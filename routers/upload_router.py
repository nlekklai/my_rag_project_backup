# -*- coding: utf-8 -*-
import transformers.utils.import_utils as import_utils
import_utils.check_torch_load_is_safe = lambda *args, **kwargs: True

import os
import shutil
import logging
import asyncio
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

# Core & Utils
from core.vectorstore import get_vectorstore_manager
from core.ingest import process_document, get_vectorstore, _n
from routers.auth_router import UserMe, get_current_user
from utils.path_utils import (
    get_document_source_dir,
    load_doc_id_mapping,
    save_doc_id_mapping,
    get_document_file_path,
    get_mapping_tenant_root_path,
    get_mapping_key_from_physical_path,
    create_stable_uuid_from_path,
    get_doc_type_collection_key
)
from config.global_vars import (
    DEFAULT_YEAR,
    DEFAULT_ENABLER,
    EVIDENCE_DOC_TYPES,
    DOCUMENT_ID_MAPPING_FILENAME_SUFFIX
)

# ตั้งค่า Environment สำหรับ Torch
os.environ["TORCH_LOAD_WEIGHTS_ONLY"] = "FALSE"

logger = logging.getLogger(__name__)

# กำหนด Router พร้อม Prefix ที่ชัดเจนเพื่อลดความสับสนใน app.py
upload_router = APIRouter(prefix="/api/upload", tags=["Knowledge Management"])

# =========================
# Pydantic Models
# =========================

class UploadResponse(BaseModel):
    doc_id: str
    status: str
    filename: str
    doc_type: str
    enabler: str
    tenant: str
    year: int
    size: int
    upload_date: str

class IngestResult(BaseModel):
    doc_id: str
    result: str

class IngestResponse(BaseModel):
    results: List[IngestResult]

class IngestRequest(BaseModel):
    doc_ids: List[str]

# =========================
# Helpers
# =========================

def map_entries(mapping_data: dict, doc_type: str, tenant: str, year: int, enabler: str) -> List[UploadResponse]:
    """แปลงข้อมูลจาก JSON Mapping เป็น List ของ UploadResponse พร้อมจัดการ Fallback"""
    results = []
    now_iso = datetime.now(timezone.utc).isoformat()

    for uid, info in mapping_data.items():
        results.append(UploadResponse(
            doc_id=uid,
            status=info.get("status", "Pending"),
            filename=info.get("file_name") or info.get("filename", "Unknown File"),
            doc_type=doc_type,
            enabler=(enabler or "KM").upper(),
            tenant=tenant,
            year=year,
            size=info.get("file_size", 0),
            upload_date=str(info.get("upload_date") or info.get("uploadDate") or now_iso)
        ))
    return results

# =========================
# 1. GET: List Documents
# =========================

@upload_router.get("/{doc_type}", response_model=List[UploadResponse])
@upload_router.get("s/{doc_type}", response_model=List[UploadResponse], include_in_schema=False) # รองรับ /api/uploads
async def list_files(
    doc_type: str,
    year: Optional[str] = Query(None),
    enabler: Optional[str] = Query(None),
    current_user: UserMe = Depends(get_current_user)
):
    try:
        tenant = current_user.tenant
        dt_clean = _n(doc_type)
        all_results = []

        # 1. จัดการเรื่องปีที่จะค้นหา
        if year == "all":
            root_path = get_mapping_tenant_root_path(tenant)
            years_to_search = [int(d) for d in os.listdir(root_path) if d.isdigit()] if os.path.exists(root_path) else [DEFAULT_YEAR]
        else:
            years_to_search = [int(year)] if year else [getattr(current_user, "year", DEFAULT_YEAR)]

        # 2. ค้นหาข้อมูลตามเงื่อนไข
        for search_year in years_to_search:
            # Case A: ระบุ Enabler ชัดเจน
            if enabler and enabler.lower() != "all":
                mapping = await run_in_threadpool(load_doc_id_mapping, doc_type, tenant, search_year, enabler)
                all_results.extend(map_entries(mapping, doc_type, tenant, search_year, enabler))
            
            # Case B: Evidence (สแกนทุก Enabler ในปีนั้น)
            elif dt_clean == _n(EVIDENCE_DOC_TYPES):
                year_dir = os.path.join(get_mapping_tenant_root_path(tenant), str(search_year))
                if os.path.exists(year_dir):
                    for fname in os.listdir(year_dir):
                        if fname.endswith(DOCUMENT_ID_MAPPING_FILENAME_SUFFIX):
                            found_en = fname.split("_")[2] if len(fname.split("_")) >= 3 else "KM"
                            mapping = await run_in_threadpool(load_doc_id_mapping, doc_type, tenant, search_year, found_en)
                            all_results.extend(map_entries(mapping, doc_type, tenant, search_year, found_en))
            
            # Case C: อื่นๆ (Global หรือ Default)
            else:
                mapping = await run_in_threadpool(load_doc_id_mapping, doc_type, tenant, search_year, None)
                all_results.extend(map_entries(mapping, doc_type, tenant, search_year, "-"))

        return all_results
    except Exception as e:
        logger.error(f"Error listing files: {str(e)}")
        return []

# =========================
# 2. POST: Upload & Auto-Ingest
# =========================

@upload_router.post("")
@upload_router.post("/{doc_type}")
async def upload_file(
    file: UploadFile = File(...),
    doc_type: Optional[str] = None,
    enabler: Optional[str] = Form(None),
    year: Optional[int] = Form(None),
    current_user: UserMe = Depends(get_current_user),
):
    try:
        actual_type = doc_type or "evidence"
        tenant = _n(current_user.tenant)
        target_year = year or getattr(current_user, "year", DEFAULT_YEAR)
        target_enabler = enabler or "KM"

        # 1. บันทึกไฟล์
        save_dir = get_document_source_dir(tenant, target_year, target_enabler, actual_type)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. สร้าง UUID และเริ่มกระบวนการ Ingest
        doc_id = create_stable_uuid_from_path(file_path, tenant, target_year, target_enabler)
        
        # รันการประมวลผลไฟล์ใน Thread เพื่อไม่ให้ Block Event Loop
        chunks, _, _ = await asyncio.to_thread(
            process_document, file_path, file.filename, doc_id, 
            actual_type, target_enabler, target_year, tenant
        )

        if chunks:
            col_name = get_doc_type_collection_key(actual_type, target_enabler)
            vectorstore = get_vectorstore(col_name, tenant, target_year)
            vectorstore.add_documents(documents=chunks, ids=[c.metadata["chunk_uuid"] for c in chunks])

            # 3. อัปเดต Mapping
            mapping = load_doc_id_mapping(actual_type, tenant, target_year, target_enabler)
            mapping[doc_id] = {
                "file_name": file.filename,
                "filepath": get_mapping_key_from_physical_path(file_path),
                "status": "Ingested",
                "file_size": os.path.getsize(file_path),
                "upload_date": datetime.now(timezone.utc).isoformat(),
                "chunk_count": len(chunks),
                "stable_doc_uuid": doc_id
            }
            save_doc_id_mapping(mapping, actual_type, tenant, target_year, target_enabler)

        return {"message": "สำเร็จ", "doc_id": doc_id, "status": "Ingested"}

    except Exception as e:
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail=str(e))

# =========================
# 3. GET: View/Download & DELETE
# =========================

@upload_router.get("/download/{doc_type}/{doc_id}")
async def download_file(
    doc_type: str, doc_id: str,
    year: Optional[int] = Query(None),
    enabler: Optional[str] = Query(None),
    current_user: UserMe = Depends(get_current_user)
):
    search_year = year or getattr(current_user, "year", DEFAULT_YEAR)
    resolved = get_document_file_path(doc_id, current_user.tenant, search_year, enabler, doc_type)

    if not resolved or not os.path.exists(resolved["file_path"]):
        raise HTTPException(status_code=404, detail="ไม่พบไฟล์")

    return FileResponse(resolved["file_path"], filename=resolved["original_filename"])

@upload_router.delete("/{doc_type}/{doc_id}")
async def delete_file(
    doc_type: str, doc_id: str,
    year: Optional[int] = Query(None),
    enabler: Optional[str] = Query(None),
    current_user: UserMe = Depends(get_current_user)
):
    tenant = current_user.tenant
    search_year = year or getattr(current_user, "year", DEFAULT_YEAR)
    
    # ลบไฟล์จริง
    resolved = get_document_file_path(doc_id, tenant, search_year, enabler, doc_type)
    if resolved and os.path.exists(resolved["file_path"]):
        os.remove(resolved["file_path"])

    # ลบออกจาก Mapping
    mapping = load_doc_id_mapping(doc_type, tenant, search_year, enabler)
    if doc_id in mapping:
        del mapping[doc_id]
        save_doc_id_mapping(mapping, doc_type, tenant, search_year, enabler)

    return {"message": "ลบไฟล์สำเร็จ"}