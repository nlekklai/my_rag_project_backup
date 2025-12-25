# -*- coding: utf-8 -*-
import os
import shutil
import uuid
import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from core.vectorstore import get_vectorstore_manager
from core.ingest import process_document, get_vectorstore, _n
import asyncio

# --- Import Auth, Path Utils & Global Vars ---
from routers.auth_router import UserMe, get_current_user
from utils.path_utils import (
    get_document_source_dir,
    load_doc_id_mapping,
    save_doc_id_mapping,
    get_document_file_path,
    get_mapping_tenant_root_path,
    get_mapping_key_from_physical_path,
    _n,
    create_stable_uuid_from_path,
    get_doc_type_collection_key
)
from config.global_vars import (
    DEFAULT_YEAR,
    DEFAULT_ENABLER,
    EVIDENCE_DOC_TYPES,
    DOCUMENT_ID_MAPPING_FILENAME_SUFFIX
)

import aiofiles
from datetime import timezone

logger = logging.getLogger(__name__)
# upload_router = APIRouter(prefix="/api/uploads", tags=["Knowledge Management"])
upload_router = APIRouter(tags=["Knowledge Management"])


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
    upload_date: str # <--- ต้องมีฟิลด์นี้เพื่อให้ FastAPI ส่งออกไปใน JSON


class IngestRequest(BaseModel):
    doc_ids: List[str]


class IngestResult(BaseModel):
    doc_id: str
    result: str


class IngestResponse(BaseModel):
    results: List[IngestResult]


# =========================
# Helper
# =========================

def map_entries(
    mapping_data: dict,
    doc_type: str,
    tenant: str,
    year: int,
    enabler: str
) -> List[UploadResponse]:
    """Map mapping_data entries to UploadResponse"""
    results = []
    
    # ใช้เวลาปัจจุบันเป็น fallback แบบ ISO สากล
    default_now = datetime.now(timezone.utc).isoformat()

    for uid, info in mapping_data.items():
        # 1. พยายามดึงวันที่ ถ้าไม่มีจริงๆ ให้ใช้ตอนนี้ (เพื่อไม่ให้หน้าจอ Invalid)
        raw_date = info.get("upload_date") or info.get("uploadDate") or default_now
        
        # 2. ตรวจสอบสถานะ (เพื่อให้แสดง Badge สีสวยๆ ใน UI)
        status = info.get("status", "Pending")

        results.append(UploadResponse(
            doc_id=uid,
            status=status,
            filename=info.get("file_name", "Unknown"),
            doc_type=doc_type,
            enabler=enabler.upper() if enabler else "-",
            tenant=tenant,
            year=year,
            size=info.get("file_size", 0),
            upload_date=raw_date  # ส่งค่านี้ไปให้ Frontend mappedFiles
        ))
    return results


# =========================
# 1. GET: List Documents
# =========================

@upload_router.get("/{doc_type}", response_model=List[UploadResponse])
async def list_files(
    doc_type: str,
    year: Optional[int] = Query(None),
    enabler: Optional[str] = Query(None),
    current_user: UserMe = Depends(get_current_user)
):
    try:
        search_year = year or getattr(current_user, "year", DEFAULT_YEAR)
        tenant = current_user.tenant
        dt_clean = _n(doc_type)

        results: List[UploadResponse] = []

        # A) Specific Enabler
        if enabler and enabler.lower() != "all":
            mapping = await run_in_threadpool(
                load_doc_id_mapping, doc_type, tenant, search_year, enabler
            )
            results.extend(map_entries(mapping, doc_type, tenant, search_year, enabler))

        # B) Evidence: scan all enablers
        elif dt_clean == _n(EVIDENCE_DOC_TYPES):
            root = get_mapping_tenant_root_path(tenant)
            year_dir = os.path.join(root, str(search_year))

            if os.path.exists(year_dir):
                for fname in os.listdir(year_dir):
                    if fname.endswith(DOCUMENT_ID_MAPPING_FILENAME_SUFFIX):
                        parts = fname.replace(
                            DOCUMENT_ID_MAPPING_FILENAME_SUFFIX, ""
                        ).split("_")
                        if len(parts) >= 3:
                            found_enabler = parts[2]
                            mapping = await run_in_threadpool(
                                load_doc_id_mapping,
                                doc_type,
                                tenant,
                                search_year,
                                found_enabler,
                            )
                            results.extend(
                                map_entries(
                                    mapping,
                                    doc_type,
                                    tenant,
                                    search_year,
                                    found_enabler,
                                )
                            )

        # C) Global docs
        else:
            mapping = await run_in_threadpool(
                load_doc_id_mapping, doc_type, tenant, search_year, None
            )
            results.extend(map_entries(mapping, doc_type, tenant, search_year, "-"))

        return results

    except Exception as e:
        logger.exception("List files error")
        return []


# =========================
# 2. POST: Upload (ฉบับปรับปรุง ID ให้ตรงกับ VectorStore)
# =========================

@upload_router.post("")              # รองรับ: POST /api/upload
@upload_router.post("/{doc_type}")   # รองรับ: POST /api/upload/evidence
@upload_router.post("/{doc_type}")
async def upload_file(
    file: UploadFile = File(...),
    doc_type: Optional[str] = None,
    enabler: Optional[str] = Form(None),
    year: Optional[int] = Form(None),
    current_user: UserMe = Depends(get_current_user),
):
    try:
        actual_type = doc_type or "document"
        tenant = _n(current_user.tenant)
        target_year = year or getattr(current_user, "year", 2568)
        target_enabler = enabler or "KM"

        # 1. บันทึกไฟล์ลง Disk
        save_dir = get_document_source_dir(tenant, target_year, target_enabler, actual_type)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, file.filename)
        
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 2. สร้าง Stable UUID (Deterministic)
        doc_id = create_stable_uuid_from_path(file_path, tenant, target_year, target_enabler)
        
        # 3. ⚡️ เรียกใช้ Logic จาก core/ingest.py โดยตรง
        # ใช้ asyncio.to_thread เพื่อไม่ให้งานหนัก (OCR/Embedding) ไปค้าง Web Server
        chunks, _, _ = await asyncio.to_thread(
            process_document,
            file_path=file_path,
            file_name=file.filename,
            stable_doc_uuid=doc_id,
            doc_type=actual_type,
            enabler=target_enabler,
            year=target_year,
            tenant=tenant
        )

        if chunks:
            # 4. บันทึกลง ChromaDB (ใช้ฟังก์ชัน get_vectorstore จาก ingest.py)
            col_name = get_doc_type_collection_key(actual_type, target_enabler)
            vectorstore = get_vectorstore(col_name, tenant, target_year)
            
            chunk_ids = [c.metadata["chunk_uuid"] for c in chunks]
            vectorstore.add_documents(documents=chunks, ids=chunk_ids)

            # 5. อัปเดต Mapping เป็น Ingested
            mapping = load_doc_id_mapping(actual_type, tenant, target_year, target_enabler)
            mapping[doc_id] = {
                "file_name": file.filename,
                "filepath": get_mapping_key_from_physical_path(file_path),
                "status": "Ingested",
                "chunk_count": len(chunks),
                "chunk_uuids": chunk_ids,
                "stable_doc_uuid": doc_id
            }
            save_doc_id_mapping(mapping, actual_type, tenant, target_year, target_enabler)

        return {"message": "Upload and Ingest complete", "doc_id": doc_id}

    except Exception as e:
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail=str(e))

# =========================
# 3. GET: View / Download
# =========================

@upload_router.get("/view/{doc_type}/{doc_id}")
@upload_router.get("/download/{doc_type}/{doc_id}")
async def get_file(
    doc_type: str,
    doc_id: str,
    year: Optional[int] = Query(None),
    enabler: Optional[str] = Query(None),
    current_user: UserMe = Depends(get_current_user),
):
    search_year = year or getattr(current_user, "year", DEFAULT_YEAR)

    resolved = get_document_file_path(
        doc_id,
        current_user.tenant,
        search_year,
        enabler,
        doc_type,
    )

    if not resolved or not os.path.exists(resolved["file_path"]):
        raise HTTPException(status_code=404, detail="ไม่พบไฟล์ในระบบ")

    return FileResponse(
        resolved["file_path"],
        filename=resolved["original_filename"],
    )


# =========================
# 4. DELETE
# =========================

@upload_router.delete("/{doc_type}/{doc_id}")
async def delete_file(
    doc_type: str,
    doc_id: str,
    year: Optional[int] = Query(None),
    enabler: Optional[str] = Query(None),
    current_user: UserMe = Depends(get_current_user),
):
    tenant = current_user.tenant
    search_year = year or getattr(current_user, "year", DEFAULT_YEAR)

    resolved = get_document_file_path(
        doc_id, tenant, search_year, enabler, doc_type
    )

    if resolved and os.path.exists(resolved["file_path"]):
        os.remove(resolved["file_path"])

    mapping = load_doc_id_mapping(
        doc_type, tenant, search_year, enabler
    )
    if doc_id in mapping:
        del mapping[doc_id]
        save_doc_id_mapping(
            mapping, doc_type, tenant, search_year, enabler
        )

    return {"message": "ลบไฟล์เรียบร้อยแล้ว"}


# =========================
# 5. POST: Ingest (Implemented)
# =========================

@upload_router.post("/ingest", response_model=IngestResponse)
async def ingest_files(
    request: IngestRequest,
    doc_type: str = Query("document"),
    year: Optional[int] = Query(None),
    enabler: Optional[str] = Query(None),
    current_user: UserMe = Depends(get_current_user),
):
    results = []
    tenant = current_user.tenant
    target_year = year or getattr(current_user, "year", DEFAULT_YEAR)
    target_enabler = enabler or DEFAULT_ENABLER
    
    # 1. เรียกใช้ VectorStoreManager สำหรับ Tenant นั้นๆ
    vsm = get_vectorstore_manager(tenant=tenant)
    
    # 2. โหลด Mapping เพื่อดูว่าไฟล์อยู่ที่ไหน (Physical Path)
    mapping = await run_in_threadpool(
        load_doc_id_mapping, doc_type, tenant, target_year, target_enabler
    )

    for doc_id in request.doc_ids:
        if doc_id not in mapping:
            results.append(IngestResult(doc_id=doc_id, result="Error: ID not found in mapping"))
            continue
            
        try:
            doc_info = mapping[doc_id]
            # ดึง Path เต็มจาก Utils
            resolved = get_document_file_path(doc_id, tenant, target_year, target_enabler, doc_type)
            
            if not resolved or not os.path.exists(resolved["file_path"]):
                results.append(IngestResult(doc_id=doc_id, result="Error: Physical file missing"))
                continue

            # 3. ⚡️ สั่ง Ingest เข้า VectorStore
            # ขั้นตอนนี้จะทำ: Load -> Chunk -> Embed -> Save to ChromaDB
            success = await asyncio.to_thread(
                vsm.ingest_document,
                file_path=resolved["file_path"],
                doc_type=doc_type,
                enabler=target_enabler,
                year=str(target_year),
                stable_doc_uuid=doc_id # ส่ง ID เดียวกันเข้าไปบันทึก
            )

            if success:
                # 4. อัปเดตสถานะใน Mapping เป็น Ingested
                mapping[doc_id]["status"] = "Ingested"
                results.append(IngestResult(doc_id=doc_id, result="Success"))
            else:
                results.append(IngestResult(doc_id=doc_id, result="Failed to process document"))

        except Exception as e:
            logger.error(f"Ingest error for {doc_id}: {str(e)}")
            results.append(IngestResult(doc_id=doc_id, result=f"Error: {str(e)}"))

    # บันทึกสถานะใหม่ลงไฟล์ JSON
    save_doc_id_mapping(mapping, doc_type, tenant, target_year, target_enabler)
    
    return IngestResponse(results=results)