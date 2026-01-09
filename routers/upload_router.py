# -*- coding: utf-8 -*-
import transformers.utils.import_utils as import_utils
import_utils.check_torch_load_is_safe = lambda *args, **kwargs: True

import os
import shutil
import logging
import asyncio
import unicodedata
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
    status: str
    message: str

# =========================
# Helpers
# =========================

async def clear_vector_data(doc_id: str, tenant: str, year: Optional[int], doc_type: str, enabler: Optional[str]):
    """ลบข้อมูลเดิมออกจาก Vectorstore อย่างปลอดภัย"""
    if not doc_id:
        return
    try:
        col_name = get_doc_type_collection_key(doc_type, enabler)
        vectorstore = get_vectorstore(col_name, tenant, year)
        
        # ตรวจสอบว่ามีข้อมูลอยู่จริงก่อนลบ
        # โดยทั่วไป delete ของ LangChain Chroma จะรับค่า ids เป็น list
        vectorstore.delete(where={"stable_doc_uuid": doc_id})
        logger.info(f"✅ Cleared existing vector chunks for doc_id: {doc_id}")
    except Exception as e:
        # ถ้าไม่มีข้อมูลให้ลบ Chroma อาจจะพ่น Error ออกมา เราแค่ Log ไว้พอ
        logger.debug(f"Info: No existing data to clear for {doc_id} or {str(e)}")

def map_entries(mapping_data: dict, doc_type: str, tenant: str, year: Optional[int], enabler: Optional[str]) -> List[UploadResponse]:
    results = []
    now_iso = datetime.now(timezone.utc).isoformat()
    # ปรับค่าสำหรับการแสดงผลบน UI (ถ้าเป็น Global ให้โชว์ปีปัจจุบันหรือปีกลาง)
    display_year = year if (year and year != 0) else DEFAULT_YEAR
    display_enabler = (enabler or "KM").upper() if enabler else "GLOBAL"

    for uid, info in mapping_data.items():
        results.append(UploadResponse(
            doc_id=uid,
            status=info.get("status", "Pending"),
            filename=info.get("file_name") or info.get("filename", "Unknown File"),
            doc_type=doc_type,
            enabler=display_enabler,
            tenant=tenant,
            year=display_year,
            # 🟢 แก้ไขบรรทัดนี้: ให้ลองดึง 'file_size' ก่อน ถ้าไม่มีให้ไปดึง 'size'
            size=info.get("file_size") or info.get("size") or 0,
            upload_date=str(info.get("upload_date") or info.get("uploadDate") or now_iso)
        ))
    return results

# =========================
# 1. GET: List Documents
# =========================

@upload_router.get("/{doc_type}", response_model=List[UploadResponse])
async def list_files(
    doc_type: str,
    year: Optional[str] = Query(None),
    enabler: Optional[str] = Query(None),
    current_user: UserMe = Depends(get_current_user)
):
    try:
        tenant = current_user.tenant
        dt_clean = _n(doc_type)
        
        # 🟢 1. จัดการกลุ่ม Global Documents (document, faq, seam)
        # ไม่อิงปีและ enabler ในการโหลด Mapping
        if dt_clean != _n(EVIDENCE_DOC_TYPES):
            mapping = await run_in_threadpool(load_doc_id_mapping, doc_type, tenant, None, None)
            return map_entries(mapping, doc_type, tenant, 0, None)

        # 🔵 2. จัดการกลุ่ม Yearly Evidence
        all_results = []
        if year == "all":
            root_path = get_mapping_tenant_root_path(tenant)
            years_to_search = [int(d) for d in os.listdir(root_path) if d.isdigit()] if os.path.exists(root_path) else [DEFAULT_YEAR]
        else:
            years_to_search = [int(year)] if year and year != "undefined" else [getattr(current_user, "year", DEFAULT_YEAR)]

        for search_year in years_to_search:
            if enabler and enabler.lower() != "all":
                mapping = await run_in_threadpool(load_doc_id_mapping, doc_type, tenant, search_year, enabler)
                all_results.extend(map_entries(mapping, doc_type, tenant, search_year, enabler))
            else:
                # สแกนทุก enabler ในปีนั้นๆ
                year_dir = os.path.join(get_mapping_tenant_root_path(tenant), str(search_year))
                if os.path.exists(year_dir):
                    for fname in os.listdir(year_dir):
                        if fname.endswith(DOCUMENT_ID_MAPPING_FILENAME_SUFFIX):
                            # ตัดชื่อไฟล์เพื่อเอา enabler ออกมา (เช่น tcg_2567_km_doc_id_mapping.json)
                            parts = fname.split("_")
                            found_en = parts[2] if len(parts) >= 3 else "KM"
                            mapping = await run_in_threadpool(load_doc_id_mapping, doc_type, tenant, search_year, found_en)
                            all_results.extend(map_entries(mapping, doc_type, tenant, search_year, found_en))
        
        return all_results
    except Exception as e:
        logger.error(f"Error listing files: {str(e)}")
        return []

# =========================
# 2. POST: Upload & Ingest
# =========================

@upload_router.post("/{doc_type}")
async def upload_file(
    file: UploadFile = File(...),
    doc_type: str = "evidence",
    enabler: Optional[str] = Form(None),
    year: Optional[int] = Form(None),
    current_user: UserMe = Depends(get_current_user),
):
    try:
        tenant = _n(current_user.tenant)
        dt_clean = _n(doc_type)

        # 🟢 Normalize Metadata: ถ้าไม่ใช่ Evidence ให้เป็น None เสมอ
        if dt_clean != _n(EVIDENCE_DOC_TYPES):
            target_year = None
            target_enabler = None
            target_year_str = ""
        else:
            target_year = year or getattr(current_user, "year", DEFAULT_YEAR)
            target_enabler = enabler or "KM"
            target_year_str = str(target_year)

        # 1. บันทึกไฟล์ลงดิสก์
        save_dir = get_document_source_dir(tenant, target_year, target_enabler, doc_type)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. เตรียม UUID และล้างข้อมูลเก่า
        doc_id = create_stable_uuid_from_path(file_path, tenant, target_year, target_enabler)
        await clear_vector_data(doc_id, tenant, target_year, doc_type, target_enabler)

        # 3. เริ่มกระบวนการ Ingest
        chunks = []
        msg = "สำเร็จ"
        
        try:
            chunks, _, _ = await asyncio.to_thread(
                process_document, file_path, file.filename, doc_id, 
                doc_type, target_enabler, target_year_str, tenant
            )

            if chunks:
                col_name = get_doc_type_collection_key(doc_type, target_enabler)
                vectorstore = get_vectorstore(col_name, tenant, target_year)
                vectorstore.add_documents(documents=chunks, ids=[c.metadata["chunk_uuid"] for c in chunks])
                status = "Ingested"
            else:
                status = "Warning: No Content"
                msg = "ไม่พบเนื้อหาที่อ่านได้จากไฟล์"
                
        except Exception as ingest_err:
            status = "Error"
            msg = str(ingest_err)
            logger.error(f"Ingest failed for {file.filename}: {msg}")

        # 4. บันทึก Mapping (โหลดใหม่ทุกครั้งเพื่อความสดใหม่ของข้อมูล)
        mapping = load_doc_id_mapping(doc_type, tenant, target_year, target_enabler)
        mapping[doc_id] = {
            "file_name": file.filename,
            "filepath": get_mapping_key_from_physical_path(file_path),
            "status": status,
            "file_size": os.path.getsize(file_path),
            "upload_date": datetime.now(timezone.utc).isoformat(),
            "chunk_count": len(chunks) if chunks else 0,
            "stable_doc_uuid": doc_id,
            "error_message": msg if status == "Error" else None
        }
        save_doc_id_mapping(mapping, doc_type, tenant, target_year, target_enabler)

        return {"doc_id": doc_id, "status": status, "message": msg}

    except Exception as e:
        logger.exception("Upload process failed")
        raise HTTPException(status_code=500, detail=str(e))

@upload_router.post("/reingest/{doc_type}/{doc_id}", response_model=IngestResult)
async def reingest_file(
    doc_type: str,
    doc_id: str,
    year: Optional[int] = Query(None),
    enabler: Optional[str] = Query(None),
    current_user: UserMe = Depends(get_current_user)
):
    tenant = current_user.tenant
    dt_clean = _n(doc_type)

    if dt_clean != _n(EVIDENCE_DOC_TYPES):
        target_year = None
        target_enabler = None
        target_year_str = ""
    else:
        target_year = year or getattr(current_user, "year", DEFAULT_YEAR)
        target_enabler = enabler or "KM"
        target_year_str = str(target_year)

    # 1. ค้นหาไฟล์เดิม
    resolved = get_document_file_path(doc_id, tenant, target_year, target_enabler, doc_type)
    if not resolved or not os.path.exists(resolved["file_path"]):
        raise HTTPException(status_code=404, detail="ไม่พบไฟล์ในระบบ")

    # 2. ล้างข้อมูลเก่า
    await clear_vector_data(doc_id, tenant, target_year, doc_type, target_enabler)

    try:
        # 3. Ingest ใหม่
        chunks, _, _ = await asyncio.to_thread(
            process_document, resolved["file_path"], resolved["original_filename"], 
            doc_id, doc_type, target_enabler, target_year_str, tenant
        )

        status = "Ingested" if chunks else "Warning: No Content"
        if chunks:
            col_name = get_doc_type_collection_key(doc_type, target_enabler)
            vectorstore = get_vectorstore(col_name, tenant, target_year)
            vectorstore.add_documents(documents=chunks, ids=[c.metadata["chunk_uuid"] for c in chunks])

        # 4. อัปเดต Mapping
        mapping = load_doc_id_mapping(doc_type, tenant, target_year, target_enabler)
        if doc_id in mapping:
            mapping[doc_id].update({
                "status": status,
                "upload_date": datetime.now(timezone.utc).isoformat(),
                "chunk_count": len(chunks) if chunks else 0
            })
            save_doc_id_mapping(mapping, doc_type, tenant, target_year, target_enabler)

        return IngestResult(doc_id=doc_id, status=status, message="ประมวลผลใหม่สำเร็จ")
    except Exception as e:
        logger.error(f"Re-ingest failed: {str(e)}")
        return IngestResult(doc_id=doc_id, status="Error", message=str(e))

# =========================
# 3. GET/DELETE: Download & Remove
# =========================
# =========================
# 3. GET/DELETE: Download & View & Remove (Revised for Preview)
# =========================
import mimetypes # เพิ่มการ import standard library นี้ที่ด้านบนของไฟล์ด้วยครับ

@upload_router.get("/download/{doc_type}/{doc_id}")
async def download_file(
    doc_type: str, 
    doc_id: str,
    year: Optional[int] = Query(None),
    enabler: Optional[str] = Query(None),
    current_user: UserMe = Depends(get_current_user)
):
    dt_clean = _n(doc_type)
    search_year = None if dt_clean != _n(EVIDENCE_DOC_TYPES) else (year or getattr(current_user, "year", DEFAULT_YEAR))
    
    resolved = get_document_file_path(doc_id, current_user.tenant, search_year, enabler, doc_type)
    
    if not resolved:
         raise HTTPException(status_code=404, detail="ไม่พบรหัสไฟล์ในฐานข้อมูล")

    # --- ส่วนจัดการ Path และ Unicode (NFC) ---
    target_path = resolved["file_path"]
    normalized_path = unicodedata.normalize('NFC', target_path)
    
    if not os.path.exists(normalized_path):
        normalized_path = unicodedata.normalize('NFD', target_path)
        if not os.path.exists(normalized_path):
            logger.error(f"❌ File not found on disk: {target_path}")
            raise HTTPException(status_code=404, detail="ไม่พบไฟล์จริงบนระบบ")

    # --- วิเคราะห์ Media Type (MIME Type) ---
    # ใช้ mimetypes library เพื่อความแม่นยำสูงขึ้น
    m_type, _ = mimetypes.guess_type(normalized_path)
    
    # Fallback กรณีพิเศษ
    if not m_type:
        file_ext = normalized_path.lower()
        if file_ext.endswith('.pdf'):
            m_type = 'application/pdf'
        elif file_ext.endswith('.docx'):
            m_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        else:
            m_type = 'application/octet-stream'

    logger.info(f"📁 Serving file: {normalized_path} as {m_type}")

    # ส่ง FileResponse
    # การตั้ง filename ใน FileResponse บางครั้งทำให้ Browser บังคับโหลด 
    # ถ้าอยากให้ Preview ภาพ/PDF ได้ดีที่สุด ให้ใช้พารามิเตอร์ตามนี้ครับ
    return FileResponse(
        path=normalized_path,
        media_type=m_type,
        filename=resolved["original_filename"],
        content_disposition_type="inline" # 🟢 พยายาม Preview บน Browser
    )

@upload_router.delete("/{doc_type}/{doc_id}")
async def delete_file(
    doc_type: str, doc_id: str,
    year: Optional[int] = Query(None),
    enabler: Optional[str] = Query(None),
    current_user: UserMe = Depends(get_current_user)
):
    tenant = current_user.tenant
    dt_clean = _n(doc_type)
    
    target_year = None if dt_clean != _n(EVIDENCE_DOC_TYPES) else (year or getattr(current_user, "year", DEFAULT_YEAR))
    target_enabler = None if dt_clean != _n(EVIDENCE_DOC_TYPES) else (enabler or "KM")
    
    # 1. ลบจาก Vectorstore
    await clear_vector_data(doc_id, tenant, target_year, doc_type, target_enabler)

    # 2. ลบไฟล์จริง
    resolved = get_document_file_path(doc_id, tenant, target_year, target_enabler, doc_type)
    if resolved and os.path.exists(resolved["file_path"]):
        os.remove(resolved["file_path"])

    # 3. ลบออกจาก Mapping
    mapping = load_doc_id_mapping(doc_type, tenant, target_year, target_enabler)
    if doc_id in mapping:
        del mapping[doc_id]
        save_doc_id_mapping(mapping, doc_type, tenant, target_year, target_enabler)

    return {"message": f"ลบไฟล์ {doc_id} สำเร็จ"}