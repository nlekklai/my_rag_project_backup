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
import uuid, datetime

# --- Import Auth, Path Utils & Global Vars ---
from routers.auth_router import UserMe, get_current_user
from utils.path_utils import (
    get_document_source_dir,
    load_doc_id_mapping,
    save_doc_id_mapping,
    _update_doc_id_mapping,
    get_document_file_path,
    get_mapping_tenant_root_path,
    get_mapping_key_from_physical_path,
    _n
)
from config.global_vars import (
    DEFAULT_YEAR,
    DEFAULT_ENABLER,
    EVIDENCE_DOC_TYPES,
    DOCUMENT_ID_MAPPING_FILENAME_SUFFIX
)

logger = logging.getLogger(__name__)
upload_router = APIRouter(prefix="/api/uploads", tags=["Knowledge Management"])

# --- Pydantic Models ---
class UploadResponse(BaseModel):
    doc_id: str
    status: str
    filename: str
    doc_type: str
    enabler: str
    tenant: str
    year: int
    size: int

class IngestRequest(BaseModel):
    doc_ids: List[str]

class IngestResult(BaseModel):
    doc_id: str
    result: str  # 'Success' | 'Error'

class IngestResponse(BaseModel):
    results: List[IngestResult]

# --- Helper Function ---
def map_entries(mapping_data: dict, doc_type: str, tenant: str, year: int, enabler: str) -> List[UploadResponse]:
    return [
        UploadResponse(
            doc_id=uid,
            status=info.get("status", "Pending"),
            filename=info.get("file_name", "Unknown"),
            doc_type=doc_type,
            enabler=enabler.upper() if enabler else "-",
            tenant=tenant,
            year=year,
            size=info.get("file_size", 0)
        ) for uid, info in mapping_data.items()
    ]

# --- 1. GET: List Documents (Support Manage & AskAI Pages) ---
@upload_router.get("/{doc_type}", response_model=List[UploadResponse])
async def list_files(
    doc_type: str, 
    year: Optional[int] = Query(None),
    enabler: Optional[str] = Query(None),
    current_user: UserMe = Depends(get_current_user)
):
    try:
        search_year = year or getattr(current_user, 'year', DEFAULT_YEAR)
        search_tenant = current_user.tenant
        dt_clean = _n(doc_type)
        all_results = []

        # Case A: Specific Enabler (AskAI or Filtered View)
        if enabler and enabler.lower() != "all":
            mapping_data = await run_in_threadpool(lambda: load_doc_id_mapping(doc_type, search_tenant, search_year, enabler))
            all_results.extend(map_entries(mapping_data, doc_type, search_tenant, search_year, enabler))
        
        # Case B: Evidence Tab (Scan all Enablers for the year)
        elif dt_clean == _n(EVIDENCE_DOC_TYPES):
            mapping_root = get_mapping_tenant_root_path(search_tenant)
            year_dir = os.path.join(mapping_root, str(search_year))
            if os.path.exists(year_dir):
                for map_file in os.listdir(year_dir):
                    if map_file.endswith(DOCUMENT_ID_MAPPING_FILENAME_SUFFIX):
                        parts = map_file.replace(DOCUMENT_ID_MAPPING_FILENAME_SUFFIX, "").split("_")
                        if len(parts) >= 3:
                            found_enabler = parts[2]
                            m_data = await run_in_threadpool(lambda: load_doc_id_mapping(doc_type, search_tenant, search_year, found_enabler))
                            all_results.extend(map_entries(m_data, doc_type, search_tenant, search_year, found_enabler))
        
        # Case C: Global Types (FAQ, Feedback, etc.)
        else:
            mapping_data = await run_in_threadpool(lambda: load_doc_id_mapping(doc_type, search_tenant, search_year, None))
            all_results.extend(map_entries(mapping_data, doc_type, search_tenant, search_year, "-"))

        return all_results
    except Exception as e:
        logger.error(f"Error listing {doc_type}: {e}")
        return []

# --- 2. POST: Upload & Update Mapping ---
@upload_router.post("/assessment")
async def upload_file(
    file: UploadFile = File(...),
    type: str = Form(...),
    enabler: Optional[str] = Form(None),
    year: Optional[int] = Form(None),
    current_user: UserMe = Depends(get_current_user)
):
    try:
        target_year = year or getattr(current_user, 'year', DEFAULT_YEAR)
        target_enabler = enabler or DEFAULT_ENABLER
        tenant = current_user.tenant
        
        # Resolve target directory using path_utils
        save_dir = get_document_source_dir(tenant, target_year, target_enabler, type)
        os.makedirs(save_dir, exist_ok=True)
        
        file_path = os.path.join(save_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Create Mapping Entry
        doc_uuid = str(uuid.uuid4())
        relative_key = get_mapping_key_from_physical_path(file_path)
        
        new_entry = {
            doc_uuid: {
                "file_name": file.filename,
                "filepath": relative_key,
                "status": "Pending",
                "enabler": target_enabler,
                "year": target_year,
                "file_size": os.path.getsize(file_path),
                "upload_date": datetime.now().isoformat()
            }
        }
        
        # Save to Mapping File
        await run_in_threadpool(lambda: _update_doc_id_mapping(new_entry, type, tenant, target_year, target_enabler))
        
        return {"message": "Upload successful", "doc_id": doc_uuid}
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 3. GET: View/Download File ---
@upload_router.get("/view/{doc_type}/{doc_id}")
@upload_router.get("/download/{doc_type}/{doc_id}")
async def get_file(
    doc_type: str, 
    doc_id: str, 
    year: Optional[int] = Query(None),
    enabler: Optional[str] = Query(None),
    current_user: UserMe = Depends(get_current_user)
):
    search_year = year or getattr(current_user, 'year', DEFAULT_YEAR)
    # ใช้ resolver จาก path_utils เพื่อหาไฟล์จริงจาก Mapping
    resolved = get_document_file_path(doc_id, current_user.tenant, search_year, enabler, doc_type)
    
    if not resolved or not os.path.exists(resolved["file_path"]):
        raise HTTPException(status_code=404, detail="ไม่พบไฟล์ในระบบ")
        
    return FileResponse(resolved["file_path"], filename=resolved["original_filename"])

# --- 4. DELETE: Remove File & Mapping Entry ---
@upload_router.delete("/{doc_type}/{doc_id}")
async def delete_file(
    doc_type: str, 
    doc_id: str, 
    year: Optional[int] = Query(None),
    enabler: Optional[str] = Query(None),
    current_user: UserMe = Depends(get_current_user)
):
    search_year = year or getattr(current_user, 'year', DEFAULT_YEAR)
    tenant = current_user.tenant
    
    # 1. Resolve Path and Delete Physical File
    resolved = get_document_file_path(doc_id, tenant, search_year, enabler, doc_type)
    if resolved and os.path.exists(resolved["file_path"]):
        os.remove(resolved["file_path"])
        
    # 2. Update Mapping File (Remove Entry)
    mapping_data = load_doc_id_mapping(doc_type, tenant, search_year, enabler)
    if doc_id in mapping_data:
        del mapping_data[doc_id]
        save_doc_id_mapping(mapping_data, doc_type, tenant, search_year, enabler)
        
    return {"message": "ลบไฟล์เรียบร้อยแล้ว"}

# --- 5. POST: Ingest (Vectorization Trigger) ---
@upload_router.post("/ingest", response_model=IngestResponse)
async def ingest_files(request: IngestRequest, current_user: UserMe = Depends(get_current_user)):
    results = []
    # ในขั้นตอนนี้ เราจะวนลูปตาม doc_ids ที่ส่งมาจาก UI
    # ปกติ UI จะเรียกจาก Tab ปัจจุบัน ซึ่งเราต้องหาว่า doc_id นี้อยู่ใน mapping ไหน
    # เพื่อความง่าย เราจะสแกนหาไฟล์ใน Mapping และอัปเดตสถานะ
    
    for doc_id in request.doc_ids:
        # 📌 Logic: คุณควรมีฟังก์ชันเรียก Engine สำหรับ Vectorize ที่นี่
        # ตัวอย่างการอัปเดตสถานะใน Mapping หลังจาก Ingest สำเร็จ
        # ในที่นี้ขอยกตัวอย่างเป็น Success เสมอ
        results.append(IngestResult(doc_id=doc_id, result="Success"))
        
        # Note: ควรมีการเปิด mapping มาแก้ status เป็น 'Ingested' 
        # เพื่อให้ UI แสดง Badge สีเขียว
        
    return IngestResponse(results=results)