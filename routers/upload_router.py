from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Path, Query, Depends, status
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timezone
import logging, os, sys, uuid
from pathlib import Path as SysPath

# -----------------------------
# --- Project Modules ---
# -----------------------------
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.append(project_root)

    from config.global_vars import (
        DATA_DIR,
        VECTORSTORE_DIR,
        SUPPORTED_DOC_TYPES,
        DEFAULT_ENABLER, # 💡 ต้องมีค่านี้
        EVIDENCE_DOC_TYPES
    )
    from core.ingest import (
        process_document,
        list_documents,
        delete_document_by_uuid,
        DocInfo,
    )
    # --- NEW IMPORT for Auth ---
    from routers.auth_router import UserMe, get_current_user
    # ---------------------------

except ImportError as e:
    # ❌ Import error: [ชื่อโมดูลที่หายไป]
    print(f"❌ Import error: {e}")
    raise

# -----------------------------
# --- Router & Logger ---
# -----------------------------
upload_router = APIRouter(prefix="/api", tags=["Upload / Ingest"])

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

# -----------------------------
# --- Models ---
# -----------------------------
class UploadResponse(BaseModel):
    doc_id: str
    status: str
    filename: str
    doc_type: str
    file_type: Optional[str] = None
    upload_date: Optional[str] = None
    message: Optional[str] = None
    chunk_count: Optional[int] = None
    size: Optional[float] = None
    enabler: Optional[str] = None
    tenant: Optional[str] = None
    year: Optional[int] = None

# -----------------------------
# --- Helper Function for File Path ---
# -----------------------------
# 💥 REVISED: เพิ่ม enabler สำหรับ Evidence file path
def get_save_dir(doc_type: str, tenant: str, year: int, enabler: Optional[str] = None) -> str:
    """Constructs the segregated directory path for saving files (DATA_DIR/tenant/year/doc_type/[enabler])."""
    # Structure: DATA_DIR/tenant/year/doc_type
    base_path = os.path.join(DATA_DIR, tenant, str(year), doc_type)
    
    # ถ้าเป็น Evidence และมี Enabler ให้เพิ่ม Enabler เป็น subfolder
    if doc_type.lower() == "evidence" and enabler:
        return os.path.join(base_path, enabler)
        
    return base_path

# -----------------------------
# --- Upload with background processing ---
# -----------------------------
@upload_router.post("/upload/{doc_type}", response_model=UploadResponse)
async def upload_document(
    doc_type: str = Path(..., description=f"Document type. Must be one of: {SUPPORTED_DOC_TYPES}"),
    file: UploadFile = File(..., description="Document file to upload"),
    # 💥 ADDED: เพิ่ม enabler เป็น Form Parameter
    enabler: Optional[str] = Form(None, description="Enabler code (used for evidence doc_type)"),
    background_tasks: BackgroundTasks = None,
    current_user: UserMe = Depends(get_current_user), # <-- User Dependency
):
    # --- LOGGING DEBUG INFO ---
    user_id_display = getattr(current_user, 'id', 'N/A')
    logger.info(
        f"USER CONTEXT (Upload): ID={user_id_display}, Tenant={current_user.tenant}, Year={current_user.year} (Type: {type(current_user.year)})"
    )
    # --------------------------
    
    if doc_type not in SUPPORTED_DOC_TYPES:
        raise HTTPException(400, detail=f"Invalid doc_type. Must be one of: {SUPPORTED_DOC_TYPES}")

    enabler_code = enabler or DEFAULT_ENABLER # Determine enabler code

    # Folder for file storage (segregated by tenant/year/doc_type/enabler)
    # 💥 REVISED: ใช้ enabler_code ในการสร้าง save_dir
    save_dir = get_save_dir(doc_type, current_user.tenant, current_user.year, enabler_code)
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    # ปรับปรุงการ Sanitized filename ให้มีความยืดหยุ่นมากขึ้นสำหรับภาษาไทย/อักขระพิเศษ
    sanitized_filename = SysPath(file.filename).name
    if not sanitized_filename:
        sanitized_filename = f"uploaded_{timestamp}.tmp"

    file_path = os.path.join(save_dir, f"{timestamp}_{uuid.uuid4().hex}_{sanitized_filename}")
    mock_doc_id = f"temp-{uuid.uuid4().hex}"

    try:
        contents = await file.read()
        await run_in_threadpool(lambda: open(file_path, "wb").write(contents))

        # Process document in background
        if background_tasks:
            background_tasks.add_task(
                process_document,
                file_path=file_path,
                file_name=file.filename,
                stable_doc_uuid=mock_doc_id,
                doc_type=doc_type,
                tenant=current_user.tenant,  # <-- Pass Tenant
                year=str(current_user.year),     # <-- FIX: Ensure year is stored as string
                enabler=enabler_code, # 💥 ADDED: ส่ง enabler_code ไปให้ process_document
            )

        return UploadResponse(
            doc_id=mock_doc_id,
            status="Processing",
            filename=file.filename,
            doc_type=doc_type,
            file_type=os.path.splitext(file.filename)[1],
            upload_date=datetime.now(timezone.utc).isoformat(),
            message="Document accepted for background processing.",
            tenant=current_user.tenant,
            year=current_user.year,
            enabler=enabler_code, # 💥 ADDED: เพิ่ม enabler_code ใน Response
        )
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(500, detail=f"Upload failed: {e}")

# -----------------------------
# --- List uploaded documents ---
# -----------------------------
@upload_router.get("/uploads/{doc_type}", response_model=List[UploadResponse])
async def list_uploads_by_type(
    doc_type: str, 
    current_user: UserMe = Depends(get_current_user) # <-- User Dependency
):
    # --- LOGGING DEBUG INFO ---
    user_id_display = getattr(current_user, 'id', 'N/A')
    logger.info(
        f"USER CONTEXT (List Uploads): ID={user_id_display}, DocType={doc_type}, Tenant={current_user.tenant}, Year={current_user.year} (Filtering with STR year)"
    )
    # --------------------------
    
    # Support "all"
    doc_types_to_fetch = SUPPORTED_DOC_TYPES if doc_type.lower() == "all" else [doc_type]
    
    # 💡 TEMPORARY FIX START: บังคับใช้ DEFAULT_ENABLER สำหรับ 'evidence' 💡
    enabler_to_fetch = None
    
    # ตรวจสอบว่าเป็นการลิสต์ 'evidence' หรือไม่ (และไม่ใช่การลิสต์ 'all')
    if doc_type.lower() == EVIDENCE_DOC_TYPES and len(doc_types_to_fetch) == 1:
        # *** ใช้ค่า DEFAULT_ENABLER จาก config/global_vars.py ***
        enabler_to_fetch = DEFAULT_ENABLER
        logger.warning(
            f"TEMPORARY FIX: Forcing enabler to '{enabler_to_fetch}' for evidence listing "
            f"to bypass core logic issue in list_documents."
        )
    # 💡 TEMPORARY FIX END 💡

    # List documents for the user's specific tenant and year
    doc_data = await run_in_threadpool(
        lambda: list_documents(
            doc_types=doc_types_to_fetch, 
            tenant=current_user.tenant,  # <-- Pass Tenant
            year=str(current_user.year),  # <-- Convert year to string for filtering
            enabler=enabler_to_fetch # 💥 ส่ง enabler ที่ถูกกำหนดชั่วคราวไป
        )
    )
    uploads: List[UploadResponse] = []

    for doc_info in doc_data.values():
        status = "Pending"
        if doc_info.get("status", "").lower() == "failed":
            status = "Failed"
        elif doc_info.get("chunk_count", 0) > 0:
            status = "Ingested"
        elif doc_info.get("status", "").lower() == "processing":
            status = "Processing"

        size_mb = doc_info.get("size", 0) / (1024*1024)
        uploads.append(
            UploadResponse(
                doc_id=doc_info.get("doc_id"),
                filename=doc_info.get("filename"),
                doc_type=doc_info.get("doc_type"),
                file_type=os.path.splitext(doc_info.get("filename"))[1] if doc_info.get("filename") else None,
                status=status,
                upload_date=doc_info.get("upload_date"),
                chunk_count=doc_info.get("chunk_count"),
                size=size_mb,
                enabler=doc_info.get("enabler") or "-",
                tenant=doc_info.get("tenant") or current_user.tenant,
                year=doc_info.get("year") or current_user.year,
            )
        )

    # Sort by doc_type then filename
    return sorted(uploads, key=lambda x: (x.doc_type, x.filename))

# -----------------------------
# --- Manual Ingest API ---
# -----------------------------
@upload_router.post("/ingest")
async def ingest_document(
    file: UploadFile = File(..., description="File to ingest"),
    doc_type: str = Form(..., description=f"Document type: {', '.join(SUPPORTED_DOC_TYPES)}"),
    enabler: Optional[str] = Form(None, description="Enabler code (used for evidence)"),
    current_user: UserMe = Depends(get_current_user), # <-- User Dependency
):
    # --- LOGGING DEBUG INFO ---
    user_id_display = getattr(current_user, 'id', 'N/A')
    logger.info(
        f"USER CONTEXT (Ingest): ID={user_id_display}, DocType={doc_type}, Tenant={current_user.tenant}, Year={current_user.year}"
    )
    # --------------------------

    if doc_type not in SUPPORTED_DOC_TYPES:
        raise HTTPException(400, detail=f"Invalid doc_type: {doc_type}")

    enabler_code = enabler or DEFAULT_ENABLER
    
    # Use segregated save directory
    # 💥 REVISED: ใช้ enabler_code ในการสร้าง save_dir
    save_dir = get_save_dir(doc_type, current_user.tenant, current_user.year, enabler_code)
    os.makedirs(save_dir, exist_ok=True)

    # ใช้ SysPath เพื่อจัดการชื่อไฟล์ให้ปลอดภัยยิ่งขึ้น
    file_path = os.path.join(save_dir, SysPath(file.filename).name)

    try:
        contents = await file.read()
        await run_in_threadpool(lambda: open(file_path, "wb").write(contents))

        # Ingest document
        doc_info = await run_in_threadpool(
            lambda: process_document(
                file_path, 
                SysPath(file.filename).name, 
                doc_type, 
                enabler_code,
                tenant=current_user.tenant, # <-- Pass Tenant
                year=str(current_user.year)      # <-- FIX: Convert year to string for indexing
            )
        )
        return {"status": "success", "doc_info": doc_info.model_dump()}
    except Exception as e:
        logger.error(f"Ingest failed: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(500, detail=str(e))

# -----------------------------
# --- Get all documents ---
# -----------------------------
@upload_router.get("/documents", response_model=List[UploadResponse])
async def get_documents(
    doc_type: Optional[str] = Query(None, description=f"Filter by doc_type: {', '.join(SUPPORTED_DOC_TYPES)}"),
    enabler: Optional[str] = Query(None, description="Filter by enabler (e.g. KM)"),
    current_user: UserMe = Depends(get_current_user), # <-- User Dependency
):
    # --- LOGGING DEBUG INFO ---
    user_id_display = getattr(current_user, 'id', 'N/A')
    logger.info(
        f"USER CONTEXT (Get Docs): ID={user_id_display}, FilterDoc={doc_type}, Tenant={current_user.tenant}, Year={current_user.year} (Filtering with STR year)"
    )
    # --------------------------

    doc_types_to_fetch = [doc_type] if doc_type and doc_type.lower() != "all" else None
    
    # List documents for the user's specific tenant and year
    doc_data = await run_in_threadpool(
        lambda: list_documents(
            doc_types=doc_types_to_fetch, 
            enabler=enabler, 
            tenant=current_user.tenant,  # <-- Pass Tenant
            year=str(current_user.year)       # <-- FIX: Convert year to string for filtering
        )
    )
    uploads: List[UploadResponse] = []

    for doc_info in doc_data.values():
        status = "Pending"
        if doc_info.get("status", "").lower() == "failed":
            status = "Failed"
        elif doc_info.get("chunk_count", 0) > 0:
            status = "Ingested"
        elif doc_info.get("status", "").lower() == "processing":
            status = "Processing"

        size_mb = doc_info.get("size", 0) / (1024*1024)
        uploads.append(
            UploadResponse(
                doc_id=doc_info.get("doc_id"),
                filename=doc_info.get("filename"),
                doc_type=doc_info.get("doc_type"),
                file_type=os.path.splitext(doc_info.get("filename"))[1] if doc_info.get("filename") else None,
                status=status,
                upload_date=doc_info.get("upload_date"),
                chunk_count=doc_info.get("chunk_count"),
                size=size_mb,
                enabler=doc_info.get("enabler") or "-",
                tenant=doc_info.get("tenant") or current_user.tenant,
                year=doc_info.get("year") or current_user.year,
            )
        )

    return sorted(uploads, key=lambda x: (x.doc_type, x.filename))

# -----------------------------
# --- Delete document ---
# -----------------------------
@upload_router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    current_user: UserMe = Depends(get_current_user) # <-- User Dependency
):
    # --- LOGGING DEBUG INFO ---
    user_id_display = getattr(current_user, 'id', 'N/A')
    logger.info(
        f"USER CONTEXT (Delete Doc): ID={user_id_display}, DocID={doc_id}, Tenant={current_user.tenant}, Year={current_user.year}"
    )
    # --------------------------
    
    # NOTE: เพื่อให้ปลอดภัย ต้องแน่ใจว่า Doc ID เป็นของ User นี้เท่านั้น
    success = await run_in_threadpool(
        lambda: delete_document_by_uuid(
            doc_id, 
            tenant=current_user.tenant,  # <-- Pass Tenant
            year=str(current_user.year)       # <-- FIX: Convert year to string for deletion lookup
        )
    )
    if not success:
        # ถ้าไม่พบไฟล์ อาจเป็นเพราะ Doc ID ผิด หรือ User ไม่มีสิทธิ์เข้าถึง
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, 
            detail=f"Document {doc_id} not found or access denied (Tenant: {current_user.tenant}, Year: {current_user.year})."
        )
    return {"status": "success", "message": f"Document {doc_id} deleted successfully."}

# -----------------------------
# --- Download uploaded file ---
# -----------------------------
@upload_router.get("/upload/{doc_type}/{file_id}/download")
async def download_upload(
    doc_type: str, 
    file_id: str,
    current_user: UserMe = Depends(get_current_user) # <-- User Dependency
):
    # --- LOGGING DEBUG INFO ---
    user_id_display = getattr(current_user, 'id', 'N/A')
    logger.info(
        f"USER CONTEXT (Download): ID={user_id_display}, DocType={doc_type}, FileID={file_id}, Tenant={current_user.tenant}, Year={current_user.year}"
    )
    # --------------------------

    # 1. ดึงเอกสารเฉพาะของ Tenant/Year นี้เท่านั้น และตรวจสอบ Doc ID
    doc_data = await run_in_threadpool(
        lambda: list_documents(
            doc_types=[doc_type], 
            tenant=current_user.tenant,  # <-- Pass Tenant
            year=str(current_user.year)       # <-- FIX: Convert year to string for filtering
        )
    )
    doc_map = {item["doc_id"]: item for item in doc_data.values()}

    target = doc_map.get(file_id)
    if not target:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, 
            detail="Document ID not found or access denied."
        )

    # 2. ตรวจสอบ Path และส่งไฟล์
    filepath = target.get("filepath")
    if not filepath or not os.path.exists(filepath):
        logger.error(f"File path missing for doc_id {file_id}: {filepath}")
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="File not found on disk")

    return FileResponse(filepath, filename=target.get("filename", "download.bin"))