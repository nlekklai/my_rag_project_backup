from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Path, Query, Depends, status
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import List, Optional, Union, Tuple
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
        SUPPORTED_DOC_TYPES,
        DEFAULT_ENABLER, 
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
    
    # 🟢 FIX: Import Path Utility และ Central Logic ที่ถูกย้ายมา
    from utils.path_utils import get_document_source_dir, get_normalized_metadata 

except ImportError as e:
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
    year: Optional[Union[int, str]] = None 

# -----------------------------
# --- Helper Function for File Path (USES CENTRAL LOGIC) ---
# -----------------------------
def get_save_dir(doc_type: str, tenant: str, year: Optional[Union[int, str]], enabler: Optional[str] = None) -> str:
    """
    Constructs the segregated directory path for saving files using Path Utility.
    The actual path is normalized based on doc_type (Global vs. Evidence).
    """
    
    # 📌 ใช้ Central Logic เพื่อหาค่า Year/Enabler ที่ถูก Normalize สำหรับ Path
    normalized_year, normalized_enabler = get_normalized_metadata(
        doc_type=doc_type,
        year_input=year, 
        enabler_input=enabler,
        default_enabler=DEFAULT_ENABLER
    )
    
    return get_document_source_dir(
        tenant=tenant,
        # ใช้ normalized values สำหรับ path construction
        year=normalized_year, 
        enabler=normalized_enabler, 
        doc_type=doc_type
    )

# -----------------------------
# --- Upload with background processing ---
# -----------------------------
@upload_router.post("/upload/{doc_type}", response_model=UploadResponse)
async def upload_document(
    doc_type: str = Path(..., description=f"Document type. Must be one of: {SUPPORTED_DOC_TYPES}"),
    file: UploadFile = File(..., description="Document file to upload"),
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
        
    # 📌 ใช้ Logic Normalized Metadata สำหรับการบันทึก (ปีและ Enabler ที่จะถูกบันทึกใน Mapping DB)
    normalized_year, normalized_enabler = get_normalized_metadata(
        doc_type=doc_type,
        year_input=current_user.year,
        enabler_input=enabler,
        default_enabler=DEFAULT_ENABLER
    )

    # Folder for file storage (ส่งค่าเดิมไปให้ get_save_dir ซึ่งจะ Normalize ภายใน)
    save_dir = get_save_dir(doc_type, current_user.tenant, current_user.year, enabler)
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
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
                tenant=current_user.tenant,
                # 📌 ส่งค่า Normalized ไปให้ process_document เพื่อบันทึกใน Mapping DB
                year=normalized_year, 
                enabler=normalized_enabler,
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
            enabler=normalized_enabler,
        )
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(500, detail=f"Upload failed: {e}")

# -----------------------------
# --- List uploaded documents (CLEANED) ---
# -----------------------------
@upload_router.get("/uploads/{doc_type}", response_model=List[UploadResponse])
async def list_uploads_by_type(
    doc_type: str, 
    filter_year: Optional[int] = Query(None, alias="year", description="Filter by year (overrides user's year)"),
    filter_enabler: Optional[str] = Query(None, alias="enabler", description="Filter by enabler code (e.g. KM)"),
    current_user: UserMe = Depends(get_current_user) # <-- User Dependency
):
    # --- LOGGING DEBUG INFO ---
    user_id_display = getattr(current_user, 'id', 'N/A')
    target_year_log = str(filter_year) if filter_year is not None else str(current_user.year)
    logger.info(
        f"USER CONTEXT (List Uploads): ID={user_id_display}, DocType={doc_type}, Tenant={current_user.tenant}, Year={target_year_log} (Filtering with STR year)"
    )
    # --------------------------
    
    # 📌 ใช้ CENTRAL LOGIC เพื่อหาค่า Year/Enabler ที่จะใช้ในการ Query
    year_to_fetch, enabler_to_fetch = get_normalized_metadata(
        doc_type=doc_type,
        year_input=filter_year, 
        enabler_input=filter_enabler,
        default_enabler=DEFAULT_ENABLER
    )
    
    tenant_to_fetch = current_user.tenant
    
    # Support "all"
    doc_types_to_fetch = SUPPORTED_DOC_TYPES if doc_type.lower() == "all" else [doc_type]
    
    # List documents for the user's specific tenant and year
    doc_data = await run_in_threadpool(
        lambda: list_documents(
            doc_types=doc_types_to_fetch, 
            tenant=tenant_to_fetch,
            year=year_to_fetch, 
            enabler=enabler_to_fetch 
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

    # 📌 ใช้ Logic Normalized Metadata สำหรับการบันทึก
    normalized_year, normalized_enabler = get_normalized_metadata(
        doc_type=doc_type,
        year_input=current_user.year,
        enabler_input=enabler,
        default_enabler=DEFAULT_ENABLER
    )
    
    # Use segregated save directory (ส่งค่าเดิมไปให้ get_save_dir ซึ่งจะ Normalize ภายใน)
    save_dir = get_save_dir(doc_type, current_user.tenant, current_user.year, enabler)
    os.makedirs(save_dir, exist_ok=True)

    # ใช้ SysPath เพื่อจัดการชื่อไฟล์ให้ปลอดภัยยิ่งขึ้น
    file_path = os.path.join(save_dir, SysPath(file.filename).name)

    try:
        contents = await file.read()
        await run_in_threadpool(lambda: open(file_path, "wb").write(contents))

        # Ingest document
        chunks, stable_doc_uuid, doc_type_result = await run_in_threadpool(
            lambda: process_document(
                file_path=file_path, 
                file_name=SysPath(file.filename).name, 
                stable_doc_uuid=str(uuid.uuid4().hex), 
                doc_type=doc_type, 
                enabler=normalized_enabler, # 📌 ใช้ค่า Normalized
                tenant=current_user.tenant, 
                year=normalized_year # 📌 ใช้ค่า Normalized
            )
        )
        
        return {
             "status": "success", 
             "doc_info": {
                 "doc_id": stable_doc_uuid,
                 "doc_type": doc_type_result,
                 "chunk_count": len(chunks) if chunks else 0
             }
        }
    except Exception as e:
        logger.error(f"Ingest failed: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(500, detail=str(e))

# -----------------------------
# --- Get all documents (CLEANED) ---
# -----------------------------
@upload_router.get("/documents", response_model=List[UploadResponse])
async def get_documents(
    doc_type: Optional[str] = Query(None, description=f"Filter by doc_type: {', '.join(SUPPORTED_DOC_TYPES)}"),
    enabler: Optional[str] = Query(None, description="Filter by enabler (e.g. KM)"),
    filter_year: Optional[int] = Query(None, alias="year", description="Filter by year (overrides user's year)"),
    current_user: UserMe = Depends(get_current_user), # <-- User Dependency
):
    # --- LOGGING DEBUG INFO ---
    user_id_display = getattr(current_user, 'id', 'N/A')
    logger.info(
        f"USER CONTEXT (Get Docs): ID={user_id_display}, FilterDoc={doc_type}, Tenant={current_user.tenant}, Year={current_user.year} (Filtering with STR year)"
    )
    # --------------------------

    doc_types_to_fetch = [doc_type] if doc_type and doc_type.lower() != "all" else None
    
    # 📌 ใช้ CENTRAL LOGIC เพื่อหาค่า Year/Enabler ที่จะใช้ในการ Query
    doc_type_for_filter = doc_type if doc_type else EVIDENCE_DOC_TYPES 
    
    year_to_fetch, enabler_to_fetch = get_normalized_metadata(
        doc_type=doc_type_for_filter,
        year_input=filter_year, 
        enabler_input=enabler,
        default_enabler=DEFAULT_ENABLER
    )
    
    tenant_to_fetch = current_user.tenant

    # List documents for the user's specific tenant and year
    doc_data = await run_in_threadpool(
        lambda: list_documents(
            doc_types=doc_types_to_fetch, 
            enabler=enabler_to_fetch, 
            tenant=tenant_to_fetch,
            year=year_to_fetch 
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
    # 💡 REVISED: เพิ่ม doc_type และ enabler เพื่อให้การลบถูกต้องตามบริบทของ Path
    doc_type: Optional[str] = Query(None, description=f"Document type of the file being deleted"),
    enabler: Optional[str] = Query(None, description="Enabler code (used for evidence)"),
    current_user: UserMe = Depends(get_current_user) # <-- User Dependency
):
    # --- LOGGING DEBUG INFO ---
    user_id_display = getattr(current_user, 'id', 'N/A')
    logger.info(
        f"USER CONTEXT (Delete Doc): ID={user_id_display}, DocID={doc_id}, Tenant={current_user.tenant}, Year={current_user.year}, DocType={doc_type}, Enabler={enabler}"
    )
    # --------------------------
    
    # 1. 💡 REVISED: ใช้ get_normalized_metadata เพื่อหาบริบทปีและ enabler ที่ใช้ในการค้นหา Mapping
    doc_type_for_lookup = doc_type if doc_type else EVIDENCE_DOC_TYPES # หากไม่ระบุ ให้เดาว่าเป็น Evidence

    normalized_year, normalized_enabler = get_normalized_metadata(
        doc_type=doc_type_for_lookup,
        year_input=current_user.year,
        enabler_input=enabler,
        default_enabler=DEFAULT_ENABLER
    )

    success = await run_in_threadpool(
        lambda: delete_document_by_uuid(
            doc_id, 
            tenant=current_user.tenant,
            # 📌 ส่งค่า Normalized ไปให้ delete_document_by_uuid
            year=normalized_year, 
            enabler=normalized_enabler
        )
    )
    if not success:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, 
            detail=(
                f"Document {doc_id} not found or access denied. "
                f"(Tenant: {current_user.tenant}, Context: {normalized_year}/{normalized_enabler})"
            )
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

    # 📌 ใช้ CENTRAL LOGIC เพื่อหาค่า Year/Enabler ในการ Query
    year_to_fetch, enabler_to_fetch = get_normalized_metadata(
        doc_type=doc_type,
        year_input=current_user.year,
        enabler_input=None, # ไม่มีการกรอง enabler จาก Query
        default_enabler=DEFAULT_ENABLER
    )

    doc_data = await run_in_threadpool(
        lambda: list_documents(
            doc_types=[doc_type], 
            tenant=current_user.tenant,
            year=year_to_fetch,
            enabler=enabler_to_fetch
        )
    )
    
    doc_map = doc_data 

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