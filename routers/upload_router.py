# routers/upload_router.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Path, Query
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
        DEFAULT_ENABLER,
    )
    from core.ingest import (
        process_document,
        list_documents,
        delete_document_by_uuid,
        DocInfo,
    )

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

# -----------------------------
# --- Upload with background processing ---
# -----------------------------
@upload_router.post("/upload/{doc_type}", response_model=UploadResponse)
async def upload_document(
    doc_type: str = Path(..., description=f"Document type. Must be one of: {SUPPORTED_DOC_TYPES}"),
    file: UploadFile = File(..., description="Document file to upload"),
    background_tasks: BackgroundTasks = None,
):
    if doc_type not in SUPPORTED_DOC_TYPES:
        raise HTTPException(400, detail=f"Invalid doc_type. Must be one of: {SUPPORTED_DOC_TYPES}")

    # Folder for file storage
    save_dir = os.path.join(DATA_DIR, doc_type)
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    sanitized_filename = "".join(
        c for c in file.filename if c.isalnum() or c in (".", "_", "-")
    ).rstrip() or f"uploaded_{timestamp}.tmp"

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
            )

        return UploadResponse(
            doc_id=mock_doc_id,
            status="Processing",
            filename=file.filename,
            doc_type=doc_type,
            file_type=os.path.splitext(file.filename)[1],
            upload_date=datetime.now(timezone.utc).isoformat(),
            message="Document accepted for background processing.",
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
async def list_uploads_by_type(doc_type: str):
    # Support "all"
    doc_types_to_fetch = SUPPORTED_DOC_TYPES if doc_type.lower() == "all" else [doc_type]
    
    doc_data = await run_in_threadpool(lambda: list_documents(doc_types=doc_types_to_fetch))
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
                enabler=doc_info.get("enabler") or "-"
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
):
    if doc_type not in SUPPORTED_DOC_TYPES:
        raise HTTPException(400, detail=f"Invalid doc_type: {doc_type}")

    enabler = enabler or DEFAULT_ENABLER
    save_dir = DATA_DIR if doc_type != "evidence" else os.path.join(DATA_DIR, f"evidence_{enabler}")
    os.makedirs(save_dir, exist_ok=True)

    file_path = os.path.join(save_dir, SysPath(file.filename).name)

    try:
        contents = await file.read()
        await run_in_threadpool(lambda: open(file_path, "wb").write(contents))

        # Ingest document
        doc_info = await run_in_threadpool(
            lambda: process_document(file_path, SysPath(file.filename).name, doc_type, enabler)
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
):
    doc_types_to_fetch = [doc_type] if doc_type and doc_type.lower() != "all" else None
    doc_data = await run_in_threadpool(lambda: list_documents(doc_types=doc_types_to_fetch, enabler=enabler))
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
                enabler=doc_info.get("enabler") or "-"
            )
        )

    return sorted(uploads, key=lambda x: (x.doc_type, x.filename))

# -----------------------------
# --- Delete document ---
# -----------------------------
@upload_router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    success = await run_in_threadpool(lambda: delete_document_by_uuid(doc_id))
    if not success:
        raise HTTPException(404, detail=f"Document {doc_id} not found or failed to delete.")
    return {"status": "success", "message": f"Document {doc_id} deleted successfully."}

# -----------------------------
# --- Download uploaded file ---
# -----------------------------
@upload_router.get("/upload/{doc_type}/{file_id}/download")
async def download_upload(doc_type: str, file_id: str):
    doc_data = await run_in_threadpool(lambda: list_documents(doc_types=[doc_type]))
    doc_map = {item["doc_id"]: item for item in doc_data.values()}

    target = doc_map.get(file_id)
    if not target:
        raise HTTPException(404, detail="Document ID not found")

    filepath = target.get("filepath")
    if not filepath or not os.path.exists(filepath):
        raise HTTPException(404, detail="File not found on disk")

    return FileResponse(filepath, filename=target.get("filename", "download.bin"))

