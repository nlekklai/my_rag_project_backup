# routers/upload_router.py
# routers/upload_router.py (Refactored)
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Path, Query
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import List, Optional, Dict, Union
from datetime import datetime, timezone
import logging
import os, sys, uuid
from pathlib import Path as SysPath

# -----------------------------
# --- Import Project Modules ---
# -----------------------------
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.append(project_root)

    # --- Global vars ---
    from config.global_vars import (
        DATA_DIR,
        VECTORSTORE_DIR,
        SUPPORTED_TYPES,
        SUPPORTED_DOC_TYPES,
        DEFAULT_ENABLER,
    )

    # --- Core logic functions ---
    from core.ingest import (
        process_document,
        list_documents,
        delete_document_by_uuid,
        DocInfo,   # Type for document metadata
    )

    # --- Vectorstore & Prompts ---
    from core.vectorstore import vectorstore_exists, FINAL_K_RERANKED
    from core.rag_prompts import QA_PROMPT, COMPARE_PROMPT

except ImportError as e:
    print(f"❌ CRITICAL IMPORT ERROR: {e}")
    raise


# -----------------------------
# --- Setup Router & Logger ---
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


# -----------------------------
# --- Upload Document ---
# -----------------------------
@upload_router.post("/upload/{doc_type}", response_model=UploadResponse)
async def upload_document(
    doc_type: str = Path(..., description=f"Document type. Must be one of: {SUPPORTED_DOC_TYPES}"),
    file: UploadFile = File(..., description="Document file to upload"),
    background_tasks: BackgroundTasks = None,
):
    """
    ✅ Upload document and process asynchronously
    """
    if doc_type not in SUPPORTED_DOC_TYPES:
        raise HTTPException(400, detail=f"Invalid doc_type. Must be one of: {SUPPORTED_DOC_TYPES}")

    os.makedirs(DATA_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    sanitized_filename = "".join(
        c for c in file.filename if c.isalnum() or c in (".", "_", "-")
    ).rstrip() or f"uploaded_{timestamp}.tmp"
    temp_file_path = os.path.join(DATA_DIR, f"{timestamp}_{uuid.uuid4().hex}_{sanitized_filename}")
    mock_doc_id = f"temp-{uuid.uuid4().hex}"

    try:
        # เขียนไฟล์ลงดิสก์
        contents = await file.read()
        await run_in_threadpool(lambda: open(temp_file_path, "wb").write(contents))

        # ประมวลผลแบบ background
        background_tasks.add_task(
            process_document,
            file_path=temp_file_path,
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
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(500, detail=f"Failed to process upload: {e}")


# -----------------------------
# --- List Uploaded Documents ---
# -----------------------------
@upload_router.get("/uploads/{doc_type}", response_model=List[UploadResponse])
async def list_uploads_by_type(doc_type: str):
    """
    🔍 List uploaded documents by type (or 'all')
    """
    doc_type_to_fetch = [doc_type] if doc_type.lower() != "all" else SUPPORTED_DOC_TYPES
    doc_data = await run_in_threadpool(lambda: list_documents(doc_types=doc_type_to_fetch))

    uploads: List[UploadResponse] = []
    if isinstance(doc_data, list):
        doc_map = {item.get("doc_id") or item.get("stable_doc_uuid"): item for item in doc_data}
    else:
        doc_map = doc_data or {}

    for doc_id, info in doc_map.items():
        file_name = info.get("filename", "Unknown")
        file_path = info.get("file_path")
        doc_type_real = info.get("doc_type", "").lower()

        if doc_type.lower() != "all" and doc_type_real != doc_type.lower():
            continue

        upload_time = datetime.now(timezone.utc)
        if file_path and os.path.exists(file_path):
            upload_time = datetime.fromtimestamp(os.path.getmtime(file_path), tz=timezone.utc)

        status = "Pending"
        if info.get("ingestion_status") == "failed":
            status = "Failed"
        elif info.get("chunk_count", 0) > 0:
            status = "Ingested"
        elif info.get("ingestion_status") == "processing":
            status = "Processing"

        uploads.append(
            UploadResponse(
                doc_id=doc_id,
                filename=file_name,
                doc_type=doc_type_real,
                file_type=os.path.splitext(file_name)[1],
                status=status,
                upload_date=upload_time.isoformat(),
                message=info.get("error_message"),
            )
        )

    return sorted(uploads, key=lambda x: x.filename)


# -----------------------------
# --- Delete Uploaded File ---
# -----------------------------
@upload_router.delete("/upload/{doc_type}/{file_id}")
async def delete_upload_by_id(doc_type: str, file_id: str):
    """
    ❌ Delete document & vectorstore
    """
    try:
        await run_in_threadpool(
            lambda: delete_document_by_uuid(file_id, doc_type=doc_type, enabler=None)
        )
        return {"status": f"Document {file_id} deletion initiated."}
    except FileNotFoundError:
        raise HTTPException(404, detail=f"File not found for ID: {file_id}")
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        raise HTTPException(500, detail=f"Failed to delete document: {e}")


# -----------------------------
# --- Download Uploaded File ---
# -----------------------------
@upload_router.get("/upload/{doc_type}/{file_id}/download")
async def download_upload(doc_type: str, file_id: str):
    """
    📥 Download uploaded file
    """
    doc_data = await run_in_threadpool(lambda: list_documents(doc_types=[doc_type]))
    doc_map = {item.get("doc_id") or item.get("stable_doc_uuid"): item for item in doc_data}
    target = doc_map.get(file_id)

    if not target:
        raise HTTPException(404, detail="Document ID not found")

    filepath = target.get("file_path")
    if not filepath or not os.path.exists(filepath):
        raise HTTPException(404, detail="File not found on disk")

    return FileResponse(filepath, filename=target.get("filename", "download.bin"))


# -----------------------------
# --- Manual Ingest (API) ---
# -----------------------------
@upload_router.post("/ingest")
async def ingest_document(
    file: UploadFile = File(..., description="ไฟล์เอกสารที่จะอัปโหลดและ ingest"),
    doc_type: str = Form(..., description=f"ประเภทเอกสาร: {', '.join(SUPPORTED_DOC_TYPES)}"),
    enabler: Optional[str] = Form(None, description="รหัส Enabler (ใช้กับ evidence เท่านั้น)"),
):
    """
    🚀 Ingest document manually (blocking)
    """
    if doc_type not in SUPPORTED_DOC_TYPES:
        raise HTTPException(400, detail=f"Invalid doc_type. Must be one of: {SUPPORTED_DOC_TYPES}")

    enabler = enabler or DEFAULT_ENABLER
    file_path = os.path.join(DATA_DIR, SysPath(file.filename).name)

    try:
        # Save file
        contents = await file.read()
        await run_in_threadpool(lambda: open(file_path, "wb").write(contents))

        # Ingest
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
# --- Get All Documents ---
# -----------------------------
@upload_router.get("/documents", response_model=List[DocInfo])
async def get_documents(
    doc_type: Optional[str] = Query(None, description=f"Filter by doc_type: {', '.join(SUPPORTED_DOC_TYPES)}"),
    enabler: Optional[str] = Query(None, description="Filter by enabler (e.g. KM)"),
):
    """📋 Return list of ingested documents"""
    return await run_in_threadpool(lambda: list_documents(doc_type, enabler))


# -----------------------------
# --- Delete Document ---
# -----------------------------
@upload_router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """🗑️ Delete document from RAG system"""
    success = await run_in_threadpool(lambda: delete_document_by_uuid(doc_id))
    if not success:
        raise HTTPException(404, detail=f"Document {doc_id} not found or failed to delete.")
    return {"status": "success", "message": f"Document {doc_id} deleted successfully."}
