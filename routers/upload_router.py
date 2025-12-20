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

# --- Import Auth, Path Utils & Global Vars ---
from routers.auth_router import UserMe, get_current_user
from utils.path_utils import (
    get_document_source_dir,
    load_doc_id_mapping,
    save_doc_id_mapping,
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
        )
        for uid, info in mapping_data.items()
    ]


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
# 2. POST: Upload
# =========================

@upload_router.post("/assessment")
async def upload_file(
    file: UploadFile = File(...),
    type: str = Form(...),
    enabler: Optional[str] = Form(None),
    year: Optional[int] = Form(None),
    current_user: UserMe = Depends(get_current_user),
):
    try:
        tenant = current_user.tenant
        target_year = year or getattr(current_user, "year", DEFAULT_YEAR)
        target_enabler = enabler or DEFAULT_ENABLER

        # 1. Save physical file
        save_dir = get_document_source_dir(
            tenant, target_year, target_enabler, type
        )
        os.makedirs(save_dir, exist_ok=True)

        file_path = os.path.join(save_dir, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 2. Create mapping entry
        doc_id = str(uuid.uuid4())
        relative_key = get_mapping_key_from_physical_path(file_path)

        new_entry = {
            "file_name": file.filename,
            "filepath": relative_key,
            "status": "Pending",
            "enabler": target_enabler,
            "year": target_year,
            "file_size": os.path.getsize(file_path),
            "upload_date": datetime.now().isoformat(),
        }

        # 3. Load → Update → Save mapping
        mapping = load_doc_id_mapping(
            type, tenant, target_year, target_enabler
        )
        mapping[doc_id] = new_entry
        save_doc_id_mapping(
            mapping, type, tenant, target_year, target_enabler
        )

        return {"message": "Upload successful", "doc_id": doc_id}

    except Exception as e:
        logger.exception("Upload error")
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
# 5. POST: Ingest
# =========================

@upload_router.post("/ingest", response_model=IngestResponse)
async def ingest_files(
    request: IngestRequest,
    current_user: UserMe = Depends(get_current_user),
):
    results = []

    for doc_id in request.doc_ids:
        # TODO: implement actual ingestion logic here (load → chunk → vectorstore)
        results.append(IngestResult(doc_id=doc_id, result="Success"))
        # Optionally update mapping status → "Ingested"

    return IngestResponse(results=results)
