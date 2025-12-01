# routers/assessment_router.py
import os
import uuid
import logging
import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Path
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field

# ------------------- Core & LLM -------------------
from core.seam_assessment import run_assessment
# from models.llm import llm as llm_instance
from models.llm import create_llm_instance

from config.global_vars import (
    DEFAULT_ENABLER,
    EVIDENCE_DOC_TYPES,
    FINAL_K_RERANKED,
    QUERY_INITIAL_K,
    QUERY_FINAL_K,
    LLM_MODEL_NAME
)


logger = logging.getLogger(__name__)

assessment_router = APIRouter(
    prefix="/api/assess",
    tags=["Assessment"]
)

# ------------------- Pydantic Models -------------------
class StartAssessmentRequest(BaseModel):
    enabler: str = Field(..., example="KM", description="รหัส Enabler เช่น KM, DG")
    sub_criteria_id: Optional[str] = Field(
        None,
        example="1.2",
        description="รหัสย่อย เช่น 1.2 หรือ all หรือเว้นว่าง = all"
    )
    sequential: bool = Field(
        True,
        description="เปิดใช้ Sequential Baseline (ส่งต่อหลักฐาน Level ก่อนหน้า) – แนะนำเปิด"
    )

class AssessmentStatus(BaseModel):
    record_id: str = Field(..., description="ID สำหรับติดตามผล")
    enabler: str
    sub_criteria_id: str
    sequential: bool
    status: str  # RUNNING | COMPLETED | FAILED
    started_at: str
    finished_at: Optional[str] = None
    overall_score: Optional[float] = None
    highest_level: Optional[int] = None
    export_path: Optional[str] = None
    message: str = "Assessment in progress..."

# ------------------- In-memory Store (รีสตาร์ทแล้วหาย) -------------------
ASSESSMENT_RECORDS: Dict[str, AssessmentStatus] = {}

# ------------------- Background Runner -------------------
async def _run_assessment_background(record_id: str, request: StartAssessmentRequest):
    record = ASSESSMENT_RECORDS[record_id]

    try:
        logger.info(
            f"Assessment STARTED → ID: {record_id} | "
            f"Enabler: {request.enabler} | Sub: {request.sub_criteria_id or 'all'} | "
            f"Sequential: {request.sequential}"
        )

        # เรียก engine จริง (รองรับ sequential flag แล้ว)
        result: dict = await run_assessment(
            enabler_id=request.enabler.upper(),
            sub_criteria_id=request.sub_criteria_id or "all",
            mode="real",
            filter_mode=False,
            export=True,
            disable_semantic_filter=False,
            allow_fallback=True,
            external_retriever=None,
            sequential=request.sequential  # ส่งเข้า engine
        )

        export_path = result.get("export_path_used")
        if not export_path or not os.path.exists(export_path):
            raise Exception("Export file was not created")

        # อัปเดต record เมื่อสำเร็จ
        overall = result.get("Overall", {})
        record.status = "COMPLETED"
        record.finished_at = datetime.now(timezone.utc).isoformat()
        record.overall_score = overall.get("overall_maturity_score", 0.0)
        record.highest_level = overall.get("overall_maturity_level", 0)
        record.export_path = export_path
        record.message = "Assessment completed successfully"

        logger.info(f"Assessment COMPLETED → {record_id} | Score: {record.overall_score}")

    except Exception as e:
        logger.exception(f"Assessment FAILED → {record_id}")
        record.status = "FAILED"
        record.finished_at = datetime.now(timezone.utc).isoformat()
        record.message = f"Error: {str(e)}"

# ------------------- API Endpoints -------------------

@assessment_router.post("/start", response_model=AssessmentStatus)
async def start_assessment(
    request: StartAssessmentRequest,
    background_tasks: BackgroundTasks
):

    
    llm = create_llm_instance(model_name=LLM_MODEL_NAME, temperature=0.0)
    if not llm:
        raise HTTPException(status_code=503, detail="LLM service unavailable")


    record_id = uuid.uuid4().hex[:12]  # สั้น ๆ อ่านง่าย
    os.makedirs("exports", exist_ok=True)

    record = AssessmentStatus(
        record_id=record_id,
        enabler=request.enabler.upper(),
        sub_criteria_id=request.sub_criteria_id or "all",
        sequential=request.sequential,
        status="RUNNING",
        started_at=datetime.now(timezone.utc).isoformat(),
        message="Assessment queued and running..."
    )
    ASSESSMENT_RECORDS[record_id] = record

    background_tasks.add_task(_run_assessment_background, record_id, request)

    return record


@assessment_router.get("/status/{record_id}", response_model=AssessmentStatus)
async def get_status(record_id: str = Path(..., description="Record ID จาก /start")):
    record = ASSESSMENT_RECORDS.get(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    return record


@assessment_router.get("/results/{record_id}")
async def get_results_json(record_id: str = Path(..., description="ดึงผลลัพธ์เป็น JSON พร้อม ai_confidence")):
    record = ASSESSMENT_RECORDS.get(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    if record.status != "COMPLETED":
        raise HTTPException(status_code=425, detail=f"ยังไม่เสร็จ (สถานะ: {record.status})")

    if not record.export_path or not os.path.exists(record.export_path):
        raise HTTPException(status_code=404, detail="ไฟล์ผลลัพธ์หาย")

    with open(record.export_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return JSONResponse(content=data)


@assessment_router.get("/download/{record_id}")
async def download_result_file(record_id: str = Path(...)):
    record = ASSESSMENT_RECORDS.get(record_id)
    if not record or record.status != "COMPLETED" or not record.export_path:
        raise HTTPException(status_code=404, detail="Result not ready or file missing")

    return FileResponse(
        path=record.export_path,
        media_type="application/json",
        filename=os.path.basename(record.export_path)
    )


@assessment_router.get("/history")
async def get_assessment_history(enabler: Optional[str] = None):
    """ดูประวัติการประเมินทั้งหมด"""
    items = list(ASSESSMENT_RECORDS.values())
    if enabler:
        items = [i for i in items if i.enabler == enabler.upper()]
    return sorted(items, key=lambda x: x.started_at, reverse=True)