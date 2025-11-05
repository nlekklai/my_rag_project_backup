# routers/assessment_router.py

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Path
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool
from typing import Optional, List
from datetime import datetime, timezone
from pydantic import BaseModel, Field
import logging, os

from core.run_assessment import run_assessment_process
from models.llm import llm as llm_instance

logger = logging.getLogger(__name__)

assessment_router = APIRouter(
    prefix="/api/assess",
    tags=["Assessment"],
)

class AssessmentRequest(BaseModel):
    enabler: str = Field(..., description="รหัส Enabler เช่น KM")
    sub_criteria_id: Optional[str] = Field(None, description="รหัสย่อยของเกณฑ์ เช่น 1.1")
    mode: str = Field(default="full", description="โหมดการประเมิน: full / partial / preview")

class AssessmentRecord(BaseModel):
    record_id: str
    enabler: str
    sub_criteria_id: Optional[str]
    mode: str
    timestamp: str
    status: str
    overall_score: float = 0.0
    highest_full_level: int = 0
    export_path: Optional[str] = None

ASSESSMENT_HISTORY: List[AssessmentRecord] = []

@assessment_router.post("/")
async def run_assessment_task(request: AssessmentRequest, background_tasks: BackgroundTasks):
    if llm_instance is None:
        raise HTTPException(status_code=503, detail="LLM service is not available.")

    record_id = os.urandom(8).hex()
    os.makedirs("exports", exist_ok=True)

    record = AssessmentRecord(
        record_id=record_id,
        enabler=request.enabler.upper(),
        sub_criteria_id=request.sub_criteria_id,
        mode=request.mode,
        timestamp=datetime.now(timezone.utc).isoformat(),
        status="RUNNING"
    )
    ASSESSMENT_HISTORY.append(record)

    background_tasks.add_task(_background_assessment_runner, record_id, request)
    return {"record_id": record_id, "status": "accepted", "message": "Assessment started."}

async def _background_assessment_runner(record_id: str, request: AssessmentRequest):
    try:
        mode_map = {"full": "real", "partial": "random", "preview": "mock"}
        mode_to_use = mode_map.get(request.mode.lower(), "real")

        result = await run_in_threadpool(
            run_assessment_process,
            request.enabler,
            request.sub_criteria_id or "all",
            mode_to_use,
            filter_mode=False,
            export=True,
            disable_semantic_filter=False,
            allow_fallback=True,
            external_retriever=None
        )

        for record in ASSESSMENT_HISTORY:
            if record.record_id == record_id:
                record.status = "COMPLETED"
                overall = result.get("Overall", {})
                record.overall_score = overall.get("overall_maturity_score", 0.0)
                record.highest_full_level = overall.get("overall_maturity_level", 0)
                record.export_path = result.get("export_path_used")
                break

    except Exception as e:
        logger.exception(f"❌ Assessment {record_id} failed: {e}")
        for record in ASSESSMENT_HISTORY:
            if record.record_id == record_id:
                record.status = "FAILED"
                break

@assessment_router.get("/history", response_model=List[AssessmentRecord])
async def get_history(enabler: Optional[str] = Query(None)):
    data = ASSESSMENT_HISTORY
    if enabler:
        data = [r for r in data if r.enabler.upper() == enabler.upper()]
    return sorted(data, key=lambda r: r.timestamp, reverse=True)

@assessment_router.get("/results/{record_id}")
async def get_result(record_id: str):
    record = next((r for r in ASSESSMENT_HISTORY if r.record_id == record_id), None)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found.")
    if record.status != "COMPLETED":
        raise HTTPException(status_code=400, detail=f"Status is {record.status}. Wait until COMPLETED.")
    if record.export_path and os.path.exists(record.export_path):
        return FileResponse(record.export_path, media_type="application/json", filename=os.path.basename(record.export_path))
    raise HTTPException(status_code=404, detail="File not found.")
