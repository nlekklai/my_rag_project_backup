# routers/assessment_router.py

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Path
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool
from typing import Optional, List, Final, Any, Dict
from datetime import datetime, timezone
from pydantic import BaseModel, Field
import logging, os
import uuid 

from core.seam_assessment import run_assessment
from models.llm import llm as llm_instance

logger = logging.getLogger(__name__)

assessment_router = APIRouter(
    prefix="/api/assess",
    tags=["Assessment"],
)

# --- Pydantic Models ---

class AssessmentRequest(BaseModel):
    enabler: str = Field(..., description="รหัส Enabler เช่น KM")
    sub_criteria_id: Optional[str] = Field(None, description="รหัสย่อยของเกณฑ์ เช่น 1.1")
    mode: str = Field(default="full", description="โหมดการประเมิน: full / partial / preview")

class AssessmentRecord(BaseModel):
    record_id: str = Field(..., description="ID เฉพาะสำหรับการรันนี้")
    enabler: str
    sub_criteria_id: Optional[str]
    mode: str
    timestamp: str
    status: str
    overall_score: float = 0.0
    highest_full_level: int = 0
    export_path: Optional[str] = None

# ⚠️ Note: This list is not persistent across restarts.
ASSESSMENT_HISTORY: List[AssessmentRecord] = []

# ---------------------------------------------------------------------

# Helper Function: ค้นหา Record จาก ID
def _find_record(record_id: str) -> Optional[AssessmentRecord]:
    """Retrieves a record from the global history list."""
    return next((r for r in ASSESSMENT_HISTORY if r.record_id == record_id), None)

@assessment_router.post("/")
async def run_assessment_task(request: AssessmentRequest, background_tasks: BackgroundTasks):
    if llm_instance is None:
        raise HTTPException(status_code=503, detail="LLM service is not available.")

    record_id: Final[str] = uuid.uuid4().hex
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
    target_record: Optional[AssessmentRecord] = _find_record(record_id)
    
    if not target_record:
        logger.error(f"Cannot find record ID {record_id} in history to run.")
        return

    try:
        mode_map: Dict[str, str] = {"full": "real", "partial": "random", "preview": "mock"}
        mode_to_use: str = mode_map.get(request.mode.lower(), "real")

        logger.info(f"Assessment {record_id} started. Mode: {mode_to_use}, Sub-Criteria: {request.sub_criteria_id or 'All'}")

        # 🟢 FIX: ใช้ Keyword Arguments ในการเรียก run_assessment เพื่อความชัดเจนและสอดคล้อง
        result: dict = await run_in_threadpool(
            run_assessment,
            enabler_id=request.enabler, # ⬅️ Argument 1
            sub_criteria_id=request.sub_criteria_id or "all", # ⬅️ Argument 2
            mode=mode_to_use, # ⬅️ Argument 3
            filter_mode=False, # ⬅️ Argument 4
            export=True, # ⬅️ Argument 5
            disable_semantic_filter=False, # ⬅️ Argument 6
            allow_fallback=True, # ⬅️ Argument 7
            external_retriever=None # ⬅️ Argument 8
        )

        # อัปเดตสถานะเมื่อเสร็จสมบูรณ์
        target_record.status = "COMPLETED"
        overall: dict = result.get("Overall", {})
        target_record.overall_score = overall.get("overall_maturity_score", 0.0)
        target_record.highest_full_level = overall.get("overall_maturity_level", 0)
        target_record.export_path = result.get("export_path_used")
        logger.info(f"✅ Assessment {record_id} COMPLETED. Score: {target_record.overall_score}")

    except Exception as e:
        logger.exception(f"❌ Assessment {record_id} failed: {e}")
        # อัปเดตสถานะเป็น FAILED หากเกิดข้อผิดพลาด
        target_record.status = "FAILED"
        target_record.overall_score = 0.0 

@assessment_router.get("/history", response_model=List[AssessmentRecord])
async def get_history(enabler: Optional[str] = Query(None)):
    data: List[AssessmentRecord] = ASSESSMENT_HISTORY
    if enabler:
        data = [r for r in data if r.enabler.upper() == enabler.upper()]
    return sorted(data, key=lambda r: r.timestamp, reverse=True)

@assessment_router.get("/results/{record_id}")
async def get_result(record_id: str = Path(..., description="Assessment Record ID")):
    record: Optional[AssessmentRecord] = _find_record(record_id)
    
    if not record:
        raise HTTPException(status_code=404, detail="Record not found.")
    
    if record.status != "COMPLETED":
        raise HTTPException(status_code=400, detail=f"Assessment {record_id} status is {record.status}. Wait until COMPLETED.")
    
    if record.export_path and os.path.exists(record.export_path):
        return FileResponse(record.export_path, media_type="application/json", filename=os.path.basename(record.export_path))
    
    # หากสถานะ COMPLETED แต่หาไฟล์ไม่เจอ
    raise HTTPException(status_code=404, detail=f"Result file for {record_id} not found at {record.export_path}.")