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
from core.seam_assessment import SEAMPDCAEngine, AssessmentConfig
from models.llm import create_llm_instance

from config.global_vars import LLM_MODEL_NAME

logger = logging.getLogger(__name__)

assessment_router = APIRouter(prefix="/api/assess", tags=["Assessment"])

# ------------------- Pydantic Models -------------------
class StartAssessmentRequest(BaseModel):
    enabler: str = Field(..., example="KM")
    sub_criteria_id: Optional[str] = Field(None, example="1.2")
    sequential: bool = Field(True, description="แนะนำเปิด")

class AssessmentStatus(BaseModel):
    record_id: str
    enabler: str
    sub_criteria_id: str
    sequential: bool
    status: str
    started_at: str
    finished_at: Optional[str] = None
    overall_score: Optional[float] = None
    highest_level: Optional[int] = None
    export_path: Optional[str] = None
    message: str = "Assessment in progress..."

# ------------------- In-memory Store -------------------
ASSESSMENT_RECORDS: Dict[str, AssessmentStatus] = {}

# ------------------- Background Runner -------------------
async def _run_assessment_background(record_id: str, request: StartAssessmentRequest):
    record = ASSESSMENT_RECORDS[record_id]
    try:
        logger.info(f"Assessment STARTED → {record_id} | {request.enabler} | Sequential: {request.sequential}")

        # สร้าง config
        config = AssessmentConfig(
            enabler=request.enabler.upper(),
            target_level=5,
            mock_mode="none",
            force_sequential=False,
            model_name=LLM_MODEL_NAME,
            temperature=0.0
        )

        # สร้าง engine
        engine = SEAMPDCAEngine(
            config=config,
            llm_instance=create_llm_instance(model_name=LLM_MODEL_NAME, temperature=0.0)
        )

        # รัน assessment
        
        # V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V
        # 🧪 HARDCODE TEST: ใช้ "1.2" เพื่อทดสอบ Engine โดยตรง (ต้องลบออกเมื่อทดสอบเสร็จ)
        target_id_to_use = "1.2"
        # ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^
        
        # โค้ดเดิมที่ถูกต้อง (ให้ลบ Hardcode ด้านบนและใช้โค้ดด้านล่างนี้เมื่อแก้ไขปัญหา Client เสร็จ)
        # target_id_to_use = (
        #     request.sub_criteria_id.strip() 
        #     if request.sub_criteria_id and request.sub_criteria_id.strip()
        #     else "all"
        # )

        result = engine.run_assessment(
            target_sub_id=target_id_to_use,
            export=True,
            sequential=request.sequential
        )

        # อัปเดต record
        export_path = result.get("export_path_used")
        if not export_path or not os.path.exists(export_path):
            raise Exception("Export file was not created")

        overall = result.get("Overall", {}) or {}
        record.status = "COMPLETED"
        record.finished_at = datetime.now(timezone.utc).isoformat()
        record.overall_score = overall.get("overall_maturity_score", 0.0)
        record.highest_level = overall.get("overall_maturity_level", 0)
        record.export_path = export_path
        record.message = "Assessment completed successfully"

        logger.info(f"Assessment COMPLETED → {record_id}")

    except Exception as e:
        logger.exception(f"Assessment FAILED → {record_id}")
        record.status = "FAILED"
        record.message = f"Error: {str(e)} ชิ้นส่วน (sub_criteria_id) ถูก Hardcode ไว้เพื่อทดสอบ"

# ------------------- API Endpoints -------------------
@assessment_router.post("/start", response_model=AssessmentStatus)
async def start_assessment(request: StartAssessmentRequest, background_tasks: BackgroundTasks):
    llm = create_llm_instance(model_name=LLM_MODEL_NAME, temperature=0.0)
    if not llm:
        raise HTTPException(status_code=503, detail="LLM service unavailable")

    record_id = uuid.uuid4().hex[:12]
    os.makedirs("exports", exist_ok=True)

    # 🟢 FIX: ตรวจสอบ String ว่าง/None ก่อนใช้ (สำหรับบันทึก Record)
    sub_id_for_record = (
        request.sub_criteria_id.strip() 
        if request.sub_criteria_id and request.sub_criteria_id.strip()
        else "all"
    )

    record = AssessmentStatus(
        record_id=record_id,
        enabler=request.enabler.upper(),
        sub_criteria_id=sub_id_for_record,
        sequential=request.sequential,
        status="RUNNING",
        started_at=datetime.now(timezone.utc).isoformat(),
        message="กำลังวิเคราะห์เอกสารกว่า 300 ฉบับด้วย AI..."
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
async def get_results_json(record_id: str = Path(...)):
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
        raise HTTPException(status_code=404, detail="Result not ready")
    return FileResponse(
        path=record.export_path,
        media_type="application/json",
        filename=os.path.basename(record.export_path)
    )

@assessment_router.get("/history")
async def get_assessment_history(enabler: Optional[str] = None):
    items = list(ASSESSMENT_RECORDS.values())
    if enabler:
        items = [i for i in items if i.enabler == enabler.upper()]
    return sorted(items, key=lambda x: x.started_at, reverse=True)