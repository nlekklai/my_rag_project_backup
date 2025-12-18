# -*- coding: utf-8 -*-
# routers/assessment_router.py

import os
import uuid
import json
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Union, List
from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

# --- 1. นำเข้าระบบ Auth และ Path Utils ---
from routers.auth_router import UserMe, get_current_user
from utils.path_utils import (
    get_assessment_export_file_path, 
    _n,
    get_tenant_year_export_root,
    load_doc_id_mapping  # ✅ ใช้ฟังก์ชันที่มีอยู่จริงใน path_utils
)

# --- 2. นำเข้า Engine และ Core Logic ---
from core.seam_assessment import SEAMPDCAEngine, AssessmentConfig
from core.vectorstore import load_all_vectorstores
from models.llm import create_llm_instance
from config.global_vars import EVIDENCE_DOC_TYPES, DEFAULT_LLM_MODEL_NAME

logger = logging.getLogger(__name__)
assessment_router = APIRouter(prefix="/api/assess", tags=["Assessment"])

# ในแรมสำหรับเก็บสถานะงานที่กำลังรัน (Running Tasks)
ACTIVE_TASKS: Dict[str, Any] = {}

# --- Schema สำหรับการรับค่า ---
class StartAssessmentRequest(BaseModel):
    tenant: str
    year: Union[int, str]
    enabler: str
    sub_criteria: Optional[str] = "all"
    sequential_mode: bool = True

# ===================================================================
# [Engine] Background Task (เรียกใช้ Engine จริง)
# ===================================================================
async def run_assessment_engine_task(record_id: str, tenant: str, year: int, enabler: str, sub_id: str, sequential: bool):
    try:
        # 1. Update Progress
        ACTIVE_TASKS[record_id]["progress_message"] = "กำลังโหลดฐานข้อมูลและ Mapping..."
        
        # โหลด VectorStore (ใช้ to_thread เพื่อไม่ให้ Block Main Thread)
        vsm = await asyncio.to_thread(
            load_all_vectorstores,
            doc_types=[EVIDENCE_DOC_TYPES],
            enabler_filter=enabler,
            tenant=tenant,
            year=year
        )
        
        # โหลด Document Map (โหลดเป็น Dict เพื่อส่งให้ Engine)
        doc_map_raw = await asyncio.to_thread(
            load_doc_id_mapping, 
            doc_type=EVIDENCE_DOC_TYPES, 
            tenant=tenant, 
            year=year, 
            enabler=enabler
        )
        
        # แปลง format ให้ Engine ใช้งานง่าย (ID -> FileName)
        doc_map = {
            doc_id: data.get("file_name", doc_id)
            for doc_id, data in doc_map_raw.items()
        }
        
        # สร้าง LLM Instance
        llm = await asyncio.to_thread(
            create_llm_instance, model_name=DEFAULT_LLM_MODEL_NAME, temperature=0.0
        )

        # 2. ตั้งค่า Engine
        config = AssessmentConfig(
            enabler=enabler,
            tenant=tenant,
            year=year,
            force_sequential=sequential
        )
        
        engine = SEAMPDCAEngine(
            config=config,
            llm_instance=llm,
            logger_instance=logger,
            doc_type=EVIDENCE_DOC_TYPES,
            vectorstore_manager=vsm,
            document_map=doc_map  # ✅ ส่ง Map ที่โหลดมาให้ Engine ตรงๆ
        )

        ACTIVE_TASKS[record_id]["progress_message"] = f"AI กำลังเริ่มวิเคราะห์หัวข้อ: {sub_id}..."

        # 3. รันการประเมิน (เรียก Method หลักของ Engine)
        final_result = await asyncio.to_thread(
            engine.run_assessment,
            target_sub_id=sub_id,
            export=True,
            vectorstore_manager=vsm,
            sequential=sequential
        )

        # 4. สรุปผลลัพธ์ลงใน RAM เพื่อให้ API Status ดึงไปแสดง
        summary = final_result.get("summary", {})
        result_payload = {
            "status": "COMPLETED",
            "progress_message": "การประเมินเสร็จสมบูรณ์",
            "level": f"L{summary.get('highest_pass_level_overall', 0)}",
            "score": round(summary.get('total_achieved_weight', 0.0), 2),
            "metrics": {
                "total_criteria": summary.get('total_subcriteria', 0),
                "completion_rate": round(summary.get('percentage_achieved_run', 0.0), 2)
            },
            "date": datetime.now(timezone.utc).isoformat(),
            "export_path": final_result.get("export_path_used")
        }

        if record_id in ACTIVE_TASKS:
            ACTIVE_TASKS[record_id].update(result_payload)
            
    except Exception as e:
        logger.exception(f"Engine Error for {record_id}: {e}")
        if record_id in ACTIVE_TASKS:
            ACTIVE_TASKS[record_id]["status"] = "FAILED"
            ACTIVE_TASKS[record_id]["progress_message"] = "เกิดข้อผิดพลาดในการประมวลผล"
            ACTIVE_TASKS[record_id]["error_message"] = str(e)

# ===================================================================
# [1] POST: Start Assessment
# ===================================================================
@assessment_router.post("/start")
async def start_assessment(
    request: StartAssessmentRequest, 
    background_tasks: BackgroundTasks, 
    current_user: UserMe = Depends(get_current_user)
):
    # 🛡️ ตรวจสอบสิทธิ์ (Normalization)
    if _n(request.tenant) != _n(current_user.tenant):
        raise HTTPException(status_code=403, detail="คุณไม่มีสิทธิ์ประเมินองค์กรอื่น")

    record_id = uuid.uuid4().hex[:12]
    
    # บันทึกสถานะเริ่มต้น
    ACTIVE_TASKS[record_id] = {
        "record_id": record_id,
        "status": "RUNNING",
        "date": datetime.now(timezone.utc).isoformat(),
        "tenant": request.tenant,
        "year": str(request.year),
        "enabler": request.enabler,
        "scope": request.sub_criteria,
        "progress_message": "กำลังเตรียมคิวงาน..."
    }
    
    # รัน Engine ใน Background
    background_tasks.add_task(
        run_assessment_engine_task, 
        record_id, request.tenant, int(request.year), 
        request.enabler, request.sub_criteria, request.sequential_mode
    )

    return {"record_id": record_id, "status": "RUNNING"}

# ===================================================================
# [2] GET: Status
# ===================================================================
@assessment_router.get("/status/{record_id}")
async def get_assessment_status(record_id: str, current_user: UserMe = Depends(get_current_user)):
    # เช็คใน RAM
    if record_id in ACTIVE_TASKS:
        task = ACTIVE_TASKS[record_id]
        if _n(task["tenant"]) == _n(current_user.tenant):
            return task

    # ถ้าไม่เจอใน RAM ลองเช็คไฟล์ Report เก่าใน Disk
    export_root = get_tenant_year_export_root(current_user.tenant, current_user.year)
    if os.path.exists(export_root):
        for root, _, files in os.walk(export_root):
            for filename in files:
                if record_id in filename and filename.endswith(".json"):
                    try:
                        with open(os.path.join(root, filename), "r", encoding="utf-8") as f:
                            data = json.load(f)
                            return {**data, "status": "COMPLETED", "record_id": record_id}
                    except: continue

    raise HTTPException(status_code=404, detail="ไม่พบข้อมูลการประเมิน")

# ===================================================================
# [3] GET: Download
# ===================================================================
@assessment_router.get("/download/{record_id}/json")
async def download_json(record_id: str, current_user: UserMe = Depends(get_current_user)):
    export_root = get_tenant_year_export_root(current_user.tenant, current_user.year)
    if os.path.exists(export_root):
        for root, _, files in os.walk(export_root):
            for f in files:
                if record_id in f and f.endswith(".json"):
                    return FileResponse(os.path.join(root, f), filename=f"report_{record_id}.json")
                
    raise HTTPException(status_code=404, detail="ไม่พบไฟล์ที่ต้องการ")