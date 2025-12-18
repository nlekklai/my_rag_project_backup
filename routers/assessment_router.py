import os
import uuid
import json
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any, Union
from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# นำเข้าระบบ Auth และ Path Utils
from routers.auth_router import UserMe, get_current_user
from utils.path_utils import (
    get_assessment_export_file_path, 
    get_export_dir,
    _n
)

logger = logging.getLogger(__name__)
assessment_router = APIRouter(prefix="/api/assess", tags=["Assessment"])

# --- Schema สำหรับการรับค่า ---
class StartAssessmentRequest(BaseModel):
    tenant: str
    year: Union[int, str]
    enabler: str
    sub_criteria: Optional[str] = ""
    sequential_mode: bool = True

# --- ในแรมสำหรับเก็บสถานะงานที่กำลังรัน (Running Tasks) ---
ACTIVE_TASKS: Dict[str, Any] = {}

# --- [1] POST: Start Assessment (พร้อมระบบเช็คสิทธิ์) ---
@assessment_router.post("/start")
async def start_assessment(
    request: StartAssessmentRequest, 
    background_tasks: BackgroundTasks, 
    current_user: UserMe = Depends(get_current_user)
):
    # 🛡️ USER CHECK 1: ตรวจสอบว่า Tenant ใน Request ตรงกับ User ที่ Login หรือไม่
    if _n(request.tenant) != _n(current_user.tenant):
        logger.warning(f"Unauthorized access attempt: User {current_user.username} tried to access tenant {request.tenant}")
        raise HTTPException(status_code=403, detail="คุณไม่มีสิทธิ์เริ่มการประเมินให้กับองค์กรอื่น")

    # 🛡️ USER CHECK 2: ตรวจสอบว่า User มีสิทธิ์ใน Enabler นั้นๆ หรือไม่ (ถ้ามีการจำกัดสิทธิ์)
    if current_user.enablers and request.enabler not in current_user.enablers:
        raise HTTPException(status_code=403, detail=f"คุณไม่มีสิทธิ์เข้าถึง Enabler: {request.enabler}")

    record_id = uuid.uuid4().hex[:12]
    
    # บันทึกข้อมูลเริ่มต้นลงใน RAM
    task_info = {
        "record_id": record_id,
        "status": "RUNNING",
        "date": datetime.now(timezone.utc).isoformat(),
        "tenant": request.tenant,
        "year": str(request.year),
        "enabler": request.enabler,
        "scope": request.sub_criteria or "ทุกข้อ",
        "progress_message": "ระบบกำลังเริ่มประมวลผล..."
    }
    
    ACTIVE_TASKS[record_id] = task_info
    
    # ส่งงานไปทำ Background Task
    background_tasks.add_task(
        run_assessment_engine, 
        record_id, request.tenant, str(request.year), request.enabler
    )

    return {"record_id": record_id, "status": "RUNNING"}

# --- [2] GET: Get Status/Result (พร้อมระบบเช็คสิทธิ์) ---
@assessment_router.get("/status/{record_id}")
async def get_assessment_status(record_id: str, current_user: UserMe = Depends(get_current_user)):
    # 1. เช็คใน RAM (งานที่กำลังรัน)
    if record_id in ACTIVE_TASKS:
        task = ACTIVE_TASKS[record_id]
        # 🛡️ USER CHECK: ต้องเป็น Tenant เดียวกันเท่านั้น
        if _n(task["tenant"]) == _n(current_user.tenant):
            return task

    # 2. เช็คใน Disk (งานที่เสร็จแล้ว)
    export_root = get_export_dir(current_user.tenant, current_user.year)
    if os.path.exists(export_root):
        for root, _, files in os.walk(export_root):
            for filename in files:
                if record_id in filename and filename.endswith(".json"):
                    with open(os.path.join(root, filename), "r", encoding="utf-8") as f:
                        data = json.load(f)
                        return {**data, "status": "COMPLETED", "record_id": record_id}

    raise HTTPException(status_code=404, detail="ไม่พบข้อมูลการประเมิน หรือคุณไม่มีสิทธิ์เข้าถึง")

# --- [3] GET: History (พร้อมระบบ Isolation) ---
@assessment_router.get("/history")
async def get_history(
    tenant: str = Query(...),
    year: str = Query(...),
    enabler: Optional[str] = Query(None),
    current_user: UserMe = Depends(get_current_user)
):
    # 🛡️ USER CHECK: บังคับให้ดูได้เฉพาะ Tenant ตัวเอง
    if _n(tenant) != _n(current_user.tenant):
        raise HTTPException(status_code=403, detail="สิทธิ์ไม่ถูกต้อง")

    history_list = []
    
    # 1. ดึงข้อมูลจาก Disk (ไฟล์ที่เซฟไว้)
    export_path = get_export_dir(tenant, year)
    if os.path.exists(export_path):
        for root, _, files in os.walk(export_path):
            current_enabler = os.path.basename(root)
            # กรองตาม enabler (ถ้ามี)
            if enabler and enabler != 'all' and _n(current_enabler) != _n(enabler):
                continue
                
            for filename in files:
                if filename.endswith(".json"):
                    try:
                        with open(os.path.join(root, filename), "r", encoding="utf-8") as f:
                            data = json.load(f)
                            history_list.append({
                                "record_id": filename.replace("report_", "").replace(".json", ""),
                                "status": "COMPLETED",
                                "tenant": tenant,
                                "year": year,
                                "enabler": current_enabler.upper(),
                                **data
                            })
                    except: pass

    # 2. ผสมกับงานใน RAM ที่ยังไม่เสร็จ
    for rid, task in ACTIVE_TASKS.items():
        if _n(task["tenant"]) == _n(tenant) and task["year"] == str(year):
            if enabler and enabler != 'all' and _n(task["enabler"]) != _n(enabler):
                continue
            history_list.append(task)

    # เรียงลำดับวันที่
    history_list.sort(key=lambda x: x.get('date', ''), reverse=True)
    return {"items": history_list}

# --- [4] GET: Download (พร้อมระบบเช็คสิทธิ์) ---
@assessment_router.get("/download/{record_id}/json")
async def download_json(record_id: str, current_user: UserMe = Depends(get_current_user)):
    # 🛡️ สแกนหาไฟล์เฉพาะใน Folder ของ Tenant ตัวเองเท่านั้น
    export_root = get_export_dir(current_user.tenant, current_user.year)
    for root, _, files in os.walk(export_root):
        for f in files:
            if record_id in f:
                return FileResponse(os.path.join(root, f), filename=f"report_{record_id}.json")
                
    raise HTTPException(status_code=404, detail="ไม่พบไฟล์")

# --- [Engine] ฟังก์ชันวิเคราะห์ (Background Task) ---
async def run_assessment_engine(record_id, tenant, year, enabler):
    try:
        # จำลองสถานะการทำงาน
        messages = ["กำลังตรวจสอบไฟล์...", "AI กำลังวิเคราะห์เกณฑ์...", "สรุปคะแนน..."]
        for msg in messages:
            if record_id in ACTIVE_TASKS:
                ACTIVE_TASKS[record_id]["progress_message"] = msg
                await asyncio.sleep(3)

        # Mock ผลลัพธ์ (เปลี่ยนเป็น AI Engine จริงที่นี่)
        result_data = {
            "level": "L3",
            "score": 3.65,
            "metrics": {"total_criteria": 10, "passed_criteria": 7, "completion_rate": 70.0},
            "radar_data": [
                {"axis": "Plan", "value": 4.0}, {"axis": "Do", "value": 3.5}, 
                {"axis": "Check", "value": 3.0}, {"axis": "Act", "value": 3.8}
            ],
            "strengths": ["ระบบจัดเก็บไฟล์ทันสมัย"],
            "weaknesses": ["ขาดการสื่อสารภายใน"]
        }

        # บันทึกลง Disk
        file_path = get_assessment_export_file_path(tenant, year, enabler, f"report_{record_id}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        save_obj = {**result_data, "date": datetime.now(timezone.utc).isoformat()}
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(save_obj, f, ensure_ascii=False, indent=4)

        # อัปเดตสถานะใน RAM ให้เป็นสำเร็จ
        if record_id in ACTIVE_TASKS:
            ACTIVE_TASKS[record_id].update(save_obj)
            ACTIVE_TASKS[record_id]["status"] = "COMPLETED"
            
    except Exception as e:
        if record_id in ACTIVE_TASKS:
            ACTIVE_TASKS[record_id]["status"] = "FAILED"
            ACTIVE_TASKS[record_id]["error_message"] = str(e)