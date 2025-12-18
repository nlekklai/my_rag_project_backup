# -*- coding: utf-8 -*-
# routers/assessment_router.py
# Final Production Version - 18 ธ.ค. 2568 (Data Mapping Match with JSON Structure)

import os
import uuid
import json
import asyncio
import logging
import unicodedata
from datetime import datetime
from typing import Optional, Dict, Any, Union, List
from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel

# --- 1. Core Imports ---
from routers.auth_router import UserMe, get_current_user
from utils.path_utils import (
    _n,
    get_tenant_year_export_root,
    get_export_dir,
    load_doc_id_mapping
)
from core.seam_assessment import SEAMPDCAEngine, AssessmentConfig
from core.vectorstore import load_all_vectorstores
from models.llm import create_llm_instance
from config.global_vars import EVIDENCE_DOC_TYPES, DEFAULT_LLM_MODEL_NAME

logger = logging.getLogger(__name__)
assessment_router = APIRouter(prefix="/api/assess", tags=["Assessment"])

# เก็บสถานะงานที่กำลังรันอยู่ใน RAM
ACTIVE_TASKS: Dict[str, Any] = {}

class StartAssessmentRequest(BaseModel):
    tenant: str
    year: Union[int, str]
    enabler: str
    sub_criteria: Optional[str] = "all"
    sequential_mode: bool = True

# ===================================================================
# [Helpers]
# ===================================================================

def parse_safe_date(raw_date_str: Any, file_path: str) -> str:
    """แปลงวันที่จากชื่อไฟล์หรือ Metadata ให้เป็น ISO สำหรับ Frontend"""
    if raw_date_str and isinstance(raw_date_str, str):
        try:
            # กรณี format 20251218_161457
            if "_" in raw_date_str and len(raw_date_str) == 15:
                dt = datetime.strptime(raw_date_str, "%Y%m%d_%H%M%S")
                return dt.isoformat()
        except: pass
    
    try:
        return datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
    except:
        return datetime.now().isoformat()

def extract_record_id(filename: str) -> str:
    f_norm = unicodedata.normalize('NFKC', filename)
    return f_norm.rsplit('.', 1)[0]

# ===================================================================
# API Endpoints (Reading & Mapping Data)
# ===================================================================

@assessment_router.get("/status/{record_id}")
async def get_assessment_status(record_id: str, current_user: UserMe = Depends(get_current_user)):
    """อ่านไฟล์ JSON และ Map ค่าเข้าสู่ Interface ของ Frontend (React)"""
    
    # 1. เช็คใน Memory Task
    if record_id in ACTIVE_TASKS:
        return ACTIVE_TASKS[record_id]
    
    # 2. ค้นหาไฟล์บน Disk
    export_root = get_tenant_year_export_root(current_user.tenant, current_user.year)
    search_id = unicodedata.normalize('NFKC', record_id).lower()
    
    for root, _, files in os.walk(export_root):
        for f in files:
            f_norm = unicodedata.normalize('NFKC', f).lower()
            if search_id in f_norm and f_norm.endswith(".json"):
                try:
                    file_path = os.path.join(root, f)
                    with open(file_path, "r", encoding="utf-8") as jf:
                        raw_data = json.load(jf)
                        summary = raw_data.get("summary", {})
                        sub_results = raw_data.get("sub_criteria_results", [])

                        # --- Mapping Data ตาม JSON จริงของคุณ ---
                        level = summary.get("Overall Maturity Level (Weighted)", "L0")
                        score = summary.get("Total Weighted Score Achieved", 0.0)
                        
                        # คำนวณ Metrics
                        total_criteria = summary.get("total_subcriteria", 12)
                        passed_criteria = summary.get("total_subcriteria_assessed", 1)
                        # แปลง 0.08 เป็น 8.0
                        completion_rate = summary.get("Overall Progress Percentage (0.0 - 1.0)", 0.0) * 100

                        # ดึง Strengths/Weaknesses จาก sub_results
                        strengths_list = []
                        weaknesses_list = []
                        
                        for res in sub_results:
                            if res.get("summary_thai"):
                                strengths_list.append(f"{res['sub_criteria_id']}: {res['summary_thai']}")
                            
                            # ดึงจุดที่ควรพัฒนาจาก suggestion_next_level
                            sugg_raw = res.get("suggestion_next_level", "")
                            if sugg_raw:
                                try:
                                    # พยายามหาข้อความข้างใน (เผื่อเป็น string dict)
                                    import ast
                                    sugg_dict = ast.literal_eval(sugg_raw)
                                    weaknesses_list.append(sugg_dict.get("description", str(sugg_dict)))
                                except:
                                    weaknesses_list.append(str(sugg_raw))

                        # ข้อมูลสำหรับ Radar Chart (สร้างตาม sub_criteria ที่ประเมิน)
                        radar_data = []
                        for res in sub_results:
                            radar_data.append({
                                "axis": res.get("sub_criteria_id", "Unknown"),
                                "value": res.get("highest_pass_level", 0),
                                "fullMark": 5
                            })

                        return {
                            "status": "COMPLETED",
                            "record_id": record_id,
                            "tenant": summary.get("tenant", current_user.tenant),
                            "year": str(summary.get("year", current_user.year)),
                            "enabler": summary.get("enabler", "KM"),
                            "level": level,
                            "score": round(float(score), 2),
                            "metrics": {
                                "total_criteria": total_criteria,
                                "passed_criteria": passed_criteria,
                                "completion_rate": round(completion_rate, 2)
                            },
                            "radar_data": radar_data,
                            "strengths": strengths_list if strengths_list else ["ประเมินสำเร็จ"],
                            "weaknesses": weaknesses_list,
                            "sub_criteria": [
                                {
                                    "code": r.get("sub_criteria_id"),
                                    "name": r.get("sub_criteria_name", "เกณฑ์ย่อย"),
                                    "level": f"L{r.get('highest_pass_level', 0)}",
                                    "score": r.get("achieved_weight", 0.0),
                                    "evidence": r.get("summary_thai", ""),
                                    "gap": r.get("gap_to_full_score", 0.0)
                                } for r in sub_results
                            ],
                            "progress_message": "โหลดข้อมูลจากไฟล์สำเร็จ"
                        }
                except Exception as e:
                    logger.error(f"Error reading JSON {f}: {e}")
                    continue

    raise HTTPException(status_code=404, detail="ไม่พบข้อมูลผลการประเมิน")

@assessment_router.get("/history")
async def get_assessment_history(tenant: str, year: Union[int, str], current_user: UserMe = Depends(get_current_user)):
    if _n(tenant) != _n(current_user.tenant):
        raise HTTPException(status_code=403, detail="Permission Denied")
    
    export_root = get_tenant_year_export_root(tenant, str(year))
    history_list = []
    if not os.path.exists(export_root): return {"items": []}

    for root, _, files in os.walk(export_root):
        for f in files:
            f_norm = unicodedata.normalize('NFKC', f)
            if f_norm.lower().endswith(".json") and "results" in f_norm.lower():
                try:
                    file_path = os.path.join(root, f)
                    with open(file_path, "r", encoding="utf-8") as jf:
                        data = json.load(jf)
                        summary = data.get("summary", {})
                        rec_id = extract_record_id(f_norm)
                        
                        history_list.append({
                            "record_id": rec_id,
                            "date": parse_safe_date(summary.get("export_timestamp"), file_path), 
                            "tenant": tenant, "year": str(year),
                            "enabler": (summary.get("enabler") or "KM").upper(),
                            "scope": summary.get("sub_criteria_id") or "ALL",
                            "level": summary.get("Overall Maturity Level (Weighted)", "L0"),
                            "score": round(float(summary.get("Total Weighted Score Achieved", 0.0)), 2),
                            "status": "COMPLETED"
                        })
                except: continue

    history_list.sort(key=lambda x: str(x.get("date") or ""), reverse=True)
    return {"items": history_list}

@assessment_router.get("/download/{record_id}/{file_type}")
async def download_assessment_file(record_id: str, file_type: str, current_user: UserMe = Depends(get_current_user)):
    ext = f".{file_type.lower()}"
    search_id = unicodedata.normalize('NFKC', record_id).lower()
    export_root = get_tenant_year_export_root(current_user.tenant, current_user.year)
    
    for root, _, files in os.walk(export_root):
        for f in files:
            f_norm = unicodedata.normalize('NFKC', f).lower()
            if search_id in f_norm and f_norm.endswith(ext):
                return FileResponse(path=os.path.join(root, f), filename=f)

    raise HTTPException(status_code=404, detail="File not found")

@assessment_router.post("/start")
async def start_assessment(request: StartAssessmentRequest, background_tasks: BackgroundTasks, current_user: UserMe = Depends(get_current_user)):
    if _n(request.tenant) != _n(current_user.tenant):
        raise HTTPException(status_code=403, detail="Permission Denied")
    
    record_id = uuid.uuid4().hex[:12]
    ACTIVE_TASKS[record_id] = {
        "status": "RUNNING", "record_id": record_id, "date": datetime.now().isoformat(),
        "tenant": request.tenant, "year": str(request.year), "enabler": request.enabler,
        "progress_message": "เริ่มการประเมิน..."
    }
    background_tasks.add_task(run_assessment_engine_task, record_id, request.tenant, int(request.year), request.enabler, request.sub_criteria, request.sequential_mode)
    return {"record_id": record_id, "status": "RUNNING"}

async def run_assessment_engine_task(record_id: str, tenant: str, year: int, enabler: str, sub_id: str, sequential: bool):
    try:
        vsm = await asyncio.to_thread(load_all_vectorstores, [EVIDENCE_DOC_TYPES], enabler, tenant, year)
        doc_map_raw = await asyncio.to_thread(load_doc_id_mapping, EVIDENCE_DOC_TYPES, tenant, year, enabler)
        doc_map = {d_id: d.get("file_name", d_id) for d_id, d in doc_map_raw.items()}
        llm = await asyncio.to_thread(create_llm_instance, model_name=DEFAULT_LLM_MODEL_NAME, temperature=0.0)
        config = AssessmentConfig(enabler=enabler, tenant=tenant, year=year, force_sequential=sequential)
        engine = SEAMPDCAEngine(config, llm, logger, EVIDENCE_DOC_TYPES, vsm, doc_map)
        await asyncio.to_thread(engine.run_assessment, sub_id, True, vsm, sequential, record_id)
        if record_id in ACTIVE_TASKS:
            # หลังจากเสร็จ ปล่อยให้ get_assessment_status อ่านจากไฟล์จริงแทน
            del ACTIVE_TASKS[record_id]
    except Exception as e:
        if record_id in ACTIVE_TASKS:
            ACTIVE_TASKS[record_id]["status"] = "FAILED"
            ACTIVE_TASKS[record_id]["error_message"] = str(e)