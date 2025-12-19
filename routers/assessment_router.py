# routers/assessment_router.py
# Production Final Version - 19 ธันวาคม 2568
# แสดง Level 1-5 ครบในช่อง "หลักฐาน" + Action Plan ละเอียดในช่อง "ช่องว่าง"
# Fit กับ UI ปัจจุบัน 100% (ไม่ต้องแก้ Frontend)

import os
import uuid
import json
import asyncio
import logging
import unicodedata
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Union, List

from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# --- Core Imports ---
from routers.auth_router import UserMe, get_current_user
from utils.path_utils import (
    _n,
    get_tenant_year_export_root,
    load_doc_id_mapping
)
from core.seam_assessment import SEAMPDCAEngine, AssessmentConfig
from core.vectorstore import load_all_vectorstores
from models.llm import create_llm_instance
from config.global_vars import EVIDENCE_DOC_TYPES, DEFAULT_LLM_MODEL_NAME

logger = logging.getLogger(__name__)
assessment_router = APIRouter(prefix="/api/assess", tags=["Assessment"])

ACTIVE_TASKS: Dict[str, Any] = {}

class StartAssessmentRequest(BaseModel):
    tenant: str
    year: Union[int, str]
    enabler: str
    sub_criteria: Optional[str] = "all"
    sequential_mode: bool = True

# ------------------- Helpers -------------------
def parse_safe_date(raw_date_str: Any, file_path: str) -> str:
    if raw_date_str and isinstance(raw_date_str, str):
        try:
            if "_" in raw_date_str:
                dt = datetime.strptime(raw_date_str, "%Y%m%d_%H%M%S")
                return dt.isoformat()
        except: pass
    try:
        return datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
    except:
        return datetime.now().isoformat()

def clean_suggestion(raw_val: Any) -> str:
    if not raw_val:
        return "ไม่มีข้อเสนอแนะเพิ่มเติม"
    if isinstance(raw_val, dict):
        return raw_val.get('description', str(raw_val))
    if isinstance(raw_val, str):
        raw_val = raw_val.strip()
        if raw_val.startswith('{'):
            try:
                data = json.loads(raw_val.replace("'", '"'))
                return data.get('description', raw_val)
            except: pass
    return raw_val

def _find_assessment_file(search_id: str, current_user: UserMe) -> str:
    export_root = get_tenant_year_export_root(current_user.tenant, current_user.year)
    norm_search = _n(search_id).lower()

    for root, _, files in os.walk(export_root):
        for f in files:
            if f.endswith(".json") and norm_search in _n(f).lower():
                return os.path.join(root, f)
    raise HTTPException(status_code=404, detail="ไม่พบไฟล์ผลการประเมิน")

def _transform_result_for_ui(raw_data: Dict[str, Any], current_user: UserMe) -> Dict[str, Any]:
    summary = raw_data.get("summary", {})
    sub_results = raw_data.get("sub_criteria_results", [])
    processed_sub_criteria = []
    radar_data = []
    strengths = []
    weaknesses = []

    overall_highest_lv = summary.get("highest_pass_level_overall", summary.get("highest_pass_level", 0))

    for res in sub_results:
        cid = res.get("sub_criteria_id", "N/A")
        cname = res.get("sub_criteria_name", f"เกณฑ์ย่อย {cid}")
        highest_full = res.get("highest_full_level", res.get("highest_pass_level", 0))

        # --- evidence: แสดง Level 1-5 แยกหัวข้อชัดเจน + คำแนะนำพัฒนา ---
        evidence_lines = []
        raw_levels = res.get("raw_results_ref", [])

        for lv_num in range(1, 6):
            lv_info = next((item for item in raw_levels if item.get("level") == lv_num), None)

            evidence_lines.append(f"**Level {lv_num}**")

            if lv_info:
                if lv_info.get("evaluation_mode") == "GAP_ONLY":
                    evidence_lines.append("⚠️ **Gap**")
                    evidence_lines.append(lv_info.get("cap_reason", "เป็น Gap จาก Level ก่อนหน้าไม่ผ่าน"))
                else:
                    status = "✅ **ผ่าน**" if lv_info.get("is_passed") else "❌ **ไม่ผ่าน**"
                    evidence_lines.append(status)
                    evidence_lines.append(lv_info.get("reason", ""))
                # คำแนะนำพัฒนาแม้ผ่านแล้ว
                if lv_info.get("is_passed"):
                    evidence_lines.append("")
                    evidence_lines.append("*คำแนะนำพัฒนาเพิ่มเติม*: แม้ผ่านแล้ว ควรเสริมหลักฐาน PDCA ให้แข็งแรงขึ้น เช่น เพิ่มการวัดผล (Check) หรือปรับปรุงต่อเนื่อง (Act)")
            else:
                if lv_num <= highest_full:
                    evidence_lines.append("✅ **ผ่าน** (จาก Sequential)")
                    evidence_lines.append("")
                    evidence_lines.append("*คำแนะนำพัฒนาเพิ่มเติม*: เสริมหลักฐานให้ชัดเจนเพื่อความมั่นคงในรอบถัดไป")
                else:
                    evidence_lines.append("⚠️ **Gap**")
                    evidence_lines.append(f"ยังไม่ประเมินเพราะ Level {highest_full} เป็นจุดอ่อน")

            evidence_lines.append("")  # เว้นบรรทัดระหว่าง Level

        full_evidence_text = "\n".join(evidence_lines)

        # --- gap: Action Plan ละเอียด ---
        action_plans = res.get("action_plan", [])
        gap_lines = ["📍 **แผนการพัฒนา (Action Plan)**"]
        if action_plans:
            for phase in action_plans:
                phase_name = phase.get("Phase", "Phase ไม่ระบุ")
                goal = phase.get("Goal", "")
                gap_lines.append(f"• **{phase_name}**: {goal}")
                actions = phase.get("Actions", [])
                for action in actions:
                    rec = action.get("Recommendation", "")
                    if rec:
                        gap_lines.append(f"  → {rec}")
                    steps = action.get("Steps", [])
                    for step in steps:
                        step_num = step.get("Step", "?")
                        desc = step.get("Description", "")
                        responsible = step.get("Responsible", "")
                        tools = step.get("Tools_Templates", "")
                        outcome = step.get("Verification_Outcome", "")
                        gap_lines.append(f"    * Step {step_num}: {desc}")
                        if responsible:
                            gap_lines.append(f"      ผู้รับผิดชอบ: {responsible}")
                        if tools:
                            gap_lines.append(f"      เครื่องมือ: {tools}")
                        if outcome:
                            gap_lines.append(f"      ผลลัพธ์ที่คาดหวัง: {outcome}")
                gap_lines.append("")  # เว้นบรรทัดระหว่าง Phase
        else:
            suggestion = clean_suggestion(res.get("suggestion_next_level"))
            gap_lines.append(f"📍 **ข้อเสนอแนะเพื่อไป Level ถัดไป**:\n{suggestion}")

        gap_text = "\n".join(gap_lines)

        # Strengths / Weaknesses
        if highest_full >= 1:
            strengths.append(f"[{cid}] บรรลุระดับ L{highest_full}")

        weaknesses.append(f"[{cid}] {gap_text}")

        processed_sub_criteria.append({
            "code": cid,
            "name": cname,
            "level": f"L{highest_full}",
            "score": res.get("weighted_score", float(highest_full)),
            "evidence": full_evidence_text,
            "gap": gap_text
        })

        radar_data.append({"axis": cid, "value": highest_full, "fullMark": 5})

    return {
        "status": "COMPLETED",
        "record_id": raw_data.get("record_id", "unknown"),
        "tenant": summary.get("tenant", current_user.tenant),
        "year": str(summary.get("year", current_user.year)),
        "enabler": (summary.get("enabler") or "KM").upper(),
        "level": f"L{overall_highest_lv}",
        "score": round(float(summary.get("Total Weighted Score Achieved", 0.0)), 2),
        "metrics": {
            "total_criteria": summary.get("total_subcriteria", 12),
            "passed_criteria": len(sub_results),
            "completion_rate": summary.get("percentage_achieved_run", 0.0)
        },
        "radar_data": radar_data,
        "strengths": strengths if strengths else ["วิเคราะห์สำเร็จ"],
        "weaknesses": weaknesses,
        "sub_criteria": processed_sub_criteria,
        "progress_message": "ประมวลผลสำเร็จ"
    }

# ------------------- API Endpoints -------------------
@assessment_router.get("/status/{record_id}")
async def get_assessment_status(record_id: str, current_user: UserMe = Depends(get_current_user)):
    if record_id in ACTIVE_TASKS:
        return ACTIVE_TASKS[record_id]

    file_path = _find_assessment_file(record_id, current_user)
    with open(file_path, "r", encoding="utf-8") as jf:
        raw_data = json.load(jf)
    return _transform_result_for_ui(raw_data, current_user)

@assessment_router.get("/history")
async def get_assessment_history(tenant: str, year: Union[int, str], current_user: UserMe = Depends(get_current_user)):
    if _n(tenant) != _n(current_user.tenant):
        raise HTTPException(status_code=403, detail="Permission Denied")

    export_root = get_tenant_year_export_root(tenant, str(year))
    history_list = []
    if not os.path.exists(export_root):
        return {"items": []}

    for root, _, files in os.walk(export_root):
        for f in files:
            if f.lower().endswith(".json") and "results" in f.lower():
                try:
                    file_path = os.path.join(root, f)
                    with open(file_path, "r", encoding="utf-8") as jf:
                        data = json.load(jf)
                        summary = data.get("summary", {})
                        history_list.append({
                            "record_id": f.rsplit('.', 1)[0],
                            "date": parse_safe_date(summary.get("export_timestamp"), file_path),
                            "tenant": tenant,
                            "year": str(year),
                            "enabler": (summary.get("enabler") or "KM").upper(),
                            "scope": summary.get("sub_criteria_id", "ALL"),
                            "level": f"L{summary.get('highest_pass_level', 0)}",
                            "score": round(float(summary.get("Total Weighted Score Achieved", 0.0)), 2),
                            "status": "COMPLETED"
                        })
                except Exception as e:
                    logger.error(f"Error reading history file {f}: {e}")

    return {"items": sorted(history_list, key=lambda x: x['date'], reverse=True)}

@assessment_router.post("/start")
async def start_assessment(request: StartAssessmentRequest, background_tasks: BackgroundTasks, current_user: UserMe = Depends(get_current_user)):
    if _n(request.tenant) != _n(current_user.tenant):
        raise HTTPException(status_code=403, detail="Permission Denied")

    record_id = uuid.uuid4().hex[:12]
    ACTIVE_TASKS[record_id] = {
        "status": "RUNNING",
        "record_id": record_id,
        "date": datetime.now().isoformat(),
        "tenant": request.tenant,
        "year": str(request.year),
        "enabler": request.enabler,
        "progress_message": f"กำลังวิเคราะห์ {request.sub_criteria or 'ทุกข้อ'}..."
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
            del ACTIVE_TASKS[record_id]
    except Exception as e:
        logger.error(f"Engine Failed: {e}")
        if record_id in ACTIVE_TASKS:
            ACTIVE_TASKS[record_id]["status"] = "FAILED"
            ACTIVE_TASKS[record_id]["error_message"] = str(e)

@assessment_router.get("/download/{record_id}/{file_type}")
async def download_assessment_file(record_id: str, file_type: str, current_user: UserMe = Depends(get_current_user)):
    file_path = _find_assessment_file(record_id, current_user)
    if not file_path.endswith(f".{file_type.lower()}"):
        raise HTTPException(status_code=404, detail="ประเภทไฟล์ไม่ถูกต้อง")
    return FileResponse(path=file_path, filename=os.path.basename(file_path))