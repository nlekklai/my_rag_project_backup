# -*- coding: utf-8 -*-
# routers/assessment_router.py
# Production Final Version - 20 ธันวาคม 2568 (Fixed parameter order + stable UUID + full assessment flow)

import os
import uuid
import json
import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any, Union, List

from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel

from routers.auth_router import UserMe, get_current_user
from utils.path_utils import _n, get_tenant_year_export_root, load_doc_id_mapping
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

# ------------------- Permission Helper -------------------
def check_user_permission(user: UserMe, tenant: str, enabler: str):
    if _n(user.tenant) != _n(tenant):
        raise HTTPException(status_code=403, detail="Tenant mismatch")
    if user.enablers and enabler.upper() not in [e.upper() for e in user.enablers]:
        raise HTTPException(status_code=403, detail=f"Enabler '{enabler}' not allowed")

# ------------------- Helpers -------------------
def parse_safe_date(raw_date_str: Any, file_path: str) -> str:
    if raw_date_str and isinstance(raw_date_str, str):
        try:
            if "_" in raw_date_str:
                dt = datetime.strptime(raw_date_str, "%Y%m%d_%H%M%S")
                return dt.isoformat()
        except:
            pass
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
            except:
                pass
    return raw_val

def _find_assessment_file(search_id: str, current_user: UserMe) -> str:
    export_root = get_tenant_year_export_root(current_user.tenant, current_user.year)
    norm_search = _n(search_id).lower()

    for root, _, files in os.walk(export_root):
        for f in files:
            if f.endswith(".json") and norm_search in _n(f).lower():
                return os.path.join(root, f)
    raise HTTPException(status_code=404, detail="ไม่พบไฟล์ผลการประเมิน")

# ------------------- Transformation Logic (For UI) -------------------
def _transform_result_for_ui(raw_data: Dict[str, Any], current_user: UserMe) -> Dict[str, Any]:
    summary = raw_data.get("summary", {})
    sub_results = raw_data.get("sub_criteria_results", [])

    processed_sub_criteria = []
    radar_data = []
    strengths = []
    weaknesses = []

    # ดึงค่าสถิติจาก summary
    total_sub_in_file = len(sub_results)
    passed_count = sum(1 for r in sub_results if int(r.get("highest_full_level", 0)) >= 1)
    total_expected = int(summary.get("total_subcriteria", 12))
    
    # คำนวณ % ความสำเร็จ
    completion_rate = float(summary.get("percentage_achieved_run", 
                                        (passed_count / total_expected * 100) if total_expected > 0 else 0))

    # ดึงระดับ Maturity โดยรวม
    overall_level = summary.get("highest_pass_level_overall") or summary.get("highest_pass_level", 0)
    try:
        overall_level = int(overall_level)
    except:
        overall_level = 0

    for res in sub_results:
        cid = res.get("sub_criteria_id", "N/A")
        cname = res.get("sub_criteria_name", f"เกณฑ์ย่อย {cid}")
        highest_pass = int(res.get("highest_full_level", 0))

        # --- 1. Evidence Transformation (ระดับ 1-5) ---
        evidence_lines = []
        raw_levels = {item.get("level"): item for item in res.get("raw_results_ref", [])}

        for lv in range(1, 6):
            lv_info = raw_levels.get(lv)
            evidence_lines.append(f"### 💠 Level {lv}")

            if lv_info:
                is_passed = lv_info.get("is_passed", False)
                status = "✅ **ผ่าน**" if is_passed else "❌ **ไม่ผ่าน**"
                reason = lv_info.get("summary_thai") or lv_info.get("reason") or "วิเคราะห์ตามเกณฑ์มาตรฐาน"
                
                pdca = lv_info.get("pdca_breakdown", {})
                pdca_str = f"P:{pdca.get('P',0)} D:{pdca.get('D',0)} C:{pdca.get('C',0)} A:{pdca.get('A',0)}"
                
                evidence_lines.append(f"{status} | **PDCA**: `{pdca_str}`\n> {reason}")
                
                if is_passed:
                    evidence_lines.append("\n*คำแนะนำ*: รักษามาตรฐานและเตรียมหลักฐานสำหรับระดับถัดไป")
            else:
                if lv <= highest_pass:
                    evidence_lines.append("✅ **ผ่าน** (โดยระบบ Sequential)")
                else:
                    evidence_lines.append("⚠️ **ยังไม่ถึงเกณฑ์**")
                    evidence_lines.append("เกณฑ์นี้จะถูกประเมินเมื่อระดับก่อนหน้าผ่านเกณฑ์ครบถ้วน")

            evidence_lines.append("\n---\n")

        # --- 2. Action Plan Transformation (Gap Analysis) ---
        gap_lines = ["📍 **แผนการพัฒนา (Action Plan)**"]
        # รองรับทั้ง 'action_plan' และ 'Action_Plan'
        plans = res.get("action_plan", res.get("Action_Plan", []))

        if plans:
            for p in plans:
                # ดึงข้อมูล Phase
                phase_name = p.get("Phase", p.get("phase", "ไม่ระบุเฟส"))
                goal = p.get("Goal", p.get("goal", ""))
                
                gap_lines.append(f"### 🚀 {phase_name}")
                if goal:
                    gap_lines.append(f"**เป้าหมาย**: {goal}\n")

                # ดึงรายการ Actions
                actions = p.get("Actions", p.get("actions", []))
                for act in actions:
                    rec = act.get("Recommendation", act.get("recommendation", "ไม่มีคำแนะนำระบุไว้"))
                    target_lv = act.get("Failed_Level", act.get("failed_level", "?"))
                    
                    gap_lines.append(f"#### 📝 ข้อเสนอแนะเพื่อแก้ไข Level {target_lv}")
                    gap_lines.append(f"> {rec}\n")
                    
                    # ข้อมูลสนับสนุน
                    ev_type = act.get("Target_Evidence_Type", act.get("target_evidence_type", "-"))
                    metric = act.get("Key_Metric", act.get("key_metric", "-"))
                    gap_lines.append(f"- 📂 **หลักฐานที่ต้องจัดทำ**: {ev_type}")
                    gap_lines.append(f"- 📊 **ตัวชี้วัดความสำเร็จ**: {metric}\n")

                    # ดึงรายการ Steps ย่อย
                    steps = act.get("Steps", act.get("steps", []))
                    if steps:
                        gap_lines.append("##### 👣 ขั้นตอนการดำเนินงาน:")
                        for s in steps:
                            s_num = s.get("Step", s.get("step", "?"))
                            s_desc = s.get("Description", s.get("description", ""))
                            s_resp = s.get("Responsible", s.get("responsible", "-"))
                            s_outcome = s.get("Verification_Outcome", s.get("verification_outcome", "-"))
                            
                            gap_lines.append(f"{s_num}. **{s_desc}**")
                            gap_lines.append(f"   - 👤 ผู้รับผิดชอบ: {s_resp}")
                            gap_lines.append(f"   - 🏁 ผลลัพธ์: {s_outcome}")
                        gap_lines.append("") # เว้นบรรทัด
        else:
            # กรณีไม่มี Action Plan ให้ดึงข้อเสนอแนะพื้นฐานมาแสดง
            suggestion = clean_suggestion(res.get("suggestion_next_level", res.get("suggestion")))
            gap_lines.append(f"\n💡 **ข้อเสนอแนะเพิ่มเติม**:\n{suggestion}")

        # รวบรวมข้อมูลลงใน processed_sub_criteria
        full_gap_text = "\n".join(gap_lines)
        
        processed_sub_criteria.append({
            "code": cid,
            "name": cname,
            "level": f"L{highest_pass}",
            "score": float(res.get("weighted_score", highest_pass)),
            "evidence": "\n".join(evidence_lines),
            "gap": full_gap_text
        })

        # เตรียมข้อมูลสำหรับ Radar Chart
        radar_data.append({
            "axis": cid,
            "value": highest_pass,
            "fullMark": 5
        })

        # แยกจุดแข็ง (ถ้าผ่าน L3 ขึ้นไป)
        if highest_pass >= 3:
            strengths.append(f"เกณฑ์ {cid}: บรรลุมาตรฐานระดับสูง (L{highest_pass})")

    # สรุปภาพรวมส่งกลับไปให้ UI
    return {
        "status": "COMPLETED",
        "record_id": raw_data.get("record_id", summary.get("record_id", "unknown")),
        "tenant": summary.get("tenant", current_user.tenant),
        "year": str(summary.get("year", current_user.year)),
        "enabler": (summary.get("enabler") or "KM").upper(),
        "level": f"L{overall_level}",
        "score": round(float(summary.get("Total Weighted Score Achieved", 0.0)), 2),
        "metrics": {
            "total_criteria": total_expected,
            "passed_criteria": passed_count,
            "completion_rate": round(completion_rate, 1)
        },
        "radar_data": radar_data,
        "strengths": strengths if strengths else ["ระบบวิเคราะห์สำเร็จ"],
        "weaknesses": [c["gap"] for c in processed_sub_criteria if "Action Plan" in c["gap"]],
        "sub_criteria": processed_sub_criteria,
        "progress_message": "ดึงข้อมูลสำเร็จ"
    }

# ------------------- API Endpoints -------------------
@assessment_router.get("/status/{record_id}")
async def get_assessment_status(record_id: str, current_user: UserMe = Depends(get_current_user)):
    if record_id in ACTIVE_TASKS:
        return ACTIVE_TASKS[record_id]

    file_path = _find_assessment_file(record_id, current_user)
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    enabler = (raw_data.get("summary", {}).get("enabler") or "KM").upper()
    check_user_permission(current_user, current_user.tenant, enabler)

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
            if f.lower().endswith(".json"):
                try:
                    file_path = os.path.join(root, f)
                    with open(file_path, "r", encoding="utf-8") as jf:
                        data = json.load(jf)
                        summary = data.get("summary", {})
                        enabler = (summary.get("enabler") or "KM").upper()
                        check_user_permission(current_user, tenant, enabler)

                        history_list.append({
                            "record_id": data.get("record_id") or summary.get("record_id") or f.rsplit('.', 1)[0],
                            "date": parse_safe_date(summary.get("export_timestamp"), file_path),
                            "tenant": tenant,
                            "year": str(year),
                            "enabler": enabler,
                            "scope": summary.get("sub_criteria_id", "ALL"),
                            "level": f"L{summary.get('highest_pass_level_overall', summary.get('highest_pass_level', 0))}",
                            "score": round(float(summary.get("Total Weighted Score Achieved", summary.get("achieved_weight", 0.0))), 2),
                            "status": "COMPLETED"
                        })
                except Exception as e:
                    logger.error(f"Error reading history file {f}: {e}")

    return {"items": sorted(history_list, key=lambda x: x['date'], reverse=True)}

@assessment_router.post("/start")
async def start_assessment(request: StartAssessmentRequest, background_tasks: BackgroundTasks, current_user: UserMe = Depends(get_current_user)):
    check_user_permission(current_user, request.tenant, request.enabler)

    record_id = uuid.uuid4().hex[:12]
    ACTIVE_TASKS[record_id] = {
        "status": "RUNNING",
        "record_id": record_id,
        "tenant": request.tenant,
        "year": str(request.year),
        "enabler": request.enabler.upper(),
        "progress_message": f"กำลังวิเคราะห์ {request.sub_criteria or 'ทุกเกณฑ์'}..."
    }

    background_tasks.add_task(
        run_assessment_engine_task,
        record_id,
        request.tenant,
        int(request.year),
        request.enabler,
        request.sub_criteria,
        request.sequential_mode
    )

    return {"record_id": record_id, "status": "RUNNING"}

async def run_assessment_engine_task(record_id: str, tenant: str, year: int, enabler: str, sub_id: str, sequential: bool):
    try:
        vsm = await asyncio.to_thread(
            load_all_vectorstores,
            doc_types=EVIDENCE_DOC_TYPES,
            enabler_filter=enabler,
            tenant=tenant,
            year=str(year)
        )
        
        doc_map_raw = await asyncio.to_thread(
            load_doc_id_mapping, 
            EVIDENCE_DOC_TYPES, 
            tenant, 
            str(year), 
            enabler
        )
        
        doc_map = {d_id: d.get("file_name", d_id) for d_id, d in doc_map_raw.items()}

        llm = await asyncio.to_thread(create_llm_instance, model_name=DEFAULT_LLM_MODEL_NAME, temperature=0.0)
        
        config = AssessmentConfig(
            enabler=enabler, 
            tenant=tenant, 
            year=str(year),
            force_sequential=sequential
        )

        # แก้ตรงนี้: ส่ง parameter ถูกต้องตาม constructor
        engine = SEAMPDCAEngine(
            config=config,
            llm_instance=llm,
            logger_instance=logger,
            doc_type=EVIDENCE_DOC_TYPES,
            vectorstore_manager=vsm,        # object จริง
            document_map=doc_map            # dict แยก
        )

        await asyncio.to_thread(engine.run_assessment, sub_id, True, vsm, sequential, record_id)

        if record_id in ACTIVE_TASKS:
            del ACTIVE_TASKS[record_id]
            
    except Exception as e:
        logger.error(f"Engine Failed: {e}", exc_info=True)
        if record_id in ACTIVE_TASKS:
            ACTIVE_TASKS[record_id]["status"] = "FAILED"
            ACTIVE_TASKS[record_id]["error_message"] = str(e)

@assessment_router.get("/download/{record_id}/{file_type}")
async def download_assessment_file(record_id: str, file_type: str, current_user: UserMe = Depends(get_current_user)):
    file_path = _find_assessment_file(record_id, current_user)

    expected_ext = f".{file_type.lower()}"
    if file_type.lower() == "word":
        expected_ext = ".docx"

    if not file_path.endswith(expected_ext):
        raise HTTPException(status_code=404, detail="ประเภทไฟล์ไม่ถูกต้อง")

    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        enabler = (raw_data.get("summary", {}).get("enabler") or "KM").upper()
        check_user_permission(current_user, current_user.tenant, enabler)

    return FileResponse(path=file_path, filename=os.path.basename(file_path))