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
from config.global_vars import EVIDENCE_DOC_TYPES, DEFAULT_LLM_MODEL_NAME, DEFAULT_YEAR


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

def _transform_result_for_ui(raw_data: Dict[str, Any], current_user: Any = None) -> Dict[str, Any]:
    """
    เวอร์ชันปรับปรุงสมบูรณ์: รองรับ PDCA Matrix, Roadmap Stepper และวิเคราะห์จุดแข็ง/อ่อน
    ให้สอดคล้องกับ UI เวอร์ชันใหม่
    """
    summary = raw_data.get("summary", {})
    sub_results = raw_data.get("sub_criteria_results", [])

    processed_sub_criteria = []
    radar_data = []
    strengths = []
    all_weaknesses = []

    # --- 1. ดึงค่า Metrics พื้นฐาน ---
    total_expected = int(summary.get("total_subcriteria") or 0)
    passed_count = int(summary.get("total_subcriteria_assessed") or len(sub_results))
    completion_rate = float(summary.get("percentage_achieved_run") or 0.0)
    overall_level = summary.get("Overall Maturity Level (Weighted)") or f"L{summary.get('highest_pass_level', 0)}"
    total_score = round(float(summary.get("Overall Maturity Score (Avg.)") or summary.get("Total Weighted Score Achieved") or 0.0), 2)
    enabler_name = (summary.get("enabler") or "N/A").upper()

    for res in sub_results:
        cid = res.get("sub_criteria_id", "N/A")
        cname = res.get("sub_criteria_name", f"เกณฑ์ย่อย {cid}")
        highest_pass = int(res.get("highest_full_level") or 0)
        raw_levels_list = res.get("raw_results_ref", [])

        # --- 2. สร้าง PDCA Matrix (สำหรับ UI Badge Grid) ---
        pdca_matrix = []
        raw_levels_map = {item.get("level"): item for item in raw_levels_list}
        
        for lv in range(1, 6):
            lv_info = raw_levels_map.get(lv)
            if lv_info:
                pdca_matrix.append({
                    "level": lv,
                    "is_passed": lv_info.get("is_passed", False),
                    "pdca": lv_info.get("pdca_breakdown", {"P": 0, "D": 0, "C": 0, "A": 0}),
                    "reason": lv_info.get("reason", "ประเมินแล้ว")
                })
            else:
                pdca_matrix.append({
                    "level": lv,
                    "is_passed": lv <= highest_pass,
                    "pdca": {"P": 1, "D": 1, "C": 1, "A": 1} if lv <= highest_pass else {"P": 0, "D": 0, "C": 0, "A": 0},
                    "reason": "ผ่านเกณฑ์มาตรฐาน" if lv <= highest_pass else "ยังไม่ถึงเกณฑ์ประเมิน"
                })

        # --- 3. สร้าง Roadmap (สำหรับ UI Stepper) ---
        ui_roadmap = []
        raw_plans = res.get("action_plan") or []
        for p in raw_plans:
            ui_roadmap.append({
                "phase": p.get("phase", "แผนงานพัฒนา"),
                "goal": p.get("goal", "เพื่อยกระดับตามเกณฑ์"),
                "tasks": [
                    {
                        "level": str(act.get("failed_level", lv + 1)),
                        "recommendation": act.get("recommendation", ""),
                        "steps": [
                            {
                                "step": str(s.get("step") or s.get("step_number") or i+1),
                                "description": s.get("description", ""),
                                "responsible": s.get("responsible", "หน่วยงานที่เกี่ยวข้อง")
                            } for i, s in enumerate(act.get("steps", []))
                        ]
                    } for act in p.get("actions", [])
                ]
            })

        # --- 4. ดึง Sources/Evidence Link ---
        all_sources = []
        seen_docs = set()
        # รวมแหล่งอ้างอิงจากทุก level
        for ref in raw_levels_list:
            for source in ref.get("temp_map_for_level", []):
                doc_id = f"{source.get('filename')}-{source.get('page_number')}"
                if doc_id not in seen_docs:
                    all_sources.append({
                        "filename": source.get("filename") or source.get("source"),
                        "page": str(source.get("page_number") or source.get("page", "1")),
                        "snippet": source.get("text", "")[:150]
                    })
                    seen_docs.add(doc_id)

        # --- 5. สรุปจุดแข็ง/จุดอ่อน รายหัวข้อ ---
        for lv_item in raw_levels_list:
            if lv_item.get("level", 0) >= 3 and lv_item.get("is_passed"):
                strengths.append(f"เกณฑ์ {cid}: บรรลุระดับ L{lv_item['level']} พร้อมหลักฐานที่ชัดเจน")
            
        for plan in raw_plans:
            for act in plan.get("actions", []):
                all_weaknesses.append(f"L{act.get('failed_level')}: {act.get('recommendation')}")

        processed_sub_criteria.append({
            "code": cid,
            "name": cname,
            "level": f"L{highest_pass}",
            "score": float(res.get("weighted_score", 0.0)),
            "progress_percent": int((highest_pass / 5) * 100),
            "pdca_matrix": pdca_matrix, # มั่นใจว่าชื่อตัวแปรตรงกับ UI
            "roadmap": ui_roadmap,      # มั่นใจว่าชื่อตัวแปรตรงกับ UI
            "sources": all_sources[:5],
            "evidence": res.get("summary_thai", ""),
            "gap": res.get("gap_analysis", "") # หรือฟิลด์ที่พี่ต้องการ
        })
        
        radar_data.append({"axis": cid, "value": highest_pass})

    return {
        "status": "COMPLETED",
        "record_id": raw_data.get("record_id", "unknown"),
        "tenant": str(summary.get("tenant", "N/A")).upper(),
        "year": str(summary.get("year", "2568")),
        "enabler": enabler_name,
        "level": overall_level,
        "score": total_score,
        "metrics": {
            "total_criteria": total_expected,
            "passed_criteria": passed_count,
            "completion_rate": int(completion_rate)
        },
        "radar_data": radar_data,
        "strengths": list(dict.fromkeys(strengths)) if strengths else ["โครงสร้างพื้นฐานมีความพร้อม"],
        "weaknesses": list(dict.fromkeys(all_weaknesses)) if all_weaknesses else ["บรรลุเกณฑ์ประเมิน"],
        "sub_criteria": processed_sub_criteria
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

    # 🟢 จัดการกรณี Year ไม่ถูกเลือกหรือส่งมาว่างๆ
    # ถ้าไม่มีค่าส่งมา ให้ใช้ปีจาก Profile ของ User หรือใช้ค่า Default ของระบบ (เช่น 2568)
    raw_year = request.year
    target_year = str(raw_year).strip() if (raw_year and str(raw_year).strip()) else str(current_user.year or DEFAULT_YEAR)

    # จัดการ sub_criteria (เหมือนเดิม)
    target_sub = request.sub_criteria.strip() if (request.sub_criteria and request.sub_criteria.strip()) else "all"

    record_id = uuid.uuid4().hex[:12]
    ACTIVE_TASKS[record_id] = {
        "status": "RUNNING",
        "record_id": record_id,
        "tenant": request.tenant,
        "year": target_year, # ใช้ปีที่ผ่านการตรวจสอบแล้ว
        "enabler": request.enabler.upper(),
        "progress_message": f"กำลังเริ่มการประเมินปี {target_year}..."
    }

    background_tasks.add_task(
        run_assessment_engine_task,
        record_id,
        request.tenant,
        int(target_year), # ส่งปีที่ชัวร์แล้วเข้าไป
        request.enabler,
        target_sub,
        request.sequential_mode
    )

    return {"record_id": record_id, "status": "RUNNING"}

async def run_assessment_engine_task(record_id: str, tenant: str, year: int, enabler: str, sub_id: str, sequential: bool):
    try:
        # --- 1. เตรียม Resource (เหมือนเดิม) ---
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

        # 🟢 แก้จุดที่ 2: Initialize Engine (ลำดับเหมือน CLI)
        engine = SEAMPDCAEngine(
            config=config,
            llm_instance=llm,
            logger_instance=logger,
            doc_type=EVIDENCE_DOC_TYPES,
            vectorstore_manager=vsm,
            document_map=doc_map
        )

        # 🟢 แก้จุดที่ 3: เรียก run_assessment แบบระบุชื่อตัวแปร (Explicit) เพื่อความชัวร์
        # ใช้ sub_id ที่ส่งมาจากจุดที่ 1 (มั่นใจว่าเป็น 'all' หรือ '3.1')
        await asyncio.to_thread(
            engine.run_assessment, 
            target_sub_id=sub_id, 
            export=True, 
            vectorstore_manager=vsm, 
            sequential=sequential, 
            record_id=record_id,
            document_map=doc_map
        )

        if record_id in ACTIVE_TASKS:
            del ACTIVE_TASKS[record_id]
            
    except Exception as e:
        logger.error(f"❌ Engine Failed for Record {record_id}: {e}", exc_info=True)
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