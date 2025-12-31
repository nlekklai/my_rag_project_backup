# -*- coding: utf-8 -*-
# routers/assessment_router.py
# Production Final Version - 20 ธันวาคม 2568 (Fixed parameter order + stable UUID + full assessment flow)

import os
import uuid
import json
import asyncio
import logging
import mimetypes
from datetime import datetime
from typing import Optional, Dict, Any, Union, List

from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel

from routers.auth_router import UserMe, get_current_user
from utils.path_utils import _n, get_tenant_year_export_root, load_doc_id_mapping, get_document_file_path
from core.seam_assessment import SEAMPDCAEngine, AssessmentConfig
from core.vectorstore import load_all_vectorstores
from models.llm import create_llm_instance
from config.global_vars import EVIDENCE_DOC_TYPES, DEFAULT_LLM_MODEL_NAME, DEFAULT_YEAR
import pytz


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
    tz = pytz.timezone('Asia/Bangkok') # กำหนด Timezone ไทย
    
    if raw_date_str and isinstance(raw_date_str, str):
        try:
            # ถ้ามีรูปแบบ %Y%m%d_%H%M%S (เช่นจากชื่อไฟล์)
            if "_" in raw_date_str:
                dt = datetime.strptime(raw_date_str, "%Y%m%d_%H%M%S")
                # บังคับให้เป็นเวลาไทย
                return tz.localize(dt).isoformat()
        except:
            pass

    try:
        # ดึงเวลาที่แก้ไขไฟล์ล่าสุดจาก Disk
        mtime = os.path.getmtime(file_path)
        dt = datetime.fromtimestamp(mtime, tz) # ระบุ Timezone ตอนดึง timestamp
        return dt.isoformat()
    except:
        # กรณีผิดพลาดให้ใช้เวลาปัจจุบันที่เป็น Thai Timezone
        return datetime.now(tz).isoformat()

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


@assessment_router.get("/evidence/{doc_type}/{document_uuid}")
async def serve_evidence_file(
    document_uuid: str,
    doc_type: str,
    tenant: str,
    year: str = None,
    enabler: str = None,
    current_user: UserMe = Depends(get_current_user)
):
    check_user_permission(current_user, tenant, enabler or "KM")

    file_info = get_document_file_path(
        document_uuid=document_uuid,
        tenant=tenant,
        year=year,
        enabler=enabler,
        doc_type_name=doc_type
    )

    if not file_info:
        raise HTTPException(status_code=404, detail="File not found")

    file_path = file_info["file_path"]
    
    # ดึงนามสกุลไฟล์
    ext = os.path.splitext(file_path)[1].lower()
    
    # 🛡️ Force MIME Type สำหรับ Mac/Safari
    mime_map = {
        ".pdf": "application/pdf",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
    }
    
    mime_type = mime_map.get(ext) or mimetypes.guess_type(file_path)[0] or "application/octet-stream"

    # ส่ง FileResponse
    response = FileResponse(
        path=file_path,
        media_type=mime_type,
        content_disposition_type="inline"
    )

    # 💡 หัวใจสำคัญสำหรับ Mac/Safari:
    # 1. ป้องกันไม่ให้ Browser ใช้ชื่อไฟล์จาก Path ซึ่งบางทีมีภาษาไทยแล้วทำให้ Header เพี้ยน
    # 2. บังคับ Header ให้ชัดเจน
    response.headers["Content-Type"] = mime_type
    response.headers["Accept-Ranges"] = "bytes" 
    
    # ถ้าเป็น PDF บน Mac ให้เติม Cache-Control เพื่อให้ Viewer ทำงานได้ดีขึ้น
    if ext == ".pdf":
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"

    return response


@assessment_router.get("/view-document")
async def view_document(filename: str, page: Optional[str] = "1", current_user: UserMe = Depends(get_current_user)):
    """ Endpoint สำหรับเปิดไฟล์ PDF ไปยังหน้าที่ระบุ """
    # ค้นหาไฟล์ในโฟลเดอร์เก็บเอกสารของ Tenant
    import os
    from utils.path_utils import get_tenant_year_import_root
    
    # สมมติว่าไฟล์เก็บอยู่ที่โฟลเดอร์ import/EVIDENCE_DOC
    base_path = os.path.join(get_tenant_year_import_root(current_user.tenant, current_user.year), "EVIDENCE_DOC")
    file_path = os.path.join(base_path, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"ไม่พบไฟล์เอกสาร: {filename}")

    # ส่งไฟล์กลับไปเพื่อให้ Browser เปิด (ระบุหน้าด้วย #page=X ในฝั่ง Frontend)
    return FileResponse(file_path, media_type="application/pdf")

def _transform_result_for_ui(raw_data: Dict[str, Any], current_user: Any = None) -> Dict[str, Any]:
    """
    เวอร์ชันแก้ไขสมบูรณ์:
    1. แก้ไข NameError: total_score is not defined
    2. Hybrid Logic: รวม 'ผลวิเคราะห์จริง' กับ 'รายละเอียดเกณฑ์ (Context)'
    3. จัดการข้อมูล summary_thai ที่มีหลายจุดโดยเลือกก้อนที่สมบูรณ์ที่สุด
    """
    summary = raw_data.get("summary", {})
    sub_results = raw_data.get("sub_criteria_results", [])

    processed_sub_criteria = []
    radar_data = []
    strengths = []

    # --- 1. Metrics & Score Calculation (จุดที่ต้องมีเพื่อแก้ NameError) ---
    total_score = round(float(summary.get("Total Weighted Score Achieved") or summary.get("achieved_weight") or 0.0), 2)
    full_score_all = round(float(summary.get("Total Possible Weight") or 40.0), 2)
    
    total_expected = int(summary.get("total_subcriteria") or 12)
    passed_count = int(summary.get("total_subcriteria_assessed") or len(sub_results))
    completion_rate = (passed_count / total_expected * 100) if total_expected > 0 else 0.0
    
    overall_level = summary.get("Overall Maturity Level (Weighted)") or f"L{summary.get('highest_pass_level', 0)}"
    enabler_name = (summary.get("enabler") or "N/A").upper()

    for res in sub_results:
        cid = res.get("sub_criteria_id", "N/A")
        cname = res.get("sub_criteria_name", f"เกณฑ์ย่อย {cid}")
        highest_pass = int(res.get("highest_full_level") or 0)
        raw_levels_list = res.get("raw_results_ref", [])
        
        # คะแนนรายข้อ
        current_sub_score = round(float(res.get("weighted_score", 0.0)), 2)
        current_sub_full = round(float(res.get("weight", 0.0)), 2)

        # --- 2. สร้าง PDCA Matrix ---
        pdca_matrix = []
        raw_levels_map = {item.get("level"): item for item in raw_levels_list}
        
        for lv_idx in range(1, 6):
            lv_info = raw_levels_map.get(lv_idx)
            if lv_info:
                pdca_matrix.append({
                    "level": lv_idx,
                    "is_passed": lv_info.get("is_passed", False),
                    "pdca": lv_info.get("pdca_breakdown", {"P": 0, "D": 0, "C": 0, "A": 0}),
                    "reason": lv_info.get("reason", "ประเมินแล้ว")
                })
            else:
                pdca_matrix.append({
                    "level": lv_idx,
                    "is_passed": lv_idx <= highest_pass,
                    "pdca": {"P": 1, "D": 1, "C": 1, "A": 1} if lv_idx <= highest_pass else {"P": 0, "D": 0, "C": 0, "A": 0},
                    "reason": "ผ่านเกณฑ์มาตรฐาน" if lv_idx <= highest_pass else "ยังไม่ถึงเกณฑ์ประเมิน"
                })

        # --- 3. จัดกลุ่ม Sources (ชื่อไฟล์จริง) ---
        grouped_sources = {str(lv): [] for lv in range(1, 6)}
        for ref in raw_levels_list:
            lv_key = str(ref.get("level"))
            seen_in_lv = set()
            for source in ref.get("temp_map_for_level", []):
                fname = source.get('source_filename') or source.get('source') or source.get('filename') or "Unknown Document"
                pnum = str(source.get('page_number') or source.get('page') or "1")
                d_uuid = source.get('document_uuid') or source.get('doc_id')
                if not d_uuid: continue
                doc_key = f"{fname}-{pnum}"
                if doc_key not in seen_in_lv:
                    grouped_sources[lv_key].append({
                        "filename": fname,
                        "page": pnum,
                        "snippet": source.get("text", "")[:150],
                        "document_uuid": d_uuid,
                        "doc_type": source.get("doc_type", "evidence")
                    })
                    seen_in_lv.add(doc_key)

        # --- 4. สร้าง Roadmap ---
        ui_roadmap = []
        raw_plans = res.get("action_plan") or []
        for p in raw_plans:
            ui_roadmap.append({
                "phase": p.get("phase", "แผนงานพัฒนา"),
                "goal": p.get("goal", "เพื่อยกระดับตามเกณฑ์"),
                "tasks": [
                    {
                        "level": str(act.get("failed_level", highest_pass + 1)),
                        "recommendation": act.get("recommendation", ""),
                        "steps": [
                            {
                                "step": str(s.get("step") or i+1),
                                "description": s.get("description", ""),
                                "responsible": s.get("responsible", "หน่วยงานที่เกี่ยวข้อง")
                            } for i, s in enumerate(act.get("steps", []))
                        ]
                    } for act in p.get("actions", [])
                ]
            })

        # --- 5. Hybrid Summary Logic (ปรับปรุงตามคำขอ: แยกผลวิเคราะห์จากไฟล์จริง) ---
        
        # 1. ดึง "ผลวิเคราะห์หลักฐาน" (Analysis)
        # วน Loop ถอยหลังเพื่อหาเหตุผลจากระดับสูงสุดที่สอบผ่าน และต้องมี Source อ้างอิง
        evidence_analysis = ""
        if pdca_matrix:
            for m in reversed(pdca_matrix):
                r_text = m.get("reason", "")
                if "[Source:" in r_text or "[อ้างอิง:" in r_text:
                    evidence_analysis = r_text
                    break

        # 2. ดึง "รายละเอียดเกณฑ์" (Context)
        # ดึงจาก summary_thai ระดับ sub-criteria (ขุมทรัพย์ข้อมูล L1-L5 ที่ยาว 10-12 บรรทัด)
        context_criteria = res.get("summary_thai") or ""

        # 3. รวมร่าง (Hybrid Merge)
        # จัด Format ให้สวยงามพร้อมใช้ ReactMarkdown ใน Frontend
        if evidence_analysis and context_criteria:
            # ตรวจสอบเบื้องต้นเผื่อข้อมูลซ้ำซ้อน
            if evidence_analysis[:40] in context_criteria:
                sthai = context_criteria
            else:
                sthai = (
                    "**ผลการวิเคราะห์หลักฐาน:**\n"
                    f"{evidence_analysis}\n\n"
                    "**รายละเอียดเกณฑ์อ้างอิง (Context):**\n"
                    f"{context_criteria}"
                )
        else:
            sthai = evidence_analysis or context_criteria or "ไม่พบข้อมูลสรุปผลวิเคราะห์"

        # --- 6. GAP Analysis (รวมจาก Action Plan ถ้าฟิลด์ GAP ว่าง) ---
        gap_data = res.get("gap_analysis") or res.get("gap") or ""
        if not gap_data.strip() and ui_roadmap:
            gap_list = []
            for phase in ui_roadmap:
                for task in phase.get("tasks", []):
                    if task.get("recommendation"):
                        gap_list.append(f"L{task['level']}: {task['recommendation']}")
            gap_data = "\n".join(gap_list)

        # เพิ่มลงใน List เพื่อส่งออกไปที่ UI
        processed_sub_criteria.append({
            "code": cid,
            "name": cname,
            "level": f"L{highest_pass}",
            "score": current_sub_score,
            "full_score": current_sub_full,
            "progress_percent": int((highest_pass / 5) * 100),
            "pdca_matrix": pdca_matrix,
            "roadmap": ui_roadmap,
            "grouped_sources": grouped_sources,
            "summary_thai": sthai.strip(),
            "gap": gap_data.strip()
        })

        radar_data.append({"axis": cid, "value": int(highest_pass)})

    return {
        "status": "COMPLETED",
        "record_id": raw_data.get("record_id", "unknown"),
        "tenant": str(summary.get("tenant", "N/A")).upper(),
        "year": str(summary.get("year", "2568")),
        "enabler": enabler_name,
        "level": overall_level,
        "score": total_score,
        "full_score": full_score_all,
        "metrics": {
            "total_criteria": total_expected,
            "passed_criteria": passed_count,
            "completion_rate": round(completion_rate, 2)
        },
        "radar_data": radar_data,
        "strengths": list(dict.fromkeys(strengths)) if strengths else ["โครงสร้างพื้นฐานมีความพร้อม"],
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

# --- แก้ไขส่วน Start Assessment Request ---
@assessment_router.post("/start")
async def start_assessment(
    request: StartAssessmentRequest, 
    background_tasks: BackgroundTasks, 
    current_user: UserMe = Depends(get_current_user)
):
    # 1. ตรวจสอบสิทธิ์ (Enabler ต้องเป็นตัวใหญ่)
    enabler_uc = request.enabler.upper()
    check_user_permission(current_user, request.tenant, enabler_uc)

    # 2. จัดการ Year: ถ้าไม่ส่งมาให้ใช้จาก User หรือ Default
    # สำคัญ: ต้องเป็น string สำหรับ path_utils
    target_year = str(request.year).strip() if request.year else str(current_user.year or DEFAULT_YEAR)
    
    # 3. จัดการ Sub-criteria: ถ้าว่างให้เป็น "all"
    target_sub = str(request.sub_criteria).strip().lower() if request.sub_criteria else "all"

    # 4. สร้าง Record ID
    record_id = uuid.uuid4().hex[:12]
    
    # 5. บันทึกลง ACTIVE_TASKS เพื่อให้ UI ดึงสถานะได้ทันที
    ACTIVE_TASKS[record_id] = {
        "status": "RUNNING",
        "record_id": record_id,
        "tenant": request.tenant,
        "year": target_year,
        "enabler": enabler_uc,
        "progress_message": f"กำลังเริ่มการประเมิน {enabler_uc} ปี {target_year}..."
    }

    # 6. เรียก Background Task (ระบุชื่อตัวแปรให้ชัดเจนป้องกันลำดับสลับ)
    background_tasks.add_task(
        run_assessment_engine_task,
        record_id=record_id,
        tenant=request.tenant,
        year=int(target_year), # แปลงเป็น int สำหรับ Engine บางส่วนที่ต้องการเลข
        enabler=enabler_uc,
        sub_id=target_sub,
        sequential=request.sequential_mode
    )

    return {"record_id": record_id, "status": "RUNNING"}


async def run_assessment_engine_task(
    record_id: str, 
    tenant: str, 
    year: int, 
    enabler: str, 
    sub_id: str, 
    sequential: bool
):
    """
    Background Task สำหรับรัน Engine
    ปรับปรุง: ตรวจสอบสถานะ FAILED จาก Engine เพื่อป้องกันการส่งข้อมูลผิดพลาดไปหน้า UI
    """
    try:
        # 1. เตรียมข้อมูลพื้นฐาน
        str_year = str(year)
        logger.info(f"🚀 [TASK START] Record: {record_id} | Enabler: {enabler} | Sub-ID: {sub_id}")

        # 2. Load Vectorstores
        vsm = await asyncio.to_thread(
            load_all_vectorstores,
            doc_types=EVIDENCE_DOC_TYPES,
            enabler_filter=enabler,
            tenant=tenant,
            year=str_year
        )
        
        # 3. Load Document Mapping
        doc_map_raw = await asyncio.to_thread(
            load_doc_id_mapping, 
            EVIDENCE_DOC_TYPES, 
            tenant, 
            str_year, 
            enabler
        )
        doc_map = {d_id: d.get("file_name", d_id) for d_id, d in doc_map_raw.items()}

        # 4. Create LLM Instance
        llm = await asyncio.to_thread(
            create_llm_instance, 
            model_name=DEFAULT_LLM_MODEL_NAME, 
            temperature=0.0
        )
        
        # 5. Initialize Engine
        config = AssessmentConfig(
            enabler=enabler, 
            tenant=tenant, 
            year=str_year,
            force_sequential=sequential
        )

        engine = SEAMPDCAEngine(
            config=config,
            llm_instance=llm,
            logger_instance=logger,
            doc_type=EVIDENCE_DOC_TYPES,
            vectorstore_manager=vsm,
            document_map=doc_map
        )

        # 6. Execution: รันการประเมินและเก็บผลลัพธ์เพื่อตรวจสอบ Error
        # เราใช้ asyncio.to_thread เพราะ Engine ทำงานเป็น Synchronous/CPU Bound
        result = await asyncio.to_thread(
            engine.run_assessment, 
            target_sub_id=sub_id, 
            export=True, 
            vectorstore_manager=vsm, 
            sequential=sequential, 
            record_id=record_id,
            document_map=doc_map
        )

        # 7. ตรวจสอบสถานะผลลัพธ์ (Critical Check)
        if isinstance(result, dict) and result.get("status") == "FAILED":
            error_msg = result.get("error_message", "Engine reported an unspecified error")
            logger.error(f"❌ [TASK FAILED] Record {record_id}: {error_msg}")
            
            if record_id in ACTIVE_TASKS:
                ACTIVE_TASKS[record_id]["status"] = "FAILED"
                ACTIVE_TASKS[record_id]["error_message"] = error_msg
                # เราไม่ลบออกจาก ACTIVE_TASKS เพื่อให้ UI แสดงหน้า Error ได้ถูก
            return

        # 8. Success Cleanup: ลบออกจาก Task ที่กำลังทำงานเพื่อให้ API ไปอ่านจาก JSON จริง
        if record_id in ACTIVE_TASKS:
            del ACTIVE_TASKS[record_id]
            logger.info(f"✅ [TASK COMPLETED] Record: {record_id}")
            
    except Exception as e:
        logger.error(f"💥 [TASK CRASH] Record {record_id}: {str(e)}", exc_info=True)
        if record_id in ACTIVE_TASKS:
            ACTIVE_TASKS[record_id]["status"] = "FAILED"
            ACTIVE_TASKS[record_id]["error_message"] = f"Internal Server Error: {str(e)}"

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
