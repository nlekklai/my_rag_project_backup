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

from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from routers.auth_router import UserMe, get_current_user
from utils.path_utils import _n, get_tenant_year_export_root, load_doc_id_mapping, get_document_file_path, get_vectorstore_collection_path
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

def _find_assessment_file(search_id: str, current_user: UserMe) -> str:
    # 1. หา root ของ tenant
    # ลองหาปี 2568 เป็นตัวตั้งต้นก่อน
    sample_path = get_tenant_year_export_root(current_user.tenant, "2568")
    tenant_export_root = os.path.dirname(sample_path)
    
    norm_search = _n(search_id).lower()

    # 2. เพิ่มการตรวจสอบ Path สำรอง (กรณีรันบน Linux/Docker แล้ว /app/ หายไป)
    search_paths = [tenant_export_root]
    if tenant_export_root.startswith("/app/"):
        search_paths.append(tenant_export_root.replace("/app/", "", 1))

    for s_path in search_paths:
        if os.path.exists(s_path):
            for root, _, files in os.walk(s_path):
                for f in files:
                    if f.endswith(".json") and norm_search in _n(f).lower():
                        return os.path.join(root, f)
                    
    raise HTTPException(status_code=404, detail=f"ไม่พบไฟล์ผลการประเมิน ID: {search_id}")


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
    เวอร์ชัน Hybrid + Bottleneck Support:
    1. รองรับ is_gap_analysis เพื่อโชว์ Badge Bottleneck ใน UI
    2. ส่ง rerank_score และ snippet แบบเต็มเพื่อทำ Tooltip
    3. แยก evaluation_mode (NORMAL/GAP_ONLY) สำหรับการระบายสี PDCA Matrix
    """
    summary = raw_data.get("summary", {})
    sub_results = raw_data.get("sub_criteria_results", [])

    processed_sub_criteria = []
    radar_data = []

    # --- 1. Metrics & Score Calculation ---
    total_score = round(float(summary.get("Total Weighted Score Achieved") or summary.get("achieved_weight") or 0.0), 2)
    full_score_all = round(float(summary.get("Total Possible Weight") or 4.0), 2)
    
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
        
        # --- 2. ตรวจสอบ Bottleneck (ข้ามเลเวล) ---
        # หาว่ามีเลเวลที่สูงกว่าเลเวลปัจจุบันที่ผ่าน (is_passed=True) หรือไม่
        has_higher_potential = any(
            int(r.get("level", 0)) > highest_pass and r.get("is_passed") 
            for r in raw_levels_list
        )

        # คะแนนรายข้อ
        current_sub_score = round(float(res.get("weighted_score", 0.0)), 2)
        current_sub_full = round(float(res.get("weight", 0.0)), 2)

        # --- 3. สร้าง PDCA Matrix (Enhanced for UI colors) ---
        pdca_matrix = []
        raw_levels_map = {item.get("level"): item for item in raw_levels_list}
        
        for lv_idx in range(1, 6):
            lv_info = raw_levels_map.get(lv_idx)
            is_passed = lv_info.get("is_passed", False) if lv_info else (lv_idx <= highest_pass)
            
            # กำหนด Mode: ถ้าผ่านแต่สูงกว่าเลเวลสูงสุดที่เป็นทางการ ให้เป็น GAP_ONLY (สีน้ำเงินใน UI)
            eval_mode = "NORMAL"
            if is_passed and lv_idx > highest_pass:
                eval_mode = "GAP_ONLY"

            pdca_matrix.append({
                "level": lv_idx,
                "is_passed": is_passed,
                "evaluation_mode": eval_mode,
                "pdca": lv_info.get("pdca_breakdown", {"P": 0, "D": 0, "C": 0, "A": 0}) if lv_info else ({"P": 1, "D": 1, "C": 1, "A": 1} if lv_idx <= highest_pass else {"P": 0, "D": 0, "C": 0, "A": 0}),
                "reason": lv_info.get("reason", "ผ่านเกณฑ์มาตรฐาน") if not lv_info and lv_idx <= highest_pass else (lv_info.get("reason", "") if lv_info else "ยังไม่ถึงเกณฑ์ประเมิน")
            })

        # --- 4. จัดกลุ่ม Sources (ฉบับแก้ไขเพื่อดึง Confidence) ---
        grouped_sources = {str(lv): [] for lv in range(1, 6)}
        for ref in raw_levels_list:
            lv_key = str(ref.get("level"))
            seen_in_lv = set()
            for source in ref.get("temp_map_for_level", []):
                # 1. ดึง Metadata ออกมา (จุดที่มีคะแนนจริง)
                meta = source.get('metadata', {})
                
                fname = source.get('filename') or meta.get('filename') or "Unknown Document"
                pnum = str(source.get('page_number') or meta.get('page') or source.get('page_label') or "1")
                d_uuid = source.get('document_uuid') or source.get('doc_id') or meta.get('doc_id')
                if not d_uuid: continue
                
                doc_key = f"{fname}-{pnum}"
                if doc_key not in seen_in_lv:
                    # ✅ แก้ไขจุดนี้: ต้องดึงจาก metadata เป็นหลัก
                    # ในไฟล์ JSON ของคุณคือ meta.get('rerank_score')
                    score_val = (
                        meta.get("rerank_score") or 
                        source.get("rerank_score") or 
                        meta.get("score") or 
                        source.get("score") or 
                        0.0
                    )

                    grouped_sources[lv_key].append({
                        "filename": fname,
                        "page": pnum,
                        "text": source.get("text", "")[:300],
                        "rerank_score": float(score_val), # ส่งเป็น float เพื่อให้ Frontend แสดงผลได้
                        "document_uuid": d_uuid,
                        "doc_type": source.get("doc_type", "evidence"),
                        "pdca_tag": source.get("pdca_tag") or meta.get("pdca_tag", "N/A")  # เพิ่ม pdca_tag เข้าไป โดยดึงจาก source หรือ metadata
                    })
                    seen_in_lv.add(doc_key)

        # --- 5. Roadmap & Action Plan ---
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
                                "responsible": s.get("responsible", "หน่วยงานหลัก")
                            } for i, s in enumerate(act.get("steps", []))
                        ]
                    } for act in p.get("actions", [])
                ]
            })

        # --- 6. Hybrid Summary & Gap ---
        evidence_analysis = ""
        if pdca_matrix:
            for m in reversed(pdca_matrix):
                if m.get("is_passed"):
                    evidence_analysis = m.get("reason", "")
                    break

        context_criteria = res.get("summary_thai") or ""
        sthai = f"**การวิเคราะห์หลักฐาน:**\n{evidence_analysis}\n\n**เกณฑ์อ้างอิง:**\n{context_criteria}" if evidence_analysis and context_criteria else (evidence_analysis or context_criteria)

        gap_data = res.get("gap_analysis") or res.get("gap") or ""
        if not gap_data.strip() and ui_roadmap:
            gap_list = [f"L{t['level']}: {t['recommendation']}" for ph in ui_roadmap for t in ph.get("tasks", []) if t.get("recommendation")]
            gap_data = "\n".join(gap_list)

        # --- 7. New Features: PDCA Coverage Summary, Avg Confidence, Potential Level ---
        avg_confidence_per_level = {}
        for lv in range(1, 6):
            sources = grouped_sources[str(lv)]
            if sources:
                avg = sum(s['rerank_score'] for s in sources) / len(sources)
                avg_confidence_per_level[lv] = round(avg, 2)
            else:
                avg_confidence_per_level[lv] = 0.0

        potential_level = max((r.get('level') for r in raw_levels_list if r.get('is_passed')), default=highest_pass)

        pdca_coverage = {}
        for m in pdca_matrix:
            pdca = m['pdca']
            covered = sum(1 for v in pdca.values() if v > 0)
            coverage_pct = (covered / 4 * 100) if covered else 0
            pdca_coverage[m['level']] = {
                'percentage': coverage_pct,
                'details': pdca
            }

        # เพิ่มลงใน List เพื่อส่งออกไปที่ UI
        processed_sub_criteria.append({
            "code": cid,
            "name": cname,
            "level": f"L{highest_pass}",
            "score": current_sub_score,
            "full_score": current_sub_full,
            "is_gap_analysis": has_higher_potential, # บ่งบอกว่าเป็นคอขวดหรือไม่
            "pdca_matrix": pdca_matrix,
            "roadmap": ui_roadmap,
            "grouped_sources": grouped_sources,
            "summary_thai": sthai.strip(),
            "gap": gap_data.strip(),
            # New Features
            "avg_confidence_per_level": avg_confidence_per_level,
            "potential_level": f"L{potential_level}",
            "pdca_coverage": pdca_coverage
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
        "sub_criteria": processed_sub_criteria
    }

# ------------------- API Endpoints -------------------
@assessment_router.get("/status/{record_id}")
async def get_assessment_status(record_id: str, current_user: UserMe = Depends(get_current_user)):
    # 1. เช็คใน Memory ก่อน (งานที่กำลังรัน)
    if record_id in ACTIVE_TASKS:
        return ACTIVE_TASKS[record_id]

    # 2. ถ้าไม่อยู่ใน Memory ให้ไปหาใน Disk (งานที่เสร็จแล้ว)
    # ฟังก์ชันนี้จะสแกนหาทุกปีให้เอง
    file_path = _find_assessment_file(record_id, current_user)
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # ดึงข้อมูล Enabler มาเช็ค Permission
        summary = raw_data.get("summary", {})
        enabler = (summary.get("enabler") or "KM").upper()
        tenant = summary.get("tenant") or current_user.tenant
        
        check_user_permission(current_user, tenant, enabler)

        # แปลงข้อมูลส่งให้ UI
        return _transform_result_for_ui(raw_data, current_user)
    except Exception as e:
        logger.error(f"Error loading status for {record_id}: {e}")
        raise HTTPException(status_code=500, detail="ไม่สามารถอ่านไฟล์ผลการประเมินได้")

@assessment_router.get("/history")
async def get_assessment_history(
    tenant: str, 
    year: Optional[str] = Query(None), # แก้จาก Union เป็น Optional และให้ default เป็น None
    current_user: UserMe = Depends(get_current_user)
):
    # 1. ตรวจสอบสิทธิ์องค์กร
    if _n(tenant) != _n(current_user.tenant):
        raise HTTPException(status_code=403, detail="Permission Denied")

    history_list = []
    
    # 2. จัดการเรื่อง "ปี" ที่ต้องการค้นหา
    # ถ้า Frontend ไม่ส่งมา หรือส่งมาเป็น "all" ให้สแกนทุกปี
    search_years = []
    
    # หา Root Path ของ Tenant เพื่อดูว่ามีโฟลเดอร์ปีไหนบ้าง
    # โดยอ้างอิงจากตำแหน่งของโฟลเดอร์ปี 2568 (หรือปีใดก็ได้)
    sample_path = get_tenant_year_export_root(tenant, "2568")
    tenant_export_root = os.path.dirname(sample_path)

    if not year or str(year).lower() == "all":
        if os.path.exists(tenant_export_root):
            # ดึงชื่อโฟลเดอร์ย่อยที่เป็นตัวเลข (ปีงบประมาณ) ทั้งหมด
            search_years = [d for d in os.listdir(tenant_export_root) if d.isdigit()]
        else:
            search_years = []
    else:
        search_years = [str(year)]

    # 3. เริ่มสแกนไฟล์ JSON ตามรายการปีที่เจอ
    for y in search_years:
        export_root = get_tenant_year_export_root(tenant, y)
        
        if not os.path.exists(export_root):
            continue

        for root, _, files in os.walk(export_root):
            for f in files:
                if f.lower().endswith(".json"):
                    try:
                        file_path = os.path.join(root, f)
                        with open(file_path, "r", encoding="utf-8") as jf:
                            data = json.load(jf)
                            summary = data.get("summary", {})
                            enabler = (summary.get("enabler") or "KM").upper()
                            
                            # เช็คสิทธิ์ราย Enabler (ถ้าพังให้ข้ามไฟล์นี้ไป)
                            try:
                                check_user_permission(current_user, tenant, enabler)
                            except:
                                continue

                            history_list.append({
                                "record_id": data.get("record_id") or summary.get("record_id") or f.rsplit('.', 1)[0],
                                "date": parse_safe_date(summary.get("export_timestamp"), file_path),
                                "tenant": tenant,
                                "year": y,
                                "enabler": enabler,
                                "scope": summary.get("sub_criteria_id", "ALL"),
                                "level": f"L{summary.get('highest_pass_level_overall', summary.get('highest_pass_level', 0))}",
                                "score": round(float(summary.get("Total Weighted Score Achieved", summary.get("achieved_weight", 0.0))), 2),
                                "status": "COMPLETED"
                            })
                    except Exception as e:
                        logger.error(f"Error reading history file {f} in year {y}: {e}")

    # 4. เรียงลำดับตามวันที่ (ใหม่ไปเก่า)
    return {"items": sorted(history_list, key=lambda x: x['date'], reverse=True)}

@assessment_router.post("/start")
async def start_assessment(
    request: StartAssessmentRequest, 
    background_tasks: BackgroundTasks, 
    current_user: UserMe = Depends(get_current_user)
):
    """
    Endpoint สำหรับเริ่มการประเมินที่รองรับการเลือกปีอย่างอิสระ
    - บังคับใช้ปีจาก Request เป็นอันดับแรก
    - ระบบตรวจสอบ Path แบบยืดหยุ่น (รองรับ Docker/Local Path)
    """
    # 1. จัดเตรียมค่า Parameter
    enabler_uc = request.enabler.upper()
    
    # --- ปรับปรุง Logic การเลือกปี (Priority: Request > User Profile > Default) ---
    raw_year = request.year if request.year else (current_user.year or DEFAULT_YEAR)
    target_year = str(raw_year).strip()
    
    target_sub = str(request.sub_criteria).strip().lower() if request.sub_criteria else "all"

    # 2. ตรวจสอบสิทธิ์
    check_user_permission(current_user, request.tenant, enabler_uc)

    # --- [ERROR DETECTION: Enhanced Pre-flight Check] ---
    from utils.path_utils import get_vectorstore_collection_path, get_vectorstore_tenant_root_path

    # หา Path ที่ระบบคาดหวัง
    vs_path = get_vectorstore_collection_path(
        tenant=request.tenant,
        year=target_year,
        doc_type="evidence",
        enabler=enabler_uc
    )

    # 🛡️ FIX: ตรวจสอบความยืดหยุ่นของ Path (กรณีรันบน Server ที่ Path อาจต่างจากใน Container)
    resolved_vs_path = vs_path
    if not os.path.exists(resolved_vs_path) and vs_path.startswith("/app/"):
        # ลองหาแบบตัด /app/ ออก (Local mode)
        alt_path = vs_path.replace("/app/", "", 1)
        if os.path.exists(alt_path):
            resolved_vs_path = alt_path

    # A. ตรวจสอบว่าโฟลเดอร์ปีนั้นๆ มีอยู่จริงไหม
    if not os.path.exists(resolved_vs_path):
        vs_tenant_root = get_vectorstore_tenant_root_path(request.tenant)
        # ลองสแกนหา Path จริงเพื่อแนะนำ User
        real_root = vs_tenant_root.replace("/app/", "", 1) if not os.path.exists(vs_tenant_root) else vs_tenant_root
        
        available_info = ""
        if os.path.exists(real_root):
            years = [d for d in os.listdir(real_root) if os.path.isdir(os.path.join(real_root, d))]
            if years:
                available_info = f" ปีที่มีข้อมูลในระบบคือ: {', '.join(years)}"
            else:
                available_info = " ระบบยังไม่มีข้อมูลปีใดๆ ในฐานข้อมูล"
        
        logger.error(f"❌ Path Not Found: {vs_path} (Resolved: {resolved_vs_path})")
        raise HTTPException(
            status_code=400, 
            detail=f"ไม่พบฐานข้อมูล {enabler_uc} ของปี {target_year}.{available_info}"
        )

    # B. ตรวจสอบไฟล์ข้างใน (ป้องกันโฟลเดอร์ว่าง)
    # เช็คทั้ง chroma.sqlite3 หรือโฟลเดอร์ UUID ของ Chroma
    db_file = os.path.join(resolved_vs_path, "chroma.sqlite3")
    has_subdirs = any(os.path.isdir(os.path.join(resolved_vs_path, d)) for d in os.listdir(resolved_vs_path)) if os.path.exists(resolved_vs_path) else False
    
    if not os.path.exists(db_file) and not has_subdirs:
        raise HTTPException(
            status_code=400, 
            detail=f"ฐานข้อมูลปี {target_year} ยังไม่ได้ถูก Ingest ข้อมูล (โฟลเดอร์ว่างเปล่า)"
        )

    # --------------------------------------------------------

    # 3. สร้าง Record ID
    record_id = uuid.uuid4().hex[:12]
    
    # 4. บันทึกลง ACTIVE_TASKS
    ACTIVE_TASKS[record_id] = {
        "status": "RUNNING",
        "record_id": record_id,
        "tenant": request.tenant,
        "year": target_year,
        "enabler": enabler_uc,
        "progress_message": f"กำลังเริ่มการประเมิน {enabler_uc} ปี {target_year}..."
    }

    # 5. ส่งเข้า Background Task
    background_tasks.add_task(
        run_assessment_engine_task,
        record_id=record_id,
        tenant=request.tenant,
        year=target_year,
        enabler=enabler_uc,
        sub_id=target_sub,
        sequential=request.sequential_mode
    )

    logger.info(f"🚀 Started Assessment: {record_id} | Year: {target_year} | Path: {resolved_vs_path}")
    return {"record_id": record_id, "status": "RUNNING"}

async def run_assessment_engine_task(
    record_id: str, 
    tenant: str, 
    year: str,  # แก้ Type Hint เป็น str
    enabler: str, 
    sub_id: str, 
    sequential: bool
):
    try:
        str_year = year # ใช้ง่ายๆ ไม่ต้องแปลง
        logger.info(f"🚀 [TASK START] Record: {record_id} | Enabler: {enabler} | Sub-ID: {sub_id} | Year: {str_year}")

        # 1. Load Vectorstores (ใช้ str_year เข้าไปหา Path)
        vsm = await asyncio.to_thread(
            load_all_vectorstores,
            doc_types=EVIDENCE_DOC_TYPES,
            enabler_filter=enabler,
            tenant=tenant,
            year=str_year
        )
        
        # 2. Load Document Mapping
        doc_map_raw = await asyncio.to_thread(
            load_doc_id_mapping, 
            EVIDENCE_DOC_TYPES, 
            tenant, 
            str_year, 
            enabler
        )
        doc_map = {d_id: d.get("file_name", d_id) for d_id, d in doc_map_raw.items()}

        # 3. Create LLM & Engine
        llm = await asyncio.to_thread(create_llm_instance, model_name=DEFAULT_LLM_MODEL_NAME, temperature=0.0)
        config = AssessmentConfig(enabler=enabler, tenant=tenant, year=str_year, force_sequential=sequential)

        engine = SEAMPDCAEngine(
            config=config,
            llm_instance=llm,
            logger_instance=logger,
            doc_type=EVIDENCE_DOC_TYPES,
            vectorstore_manager=vsm,
            document_map=doc_map
        )

        # 4. Execution
        result = await asyncio.to_thread(
            engine.run_assessment, 
            target_sub_id=sub_id, 
            export=True, 
            vectorstore_manager=vsm, 
            sequential=sequential, 
            record_id=record_id,
            document_map=doc_map
        )

        if isinstance(result, dict) and result.get("status") == "FAILED":
            error_msg = result.get("error_message", "Engine reported an error")
            logger.error(f"❌ [TASK FAILED] {record_id}: {error_msg}")
            if record_id in ACTIVE_TASKS:
                ACTIVE_TASKS[record_id]["status"] = "FAILED"
                ACTIVE_TASKS[record_id]["error_message"] = error_msg
            return

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
