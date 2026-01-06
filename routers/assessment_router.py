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

import tempfile
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE

from docx.shared import Pt, RGBColor, Inches
from docx.oxml.ns import qn

from routers.auth_router import UserMe, get_current_user
from utils.path_utils import (
    _n, 
    get_tenant_year_export_root, 
    load_doc_id_mapping, 
    get_document_file_path,
    get_vectorstore_collection_path,
    get_vectorstore_tenant_root_path
    )

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
    [FULL STABLE VERSION] สำหรับส่งข้อมูลให้ React Frontend:
    - เพิ่มการส่ง audit_confidence (Independence, Traceability, Consistency)
    - ปรับปรุง PDCA Tag Priority และ Coverage Matrix
    - สรุป Roadmap และ Gap Analysis สำหรับ UI
    """
    summary = raw_data.get("summary", {})
    sub_results = raw_data.get("sub_criteria_results", [])

    processed_sub_criteria = []
    radar_data = []

    # --- 1. ข้อมูลภาพรวม (Header) ---
    enabler_name = (summary.get("enabler") or "KM").upper()
    overall_level = str(summary.get("Overall Maturity Level (Weighted)") or f"L{summary.get('highest_pass_level_overall', 0)}")
    
    total_score = round(float(summary.get("Total Weighted Score Achieved") or 0.0), 2)
    full_score_all = round(float(summary.get("Total Possible Weight") or 40.0), 2)
    total_expected = int(summary.get("total_subcriteria") or 12)
    passed_count = int(summary.get("total_subcriteria_assessed") or len(sub_results))
    completion_rate = (passed_count / total_expected * 100) if total_expected > 0 else 0.0

    for res in sub_results:
        cid = res.get("sub_criteria_id", "N/A")
        cname = res.get("sub_criteria_name", f"เกณฑ์ย่อย {cid}")
        highest_pass = int(res.get("highest_full_level") or 0)
        raw_levels_list = res.get("raw_results_ref", [])
        
        # --- 2. ดึง Audit Confidence (ค่าความมั่นใจสำหรับโชว์ 3 กล่อง) ---
        # ดึงจากระดับ Sub-criteria ตรงๆ (ที่คำนวณจาก Logic v2)
        raw_audit_conf = res.get("audit_confidence") or {}
        
        # กรณีใน Root ไม่มี ให้พยายามหาจาก Level ล่าสุดที่ประเมิน
        if not raw_audit_conf and raw_levels_list:
            raw_audit_conf = raw_levels_list[-1].get("audit_confidence") or {}

        ui_audit_confidence = {
            "level": raw_audit_conf.get("level", "LOW"),
            "source_count": int(raw_audit_conf.get("source_count", 0)),
            "traceability_score": float(raw_audit_conf.get("traceability_score", 0.0)),
            "consistency_check": bool(raw_audit_conf.get("consistency_check", True)),
            "reason": raw_audit_conf.get("reason", "")
        }

        # --- 3. PDCA Matrix & Coverage ---
        pdca_matrix = []
        pdca_coverage = {str(lv): {"percentage": 0} for lv in range(1, 6)} 
        avg_conf_per_lv = {str(lv): 0 for lv in range(1, 6)}
        raw_levels_map = {item.get("level"): item for item in raw_levels_list}
        
        for lv_idx in range(1, 6):
            lv_info = raw_levels_map.get(lv_idx)
            is_passed = lv_info.get("is_passed", False) if lv_info else (lv_idx <= highest_pass)
            
            # กำหนดสถานะการประเมิน (NORMAL, GAP_ONLY, FAILED)
            eval_mode = "NORMAL"
            if is_passed and lv_idx > highest_pass:
                eval_mode = "GAP_ONLY" 
            elif not is_passed and lv_info:
                eval_mode = "FAILED"   
            elif not is_passed:
                eval_mode = "INACTIVE" 

            # ดึงคะแนนราย PDCA (ถ้าผ่านแบบมาตรฐาน Base Level ให้เต็ม 1)
            pdca_raw = lv_info.get("pdca_breakdown", {}) if lv_info else {}
            pdca_final = {k: (1 if float(pdca_raw.get(k, 0)) > 0 else 0) for k in ["P", "D", "C", "A"]}
            
            if not lv_info and lv_idx <= highest_pass:
                pdca_final = {"P": 1, "D": 1, "C": 1, "A": 1}

            pdca_matrix.append({
                "level": lv_idx,
                "is_passed": is_passed,
                "evaluation_mode": eval_mode,
                "pdca": pdca_final,
                "reason": lv_info.get("reason") or ("ผ่านเกณฑ์มาตรฐาน" if lv_idx <= highest_pass else "ยังไม่มีข้อมูลประเมิน")
            })

            covered_count = sum(pdca_final.values())
            pdca_coverage[str(lv_idx)]["percentage"] = (covered_count / 4) * 100

        # --- 4. Evidence Sources (Grouped by Level) ---
        grouped_sources = {str(lv): [] for lv in range(1, 6)}
        all_scores = []
        
        for lv_idx in range(1, 6):
            lv_scores = []
            lv_refs = [r for r in raw_levels_list if r.get("level") == lv_idx]
            for ref in lv_refs:
                sources = ref.get("temp_map_for_level", []) or [ref]
                for s in sources:
                    meta = s.get('metadata', {})
                    d_uuid = s.get('document_uuid') or meta.get('stable_doc_uuid')
                    if not d_uuid: continue
                    
                    score_val = float(s.get("rerank_score") or meta.get("rerank_score") or 0.0)
                    if score_val > 0: 
                        all_scores.append(score_val)
                        lv_scores.append(score_val)

                    raw_pdca = s.get("pdca_tag")
                    if not raw_pdca or str(raw_pdca).upper() == "OTHER":
                        raw_pdca = meta.get("pdca_tag") or "OTHER"
                    
                    grouped_sources[str(lv_idx)].append({
                        "filename": s.get('filename') or meta.get('source') or "Evidence Document",
                        "page": str(meta.get('page') or meta.get('page_label') or "1"),
                        "text": s.get("text", "")[:300],
                        "rerank_score": round(score_val * 100, 1),
                        "document_uuid": d_uuid,
                        "pdca_tag": str(raw_pdca).upper()
                    })
            if lv_scores:
                avg_conf_per_lv[str(lv_idx)] = (sum(lv_scores)/len(lv_scores)*100)

        # --- 5. Roadmap & Gap Analysis ---
        ui_roadmap = []
        all_gaps = []
        raw_plans = res.get("action_plan") or []
        for p in raw_plans:
            phase_actions = []
            actions_list = p.get("Actions") or p.get("actions") or []
            for act in actions_list:
                rec = act.get("Recommendation") or act.get("recommendation") or "ควรดำเนินการตามเกณฑ์"
                lv_num = str(act.get("Failed_Level") or act.get("level") or (highest_pass + 1))
                all_gaps.append(f"**L{lv_num}**: {rec}")

                raw_steps = act.get("Steps") or act.get("steps") or []
                clean_steps = [s.get("Description") or s.get("Step") or str(s) if isinstance(s, dict) else str(s) for s in raw_steps]

                phase_actions.append({
                    "level": lv_num,
                    "recommendation": rec,
                    "steps": clean_steps
                })
            ui_roadmap.append({
                "phase": p.get("Phase") or p.get("phase") or "ระยะการพัฒนา",
                "actions": phase_actions
            })

        # --- 6. Sub-Criteria Summary ---
        potential_level = max([r.get('level') for r in raw_levels_list if r.get('is_passed')] + [highest_pass, 0])
        strength_summary = res.get("summary_thai") or (raw_levels_list[-1].get("reason") if raw_levels_list else "ดำเนินงานได้ตามเกณฑ์")

        processed_sub_criteria.append({
            "code": cid,
            "name": cname,
            "level": f"L{highest_pass}",
            "potential_level": f"L{potential_level}",
            "is_gap_analysis": potential_level > highest_pass,
            "pdca_matrix": pdca_matrix,
            "pdca_coverage": pdca_coverage,
            "avg_confidence_per_level": avg_conf_per_lv,
            "audit_confidence": ui_audit_confidence, # 🔥 ส่งให้ UI ตรงนี้
            "roadmap": ui_roadmap,
            "grouped_sources": grouped_sources,
            "summary_thai": strength_summary.strip(),
            "gap": "\n\n".join(all_gaps) if all_gaps else "ไม่พบช่องว่างในการพัฒนาที่เด่นชัด",
            "confidence_score": round((sum(all_scores)/len(all_scores)*100) if all_scores else 0, 1)
        })
        radar_data.append({"axis": cid, "value": highest_pass})

    return {
        "status": "COMPLETED",
        "record_id": raw_data.get("record_id", "unknown"),
        "tenant": str(summary.get("tenant", "PEA")).upper(),
        "year": str(summary.get("year", "2567")),
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


def create_docx_report_similar_to_ui(ui_data: dict) -> Document:
    doc = Document()

    # --- ตั้งค่าหน้ากระดาษ ---
    section = doc.sections[0]
    section.top_margin = Inches(0.5)
    section.bottom_margin = Inches(0.5)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)

    def set_thai_font(run, name='TH Sarabun New', size=14, bold=False, color=None):
        run.font.name = name
        run._element.rPr.rFonts.set(qn('w:eastAsia'), name)
        run.font.size = Pt(size)
        run.bold = bold
        if color:
            run.font.color.rgb = color

    # --- 1. หน้าปก / หัวรายงาน ---
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title_p.add_run(f"{ui_data.get('enabler', 'KM')} ASSESSMENT REPORT\n")
    set_thai_font(run, size=24, bold=True, color=RGBColor(30, 58, 138))
    
    # สรุปภาพรวม
    summary_table = doc.add_table(rows=0, cols=2)
    summary_table.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    summary_data = [
        ("Record ID", ui_data.get('record_id', '-')),
        ("หน่วยงาน", ui_data.get('tenant', '-')),
        ("ปีงบประมาณ", ui_data.get('year', '-')),
        ("ระดับความสามารถโดยรวม", f"L{ui_data.get('level', '0')}"),
        ("คะแนนรวม / คะแนนเต็ม", f"{ui_data.get('score', 0)} / {ui_data.get('full_score', 40)}"),
        ("ความครบถ้วน (Completion)", f"{ui_data.get('metrics', {}).get('completion_rate', 0):.1f}%")
    ]

    for label, value in summary_data:
        row = summary_table.add_row().cells
        set_thai_font(row[0].paragraphs[0].add_run(label), size=14, bold=True)
        set_thai_font(row[1].paragraphs[0].add_run(str(value)), size=14)

    doc.add_page_break()

    # --- 2. รายละเอียดรายเกณฑ์ย่อย ---
    sub_criteria = ui_data.get('sub_criteria', [])
    for item in sub_criteria:
        # หัวข้อเกณฑ์
        h = doc.add_paragraph()
        run = h.add_run(f"เกณฑ์ย่อย {item.get('code', '')}: {item.get('name', '')}")
        set_thai_font(run, size=18, bold=True, color=RGBColor(30, 58, 138))

        # --- ส่วนที่เพิ่ม: Audit Confidence Metrics (เหมือนบน UI) ---
        conf_table = doc.add_table(rows=1, cols=3)
        conf_table.style = 'Table Grid'
        cells = conf_table.rows[0].cells
        
        # กล่อง 1: Independence
        p1 = cells[0].paragraphs[0]
        p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_thai_font(p1.add_run("Independence"), size=10, bold=True)
        p1.add_run(f"\n{item.get('audit_confidence', {}).get('source_count', 0)} Files").font.size = Pt(14)
        
        # กล่อง 2: Traceability
        p2 = cells[1].paragraphs[0]
        p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_thai_font(p2.add_run("Traceability"), size=10, bold=True)
        trace_val = int(item.get('audit_confidence', {}).get('traceability_score', 0) * 100)
        p2.add_run(f"\n{trace_val}%").font.size = Pt(14)
        
        # กล่อง 3: Consistency
        p3 = cells[2].paragraphs[0]
        p3.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_thai_font(p3.add_run("Consistency"), size=10, bold=True)
        consist_txt = "VERIFIED" if item.get('audit_confidence', {}).get('consistency_check') else "CONFLICT"
        p3.add_run(f"\n{consist_txt}").font.size = Pt(14)

        doc.add_paragraph() # เว้นบรรทัด

        # สรุปผลของ AI (Strength & Gap)
        # Strength
        s_title = doc.add_paragraph()
        set_thai_font(s_title.add_run("บทสรุปจุดแข็ง (AI Strength Summary):"), size=14, bold=True, color=RGBColor(22, 101, 52))
        set_thai_font(doc.add_paragraph(item.get('summary_thai', '-')).runs[0], size=13)

        # Gap
        g_title = doc.add_paragraph()
        set_thai_font(g_title.add_run("ข้อเสนอแนะเพื่อการปรับปรุง (Critical Gaps):"), size=14, bold=True, color=RGBColor(154, 52, 18))
        set_thai_font(doc.add_paragraph(item.get('gap', 'ไม่พบข้อบกพร่องที่สำคัญ')).runs[0], size=13)

        # Roadmap (ถ้ามี)
        if item.get('roadmap'):
            r_title = doc.add_paragraph()
            set_thai_font(r_title.add_run("Roadmap การพัฒนาเชิงกลยุทธ์:"), size=14, bold=True, color=RGBColor(30, 58, 138))
            
            for phase in item['roadmap']:
                p_text = f"ระยะ: {phase.get('phase', '')}"
                phase_p = doc.add_paragraph(style='List Bullet')
                set_thai_font(phase_p.add_run(p_text), size=13, bold=True)

                for act in phase.get('actions', []):
                    # Recommendation
                    act_p = doc.add_paragraph(style='List Bullet 2')
                    set_thai_font(act_p.add_run(f"เป้าหมาย L{act.get('level')}: {act.get('recommendation')}"), size=12, bold=True)
                    
                    # Steps
                    for step in act.get('steps', []):
                        step_p = doc.add_paragraph(style='List Bullet 3')
                        set_thai_font(step_p.add_run(str(step)), size=11)

        doc.add_page_break()

    return doc
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
async def download_assessment_file(
    record_id: str,
    file_type: str,
    background_tasks: BackgroundTasks, # 1. เพิ่มตัวนี้เข้ามาใน Parameter
    current_user: UserMe = Depends(get_current_user)
):
    logger.info(f"Download request: record_id={record_id}, file_type={file_type}")

    # 1. หาไฟล์ JSON
    json_path = _find_assessment_file(record_id, current_user)

    # 2. อ่านข้อมูล
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # 3. ตรวจ permission
    enabler = (raw_data.get("summary", {}).get("enabler") or "KM").upper()
    check_user_permission(current_user, current_user.tenant, enabler)

    file_type = file_type.lower()

    # 4. JSON
    if file_type == "json":
        return FileResponse(
            path=json_path,
            filename=f"assessment-{record_id}.json",
            media_type="application/json"
        )

    # 5. Word Report
    elif file_type in ["word", "docx"]:
        logger.info(f"Generating on-the-fly Word report for {record_id}")

        ui_data = _transform_result_for_ui(raw_data)
        doc = create_docx_report_similar_to_ui(ui_data)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            doc.save(tmp.name)
            temp_path = tmp.name

        logger.info(f"Word report generated: {os.path.basename(temp_path)}")

        # 2. ใช้ background_tasks.add_task แทน lambda
        background_tasks.add_task(os.remove, temp_path)

        # 3. ส่ง FileResponse โดยเอาพารามิเตอร์ background ออก
        return FileResponse(
            path=temp_path,
            filename=f"{ui_data['enabler']}_Assessment_Report_{record_id}.docx",
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

    else:
        raise HTTPException(status_code=400, detail="รองรับเฉพาะ json และ word")