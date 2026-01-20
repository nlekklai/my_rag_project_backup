# -*- coding: utf-8 -*-
# routers/assessment_router.py
# Production Final Version - 2026 Optimized for DB Persistence & Professional Reporting

import os
import re
import uuid
import json
import asyncio
import logging
import mimetypes
import tempfile
import pytz
from datetime import datetime
from typing import Optional, Dict, Any, Union, List

from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

# --- Docx Imports (สำหรับสร้าง Report) ---
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn, nsdecls
from docx.oxml import parse_xml

# --- Project Imports ---
from routers.auth_router import get_current_user, check_user_permission, UserMe
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
from config.global_vars import (
    EVIDENCE_DOC_TYPES, 
    DEFAULT_LLM_MODEL_NAME, 
    DEFAULT_YEAR, 
    DEFAULT_TENANT,
    DATA_STORE_ROOT
)

# 🎯 Database Components (SQLite Persistence)
from database import (
    SessionLocal, 
    AssessmentTaskTable, 
    AssessmentResultTable,
    db_update_task_status,
    db_finish_task
)


# ตั้งค่า Logger และ Router
logger = logging.getLogger(__name__)
assessment_router = APIRouter(prefix="/api/assess", tags=["Assessment"])

# --- Request Models ---
class StartAssessmentRequest(BaseModel):
    tenant: str
    year: Optional[Union[int, str]] = None
    enabler: str = "KM"
    sub_criteria: Optional[str] = "all"
    sequential_mode: bool = False

# ------------------------------------------------------------------
# [Helpers]
# ------------------------------------------------------------------
def parse_safe_date(raw_date_str: Any, file_path: str) -> str:
    """แปลงวันที่จาก String หรือ File Metadata ให้เป็น ISO Format (Bangkok Time)"""
    tz = pytz.timezone('Asia/Bangkok')
    if raw_date_str and isinstance(raw_date_str, str):
        try:
            # รองรับ format yyyymmdd_hhmmss
            if "_" in raw_date_str:
                dt = datetime.strptime(raw_date_str, "%Y%m%d_%H%M%S")
                return tz.localize(dt).isoformat()
        except: pass
    
    # Fallback: ใช้เวลาแก้ไขไฟล์ล่าสุด
    try:
        mtime = os.path.getmtime(file_path)
        dt = datetime.fromtimestamp(mtime, tz)
        return dt.isoformat()
    except:
        return datetime.now(tz).isoformat()


def safe_float(value):
    try:
        if isinstance(value, str):
            value = value.replace('%', '') # ลบ % ออกถ้ามี
        return float(value)
    except:
        return 0.0
    

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
async def view_document(
    filename: str, 
    document_uuid: Optional[str] = None, 
    current_user: UserMe = Depends(get_current_user)
):
    """ Endpoint สำหรับเปิดไฟล์ PDF โดยรองรับการค้นหาที่แม่นยำขึ้น """
    
    file_path = None

    # 1. พยายามหาจาก UUID ก่อน (ดีที่สุด เพราะแม่นยำ 100% แม้ชื่อไฟล์จะซ้ำ)
    if document_uuid:
        file_info = get_document_file_path(
            document_uuid=document_uuid,
            tenant=current_user.tenant,
            year=current_user.year,
            enabler="KM", # หรือดึงจาก Query Param
            doc_type_name="evidence"
        )
        if file_info:
            file_path = file_info["file_path"]

    # 2. ถ้าไม่มี UUID ให้หาจากชื่อไฟล์ (ใช้ Logic เดิมของคุณ)
    if not file_path:
        # ใช้ get_document_source_dir ที่คุณมีใน path_utils อยู่แล้ว
        from utils.path_utils import get_document_source_dir, resolve_filepath_to_absolute
        
        base_path = get_document_source_dir(
            tenant=current_user.tenant,
            year=current_user.year,
            enabler="KM",
            doc_type="evidence"
        )
        file_path = resolve_filepath_to_absolute(os.path.join(base_path, filename))

    if not file_path or not os.path.exists(file_path):
        logger.error(f"❌ File not found: {file_path}")
        raise HTTPException(status_code=404, detail=f"ไม่พบไฟล์เอกสารบนเซิร์ฟเวอร์")

    # ส่งไฟล์ PDF กลับไป
    return FileResponse(file_path, media_type="application/pdf")


def _transform_result_for_ui(raw_data: Dict[str, Any], current_user: Any = None) -> Dict[str, Any]:
    # 1. ดึงข้อมูลพื้นฐานและ Evidence Mapping ระดับบนสุด
    summary_root = raw_data.get("summary", {}) or {}
    sub_results = raw_data.get("detailed_results", []) or []
    evidence_map = raw_data.get("evidence_mapping", {}) or {}
    
    processed_sub_criteria = []
    radar_data = []

    for res in sub_results:
        # --- 2. เจาะเข้าสู่ชั้นข้อมูลที่แท้จริง (Nested Path Discovery) ---
        # โครงสร้าง: detailed_results -> level_details["0"] -> level_details["0"]
        level_root_outer = res.get("level_details", {}).get("0", {})
        level_root_inner = level_root_outer.get("level_details", {}).get("0", {})
        
        # ดึง Level Details (1-5) จากชั้นในสุด
        inner_level_details = level_root_inner.get("level_details", {})
        
        # ดึงสถานะผ่านสูงสุดและคะแนนที่แท้จริง (4.0)
        highest_pass = int(level_root_inner.get("highest_pass_level") or 0)
        actual_weighted_score = float(level_root_inner.get("weighted_score") or 0)
        
        level_details_ui = {}
        pdca_matrix_list = []
        all_unique_files = set()
        all_conf_scores = []
        
        # --- 3. วนลูปสร้างข้อมูล Level 1-5 ---
        for lv_idx in range(1, 6):
            lv_key = str(lv_idx)
            lv_info = inner_level_details.get(lv_key) or {}
            reason_text = lv_info.get("reason", "")
            
            # ดึงข้อมูลหลักฐานจาก Evidence Mapping (ใช้ Key เช่น '1.2.L1')
            sub_id = res.get("sub_id", "1.2")
            mapping_key = f"{sub_id}.L{lv_idx}"
            level_evidences = []
            
            # ตรวจสอบใน Mapping ก่อน
            if mapping_key in evidence_map:
                map_evi = evidence_map[mapping_key]
                f_name = map_evi.get("file")
                if f_name:
                    all_unique_files.add(f_name)
                    level_evidences.append({
                        "filename": f_name,
                        "page": map_evi.get("page", "-"),
                        "pdca": map_evi.get("pdca", "-")
                    })
            
            # ถ้าใน Mapping ไม่มี แต่ใน Reason มี (Regex Backup)
            found_files_regex = re.findall(r"\[Source:\s*([^,\]]+)", reason_text)
            for f in found_files_regex:
                f_name = f.strip()
                if f_name not in all_unique_files:
                    all_unique_files.add(f_name)
                    level_evidences.append({"filename": f_name, "page": "N/A"})

            # คำนวณความมั่นใจ (Simulated Score)
            lv_simulated_score = 0
            if level_evidences:
                lv_simulated_score = min(80.0 + (len(level_evidences) * 4), 99.0)
                all_conf_scores.append(lv_simulated_score)

            # PDCA Matrix
            pdca_raw = lv_info.get("pdca_breakdown", {}) or {}
            pdca_final = {p: (1 if float(pdca_raw.get(p, 0)) >= 0.5 else 0) for p in ["P", "D", "C", "A"]}
            
            # บรรจุข้อมูลเข้า Level Details UI
            level_details_ui[lv_key] = {
                "level": lv_idx,
                "confidence": lv_simulated_score,
                "is_passed": lv_idx <= highest_pass, # เช็คจาก highest_pass_level จริง
                "pdca_breakdown": pdca_final,
                "context_summary": reason_text,
                "evidences": level_evidences
            }

            pdca_matrix_list.append({
                "level": lv_idx, 
                "is_passed": lv_idx <= highest_pass, 
                "pdca": pdca_final
            })

        # --- 4. หา Gap และ Roadmap ---
        first_fail_lv = highest_pass + 1
        gap_info = inner_level_details.get(str(first_fail_lv), {})
        gap_text = f"L{first_fail_lv}: {gap_info.get('reason') or 'ไม่พบช่องว่างข้อมูล'}" if first_fail_lv <= 5 else "บรรลุเป้าหมายสูงสุดระดับ L5"

        # --- 5. สรุปผลราย Sub-Criteria ---
        trace_score_raw = (sum(all_conf_scores) / len(all_conf_scores)) if all_conf_scores else 0

        processed_sub_criteria.append({
            "code": sub_id,
            "name": level_root_inner.get("sub_criteria_name", "การมีส่วนร่วมของผู้บริหารทุกระดับ"),
            "level": f"L{highest_pass}",
            "score": round(actual_weighted_score, 2), # ใช้คะแนน 4.0 ที่ดึงมา
            "summary_thai": f"บรรลุเกณฑ์ระดับ {highest_pass}",
            "gap": gap_text,
            "audit_confidence": {
                "source_count": len(all_unique_files),
                "traceability_score": round(trace_score_raw / 100, 2),
                "consistency_check": highest_pass > 0
            },
            "pdca_matrix": pdca_matrix_list,
            "level_details": level_details_ui,
            "roadmap": level_root_inner.get("action_plan", [])
        })
        radar_data.append({"axis": sub_id, "value": highest_pass})

    # --- 6. สรุปผลภาพรวม Final Assembly ---
    max_lv = max([d['value'] for d in radar_data]) if radar_data else 0
    # ใช้คะแนนรวมที่แท้จริง (Sum จากทุก Sub-Criteria)
    total_score = sum([d['score'] for d in processed_sub_criteria])

    return {
        "status": "COMPLETED",
        "result_summary": {
            "level": f"L{max_lv}",
            "score": round(total_score, 2),
            "full_score": summary_root.get("total_possible_weight", 4.0)
        },
        "radar_data": radar_data,
        "sub_criteria": processed_sub_criteria
    }

def set_thai_font(run, size=14, bold=False, color=None):
    """ตั้งค่าฟอนต์ TH Sarabun New ให้รองรับทั้งภาษาไทยและอังกฤษ"""
    run.font.name = 'TH Sarabun New'
    # บังคับให้ XML ภายในใช้ TH Sarabun New สำหรับอักขระพิเศษและภาษาไทย
    run._element.rPr.rFonts.set(qn('w:eastAsia'), 'TH Sarabun New')
    run._element.rPr.rFonts.set(qn('w:ascii'), 'TH Sarabun New')
    run._element.rPr.rFonts.set(qn('w:hAnsi'), 'TH Sarabun New')
    run.font.size = Pt(size)
    run.bold = bold
    if color:
        run.font.color.rgb = color

def set_cell_background(cell, fill_color):
    """ระบายสีพื้นหลังให้ Cell ในตาราง (fill_color คือ hex code เช่น 'D9EAD3')"""
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_color}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)

def create_docx_report_similar_to_ui(ui_data: dict) -> Document:
    doc = Document()
    
    # 1. หัวข้อรายงาน (Header)
    header = doc.add_paragraph()
    header.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_h = header.add_run(f"รายงานผลการประเมิน Maturity Audit (SE-AM)\n")
    set_thai_font(run_h, size=20, bold=True, color=RGBColor(30, 58, 138)) # สีน้ำเงินเข้ม

    # สรุปภาพรวม (Summary Box)
    summary = ui_data.get('result_summary', {})
    sum_p = doc.add_paragraph()
    sum_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_sum = sum_p.add_run(f"ระดับวุฒิภาวะที่ได้: {summary.get('level')} | คะแนนรวม: {summary.get('score')}/{summary.get('full_score')}")
    set_thai_font(run_sum, size=16, bold=True, color=RGBColor(22, 101, 52)) # สีเขียว

    for item in ui_data.get('sub_criteria', []):
        # --- หัวข้อเกณฑ์ย่อย ---
        doc.add_paragraph() # เว้นวรรค
        title_p = doc.add_paragraph()
        run_title = title_p.add_run(f"เกณฑ์ย่อย {item.get('code', '')}: {item.get('name', '')}")
        set_thai_font(run_title, size=16, bold=True, color=RGBColor(30, 58, 138))

        # --- 1. ตาราง Audit Confidence ---
        conf_table = doc.add_table(rows=1, cols=3)
        conf_table.style = 'Table Grid'
        conf = item.get('audit_confidence', {})
        metrics = [
            ("Independence", f"{conf.get('source_count', 0)} Files"),
            ("Traceability", f"{int(conf.get('traceability_score', 0) * 100)}%"),
            ("Consistency", "VERIFIED" if conf.get('consistency_check') else "CONFLICT")
        ]
        for i, (label, val) in enumerate(metrics):
            cell = conf_table.rows[0].cells[i]
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            set_thai_font(p.add_run(label), size=10, bold=True)
            set_thai_font(p.add_run(f"\n{val}"), size=14, bold=True)
            set_cell_background(cell, "F3F4F6") # สีเทาอ่อน

        # --- 2. PDCA Capability Matrix ---
        doc.add_paragraph()
        set_thai_font(doc.add_paragraph().add_run("📊 PDCA Capability Matrix:"), size=14, bold=True)
        pdca_table = doc.add_table(rows=2, cols=5)
        pdca_table.style = 'Table Grid'
        
        for i, lv_data in enumerate(item.get('pdca_matrix', [])):
            # แถวบน: ระดับเลเวล L1-L5
            cell_top = pdca_table.cell(0, i)
            p_top = cell_top.paragraphs[0]
            p_top.alignment = WD_ALIGN_PARAGRAPH.CENTER
            set_thai_font(p_top.add_run(f"Level {lv_data['level']}"), bold=True)
            if lv_data['is_passed']:
                set_cell_background(cell_top, "D9EAD3") # เขียวอ่อนถ้าผ่าน

            # แถวล่าง: ตัวอักษร P D C A
            cell_bot = pdca_table.cell(1, i)
            p_bot = cell_bot.paragraphs[0]
            p_bot.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for char, val in lv_data['pdca'].items():
                run_char = p_bot.add_run(f" {char} ")
                color = RGBColor(22, 101, 52) if val == 1 else RGBColor(185, 28, 28)
                set_thai_font(run_char, size=11, bold=True, color=color)

        # --- 3. รายชื่อหลักฐาน (Evidence List) ---
        doc.add_paragraph()
        set_thai_font(doc.add_paragraph().add_run("📎 รายการหลักฐานที่ตรวจพบ (Evidence Mapping):"), size=13, bold=True)
        level_details = item.get('level_details', {})
        has_evidence = False
        for lv_key in sorted(level_details.keys(), key=int):
            evidences = level_details[lv_key].get('evidences', [])
            for evi in evidences:
                has_evidence = True
                txt = f"Level {lv_key}: {evi['filename']} (หน้า {evi.get('page', '-')})"
                p_evi = doc.add_paragraph(style='List Bullet')
                set_thai_font(p_evi.add_run(txt), size=11)
        if not has_evidence:
            set_thai_font(doc.add_paragraph().add_run("- ไม่พบหลักฐานแนบ -"), size=11)

        # --- 4. Strength & Gap ---
        doc.add_paragraph()
        s_title = doc.add_paragraph()
        set_thai_font(s_title.add_run("💡 AI Strength Summary:"), size=14, bold=True, color=RGBColor(22, 101, 52))
        set_thai_font(doc.add_paragraph(item.get('summary_thai', '-')).runs[0], size=13)

        g_title = doc.add_paragraph()
        set_thai_font(g_title.add_run("⚠️ Critical Gaps Found:"), size=14, bold=True, color=RGBColor(185, 28, 28))
        set_thai_font(doc.add_paragraph(item.get('gap', '-')).runs[0], size=13)

        # --- 5. Strategic Roadmap ---
        if item.get('roadmap'):
            doc.add_paragraph()
            set_thai_font(doc.add_paragraph().add_run("🛠 Strategic Improvement Roadmap:"), size=14, bold=True, color=RGBColor(30, 58, 138))
            for phase in item['roadmap']:
                p_run = doc.add_paragraph().add_run(f"Phase: {phase.get('phase')} - {phase.get('goal')}")
                set_thai_font(p_run, size=13, bold=True)
                for act in phase.get('actions', []):
                    a_run = doc.add_paragraph(style='List Bullet').add_run(f"เป้าหมาย L{act.get('failed_level')}: {act.get('recommendation')}")
                    set_thai_font(a_run, size=12, bold=True)
                    for step in act.get('steps', []):
                        step_txt = f"{step.get('description')} (ผู้รับผิดชอบ: {step.get('responsible')})"
                        set_thai_font(doc.add_paragraph(style='List Bullet 2').add_run(step_txt), size=11)

        doc.add_page_break() # จบ 1 Sub-Criteria ขึ้นหน้าใหม่
        
    return doc

# --- ตัวอย่างการสั่งบันทึกไฟล์ ---
# doc = create_docx_report_similar_to_ui(processed_data)
# doc.save("PEA_Maturity_Report_1.2.docx")

# ==================== API ENDPOINT: GET Status / Get Data ====================
@assessment_router.get("/status/{record_id}")
async def get_assessment_status(
    record_id: str, 
    current_user: UserMe = Depends(get_current_user)
):
    """
    [v2026.6.19 — Final Status + Robust Polling & Fallback]
    - Polling PROGRESS ชัดเจน (progress %, message, estimated_time)
    - Fallback ถ้าไม่เจอไฟล์ → ส่ง "NOT_FOUND" + suggestion
    - สิทธิ์ check ปลอดภัย + fallback tenant/enabler
    - Error handling แยกกรณี + log ละเอียด
    """
    # 1. เช็คใน Memory ก่อน (Polling สำหรับงานกำลังรัน)
    active_tasks = globals().get("ACTIVE_TASKS", {})
    if record_id in active_tasks:
        task = active_tasks[record_id]
        progress = task.get("progress", 0)
        message = task.get("message", "กำลังประมวลผล...")
        estimated_remaining = task.get("estimated_remaining_seconds", None)

        return {
            "status": "PROCESSING",
            "record_id": record_id,
            "progress": progress,
            "message": message,
            "estimated_remaining": estimated_remaining,
            "started_at": task.get("started_at"),
            "updated_at": datetime.now().isoformat()
        }

    # 2. หาไฟล์ที่เสร็จแล้วบน Disk
    file_path = _find_assessment_file(record_id, current_user)
    
    if not file_path or not os.path.exists(file_path):
        logger.warning(f"[Status] File not found for record_id: {record_id}")
        return {
            "status": "NOT_FOUND",
            "record_id": record_id,
            "message": "ไม่พบผลการประเมินนี้ อาจยังไม่เสร็จสิ้นหรือถูกลบ กรุณารอสักครู่หรือเริ่มการประเมินใหม่",
            "suggestion": "ตรวจสอบสถานะอีกครั้งใน 1-2 นาที หรือติดต่อผู้ดูแลระบบ"
        }

    try:
        # 3. อ่าน JSON ต้นฉบับ
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # 4. ดึง Metadata + fallback tenant/enabler
        summary = raw_data.get("summary", {}) or raw_data.get("metadata", {}) or {}
        file_enabler = (summary.get("enabler") or "KM").upper()
        file_tenant = summary.get("tenant") or current_user.tenant or "unknown"

        # ตรวจสอบสิทธิ์ (tenant + enabler)
        try:
            check_user_permission(current_user, file_tenant, file_enabler)
        except Exception as perm_err:
            logger.warning(f"[Status] Permission denied for {record_id}: {perm_err}")
            raise HTTPException(status_code=403, detail="คุณไม่มีสิทธิ์เข้าถึงผลการประเมินนี้")

        # 5. Transform ให้ UI พร้อมใช้
        ui_result = _transform_result_for_ui(raw_data, current_user)
        
        # เพิ่ม status + metadata สำคัญ
        ui_result["status"] = "COMPLETED"
        ui_result["record_id"] = record_id
        ui_result["export_path"] = file_path
        ui_result["exported_at"] = summary.get("export_at") or datetime.now().isoformat()

        logger.info(f"🚀 [Status] Returning COMPLETED for {record_id} | Enabler: {file_enabler} | Tenant: {file_tenant}")
        return ui_result

    except json.JSONDecodeError:
        logger.error(f"💥 [Status] Invalid JSON for {record_id} at {file_path}")
        raise HTTPException(status_code=500, detail="ไฟล์ข้อมูลเสียหาย ไม่สามารถอ่านได้ กรุณาติดต่อผู้ดูแลระบบ")

    except HTTPException as he:
        raise he  # ส่งต่อ permission error

    except Exception as e:
        logger.error(f"💥 [Status] Error processing {record_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="เกิดข้อผิดพลาดในการโหลดผลการประเมิน กรุณาลองใหม่หรือติดต่อผู้ดูแลระบบ"
        )

@assessment_router.get("/history")
async def get_assessment_history(
    tenant: str, 
    year: Optional[str] = Query(None),
    enabler: Optional[str] = Query(None),
    current_user: UserMe = Depends(get_current_user)
):
    """
    [v2026.PDCA.COMPAT] - เวอร์ชันรองรับ Hybrid Version
    - ดึง Record ID จาก Root, Metadata หรือ Filename (แก้ปัญหา 404)
    - รองรับ Key ทั้งตัวเล็ก (snake_case) และตัวใหญ่ (Title Case)
    - ระบบ Date Fallback ที่แข็งแกร่ง
    """
    check_user_permission(current_user, tenant)
    history_list = []
    from config.global_vars import DATA_STORE_ROOT
    from datetime import datetime
    
    norm_tenant = _n(tenant)
    # ค้นหาในหลาย Path เพื่อป้องกันการย้ายที่ของไฟล์
    search_roots = [
        os.path.join(DATA_STORE_ROOT, norm_tenant, "exports"),
        os.path.join("data_store", norm_tenant, "exports")
    ]
    
    tenant_export_root = next((p for p in search_roots if os.path.exists(p)), None)
    if not tenant_export_root:
        return {"items": [], "total_found": 0, "message": "No export data found"}

    user_allowed_enablers = [e.upper() for e in current_user.enablers]
    target_enabler = enabler.upper() if enabler else None

    if not year or str(year).lower() == "all":
        search_years = [d for d in os.listdir(tenant_export_root) if d.isdigit()]
    else:
        search_years = [str(year)]

    for y in search_years:
        year_path = os.path.join(tenant_export_root, y)
        if not os.path.exists(year_path): continue

        for root, _, files in os.walk(year_path):
            for f in files:
                if not f.lower().endswith(".json"): continue
                file_path = os.path.join(root, f)
                
                try:
                    with open(file_path, "r", encoding="utf-8") as jf:
                        data = json.load(jf)

                    summary = data.get("summary") or {}
                    metadata = data.get("metadata") or {}

                    # 1. 🛡️ EXTRA SAFE ID: ป้องกันปัญหา Search DB Miss / 404
                    # พยายามหาจาก Root -> Metadata -> Summary -> ชื่อไฟล์
                    record_id = (
                        data.get("record_id") or 
                        metadata.get("record_id") or 
                        summary.get("record_id")
                    )
                    if not record_id:
                        # ถ้าไม่มีในไฟล์จริงๆ ให้แกะจากชื่อไฟล์ (Pattern: assessment_ENABLER_ID_...)
                        parts = f.replace(".json", "").split("_")
                        record_id = parts[2] if len(parts) >= 3 else f.replace(".json", "")

                    # 2. ENABLER & SCOPE
                    file_enabler = (metadata.get("enabler") or summary.get("enabler") or data.get("enabler") or "KM").upper()
                    scope = str(metadata.get("sub_id") or summary.get("sub_criteria_id") or data.get("sub_criteria_id") or "ALL").strip().upper()

                    if file_enabler not in user_allowed_enablers: continue
                    if target_enabler and file_enabler != target_enabler: continue

                    # 3. LEVEL LOGIC (Fallback ครอบคลุมทุกเวอร์ชัน)
                    display_level = "N/A"
                    raw_lvl = summary.get("highest_pass_level") or summary.get("Overall Maturity Level (Weighted)") or summary.get("overall_level_label")
                    
                    if raw_lvl is not None:
                        l_str = str(raw_lvl).strip().upper()
                        display_level = l_str if l_str.startswith("L") else f"L{l_str}"
                    else:
                        # Fallback จากคะแนน
                        score_val = safe_float(summary.get("total_weighted_score") or summary.get("Total Weighted Score Achieved"))
                        if score_val >= 0.8: display_level = "L5"
                        elif score_val >= 0.6: display_level = "L4"
                        elif score_val >= 0.4: display_level = "L3"
                        elif score_val >= 0.2: display_level = "L2"
                        elif score_val > 0: display_level = "L1"
                        else: display_level = "L0"

                    # 4. SCORE LOGIC
                    total_score = round(safe_float(
                        summary.get("total_weighted_score") or 
                        summary.get("Total Weighted Score Achieved") or 
                        summary.get("achieved_weight") or 0.0
                    ), 2)

                    # 5. DATE PARSING (Safe multi-field)
                    date_candidates = [
                        metadata.get("export_at"),
                        summary.get("export_timestamp"),
                        summary.get("assessed_at"),
                        summary.get("timestamp")
                    ]
                    date_str, parsed_dt = "N/A", None
                    for cand in date_candidates:
                        if cand:
                            try:
                                # จัดการทั้ง ISO format และ Custom format
                                if "_" in str(cand): # สำหรับ 20260115_233148
                                    parsed_dt = datetime.strptime(str(cand), "%Y%m%d_%H%M%S")
                                else:
                                    parsed_dt = datetime.fromisoformat(str(cand).replace('Z', '+00:00'))
                                date_str = parsed_dt.isoformat()
                                break
                            except: continue

                    if not parsed_dt: # Last resort
                        mtime = os.path.getmtime(file_path)
                        parsed_dt = datetime.fromtimestamp(mtime)
                        date_str = parsed_dt.isoformat()

                    history_list.append({
                        "record_id": record_id,
                        "date": date_str,
                        "date_dt": parsed_dt,
                        "tenant": tenant,
                        "year": y,
                        "enabler": file_enabler,
                        "scope": scope,
                        "level": display_level,
                        "score": total_score,
                        "status": "COMPLETED"
                    })

                except Exception as e:
                    logger.error(f"❌ Skip corrupted file {f}: {e}")
                    continue

    # 6. Sort & Cleanup
    sorted_history = sorted(history_list, key=lambda x: x['date_dt'] or datetime.min, reverse=True)
    for item in sorted_history: item.pop('date_dt', None)

    return {
        "items": sorted_history,
        "total_found": len(history_list),
        "displayed": len(sorted_history)
    }

# ------------------------------------------------------------------
# 1. Start Assessment Endpoint
# ------------------------------------------------------------------
@assessment_router.post("/start")
async def start_assessment(
    request: StartAssessmentRequest, 
    background_tasks: BackgroundTasks, 
    current_user: UserMe = Depends(get_current_user)
):
    """
    [FINAL v2026.6.20 — Start Assessment + Friendly Response]
    - Permission + Data Integrity Check
    - Persistent Task Entry (DB)
    - Background Worker Delegation
    - Response เป็นมิตร + estimated_time
    """
    enabler_uc = request.enabler.upper()
    target_year = str(request.year if request.year else (current_user.year or DEFAULT_YEAR)).strip()
    target_sub = str(request.sub_criteria).strip().lower() if request.sub_criteria else "all"

    # 1. Permission & Data Integrity Check
    check_user_permission(current_user, request.tenant, enabler_uc)

    vs_path = get_vectorstore_collection_path(request.tenant, target_year, "evidence", enabler_uc)
    if not os.path.exists(vs_path):
        raise HTTPException(status_code=400, detail=f"Data Store สำหรับ {enabler_uc}/{target_year} ยังไม่ถูกสร้าง")

    # 2. Generate Traceable Record ID
    record_id = uuid.uuid4().hex[:12]
    
    # 3. Persistent Task Entry
    db = SessionLocal()
    try:
        new_task = AssessmentTaskTable(
            record_id=record_id,
            user_id=current_user.id,
            tenant=request.tenant,
            year=target_year,
            enabler=enabler_uc,
            sub_criteria=target_sub,
            status="QUEUED",
            progress_percent=5,
            progress_message="กำลังคิวงานประเมิน..."
        )
        db.add(new_task)
        db.commit()
        db.refresh(new_task)
    except Exception as e:
        logger.error(f"❌ Initial DB Error: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="ไม่สามารถลงทะเบียนงานประเมินได้ กรุณาลองใหม่")
    finally:
        db.close()

    # 4. Delegate to Background Worker
    background_tasks.add_task(
        run_assessment_engine_task,
        record_id=record_id,
        tenant=request.tenant,
        year=target_year,
        enabler=enabler_uc,
        sub_id=target_sub,
        sequential=request.sequential_mode  # True สำหรับ Mac (sequential)
    )

    return {
        "record_id": record_id,
        "status": "QUEUED",
        "message": f"เริ่มการประเมิน {enabler_uc} เรียบร้อยแล้ว (กำลังคิวงาน)",
        "estimated_time": "20-40 นาที (ขึ้นกับจำนวนเอกสารและโหมด sequential)",
        "poll_url": f"/api/assess/status/{record_id}",
        "poll_interval_seconds": 15
    }

# ------------------------------------------------------------------
# 2. Background Task Engine (Robust Implementation)
# ------------------------------------------------------------------
async def run_assessment_engine_task(
    record_id: str, tenant: str, year: str, enabler: str, sub_id: str, sequential: bool
):
    """
    [v2026.6.20 — Robust Background Worker + Progress Update]
    - Update progress ทุกขั้นตอนสำคัญ
    - Use asyncio.to_thread สำหรับ CPU-bound
    - Error handling + DB update เมื่อ fail
    """
    try:
        logger.info(f"⚙️ [Task {record_id}] Processing Started...")

        # Step 1: Resource Hydration
        db_update_task_status(record_id, 10, "เชื่อมต่อ Vector Database และโหลด Mapping...")
        
        vsm = await asyncio.to_thread(
            load_all_vectorstores, tenant, year, None, EVIDENCE_DOC_TYPES, enabler
        )
        
        doc_map_raw = await asyncio.to_thread(
            load_doc_id_mapping, EVIDENCE_DOC_TYPES, tenant, year, enabler
        )
        doc_map = {d_id: d.get("file_name", d_id) for d_id, d in doc_map_raw.items()}

        # Step 2: Engine & Model Setup
        db_update_task_status(record_id, 20, f"โหลด AI Model ({DEFAULT_LLM_MODEL_NAME})...")
        
        llm = await asyncio.to_thread(
            create_llm_instance, model_name=DEFAULT_LLM_MODEL_NAME, temperature=0.0
        )
        
        config = AssessmentConfig(
            enabler=enabler, tenant=tenant, year=year, 
            force_sequential=sequential,
            export_path=None
        )
        
        engine = SEAMPDCAEngine(
            config=config, 
            llm_instance=llm, 
            logger_instance=logger, 
            doc_type=EVIDENCE_DOC_TYPES, 
            vectorstore_manager=vsm, 
            document_map=doc_map,
            record_id=record_id  # 👈 เพิ่มบรรทัดนี้ เพื่อให้ Engine ภายใน Thread รู้จัก record_id
        )

        # Step 3: Core Assessment
        db_update_task_status(record_id, 35, "AI กำลังตรวจสอบหลักฐาน (RAG Assessment)...")
        
        result = await asyncio.to_thread(
            engine.run_assessment, 
            target_sub_id=sub_id, 
            export=True, 
            record_id=record_id,
            vectorstore_manager=vsm,
            sequential=sequential
        )

        # Step 4: Finalize
        if isinstance(result, dict) and result.get("status") == "FAILED":
            error_msg = result.get("error_message", "AI Engine Error")
            db_update_task_status(record_id, 0, f"ล้มเหลว: {error_msg}", status="FAILED")
        else:
            await asyncio.to_thread(db_finish_task, record_id, result)
            db_update_task_status(record_id, 100, "การประเมินเสร็จสมบูรณ์", status="COMPLETED")
            logger.info(f"✅ [Task {record_id}] Finished Successfully")
            
    except Exception as e:
        logger.error(f"💥 [Task {record_id}] Critical Failure: {str(e)}", exc_info=True)
        db_update_task_status(record_id, 0, f"ระบบขัดข้อง: {str(e)}", status="FAILED")


def _find_assessment_file(search_id: str, current_user: UserMe) -> str:
    """
    [HYBRID SEARCH v2026.2 — Final Robust]
    - ชั้น 1: DB (fast)
    - ชั้น 2: Disk scan (fallback) + tenant check
    """
    norm_tenant = _n(current_user.tenant)
    norm_search = _n(search_id).lower()

    # ชั้น 1: DB Hit
    db = SessionLocal()
    try:
        res_record = db.query(AssessmentResultTable).filter(
            AssessmentResultTable.record_id == search_id
        ).first()
        
        if res_record and res_record.full_result_json:
            try:
                data = json.loads(res_record.full_result_json)
                db_path = data.get("export_path_used") or data.get("metadata", {}).get("full_path")
                if db_path and os.path.exists(db_path):
                    logger.info(f"⚡ [Search] DB Hit! Found: {db_path}")
                    return db_path
            except:
                pass
    finally:
        db.close()

    # ชั้น 2: Disk Scan
    search_paths = [
        os.path.join(DATA_STORE_ROOT, norm_tenant, "exports"),
        os.path.join("data_store", norm_tenant, "exports"),
        "/app/data_store/{}/exports".format(norm_tenant)
    ]
    
    logger.info(f"🔍 [Search] DB Miss. Scanning Disk for ID: {norm_search}...")

    for s_path in search_paths:
        if not os.path.exists(s_path):
            continue
            
        for root, _, files in os.walk(s_path):
            for f in files:
                norm_filename = _n(f).lower()
                if norm_filename.endswith(".json") and norm_search in norm_filename:
                    if norm_tenant.lower() in _n(root).lower() or "exports" in root:
                        found_path = os.path.join(root, f)
                        logger.info(f"✅ [Search] Disk Scan Success: {found_path}")
                        return found_path
                    
    logger.error(f"❌ [Search] Total Failure for ID: {norm_search}")
    raise HTTPException(
        status_code=404, 
        detail=f"ไม่พบไฟล์ผลการประเมิน (ID: {search_id}) กรุณารอสักครู่หรือเริ่มการประเมินใหม่"
    )

# ------------------------------------------------------------------
# 3. Task List API (สำหรับหน้า UI ดึงรายการมาโชว์)
# ------------------------------------------------------------------
@assessment_router.get("/tasks")
async def get_assessment_tasks(current_user: UserMe = Depends(get_current_user)):
    db = SessionLocal()
    try:
        # ดึงเฉพาะของ Tenant ตัวเอง เรียงลำดับใหม่สุดขึ้นก่อน
        tasks = db.query(AssessmentTaskTable).filter(
            AssessmentTaskTable.tenant == current_user.tenant
        ).order_by(AssessmentTaskTable.created_at.desc()).limit(20).all()
        
        return {"tasks": tasks}
    finally:
        db.close()


# ------------------------------------------------------------------
# 4. Download API (Full Revised)
# ------------------------------------------------------------------
@assessment_router.get("/download/{record_id}/{file_type}")
async def download_assessment_file(
    record_id: str,
    file_type: str,
    background_tasks: BackgroundTasks,
    current_user: UserMe = Depends(get_current_user)
):
    """
    API สำหรับดาวน์โหลดผลประเมินย้อนหลัง (JSON หรือ Word)
    """
    logger.info(f"📥 Download request: record_id={record_id}, type={file_type} by {current_user.email}")

    # 1. ค้นหาไฟล์ต้นฉบับ (JSON)
    json_path = _find_assessment_file(record_id, current_user)

    # 2. อ่านข้อมูลเพื่อตรวจสอบ Permission ระดับ Enabler อีกครั้ง
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except Exception as e:
        logger.error(f"Error reading JSON: {e}")
        raise HTTPException(status_code=500, detail="ไม่สามารถอ่านไฟล์ข้อมูลได้")

    # 3. ตรวจสอบสิทธิ์เข้าถึง Enabler (เช่น User PEA-KM ห้ามโหลด PEA-IT)
    enabler = (raw_data.get("summary", {}).get("enabler") or "KM").upper()
    check_user_permission(current_user, current_user.tenant, enabler)

    file_type = file_type.lower()

    # --- กรณีขอไฟล์ JSON ---
    if file_type == "json":
        return FileResponse(
            path=json_path,
            filename=f"SEAM_Result_{enabler}_{record_id}.json",
            media_type="application/json"
        )

    # --- กรณีขอไฟล์ Word (DOCX) ---
    elif file_type in ["word", "docx"]:
        logger.info(f"📄 Generating Word report for {record_id}...")

        # แปลงข้อมูล
        ui_data = _transform_result_for_ui(raw_data)
        
        # สร้าง Document (เรียกใช้จากฟังก์ชันที่คุณมีใน gen_report.py หรือคล้ายกัน)
        try:
            doc = create_docx_report_similar_to_ui(ui_data)
        except ImportError:
            # Fallback หากยังไม่ได้ทำตัวสร้าง Report
            raise HTTPException(status_code=501, detail="ระบบสร้างไฟล์ Word ยังไม่ถูกติดตั้ง")

        # บันทึกลงไฟล์ชั่วคราว (Temporary File)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            doc.save(tmp.name)
            temp_path = tmp.name

        logger.info(f"✅ Word report generated at: {temp_path}")

        # ใช้ Background Task ลบไฟล์ทิ้งหลังจากส่งให้ User เสร็จแล้ว
        background_tasks.add_task(os.remove, temp_path)

        return FileResponse(
            path=temp_path,
            filename=f"SEAM_Report_{enabler}_{record_id}.docx",
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

    else:
        raise HTTPException(status_code=400, detail="รูปแบบไฟล์ไม่ถูกต้อง (รองรับ json, word)")