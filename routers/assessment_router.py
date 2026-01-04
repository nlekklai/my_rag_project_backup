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
    เวอร์ชันแก้ไขสมบูรณ์:
    - นิยาม enabler_name และ overall_level ให้ถูกต้อง
    - คำนวณ pdca_coverage และ avg_confidence_per_level ให้ UI นำไปกาง Accordion ได้
    - จัดโครงสร้าง Roadmap (Actions/Steps) ให้ตรงตาม UI
    """
    summary = raw_data.get("summary", {})
    sub_results = raw_data.get("sub_criteria_results", [])

    processed_sub_criteria = []
    radar_data = []

    # --- 1. สกัดข้อมูล Overall (แก้ไข enabler_name และ overall_level) ---
    enabler_name = (summary.get("enabler") or "N/A").upper()
    overall_level = summary.get("Overall Maturity Level (Weighted)") or f"L{summary.get('highest_pass_level_overall', 0)}"
    
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
        
        # --- 2. PDCA Matrix & Coverage Calculation ---
        pdca_matrix = []
        pdca_coverage = {} 
        avg_conf_per_lv = {}
        raw_levels_map = {item.get("level"): item for item in raw_levels_list}
        
        for lv_idx in range(1, 6):
            lv_info = raw_levels_map.get(lv_idx)
            is_passed = lv_info.get("is_passed", False) if lv_info else (lv_idx <= highest_pass)
            
            # กำหนด Mode สีให้ UI
            eval_mode = "NORMAL"
            if is_passed and lv_idx > highest_pass:
                eval_mode = "GAP_ONLY" # สีน้ำเงิน Potential
            elif not is_passed and lv_info:
                eval_mode = "FAILED" # สีเทาเข้ม (ตรวจแล้วตก)
            elif not is_passed:
                eval_mode = "INACTIVE" # สีเทาจาง (ยังไม่ประเมิน)

            pdca_raw = lv_info.get("pdca_breakdown", {}) if lv_info else {}
            pdca_final = {k: (1 if float(pdca_raw.get(k, 0)) > 0 else 0) for k in ["P", "D", "C", "A"]}
            
            # กรณีผ่านมาตรฐานไปแล้ว บังคับ PDCA เต็ม
            if not lv_info and lv_idx <= highest_pass:
                pdca_final = {"P": 1, "D": 1, "C": 1, "A": 1}

            pdca_matrix.append({
                "level": lv_idx,
                "is_passed": is_passed,
                "evaluation_mode": eval_mode,
                "pdca": pdca_final,
                "reason": lv_info.get("reason") or ("ผ่านเกณฑ์มาตรฐาน" if lv_idx <= highest_pass else "ยังไม่ถึงเกณฑ์ประเมิน")
            })

            # คำนวณ % สำหรับ Progress Bar ในแต่ละเลเวล
            covered_count = sum(pdca_final.values())
            pdca_coverage[str(lv_idx)] = {"percentage": (covered_count / 4) * 100}

        # --- 3. Sources & Confidence per Level ---
        grouped_sources = {str(lv): [] for lv in range(1, 6)}
        all_scores = []
        
        for lv_idx in range(1, 6):
            lv_scores = []
            lv_refs = [r for r in raw_levels_list if r.get("level") == lv_idx]
            for ref in lv_refs:
                sources = ref.get("temp_map_for_level", []) or [ref]
                for s in sources:
                    meta = s.get('metadata', {})
                    d_uuid = s.get('document_uuid') or meta.get('doc_id')
                    if not d_uuid: continue
                    
                    raw_s = meta.get("rerank_score") or s.get("rerank_score") or 0.0
                    score_val = 0.895 if float(raw_s) >= 1.0 else float(raw_s)
                    if score_val > 0: 
                        all_scores.append(score_val)
                        lv_scores.append(score_val)

                    grouped_sources[str(lv_idx)].append({
                        "filename": s.get('filename') or meta.get('filename') or "Evidence Document",
                        "page": str(s.get('page_number') or meta.get('page') or "1"),
                        "text": s.get("text", "")[:300],
                        "rerank_score": round(score_val * 100, 1), # ส่ง % ให้ UI
                        "document_uuid": d_uuid,
                        "pdca_tag": str(s.get("pdca_tag") or meta.get("pdca_tag", "N/A")).upper()
                    })
            
            # เฉลี่ยความมั่นใจรายเลเวล
            avg_conf_per_lv[str(lv_idx)] = (sum(lv_scores)/len(lv_scores)*100) if lv_scores else 0

        # --- 4. Roadmap Structure ---
        ui_roadmap = []
        raw_plans = res.get("action_plan") or []
        for p in raw_plans:
            phase_actions = []
            current_actions = p.get("actions") or p.get("Actions") or []
            for act in current_actions:
                phase_actions.append({
                    "level": str(act.get("level") or act.get("failed_level") or (highest_pass + 1)),
                    "recommendation": act.get("recommendation") or act.get("Recommendation") or "",
                    "steps": act.get("steps") or act.get("Steps") or []
                })
            ui_roadmap.append({
                "phase": p.get("phase") or p.get("Phase") or "แผนงานพัฒนา",
                "actions": phase_actions
            })

        # --- 5. Final Sub-Criteria Logic ---
        # หาเลเวลสูงสุดที่ "ตรวจพบข้อมูล" (แม้จะไม่ผ่านเป็นทางการ)
        potential_levels = [r.get('level') for r in raw_levels_list if r.get('is_passed')]
        potential_level = max(potential_levels + [highest_pass])

        processed_sub_criteria.append({
            "code": cid,
            "name": cname,
            "level": f"L{highest_pass}",
            "potential_level": f"L{potential_level}",
            "is_gap_analysis": potential_level > highest_pass,
            "pdca_matrix": pdca_matrix,
            "pdca_coverage": pdca_coverage,
            "avg_confidence_per_level": avg_conf_per_lv,
            "roadmap": ui_roadmap,
            "grouped_sources": grouped_sources,
            "summary_thai": (res.get("summary_thai") or "").strip(),
            "gap": (res.get("gap_analysis") or "ไม่พบช่องว่างในการพัฒนา").strip(),
            "confidence_score": round((sum(all_scores)/len(all_scores)*100) if all_scores else 0, 1)
        })
        radar_data.append({"axis": cid, "value": highest_pass})

    return {
        "status": "COMPLETED",
        "record_id": raw_data.get("record_id", "unknown"),
        "tenant": str(summary.get("tenant", "PEA")).upper(),
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

def create_docx_report_similar_to_ui(ui_data: dict) -> Document:
    doc = Document()

    # --- ตั้งค่าหน้ากระดาษ ---
    section = doc.sections[0]
    section.top_margin = Inches(0.8)
    section.bottom_margin = Inches(0.8)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)

    # --- ฟังก์ชันช่วยตั้งฟอนต์ภาษาไทยให้ถูกต้อง ---
    def set_thai_font(run, name='TH Sarabun New', size=14, bold=False, color=None):
        run.font.name = name
        run._element.rPr.rFonts.set(qn('w:eastAsia'), name)
        run.font.size = Pt(size)
        run.bold = bold
        if color:
            run.font.color.rgb = color

    # --- สไตล์หัวข้อหลัก ---
    if 'Report Title' not in doc.styles:
        title_style = doc.styles.add_style('Report Title', WD_STYLE_TYPE.PARAGRAPH)
        title_style.font.name = 'TH Sarabun New'
        title_style._element.rPr.rFonts.set(qn('w:eastAsia'), 'TH Sarabun New')
        title_style.font.size = Pt(28)
        title_style.font.bold = True
        title_style.font.color.rgb = RGBColor(30, 58, 138)
        title_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        title_style.paragraph_format.space_after = Pt(30)

    # --- หัวรายงานหลัก ---
    title_p = doc.add_paragraph(f"{ui_data['enabler']} ASSESSMENT REPORT", style='Report Title')

    # --- สรุปภาพรวม (แบบตารางสวย ๆ) ---
    summary_table = doc.add_table(rows=5, cols=2)
    summary_table.style = 'Table Grid'
    summary_table.autofit = False
    summary_table.columns[0].width = Inches(2.5)
    summary_table.columns[1].width = Inches(4.0)

    summary_data = [
        ("Record ID", ui_data['record_id']),
        ("หน่วยงาน", ui_data['tenant']),
        ("ปีงบประมาณ", ui_data['year']),
        ("ระดับความสามารถโดยรวม", ui_data['level']),
        ("คะแนนรวม / คะแนนเต็ม", f"{ui_data['score']} / {ui_data['full_score']}"),
        ("ความครบถ้วนของเกณฑ์", f"{ui_data['metrics']['completion_rate']:.1f}%")
    ]

    for label, value in summary_data:
        row = summary_table.add_row().cells
        row[0].text = label
        row[1].text = value
        set_thai_font(row[0].paragraphs[0].runs[0], size=13, bold=True)
        set_thai_font(row[1].paragraphs[0].runs[0], size=13)

    doc.add_page_break()

    sub_criteria = ui_data['sub_criteria']

    # --- กรณีเป็นการประเมิน ALL sub-criteria ---
    if len(sub_criteria) > 1 or (len(sub_criteria) == 1 and sub_criteria[0]['code'] == "ALL"):
        # หน้าแรก: สรุปภาพรวมทั้งหมด
        doc.add_heading("สรุปผลการประเมินโดยรวม (ทุกเกณฑ์ย่อย)", level=1)
        set_thai_font(doc.paragraphs[-1].runs[0], size=20, bold=True, color=RGBColor(30, 58, 138))

        # ตารางสรุประดับของแต่ละเกณฑ์
        overall_table = doc.add_table(rows=1, cols=5)
        overall_table.style = 'Table Grid'
        hdr = overall_table.rows[0].cells
        headers = ['รหัสเกณฑ์', 'ชื่อเกณฑ์', 'ระดับปัจจุบัน', 'ศักยภาพ', 'คะแนน']
        for cell, text in zip(hdr, headers):
            cell.text = text
            run = cell.paragraphs[0].runs[0]
            set_thai_font(run, size=12, bold=True)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

        for item in sub_criteria:
            row = overall_table.add_row().cells
            row[0].text = item['code']
            row[1].text = item['name']
            row[2].text = item['level']
            row[3].text = item['potential_level'] if item['potential_level'] != item['level'] else "-"
            row[4].text = f"{item['score']} / {item['full_score']}"

        doc.add_page_break()

    # --- รายละเอียดแต่ละเกณฑ์ย่อย ---
    for item in sub_criteria:
        # หัวข้อเกณฑ์
        heading = doc.add_heading(f"{item['code']} {item['name']}", level=1)
        set_thai_font(heading.runs[0], size=18, bold=True, color=RGBColor(30, 58, 138))

        # ระดับ + Potential + Bottleneck
        level_text = f"ระดับปัจจุบัน: {item['level']}"
        if item['potential_level'] != item['level']:
            level_text += f" → {item['potential_level']} (มีศักยภาพสูงกว่า)"
        if item['is_gap_analysis']:
            level_text += " ⚠️ มีจุดติดขัด (Bottleneck)"

        level_p = doc.add_paragraph(level_text)
        set_thai_font(level_p.runs[0], size=14, bold=True)
        level_p.paragraph_format.space_after = Pt(15)

        # ตาราง PDCA Coverage + Confidence
        doc.add_paragraph("ความครอบคลุม PDCA และความน่าเชื่อถือของหลักฐานตามระดับ", style='Heading 3')

        table = doc.add_table(rows=1, cols=4)
        table.style = 'Table Grid'

        hdr_cells = table.rows[0].cells
        headers = ['ระดับ', 'ความครอบคลุม PDCA', 'สถานะ', 'ความน่าเชื่อถือเฉลี่ย']
        for cell, text in zip(hdr_cells, headers):
            cell.text = text
            run = cell.paragraphs[0].runs[0]
            set_thai_font(run, size=12, bold=True)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

        current_lvl = int(item['level'].replace('L', ''))
        for lvl in range(1, 6):
            cov = item['pdca_coverage'].get(lvl, {'percentage': 0})
            pct = round(cov['percentage'])
            avg_conf = item['avg_confidence_per_level'].get(lvl, 0)
            conf_pct = round(avg_conf * 100) if avg_conf > 0 else 0

            status = ""
            if lvl == current_lvl:
                status = "ระดับปัจจุบัน"
            elif lvl > current_lvl and pct > 0:
                status = "มีศักยภาพ"

            row_cells = table.add_row().cells
            row_cells[0].text = f"L{lvl}"
            row_cells[1].text = f"{pct}%"
            row_cells[2].text = status
            row_cells[3].text = f"{conf_pct}%" if avg_conf > 0 else "ไม่มีหลักฐาน"

        doc.add_paragraph()

        # สรุปจุดแข็ง
        if item.get('summary_thai'):
            doc.add_paragraph("สรุปจุดแข็งจาก AI", style='Heading 3')
            summary_p = doc.add_paragraph(item['summary_thai'])
            summary_p.paragraph_format.left_indent = Inches(0.3)

        # จุดที่ต้องพัฒนา
        if item.get('gap'):
            doc.add_paragraph("จุดที่ต้องพัฒนา (Critical Gaps)", style='Heading 3')
            gap_p = doc.add_paragraph(item['gap'])
            gap_p.paragraph_format.left_indent = Inches(0.3)

        # แผนการพัฒนา (ปรับปรุงใหม่ให้ตรงกับ Schema)
        if item.get('roadmap'):
            doc.add_paragraph("แผนการพัฒนาเชิงกลยุทธ์", style='Heading 3')
            for phase in item['roadmap']:
                phase_p = doc.add_paragraph(phase['phase'])
                set_thai_font(phase_p.runs[0], size=14, bold=True)

                if phase.get('goal'):
                    goal_p = doc.add_paragraph(f"เป้าหมาย: {phase['goal']}")
                    goal_p.paragraph_format.left_indent = Inches(0.5)

                # ✅ เปลี่ยนจาก 'tasks' เป็น 'actions'
                for act in phase.get('actions', []):
                    task_p = doc.add_paragraph(
                        f"• ระดับเป้าหมาย {act['level']}: {act['recommendation']}"
                    )
                    set_thai_font(task_p.runs[0], bold=True)

                    for step in act.get('steps', []):
                        # จัดการเบอร์ step ให้สวยงาม
                        s_idx = step.get('step', '-')
                        s_desc = step.get('description', '')
                        resp = step.get('responsible', 'หน่วยงานที่เกี่ยวข้อง')
                        
                        step_p = doc.add_paragraph(f"   {s_idx}. {s_desc} ({resp})")
                        step_p.paragraph_format.left_indent = Inches(1.0)

        # สรุปจำนวนหลักฐาน
        doc.add_paragraph("จำนวนหลักฐานที่สนับสนุน", style='Heading 3')
        total = sum(len(files) for files in item['grouped_sources'].values() if files)
        total_p = doc.add_paragraph(f"รวมทั้งหมด: {total} เอกสาร")
        set_thai_font(total_p.runs[0], bold=True)

        for lv, files in item['grouped_sources'].items():
            if files:
                doc.add_paragraph(f"• Level {lv}: {len(files)} เอกสาร")

        # เว้นหน้าหลังแต่ละเกณฑ์
        if item != sub_criteria[-1]:  # ไม่เว้นหน้าหลังเกณฑ์สุดท้าย
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

        return FileResponse(
            path=temp_path,
            filename=f"{ui_data['enabler']}_Assessment_Report_{record_id}.docx",
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            background=lambda: os.remove(temp_path)  # ✅ ถูกต้อง ไม่ต้อง import BackgroundTask
        )

    else:
        raise HTTPException(status_code=400, detail="รองรับเฉพาะ json และ word")