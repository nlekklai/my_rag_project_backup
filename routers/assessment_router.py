# -*- coding: utf-8 -*-
# routers/assessment_router.py
# Production Final Version - 2026 Optimized for DB Persistence & Professional Reporting

import os
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
from docx.oxml.ns import qn

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
    """
    [PRODUCTION READY - v2026.6.18 — Final UI Ready]
    - ดึง temp_map_for_level ครบ + fallback ถ้าขาด
    - grouped_sources เรียงตาม score + text snippet
    - strength_summary สมจริง + fallback
    - Roadmap Steps ครบ 4 fields + fallback
    - เพิ่ม evidences_by_level สำหรับ UI แสดงหลักฐานทุก level
    """
    summary = raw_data.get("summary", {}) or {}
    sub_results = raw_data.get("sub_criteria_results", []) or []

    processed_sub_criteria: List[Dict[str, Any]] = []
    radar_data: List[Dict[str, Any]] = []

    # --- 1. Header & Global Metrics ---
    enabler_name = (summary.get("enabler") or "KM").upper()
    overall_level = str(summary.get("Overall Maturity Level (Weighted)") or
                        f"L{summary.get('highest_pass_level_overall', 0)}")

    total_score = round(safe_float(summary.get("Total Weighted Score Achieved")), 2)
    full_score_all = round(float(summary.get("Total Possible Weight") or 40.0), 2)
    total_expected = int(summary.get("total_subcriteria") or 12)
    passed_count = int(summary.get("total_subcriteria_assessed") or len(sub_results))
    completion_rate = (passed_count / total_expected * 100) if total_expected > 0 else 0.0

    for res in sub_results:
        cid = res.get("sub_criteria_id", "N/A")
        cname = res.get("sub_criteria_name", f"เกณฑ์ย่อย {cid}")
        highest_pass = int(res.get("highest_full_level") or res.get("highest_pass_level") or 0)
        raw_levels_list = res.get("raw_results_ref", []) or []

        # --- 2. Audit Confidence ---
        raw_audit_conf = res.get("audit_confidence") or {}
        if not raw_audit_conf and raw_levels_list:
            raw_audit_conf = raw_levels_list[-1].get("audit_confidence") or {}

        ui_audit_confidence = {
            "level": raw_audit_conf.get("level", "LOW"),
            "source_count": int(raw_audit_conf.get("source_count", 0)),
            "traceability_score": float(raw_audit_conf.get("traceability_score", 0.0)),
            "consistency_check": bool(raw_audit_conf.get("consistency_check", True)),
            "reason": raw_audit_conf.get("reason", "ผ่านการตรวจสอบตามมาตรฐาน SE-AM")
        }

        # --- 3. PDCA Matrix & Coverage ---
        pdca_matrix = []
        pdca_coverage = {str(lv): {"percentage": 0} for lv in range(1, 6)}
        raw_levels_map = {item.get("level"): item for item in raw_levels_list}

        for lv_idx in range(1, 6):
            lv_info = raw_levels_map.get(lv_idx)
            is_passed = lv_info.get("is_passed", False) if lv_info else (lv_idx <= highest_pass)

            eval_mode = "NORMAL"
            if is_passed and lv_idx > highest_pass:
                eval_mode = "GAP_ONLY"
            elif not is_passed and lv_info:
                eval_mode = "FAILED"
            elif not is_passed:
                eval_mode = "INACTIVE"

            pdca_raw = lv_info.get("pdca_breakdown", {}) if lv_info else {}
            pdca_final = {k: (1 if float(pdca_raw.get(k, 0)) > 0 else 0) for k in ["P", "D", "C", "A"]}

            if not lv_info and lv_idx <= highest_pass:
                pdca_final = {"P": 1, "D": 1, "C": 1, "A": 1}

            pdca_matrix.append({
                "level": lv_idx,
                "is_passed": is_passed,
                "evaluation_mode": eval_mode,
                "pdca": pdca_final
            })

            pdca_coverage[str(lv_idx)]["percentage"] = (sum(pdca_final.values()) / 4) * 100

        # --- 4. Grouped Evidence & Confidence ---
        grouped_sources = {str(lv): [] for lv in range(1, 6)}
        all_scores = []
        avg_confidence_per_level = {}
        evidences_by_level = {}

        for lv_idx in range(1, 6):
            lv_scores = []
            lv_refs = [r for r in raw_levels_list if r.get("level") == lv_idx]
            evidences = []

            for ref in lv_refs:
                sources = ref.get("temp_map_for_level", []) or ref.get("evidence_sources", []) or [ref]
                
                for s in sources:
                    meta = s.get("metadata", {})
                    d_uuid = s.get("stable_doc_uuid") or meta.get("stable_doc_uuid") or s.get("doc_id") or "N/A"
                    if not d_uuid or d_uuid == "N/A":
                        continue

                    score_val = float(s.get("rerank_score") or meta.get("rerank_score") or s.get("score") or 0.0)
                    if score_val > 0:
                        all_scores.append(score_val)
                        lv_scores.append(score_val)

                    pdca_tag = s.get("pdca_tag") or meta.get("pdca_tag") or "OTHER"

                    evidence_item = {
                        "filename": s.get("source_filename") or meta.get("source_filename") or s.get("filename") or meta.get("source") or "Evidence Document",
                        "page": str(meta.get("page") or meta.get("page_label") or "1"),
                        "text": (s.get("text") or "")[:300] + ("..." if len(s.get("text") or "") > 300 else ""),
                        "rerank_score": round(score_val * 100, 1),
                        "document_uuid": d_uuid,
                        "pdca_tag": str(pdca_tag).upper(),
                        "doc_type": s.get("doc_type", "evidence")
                    }

                    evidences.append(evidence_item)
                    grouped_sources[str(lv_idx)].append(evidence_item)

            # เรียงตาม score
            grouped_sources[str(lv_idx)] = sorted(grouped_sources[str(lv_idx)], key=lambda x: x["rerank_score"], reverse=True)

            avg_confidence_per_level[str(lv_idx)] = round((sum(lv_scores) / len(lv_scores) * 100), 1) if lv_scores else 0.0
            evidences_by_level[str(lv_idx)] = evidences

        # --- 5. Roadmap ---
        ui_roadmap = []
        all_gaps = []
        raw_plans = res.get("action_plan") or []

        for p in raw_plans:
            phase_name = p.get("Phase") or p.get("phase") or "Phase การพัฒนา"
            goal = p.get("Goal") or p.get("goal") or "ปิดช่องว่างและยกระดับเกณฑ์"

            phase_actions = []
            actions_list = p.get("Actions") or p.get("actions") or []

            for act in actions_list:
                recommendation = act.get("Recommendation") or act.get("recommendation") or "ควรดำเนินการตามเกณฑ์"
                failed_level = str(act.get("Failed_Level") or act.get("failed_level") or (highest_pass + 1))
                all_gaps.append(f"**L{failed_level}**: {recommendation}")

                formatted_steps = []
                raw_steps = act.get("Steps") or act.get("steps") or []

                for s_idx, s in enumerate(raw_steps):
                    if isinstance(s, dict):
                        formatted_steps.append({
                            "step": s.get("Step") or s.get("step") or (s_idx + 1),
                            "description": s.get("Description") or s.get("description") or "ดำเนินการตามคำแนะนำ",
                            "responsible": s.get("Responsible") or s.get("responsible") or "คณะทำงานที่เกี่ยวข้อง",
                            "verification_outcome": s.get("Verification_Outcome") or s.get("verification_outcome") or "เอกสารหลักฐานผลการดำเนินงาน"
                        })
                    else:
                        formatted_steps.append({
                            "step": s_idx + 1,
                            "description": str(s),
                            "responsible": "คณะทำงานที่เกี่ยวข้อง",
                            "verification_outcome": "เอกสารหลักฐานผลการดำเนินงาน"
                        })

                phase_actions.append({
                    "failed_level": failed_level,
                    "recommendation": recommendation,
                    "target_evidence_type": act.get("Target_Evidence_Type") or "Report/Policy/Document",
                    "steps": formatted_steps
                })

            ui_roadmap.append({
                "phase": phase_name,
                "goal": goal,
                "actions": phase_actions
            })

        # --- 6. 🎯 AI STRENGTH SUMMARY ---
        base_reason = ui_audit_confidence["reason"].strip()

        level_num = highest_pass
        if level_num == 1:
            prefix = f"ในระดับพื้นฐาน (L{level_num}): พบหลักฐานเริ่มต้น"
            adjusted_reason = base_reason.replace("ครบถ้วนตามวงจร PDCA", "สอดคล้องกับเกณฑ์พื้นฐาน")
        elif level_num == 2:
            prefix = f"ในระดับเริ่มต้น (L{level_num}): "
            adjusted_reason = base_reason
        elif level_num == 3:
            prefix = f"ในระดับพัฒนา (L{level_num}): "
            adjusted_reason = base_reason
        else:
            prefix = f"ในระดับสูง (L{level_num}): "
            adjusted_reason = base_reason

        strength_summary = f"{prefix} {adjusted_reason}"

        content_analysis = res.get("summary_thai", "").strip()
        if content_analysis and len(content_analysis) > 20 and content_analysis not in strength_summary:
            strength_summary += f" {content_analysis}"

        if not strength_summary or len(strength_summary) < 20:
            strength_summary = f"ในระดับ L{level_num}: หลักฐานมีความน่าเชื่อถือตามเกณฑ์พื้นฐาน"

        # --- 7. Final Mapping ---
        potential_level = max(
            [r.get("level") for r in raw_levels_list if r.get("is_passed")] + [highest_pass, 0]
        )
        current_score = float(raw_levels_list[-1].get("score") or 0.0) if raw_levels_list else (highest_pass * 0.2)

        level_details_ui = {}
        for lv_idx in range(1, 6):
            level_details_ui[str(lv_idx)] = {
                "level": lv_idx,
                "is_passed": lv_idx <= highest_pass,
                "score": 0.0,
                "pdca_breakdown": pdca_matrix[lv_idx-1]["pdca"],
                "evidences": evidences_by_level.get(str(lv_idx), [])
            }

        processed_sub_criteria.append({
            "code": cid,
            "name": cname,
            "level": f"L{highest_pass}",
            "score": round(current_score, 1),
            "potential_level": f"L{potential_level}",
            "is_gap_analysis": potential_level > highest_pass,
            "pdca_matrix": pdca_matrix,
            "pdca_coverage": pdca_coverage,
            "avg_confidence_per_level": avg_confidence_per_level,
            "audit_confidence": ui_audit_confidence,
            "roadmap": ui_roadmap,
            "grouped_sources": grouped_sources,
            "summary_thai": strength_summary,
            "gap": "\n\n".join(all_gaps) if all_gaps else "บรรลุเป้าหมายตามเกณฑ์ปัจจุบัน",
            "confidence_score": round((sum(all_scores) / len(all_scores) * 100) if all_scores else 0, 1),
            "level_details": level_details_ui
        })

        radar_data.append({"axis": cid, "value": highest_pass})

    return {
        "status": "COMPLETED",
        "record_id": raw_data.get("record_id", "unknown"),
        "tenant": str(summary.get("tenant", "unknown")).upper(),
        "year": str(summary.get("year", "unknown")),
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

    # ตั้งค่าหน้ากระดาษ
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

    # 1. หน้าปก
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
        ("ระดับความสามารถโดยรวม", ui_data.get('level', 'L0')),
        ("คะแนนรวม / คะแนนเต็ม", f"{ui_data.get('score', 0)} / {ui_data.get('full_score', 40)}"),
        ("ความครบถ้วน", f"{ui_data.get('metrics', {}).get('completion_rate', 0):.1f}%")
    ]

    for label, value in summary_data:
        row = summary_table.add_row().cells
        set_thai_font(row[0].paragraphs[0].add_run(label), size=14, bold=True)
        set_thai_font(row[1].paragraphs[0].add_run(str(value)), size=14)

    doc.add_page_break()

    # 2. Radar Chart (ถ้ามี radar_data)
    if ui_data.get('radar_data'):
        r_title = doc.add_paragraph()
        set_thai_font(r_title.add_run("ภาพรวมระดับความสามารถแต่ละเกณฑ์ (Radar Chart)"), size=16, bold=True)
        # เพิ่ม placeholder สำหรับ radar (ใน docx จริงอาจใช้ python-docx + matplotlib แทรกภาพ)
        doc.add_paragraph("[Radar Chart Placeholder - แทรกภาพจาก UI]")

    # 3. รายละเอียดรายเกณฑ์
    sub_criteria = ui_data.get('sub_criteria', [])
    for item in sub_criteria:
        h = doc.add_paragraph()
        run = h.add_run(f"เกณฑ์ย่อย {item.get('code', '')}: {item.get('name', '')}")
        set_thai_font(run, size=18, bold=True, color=RGBColor(30, 58, 138))

        # Audit Confidence
        conf_table = doc.add_table(rows=1, cols=3)
        conf_table.style = 'Table Grid'
        cells = conf_table.rows[0].cells
        
        p1 = cells[0].paragraphs[0]
        p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_thai_font(p1.add_run("Independence"), size=10, bold=True)
        p1.add_run(f"\n{item.get('audit_confidence', {}).get('source_count', 0)} Files").font.size = Pt(14)
        
        p2 = cells[1].paragraphs[0]
        p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_thai_font(p2.add_run("Traceability"), size=10, bold=True)
        trace_val = int(item.get('audit_confidence', {}).get('traceability_score', 0) * 100)
        p2.add_run(f"\n{trace_val}%").font.size = Pt(14)
        
        p3 = cells[2].paragraphs[0]
        p3.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_thai_font(p3.add_run("Consistency"), size=10, bold=True)
        consist_txt = "VERIFIED" if item.get('audit_confidence', {}).get('consistency_check') else "CONFLICT"
        p3.add_run(f"\n{consist_txt}").font.size = Pt(14)

        doc.add_paragraph()

        # Strength & Gap
        s_title = doc.add_paragraph()
        set_thai_font(s_title.add_run("บทสรุปจุดแข็ง:"), size=14, bold=True, color=RGBColor(22, 101, 52))
        set_thai_font(doc.add_paragraph(item.get('summary_thai', '-')).runs[0], size=13)

        g_title = doc.add_paragraph()
        set_thai_font(g_title.add_run("ข้อเสนอแนะเพื่อการปรับปรุง:"), size=14, bold=True, color=RGBColor(154, 52, 18))
        set_thai_font(doc.add_paragraph(item.get('gap', 'ไม่พบข้อบกพร่องที่สำคัญ')).runs[0], size=13)

        # PDCA Matrix
        if item.get('pdca_matrix'):
            pdca_title = doc.add_paragraph()
            set_thai_font(pdca_title.add_run("PDCA Matrix:"), size=14, bold=True)
            pdca_table = doc.add_table(rows=1, cols=5)
            pdca_table.style = 'Table Grid'
            headers = ["Level", "Passed", "P", "D", "C", "A"]
            hdr_cells = pdca_table.add_row().cells
            for i, h in enumerate(headers):
                set_thai_font(hdr_cells[i].paragraphs[0].add_run(h), size=11, bold=True)

            for entry in item['pdca_matrix']:
                row = pdca_table.add_row().cells
                set_thai_font(row[0].paragraphs[0].add_run(str(entry['level'])), size=11)
                set_thai_font(row[1].paragraphs[0].add_run("Yes" if entry['is_passed'] else "No"), size=11)
                pdca = entry['pdca']
                for i, k in enumerate(["P", "D", "C", "A"], 2):
                    set_thai_font(row[i].paragraphs[0].add_run("✔" if pdca.get(k) else "-"), size=11)

        # Evidences (แสดง 3 อันดับแรกต่อ level)
        if item.get('level_details'):
            ev_title = doc.add_paragraph()
            set_thai_font(ev_title.add_run("หลักฐานหลัก (Top Evidences):"), size=14, bold=True)
            for lv, details in item['level_details'].items():
                evs = details.get('evidences', [])[:3]  # แสดง 3 อันดับแรก
                if evs:
                    lv_p = doc.add_paragraph()
                    set_thai_font(lv_p.add_run(f"Level {lv}:"), size=12, bold=True)
                    for ev in evs:
                        ev_p = doc.add_paragraph(style='List Bullet')
                        txt = f"{ev.get('filename', '')} หน้า {ev.get('page', '1')} | Score: {ev.get('rerank_score', 0)}% | {ev.get('text_snippet', '')}"
                        set_thai_font(ev_p.add_run(txt), size=11)

        # Roadmap
        if item.get('roadmap'):
            r_title = doc.add_paragraph()
            set_thai_font(r_title.add_run("Roadmap การพัฒนา:"), size=14, bold=True)
            for phase in item['roadmap']:
                p_text = f"ระยะ: {phase.get('phase', '')} - {phase.get('goal', '')}"
                phase_p = doc.add_paragraph(style='List Bullet')
                set_thai_font(phase_p.add_run(p_text), size=13, bold=True)

                for act in phase.get('actions', []):
                    act_p = doc.add_paragraph(style='List Bullet 2')
                    set_thai_font(act_p.add_run(f"เป้าหมาย L{act.get('failed_level')}: {act.get('recommendation')}"), size=12, bold=True)
                    
                    for step in act.get('steps', []):
                        step_p = doc.add_paragraph(style='List Bullet 3')
                        set_thai_font(step_p.add_run(str(step)), size=11)

        doc.add_page_break()

    return doc

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
    [v2026.6.17 — Ultra Safe + Complete History]
    - ดึง Level อย่างชาญฉลาด + fallback ทุก field
    - Date parsing ปลอดภัยสุด (หลาย field + mtime fallback)
    - Normalize scope/level/score/date ให้ไม่ crash
    - Sort ด้วย datetime จริง + fallback วันที่เก่า
    - Log ละเอียด + response stats (total_found, filtered, displayed)
    """
    # 1. ตรวจสอบสิทธิ์องค์กร
    check_user_permission(current_user, tenant)

    history_list = []
    from config.global_vars import DATA_STORE_ROOT
    from datetime import datetime
    
    # 2. จัดการ Path (Tenant & Exports)
    norm_tenant = _n(tenant)
    tenant_export_root = os.path.join(DATA_STORE_ROOT, norm_tenant, "exports")
    
    # Fallback สำหรับรัน Local
    if not os.path.exists(tenant_export_root):
        alt_path = os.path.join("data_store", norm_tenant, "exports")
        if os.path.exists(alt_path): 
            tenant_export_root = alt_path

    if not os.path.exists(tenant_export_root):
        logger.warning(f"⚠️ [History] ไม่พบข้อมูลของ {norm_tenant}")
        return {"items": [], "total_found": 0, "filtered_by_permission": 0, "displayed": 0}

    # 3. เตรียม Filter
    user_allowed_enablers = [e.upper() for e in current_user.enablers]
    target_enabler = enabler.upper() if enabler else None

    # 4. กำหนดช่วงปี
    if not year or str(year).lower() == "all":
        search_years = [d for d in os.listdir(tenant_export_root) if d.isdigit()]
    else:
        search_years = [str(year)]

    filtered_count = 0

    # 5. สแกนไฟล์และดึงข้อมูล
    for y in search_years:
        year_path = os.path.join(tenant_export_root, y)
        if not os.path.exists(year_path): continue

        for root, _, files in os.walk(year_path):
            for f in files:
                if f.lower().endswith(".json"):
                    file_path = os.path.join(root, f)
                    try:
                        with open(file_path, "r", encoding="utf-8") as jf:
                            data = json.load(jf)

                        # ดึง summary + metadata fallback
                        summary = data.get("summary", {}) or data.get("metadata", {}) or {}
                        file_enabler = (summary.get("enabler") or data.get("enabler") or "KM").upper()
                        scope_raw = summary.get("sub_criteria_id") or data.get("sub_criteria_id") or "ALL"
                        scope = str(scope_raw).strip().upper()

                        # 🛡️ สิทธิ์การเข้าถึง Enabler
                        if file_enabler not in user_allowed_enablers:
                            logger.debug(f"Skip: {file_path} - No permission for enabler '{file_enabler}'")
                            filtered_count += 1
                            continue

                        # 🎯 กรองตามที่ User เลือก
                        if target_enabler and file_enabler != target_enabler:
                            continue

                        # --- 🛠️ Logic การจัดการ Level (fallback ทุก field) ---
                        display_level = "N/A"
                        
                        if scope != "ALL":
                            # 1. Overall Maturity Level (Weighted)
                            raw_weighted = summary.get("Overall Maturity Level (Weighted)") or summary.get("overall_level_label")
                            if raw_weighted:
                                display_level = str(raw_weighted).strip()
                            
                            # 2. highest_pass_level
                            elif "highest_pass_level" in summary:
                                raw_highest = summary.get("highest_pass_level")
                                if raw_highest is not None:
                                    try:
                                        display_level = f"L{int(raw_highest)}"
                                    except:
                                        display_level = f"L{str(raw_highest)}"
                            
                            # 3. Fallback จาก score
                            else:
                                score_val = safe_float(summary.get("Total Weighted Score Achieved") or summary.get("total_weighted_score"))
                                if score_val >= 0.8: display_level = "L5"
                                elif score_val >= 0.6: display_level = "L4"
                                elif score_val >= 0.4: display_level = "L3"
                                elif score_val >= 0.2: display_level = "L2"
                                elif score_val > 0: display_level = "L1"
                                else: display_level = "L0"

                        # จัดการคะแนน (Score) - ปลอดภัย
                        total_score = round(safe_float(summary.get("Total Weighted Score Achieved") or summary.get("total_weighted_score")), 2)

                        # --- Date (safe parse + multi fallback) ---
                        date_candidates = [
                            summary.get("export_at"),
                            summary.get("export_timestamp"),
                            summary.get("timestamp"),
                            summary.get("assessed_at"),
                            summary.get("created_at")
                        ]
                        date_str = "N/A"
                        parsed_dt = None

                        for cand in date_candidates:
                            if cand:
                                try:
                                    parsed_dt = datetime.fromisoformat(str(cand).replace('Z', '+00:00'))
                                    date_str = parsed_dt.isoformat()
                                    break
                                except:
                                    continue

                        # Ultimate fallback: ใช้ mtime ของไฟล์
                        if date_str == "N/A":
                            try:
                                mtime = os.path.getmtime(file_path)
                                parsed_dt = datetime.fromtimestamp(mtime)
                                date_str = parsed_dt.isoformat()
                            except:
                                pass

                        history_list.append({
                            "record_id": data.get("record_id") or data.get("metadata", {}).get("record_id") or f.replace(".json", ""),
                            "date": date_str,
                            "date_dt": parsed_dt,  # สำหรับ sort (ไม่ส่งให้ client)
                            "tenant": tenant,
                            "year": y,
                            "enabler": file_enabler,
                            "scope": scope,
                            "level": display_level,
                            "score": total_score,
                            "status": "COMPLETED",
                            "file_path": file_path  # debug เท่านั้น
                        })

                    except json.JSONDecodeError as je:
                        logger.error(f"❌ JSON Error in {file_path}: {je}")
                        continue
                    except Exception as e:
                        logger.error(f"❌ Error parsing {file_path}: {e}")
                        continue

    # 6. Sort ด้วย datetime จริง (fallback วันที่เก่า)
    def parse_date_safe(item):
        dt = item.get('date_dt')
        return dt if dt else datetime.min

    sorted_history = sorted(
        history_list,
        key=parse_date_safe,
        reverse=True
    )

    # ลบ date_dt ก่อนส่ง response
    for item in sorted_history:
        item.pop('date_dt', None)
        item.pop('file_path', None)  # ไม่ส่ง path จริงให้ client

    total_found = len(history_list)
    return {
        "items": sorted_history,
        "total_found": total_found,
        "filtered_by_permission": filtered_count,
        "displayed": len(sorted_history),
        "message": f"Found {total_found} assessments, filtered {filtered_count} by permission"
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
            document_map=doc_map
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