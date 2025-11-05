import json
import os
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from datetime import datetime

# ==========================
# CONFIG
# ==========================
EXPORT_DIR = "exports"
SUMMARY_FILE = os.path.join(EXPORT_DIR, "KM_summary_all_20251104_031342.json")
RAW_FILE = os.path.join(EXPORT_DIR, "KM_raw_details_all_20251104_031342.json")
REPORT_DATE = datetime.now().strftime("%Y-%m-%d")

# ==========================
# UTILITIES
# ==========================
def set_heading(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if level == 1 else WD_ALIGN_PARAGRAPH.LEFT

def add_paragraph(doc, text, bold=False, italic=False, color=None):
    run = doc.add_paragraph().add_run(text)
    run.bold = bold
    run.italic = italic
    if color:
        run.font.color.rgb = RGBColor(*color)
    run.font.size = Pt(11)

def save_doc(doc, name):
    output_path = os.path.join(EXPORT_DIR, name)
    doc.save(output_path)
    print(f"✅ Created: {output_path}")

# ==========================
# REPORT 1: ภาพรวม (Overall)
# ==========================
def create_overall_report(summary):
    doc = Document()
    set_heading(doc, "รายงานภาพรวมการจัดการความรู้ (KM)")

    overall = summary.get("Overall", {})
    add_paragraph(doc, f"วันที่จัดทำรายงาน: {REPORT_DATE}")
    add_paragraph(doc, f"คะแนนรวมถ่วงน้ำหนัก: {overall.get('total_weighted_score', '-')}")
    add_paragraph(doc, f"คะแนนเต็มรวม: {overall.get('total_possible_weight', '-')}")
    add_paragraph(doc, f"ร้อยละความก้าวหน้า: {overall.get('overall_progress_percent', '-'):.2f}%")
    add_paragraph(doc, f"ระดับความเป็นผู้ใหญ่ (Maturity Score): {overall.get('overall_maturity_score', '-')}")

    doc.add_paragraph("\nรายละเอียดเกณฑ์ย่อย (Sub-Criteria):", style="List Bullet")
    for sid, sdata in summary.get("SubCriteria_Breakdown", {}).items():
        add_paragraph(doc, f"{sid} - {sdata['name']}")
        add_paragraph(doc, f"   คะแนน: {sdata['score']}/{sdata['weight']}")
        add_paragraph(doc, f"   ระดับสูงสุดที่ผ่านเต็ม: L{sdata['highest_full_level']}")
        add_paragraph(doc, f"   ช่องว่างพัฒนา: {'มี' if sdata['development_gap'] else 'ไม่มี'}")
    save_doc(doc, "KM_Report_Overall.docx")

# ==========================
# REPORT 2: GAP Analysis
# ==========================
def create_gap_report(summary):
    doc = Document()
    set_heading(doc, "รายงานช่องว่างการพัฒนา (KM Gap Analysis)")

    gaps = [s for s in summary["SubCriteria_Breakdown"].values() if s.get("development_gap")]
    if not gaps:
        add_paragraph(doc, "ไม่พบช่องว่างการพัฒนา ✅")
    else:
        for s in gaps:
            set_heading(doc, f"{s['name']}", level=2)
            add_paragraph(doc, f"คะแนน: {s['score']} / {s['weight']}")
            add_paragraph(doc, f"Action Item: {s.get('action_item', '-')}")
            if "evidence_summary_L5" in s:
                add_paragraph(doc, "สรุปหลักฐานระดับ L5:", bold=True)
                add_paragraph(doc, s["evidence_summary_L5"].get("summary", ""))
                add_paragraph(doc, "ข้อเสนอแนะ:", bold=True)
                add_paragraph(doc, s["evidence_summary_L5"].get("suggestion_for_next_level", ""))
            if "evidence_summary_L4" in s:
                add_paragraph(doc, "สรุปหลักฐานระดับ L4:", bold=True)
                add_paragraph(doc, s["evidence_summary_L4"].get("summary", ""))
    save_doc(doc, "KM_Report_Gap.docx")

# ==========================
# REPORT 3: หลักฐานรายข้อ (Evidence Detail)
# ==========================
def create_evidence_detail_report(raw):
    doc = Document()
    set_heading(doc, "รายงานหลักฐานรายข้อ (KM Evidence Details)")

    for item in raw:
        sid = item.get("statement_id")
        add_paragraph(doc, f"รหัสข้อ: {sid}", bold=True)
        add_paragraph(doc, f"หัวข้อ: {item.get('statement')}")
        add_paragraph(doc, f"ระดับ: {item.get('level')}")
        add_paragraph(doc, f"ผลการประเมิน: {'ผ่าน ✅' if item.get('pass_status') else 'ไม่ผ่าน ❌'}")
        add_paragraph(doc, f"เหตุผล: {item.get('reason')}")
        if item.get("retrieved_sources_list"):
            add_paragraph(doc, "แหล่งหลักฐานที่ใช้:")
            for src in item["retrieved_sources_list"]:
                add_paragraph(doc, f" - {src['source_name']} (p.{src.get('location')})")
        doc.add_paragraph("-" * 50)
    save_doc(doc, "KM_Report_EvidenceDetails.docx")

# ==========================
# REPORT 4: Executive Summary
# ==========================
def create_executive_summary(summary):
    doc = Document()
    set_heading(doc, "รายงานสรุปสำหรับผู้บริหาร (Executive Summary)")

    overall = summary.get("Overall", {})
    add_paragraph(doc, f"คะแนนรวม: {overall.get('total_weighted_score', '-')}/{overall.get('total_possible_weight', '-')}")
    add_paragraph(doc, f"ร้อยละความสำเร็จ: {overall.get('overall_progress_percent', 0):.2f}%")
    add_paragraph(doc, f"ระดับความเป็นผู้ใหญ่: {overall.get('overall_maturity_score', '-')}")
    add_paragraph(doc, "")

    add_paragraph(doc, "✅ จุดแข็งที่โดดเด่น (Top Strengths):", bold=True)
    top_strengths = sorted(summary["SubCriteria_Breakdown"].values(), key=lambda x: -x["score"])[:3]
    for s in top_strengths:
        add_paragraph(doc, f"- {s['name']} ({s['score']}/{s['weight']})")

    add_paragraph(doc, "")
    add_paragraph(doc, "⚠️ จุดที่ควรพัฒนา (Development Areas):", bold=True)
    gaps = [s for s in summary["SubCriteria_Breakdown"].values() if s.get("development_gap")]
    for s in gaps[:3]:
        add_paragraph(doc, f"- {s['name']} (ระดับสูงสุดผ่าน: {s['highest_full_level']})")

    save_doc(doc, "KM_Report_ExecutiveSummary.docx")

# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    os.makedirs(EXPORT_DIR, exist_ok=True)

    with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
        summary_data = json.load(f)
    with open(RAW_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    create_overall_report(summary_data)
    create_gap_report(summary_data)
    create_evidence_detail_report(raw_data)
    create_executive_summary(summary_data)

    print("\n🎉 รายงานทั้งหมดถูกสร้างเสร็จสมบูรณ์ในโฟลเดอร์ exports/ แล้วครับ!")
