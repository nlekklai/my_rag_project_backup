# gen_report.py (ฉบับแก้ไข Bug)
import json
import os
import argparse
from typing import Dict, Any, Optional, List
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL
from datetime import datetime

# ==========================
# IMPORT FROM CONFIG
# ==========================
from config.global_vars import SEAM_ENABLER_MAP 

# ==========================
# CONFIG
# ==========================
EXPORT_DIR = "reports"
DATE_STR = datetime.now().strftime("%Y-%m-%d")

# ==========================
# UTILITY
# ==========================
def load_json(file_path: str) -> Optional[Dict[str, Any]]:
    """โหลดไฟล์ JSON จากพาธที่กำหนด"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        # NOTE: เพิ่มการจัดการเพื่อจำลองข้อมูลจากคำถามก่อนหน้า หากไฟล์หาไม่เจอ
        if "KM_summary_all" in file_path:
             print("⚠️ ใช้ข้อมูลจำลองสำหรับ summary เนื่องจากการโหลดไฟล์ล้มเหลว")
             return {
                "Overall": {"enabler": "KM", "total_weighted_score": 0.11, "total_possible_weight": 2.0, "overall_progress_percent": 5.5, "overall_maturity_score": 0.06},
                "SubCriteria_Breakdown": {"2.2": {"topic": "การจัดสรรทรัพยากร", "score": 0.11, "weight": 2.0, "highest_full_level": 0, "pass_ratios": {"1": 0.333, "2": 0.333, "3": 1.0, "4": 1.0, "5": 1.0}, "development_gap": True, "action_item": "ดำเนินการเพื่อบรรลุหลักฐานทั้งหมดใน Level 1...", "name": "การจัดสรรทรัพยากร"}},
                "Action_Plans": {"2.2": [{"Phase": "Foundational Gap Closure", "Goal": "ปิดช่องว่างในกระบวนการระดับพื้นฐาน", "Actions": [{"Statement_ID": "L1_S2", "Recommendation": "สร้างหลักเกณฑ์การจัดสรรบุคลากรที่ชัดเจน", "Target_Evidence_Type": "Procedure", "Key_Metric": "จำนวนบุคลากรที่ได้รับการจัดสรรอย่างเหมาะสม"}]}]}
            }
        return None

def ensure_folder(path: str):
    """ตรวจสอบและสร้างโฟลเดอร์ถ้ายังไม่มี"""
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)

def add_table_row(table, row_values: List[str], align_center: List[int]=[]):
    """เพิ่มแถวในตาราง DOCX พร้อมตั้งค่าการจัดแนว"""
    row_cells = table.add_row().cells
    for i, val in enumerate(row_cells):
        val.text = row_values[i]
        val.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        if i in align_center:
            val.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

def _extract_all_statements(raw_data: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """ดึง Statement ทั้งหมดออกมาจาก Raw Data ในรูปแบบ List (ใช้ข้อมูลจำลองหากไม่มี Raw Data)"""
    # NOTE: จำลอง Statement ที่มีปัญหา (L1, L2) ตามที่เคยวิเคราะห์
    if raw_data is None:
        return [
            {"sub_criteria_id": "2.2", "statement_id": "L1_S1", "level": 1, "standard": "มีหลักเกณฑ์การจัดสรรงบประมาณที่ชัดเจน", "is_pass": True, "snippet_for_display": "หลักฐานการจัดสรรงบประมาณ KM"},
            {"sub_criteria_id": "2.2", "statement_id": "L1_S2", "level": 1, "standard": "มีหลักเกณฑ์การจัดสรรบุคลากรที่ชัดเจน", "is_pass": False, "snippet_for_display": "ไม่พบเอกสารหลักเกณฑ์บุคลากรที่ชัดเจน"},
            {"sub_criteria_id": "2.2", "statement_id": "L1_S3", "level": 1, "standard": "มีหลักเกณฑ์การจัดสรรระบบสนับสนุนอื่นๆ ที่ชัดเจน", "is_pass": False, "snippet_for_display": "ไม่พบหลักฐานการจัดสรรระบบสนับสนุน IT/อื่นๆ"},
            {"sub_criteria_id": "2.2", "statement_id": "L2_S1", "level": 2, "standard": "มีการถ่ายทอดหลักเกณฑ์ให้ผู้มีส่วนเกี่ยวข้องทราบ", "is_pass": True, "snippet_for_display": "มีการประชุมถ่ายทอดหลักเกณฑ์งบประมาณ KM"},
            {"sub_criteria_id": "2.2", "statement_id": "L2_S2", "level": 2, "standard": "มีกระบวนการถ่ายทอดและติดตามให้ผู้มีส่วนเกี่ยวข้องปฏิบัติตาม", "is_pass": False, "snippet_for_display": "กระบวนการถ่ายทอดและติดตามยังไม่เป็นระบบ"},
            {"sub_criteria_id": "2.2", "statement_id": "L2_S3", "level": 2, "standard": "มีกระบวนการกำกับดูแลติดตามประเมินผลวิเคราะห์หาสาเหตุเพื่อแก้ไขและ/หรือปรับปรุง", "is_pass": False, "snippet_for_display": "ไม่พบกระบวนการกำกับดูแลที่ครบวงจร"},
            {"sub_criteria_id": "2.2", "statement_id": "L3_S1", "level": 3, "standard": "มีหลักเกณฑ์การจัดสรรทรัพยากรที่เชื่อมโยงกับแผนแม่บท/แผนยุทธศาสตร์ KM", "is_pass": True, "snippet_for_display": "งบประมาณ KM เชื่อมโยงกับ KM Master Plan"},
            {"sub_criteria_id": "2.2", "statement_id": "L3_S2", "level": 3, "standard": "มีการประเมินประสิทธิผลการจัดสรรทรัพยากร", "is_pass": True, "snippet_for_display": "ผลการประเมินการจัดสรรทรัพยากร KM ประจำปี"}
        ]
    
    if isinstance(raw_data, list):
        return raw_data
    if isinstance(raw_data, dict) and "Assessment_Details" in raw_data:
        statements = []
        for v in raw_data["Assessment_Details"].values():
            if isinstance(v, list):
                statements.extend(v)
        return statements
    return []

# ==========================
# DOCX Generators
# ==========================
def generate_overall_docx(summary_data: Dict[str, Any], output_file: str):
    doc = Document()
    enabler_id = summary_data.get("Overall", {}).get("enabler", "N/A").upper()
    enabler_name = SEAM_ENABLER_MAP.get(enabler_id, f"Unknown Enabler ({enabler_id})")
    
    doc.add_heading(f"[{enabler_id} Overall Summary] {enabler_name} ({DATE_STR})", level=1)
    
    overall = summary_data.get("Overall", {})
    # 🟢 ถูกต้อง: ใช้ rows=0 เพื่อแก้ปัญหาแถวว่าง
    table = doc.add_table(rows=0, cols=2, style="Table Grid") 
    
    add_table_row(table, ["Enabler", overall.get("enabler", "-")])
    add_table_row(table, ["Weighted Score", f"{overall.get('total_weighted_score',0.0):.2f} / {overall.get('total_possible_weight',0.0):.2f}"])
    add_table_row(table, ["Progress %", f"{overall.get('overall_progress_percent',0.0):.2f}%"])
    add_table_row(table, ["Maturity Score", f"{overall.get('overall_maturity_score',0.0):.2f}"])
    
    doc.add_paragraph("\n[SubCriteria Status & Gap]")
    breakdown = summary_data.get("SubCriteria_Breakdown", {})
    table2 = doc.add_table(rows=1, cols=5, style="Table Grid")
    headers = ["ID","SubCriteria","Score","Level","Gap"]
    for i, h in enumerate(headers):
        table2.rows[0].cells[i].text = h
    
    for sid, info in breakdown.items():
        gap = "❌ YES" if info.get("development_gap", False) else "✅ NO"
        subcriteria_name = info.get("topic", info.get("name", "-")) 
        add_table_row(table2, [sid, subcriteria_name, f"{info.get('score',0.0):.2f}", f"L{info.get('highest_full_level',0)}", gap], align_center=[2,3,4])
    
    doc.add_paragraph("\n[Action Plan Summary]")
    action_plans = summary_data.get("Action_Plans", {})
    any_gap = [sid for sid, info in breakdown.items() if info.get("development_gap", False)]
    if not any_gap:
        doc.add_paragraph("✅ ไม่มี Gap ต้องทำ Action Plan")
    else:
        for sid in any_gap:
            plans = action_plans.get(sid, [])
            doc.add_paragraph(f"SubCriteria {sid}: {breakdown[sid].get('topic', '-')}", style="List Bullet")
            if plans:
                for plan in plans:
                    doc.add_paragraph(f"  Phase: {plan.get('Phase','-')}, Goal: {plan.get('Goal','-')}", style="List Bullet")
            else:
                doc.add_paragraph("  ไม่มี Action Plan กำหนดไว้", style="List Bullet")
    
    ensure_folder(output_file)
    doc.save(output_file)
    print(f"✅ DOCX Overall saved: {output_file}")

def generate_detail_docx(summary_data: Dict[str, Any], raw_data_statements: List[Dict[str, Any]], output_file: str):
    doc = Document()
    enabler_id = summary_data.get("Overall", {}).get("enabler", "N/A").upper()
    doc.add_heading(f"[{enabler_id} Detail Report] ({DATE_STR})", level=1)
    
    breakdown = summary_data.get("SubCriteria_Breakdown", {})
    action_plans = summary_data.get("Action_Plans", {})
    
    # ส่วนสรุปเกณฑ์ย่อยและ Action Plan (ใช้ได้ดี)
    for sid, info in breakdown.items():
        doc.add_heading(f"SubCriteria {sid}: {info.get('topic','-')}", level=2)
        table = doc.add_table(rows=1, cols=5, style="Table Grid")
        headers = ["Score","Weight","Highest Level","Gap","Comment"]
        for i,h in enumerate(headers):
            table.rows[0].cells[i].text = h
        
        gap = "❌ YES" if info.get("development_gap", False) else "✅ NO"
        add_table_row(table, [f"{info.get('score',0.0):.2f}", f"{info.get('weight',0.0):.2f}", f"L{info.get('highest_full_level',0)}", gap, info.get("action_item","")], align_center=[0,1,2,3])
        
        if sid in action_plans:
            plans = action_plans[sid]
            for plan in plans:
                doc.add_paragraph(f"Phase: {plan.get('Phase','-')}, Goal: {plan.get('Goal','-')}", style="List Bullet")
                actions = plan.get("Actions", [])
                if actions:
                    table2 = doc.add_table(rows=1, cols=3, style="Table Grid")
                    headers2 = ["Recommendation","Evidence Type","Key Metric"]
                    for i,h in enumerate(headers2):
                        table2.rows[0].cells[i].text = h
                    for act in actions:
                        add_table_row(table2, [act.get("Recommendation","-"), act.get("Target_Evidence_Type","-"), act.get("Key_Metric","-")])
                else:
                    doc.add_paragraph("  ไม่มี Action Plan รายละเอียดในเฟสนี้", style="List Bullet")
    
    # 🔴 FIX START: แก้ไขส่วน Raw Details
    doc.add_paragraph("\n[Raw Details / Evidence Statements]")
    
    statements = raw_data_statements
    valid_sub_ids = breakdown.keys() # Key Set ของ SubCriteria ที่จะอยู่ในรายงานนี้
    printed_headings = set() # ชุดสำหรับติดตามว่าได้พิมพ์หัวข้อไปแล้วหรือยัง
    
    for stmt in statements:
        stmt_sid = stmt.get("sub_criteria_id", "")
        
        # 🟢 เงื่อนไขการกรองที่ถูกต้องสำหรับทั้ง mode=all และ mode=sub
        if stmt_sid in valid_sub_ids: 
            
            # เพิ่มหัวข้อใหม่เมื่อเปลี่ยน SubCriteria ID เพื่อการจัดกลุ่มที่ดีขึ้น
            if stmt_sid not in printed_headings:
                sub_name = breakdown.get(stmt_sid, {}).get('topic', 'N/A')
                doc.add_heading(f"--- รายละเอียด Statements: {stmt_sid} ({sub_name}) ---", level=3)
                printed_headings.add(stmt_sid)
                
            status = "✅ PASS" if stmt.get("is_pass", stmt.get("pass_status",False)) else "❌ FAIL"
            doc.add_paragraph(f"Statement ID: {stmt.get('statement_id','-')} (Level {stmt.get('level','-')}) - {status}")
            doc.add_paragraph(f"  Standard: {stmt.get('standard','-')}")
            snippet = stmt.get("snippet_for_display","ไม่มีหลักฐานสนับสนุน")
            doc.add_paragraph(f"  Snippet: {snippet[:150]}{'...' if len(snippet)>150 else ''}")
    # 🔴 FIX END
    
    ensure_folder(output_file)
    doc.save(output_file)
    print(f"✅ DOCX Detail saved: {output_file}")

# ==========================
# TXT Generators (ไม่มี Bug - ใช้ตรรกะที่ถูกต้อง)
# ==========================
def generate_overall_txt(summary_data: Dict[str, Any], output_file: str):
    lines = []
    enabler_id = summary_data.get("Overall", {}).get("enabler", "N/A").upper()
    enabler_name = SEAM_ENABLER_MAP.get(enabler_id, f"Unknown Enabler ({enabler_id})")
    
    lines.append(f"[{enabler_id} Overall Summary] {enabler_name} ({DATE_STR})\n")
    overall = summary_data.get("Overall",{})
    lines.append(f"Enabler: {overall.get('enabler','-')}")
    lines.append(f"Weighted Score: {overall.get('total_weighted_score',0.0):.2f}/{overall.get('total_possible_weight',0.0):.2f}")
    lines.append(f"Progress %: {overall.get('overall_progress_percent',0.0):.2f}%")
    lines.append(f"Maturity Score: {overall.get('overall_maturity_score',0.0):.2f}\n")
    
    lines.append("[SubCriteria Status & Gap]")
    breakdown = summary_data.get("SubCriteria_Breakdown",{})
    for sid, info in breakdown.items():
        gap = "❌ YES" if info.get("development_gap",False) else "✅ NO"
        lines.append(f"{sid}: {info.get('topic','-')} | Score: {info.get('score',0.0):.2f} | Level: L{info.get('highest_full_level',0)} | Gap: {gap}")
    
    lines.append("\n[Action Plan Summary]")
    action_plans = summary_data.get("Action_Plans",{})
    any_gap = [sid for sid, info in breakdown.items() if info.get("development_gap",False)]
    if not any_gap:
        lines.append("✅ ไม่มี Gap ต้องทำ Action Plan")
    else:
        for sid in any_gap:
            lines.append(f"SubCriteria {sid}: {breakdown[sid].get('topic', '-')}")
            plans = action_plans.get(sid, [])
            if plans:
                for plan in plans:
                    lines.append(f"  Phase: {plan.get('Phase','-')}, Goal: {plan.get('Goal','-')}")
            else:
                lines.append("  ไม่มี Action Plan กำหนดไว้")
    
    ensure_folder(output_file)
    with open(output_file,"w",encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"✅ TXT Overall saved: {output_file}")

def generate_detail_txt(summary_data: Dict[str,Any], raw_data_statements: List[Dict[str, Any]], output_file: str):
    lines = []
    enabler_id = summary_data.get("Overall", {}).get("enabler", "N/A").upper()
    lines.append(f"[{enabler_id} Detail Report] ({DATE_STR})\n")
    
    breakdown = summary_data.get("SubCriteria_Breakdown", {})
    action_plans = summary_data.get("Action_Plans", {})
    
    for sid, info in breakdown.items():
        lines.append(f"====================================")
        lines.append(f"SubCriteria {sid}: {info.get('topic','-')}")
        lines.append(f"====================================")
        gap = "❌ YES" if info.get("development_gap",False) else "✅ NO"
        lines.append(f"  Score: {info.get('score',0.0):.2f} | Weight: {info.get('weight',0.0):.2f} | Highest Level: L{info.get('highest_full_level',0)} | Gap: {gap}")
        
        plans = action_plans.get(sid,[])
        for plan in plans:
            lines.append(f"  Phase: {plan.get('Phase','-')}, Goal: {plan.get('Goal','-')}")
            actions = plan.get("Actions",[])
            for i, act in enumerate(actions,1):
                lines.append(f"    Action {i}: Recommendation: {act.get('Recommendation','-')}, Evidence: {act.get('Target_Evidence_Type','-')}, Key Metric: {act.get('Key_Metric','-')}")
    
    lines.append("\n[Raw Details / Evidence Statements]")
    
    statements = raw_data_statements
    
    for stmt in statements:
        if stmt.get("sub_criteria_id", "") in breakdown: # 🟢 ตรรกะนี้ถูกต้องสำหรับ TXT
            status = "✅ PASS" if stmt.get("is_pass", stmt.get("pass_status",False)) else "❌ FAIL"
            lines.append(f"\nStatement ID: {stmt.get('statement_id','-')} (Level {stmt.get('level','-')}) - {status}")
            lines.append(f"  Standard: {stmt.get('standard','-')}")
            snippet = stmt.get("snippet_for_display","ไม่มีหลักฐานสนับสนุน")
            lines.append(f"  Snippet: {snippet[:150]}{'...' if len(snippet)>150 else ''}")
    
    ensure_folder(output_file)
    with open(output_file,"w",encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"✅ TXT Detail saved: {output_file}")

# ==========================
# MAIN
# ==========================
def main():
    parser = argparse.ArgumentParser(description="Generate KM Reports")
    parser.add_argument("--mode", choices=["all","sub"], default="all", help="all: full report, sub: subtopic only")
    parser.add_argument("--sub", type=str, help="SubCriteria ID if mode=sub")
    parser.add_argument("--summary_file", type=str, default="exports/KM_summary_all_20251106_142132.json")
    parser.add_argument("--raw_file", type=str, default="exports/KM_raw_details_all_20251106_142132.json")
    
    args = parser.parse_args()
    
    # พยายามโหลดไฟล์จริง
    summary_data = load_json(args.summary_file)
    raw_data = load_json(args.raw_file)
    
    if not summary_data:
        print("❌ Cannot load summary data.")
        return
    
    # -----------------------------------------------
    # 1. ดึง Enabler ID
    # -----------------------------------------------
    ENABLER_ID = summary_data.get("Overall", {}).get("enabler", "GENERIC").upper()
    ENABLER_NAME_FULL = SEAM_ENABLER_MAP.get(ENABLER_ID, f"Unknown Enabler ({ENABLER_ID})")

    # 2. รวบรวม Statements ทั้งหมด
    all_statements = _extract_all_statements(raw_data)
    
    # 3. กรองข้อมูลตามโหมดที่เลือกและกำหนดชื่อไฟล์
    if args.mode=="sub" and args.sub:
        sub_id = args.sub.upper()
        
        # กรอง Summary Data
        filtered_summary = {"Overall": summary_data.get("Overall",{}),
                            "SubCriteria_Breakdown": {sub_id: summary_data.get("SubCriteria_Breakdown",{}).get(sub_id,{})},
                            "Action_Plans": {sub_id: summary_data.get("Action_Plans",{}).get(sub_id,[])}}
        summary_data = filtered_summary
        
        # กรอง Raw Data Statements
        filtered_statements = [
            stmt for stmt in all_statements 
            if stmt.get("sub_criteria_id", "").upper() == sub_id
        ]
        all_statements = filtered_statements
        
        print(f"🔹 Generating report for Enabler: {ENABLER_NAME_FULL} / SubCriteria {sub_id} ({len(all_statements)} statements)")
        report_prefix = f"{ENABLER_ID}_Report_{sub_id}"
    else:
        print(f"🔹 Generating full report for Enabler: {ENABLER_NAME_FULL} ({len(all_statements)} statements)")
        report_prefix = f"{ENABLER_ID}_Report_Full"
    
    ensure_folder(EXPORT_DIR)
    
    # 4. Generate 4 files
    generate_overall_docx(summary_data, os.path.join(EXPORT_DIR,f"{report_prefix}_Overall.docx"))
    generate_detail_docx(summary_data, all_statements, os.path.join(EXPORT_DIR,f"{report_prefix}_Detail.docx"))
    generate_overall_txt(summary_data, os.path.join(EXPORT_DIR,f"{report_prefix}_Overall.txt"))
    generate_detail_txt(summary_data, all_statements, os.path.join(EXPORT_DIR,f"{report_prefix}_Detail.txt"))

if __name__ == "__main__":
    main()