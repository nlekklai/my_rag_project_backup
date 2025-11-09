import json
import os
import argparse
from typing import Dict, Any, Optional, List
from datetime import datetime

# Import libraries for DOCX generation
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL

# ==========================
# 1. CONFIGURATION & GLOBAL VARS
# ==========================
EXPORT_DIR = "reports"
REPORT_DATE = datetime.now().strftime("%Y-%m-%d")
THAI_FONT_NAME = "Angsana New" 

# Required Import: พยายาม Import SEAM_ENABLER_MAP จาก config/global_vars.py
try:
    from config.global_vars import SEAM_ENABLER_MAP
except ImportError:
    # Fallback/Placeholder: หากรันโค้ดนอกโครงสร้างโปรเจกต์
    print("⚠️ ไม่พบ config.global_vars. ใช้ SEAM_ENABLER_MAP จำลอง.")
    SEAM_ENABLER_MAP = {
        "KM": "7.1 การจัดการความรู้ (Knowledge Management)",
        "IT": "7.2 เทคโนโลยีดิจิทัล",
        "HR": "1.1 การบริหารทรัพยากรบุคคล",
        "GENERIC": "ตัวขับเคลื่อนทั่วไป"
    }

# ==========================
# 2. DATA LOADING & UTILITY
# ==========================

def load_data(file_path: str, file_type: str) -> Optional[Dict[str, Any]]:
    """โหลดข้อมูลจากไฟล์ JSON และจัดการข้อผิดพลาด"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"❌ ข้อผิดพลาดในการโหลดไฟล์ {file_type}: ไม่พบไฟล์ '{file_path}'") 
        return None
    except Exception as e:
        print(f"❌ ข้อผิดพลาดในการโหลดไฟล์ {file_path} '{file_path}': {e}") 
        return None

def setup_output_folder(file_path):
    """ตรวจสอบและสร้าง folder output"""
    output_dir = os.path.dirname(file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

def setup_document(doc):
    """Sets up document-wide formatting like margins and default font."""
    # 1. ตั้งค่า Margins: ลดระยะขอบซ้าย/ขวาเหลือ 0.75 นิ้ว
    section = doc.sections[0]
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(0.75) 
    section.right_margin = Inches(0.75)

    # 2. ตั้งค่า Default Font เป็น Angsana New
    doc.styles['Normal'].font.name = THAI_FONT_NAME
    
def add_paragraph(doc, text, bold=False, italic=False, color=None, style=None):
    """ฟังก์ชัน Utility สำหรับการเพิ่ม Paragraph อย่างง่าย (พร้อมกำหนด Angsana New)"""
    p = doc.add_paragraph(style=style) if style else doc.add_paragraph()
    run = p.add_run(text)
    
    run.font.name = THAI_FONT_NAME 
    
    run.bold = bold
    run.italic = italic
    if color:
        run.font.color.rgb = RGBColor(*color)
    run.font.size = Pt(11)
    return p

def set_heading(doc, text, level=1):
    """ฟังก์ชัน Utility สำหรับการเพิ่ม Heading (พร้อมกำหนด Angsana New)"""
    p = doc.add_heading(text, level=level)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if level == 1 else WD_ALIGN_PARAGRAPH.LEFT
    
    # กำหนดฟอนต์ให้กับ Heading
    for run in p.runs:
        run.font.name = THAI_FONT_NAME 

def flatten_raw_data(raw_data_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    ดึง Statement ทั้งหมดออกมาจาก Raw Data Dictionary 
    ให้อยู่ในรูปแบบ List ที่สามารถวนซ้ำได้ง่าย
    """
    statements = []
    # โครงสร้าง 1: {"Assessment_Details": {"2.2": [...], ...}}
    details = raw_data_dict.get("Assessment_Details") if isinstance(raw_data_dict, dict) else None
    if isinstance(details, dict):
        for sub_id_statements in details.values():
            if isinstance(sub_id_statements, list):
                statements.extend(sub_id_statements)
    # โครงสร้าง 2: List ของ Statements ตรงๆ
    elif isinstance(raw_data_dict, list):
        statements = raw_data_dict
        
    return statements

# ==========================
# 3. REPORT GENERATION FUNCTIONS (DOCX)
# ==========================

def generate_overall_summary_docx(document: Document, data: Dict[str, Any], enabler_name_full: str): 
    """สร้างส่วนสรุปผลการประเมินโดยรวม (Overall) [SECTION 1] ใน DOCX"""
    overall = data.get("Overall", {})
    
    set_heading(document, f'[SECTION 1] สรุปผลการประเมิน {enabler_name_full} โดยรวม', level=1)
    
    table = document.add_table(rows=4, cols=2) 
    table.style = 'Table Grid'
    
    def add_summary_row(row_index, label, value):
        table.cell(row_index, 0).text = label
        table.cell(row_index, 1).text = value
        table.cell(row_index, 0).paragraphs[0].runs[0].font.bold = True
        table.cell(row_index, 1).paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
    
    add_summary_row(0, "ตัวขับเคลื่อน (Enabler):", f"{overall.get('enabler', '-')} ({enabler_name_full})") 
    add_summary_row(1, "คะแนนรวมถ่วงน้ำหนักที่ได้:", f"{overall.get('total_weighted_score', 0.0):.2f} / {overall.get('total_possible_weight', 0.0):.2f}")
    add_summary_row(2, "เปอร์เซ็นต์ความคืบหน้าโดยรวม:", f"{overall.get('overall_progress_percent', 0.0):.2f}%")
    add_summary_row(3, "คะแนนวุฒิภาวะโดยรวม (Maturity Score):", f"{overall.get('overall_maturity_score', 0.0):.2f}")
    
    # กำหนดฟอนต์ให้ Table Headers และ Content
    for row in table.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.name = THAI_FONT_NAME
    
    document.add_paragraph() 

def generate_executive_summary_docx(document: Document, summary: Dict[str, Any]):
    """สร้างรายงานสรุปสำหรับผู้บริหาร (Executive Summary) [SECTION 2] ใน DOCX"""
    if not summary: return
    set_heading(document, "[SECTION 2] สรุปสำหรับผู้บริหาร (Executive Summary)", level=1)

    overall = summary.get("Overall", {})
    add_paragraph(document, f"✅ คะแนนรวม: {overall.get('total_weighted_score', 0):.2f} / {overall.get('total_possible_weight', 0):.2f}")
    add_paragraph(document, f"✅ ร้อยละความสำเร็จ: {overall.get('overall_progress_percent', 0):.2f}%")
    add_paragraph(document, f"✅ ระดับความเป็นผู้ใหญ่: {overall.get('overall_maturity_score', 0):.2f}")
    document.add_paragraph()

    breakdown = summary.get("SubCriteria_Breakdown", {})
    if breakdown:
        # Strength: Top 3 highest scoring
        add_paragraph(document, "📈 จุดแข็งที่โดดเด่น (Top Strengths):", bold=True, color=(0x00, 0x70, 0xC0))
        top_strengths = sorted(breakdown.values(), key=lambda x: x.get("score", 0), reverse=True)[:3]
        for s in top_strengths:
            sub_name = s.get('name', s.get('topic', 'N/A'))
            add_paragraph(document, f"• {sub_name} ({s.get('score', 0):.2f}/{s.get('weight', 0):.2f})", style="List Bullet")

        document.add_paragraph()
        
        # Weakness: Top 3 with Gap (or lowest scoring with Gap)
        add_paragraph(document, "🚨 จุดที่ควรพัฒนา (Development Areas):", bold=True, color=(0xFF, 0x00, 0x00))
        gaps = [s for s in breakdown.values() if s.get("development_gap")]
        top_weaknesses = sorted(gaps, key=lambda x: x.get("score", 0))[:3]
        for s in top_weaknesses:
            sub_name = s.get('name', s.get('topic', 'N/A'))
            add_paragraph(document, f"• {sub_name} (ระดับสูงสุดผ่าน: L{s.get('highest_full_level', 0)})", style="List Bullet")
    
    document.add_paragraph()

def generate_sub_criteria_status_docx(document: Document, data: Dict[str, Any]) -> Dict[str, Any]:
    """สร้างตารางสถานะการประเมินรายเกณฑ์ย่อย [SECTION 3] ใน DOCX และคืนค่าเกณฑ์ที่มี Gap"""
    breakdown = data.get("SubCriteria_Breakdown", {})
    
    document.add_heading('[SECTION 3] สถานะการประเมินรายเกณฑ์ย่อยและ Gap', level=1)
    
    table = document.add_table(rows=1, cols=5)
    table.style = 'Table Grid'
    
    header_cells = table.rows[0].cells
    headers = ["ID", "ชื่อเกณฑ์ย่อย", "คะแนน", "Level", "Gap"]
    for i, h in enumerate(headers):
        header_cells[i].text = h
        header_cells[i].paragraphs[0].runs[0].font.bold = True
        header_cells[i].paragraphs[0].runs[0].font.name = THAI_FONT_NAME 
        header_cells[i].vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    gap_criteria = {}
    
    for sub_id, info in breakdown.items():
        row_cells = table.add_row().cells
        
        name = info.get('name', info.get('topic', 'N/A')) 
        score = info.get('score', 0.0)
        level = info.get('highest_full_level', 0)
        has_gap = "❌ YES" if info.get('development_gap', False) else "✅ NO"
        
        if info.get('development_gap', False):
            gap_criteria[sub_id] = info # เก็บ info ทั้งหมด รวมถึง L4/L5 summary
            
        row_cells[0].text = sub_id
        row_cells[1].text = name
        row_cells[2].text = f"{score:.2f}"
        row_cells[3].text = f"L{level}"
        row_cells[4].text = has_gap
        
        # กำหนดฟอนต์สำหรับแถวข้อมูล
        for cell in row_cells:
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.name = THAI_FONT_NAME

        row_cells[0].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        row_cells[2].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        row_cells[3].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        row_cells[4].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    document.add_paragraph() 
    return gap_criteria

def generate_action_plan_report_docx(document: Document, data: Dict[str, Any], gap_criteria: Dict[str, Any]):
    """สร้างรายงานรายละเอียดแผนปฏิบัติการ (Action Plan) [SECTION 4] ใน DOCX (รวม L4/L5 Summary)"""
    action_plans = data.get("Action_Plans", {})
    
    document.add_heading('[SECTION 4] รายงานรายละเอียดแผนปฏิบัติการเพื่อปิดช่องว่าง (Action Plan)', level=1)
    
    if not gap_criteria:
        document.add_paragraph("✅ ทุกเกณฑ์ย่อยผ่านครบถ้วนแล้ว ไม่จำเป็นต้องมี Action Plan เพิ่มเติม")
        return

    for sub_id, sub_info in gap_criteria.items():
        sub_name = sub_info.get('name', sub_info.get('topic', 'N/A'))
        
        document.add_heading(f"• เกณฑ์ย่อย {sub_id}: {sub_name} (Highest Full Level: L{sub_info.get('highest_full_level', 0)})", level=2)
        
        # --- NEW FEATURE: เพิ่ม L4/L5 Summary จาก gen_gpt_report.py ---
        if "evidence_summary_L5" in sub_info:
            add_paragraph(document, "💡 ข้อมูลเชิงลึกหลักฐานระดับ L5 (เป้าหมายสูงสุด):", bold=True, color=(0x00, 0x70, 0xC0))
            add_paragraph(document, f"   - สรุปหลักฐาน: {sub_info['evidence_summary_L5'].get('summary', 'ไม่มีสรุป L5')}", italic=True)
            add_paragraph(document, f"   - ข้อเสนอแนะ: {sub_info['evidence_summary_L5'].get('suggestion_for_next_level', 'ไม่มีข้อเสนอแนะ')}", italic=True)
            document.add_paragraph()
        
        if "evidence_summary_L4" in sub_info:
            add_paragraph(document, "💡 ข้อมูลเชิงลึกหลักฐานระดับ L4:", bold=True, color=(0x00, 0x70, 0xC0))
            add_paragraph(document, f"   - สรุปหลักฐาน: {sub_info['evidence_summary_L4'].get('summary', 'ไม่มีสรุป L4')}", italic=True)
            document.add_paragraph()
        # --- END NEW FEATURE ---
        
        if sub_id in action_plans:
            
            for plan_phase in action_plans[sub_id]:
                phase = plan_phase.get('Phase', '-')
                goal = plan_phase.get('Goal', '-')
                actions_list = plan_phase.get('Actions', [])
                
                add_paragraph(document, f"🛠️ เฟส/ขั้นตอน: {phase}", style='List Bullet')
                add_paragraph(document, f"🎯 เป้าหมายหลัก: {goal}", style='List Bullet')

                if actions_list:
                    document.add_paragraph("แผนปฏิบัติการ:")
                    
                    action_table = document.add_table(rows=1, cols=3, style='Table Grid')
                    header_cells = action_table.rows[0].cells
                    header_cells[0].text = "คำแนะนำ (Recommendation)"
                    header_cells[1].text = "หลักฐานเป้าหมาย (Evidence Type)"
                    header_cells[2].text = "ตัวชี้วัดสำคัญ (Key Metric)"
                    
                    for cell in action_table.rows[0].cells:
                         cell.paragraphs[0].runs[0].font.bold = True
                         cell.paragraphs[0].runs[0].font.name = THAI_FONT_NAME 
                    
                    for action in actions_list:
                        row_cells = action_table.add_row().cells
                        row_cells[0].text = action.get('Recommendation', '-')
                        row_cells[1].text = action.get('Target_Evidence_Type', '-')
                        row_cells[2].text = action.get('Key_Metric', '-')
                        
                        # กำหนดฟอนต์สำหรับแถวข้อมูล
                        for cell in row_cells:
                            for p in cell.paragraphs:
                                for run in p.runs:
                                    run.font.name = THAI_FONT_NAME
                
                document.add_paragraph() 
        else:
            add_paragraph(document, ">>> [ข้อมูล]: ไม่มี Action Plan ถูกกำหนดไว้ในส่วน Action_Plans", style='List Bullet')

def generate_raw_details_report_docx(document: Document, raw_data: Optional[Dict[str, Any]]): 
    """สร้างรายงานการตรวจสอบความถูกต้องเชิงลึก (Raw Details) [SECTION 5] ใน DOCX (เพิ่ม Reason และ Source)"""
    
    raw_data_base = raw_data 
    if raw_data is None:
        document.add_paragraph(f"⚠️ ไม่สามารถโหลดไฟล์ Raw Details ได้ หรือไฟล์ว่างเปล่า") 
        return
        
    assessment_details = {}
    
    # ตรวจสอบว่าเป็น Dict หรือ List ก่อน
    if isinstance(raw_data_base, dict):
        # Case 1: Standard Dictionary structure
        assessment_details = raw_data.get('Assessment_Details', {})
    elif isinstance(raw_data_base, list):
        # Case 2: List of statements structure (โหมด 'sub')
        statements_list = raw_data_base
        # พยายามตั้งชื่อ sub_id จาก statement แรก (เนื่องจากโหมด sub จะมี sub เดียว)
        sub_id = statements_list[0].get('sub_criteria_id', 'N/A') if statements_list else 'N/A'
        if sub_id != 'N/A':
            assessment_details[sub_id] = statements_list
        else:
             add_paragraph(document, "ℹ️ ข้อมูล Raw Details ถูกโหลดแล้ว แต่ไม่พบ 'sub_criteria_id' ใน Statement")
             return
    else:
        add_paragraph(document, f"⚠️ โครงสร้างข้อมูล Raw Details ไม่ถูกต้อง (ไม่ใช่ Dict หรือ List)") 
        return

    if not assessment_details:
         add_paragraph(document, "ℹ️ ข้อมูล Raw Details ว่างเปล่าหลังจากตรวจสอบโครงสร้าง")
         return
    
    # โค้ดแสดงผลรายละเอียด
    for sub_id, statements in assessment_details.items():
        document.add_heading(f"รายละเอียดการประเมินเกณฑ์ย่อย: {sub_id}", level=2)
        
        # สร้างตารางสำหรับแต่ละ Sub-criteria (เปลี่ยนจาก 4 คอลัมน์เป็น 6 คอลัมน์)
        table = document.add_table(rows=1, cols=6, style='Table Grid')
        header_cells = table.rows[0].cells
        # เพิ่ม 2 คอลัมน์ใหม่: เหตุผล/วิเคราะห์, แหล่งที่มา
        headers = ["Statement ID (Level)", "ผลการประเมิน", "เกณฑ์มาตรฐาน (Standard)", "เหตุผล/วิเคราะห์", "แหล่งที่มา", "หลักฐาน/บริบท (Snippet)"] 
        for i, h in enumerate(headers):
            header_cells[i].text = h
            header_cells[i].paragraphs[0].runs[0].font.bold = True
            header_cells[i].paragraphs[0].runs[0].font.name = THAI_FONT_NAME 
            
        for statement in statements:
            status = "✅ PASS" if statement.get('is_pass', statement.get('pass_status', False)) else "❌ FAIL"
            level = statement.get('level', '-')
            
            # --- ดึงข้อมูลใหม่ ---
            reason_text = statement.get('reason', 'N/A')
            sources_list = statement.get('retrieved_sources_list', [])
            sources_text = "\n".join([
                f"{src.get('source_name', 'N/A')} (p.{src.get('location', 'N/A')})"
                for src in sources_list
            ]) if sources_list else 'ไม่มีแหล่งที่มา'
            # -------------------
            
            row_cells = table.add_row().cells
            
            row_cells[0].text = f"{statement.get('statement_id', '-')}\n(L{level})"
            row_cells[1].text = status
            row_cells[2].text = statement.get('standard', 'N/A')
            
            # --- ใส่ข้อมูลใหม่ในคอลัมน์ 3 และ 4 ---
            row_cells[3].text = reason_text 
            row_cells[4].text = sources_text 
            # ------------------------------------
            
            row_cells[5].text = statement.get('context_retrieved_snippet', 'ไม่มีหลักฐานสนับสนุนที่ชัดเจน')

            # กำหนดฟอนต์สำหรับแถวข้อมูล
            for cell in row_cells:
                for p in cell.paragraphs:
                    for run in p.runs:
                        run.font.name = THAI_FONT_NAME

            row_cells[1].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            if not statement.get('is_pass', statement.get('pass_status', False)):
                row_cells[1].paragraphs[0].runs[0].font.bold = True 

        document.add_paragraph() 

# ==========================
# 4. REPORT GENERATION FUNCTIONS (TXT) - ปรับปรุง Section 5
# ==========================

# (ฟังก์ชัน TXT อื่น ๆ เช่น generate_overall_summary_txt, generate_executive_summary_txt, generate_sub_criteria_status_txt, generate_action_plan_report_txt ไม่ได้ถูกแก้ไข)

def generate_overall_summary_txt(data: Dict[str, Any], report_lines: List[str], enabler_name_full: str): 
    """สร้างส่วนสรุปผลการประเมินโดยรวม (Overall) [SECTION 1] สำหรับ TXT"""
    overall = data.get("Overall", {})
    
    report_lines.append("="*80)
    report_lines.append(f"    [SECTION 1] สรุปผลการประเมิน {enabler_name_full} โดยรวม") 
    report_lines.append("="*80)
    report_lines.append(f"ตัวขับเคลื่อน (Enabler):        {overall.get('enabler', '-')} ({enabler_name_full})") 
    report_lines.append(f"คะแนนรวมถ่วงน้ำหนักที่ได้:     {overall.get('total_weighted_score', 0.0):.2f} / {overall.get('total_possible_weight', 0.0):.2f}")
    report_lines.append(f"เปอร์เซ็นต์ความคืบหน้าโดยรวม:  {overall.get('overall_progress_percent', 0.0):.2f}%")
    report_lines.append(f"คะแนนวุฒิภาวะโดยรวม (Maturity Score): {overall.get('overall_maturity_score', 0.0):.2f}")
    report_lines.append("="*80)

def generate_executive_summary_txt(summary: Dict[str, Any], report_lines: List[str]):
    """สร้างรายงานสรุปสำหรับผู้บริหาร (Executive Summary) [SECTION 2] สำหรับ TXT"""
    if not summary: return
    
    report_lines.append("\n" + "#"*80)
    report_lines.append("          [SECTION 2] สรุปสำหรับผู้บริหาร (Executive Summary)")
    report_lines.append("#"*80)

    overall = summary.get("Overall", {})
    report_lines.append(f"✅ คะแนนรวม: {overall.get('total_weighted_score', 0):.2f} / {overall.get('total_possible_weight', 0):.2f}")
    report_lines.append(f"✅ ร้อยละความสำเร็จ: {overall.get('overall_progress_percent', 0):.2f}%")
    report_lines.append(f"✅ ระดับความเป็นผู้ใหญ่: {overall.get('overall_maturity_score', 0):.2f}")
    report_lines.append("-" * 30)

    breakdown = summary.get("SubCriteria_Breakdown", {})
    if breakdown:
        # Strength: Top 3 highest scoring
        report_lines.append("\n📈 จุดแข็งที่โดดเด่น (Top Strengths):")
        top_strengths = sorted(breakdown.values(), key=lambda x: x.get("score", 0), reverse=True)[:3]
        for s in top_strengths:
            sub_name = s.get('name', s.get('topic', 'N/A'))
            report_lines.append(f"  • {sub_name} ({s.get('score', 0):.2f}/{s.get('weight', 0):.2f})")
        
        # Weakness: Top 3 with Gap (or lowest scoring with Gap)
        report_lines.append("\n🚨 จุดที่ควรพัฒนา (Development Areas):")
        gaps = [s for s in breakdown.values() if s.get("development_gap")]
        top_weaknesses = sorted(gaps, key=lambda x: x.get("score", 0))[:3]
        for s in top_weaknesses:
            sub_name = s.get('name', s.get('topic', 'N/A'))
            report_lines.append(f"  • {sub_name} (ระดับสูงสุดผ่าน: L{s.get('highest_full_level', 0)})")
    
    report_lines.append("#"*80)

def generate_sub_criteria_status_txt(data: Dict[str, Any], report_lines: List[str]) -> Dict[str, Any]:
    """สร้างตารางสถานะการประเมินรายเกณฑ์ย่อย [SECTION 3] สำหรับ TXT และคืนค่าเกณฑ์ที่มี Gap"""
    breakdown = data.get("SubCriteria_Breakdown", {})
    
    report_lines.append("\n" + "#"*80)
    report_lines.append("          [SECTION 3] สถานะการประเมินรายเกณฑ์ย่อยและ Gap")
    report_lines.append("#"*80)
    
    header_format = "{:<5} | {:<50} | {:<5} | {:<7} | {:<10}"
    separator = "-"*80
    
    report_lines.append(separator)
    report_lines.append(header_format.format("ID", "ชื่อเกณฑ์ย่อย", "คะแนน", "Level", "Gap"))
    report_lines.append(separator)
    
    gap_criteria = {}
    
    for sub_id, info in breakdown.items():
        name = info.get('name', info.get('topic', 'N/A'))
        score = info.get('score', 0.0)
        level = info.get('highest_full_level', 0)
        has_gap = "❌ YES" if info.get('development_gap', False) else "✅ NO"
        
        if info.get('development_gap', False):
            gap_criteria[sub_id] = info
        
        report_lines.append(header_format.format(
            sub_id, 
            name[:48], 
            f"{score:.2f}", 
            f"L{level}", 
            has_gap
        ))
    
    report_lines.append(separator)
    report_lines.append("")
    return gap_criteria

def generate_action_plan_report_txt(data: Dict[str, Any], gap_criteria: Dict[str, Any], report_lines: List[str]):
    """สร้างรายงานรายละเอียดแผนปฏิบัติการ (Action Plan) [SECTION 4] สำหรับ TXT (รวม L4/L5 Summary)"""
    action_plans = data.get("Action_Plans", {})
    
    report_lines.append("\n" + "*"*90)
    report_lines.append("       [SECTION 4] รายงานรายละเอียดแผนปฏิบัติการเพื่อปิดช่องว่าง (Action Plan)")
    report_lines.append("*"*90)

    if not gap_criteria:
        report_lines.append("✅ ทุกเกณฑ์ย่อยผ่านครบถ้วนแล้ว ไม่จำเป็นต้องมี Action Plan เพิ่มเติม")
        return
        
    for sub_id, sub_info in gap_criteria.items():
        sub_name = sub_info.get('name', sub_info.get('topic', 'N/A'))
        
        report_lines.append(f"\n[เกณฑ์ย่อย {sub_id}: {sub_name}] (Highest Full Level: L{sub_info.get('highest_full_level', 0)})")
        report_lines.append("-" * (len(sub_name) + 15))
        
        # --- NEW FEATURE: เพิ่ม L4/L5 Summary ---
        if "evidence_summary_L5" in sub_info:
            report_lines.append(f"  > 💡 L5 (เป้าหมายสูงสุด) สรุป: {sub_info['evidence_summary_L5'].get('summary', 'ไม่มีสรุป L5')[:100]}...")
            report_lines.append(f"  > 🎯 L5 ข้อเสนอแนะ: {sub_info['evidence_summary_L5'].get('suggestion_for_next_level', 'ไม่มีข้อเสนอแนะ')[:100]}...")
        if "evidence_summary_L4" in sub_info:
            report_lines.append(f"  > 💡 L4 สรุป: {sub_info['evidence_summary_L4'].get('summary', 'ไม่มีสรุป L4')[:100]}...")
        # --- END NEW FEATURE ---
        
        if sub_id in action_plans:
            
            for plan_phase in action_plans[sub_id]:
                phase = plan_phase.get('Phase', '-')
                goal = plan_phase.get('Goal', '-')
                actions_list = plan_phase.get('Actions', [])

                report_lines.append(f"  > 🛠️ เฟส/ขั้นตอน (Phase): {phase}")
                report_lines.append(f"  > 🎯 เป้าหมายหลัก (Goal): {goal}")
                
                if actions_list:
                    report_lines.append(f"  >>> แผนปฏิบัติการ {len(actions_list)} รายการ:")
                    for i, action in enumerate(actions_list, 1):
                        report_lines.append(f"    - Action {i}:")
                        report_lines.append(f"      - แนะนำ (Recommendation): {action.get('Recommendation', '-')}")
                        report_lines.append(f"      - หลักฐานเป้าหมาย (Evidence Type): {action.get('Target_Evidence_Type', '-')}")
                        report_lines.append(f"      - ตัวชี้วัดสำคัญ (Key Metric): {action.get('Key_Metric', '-')}")
                else:
                    report_lines.append("  >>> [ข้อมูล]: ไม่มี Action Plan ที่ต้องดำเนินการในเฟสนี้")
        else:
            report_lines.append("  >>> [ข้อมูล]: ไม่มี Action Plan ถูกกำหนดไว้ในส่วน Action_Plans")
    
    report_lines.append("\n" + "*"*90)

def generate_raw_details_report_txt(raw_data: Optional[Dict[str, Any]], report_lines: List[str]): 
    """สร้างรายงานการตรวจสอบความถูกต้องเชิงลึก (Raw Details) [SECTION 5] สำหรับ TXT (เพิ่ม Reason และ Source)"""
    
    report_lines.append("\n" + "="*80)
    report_lines.append("       [SECTION 5] รายงานการตรวจสอบความถูกต้องเชิงลึก (Raw Details)")
    report_lines.append("="*80)

    raw_data_base = raw_data 
    if raw_data is None:
        report_lines.append(f"⚠️ ไม่สามารถโหลดไฟล์ Raw Details ได้ หรือไฟล์ว่างเปล่า") 
        report_lines.append("="*80)
        return

    # พยายามดึงจากโครงสร้างหลักที่ใช้คีย์ 'Assessment_Details'
    assessment_details = {}
    
    # ตรวจสอบว่าเป็น Dict หรือ List ก่อน
    if isinstance(raw_data_base, dict):
        # Case 1: Standard Dictionary structure
        assessment_details = raw_data.get('Assessment_Details', {})
    elif isinstance(raw_data_base, list):
        # Case 2: List of statements structure (โหมด 'sub')
        statements_list = raw_data_base
        sub_id = statements_list[0].get('sub_criteria_id', 'N/A') if statements_list else 'N/A'
        if sub_id != 'N/A':
            assessment_details[sub_id] = statements_list
        else:
             report_lines.append("ℹ️ ข้อมูล Raw Details ถูกโหลดแล้ว แต่ไม่พบ 'sub_criteria_id' ใน Statement")
             return
    else:
        report_lines.append(f"⚠️ โครงสร้างข้อมูล Raw Details ไม่ถูกต้อง (ไม่ใช่ Dict หรือ List)") 
        return

    if not assessment_details:
         report_lines.append("ℹ️ ข้อมูล Raw Details ว่างเปล่าหลังจากตรวจสอบโครงสร้าง")
         report_lines.append("="*80)
         return
    
    # โค้ดแสดงผลรายละเอียด
    for sub_id, statements in assessment_details.items():
        report_lines.append(f"\n=======================================================")
        report_lines.append(f"| รายละเอียดการประเมินเกณฑ์ย่อย: {sub_id} |")
        report_lines.append(f"=======================================================")
        
        for statement in statements:
            status = "✅ PASS" if statement.get('is_pass', statement.get('pass_status', False)) else "❌ FAIL"
            level = statement.get('level', '-')
            snippet = statement.get('context_retrieved_snippet', 'ไม่มีหลักฐานสนับสนุนที่ชัดเจน')
            
            # --- NEW FEATURE: เพิ่ม Reason และ Source ---
            reason = statement.get('reason', 'N/A')
            sources_list = statement.get('retrieved_sources_list', [])
            sources_text = "; ".join([
                f"{src.get('source_name', 'N/A')} (p.{src.get('location', 'N/A')})"
                for src in sources_list
            ]) if sources_list else 'ไม่มีแหล่งที่มา'
            # --- END NEW FEATURE ---
            
            report_lines.append(f"\n[Statement ID: {statement.get('statement_id', '-')}] (Level {level}) - {status}")
            report_lines.append(f"  - เกณฑ์มาตรฐาน (Standard): {statement.get('standard', 'N/A')}")
            report_lines.append(f"  - เหตุผล/วิเคราะห์ (Reason): {reason}") # NEW
            report_lines.append(f"  - แหล่งที่มา (Sources): {sources_text}") # NEW
            # จำกัดความยาว Snippet ใน TXT และเพิ่ม ... หากยาวเกิน
            report_lines.append(f"  - หลักฐาน/บริบท (Snippet): {snippet[:150]}{'...' if len(snippet) > 150 else ''}") 
            
    report_lines.append("\n" + "="*80)


# ==========================
# 5. MAIN EXECUTION
# ==========================
def main():
    """ฟังก์ชันหลักในการสร้างรายงานทั้งหมด"""
    
    parser = argparse.ArgumentParser(description="Generate Comprehensive Assessment Reports.")
    parser.add_argument("--mode", choices=["all", "sub"], default="all", help="all: Generate full report. sub: Generate report for a specific sub-criteria.")
    parser.add_argument("--sub", type=str, help="SubCriteria ID (e.g., 2.2) if mode=sub.")
    parser.add_argument("--summary_file", type=str, required=True, help="Path to the Summary JSON file.")
    parser.add_argument("--raw_file", type=str, required=True, help="Path to the Raw Details JSON file.")
    parser.add_argument("--output_docx", type=str, default="reports/KM_Comprehensive_Report.docx", help="Output path for the DOCX file prefix.")
    parser.add_argument("--output_txt", type=str, default="reports/KM_Comprehensive_Report.txt", help="Output path for the TXT file.")
    
    args = parser.parse_args()
    
    # 1. จัดการ Folder Output
    setup_output_folder(args.output_docx)
    
    # 2. โหลดไฟล์
    summary_data = load_data(args.summary_file, "Summary Data")
    raw_data = load_data(args.raw_file, "Raw Details Data") 
    
    if not summary_data:
        print("🚨 ไม่สามารถสร้างรายงานได้เนื่องจากไฟล์ Summary Core Data ไม่พร้อม")
        return
    
    # --- 3. ดึง ENABLER และกำหนดค่าเริ่มต้น ---
    enabler_id = summary_data.get("Overall", {}).get("enabler", "GENERIC").upper()
    enabler_name_full = SEAM_ENABLER_MAP.get(enabler_id, f"Unknown Enabler ({enabler_id})")
    
    final_summary_data = summary_data
    final_raw_data = raw_data
    output_docx_path = args.output_docx
    output_txt_path = args.output_txt
    
    # --- 4. การกรองข้อมูลสำหรับโหมด 'sub' ---
    if args.mode == "sub" and args.sub:
        sub_id = args.sub.upper()
        print(f"🔹 โหมด: รายงานเฉพาะเกณฑ์ย่อย {sub_id} สำหรับ {enabler_name_full}")
        
        # 4.1. กรอง Summary Data
        if sub_id not in summary_data.get("SubCriteria_Breakdown", {}):
            print(f"⚠️ ไม่พบข้อมูลสำหรับเกณฑ์ย่อย {sub_id} ใน Summary Data. ใช้ข้อมูลทั้งหมดแทน.")
        else:
            final_summary_data = {
                "Overall": summary_data.get("Overall",{}),
                "SubCriteria_Breakdown": {sub_id: summary_data["SubCriteria_Breakdown"].get(sub_id,{})},
                "Action_Plans": {sub_id: summary_data.get("Action_Plans",{}).get(sub_id,[])}
            }
            
        # 4.2. กรอง Raw Data 
        if raw_data is not None:
            # ดึง statements ทั้งหมดออกมาเป็น list ก่อน
            all_statements = flatten_raw_data(raw_data)
            
            # กรอง statements เฉพาะ sub_id ที่ต้องการ
            filtered_statements = [
                stmt for stmt in all_statements 
                if stmt.get("sub_criteria_id", "").upper() == sub_id
            ]
            
            # กำหนด final_raw_data เป็น List ของ Statements ที่ถูกกรองแล้ว
            final_raw_data = filtered_statements if filtered_statements else None
                
        # 4.3. กำหนดชื่อ Output ใหม่
        report_prefix = f"{enabler_id}_Report_{sub_id}"
        output_docx_path = os.path.join(os.path.dirname(output_docx_path), f"{report_prefix}.docx")
        output_txt_path = os.path.join(os.path.dirname(output_txt_path), f"{report_prefix}.txt")
    
    else:
        print(f"🔹 โหมด: รายงานฉบับเต็มสำหรับ {enabler_name_full}")

    # --- A. การสร้างไฟล์ DOCX (แยกเป็น 2 ไฟล์: Strategic และ Raw Details) ---
    
    # 1. สร้าง Strategic Report (Sections 1-4)
    strategic_doc = Document()
    setup_document(strategic_doc) 
    
    # SECTION 1: Overall Summary
    generate_overall_summary_docx(strategic_doc, final_summary_data, enabler_name_full) 
    # SECTION 2: Executive Summary
    generate_executive_summary_docx(strategic_doc, final_summary_data)
    # SECTION 3: Sub-Criteria Status & Gap
    gap_criteria_docx = generate_sub_criteria_status_docx(strategic_doc, final_summary_data)
    # SECTION 4: Action Plan Report (พร้อม L4/L5 Summary)
    generate_action_plan_report_docx(strategic_doc, final_summary_data, gap_criteria_docx)

    # บันทึกไฟล์ Strategic Report
    strategic_path = output_docx_path.rsplit('.', 1)[0] + "_Strategic.docx"
    strategic_doc.save(strategic_path)
    print(f"🎉 สร้างไฟล์ DOCX [Strategic Report] สำเร็จ! บันทึกที่: {strategic_path}")


    # 2. สร้าง Raw Details Working Document (Section 5)
    detail_doc = Document()
    setup_document(detail_doc) 
    detail_doc.add_heading(f"[SECTION 5] รายงานหลักฐานเชิงลึก (Raw Details) - {enabler_name_full} ({REPORT_DATE})", level=1)
    # SECTION 5: Raw Details (พร้อม Reason และ Source)
    generate_raw_details_report_docx(detail_doc, final_raw_data) 

    # บันทึกไฟล์ Raw Details
    detail_path = output_docx_path.rsplit('.', 1)[0] + "_RawDetails.docx"
    detail_doc.save(detail_path)
    print(f"🎉 สร้างไฟล์ DOCX [Raw Details] สำเร็จ! บันทึกที่: {detail_path}")

    # --- B. การสร้างไฟล์ TXT (ฉบับรวม 5 Sections) ---
    if os.path.exists(output_txt_path):
        os.remove(output_txt_path)
        
    txt_report_lines = []
    
    # SECTION 1: Overall Summary
    generate_overall_summary_txt(final_summary_data, txt_report_lines, enabler_name_full) 
    # SECTION 2: Executive Summary
    generate_executive_summary_txt(final_summary_data, txt_report_lines)
    # SECTION 3: Sub-Criteria Status & Gap
    gap_criteria_txt = generate_sub_criteria_status_txt(final_summary_data, txt_report_lines)
    # SECTION 4: Action Plan Report (พร้อม L4/L5 Summary)
    generate_action_plan_report_txt(final_summary_data, gap_criteria_txt, txt_report_lines)
    # SECTION 5: Raw Details (พร้อม Reason และ Source)
    generate_raw_details_report_txt(final_raw_data, txt_report_lines) 
    
    # บันทึกไฟล์ TXT
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(txt_report_lines))
    
    print(f"🎉 สร้างไฟล์ TXT สำเร็จ! บันทึกที่: {output_txt_path}")

if __name__ == "__main__":
    main()