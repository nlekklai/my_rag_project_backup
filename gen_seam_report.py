# gen_seam_report.py (โค้ดฉบับแก้ไขรอบที่ 4: ดึงชื่อ Source จาก Snippet หากไม่พบใน Metadata)

import json
import os
import argparse
from typing import Dict, Any, Optional, List
from datetime import datetime
import re 

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

# พจนานุกรมสำหรับชื่อ Enabler ฉบับเต็ม
SEAM_ENABLER_MAP = {
    "KM": "7.1 การจัดการความรู้ (Knowledge Management)",
    "IT": "7.2 เทคโนโลยีดิจิทัล",
    "HR": "6.1 การบริหารทุนมนุษย์ (Human Capital Management)",
    "CG": "1.1 การกำกับดูแลที่ดีและการนำองค์กร",
    "SP": "2.1 การวางแผนเชิงยุทธศาสตร์",
    "RM": "3.1 การบริหารความเสี่ยงและการควบคุมภายใน",
    "SCM": "4.1 การมุ่งเน้นผู้มีส่วนได้ส่วนเสียและลูกค้า",
    "IM": "7.2 การจัดการนวัตกรรม",
    "IA": "8.1 การตรวจสอบภายใน"
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
        print(f"❌ ข้อผิดพลาดในการโหลดไฟล์ {file_type} '{file_path}': {e}") 
        return None

def setup_output_folder(file_path):
    """ตรวจสอบและสร้าง folder output"""
    output_dir = os.path.dirname(file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

def setup_document(doc):
    """Sets up document-wide formatting like margins and default font."""
    section = doc.sections[0]
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(0.75) 
    section.right_margin = Inches(0.75)

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
    
    for run in p.runs:
        run.font.name = THAI_FONT_NAME 

def flatten_raw_data(raw_data_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    ดึง Statement ทั้งหมดออกมาจาก Raw Data Dictionary 
    (รองรับโครงสร้าง List ของ Statements โดยตรง)
    """
    statements = []
    
    # 1. โครงสร้าง List ของ Statements โดยตรง (สำหรับ New Single-File Export)
    if isinstance(raw_data_dict, list):
        statements = [s for s in raw_data_dict if isinstance(s, dict)]
        if statements:
            return statements
            
    # 2. โครงสร้างเดิมที่เผื่อไว้ (สำหรับ Raw Details ที่เคยเป็น Dictionary)
    if isinstance(raw_data_dict, dict) and 'sub_criteria_results' in raw_data_dict:
        sub_results = raw_data_dict.get('sub_criteria_results', [])
        for sub_item in sub_results:
            if isinstance(sub_item, dict) and 'raw_results_ref' in sub_item:
                raw_statements = sub_item['raw_results_ref']
                if isinstance(raw_statements, list):
                    statements.extend([s for s in raw_statements if isinstance(s, dict)])
        if statements:
            return statements
            
    # 3. โครงสร้างเก่ามาก (เผื่อไว้)
    details = raw_data_dict.get("Assessment_Details") if isinstance(raw_data_dict, dict) else None
    if isinstance(details, dict):
        for sub_id_statements in details.values():
            if isinstance(sub_id_statements, list):
                statements.extend([s for s in sub_id_statements if isinstance(s, dict)])
                
    return statements

# ==========================
# 3. REPORT GENERATION FUNCTIONS (DOCX)
# ==========================

def generate_overall_summary_docx(document: Document, summary_data: Dict[str, Any], enabler_name_full: str): 
    """สร้างส่วนสรุปผลการประเมินโดยรวม (Overall) [SECTION 1] ใน DOCX"""
    
    set_heading(document, f'[SECTION 1] สรุปผลการประเมิน {enabler_name_full} โดยรวม', level=1)
    
    # ดึงคีย์ที่ถูกต้องจาก New Single-File structure (ซึ่งมาจาก 'summary')
    achieved_score = summary_data.get('Total Weighted Score Achieved', 0.0)
    overall_max_score = summary_data.get('Total Possible Weight', 0.0)

    # Percentage calculation
    overall_percent = summary_data.get('Overall Progress Percentage (0.0 - 1.0)', 0.0) * 100

    # Maturity Score & Level: 
    maturity_score = summary_data.get('Overall Maturity Score (Avg.)', 0.0) 
    maturity_level = summary_data.get('Overall Maturity Level (Weighted)', 'N/A')
    
    table = document.add_table(rows=5, cols=2) 
    table.style = 'Table Grid'
    
    def add_summary_row(row_index, label, value):
        table.cell(row_index, 0).text = label
        table.cell(row_index, 1).text = value
        table.cell(row_index, 0).paragraphs[0].runs[0].font.bold = True
        table.cell(row_index, 1).paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
        
    enabler_id = summary_data.get("enabler", summary_data.get("enabler_id", "N/A")).upper()
    add_summary_row(0, "ตัวขับเคลื่อน (Enabler):", f"{enabler_id}\n({enabler_name_full})") 
    add_summary_row(1, "คะแนนรวมถ่วงน้ำหนักที่ได้:", f"{achieved_score:.2f} / {overall_max_score:.2f}")
    add_summary_row(2, "เปอร์เซ็นต์ความคืบหน้าโดยรวม:", f"{overall_percent:.2f}%")
    add_summary_row(3, "คะแนนวุฒิภาวะโดยรวม (Maturity Score):", f"{maturity_score:.2f}")
    add_summary_row(4, "ระดับวุฒิภาวะโดยรวม (Maturity Level):", maturity_level)
    
    # กำหนดฟอนต์ให้ Table Headers และ Content
    for row in table.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.name = THAI_FONT_NAME
    
    document.add_paragraph() 

def generate_executive_summary_docx(document: Document, summary_data: Dict[str, Any], sub_results: List[Dict[str, Any]]):
    """สร้างรายงานสรุปสำหรับผู้บริหาร (Executive Summary) [SECTION 2] ใน DOCX"""
    if not summary_data: return
    set_heading(document, "[SECTION 2] สรุปสำหรับผู้บริหาร (Executive Summary)", level=1)

    # ดึงข้อมูลจาก Summary Section 
    achieved_score = summary_data.get('Total Weighted Score Achieved', 0.0)
    overall_max_score = summary_data.get('Total Possible Weight', 0.0)
    
    overall_percent = summary_data.get('Overall Progress Percentage (0.0 - 1.0)', 0.0) * 100
    maturity_score = summary_data.get('Overall Maturity Score (Avg.)', 0.0)
    maturity_level = summary_data.get('Overall Maturity Level (Weighted)', 'N/A')

    add_paragraph(document, f"✅ คะแนนรวม: {achieved_score:.2f} / {overall_max_score:.2f}")
    add_paragraph(document, f"✅ ร้อยละความสำเร็จ: {overall_percent:.2f}%")
    add_paragraph(document, f"✅ ระดับความเป็นผู้ใหญ่: {maturity_score:.2f} ({maturity_level})")
    document.add_paragraph()

    if sub_results:
        # Strength: Top 3 highest scoring
        add_paragraph(document, "📈 จุดแข็งที่โดดเด่น (Top Strengths):", bold=True, color=(0x00, 0x70, 0xC0))
        top_strengths = sorted(sub_results, key=lambda x: (x.get("weighted_score", 0) / x.get("weight", 1)) if x.get("weight", 1) > 0 else 0, reverse=True)[:3]
        for s in top_strengths:
            sub_id = s.get('sub_criteria_id', 'N/A')
            sub_name = s.get('sub_criteria_name', 'N/A')
            add_paragraph(document, f"• {sub_id} - {sub_name} (L{s.get('highest_full_level', 0)} ได้ {s.get('weighted_score', 0):.2f}/{s.get('weight', 0):.2f})", style="List Bullet")

        document.add_paragraph()
        
        # Weakness: Top 3 with Gap (or lowest scoring with Gap)
        add_paragraph(document, "🚨 จุดที่ควรพัฒนา (Development Areas):", bold=True, color=(0xFF, 0x00, 0x00))
        gaps = [s for s in sub_results if not s.get("target_level_achieved", True)]
        # เรียงตามระดับที่ผ่านได้สูงสุด (Highest Full Level) น้อยที่สุด
        top_weaknesses = sorted(gaps, key=lambda x: x.get("highest_full_level", 0))[:3] 
        for s in top_weaknesses:
            sub_id = s.get('sub_criteria_id', 'N/A')
            sub_name = s.get('sub_criteria_name', 'N/A')
            add_paragraph(document, f"• {sub_id} - {sub_name} (ระดับสูงสุดผ่าน: L{s.get('highest_full_level', 0)})", style="List Bullet")
    
    document.add_paragraph()

def generate_sub_criteria_status_docx(document: Document, sub_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """สร้างตารางสถานะการประเมินรายเกณฑ์ย่อย [SECTION 3] ใน DOCX และคืนค่าเกณฑ์ที่มี Gap"""
    
    document.add_heading('[SECTION 3] สถานะการประเมินรายเกณฑ์ย่อยและ Gap', level=1)
    
    table = document.add_table(rows=1, cols=5)
    table.style = 'Table Grid'
    
    # ตั้งค่าความกว้างคอลัมน์
    table.columns[0].width = Inches(0.5) 
    table.columns[1].width = Inches(4.5) 
    table.columns[2].width = Inches(0.8) # คะแนน/น้ำหนัก
    table.columns[3].width = Inches(0.7) 
    table.columns[4].width = Inches(0.8) # Gap
    
    header_cells = table.rows[0].cells
    headers = ["ID", "ชื่อเกณฑ์ย่อย", "คะแนน/น้ำหนัก", "Level", "Gap"] 
    for i, h in enumerate(headers):
        header_cells[i].text = h
        header_cells[i].paragraphs[0].runs[0].font.bold = True
        header_cells[i].paragraphs[0].runs[0].font.name = THAI_FONT_NAME 
        header_cells[i].vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    gap_criteria = {}
    
    for info in sub_results:
        sub_id = info.get('sub_criteria_id')
        if not sub_id: continue

        row_cells = table.add_row().cells
        
        name = info.get('sub_criteria_name', 'N/A') 
        score = info.get('weighted_score', 0.0) 
        weight = info.get('weight', 0.0) 
        level = info.get('highest_full_level', 0)
        has_gap = "❌ YES" if not info.get('target_level_achieved', True) else "✅ NO"
        
        if not info.get('target_level_achieved', True):
            # เก็บข้อมูล Sub-criteria ที่มี Gap
            gap_criteria[sub_id] = info 
            
        row_cells[0].text = sub_id
        row_cells[1].text = name
        row_cells[2].text = f"{score:.2f} / {weight:.2f}" 
        row_cells[3].text = f"L{level}"
        row_cells[4].text = has_gap
        
        # กำหนดฟอนต์สำหรับแถวข้อมูล
        for row in table.rows: # Re-apply font to all rows/cells for consistency
            for cell in row.cells:
                for p in cell.paragraphs:
                    for run in p.runs:
                        run.font.name = THAI_FONT_NAME

        row_cells[0].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        row_cells[2].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        row_cells[3].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        row_cells[4].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    document.add_paragraph() 
    return gap_criteria

def generate_action_plan_report_docx(document: Document, gap_criteria: Dict[str, Any]):
    """สร้างรายงานรายละเอียดแผนปฏิบัติการ (Action Plan) [SECTION 4] ใน DOCX"""
    
    document.add_heading('[SECTION 4] รายงานรายละเอียดแผนปฏิบัติการเพื่อปิดช่องว่าง (Action Plan)', level=1)
    
    if not gap_criteria:
        add_paragraph(document, "✅ ทุกเกณฑ์ย่อยผ่านครบถ้วนแล้ว ไม่จำเป็นต้องมี Action Plan เพิ่มเติม")
        return

    add_paragraph(document, 
        "ℹ️ คำแนะนำเหล่านี้เป็นการระบุ 'ช่องว่างเชิงระบบ' (Systemic Gap) เพื่อให้องค์กรบรรลุวุฒิภาวะ Level ถัดไป องค์กรควรนำคำแนะนำเหล่านี้ไปแตกเป็น 'แผนปฏิบัติการย่อย (Detailed Work Plan)' ที่ระบุรายละเอียดกิจกรรม, ผู้รับผิดชอบ, และไทม์ไลน์ ภายใต้กรอบ PDCA ที่กำหนดไว้",
        italic=True,
        color=(0x80, 0x80, 0x80)
    )
    document.add_paragraph()

    for sub_id, sub_info in gap_criteria.items():
        sub_name = sub_info.get('sub_criteria_name', 'N/A')
        
        document.add_heading(f"• เกณฑ์ย่อย {sub_id}: {sub_name} (Highest Full Level: L{sub_info.get('highest_full_level', 0)})", level=2)
        
        # ดึง Summary Evidence ที่สำคัญ (ถ้ามี)
        evidence_summary_L5 = sub_info.get("evidence_summary_L5", {})
        evidence_summary_L4 = sub_info.get("evidence_summary_L4", {})
        
        if evidence_summary_L5.get('summary'):
            add_paragraph(document, "💡 ข้อมูลเชิงลึกหลักฐานระดับ L5 (เป้าหมายสูงสุด):", bold=True, color=(0x00, 0x70, 0xC0))
            add_paragraph(document, f"   - สรุปหลักฐาน: {evidence_summary_L5.get('summary', 'ไม่มีสรุป L5')}", italic=True)
            p_sugg = add_paragraph(document, f"   - ข้อเสนอแนะ: ", italic=True)
            run_sugg = p_sugg.add_run(evidence_summary_L5.get('suggestion_for_next_level', 'ไม่มีข้อเสนอแนะ'))
            run_sugg.font.bold = True
            run_sugg.font.name = THAI_FONT_NAME
            document.add_paragraph()
        
        if evidence_summary_L4.get('summary'):
            add_paragraph(document, "💡 ข้อมูลเชิงลึกหลักฐานระดับ L4:", bold=True, color=(0x00, 0x70, 0xC0))
            add_paragraph(document, f"   - สรุปหลักฐาน: {evidence_summary_L4.get('summary', 'ไม่มีสรุป L4')}", italic=True)
            document.add_paragraph()
        
        action_plans = sub_info.get('action_plan', [])
        
        # จัดกลุ่ม Action Plan ตาม PDCA (เผื่อกรณี LLM สร้าง Action Plan แบบไม่มี Phase/Goal)
        pdca_actions = {
            'P (Plan / วางแผน)': [],
            'D (Do / ปฏิบัติ)': [],
            'C (Check / ตรวจสอบ)': [],
            'A (Act / ปรับปรุง)': []
        }

        # 1. ตรวจสอบโครงสร้างแบบซับซ้อน (Phase/Goal/Actions)
        is_complex_structure = (
             isinstance(action_plans, list) and 
             action_plans and 
             isinstance(action_plans[0], dict) and 
             'Phase' in action_plans[0]
        )

        if is_complex_structure:
            # วนลูปและดึงข้อมูลจากโครงสร้าง Phase/Goal/Actions
            for plan in action_plans:
                # Basic mapping based on name prefix
                phase = plan.get('Phase', 'D (Do / ปฏิบัติ)')
                if 'P (' in phase: pdca_key = 'P (Plan / วางแผน)'
                elif 'D (' in phase: pdca_key = 'D (Do / ปฏิบัติ)'
                elif 'C (' in phase: pdca_key = 'C (Check / ตรวจสอบ)'
                elif 'A (' in phase: pdca_key = 'A (Act / ปรับปรุง)'
                else: pdca_key = 'D (Do / ปฏิบัติ)' 
                    
                for action in plan.get('Actions', []):
                    # สมมติว่า Action Plan ในโครงสร้างซับซ้อนมีแค่ Recommendation
                    pdca_actions[pdca_key].append({
                        'rec': action.get('Recommendation', ''), 
                        'target_evidence': '-', # ไม่มีข้อมูลในโครงสร้างนี้
                        'key_metric': '-' # ไม่มีข้อมูลในโครงสร้างนี้
                    })
        
        # 2. ตรวจสอบโครงสร้างแบบเรียบง่าย (List of Recommendations ที่มีคีย์ Target_Evidence_Type, Key_Metric)
        elif isinstance(action_plans, list) and all(isinstance(a, dict) and 'Recommendation' in a for a in action_plans):
            for action in action_plans:
                rec = action.get('Recommendation', '')
                failed_level = action.get('Failed_Level', 5)
                target_evidence = action.get('Target_Evidence_Type', '-')
                key_metric = action.get('Key_Metric', '-')
                
                # Logic การจัดกลุ่ม PDCA (ใช้ Logic เดิมที่เคยทำไว้สำหรับ Action Plan)
                if 'ปรับปรุง' in rec or 'ข้อมูลป้อนกลับ' in rec or failed_level == 5:
                    pdca_key = 'A (Act / ปรับปรุง)'
                elif 'ติดตาม' in rec or 'ประเมินผล' in rec or 'ทบทวน' in rec:
                    pdca_key = 'C (Check / ตรวจสอบ)'
                elif failed_level in [1, 2] or 'กำหนดแผน' in rec or 'กำหนดกลยุทธ์' in rec:
                    pdca_key = 'P (Plan / วางแผน)' 
                else:
                    pdca_key = 'D (Do / ปฏิบัติ)'

                pdca_actions[pdca_key].append({
                    'rec': rec, 
                    'target_evidence': target_evidence,
                    'key_metric': key_metric
                })
                
        # 3. สร้างตารางรายงาน PDCA จากข้อมูลที่จัดกลุ่มแล้ว
        if any(actions for actions in pdca_actions.values()):
            document.add_paragraph()
            add_paragraph(document, "📋 แผนปฏิบัติการเพื่อปิดช่องว่างตามวงจร PDCA (User Guideline)", bold=True, color=(0x00, 0x00, 0x00))

            for phase, actions in pdca_actions.items():
                if actions:
                    document.add_heading(f"--- {phase} ---", level=4)
                    
                    # สร้างตารางเฉพาะเมื่อมีข้อมูล (สำหรับโครงสร้างแบบเรียบง่ายที่มี Target/Metric)
                    if not is_complex_structure and all(a.get('target_evidence') != '-' for a in actions):
                        action_table = document.add_table(rows=1, cols=3, style='Table Grid')
                        
                        header_cells = action_table.rows[0].cells
                        header_cells[0].text = "คำแนะนำ (Recommendation)"
                        header_cells[1].text = "หลักฐานเป้าหมาย (Evidence Type)"
                        header_cells[2].text = "ตัวชี้วัดสำคัญ (Key Metric)"
                        
                        for cell in action_table.rows[0].cells:
                             cell.paragraphs[0].runs[0].font.bold = True
                             cell.paragraphs[0].runs[0].font.name = THAI_FONT_NAME 
                        
                        for action in actions:
                            row_cells = action_table.add_row().cells
                            
                            row_cells[0].text = action['rec']
                            
                            p_evidence = row_cells[1].paragraphs[0]
                            run_evidence = p_evidence.add_run(action['target_evidence'])
                            run_evidence.font.bold = True
                            run_evidence.font.color.rgb = RGBColor(0x00, 0x70, 0xC0) 
                            
                            p_metric = row_cells[2].paragraphs[0]
                            run_metric = p_metric.add_run(action['key_metric'])
                            run_metric.font.bold = True
                            run_metric.font.color.rgb = RGBColor(0xFF, 0x00, 0x00)
                            
                            for cell in row_cells:
                                cell.vertical_alignment = WD_ALIGN_VERTICAL.TOP
                                for p_cell in cell.paragraphs:
                                    for run in p_cell.runs:
                                        run.font.name = THAI_FONT_NAME
                                        run.font.size = Pt(11) 
                    
                    # ถ้าเป็นโครงสร้างซับซ้อน หรือขาด Target/Metric ให้แสดงเป็น List Bullet ธรรมดา
                    else:
                        for action in actions:
                            add_paragraph(document, f"• {action['rec']}", style="List Bullet")
                    
                    document.add_paragraph() 
        else:
            add_paragraph(document, f">>> [ข้อมูล]: ไม่มี Action Plan ถูกกำหนดไว้สำหรับเกณฑ์ย่อย {sub_id}", style='List Bullet')
        document.add_paragraph()

def generate_raw_details_report_docx(document: Document, raw_data_list: List[Dict[str, Any]]): 
    """
    สร้างรายงานการตรวจสอบความถูกต้องเชิงลึก (Raw Details) [SECTION 5] ใน DOCX
    หมายเหตุ: raw_data_list คือ List ของ Statements ที่มาจาก 'raw_llm_results'
    """
    
    document.add_heading('[SECTION 5] รายงานหลักฐานเชิงลึก (Raw Details)', level=1)
    
    if not raw_data_list or not isinstance(raw_data_list, list):
        add_paragraph(document, f"⚠️ ไม่พบ Raw Statements ในไฟล์ Input (คีย์ 'raw_llm_results' อาจว่างเปล่า)", bold=True, color=(0xFF, 0x80, 0x00)) 
        return
        
    assessment_details = {}
    for statement in raw_data_list:
        sub_id = statement.get('sub_criteria_id', 'N/A')
        if sub_id != 'N/A' and isinstance(statement, dict):
            if sub_id not in assessment_details:
                assessment_details[sub_id] = []
            assessment_details[sub_id].append(statement)
    
    sorted_assessment_details = dict(sorted(assessment_details.items()))
    
    if not sorted_assessment_details:
        add_paragraph(document, "🚨 **ไม่พบ Raw Statements ใดๆ** (โปรดตรวจสอบว่าไฟล์ Input JSON ถูกต้อง)", bold=True, color=(0xFF, 0x00, 0x00))
        return 

    for sub_id, statements in sorted_assessment_details.items():
        document.add_heading(f"รายละเอียดการประเมินเกณฑ์ย่อย: {sub_id}", level=2)
        
        table = document.add_table(rows=1, cols=5, style='Table Grid')

        table.columns[0].width = Inches(1.0)  
        table.columns[1].width = Inches(1.0)  
        table.columns[2].width = Inches(2.5)  
        table.columns[3].width = Inches(2.0)  
        table.columns[4].width = Inches(3.0)  
        
        header_cells = table.rows[0].cells
        
        headers = ["Statement ID (Level)", "ผลการประเมิน", "เหตุผล/วิเคราะห์", "แหล่งที่มา", "หลักฐาน/บริบท (Snippet)"] 
        for i, h in enumerate(headers):
            header_cells[i].text = h
            header_cells[i].paragraphs[0].runs[0].font.bold = True
            header_cells[i].paragraphs[0].runs[0].font.name = THAI_FONT_NAME 

        if not statements or not all(isinstance(s, dict) for s in statements):
            row_cells = table.add_row().cells
            row_cells[0].merge(row_cells[4]) 
            merged_cell = row_cells[0]
            merged_cell.text = "⚠️ ไม่พบ Raw Statements สำหรับเกณฑ์ย่อยนี้"
            
            for p_cell in merged_cell.paragraphs:
                run = p_cell.runs[0]
                run.font.name = THAI_FONT_NAME
                run.font.size = Pt(11)
                run.font.color.rgb = RGBColor(0xFF, 0x80, 0x00) 
            
            document.add_paragraph()
            continue 
            
        # Regex pattern to find source entries in the raw context snippet
        # [SOURCE: filename.ext (ID:hash)]
        SOURCE_PATTERN = re.compile(r'\[SOURCE:\s*(.*?)\s*\(ID:([0-9a-f]+)...?\)\s*\]', re.DOTALL)


        for statement in statements:
            is_passed = statement.get('is_passed', statement.get('is_pass', False)) 
            status = "✅ PASS" if is_passed else "❌ FAIL"
            level = statement.get('level', '-')
            reason_text = statement.get('reason', 'N/A')

            # --- START: ดึง Source/Snippet จากคีย์ Custom ---
            sources_list_raw = statement.get('retrieved_full_source_info', []) 
            
            # 1. ดึง Snippet มาเพื่อหาชื่อไฟล์ที่ถูกซ่อนไว้
            context_snippet_raw = statement.get('aggregated_context_used', 'ไม่มีหลักฐานสนับสนุนที่ชัดเจน') 
            extracted_names_from_snippet = {} # {doc_id_prefix: file_name}
            
            for match in SOURCE_PATTERN.finditer(context_snippet_raw):
                file_name = match.group(1).strip()
                doc_id_prefix = match.group(2).strip() # Hash ID Prefix
                # ใช้ Prefix 8 ตัวแรกเป็นคีย์ในการจับคู่
                extracted_names_from_snippet[doc_id_prefix] = file_name 
            
            
            valid_sources = []
            if isinstance(sources_list_raw, list): 
                for src in sources_list_raw:
                    # 2.1 พยายามหาชื่อไฟล์จากคีย์ที่ชัดเจนก่อน
                    name = (
                        src.get('file_name') or 
                        src.get('source_name') or 
                        src.get('title') # ค้นหา 'title'
                    )
                    
                    # 2.2 ดึง doc_id สำหรับการจับคู่
                    doc_id = str(src.get('doc_id') or src.get('document_id', '')).strip()
                    doc_id_prefix = doc_id[:8]

                    # 2.3 ถ้าไม่มีชื่อที่ชัดเจน ให้ใช้ชื่อที่ดึงมาจาก Snippet แทน (ถ้าพบการจับคู่ Hash Prefix)
                    if not name and doc_id_prefix and doc_id_prefix in extracted_names_from_snippet:
                         name = extracted_names_from_snippet[doc_id_prefix] # ใช้ชื่อที่ดึงมา
                    
                    
                    location = src.get('page')
                    rank = src.get('retrieved_rank')
                    
                    if name: # พบชื่อที่อ่านได้ (ทั้งแบบตรงและแบบที่ดึงจาก snippet)
                        location_display = f" (p.{location})" if location and str(location).strip() != "" else ''
                        rank_display = f" (Rank {rank})" if rank is not None else ""
                        
                        valid_sources.append(f"{name}{location_display}{rank_display}")
                    else:
                        # Fallback message (ถ้ายังไม่พบชื่อ)
                        if doc_id:
                            # Check if it's the long hash the user complained about
                            if len(doc_id) > 20:
                                 valid_sources.append(f"[ERROR: ไม่พบชื่อไฟล์ (Hash: {doc_id_prefix}...)]")
                            else:
                                 valid_sources.append(f"[ERROR: ไม่พบชื่อไฟล์ (ID: {doc_id})]")
                        else:
                            valid_sources.append("[ERROR: ไม่พบชื่อแหล่งที่มา]")

            sources_text = "\n".join(valid_sources) if valid_sources else 'ไม่มีแหล่งที่มา'
            
            # --- END: ดึง Source/Snippet ---

            row_cells = table.add_row().cells
            
            # Column 1
            row_cells[0].text = f"{statement.get('sub_criteria_id', '-')}\n(L{level})"
            
            # Column 2
            llm_score = statement.get('llm_score', '-')
            if llm_score != '-':
                row_cells[1].text = f"{status}\n({llm_score}/1)" 
            else:
                 row_cells[1].text = status
            
            # Column 3
            row_cells[2].text = reason_text 
            
            # Column 4 (Source)
            row_cells[3].text = sources_text 
            
            # Column 5 (Snippet) - ต้องทำความสะอาด Source Prefix ออกจาก Snippet
            context_snippet_cleaned = context_snippet_raw
            
            # FIX 2: Clean the Snippet by removing the [SOURCE: ...] prefix if present 
            # เราใช้ Regex เดียวกันนี้เพื่อทำความสะอาด
            # เราใช้ sub() เพื่อแทนที่ทุกครั้งที่พบ pattern ด้วยสตริงว่าง
            context_snippet_cleaned = SOURCE_PATTERN.sub('', context_snippet_raw).strip() 

            if not context_snippet_cleaned:
                context_snippet_cleaned = 'ไม่มีหลักฐานสนับสนุนที่ชัดเจน (ถูกลบ Source Prefix ออกไป)'
            
            # Take only the first 300 characters for a snippet if full context is too long
            if len(context_snippet_cleaned) > 300:
                row_cells[4].text = context_snippet_cleaned[:300] + "..."
            else:
                row_cells[4].text = context_snippet_cleaned
            
            # กำหนดฟอนต์สำหรับแถวข้อมูล
            for cell in row_cells:
                cell.vertical_alignment = WD_ALIGN_VERTICAL.TOP
                for p_cell in cell.paragraphs:
                    for run in p_cell.runs:
                        run.font.name = THAI_FONT_NAME
                        run.font.size = Pt(11)

            row_cells[1].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            if not is_passed:
                row_cells[1].paragraphs[0].runs[0].font.bold = True 

        document.add_paragraph()

# ==========================
# 4. MAIN EXECUTION (Updated for Single-File Input)
# ==========================
def main():
    """ฟังก์ชันหลักในการสร้างรายงานทั้งหมด (รองรับ Input 1 ไฟล์ในโครงสร้างใหม่)"""
    
    parser = argparse.ArgumentParser(description="Generate Comprehensive SEAM Assessment Reports (Using the new Single-File Export).")
    parser.add_argument("--mode", choices=["all", "sub"], default="all", help="all: Generate full report. sub: Generate report for a specific sub-criteria.")
    parser.add_argument("--sub", type=str, help="SubCriteria ID (e.g., 2.2) if mode=sub.")
    
    # *** 🟢 รับ 1 ไฟล์ Input ***
    parser.add_argument("--results_file", type=str, required=True, help="Path to the unified JSON results file (e.g., seam_assessment_results_km_L5_...json).") 
    
    parser.add_argument("--output_path", type=str, default="reports/SEAM_Comprehensive_Report", help="Output directory and base filename prefix (e.g., reports/SEAM_Comprehensive_Report).")
    
    args = parser.parse_args()
    
    # 1. จัดการ Folder Output และแยก Directory
    output_dir = os.path.dirname(args.output_path)
    if not output_dir:
         output_dir = EXPORT_DIR 
    setup_output_folder(output_dir)
    
    # 2. โหลดไฟล์ Single Results File
    results_data_loaded = load_data(args.results_file, "Unified Results File")
    
    if not results_data_loaded:
        print("🚨 ไม่สามารถสร้างรายงานได้เนื่องจากไฟล์ Results Data โหลดไม่ได้")
        return
        
    # --- 3. การดึงข้อมูลจากโครงสร้าง Single-File Export ---
    
    # 3.1 GOAL 1: Get Summary Section (for Sections 1 & 2)
    summary_section = results_data_loaded.get("summary", {})
    if not summary_section:
        print("🚨 ไม่พบคีย์ 'summary' ในไฟล์ Results Data (โปรดตรวจสอบโครงสร้างไฟล์)")
        return
    
    # 3.2 GOAL 2: Synthesize Sub Results Full (for Sections 2, 3, 4)
    # FIX: Access 'sub_criteria_results' as a LIST
    sub_results_list = results_data_loaded.get("sub_criteria_results")
    sub_results_full = []
    
    if sub_results_list and isinstance(sub_results_list, list):
        print("✅ ตรวจพบโครงสร้าง Results File (summary/sub_criteria_results เป็น List) กำลังเตรียมข้อมูล...")
        
        for item in sub_results_list:
            if isinstance(item, dict):
                sub_results_full.append({
                    "sub_criteria_id": item.get('sub_criteria_id', 'N/A'),
                    "sub_criteria_name": item.get('sub_criteria_name', 'N/A'),
                    "weighted_score": item.get('weighted_score', 0.0), 
                    "weight": item.get('weight', 0.0),      
                    "highest_full_level": item.get('highest_full_level', 0),
                    "target_level_achieved": item.get('target_level_achieved', False), 
                    "action_plan": item.get('action_plan', []),
                    "evidence_summary_L5": item.get('evidence_summary_L5', {}),
                    "evidence_summary_L4": item.get('evidence_summary_L4', {}),
                })
    else:
         print("🚨 ไม่พบหรือโครงสร้าง 'sub_criteria_results' ไม่ถูกต้อง")
         return
         
    # 3.3 GOAL 3: Raw Data (for Section 5)
    # FIX: Correct key name to 'raw_llm_results'
    raw_data_for_section5 = results_data_loaded.get("raw_llm_results") 
    
    if not raw_data_for_section5:
         print("🚨 ไม่พบ Raw Data ในคีย์ 'raw_llm_results' (Section 5 จะถูกข้าม)")
         raw_data_for_section5 = [] 
         
    # ดึง ENABLER และกำหนดค่าเริ่มต้น
    enabler_id = summary_section.get("enabler", summary_section.get("enabler_id", "KM")).upper()
    enabler_name_full = SEAM_ENABLER_MAP.get(enabler_id, f"Unknown Enabler ({enabler_id})")
    
    final_sub_results = sub_results_full
    
    # --- 4. การจัดการชื่อไฟล์และการกรองข้อมูลสำหรับโหมด 'sub' ---
    
    base_prefix = os.path.basename(args.output_path)
    if not base_prefix or base_prefix == "SEAM_Comprehensive_Report":
        base_prefix = f"{enabler_id}_Comprehensive_Report"
        
    if args.mode == "sub" and args.sub:
        sub_id_filter = args.sub.upper()
        print(f"🔹 โหมด: รายงานเฉพาะเกณฑ์ย่อย {sub_id_filter} สำหรับ {enabler_name_full}")
        
        # กรอง Sub Results
        final_sub_results = [
            res for res in sub_results_full 
            if res.get("sub_criteria_id", "").upper() == sub_id_filter
        ]
        
        # กรอง Raw Data 
        if raw_data_for_section5:
            raw_data_for_section5 = [
                stmt for stmt in raw_data_for_section5 
                if stmt.get("sub_criteria_id", "").upper() == sub_id_filter
            ]
            
        base_prefix = f"{enabler_id}_Report_{sub_id_filter}"
    
    else:
        print(f"🔹 โหมด: รายงานฉบับเต็มสำหรับ {enabler_name_full}")
        
    
    # กำหนดชื่อ Output สุดท้าย (รวมวันที่)
    final_base_name = f"{base_prefix}_{REPORT_DATE}"
    
    # ** ในเวอร์ชันนี้ จะสร้าง 2 ไฟล์ DOCX **
    strategic_path = os.path.join(output_dir, f"{final_base_name}_Strategic.docx")
    detail_path = os.path.join(output_dir, f"{final_base_name}_RawDetails.docx")

    # --- A. การสร้างไฟล์ DOCX ---
    
    # 1. สร้าง Strategic Report (Sections 1-4)
    if not final_sub_results:
        print("🚨 ไม่สามารถสร้าง Strategic Report ได้เนื่องจากไม่มีข้อมูล Sub-Criteria ที่กรองแล้ว")
        if args.mode == "sub":
             print(f"   (ข้อผิดพลาด: ไม่พบข้อมูลสำหรับเกณฑ์ย่อย {args.sub})")
        # ยังคงไปขั้นตอน 2 เพื่อลองสร้าง Raw Details (เผื่อมีข้อมูล)
    else:
        print(f"\nกำลังสร้างไฟล์ DOCX [Strategic Report]...")
        strategic_doc = Document()
        setup_document(strategic_doc) 
        
        # SECTION 1: Overall Summary 
        generate_overall_summary_docx(strategic_doc, summary_section, enabler_name_full) 
        # SECTION 2: Executive Summary
        generate_executive_summary_docx(strategic_doc, summary_section, final_sub_results)
        # SECTION 3: Sub-Criteria Status & Gap 
        gap_criteria_docx = generate_sub_criteria_status_docx(strategic_doc, final_sub_results)
        # SECTION 4: Action Plan Report
        generate_action_plan_report_docx(strategic_doc, gap_criteria_docx)

        strategic_doc.save(strategic_path)
        print(f"🎉 สร้างไฟล์ DOCX [Strategic Report] สำเร็จ! บันทึกที่: {strategic_path}")


    # 2. สร้าง Raw Details Working Document (Section 5)
    print(f"กำลังสร้างไฟล์ DOCX [Raw Details]...")
    detail_doc = Document()
    setup_document(detail_doc) 
    
    # SECTION 5: Raw Details (ส่ง List ของ Statements เข้าไปโดยตรง)
    generate_raw_details_report_docx(detail_doc, raw_data_for_section5)
    
    detail_doc.save(detail_path)
    print(f"🎉 สร้างไฟล์ DOCX [Raw Details] สำเร็จ! บันทึกที่: {detail_path}")

    print("\n✅ การสร้างรายงาน SEAM Assessment ทั้งหมดเสร็จสมบูรณ์")


if __name__ == "__main__":
    main()