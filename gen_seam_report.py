# gen_seam_report.py (โค้ดฉบับแก้ไข: เพิ่ม PDCA Grouping และเน้น Actionability)

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

# พจนานุกรมสำหรับชื่อ Enabler ฉบับเต็ม (ใช้เป็น Fallback)
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
    (รองรับโครงสร้างใหม่: sub_criteria_results -> raw_results_ref และโครงสร้างเดิม)
    """
    statements = []
    
    # *** โครงสร้างใหม่ที่ต้องรองรับ: ดึงจาก 'sub_criteria_results' -> 'raw_results_ref' ***
    if isinstance(raw_data_dict, dict) and 'sub_criteria_results' in raw_data_dict:
        sub_results = raw_data_dict.get('sub_criteria_results', [])
        for sub_item in sub_results:
            if isinstance(sub_item, dict) and 'raw_results_ref' in sub_item:
                raw_statements = sub_item['raw_results_ref']
                if isinstance(raw_statements, list):
                    statements.extend(raw_statements)
        # ถ้าดึงได้จากโครงสร้างใหม่นี้ ให้ return เลย
        if statements:
            return statements
            
    # โครงสร้างเดิม 1: {"Assessment_Details": {"2.2": [...], ...}}
    details = raw_data_dict.get("Assessment_Details") if isinstance(raw_data_dict, dict) else None
    if isinstance(details, dict):
        for sub_id_statements in details.values():
            if isinstance(sub_id_statements, list):
                statements.extend(sub_id_statements)
    # โครงสร้างเดิม 2: List ของ Statements ตรงๆ
    elif isinstance(raw_data_dict, list):
        statements = raw_data_dict
        
    return statements

# ==========================
# 3. REPORT GENERATION FUNCTIONS (DOCX - Native to New Structure)
# ==========================

def generate_overall_summary_docx(document: Document, summary_data: Dict[str, Any], enabler_name_full: str): 
    """สร้างส่วนสรุปผลการประเมินโดยรวม (Overall) [SECTION 1] ใน DOCX"""
    
    set_heading(document, f'[SECTION 1] สรุปผลการประเมิน {enabler_name_full} โดยรวม', level=1)
    
    # 🎯 FIX 1: ดึงข้อมูลจาก 'summary' โดยใช้คีย์ที่แก้ไขใน core/seam_assessment.py
    achieved_score = summary_data.get('final_score_achieved', 0.0)
    # ใช้ overall_enabler_max_score เพื่อสะท้อนคะแนนเต็ม 40 (สำหรับ KM)
    overall_max_score = summary_data.get('overall_enabler_max_score', 0.0) 

    # Percentage calculation based on the Overall Enabler Max Score (40)
    overall_percent = (achieved_score / overall_max_score) * 100 if overall_max_score > 0 else 0.0

    # Maturity Score = (Achieved Score / Overall Enabler Max Score) * 5 (สมมติว่า 5 คือ Level สูงสุด)
    maturity_score = (achieved_score / overall_max_score) * 5 if overall_max_score > 0 else 0.0
    
    table = document.add_table(rows=4, cols=2) 
    table.style = 'Table Grid'
    
    def add_summary_row(row_index, label, value):
        table.cell(row_index, 0).text = label
        table.cell(row_index, 1).text = value
        table.cell(row_index, 0).paragraphs[0].runs[0].font.bold = True
        table.cell(row_index, 1).paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
    
    add_summary_row(0, "ตัวขับเคลื่อน (Enabler):", f"{summary_data.get('enabler', '-')}\n({enabler_name_full})") 
    # ใช้ achieved_score / overall_max_score
    add_summary_row(1, "คะแนนรวมถ่วงน้ำหนักที่ได้:", f"{achieved_score:.2f} / {overall_max_score:.2f}")
    # ใช้ overall_percent
    add_summary_row(2, "เปอร์เซ็นต์ความคืบหน้าโดยรวม:", f"{overall_percent:.2f}%")
    add_summary_row(3, "คะแนนวุฒิภาวะโดยรวม (Maturity Score):", f"{maturity_score:.2f}")
    
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

    # 🎯 FIX 2: ใช้ achieved_score และ overall_max_score เพื่อคำนวณตามคะแนนเต็ม 40
    achieved_score = summary_data.get('final_score_achieved', 0.0)
    overall_max_score = summary_data.get('overall_enabler_max_score', 0.0)
    
    overall_percent = (achieved_score / overall_max_score) * 100 if overall_max_score > 0 else 0.0
    maturity_score = (achieved_score / overall_max_score) * 5 if overall_max_score > 0 else 0.0

    add_paragraph(document, f"✅ คะแนนรวม: {achieved_score:.2f} / {overall_max_score:.2f}")
    add_paragraph(document, f"✅ ร้อยละความสำเร็จ: {overall_percent:.2f}%")
    add_paragraph(document, f"✅ ระดับความเป็นผู้ใหญ่: {maturity_score:.2f}")
    document.add_paragraph()

    if sub_results:
        # Strength: Top 3 highest scoring
        add_paragraph(document, "📈 จุดแข็งที่โดดเด่น (Top Strengths):", bold=True, color=(0x00, 0x70, 0xC0))
        top_strengths = sorted(sub_results, key=lambda x: x.get("weighted_score", 0), reverse=True)[:3]
        for s in top_strengths:
            sub_id = s.get('sub_criteria_id', 'N/A')
            sub_name = s.get('sub_criteria_name', 'N/A')
            add_paragraph(document, f"• {sub_id} - {sub_name} ({s.get('weighted_score', 0):.2f}/{s.get('weight', 0):.2f})", style="List Bullet")

        document.add_paragraph()
        
        # Weakness: Top 3 with Gap (or lowest scoring with Gap)
        add_paragraph(document, "🚨 จุดที่ควรพัฒนา (Development Areas):", bold=True, color=(0xFF, 0x00, 0x00))
        gaps = [s for s in sub_results if not s.get("target_level_achieved", True)]
        # เรียงตาม Level ที่ผ่านได้ต่ำสุด (Highest Full Level)
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
    table.columns[0].width = Inches(0.5) # ID
    table.columns[1].width = Inches(4.5) # ชื่อเกณฑ์ย่อย
    table.columns[2].width = Inches(0.7) # คะแนน
    table.columns[3].width = Inches(0.7) # Level
    table.columns[4].width = Inches(1.0) # Gap
    
    header_cells = table.rows[0].cells
    headers = ["ID", "ชื่อเกณฑ์ย่อย", "คะแนน", "Level", "Gap"]
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
        level = info.get('highest_full_level', 0)
        has_gap = "❌ YES" if not info.get('target_level_achieved', True) else "✅ NO"
        
        if not info.get('target_level_achieved', True):
            gap_criteria[sub_id] = info # เก็บ info ทั้งหมด
            
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

def generate_action_plan_report_docx(document: Document, gap_criteria: Dict[str, Any]):
    """สร้างรายงานรายละเอียดแผนปฏิบัติการ (Action Plan) [SECTION 4] ใน DOCX (รวม L4/L5 Summary และจัดกลุ่มตาม PDCA)"""
    
    document.add_heading('[SECTION 4] รายงานรายละเอียดแผนปฏิบัติการเพื่อปิดช่องว่าง (Action Plan)', level=1)
    
    if not gap_criteria:
        add_paragraph(document, "✅ ทุกเกณฑ์ย่อยผ่านครบถ้วนแล้ว ไม่จำเป็นต้องมี Action Plan เพิ่มเติม")
        return

    # เพิ่มย่อหน้าแนะนำเพื่อเพิ่ม Actionability
    add_paragraph(document, 
        "ℹ️ คำแนะนำเหล่านี้เป็นการระบุ 'ช่องว่างเชิงระบบ' (Systemic Gap) เพื่อให้องค์กรบรรลุวุฒิภาวะ Level ถัดไป องค์กรควรนำคำแนะนำเหล่านี้ไปแตกเป็น 'แผนปฏิบัติการย่อย (Detailed Work Plan)' ที่ระบุรายละเอียดกิจกรรม, ผู้รับผิดชอบ, และไทม์ไลน์ ภายใต้กรอบ PDCA ที่กำหนดไว้",
        italic=True,
        color=(0x80, 0x80, 0x80) # สีเทาสำหรับข้อแนะนำ
    )
    document.add_paragraph()

    for sub_id, sub_info in gap_criteria.items():
        sub_name = sub_info.get('sub_criteria_name', 'N/A')
        
        document.add_heading(f"• เกณฑ์ย่อย {sub_id}: {sub_name} (Highest Full Level: L{sub_info.get('highest_full_level', 0)})", level=2)
        
        # --- ดึง L4/L5 Summary (โค้ดเดิม) ---
        evidence_summary_L5 = sub_info.get("evidence_summary_L5", {})
        evidence_summary_L4 = sub_info.get("evidence_summary_L4", {})
        
        if evidence_summary_L5.get('summary'):
            add_paragraph(document, "💡 ข้อมูลเชิงลึกหลักฐานระดับ L5 (เป้าหมายสูงสุด):", bold=True, color=(0x00, 0x70, 0xC0))
            add_paragraph(document, f"   - สรุปหลักฐาน: {evidence_summary_L5.get('summary', 'ไม่มีสรุป L5')}", italic=True)
            # เน้นข้อเสนอแนะหลัก
            p_sugg = add_paragraph(document, f"   - ข้อเสนอแนะ: ", italic=True)
            run_sugg = p_sugg.add_run(evidence_summary_L5.get('suggestion_for_next_level', 'ไม่มีข้อเสนอแนะ'))
            run_sugg.font.bold = True
            run_sugg.font.name = THAI_FONT_NAME
            document.add_paragraph()
        
        if evidence_summary_L4.get('summary'):
            add_paragraph(document, "💡 ข้อมูลเชิงลึกหลักฐานระดับ L4:", bold=True, color=(0x00, 0x70, 0xC0))
            add_paragraph(document, f"   - สรุปหลักฐาน: {evidence_summary_L4.get('summary', 'ไม่มีสรุป L4')}", italic=True)
            document.add_paragraph()
        # --- END L4/L5 Summary ---
        
        # --- START: จัดกลุ่ม Action Plan ตาม PDCA (User Guideline) ---
        action_plans = sub_info.get('action_plan', [])
        
        # โครงสร้างสำหรับจัดเก็บ Action Plan ตาม PDCA
        pdca_actions = {
            'P (Plan / วางแผน)': [],
            'D (Do / ปฏิบัติ)': [],
            'C (Check / ตรวจสอบ)': [],
            'A (Act / ปรับปรุง)': []
        }

        # รวบรวมและจัดกลุ่ม Action Plan ทั้งหมด
        for plan_phase in action_plans:
            actions_list = plan_phase.get('Actions', [])
            for action in actions_list:
                rec = action.get('Recommendation', '')
                failed_level = action.get('Failed_Level', 5)
                target_evidence = action.get('Target_Evidence_Type', '-')
                key_metric = action.get('Key_Metric', '-')
                
                # Logic การแมปเข้าสู่ PDCA
                pdca_key = 'D (Do / ปฏิบัติ)' # กำหนดค่าเริ่มต้นเป็น D

                if 'ปรับปรุง' in rec or 'ข้อมูลป้อนกลับ' in rec or failed_level == 5:
                    # ถ้าล้มเหลวที่ L5 หรือมีคำว่าปรับปรุง/ข้อมูลป้อนกลับ จะเป็น Act
                    pdca_key = 'A (Act / ปรับปรุง)'
                elif 'ติดตาม' in rec or 'ประเมินผล' in rec or 'ทบทวน' in rec:
                    # ถ้ามีคำว่า ติดตาม/ประเมินผล/ทบทวน จะจัดอยู่ใน Check
                    pdca_key = 'C (Check / ตรวจสอบ)'
                elif failed_level in [1, 2] or 'กำหนดแผน' in rec or 'กำหนดกลยุทธ์' in rec:
                    # ถ้าล้มเหลวที่ระดับต้นๆ หรือเน้นคำว่าวางแผน จะเป็น Plan
                    pdca_key = 'P (Plan / วางแผน)' 
                else:
                    # ที่เหลือส่วนใหญ่มักเป็น D (การสร้าง การดำเนินการตามปกติ)
                    pdca_key = 'D (Do / ปฏิบัติ)'

                pdca_actions[pdca_key].append({
                    'rec': rec, 
                    'target_evidence': target_evidence,
                    'key_metric': key_metric
                })
        
        # แสดงผลใน Report ตามหมวด PDCA
        if any(pdca_actions.values()):
            document.add_paragraph()
            add_paragraph(document, "📋 แผนปฏิบัติการเพื่อปิดช่องว่างตามวงจร PDCA (User Guideline)", bold=True, color=(0x00, 0x00, 0x00))

            for phase, actions in pdca_actions.items():
                if actions:
                    document.add_heading(f"--- {phase} ---", level=4)
                    
                    # สร้างตาราง Action Plan
                    action_table = document.add_table(rows=1, cols=3, style='Table Grid')
                    
                    # ตั้งค่า Header
                    header_cells = action_table.rows[0].cells
                    header_cells[0].text = "คำแนะนำ (Recommendation)"
                    header_cells[1].text = "หลักฐานเป้าหมาย (Evidence Type)"
                    header_cells[2].text = "ตัวชี้วัดสำคัญ (Key Metric)"
                    
                    # Format Header
                    for cell in action_table.rows[0].cells:
                         cell.paragraphs[0].runs[0].font.bold = True
                         cell.paragraphs[0].runs[0].font.name = THAI_FONT_NAME 
                    
                    # เพิ่ม Action Rows
                    for action in actions:
                        row_cells = action_table.add_row().cells
                        
                        # Column 1: Recommendation
                        row_cells[0].text = action['rec']
                        
                        # Column 2: Target Evidence (เน้นด้วยสีน้ำเงินและตัวหนา)
                        p_evidence = row_cells[1].paragraphs[0]
                        run_evidence = p_evidence.add_run(action['target_evidence'])
                        run_evidence.font.bold = True
                        run_evidence.font.color.rgb = RGBColor(0x00, 0x70, 0xC0) # Blue
                        
                        # Column 3: Key Metric (เน้นด้วยสีแดงและตัวหนา)
                        p_metric = row_cells[2].paragraphs[0]
                        run_metric = p_metric.add_run(action['key_metric'])
                        run_metric.font.bold = True
                        run_metric.font.color.rgb = RGBColor(0xFF, 0x00, 0x00) # Red
                        
                        # กำหนดฟอนต์สำหรับแถวข้อมูลทั้งหมด
                        for cell in row_cells:
                            for p_cell in cell.paragraphs:
                                for run in p_cell.runs:
                                    run.font.name = THAI_FONT_NAME
                                    run.font.size = Pt(11) # กำหนดขนาดฟอนต์
                
            document.add_paragraph() 
        else:
            add_paragraph(document, ">>> [ข้อมูล]: ไม่มี Action Plan ถูกกำหนดไว้สำหรับเกณฑ์นี้", style='List Bullet')
        # --- END: จัดกลุ่ม Action Plan ตาม PDCA ---

def generate_raw_details_report_docx(document: Document, raw_data: Optional[Dict[str, Any]]): 
    """สร้างรายงานการตรวจสอบความถูกต้องเชิงลึก (Raw Details) [SECTION 5] ใน DOCX"""
    
    raw_data_base = raw_data 
    if raw_data is None:
        document.add_paragraph(f"⚠️ ไม่สามารถโหลดไฟล์ Raw Details ได้ หรือไฟล์ว่างเปล่า") 
        return
        
    assessment_details = {}
    
    # ดึงข้อมูล Raw Data และจัดกลุ่มตาม Sub-criteria ID
    if isinstance(raw_data_base, dict):
        assessment_details = raw_data_base.get('Assessment_Details', {})
    
    if not assessment_details:
         statements_list = flatten_raw_data(raw_data_base) # ลองดึงแบบ 'all statements'
         if statements_list:
             for statement in statements_list:
                sub_id = statement.get('sub_criteria_id', 'N/A')
                if sub_id not in assessment_details:
                    assessment_details[sub_id] = []
                assessment_details[sub_id].append(statement)
         else:
            add_paragraph(document, "ℹ️ ข้อมูล Raw Details ว่างเปล่าหลังจากตรวจสอบโครงสร้าง (โปรดตรวจสอบว่า 'raw_results_ref' มีข้อมูลหรือไม่)")
            return
    
    # โค้ดแสดงผลรายละเอียด
    document.add_heading('[SECTION 5] รายงานหลักฐานเชิงลึก (Raw Details)', level=1)
    
    for sub_id, statements in assessment_details.items():
        document.add_heading(f"รายละเอียดการประเมินเกณฑ์ย่อย: {sub_id}", level=2)
        
        # สร้างตารางสำหรับแต่ละ Sub-criteria (5 คอลัมน์)
        table = document.add_table(rows=1, cols=5, style='Table Grid')
        header_cells = table.rows[0].cells
        
        # Headers (เหลือ 5 คอลัมน์: ลบคอลัมน์ Statement / Standard ออก)
        headers = ["Statement ID (Level)", "ผลการประเมิน", "เหตุผล/วิเคราะห์", "แหล่งที่มา", "หลักฐาน/บริบท (Snippet)"] 
        for i, h in enumerate(headers):
            header_cells[i].text = h
            header_cells[i].paragraphs[0].runs[0].font.bold = True
            header_cells[i].paragraphs[0].runs[0].font.name = THAI_FONT_NAME 
            
        for statement in statements:
            # ใช้คีย์ที่คาดว่าจะมาจาก raw_results_ref: is_passed และ llm_score
            is_passed = statement.get('is_passed', statement.get('is_pass', False)) 
            status = "✅ PASS" if is_passed else "❌ FAIL"
            level = statement.get('level', '-')
            reason_text = statement.get('reason', 'N/A')

            # --- START: แก้ไขการดึง Source และ Location (Column 4 - Index 3) ให้รองรับโครงสร้าง JSON ที่ซ้อน metadata ---
            sources_list = statement.get('retrieved_full_source_info', [])
            
            valid_sources = []
            for src in sources_list:
                metadata = src.get('metadata', {}) # ดึง metadata ออกมา
                
                # 1. พยายามดึงชื่อ: file_name > source > source_file > uuid (จาก metadata)
                name = metadata.get('file_name') or metadata.get('source') or metadata.get('source_file') or metadata.get('uuid')
                
                # 2. พยายามดึงตำแหน่ง: page_label > page (จาก metadata)
                location = metadata.get('page_label') or metadata.get('page')
                
                # 3. สร้างข้อความ
                if name:
                    name_display = name
                    
                    location_str = None
                    if location is not None:
                        # ถ้า page เป็น 0 และไม่มี page_label (สันนิษฐานว่าเป็นหน้าแรก p.1)
                        if str(location) == '0' and not metadata.get('page_label'): 
                             location_str = "p.1"
                        # ถ้าเป็นค่าตัวเลข/ข้อความปกติ
                        elif str(location).strip() != "":
                            location_str = f"p.{location}"
                    
                    location_display = f"({location_str})" if location_str else ''
                    valid_sources.append(f"{name_display}{location_display}")
                
            sources_text = "\n".join(valid_sources) if valid_sources else 'ไม่มีแหล่งที่มา'
            # --- END: แก้ไขการดึง Source และ Location (New Fix) ---

            row_cells = table.add_row().cells
            
            # Column 1 (Index 0): Statement ID (Level)
            row_cells[0].text = f"{statement.get('statement_id', '-')}\n(L{level})"
            
            # Column 2 (Index 1): ผลการประเมิน (Status)
            llm_score = statement.get('llm_score', '-')
            if llm_score != '-':
                # llm_score คือ 0/1; คะแนนเต็มจริงคือ 5 แต่เนื่องจากเราไม่ทราบระดับคะแนนต่อข้อที่นี่ จึงแสดงเป็น 0/1
                row_cells[1].text = f"{status}\n({llm_score}/1)" 
            else:
                 row_cells[1].text = status
            
            # Column 3 (Index 2): เหตุผล/วิเคราะห์ (Reason)
            row_cells[2].text = reason_text 
            
            # Column 4 (Index 3): แหล่งที่มา (Source)
            row_cells[3].text = sources_text 
            
            # Column 5 (Index 4): หลักฐาน/บริบท (Snippet)
            row_cells[4].text = statement.get('aggregated_context_used', 'ไม่มีหลักฐานสนับสนุนที่ชัดเจน')
            
            # กำหนดฟอนต์สำหรับแถวข้อมูล
            for cell in row_cells:
                for p_cell in cell.paragraphs:
                    for run in p_cell.runs:
                        run.font.name = THAI_FONT_NAME

            row_cells[1].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            if not is_passed:
                row_cells[1].paragraphs[0].runs[0].font.bold = True 

        document.add_paragraph()

# ==========================
# 4. MAIN EXECUTION (Updated to accept 1 input file)
# ==========================
def main():
    """ฟังก์ชันหลักในการสร้างรายงานทั้งหมด"""
    
    parser = argparse.ArgumentParser(description="Generate Comprehensive SEAM Assessment Reports (New Structure V2).")
    parser.add_argument("--mode", choices=["all", "sub"], default="all", help="all: Generate full report. sub: Generate report for a specific sub-criteria.")
    parser.add_argument("--sub", type=str, help="SubCriteria ID (e.g., 2.2) if mode=sub.")
    
    # *** เปลี่ยนเป็นรับไฟล์เดียว ***
    parser.add_argument("--input_file", type=str, required=True, help="Path to the single JSON file containing both summary and raw details.") 
    
    # *** ไม่มีการรับ --raw_file แล้ว ***
    parser.add_argument("--output_path", type=str, default="reports/SEAM_Comprehensive_Report", help="Output directory and base filename prefix (e.g., reports/SEAM_Comprehensive_Report).")
    
    args = parser.parse_args()
    
    # 1. จัดการ Folder Output และแยก Directory
    output_dir = os.path.dirname(args.output_path)
    if not output_dir:
         output_dir = EXPORT_DIR # ใช้ Default หาก Path ไม่มี Directory
    setup_output_folder(output_dir)
    
    # 2. โหลดไฟล์ (โหลดเพียงครั้งเดียว)
    full_data = load_data(args.input_file, "Comprehensive Input Data")
    
    if not full_data or 'summary' not in full_data or 'sub_criteria_results' not in full_data:
        print("🚨 ไม่สามารถสร้างรายงานได้เนื่องจากไฟล์ Input Core Data ไม่พร้อมหรือโครงสร้างไม่ถูกต้อง (ขาด 'summary' หรือ 'sub_criteria_results')")
        return
        
    # *** Assign ตัวแปรทั้งสองไปยังข้อมูลที่โหลดมาเดียวกัน ***
    full_summary_data = full_data
    raw_data = full_data
    
    # --- 3. ดึงและกรองข้อมูลหลัก ---
    summary_section = full_summary_data["summary"]
    sub_results_full = full_summary_data["sub_criteria_results"]
    
    # ดึง ENABLER และกำหนดค่าเริ่มต้น
    enabler_id = summary_section.get("enabler", "KM").upper() 
    enabler_name_full = SEAM_ENABLER_MAP.get(enabler_id, f"Unknown Enabler ({enabler_id})")
    
    final_sub_results = sub_results_full
    final_raw_data = raw_data
    
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
        if raw_data is not None:
            # *** FIX: flatten_raw_data ถูกแก้ไขให้ดึงจาก 'raw_results_ref' ได้แล้ว ***
            all_statements = flatten_raw_data(raw_data)
            filtered_statements = [
                stmt for stmt in all_statements 
                if stmt.get("sub_criteria_id", "").upper() == sub_id_filter
            ]
            
            # ถ้า Raw Data ที่กรองแล้วมีข้อมูล ให้สร้างโครงสร้าง Dict ใหม่เพื่อให้ generate_raw_details_report_docx ทำงานได้
            if filtered_statements:
                 # สร้างโครงสร้างใหม่: {"Assessment_Details": {sub_id_filter: [...]}}
                 final_raw_data = {"Assessment_Details": {sub_id_filter: filtered_statements}}
            else:
                 final_raw_data = None
            
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
    print(f"\nกำลังสร้างไฟล์ DOCX [Strategic Report]...")
    strategic_doc = Document()
    setup_document(strategic_doc) 
    
    # SECTION 1: Overall Summary
    generate_overall_summary_docx(strategic_doc, summary_section, enabler_name_full) 
    # SECTION 2: Executive Summary
    generate_executive_summary_docx(strategic_doc, summary_section, final_sub_results)
    # SECTION 3: Sub-Criteria Status & Gap
    gap_criteria_docx = generate_sub_criteria_status_docx(strategic_doc, final_sub_results)
    # SECTION 4: Action Plan Report (พร้อม L4/L5 Summary) -> โค้ดได้รับการปรับปรุง PDCA
    generate_action_plan_report_docx(strategic_doc, gap_criteria_docx)

    strategic_doc.save(strategic_path)
    print(f"🎉 สร้างไฟล์ DOCX [Strategic Report] สำเร็จ! บันทึกที่: {strategic_path}")


    # 2. สร้าง Raw Details Working Document (Section 5)
    print(f"กำลังสร้างไฟล์ DOCX [Raw Details]...")
    detail_doc = Document()
    setup_document(detail_doc) 
    detail_doc.add_heading(f"[SECTION 5] รายงานหลักฐานเชิงลึก (Raw Details) - {enabler_name_full} ({REPORT_DATE})", level=1)
    # SECTION 5: Raw Details 
    generate_raw_details_report_docx(detail_doc, final_raw_data) 

    detail_doc.save(detail_path)
    print(f"🎉 สร้างไฟล์ DOCX [Raw Details] สำเร็จ! บันทึกที่: {detail_path}")


if __name__ == "__main__":
    main()