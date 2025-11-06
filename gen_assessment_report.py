import json
import os
from typing import Dict, Any, Optional, List
# 🟢 Import libraries for DOCX generation
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL

# --- 1. CONFIGURATION ---
# ไฟล์ Input JSON ทั้งสองไฟล์ (อยู่ใน exports/)
# 🟢 แก้ไขให้ใช้ไฟล์ 'all' ทั้งคู่ ตามที่ผู้ใช้ยืนยัน
DATA_FILE_SUMMARY = "exports/KM_summary_all_20251106_142132.json" 
DATA_FILE_RAW = "exports/KM_raw_details_all_20251106_142132.json" 

# การตั้งค่า Output สำหรับทั้งสองรูปแบบ
OUTPUT_FILE_PATH_DOCX = "reports/KM_Comprehensive_Report.docx" 
OUTPUT_FILE_PATH_TXT = "reports/KM_Comprehensive_Report.txt"


# --- 2. DATA LOADING & UTILITY ---

def load_data(file_path: str, file_type: str) -> Optional[Dict[str, Any]]:
    """โหลดข้อมูลจากไฟล์ JSON และจัดการข้อผิดพลาด"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"❌ ข้อผิดพลาดในการโหลดไฟล์ {file_type} '{file_path}': {e}")
        return None

def setup_output_folder(file_path):
    """ตรวจสอบและสร้าง folder output"""
    output_dir = os.path.dirname(file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

# --- 3. REPORT GENERATION FUNCTIONS (DOCX) ---

def generate_overall_summary_docx(document: Document, data: Dict[str, Any]):
    """สร้างส่วนสรุปผลการประเมินโดยรวม (Overall) ใน DOCX"""
    overall = data.get("Overall", {})
    
    document.add_heading('[SECTION 1] สรุปผลการประเมินการจัดการความรู้ (KM) โดยรวม', level=1)
    
    table = document.add_table(rows=4, cols=2)
    table.style = 'Table Grid'
    
    def add_summary_row(row_index, label, value):
        table.cell(row_index, 0).text = label
        table.cell(row_index, 1).text = value
        table.cell(row_index, 0).paragraphs[0].runs[0].font.bold = True
        table.cell(row_index, 1).paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
    
    add_summary_row(0, "ตัวขับเคลื่อน (Enabler):", overall.get('enabler', '-'))
    add_summary_row(1, "คะแนนรวมถ่วงน้ำหนักที่ได้:", f"{overall.get('total_weighted_score', 0.0):.2f} / {overall.get('total_possible_weight', 0.0):.2f}")
    add_summary_row(2, "เปอร์เซ็นต์ความคืบหน้าโดยรวม:", f"{overall.get('overall_progress_percent', 0.0):.2f}%")
    add_summary_row(3, "คะแนนวุฒิภาวะโดยรวม (Maturity Score):", f"{overall.get('overall_maturity_score', 0.0):.2f}")
    
    document.add_paragraph() 

def generate_sub_criteria_status_docx(document: Document, data: Dict[str, Any]) -> Dict[str, Any]:
    """สร้างตารางสถานะการประเมินรายเกณฑ์ย่อยใน DOCX และคืนค่าเกณฑ์ที่มี Gap"""
    breakdown = data.get("SubCriteria_Breakdown", {})
    
    document.add_heading('[SECTION 2] สถานะการประเมินรายเกณฑ์ย่อยและ Gap', level=1)
    
    table = document.add_table(rows=1, cols=5)
    table.style = 'Table Grid'
    
    header_cells = table.rows[0].cells
    headers = ["ID", "ชื่อเกณฑ์ย่อย", "คะแนน", "Level", "Gap"]
    for i, h in enumerate(headers):
        header_cells[i].text = h
        header_cells[i].paragraphs[0].runs[0].font.bold = True
        header_cells[i].vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    gap_criteria = {}
    
    for sub_id, info in breakdown.items():
        row_cells = table.add_row().cells
        
        name = info.get('name', 'N/A')
        score = info.get('score', 0.0)
        level = info.get('highest_full_level', 0)
        has_gap = "❌ YES" if info.get('development_gap', False) else "✅ NO"
        
        if info.get('development_gap', False):
            gap_criteria[sub_id] = info
            
        row_cells[0].text = sub_id
        row_cells[1].text = name
        row_cells[2].text = f"{score:.2f}"
        row_cells[3].text = f"L{level}"
        row_cells[4].text = has_gap
        
        row_cells[0].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        row_cells[2].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        row_cells[3].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        row_cells[4].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    document.add_paragraph() 
    return gap_criteria

def generate_action_plan_report_docx(document: Document, data: Dict[str, Any], gap_criteria: Dict[str, Any]):
    """สร้างรายงานรายละเอียดแผนปฏิบัติการ (Action Plan) ใน DOCX"""
    action_plans = data.get("Action_Plans", {})
    
    document.add_heading('[SECTION 3] รายงานรายละเอียดแผนปฏิบัติการเพื่อปิดช่องว่าง (Action Plan)', level=1)
    
    if not gap_criteria:
        document.add_paragraph("✅ ทุกเกณฑ์ย่อยผ่านครบถ้วนแล้ว ไม่จำเป็นต้องมี Action Plan เพิ่มเติม")
        return

    for sub_id, sub_info in gap_criteria.items():
        sub_name = sub_info.get('name', 'N/A')
        
        document.add_heading(f"• เกณฑ์ย่อย {sub_id}: {sub_name} (Highest Full Level: L{sub_info.get('highest_full_level', 0)})", level=2)
        
        if sub_id in action_plans:
            
            for plan_phase in action_plans[sub_id]:
                phase = plan_phase.get('Phase', '-')
                goal = plan_phase.get('Goal', '-')
                actions_list = plan_phase.get('Actions', [])
                
                document.add_paragraph(f"🛠️ เฟส/ขั้นตอน: {phase}", style='List Bullet')
                document.add_paragraph(f"🎯 เป้าหมายหลัก: {goal}", style='List Bullet')

                if actions_list:
                    document.add_paragraph("แผนปฏิบัติการ:")
                    
                    action_table = document.add_table(rows=1, cols=3, style='Table Grid')
                    header_cells = action_table.rows[0].cells
                    header_cells[0].text = "คำแนะนำ (Recommendation)"
                    header_cells[1].text = "หลักฐานเป้าหมาย (Evidence Type)"
                    header_cells[2].text = "ตัวชี้วัดสำคัญ (Key Metric)"
                    
                    for cell in action_table.rows[0].cells:
                         cell.paragraphs[0].runs[0].font.bold = True
                    
                    for action in actions_list:
                        row_cells = action_table.add_row().cells
                        row_cells[0].text = action.get('Recommendation', '-')
                        row_cells[1].text = action.get('Target_Evidence_Type', '-')
                        row_cells[2].text = action.get('Key_Metric', '-')
                
                document.add_paragraph() 
        else:
            document.add_paragraph(">>> [ข้อมูล]: ไม่มี Action Plan ถูกกำหนดไว้ในส่วน Action_Plans", style='List Bullet')

def generate_raw_details_report_docx(document: Document, raw_data: Optional[Dict[str, Any]]):
    """สร้างรายงานการตรวจสอบความถูกต้องเชิงลึก (Raw Details) ใน DOCX"""
    
    document.add_heading('[SECTION 4] รายงานการตรวจสอบความถูกต้องเชิงลึก (Raw Details)', level=1)
    
    # 🟢 FIX: จัดการกับโครงสร้าง Raw Data ที่เป็น List/Dict
    raw_data_base = raw_data # เก็บตัวแปรต้นฉบับไว้
    if isinstance(raw_data, list) and raw_data:
        raw_data = raw_data[0] # พยายามดึง Dict ตัวแรก
    elif isinstance(raw_data, list) and not raw_data:
        raw_data = None
    # 🟢 END FIX
    
    if raw_data is None:
        document.add_paragraph(f"⚠️ ไม่สามารถโหลดไฟล์ Raw Details ได้ ({DATA_FILE_RAW}) หรือไฟล์ว่างเปล่า")
        document.add_paragraph("⚠️ หากต้องการส่วนนี้ โปรดตรวจสอบความพร้อมของไฟล์ Raw Data")
        return
        
    # โครงสร้างสำหรับไฟล์ KM_raw_details_all...json คือการมองหาคีย์ 'Assessment_Details'
    assessment_details = raw_data.get('Assessment_Details', {})
    
    if not assessment_details:
        document.add_paragraph("ℹ️ ข้อมูล Raw Details ถูกโหลดแล้ว แต่ส่วน 'Assessment_Details' ว่างเปล่า")
        document.add_paragraph("ℹ️ หรือไฟล์ Raw Details ที่โหลดไม่มีคีย์ 'Assessment_Details' ซึ่งอาจเป็น Raw Detail ของเกณฑ์เดียว")
        document.add_paragraph()
        
        # 🟢 NEW: หากไม่มีคีย์ 'Assessment_Details' อาจเป็นเพราะมันเป็น List ของ Statements ตรงๆ
        statements_list = []
        if isinstance(raw_data_base, list) and all(isinstance(item, dict) and 'statement_id' in item for item in raw_data_base):
             # กรณีที่ Raw Data เป็น List ของ Statements ตรงๆ
             statements_list = raw_data_base
             
             # สร้างข้อมูล Assessment_Details จำลองจาก List เพื่อให้โค้ดส่วนแสดงผลทำงานได้
             # Assumption: ทุก statement ใน list นี้เป็น sub_criteria_id เดียวกัน (จาก KM_raw_details_1.1...)
             sub_id = statements_list[0].get('sub_criteria_id', 'N/A')
             assessment_details[sub_id] = statements_list
        else:
            return # หากไม่ใช่ทั้งสองโครงสร้าง ก็จบ

    # โค้ดแสดงผลรายละเอียด
    for sub_id, statements in assessment_details.items():
        document.add_heading(f"รายละเอียดการประเมินเกณฑ์ย่อย: {sub_id}", level=2)
        
        # สร้างตารางสำหรับแต่ละ Sub-criteria
        table = document.add_table(rows=1, cols=4, style='Table Grid')
        header_cells = table.rows[0].cells
        headers = ["Statement ID (Level)", "ผลการประเมิน", "เกณฑ์มาตรฐาน (Standard)", "หลักฐาน/บริบท (Snippet)"]
        for i, h in enumerate(headers):
            header_cells[i].text = h
            header_cells[i].paragraphs[0].runs[0].font.bold = True
            
        for statement in statements:
            # ใช้ 'is_pass' สำหรับไฟล์ 'all' หรือ 'pass_status' สำหรับไฟล์เกณฑ์เดียว
            status = "✅ PASS" if statement.get('is_pass', statement.get('pass_status', False)) else "❌ FAIL"
            level = statement.get('level', '-')
            
            row_cells = table.add_row().cells
            
            row_cells[0].text = f"{statement.get('statement_id', '-')}\n(L{level})"
            row_cells[1].text = status
            row_cells[2].text = statement.get('standard', 'N/A')
            row_cells[3].text = statement.get('snippet_for_display', 'ไม่มีหลักฐานสนับสนุนที่ชัดเจน')

            # จัดรูปแบบเซลล์ผลการประเมิน
            row_cells[1].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            if not statement.get('is_pass', statement.get('pass_status', False)):
                row_cells[1].paragraphs[0].runs[0].font.bold = True # ทำตัวหนาถ้า Fail

        document.add_paragraph() # เพิ่มพื้นที่ว่างหลังตาราง

# --- 4. REPORT GENERATION FUNCTIONS (TXT) ---
# ฟังก์ชันสำหรับ TXT (สร้างรายการข้อความ)

def generate_overall_summary_txt(data: Dict[str, Any], report_lines: List[str]):
    """สร้างส่วนสรุปผลการประเมินโดยรวม (Overall) สำหรับ TXT"""
    overall = data.get("Overall", {})
    
    report_lines.append("="*80)
    report_lines.append("          [SECTION 1] สรุปผลการประเมินการจัดการความรู้ (KM) โดยรวม")
    report_lines.append("="*80)
    report_lines.append(f"ตัวขับเคลื่อน (Enabler):        {overall.get('enabler', '-')}")
    report_lines.append(f"คะแนนรวมถ่วงน้ำหนักที่ได้:     {overall.get('total_weighted_score', 0.0):.2f} / {overall.get('total_possible_weight', 0.0):.2f}")
    report_lines.append(f"เปอร์เซ็นต์ความคืบหน้าโดยรวม:  {overall.get('overall_progress_percent', 0.0):.2f}%")
    report_lines.append(f"คะแนนวุฒิภาวะโดยรวม (Maturity Score): {overall.get('overall_maturity_score', 0.0):.2f}")
    report_lines.append("="*80)

def generate_sub_criteria_status_txt(data: Dict[str, Any], report_lines: List[str]) -> Dict[str, Any]:
    """สร้างตารางสถานะการประเมินรายเกณฑ์ย่อยสำหรับ TXT และคืนค่าเกณฑ์ที่มี Gap"""
    breakdown = data.get("SubCriteria_Breakdown", {})
    
    report_lines.append("\n" + "#"*80)
    report_lines.append("          [SECTION 2] สถานะการประเมินรายเกณฑ์ย่อยและ Gap")
    report_lines.append("#"*80)
    
    header_format = "{:<5} | {:<50} | {:<5} | {:<7} | {:<10}"
    separator = "-"*80
    
    report_lines.append(separator)
    report_lines.append(header_format.format("ID", "ชื่อเกณฑ์ย่อย", "คะแนน", "Level", "Gap"))
    report_lines.append(separator)
    
    gap_criteria = {}
    
    for sub_id, info in breakdown.items():
        name = info.get('name', 'N/A')
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
    """สร้างรายงานรายละเอียดแผนปฏิบัติการ (Action Plan) สำหรับ TXT"""
    action_plans = data.get("Action_Plans", {})
    
    report_lines.append("\n" + "*"*90)
    report_lines.append("       [SECTION 3] รายงานรายละเอียดแผนปฏิบัติการเพื่อปิดช่องว่าง (Action Plan)")
    report_lines.append("*"*90)

    if not gap_criteria:
        report_lines.append("✅ ทุกเกณฑ์ย่อยผ่านครบถ้วนแล้ว ไม่จำเป็นต้องมี Action Plan เพิ่มเติม")
        return
        
    for sub_id, sub_info in gap_criteria.items():
        sub_name = sub_info.get('name', 'N/A')
        
        report_lines.append(f"\n[เกณฑ์ย่อย {sub_id}: {sub_name}] (Highest Full Level: L{sub_info.get('highest_full_level', 0)})")
        report_lines.append("-" * (len(sub_name) + 15))

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
    """สร้างรายงานการตรวจสอบความถูกต้องเชิงลึก (Raw Details) สำหรับ TXT"""
    
    report_lines.append("\n" + "="*80)
    report_lines.append("       [SECTION 4] รายงานการตรวจสอบความถูกต้องเชิงลึก (Raw Details)")
    report_lines.append("="*80)

    # 🟢 FIX: จัดการกับโครงสร้าง Raw Data ที่เป็น List/Dict
    raw_data_base = raw_data # เก็บตัวแปรต้นฉบับไว้
    if isinstance(raw_data, list) and raw_data:
        raw_data = raw_data[0] # พยายามดึง Dict ตัวแรก
    elif isinstance(raw_data, list) and not raw_data:
        raw_data = None
    # 🟢 END FIX
    
    if raw_data is None:
        report_lines.append(f"⚠️ ไม่สามารถโหลดไฟล์ Raw Details ได้ ({DATA_FILE_RAW}) หรือไฟล์ว่างเปล่า")
        report_lines.append("⚠️ หากต้องการส่วนนี้ โปรดตรวจสอบความพร้อมของไฟล์ Raw Data")
        report_lines.append("="*80)
        return

    # พยายามดึงจากโครงสร้างหลักที่ใช้คีย์ 'Assessment_Details'
    assessment_details = raw_data.get('Assessment_Details', {})
    
    if not assessment_details:
        report_lines.append("ℹ️ ข้อมูล Raw Details ถูกโหลดแล้ว แต่ส่วน 'Assessment_Details' ว่างเปล่า")
        report_lines.append("ℹ️ หรือไฟล์ Raw Details ที่โหลดไม่มีคีย์ 'Assessment_Details' ซึ่งอาจเป็น Raw Detail ของเกณฑ์เดียว")
        
        # 🟢 NEW: หากไม่มีคีย์ 'Assessment_Details' อาจเป็นเพราะมันเป็น List ของ Statements ตรงๆ
        statements_list = []
        if isinstance(raw_data_base, list) and all(isinstance(item, dict) and 'statement_id' in item for item in raw_data_base):
             # กรณีที่ Raw Data เป็น List ของ Statements ตรงๆ (เช่น KM_raw_details_1.1...)
             statements_list = raw_data_base
             
             # สร้างข้อมูล Assessment_Details จำลองจาก List เพื่อให้โค้ดส่วนแสดงผลทำงานได้
             sub_id = statements_list[0].get('sub_criteria_id', 'N/A')
             assessment_details[sub_id] = statements_list
        else:
            return # หากไม่ใช่ทั้งสองโครงสร้าง ก็จบ
    
    # โค้ดแสดงผลรายละเอียด
    for sub_id, statements in assessment_details.items():
        report_lines.append(f"\n=======================================================")
        report_lines.append(f"| รายละเอียดการประเมินเกณฑ์ย่อย: {sub_id} |")
        report_lines.append(f"=======================================================")
        
        for statement in statements:
            # ใช้ 'is_pass' สำหรับไฟล์ 'all' หรือ 'pass_status' สำหรับไฟล์เกณฑ์เดียว
            status = "✅ PASS" if statement.get('is_pass', statement.get('pass_status', False)) else "❌ FAIL"
            level = statement.get('level', '-')
            snippet = statement.get('snippet_for_display', 'ไม่มีหลักฐานสนับสนุนที่ชัดเจน')
            
            report_lines.append(f"\n[Statement ID: {statement.get('statement_id', '-')}] (Level {level}) - {status}")
            report_lines.append(f"  - เกณฑ์มาตรฐาน (Standard): {statement.get('standard', 'N/A')}")
            # แสดง snippet ไม่เกิน 150 ตัวอักษร
            report_lines.append(f"  - หลักฐาน/บริบท (Snippet): {snippet[:150]}{'...' if len(snippet) > 150 else ''}") 
            
    report_lines.append("\n" + "="*80)


# --- 5. MAIN EXECUTION ---
def main():
    """ฟังก์ชันหลักในการสร้างรายงานทั้งหมด"""
    
    # 1. จัดการ Folder Output
    setup_output_folder(OUTPUT_FILE_PATH_DOCX)
    
    # 2. โหลดไฟล์
    summary_data = load_data(DATA_FILE_SUMMARY, "Summary Data")
    raw_data = load_data(DATA_FILE_RAW, "Raw Details Data") 
    
    if summary_data:
        print(f"✅ เริ่มสร้างรายงานฉบับสมบูรณ์จาก {DATA_FILE_SUMMARY}")

        # --- A. การสร้างไฟล์ DOCX ---
        if os.path.exists(OUTPUT_FILE_PATH_DOCX):
            os.remove(OUTPUT_FILE_PATH_DOCX)
            
        document = Document()
        
        # รัน DOCX Generation
        generate_overall_summary_docx(document, summary_data)
        gap_criteria_docx = generate_sub_criteria_status_docx(document, summary_data)
        generate_action_plan_report_docx(document, summary_data, gap_criteria_docx)
        generate_raw_details_report_docx(document, raw_data) 
        
        # บันทึกไฟล์ DOCX
        document.save(OUTPUT_FILE_PATH_DOCX)
        print(f"🎉 สร้างไฟล์ DOCX สำเร็จ! บันทึกที่: {OUTPUT_FILE_PATH_DOCX}")

        # --- B. การสร้างไฟล์ TXT ---
        if os.path.exists(OUTPUT_FILE_PATH_TXT):
            os.remove(OUTPUT_FILE_PATH_TXT)
            
        txt_report_lines = []
        
        # รัน TXT Generation
        generate_overall_summary_txt(summary_data, txt_report_lines)
        gap_criteria_txt = generate_sub_criteria_status_txt(summary_data, txt_report_lines)
        generate_action_plan_report_txt(summary_data, gap_criteria_txt, txt_report_lines)
        generate_raw_details_report_txt(raw_data, txt_report_lines) 
        
        # บันทึกไฟล์ TXT
        with open(OUTPUT_FILE_PATH_TXT, 'w', encoding='utf-8') as f:
            f.write('\n'.join(txt_report_lines))
        
        print(f"🎉 สร้างไฟล์ TXT สำเร็จ! บันทึกที่: {OUTPUT_FILE_PATH_TXT}")

    else:
        print("🚨 ไม่สามารถสร้างรายงานได้เนื่องจากไฟล์ Summary Core Data ไม่พร้อม")

if __name__ == "__main__":
    main()