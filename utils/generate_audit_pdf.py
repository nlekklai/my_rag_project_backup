# ==============================================================================
# โค้ดที่ต้องนำเข้าเพิ่ม (Import)
# ==============================================================================
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from datetime import datetime
import os
import re # นำเข้า regex สำหรับทำความสะอาดข้อความ

# ==============================================================================
# ฟังก์ชันหลักที่ปรับปรุง
# ==============================================================================

# กำหนด Styles
styles = getSampleStyleSheet()
style_normal = styles['Normal']
style_heading = styles['Heading2']
style_subheading = styles['h3']

# ฟังก์ชันทำความสะอาดข้อความจาก Markdown / JSON (บางครั้ง LLM output ก็มี \n หรือ quote)
def clean_text(text: str) -> str:
    if not text:
        return ""
    # แทนที่ \n ด้วยช่องว่าง และลบเครื่องหมาย quotation ที่ไม่จำเป็น
    text = text.replace('\n', ' ').strip()
    text = re.sub(r'["\']', '', text) 
    return text

def generate_audit_trail_pdf(result_data: dict, output_dir: str = "audit_trails"):
    """
    สร้างรายงาน Audit Trail ในรูปแบบ PDF โดยใช้ SimpleDocTemplate 
    เพื่อรองรับข้อความยาวและการขึ้นหน้าใหม่โดยอัตโนมัติ
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # วนซ้ำ sub_criteria_results ซึ่งคาดว่าเป็น List ตามโครงสร้าง Engine
    sub_criteria_list = result_data.get("sub_criteria_results", [])
    
    for sub_result in sub_criteria_list:
        sub_id = sub_result.get("sub_criteria_id", "N/A")
        
        for level_result in sub_result.get("level_results", []):
            level = level_result.get("level", 0)
            
            # กรองเฉพาะ Level ที่ PASS
            if not level_result.get("is_passed", False):
                continue
            
            # เตรียมไฟล์ PDF
            filename = f"{output_dir}/Audit_Trail_{sub_id}_L{level}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            doc = SimpleDocTemplate(
                filename, 
                pagesize=A4, 
                leftMargin=50, 
                rightMargin=50, 
                topMargin=50, 
                bottomMargin=50
            )
            Story = []
            
            # ==================== 1. Header และ Summary Table ====================
            Story.append(Paragraph(f"AUDIT TRAIL REPORT", styles['Title']))
            Story.append(Spacer(1, 12))
            
            # ข้อมูลสรุป
            summary_data = [
                ['Sub-Criteria:', sub_id],
                ['Level:', level],
                ['Statement:', level_result.get("statement", "N/A")],
                ['ผลการประเมิน:', 'PASS'],
                ['คะแนน:', f"{level_result.get('score', 0.0)}/100"],
                ['AI Confidence:', level_result.get('ai_confidence', 'N/A')],
                ['Strength Score:', f"{level_result.get('evidence_strength', 0.0)}/10"],
                ['วันที่สร้าง:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
            ]
            
            table_style = TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('GRID', (0,0), (-1,-1), 0.5, colors.black),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('BACKGROUND', (0,0), (0,-1), colors.lightgrey),
            ])
            
            table = Table(summary_data, colWidths=[120, 400])
            table.setStyle(table_style)
            Story.append(table)
            Story.append(Spacer(1, 24))
            
            # ==================== 2. LLM Rationale (คำอธิบายเหตุผล) ====================
            Story.append(Paragraph("2. LLM Rationale and Explanation", style_heading))
            Story.append(Spacer(1, 12))
            
            # ดึง Rationale จาก llm_result_full
            llm_result_full = level_result.get("llm_result_full", {})
            llm_explanation = llm_result_full.get("explanation", "ไม่พบคำอธิบายจาก LLM.")
            
            Story.append(Paragraph(clean_text(llm_explanation), style_normal))
            Story.append(Spacer(1, 24))
            
            # ==================== 3. Baseline Context Summary (หลักฐาน Level ก่อนหน้า) ====================
            Story.append(Paragraph(f"3. Baseline Context Summary (Evidence from L1 - L{level-1})", style_heading))
            Story.append(Spacer(1, 12))

            # ดึง Baseline Summary จาก llm_result_full
            baseline_summary = llm_result_full.get("baseline_summary_used", "ไม่มีหลักฐาน Level ก่อนหน้าใช้เป็นบริบท (หรือ L1).")
            
            Story.append(Paragraph(clean_text(baseline_summary), style_normal))
            Story.append(Spacer(1, 24))
            
            # ==================== 4. Direct Evidence References ====================
            Story.append(Paragraph("4. Direct Evidence References (Top 20 Used)", style_heading))
            Story.append(Spacer(1, 12))
            
            evidences = level_result.get("top_evidences_ref", [])[:20]
            
            if evidences:
                evidence_data = [
                    ['#', 'Filename', 'Document ID']
                ]
                for i, ev in enumerate(evidences):
                    # ใช้ os.path.basename เพื่อให้ชื่อไฟล์ไม่ยาวเกินไป
                    filename_display = os.path.basename(ev.get('filename', 'UNKNOWN_FILE'))
                    evidence_data.append([
                        i + 1, 
                        filename_display, 
                        ev.get('doc_id', 'N/A')
                    ])
                
                table_style_ev = TableStyle([
                    ('GRID', (0,0), (-1,-1), 0.5, colors.black),
                    ('BACKGROUND', (0,0), (-1,0), colors.fidblue),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0,0), (-1,0), 10),
                    ('FONTSIZE', (0,1), (-1,-1), 9),
                ])
                
                evidence_table = Table(evidence_data, colWidths=[20, 350, 140])
                evidence_table.setStyle(table_style_ev)
                Story.append(evidence_table)
            else:
                Story.append(Paragraph("ไม่พบหลักฐานที่ใช้ในการตัดสินใน Level นี้", style_normal))

            # สร้าง PDF
            doc.build(Story)
            print(f"Generated: {filename}")