# generate_report.py — Full Multi-Tenant Support Version
import json
import os
import sys
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT

def generate_pdf_report(json_path: str):
    if not os.path.exists(json_path):
        print(f"ไม่พบไฟล์: {json_path}")
        return

    # โหลด JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    summary = data.get('summary', {})
    sub_results = data.get('sub_criteria_results', [])
    
    if not sub_results:
        print("ไม่พบข้อมูลผลการประเมินใน JSON")
        return
    
    result = sub_results[0]  # สำหรับ single sub-criteria
    
    # ดึงข้อมูลจาก JSON แทน hardcode
    tenant_name = summary.get('tenant', 'ไม่ระบุ').upper()
    year = summary.get('year', 'ไม่ระบุ')
    enabler = summary.get('enabler', 'ไม่ระบุ').upper()
    sub_id = result.get('sub_id', 'ไม่ระบุ')
    sub_name = result.get('sub_criteria_name', 'ไม่ระบุ')
    highest_level = summary.get('highest_pass_level', 0)
    achieved_score = summary.get('achieved_weight', 0.0)
    total_weight = summary.get('total_weight', 4.0)
    percentage = summary.get('overall_percentage', 0.0)

    # ชื่อไฟล์ PDF
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_name = f"รายงาน_SEAM_{tenant_name}_{year}_{enabler}_{sub_id}_{timestamp}.pdf"
    pdf_path = os.path.join(os.path.dirname(json_path), pdf_name)

    doc = SimpleDocTemplate(pdf_path, pagesize=A4, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()
    
    # สไตล์ภาษาไทย
    styles.add(ParagraphStyle(name='TitleTH', fontName='Helvetica-Bold', fontSize=22, alignment=TA_CENTER, spaceAfter=30, leading=28))
    styles.add(ParagraphStyle(name='HeadingTH', fontName='Helvetica-Bold', fontSize=16, spaceAfter=12, leading=20))
    styles.add(ParagraphStyle(name='NormalTH', fontName='Helvetica', fontSize=12, leading=18, spaceAfter=10))
    styles.add(ParagraphStyle(name='SmallTH', fontName='Helvetica', fontSize=10, leading=14))

    story = []

    # หน้าปก
    story.append(Paragraph("รายงานผลการประเมินความพร้อม", styles['TitleTH']))
    story.append(Paragraph("ตามเกณฑ์ SE-AM (State Enterprise Assessment Model)", styles['TitleTH']))
    story.append(Spacer(1, 40))
    story.append(Paragraph(f"องค์กร: {tenant_name}", styles['HeadingTH']))
    story.append(Paragraph(f"ปีงบประมาณ: {year}", styles['NormalTH']))
    story.append(Paragraph(f"Enabler: {enabler}", styles['NormalTH']))
    story.append(Paragraph(f"เกณฑ์: {sub_name} ({sub_id})", styles['NormalTH']))
    story.append(Paragraph(f"วันที่จัดทำรายงาน: {datetime.now().strftime('%d %B %Y')}", styles['NormalTH']))
    story.append(PageBreak())

    # สรุปผู้บริหาร
    story.append(Paragraph("สรุปผลผู้บริหาร", styles['HeadingTH']))
    story.append(Paragraph(f"<b>ระดับสูงสุดที่ผ่าน:</b> Level {highest_level}", styles['NormalTH']))
    story.append(Paragraph(f"<b>คะแนนที่ได้รับ:</b> {achieved_score:.2f} / {total_weight:.2f} คะแนน ({percentage:.2f}%)", styles['NormalTH']))
    story.append(Spacer(1, 20))

    # ตารางผลแต่ละ Level
    table_data = [["ระดับ", "ผลลัพธ์", "คะแนน", "สรุปเหตุผล"]]
    levels = result.get('levels', {})
    for level in sorted(levels.keys(), key=int):
        lvl = levels[level]
        status = "🟢 ผ่าน" if lvl.get('passed') else "🔴 ไม่ผ่าน"
        score = lvl.get('score', 0)
        reason = lvl.get('summary', 'ไม่มีข้อมูล').replace('<br/>', ' ')[:150]
        table_data.append([f"Level {level}", status, f"{score:.1f}", reason + ("..." if len(reason) > 150 else "")])

    table = Table(table_data, colWidths=[2.5*cm, 3*cm, 2*cm, 9*cm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 12),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE')
    ]))
    story.append(table)
    story.append(Spacer(1, 30))

    # Action Plan
    action_plan = result.get('action_plan', [])
    if action_plan:
        story.append(Paragraph("แผนปฏิบัติการยกระดับ", styles['HeadingTH']))
        for i, phase in enumerate(action_plan, 1):
            story.append(Paragraph(f"ระยะที่ {i}: {phase.get('Phase', 'ไม่ระบุ')}", styles['NormalTH']))
            story.append(Paragraph(f"เป้าหมาย: {phase.get('Goal', '')}", styles['NormalTH']))
            for j, action in enumerate(phase.get('Actions', []), 1):
                story.append(Paragraph(f"  • {j}. {action.get('Recommendation', '')}", styles['NormalTH']))
                steps = action.get('Steps', [])
                for step in steps:
                    story.append(Paragraph(f"     - ขั้นตอน {step.get('Step')}: {step.get('Description')}", styles['SmallTH']))
                    story.append(Paragraph(f"       รับผิดชอบ: {step.get('Responsible', 'ไม่ระบุ')}", styles['SmallTH']))

    # สร้าง PDF
    doc.build(story)
    print(f"✅ สร้างรายงาน PDF สำเร็จ: {pdf_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_report.py <path_to_json_file>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    generate_pdf_report(json_file)