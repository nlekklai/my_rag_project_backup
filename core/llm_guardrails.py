# core/llm_guardrails.py
import re
from typing import Dict, Optional

# =============================
#    Intent Detection (ฉลาด + แม่นสุด ๆ)
# =============================
def detect_intent(question: str, doc_type: str = "document") -> Dict[str, bool]:
    """
    ตรวจจับ intent ได้แม่นยำสูงมาก รองรับภาษาไทยเต็มรูปแบบ + คำพูดจริงของคน
    """
    q = question.strip().lower()

    intent = {
        "is_faq": False,
        "is_synthesis": False,
        "is_evidence": False
    }

    # --------------------
    # 1. Intent จาก Keyword + Pattern matching (Priority สูงสุด)
    # --------------------
    
    # Synthesis/Compare Signals (Priority 1: ชัดเจน ไม่ควรมี Fallback)
    synthesis_signals = [
        "เปรียบเทียบ", "ต่างกัน", "ความแตกต่าง", "ความต่าง", "เทียบ", "vs", "versus",
        "compare", "difference", "ต่างกันยังไง", "ต่างกันอย่างไร", "เทียบกับ",
        "สรุปความเหมือน", "สรุปความต่าง", "ไฮไลต์", "highlight"
    ]
    
    if any(word in q for word in synthesis_signals):
        intent["is_synthesis"] = True
        return intent # Synthesis มี priority สูงสุด
        
    # FAQ/Definition Signals (Priority 2: ถ้าไม่ใช่ Synthesis)
    faq_signals = [
        "คืออะไร", "คือ", "อะไร", "ใคร", "เมื่อไร", "ที่ไหน", "อย่างไร", "ทำไม", "หมายถึง",
        "what ", "who ", "when ", "where ", "why ", "how ", "faq", "คือยังไง", "แปลว่า", "แปลว่าอะไร",
        "สรุป", "ภาพรวม" # เพิ่มคำว่า "สรุป" และ "ภาพรวม" เพื่อจับคำถามระดับสูง/นิยาม
    ]
    if any(sig in q for sig in faq_signals):
        intent["is_faq"] = True
        
    # Evidence/Detail signals (Priority 3: ถ้าไม่ใช่ Synthesis หรือ FAQ)
    # เพิ่มคำที่บ่งชี้การค้นหารายละเอียดเชิงลึก หรือการประเมิน
    evidence_signals = [
        "ตามเอกสาร", "ในเอกสาร", "เอกสารบอก", "หลักฐาน", "อ้างอิง", "source", "reference",
        "จากไฟล์", "ระบุแหล่ง", "อิงจาก", "ตามที่ระบุ", "ดำเนินการ", "รายงาน", "ผลลัพธ์",
        "ประเมิน", "คะแนน", "PDCA" # เพิ่มคำที่เกี่ยวข้องกับ SEAM/KM Evidence
    ]
    if any(sig in q for sig in evidence_signals):
        intent["is_evidence"] = True

    # --------------------
    # 2. Intent จาก doc_type (Default/Fallback)
    # --------------------
    
    # Fallback Logic:
    
    # 2.1 ถ้าเป็น Doc Type ที่เน้น FAQ (และไม่ใช่ Synthesis) ให้เป็น FAQ เสมอ
    if doc_type in ["faq"] and not intent["is_synthesis"]:
        intent["is_faq"] = True
        intent["is_evidence"] = False # ล้าง Evidence/Detail ที่อาจถูกจับได้
        
    # 2.2 ถ้าเป็น Doc Type ที่เน้น Evidence/Detail/Document
    elif doc_type in ["document", "evidence", "seam"]:
        # ถ้ายังไม่มี Intent หลักถูกจับได้ (ไม่ใช่ Synthesis/FAQ ที่ชัดเจน)
        if not intent["is_synthesis"] and not intent["is_faq"]:
            intent["is_evidence"] = True
            
    # 2.3 Fallback สุดท้าย: ถ้ายังไม่มี Intent ใดถูกจับได้เลย
    if not any(intent.values()):
        # ให้เป็น FAQ (Default สำหรับการตอบแบบสรุป) ดีกว่าการเป็น Evidence (ที่ต้องอ้างอิง)
        intent["is_faq"] = True 
        
    return intent


# =============================
#    Prompt Builder
# =============================
def build_prompt(context: str, question: str, intent: Dict[str, bool]) -> str:
    sections = []

    # 1. บทบาทหลัก
    if intent["is_synthesis"]:
        role = ("คุณคือผู้เชี่ยวชาญด้านการวิเคราะห์และเปรียบเทียบเอกสารอย่างละเอียด "
                "โปรดตอบอย่างเป็นระบบ ระบุความเหมือน ความต่าง และข้อสรุปที่ชัดเจน")
    elif intent["is_faq"]:
        role = ("คุณคือผู้ช่วยที่ตอบคำถามแบบสรุป (FAQ/Summary) ให้กระชับ อ่านง่าย ใช้ภาษาเป็นมิตร "
                "โปรดใช้ข้อมูลจากเอกสารที่ให้มาเท่านั้น")
    else:
        role = ("คุณคือผู้ช่วยวิเคราะห์ที่ตอบคำถามโดยยึดหลักฐานจากเอกสารเท่านั้น "
                "ห้ามแต่งข้อมูลเพิ่ม ห้ามสรุปเกินกว่าที่มี")

    sections.append(role)

    # 2. ข้อมูลอ้างอิง
    if context.strip():
        sections.append(f"ข้อมูลจากเอกสาร:\n{context}")

    # 3. คำถามผู้ใช้
    sections.append(f"คำถาม:\n{question.strip()}")

    # 4. กฎการตอบเฉพาะเจาะจง
    if intent["is_synthesis"]:
        sections.append("""
รูปแบบคำตอบที่ต้องการ:
• ใช้หัวข้อชัดเจน เช่น "ความเหมือน", "ความแตกต่าง", "ข้อสรุป"
• อ้างอิงแหล่งที่มาเสมอ เช่น (Source 1: SEAM 101.pdf)
• ตอบเป็นข้อ ๆ อ่านง่าย
""")
    elif intent["is_evidence"]:
        sections.append("""
กฎสำคัญ:
• ทุกข้อมูลที่ใช้ ต้องระบุแหล่งที่มาในวงเล็บท้ายประโยค เช่น (Source 2)
• **[CRITICAL OVERRIDE]** หากคำถามระบุปี (เช่น 2568) แต่หลักฐานที่ดึงมามีเนื้อหาคล้ายกันแต่ระบุปีที่ใกล้เคียงที่สุด (เช่น 2567) **คุณต้องใช้ Context นั้นตอบ** โดยตอบนโยบายปีที่พบในเอกสาร (2567) และ **ต้องระบุปีที่พบใน Context** อย่างชัดเจนในคำตอบ (เช่น 'ข้อมูลนี้เป็นของปี 2567...')
• ถ้าไม่พบข้อมูลในเอกสารที่ให้มา ให้ตอบว่า "ไม่พบข้อมูลในเอกสารที่เกี่ยวข้อง"
• ห้ามเดา ห้ามแต่งข้อมูล
""")
    else: # is_faq:
        sections.append("""
กฎสำคัญ:
• ตอบกระชับและสุภาพ
• ใช้ข้อมูลจากเอกสารที่ให้มาเท่านั้น
• ห้ามเดา ห้ามแต่งข้อมูล
""")

    return "\n\n".join(sections)