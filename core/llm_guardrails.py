# core/llm_guardrails.py
import re
from typing import Dict

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

    # 1. จาก doc_type (priority สูง)
    if doc_type in ["faq", "qa", "question"]:
        intent["is_faq"] = True
        return intent
    if doc_type in ["document", "rubric", "evidence", "feedback"]:
        intent["is_evidence"] = True

    # 2. Keyword + Pattern matching (เรียงจากความสำคัญสูง → ต่ำ)
    synthesis_signals = [
        "เปรียบเทียบ", "ต่างกัน", "ความแตกต่าง", "ความต่าง", "เทียบ", "vs", "versus",
        "compare", "difference", "ต่างกันยังไง", "ต่างกันอย่างไร", "เทียบกับ", "กับ",
        "และ", "สรุปความเหมือน", "สรุปความต่าง", "ไฮไลต์", "highlight"
    ]

    # ถ้ามีคำว่า "กับ" หรือ "และ" แล้วคำถามไม่ยาวเกินไป → น่าจะเปรียบเทียบ
    has_compare_word = any(word in q for word in synthesis_signals)
    has_and_connector = bool(re.search(r"\b(กับ|และ|vs|versus)\b", q))
    is_short_compare = len(q.split()) <= 30

    if has_compare_word or (has_and_connector and is_short_compare):
        intent["is_synthesis"] = True
        return intent  # Synthesis มี priority สูงสุด

    # FAQ signals
    faq_signals = [
        "คืออะไร", "คือ", "อะไร", "ใคร", "เมื่อไร", "ที่ไหน", "อย่างไร", "ทำไม", "หมายถึง",
        "what ", "who ", "when ", "where ", "why ", "how ", "faq", "คือยังไง", "แปลว่า"
    ]
    if any(sig in q for sig in faq_signals):
        intent["is_faq"] = True

    # Evidence signals
    evidence_signals = [
        "ตามเอกสาร", "ในเอกสาร", "เอกสารบอก", "หลักฐาน", "อ้างอิง", "source", "reference",
        "จากไฟล์", "ระบุแหล่ง", "อิงจาก", "ตามที่ระบุ"
    ]
    if any(sig in q for sig in evidence_signals):
        intent["is_evidence"] = True

    return intent


# =============================
#    Prompt Builder (สวย + LLM ตอบตรงเป๊ะ)
# =============================
def build_prompt(context: str, question: str, intent: Dict[str, bool]) -> str:
    sections = []

    # 1. บทบาทหลัก
    if intent["is_synthesis"]:
        role = ("คุณคือผู้เชี่ยวชาญด้านการวิเคราะห์และเปรียบเทียบเอกสารอย่างละเอียด "
                "โปรดตอบอย่างเป็นระบบ ระบุความเหมือน ความต่าง และข้อสรุปที่ชัดเจน")
    elif intent["is_faq"]:
        role = "คุณคือผู้ช่วยที่ตอบคำถามแบบ FAQ ให้กระชับ อ่านง่าย ใช้ภาษาเป็นมิตร"
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
• ถ้าไม่พบข้อมูลในเอกสารที่ให้มา ให้ตอบว่า "ไม่พบข้อมูลในเอกสารที่เกี่ยวข้อง"
• ห้ามเดา ห้ามแต่งข้อมูล
""")
    else:
        sections.append("โปรดตอบให้กระชับ สุภาพ และเป็นธรรมชาติ")

    return "\n\n".join(sections)