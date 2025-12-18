import re
import logging
from typing import Dict, Optional, Any, List

# ดึงค่าคอนฟิกจาก global_vars
from config.global_vars import (
    ACTION_PLAN_LANGUAGE,
    SUPPORTED_ENABLERS,
    DEFAULT_ENABLER
)

logger = logging.getLogger(__name__)

# =============================
#    Intent Detection
# =============================
def detect_intent(question: str, doc_type: str = "document") -> Dict[str, Any]:
    """
    ตรวจจับ intent แม่นยำสูง รองรับภาษาไทย + ดึงข้อมูลจาก global_vars
    """
    q = question.strip().lower()

    intent = {
        "is_faq": False,
        "is_synthesis": False,
        "is_evidence": False,
        "sub_topic": None,
        "enabler_hint": None
    }

    # 1. Extract sub_topic (e.g. "KM 4.1" -> "KM-4.1")
    # ใช้ SUPPORTED_ENABLERS จาก global_vars มาสร้าง regex
    enabler_pattern = "|".join(SUPPORTED_ENABLERS).lower()
    sub_topic_match = re.search(fr"({enabler_pattern}|topic)\s*(?:topic\s*)?(\d+\.\d+)", q)
    
    if sub_topic_match:
        found_enabler = sub_topic_match.group(1).upper()
        # ถ้าเจอคำว่า 'topic' ให้ใช้ DEFAULT_ENABLER (เช่น KM)
        intent["enabler_hint"] = found_enabler if found_enabler != "TOPIC" else DEFAULT_ENABLER
        intent["sub_topic"] = f"{intent['enabler_hint']}-{sub_topic_match.group(2)}"

    # --------------------
    # 2. Intent Signals
    # --------------------
    
    # Synthesis/Compare
    synthesis_signals = [
        "เปรียบเทียบ", "ต่างกัน", "ความแตกต่าง", "ความต่าง", "เทียบ", "vs", "versus",
        "compare", "difference", "ต่างกันยังไง", "ต่างกันอย่างไร", "เทียบกับ",
        "สรุปความเหมือน", "สรุปความต่าง", "ไฮไลต์", "highlight"
    ]
    if any(word in q for word in synthesis_signals):
        intent["is_synthesis"] = True
        return intent
        
    # FAQ/Definition
    faq_signals = [
        "คืออะไร", "คือ", "อะไร", "ใคร", "เมื่อไร", "ที่ไหน", "อย่างไร", "ทำไม", "หมายถึง",
        "what", "who", "when", "where", "why", "how", "faq", "คือยังไง", "แปลว่า",
        "สรุป", "ภาพรวม"
    ]
    if any(sig in q for sig in faq_signals):
        intent["is_faq"] = True
        
    # Evidence/Detail (SEAM Context)
    evidence_signals = [
        "ตามเอกสาร", "ในเอกสาร", "เอกสารบอก", "หลักฐาน", "อ้างอิง", "source", "reference",
        "จากไฟล์", "ระบุแหล่ง", "อิงจาก", "ตามที่ระบุ", "ดำเนินการ", "รายงาน", "ผลลัพธ์",
        "ประเมิน", "คะแนน", "pdca", "เกณฑ์", "ระดับ", "รายละเอียด", "แยกย่อย"
    ]
    if any(sig in q for sig in evidence_signals):
        intent["is_evidence"] = True

    # --------------------
    # 3. Fallback Logic
    # --------------------
    if not any([intent["is_faq"], intent["is_synthesis"], intent["is_evidence"]]):
        if doc_type in ["seam", "evidence", "document"]:
            intent["is_evidence"] = True
        else:
            intent["is_faq"] = True 
        
    return intent


# =============================
#    Prompt Builder
# =============================
def build_prompt(context: str, question: str, intent: Dict[str, Any]) -> str:
    """
    สร้าง Prompt โดยใช้เงื่อนไขภาษาจาก global_vars.ACTION_PLAN_LANGUAGE
    """
    sections = []

    # 0. Language Instruction (ดึงจาก global_vars)
    lang_map = {"th": "ภาษาไทย (Thai)", "en": "English"}
    target_lang = lang_map.get(ACTION_PLAN_LANGUAGE, "ภาษาไทย (Thai)")
    sections.append(f"CRITICAL: Always respond in {target_lang} only.")

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
    else:
        sections.append("หมายเหตุ: ไม่พบข้อมูลที่เกี่ยวข้องในฐานข้อมูล")

    # 3. คำถามผู้ใช้
    sections.append(f"คำถาม:\n{question.strip()}")

    # 4. กฎการตอบเฉพาะเจาะจง
    if intent["is_synthesis"]:
        sections.append("""
รูปแบบคำตอบที่ต้องการ:
• ใช้หัวข้อชัดเจน เช่น "ความเหมือน", "ความแตกต่าง", "ข้อสรุป"
• อ้างอิงแหล่งที่มาเสมอ เช่น (Source: filename.pdf)
• ตอบเป็นข้อ ๆ หรือตารางเพื่อความชัดเจน
""")
    elif intent["is_evidence"]:
        sections.append("""
กฎสำคัญ:
• ทุกข้อมูลที่ใช้ ต้องระบุแหล่งที่มาในวงเล็บท้ายประโยค เช่น (Source: filename.pdf)
• หากคำถามระบุปี แต่ในเอกสารเป็นปีที่ใกล้เคียงที่สุด ให้ใช้ข้อมูลนั้นตอบและระบุปีที่พบจริงอย่างชัดเจน
• หากไม่พบข้อมูล ให้ตอบตรงๆ ว่า "ไม่พบข้อมูลที่เกี่ยวข้องในเอกสาร"
• สำหรับรายละเอียดเกณฑ์ 5 ระดับ: แยกเป็น Bullet points 1-5 โดยใช้ข้อความจากเอกสารเท่านั้น
""")
    else:
        sections.append("กฎสำคัญ: ตอบให้กระชับ สุภาพ และอ้างอิงข้อมูลจากเอกสารเท่านั้น")

    # 5. Subtopic Focus (ถ้ามี)
    if intent.get("sub_topic"):
        sections.append(f"**[FOCUS AREA]** คำถามนี้เน้นเฉพาะหัวข้อ: {intent['sub_topic']} ห้ามนำข้อมูลหัวข้ออื่นมาปน")

    return "\n\n".join(sections)