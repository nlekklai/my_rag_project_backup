# -*- coding: utf-8 -*-
# core/llm_guardrails.py

import re
import logging
from typing import Dict, Optional, Any, List

# ดึงค่าคอนฟิกจาก global_vars
from config.global_vars import (
    ACTION_PLAN_LANGUAGE,
    SUPPORTED_ENABLERS,
    DEFAULT_ENABLER,
    PDCA_ANALYSIS_SIGNALS,  # ตัวแปรใหม่ที่เพิ่มใน global_vars
    ANALYSIS_FRAMEWORK      # ตัวแปรใหม่ที่เพิ่มใน global_vars
)

logger = logging.getLogger(__name__)

# =================================================================
# 1. INTENT DETECTION (DETECTING WHAT USER WANTS)
# =================================================================
def detect_intent(question: str, doc_type: str = "document") -> Dict[str, Any]:
    """
    ตรวจจับความต้องการของผู้ใช้ (Intent) เพื่อเลือก Prompt ที่เหมาะสม
    รองรับ: FAQ, Synthesis, Evidence Search, และ PDCA Analysis
    """
    q = question.strip().lower()

    intent = {
        "is_faq": False,
        "is_synthesis": False,
        "is_evidence": False,
        "is_analysis": False,  # บ่งบอกว่าต้องการให้ "วิเคราะห์" (เช่น PDCA)
        "sub_topic": None,
        "enabler_hint": None
    }

    # --- A. Extract Enabler & Sub-topic (e.g., KM 1.1) ---
    enabler_pattern = "|".join(SUPPORTED_ENABLERS).lower()
    sub_topic_match = re.search(fr"({enabler_pattern}|topic)\s*(?:topic\s*)?(\d+\.\d+)", q)
    
    if sub_topic_match:
        found_enabler = sub_topic_match.group(1).upper()
        intent["enabler_hint"] = found_enabler if found_enabler != "TOPIC" else DEFAULT_ENABLER
        intent["sub_topic"] = f"{intent['enabler_hint']}-{sub_topic_match.group(2)}"

    # --- B. Check Analysis Intent (PDCA Inquiry) ---
    # ใช้สัญญาณจาก global_vars เพื่อตัดสินใจว่าเป็นคำถามเชิงวิเคราะห์หรือไม่
    if any(sig in q for sig in PDCA_ANALYSIS_SIGNALS):
        intent["is_analysis"] = True
        return intent # Priority สูงสุดสำหรับการวิเคราะห์หลักฐาน

    # --- C. Check Synthesis/Comparison Intent ---
    synthesis_signals = [
        "เปรียบเทียบ", "ต่างกัน", "ความแตกต่าง", "ความต่าง", "เทียบ", "vs", "versus",
        "compare", "difference", "สรุปความเหมือน", "สรุปความต่าง"
    ]
    if any(word in q for word in synthesis_signals):
        intent["is_synthesis"] = True
        return intent
        
    # --- D. Check FAQ/Definition Intent ---
    faq_signals = [
        "คืออะไร", "คือ", "อะไร", "ใคร", "หมายถึง", "คือยังไง", "แปลว่า", "สรุปภาพรวม"
    ]
    if any(sig in q for sig in faq_signals):
        intent["is_faq"] = True
        
    # --- E. Check Evidence/Source Intent ---
    evidence_signals = [
        "ตามเอกสาร", "ในเอกสาร", "หลักฐาน", "อ้างอิง", "source", "reference",
        "จากไฟล์", "ระบุแหล่ง", "อิงจาก", "คะแนน", "ระดับ"
    ]
    if any(sig in q for sig in evidence_signals):
        intent["is_evidence"] = True

    # --- F. Fallback Logic ---
    if not any([intent["is_faq"], intent["is_synthesis"], intent["is_evidence"], intent["is_analysis"]]):
        if doc_type in ["seam", "evidence", "document"]:
            intent["is_evidence"] = True
        else:
            intent["is_faq"] = True 
        
    return intent


# =================================================================
# 2. PROMPT BUILDER (CRAFTING THE FINAL PROMPT)
# =================================================================
def build_prompt(context: str, question: str, intent: Dict[str, Any]) -> str:
    """
    สร้าง Prompt ตาม Intent ที่ตรวจจับได้ พร้อมแทรก Logic การวิเคราะห์ PDCA
    """
    sections = []

    # 0. Language Instruction
    lang_map = {"th": "ภาษาไทย (Thai)", "en": "English"}
    target_lang = lang_map.get(ACTION_PLAN_LANGUAGE, "ภาษาไทย (Thai)")
    sections.append(f"CRITICAL: Always respond in {target_lang} only.")

    # 1. Role Assignment (กำหนดบทบาท)
    if intent["is_analysis"]:
        role = (f"คุณคือผู้เชี่ยวชาญด้านการตรวจสอบคุณภาพหลักฐานตามกรอบ {ANALYSIS_FRAMEWORK} "
                "โปรดวิเคราะห์ข้อมูลที่ได้รับอย่างละเอียดและระบุความครบถ้วนของกระบวนการ")
    elif intent["is_synthesis"]:
        role = ("คุณคือผู้เชี่ยวชาญด้านการวิเคราะห์และเปรียบเทียบเอกสาร "
                "ระบุความเหมือน ความต่าง และข้อสรุปอย่างเป็นระบบ")
    elif intent["is_faq"]:
        role = ("คุณคือผู้ช่วยตอบคำถามสรุป (FAQ) ที่กระชับและเข้าใจง่าย "
                "ใช้ข้อมูลจากเอกสารที่ให้มาเท่านั้น")
    else:
        role = ("คุณคือผู้ช่วยวิเคราะห์ที่ตอบคำถามโดยยึดหลักฐานจากเอกสารอย่างเคร่งครัด "
                "ห้ามแต่งข้อมูลเพิ่มเอง")
    sections.append(role)

    # 2. Context Information
    if context.strip():
        sections.append(f"ข้อมูลอ้างอิงจากฐานข้อมูล:\n{context}")
    else:
        sections.append("หมายเหตุ: ไม่พบข้อมูลที่เกี่ยวข้องโดยตรงในระบบ")

    # 3. User Question
    sections.append(f"คำถามที่ต้องตอบ:\n{question.strip()}")

    # 4. Response Guidelines (กฎการตอบ)
    if intent["is_analysis"]:
        # 
        sections.append(f"""
คำแนะนำในการวิเคราะห์แบบ {ANALYSIS_FRAMEWORK}:
1. P (Plan): มีการระบุแผน วัตถุประสงค์ หรือเป้าหมายชัดเจนหรือไม่?
2. D (Do): มีหลักฐานการปฏิบัติงานจริง หรือบันทึกกิจกรรมหรือไม่?
3. C (Check): มีการวัดผล ติดตามผล หรือสรุปผลสำเร็จเทียบกับเป้าหมายหรือไม่?
4. A (Act): มีการนำผลมาปรับปรุง (Lesson Learned) หรือแนวทางแก้ไขปัญหาหรือไม่?

**ให้สรุปว่าส่วนใด "มี" และส่วนใด "ขาด" พร้อมคำแนะนำในการเพิ่มหลักฐานให้ครบถ้วน**
""")
    elif intent["is_synthesis"]:
        sections.append("""
รูปแบบคำตอบ:
- แยกเป็นหัวข้อ "ความเหมือน", "ความแตกต่าง", "ข้อสรุป"
- อ้างอิงชื่อไฟล์ (Source: filename) เสมอ
""")
    elif intent["is_evidence"]:
        sections.append("""
กฎสำคัญ:
- ระบุชื่อไฟล์แหล่งอ้างอิงท้ายประโยคเสมอ เช่น (Source: filename.pdf)
- หากข้อมูลเป็นเกณฑ์ระดับ 1-5 ให้ตอบเป็น Bullet points โดยใช้ข้อความจริงจากเอกสาร
- หากไม่พบข้อมูล ให้ตอบว่า "ไม่พบข้อมูลที่เกี่ยวข้องในเอกสาร"
""")
    else:
        sections.append("กฎสำคัญ: ตอบอย่างสุภาพ กระชับ และใช้ข้อมูลจากเอกสารเท่านั้น")

    # 5. Focus Topic
    if intent.get("sub_topic"):
        sections.append(f"**[FOCUS]** คำถามนี้เกี่ยวข้องกับหัวข้อ: {intent['sub_topic']}")

    return "\n\n".join(sections)