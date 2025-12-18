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
    PDCA_ANALYSIS_SIGNALS,  
    ANALYSIS_FRAMEWORK      
)

logger = logging.getLogger(__name__)

# =================================================================
# 1. INTENT DETECTION (DETECTING WHAT USER WANTS)
# =================================================================
def detect_intent(question: str, doc_type: str = "document", contextual_rules: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    ตรวจจับความต้องการของผู้ใช้ (Intent) พร้อมใช้ Contextual Rules ช่วยเพิ่มความแม่นยำ
    """
    q = question.strip().lower()

    intent = {
        "is_faq": False,
        "is_synthesis": False,
        "is_evidence": False,
        "is_analysis": False,  
        "sub_topic": None,
        "enabler_hint": None
    }

    # --- A. Extract Enabler & Sub-topic (e.g., KM 1.1) ---
    enabler_pattern = "|".join(SUPPORTED_ENABLERS).lower()
    # ปรับ Regex ให้ครอบคลุมรูปแบบการพิมพ์ที่หลากหลาย
    sub_topic_match = re.search(fr"({enabler_pattern}|topic)\s*(?:topic\s*)?(\d+\.\d+)", q)
    
    if sub_topic_match:
        found_enabler = sub_topic_match.group(1).upper()
        intent["enabler_hint"] = found_enabler if found_enabler != "TOPIC" else DEFAULT_ENABLER
        intent["sub_topic"] = sub_topic_match.group(2) # เก็บเฉพาะเลขข้อ เช่น "1.1"

    # --- B. Check Analysis Intent (PDCA Inquiry) ---
    # ใช้ทั้ง Global Signals และ Keywords จาก Contextual Rules
    analysis_keywords = list(PDCA_ANALYSIS_SIGNALS) if PDCA_ANALYSIS_SIGNALS else ["plan", "do", "check", "act", "คะแนน", "ระดับ"]
    
    # ถ้ามี Contextual Rules ของข้อนั้นๆ ให้ดึง Keyword P-D-C-A มาช่วย Detect
    if intent["sub_topic"] and contextual_rules:
        rules = contextual_rules.get(intent["sub_topic"], {})
        for stage in ["Plan", "Do", "Check", "Act"]:
            analysis_keywords.extend([k.lower() for k in rules.get(stage, [])])

    if any(sig in q for sig in analysis_keywords):
        intent["is_analysis"] = True
        return intent 

    # --- C. Check Synthesis/Comparison Intent ---
    synthesis_signals = ["เปรียบเทียบ", "ต่างกัน", "ความแตกต่าง", "เทียบ", "vs", "compare", "diff"]
    if any(word in q for word in synthesis_signals):
        intent["is_synthesis"] = True
        return intent
        
    # --- D. Check FAQ/Definition Intent ---
    faq_signals = ["คืออะไร", "คือ", "หมายถึง", "แปลว่า", "สรุปภาพรวม"]
    if any(sig in q for sig in faq_signals):
        intent["is_faq"] = True
        
    # --- E. Check Evidence/Source Intent ---
    evidence_signals = ["หลักฐาน", "อ้างอิง", "source", "จากไฟล์", "ขอดูไฟล์"]
    if any(sig in q for sig in evidence_signals):
        intent["is_evidence"] = True

    # --- F. Fallback ---
    if not any([intent["is_faq"], intent["is_synthesis"], intent["is_evidence"], intent["is_analysis"]]):
        intent["is_evidence"] = True if doc_type in ["seam", "evidence"] else intent.update({"is_faq": True}) or True
        
    return intent


# =================================================================
# 2. PROMPT BUILDER (CRAFTING THE FINAL PROMPT)
# =================================================================
def build_prompt(context: str, question: str, intent: Dict[str, Any], contextual_rules: Dict[str, Any] = None) -> str:
    """
    สร้าง Prompt ตาม Intent และแทรกเกณฑ์จาก Contextual Rules เข้าไปในแนวทางการวิเคราะห์
    """
    sections = []

    # 0. Language Instruction
    sections.append("CRITICAL: Always respond in THAI language only.")

    # 1. Role Assignment
    if intent["is_analysis"]:
        role = f"คุณคือผู้เชี่ยวชาญด้านการตรวจสอบคุณภาพหลักฐานตามกรอบ {ANALYSIS_FRAMEWORK} และเกณฑ์ SE-AM"
    elif intent["is_synthesis"]:
        role = "คุณคือผู้เชี่ยวชาญด้านการวิเคราะห์เปรียบเทียบข้อมูล"
    else:
        role = "คุณคือผู้ช่วยที่ตอบคำถามโดยอิงจากเอกสารอ้างอิงอย่างเคร่งครัด"
    sections.append(role)

    # 2. Contextual Guidelines (กฎเฉพาะของ Enabler/Sub-topic)
    if intent["sub_topic"] and contextual_rules:
        rules = contextual_rules.get(intent["sub_topic"])
        if rules:
            rule_text = f"### เกณฑ์เฉพาะสำหรับข้อ {intent['sub_topic']}:\n"
            for stage in ["Plan", "Do", "Check", "Act"]:
                keywords = rules.get(stage, [])
                if keywords:
                    rule_text += f"- **{stage}**: ควรมีหลักฐานเกี่ยวกับ {', '.join(keywords)}\n"
            sections.append(rule_text)

    # 3. Reference Context
    sections.append(f"ข้อมูลอ้างอิงจากระบบ:\n{context if context.strip() else '--- ไม่พบข้อมูลที่เกี่ยวข้อง ---'}")

    # 4. User Question
    sections.append(f"คำถามของผู้ใช้: {question.strip()}")

    # 5. Specific Guidelines based on Intent
    if intent["is_analysis"]:
        sections.append(f"""
### คำแนะนำในการวิเคราะห์ ({ANALYSIS_FRAMEWORK}):
1. วิเคราะห์ว่าจากข้อมูลที่มี มีส่วนใดที่ตรงกับเกณฑ์ P, D, C หรือ A บ้าง
2. **จุดแข็ง**: ระบุหลักฐานที่พบชัดเจนพร้อมชื่อไฟล์
3. **สิ่งที่ขาด**: ระบุส่วนที่ยังไม่มีหลักฐานสนับสนุนตามเกณฑ์เฉพาะข้างต้น
4. **ข้อเสนอแนะ**: แนะนำประเภทเอกสารที่ควรเพิ่มเพื่อให้ได้คะแนนสูงขึ้น
""")
    elif intent["is_evidence"]:
        sections.append("""
### กฎการตอบ:
- ระบุชื่อไฟล์ (Source: ...) ทุกครั้งที่นำข้อมูลมาตอบ
- หากเป็นเกณฑ์ระดับคะแนน ให้สรุปเป็นข้อๆ ให้ชัดเจน
""")
    else:
        sections.append("### กฎการตอบ: ตอบให้ตรงประเด็น สุภาพ และไม่ออกนอกเหนือจากข้อมูลที่ให้")

    return "\n\n".join(sections)