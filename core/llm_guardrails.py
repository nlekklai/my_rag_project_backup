# -*- coding: utf-8 -*-
# core/llm_guardrails.py

import re
import logging
from typing import Dict, Any

from config.global_vars import (
    SUPPORTED_ENABLERS,
    DEFAULT_ENABLER,
    PDCA_ANALYSIS_SIGNALS,
    ANALYSIS_FRAMEWORK
)

logger = logging.getLogger(__name__)

# ======================================================================
# Intent Detection
# ======================================================================

def detect_intent(
    question: str,
    doc_type: str = "document",
    contextual_rules: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """
    วิเคราะห์ intent ของคำถาม เพื่อใช้เลือก prompt / chain ให้เหมาะสม
    (ไม่สร้าง prompt ที่นี่)
    """
    q = question.strip().lower()

    intent = {
        "is_faq": False,
        "is_synthesis": False,
        "is_summary": False,
        "is_comparison": False,
        "is_evidence": False,
        "is_analysis": False,
        "sub_topic": None,
        "enabler_hint": None,
    }

    # --------------------------------------------------
    # 1) Comparison
    # --------------------------------------------------
    comparison_signals = [
        "เปรียบเทียบ", "ความแตกต่าง", "ต่างจาก", "เทียบ",
        "vs", "compare", "comparison", "different"
    ]
    if any(sig in q for sig in comparison_signals):
        intent["is_comparison"] = True
        return intent

    # --------------------------------------------------
    # 2) Summary / Synthesis
    # --------------------------------------------------
    summary_signals = [
        "สรุป", "ภาพรวม", "ทั้งหมด", "ทุกไฟล์",
        "summary", "overview", "high level"
    ]
    if any(sig in q for sig in summary_signals):
        intent["is_summary"] = True
        return intent

    # --------------------------------------------------
    # 3) Enabler / Sub-criteria detection
    # --------------------------------------------------
    enabler_pattern = "|".join([e.lower() for e in SUPPORTED_ENABLERS])
    match = re.search(
        rf"(?:^|\s)({enabler_pattern})\s*[:\-]?\s*(\d+\.\d+)|(\d+\.\d+)",
        q
    )
    if match:
        found_enabler = (match.group(1) or "").upper()
        intent["enabler_hint"] = (
            found_enabler if found_enabler in SUPPORTED_ENABLERS else DEFAULT_ENABLER
        )
        intent["sub_topic"] = match.group(2) or match.group(3)

    # --------------------------------------------------
    # 4) PDCA / Analysis
    # --------------------------------------------------
    analysis_keywords = set(PDCA_ANALYSIS_SIGNALS or [])
    analysis_keywords.update([
        "pdca", "plan", "do", "check", "act",
        "วิเคราะห์", "ประเมิน", "ตรวจสอบ", "เกณฑ์", "analyze"
    ])
    if any(sig in q for sig in analysis_keywords):
        intent["is_analysis"] = True
        return intent

    # --------------------------------------------------
    # 5) FAQ / Evidence
    # --------------------------------------------------
    if any(sig in q for sig in ["คืออะไร", "คือ", "หมายถึง", "definition"]):
        intent["is_faq"] = True
    else:
        intent["is_evidence"] = True

    return intent


# ======================================================================
# Prompt Guardrails Builder
# ======================================================================

def build_prompt(
    context: str,
    question: str,
    intent: Dict[str, Any],
    contextual_rules: Dict[str, Any] | None = None
) -> str:
    """
    สร้าง system-style instruction เพื่อครอบ (guardrail)
    prompt หลักจาก rag_prompts.py
    """

    sections: list[str] = []

    # --------------------------------------------------
    # HARD LANGUAGE RULE (ROOT CAUSE FIX)
    # --------------------------------------------------
    sections.append(
        "ข้อบังคับสูงสุด: "
        "คุณต้องตอบเป็นภาษาไทยเท่านั้น "
        "ห้ามใช้ภาษาอังกฤษหรือคำอธิบายภาษาอังกฤษ "
        "ยกเว้นคำศัพท์ที่ปรากฏอยู่ในเอกสารอ้างอิงโดยตรง"
    )

    # --------------------------------------------------
    # ROLE DEFINITION
    # --------------------------------------------------
    role = "ผู้เชี่ยวชาญด้านการวิเคราะห์เอกสารและองค์ความรู้ขององค์กร"

    if intent.get("is_comparison"):
        role = "ผู้เชี่ยวชาญด้านการเปรียบเทียบเอกสารและการวิเคราะห์ความแตกต่างเชิงนโยบาย"
    elif intent.get("is_summary"):
        role = "ผู้เชี่ยวชาญด้านการสรุปสาระสำคัญเชิงผู้บริหาร"
    elif intent.get("is_analysis"):
        role = f"ผู้เชี่ยวชาญด้านการประเมินหลักฐานตามกรอบ {ANALYSIS_FRAMEWORK}"

    sections.append(f"บทบาทของคุณ: {role}")

    # --------------------------------------------------
    # CONTEXT & QUESTION
    # --------------------------------------------------
    sections.append("ข้อมูลอ้างอิงจากระบบคลังความรู้:")
    sections.append(context)

    sections.append("คำถามจากผู้ใช้:")
    sections.append(question)

    # --------------------------------------------------
    # TASK INSTRUCTION BY INTENT
    # --------------------------------------------------
    if intent.get("is_comparison"):
        sections.append(
            "คำสั่ง: "
            "ให้เปรียบเทียบเอกสารที่ให้มา "
            "โดยระบุจุดที่เหมือนและแตกต่างอย่างชัดเจน "
            "ต้องอ้างอิงข้อความจริงจากแต่ละเอกสาร "
            "และสรุปผลในรูปแบบตาราง Markdown "
            "หากประเด็นใดไม่ปรากฏในเอกสาร ให้ระบุว่า "
            "'ไม่ปรากฏข้อมูลในเอกสารนี้'"
        )

    elif intent.get("is_summary"):
        sections.append(
            "คำสั่ง: "
            "สรุปภาพรวมของเอกสารทั้งหมด "
            "เน้นสาระสำคัญที่ผู้บริหารควรทราบ "
            "ห้ามสรุปนอกเหนือจากข้อมูลที่ปรากฏในเอกสาร"
        )

    elif intent.get("is_analysis"):
        sections.append(
            "คำสั่ง: "
            "วิเคราะห์คุณภาพและความครบถ้วนของหลักฐาน "
            "ตามกรอบ PDCA (Plan-Do-Check-Act) "
            "ระบุจุดแข็ง ช่องว่าง และข้อควรปรับปรุง "
            "โดยอ้างอิงจากเอกสารเท่านั้น"
        )

    else:
        sections.append(
            "คำสั่ง: "
            "ตอบคำถามให้ตรงประเด็น "
            "โดยอ้างอิงข้อมูลจากเอกสารที่ให้มา "
            "หากข้อมูลไม่เพียงพอ ให้ตอบว่า "
            "'ไม่พบข้อมูลที่เพียงพอในเอกสารที่เกี่ยวข้อง'"
        )

    return "\n\n".join(sections)


# ======================================================================
# Optional: Post-response Validation (Thai-only Safety Net)
# ======================================================================
def enforce_thai_primary_language(response_text: str) -> str:
    """
    Policy:
    - อนุญาตภาษาอังกฤษใน table / heading / technical term
    - Narrative explanation ต้องเป็นภาษาไทยเป็นหลัก
    """

    lines = response_text.splitlines()
    narrative_lines = []

    for line in lines:
        line = line.strip()

        # ข้าม markdown / table / list / header
        if (
            not line
            or line.startswith("|")
            or line.startswith("-")
            or line.startswith("###")
            or line.startswith("##")
            or line.startswith("#")
            or line.startswith("```")
        ):
            continue

        # ข้ามประโยคสั้น ๆ ที่เป็น instruction
        if len(line.split()) <= 5:
            continue

        narrative_lines.append(line)

    narrative_text = " ".join(narrative_lines)

    # ไม่มี narrative จริง → ผ่าน
    if not narrative_text:
        return response_text

    thai_chars = len(re.findall(r"[ก-๙]", narrative_text))
    eng_chars = len(re.findall(r"[A-Za-z]", narrative_text))

    # ต้องให้ไทยมากกว่าอังกฤษ "พอสมควร"
    if eng_chars > thai_chars * 1.2:
        logger.warning("English dominates narrative content")
        return (
            "ไม่สามารถแสดงผลคำตอบได้ "
            "เนื่องจากคำอธิบายหลักไม่เป็นภาษาไทยตามนโยบายเอกสารราชการ"
        )

    return response_text
