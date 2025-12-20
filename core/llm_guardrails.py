# -*- coding: utf-8 -*-
# core/llm_guardrails.py (Enhanced Multi-User Version)

import re
import logging
from typing import Dict, Any, List

from config.global_vars import (
    SUPPORTED_ENABLERS,
    DEFAULT_ENABLER,
    PDCA_ANALYSIS_SIGNALS,
    ANALYSIS_FRAMEWORK
)

logger = logging.getLogger(__name__)

# ======================================================================
# Intent Detection (Enhanced: multi-user aware)
# ======================================================================

def detect_intent(
    question: str,
    doc_type: str = "document",
    contextual_rules: Dict[str, Any] | None = None,
    user_context: List[Dict[str, Any]] | None = None  # conversation history
) -> Dict[str, Any]:
    """
    วิเคราะห์ intent ของคำถาม โดยสามารถใช้ conversation history ของผู้ใช้
    เพื่อ infer sub-topic / enabler / analysis signals ได้แม่นยำขึ้น
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

    # ใช้ conversation history ช่วย infer
    if user_context:
        # ตัวอย่างง่าย: หา last referenced enabler/sub-topic
        for msg in reversed(user_context):
            text = msg.get("content", "").lower()
            enabler_pattern = "|".join([e.lower() for e in SUPPORTED_ENABLERS])
            match = re.search(rf"(?:^|\s)({enabler_pattern})\s*[:\-]?\s*(\d+\.\d+)|(\d+\.\d+)", text)
            if match:
                found_enabler = (match.group(1) or "").upper()
                intent["enabler_hint"] = found_enabler if found_enabler in SUPPORTED_ENABLERS else DEFAULT_ENABLER
                intent["sub_topic"] = match.group(2) or match.group(3)
                break

    # --------------------------------------------------
    # Comparison
    # --------------------------------------------------
    comparison_signals = ["เปรียบเทียบ", "ความแตกต่าง", "ต่างจาก", "เทียบ", "vs", "compare"]
    if any(sig in q for sig in comparison_signals):
        intent["is_comparison"] = True
        return intent

    # --------------------------------------------------
    # Summary / Synthesis
    # --------------------------------------------------
    summary_signals = ["สรุป", "ภาพรวม", "ทั้งหมด", "ทุกไฟล์", "summary", "overview"]
    if any(sig in q for sig in summary_signals):
        intent["is_summary"] = True
        return intent

    # --------------------------------------------------
    # Enabler / Sub-criteria detection (จาก question)
    # --------------------------------------------------
    enabler_pattern = "|".join([e.lower() for e in SUPPORTED_ENABLERS])
    match = re.search(rf"(?:^|\s)({enabler_pattern})\s*[:\-]?\s*(\d+\.\d+)|(\d+\.\d+)", q)
    if match:
        found_enabler = (match.group(1) or "").upper()
        intent["enabler_hint"] = found_enabler if found_enabler in SUPPORTED_ENABLERS else DEFAULT_ENABLER
        intent["sub_topic"] = match.group(2) or match.group(3)

    # --------------------------------------------------
    # PDCA / Analysis
    # --------------------------------------------------
    analysis_keywords = set(PDCA_ANALYSIS_SIGNALS or [])
    analysis_keywords.update(["pdca", "plan", "do", "check", "act", "วิเคราะห์", "ประเมิน", "ตรวจสอบ", "เกณฑ์", "analyze"])
    if any(sig in q for sig in analysis_keywords):
        intent["is_analysis"] = True
        return intent

    # --------------------------------------------------
    # FAQ / Evidence
    # --------------------------------------------------
    if any(sig in q for sig in ["คืออะไร", "คือ", "หมายถึง", "definition"]):
        intent["is_faq"] = True
    else:
        intent["is_evidence"] = True

    return intent


# ======================================================================
# Prompt Guardrails Builder (Enhanced: conversation-aware)
# ======================================================================

def build_prompt(
    context: str,
    question: str,
    intent: Dict[str, Any],
    user_context: List[Dict[str, Any]] | None = None,
    contextual_rules: Dict[str, Any] | None = None
) -> str:
    """
    สร้าง system-style instruction แบบ multi-user aware
    - ใส่ conversation snapshot ของผู้ใช้ เพื่อให้ LLM เข้าใจบริบทต่อเนื่อง
    """
    sections: List[str] = []

    # --------------------------------------------------
    # HARD LANGUAGE RULE
    # --------------------------------------------------
    sections.append(
        "ข้อบังคับสูงสุด: "
        "ตอบเป็นภาษาไทยเท่านั้น "
        "ยกเว้นคำศัพท์ที่ปรากฏในเอกสารอ้างอิงโดยตรง"
    )

    # --------------------------------------------------
    # ROLE DEFINITION
    # --------------------------------------------------
    role = "ผู้เชี่ยวชาญด้านการวิเคราะห์เอกสารและองค์ความรู้ขององค์กร"
    if intent.get("is_comparison"):
        role = "ผู้เชี่ยวชาญด้านการเปรียบเทียบเอกสารและวิเคราะห์ความแตกต่างเชิงนโยบาย"
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
    # Conversation snapshot
    # --------------------------------------------------
    if user_context:
        snapshot_lines = []
        for msg in user_context[-6:]:  # limit last 6 turns
            role_label = "ผู้ใช้" if msg.get("type") == "user" else "AI"
            snapshot_lines.append(f"[{role_label}] {msg.get('content')}")
        sections.append("ข้อมูลจากการสนทนาก่อนหน้า:")
        sections.append("\n".join(snapshot_lines))

    # --------------------------------------------------
    # TASK INSTRUCTION BY INTENT
    # --------------------------------------------------
    if intent.get("is_comparison"):
        sections.append(
            "คำสั่ง: เปรียบเทียบเอกสารโดยระบุจุดเหมือน/แตกต่าง "
            "อ้างอิงข้อความจริงจากเอกสารแต่ละฉบับ "
            "สรุปผลเป็น Markdown table "
            "หากประเด็นไม่ปรากฏในเอกสาร ให้ระบุ 'ไม่ปรากฏข้อมูลในเอกสารนี้'"
        )
    elif intent.get("is_summary"):
        sections.append(
            "คำสั่ง: สรุปภาพรวมเอกสารทั้งหมด "
            "เน้นสาระสำคัญสำหรับผู้บริหาร "
            "ห้ามสรุปนอกเหนือจากเอกสาร"
        )
    elif intent.get("is_analysis"):
        sections.append(
            "คำสั่ง: วิเคราะห์คุณภาพและความครบถ้วนของหลักฐาน "
            "ตามกรอบ PDCA (Plan-Do-Check-Act) "
            "ระบุจุดแข็ง ช่องว่าง และข้อปรับปรุง "
            "อ้างอิงจากเอกสารเท่านั้น"
        )
    else:
        sections.append(
            "คำสั่ง: ตอบคำถามตรงประเด็น "
            "อ้างอิงข้อมูลจากเอกสารที่ให้มา "
            "หากไม่เพียงพอ ให้ตอบว่า 'ไม่พบข้อมูลที่เพียงพอในเอกสารที่เกี่ยวข้อง'"
        )

    return "\n\n".join(sections)


# ======================================================================
# Post-response Validation (Thai-only Safety Net)
# ======================================================================

# core/llm_guardrails.py

def enforce_thai_primary_language(response_text: str) -> str:
    """
    ป้องกันการตอบเป็นภาษาอังกฤษทั้งหมด 
    แต่ยอมรับกรณี compare/analysis ที่อาจมีชื่อไฟล์หรือคำเทคนิคอังกฤษ
    """
    lines = [line.strip() for line in response_text.splitlines() if line.strip()]
    if not lines:
        return response_text

    # คัดกรองเอาเฉพาะบรรทัดที่เป็นบทบรรยาย (Narrative)
    narrative_lines = []
    for line in lines:
        # ข้าม Markdown Table, Headers, และ List tags
        if line.startswith(("#", "|", "-", "*")) or "```" in line:
            continue
        # ข้ามบรรทัดสั้นๆ ที่มักเป็นชื่อไฟล์หรือชื่อตัวแปร
        if len(line.split()) <= 4:
            continue
        narrative_lines.append(line)

    # หากคำตอบมีแต่ตารางหรือหัวข้อ (ไม่มีบทบรรยายยาวๆ) ให้ปล่อยผ่าน
    if not narrative_lines:
        return response_text

    narrative_text = " ".join(narrative_lines)
    
    # ตรวจสอบว่าในบทบรรยายมีตัวอักษรภาษาไทยหรือไม่
    thai_count = len(re.findall(r"[ก-๙]", narrative_text))
    
    if thai_count == 0:
        logger.warning("Response has no Thai characters in narrative - blocked")
        return "ไม่สามารถแสดงผลคำตอบได้ เนื่องจากคำอธิบายหลักไม่เป็นภาษาไทยตามนโยบาย"

    return response_text
