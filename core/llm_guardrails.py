# -*- coding: utf-8 -*-
# core/llm_guardrails.py (Ultimate Final Version - 21 ธันวาคม 2568)

import re
from typing import Dict, Any, List
import logging

from config.global_vars import (
    SUPPORTED_ENABLERS,
    DEFAULT_ENABLER,
    PDCA_ANALYSIS_SIGNALS,
    ANALYSIS_FRAMEWORK
)

logger = logging.getLogger(__name__)

# ======================================================================
# Intent Detection (Full Production Version with Greeting + Capabilities)
# ======================================================================

def detect_intent(
    question: str,
    user_context: List[Dict[str, Any]] | None = None  # conversation history
) -> Dict[str, Any]:
    """
    วิเคราะห์ความตั้งใจของผู้ใช้ (Intent) พร้อมสกัด Metadata สำคัญ
    เพื่อส่งต่อให้ llm_router ตัดสินใจเลือกใช้ Endpoint หรือตอบกลับอย่างเหมาะสม
    """
    q = question.strip().lower()
    intent = {
        "is_greeting": False,
        "is_capabilities": False,    # NEW: ถามความสามารถของระบบ
        "is_faq": False,
        "is_summary": False,
        "is_comparison": False,
        "is_analysis": False,
        "is_criteria_query": False,
        "sub_topic": None,
        "enabler_hint": None,
    }

    # --- 1. สกัด Enabler/Sub-topic จาก Question หรือ History ---
    enabler_pattern = "|".join([e.lower() for e in SUPPORTED_ENABLERS])
    match = re.search(rf"(?:^|\s)({enabler_pattern})\s*[:\-]?\s*(\d+\.\d+)|(\d+\.\d+)", q)
    
    if match:
        found_enabler = (match.group(1) or "").upper()
        intent["enabler_hint"] = found_enabler if found_enabler in SUPPORTED_ENABLERS else None
        intent["sub_topic"] = match.group(2) or match.group(3)
    
    if not intent["sub_topic"] and user_context:
        for msg in reversed(user_context):
            text = msg.get("content", "").lower()
            m = re.search(rf"(?:^|\s)({enabler_pattern})\s*[:\-]?\s*(\d+\.\d+)|(\d+\.\d+)", text)
            if m:
                found_enabler = (m.group(1) or "").upper()
                intent["enabler_hint"] = found_enabler if found_enabler in SUPPORTED_ENABLERS else None
                intent["sub_topic"] = m.group(2) or m.group(3)
                break

    # --- 2. Greeting Intent (สูงสุด - ตรวจก่อนอย่างอื่น) ---
    greeting_signals = [
        "สวัสดี", "ดีครับ", "ดีค่ะ", "ดีเช้า", "ดีบ่าย", "ดีเย็น",
        "hello", "hi", "hey", "สวัสดีตอนเช้า", "สวัสดีตอนบ่าย", "สวัสดีตอนเย็น",
        "ยินดีที่ได้รู้จัก", "สบายดีไหม", "สบายดีมั้ย"
    ]
    if any(sig in q for sig in greeting_signals):
        intent["is_greeting"] = True
        return intent

    # --- 3. Capabilities / Self-Introduction Intent ---
    capabilities_signals = [
        "ทำอะไรได้บ้าง", "ช่วยอะไรได้", "ทำอะไรได้", "ช่วยได้ไหม",
        "capabilities", "features", "function", "ช่วยเหลืออะไร",
        "คุณทำอะไร", "คุณช่วยอะไร", "คุณคือใคร", "แนะนำตัว"
    ]
    if any(sig in q for sig in capabilities_signals):
        intent["is_capabilities"] = True
        return intent

    # --- 4. Comparison Intent ---
    comparison_signals = ["เปรียบเทียบ", "ความแตกต่าง", "ต่างจาก", "เทียบ", "vs", "compare"]
    if any(sig in q for sig in comparison_signals):
        intent["is_comparison"] = True
        return intent

    # --- 5. Summary Intent ---
    summary_signals = [
        "สรุป", "ภาพรวม", "ทั้งหมด", "executive summary", "overview", "สาระสำคัญ", "key points",
        "summary", "summarize", "summarise", "comprehensive summary", 
        "provide a summary", "give me a summary", "summarize all", "summary of all"
    ]
    if any(sig in q for sig in summary_signals):
        intent["is_summary"] = True
        return intent

    # --- 6. SE-AM Criteria / Evidence Analysis Intent ---
    criteria_signals = [
        "ผ่านเกณฑ์", "sub criteria", "ผ่าน level", "สนับสนุนเกณฑ์", "evidence ผ่าน",
        "เกณฑ์อะไรบ้าง", "level เท่าไหร่", "ครบ level", "ขาดเกณฑ์", "criteria"
    ]
    if any(sig in q for sig in criteria_signals):
        intent["is_analysis"] = True
        intent["is_criteria_query"] = True
        return intent

    # --- 7. PDCA / Deep Analysis Intent ---
    analysis_keywords = set(PDCA_ANALYSIS_SIGNALS or [])
    analysis_keywords.update([
        "pdca", "plan", "do", "check", "act", "วิเคราะห์", "ประเมิน", "ตรวจสอบ",
        "analyze", "จุดแข็ง", "ช่องว่าง", "gap", "strength", "weakness"
    ])
    if any(sig in q for sig in analysis_keywords):
        intent["is_analysis"] = True
        return intent

    # --- 8. Default: FAQ / General Evidence ---
    if any(sig in q for sig in ["คืออะไร", "คือ", "หมายถึง", "definition"]):
        intent["is_faq"] = True

    return intent


# ======================================================================
# Prompt Guardrails Builder (Conversation-aware + Intent-specific)
# ======================================================================

def build_prompt(
    context: str,
    question: str,
    intent: Dict[str, Any],
    user_context: List[Dict[str, Any]] | None = None
) -> str:
    """
    สร้าง system-style instruction แบบ dynamic ตาม intent และประวัติการสนทนา
    """
    sections = []

    # กฎภาษาเข้มงวด
    sections.append("### กฎเหล็ก: ตอบเป็นภาษาไทยเท่านั้น (ยกเว้นชื่อไฟล์หรือคำเทคนิคในเอกสาร) ###")

    # กำหนดบทบาทตาม intent
    role = "ผู้เชี่ยวชาญด้านองค์ความรู้และเอกสารองค์กร"
    if intent.get("is_comparison"):
        role = "ผู้เชี่ยวชาญด้านการเปรียบเทียบเอกสารและวิเคราะห์ความแตกต่างเชิงนโยบาย"
    elif intent.get("is_summary"):
        role = "ผู้เชี่ยวชาญด้านการสรุปสาระสำคัญเชิงผู้บริหาร"
    elif intent.get("is_analysis") or intent.get("is_criteria_query"):
        role = f"ผู้ประเมินคุณภาพหลักฐานตามกรอบ {ANALYSIS_FRAMEWORK} และเกณฑ์ SE-AM"

    sections.append(f"บทบาทของคุณ: {role}")

    # ข้อมูลอ้างอิง
    sections.append("ข้อมูลอ้างอิงจากระบบคลังความรู้:")
    sections.append(context)

    # ประวัติการสนทนา (ถ้ามี)
    if user_context:
        snapshot = "\n".join([f"- {m.get('content')}" for m in user_context[-6:]])
        sections.append("บริบทการสนทนาก่อนหน้า:")
        sections.append(snapshot)

    # คำถามปัจจุบัน
    sections.append(f"คำถามของผู้ใช้: {question}")

    # คำสั่งเฉพาะตาม intent
    if intent.get("is_comparison"):
        sections.append(
            "คำสั่ง: เปรียบเทียบเอกสารโดยระบุจุดเหมือนและจุดต่าง "
            "นำเสนอผลเป็นตาราง Markdown "
            "หากประเด็นใดไม่ปรากฏในเอกสาร ให้ระบุ 'ไม่ปรากฏข้อมูลในเอกสารนี้'"
        )
    elif intent.get("is_summary"):
        sections.append(
            "คำสั่ง: สรุปเอกสารทั้งหมดอย่างละเอียดเป็นภาษาไทยในรูปแบบ Executive Summary\n"
            "ต้องมีหัวข้อชัดเจน เช่น วัตถุประสงค์ ขอบเขต กลยุทธ์หลัก ตัวชี้วัด ความเสี่ยง และสรุปสำหรับผู้บริหาร"
        )
    elif intent.get("is_analysis") or intent.get("is_criteria_query"):
        sections.append(
            "คำสั่ง: วิเคราะห์คุณภาพและความครบถ้วนของหลักฐานตามกรอบ PDCA\n"
            "ระบุจุดแข็ง ช่องว่าง และระดับที่ผ่านตามเกณฑ์ SE-AM\n"
            "หากถามว่า 'ผ่านเกณฑ์อะไรบ้าง' ให้ระบุรหัสเกณฑ์และระดับที่ผ่านอย่างชัดเจน"
        )
    else:
        sections.append(
            "คำสั่ง: ตอบคำถามให้ตรงประเด็น อ้างอิงข้อมูลจากเอกสารที่ให้มา\n"
            "หากไม่พบข้อมูล ให้ตอบว่า 'ไม่พบข้อมูลที่เพียงพอในเอกสารที่เกี่ยวข้อง'"
        )

    return "\n\n".join(sections)


# ======================================================================
# Post-response Validation (Enhanced Thai Safety Net)
# ======================================================================

def enforce_thai_primary_language(response_text: str) -> str:
    """
    ป้องกันการตอบเป็นภาษาอังกฤษทั้งหมด
    แต่ยอมรับกรณีที่มีตาราง Markdown, JSON, หรือชื่อไฟล์ภาษาอังกฤษ
    """
    if not response_text.strip():
        return response_text

    lines = [line.strip() for line in response_text.splitlines() if line.strip()]

    # คัดกรองเฉพาะบรรทัดที่เป็น narrative จริง ๆ
    narrative_lines = []
    for line in lines:
        # ข้ามโครงสร้างที่ไม่ใช่ narrative
        if any(line.startswith(prefix) for prefix in ["#", "##", "###", "|", "-", "*", ">", "```", "<", "["]):
            continue
        # ข้ามบรรทัดสั้น ๆ (ชื่อไฟล์, JSON key)
        if len(line.split()) <= 5:
            continue
        narrative_lines.append(line)

    # ถ้าไม่มี narrative เลย (เช่น ตอบแต่ตารางหรือ JSON) → ปล่อยผ่าน
    if not narrative_lines:
        return response_text

    narrative_text = " ".join(narrative_lines)
    thai_count = len(re.findall(r"[ก-๙]", narrative_text))

    # ถ้ามีตัวอักษรไทยน้อยเกินไป → เตือนแต่ไม่บล็อก (เพื่อความยืดหยุ่น)
    if thai_count < 10 and len(narrative_text) > 100:
        logger.warning("Detected low Thai content in narrative response - consider re-prompting for Thai output")
        return response_text

    return response_text