# -*- coding: utf-8 -*-
# core/llm_guardrails.py (Ultimate Revised Version - 20 ธันวาคม 2568)

import re
import logging
from typing import Dict, Any, List, Optional

from config.global_vars import (
    SUPPORTED_ENABLERS,
    DEFAULT_ENABLER,
    PDCA_ANALYSIS_SIGNALS,
    ANALYSIS_FRAMEWORK
)

logger = logging.getLogger(__name__)

# ======================================================================
# Intent Detection (Smart Logic for Multi-Path Routing)
# ======================================================================

def detect_intent(
    question: str,
    user_context: List[Dict[str, Any]] | None = None  # conversation history
) -> Dict[str, Any]:
    """
    วิเคราะห์ความตั้งใจของผู้ใช้ (Intent) พร้อมสกัด Metadata สำคัญ
    เพื่อส่งต่อให้ llm_router ตัดสินใจเลือกใช้ Endpoint (/query, /compare, /analysis)
    """
    q = question.strip().lower()
    intent = {
        "is_faq": False,
        "is_summary": False,
        "is_comparison": False,
        "is_analysis": False,
        "is_criteria_query": False,
        "sub_topic": None,
        "enabler_hint": None,
    }

    # --- 1. สกัด Enabler/Sub-topic จาก Question หรือ History ---
    # Regex สำหรับตรวจจับรหัส เช่น KM 1.1, HR 2.2 หรือแค่ 1.1
    enabler_pattern = "|".join([e.lower() for e in SUPPORTED_ENABLERS])
    # ค้นหาในคำถามปัจจุบันก่อน
    match = re.search(rf"(?:^|\s)({enabler_pattern})\s*[:\-]?\s*(\d+\.\d+)|(\d+\.\d+)", q)
    
    if match:
        found_enabler = (match.group(1) or "").upper()
        intent["enabler_hint"] = found_enabler if found_enabler in SUPPORTED_ENABLERS else None
        intent["sub_topic"] = match.group(2) or match.group(3)
    
    # ถ้าคำถามไม่มีรหัส แต่ในประวัติมี ให้ดึงจากประวัติมาเติม (Context Awareness)
    if not intent["sub_topic"] and user_context:
        for msg in reversed(user_context):
            text = msg.get("content", "").lower()
            m = re.search(rf"(?:^|\s)({enabler_pattern})\s*[:\-]?\s*(\d+\.\d+)|(\d+\.\d+)", text)
            if m:
                found_enabler = (m.group(1) or "").upper()
                intent["enabler_hint"] = found_enabler if found_enabler in SUPPORTED_ENABLERS else None
                intent["sub_topic"] = m.group(2) or m.group(3)
                break

    # --- 2. การตรวจสอบ Comparison Intent ---
    comparison_signals = ["เปรียบเทียบ", "ความแตกต่าง", "ต่างจาก", "เทียบ", "vs", "compare"]
    if any(sig in q for sig in comparison_signals):
        intent["is_comparison"] = True
        return intent

    # --- 3. การตรวจสอบ Analysis & Criteria Intent ---
    criteria_signals = [
        "ผ่านเกณฑ์", "sub criteria", "ผ่าน level", "สนับสนุนเกณฑ์", "evidence ผ่าน",
        "เกณฑ์อะไรบ้าง", "level เท่าไหร่", "ครบ level", "ขาดเกณฑ์", "criteria"
    ]
    analysis_keywords = set(PDCA_ANALYSIS_SIGNALS or [])
    analysis_keywords.update([
        "pdca", "plan", "do", "check", "act", "วิเคราะห์", "ประเมิน", "ตรวจสอบ",
        "analyze", "จุดแข็ง", "ช่องว่าง", "gap", "strength", "weakness"
    ])

    if any(sig in q for sig in criteria_signals):
        intent["is_analysis"] = True
        intent["is_criteria_query"] = True
        return intent
    
    if any(sig in q for sig in analysis_keywords):
        intent["is_analysis"] = True
        return intent

    # --- 4. การตรวจสอบ Summary Intent ---
    summary_signals = ["สรุป", "ภาพรวม", "ทั้งหมด", "summary", "overview"]
    if any(sig in q for sig in summary_signals):
        intent["is_summary"] = True
        return intent

    # --- 5. Default Case (FAQ / General QA) ---
    if any(sig in q for sig in ["คืออะไร", "คือ", "หมายถึง", "definition"]):
        intent["is_faq"] = True

    return intent


# ======================================================================
# Prompt Guardrails Builder
# ======================================================================

def build_prompt(
    context: str,
    question: str,
    intent: Dict[str, Any],
    user_context: List[Dict[str, Any]] | None = None
) -> str:
    """
    สร้าง Prompt ที่ประกอบด้วยบทบาท (Role) ข้อมูลอ้างอิง และประวัติการคุย
    """
    sections = []

    # กฎภาษาเข้มงวด
    sections.append("### RULE: ANSWER IN THAI ONLY. (Except file names or technical codes) ###")

    # กำหนดบทบาทตาม Intent
    role = "ผู้เชี่ยวชาญด้านองค์ความรู้องค์กร"
    if intent.get("is_analysis"):
        role = f"ผู้ประเมินคุณภาพหลักฐานตามกรอบ {ANALYSIS_FRAMEWORK} และเกณฑ์ SE-AM"
    elif intent.get("is_comparison"):
        role = "นักวิเคราะห์นโยบายและเปรียบเทียบเอกสารอ้างอิง"
    
    sections.append(f"บทบาทของคุณ: {role}")
    sections.append(f"ข้อมูลอ้างอิง:\n{context}")

    # แทรกประวัติการสนทนาล่าสุด (ถ้ามี)
    if user_context:
        snapshot = "\n".join([f"- {m.get('content')}" for m in user_context[-4:]])
        sections.append(f"บริบทการสนทนาก่อนหน้า:\n{snapshot}")

    sections.append(f"คำถามของผู้ใช้: {question}")

    # คำสั่งเฉพาะ (Task Instructions)
    if intent.get("is_comparison"):
        sections.append("คำสั่ง: เปรียบเทียบข้อมูลจุดต่อจุด แสดงผลเป็นตาราง Markdown และระบุความแตกต่าง")
    elif intent.get("is_analysis"):
        sections.append("คำสั่ง: วิเคราะห์ตาม PDCA ระบุจุดแข็งและช่องว่าง (Gap) โดยอ้างอิงรหัสเกณฑ์ให้ถูกต้อง")
    else:
        sections.append("คำสั่ง: ตอบคำถามให้กระชับ อ้างอิงแหล่งข้อมูลจากคลังความรู้เท่านั้น")

    return "\n\n".join(sections)


# ======================================================================
# Post-response Validation (Enhanced Safety Net)
# ======================================================================

def enforce_thai_primary_language(response_text: str) -> str:
    """
    คัดกรองคำตอบเพื่อป้องกันการตอบเป็นภาษาอังกฤษล้วน 
    โดยละเว้นส่วนที่เป็นตาราง รหัส หรือชื่อไฟล์
    """
    if not response_text:
        return response_text

    # แยกบรรทัดที่เป็นข้อความอธิบาย (Narrative) ออกมาตรวจสอบ
    narrative_lines = []
    for line in response_text.splitlines():
        line = line.strip()
        # ข้ามบรรทัดที่เป็น Markdown Structure หรือสั้นเกินไป
        if not line or any(line.startswith(c) for c in ["#", "|", "-", "*", "`", "["]):
            continue
        if len(line.split()) < 4:
            continue
        narrative_lines.append(line)

    if not narrative_lines:
        return response_text # ถ้าตอบแต่ตารางหรือ JSON ปล่อยผ่าน

    # ตรวจสอบสัดส่วนอักษรไทย
    narrative_text = " ".join(narrative_lines)
    thai_chars = re.findall(r"[ก-๙]", narrative_text)
    
    # หากมีตัวอักษรไทยน้อยมากเมื่อเทียบกับความยาวข้อความ (Narrative)
    if len(thai_chars) < 10 and len(narrative_text) > 40:
        logger.warning("English narrative detected! Blocking response.")
        return "ขออภัย ระบบสามารถตอบได้เฉพาะภาษาไทยเท่านั้น กรุณาตรวจสอบคำถามของคุณอีกครั้ง"

    return response_text