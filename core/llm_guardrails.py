# -*- coding: utf-8 -*-
# core/llm_guardrails.py

import re
import logging
from typing import Dict, Optional, Any, List
from config.global_vars import (
    SUPPORTED_ENABLERS, DEFAULT_ENABLER,
    PDCA_ANALYSIS_SIGNALS, ANALYSIS_FRAMEWORK
)

logger = logging.getLogger(__name__)

def detect_intent(question: str, doc_type: str = "document", contextual_rules: Dict[str, Any] = None) -> Dict[str, Any]:
    q = question.strip().lower()
    intent = {
        "is_faq": False, "is_synthesis": False, "is_summary": False,
        "is_comparison": False, "is_evidence": False, "is_analysis": False,
        "sub_topic": None, "enabler_hint": None
    }

    # 1. เปรียบเทียบ (Comparison)
    comparison_signals = ["เปรียบเทียบ", "ต่างจาก", "ความแตกต่าง", "เทียบ", "vs", "compare"]
    if any(sig in q for sig in comparison_signals):
        intent["is_comparison"] = True
        return intent

    # 2. สรุปภาพรวม (Summary)
    summary_signals = ["สรุปทั้งหมด", "ภาพรวม", "ทุกไฟล์", "summary"]
    if any(sig in q for sig in summary_signals):
        intent["is_summary"] = True
        return intent

    # 3. ตรวจจับหัวข้อ/Enabler
    enabler_pattern = "|".join([e.lower() for e in SUPPORTED_ENABLERS] + ["km", "cg", "hcm", "sp", "it", "risk"])
    match = re.search(fr"(?:^|\s)({enabler_pattern})\s*[:\-]?\s*(\d+\.\d+)|(\d+\.\d+)", q)
    if match:
        found_enabler = (match.group(1) or "").strip().upper()
        intent["enabler_hint"] = found_enabler if found_enabler in SUPPORTED_ENABLERS else DEFAULT_ENABLER
        intent["sub_topic"] = match.group(2) or match.group(3)

    # 4. วิเคราะห์ PDCA
    analysis_keywords = set(PDCA_ANALYSIS_SIGNALS or [])
    analysis_keywords.update(["plan", "do", "check", "act", "pdca", "เกณฑ์", "ประเมิน"])
    if any(sig in q for sig in analysis_keywords):
        intent["is_analysis"] = True
        return intent

    # 5. ทั่วไป
    if any(sig in q for sig in ["คืออะไร", "คือ", "หมายถึง"]): intent["is_faq"] = True
    else: intent["is_evidence"] = True

    return intent

def build_prompt(context: str, question: str, intent: Dict[str, Any], contextual_rules: Dict[str, Any] = None) -> str:
    sections = ["CRITICAL: Always respond in THAI language only."]
    
    # บทบาท
    role = "ผู้เชี่ยวชาญด้านการวิเคราะห์เอกสาร SE-AM"
    if intent["is_comparison"]: role = "ผู้เชี่ยวชาญการเปรียบเทียบและวิเคราะห์จุดต่าง"
    elif intent["is_summary"]: role = "ผู้เชี่ยวชาญการสรุปประเด็นยุทธศาสตร์"
    elif intent["is_analysis"]: role = f"ผู้เชี่ยวชาญการตรวจสอบหลักฐานตามกรอบ {ANALYSIS_FRAMEWORK}"
    sections.append(f"บทบาทของคุณ: {role}")

    # บริบท
    sections.append(f"ข้อมูลอ้างอิง:\n{context}")
    sections.append(f"คำถามจากผู้ใช้: {question}")

    # คำสั่งตาม Intent
    if intent["is_comparison"]:
        sections.append("คำสั่ง: วิเคราะห์เปรียบเทียบจุดเหมือนและจุดต่างระหว่างไฟล์อ้างอิง ระบุชื่อไฟล์ประกอบ และสรุปเป็นข้อๆ")
    elif intent["is_summary"]:
        sections.append("คำสั่ง: สรุปภาพรวมเนื้อหาของทุกเอกสารที่เลือก โดยเน้นสาระสำคัญที่ผู้บริหารควรรู้")
    elif intent["is_analysis"]:
        sections.append("คำสั่ง: วิเคราะห์คุณภาพหลักฐานตามวงจร PDCA ระบุจุดแข็งและสิ่งที่ควรปรับปรุง")
    else:
        sections.append("คำสั่ง: ตอบคำถามให้ตรงประเด็นโดยอ้างอิงจากข้อมูลที่มี")

    return "\n\n".join(sections)