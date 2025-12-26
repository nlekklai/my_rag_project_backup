# -*- coding: utf-8 -*-
# core/llm_guardrails.py

import re
from typing import Dict, Any, List, Optional
import logging

from config.global_vars import (
    SUPPORTED_ENABLERS,
    SEAM_ENABLER_MAP,
    SEAM_SUBTOPIC_MAP,
    PDCA_ANALYSIS_SIGNALS,
    ANALYSIS_FRAMEWORK
)

logger = logging.getLogger(__name__)

# ======================================================================
# 1. Intent Detection (Multi-Enabler Support)
# ======================================================================

def detect_intent(
    question: str,
    user_context: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    q = question.strip().lower()
    intent = {
        "is_greeting": False,
        "is_capabilities": False,    
        "is_faq": False,
        "is_summary": False,
        "is_comparison": False,
        "is_analysis": False,
        "is_criteria_query": False,
        "sub_topic": None,
        "enabler_hint": None,
        "enabler_full_name": None
    }

    # --- 1. สกัด Enabler & Sub-topic (Dynamic Lookup) ---
    enabler_pattern = "|".join([e.lower() for e in SUPPORTED_ENABLERS])
    # ค้นหา Enabler หรือรหัสเกณฑ์ (เช่น 6.1, 7.1)
    match = re.search(rf"(?:^|\s)({enabler_pattern})\s*[:\-]?\s*(\d+\.\d+)|(\d+\.\d+)", q)
    
    found_key = None
    if match:
        found_enabler = (match.group(1) or "").upper()
        intent["enabler_hint"] = found_enabler if found_enabler in SUPPORTED_ENABLERS else None
        intent["sub_topic"] = match.group(2) or match.group(3)
        found_key = intent["sub_topic"]
    
    # ถ้าไม่เจอในคำถาม ให้ดูจาก History
    if not intent["sub_topic"] and user_context:
        for msg in reversed(user_context):
            text = msg.get("content", "").lower()
            m = re.search(rf"(?:^|\s)({enabler_pattern})\s*[:\-]?\s*(\d+\.\d+)|(\d+\.\d+)", text)
            if m:
                intent["enabler_hint"] = (m.group(1) or "").upper()
                intent["sub_topic"] = m.group(2) or m.group(3)
                found_key = intent["sub_topic"]
                break

    # แมพชื่อเต็มของ Enabler จาก SEAM_ENABLER_MAP
    if intent["enabler_hint"] in SEAM_ENABLER_MAP:
        intent["enabler_full_name"] = SEAM_ENABLER_MAP[intent["enabler_hint"]]

    # --- 2. Route Intent (Priority Check) ---
    if any(sig in q for sig in ["สรุป", "ภาพรวม", "summary", "overview"]):
        intent["is_summary"] = True
        return intent

    if any(sig in q for sig in ["เปรียบเทียบ", "ความแตกต่าง", "vs", "compare", "difference"]):
        intent["is_comparison"] = True
        return intent

    if any(re.search(rf"\b{re.escape(sig)}\b", q) for sig in ["สวัสดี", "hello", "hi", "hey"]):
        intent["is_greeting"] = True
        return intent

    if any(sig in q for sig in ["ทำอะไรได้บ้าง", "ช่วยอะไรได้", "capabilities", "features"]):
        intent["is_capabilities"] = True
        return intent

    # Analysis Intent: ถ้ามีคำกลุ่มประเมิน หรือมีรหัสเกณฑ์ SE-AM
    analysis_signals = ["ผ่านเกณฑ์", "criteria", "สอดคล้อง", "วิเคราะห์", "ประเมิน", "pdca", "gap"]
    if any(sig in q for sig in analysis_signals) or intent["sub_topic"]:
        intent["is_analysis"] = True
        return intent

    return intent

# ======================================================================
# 2. Prompt Builder (Dynamic Role & Instruction)
# ======================================================================

def build_prompt(
    context: str,
    question: str,
    intent: Dict[str, Any],
    user_context: Optional[List[Dict[str, Any]]] = None
) -> str:
    sections = []
    sections.append("### กฎเหล็ก: ตอบเป็นภาษาไทยเท่านั้น ###")

    # กำหนดบทบาทแบบ Dynamic ตาม Enabler ที่ตรวจพบ
    enabler_name = intent.get("enabler_full_name") or "องค์ความรู้และมาตรฐานองค์กร"
    
    if intent.get("is_summary"):
        role = f"ผู้เชี่ยวชาญด้านการสรุปสาระสำคัญ ({enabler_name})"
    elif intent.get("is_comparison"):
        role = f"ผู้เชี่ยวชาญด้านการวิเคราะห์ความแตกต่างเอกสาร ({enabler_name})"
    elif intent.get("is_analysis"):
        role = f"ผู้ประเมินคุณภาพหลักฐานตามเกณฑ์ SE-AM ด้าน {enabler_name}"
    else:
        role = "ผู้ช่วยอัจฉริยะด้านการจัดการข้อมูลองค์กร"

    sections.append(f"บทบาทของคุณ: {role}")
    sections.append(f"--- ข้อมูลอ้างอิง ({intent.get('enabler_hint', 'General')}) ---\n{context}\n---")

    # เพิ่ม Instruction เฉพาะทาง
    if intent.get("is_analysis"):
        sections.append(
            f"คำสั่ง: วิเคราะห์หลักฐานตามกรอบ {ANALYSIS_FRAMEWORK}\n"
            "1. ตรวจสอบความครบถ้วนตามวงจร Plan, Do, Check, Act\n"
            "2. ระบุจุดแข็งและสิ่งที่ควรปรับปรุง (Gaps)\n"
            "3. ประเมินระดับ Maturity (L1-L5) ตามหลักฐานที่มี"
        )
    elif intent.get("is_comparison"):
        sections.append("คำสั่ง: เปรียบเทียบข้อมูลในรูปแบบตาราง Markdown ภาษาไทย พร้อมสรุปประเด็นสำคัญ")

    sections.append(f"### คำถาม: {question}")
    sections.append("### กฎการตอบ: ตอบเป็นภาษาไทยสละสลวย หากถามภาษาอังกฤษมาให้แปลและสรุปเป็นไทยเสมอ")

    return "\n\n".join(sections)


# ======================================================================
# Post-response Validation (Enhanced Thai Safety Net)
# ======================================================================

def enforce_thai_primary_language(response_text: str) -> str:
    """
    Revised Version: ตรวจสอบและบังคับทิศทางภาษาไทยแบบเข้มงวด
    """
    if not response_text or not response_text.strip():
        return response_text

    # 1. ล้างบรรทัดที่ไม่ใช่เนื้อความเพื่อตรวจสัดส่วนภาษา
    lines = [line.strip() for line in response_text.splitlines() if line.strip()]
    narrative_lines = []
    for line in lines:
        if any(line.startswith(p) for p in ["#", "|", "-", "*", ">", "```", "["]):
            continue
        if len(line.split()) <= 3: # ข้ามพวกชื่อหัวข้อสั้นๆ
            continue
        narrative_lines.append(line)

    if not narrative_lines:
        return response_text

    narrative_text = " ".join(narrative_lines)
    
    # 2. นับตัวอักษรไทย
    thai_count = len(re.findall(r"[ก-๙]", narrative_text))
    # นับคำภาษาอังกฤษ (ที่มีความยาว > 2 ตัวอักษร)
    eng_words = re.findall(r"\b[a-zA-Z]{3,}\b", narrative_text)

    # 🎯 วิเคราะห์อาการดื้อ (Stubbornness Detection)
    # ถ้าคำอังกฤษเยอะกว่าตัวอักษรไทย (ซึ่งปกติ 1 คำไทยมีหลายตัวอักษร) แสดงว่าดื้อแน่นอน
    if len(eng_words) > 10 and thai_count < 20:
        logger.error(f"🚨 AI ดื้อตอบภาษาอังกฤษล้วน! (ENG Words: {len(eng_words)}, Thai Chars: {thai_count})")
        
        # ยาแรง: บังคับแทรกคำเตือน และสรุปสั้นๆ (ถ้าทำได้)
        stubborn_msg = (
            "⚠️ **[ระบบตรวจพบว่า AI ตอบเป็นภาษาอังกฤษ]**\n"
            "*คำสั่งบังคับภาษาไทยถูกละเลยโดย Model โปรดลองถามใหม่ด้วยภาษาไทย หรือตรวจสอบ System Prompt*\n\n"
            "---\n"
        )
        return stubborn_msg + response_text

    return response_text