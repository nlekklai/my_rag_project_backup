# -*- coding: utf-8 -*-
# core/llm_guardrails.py
import re
from typing import Dict, Any, List, Optional
import logging

from config.global_vars import (
    SUPPORTED_ENABLERS,
    SEAM_ENABLER_MAP,
    ANALYSIS_FRAMEWORK,
    PDCA_ANALYSIS_SIGNALS
)

logger = logging.getLogger(__name__)

# ======================================================================
# 1. Intent Detection (Revised Priority for Capabilities)
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

    # --- Step 1: Extract Enabler/Sub-topic First (For Role Identity) ---
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
                intent["enabler_hint"] = (m.group(1) or "").upper()
                intent["sub_topic"] = m.group(2) or m.group(3)
                break

    if intent["enabler_hint"] in SEAM_ENABLER_MAP:
        intent["enabler_full_name"] = SEAM_ENABLER_MAP[intent["enabler_hint"]]

    # --- Step 2: Route Intent by Priority ---

    # [Priority 1] Capabilities & Intro (ต้องดักก่อนเพื่อให้ AI ตอบบทบาทตัวเอง)
    cap_signals = ["ทำอะไรได้บ้าง", "ช่วยอะไรได้", "แนะนำตัว", "what can you do", "who are you", "capabilities", "features", "how to use"]
    if any(sig in q for sig in cap_signals):
        intent["is_capabilities"] = True
        return intent

    # [Priority 2] Greeting
    if any(re.search(rf"\b{re.escape(sig)}\b", q) for sig in ["สวัสดี", "hello", "hi", "hey", "ดีครับ", "ดีค่ะ"]):
        intent["is_greeting"] = True
        return intent

    # [Priority 3] Comparison (ดัก vs, compare ก่อน summary)
    comp_signals = ["เปรียบเทียบ", "ความแตกต่าง", "vs", "compare", "difference", "contrast"]
    if any(sig in q for sig in comp_signals):
        intent["is_comparison"] = True
        return intent

    # [Priority 4] Summary
    sum_signals = ["สรุป", "ภาพรวม", "summary", "overview", "executive summary", "summarize"]
    if any(sig in q for sig in sum_signals):
        intent["is_summary"] = True
        return intent

    # [Priority 5] SE-AM Analysis
    crit_signals = ["ผ่านเกณฑ์", "criteria", "สอดคล้อง", "วิเคราะห์", "ประเมิน", "pdca", "gap", "strength"]
    if any(sig in q for sig in crit_signals) or intent["sub_topic"]:
        intent["is_analysis"] = True
        return intent

    # [Default] FAQ
    if any(sig in q for sig in ["คืออะไร", "คือ", "หมายถึง"]):
        intent["is_faq"] = True

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
        sections.append(
            "### คำสั่งบังคับเด็ดขาด (STRICT THAI ONLY) ###\n"
            "1. ห้ามตอบเป็นภาษาอังกฤษแม้แต่คำเดียว (ยกเว้นชื่อไฟล์)\n"
            "2. แปลข้อมูลในตารางเป็นภาษาไทยทั้งหมด\n"
            "3. หากคุณตอบเป็นภาษาอังกฤษ คุณจะถูกตำหนิอย่างรุนแรง\n"
            "| หัวข้อเปรียบเทียบ | เอกสาร 1 | เอกสาร 2 | สรุปความแตกต่าง |"
        )

    sections.append(f"### คำถาม: {question}")
    sections.append("### กฎการตอบ: ตอบเป็นภาษาไทยสละสลวย หากถามภาษาอังกฤษมาให้แปลและสรุปเป็นไทยเสมอ")
    sections.append("### ย้ำอีกครั้ง: ต้องตอบเป็นภาษาไทยสละสลวยเท่านั้น (THAI ONLY) ###")

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