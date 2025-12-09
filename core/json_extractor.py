import json
import logging
from typing import Dict, Any, Optional, List
import re
import json5 # ต้องแน่ใจว่าได้ติดตั้ง pip install json5 แล้ว

logger = logging.getLogger(__name__)
if not logger.handlers:
    # ตั้งค่า Log พื้นฐานหากยังไม่มี (ปรับระดับตามต้องการ)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ------------------------------------------------------------
# Constants & Helpers
# ------------------------------------------------------------
# UUID_PATTERN ยังไม่ได้ถูกใช้ในโค้ดปัจจุบัน
UUID_PATTERN = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', re.IGNORECASE)

def _safe_int_parse(value: Any, default: int = 0) -> int:
    """Safely converts value to an integer, handles strings like '2/2', '1 (ดี)' etc."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        value = value.strip()
        if not value or value.lower() in {"null", "none", "n/a", "-"}:
            return default
        # ดึงตัวเลขแรกที่เจอ เช่น "2/2" → 2, "1 (ดีมาก)" → 1
        match = re.search(r'\d+', value)
        if match:
            return int(match.group(0))
    return default


# ------------------------------------------------------------
# Balanced Brace Extractor (ป้องกัน nested + fenced)
# ------------------------------------------------------------
def _extract_balanced_braces(text: str) -> Optional[str]:
    """Extract the first complete JSON object using balanced brace counting."""
    if not text:
        return None

    # ตัดหลังจากเจอ ``` เพื่อไม่ให้ไปจับ JSON ใน code block ตัวอย่าง
    fence_pos = text.find("```")
    scan_text = text if fence_pos == -1 else text[:fence_pos]

    start = scan_text.find('{')
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(scan_text)):
        if scan_text[i] == '{':
            depth += 1
        elif scan_text[i] == '}':
            depth -= 1
            if depth == 0:
                return scan_text[start:i+1]
    return None


# ------------------------------------------------------------
# Key Normalization
# ------------------------------------------------------------
def _normalize_keys(data: Any) -> Any:
    """Recursively normalize dictionary keys to standard assessment format."""
    mapping = {
        # Score & Status
        "llm_score": "score", "total_score": "score", "final_score": "score",
        "reasoning": "reason", "llm_reasoning": "reason", "assessment_reason": "reason",
        "explanation": "reason", "comment": "reason", "rationale": "reason",
        "pass": "is_passed", "is_pass": "is_passed", "passed": "is_passed", "result": "is_passed",

        # PDCA Scores (รองรับทุกชื่อที่เคยเจอจริง)
        "p_score": "P_Plan_Score", "plan_score": "P_Plan_Score", "p": "P_Plan_Score", "plan": "P_Plan_Score",
        "d_score": "D_Do_Score", "do_score": "D_Do_Score", "d": "D_Do_Score", "do": "D_Do_Score",
        "c_score": "C_Check_Score", "check_score": "C_Check_Score", "c": "C_Check_Score", "check": "C_Check_Score",
        "a_score": "A_Act_Score", "act_score": "A_Act_Score", "a": "A_Act_Score", "act": "A_Act_Score",

        "p_plan_score": "P_Plan_Score", "d_do_score": "D_Do_Score",
        "c_check_score": "C_Check_Score", "a_act_score": "A_Act_Score",
    }

    if isinstance(data, dict):
        normalized = {}
        for k, v in data.items():
            key_lower = k.strip().lower() if isinstance(k, str) else k
            std_key = mapping.get(key_lower, k)
            normalized[std_key] = _normalize_keys(v)
        return normalized

    if isinstance(data, list):
        return [_normalize_keys(item) for item in data]

    return data


# ------------------------------------------------------------
# Core Extractor
# ------------------------------------------------------------

def _extract_normalized_dict(llm_response: str) -> Optional[Dict[str, Any]]:
    """
    Extract and normalize JSON from LLM response using balanced brace counting 
    and robust JSON parsing (json5 -> json).
    """
    raw = (llm_response or "").strip()
    if not raw:
        return None

    # 1. PRIORITY 1: ใช้ Balanced Brace Extractor (ดีกว่า regex ดิบสำหรับ nested JSON)
    best_match: Optional[str] = _extract_balanced_braces(raw)

    # 2. PRIORITY 2: ถ้า Balanced Brace ไม่เจอ → ลองใช้ Regex ดิบเพื่อหา Object ที่ยาวที่สุด
    if not best_match:
        # \{[\s\S]*?\} : หาตั้งแต่ { แรก จนถึง } สุดท้าย
        # re.DOTALL: สำคัญมาก เพื่อให้ . match \n (multiline JSON)
        matches = re.findall(r"\{[\s\S]*?\}", raw, re.DOTALL)
        
        if not matches:
            return None

        # เลือก match ที่ยาวที่สุด (น่าจะเป็น JSON object หลัก)
        best_match = max(matches, key=len)
    
    if not best_match:
        return None

    # 3. Parse JSON (ใช้ json5 ก่อน เพราะทนทานกว่า)
    data = None
    try:
        data = json5.loads(best_match)
    except Exception as e_json5:
        logger.debug(f"json5 failed on best match: {e_json5}")
        # Fallback 4: ลองใช้ standard json
        try:
            data = json.loads(best_match)
        except Exception as e_json:
            logger.debug(f"Standard json failed: {e_json}")
            return None # Cannot parse

    # กรณีที่ LLM ตอบกลับมาเป็น Array ที่มี Object เดียว (เจอบ่อยใน Action Plan)
    if isinstance(data, list) and data and isinstance(data[0], dict):
        logger.warning("Extracted JSON is an array containing dict(s). Using the first dict.")
        data = data[0]

    if not isinstance(data, dict):
        return None

    # 4. Normalize keys (ใช้ฟังก์ชันเดิมของคุณ)
    return _normalize_keys(data)


# ------------------------------------------------------------
# Final Assessment JSON Extractor (Production Ready)
# ------------------------------------------------------------
def _robust_extract_json(llm_response: str) -> Dict[str, Any]:
    """
    Ultimate robust JSON extractor for SEAM Assessment.
    รับประกันคืน dict ที่มี key ครบ + score ถูกต้อง เสมอ
    """
    fallback = {
        "score": 0,
        "reason": "Failed to extract valid JSON from LLM response.",
        "is_passed": False,
        "P_Plan_Score": 0,
        "D_Do_Score": 0,
        "C_Check_Score": 0,
        "A_Act_Score": 0,
        # ต้องใส่ Fields ใหม่ เพื่อป้องกัน KeyError ใน Logic ภายนอก
        "low_confidence_reason": "N/A",  
        "suggested_action_on_low_conf": "N/A",  
        "suggested_action_on_failure": "Failed to extract JSON/data." 
    }

    if not llm_response or not isinstance(llm_response, str):
        return fallback

    data = _extract_normalized_dict(llm_response)
    if not data:
        return fallback

    result = {}

    # 1. Extract PDCA scores
    result["P_Plan_Score"] = _safe_int_parse(data.get("P_Plan_Score", 0))
    result["D_Do_Score"]   = _safe_int_parse(data.get("D_Do_Score", 0))
    result["C_Check_Score"] = _safe_int_parse(data.get("C_Check_Score", 0))
    result["A_Act_Score"]   = _safe_int_parse(data.get("A_Act_Score", 0))

    # 2. Reason
    reason = data.get("reason") or data.get("explanation") or ""
    result["reason"] = str(reason).strip() or "No reason provided by LLM."

    # 3. is_passed
    isp = data.get("is_passed")
    if isinstance(isp, str):
        result["is_passed"] = isp.strip().lower() in {"true", "yes", "pass", "passed", "ผ่าน"}
    else:
        result["is_passed"] = bool(isp)

    # 4. Final score: ใช้ score จาก LLM ก่อน → ถ้าไม่มีค่อยรวม P+D+C+A
    explicit_score = data.get("score")
    if explicit_score is not None:
        result["score"] = _safe_int_parse(explicit_score)
    else:
        result["score"] = sum([
            result["P_Plan_Score"],
            result["D_Do_Score"],
            result["C_Check_Score"],
            result["A_Act_Score"]
        ])

    # Optional: ป้องกัน score เกิน (L5 max = 10)
    result["score"] = min(result["score"], 10)

    # 5. รวม Key ที่ไม่ได้ Normalize (เช่น low_confidence_reason, suggested_action_on_failure) กลับเข้ามา
    for k, v in data.items():
        # ป้องกันการเขียนทับ key หลักที่ถูก Parse แล้ว (score, reason, is_passed, PDCA scores)
        if k not in result:
             # สำหรับ Actionable Fields และ Confidence Fields
             result[k] = v 
    
    # 6. รับประกันว่า Field ใหม่มีค่าเสมอแม้ LLM จะไม่ส่งมา (ป้องกัน KeyError ใน Logic ภายนอก)
    result["low_confidence_reason"] = result.get("low_confidence_reason", "N/A")
    result["suggested_action_on_low_conf"] = result.get("suggested_action_on_low_conf", "N/A")
    result["suggested_action_on_failure"] = result.get("suggested_action_on_failure", "ระบุหลักฐานที่ขาดหาย")
    

    return result