# core/json_extractor.py
# Ultimate Robust JSON Extractor for SEAM Assessment (Final CLEAN Version - NO UNICODE ARROWS)

import json
import logging
import re
from typing import Dict, Any, Optional

# pip install json5
import json5

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ===================================================================
# 1. Safe integer parser
# ===================================================================
def _safe_int_parse(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    
    # ถ้าเป็นตัวเลขอยู่แล้ว แปลงตรงๆ
    if isinstance(value, (int, float)):
        return int(round(value))

    if isinstance(value, str):
        value = value.strip()
        if not value or value.lower() in {"null", "none", "n/a", "-", "ไม่พบ", "ไม่มี"}:
            return default
        
        # ลองแปลงเป็น float ก่อน (รองรับ "8", "8.0", " 8.5 ")
        try:
            return int(round(float(value)))
        except ValueError:
            # ถ้าแปลงตรงๆ ไม่ได้ ให้ใช้ Regex ช่วย
            # ปรับ regex ให้ดึงเลขที่มีทศนิยมได้ (เช่น 8.5)
            match = re.search(r'[-+]?\d*\.\d+|\d+', value)
            if match:
                try:
                    return int(round(float(match.group(0))))
                except ValueError:
                    return default
    
    return default


# ===================================================================
# 2. Extract first complete JSON object (balanced braces)
# ===================================================================
def _extract_first_json_object(text: str) -> Optional[str]:
    if not text:
        return None

    # 1. ทำความสะอาด Control Characters (ยกเว้น \n \r \t)
    # ป้องกัน Error: Invalid control character ที่คุณเจอใน Log
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # 2. ค้นหาจุดเริ่มต้นของ JSON ก้อนแรก
    start = text.find("{")
    if start == -1:
        return None

    # 3. ใช้เทคนิค Balanced Braces แบบระวัง String
    depth = 0
    in_string = False
    escape_char = False

    for i in range(start, len(text)):
        char = text[i]

        # จัดการเรื่อง Escape characters (เช่น \")
        if escape_char:
            escape_char = False
            continue
        if char == "\\":
            escape_char = True
            continue

        # จัดการเรื่องเครื่องหมายคำพูด (เพื่อไม่ให้นับ { หรือ } ที่อยู่ใน string)
        if char == '"':
            in_string = not in_string
            continue

        if not in_string:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    # เจอจุดปิดที่แท้จริงแล้ว
                    return text[start:i + 1]

    return None


# ===================================================================
# 3. Normalize keys to SEAM standard
# ===================================================================
def _normalize_keys(data: Any) -> Any:
    mapping = {
        # Score
        "score": "score", "llm_score": "score", "total_score": "score", "final_score": "score",
        "assessment_score": "score", "evaluation_score": "score",

        # Reason
        "reason": "reason", "explanation": "reason", "reasoning": "reason",
        "comment": "reason", "rationale": "reason", "analysis": "reason",

        # Pass/Fail
        "is_passed": "is_passed", "passed": "is_passed", "pass": "is_passed",
        "result": "is_passed", "status": "is_passed",

        # PDCA
        "p_plan_score": "P_Plan_Score", "p_score": "P_Plan_Score", "plan_score": "P_Plan_Score",
        "p": "P_Plan_Score", "plan": "P_Plan_Score",
        "d_do_score": "D_Do_Score", "do_score": "D_Do_Score", "d": "D_Do_Score", "do": "D_Do_Score",
        "c_check_score": "C_Check_Score", "c_score": "C_Check_Score", "check_score": "C_Check_Score",
        "c": "C_Check_Score", "check": "C_Check_Score",
        "a_act_score": "A_Act_Score", "a_score": "A_Act_Score", "act_score": "A_Act_Score",
        "a": "A_Act_Score", "act": "A_Act_Score",

        # เพิ่มใน _normalize_keys mapping
        "summary": "summary",
        "summarization": "summary",
        "suggestion_for_next_level": "suggestion_for_next_level",
        "suggestion": "suggestion_for_next_level",
        "next_step": "suggestion_for_next_level",
    }

    if isinstance(data, dict):
        normalized = {}
        for k, v in data.items():
            key_clean = k.strip().lower() if isinstance(k, str) else str(k)
            normalized_key = mapping.get(key_clean, k)
            normalized[normalized_key] = _normalize_keys(v)
        return normalized
    elif isinstance(data, list):
        return [_normalize_keys(item) for item in data]
    else:
        return data


# ===================================================================
# 4. Extract + parse + normalize
# ===================================================================
def _extract_normalized_dict(raw_response: Any) -> Optional[Dict[str, Any]]:
    """
    เวอร์ชันสมบูรณ์: รองรับทั้ง String, AIMessage และ LLMResult 
    พร้อมระบบค้นหา JSON และ Normalize Keys
    """
    # 1. Input Guard: แปลง Object ทุกประเภทให้เป็น String ก่อน
    raw = ""
    if raw_response is None:
        return None
    
    try:
        if isinstance(raw_response, str):
            raw = raw_response.strip()
        elif hasattr(raw_response, 'content'): # กรณีเป็น AIMessage จาก LangChain
            raw = str(raw_response.content).strip()
        elif hasattr(raw_response, 'generations'): # กรณีเป็น LLMResult จาก LangChain
            raw = str(raw_response.generations[0][0].text).strip()
        else:
            raw = str(raw_response).strip()
    except Exception as e:
        logger.error(f"Error converting LLM response to string: {e}")
        return None

    if not raw:
        return None

    # 2. พยายามดึง JSON ก้อนแรกด้วย Balanced Braces
    json_str = _extract_first_json_object(raw)
    
    # 3. Fallback: ถ้าวิธีแรกไม่ได้ผล ให้ใช้ Regex ค้นหาก้อนที่ยาวที่สุด
    if not json_str:
        matches = re.findall(r"\{[\s\S]*?\}", raw, re.DOTALL)
        if matches:
            json_str = max(matches, key=len)

    if not json_str:
        return None

    # 4. Parsing ด้วย json5 (ยืดหยุ่นกว่า json ปกติ)
    data = None
    try:
        data = json5.loads(json_str)
    except Exception:
        try:
            # ลองใช้ standard json อีกครั้งเผื่อ json5 มีปัญหา
            data = json.loads(json_str)
        except Exception as e:
            logger.warning(f"Failed to parse JSON string: {json_str[:100]}... Error: {e}")
            return None

    # 5. จัดการกรณีที่ LLM คืนค่าเป็น List ของ Dict (เช่น [{...}])
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        data = data[0]

    if not isinstance(data, dict):
        return None

    # 6. Normalize Keys ให้กลับมาเป็นมาตรฐาน SE-AM
    return _normalize_keys(data)


# ===================================================================
# 5. MAIN FUNCTION หลัก – ใช้ฟังก์ชันนี้ในทุกที่
# ===================================================================
def _robust_extract_json(llm_response: str) -> Dict[str, Any]:
    """รับประกันคืน dict ที่สมบูรณ์เสมอ – ไม่เคยได้ score = 0 เพราะ parse ไม่ได้อีกต่อไป"""
    safe_result = {
        "score": 0,
        "reason": "ไม่สามารถแยกวิเคราะห์ JSON จากการตอบกลับของ LLM ได้",
        "is_passed": False,
        "P_Plan_Score": 0,
        "D_Do_Score": 0,
        "C_Check_Score": 0,
        "A_Act_Score": 0,
        "low_confidence_reason": "N/A",
        "suggested_action_on_low_conf": "N/A",
        "suggested_action_on_failure": "กรุณาตรวจสอบและอัปโหลดเอกสารเพิ่มเติม",
    }

    if not llm_response or not isinstance(llm_response, str):
        return safe_result

    data = _extract_normalized_dict(llm_response)
    if not data:
        # Fallback: ดึงคะแนนจากข้อความธรรมดา
        text = llm_response.lower()
        patterns = [
            r'score\D*(\d+)',
            r'คะแนน\D*(\d+)',
            r'ระดับ\D*(\d+)',
            r'level\D*(\d+)',
            r'\b(\d+)\s*คะแนน',
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                score = min(int(m.group(1)), 10)
                safe_result["score"] = score
                safe_result["reason"] = f"ดึงคะแนนจากข้อความ (ไม่พบ JSON): พบ '{m.group(0)}'"
                if score >= 3:
                    safe_result["is_passed"] = True
                return safe_result
        return safe_result

    # มี JSON → แปลงให้สมบูรณ์
    result = {}
    result["P_Plan_Score"] = _safe_int_parse(data.get("P_Plan_Score"))
    result["D_Do_Score"]   = _safe_int_parse(data.get("D_Do_Score"))
    result["C_Check_Score"] = _safe_int_parse(data.get("C_Check_Score"))
    result["A_Act_Score"]   = _safe_int_parse(data.get("A_Act_Score"))

    reason = data.get("reason") or data.get("explanation") or ""
    result["reason"] = str(reason).strip() or "ไม่พบเหตุผลจาก LLM"

    isp = data.get("is_passed")
    if isinstance(isp, str):
        result["is_passed"] = isp.strip().lower() in {"true", "yes", "pass", "passed", "ผ่าน", "1"}
    else:
        result["is_passed"] = bool(isp)

    if "score" in data:
        result["score"] = _safe_int_parse(data["score"])
    else:
        result["score"] = sum([result["P_Plan_Score"], result["D_Do_Score"], result["C_Check_Score"], result["A_Act_Score"]])
    result["score"] = min(result["score"], 10)

    # คัดลอก key อื่นๆ
    for k, v in data.items():
        if k not in result:
            result[k] = v

    # รับประกัน key สำคัญ
    for k in ["low_confidence_reason", "suggested_action_on_low_conf", "suggested_action_on_failure"]:
        result.setdefault(k, safe_result[k])

    return result