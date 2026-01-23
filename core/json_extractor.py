# core/json_extractor.py
# Ultimate Robust JSON Extractor for SEAM Assessment (Final CLEAN Version - NO UNICODE ARROWS)

import json
import logging
import re
from typing import Dict, Any, Optional, List

# pip install json5
import json5

# ใช้ json_repair ถ้ามี (คุณติดตั้งแล้ว)
try:
    from json_repair import repair_json
except ImportError:
    repair_json = None


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

def _extract_first_json_object(text: str) -> Optional[str]:
    if not text:
        return None

    # 1. ทำความสะอาด Control Characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    # 2. ค้นหาจุดเริ่มต้น (รองรับทั้ง { และ [)
    start_brace = text.find("{")
    start_bracket = text.find("[")
    
    if start_brace == -1 and start_bracket == -1: return None
    
    if start_brace == -1: start = start_bracket
    elif start_bracket == -1: start = start_brace
    else: start = min(start_brace, start_bracket)
    
    opening_char = text[start]
    closing_char = "}" if opening_char == "{" else "]"

    # 3. ใช้เทคนิค Balanced Braces
    depth = 0
    in_string = False
    escape_char = False

    for i in range(start, len(text)):
        char = text[i]

        if escape_char:
            escape_char = False
            continue
        if char == "\\":
            escape_char = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if not in_string:
            if char == opening_char:   # ✅ แก้ไขตรงนี้
                depth += 1
            elif char == closing_char: # ✅ แก้ไขตรงนี้
                depth -= 1
                if depth == 0:
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
    """
    [ULTIMATE ROBUST REVISE - v2026.1.23]
    - รองรับ Nested Braces (วงเล็บปีกกาซ้อน) ด้วย Recursive Regex
    - แก้ปัญหา JSON Syntax Error จากเครื่องหมายคำพูด (Quotes)
    - ระบบ Multi-Key Aliasing สำหรับดึง Reason/Score
    """
    logger = logging.getLogger(__name__)
    
    # 1. 🛡️ Default Safe Structure
    safe_result = {
        "score": 0.0,
        "reason": "ไม่สามารถแยกวิเคราะห์ JSON ได้ (System Fallback)",
        "is_passed": False,
        "summary_thai": "ไม่พบข้อมูลสรุป",
        "coaching_insight": "ไม่พบข้อมูล",
        "P_Plan_Score": 0.0, "D_Do_Score": 0.0, "C_Check_Score": 0.0, "A_Act_Score": 0.0,
        "atomic_action_plan": []
    }

    if not llm_response:
        return safe_result

    raw_text = getattr(llm_response, 'content', str(llm_response)).strip()
    
    # Pre-Sanitize: ล้าง Smart Quotes และอักขระพิเศษ
    processed_text = raw_text.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
    processed_text = processed_text.replace('```json', '').replace('```', '').strip()

    # 2. 🧩 Extraction Strategy: หา { ... } ก้อนที่สมบูรณ์ที่สุด
    data = {}
    try:
        # ใช้ Greedy Match หาตั้งแต่ { จนถึง } ตัวสุดท้าย
        match = re.search(r'(\{.*\})', processed_text, re.DOTALL)
        if match:
            json_str = match.group(1)
            # พยายาม Parse (ใช้ json5 ถ้ามีจะดีมาก แต่ถ้าไม่มีใช้ json ปกติ)
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                # ถ้าพัง พยายามซ่อมแซม Common Issues เช่น ลืมใส่คอมม่า หรือ Quote ซ้อน
                # (ขั้นตอนนี้ใช้ความสามารถของ Regex ช่วยเบื้องต้น)
                clean_json_str = re.sub(r',\s*\}', '}', json_str) # ลบคอมม่าเกินหน้า }
                data = json.loads(clean_json_str)
    except Exception:
        # 3. 📉 Regex Fallback Layer: ถ้า JSON พัง 100% ให้ควานหาทีละ Key
        logger.warning("⚠️ JSON Parse failed. Engaging Regex Key-Value scavenging.")
        # หา Score
        score_m = re.search(r'"score"\s*:\s*([\d\.]+)', processed_text)
        if score_m: data["score"] = float(score_m.group(1))
        
        # หา is_passed
        pass_m = re.search(r'"is_passed"\s*:\s*(true|false)', processed_text, re.I)
        if pass_m: data["is_passed"] = pass_m.group(1).lower() == "true"
        
        # หา Reason
        reason_m = re.search(r'"reason"\s*:\s*"([^"]+)"', processed_text)
        if reason_m: data["reason"] = reason_m.group(1)

    # 4. 🏗️ Normalization & Mapping (แมตช์ข้อมูลเข้า UI Engine)
    result = {}
    
    # 💡 Key-Aliasing Logic: ดักคำที่ AI ชอบเขียนเพี้ยน
    result["reason"] = (data.get("reason") or data.get("summary_thai") or data.get("explanation") or safe_result["reason"])
    result["coaching_insight"] = (data.get("coaching_insight") or data.get("insight") or result["reason"])
    result["summary_thai"] = (data.get("summary_thai") or result["reason"][:100])

    # จัดการคะแนน PDCA (แปลงเป็น Float เพื่อความละเอียด)
    for k in ["P_Plan_Score", "D_Do_Score", "C_Check_Score", "A_Act_Score"]:
        val = data.get(k, 0.0)
        try:
            result[k] = float(val)
        except:
            result[k] = 0.0

    # จัดการ Score รวม
    try:
        if "score" in data:
            result["score"] = float(data["score"])
        else:
            # ถ้าไม่มี Score รวม ให้คำนวณจาก PDCA เฉลี่ย (Max 2.0 per phase = 8.0)
            result["score"] = sum([result[k] for k in ["P_Plan_Score", "D_Do_Score", "C_Check_Score", "A_Act_Score"]])
    except:
        result["score"] = 0.0

    # จัดการ is_passed (ถ้าไม่ส่งมา ให้ใช้เกณฑ์คะแนน >= 1.0 สำหรับ L1-L2)
    isp = data.get("is_passed")
    if isp is not None:
        result["is_passed"] = bool(isp) if not isinstance(isp, str) else isp.lower() == "true"
    else:
        result["is_passed"] = result["score"] >= 1.0

    # จัดการ Atomic Action Plan
    raw_actions = data.get("atomic_action_plan") or data.get("action_plan") or []
    result["atomic_action_plan"] = raw_actions if isinstance(raw_actions, list) else []

    # 5. 📎 ประกันความสมบูรณ์ (Merge with Safe Defaults)
    final_output = {**safe_result, **result}
    
    return final_output

def _robust_extract_json_list(raw_text: str) -> List[Dict[str, Any]]:
    """
    [HELPER - FULL REVISED v2026.1.25]
    สกัด List ของ JSON ออกจากข้อความอย่างปลอดภัยที่สุด
    - ลอง parse ตรง ๆ ก่อน
    - ใช้ json_repair กู้ถ้าพัง
    - Regex หา block JSON + manual clean-up
    - Log error ชัดเจนเพื่อ debug
    - คืน [] ถ้าทำอะไรไม่ได้เลย
    """
    if not raw_text or len(raw_text.strip()) < 2:
        return []

    original_text = raw_text  # เก็บไว้ log

    # Stage 1: ลบ markdown/code fences ก่อน
    raw_text = re.sub(r'```(?:json)?\s*|\s*```', '', raw_text).strip()

    # Stage 2: ลบ whitespace เกิน + trailing comma
    raw_text = re.sub(r',\s*([}\]])', r'\1', raw_text)
    raw_text = re.sub(r'\s+', ' ', raw_text).strip()

    # Stage 3: ลอง parse ตรง ๆ
    try:
        data = json.loads(raw_text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]  # หุ้มเป็น list ถ้าได้ object เดียว
    except json.JSONDecodeError as e:
        pass  # ไป stage ถัดไป

    # Stage 4: ใช้ json_repair ถ้ามี (ดีมากสำหรับ LLM output พัง)
    if repair_json:
        try:
            repaired = repair_json(raw_text)
            data = json.loads(repaired)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return [data]
        except Exception as repair_err:
            pass  # ถ้า repair พัง ไป manual

    # Stage 5: Manual regex + clean-up
    try:
        # หา [ ... ] ก้อนใหญ่สุด
        list_match = re.search(r'(\[[\s\S]*?\])', raw_text, re.DOTALL)
        if list_match:
            block = list_match.group(1)
            # ลบ control chars + unbalanced
            block = re.sub(r'[\x00-\x1F\x7F]', '', block)
            data = json.loads(block)
            if isinstance(data, list):
                return data

        # หา { ... } แล้วหุ้มเป็น list
        dict_match = re.search(r'(\{[\s\S]*?\})', raw_text, re.DOTALL)
        if dict_match:
            block = dict_match.group(1)
            block = re.sub(r'[\x00-\x1F\x7F]', '', block)
            data = json.loads(block)
            if isinstance(data, dict):
                return [data]

    except json.JSONDecodeError as je:
        # Log error ชัดเจน
        logger.warning(f"[ROBUST-EXTRACT-FAIL] Failed to parse JSON block: {str(je)}")
        logger.debug(f"[RAW-TEXT-SAMPLE] {original_text[:300]}...")

    # Ultimate fallback
    return []