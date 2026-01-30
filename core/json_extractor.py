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
def _robust_extract_json(llm_response: str | Any) -> Dict[str, Any]:
    """
    [ULTIMATE ROBUST v2026.02.03 - Markdown/Thai/Chaos Safe]
    - รองรับ LLM ตอบแบบมี Markdown นำหน้า
    - ล้าง smart quotes, escape chars, trailing comma, control chars
    - Multi-stage extraction: code fence → greedy {..} → object แรก
    - json_repair เป็น rescue layer (ถ้ามี library)
    - Fallback ยังสมบูรณ์ + log raw sample
    """
    logger = logging.getLogger(__name__)

    # 1. Default Safe Structure
    safe_result = {
        "score": 0.0,
        "reason": "ไม่สามารถแยกวิเคราะห์ JSON ได้ (System Fallback)",
        "is_passed": False,
        "summary_thai": "ไม่พบข้อมูลสรุป",
        "coaching_insight": "ไม่พบข้อมูล",
        "P_Plan_Score": 0.0,
        "D_Do_Score": 0.0,
        "C_Check_Score": 0.0,
        "A_Act_Score": 0.0,
        "atomic_action_plan": []
    }

    # 2. Normalize input เป็น string
    if hasattr(llm_response, "content"):
        raw_text = llm_response.content
    else:
        raw_text = str(llm_response)

    if not raw_text.strip():
        logger.debug("[ROBUST-JSON] Empty response")
        return safe_result

    # 3. Pre-clean: ล้างสิ่งที่ LLM ชอบปนมา
    processed_text = raw_text.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
    processed_text = processed_text.replace('```json', '').replace('```markdown', '').replace('```', '').strip()
    processed_text = re.sub(r'[\x00-\x1F\x7F]', '', processed_text)  # ลบ control chars
    processed_text = re.sub(r',\s*([}\]])', r'\1', processed_text)  # ลบ trailing comma

    # 4. Multi-stage extraction
    json_str = None

    # Stage 4.1: หา block ใน ```json ... ```
    match = re.search(r'```json\s*([\s\S]*?)\s*```', processed_text, re.DOTALL | re.IGNORECASE)
    if match:
        json_str = match.group(1).strip()

    # Stage 4.2: หา { ... } ก้อนใหญ่สุด (greedy)
    if not json_str:
        match = re.search(r'(\{[\s\S]*\})', processed_text, re.DOTALL)
        if match:
            json_str = match.group(1).strip()

    # Stage 4.3: หา object แรก (กรณี LLM ตัดท้าย)
    if not json_str:
        match = re.search(r'\{[^}]*\}', processed_text)
        if match:
            json_str = match.group(0)

    # 5. พยายาม parse
    if json_str:
        try:
            data = json.loads(json_str)
            logger.debug(f"[ROBUST-JSON] Success - Keys: {list(data.keys())}")
            return data
        except json.JSONDecodeError as e:
            logger.warning(f"[ROBUST-JSON] Parse failed: {str(e)} → Trying repair/cleanup")

            # Cleanup เพิ่มเติม
            json_str_clean = re.sub(r',\s*([}\]])', r'\1', json_str)
            try:
                data = json.loads(json_str_clean)
                logger.debug("[ROBUST-JSON] Cleanup success")
                return data
            except:
                pass

            # Stage 5: json_repair (ถ้ามี library)
            try:
                from json_repair import repair_json
                repaired = repair_json(json_str)
                data = json.loads(repaired)
                logger.debug("[ROBUST-JSON] json_repair success")
                return data
            except (ImportError, Exception):
                pass

    # 6. Regex fallback (scavenge key-value)
    logger.warning("⚠️ JSON Parse failed → Regex scavenging")
    data = {}

    # หา score
    score_m = re.search(r'"score"\s*:\s*([\d\.]+)', processed_text)
    if score_m:
        data["score"] = float(score_m.group(1))

    # หา is_passed
    pass_m = re.search(r'"is_passed"\s*:\s*(true|false)', processed_text, re.I)
    if pass_m:
        data["is_passed"] = pass_m.group(1).lower() == "true"

    # หา reason / coaching_insight
    reason_m = re.search(r'"(reason|coaching_insight|summary_thai)"\s*:\s*"([^"]+)"', processed_text)
    if reason_m:
        data[reason_m.group(1)] = reason_m.group(2)

    # 7. Normalization & Merge with safe defaults
    result = {}
    result["reason"] = data.get("reason") or data.get("summary_thai") or data.get("explanation") or safe_result["reason"]
    result["coaching_insight"] = data.get("coaching_insight") or data.get("insight") or result["reason"]
    result["summary_thai"] = data.get("summary_thai") or result["reason"][:100]

    for k in ["P_Plan_Score", "D_Do_Score", "C_Check_Score", "A_Act_Score"]:
        val = data.get(k, 0.0)
        try:
            result[k] = float(val)
        except:
            result[k] = 0.0

    try:
        if "score" in data:
            result["score"] = float(data["score"])
        else:
            result["score"] = sum(result.get(k, 0.0) for k in ["P_Plan_Score", "D_Do_Score", "C_Check_Score", "A_Act_Score"])
    except:
        result["score"] = 0.0

    isp = data.get("is_passed")
    if isp is not None:
        result["is_passed"] = bool(isp) if not isinstance(isp, str) else isp.lower() == "true"
    else:
        result["is_passed"] = result["score"] >= 1.0

    raw_actions = data.get("atomic_action_plan") or data.get("action_plan") or []
    result["atomic_action_plan"] = raw_actions if isinstance(raw_actions, list) else []

    final_output = {**safe_result, **result}

    # Log raw sample ถ้า fallback
    if final_output["reason"] == safe_result["reason"]:
        logger.debug(f"[RAW-RESPONSE-SAMPLE] {processed_text[:300]}...")

    return final_output

def _robust_extract_json_list(raw_text: str) -> List[Dict[str, Any]]:
    """
    [HELPER - FINAL v2026.02.03]
    - ลบ fences, trailing comma, control chars
    - json_repair เป็น rescue
    - Regex หา [ ... ] หรือ { ... } ที่ใหญ่ที่สุด
    - Log ชัดเจน + sample text
    """
    if not raw_text or len(raw_text.strip()) < 2:
        return []

    original_sample = raw_text[:500] + "..." if len(raw_text) > 500 else raw_text

    # Pre-clean
    raw_text = re.sub(r'```(?:json)?\s*|\s*```', '', raw_text).strip()
    raw_text = re.sub(r',\s*([}\]])', r'\1', raw_text)
    raw_text = re.sub(r'[\x00-\x1F\x7F]', '', raw_text)

    # Stage 1: Direct parse
    try:
        data = json.loads(raw_text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass

    # Stage 2: json_repair (ถ้ามี)
    try:
        from json_repair import repair_json
        repaired = repair_json(raw_text)
        data = json.loads(repaired)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except (ImportError, Exception):
        pass

    # Stage 3: Regex หา array block ใหญ่สุด
    try:
        match = re.search(r'(\[[\s\S]*?\])', raw_text, re.DOTALL)
        if match:
            block = match.group(1)
            data = json.loads(block)
            if isinstance(data, list):
                return data
    except:
        pass

    # Stage 4: หา object แล้วหุ้มเป็น list
    try:
        match = re.search(r'(\{[\s\S]*?\})', raw_text, re.DOTALL)
        if match:
            block = match.group(1)
            data = json.loads(block)
            if isinstance(data, dict):
                return [data]
    except:
        pass

    # Fallback
    logger.warning(f"[ROBUST-LIST-FAIL] Cannot extract JSON list")
    logger.debug(f"[RAW-SAMPLE] {original_sample}")

    return []