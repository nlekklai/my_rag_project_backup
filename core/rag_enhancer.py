# rag_enhancer_v15.1.py (PDCA-aware, safe-override, main_criteria auto-derive)
import logging
import re
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------
# OVERRIDE MAP (minimal, keywords to APPEND)
# Keep overrides concise and content-focused (not repeated tokens)
# ---------------------------
OVERRIDE_MAP = {
    "KM": {
        "1.1": {
            2: "ปฏิบัติจริง การดำเนินงาน กิจกรรม ประชุม ผลการดำเนินงาน คณะทำงาน KM คณะกรรมการ KM ToR บทบาท หน้าที่",
            3: "รายงานผล การประเมิน KPI การติดตาม สรุปผล การทบทวน วิสัยทัศน์ นโยบาย"
        }
    }
}

# ---------------------------
# BOOST MAP per criteria/level (concise, content + verbs)
# ---------------------------
BOOST_MAP = {
    "KM": {
        "1.1": {
            1: "นโยบาย วิสัยทัศน์ ทิศทาง แผนกลยุทธ์ การกำหนดทิศทาง",
            4: "ปรับปรุง บูรณาการ ขยายผล การวัดผล ผลกระทบระยะยาว",
            5: "นวัตกรรม ผลลัพธ์เชิงกลยุทธ์ คุณค่าทางธุรกิจ ความยั่งยืน"
        },
        "1.2": {
            1: "ผู้บริหาร นโยบาย การสนับสนุน การกำกับ",
            2: "การมีส่วนร่วม ผู้บริหารระดับสูง คำสั่ง แต่งตั้ง",
            3: "ประเมินผล รายงานสรุป การติดตาม การวัดผล",
            4: "ปรับปรุง บูรณาการ การขยายผล",
            5: "ผลกระทบระยะยาว นวัตกรรม"
        },
        "2.1": {
            1: "ขอบเขตความรู้ แผนการจัดการความรู้ Knowledge Map",
            2: "การรวบรวมความรู้ tacit explicit capture กระบวนการเก็บ",
            3: "ระบบจัดเก็บ การตรวจสอบบันทึก documentation workflow",
            4: "รายงานการเก็บความรู้ ผลการ capture การปรับปรุง",
            5: "การใช้ความรู้สร้างคุณค่า รายงานผลเชิงธุรกิจ"
        },
        "2.2": {
            1: "แนวทางการจัดสรรทรัพยากร งบประมาณ บุคลากร เครื่องมือ",
            2: "การจัดสรรจริง การอนุมัติงบประมาณ การจัดหาเครื่องมือ",
            3: "การติดตามการใช้ทรัพยากร รายงานการใช้ ตรวจสอบงบประมาณ",
            4: "ปรับแผนงบประมาณ การทบทวนความคุ้มค่า",
            5: "การใช้ทรัพยากรร่วมกันอย่างยั่งยืน"
        }
        # add other criteria as needed...
    }
}

# ---------------------------
# PDCA CORE KEYWORDS (verbs / intents) — will be appended per level
# These ensure RAG focuses on the PDCA action type required by level
# ---------------------------
PDCA_CORE_KEYWORDS = {
    1: "กำหนด จัดทำ วางแผน กรอบแนวทาง เริ่มต้น",
    2: "ดำเนินการ ปฏิบัติ ขับเคลื่อน ประชุม จัดกิจกรรม",
    3: "ติดตาม ตรวจสอบ ประเมิน สรุปผล รายงาน",
    4: "ปรับปรุง แก้ไข บูรณาการ ขยายผล พัฒนา",
    5: "นวัตกรรม สร้างมูลค่า ผลเชิงกลยุทธ์ ความยั่งยืน"
}


# ---------------------------
# Helper: derive main_criteria from statement_id if missing
# Expecting formats like "1.1", "2.2", or "1.1.L2" or "1.1_L2" or "1.1.L2.uuid"
# We'll return e.g. "1.1" or fallback to provided default
# ---------------------------
def derive_main_criteria(main_criteria_id: Optional[str], statement_id: Optional[str]) -> str:
    if main_criteria_id:
        return main_criteria_id
    if not statement_id:
        return "1.1"
    # try common separators
    try:
        # extract leading "X.Y" pattern
        m = re.search(r"(\d+\.\d+)", statement_id)
        if m:
            return m.group(1)
        # fallback: first token before separator
        parts = re.split(r"[._\-]", statement_id)
        if parts:
            # if first is like '1' and second is '1' then join
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                return f"{parts[0]}.{parts[1]}"
            return parts[0]
    except Exception:
        pass
    return "1.1"


# ---------------------------
# get override keywords (if any) — returns string to append (not replace)
# ---------------------------
def override_query_if_needed(enabler_id: str, main_criteria_id: str, current_level: int) -> Optional[str]:
    return OVERRIDE_MAP.get(enabler_id, {}).get(main_criteria_id, {}).get(current_level, None)


# ---------------------------
# get boost keywords from map (concise)
# ---------------------------
def get_boost_keywords(enabler_id: str, main_criteria_id: str, current_level: int) -> str:
    return BOOST_MAP.get(enabler_id, {}).get(main_criteria_id, {}).get(current_level, "")


# ---------------------------
# Core enhancer
# Returns (final_query, debug_info)
# ---------------------------
def enhance_query(enabler_id: str, main_criteria_id: str, current_level: int, statement_text: str, statement_id: Optional[str] = None):
    debug: Dict[str, Any] = {}

    # Derive main criteria if not provided
    m_id = derive_main_criteria(main_criteria_id, statement_id)
    debug["derived_main_criteria"] = m_id

    # If current_level is 0 try to parse from statement_text or statement_id
    if current_level == 0:
        level = 0
        # try statement_text first
        try:
            lm = re.search(r"L(\d)", statement_text, re.IGNORECASE)
            if lm:
                level = int(lm.group(1))
        except Exception:
            level = 0
        # fallback to statement_id
        if level == 0 and statement_id:
            try:
                lm2 = re.search(r"L(\d)", statement_id, re.IGNORECASE)
                if lm2:
                    level = int(lm2.group(1))
            except Exception:
                level = 0
        # final fallback to 1
        current_level = level if level > 0 else 1
    debug["resolved_level"] = current_level

    base_query = (statement_text or "").strip()
    debug["base_query"] = base_query

    # Step A: get override keywords (if any) and append (do NOT replace statement)
    override_kw = override_query_if_needed(enabler_id, m_id, current_level)
    if override_kw:
        debug["override_used"] = True
    else:
        debug["override_used"] = False

    # Step B: get boost keywords from BOOST_MAP
    boost_kw = get_boost_keywords(enabler_id, m_id, current_level)
    debug["boost_present"] = bool(boost_kw)

    # Step C: PDCA core keywords (verbs/intents)
    pdca_kw = PDCA_CORE_KEYWORDS.get(current_level, "")
    debug["pdca_core"] = pdca_kw

    # Merge pieces: base + (override if any) + boost + pdca_core
    # ORDER: base_statement -> override (high-priority append) -> boost (criteria-level) -> pdca verbs
    parts = [base_query] if base_query else []
    if override_kw:
        parts.append(override_kw)
    if boost_kw:
        parts.append(boost_kw)
    if pdca_kw:
        parts.append(pdca_kw)

    final_query = " ".join(parts)

    # Normalize whitespace and remove duplicated spaces
    final_query = re.sub(r"\s+", " ", final_query).strip()

    # Lower-risk: avoid excessive quotes/characters for the vector search
    # (do not lowercase Thai, but normalize ASCII punctuation)
    final_query = final_query.replace('"', '')

    debug["final_query"] = final_query
    debug["mode"] = "override_append" if override_kw else ("boost" if boost_kw else "base")

    logger.debug(f"[RAG v15.1] Final Query → {final_query} | DEBUG: {debug}")
    return final_query, debug


# ---------------------------
# Wrapper (compatibility)
# signature kept same as previous versions
# ---------------------------
def enhance_query_for_statement(
    statement_text: str,
    enabler_id: Optional[str] = None,
    main_criteria_id: Optional[str] = None,
    current_level: int = 0,
    statement_id: Optional[str] = None,
    focus_hint: Optional[str] = None
) -> str:
    """
    External wrapper used by retrieval pipeline.
    - Auto-derives main_criteria_id from statement_id if not provided.
    - Ensures override is appended (not replacing base statement).
    - Adds PDCA verb-focused keywords.
    """
    e_id = enabler_id or "KM"
    m_id = derive_main_criteria(main_criteria_id, statement_id)
    c_level = current_level or 0

    final_query, dbg = enhance_query(e_id, m_id, c_level, statement_text or "", statement_id=statement_id)
    # keep an info-level log of the produced query (helpful in production)
    logger.info(f"[RAG v15.1] Query Generated for {m_id} L{c_level} → {final_query}")
    return final_query
