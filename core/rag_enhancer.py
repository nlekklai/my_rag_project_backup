# rag_enhancer_v14.7.py

import logging
import re
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# ----------------------------------------------------------
# 1) Override เฉพาะจุด Critical (Rubric-based Override)
# ----------------------------------------------------------

OVERRIDE_MAP = {
    "KM": {
        "1.1": {
            2: (
                "**ปฏิบัติจริง** " * 7 +
                "**การทำงานจริง** " * 7 +
                "**การดำเนินงาน** " * 7 +
                "**กิจกรรม** " * 5 +
                "**ประชุม** " * 5 +
                "**ผลการดำเนินงาน** " * 5 +
                "คณะทำงาน KM คณะกรรมการ KM ทีม KM "
                "แต่งตั้งคณะทำงาน บทบาท หน้าที่ ความรับผิดชอบ ToR "
                "ตัวแทนสายงาน ตัวแทนหน่วยงาน cross-functional "
                "steering committee คณะกรรมการอำนวยการ KM"
            ),
            3: (
                "**การติดตามผลลัพธ์ตามวิสัยทัศน์** " * 5 +
                "**การประเมินผลโดยเทียบกับนโยบาย** " * 5 +
                "**KPI** " * 5 +
                "**KMA** " * 5 +
                "**รายงานผลการดำเนินงาน** " * 5 +
                "สรุปผล การประเมินผล การติดตามผล การทบทวน การตรวจสอบ "
                "วิสัยทัศน์ นโยบาย"
            ),
        }
    }
}

def override_query_if_needed(enabler_id: str, main_criteria_id: str, current_level: int) -> Optional[str]:
    """Return override query if required, else None."""
    return OVERRIDE_MAP.get(enabler_id, {}).get(main_criteria_id, {}).get(current_level, None)


# ----------------------------------------------------------
# 2) Boost Keyword ตาม Rubric ของแต่ละ Level
# ----------------------------------------------------------

BOOST_MAP = {
    "KM": {
        "1.1": {
            1: (
                "นโยบาย นโยบาย นโยบาย นโยบาย นโยบาย "
                "วิสัยทัศน์ ทิศทางกลยุทธ์ แผนกลยุทธ์ การกำหนดทิศทาง"
            ),
            4: (
                "**นวัตกรรม** นวัตกรรม นวัตกรรม นวัตกรรม นวัตกรรม "
                "**ปรับปรุง** ปรับปรุง **ทบทวน** ทบทวน **บูรณาการ** บูรณาการ "
                "ผลกระทบระยะยาว ความยั่งยืน "
                "การทบทวน การปรับปรุง การวัดผล การประเมิน การกำกับดูแล"
            ),
            5: (
                "**นวัตกรรม** นวัตกรรม นวัตกรรม นวัตกรรม นวัตกรรม "
                "**ปรับปรุง** ปรับปรุง **ทบทวน** ทบทวน **บูรณาการ** บูรณาการ "
                "ผลกระทบระยะยาว ความยั่งยืน "
                "การทบทวน การปรับปรุง การวัดผล การประเมิน การกำกับดูแล"
            )
        },
        "2.1": {
            2: "การรวบรวมองค์ความรู้ แหล่งความรู้ tacit explicit capture",
            3: "กระบวนการจัดเก็บ วงจร KM documentation workflow",
            4: "ผลการเก็บความรู้จริง รายงาน capture log",
            5: "ผลการเก็บความรู้จริง รายงาน capture log"
        }
    }
}

def get_boost_keywords(enabler_id: str, main_criteria_id: str, current_level: int) -> str:
    """Return keyword boost string aligned with rubric."""
    return BOOST_MAP.get(enabler_id, {}).get(main_criteria_id, {}).get(current_level, "")


# ----------------------------------------------------------
# 3) Final Query Generator
# ----------------------------------------------------------

def enhance_query(enabler_id: str, main_criteria_id: str, current_level: int, statement_text: str):
    """Core function used by Retrieval Pipeline."""

    debug = {}

    # หาก level ไม่ได้ส่งมา ให้ลอง parse จาก statement_text
    if current_level == 0:
        try:
            level_match = re.search(r'L(\d)', statement_text)
            current_level = int(level_match.group(1)) if level_match else 0
        except Exception as e:
            logger.warning(f"[RAG v14.7] Could not parse level from statement_text: {e}")
            current_level = 0

    base_query = statement_text.strip()
    debug["base_query"] = base_query

    # Step 1: Override?
    override = override_query_if_needed(enabler_id, main_criteria_id, current_level)
    if override:
        final_query = override
        debug["mode"] = "override"
        debug["final"] = final_query
        return final_query, debug

    # Step 2: Boost logic
    boost = get_boost_keywords(enabler_id, main_criteria_id, current_level)
    debug["boost"] = boost

    if boost:
        final_query = f"{base_query} {boost}"
        debug["mode"] = "boost"
    else:
        final_query = base_query
        debug["mode"] = "base"

    debug["final"] = final_query
    return final_query, debug


# ----------------------------------------------------------
# 4) Wrapper สำหรับ external call
# ----------------------------------------------------------

def enhance_query_for_statement(
    statement_text: str,
    enabler_id: Optional[str] = None,
    main_criteria_id: Optional[str] = None,
    current_level: int = 0,
    statement_id: Optional[str] = None,
    focus_hint: Optional[str] = None
) -> str:
    """Wrapper ที่รับ input ตาม V11/V12/V13/V14 และเรียกใช้ enhance_query"""

    e_id = enabler_id or "KM"
    m_id = main_criteria_id or "1.1"
    c_level = current_level or 0

    final_query, dbg = enhance_query(e_id, m_id, c_level, statement_text)
    logger.debug(f"[RAG v14.7] Query Generated → {final_query} | DEBUG: {dbg}")
    return final_query
