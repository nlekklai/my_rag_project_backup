# assessments/mocking_assessment.py
"""
Mocking Assessment Utilities
ใช้สำหรับทดสอบระบบ KM/Enabler Assessment โดยไม่ต้องเรียก LLM จริง
"""

import random
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


# -------------------------------------------------------
# MOCK: Controlled LLM Evaluation (Deterministic)
# -------------------------------------------------------
def evaluate_with_llm_CONTROLLED_MOCK(
    statement: str,
    context: str,
    level: int = 1,
    sub_criteria_id: str = "UNKNOWN",
    statement_number: int = 1,
    **kwargs
) -> Dict[str, Any]:
    """
    Mock การประเมินผลลัพธ์แบบควบคุมลำดับ (ใช้แทน LLM จริง)
    รุ่น Debug + Strict Mock สำหรับ sub_criteria 1.2
    """
    # ตรวจสอบ kwargs
    level = kwargs.get("level", level)
    sub_criteria_id = kwargs.get("sub_criteria_id", sub_criteria_id)
    statement_number = kwargs.get("statement_number", statement_number)

    # Trim & debug log
    sub_id = str(sub_criteria_id).strip()
    logger.info(f"[DEBUG MOCK INPUT] sub_criteria_id={sub_id}, level={level}, statement_number={statement_number}")

    # ✅ Logic controlled mock
    if sub_id == "1.2":
        # L1-L3 ผ่านหมด
        if level in [1,2,3]:
            score = 1
        elif level == 4:
            # Fail statement 2 ของ L4
            score = 0 if statement_number == 2 else 1
        elif level == 5:
            # Fail statements 1 & 3 ของ L5
            score = 0 if statement_number in [1,3] else 1
        else:
            score = 0
    else:
        # Default mock logic
        if level == 1:
            score = 1
        elif level == 2:
            score = 1 if statement_number % 2 == 1 else 0
        elif level == 3:
            score = 1
        elif level in [4,5]:
            score = 0
        else:
            score = 0

    is_passed = score == 1

    mock_context_snippet = f"[MOCK CONTEXT SNIPPET] Evidence found for {sub_id} L{level} S{statement_number}." if is_passed else ""
    mock_sources = [
        {"source_name": f"mock_doc_{sub_id}_L{level}_S{statement_number}.pdf",
         "location": f"page_{10+statement_number}",
         "doc_id": f"DOC_{sub_id}"}
    ] if is_passed else []

    return {
        "sub_criteria_id": sub_id,
        "level": level,
        "statement_number": statement_number,
        "statement": statement,
        "context_retrieved_snippet": mock_context_snippet,
        "retrieved_sources_list": mock_sources,
        "llm_score": score,
        "score": score,
        "reason": f"MOCK reason for L{level} S{statement_number} → {'PASS' if is_passed else 'FAIL'} (Controlled Mock)",
        "pass_status": is_passed,
        "status_th": "ผ่าน" if is_passed else "ไม่ผ่าน",
        "llm_result": {"is_passed": is_passed, "score": float(score)}
    }



# -------------------------------------------------------
# MOCK: Retrieval
# -------------------------------------------------------
def retrieve_context_MOCK(statement: str, sub_criteria_id: str, level: int, statement_number: int, mapping_data=None, **kwargs) -> Dict[str, Any]:
    """
    Mock retrieval context จาก Vectorstore (ไม่มีการเรียกฐานข้อมูลจริง)
    """
    fake_sources = [
        {"source_name": f"MOCK_DOC_{sub_criteria_id}_L{level}.pdf", "location": f"page_{10+statement_number}", "doc_id": f"DOC_{sub_criteria_id}"}
    ]
    # จำลองผลลัพธ์ที่ซับซ้อนขึ้นเพื่อทดสอบ Reranking Logic (หากมี)
    return {
        "top_evidences": [
            {"content": f"[MOCK EVIDENCE 1] Primary evidence for {sub_criteria_id} L{level} S{statement_number}.", "score": 0.9, "source": fake_sources[0]['source_name'], "doc_id": fake_sources[0]['doc_id'], "metadata": {"page_number": 10+statement_number}},
            {"content": f"[MOCK EVIDENCE 2] Secondary evidence for {sub_criteria_id} L{level} S{statement_number}.", "score": 0.7, "source": fake_sources[0]['source_name'], "doc_id": fake_sources[0]['doc_id'], "metadata": {"page_number": 11+statement_number}},
        ]
    }


# -------------------------------------------------------
# MOCK: Action Plan Generation
# -------------------------------------------------------
def create_structured_action_plan_MOCK(failed_statements_data: List[Dict[str, Any]], sub_id: str, target_level: int) -> List[Dict[str, Any]]:
    """
    Mock LLM Action Plan
    """
    actions = []
    
    # 🚨 FIX: ใช้ Action Plan ที่ง่ายขึ้นเพื่อบ่งชี้ถึง Gap 
    actions.append({
        "Statement_ID": "ALL_L1",
        "Failed_Level": target_level,
        "Recommendation": f"รวบรวมหลักฐานใหม่สำหรับ Level {target_level} (และ Level ที่มี Gap อื่นๆ: 1, 2, 3, 4, 5) และนำเข้า Vector Store และรัน AI Assessment ในโหมด **FULLSCOPE** อีกครั้งเพื่อยืนยันว่า Level ที่ต้องการผ่านเกณฑ์",
        "Target_Evidence_Type": "Rerunning Assessment & New Evidence",
        "Key_Metric": f"Overall Score ของ {sub_id} ต้องเพิ่มขึ้นและ Highest Full Level ต้องเป็น L{target_level}"
    })
    
    # Action Plan จะถูกส่งคืนเป็น List ของ Phase (เพื่อให้สอดคล้องกับ run_assessment.py)
    return [
        {
            "Phase": "2. AI Validation & Maintenance",
            "Goal": f"ยืนยันการ Level-Up และรักษาความต่อเนื่องของหลักฐานสำหรับ L{target_level}",
            "Actions": actions
        }
    ]


# -------------------------------------------------------
# MOCK: Summarize Context (ใช้ใน Evidence Summary)
# -------------------------------------------------------
def summarize_context_with_llm_MOCK(
    context: str, 
    sub_criteria_name: str, 
    level: int, 
    **kwargs # รับ Argument ที่ไม่คาดคิดทั้งหมด
) -> Dict[str, Any]:
    """
    Mock การสรุปหลักฐานในระดับสูงสุด (L5)
    """
    
    sub_id = kwargs.get('sub_id', 'N/A')
    
    return {
        "summary": f"[MOCK SUMMARY] สรุปหลักฐานของ {sub_criteria_name} ({sub_id}) สำหรับ Level {level}. เนื้อหาจำลอง ไม่พบหลักฐานที่เกี่ยวข้องใน Vector Store สำหรับเกณฑ์ {sub_id} Level {level}...",
        "suggestion_for_next_level": f"[MOCK SUGGESTION] ควรเก็บหลักฐานเพิ่มเติมเพื่อความต่อเนื่องของ L{level+1}"
    }


# -------------------------------------------------------
# MOCK: Set Control Mode
# -------------------------------------------------------
def set_mock_control_mode(mode: str = "default"):
    """
    สำหรับตั้งค่า behavior ของ mock (เผื่อในอนาคตมีหลาย pattern)
    """
    logger.info(f"[MOCK MODE] Using mock control mode = {mode}")
    random.seed(42)
    return True

 # ----------------------------------------------------
    ## 🌟 NEW FEATURE: Generate Action Plan (Mock Handler)
    # ----------------------------------------------------
def generate_action_plan_mock(
    sub_criteria_id: str,
    raw_llm_results: List[Dict[str, Any]] = None,
    final_subcriteria_results: List[Dict[str, Any]] = None,
) -> List[Dict]:
    """
    สร้าง Action Plan สำหรับ Sub-Criteria ที่กำหนด (ฟังก์ชัน Mock แบบไม่ใช้ self)
    ใช้จำลองการสร้าง Action Plan หลังประเมินผล LLM Mock เสร็จ
    """

    logger.info(f"[MOCK ACTION PLAN] Generating mock action plan for {sub_criteria_id}")

    # ✅ ตรวจสอบ input
    if not raw_llm_results:
        logger.warning("generate_action_plan_mock: raw_llm_results is empty or None")
        return []

    if not final_subcriteria_results:
        final_subcriteria_results = []

    # ✅ ดึง statement ที่ไม่ผ่าน
    failed_statements_data = [
        r for r in raw_llm_results
        if r.get("sub_criteria_id") == sub_criteria_id and not r.get("pass_status", False)
    ]

    # ✅ หา target_level จากผลลัพธ์สรุป
    sub_criteria_result = next(
        (r for r in final_subcriteria_results if r.get("sub_criteria_id") == sub_criteria_id),
        {}
    )
    target_level = sub_criteria_result.get("highest_full_level", 0) + 1

    logger.info(f"[MOCK ACTION PLAN] Found {len(failed_statements_data)} failed statements for {sub_criteria_id}, target_level={target_level}")

    # ✅ เรียกใช้ Mock Generator จริง
    try:
        return create_structured_action_plan_MOCK(
            failed_statements_data=failed_statements_data,
            sub_id=sub_criteria_id,
            target_level=target_level
        )
    except Exception as e:
        logger.exception(f"[MOCK ACTION PLAN] Error creating action plan for {sub_criteria_id}: {e}")
        return []
