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
    # 💡 NOTE: level และ statement_number ที่นี่จะเป็นค่า Default หากไม่ได้ส่งผ่าน kwargs
    level: int = 1,
    sub_criteria_id: str = "UNKNOWN",
    statement_number: int = 1,
    **kwargs
) -> Dict[str, Any]:
    """
    Mock การประเมินผลลัพธ์แบบควบคุมลำดับ (ใช้แทน LLM จริง)

    - L1: ผ่านทั้งหมด (100%)
    - L2: ผ่านเฉพาะ Statement ลำดับคี่ (2/3)
    - L3: ผ่านทั้งหมด (100%)
    - L4, L5: ไม่ผ่านทั้งหมด (0%)
    """
    
    # 🚨 FIX 1: ดึงค่า level, sub_criteria_id, statement_number จาก kwargs 
    # ซึ่งเป็นรูปแบบที่ EnablerAssessment.run_assessment ส่งเข้ามา
    level = kwargs.get("level", level)
    sub_criteria_id = kwargs.get("sub_criteria_id", sub_criteria_id)
    statement_number = kwargs.get("statement_number", statement_number)

    # 🎯 ตรรกะการให้คะแนนแบบควบคุม
    if level == 1:
        score = 1 # L1: ผ่านทั้งหมด
    elif level == 2:
        score = 1 if statement_number % 2 == 1 else 0 # L2: ผ่านเฉพาะ Statement ลำดับคี่ (S1, S3)
    elif level == 3:
        score = 1 # L3: ผ่านทั้งหมด
    elif level in [4, 5]:
        score = 0 # L4, L5: ไม่ผ่านทั้งหมด
    else:
        score = 0 

    is_passed = score == 1
    
    # ⭐ Mock Context และ Sources เพื่อให้ EnablerAssessment.run_assessment บันทึกหลักฐาน
    mock_context_snippet = f"[MOCK CONTEXT SNIPPET] Evidence found for {sub_criteria_id} L{level} S{statement_number}." if is_passed else ""
    mock_sources = [
        {"source_name": f"mock_doc_L{level}_S{statement_number}.pdf", "location": f"page_{10+statement_number}", "doc_id": f"DOC_{sub_criteria_id}"}
    ] if is_passed else []


    result = {
        "sub_criteria_id": sub_criteria_id,
        "level": level,
        "statement_number": statement_number,
        "statement": statement,
        
        # 3. ใช้ Mock Context/Sources ที่มีค่า
        "context_retrieved_snippet": mock_context_snippet, 
        "retrieved_sources_list": mock_sources,
        
        # 🚨 FIX 2: score ต้องถูกกำหนดตามตรรกะ Mock (1 หรือ 0)
        "llm_score": score, 
        
        # 🚨 NEW: ใส่ 'score' ด้วยเพื่อความเข้ากันได้กับ Real LLM Eval Function
        "score": score, 
        
        "reason": f"MOCK reason for L{level} S{statement_number} → {'PASS' if is_passed else 'FAIL'} (Controlled Mock)",
        
        # ⭐ กำหนด pass_status อย่างชัดเจน
        "pass_status": is_passed,
        "status_th": "ผ่าน" if is_passed else "ไม่ผ่าน",
        "llm_result": {"is_passed": is_passed, "score": float(score)}
    }

    return result


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
def generate_action_plan_MOCK(failed_statements_data: List[Dict[str, Any]], sub_id: str, target_level: int) -> List[Dict[str, Any]]:
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