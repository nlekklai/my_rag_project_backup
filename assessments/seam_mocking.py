"""
Mocking Assessment Utilities
ใช้สำหรับทดสอบระบบ KM/Enabler Assessment โดยไม่ต้องเรียก LLM จริง
(Revised for SEAM PDCA Engine)
"""

import logging
import random
from typing import Dict, Any, List, Optional, Type, TypeVar
from pydantic import BaseModel 

# 💡 เพิ่มการ Import LcDocument และ Config เพื่อให้ Mock Object ที่คืนค่าตรงตามที่ Engine ต้องการ
try:
    from langchain_core.documents import Document as LcDocument 
    from config.global_vars import FINAL_K_RERANKED 
except ImportError:
    # Fallback to a simple dictionary and default value if imports fail
    LcDocument = dict 
    FINAL_K_RERANKED = 5 

# นำเข้า PDCA_PHASE_MAP จาก seam_prompts เพื่อใช้ใน Mock Reason/Action Plan
try:
    from core.seam_prompts import PDCA_PHASE_MAP
except ImportError:
    PDCA_PHASE_MAP = {1: "Plan", 2: "Plan + Do", 3: "Plan + Do + Check", 4: "PDCA Cycle", 5: "Advanced"}

logger = logging.getLogger(__name__)

# Mock TypeVars for consistency with llm_data_utils
T = TypeVar("T", bound=BaseModel)


# -------------------------------------------------------
# MOCK: Controlled LLM Evaluation (Deterministic)
# Mocks: llm_data_utils.evaluate_with_llm
# -------------------------------------------------------
def evaluate_with_llm_CONTROLLED_MOCK(
    context: str, 
    sub_criteria_name: str, 
    level: int, 
    statement_text: str, 
    sub_id: str, 
    **kwargs
) -> Dict[str, Any]:
    """
    Mock การประเมินผลลัพธ์แบบควบคุมลำดับ (Deterministic)
    คืนค่าตาม Schema ของ StatementAssessment (score, reason, is_passed)
    """
    
    # Extract PDCA Phase (เพื่อการ Debug/Log)
    pdca_phase = kwargs.get('pdca_phase', PDCA_PHASE_MAP.get(level, f"L{level} Concept"))
    sub_id_clean = str(sub_id).strip()
    logger.info(f"[MOCK LLM] Evaluating sub={sub_id_clean}, level={level}, PDCA={pdca_phase}")

    # --- Controlled Logic ---
    score = 0
    if sub_id_clean == "1.2":
        # L1, L2, L3 ผ่าน; L4, L5 ไม่ผ่าน (Highest Full Level จะจบที่ L3)
        if level in [1, 2, 3]:
            score = 1
        else:
            score = 0
    elif sub_id_clean == "3.1":
        # L1, L2 ผ่าน; L3, L4, L5 ไม่ผ่าน (Highest Full Level จะจบที่ L2)
        if level in [1, 2]:
            score = 1
        else:
            score = 0
    else:
        # Default Logic: L1 ผ่าน, ที่เหลือไม่ผ่าน (Highest Full Level จะจบที่ L1)
        if level == 1:
            score = 1
        else:
            score = 0

    is_passed = score == 1
    reason = f"[MOCK] Statement passed the {pdca_phase} check (Controlled Mock). หลักฐานจำลองชี้ว่าองค์กรมีหลักฐานครบถ้วนถึง L{level} แล้ว. Result: {'PASS' if is_passed else 'FAIL'}"

    # คืนค่าตามที่ llm_data_utils.py คาดหวัง
    return {
        "score": score,
        "reason": reason,
        "is_passed": is_passed, 
    }


# -------------------------------------------------------
# MOCK: Retrieval
# Mocks: llm_data_utils.retrieve_context_with_filter
# -------------------------------------------------------
# 🎯 FIX: เพิ่ม Argument 'vsm_manager' เพื่อให้ Signature ตรงกับ llm_data_utils.py
def retrieve_context_with_filter_MOCK(
    vsm_manager: Optional[Any], # 🟢 Argument ที่เพิ่มเข้ามาเพื่อรับ VSM Instance
    query: str,
    collection_name: str,
    doc_uuid_filter: Optional[List[str]] = None,
    disable_semantic_filter: bool = False,
    top_k: int = FINAL_K_RERANKED 
) -> Dict[str, Any]:
    """
    Mock retrieval context จาก Vectorstore (ไม่มีการเรียกฐานข้อมูลจริง)
    คืนค่า Dict ในรูปแบบที่ LLM Engine คาดหวัง: {"top_evidences": [...], "aggregated_context": "..." }
    """
    
    # Mock does not use vsm_manager, but must accept it.
    sub_id = collection_name.split('_')[-1] 
    logger.info(f"[MOCK RAG] Retrieving {top_k} chunks for query on {sub_id}...")

    top_evidences = []
    aggregated_parts = []
    
    for i in range(top_k):
        metadata = {
            "stable_doc_uuid": f"mock-stable-uuid-{sub_id}-{i+1}",
            "file_name": f"MOCK_DOC_{sub_id}_Chunk_{i+1}.pdf", 
            "location": f"/path/to/docs/{sub_id}/chunk_{i+1}", 
            "chunk_uuid": f"mock-uuid-{sub_id}-{i+1}"
        }
        page_content = f"[MOCK CHUNK {i+1}] Relevant evidence for topic {sub_id} (Query: {query[:30]}...)"
        
        # โครงสร้างการคืนค่า Evidence ต้องเป็น Dict ที่มี content และ metadata
        top_evidences.append({"content": page_content, "metadata": metadata})
        aggregated_parts.append(page_content)

    aggregated_context = "\n---\n".join(aggregated_parts)

    return {
        "top_evidences": top_evidences,
        "aggregated_context": aggregated_context
    }


# -------------------------------------------------------
# MOCK: Action Plan Generation
# Mocks: llm_data_utils.create_structured_action_plan
# -------------------------------------------------------
def create_structured_action_plan_MOCK(
    failed_statements_data: List[Dict[str, Any]], 
    sub_id: str, 
    enabler: str, 
    target_level: int, 
    max_retries: int = 2 
) -> List[Dict[str, Any]]:
    """
    Mock LLM Action Plan. คืนค่าตาม ActionPlanActions Schema (List ของ Phase)
    """
    logger.info(f"[MOCK ACTION PLAN] Generating mock plan for {sub_id} (Target L{target_level})")

    first_fail_reason = "ไม่พบสาเหตุความล้มเหลว"
    if failed_statements_data:
        first_fail_reason = failed_statements_data[0].get('reason', 'Missing reason').strip()
    
    # ใช้ PDCA Phase ที่เป็นเป้าหมาย
    target_phase = PDCA_PHASE_MAP.get(target_level, f"Level {target_level} Requirements")
    
    # คืนค่าตาม Action Plan Schema
    return [
        {
            "Phase": f"1. Gap Closure & Planning ({target_phase})",
            "Goal": f"จัดทำหลักฐานใหม่เพื่อปิด Gap ที่ L{target_level} ในเกณฑ์ {sub_id}",
            "Actions": [
                {
                    "Statement_ID": sub_id,
                    "Recommendation": f"ทบทวน Gap: {first_fail_reason[:50]}... และจัดทำเอกสารที่แสดงถึงการปฏิบัติตามขั้นตอน {target_phase} อย่างครบถ้วน",
                    "Responsible": f"{enabler} Enabler Lead",
                    "Key_Metric": f"Evidence Quality Score L{target_level}",
                    "Tools_Templates": "Gap Analysis Template, Action Plan Form",
                    "Verification_Outcome": "หลักฐานใหม่ถูกรวบรวมและนำเข้า Vector Store"
                }
            ]
        }
    ]


# -------------------------------------------------------
# MOCK: Summarize Context
# Mocks: llm_data_utils.summarize_context_with_llm
# -------------------------------------------------------
def summarize_context_with_llm_MOCK(
    context: str, 
    sub_criteria_name: str, 
    level: int, 
    sub_id: str, 
    schema: Optional[Type[T]] = None
) -> Dict[str, Any]:
    """
    Mock การสรุปหลักฐาน (Evidence Summary)
    """
    logger.info(f"[MOCK SUMM] Summarizing evidence for {sub_id} L{level}")
    
    mock_context_len = context.count("[MOCK CHUNK") 
    
    return {
        "summary": f"[MOCK SUMMARY] พบหลักฐานจำลอง {mock_context_len} ชิ้นสำหรับ {sub_criteria_name} ({sub_id}) L{level} เนื้อหาครอบคลุมถึงขั้นตอนการวางแผนและการดำเนินงานในระดับพื้นฐาน",
        "suggestion_for_next_level": f"[MOCK SUGGESTION] ควรเน้นการเก็บหลักฐานที่แสดงถึง **การตรวจสอบและปรับปรุง (Check & Act Phase)** เพื่อเตรียมพร้อมสำหรับ L{level+1}"
    }


# -------------------------------------------------------
# MOCK: Set Control Mode (Used by SEAMPDCAEngine)
# -------------------------------------------------------
def set_mock_control_mode(enable: bool):
    """
    Mock setting the global mock control flag.
    Note: The real mock flag is controlled in core/llm_data_utils.py
    """
    logger.info(f"[MOCK MODE] Setting mock control mode to {enable}")
    if enable:
        random.seed(42) # Ensure deterministic "random" for other components (if any)