# -------------------- core/retrieval_utils.py (FINAL FIXED VERSION) --------------------
import logging
import random 
import json   
from typing import List, Dict, Any, Optional, Union
# from langchain.schema import Document # NOTE: ถูกยกไปรวมในด้านบนของไฟล์แล้ว

# NOTE: สมมติว่าไฟล์นี้สามารถเข้าถึง VectorStoreManager, load_all_vectorstores และ RAG Prompts ได้
from core.vectorstore import VectorStoreManager, load_all_vectorstores 
from core.rag_prompts import SYSTEM_ASSESSMENT_PROMPT 

# Import LLM Instance Explicitly to avoid module name conflict
from models.llm import llm as llm_instance 
from langchain.schema import SystemMessage, HumanMessage # สำหรับ LLM Call
from langchain.schema import Document # เพิ่มการ Import Document เพื่อให้โค้ดส่วน Retrieval ทำงานได้

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


# =================================================================
# === MOCKING LOGIC AND GLOBAL FLAGS ===
# =================================================================

_MOCK_CONTROL_FLAG = False
_MOCK_COUNTER = 0

def set_mock_control_mode(enable: bool):
    """ฟังก์ชันสำหรับเปิด/ปิดโหมดควบคุมคะแนน (CONTROLLED MOCK)"""
    global _MOCK_CONTROL_FLAG, _MOCK_COUNTER
    _MOCK_CONTROL_FLAG = enable
    if enable:
        # รีเซ็ตตัวนับเสมอเมื่อเปิดโหมดควบคุม
        _MOCK_COUNTER = 0
        logger.info("🔑 CONTROLLED MOCK Mode ENABLED. Score will be 1 for first 5 statements, then 0.")
    else:
        logger.info("❌ CONTROLLED MOCK Mode DISABLED.")


# =================================================================
# === RETRIEVAL FUNCTIONS ===
# NOTE: โค้ดส่วนนี้ไม่ได้ถูกแก้ไขหลักๆ แต่ถูกรวมเพื่อให้สมบูรณ์
# =================================================================

def retrieve_statements(statements: List[str], doc_id: Optional[str] = None) -> Dict[str, List[Document]]:
    """
    Retrieve documents จาก vectorstore สำหรับ list ของ statements
    """
    vs_manager = VectorStoreManager()
    retriever = vs_manager.get_retriever(k=5)
    if retriever is None:
        logger.error("Retriever not initialized.")
        return {stmt: [] for stmt in statements}

    results: Dict[str, List[Document]] = {}
    for stmt in statements:
        try:
            # NOTE: ใช้ retriever.get_relevant_documents
            docs = retriever.get_relevant_documents(stmt)
            if not docs:
                logger.warning(f"⚠️ No results found for statement: {stmt[:50]}...")
            results[stmt] = docs
        except Exception as e:
            logger.error(f"Retrieval failed for statement: {stmt[:50]}... Error: {e}")
            results[stmt] = []
    return results


def retrieve_context(statement: str,
                     doc_ids: Optional[List[str]] = None,
                     doc_type: str = "document",
                     top_k: int = 10,
                     final_k: int = 3) -> Dict[str, Any]:
    """
    🔍 Retrieve top evidences for a given KM statement.
    """
    try:
        retriever = load_all_vectorstores(
            doc_ids=doc_ids,
            top_k=top_k,
            final_k=final_k,
            doc_type=doc_type
        )
        # NOTE: ตรวจสอบการเรียก get_relevant_documents
        docs: List[Document] = retriever.get_relevant_documents(statement) if retriever else []

        results = []
        for d in docs:
            meta = d.metadata
            # NOTE: ควรตรวจสอบว่า metadata มีค่าหรือไม่ก่อนเรียก .get()
            results.append({
                "doc_id": meta.get("doc_id"),
                "doc_type": meta.get("doc_type"),
                "chunk_index": meta.get("chunk_index"),
                "score": meta.get("relevance_score", None),
                "source": meta.get("source"),
                "content": d.page_content.strip()
            })

        logger.debug(f"✅ Found {len(results)} context items for statement: '{statement[:60]}...'")
        return {"statement": statement, "context_count": len(results), "top_evidences": results}

    except Exception as e:
        logger.error(f"⚠️ Retrieval failed for statement='{statement[:60]}...': {e}")
        return {"statement": statement, "context_count": 0, "top_evidences": [], "error": str(e)}


def batch_retrieve_from_checklist(checklist_json_path: str, doc_type: str = "km_document") -> List[Dict[str, Any]]:
    """
    Loop ผ่าน checklist JSON แล้ว retrieve context ให้ครบทุก statement
    """
    # NOTE: จำเป็นต้อง Import json ภายในฟังก์ชันนี้หากรันแยก แต่เนื่องจาก import ไว้ module level แล้ว จึงไม่เป็นปัญหา
    try:
        with open(checklist_json_path, "r", encoding="utf-8") as f:
            checklist = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load checklist JSON: {e}")
        return []

    results = []
    for enabler in checklist:
        for level in range(1, 6):
            level_key = f"Level_{level}_Statements"
            statements: List[str] = enabler.get(level_key, [])
            
            for stmt in statements:
                result = retrieve_context(stmt, doc_type=doc_type)
                results.append({
                    "enabler_id": enabler.get("Enabler_ID"),
                    "sub_criteria_id": enabler.get("Sub_Criteria_ID"),
                    "level": level,
                    **result
                })
    return results


# =================================================================
# === EVALUATION FUNCTION (MOCK & REAL LLM) ===
# =================================================================

def evaluate_with_llm(statement: str, context: str, standard: str) -> Dict[str, Any]:
    """
    ฟังก์ชันสำหรับเรียกใช้ LLM เพื่อประเมินความสอดคล้อง
    """
    global _MOCK_CONTROL_FLAG, _MOCK_COUNTER
    
    # 1. MOCK CONTROL LOGIC
    if _MOCK_CONTROL_FLAG:
        _MOCK_COUNTER += 1
        
        # Logic: Pass L1, Partially Pass L2 (3 Pass, 2 Pass, 1 Fail)
        if _MOCK_COUNTER <= 3: # L1 Statements
            score = 1
            reason_text = f"MOCK: FORCED PASS (L1)"
        elif _MOCK_COUNTER in [4, 5]: # L2 Statements (2/3 Pass)
            score = 1
            reason_text = f"MOCK: FORCED PASS (L2)"
        else: # L2 S3 and all L3-L5 (Counter > 5)
            score = 0
            reason_text = f"MOCK: FORCED FAIL (L2+)"

        logger.debug(f"MOCK COUNT: {_MOCK_COUNTER} | SCORE: {score} | STMT: '{statement[:20]}...'")
        return {"score": score, "reason": reason_text}

    
    # 2. REAL LLM CALL LOGIC

    if llm_instance is None:
        logger.error("❌ LLM Instance is not initialized (likely failed to connect to Ollama).")
        score = random.choice([0, 1])
        reason = f"LLM Initialization Failed (Fallback to Random Score {score})"
        return {"score": score, "reason": reason}

    # 1. จัดรูปแบบ User Prompt (สำหรับ LLM)
    user_prompt = f"""
    --- Statement ที่ต้องการประเมิน (หลักฐานที่ควรมี) ---
    {statement}

    --- เกณฑ์ (Standard/Rubric) ---
    {standard}

    --- หลักฐานที่พบในเอกสารจริง (Context จาก Semantic Search) ---
    {context if context else "ไม่พบหลักฐานในเอกสารที่เกี่ยวข้อง"}

    --- คำสั่ง ---
    โปรดประเมินโดยใช้บทบาท Se-AM Consultant ว่าหลักฐานที่พบ (Context) สอดคล้อง 
    กับ Statement และเกณฑ์ที่กำหนดหรือไม่
    
    โปรดตอบในรูปแบบ JSON ที่มี key: 'score' (0 หรือ 1) และ 'reason' เท่านั้น!
    ตัวอย่าง: {{"score": 1, "reason": "หลักฐาน X ใน Context ยืนยัน Statement Y..."}}
    """
    
    try:
        response = llm_instance.invoke([
            SystemMessage(content=SYSTEM_ASSESSMENT_PROMPT),
            HumanMessage(content=user_prompt)
        ])
        
        llm_response_content = response.content if hasattr(response, 'content') else str(response)

        # 2. Parse JSON string ที่ได้จาก LLM (Robust Parsing)
        if llm_response_content.strip().startswith("```json"):
            llm_response_content = llm_response_content.strip().replace("```json", "").replace("```", "")
            
        llm_output = json.loads(llm_response_content.strip())
        
        score = int(llm_output.get("score", 0)) 
        reason = llm_output.get("reason", "No reason provided by LLM.")
        
        return {"score": score, "reason": reason}

    except Exception as e:
        logger.error(f"❌ LLM Evaluation failed. Using RANDOM SCORE as fallback. Error: {e}")
        score = random.choice([0, 1])
        reason = f"LLM Call Failed (Fallback to Random Score {score}): {str(e)}"
        return {"score": score, "reason": reason}