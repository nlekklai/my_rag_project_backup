import logging
import random 
import json   
from typing import List, Dict, Any, Optional, Union

# ต้องมั่นใจว่า vectorstore และ rag_prompts ถูก import ได้
from core.vectorstore import VectorStoreManager, load_all_vectorstores 
from core.rag_prompts import SYSTEM_ASSESSMENT_PROMPT 

# Import LLM Instance Explicitly to avoid module name conflict
from models.llm import llm as llm_instance 
from langchain.schema import SystemMessage, HumanMessage 
from langchain.schema import Document 

logger = logging.getLogger(__name__)
# 🚨 เปลี่ยนเป็น DEBUG เพื่อดู Log การกรองเอกสาร
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


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
        _MOCK_COUNTER = 0
        logger.info("🔑 CONTROLLED MOCK Mode ENABLED. Score will be 1 for first 5 statements, then 0.")
    else:
        logger.info("❌ CONTROLLED MOCK Mode DISABLED.")


# =================================================================
# === RETRIEVAL FUNCTIONS (INCLUDING THE NEW FILTER FUNCTION) ===
# =================================================================

def retrieve_statements(statements: List[str], doc_id: Optional[str] = None) -> Dict[str, List[Document]]:
    """
    Retrieve documents จาก vectorstore สำหรับ list ของ statements
    NOTE: ฟังก์ชันนี้อาจไม่ได้ถูกเรียกใช้โดยตรงใน Assessment Process
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
            # (ในโค้ดใหม่ เราควรใช้ .invoke() เพื่อหลีกเลี่ยง DeprecationWarning)
            docs = retriever.invoke(stmt) 
            if not docs:
                logger.warning(f"⚠️ No results found for statement: {stmt[:50]}...")
            results[stmt] = docs
        except Exception as e:
            logger.error(f"Retrieval failed for statement: {stmt[:50]}... Error: {e}")
            results[stmt] = []
    return results


# 🚨 NEW FUNCTION: retrieve_context_with_filter (ใช้ในการประเมินหลัก)
def retrieve_context_with_filter(query: str, retriever: Any, filter_document_ids: List[str]) -> Dict[str, Any]:
    """
    Retrieval function ที่เพิ่มความสามารถในการกรองเอกสาร (Document ID Filter)
    โดยใช้ retriever ที่โหลดมาแล้ว
    """
    if retriever is None:
        return {"top_evidences": []}

    # 1. สร้าง Filter Metadata (ส่วนนี้ใช้สำหรับ Log เท่านั้น ในการ Implement ปัจจุบัน)
    metadata_filter = None
    if filter_document_ids:
        metadata_filter = {
            "doc_id": {"$in": filter_document_ids}
        }
        logger.debug(f"RAG Filter Applied: {len(filter_document_ids)} documents for query: '{query[:30]}...'")

    try:
        # 2. เรียกใช้ Search (เปลี่ยนเป็น .invoke() ตามคำแนะนำของ LangChain)
        # LangChainDeprecationWarning เกิดขึ้นที่นี่
        docs: List[Document] = retriever.invoke(query) 
        
        # 3. กรองเอกสารด้วยมือ หาก VectorStore ไม่รองรับ Filter ใน .get_relevant_documents()
        filtered_docs = []
        if filter_document_ids:
            # 💡 [DEBUG FIX] เพิ่ม Log เพื่อตรวจสอบรายการ Filter
            logger.debug(f"Filter List: {filter_document_ids}") 

            for doc in docs:
                doc_id_in_metadata = doc.metadata.get("doc_id") # ดึงค่า doc_id
                
                # 💡 [DEBUG FIX] Log ค่าที่ไม่ตรง/หายไป
                if doc_id_in_metadata is None:
                    logger.debug(f"Document missing 'doc_id' key. Source: {doc.metadata.get('source')}")
                elif doc_id_in_metadata not in filter_document_ids:
                    logger.debug(f"Doc ID mismatch: Metadata='{doc_id_in_metadata}' not in Filter.")

                # โค้ดกรองหลัก
                if doc_id_in_metadata in filter_document_ids:
                    filtered_docs.append(doc)
            
            docs = filtered_docs
            logger.debug(f"Found {len(docs)} documents after manual filtering.")
        
        top_evidences = []
        for d in docs:
            meta = d.metadata
            top_evidences.append({
                "doc_id": meta.get("doc_id"),
                "source": meta.get("source"),
                "content": d.page_content.strip()
            })
            
        return {"top_evidences": top_evidences}
        
    except Exception as e:
        logger.error(f"Error during RAG retrieval with filter: {e}")
        return {"top_evidences": []}


# =================================================================
# === EVALUATION FUNCTION (MOCK & REAL LLM) ===
# =================================================================

def evaluate_with_llm(statement: str, context: str, standard: str) -> Dict[str, Any]:
    """
    ฟังก์ชันสำหรับเรียกใช้ LLM เพื่อประเมินความสอดคล้อง (โค้ดเดิม)
    """
    global _MOCK_CONTROL_FLAG, _MOCK_COUNTER
    
    # 1. MOCK CONTROL LOGIC
    if _MOCK_CONTROL_FLAG:
        _MOCK_COUNTER += 1
        
        if _MOCK_COUNTER <= 3: 
            score = 1
            reason_text = f"MOCK: FORCED PASS (L1, Statement {_MOCK_COUNTER})"
        elif _MOCK_COUNTER in [4, 5]:
            score = 1
            reason_text = f"MOCK: FORCED PASS (L2, Statement {_MOCK_COUNTER})"
        else:
            score = 0
            reason_text = f"MOCK: FORCED FAIL (L2+, Statement {_MOCK_COUNTER})"

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