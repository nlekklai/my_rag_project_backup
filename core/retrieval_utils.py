import logging
import random 
import json   
import time # <--- เพิ่ม import time ที่นี่
from typing import List, Dict, Any, Optional, Union

# ต้องมั่นใจว่า vectorstore และ rag_prompts ถูก import ได้
from core.vectorstore import VectorStoreManager, load_all_vectorstores 
from core.rag_prompts import SYSTEM_ASSESSMENT_PROMPT 

# Import LLM Instance Explicitly to avoid module name conflict
from models.llm import llm as llm_instance 
from langchain.schema import SystemMessage, HumanMessage 
from langchain.schema import Document 

logger = logging.getLogger(__name__)
# ปรับ level เป็น DEBUG ตามที่คุณต้องการ
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
            docs = retriever.invoke(stmt) 
            if not docs:
                logger.warning(f"⚠️ No results found for statement: {stmt[:50]}...")
            results[stmt] = docs
        except Exception as e:
            logger.error(f"Retrieval failed for statement: {stmt[:50]}... Error: {e}")
            results[stmt] = []
    return results


# 🚨 FIXED FUNCTION: retrieve_context_with_filter
def retrieve_context_with_filter(
    query: str, 
    retriever: Any, 
    # 🚨 CRITICAL FIX: ต้องเพิ่มชื่อ Argument นี้เพื่อให้ตรงกับการเรียกใน enabler_assessment.py
    metadata_filter: Optional[List[str]] = None, 
) -> Dict[str, Any]:
    """
    Retrieval function ที่เพิ่มความสามารถในการกรองเอกสาร (Document ID Filter)
    โดยใช้ retriever ที่โหลดมาแล้ว
    """
    if retriever is None:
        return {"top_evidences": []}
    
    # 1. กำหนดชื่อตัวแปรที่ใช้งานภายในฟังก์ชัน (เพื่อความชัดเจน)
    filter_document_ids = metadata_filter 

    # 2. Log การกรอง
    if filter_document_ids:
        logger.debug(f"RAG Filter Applied: {len(filter_document_ids)} documents for query: '{query[:30]}...'")

    try:
        # 3. เรียกใช้ Search (ใช้ .invoke() เพื่อความเข้ากันได้)
        docs: List[Document] = retriever.invoke(query) 
        
        # 4. กรองเอกสารด้วยมือ (Manual Filtering)
        # เนื่องจาก filter_document_ids (ชื่อไฟล์) ถูกส่งมาแล้ว จึงกรองตาม doc_id ใน metadata
        if filter_document_ids:
            logger.debug(f"Filter List Length: {len(filter_document_ids)}") 

            filtered_docs = []
            for doc in docs:
                doc_id_in_metadata = doc.metadata.get("doc_id") # ดึงค่า doc_id

                # โค้ดกรองหลัก: ถ้า doc_id อยู่ในรายการที่อนุญาต ให้นำไปใช้
                if doc_id_in_metadata in filter_document_ids:
                    filtered_docs.append(doc)
                # 💡 Debug Logs:
                # elif doc_id_in_metadata is None:
                #     logger.debug(f"Document missing 'doc_id' key. Source: {doc.metadata.get('source')}")
                # else:
                #     logger.debug(f"Doc ID mismatch: Metadata='{doc_id_in_metadata}' not in Filter.")
            
            docs = filtered_docs
            logger.debug(f"Found {len(docs)} documents after manual filtering.")
        
        # 5. จัดรูปแบบผลลัพธ์
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

MAX_LLM_RETRIES = 3 

def evaluate_with_llm(statement: str, context: str, standard: str) -> Dict[str, Any]:
    """
    ฟังก์ชันสำหรับเรียกใช้ LLM เพื่อประเมินความสอดคล้อง (เพิ่ม Logic Retry)
    """
    global _MOCK_CONTROL_FLAG, _MOCK_COUNTER
    
    # 1. MOCK CONTROL LOGIC (เดิม)
    if _MOCK_CONTROL_FLAG:
        # ... (โค้ด MOCK เดิม) ...
        # ...
        logger.debug(f"MOCK COUNT: {_MOCK_COUNTER} | SCORE: {score} | STMT: '{statement[:20]}...'")
        return {"score": score, "reason": reason_text}

    
    # 2. REAL LLM CALL LOGIC
    if llm_instance is None:
        logger.error("❌ LLM Instance is not initialized (likely failed to connect to Ollama).")
        score = random.choice([0, 1])
        reason = f"LLM Initialization Failed (Fallback to Random Score {score})"
        return {"score": score, "reason": reason}

    # 1. จัดรูปแบบ User Prompt (เดิม)
    # ... (user_prompt remains the same) ...
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
    
    # === NEW RETRY LOOP ===
    for attempt in range(MAX_LLM_RETRIES):
        try:
            # A. เรียกใช้ LLM
            response = llm_instance.invoke([
                SystemMessage(content=SYSTEM_ASSESSMENT_PROMPT),
                HumanMessage(content=user_prompt)
            ])
            
            llm_response_content = response.content if hasattr(response, 'content') else str(response)

            # B. Parse JSON string ที่ได้จาก LLM
            if llm_response_content.strip().startswith("```json"):
                llm_response_content = llm_response_content.strip().replace("```json", "").replace("```", "")
                
            llm_output = json.loads(llm_response_content.strip()) # 🚨 จุดที่เกิด Error
            
            score = int(llm_output.get("score", 0)) 
            reason = llm_output.get("reason", "No reason provided by LLM.")
            
            # ถ้าสำเร็จ: RETURN ทันที (ไม่จำเป็นต้องลองซ้ำ)
            return {"score": score, "reason": reason}

        except json.JSONDecodeError as e:
            # C. จัดการ JSON Error
            if attempt < MAX_LLM_RETRIES - 1:
                logger.warning(f"❌ JSON Parse Failed (Attempt {attempt + 1}/{MAX_LLM_RETRIES}). Retrying in 1s... Error: {e}")
                time.sleep(1) # หน่วงเวลาเล็กน้อยก่อนลองใหม่
                continue
            else:
                # ถ้าถึงรอบสุดท้ายแล้ว ให้ข้ามไปใช้ Fallback
                logger.error(f"❌ LLM Evaluation failed after {MAX_LLM_RETRIES} attempts. JSON format failure.")
                break # ออกจาก Loop
        
        except Exception as e:
            # D. จัดการ Error อื่นๆ (เช่น Connection Error)
            logger.error(f"❌ LLM Evaluation failed due to unexpected error. Error: {e}")
            break # ออกจาก Loop และใช้ Fallback ทันที

    # === FALLBACK LOGIC (ทำงานเมื่อ Loop จบด้วย Break) ===
    logger.error("❌ Using RANDOM SCORE as final fallback.")
    score = random.choice([0, 1])
    reason = f"LLM Call Failed (Fallback to Random Score {score}) after {MAX_LLM_RETRIES} attempts."
    return {"score": score, "reason": reason}