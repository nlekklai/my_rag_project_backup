"""
llm_data_utils.py
Robust LLM + RAG utilities for SEAM assessment.
Responsibilities:
- Retrieval wrapper: retrieve_context_with_filter & retrieve_context_by_doc_ids
- Robust JSON extraction & normalization (_robust_extract_json, _normalize_keys)
- LLM invocation wrappers with retries (_fetch_llm_response)
- evaluate_with_llm: produce {score, reason, is_passed}
- summarize_context_with_llm: produce evidence summary
- create_structured_action_plan: generate action plan JSON list
- Mock control helper: set_mock_control_mode
"""
import logging, time, json, json5, random, hashlib, regex as re
from typing import List, Dict, Any, Optional, TypeVar, Final 
from pydantic import BaseModel, ConfigDict, Field, RootModel # RootModel ถูก Import เพื่อความถูกต้อง

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ------------------------
# Imports (project-specific)
# ------------------------
try:
    from core.seam_prompts import (
        SYSTEM_ASSESSMENT_PROMPT, USER_ASSESSMENT_PROMPT,
        SYSTEM_ACTION_PLAN_PROMPT, ACTION_PLAN_PROMPT,
        SYSTEM_EVIDENCE_DESCRIPTION_PROMPT, EVIDENCE_DESCRIPTION_PROMPT,
        SYSTEM_LOW_LEVEL_PROMPT, USER_LOW_LEVEL_PROMPT 
    )
    # NOTE: Assuming ChromaRetriever is defined or imported correctly
    from core.vectorstore import VectorStoreManager, get_global_reranker, _get_collection_name, ChromaRetriever
    from core.assessment_schema import StatementAssessment, EvidenceSummary
    from core.action_plan_schema import ActionPlanActions
    from config.global_vars import DEFAULT_ENABLER, INITIAL_TOP_K, FINAL_K_RERANKED
    from langchain_core.documents import Document as LcDocument
except Exception as e:
    logger.error(f"Missing dependency: {e}")
    # Define necessary placeholders for the code to run if imports fail
    class VectorStoreManager: pass
    # Mock Reranker needs to handle compress_documents (with query, documents, top_n)
    class MockReranker:
         def __init__(self, k): self.k = k
         def compress_documents(self, documents: List[Any], query: str, top_n: int) -> List[Any]:
             return documents[:top_n]
    def get_global_reranker(k): 
        # Return a mock object that can be checked by 'hasattr(reranker, 'compress_documents')'
        return type('MockRerankerWrapper', (), {'compress_documents': MockReranker(k).compress_documents, 'base_reranker': MockReranker(k)})()

    def _get_collection_name(doc_type, enabler): return f"{doc_type}_{enabler}"
    class ChromaRetriever: pass
    class StatementAssessment(BaseModel): score: int; reason: str
    class EvidenceSummary(BaseModel): summary: str; suggestion_for_next_level: str
    
    # 🟢 FIX: แก้ไข Pydantic V2 Syntax ใน Placeholder (เปลี่ยน __root__ เป็น BaseModel ธรรมดา)
    class ActionPlanActions(BaseModel): 
        Phase: str = "Mock Phase"
        Goal: str = "Mock Goal"
        Actions: List[Dict[str,Any]] = []
        
    class LcDocument: 
        def __init__(self, page_content, metadata): self.page_content=page_content; self.metadata=metadata
    # DEFAULT_ENABLER = "KM"
    # INITIAL_TOP_K = 10
    # FINAL_K_RERANKED = 3
    # Define mock prompts to prevent crash if real ones are missing
    SYSTEM_ASSESSMENT_PROMPT = "Assess the statement based on the provided context."
    USER_ASSESSMENT_PROMPT = "Context: {context}\nStatement: {statement_text}"
    SYSTEM_ACTION_PLAN_PROMPT = "Generate an action plan."
    ACTION_PLAN_PROMPT = "Failed statements: {failed_statements_list}"
    SYSTEM_EVIDENCE_DESCRIPTION_PROMPT = "Summarize evidence."
    EVIDENCE_DESCRIPTION_PROMPT = "Context: {context}"
    SYSTEM_LOW_LEVEL_PROMPT = "Assess L1/L2 simply."
    USER_LOW_LEVEL_PROMPT = "Context: {context}\nL1/L2 Statement: {statement_text}"


try:
    # Use a mock LLM instance if the real one isn't available
    from models.llm import llm as llm_instance
except Exception:
    logger.warning("Using Mock LLM Instance.")
    class MockLLM:
        def invoke(self, messages, config):
            global _MOCK_COUNTER
            _MOCK_COUNTER += 1
            # Simulate a pass/fail pattern for controlled mock
            is_pass = (_MOCK_COUNTER % 3 != 0) if _MOCK_FLAG else True 
            score = 1 if is_pass else 0
            reason = f"Mock assessment: {'Passed' if is_pass else 'Failed'} (Count: {_MOCK_COUNTER})"
            
            # Simulate JSON output based on the prompt's intent
            if "JSON ARRAY" in messages[0]['content']: # Action Plan
                return json.dumps([{"Phase":f"Mock Phase {_MOCK_COUNTER}","Goal":reason}])
            if "score" in messages[0]['content']: # Assessment
                return json.dumps({"score": score, "reason": reason, "is_passed": is_pass})
            
            return f"Mock Response {_MOCK_COUNTER}"
    llm_instance = MockLLM()


# ------------------------
# Constants for Phase 2 Optimization
# ------------------------
LOW_LEVEL_K: Final[int] = 3 

# ------------------------
# Mock control
# ------------------------
_MOCK_FLAG = False
_MOCK_COUNTER = 0
_MAX_LLM_RETRIES = 3

def set_mock_control_mode(enable: bool):
    global _MOCK_FLAG, _MOCK_COUNTER
    _MOCK_FLAG = bool(enable)
    _MOCK_COUNTER = 0
    logger.info(f"Mock control mode: {_MOCK_FLAG}")

# ------------------------
# ID normalization
# ------------------------
def _hash_stable_id_to_64_char(stable_id: str) -> str:
    return hashlib.sha256(stable_id.lower().encode('utf-8')).hexdigest()

def normalize_stable_ids(ids: List[str]) -> List[str]:
    return [i.lower() if len(i)==64 else _hash_stable_id_to_64_char(i) for i in ids]

# ------------------------
# Retrieval
# ------------------------
def retrieve_context_by_doc_ids(
    doc_uuids: List[str], 
    doc_type: str, 
    enabler: Optional[str] = None, 
    # 🟢 เพิ่ม Argument นี้
    vectorstore_manager: Optional['VectorStoreManager'] = None 
) -> Dict[str, Any]:
    
    # 📌 การตรวจสอบและเตรียม manager
    if not doc_uuids:
        return {"top_evidences": []}
    
    # 🟢 ใช้ VSM ที่ส่งเข้ามา ถ้ามี (Priority) หรือสร้างใหม่/ดึง Instance
    manager = vectorstore_manager if vectorstore_manager is not None else VectorStoreManager()
    
    if manager is None:
        logger.error("VectorStoreManager is None.")
        return {"top_evidences": []}
        
    try:
        # manager = VectorStoreManager() # ❌ ลบหรือคอมเมนต์บรรทัดนี้ออก
        
        normalized_uuids = normalize_stable_ids(doc_uuids)
        
        # Note: manager.get_documents_by_id must support list of normalized IDs
        docs: List[LcDocument] = manager.get_documents_by_id(normalized_uuids, doc_type, enabler)
        
        # ... (ส่วนที่เหลือของโค้ดเหมือนเดิม) ...
        top_evidences = [{
            "doc_id": d.metadata.get("stable_doc_uuid"),
            "doc_type": d.metadata.get("doc_type"),
            "chunk_uuid": d.metadata.get("chunk_uuid"),
            "source": d.metadata.get("source") or d.metadata.get("doc_source"),
            "content": d.page_content.strip(),
            "chunk_index": d.metadata.get("chunk_index")
        } for d in docs]
        
        return {"top_evidences": top_evidences}
        
    except Exception as e:
        logger.error(f"retrieve_context_by_doc_ids error: {e}")
        return {"top_evidences": []}


import time # 🟢 ต้องมี import time สำหรับ start_time และ duration

def retrieve_context_with_filter(query: str, doc_type: str, enabler: Optional[str]=None,
                                 vectorstore_manager: Optional['VectorStoreManager']=None,
                                 top_k: int=FINAL_K_RERANKED, initial_k: int=INITIAL_TOP_K,
                                 # 📌 NEW ARGUMENT: Persistent Mapping UUIDs
                                 mapped_uuids: Optional[List[str]]=None,
                                 stable_doc_ids: Optional[List[str]]=None, 
                                 # 🟢 FIX: เพิ่ม priority_docs_input เข้ามาในลายเซ็นต์หลัก
                                 priority_docs_input: Optional[List[Any]] = None, 
                                 sub_id: Optional[str]=None, level: Optional[int]=None) -> Dict[str, Any]:

    """
    Retrieves and reranks relevant context from the specified VectorStore collection, 
    GUARANTEEING the inclusion of pre-mapped priority documents in the final set.
    """
    start_time = time.time()
    try:
        # 1-3. Setup & K Setting (โค้ดเดิม)
        manager = vectorstore_manager
        if manager is None: raise ValueError("VectorStoreManager is not initialized.")
        collection_name = _get_collection_name(doc_type, enabler).lower() 
        retriever = manager.get_retriever(collection_name)
        if retriever is None: raise ValueError(f"Retriever initialization failed for {collection_name}.")
        
        logger.critical(f"🧭 DEBUG: Successfully retrieved Core Retriever. Starting query...")
        
        if hasattr(retriever, 'k'): retriever.k = initial_k
        if hasattr(retriever, 'search_kwargs') and isinstance(retriever.search_kwargs, dict):
            retriever.search_kwargs['k'] = initial_k

        # ----------------------------------------------------
        # 4. 📌 Persistent Mapping (Priority Context Retrieval)
        # ----------------------------------------------------
        # 🟢 FIX 1: ประกาศตัวแปรอย่างชัดเจน
        guaranteed_priority_chunks: List[Any] = [] 
        
        if priority_docs_input:
            logger.critical(f"🧭 DEBUG: Using {len(priority_docs_input)} pre-calculated priority chunks (Limited Chunks Mode).")
            guaranteed_priority_chunks = priority_docs_input
        
        elif mapped_uuids and manager and collection_name: 
            # ⚠️ Fallback Logic (โค้ดเดิม - ใช้ mapped_uuids ในการดึงเอกสารเต็ม)
            normalized_mapped_uuids = normalize_stable_ids(mapped_uuids) 
            logger.critical(f"🧭 DEBUG: [CHECK] Input mapped_uuids count: {len(normalized_mapped_uuids)}")
            logger.critical(f"🧭 DEBUG: Retrieving {len(normalized_mapped_uuids)} priority documents by DOCUMENT ID...") 
            try:
                if hasattr(manager, 'get_documents_by_id'):
                    retrieved_mapped_docs = manager.get_documents_by_id(
                        stable_doc_ids=normalized_mapped_uuids, 
                        doc_type=doc_type, 
                        enabler=enabler
                    )
                    # 🟢 FIX 2: กำหนดค่าเข้าตัวแปรที่ถูกต้อง
                    guaranteed_priority_chunks = retrieved_mapped_docs 
                else:
                    logger.error("❌ CRITICAL: VSM lacks 'get_documents_by_id' method.")

                logger.critical(f"🧭 DEBUG: [CHECK] Retrieved priority_docs count: {len(guaranteed_priority_chunks)}")
                logger.critical(f"🧭 DEBUG: Retrieved {len(guaranteed_priority_chunks)} documents from persistent mapping (Actual Mapped Docs).")
            except Exception as e:
                logger.error(f"Priority retrieval by UUID failed: {e}")
        
        # 🟢 FIX 3: ใช้ตัวแปรที่ถูกต้องในการนับ
        correct_mapped_count = len(guaranteed_priority_chunks) 
        
        logger.critical(f"🧭 DEBUG: [CHECK FINAL] priority_docs size BEFORE RAG Search: {correct_mapped_count}") 

        # ----------------------------------------------------
        # 5. Invoke Retrieval (Standard RAG Search) (โค้ดเดิม)
        # ----------------------------------------------------
        search_kwargs = {"k": initial_k} 
        if stable_doc_ids:
            normalized_uuids = normalize_stable_ids(stable_doc_ids) 
            search_kwargs["filter"] = {"stable_doc_uuid": {"$in": normalized_uuids}}
            logger.critical(f"🧭 DEBUG: RAG Filter by Stable Doc IDs activated ({len(normalized_uuids)} IDs).")

        retrieved_docs: List[Any] = []
        if callable(getattr(retriever, 'get_relevant_documents', None)):
            retrieved_docs = retriever.get_relevant_documents(query) 
        elif callable(getattr(retriever, 'invoke', None)):
            retrieved_docs = retriever.invoke(query, config={"configurable": {"search_kwargs": search_kwargs}})
        else:
            raise AttributeError("Retriever object lacks methods.")

        # Filter to only clean documents
        cleaned_rag_docs: List[Any] = [] 
        for doc in retrieved_docs:
            if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                cleaned_rag_docs.append(doc)
        
        # ----------------------------------------------------
        # 6. 🛠️ Rerank Strategy: Guaranteed Inclusion
        # ----------------------------------------------------
        
        # 6a. Filter RAG Search Results to exclude Priority Chunks (Dedup)
        # 🟢 FIX 4: ใช้ตัวแปรที่ถูกต้องในการทำ Dedup
        priority_chunk_uuids = set([d.metadata.get('chunk_uuid') for d in guaranteed_priority_chunks])
        
        rag_search_results_only = []
        newly_added_rag_count = 0
        
        for doc in cleaned_rag_docs:
            chunk_uuid = doc.metadata.get('chunk_uuid')
            if chunk_uuid and chunk_uuid in priority_chunk_uuids:
                continue # ข้าม Chunks ที่เป็น Priority ไปแล้ว
            rag_search_results_only.append(doc)
            newly_added_rag_count += 1
            
        logger.info(f"    - Dedup Merged: {correct_mapped_count} Priority Chunks + {newly_added_rag_count} New RAG Chunks.")


        # 6b. Determine Reranker Slots
        slots_available = max(0, top_k - correct_mapped_count) 
        
        # 6c. Rerank Logic
        # 🟢 FIX 5: ใช้ตัวแปรที่ถูกต้องในการเริ่มต้น
        final_selected_docs: List[Any] = guaranteed_priority_chunks 
        reranked_rag_results = []
        
        if slots_available > 0 and rag_search_results_only:
            reranker = get_global_reranker(top_k) 
            
            if reranker is None or not hasattr(reranker, 'compress_documents'):
                logger.error("🚨 CRITICAL FALLBACK: Reranker failed to load. Using simple truncation of NEW RAG results.")
                reranked_rag_results = rag_search_results_only[:slots_available]
            else:
                reranked_rag_results = reranker.compress_documents(
                    query=query, 
                    documents=rag_search_results_only, 
                    top_n=slots_available
                ) 
                
            final_selected_docs.extend(reranked_rag_results)
        
        # 7. Truncate (Safety measure)
        final_selected_docs = final_selected_docs[:top_k]
        
        # 8. Output Formatting (โค้ดเดิม แต่ใช้ final_selected_docs)
        top_evidences = []
        aggregated_context_list = []
        
        for doc in final_selected_docs: 
            source = doc.metadata.get("source") or doc.metadata.get("doc_source")
            content = doc.page_content.strip()
            relevance_score_raw = doc.metadata.get("relevance_score")
            
            relevance_score = f"{float(relevance_score_raw):.4f}" if relevance_score_raw is not None else "N/A"
            doc_uuid = doc.metadata.get("chunk_uuid") or doc.metadata.get("doc_uuid")
            
            top_evidences.append({
                "doc_uuid": doc_uuid,
                "doc_id": doc.metadata.get("stable_doc_uuid"),
                "doc_type": doc.metadata.get("doc_type"),
                "chunk_uuid": doc.metadata.get("chunk_uuid"),
                "source": source,
                "text": content,
                "relevance_score": relevance_score, 
                "chunk_index": doc.metadata.get("chunk_index")
            })
            doc_id_short = doc.metadata.get('stable_doc_uuid', 'N/A')[:8]
            aggregated_context_list.append(f"[SOURCE: {source} (ID:{doc_id_short}...)] {content}")

        aggregated_context = "\n\n---\n\n".join(aggregated_context_list)
        duration = time.time() - start_time 
        
        if level is not None:
            logger.critical(f"🧭 DEBUG: Aggregated Context Length for L{level} ({sub_id}) = {len(aggregated_context)}. Retrieval Time: {duration:.2f}s")


        return {
            "top_evidences": top_evidences,
            "aggregated_context": aggregated_context
        }
    
    except Exception as e:
        logger.error(f"retrieve_context_with_filter error: {type(e).__name__}: {e}")
        return {"top_evidences": [], "aggregated_context": f"ERROR: RAG retrieval failed due to {type(e).__name__}: {e}"}
        
def retrieve_context_for_low_levels(query: str, doc_type: str, enabler: Optional[str]=None,
                                 vectorstore_manager: Optional['VectorStoreManager']=None,
                                 top_k: int=LOW_LEVEL_K, initial_k: int=INITIAL_TOP_K, 
                                 # 🟢 NEW: ต้องรับ 2 arguments นี้
                                 mapped_uuids: Optional[List[str]]=None,
                                 priority_docs_input: Optional[List[Any]] = None, 
                                 sub_id: Optional[str]=None, level: Optional[int]=None) -> Dict[str, Any]:
    """
    Retrieves a small, focused context for low levels (L1, L2) using a reduced k (LOW_LEVEL_K).
    """
    # ใช้ฟังก์ชันหลัก แต่บังคับใช้ k ที่เหมาะสม
    return retrieve_context_with_filter(
        query=query,
        doc_type=doc_type,
        enabler=enabler,
        vectorstore_manager=vectorstore_manager,
        top_k=LOW_LEVEL_K,
        initial_k=initial_k, 
        # 🟢 NEW: ส่งต่อ arguments
        mapped_uuids=mapped_uuids, 
        priority_docs_input=priority_docs_input,
        sub_id=sub_id,
        level=level
    )

# ------------------------
# Robust JSON
# ------------------------
UUID_PATTERN = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', re.IGNORECASE)

def _robust_extract_json(text: str) -> Optional[Any]:
    if not text: return None
    # ใช้ re.sub เพื่อลบ code fences และปรับปรุงการค้นหา JSON
    txt = re.sub(r'^\s*```(?:json)?\s*|\s*```\s*$', '', text.strip(), flags=re.MULTILINE)
    
    # 1. ค้นหารูปแบบ JSON ทั่วไป
    for pattern in [r'(\{.*\})', r'(\[.*\])']:
        m = re.search(pattern, txt, flags=re.DOTALL)
        if m:
            try: return json.loads(m.group(1))
            except:
                try: return json5.loads(m.group(1)) # ลองใช้ json5 เพื่อ handle trailing commas, comments
                except: pass # ให้ลองไปขั้นตอนต่อไป
    
    # 2. ลองโหลดทั้งหมดโดยตรง (อาจมีข้อความอื่นปนมาเล็กน้อย)
    try: return json5.loads(txt)
    except: return None

def _normalize_keys(data: Any) -> Any:
    """Recursively normalizes common key variations to a standard set."""
    if isinstance(data, dict):
        mapping = {
            "llm_score": "score", 
            "reasoning": "reason", 
            "llm_reasoning": "reason", 
            "assessment_reason": "reason", 
            "comment": "reason",
            "pass": "is_passed", 
            "is_pass": "is_passed"
        }
        return {mapping.get(k.lower(), k): _normalize_keys(v) for k,v in data.items()}
    if isinstance(data, list): return [_normalize_keys(x) for x in data]
    return data

# ------------------------
# LLM fetcher
# ------------------------
def _fetch_llm_response(system_prompt: str, user_prompt: str, max_retries: int=_MAX_LLM_RETRIES) -> str:
    global _MOCK_FLAG
    
    if _MOCK_FLAG:
        # ใช้ Mock LLM ที่ถูกตั้งค่าไว้
        try:
             # เรียกใช้ Mock LLM โดยตรง
             resp = llm_instance.invoke([{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}], config={"temperature": 0.0})
             if hasattr(resp, "content"): return resp.content.strip()
             return str(resp).strip()
        except Exception as e:
            logger.error(f"Mock LLM invocation failed: {e}")
            raise ConnectionError("Mock LLM failed to respond.")


    if llm_instance is None: raise ConnectionError("LLM instance not initialized") # แก้ไขจาก throw เป็น raise แล้ว
    
    config = {"temperature": 0.0}
    for attempt in range(max_retries):
        try:
            resp = llm_instance.invoke([{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}], config=config)
            if hasattr(resp, "content"): return resp.content.strip()
            if isinstance(resp, dict) and "content" in resp: return resp["content"].strip()
            if isinstance(resp, str): return resp.strip()
            return str(resp).strip()
        except Exception as e:
            logger.warning(f"LLM attempt {attempt+1} failed: {e}")
            time.sleep(0.5)
    raise ConnectionError("LLM calls failed after retries")

# ------------------------
# Evaluation
# ------------------------
T = TypeVar("T", bound=BaseModel)

def _check_and_handle_empty_context(context: str, sub_id: str, level: int) -> Optional[Dict[str, Any]]:
    """Returns Failure result if context is empty or contains known error strings."""
    if not context or "ไม่มีหลักฐานที่เกี่ยวข้อง" in context or "ERROR:" in context.upper():
        logger.warning(f"Auto-FAIL L{level} for {sub_id}: Empty or Error Context detected from RAG.")
        # ป้องกันการแสดง context ยาวๆ ใน log
        context_preview = context.strip()[:100].replace("\n", " ") if context else "Empty Context"
        return {
            "score": 0, 
            "reason": f"หลักฐานที่ค้นหาได้ว่างเปล่าหรือไม่เกี่ยวข้อง (Context: {context_preview}).", 
            "is_passed": False
        }
    return None

def evaluate_with_llm(context: str, sub_criteria_name: str, level: int, statement_text: str, sub_id: str, **kwargs) -> Dict[str, Any]:
    
    # ตรวจสอบ Context ก่อนส่งให้ LLM
    failure_result = _check_and_handle_empty_context(context, sub_id, level)
    if failure_result:
        return failure_result
        
    # L3+ (Standard Evaluation)
    user_prompt = USER_ASSESSMENT_PROMPT.format(
        sub_criteria_name=sub_criteria_name, level=level, statement_text=statement_text, sub_id=sub_id,
        context=context or "ไม่มีหลักฐานที่เกี่ยวข้อง", pdca_phase=kwargs.get("pdca_phase",""), level_constraint=kwargs.get("level_constraint","")
    )
    try:
        # ใช้ .model_json_schema() เพื่อความเข้ากันได้กับ Pydantic v2+
        schema_json = json.dumps(StatementAssessment.model_json_schema(), ensure_ascii=False, indent=2)
    except: schema_json = '{"score":0,"reason":"string"}'
    
    system_prompt = SYSTEM_ASSESSMENT_PROMPT + "\n\n--- JSON SCHEMA ---\n" + schema_json + "\nIMPORTANT: Respond only with valid JSON."
    
    try:
        raw = _fetch_llm_response(system_prompt, user_prompt, _MAX_LLM_RETRIES)
        parsed = _normalize_keys(_robust_extract_json(raw) or {})
        
        score = int(parsed.get("score",0))
        is_passed = parsed.get("is_passed", score >= 1) # ใช้ score >= 1 เป็นค่า default ถ้า LLM ไม่ได้ส่ง is_passed
        
        return {"score":score,"reason":parsed.get("reason",""),"is_passed":is_passed}
        
    except Exception as e:
        logger.exception(f"evaluate_with_llm failed for {sub_id} L{level}: {e}")
        return {"score":0,"reason":f"LLM error: {e}","is_passed":False}

# NEW: Low-Level Evaluation (Simplified Prompt)
def evaluate_with_llm_low_level(context: str, sub_criteria_name: str, level: int, statement_text: str, sub_id: str, **kwargs) -> Dict[str, Any]:
    """
    Uses a simplified prompt for L1/L2 assessment to reduce complexity and cost.
    """
    
    # ตรวจสอบ Context ก่อนส่งให้ LLM
    failure_result = _check_and_handle_empty_context(context, sub_id, level)
    if failure_result:
        return failure_result

    # 🟢 ดึง Level Constraint ออกจาก kwargs
    level_constraint = kwargs.get("level_constraint", "")

    # L1/L2 (Low-Level Evaluation)
    user_prompt = USER_LOW_LEVEL_PROMPT.format(
        sub_criteria_name=sub_criteria_name, 
        level=level, 
        statement_text=statement_text, 
        sub_id=sub_id,
        context=context, 
        pdca_phase=kwargs.get("pdca_phase", ""),
        # 🟢 เพิ่ม Level Constraint เข้าไปใน Format
        level_constraint=level_constraint 
    )
    try:
        # ใช้ StatementAssessment Schema เดียวกัน
        schema_json = json.dumps(StatementAssessment.model_json_schema(), ensure_ascii=False, indent=2)
    except: schema_json = '{"score":0,"reason":"string"}'
    
    system_prompt = SYSTEM_LOW_LEVEL_PROMPT + "\n\n--- JSON SCHEMA ---\n" + schema_json + "\nIMPORTANT: Respond only with valid JSON."
    
    try:
        raw = _fetch_llm_response(system_prompt, user_prompt, _MAX_LLM_RETRIES)
        parsed = _normalize_keys(_robust_extract_json(raw) or {})
        
        score = int(parsed.get("score",0))
        # ⚠️ Note: is_passed ถูกส่งมาใน parsed จาก LLM หรือคำนวณจาก score >= 1
        is_passed = parsed.get("is_passed", score >= 1) 
        
        return {"score":score,"reason":parsed.get("reason",""),"is_passed":is_passed}
        
    except Exception as e:
        logger.exception(f"evaluate_with_llm_low_level failed for {sub_id} L{level}: {e}")
        return {"score":0,"reason":f"LLM error: {e}","is_passed":False}

# ------------------------
# Summarize
# ------------------------
def summarize_context_with_llm(context: str, sub_criteria_name: str, level: int, sub_id: str) -> Dict[str, Any]:
    if llm_instance is None: return {"summary":"LLM not available","suggestion_for_next_level":"Check LLM"}
    
    # จำกัด Context ให้สั้นลงเพื่อความเร็วและความเสถียร (4000 tokens)
    human_prompt = EVIDENCE_DESCRIPTION_PROMPT.format(sub_criteria_name=sub_criteria_name, level=level, context=(context or "")[:4000], sub_id=sub_id)
    
    try: schema_json = json.dumps(EvidenceSummary.model_json_schema(), ensure_ascii=False, indent=2)
    except: schema_json = "{}"
    
    system_prompt = SYSTEM_EVIDENCE_DESCRIPTION_PROMPT + "\n\n--- JSON SCHEMA ---\n" + schema_json + "\nRespond only with valid JSON."
    
    try:
        raw = _fetch_llm_response(system_prompt, human_prompt, 2)
        return _normalize_keys(_robust_extract_json(raw) or {})
    except Exception as e:
        logger.exception(f"summarize_context_with_llm failed: {e}")
        return {"summary":"LLM error","suggestion_for_next_level": str(e)}

# ------------------------
# Action plan
# ------------------------
def create_structured_action_plan(failed_statements_data: List[Dict[str,Any]], sub_id:str, enabler:str, target_level:int, max_retries:int=5) -> List[Dict[str,Any]]:
    if _MOCK_FLAG: return [{"Phase":"MOCK","Goal":f"MOCK plan for {sub_id}","Actions":[]}]
    
    try:
        # ใช้ .model_json_schema() เพื่อความเข้ากันได้กับ Pydantic v2+
        schema_json = json.dumps(ActionPlanActions.model_json_schema(), ensure_ascii=False, indent=2)
    except: schema_json = "{}"
    
    system_prompt = SYSTEM_ACTION_PLAN_PROMPT + "\n\n--- JSON SCHEMA ---\n" + schema_json + "\nRespond ONLY with a valid JSON ARRAY."
    
    statements_text = []
    for s in failed_statements_data:
        # จำกัดความยาวของ Statement และ Reason เพื่อป้องกันการล้น Token
        st = (s.get('statement','') or '')[:1000]
        rs = (s.get('reason','') or '')[:500]
        statements_text.append(f"Level:{s.get('level','N/A')}\nStatement:{st}\nReason:{rs}")
        
    human_prompt = ACTION_PLAN_PROMPT.format(sub_id=sub_id, target_level=target_level, failed_statements_list="\n\n".join(statements_text))
    
    for attempt in range(max_retries+1):
        try:
            raw = _fetch_llm_response(system_prompt, human_prompt,1)
            
            parsed = _robust_extract_json(raw) or []
            if isinstance(parsed, dict): parsed = [parsed] # แปลง dict เดี่ยวให้เป็น list
            
            valid_items = []
            for item in parsed:
                if not isinstance(item, dict): continue
                # ใช้ .get() เพื่อให้โค้ดรันได้แม้ไม่มี key
                item.setdefault("Phase",f"Fallback L{target_level}")
                item.setdefault("Goal","N/A")
                item.setdefault("Actions",[])
                valid_items.append(item)
            
            if valid_items: return valid_items
            
        except Exception as e:
            logger.warning(f"Action plan attempt {attempt+1} failed: {e}")
            time.sleep(0.5)
            
    # Fallback กรณีล้มเหลวทุกครั้ง
    return [{"Phase":f"Fallback L{target_level}","Goal":f"Manual review for {sub_id}","Actions":[{"Statement_ID":"LLM_ERROR","Recommendation":"Manual review required"}]}]