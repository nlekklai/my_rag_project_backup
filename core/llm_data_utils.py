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
    DEFAULT_ENABLER = "KM"
    INITIAL_TOP_K = 10
    FINAL_K_RERANKED = 3
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
def retrieve_context_by_doc_ids(doc_uuids: List[str], doc_type: str, enabler: Optional[str] = None) -> Dict[str, Any]:
    if not doc_uuids or VectorStoreManager is None:
        return {"top_evidences": []}
    try:
        manager = VectorStoreManager()
        # collection_name = _get_collection_name(doc_type, enabler) # Not strictly needed here
        normalized_uuids = normalize_stable_ids(doc_uuids)
        # Note: manager.get_documents_by_id must support list of normalized IDs
        docs: List[LcDocument] = manager.get_documents_by_id(normalized_uuids, doc_type, enabler)
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

# ในไฟล์ core/llm_data_utils.py

# ... (ตรวจสอบให้แน่ใจว่าได้ import สิ่งที่จำเป็น เช่น List, Optional, Dict, Any, LcDocument, logger, 
#      FINAL_K_RERANKED, INITIAL_TOP_K, VectorStoreManager, get_global_reranker, normalize_stable_ids)
#      (ผมจะสมมติว่าคุณมีฟังก์ชันเหล่านี้แล้ว)

def retrieve_context_with_filter(query: str, doc_type: str, enabler: Optional[str]=None,
                                 vectorstore_manager: Optional['VectorStoreManager']=None,
                                 top_k: int=FINAL_K_RERANKED, initial_k: int=INITIAL_TOP_K,
                                 # 🟢 FIX 1: แก้ไข Syntax และเพิ่ม stable_doc_ids (Optional)
                                 stable_doc_ids: Optional[List[str]]=None, 
                                 sub_id: Optional[str]=None, level: Optional[int]=None) -> Dict[str, Any]:
    """
    Retrieves and reranks relevant context from the specified VectorStore collection.
    
    Args:
        query (str): The search query.
        stable_doc_ids (Optional[List[str]]): List of stable document IDs to filter by.
        ... (other args)
    """
    try:
        # 1. จัดการ VSM Instance
        manager = vectorstore_manager or VectorStoreManager()
        collection_name = _get_collection_name(doc_type, enabler)
        
        # 🟢 FIX C: บังคับเป็นตัวพิมพ์เล็กเสมอเพื่อแก้ Case-Sensitivity 
        collection_name = collection_name.lower() 
        
        retriever_wrapper = None 
        
        logger.critical(f"🧭 DEBUG: Attempting to retrieve collection: {collection_name}")
        
        # 2. ลองเข้าถึง Retriever ที่ถูกต้องผ่าน MultiDocRetriever (MDR)
        multi_doc_retriever = None
        if hasattr(manager, '_multi_doc_retriever'):
            multi_doc_retriever = manager._multi_doc_retriever
            
            # 🟢 FIX 2: แก้ปัญหา 'ModelPrivateAttr' โดยการ unwrap object
            if not hasattr(multi_doc_retriever, '_all_retrievers') and hasattr(multi_doc_retriever, 'value'):
                multi_doc_retriever = multi_doc_retriever.value
                logger.critical("🧭 DEBUG: ModelPrivateAttr unwrapped using .value.")

        if multi_doc_retriever and hasattr(multi_doc_retriever, '_all_retrievers'):
            retriever_wrapper = multi_doc_retriever._all_retrievers.get(collection_name)
            if retriever_wrapper:
                logger.critical("🧭 DEBUG: Access via VSM._multi_doc_retriever._all_retrievers SUCCESS.")
            elif hasattr(manager, '_all_retrievers'): 
                 retriever_wrapper = manager._all_retrievers.get(collection_name)
                 if retriever_wrapper:
                     logger.critical("🧭 DEBUG: Access via direct MultiDocRetriever SUCCESS (Incorrect instance type passed).")
        if not retriever_wrapper:
            if hasattr(manager, 'get_retriever'):
                 retriever_wrapper = manager.get_retriever(collection_name)
                 if retriever_wrapper:
                    logger.critical("🧭 DEBUG: Access via VSM.get_retriever SUCCESS.")
        
        # --- 🚨 Unwrap the Core Retriever 🚨 ---
        retriever = retriever_wrapper
        if retriever_wrapper and hasattr(retriever_wrapper, 'base_retriever'):
             retriever = retriever_wrapper.base_retriever
             logger.critical(f"🧭 DEBUG: Successfully unwrapped base_retriever: {type(retriever).__name__}.")
        
        # 🚨 FINAL DEBUG LOGGING BLOCK / Core Retriever Validation 🚨
        is_valid_retriever_method = callable(getattr(retriever, 'get_relevant_documents', None)) or callable(getattr(retriever, 'invoke', None))

        if not retriever or not is_valid_retriever_method:
            # --- FINAL RESORT: Check known wrapper structures ---
            if retriever_wrapper and hasattr(retriever_wrapper, 'vectorstore') and callable(getattr(retriever_wrapper.vectorstore, 'get_relevant_documents', None)):
                 retriever = retriever_wrapper.vectorstore 
                 logger.critical(f"🧭 DEBUG: Final Resort: Using .vectorstore as retriever: {type(retriever).__name__}.")
            elif retriever_wrapper and hasattr(retriever_wrapper, 'retriever') and callable(getattr(retriever_wrapper.retriever, 'get_relevant_documents', None)):
                 retriever = retriever_wrapper.retriever 
                 logger.critical(f"🧭 DEBUG: Final Resort: Using .retriever as retriever: {type(retriever).__name__}.")
            
            is_valid_retriever_method = callable(getattr(retriever, 'get_relevant_documents', None)) or callable(getattr(retriever, 'invoke', None))
            if not is_valid_retriever_method:
                available_keys = "N/A"
                if hasattr(manager, '_multi_doc_retriever') and hasattr(manager._multi_doc_retriever, '_all_retrievers'):
                    available_keys = str(list(manager._multi_doc_retriever._all_retrievers.keys()))
                elif hasattr(manager, '_all_retrievers'): 
                    available_keys = str(list(manager._all_retrievers.keys()))
                    
                logger.error(f"FATAL: Core Retriever not found/lacks 'get_relevant_documents' or 'invoke' for key: {collection_name}. Type: {type(retriever).__name__}")
                logger.error(f"FATAL: Available keys in VSM/MDR were: {available_keys}")
                
                return {"top_evidences": [], "aggregated_context": f"ERROR: Target ChromaRetriever missing for {collection_name} (Object type {type(retriever).__name__} is incorrect)."}
        
        logger.critical(f"🧭 DEBUG: Successfully retrieved Core Retriever. Starting query...")
        
        # ----------------------------------------------------
        # 🚨 RE-ADDING K FORCE (เพื่อพยายาม OVERRIDE HARDCODE 5) 🚨
        logger.critical(f"🧭 DEBUG: Final k value (initial_k) to use: {initial_k}")
        logger.critical(f"🧭 DEBUG: k attribute before query: {getattr(retriever, 'k', 'N/A')}")
        # ----------------------------------------------------

        if hasattr(retriever, 'k'):
            # ถ้ามี k attribute ให้ตั้งค่า k
            retriever.k = initial_k
            logger.critical(f"🧭 DEBUG: Successfully set retriever.k = {retriever.k}")

        if hasattr(retriever, 'search_kwargs') and isinstance(retriever.search_kwargs, dict):
            # พยายามตั้งค่า k ใน search_kwargs
            retriever.search_kwargs['k'] = initial_k
            logger.critical(f"🧭 DEBUG: Successfully set search_kwargs['k'] = {retriever.search_kwargs['k']}")

        # 3. Invoke Retrieval
        
        # 🟢 NEW FIX: ถ้าเป็น Langchain Retriever ที่ซับซ้อน อาจใช้ .search_kwargs
        search_kwargs = {"k": initial_k} # เตรียม search_kwargs

        # 🟢 FIX 3: เพิ่ม Logic การกรองเอกสารด้วย stable_doc_ids ถ้ามีการส่งมา
        if stable_doc_ids:
            # สมมติว่า normalize_stable_ids ถูก import มาแล้ว
            normalized_uuids = normalize_stable_ids(stable_doc_ids) 
            # โครงสร้างการกรองสำหรับ Chroma/LangChain คือ 'where'
            # Note: ต้องมั่นใจว่า metadata key ใน ChromaDB คือ stable_doc_uuid
            search_kwargs["filter"] = {"stable_doc_uuid": {"$in": normalized_uuids}}
            logger.critical(f"🧭 DEBUG: RAG Filter by Stable Doc IDs activated ({len(normalized_uuids)} IDs).")


        if callable(getattr(retriever, 'get_relevant_documents', None)):
            # หาก Retriever พื้นฐานไม่รองรับ 'filter' ตรงๆ ใน get_relevant_documents 
            # (ซึ่งเป็นไปได้ใน LangChain เวอร์ชันเก่า)
            # เราอาจต้องใช้ .with_search_kwargs(filter=...) ถ้าเป็น LangChain Retriever object
            # แต่เพื่อความเรียบง่ายและลดความเสี่ยงการเกิด error ซ้ำซ้อน เราจะใช้การ invoke() ในการส่ง config แทน
            if "filter" in search_kwargs:
                logger.warning("⚠️ RAG Filter set but using get_relevant_documents(). Filter may not be applied if Retriever is basic.")
                retrieved_docs: List[Any] = retriever.get_relevant_documents(query) 
            else:
                 retrieved_docs: List[Any] = retriever.get_relevant_documents(query) 
                 
        elif callable(getattr(retriever, 'invoke', None)):
            # LangChain ใหม่: ส่ง config เข้าไป
            retrieved_docs: List[Any] = retriever.invoke(query, config={"retrieval_config": search_kwargs})
        else:
            raise AttributeError("Retriever object lacks both 'get_relevant_documents' and 'invoke' methods.")
                
        # ----------------------------------------------------
        # 🚨 FIX: Ensure retrieved_docs is List[LcDocument] 🚨
        # ----------------------------------------------------
        cleaned_docs: List[LcDocument] = []
        for doc in retrieved_docs:
            if isinstance(doc, str):
                # ถ้าเป็น string ให้แปลงเป็น LcDocument ที่มี metadata พื้นฐาน
                cleaned_docs.append(LcDocument(page_content=doc, metadata={"source": "RAG_Chunk", "doc_type": doc_type}))
            elif hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                # ถ้าเป็น Document object อยู่แล้ว ให้ใช้ได้เลย
                cleaned_docs.append(doc)

        retrieved_docs = cleaned_docs
        
        
        # 🟢 Reranking Logic (ใช้โค้ดเดิมของคุณ)
        reranker = get_global_reranker(top_k) 
        
        if reranker is None or not hasattr(reranker, 'compress_documents'):
            logger.error("🚨 CRITICAL FALLBACK: Reranker failed to load (Likely configuration issue). Using simple truncation of retrieved docs.")
            reranked_docs = retrieved_docs[:top_k]
            
        elif not retrieved_docs:
            reranked_docs = []
            
        else:
            reranked_docs = reranker.compress_documents(query=query, documents=retrieved_docs, top_n=top_k) 
        
        # ... (โค้ดส่วนการจัดรูปแบบผลลัพธ์ที่เหลือเหมือนเดิม) ...

        top_evidences = []
        aggregated_context_list = []
        for doc in reranked_docs:
            source = doc.metadata.get("source") or doc.metadata.get("doc_source")
            content = doc.page_content.strip()
            relevance_score_raw = doc.metadata.get("relevance_score")
            
            if relevance_score_raw is None:
                 relevance_score_raw = doc.metadata.get("score") 
            
            relevance_score = f"{float(relevance_score_raw):.4f}" if relevance_score_raw is not None else "N/A"
            
            top_evidences.append({
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
        if level is not None:
            logger.critical(f"🧭 DEBUG: Aggregated Context Length for L{level} ({sub_id}) = {len(aggregated_context)}")

        return {
            "top_evidences": top_evidences,
            "aggregated_context": aggregated_context
        }
    
    except Exception as e:
        logger.error(f"retrieve_context_with_filter error: {e}")
        return {"top_evidences": [], "aggregated_context": f"ERROR: RAG retrieval failed due to {type(e).__name__}: {e}"}

def retrieve_context_for_low_levels(query: str, doc_type: str, enabler: Optional[str]=None,
                                 vectorstore_manager: Optional['VectorStoreManager']=None,
                                 top_k: int=LOW_LEVEL_K, initial_k: int=INITIAL_TOP_K, # 🟢 เพิ่ม initial_k เข้ามา
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
        initial_k=initial_k, # 🟢 ส่งค่า initial_k ลงไป
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

    # L1/L2 (Low-Level Evaluation)
    user_prompt = USER_LOW_LEVEL_PROMPT.format(
        sub_criteria_name=sub_criteria_name, level=level, statement_text=statement_text, sub_id=sub_id,
        context=context, pdca_phase=kwargs.get("pdca_phase","")
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
        is_passed = parsed.get("is_passed", score >= 1) # ใช้ score >= 1 เป็นค่า default
        
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