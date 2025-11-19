"""
llm_data_utils.py
Robust LLM + RAG utilities for SEAM assessment.
Responsibilities:
- Retrieval wrapper: retrieve_context_with_filter & retrieve_context_by_doc_ids
- Robust JSON extraction & normalization (_robust_extract_json, _normalize_keys)
- LLM invocation wrappers with retries (_fetch_llm_response)
- evaluate_with_llm: produce {score, reason, is_passed, P/D/C/A breakdown}
- summarize_context_with_llm: produce evidence summary
- create_structured_action_plan: generate action plan JSON list
- enhance_query_for_statement: Multi-Query generation for RAG
- Mock control helper: set_mock_control_mode
"""
import logging, time, json, json5, random, hashlib, regex as re
from typing import List, Dict, Any, Optional, TypeVar, Final, Union
from pydantic import BaseModel, ConfigDict, Field, RootModel 
import uuid 
import sys 

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
    # NOTE: Assuming the correct schemas are available in core.assessment_schema
    from core.vectorstore import VectorStoreManager, get_global_reranker, _get_collection_name, ChromaRetriever
    # 📌 ASSUMED: We now import the comprehensive schema
    from core.assessment_schema import CombinedAssessment, EvidenceSummary
    # StatementAssessment is no longer primarily used, but might be for compatibility
    try:
        from core.assessment_schema import StatementAssessment
    except ImportError:
        class StatementAssessment(BaseModel): score: int; reason: str

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

    # 🟢 FIX: Define missing constants here
    INITIAL_TOP_K: Final[int] = 10
    FINAL_K_RERANKED: Final[int] = 3
    DEFAULT_ENABLER = "KM"
    
    # 🟢 PLACEHOLDER: NEW COMBINED ASSESSMENT SCHEMA
    class CombinedAssessment(BaseModel):
        model_config = ConfigDict(extra='allow')
        score: int = Field(0, description="Overall Score (0-4)")
        reason: str = Field("Mock reason", description="Detailed reasoning.")
        is_passed: bool = Field(False, description="Pass status.")
        P_Plan_Score: int = Field(0, description="Score for Plan (0-2)")
        D_Do_Score: int = Field(0, description="Score for Do (0-2)")
        C_Check_Score: int = Field(0, description="Score for Check (0-2)")
        A_Act_Score: int = Field(0, description="Score for Act (0-2)")
        assessment_comment: Optional[str] = None
        
    class StatementAssessment(BaseModel): score: int = 0; reason: str = "Mock reason"
    class EvidenceSummary(BaseModel): summary: str; suggestion_for_next_level: str
    
    # 🟢 FIX: แก้ไข Pydantic V2 Syntax ใน Placeholder
    class ActionPlanActions(BaseModel):
        Phase: str = "Mock Phase"
        Goal: str = "Mock Goal"
        Actions: List[Dict[str,Any]] = []
        
    class LcDocument:
        def __init__(self, page_content, metadata): self.page_content=page_content; self.metadata=metadata
    
    # Define mock prompts to prevent crash if real ones are missing
    SYSTEM_ASSESSMENT_PROMPT = "Assess the statement based on the provided context."
    USER_ASSESSMENT_PROMPT = "Context: {context}\nStatement: {statement_text}\nLevel Constraint: {level_constraint}\nContextual Rules: {contextual_rules_prompt}"
    SYSTEM_ACTION_PLAN_PROMPT = "Generate an action plan."
    ACTION_PLAN_PROMPT = "Failed statements: {failed_statements_list}"
    SYSTEM_EVIDENCE_DESCRIPTION_PROMPT = "Summarize evidence."
    EVIDENCE_DESCRIPTION_PROMPT = "Context: {context}"
    SYSTEM_LOW_LEVEL_PROMPT = "Assess L1/L2 simply."
    USER_LOW_LEVEL_PROMPT = "Context: {context}\nL1/L2 Statement: {statement_text}\nLevel Constraint: {level_constraint}\nContextual Rules: {contextual_rules_prompt}"


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

# -------------------- RAG Retrieval Functions --------------------

def retrieve_context_with_filter(
    query: Union[str, List[str]], 
    doc_type: str, 
    enabler: Optional[str]=None,
    vectorstore_manager: Optional['VectorStoreManager']=None,
    top_k: int=FINAL_K_RERANKED, 
    initial_k: int=INITIAL_TOP_K,
    mapped_uuids: Optional[List[str]]=None,
    stable_doc_ids: Optional[List[str]]=None,
    priority_docs_input: Optional[List[Any]] = None,
    # 🟢 FIX: เพิ่ม arguments ที่ขาดหายไป (จาก seam_assessment.py)
    sequential_chunk_uuids: Optional[List[str]] = None, 
    sub_id: Optional[str]=None, 
    level: Optional[int]=None,
    logger: logging.Logger = logging.getLogger(__name__) 
) -> Dict[str, Any]:

    """
    Retrieves and reranks relevant context from the specified VectorStore collection,
    supporting MULTI-QUERY search and GUARANTEEING the inclusion of pre-mapped 
    priority documents (Hybrid Retrieval).
    """
    start_time = time.time()
    all_retrieved_chunks: List[Any] = []
    
    # 📌 FIX: รวบรวม sequential_chunk_uuids เข้าใน priority_docs_input
    if sequential_chunk_uuids:
        if mapped_uuids:
             mapped_uuids.extend(sequential_chunk_uuids)
        else:
             mapped_uuids = sequential_chunk_uuids

    try:
        # 1. Setup & Query List
        manager = vectorstore_manager
        if manager is None: raise ValueError("VectorStoreManager is not initialized.")
        collection_name = _get_collection_name(doc_type, enabler).lower()
        queries_to_run = [query] if isinstance(query, str) else query
        
        # 2. Persistent Mapping (Hybrid Retrieval) SETUP
        guaranteed_priority_chunks: List[Any] = []
        mapped_uuids_for_vsm_search: Optional[List[str]] = None
        
        if priority_docs_input:
            logger.critical(f"🧭 DEBUG: Using {len(priority_docs_input)} pre-calculated priority chunks (Limited Chunks Mode).")
            guaranteed_priority_chunks = priority_docs_input
            mapped_uuids_for_vsm_search = None 
        elif mapped_uuids:
            # 2b. Fallback: ใช้ mapped_uuids เป็นตัวกรองใน VSM Search (Step 3)
            normalized_mapped_uuids = normalize_stable_ids(mapped_uuids)
            mapped_uuids_for_vsm_search = normalized_mapped_uuids
            logger.critical(f"🧭 DEBUG: [FALLBACK] Using {len(normalized_mapped_uuids)} UUIDs as search filter.")
            
        # 3. 🎯 Invoke Retrieval (MULTI-QUERY Standard RAG Search)
        rag_search_filters = mapped_uuids_for_vsm_search
        if stable_doc_ids:
            additional_uuids = set(normalize_stable_ids(stable_doc_ids))
            if rag_search_filters:
                rag_search_filters = list(set(rag_search_filters) | additional_uuids)
            else:
                rag_search_filters = list(additional_uuids)
        
        retriever = manager.get_retriever(collection_name)
        if retriever is None: raise ValueError(f"Retriever initialization failed for {collection_name}.")
        
        search_kwargs = {"k": initial_k}
        
        for q in queries_to_run:
            search_kwargs_for_query = search_kwargs.copy()
            
            # (❌ คอมเมนต์ส่วน Filter ออกไปแล้ว)

            retrieved_docs: List[Any] = []
            
            # 📌 เรียกใช้ Retriever
            if callable(getattr(retriever, 'invoke', None)):
                retrieved_docs = retriever.invoke(q, config={"configurable": {"search_kwargs": search_kwargs_for_query}})
            elif callable(getattr(retriever, 'get_relevant_documents', None)):
                retrieved_docs = retriever.get_relevant_documents(q)
            else:
                raise AttributeError("Retriever object lacks methods.")

            # 📌 รวบรวมผลลัพธ์ทั้งหมด
            all_retrieved_chunks.extend(retrieved_docs)
            
        logger.critical(f"🧭 DEBUG: Total chunks retrieved before dedup: {len(all_retrieved_chunks)}")

        # 4. 🛠️ Rerank Strategy: Guaranteed Inclusion (รวม Priority Chunks)

        all_chunks_to_process = all_retrieved_chunks + guaranteed_priority_chunks

        # 4b. FIX: Filter RAG Search Results to exclude Priority Chunks (Dedup)
        unique_chunks_map = {}
        for doc in all_chunks_to_process:
            chunk_uuid = doc.metadata.get('chunk_uuid')
            
            # 🟢 FIX (NameError): ใช้ uuid.uuid4()
            if not chunk_uuid:
                # Fallback Dedup: (Content + Source)
                content_hash = hash(doc.page_content.strip())
                source_id = doc.metadata.get('doc_uuid') or doc.metadata.get('stable_doc_uuid') or doc.metadata.get('source', 'unknown')
                dedup_key = f"{source_id}-{content_hash}"
                
                if dedup_key not in unique_chunks_map:
                    # สร้าง UUID จำลองสำหรับ document ที่ไม่มี chunk_uuid
                    doc.metadata['chunk_uuid'] = str(uuid.uuid4())
                    unique_chunks_map[dedup_key] = doc
                    # ใช้ chunk_uuid ที่สร้างใหม่สำหรับ Logic ต่อไป
                    chunk_uuid = doc.metadata['chunk_uuid']
                else:
                    # ถ้าซ้ำกันด้วย content hash ก็ข้ามไป
                    continue
            
            # Logic Dedup:
            if chunk_uuid and chunk_uuid not in unique_chunks_map:
                 if doc in guaranteed_priority_chunks and doc.metadata.get('relevance_score') is None:
                     doc.metadata['relevance_score'] = 1.0 
                 unique_chunks_map[chunk_uuid] = doc
            
        deduplicated_chunks = list(unique_chunks_map.values())
        
        # 4c. กำหนดจำนวน Slots Reranker 
        correct_mapped_count = len([d for d in deduplicated_chunks if d in guaranteed_priority_chunks]) # นับเฉพาะที่ยังอยู่หลัง Dedup
        slots_available = max(0, top_k - correct_mapped_count)
        
        logger.info(f"    - Dedup Merged: Total unique chunks = {len(deduplicated_chunks)}. Priority Chunks (Guaranteed) = {correct_mapped_count}.")


        # 4d. Rerank Logic
        final_selected_docs: List[Any] = [d for d in deduplicated_chunks if d in guaranteed_priority_chunks] # ใช้ Chunks ที่ถูก Guarantee และ Dedup แล้ว

        priority_chunk_uuids = set([d.metadata.get('chunk_uuid') for d in final_selected_docs])
        rerank_candidates = [d for d in deduplicated_chunks if d.metadata.get('chunk_uuid') not in priority_chunk_uuids]

        if slots_available > 0 and rerank_candidates:
            reranker = get_global_reranker(top_k)

            if reranker is None or not hasattr(reranker, 'compress_documents'):
                logger.error("🚨 CRITICAL FALLBACK: Reranker failed to load. Using simple truncation of NEW RAG results.")
                # 📌 FIX: ใช้ score ภายใน metadata (ถ้ามี) สำหรับการจัดเรียง Fallback
                reranked_rag_results = sorted(rerank_candidates, key=lambda x: x.metadata.get('relevance_score', 0.0), reverse=True)[:slots_available]
            else:
                rerank_query = queries_to_run[0] 
                
                # 📌 Reranker จะเพิ่ม score กลับเข้าไปใน metadata
                reranked_rag_results = reranker.compress_documents(
                    query=rerank_query,
                    documents=rerank_candidates,
                    top_n=slots_available
                )

            final_selected_docs.extend(reranked_rag_results)

        # 5. Output Formatting
        top_evidences = []
        aggregated_context_list = []

        for doc in final_selected_docs[:top_k]:
            source = doc.metadata.get("source") or doc.metadata.get("doc_source")
            content = doc.page_content.strip()
            # 🟢 FIX: ใช้ 'relevance_score' จาก Metadata ซึ่งถูกตั้งโดย Reranker
            relevance_score_raw = doc.metadata.get("relevance_score") 

            relevance_score = f"{float(relevance_score_raw):.4f}" if relevance_score_raw is not None else "N/A"
            doc_uuid = doc.metadata.get("doc_uuid") or doc.metadata.get("chunk_uuid")

            top_evidences.append({
                # 🟢 FIX: ใช้ chunk_uuid
                "doc_uuid": doc_uuid,
                "doc_id": doc.metadata.get("stable_doc_uuid"),
                "doc_type": doc.metadata.get("doc_type"),
                "chunk_uuid": doc.metadata.get("chunk_uuid"), 
                "source": source,
                "text": content,
                "relevance_score": relevance_score,
                "chunk_index": doc.metadata.get("chunk_index"),
                # 🟢 FIX: เพิ่ม 'score' สำหรับใช้ใน Logic ของ _run_single_assessment
                "score": float(relevance_score_raw) if relevance_score_raw is not None else 0.0 
            })
            doc_id_short = doc.metadata.get('stable_doc_uuid', 'N/A')[:8]
            aggregated_context_list.append(f"[SOURCE: {source} (ID:{doc_id_short}...)] {content}")

        aggregated_context = "\n\n---\n\n".join(aggregated_context_list)
        duration = time.time() - start_time

        if level is not None:
            logger.critical(f"🧭 DEBUG: Aggregated Context Length for L{level} ({sub_id}) = {len(aggregated_context)}. Retrieval Time: {duration:.2f}s")


        return {
            "top_evidences": top_evidences,
            "aggregated_context": aggregated_context,
            "retrieval_time": duration # 🟢 FIX: เพิ่ม retrieval_time กลับมาในผลลัพธ์
        }

    except Exception as e:
        logger.error(f"retrieve_context_with_filter error: {type(e).__name__}: {e}")
        return {"top_evidences": [], "aggregated_context": f"ERROR: RAG retrieval failed due to {type(e).__name__}: {e}", "retrieval_time": time.time() - start_time}

def retrieve_context_for_low_levels(query: str, doc_type: str, enabler: Optional[str]=None,
                                 vectorstore_manager: Optional['VectorStoreManager']=None,
                                 top_k: int=LOW_LEVEL_K, initial_k: int=INITIAL_TOP_K,
                                 # 🟢 NEW: ส่งต่อ arguments
                                 mapped_uuids: Optional[List[str]]=None,
                                 priority_docs_input: Optional[List[Any]] = None,
                                 sequential_chunk_uuids: Optional[List[str]] = None, 
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
        mapped_uuids=mapped_uuids,
        priority_docs_input=priority_docs_input,
        sequential_chunk_uuids=sequential_chunk_uuids, 
        sub_id=sub_id,
        level=level
    )
# -------------------- Query Enhancement Functions --------------------
def enhance_query_for_statement(
    statement_text: str,
    sub_id: str,
    # 🟢 FIX: Argument list ที่ถูกต้องตามการเรียกใน seam_assessment.py
    statement_id: str, 
    level: int,
    enabler_id: str,
    focus_hint: str,
    llm_executor: Any = None
) -> List[str]:
    """
    Generates a list of tailored queries (Multi-Query strategy) based on the statement 
    and PDCA focus. The logic is hardcoded here to generate P/D, C, and A queries 
    based on the assessment level (L3+ gets C/A queries).
    
    Returns: List[str] of queries.
    """
    
    # Q1: Base Query (P/D Focus)
    # เน้นที่ statement หลักและข้อจำกัดของ level, เป็นคำค้นหาหลัก
    base_query = (
        f"{statement_text}. {focus_hint} หลักฐานแสดงแผน การดำเนินการ และโครงสร้างของ {statement_id} "
        f"ตามบริบทของ {enabler_id}"
    )
    
    queries = [base_query]

    # Q2 & Q3: เพิ่ม C/A Focus Queries สำหรับระดับ L3 ขึ้นไป
    # เพื่อให้แน่ใจว่า RAG จะหาหลักฐานการตรวจสอบ (Check) และการปรับปรุง (Act)
    if level >= 3:
        
        # 🟢 C (Check/Evaluation) Focus Query
        # เน้นหาหลักฐานการวัดผล ประเมินผล
        c_query = (
            f"หลักฐานการวัดผล ประเมินผล หรือการตรวจสอบ ว่า {statement_id} "
            f"ดำเนินการตามแผนหรือไม่ รายงานการตรวจสอบ รายงานการวัดผลความเข้าใจ "
            f"แบบสอบถามผลตอบรับ การวิเคราะห์ช่องว่าง ผลลัพธ์ของการประเมิน"
        )
        queries.append(c_query)

        # 🟢 A (Act/Improvement) Focus Query
        # เน้นหาหลักฐานการปรับปรุง การทบทวน การเปลี่ยนแปลง
        a_query = (
            f"หลักฐานการปรับปรุง การทบทวน หรือการเปลี่ยนแปลงวิธีการดำเนินการของ {statement_id} "
            f"ตามผลการประเมิน ข้อเสนอแนะ หรือผลตอบรับ การปรับปรุงแผนงาน "
            f"บทเรียนที่ได้รับ และการวางแผนสำหรับการดำเนินการรอบถัดไป"
        )
        queries.append(a_query)
    
    # สำหรับ L1/L2 จะคืนค่าเฉพาะ Base Query เดียว
    logger.info(f"Generated {len(queries)} queries for {sub_id} L{level} (ID: {statement_id}).")
    return queries


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
            "is_pass": "is_passed",
            # 🟢 NEW: Normalize PDCA keys for consistency
            "p_score": "P_Plan_Score",
            "d_score": "D_Do_Score",
            "c_score": "C_Check_Score",
            "a_score": "A_Act_Score",
            "p_plan": "P_Plan_Score",
            "d_do": "D_Do_Score",
            "c_check": "C_Check_Score",
            "a_act": "A_Act_Score"
        }
        return {mapping.get(k.lower(), k): _normalize_keys(v) for k,v in data.items()}
    if isinstance(data, list): return [_normalize_keys(x) for x in data]
    return data

# ------------------------
# LLM fetcher
# ------------------------
def _fetch_llm_response(
    system_prompt: str, 
    user_prompt: str, 
    max_retries: int=_MAX_LLM_RETRIES,
    llm_executor: Any = None 
) -> str:
    global _MOCK_FLAG

    llm = llm_executor
    
    if llm is None and not _MOCK_FLAG: 
        raise ConnectionError("LLM instance not initialized (Missing llm_executor).")

    if _MOCK_FLAG:
        # ใช้ Mock LLM ที่ถูกตั้งค่าไว้
        try:
             resp = llm.invoke([{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}], config={"temperature": 0.0})
             if hasattr(resp, "content"): return resp.content.strip()
             return str(resp).strip()
        except Exception as e:
            logger.error(f"Mock LLM invocation failed: {e}")
            raise ConnectionError("Mock LLM failed to respond.")

    config = {"temperature": 0.0}
    for attempt in range(max_retries):
        try:
            resp = llm.invoke([{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}], config=config)
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
        # 🟢 NEW: Return all PDCA keys with 0 for consistency with CombinedAssessment
        return {
            "score": 0,
            "reason": f"หลักฐานที่ค้นหาได้ว่างเปล่าหรือไม่เกี่ยวข้อง (Context: {context_preview}).",
            "is_passed": False,
            "P_Plan_Score": 0,
            "D_Do_Score": 0,
            "C_Check_Score": 0,
            "A_Act_Score": 0,
        }
    return None

def _extract_combined_assessment(parsed: Dict[str, Any], score_default_key: str = "score") -> Dict[str, Any]:
    """Helper to safely extract combined assessment results."""
    # 🟢 NEW: Extract all scores needed by seam_assessment.py (Action #1 logic)
    score = int(parsed.get(score_default_key, 0))
    is_passed = parsed.get("is_passed", score >= 1) # ใช้ score >= 1 เป็นค่า default ถ้า LLM ไม่ได้ส่ง is_passed

    result = {
        "score": score,
        "reason": parsed.get("reason", "No reason provided by LLM."),
        "is_passed": is_passed,
        "P_Plan_Score": int(parsed.get("P_Plan_Score", 0)),
        "D_Do_Score": int(parsed.get("D_Do_Score", 0)),
        "C_Check_Score": int(parsed.get("C_Check_Score", 0)),
        "A_Act_Score": int(parsed.get("A_Act_Score", 0)),
    }
    return result

def evaluate_with_llm(context: str, sub_criteria_name: str, level: int, statement_text: str, sub_id: str, llm_executor: Any, **kwargs) -> Dict[str, Any]:
    """Standard Evaluation for L3+"""

    # ตรวจสอบ Context ก่อนส่งให้ LLM
    failure_result = _check_and_handle_empty_context(context, sub_id, level)
    if failure_result:
        return failure_result

    contextual_rules_prompt = kwargs.get("contextual_rules_prompt", "")

    # L3+ (Standard Evaluation)
    user_prompt = USER_ASSESSMENT_PROMPT.format(
        sub_criteria_name=sub_criteria_name, 
        level=level, 
        statement_text=statement_text, 
        sub_id=sub_id,
        context=context or "ไม่มีหลักฐานที่เกี่ยวข้อง", 
        pdca_phase=kwargs.get("pdca_phase",""), 
        level_constraint=kwargs.get("level_constraint",""),
        contextual_rules_prompt=contextual_rules_prompt 
    )
    try:
        schema_json = json.dumps(CombinedAssessment.model_json_schema(), ensure_ascii=False, indent=2)
    except: schema_json = '{"score":0,"reason":"string"}'

    system_prompt = SYSTEM_ASSESSMENT_PROMPT + "\n\n--- JSON SCHEMA ---\n" + schema_json + "\nIMPORTANT: Respond only with valid JSON."

    try:
        # 1. เรียก LLM และรับ raw response
        raw = _fetch_llm_response(system_prompt, user_prompt, _MAX_LLM_RETRIES, llm_executor=llm_executor)
        
        # 2. Extract และ Normalize JSON
        parsed = _normalize_keys(_robust_extract_json(raw) or {})
        
        # 3. คืนผลลัพธ์
        return _extract_combined_assessment(parsed, score_default_key="score")

    except Exception as e:
        logger.exception(f"evaluate_with_llm failed for {sub_id} L{level}: {e}")
        # 🟢 NEW: Return all PDCA keys with 0 for error case
        return {
            "score":0,
            "reason":f"LLM error: {e}",
            "is_passed":False,
            "P_Plan_Score": 0,
            "D_Do_Score": 0,
            "C_Check_Score": 0,
            "A_Act_Score": 0,
        }

def evaluate_with_llm_low_level(context: str, sub_criteria_name: str, level: int, statement_text: str, sub_id: str, llm_executor: Any, **kwargs) -> Dict[str, Any]:
    """
    Uses a simplified prompt for L1/L2 assessment to reduce complexity and cost.
    """

    # ตรวจสอบ Context ก่อนส่งให้ LLM
    failure_result = _check_and_handle_empty_context(context, sub_id, level)
    if failure_result:
        return failure_result

    level_constraint = kwargs.get("level_constraint", "")
    contextual_rules_prompt = kwargs.get("contextual_rules_prompt", "") 

    # L1/L2 (Low-Level Evaluation)
    user_prompt = USER_LOW_LEVEL_PROMPT.format(
        sub_criteria_name=sub_criteria_name,
        level=level,
        statement_text=statement_text,
        sub_id=sub_id,
        context=context,
        pdca_phase=kwargs.get("pdca_phase", ""),
        level_constraint=level_constraint,
        contextual_rules_prompt=contextual_rules_prompt 
    )
    try:
        schema_json = json.dumps(CombinedAssessment.model_json_schema(), ensure_ascii=False, indent=2)
    except: schema_json = '{"score":0,"reason":"string"}'

    system_prompt = SYSTEM_LOW_LEVEL_PROMPT + "\n\n--- JSON SCHEMA ---\n" + schema_json + "\nIMPORTANT: Respond only with valid JSON."

    try:
        # 1. เรียก LLM และรับ raw response
        raw = _fetch_llm_response(system_prompt, user_prompt, _MAX_LLM_RETRIES, llm_executor=llm_executor)
        
        # 2. Extract และ Normalize JSON
        parsed = _normalize_keys(_robust_extract_json(raw) or {})

        # 3. คืนผลลัพธ์
        return _extract_combined_assessment(parsed, score_default_key="score")

    except Exception as e:
        logger.exception(f"evaluate_with_llm_low_level failed for {sub_id} L{level}: {e}")
        # 🟢 NEW: Return all PDCA keys with 0 for error case
        return {
            "score":0,
            "reason":f"LLM error: {e}",
            "is_passed":False,
            "P_Plan_Score": 0,
            "D_Do_Score": 0,
            "C_Check_Score": 0,
            "A_Act_Score": 0,
        }

# ------------------------
# Summarize
# ------------------------
def create_context_summary_llm(
    context: str, 
    sub_criteria_name: str, 
    level: int, 
    sub_id: str, 
    llm_executor: Any 
) -> Dict[str, Any]:
    """
    ใช้ LLM เพื่อสรุปเนื้อหา Context...
    """
    # 0. ตรวจสอบ llm_executor
    if llm_executor is None: 
        logger.error("LLM instance is None. Cannot summarize context.")
        return {"summary":"LLM not available","suggestion_for_next_level":"Check LLM"}

    # 0.1 ตรวจสอบ Context สั้นเกินไป
    context_limited = (context or "").strip()
    if not context_limited or len(context_limited) < 50:
        logger.info(f"Context too short for summarization L{level} {sub_id}. Skipping LLM call.")
        return {
            "summary": "หลักฐานที่ค้นหาได้มีข้อความสั้นเกินไปหรือไม่พบข้อความที่เกี่ยวข้อง",
            "suggestion_for_next_level": "ตรวจสอบแหล่งข้อมูลหรือปรับปรุงคำค้นหา RAG"
        }

    # 1. จำกัด Context ให้สั้นลงเพื่อความเร็วและความเสถียร (4000 tokens)
    context_to_send = context_limited[:4000]
    
    human_prompt = EVIDENCE_DESCRIPTION_PROMPT.format(
        sub_criteria_name=sub_criteria_name, 
        level=level, 
        context=context_to_send, 
        sub_id=sub_id
    )

    # 2. สร้าง System Prompt พร้อม JSON Schema
    try: 
        schema_json = json.dumps(EvidenceSummary.model_json_schema(), ensure_ascii=False, indent=2)
    except: 
        schema_json = '{"summary":"string", "suggestion_for_next_level":"string"}'

    system_prompt = SYSTEM_EVIDENCE_DESCRIPTION_PROMPT + "\n\n--- JSON SCHEMA ---\n" + schema_json + "\nIMPORTANT: Respond only with valid JSON."

    # 3. เรียกใช้ LLM พร้อม Retries
    try:
        raw = _fetch_llm_response(system_prompt, human_prompt, 2, llm_executor=llm_executor)
        
        # 4. แปลงผลลัพธ์ JSON
        parsed = _normalize_keys(_robust_extract_json(raw) or {})
        
        # 5. ตรวจสอบความถูกต้องของ Schema เบื้องต้น
        if not all(k in parsed for k in ["summary", "suggestion_for_next_level"]):
             logger.warning(f"LLM Summary: Missing expected keys in JSON. Raw: {raw[:100]}...")
             
        return parsed
        
    except Exception as e:
        logger.exception(f"create_context_summary_llm failed for {sub_id} L{level}: {e}")
        # Fallback กรณีเกิดข้อผิดพลาด
        return {"summary":f"LLM Error during summarization: {e.__class__.__name__}","suggestion_for_next_level": "Manual review required due to LLM failure."}

# ------------------------
# Action plan
# ------------------------
def create_structured_action_plan(failed_statements_data: List[Dict[str,Any]], sub_id:str, enabler:str, target_level:int, llm_executor: Any, max_retries:int=5) -> List[Dict[str,Any]]:
    """
    ใช้ LLM เพื่อสร้าง Action Plan ที่มีโครงสร้างจากรายการ Statement ที่สอบตก
    """
    if not failed_statements_data:
        return []

    if llm_executor is None:
        logger.error("LLM instance is None. Cannot create action plan.")
        return []

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

    # 📌 แก้ไข ACTION_PLAN_PROMPT ให้รองรับ sub_id และ target_level
    human_prompt = ACTION_PLAN_PROMPT.format(sub_id=sub_id, target_level=target_level, failed_statements_list="\n\n".join(statements_text))

    for attempt in range(max_retries+1):
        try:
            # 🟢 FIX: ส่ง llm_executor เข้าไป
            raw = _fetch_llm_response(system_prompt, human_prompt, 1, llm_executor=llm_executor)

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