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
from typing import List, Dict, Any, Optional, TypeVar, Final, Union, Callable
from pydantic import BaseModel, ConfigDict, Field, RootModel 
import uuid 
import sys 
import hashlib
from datetime import datetime
import textwrap

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
    from config.global_vars import (
        DEFAULT_ENABLER, 
        FINAL_K_RERANKED, 
        INITIAL_TOP_K,
        MAX_EVAL_CONTEXT_LENGTH 
    )

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

# -----------------------
# retrieve_context_with_filter
# -----------------------
def retrieve_context_with_filter(
    query: Union[str, List[str]], 
    doc_type: str, 
    enabler: Optional[str]=None,
    vectorstore_manager: Optional['VectorStoreManager']=None,
    top_k: int = None,
    initial_k: int = None,
    mapped_uuids: Optional[List[str]]=None,
    stable_doc_ids: Optional[List[str]]=None,
    priority_docs_input: Optional[List[Any]] = None,
    sequential_chunk_uuids: Optional[List[str]] = None, 
    sub_id: Optional[str]=None, 
    level: Optional[int]=None,
    get_previous_level_docs: Optional[Callable[[int, str], List[Any]]] = None,  # L3 Fallback
    logger: logging.Logger = logging.getLogger(__name__) 
) -> Dict[str, Any]:
    """
    L3-ready retrieval + fallback context + guaranteed-priority-chunks + rerank.
    """
    start_time = time.time()
    try:
        from core.constants import FINAL_K_RERANKED as CONST_FINAL_K, INITIAL_TOP_K as CONST_INITIAL_K
    except Exception:
        CONST_FINAL_K, CONST_INITIAL_K = 5, 25
    if top_k is None: top_k = CONST_FINAL_K
    if initial_k is None: initial_k = CONST_INITIAL_K

    all_retrieved_chunks: List[Any] = []
    used_chunk_uuids: List[str] = []

    # merge sequential_chunk_uuids into mapped_uuids
    if sequential_chunk_uuids:
        mapped_uuids = (list(mapped_uuids) if mapped_uuids else []) + list(sequential_chunk_uuids)

    manager = vectorstore_manager
    if manager is None:
        raise ValueError("VectorStoreManager is not initialized.")

    collection_name = _get_collection_name(doc_type, enabler).lower()
    queries_to_run = [query] if isinstance(query, str) else list(query or [])

    # --- L3: Fallback from previous level ---
    fallback_chunks: List[Any] = []
    if level == 3 and callable(get_previous_level_docs):
        try:
            fallback_chunks = get_previous_level_docs(level - 1, sub_id) or []
            logger.critical(f"🧭 DEBUG: Fallback context from previous level: {len(fallback_chunks)} chunks")
        except Exception as e:
            logger.warning(f"Fallback previous level docs failed: {e}")

    # --- Priority / mapped UUIDs ---
    guaranteed_priority_chunks: List[Any] = []
    mapped_uuids_for_vsm_search: Optional[List[str]] = None

    if priority_docs_input:
        try:
            from langchain_core.documents import Document as LcDocument
        except Exception:
            priority_docs_input = []
        else:
            transformed = []
            for doc in priority_docs_input:
                if doc is None: continue
                if isinstance(doc, dict):
                    pc = doc.get('page_content') or doc.get('text') or ''
                    meta = doc.get('metadata') or {}
                    if pc: transformed.append(LcDocument(page_content=pc, metadata=meta))
                elif isinstance(doc, LcDocument):
                    transformed.append(doc)
            guaranteed_priority_chunks = transformed
    elif mapped_uuids:
        mapped_uuids_for_vsm_search = normalize_stable_ids(mapped_uuids)
        logger.critical(f"🧭 DEBUG: Using {len(mapped_uuids_for_vsm_search)} UUIDs as search filter.")

    # --- Retriever ---
    retriever = manager.get_retriever(collection_name)
    if retriever is None:
        raise ValueError(f"Retriever init failed for collection '{collection_name}'")

    retrieved_chunks: List[Any] = []
    for q in queries_to_run:
        try:
            if callable(getattr(retriever, "invoke", None)):
                resp = retriever.invoke(q, config={"configurable": {"search_kwargs": {"k": initial_k}}})
            elif callable(getattr(retriever, "get_relevant_documents", None)):
                resp = retriever.get_relevant_documents(q)
            else:
                resp = []
        except Exception as e:
            logger.error(f"Retriever invocation error for query '{q}': {e}")
            resp = []
        retrieved_chunks.extend(resp or [])

    # merge fallback_chunks + retrieved + guaranteed_priority
    all_chunks_to_process = list(retrieved_chunks) + list(fallback_chunks) + list(guaranteed_priority_chunks)

    # --- Dedup + PDCA default + truncation ---
    unique_chunks_map: Dict[str, Any] = {}
    for doc in all_chunks_to_process:
        if doc is None: continue
        # ensure metadata dict
        md = getattr(doc, "metadata", {}) or {}
        setattr(doc, "metadata", md)
        # PDCA default
        if "pdca_tag" not in md or not md.get("pdca_tag"):
            md["pdca_tag"] = "Other"
        # truncated content for L3
        pc = (getattr(doc, "page_content", None) or getattr(doc, "text", "") or "")
        if level == 3:
            pc = pc[:500]
            setattr(doc, "page_content", pc)
        # chunk_uuid key
        chunk_uuid = md.get("chunk_uuid") or md.get("doc_uuid") or f"HASH-{hash(pc)}"
        if chunk_uuid not in unique_chunks_map:
            unique_chunks_map[chunk_uuid] = doc

    dedup_chunks = list(unique_chunks_map.values())
    logger.info(f"    - Dedup Merged: Total unique chunks = {len(dedup_chunks)}. Guaranteed chunks = {len(guaranteed_priority_chunks)}")

    # --- Rerank ---
    final_selected_docs: List[Any] = list(guaranteed_priority_chunks)
    slots_available = max(0, top_k - len(final_selected_docs))
    rerank_candidates = [d for d in dedup_chunks if d not in final_selected_docs]

    if slots_available > 0 and rerank_candidates:
        reranker = get_global_reranker(top_k)
        if reranker and hasattr(reranker, "compress_documents"):
            try:
                reranked = reranker.compress_documents(query=queries_to_run[0] if queries_to_run else "", documents=rerank_candidates, top_n=slots_available)
                final_selected_docs.extend(reranked or [])
            except Exception:
                final_selected_docs.extend(rerank_candidates[:slots_available])
        else:
            final_selected_docs.extend(rerank_candidates[:slots_available])

    # --- Prepare outputs ---
    top_evidences: List[Dict[str, Any]] = []
    aggregated_list: List[str] = []

    for doc in final_selected_docs[:top_k]:
        if doc is None: continue
        md = getattr(doc, "metadata", {}) or {}
        source = md.get("source") or md.get("filename") or "Unknown"
        pc = getattr(doc, "page_content", "") or ""
        chunk_uuid = md.get("chunk_uuid") or f"HASH-{hash(pc)}"
        used_chunk_uuids.append(chunk_uuid)
        top_evidences.append({
            "doc_uuid": md.get("doc_uuid"),
            "doc_id": md.get("doc_id") or md.get("stable_doc_uuid"),
            "chunk_uuid": chunk_uuid,
            "source": source,
            "text": pc,
            "pdca_tag": md.get("pdca_tag", "Other"),
            "score": md.get("relevance_score", 0.0)
        })
        aggregated_list.append(f"[{md.get('pdca_tag','Other')}] [SOURCE: {source}] {pc}")

    aggregated_context = "\n\n---\n\n".join(aggregated_list)
    duration = time.time() - start_time
    if level is not None:
        logger.critical(f"🧭 DEBUG: Aggregated Context Length for L{level} ({sub_id}) = {len(aggregated_context)}. Retrieval Time: {duration:.2f}s")

    return {
        "top_evidences": top_evidences,
        "aggregated_context": aggregated_context,
        "retrieval_time": duration,
        "used_chunk_uuids": used_chunk_uuids
    }


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

def _summarize_evidence_list_short(evidences: List[Dict[str, Any]], max_sentences: int = 3) -> str:
    """
    Create a concise human-readable 1-3 sentence summary of a list of evidence chunks.
    Use filename + key phrase from text (first 120 chars).
    """
    if not evidences:
        return ""
    parts = []
    for ev in evidences[:max(1, min(len(evidences), max_sentences))]:
        fn = ev.get("source_filename") or ev.get("source") or ev.get("doc_id", "unknown")
        txt = ev.get("text", "")[:120].replace("\n", " ").strip()
        if txt:
            parts.append(f"จากไฟล์ `{fn}`: {txt}...")
        else:
            parts.append(f"จากไฟล์ `{fn}`")
    return " | ".join(parts)

def build_multichannel_context_for_level(
    level: int,
    top_evidences: List[Dict[str, Any]],
    previous_levels_map: Dict[int, List[Dict[str, Any]]],
    max_main_context_tokens: int = 3000,
    max_summary_sentences: int = 3
) -> Dict[str, Any]:
    """
    Build structured 'channels' of context for LLM:
      - baseline_L1_L2: short summary of passed evidence from L1/L2 (separate by level)
      - direct_Ln_evidence: evidence retrieved specifically for this level (ranked)
      - aux_other: other evidence (low-priority) but truncated

    Args:
        level: current evaluating level (int)
        top_evidences: list of evidence dicts retrieved for this level (already reranked)
        previous_levels_map: mapping {level: [evidence_dicts]} from passed levels (1..level-1)
    Returns:
        dict with keys:
            - baseline_summary: short text (L1/L2)
            - direct_context: aggregated text for main evaluation (truncated to max_main_context_tokens chars)
            - aux_summary: optional other evidences summary
            - debug_meta: meta info (counts, filenames)
    """
    # 1) Baseline summary: gather L1 and L2 passed evidences only (preserve by-level order)
    baseline_evidence = []
    for lev in sorted(previous_levels_map.keys()):
        if lev <= 2:
            baseline_evidence.extend(previous_levels_map.get(lev, []))

    baseline_summary = _summarize_evidence_list_short(baseline_evidence, max_sentences=max_summary_sentences)

    # 2) Direct evidence: prefer evidences that have PDCA tags relevant to this level
    # Expect top_evidences already contains pdca_tag or similar; fallback: use first K
    direct = []
    aux = []
    K_MAIN = 5  # how many chunks to include prominently
    for ev in top_evidences:
        tag = ev.get("pdca_tag", "").upper()
        # if this level >=3, prefer C/A blocks for main context
        if level >= 3 and tag in ("C", "A"):
            direct.append(ev)
        else:
            aux.append(ev)

    # fill direct up to K_MAIN with remaining if needed
    if len(direct) < K_MAIN:
        # take from aux to fill
        take = K_MAIN - len(direct)
        direct.extend(aux[:take])
        aux = aux[take:]

    # 3) Build textual aggregated direct_context (ordered)
    def _join_chunks(chunks: List[Dict[str, Any]], max_chars: int):
        out = []
        used = 0
        for c in chunks:
            txt = c.get("text") or c.get("content") or ""
            if not txt: continue
            piece = txt.strip()
            # simple truncation to avoid huge contexts
            if used + len(piece) > max_chars:
                remaining = max_chars - used
                if remaining <= 0:
                    break
                out.append(piece[:remaining] + "...")
                used += remaining
                break
            out.append(piece)
            used += len(piece)
        return "\n\n".join(out)

    direct_context = _join_chunks(direct, max_main_context_tokens)
    aux_summary = _summarize_evidence_list_short(aux, max_sentences=3)

    debug_meta = {
        "direct_count": len(direct),
        "aux_count": len(aux),
        "baseline_count": len(baseline_evidence),
        "direct_files": [d.get("source_filename") or d.get("source") or d.get("doc_id") for d in direct][:10],
        "aux_files": [d.get("source_filename") or d.get("source") or d.get("doc_id") for d in aux][:10]
    }

    return {
        "baseline_summary": baseline_summary,
        "direct_context": direct_context,
        "aux_summary": aux_summary,
        "debug_meta": debug_meta
    }


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

def _safe_int_parse(value: Any, default: int = 0) -> int:
    """Safely converts value to an integer."""
    if value is None:
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


# ------------------------------------------------------------
# Balanced Extractor (ป้องกัน nested + fenced)
# ------------------------------------------------------------
def _extract_balanced_braces(text: str) -> Optional[str]:
    if not text:
        return None

    # ตัด scanning หลังจากเจอ ``` (กันการจับ JSON ผิดชุด)
    fence_pos = text.find("```")
    scan_text = text if fence_pos == -1 else text[:fence_pos]

    start = scan_text.find('{')
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(scan_text)):
        if scan_text[i] == '{':
            depth += 1
        elif scan_text[i] == '}':
            depth -= 1
            if depth == 0:
                return scan_text[start:i+1]

    return None


def _robust_extract_json(llm_response: str) -> Dict[str, Any]:
    """
    Assessment-specific JSON extraction. Handles P/D/C/A key completion and score calculation.
    """
    # 📌 1. กำหนด Fallback Dict ที่สมบูรณ์
    # ใช้ค่า Fallback นี้หากเกิดข้อผิดพลาดร้ายแรงหรือการดึง JSON ล้มเหลว
    fallback = {
        "score": 0,
        "reason": "LLM Fatal Error in JSON extraction.",
        "is_passed": False,
        "P_Plan_Score": 0,
        "D_Do_Score": 0,
        "C_Check_Score": 0,
        "A_Act_Score": 0
    }

    # Step 1: Extract JSON และ Normalize key ด้วย Helper ตัวใหม่
    data = _extract_normalized_dict(llm_response)
    
    # หากดึง JSON ไม่ได้เลย ให้คืนค่า Fallback ทันที
    if not data:
        return fallback 
        # เราจะไม่ใช้ data = {} เหมือนที่คุณเสนอ เพราะมันขาด keys สำคัญ

    # Step 2: Logic สำหรับ Assessment (Clean and Complete keys)
    final = {}

    # 2.1 P/D/C/A Scores (ต้องเป็น int)
    final["P_Plan_Score"] = _safe_int_parse(data.get("P_Plan_Score"))
    final["D_Do_Score"]   = _safe_int_parse(data.get("D_Do_Score"))
    final["C_Check_Score"] = _safe_int_parse(data.get("C_Check_Score"))
    final["A_Act_Score"]   = _safe_int_parse(data.get("A_Act_Score")) # ใช้ A_Act_Score ตาม schema เดิม

    # 2.2 Reason และ is_passed
    final["reason"] = str(data.get("reason")) if data.get("reason") else "Fallback: Missing reason."
    isp = data.get("is_passed")
    final["is_passed"] = (isinstance(isp, str) and isp.lower() == "true") or bool(isp)

    # 2.3 Total Score Logic (สำคัญ: มี Fallback การรวม P+D+C+A)
    llm_score = data.get("score")
    
    if llm_score is None:
        # Fallback: หาก LLM ไม่ให้ Total Score มา ให้รวมจาก P+D+C+A
        final["score"] = (
            final["P_Plan_Score"]
            + final["D_Do_Score"]
            + final["C_Check_Score"]
            + final["A_Act_Score"]
        )
    else:
        # ใช้ score ที่ LLM ให้มา (ต้องเป็น int ตามมาตรฐาน Assessment)
        final["score"] = _safe_int_parse(llm_score)
        
    return final


def _extract_normalized_dict(llm_response: str) -> Optional[Dict[str, Any]]:
    """
    Performs robust JSON extraction and key normalization ONLY. 
    It does not perform assessment-specific key completion or scoring logic.
    """
    raw = (llm_response or "").strip()
    if not raw:
        return None

    # 1) Fenced JSON
    fence_regex = r'```(?:json|JSON)?\s*(\{.*?})\s*```'
    fenced = re.search(fence_regex, raw, flags=re.DOTALL)
    if fenced:
        json_str = fenced.group(1)
    else:
        # 2) Balanced JSON scan
        json_str = _extract_balanced_braces(raw)
        if json_str is None:
            return None

    # 3) JSON Decode (JSON → JSON5 fallback)
    try:
        data = json.loads(json_str)
    except Exception:
        try:
            import json5
            data = json5.loads(json_str)
        except:
            return None # Failed both json and json5

    if not isinstance(data, dict):
        return None

    # 4) Normalize keys
    return _normalize_keys(data)

def _normalize_keys(data: Any) -> Any:
    mapping = {
        "llm_score": "score",
        "reasoning": "reason",
        "llm_reasoning": "reason",
        "assessment_reason": "reason",
        "comment": "reason",
        "pass": "is_passed",
        "is_pass": "is_passed",

        "p_score": "P_Plan_Score",
        "d_score": "D_Do_Score",
        "c_score": "C_Check_Score",
        "a_score": "A_Act_Score",

        "p_plan": "P_Plan_Score",
        "d_do": "D_Do_Score",
        "c_check": "C_Check_Score",
        "a_act": "A_Act_Score",
    }

    if isinstance(data, dict):
        return {mapping.get(k.lower(), k): _normalize_keys(v) for k, v in data.items()}

    if isinstance(data, list):
        return [_normalize_keys(x) for x in data]

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
    """
    Returns Failure result if context is empty or contains known error strings.
    Auto-fail with PDCA keys all set to 0.
    """
    if not context or "ไม่มีหลักฐานที่เกี่ยวข้อง" in context or "ERROR:" in context.upper():
        logger.warning(f"Auto-FAIL L{level} for {sub_id}: Empty or Error Context detected from RAG.")
        context_preview = context.strip()[:100].replace("\n", " ") if context else "Empty Context"
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


def evaluate_with_llm(context: str, sub_criteria_name: str, level: int, statement_text: str, sub_id: str, check_evidence: str = "", act_evidence: str = "", llm_executor: Any = None, **kwargs) -> Dict[str, Any]:
    """Standard Evaluation for L3+ with robust handling for missing keys."""
    
    context_to_send_eval = context[:MAX_EVAL_CONTEXT_LENGTH] if context else ""
    # 1. ตรวจสอบ Context ก่อนส่ง LLM
    failure_result = _check_and_handle_empty_context(context, sub_id, level)
    if failure_result:
        return failure_result

    contextual_rules_prompt = kwargs.get("contextual_rules_prompt", "")
    # inside evaluate_with_llm before formatting
    baseline_summary = kwargs.get("baseline_summary", "")
    aux_summary = kwargs.get("aux_summary", "")
    
    # 2. Prepare User & System Prompts
    user_prompt = USER_ASSESSMENT_PROMPT.format(
        sub_criteria_name=sub_criteria_name, 
        level=level, 
        statement_text=statement_text, 
        sub_id=sub_id,
        context=context_to_send_eval or "ไม่มีหลักฐานที่เกี่ยวข้อง",
        pdca_phase=kwargs.get("pdca_phase",""), 
        level_constraint=kwargs.get("level_constraint",""),
        contextual_rules_prompt=contextual_rules_prompt,
        check_evidence=check_evidence, 
        act_evidence=act_evidence, 
    )

    # Insert baseline_summary into the prompt explicitly:
    if baseline_summary:
        user_prompt = user_prompt + "\n\n--- Baseline summary (จาก L1-L2): ---\n" + baseline_summary

    if aux_summary:
        user_prompt = user_prompt + "\n\n--- Auxiliary evidence summary (low-priority): ---\n" + aux_summary

    try:
        schema_json = json.dumps(CombinedAssessment.model_json_schema(), ensure_ascii=False, indent=2)
    except:
        schema_json = '{"score":0,"reason":"string"}'

    system_prompt = SYSTEM_ASSESSMENT_PROMPT + "\n\n--- JSON SCHEMA ---\n" + schema_json + "\nIMPORTANT: Respond only with valid JSON."

    try:
        # 3. เรียก LLM
        raw = _fetch_llm_response(system_prompt, user_prompt, _MAX_LLM_RETRIES, llm_executor=llm_executor)
        
        # 4. Extract JSON และ normalize keys
        parsed = _normalize_keys(_robust_extract_json(raw) or {})

        # 5. คืนผลลัพธ์, เติม default หาก key ขาด
        return {
            "score": int(parsed.get("score", 0)),
            "reason": parsed.get("reason", "No reason provided by LLM."),
            "is_passed": parsed.get("is_passed", False),
            "P_Plan_Score": int(parsed.get("P_Plan_Score", 0)),
            "D_Do_Score": int(parsed.get("D_Do_Score", 0)),
            "C_Check_Score": int(parsed.get("C_Check_Score", 0)),
            "A_Act_Score": int(parsed.get("A_Act_Score", 0)),
        }

    except Exception as e:
        logger.exception(f"evaluate_with_llm failed for {sub_id} L{level}: {e}")
        return {
            "score":0,
            "reason":f"LLM error: {e}",
            "is_passed":False,
            "P_Plan_Score": 0,
            "D_Do_Score": 0,
            "C_Check_Score": 0,
            "A_Act_Score": 0,
        }


# =========================
# Patch for L1-L2 evaluation
# =========================

# 1️⃣ เพิ่ม context limit สำหรับ L1/L2
def _get_context_for_level(context: str, level: int) -> str:
    """Return context string with appropriate length limit for each level."""
    if not context:
        return ""
    if level <= 2:
        return context[:6000]  # L1-L2 ใช้ context ยาวขึ้น
    return context[:MAX_EVAL_CONTEXT_LENGTH]  # L3-L5

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

# 3️⃣ ปรับ evaluate_with_llm_low_level ให้เรียกฟังก์ชันใหม่
def evaluate_with_llm_low_level(context: str, sub_criteria_name: str, level: int, statement_text: str, sub_id: str, llm_executor: Any, **kwargs) -> Dict[str, Any]:
    """
    Evaluation สำหรับ L1/L2 แบบ robust และ schema uniform
    """

    failure_result = _check_and_handle_empty_context(context, sub_id, level)
    if failure_result:
        return failure_result

    level_constraint = kwargs.get("level_constraint", "")
    contextual_rules_prompt = kwargs.get("contextual_rules_prompt", "") 

    # จำกัด context ตาม level
    context_to_send = _get_context_for_level(context, level)

    user_prompt = USER_LOW_LEVEL_PROMPT.format(
        sub_criteria_name=sub_criteria_name,
        level=level,
        statement_text=statement_text,
        sub_id=sub_id,
        context=context_to_send,
        level_constraint=level_constraint,
        contextual_rules_prompt=contextual_rules_prompt
    )

    try:
        schema_json = json.dumps(CombinedAssessment.model_json_schema(), ensure_ascii=False, indent=2)
    except:
        schema_json = '{"score":0,"reason":"string"}'

    system_prompt = SYSTEM_LOW_LEVEL_PROMPT + "\n\n--- JSON SCHEMA ---\n" + schema_json + "\nIMPORTANT: Respond only with valid JSON."

    try:
        raw = _fetch_llm_response(system_prompt, user_prompt, _MAX_LLM_RETRIES, llm_executor=llm_executor)
        parsed = _normalize_keys(_robust_extract_json(raw) or {})

        # ใช้ extraction สำหรับ L1/L2
        return _extract_combined_assessment_low_level(parsed)

    except Exception as e:
        logger.exception(f"evaluate_with_llm_low_level failed for {sub_id} L{level}: {e}")
        return {
            "score":0,
            "reason":f"LLM error: {e}",
            "is_passed":False,
            "P_Plan_Score": 0,
            "D_Do_Score": 0,
            "C_Check_Score": 0,
            "A_Act_Score": 0,
        }

def _extract_combined_assessment_low_level(parsed: dict) -> dict:
    """
    Safely extracts combined assessment results for L1/L2, 
    ensuring all keys exist AND enforcing the C=0, A=0 rule.
    """
    # 1. สร้าง Dictionary ผลลัพธ์โดยดึงค่าจาก LLM และใส่ค่าเริ่มต้น (Default)
    # ใช้ .get() เพื่อให้โค้ดรันได้แม้ Key จะหายไป
    result = {
        "score": int(parsed.get("score", 0)),
        "reason": parsed.get("reason", "No reason provided by LLM (Low Level)."),
        "is_passed": parsed.get("is_passed", False),
        "P_Plan_Score": int(parsed.get("P_Plan_Score", 0)),
        "D_Do_Score": int(parsed.get("D_Do_Score", 0)),
        # ดึงค่า C/A มาก่อน เผื่อใช้ Debug แต่...
        "C_Check_Score": int(parsed.get("C_Check_Score", 0)),
        "A_Act_Score": int(parsed.get("A_Act_Score", 0)),
    }
    
    # 2. ENFORCE L1/L2 HARD RULE: C and A must be 0
    # บังคับค่าให้เป็น 0 เสมอ เพื่อป้องกันการ Over-scoring
    result["C_Check_Score"] = 0
    result["A_Act_Score"] = 0
    
    # 3. Final check for is_passed default logic
    # หาก LLM ไม่ได้ให้ is_passed มา ให้ใช้ score >= 1 เป็นเกณฑ์ (ตามที่ทำใน _extract_combined_assessment)
    if result["is_passed"] == False and result["score"] >= 1:
         result["is_passed"] = True

    return result

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

    # system_prompt = SYSTEM_EVIDENCE_DESCRIPTION_PROMPT + "\n\n--- JSON SCHEMA ---\n" + schema_json + "\nIMPORTANT: Respond only with valid JSON."
    system_prompt = (
        SYSTEM_EVIDENCE_DESCRIPTION_PROMPT
        + "\n\n--- JSON SCHEMA ---\n"
        + schema_json
        + "\nIMPORTANT: Respond only with valid JSON. เนื้อหาในทุก key ต้องเป็นภาษาไทยเท่านั้น ห้ามใช้ภาษาอังกฤษ."
    )


    # 3. เรียกใช้ LLM พร้อม Retries
    try:
        raw = _fetch_llm_response(system_prompt, human_prompt, 2, llm_executor=llm_executor)
        
        # 4. แปลงผลลัพธ์ JSON
        parsed = _extract_normalized_dict(raw) or {}
        parsed.setdefault("summary", "Fallback: No summary provided by LLM.")
        parsed.setdefault("suggestion_for_next_level", "Fallback: No suggestion provided.")
        
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

def create_structured_action_plan(
    failed_statements: List[Dict[str, Any]], 
    sub_id: str, 
    target_level: int, 
    llm_executor: Any, # <-- เพิ่มอาร์กิวเมนต์นี้
    max_retries: int = 3
) -> List[Dict[str, Any]]:
    """
    ใช้ LLM เพื่อสร้าง Action Plan ที่มีโครงสร้างจากรายการ Statement ที่สอบตก
    """
    # 📌 FIX: เปลี่ยน failed_statements_data เป็น failed_statements
    if not failed_statements: 
        return []

    if llm_executor is None:
        logger.error("LLM instance is None. Cannot create action plan.")
        return []

    try:
        # ใช้ .model_json_schema() เพื่อความเข้ากันได้กับ Pydantic v2+
        # NOTE: ActionPlanActions ต้องเป็น Pydantic Model
        schema_json = json.dumps(ActionPlanActions.model_json_schema(), ensure_ascii=False, indent=2)
    except Exception:
        schema_json = "{}"

    system_prompt = SYSTEM_ACTION_PLAN_PROMPT + "\n\n--- JSON SCHEMA ---\n" + schema_json + "\nRespond ONLY with a valid JSON ARRAY."

    statements_text = []
    # 📌 FIX: เปลี่ยน failed_statements_data เป็น failed_statements
    for s in failed_statements: 
        # จำกัดความยาวของ Statement และ Reason เพื่อป้องกันการล้น Token
        st = (s.get('statement','') or '')[:1000]
        rs = (s.get('reason','') or '')[:500]
        statements_text.append(f"Level:{s.get('level','N/A')}\nStatement:{st}\nReason:{rs}")

    # 📌 แก้ไข ACTION_PLAN_PROMPT ให้รองรับ sub_id และ target_level
    human_prompt = ACTION_PLAN_PROMPT.format(sub_id=sub_id, target_level=target_level, failed_statements_list="\n\n".join(statements_text))

    for attempt in range(max_retries + 1):
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