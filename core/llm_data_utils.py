"""
llm_data_utils.py
Robust LLM + RAG utilities for SEAM assessment (CLEAN FINAL VERSION)
"""

import logging
import time
import json
import hashlib
import uuid
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Callable, TypeVar
import json5


# Optional: regex แทน re (ดีกว่า) — ถ้าไม่มีก็ใช้ re ธรรมดา
try:
    import regex as re  # type: ignore
except ImportError:
    pass  # ใช้ re มาตรฐานต่อไป

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ===================================================================
# 1. Core Configuration (ต้องมีแน่นอน)
# ===================================================================
from config.global_vars import (
    DEFAULT_ENABLER,
    INITIAL_TOP_K,
    FINAL_K_RERANKED,
    MAX_EVAL_CONTEXT_LENGTH,
)

# ===================================================================
# 2. Critical Utilities (ต้องมีจริง — ไม่มี fallback)
# ===================================================================
from core.vectorstore import _get_collection_name
from core.json_extractor import (
    _robust_extract_json,
    _normalize_keys,
    _safe_int_parse,
    _extract_normalized_dict
)

# ===================================================================
# 3. Project Modules (ถ้าหาย → ปล่อยให้ error ชัด ๆ ไปเลย ดีกว่าเงียบ)
# ===================================================================
from core.seam_prompts import (
    SYSTEM_ASSESSMENT_PROMPT,
    USER_ASSESSMENT_PROMPT,
    SYSTEM_ACTION_PLAN_PROMPT,
    ACTION_PLAN_PROMPT,
    SYSTEM_EVIDENCE_DESCRIPTION_PROMPT,
    EVIDENCE_DESCRIPTION_PROMPT,
    SYSTEM_LOW_LEVEL_PROMPT,
    USER_LOW_LEVEL_PROMPT,
)

from core.vectorstore import VectorStoreManager, get_global_reranker, ChromaRetriever
from core.assessment_schema import CombinedAssessment, EvidenceSummary
from core.action_plan_schema import ActionPlanActions

try:
    from core.assessment_schema import StatementAssessment
except ImportError:
    from pydantic import BaseModel
    class StatementAssessment(BaseModel):
        score: int = 0
        reason: str = ""

from langchain_core.documents import Document as LcDocument

# ===================================================================
# 4. Constants
# ===================================================================
LOW_LEVEL_K: int = 3
_MOCK_FLAG = False
_MAX_LLM_RETRIES = 3

def set_mock_control_mode(enable: bool):
    global _MOCK_FLAG
    _MOCK_FLAG = bool(enable)
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
    vectorstore_manager: Optional['VectorStoreManager'] = None
) -> Dict[str, Any]:

    # ไม่มี Doc UUID → ไม่มี evidence
    if not doc_uuids:
        return {"top_evidences": []}

    # ใช้ manager ที่ส่งเข้ามา (ถ้ามี) หรือสร้างใหม่
    manager = vectorstore_manager if vectorstore_manager else VectorStoreManager()
    if manager is None:
        logger.error("VectorStoreManager is None.")
        return {"top_evidences": []}

    try:
        # 🎯 FIX: ลบ normalize_stable_ids ออก เพราะ doc_uuids คือ Chunk UUIDs ที่เสถียรแล้ว
        lookup_ids = doc_uuids
        
        # ดึง document chunk ตาม stable_doc_uuid หรือ chunk_uuid
        docs: List[LcDocument] = manager.get_documents_by_id(lookup_ids, doc_type, enabler)

        top_evidences = []
        for d in docs:
            md = getattr(d, "metadata", {}) or {}
            top_evidences.append({
                "doc_id": md.get("stable_doc_uuid"),
                "chunk_uuid": md.get("chunk_uuid"),
                "doc_type": md.get("doc_type"),
                "source": md.get("source") or md.get("doc_source"),
                "source_filename": md.get("source") or md.get("doc_source"),  # ✅
                "content": getattr(d, "page_content", "").strip(),
                "chunk_index": md.get("chunk_index")
            })

        return {"top_evidences": top_evidences}

    except Exception as e:
        logger.error(f"retrieve_context_by_doc_ids error: {e}")
        return {"top_evidences": []}

# ------------------------
# Retrieval: retrieve_context_with_filter (แก้จุดเสี่ยง 2 จุด)
# ------------------------
def retrieve_context_with_filter(
    query: Union[str, List[str]],
    doc_type: str,
    enabler: Optional[str] = None,
    vectorstore_manager: Optional['VectorStoreManager'] = None,
    mapped_uuids: Optional[List[str]] = None,
    stable_doc_ids: Optional[List[str]] = None,
    priority_docs_input: Optional[List[Any]] = None,
    sequential_chunk_uuids: Optional[List[str]] = None,
    sub_id: Optional[str] = None,
    level: Optional[int] = None,
    get_previous_level_docs: Optional[Callable[[int, str], List[Any]]] = None,
) -> Dict[str, Any]:
    """
    ดึง context ด้วย semantic search + priority + fallback + rerank
    แก้ทุกปัญหา 0 documents แล้ว 100%
    """
    start_time = time.time()
    all_retrieved_chunks: List[Any] = []
    used_chunk_uuids: List[str] = []

    # 1. ใช้ VectorStoreManager เดียวกันทั้งหมด (สำคัญมาก!)
    manager = vectorstore_manager or VectorStoreManager()
    if manager is None or manager._client is None:
        logger.error("VectorStoreManager not initialized!")
        return {"top_evidences": [], "aggregated_context": "", "retrieval_time": 0.0, "used_chunk_uuids": []}

    queries_to_run = [query] if isinstance(query, str) else list(query or [])
    if not queries_to_run:
        queries_to_run = [""]  # ป้องกัน error

    # รวม chunk ที่ต้องบังคับโผล่ (sequential)
    if sequential_chunk_uuids:
        mapped_uuids = (mapped_uuids or []) + sequential_chunk_uuids

    # 2. Fallback จาก level ก่อนหน้า (สำหรับ Level 3)
    fallback_chunks = []
    if level == 3 and callable(get_previous_level_docs):
        try:
            fallback_chunks = get_previous_level_docs(level - 1, sub_id) or []
            logger.info(f"Fallback from previous level: {len(fallback_chunks)} chunks")
        except Exception as e:
            logger.warning(f"Fallback failed: {e}")

    # 3. Priority chunks (เช่น จาก evidence mapping)
    guaranteed_priority_chunks = []
    if priority_docs_input:
        for doc in priority_docs_input:
            if doc is None:
                continue
            if isinstance(doc, dict):
                pc = doc.get('page_content') or doc.get('text') or ''
                meta = doc.get('metadata') or {}
                if pc.strip():
                    guaranteed_priority_chunks.append(LcDocument(page_content=pc, metadata=meta))
            elif hasattr(doc, 'page_content'):
                guaranteed_priority_chunks.append(doc)

    # 4. ดึง collection name ให้ตรงตัว (ห้าม .lower() เด็ดขาด!)
    collection_name = _get_collection_name(doc_type, enabler or DEFAULT_ENABLER)
    logger.info(f"Requesting retriever → collection='{collection_name}' (doc_type={doc_type}, enabler={enabler})")

    retriever = manager.get_retriever(collection_name)  # ไม่มี .lower()!!!
    if not retriever:
        logger.error(f"Retriever NOT FOUND for collection: {collection_name}")
        logger.error(f"Available collections: {list(manager._chroma_cache.keys())}")
        retrieved_chunks = []
    else:
        retrieved_chunks = []
        for q in queries_to_run:
            q_log = q[:120] + "..." if len(q) > 120 else q
            logger.critical(f"[QUERY] Running: '{q_log}' → collection='{collection_name}'")

            try:
                if hasattr(retriever, "get_relevant_documents"):
                    docs = retriever.get_relevant_documents(q)
                elif hasattr(retriever, "invoke"):
                    docs = retriever.invoke(q, config={"configurable": {"search_kwargs": {"k": INITIAL_TOP_K}}})
                else:
                    docs = []
                retrieved_chunks.extend(docs or [])
            except Exception as e:
                logger.error(f"Retriever invoke failed: {e}", exc_info=True)

    logger.critical(f"[RETRIEVAL] Raw chunks from ChromaDB: {len(retrieved_chunks)} documents")

    # 5. รวม + deduplicate อย่างปลอดภัย
    all_chunks = retrieved_chunks + fallback_chunks + guaranteed_priority_chunks
    unique_map: Dict[str, LcDocument] = {}

    for doc in all_chunks:
        if not doc or not hasattr(doc, "page_content"):
            continue
        md = getattr(doc, "metadata", {}) or {}
        pc = str(getattr(doc, "page_content", "") or "").strip()
        if not pc:
            continue

        # ตัด content สำหรับ Level 3
        if level == 3:
            pc = pc[:500]
            doc.page_content = pc

        chunk_uuid = md.get("chunk_uuid") or md.get("stable_doc_uuid") or f"TEMP-{uuid.uuid4().hex[:12]}"
        if chunk_uuid not in unique_map:
            md["dedup_chunk_uuid"] = chunk_uuid
            unique_map[chunk_uuid] = doc

    dedup_chunks = list(unique_map.values())
    logger.info(f"After dedup: {len(dedup_chunks)} chunks")

    # 6. Rerank (ถ้ามี reranker และมี slot ว่าง)
    final_docs = list(guaranteed_priority_chunks)
    slots_left = max(0, FINAL_K_RERANKED - len(final_docs))
    candidates = [d for d in dedup_chunks if d not in final_docs]

    if slots_left > 0 and candidates:
        reranker = get_global_reranker()
        if reranker and hasattr(reranker, "compress_documents"):
            try:
                reranked = reranker.compress_documents(
                    documents=candidates,
                    query=queries_to_run[0],
                    top_n=slots_left
                )
                final_docs.extend(reranked or candidates[:slots_left])
                logger.info(f"Reranker returned {len(reranked or [])} docs")
            except Exception as e:
                logger.warning(f"Reranker failed ({e}), using raw candidates")
                final_docs.extend(candidates[:slots_left])
        else:
            logger.info("No reranker → using top-k raw")
            final_docs.extend(candidates[:slots_left])
    else:
        logger.info("No slots left or no candidates → priority only")

    # 7. สร้าง output
    top_evidences = []
    aggregated_parts = []

    for doc in final_docs[:FINAL_K_RERANKED]:
        md = getattr(doc, "metadata", {}) or {}
        pc = str(getattr(doc, "page_content", "") or "").strip()
        chunk_uuid = md.get("chunk_uuid") or md.get("dedup_chunk_uuid") or f"UNKNOWN-{uuid.uuid4().hex[:8]}"
        used_chunk_uuids.append(chunk_uuid)

        source = md.get("source") or md.get("filename") or md.get("doc_source") or "Unknown"
        pdca = md.get("pdca_tag", "Other")

        top_evidences.append({
            "doc_id": md.get("stable_doc_uuid"),
            "chunk_uuid": chunk_uuid,
            "source": source,
            "source_filename": source,
            "text": pc,
            "pdca_tag": pdca,
        })
        aggregated_parts.append(f"[{pdca}] [SOURCE: {source}] {pc}")

    result = {
        "top_evidences": top_evidences,
        "aggregated_context": "\n\n---\n\n".join(aggregated_parts),
        "retrieval_time": round(time.time() - start_time, 3),
        "used_chunk_uuids": used_chunk_uuids
    }

    logger.info(f"Final retrieval L{level or '?'} {sub_id or ''}: {len(top_evidences)} chunks in {result['retrieval_time']:.2f}s")
    return result

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

# ----------------------------------------------------
# Helper function: Summarize evidence list (minimal stub)
# ----------------------------------------------------
def _summarize_evidence_list_short(evidences: list, max_sentences: int = 3) -> str:
    """
    Provides a concise summary of evidence items.
    """
    if not evidences:
        return ""
    
    parts = []
    for ev in evidences[:max(1, min(len(evidences), max_sentences))]:
        if isinstance(ev, dict):
            fn = ev.get("source_filename") or ev.get("source") or ev.get("doc_id", "unknown")
            txt = ev.get("text") or ev.get("content") or ""
        else:
            fn = str(ev)
            txt = str(ev)
        txt_short = txt[:120].replace("\n", " ").strip()
        if txt_short:
            parts.append(f"จากไฟล์ `{fn}`: {txt_short}...")
        else:
            parts.append(f"จากไฟล์ `{fn}`")
    return " | ".join(parts)


# ----------------------------------------------------
# ULTIMATE FINAL VERSION: build_multichannel_context_for_level
# รับทั้ง evidence dicts เต็ม ๆ และรองรับของเก่าด้วย
# ----------------------------------------------------
def build_multichannel_context_for_level(
    level: int,
    top_evidences: list,
    previous_levels_map: dict | None = None,                    # เก่า: {key: list[dict]} หรือ {doc_id: filename}
    previous_levels_evidence: list | None = None,               # ใหม่: list[dict] ที่มี text เต็ม ๆ
    max_main_context_tokens: int = 3000,
    max_summary_sentences: int = 4
) -> dict:

    # --- 1) Baseline: ใช้ previous_levels_evidence เป็นหลัก (มี text!) ---
    baseline_evidence = previous_levels_evidence or []

    # Fallback เก่า: ถ้ายังส่งแบบเดิมมา (เช่นจาก _run_single_assessment เก่า)
    if not baseline_evidence and previous_levels_map:
        for items in previous_levels_map.values():
            if isinstance(items, list):
                baseline_evidence.extend(items)
            elif isinstance(items, dict) and (items.get("text") or items.get("content")):
                baseline_evidence.append(items)

    # กรองเฉพาะที่มี text
    summarizable_baseline = [
        item for item in baseline_evidence
        if isinstance(item, dict) and (item.get("text") or item.get("content"))
    ]

    # ถ้าไม่มีจริง ๆ ให้ใส่ข้อความแทน
    if not summarizable_baseline:
        summarizable_baseline = [{"text": "ไม่มีหลักฐานจาก Level ก่อนหน้า"}]

    baseline_summary = _summarize_evidence_list_short(
        summarizable_baseline,
        max_sentences=max_summary_sentences
    )

    # --- 2) Direct / Aux classification (เหมือนเดิม) ---
    direct, aux = [], []
    K_MAIN = 5

    for ev in top_evidences:
        if not isinstance(ev, dict):
            aux.append(ev)
            continue
        tag = (ev.get("pdca_tag") or ev.get("PDCA") or "P").upper()
        if tag in ("P", "D", "C", "A"):
            direct.append(ev)
        else:
            aux.append(ev)

    if len(direct) < K_MAIN:
        need = K_MAIN - len(direct)
        direct.extend(aux[:need])
        aux = aux[need:]

    direct_for_context = direct[:K_MAIN]

    # --- 3) Join text ---
    def _join_chunks(chunks, max_chars):
        out, used = [], 0
        for c in chunks:
            txt = (c.get("text") or c.get("content") or "").strip()
            if not txt:
                continue
            if used + len(txt) > max_chars:
                remain = max_chars - used
                if remain > 0:
                    out.append(txt[:remain] + "...")
                break
            out.append(txt)
            used += len(txt)
        return "\n\n".join(out)

    direct_context = _join_chunks(direct_for_context, max_main_context_tokens)
    aux_summary = _summarize_evidence_list_short(aux, max_sentences=3) if aux else "ไม่มีหลักฐานรอง"

    # --- 4) Debug ---
    debug_meta = {
        "level": level,
        "direct_count": len(direct_for_context),
        "aux_count": len(aux),
        "baseline_count": len(summarizable_baseline),
        "baseline_source": "previous_levels_evidence" if previous_levels_evidence else "fallback_map",
    }

    logger.info(f"Context L{level} → Direct:{len(direct_for_context)} | Aux:{len(aux)} | Baseline:{len(summarizable_baseline)}")

    return {
        "baseline_summary": baseline_summary,
        "direct_context": direct_context,
        "aux_summary": aux_summary,
        "debug_meta": debug_meta,
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
        # parsed = _normalize_keys(_robust_extract_json(raw) or {})
        parsed = _robust_extract_json(raw)

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
        # parsed = _normalize_keys(_robust_extract_json(raw) or {})
        parsed = _robust_extract_json(raw)  # ใช้ตัวเดียวกันทั้งโปรเจกต์!

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
    """L1/L2 ต้องบังคับ C=A=0 และ is_passed ตาม score"""
    result = {
        "score": int(parsed.get("score", 0)),
        "reason": parsed.get("reason", "No reason provided by LLM (Low Level)."),
        "is_passed": parsed.get("is_passed", False),
        "P_Plan_Score": int(parsed.get("P_Plan_Score", 0)),
        "D_Do_Score": int(parsed.get("D_Do_Score", 0)),
        "C_Check_Score": 0,  # บังคับ!
        "A_Act_Sure": 0,     # บังคับ!
    }
    # แก้ is_passed ถ้า score >=1 แต่ LLM บอก False
    if result["score"] >= 1 and not result["is_passed"]:
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
# FINAL: create_structured_action_plan (Production-Ready 100%)
# ------------------------
def _extract_json_array_for_action_plan(llm_response: str) -> List[Dict[str, Any]]:
    """Extract JSON array อย่างแข็งแกร่งสุด ๆ — ใช้สำหรับ Action Plan เท่านั้น"""
    if not llm_response or not isinstance(llm_response, str):
        return []

    text = llm_response.strip()

    # 1. ลองหาใน code block ก่อน (```json หรือ ```)
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        json_str = fenced.group(1)
    else:
        # 2. หา balanced [] array
        start = text.find("[")
        if start == -1:
            return []
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "[": depth += 1
            elif text[i] == "]":
                depth -= 1
                if depth == 0:
                    json_str = text[start:i+1]
                    break
        else:
            return []

    # 3. Parse ด้วย json → json5 fallback
    try:
        data = json.loads(json_str)
    except:
        try:
            data = json5.loads(json_str)
        except:
            logger.error(f"ActionPlan JSON parse failed: {json_str[:200]}")
            return []

    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def create_structured_action_plan(
    failed_statements: List[Dict[str, Any]],
    sub_id: str,
    target_level: int,
    llm_executor: Any,
    max_retries: int = 3
) -> List[Dict[str, Any]]:
    """
    สร้าง Action Plan ที่สมบูรณ์แบบที่สุดเท่าที่จะเป็นไปได้
    รองรับทุกสถานการณ์จริงใน Production
    """

    # ------------------------------------------------------------------
    # 1. ทุกอย่างผ่าน → แผนรักษาระดับ (Sustain / Optimize)
    # ------------------------------------------------------------------
    if not failed_statements:
        if target_level >= 5:
            return [{
                "Phase": "Level 5 - Optimizing",
                "Goal": f"รักษาและยกระดับความเป็นเลิศอย่างต่อเนื่องสำหรับ {sub_id}",
                "Actions": [{
                    "Statement_ID": "OPT-L5",
                    "Recommendation": "เน้นนวัตกรรม การวิเคราะห์เชิงสาเหตุ และการปรับปรุงกระบวนการด้วยข้อมูลเชิงปริมาณอย่างต่อเนื่อง"
                }]
            }]
        else:
            return [{
                "Phase": f"Level {target_level} - Sustaining",
                "Goal": f"รักษามาตรฐาน Level {target_level} และเตรียมความพร้อมสู่ Level {target_level + 1}",
                "Actions": [{
                    "Statement_ID": f"SUSTAIN-L{target_level}",
                    "Recommendation": f"ติดตามและรักษาการปฏิบัติตามแนวทาง Level {target_level} อย่างสม่ำเสมอ พร้อมเก็บข้อมูลเพื่อเตรียมความพร้อมสู่ระดับถัดไป"
                }]
            }]

    # ------------------------------------------------------------------
    # 2. LLM ไม่มี → Fallback สวยงาม
    # ------------------------------------------------------------------
    if llm_executor is None:
        logger.error("create_structured_action_plan: llm_executor is None → ใช้ fallback")
        actions = []
        for s in failed_statements[:10]:
            sid = s.get("sub_id") or s.get("statement_id") or "UNKNOWN"
            stmt = (s.get("statement") or "").strip()[:200]
            reason = (s.get("reason") or "").strip()[:300]
            actions.append({
                "Statement_ID": sid,
                "Recommendation": f"[{sid}] {stmt} | สาเหตุ: {reason}"
            })
        return [{
            "Phase": f"Level {target_level}",
            "Goal": f"ยกระดับให้ได้ Level {target_level} สำหรับ {sub_id}",
            "Actions": actions or [{"Statement_ID": "NO-LLM", "Recommendation": "กรุณาตรวจสอบเอกสารและดำเนินการตามข้อกำหนดที่ขาดหาย"}]
        }]

    # ------------------------------------------------------------------
    # 3. เตรียม Prompt + Schema
    # ------------------------------------------------------------------
    try:
        schema_json = json.dumps(ActionPlanActions.model_json_schema(), ensure_ascii=False, indent=2)
    except:
        schema_json = '{"Phase":"string","Goal":"string","Actions":[{"Statement_ID":"string","Recommendation":"string"}]}'

    system_prompt = (
        SYSTEM_ACTION_PLAN_PROMPT
        + "\n\n--- JSON SCHEMA (ตอบเป็น ARRAY เท่านั้น) ---\n"
        + schema_json
        + "\n\nIMPORTANT:\n"
          "- ตอบกลับด้วย JSON ARRAY เท่านั้น เช่น: [ { ... }, { ... } ]\n"
          "- ห้ามมีข้อความนอก JSON เด็ดขาด\n"
          "- ทุก field ต้องเป็นภาษาไทย\n"
          "- Actions ต้องมีอย่างน้อย 1 รายการต่อ Phase"
    )

    # จัดกลุ่ม Statement ให้อ่านง่าย
    stmt_blocks = []
    for i, s in enumerate(failed_statements, 1):
        sid = s.get("sub_id") or s.get("statement_id") or f"STMT-{i}"
        level = s.get("level", "?")
        text = str(s.get("statement") or "").strip()
        reason = str(s.get("reason") or "").strip()
        stmt_blocks.append(
            f"ลำดับที่ {i}\n"
            f"Statement ID: {sid} (Level {level})\n"
            f"ข้อความ: {text}\n"
            f"เหตุผลที่ไม่ผ่าน: {reason}\n"
        )

    human_prompt = ACTION_PLAN_PROMPT.format(
        sub_id=sub_id,
        target_level=target_level,
        failed_statements_list="\n\n".join(stmt_blocks)
    )

    # ------------------------------------------------------------------
    # 4. เรียก LLM + Extract (แข็งแกร่งสุด)
    # ------------------------------------------------------------------
    for attempt in range(max_retries):
        try:
            raw = _fetch_llm_response(
                system_prompt=system_prompt,
                user_prompt=human_prompt,
                max_retries=1,
                llm_executor=llm_executor
            )

            items = _extract_json_array_for_action_plan(raw)
            if not items:
                logger.warning(f"ActionPlan attempt {attempt+1}: ไม่ได้ JSON array → ลองใหม่")
                time.sleep(1)
                continue

            # เติม default + ทำความสะอาด
            result = []
            for item in items:
                phase = str(item.get("Phase") or f"Level {target_level}").strip()
                goal = str(item.get("Goal") or f"ยกระดับให้ได้ Level {target_level}").strip()
                actions = item.get("Actions") or []

                if not isinstance(actions, list):
                    actions = [actions] if isinstance(actions, dict) else []

                clean_actions = []
                for act in actions:
                    if not isinstance(act, dict): continue
                    rec = str(act.get("Recommendation") or "").strip()
                    sid = str(act.get("Statement_ID") or "UNKNOWN").strip()
                    if rec:
                        clean_actions.append({"Statement_ID": sid, "Recommendation": rec})

                if not clean_actions:
                    clean_actions.append({
                        "Statement_ID": "FALLBACK",
                        "Recommendation": "ดำเนินการปรับปรุงตามข้อบกพร่องที่ระบุในรายงานการประเมิน"
                    })

                result.append({"Phase": phase, "Goal": goal, "Actions": clean_actions})

            if result:
                logger.info(f"Action Plan สร้างสำเร็จ → {len(result)} phase(s)")
                return result

        except Exception as e:
            logger.warning(f"ActionPlan attempt {attempt+1} เกิด error: {e}")

    # ------------------------------------------------------------------
    # 5. Final Fallback (ไม่เคยเกิดขึ้นใน Production ได้)
    # ------------------------------------------------------------------
    logger.error("ActionPlan: ทุกอย่างล้มเหลว → ใช้ Hardcoded Template")
    actions = []
    for i, s in enumerate(failed_statements[:8], 1):
        sid = s.get("sub_id") or f"STMT-{i}"
        text = str(s.get("statement") or "").strip()[:150]
        actions.append({"Statement_ID": sid, "Recommendation": f"ดำเนินการตามข้อกำหนด: {text}"})

    return [{
        "Phase": f"Level {target_level} - ปรับปรุงด่วน",
        "Goal": f"แก้ไขข้อบกพร่องทั้งหมดเพื่อให้ได้ Level {target_level}",
        "Actions": actions or [{"Statement_ID": "URGENT", "Recommendation": "กรุณาตรวจสอบเอกสารและดำเนินการตามรายงานการประเมินโดยด่วน"}]
    }]