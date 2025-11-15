#core/llm_data_utils.py
"""
Robust LLM + RAG utilities for SEAM assessment.
Responsibilities:
 - Retrieval wrapper: retrieve_context_with_filter (calls core.vectorstore)
 - Robust JSON extraction & normalization (_robust_extract_json, _normalize_keys)
 - LLM invocation wrappers with retries (_fetch_llm_response)
 - evaluate_with_llm: produce {score, reason, is_passed}
 - summarize_context_with_llm: produce evidence summary
 - create_structured_action_plan: generate action plan JSON list
 - Mock control helper: set_mock_control_mode
Design goals: deterministic prompts, strict JSON schema expectation, resilient fallbacks.
"""

import logging
import time
import json
import json5
import random
import hashlib
import regex as re
from typing import List, Dict, Any, Optional, Type, TypeVar, Union, List as TList
from pydantic import BaseModel, ValidationError

# project imports (these must exist)

logger = logging.getLogger(__name__)
# 🟢 ตั้งค่า logging ให้ไม่ซ้ำซ้อน
try:
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
except Exception:
    pass


try:
    from core.seam_prompts import (
        SYSTEM_ASSESSMENT_PROMPT,
        USER_ASSESSMENT_PROMPT,
        SYSTEM_ACTION_PLAN_PROMPT,
        ACTION_PLAN_PROMPT,
        SYSTEM_EVIDENCE_DESCRIPTION_PROMPT,
        EVIDENCE_DESCRIPTION_PROMPT
    )
    # Import VectorStoreManager, LcDocument (LangChain) for type hinting and functions
    from core.vectorstore import VectorStoreManager, get_global_reranker, _get_collection_name
    from core.assessment_schema import StatementAssessment, EvidenceSummary
    from core.action_plan_schema import ActionPlanActions
    from config.global_vars import DEFAULT_ENABLER, INITIAL_TOP_K, FINAL_K_RERANKED
    # 🟢 เพิ่ม LcDocument เข้ามาเพื่อให้ retrieve_context_by_doc_ids ใช้งานได้
    from langchain_core.documents import Document as LcDocument 
except Exception as e:
    # ⚠️ เพิ่ม logging เพื่อช่วย debug หาก import ล้มเหลว
    logger.error(f"FATAL: Missing dependency in llm_data_utils.py: {e}")
    raise ImportError(f"Missing dependency in llm_data_utils.py: {e}")

# LLM instance must be provided in project (models/llm.py exposes llm)
try:
    from models.llm import llm as llm_instance
except Exception:
    llm_instance = None


# Mock control flags
_MAX_LLM_RETRIES = 3
_MOCK_CONTROL_FLAG = False
_MOCK_COUNTER = 0

def set_mock_control_mode(enable: bool):
    global _MOCK_CONTROL_FLAG, _MOCK_COUNTER
    _MOCK_CONTROL_FLAG = bool(enable)
    _MOCK_COUNTER = 0
    logger.info(f"Mock control mode set to {_MOCK_CONTROL_FLAG}")


# ------------------------------------------------------------------
# ID Normalization and Hashing
# ------------------------------------------------------------------
def _hash_stable_id_to_64_char(stable_id: str) -> str:
    # 🟢 ใช้ hashlib ที่ถูก Import ด้านบนแล้ว
    return hashlib.sha256(stable_id.lower().encode('utf-8')).hexdigest()

def normalize_stable_ids(ids: List[str]) -> List[str]:
    normalized = []
    for i in ids:
        if len(i) == 64:
            normalized.append(i.lower())
        else:
            normalized.append(_hash_stable_id_to_64_char(i))
    return normalized

def _hash_to_64(s: str) -> str:
    """Helper used internally by retrieve_context_with_filter (for backward compatibility)."""
    if not s:
        return ""
    try:
        if len(s) == 64 and all(c in "0123456789abcdef" for c in s.lower()):
            return s.lower()
        # 🟢 FIX 3: ลบ import hashlib ที่ซ้ำซ้อน
        return hashlib.sha256(s.lower().encode("utf-8")).hexdigest()
    except Exception:
        return s

# ------------------------------------------------------------------
# Retrieval Wrappers
# ------------------------------------------------------------------

def retrieve_context_by_doc_ids(doc_uuids: List[str], doc_type: str, enabler: Optional[str] = None) -> Dict[str, Any]:
    """
    Retrieve documents by UUIDs from a specific collection, calculating the collection name internally.
    """
    if VectorStoreManager is None:
        logger.error("❌ VectorStoreManager is not available.")
        return {"top_evidences": []}
        
    if not doc_uuids:
        logger.warning("⚠️ No document UUIDs provided for retrieval.")
        return {"top_evidences": []}

    try:
        manager = VectorStoreManager()
        
        # 🟢 ยังคงคำนวณชื่อ Collection เพื่อใช้ใน Log เท่านั้น
        collection_name = _get_collection_name(doc_type=doc_type, enabler=enabler)
        logger.info(f"Targeting collection: '{collection_name}' for UUID retrieval.")
        
        # 2. Normalize ID เป็น 64-char Stable UUIDs
        normalized_uuids = normalize_stable_ids(doc_uuids) 
        
        # 3. ดึงข้อมูลจาก Vector Store
        docs: List[LcDocument] = manager.get_documents_by_id(
            stable_doc_ids=normalized_uuids, 
            doc_type=doc_type,          # <-- ใช้ doc_type เดิม
            enabler=enabler             # <-- ใช้ enabler เดิม
        )

        # 4. จัดรูปแบบผลลัพธ์
        top_evidences = []
        for d in docs:
            meta = d.metadata
            top_evidences.append({
                "doc_id": meta.get("stable_doc_uuid"), 
                "doc_type": meta.get("doc_type"),
                "chunk_uuid": meta.get("chunk_uuid"),      # ID เฉพาะของ Chunk
                "source": meta.get("source") or meta.get("doc_source"), # ชื่อไฟล์/ที่มา
                "content": d.page_content.strip(),
                "chunk_index": meta.get("chunk_index")
            })

        logger.info(f"✅ Successfully retrieved {len(top_evidences)} evidences by UUIDs from collection '{collection_name}'.")
        return {"top_evidences": top_evidences}

    except Exception as e:
        logger.error(f"Error during UUID-based retrieval: {e}", exc_info=True)
        return {"top_evidences": []}
    
# --------------------
# RAG retrieval wrapper (expects VectorStoreManager implemented elsewhere)
# --------------------
def retrieve_context_with_filter(
    query: str,
    doc_type: str, # 🟢 FIX 1: รับ doc_type แทน vsm_manager
    enabler: Optional[str] = None, # 🟢 FIX 1: รับ enabler แทน collection_name
    doc_uuid_filter: Optional[List[str]] = None,
    disable_semantic_filter: bool = False,
    top_k: int = FINAL_K_RERANKED
) -> Dict[str, Any]:
    """
    Wrapper to retrieve top_k chunks from vectorstore, with optional UUID hard filter and rerank.
    Returns {"top_evidences": [...], "aggregated_context": "..." }
    """
    try:
        # 🟢 FIX 1: สร้าง Manager และคำนวณ Collection ภายใน
        if VectorStoreManager is None:
             logger.error("VectorStoreManager is not available.")
             return {"top_evidences": [], "aggregated_context": ""}
        
        manager = VectorStoreManager()
        collection_name = _get_collection_name(doc_type=doc_type, enabler=enabler)

        # 1. Initial check
        if not manager:
            logger.error("Invalid VectorStoreManager instance provided for retrieval (VSM is None).")
            return {"top_evidences": [], "aggregated_context": ""}
        
        base_retriever = None
        
        # 2. Dynamic check: Is it a VSM or is it the final Retriever object?
        if hasattr(manager, '_load_chroma_instance'):
            # Case A: It is the intended VectorStoreManager. Use it to get the retriever.
            try:
                vectorstore = manager._load_chroma_instance(collection_name)
                if not vectorstore:
                    logger.error(f"Vectorstore not found: {collection_name}")
                    return {"top_evidences": [], "aggregated_context": ""}
                
                # build search kwargs
                search_kwargs = {"k": max(INITIAL_TOP_K, top_k)}
                if doc_uuid_filter:
                    # 🟢 FIX 2: ใช้ normalize_stable_ids แทน _hash_to_64
                    normalized = normalize_stable_ids(doc_uuid_filter) 
                    # many vectorstores support 'filter' param; implementation-dependent
                    search_kwargs["filter"] = {"stable_doc_uuid": {"$in": normalized}}

                base_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs=search_kwargs)
                
            except Exception as e:
                logger.error(f"VSM logic failed during load or as_retriever: {e}")
                return {"top_evidences": [], "aggregated_context": ""}
        
        else:
            # Case B: The calling code passed the retriever directly.
            base_retriever = manager
            logger.warning("RAG: Passed object is missing '_load_chroma_instance'; assuming a Retriever was passed directly.")
            
        
        # 3. Unified Retrieval Invocation
        if not base_retriever:
             logger.error("RAG: Could not initialize base_retriever.")
             return {"top_evidences": [], "aggregated_context": ""}
        
        try:
            documents = base_retriever.invoke(query)
        except Exception as e:
            logger.error(f"Retriever invoke error: {e}")
            return {"top_evidences": [], "aggregated_context": ""}

        if not isinstance(documents, list):
            documents = list(documents) if hasattr(documents, "__iter__") else []

        # optional rerank
        if not disable_semantic_filter:
            reranker = get_global_reranker(top_k)
            if reranker and hasattr(reranker, "rerank"):
                try:
                    documents = reranker.rerank(query, documents)
                except Exception as e:
                    logger.warning(f"Rerank failed: {e}")
                    documents = documents[:top_k]
            else:
                documents = documents[:top_k]
        else:
            documents = documents[:top_k]

        top_evidences = []
        aggregated_parts = []
        for d in documents:
            meta = getattr(d, "metadata", {}) or {}
            page_content = getattr(d, "page_content", "") or ""
            
            normalized_meta = {
                "stable_doc_uuid": meta.get("stable_doc_uuid") or meta.get("stable_id") or getattr(d, "id", "N/A"),
                # FIX 4: Simplifies confusing placeholder and ensures a proper fallback string
                "file_name": meta.get("file_name") or meta.get("source") or "unknown_file",
                **meta
            }
            top_evidences.append({"content": page_content, "metadata": normalized_meta})
            
            # ADJUSTED LOGIC: เพิ่ม Source Metadata เข้าไปใน Context
            doc_id_short = normalized_meta['stable_doc_uuid'][:8] + "..." if normalized_meta['stable_doc_uuid'] and len(normalized_meta['stable_doc_uuid']) > 8 else normalized_meta['stable_doc_uuid']
            source_info = f"SOURCE: {normalized_meta['file_name']} (ID: {doc_id_short})"
            aggregated_parts.append(f"[{source_info}]\n{page_content}") # ใส่ Source Info ก่อนเนื้อหา
            
        aggregated_context = "\n\n---\n\n".join(aggregated_parts) # ใช้ตัวคั่นที่ชัดเจนขึ้น
        return {"top_evidences": top_evidences, "aggregated_context": aggregated_context}
    except Exception as e:
        logger.exception(f"retrieve_context_with_filter failed: {e}")
        return {"top_evidences": [], "aggregated_context": ""}


# --------------------
# Robust JSON extraction
# --------------------
UUID_PATTERN = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', re.IGNORECASE)

def _robust_extract_json(text: str) -> Optional[Any]:
    if not text:
        return None
    txt = text.strip()
    # remove fences
    txt = re.sub(r'^\s*```(?:json)?\s*', '', txt, flags=re.IGNORECASE | re.MULTILINE)
    txt = re.sub(r'\s*```\s*$', '', txt, flags=re.MULTILINE)
    # try find balanced {...} or [...]
    try:
        # first try full json
        return json.loads(txt)
    except Exception:
        pass
    # try to extract first balanced {...} or [...]
    obj_match = re.search(r'(\{.*\})', txt, flags=re.DOTALL)
    arr_match = re.search(r'(\[.*\])', txt, flags=re.DOTALL)
    candidate = None
    if obj_match:
        candidate = obj_match.group(1)
    elif arr_match:
        candidate = arr_match.group(1)
    if candidate:
        try:
            return json.loads(candidate)
        except Exception:
            try:
                return json5.loads(candidate)
            except Exception:
                return None
    # fallback try json5 on whole text
    try:
        return json5.loads(txt)
    except Exception:
        return None

def _normalize_keys(data: Any) -> Any:
    if isinstance(data, dict):
        out = {}
        mapping = {
            "llm_score": "score",
            "reasoning": "reason",
            "llm_reasoning": "reason",
            "assessment_reason": "reason",
            "comment": "reason"
        }
        for k, v in data.items():
            lk = k.lower()
            nk = mapping.get(lk, k)
            out[nk] = _normalize_keys(v)
        return out
    if isinstance(data, list):
        return [_normalize_keys(x) for x in data]
    return data

# --------------------
# LLM low-level fetcher
# --------------------
def _fetch_llm_response(system_prompt: str, user_prompt: str, max_retries: int = 2) -> str:
    if llm_instance is None:
        raise ConnectionError("LLM instance not initialized.")
    config = {"temperature": 0.0}
    for attempt in range(max_retries):
        try:
            resp = llm_instance.invoke([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], config=config)
            # support different return shapes
            if hasattr(resp, "content"):
                return resp.content.strip()
            if isinstance(resp, str):
                return resp.strip()
            # try dict
            if isinstance(resp, dict) and "content" in resp:
                return resp["content"].strip()
            return str(resp).strip()
        except Exception as e:
            logger.warning(f"LLM call failed (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(0.5)
    raise ConnectionError("LLM calls failed after retries.")

# --------------------
# evaluate_with_llm
# --------------------
T = TypeVar("T", bound=BaseModel)

def evaluate_with_llm(context: str, sub_criteria_name: str, level: int, statement_text: str, sub_id: str, **kwargs) -> Dict[str, Any]:
    global _MOCK_CONTROL_FLAG, _MOCK_COUNTER
    # NEW: Extract PDCA phase from kwargs for prompt injection
    pdca_phase = kwargs.get('pdca_phase', f"L{level} Concept") 
    
    if _MOCK_CONTROL_FLAG:
        _MOCK_COUNTER += 1
        score = 1 if _MOCK_COUNTER <= 9 else 0
        return {"score": score, "reason": f"MOCK {'PASS' if score else 'FAIL'}", "is_passed": score >= 1}
    
    # ✅ FIX 1: แก้ไข Syntax Error จาก === เป็น is None
    if llm_instance is None:
        # fallback random deterministic-ish
        score = random.choice([0, 1])
        return {"score": score, "reason": "LLM unavailable — fallback random", "is_passed": score >= 1}

    # FIX & UPDATE: Pass all required arguments including 'pdca_phase' and use 'statement_text' consistently
    user_prompt = USER_ASSESSMENT_PROMPT.format(
        sub_criteria_name=sub_criteria_name,
        level=level,
        statement_text=statement_text, # ใช้ 'statement_text' ถูกต้องตามที่ตกลง
        sub_id=sub_id,
        context=context or "ไม่มีหลักฐานที่เกี่ยวข้อง",
        pdca_phase=pdca_phase # Inject the new PDCA phase
    )
    # include schema for strict JSON expectation
    try:
        schema_json = json.dumps(StatementAssessment.model_json_schema(), ensure_ascii=False, indent=2)
    except Exception:
        schema_json = '{"score": 0, "reason": "string"}'

    system_prompt = SYSTEM_ASSESSMENT_PROMPT + "\n\n--- JSON SCHEMA ---\n" + schema_json + "\nIMPORTANT: Respond only with valid JSON following the schema."

    try:
        raw_out = _fetch_llm_response(system_prompt=system_prompt, user_prompt=user_prompt, max_retries=_MAX_LLM_RETRIES)
        parsed = _robust_extract_json(raw_out)
        if not parsed:
            # best-effort: try to parse any numbers in response
            logger.warning("LLM response JSON extract failed; using fallback score detection.")
            # fallback to fail
            return {"score": 0, "reason": "Could not parse LLM JSON", "is_passed": False}
        parsed = _normalize_keys(parsed)
        # FIX & UPDATE: ใช้ 'score' หลังการ Normalize
        score = parsed.get("score", 0) 
        try:
            score_int = int(str(score))
        except Exception:
            score_int = 0
        return {"score": score_int, "reason": parsed.get("reason", parsed.get("comment", "")), "is_passed": score_int >= 1}
    except Exception as e:
        logger.exception(f"evaluate_with_llm failed: {e}")
        return {"score": 0, "reason": f"LLM error: {e}", "is_passed": False}

# --------------------
# summarize_context_with_llm
# --------------------
def summarize_context_with_llm(context: str, sub_criteria_name: str, level: int, sub_id: str, schema: Optional[Type[T]] = None) -> Dict[str, Any]:
    # ✅ FIX 2: แก้ไข Syntax Error จาก === เป็น is None
    if llm_instance is None:
        return {"summary": "LLM not available", "suggestion_for_next_level": "Check LLM service."}
        
    human_prompt = EVIDENCE_DESCRIPTION_PROMPT.format(sub_criteria_name=sub_criteria_name, level=level, context=(context or "")[:4000], sub_id=sub_id)
    try:
        schema_json = json.dumps(EvidenceSummary.model_json_schema(), ensure_ascii=False, indent=2)
    except Exception:
        schema_json = "{}"
    system_prompt = SYSTEM_EVIDENCE_DESCRIPTION_PROMPT + "\n\n--- JSON SCHEMA ---\n" + schema_json + "\nRespond only with valid JSON."
    try:
        raw = _fetch_llm_response(system_prompt=system_prompt, user_prompt=human_prompt, max_retries=2)
        parsed = _robust_extract_json(raw)
        if not parsed:
            return {"summary": "Could not parse LLM summary", "suggestion_for_next_level": "Manual review"}
        parsed = _normalize_keys(parsed)
        # convert to simple dict if pydantic available
        try:
            if isinstance(parsed, dict):
                return parsed
            return {"summary": str(parsed)}
        except Exception:
            return {"summary": str(parsed)}
    except Exception as e:
        logger.exception(f"summarize_context_with_llm failed: {e}")
        return {"summary": "LLM error during summarization", "suggestion_for_next_level": str(e)}

# --------------------
# create_structured_action_plan
# --------------------
def create_structured_action_plan(failed_statements_data: List[Dict[str, Any]], sub_id: str, enabler: str, target_level: int, max_retries: int = 2) -> List[Dict[str, Any]]:
    if _MOCK_CONTROL_FLAG:
        return [{"Phase": "MOCK", "Goal": f"MOCK plan for {sub_id}", "Actions": []}]

    try:
        schema_json = json.dumps(ActionPlanActions.model_json_schema(), ensure_ascii=False, indent=2)
    except Exception:
        schema_json = "{}"
    system_prompt = SYSTEM_ACTION_PLAN_PROMPT + "\n\n--- JSON SCHEMA ---\n" + schema_json + "\nRespond only in valid JSON array."

    statements_text = []
    for s in failed_statements_data:
        # ✅ FIX 3: เปลี่ยนจาก 'statement_text' เป็น 'statement' เพื่อให้สอดคล้องกับโครงสร้างข้อมูลใน Engine
        statements_text.append(f"Level: {s.get('level')}\nStatement: {s.get('statement')}\nReason: {s.get('reason')}\n") 

    human_prompt = ACTION_PLAN_PROMPT.format(sub_id=sub_id, target_level=target_level, failed_statements_list="\n".join(statements_text))

    for attempt in range(max_retries + 1):
        try:
            raw = _fetch_llm_response(system_prompt=system_prompt, user_prompt=human_prompt, max_retries=1)
            cleaned = raw.strip()
            parsed = _robust_extract_json(cleaned)
            if not parsed:
                raise ValueError("LLM returned non-JSON for action plan.")
            # validate
            parsed = _normalize_keys(parsed)
            # if it's dict, wrap into list
            if isinstance(parsed, dict):
                parsed = [parsed]
            # return as-is (assume shape matches ActionPlanActions)
            return parsed
        except Exception as e:
            logger.warning(f"Action plan attempt {attempt+1} failed: {e}")
            time.sleep(0.5)
            if attempt == max_retries:
                logger.error(f"Failed to generate action plan after {max_retries+1} attempts: {e}")
    # fallback
    return [{
        "Phase": f"Fallback L{target_level}",
        "Goal": f"Manual review for {sub_id}",
        "Actions": [{"Statement_ID": "LLM_ERROR", "Recommendation": "Manual consultant review required."}]
    }]