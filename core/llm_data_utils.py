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
from typing import List, Dict, Any, Optional, Union, Callable, TypeVar, Set
import json5
from utils.enabler_keyword_map import ENABLER_KEYWORD_MAP, DEFAULT_KEYWORDS
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever # FIX: Import BM25 จาก community
from core.action_plan_schema import get_clean_action_plan_schema


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
    USE_HYBRID_SEARCH, 
    HYBRID_VECTOR_WEIGHT, 
    HYBRID_BM25_WEIGHT,
    MAX_ACTION_PLAN_PHASES,
    MAX_STEPS_PER_ACTION,
    ACTION_PLAN_STEP_MAX_WORDS,
    ACTION_PLAN_LANGUAGE
)

# ===================================================================
# 2. Critical Utilities (ต้องมีจริง — ไม่มี fallback)
# ===================================================================
# 🎯 FIX 1: เปลี่ยน Import จาก _get_collection_name ไปเป็น get_doc_type_collection_key
from core.vectorstore import get_hf_embeddings
from utils.path_utils import get_doc_type_collection_key # <--- นำเข้าฟังก์ชันใหม่จาก utils/path_utils
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
    USER_LOW_LEVEL_PROMPT_TEMPLATE
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

# Helper: สร้าง Chroma where filter
def _create_where_filter(stable_doc_ids: Optional[Set[str]] = None, 
                         subject: Optional[str] = None,
                         sub_topic: Optional[str] = None) -> Dict[str, Any]:
    """สร้าง where filter สำหรับ ChromaDB ที่แข็งแกร่งที่สุด"""
    filters = []
    
    if stable_doc_ids:
        filters.append({"stable_doc_uuid": {"$in": list(stable_doc_ids)}})
    
    if subject:
        cleaned = subject.strip()
        if cleaned:
            filters.append({"subject": cleaned})
    
    if sub_topic:
        filters.append({"sub_topic": {"$eq": sub_topic}})
    
    if len(filters) > 1:
        return {"$and": filters}
    elif filters:
        return filters[0]
    else:
        return {}

def retrieve_context_for_endpoint(
    vectorstore_manager,
    query: str = "",
    tenant: Optional[str] = None,
    year: Optional[Union[int, str]] = None,
    stable_doc_ids: Optional[Set[str]] = None,
    doc_type: Optional[str] = None,
    enabler: Optional[str] = None,
    subject: Optional[str] = None,
    sub_topic: Optional[str] = None,  # ใหม่: เช่น "KM-4.1"
    k_to_retrieve: int = INITIAL_TOP_K,
    k_to_rerank: int = FINAL_K_RERANKED,
) -> Dict[str, Any]:
    """
    ดึง context จากเอกสารที่เลือกมาแล้ว (stable_doc_ids) หรือ filter แม่น ๆ
    รองรับ sub_topic เช่น "KM-4.1" → แม่นสุด
    """
    start_time = time.time()
    vsm = vectorstore_manager

    # 1. กำหนด collection และแก้ไขปัญหา doc_type เป็น List/String Literal
    
    clean_doc_type = doc_type or 'seam'
    
    # 💡 FIX A: ตรวจสอบและพยายามแปลง String Literal ที่มาจาก curl เช่น '["seam"]'
    if isinstance(clean_doc_type, str) and clean_doc_type.strip().startswith('['):
        try:
            # พยายามโหลดเป็น JSON Array
            parsed_list = json.loads(clean_doc_type.strip())
            
            if isinstance(parsed_list, (list, tuple)) and parsed_list:
                # ถ้าโหลดได้เป็น List/Tuple ให้ใช้สมาชิกตัวแรก
                clean_doc_type = parsed_list[0]
            elif isinstance(parsed_list, str):
                # ถ้าโหลดได้เป็น String (อาจเกิดขึ้นได้)
                clean_doc_type = parsed_list
                
        except json.JSONDecodeError:
            # ถ้าโหลดไม่ได้ ให้ข้ามไปใช้ String เดิม
            logger.debug(f"Could not parse doc_type string literal: {clean_doc_type}")
            pass

    # 💡 FIX B: จัดการกับ List/Tuple ที่เข้ามาปกติ (กรณี Router ส่งมาถูก หรือหลังจากการ Parse JSON)
    if isinstance(clean_doc_type, (list, tuple)):
        # ใช้ element แรกเท่านั้น
        clean_doc_type = str(clean_doc_type[0]) if clean_doc_type else 'seam'
    elif not isinstance(clean_doc_type, str):
        # บังคับเป็น String
        clean_doc_type = str(clean_doc_type)

    # 💡 FIX C: ทำความสะอาด Quote ที่อาจติดมา (เช่น 'seam' หรือ "seam")
    clean_doc_type = str(clean_doc_type).strip().strip("'\"")

    # 🎯 ใช้ get_doc_type_collection_key จาก utils/path_utils.py
    collection_name = get_doc_type_collection_key(
        doc_type=clean_doc_type, 
        enabler=enabler
    )

    chroma = vsm._load_chroma_instance(collection_name)
    if not chroma:
        # 📌 NOTE: ใช้ clean_doc_type ใน Log เพื่อความถูกต้อง
        logger.error(f"Collection {collection_name} (Doc Type: {clean_doc_type}) not found!")
        return {"top_evidences": [], "aggregated_context": "", "retrieval_time": 0, "used_chunk_uuids": []}

    # 2. สร้าง filter ที่แข็งแกร่ง
    where_filter = _create_where_filter(stable_doc_ids, subject, sub_topic)
    logger.info(f"Retrieval → Collection: {collection_name} | Filter: {where_filter} | Query: {query[:80]}...")

    # 3. Embed query
    try:
        emb = get_hf_embeddings()
        # BGE-M3 แนะนำให้ใช้ prefix
        query_emb = emb.embed_query(f"query: {query}") 
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return {"top_evidences": [], "aggregated_context": "", "retrieval_time": 0, "used_chunk_uuids": []}

    # 4. Query Chroma
    try:
        results = chroma._collection.query(
            query_embeddings=[query_emb],
            n_results=k_to_retrieve,
            where=where_filter if where_filter else None,
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
        logger.error(f"Chroma query failed: {e}")
        return {"top_evidences": [], "aggregated_context": "", "retrieval_time": 0, "used_chunk_uuids": []}

    # 5. แปลงเป็น LcDocument
    raw_chunks: List[LcDocument] = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        meta["retrieval_distance"] = float(dist)
        raw_chunks.append(LcDocument(page_content=doc, metadata=meta))

    logger.info(f"Raw retrieval: {len(raw_chunks)} chunks")

    # 6. Rerank (สำคัญมาก!)
    final_chunks = raw_chunks
    reranker = get_global_reranker()
    if reranker and len(raw_chunks) > k_to_rerank:
        try:
            reranked = reranker.compress_documents(
                documents=raw_chunks,
                query=query,
                top_n=k_to_rerank
            )
            # ดึง Document ออกมา
            final_chunks = [getattr(r, "document", r) for r in reranked]
            logger.info(f"Reranked → {len(final_chunks)} chunks")
        except Exception as e:
            logger.warning(f"Reranker failed: {e}")

    # 7. สร้าง output
    top_evidences = []
    aggregated_parts = []
    used_chunk_uuids = []

    for doc in final_chunks[:k_to_rerank]:
        md = doc.metadata or {}
        text = str(doc.page_content or "").strip()
        if not text:
            continue

        chunk_uuid = md.get("chunk_uuid") or md.get("dedup_chunk_uuid")
        if not chunk_uuid or len(chunk_uuid) < 32:
            continue  # กรอง TEMP ID

        used_chunk_uuids.append(chunk_uuid)

        top_evidences.append({
            "doc_id": md.get("stable_doc_uuid"),
            "chunk_uuid": chunk_uuid,
            "source": md.get("source") or md.get("filename") or "Unknown",
            "text": text,
            "pdca_tag": md.get("pdca_tag", "Other"),
            "retrieval_distance": md.get("retrieval_distance", 1.0),
            "sub_topic": md.get("sub_topic"),
        })
        aggregated_parts.append(f"[SOURCE: {md.get('source', 'Unknown')}] {text}")

    result = {
        "top_evidences": top_evidences,
        "aggregated_context": "\n\n---\n\n".join(aggregated_parts),
        "retrieval_time": round(time.time() - start_time, 3),
        "used_chunk_uuids": used_chunk_uuids
    }
    logger.info(f"Final retrieval: {len(top_evidences)} chunks | Sub-topic: {sub_topic}")
    return result

# ========================
# 2. retrieve_context_by_doc_ids (สำหรับ hydration ใน router)
# ========================
def retrieve_context_by_doc_ids(
    doc_uuids: List[str],
    doc_type: str,
    enabler: Optional[str] = None,
    vectorstore_manager = None,
    limit: int = 50,
    tenant: Optional[str] = None, # <-- ต้องมี
    year: Optional[Union[int, str]] = None, # <-- ต้องมี (แก้ไขล่าสุด)
) -> Dict[str, Any]:
    """
    ดึง chunks จาก stable_doc_uuid หลายตัว (ใช้ตอน hydration sources)
    """
    start_time = time.time()
    vsm = vectorstore_manager or VectorStoreManager()
    
    # 🎯 FIX: แทนที่การสร้าง collection_name ด้วยตนเอง
    # collection_name = f"{doc_type}"
    # if enabler and enabler != DEFAULT_ENABLER:
    #     collection_name = f"{doc_type}_{enabler.lower()}"
    
    # 🟢 ใช้ get_doc_type_collection_key เพื่อความสม่ำเสมอและถูกต้อง
    collection_name = get_doc_type_collection_key(doc_type=doc_type, enabler=enabler)

    chroma = vsm._load_chroma_instance(collection_name)
    if not chroma:
        logger.error(f"Collection {collection_name} not found for hydration")
        return {"top_evidences": []}

    if not doc_uuids:
        return {"top_evidences": []}

    logger.info(f"Hydration → {len(doc_uuids)} doc IDs from {collection_name}")

    try:
        # การดึงข้อมูลด้วย stable_doc_uuid สำหรับ hydration นั้นถูกต้องแล้ว
        results = chroma._collection.get(
            where={"stable_doc_uuid": {"$in": doc_uuids}},
            limit=limit,
            include=["documents", "metadatas"]
        )
    except Exception as e:
        logger.error(f"Hydration query failed: {e}")
        return {"top_evidences": []}

    evidences = []
    for doc, meta in zip(results["documents"], results["metadatas"]):
        if not doc.strip():
            continue
        evidences.append({
            "doc_id": meta.get("stable_doc_uuid"),
            "chunk_uuid": meta.get("chunk_uuid"),
            "source": meta.get("source") or meta.get("filename") or "Unknown",
            "text": doc,
            "pdca_tag": meta.get("pdca_tag", "Other"),
        })

    logger.info(f"Hydration success: {len(evidences)} chunks from {len(doc_uuids)} docs")
    return {"top_evidences": evidences}

# ------------------------
# Retrieval: retrieve_context_with_filter (แก้ไขประสิทธิภาพ + Logger Fix)
# ------------------------
def retrieve_context_with_filter(
    query: Union[str, List[str]],
    doc_type: str,
    tenant: Optional[str] = None,
    year: Optional[Union[int, str]] = None,
    enabler: Optional[str] = None,
    subject: Optional[str] = None,
    # ต้องเป็น Instance ของ Manager ที่มี create_hybrid_retriever
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
    เวอร์ชันแก้ไขแล้ว: ใช้ Hybrid Search (BM25 + Vector) ที่สร้างและ Cache ไว้ใน Manager
    """
    start_time = time.time()
    all_retrieved_chunks: List[Any] = []
    used_chunk_uuids: List[str] = []

    # 1. ใช้ VectorStoreManager เดียวกันทั้งหมด
    # สมมติว่า VectorStoreManager() มี Logic ในการ Initialise Chroma Client (self._client)
    manager = vectorstore_manager or VectorStoreManager() 
    if manager is None or not hasattr(manager, '_client') or manager._client is None:
        logger.error("VectorStoreManager not initialized or _client is missing!")
        return {"top_evidences": [], "aggregated_context": "", "retrieval_time": 0.0, "used_chunk_uuids": []}

    # 🟢 NEW FIX: ตรวจสอบและกำหนด Logger ให้กับ VSM Instance
    # เพื่อให้เมธอดภายใน (เช่น create_hybrid_retriever) สามารถใช้ self.logger ได้
    if not hasattr(manager, 'logger') or manager.logger is None:
        try:
            manager.logger = logger # กำหนด logger ของ module นี้ให้ VSM
            logger.info("Assigned module logger to VectorStoreManager instance (Worker/Fallback Fix).")
        except NameError:
            # กรณีที่ logger ไม่ถูก import ใน module นี้ (ซึ่งไม่ควรเกิดขึ้น)
            pass

    queries_to_run = [query] if isinstance(query, str) else list(query or [])
    if not queries_to_run:
        queries_to_run = [""]

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

    # 3. Priority chunks (จาก evidence mapping)
    guaranteed_priority_chunks = []
    if priority_docs_input:
        for doc in priority_docs_input:
            if doc is None:
                continue
            if isinstance(doc, dict):
                pc = doc.get('page_content') or doc.get('text') or ''
                meta = doc.get('metadata') or {}
                if 'chunk_uuid' in doc:
                    meta['chunk_uuid'] = doc['chunk_uuid']
                if 'doc_id' in doc:
                    meta['stable_doc_uuid'] = doc['doc_id']
                if 'pdca_tag' in doc:
                    meta['pdca_tag'] = doc['pdca_tag']
                if pc.strip():
                    guaranteed_priority_chunks.append(LcDocument(page_content=pc, metadata=meta))
            elif hasattr(doc, 'page_content'):
                guaranteed_priority_chunks.append(doc)

    # 4. Collection name
    collection_name = get_doc_type_collection_key(doc_type, enabler or "KM")
    logger.info(f"Requesting retriever → collection='{collection_name}' (doc_type={doc_type}, enabler={enabler})")

    # 5. สร้าง Filter
    where_filter: Dict[str, Any] = {}
    if stable_doc_ids:
        logger.info(f"Applying Stable Doc ID filter: {len(stable_doc_ids)} IDs")
        where_filter = {"stable_doc_uuid": {"$in": stable_doc_ids}}

    if subject:
        subject_filter = {"subject": {"$eq": subject}}
        if where_filter:
            where_filter = {"$and": [where_filter, subject_filter]}
            logger.info(f"Adding Subject filter (AND logic): {subject}")
        else:
            where_filter = subject_filter

    # === HYBRID SEARCH MODE (BM25 + Vector) ===
    hybrid_retriever = None

    if USE_HYBRID_SEARCH:
        try:
            # 🎯 FIX: เรียกใช้ Manager ที่ Cache Hybrid Retriever ไว้แล้ว
            logger.info(f"Requesting Hybrid Retriever from Manager for {collection_name} (Cached)...")
            hybrid_retriever = manager.create_hybrid_retriever(collection_name=collection_name)
            logger.info(f"HYBRID mode activated: Vector 70% + BM25 30% for {collection_name} (Cached)")

        except Exception as e:
            # Fallback หาก Manager ไม่สามารถสร้าง Hybrid Retriever ได้ (เช่น ไม่มี BM25 Index)
            # 🚨 BUG: บันทึก Error จาก VSM ที่ไม่มี Logger (แต่ตอนนี้แก้ไขแล้ว)
            logger.warning(f"Hybrid mode failed (Error calling manager.create_hybrid_retriever: {e}), falling back to vector only")
            use_hybrid = False

    # 6. เลือก Retriever ที่จะใช้
    if USE_HYBRID_SEARCH and hybrid_retriever:
        retriever = hybrid_retriever
    else:
        # ใช้ Vector Retriever อย่างเดียว
        retriever = manager.get_retriever(collection_name)
        logger.info("Using VECTOR ONLY mode.")

    # 7. ดึงข้อมูลจาก VectorStore
    retrieved_chunks = []
    if retriever:
        for q in queries_to_run:
            q_log = q[:120] + "..." if len(q) > 120 else q
            logger.critical(f"[QUERY] Running: '{q_log}' → collection='{collection_name}'")

            try:
                search_kwargs = {"k": INITIAL_TOP_K}  # INITIAL_TOP_K
                if where_filter:
                    search_kwargs["where"] = where_filter
                
                # 🎯 FIX: ใช้ get_relevant_documents ที่รับ **search_kwargs โดยตรงจะเสถียรกว่า
                if hasattr(retriever, "get_relevant_documents"):
                    # EnsembleRetriever และ ChromaRetriever มักจะรับ kwargs โดยตรง
                    docs = retriever.get_relevant_documents(q, **search_kwargs)
                elif hasattr(retriever, "invoke"):
                    # Fallback สำหรับ LangChain Runnable API 
                    docs = retriever.invoke(q, config={"configurable": {"search_kwargs": search_kwargs}})
                else:
                    docs = []
                    
                retrieved_chunks.extend(docs or [])
            except Exception as e:
                logger.error(f"Retriever invoke failed: {e}", exc_info=True)
    else:
        logger.error(f"Retriever NOT FOUND for collection: {collection_name}")

    logger.critical(f"[RETRIEVAL] Raw chunks from ChromaDB: {len(retrieved_chunks)} documents")

    # 8. รวม + deduplicate
    all_chunks = retrieved_chunks + fallback_chunks + guaranteed_priority_chunks
    unique_map: Dict[str, LcDocument] = {}

    for doc in all_chunks:
        if not doc or not hasattr(doc, "page_content"):
            continue
        md = getattr(doc, "metadata", {}) or {}
        pc = str(getattr(doc, "page_content", "") or "").strip()
        if not pc:
            continue

        if level == 3:
            pc = pc[:500]
            doc.page_content = pc

        chunk_uuid = md.get("chunk_uuid") or md.get("stable_doc_uuid") or f"TEMP-{uuid.uuid4().hex[:12]}"
        if chunk_uuid not in unique_map:
            md["dedup_chunk_uuid"] = chunk_uuid
            unique_map[chunk_uuid] = doc

    dedup_chunks = list(unique_map.values())
    logger.info(f"After dedup: {len(dedup_chunks)} chunks")

    # 9. Rerank
    final_docs = list(guaranteed_priority_chunks)
    slots_left = max(0, 12 - len(final_docs))  # FINAL_K_RERANKED
    candidates = [d for d in dedup_chunks if d not in final_docs]

    # Patch metadata ที่หายไปหลัง rerank
    candidate_metadata_map = {
        doc.page_content: getattr(doc, 'metadata', {})
        for doc in candidates if hasattr(doc, 'page_content') and doc.page_content.strip()
    }

    if slots_left > 0 and candidates:
        reranker = get_global_reranker()
        if reranker and hasattr(reranker, "compress_documents"):
            try:
                reranked_results = reranker.compress_documents(
                    documents=candidates,
                    query=queries_to_run[0],
                    top_n=slots_left
                )
                for result in reranked_results:
                    doc_to_add = getattr(result, 'document', result)
                    if doc_to_add and hasattr(doc_to_add, 'page_content') and doc_to_add.page_content.strip():
                        current_md = getattr(doc_to_add, 'metadata', {})
                        if not current_md.get("chunk_uuid") and doc_to_add.page_content in candidate_metadata_map:
                            doc_to_add.metadata = candidate_metadata_map[doc_to_add.page_content]
                        final_docs.append(doc_to_add)
                logger.info(f"Reranker returned {len(reranked_results)} docs")
            except Exception as e:
                logger.warning(f"Reranker failed ({e}), using raw candidates")
                final_docs.extend(candidates[:slots_left])
        else:
            final_docs.extend(candidates[:slots_left])
    else:
        logger.info("No slots left or no candidates → priority only")

    # 10. สร้างผลลัพธ์สุดท้าย
    top_evidences = []
    aggregated_parts = []
    used_chunk_uuids = []

    VALID_CHUNK_ID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$|^[0-9a-f]{64}(-[0-9]+)?$", re.IGNORECASE)
    VALID_STABLE_ID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$|^[0-9a-f]{64}$", re.IGNORECASE)

    for doc in final_docs[:12]:
        md = getattr(doc, "metadata", {}) or {}
        pc = str(getattr(doc, "page_content", "") or "").strip()
        if not pc:
            continue

        chunk_uuid = md.get("chunk_uuid") or md.get("dedup_chunk_uuid") or md.get("id")
        stable_doc_uuid = md.get("stable_doc_uuid") or md.get("source_doc_id")

        primary_id = None
        if stable_doc_uuid and VALID_STABLE_ID.match(str(stable_doc_uuid)):
            primary_id = stable_doc_uuid
        elif chunk_uuid and VALID_CHUNK_ID.match(str(chunk_uuid)):
            primary_id = chunk_uuid
        else:
            logger.warning(f"Chunk has no valid ID! Stable: {stable_doc_uuid}, Chunk: {chunk_uuid}")
            primary_id = f"TEMP-{uuid.uuid4().hex[:8]}"

        if not str(primary_id).startswith("TEMP-"):
            used_chunk_uuids.append(str(primary_id))

        source = md.get("source_filename") or md.get("source") or md.get("filename") or "Unknown File"
        pdca = md.get("pdca_tag", "Other")
        rerank_score = float(md.get("_rerank_score_force") or md.get("relevance_score") or 0.0)

        top_evidences.append({
            "doc_id": stable_doc_uuid or primary_id,
            "chunk_uuid": chunk_uuid or primary_id,
            "stable_doc_uuid": stable_doc_uuid,
            "source": source,
            "source_filename": source,
            "text": pc,
            "pdca_tag": pdca,
            "rerank_score": rerank_score,
        })
        aggregated_parts.append(f"[{pdca}] [SOURCE: {source}] {pc}")

    result = {
        "top_evidences": top_evidences,
        "aggregated_context": "\n\n---\n\n".join(aggregated_parts) if aggregated_parts else "ไม่มีหลักฐานที่เกี่ยวข้อง",
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


# ULTIMATE FINAL VERSION: build_multichannel_context_for_level (OPTIMIZED)
# บทบาทใหม่: สร้างแค่ BASELINE และ AUXILIARY summaries เท่านั้น
# ----------------------------------------------------
def build_multichannel_context_for_level(
    level: int,
    top_evidences: List[Dict[str, Any]],
    previous_levels_map: Optional[Dict[str, Any]] = None,
    previous_levels_evidence: Optional[List[Dict[str, Any]]] = None, # List ของ Chunks ทั้งหมดจาก Level ก่อนหน้า
    max_main_context_tokens: int = 3000, 
    max_summary_sentences: int = 4,
    max_context_length: Optional[int] = None, 
    **kwargs
) -> Dict[str, Any]:
    """
    ฟังก์ชันที่ทำหน้าที่หลักในการสร้าง Context Summary จากหลักฐานที่ดึงมา
    โดยเน้นสร้างเฉพาะ Baseline Summary (จาก Level ก่อนหน้า) และ Auxiliary Summary (จาก Level ปัจจุบัน)
    """
    logger = logging.getLogger(__name__)
    K_MAIN = 5
    MIN_RELEVANCE_FOR_AUX = 0.4  # กรอง aux ที่ต่ำเกินไป

    # --- 1) Baseline Summary ---
    # ใช้ List ที่รวมมาแล้ว (previous_levels_evidence_list) เป็นหลัก
    baseline_evidence = previous_levels_evidence or [] 

    summarizable_baseline = [
        item for item in baseline_evidence
        if isinstance(item, dict) and (item.get("text") or item.get("content"))
    ]
    
    # 🟢 FIX: ปรับการจัดการกรณีไม่มีหลักฐาน baseline
    if not summarizable_baseline:
        baseline_summary = "ไม่มีหลักฐานจาก Level ก่อนหน้า"
    else:
        baseline_summary = _summarize_evidence_list_short(
            summarizable_baseline,
            max_sentences=max_summary_sentences
        )

    # --- 2) Auxiliary Summary ---
    direct, aux_candidates = [], []

    for ev in top_evidences:
        if not isinstance(ev, dict):
            # อาจเป็นกรณีที่ข้อมูลไม่สมบูรณ์
            continue # ข้าม chunks ที่ไม่เป็น dict ไปเลย

        # NEW: รองรับ tag ทั้งแบบเต็มและย่อ
        tag = (ev.get("pdca_tag") or ev.get("PDCA") or "Other").upper()
        relevance = ev.get("rerank_score") or ev.get("score", 0.0)

        # PDCA Chunks ถูกส่งไปเป็น Direct Context (สร้างใน Engine)
        # โค้ดนี้ทำหน้าที่แค่แยกแยะ ไม่ใช่การสร้าง Direct Context
        if tag in {"P", "PLAN", "D", "DO", "C", "CHECK", "A", "ACT"}:
            direct.append(ev)
        elif relevance >= MIN_RELEVANCE_FOR_AUX:  # กรอง aux ที่ต่ำเกิน
            aux_candidates.append(ev)

    # Logic การย้ายจาก aux ไป direct (K_MAIN) ยังคงอยู่ (ใช้ในการวัด/Debug แต่ไม่ได้ส่ง Direct ออกไป)
    if len(direct) < K_MAIN:
        need = K_MAIN - len(direct)
        direct.extend(aux_candidates[:need])
        aux_candidates = aux_candidates[need:]
        
    if len(direct) < K_MAIN:
        logger.warning(f"L{level}: Direct PDCA chunks ยังน้อย ({len(direct)}) หลังย้ายจาก aux")

    aux_summary = _summarize_evidence_list_short(aux_candidates, max_sentences=3) if aux_candidates else "ไม่มีหลักฐานรอง"

    # --- 3) Return ---
    debug_meta = {
        "level": level,
        "direct_count": len(direct),
        "aux_count": len(aux_candidates),
        # 🟢 FIX: นับจำนวนหลักฐานที่ใช้สรุปจริง
        "baseline_count": len(summarizable_baseline), 
        "max_context_length_received": max_context_length 
    }
    logger.info(f"Context L{level} → Direct:{len(direct)} | Aux:{len(aux_candidates)} | Baseline:{len(summarizable_baseline)}")

    return {
        "baseline_summary": baseline_summary,
        "direct_context": "",  
        "aux_summary": aux_summary,
        "debug_meta": debug_meta,
    }

# ------------------------
# LLM fetcher
# ------------------------
def _fetch_llm_response(
    system_prompt: str, 
    user_prompt: str, 
    max_retries: int = 3,
    llm_executor: Any = None 
) -> str:
    """
    เรียก LLM ผ่าน LangChain/Ollama พร้อมระบบป้องกัน Format ผิดเพี้ยน:
    - บังคับ JSON output ด้วย Strict English Prompt
    - ใช้ Regex Extraction ดึงเฉพาะส่วน { ... } เพื่อตัดคำบรรยายภาษาอังกฤษออก
    - Log raw response เต็มๆ เพื่อใช้ในการ Debug
    - Retry พร้อม Exponential Backoff เมื่อเกิด Error
    """
    global _MOCK_FLAG

    # ตรวจสอบว่ามี LLM Instance พร้อมใช้งานหรือไม่
    if llm_executor is None and not _MOCK_FLAG: 
        raise ConnectionError("LLM instance not initialized (Missing llm_executor).")

    # 1. 🛠️ ENFORCED PROMPT (ภาษาอังกฤษมักคุม Format ได้ดีกว่าสำหรับโมเดลขนาดเล็ก)
    enforced_system_prompt = system_prompt.strip() + (
        "\n\n"
        "### STRICT OUTPUT RULES ###\n"
        "1. ANSWER IN VALID JSON OBJECT ONLY.\n"
        "2. NO EXPLANATIONS, NO PREFACE, NO CONVERSATION.\n"
        "3. START WITH '{' AND END WITH '}'.\n"
        "4. DO NOT USE MARKDOWN CODE BLOCKS (```json).\n"
        "5. IF NO EVIDENCE FOUND, RETURN: {\"score\": 0, \"reason\": \"No evidence\", \"is_passed\": false}"
    )

    messages = [
        {"role": "system", "content": enforced_system_prompt},
        {"role": "user",   "content": user_prompt}
    ]

    for attempt in range(1, max_retries + 1):
        try:
            # --- MOCK MODE CASE ---
            if _MOCK_FLAG:
                mock_json = '{"score": 1, "reason": "Mock mode active", "is_passed": true}'
                logger.critical(f"LLM RAW RESPONSE (DEBUG MOCK): {mock_json}")
                return mock_json

            # --- ACTUAL LLM CALL (OLLAMA / LANGCHAIN) ---
            # ใช้ temperature=0.0 เพื่อความแม่นยำสูงสุด
            response = llm_executor.invoke(messages, config={"temperature": 0.0})
            
            # ดึง Text ออกมาจาก Response Object
            raw_text = ""
            if hasattr(response, "content"):
                raw_text = str(response.content)
            elif isinstance(response, str):
                raw_text = response
            else:
                raw_text = str(response)

            # 🔍 บรรทัดที่คุณเห็นใน Log คือบรรทัดนี้ (Log ก่อน Clean เพื่อดูพฤติกรรมโมเดล)
            logger.critical(f"LLM RAW RESPONSE (DEBUG): {raw_text[:1000]}{'...' if len(raw_text) > 1000 else ''}")

            # 2. 🧹 CLEANING LOGIC (Regex Extraction)
            # แก้ปัญหา LLM ตอบ "Based on the text... { ... }"
            raw_text_stripped = raw_text.strip()
            
            # ค้นหาข้อความที่อยู่ในปีกกาคู่แรก { ... }
            json_match = re.search(r'(\{.*\})', raw_text_stripped, re.DOTALL)
            
            if json_match:
                extracted_json = json_match.group(1)
                try:
                    # ทดสอบว่าสิ่งที่ดึงมาเป็น JSON ที่ถูกต้องหรือไม่
                    json.loads(extracted_json) 
                    return extracted_json
                except json.JSONDecodeError:
                    logger.warning(f"Extracted string is not valid JSON: {extracted_json[:100]}")
            
            # 3. 🛡️ FALLBACK: ถ้า Regex ไม่เจอ หรือ Parse ไม่ได้
            # ตรวจดูว่า raw_text (ที่ตัดหัวท้าย) พอลุ้นเป็น JSON ได้ไหม
            return raw_text_stripped

        except Exception as e:
            logger.error(f"LLM call failed (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                # Exponential backoff: 2s, 4s, 8s...
                time.sleep(2 ** attempt)  
            else:
                logger.critical("All LLM attempts failed – returning safe fallback JSON")
                return '{"score": 0, "reason": "LLM_TIMEOUT_OR_FAILURE", "is_passed": false}'

    return '{"score": 0, "reason": "Unknown execution error"}'

# # ------------------------------------------------------------------
# # ฟังก์ชันตัวช่วยใหม่: ทำความสะอาดและดึงค่า String/Dict ออกจาก Response
# # ------------------------------------------------------------------
# def _clean_llm_response_content(resp: Any) -> str:
#     """
#     พยายามดึง content ออกมาในรูปแบบ string ที่สะอาดที่สุด
#     รองรับการห่อหุ้มแบบ Tuple/List ที่มี Dict/String อยู่ภายใน และใช้ Regex Cleanup 
#     เพื่อดึงเฉพาะ JSON Object
#     """
    
#     # --- 1. การทำความสะอาดเบื้องต้น (Existing Logic) ---
#     cleaned_resp_str: str = ""

#     # 1.1 จัดการการห่อหุ้ม (Handle Tuple/List wrapper)
#     if isinstance(resp, (list, tuple)) and resp:
#         resp = resp[0]
#         logger.debug(f"LLM Response was wrapped in {type(resp).__name__}, extracted first element.")

#     # 1.2 จัดการ Response Object/Dict ที่มี 'content' field
#     if hasattr(resp, "content"): 
#         cleaned_resp_str = str(resp.content).strip()
#     elif isinstance(resp, dict) and "content" in resp: 
#         cleaned_resp_str = str(resp["content"]).strip()
#     elif isinstance(resp, str): 
#         cleaned_resp_str = resp.strip()
#     else: 
#         # 1.3 Fallback: แปลงเป็น String
#         cleaned_resp_str = str(resp).strip()
    
#     # --- 2. การทำความสะอาด Regex (The CRITICAL Fix for Malform) ---
    
#     # 2.1 ค้นหาและดึงเฉพาะส่วนที่อยู่ในเครื่องหมายปีกกา { ... }
#     # re.DOTALL: เพื่อให้ . จับคู่ได้แม้กระทั่งอักขระขึ้นบรรทัดใหม่
#     match = re.search(r'\{.*\}', cleaned_resp_str, re.DOTALL)
    
#     if match:
#         json_string_only = match.group(0)
#         logger.debug("Regex Cleanup performed: Extracted pure JSON string.")
#         return json_string_only
    
#     # 2.2 หากไม่พบ JSON Object: คืนค่า String ที่ทำความสะอาดเบื้องต้นไป
#     logger.warning("Regex Cleanup failed: Could not find JSON object. Returning original cleaned string.")
#     return cleaned_resp_str

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

def _get_context_for_level(context: str, level: int) -> str:
    """Return context string with appropriate length limit for each level."""
    if not context:
        return ""
    # L1-L2 ใช้ context ยาวขึ้น
    if level <= 2:
        return context[:6000]  
    # L3-L5 ใช้ค่าที่กำหนดใน global_vars เพื่อลด Latency
    return context[:MAX_EVAL_CONTEXT_LENGTH]  

# =========================
# Main Evaluation Function
# =========================

def evaluate_with_llm(
    context: str, 
    sub_criteria_name: str, 
    level: int, 
    statement_text: str, 
    sub_id: str, 
    llm_executor: Any = None, 
    pdca_phase: str = "",
    level_constraint: str = "",
    must_include_keywords: str = "",
    avoid_keywords: str = "",
    max_rerank_score: float = 0.0,
    max_evidence_strength: float = 10.0,
    **kwargs
) -> Dict[str, Any]:
    """Standard Evaluation for L3+ with robust handling."""
    
    # 🎯 แก้ไข: ใช้ logic การตัด Context ตาม Level
    context_to_send_eval = _get_context_for_level(context, level)
    
    # ตรวจสอบ context ว่าง
    failure_result = _check_and_handle_empty_context(context, sub_id, level)
    if failure_result:
        return failure_result

    # ดึงค่าจาก kwargs (ถ้ายังใช้)
    baseline_summary = kwargs.get("baseline_summary", "")
    aux_summary = kwargs.get("aux_summary", "")

    # สร้าง User Prompt
    try:
        user_prompt = USER_ASSESSMENT_PROMPT.format(
            sub_criteria_name=sub_criteria_name,
            sub_id=sub_id,
            level=level,
            pdca_phase=pdca_phase,
            statement_text=statement_text,
            context=context_to_send_eval, # ใช้ Context ที่ถูกตัดแล้ว
            level_constraint=level_constraint,
            must_include_keywords=must_include_keywords or "ไม่มี",
            avoid_keywords=avoid_keywords or "ไม่มี",
            max_rerank_score=max_rerank_score,
            max_evidence_strength=max_evidence_strength
        )
    except KeyError as e:
        logger.error(f"Missing placeholder in prompt template: {e}")
        user_prompt = f"เกณฑ์: {sub_criteria_name} L{level}\nคำถาม: {statement_text}\nหลักฐาน: {context_to_send_eval}"

    # เพิ่ม summary ถ้ามี
    if baseline_summary:
        user_prompt += f"\n\n--- Baseline summary (จาก Level ก่อนหน้า): ---\n{baseline_summary}"
    if aux_summary:
        user_prompt += f"\n\n--- Auxiliary evidence summary: ---\n{aux_summary}"

    # System Prompt (ต้องมี placeholder)
    try:
        # 🎯 แก้ไข: ใช้ชื่อ key ตรงกับ SYSTEM_ASSESSMENT_PROMPT (คือ max_evidence_strength)
        system_prompt = SYSTEM_ASSESSMENT_PROMPT.format(
            max_evidence_strength=max_evidence_strength 
        )
    except KeyError:
        system_prompt = SYSTEM_ASSESSMENT_PROMPT  # fallback ถ้าไม่มี

    # เพิ่ม schema
    try:
        schema_json = json.dumps(CombinedAssessment.model_json_schema(), ensure_ascii=False, indent=2)
    except Exception:
        schema_json = '{"score":0,"reason":"string"}'

    system_prompt += "\n\n--- JSON SCHEMA ---\n" + schema_json + "\nIMPORTANT: Respond only with valid JSON."

    # เรียก LLM
    try:
        raw = _fetch_llm_response(system_prompt, user_prompt, _MAX_LLM_RETRIES, llm_executor=llm_executor)
        parsed = _robust_extract_json(raw)
        
        if not isinstance(parsed, dict):
            logger.error(f"Parsed result is not dict: {type(parsed)}")
            parsed = {}

        return {
            "score": int(parsed.get("score", 0)),
            "reason": parsed.get("reason", "No reason provided."),
            "is_passed": parsed.get("is_passed", False),
            "P_Plan_Score": int(parsed.get("P_Plan_Score", 0)),
            "D_Do_Score": int(parsed.get("D_Do_Score", 0)),
            "C_Check_Score": int(parsed.get("C_Check_Score", 0)),
            "A_Act_Score": int(parsed.get("A_Act_Score", 0)),
        }

    except Exception as e:
        logger.exception(f"evaluate_with_llm failed for {sub_id} L{level}: {e}")
        return {
            "score": 0,
            "reason": f"LLM error: {str(e)}",
            "is_passed": False,
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


# =================================================================
# ULTIMATE PRODUCTION: evaluate_with_llm_low_level (L1/L2 Multi-Enabler)
# =================================================================
def evaluate_with_llm_low_level(
    context: str,
    sub_criteria_name: str,
    level: int,
    statement_text: str,
    sub_id: str,
    llm_executor: Any = None,
    pdca_phase: str = "",
    level_constraint: str = "",
    must_include_keywords: str = "",
    avoid_keywords: str = "",
    max_rerank_score: float = 0.0,
    max_evidence_strength: float = 10.0,
    contextual_rules_map: Optional[Dict[str, Any]] = None,
    enabler_id: str = "KM",
    **kwargs
) -> Dict[str, Any]:
    """
    Standard Evaluation for L1/L2 using LOW_LEVEL_PROMPT (Dynamic Multi-Enabler)
    - ดึง planning_keywords จาก contextual_rules_map (จาก pea_km_contextual_rules.json)
    - ไม่ hardcode อีกต่อไป
    - ส่งค่า P/D/C/A ดิบจาก LLM → ให้ _run_single_assessment บังคับกฎ L1/L2 ขั้นสุดท้าย
    """
    
    # -------------------- 1. Setup & Context Check --------------------
    context_to_send_eval = context[:MAX_EVAL_CONTEXT_LENGTH] if context else "ไม่มีหลักฐานที่เกี่ยวข้อง"
    
    failure_result = _check_and_handle_empty_context(context, sub_id, level)
    if failure_result:
        return failure_result

    # -------------------- 2. ดึง planning_keywords จาก pea_km_contextual_rules.json --------------------
    planning_keywords = "วิสัยทัศน์, นโยบาย, ทิศทาง, เป้าหมาย"  # fallback พื้นฐาน

    if contextual_rules_map:
        # 2.1 ดึงจาก sub-criteria เฉพาะก่อน (เช่น 1.1 → L1)
        sub_rules = contextual_rules_map.get(sub_id, {})
        l1_rules = sub_rules.get("L1", {})
        if l1_rules and "planning_keywords" in l1_rules:
            planning_keywords = l1_rules["planning_keywords"]
        else:
            # 2.2 Fallback ไปใช้ _enabler_defaults (เช่น KM, DX)
            default_rules = contextual_rules_map.get("_enabler_defaults", {})
            if "planning_keywords" in default_rules:
                planning_keywords = default_rules["planning_keywords"]

    logger.debug(f"[L{level}] Using planning_keywords: {planning_keywords}")

    # -------------------- 3. Prompt Building --------------------
    try:
        # System Prompt: ใส่ planning_keywords ด้วย .format()
        system_prompt = SYSTEM_LOW_LEVEL_PROMPT.format(planning_keywords=planning_keywords)
        system_prompt += "\n\nIMPORTANT: Respond only with valid JSON."

        # User Prompt: ไม่ต้องส่ง planning_keywords (เพราะอยู่ใน system แล้ว)
        user_prompt = USER_LOW_LEVEL_PROMPT_TEMPLATE.format(
            sub_id=sub_id,
            sub_criteria_name=sub_criteria_name,
            level=level,
            statement_text=statement_text,
            level_constraint=level_constraint or "ไม่มี",
            must_include_keywords=must_include_keywords or "ไม่มี",
            avoid_keywords=avoid_keywords or "ไม่มี",
            context=context_to_send_eval
        )
    except Exception as e:
        logger.error(f"Error formatting LOW_LEVEL_PROMPT: {e}. Using fallback prompt.")
        system_prompt = SYSTEM_LOW_LEVEL_PROMPT + "\n\nIMPORTANT: Respond only with valid JSON."
        user_prompt = f"เกณฑ์: {sub_id} L{level}\nหลักฐาน: {context_to_send_eval}\nตอบ JSON เท่านั้น"

    # -------------------- 4. LLM Call --------------------
    try:
        raw = _fetch_llm_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_retries=_MAX_LLM_RETRIES,
            llm_executor=llm_executor
        )
        
        parsed = _robust_extract_json(raw)
        
        if not isinstance(parsed, dict):
            logger.error(f"LLM L{level} response parsed to non-dict: {type(parsed)}. Using empty dict.")
            parsed = {}

        # -------------------- 5. ส่งค่าดิบจาก LLM ทั้งหมด (ไม่บังคับ C/A=0 ที่นี่) --------------------
        return {
            "score": int(parsed.get("score", 0)),
            "reason": parsed.get("reason", "ไม่พบเหตุผลจาก LLM"),
            "is_passed": parsed.get("is_passed", False),
            "P_Plan_Score": int(parsed.get("P_Plan_Score", 0)),
            "D_Do_Score": int(parsed.get("D_Do_Score", 0)),
            "C_Check_Score": int(parsed.get("C_Check_Score", 0)),
            "A_Act_Score": int(parsed.get("A_Act_Score", 0)),
        }

    except Exception as e:
        logger.exception(f"evaluate_with_llm_low_level failed for {sub_id} L{level}: {e}")
        return {
            "score": 0,
            "reason": f"เกิดข้อผิดพลาดใน LLM: {str(e)}",
            "is_passed": False,
            "P_Plan_Score": 0,
            "D_Do_Score": 0,
            "C_Check_Score": 0,
            "A_Act_Score": 0,
        }
    
# ------------------------
# Summarize (FULL VERSION)
# ------------------------
def create_context_summary_llm(
    context: str, 
    sub_criteria_name: str, 
    level: int, 
    sub_id: str, 
    llm_executor: Any 
) -> Dict[str, Any]:
    """
    ใช้ LLM เพื่อสรุปเนื้อหาหลักฐานเป็นภาษาไทย และให้คำแนะนำราย Level
    รองรับการจัดการผลลัพธ์ทั้งแบบ String และ Object (LLMResult/AIMessage)
    """
    logger = logging.getLogger("AssessmentApp")

    # 0. ตรวจสอบความพร้อมของ LLM
    if llm_executor is None: 
        logger.error("LLM instance is None. Cannot summarize context.")
        return {
            "summary": "ไม่สามารถสรุปได้เนื่องจากระบบ LLM ไม่พร้อมใช้งาน",
            "suggestion_for_next_level": "โปรดตรวจสอบการเชื่อมต่อ LLM"
        }

    # 1. ตรวจสอบ Context และเตรียมข้อมูล
    # ป้องกันกรณี context เป็น None
    context_safe = context or ""
    context_limited = context_safe.strip()
    
    if not context_limited or len(context_limited) < 50:
        logger.info(f"Context too short for summarization L{level} {sub_id}. Skipping LLM call.")
        return {
            "summary": "หลักฐานที่ค้นหาได้มีข้อความสั้นเกินไปหรือไม่พบข้อความที่เกี่ยวข้องชัดเจนในระดับนี้",
            "suggestion_for_next_level": "ตรวจสอบความครบถ้วนของหลักฐานในฐานข้อมูล KM"
        }

    # Cap context เพื่อไม่ให้เกิน Token Limit (ประมาณ 4000 ตัวอักษร)
    context_to_send = context_limited[:4000] 
    next_level = min(level + 1, 5)

    # 2. ดึง Prompt Template
    from seam_prompts import USER_EVIDENCE_DESCRIPTION_TEMPLATE, SYSTEM_EVIDENCE_DESCRIPTION_PROMPT
    
    try:
        human_prompt = USER_EVIDENCE_DESCRIPTION_TEMPLATE.format(
            sub_id=f"{sub_id} - {sub_criteria_name}",
            level=level,
            next_level=next_level,
            context=context_to_send
        )
    except Exception as e:
        logger.error(f"Error formatting prompt template: {e}")
        return {"summary": "Error formatting prompt", "suggestion_for_next_level": "Check template variables"}

    system_instruction = SYSTEM_EVIDENCE_DESCRIPTION_PROMPT + "\nIMPORTANT: ตอบเป็น JSON ภาษาไทยเท่านั้น ห้ามมีคำอธิบายอื่นนอก JSON."

    # 3. เรียกใช้ LLM พร้อมจัดการ Retries และ Object Parsing
    max_retries = 2
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Generating Thai Summary for {sub_id} L{level} (Attempt {attempt})")
            
            # เรียกใช้ LLM
            raw_response_obj = llm_executor.generate(
                system=system_instruction, 
                prompts=[human_prompt]
            )

            # --- CRITICAL FIX START: ดึง String ออกจาก Object ---
            raw_response_str = ""
            if hasattr(raw_response_obj, 'generations'): # LLMResult
                raw_response_str = raw_response_obj.generations[0][0].text
            elif hasattr(raw_response_obj, 'content'):   # AIMessage
                raw_response_str = raw_response_obj.content
            else:
                raw_response_str = str(raw_response_obj)
            # --- CRITICAL FIX END ---

            # 4. Extract และ Normalize JSON
            # เรียกใช้ _extract_normalized_dict จาก core/json_extractor.py
            parsed = _extract_normalized_dict(raw_response_str)
            
            if parsed and isinstance(parsed, dict) and "summary" in parsed:
                # ทำความสะอาดข้อมูล String ขั้นสุดท้าย
                summary_val = str(parsed.get("summary", "")).strip()
                suggestion_val = str(parsed.get("suggestion_for_next_level", "")).strip()
                
                return {
                    "summary": summary_val if summary_val else "ไม่พบข้อมูลสรุป",
                    "suggestion_for_next_level": suggestion_val if suggestion_val else "ไม่พบคำแนะนำ"
                }
            
            logger.warning(f"Attempt {attempt}: LLM returned invalid summary format.")
            
        except Exception as e:
            logger.error(f"Attempt {attempt}: create_context_summary_llm failed: {str(e)}")
            time.sleep(1)

    # 5. Fallback สุดท้ายหากรันไม่สำเร็จ
    return {
        "summary": f"ระบบประเมินพบหลักฐานในระดับ {level} แต่ไม่สามารถสรุปเนื้อหาได้โดยอัตโนมัติ (LLM Parse Error)",
        "suggestion_for_next_level": f"ตรวจสอบข้อกำหนดเป้าหมายของ Level {next_level} ในคู่มือ SE-AM"
    }


# =================================================================
# 1. JSON Extractor (ทนทานที่สุด)
# =================================================================
def _extract_json_array_for_action_plan(text: Any, logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    สกัด JSON Array ออกจาก Text โดยรองรับการซ่อมแซมโครงสร้าง (Auto-Repair)
    และจัดการปัญหา Delimiter Error/Control Characters
    """
    try:
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        if not text.strip():
            return []

        # 1. ลบ Markdown Block (ถ้ามี)
        clean_text = re.sub(r'```(?:json)?\s*([\s\S]*?)\s*```', r'\1', text, flags=re.IGNORECASE).strip()

        # 2. ค้นหาขอบเขตที่กว้างที่สุดของ [ ] หรือ { }
        start_idx = clean_text.find('[')
        end_idx = clean_text.rfind(']')

        if start_idx == -1:
            # กรณี LLM ส่งมาเป็น Object เดียว (Single Phase)
            start_idx = clean_text.find('{')
            end_idx = clean_text.rfind('}')
            if start_idx == -1: return []
            json_candidate = clean_text[start_idx:end_idx + 1]
        else:
            json_candidate = clean_text[start_idx:end_idx + 1]

        # 3. ล้างอักขระควบคุม (Control Characters) ที่มักทำให้ JSON Parse พัง
        # ลบ ASCII 0-31 ยกเว้น newline, tab, carriage return
        json_candidate = "".join(char for char in json_candidate if ord(char) >= 32 or char in "\n\r\t")

        # 4. ฟังก์ชันย่อยสำหรับพยายาม Parse
        def try_parse(content):
            try:
                # json5 รองรับ Trailing Comma และ Single Quote
                data = json5.loads(content)
                return data if isinstance(data, list) else [data]
            except Exception:
                return None

        # --- ลองครั้งที่ 1: Parse ปกติ ---
        result = try_parse(json_candidate)
        if result: return result

        # --- ลองครั้งที่ 2: ซ่อมแซมเครื่องหมายคำพูด (Smart Quotes) ---
        repaired_quotes = json_candidate.replace('“', '"').replace('”', '"').replace("'", '"')
        result = try_parse(repaired_quotes)
        if result: return result

        # --- ลองครั้งที่ 3: กรณี JSON ตัดจบ (Truncated Repair) ---
        # พยายามปิด Bracket ที่ LLM เจนไม่จบ
        logger.warning("JSON truncated or malformed, attempting brute-force closure...")
        for suffix in ["]", "}", "}]", "}]}]", "}\n]"]:
            result = try_parse(json_candidate + suffix)
            if result:
                logger.info(f"✅ Auto-repaired JSON success with suffix: {suffix}")
                return result

        # --- ลองครั้งที่ 4: สุดท้าย ใช้ Regex ดึง Object ทีละตัว (Fallback) ---
        logger.warning("Falling back to Regex Object Extraction...")
        # ค้นหา pattern { ... } ที่ดูเหมือนจะเป็น object
        objects = re.findall(r'\{(?:[^{}]|(?R))*\}', json_candidate) # ต้องการ regex module พิเศษ หรือเขียนแบบง่าย:
        if not objects:
            objects = re.findall(r'\{[\s\S]*?\}', json_candidate)
            
        fallback_results = []
        for obj_str in objects:
            try:
                obj_data = json5.loads(obj_str)
                if isinstance(obj_data, dict):
                    fallback_results.append(obj_data)
            except:
                continue
        
        if fallback_results:
            logger.info(f"✅ Recovered {len(fallback_results)} objects via regex")
            return fallback_results

        logger.error(f"Failed to parse JSON. Snippet: {json_candidate[:200]}...")
        return []

    except Exception as e:
        logger.error(f"Extraction logic failed: {str(e)}", exc_info=True)
        return []

# =================================================================
# 2. Key Normalizer (ตรงกับ schema ล่าสุด)
# =================================================================

def action_plan_normalize_keys(obj: Any) -> Any:
    """
    แปลง key ให้ตรงกับ schema และบังคับประเภทข้อมูล (Data Type Enforcement)
    โดยเฉพาะ failed_level และ Step ที่ต้องเป็น Integer 100%
    """
    if isinstance(obj, list):
        return [action_plan_normalize_keys(i) for i in obj]
    
    if isinstance(obj, dict):
        field_mapping = {
            # Phase & Action level
            'phase': 'phase', 'Phase': 'phase',
            'goal': 'goal', 'Goal': 'goal',
            'actions': 'actions', 'Actions': 'actions',
            
            'statement_id': 'statement_id', 'Statement_ID': 'statement_id',
            'statement id': 'statement_id', 'title': 'statement_id', 'id': 'statement_id',
            
            'failed_level': 'failed_level', 'Failed_Level': 'failed_level',
            'failed level': 'failed_level', 'level': 'failed_level',
            
            'recommendation': 'recommendation', 'Recommendation': 'recommendation',
            'recommend': 'recommendation',
            
            'target_evidence_type': 'target_evidence_type', 'Target_Evidence_Type': 'target_evidence_type',
            'evidence_type': 'target_evidence_type', 'evidence': 'target_evidence_type',
            
            'key_metric': 'key_metric', 'Key_Metric': 'key_metric',
            'metric': 'key_metric',
            
            'steps': 'steps', 'Steps': 'steps',
            
            # StepDetail (Capitalized per schema)
            'step': 'Step', 'Step': 'Step',
            'description': 'Description', 'Description': 'Description', 'desc': 'Description',
            'responsible': 'Responsible', 'Responsible': 'Responsible', 'owner': 'Responsible',
            'tools_templates': 'Tools_Templates', 'Tools_Templates': 'Tools_Templates', 'tools': 'Tools_Templates',
            'verification_outcome': 'Verification_Outcome', 'Verification_Outcome': 'Verification_Outcome', 'outcome': 'Verification_Outcome',
        }
        
        new_obj = {}
        for k, v in obj.items():
            # ทำความสะอาด Key
            k_clean = k.lower().replace('_', ' ').replace('-', ' ').strip()
            k_no_space = k_clean.replace(' ', '')
            target_key = field_mapping.get(k_clean) or field_mapping.get(k_no_space) or k
            
            # --- [CRITICAL FIX] บังคับประเภทข้อมูลตัวเลข ---
            if target_key in ['failed_level', 'Step']:
                try:
                    if isinstance(v, str):
                        # ดึงเฉพาะตัวเลขที่เจอใน string เช่น "Level 3" -> 3
                        nums = re.findall(r'\d+', v)
                        v = int(nums[0]) if nums else 0
                    else:
                        v = int(v) if v is not None else 0
                except (ValueError, IndexError):
                    v = 0 # Fallback default
            
            new_obj[target_key] = action_plan_normalize_keys(v)
        
        return new_obj
    
    return obj


# =================================================================
# 3. Main Function: create_structured_action_plan
# =================================================================
def create_structured_action_plan(
    recommendation_statements: List[Dict[str, Any]],
    sub_id: str,
    sub_criteria_name: str,
    target_level: int,
    llm_executor: Any,
    logger: logging.Logger,
    max_retries: int = 3
) -> List[Dict[str, Any]]:
    """
    สร้าง Action Plan ที่ผ่าน Pydantic validation 100%
    ควบคุมจาก config.global_vars อย่างเต็มรูปแบบ:
    - จำนวน Phase สูงสุด
    - จำนวน Steps ต่อ Action
    - ความยาว Step (คำ)
    - ภาษา
    """
    from config import global_vars as gv

    # --- Sustain Mode (ไม่มี Gap) ---
    if not recommendation_statements:
        logger.info(f"[Sustain Mode] No gaps found → Level {target_level}")
        return [{
            "phase": f"Level {target_level} Sustain & Innovation",
            "goal": f"รักษามาตรฐานและยกระดับ {sub_criteria_name} สู่ความเป็นเลิศอย่างต่อเนื่อง",
            "actions": [{
                "statement_id": f"SUSTAIN_L{target_level}",
                "failed_level": target_level,
                "recommendation": "รักษามาตรฐานการดำเนินงาน พร้อมทำ Benchmarking กับ Best Practice สากล",
                "target_evidence_type": "Internal Audit Report / External Benchmarking Report",
                "key_metric": f"Maintain Maturity ≥ Level {target_level}",
                "steps": [{
                    "Step": "1",
                    "Description": "ทบทวนกระบวนการและ KPI รายไตรมาส พร้อมปรับปรุงตาม PDCA",
                    "Responsible": "KM Committee / Top Management",
                    "Tools_Templates": "PDCA Dashboard / Quarterly Review Template",
                    "Verification_Outcome": "Quarterly KM Review Report"
                }, {
                    "Step": "2",
                    "Description": "ศึกษาค้นคว้า Best Practices จากองค์กรชั้นนำทั้งในและต่างประเทศ",
                    "Responsible": "KM Team",
                    "Tools_Templates": "Benchmarking Framework",
                    "Verification_Outcome": "Benchmarking Study Report"
                }]
            }]
        }]

    # --- วิเคราะห์ Gap และกำหนดจำนวน Phase ตามระดับ ---
    max_failed_level = max([s.get('level', 0) for s in recommendation_statements] or [1])

    if max_failed_level >= 5:
        advice_focus = "Innovation, External Benchmarking, Digital Transformation และ Continuous Improvement"
    elif max_failed_level >= 3:
        advice_focus = "Standardization, KPI Monitoring, PDCA Cycle และ Evidence Strengthening"
    else:
        advice_focus = "Policy Establishment, Resource Allocation, Communication และ Basic Training"

    stmt_blocks = [
        f"- [Level {s.get('level')}] {s.get('statement')} (Gap: {s.get('reason')})"
        for s in recommendation_statements
    ]

    # --- เตรียม JSON Schema ---
    try:
        from core.action_plan_schema import get_clean_action_plan_schema
        schema_json = json.dumps(get_clean_action_plan_schema(), ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Schema load failed: {e}")
        return []

    # --- สร้าง Prompt พร้อมส่ง config ทั้งหมดเข้าไปบังคับ LLM ---
    human_prompt = ACTION_PLAN_PROMPT.format(
        sub_id=sub_id,
        sub_criteria_name=sub_criteria_name,
        target_level=target_level,
        advice_focus=advice_focus,
        recommendation_statements_list="\n".join(stmt_blocks),
        json_schema=schema_json,
        max_phases=gv.MAX_ACTION_PLAN_PHASES,
        max_steps=gv.MAX_STEPS_PER_ACTION,
        max_words_per_step=gv.ACTION_PLAN_STEP_MAX_WORDS,
        language="ภาษาไทย" if gv.ACTION_PLAN_LANGUAGE == "th" else "English"
    )

    # --- Retry Loop ---
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Action Plan Generation | Attempt {attempt}/{gv.OLLAMA_MAX_RETRIES}")
            response = llm_executor.generate(
                system=SYSTEM_ACTION_PLAN_PROMPT,
                prompts=[human_prompt],
                temperature=gv.LLM_TEMPERATURE,
                max_tokens=3000
            )

            raw_text = ""
            if hasattr(response, 'generations') and response.generations:
                raw_text = response.generations[0][0].text
            elif hasattr(response, 'text'):
                raw_text = response.text
            else:
                raw_text = str(response)

            if attempt == 1:
                logger.debug(f"Raw Response (first 800 chars):\n{raw_text[:800]}")

            items = _extract_json_array_for_action_plan(raw_text, logger)
            if not items:
                logger.warning(f"Attempt {attempt}: No JSON extracted")
                continue

            validated_output = []
            for idx, entry in enumerate(items):
                try:
                    clean_entry = action_plan_normalize_keys(entry)
                    validated = ActionPlanActions.model_validate(clean_entry)
                    validated_output.append(validated.model_dump(by_alias=False))
                except Exception as ve:
                    logger.error(f"Entry {idx} validation failed: {ve}")
                    if idx < 3:
                        logger.debug(f"Failed Entry:\n{json.dumps(clean_entry, ensure_ascii=False, indent=2)[:1500]}")

            if validated_output:
                logger.info(f"✅ Success: {len(validated_output)} valid phase(s) on attempt {attempt}")
                return validated_output

        except Exception as e:
            logger.error(f"Attempt {attempt} error: {e}", exc_info=True)

    # --- Emergency Fallback ---
    logger.warning("All attempts failed → returning emergency fallback plan")
    return [{
        "phase": "Phase 1: Immediate Action Required",
        "goal": f"เริ่มต้นแก้ไขช่องว่างหลักใน {sub_criteria_name} อย่างเร่งด่วน",
        "actions": [{
            "statement_id": f"GAP_L{max_failed_level}",
            "failed_level": max_failed_level,
            "recommendation": f"แต่งตั้งทีมงานและจัดทำแผนพัฒนาอย่างเป็นระบบ โดยเน้น {advice_focus}",
            "target_evidence_type": "คำสั่งแต่งตั้ง / แผนปฏิบัติการ",
            "key_metric": "จัดตั้งทีมและอนุมัติแผนภายใน 3 เดือน",
            "steps": [
                {"Step": "1", "Description": "แต่งตั้งคณะทำงานพัฒนา KM เฉพาะเกณฑ์นี้", "Responsible": "ผู้บริหารสูงสุด", "Tools_Templates": "คำสั่งแต่งตั้ง", "Verification_Outcome": "คำสั่งแต่งตั้งอย่างเป็นทางการ"},
                {"Step": "2", "Description": "จัดประชุม Kick-off และวิเคราะห์ Gap ร่วมกัน", "Responsible": "หัวหน้าทีม KM", "Tools_Templates": "Gap Analysis Template", "Verification_Outcome": "รายงานผลการประชุม"}
            ]
        }]
    }]