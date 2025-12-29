"""
llm_data_utils.py
Robust LLM + RAG utilities for SEAM assessment (CLEAN FINAL VERSION)
"""

import logging

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


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
import os
# --- เตรียม JSON Schema ---
try:
    from core.action_plan_schema import get_clean_action_plan_schema
    schema_json = json.dumps(get_clean_action_plan_schema(), ensure_ascii=False, indent=2)
except Exception as e:
    logger.error(f"Schema load failed: {e}")


# Optional: regex แทน re (ดีกว่า) — ถ้าไม่มีก็ใช้ re ธรรมดา
try:
    import regex as re  # type: ignore
except ImportError:
    pass  # ใช้ re มาตรฐานต่อไป

# ===================================================================
# 1. Core Configuration (ต้องมีแน่นอน)
# ===================================================================
from config.global_vars import (
    DEFAULT_ENABLER,
    INITIAL_TOP_K,
    MAX_EVAL_CONTEXT_LENGTH,
    USE_HYBRID_SEARCH, 
    HYBRID_VECTOR_WEIGHT, 
    HYBRID_BM25_WEIGHT,
    MAX_ACTION_PLAN_PHASES,
    MAX_STEPS_PER_ACTION,
    ACTION_PLAN_STEP_MAX_WORDS,
    ACTION_PLAN_LANGUAGE,
    QUERY_INITIAL_K
)

# ===================================================================
# 2. Critical Utilities (ต้องมีจริง — ไม่มี fallback)
# ===================================================================
# 🎯 FIX 1: เปลี่ยน Import จาก _get_collection_name ไปเป็น get_doc_type_collection_key
from core.vectorstore import get_hf_embeddings
from utils.path_utils import (
    get_doc_type_collection_key, 
    _n, get_mapping_file_path, # <--- นำเข้าฟังก์ชันใหม่จาก utils/path_utils
    get_vectorstore_collection_path,
    get_vectorstore_tenant_root_path,
    get_rubric_file_path
)
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
    USER_LOW_LEVEL_PROMPT_TEMPLATE,
    USER_EVIDENCE_DESCRIPTION_TEMPLATE,
    EXCELLENCE_ADVICE_PROMPT, 
    SYSTEM_EXCELLENCE_PROMPT,
    SYSTEM_QUALITY_PROMPT,
    QUALITY_REFINEMENT_PROMPT
)

from core.vectorstore import VectorStoreManager, get_global_reranker, ChromaRetriever
from core.assessment_schema import CombinedAssessment, EvidenceSummary
from core.action_plan_schema import ActionPlanActions, ActionPlanResult

try:
    from core.assessment_schema import StatementAssessment
except ImportError:
    from pydantic import BaseModel
    class StatementAssessment(BaseModel):
        score: int = 0
        reason: str = ""

from langchain_core.documents import Document as LcDocument
# แก้ไขบรรทัด Import ใน routers/llm_router.py

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

# แก้ไขใน core/llm_data_utils.py

def _create_where_filter(
    stable_doc_ids: Optional[Union[Set[str], List[str]]] = None,
    subject: Optional[str] = None,
    sub_topic: Optional[str] = None,  # 👈 เพิ่มบรรทัดนี้เพื่อรองรับค่าที่ส่งมา
    year: Optional[Union[int, str]] = None,
    enabler: Optional[str] = None,
    **kwargs  # 👈 หรือเพิ่ม **kwargs ไว้เพื่อกันเหนียวในอนาคต
) -> Dict[str, Any]:
    """
    สร้าง Filter สำหรับ ChromaDB ที่ยืดหยุ่น:
    """
    filters: List[Dict[str, Any]] = []

    # --- 1. การจัดการ Stable Doc IDs (ลำดับความสำคัญสูงสุด) ---
    if stable_doc_ids:
        ids_list = [str(i).strip() for i in (stable_doc_ids if isinstance(stable_doc_ids, (list, set)) else [stable_doc_ids]) if i]
        if ids_list:
            if len(ids_list) == 1:
                return {"stable_doc_uuid": ids_list[0]}
            else:
                return {"stable_doc_uuid": {"$in": ids_list}}

    # --- 2. การจัดการ Metadata อื่นๆ ---
    if year and str(year).strip():
        filters.append({"year": str(year).strip()})
    
    if enabler and str(enabler).strip():
        filters.append({"enabler": enabler.strip().upper()})

    if subject and str(subject).strip():
        filters.append({"subject": str(subject).strip()})

    # --- 3. การจัดการ sub_topic (ถ้ามีส่งมา) ---
    if sub_topic and str(sub_topic).strip():
        filters.append({"sub_topic": str(sub_topic).strip()})

    if not filters:
        return {}

    return filters[0] if len(filters) == 1 else {"$and": filters}

def retrieve_context_for_endpoint(
    vectorstore_manager,
    query: str = "",
    tenant: Optional[str] = None,
    year: Optional[Union[int, str]] = None,
    stable_doc_ids: Optional[Set[str]] = None,
    doc_type: Optional[str] = None,
    enabler: Optional[str] = None,
    subject: Optional[str] = None,
    sub_topic: Optional[str] = None,
    k_to_retrieve: int = 150, # 🚀 ดึงมาเยอะเพื่อให้ Reranker มีวัตถุดิบ
    k_to_rerank: int = 30,    # 🚀 จำนวนสุดท้ายที่จะส่งให้ LLM
    strict_filter: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    [REVISED] Retrieval with Anchor Support, Content-Based Dedup, and Batch Reranking.
    """
    start_time = time.time()
    vsm = vectorstore_manager

    # 1. Resolve collection & Check existence
    clean_doc_type = str(doc_type or "document").strip().lower()
    collection_name = get_doc_type_collection_key(doc_type=clean_doc_type, enabler=enabler)
    
    chroma = vsm._load_chroma_instance(collection_name)
    if not chroma:
        logger.error(f"❌ Collection {collection_name} not found.")
        return {"top_evidences": [], "aggregated_context": "ไม่พบฐานข้อมูล", "retrieval_time": 0}

    # 2. Create where_filter (ID Filter is CRITICAL for Level 5 accuracy)
    where_filter = _create_where_filter(
        stable_doc_ids=stable_doc_ids, subject=subject, sub_topic=sub_topic, year=year
    )

    # 🎯 ใช้ Dictionary เก็บเพื่อทำ Deduplication ด้วย Content Hash
    # ป้องกันกรณีที่ ID ต่างกันแต่เนื้อหาซ้ำ หรือ ID ซ้ำแต่เนื้อหาต่าง
    unique_map: Dict[str, LcDocument] = {}

    # =====================================================
    # ⚓ 2.1 ANCHOR RETRIEVAL (Fetching Structure/Table of Contents)
    # =====================================================
    if stable_doc_ids:
        logger.info(f"⚓ Fetching Anchor Chunks for structure from {len(stable_doc_ids)} files...")
        # ดึงหน้าแรกๆ ของไฟล์มาเป็นบริบทพื้นฐาน
        anchors = chroma.get(where=where_filter, limit=10) 
        if anchors and anchors.get('documents'):
            for i in range(len(anchors['documents'])):
                content = anchors['documents'][i]
                md = anchors['metadatas'][i]
                # สร้าง UID จากเนื้อหา
                content_hash = str(hash(content))
                uid = md.get("chunk_uuid") or f"anchor-{content_hash}"
                
                if uid not in unique_map:
                    unique_map[uid] = LcDocument(
                        page_content=content,
                        metadata={**md, "score": 0.9, "is_anchor": True}
                    )

    # =====================================================
    # 🔍 2.2 SEMANTIC SEARCH
    # =====================================================
    search_query = query if (query and query != "*" and len(query) > 2) else ""
    
    if search_query:
        docs = chroma.similarity_search(search_query, k=k_to_retrieve, filter=where_filter)
        for d in docs:
            content_hash = str(hash(d.page_content))
            uid = d.metadata.get("chunk_uuid") or content_hash
            if uid not in unique_map:
                unique_map[uid] = d
    elif not unique_map: 
        # Fallback กรณีไม่มี Query ให้ดึงแบบกวาดตาม Filter
        docs = chroma.similarity_search("*", k=k_to_retrieve, filter=where_filter)
        for d in docs:
            content_hash = str(hash(d.page_content))
            uid = d.metadata.get("chunk_uuid") or content_hash
            if uid not in unique_map:
                unique_map[uid] = d

    candidates = list(unique_map.values())

    # 🎯 Double Check Guardrail: กรองเฉพาะไฟล์ที่ระบุ (ถ้ามี)
    if stable_doc_ids:
        target_ids = {str(i).lower() for i in stable_doc_ids}
        candidates = [
            d for d in candidates 
            if str(d.metadata.get("stable_doc_uuid") or d.metadata.get("doc_id")).lower() in target_ids
        ]

    # =====================================================
    # 🚀 3. BATCH RERANKING (ป้องกัน CUDA OOM)
    # =====================================================
    final_chunks = []
    reranker = get_global_reranker()
    
    if reranker and candidates and search_query:
        try:
            batch_size = 100 # แบ่ง Batch เพื่อความปลอดภัยของ VRAM
            scored_candidates = []
            
            logger.info(f"🚀 Reranking {len(candidates)} candidates in batches...")
            for i in range(0, len(candidates), batch_size):
                batch = candidates[i : i + batch_size]
                # ทำ Rerank ทีละชุด
                reranked_batch = reranker.compress_documents(documents=batch, query=search_query)
                scored_candidates.extend(reranked_batch)
            
            # Sort ตามคะแนน และเลือกเอาตัวท็อป
            scored_candidates = sorted(
                scored_candidates, 
                key=lambda x: getattr(x, "relevance_score", 0), 
                reverse=True
            )
            
            # เลือกเฉพาะ k_to_rerank ตัวแรก
            for res in scored_candidates[:k_to_rerank]:
                doc = res if isinstance(res, LcDocument) else res.document
                score = getattr(res, "relevance_score", 0)
                doc.metadata["rerank_score"] = score
                final_chunks.append(doc)
                
        except Exception as e:
            logger.error(f"⚠️ Rerank failed: {e}")
            final_chunks = candidates[:k_to_rerank]
    else:
        final_chunks = candidates[:k_to_rerank]

    # =====================================================
    # 4. RESPONSE BUILD
    # =====================================================
    top_evidences = []
    aggregated_parts = []
    
    for doc in final_chunks:
        md = doc.metadata or {}
        text = doc.page_content.strip()
        s_uuid = md.get("stable_doc_uuid") or md.get("doc_id")
        p_val = md.get("page_label") or md.get("page_number") or md.get("page") or "N/A"
        
        # 🎯 Sync Score ให้ลงตัวแปรที่หลากหลายเพื่อให้ Logic ขั้นต่อไปหาเจอ
        score = md.get("rerank_score") or md.get("score") or 0.0
        
        top_evidences.append({
            "doc_id": s_uuid,
            "chunk_uuid": md.get("chunk_uuid"),
            "source": md.get("source") or md.get("file_name") or "Unknown",
            "text": text,
            "page": str(p_val),
            "score": score,
            "pdca_tag": md.get("pdca_tag", "Other"),
            "metadata": md # แนบ Metadata ตัวเต็มไปด้วยเพื่อความ Robust
        })
        
        source_name = md.get('source') or md.get('file_name') or 'Unknown'
        aggregated_parts.append(f"[ไฟล์: {source_name}, หน้า: {p_val}] {text}")

    retrieval_time = round(time.time() - start_time, 3)
    logger.info(f"🏁 Finished: {len(top_evidences)} chunks in {retrieval_time}s")

    return {
        "top_evidences": top_evidences,
        "aggregated_context": "\n\n".join(aggregated_parts) if aggregated_parts else "ไม่พบข้อมูลที่เกี่ยวข้อง",
        "retrieval_time": retrieval_time,
        "used_chunk_uuids": [e["chunk_uuid"] for e in top_evidences if e.get("chunk_uuid")]
    }

# ------------------------
# Retrieval: retrieve_context_with_filter (Revised)
# ------------------------

def retrieve_context_with_filter(
    query: Union[str, List[str]],
    doc_type: str,
    tenant: Optional[str] = None,
    year: Optional[Union[int, str]] = None,
    enabler: Optional[str] = None,
    subject: Optional[str] = None,
    vectorstore_manager: Optional[Any] = None, 
    mapped_uuids: Optional[List[str]] = None,
    stable_doc_ids: Optional[List[str]] = None,
    priority_docs_input: Optional[List[Any]] = None,
    sequential_chunk_uuids: Optional[List[str]] = None,
    sub_id: Optional[str] = None,
    level: Optional[int] = None,
    get_previous_level_docs: Optional[Callable[[int, str], List[Any]]] = None,
    top_k: int = 150, # 🚀 แนะนำให้ใช้ 100-200 เพื่อให้ Reranker มีตัวเลือก
) -> Dict[str, Any]:
    """
    [FINAL ROBUST VERSION] Retrieval + Deduplication + Batch Reranking
    แก้ไขปัญหาหลักฐานหาย (Deduplication Fix) และป้องกัน CUDA OOM (Batch Fix)
    """
    start_time = time.time()
    
    # 1. Setup Manager & Configuration
    manager = vectorstore_manager
    queries_to_run = [query] if isinstance(query, str) else list(query or [""])
    collection_name = get_doc_type_collection_key(doc_type, enabler or "KM")
    
    # 2. จัดการ Filter (Target IDs)
    target_ids = set()
    if stable_doc_ids: target_ids.update([str(i) for i in stable_doc_ids])
    if mapped_uuids: target_ids.update([str(i) for i in mapped_uuids])
    if sequential_chunk_uuids: target_ids.update([str(i) for i in sequential_chunk_uuids])
    
    where_filter = _create_where_filter(
        stable_doc_ids=list(target_ids) if target_ids else None,
        subject=subject,
        year=year
    )

    # 3. รวบรวมข้อมูลเริ่มต้น (Base Chunks)
    all_source_chunks = []

    # 3.1 Priority Docs (จาก Baseline หรือ Level ก่อนหน้า)
    if priority_docs_input:
        for doc in priority_docs_input:
            if not doc: continue
            if isinstance(doc, dict):
                pc = doc.get('page_content') or doc.get('text') or ''
                meta = doc.get('metadata') or {}
                meta['chunk_uuid'] = doc.get('chunk_uuid') or meta.get('chunk_uuid')
                meta['stable_doc_uuid'] = doc.get('doc_id') or meta.get('stable_doc_uuid')
                if pc.strip():
                    all_source_chunks.append(LcDocument(page_content=pc, metadata=meta))
            elif hasattr(doc, 'page_content'):
                all_source_chunks.append(doc)

    # 3.2 L3 Fallback (ดึงจาก L2)
    if level == 3 and callable(get_previous_level_docs):
        try:
            fallback_chunks = get_previous_level_docs(level - 1, sub_id) or []
            all_source_chunks.extend(fallback_chunks)
            logger.info(f"L3 Fallback: Added {len(fallback_chunks)} chunks from L2")
        except Exception as e:
            logger.warning(f"L3 Fallback failed: {e}")

    # 3.3 Vector Search Retrieval (ดึงจาก ChromaDB/Pinecone)
    try:
        full_retriever = manager.get_retriever(collection_name=collection_name)
        # ดึง Base Retriever เพื่อดึง Chunk จำนวนมากมา Rerank เอง
        base_retriever = getattr(full_retriever, "base_retriever", full_retriever)
        
        search_kwargs = {"k": top_k} 
        if where_filter: search_kwargs["where"] = where_filter

        for q in queries_to_run:
            if not q: continue
            docs = base_retriever.invoke(q, config={"configurable": {"search_kwargs": search_kwargs}})
            all_source_chunks.extend(docs or [])
    except Exception as e:
        logger.error(f"Retrieval error for {collection_name}: {e}")

    # 4. Deduplicate (CRITICAL FIX: ป้องกันข้อมูลหายจาก ID ซ้ำ)
    unique_map: Dict[str, LcDocument] = {}
    for doc in all_source_chunks:
        if not doc or not doc.page_content.strip(): continue
        md = doc.metadata or {}
        
        # 🎯 ใช้ Hash เนื้อหาผสมกับ ID เพื่อให้มั่นใจว่า Chunk ในไฟล์เดียวกันไม่โดนยุบรวมกัน
        content_hash = str(hash(doc.page_content))
        uid = str(md.get("chunk_uuid") or f"{md.get('stable_doc_uuid', 'unknown')}-{content_hash}")

        if uid not in unique_map:
            if level == 3:
                doc.page_content = doc.page_content[:1200] # เพิ่มความยาวเล็กน้อยให้ L3
            unique_map[uid] = doc

    candidates = list(unique_map.values())

    # 5. [BATCH RERANKING] ป้องกัน OOM และจัดลำดับความสำคัญ
    final_scored_docs = []
    batch_size = 150 # ค่าที่ปลอดภัยสำหรับ GPU 
    
    # พยายามดึง Reranker จาก Manager หรือ Global
    reranker_compressor = getattr(manager, "reranker", None)

    if reranker_compressor and len(candidates) > 0:
        logger.info(f"🚀 Batch Reranking {len(candidates)} chunks in batches of {batch_size}")
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i : i + batch_size]
            try:
                # รัน Reranker ราย Batch
                scored_batch = reranker_compressor.compress_documents(batch, queries_to_run[0])
                final_scored_docs.extend(scored_batch)
            except Exception as e:
                logger.error(f"Rerank Batch Error at index {i}: {e}")
                final_scored_docs.extend(batch)
    else:
        final_scored_docs = candidates

    # 6. Sorting & Final Formatting
    # เรียงลำดับตามคะแนน Rerank (พยายามดึงจากทุก Key ที่เป็นไปได้)
    def get_score(d):
        m = d.metadata or {}
        return float(getattr(d, "relevance_score", m.get("relevance_score", m.get("score", 0.0))))

    final_scored_docs = sorted(final_scored_docs, key=get_score, reverse=True)

    # คัดเลือกเฉพาะ K ตัวแรกที่ต้องการ (เช่น 12-15)
    final_k = 15 
    top_evidences = []
    aggregated_parts = []
    used_uuids = []
    VALID_ID = re.compile(r"^[0-9a-f\-]{36}$|^[0-9a-f]{64}$", re.IGNORECASE)

    for doc in final_scored_docs[:final_k]:
        md = doc.metadata or {}
        text = doc.page_content.strip()
        score = get_score(doc)

        # จัดการ IDs สำหรับเชื่อมโยงไฟล์
        c_uuid = str(md.get("chunk_uuid", ""))
        s_uuid = str(md.get("stable_doc_uuid") or md.get("doc_id") or "")
        best_id = s_uuid if VALID_ID.match(s_uuid) else (c_uuid if VALID_ID.match(c_uuid) else f"temp-{uuid.uuid4().hex[:8]}")
        
        if not best_id.startswith("temp-"): used_uuids.append(best_id)
            
        source = md.get("source") or md.get("source_filename") or "Unknown"
        pdca = md.get("pdca_tag", "Other")
        page = str(md.get("page_label") or md.get("page_number") or md.get("page") or "N/A")
        
        # 🎯 Sync คะแนนกลับเข้าไปใน Metadata ทุก Key เพื่อให้ _run_single_assessment หาเจอ
        md["score"] = score
        md["relevance_score"] = score
        md["rerank_score"] = score

        top_evidences.append({
            "doc_id": s_uuid or best_id,
            "chunk_uuid": c_uuid or best_id,
            "source": source,
            "text": text,
            "page": page,
            "pdca_tag": pdca,
            "score": score,
            "metadata": md # ส่ง metadata กลับไปให้ครบ
        })
        aggregated_parts.append(f"[{pdca}] [ไฟล์: {source} หน้า: {page}] {text}")

    return {
        "top_evidences": top_evidences,
        "aggregated_context": "\n\n---\n\n".join(aggregated_parts) if aggregated_parts else "ไม่พบหลักฐาน",
        "retrieval_time": round(time.time() - start_time, 3),
        "used_chunk_uuids": list(set(used_uuids))
    }


# =====================================================================
# 🛠 Helper: check_rubric_readiness (ปรับปรุงให้เงียบลง)
# =====================================================================
def is_rubric_ready(tenant: str) -> bool:
    """ ตรวจสอบการมีอยู่ของ seam collection โดยไม่พ่น Warning กวนใจ """
    if not tenant:
        return False
    
    tenant_vs_root = get_vectorstore_tenant_root_path(tenant)
    chroma_path = os.path.join(tenant_vs_root, "seam")
    
    # ส่งคืนค่า True/False เงียบๆ เพื่อให้ระบบตัดสินใจว่าจะดึง Rubric หรือไม่
    return os.path.exists(chroma_path)


# =====================================================================
# 🚀 Ultimate Version: retrieve_context_with_rubric (FIXED & REVISED)
# =====================================================================
def retrieve_context_with_rubric(
    vectorstore_manager,
    query: str,
    doc_type: str,
    enabler: Optional[str] = None,
    stable_doc_ids: Optional[Set[str]] = None,
    tenant: Optional[str] = None,
    year: Optional[Union[int, str]] = None,
    subject: Optional[str] = None,
    rubric_vectorstore_name: str = "seam", 
    top_k: int = 150,         # 🚀 เพิ่มจำนวนดึงเบื้องต้นเพื่อให้ Reranker มีตัวเลือกมากขึ้น
    rubric_top_k: int = 15,  
    strict_filter: bool = True,
    k_to_rerank: int = 30    
) -> Dict[str, Any]:
    """
    [REVISED VERSION] รองรับ Rubric + Evidence Retrieval พร้อมระบบ Batch Reranking 
    และ Content-Based Deduplication เพื่อป้องกันคะแนน Level 5 หาย
    """
    start_time = time.time()
    vsm = vectorstore_manager

    # --- 1. ตรวจสอบและสลับ Collection ---
    if hasattr(vsm, 'doc_type') and vsm.doc_type != doc_type:
        logger.info(f"🔄 Switching VSM doc_type to: {doc_type}")
        vsm.close()
        vsm.__init__(tenant=tenant, year=year, doc_type=doc_type, enabler=enabler)

    evidence_collection = get_doc_type_collection_key(doc_type, enabler or "KM")
    
    rubric_results = []
    # ใช้ Dict เพื่อทำ Deduplication ด้วย Content Hash ป้องกันหน้าซ้ำแต่ ID ต่างกัน
    unique_evidence_map: Dict[str, LcDocument] = {}

    # --- 2. การดึง Rubrics (เกณฑ์มาตรฐาน) ---
    try:
        rubric_chroma = vsm._load_chroma_instance(rubric_vectorstore_name)
        if rubric_chroma:
            rubric_query = f"เกณฑ์ SE-AM {enabler} {subject or ''}: {query}"
            r_docs = rubric_chroma.similarity_search(rubric_query, k=rubric_top_k)
            for rd in r_docs:
                rubric_results.append({
                    "text": rd.page_content, 
                    "metadata": rd.metadata, 
                    "is_rubric": True
                })
    except Exception as e:
        logger.warning(f"⚠️ Rubric Retrieval Error: {e}")

    # --- 3. การดึง Evidence (หลักฐานการดำเนินงาน) ---
    try:
        evidence_chroma = vsm._load_chroma_instance(evidence_collection)
        if not evidence_chroma:
            return {"top_evidences": [], "rubric_context": rubric_results, "retrieval_time": 0}

        where_filter = None
        if stable_doc_ids:
            ids_list = [str(i).strip().lower() for i in stable_doc_ids if i]
            where_filter = {"stable_doc_uuid": ids_list[0]} if len(ids_list) == 1 else {"stable_doc_uuid": {"$in": ids_list}}
            
            # ⚓ 3.1 Fetch Anchor Chunks (ส่วนสำคัญของไฟล์ เช่น หน้า 1-5)
            anchors = evidence_chroma.get(where=where_filter, limit=10)
            if anchors and anchors.get('documents'):
                for i in range(len(anchors['documents'])):
                    content = anchors['documents'][i]
                    md = anchors['metadatas'][i]
                    content_hash = str(hash(content))
                    uid = md.get("chunk_uuid") or f"anchor-{content_hash}"
                    
                    if uid not in unique_evidence_map:
                        unique_evidence_map[uid] = LcDocument(
                            page_content=content,
                            metadata={**md, "score": 0.95, "is_anchor": True}
                        )

        # 🔍 3.2 Semantic Search (ค้นหาตามความหมาย)
        search_results = evidence_chroma.similarity_search(query, k=top_k, filter=where_filter)
        for d in search_results:
            content_hash = str(hash(d.page_content))
            uid = d.metadata.get("chunk_uuid") or content_hash
            if uid not in unique_evidence_map:
                unique_evidence_map[uid] = d

        candidates = list(unique_evidence_map.values())

        # --- 4. BATCH RERANKING (ป้องกัน OOM และเพิ่มความแม่นยำ) ---
        evidence_results = []
        reranker = get_global_reranker()
        
        if reranker and candidates and query:
            try:
                batch_size = 100 # 🚀 แบ่ง Batch เพื่อความปลอดภัยของ VRAM
                scored_candidates = []
                
                logger.info(f"🚀 Batch Reranking {len(candidates)} chunks...")
                for i in range(0, len(candidates), batch_size):
                    batch = candidates[i : i + batch_size]
                    reranked_batch = reranker.compress_documents(documents=batch, query=query)
                    scored_candidates.extend(reranked_batch)
                
                # เรียงลำดับคะแนนรวมจากทุก Batch
                scored_candidates = sorted(
                    scored_candidates, 
                    key=lambda x: getattr(x, "relevance_score", 0), 
                    reverse=True
                )
                
                for r in scored_candidates[:k_to_rerank]:
                    doc = r.document if hasattr(r, "document") else r
                    m = doc.metadata or {}
                    score = getattr(r, "relevance_score", 0.0)
                    
                    # 🎯 Sync Score กลับเข้า Metadata เพื่อให้โมเดลประเมินใช้งานได้
                    m["rerank_score"] = score
                    m["score"] = score
                    
                    evidence_results.append({
                        "text": doc.page_content,
                        "source_filename": m.get("source_filename") or m.get("source") or "Evidence",
                        "page_label": str(m.get("page_label") or m.get("page_number") or m.get("page") or "N/A"),
                        "doc_id": m.get("stable_doc_uuid") or m.get("doc_id"),
                        "chunk_uuid": m.get("chunk_uuid") or str(uuid.uuid4()),
                        "pdca_tag": m.get("pdca_tag") or "Content",
                        "rerank_score": score,
                        "is_evidence": True,
                        "metadata": m
                    })
            except Exception as e:
                logger.error(f"⚠️ Rerank failed: {e}")
                candidates = candidates[:k_to_rerank] # Fallback
        
        # กรณีไม่มี Reranker หรือ Error ให้ใช้ลำดับปกติ
        if not evidence_results:
            for d in candidates[:k_to_rerank]:
                m = d.metadata or {}
                evidence_results.append({
                    "text": d.page_content,
                    "source_filename": m.get("source_filename") or m.get("source") or "Evidence",
                    "page_label": str(m.get("page_label") or m.get("page_number") or m.get("page") or "N/A"),
                    "doc_id": m.get("stable_doc_uuid") or m.get("doc_id"),
                    "chunk_uuid": m.get("chunk_uuid") or str(uuid.uuid4()),
                    "pdca_tag": m.get("pdca_tag") or "Content",
                    "rerank_score": 0.0,
                    "is_evidence": True,
                    "metadata": m
                })

    except Exception as e:
        logger.error(f"❌ Evidence Retrieval Error: {e}", exc_info=True)

    retrieval_time = round(time.time() - start_time, 3)
    logger.info(f"✅ Success: Retrieved {len(evidence_results)} evidence chunks in {retrieval_time}s")

    return {
        "top_evidences": evidence_results,
        "rubric_context": rubric_results,
        "retrieval_time": retrieval_time,
        "used_chunk_uuids": [e["chunk_uuid"] for e in evidence_results if e.get("chunk_uuid")]
    }

# ========================
#  retrieve_context_by_doc_ids (สำหรับ hydration ใน router)
# ========================
def retrieve_context_by_doc_ids(
    doc_uuids: List[str],
    doc_type: str,
    enabler: Optional[str] = None,
    vectorstore_manager = None,
    limit: int = 100, # 🚀 เพิ่ม limit เพื่อให้ดึงข้อมูลได้ครอบคลุมมากขึ้น
    tenant: Optional[str] = None,
    year: Optional[Union[int, str]] = None,
) -> Dict[str, Any]:
    """
    [REVISED] ดึง chunks จาก stable_doc_uuid หลายตัว (ใช้ตอน hydration sources)
    รองรับการระบุปีเพื่อหา Collection ที่ถูกต้อง และรักษา Metadata ให้ครบถ้วน
    """
    start_time = time.time()
    vsm = vectorstore_manager or VectorStoreManager(tenant=tenant, year=year)
    
    # Resolve collection name ให้ถูกต้องตามโครงสร้างปีและ enabler
    collection_name = get_doc_type_collection_key(doc_type=doc_type, enabler=enabler)

    chroma = vsm._load_chroma_instance(collection_name)
    if not chroma:
        logger.error(f"❌ Collection {collection_name} not found for hydration")
        return {"top_evidences": []}

    if not doc_uuids:
        return {"top_evidences": []}

    logger.info(f"💧 Hydration → {len(doc_uuids)} doc IDs from {collection_name}")

    try:
        # ใช้ Metadata filter ดึง chunks ทั้งหมดที่อยู่ในลิสต์ ID ที่เลือก
        # ปรับเป็น list comprehension เพื่อความชัวร์ของ Type
        ids_to_query = [str(u) for u in doc_uuids if u]
        
        results = chroma._collection.get(
            where={"stable_doc_uuid": {"$in": ids_to_query}},
            limit=limit,
            include=["documents", "metadatas"]
        )
    except Exception as e:
        logger.error(f"⚠️ Hydration query failed: {e}")
        return {"top_evidences": []}

    evidences = []
    # 🎯 ใช้ Content-based Deduplication เบื้องต้นกันหน้าซ้ำ
    seen_contents = set()

    for doc_content, meta in zip(results.get("documents", []), results.get("metadatas", [])):
        if not doc_content or not doc_content.strip():
            continue
            
        content_hash = str(hash(doc_content))
        if content_hash in seen_contents:
            continue
        seen_contents.add(content_hash)

        p_val = meta.get("page_label") or meta.get("page_number") or meta.get("page") or "N/A"
        
        # 🎯 Sync Score หลอก (เพราะ Hydration ไม่ได้ผ่าน Rerank แต่ต้องมีเพื่อให้ Logic อื่นไม่พัง)
        score = meta.get("score") or meta.get("rerank_score") or 0.85

        evidences.append({
            "doc_id": meta.get("stable_doc_uuid") or meta.get("doc_id"),
            "chunk_uuid": meta.get("chunk_uuid"),
            "source": meta.get("source") or meta.get("source_filename") or "Unknown",
            "page": str(p_val),
            "text": doc_content.strip(),
            "pdca_tag": meta.get("pdca_tag", "Other"),
            "score": score,
            "metadata": meta # แนบตัวเต็มไว้เสมอ
        })

    logger.info(f"✅ Hydration success: {len(evidences)} chunks from {len(doc_uuids)} docs")
    return {
        "top_evidences": evidences,
        "retrieval_time": round(time.time() - start_time, 3)
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
            max_evidence_strength=max_evidence_strength,
            target_score_threshold=kwargs.get("target_score_threshold", 2)
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
    [REVISED v21.2] Standard Evaluation for L1/L2 (Low Level)
    - รองรับ Float scores และแก้ปัญหาคะแนนหายจากการปัดเศษ
    - ส่งค่าน้ำหนักหลักฐาน (Rerank/Strength) เข้า Prompt
    """
    
    # -------------------- 1. Setup & Context Check --------------------
    # ใช้ฟังก์ชันตัด Context ตาม Level เพื่อลด Token และเน้นความแม่นยำ
    context_to_send_eval = _get_context_for_level(context, level)
    
    failure_result = _check_and_handle_empty_context(context, sub_id, level)
    if failure_result:
        return failure_result

    # -------------------- 2. ดึง plan_keywords (Dynamic Logic) --------------------
    plan_keywords = "วิสัยทัศน์, นโยบาย, ทิศทาง, เป้าหมาย"
    if contextual_rules_map:
        sub_rules = contextual_rules_map.get(sub_id, {})
        l1_rules = sub_rules.get("L1", {})
        if l1_rules and "plan_keywords" in l1_rules:
            plan_keywords = l1_rules["plan_keywords"]
        else:
            default_rules = contextual_rules_map.get("_enabler_defaults", {})
            if "plan_keywords" in default_rules:
                plan_keywords = default_rules["plan_keywords"]

    # -------------------- 3. Prompt Building --------------------
    try:
        # System Prompt สำหรับ L1-L2 โดยเฉพาะ
        system_prompt = SYSTEM_LOW_LEVEL_PROMPT.format(
            plan_keywords=plan_keywords,
            avoid_keywords=avoid_keywords or "ไม่มี"
        )
        system_prompt += "\n\nIMPORTANT: Respond only with valid JSON."

        # User Prompt พร้อมระบุน้ำหนักหลักฐาน
        user_prompt = USER_LOW_LEVEL_PROMPT_TEMPLATE.format(
            sub_id=sub_id,
            sub_criteria_name=sub_criteria_name,
            level=level,
            statement_text=statement_text,
            level_constraint=level_constraint or "ไม่มี",
            must_include_keywords=must_include_keywords or "ไม่มี",
            avoid_keywords=avoid_keywords or "ไม่มี",
            context=context_to_send_eval,
            # เพิ่มเพื่อให้ AI ทราบ Quality ของ Retrieval
            max_rerank_score=f"{max_rerank_score:.4f}",
            max_evidence_strength=f"{max_evidence_strength:.1f}"
        )

    except Exception as e:
        logger.error(f"Error formatting LOW_LEVEL_PROMPT: {e}")
        # Robust Fallback Prompt
        system_prompt = f"{SYSTEM_LOW_LEVEL_PROMPT}\n\nIMPORTANT: Respond only with valid JSON."
        user_prompt = f"Sub-ID: {sub_id} Level: {level}\nเกณฑ์: {sub_criteria_name}\nคำถาม: {statement_text}\nหลักฐาน: {context_to_send_eval}"

    # -------------------- 4. LLM Execution --------------------
    try:
        raw = _fetch_llm_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_retries=_MAX_LLM_RETRIES,
            llm_executor=llm_executor
        )
        
        parsed = _robust_extract_json(raw)
        
        if not isinstance(parsed, dict):
            logger.error(f"LLM L{level} response parsed to non-dict: {type(parsed)}")
            parsed = {}

        # -------------------- 5. Return Results (Float Stability) --------------------
        # แก้ไขจาก int() เป็น float() เพื่อรักษาทศนิยมของคะแนน
        return {
            "score": float(parsed.get("score", 0.0)),
            "reason": parsed.get("reason", "ไม่พบเหตุผลจาก AI"),
            "is_passed": parsed.get("is_passed", False),
            "P_Plan_Score": float(parsed.get("P_Plan_Score", 0.0)),
            "D_Do_Score": float(parsed.get("D_Do_Score", 0.0)),
            "C_Check_Score": float(parsed.get("C_Check_Score", 0.0)),
            "A_Act_Score": float(parsed.get("A_Act_Score", 0.0)),
        }

    except Exception as e:
        logger.exception(f"evaluate_with_llm_low_level failed: {e}")
        return {
            "score": 0.0,
            "reason": f"เกิดข้อผิดพลาดในการประมวลผล: {str(e)}",
            "is_passed": False,
            "P_Plan_Score": 0.0,
            "D_Do_Score": 0.0,
            "C_Check_Score": 0.0,
            "A_Act_Score": 0.0,
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
    logger = logging.getLogger("AssessmentApp")

    if llm_executor is None: 
        return {
            "summary": "ไม่สามารถสรุปได้เนื่องจากระบบ LLM ไม่พร้อมใช้งาน",
            "suggestion_for_next_level": "โปรดตรวจสอบการเชื่อมต่อ LLM"
        }

    context_safe = context or ""
    context_limited = context_safe.strip()
    
    if not context_limited or len(context_limited) < 50:
        return {
            "summary": "หลักฐานที่ค้นหาได้มีข้อความสั้นเกินไปหรือไม่พบข้อความที่เกี่ยวข้องชัดเจน",
            "suggestion_for_next_level": "ตรวจสอบความครบถ้วนของหลักฐานในฐานข้อมูล"
        }

    # Cap context ให้เหมาะสมกับสถาปัตยกรรม Model (4000-8000 chars)
    context_to_send = context_limited[:6000] 
    next_level = min(level + 1, 5)

    try:
        human_prompt = USER_EVIDENCE_DESCRIPTION_TEMPLATE.format(
            sub_id=f"{sub_id} - {sub_criteria_name}",
            level=level,
            next_level=next_level,
            context=context_to_send
        )
    except Exception as e:
        logger.error(f"Error formatting prompt: {e}")
        return {"summary": "Error formatting prompt", "suggestion_for_next_level": "Check template"}

    # ปรับ System Instruction ให้ดุดันขึ้นเพื่อลด Invalid Format
    system_instruction = (
        f"{SYSTEM_EVIDENCE_DESCRIPTION_PROMPT}\n"
        "STRICT RULE: ตอบในรูปแบบ JSON เท่านั้น ห้ามมี Markdown หรือคำเกริ่นนำ\n"
        "EXPECTED FORMAT: {\"summary\": \"...\", \"suggestion_for_next_level\": \"...\"}"
    )

    max_retries = 2
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Generating Thai Summary for {sub_id} L{level} (Attempt {attempt})")
            
            raw_response_obj = llm_executor.generate(
                system=system_instruction, 
                prompts=[human_prompt]
            )

            # ดึง Text ออกจาก Response Object
            raw_response_str = ""
            if hasattr(raw_response_obj, 'generations'): 
                raw_response_str = raw_response_obj.generations[0][0].text
            elif hasattr(raw_response_obj, 'content'):   
                raw_response_str = raw_response_obj.content
            else:
                raw_response_str = str(raw_response_obj)

            # ใช้ Regex Extract JSON (เผื่อ LLM ใส่ข้อความแถมมา)
            parsed = _extract_normalized_dict(raw_response_str)
            
            if parsed and isinstance(parsed, dict):
                # ใช้ .get() พร้อม Default Value เพื่อป้องกัน KeyMissingError
                sum_text = parsed.get("summary") or parsed.get("สรุป") or ""
                sug_text = parsed.get("suggestion_for_next_level") or parsed.get("คำแนะนำ") or ""

                if sum_text:
                    return {
                        "summary": str(sum_text).strip(),
                        "suggestion_for_next_level": str(sug_text).strip() if sug_text else "พัฒนาตามเกณฑ์ถัดไป"
                    }
            
            logger.warning(f"Attempt {attempt}: LLM returned invalid summary format.")
            # ถ้าเป็นรอบแรกแล้วพัง รอบสองเราจะย้ำ Force JSON เข้าไปอีกใน prompt
            human_prompt += "\nReminder: Return ONLY JSON."
            
        except Exception as e:
            logger.error(f"Attempt {attempt} failed: {str(e)}")
            time.sleep(0.5)

    return {
        "summary": f"พบหลักฐานระดับ {level} แต่ระบบสรุปขัดข้อง (Parse Error)",
        "suggestion_for_next_level": f"ตรวจสอบเกณฑ์ของ Level {next_level} ในคู่มือ"
    }


# =================================================================
# 1. Main Function: create_structured_action_plan
# =================================================================
def create_structured_action_plan(
    recommendation_statements: List[Dict[str, Any]],
    sub_id: str,
    sub_criteria_name: str,
    target_level: int,
    llm_executor: Any,
    logger: logging.Logger,
    max_retries: int = 3,
    enabler_rules: Dict[str, Any] = {}
) -> List[Dict[str, Any]]:
    """
    สร้าง Action Plan ที่มีการจัดกลุ่มประเด็นไม่ให้ซ้ำซ้อน และเลือก Mode ตามผลประเมินจริง
    """
    
    # --- 1. วิเคราะห์สถานะเพื่อเลือกโหมด ---
    is_sustain_mode = not recommendation_statements
    
    is_quality_refinement = False
    if not is_sustain_mode:
        types = [s.get('recommendation_type') for s in recommendation_statements]
        # ตรวจสอบกรณี "ผ่านหมดแล้วแต่ต้องเสริมคุณภาพ" (ไม่มีข้อที่ตกจริง)
        if 'FAILED' not in types and 'GAP_ANALYSIS' not in types:
            is_quality_refinement = True

    # --- 2. การเลือก Prompt ตามโหมด พร้อมคำสั่งคุมกำเนิดความซ้ำซ้อน ---
    if is_sustain_mode:
        current_system_prompt = SYSTEM_EXCELLENCE_PROMPT + "\nสำคัญ: เน้นนวัตกรรมที่ยั่งยืน ห้ามเขียนแผนงานซ้ำซ้อน"
        current_prompt_template = EXCELLENCE_ADVICE_PROMPT
        advice_focus = "การรักษาความเป็นเลิศและสร้างนวัตกรรมต่อเนื่อง"
        assessment_context = f"ผ่านเกณฑ์ระดับ 5 (สูงสุด) อย่างสมบูรณ์ในหัวข้อ {sub_criteria_name}"
        max_steps = 5
        
    elif is_quality_refinement:
        current_system_prompt = SYSTEM_QUALITY_PROMPT + "\nสำคัญ: รวมประเด็นที่คล้ายกันเข้าด้วยกัน ห้ามสร้าง Action Item ที่ซ้ำซ้อน"
        current_prompt_template = QUALITY_REFINEMENT_PROMPT
        advice_focus = "การเสริมความแข็งแกร่งของหลักฐานและวงจร PDCA ให้สมบูรณ์ 100%"
        assessment_context = f"ผ่านเกณฑ์ในระดับสูงแล้ว แต่ควรเพิ่มคุณภาพและความน่าเชื่อถือของหลักฐาน"
        max_steps = 3
        
    else:
        current_system_prompt = SYSTEM_ACTION_PLAN_PROMPT + "\nสำคัญ: สรุปประเด็นบกพร่องที่คล้ายกันให้เป็นหนึ่งแผนงาน (Consolidate)"
        current_prompt_template = ACTION_PLAN_PROMPT
        advice_focus = "การแก้ไขช่องว่างและสร้างระบบงานตามมาตรฐาน"
        assessment_context = f"อยู่ระหว่างการพัฒนาสู่ระดับ {target_level} และแก้ไขจุดบกพร่องที่พบ"
        max_steps = 3

    # --- 3. จัดเตรียมเนื้อหา Statements (REVISED: Logic การลบรายการซ้ำ) ---
    if is_sustain_mode:
        stmt_content = "บรรลุเกณฑ์มาตรฐานสูงสุดอย่างครบถ้วน"
    else:
        unique_statements = {}
        for s in recommendation_statements:
            reason = (s.get('reason') or s.get('statement') or "").strip()
            lvl = s.get('level', 0)
            if not reason: continue
            
            # หากข้อความเหมือนกัน ให้ยึดอันที่มี Level สูงกว่า (เพื่อคลุมเกณฑ์ที่ยากกว่า)
            if reason not in unique_statements or lvl > unique_statements[reason]:
                unique_statements[reason] = lvl
        
        stmt_blocks = [f"- [Level {v}] {k}" for k, v in unique_statements.items()]
        stmt_content = "\n".join(stmt_blocks)

    # --- 4. ประกอบ Human Prompt ---
    human_prompt = current_prompt_template.format(
        sub_id=sub_id, 
        sub_criteria_name=sub_criteria_name, 
        target_level=target_level,
        assessment_context=assessment_context,
        advice_focus=advice_focus, 
        recommendation_statements_list=stmt_content,
        max_phases=1, 
        max_steps=max_steps, 
        max_words_per_step=150,
        language="ภาษาไทย"
    )

    # --- 5. EXECUTION & VALIDATION LOOP ---
    for attempt in range(1, max_retries + 1):
        try:
            response = llm_executor.generate(
                system=current_system_prompt, 
                prompts=[human_prompt],
                temperature=0.2 # ค่านิ่งขึ้น ลดความเพ้อเจ้อ
            )
            raw_text = response.generations[0][0].text if hasattr(response, 'generations') else str(response)
            
            # สกัด JSON จาก Text
            items = _extract_json_array_for_action_plan(raw_text, logger)
            if not items: continue

            # Normalize Keys ให้เข้ากับ Schema (ป้องกัน LLM พ่น Key ผิด/มีวรรค)
            clean_items = action_plan_normalize_keys(items)
            
            # Validate ด้วย Pydantic
            validated_result = ActionPlanResult.model_validate(clean_items)
            
            return validated_result.model_dump(by_alias=True)

        except Exception as e:
            logger.error(f"⚠️ Action Plan Attempt {attempt} failed for {sub_id}: {e}")

    # --- 6. EMERGENCY FALLBACK ---
    return _get_emergency_fallback_plan(sub_id, sub_criteria_name, target_level, is_sustain_mode, is_quality_refinement)

# =================================================================
# 2. Key Normalizer: แก้ไขปัญหา LLM พ่น Key ไม่นิ่ง
# =================================================================
def action_plan_normalize_keys(obj: Any) -> Any:
    if isinstance(obj, list): return [action_plan_normalize_keys(i) for i in obj]
    if isinstance(obj, dict):
        field_mapping = {
            'phase': 'phase', 'goal': 'goal', 'actions': 'actions',
            'statementid': 'statement_id', 'statement_id': 'statement_id',
            'failedlevel': 'failed_level', 'failed_level': 'failed_level',
            'recommendation': 'recommendation',
            'targetevidencetype': 'target_evidence_type', 'target_evidence_type': 'target_evidence_type',
            'keymetric': 'key_metric', 'key_metric': 'key_metric',
            'steps': 'steps', 'step': 'step', 
            'description': 'description', 'responsible': 'responsible',
            'toolstemplates': 'tools_templates', 'tools_templates': 'tools_templates',
            'verificationoutcome': 'verification_outcome', 'verification_outcome': 'verification_outcome'
        }
        
        new_obj = {}
        for k, v in obj.items():
            # กวาดล้าง Key ให้เหลือแต่ตัวพิมพ์เล็กและตัวเลขเพื่อเปรียบเทียบ
            k_raw = str(k).lower().replace(' ', '').replace('_', '').strip()
            k_raw = re.sub(r'[^a-z0-9]', '', k_raw)
            
            target_key = field_mapping.get(k_raw) or k_raw
            
            # Enforcement: บังคับ Integer สำหรับ Level และ Step
            if target_key in ['failed_level', 'step']:
                try:
                    if isinstance(v, (int, float)): v = int(v)
                    else:
                        nums = re.findall(r'\d+', str(v))
                        v = int(nums[0]) if nums else 0
                except: v = 0
            
            new_obj[target_key] = action_plan_normalize_keys(v)
        return new_obj
    return obj

# =================================================================
# 3. JSON Extractor: ระบบกู้คืน JSON ที่พังหรือเจนไม่จบ
# =================================================================
def _extract_json_array_for_action_plan(text: Any, logger: logging.Logger) -> List[Dict[str, Any]]:
    try:
        if not isinstance(text, str): text = str(text) if text is not None else ""
        if not text.strip(): return []

        # ลบ Markdown tags
        clean_text = re.sub(r'```(?:json)?\s*([\s\S]*?)\s*```', r'\1', text, flags=re.IGNORECASE).strip()

        # ค้นหาขอบเขต JSON
        start_idx = clean_text.find('[')
        end_idx = clean_text.rfind(']')

        if start_idx == -1:
            start_idx = clean_text.find('{')
            end_idx = clean_text.rfind('}')
            if start_idx == -1: return []
            json_candidate = clean_text[start_idx:end_idx + 1]
        else:
            json_candidate = clean_text[start_idx:end_idx + 1]

        # ล้าง Control characters
        json_candidate = "".join(char for char in json_candidate if ord(char) >= 32 or char in "\n\r\t")

        def try_parse(content):
            try:
                data = json5.loads(content)
                return data if isinstance(data, list) else [data]
            except Exception: return None

        # พยายาม Parse และซ่อมแซม
        result = try_parse(json_candidate)
        if not result:
            repaired = json_candidate.replace('“', '"').replace('”', '"').replace("'", '"')
            result = try_parse(repaired)
        
        # กรณี LLM ตัดจบ (Truncated) พยายามเติมปิดท้ายให้
        if not result:
            for suffix in ["]", "}", "}]", "}\n]"]:
                result = try_parse(json_candidate + suffix)
                if result: break

        return result or []
    except Exception as e:
        logger.error(f"Extraction failed: {str(e)}")
        return []

# =================================================================
# 4. Emergency Fallback (Revised for All Modes)
# =================================================================
def _get_emergency_fallback_plan(sub_id, sub_criteria_name, target_level, is_sustain_mode, is_quality_refinement):
    if is_sustain_mode:
        title = "Continuous Excellence Plan"
        rec = "รักษามาตรฐานระดับสูงสุดและดำเนินการแบ่งปันองค์ความรู้สู่ภายนอก (Best Practice Sharing)"
    elif is_quality_refinement:
        title = "Quality Evidence Reinforcement"
        rec = "จัดรวบรวมหลักฐานย้อนหลังและประเมินประสิทธิภาพกระบวนการ (Check) ให้ชัดเจนยิ่งขึ้น"
    else:
        title = "Gap Remediation Roadmap"
        rec = f"เร่งดำเนินการตามเกณฑ์ระดับ {target_level} และจัดทำเอกสารประกอบวงจร PDCA ให้ครบสมบูรณ์"
        
    return [{
        "phase": f"Phase: {title}",
        "goal": f"ยกระดับและรักษามาตรฐาน {sub_criteria_name}",
        "actions": [{
            "statement_id": sub_id, 
            "failed_level": target_level,
            "recommendation": rec, 
            "target_evidence_type": "Evidence Pack / KM Dashboard",
            "key_metric": "ความสมบูรณ์ของหลักฐาน 100%",
            "steps": [
                {
                    "Step": 1, 
                    "Description": "ทบทวนช่องว่างของหลักฐานและจัดทำสรุปผลการดำเนินงาน", 
                    "Responsible": "KM Working Team", 
                    "Tools_Templates": "Gap Analysis Template", 
                    "Verification_Outcome": "รายงานสรุปช่องว่างและการแก้ไข"
                }
            ]
        }]
    }]
