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
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
import os
import unicodedata


# Optional: regex แทน re (ดีกว่า) — ถ้าไม่มีก็ใช้ re ธรรมดา
try:
    import regex as re  # type: ignore
except ImportError:
    pass  # ใช้ re มาตรฐานต่อไป

# ===================================================================
# 1. Core Configuration (ต้องมีแน่นอน)
# ===================================================================
from config.global_vars import (
    INITIAL_TOP_K,
    MAX_EVAL_CONTEXT_LENGTH,
    DEFAULT_EMBED_BATCH_SIZE,
    RERANK_THRESHOLD,
    ANALYSIS_FINAL_K,
    ACTION_PLAN_STEP_MAX_WORDS,
    LLM_TEMPERATURE,
    MAX_ACTION_PLAN_TOKENS
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
    SYSTEM_EVIDENCE_DESCRIPTION_PROMPT,
    EVIDENCE_DESCRIPTION_PROMPT,
    USER_LOW_LEVEL_PROMPT,
    USER_EVIDENCE_DESCRIPTION_TEMPLATE,
)

from core.vectorstore import VectorStoreManager, get_global_reranker, ChromaRetriever
from core.assessment_schema import CombinedAssessment, EvidenceSummary

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

def _create_where_filter(
    stable_doc_ids: Optional[Union[Set[str], List[str]]] = None,
    subject: Optional[str] = None,
    sub_topic: Optional[str] = None,
    year: Optional[Union[int, str]] = None,
    enabler: Optional[str] = None,
    tenant: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    [PRODUCTION VERSION] สร้าง Filter สำหรับ ChromaDB ที่แม่นยำ
    แก้ปัญหาเรื่อง Data Type Mismatch (Int/Str) และรองรับ Multi-tenant
    """
    filters: List[Dict[str, Any]] = []

    # --- 1. การจัดการ Stable Doc IDs (ลำดับความสำคัญสูงสุด) ---
    if stable_doc_ids:
        ids_list = [str(i).strip() for i in (stable_doc_ids if isinstance(stable_doc_ids, (list, set)) else [stable_doc_ids]) if i]
        if ids_list:
            if len(ids_list) == 1:
                # กรณีเลือกไฟล์เดียว ไม่ต้องใช้ $and
                return {"stable_doc_uuid": ids_list[0]}
            else:
                return {"stable_doc_uuid": {"$in": ids_list}}

    # --- 2. การจัดการ Year (จุดที่ทำให้ Local Mac หาไม่เจอ) ---
    if year is not None:
        year_str = str(year).strip()
        if year_str and year_str.lower() != "none":
            # 🎯 แก้ไข: ส่งทั้งแบบ Int และ Str หรือพยายามแปลงเป็น Int ตามข้อมูลที่ Peek เจอใน Mac
            try:
                # ถ้าแปลงเป็นตัวเลขได้ ให้ส่งแบบ Integer (ChromaDB Local มักเก็บเป็น Int)
                val_year = int(year_str)
                filters.append({"year": val_year})
            except (ValueError, TypeError):
                # ถ้าแปลงไม่ได้จริงๆ ให้ส่งแบบ String
                filters.append({"year": year_str})
    
    # --- 3. การจัดการ Tenant (ป้องกันข้ามบริษัท) ---
    effective_tenant = tenant or kwargs.get("tenant")
    if effective_tenant and str(effective_tenant).strip():
        filters.append({"tenant": str(effective_tenant).strip()})

    # --- 4. การจัดการ Enabler (KM, IM, etc.) ---
    if enabler and str(enabler).strip():
        filters.append({"enabler": enabler.strip().upper()})

    # --- 5. การจัดการ Subject (หัวข้อเกณฑ์) ---
    if subject and str(subject).strip():
        filters.append({"subject": str(subject).strip()})

    # --- 6. การจัดการ Sub Topic ---
    if sub_topic and str(sub_topic).strip():
        filters.append({"sub_topic": str(sub_topic).strip()})

    # --- สรุปผลการสร้าง Filter ---
    if not filters:
        return {}

    # ถ้ามีเงื่อนไขเดียวส่งคืนได้เลย ถ้ามีหลายอันต้องเชื่อมด้วย $and
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
    k_to_retrieve: int = 150, 
    k_to_rerank: int = 30,    
    strict_filter: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    [ULTIMATE REVISED] Retrieval for Search Endpoint
    - ยึด Metadata เป็นหลักเพื่อความปลอดภัย (No Pydantic Error)
    - ใช้ Deterministic MD5 Hash สำหรับ Deduplication
    - รองรับ Anchor Chunks เพื่อสร้าง Context ที่มั่นคง
    """
    start_time = time.time()
    vsm = vectorstore_manager

    # 1. Resolve collection
    clean_doc_type = str(doc_type or "document").strip().lower()
    collection_name = get_doc_type_collection_key(doc_type=clean_doc_type, enabler=enabler)
    
    chroma = vsm._load_chroma_instance(collection_name)
    if not chroma:
        logger.error(f"❌ Collection {collection_name} not found.")
        return {"top_evidences": [], "aggregated_context": "ไม่พบฐานข้อมูล", "retrieval_time": 0}

    # 2. Create where_filter
    where_filter = _create_where_filter(
        stable_doc_ids=list(stable_doc_ids) if stable_doc_ids else None, 
        subject=subject, 
        sub_topic=sub_topic, 
        year=year
    )

    # Dictionary สำหรับ Deduplication (ใช้ MD5 ของเนื้อหาเป็น Key)
    unique_map: Dict[str, LcDocument] = {}

    # =====================================================
    # ⚓ 2.1 ANCHOR RETRIEVAL (โครงสร้างไฟล์)
    # =====================================================
    if stable_doc_ids:
        # ดึงหน้าแรกๆ ของแต่ละไฟล์มาเป็นพื้นฐาน
        anchors = chroma.get(where=where_filter, limit=10) 
        if anchors and anchors.get('documents'):
            for i in range(len(anchors['documents'])):
                content = anchors['documents'][i]
                md = anchors['metadatas'][i]
                
                # Deterministic MD5 Hash
                c_hash = hashlib.md5(content.encode()).hexdigest()
                uid = md.get("chunk_uuid") or f"anchor-{c_hash}"
                
                if uid not in unique_map:
                    # ฉีดคะแนนเริ่มต้นให้ Anchor เพื่อให้ติดอันดับต้นๆ หากเนื้อหาใกล้เคียง
                    md["score"] = 0.5 
                    md["is_anchor"] = True
                    unique_map[uid] = LcDocument(page_content=content, metadata=md)

    # =====================================================
    # 🔍 2.2 SEMANTIC SEARCH
    # =====================================================
    search_query = query if (query and query != "*" and len(query) > 2) else ""
    
    # ดึงข้อมูลดิบจาก Vector DB
    if search_query:
        docs = chroma.similarity_search(search_query, k=k_to_retrieve, filter=where_filter)
    else:
        # Fallback กรณีไม่มี Query ให้ดึงแบบกวาด
        docs = chroma.similarity_search("*", k=k_to_retrieve, filter=where_filter)

    for d in docs:
        c_hash = hashlib.md5(d.page_content.encode()).hexdigest()
        md = d.metadata or {}
        uid = md.get("chunk_uuid") or c_hash
        if uid not in unique_map:
            unique_map[uid] = d

    candidates = list(unique_map.values())

    # =====================================================
    # 🚀 3. BATCH RERANKING
    # =====================================================
    final_scored_docs = []
    reranker = getattr(vsm, "reranker", None)
    
    if reranker and candidates and search_query:
        try:
            batch_size = 100 
            logger.info(f"🚀 Reranking {len(candidates)} candidates in batches...")
            for i in range(0, len(candidates), batch_size):
                batch = candidates[i : i + batch_size]
                # ใช้ Reranker ของ LangChain (Compressor)
                scored_batch = reranker.compress_documents(documents=batch, query=search_query)
                final_scored_docs.extend(scored_batch)
        except Exception as e:
            logger.error(f"⚠️ Rerank failed: {e}")
            final_scored_docs = candidates
    else:
        final_scored_docs = candidates

    # =====================================================
    # 4. SORTING & SCORE INJECTION (หัวใจหลักที่แก้ 0.0000)
    # =====================================================
    def get_score(d) -> float:
        m = d.metadata or {}
        # ดึงคะแนนจากทึกที่เป็นไปได้
        s = m.get("relevance_score") or m.get("score") or m.get("rerank_score") or 0.0
        try: return float(s)
        except: return 0.0

    final_scored_docs.sort(key=get_score, reverse=True)

    # =====================================================
    # 5. RESPONSE BUILD
    # =====================================================
    top_evidences = []
    aggregated_parts = []
    
    for doc in final_scored_docs[:k_to_rerank]:
        md = doc.metadata or {}
        text = doc.page_content.strip()
        score = get_score(doc)
        
        # จัดการเลขหน้าและแหล่งที่มาให้ Robust
        p_val = md.get("page") or md.get("page_label") or md.get("page_number") or "N/A"
        source_name = md.get('source') or md.get('source_filename') or md.get('file_name') or 'Unknown'
        s_uuid = md.get("stable_doc_uuid") or md.get("doc_id") or ""
        
        # SYNC SCORE กลับเข้า Metadata ทุกตัว
        md["score"] = score
        md["relevance_score"] = score
        
        evidence_item = {
            "doc_id": str(s_uuid),
            "chunk_uuid": str(md.get("chunk_uuid") or ""),
            "source": source_name,
            "text": text,
            "page": str(p_val),
            "score": score,
            "pdca_tag": md.get("pdca_tag", "Other"),
            "metadata": md
        }
        
        top_evidences.append(evidence_item)
        aggregated_parts.append(f"[ไฟล์: {source_name}, หน้า: {p_val}] {text}")

    retrieval_time = round(time.time() - start_time, 3)
    logger.info(f"🏁 Finished: {len(top_evidences)} chunks in {retrieval_time}s")

    return {
        "top_evidences": top_evidences,
        "aggregated_context": "\n\n---\n\n".join(aggregated_parts) if aggregated_parts else "ไม่พบข้อมูล",
        "retrieval_time": retrieval_time,
        "total_candidates": len(candidates)
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
    top_k: int = 150, 
) -> Dict[str, Any]:
    """
    [FIXED VERSION] ยึดโครงสร้างเดิมที่ทำงานได้ (No Error) 
    แต่เพิ่มการยัดคะแนนลง Metadata เพื่อแก้ปัญหา Rerank 0.0000
    """
    start_time = time.time()
    manager = vectorstore_manager
    queries_to_run = [query] if isinstance(query, str) else list(query or [""])
    
    # 1. Resolve Collection & Filter
    # ใช้ helper จาก utils เพื่อความแม่นยำของชื่อ collection
    collection_name = get_doc_type_collection_key(doc_type, enabler or "KM")
    
    target_ids = set()
    if stable_doc_ids: target_ids.update([str(i) for i in stable_doc_ids])
    if mapped_uuids: target_ids.update([str(i) for i in mapped_uuids])
    if sequential_chunk_uuids: target_ids.update([str(i) for i in sequential_chunk_uuids])
    
    where_filter = _create_where_filter(
        stable_doc_ids=list(target_ids) if target_ids else None,
        subject=subject,
        year=year
    )

    # 2. Collect Chunks
    all_source_chunks = []

    # 2.1 Priority Docs
    if priority_docs_input:
        for doc in priority_docs_input:
            if not doc: continue
            if isinstance(doc, dict):
                pc = doc.get('page_content') or doc.get('text') or ''
                meta = doc.get('metadata') or {}
                if pc.strip():
                    all_source_chunks.append(LcDocument(page_content=pc, metadata=meta))
            elif hasattr(doc, 'page_content'):
                all_source_chunks.append(doc)

    # 2.2 Vector Retrieval
    try:
        full_retriever = manager.get_retriever(collection_name=collection_name)
        base_retriever = getattr(full_retriever, "base_retriever", full_retriever)
        
        search_kwargs = {"k": top_k}
        if where_filter: 
            search_kwargs["where"] = where_filter

        for q in queries_to_run:
            if not q: continue
            docs = base_retriever.invoke(q, config={"configurable": {"search_kwargs": search_kwargs}})
            if docs: all_source_chunks.extend(docs)
    except Exception as e:
        logger.error(f"Retrieval error: {e}")

    # 3. Deduplicate (ใช้ Hash เพื่อความแม่นยำ)
    unique_map: Dict[str, LcDocument] = {}
    for doc in all_source_chunks:
        if not doc or not doc.page_content.strip(): continue
        md = doc.metadata or {}
        # ใช้ hashlib เพื่อให้ได้ ID ที่คงที่เหมือนกันทุกรอบ
        c_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
        uid = str(md.get("chunk_uuid") or f"{md.get('stable_doc_uuid', 'unknown')}-{c_hash}")
        
        if uid not in unique_map:
            unique_map[uid] = doc

    candidates = list(unique_map.values())

    # 4. Batch Reranking
    final_scored_docs = []
    batch_size = 100 
    reranker = getattr(manager, "reranker", None)

    if reranker and candidates:
        main_query = queries_to_run[0]
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i : i + batch_size]
            try:
                # 📌 กุญแจสำคัญ: Reranker จะคืนค่าพร้อมคะแนนใน metadata
                scored_batch = reranker.compress_documents(batch, main_query)
                final_scored_docs.extend(scored_batch)
            except Exception as e:
                logger.error(f"Rerank Error: {e}")
                final_scored_docs.extend(batch)
    else:
        final_scored_docs = candidates

    # 5. Sorting & Score Injection (จุดที่แก้ปัญหา 0.0000)
    def get_score(d) -> float:
        m = d.metadata or {}
        # ไล่หาคะแนนจากทุกชื่อที่เป็นไปได้
        s = m.get("relevance_score") or m.get("score") or m.get("rerank_score") or 0.0
        try: return float(s)
        except: return 0.0

    final_scored_docs.sort(key=get_score, reverse=True)

    # 6. Final Formatting (คัดเลือก K ตัวแรก)
    top_evidences = []
    aggregated_parts = []
    final_k = ANALYSIS_FINAL_K # ใช้ค่าจาก config

    for doc in final_scored_docs[:final_k]:
        score = get_score(doc)
        # กรอง Threshold ตามที่ตั้งไว้ใน .env
        if score < RERANK_THRESHOLD and RERANK_THRESHOLD > 0:
            continue

        md = doc.metadata or {}
        text = doc.page_content.strip()
        
        # จัดการเลขหน้าให้ตรงกับ ingest.py
        page = str(md.get("page_label") or md.get("page_number") or md.get("page") or "N/A")
        source = md.get("source") or md.get("source_filename") or "Unknown"
        pdca = md.get("pdca_tag", "Other")

        # 🎯 SYNC SCORE เข้า Metadata (เพื่อให้ Loop อื่นๆ เรียกใช้แล้วไม่เจอ 0)
        md["score"] = score
        md["relevance_score"] = score
        md["rerank_score"] = score

        top_evidences.append({
            "doc_id": str(md.get("stable_doc_uuid") or md.get("doc_id") or ""),
            "chunk_uuid": str(md.get("chunk_uuid") or ""),
            "source": source,
            "text": text,
            "page": page,
            "pdca_tag": pdca,
            "score": score,
            "metadata": md
        })
        aggregated_parts.append(f"[{pdca}] [ไฟล์: {source} หน้า: {page}] {text}")

    return {
        "top_evidences": top_evidences,
        "aggregated_context": "\n\n---\n\n".join(aggregated_parts) if aggregated_parts else "ไม่พบหลักฐาน",
        "retrieval_time": round(time.time() - start_time, 3),
        "total_candidates": len(candidates)
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
                    md = dict(anchors['metadatas'][i] or {}) # ป้องกัน metadata เป็น None
                    
                    # 🎯 แก้จุดที่ 1: ใช้ MD5 แทน hash() เพื่อให้ ID คงที่ (Deterministic)
                    content_hash = hashlib.md5(content.encode()).hexdigest()
                    uid = str(md.get("chunk_uuid") or f"anchor-{content_hash}")
                    
                    if uid not in unique_evidence_map:
                        # 🎯 แก้จุดที่ 2: ฉีดคะแนนลงใน metadata โดยตรง (Safe Injection)
                        # กำหนดให้ Anchor มีคะแนนสูง (0.95) เพื่อให้เป็นบริบทหลัก
                        md["score"] = 0.95
                        md["relevance_score"] = 0.95
                        md["is_anchor"] = True
                        
                        unique_evidence_map[uid] = LcDocument(
                            page_content=content,
                            metadata=md
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
    Ultimate Robust LLM Fetcher v2026.1.19-final
    - คืน STRING เสมอ ไม่เคยคืน None
    - ป้องกัน NoneType.strip() ทุกจุด
    - Log ชัดเจนเพื่อ debug
    """

    if llm_executor is None and not _MOCK_FLAG:
        raise ConnectionError("LLM instance not initialized.")

    # Enforced prompt (ปรับให้เข้มขึ้นอีกนิด)
    enforced_system_prompt = (system_prompt or "").strip() + (
        "\n\n### STRICT OUTPUT RULES - FOLLOW EXACTLY ###\n"
        "1. Respond with ONLY valid JSON object. No other text.\n"
        "2. Start with '{' and end with '}'.\n"
        "3. No markdown, no explanations, no prefixes.\n"
        "4. If no evidence: {\"score\": 0, \"reason\": \"No evidence\", \"is_passed\": false}"
    )

    messages = [
        {"role": "system", "content": enforced_system_prompt},
        {"role": "user",   "content": (user_prompt or "").strip()}
    ]

    for attempt in range(1, max_retries + 1):
        try:
            if _MOCK_FLAG:
                mock = '{"score": 1, "reason": "Mock active", "is_passed": true}'
                logger.critical(f"LLM RAW (MOCK): {mock}")
                return mock

            # LLM CALL
            response = llm_executor.invoke(messages, config={"temperature": 0.0})

            # SAFE EXTRACTION - รวม patch ของคุณ
            raw_text = ""
            if response is None:
                logger.warning("Response object is None")
            elif hasattr(response, "content"):
                raw_text = str(response.content or "")
            elif hasattr(response, "text"):
                raw_text = str(response.text or "")
            elif isinstance(response, str):
                raw_text = response
            else:
                raw_text = str(response or "")

            # Log raw ก่อน clean
            preview = (raw_text[:1000] + "...") if len(raw_text) > 1000 else raw_text
            logger.critical(f"LLM RAW RESPONSE (attempt {attempt}): {preview}")

            # Clean - ปลอดภัยแน่นอน
            raw_text_stripped = (raw_text or "").strip()

            # หา JSON block
            json_match = re.search(r'\{[\s\S]*?\}', raw_text_stripped, re.DOTALL)
            if json_match:
                extracted = json_match.group(0)
                try:
                    json.loads(extracted)  # ทดสอบก่อน
                    return extracted
                except json.JSONDecodeError:
                    logger.warning(f"Extracted not valid JSON: {extracted[:120]}...")

            # Fallback: ลอง parse ทั้งหมด
            try:
                parsed = json.loads(raw_text_stripped)
                return json.dumps(parsed, ensure_ascii=False)
            except json.JSONDecodeError:
                pass

            # Ultimate safe return - บังคับ string เสมอ (ตามที่คุณแนะนำ)
            final_return = str(raw_text_stripped or "").strip()
            logger.debug(f"No valid JSON → returning cleaned string (len={len(final_return)})")
            return final_return

        except Exception as e:
            logger.error(f"Attempt {attempt} failed: {str(e)}", exc_info=True)
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                logger.critical("All retries failed")
                return '{"score": 0, "reason": "LLM failed after retries", "is_passed": false}'

    return '{"score": 0, "reason": "Unknown error", "is_passed": false}'

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

def _get_context_for_level(
    context: str,
    level: int,
    chunks: list = None,  # ส่ง chunks มาจาก _run_single_assessment
    max_chars_l1_l2: int = 15000,  # เพิ่มจาก 6000
    max_chars_l3_up: int = 10000,
    max_chunks_l1_l2: int = 40,
    max_chunks_l3_up: int = 25
) -> str:
    if not context:
        return "ไม่พบข้อมูลหลักฐาน"

    # ถ้ามี chunks → เรียงตาม score แล้วเลือก top
    if chunks:
        sorted_chunks = sorted(
            chunks,
            key=lambda c: float(c.get('rerank_score', 0) or c.get('score', 0)),
            reverse=True
        )
        max_chunks = max_chunks_l1_l2 if level <= 2 else max_chunks_l3_up
        selected = sorted_chunks[:max_chunks]

        parts = []
        for i, c in enumerate(selected, 1):
            score = c.get('rerank_score', 'N/A')
            source = c.get('source', 'ไม่ระบุ')
            text = c.get('text', '').strip()
            if text:
                parts.append(f"[Chunk {i} | Score: {score} | {source}]\n{text}\n{'-'*80}\n")

        final = "".join(parts)
        max_chars = max_chars_l1_l2 if level <= 2 else max_chars_l3_up
        if len(final) > max_chars:
            final = final[:max_chars] + "\n... (ตัดเพื่อความเหมาะสม)"
        return final

    # fallback เดิม
    max_chars = max_chars_l1_l2 if level <= 2 else max_chars_l3_up
    return context[:max_chars] + ("... [truncated]" if len(context) > max_chars else "")

def evaluate_with_llm(
    context: str, 
    sub_criteria_name: str, 
    level: int, 
    statement_text: str, 
    sub_id: str, 
    llm_executor: Any = None, 
    required_phases: List[str] = None,
    specific_contextual_rule: str = "พิจารณาตามเกณฑ์มาตรฐาน",
    ai_confidence: str = "MEDIUM",
    confidence_reason: str = "N/A",
    pdca_context: str = "", # <--- [ADD] รับข้อมูลแยกหมวดหมู่ P-D-C-A
    **kwargs
) -> Dict[str, Any]:
    """
    [REVISED v2026.3.5 — PDCA Block Enabled]
    - รองรับการฉีด pdca_context เพื่อเพิ่มความแม่นยำในการสกัดหลักฐาน
    - เสริมระบบความปลอดภัยในการ Parse JSON และจัดการ Multi-Enabler
    """
    logger = logging.getLogger(__name__)

    # 1. Safe casting + defaults
    ctx_raw = str(context or "ไม่พบข้อมูลหลักฐาน")
    pdca_ctx = str(pdca_context or "ไม่มีข้อมูลแยกหมวดหมู่ (โปรดพิจารณาจาก Full Context)")
    s_name = str(sub_criteria_name or "N/A")
    sid = str(sub_id or "N/A")
    s_text = str(statement_text or "N/A")
    
    # Enabler info
    enabler_full_name = str(kwargs.get("enabler_full_name", "Unknown Enabler"))
    enabler_code = str(kwargs.get("enabler_code", "UNK"))
    
    logger.info(f"[EVAL START] Enabler: {enabler_full_name} ({enabler_code}) | Sub: {sid} | L{level}")

    # ดึง Context ที่เหมาะสมตามระดับ (Slice ตามความยาวที่กำหนด)
    context_to_send_eval = _get_context_for_level(ctx_raw, level) or ""
    
    # 2. Safe phases
    phases_str = ", ".join(str(p).strip() for p in (required_phases or [])) if required_phases else "P, D, C, A"

    # 3. Baseline clean
    baseline_raw = kwargs.get("baseline_summary")
    baseline_summary = str(baseline_raw or "").strip()

    try:
        # Build prompt with PDCA Context support
        # หมายเหตุ: ใน USER_ASSESSMENT_PROMPT ต้องมี placeholder {pdca_context}
        full_prompt = USER_ASSESSMENT_PROMPT.format(
            sub_criteria_name=s_name,
            sub_id=sid,
            level=int(level),
            statement_text=s_text,
            context=context_to_send_eval[:28000], # เผื่อพื้นที่ให้ PDCA Block
            pdca_context=pdca_ctx[:8000],         # [CRITICAL] ฉีดบล็อกข้อมูลแยกหมวด
            required_phases=phases_str,
            specific_contextual_rule=str(specific_contextual_rule or "พิจารณาตามเกณฑ์"),
            ai_confidence=str(ai_confidence or "MEDIUM"),
            confidence_reason=str(confidence_reason or "N/A"),
            enabler_full_name=enabler_full_name,
            enabler_code=enabler_code
        )

        if baseline_summary:
            full_prompt += f"\n\n--- BASELINE DATA (จากระดับก่อนหน้า) ---\n{baseline_summary}"

        # 4. LLM call with guard
        if llm_executor is None:
            raise ValueError("No LLM executor provided")

        raw_response = _fetch_llm_response(None, full_prompt, llm_executor=llm_executor)
        raw_response = str(raw_response or "").strip()

        # 5. Parse with fallback
        parsed = _robust_extract_json(raw_response)
        if not parsed or not isinstance(parsed, dict):
            logger.warning(f"[JSON PARSE FAIL] Sub {sid} L{level} - Using heuristic fallback")
            parsed = _heuristic_fallback_parse(raw_response)

        # 6. Build final result object
        result = _build_audit_result_object(
            parsed, raw_response, context_to_send_eval, ai_confidence, 
            level=level, sub_id=sid, enabler_full_name=enabler_full_name, enabler_code=enabler_code
        )
        return result

    except Exception as e:
        logger.error(f"🛑 Evaluation Error Enabler:{enabler_code} Sub:{sid} L{level}: {str(e)}", exc_info=True)
        return _create_fallback_error(sid, level, e, context_to_send_eval, enabler_full_name, enabler_code)


def evaluate_with_llm_low_level(
    context: str,
    sub_criteria_name: str,
    level: int,
    statement_text: str,
    sub_id: str,
    llm_executor: Any = None,
    required_phases: List[str] = None,
    specific_contextual_rule: str = "พิจารณาตามเกณฑ์มาตรฐาน",
    ai_confidence: str = "MEDIUM",
    pdca_context: str = "", # <--- [ADD]
    **kwargs
) -> Dict[str, Any]:
    """
    [REVISED v2026.3.5 — Low Level PDCA Enabled]
    - รองรับ pdca_context สำหรับระดับ L1-L2
    - เน้นความแม่นยำในการแยกหมวดหมู่ 'แผน' และ 'กิจกรรม'
    """
    logger = logging.getLogger(__name__)

    # 1. Safe casting
    ctx = str(context or "ไม่พบข้อมูลหลักฐาน")
    pdca_ctx = str(pdca_context or "ไม่มีข้อมูลแยกหมวดหมู่")
    s_name = str(sub_criteria_name or "N/A")
    s_text = str(statement_text or "N/A")
    sid = str(sub_id or "N/A")
    
    # Enabler info
    enabler_full_name = str(kwargs.get("enabler_full_name", "Unknown Enabler"))
    enabler_code = str(kwargs.get("enabler_code", "UNK"))
    
    logger.info(f"[LOW EVAL START] Enabler: {enabler_full_name} ({enabler_code}) | Sub: {sid} | L{level}")

    plan_kws = str(kwargs.get("plan_keywords") or "นโยบาย, แผนงาน, ยุทธศาสตร์")
    baseline_summary = str(kwargs.get("baseline_summary") or "ไม่มีข้อมูลระดับก่อนหน้า").strip()
    conf_reason = str(kwargs.get("confidence_reason") or "วิเคราะห์ตามเนื้องาน")
    
    phases_str = ", ".join(str(p) for p in (required_phases or [])) if required_phases else "P, D"

    try:
        full_prompt = USER_LOW_LEVEL_PROMPT.format(
            sub_id=sid,
            sub_criteria_name=s_name,
            level=int(level),
            statement_text=s_text,
            context=ctx[:28000],
            pdca_context=pdca_ctx[:8000], # [CRITICAL]
            required_phases=phases_str,
            plan_keywords=plan_kws,
            baseline_summary=baseline_summary,
            specific_contextual_rule=str(specific_contextual_rule or "ตรวจตามเกณฑ์"),
            ai_confidence=str(ai_confidence or "MEDIUM"),
            confidence_reason=conf_reason,
            enabler_full_name=enabler_full_name,
            enabler_code=enabler_code
        )

        if llm_executor is None:
            raise ValueError("No LLM executor provided")

        raw_response = _fetch_llm_response(None, full_prompt, llm_executor=llm_executor)
        raw_response = str(raw_response or "").strip()

        parsed = _robust_extract_json(raw_response)
        if not parsed or not isinstance(parsed, dict):
            logger.warning(f"[LOW JSON PARSE FAIL] Sub {sid} L{level} - Using heuristic fallback")
            parsed = _heuristic_fallback_parse(raw_response)

        result = _build_audit_result_object(
            parsed, raw_response, ctx, ai_confidence, 
            level=level, sub_id=sid, enabler_full_name=enabler_full_name, enabler_code=enabler_code
        )
        return result

    except Exception as e:
        logger.error(f"🛑 Low-Level Eval Error Enabler:{enabler_code} Sub:{sid} L{level}: {str(e)}", exc_info=True)
        return _create_fallback_error(sid, level, e, ctx, enabler_full_name, enabler_code)
    
def _build_audit_result_object(parsed: Dict, raw_response: str, context: str, confidence: str, **kwargs) -> Dict[str, Any]:
    """
    [ULTIMATE-SYNC v2026.1.22] — Multi-Enabler + Zero-Error
    - เพิ่ม enabler_full_name & enabler_code ใน result
    - Robust string handling (str(val or "") ก่อน strip)
    - รองรับ key หลากหลายรูปแบบจาก LLM
    """
    level = kwargs.get('level', 1)
    sub_id = kwargs.get('sub_id', 'Unknown')
    enabler_full_name = kwargs.get('enabler_full_name', 'Unknown Enabler')
    enabler_code = kwargs.get('enabler_code', 'UNK')

    def clean_score(val, default=0.0):
        if val is None: return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    if not isinstance(parsed, dict):
        parsed = {}

    score = clean_score(parsed.get("score"))
    is_passed = parsed.get("is_passed")
    if is_passed is None:
        is_passed = score >= 0.7 if level <= 2 else score >= 1.0

    # Robust extraction (ป้องกัน NoneType.strip)
    ext_p = str(parsed.get("Extraction_P") or parsed.get("หลักฐาน P") or parsed.get("p_plan_extraction") or "-").strip()
    ext_d = str(parsed.get("Extraction_D") or parsed.get("หลักฐาน D") or parsed.get("d_do_extraction") or "-").strip()
    ext_c = str(parsed.get("Extraction_C") or parsed.get("หลักฐาน C") or parsed.get("c_check_extraction") or "-").strip()
    ext_a = str(parsed.get("Extraction_A") or parsed.get("หลักฐาน A") or parsed.get("a_act_extraction") or "-").strip()

    # PDCA scores — รองรับ key หลากหลาย
    p_plan_score = clean_score(parsed.get("P_Plan_Score") or parsed.get("P_Score") or parsed.get("plan_score") or parsed.get("P"))
    d_do_score = clean_score(parsed.get("D_Do_Score") or parsed.get("D_Score") or parsed.get("do_score") or parsed.get("D"))
    c_check_score = clean_score(parsed.get("C_Check_Score") or parsed.get("C_Score") or parsed.get("check_score") or parsed.get("C"))
    a_act_score = clean_score(parsed.get("A_Act_Score") or parsed.get("A_Score") or parsed.get("act_score") or parsed.get("A"))

    # Fallback สำหรับ L1-L2
    if bool(is_passed) and level <= 2 and p_plan_score == 0:
        p_plan_score = score

    return {
        "sub_id": str(sub_id),
        "level": int(level),
        "score": score,
        "is_passed": bool(is_passed),
        "reason": str(parsed.get("reason") or parsed.get("เหตุผล") or "ไม่พบเหตุผลจาก LLM").strip(),
        "summary_thai": str(parsed.get("summary_thai") or parsed.get("บทสรุป") or "").strip(),
        "coaching_insight": str(parsed.get("coaching_insight") or parsed.get("ข้อแนะนำ") or "").strip(),
        
        "P_Plan_Score": p_plan_score,
        "D_Do_Score": d_do_score,
        "C_Check_Score": c_check_score,
        "A_Act_Score": a_act_score,

        "Extraction_P": ext_p,
        "Extraction_D": ext_d,
        "Extraction_C": ext_c,
        "Extraction_A": ext_a,
        
        "final_llm_context": str(context or ""),
        "raw_llm_response": str(raw_response or ""),
        "ai_confidence_at_eval": str(confidence or "MEDIUM"),
        "consistency_check": bool(parsed.get("consistency_check", True)),
        
        # Multi-Enabler Traceability
        "enabler_at_eval": f"{enabler_full_name} ({enabler_code})"
    }


def _create_fallback_error(sub_id: str, level: int, error: Exception, context: str, 
                          enabler_full_name: str = "Unknown", enabler_code: str = "UNK") -> Dict[str, Any]:
    """[SAFETY NET] จัดการกรณี LLM หรือ Prompt พัง"""
    logger = logging.getLogger(__name__)
    logger.error(f"🛑 Critical Audit Failure Enabler:{enabler_code} Sub:{sub_id} L{level}: {str(error)}")
    
    return {
        "sub_id": sub_id,
        "level": level,
        "score": 0.0,
        "reason": f"Audit Engine Error: {str(error)}",
        "is_passed": False,
        "consistency_check": False,
        "P_Plan_Score": 0.0, "D_Do_Score": 0.0, "C_Check_Score": 0.0, "A_Act_Score": 0.0,
        "Extraction_P": "-", "Extraction_D": "-", "Extraction_C": "-", "Extraction_A": "-",
        "final_llm_context": str(context or ""),
        "raw_llm_response": "",
        "ai_confidence_at_eval": "ERROR",
        "enabler_at_eval": f"{enabler_full_name} ({enabler_code})"
    }


def _heuristic_fallback_parse(raw_text: str) -> Dict:
    """
    [ENHANCED v2026.1.22] Fallback parse — หา score/PDCA จาก raw text ด้วย regex + keyword
    """
    parsed = {
        "score": 0.0,
        "is_passed": False,
        "reason": "JSON parse failed - fallback heuristic",
        "summary_thai": "ไม่สามารถแยกผลลัพธ์ได้อย่างสมบูรณ์",
        "P_Plan_Score": 0.0,
        "D_Do_Score": 0.0,
        "C_Check_Score": 0.0,
        "A_Act_Score": 0.0,
        "consistency_check": False
    }

    import re

    # หา score หลัก
    score_match = re.search(r"(?:score|คะแนน|total score)\D*([\d\.]+)", raw_text, re.IGNORECASE)
    if score_match:
        try:
            parsed["score"] = float(score_match.group(1))
            parsed["is_passed"] = parsed["score"] >= 0.7
        except:
            pass

    # หา PDCA scores
    pdca_patterns = {
        "P_Plan_Score": r"(?:P_Plan|P|Plan|แผน)\D*([\d\.]+)",
        "D_Do_Score": r"(?:D_Do|D|Do|ปฏิบัติ)\D*([\d\.]+)",
        "C_Check_Score": r"(?:C_Check|C|Check|ตรวจสอบ)\D*([\d\.]+)",
        "A_Act_Score": r"(?:A_Act|A|Act|ปรับปรุง)\D*([\d\.]+)"
    }

    for key, pattern in pdca_patterns.items():
        match = re.search(pattern, raw_text, re.IGNORECASE)
        if match:
            try:
                parsed[key] = float(match.group(1))
            except:
                pass

    parsed["reason"] += f" | Raw snippet: {raw_text[:300]}..."
    return parsed

# ------------------------
# Summarize (FULL VERSION - v2026.4 Ultra-Robust & Zero-Error)
# ------------------------
def create_context_summary_llm(
    context: str,
    sub_criteria_name: str,
    level: int,
    sub_id: str,
    statement_text: str = "",           # Default ว่าง → ไม่ error ถ้าไม่ส่ง
    next_level: int = None,             # Default คำนวณเอง
    llm_executor: Any = None
) -> Dict[str, Any]:
    """
    [SUMMARIZER v2026.4 — Ultra-Robust & Zero-Error]
    - เพิ่ม default สำหรับ statement_text และ next_level → ไม่ error ถ้าเรียกไม่ครบ
    - Fallback ใน prompt ถ้า statement_text ว่าง (สรุปทั่วไป)
    - Log เฉพาะเมื่อใช้ fallback + clean บริบทก่อนส่ง
    - รองรับ retry 4 ครั้ง + hint ใน prompt รอบ retry
    """
    logger = logging.getLogger("AssessmentApp")

    # 1. Validation เบื้องต้น
    if llm_executor is None:
        logger.warning("⚠️ LLM executor is None - returning fallback")
        return {
            "summary": "ระบบ LLM ไม่พร้อมใช้งาน",
            "suggestion_for_next_level": "โปรดตรวจสอบการเชื่อมต่อ LLM",
            "compliance_note": "ไม่สามารถประมวลผลได้",
            "evidence_integrity_score": 0.0
        }

    context_safe = (context or "").strip()
    if len(context_safe) < 30:
        return {
            "summary": "หลักฐานที่พบมีเนื้อหาน้อยเกินกว่าจะสรุปได้ชัดเจน",
            "suggestion_for_next_level": "กรุณาเพิ่มเอกสารหลักฐานที่เกี่ยวข้อง",
            "compliance_note": "ไม่เพียงพอต่อการประเมิน",
            "evidence_integrity_score": 0.1
        }

    # Fallback next_level ถ้าไม่ส่ง
    if next_level is None:
        next_level = min(level + 1, 5)
        logger.debug(f"[SUMMARY] next_level fallback to {next_level} for {sub_id} L{level}")

    # 2. Clean context ก่อนส่ง (เพิ่มเติมจากเดิม)
    context_to_send = re.sub(r'[\x00-\x1F\x7F-\x9F]', ' ', context_safe)[:6500]

    # 3. เตรียม Prompt
    try:
        fallback_statement = statement_text or "ไม่ระบุ statement (สรุปหลักฐานโดยรวม)"
        human_prompt = USER_EVIDENCE_DESCRIPTION_TEMPLATE.format(
            sub_id=f"{sub_id} - {sub_criteria_name}",
            sub_criteria_name=sub_criteria_name,
            level=level,
            statement_text=fallback_statement,
            next_level=next_level,
            context=context_to_send
        )
    except Exception as e:
        logger.error(f"❌ Formatting Error in Summary Prompt: {e}")
        return {
            "summary": "เกิดข้อผิดพลาดในการจัดรูปแบบ prompt",
            "suggestion_for_next_level": "N/A",
            "compliance_note": "ไม่สามารถประมวลผลได้",
            "evidence_integrity_score": 0.0
        }

    # System Instruction เข้มงวด + รองรับ fallback
    system_instruction = (
        f"{SYSTEM_EVIDENCE_DESCRIPTION_PROMPT}\n"
        "### STRICT RULES (ต้องปฏิบัติตามทุกข้อ) ###\n"
        "1. RETURN ONLY VALID JSON OBJECT. ห้ามมีข้อความใด ๆ นอก JSON\n"
        "2. ห้ามใช้ ```json หรือ markdown block\n"
        "3. ใช้ภาษาไทยล้วนในทุก value\n"
        "4. ห้ามมโนข้อมูลที่ไม่มีใน context\n"
        "5. ถ้า statement_text ว่างหรือเป็น 'ไม่ระบุ' ให้สรุปหลักฐานโดยรวมโดยไม่ประเมิน compliance\n"
        "6. ถ้ามี statement_text ให้ประเมิน compliance กับ statement จริง ๆ"
    )

    # 4. Execution Loop with Advanced Parsing + Retry
    max_retries = 4  # เพิ่มเป็น 4 รอบ
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"🔄 Generating Summary {sub_id} L{level} (Attempt {attempt})")

            # เรียก LLM (รองรับทั้ง LangChain & Ollama-style)
            if hasattr(llm_executor, 'generate'):
                raw_response = llm_executor.generate(system=system_instruction, prompts=[human_prompt])
            elif hasattr(llm_executor, 'invoke'):
                raw_response = llm_executor.invoke(human_prompt)
            else:
                raw_response = llm_executor(system_instruction + "\n" + human_prompt)

            # Robust Text Extraction
            res_text = ""
            if hasattr(raw_response, 'generations'):
                res_text = raw_response.generations[0][0].text.strip()
            elif hasattr(raw_response, 'content'):
                res_text = str(raw_response.content).strip()
            else:
                res_text = str(raw_response).strip()

            # 5. ขั้นตอนทำความสะอาดข้อความ (Thai-safe)
            if res_text:
                res_text = res_text.replace('\xa0', ' ').replace('\u200b', '')
                res_text = "".join(c for c in res_text if ord(c) >= 32 or c in "\n\r\t")
                res_text = re.sub(r'```(?:json)?\s*|\s*```', '', res_text).strip()
                res_text = re.sub(r'^[^{\[]+', '', res_text).strip()

            # 6. Robust JSON Extraction
            parsed = _extract_normalized_dict(res_text)

            if parsed and isinstance(parsed, dict):
                summary_val = parsed.get("summary") or parsed.get("สรุป") or ""
                suggestion = parsed.get("suggestion_for_next_level") or parsed.get("คำแนะนำระดับถัดไป") or ""
                compliance = parsed.get("compliance_note") or parsed.get("หมายเหตุความสอดคล้อง") or "ไม่ระบุ"
                score = float(parsed.get("evidence_integrity_score", 0.5))

                if summary_val.strip():
                    logger.info(f"✅ Summary Generated Successfully (Attempt {attempt})")
                    return {
                        "summary": str(summary_val).strip(),
                        "suggestion_for_next_level": str(suggestion).strip() or "ดำเนินการตามเกณฑ์ระดับถัดไป",
                        "compliance_note": str(compliance).strip(),
                        "evidence_integrity_score": max(0.0, min(1.0, score))
                    }

            logger.warning(f"⚠️ Attempt {attempt}: Invalid/empty JSON. Retrying...")
            human_prompt += "\n\n(สำคัญมาก: ตอบเฉพาะ JSON เริ่มต้นด้วย { และจบด้วย } ห้ามมีข้อความอื่นใด)"

            time.sleep(0.8)  # พักนานขึ้นนิดหน่อย

        except Exception as e:
            logger.error(f"❌ Attempt {attempt} Error: {str(e)}")
            time.sleep(1.2)

    # 7. Ultimate Fallback
    logger.error(f"❌ All attempts failed for {sub_id} L{level}")
    return {
        "summary": f"ตรวจพบหลักฐานการดำเนินงานในระดับ {level} แต่ระบบไม่สามารถสรุปได้อย่างสมบูรณ์",
        "suggestion_for_next_level": f"กรุณาตรวจสอบเอกสารเพิ่มเติมสำหรับเกณฑ์ระดับ {next_level or level+1}",
        "compliance_note": "ไม่สามารถประเมินความสอดคล้องได้เนื่องจากข้อผิดพลาด",
        "evidence_integrity_score": 0.3
    }
    
# =================================================================
# 2. Key Normalizer: แก้ไขปัญหา LLM พ่น Key ไม่นิ่ง
# =================================================================
def action_plan_normalize_keys(obj: Any) -> Any:
    """
    [ULTIMATE NORMALIZER v2026.3.26]
    - จัดการปัญหา Key ภาษาไทยที่ LLM อาจเผลอพ่นออกมา (เช่น 'ขั้นตอน' -> 'steps')
    - ล้างอักขระพิเศษและ Newline ที่ทำให้ JSON พัง
    - บังคับ Type ข้อมูลให้ตรงเกณฑ์ (Coercion) เพื่อความปลอดภัยของระบบ UI/Frontend
    """
    if isinstance(obj, list):
        return [action_plan_normalize_keys(i) for i in obj]

    if isinstance(obj, dict):
        # แผนผังการแปลง Key แบบครอบจักรวาล (รวมภาษาไทยและตัวย่อ)
        FIELD_MAPPING = {
            # Level 1: Phase
            "phase": "phase", "เฟส": "phase", "ขั้นตอนหลัก": "phase",
            "goal": "goal", "เป้าหมาย": "goal", "วัตถุประสงค์": "goal",
            "actions": "actions", "กิจกรรม": "actions", "รายการแก้ไข": "actions",

            # Level 2: Action Detail
            "statementid": "statement_id", "id": "statement_id",
            "failedlevel": "failed_level", "ระดับที่ไม่ผ่าน": "failed_level",
            "recommendation": "recommendation", "คำแนะนำ": "recommendation",
            "coachinginsight": "coaching_insight", "insight": "coaching_insight",
            "targetevidencetype": "target_evidence_type", "เอกสารที่ต้องมี": "target_evidence_type",
            "keymetric": "key_metric", "ตัวชี้วัด": "key_metric",

            # Level 3: Steps
            "steps": "steps", "ขั้นตอนย่อย": "steps",
            "step": "step", "ลำดับ": "step",
            "description": "description", "รายละเอียด": "description",
            "responsible": "responsible", "ผู้รับผิดชอบ": "responsible",
            "verificationoutcome": "verification_outcome", "ผลลัพธ์ที่ต้องเห็น": "verification_outcome"
        }

        new_obj = {}
        for raw_key, raw_value in obj.items():
            # 1. Clean Key: ตัดช่องว่าง, Newline และแปลงเป็น Lowercase
            clean_k = str(raw_key).strip().lower().replace("_", "").replace(" ", "")
            
            # 2. Map Key: ถ้าไม่เจอใน Map ให้ใช้ Key เดิมที่ล้างแล้ว
            target_key = FIELD_MAPPING.get(clean_k, clean_k)

            # 3. Value Normalization: ถ้าเป็นเลข ต้องเป็นเลขจริงๆ
            if target_key in ["failed_level", "step"]:
                try:
                    if isinstance(raw_value, str):
                        nums = re.findall(r"\d+", raw_value)
                        value = int(nums[0]) if nums else 0
                    else:
                        value = int(raw_value)
                except: value = 0
            else:
                value = raw_value

            # 4. Recursive Call: ทำต่อในชั้นลูก
            new_obj[target_key] = action_plan_normalize_keys(value)
        return new_obj

    return obj
