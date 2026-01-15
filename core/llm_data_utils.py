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
from pydantic import ValidationError

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
    INITIAL_TOP_K,
    MAX_EVAL_CONTEXT_LENGTH,
    DEFAULT_EMBED_BATCH_SIZE,
    RERANK_THRESHOLD,
    ANALYSIS_FINAL_K
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

def build_multichannel_context_for_level(
    level: int,
    top_evidences: List[Dict[str, Any]],
    previous_levels_map: Optional[Dict[str, Any]] = None,
    previous_levels_evidence: Optional[List[Dict[str, Any]]] = None,
    max_main_context_tokens: int = 3000, 
    max_summary_sentences: int = 4,
    max_context_length: Optional[int] = None, 
    **kwargs
) -> Dict[str, Any]:
    """
    [ULTIMATE OPTIMIZED v2026.8 - ULTRA L1 FALLBACK & MAX DEBUG]
    - สร้าง baseline_summary + aux_summary เท่านั้น (direct ว่างให้ engine จัดการ)
    - ลด threshold หนักสำหรับ L1 (0.15) เพื่อให้มีข้อมูลมากที่สุด
    - Ultra fallback สำหรับ L1: ถ้า direct ยัง 0 → force top chunks เข้า direct + pdca_groups (P/D)
    - Debug log ละเอียดสุด: tag, relevance, text preview, force detail
    - Robust handling: ว่าง → ข้อความ fallback ที่ LLM เข้าใจว่า "มีหลักฐานแต่ไม่ชัดเจน"
    """
    logger = logging.getLogger(__name__)
    K_MAIN = 5
    MIN_RELEVANCE_FOR_AUX = 0.15 if level == 1 else 0.4  # ลดหนักสำหรับ L1

    # 1. Baseline Summary (จาก previous)
    baseline_evidence = previous_levels_evidence or []
    if previous_levels_map:
        for lvl_ev in previous_levels_map.values():
            baseline_evidence.extend(lvl_ev)

    summarizable_baseline = [
        item for item in baseline_evidence[:40]  # เพิ่มเป็น 40 เพื่อครบถ้วน
        if isinstance(item, dict) and (item.get("text") or item.get("content", "")).strip()
    ]

    if not summarizable_baseline:
        baseline_summary = "ไม่มีหลักฐานจากระดับก่อนหน้า (เริ่มต้นที่ Level 1)"
    else:
        baseline_summary = _summarize_evidence_list_short(
            summarizable_baseline,
            max_sentences=max_summary_sentences
        )

    # 2. Direct + Aux Separation + Ultra Debug
    direct, aux_candidates = [], []

    for idx, ev in enumerate(top_evidences[:40], 1):  # จำกัด 40 เพื่อความเร็ว
        if not isinstance(ev, dict):
            continue

        tag = (ev.get("pdca_tag") or ev.get("PDCA") or "Other").upper()
        relevance = ev.get("rerank_score") or ev.get("score", 0.0)
        text_preview = (ev.get('text', '')[:80] + "...") if ev.get('text') else "[No text]"

        # Debug ทุก chunk (ช่วยหาว่าทำไมไม่เข้า direct)
        logger.debug(f"[TAG-CHECK L{level} #{idx}] Rel: {relevance:.3f} | Tag: {tag} | Preview: {text_preview}")

        if tag in {"P", "PLAN", "D", "DO", "C", "CHECK", "A", "ACT"}:
            direct.append(ev)
        elif relevance >= MIN_RELEVANCE_FOR_AUX:
            aux_candidates.append(ev)

    # 3. ย้าย aux ไปเติม direct ถ้าน้อยเกิน
    if len(direct) < K_MAIN:
        need = K_MAIN - len(direct)
        moved = aux_candidates[:need]
        direct.extend(moved)
        aux_candidates = aux_candidates[need:]
        logger.info(f"[DIRECT-FILL] Moved {need} aux chunks to direct (total direct now: {len(direct)})")

    # 4. L1 ULTRA FALLBACK: ถ้า direct ยัง 0 → force top chunks เข้า direct + log ละเอียด
    if level == 1 and len(direct) == 0 and top_evidences:
        need = min(K_MAIN, len(top_evidences))
        forced_chunks = sorted(top_evidences, key=lambda e: e.get("rerank_score", 0) or e.get("score", 0), reverse=True)[:need]
        direct.extend(forced_chunks)
        logger.warning(f"[L1-ULTRA-FALLBACK] No PDCA tag at all → Forced top {need} chunks to direct (ignore tag)")
        for i, ev in enumerate(forced_chunks, 1):
            rel = ev.get("rerank_score", ev.get("score", 'N/A'))
            text = ev.get('text', '')[:80] + "..." if ev.get('text') else "[No text]"
            logger.info(f"[FORCED-DIRECT {i}] Rel: {rel} | Text: {text}")

    # Warning ถ้า direct ยังน้อยมาก (แทบเป็นไปไม่ได้หลัง ultra fallback)
    if len(direct) < 1:
        logger.critical(f"L{level}: CRITICAL - Direct PDCA chunks = 0 หลัง ultra fallback ทั้งหมด!")

    # 5. สร้าง aux_summary
    aux_summary = _summarize_evidence_list_short(aux_candidates, max_sentences=3) if aux_candidates else \
        "ไม่มีหลักฐานรองที่มีคุณภาพเพียงพอ (อาจต้องปรับเกณฑ์หรือเพิ่มเอกสาร)"

    # 6. Return ผลลัพธ์
    debug_meta = {
        "level": level,
        "direct_count": len(direct),
        "aux_count": len(aux_candidates),
        "baseline_count": len(summarizable_baseline),
        "min_relevance_used": MIN_RELEVANCE_FOR_AUX,
        "top_relevance": max((ev.get("rerank_score", 0) or ev.get("score", 0) for ev in top_evidences), default=0)
    }
    logger.info(f"Context L{level} → Direct:{len(direct)} | Aux:{len(aux_candidates)} | Baseline:{len(summarizable_baseline)}")

    return {
        "baseline_summary": baseline_summary,
        "direct_context": "",  # ว่างให้ engine จัดการเอง
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
    # L1-L2: เน้นกว้างเพื่อหาหลักฐานการมีอยู่
    if level <= 2:
        return context[:6000]  
    # L3-L5: ใช้ค่าคงที่ที่กำหนดไว้ (ลด Latency บน Server)
    return context[:MAX_EVAL_CONTEXT_LENGTH]  

def evaluate_with_llm(
    context: str, 
    sub_criteria_name: str, 
    level: int, 
    statement_text: str, 
    sub_id: str, 
    llm_executor: Any = None, 
    require_phase: List[str] = None,
    specific_contextual_rule: str = "พิจารณาตามเกณฑ์มาตรฐาน",
    ai_confidence: str = "MEDIUM",
    confidence_reason: str = "N/A",
    **kwargs
) -> Dict[str, Any]:
    """[SENIOR-AUDIT v36.4] สำหรับระดับ Maturity L3-L5"""
    context_to_send_eval = _get_context_for_level(context, level)
    phases_str = ", ".join(require_phase) if require_phase else "P, D, C, A"
    baseline_summary = kwargs.get("baseline_summary", "").strip()

    try:
        # แมปค่าให้ตรงกับ USER_ASSESSMENT_PROMPT ใน seam_prompts.py
        system_prompt = SYSTEM_ASSESSMENT_PROMPT.format(
            level=level,
            ai_confidence=ai_confidence,
            confidence_reason=confidence_reason
        )

        user_prompt = USER_ASSESSMENT_PROMPT.format(
            sub_criteria_name=sub_criteria_name,
            sub_id=sub_id,
            level=level,
            statement_text=statement_text,
            context=context_to_send_eval[:32000],
            required_phases=phases_str,
            specific_contextual_rule=specific_contextual_rule,
            ai_confidence=ai_confidence,
            confidence_reason=confidence_reason
        )

        if baseline_summary:
            user_prompt += f"\n\n--- BASELINE DATA ---\n{baseline_summary}"

        raw_response = _fetch_llm_response(system_prompt, user_prompt, llm_executor=llm_executor)
        parsed = _robust_extract_json(raw_response)

        return _build_audit_result_object(
            parsed, raw_response, context_to_send_eval, ai_confidence, 
            level=level, sub_id=sub_id
        )

    except Exception as e:
        return _create_fallback_error(sub_id, level, e, context_to_send_eval)

def evaluate_with_llm_low_level(
    context: str,
    sub_criteria_name: str,
    level: int,
    statement_text: str,
    sub_id: str,
    llm_executor: Any = None,
    require_phase: List[str] = None,
    ai_confidence: str = "MEDIUM",
    confidence_reason: str = "N/A",
    **kwargs
) -> Dict[str, Any]:
    """[FOUNDATION-AUDIT v36.4] สำหรับระดับ L1/L2"""
    context_to_send_eval = _get_context_for_level(context, level)
    phases_str = ", ".join(require_phase) if require_phase else "P, D"
    plan_keywords = kwargs.get("plan_keywords", "วิสัยทัศน์, นโยบาย, แผนงาน, คำสั่ง")

    try:
        system_prompt = SYSTEM_LOW_LEVEL_PROMPT.format(
            ai_confidence=ai_confidence,
            confidence_reason=confidence_reason
        )

        user_prompt = USER_LOW_LEVEL_PROMPT.format(
            sub_id=sub_id,
            sub_criteria_name=sub_criteria_name,
            level=level,
            statement_text=statement_text,
            context=context_to_send_eval[:32000],
            required_phases=phases_str,
            plan_keywords=plan_keywords,
            ai_confidence=ai_confidence,
            confidence_reason=confidence_reason
        )

        raw_response = _fetch_llm_response(system_prompt, user_prompt, llm_executor=llm_executor)
        parsed = _robust_extract_json(raw_response)

        return _build_audit_result_object(
            parsed, raw_response, context_to_send_eval, ai_confidence, 
            level=level, sub_id=sub_id
        )

    except Exception as e:
        return _create_fallback_error(sub_id, level, e, context_to_send_eval)

def _build_audit_result_object(parsed: Dict, raw_response: str, context: str, confidence: str, **kwargs):
    """ประกอบร่าง JSON Object พร้อมระบบป้องกันข้อมูลหาย"""
    level = kwargs.get('level', 1)
    sub_id = kwargs.get('sub_id', 'Unknown')

    def clean_score(val, default=0.0):
        try:
            return float(val)
        except:
            return default

    score = clean_score(parsed.get("score"))
    is_passed = parsed.get("is_passed")
    
    # ถ้า AI ตอบไม่ชัดเจน ให้ใช้คะแนนเป็นตัวตัดสิน (Logic ของ New Branch)
    if is_passed is None:
        is_passed = score >= 0.7 if level <= 2 else score >= 1.0

    return {
        "sub_id": str(sub_id),
        "level": int(level),
        "score": score,
        "is_passed": bool(is_passed),
        "reason": parsed.get("reason", "ไม่พบเหตุผลจาก LLM"),
        "P_Plan_Score": clean_score(parsed.get("P_Plan_Score"), score if is_passed else 0.0),
        "D_Do_Score": clean_score(parsed.get("D_Do_Score"), score if is_passed else 0.0),
        "C_Check_Score": clean_score(parsed.get("C_Check_Score")),
        "A_Act_Score": clean_score(parsed.get("A_Act_Score")),
        "Extraction_P": parsed.get("Extraction_P", "-"),
        "Extraction_D": parsed.get("Extraction_D", "-"),
        "Extraction_C": parsed.get("Extraction_C", "-"),
        "Extraction_A": parsed.get("Extraction_A", "-"),
        "final_llm_context": context,
        "raw_llm_response": raw_response,
        "ai_confidence_at_eval": confidence,
        "consistency_check": parsed.get("consistency_check", True)
    }

def _create_fallback_error(sub_id: str, level: int, error: Exception, context: str) -> Dict[str, Any]:
    """[SAFETY NET] จัดการกรณี LLM หรือ Prompt พัง"""
    logger.error(f"🛑 Critical Audit Failure {sub_id} L{level}: {str(error)}")
    
    return {
        "sub_id": sub_id,
        "level": level,
        "score": 0.0,
        "reason": f"Audit Engine Error: {str(error)}",
        "is_passed": False,
        "consistency_check": False,
        "P_Plan_Score": 0.0, "D_Do_Score": 0.0, "C_Check_Score": 0.0, "A_Act_Score": 0.0,
        "Extraction_P": "-", "Extraction_D": "-", "Extraction_C": "-", "Extraction_A": "-",
        "final_llm_context": context,
        "raw_llm_response": "",
        "ai_confidence_at_eval": "ERROR"
    }

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

def create_structured_action_plan(
    recommendation_statements: List[Dict[str, Any]],
    sub_id: str,
    sub_criteria_name: str,
    enabler: str = "KM",
    target_level: int = 5,
    llm_executor: Any = None,
    logger: logging.Logger = None,
    max_retries: int = 3,
    enabler_rules: Dict[str, Any] = {}
) -> List[Dict[str, Any]]:
    """
    [STRATEGIC ROADMAP ENGINE v2026.6.15]
    - สร้างแผนงานเชิงยุทธศาสตร์แบบ Phase-based
    - บูรณาการ Gaps รายเลเวลเข้ากับ Coaching Insights
    - มีระบบ Safety Net (Emergency Fallback) ที่แม่นยำ
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"🚀 [ROADMAP START] Generating plan for {enabler} | {sub_id} (Target L{target_level})")

    # --- 1. วิเคราะห์สถานะ (Condition Mode) ---
    is_sustain_mode = not recommendation_statements
    is_quality_refinement = False
    
    avg_score = 0.0
    if recommendation_statements:
        scores = [float(s.get('score', 0.0)) for s in recommendation_statements]
        avg_score = sum(scores) / len(scores)
        types = [str(s.get('recommendation_type', '')) for s in recommendation_statements]
        
        # ถ้าผ่านเกณฑ์พื้นฐานแต่คะแนนรายข้อไม่เต็ม (เฉลี่ย < 80%)
        if all(t not in ['FAILED_REMEDIATION'] for t in types) and avg_score < 0.8:
            is_quality_refinement = True

    # --- 2. ตั้งค่า Advice Focus (ตัวชี้นำ AI) ---
    specific_rule = enabler_rules.get(enabler, enabler_rules.get("DEFAULT", ""))
    
    if is_sustain_mode:
        advice_focus = f"รักษามาตรฐานความเป็นเลิศ (Best Practice) และขยายผลสู่นวัตกรรมระดับองค์กรสำหรับ {sub_criteria_name}"
        dynamic_max_phases, max_steps = 1, 4
    elif is_quality_refinement:
        advice_focus = f"ยกระดับคุณภาพของหลักฐานเชิงประจักษ์ (Evidence Quality) และการวัดผลความสำเร็จ (KPI) ให้คมชัดขึ้น"
        dynamic_max_phases, max_steps = 1, 3
    else:
        # 🎯 CORE LOGIC: ต้องซ่อมฐาน (Level ต่ำ) ก่อนปีน Level สูง
        advice_focus = (f"เน้นการปิดช่องว่าง (Gap Remediation) ตามวงจร PDCA "
                        f"โดยเริ่มจากพื้นฐานนโยบาย (P) สู่การปฏิบัติจริง (D) และการวัดผล (C/A)")
        dynamic_max_phases = 3 if target_level >= 4 else 2
        max_steps = 3

    if specific_rule:
        advice_focus += f" (กฎเฉพาะ: {specific_rule})"

    # --- 3. เตรียมข้อมูล Gaps & Insights (เรียงลำดับ Level) ---
    if is_sustain_mode:
        stmt_content = f"บรรลุเกณฑ์ระดับ {target_level} อย่างสมบูรณ์แบบ เน้นแผนงานเพื่อความยั่งยืนและการเป็น Role Model"
    else:
        # รวบรวมและกำจัดความซ้ำซ้อน โดยให้ความสำคัญกับเหตุผลที่ชัดเจนที่สุด
        unique_gaps = {}
        for s in recommendation_statements:
            lvl = s.get('level', 0)
            reason = (s.get('context') or s.get('reason') or "ไม่ระบุช่องว่าง").strip()
            # ดึง Missing PDCA Phases มาโชว์ใน Roadmap ด้วย
            missing = s.get('missing_phases', [])
            pdca_suffix = f" [Missing: {','.join(missing)}]" if missing else ""
            
            unique_gaps[lvl] = f"{reason}{pdca_suffix}"
        
        # เรียงจาก L1 -> L5 เพื่อให้ AI เขียนแผนเป็นขั้นตอน
        stmt_content = "\n".join([f"- Level {l}: {unique_gaps[l]}" for l in sorted(unique_gaps.keys())])

    # --- 4. ประกอบ Prompt สำหรับ AI Engine ---
    # ใช้ Global Variable ACTION_PLAN_PROMPT ที่พี่กำหนดไว้
    human_prompt = ACTION_PLAN_PROMPT.format(
        enabler=enabler,
        sub_id=sub_id,
        sub_criteria_name=sub_criteria_name,
        target_level=target_level,
        recommendation_statements_list=stmt_content,
        advice_focus=advice_focus,
        max_phases=dynamic_max_phases,
        max_steps=max_steps,
        max_words_per_step=120,
        language="ภาษาไทย"
    )

    # --- 5. Execution Loop (Retry & Recovery) ---
    for attempt in range(1, max_retries + 1):
        try:
            response = llm_executor.generate(
                system=SYSTEM_ACTION_PLAN_PROMPT,
                prompts=[human_prompt],
                temperature=0.0 # ใช้ 0 เพื่อความนิ่งของโครงสร้าง JSON
            )

            # Extract text safely
            raw_text = getattr(response, 'content', str(response))
            if hasattr(response, 'generations'):
                raw_text = response.generations[0][0].text

            # กู้คืน JSON Array
            items = _extract_json_array_for_action_plan(raw_text, logger)
            if not items: continue

            # Normalize Keys & Validate
            clean_items = action_plan_normalize_keys(items)
            
            try:
                # ตรวจสอบกับ Pydantic Schema
                validated = ActionPlanResult.validate_flexible(clean_items)
                logger.info(f"✅ [SUCCESS] Roadmap Built for {sub_id} (Attempt {attempt})")
                
                # คืนค่าเป็น List ของ Dict
                if hasattr(validated, 'root'):
                    return [i.model_dump() for i in validated.root]
                return [v.model_dump() for v in validated]

            except Exception as ve:
                logger.warning(f"⚠️ Schema mismatch (Attempt {attempt}): {ve}")
                continue

        except Exception as e:
            logger.error(f"💥 Attempt {attempt} failed: {str(e)}")
            time.sleep(0.3)

    # --- 6. Emergency Fallback ---
    logger.warning(f"🆘 Using Predefined Emergency Roadmap for {sub_id}")
    return _get_emergency_fallback_plan(
        sub_id, sub_criteria_name, target_level, is_sustain_mode, is_quality_refinement, enabler
    )

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
def _get_emergency_fallback_plan(
    sub_id: str, 
    sub_criteria_name: str, 
    target_level: int, 
    is_sustain_mode: bool, 
    is_quality_refinement: bool, 
    enabler: str = "KM"
) -> List[Dict[str, Any]]:
    """
    [SAFEGUARD] คืนค่าแผนงานมาตรฐานกรณี LLM พลาด เพื่อให้ระบบไม่ Error
    """
    responsible_party = f"คณะทำงาน {enabler} / หน่วยงานเจ้าของเรื่อง"
    
    title = "แผนการปิดช่องว่าง (Gap Remediation)"
    rec = f"เร่งดำเนินการตามเกณฑ์ {sub_criteria_name} ให้ครบถ้วนตามวงจร PDCA"
    
    if is_sustain_mode:
        title = "แผนรักษามาตรฐาน (Sustainability Plan)"
        rec = f"รักษามาตรฐานความเป็นเลิศในหัวข้อ {sub_criteria_name} และสร้างต้นแบบนวัตกรรม"
    elif is_quality_refinement:
        title = "แผนเสริมความแข็งแกร่ง (Quality Enhancement)"
        rec = f"ปรับปรุงรายละเอียดหลักฐานเชิงประจักษ์ของ {sub_criteria_name} ให้ชัดเจนยิ่งขึ้น"
    
    return [{
        "phase": f"Phase 1: {title}",
        "goal": f"ยกระดับและรักษามาตรฐาน {sub_criteria_name} ({enabler}) สู่ระดับ {target_level}",
        "actions": [{
            "statement_id": sub_id, 
            "failed_level": target_level,
            "recommendation": rec, 
            "target_evidence_type": "Evidence Package / รายงานสรุปผล",
            "key_metric": "ความครบถ้วนของหลักฐานตามเกณฑ์ระดับเป้าหมาย 100%",
            "steps": [
                {
                    "step": 1, 
                    "description": f"ทบทวนช่องว่างจากรายงานผลการประเมิน และจัดทำแผนปฏิบัติการเฉพาะหน้าเพื่อเก็บรวบรวมหลักฐาน", 
                    "responsible": responsible_party, 
                    "tools_templates": "SE-AM Gap Analysis Template", 
                    "verification_outcome": "remediation_plan_report.pdf"
                },
                {
                    "step": 2, 
                    "description": "รวบรวมและตรวจสอบความถูกต้องของหลักฐานให้สอดคล้องกับหัวข้อที่ยังขาดหาย (PDCA)", 
                    "responsible": responsible_party, 
                    "tools_templates": "Checklist เกณฑ์ SE-AM", 
                    "verification_outcome": "evidence_package_final.pdf"
                }
            ]
        }]
    }]

def _extract_json_array_for_action_plan(text: Any, logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    ระบบกู้คืน JSON ที่พังหรือเจนไม่จบ (Robust JSON Extractor)
    """
    try:
        if not isinstance(text, str): text = str(text) if text is not None else ""
        if not text.strip(): return []

        # 1. ลบ Markdown tags
        clean_text = re.sub(r'```(?:json)?\s*([\s\S]*?)\s*```', r'\1', text, flags=re.IGNORECASE).strip()

        # 2. ค้นหาขอบเขต JSON
        start_idx = clean_text.find('[')
        end_idx = clean_text.rfind(']')

        if start_idx == -1:
            # ลองหาแบบ Object เดียว {} แล้วครอบด้วย []
            start_idx = clean_text.find('{')
            end_idx = clean_text.rfind('}')
            if start_idx == -1: return []
            json_candidate = f"[{clean_text[start_idx:end_idx + 1]}]"
        else:
            json_candidate = clean_text[start_idx:end_idx + 1]

        # 3. ล้าง Control characters และ Trailing commas
        json_candidate = "".join(char for char in json_candidate if ord(char) >= 32 or char in "\n\r\t")
        json_candidate = re.sub(r',\s*([\]}])', r'\1', json_candidate) # ลบ comma ตัวสุดท้าย

        def try_parse(content):
            try:
                data = json5.loads(content)
                return data if isinstance(data, list) else [data]
            except: return None

        # 4. พยายาม Parse และซ่อมแซมกรณีตัดจบ
        result = try_parse(json_candidate)
        if not result:
            # กรณี LLM พ่นมาไม่จบ พยายามเติมวงเล็บปิดให้
            for suffix in ["]", "}", "}]", "}\n]"]:
                result = try_parse(json_candidate + suffix)
                if result: break

        return result or []
    except Exception as e:
        logger.error(f"JSON Extraction failed: {str(e)}")
        return []
