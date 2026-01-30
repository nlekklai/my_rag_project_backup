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
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Callable, TypeVar, Set
import json5
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
import os
import unicodedata
# แนะนำ: pip install json-repair (ถ้ายังไม่มี) เพื่อกู้ JSON พังได้ดีมาก
try:
    from json_repair import repair_json
except ImportError:
    repair_json = None  # ถ้าไม่มี จะใช้ manual repair แทน

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
# Retrieval: retrieve_context_with_filter (Revised & Optimized)
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
    top_k: int = 100, 
) -> Dict[str, Any]:
    """
    [ULTIMATE REVISED] 
    - รองรับการทำ Batch Reranking ตาม DEFAULT_EMBED_BATCH_SIZE (Mac 16 / CUDA 128)
    - ปรับปรุงการ Sync คะแนนให้แม่นยำ ป้องกันปัญหาคะแนนลดลงเกินจริง
    - ใช้ Retriever Caching เพื่อลดภาระการโหลด BM25 บน Mac
    """
    start_time = time.time()
    manager = vectorstore_manager
    if not manager:
        logger.error("❌ VectorStoreManager is missing!")
        return {"top_evidences": [], "aggregated_context": "Missing VSM", "retrieval_time": 0}

    # 1. เตรียม Query และ Filter
    queries_to_run = [query] if isinstance(query, str) else list(query or [""])
    collection_name = get_doc_type_collection_key(doc_type, enabler or "KM")
    
    target_ids = set()
    if stable_doc_ids: target_ids.update([str(i) for i in stable_doc_ids])
    if mapped_uuids: target_ids.update([str(i) for i in mapped_uuids])
    if sequential_chunk_uuids: target_ids.update([str(i) for i in sequential_chunk_uuids])
    
    where_filter = _create_where_filter(
        stable_doc_ids=list(target_ids) if target_ids else None,
        subject=subject,
        year=year,
        tenant=tenant
    )

    # 2. Hybrid Retrieval (Vector + BM25) พร้อมระบบ Cache
    all_source_chunks = []

    # 2.1 แทรก Priority Docs (ถ้ามี)
    if priority_docs_input:
        for doc in priority_docs_input:
            if not doc: continue
            if isinstance(doc, dict):
                pc = doc.get('page_content') or doc.get('text') or ''
                meta = doc.get('metadata') or {}
                if pc.strip(): all_source_chunks.append(LcDocument(page_content=pc, metadata=meta))
            elif hasattr(doc, 'page_content'):
                all_source_chunks.append(doc)

    # 2.2 ค้นหาจริง (ใช้ Retriever Cache เพื่อประหยัด RAM บน Mac)
    try:
        if not hasattr(manager, '_retriever_cache'):
            manager._retriever_cache = {}
        
        if collection_name not in manager._retriever_cache:
            logger.info(f"🧬 [CACHE-MISS] Initializing Hybrid Retriever for: {collection_name}")
            manager._retriever_cache[collection_name] = manager.get_retriever(collection_name=collection_name)
        
        full_retriever = manager._retriever_cache[collection_name]
        base_retriever = getattr(full_retriever, "base_retriever", full_retriever)
        
        search_kwargs = {"k": top_k}
        if where_filter: search_kwargs["where"] = where_filter

        for q in queries_to_run:
            if not q or len(q.strip()) < 2: continue
            docs = base_retriever.invoke(q, config={"configurable": {"search_kwargs": search_kwargs}})
            if docs: all_source_chunks.extend(docs)
    except Exception as e:
        logger.error(f"❌ Retrieval failure: {e}")

    # 3. Deduplication (Deterministic MD5)
    unique_map: Dict[str, LcDocument] = {}
    for doc in all_source_chunks:
        if not doc or not doc.page_content.strip(): continue
        md = doc.metadata or {}
        c_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
        uid = str(md.get("chunk_uuid") or f"hash-{c_hash}")
        if uid not in unique_map:
            unique_map[uid] = doc

    candidates = list(unique_map.values())

    # 4. Batch Reranking (ปรับปรุงเพื่อรองรับ Batch Size ตาม Device)
    final_scored_docs = []
    reranker = get_global_reranker()

    if reranker and candidates and queries_to_run:
        try:
            # ใช้ Query แรกเป็นแกนกลางในการ Rerank
            main_query = queries_to_run[0]
            
            # [CRITICAL] เรียกใช้ Batch Size จาก global_vars (16 สำหรับ Mac) 
            # เพื่อป้องกันความเร็วตกหรือคะแนนเพี้ยนกรณีส่งเข้าไปเยอะเกินไปทีเดียว
            final_scored_docs = reranker.compress_documents(
                documents=candidates, 
                query=main_query,
                batch_size=DEFAULT_EMBED_BATCH_SIZE
            )
        except Exception as e:
            logger.error(f"⚠️ Rerank Error: {e}")
            final_scored_docs = candidates
    else:
        final_scored_docs = candidates

    # 5. Ranking & Score Extraction (แม่นยำขึ้น)
    def extract_score(d) -> float:
        m = d.metadata or {}
        # ดึงคะแนนจาก attribute ของ Reranker ก่อน ถ้าไม่มีค่อยไปดูใน Metadata
        if hasattr(d, "relevance_score"): return float(d.relevance_score)
        return float(m.get("relevance_score") or m.get("score") or m.get("rerank_score") or 0.0)

    final_scored_docs.sort(key=extract_score, reverse=True)

    # 6. คัดเลือกผลลัพธ์และจัดรูปแบบ
    top_evidences = []
    aggregated_parts = []
    final_limit = ANALYSIS_FINAL_K

    for doc in final_scored_docs:
        if len(top_evidences) >= final_limit:
            break
            
        score = extract_score(doc)
        
        # กรองทิ้งหากคะแนนต่ำกว่า Threshold (0.20)
        if score < RERANK_THRESHOLD and RERANK_THRESHOLD > 0:
            continue

        md = doc.metadata or {}
        text = doc.page_content.strip()
        
        # สกัด Metadata (Page / Source / PDCA)
        page = str(md.get("page_label") or md.get("page_number") or md.get("page") or "N/A")
        source = md.get("source_filename") or md.get("source") or "Unknown"
        pdca = md.get("pdca_tag", "Other")

        # [IMPORTANT] Sync คะแนนกลับเข้า metadata เพื่อให้ Prompt Assessment เห็นคะแนนจริง
        md["score"] = score
        md["relevance_score"] = score

        top_evidences.append({
            "doc_id": str(md.get("stable_doc_uuid") or md.get("doc_id") or ""),
            "chunk_uuid": str(md.get("chunk_uuid") or str(uuid.uuid4())),
            "source": source,
            "text": text,
            "page": page,
            "pdca_tag": pdca,
            "score": score,
            "metadata": md
        })
        aggregated_parts.append(f"[{pdca}] [ไฟล์: {source} หน้า: {page}] {text}")

    total_time = round(time.time() - start_time, 3)
    max_score = extract_score(final_scored_docs[0]) if final_scored_docs else 0.0
    
    logger.info(f"🏁 Retrieval Finished: {len(top_evidences)} chunks | Max Score: {max_score:.4f} | Time: {total_time}s")

    return {
        "top_evidences": top_evidences,
        "aggregated_context": "\n\n---\n\n".join(aggregated_parts) if aggregated_parts else "ไม่พบหลักฐาน",
        "retrieval_time": total_time,
        "max_score": max_score
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

def _format_evidence_item(doc: LcDocument, score: float) -> Dict[str, Any]:
    """ Helper สำหรับจัดรูปแบบข้อมูล Output ให้เป็นมาตรฐานเดียวกัน """
    m = doc.metadata or {}
    return {
        "text": doc.page_content,
        "source_filename": m.get("source_filename") or m.get("source") or "Evidence",
        "page_label": str(m.get("page_label") or m.get("page_number") or m.get("page") or "N/A"),
        "doc_id": str(m.get("stable_doc_uuid") or m.get("doc_id") or ""),
        "chunk_uuid": str(m.get("chunk_uuid") or str(uuid.uuid4())),
        "pdca_tag": m.get("pdca_tag") or "Content",
        "rerank_score": score,
        "is_evidence": True,
        "metadata": m
    }

# =====================================================================
# 🚀 Ultimate Version: retrieve_context_with_rubric (FULL REVISED)
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
    top_k: int = 150,         
    rubric_top_k: int = 15,  
    k_to_rerank: int = 30    
) -> Dict[str, Any]:
    """
    [PRODUCTION REVISED] ระบบดึงบริบท (Context) แบบ Hybrid: Rubric + Evidence
    - ป้องกันความซ้ำซ้อนด้วย Content-Based MD5 Deduplication
    - ใช้ Batch Reranking เพื่อความเสถียรบน Mac/Server
    - รักษาความคงที่ของ ID สำหรับการประเมิน SE-AM
    """
    start_time = time.time()
    vsm = vectorstore_manager

    # --- 1. การบริหารจัดการ Collection (Auto-Switch) ---
    if hasattr(vsm, 'doc_type') and vsm.doc_type != doc_type:
        logger.info(f"🔄 Switching VSM doc_type to: {doc_type}")
        vsm.close()
        vsm.__init__(tenant=tenant, year=year, doc_type=doc_type, enabler=enabler)

    evidence_collection = get_doc_type_collection_key(doc_type, enabler or "KM")
    
    rubric_results = []
    unique_evidence_map: Dict[str, LcDocument] = {}

    # --- 2. การดึง Rubrics (เกณฑ์มาตรฐาน SE-AM) ---
    # ใช้ Helper check_rubric_ready เพื่อความเงียบ (Silent Mode)
    if is_rubric_ready(tenant): 
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
    else:
        logger.info(f"ℹ️ Rubric skip: Collection 'seam' not found for tenant: {tenant}")

    # --- 3. การดึง Evidence (หลักฐานการดำเนินงาน) ---
    try:
        evidence_chroma = vsm._load_chroma_instance(evidence_collection)
        if not evidence_chroma:
            return {"top_evidences": [], "rubric_context": rubric_results, "retrieval_time": 0}

        # สร้าง Filter ให้สอดคล้องกับ DataType (Int/Str)
        where_filter = None
        if stable_doc_ids:
            ids_list = [str(i).strip().lower() for i in stable_doc_ids if i]
            if len(ids_list) == 1:
                where_filter = {"stable_doc_uuid": ids_list[0]}
            else:
                where_filter = {"stable_doc_uuid": {"$in": ids_list}}
            
            # ⚓ 3.1 Anchor Chunks (ส่วนโครงสร้างหลัก เช่น หน้า 1-5)
            # ดันคะแนนให้สูง (0.95) เพื่อให้เป็นบริบทพื้นฐานที่ AI ต้องเห็น
            anchors = evidence_chroma.get(where=where_filter, limit=10)
            if anchors and anchors.get('documents'):
                for i in range(len(anchors['documents'])):
                    content = anchors['documents'][i]
                    md = dict(anchors['metadatas'][i] or {}) 
                    
                    # ใช้ MD5 เพื่อให้ ID นิ่งตลอดการรัน (Deterministic)
                    content_hash = hashlib.md5(content.encode()).hexdigest()
                    uid = str(md.get("chunk_uuid") or f"anchor-{content_hash}")
                    
                    if uid not in unique_evidence_map:
                        md.update({
                            "score": 0.95,
                            "relevance_score": 0.95,
                            "is_anchor": True
                        })
                        unique_evidence_map[uid] = LcDocument(page_content=content, metadata=md)

        # 🔍 3.2 Semantic Search (ค้นหาตามความหมายของ Query)
        search_results = evidence_chroma.similarity_search(query, k=top_k, filter=where_filter)
        for d in search_results:
            # เปลี่ยนจาก hash() เป็น MD5 เพื่อความแม่นยำข้าม Environment (Mac/Server)
            c_hash = hashlib.md5(d.page_content.encode()).hexdigest()
            uid = d.metadata.get("chunk_uuid") or c_hash
            if uid not in unique_evidence_map:
                unique_evidence_map[uid] = d

        candidates = list(unique_evidence_map.values())

        # --- 4. BATCH RERANKING (ระบบคัดเลือกความหมายขั้นสูง) ---
        evidence_results = []
        reranker = get_global_reranker()
        
        if reranker and candidates and query:
            try:
                # ดึง Batch Size จาก ENV (Mac: 16 / Server: 128)
                from config.global_vars import DEFAULT_EMBED_BATCH_SIZE
                batch_size = DEFAULT_EMBED_BATCH_SIZE
                scored_candidates = []
                
                logger.info(f"🚀 Batch Reranking {len(candidates)} chunks...")
                for i in range(0, len(candidates), batch_size):
                    batch = candidates[i : i + batch_size]
                    reranked_batch = reranker.compress_documents(documents=batch, query=query)
                    scored_candidates.extend(reranked_batch)
                
                # เรียงลำดับคะแนน (Desc)
                scored_candidates = sorted(
                    scored_candidates, 
                    key=lambda x: getattr(x, "relevance_score", 0), 
                    reverse=True
                )
                
                for r in scored_candidates[:k_to_rerank]:
                    doc = r.document if hasattr(r, "document") else r
                    m = doc.metadata or {}
                    score = getattr(r, "relevance_score", 0.0)
                    
                    # Sync คะแนนกลับเข้า Metadata เพื่อให้ระบบประเมินนำไปใช้ต่อ
                    m["rerank_score"] = score
                    m["score"] = score
                    
                    evidence_results.append({
                        "text": doc.page_content,
                        "source_filename": m.get("source_filename") or m.get("source") or "Evidence",
                        "page_label": str(m.get("page_label") or m.get("page_number") or m.get("page") or "N/A"),
                        "doc_id": str(m.get("stable_doc_uuid") or m.get("doc_id") or ""),
                        "chunk_uuid": str(m.get("chunk_uuid") or str(uuid.uuid4())),
                        "pdca_tag": m.get("pdca_tag") or "Content",
                        "rerank_score": score,
                        "is_evidence": True,
                        "metadata": m
                    })
            except Exception as e:
                logger.error(f"⚠️ Rerank failed: {e}")
                # Fallback: ใช้ผลการดึงแบบปกติหาก Reranker มีปัญหา
                for d in candidates[:k_to_rerank]:
                    evidence_results.append(_format_evidence_item(d, 0.0))
        
        # กรณีไม่มี Reranker หรือดึงข้อมูลดิบ
        elif candidates:
            for d in candidates[:k_to_rerank]:
                evidence_results.append(_format_evidence_item(d, 0.0))

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


# =====================================================================
# 🚀 Ultimate Version: retrieve_context_by_doc_ids (FULL REVISED)
# =====================================================================
def retrieve_context_by_doc_ids(
    doc_uuids: List[str],
    doc_type: str,
    enabler: Optional[str] = None,
    vectorstore_manager = None,
    limit: int = 200,          # 🚀 เพิ่ม limit เพื่อให้ครอบคลุมการประเมินไฟล์ใหญ่
    tenant: Optional[str] = None,
    year: Optional[Union[int, str]] = None,
) -> Dict[str, Any]:
    """
    [PRODUCTION REVISED] ดึง Chunks ทั้งหมดจากเอกสารที่ระบุ (Hydration)
    - ใช้ MD5 Hashing เพื่อทำ Deduplication ป้องกันหน้าซ้ำ
    - รักษาโครงสร้าง Metadata ให้เหมือนกับฟังก์ชัน Search หลัก
    - รองรับ Multi-tenant และการแยก Collection ตาม Enabler/Year
    """
    start_time = time.time()
    
    # 1. จัดการ VectorStore Manager
    vsm = vectorstore_manager
    if not vsm:
        from core.vectorstore import VectorStoreManager # ป้องกัน Circular Import
        vsm = VectorStoreManager(tenant=tenant, year=year)

    # 2. ค้นหาชื่อ Collection ที่ถูกต้อง
    collection_name = get_doc_type_collection_key(doc_type=doc_type, enabler=enabler or "KM")

    chroma = vsm._load_chroma_instance(collection_name)
    if not chroma:
        logger.error(f"❌ Collection '{collection_name}' not found for hydration")
        return {"top_evidences": [], "retrieval_time": 0}

    if not doc_uuids:
        return {"top_evidences": [], "retrieval_time": 0}

    # ล้างค่า ID ให้สะอาด (Trim & Lower)
    ids_to_query = [str(u).strip().lower() for u in doc_uuids if u]
    logger.info(f"💧 Hydrating Context: {len(ids_to_query)} docs from '{collection_name}'")

    try:
        # 3. ดึงข้อมูลจาก ChromaDB โดยตรง (Direct Get)
        # ใช้ Metadata Filter เพื่อเจาะจงเฉพาะไฟล์ที่เลือก
        results = chroma._collection.get(
            where={"stable_doc_uuid": {"$in": ids_to_query}},
            limit=limit,
            include=["documents", "metadatas"]
        )
    except Exception as e:
        logger.error(f"⚠️ Hydration query failed: {e}")
        return {"top_evidences": [], "retrieval_time": 0}

    evidences = []
    seen_contents = set()

    # 4. จัดรูปแบบข้อมูล (Formatting & Deduplication)
    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []

    for doc_content, meta in zip(documents, metadatas):
        if not doc_content or not doc_content.strip():
            continue
            
        # 🎯 ใช้ MD5 แทน hash() เพื่อความคงที่ (Deterministic)
        content_hash = hashlib.md5(doc_content.encode()).hexdigest()
        if content_hash in seen_contents:
            continue
        seen_contents.add(content_hash)

        # ดึง Metadata ที่สำคัญ (Fallback แบบเดียวกับ Search)
        m = meta or {}
        p_val = str(m.get("page_label") or m.get("page_number") or m.get("page") or "N/A")
        
        # 🎯 ปรับ Key ให้ตรงกับ _format_evidence_item เพื่อให้ Router ใช้งานได้ทันที
        evidences.append({
            "text": doc_content.strip(),
            "source_filename": m.get("source_filename") or m.get("source") or "Evidence",
            "page_label": p_val,
            "doc_id": str(m.get("stable_doc_uuid") or m.get("doc_id") or ""),
            "chunk_uuid": str(m.get("chunk_uuid") or content_hash),
            "pdca_tag": m.get("pdca_tag") or "Content",
            "rerank_score": float(m.get("score") or m.get("rerank_score") or 0.85), # Hydration Score
            "is_evidence": True,
            "metadata": m
        })

    # เรียงตามหน้าเพื่อให้ AI อ่านง่ายขึ้น (ถ้ามีเลขหน้า)
    try:
        evidences.sort(key=lambda x: int(x['page_label']) if x['page_label'].isdigit() else 999)
    except:
        pass

    retrieval_time = round(time.time() - start_time, 3)
    logger.info(f"✅ Hydration success: {len(evidences)} chunks in {retrieval_time}s")

    return {
        "top_evidences": evidences,
        "retrieval_time": retrieval_time
    }


def _fetch_llm_response(
    system_prompt: str = "",
    user_prompt: str = "",
    max_retries: int = 3,
    llm_executor: Any = None
) -> str:
    """
    [IRONCLAD LLM FETCHER - FINAL POLISH v2026.1.26]
    - บังคับ LLM ให้ส่ง VALID JSON เท่านั้น (prompt เข้มงวด + ตัวอย่างหลายรูปแบบ)
    - Multi-stage clean-up + greedy extraction + json_repair (ถ้ามี)
    - Retry ฉลาด + exponential backoff + prompt variation เมื่อล้มเหลว
    - Log raw + cleaned ทุก attempt เพื่อ debug (ภาษาไทยอ่านได้ทันที)
    - คืน JSON string ที่สะอาดเสมอ (fallback ถ้าพังหมด)
    - เพิ่ม: Avoid Unicode escape sequences ใน Thai text
    """
    if llm_executor is None:
        raise ConnectionError("LLM instance not initialized.")

    # 1. System Prompt เข้มงวดสุด (รวม Avoid Unicode escape + ตัวอย่างชัดเจน)
    enforced_system = (
        (system_prompt or "").strip() + "\n\n"
        "### ABSOLUTE RULES - MUST FOLLOW OR FAIL ###\n"
        "1. Respond with **ONLY** valid JSON. NO text before or after.\n"
        "2. NO markdown (```json, ```), NO explanations, NO 'Here is...', NO apologies.\n"
        "3. Use double quotes for ALL keys and string values.\n"
        "4. If string contains double quote, escape it as \\\" or use single quote instead.\n"
        "5. All braces { } and brackets [ ] MUST be balanced.\n"
        "6. IMPORTANT: For Thai text, use normal Thai characters ONLY.\n"
        "   DO NOT use Unicode escape sequences (e.g. \\u0e23 \\u0e35 \\u0e07).\n"
        "   Output readable Thai directly (เช่น \"จัดทำคำสั่ง\" ไม่ใช่ \"\\u0e08\\u0e31\\u0e14\\u0e17\\u0e33\").\n"
        "7. Return ONLY array or object. Examples:\n"
        '   [{"action": "จัดทำคำสั่งแต่งตั้งคณะทำงาน KM", "target_evidence": "ประกาศองค์กร ฉบับที่..."}]\n'
        '   [{"score": 1.0, "is_passed": true, "reason": "เหตุผลเป็นภาษาไทยชัดเจน"}]\n'
        "FAILURE TO COMPLY = INVALID RESPONSE"
    )

    messages = [
        {"role": "system", "content": enforced_system},
        {"role": "user",   "content": (user_prompt or "").strip()}
    ]

    for attempt in range(1, max_retries + 1):
        try:
            # 2. LLM Call (temperature 0.0 เพื่อความ deterministic สูงสุด)
            response = llm_executor.invoke(messages, config={"temperature": 0.0})

            raw_text = ""
            if hasattr(response, "content"):
                raw_text = str(response.content).strip()
            elif hasattr(response, "text"):
                raw_text = str(response.text).strip()
            else:
                raw_text = str(response or "").strip()

            # Log raw response เต็ม (จำกัด 1500 ตัวอักษร)
            logger.critical(f"[LLM-RAW attempt {attempt}] (len={len(raw_text)}):\n{raw_text[:1500]}{'...' if len(raw_text) > 1500 else ''}")

            # 3. Multi-stage Clean-up
            # Stage 1: ลบ markdown และ code fences
            cleaned = re.sub(r'```(?:json)?\s*|\s*```', '', raw_text).strip()

            # Stage 2: ลบ whitespace เกิน + trailing comma + unbalanced quotes
            cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()

            # Stage 3: Greedy หา JSON block ใหญ่สุด (รองรับทั้ง object และ array)
            json_match = re.search(r'(\{[\s\S]*?\}|\[[\s\S]*?\])', cleaned, re.DOTALL)
            if json_match:
                extracted = json_match.group(1)
            else:
                extracted = cleaned  # ถ้าไม่เจอ ให้ลองทั้งหมด

            # Stage 4: ใช้ json_repair ถ้ามี (ดีมากสำหรับ LLM output พัง)
            if repair_json:
                try:
                    repaired = repair_json(extracted)
                    json.loads(repaired)  # ทดสอบ parse
                    logger.debug(f"[JSON-REPAIR-SUCCESS] attempt {attempt}")
                    return repaired
                except Exception as repair_err:
                    logger.debug(f"[JSON-REPAIR-FAIL] {str(repair_err)} → fallback to manual")

            # Stage 5: Manual salvage (ลบ control chars + unbalanced)
            extracted = re.sub(r'[\x00-\x1F\x7F]', '', extracted)  # ลบ control chars
            try:
                json.loads(extracted)
                logger.debug(f"[MANUAL-PARSE-SUCCESS] attempt {attempt}")
                return extracted
            except json.JSONDecodeError as je:
                logger.warning(f"[JSON-PARSE-FAIL attempt {attempt}]: {str(je)}")

            # ถ้าพังหมด → retry ด้วย prompt variation
            if attempt < max_retries:
                messages[1]["content"] += (
                    f"\n\nPrevious attempt failed (invalid JSON). "
                    f"Fix it now and return **ONLY** valid JSON. No extra text."
                )
                time.sleep(1.5 ** attempt)  # exponential backoff
                continue

        except Exception as e:
            logger.error(f"[LLM-EXCEPTION attempt {attempt}]: {str(e)}")
            if attempt < max_retries:
                time.sleep(1.5 ** attempt)
            else:
                break

    # Ultimate Fallback (คืน JSON มาตรฐานเสมอ)
    logger.critical(f"[LLM-FINAL-FALLBACK] All {max_retries} attempts failed")
    return json.dumps({
        "score": 0.0,
        "is_passed": False,
        "reason": "Failed to generate valid JSON after retries (system fallback)",
        "fallback": True
    }, ensure_ascii=False)  # ensure_ascii=False เพื่อให้ภาษาไทยไม่ escape

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
    chunks: list = None,
    **kwargs
) -> str:
    """
    [DYNAMIC REVISED 2026] 
    ปรับแต่งการดึง Context ตาม RAG_RUN_MODE ใน .env อัตโนมัติ
    - Mac (LOCAL_OLLAMA): ประหยัด Context เพื่อความเร็ว (8B Model)
    - Server (PRODUCTION): ขยาย Context เต็มศักยภาพ (70B Model)
    """
    if not context:
        return "ไม่พบข้อมูลหลักฐาน"

    import os
    # 1. ตรวจสอบโหมดการรันจาก .env ที่มีอยู่แล้ว
    run_mode = os.getenv("RAG_RUN_MODE", "LOCAL_OLLAMA")
    is_server = run_mode == "PRODUCTION"

    # 2. กำหนดขีดจำกัด (Limits) ตามความแรงของเครื่อง (Detect จาก env โดยตรง)
    if is_server:
        # ฝั่ง Server: ใช้ ANALYSIS_FINAL_K จาก .env หรือ Default 35
        max_chunks = int(os.getenv("ANALYSIS_FINAL_K", 35))
        max_chars = 20000 if level > 2 else 25000  # 70B รับบริบทได้กว้างกว่า
    else:
        # ฝั่ง Mac: บีบให้เล็กลงเพื่อความลื่นไหลของ 8B
        max_chunks = int(os.getenv("ANALYSIS_FINAL_K", 15))
        max_chars = 8000 if level > 2 else 10000

    # 3. กรณีมี Chunks ( List of Dicts )
    if chunks:
        # เรียงลำดับตาม Score (ดีที่สุดไว้บนสุด)
        sorted_chunks = sorted(
            chunks,
            key=lambda c: float(c.get('rerank_score') or c.get('score') or 0.0),
            reverse=True
        )
        
        selected = sorted_chunks[:max_chunks]
        parts = []
        
        for i, c in enumerate(selected, 1):
            score = c.get('rerank_score') or c.get('score', 'N/A')
            source = c.get('source') or c.get('source_filename', 'Unknown')
            page = c.get('page') or c.get('page_label', 'N/A')
            text = c.get('text', '').strip()
            pdca = c.get('pdca_tag', 'N/A')
            
            if text:
                # จัด Format ให้ AI อ่านง่าย เพื่อให้นำไปเขียน Markdown Table ต่อได้
                parts.append(
                    f"### หลักฐานชิ้นที่ {i} | Tag: {pdca} | Score: {score} | แหล่งที่มา: {source} หน้า {page}\n"
                    f"{text}\n"
                    f"{'-'*40}"
                )

        final_text = "\n\n".join(parts)
        
        # Hard Cap ตัวอักษรป้องกัน Context Overload
        if len(final_text) > max_chars:
            final_text = final_text[:max_chars] + "\n... [ข้อมูลถูกตัดเพื่อความเหมาะสม]"
        return final_text

    # 4. Fallback กรณีส่งมาเป็น String ยาวๆ
    return context[:max_chars] + ("... [truncated]" if len(context) > max_chars else "")

# =================================================================
# 1. CORE LLM EVALUATION FUNCTIONS (Revised for New Prompts)
# =================================================================
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
    confidence_reason: str = "N/A", # ✅ ประกาศที่หัวฟังก์ชันชัดเจนตามต้องการ
    pdca_context: str = "", 
    **kwargs
) -> Dict[str, Any]:
    """
    [EXPLICIT REVISED v2026.01.27] - การประเมินระดับสูง (L3-L5)
    - STRATEGY: ประกาศ Argument ชัดเจน และล้างความซ้ำซ้อนใน kwargs
    """
    logger = logging.getLogger(__name__)

    # 1. 🛡️ [SHIELDING] ล้างค่าที่อาจซ้ำซ้อนใน kwargs เพื่อป้องกัน TypeError 
    # เราใช้ค่าจาก Argument ที่ประกาศข้างบนเป็นหลัก ส่วนใน kwargs ให้ลบทิ้งไป
    kwargs.pop("confidence_reason", None)
    
    # ดึงค่า Enabler (ถ้าไม่มีให้ใช้ค่า Default)
    e_code = str(kwargs.pop("enabler", "UNK")).upper()
    e_name_th = str(kwargs.pop("enabler_name_th", f"ด้าน {e_code}"))

    # 2. [PREPARING] เตรียมข้อมูลพื้นฐาน
    ctx_raw = str(context or "ไม่พบข้อมูลหลักฐาน")
    pdca_ctx = str(pdca_context or "ไม่มีข้อมูลแยกหมวดหมู่")
    context_to_send_eval = _get_context_for_level(ctx_raw, level) or ""
    phases_str = ", ".join(str(p).strip() for p in (required_phases or [])) if required_phases else "P, D, C, A"

    try:
        # 3. 🎯 [FORMATTING] ฉีดข้อมูลเข้า Prompt
        full_prompt = USER_ASSESSMENT_PROMPT.format(
            sub_criteria_name=sub_criteria_name,
            sub_id=sub_id,
            level=level,
            statement_text=statement_text,
            context=context_to_send_eval[:25000], 
            pdca_context=pdca_ctx[:8000],         
            required_phases=phases_str,
            specific_contextual_rule=specific_contextual_rule,
            ai_confidence=ai_confidence,
            confidence_reason=confidence_reason, # ✅ ใช้จาก Argument โดยตรง
            enabler=e_code,
            enabler_name_th=e_name_th,
            **kwargs # ✅ ข้อมูลที่เหลือ (focus_points, evidence_guidelines) จะถูกฉีดเข้าที่นี่
        )
        
        # เพิ่ม Baseline Summary (ถ้ามีส่งมาใน kwargs)
        if kwargs.get("baseline_summary"):
            full_prompt += f"\n\n--- BASELINE DATA ---\n{kwargs['baseline_summary']}"

        system_msg = f"Expert SE-AM Auditor for {e_name_th} ({e_code})"
        
        # 4. [EXECUTION] เรียก LLM
        raw_response = _fetch_llm_response(
            system_prompt=system_msg,
            user_prompt=full_prompt,
            llm_executor=llm_executor
        )

        # 5. [PARSING] แปลงเป็น Audit Object
        parsed = _robust_extract_json(raw_response)
        return _build_audit_result_object(
            parsed, raw_response, context_to_send_eval, ai_confidence, 
            level=level, sub_id=sub_id, enabler_full_name=e_name_th, enabler_code=e_code
        )

    except Exception as e:
        logger.error(f"🛑 Evaluation Error Sub:{sub_id} L{level}: {str(e)}")
        return _create_fallback_error(sub_id, level, e, context_to_send_eval, e_name_th, e_code)


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
    confidence_reason: str = "N/A", # ✅ ประกาศที่หัวฟังก์ชันชัดเจน
    pdca_context: str = "",
    **kwargs
) -> Dict[str, Any]:
    """
    [EXPLICIT REVISED v2026.01.27] - การประเมินระดับพื้นฐาน (L1-L2)
    """
    logger = logging.getLogger(__name__)

    # 1. 🛡️ [SHIELDING] ล้างค่าซ้ำใน kwargs
    kwargs.pop("confidence_reason", None)
    e_code = str(kwargs.pop("enabler", "UNK")).upper()
    e_name_th = str(kwargs.pop("enabler_name_th", f"ด้าน {e_code}"))
    
    # ดึง Keywords เฉพาะของ L1-L2
    plan_keywords = kwargs.pop("plan_keywords", "แผนงาน, นโยบาย, คำสั่ง, การดำเนินงาน")
    
    pdca_ctx = str(pdca_context or "ไม่มีข้อมูลแยกหมวดหมู่")
    phases_str = ", ".join(str(p) for p in (required_phases or [])) if required_phases else "P, D"

    try:
        # 2. 🎯 [FORMATTING]
        full_prompt = USER_LOW_LEVEL_PROMPT.format(
            sub_id=sub_id,
            sub_criteria_name=sub_criteria_name,
            level=level,
            statement_text=statement_text,
            context=str(context)[:25000],
            pdca_context=pdca_ctx[:8000],
            required_phases=phases_str,
            specific_contextual_rule=specific_contextual_rule,
            ai_confidence=ai_confidence,
            confidence_reason=confidence_reason, # ✅ ใช้จาก Argument โดยตรง
            plan_keywords=plan_keywords,
            enabler=e_code,
            enabler_name_th=e_name_th,
            **kwargs 
        )

        system_msg = f"Foundation Auditor for {e_name_th} ({e_code})"
        
        # 3. [EXECUTION]
        raw_response = _fetch_llm_response(
            system_prompt=system_msg,
            user_prompt=full_prompt,
            llm_executor=llm_executor
        )

        parsed = _robust_extract_json(raw_response)
        return _build_audit_result_object(
            parsed, raw_response, context, ai_confidence, 
            level=level, sub_id=sub_id, enabler_full_name=e_name_th, enabler_code=e_code
        )

    except Exception as e:
        logger.error(f"🛑 Low-Level Eval Error Sub:{sub_id} L{level}: {str(e)}")
        return _create_fallback_error(sub_id, level, e, context, e_name_th, e_code)
    
def _build_audit_result_object(
    parsed: Dict, 
    raw_response: str, 
    context: str, 
    confidence: str, 
    **kwargs
) -> Dict[str, Any]:
    """
    [ULTIMATE-SYNC v2026.01.27] — THE COMPLETE AUDITOR OBJECT
    - 👔 Integrated 'executive_summary' as primary narrative output.
    - 📎 Enhanced 'evidence_sources' mapping for UI linking.
    - 🛡️ PDCA Coercion & Safety Fallback for scoring.
    """
    from datetime import datetime
    
    # 1. [EXTRACT METADATA]
    level = int(kwargs.get('level', 1))
    sub_id = str(kwargs.get('sub_id', 'Unknown'))
    enabler_full_name = kwargs.get('enabler_full_name', 'Unknown Enabler')
    enabler_code = kwargs.get('enabler_code', 'UNK')

    def clean_score(val, default=0.0):
        if val is None: return default
        try:
            return round(float(val), 2)
        except (ValueError, TypeError):
            return default

    # ประกันความเสี่ยงกรณี parsed ไม่ใช่ dict
    if not isinstance(parsed, dict):
        parsed = {}

    # 2. [SCORING & STATUS] 📊
    # ดึงคะแนนหลัก และคำนวณ is_passed หาก AI ไม่ได้ส่งมา
    score = clean_score(parsed.get("score"))
    is_passed = parsed.get("is_passed")
    if is_passed is None:
        # Fallback เกณฑ์ผ่านมาตรฐาน: L1-L2 (0.7), L3-L5 (1.0)
        is_passed = score >= 0.7 if level <= 2 else score >= 1.0
    else:
        is_passed = bool(is_passed)

    # 3. [EVIDENCE SOURCES & SOURCES] 📎
    # ระบบใหม่ใช้ 'evidence_sources' สำหรับ Object เต็ม และ 'sources' สำหรับรายชื่อ Doc ID
    # พยายามดึงจากทุก Key ที่เป็นไปได้เพื่อให้ครอบคลุมทุก Prompt Version
    evidence_sources = (
        parsed.get("evidence_sources") or 
        parsed.get("top_chunks_data") or 
        []
    )
    
    sources = (
        parsed.get("sources") or 
        parsed.get("evidence") or 
        parsed.get("doc_ids") or 
        parsed.get("reference_documents") or []
    )
    # Normalize 'sources' ให้เป็น List ของ String เสมอ
    if isinstance(sources, str):
        sources = [s.strip() for s in sources.split(',') if s.strip()]
    elif not isinstance(sources, list):
        sources = []

    # 4. [PDCA BREAKDOWN NORMALIZATION] 🧩
    # ดึงคะแนนราย Phase (รองรับทั้งชื่อภาษาไทยและอังกฤษ)
    p_val = parsed.get("P_Plan_Score") or parsed.get("P_Score") or parsed.get("plan_score") or parsed.get("P", 0)
    d_val = parsed.get("D_Do_Score") or parsed.get("D_Score") or parsed.get("do_score") or parsed.get("D", 0)
    c_val = parsed.get("C_Check_Score") or parsed.get("C_Score") or parsed.get("check_score") or parsed.get("C", 0)
    a_val = parsed.get("A_Act_Score") or parsed.get("A_Score") or parsed.get("act_score") or parsed.get("A", 0)

    p_score = clean_score(p_val)
    d_score = clean_score(d_val)
    c_score = clean_score(c_val)
    a_score = clean_score(a_val)

    # กฎเหล็ก: ถ้าผ่าน L1-L2 แต่ AI ลืมใส่คะแนน P ให้ดึงจากคะแนนหลักมาใส่
    if is_passed and level <= 2 and p_score == 0:
        p_score = score

    # 5. [TEXTUAL CONTENT] 📝
    # ดึงการวิเคราะห์เนื้อหาแยกตาม Phase
    ext_p = str(parsed.get("Extraction_P") or parsed.get("หลักฐาน P") or "-").strip()
    ext_d = str(parsed.get("Extraction_D") or parsed.get("หลักฐาน D") or "-").strip()
    ext_c = str(parsed.get("Extraction_C") or parsed.get("หลักฐาน C") or "-").strip()
    ext_a = str(parsed.get("Extraction_A") or parsed.get("หลักฐาน A") or "-").strip()

    # 6. [EXECUTIVE & COACHING NARRATIVE] 👔
    # นี่คือส่วนที่สำคัญที่สุดสำหรับรายงานระดับผู้บริหาร
    executive_summary = str(
        parsed.get("executive_summary") or 
        parsed.get("summary_thai") or 
        parsed.get("บทสรุป") or ""
    ).strip()
    
    reason = str(parsed.get("reason") or parsed.get("เหตุผล") or "ไม่พบเหตุผลสรุปจาก AI").strip()
    coaching_insight = str(parsed.get("coaching_insight") or parsed.get("ข้อแนะนำ") or "").strip()

    # 7. [FINAL ASSEMBLY] 🏛️
    return {
        "sub_id": sub_id,
        "level": level,
        "score": score,
        "is_passed": is_passed,
        "reason": reason,
        "executive_summary": executive_summary,
        "coaching_insight": coaching_insight,
        
        # สำหรับ Flat Report
        "P_Plan_Score": p_score,
        "D_Do_Score": d_score,
        "C_Check_Score": c_score,
        "A_Act_Score": a_score,

        # สำหรับ UI/Dashboard Radar Chart
        "pdca_breakdown": {
            "P": p_score,
            "D": d_score,
            "C": c_score,
            "A": a_score
        },

        # รายละเอียดการสกัดข้อมูล
        "Extraction_P": ext_p,
        "Extraction_D": ext_d,
        "Extraction_C": ext_c,
        "Extraction_A": ext_a,
        
        # ข้อมูลหลักฐาน (สำคัญมากสำหรับการทำ Merge Mapping)
        "evidence_sources": evidence_sources, 
        "sources": sources, 
        
        # Metadata สำหรับ Audit Trail
        "ai_confidence_at_eval": str(confidence or "MEDIUM"),
        "enabler_at_eval": f"{enabler_full_name} ({enabler_code})",
        "generated_at": datetime.now().isoformat(),
        "is_safety_pass": parsed.get("is_safety_pass", True) # สำหรับ Judicial Review
    }

def _create_fallback_error(sub_id: str, level: int, error: Exception, context: str, 
                          enabler_full_name: str = "Unknown", enabler_code: str = "UNK") -> Dict[str, Any]:
    """[SAFETY NET] จัดการกรณี LLM หรือ Network พังแบบ 100%"""
    logger = logging.getLogger(__name__)
    logger.error(f"🛑 Critical Audit Failure Enabler:{enabler_code} Sub:{sub_id} L{level}: {str(error)}")
    
    return {
        "sub_id": str(sub_id),
        "level": int(level),
        "score": 0.0,
        "is_passed": False,
        "reason": f"System Error: {str(error)}",
        "executive_summary": "ไม่สามารถประเมินได้เนื่องจากข้อผิดพลาดทางระบบ", # เพิ่มเพื่อให้ Word มีข้อมูล
        "coaching_insight": "โปรดตรวจสอบการเชื่อมต่อ LLM หรือตรวจสอบหลักฐานอีกครั้ง", # เพิ่มเพื่อให้ Word มีข้อมูล
        "consistency_check": False,
        "P_Plan_Score": 0.0, "D_Do_Score": 0.0, "C_Check_Score": 0.0, "A_Act_Score": 0.0,
        "Extraction_P": "ERR", "Extraction_D": "ERR", "Extraction_C": "ERR", "Extraction_A": "ERR",
        "final_llm_context": str(context or ""),
        "raw_llm_response": "SYSTEM_CRASH",
        "ai_confidence_at_eval": "ERROR",
        "enabler_at_eval": f"{enabler_full_name} ({enabler_code})"
    }

def _heuristic_fallback_parse(raw_text: str) -> Dict:
    """
    [ENHANCED v2026.1.23] Heuristic Fallback 
    - กู้คืนคะแนนจากข้อความดิบเมื่อ JSON พัง
    - รองรับภาษาไทยใน Regex
    """
    parsed = {
        "score": 0.0,
        "is_passed": False,
        "reason": "JSON Parse Failed (Heuristic Applied)",
        "executive_summary": "สกัดผลลัพธ์จากข้อความดิบ",
        "coaching_insight": "ตรวจสอบเนื้อหาใน Raw Response",
        "P_Plan_Score": 0.0, "D_Do_Score": 0.0, "C_Check_Score": 0.0, "A_Act_Score": 0.0,
        "consistency_check": False
    }

    import re
    # 🎯 Regex ที่แข็งแกร่งขึ้นสำหรับหา Score รวม
    # ดักได้ทั้ง: "Score: 1.5", "คะแนนรวม: 2", "Total = 0.5"
    score_match = re.search(r"(?:score|คะแนน|total|ผลรวม)\D*([\d\.]+)", raw_text, re.I)
    if score_match:
        try:
            val = float(score_match.group(1))
            parsed["score"] = min(val, 10.0) # กันคะแนนเกิน
            parsed["is_passed"] = parsed["score"] >= 0.7
        except: pass

    # 🎯 Regex สำหรับ PDCA แยกรายตัว
    patterns = {
        "P_Plan_Score": r"[Pp](?:lan|score)?\D*([\d\.]+)",
        "D_Do_Score": r"[Dd](?:o|score)?\D*([\d\.]+)",
        "C_Check_Score": r"[Cc](?:heck|score)?\D*([\d\.]+)",
        "A_Act_Score": r"[Aa](?:ct|score)?\D*([\d\.]+)"
    }

    for key, pat in patterns.items():
        m = re.search(pat, raw_text)
        if m:
            try: parsed[key] = float(m.group(1))
            except: pass

    # พยายามดึง "เหตุผล" จากบรรทัดแรกๆ
    lines = [l.strip() for l in raw_text.split('\n') if len(l.strip()) > 10]
    if lines:
        parsed["executive_summary"] = lines[0][:200]
        
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
