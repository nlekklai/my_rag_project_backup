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
from core.vectorstore import _get_collection_name, get_hf_embeddings
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
# Retrieval: retrieve_context_by_doc_ids (Level 2 Hydration)
# ------------------------
def retrieve_context_by_doc_ids(
    doc_uuids: List[str], # <--- Input คือ Chunk UUIDs (64-char_index) หรือ Stable Doc UUID (64-char)
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

    # 🟢 NEW FIX: การแปลง ID
    chunk_uuids_for_chroma = []
    
    # ตรวจสอบว่ามี Doc ID Map ไหม
    if not hasattr(manager, 'doc_id_map') or not manager.doc_id_map:
        logger.warning("VSM Doc ID Map is missing or empty! Using input IDs directly (may fail Hydration).")
        # Fallback (เพื่อไม่ให้โค้ดพัง)
        chunk_uuids_for_chroma = doc_uuids
        
    else:
        for input_id in doc_uuids:
            input_id_str = str(input_id).strip()
            # 1. ตรวจสอบว่าเป็น Stable Doc ID (64 ตัว) ที่ต้อง Map หรือไม่
            if len(input_id_str) == 64 and input_id_str in manager.doc_id_map:
                # 🎯 แปลง: ใช้ Stable Doc ID ค้นหา Chunk UUIDs ที่ถูกต้อง
                mapped_info = manager.doc_id_map.get(input_id_str, {})
                full_chunk_list = mapped_info.get('chunk_uuids', [])
                chunk_uuids_for_chroma.extend(full_chunk_list)
            # 2. ถ้าไม่ใช่ 64 ตัว หรือไม่ตรงกับ Stable Doc ID (อาจเป็น Chunk ID ที่ถูกต้องแล้ว) ให้ใช้โดยตรง
            else:
                chunk_uuids_for_chroma.append(input_id_str) 

        # Log เพื่อความชัดเจน
        if len(chunk_uuids_for_chroma) > len(doc_uuids):
            logger.info(f"VSM: Mapped {len(doc_uuids)} Stable IDs to {len(chunk_uuids_for_chroma)} full Chunk UUIDs for Chroma.")

    # ลบซ้ำก่อนส่งเข้า Chroma
    final_uuids_to_retrieve = list(set(chunk_uuids_for_chroma))
    if not final_uuids_to_retrieve:
        logger.warning("VSM: No valid Chunk UUIDs found after mapping and cleaning.")
        return {"top_evidences": []}
    
    # END OF NEW FIX: ใช้ final_uuids_to_retrieve แทน doc_uuids

    try:
        # 🎯 FIX: เปลี่ยนไปใช้ retrieve_by_chunk_uuids เพื่อดึงเฉพาะ Chunk ที่ระบุเท่านั้น (1:1 Hydration)
        collection_name = _get_collection_name(doc_type, enabler or DEFAULT_ENABLER)
        
        # docs จะเป็นรายการ LcDocument ที่มี page_content และ metadata ที่เกี่ยวข้องกับ Chunk ID นั้น ๆ
        # ใช้ ID ที่ถูกแปลงแล้ว
        docs: List[LcDocument] = manager.retrieve_by_chunk_uuids(final_uuids_to_retrieve, collection_name) 

        top_evidences = []
        for d in docs:
            md = getattr(d, "metadata", {}) or {}
            
            # ✅ FIX: ตรวจสอบและใช้ chunk_uuid ซึ่งตอนนี้ควรเป็น 64-char_index ที่ถูกดึงมา
            # ใน retrieve_by_chunk_uuids, chunk_uuid จะถูกใส่กลับเข้าไปใน metadata
            final_chunk_uuid = md.get("chunk_uuid") or md.get("stable_doc_uuid") 
            
            top_evidences.append({
                # doc_id: ID เอกสารหลัก (64 ตัว)
                "doc_id": md.get("stable_doc_uuid"),
                # chunk_uuid: ID ที่ใช้ในการค้นหาและอ้างอิง Chunk (64-char_index)
                "chunk_uuid": final_chunk_uuid, 
                "doc_type": md.get("doc_type"),
                "source": md.get("source") or md.get("doc_source"),
                "source_filename": md.get("source") or md.get("doc_source"),  
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
# ------------------------
# Retrieval: retrieve_context_with_filter (แก้จุดเสี่ยง 2 จุด)
# ------------------------
def retrieve_context_with_filter(
    query: Union[str, List[str]],
    doc_type: str,
    enabler: Optional[str] = None,
    subject: Optional[str] = None, # 🟢 เพิ่ม subject
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
                
                # 🎯 FIX C: นำ chunk_uuid และ doc_id (stable_doc_uuid) เข้ามาใน metadata
                if 'chunk_uuid' in doc:
                    meta['chunk_uuid'] = doc['chunk_uuid']
                if 'doc_id' in doc:
                    meta['stable_doc_uuid'] = doc['doc_id']
                if 'pdca_tag' in doc:
                     meta['pdca_tag'] = doc['pdca_tag'] # ต้องเก็บ PDCA tag ด้วย

                if pc.strip():
                    guaranteed_priority_chunks.append(LcDocument(page_content=pc, metadata=meta))
            elif hasattr(doc, 'page_content'):
                guaranteed_priority_chunks.append(doc)

    # 4. ดึง collection name ให้ตรงตัว
    collection_name = _get_collection_name(doc_type, enabler or DEFAULT_ENABLER)
    logger.info(f"Requesting retriever → collection='{collection_name}' (doc_type={doc_type}, enabler={enabler})")

    # 🟢 Logic สร้าง Filter WHERE จาก stable_doc_ids และ subject
    where_filter: Dict[str, Any] = {}
    doc_id_filter: Dict[str, Any] = {}
    
    # 4.1 Filter: Stable Doc IDs (Hard Filter)
    if stable_doc_ids:
        logger.info(f"Applying Stable Doc ID filter: {len(stable_doc_ids)} IDs")
        doc_id_filter = {"stable_doc_uuid": {"$in": stable_doc_ids}} 
        where_filter = doc_id_filter # เริ่มต้นด้วย Doc ID Filter

    # 4.2 Filter: Subject (Soft Filter)
    if subject:
        subject_filter = {"subject": {"$eq": subject}}
        
        if where_filter:
            # ใช้ $and เพื่อรวมเงื่อนไข: (ID ต้องตรง AND Subject ต้องตรง)
            where_filter = {"$and": [where_filter, subject_filter]}
            logger.info(f"Adding Subject filter (AND logic): {subject}")
        else:
            # ถ้าไม่มี stable_doc_ids ให้ใช้ subject เป็น filter หลัก
            where_filter = subject_filter
            logger.warning("Applying Subject filter only (no Stable Doc IDs).")


    retriever = manager.get_retriever(collection_name) 
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
                # 🎯 FIX: รวม Filter เข้าใน search_kwargs
                search_kwargs = {"k": INITIAL_TOP_K}
                if where_filter:
                    search_kwargs["where"] = where_filter

                if hasattr(retriever, "get_relevant_documents"):
                    docs = retriever.get_relevant_documents(q, search_kwargs=search_kwargs) 
                elif hasattr(retriever, "invoke"):
                    docs = retriever.invoke(q, config={"configurable": {"search_kwargs": search_kwargs}})
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

        # 🎯 FIX: ใช้ chunk_uuid ซึ่งตอนนี้คือ ID 64-char_index สำหรับ Dedup
        # TEMP-ID ยังคงถูกใช้สำหรับ dedup ชั่วคราว แต่จะถูกกรองออกในขั้นตอนที่ 7
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

    # **NEW:** 6.0. สร้าง Map เพื่อ Patch Metadata ที่หายไปกลับคืนมา (ป้องกัน Reranker ล้าง metadata)
    candidate_metadata_map = {
        doc.page_content: getattr(doc, 'metadata', {}) 
        for doc in candidates if hasattr(doc, 'page_content') and doc.page_content.strip()
    }

    if slots_left > 0 and candidates:
        reranker = get_global_reranker()
        if reranker and hasattr(reranker, "compress_documents"):
            try:
                # 6.1. เรียก Reranker (จะคืนค่าเป็น DocumentWithScore object)
                reranked_results = reranker.compress_documents(
                    documents=candidates,
                    query=queries_to_run[0],
                    top_n=slots_left
                )
                
                reranked_docs_with_metadata = []
                for result in reranked_results:
                    # 🎯 FIX A: แตก Wrapper Object (ใช้ getattr เพื่อให้โค้ดสั้นลงและยืดหยุ่น)
                    doc_to_add = getattr(result, 'document', result)
                        
                    # 2. ตรวจสอบความถูกต้อง
                    if doc_to_add and hasattr(doc_to_add, 'page_content') and doc_to_add.page_content.strip():
                        
                        # 3. **CRITICAL FIX**: Patch Metadata ถ้าพบว่า ID หายไป
                        current_metadata = getattr(doc_to_add, 'metadata', {})
                        chunk_uuid_check = current_metadata.get("chunk_uuid") or current_metadata.get("dedup_chunk_uuid")

                        # ถ้า ID หายไป และ Content สามารถ Map กลับไปหา Original ได้
                        if not chunk_uuid_check and doc_to_add.page_content in candidate_metadata_map:
                            original_metadata = candidate_metadata_map[doc_to_add.page_content]
                            
                            # Patch metadata กลับเข้าไปใน document object
                            if hasattr(doc_to_add, 'metadata'):
                                doc_to_add.metadata = original_metadata
                                logger.debug("Patched metadata back to reranked document.")
                            # กรณีที่เป็น object ชนิดอื่นที่แก้ไข metadata ไม่ได้ ให้ข้ามไป (กรณีนี้ไม่ควรเกิด)
                        
                        # 4. เพิ่มเข้าลิสต์
                        reranked_docs_with_metadata.append(doc_to_add)
                
                # 6.2. ใช้ Documents ที่ถูกแตกและ Patch Metadata แล้ว
                final_docs.extend(reranked_docs_with_metadata or candidates[:slots_left])
                logger.info(f"Reranker returned {len(reranked_docs_with_metadata)} docs (after extraction and patching)")
                
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
    used_chunk_uuids: List[str] = [] # ต้องประกาศใหม่ที่นี่เพื่อรับเฉพาะ ID ที่ถูกเลือก

    # 🟢 NEW FIX: กรอง Chunk ที่ไม่มี ID ที่ถูกต้องออกไป
    valid_final_docs = []
    for doc in final_docs[:FINAL_K_RERANKED]:
        md = getattr(doc, "metadata", {}) or {}
        chunk_uuid_candidate = md.get("chunk_uuid") or md.get("dedup_chunk_uuid")
        
        # เงื่อนไขการยอมรับ: ต้องมี ID, ต้องมีความยาวอย่างน้อย 32 ตัวอักษร (เพื่อกรอง TEMP-), และต้องไม่เป็น ID ชั่วคราว (UNKNOWN/TEMP)
        is_valid_hash = bool(chunk_uuid_candidate and len(chunk_uuid_candidate) >= 32 and not re.match(r"^(TEMP|UNKNOWN)-", str(chunk_uuid_candidate)))
        
        if is_valid_hash:
            valid_final_docs.append(doc)
        else:
            logger.warning(
                f"Skipping chunk in final output due to invalid/temporary ID: {chunk_uuid_candidate}. "
                f"Source Doc ID: {md.get('stable_doc_uuid')}"
            )

    # 7.1 วนลูปเฉพาะ Chunk ที่มี ID ที่ถูกต้องเพื่อสร้าง Final Output
    for doc in valid_final_docs:
        md = getattr(doc, "metadata", {}) or {}
        pc = str(getattr(doc, "page_content", "") or "").strip()
        
        # 🎯 FIX B: ใช้ ID ที่ถูกกรองแล้ว (chunk_uuid_final)
        chunk_uuid_final = md.get("chunk_uuid") or md.get("dedup_chunk_uuid")
        
        used_chunk_uuids.append(str(chunk_uuid_final)) # บันทึกเฉพาะ ID ที่ถูกต้อง

        source = md.get("source") or md.get("filename") or md.get("doc_source") or "Unknown"
        pdca = md.get("pdca_tag", "Other")

        top_evidences.append({
            "doc_id": md.get("stable_doc_uuid"),
            "chunk_uuid": chunk_uuid_final, # ID ที่ Level 2 จะใช้ค้นหา (ตอนนี้มั่นใจว่าเป็น 64-char Hash)
            "source": source,
            "source_filename": source,
            "text": pc,
            "pdca_tag": pdca,
            # 🔑 CRITICAL FIX: เพิ่มการคัดลอกคะแนน Rerank เข้ามาในผลลัพธ์
            "rerank_score": md.get("relevance_score", 0.0), 
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

# ------------------------------------------------------------------
# Helper Function: Create ChromaDB Where Filter
# ------------------------------------------------------------------
def _create_where_filter(doc_ids: Optional[Set[str]]) -> Dict[str, Any]:
    """
    Creates a ChromaDB 'where' filter dictionary to filter by stable document IDs.
    Assumes the stable document ID is stored in the metadata key 'stable_doc_uuid'.
    """
    if not doc_ids:
        # 🟢 FIX: คืนค่าเป็น Dict ว่าง เมื่อไม่มี ID เพื่อป้องกัน Chroma Error
        return {}
    
    return {
        "stable_doc_uuid": {
            "$in": list(doc_ids)
        }
    }


# ------------------------
# Retrieval: retrieve_context_for_endpoint (Final, Robust Version)
# ------------------------
def retrieve_context_for_endpoint(
    vectorstore_manager: VectorStoreManager, 
    collection_name: Optional[str] = None, 
    query: str = "", 
    stable_doc_ids: Optional[Set[str]] = None, 
    doc_type: Optional[str] = None, 
    enabler: Optional[str] = None, 
    subject: Optional[str] = None, # 🟢 เพิ่ม subject เข้ามาใน Signature
    **kwargs: Any, # รับ k_to_retrieve และ k_to_rerank ที่ Router อาจส่งมา
) -> Dict[str, Any]: 
    """
    Directly query a Chroma collection using stable doc IDs (Hard Filter)
    This is used for endpoints that require specific, already selected documents.
    """
    start_time = time.time() 
    
    # ------------------------------------------------------------------
    # 1. จัดการ Collection Name (Fallback Logic)
    # ------------------------------------------------------------------
    if not collection_name and doc_type:
        try:
            # 💡 Derive collection name จาก doc_type และ enabler
            collection_name = _get_collection_name(doc_type, enabler or DEFAULT_ENABLER)
            logger.info(f"Derived collection_name: '{collection_name}' from doc_type='{doc_type}', enabler='{enabler}'")
        except Exception as e:
            logger.error(f"Cannot derive collection_name from doc_type/enabler: {e}")
            collection_name = None 

    if not collection_name:
        logger.error("FATAL: Cannot determine collection_name. Exiting retrieval.")
        return {"top_evidences": [], "aggregated_context": "", "retrieval_time": 0.0, "used_chunk_uuids": []}
    
    logger.critical(f"[QUERY] Running Endpoint Query: '{query[:50]}...' → collection='{collection_name}' (Type: {doc_type or '?'})")

    # 2. โหลด Chroma Instance
    chroma_instance = vectorstore_manager._load_chroma_instance(collection_name)
    if chroma_instance is None:
        logger.error(f"Cannot load Chroma instance for collection: {collection_name}")
        return {"top_evidences": [], "aggregated_context": "", "retrieval_time": 0.0, "used_chunk_uuids": []}
    
    # 3. เตรียม Where Filter (รวม Stable Doc IDs และ Subject) 🟢 จุดแก้ไข
    # ใช้ฟังก์ชันช่วยสร้าง filter สำหรับ Doc IDs (ถ้ามี)
    where_filter = _create_where_filter(stable_doc_ids)

    # 3.2 Filter: Subject (Secondary Safety Filter) 
    # 🟢 FIX: Clean Subject String ที่รับมาทันที และใช้ Exact Match (Final Version)
    cleaned_subject = subject.strip() if subject else None

    if cleaned_subject:
        # 🎯 ใช้ Exact Match: {"subject": value}
        subject_filter = {"subject": cleaned_subject}
        
        if where_filter:
            # ใช้ $and เพื่อรวมเงื่อนไข: ID ต้องตรง AND Subject ต้องตรง
            where_filter = {"$and": [where_filter, subject_filter]}
            logger.info(f"Applying combined filter: {len(stable_doc_ids or [])} IDs AND Subject='{cleaned_subject}'")
        else:
            # กรณีที่ไม่ได้เลือก Doc ID มา
            where_filter = subject_filter
            logger.warning(f"Applying Subject filter only: '{cleaned_subject}'")
    # ------------------------------------------------------------------
    
    # 4. Embed Query (แก้ Dimension Mismatch)
    try:
        embedding_func = get_hf_embeddings()
        query_text_with_prefix = "query: " + query
        query_embeddings = embedding_func.embed_query(query_text_with_prefix)
        logger.info("✅ Successfully embedded query with 768 dimension.")
    except Exception as e:
        logger.error(f"FATAL: Failed to embed query with 768-dim model: {e}")
        return {"top_evidences": [], "aggregated_context": "", "retrieval_time": 0.0, "used_chunk_uuids": []}

    # ------------------------------------------------------------------
    # 5. Query Chroma DB โดยตรง (ใช้ Hard Filter หรือ Query ทั้งหมด)
    # ------------------------------------------------------------------
    results = {'ids': [[]], 'documents': [[]], 'metadatas': [[]], 'distances': [[]]} # Placeholder
    
    # 💡 ใช้ INITIAL_TOP_K หรือค่า k_to_retrieve ที่ส่งมาจาก Router (ถ้ามี)
    n_results = kwargs.get("k_to_retrieve", INITIAL_TOP_K)
    
    try:
        query_params = {
            "query_embeddings": [query_embeddings], 
            "n_results": n_results,
            "include": ['documents', 'metadatas', 'distances']
        }
        
        # 🎯 FIX: ส่ง 'where' ไปก็ต่อเมื่อมี Filter เท่านั้น (แก้ Chroma Error)
        if where_filter: 
            query_params["where"] = where_filter
            filter_summary = f"Doc IDs:{len(stable_doc_ids or [])}"
            if subject:
                 filter_summary += f", Subject:'{subject}'"
            logger.info(f"Running Chroma query with Filter ({filter_summary}) and n_results={n_results}") # 🟢 ปรับปรุง Log
        else:
            logger.warning("No stable_doc_ids or subject provided. Querying entire collection (may be slow/incorrect usage).")

        results = chroma_instance._collection.query(**query_params)
        
    except Exception as e:
        logger.error(f"Chroma direct query failed (Endpoint): {e}", exc_info=False)
        
    # ------------------------------------------------------------------
    # 6. Post-process: Convert Chroma results to LcDocument
    # ------------------------------------------------------------------
    raw_chunks: List[LcDocument] = []
    if results and results.get('documents') and results['documents'][0]:
        for doc_content, metadata, distance in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            if not metadata:
                metadata = {}
            
            metadata['retrieval_distance'] = float(distance)
            metadata['collection_name'] = collection_name
            
            raw_chunks.append(LcDocument(page_content=doc_content, metadata=metadata))
    
    logger.critical(f"[RETRIEVAL] Raw chunks from ChromaDB (Direct): {len(raw_chunks)} documents")
    final_chunks = list(raw_chunks) 
    
    # ------------------------------------------------------------------
    # 7. Final Output: Convert LcDocument list to expected DICT format
    # ------------------------------------------------------------------
    top_evidences = []
    aggregated_parts = []
    used_chunk_uuids = []
    
    for doc in final_chunks:
        md = getattr(doc, "metadata", {}) or {}
        pc = str(doc.page_content or "").strip()
        
        chunk_uuid = md.get("chunk_uuid") or md.get("dedup_chunk_uuid")
        source = md.get("source") or md.get("filename") or md.get("doc_source") or "Unknown"
        pdca = md.get("pdca_tag", "Other")

        if chunk_uuid:
            used_chunk_uuids.append(str(chunk_uuid))

        top_evidences.append({
            "doc_id": md.get("stable_doc_uuid"),
            "chunk_uuid": chunk_uuid, 
            "source": source,
            "source_filename": source,
            "text": pc,
            "pdca_tag": pdca,
            "retrieval_distance": md.get("retrieval_distance", 0.0),
        })
        aggregated_parts.append(f"[{pdca}] [SOURCE: {source}] {pc}")

    end_time = time.time()
    result = {
        "top_evidences": top_evidences,
        "aggregated_context": "\n\n---\n\n".join(aggregated_parts),
        "retrieval_time": round(end_time - start_time, 3),
        "used_chunk_uuids": used_chunk_uuids 
    }
    
    source_count = len({c.metadata.get('stable_doc_uuid') for c in final_chunks if c.metadata and c.metadata.get('stable_doc_uuid')}) 
    logger.info(f"Final retrieval (Endpoint): {len(top_evidences)} chunks in {result['retrieval_time']:.2f}s (Sources: {source_count})")
    
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
    statement_id: str, 
    level: int,
    enabler_id: str,
    focus_hint: str,
    llm_executor: Any = None
) -> List[str]:
    """
    Generates a list of tailored queries (Multi-Query strategy) based on the statement 
    และ PDCA focus, โดยกำหนด Query ที่เฉพาะเจาะจงสำหรับแต่ละระดับ
    
    Returns: List[str] of queries.
    """
    
    # Q1: Base Query (P/D Focus) - แม่แบบเริ่มต้น
    base_query_template = (
        f"{statement_text}. {focus_hint} หลักฐานแสดงแผน การดำเนินการ และโครงสร้างของ {statement_id} "
        f"ตามบริบทของ {enabler_id}"
    )
    
    queries = []

    # 1. Level 5 Query Refinement (ปรับ Base Query สำหรับ L5 เท่านั้น)
    if level == 5:
        # สำหรับ L5, ปรับ Base Query Q1 ให้เน้น L5 มากขึ้น
        base_query = base_query_template + ". **การบูรณาการ, ความยั่งยืน, การขยายผล, โครงการนำร่อง, นวัตกรรม**"
        queries.append(base_query)
        
        # Q4 (Innovation/Sustainability Focus) - เพิ่ม Query ที่ 4 เฉพาะ L5
        l5_innovation_query = (
            f"หลักฐานนวัตกรรม ความยั่งยืน การขยายผล หรือโครงการนำร่องที่เกี่ยวข้องกับ {statement_id}. "
            f"การใช้ **Best Practice**, **ผลกระทบระยะยาว**, **การบูรณาการข้ามสายงาน**"
        )
        queries.append(l5_innovation_query)
    
    else:
        # สำหรับ L1-L4, ใช้ Base Query ปกติ
        base_query = base_query_template
        queries.append(base_query)


    # 2. Level 3+ (C/A) Query Refinement (เพิ่ม C และ A สำหรับ L3 ขึ้นไป)
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


def evaluate_with_llm(
    context: str, 
    sub_criteria_name: str, 
    level: int, 
    statement_text: str, 
    sub_id: str, 
    check_evidence: str = "", 
    act_evidence: str = "", 
    llm_executor: Any = None, 
    max_evidence_strength: float = 10.0, # 🟢 NEW: รับค่า Capping โดยตรง
    **kwargs
) -> Dict[str, Any]:
    """Standard Evaluation for L3+ with robust handling for missing keys."""
    
    context_to_send_eval = context[:MAX_EVAL_CONTEXT_LENGTH] if context else ""
    # 1. ตรวจสอบ Context ก่อนส่ง LLM
    failure_result = _check_and_handle_empty_context(context, sub_id, level)
    if failure_result:
        return failure_result

    contextual_rules_prompt = kwargs.get("contextual_rules_prompt", "")
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
        # 🟢 NEW: ส่งค่า Cap ให้ User Prompt (เผื่อใช้ในส่วนของ User)
        max_evi_str_cap_for_llm=max_evidence_strength,
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

    # 🟢 FIX: จัดรูปแบบ SYSTEM_ASSESSMENT_PROMPT ด้วยค่า Cap ก่อนรวมกับ Schema
    # (ต้องมั่นใจว่า SYSTEM_ASSESSMENT_PROMPT มี placeholder {max_evi_str_cap_for_llm} แล้ว)
    system_prompt_formatted = SYSTEM_ASSESSMENT_PROMPT.format(
        max_evi_str_cap_for_llm=max_evidence_strength
    )

    system_prompt = system_prompt_formatted + "\n\n--- JSON SCHEMA ---\n" + schema_json + "\nIMPORTANT: Respond only with valid JSON."

    try:
        # 3. เรียก LLM
        raw = _fetch_llm_response(system_prompt, user_prompt, _MAX_LLM_RETRIES, llm_executor=llm_executor)
        
        # 4. Extract JSON และ normalize keys
        parsed = _robust_extract_json(raw)
        
        # 🎯 FIX 1: ตรวจสอบและบังคับให้ 'parsed' เป็น dict ก่อนใช้งาน
        if not isinstance(parsed, dict):
            logger.error(f"LLM L{level} response parsed to non-dict type: {type(parsed).__name__}. Falling back to empty dict.")
            parsed = {}

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

def evaluate_with_llm_low_level(
    context: str, 
    sub_criteria_name: str, 
    level: int, 
    statement_text: str, 
    sub_id: str, 
    llm_executor: Any, 
    max_evidence_strength: float = 10.0, # 🟢 NEW: รับค่า Capping โดยตรง (แม้จะไม่ใช้ แต่รับเพื่อป้องกัน error)
    **kwargs
) -> Dict[str, Any]:
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

    # 🟢 FIX: จัดรูปแบบ SYSTEM_LOW_LEVEL_PROMPT ด้วยค่า Cap ก่อนรวมกับ Schema
    system_prompt_formatted = SYSTEM_LOW_LEVEL_PROMPT.format(
        max_evi_str_cap_for_llm=max_evidence_strength
    )
    system_prompt = system_prompt_formatted + "\n\n--- JSON SCHEMA ---\n" + schema_json + "\nIMPORTANT: Respond only with valid JSON."

    try:
        raw = _fetch_llm_response(system_prompt, user_prompt, _MAX_LLM_RETRIES, llm_executor=llm_executor)
        parsed = _robust_extract_json(raw)

        # 🎯 FIX 1: ตรวจสอบและบังคับให้ 'parsed' เป็น dict ก่อนส่งไป extraction
        if not isinstance(parsed, dict):
            logger.error(f"LLM L{level} response parsed to non-dict type: {type(parsed).__name__}. Falling back to empty dict.")
            parsed = {}
        
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
        "A_Act_Score": 0,    # 🎯 FIX 2: เปลี่ยนจาก 'A_Act_Sure' เป็น 'A_Act_Score'
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