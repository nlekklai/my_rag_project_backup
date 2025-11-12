# tools/mapping_suggestor.py (ฉบับแก้ไข: ใช้ VSM และ Hard Filter Logic ใหม่)

import os
import logging
from typing import List, Dict, Any, Optional, Tuple, Union

# Project imports
from config.global_vars import (
    DEFAULT_ENABLER,
    EVIDENCE_DOC_TYPES,
    INITIAL_TOP_K,
    FINAL_K_RERANKED,
    FINAL_K_NON_RERANKED,
    MAPPING_FILE_PATH,
)

# 💡 NEW: นำเข้า Logic RAG ที่แก้ไขแล้ว
from core.retrieval_utils import retrieve_context_with_filter 

# 💡 NEW: นำเข้า VSM และ MultiDocRetriever (สำหรับกรณีการดึงข้อมูลหลายเอกสาร)
from core.vectorstore import (
    VectorStoreManager,
    NamedRetriever,
    MultiDocRetriever,
    get_vectorstore_manager,
    load_vectorstore,
    get_global_reranker # เผื่อใช้ rerank เองนอก retrieval_utils
)

# 💡 NEW: นำเข้า Helper ในการโหลด Mapping
from core.ingest import load_doc_id_mapping # ใช้ฟังก์ชันที่คุณมีใน core/ingest.py

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# RAG Context Generation (ใช้ Logic ที่ถูกแก้ไขแล้ว)
# ------------------------------------------------------------------

def get_rag_context(
    query: str, 
    doc_type: str, 
    enabler: str, 
    stable_doc_ids: Optional[list] = None, # Hard Filter
    top_k: int = FINAL_K_RERANKED
) -> List[Dict[str, Any]]:
    """
    Wrapper function to get RAG context using the central retrieval logic.
    ใช้ retrieve_context_with_filter จาก core/retrieval_utils.py 
    ซึ่งรองรับ Hard Filter และ Custom Rerank Logic ของเราแล้ว
    """
    logger.info(f"🔍 Starting RAG retrieval for doc_type='{doc_type}', enabler='{enabler}', query='{query[:50]}...'")
    
    # 🎯 เรียกใช้ฟังก์ชันที่เราแก้ไขแล้ว
    result = retrieve_context_with_filter(
        query=query,
        doc_type=doc_type,
        enabler=enabler,
        stable_doc_ids=stable_doc_ids,
        top_k_reranked=top_k,
        disable_semantic_filter=False # ใช้งาน Rerank Logic ที่เรายืนยันแล้ว
    )
    
    return result.get("top_evidences", [])


# ------------------------------------------------------------------
# Assessment Core Logic (สมมติฐาน)
# ------------------------------------------------------------------

def assess_statement_level(
    statement_id: str,
    target_level: int,
    rubric_text: str,
    evidence_query: str,
    stable_doc_ids: Optional[List[str]] = None,
    doc_type: str = EVIDENCE_DOC_TYPES,
    enabler: str = DEFAULT_ENABLER,
    is_multi_doc: bool = False,
    top_k: int = FINAL_K_RERANKED
) -> Dict[str, Any]:
    """
    Performs the core LLM assessment for a single statement/level.
    """
    logger.info(f"--- ASSESSING: {statement_id} (Level {target_level}) ---")
    
    # 1. Prepare RAG Context
    rag_context = get_rag_context(
        query=evidence_query,
        doc_type=doc_type,
        enabler=enabler,
        stable_doc_ids=stable_doc_ids, # ส่ง Hard Filter เข้าไป
        top_k=top_k
    )
    
    if not rag_context:
        logger.warning(f"⚠️ No RAG context found for {statement_id}. Skipping LLM call.")
        return {
            "statement_id": statement_id,
            "level": target_level,
            "assessment_result": "NO_EVIDENCE_FOUND",
            "context_docs": [],
            "error": "No relevant evidence documents were retrieved."
        }

    # 2. Format Context for LLM (ตัวอย่างการรวม context)
    formatted_context = "\n---\n".join([
        f"Document ID: {doc['metadata'].get('doc_id') or 'N/A'}\n"
        f"Source: {doc['metadata'].get('source') or 'Unknown'}\n"
        f"Content:\n{doc['content']}"
        for doc in rag_context
    ])
    
    # 3. LLM Call (Placeholder)
    # 💡 ในการใช้งานจริง, Logic ตรงนี้จะเรียก LLM เพื่อทำการประเมิน
    # ตัวอย่าง: llm_response = call_llm_for_assessment(rubric_text, evidence_query, formatted_context)
    
    llm_assessment_data = {
        "statement_id": statement_id,
        "level": target_level,
        "assessment_result": "PASSED" if rag_context else "FAILED", # สมมติฐาน
        "llm_reasoning": "Context was successfully retrieved and reranked via the new VSM logic.",
    }

    # 4. Final Output Formatting
    return {
        "statement_id": statement_id,
        "level": target_level,
        "assessment_result": llm_assessment_data["assessment_result"],
        "llm_reasoning": llm_assessment_data["llm_reasoning"],
        "context_docs": [
            {
                "doc_id": doc['metadata'].get('doc_id'),
                "source": doc['metadata'].get('source'),
                "chunk_uuid": doc['metadata'].get('chunk_uuid'),
            } for doc in rag_context
        ]
    }

# ------------------------------------------------------------------
# Multi-Document Assessment / Multi-Retriever (ถ้าใช้)
# ------------------------------------------------------------------

def assess_multiple_statements(
    assessment_list: List[Dict[str, Any]],
    doc_mapping_db: Optional[Dict[str, Any]] = None # ใช้เพื่อตรวจสอบ doc_ids
) -> List[Dict[str, Any]]:
    """
    Process a list of statements for assessment.
    """
    if doc_mapping_db is None:
        doc_mapping_db = load_doc_id_mapping(MAPPING_FILE_PATH)
        
    results = []

    for item in assessment_list:
        stable_doc_ids = item.get("stable_doc_ids")
        # 💡 Verification: ตรวจสอบว่า Stable IDs มีอยู่ใน Mapping หรือไม่
        if stable_doc_ids:
            verified_ids = [
                doc_id for doc_id in stable_doc_ids 
                if doc_id in doc_mapping_db and doc_mapping_db[doc_id].get("chunk_uuids")
            ]
            if len(verified_ids) < len(stable_doc_ids):
                logger.warning(f"⚠️ Some Stable IDs not found/mapped for {item['statement_id']}. Using only {len(verified_ids)} IDs.")
            stable_doc_ids = verified_ids
            
        result = assess_statement_level(
            statement_id=item["statement_id"],
            target_level=item["target_level"],
            rubric_text=item["rubric_text"],
            evidence_query=item["evidence_query"],
            stable_doc_ids=stable_doc_ids, # ส่ง Hard Filter ที่ถูกตรวจสอบแล้ว
            doc_type=item.get("doc_type", EVIDENCE_DOC_TYPES),
            enabler=item.get("enabler", DEFAULT_ENABLER),
            top_k=FINAL_K_RERANKED
        )
        results.append(result)

    return results

# ------------------------------------------------------------------
# Main Execution (ตัวอย่าง)
# ------------------------------------------------------------------

def run_mapping_suggestor(assessment_list: List[Dict[str, Any]]):
    """
    Main function to run the assessment process.
    """
    if not assessment_list:
        logger.error("❌ Assessment list is empty. Exiting.")
        return []

    # 💡 Initialize VSM to ensure models/embeddings are loaded once
    try:
        get_vectorstore_manager()
        logger.info("✅ VectorStoreManager initialized successfully.")
    except Exception as e:
        logger.critical(f"❌ FATAL: VSM initialization failed: {e}")
        return []

    # Run the assessment
    assessment_results = assess_multiple_statements(assessment_list)
    
    logger.info(f"Completed assessment for {len(assessment_results)} statements.")
    return assessment_results

# Example usage (Optional, สำหรับการทดสอบ):
if __name__ == '__main__':
    # 💡 NOTE: ในการใช้งานจริง โค้ดนี้จะรับข้อมูลจากภายนอก
    example_assessments = [
        {
            "statement_id": "S1.1",
            "target_level": 3,
            "rubric_text": "Criteria for S1.1 Level 3...",
            "evidence_query": "What are the procedures for risk assessment?",
            "stable_doc_ids": ["DOC-12345", "DOC-99999"], # ตัวอย่าง Hard Filter
            "doc_type": "evidence",
            "enabler": "KM"
        },
        # ... (รายการอื่นๆ)
    ]
    
    # final_results = run_mapping_suggestor(example_assessments)
    # print(json.dumps(final_results, indent=2, ensure_ascii=False))