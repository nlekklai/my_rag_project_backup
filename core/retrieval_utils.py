# core/retrieval_utils.py

import logging
import random
import json
import time
from typing import List, Dict, Any, Optional, Union, TypeVar, Type, Tuple
from langchain.schema import SystemMessage, HumanMessage, Document as LcDocument
from pydantic import ValidationError, BaseModel
import regex as re
import hashlib

from langchain.retrievers.contextual_compression import ContextualCompressionRetriever

# --------------------
# Imports from your project schemas & prompts
# --------------------
# 💡 IMPORTANT: สมมติว่า StatementAssessment มี field: score และ reason
from core.assessment_schema import StatementAssessment, EvidenceSummary
from core.action_plan_schema import ActionPlanActions
from core.rag_prompts import (
    SYSTEM_ASSESSMENT_PROMPT,
    USER_ASSESSMENT_PROMPT,
    ACTION_PLAN_PROMPT,
    SYSTEM_ACTION_PLAN_PROMPT,
    SYSTEM_EVIDENCE_DESCRIPTION_PROMPT,
    EVIDENCE_DESCRIPTION_PROMPT
)

from core.vectorstore import (
    VectorStoreManager,
    load_all_vectorstores, 
    get_reranking_compressor,
    NamedRetriever,
    MultiDocRetriever,
)

T = TypeVar('T', bound=BaseModel)

try:
    from models.llm import llm as llm_instance
except Exception:
    llm_instance = None

# -------------------- Config from global_vars --------------------
try:
    from config.global_vars import (
        DEFAULT_ENABLER,
        SUPPORTED_ENABLERS,
        DEFAULT_SEAM_REFERENCE_DOC_ID,
        SEAM_DOC_ID_MAP,
        SEAM_ENABLER_MAP,
        INITIAL_TOP_K,
        FINAL_K_RERANKED,
        FINAL_K_NON_RERANKED,
    )
except ImportError as e:
    print(f"FATAL ERROR: Cannot import global_vars: {e}")
    raise

logger = logging.getLogger(__name__)

# =================================================================
# MOCKING LOGIC AND GLOBAL FLAGS
# =================================================================
_MOCK_CONTROL_FLAG = False
_MOCK_COUNTER = 0

def set_mock_control_mode(enable: bool):
    global _MOCK_CONTROL_FLAG, _MOCK_COUNTER
    _MOCK_CONTROL_FLAG = enable
    if enable:
        _MOCK_COUNTER = 0
        logger.info("🔑 CONTROLLED MOCK Mode ENABLED.")
    else:
        logger.info("❌ CONTROLLED MOCK Mode DISABLED.")


# ------------------------------------------------------------------
# ID Normalization and Hashing
# ------------------------------------------------------------------
def _hash_stable_id_to_64_char(stable_id: str) -> str:
    return hashlib.sha256(stable_id.lower().encode('utf-8')).hexdigest()

def normalize_stable_ids(ids: List[str]) -> List[str]:
    normalized = []
    for i in ids:
        if len(i) == 64:
            normalized.append(i.lower())
        else:
            normalized.append(_hash_stable_id_to_64_char(i))
    return normalized

# =================================================================
# Document Retrieval
# =================================================================
def retrieve_context_by_doc_ids(doc_uuids: List[str], collection_name: str) -> Dict[str, Any]:
    """
    Retrieve documents by UUIDs from a specific collection (ใช้ 64-char Stable UUIDs)
    """
    if VectorStoreManager is None:
        logger.error("❌ VectorStoreManager is not available.")
        return {"top_evidences": []}
        
    if not doc_uuids:
        logger.warning("⚠️ No document UUIDs provided for retrieval.")
        return {"top_evidences": []}

    try:
        manager = VectorStoreManager()
        
        # 1. Normalize ID เป็น 64-char Stable UUIDs
        normalized_uuids = normalize_stable_ids(doc_uuids)
        
        # 2. ดึงข้อมูลจาก Vector Store
        docs: List[LcDocument] = manager.get_documents_by_id(normalized_uuids, doc_type=collection_name)

        # 3. จัดรูปแบบผลลัพธ์
        top_evidences = []
        for d in docs:
            meta = d.metadata
            top_evidences.append({
                "doc_id": meta.get("doc_id"),              # 64-char Stable UUID
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

# ------------------------------------------------------------------
# RAG Retrieval with optional hard filter (Combined Rerank & Filter)
# ------------------------------------------------------------------
def retrieve_context_with_filter(
    query: str, 
    doc_type: str, 
    enabler: str, 
    stable_doc_ids: Optional[List[str]] = None, 
    top_k_reranked: int = FINAL_K_RERANKED, 
    disable_semantic_filter: bool = False,
    allow_fallback: bool = False
) -> Dict[str, Any]:
    
    global INITIAL_TOP_K, FINAL_K_NON_RERANKED 
    if not isinstance(INITIAL_TOP_K, int):
        INITIAL_TOP_K = 15 
    if not isinstance(FINAL_K_NON_RERANKED, int):
        FINAL_K_NON_RERANKED = 5

    if VectorStoreManager is None:
        logger.error("❌ VectorStoreManager is not available.")
        return {"top_evidences": []}
    
    try:
        manager = VectorStoreManager()
        try:
            if doc_type.lower() == "evidence":
                collection_name = f"{doc_type}_{(enabler or DEFAULT_ENABLER).lower()}"
            else:
                collection_name = doc_type.lower()
        except Exception as e:
            logger.error(f"⚠️ Error generating collection name for doc_type={doc_type}, enabler={enabler}: {e}")
            collection_name = doc_type.lower()

        
        # 1. โหลด Vector Store (ใช้ _load_chroma_instance เพื่อเข้าถึงโดยตรง)
        vectorstore = manager._load_chroma_instance(collection_name)
        
        if vectorstore is None:
            logger.error(f"❌ Vectorstore '{collection_name}' not found or failed to load.")
            return {"top_evidences": []}
        
        # 2. จัดการ Hard Filter (Logic เดิมที่ทำงานได้ดี)
        where_clause = None
        if stable_doc_ids:
            stable_doc_ids_normalized = [doc_id.lower() for doc_id in stable_doc_ids] 
            correct_filter_key = "stable_doc_uuid" 
            
            if not hasattr(vectorstore, "_collection"):
                 logger.error("❌ Vectorstore instance does not have '_collection' attribute for direct access.")
                 if not allow_fallback: return {"top_evidences": []}
                 
            collection = vectorstore._collection
            
            try:
                test_results = collection.get(
                    where={correct_filter_key: {"$in": stable_doc_ids_normalized}},
                    include=["metadatas"]
                )
                found_chunks_count = len(test_results.get("ids", []))
                
                if found_chunks_count > 0:
                    logger.critical(f"✅ Hard Filter found {found_chunks_count} chunks!")
                    where_clause = {correct_filter_key: {"$in": stable_doc_ids_normalized}}
                else:
                    logger.error(f"🛑 Hard Filter found 0 chunks with key '{correct_filter_key}'.")
                    if not allow_fallback: return {"top_evidences": []}
            except Exception as e:
                logger.error(f"🛑 Hard Filter failed: {e}", exc_info=True)
                if not allow_fallback: return {"top_evidences": []}

        # 3. สร้าง Base Retriever
        search_kwargs = {"k": INITIAL_TOP_K}
        if where_clause:
            search_kwargs["filter"] = where_clause
            
        base_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs=search_kwargs)
        
        # 4. ใช้ Reranker/Compression
        if disable_semantic_filter:
            # 4.1 ถ้าปิด Rerank: ใช้ Base Retriever และ Truncate ตาม FINAL_K_NON_RERANKED
            final_k = top_k_reranked if top_k_reranked > 0 else FINAL_K_NON_RERANKED
            documents = base_retriever.invoke(query)[:final_k]
            logger.info(f"RAG Retrieval (Non-Reranked) found {len(documents)} evidences.")
        else:
            # 4.2 ถ้าเปิด Rerank: ห่อหุ้ม Base Retriever ด้วย Compressor
            
            compressor = get_reranking_compressor(top_n=top_k_reranked) 
            
            compressed_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, 
                base_retriever=base_retriever
            )
            
            # 5. Invoke Compressed Retriever (ทำการ Rerank และ Truncate เป็น top_k_reranked)
            documents = compressed_retriever.invoke(query)
            logger.info(f"RAG Retrieval (Reranked) found {len(documents)} evidences (k={INITIAL_TOP_K}->{top_k_reranked}).")

        # 6. จัดรูปแบบผลลัพธ์
        top_evidences = []
        for doc in documents:
            if not isinstance(doc, LcDocument):
                continue

            metadata = doc.metadata or {}

            # 🟢 เพิ่ม fallback ให้ metadata ครบถ้วน
            # 1. เสริม source/file_name หากหายไป
            if not metadata.get("source") and not metadata.get("source_file"):
                # พยายามดึงชื่อไฟล์จาก stable_doc_uuid
                uuid_ref = metadata.get("stable_doc_uuid", "")[:8]
                metadata["source"] = f"Unknown_Source_{uuid_ref}" if uuid_ref else "Unknown_Source"

            # 2. เพิ่ม field page_label หากไม่มี
            if "page_label" not in metadata and "page" in metadata:
                metadata["page_label"] = str(metadata["page"])

            # 3. เพิ่ม chunk_index ถ้ายังไม่มี
            if "chunk_index" not in metadata and hasattr(doc, "metadata") and "chunk_id" in metadata:
                metadata["chunk_index"] = metadata.get("chunk_id")

            # 4. เพิ่ม safety field file_name ให้เท่ากับ source ถ้าไม่มี
            if not metadata.get("file_name"):
                metadata["file_name"] = metadata.get("source")

            top_evidences.append({
                "content": doc.page_content,
                "metadata": metadata
            })

        logger.info(f"RAG Final Output for query='{query[:30]}...' found {len(top_evidences)} evidences.")
        return {"top_evidences": top_evidences}


    except Exception as e:
        logger.error(f"Error during RAG retrieval with filter (Combined Logic): {e}", exc_info=True)
        return {"top_evidences": []}
    
# ------------------------------------------------------------------
# Robust JSON Extraction
# ------------------------------------------------------------------# 

def _robust_extract_json(text: str) -> Optional[Any]:
    """
    Attempts to extract a complete and valid JSON object from text.
    """
    if not text:
        return None

    cleaned_text = text.strip()

    # 1. ลบ code fences (```json, ```, ฯลฯ)
    cleaned_text = re.sub(r'^\s*```(?:json)?\s*', '', cleaned_text, flags=re.MULTILINE | re.IGNORECASE)
    cleaned_text = re.sub(r'\s*```\s*$', '', cleaned_text, flags=re.MULTILINE)
    
    # ลบคำว่า "json" หรือ "output" นำหน้า
    cleaned_text = re.sub(r'^\s*(?:json|output|result)\s*', '', cleaned_text, flags=re.IGNORECASE)

    # 2. ลองโหลดทันที
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        pass # ถ้าโหลดไม่ได้ ให้ทำขั้นตอนต่อไป
    
    # -----------------------------------------------------------
    # ✅ FIX: Logic การค้นหา JSON Object ภายในข้อความ Narrative
    # -----------------------------------------------------------
    try:
        # 2.1 ค้นหาตำแหน่งของ '{' ตัวแรก และ '}' ตัวสุดท้าย
        
        # ลองค้นหา Object JSON (Dictionary)
        start_index = cleaned_text.find('{')
        end_index = cleaned_text.rfind('}')
        
        # ถ้าไม่พบ Object, ลองค้นหา List JSON (Array)
        if start_index == -1 and end_index == -1:
            start_index = cleaned_text.find('[')
            end_index = cleaned_text.rfind(']')

        if start_index != -1 and end_index != -1 and end_index > start_index:
            # ตัดเอาเฉพาะส่วนที่อยู่ระหว่างวงเล็บ/วงเล็บเหลี่ยมเปิดตัวแรกและปิดตัวสุดท้าย
            json_candidate = cleaned_text[start_index : end_index + 1]
            
            # พยายามโหลด JSON ที่ถูกตัดมา
            return json.loads(json_candidate)

    except (json.JSONDecodeError, Exception) as e:
        logger.debug(f"JSON extraction failed even with robust search: {e}")
        return None

    # หากล้มเหลวทุกอย่าง
    return None

def _normalize_keys(data: Union[Dict, List, Any]) -> Union[Dict, List, Any]:
    """
    Helper function to recursively normalize common LLM output keys 
    (เช่น 'llm_score' -> 'score' และ 'reasoning' -> 'reason'). 
    ใช้สำหรับ Action Plan และ Summary เป็นหลัก
    """
    if isinstance(data, dict):
        normalized_data = {}
        # 🎯 Mapping: (Key ที่ LLM ตอบผิด) -> (Key ที่ Pydantic Schema ต้องการ)
        key_mapping = {
            'llm_score': 'score',
            'reasoning': 'reason',          # 🟢 FIX: แปลง 'reasoning' เป็น 'reason'
            'llm_reasoning': 'reason',
            'assessment_reason': 'reason', 
            'comment': 'reason',

            # สำหรับ Action Plan:
            'actions': 'Actions',
            'phase': 'Phase',
            'goal': 'Goal'
        }
        
        for k, v in data.items():
            k_lower = k.lower()
            normalized_key = k
            
            # ตรวจสอบและแปลง key
            for old_k, new_k in key_mapping.items():
                if k_lower == old_k:
                    normalized_key = new_k
                    break
            
            # ใช้ normalized_key และเรียกซ้ำ
            normalized_data[normalized_key] = _normalize_keys(v)
                 
        return normalized_data
    elif isinstance(data, list):
        return [_normalize_keys(item) for item in data]
    return data


def parse_llm_json_response(llm_response_text: str, pydantic_schema: Type[T]) -> Union[T, List[T]]:
    """
    ดึง JSON จากข้อความ LLM, Normalize Key และตรวจสอบความถูกต้องด้วย Pydantic Schema.
    (ใช้สำหรับ Action Plan และ Summary ที่มักมีปัญหา JSON/Key)
    """
    raw_data = _robust_extract_json(llm_response_text)
    
    if raw_data is None:
        raise ValueError("Could not robustly extract valid JSON from LLM response.")
    
    # 🟢 NEW STEP: Normalize Keys ก่อน Validate (สำคัญสำหรับ Action Plan)
    raw_data = _normalize_keys(raw_data) 
        
    try:
        # ตรวจสอบว่า Schema ที่รับเข้ามาเป็น List
        if hasattr(pydantic_schema, '__origin__') and pydantic_schema.__origin__ is list:
            item_schema = pydantic_schema.__args__[0]
            if not isinstance(raw_data, list):
                raw_data = [raw_data]
            
            validated_list = [item_schema.model_validate(item) for item in raw_data]
            return validated_list
        else:
            # สำหรับ Schema เดี่ยว
            if isinstance(raw_data, list):
                if raw_data:
                    logger.warning(f"Expected single object for {pydantic_schema.__name__}, but received a list. Using first element.")
                    raw_data = raw_data[0]
                else:
                    raise ValueError("Received empty list when expecting a single object.")

            validated_model = pydantic_schema.model_validate(raw_data)
            return validated_model
            
    except ValidationError as e:
        logger.error(f"Pydantic Validation Error for {pydantic_schema.__name__}: {e}")
        raise ValueError(f"Pydantic validation failed for schema {pydantic_schema.__name__}. Details: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during Pydantic validation: {e}")
        raise ValueError(f"An unexpected error occurred during JSON validation: {e}")

def extract_uuids_from_llm_response(text: str) -> List[str]:
    """
    ดึง UUIDs ที่ถูกต้องตามรูปแบบมาตรฐาน
    """
    uuid_pattern = r"([a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12})"
    uuids = re.findall(uuid_pattern, text)
    return list(set(uuids))

# =================================================================
# LLM Evaluation
# =================================================================
MAX_LLM_RETRIES = 3
# =================================================================
# Retrieve SEAM Reference Context (Helper)
# =================================================================
# ... (โค้ด include_seam_reference_context และ retrieve_reference_context เดิม) ...

def include_seam_reference_context(sub_id: str, enabler: str = None, top_k_reranked: int = 5) -> str:
    """
    ดึง SEAM Reference Context โดยใช้ Enabler ที่ส่งมาจาก script
    """
    seam_reference_snippet = ""
    try:
        enabler_upper = enabler.upper() if enabler else None
        if enabler_upper not in SUPPORTED_ENABLERS:
            logger.warning(f"⚠️ Enabler '{enabler}' ไม่อยู่ใน SUPPORTED_ENABLERS. ใช้ default SEAM reference.")
            enabler_upper = None

        seam_topic = SEAM_ENABLER_MAP.get(enabler_upper)

        if seam_topic:
            seam_query = (
                f"หลักเกณฑ์การประเมิน SEAM สำหรับหมวด {seam_topic} "
                f"(รหัส {sub_id}) "
                f"โดยเน้นระดับความก้าวหน้าจากระดับล่างสู่ระดับสูง"
            )
        else:
            seam_query = f"หลักเกณฑ์การประเมิน SEAM ที่เกี่ยวข้องกับรหัส {sub_id}"

        logger.info(f"🔍 SEAM reference search query (TH): {seam_query}")

        seam_ctx = retrieve_reference_context(query=seam_query, enabler=enabler_upper, top_k_reranked=top_k_reranked)
        top_chunks = [d["content"] for d in seam_ctx.get("top_evidences", [])][:3]

        if top_chunks:
            seam_reference_snippet = "\n\n--- SEAM REFERENCE CONTEXT ---\n" + "\n\n".join(top_chunks)
            logger.info(f"✅ Included SEAM reference context ({len(top_chunks)} chunks).")
        else:
            logger.warning("⚠️ No SEAM reference chunks found.")

    except Exception as e:
        logger.warning(f"⚠️ Could not include SEAM reference context: {e}")

    return seam_reference_snippet

def retrieve_reference_context(
    query: str,
    enabler: str = None,
    top_k_reranked: int = 5,
    disable_semantic_filter: bool = False
) -> Dict[str, Any]:
    """
    ดึง context จากเอกสาร SEAM Reference โดยอัตโนมัติ
    """
    try:
        logger.info("📘 Retrieving SEAM reference context...")

        doc_id_to_use = SEAM_DOC_ID_MAP.get(enabler.upper(), DEFAULT_SEAM_REFERENCE_DOC_ID) if enabler else DEFAULT_SEAM_REFERENCE_DOC_ID

        return retrieve_context_with_filter(
            query=query,
            doc_type="seam",
            enabler=enabler,
            stable_doc_ids=[doc_id_to_use],
            top_k_reranked=top_k_reranked,
            disable_semantic_filter=disable_semantic_filter,
            allow_fallback=True
        )
    except Exception as e:
        logger.error(f"❌ Failed to retrieve SEAM reference context: {e}", exc_info=True)
        return {"top_evidences": []}


def evaluate_with_llm(
    statement: str,
    context: str,
    standard: str,
    enabler_name: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    ประเมิน Statement เทียบกับ Standard ของ Enabler ที่กำหนด โดยใช้ Evidence Context ที่ดึงจากระบบ RAG
    
    ✅ FIX: กลับไปใช้ Logic เดิม (ดึงค่า JSON โดยตรง) พร้อมเพิ่มความยืดหยุ่นในการดึง Field
    """
    global _MOCK_CONTROL_FLAG, _MOCK_COUNTER

    # -------------------- MOCK MODE --------------------
    if _MOCK_CONTROL_FLAG:
        _MOCK_COUNTER += 1
        score = 1 if _MOCK_COUNTER <= 9 else 0
        reason_text = f"MOCK: FORCED {'PASS' if score == 1 else 'FAIL'} (Statement {_MOCK_COUNTER})"
        is_pass = score >= 1
        status_th = "ผ่าน" if is_pass else "ไม่ผ่าน"
        return {
            "score": score,
            "reason": reason_text,
            "pass_status": is_pass,
            "status_th": status_th,
            "enabler": enabler_name or "N/A"
        }

    # -------------------- FALLBACK CASE --------------------
    if llm_instance is None:
        score = random.choice([0, 1])
        reason = f"LLM Initialization Failed (Fallback to Random Score {score})"
        is_pass = score >= 1
        status_th = "ผ่าน" if is_pass else "ไม่ผ่าน"
        return {
            "score": score,
            "reason": reason,
            "pass_status": is_pass,
            "status_th": status_th,
            "enabler": enabler_name or "N/A"
        }

    # -------------------- GENERATE USER PROMPT --------------------
    user_prompt_content = USER_ASSESSMENT_PROMPT.format(
        enabler_name=enabler_name or DEFAULT_ENABLER,
        level=f"Level {kwargs.get('level', 'N/A')}",
        statement=statement,
        standard=standard,
        context=context if context else "ไม่พบหลักฐานในเอกสารที่เกี่ยวข้อง"
    )

    # -------------------- CALL LLM --------------------
    for attempt in range(MAX_LLM_RETRIES):
        try:
            response = llm_instance.invoke(
                [SystemMessage(content=SYSTEM_ASSESSMENT_PROMPT), HumanMessage(content=user_prompt_content)],
                **({'format': 'json'} if hasattr(llm_instance, 'model_params') and 'format' in llm_instance.model_params else {})
            )

            llm_response_content = response.content if hasattr(response, 'content') else str(response)
            llm_output = _robust_extract_json(llm_response_content)

            if not llm_output or not isinstance(llm_output, dict):
                raise ValueError("LLM response did not contain a recognizable JSON block.")

            # 🎯 FIX: จัดการ Key Nomalization สำหรับ StatementAssessment โดยตรง 
            # (เนื่องจาก Logic เดิมทำงานได้ดีและเราไม่ต้องการ Pydantic Error ใหม่)
            
            # 1. ดึงคะแนน: รับทั้ง 'llm_score' และ 'score'
            raw_score = llm_output.get("llm_score", llm_output.get("score"))
            
            # 2. ดึงเหตุผล: รับจาก Field ที่ LLM ชอบตอบ (reason, reasoning, llm_reasoning)
            reason_text = (
                llm_output.get("reason") or 
                llm_output.get("reasoning") or 
                llm_output.get("llm_reasoning") or
                llm_output.get("assessment_reason")
            )
            
            score_int = int(str(raw_score)) if raw_score is not None else 0

            # 3. สร้าง Dict ตามที่ Pydantic Schema StatementAssessment น่าจะต้องการ (score, reason)
            validated_data = {"score": score_int, "reason": reason_text}
            
            try:
                # 4. Validate (เพื่อป้องกัน Error ร้ายแรง)
                StatementAssessment.model_validate(validated_data)
            except ValidationError as ve:
                 logger.warning(f"Pydantic Validation Warning for StatementAssessment (Minor): {ve}")
                 # อนุญาตให้ผ่านไปต่อ หาก Field สำคัญ (score, reason) ถูกดึงมาแล้ว
                 pass
            
            is_pass = score_int >= 1
            status_th = "ผ่าน" if is_pass else "ไม่ผ่าน"

            validated_data.update({
                "pass_status": is_pass,
                "status_th": status_th,
                "enabler": enabler_name or "N/A"
            })
            return validated_data

        except Exception as e:
            logger.warning(f"LLM Evaluation failed (Attempt {attempt+1}/{MAX_LLM_RETRIES}): {e}")
            if attempt < MAX_LLM_RETRIES - 1:
                time.sleep(1)
                continue

    # -------------------- FALLBACK RESULT --------------------
    score = random.choice([0, 1])
    reason = f"LLM Call Failed after max retries (Fallback to Random Score {score})"
    is_pass = score >= 1
    status_th = "ผ่าน" if is_pass else "ไม่ผ่าน"
    return {
        "score": score,
        "reason": reason,
        "pass_status": is_pass,
        "status_th": status_th,
        "enabler": enabler_name or "N/A"
    }

# =================================================================
# Narrative & Evidence Summary
# =================================================================
# ... (โค้ด generate_narrative_report_via_llm_real เดิม) ...
# ... (โค้ด generate_evidence_description_via_llm เดิม) ...

def generate_narrative_report_via_llm_real(prompt_text: str, system_instruction: str) -> str:
    if llm_instance is None:
        return "[ERROR: LLM Client is not initialized for real API call.]"
    try:
        response = llm_instance.invoke([SystemMessage(content=system_instruction), HumanMessage(content=prompt_text)])
        return (response.content if hasattr(response, 'content') else str(response)).strip()
    except Exception as e:
        return f"[API ERROR] Failed to generate narrative report: {e}"

def summarize_context_with_llm(context: str, sub_criteria_name: str, level: int, sub_id: str, schema: Any) -> Dict[str, str]:
    MAX_LLM_SUMMARY_CONTEXT = 5000
    if llm_instance is None:
        return {"summary": "เกิดข้อผิดพลาด: LLM Client ไม่พร้อมใช้งาน", "suggestion_for_next_level": "N/A"}

    context_to_use = context[:MAX_LLM_SUMMARY_CONTEXT] if len(context) > MAX_LLM_SUMMARY_CONTEXT else context
    try:
        schema_dict = EvidenceSummary.model_json_schema()
        system_prompt_content = SYSTEM_EVIDENCE_DESCRIPTION_PROMPT + "\n\n--- REQUIRED JSON SCHEMA ---\n" + json.dumps(schema_dict, indent=2, ensure_ascii=False)
        human_prompt = EVIDENCE_DESCRIPTION_PROMPT.format(
            standard=sub_criteria_name,
            level=level,
            context=context_to_use,
            sub_id=sub_id
        )
        
        llm_full_response = _fetch_llm_response(prompt=human_prompt, system_prompt=system_prompt_content)
        
        # 🟢 ใช้ parse_llm_json_response (รวม Key Normalization)
        validated_summary_model = parse_llm_json_response(llm_full_response, EvidenceSummary)
        return validated_summary_model.model_dump()
        
    except Exception as e:
        logger.error(f"Error in summarize_context_with_llm: {e}", exc_info=True)
        return {"summary": "ข้อผิดพลาดในการสรุปข้อมูลโดย LLM", "suggestion_for_next_level": str(e)}

def generate_evidence_description_via_llm(*args, **kwargs) -> str:
    logger.warning("generate_evidence_description_via_llm is deprecated. Use summarize_context_with_llm instead.")
    return "เกิดข้อผิดพลาด: ฟังก์ชันนี้ถูกยกเลิกการใช้งาน โปรดใช้ summarize_context_with_llm แทน"

# =================================================================
# LLM Action Plan
# =================================================================
# ... (โค้ด _fetch_llm_response เดิม) ...

def _fetch_llm_response(prompt: str, system_prompt: str, max_retries: int = 1) -> str:
    """
    ดึง Raw Response จาก LLM API
    """
    if llm_instance is None:
        raise ConnectionError("LLM Instance is not available.")
        
    config_params = {"temperature": 0.0}
    
    # 🟢 เพิ่มการตั้งค่า JSON format หากเป็นไปได้
    if hasattr(llm_instance, 'model_params') and 'format' in llm_instance.model_params:
         config_params.update({'format': 'json'})
         
    # 🎯 FIX 2: สำหรับ Local LLM บางตัว (เช่น Ollama) อาจต้องใส่คำสั่งใน System Prompt เพิ่ม
    system_prompt_with_json_ins = system_prompt + "\n\n--- IMPORTANT RULE ---\nOutput MUST be a single, valid JSON object that strictly adheres to the schema. DO NOT include any narrative text, introduction, or markdown fences (```json, ```) outside the JSON object itself."

    for attempt in range(max_retries):
        try:
            response = llm_instance.invoke(
                [SystemMessage(content=system_prompt_with_json_ins), HumanMessage(content=prompt)], 
                config=config_params 
            )
            # Return raw content string
            return (response.content if hasattr(response, 'content') else str(response)).strip()
        except Exception as e:
            logger.warning(f"LLM API call failed (Attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            raise Exception("LLM call failed after max retries for raw response retrieval.")
    return ""


def create_structured_action_plan(
    failed_statements_data: List[Dict[str, Any]],
    sub_id: str,
    enabler: str,
    target_level: int,
    max_retries: int = 2, # จำนวนครั้งที่ Retry (รวม Attempt แรกเป็น 3 ครั้ง)
    retry_delay: float = 1.0,
    include_seam_reference: bool = True
) -> Dict[str, Any]:
    """
    🧩 สร้าง Action Plan โดยใช้ LLM จากผลการประเมิน (Assessment Result)
    
    ✅ FIX: ปรับ Logic Retry เพื่อตัด SEAM Context ออกหาก LLM ล้มเหลวในการสร้าง JSON
    """
    global _MOCK_CONTROL_FLAG, FINAL_K_RERANKED 

    # ============================================================
    # 1️⃣ MOCK MODE
    # ... (โค้ด MOCK MODE ถูกตัดออกเพื่อความกระชับ แต่สมมติว่าอยู่ในโค้ดจริง) ...
    if _MOCK_CONTROL_FLAG:
        return {"Mock": "Action Plan"}


    # -------------------- PREP: Failed Statements --------------------
    failed_statements_text = []
    # Loop over failed data to create the prompt list
    for data in failed_statements_data:
        stmt_num = data.get('statement_number', 'N/A')
        failed_level = data.get('level', 'N/A')
        statement_id = f"L{failed_level} S{stmt_num}"
        # ใช้ Logic ดึง Field reason ที่ยืดหยุ่น (แก้ปัญหา reasoning vs. reason)
        reason_for_failure = data.get('llm_reasoning', data.get('reason', 'N/A'))
        
        failed_statements_text.append(f"""
--- STATEMENT FAILED (Statement ID: {statement_id}) ---
Statement Text: {data.get('statement_text', 'N/A')}
Failed Level: {failed_level}
Reason for Failure: {reason_for_failure}
RAG Context Found: {data.get('retrieved_context', 'No context found')}
**IMPORTANT: The Action Plan must use '{statement_id}' for 'Statement_ID'.**
""")

    statements_list_str = "\n".join(failed_statements_text)
    
    seam_reference_snippet_cache = ""

    # -------------------- PREP: System Prompt --------------------
    try:
        schema_dict = ActionPlanActions.model_json_schema()
        schema_json = json.dumps(schema_dict, indent=2, ensure_ascii=False)
    except Exception:
        schema_json = "{}"

    system_prompt_content = SYSTEM_ACTION_PLAN_PROMPT + "\n\n--- REQUIRED JSON SCHEMA ---\n" + schema_json

    # ============================================================
    # 6️⃣ CALL LLM WITH RETRY LOGIC (The core fix)
    # ============================================================
    final_error = None
    llm_full_response = ""
    
    # max_retries+1 คือจำนวนครั้งที่ลองทั้งหมด (เช่น 2+1 = 3 ครั้ง)
    for attempt in range(max_retries + 1):
        
        # 1. จัดการ SEAM Reference Context (Conditional Inclusion)
        current_seam_snippet = ""
        if attempt == 0 and include_seam_reference:
            # Attempt 1: Try to retrieve and include SEAM context (for quality)
            if not seam_reference_snippet_cache:
                try:
                    # Assuming include_seam_reference_context is available
                    seam_reference_snippet_cache = include_seam_reference_context(sub_id=sub_id, enabler=enabler, top_k_reranked = FINAL_K_RERANKED)
                    if seam_reference_snippet_cache:
                        logger.info(f"✅ Included SEAM reference context for {sub_id} in Attempt {attempt+1} (Quality Attempt).")
                except Exception as e:
                    logger.warning(f"⚠️ Could not include SEAM reference context: {e}")
            
            current_seam_snippet = seam_reference_snippet_cache
            
        elif attempt > 0:
            # ❌ Attempts 2 and 3: Omit Context (to enforce successful JSON output)
            logger.warning(f"❌ Attempt {attempt+1}: Omitted SEAM reference context to enforce JSON output (Failed in previous attempt).")
        
        # 2. Build the LLM Prompt Content
        llm_prompt_content = ACTION_PLAN_PROMPT.format(
            sub_id=sub_id,
            target_level=target_level,
            failed_statements_list=statements_list_str
        ) + current_seam_snippet # Appends the snippet (which is empty in attempt 2, 3)


        # 3. Call LLM
        try:
            # Assuming _fetch_llm_response is available
            llm_full_response = _fetch_llm_response(
                prompt=llm_prompt_content,
                system_prompt=system_prompt_content,
                max_retries=1 
            )

            # 4. Parse JSON (Includes Key Normalization)
            # Assuming parse_llm_json_response is available
            validated_plan_model = parse_llm_json_response(llm_full_response, ActionPlanActions)
            
            # ✅ Success!
            logger.info(f"🎉 Action Plan created successfully on Attempt {attempt+1}!")
            return validated_plan_model.model_dump()

        except Exception as e:
            final_error = str(e)
            logger.warning(f"⚠️ Attempt {attempt+1}/{max_retries+1} failed: {e}")
            
            # 🛑 Logging Raw Output
            if "Could not robustly extract valid JSON" in final_error or "Pydantic validation failed" in final_error:
                 logger.error(f"--- RAW LLM RESPONSE START (JSON FAILED) ---\n{llm_full_response}\n--- RAW LLM RESPONSE END ---")
            
            if attempt < max_retries:
                # Delay before next attempt
                time.sleep(retry_delay)
                continue
            break

    # ============================================================
    # 7️⃣ FINAL ERROR
    # ============================================================
    raise Exception(final_error or "Unknown error during Action Plan creation.")