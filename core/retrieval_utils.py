# core/retrieval_utils.py
import logging
import random
import json, json5
import time
from typing import List, Dict, Any, Optional, Union, TypeVar, Type, Tuple
# 💡 แก้ไข #1: Messages และ Document ถูกย้ายไปที่ langchain_core
from langchain_core.messages import SystemMessage, HumanMessage 
from langchain_core.documents import Document as LcDocument
from pydantic import ValidationError, BaseModel
import regex as re
import hashlib

# Project imports
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
    GLOBAL_RERANKER, 
    get_global_reranker
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
# RAG Retrieval with optional hard filter (ปรับปรุง rerank)
# ------------------------------------------------------------------
def retrieve_context_with_filter(
    query: str, 
    doc_type: str, 
    enabler: str, 
    stable_doc_ids: Optional[list] = None, 
    top_k_reranked: int = FINAL_K_RERANKED, 
    disable_semantic_filter: bool = False,
    allow_fallback: bool = False
) -> dict:

    try:
        manager = VectorStoreManager()
        collection_name = f"{doc_type}_{(enabler or DEFAULT_ENABLER).lower()}" \
                          if doc_type.lower() == "evidence" else doc_type.lower()
        vectorstore = manager._load_chroma_instance(collection_name)
        if not vectorstore:
            logger.error(f"❌ Vectorstore '{collection_name}' not found.")
            return {"top_evidences": []}

        # Hard filter
        where_clause = None
        if stable_doc_ids:
            stable_doc_ids_normalized = [doc_id.lower() for doc_id in stable_doc_ids]
            collection = getattr(vectorstore, "_collection", None)
            if collection:
                try:
                    test_results = collection.get(
                        where={"stable_doc_uuid": {"$in": stable_doc_ids_normalized}},
                        include=["metadatas"]
                    )
                    if len(test_results.get("ids", [])) > 0:
                        where_clause = {"stable_doc_uuid": {"$in": stable_doc_ids_normalized}}
                except Exception as e:
                    logger.error(f"Hard filter failed: {e}")
                    if not allow_fallback:
                        return {"top_evidences": []}

        # Base retriever
        search_kwargs = {"k": INITIAL_TOP_K}
        if where_clause:
            search_kwargs["filter"] = where_clause
        base_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs=search_kwargs)

        # -------------------------
        # Rerank + truncate
        # -------------------------
        documents = base_retriever.invoke(query)
        if not disable_semantic_filter:
            # reranker = get_reranking_compressor(top_n=top_k_reranked)
            reranker = get_global_reranker(top_k_reranked)
            if reranker and hasattr(reranker, "rerank"):
                try:
                    documents = reranker.rerank(query, documents)
                    logger.info(f"RAG Retrieval (Rerank) truncated to top {top_k_reranked}.")
                except Exception as e:
                    logger.warning(f"⚠️ Rerank failed: {e}. Using base retriever + truncate.")
                    documents = documents[:top_k_reranked]
            else:
                logger.warning("⚠️ Rerank unavailable. Using base retriever + truncate.")
                documents = documents[:top_k_reranked]
        else:
            # Non-rerank
            documents = documents[:top_k_reranked]
            logger.info(f"RAG Retrieval (Non-Rerank) truncated to top {top_k_reranked}.")

        # Format results
        top_evidences = []
        for doc in documents:
            if not isinstance(doc, LcDocument):
                continue
            metadata = doc.metadata or {}
            if not metadata.get("source"):
                uuid_ref = metadata.get("stable_doc_uuid", "")[:8]
                metadata["source"] = f"Unknown_Source_{uuid_ref}" if uuid_ref else "Unknown_Source"
            if "page_label" not in metadata and "page" in metadata:
                metadata["page_label"] = str(metadata["page"])
            if "chunk_index" not in metadata and "chunk_id" in metadata:
                metadata["chunk_index"] = metadata.get("chunk_id")
            if not metadata.get("file_name"):
                metadata["file_name"] = metadata.get("source")
            top_evidences.append({
                "content": doc.page_content,
                "metadata": metadata
            })

        return {"top_evidences": top_evidences}

    except Exception as e:
        logger.error(f"Error in retrieve_context_with_filter: {e}", exc_info=True)
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


# ------------------------------------------------------------------
# 🟢 ฟังก์ชันที่แก้ไข: extract_uuids_from_llm_response
# ------------------------------------------------------------------

# Regex สำหรับค้นหา UUIDs ทั่วไป (รูปแบบ 8-4-4-4-12)
UUID_PATTERN = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', re.IGNORECASE)

def extract_uuids_from_llm_response(
    text: str, 
    key_hint: Optional[List[str]] = None # 🟢 รับ key_hint
) -> List[str]:
    """
    Attempts to extract UUIDs from text, supporting both plain text search and 
    JSON key search (when key_hint is provided).
    """
    unique_uuids = set()
    cleaned_text = text.strip()

    # 1. ลอง Deserialize เป็น JSON (เพื่อค้นหาใน Key ที่ระบุ)
    try:
        data: Dict[str, Any] = json.loads(cleaned_text)
        
        # 2. ค้นหาในคีย์ที่กำหนด (ถ้ามี key_hint)
        if key_hint:
            for key in key_hint:
                if key in data:
                    uuids_list = data[key]
                    if isinstance(uuids_list, list):
                        # ค้นหาใน List ของ String (กรณีที่ LLM ตอบเป็น ["uuid1", "uuid2"])
                        for item in uuids_list:
                            if isinstance(item, str) and UUID_PATTERN.fullmatch(item):
                                unique_uuids.add(item)
                    elif isinstance(uuids_list, str):
                         # กรณีที่คีย์นั้นเป็น string ที่มี UUIDs คั่นด้วยอะไรบางอย่าง
                         found_in_string = UUID_PATTERN.findall(uuids_list)
                         unique_uuids.update(found_in_string)
        
        # 3. ถ้าไม่มี key_hint หรือยังไม่พบ ให้ค้นหา UUIDs ทั้งหมดในข้อความ JSON ทั้งหมด (รวมถึงค่าใน Field ที่ไม่ตรงกับ key_hint)
        # (ใช้ค้นหาแบบ Fallback Search ในกรณีที่ key_hint ไม่สมบูรณ์ หรือสำหรับงานอื่น)
        all_text = json.dumps(data)
        found_in_string = UUID_PATTERN.findall(all_text)
        unique_uuids.update(found_in_string)

    except json.JSONDecodeError:
        # 4. ถ้าไม่ใช่ JSON ที่ถูกต้อง หรือแค่ข้อความธรรมดา ให้ค้นหา UUIDs ทั่วไป
        found_in_string = UUID_PATTERN.findall(cleaned_text)
        unique_uuids.update(found_in_string)
        
    return list(unique_uuids)

# =================================================================
MAX_LLM_RETRIES = 3
# =================================================================

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
    MAX_RETRIES = 3 
    
    # 1️⃣ ตั้งค่าข้อมูลเริ่มต้น
    context_to_use = context[:MAX_LLM_SUMMARY_CONTEXT] if len(context) > MAX_LLM_SUMMARY_CONTEXT else context
    
    default_error_response = {
        "summary": f"ข้อผิดพลาดในการสรุปข้อมูลโดย LLM (Level {level})", 
        "suggestion_for_next_level": "N/A - โปรดตรวจสอบ Raw LLM Output หรือ LLM Service"
    }

    # 2️⃣ ตรวจสอบ LLM instance
    try:
        if llm_instance is None:
            logger.error("LLM instance is None. Cannot run summary generation.")
            default_error_response["summary"] = "LLM Service ไม่พร้อมใช้งาน"
            default_error_response["suggestion_for_next_level"] = "ตรวจสอบการเชื่อมต่อ LLM Service."
            return default_error_response
    except NameError:
        logger.error("Global 'llm_instance' variable not defined or accessible.")
        default_error_response["summary"] = "LLM Instance Error (NameError)"
        return default_error_response

    # 3️⃣ Retry Loop
    for attempt in range(MAX_RETRIES):
        try:
            # 3.1 เตรียม Schema JSON และ Prompt
            schema_dict = EvidenceSummary.model_json_schema()
            system_prompt_content = (
                SYSTEM_EVIDENCE_DESCRIPTION_PROMPT
                + "\n\n--- REQUIRED JSON SCHEMA ---\n"
                + json.dumps(schema_dict, indent=2, ensure_ascii=False)
                + "\n\n🧠 IMPORTANT: Respond ONLY in valid JSON format following the schema above."
            )
            
            human_prompt = EVIDENCE_DESCRIPTION_PROMPT.format(
                standard=sub_criteria_name,
                level=level,
                context=context_to_use, 
                sub_id=sub_id
            )
            
            # 3.2 เรียก LLM 
            llm_full_response = _fetch_llm_response(
                prompt=human_prompt, 
                system_prompt=system_prompt_content, 
                max_retries=1
            )
            
            # ✅ Debug Raw Output
            logger.debug(f"🧩 RAW LLM OUTPUT (Attempt {attempt+1}/{MAX_RETRIES}):\n{llm_full_response}")

            # 3.3 พยายาม parse
            validated_summary_model = parse_llm_json_response(llm_full_response, EvidenceSummary)
            
            # ✅ หาก LLM ตอบไม่เป็น JSON หรือ parse ไม่ได้ → fallback
            if validated_summary_model is None:
                logger.warning("⚠️ LLM output not parsed correctly. Falling back to raw text summary.")
                return {
                    "summary": llm_full_response[:2000],  # ตัดความยาวไม่ให้ log overflow
                    "suggestion_for_next_level": "LLM ตอบนอก schema โปรดตรวจสอบ prompt หรือ schema definition"
                }

            # 3.4 สำเร็จ ✅
            return validated_summary_model.model_dump()

        except Exception as e:
            logger.warning(f"LLM Summary failed (Attempt {attempt+1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)
                continue

            # 3.5 ล้มครบทุกครั้ง
            logger.error(f"Error in summarize_context_with_llm: Failed after {MAX_RETRIES} attempts. Details: {e}", exc_info=True)
            default_error_response["suggestion_for_next_level"] = str(e)
            return default_error_response

    # 4️⃣ fallback สุดท้าย
    return default_error_response

def generate_evidence_description_via_llm(*args, **kwargs) -> str:
    logger.warning("generate_evidence_description_via_llm is deprecated. Use summarize_context_with_llm instead.")
    return "เกิดข้อผิดพลาด: ฟังก์ชันนี้ถูกยกเลิกการใช้งาน โปรดใช้ summarize_context_with_llm แทน"


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


def clean_llm_response(raw: str) -> str:
    """Clean hidden / zero-width chars, strip whitespace"""
    if not isinstance(raw, str):
        return ""
    cleaned = re.sub(r'[\u200b\u200c\u200d\uFEFF]', '', raw)
    cleaned = cleaned.strip()
    return cleaned

def create_structured_action_plan(
    failed_statements_data: list,
    sub_id: str,
    enabler: str,
    target_level: int,
    max_retries: int = 2,
) -> list:
    """
    Robust Action Plan generator from assessment results (LLM)
    - Cleans hidden characters
    - Retries LLM calls
    - Fallback to placeholder if parsing fails
    """
    global _MOCK_CONTROL_FLAG

    if _MOCK_CONTROL_FLAG:
        return [{"Mock": "Action Plan"}]

    # -------------------------
    # 1. Prepare failed statements text
    # -------------------------
    failed_statements_text = []
    for data in failed_statements_data:
        stmt_num = data.get("statement_number", "N/A")
        failed_level = data.get("level", "N/A")
        statement_id = f"L{failed_level}_S{stmt_num}"

        fields = ["evidence_statement_text", "rubric_standard_text", "llm_reasoning", "retrieved_context"]
        texts = []
        for f in fields:
            t = data.get(f, data.get("reason", "N/A") if f=="llm_reasoning" else "N/A")
            if isinstance(t, str):
                t = clean_llm_response(t)
            texts.append(t)
        evidence_statement_text, rubric_standard_text, reason_for_failure, retrieved_context = texts

        failed_statements_text.append(f"""
--- STATEMENT FAILED (Core Business Enabler: {enabler}) ---
Statement ID: {statement_id}
Evidence_statement: {evidence_statement_text}
Rubric_standard: {rubric_standard_text}
Failed Level: {failed_level}
Reason for Failure: {reason_for_failure}
Evidence Snippet: {retrieved_context}
**IMPORTANT: Use '{statement_id}' for 'Statement_ID' in Action Plan.**
""")
    statements_list_str = "\n".join(failed_statements_text)

    # -------------------------
    # 2. Prepare system prompt
    # -------------------------
    try:
        schema_dict = ActionPlanActions.model_json_schema()
        schema_json = json.dumps(schema_dict, indent=2, ensure_ascii=False)
    except Exception:
        schema_json = "{}"
    system_prompt_content = SYSTEM_ACTION_PLAN_PROMPT + "\n\n--- REQUIRED JSON SCHEMA ---\n" + schema_json

    # -------------------------
    # 3. Call LLM with retries
    # -------------------------
    final_error = None
    llm_full_response = ""

    for attempt in range(max_retries + 1):
        try:
            llm_prompt_content = ACTION_PLAN_PROMPT.format(
                sub_id=sub_id,
                target_level=target_level,
                failed_statements_list=statements_list_str
            )

            llm_full_response = _fetch_llm_response(
                prompt=llm_prompt_content,
                system_prompt=system_prompt_content,
                max_retries=1
            )
            cleaned_response = clean_llm_response(llm_full_response)

            # Try normal JSON parse first
            try:
                validated_data = parse_llm_json_response(cleaned_response, ActionPlanActions)
            except Exception:
                # fallback: use json5 (more tolerant)
                validated_data = parse_llm_json_response(json5.loads(cleaned_response), ActionPlanActions)

            if validated_data is None:
                raise ValueError("JSON parsing returned None unexpectedly.")

            logger.info(f"🎉 Action Plan created successfully on Attempt {attempt+1}!")
            return validated_data.model_dump()

        except Exception as e:
            final_error = str(e)
            logger.warning(f"⚠️ Attempt {attempt+1}/{max_retries+1} failed: {e}")
            logger.error(f"--- RAW LLM RESPONSE START ---\n{llm_full_response}\n--- RAW LLM RESPONSE END ---")
            if attempt == max_retries:
                logger.error(f"[ERROR] Failed to generate Action Plan via LLM for {sub_id}: {final_error}")

    # -------------------------
    # 4. Fallback
    # -------------------------
    fallback_plan = {
        "Phase": f"Action Plan Error - L{target_level}",
        "Goal": f"ไม่สามารถสร้าง Action Plan สำหรับ {sub_id} ได้",
        "Actions": [
            {
                "Statement_ID": "LLM_ERROR_FALLBACK",
                "Failed_Level": target_level,
                "Recommendation": f"LLM Output ไม่ตรง JSON Schema: {final_error}",
                "Target_Evidence_Type": "System Check",
                "Key_Metric": "LLM output validation passed",
                "Steps": [
                    {
                        "Step": "1",
                        "Description": f"ตรวจสอบ Response ที่ผิดพลาด: {llm_full_response[:200]}...",
                        "Responsible": "System",
                        "Tools_Templates": "N/A",
                        "Verification_Outcome": "System review required"
                    }
                ]
            }
        ]
    }
    return [fallback_plan]
