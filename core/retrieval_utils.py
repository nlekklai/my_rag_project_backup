# core/retrieval_utils.py
import logging
import random
import json
import time
from typing import List, Dict, Any, Optional, Union, TypeVar, Type, Tuple
from langchain.schema import SystemMessage, HumanMessage, Document as LcDocument
from pydantic import ValidationError
import regex as re
import hashlib

from langchain.schema import SystemMessage, HumanMessage, Document as LcDocument
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever # 🟢 ต้องเพิ่ม

# --------------------
# Imports from your project schemas & prompts
# --------------------
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
    INITIAL_TOP_K,
    load_all_vectorstores, 
    INITIAL_TOP_K, 
    get_reranking_compressor,
    FINAL_K_RERANKED,
    FINAL_K_NON_RERANKED,
    NamedRetriever,
    MultiDocRetriever,
)

from pydantic import ValidationError, BaseModel
T = TypeVar('T', bound=BaseModel)

try:
    from models.llm import llm as llm_instance
except Exception:
    llm_instance = None

# -------------------- Config --------------------
DEFAULT_ENABLER ="KM"

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
        # (ฟังก์ชัน normalize_stable_ids ควรแปลง ID ที่เข้ามาให้เป็น 64-char Hash)
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
    top_k_reranked: int = FINAL_K_RERANKED, # 🟢 ใช้ FINAL_K_RERANKED เป็น default
    disable_semantic_filter: bool = False,
    allow_fallback: bool = False
) -> Dict[str, Any]:
    
    global INITIAL_TOP_K, FINAL_K_NON_RERANKED # 🟢 ใช้ FINAL_K_NON_RERANKED ด้วย
    if not isinstance(INITIAL_TOP_K, int):
        INITIAL_TOP_K = 15 
    if not isinstance(FINAL_K_NON_RERANKED, int):
        FINAL_K_NON_RERANKED = 5

    if VectorStoreManager is None:
        logger.error("❌ VectorStoreManager is not available.")
        return {"top_evidences": []}
    
    try:
        manager = VectorStoreManager()
        # collection_name = f"{doc_type}_{enabler.lower()}"
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
            
            # ❗️ Assumption: get_reranking_compressor ถูก Import จาก core.vectorstore
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
            if isinstance(doc, LcDocument):
                top_evidences.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })

        logger.info(f"RAG Final Output for query='{query[:30]}...' found {len(top_evidences)} evidences.")
        return {"top_evidences": top_evidences}

    except Exception as e:
        logger.error(f"Error during RAG retrieval with filter (Combined Logic): {e}", exc_info=True)
        return {"top_evidences": []}
    
# ------------------------------------------------------------------
# Robust JSON Extraction
# ------------------------------------------------------------------
def _robust_extract_json(text: str) -> Optional[Any]:
    if not text:
        return None
    cleaned_text = text.strip()
    cleaned_text = re.sub(r'^\s*```(?:json)?\s*', '', cleaned_text, flags=re.MULTILINE | re.IGNORECASE)
    cleaned_text = re.sub(r'\s*```\s*$', '', cleaned_text, flags=re.MULTILINE)
    cleaned_text = re.sub(r'^\s*json[:\s]*', '', cleaned_text, flags=re.IGNORECASE)
    brace_idx = min([idx for idx in (cleaned_text.find('{'), cleaned_text.find('[')) if idx != -1], default=-1)
    if brace_idx == -1:
        return None
    candidate = cleaned_text[brace_idx:].strip()
    try:
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(candidate)
        return obj
    except json.JSONDecodeError:
        try:
            last_brace = max(candidate.rfind('}'), candidate.rfind(']'))
            if last_brace != -1:
                json_str = candidate[:last_brace + 1]
                json_str = re.sub(r',\s*([\]\}])', r'\1', json_str)
                return json.loads(json_str)
        except Exception:
            pass
    return None

def parse_llm_json_response(llm_response_text: str, pydantic_schema: Type[T]) -> Union[T, List[T]]:
    """
    ดึง JSON จากข้อความ LLM และตรวจสอบความถูกต้องด้วย Pydantic Schema.
    """
    raw_data = _robust_extract_json(llm_response_text)
    
    if raw_data is None:
        raise ValueError("Could not robustly extract valid JSON from LLM response.")
        
    try:
        if hasattr(pydantic_schema, '__origin__') and pydantic_schema.__origin__ is list:
            item_schema = pydantic_schema.__args__[0]
            if not isinstance(raw_data, list):
                raw_data = [raw_data]
            
            validated_list = [item_schema.model_validate(item) for item in raw_data]
            return validated_list
        else:
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

def evaluate_with_llm(statement: str, context: str, standard: str, **kwargs) -> Dict[str, Any]:
    global _MOCK_CONTROL_FLAG, _MOCK_COUNTER
    if _MOCK_CONTROL_FLAG:
        _MOCK_COUNTER += 1
        score = 1 if _MOCK_COUNTER <= 9 else 0
        reason_text = f"MOCK: FORCED {'PASS' if score == 1 else 'FAIL'} (Statement {_MOCK_COUNTER})"
        is_pass = score >= 1
        status_th = "ผ่าน" if is_pass else "ไม่ผ่าน"
        return {"score": score, "reason": reason_text, "pass_status": is_pass, "status_th": status_th}

    if llm_instance is None:
        score = random.choice([0, 1])
        reason = f"LLM Initialization Failed (Fallback to Random Score {score})"
        is_pass = score >= 1
        status_th = "ผ่าน" if is_pass else "ไม่ผ่าน"
        return {"score": score, "reason": reason, "pass_status": is_pass, "status_th": status_th}

    user_prompt_content = USER_ASSESSMENT_PROMPT.format(
        statement=statement,
        standard=standard,
        context=context if context else "ไม่พบหลักฐานในเอกสารที่เกี่ยวข้อง"
    )

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

            raw_score = llm_output.get("llm_score", llm_output.get("score"))
            reason = llm_output.get("reason")
            score_int = int(str(raw_score)) if raw_score is not None else 0
            validated_data = {"score": score_int, "reason": reason}
            try:
                StatementAssessment.model_validate(validated_data)
            except ValidationError:
                pass
            is_pass = score_int >= 1
            status_th = "ผ่าน" if is_pass else "ไม่ผ่าน"
            validated_data.update({"pass_status": is_pass, "status_th": status_th})
            return validated_data
        except Exception as e:
            if attempt < MAX_LLM_RETRIES - 1:
                time.sleep(1)
                continue
    # fallback
    score = random.choice([0, 1])
    reason = f"LLM Call Failed (Fallback to Random Score {score})"
    is_pass = score >= 1
    status_th = "ผ่าน" if is_pass else "ไม่ผ่าน"
    return {"score": score, "reason": reason, "pass_status": is_pass, "status_th": status_th}

# =================================================================
# Narrative & Evidence Summary
# =================================================================
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
        llm_full_response = _call_llm_for_json_output(prompt=human_prompt, system_prompt=system_prompt_content)
        llm_result_dict = _robust_extract_json(llm_full_response)
        validated_summary_model = EvidenceSummary.model_validate(llm_result_dict)
        return validated_summary_model.model_dump()
    except Exception as e:
        return {"summary": "ข้อผิดพลาดในการสรุปข้อมูลโดย LLM", "suggestion_for_next_level": str(e)}

def generate_evidence_description_via_llm(*args, **kwargs) -> str:
    logger.warning("generate_evidence_description_via_llm is deprecated. Use summarize_context_with_llm instead.")
    return "เกิดข้อผิดพลาด: ฟังก์ชันนี้ถูกยกเลิกการใช้งาน โปรดใช้ summarize_context_with_llm แทน"

# =================================================================
# LLM Action Plan
# =================================================================
def _call_llm_for_json_output(prompt: str, system_prompt: str, max_retries: int = 1) -> str:
    if llm_instance is None:
        raise ConnectionError("LLM Instance is not available.")
    for attempt in range(max_retries):
        try:
            response = llm_instance.invoke([SystemMessage(content=system_prompt), HumanMessage(content=prompt)], config={"temperature": 0.0})
            cleaned_output = (response.content if hasattr(response, 'content') else str(response)).strip().lstrip('`')
            if cleaned_output.lower().startswith('json'):
                cleaned_output = cleaned_output[4:].lstrip()
            cleaned_output = cleaned_output.rstrip('`').rstrip()
            first_brace_index = cleaned_output.find('{')
            last_brace_index = cleaned_output.rfind('}')
            if first_brace_index != -1 and last_brace_index > first_brace_index:
                return cleaned_output[first_brace_index:last_brace_index+1]
            return cleaned_output
        except Exception as e:
            time.sleep(1)
    raise Exception("LLM call failed after max retries for raw JSON generation.")

def generate_action_plan_via_llm(failed_statements_data: List[Dict[str, Any]], sub_id: str, target_level: int, max_retries: int = 2, retry_delay: float = 1.0) -> Dict[str, Any]:
    global _MOCK_CONTROL_FLAG
    if _MOCK_CONTROL_FLAG:
        actions = []
        try:
            ActionItemType = ActionPlanActions.model_fields['Actions'].annotation.__args__[0]
        except Exception:
            ActionItemType = None
        for i, data in enumerate(failed_statements_data):
            statement_id = f"L{data.get('level', target_level)} S{data.get('statement_number', i+1)}"
            failed_level = data.get('level', target_level)
            stmt_text = data.get('statement_text', 'N/A')[:50]
            reason_text = data.get('llm_reasoning', 'No reason')[:50]
            if ActionItemType:
                actions.append(ActionItemType(
                    Statement_ID=statement_id,
                    Failed_Level=failed_level,
                    Recommendation=f"MOCK: [Action] '{stmt_text}...' เพื่อแก้ GAP: {reason_text}...",
                    Target_Evidence_Type="MOCK: Policy Document (Guideline)",
                    Key_Metric="Policy Approved and Published"
                ).model_dump())
            else:
                actions.append({
                    "Statement_ID": statement_id,
                    "Failed_Level": failed_level,
                    "Recommendation": f"MOCK: [Action] '{stmt_text}...' เพื่อแก้ GAP: {reason_text}...",
                    "Target_Evidence_Type": "MOCK: Policy Document (Guideline)",
                    "Key_Metric": "Policy Approved and Published"
                })
        return ActionPlanActions(
            Phase=f"1. Strategic Gap Closure (Target L{target_level})",
            Goal=f"บรรลุหลักฐานที่จำเป็นทั้งหมดใน Level {target_level} ของ {sub_id}",
            Actions=actions
        ).model_dump()

    failed_statements_text = []
    for data in failed_statements_data:
        stmt_num = data.get('statement_number', 'N/A')
        failed_level = data.get('level', 'N/A')
        statement_id = f"L{failed_level} S{stmt_num}"
        failed_statements_text.append(f"""
--- STATEMENT FAILED (Statement ID: {statement_id}) ---
Statement Text: {data.get('statement_text', 'N/A')}
Failed Level: {failed_level}
Reason for Failure: {data.get('llm_reasoning', 'N/A')}
RAG Context Found: {data.get('retrieved_context', 'No context found')}
**IMPORTANT: The Action Plan must use '{statement_id}' for 'Statement_ID'.**
""")
    statements_list_str = "\n".join(failed_statements_text)
    llm_prompt_content = ACTION_PLAN_PROMPT.format(sub_id=sub_id, target_level=target_level, failed_statements_list=statements_list_str)
    try:
        schema_dict = ActionPlanActions.model_json_schema()
        schema_json = json.dumps(schema_dict, indent=2, ensure_ascii=False)
    except Exception:
        schema_json = "{}"
    system_prompt_content = SYSTEM_ACTION_PLAN_PROMPT + "\n\n--- REQUIRED JSON SCHEMA ---\n" + schema_json
    final_error = None
    for attempt in range(max_retries + 1):
        try:
            llm_full_response = _call_llm_for_json_output(prompt=llm_prompt_content, system_prompt=system_prompt_content)
            llm_result = _robust_extract_json(llm_full_response)
            if not llm_result:
                raise ValueError("Failed to extract JSON for Action Plan.")
            if isinstance(llm_result, dict):
                if 'actions' in llm_result and 'Actions' not in llm_result:
                    llm_result['Actions'] = llm_result.pop('actions')
                if 'phase' in llm_result and 'Phase' not in llm_result:
                    llm_result['Phase'] = llm_result.pop('phase')
                if 'goal' in llm_result and 'Goal' not in llm_result:
                    llm_result['Goal'] = llm_result.pop('goal')
            validated_plan_model = ActionPlanActions.model_validate(llm_result)
            return validated_plan_model.model_dump()
        except Exception as e:
            final_error = str(e)
            if attempt < max_retries:
                time.sleep(retry_delay)
                continue
            else:
                break
    raise Exception(final_error if final_error else "Unknown Error during LLM Action Plan generation.")
