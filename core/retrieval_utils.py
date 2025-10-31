#core/retrieval_utils.py
import logging
import random
import json
import time
from typing import List, Dict, Any, Optional
from langchain.schema import SystemMessage, HumanMessage, Document
from pydantic import ValidationError
import regex as re  # use regex for more robust patterns

# --------------------
# Imports from your project schemas & prompts
# --------------------
from core.assessment_schema import StatementAssessment, EvidenceSummary
from core.action_plan_schema import ActionPlanActions  # imported schema (capitalized fields)
from core.rag_prompts import (
    SYSTEM_ASSESSMENT_PROMPT,
    USER_ASSESSMENT_PROMPT,
    ACTION_PLAN_PROMPT,
    SYSTEM_ACTION_PLAN_PROMPT,
    SYSTEM_EVIDENCE_DESCRIPTION_PROMPT,
    EVIDENCE_DESCRIPTION_PROMPT
)

try:
    from core.vectorstore import (
        VectorStoreManager, 
        load_all_vectorstores,
        # 🟢 NEW IMPORTS: จำเป็นสำหรับการสร้าง Retriever
        get_vectorstore_manager, 
        _get_collection_name, 
        INITIAL_TOP_K, 
        FINAL_K_RERANKED, 
        FINAL_K_NON_RERANKED, # ตัวแปร K สำหรับกรณีไม่มี Reranker
        NamedRetriever,
        MultiDocRetriever,
    )
except Exception:
    VectorStoreManager = None
    load_all_vectorstores = None

try:
    from models.llm import llm as llm_instance
except Exception:
    llm_instance = None

logger = logging.getLogger(__name__)
# 🔴 ลบ logging.basicConfig ออกเพื่อไม่ให้ไปทับ config หลัก
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') 

# =================================================================
# MOCKING LOGIC AND GLOBAL FLAGS
# =================================================================

_MOCK_CONTROL_FLAG = False
_MOCK_COUNTER = 0

def set_mock_control_mode(enable: bool):
    """Enable/disable controlled mock mode for deterministic tests."""
    global _MOCK_CONTROL_FLAG, _MOCK_COUNTER
    _MOCK_CONTROL_FLAG = enable
    if enable:
        _MOCK_COUNTER = 0
        logger.info("🔑 CONTROLLED MOCK Mode ENABLED.")
    else:
        logger.info("❌ CONTROLLED MOCK Mode DISABLED.")


# --------------------
# NEW FUNCTION: Retrieve documents directly by UUIDs
# --------------------


def retrieve_context_by_doc_ids(
    doc_uuids: List[str],
    # 🛑 MODIFIED: เปลี่ยนชื่อเป็น collection_name เพื่อรับชื่อ collection ที่รวม enabler แล้ว
    collection_name: str 
) -> Dict[str, Any]:
    """
    Retrieves documents (chunks) directly by their UUIDs from a specific 
    Chroma collection using VectorStoreManager.
    
    Returns: {"top_evidences": List[Dict[str, Any]]}
    """
    if VectorStoreManager is None:
        logger.error("❌ VectorStoreManager is not available.")
        return {"top_evidences": []}
        
    if not doc_uuids:
        return {"top_evidences": []}

    try:
        # 🚨 NOTE: Assuming VectorStoreManager can be initialized without arguments 
        # or it handles loading all necessary vector stores internally.
        manager = VectorStoreManager() 
        
        # 🛑 MODIFIED: ส่ง collection_name ไปเป็น doc_type เพื่อเข้าถึง Collection ที่ถูกต้อง
        docs: List[Document] = manager.get_documents_by_id(doc_uuids, doc_type=collection_name)

        # Format result
        top_evidences = []
        for d in docs:
            meta = d.metadata
            top_evidences.append({
                "doc_id": meta.get("doc_id"),
                "doc_type": meta.get("doc_type"), # 🟢 เพิ่ม doc_type
                "chunk_uuid": meta.get("chunk_uuid"), # 🟢 เพิ่ม chunk_uuid
                "source": meta.get("source") or meta.get("doc_source"),
                "content": d.page_content.strip(),
                "chunk_index": meta.get("chunk_index")
            })

        logger.info(f"✅ Successfully retrieved {len(top_evidences)} evidences by UUIDs from collection '{collection_name}'.") # 🛑 ใช้ collection_name
        return {"top_evidences": top_evidences}

    except Exception as e:
        logger.error(f"Error during UUID-based retrieval: {e}", exc_info=True)
        return {"top_evidences": []}

# --------------------
# RETRIEVER FUNCTION (RAG Search)
# --------------------
# --------------------
# RETRIEVER FUNCTION (RAG Search)
# --------------------
def retrieve_context_with_filter(
    query: str, 
    doc_type: str, 
    enabler: str, 
    stable_doc_ids: Optional[List[str]] = None, 
    disable_semantic_filter: bool = False,
    # 🟢 NEW PARAMETER: ควบคุม Fallback Logic
    allow_fallback: bool = False 
) -> Dict[str, Any]:
    """
    Retrieves documents for a given query, creates the appropriate retriever 
    (MultiDocRetriever with Reranker or not) and filters results based on stable_doc_ids.
    
    In strict filter mode (allow_fallback=False), if stable_doc_ids is provided, 
    it strictly enforces the hard filter and prevents any fallback to full search.
    """
    if VectorStoreManager is None or 'get_vectorstore_manager' not in globals():
        logger.error("❌ VectorStoreManager or required components are not available. Skipping RAG retrieval.")
        return {"top_evidences": []}
        
    try:
        # 1. เตรียม Vector Store
        manager = get_vectorstore_manager()
        collection_name = _get_collection_name(doc_type, enabler)

        # 🟢 NEW: Logic สำหรับ Strict Filter (โหมดที่ต้องห้าม Fallback)
        is_strict_mode = not allow_fallback
        
        # 🟢 NEW: ใน Strict Mode ถ้าไม่มีการ Map ID เลย ให้คืนค่าว่างทันที (ห้ามค้นหา)
        if is_strict_mode and (stable_doc_ids is None or not stable_doc_ids):
             logger.info("🛑 Strict Filter Mode: No mapped stable_doc_ids. Returning empty context to enforce manual mapping check.")
             return {"top_evidences": []}


        # 2. เลือก Final K ที่เหมาะสมตาม flag
        final_k = FINAL_K_NON_RERANKED if disable_semantic_filter else FINAL_K_RERANKED
        
        # 3. สร้าง NamedRetriever spec
        retriever_spec = NamedRetriever(
            doc_id=f"{doc_type}_{enabler}",
            doc_type=collection_name,
            top_k=INITIAL_TOP_K, 
            final_k=final_k 
        )
        
        # 4. ใช้ MultiDocRetriever
        # 🚨 KEY CHANGE: MultiDocRetriever มี doc_ids_filter 
        # ถ้า stable_doc_ids ถูกส่งมา มันจะทำ Hard Filter 
        # เราต้องมั่นใจว่า MultiDocRetriever ไม่มี Fallback Logic ภายใน (ซึ่งโดยทั่วไป LangChain จะไม่ทำ Fallback อัตโนมัติ)
        multi_retriever = MultiDocRetriever(
            retrievers_list=[retriever_spec],
            k_per_doc=INITIAL_TOP_K, 
            doc_ids_filter=stable_doc_ids 
        )
        
        # 5. ดำเนินการค้นหา
        documents = multi_retriever.get_relevant_documents(query)
        
        # 🟢 NEW: Logic สำหรับการจัดการผลลัพธ์ใน Strict Mode
        if is_strict_mode and stable_doc_ids:
            if not documents:
                # 🛑 ใน Strict Mode ถ้า Map ID มาแล้ว แต่ค้นหาไม่พบ (เพราะไม่ใกล้เคียงทางความหมาย)
                # เราถือว่าหลักฐานที่ Map ไว้ "ไม่เพียงพอ" หรือ "ไม่ตรง"
                logger.info(f"🛑 Strict Filter Mode: Found 0 documents despite stable_doc_ids being provided. Returning empty context.")
                return {"top_evidences": []}
        
        # 6. Format result
        # ... (ส่วนนี้ใช้โค้ดเดิม) ...
        top_evidences = []
        for d in documents:
            meta = d.metadata
            top_evidences.append({
                "doc_id": meta.get("doc_id") or meta.get("doc_source"), 
                "doc_type": meta.get("doc_type"),
                "relevance_score": meta.get("relevance_score"),
                "source": meta.get("source") or meta.get("doc_source"),
                "content": d.page_content.strip()
            })
            
        logger.info(f"RAG Retrieval for query='{query[:30]}...' found {len(documents)} evidences (Strict Mode: {is_strict_mode}, Filtered by ID: {bool(stable_doc_ids)})")
            
        return {"top_evidences": top_evidences}

    except Exception as e:
        logger.error(f"Error during RAG retrieval with filter: {e}", exc_info=True)
        return {"top_evidences": []}


# =================================================================
# ROBUST JSON EXTRACTION (single, canonical implementation)
# =================================================================

def _robust_extract_json(text: str) -> Optional[Any]:
    """
    Attempts to extract a complete and valid JSON object from the LLM response text.
    - Removes common fences and trailing garbage.
    - Uses json.JSONDecoder.raw_decode to locate and decode the first JSON object.
    - Returns dict/list if found, otherwise None.
    """
    if not text:
        return None

    cleaned_text = text.strip()

    # Remove code fences (```json or ```)
    cleaned_text = re.sub(r'^\s*```(?:json)?\s*', '', cleaned_text, flags=re.MULTILINE | re.IGNORECASE)
    cleaned_text = re.sub(r'\s*```\s*$', '', cleaned_text, flags=re.MULTILINE)

    # Remove any leading "json" or similar tokens
    cleaned_text = re.sub(r'^\s*json[:\s]*', '', cleaned_text, flags=re.IGNORECASE)

    # Find first '{' or '[' as potential JSON start
    brace_idx = min(
        [idx for idx in (cleaned_text.find('{'), cleaned_text.find('[')) if idx != -1],
        default=-1
    )
    if brace_idx == -1:
        return None

    candidate = cleaned_text[brace_idx:].strip()

    # Try JSONDecoder.raw_decode
    try:
        decoder = json.JSONDecoder()
        obj, end_idx = decoder.raw_decode(candidate)
        return obj
    except json.JSONDecodeError:
        # Fallback: try to find last closing brace and clean trailing commas
        try:
            last_brace = max(candidate.rfind('}'), candidate.rfind(']'))
            if last_brace != -1:
                json_str = candidate[:last_brace + 1]
                # remove trailing commas before closing braces/brackets
                json_str = re.sub(r',\s*([\]\}])', r'\1', json_str)
                return json.loads(json_str)
        except Exception:
            pass

    return None


# =================================================================
# EVALUATION FUNCTION (MOCK & REAL LLM)
# =================================================================

MAX_LLM_RETRIES = 3

def evaluate_with_llm(statement: str, context: str, standard: str, **kwargs) -> Dict[str, Any]:
    """
    Use LLM to evaluate a statement against standard. Returns dict with:
    - score (int 0/1)
    - reason (str)
    - pass_status (bool)
    - status_th (Thai text)
    """
    global _MOCK_CONTROL_FLAG, _MOCK_COUNTER

    # MOCK mode
    if _MOCK_CONTROL_FLAG:
        _MOCK_COUNTER += 1
        score = 1 if _MOCK_COUNTER <= 9 else 0
        reason_text = f"MOCK: FORCED {'PASS' if score == 1 else 'FAIL'} (Statement {_MOCK_COUNTER})"
        is_pass = score >= 1
        status_th = "ผ่าน" if is_pass else "ไม่ผ่าน"
        logger.debug(f"MOCK COUNT: {_MOCK_COUNTER} | SCORE: {score} | STMT: '{statement[:20]}...'")
        return {"score": score, "reason": reason_text, "pass_status": is_pass, "status_th": status_th}

    # LLM not available fallback
    if llm_instance is None:
        logger.error("❌ LLM Instance is not initialized.")
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
                [
                    SystemMessage(content=SYSTEM_ASSESSMENT_PROMPT),
                    HumanMessage(content=user_prompt_content)
                ],
                **({'format': 'json'} if hasattr(llm_instance, 'model_params') and 'format' in llm_instance.model_params else {})
            )

            llm_response_content = response.content if hasattr(response, 'content') else str(response)
            llm_output = _robust_extract_json(llm_response_content)

            if not llm_output or not isinstance(llm_output, dict):
                raise ValueError("LLM response did not contain a recognizable JSON block.")

            raw_score = llm_output.get("llm_score", llm_output.get("score"))
            reason = llm_output.get("reason")

            if raw_score is not None and reason is not None:
                try:
                    score_int = int(str(raw_score))
                except ValueError:
                    score_int = 0

                validated_data = {"score": score_int, "reason": reason}

                # Validate with StatementAssessment (best-effort)
                try:
                    StatementAssessment.model_validate(validated_data)
                except ValidationError as ve:
                    logger.warning(f"Partial Validation Error in Statement Assessment (Keys are OK): {ve}")

                final_score = validated_data["score"]
                is_pass = final_score >= 1
                status_th = "ผ่าน" if is_pass else "ไม่ผ่าน"
                result_data = validated_data
                result_data["pass_status"] = is_pass
                result_data["status_th"] = status_th
                return result_data
            else:
                raise ValueError("LLM response JSON is missing 'score' or 'reason' keys.")

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"❌ Format/JSON Parse Failed (Attempt {attempt + 1}/{MAX_LLM_RETRIES}). Retrying in 1s... Error: {str(e)[:150]}...")
            if attempt < MAX_LLM_RETRIES - 1:
                time.sleep(1)
                continue
            else:
                logger.error(f"❌ LLM Evaluation failed after {MAX_LLM_RETRIES} attempts. JSON/Format failure.")
                break
        except Exception as e:
            logger.error(f"❌ LLM Evaluation failed due to unexpected error: {e}", exc_info=True)
            break

    # fallback random
    logger.error("❌ Using RANDOM SCORE as final fallback.")
    score = random.choice([0, 1])
    reason = f"LLM Call Failed (Fallback to Random Score {score}) after {MAX_LLM_RETRIES} attempts."
    is_pass = score >= 1
    status_th = "ผ่าน" if is_pass else "ไม่ผ่าน"
    return {"score": score, "reason": reason, "pass_status": is_pass, "status_th": status_th}


# =================================================================
# ACTION PLAN GENERATION UTILITIES (minimal patch)
# =================================================================

def _call_llm_for_json_output(prompt: str, system_prompt: str, max_retries: int = 1) -> str:
    """Call LLM and aggressively try to return the raw JSON block as string."""
    if llm_instance is None:
        raise ConnectionError("LLM Instance is not available for Action Plan Generation.")

    for attempt in range(max_retries):
        try:
            response = llm_instance.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=prompt)],
                config={"temperature": 0.0}
            )
            llm_response_content = response.content if hasattr(response, 'content') else str(response)
            logger.debug(f"Raw LLM Response (Attempt {attempt + 1}): {llm_response_content}")

            cleaned_output = llm_response_content.strip()
            # remove leading backticks and 'json' token
            cleaned_output = cleaned_output.lstrip('`')
            if cleaned_output.lower().startswith('json'):
                cleaned_output = cleaned_output[4:].lstrip()
            # remove trailing backticks
            cleaned_output = cleaned_output.rstrip('`').rstrip()

            first_brace_index = cleaned_output.find('{')
            last_brace_index = cleaned_output.rfind('}')
            if first_brace_index != -1 and last_brace_index > first_brace_index:
                return cleaned_output[first_brace_index:last_brace_index + 1]

            return cleaned_output

        except Exception as e:
            logger.warning(f"LLM call attempt {attempt+1} failed: {e}")
            time.sleep(1)

    raise Exception("LLM call failed after max retries for raw JSON generation.")


def generate_action_plan_via_llm(
    failed_statements_data: List[Dict[str, Any]],
    sub_id: str,
    target_level: int,
    max_retries: int = 2,
    retry_delay: float = 1.0
) -> Dict[str, Any]:
    """
    Generate Action Plan by calling LLM. Minimal changes:
    - Use imported ActionPlanActions schema (from core.action_plan_schema)
    - Keep existing retry & validation behavior
    """
    global _MOCK_CONTROL_FLAG

    # MOCK mode
    if _MOCK_CONTROL_FLAG:
        logger.warning("MOCK: Generating dummy Action Plan via MOCK Logic.")
        actions = []
        # Use imported schema's model_fields - ActionPlanActions uses 'Actions' key in your schema
        # NOTE: If your imported schema's list field name differs, adjust below.
        try:
            ActionItemType = ActionPlanActions.model_fields['Actions'].annotation.__args__[0]
        except Exception:
            # As a safe fallback, build a simple dict
            ActionItemType = None

        for i, data in enumerate(failed_statements_data):
            statement_id = f"L{data.get('level', target_level)} S{data.get('statement_number', i+1)}"
            failed_level = data.get('level', target_level)
            stmt_text = data.get('statement_text', 'N/A')[:50]
            reason_text = data.get('llm_reasoning', 'No reason')[:50]

            if ActionItemType:
                # construct via Pydantic model if available
                actions.append(ActionItemType(
                    Statement_ID=statement_id,
                    Failed_Level=failed_level,
                    Recommendation=f"MOCK: [Specific Action] จัดทำ Policy '{stmt_text}...' เพื่อแก้ GAP: {reason_text}...",
                    Target_Evidence_Type="MOCK: Policy Document (Guideline)",
                    Key_Metric="Policy Approved and Published"
                ).model_dump())
            else:
                actions.append({
                    "Statement_ID": statement_id,
                    "Failed_Level": failed_level,
                    "Recommendation": f"MOCK: [Specific Action] จัดทำ Policy '{stmt_text}...' เพื่อแก้ GAP: {reason_text}...",
                    "Target_Evidence_Type": "MOCK: Policy Document (Guideline)",
                    "Key_Metric": "Policy Approved and Published"
                })

        return ActionPlanActions(
            Phase=f"1. Strategic Gap Closure (Target L{target_level})",
            Goal=f"บรรลุหลักฐานที่จำเป็นทั้งหมดใน Level {target_level} ของ {sub_id}",
            Actions=actions
        ).model_dump()

    # REAL LLM logic: prepare prompt
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
RAG Context Found: {data.get('retrieved_context', 'No context found or saved')}
**IMPORTANT: The Action Plan must use '{statement_id}' for the 'Statement_ID' field.**
""")
    statements_list_str = "\n".join(failed_statements_text)

    llm_prompt_content = ACTION_PLAN_PROMPT.format(
        sub_id=sub_id,
        target_level=target_level,
        failed_statements_list=statements_list_str
    )

    # Include schema in system prompt to help LLM match structure
    try:
        schema_dict = ActionPlanActions.model_json_schema()
        schema_json = json.dumps(schema_dict, indent=2, ensure_ascii=False)
    except Exception:
        schema_json = "{}"

    system_prompt_content = (
        SYSTEM_ACTION_PLAN_PROMPT
        + "\n\n--- REQUIRED JSON SCHEMA (STRICTLY FOLLOW) ---\n"
        + schema_json
        + "\n\n**WARNING: Response MUST contain ONLY the JSON object. Every action item MUST include Statement_ID, Failed_Level, Recommendation, Target_Evidence_Type, and Key_Metric.**"
    )

    final_error = None

    for attempt in range(max_retries + 1):
        try:
            logger.info(f"LLM Call Attempt {attempt+1}/{max_retries+1} for Action Plan generation (Sub: {sub_id} L{target_level})")

            llm_full_response = _call_llm_for_json_output(
                prompt=llm_prompt_content,
                system_prompt=system_prompt_content,
                max_retries=1
            )

            if not llm_full_response or not llm_full_response.strip():
                raise ValueError("LLM returned empty or invalid response for Action Plan.")

            llm_result = _robust_extract_json(llm_full_response)
            if not llm_result:
                raise ValueError("Failed to extract JSON for Action Plan from LLM response.")

            # Normalize possible lowercase keys -> make sure keys match schema expected names
            # If LLM returned 'actions' instead of 'Actions', convert it.
            if isinstance(llm_result, dict):
                if 'actions' in llm_result and 'Actions' not in llm_result:
                    llm_result['Actions'] = llm_result.pop('actions')
                if 'phase' in llm_result and 'Phase' not in llm_result:
                    llm_result['Phase'] = llm_result.pop('phase')
                if 'goal' in llm_result and 'Goal' not in llm_result:
                    llm_result['Goal'] = llm_result.pop('goal')

            # Validate using imported Pydantic schema
            validated_plan_model = ActionPlanActions.model_validate(llm_result)
            final_action_plan_result = validated_plan_model.model_dump()
            logger.debug(f"✅ Successfully generated Action Plan for {sub_id} L{target_level}")
            return final_action_plan_result

        except (json.JSONDecodeError, ValueError, ValidationError) as e:
            final_error = str(e)
            error_type = "Validation Error" if isinstance(e, ValidationError) else "JSON/Value Error"
            logger.warning(f"❌ Action Plan {error_type} (Attempt {attempt + 1}/{max_retries + 1}). Retrying in {retry_delay}s... Error: {str(e)[:200]}")

            if attempt < max_retries:
                time.sleep(retry_delay)
                continue
            else:
                break

        except Exception as e:
            final_error = str(e)
            logger.error(f"❌ Action Plan Generation Failed (Unexpected Error): {e}", exc_info=True)
            break

    final_error = final_error if final_error else "Unknown Error during LLM Generation."
    # raise to caller to handle (keeps behavior consistent with your pipeline)
    raise Exception(final_error)


# =================================================================
# NARRATIVE & EVIDENCE SUMMARY HELPERS
# =================================================================

def generate_narrative_report_via_llm_real(prompt_text: str, system_instruction: str) -> str:
    """
    Real LLM call for narrative report. Kept as-is (minimal patch).
    """
    if llm_instance is None:
        logger.error("❌ LLM Instance is not initialized for real API call.")
        return "[ERROR: LLM Client is not initialized for real API call.]"

    try:
        response = llm_instance.invoke([SystemMessage(content=system_instruction), HumanMessage(content=prompt_text)])
        generated_text = response.content if hasattr(response, 'content') else str(response)
        return generated_text.strip()
    except Exception as e:
        logger.error(f"Real LLM API call failed during narrative report generation: {e}")
        return f"[API ERROR] Failed to generate narrative report via real LLM API: {e}"


def summarize_context_with_llm(context: str, sub_criteria_name: str, level: int, sub_id: str, schema: Any) -> Dict[str, str]:
    """
    Use LLM to create EvidenceSummary and validate via imported EvidenceSummary Pydantic model.
    """
    MAX_LLM_SUMMARY_CONTEXT = 5000

    if llm_instance is None:
        return {"summary": "เกิดข้อผิดพลาด: LLM Client ไม่พร้อมใช้งาน", "suggestion_for_next_level": "N/A"}

    context_to_use = context
    if len(context) > MAX_LLM_SUMMARY_CONTEXT:
        logger.warning(f"Context for summary L{level} is too long ({len(context)}), truncating to {MAX_LLM_SUMMARY_CONTEXT}.")
        context_to_use = context[:MAX_LLM_SUMMARY_CONTEXT]

    try:
        schema_dict = EvidenceSummary.model_json_schema()
        system_prompt_content = (
            SYSTEM_EVIDENCE_DESCRIPTION_PROMPT + "\n\n--- REQUIRED JSON SCHEMA (STRICTLY FOLLOW) ---\n" + json.dumps(schema_dict, indent=2, ensure_ascii=False)
        )

        human_prompt = EVIDENCE_DESCRIPTION_PROMPT.format(
            standard=sub_criteria_name,
            level=level,
            context=context_to_use,
            sub_id=sub_id
        )

        llm_full_response = _call_llm_for_json_output(prompt=human_prompt, system_prompt=system_prompt_content)
        llm_result_dict = _robust_extract_json(llm_full_response)

        if not llm_result_dict or not isinstance(llm_result_dict, dict):
            raise ValueError("LLM response did not contain a recognizable JSON block for Evidence Summary.")

        validated_summary_model = EvidenceSummary.model_validate(llm_result_dict)
        logger.info(f"✅ Generated Evidence Summary for {sub_criteria_name} L{level}")
        return validated_summary_model.model_dump()

    except Exception as e:
        logger.error(f"LLM Summary generation failed: {e}", exc_info=True)
        return {"summary": "ข้อผิดพลาดในการสรุปข้อมูลโดย LLM", "suggestion_for_next_level": f"เกิดข้อผิดพลาดในการเรียก LLM: {str(e)}"}


def generate_evidence_description_via_llm(*args, **kwargs) -> str:
    logger.warning("generate_evidence_description_via_llm is deprecated. Use summarize_context_with_llm instead.")
    return "เกิดข้อผิดพลาด: ฟังก์ชันนี้ถูกยกเลิกการใช้งาน โปรดใช้ summarize_context_with_llm แทน"