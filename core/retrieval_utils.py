#core/retrieval_utils.py
import logging
import random
import json
import time 
from typing import List, Dict, Any, Optional, Union
from langchain.schema import SystemMessage, HumanMessage 
from langchain.schema import Document 

# 🚨 IMPORT: นำเข้า Regex (จำเป็นสำหรับการแก้ไขนี้)
import re 
# 🚨 IMPORT: นำเข้า Pydantic Model จากไฟล์ action_plan_schema.py
from core.action_plan_schema import ActionPlanActions 
# 🟢 NEW: นำเข้า Pydantic Model จากไฟล์ assessment_schema.py
from core.assessment_schema import StatementAssessment, EvidenceSummary
# 🚨 IMPORT: นำเข้า Prompts
from core.rag_prompts import (
    SYSTEM_ASSESSMENT_PROMPT, 
    USER_ASSESSMENT_PROMPT, 
    ACTION_PLAN_PROMPT,
    SYSTEM_ACTION_PLAN_PROMPT,
    SYSTEM_EVIDENCE_DESCRIPTION_PROMPT, 
    EVIDENCE_DESCRIPTION_PROMPT         
) 
from core.vectorstore import VectorStoreManager, load_all_vectorstores 
from models.llm import llm as llm_instance 

logger = logging.getLogger(__name__)
# เนื่องจากคุณรันในโหมด real และต้องการ debug/info จึงใช้ level INFO
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# =================================================================
# === MOCKING LOGIC AND GLOBAL FLAGS ===
# =================================================================

_MOCK_CONTROL_FLAG = False
_MOCK_COUNTER = 0

def set_mock_control_mode(enable: bool):
    """ฟังก์ชันสำหรับเปิด/ปิดโหมดควบคุมคะแนน (CONTROLLED MOCK)"""
    global _MOCK_CONTROL_FLAG, _MOCK_COUNTER
    _MOCK_CONTROL_FLAG = enable
    if enable:
        _MOCK_COUNTER = 0
        logger.info("🔑 CONTROLLED MOCK Mode ENABLED.")
    else:
        logger.info("❌ CONTROLLED MOCK Mode DISABLED.")

def retrieve_context_with_filter(
    query: str, 
    retriever: Any, 
    metadata_filter: Optional[List[str]] = None, 
) -> Dict[str, Any]:
    """Retrieves documents from the vector store, optionally filtering by document ID."""
    if retriever is None:
        return {"top_evidences": []}
    
    filter_document_ids = metadata_filter 

    try:
        docs: List[Document] = retriever.invoke(query) 
        
        # Manual Filtering
        if filter_document_ids:
            filter_set = set(filter_document_ids)
            
            filtered_docs = []
            for doc in docs:
                doc_id_in_metadata = doc.metadata.get("doc_id") 

                if doc_id_in_metadata in filter_set:
                    filtered_docs.append(doc)
            
            docs = filtered_docs
        
        # จัดรูปแบบผลลัพธ์
        top_evidences = []
        for d in docs:
            meta = d.metadata
            top_evidences.append({
                "doc_id": meta.get("doc_id"),
                "source": meta.get("source"),
                "content": d.page_content.strip()
            })
            
        return {"top_evidences": top_evidences}
        
    except Exception as e:
        logger.error(f"Error during RAG retrieval with filter: {e}")
        return {"top_evidences": []}


# =================================================================
# === EVALUATION FUNCTION (MOCK & REAL LLM) (แก้ไข JSON Robustness) ===
# =================================================================

MAX_LLM_RETRIES = 3 

def evaluate_with_llm(statement: str, context: str, standard: str) -> Dict[str, Any]:
    """
    Performs the LLM evaluation, extracting a score and reason, 
    with robust JSON parsing and retry logic.
    """
    global _MOCK_CONTROL_FLAG, _MOCK_COUNTER
    
    # 1. MOCK CONTROL LOGIC
    if _MOCK_CONTROL_FLAG:
        _MOCK_COUNTER += 1
        
        # MOCK LOGIC: ให้ Level 1-3 (9 statements) ผ่าน เพื่อให้สอดคล้องกับ highest_full_level=3
        score = 1 if _MOCK_COUNTER <= 9 else 0
        reason_text = f"MOCK: FORCED {'PASS' if score == 1 else 'FAIL'} (Statement {_MOCK_COUNTER})"
        
        # 🟢 FIX: Calculate Pass Status for MOCK
        is_pass = score >= 1
        status_th = "ผ่าน" if is_pass else "ไม่ผ่าน"

        logger.debug(f"MOCK COUNT: {_MOCK_COUNTER} | SCORE: {score} | STMT: '{statement[:20]}...'")
        
        # 🟢 FIX: Return the full data structure
        return {
            "score": score, 
            "reason": reason_text,
            "pass_status": is_pass, # ⬅️ เพิ่ม pass status
            "status_th": status_th  # ⬅️ เพิ่ม status ภาษาไทย
        }

    
    # 2. REAL LLM CALL LOGIC
    if llm_instance is None:
        logger.error("❌ LLM Instance is not initialized.")
        score = random.choice([0, 1])
        reason = f"LLM Initialization Failed (Fallback to Random Score {score})"
        
        # Fallback Logic
        is_pass = score >= 1
        status_th = "ผ่าน" if is_pass else "ไม่ผ่าน"
        
        return {
            "score": score, 
            "reason": reason,
            "pass_status": is_pass,
            "status_th": status_th
        }

    # 🟢 NEW: ใช้ PromptTemplate เพื่อสร้าง HumanMessage Content
    user_prompt_content = USER_ASSESSMENT_PROMPT.format(
        statement=statement,
        standard=standard,
        # จัดการเงื่อนไข "ไม่พบหลักฐาน" ที่นี่ ก่อนส่งให้ PromptTemplate
        context=context if context else "ไม่พบหลักฐานในเอกสารที่เกี่ยวข้อง" 
    )
    
    # === NEW RETRY LOOP ===
    for attempt in range(MAX_LLM_RETRIES):
        try:
            # A. เรียกใช้ LLM
            response = llm_instance.invoke([
                SystemMessage(content=SYSTEM_ASSESSMENT_PROMPT),
                HumanMessage(content=user_prompt_content)
            ])
            
            llm_response_content = response.content if hasattr(response, 'content') else str(response)
            
            # 🛑 B. FIX: ใช้ Regex เพื่อ Clean และค้นหา JSON Block
            
            # 1. Clean up markdown fences (```json...```) 
            cleaned_content = llm_response_content.strip()
            if cleaned_content.startswith("```json"):
                cleaned_content = cleaned_content.replace("```json", "", 1).rstrip('`')
            elif cleaned_content.startswith("```"):
                cleaned_content = cleaned_content.replace("```", "", 1).rstrip('`')
            
            # 2. ใช้ Regex เพื่อกรองหา JSON block ที่สมบูรณ์
            # NOTE: โค้ดที่ได้มามีการใช้ key 'llm_score'
            json_match = re.search(r'\{.*\}', cleaned_content, re.DOTALL)
            
            final_json_string = None
            if json_match:
                final_json_string = json_match.group(0)
            
            if not final_json_string:
                raise ValueError("LLM response did not contain a recognizable JSON block.")
                
            # 3. Parse JSON string
            llm_output = json.loads(final_json_string) # ⬅️ Parse string ที่ถูกกรองแล้ว
            
            # 4. 🟢 NEW: Validate against StatementAssessment Pydantic Schema
            # ตรวจสอบว่ามี Key 'score' หรือ 'llm_score' และ 'reason' ครบถ้วน
            raw_score = llm_output.get("llm_score") or llm_output.get("score")
            reason = llm_output.get("reason")
            
            if raw_score is not None and reason is not None:
                # สร้าง Dict ที่ใช้ Pydantic Key
                validated_data = {
                    "score": int(str(raw_score)) if str(raw_score).isdigit() else 0,
                    "reason": reason
                }
                
                # 5. ใช้ Pydantic Model Validate (ได้ score และ reason)
                validated_assessment = StatementAssessment.model_validate(validated_data)
                
                # 🟢 FIX: Calculate Pass Status based on score
                final_score = validated_assessment.score
                # Assumption: score >= 1 is a Pass
                is_pass = final_score >= 1 
                status_th = "ผ่าน" if is_pass else "ไม่ผ่าน"
                
                # 🟢 FIX: Return the complete assessment data, including calculated pass status
                result_data = validated_assessment.model_dump()
                result_data["pass_status"] = is_pass # ⬅️ เพิ่ม pass status
                result_data["status_th"] = status_th  # ⬅️ เพิ่ม status ภาษาไทย
                
                return result_data
            
            else:
                raise ValueError("LLM response JSON is missing 'score' or 'reason' keys.")

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"❌ Format/JSON Parse Failed (Attempt {attempt + 1}/{MAX_LLM_RETRIES}). Retrying in 1s... Error: {e}")
            if attempt < MAX_LLM_RETRIES - 1:
                time.sleep(1) 
                continue
            else:
                logger.error(f"❌ LLM Evaluation failed after {MAX_LLM_RETRIES} attempts. JSON/Format failure.")
                break 
        
        except Exception as e:
            logger.error(f"❌ LLM Evaluation failed due to unexpected error (Connection/Runtime). Error: {e}")
            break 

    # === FALLBACK LOGIC ===
    logger.error("❌ Using RANDOM SCORE as final fallback.")
    score = random.choice([0, 1])
    reason = f"LLM Call Failed (Fallback to Random Score {score}) after {MAX_LLM_RETRIES} attempts."
    
    # Fallback Logic
    is_pass = score >= 1
    status_th = "ผ่าน" if is_pass else "ไม่ผ่าน"
    
    # 🟢 Fallback ควรคืนค่าที่ตรงกับ StatementAssessment.model_dump() + pass status
    return {
        "score": score, 
        "reason": reason,
        "pass_status": is_pass, # ⬅️ เพิ่ม pass status
        "status_th": status_th  # ⬅️ เพิ่ม status ภาษาไทย
    }


# =================================================================
# === ACTION PLAN GENERATION UTILITY (ปรับปรุง Logic) ===
# =================================================================

def _call_llm_for_json_output(prompt: str, system_prompt: str) -> str:
    """Basic LLM call for JSON output, relies on the System Prompt to enforce JSON."""
    if llm_instance is None:
        raise ConnectionError("LLM Instance is not available for Action Plan Generation.")

    response = llm_instance.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ])
    
    llm_response_content = response.content if hasattr(response, 'content') else str(response)
    
    # Clean up markdown fences (```json...```)
    cleaned_content = llm_response_content.strip()
    if cleaned_content.startswith("```json"):
        cleaned_content = cleaned_content.replace("```json", "", 1).rstrip('`')
    elif cleaned_content.startswith("```"):
        cleaned_content = cleaned_content.replace("```", "", 1).rstrip('`')
        
    return cleaned_content.strip()

def generate_action_plan_via_llm(
    failed_statements_data: List[Dict[str, Any]], 
    sub_id: str, 
    target_level: int, 
) -> Dict[str, Any]:
    
    # 1. MOCK LOGIC
    global _MOCK_CONTROL_FLAG
    if _MOCK_CONTROL_FLAG:
        logger.warning("MOCK: Generating dummy Action Plan via MOCK Logic.")
        actions = []
        for i, data in enumerate(failed_statements_data):
            statement_id = f"L{data.get('level', target_level)} S{data.get('statement_number', i+1)}"
            failed_level = data.get('level', target_level)
            
            # NOTE: ใช้ model_dump() เพื่อสร้าง dict ที่ตรงกับ ActionPlanActions.Actions.item_type
            actions.append(ActionPlanActions.Actions.item_type( 
                Statement_ID=statement_id,
                Failed_Level=failed_level,
                Recommendation=f"MOCK: [Specific Action] จัดทำ Policy '{data.get('statement_text', 'N/A')[:20]}...' เพื่อแก้ไข GAP จากเหตุผล: {data.get('llm_reasoning', 'No reason')[:20]}...",
                Target_Evidence_Type="MOCK: Policy Document (Type: Guideline)",
                Key_Metric="Policy Approved and Published"
            ).model_dump())
        
        return ActionPlanActions(
            Phase=f"1. Strategic Gap Closure (Target L{target_level})",
            Goal=f"บรรลุหลักฐานที่จำเป็นทั้งหมดใน Level {target_level} ของ {sub_id}",
            Actions=actions
        ).model_dump()


    # --- REAL IMPLEMENTATION (LLM Call for Action Plan) ---
    try:
        # 2. Prepare Prompt Input Variables
        failed_statements_text = []
        for data in failed_statements_data:
            stmt_num = data.get('statement_number', 'N/A')
            failed_statements_text.append(f"""
            --- STATEMENT FAILED (L{data.get('level', 'N/A')} S{stmt_num}) ---
            Statement Text: {data.get('statement_text', 'N/A')}
            Reason for Failure: {data.get('llm_reasoning', 'N/A')}
            RAG Context Found: {data.get('retrieved_context', 'No context found or saved')}
            """)
            
        statements_list_str = "\n".join(failed_statements_text)

        # 3. Format the Prompt
        llm_prompt_content = ACTION_PLAN_PROMPT.format(
            sub_id=sub_id,
            target_level=target_level,
            failed_statements_list=statements_list_str 
        )
        

        # 4. Define System Prompt with JSON Schema
        schema_dict = ActionPlanActions.model_json_schema()
        
        # 🟢 ปรับปรุง: ใช้ SYSTEM_ACTION_PLAN_PROMPT และผนวก JSON Schema เข้าไป
        system_prompt_content = (
            SYSTEM_ACTION_PLAN_PROMPT + # ⬅️ ใช้ System Prompt ภาษาไทยที่ชัดเจน
            "\n\n--- REQUIRED JSON SCHEMA (STRICTLY FOLLOW) ---\n" +
            json.dumps(schema_dict, indent=2)
        )

        # 5. CALL LLM
        llm_response_json_str = _call_llm_for_json_output(
            prompt=llm_prompt_content,
            system_prompt=system_prompt_content # ⬅️ ใช้ system prompt ที่ปรับปรุงแล้ว
        )
        
        
        # 🛑 6. FIX: ใช้ Regex เพื่อค้นหา JSON Block ก่อน Parse
        
        # ตรวจสอบว่ามี String อยู่จริงหรือไม่
        if not llm_response_json_str or not llm_response_json_str.strip():
             raise ValueError("LLM returned an empty response for Action Plan.")
             
        # ใช้ Regex เพื่อค้นหา JSON Block
        json_match = re.search(r'\{.*\}', llm_response_json_str.strip(), re.DOTALL)

        cleaned_content = None
        if json_match:
            cleaned_content = json_match.group(0)

        if not cleaned_content:
            logger.error("❌ Failed to find a valid JSON block for Action Plan using Regex.")
            raise ValueError("LLM response did not contain a recognizable JSON block for Action Plan.")

        # 7. Process and Parse
        llm_result_dict = json.loads(cleaned_content) # ⬅️ ใช้ string ที่ถูกกรองแล้ว
        
        # 8. Validate and Dump
        validated_plan_model = ActionPlanActions.model_validate(llm_result_dict)

        final_action_plan_result = validated_plan_model.model_dump()
        
        return final_action_plan_result

    except Exception as e:
        logger.error(f"❌ Action Plan Generation Failed: {e}", exc_info=True)
        return {
             "Phase": "Error",
             "Goal": f"Failed to generate Action Plan for {sub_id} (Target L{target_level})",
             "Actions": [{
                "Statement_ID": "N/A", 
                "Failed_Level": target_level, 
                "Recommendation": f"System Error: {str(e)}", 
                "Target_Evidence_Type": "Check Logs", 
                "Key_Metric": "Error"
             }]
        }
    

# ----------------------------------------------------------------------
# === NARRATIVE REPORT GENERATION ===
# ----------------------------------------------------------------------

def generate_narrative_report_via_llm_real(prompt_text: str, system_instruction: str) -> str:
    """
    ฟังก์ชันเรียก LLM API จริงเพื่อสังเคราะห์รายงานเชิงบรรยายสำหรับ CEO
    
    :param prompt_text: Prompt ที่มีข้อมูล JSON ผลการประเมินอยู่ภายใน (Human Message)
    :param system_instruction: System Instruction ที่กำหนดบทบาทและรูปแบบการตอบ (มาจาก rag_prompts.py)
    :return: ข้อความรายงานเชิงบรรยายที่สร้างโดย LLM
    """
    # 🚨 ตรวจสอบการเข้าถึง LLM instance (ต้องมีใน environment จริง)
    try:
        from langchain.schema import SystemMessage, HumanMessage 
        # สมมติว่า llm ถูก import จาก models.llm
        from models.llm import llm as llm_instance 
    except ImportError:
        logger.error("❌ Cannot import LLM dependencies (SystemMessage, HumanMessage, llm_instance).")
        return "[ERROR: LLM Dependencies missing for real API call.]"
        
    if llm_instance is None:
        logger.error("❌ LLM Instance is not initialized for real API call.")
        return "[ERROR: LLM Client is not initialized for real API call.]"
        
    logger.info("Executing REAL LLM call for narrative report synthesis...")
    
    try:
        # System Message ถูกส่งเข้ามาเป็น Argument
        
        response = llm_instance.invoke([
            SystemMessage(content=system_instruction),
            HumanMessage(content=prompt_text)
        ])
        
        # 🟢 FIX: ตรวจสอบประเภท Response ก่อนเข้าถึง .content
        # เพื่อรองรับการตอบกลับที่เป็น String โดยตรง หรือ LangChain Response Object
        if hasattr(response, 'content'):
            # ถ้าเป็น LangChain/SDK Response Object
            generated_text = response.content
        elif isinstance(response, str):
            # ถ้า Wrapper คืนค่าเป็น String ตรงๆ
            generated_text = response
        else:
            # กรณีที่ไม่รู้จัก
            raise TypeError(f"LLM response type {type(response)} is not supported for content extraction.")
            
        # คืนค่าข้อความที่ได้จาก LLM
        return generated_text.strip()
        
    except Exception as e:
        logger.error(f"Real LLM API call failed during narrative report generation: {e}")
        # Fallback message
        return f"[API ERROR] Failed to generate narrative report via real LLM API: {e}"

# =================================================================
# === EVIDENCE DESCRIPTION GENERATION (NEW FUNCTION) ===
# =================================================================

def summarize_context_with_llm(context: str, sub_criteria_name: str, level: int, sub_id: str, schema: Any) -> Dict[str, str]:
    """
    ใช้ LLM เพื่อสรุปบริบทหลักฐานตามเกณฑ์และระดับที่กำหนด 
    และ Validate ผลลัพธ์ด้วย EvidenceSummary Schema.
    """
    
    # 🚨 FIX 1: จำกัดความยาว Context ก่อนส่งเข้า LLM
    MAX_LLM_SUMMARY_CONTEXT = 3000 
    
    if llm_instance is None:
        return {"summary": "เกิดข้อผิดพลาด: LLM Client ไม่พร้อมใช้งาน", "suggestion_for_next_level": "N/A"}
        
    context_to_use = context
    if len(context) > MAX_LLM_SUMMARY_CONTEXT:
        logger.warning(f"Context for summary L{level} is too long ({len(context)}), truncating to {MAX_LLM_SUMMARY_CONTEXT}.")
        context_to_use = context[:MAX_LLM_SUMMARY_CONTEXT]
        
    # 🚨 FIX 2: ปรับ System Prompt เพื่อบังคับ JSON Output ตาม EvidenceSummary
    schema_dict = EvidenceSummary.model_json_schema()
    system_prompt_content = (
        SYSTEM_EVIDENCE_DESCRIPTION_PROMPT + 
        "\n\n--- REQUIRED JSON SCHEMA (STRICTLY FOLLOW) ---\n" +
        json.dumps(schema_dict, indent=2)
    )

    try:
        # 1. จัดเตรียม Human Prompt
        human_prompt = EVIDENCE_DESCRIPTION_PROMPT.format(
            standard=sub_criteria_name,
            level=level,
            context=context_to_use,
            sub_id=sub_id # 🚨 FIX 2: เพิ่ม sub_id ในการ format
        )
        
        # 2. เรียกใช้ LLM 
        llm_response_json_str = _call_llm_for_json_output(
            prompt=human_prompt,
            system_prompt=system_prompt_content
        )
        
        # 3. ใช้ Regex ค้นหา JSON Block
        json_match = re.search(r'\{.*\}', llm_response_json_str.strip(), re.DOTALL)
        
        cleaned_content = None
        if json_match:
            cleaned_content = json_match.group(0)

        if not cleaned_content:
            logger.error("❌ Failed to find a valid JSON block for Evidence Summary using Regex.")
            raise ValueError("LLM response did not contain a recognizable JSON block for Evidence Summary.")

        # 4. Process and Parse
        llm_result_dict = json.loads(cleaned_content) 
        
        # 5. Validate against EvidenceSummary Schema
        validated_summary_model = EvidenceSummary.model_validate(llm_result_dict)
        
        logger.info(f"✅ Generated Evidence Summary for {sub_criteria_name} L{level}")
        
        # 6. Return the validated Dict
        return validated_summary_model.model_dump()
        
    except Exception as e:
        logger.error(f"LLM Summary generation failed: {e}", exc_info=True)
        return {"summary": "ข้อผิดพลาดในการสรุปข้อมูลโดย LLM", "suggestion_for_next_level": f"เกิดข้อผิดพลาดในการเรียก LLM: {str(e)}"}
    
def generate_evidence_description_via_llm(*args, **kwargs) -> str:
    """
    Deprecated: ฟังก์ชันนี้ถูกแทนที่ด้วย summarize_context_with_llm เพื่อให้รองรับ Pydantic Schema
    """
    logger.warning("generate_evidence_description_via_llm is deprecated. Using summarize_context_with_llm instead.")
    # Fallback/Error message based on the old usage pattern (if accidentally called)
    return "เกิดข้อผิดพลาด: ฟังก์ชันนี้ถูกยกเลิกการใช้งาน โปรดใช้ summarize_context_with_llm แทน"