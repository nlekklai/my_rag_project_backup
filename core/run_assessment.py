import os
import sys
import logging
import argparse
import random
import json
import time
from typing import List, Dict, Any, Optional, Union, Tuple
from unittest.mock import patch

# -------------------- PATH SETUP --------------------
# 🚨 NOTE: run_assessment.py อยู่ใน core/ ดังนั้น project_root คือ directory แม่
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# -------------------- Global Vars --------------------
try:
    from config.global_vars import (
        EVIDENCE_DOC_TYPES
    )
except ImportError as e:
    print(f"FATAL ERROR: Cannot import global_vars: {e}", file=sys.stderr)
    sys.exit(1)


# -------------------- Core & Assessment Imports --------------------
try:
    from assessments.enabler_assessment import EnablerAssessment
    from core.retrieval_utils import set_mock_control_mode
    from core.vectorstore import load_all_vectorstores

    # --- Mocking functions ---
    from assessments.mocking_assessment import (
        summarize_context_with_llm_MOCK,
        create_structured_action_plan_MOCK,
        retrieve_context_MOCK,
        evaluate_with_llm_CONTROLLED_MOCK,
    )

except ImportError as e:
    print(f"FATAL ERROR: Failed to import required modules: {e}", file=sys.stderr)
    sys.exit(1)

# -------------------- Logging --------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# -------------------- MOCK COUNTER --------------------
_MOCK_EVALUATION_COUNTER = 0

def get_default_assessment_file_paths(enabler_abbr: str) -> Dict[str, str]:
    """สร้างพาธไฟล์ตั้งค่าที่ใช้ในปัจจุบันสำหรับ Enabler."""
    
    # 🟢 [FIXED PATH]: ใช้ project_root และ evidence_checklist โดยตรง
    BASE_DIR = os.path.abspath(os.path.join(project_root, "evidence_checklist"))
    
    return {
        "evidence_file_path": os.path.join(BASE_DIR, f"{enabler_abbr.lower()}_evidence_statements_checklist.json"),
        "rubric_file_path": os.path.join(BASE_DIR, f"{enabler_abbr.lower()}_rating_criteria_rubric.json"),
        "level_fractions_file_path": os.path.join(BASE_DIR, f"{enabler_abbr.lower()}_scoring_level_fractions.json"),
        # ใช้ mapping_new.json เป็นค่า default
        "mapping_file_path": os.path.join(BASE_DIR, f"{enabler_abbr.lower()}_evidence_mapping_pom.json"), 
    }

# -------------------- SUB CRITERIA UTILITIES & ACTION PLAN --------------------
def get_sub_criteria_data(sub_id: str, criteria_list: List[Dict]) -> Dict:
    """Finds the sub-criteria dictionary from the full list."""
    for criteria in criteria_list:
        if criteria.get('Sub_Criteria_ID') == sub_id:
            return criteria
    return {}


# -------------------- L5 SUMMARY --------------------
def generate_and_integrate_l5_summary(assessor: EnablerAssessment, results: Dict) -> Dict:
    """
    Generate L5 Summary และ Highest Full Level Summary 
    (จะเรียก generate_evidence_summary_for_level และ summarize_context_with_llm ซึ่งจะถูก Patch)
    """
    updated_breakdown = {}
    sub_criteria_breakdown = results.get("SubCriteria_Breakdown", {})
    for sub_id, sub_data in sub_criteria_breakdown.items():
        try:
            if isinstance(sub_data, str):
                sub_data = {"name": sub_data}
            
            # 🟢 NEW: ดึง Highest Full Level
            highest_full_level = sub_data.get('highest_full_level', 0)
            
            # 1. --- Generate L5 Summary (ยังคงทำเหมือนเดิม) ---
            try:
                # 🛑 ใช้ generate_evidence_summary_for_level() ของ EnablerAssessment สำหรับ Level 5
                l5_context_info = assessor.generate_evidence_summary_for_level(sub_id, 5)
            except Exception:
                l5_context_info = None
            
            # 🛑 การจัดการผลลัพธ์ L5 (ตาม logic เดิม)
            l5_summary_result = l5_context_info

            if isinstance(l5_summary_result, str):
                if "ไม่พบหลักฐาน" in l5_summary_result:
                    l5_summary_result = {"summary": l5_summary_result, "suggestion_for_next_level": "N/A (No Evidence Found)"}
                else:
                    l5_summary_result = {"summary": l5_summary_result, "suggestion_for_next_level": "N/A"}
            
            if not l5_summary_result.get("summary", "").strip():
                l5_summary_result = {"summary": "ไม่พบหลักฐานที่เกี่ยวข้องสำหรับ Level 5 ในฐานข้อมูล RAG.", "suggestion_for_next_level": "N/A"}

            # ผสานรวม L5 Summary
            sub_data["evidence_summary_L5"] = l5_summary_result


            # 2. --- Generate Highest Full Level Summary (NEW LOGIC) ---
            highest_summary_result = {"summary": "N/A (Level 0 or Error)", "suggestion_for_next_level": "N/A"}
            
            if highest_full_level > 0 and highest_full_level <= 5:
                 if highest_full_level == 5:
                     # ถ้า Highest Full Level คือ 5 ให้ใช้ผลลัพธ์เดียวกับ L5 (ลดการเรียก LLM ซ้ำ)
                     highest_summary_result = l5_summary_result
                 else:
                     try:
                        # 🛑 ใช้ generate_evidence_summary_for_level() ของ EnablerAssessment สำหรับ Level ที่ผ่านสูงสุด
                        highest_context_info = assessor.generate_evidence_summary_for_level(sub_id, highest_full_level)
                     except Exception:
                         highest_context_info = None

                     # 🛑 การจัดการผลลัพธ์ Highest Level
                     highest_summary_result = highest_context_info

                     if isinstance(highest_summary_result, str):
                        if "ไม่พบหลักฐาน" in highest_summary_result:
                            highest_summary_result = {"summary": highest_summary_result, "suggestion_for_next_level": "N/A (No Evidence Found)"}
                        else:
                            highest_summary_result = {"summary": highest_summary_result, "suggestion_for_next_level": "N/A"}
                    
                     if not highest_summary_result.get("summary", "").strip():
                         highest_summary_result = {"summary": f"ไม่พบหลักฐานที่เกี่ยวข้องสำหรับ Level {highest_full_level} ในฐานข้อมูล RAG.", "suggestion_for_next_level": "N/A"}

            # 🟢 ผสานรวม Highest Full Level Summary โดยใช้คีย์ที่สร้างแบบ Dynamic
            highest_summary_key = f"evidence_summary_L{highest_full_level}"
            sub_data[highest_summary_key] = highest_summary_result

            updated_breakdown[sub_id] = sub_data
            
        except Exception as e_outer:
            # Handle error gracefully
            sub_data["evidence_summary_L5"] = {"summary": f"Error: {e_outer}", "suggestion_for_next_level": "N/A"}
            # Ensure the highest_summary key is also updated in case of error
            highest_summary_key = f"evidence_summary_L{sub_data.get('highest_full_level', 0)}"
            sub_data[highest_summary_key] = {"summary": f"Error: {e_outer} (Highest Level Summary)", "suggestion_for_next_level": "N/A"}
            updated_breakdown[sub_id] = sub_data
            logger.error(f"[ERROR] during summary integration for {sub_id}: {e_outer}", exc_info=True)
            
    results["SubCriteria_Breakdown"] = updated_breakdown
    return results

# -------------------- EXPORT UTILITIES (NEW) --------------------
# 🟢 NEW: JSON Serializer Helper (แก้ไขเพื่อแก้ TypeError: Object is not JSON serializable)
def _json_default_serializer(obj: Any) -> Dict[str, Any]:
    """
    Default serializer สำหรับ json.dump() เพื่อแปลง Custom Objects 
    (เช่น ActionPlanActions) ให้เป็น Dictionary ที่ JSON เข้าใจ
    """
    if hasattr(obj, '__dict__'):
        try:
            # ใช้ model_dump() ถ้าเป็น Pydantic V2/Dataclass
            if hasattr(obj, 'model_dump'):
                return obj.model_dump()
        except TypeError:
             pass 
        
        # fallback เป็น __dict__
        return obj.__dict__
    
    # ถ้ายังหาไม่ได้ ให้ raise TypeError เพื่อใช้ default behavior ของ json.dump
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

def _export_results_to_json(
    summary: Dict[str, Any], 
    enabler_type: str, 
    sub_id: Optional[str] = None
) -> Dict[str, Optional[str]]:
    """
    จัดการการบันทึกไฟล์สรุปและข้อมูลดิบ โดยใช้ enabler_type และ sub_id 
    (ถ้ามี) ในการตั้งชื่อไฟล์ และคืนค่าพาธที่ใช้
    """
    export_paths = {
        'export_path_used': None,
        'raw_export_path_used': None
    }
    
    try:
        # 1. เตรียม Export Directory และ Timestamp
        scope_prefix = f"_{sub_id}" if sub_id and sub_id != "all" else "_all"
        
        export_dir = os.path.abspath(os.path.join(project_root, "exports"))
        os.makedirs(export_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 2. Export Summary Report (ไม่รวม raw_llm_results)
        summary_to_save = {k: v for k, v in summary.items() if k != 'raw_llm_results'}
        
        summary_filename = f"{enabler_type}_summary{scope_prefix}_{timestamp}.json" 
        summary_file_path = os.path.join(export_dir, summary_filename)
        
        with open(summary_file_path, 'w', encoding='utf-8') as f:
            # 🛑 แก้ไข: เพิ่ม default=_json_default_serializer 
            json.dump(summary_to_save, f, ensure_ascii=False, indent=4, default=_json_default_serializer)
        export_paths['export_path_used'] = summary_file_path
        logger.info(f"Report successfully exported to: {summary_file_path}")
        
        # 3. Export Raw Evaluation Data
        raw_data = summary.get('raw_llm_results', [])
        if raw_data:
            raw_filename = f"{enabler_type}_raw_details{scope_prefix}_{timestamp}.json" 
            raw_file_path = os.path.join(export_dir, raw_filename)
            
            with open(raw_file_path, 'w', encoding='utf-8') as f:
                # 🛑 แก้ไข: เพิ่ม default=_json_default_serializer 
                json.dump(raw_data, f, ensure_ascii=False, indent=4, default=_json_default_serializer)
            export_paths['raw_export_path_used'] = raw_file_path
            logger.info(f"Raw evaluation data successfully exported to: {raw_file_path}")
            
    except Exception as e:
        # 🛑 แก้ไข Log Message ให้ชัดเจน
        logger.error(f"❌ ERROR during file export: Object of type {e.__class__.__name__} is not JSON serializable (check ActionPlanActions or other custom objects)", exc_info=True)
        
    return export_paths

# -------------------- MAIN ASSESSMENT --------------------
def run_assessment_process(
    enabler: str,
    sub_criteria_id: str,
    mode: str = "real",
    filter_mode: bool = False,
    export: bool = False,
    disable_semantic_filter: bool = False,
    allow_fallback: bool = False,
    external_retriever: Optional[Any] = None  # 🟢 ใช้ retriever จากภายนอก (FastAPI)
) -> Tuple[Dict[str, Any], EnablerAssessment]: # 🛑 เปลี่ยน Return Type เป็น Tuple
    start_time_global = time.perf_counter()
    summary: Dict[str, Any] = {'raw_export_path_used': None}
    original_mode = mode
    retriever = external_retriever
    assessment_engine: Optional[EnablerAssessment] = None # 🟢 กำหนดค่าเริ่มต้น

    # -------------------- Mock Setup --------------------
    set_mock_control_mode(original_mode == "mock")
    llm_eval_func = evaluate_with_llm_CONTROLLED_MOCK if original_mode == "mock" else None
    llm_summarize_func = summarize_context_with_llm_MOCK if original_mode == "mock" else None
    llm_action_plan_func = create_structured_action_plan_MOCK if original_mode == "mock" else None

    file_paths = get_default_assessment_file_paths(enabler)
    
    # 🟢 [NEW] Data container สำหรับเก็บข้อมูลที่โหลดจาก temp_loader
    evidence_data_from_loader = []
    evidence_mapping_data = {}


    # -------------------- Load Vectorstore & Mapping --------------------
    try:
        if mode == "real" and external_retriever is None:
            logger.warning("⚠️ Running in REAL mode without external retriever. Loading vector store inside function (slow).")
            
            # 🟢 [FIX 1]: สร้าง temp_loader โดยส่ง **file_paths ทั้งหมด
            temp_loader = EnablerAssessment(enabler_abbr=enabler, 
                                            vectorstore_retriever=None, 
                                            **file_paths) # 💡 ส่งพาธทั้งหมดเข้าไป
            
            # 💡 ดึง Evidence Data และ Mapping Data ที่โหลดมาแล้ว
            evidence_data_from_loader = temp_loader.evidence_data or []
            evidence_mapping_data = temp_loader.evidence_mapping_data or {}
            
            # 💡 [DEBUG CHECK]: Log การโหลด Evidence
            if not evidence_data_from_loader:
                 logger.critical("❌ CRITICAL: temp_loader failed to load Evidence (Count: 0). Proceeding with vectorstore load only.")
            else:
                 logger.info(f"✅ temp_loader loaded Evidence successfully (Count: {len(evidence_data_from_loader)}).")

            # Filter document IDs
            file_ids_to_load = []
            if filter_mode and sub_criteria_id != "all":
                if not evidence_mapping_data:
                     logger.warning("⚠️ Warning: Mapping data is empty. Cannot apply document filtering.")
                
                for key, data in evidence_mapping_data.items():
                    if key.startswith(f"{sub_criteria_id}_L"):
                        # ใช้ EVIDENCE_DOC_TYPES เพื่อระบุคีย์
                        for ev in data.get(EVIDENCE_DOC_TYPES, []): 
                            doc_uuid = ev.get('stable_doc_uuid', ev.get('doc_id'))
                            if doc_uuid:
                                file_ids_to_load.append(doc_uuid)
                logger.info(f"DEBUG: doc_ids to load for {sub_criteria_id}: {len(file_ids_to_load)} unique documents.")


            retriever = load_all_vectorstores(
                doc_types=[EVIDENCE_DOC_TYPES],
                evidence_enabler=enabler.lower(),
                doc_ids=file_ids_to_load if file_ids_to_load else None
            )
            logger.info(f"✅ Vectorstore loaded for enabler {enabler}")

        elif mode == "real" and external_retriever is not None:
            logger.info("✅ Using external retriever provided by API/Caller.")

        elif mode != "real":
            retriever = None

    except Exception as e:
        logger.error(f"❌ ERROR: Failed to load Vectorstores: {e}", exc_info=True)
        mode = "random"
        logger.warning(f"⚠️ MODE CHANGED TO: {mode.upper()} due to Vectorstore Load Failure.")
        # 🛑 ถ้าโหลดไม่สำเร็จ ให้สร้าง Engine ที่ไม่สมบูรณ์เพื่อคืนค่า (ถ้าจำเป็น)
        assessment_engine = EnablerAssessment(enabler_abbr=enabler, **file_paths)


    # -------------------- Load & Filter Evidence --------------------
    try:
        if 'temp_loader' in locals() and mode == "real" and external_retriever is None:
            # 💡 ใช้ข้อมูลที่โหลดมาแล้วจาก temp_loader
            filtered_evidence = evidence_data_from_loader 
            enabler_loader = temp_loader # ใช้ temp_loader เป็น loader หลัก
        else:
            # 🛑 ถ้าไม่ใช้ temp_loader ให้สร้าง loader ใหม่และใช้ **file_paths
            enabler_loader = EnablerAssessment(enabler_abbr=enabler, 
                                                vectorstore_retriever=retriever,
                                                **file_paths)
            filtered_evidence = enabler_loader.evidence_data

        # 🟢 [DEBUG FIX] Log ตรวจสอบจำนวน Evidence ที่โหลดมา
        logger.info(f"DEBUG: Initial Evidence loaded (enabler_loader.evidence_data count): {len(filtered_evidence)}")


        if sub_criteria_id != "all":
            # 1. กรองตาม Sub-Criteria ID
            initial_count = len(filtered_evidence)
            filtered_evidence = [e for e in filtered_evidence if e.get("Sub_Criteria_ID") == sub_criteria_id]
            
            # 2. กรองตาม Mapping (ถ้าเปิด filter_mode)
            if filter_mode:
                evidence_mapping_data_local = enabler_loader.evidence_mapping_data
                valid_level_keys = {k for k, v in evidence_mapping_data_local.items() if k.startswith(sub_criteria_id) and v.get(EVIDENCE_DOC_TYPES)}
                statements_to_assess = []
                skipped_statements = 0

                for statement_dict in filtered_evidence:
                    added = False
                    for lvl in range(1, 6):
                        level_key = f"{statement_dict['Sub_Criteria_ID']}_L{lvl}"
                        # ตรวจสอบว่า statement นั้นอยู่ใน mapping หรือไม่ (มีการระบุเอกสารหลักฐาน)
                        if level_key in valid_level_keys and statement_dict.get(f"Level_{lvl}_Statements"):
                            statements_to_assess.append(statement_dict)
                            added = True
                            break
                    if not added:
                        skipped_statements += 1

                filtered_evidence = statements_to_assess
            else:
                skipped_statements = initial_count - len(filtered_evidence)
                
            logger.info(f"DEBUG: Statements after Strict Filter: {len(filtered_evidence)} (Skipped: {skipped_statements})")

    except Exception as e:
        # 🛑 ถ้าเกิด Error ในขั้นตอนนี้ และ assessment_engine ยังไม่ถูกสร้าง
        if assessment_engine is None:
            assessment_engine = EnablerAssessment(enabler_abbr=enabler, **file_paths)
        summary.update(assessment_engine.summarize_results())
        summary['Error'] = str(e)
        summary['mode_used'] = mode
        return summary, assessment_engine # 🛑 คืนค่า Engine ด้วย


    # -------------------- Create Assessment Engine (ใช้ข้อมูลที่โหลด/กรองแล้ว) --------------------
    # 💡 ใช้ filtered_evidence และ data จาก enabler_loader
    assessment_engine = EnablerAssessment(
        enabler_abbr=enabler,
        evidence_data=filtered_evidence,
        rubric_data=enabler_loader.rubric_data,
        level_fractions=enabler_loader.level_fractions,
        evidence_mapping_data=enabler_loader.evidence_mapping_data,
        vectorstore_retriever=retriever,
        use_mapping_filter=filter_mode,
        target_sub_id=sub_criteria_id if sub_criteria_id != "all" else None,
        mock_llm_eval_func=llm_eval_func,
        mock_llm_summarize_func=llm_summarize_func,
        mock_llm_action_plan_func=llm_action_plan_func,
        disable_semantic_filter=disable_semantic_filter,
        **file_paths # 💡 ส่งพาธไฟล์ทั้งหมด
    )

    if original_mode == "mock":
        assessment_engine._retrieve_context = lambda **kwargs: retrieve_context_MOCK(
            statement=kwargs.get('query'),
            sub_criteria_id=kwargs['sub_criteria_id'],
            level=kwargs.get('level'),
            statement_number=kwargs.get('statement_number', 0),
        )

    # -------------------- Run Assessment --------------------
    try:
        if not filtered_evidence:
            logger.warning(f"⚠️ No evidence statements found for Enabler: {enabler} and Sub-Criteria: {sub_criteria_id}. Skipping assessment.")
        else:
            assessment_engine.run_assessment(filtered_evidence) # ส่ง filtered_evidence เข้าไป

        
        # 🟢 NEW: ประมวลผลสถานะการผ่าน/ไม่ผ่านและดึง UUIDs
        assessment_engine._add_pass_status_and_extract_uuids()
        
        # 🟢 NEW: ดึงข้อมูล Source เต็มจาก UUIDs
        assessment_engine._retrieve_full_source_info_and_update()
        
        summary = assessment_engine.summarize_results()
        summary['raw_llm_results'] = assessment_engine.raw_llm_results

        logger.info(f"[DEBUG] raw_llm_results count = {len(summary.get('raw_llm_results', []))}")

        # # 🟢 Generate & Integrate L5 Summary
        summary = assessment_engine._integrate_evidence_summaries(summary)

        # 🟢 Generate & Integrate Action Plans
        action_plans: Dict[str, Any] = {}
        for sub_id, summary_data in summary.get('SubCriteria_Breakdown', {}).items():
            try:
                action_plan = assessment_engine.generate_action_plan_sub(sub_id, enabler, summary_data, summary)
                if action_plan:
                    action_plans[sub_id] = action_plan
            except Exception as e:
                logger.error(f"[ERROR] Action Plan failed for {sub_id}: {e}", exc_info=True)
        logger.info(f"[DEBUG] Action Plan Keys: {list(action_plans.keys())}")

        summary['Action_Plans'] = action_plans

        logger.info(f"[ACTION PLAN READY] Generated {len(action_plans)} plans.")

    except Exception as e:
        logger.exception("[ERROR] run_assessment_process failed.")
        summary.update(assessment_engine.summarize_results())
        summary['Error_Details'] = str(e)

    # -------------------- Export Option --------------------
    if export:
        export_paths = _export_results_to_json(summary, enabler, sub_criteria_id)
        summary.update(export_paths)

    end_time = time.perf_counter()
    summary['runtime_seconds'] = round(end_time - start_time_global, 2)
    summary['mode_used'] = mode

    logger.info(f"✅ run_assessment_process finished in {summary['runtime_seconds']}s (mode={mode})")
    
    # 🛑 คืนค่า Engine ด้วย
    return summary, assessment_engine 


# -------------------- CLI Entry Point (Adapter) --------------------
if __name__ == "__main__":  
    try:
        # 1. ⚙️ การตั้งค่า Argument Parser
        parser = argparse.ArgumentParser(description="Automated Enabler Maturity Assessment System.")
        parser.add_argument("--mode", 
                            choices=["mock", "random", "real"], 
                            default="real", # 💡 เปลี่ยน default เป็น real
                            help="Assessment mode: 'mock', 'random', or 'real'.")
                            
        parser.add_argument("--enabler", 
                            type=str, 
                            default="KM",
                            choices=["CG", "L", "SP", "RM&IC", "SCM", "DT", "HCM", "KM", "IM", "IA"],
                            help="The core business enabler abbreviation (e.g., 'KM', 'SCM').")
                            
        parser.add_argument("--sub", 
                            type=str, 
                            default="all",
                            help="Filter to a specific Sub-Criteria ID (e.g., '1.1'). Default is 'all'.")
        
        parser.add_argument("--filter", 
                            action="store_true", 
                            help="Enable metadata filtering based on the KM mapping file (Strict Filter Mode).")
        
        parser.add_argument("--export", 
                            action="store_true",
                            help="Export the final summary results to a JSON file.")

        parser.add_argument("--disable-semantic-filter",
                            action="store_true",
                            help="Disable semantic reranking / semantic filter in RAG (for debugging retrieval).")
        
        parser.add_argument("--allow-fallback",
                            action="store_true",
                            help="Allow assessment to fallback to a random/mock mode if a fatal error occurs during RAG/LLM.")
        
        args = parser.parse_args()
        
        
        # 2. 🚀 CLI Call: เรียกใช้ฟังก์ชัน run_assessment_process และรับค่า 2 ตัว
        final_results, assessment_engine = run_assessment_process( # 🛑 รับค่า 2 ตัว
            enabler=args.enabler,
            sub_criteria_id=args.sub,
            mode=args.mode, 
            filter_mode=args.filter,
            export=args.export,
            disable_semantic_filter=args.disable_semantic_filter,
            allow_fallback=args.allow_fallback, 
            external_retriever=None
        )

        enabler_abbr = args.enabler
        target_sub_id_for_print = args.sub
        mode_used_for_report = final_results.get('mode_used', args.mode)

        # -------------------- Output Summary for CLI --------------------
        error_details = final_results.get('Error_Details')
        
        if error_details:
            # ❌ แสดง Error Details ก่อน Summary
            print(f"\n❌ FATAL ERROR: Assessment failed during execution: {error_details}", file=sys.stderr)
            
        
        summary = final_results
        overall_data = summary.get('Overall', {})
        sub_breakdown = summary.get('SubCriteria_Breakdown', {})

        # (โค้ดการแสดงผล Summary)
        print("\n=====================================================")
        print(f"      SUMMARY OF SCORING RESULTS ({mode_used_for_report.upper()} MODE) ")
        print(f"      ENABLER: {enabler_abbr.upper()}")
        print(f"      SCOPE: {target_sub_id_for_print.upper()}")
        print("=====================================================")
            
        if overall_data or sub_breakdown: 
            avg_score = overall_data.get('overall_maturity_score')
            avg_score_text = f"{avg_score:.3f}" if isinstance(avg_score, (int, float)) else "0.000"
            
            print(f"Overall Maturity Score (Avg.): {avg_score_text}")
            print(f"Overall Maturity Level (Weighted): {overall_data.get('overall_maturity_level', 'N/A')}")
            print(f"Number of Sub-Criteria Assessed: {overall_data.get('num_sub_criteria', len(sub_breakdown) if sub_breakdown else 'N/A')}")
            
        print("\n--- Sub-Criteria Breakdown ---")
        if sub_breakdown:
            for sub_id, data in sub_breakdown.items():
                highest_full = data.get('highest_full_level', 0)
                score = data.get('weighted_score', 0.0)
                print(f"  {sub_id} (Score: {score:.3f}, Highest Full Lvl: L{highest_full}) - {data.get('name', 'N/A')}")
        else:
            # 🟢 แสดงผลในกรณีที่ Sub-Criteria Breakdown ไม่มีข้อมูล
            print("No Sub-Criteria results available.")
            
        # -------------------- Print Detailed Results --------------------
        # 🛑 เรียกเมธอดของ Engine โดยตรง
        if args.sub != "all" and assessment_engine and assessment_engine.raw_llm_results:
            assessment_engine.print_detailed_results(
                target_sub_id=target_sub_id_for_print
            )

        # -------------------- Print Export Path --------------------
        export_path = final_results.get('export_path_used')
        raw_export_path = final_results.get('raw_export_path_used')

        if args.export and export_path:
            print(f"\n✨ Assessment Report Saved: {export_path}")
            if raw_export_path:
                print(f"✨ Raw Details Saved: {raw_export_path}")

        # -------------------- Exit on Fatal Error --------------------
        # 🚨 FIX: ถ้าเกิด Fatal Error ต้อง Exit(1)
        if error_details:
            print("\n(Script exiting due to Fatal Error.)")
            sys.exit(1)

        
    except Exception as e:
        logger.error(f"FATAL ERROR in CLI execution: {e}", exc_info=True)
        sys.exit(1)