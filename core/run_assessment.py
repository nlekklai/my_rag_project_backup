import os
import json
import logging
import sys
import argparse
import random
from typing import List, Dict, Any, Optional, Union
import time
import re
from unittest.mock import patch

# --- PATH SETUP (Must be executed first for imports to work) ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    from assessments.enabler_assessment import EnablerAssessment 
    import core.retrieval_utils 
    from core.retrieval_utils import set_mock_control_mode
    from core.vectorstore import load_all_vectorstores 

    # 🛑 FIX: Import Mock Functions ทั้งหมด
    from assessments.mocking_assessment import (
        summarize_context_with_llm_MOCK,
        generate_action_plan_MOCK,
        retrieve_context_MOCK,
        evaluate_with_llm_CONTROLLED_MOCK,
    )
    from core.assessment_schema import EvidenceSummary, StatementAssessment
    from core.action_plan_schema import ActionPlanActions 
    
except ImportError as e:
    print(f"FATAL ERROR: Failed to import required modules. Check sys.path and file structure. Error: {e}", file=sys.stderr)
    sys.exit(1)

# 1. Setup Logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -------------------- MOCK COUNTER --------------------
_MOCK_EVALUATION_COUNTER = 0

# -------------------- DETAILED RESULTS --------------------
def print_detailed_results(raw_llm_results: List[Dict], target_sub_id: str):
    """แสดงผลลัพธ์การประเมิน LLM ราย Statement พร้อม Source File"""
    if not raw_llm_results:
        logger.info("\n[Detailed Results] No raw LLM results found to display.")
        return

    grouped: Dict[str, Dict[int, List[Dict]]] = {}
    for r in raw_llm_results:
        sub_id = r.get('sub_criteria_id')
        level = r.get('level')
        if sub_id and level is not None:
             grouped.setdefault(sub_id, {}).setdefault(level, []).append(r)
        
    for sub_id in sorted(grouped.keys()):
        if target_sub_id != "all" and sub_id != target_sub_id:
             continue
        
        print(f"\n--- Sub-Criteria ID: {sub_id} ---")
        for level in sorted(grouped[sub_id].keys()):
            level_results = grouped[sub_id][level]
            total_statements = len(level_results)
            passed_statements = sum(r.get('llm_score', r.get('score', 0)) for r in level_results)
            pass_ratio = passed_statements / total_statements if total_statements > 0 else 0.0
            print(f"  > Level {level} ({passed_statements}/{total_statements}, Pass Ratio: {pass_ratio:.3f})")
            for r in level_results:
                llm_score = r.get('llm_score', r.get('score', 0))
                score_text = f"✅ PASS | Score: {llm_score}" if llm_score == 1 else f"❌ FAIL | Score: {llm_score}"
                fail_status = "" if llm_score == 1 else "❌ FAIL |"
                statement_num = r.get('statement_number', 'N/A')
                print(f"    - S{statement_num}: {fail_status} Statement: {r.get('statement', 'N/A')[:100]}...")
                print(f"      [Score]: {score_text}")
                print(f"      [Reason]: {r.get('reason', 'N/A')[:120]}...")
                sources = r.get('retrieved_sources_list', [])
                if sources:
                     print("      [SOURCE FILES]:")
                     for src in sources:
                         source_name = src.get('source_name', 'Unknown File')
                         location = src.get('location', 'N/A')
                         print(f"        -> {source_name} (Location: {location})")
                print(f"      [Context]: {r.get('context_retrieved_snippet', 'N/A')}")

# -------------------- PASS STATUS --------------------
def add_pass_status_to_raw_results(raw_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    updated_results = []
    for item in raw_results:
        llm_res = item.get('llm_result', {})
        score = item.get('llm_score') or item.get('score')
        passed = False
        is_passed_from_sub = llm_res.get('is_passed')
        
        if isinstance(is_passed_from_sub, bool):
            passed = is_passed_from_sub
        elif score is not None:
            try:
                if int(score) >= 1:
                    passed = True
            except (ValueError, TypeError):
                pass
                
        item['pass_status'] = passed
        item['status_th'] = "ผ่าน" if passed else "ไม่ผ่าน"
        item['sub_criteria_id'] = item.get('sub_criteria_id', 'N/A')
        item['level'] = item.get('level', 0)
        item['statement_id'] = item.get('statement_id', 'N/A')
        updated_results.append(item)
    return updated_results

# -------------------- SUB CRITERIA UTILITIES & ACTION PLAN --------------------
def get_sub_criteria_data(sub_id: str, criteria_list: List[Dict]) -> Dict:
    for criteria in criteria_list:
        if criteria.get('Sub_Criteria_ID') == sub_id:
            return criteria
    return {}

def get_all_failed_statements(summary: Dict) -> List[Dict[str, Any]]:
    all_failed = []
    raw_results = summary.get('raw_llm_results', []) 
    for r in raw_results:
        score_val = r.get('llm_score', r.get('score', 1))
        try:
            if int(score_val) == 0:
                all_failed.append({
                    "sub_id": r.get('sub_criteria_id', 'N/A'),
                    "level": r.get('level', 0),
                    "statement_number": r.get('statement_number', 0),
                    "statement_text": r.get('statement', 'N/A'),
                    "llm_reasoning": r.get('reason', 'No reason saved'),
                    "retrieved_context": r.get('context_retrieved_snippet', 'No context saved') 
                })
        except (ValueError, TypeError):
            pass
    return all_failed

def generate_action_plan_for_sub(sub_id: str, summary_data: Dict, full_summary: Dict) -> List[Dict]:
    """Generate Action Plan per Sub-Criteria. (จะใช้ generate_action_plan_via_llm ซึ่งจะถูก Patch)"""

    highest_full_level = summary_data.get('highest_full_level', 0)
    target_level = highest_full_level + 1

    if not summary_data.get('development_gap', False):
        return [{
            "Phase": "No Action Needed",
            "Goal": f"Sub-Criteria {sub_id} ผ่านครบถึง Level {highest_full_level}",
            "Actions": []
        }]

    if target_level > 5:
        return [{
            "Phase": "L5 Maturity Maintenance",
            "Goal": "มุ่งเน้นการรักษาความสม่ำเสมอและการปรับปรุงอย่างต่อเนื่องในระดับ 5",
            "Actions": []
        }]

    all_failed_statements = get_all_failed_statements(full_summary)

    failed_statements_for_sub = [
        s for s in all_failed_statements
        if s['sub_id'] == sub_id
    ]

    action_plan = []

    if not failed_statements_for_sub:
        llm_action_plan_result = {
            "Phase": "1. General Evidence Collection",
            "Goal": f"รวบรวมหลักฐานเพิ่มเติมเพื่อผ่านเกณฑ์ Level {target_level}",
            "Actions": [{
                "Statement_ID": f"ALL_L{target_level}",
                "Failed_Level": target_level,
                "Recommendation": (
                    f"ไม่พบ Statement ที่ล้มเหลวใน Raw Data สำหรับ L{target_level}. "
                    f"โปรดตรวจสอบหลักฐานทั่วไปของ Level นี้อีกครั้ง."
                ),
                "Target_Evidence_Type": "Policy, Record, Report",
                "Key_Metric": "Pass Rate 100% on Rerunning Assessment"
            }]
        }

    else:
        try:
            # 🛑 ใช้ LLM จริง หรือ Mock (ตามที่ถูก Patch ใน run_assessment_process และ Logic ใน retrieval_utils)
            llm_action_plan_result = core.retrieval_utils.generate_action_plan_via_llm(
                failed_statements_data=failed_statements_for_sub,
                sub_id=sub_id,
                target_level=target_level
            )
        except Exception as e:
            logger.error(f"[ERROR] Failed to generate Action Plan via LLM for {sub_id}: {e}", exc_info=True)
            llm_action_plan_result = {
                "Phase": "Error - LLM Response Issue",
                "Goal": f"Failed to generate Action Plan for {sub_id} (Target L{target_level})",
                "Actions": [{
                    "Statement_ID": "LLM_ERROR",
                    "Failed_Level": target_level,
                    "Recommendation": (
                        f"System Error: {str(e)}. "
                        f"Manual Action Required: รวบรวมหลักฐานใหม่สำหรับ Level {target_level}."
                    ),
                    "Target_Evidence_Type": "System Check / Manual Collection",
                    "Key_Metric": "Error Fix"
                }]
            }

    if llm_action_plan_result and 'Actions' in llm_action_plan_result:
        action_plan.append(llm_action_plan_result)

    failed_levels_with_gap = [
        lvl for lvl, ratio in summary_data.get('pass_ratios', {}).items() if ratio < 1.0
    ]
    recommend_action_text = f"รวบรวมหลักฐานใหม่สำหรับ Level {target_level} และนำเข้า Vector Store"

    action_plan.append({
        "Phase": "2. AI Validation & Maintenance",
        "Goal": f"ยืนยัน Level-Up และรักษาความต่อเนื่องของหลักฐานใน L{target_level}",
        "Actions": [{
            "Statement_ID": f"ALL_L{target_level}",
            "Failed_Level": target_level,
            "Recommendation": f"{recommend_action_text} จากนั้นรัน AI Assessment อีกครั้งเพื่อยืนยันผล",
            "Target_Evidence_Type": "Rerunning Assessment & New Evidence",
            "Key_Metric": f"Overall Score ของ {sub_id} ต้องเพิ่มขึ้นและ Highest Full Level เป็น L{target_level}"
        }]
    })

    logger.info(
        f"[ACTION PLAN READY] {sub_id} → {len(failed_statements_for_sub)} failed statements "
        f"at levels: {[s['level'] for s in failed_statements_for_sub]}"
    )

    return action_plan

# -------------------- L5 SUMMARY --------------------
def generate_and_integrate_l5_summary(assessor, results):
    """
    Generate L5 Summary (จะเรียก generate_evidence_summary_for_level และ summarize_context_with_llm ซึ่งจะถูก Patch)
    """
    updated_breakdown = {}
    sub_criteria_breakdown = results.get("SubCriteria_Breakdown", {})
    for sub_id, sub_data in sub_criteria_breakdown.items():
        try:
            if isinstance(sub_data, str):
                sub_data = {"name": sub_data}
            sub_name = sub_data.get("name", sub_id)
            try:
                # 🛑 ใช้ generate_evidence_summary_for_level() ของ EnablerAssessment
                l5_context_info = assessor.generate_evidence_summary_for_level(sub_id, 5)
            except Exception:
                l5_context_info = None
            
            # NOTE: l5_context_info ใน EnablerAssessment.generate_evidence_summary_for_level คาดว่าจะ return string
            l5_context = l5_context_info.get("combined_context", "") if isinstance(l5_context_info, dict) else (l5_context_info if isinstance(l5_context_info, str) else "")
            
            if "MOCK SUMMARY" in l5_context:
                l5_summary_result = {"summary": l5_context, "suggestion_for_next_level": "N/A (MOCK)"}
            elif not l5_context.strip():
                l5_summary_result = {"summary": "ไม่พบหลักฐานที่เกี่ยวข้องสำหรับ Level 5 ในฐานข้อมูล RAG.", "suggestion_for_next_level": "N/A"}
            else:
                try:
                    # 🛑 ใช้ summarize_context_with_llm() ของ core.retrieval_utils ซึ่งจะถูก Patch
                    l5_summary_result = core.retrieval_utils.summarize_context_with_llm(
                        context=l5_context,
                        sub_criteria_name=sub_name,
                        level=5,
                        sub_id=sub_id,
                        schema=EvidenceSummary
                    )
                    if not isinstance(l5_summary_result, dict):
                        l5_summary_result = {"summary": str(l5_summary_result), "suggestion_for_next_level": "N/A"}
                except Exception as e:
                    l5_summary_result = {"summary": f"Error generating L5 summary: {e}", "suggestion_for_next_level": "N/A"}
                    
            sub_data["evidence_summary_L5"] = l5_summary_result
            updated_breakdown[sub_id] = sub_data
        except Exception as e_outer:
            updated_breakdown[sub_id] = {"name": str(sub_data), "evidence_summary_L5": {"summary": f"Error: {e_outer}", "suggestion_for_next_level": "N/A"}}
    results["SubCriteria_Breakdown"] = updated_breakdown
    return results

# -------------------- MAIN ASSESSMENT --------------------
def run_assessment_process(enabler: str, sub_criteria_id: str, mode: str = "real", filter_mode: bool = False, export: bool = False) -> Dict[str, Any]:
    start_time_global = time.perf_counter()
    global _MOCK_EVALUATION_COUNTER
    _MOCK_EVALUATION_COUNTER = 0

    retriever = None
    summary: Dict[str, Any] = {}
    
    original_mode = mode 
    
    # 🛑 FIX 1: แก้ไขการส่งค่า Boolean ไปยัง set_mock_control_mode
    set_mock_control_mode(original_mode == "mock")
    
    # 🛑 FIX: กำหนด Mock Functions ทั้งหมดที่นี่
    llm_eval_func = evaluate_with_llm_CONTROLLED_MOCK if original_mode=="mock" else None
    llm_summarize_func = summarize_context_with_llm_MOCK if original_mode=="mock" else None 
    llm_action_plan_func = generate_action_plan_MOCK if original_mode=="mock" else None 


    # 1. Setup RAG/Vectorstore
    try:
        if mode == "real":
            temp_loader = EnablerAssessment(enabler_abbr=enabler, vectorstore_retriever=None)
            evidence_mapping = temp_loader.evidence_mapping_data
            all_enabler_file_ids = []
            for key, data in evidence_mapping.items():
                all_enabler_file_ids.extend(data.get('filter_ids', []))
            base_enabler_files = list(set(all_enabler_file_ids))
            target_collection_names = base_enabler_files
            if filter_mode and sub_criteria_id != "all":
                file_ids_to_load = []
                for key, data in evidence_mapping.items():
                    if key.startswith(f"{sub_criteria_id}_L"): 
                        file_ids_to_load.extend(data.get('filter_ids', []))
                target_collection_names = list(set(file_ids_to_load))
            
            retriever = load_all_vectorstores(doc_ids=target_collection_names, doc_type=["evidence"])
            
    except Exception as e:
        logger.error(f"❌ ERROR: Failed to load Vectorstores in REAL mode: {e}", exc_info=True)
        mode = "random" 
        logger.warning(f"⚠️ MODE CHANGED TO: {mode.upper()} due to Vectorstore Load Failure. Assessment will run in RANDOM mode.")


    # 2. Load & Filter Evidence
    try:
        if 'temp_loader' in locals() and mode=="real":
            enabler_loader = temp_loader
        else:
            enabler_loader = EnablerAssessment(enabler_abbr=enabler, vectorstore_retriever=retriever)
        
        filtered_evidence = enabler_loader.evidence_data
        if sub_criteria_id != "all":
            filtered_evidence = [e for e in filtered_evidence if e.get("Sub_Criteria_ID")==sub_criteria_id]
            
    except Exception as e:
        summary.update(EnablerAssessment(enabler_abbr=enabler).summarize_results())
        summary['Error'] = str(e)
        summary['mode_used'] = mode 
        return summary

    # 3. Create Assessment Engine
    assessment_engine = EnablerAssessment(
        enabler_abbr=enabler,
        evidence_data=filtered_evidence,
        rubric_data=enabler_loader.rubric_data,
        level_fractions=enabler_loader.level_fractions,
        evidence_mapping_data=enabler_loader.evidence_mapping_data,
        vectorstore_retriever=retriever,
        use_retrieval_filter=filter_mode,
        target_sub_id=sub_criteria_id if sub_criteria_id!="all" else None,
        # 🛑 FIX: ส่ง Mock Functions ทั้ง 3 ตัว
        mock_llm_eval_func=llm_eval_func,
        mock_llm_summarize_func=llm_summarize_func, # <--- NEW
        mock_llm_action_plan_func=llm_action_plan_func # <--- NEW
    )

    # 4. Run Assessment
    try:
        if mode=="mock":
            # Patch retrieve_context เฉพาะใน Mock Mode
            assessment_engine._retrieve_context = lambda **kwargs: retrieve_context_MOCK(
                statement=kwargs.get('query'), 
                sub_criteria_id=kwargs['sub_criteria_id'],
                level=kwargs.get('level'),
                statement_number=kwargs.get('statement_number', 0), 
                mapping_data=kwargs.get('mapping_data') 
            )
        assessment_engine.run_assessment()
        summary = assessment_engine.summarize_results()
        summary['raw_llm_results'] = assessment_engine.raw_llm_results
        
    except Exception as e:
        summary.update(assessment_engine.summarize_results())
        summary['Error_Details'] = str(e)
        

    # 5. Generate Evidence Summaries (จัดการ Patch)
    summary_patcher_enabler = None
    summary_patcher_utils = None
    if original_mode=="mock":
        # Patch ใน enabler_assessment.py (สำหรับ generate_evidence_summary_for_level)
        summary_patcher_enabler = patch('assessments.enabler_assessment.summarize_context_with_llm', new=summarize_context_with_llm_MOCK)
        # Patch ใน core.retrieval_utils (สำหรับ generate_and_integrate_l5_summary)
        summary_patcher_utils = patch('core.retrieval_utils.summarize_context_with_llm', new=summarize_context_with_llm_MOCK)
        summary_patcher_enabler.start()
        summary_patcher_utils.start()

    try:
        breakdown = summary.get("SubCriteria_Breakdown", {})
        for sub_id, data in breakdown.items():
            target_level = data.get("highest_full_level", 0)
            summary_key_name = f"evidence_summary_L{target_level}"
            if target_level>0:
                summary_text = assessment_engine.generate_evidence_summary_for_level(sub_id, target_level)
                data[summary_key_name] = summary_text
            else:
                data[summary_key_name] = "ไม่พบหลักฐานที่ผ่านเกณฑ์ Level 1"
        summary = generate_and_integrate_l5_summary(assessment_engine, summary)
    finally:
        if original_mode=="mock":
            summary_patcher_enabler.stop()
            summary_patcher_utils.stop()

    # 6. Generate Action Plans (จัดการ Patch)
    action_patcher = None
    full_summary_data = summary
    if original_mode=="mock":
        # Patch generate_action_plan_via_llm ใน core.retrieval_utils (เฉพาะใน mock mode)
        action_patcher = patch('core.retrieval_utils.generate_action_plan_via_llm', new=generate_action_plan_MOCK)
        action_patcher.start()
    
    # 🛑 NOTE: เมื่อไม่ใช้ mock mode การเรียก generate_action_plan_for_sub จะเรียก LLM จริง
    try:
        all_action_plans = {}
        for sub_id_key, summary_data in summary.get('SubCriteria_Breakdown', {}).items():
            action_plan_data = generate_action_plan_for_sub(sub_id_key, summary_data, full_summary_data)
            all_action_plans[sub_id_key] = action_plan_data
        summary['Action_Plans'] = all_action_plans
    finally:
        # 🛑 NOTE: ลบเงื่อนไข action_patcher ออกไปถ้าเราไม่ใช้ mock mode
        if original_mode=="mock" and action_patcher:
            action_patcher.stop()

    # 7. Export JSON
    if export and "Overall" in summary:
        EXPORT_DIR = os.path.join(project_root, "results")
        os.makedirs(EXPORT_DIR, exist_ok=True)
        
        mode_suffix = original_mode.upper() 
        filter_suffix = "STRICTFILTER" if filter_mode else "FULLSCOPE" 
        random_suffix = os.urandom(4).hex()
        
        EXPORT_FILENAME = f"assessment_report_{enabler}_{sub_criteria_id}_{mode_suffix}_{filter_suffix}_{random_suffix}.json" 
        FULL_EXPORT_PATH = os.path.join(EXPORT_DIR, EXPORT_FILENAME)
        
        try:
            export_summary = summary.copy()
            raw_data_to_export = export_summary.pop('raw_llm_results', None)
            with open(FULL_EXPORT_PATH, 'w', encoding='utf-8') as f:
                json.dump(export_summary, f, ensure_ascii=False, indent=4)
            summary['export_path_used'] = FULL_EXPORT_PATH
            if raw_data_to_export:
                raw_data_to_export = add_pass_status_to_raw_results(raw_data_to_export)
                raw_filename = FULL_EXPORT_PATH.replace(".json", "_RAW_EVAL.json")
                with open(raw_filename, 'w', encoding='utf-8') as f:
                    json.dump({"raw_llm_results": raw_data_to_export}, f, ensure_ascii=False, indent=4)
                summary['raw_export_path_used'] = raw_filename
        except Exception as e:
            logger.error(f"Failed to export JSON report: {e}")

    summary['Execution_Time'] = {
        "total": time.perf_counter()-start_time_global
    }
    summary['mode_used'] = mode 
    return summary


# -------------------- CLI Entry Point (Adapter) --------------------
if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Automated Enabler Maturity Assessment System.")
        parser.add_argument("--mode", 
                            choices=["mock", "random", "real"], 
                            default="mock",
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
        
        args = parser.parse_args()
        
        
        # CLI Call: เรียกใช้ฟังก์ชัน run_assessment_process
        final_results = run_assessment_process(
            enabler=args.enabler,
            sub_criteria_id=args.sub,
            mode=args.mode, 
            filter_mode=args.filter,
            export=args.export
        )
        
        
        mode_used_for_report = final_results.get('mode_used', args.mode)
        
        # -------------------- Output Summary for CLI --------------------
        if "Error_Details" in final_results:
            print(f"\n❌ FATAL ERROR: Assessment failed during execution: {final_results['Error_Details']}", file=sys.stderr)
            
        
        summary = final_results
        overall_data = summary.get('Overall', {})
        sub_breakdown = summary.get('SubCriteria_Breakdown', {})
        
        print("\n=====================================================")
        print(f"      SUMMARY OF SCORING RESULTS ({mode_used_for_report.upper()} MODE) ")
        print(f"      ENABLER: {args.enabler.upper()} ")
        print("=====================================================")
        
        if overall_data:
            print(f"Overall Maturity Score (Avg.): {overall_data.get('overall_maturity_score', 0.0):.2f} (Scale: 0.0-1.0)")
            print(f"Total Score (Weighted): {overall_data.get('total_weighted_score', 0.0):.2f}/{overall_data.get('total_possible_weight', 0.0):.2f} (Progress: {overall_data.get('overall_progress_percent', 0.0):.2f}%)")
        else:
            print("⚠️ Overall Summary Data Missing.")

        print("\n-----------------------------------------------------")
        
        if sub_breakdown:
            for sub_id, data in sub_breakdown.items():
                
                if args.sub != "all" and sub_id != args.sub:
                    continue
                
                highest_full_level = data.get('highest_full_level', 0)
                summary_key = f"evidence_summary_L{highest_full_level}"
                
                evidence_summary_raw = data.get(summary_key, "N/A")
                if isinstance(evidence_summary_raw, dict):
                    evidence_summary = evidence_summary_raw.get('summary', 'N/A')
                elif isinstance(evidence_summary_raw, str):
                    evidence_summary = evidence_summary_raw
                else:
                    evidence_summary = "N/A"
                
                ratios = data.get('pass_ratios', {})
                ratios_display = []
                for lvl in range(1, 6):
                    ratio = ratios.get(str(lvl), 0.0)
                    symbol = "🟢" if ratio == 1.0 else "🟡" if ratio > 0 and ratio < 1.0 else "🔴"
                    ratios_display.append(f"L{lvl}: {symbol}{ratio:.2f}")
                
                print(f"| {sub_id}: {data.get('name', 'N/A')}")
                print(f"| - Score: {data.get('score', 0.0):.2f}/{data.get('weight', 0.0):.2f} | Full Lvl: L{highest_full_level} | Gap: {'YES' if data.get('development_gap') else 'NO'}")
                print(f"| - Ratios (L1-L5): {' | '.join(ratios_display)}") 
                
                print(f"| - Summary L{highest_full_level}: {evidence_summary}") 

                if data.get('development_gap'):
                    print(f"| - Action: {data.get('action_item', 'See Action Plans section.')}") 
                print("-----------------------------------------------------")
        else:
            print("⚠️ No Sub-Criteria breakdown results found.")


        print("\n\n=====================================================")
        print("        GENERATING ACTION PLAN...")
        print("=====================================================")
        
        if 'Action_Plans' in final_results:
            for sub_id, action_plan_phases in final_results.get('Action_Plans', {}).items():
                
                if args.sub != "all" and sub_id != args.sub:
                    continue
                
                summary_data = sub_breakdown.get(sub_id, {})
                highest_full_level = summary_data.get('highest_full_level', 0)
                target_level = highest_full_level + 1

                print(f"\n--- ACTION PLAN FOR {args.enabler.upper()} - {sub_id} (Target L{target_level}) ---")
                
                if isinstance(action_plan_phases, List):
                    for phase in action_plan_phases:
                        print(f"\n[PHASE] {phase.get('Phase', 'N/A')}")
                        print(f"[GOAL] {phase.get('Goal', 'N/A')}")
                        
                        if phase.get('Actions'):
                            print("\n[ACTIONS]")
                            for action in phase['Actions']:
                                stmt_id = action.get('Statement_ID', 'N/A')
                                failed_lvl = action.get('Failed_Level', 'N/A')
                                
                                print(f"  - Statement: {stmt_id} (L{failed_lvl})") 
                                print(f"    - Recommendation: {action.get('Recommendation', 'N/A')}")
                                print(f"    - Target Evidence: {action.get('Target_Evidence_Type', 'N/A')}")
                                print(f"    - Key Metric: {action.get('Key_Metric', 'N/A')}")
                        else:
                            print("[ACTIONS] No specific actions listed in this phase.")
                else:
                    print(f"Error: Action plan for {sub_id} is not a valid list. Details: {action_plan_phases}")

        
        print(f"\n[⏱️ TOTAL EXECUTION TIME] All processes completed in: {final_results['Execution_Time']['total']:.2f} seconds.")
        
        if args.export and 'export_path_used' in final_results:
            print(f"\n[✅ EXPORT SUCCESS] Report JSON exported to: {final_results['export_path_used']}")
            if 'raw_export_path_used' in final_results:
                print(f"[✅ RAW DATA EXPORTED] Raw Evaluation JSON exported to: {final_results['raw_export_path_used']}")
                
        print("\n=====================================================")
        
        print_detailed_results(summary.get('raw_llm_results', []), args.sub) 
        
    except Exception as e:
        logger.error(f"An unexpected error occurred during CLI execution: {e}")