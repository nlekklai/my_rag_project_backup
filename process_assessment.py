# -------------------- process_assessment.py (FINAL FULL CODE) --------------------
import os
import json
import logging
import sys
import argparse
import random
from typing import List, Dict, Any, Optional

# --- CORRECT IMPORTS ---
try:
    # Ensure the parent directory is in sys.path for correct relative imports
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    # IMPORT NEW CLASS: นำเข้า EnablerAssessment
    from assessments.enabler_assessment import EnablerAssessment
    
    # Import the module, not the function, for successful patching
    import core.retrieval_utils
    from core.retrieval_utils import set_mock_control_mode
    
except ImportError as e:
    logger.error(f"FATAL ERROR: Failed to import required modules. Check sys.path and file structure. Error: {e}")
    sys.exit(1)


# 1. Setup Logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# -------------------- Mock Evaluation Logic --------------------
_MOCK_EVALUATION_COUNTER = 0

def evaluate_with_llm_CONTROLLED_MOCK(statement: str, context: str, standard: str) -> Dict[str, Any]:
    """Returns controlled scores: L1(1,1,1), L2(1,1,0), L3-L5(0)"""
    global _MOCK_EVALUATION_COUNTER
    _MOCK_EVALUATION_COUNTER += 1
    
    score = 0
    
    # Logic Controlled Mock ตามที่เคยตกลงกันไว้
    if _MOCK_EVALUATION_COUNTER <= 3: # L1 Statements (Counter 1, 2, 3)
        score = 1
    elif _MOCK_EVALUATION_COUNTER in [4, 5]: # L2 Statements (Counter 4, 5)
        score = 1
    else: # L2 S3 (Counter 6) และ L3-L5 (Counter > 6)
        score = 0 
    
    reason_text = f"MOCK: FORCED {'PASS' if score == 1 else 'FAIL'} (Statement {_MOCK_EVALUATION_COUNTER})"
    logger.debug(f"MOCK COUNT: {_MOCK_EVALUATION_COUNTER} | SCORE: {score} | STMT: '{statement[:20]}...'")
    
    return {"score": score, "reason": reason_text}

# 🚨 EDITED: retrieve_context_MOCK (รับ mapping_data และแสดง filter_ids)
def retrieve_context_MOCK(statement: str, sub_criteria_id: str, level: int, mapping_data: Optional[Dict] = None) -> str:
    """Mock retrieval function returns a string indicating the requested filter key and the intended filter_ids."""
    
    mapping_key = f"{sub_criteria_id}_L{level}"
    filter_ids: List[str] = []
    
    # 1. จำลองการดึง filter_ids จาก Mapping Data
    if mapping_data:
        # ใช้ mapping_data ที่ส่งมาเพื่อดึง filter_ids
        filter_ids = mapping_data.get(mapping_key, {}).get("filter_ids", [])
    
    # 2. จัดรูปแบบข้อมูล Filter ที่จะแสดงใน Context
    if not filter_ids:
         filter_info = "NO FILTER IDS FOUND IN MAPPING."
    else:
         # แสดงจำนวนไฟล์ทั้งหมด และ 2 ไฟล์แรก
         top_ids_snippet = ', '.join([f"'{id}'" for id in filter_ids[:2]])
         filter_info = f"Total {len(filter_ids)} IDs. Top 2: [{top_ids_snippet}, ...]"
    
    # 3. สร้าง Mock Context สำหรับแสดงผล
    mock_context = (
        f"MOCK CONTEXT SNIPPET. [Key: {mapping_key}] "
        f"[Filter Info: {filter_info}]"
    )
    
    return mock_context

# -------------------- Detailed Output Function --------------------

def print_detailed_results(raw_llm_results: List[Dict]):
    """
    แสดงผลลัพธ์การประเมิน LLM ราย Statement อย่างละเอียด
    """
    if not raw_llm_results:
        logger.info("\n[Detailed Results] No raw LLM results found to display.")
        return

    print("\n\n=========================================================================================")
    print("        DETAILED LLM EVALUATION RESULTS (Statement-by-Statement)")
    print("=========================================================================================")

    grouped: Dict[str, List[Dict]] = {}
    for r in raw_llm_results:
        key = f"{r['sub_criteria_id']}: L{r['level']}"
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(r)
    
    sorted_keys = sorted(grouped.keys())

    for key in sorted_keys:
        sub_name = next((r['sub_criteria_name'] for r in grouped[key] if 'sub_criteria_name' in r), 'N/A')
        print(f"\n--- Sub-Criteria / Level: {key} - {sub_name} (Total: {len(grouped[key])} Statements) ---")
        
        for i, r in enumerate(grouped[key]):
            statement_snippet = r['statement'][:120] + "..." if len(r['statement']) > 120 else r['statement']
            reason_snippet = r['reason'][:120] + "..." if len(r['reason']) > 120 else r['reason']
            # 🚨 NOTE: ใช้ context_retrieved_snippet ที่ EnablerAssessment ส่งมา
            context_snippet = r.get('context_retrieved_snippet', 'N/A')
            
            score_text = "✅ PASS (1)" if r['llm_score'] == 1 else "❌ FAIL (0)"
            
            print(f"\n[STMT {i+1}] Result: {score_text}")
            print(f"  > Statement: {statement_snippet}")
            print(f"  > Standard:  {r['standard'][:120] + '...' if len(r['standard']) > 120 else r['standard']}")
            print(f"  > Context:   {context_snippet}")
            print(f"  > Reason:    {reason_snippet}")
            
    print("\n=========================================================================================")


# -------------------- Main CLI Entry Point --------------------
if __name__ == "__main__":
    
    # 1. Setup Argparse
    parser = argparse.ArgumentParser(description="Automated Enabler Maturity Assessment System.")
    parser.add_argument("--mode", 
                        choices=["mock", "random", "real"], 
                        default="random",
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
    parser.add_argument("--export", 
                        action="store_true",
                        help="Export the final summary results to a JSON file.")
    
    args = parser.parse_args()
    
    _MOCK_EVALUATION_COUNTER = 0 
    retriever = None 
    
    # 2. Setup LLM/RAG Mode (Mode-specific setup)
    if args.mode == "mock":
        set_mock_control_mode(True)
        original_eval_func = core.retrieval_utils.evaluate_with_llm
        core.retrieval_utils.evaluate_with_llm = evaluate_with_llm_CONTROLLED_MOCK
        logger.info(f"Using CONTROLLED MOCK Mode for Enabler: {args.enabler.upper()} on Sub-Criteria: {args.sub}.")
    
    elif args.mode == "random":
        logger.info(f"Using RANDOM LLM Scores (0/1) for Enabler: {args.enabler.upper()}.")
        pass

    elif args.mode == "real":
        try:
            from core.vectorstore import load_all_vectorstores 
            retriever = load_all_vectorstores()
            logger.info(f"Using REAL RAG/LLM mode for Enabler: {args.enabler.upper()}.")
        except Exception as e:
            logger.warning(f"Error loading vectorstore for REAL mode: {e}. Falling back to RANDOM.")
            args.mode = "random"


    # 3. Load & Filter Evidence Data (Load all required data first)
    try:
        enabler_loader = EnablerAssessment(
            enabler_abbr=args.enabler,
            vectorstore_retriever=retriever
        )
    except Exception as e:
        logger.error(f"Error during EnablerAssessment init: {e}")
        sys.exit(1)

    if not enabler_loader.evidence_data:
        logger.error(f"❌ EnablerAssessment failed to load evidence data for '{args.enabler}'. Cannot run test.")
        sys.exit(1)
        
    filtered_evidence = enabler_loader.evidence_data
    if args.sub != "all":
        filtered_evidence = [
            e for e in enabler_loader.evidence_data 
            if e.get("Sub_Criteria_ID") == args.sub
        ]
        if not filtered_evidence:
            logger.error(f"❌ Sub-Criteria ID '{args.sub}' not found in JSON for Enabler '{args.enabler}'.")
            sys.exit(1)
        
    # 4. Create Final EnablerAssessment Object (with filtered data)
    enabler = EnablerAssessment(
        enabler_abbr=args.enabler,
        evidence_data=filtered_evidence, # ส่งข้อมูลที่ถูกกรอง
        rubric_data=enabler_loader.rubric_data,
        level_fractions=enabler_loader.level_fractions,
        evidence_mapping_data=enabler_loader.evidence_mapping_data, # ส่ง Mapping Data
        vectorstore_retriever=retriever
    )
    
    print(f"✅ Loaded {len(filtered_evidence)} Sub-Criteria (Filtered to {args.sub}) for assessment of ENABLER: {args.enabler.upper()}.")
    
    
    # 5. Run Assessment
    try:
        if args.mode == "mock":
            # 🚨 Patching retrieve_context: ใช้ฟังก์ชัน Mock ที่รับ 4 Argument
            enabler._retrieve_context = retrieve_context_MOCK 
        
        results = enabler.run_assessment()
        
    finally:
        # Restore original function
        if args.mode == "mock":
            core.retrieval_utils.evaluate_with_llm = original_eval_func
        
    summary = enabler.summarize_results()

    
    # 6. Output Summary
    print("\n=====================================================")
    print(f"      SUMMARY OF SCORING RESULTS ({args.mode.upper()} MODE) ")
    print(f"      ENABLER: {args.enabler.upper()} ")
    print("=====================================================")
    print(f"Overall Maturity Score (Avg.): {summary['Overall']['overall_maturity_score']} (Scale: 0.0-1.0)")
    print(f"Total Score (Weighted): {summary['Overall']['total_weighted_score']}/{summary['Overall']['total_possible_weight']} (Progress: {summary['Overall']['overall_progress_percent']}%)")
    print("\n-----------------------------------------------------")
    
    for sub_id, data in summary['SubCriteria_Breakdown'].items():
        print(f"| {sub_id}: {data['name']}")
        print(f"| - Score: {data['score']}/{data['weight']} | Full Lvl: L{data['highest_full_level']} | Gap: {'YES' if data['development_gap'] else 'NO'}")
        print(f"| - Ratios (L1-L5): {data['pass_ratios']}") 
        if data['development_gap']:
            print(f"| - Action: {data['action_item']}")
        print("-----------------------------------------------------")
        
    # 7. Export JSON 
    if args.export:
        EXPORT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))
        try:
            os.makedirs(EXPORT_DIR, exist_ok=True)
            logger.info(f"Using export directory: {EXPORT_DIR}")
        except OSError as e:
            logger.error(f"❌ Failed to create results directory {EXPORT_DIR}: {e}")
            sys.exit(1)
        
        EXPORT_FILENAME = f"assessment_summary_{args.enabler}_{args.sub}_{args.mode}_{os.urandom(4).hex()}.json"
        FULL_EXPORT_PATH = os.path.join(EXPORT_DIR, EXPORT_FILENAME)

        try:
            with open(FULL_EXPORT_PATH, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=4)
            print(f"\n✅ Successfully exported summary to {FULL_EXPORT_PATH}")
        except Exception as e:
            logger.error(f"❌ Failed to export JSON summary to {FULL_EXPORT_PATH}: {e}")
            
    # 7.5. เรียกใช้ฟังก์ชันแสดงผลลัพธ์ละเอียด
    print_detailed_results(enabler.raw_llm_results)