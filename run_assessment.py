# -------------------- run_assessment.py (FINAL FIXED VERSION) --------------------
import os
import json
import logging
import sys
import argparse
import random
from typing import List, Dict, Any

# 1. Setup Logging (Need to set it up here since it's the main entry point)
logger = logging.getLogger(__name__)
# กำหนด logging format ให้ชัดเจนขึ้น
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CORRECT IMPORTS ---
try:
    # Ensure the parent directory is in sys.path for correct relative imports
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    # Import the main class from the assessments module
    from assessments.km_assessment import KMAssessment
    
    # ❗ IMPORTANT: Import the module, not the function, for successful patching
    import core.retrieval_utils
    from core.retrieval_utils import set_mock_control_mode
    
except ImportError as e:
    logger.error(f"FATAL ERROR: Failed to import required modules. Check sys.path and file structure. Error: {e}")
    sys.exit(1)


# -------------------- Mock Evaluation Logic --------------------
# Counter ถูกกำหนดให้เป็น Global ในไฟล์นี้
_MOCK_EVALUATION_COUNTER = 0

def evaluate_with_llm_CONTROLLED_MOCK(statement: str, context: str, standard: str) -> Dict[str, Any]:
    """Returns controlled scores: L1(1,1,1), L2(1,1,0), L3-L5(0)"""
    global _MOCK_EVALUATION_COUNTER
    _MOCK_EVALUATION_COUNTER += 1
    
    # 🎯 ULTIMATE MOCK LOGIC: (L1: 100%, L2: 66% Pass)
    # Sub-Criteria 1.1 มี 3 Statements/Level (รวม 15 Statements)
    
    score = 0
    
    if _MOCK_EVALUATION_COUNTER <= 3: # L1 Statements (Counter 1, 2, 3)
        score = 1
    elif _MOCK_EVALUATION_COUNTER in [4, 5]: # L2 Statements (Counter 4, 5)
        score = 1
    else: # L2 S3 (Counter 6) และ L3-L5 (Counter > 6)
        score = 0 # FORCED FAIL for easy verification
    
    logger.debug(f"MOCK COUNT: {_MOCK_EVALUATION_COUNTER} | SCORE: {score} | STMT: '{statement[:20]}...'")
    
    return {"score": score, "reason": f"CONTROLLED MOCK Score {score}"}

def retrieve_context_MOCK(statement: str) -> str:
    """Mock retrieval function returns empty string for speed."""
    # ใน Mock Mode เราใช้ Snippet สั้นๆ เพื่อให้ Context ไม่ว่างเปล่า
    return "MOCK CONTEXT SNIPPET: The organization has a formal KM policy approved by the CEO."

# -------------------- Detailed Output Function --------------------

def print_detailed_results(raw_llm_results: List[Dict]):
    """
    แสดงผลลัพธ์การประเมิน LLM ราย Statement อย่างละเอียด
    """
    if not raw_llm_results:
        logger.info("\n[Detailed Results] No raw LLM results found to display.")
        return

    # ❗ สลับการแสดงผลให้ Detailed อยู่หลัง Summary
    print("\n\n=========================================================================================")
    print("        DETAILED LLM EVALUATION RESULTS (Statement-by-Statement)")
    print("=========================================================================================")

    # จัดกลุ่มตาม Sub-Criteria ID และ Level เพื่อให้อ่านง่าย
    grouped: Dict[str, List[Dict]] = {}
    for r in raw_llm_results:
        key = f"{r['sub_criteria_id']}: L{r['level']}"
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(r)
    
    # เรียงลำดับตาม Sub ID และ Level
    sorted_keys = sorted(grouped.keys())

    for key in sorted_keys:
        # ใช้ Sub-Criteria Name (ถ้าดึงได้)
        sub_name = next((r['sub_criteria_name'] for r in grouped[key] if 'sub_criteria_name' in r), 'N/A')
        print(f"\n--- Sub-Criteria / Level: {key} - {sub_name} (Total: {len(grouped[key])} Statements) ---")
        
        for i, r in enumerate(grouped[key]):
            # ตัดข้อความยาวๆ เพื่อให้อ่านง่ายใน Console
            statement_snippet = r['statement'][:120] + "..." if len(r['statement']) > 120 else r['statement']
            reason_snippet = r['reason'][:120] + "..." if len(r['reason']) > 120 else r['reason']
            # ใน km_assessment.py เราเก็บ context_retrieved_snippet ไว้
            context_snippet = r.get('context_retrieved_snippet', 'N/A')
            
            # การแสดงผล
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
    parser = argparse.ArgumentParser(description="Automated KM Maturity Assessment System.")
    parser.add_argument("--mode", 
                        choices=["mock", "random", "real"], 
                        default="random",
                        help="Assessment mode: 'mock' (controlled scores for 1.1), 'random' (random 0/1 scores), or 'real' (requires actual RAG setup).")
    parser.add_argument("--sub", 
                        type=str, 
                        default="all",
                        help="Filter to a specific Sub-Criteria ID (e.g., '1.1'). Default is 'all'.")
    parser.add_argument("--export", 
                        action="store_true",
                        help="Export the final summary results to a JSON file.")
    
    args = parser.parse_args()
    
    retriever = None 
    
    # 2. Setup LLM/RAG Mode (Mode-specific setup)
    if args.mode == "mock":
        # 🟢 FIX 1: ตั้งค่า Global Flag และรีเซ็ต Counter ใน core/retrieval_utils
        set_mock_control_mode(True)
        # 🟢 FIX 2: Patch evaluation function ในโมดูล core.retrieval_utils โดยตรง
        original_eval_func = core.retrieval_utils.evaluate_with_llm
        core.retrieval_utils.evaluate_with_llm = evaluate_with_llm_CONTROLLED_MOCK
        logger.info(f"Using CONTROLLED MOCK Mode on Sub-Criteria: {args.sub}.")
    
    elif args.mode == "random":
        logger.info("Using RANDOM LLM Scores (0/1) for full test.")
        pass

    elif args.mode == "real":
        try:
            from core.vectorstore import load_all_vectorstores 
            retriever = load_all_vectorstores()
            logger.info("Using REAL RAG/LLM mode.")
        except Exception as e:
            logger.warning(f"Error loading vectorstore for REAL mode: {e}. Falling back to RANDOM.")
            args.mode = "random"


    # 3. Load & Filter Evidence Data
    try:
        km_loader = KMAssessment(vectorstore_retriever=retriever)
    except Exception as e:
        logger.error(f"Error during KMAssessment init: {e}")
        sys.exit(1)

    if not km_loader.evidence_data:
        logger.error("❌ KMAssessment failed to load evidence data. Cannot run test.")
        sys.exit(1)
        
    filtered_evidence = km_loader.evidence_data
    if args.sub != "all":
        filtered_evidence = [
            e for e in km_loader.evidence_data 
            if e.get("Sub_Criteria_ID") == args.sub
        ]
        if not filtered_evidence:
            logger.error(f"❌ Sub-Criteria ID '{args.sub}' not found in JSON.")
            sys.exit(1)
        
    # 4. Create Final KMAssessment Object (with filtered data)
    km = KMAssessment(
        evidence_data=filtered_evidence,
        rubric_data=km_loader.rubric_data,
        level_fractions=km_loader.level_fractions,
        vectorstore_retriever=retriever
    )
    
    print(f"✅ Loaded {len(filtered_evidence)} Sub-Criteria (Filtered to {args.sub}) for assessment.")
    
    
    # 5. Run Assessment
    try:
        # Patch retrieval for speed in mock/random mode
        if args.mode == "mock" or args.mode == "random":
            km._retrieve_context = retrieve_context_MOCK 
        
        results = km.run_assessment()
        
    finally:
        # 🟢 FIX 3: Clean up patch
        if args.mode == "mock":
            core.retrieval_utils.evaluate_with_llm = original_eval_func
        
    summary = km.summarize_results()

    
    # 6. Output Summary (Code is the same as provided)
    print("\n=====================================================")
    print(f"      SUMMARY OF SCORING RESULTS ({args.mode.upper()} MODE) ")
    print("=====================================================")
    print(f"Overall Maturity Score (Avg.): {summary['Overall']['overall_maturity_score']}")
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
        
        EXPORT_FILENAME = f"km_assessment_summary_{args.sub}_{args.mode}_{os.urandom(4).hex()}.json"
        FULL_EXPORT_PATH = os.path.join(EXPORT_DIR, EXPORT_FILENAME)

        try:
            with open(FULL_EXPORT_PATH, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=4)
            print(f"\n✅ Successfully exported summary to {FULL_EXPORT_PATH}")
        except Exception as e:
            logger.error(f"❌ Failed to export JSON summary to {FULL_EXPORT_PATH}: {e}")
            
    # 7.5. ❗ เรียกใช้ฟังก์ชันแสดงผลลัพธ์ละเอียด
    # NOTE: เราเรียกใช้ที่ท้ายสุดเพื่อให้ Summary ขึ้นก่อนเสมอ
    print_detailed_results(km.raw_llm_results)