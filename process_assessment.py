import os
import json
import logging
import sys
import argparse
import random
from typing import List, Dict, Any, Optional
import time 

# --- CORRECT IMPORTS ---
try:
    # Ensure the parent directory is in sys.path for correct relative imports
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    # IMPORT REQUIRED CLASSES/FUNCTIONS
    from assessments.enabler_assessment import EnablerAssessment
    import core.retrieval_utils
    from core.retrieval_utils import set_mock_control_mode
    from core.vectorstore import load_all_vectorstores 
    
except ImportError as e:
    # หากรันจากตำแหน่งที่ผิด หรือ Path ไม่ถูกต้อง
    print(f"FATAL ERROR: Failed to import required modules. Check sys.path and file structure. Error: {e}", file=sys.stderr)
    sys.exit(1)


# 1. Setup Logging
logger = logging.getLogger(__name__)
# ตั้งค่า Logging ให้แสดงผลลัพธ์ตามที่คุณต้องการ
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# -------------------- Mock Evaluation Logic --------------------
_MOCK_EVALUATION_COUNTER = 0

def evaluate_with_llm_CONTROLLED_MOCK(statement: str, context: str, standard: str) -> Dict[str, Any]:
    """Returns controlled scores for mock testing: L1(1,1,1), L2(1,1,0), L3-L5(0)"""
    global _MOCK_EVALUATION_COUNTER
    _MOCK_EVALUATION_COUNTER += 1
    
    # กำหนดคะแนนจำลองตาม Statement Counter
    score = 0
    if _MOCK_EVALUATION_COUNTER <= 3: 
        score = 1
    elif _MOCK_EVALUATION_COUNTER in [4, 5]: 
        score = 1
    else: 
        score = 0 
    
    reason_text = f"MOCK: FORCED {'PASS' if score == 1 else 'FAIL'} (Statement {_MOCK_EVALUATION_COUNTER})"
    logger.debug(f"MOCK COUNT: {_MOCK_EVALUATION_COUNTER} | SCORE: {score} | STMT: '{statement[:20]}...'")
    return {"score": score, "reason": reason_text}

def retrieve_context_MOCK(statement: str, sub_criteria_id: str, level: int, mapping_data: Optional[Dict] = None) -> str:
    """Mock retrieval function returns a string indicating the requested filter key and the intended filter_ids."""
    
    mapping_key = f"{sub_criteria_id}_L{level}"
    filter_ids: List[str] = []
    
    if mapping_data:
        filter_ids = mapping_data.get(mapping_key, {}).get("filter_ids", [])
    
    if not filter_ids:
         filter_info = "NO FILTER IDS FOUND IN MAPPING."
    else:
         top_ids_snippet = ', '.join([f"'{id}'" for id in filter_ids[:2]])
         filter_info = f"Total {len(filter_ids)} IDs. Top 2: [{top_ids_snippet}, ...]"
    
    mock_context = (
        f"MOCK CONTEXT SNIPPET. [Key: {mapping_key}] "
        f"[Filter Info: {filter_info}]"
    )
    return mock_context

# -------------------- Detailed Output Function --------------------
def print_detailed_results(raw_llm_results: List[Dict]):
    """แสดงผลลัพธ์การประเมิน LLM ราย Statement อย่างละเอียด"""
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
            # หากมีการดึง Context, ให้แสดง snippet
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
    
    # Start global timer
    start_time_global = time.perf_counter()
    
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
    
    parser.add_argument("--filter", 
                        action="store_true", 
                        help="Enable metadata filtering based on the KM mapping file (Strict Filter Mode).")
    
    parser.add_argument("--export", 
                        action="store_true",
                        help="Export the final summary results to a JSON file.")
    
    args = parser.parse_args()
    
    _MOCK_EVALUATION_COUNTER = 0 
    retriever = None 
    
    # 2. Setup LLM/RAG Mode (Mode-specific setup)
    start_time_setup = time.perf_counter() 
    
    if args.mode == "mock":
        set_mock_control_mode(True)
        # 🟢 CRITICAL FIX: แทนที่ฟังก์ชันประเมิน LLM ด้วยฟังก์ชัน Mock
        original_eval_func = core.retrieval_utils.evaluate_with_llm
        core.retrieval_utils.evaluate_with_llm = evaluate_with_llm_CONTROLLED_MOCK
        logger.info(f"Using CONTROLLED MOCK Mode for Enabler: {args.enabler.upper()} on Sub-Criteria: {args.sub}.")
    
    elif args.mode == "random":
        logger.info(f"Using RANDOM LLM Scores (0/1) for Enabler: {args.enabler.upper()}.")
        pass

    elif args.mode == "real":
        try:
            # 🚨 FIX 1: โหลด EnablerAssessment ชั่วคราวเพื่อเข้าถึง mapping data
            temp_loader = EnablerAssessment(enabler_abbr=args.enabler, vectorstore_retriever=None)
            evidence_mapping = temp_loader.evidence_mapping_data
            
            target_collection_names = None
            
            # 1. รวบรวมชื่อไฟล์ทั้งหมดที่อยู่ใน Evidence Mapping ของ Enabler นี้ (KM)
            #    ใช้เป็นฐานสำหรับการโหลดเสมอ (KM-Only Scope)
            all_enabler_file_ids = []
            for key, data in evidence_mapping.items():
                all_enabler_file_ids.extend(data.get('filter_ids', []))
            
            base_enabler_files = list(set(all_enabler_file_ids))


            if args.filter and args.sub != "all":
                # --- Scenario 1: Strict Filter Mode (Load Only Specified Sub-Criteria Files) ---
                
                file_ids_to_load = []
                for key, data in evidence_mapping.items():
                    if key.startswith(f"{args.sub}_L"): 
                        file_ids_to_load.extend(data.get('filter_ids', []))

                target_collection_names = list(set(file_ids_to_load)) 
                
                logger.info(f"⚡ Hard-limiting Vectorstore load (due to --filter) to {len(target_collection_names)} documents of {args.sub}.")
                logger.info(f"   RAG Search Mode: STRICT FILTER")
            
            else:
                # --- Scenario 2: Full Scope Search WITHIN Enabler (No Filter) ---
                # ใช้ไฟล์ทั้งหมดของ Enabler (KM)
                target_collection_names = base_enabler_files
                
                logger.info(f"⚡ Loading ALL Vectorstore Collections for {args.enabler} ({len(target_collection_names)} documents).")
                logger.info(f"   RAG Search Mode: FULL SCOPE SEARCH (KM ONLY)")
                
            # -------------------------------------------------------------

            # 🚨 FIX 2: ส่ง doc_ids ที่ถูกกำหนดขอบเขตแล้วไปยัง load_all_vectorstores
            retriever = load_all_vectorstores(
                doc_ids=target_collection_names, 
                doc_type=["evidence"]
            )
            
            logger.info(f"✅ Loaded {len(retriever.retrievers_list)} RAG Retrivers for assessment.")
            logger.info(f"Using REAL RAG/LLM mode for Enabler: {args.enabler.upper()}.")

        except Exception as e:
            # จัดการข้อผิดพลาดและ Fallback ไป RANDOM Mode
            logger.warning(f"Error loading vectorstore for REAL mode: {e}. Falling back to RANDOM.")
            args.mode = "random"


    # Stop setup timer and print
    end_time_setup = time.perf_counter()
    setup_duration = end_time_setup - start_time_setup
    print(f"\n[⏱️ Setup Time] LLM/RAG/Vectorstore Loading took: {setup_duration:.2f} seconds.")


    # 3. Load & Filter Evidence Data (Load all required data first)
    try:
        # หาก temp_loader ถูกสร้างสำเร็จแล้ว ให้ใช้ข้อมูลนั้น
        if 'temp_loader' in locals() and temp_loader.evidence_data:
            enabler_loader = temp_loader
        else:
            # กรณีที่ REAL mode ล้มเหลวและกลับไป RANDOM
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
        # การกรอง Evidence Statements ที่จะใช้ในการประเมิน (Scope Limiter ที่ถูกต้อง)
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
        evidence_data=filtered_evidence, 
        rubric_data=enabler_loader.rubric_data,
        level_fractions=enabler_loader.level_fractions,
        evidence_mapping_data=enabler_loader.evidence_mapping_data, 
        vectorstore_retriever=retriever,
        
        # 🟢 CRITICAL FIX: ส่งสถานะ Filter และ Sub ID ไปยัง Class 
        # เพื่อควบคุม Metadata Filter ใน RAG Query ภายใน Class Assessment
        use_retrieval_filter=args.filter,
        target_sub_id=args.sub if args.sub != "all" else None
    )
    
    print(f"✅ Loaded {len(filtered_evidence)} Statements (Filtered to {args.sub}) for assessment of ENABLER: {args.enabler.upper()}.")
    
    
    # 5. Run Assessment
    start_time_assessment = time.perf_counter() 
    
    try:
        if args.mode == "mock":
            # 🟢 ต้องแทนที่ฟังก์ชัน retrieve_context ด้วย MOCK ด้วย
            enabler._retrieve_context = retrieve_context_MOCK 
        
        results = enabler.run_assessment()
        
    finally:
        # คืนค่าฟังก์ชัน LLM Evaluation กลับไปเป็นค่าเดิมหลังจบการรัน (เฉพาะ Mock Mode)
        if args.mode == "mock":
            core.retrieval_utils.evaluate_with_llm = original_eval_func
            
    # สรุปเวลา
    end_time_assessment = time.perf_counter()
    assessment_duration = end_time_assessment - start_time_assessment
    print(f"\n[⏱️ Assessment Time] LLM Evaluation and RAG Retrieval took: {assessment_duration:.2f} seconds.")

    summary = enabler.summarize_results()

    
    # 6. Output Summary 
    print("\n=====================================================")
    print(f"      SUMMARY OF SCORING RESULTS ({args.mode.upper()} MODE) ")
    print(f"      ENABLER: {args.enabler.upper()} ")
    print("=====================================================")
    print(f"Overall Maturity Score (Avg.): {summary['Overall']['overall_maturity_score']:.2f} (Scale: 0.0-1.0)")
    print(f"Total Score (Weighted): {summary['Overall']['total_weighted_score']:.2f}/{summary['Overall']['total_possible_weight']:.2f} (Progress: {summary['Overall']['overall_progress_percent']:.2f}%)")
    print("\n-----------------------------------------------------")
    
    for sub_id, data in summary['SubCriteria_Breakdown'].items():
        print(f"| {sub_id}: {data['name']}")
        print(f"| - Score: {data['score']:.2f}/{data['weight']:.2f} | Full Lvl: L{data['highest_full_level']} | Gap: {'YES' if data['development_gap'] else 'NO'}")
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
        
        mode_suffix = "REAL" if args.mode == "real" else args.mode.upper()
        # ใช้ FULLSCOPE หรือ STRICTFILTER เพื่อระบุโหมด
        filter_suffix = "STRICTFILTER" if args.filter else "FULLSCOPE" 
        # ใช้ os.urandom(4).hex() เพื่อสร้าง unique id สำหรับชื่อไฟล์
        EXPORT_FILENAME = f"assessment_summary_{args.enabler}_{args.sub}_{mode_suffix}_{filter_suffix}_{os.urandom(4).hex()}.json"
        FULL_EXPORT_PATH = os.path.join(EXPORT_DIR, EXPORT_FILENAME)

        try:
            with open(FULL_EXPORT_PATH, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=4)
            print(f"\n✅ Successfully exported summary to {FULL_EXPORT_PATH}")
        except Exception as e:
            logger.error(f"❌ Failed to export JSON summary to {FULL_EXPORT_PATH}: {e}")
            
    # 7.5. เรียกใช้ฟังก์ชันแสดงผลลัพธ์ละเอียด
    print_detailed_results(enabler.raw_llm_results)

    # Stop global timer and print total
    end_time_global = time.perf_counter()
    global_duration = end_time_global - start_time_global
    print(f"\n[⏱️ TOTAL EXECUTION TIME] All processes completed in: {global_duration:.2f} seconds.")