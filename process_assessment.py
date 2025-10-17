import os
import json
import logging
import sys
import argparse
import random
from typing import List, Dict, Any, Optional, Union
import time 

# --- CORRECT IMPORTS ---
# สมมติว่าไฟล์นี้อยู่ในระดับเดียวกันกับ assessments/ และ core/
try:
    # Ensure the parent directory is in sys.path for correct relative imports
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    # IMPORT REQUIRED CLASSES/FUNCTIONS (Assuming these paths are correct in your environment)
    from assessments.enabler_assessment import EnablerAssessment # นำเข้า Class EnablerAssessment ที่คุณกำหนด
    import core.retrieval_utils
    from core.retrieval_utils import set_mock_control_mode, evaluate_with_llm
    from core.vectorstore import load_all_vectorstores 
    
except ImportError as e:
    # หากรันจากตำแหน่งที่ผิด หรือ Path ไม่ถูกต้อง
    print(f"FATAL ERROR: Failed to import required modules. Check sys.path and file structure. Error: {e}", file=sys.stderr)
    # ในสภาพแวดล้อมจริง เราอาจจะต้องยกเลิกการทำงาน
    # sys.exit(1)
    # For a unified example, we will proceed assuming the imports are available or mocked later.
    pass


# 1. Setup Logging
logger = logging.getLogger(__name__)
# ตั้งค่า Logging ให้แสดงผลลัพธ์ตามที่คุณต้องการ
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# -------------------- Mock Evaluation Logic --------------------
_MOCK_EVALUATION_COUNTER = 0

def evaluate_with_llm_CONTROLLED_MOCK(statement: str, context: str, standard: str) -> Dict[str, Any]:
    """
    Returns controlled scores for mock testing: L1(1,1,1), L2(1,1,0), L3-L5(0).
    (Used to simulate the target scenario 1.1 L2 Fail on S3)
    """
    global _MOCK_EVALUATION_COUNTER
    _MOCK_EVALUATION_COUNTER += 1
    
    # Simulate L1 PASS, L2 S1/S2 PASS, L2 S3 FAIL (for Sub-Criteria 1.1, total 6 statements)
    score = 0
    if _MOCK_EVALUATION_COUNTER <= 3: # L1 Statements
        score = 1
    elif _MOCK_EVALUATION_COUNTER in [4, 5]: # L2 S1, S2
        score = 1
    else: # L2 S3 and all L3-L5 Statements
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
    # (ฟังก์ชันนี้เหมือนเดิม)
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


# -----------------------------------------------------------
# --- Action Plan Generation Logic (General for all Enablers) ---
# -----------------------------------------------------------

def get_sub_criteria_data(sub_id: str, criteria_list: List[Dict]) -> Dict:
    """ดึงข้อมูล Rubric เฉพาะ Sub-Criteria ที่ต้องการจาก list ของ Statements (evidence_data)"""
    for criteria in criteria_list:
        if criteria.get('Sub_Criteria_ID') == sub_id:
            return criteria
    return {}

def generate_action_plan_for_sub(sub_id: str, summary_data: Dict, full_rubric_data: List[Dict]) -> List[Dict]:
    """
    สร้าง Action Plan อัตโนมัติจากผลลัพธ์ Summary Assessment โดยใช้ Rubric Data
    [GENERALIZED FOR ALL ENABLERS, with specific BEST PRACTICE for 1.1 L2]
    """
    
    # ------------------ 1. ตรวจสอบ Gap และหา Target Level ------------------
    if not summary_data.get('development_gap'):
        return [{"Phase": "No Action Needed", "Goal": f"Sub-Criteria {sub_id} มีระดับวุฒิภาวะสูงสุดแล้ว (L5) หรือไม่มี Gap ที่ชัดเจน"}]
        
    highest_full_level = summary_data['highest_full_level']
    target_level = highest_full_level + 1
    
    if target_level > 5:
        return [{"Phase": "L5 Maturity Maintenance", "Goal": "มุ่งเน้นการรักษาความสม่ำเสมอ การสร้างนวัตกรรม และการปรับปรุงอย่างยั่งยืน"}]

    criteria_data = get_sub_criteria_data(sub_id, full_rubric_data)
    if not criteria_data:
        return [{"Phase": "Error", "Goal": f"ไม่พบข้อมูล Rubric/Statements สำหรับ {sub_id}"}]
        
    target_statements_key = f"Level_{target_level}_Statements"
    target_statements = criteria_data.get(target_statements_key, [])
    
    action_plan = []
    
    # ------------------ 2. PHASE 1: GAP CLOSURE (สร้าง Action หลัก) ------------------
    action_plan.append({
        "Phase": f"1. Strategic Gap Closure (Level {target_level})",
        "Goal": f"บรรลุหลักฐานที่จำเป็นทั้งหมดใน Level {target_level} ของ {sub_id}",
        "Target_Pass_Ratio": f"1.0 (ปัจจุบัน: {summary_data['pass_ratios'].get(str(target_level), 0.0)})",
        "Actions": []
    })
    
    # Logic การสร้าง Action Item แบบ General
    for i, statement in enumerate(target_statements):
         if not statement.strip(): 
             continue
             
         recommendation = f"จัดหาหรือสร้าง Evidence ที่สอดคล้องกับ Statement นี้ เพื่อให้บรรลุ Level {target_level}"
         target_evidence_type = "Required Documentation/Process Improvement"
         
         # *** Specific BEST PRACTICE OVERRIDE for L2/1.1 Governance Gap ***
         if sub_id == "1.1" and target_level == 2:
             if "แต่งตั้งคณะทำงาน" in statement or "มอบหมาย" in statement:
                 recommendation = "จัดทำ/รวบรวม **'คำสั่ง/ประกาศแต่งตั้งคณะทำงาน'** อย่างเป็นทางการ ที่ระบุอำนาจหน้าที่ชัดเจน และนำเข้าสู่ Evidence Store"
                 target_evidence_type = "Governance Document (Official Appointment)"
             elif "บทบาทหน้าที่ชัดเจน" in statement or "ขอบเขต" in statement:
                 recommendation = "ทบทวน/จัดทำ **'ขอบเขตอำนาจหน้าที่ (ToR)'** ของคณะทำงานให้ชัดเจนและเป็นทางการ"
                 target_evidence_type = "Formal ToR Document"
             elif "ประชุมต่อเนื่อง" in statement or "ขับเคลื่อน" in statement:
                 recommendation = "จัดให้มีการประชุมคณะทำงานอย่างสม่ำเสมอ และจัดเก็บ **'รายงานการประชุม (MoM)'** ที่แสดงความต่อเนื่องในการขับเคลื่อน"
                 target_evidence_type = "Meeting Minutes/MoM Logs"
         
         # --------------------------------------------------------------------------
         
         action_plan[-1]['Actions'].append({
             "Statement": f"L{target_level} S{i+1}: {statement}",
             "Recommendation": recommendation,
             "Target_Evidence_Type": target_evidence_type
         })

    # ------------------ 3. PHASE 2: AI Verification & Consolidation ------------------
    action_plan.append({
        "Phase": "2. AI Validation & Maintenance",
        "Goal": "ยืนยันการ Level-Up และรักษาความต่อเนื่องของหลักฐาน",
        "Actions": [
            {
                "Recommendation": "หลังรวบรวมหลักฐานใหม่ทั้งหมดแล้ว ให้นำเข้า Vector Store และรัน AI Assessment ในโหมด **FULLSCOPE** อีกครั้งเพื่อยืนยันว่า Level ที่ต้องการผ่านเกณฑ์",
                "Key_Metric": f"Overall Score ของ {sub_id} ต้องเพิ่มขึ้นและ Highest Full Level ต้องเป็น L{target_level}"
            }
        ]
    })
    
    return action_plan


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
            # NOTE: ต้องมั่นใจว่า Class EnablerAssessment ถูก import มาจาก assessments.enabler_assessment
            temp_loader = EnablerAssessment(enabler_abbr=args.enabler, vectorstore_retriever=None)
            evidence_mapping = temp_loader.evidence_mapping_data
            
            target_collection_names = None
            
            # 1. รวบรวมชื่อไฟล์ทั้งหมดที่อยู่ใน Evidence Mapping ของ Enabler นี้ 
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
                target_collection_names = base_enabler_files
                
                logger.info(f"⚡ Loading ALL Vectorstore Collections for {args.enabler} ({len(target_collection_names)} documents).")
                logger.info(f"   RAG Search Mode: FULL SCOPE SEARCH ({args.enabler.upper()} ONLY)")
                
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
        use_retrieval_filter=args.filter,
        target_sub_id=args.sub if args.sub != "all" else None
    )
    
    print(f"✅ Loaded {len(filtered_evidence)} Statements (Filtered to {args.sub}) for assessment of ENABLER: {args.enabler.upper()}.")
    
    
    # 5. Run Assessment
    start_time_assessment = time.perf_counter() 
    
    try:
        if args.mode == "mock":
            # 🟢 ต้องแทนที่ฟังก์ชัน retrieve_context ด้วย MOCK ด้วย
            # Note: _retrieve_context เป็น internal method ของ Class EnablerAssessment
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

    
    # 6. Output Summary (Print to terminal)
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


    # 7. NEW: GENERATE ACTION PLAN AND MERGE INTO SUMMARY
    print("\n\n=====================================================")
    print("        GENERATING ACTION PLAN...")
    print("=====================================================")

    try:
        all_action_plans: Dict[str, List] = {}
        # ใช้ enabler_loader.evidence_data ซึ่งเป็น list ของ Dictionaries ที่มี Rubric Statements
        full_rubric_data = enabler_loader.evidence_data 
        
        # Loop ผ่าน Sub-Criteria ที่ถูกประเมินในรอบนี้
        for sub_id, summary_data in summary['SubCriteria_Breakdown'].items():
            
            # สร้าง Action Plan สำหรับ Sub-Criteria นี้
            action_plan_data = generate_action_plan_for_sub(sub_id, summary_data, full_rubric_data)
            all_action_plans[sub_id] = action_plan_data

            # 🚨 NEW: Print Action Plan to Terminal for immediate viewing
            print(f"\n--- ACTION PLAN FOR {args.enabler.upper()} - {sub_id} ---")
            print(json.dumps(action_plan_data, indent=4, ensure_ascii=False))
        
        # 🚨 CRITICAL STEP: MERGE Action Plan data into the main summary dictionary
        summary['Action_Plans'] = all_action_plans
            
    except Exception as e:
        logger.error(f"❌ Failed to generate or merge Action Plan: {e}")
        # หากเกิด Error ในการสร้าง Action Plan ให้สร้าง Key เปล่าไว้เพื่อไม่ให้ export ล้มเหลว
        summary['Action_Plans'] = {"Error": str(e)}


    # 8. EXPORT FINAL JSON (รวม Summary และ Action Plan)
    if args.export:
        EXPORT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))
        try:
            os.makedirs(EXPORT_DIR, exist_ok=True)
            logger.info(f"Using export directory: {EXPORT_DIR}")
        except OSError as e:
            logger.error(f"❌ Failed to create results directory {EXPORT_DIR}: {e}")
            # ไม่ต้อง exit แต่ควร log error
        
        mode_suffix = "REAL" if args.mode == "real" else args.mode.upper()
        filter_suffix = "STRICTFILTER" if args.filter else "FULLSCOPE" 
        EXPORT_FILENAME = f"assessment_report_{args.enabler}_{args.sub}_{mode_suffix}_{filter_suffix}_{os.urandom(4).hex()}.json"
        FULL_EXPORT_PATH = os.path.join(EXPORT_DIR, EXPORT_FILENAME)

        try:
            # ใช้ summary ที่ถูกอัปเดตด้วย Action_Plans แล้ว
            with open(FULL_EXPORT_PATH, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=4)
            print(f"\n✅ Successfully exported Summary and Action Plan to {FULL_EXPORT_PATH}")
        except Exception as e:
            logger.error(f"❌ Failed to export JSON report to {FULL_EXPORT_PATH}: {e}")
            
    # 9. เรียกใช้ฟังก์ชันแสดงผลลัพธ์ละเอียด
    print_detailed_results(enabler.raw_llm_results)


    # Stop global timer and print total
    end_time_global = time.perf_counter()
    global_duration = end_time_global - start_time_global
    print(f"\n[⏱️ TOTAL EXECUTION TIME] All processes completed in: {global_duration:.2f} seconds.")
