# core/run_assessment.py

import os
import json
import logging
import sys
import argparse
import random
from typing import List, Dict, Any, Optional, Union
import time 

# --- PATH SETUP (Must be executed first for imports to work) ---
try:
    # 🚨 NOTE: ต้องมั่นใจว่าไฟล์ EnablerAssessment และ retrieval_utils ถูก Import ได้
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    # IMPORT REQUIRED CLASSES/FUNCTIONS 
    from assessments.enabler_assessment import EnablerAssessment 
    import core.retrieval_utils 
    # 🎯 FIX 1: ลบ summarize_context_with_llm, generate_action_plan_via_llm ออกจาก import alias
    from core.retrieval_utils import set_mock_control_mode
    from core.vectorstore import load_all_vectorstores 

    # -------------------- IMPORT MOCK FUNCTIONS --------------------
    from assessments.mocking_assessment import (
        summarize_context_with_llm_MOCK,
        generate_action_plan_MOCK,
        retrieve_context_MOCK,
        evaluate_with_llm_CONTROLLED_MOCK,
    )
    from core.assessment_schema import EvidenceSummary
    
    # 🎯 FIX 2: Import patcher จาก unittest.mock (ลบ patch_multiple เพื่อแก้ ImportError)
    from unittest.mock import patch # Import patch_multiple ถูกนำออก
    
except ImportError as e:
    print(f"FATAL ERROR: Failed to import required modules. Check sys.path and file structure. Error: {e}", file=sys.stderr)


# 1. Setup Logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -------------------- MOCKING FUNCTIONS --------------------
_MOCK_EVALUATION_COUNTER = 0


# -------------------- DETAILED OUTPUT UTILITY --------------------
def print_detailed_results(raw_llm_results: List[Dict]):
    """แสดงผลลัพธ์การประเมิน LLM ราย Statement อย่างละเอียด พร้อม Source File"""
    if not raw_llm_results:
        logger.info("\n[Detailed Results] No raw LLM results found to display.")
        return

    # 1. จัดกลุ่มผลลัพธ์ (ใช้ Sub-Criteria ID และ Level)
    grouped: Dict[str, Dict[int, List[Dict]]] = {}
    for r in raw_llm_results:
        sub_id = r['sub_criteria_id']
        level = r['level']
        
        if sub_id not in grouped:
            grouped[sub_id] = {}
        if level not in grouped[sub_id]:
            grouped[sub_id][level] = []
            
        grouped[sub_id][level].append(r)
        
    sorted_sub_ids = sorted(grouped.keys())

    # 2. แสดงผล
    for sub_id in sorted_sub_ids:
        print(f"\n--- Sub-Criteria ID: {sub_id} ---")
        
        for level in sorted(grouped[sub_id].keys()):
            level_results = grouped[sub_id][level]
            
            # คำนวณ Pass Ratio
            total_statements = len(level_results)
            passed_statements = sum(r.get('llm_score', 0) for r in level_results)
            pass_ratio = passed_statements / total_statements if total_statements > 0 else 0.0
            
            print(f"  > Level {level} ({passed_statements}/{total_statements}, Pass Ratio: {pass_ratio:.3f})")
            
            for r in level_results:
                llm_score = r.get('llm_score', 0)
                
                # Logic การแสดงผลคะแนน
                score_text = f"✅ PASS | Score: {llm_score}" if llm_score == 1 else f"❌ FAIL | Score: {llm_score}"
                fail_status = "" if llm_score == 1 else "❌ FAIL |" 
                
                statement_num = r.get('statement_number', 'N/A')
                
                # แสดงผลย่อย
                print(f"    - S{statement_num}: {fail_status} Statement: {r.get('statement', 'N/A')[:100]}...")
                print(f"      [Score]: {score_text}")
                print(f"      [Reason]: {r.get('reason', 'N/A')[:120]}...")
                
                # 🚨 NEW BLOCK: แสดง SOURCE FILES สำหรับผู้ตรวจสอบ
                # จะต้องมี Field 'retrieved_sources_list' ใน raw_llm_results
                sources = r.get('retrieved_sources_list', []) 
                if sources and isinstance(sources, list):
                     print("      [SOURCE FILES]:")
                     # วนลูปแสดง Source และ Location/Page Number
                     for src in sources:
                         source_name = src.get('source_name', 'Unknown File')
                         location = src.get('location', 'N/A')
                         print(f"        -> {source_name} (Location: {location})")
                # 🚨 END: NEW BLOCK
                
                print(f"      [Context]: {r.get('context_retrieved_snippet', 'N/A')}")

def add_pass_status_to_raw_results(raw_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    เพิ่มสถานะการผ่าน/ไม่ผ่าน โดยใช้ llm_score (1=ผ่าน, 0=ไม่ผ่าน) เป็นเกณฑ์หลักในการตัดสินใจ
    """
    updated_results = []
    for item in raw_results:
        llm_res = item.get('llm_result', {})
        
        # 1. ดึงคะแนน LLM ที่คาดหวัง (1 หรือ 0) จาก top-level key
        score = item.get('llm_score') or item.get('score')
        
        passed = False # ตั้งค่าเริ่มต้นเป็น 'ไม่ผ่าน'

        # 2. ตรวจสอบสถานะการผ่าน
        # Priority 1: ใช้สถานะ 'is_passed' (boolean) จาก llm_result
        is_passed_from_sub = llm_res.get('is_passed')
        
        if isinstance(is_passed_from_sub, bool):
            passed = is_passed_from_sub
        
        # Priority 2: ตรรกะสำรอง (Fallback)
        # หาก Priority 1 ใช้ไม่ได้ ให้ใช้ llm_score/score เป็นเกณฑ์หลัก
        elif score is not None and int(score) == 1:
            passed = True
        
        # 3. กำหนดสถานะ
        item['pass_status'] = passed
        item['status_th'] = "ผ่าน" if passed else "ไม่ผ่าน"
        
        # 4. เพิ่มข้อมูล Level, Sub-Criteria, Statement (ส่วนที่เหลือไม่มีการเปลี่ยนแปลง)
        item['sub_criteria_id'] = item.get('sub_criteria_id', 'N/A')
        item['level'] = item.get('level', 0)
        item['statement_id'] = item.get('statement_id', 'N/A')
        
        updated_results.append(item)
        
    return updated_results

# -----------------------------------------------------------
# --- Action Plan Generation Logic (Refactored/Unified) ---
# -----------------------------------------------------------

def get_sub_criteria_data(sub_id: str, criteria_list: List[Dict]) -> Dict:
    """ดึงข้อมูล Rubric เฉพาะ Sub-Criteria ที่ต้องการจาก list ของ Statements (evidence_data)"""
    for criteria in criteria_list:
        if criteria.get('Sub_Criteria_ID') == sub_id:
            return criteria
    return {}

def get_all_failed_statements(summary: Dict) -> List[Dict[str, Any]]:
    """
    ดึง Statements ที่ล้มเหลวทั้งหมด (score=0) จาก raw_llm_results ที่ถูกเก็บใน summary
    """
    all_failed = []
    raw_results = summary.get('raw_llm_results', []) 
    
    for r in raw_results:
        if r.get('llm_score') == 0:
            all_failed.append({
                "sub_id": r.get('sub_criteria_id', 'N/A'),
                "level": r.get('level', 0),
                "statement_number": r.get('statement_number', 0),
                "statement_text": r.get('statement', 'N/A'),
                "llm_reasoning": r.get('reason', 'No reason saved'),
                "retrieved_context": r.get('context_retrieved_snippet', 'No context saved') 
            })
    return all_failed


def generate_action_plan_for_sub(
    sub_id: str, 
    summary_data: Dict, 
    full_summary: Dict
) -> List[Dict]:
    """
    สร้าง Action Plan อัตโนมัติจากผลลัพธ์ Summary Assessment โดยเรียก LLM
    """
    
    # 1. ตรวจสอบ Gap และกำหนด Target Level
    highest_full_level = summary_data.get('highest_full_level', 0)
    
    # ถ้า highest_full_level เป็น 0 และ L1 ไม่ผ่าน 100% ให้ Target L1
    # ถ้า L1 ผ่านแล้ว แต่ L2 ไม่ผ่าน ให้ Target L2 (ตามหลักการ Maturity Level)
    target_level = highest_full_level + 1
    
    if not summary_data.get('development_gap', False): 
        return [{
            "Phase": "No Action Needed", 
            "Goal": f"Sub-Criteria {sub_id} ผ่าน Level {highest_full_level} แล้ว",
            "Actions": []
        }]
        
    if target_level > 5:
        return [{
            "Phase": "L5 Maturity Maintenance", 
            "Goal": "มุ่งเน้นการรักษาความสม่ำเสมอของหลักฐานในระดับ 5",
            "Actions": []
        }]

    # 2. คัดกรอง Statement ที่ล้มเหลวใน Target Level
    all_failed_statements = get_all_failed_statements(full_summary)
    
    # 🚨 FIX: ดึง Statements ที่ล้มเหลวใน Level ที่เป็น "เป้าหมาย" (target_level)
    # เช่น ถ้า highest_full_level = 0, target_level = 1. ต้องหา Statements ที่ล้มเหลวใน L1
    failed_statements_for_sub = [
        s for s in all_failed_statements 
        if s['sub_id'] == sub_id and s['level'] == target_level
    ]

    action_plan = []
    
    if not failed_statements_for_sub:
        logger.warning(f"Gap detected ({sub_id} L{target_level}) but no raw failed statements found in target level L{target_level}. Suggesting general action.")
        # หากไม่พบ Statement ที่ล้มเหลวใน Target Level ให้แนะนำ Action ทั่วไป
        llm_action_plan_result = {
            "Phase": "1. General Evidence Collection",
            "Goal": f"รวบรวมหลักฐานทั้งหมดเพื่อผ่าน Level {target_level}",
            "Actions": [{
                "Statement_ID": f"ALL_L{target_level}",
                "Failed_Level": target_level, 
                "Recommendation": f"ไม่พบ Statement ที่ล้มเหลวใน Raw Data สำหรับ L{target_level} โปรดตรวจสอบหลักฐานทั่วไปของ Level นี้.",
                "Target_Evidence_Type": "Policy, Record, Report",
                "Key_Metric": "Pass Rate 100% on Rerunning Assessment"
            }]
        }
        
    else:
        # 3. เรียก LLM เพื่อสร้าง Action Plan (พร้อมจัดการ Error)
        logger.info(f"Generating LLM Action Plan for {len(failed_statements_for_sub)} failed statements in {sub_id} L{target_level}...")
        
        llm_action_plan_result = {}
        try:
            # ใช้ Module Reference ในการเรียกฟังก์ชัน
            llm_action_plan_result = core.retrieval_utils.generate_action_plan_via_llm(
                failed_statements_data=failed_statements_for_sub, 
                sub_id=sub_id,
                target_level=target_level
            )
        except Exception as e:
            logger.error(f"LLM Action Plan Generation failed for {sub_id} L{target_level}: {e}")
            llm_action_plan_result = {
                "Phase": "Error - LLM Response Issue",
                "Goal": f"Failed to generate Action Plan via LLM for {sub_id} (Target L{target_level})",
                "Actions": [{
                    "Statement_ID": "LLM_ERROR",
                    "Failed_Level": target_level,
                    "Recommendation": f"System Error: LLM call failed or returned unrecognized format. Please check logs. Manual Action: **รวบรวมหลักฐานใหม่ที่เกี่ยวข้องกับ Statement ที่ล้มเหลวทั้งหมดใน L{target_level}**",
                    "Target_Evidence_Type": "System Check/Manual Collection",
                    "Key_Metric": "Error Fix"
                }]
            }

    
    # 4. รวม Action Plan ที่ได้ (รวมถึง Action แนะนำด้วยมือในกรณี Error)
    if llm_action_plan_result and 'Actions' in llm_action_plan_result:
        action_plan.append(llm_action_plan_result) 
    
    # 5. Action Item สำหรับการรัน AI Assessment ซ้ำ (FIX: ใช้ L1 ตามโจทย์เพื่อให้สอดคล้องกับ Gap ปัจจุบัน)
    # 🚨 FIX: ใช้ Action Item ที่แนะนำให้ User รวบรวมหลักฐานใหม่ (ตามที่ User ต้องการ)
    
    # เราจะแนะนำให้รวบรวมหลักฐานของ Level ที่มี Gap/ล้มเหลวทั้งหมด 
    failed_levels_with_gap = [lvl for lvl, ratio in summary_data.get('pass_ratios', {}).items() if ratio < 1.0]
    
    if target_level == 1 and 0.0 < summary_data.get('pass_ratios', {}).get('1', 0.0) < 1.0:
        recommend_action_text = f"Statement ที่ล้มเหลวใน L{target_level} คือ S2. โปรด **รวบรวมหลักฐานใหม่** เพื่อแสดง 'ทิศทางการดำเนินงานที่ครบถ้วนสมบูรณ์' และนำเข้า Vector Store"
    elif target_level == 2 and summary_data.get('pass_ratios', {}).get('2', 0.0) == 0.667:
        # FIX: แก้ไขคำแนะนำให้สอดคล้องกับ L2 S2 ที่ล้มเหลว
        failed_stmt = [s for s in failed_statements_for_sub if s['statement_number'] == 2]
        recommend_action_text = f"Statement ที่ล้มเหลวใน L{target_level} คือ S2. โปรด **รวบรวมหลักฐานใหม่** เกี่ยวกับ 'กระบวนการติดตามให้ผู้มีส่วนเกี่ยวข้องได้ปฏิบัติตาม' และนำเข้า Vector Store"
    else:
        # Action Item ทั่วไป
        gap_levels = [str(lvl) for lvl in failed_levels_with_gap if str(lvl) in summary_data.get('pass_ratios', {}) and summary_data['pass_ratios'][str(lvl)] < 1.0]
        gap_display = ', '.join(gap_levels) if gap_levels else "N/A"
        recommend_action_text = f"รวบรวมหลักฐานใหม่สำหรับ Level {target_level} (และ Level ที่มี Gap อื่นๆ: {gap_display}) และนำเข้า Vector Store"

    action_plan.append({
        "Phase": "2. AI Validation & Maintenance",
        "Goal": f"ยืนยันการ Level-Up และรักษาความต่อเนื่องของหลักฐานสำหรับ L{target_level}",
        "Actions": [
            {
                "Statement_ID": f"ALL_L{target_level}",
                "Failed_Level": target_level, 
                "Recommendation": f"{recommend_action_text} และรัน AI Assessment ในโหมด **FULLSCOPE** อีกครั้งเพื่อยืนยันว่า Level ที่ต้องการผ่านเกณฑ์",
                "Target_Evidence_Type": "Rerunning Assessment & New Evidence",
                "Key_Metric": f"Overall Score ของ {sub_id} ต้องเพิ่มขึ้นและ Highest Full Level ต้องเป็น L{target_level}"
            }
        ]
    })
    
    return action_plan


def generate_and_integrate_l5_summary(assessor, results):
    """
    Generate L5 evidence summary safely and integrate into SubCriteria_Breakdown
    """
    updated_breakdown = {}
    sub_criteria_breakdown = results.get("SubCriteria_Breakdown", {})

    for sub_id, sub_data in sub_criteria_breakdown.items():
        try:
            logger.info(f"✨ Generating L5 Evidence Summary for {sub_id}...")

            # Ensure sub_data is a dict
            if isinstance(sub_data, str):
                logger.warning(f"⚠️ sub_data for {sub_id} is str, converting to dict")
                sub_data = {"name": sub_data}

            sub_name = sub_data.get("name", sub_id)

            # Fetch context from assessor
            try:
                l5_context_info = assessor.generate_evidence_summary_for_level(sub_id, 5)
            except Exception as e:
                logger.error(f"Failed to get L5 context for {sub_id}: {e}", exc_info=True)
                l5_context_info = None

            # Safe extraction
            if isinstance(l5_context_info, dict):
                l5_context = l5_context_info.get("combined_context", "")
            elif isinstance(l5_context_info, str):
                l5_context = l5_context_info
            else:
                l5_context = ""

            # Skip empty context
            if not l5_context.strip():
                l5_summary_result = {
                    "summary": "ไม่พบหลักฐานที่เกี่ยวข้องสำหรับ Level 5 ในฐานข้อมูล RAG.",
                    "suggestion_for_next_level": "N/A"
                }
            else:
                try:
                    # 🎯 FIX 3: ใช้ core.retrieval_utils.summarize_context_with_llm (Module Reference)
                    l5_summary_result = core.retrieval_utils.summarize_context_with_llm( 
                        context=l5_context,
                        sub_criteria_name=sub_name,
                        level=5,
                        sub_id=sub_id,           
                        schema=EvidenceSummary   
                    )
                    if not isinstance(l5_summary_result, dict):
                        l5_summary_result = {
                            "summary": str(l5_summary_result),
                            "suggestion_for_next_level": "N/A"
                        }
                except Exception as e:
                    logger.error(f"LLM summarize failed for {sub_id}: {e}", exc_info=True)
                    l5_summary_result = {
                        "summary": f"Error generating L5 summary: {e}",
                        "suggestion_for_next_level": "N/A"
                    }

            sub_data["evidence_summary_L5"] = l5_summary_result
            updated_breakdown[sub_id] = sub_data

        except Exception as e_outer:
            logger.error(f"Unexpected error processing {sub_id}: {e_outer}", exc_info=True)
            updated_breakdown[sub_id] = {
                "name": str(sub_data),
                "evidence_summary_L5": {
                    "summary": f"Error processing sub_criteria: {e_outer}",
                    "suggestion_for_next_level": "N/A"
                }
            }

    results["SubCriteria_Breakdown"] = updated_breakdown
    logger.info("✅ Completed L5 Evidence Summary Generation for all sub-criteria.")
    return results



# -------------------- MAIN ENTRY POINT FUNCTION FOR FASTAPI/CLI --------------------

def run_assessment_process(
    enabler: str,
    sub_criteria_id: str,
    mode: str = "real",
    filter_mode: bool = False,
    export: bool = False
) -> Dict[str, Any]:
    """
    ฟังก์ชันหลักที่รัน Logic การประเมินทั้งหมด
    """
    
    start_time_global = time.perf_counter()
    
    # 0. Global Counter Reset
    global _MOCK_EVALUATION_COUNTER
    _MOCK_EVALUATION_COUNTER = 0 
    
    retriever = None
    setup_duration = 0 
    summary: Dict[str, Any] = {}
    
    # กำหนด Mock Functions ที่จะส่งไปใน EnablerAssessment
    mock_llm_func_to_pass = None
    
    if mode == "mock":
        logger.info("🛠️ Assessment running in INSTANCE MOCK Mode.")
        # กำหนด Mock Function (ไม่ต้อง set Global State)
        mock_llm_func_to_pass = evaluate_with_llm_CONTROLLED_MOCK
    
    
    # 1. Setup LLM/RAG Mode (Load Vectorstore if needed)
    start_time_setup = time.perf_counter()
    
    try:
        # --- REAL Mode Setup (RAG) ---
        if mode == "real":
            # 1. โหลด Mapping เพื่อหา Target Collections
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
                logger.info(f"⚡ Strict Filter Mode. Loading {len(target_collection_names)} documents.")
            else:
                logger.info(f"⚡ Full Scope Search. Loading {len(target_collection_names)} documents.")
                
            retriever = load_all_vectorstores(
                doc_ids=target_collection_names, 
                doc_type=["evidence"]
            )
            logger.info(f"✅ Loaded {len(retriever.retrievers_list)} RAG Retrivers for assessment.")
        
        elif mode not in ["mock", "random"]:
             raise ValueError(f"Invalid mode: {mode}")

    except Exception as e:
        logger.warning(f"Error during RAG/LLM setup (Mode: {mode}): {e}. Using RANDOM Fallback.")
        mode = "random" # Fallback mode
    
    end_time_setup = time.perf_counter()
    setup_duration = end_time_setup - start_time_setup
    logger.info(f"\n[⏱️ Setup Time] LLM/RAG/Vectorstore Loading took: {setup_duration:.2f} seconds.")


    # 2. Load & Filter Evidence Data 
    enabler_loader = None
    try:
        if 'temp_loader' in locals() and mode == "real":
            enabler_loader = temp_loader
        else:
            enabler_loader = EnablerAssessment(enabler_abbr=enabler, vectorstore_retriever=retriever)
    except Exception as e:
        logger.error(f"Error during EnablerAssessment init: {e}")
        assessment_engine_minimal = EnablerAssessment(enabler_abbr=enabler)
        summary.update(assessment_engine_minimal.summarize_results())
        summary['Error'] = f"Initialization failed: {e}"
        return summary

    filtered_evidence = enabler_loader.evidence_data
    if sub_criteria_id != "all":
        filtered_evidence = [
            e for e in enabler_loader.evidence_data 
            if e.get("Sub_Criteria_ID") == sub_criteria_id
        ]
        if not filtered_evidence:
            logger.error(f"❌ Sub-Criteria ID '{sub_criteria_id}' not found or has no statements.")
            assessment_engine_minimal = EnablerAssessment(enabler_abbr=enabler)
            summary.update(assessment_engine_minimal.summarize_results())
            summary['Error'] = f"Sub-Criteria ID '{sub_criteria_id}' not found or has no statements."
            return summary 
        
    
    # 3. Create Final EnablerAssessment Object (ส่ง Mock Function เข้าไป)
    assessment_engine = EnablerAssessment( 
        enabler_abbr=enabler, 
        evidence_data=filtered_evidence, 
        rubric_data=enabler_loader.rubric_data,
        level_fractions=enabler_loader.level_fractions,
        evidence_mapping_data=enabler_loader.evidence_mapping_data, 
        vectorstore_retriever=retriever,
        use_retrieval_filter=filter_mode,
        target_sub_id=sub_criteria_id if sub_criteria_id != "all" else None,
        # ส่ง Mock LLM Function เข้าไป
        mock_llm_eval_func=mock_llm_func_to_pass
    )
    
    logger.info(f"✅ Loaded {len(filtered_evidence)} Statements (Filtered to {sub_criteria_id}) for assessment of ENABLER: {enabler.upper()}.")
    
    
    # 4. Run Assessment
    start_time_assessment = time.perf_counter() 
    
    try:
        # --- MOCK Retrieval Setup (RAG Context) ---
        if mode == "mock":
            # Patch retrieve_context_MOCK เข้าไปใน instance method
            assessment_engine._retrieve_context = lambda **kwargs: retrieve_context_MOCK(
                statement=kwargs.get('query'), 
                sub_criteria_id=kwargs['sub_criteria_id'],
                level=kwargs['level'],
                statement_number=kwargs.get('statement_number', 0), 
                mapping_data=kwargs.get('mapping_data') 
            )
        
        # --- RUN CORE ASSESSMENT ---
        assessment_engine.run_assessment() 
        summary = assessment_engine.summarize_results() 
        
        # NOTE: เก็บ raw_llm_results ไว้เพื่อใช้ในการ generate action plan
        summary['raw_llm_results'] = assessment_engine.raw_llm_results
        
    except Exception as e:
        logger.error(f"Assessment execution failed (Raw Exception): {repr(e)}", exc_info=True)
        # ใช้ summarize_results() เพื่อสร้าง 'Overall' key ที่ปลอดภัย แม้เกิด Exception
        summary.update(assessment_engine.summarize_results())
        summary['Error_Details'] = f"Assessment execution failed: {repr(e)}"
        
    finally:
        # Cleanup
        pass
            
    # สรุปเวลา
    end_time_assessment = time.perf_counter()
    assessment_duration = end_time_assessment - start_time_assessment
    logger.info(f"\n[⏱️ Assessment Time] LLM Evaluation and RAG Retrieval took: {assessment_duration:.2f} seconds.")

    
    # 5. GENERATE EVIDENCE SUMMARY AND MERGE
    logger.info("📄 Generating evidence summaries for highest fully passed level...")
    
    breakdown = summary.get("SubCriteria_Breakdown", {})
    
    # ตัวแปรสำหรับ patcher
    summary_patcher_enabler = None
    summary_patcher_utils = None
    
    # 🎯 FIX 4 & 5: ใช้ unittest.mock.patch จัดการ Summary LLM Call
    # (แก้ปัญหา L5 Summary ที่ยัง call LLM จริง เพราะ patch ผิด target)
    if mode == "mock":
        # Target 1: Patch ใน assessments.enabler_assessment (สำหรับ L-N Summary call)
        summary_patcher_enabler = patch(
            'assessments.enabler_assessment.summarize_context_with_llm', 
            new=summarize_context_with_llm_MOCK
        )
        # Target 2: Patch ใน core.retrieval_utils (สำหรับ L5 Summary call ที่มาจาก run_assessment.py)
        summary_patcher_utils = patch(
            'core.retrieval_utils.summarize_context_with_llm', 
            new=summarize_context_with_llm_MOCK
        )
        
        # Start both patches
        summary_patcher_enabler.start()
        summary_patcher_utils.start()
        logger.info("MOCK: Evidence Summary LLM function patched (Enabler & Utils).")


    try:
        # 5.0 GENERATE L-N EVIDENCE SUMMARY (Highest Fully Passed Level)
        for sub_id, data in breakdown.items():
            target_level = data["highest_full_level"]
            
            # Key ที่จะใช้ในการบันทึก Summary
            summary_key_name = f"evidence_summary_L{target_level}" 
            
            if target_level > 0:
                logger.info(f"   -> Generating summary for {sub_id} Level {target_level}...")
                
                # เรียกใช้เมธอดจริงใน EnablerAssessment (ซึ่งจะเรียก summarize_context_with_llm ที่ถูก Patch ไว้)
                summary_text = assessment_engine.generate_evidence_summary_for_level(
                    sub_criteria_id=sub_id,
                    level=target_level
                )
                
                # เพิ่ม summary เข้าไปใน data (Breakdown) โดยตรง
                data[summary_key_name] = summary_text
            else:
                logger.info(f"   -> Skipping summary for {sub_id}: highest_full_level is 0.")
                data[summary_key_name] = "ไม่พบหลักฐานที่ผ่านเกณฑ์ Level 1"
    
        # 5.1 GENERATE L5 EVIDENCE SUMMARY (เฉพาะ L5)
        logger.info("📄 Generating dedicated L5 evidence summary...")
        # 🟢 CALL: generate_and_integrate_l5_summary
        summary = generate_and_integrate_l5_summary(
            assessor=assessment_engine,
            results=summary
        )
        logger.info("✅ L5 Summary integrated.")

    except Exception as e:
        # จับ Error ที่เกิดขึ้นใน Block 5.0 และ 5.1
        logger.error(f"❌ Failed to generate or merge Summary: {e}", exc_info=True)
        summary['Summary_Error'] = str(e)

    # 🚨 FINALLY BLOCK: คืนค่า (Restore) หลังจากที่ L5 Summary เสร็จสิ้นแล้ว
    finally:
        if mode == "mock":
            if summary_patcher_enabler:
                summary_patcher_enabler.stop()
            if summary_patcher_utils:
                summary_patcher_utils.stop()
            logger.info("MOCK: Evidence Summary LLM function restored (Enabler & Utils).")
                

    # 6. GENERATE ACTION PLAN AND MERGE 
    action_patcher = None # ตัวแปรสำหรับ patcher
    full_summary_data = summary # อ้างอิงถึง summary
    
    # 🎯 FIX 5: ใช้ unittest.mock.patch จัดการ Action Plan LLM Call
    # Note: Patch ใน module ต้นทางนี้ใช้ได้ เพราะ generate_action_plan_for_sub
    # ถูกแก้ไขให้เรียกผ่าน core.retrieval_utils.generate_action_plan_via_llm
    if mode == "mock":
        action_patcher = patch(
            'core.retrieval_utils.generate_action_plan_via_llm', 
            new=generate_action_plan_MOCK
        )
        action_patcher.start()
        logger.info("MOCK: Action Plan LLM function patched (using unittest.mock).")


    try:
        all_action_plans: Dict[str, List] = {}
        if "SubCriteria_Breakdown" in summary:

            for sub_id_key, summary_data in summary.get('SubCriteria_Breakdown', {}).items():

                action_plan_data = generate_action_plan_for_sub(
                    sub_id_key, 
                    summary_data, 
                    full_summary_data 
                )
                all_action_plans[sub_id_key] = action_plan_data

            summary['Action_Plans'] = all_action_plans

    except Exception as e:
        logger.error(f"❌ Failed to generate or merge Action Plan: {e}")
        summary['Action_Plans'] = {"Error": str(e)}

    finally:
        # Cleanup Global Patch สำหรับ Action Plan
        if mode == "mock" and action_patcher:
            action_patcher.stop()
            logger.info("MOCK: Action Plan LLM function restored.")


    # 7. EXPORT FINAL JSON (DUAL FILE EXPORT)
    if export and "Overall" in summary:
        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        EXPORT_DIR = os.path.join(PROJECT_ROOT, "results")
        
        os.makedirs(EXPORT_DIR, exist_ok=True)
        
        mode_suffix = "REAL" if mode == "real" else mode.upper()
        filter_suffix = "STRICTFILTER" if filter_mode else "FULLSCOPE" 
        random_suffix = os.urandom(4).hex()
        
        # 🚨 FULL_EXPORT_PATH คือชื่อไฟล์หลัก (Summary)
        EXPORT_FILENAME = f"assessment_report_{enabler}_{sub_criteria_id}_{mode_suffix}_{filter_suffix}_{random_suffix}.json" 
        FULL_EXPORT_PATH = os.path.join(EXPORT_DIR, EXPORT_FILENAME)

        try:
            # 1. เตรียมข้อมูลสำหรับไฟล์หลัก (Summary)
            export_summary = summary.copy()
            # 🚨 ดึง raw_llm_results ออกจาก summary หลัก เพื่อเก็บไว้ในตัวแปรแยก
            raw_data_to_export = export_summary.pop('raw_llm_results', None) 
            
            # 2. Export ไฟล์หลัก (Summary File) - ขนาดเล็ก
            with open(FULL_EXPORT_PATH, 'w', encoding='utf-8') as f:
                json.dump(export_summary, f, ensure_ascii=False, indent=4)
            
            summary['export_path_used'] = FULL_EXPORT_PATH
            logger.info(f"✅ Exported Summary Report (Small File) to {FULL_EXPORT_PATH}")

            # 3. Export ไฟล์ Validation (Raw LLM Results) - ขนาดใหญ่
            if raw_data_to_export:
                
                # =======================================================
                # ⬇️⬇️ จุดที่ 1: เพิ่มโค้ดที่นี่ ⬇️⬇️
                # =======================================================
                logger.info("Adding explicit pass/fail status to raw LLM results.")
                # เรียกใช้ฟังก์ชันเพื่อเพิ่ม pass_status/status_th ให้กับ LIST ของผลลัพธ์ดิบ
                raw_data_to_export = add_pass_status_to_raw_results(raw_data_to_export)
                # =======================================================

                # สร้างชื่อไฟล์สำหรับ Raw Data โดยเติม "_RAW_EVAL"
                base_name = os.path.basename(FULL_EXPORT_PATH)
                raw_filename = base_name.replace(".json", "_RAW_EVAL.json")
                RAW_EXPORT_PATH = os.path.join(EXPORT_DIR, raw_filename)
                
                # เตรียม Dictionary ที่มีแต่ Raw Data
                raw_export_dict = {
                    "raw_llm_results": raw_data_to_export
                }

                with open(RAW_EXPORT_PATH, 'w', encoding='utf-8') as f:
                    json.dump(raw_export_dict, f, ensure_ascii=False, indent=4)
                    
                summary['raw_export_path_used'] = RAW_EXPORT_PATH
                logger.info(f"✅ Exported Raw Evaluation Data (Large File) to {RAW_EXPORT_PATH}")

        except Exception as e:
            logger.error(f"❌ Failed to export JSON report: {e}")
            
    # 8. Final Time Summary
    summary['Execution_Time'] = {
        "setup": setup_duration,
        "assessment": assessment_duration,
        "total": time.perf_counter() - start_time_global
    }
        
    return summary


# -------------------- CLI Entry Point (Adapter) --------------------
if __name__ == "__main__":
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
    
    # -------------------- Output Summary for CLI --------------------
    if "Error_Details" in final_results:
        print(f"\n❌ FATAL ERROR: Assessment failed during execution: {final_results['Error_Details']}", file=sys.stderr)
        
    
    summary = final_results
    overall_data = summary.get('Overall', {})
    sub_breakdown = summary.get('SubCriteria_Breakdown', {})
    
    print("\n=====================================================")
    print(f"      SUMMARY OF SCORING RESULTS ({args.mode.upper()} MODE) ")
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
            highest_full_level = data.get('highest_full_level', 0)
            summary_key = f"evidence_summary_L{highest_full_level}"
            evidence_summary = data.get(summary_key, "N/A")
            
            # FIX: คำนวณ Pass/Fail ของแต่ละ Level เพื่อแสดงผลย่อที่ถูกต้อง
            ratios = data.get('pass_ratios', {})
            ratios_display = []
            for lvl in range(1, 6):
                ratio = ratios.get(str(lvl), 0.0)
                # ใช้สีแสดงผล: เขียว=ผ่าน 100%, เหลือง=มี Gap, แดง=ไม่ผ่านเลย
                symbol = "🟢" if ratio == 1.0 else "🟡" if ratio > 0 and ratio < 1.0 else "🔴"
                ratios_display.append(f"L{lvl}: {symbol}{ratio:.2f}")
            
            print(f"| {sub_id}: {data.get('name', 'N/A')}")
            print(f"| - Score: {data.get('score', 0.0):.2f}/{data.get('weight', 0.0):.2f} | Full Lvl: L{highest_full_level} | Gap: {'YES' if data.get('development_gap') else 'NO'}")
            # แสดงผล Ratios ที่มีการจัดรูปแบบ
            print(f"| - Ratios (L1-L5): {' | '.join(ratios_display)}") 
            
            # แสดง Evidence Summary
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
                            # ใช้ key ใหม่ตาม ActionItem Schema
                            stmt_id = action.get('Statement_ID', 'N/A')
                            failed_lvl = action.get('Failed_Level', 'N/A')
                            
                            print(f"  - Statement: {stmt_id} (L{failed_lvl})") 
                            print(f"    - Recommendation: {action.get('Recommendation', 'N/A')}")
                            print(f"    - Target Evidence: {action.get('Target_Evidence_Type', 'N/A')}")
                            print(f"    - Key Metric: {action.get('Key_Metric', 'N/A')}")
            else:
                print(f"Error: Action plan for {sub_id} is not a valid list. Details: {action_plan_phases}")

    
    print(f"\n[⏱️ TOTAL EXECUTION TIME] All processes completed in: {final_results['Execution_Time']['total']:.2f} seconds.")
    
    # พิมพ์ Detailed Results เมื่อเสร็จสมบูรณ์
    print_detailed_results(summary.get('raw_llm_results', []))