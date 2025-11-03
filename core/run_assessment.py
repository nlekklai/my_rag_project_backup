import os
import json
import logging
import sys
import argparse
import random
from typing import List, Dict, Any, Optional, Union
import time
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

    # 🟢 NEW IMPORT: Import UUID extraction and retrieval by ID
    from core.llm_utils import extract_uuids_from_llm_response
    from core.retrieval_utils import retrieve_context_by_doc_ids

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
def print_detailed_results(raw_llm_results: List[Dict], target_sub_id: str, enabler_abbr: str):
    """
    แสดงผลลัพธ์การประเมิน LLM ราย Statement พร้อม Source File
    🛑 MODIFIED: เพิ่ม enabler_abbr เพื่อส่งต่อให้ _retrieve_full_source_info
    """
    if not raw_llm_results:
        logger.info("\n[Detailed Results] No raw LLM results found to display.")
        return

    # 🛑 NEW: ดึงข้อมูลไฟล์ต้นฉบับทั้งหมดจาก UUIDs ที่ LLM แนะนำ, passing enabler_abbr
    updated_raw_llm_results = _retrieve_full_source_info(raw_llm_results, enabler_abbr)

    grouped: Dict[str, Dict[int, List[Dict]]] = {}
    for r in updated_raw_llm_results:
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

                # 🟢 ใช้ sources จาก UUIDs ที่ถูกดึงมาใหม่
                sources = r.get('retrieved_full_source_info', []) # NEW KEY
                if sources:
                     print("      [SOURCE FILES]:")
                     for src in sources:
                         source_name = src.get('source_name', 'Unknown File')
                         location = src.get('location', 'N/A')
                         chunk_uuid = src.get('chunk_uuid', 'N/A') # 🟢 NEW: แสดง UUID ของ Chunk
                         uuid_short = chunk_uuid[:8] + "..." if chunk_uuid else "N/A"
                         print(f"        -> {source_name} (Location: {location}, UUID: {uuid_short})")
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

        # 🟢 NEW: Extract UUIDs from the LLM's raw response (if available)
        llm_raw_response = item.get('llm_raw_response_content', '')
        if llm_raw_response:
            # We look for UUIDs in any key, as the exact key name might vary 
            # (though 'evidence_uuids' or 'chunk_uuids' is expected)
            extracted_uuids = extract_uuids_from_llm_response(
                llm_raw_response, 
                key_hint=["chunk_uuids", "evidence_uuids", "doc_uuids"] # Keys LLM is instructed to use
            )
            item['llm_extracted_chunk_uuids'] = extracted_uuids
        else:
            item['llm_extracted_chunk_uuids'] = []

        updated_results.append(item)
    return updated_results

# -------------------- POST-ASSESSMENT RETRIEVAL --------------------

def _retrieve_full_source_info(raw_llm_results: List[Dict[str, Any]], enabler_abbr: str) -> List[Dict[str, Any]]:
    """
    🟢 NEW FUNCTION: Collects all unique chunk UUIDs and retrieves full document details 
    using the dedicated function, then merges the results back.
    🛑 MODIFIED: เพิ่ม enabler_abbr เพื่อใช้ในการกำหนด Collection Name
    """
    all_uuids = set()
    for r in raw_llm_results:
        uuids = r.get('llm_extracted_chunk_uuids')
        if uuids:
            all_uuids.update(uuids)

    if not all_uuids:
        logger.info("ℹ️ No Chunk UUIDs extracted from LLM results for post-retrieval.")
        return raw_llm_results

    logger.info(f"🔎 Attempting to retrieve full source info for {len(all_uuids)} unique chunks...")
    
    # 🛑 CONSTRUCT COLLECTION NAME: evidence_<ENABLER_ABBR_LOWER>
    collection_name = f"evidence_{enabler_abbr.lower()}"
    logger.info(f"Retrieving source info from collection: {collection_name}")
    
    # 🛑 Call the new function from retrieval_utils, passing the specific collection name
    try:
        retrieval_result = retrieve_context_by_doc_ids(list(all_uuids), collection_name) 
    except Exception as e:
        logger.error(f"Failed to retrieve full source info by doc ids: {e}", exc_info=True)
        return raw_llm_results
    
    full_docs_map: Dict[str, Dict[str, Any]] = {
        doc.get("chunk_uuid"): {
            "chunk_uuid": doc.get("chunk_uuid"),
            "doc_id": doc.get("doc_id"),
            "source_name": doc.get("source"), # Use 'source' field as the file name/source name
            "location": doc.get("doc_type"), # Use 'doc_type' or a similar field for location hint
            "content_snippet": (doc.get("content", "")[:100] + "...") if doc.get("content") else ""
        } for doc in retrieval_result.get("top_evidences", []) if doc.get("chunk_uuid")
    }
    
    # Merge results back into raw_llm_results
    updated_results = []
    for r in raw_llm_results:
        chunk_uuids = r.get('llm_extracted_chunk_uuids', [])
        source_info_list = []
        for uuid in chunk_uuids:
            if uuid in full_docs_map:
                source_info_list.append(full_docs_map[uuid])
        
        # 🟢 Store the full source info list
        r['retrieved_full_source_info'] = source_info_list
        updated_results.append(r)
        
    return updated_results


# -------------------- SUB CRITERIA UTILITIES & ACTION PLAN --------------------
def get_sub_criteria_data(sub_id: str, criteria_list: List[Dict]) -> Dict:
    """Finds the sub-criteria dictionary from the full list."""
    for criteria in criteria_list:
        if criteria.get('Sub_Criteria_ID') == sub_id:
            return criteria
    return {}

def get_all_failed_statements(summary: Dict) -> List[Dict[str, Any]]:
    """Extracts all statements with a score of 0 (Fail)."""
    all_failed = []
    raw_results = summary.get('raw_llm_results', []) 
    for r in raw_results:
        score_val = r.get('llm_score', r.get('score', 1))
        try:
            # Statements that failed (score 0)
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
    Generate L5 Summary และ Highest Full Level Summary 
    (จะเรียก generate_evidence_summary_for_level และ summarize_context_with_llm ซึ่งจะถูก Patch)

    🛑 MODIFIED: เพิ่มการสร้างและผสานรวม Summary สำหรับ Level ที่ผ่านครบสูงสุด (Highest Full Level)
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
def _export_results_to_json(
    summary: Dict[str, Any], 
    enabler_type: str, 
    sub_id: Optional[str] = None
) -> Dict[str, Optional[str]]:
    """
    จัดการการบันทึกไฟล์สรุปและข้อมูลดิบ โดยใช้ enabler_type และ sub_id 
    (ถ้ามี) ในการตั้งชื่อไฟล์ และคืนค่าพาธที่ใช้

    :param summary: ข้อมูลสรุปและข้อมูลดิบ (รวม raw_llm_results) ที่จะ export
    :param enabler_type: ประเภทของ Enabler (เช่น 'KM' หรือ 'HCR')
    :param sub_id: ID ของเกณฑ์ย่อย (เช่น '1.1') ถ้าเป็น None คือรันทั้งหมด
    :return: Dict ที่มีพาธไฟล์ที่ export แล้ว
    """
    export_paths = {
        'export_path_used': None,
        'raw_export_path_used': None
    }
    
    try:
        # 1. เตรียม Export Directory และ Timestamp
        # project_root ถูกกำหนดไว้ในส่วน PATH SETUP
        # NOTE: ตรวจสอบให้แน่ใจว่า project_root และ logger ถูกกำหนดไว้ก่อนเรียกใช้ฟังก์ชันนี้
        
        # 💡 สร้างชื่อส่วนขยาย scope_prefix
        scope_prefix = f"_{sub_id}" if sub_id else "_All"
        
        export_dir = os.path.abspath(os.path.join(project_root, "exports"))
        os.makedirs(export_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 2. Export Summary Report (ไม่รวม raw_llm_results)
        summary_to_save = {k: v for k, v in summary.items() if k != 'raw_llm_results'}
        
        # 💡 ปรับปรุงชื่อไฟล์: [TYPE]_[SUMMARY]_[SCOPE]_[TIMESTAMP].json
        # เช่น 'KM_summary_1.1_20251031_143851.json'
        summary_filename = f"{enabler_type}_summary{scope_prefix}_{timestamp}.json" 
        summary_file_path = os.path.join(export_dir, summary_filename)
        
        with open(summary_file_path, 'w', encoding='utf-8') as f:
            json.dump(summary_to_save, f, ensure_ascii=False, indent=4)
        export_paths['export_path_used'] = summary_file_path
        logger.info(f"Report successfully exported to: {summary_file_path}")
        
        # 3. Export Raw Evaluation Data
        raw_data = summary.get('raw_llm_results', [])
        if raw_data:
            # 💡 ปรับปรุงชื่อไฟล์: [TYPE]_[RAW_DETAILS]_[SCOPE]_[TIMESTAMP].json
            # เช่น 'KM_raw_details_1.1_20251031_143851.json'
            raw_filename = f"{enabler_type}_raw_details{scope_prefix}_{timestamp}.json" 
            raw_file_path = os.path.join(export_dir, raw_filename)
            
            # 🛑 NOTE: บันทึกเฉพาะ raw_llm_results array
            with open(raw_file_path, 'w', encoding='utf-8') as f:
                json.dump(raw_data, f, ensure_ascii=False, indent=4)
            export_paths['raw_export_path_used'] = raw_file_path
            logger.info(f"Raw evaluation data successfully exported to: {raw_file_path}")
            
    except Exception as e:
        logger.error(f"❌ ERROR during file export: {e}", exc_info=True)
        
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
    external_retriever: Optional[Any] = None # 🟢 [REVISED] เพิ่ม Argument สำหรับ FastAPI
) -> Dict[str, Any]:
    start_time_global = time.perf_counter()
    summary: Dict[str, Any] = {'raw_export_path_used': None}
    original_mode = mode
    retriever = external_retriever # 🟢 [REVISED] ใช้ external_retriever ที่ส่งเข้ามา

    # -------------------- Mock Setup --------------------
    set_mock_control_mode(original_mode == "mock")
    llm_eval_func = evaluate_with_llm_CONTROLLED_MOCK if original_mode=="mock" else None
    llm_summarize_func = summarize_context_with_llm_MOCK if original_mode=="mock" else None
    llm_action_plan_func = generate_action_plan_MOCK if original_mode=="mock" else None

    # -------------------- Load Vectorstore --------------------
    try:
        if mode == "real" and external_retriever is None:
            # 💡 [FALLBACK] โหลด Vector Store เองเมื่อรันผ่าน CLI/Local (ถ้า external_retriever เป็น None)
            logger.warning("⚠️ Running in REAL mode without external retriever. Loading vector store inside function (slow).")
            
            temp_loader = EnablerAssessment(enabler_abbr=enabler, vectorstore_retriever=None)
            evidence_mapping_data = temp_loader.evidence_mapping_data
            if evidence_mapping_data is None:
                evidence_mapping_data = {}

            # Apply filter_mode doc_ids if requested
            file_ids_to_load = []
            if filter_mode and sub_criteria_id != "all":
                for key, data in evidence_mapping_data.items():
                    if key.startswith(f"{sub_criteria_id}_L"):
                        for ev in data.get('evidences', []):
                            doc_id = ev.get('doc_id')
                            if doc_id:
                                file_ids_to_load.append(doc_id)
                logger.info(f"DEBUG: doc_ids to load for {sub_criteria_id}: {file_ids_to_load}")

            # Load retriever with filter_doc_ids (เฉพาะ doc_ids ที่ต้องการ)
            retriever = load_all_vectorstores(
                doc_types=["evidence"],
                evidence_enabler=enabler.lower(),
                doc_ids=file_ids_to_load if file_ids_to_load else None
            )

            target_collection_names = [f"evidence_{enabler.lower()}"]
            logger.info(f"✅ Vectorstore loaded for enabler {enabler}. Collections: {target_collection_names}")
        
        elif mode == "real" and external_retriever is not None:
             logger.info("✅ Using external retriever provided by API/Caller.")
             # retriever ถูกกำหนดไว้แล้วจาก external_retriever
        
        elif mode != "real":
             retriever = None

    except Exception as e:
        logger.error(f"❌ ERROR: Failed to load Vectorstores in REAL mode: {e}", exc_info=True)
        mode = "random"
        logger.warning(f"⚠️ MODE CHANGED TO: {mode.upper()} due to Vectorstore Load Failure.")

    # -------------------- Load & Filter Evidence --------------------
    try:
        # 💡 [REVISED] ใช้ temp_loader ที่สร้างมาใน Fallback หรือสร้าง EnablerAssessment ใหม่
        if 'temp_loader' in locals() and mode=="real" and external_retriever is None:
            enabler_loader = temp_loader
        else:
            enabler_loader = EnablerAssessment(enabler_abbr=enabler, vectorstore_retriever=retriever)
        
        filtered_evidence = enabler_loader.evidence_data
        if sub_criteria_id != "all":
            filtered_evidence = [e for e in filtered_evidence if e.get("Sub_Criteria_ID")==sub_criteria_id]

        # Strict Filter Mode: remove statements without mapped evidence
        if filter_mode and sub_criteria_id != "all":
            evidence_mapping_data = enabler_loader.evidence_mapping_data
            valid_level_keys = {k for k, v in evidence_mapping_data.items() if k.startswith(sub_criteria_id) and v.get('evidences')}
            statements_to_assess = []
            skipped_statements = 0

            for statement_dict in filtered_evidence:
                added = False
                for lvl in range(1, 6):
                    level_key = f"{statement_dict['Sub_Criteria_ID']}_L{lvl}"
                    level_statements = statement_dict.get(f"Level_{lvl}_Statements", [])
                    if level_statements and level_key in valid_level_keys:
                        statements_to_assess.append(statement_dict)
                        added = True
                        break
                if not added:
                    skipped_statements += 1

            filtered_evidence = statements_to_assess
            logger.info(f"DEBUG: Statements after Strict Filter: {len(filtered_evidence)} (Skipped: {skipped_statements})")

    except Exception as e:
        summary.update(EnablerAssessment(enabler_abbr=enabler).summarize_results())
        summary['Error'] = str(e)
        summary['mode_used'] = mode
        return summary

    # -------------------- Create Assessment Engine --------------------
    assessment_engine = EnablerAssessment(
        enabler_abbr=enabler,
        evidence_data=filtered_evidence,
        rubric_data=enabler_loader.rubric_data,
        level_fractions=enabler_loader.level_fractions,
        evidence_mapping_data=enabler_loader.evidence_mapping_data,
        vectorstore_retriever=retriever,
        use_mapping_filter=filter_mode, # 🟢 [FIXED] แก้ชื่อ Argument แล้ว
        target_sub_id=sub_criteria_id if sub_criteria_id!="all" else None,
        mock_llm_eval_func=llm_eval_func,
        mock_llm_summarize_func=llm_summarize_func,
        mock_llm_action_plan_func=llm_action_plan_func,
        disable_semantic_filter=disable_semantic_filter
    )

    # -------------------- Override _retrieve_context สำหรับ Mock --------------------
    if original_mode=="mock":
        assessment_engine._retrieve_context = lambda **kwargs: retrieve_context_MOCK(
            statement=kwargs.get('query'),
            sub_criteria_id=kwargs['sub_criteria_id'],
            level=kwargs.get('level'),
            statement_number=kwargs.get('statement_number', 0),
        )

    # -------------------- Run Assessment --------------------
    try:
        assessment_engine.run_assessment()
        summary = assessment_engine.summarize_results() # <--- 1. สรุปคะแนนเบื้องต้น
        summary['raw_llm_results'] = assessment_engine.raw_llm_results
        
        # 🟢 NEW: 1. Generate & Integrate L5/Highest Full Level Summary (<< ถูกเรียกใช้ตรงนี้)
        summary = generate_and_integrate_l5_summary(assessment_engine, summary) # <--- 2. ถูกเรียกใช้ตรงนี้
        
        # 🟢 NEW: 2. Generate & Integrate Action Plans
        action_plans: Dict[str, Any] = {}
        for sub_id, sub_data in summary.get('SubCriteria_Breakdown', {}).items():
            action_plan = generate_action_plan_for_sub(sub_id, sub_data, summary)
            action_plans[sub_id] = action_plan
        summary['Action_Plans'] = action_plans # <--- 3. ตามด้วย Action Plan
    except Exception as e:
        summary.update(assessment_engine.summarize_results())
        summary['Error_Details'] = str(e)

    if export:
        export_paths = _export_results_to_json(summary, enabler, sub_criteria_id)
        # 🛑 อัปเดต Dict summary ด้วยพาธที่ได้จากการ Export
        summary.update(export_paths)

    return summary


# -------------------- CLI Entry Point (Adapter) --------------------
if __name__ == "__main__":  
    try:
        # 1. ⚙️ การตั้งค่า Argument Parser
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

        parser.add_argument("--disable-semantic-filter",
                            action="store_true",
                            help="Disable semantic reranking / semantic filter in RAG (for debugging retrieval).")
        
        # 🟢 [FIX 1] เพิ่ม Argument --allow-fallback เข้าไปใน argparse
        parser.add_argument("--allow-fallback",
                            action="store_true",
                            help="Allow assessment to fallback to a random/mock mode if a fatal error occurs during RAG/LLM.")
        
        args = parser.parse_args()
        
        
        # 2. 🚀 CLI Call: เรียกใช้ฟังก์ชัน run_assessment_process
        final_results = run_assessment_process(
            enabler=args.enabler,
            sub_criteria_id=args.sub,
            mode=args.mode, 
            filter_mode=args.filter,
            export=args.export,
            disable_semantic_filter=args.disable_semantic_filter,
            # 🟢 [FIX 2] ส่ง allow_fallback โดยตรง
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
        if args.sub != "all" and final_results.get('raw_llm_results'):
            print_detailed_results(
                raw_llm_results=add_pass_status_to_raw_results(final_results['raw_llm_results']), 
                target_sub_id=target_sub_id_for_print,
                enabler_abbr=args.enabler
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