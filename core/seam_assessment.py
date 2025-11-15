# core/seam_assessment.py
"""
Core Engine for SEAM PDCA Assessment.
Implements the sequential maturity check (L1 -> L2 -> L3...) logic,
handles configuration (including mocking), RAG/LLM invocation,
result aggregation, and export.
"""
import os
import sys
import json
import logging
import time
from typing import List, Dict, Any, Optional, Union, Tuple, Set
from datetime import datetime
from dataclasses import dataclass, field

# -------------------- PATH SETUP (แก้ไข NameError) --------------------
try:
    # 🎯 FIX: เปลี่ยนชื่อตัวแปรเป็น PROJECT_ROOT เพื่อให้โค้ดข้างล่างเรียกใช้ได้
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)

    # -------------------- IMPORT CONFIG --------------------
    from config.global_vars import (
        EXPORTS_DIR,
        MAX_LEVEL,
        INITIAL_LEVEL,
        FINAL_K_RERANKED,  
        RUBRIC_FILENAME_PATTERN,
        RUBRIC_CONFIG_DIR,
        DEFAULT_ENABLER,
        EVIDENCE_DOC_TYPES, # 🟢 ต้อง Import เพื่อใช้ในการกำหนดชื่อ Collection
        # ALL_SUB_CRITERIA_IDS ถูกลบออกแล้ว
    )
    
    # -------------------- IMPORT CORE LOGIC --------------------
    from core.llm_data_utils import ( 
        create_structured_action_plan, 
        evaluate_with_llm,
        retrieve_context_with_filter,
        set_mock_control_mode as set_llm_data_mock_mode 
    )
    from core.vectorstore import VectorStoreManager # Import for type hinting/usage
    from core.seam_prompts import PDCA_PHASE_MAP # ใช้สำหรับ mapping level -> PDCA phase

    # -------------------- IMPORT MOCK LOGIC (Conditional) --------------------
    import assessments.seam_mocking as seam_mocking 
    
except ImportError as e:
    print(f"FATAL ERROR: Failed to import required modules. Error: {e}", file=sys.stderr)
    raise

logger = logging.getLogger(__name__)


# =================================================================
# Configuration Class
# =================================================================
@dataclass
class AssessmentConfig:
    """Configuration for the SEAM PDCA Assessment Run."""
    enabler: str = DEFAULT_ENABLER
    target_level: int = MAX_LEVEL
    mock_mode: str = "none" # 'none', 'random', 'control'


# =================================================================
# SEAM Assessment Engine (PDCA Focused)
# =================================================================
class SEAMPDCAEngine:
    
    def __init__(self, config: AssessmentConfig):
        self.config = config
        self.enabler_id = config.enabler
        self.target_level = config.target_level
        self.rubric = self._load_rubric()
        
        # Assessment results storage
        self.raw_llm_results: List[Dict[str, Any]] = []
        self.final_subcriteria_results: List[Dict[str, Any]] = []
        self.total_stats: Dict[str, Any] = {}

        # Mock function pointers (will point to real functions by default)
        self.llm_evaluator = evaluate_with_llm
        self.rag_retriever = retrieve_context_with_filter
        self.action_plan_generator = create_structured_action_plan
        
        # Apply mocking if enabled
        if config.mock_mode in ["random", "control"]:
            self._set_mock_handlers(config.mock_mode)
        
        # Set global mock control mode for llm_data_utils if using 'control'
        if config.mock_mode == "control":
            logger.info("Enabling global LLM data utils mock control mode.")
            set_llm_data_mock_mode(True)
        elif config.mock_mode == "random":
             logger.warning("Mock mode 'random' is not fully implemented. Using 'control' logic if available.")
             # Ensure the global mock flag is reset if we aren't using controlled mode
             if hasattr(seam_mocking, 'set_mock_control_mode'):
                 seam_mocking.set_mock_control_mode(False)
                 set_llm_data_mock_mode(False)
            
        logger.info(f"Engine initialized for Enabler: {self.enabler_id}, Mock Mode: {config.mock_mode}")

    # -------------------- Initialization Helpers --------------------
    def _load_rubric(self) -> List[Dict[str, Any]]:
        """
        Loads the SE-AM Rubric JSON file for the specific enabler.
        Transforms Dictionary root (Criteria ID as keys) to a List of Sub-Criteria.
        """
        filename = RUBRIC_FILENAME_PATTERN.format(enabler=self.enabler_id.lower())
        # 🎯 FIX: PROJECT_ROOT ถูก define ไว้ด้านบนแล้ว
        filepath = os.path.join(PROJECT_ROOT, RUBRIC_CONFIG_DIR, filename) 
        
        if not os.path.exists(filepath):
            logger.error(f"Rubric file not found for {self.enabler_id}: {filepath}")
            if self.config.mock_mode != "none":
                logger.warning("Using minimal mock rubric for testing.")
                return [{
                    "sub_id": "1.1",
                    "name": "Mock Sub-Criteria 1.1",
                    "weight": 4, # ต้องมี weight สำหรับการคำนวณคะแนน
                    "levels": [
                        {"level": 1, "statement": "Mock L1 statement"},
                        {"level": 2, "statement": "Mock L2 statement"}
                    ]
                }]
            raise FileNotFoundError(f"Rubric not found at {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # --- START FIX: Transform Dictionary Root to List of Sub-Criteria ---
            if isinstance(data, dict):
                logger.info("Rubric file detected as Dictionary root. Extracting Sub-Criteria list.")
                
                extracted_list = []
                for criteria_id, criteria_data in data.items():
                    sub_criteria_map = criteria_data.get('subcriteria', {})
                    criteria_name = criteria_data.get('name')
                    
                    for sub_id, sub_data in sub_criteria_map.items():
                        # รวมข้อมูลที่จำเป็นของเกณฑ์หลักเข้ากับเกณฑ์ย่อย
                        sub_data['criteria_id'] = criteria_id
                        sub_data['criteria_name'] = criteria_name
                        sub_data['sub_id'] = sub_id # เพิ่ม sub_id เข้าไปเพื่อให้เรียกใช้งานง่าย
                        sub_data['sub_criteria_name'] = sub_data.get('name', criteria_name + ' sub') # ใช้ 'name' ของเกณฑ์ย่อย
                        
                        # ตรวจสอบว่ามี 'weight' หรือไม่ ถ้าไม่มี ให้ใช้ค่า default หรือค่าจากเกณฑ์หลัก
                        if 'weight' not in sub_data:
                            sub_data['weight'] = criteria_data.get('weight', 0)
                        
                        extracted_list.append(sub_data)
                
                data = extracted_list
            # --- END FIX ---


            if not isinstance(data, list):
                raise ValueError(f"Rubric file {filepath} has invalid root structure (expected list after transformation).")

            # Check for missing levels and sort, and transform levels dict to list
            for sub_criteria in data:
                if "levels" in sub_criteria and isinstance(sub_criteria["levels"], dict):
                    levels_list = []
                    for level_str, statement in sub_criteria["levels"].items():
                        levels_list.append({
                            "level": int(level_str),
                            "statement": statement
                        })
                    
                    sub_criteria["levels"] = levels_list
                
                # จัดเรียง levels ตาม level number
                if "levels" in sub_criteria and isinstance(sub_criteria["levels"], list):
                    sub_criteria["levels"].sort(key=lambda x: x.get("level", 0))
            
            return data

    def _set_mock_handlers(self, mode: str):
        """Replaces real LLM/RAG functions with mock versions."""
        if mode == "control" or mode == "random":
            # ตรวจสอบและแทนที่ฟังก์ชันด้วย Mock (หากมี)
            if hasattr(seam_mocking, 'evaluate_with_llm_CONTROLLED_MOCK'):
                self.llm_evaluator = seam_mocking.evaluate_with_llm_CONTROLLED_MOCK
            if hasattr(seam_mocking, 'retrieve_context_with_filter_MOCK'):
                self.rag_retriever = seam_mocking.retrieve_context_with_filter_MOCK
            if hasattr(seam_mocking, 'create_structured_action_plan_MOCK'):
                self.action_plan_generator = seam_mocking.create_structured_action_plan_MOCK
            if hasattr(seam_mocking, 'set_mock_control_mode'):
                seam_mocking.set_mock_control_mode(mode == "control") 

        logger.warning(f"Engine is running in MOCK mode: {mode}")

    def _get_pdca_phase(self, level: int) -> str:
        """Helper to get the PDCA phase string from the map."""
        return PDCA_PHASE_MAP.get(level, f"Level {level} Requirement")
    
    # -------------------- Core Assessment Logic --------------------
    def _run_single_assessment(
        self,
        sub_criteria: Dict[str, Any],
        statement_data: Dict[str, Any],
        vectorstore_manager: Optional['VectorStoreManager']
    ) -> Dict[str, Any]:
        """
        Runs RAG retrieval and LLM evaluation for a single statement (Level).
        Returns a comprehensive result dictionary.
        """
        sub_id = sub_criteria['sub_id']
        level = statement_data['level']
        statement_text = statement_data['statement']
        sub_criteria_name = sub_criteria['sub_criteria_name']
        
        logger.info(f"  > Starting assessment for {sub_id} L{level}...")

        # 1. Determine PDCA Phase for the prompt
        pdca_phase = self._get_pdca_phase(level)

        # 2. RAG Retrieval 
        # 🎯 FIX: Collection name MUST match the one used during INGEST (e.g., evidence_km)
        # เนื่องจาก vectorstore มีคอลเลกชัน evidence_km อยู่
        collection_name = f"{EVIDENCE_DOC_TYPES}_{self.enabler_id}".lower() 
        rag_query = f"{sub_criteria_name} Level {level} - {statement_text}"

        retrieval_start = time.time()
        
        # Check for vectorstore manager dependency in non-mock mode
        if self.config.mock_mode == "none" and not vectorstore_manager:
            logger.error(f"Cannot run RAG for {sub_id} L{level}: VectorstoreManager is None in non-mock mode.")
            retrieval_result = {"top_evidences": [], "aggregated_context": "ERROR: No vectorstore manager."}
        else:
            try:
                # 🎯 ใช้ self.rag_retriever ที่ถูกตั้งค่าไว้ (Mock หรือ Real)
                retrieval_result = self.rag_retriever(
                    query=rag_query,
                    collection_name=collection_name, # 🟢 ใช้ collection_name ที่แก้ไขแล้ว
                    top_k=FINAL_K_RERANKED 
                )
            except Exception as e:
                logger.error(f"RAG retrieval failed for {sub_id} L{level}: {e}")
                retrieval_result = {"top_evidences": [], "aggregated_context": "ERROR: RAG failure."}
        
        retrieval_duration = time.time() - retrieval_start
        
        aggregated_context = retrieval_result.get("aggregated_context", "")
        top_evidences = retrieval_result.get("top_evidences", [])
        
        logger.info(f"    - Retrieval found {len(top_evidences)} evidences in {retrieval_duration:.2f}s.")

        # 3. LLM Evaluation
        llm_start = time.time()
        try:
            # 🎯 ใช้ self.llm_evaluator ที่ถูกตั้งค่าไว้ (Mock หรือ Real)
            llm_result = self.llm_evaluator(
                context=aggregated_context,
                sub_criteria_name=sub_criteria_name,
                level=level,
                statement_text=statement_text,
                sub_id=sub_id,
                # Pass extra kwargs for PDCA context
                pdca_phase=pdca_phase 
            )
        except Exception as e:
            logger.error(f"LLM evaluation failed for {sub_id} L{level}: {e}")
            llm_result = {"score": 0, "reason": f"LLM Fatal Error: {e}", "is_passed": False}
        
        llm_duration = time.time() - llm_start

        # 4. Construct Final Result
        is_passed = llm_result.get('is_passed', False)
        pass_status = "✅ PASS" if is_passed else "❌ FAIL"
        
        result = {
            "sub_criteria_id": sub_id,
            "sub_criteria_name": sub_criteria_name,
            "level": level,
            "statement": statement_text,
            "pdca_phase": pdca_phase,
            "llm_score": llm_result.get('score', 0),
            "reason": llm_result.get('reason', 'N/A'),
            "is_passed": is_passed,
            "rag_query": rag_query,
            "retrieval_duration_s": retrieval_duration,
            "llm_duration_s": llm_duration,
            "retrieved_evidences_count": len(top_evidences),
            "retrieved_full_source_info": top_evidences, 
            "aggregated_context_used": aggregated_context 
        }
        
        logger.info(f"    - Result: {pass_status} ({llm_result.get('score', 0)}/1) in {llm_duration:.2f}s. Reason: {llm_result.get('reason', 'N/A')[:50]}...")
        
        return result

    # -------------------- Main Execution --------------------
    def run_assessment(
        self, 
        target_sub_id: str = "all", 
        export: bool = False, 
        vectorstore_manager: Optional['VectorStoreManager'] = None
    ) -> Dict[str, Any]:
        """
        Main runner for the assessment engine.
        Implements the sequential maturity check (L1 -> L2 -> L3...) logic.
        """
        start_ts = time.time()
        
        # 1. Filter Rubric based on target_sub_id
        if target_sub_id.lower() == "all":
            sub_criteria_list = self.rubric
        else:
            sub_criteria_list = [
                s for s in self.rubric if s.get('sub_id') == target_sub_id
            ]
            if not sub_criteria_list:
                logger.error(f"Sub-Criteria ID '{target_sub_id}' not found in rubric.")
                return {"error": f"Sub-Criteria ID '{target_sub_id}' not found."}

        # Reset storage
        self.raw_llm_results = []
        self.final_subcriteria_results = []
        
        # 2. Iterate through filtered sub-criteria
        for sub_criteria in sub_criteria_list:
            sub_id = sub_criteria['sub_id']
            sub_criteria_name = sub_criteria['sub_criteria_name']
            sub_weight = sub_criteria.get('weight', 0)
            
            logger.info(f"\n[START] Assessing Sub-Criteria: {sub_id} - {sub_criteria_name} (Weight: {sub_weight})")
            
            highest_full_level = INITIAL_LEVEL - 1 # Start at 0
            is_passed_current_level = True
            
            # 3. Sequential Level Check (L1, L2, L3, ...)
            for statement_data in sub_criteria.get('levels', []):
                level = statement_data.get('level')
                
                if level is None or level > self.target_level:
                    continue 
                
                # Check for strict sequential progression: Stop if the previous level failed
                if not is_passed_current_level:
                    logger.warning(f"  > Skipping L{level}: L{level-1} already failed. Sequential check terminated.")
                    break 

                # Run Assessment for this level
                result = self._run_single_assessment(
                    sub_criteria=sub_criteria,
                    statement_data=statement_data,
                    vectorstore_manager=vectorstore_manager
                )
                
                self.raw_llm_results.append(result)
                is_passed_current_level = result.get('is_passed', False)
                
                if is_passed_current_level:
                    highest_full_level = level # Update highest passed level
                # หากไม่ผ่าน การวนซ้ำจะถูกหยุดโดย check ด้านบนในรอบถัดไป
            
            # 4. Generate Action Plan & Final Summary for this Sub-Criteria
            
            # Target the next level after the highest one passed 
            target_plan_level = highest_full_level + 1
            action_plan = []
            
            # Generate action plan only if we didn't achieve the target level
            if highest_full_level < self.target_level: 
                logger.info(f"  > Generating Action Plan: Target L{target_plan_level}...")
                
                # Collect the statement results for the level we need to plan for (ซึ่งคือระดับแรกที่สอบไม่ผ่าน)
                failed_statements_for_plan = [
                    r for r in self.raw_llm_results
                    if r.get("sub_criteria_id") == sub_id and r.get("level") == target_plan_level
                ]
                
                if failed_statements_for_plan:
                    try:
                         # 🎯 ใช้ self.action_plan_generator ที่ถูกตั้งค่าไว้ (Mock หรือ Real)
                         action_plan = self.action_plan_generator(
                            failed_statements_data=failed_statements_for_plan,
                            sub_id=sub_id,
                            enabler=self.enabler_id,
                            target_level=target_plan_level 
                         )
                    except Exception as e:
                        logger.error(f"Action Plan Generation failed for {sub_id}: {e}")
                        action_plan = [{"Phase": "ERROR", "Goal": "Action Plan generation failed."}]
                
            # 5. Aggregate Final Results for the Sub-Criteria
            final_sub_result = {
                "sub_criteria_id": sub_id,
                "sub_criteria_name": sub_criteria_name,
                "highest_full_level": highest_full_level,
                "weight": sub_weight,
                "target_level_achieved": highest_full_level >= self.target_level,
                "weighted_score": self._calculate_weighted_score(highest_full_level, sub_weight),
                "action_plan": action_plan,
                "raw_results_ref": [r for r in self.raw_llm_results if r.get('sub_criteria_id') == sub_id] 
            }
            self.final_subcriteria_results.append(final_sub_result)
            
            logger.info(f"[END] Assessment for {sub_id} finished. Highest Full Level: L{highest_full_level}")

        # 6. Calculate Overall Statistics & Finalize
        self._calculate_overall_stats(target_sub_id)
        
        final_results = {
            "summary": self.total_stats,
            "sub_criteria_results": self.final_subcriteria_results,
            "raw_llm_results": self.raw_llm_results,
            "run_time_seconds": time.time() - start_ts,
            "timestamp": datetime.now().isoformat(),
        }
        
        # 7. Handle Export
        if export:
             export_path = self._export_results(final_results)
             final_results["export_path_used"] = export_path

        return final_results
    
    # -------------------- Score & Stats Helpers --------------------
    def _calculate_weighted_score(self, highest_full_level: int, sub_weight: int) -> float:
        """Calculates a weighted score based on the highest level achieved."""
        if MAX_LEVEL == 0:
            return 0.0
        # Score = (Level Achieved / MAX_LEVEL) * Sub_Weight
        return (highest_full_level / MAX_LEVEL) * sub_weight

    def _calculate_overall_stats(self, target_sub_id: str):
        """Aggregates scores and stats across all sub-criteria run."""
        total_subcriteria = len(self.final_subcriteria_results)
        
        # รวมน้ำหนักสูงสุดที่เป็นไปได้ของเกณฑ์ที่รัน (ใช้ในการคำนวณ % โดยรวม)
        max_possible_weighted_score_run = sum(r['weight'] * MAX_LEVEL for r in self.rubric if r.get('sub_id') in [r['sub_criteria_id'] for r in self.final_subcriteria_results])
        
        total_achieved_weighted_score = sum(r['weighted_score'] for r in self.final_subcriteria_results)
        
        avg_weighted_score_percent = (total_achieved_weighted_score / max_possible_weighted_score_run) * 100 if max_possible_weighted_score_run > 0 else 0.0
        
        self.total_stats = {
            "enabler": self.enabler_id,
            "target_level": self.target_level,
            "target_sub_id_run": target_sub_id,
            "total_subcriteria": total_subcriteria,
            "total_achieved_weighted_score": total_achieved_weighted_score,
            "max_possible_weighted_score_run": max_possible_weighted_score_run,
            "avg_weighted_score_percent": avg_weighted_score_percent,
            "overall_status": "COMPLETED",
        }

    # -------------------- Output & Export --------------------
    def print_detailed_results(self, target_sub_id: str):
        """Prints the detailed results and action plan for a specific sub-criteria ID."""
        
        sub_result = next(
            (r for r in self.final_subcriteria_results if r['sub_criteria_id'] == target_sub_id),
            None
        )
        
        if not sub_result:
            print(f"\n[ERROR] Detailed results not found for {target_sub_id}.")
            return
            
        print("\n" + "="*60)
        print(f"DETAILED ASSESSMENT RESULTS: {sub_result['sub_criteria_name']} ({target_sub_id})")
        print("="*60)
        print(f"-> HIGHEST FULL LEVEL: L{sub_result['highest_full_level']} / L{self.target_level}")
        print(f"-> WEIGHTED SCORE: {sub_result['weighted_score']:.3f}")
        print("-" * 60)

        raw_results = sub_result['raw_results_ref']
        
        results_by_level = {}
        for r in raw_results:
            level = r.get('level', 0)
            if level not in results_by_level:
                results_by_level[level] = []
            results_by_level[level].append(r)

        for level in sorted(results_by_level.keys()):
            level_results = results_by_level[level]
            r = level_results[0] 
            
            is_passed = r.get('is_passed', False)
            pass_status = "✅ PASS" if is_passed else "❌ FAIL"
            
            print(f"  > Level {level} ({r.get('pdca_phase', 'N/A')}): {pass_status}")
            
            print(f"    - Statement: {r.get('statement', 'N/A')[:100]}...")
            print(f"      [Reason]: {r.get('reason', 'N/A')[:120]}...")

            sources = r.get('retrieved_full_source_info', []) 
            if sources:
                print("      [SOURCE FILES] (Top {} Chunks):".format(FINAL_K_RERANKED))
                for src in sources:
                    metadata = src.get('metadata', {})
                    source_name = metadata.get('file_name', 'Unknown File')
                    chunk_uuid = metadata.get('stable_doc_uuid', 'N/A') 
                    uuid_short = chunk_uuid[:8] + "..." if chunk_uuid else "N/A"
                    print(f"        -> {source_name} (UUID: {uuid_short})")
            
            # Stop detailed print if the level failed and was the next expected level
            if not is_passed and level == sub_result['highest_full_level'] + 1:
                print(f"\n[ASSESSMENT STOPPED] Failure detected at L{level}.")
                break
        
        # Print Action Plan if available
        action_plan = sub_result.get('action_plan', [])
        if action_plan:
             print("\n" + "#"*20 + " ACTION PLAN " + "#"*20)
             for phase in action_plan:
                 print(f"PHASE: {phase.get('Phase', 'N/A')} | GOAL: {phase.get('Goal', 'N/A')}")
                 for action in phase.get('Actions', []):
                     print(f"  > RECOMMENDATION: {action.get('Recommendation', 'N/A')}")
                     print(f"    - Responsible: {action.get('Responsible', 'N/A')} | Metric: {action.get('Key_Metric', 'N/A')}")
             print("#"*53 + "\n")


    def _export_results(self, final_results: Dict[str, Any]) -> str:
        """Exports the final results dictionary to a JSON file."""
        try:
            os.makedirs(EXPORTS_DIR, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            enabler = self.enabler_id
            target_sub = final_results['summary']['target_sub_id_run'].replace('.', '_')
            
            filename = f"seam_assessment_{enabler}_{target_sub}_{timestamp}.json"
            filepath = os.path.join(EXPORTS_DIR, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Report exported successfully to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to export results: {e}")
            return f"EXPORT_FAILED: {e}"