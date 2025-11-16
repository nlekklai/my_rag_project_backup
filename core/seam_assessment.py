# core/seam_assessment.py
import sys
import json
import logging
import time
import os
from typing import List, Dict, Any, Optional, Union, Tuple, Set, Final
from datetime import datetime
from dataclasses import dataclass, field
import multiprocessing # NEW: Import for parallel execution
from core.rag_enhancer import enhance_query_for_statement
import pathlib, uuid

# -------------------- PATH SETUP & IMPORTS --------------------
try:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)

    from config.global_vars import (
        EXPORTS_DIR, MAX_LEVEL, INITIAL_LEVEL, FINAL_K_RERANKED,
        RUBRIC_FILENAME_PATTERN, RUBRIC_CONFIG_DIR, DEFAULT_ENABLER,
        EVIDENCE_DOC_TYPES, INITIAL_TOP_K
    )
    
    from core.llm_data_utils import ( 
        create_structured_action_plan, evaluate_with_llm,
        retrieve_context_with_filter, retrieve_context_for_low_levels,
        evaluate_with_llm_low_level, LOW_LEVEL_K, 
        set_mock_control_mode as set_llm_data_mock_mode 
    )
    from core.vectorstore import VectorStoreManager, load_all_vectorstores 
    from core.seam_prompts import PDCA_PHASE_MAP 
        
    import assessments.seam_mocking as seam_mocking 
    
except ImportError as e:
    print(f"FATAL ERROR: Failed to import required modules. Error: {e}", file=sys.stderr)
    
    # Define placeholder variables if imports fail
    EXPORTS_DIR = "exports"
    MAX_LEVEL = 5
    INITIAL_LEVEL = 1
    FINAL_K_RERANKED = 3
    RUBRIC_FILENAME_PATTERN = "{enabler}_rubric.json"
    RUBRIC_CONFIG_DIR = "config/rubrics"
    DEFAULT_ENABLER = "KM"
    EVIDENCE_DOC_TYPES = "evidence"
    INITIAL_TOP_K = 10
    
    def create_structured_action_plan(*args, **kwargs): return [{"Phase": "Mock Plan", "Goal": "Resolve issue"}]
    def evaluate_with_llm(*args, **kwargs): return {"score": 1, "reason": "Mock pass", "is_passed": True}
    def retrieve_context_with_filter(*args, **kwargs): return {"top_evidences": [], "aggregated_context": "Mock Context"}
    def retrieve_context_for_low_levels(*args, **kwargs): return {"top_evidences": [], "aggregated_context": "Mock Low Context"}
    def evaluate_with_llm_low_level(*args, **kwargs): return {"score": 1, "reason": "Mock pass L1/L2", "is_passed": True}
    LOW_LEVEL_K = 2
    def set_llm_data_mock_mode(mode): pass
    class VectorStoreManager: pass
    def load_all_vectorstores(*args, **kwargs): return VectorStoreManager()
    PDCA_PHASE_MAP = {1: "Plan", 2: "Do", 3: "Check", 4: "Act", 5: "Innovate"}
    class seam_mocking:
        @staticmethod
        def evaluate_with_llm_CONTROLLED_MOCK(*args, **kwargs): return {"score": 0, "reason": "Mock fail", "is_passed": False}
        @staticmethod
        def retrieve_context_with_filter_MOCK(*args, **kwargs): return {"top_evidences": [], "aggregated_context": "Mock Context"}
        @staticmethod
        def create_structured_action_plan_MOCK(*args, **kwargs): return [{"Phase": "Mock Plan", "Goal": "Resolve issue"}]
        @staticmethod
        def set_mock_control_mode(mode): pass
    
    if "FATAL ERROR" in str(e):
        pass 


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# =================================================================
# Configuration Class
# =================================================================
@dataclass
class AssessmentConfig:
    """Configuration for the SEAM PDCA Assessment Run."""
    enabler: str = DEFAULT_ENABLER
    target_level: int = MAX_LEVEL
    mock_mode: str = "none" # 'none', 'random', 'control'
    force_sequential: bool = field(default=False) # Flag to force sequential run


# =================================================================
# SEAM Assessment Engine (PDCA Focused)
# =================================================================
class SEAMPDCAEngine:
    
    # 🎯 Mapping for RAG Query Augmentation at Level 1 (Plan)
    ENABLER_L1_AUGMENTATION = {
        "KM": "นโยบาย วิสัยทัศน์ ทิศทางกลยุทธ์ แผนกลยุทธ์ ความมุ่งมั่น",
        "HCM": "นโยบายการบริหารบุคคล แผนกำลังคน ยุทธศาสตร์ทรัพยากรบุคคล การมีส่วนร่วมของผู้บริหาร",
        "DT": "นโยบายเทคโนโลยีดิจิทัล แผนแม่บทดิจิทัล ทิศทาง IT แผนกลยุทธ์เทคโนโลยี",
        "SP": "นโยบายองค์กร แผนยุทธศาสตร์องค์กร ทิศทางกลยุทธ์ แผนกลยุทธ์",
        "DEFAULT": "นโยบาย วิสัยทัศน์ ทิศทางกลยุทธ์ แผนกลยุทธ์ ความมุ่งมั่น" 
    }
    
    L1_INITIAL_TOP_K_RAG: int = 50 
    
    def __init__(self, config: AssessmentConfig):
            self.config = config
            self.enabler_id = config.enabler
            self.target_level = config.target_level
            self.rubric = self._load_rubric()

            # 🟢 FIX: Disable Strict Filter (Permanent Bypass)
            self.initial_evidence_ids: Set[str] = self._load_initial_evidence_info()
            all_statements = self._flatten_rubric_to_statements()
            initial_count = len(all_statements)

            logger.info(f"DEBUG: Statements found: {initial_count}. Strict Filter is **DISABLED**.")

            # all_statements = self._apply_strict_filter(all_statements, self.initial_evidence_ids) 
            self.statements_to_assess = all_statements
            logger.info(f"DEBUG: Statements selected for assessment: {len(self.statements_to_assess)} (Skipped: {initial_count - len(self.statements_to_assess)})")

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
                if hasattr(seam_mocking, 'set_mock_control_mode'):
                    seam_mocking.set_mock_control_mode(False)
                    set_llm_data_mock_mode(False)

            logger.info(f"Engine initialized for Enabler: {self.enabler_id}, Mock Mode: {config.mock_mode}")

    # -------------------- Initialization Helpers --------------------
    def _load_rubric(self) -> List[Dict[str, Any]]:
        """Loads the SE-AM Rubric JSON file."""
        filename = RUBRIC_FILENAME_PATTERN.format(enabler=self.enabler_id.lower())
        filepath = os.path.join(PROJECT_ROOT, RUBRIC_CONFIG_DIR, filename) 
        
        if not os.path.exists(filepath):
            logger.error(f"Rubric file not found for {self.enabler_id}: {filepath}")
            if self.config.mock_mode != "none":
                logger.warning("Using minimal mock rubric for testing.")
                return [{
                    "sub_id": "1.1", "name": "Mock Sub-Criteria 1.1", "weight": 4, 
                    "levels": [{"level": 1, "statement": "Mock L1 statement"}, {"level": 2, "statement": "Mock L2 statement"}]
                }]
            raise FileNotFoundError(f"Rubric not found at {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # --- FIX: Transform Dictionary Root to List of Sub-Criteria ---
            if isinstance(data, dict):
                logger.info("Rubric file detected as Dictionary root. Extracting Sub-Criteria list.")
                extracted_list = []
                for criteria_id, criteria_data in data.items():
                    sub_criteria_map = criteria_data.get('subcriteria', {})
                    criteria_name = criteria_data.get('name')
                    
                    for sub_id, sub_data in sub_criteria_map.items():
                        sub_data['criteria_id'] = criteria_id
                        sub_data['criteria_name'] = criteria_name
                        sub_data['sub_id'] = sub_id 
                        sub_data['sub_criteria_name'] = sub_data.get('name', criteria_name + ' sub') 
                        
                        if 'weight' not in sub_data:
                            sub_data['weight'] = criteria_data.get('weight', 0)
                        
                        extracted_list.append(sub_data)
                data = extracted_list

            if not isinstance(data, list):
                raise ValueError(f"Rubric file {filepath} has invalid root structure (expected list after transformation).")

            # Check for missing levels and sort, and transform levels dict to list
            for sub_criteria in data:
                if "levels" in sub_criteria and isinstance(sub_criteria["levels"], dict):
                    levels_list = []
                    for level_str, statement in sub_criteria["levels"].items():
                        levels_list.append({"level": int(level_str), "statement": statement})
                    sub_criteria["levels"] = levels_list
                
                if "levels" in sub_criteria and isinstance(sub_criteria["levels"], list):
                    sub_criteria["levels"].sort(key=lambda x: x.get("level", 0))
            
            return data

    def _set_mock_handlers(self, mode: str):
        """Replaces real LLM/RAG functions with mock versions."""
        if mode == "control" or mode == "random":
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
    
    # 🟢 MODIFIED: Level Constraint Prompt Generator
    def _get_level_constraint_prompt(self, level: int) -> str:
        """
        สร้าง Prompt Constraint เพื่อจำกัดขอบเขตของหลักฐานให้เหมาะสมกับระดับวุฒิภาวะที่กำลังประเมิน
        """
        if level == 1:
            return "ข้อจำกัด: หลักฐานต้องแสดงถึง 'การกำหนดนโยบาย/วิสัยทัศน์', 'การวางแผนกลยุทธ์', 'การจัดทำกรอบแนวทาง', หรือ 'การเริ่มต้นดำเนินการ' เท่านั้น (L1-Focus)"
        elif level == 2:
            return "ข้อจำกัด: หลักฐานต้องเน้นเฉพาะ 'การดำเนินงาน', 'การขับเคลื่อน', 'การทำให้เป็นรูปธรรม', หรือ 'การมีส่วนร่วม' ตามแผนเท่านั้น (L2-Focus)"
        elif level == 3:
            return "ข้อจำกัด: หลักฐานต้องเน้นเฉพาะ 'การตรวจสอบ', 'การติดตาม', 'การประเมินผล', หรือ 'การสื่อสารผลลัพธ์ภายใน' เท่านั้น (L3-Focus)"
        elif level == 4:
            return "ข้อจำกัด: หลักฐานควรแสดงถึง 'การบูรณาการ', 'การปรับปรุงอย่างต่อเนื่อง', หรือ 'การประยุกต์ใช้กับยุทธศาสตร์' เท่านั้น (L4-Focus)"
        elif level == 5:
            return "ข้อจำกัด: หลักฐานควรแสดงถึง 'นวัตกรรม', 'การสร้างคุณค่าทางธุรกิจ', หรือ 'ผลลัพธ์ระยะยาว' โดยชัดเจน (L5-Focus)"
        else:
            return "" 
        
    # -------------------- Statement Preparation & Filtering Helpers --------------------
    def _flatten_rubric_to_statements(self) -> List[Dict[str, Any]]:
        """Extracts a flat list of all individual level statements from the rubric."""
        flat_list = []
        for sub_criteria in self.rubric:
            sub_id = sub_criteria['sub_id']
            sub_criteria_name = sub_criteria['sub_criteria_name']
            
            for statement_data in sub_criteria.get('levels', []):
                flat_list.append({
                    "sub_id": sub_id,
                    "sub_criteria_name": sub_criteria_name,
                    "level": statement_data['level'],
                    "statement": statement_data['statement'],
                    "evidence_doc_ids": statement_data.get('evidence_doc_ids', []) 
                })
        return flat_list

    def _load_initial_evidence_info(self) -> Set[str]:
        """Retrieves the set of all available stable document IDs in the VectorStore."""
        if self.config.mock_mode != "none":
            return {"00000000-0000-0000-0000-000000000001"}
        return set() 

    def _apply_strict_filter(self, statements: List[Dict[str, Any]], available_evidence_ids: Set[str]) -> List[Dict[str, Any]]:
        """
        Filters out statements that have no specified evidence_doc_ids 
        that match any ID found in the available_evidence_ids set.
        """
        if not available_evidence_ids:
            logger.warning("Strict Filter bypassed: No available evidence IDs loaded.")
            return statements

        filtered_statements = []
        for stmt in statements:
            required_ids = set(stmt.get('evidence_doc_ids', []))
            
            if not required_ids or required_ids.isdisjoint(available_evidence_ids):
                 logger.debug(f"Strict Filter: Skipping {stmt['sub_id']} L{stmt['level']} (No evidence match)")
                 continue
            
            filtered_statements.append(stmt)
            
        return filtered_statements
        
    # -------------------- Core Assessment Logic --------------------
    def _run_single_assessment(
        self,
        sub_criteria: Dict[str, Any],
        statement_data: Dict[str, Any],
        vectorstore_manager: Optional['VectorStoreManager'] 
    ) -> Dict[str, Any]:
        """Runs RAG retrieval and LLM evaluation for a single statement (Level)."""
        sub_id = sub_criteria['sub_id']
        level = statement_data['level']
        statement_text = statement_data['statement']
        sub_criteria_name = sub_criteria['sub_criteria_name']
        
        logger.info(f"  > Starting assessment for {sub_id} L{level}...")

        # 1. Determine PDCA Phase and LEVEL CONSTRAINT
        pdca_phase = self._get_pdca_phase(level)
        level_constraint = self._get_level_constraint_prompt(level)

        # 2. RAG Retrieval SETUP
        # rag_query = f"{sub_criteria_name} Level {level} - {statement_text}. {level_constraint}"

        # ฟังก์ชันนี้จะจัดการ Query Boosting สำหรับ 1.1 KM และส่ง Query มาตรฐานสำหรับส่วนอื่น ๆ
        rag_query = enhance_query_for_statement(
            statement_id=sub_id,
            enabler_id=self.enabler_id,
            statement_text=statement_text,
            focus_hint=level_constraint # level_constraint ถูกใช้เป็น focus_hint ในฟังก์ชัน enhancer
        )
        # === END OF MODIFICATION ===

        current_final_k = FINAL_K_RERANKED
        current_rag_retriever = self.rag_retriever 
        current_llm_evaluator = self.llm_evaluator 
        initial_k_to_use = INITIAL_TOP_K

        # 🟢 PHASE 2 OPTIMIZATION: Use specialized retrieval/evaluation for L1/L2
        if level <= 2:
            current_llm_evaluator = evaluate_with_llm_low_level
            current_final_k = LOW_LEVEL_K 
            initial_k_to_use = self.L1_INITIAL_TOP_K_RAG
            
            # if level == 1:
            #     enabler_keywords = self.ENABLER_L1_AUGMENTATION.get(
            #         self.enabler_id, 
            #         self.ENABLER_L1_AUGMENTATION["DEFAULT"]
            #     )
            #     rag_query += " " + enabler_keywords
            pass
        else:
             current_final_k = FINAL_K_RERANKED

        retrieval_start = time.time()
        
        if self.config.mock_mode == "none" and not vectorstore_manager:
            logger.error(f"Cannot run RAG for {sub_id} L{level}: VectorstoreManager is None in non-mock mode.")
            retrieval_result = {"top_evidences": [], "aggregated_context": "ERROR: No vectorstore manager."}
        else:
            try:
                retrieval_result = current_rag_retriever(
                    query=rag_query,
                    doc_type=EVIDENCE_DOC_TYPES, 
                    enabler=self.enabler_id,     
                    top_k=current_final_k,
                    initial_k=initial_k_to_use,
                    sub_id=sub_id, 
                    level=level,
                    vectorstore_manager=vectorstore_manager 
                )
            except Exception as e:
                logger.error(f"RAG retrieval failed for {sub_id} L{level}: {e}")
                retrieval_result = {"top_evidences": [], "aggregated_context": "ERROR: RAG failure."}

        retrieval_duration = time.time() - retrieval_start
        aggregated_context = retrieval_result.get("aggregated_context", "")
        top_evidences = retrieval_result.get("top_evidences", [])

        logger.info(f"    - Retrieval found {len(top_evidences)} evidences in {retrieval_duration:.2f}s (K={current_final_k}).")

        # 3. LLM Evaluation
        llm_start = time.time()
        try:
            llm_result = current_llm_evaluator(
                context=aggregated_context,
                sub_criteria_name=sub_criteria_name,
                level=level,
                statement_text=statement_text,
                sub_id=sub_id,
                pdca_phase=pdca_phase,
                level_constraint=level_constraint
            )
        except Exception as e:
            logger.error(f"LLM evaluation failed for {sub_id} L{level}: {e}")
            llm_result = {"score": 0, "reason": f"LLM Fatal Error: {e}", "is_passed": False}

        llm_duration = time.time() - llm_start

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
    
    # -------------------- Calculation Helpers (ADDED) --------------------
    def _calculate_weighted_score(self, highest_full_level: int, weight: int) -> float:
        """
        Calculates the weighted score based on the highest full level achieved.
        Score is calculated by: (Level / 5) * Weight
        """
        MAX_LEVEL_CALC = 5  
        
        if highest_full_level <= 0:
            return 0.0
        
        level_for_calc = min(highest_full_level, MAX_LEVEL_CALC)
        score = (level_for_calc / MAX_LEVEL_CALC) * weight
        return score

    def _calculate_overall_stats(self, target_sub_id: str = "all"):
        """
        Calculates the total weighted score, total possible weight, and overall maturity score/level.
        """
        total_weighted_score = 0.0
        total_possible_weight = 0.0
        assessed_count = 0
        
        for result in self.final_subcriteria_results:
            if target_sub_id.lower() != "all" and result.get('sub_criteria_id') != target_sub_id:
                continue

            weighted_score = result.get('weighted_score', 0.0)
            weight = result.get('weight', 0)
            
            total_weighted_score += weighted_score
            total_possible_weight += weight
            assessed_count += 1
            
        overall_maturity_score_avg = 0.0
        overall_maturity_level = "N/A"
        overall_progress_percent = 0.0 # ตั้งค่าเริ่มต้นที่ 0.0
        
        if total_possible_weight > 0:
            overall_progress_percent = total_weighted_score / total_possible_weight
            
            MAX_LEVEL_STATS = 5 
            overall_maturity_score_avg = overall_progress_percent * MAX_LEVEL_STATS 

            # Determine Overall Maturity Level (Mapping Placeholder)
            if overall_maturity_score_avg >= 4.5:
                overall_maturity_level = "L5"
            elif overall_maturity_score_avg >= 3.5:
                overall_maturity_level = "L4"
            elif overall_maturity_score_avg >= 2.5:
                overall_maturity_level = "L3"
            elif overall_maturity_score_avg >= 1.5:
                overall_maturity_level = "L2"
            elif overall_maturity_score_avg >= 0.5:
                overall_maturity_level = "L1"
            else:
                overall_maturity_level = "L0"
        
        # 📌 ปรับแก้ Key name และคำนวณ % Achieved เป็น 0-100% สำหรับ CLI
        percentage_achieved_for_cli = overall_progress_percent * 100 

        # Store Results
        self.total_stats = {
            "Overall Maturity Score (Avg.)": overall_maturity_score_avg,
            "Overall Maturity Level (Weighted)": overall_maturity_level,
            "Number of Sub-Criteria Assessed": assessed_count,
            "Total Weighted Score Achieved": total_weighted_score,
            "Total Possible Weight": total_possible_weight,
            "Overall Progress Percentage (0.0 - 1.0)": overall_progress_percent,
            # 🟢 KEY ใหม่ที่ CLI ใช้ (ค่า 0-100)
            "percentage_achieved_run": percentage_achieved_for_cli, 
            # 🟢 KEY ใหม่ที่ CLI ใช้ (อ้างอิงจาก start_assessment.py)
            "total_subcriteria": assessed_count, 
            # 🟢 KEY สำหรับ Target Level
            "target_level": self.config.target_level,
        }
    
    # -------------------- Export Results --------------------
    def _export_results(self, data: Dict[str, Any], target_id: str) -> str:
        """Exports the final results to a JSON file."""
        
        # 🟢 1. สร้างชื่อไฟล์ตาม Enabler และ Target ID
        # ใช้ ID ที่ถูกส่งมาแทนค่า Default
        file_name = f"assessment_results_{self.config.enabler}_{target_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:4]}.json"
        
        # กำหนดที่จัดเก็บ (สมมติว่าเก็บในโฟลเดอร์ 'exports' ที่ Root)
        export_dir = pathlib.Path(pathlib.Path(__file__).parent.parent, "exports")
        export_dir.mkdir(exist_ok=True) # สร้างโฟลเดอร์ถ้ายังไม่มี
        
        export_path = export_dir / file_name
        
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                # ใช้ indent 4 เพื่อให้อ่านง่าย
                json.dump(data, f, ensure_ascii=False, indent=4)
            
            logger.info(f"Successfully exported results for {target_id} to: {export_path}")
            return str(export_path)
            
        except Exception as e:
            logger.error(f"Error during file export to {export_path}: {e}")
            return f"EXPORT_FAILED: {e}"
    
    def print_detailed_results(self, target_sub_id: str = "all"):
            """พิมพ์สรุปผลการประเมินโดยละเอียดแบบระดับต่อระดับ"""
            import sys
            
            results_to_print = []
            if hasattr(self, 'final_subcriteria_results') and isinstance(self.final_subcriteria_results, list):
                if target_sub_id.lower() == "all":
                    results_to_print = self.final_subcriteria_results
                else:
                    results_to_print = [
                        res for res in self.final_subcriteria_results 
                        if res.get('sub_criteria_id') == target_sub_id
                    ]

            if not results_to_print:
                print(f"\n[DETAIL] ไม่พบผลลัพธ์สำหรับ Sub-Criteria ID: {target_sub_id} (หรือยังไม่ได้รันการประเมิน)", file=sys.stderr)
                return

            print("\n\n============================================================")
            print("                 DETAILED ASSESSMENT RESULTS                  ")
            print("============================================================")

            for sub_result in results_to_print:
                sub_id = sub_result.get('sub_criteria_id', 'N/A')
                sub_name = sub_result.get('sub_criteria_name', 'N/A')
                achieved_level = sub_result.get('highest_full_level', 0)
                target_achieved = sub_result.get('target_level_achieved', False)
                weighted_score = sub_result.get('weighted_score', 0.0)
                raw_results = sub_result.get('raw_results_ref', []) 

                print(f"\n--- Sub-Criteria ID: {sub_id} - {sub_name} (Weight: {sub_result.get('weight', 0)}) ---")
                print(f"  > 🏆 Highest Full Level Achieved: L{achieved_level}")
                print(f"  > ✅ Target Level Achieved (Target L{self.config.target_level}): {'YES' if target_achieved else 'NO'}")
                print(f"  > 💰 Weighted Score: {weighted_score:.2f}")

                print("\n  >> Level Check Status (L1 -> L5):")
                for raw_res in raw_results:
                    level = raw_res.get('level', 'N/A')
                    status = "✅ PASS" if raw_res.get('is_passed', False) else "❌ FAIL"
                    reason = raw_res.get('reason', 'N/A')
                    duration = raw_res.get('llm_duration_s', 0.0)

                    print(f"    - L{level}: {status} (Duration: {duration:.2f}s)")
                    short_reason = reason[:100] + "..." if len(reason) > 100 else reason
                    print(f"      - Reason: {short_reason}")
                
                # พิมพ์ Action Plan
                action_plan = sub_result.get('action_plan')
                if action_plan and isinstance(action_plan, list):
                    print("\n  >> 🚨 ACTION PLAN (To Achieve Next Level):")
                    
                    is_complex_list = action_plan and isinstance(action_plan[0], dict) and 'Phase' in action_plan[0]
                    
                    if is_complex_list:
                        for plan in action_plan:
                            print(f"    - 🎯 Goal ({plan.get('Phase', 'N/A') or 'N/A'}): {plan.get('Goal', 'N/A')}")
                            for action in plan.get('Actions', []):
                                print(f"      • [L{action.get('Failed_Level', 'N/A')}] {action.get('Recommendation', 'N/A')}")
                    else:
                        is_simple_list = all(isinstance(item, dict) and 'Recommendation' in item for item in action_plan)
                        
                        if is_simple_list:
                            for action in action_plan:
                                failed_level = action.get('Failed_Level', 'N/A')
                                rec = action.get('Recommendation', 'N/A')
                                print(f"    • [L{failed_level}] {rec}")
                        else:
                            print(f"      [WARNING: Unknown AP Structure] {action_plan}")
                
                elif action_plan:
                    print(f"  >> 🚨 ACTION PLAN:")
                    print(f"      {action_plan}")

                print("-" * 60)
    

    # -------------------- Multiprocessing Worker Method --------------------
    @staticmethod
    def _assess_single_sub_criteria_worker(
        sub_criteria: Dict[str, Any],
        engine_config_dict: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Worker function executed in parallel processes."""
        
        # 🟢 FIX: Disable logging in worker to prevent multiprocessing deadlocks
        worker_logger = logging.getLogger()
        if worker_logger.handlers:
            worker_logger.handlers.clear()
        worker_logger.addHandler(logging.NullHandler())
        
        sub_id = sub_criteria['sub_id']
        print(f"[WORKER {os.getpid()}] Starting assessment for {sub_id}...", file=sys.stderr)
        
        # 1. Re-instantiate Engine and VSM LOCALLY
        try:
            config = AssessmentConfig(**engine_config_dict)
            worker_engine = SEAMPDCAEngine(config=config)
            
            worker_vsm = load_all_vectorstores(
                doc_types=[EVIDENCE_DOC_TYPES], 
                evidence_enabler=config.enabler
            )
            
        except Exception as e:
            print(f"[WORKER {os.getpid()}] Init Failed for {sub_id}: {e}", file=sys.stderr)
            return (
                [], 
                {
                    "sub_criteria_id": sub_id, "sub_criteria_name": sub_criteria.get('sub_criteria_name', 'N/A'),
                    "highest_full_level": 0, "weight": sub_criteria.get('weight', 0),
                    "target_level_achieved": False, "weighted_score": 0.0,
                    "action_plan": [], "raw_results_ref": [],
                    "error": f"Worker Initialization Failed: {e}"
                }
            )

        # 2. Run sequential Level Check (L1 -> L2 -> L3...)
        highest_full_level = INITIAL_LEVEL - 1
        is_passed_current_level = True
        raw_results_for_sub = []
        
        for statement_data in sub_criteria.get('levels', []):
            level = statement_data.get('level')
            
            if level is None or level > config.target_level:
                continue 
            
            if not is_passed_current_level:
                print(f"[WORKER {os.getpid()}] > Skipping L{level}: L{level-1} already failed. Sequential check terminated.", file=sys.stderr)
                break 

            result = worker_engine._run_single_assessment(
                sub_criteria=sub_criteria,
                statement_data=statement_data,
                vectorstore_manager=worker_vsm 
            )
            
            raw_results_for_sub.append(result)
            is_passed_current_level = result.get('is_passed', False)
            
            if is_passed_current_level:
                highest_full_level = level
        
        # 3. Generate Action Plan & Aggregate Final Results
        target_plan_level = highest_full_level + 1
        action_plan = []
        
        if target_plan_level <= MAX_LEVEL and highest_full_level < config.target_level: 
            failed_statements_for_plan = [
                r for r in raw_results_for_sub
                if r.get("level") == target_plan_level
            ]
            
            if failed_statements_for_plan:
                try:
                     action_plan = worker_engine.action_plan_generator(
                        failed_statements_data=failed_statements_for_plan,
                        sub_id=sub_id, enabler=config.enabler, target_level=target_plan_level 
                     )
                except Exception as e:
                    print(f"[WORKER {os.getpid()}] Action Plan Generation failed for {sub_id}: {e}", file=sys.stderr)
                    action_plan = [{"Phase": "ERROR", "Goal": "Action Plan generation failed."}]

        # 4. Aggregate Final Results
        final_sub_result = {
            "sub_criteria_id": sub_id,
            "sub_criteria_name": sub_criteria.get('sub_criteria_name', 'N/A'),
            "highest_full_level": highest_full_level,
            "weight": sub_criteria.get('weight', 0),
            "target_level_achieved": highest_full_level >= config.target_level,
            "weighted_score": worker_engine._calculate_weighted_score(highest_full_level, sub_criteria.get('weight', 0)),
            "action_plan": action_plan,
            "raw_results_ref": raw_results_for_sub,
        }
        
        print(f"[WORKER {os.getpid()}] Finished assessment for {sub_id}. L{highest_full_level} achieved.", file=sys.stderr)
        return raw_results_for_sub, final_sub_result
    

    # -------------------- Main Execution --------------------
    def run_assessment(
        self, 
        target_sub_id: str = "all", 
        export: bool = False, 
        vectorstore_manager: Optional['VectorStoreManager'] = None
    ) -> Dict[str, Any]:
        """
        Main runner for the assessment engine.
        Implements sequential maturity check (L1 -> L2 -> L3...) and multiprocessing.
        """
        start_ts = time.time()
        
        # 1. Filter Rubric based on target_sub_id
        if target_sub_id.lower() == "all":
            sub_criteria_list = self.rubric
        else:
            sub_criteria_list = [s for s in self.rubric if s.get('sub_id') == target_sub_id]
            if not sub_criteria_list:
                logger.error(f"Sub-Criteria ID '{target_sub_id}' not found in rubric.")
                return {"error": f"Sub-Criteria ID '{target_sub_id}' not found."}

        # Reset storage
        self.raw_llm_results = []
        self.final_subcriteria_results = []
        
        # 🟢 Core Logic Switch for Parallel Execution
        run_parallel = (target_sub_id.lower() == "all" and not self.config.force_sequential)
        
        if run_parallel:
            logger.info("Starting Parallel Assessment (All Sub-Criteria) with Multiprocessing Pool...")
            
            sub_criteria_data_list = sub_criteria_list 
            engine_config_dict = self.config.__dict__ 
            worker_args = [(sub_data, engine_config_dict) for sub_data in sub_criteria_data_list]
            
            try:
                with multiprocessing.Pool(processes=max(1, os.cpu_count() - 1)) as pool:
                    results_tuples = pool.starmap(self._assess_single_sub_criteria_worker, worker_args)
            except Exception as e:
                logger.critical(f"Multiprocessing Pool Execution Failed: {e}")
                logger.exception("FATAL: Multiprocessing pool failed to execute worker functions.")
                raise
            
            for raw_results_for_sub, final_sub_result in results_tuples:
                self.raw_llm_results.extend(raw_results_for_sub) 
                self.final_subcriteria_results.append(final_sub_result)

        else:
            run_mode_desc = target_sub_id if target_sub_id.lower() != 'all' else 'All Sub-Criteria (Forced Sequential)'
            logger.info(f"Starting Sequential Assessment for: {run_mode_desc}")
            
            local_vsm = vectorstore_manager
            
            if self.config.mock_mode == "none":
                logger.info("Sequential run: Re-instantiating VectorStoreManager locally in main process for robustness.")
                try:
                    local_vsm = load_all_vectorstores(
                        doc_types=[EVIDENCE_DOC_TYPES], 
                        evidence_enabler=self.config.enabler
                    )
                except Exception as e:
                    logger.error(f"FATAL: Local VSM Re-instantiation Failed for Sequential Run: {e}")
                    raise
            
            if self.config.mock_mode == "none" and not local_vsm:
                logger.error("VectorStoreManager is required for sequential execution in non-mock mode.")
                raise ValueError("VSM missing in sequential non-mock mode.")

            for sub_criteria in sub_criteria_list:
                sub_id = sub_criteria['sub_id']
                sub_criteria_name = sub_criteria['sub_criteria_name']
                sub_weight = sub_criteria.get('weight', 0)
                
                logger.info(f"\n[START] Assessing Sub-Criteria: {sub_id} - {sub_criteria_name} (Weight: {sub_weight})")
                
                highest_full_level = INITIAL_LEVEL - 1 
                is_passed_current_level = True
                raw_results_for_sub_seq = [] 
                
                for statement_data in sub_criteria.get('levels', []):
                    level = statement_data.get('level')
                    
                    if level is None or level > self.target_level:
                        continue 
                    
                    if not is_passed_current_level:
                        logger.warning(f"  > Skipping L{level}: L{level-1} already failed. Sequential check terminated.")
                        break 

                    result = self._run_single_assessment(
                        sub_criteria=sub_criteria,
                        statement_data=statement_data,
                        vectorstore_manager=local_vsm 
                    )
                    
                    self.raw_llm_results.append(result)
                    raw_results_for_sub_seq.append(result)
                    is_passed_current_level = result.get('is_passed', False)
                    
                    if is_passed_current_level:
                        highest_full_level = level
                
                target_plan_level = highest_full_level + 1
                action_plan = []
                
                if target_plan_level <= MAX_LEVEL and highest_full_level < self.target_level: 
                    logger.info(f"  > Generating Action Plan: Target L{target_plan_level}...")
                    
                    failed_statements_for_plan = [
                        r for r in raw_results_for_sub_seq
                        if r.get("level") == target_plan_level
                    ]
                    
                    if failed_statements_for_plan:
                        try:
                             action_plan = self.action_plan_generator(
                                failed_statements_data=failed_statements_for_plan,
                                sub_id=sub_id, enabler=self.enabler_id, target_level=target_plan_level 
                             )
                        except Exception as e:
                            logger.error(f"Action Plan Generation failed for {sub_id}: {e}")
                            action_plan = [{"Phase": "ERROR", "Goal": "Action Plan generation failed."}]
                    
                final_sub_result = {
                    "sub_criteria_id": sub_id,
                    "sub_criteria_name": sub_criteria_name,
                    "highest_full_level": highest_full_level,
                    "weight": sub_weight,
                    "target_level_achieved": highest_full_level >= self.target_level,
                    "weighted_score": self._calculate_weighted_score(highest_full_level, sub_weight),
                    "action_plan": action_plan,
                    "raw_results_ref": raw_results_for_sub_seq 
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
        
        if export:
             # อัปเดตการเรียกใช้เมธอดให้ส่ง Sub-ID เข้าไปด้วย
             export_path = self._export_results(
                 data=final_results, 
                 target_id=target_sub_id # <--- เพิ่มตัวแปรนี้
             )
             final_results["export_path_used"] = export_path

        return final_results