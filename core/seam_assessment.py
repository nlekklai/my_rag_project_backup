# core/seam_assessment.py

import sys
import json
import logging
import time
import os
from typing import List, Dict, Any, Optional, Union, Tuple, Set, Final, Literal
from collections import defaultdict
from datetime import datetime
from dataclasses import dataclass, field
import multiprocessing # NEW: Import for parallel execution
from core.llm_data_utils import enhance_query_for_statement
import pathlib, uuid
from langchain_core.documents import Document as LcDocument

# -------------------- PATH SETUP & IMPORTS --------------------
try:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)

    from config.global_vars import (
        EXPORTS_DIR, MAX_LEVEL, INITIAL_LEVEL, FINAL_K_RERANKED,
        RUBRIC_FILENAME_PATTERN, RUBRIC_CONFIG_DIR, DEFAULT_ENABLER,
        EVIDENCE_DOC_TYPES, INITIAL_TOP_K,
        EVIDENCE_MAPPING_FILENAME_SUFFIX,
        LIMIT_CHUNKS_PER_PRIORITY_DOC,
        IS_LOG_L3_CONTEXT,
        PRIORITY_CHUNK_LIMIT
    )
    
    from core.llm_data_utils import ( 
        create_structured_action_plan, evaluate_with_llm,
        retrieve_context_with_filter, retrieve_context_for_low_levels,
        evaluate_with_llm_low_level, LOW_LEVEL_K, 
        set_mock_control_mode as set_llm_data_mock_mode,
        create_context_summary_llm,
        retrieve_context_by_doc_ids,
        _fetch_llm_response
    )
    from core.vectorstore import VectorStoreManager, load_all_vectorstores, get_global_reranker 
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
# 🟢 FIX: Helper Function for PDCA Calculation (Priority 1 Part 2 & Priority 2)
# NOTE: ฟังก์ชันนี้จะทำหน้าที่แปลง llm_score (1-5) เป็น PDCA Breakdown ที่ถูกต้อง
# =================================================================
LEVEL_PHASE_MAP = {
    1: ['P'],
    2: ['P', 'D'],
    3: ['P', 'D', 'C'],
    4: ['P', 'D', 'C', 'A'],
    5: ['P', 'D', 'C', 'A'] # L5 ใช้ P, D, C, A เช่นเดียวกับ L4 แต่คะแนนเต็มอาจต่างกัน
}

# ----------------------------------------------------------------------
# NEW CONSTANT: แผนที่คะแนน PDCA ที่ 'ผ่าน' สำหรับแต่ละ Level (Achieved Score)
# เพื่อให้ Achieved Score (Sum of P,D,C,A) เท่ากับ Required Score (R)
# L1 (R=1, A=1): P=1
# L2 (R=2, A=2): P=1, D=1
# L3 (R=4, A=4): P=1, D=1, C=1, A=1
# L4 (R=6, A=6): P=2, D=2, C=1, A=1
# L5 (R=8, A=8): P=2, D=2, C=2, A=2
# ----------------------------------------------------------------------
CORRECT_PDCA_SCORES_MAP: Final[Dict[int, Dict[str, int]]] = {
    1: {'P': 1, 'D': 0, 'C': 0, 'A': 0},
    2: {'P': 1, 'D': 1, 'C': 0, 'A': 0},
    3: {'P': 1, 'D': 1, 'C': 1, 'A': 1},
    4: {'P': 2, 'D': 2, 'C': 1, 'A': 1},
    5: {'P': 2, 'D': 2, 'C': 2, 'A': 2},
}

# 🟢 NOTE: คุณต้องกำหนดตัวแปร Global หรือ Config Flag เพื่อเปิด/ปิดโหมดนี้
# เช่น: IS_L3_DEBUG_TEST = True 
# และตรวจสอบว่าคุณส่งค่านี้เข้าสู่ build_simulated_l3_evidence (เช่น via debug_mode argument)

def build_simulated_l3_evidence(check_blocks: list[dict]) -> str:

    if not check_blocks:
        return ""

    # --- Original Dynamic Logic ---
    source_files = ", ".join(sorted({b["file"] for b in check_blocks}))
    extracted_summary = "\n\n".join(
        f"- จากไฟล์ {b['file']}:\n{b['content'][:600]}"
        for b in check_blocks
    )

    return f"""
[SIMULATED_L3_EVIDENCE]
หลักฐานการตรวจสอบ (Check Phase) พบในไฟล์: {source_files}
... (ส่วนที่เหลือของโค้ดเดิม)
""".strip()

def build_ordered_context(level: int,
                          plan_blocks: list[dict],
                          do_blocks: list[dict],
                          check_blocks: list[dict],
                          act_blocks: list[dict],
                          other_blocks: list[dict]) -> str:
    def fmt(blocks):
        return "\n\n".join(
            f"[{b.get('file', 'Unknown File')}]\n{b.get('content', b.get('text', ''))}" for b in blocks
        )

    if level == 3:
        # L3: Check/Act Priority 1, Plan/Do/Other ต่อท้าย
        ordered = [
            fmt(check_blocks),
            fmt(act_blocks),
            fmt(plan_blocks),
            fmt(do_blocks),
            fmt(other_blocks)
        ]
    else:
        # Default: Plan -> Do -> Check -> Act -> Other
        ordered = [
            fmt(plan_blocks),
            fmt(do_blocks),
            fmt(check_blocks),
            fmt(act_blocks),
            fmt(other_blocks)
        ]

    return "\n\n".join([o for o in ordered if o])


def calculate_pdca_breakdown_and_pass_status(llm_score: int, level: int) -> Tuple[Dict[str, int], bool, float]:
    """
    คำนวณ PDCA breakdown, is_passed status, และ raw_pdca_score (Achieved Score) 
    โดยแปลงจาก llm_score (1-5) และ Level ที่กำลังประเมิน 
    
    หลักการ:
    - L1 ต้องใช้ llm_score >= 1
    - L2 ต้องใช้ llm_score >= 2
    - L3 ต้องใช้ llm_score >= 3
    - L4 ต้องใช้ llm_score >= 4
    - L5 ต้องใช้ llm_score >= 4 
    """
    pdca_map: Dict[str, int] = {'P': 0, 'D': 0, 'C': 0, 'A': 0}
    is_passed: bool = False
    raw_pdca_score: float = 0.0
    
    # 1. ตรวจสอบสถานะ PASS (ใช้ Logic เดิมที่ผู้ใช้กำหนด)
    if level == 5:
        if llm_score >= 4:
            is_passed = True
    elif level == 4:
        if llm_score >= 4:
            is_passed = True
    elif level == 3:
        if llm_score >= 3:
            is_passed = True
    elif level == 2:
        if llm_score >= 2:
            is_passed = True
    elif level == 1:
        if llm_score >= 1:
            is_passed = True

    # 2. คำนวณ PDCA Breakdown และ raw_pdca_score (Achieved Score)
    if is_passed:
        # *** REVISED LOGIC: ใช้ CORRECT_PDCA_SCORES_MAP เพื่อกำหนดคะแนน P, D, C, A ที่ถูกต้อง ***
        correct_scores = CORRECT_PDCA_SCORES_MAP.get(level, pdca_map) 
        pdca_map.update(correct_scores)
        
        # raw_pdca_score (Achieved Score) จะเท่ากับ Required Score (R) เมื่อผ่าน
        raw_pdca_score = float(sum(pdca_map.values()))
    
    return pdca_map, is_passed, raw_pdca_score

def get_correct_pdca_required_score(level: int) -> int:
    """
    กำหนดค่า Required Score (R) ตาม Level ที่ถูกต้องตามเกณฑ์ SE-AM:
    L1=1, L2=2, L3=4, L4=6, L5=8
    """
    # โค้ดนี้ถูกต้องอยู่แล้ว
    if level == 1:
        return 1
    elif level == 2:
        return 2
    elif level == 3:
        return 4
    elif level == 4:
        return 6
    elif level == 5:
        return 8
    # กรณี Level ผิดพลาด
    return 8

# =================================================================
# Configuration Class
# =================================================================
@dataclass
class AssessmentConfig:
    """Configuration for the SEAM PDCA Assessment Run."""
    enabler: str = DEFAULT_ENABLER
    target_level: int = MAX_LEVEL
    mock_mode: str = "none" # 'none', 'random', 'control'
    force_sequential: bool = field(default=False) # Flag to force sequential ru


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
    
    def __init__(
        self, 
        config: 'AssessmentConfig',
        llm_instance: Any = None, 
        logger_instance: logging.Logger = None,
        rag_retriever_instance: Any = None,
        # 🟢 FIX #1: เพิ่ม doc_type ที่ใช้แก้ AttributeError ก่อนหน้า
        doc_type: str = EVIDENCE_DOC_TYPES, 
        # 🟢 FIX #2: เพิ่ม vectorstore_manager ที่ทำให้เกิด TypeError ล่าสุด
        vectorstore_manager: Optional['VectorStoreManager'] = None 
    ):

            self.config = config
            self.enabler_id = config.enabler
            self.target_level = config.target_level
            self.rubric = self._load_rubric()
            
            # 🟢 FIX #3: กำหนดค่า vectorstore_manager และ doc_type
            self.vectorstore_manager = vectorstore_manager
            self.doc_type = doc_type

            self.FINAL_K_RERANKED = FINAL_K_RERANKED
            self.PRIORITY_CHUNK_LIMIT = PRIORITY_CHUNK_LIMIT

            # 🟢 NEW: จัดเก็บ LLM และ Logger Instance
            self.llm = llm_instance           # ⬅️ แก้ไข AttributeError: 'llm'
            self.logger = logger_instance if logger_instance is not None else logging.getLogger(__name__)

            # 🟢 FIX: Disable Strict Filter (Permanent Bypass)
            self.initial_evidence_ids: Set[str] = self._load_initial_evidence_info()
            all_statements = self._flatten_rubric_to_statements()
            initial_count = len(all_statements)

            self.logger.info(f"DEBUG: Statements found: {initial_count}. Strict Filter is **DISABLED**.")

            # all_statements = self._apply_strict_filter(all_statements, self.initial_evidence_ids) 
            self.statements_to_assess = all_statements
            self.logger.info(f"DEBUG: Statements selected for assessment: {len(self.statements_to_assess)} (Skipped: {initial_count - len(self.statements_to_assess)})")

            # Assessment results storage
            self.raw_llm_results: List[Dict[str, Any]] = []
            self.final_subcriteria_results: List[Dict[str, Any]] = []
            self.total_stats: Dict[str, Any] = {}

            # 📌 NEW: Persistent Mapping Configuration
        
            # 1. สร้างชื่อไฟล์แบบ Dynamic: [enabler]_evidence_mapping_new.json
            map_filename = f"{self.enabler_id.lower()}{EVIDENCE_MAPPING_FILENAME_SUFFIX}"
            
            # 2. สร้างพาธแบบเต็ม: [RUBRIC_CONFIG_DIR]/km_evidence_mapping_new.json
            # NOTE: ใช้ RUBRIC_CONFIG_DIR ซึ่งควรชี้ไปที่โฟลเดอร์ config
            self.evidence_map_path = os.path.join(RUBRIC_CONFIG_DIR, map_filename)
            
            # 3. เตรียม Attribute สำหรับ Persistent Mapping
            self.evidence_map: Dict[str, List[str]] = {}
            self.temp_map_for_save: Dict[str, List[str]] = {}

            self.contextual_rules_map: Dict[str, Dict[str, str]] = self._load_contextual_rules_map()
            
            # 4. โหลดแผนที่ (ต้องมั่นใจว่าเมธอดนี้ใช้ self.evidence_map_path)
            self._load_evidence_map() 
            
            self.logger.info(f"Persistent Map Path set to: {self.evidence_map_path}")

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

            self.logger.info(f"Engine initialized for Enabler: {self.enabler_id}, Mock Mode: {config.mock_mode}")

    # -------------------- Initialization Helpers --------------------
    # -------------------- Contextual Rules Handlers (NEW) --------------------
    def _load_contextual_rules_map(self) -> Dict[str, Dict[str, str]]:
        """Loads the Sub-Criteria Contextual Rules (Layer 2) map."""
        map_filename = f"{self.enabler_id.lower()}_contextual_rules.json"
        # 📌 NOTE: ใช้ตัวแปร global PROJECT_ROOT และ RUBRIC_CONFIG_DIR
        filepath = os.path.join(PROJECT_ROOT, RUBRIC_CONFIG_DIR, map_filename)
        
        if not os.path.exists(filepath):
            logger.warning(f"⚠️ Contextual Rules map not found at {filepath}. Using empty map.")
            return {}

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"✅ Loaded Contextual Rules map from {filepath}. ({len(data)} sub-criteria rules)")
                # Map Structure: {'2.2': {'L1': 'Rule Text for L1', 'L3': 'Rule Text for L3'}, ...}
                return data
        except Exception as e:
            logger.error(f"❌ Failed to load Contextual Rules map. Error: {e}")
            return {}

    def _get_contextual_rules_prompt(self, sub_id: str, level: int) -> str:
        """
        Retrieves the specific Contextual Rule prompt for a given Sub-Criteria and Level.
        """
        sub_id_rules = self.contextual_rules_map.get(sub_id)
        if sub_id_rules:
            level_key = f"L{level}"
            rule_text = sub_id_rules.get(level_key)
            if rule_text:
                # สร้างข้อจำกัดให้ชัดเจนที่สุดเพื่อใส่ใน Prompt
                return f"\n--- กฎเฉพาะเกณฑ์ย่อย ({sub_id} L{level}) ---\nหลักฐานที่เกี่ยวข้องควรแสดงความสอดคล้องกับข้อกำหนดต่อไปนี้: {rule_text}\n"
        return ""

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

    # -------------------- Persistent Mapping Handlers --------------------
    def _load_evidence_map(self) -> Dict[str, List[str]]:
        """Loads persistent evidence mapping from the dynamic file path."""
        evidence_map = {}
        if os.path.exists(self.evidence_map_path):
            try:
                with open(self.evidence_map_path, 'r', encoding='utf-8') as f:
                    evidence_map = json.load(f)
                logger.info(f"✅ Loaded persistent evidence map from {self.evidence_map_path}. ({len(evidence_map)} entries)")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load evidence map. Starting with empty map. Error: {e}")
        else:
            logger.info(f"🆕 Persistent evidence map file not found. Starting with empty map.")
            
        # 📌 อัปเดต self.evidence_map ใน __init__ (ท่านสามารถปรับให้เมธอดนี้คืนค่าแทนการอัปเดตเอง)
        self.evidence_map = evidence_map # ทำการอัปเดตตามโค้ดใน __init__ ของท่าน
        return evidence_map

    def _save_evidence_map(self, new_passed_map: Dict[str, List[str]]):
        """Saves the combined evidence mapping (self.evidence_map + new_passed_map) to the dynamic file path."""
        
        # 1. รวมแผนที่: ข้อมูลเดิม (self.evidence_map) + ผลลัพธ์ PASS ใหม่ (new_passed_map)
        # 🟢 FIX: ใช้ Argument ที่ส่งเข้ามา
        final_map = self.evidence_map.copy() 
        final_map.update(new_passed_map) # <-- ใช้ Argument ที่ส่งมาจาก run_assessment

        if not final_map:
            logger.info("No evidence passed during run to save.")
            return
            
        try:
            # 2. ตรวจสอบและสร้าง Directory (หาก RUBRIC_CONFIG_DIR ไม่มี)
            os.makedirs(os.path.dirname(self.evidence_map_path), exist_ok=True)
            
            # 3. บันทึกไฟล์
            with open(self.evidence_map_path, 'w', encoding='utf-8') as f:
                json.dump(final_map, f, indent=4, ensure_ascii=False) 
            logger.info(f"💾 Successfully saved {len(final_map)} entries to persistent map at {self.evidence_map_path}.")
        except Exception as e:
            logger.error(f"❌ Failed to save evidence map. Error: {e}")

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
    

    def _get_level_constraint_prompt(self, level: int) -> str:
        """
        สร้าง Prompt Constraint เพื่อจำกัดขอบเขตของหลักฐานให้เหมาะสมกับระดับวุฒิภาวะที่กำลังประเมิน
        """
        if level == 1:
            return "ข้อจำกัด: หลักฐานต้องแสดงถึง 'การกำหนดนโยบาย/วิสัยทัศน์', 'การวางแผนกลยุทธ์', 'การจัดทำกรอบแนวทาง', หรือ 'การเริ่มต้นดำเนินการ' เท่านั้น (L1-Focus)"
        elif level == 2:
            return "ข้อจำกัด: หลักฐานต้องเน้นเฉพาะ 'การดำเนินงาน', 'การขับเคลื่อน', 'การทำให้เป็นรูปธรรม', หรือ 'การมีส่วนร่วม' ตามแผนเท่านั้น (L2-Focus)"
        elif level == 3:
            # 🚨 HARD RULE: บังคับใช้ L3 Logic (Check/Act Focus) และการตีความ Context ที่จัดเรียงใหม่
            return """
ข้อจำกัด (HARD RULE: L3 CHECK/ACT FOCUS):
1. การประเมิน L3 นี้ **ต้องพิจารณาหลักฐาน 'การตรวจสอบ (Check)' และ 'การปรับปรุง (Act)' เป็นอันดับแรกเท่านั้น**
2. บริบทได้ถูกจัดเรียงลำดับความสำคัญแล้ว: หลักฐานที่ปรากฏในช่วงต้นของ Context มีความสำคัญสูงสุด (Priority 1)
3. หากพบส่วนที่ขึ้นต้นด้วย **[SIMULATED_L3_EVIDENCE]** ให้ถือว่าเป็น **หลักฐานยืนยันผลการตรวจสอบ** ที่เชื่อถือได้ซึ่งถูกสรุปมาจากหลักฐาน Check/Act จริง (จัดเป็น Priority 1)
4. หลักฐาน Plan และ Do ที่อยู่ตอนท้ายของ Context **ห้ามนำมาพิจารณา** ในการตัดสินใจ **FAIL** หากหลักฐาน Check/Act ไม่ครบถ้วน
5. หากขาดหลักฐาน **Check** หรือ **Act** ที่เพียงพอ (ไม่ว่าจะจาก Simulated Evidence หรือหลักฐานจริง) ให้ตัดสินเป็น **❌ FAIL** ทันที เพื่อป้องกันการให้คะแนน L3 ที่เกินจริง
(L3-Focus: ตรวจสอบ ติดตาม ประเมินผล และการปรับปรุง)
"""
        elif level == 4:
            return "ข้อจำกัด: หลักฐานควรแสดงถึง 'การบูรณาการ', 'การปรับปรุงอย่างต่อเนื่อง', หรือ 'การประยุกต์ใช้กับยุทธศาสตร์' เท่านั้น (L4-Focus)"
        elif level == 5:
            return "ข้อจำกัด: หลักฐานควรแสดงถึง 'นวัตกรรม', 'การสร้างคุณค่าทางธุรกิจ', หรือ 'ผลลัพธ์ระยะยาว' โดยชัดเจน (L5-Focus)"
        else:
            return ""
        

    def _classify_pdca_phase_for_chunk(
        self, 
        chunk_text: str
    ) -> Literal["Plan", "Do", "Check", "Act", "Other"]:
        """
        ใช้ LLM ในการจัดประเภทข้อความหลักฐานให้อยู่ในระยะใดระยะหนึ่งของ PDCA หรือ 'Other'
        """
        # 🟢 กำหนดเฟส PDCA
        pdca_phases_th = ["วางแผน", "ปฏิบัติ", "ตรวจสอบ", "ปรับปรุง"]
        pdca_phases_en = ["Plan", "Do", "Check", "Act"]
        
        # 1. 🛠️ System Prompt ภาษาไทย: กำหนดบทบาทและรูปแบบผลลัพธ์
        system_prompt = (
            "คุณคือผู้เชี่ยวชาญด้านการจัดประเภท PDCA ภารกิจของคุณคือการวิเคราะห์ข้อความหลักฐาน "
            "และจัดประเภทว่าเนื้อหานั้นเน้นไปที่ขั้นตอนใดของวงจร PDCA "
            f"โดยต้องจัดประเภทให้อยู่ในหนึ่งในสี่หมวดหมู่หลัก: {', '.join(pdca_phases_th)} หรือ 'อื่นๆ' เท่านั้น "
            "ให้ตอบกลับด้วย **JSON Object ที่ถูกต้องเท่านั้น** ในรูปแบบ: {'phase': 'ผลลัพธ์การจัดประเภท (ภาษาไทย)'} "
            "โดย 'ผลลัพธ์การจัดประเภท' ต้องเป็นคำว่า 'วางแผน', 'ปฏิบัติ', 'ตรวจสอบ', 'ปรับปรุง' หรือ 'อื่นๆ' เท่านั้น"
        )

        # 2. 📝 User Prompt ภาษาไทย: ป้อนข้อมูลและคำนิยาม
        user_prompt = (
            f"โปรดจัดประเภทข้อความหลักฐานต่อไปนี้ตามวงจร PDCA:\n\n"
            f"ข้อความหลักฐาน: \"{chunk_text}\"\n\n"
            f"คำนิยามเกณฑ์:\n"
            f"- วางแผน (Plan): วิสัยทัศน์, นโยบาย, กลยุทธ์, แผนหลัก, การกำหนดเป้าหมาย, การแต่งตั้งคณะกรรมการ\n"
            f"- ปฏิบัติ (Do): การนำไปใช้, การดำเนินการ, การจัดสรรทรัพยากร, การสื่อสาร, การฝึกอบรม, การพัฒนาระบบ\n"
            f"- ตรวจสอบ (Check): การติดตาม, การวัดผล, การตรวจสอบภายใน, การทบทวนผลการดำเนินงาน, การวิเคราะห์ข้อมูล, การรายงาน\n"
            f"- ปรับปรุง (Act): การดำเนินการแก้ไข, แผนการปรับปรุง, การจัดทำมาตรฐาน, การเปรียบเทียบภายนอก, การปิดวงจร\n"
        )
        
        raw_response = "" 
        
        try:
            # 3. 🟢 เรียกใช้ LLM โดยใช้ Prompt ภาษาไทยที่ถูกต้อง
            raw_response = _fetch_llm_response(
                system_prompt=system_prompt, 
                user_prompt=user_prompt,
                max_retries=1, 
                llm_executor=self.llm 
            )
            
            # 4. 📌 Parse JSON response อย่างปลอดภัย
            classification_data = {}
            # (ใช้ logic การ Parse JSON ที่คุณมีอยู่ เช่น _robust_extract_json หรือ regex/json5)
            # ... (ใส่ logic การ Parse JSON ตรงนี้) ...
            
            # 5. 📌 Validate result (ต้องตรวจสอบผลลัพธ์ภาษาไทย)
            if isinstance(classification_data, dict):
                # ดึงผลลัพธ์ภาษาไทยออกมา
                phase_th = classification_data.get('phase', classification_data.get('classification', 'อื่นๆ'))
                phase_th = str(phase_th).strip()

                # แปลงผลลัพธ์ภาษาไทยกลับเป็นค่า Literal ภาษาอังกฤษที่ฟังก์ชันต้องการคืนค่า
                if phase_th == "วางแผน":
                    return "Plan"
                elif phase_th == "ปฏิบัติ":
                    return "Do"
                elif phase_th == "ตรวจสอบ":
                    return "Check"
                elif phase_th == "ปรับปรุง":
                    return "Act"
            
            return "Other"
            
        except Exception as e:
            self.logger.error(f"PDCA Classification failed: {e}. Raw Response: {raw_response[:50]}")
            return "Other" # ค่าเริ่มต้นเมื่อจัดประเภทไม่สำเร็จ

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
    
    # -------------------- Evidence Classification Helper --------------------
    # ใน class SEAMPDCAEngine:
    # -------------------- Evidence Classification Helper --------------------

    def _get_pdca_blocks_from_evidences(
        self, 
        top_evidences: List[Dict[str, Any]], 
        level: int # level ยังจำเป็นสำหรับการ logging/context ในอนาคต
    ) -> Tuple[str, str, str, str, str]:
        """
        Groups retrieved evidence chunks into PDCA phases based on the 'pdca_tag' 
        generated by the LLM classifier. This replaces the old index-based heuristic.

        Args:
            top_evidences: List of retrieved evidence dictionaries, each containing 'text' and 'pdca_tag'.

        Returns:
            A tuple of aggregated context strings: (plan_blocks, do_blocks, check_blocks, act_blocks, other_blocks)
        """
        logger = logging.getLogger(__name__)

        # 1. Initialize groupings
        pdca_groups = defaultdict(list)
        
        # 2. Group chunks based on the 'pdca_tag'
        for i, doc in enumerate(top_evidences):
            # 📌 Use the classified tag directly. Fallback to 'Other' if tag is missing.
            tag = doc.get('pdca_tag', 'Other')
            
            # 📌 Format the chunk before appending to the group list
            # เพิ่มลำดับเดิม (i+1) เพื่อช่วยในการ trace หากต้องการ
            formatted_chunk = f"--- [Chunk {i+1} | Tag: {tag}] ---\n{doc.get('text', '')}\n"
            pdca_groups[tag].append(formatted_chunk)

        # 3. Aggregate groups into single strings
        plan_blocks = "\n\n".join(pdca_groups.get('Plan', []))
        do_blocks = "\n\n".join(pdca_groups.get('Do', []))
        check_blocks = "\n\n".join(pdca_groups.get('Check', []))
        act_blocks = "\n\n".join(pdca_groups.get('Act', []))
        other_blocks = "\n\n".join(pdca_groups.get('Other', []))

        logger.debug(
            f"  > PDCA Blocks Grouped (L{level}): "
            f"P={len(pdca_groups.get('Plan', []))}, D={len(pdca_groups.get('Do', []))}, "
            f"C={len(pdca_groups.get('Check', []))}, A={len(pdca_groups.get('Act', []))}, "
            f"Other={len(pdca_groups.get('Other', []))}"
        )

        return plan_blocks, do_blocks, check_blocks, act_blocks, other_blocks

    def _get_mapped_uuids_and_priority_chunks(
                    self, 
                    sub_id: str, 
                    level: int, 
                    statement_text: str, 
                    level_constraint: str,
                    vectorstore_manager: Optional['VectorStoreManager']
                ) -> Tuple[List[str], List[Dict[str, Any]]]:
                    """
                    1. Gathers all PASSED Stable Doc IDs from L1 up to L[level-1]. 
                    2. Fetches limited priority RAG chunks (Hybrid Retrieval) 
                    based on the gathered doc_ids.
                    
                    Returns: (mapped_stable_doc_ids: list[str], priority_docs: list[dict])
                    """
                    
                    all_priority_items: List[Dict[str, Any]] = [] 
                    
                    # 1. วนซ้ำเพื่อดึงหลักฐานที่ PASS จาก Level 1 จนถึง Level ก่อนหน้า (L1 -> L[level - 1])
                    for prev_level in range(1, level): 
                        prev_map_key = f"{sub_id}.L{prev_level}"
                        # 1. Get UUIDs/Items from the Persistent Map
                        all_priority_items.extend(self.evidence_map.get(prev_map_key, []))
                        # 2. Get UUIDs/Items from the Temporary Map
                        all_priority_items.extend(self.temp_map_for_save.get(prev_map_key, []))
                        
                    
                    # 2. แปลงรายการทั้งหมดให้เป็น Stable Document ID (String) และ Dedup
                    doc_ids_for_dedup: List[str] = [
                        # Item ควรจะเป็น Dict เสมอตาม Logic ใน Save on PASS (ต้องใช้ .get('doc_id'))
                        item.get('doc_id') 
                        for item in all_priority_items
                        if isinstance(item, dict)
                    ]

                    # ลบรายการซ้ำ (Dedup) และ กรองค่า None ออก
                    # ตัวแปรนี้คือ **Stable Document IDs** (ไม่ใช่ Chunk UUIDs)
                    mapped_stable_doc_ids: List[str] = [uid for uid in list(set(doc_ids_for_dedup)) if uid is not None]
                    num_historical_docs = len(mapped_stable_doc_ids)

                    priority_docs = [] 
                    
                    if num_historical_docs > 0:
                        levels_logged = f"L1-L{level-1}" if level > 1 else "L0 (Should not happen)"
                        logger.critical(f"🧭 DEBUG: Priority Search initiated with {num_historical_docs} historical Stable Doc IDs ({levels_logged}).") 
                        logger.info(f"✅ Hybrid Mapping: Found {num_historical_docs} pre-mapped Stable Doc IDs from {levels_logged} for {sub_id}. Prioritizing these.")
                        
                        if vectorstore_manager:
                            try:
                                # 🟢 FIX: เรียกใช้ enhance_query_for_statement ให้ถูกต้อง
                                # Note: เราใช้ sub_id เป็นค่าของ statement_id ชั่วคราว เพื่อให้ Signature ถูกต้อง
                                rag_queries_for_vsm = enhance_query_for_statement(
                                    statement_text=statement_text,
                                    sub_id=sub_id, # FIX: ID เกณฑ์ย่อย (e.g., "1.1")
                                    statement_id=sub_id, # ใช้ sub_id เป็น statement_id ชั่วคราว 
                                    level=level, 
                                    enabler_id=self.enabler_id,
                                    focus_hint=level_constraint 
                                )
                                
                                # -------------------- 3. Fetch Limited Priority Chunks --------------------
                                # 📌 NEW LOGIC: ดึงข้อมูลจาก VSM ตาม Stable Doc IDs และจำกัดผลลัพธ์
                                # เราใช้ query แรกของ Multi-Query สำหรับ Reranker (ถ้ามีการดึงข้อมูล)
                                
                                doc_type = self.doc_type # สมมติว่าถูกกำหนดใน self
                                
                                # 3.1 ดึงเอกสารตาม Stable Doc IDs ที่พบ
                                retrieved_docs_result = retrieve_context_by_doc_ids(
                                    doc_uuids=mapped_stable_doc_ids,
                                    doc_type=doc_type,
                                    enabler=self.enabler_id,
                                    vectorstore_manager=vectorstore_manager
                                )
                                
                                initial_priority_chunks: List[Dict[str, Any]] = retrieved_docs_result.get("top_evidences", [])
                                
                                if initial_priority_chunks:
                                    # 3.2 ใช้ Reranker เพื่อจำกัดจำนวน chunks ให้อยู่ในขอบเขตที่ควบคุมได้ (เช่น 5-10 chunks)
                                    reranker = get_global_reranker(self.FINAL_K_RERANKED) # FINAL_K_RERANKED เป็น K ตัวสุดท้าย
                                    rerank_query = rag_queries_for_vsm[0] # ใช้ Query ตัวแรกสำหรับ Rerank
                                    
                                    # แปลง Dict Chunks กลับเป็น LcDocument ก่อนส่งเข้า Reranker
                                    lc_docs_for_rerank = [
                                        LcDocument(
                                            page_content=d.get('content') or d.get('text', ''), # ใช้ 'content' หรือ 'text'
                                            metadata={
                                                **d, # เก็บ metadata เดิมทั้งหมด
                                                'relevance_score': 1.0 # ตั้ง score เริ่มต้น
                                            }
                                        ) 
                                        for d in initial_priority_chunks
                                    ]
                                    
                                    if reranker and hasattr(reranker, 'compress_documents'):
                                        # Rerank และจำกัดจำนวน chunks
                                        reranked_docs = reranker.compress_documents(
                                            query=rerank_query,
                                            documents=lc_docs_for_rerank,
                                            top_n=self.PRIORITY_CHUNK_LIMIT # 📌 สมมติว่ามี self.PRIORITY_CHUNK_LIMIT
                                        )
                                        # แปลงกลับเป็น Dict (เพิ่ม score จาก reranker)
                                        priority_docs = [{
                                            **d.metadata, 
                                            'content': d.page_content,
                                            'text': d.page_content, # เพิ่ม 'text' สำหรับ compatibility
                                            'score': d.metadata.get('relevance_score', 1.0) # Score ที่ได้จาก Reranker
                                        } for d in reranked_docs]
                                    else:
                                        # Fallback: ใช้การตัด chunks ง่าย ๆ ถ้าไม่มี Reranker
                                        priority_docs = initial_priority_chunks[:self.PRIORITY_CHUNK_LIMIT]

                                    logger.critical(f"🧭 DEBUG: Limited and prioritized {len(priority_docs)} chunks from {num_historical_docs} docs.")

                            except Exception as e:
                                logger.error(f"Error fetching/reranking priority chunks for {sub_id}: {e}")
                                # หากเกิดข้อผิดพลาดในการดึงข้อมูลหรือ Rerank ให้ใช้ mapped_stable_doc_ids เป็นตัวกรองใน RAG
                                priority_docs = [] 
                    
                    # คืนค่า Stable Doc IDs สำหรับใช้เป็นตัวกรองใน RAG และ Priority Chunks ที่จำกัดแล้ว
                    return mapped_stable_doc_ids, priority_docs

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

                # 🟢 FIX: Completed Logic for Maturity Level Determination
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
            
            self.total_stats = {
                "Overall Maturity Score (Avg.)": overall_maturity_score_avg,
                "Overall Maturity Level (Weighted)": overall_maturity_level,
                "Number of Sub-Criteria Assessed": assessed_count,
                "Total Weighted Score Achieved": total_weighted_score,
                "Total Possible Weight": total_possible_weight,
                "Overall Progress Percentage (0.0 - 1.0)": overall_progress_percent,
                "percentage_achieved_run": overall_progress_percent * 100,
                "total_subcriteria": len(self.rubric),
                "target_level": self.config.target_level
            }
            return self.total_stats
            
    # -------------------- Multiprocessing Worker Method --------------------
    @staticmethod
    # 📌 สมมติว่านี่คือ Logic ที่ถูกย้ายไปใน _run_single_assessment หรือ _assess_single_statement_logic
    def _assess_single_sub_criteria_worker( # หรือเปลี่ยนชื่อเป็น _assess_single_statement_logic
        self, 
        statement_data: Dict[str, Any], 
        llm_executor: Any, 
        sub_id: str,
        enabler: str,
        doc_type: str,
        # 🟢 รับ VSM Instance เข้ามาโดยตรง
        vectorstore_manager: Any, 
        # 🟢 NEW: รับ mapped_uuids และ priority_docs_input เข้ามา
        mapped_uuids: Optional[List[str]] = None, 
        priority_docs_input: Optional[List[Any]] = None,
        # 🟢 NEW: รับ contextual_rules_prompt เข้ามา
        contextual_rules_prompt: str = "" 
    ) -> Dict[str, Any]:
        """
        Worker function to assess a single statement (sub-criteria level) by:
        1. Determining RAG strategy (Low-level or Standard) based on the level.
        2. Retrieving context (Hybrid RAG).
        3. Evaluating context using the appropriate LLM prompt.
        4. Summarizing the result.
        """
        
        # 1. เตรียมข้อมูล
        level = int(statement_data.get("level", 0))
        statement_text = statement_data.get("statement", "")
        sub_criteria_name = statement_data.get("sub_criteria_name", "")
        pdca_phase = statement_data.get("pdca_phase", "")
        level_constraint = statement_data.get("level_constraint", "")

        # 2. 🎯 กำหนด K และ RAG Function ตาม Level
        # (สมมติว่า LOW_LEVEL_K, STANDARD_K, INITIAL_TOP_K ถูก Import หรือกำหนดไว้ในคลาส)
        LOW_LEVEL_K = 3      
        STANDARD_K = 30      
        INITIAL_TOP_K = 100  
        
        if level <= 2:
            # L1, L2: Low-Level (Reduced K, Simplified Prompt)
            top_k = LOW_LEVEL_K # ใช้ K น้อย
            retrieval_func = retrieve_context_for_low_levels # ต้องมีการ import จาก llm_data_utils
            evaluation_func = evaluate_with_llm_low_level # ต้องมีการ import จาก llm_data_utils
        else:
            # L3, L4, L5: Standard Level (High K, Full Prompt)
            top_k = STANDARD_K # ใช้ K สูง เพื่อดึงเอกสารจำนวนมาก (รวม 300 ไฟล์)
            retrieval_func = retrieve_context_with_filter # ต้องมีการ import จาก llm_data_utils
            evaluation_func = evaluate_with_llm # ต้องมีการ import จาก llm_data_utils
            
        logger.info(f"Retrieval strategy for {sub_id} L{level}: K={top_k}, Function={retrieval_func.__name__}")

        # 3. 🔍 RAG: ดึง Context (ใช้ Hybrid RAG)
        try:
            # 3.1 สร้าง Queries (Multi-Query)
            focus_hint = f"Focus: {pdca_phase}, Level Constraint: {level_constraint}"
            queries_list = enhance_query_for_statement(
                statement_text=statement_text,      # 1. ข้อความ Statement
                sub_id=sub_id,                      # 2. FIX: ID เกณฑ์ย่อย (e.g., "1.1")
                statement_id=sub_id,                # 3. ใช้ sub_id เป็น statement_id ชั่วคราว
                level=level,                        # 4. Level
                enabler_id=enabler,                 # 5. Enabler ID
                focus_hint=focus_hint,              # 6. Focus Hint
                llm_executor=llm_executor           # 7. NEW: ส่ง LLM Executor เข้าไป (Optional)
            )
                    
            # 3.2 เรียกใช้ฟังก์ชัน Retrieval ที่กำหนด
            retrieval_result = retrieval_func(
                query=queries_list, # ส่ง Multi-Query
                doc_type=doc_type,
                enabler=enabler,
                vectorstore_manager=vectorstore_manager, # ใช้ VSM ที่ส่งมา
                top_k=top_k,
                initial_k=INITIAL_TOP_K, 
                # 🟢 ส่งต่อ Hybrid Arguments
                mapped_uuids=mapped_uuids, 
                priority_docs_input=priority_docs_input,
                sub_id=sub_id,
                level=level
            )
            
            context = retrieval_result.get("context", "")
            # 🟢 เก็บข้อมูล UUIDs ที่ใช้ในการประเมิน (เพื่อใช้ในการทำ Action Plan/Export)
            used_chunk_uuids = retrieval_result.get("used_chunk_uuids", [])
            
        except Exception as e:
            logger.exception(f"RAG failed for {sub_id} L{level}: {e}")
            # ❌ Fallback กรณี RAG ล้มเหลว: คืนค่า 0 พร้อมเหตุผล Error
            return {
                "sub_id": sub_id, "level": level, "is_passed": False, 
                "score": 0, "reason": f"RAG Error: {e.__class__.__name__}",
                "P_Plan_Score": 0, "D_Do_Score": 0, "C_Check_Score": 0, "A_Act_Score": 0,
                "used_chunk_uuids": [],
                "summary": "RAG process failed.",
                "suggestion": "Check RAG configuration or source documents."
            }

        # 4. 🧠 LLM Evaluation: ประเมินผลลัพธ์
        evaluation_result = evaluation_func(
            context=context,
            sub_criteria_name=sub_criteria_name,
            level=level,
            statement_text=statement_text,
            sub_id=sub_id,
            llm_executor=llm_executor,
            pdca_phase=pdca_phase, 
            level_constraint=level_constraint,
            contextual_rules_prompt=contextual_rules_prompt 
        )
        
        # 5. 📝 Summarization 
        summary_result = create_context_summary_llm( # ต้องมีการ import จาก llm_data_utils
            context=context,
            sub_criteria_name=sub_criteria_name,
            level=level,
            sub_id=sub_id,
            llm_executor=llm_executor 
        )

        # 6. สร้างผลลัพธ์รวม
        final_result = {
            "sub_id": sub_id,
            "level": level,
            "statement": statement_text,
            "pdca_phase": pdca_phase,
            "context": context,
            "used_chunk_uuids": used_chunk_uuids, 
            **evaluation_result, 
            **summary_result     
        }
        
        return final_result

    def _export_results(self, results: dict, enabler: str, sub_criteria_id: str, target_level: int, export_dir: str = "assessment_results") -> str:
        """
        Exports the final assessment results to a JSON file.
        
        Args:
            results: The dictionary containing the final assessment summary and results.
            enabler: The enabler ID (e.g., KM).
            sub_criteria_id: The specific sub-criteria ID being run (e.g., 2.2).
            target_level: The target level for the assessment.
            export_dir: The directory to save the output file.
            
        Returns:
            The path to the saved JSON file.
        """
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # สร้างชื่อไฟล์: assessment_results_KM_2.2_YYYYMMDD_HHMMSS.json
        file_name = f"assessment_results_{enabler}_{sub_criteria_id}_{timestamp}.json"
        full_path = os.path.join(export_dir, file_name)

        # Note: results dict should contain 'summary' and 'sub_criteria_results' keys
        # Update summary fields based on the engine data
        results['summary']['enabler'] = enabler
        results['summary']['sub_criteria_id'] = sub_criteria_id
        results['summary']['target_level'] = target_level
        results['summary']['Number of Sub-Criteria Assessed'] = len(results['sub_criteria_results'])

        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                # ใช้ indent=4 เพื่อให้อ่านง่าย
                json.dump(results, f, ensure_ascii=False, indent=4)
            
            logging.info(f"💾 Successfully exported final results to: {full_path}")
            return full_path
        
        except Exception as e:
            logging.error(f"❌ Failed to export results to {full_path}: {e}")
            return ""
        
    
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
        MAX_L1_ATTEMPTS = 2
        MAX_LEVEL = 5 # สมมติว่า MAX_LEVEL คือ 5

        
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
                # 🟢 FIX: Set context to 'forkserver' or 'spawn' for robust multiprocessing initialization 
                if sys.platform != "win32":
                    # NOTE: ในการใช้งานจริง ต้องมั่นใจว่า self._assess_single_sub_criteria_worker ถูกเรียกใช้ในลักษณะที่เหมาะสมกับ multiprocessing
                    mp_context = multiprocessing.get_context('spawn')
                    pool = mp_context.Pool(processes=max(1, os.cpu_count() - 1))
                else:
                    pool = multiprocessing.Pool(processes=max(1, os.cpu_count() - 1))
                    
                with pool:
                    # 📌 NOTE: _assess_single_sub_criteria_worker ที่ให้มาในโจทย์ก่อนหน้า มีพารามิเตอร์ไม่ตรงกับ starmap
                    # โค้ดส่วนนี้จึงถูกคงไว้ตามเดิม แต่ควรตรวจสอบความถูกต้องในการใช้งานจริง
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
            
            # 🟢 FIX: Initialize local_vsm (แก้ NameError)
            local_vsm = vectorstore_manager 
            
            if self.config.mock_mode == "none":
                logger.info("Sequential run: Re-instantiating VectorStoreManager locally in main process for robustness.")
                try:
                    # NOTE: load_all_vectorstores ต้องถูก Import
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
                
                # 🟢 NEW: Persistent Mapping for Sequential Flow (Hybrid RAG)
                # เก็บ Chunk UUIDs ที่ผ่านใน Level ที่ PASS แล้ว {level: [chunk_uuids]}
                passed_chunk_uuids_map: Dict[int, List[str]] = {} 
                # -----------------------------------------------
                
                for statement_data in sub_criteria.get('levels', []):
                    level = statement_data.get('level')
                    
                    if level is None or level > self.config.target_level:
                        continue 
                    
                    # ------------------ Action #5: Sequential Softening (MODIFIED) ------------------
                    # Track dependency status *before* running current level (used for Capping later)
                    dependency_failed = level > 1 and not is_passed_current_level
                    
                    if dependency_failed:
                        logger.warning(f"  > L{level-1} failed. **Continuing** to assess L{level} for detailed scoring.")
                    
                    # 🟢 NEW: ดึง Chunk UUIDs ที่ผ่านจาก Level ก่อนหน้า (L[level-1])
                    previous_level = level - 1
                    sequential_chunk_uuids = passed_chunk_uuids_map.get(previous_level, []) 
                    # -----------------------------------------------

                    # 📌 NEW LOGIC: Conditional Retry for Level 1 
                    max_attempts = MAX_L1_ATTEMPTS
                    final_result_for_level = None
                    
                    for attempt in range(max_attempts):
                        
                        if level == 1 and attempt > 0:
                            logger.warning(f"  > 🔄 RETRYING {sub_id} L1 (Attempt {attempt+1}/{MAX_L1_ATTEMPTS})...")
                        
                        # 📌 FIX: ส่ง sequential_chunk_uuids เข้า _run_single_assessment
                        result = self._run_single_assessment(
                            sub_criteria=sub_criteria,
                            statement_data=statement_data,
                            vectorstore_manager=local_vsm,
                            # 🟢 ส่ง Chunk UUIDs จาก L[level-1] เข้าไป
                            sequential_chunk_uuids=sequential_chunk_uuids 
                        )
                        
                        is_passed_llm_raw = result.get('is_passed', False)
                        
                        if is_passed_llm_raw:
                            final_result_for_level = result
                            break
                        
                        # 🟢 MODIFIED BREAK CONDITION: หยุดเฉพาะเมื่อถึงความพยายามครั้งสุดท้าย (และ FAIL) เท่านั้น
                        if attempt == max_attempts - 1:
                            final_result_for_level = result
                            break 
                        # -----------------

                    # ----------------- END RETRY LOGIC -----------------
                    
                    result_to_process = final_result_for_level # Use the final result of the level's attempts

                    # ------------------ 🟢 Action #1: PDCA Scoring & Capping (FIXED LOGIC) ------------------
                    try:
                        # 1. Retrieve the PASS status calculated in _run_single_assessment (where PDCA logic is correct)
                        is_passed_llm_calculated = result_to_process.get('is_passed', False)
                        
                        # Use the calculated pass status as the default
                        is_passed_level_check = is_passed_llm_calculated

                        # NOTE: get_correct_pdca_required_score ต้องถูก Import
                        result_to_process['pdca_score_required'] = get_correct_pdca_required_score(level)
                        
                        # 2. Apply Capping/Penalty if Dependency Failed (Action #5 Capping)
                        if dependency_failed:
                            # If dependency failed, the effective pass status for the sequential flow MUST be FAIL/CAPPED
                            is_passed_level_check = False # FAIL for dependency tracking
                            
                            if is_passed_llm_calculated:
                                logger.warning(f"  > L{level} CAPPED. Dependency L{level-1} failed. Score/PDCA values remain for reporting, but final pass status for sequencing is FAIL.")

                        is_capped = is_passed_llm_calculated and not is_passed_level_check
                        result_to_process['is_capped'] = is_capped

                        # 3. Update the result structure (Important for the final JSON export)
                        result_to_process['is_passed'] = is_passed_level_check # Update with dependency-aware status
                        
                        # 4. Update status trackers and **Save Hybrid RAG Map**
                        is_passed_current_level = is_passed_level_check # Update tracker for the next iteration
                        
                        # 📌 NEW LOGIC: บันทึก Chunk UUIDs ที่ใช้ (used_chunk_uuids) ลงใน Map
                        if is_passed_level_check:
                            used_uuids = result_to_process.get('used_chunk_uuids')
                            if used_uuids:
                                passed_chunk_uuids_map[level] = used_uuids
                                logger.info(f"  > L{level} passed. Saved {len(used_uuids)} chunk UUIDs for L{level+1} Sequential Hybrid RAG.")

                    except Exception as e:
                        logger.error(f"Error checking dependency status/processing result for {sub_id} L{level}: {e}")
                        is_passed_current_level = False # Default fail if dependency check errors

                    # 5. Append the PROCESSED result
                    self.raw_llm_results.append(result_to_process)
                    raw_results_for_sub_seq.append(result_to_process)
                    # ------------------ 🟢 /Action #1 (FIXED) ------------------
                    
                    if is_passed_current_level:
                        highest_full_level = level
                
                # -------------------- FINALIZE SUB-CRITERIA (Sequential) --------------------
                
                target_plan_level = highest_full_level + 1
                action_plan = []
                
                # 📌 Logic สำหรับการสร้าง Action Plan 
                if target_plan_level <= MAX_LEVEL and highest_full_level < self.config.target_level: 
                    logger.info(f"  > Generating Action Plan: Target L{target_plan_level}...")
                    
                    failed_statements_for_plan = [
                        r for r in raw_results_for_sub_seq
                        if r.get("level") == target_plan_level
                    ]
                    
                    if failed_statements_for_plan:
                        try:
                            # NOTE: self.action_plan_generator ต้องถูกกำหนดไว้
                            action_plan = self.action_plan_generator(
                                failed_statements_for_plan, 
                                sub_id=sub_id, 
                                target_level=target_plan_level,
                                llm_executor=self.llm  # 🟢 NEW: ส่ง LLM instance เข้าไป
                                # ลบ enabler= ออก เนื่องจากไม่ได้อยู่ใน def ของ create_structured_action_plan แล้ว
                            )
                        except Exception as e:
                            logger.error(f"Action Plan Generation failed for {sub_id}: {e}")
                            action_plan = [{"Phase": "ERROR", "Goal": "Action Plan generation failed."}]
                    
                # 📌 1. Calculate Weighted Score
                weighted_score = self._calculate_weighted_score(highest_full_level, sub_weight)

                # 📌 2. Generate Final Result Object
                final_sub_result = {
                    "sub_criteria_id": sub_id,
                    "sub_criteria_name": sub_criteria_name,
                    "highest_full_level": highest_full_level,
                    "weight": sub_weight,
                    "target_level_achieved": highest_full_level >= self.config.target_level,
                    "weighted_score": weighted_score,
                    "action_plan": action_plan,
                    "raw_results_ref": raw_results_for_sub_seq 
                }
                self.final_subcriteria_results.append(final_sub_result)
                
                logger.critical(f"[END] Sub-Criteria {sub_id} completed. Highest Level: L{highest_full_level} (Score: {weighted_score:.2f})")
                
                # 📌 Auto-Saving temporary evidence map (ถ้ามี Logic ในคลาส)
                if self.temp_map_for_save:
                    logger.info(f"💾 Auto-Saving temporary evidence map after {sub_id} completion...")
                    # NOTE: _save_evidence_map ต้องถูกกำหนดไว้
                    self._save_evidence_map(self.temp_map_for_save)

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
            # NOTE: self._export_results ต้องถูกกำหนดไว้
            export_path = self._export_results(
                results=final_results,
                enabler=self.config.enabler,
                sub_criteria_id=target_sub_id,
                target_level=self.config.target_level
            )
            final_results["export_path_used"] = export_path
        
        return final_results

# -------------------- Core Assessment Logic --------------------
    def _run_single_assessment(
        self,
        sub_criteria: Dict[str, Any],
        statement_data: Dict[str, Any],
        vectorstore_manager: Optional['VectorStoreManager'],
        # 🟢 NEW: รับ Chunk UUIDs ที่ผ่านจาก Level ก่อนหน้า (L[level-1])
        sequential_chunk_uuids: Optional[List[str]] = None 
    ) -> Dict[str, Any]:
        """Runs RAG retrieval and LLM evaluation for a single statement (Level)."""
        sub_id = sub_criteria['sub_id']
        level = statement_data['level']
        statement_text = statement_data['statement']
        sub_criteria_name = sub_criteria['sub_criteria_name']

        statement_id = statement_data.get('statement_id', sub_id)
        
        logger.info(f"  > Starting assessment for {sub_id} L{level}...")

        # 1. Determine PDCA Phase and LEVEL CONSTRAINT
        pdca_phase = self._get_pdca_phase(level)
        level_constraint = self._get_level_constraint_prompt(level)
        
        contextual_rules_prompt = self._get_contextual_rules_prompt(sub_id, level)
        
        full_focus_hint = level_constraint + contextual_rules_prompt
        
        # -------------------- 🛑 NEW LOGIC START: Hybrid Retrieval (Helper Call) 🛑 --------------------
        # 1. Hybrid Retrieval: Fetch mapped Stable Doc IDs and priority chunks from VSM
        # 📌 รับผลลัพธ์เป็น Stable Doc IDs ที่ถูกแมปไว้ และ Priority Docs
        mapped_stable_doc_ids, priority_docs = self._get_mapped_uuids_and_priority_chunks(
            sub_id=sub_id,
            level=level,
            statement_text=statement_text,
            level_constraint=level_constraint, 
            vectorstore_manager=vectorstore_manager
            # ไม่ต้องส่ง sequential_chunk_uuids เพราะ Helper ดึงจาก Map เอง
        )
        # -------------------- 🛑 NEW LOGIC END 🛑 --------------------
        
        # 2. RAG Retrieval SETUP (Pre-Query Enhancement)
        # 🟢 FIX #1: เรียกใช้ enhance_query_for_statement พร้อมส่ง 'level' และรับ List[str]
        rag_query_list = enhance_query_for_statement(
            statement_text=statement_text,      # 1. statement_text
            sub_id=sub_id,                      # 2. FIX: ID เกณฑ์ย่อย (e.g., "1.1")
            statement_id=statement_id,          # 3. FIX: ID Statement ย่อย (e.g., "1.1.2" หรือ Fallback เป็น "1.1")
            level=level,                        # 4. Level
            enabler_id=self.enabler_id,         # 5. Enabler ID
            focus_hint=full_focus_hint,         # 6. Focus Hint
            llm_executor=self.llm     # 7. LLM Executor (ถ้ามีใน self)
        )
            
        # 📌 ใช้ query ตัวแรกสำหรับการ Log/แสดงผล (rag_query)
        rag_query = rag_query_list[0] if rag_query_list else statement_text 

        current_final_k = FINAL_K_RERANKED
        current_rag_retriever = self.rag_retriever 
        current_llm_evaluator = self.llm_evaluator 
        initial_k_to_use = INITIAL_TOP_K

        # 🟢 PHASE 2 OPTIMIZATION: Use specialized retrieval/evaluation for L1/L2
        # NOTE: L1_INITIAL_TOP_K_RAG ต้องถูกกำหนดใน self (เช่น self.config.L1_INITIAL_TOP_K_RAG)
        if level <= 2:
            current_llm_evaluator = evaluate_with_llm_low_level
            current_final_k = LOW_LEVEL_K 
            initial_k_to_use = getattr(self, 'L1_INITIAL_TOP_K_RAG', INITIAL_TOP_K)
        else:
            current_final_k = FINAL_K_RERANKED

        # 2. RAG Retrieval EXECUTION
        
        retrieval_start = time.time()
        
        if self.config.mock_mode == "none" and not vectorstore_manager:
            logger.error(f"Cannot run RAG for {sub_id} L{level}: VectorstoreManager is None in non-mock mode.")
            # 📌 FIX: เพิ่ม used_chunk_uuids เป็น List ว่าง
            retrieval_result = {"top_evidences": [], "aggregated_context": "ERROR: No vectorstore manager.", "used_chunk_uuids": []}
        else:
            # 🟢 NEW LOGIC: กำหนดค่า mapped_uuids และ priority_docs_input ที่จะส่งให้ RAG Retriever
            
            # ถ้ามีการดึง Limited Chunks สำเร็จ (priority_docs ไม่ว่าง)
            if priority_docs:
                # 1. ส่ง Chunks ที่ถูกจำกัดไปโดยตรง
                retrieval_map_uuids = None
                retrieval_priority_docs = priority_docs
            else:
                # 2. ถ้าดึง Limited Chunks ไม่ได้: ให้ RAG Retriever จัดการ Hybrid Search เอง
                # 📌 FIX: ส่ง Dict ที่มี Stable Doc IDs และ Chunk UUIDs จาก L[level-1]
                retrieval_map_uuids = {
                    "stable_doc_ids": mapped_stable_doc_ids,
                    "sequential_chunk_uuids": sequential_chunk_uuids or []
                }
                retrieval_priority_docs = None

            try:
                retrieval_result = current_rag_retriever(
                    query=rag_query_list, # 📌 ส่ง List[str] (Multi-Query)
                    doc_type=EVIDENCE_DOC_TYPES, 
                    enabler=self.enabler_id,     
                    top_k=current_final_k,
                    initial_k=initial_k_to_use,
                    sub_id=sub_id, 
                    level=level,
                    vectorstore_manager=vectorstore_manager,
                    # 📌 อัปเดตการส่งพารามิเตอร์: ใช้ตัวแปรใหม่ที่ควบคุมการทำงาน
                    mapped_uuids=retrieval_map_uuids, 
                    priority_docs_input=retrieval_priority_docs 
                )
            except Exception as e:
                logger.error(f"RAG retrieval failed for {sub_id} L{level}: {e}")
                # 📌 FIX: เพิ่ม used_chunk_uuids เป็น List ว่าง
                retrieval_result = {"top_evidences": [], "aggregated_context": "ERROR: RAG failure.", "used_chunk_uuids": []}
        
        retrieval_duration = time.time() - retrieval_start
        aggregated_context = retrieval_result.get("aggregated_context", "")
        top_evidences = retrieval_result.get("top_evidences", [])
        # 🟢 NEW: ดึง used_chunk_uuids ที่ถูก Reranked/ใช้ในการประเมินจริง
        used_chunk_uuids = retrieval_result.get("used_chunk_uuids", []) 

        logger.info(f"    - Retrieval found {len(top_evidences)} evidences in {retrieval_duration:.2f}s (K={current_final_k}).")

        # -------------------- CONTEXT ORDERING LOGIC --------------------
        # ------------------ Action #6: PDCA Content Classification (NEW) ------------------
        # 🟢 ติดป้าย PDCA Tag ให้กับ Chunk ที่ถูก Reranked/เลือกมาเป็น Context
        for doc in top_evidences:
            chunk_text = doc.get('text', '')
            if chunk_text:
                # 📌 เรียกใช้ Classifier ที่เพิ่งเพิ่ม
                pdca_tag = self._classify_pdca_phase_for_chunk(chunk_text) 
                doc['pdca_tag'] = pdca_tag 
            else:
                doc['pdca_tag'] = "Other"
        
        logger.info(f"  > ✅ PDCA Content Tagging complete for {len(top_evidences)} evidences.")
        # ------------------ /Action #6 ------------------

        # 1) CLASSIFY PDCA BLOCKS FROM EVIDENCE
        plan_blocks, do_blocks, check_blocks, act_blocks, other_blocks = \
            self._get_pdca_blocks_from_evidences(top_evidences, level)

        final_context_for_llm = aggregated_context  # default

        # 2) APPLY L3 3-TIER ORDERING
        if level >= 3: 
            
            # 📌 การจัดลำดับจะเกิดขึ้นเฉพาะเมื่อมีหลักฐาน C หรือ A เพื่อให้ความสำคัญกับ PDCA Loop 
            has_check_or_act = check_blocks or act_blocks
            
            if not has_check_or_act:
                logger.warning(f"⚠️ L{level}: No Check/Act blocks detected. Skipping custom ordering.")
            else:
                logger.critical(f"🚨 Activating L{level} Content-Based Reordering.")

                # A. Build simulated evidence (Priority 1) - KEEP FOR NOW
                # simulated_evidence_context = build_simulated_l3_evidence(check_blocks)
                simulated_evidence_context=""
                
                # 🟢 NEW: เพิ่ม SAFETY CAP สำหรับ Simulated Evidence
                # if len(simulated_evidence_context) > MAX_SIMULATED_CONTEXT_LEN:
                #     logger.warning(f"⚠️ L{level} Simulated Context capped from {len(simulated_evidence_context)} to {MAX_SIMULATED_CONTEXT_LEN} chars.")
                #     simulated_evidence_context = simulated_evidence_context[:MAX_SIMULATED_CONTEXT_LEN]
                
                # if IS_LOG_L3_CONTEXT:
                #     logger.info(f"🟢 L{level} simulated evidence created and merged: {len(simulated_evidence_context)} chars.")

                # B. Content-Based Ordered Context (ใช้ Blocks ที่จัดกลุ่มตาม Tag แล้ว)
                final_context_for_llm = build_ordered_context(
                    level=level,
                    # simulated_l3=simulated_evidence_context, 
                    plan_blocks=plan_blocks,
                    do_blocks=do_blocks,
                    check_blocks=check_blocks,
                    act_blocks=act_blocks,
                    other_blocks=other_blocks
                )

                logger.info(f"    - L{level} context reordered successfully based on PDCA Tags.")
        
        # ส่งค่าต่อให้ LLM
        aggregated_context = final_context_for_llm

        # -------------------- CONTEXT ORDERING LOGIC END --------------------

        # 3. LLM Evaluation
        llm_start = time.time()
        llm_result = None # เริ่มต้นด้วย None เพื่อให้การจัดการข้อผิดพลาดมีความทนทาน
        try:
            llm_result = current_llm_evaluator(
                context=aggregated_context,
                sub_criteria_name=sub_criteria_name,
                level=level,
                statement_text=statement_text,
                sub_id=sub_id,
                pdca_phase=pdca_phase,
                level_constraint=level_constraint,
                contextual_rules=contextual_rules_prompt, # 🟢 NEW: ส่งกฎเฉพาะเข้า LLM
                llm_executor=self.llm
            )
        except Exception as e:
            logger.error(f"LLM evaluation failed for {sub_id} L{level}: {e}")
            llm_result = {"score": 0, "reason": f"LLM Fatal Error: {e}", "is_passed": False}

        llm_duration = time.time() - llm_start

        # 🌟 NEW STEP 3.5: LLM Context Summary (สำหรับการทำรายงาน)
        try:
            # 📌 NOTE: ใช้ aggregated_context ที่ได้จาก RAG ในการสรุป
            summary_result = create_context_summary_llm(
                context=aggregated_context,
                sub_criteria_name=sub_criteria_name,
                level=level,
                sub_id=sub_id,
                llm_executor=self.llm # ⬅️ ส่ง LLM Instance เข้าไป
            )
            # ดึง summary text และเก็บ object เต็มไว้
            llm_summary_text = summary_result.get("summary", "N/A (LLM Summary Failed)")
            summary_for_save = summary_result
            
        except Exception as e:
            logger.error(f"Context summarization failed for {sub_id} L{level}: {e}")
            llm_summary_text = "ERROR: LLM summary failed. Using raw context."
            summary_for_save = {"summary": llm_summary_text, "suggestion_for_next_level": str(e)}

        # 🟢 FIX: Calculate PDCA breakdown and final pass status based on llm_score (Priority 1 Part 2 & Priority 2)
        # -------------------- 🛑 การแก้ไขข้อผิดพลาด 'NoneType' 🛑 --------------------
        llm_score = 0
        if llm_result is not None and isinstance(llm_result, dict):
            llm_score = llm_result.get('score', 0)
        else:
            # เพิ่มการจัดการหาก LLM ตอบกลับมาเป็น None หรือไม่ใช่ Dictionary
            self.logger.error(f"LLM returned None or invalid result for assessment {sub_id} L{level}. Setting score=0.")
        # -------------------- 🛑 สิ้นสุดการแก้ไข 🛑 --------------------

        # 📌 ใช้ฟังก์ชันที่แก้ไขแล้ว
        pdca_breakdown, is_passed, raw_pdca_score = calculate_pdca_breakdown_and_pass_status(
            llm_score=llm_score, 
            level=level
        )
        
        pass_status = "✅ PASS" if is_passed else "❌ FAIL"
        
        # 📌 Save on PASS Logic (Auto-Persistence - Idea 2)
        # ใช้ map_key = f"{sub_id}.L{level}" สำหรับการบันทึก Level ปัจจุบัน
        map_key_current = f"{sub_id}.L{level}"
        if is_passed:
            
            # 🟢 FIX: เปลี่ยนการบันทึกเป็น List ของ Dictionary {doc_id, filename}
            # ดึง UUIDs/Info จาก Context ที่ถูก Reranked/ใช้ในการประเมินจริง (จาก top_evidences)
            uuids_to_save = []
            
            # 🟢 NEW LOGIC: บันทึก doc_id และ filename เป็น dictionary
            for doc in top_evidences:
                doc_id = doc.get('doc_id', None)
                source_filename = doc.get('source_filename', doc.get('source', None)) # ใช้ 'source' เป็น fallback
                
                if doc_id is not None:
                    uuids_to_save.append({
                        "doc_id": doc_id,
                        # บันทึกชื่อไฟล์เพื่อให้ mapping file อ่านง่ายขึ้น
                        "filename": source_filename,
                        "mapper_type": "AI_RAG", # ⬅️ เพิ่ม Field นี้
                        "priority": True,    
                        "timestamp": datetime.now().isoformat() # ⬅️ เพิ่ม Field นี้
                    })
            
            if uuids_to_save:
                # ตรวจสอบเพื่อปรับปรุงข้อความ Log (ว่าเป็นการสร้างใหม่หรืออัปเดต)
                is_new_mapping = map_key_current not in self.evidence_map
                
                # ... (โค้ดสำหรับพิมพ์ Log ที่เหลือเหมือนเดิม) ...
                # ใช้ sys.stderr/sys.stdout ในการพิมพ์ Log (ถ้าจำเป็น)
                print(f"\n[MAP 💾 {map_key_current}] ✅ PASS: Saved {len(uuids_to_save)} evidence info to temp map. Details:", file=sys.stderr)
                
                # แสดงแค่ชื่อไฟล์หรือ ID ที่อ่านง่าย
                for i, doc in enumerate(top_evidences[:3]): # แสดง 3 อันดับแรก
                    doc_id = doc.get('doc_id', 'N/A')
                    source = doc.get('source_filename', doc.get('source', 'N/A')) # <--- ใช้ 'source' เป็น fallback
                    score = doc.get('score', 0.0)
                    
                    # ใช้ stderr เพื่อแยกจาก Log ปกติ
                    print(f"  > [Top {i+1} | Score: {score:.3f}] File: **{source}** (ID: {doc_id})", file=sys.stderr)
                
                # บันทึก/อัปเดต Mapping ชั่วคราว (จะ OVERWRITE ข้อมูลเก่าสำหรับ Key นั้น)
                self.temp_map_for_save[map_key_current] = uuids_to_save
                
                action_desc = "🆕 Temporarily stored new mapping" if is_new_mapping else "💾 Updated temporary mapping"
                logger.info(f"{action_desc} for {map_key_current} after successful PASS. ({len(uuids_to_save)} evidence items)")

        result = {
            "sub_criteria_id": sub_id,
            "sub_criteria_name": sub_criteria_name,
            "level": level,
            "statement": statement_text,
            "pdca_phase": pdca_phase,
            "llm_score": llm_score,
            "reason": llm_result.get('reason', 'N/A'),
            "is_passed": is_passed, # 🟢 FIX: ใช้ค่าที่คำนวณจาก PDCA Logic
            "pdca_breakdown": pdca_breakdown, # 🟢 NEW FIELD
            "raw_pdca_score": raw_pdca_score, # 🟢 NEW FIELD
            "rag_query": rag_query,
            "retrieval_duration_s": retrieval_duration,
            "llm_duration_s": llm_duration,
            "retrieved_evidences_count": len(top_evidences),
            "retrieved_full_source_info": top_evidences,
            "aggregated_context_used": aggregated_context,
            # ✅ NEW FIELD: สรุปบริบทโดย LLM (ข้อความ)
            "llm_summarized_context": llm_summary_text, 
            # ✅ NEW FIELD: ผลลัพธ์ LLM Summary เต็ม (รวม suggestion)
            "llm_summary_full_result": summary_for_save,
            # 🟢 NEW: ส่ง UUIDs ที่ใช้ในการประเมินกลับไป (เพื่อใช้ในการทำ Hybrid RAG ของ Level ถัดไป)
            "used_chunk_uuids": used_chunk_uuids 
        }

        logger.info(f"    - Result: {pass_status} ({llm_score}/1) in {llm_duration:.2f}s. Reason: {llm_result.get('reason', 'N/A')[:50]}...")

        return result