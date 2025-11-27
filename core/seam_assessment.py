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
from core.retry_policy import RetryPolicy, RetryResult


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
        _fetch_llm_response,
        build_multichannel_context_for_level
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
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

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

            self.is_sequential = False  

            self.retry_policy = RetryPolicy(
                max_attempts=3,            # ปกติ L3–L5 รีรัน 3 ครั้ง
                base_delay=2.0,            # 2 วินาที
                jitter=True,               # สุ่มหน่วงเวลาเล็กน้อย
                escalate_context=True,     # ขยาย context เมื่อ fail ครั้งที่ 2+
                shorten_prompt_on_fail=True,  # ตัด prompt ให้สั้นเมื่อ fail
                exponential_backoff=True,  # backoff 2s → 4s → 8s
            )

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

    def _collect_previous_level_evidences(self, sub_id: str) -> Dict[int, List[Dict[str, Any]]]:
        """
        Return mapping {level: [evidence_dicts]} for levels 1..(current-1) from self.evidence_map and temp_map_for_save.
        Evidence dict: {"doc_id":..., "source_filename":..., "text": "..."} -- text may be empty (VSM retriever can fetch later)
        """
        levels_map = {}
        # temp_map_for_save keys are like "1.1.L1" => list of dicts with doc_id & filename
        combined_map = {}
        combined_map.update(self.evidence_map or {})
        combined_map.update(self.temp_map_for_save or {})

        for key, items in combined_map.items():
            # expected key format "1.1.L1" or "1.1.L2"
            try:
                parts = key.split(".L")
                if len(parts) != 2:
                    continue
                k_sub = parts[0]
                level_num = int(parts[1])
                if k_sub != sub_id:
                    continue
                evid_list = []
                for it in items:
                    # if stored as dict with doc_id, filename
                    if isinstance(it, dict):
                        evid_list.append({
                            "doc_id": it.get("doc_id"),
                            "source_filename": it.get("filename"),
                            "text": it.get("snippet", "")  # optional
                        })
                    else:
                        # fallback: doc_id string
                        evid_list.append({"doc_id": str(it), "source_filename": None, "text": ""})
                levels_map[level_num] = evid_list
            except Exception:
                continue
        return levels_map


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
    def _save_evidence_map(self):
        """
        Save the temporary evidence map (UUIDs only) to the persistent JSON file.
        Only includes entries that were generated during the current assessment run.
        """
        try:
            if not self.temp_map_for_save:
                logger.info("No evidence to save.")
                return

            existing_map = {}
            if os.path.exists(self.evidence_map_path):
                with open(self.evidence_map_path, "r", encoding="utf-8") as f:
                    existing_map = json.load(f)

            # Merge temp map into existing map
            for key, entries in self.temp_map_for_save.items():
                existing_map[key] = entries

            # Write updated map
            with open(self.evidence_map_path, "w", encoding="utf-8") as f:
                json.dump(existing_map, f, ensure_ascii=False, indent=4)

            logger.info(f"✅ Evidence map saved successfully: {self.evidence_map_path}")
        except Exception as e:
            logger.error(f"Failed to save evidence map: {e}")


    def _load_evidence_map(self):
        """
        Load the persistent evidence map from JSON file.
        Returns a dict mapping {sub_criteria.level: [uuid entries]}.
        """
        if not os.path.exists(self.evidence_map_path):
            logger.info("Evidence map file not found. Starting with empty map.")
            return {}

        try:
            with open(self.evidence_map_path, "r", encoding="utf-8") as f:
                loaded_map = json.load(f)
            logger.info(f"✅ Evidence map loaded: {len(loaded_map)} entries")
            return loaded_map
        except Exception as e:
            logger.error(f"Failed to load evidence map: {e}")
            return {}


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

# --------------------------------------------------------------------------------------
# เมธอด: _get_mapped_uuids_and_priority_chunks (ใน seam_assessment.py)
# --------------------------------------------------------------------------------------
    def _get_mapped_uuids_and_priority_chunks(
        self, 
        sub_id: str, 
        level: int, 
        statement_text: str, 
        level_constraint: str,
        vectorstore_manager: Optional['VectorStoreManager']
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        1. Gathers all PASSED Stable Chunk UUIDs (doc_id) from L1 up to L[level-1]. 
        2. Fetches limited priority RAG chunks (Hybrid Retrieval) 
        based on the gathered Chunk UUIDs.
        
        Returns: (mapped_chunk_uuids: list[str], priority_docs: list[dict])
        """
        
        all_priority_items: List[Dict[str, Any]] = [] 
        
        # 1. วนซ้ำเพื่อดึงหลักฐานที่ PASS จาก Level 1 จนถึง Level ก่อนหน้า (L1 -> L[level - 1])
        for prev_level in range(1, level): 
            prev_map_key = f"{sub_id}.L{prev_level}"
            # 🎯 ตอนนี้ self.evidence_map และ self.temp_map_for_save เก็บ Chunk UUIDs จริง
            all_priority_items.extend(self.evidence_map.get(prev_map_key, []))
            all_priority_items.extend(self.temp_map_for_save.get(prev_map_key, []))
            
        
        # 2. แปลงรายการทั้งหมดให้เป็น Chunk UUID (String) และ Dedup
        doc_ids_for_dedup: List[str] = [
            # 🎯 FIX: ดึง doc_id โดยตรง (ตอนนี้คือ Chunk UUID จริง)
            item.get('doc_id') 
            for item in all_priority_items
            if isinstance(item, dict) and item.get('doc_id')
        ]

        # 🎯 FIX: เปลี่ยนชื่อตัวแปรให้สื่อถึง Chunk UUIDs
        mapped_chunk_uuids: List[str] = list(set(doc_ids_for_dedup))
        num_historical_chunks = len(mapped_chunk_uuids)

        priority_docs = [] 
        
        if num_historical_chunks > 0:
            levels_logged = f"L1-L{level-1}" if level > 1 else "L0 (Should not happen)"
            logger.critical(f"🧭 DEBUG: Priority Search initiated with {num_historical_chunks} historical Chunk UUIDs ({levels_logged}).") 
            logger.info(f"✅ Hybrid Mapping: Found {num_historical_chunks} pre-mapped Chunk UUIDs from {levels_logged} for {sub_id}. Prioritizing these.")
            
            if vectorstore_manager:
                try:
                    # Assuming enhance_query_for_statement is available
                    rag_queries_for_vsm = enhance_query_for_statement(
                        statement_text=statement_text,
                        sub_id=sub_id, 
                        statement_id=sub_id, 
                        level=level, 
                        enabler_id=self.enabler_id,
                        focus_hint=level_constraint 
                    )
                    
                    doc_type = self.doc_type 
                    
                    # 3.1 ดึงเอกสารตาม Chunk UUIDs ที่พบ
                    # retrieve_context_by_doc_ids ได้รับการแก้ไขให้ใช้ UUIDs โดยตรงแล้ว
                    retrieved_docs_result = retrieve_context_by_doc_ids(
                        doc_uuids=mapped_chunk_uuids, # <-- ใช้ Chunk UUIDs
                        doc_type=doc_type,
                        enabler=self.enabler_id,
                        vectorstore_manager=vectorstore_manager
                    )
                    
                    initial_priority_chunks: List[Dict[str, Any]] = retrieved_docs_result.get("top_evidences", [])
                    
                    if initial_priority_chunks:
                        # Rerank เพื่อเลือก Chunk ที่เกี่ยวข้องที่สุด
                        reranker = get_global_reranker(self.FINAL_K_RERANKED) 
                        rerank_query = rag_queries_for_vsm[0] 
                        
                        # สร้าง LcDocument list สำหรับ Rerank
                        lc_docs_for_rerank = [
                            LcDocument(
                                page_content=d.get('content') or d.get('text', ''), 
                                metadata={
                                    **d, 
                                    'relevance_score': 1.0 
                                }
                            ) 
                            for d in initial_priority_chunks
                        ]
                        
                        if reranker and hasattr(reranker, 'compress_documents'):
                            reranked_docs = reranker.compress_documents(
                                query=rerank_query,
                                documents=lc_docs_for_rerank,
                                top_n=self.PRIORITY_CHUNK_LIMIT 
                            )
                            # แปลงกลับเป็น Dict
                            priority_docs = [{
                                **d.metadata, 
                                'content': d.page_content,
                                'text': d.page_content, 
                                'score': d.metadata.get('relevance_score', 1.0) 
                            } for d in reranked_docs]
                        else:
                            priority_docs = initial_priority_chunks[:self.PRIORITY_CHUNK_LIMIT]

                        logger.critical(f"🧭 DEBUG: Limited and prioritized {len(priority_docs)} chunks from {num_historical_chunks} mapped UUIDs.")

                except Exception as e:
                    logger.error(f"Error fetching/reranking priority chunks for {sub_id}: {e}")
                    priority_docs = [] 
        
        # คืนค่า Chunk UUIDs และ Chunks ที่ถูกดึงและจัดลำดับความสำคัญแล้ว
        return mapped_chunk_uuids, priority_docs

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
    def _assess_single_sub_criteria_worker(
        self,
        statement_data: Dict[str, Any],
        llm_executor: Any,
        sub_id: str,
        enabler: str,
        doc_type: str,
        vectorstore_manager: Any,
        mapped_uuids: Optional[List[str]] = None,
        priority_docs_input: Optional[List[Any]] = None,
        contextual_rules_prompt: str = ""
    ) -> Dict[str, Any]:
        level = int(statement_data.get("level", 0))
        statement_text = statement_data.get("statement", "")
        sub_criteria_name = statement_data.get("sub_criteria_name", "")
        pdca_phase = statement_data.get("pdca_phase", "")
        level_constraint = statement_data.get("level_constraint", "")

        # choose retrieval function based on level (you had this)
        if level <= 2:
            retrieval_func = retrieve_context_for_low_levels
            evaluation_func = evaluate_with_llm_low_level
        else:
            retrieval_func = retrieve_context_with_filter
            evaluation_func = evaluate_with_llm

        # retrieval: get top_evidences (list) and aggregated_context (but only used as fallback)
        retrieval_result = retrieval_func(
            query=enhance_query_for_statement(...),  # use your existing call
            doc_type=doc_type,
            enabler=enabler,
            vectorstore_manager=vectorstore_manager,
            top_k=...,
            mapped_uuids=mapped_uuids,
            priority_docs_input=priority_docs_input,
            sub_id=sub_id,
            level=level
        )

        top_evidences = retrieval_result.get("top_evidences", [])
        aggregated_context = retrieval_result.get("aggregated_context", "")

        # Build multichannel context locally (use previous_levels_map from engine)
        previous_levels_map = {}
        try:
            previous_levels_map = self._collect_previous_level_evidences(sub_id)
        except Exception:
            previous_levels_map = {}

        channels = build_multichannel_context_for_level(level, top_evidences, previous_levels_map)

        # Evaluate using direct_context and baseline_summary
        evaluation_result = evaluation_func(
            context=channels.get("direct_context", "") or aggregated_context,
            sub_criteria_name=sub_criteria_name,
            level=level,
            statement_text=statement_text,
            sub_id=sub_id,
            llm_executor=llm_executor,
            pdca_phase=pdca_phase,
            level_constraint=level_constraint,
            contextual_rules_prompt=contextual_rules_prompt,
            baseline_summary=channels.get("baseline_summary", ""),
            aux_summary=channels.get("aux_summary", "")
        )

        # Summarize context for report (use aggregated_context or direct_context)
        summary_result = create_context_summary_llm(
            context=channels.get("direct_context", "") or aggregated_context,
            sub_criteria_name=sub_criteria_name,
            level=level,
            sub_id=sub_id,
            llm_executor=llm_executor
        )

        used_doc_ids = [d.get("doc_id") for d in top_evidences if d.get("doc_id")]

        final = {
            "sub_criteria_id": sub_id,
            "sub_criteria_name": sub_criteria_name,
            "level": level,
            "statement": statement_text,
            "pdca_phase": pdca_phase,
            "llm_result": evaluation_result,
            "used_doc_ids": used_doc_ids,
            "channels_debug": channels.get("debug_meta", {}),
            "summary": summary_result
        }
        return final


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
            
# --- -------------------- Main Execution (FIXED & MODIFIED) -------------------- ---
    def run_assessment(
        self, 
        target_sub_id: str = "all", 
        export: bool = False, 
        vectorstore_manager: Optional['VectorStoreManager'] = None,
        sequential: bool = False  # <-- เพิ่มตรงนี้
    ) -> Dict[str, Any]:
        """
        Main runner for the assessment engine.
        Implements sequential maturity check (L1 -> L2 -> L3...) and multiprocessing.
        """
        start_ts = time.time()
        MAX_L1_ATTEMPTS = 2
        MAX_LEVEL = 5 

        self.is_sequential = sequential

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
        # self.temp_map_for_save = {}
        
        # 🟢 Core Logic Switch for Parallel Execution
        run_parallel = (target_sub_id.lower() == "all" and not self.config.force_sequential)
        
        if run_parallel:
            # ... (Logic Multiprocessing คงเดิม - ต้องตรวจสอบความเข้ากันได้ของ starmap) ...
            logger.info("Starting Parallel Assessment (All Sub-Criteria) with Multiprocessing Pool...")
            
            sub_criteria_data_list = sub_criteria_list 
            engine_config_dict = self.config.__dict__ 
            worker_args = [(sub_data, engine_config_dict) for sub_data in sub_criteria_data_list]
            
            try:
                if sys.platform != "win32":
                    mp_context = multiprocessing.get_context('spawn')
                    pool = mp_context.Pool(processes=max(1, os.cpu_count() - 1))
                else:
                    pool = multiprocessing.Pool(processes=max(1, os.cpu_count() - 1))
                    
                with pool:
                    # NOTE: _assess_single_sub_criteria_worker ที่ให้มาในโจทย์ก่อนหน้า มีพารามิเตอร์ไม่ตรงกับ starmap
                    # โค้ดส่วนนี้จึงถูกคงไว้ตามเดิม แต่ควรตรวจสอบความถูกต้องในการใช้งานจริง
                    results_tuples = pool.starmap(self._assess_single_sub_criteria_worker, worker_args)
                    
            except Exception as e:
                logger.critical(f"Multiprocessing Pool Execution Failed: {e}")
                logger.exception("FATAL: Multiprocessing pool failed to execute worker functions.")
                raise
            

            # --- START PATCH C: normalize parallel worker outputs ---
            for raw_results_for_sub, final_sub_result in results_tuples:
                # Safety: ensure raw_results_for_sub is a list
                if not isinstance(raw_results_for_sub, list):
                    logger.warning("Parallel worker returned non-list raw_results_for_sub; normalizing.")
                    raw_results_for_sub = [raw_results_for_sub] if raw_results_for_sub else []

                # Append raw results to global list
                self.raw_llm_results.extend(raw_results_for_sub)

                # Ensure final_sub_result contains raw_results_ref; if not, attach from worker raw results
                if not final_sub_result.get("raw_results_ref"):
                    final_sub_result["raw_results_ref"] = raw_results_for_sub

                # Compute per-sub summary if missing
                if "sub_summary" not in final_sub_result:
                    num_statements = len(final_sub_result["raw_results_ref"])
                    num_passed = sum(1 for r in final_sub_result["raw_results_ref"] if r.get("is_passed", False))
                    final_sub_result["sub_summary"] = {
                        "num_statements": num_statements,
                        "num_passed": num_passed,
                        "num_failed": num_statements - num_passed,
                        "pass_rate": (num_passed / num_statements) if num_statements else 0.0
                    }

                self.final_subcriteria_results.append(final_sub_result)
            # --- END PATCH C ---

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

                for statement_data in sub_criteria.get('levels', []):
                    level = statement_data.get('level')
                    if level is None or level > self.config.target_level:
                        continue

                    dependency_failed = level > 1 and not is_passed_current_level
                    if dependency_failed:
                        logger.warning(f"  > L{level-1} failed. **Continuing** to assess L{level} for detailed scoring.")

                    previous_level = level - 1
                    sequential_chunk_uuids = passed_chunk_uuids_map.get(previous_level, [])

                
                    # ------------------ CONTEXT SOURCE PREP ------------------
                    # 1) ยืนยันว่ามีตัวแปรจาก retrieval
                    top_evidences = locals().get("top_evidences", [])
                    channels = locals().get("channels", [])

                    # 2) sequential_chunk_uuids พร้อมอยู่แล้วจากขั้นก่อนหน้า
                    #    ถ้ายังไม่มีให้ fallback เป็น []
                    sequential_chunk_uuids = sequential_chunk_uuids if "sequential_chunk_uuids" in locals() else []

                    # ------------------ 🟢 PREP FOR LLM ------------------
                    MAX_CHUNKS = 20  # จำกัดจำนวน evidence chunks
                    MAX_STATEMENT_LEN = 200  # จำกัดความยาว statement สำหรับ retry

                    # 1️⃣ Shorten top_evidences if too many
                    if len(top_evidences) > MAX_CHUNKS:
                        logger.warning(f"  > Truncating top_evidences from {len(top_evidences)} to {MAX_CHUNKS}")
                        top_evidences = top_evidences[:MAX_CHUNKS]

                    # 2️⃣ Shorten sequential_chunk_uuids if too many
                    if len(sequential_chunk_uuids) > MAX_CHUNKS:
                        logger.warning(f"  > Truncating sequential_chunk_uuids from {len(sequential_chunk_uuids)} to {MAX_CHUNKS}")
                        sequential_chunk_uuids = sequential_chunk_uuids[:MAX_CHUNKS]

                    # 3️⃣ Shorten statement text
                    statement_text = statement_data.get("statement", "")
                    if len(statement_text) > MAX_STATEMENT_LEN:
                        logger.warning(f"  > Shortening statement from {len(statement_text)} to {MAX_STATEMENT_LEN} chars")
                        statement_text = statement_text[:MAX_STATEMENT_LEN] + " ..."


                    # 3) Build aggregated_context ให้ RetryPolicy ใช้
                    aggregated_context = {
                        "top_evidences": top_evidences,
                        "channels": channels,
                        "sequential_chunk_uuids": sequential_chunk_uuids
                    }

                    # ------------------ Retry Logic ------------------
                    final_result_for_level = None

                    if level >= 3:
                        # ใช้ RetryPolicy สำหรับ L3-L5
                        final_result_for_level = self.retry_policy.run(
                            fn=lambda attempt: self._run_single_assessment(
                                sub_criteria=sub_criteria,
                                statement_data=statement_data,
                                vectorstore_manager=local_vsm,
                                sequential_chunk_uuids=sequential_chunk_uuids
                            ),
                            level=level,
                            statement=statement_data.get('statement', ''),
                            context_blocks=aggregated_context,  # หรือ channels / top_evidences ตามต้องการ
                            logger=logger
                        )
                    else:
                        # L1-L2 ใช้ retry loop เดิม
                        max_attempts = MAX_L1_ATTEMPTS
                        for attempt in range(max_attempts):
                            if level == 1 and attempt > 0:
                                logger.warning(f"  > 🔄 RETRYING {sub_id} L1 (Attempt {attempt+1}/{MAX_L1_ATTEMPTS})...")

                            result = self._run_single_assessment(
                                sub_criteria=sub_criteria,
                                statement_data=statement_data,
                                vectorstore_manager=local_vsm,
                                sequential_chunk_uuids=sequential_chunk_uuids
                            )

                            if result.get('is_passed', False):
                                final_result_for_level = result
                                break

                            if attempt == max_attempts - 1:
                                final_result_for_level = result
                                break

                    # ----------------- END RETRY LOGIC -----------------
                    
                    # result_to_process = final_result_for_level # Use the final result of the level's attempts

                    if isinstance(final_result_for_level, RetryResult):
                        # แปลงเป็น dict เพื่อใช้งานเหมือนเดิม
                        if final_result_for_level.result is None:
                            result_to_process = {}
                        else:
                            result_to_process = final_result_for_level.result
                    else:
                        result_to_process = final_result_for_level  # dict เดิมสำหรับ L1-L2

                    # ตอนนี้สามารถใช้ setdefault / get ปกติได้
                    result_to_process.setdefault("used_chunk_uuids", result_to_process.get("used_chunk_uuids", []))

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
                            # 🟢 ใช้ 'used_chunk_uuids' ที่ส่งกลับมาจาก _run_single_assessment (List[str] ของ UUIDs)
                            used_evidence = result_to_process.get('supporting_evidence') 
                            if used_evidence:
                                # บันทึก List ของ Evidence Dicts ลงใน Map ท้องถิ่น
                                # Map ท้องถิ่น: passed_chunk_uuids_map
                                passed_chunk_uuids_map[level] = used_evidence 
                                logger.info(f"  > L{level} passed. Saved {len(used_evidence)} supporting evidence items for L{level+1} Sequential Hybrid RAG.")

                    except Exception as e:
                        logger.error(f"Error checking dependency status/processing result for {sub_id} L{level}: {e}")
                        is_passed_current_level = False # Default fail if dependency check errors

                    # 5. Append the PROCESSED result (ENHANCED)
                    # --- START PATCH A ---
                    # Ensure result_to_process has stable keys (avoid later KeyError)
                    result_to_process.setdefault("used_chunk_uuids", result_to_process.get("used_chunk_uuids", []))
                    result_to_process.setdefault("is_passed", result_to_process.get("is_passed", False))
                    result_to_process.setdefault("level", result_to_process.get("level", level))
                    result_to_process.setdefault("pdca_breakdown", result_to_process.get("pdca_breakdown", {}))
                    result_to_process.setdefault("llm_score", result_to_process.get("llm_score", 0))

                    # Add execution index to keep order and differentiate retries
                    exec_index = len(raw_results_for_sub_seq)
                    result_to_process["execution_index"] = exec_index

                    # Append to master lists
                    self.raw_llm_results.append(result_to_process)
                    raw_results_for_sub_seq.append(result_to_process)

                    logger.debug(f"    - Appended result (level={result_to_process['level']}, passed={result_to_process['is_passed']}, exec_idx={exec_index})")
                    # --- END PATCH A ---


                    # ------------------ 🟢 /Action #1 (FIXED) ------------------
                    
                    if is_passed_current_level:
                        highest_full_level = level
                
                
                # -------------------- FINALIZE SUB-CRITERIA (Sequential) --------------------

                target_plan_level = highest_full_level + 1
                action_plan = []

                # 📌 Logic สำหรับการสร้าง Action Plan 
                if target_plan_level <= MAX_LEVEL and highest_full_level < self.config.target_level: 
                    logger.info(f"  > Generating Action Plan: Target L{target_plan_level}...")
                    
                    # ดึง failed statements ของ level ต่อไป
                    # failed_statements_for_plan = [
                    #     r for r in raw_results_for_sub_seq
                    #     if r.get("level") == target_plan_level
                    # ]

                    failed_statements_for_plan = [
                        r for r in raw_results_for_sub_seq
                        if r.get("level") == target_plan_level and not r.get("is_passed", False)
                    ]

                    # ถ้าไม่มี failed statement ให้สร้าง template default
                    if not failed_statements_for_plan:
                        failed_statements_for_plan = [{
                            "statement": "No failed statement, use template recommendation",
                            "level": target_plan_level,
                            "reason": "Auto-generated default for missing failures",
                            "statement_id": f"{sub_id}_L{target_plan_level}"
                        }]

                    try:
                        # NOTE: self.action_plan_generator ต้องถูกกำหนดไว้
                        action_plan = self.action_plan_generator(
                            failed_statements_for_plan, 
                            sub_id=sub_id, 
                            target_level=target_plan_level,
                            llm_executor=self.llm  # ส่ง LLM instance เข้าไป
                        )

                        # ถ้า generator ล้มเหลวหรือ return ว่าง ให้ใช้ fallback template
                        if not action_plan:
                            action_plan = [{
                                "Phase": f"L{target_plan_level}",
                                "Goal": f"Reach Level {target_plan_level} for {sub_id}",
                                "Actions": [{"Statement_ID": "TEMPLATE", "Recommendation": "Review and implement missing practices."}],
                                "P_Plan_Score": 0,
                                "D_Do_Score": 0,
                                "C_Check_Score": 0,
                                "A_Act_Score": 0,
                                "is_passed": False,
                                "score": 0,
                                "reason": "Auto-generated template"
                            }]

                    except Exception as e:
                        logger.error(f"Action Plan Generation failed for {sub_id}: {e}")
                        action_plan = [{
                            "Phase": f"L{target_plan_level}",
                            "Goal": f"Action Plan generation failed for {sub_id}",
                            "Actions": [{"Statement_ID": "LLM_ERROR", "Recommendation": "Manual review required"}],
                            "P_Plan_Score": 0,
                            "D_Do_Score": 0,
                            "C_Check_Score": 0,
                            "A_Act_Score": 0,
                            "is_passed": False,
                            "score": 0,
                            "reason": str(e)
                        }]

                
                # --- START PATCH B: finalize raw_results_for_sub_seq, compute sub-stats ---
                # Sort raw results by level then execution_index for deterministic order
                raw_results_for_sub_seq.sort(key=lambda r: (r.get("level", 0), r.get("execution_index", 0)))

                # Compute basic stats per sub-criteria (useful for export & summary)
                num_statements = len(raw_results_for_sub_seq)
                num_passed = sum(1 for r in raw_results_for_sub_seq if r.get("is_passed", False))
                num_failed = num_statements - num_passed

                # Attach computed summary into self.total_stats or per-sub structure if desired
                # We'll keep per-sub summary inside the final_sub_result (constructed later)
                sub_summary = {
                    "num_statements": num_statements,
                    "num_passed": num_passed,
                    "num_failed": num_failed,
                    "pass_rate": (num_passed / num_statements) if num_statements else 0.0
                }
                # --- END PATCH B ---

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
                    "raw_results_ref": raw_results_for_sub_seq,
                    "sub_summary": sub_summary   # <-- เพิ่มตรงนี้ 
                }
                
                # 📌 NEW FIX: ย้าย Evidence ที่ผ่านแล้วจาก Map ท้องถิ่นไป Map สมาชิก
                # Key Format: "sub_id.L[level]"
                for level, evidence_list in passed_chunk_uuids_map.items():
                    key = f"{sub_id}.L{level}"
                    # evidence_list ตอนนี้คือ List[Dict] ที่มี doc_id และ filename (ถ้าคุณแก้ในลูป Level แล้ว)
                    self.temp_map_for_save[key] = evidence_list
                # --------------------------------------------------------------------------

                logger.critical(f"[END] Sub-Criteria {sub_id} completed. Highest Level: L{highest_full_level} (Score: {weighted_score:.2f})")
                
                # 📌 Auto-Saving temporary evidence map (ถ้ามี Logic ในคลาส)
                if self.temp_map_for_save:
                    logger.info(f"✅ DEBUG: self.temp_map_for_save has {len(self.temp_map_for_save)} entries. Proceeding to save.")
                    logger.info(f"💾 Auto-Saving temporary evidence map after {sub_id} completion...")
                    # NOTE: _save_evidence_map ต้องถูกกำหนดไว้
                    self._save_evidence_map()
                    self.temp_map_for_save.clear()
                else:
                    # ❌ Log นี้จะบอกเราว่า Map ว่างเมื่อมาถึงจุดนี้
                    logger.warning(f"❌ DEBUG: self.temp_map_for_save is EMPTY for {sub_id}. Skipping evidence save.")
                # ------------------------------------------------------------

                self.final_subcriteria_results.append(final_sub_result)

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

    # -------------------- Core Assessment Logic (FINAL ROBUST FIX) --------------------
    def _run_single_assessment(
            self,
            sub_criteria: Dict[str, Any],
            statement_data: Dict[str, Any],
            vectorstore_manager: Optional['VectorStoreManager'],
            sequential_chunk_uuids: Optional[List[str]] = None
        ) -> Dict[str, Any]:
        """Runs RAG retrieval and LLM evaluation for a single statement (Level) and saves evidence (Chunk UUIDs only) if PASS."""

        sub_id = sub_criteria['sub_id']
        level = statement_data['level']
        statement_text = statement_data['statement']
        sub_criteria_name = sub_criteria['sub_criteria_name']
        statement_id = statement_data.get('statement_id', sub_id)

        logger.info(f"  > Starting assessment for {sub_id} L{level}...")

        # -------------------- 1. PDCA & Level Prompt --------------------
        pdca_phase = self._get_pdca_phase(level)
        level_constraint = self._get_level_constraint_prompt(level)
        contextual_rules_prompt = self._get_contextual_rules_prompt(sub_id, level)
        full_focus_hint = level_constraint + contextual_rules_prompt

        # -------------------- 2. Hybrid Retrieval --------------------
        mapped_stable_doc_ids, priority_docs = self._get_mapped_uuids_and_priority_chunks(
            sub_id=sub_id,
            level=level,
            statement_text=statement_text,
            level_constraint=level_constraint,
            vectorstore_manager=vectorstore_manager
        )

        # -------------------- 3. Enhance Query --------------------
        rag_query_list = enhance_query_for_statement(
            statement_text=statement_text,
            sub_id=sub_id,
            statement_id=statement_id,
            level=level,
            enabler_id=self.enabler_id,
            focus_hint=full_focus_hint,
            llm_executor=self.llm
        )
        rag_query = rag_query_list[0] if rag_query_list else statement_text

        # -------------------- 4. LLM Evaluator Setup --------------------
        current_final_k = FINAL_K_RERANKED
        initial_k_to_use = INITIAL_TOP_K
        llm_evaluator_to_use = self.llm_evaluator

        if level <= 2:
            llm_evaluator_to_use = evaluate_with_llm_low_level
            current_final_k = LOW_LEVEL_K
            initial_k_to_use = getattr(self.config, 'L1_INITIAL_TOP_K_RAG', INITIAL_TOP_K)

        # -------------------- 5. RAG Retrieval --------------------
        # ------------------ 🟢 LOG: RAG Queries for current level ------------------
        if rag_query_list:
            # Log จำนวน Query ทั้งหมด
            logger.info(f"  > RAG Query List for {sub_id} L{level} ({len(rag_query_list)} total):")
            
            # Log เนื้อหาของแต่ละ Query (จำกัดความยาวเพื่อไม่ให้ Log ยาวเกินไป)
            for i, q in enumerate(rag_query_list):
                # ใช้ CRITICAL/DEBUG level เพื่อให้ Log ไม่รบกวนการทำงานหลัก (แต่ถ้าต้องการเห็นชัดเจน ใช้ INFO)
                logger.info(f"    - Query {i+1}: \"{q[:150]}...\"") # Log 150 ตัวอักษรแรก
        # -------------------------------------------------------------------------

        retrieval_start = time.time()
        try:
            retrieval_result = self.rag_retriever(
                query=rag_query_list,
                doc_type=EVIDENCE_DOC_TYPES,
                enabler=self.enabler_id,
                top_k=current_final_k,
                initial_k=initial_k_to_use,
                sub_id=sub_id,
                level=level,
                vectorstore_manager=vectorstore_manager,
                mapped_uuids=mapped_stable_doc_ids, 
                priority_docs_input=priority_docs 
            )
        except Exception as e:
            logger.error(f"RAG retrieval failed for {sub_id} L{level}: {e}")
            retrieval_result = {"top_evidences": [], "aggregated_context": "ERROR: RAG failure.", "used_chunk_uuids": []}

        retrieval_duration = time.time() - retrieval_start
        aggregated_context = retrieval_result.get("aggregated_context", "")
        top_evidences = retrieval_result.get("top_evidences", [])
        used_chunk_uuids = retrieval_result.get("used_chunk_uuids", [])
        top_evidences_count = len(top_evidences)

        if top_evidences_count > 0:
            # พยายามดึงชื่อไฟล์เฉพาะที่ไม่ซ้ำกัน
            # 💡 LOG FIX: เพิ่มการกรองค่า 'Unknown' ออกจากชื่อไฟล์ที่ดึงมา
            unique_files = sorted(list(set([
                filename for d in top_evidences 
                if (filename := d.get("source_filename") or d.get("source")) and filename != 'Unknown'
            ])))
            
            logger.info(f"  > RAG Search for {sub_id} L{level}: **{top_evidences_count}** new top evidences selected.")
            if unique_files:
                # จำกัดการแสดงผลชื่อไฟล์ไม่ให้ยาวเกินไป
                file_list_str = ', '.join(unique_files[:5])
                if len(unique_files) > 5:
                    file_list_str += f", ... (and {len(unique_files) - 5} more files)"
                    
                logger.info(f"    - NEW Source Files: [{file_list_str}]")
        elif top_evidences_count == 0:
            logger.info(f"  > RAG Search for {sub_id} L{level}: **0** new top evidences selected (RAG was run).")

        # -------------------- 6. Build Multi-Channel Context (Sequential RAG Fix) --------------------
        try:
            # 🎯 VITAL FIX: Force Evidence Map Reload in Sequential Mode
            if level > 1 and self.is_sequential: 
                logger.info(f"L{level} Sequential: Forcing evidence map reload from file for consistency.")
                # ✅ เรียกฟังก์ชัน _load_evidence_map() ที่ถูกต้อง และอัปเดต self.evidence_mapping
                self.evidence_mapping = self._load_evidence_map() 
                
            previous_levels_map_raw = self._collect_previous_level_evidences(sub_id)
        except Exception as e:
            logger.error(f"Error collecting previous evidences for {sub_id} L{level}: {e}")
            previous_levels_map_raw = {}

        # Flatten {chunk_uuid/doc_id: filename} for fallback
        previous_levels_map = {}
        for level_chunks in previous_levels_map_raw.values():
            for c in level_chunks:
                doc_id = c.get("doc_id") or c.get("chunk_uuid")
                filename = c.get("source_filename") or c.get("source") or c.get("filename") or doc_id
                if doc_id and filename:
                    previous_levels_map[doc_id] = filename

        logger.info(f"L{level} Flattened previous_levels_map ({len(previous_levels_map)} entries)")

        # -------------------- 6a. Sequential Fallback for Level >= 2 (Merge previous top evidences) --------------------
        if level > 1 and self.is_sequential:
            prev_top_evidences = []
            for ev_list in previous_levels_map_raw.values():
                for ev in ev_list:
                    chunk_id = ev.get('doc_id') or ev.get('chunk_uuid')
                    filename = ev.get('filename') or ev.get('source_filename') or ev.get('source')
                    if chunk_id and filename:
                        # สร้าง dict ที่จำลองโครงสร้างของ top_evidences
                        prev_top_evidences.append({
                            "chunk_uuid": chunk_id,
                            "doc_id": chunk_id,
                            "source_filename": filename,
                            "pdca_tag": "Other" # Tag เป็น Other/Priority ได้
                        })
            
            # เติม top_evidences ถ้า RAG ไม่ได้หลักฐานมาเลย (Fallback/Enhance)
            if not top_evidences and prev_top_evidences:
                top_evidences = prev_top_evidences
                logger.warning(f"L{level} Sequential: RAG returned empty. Fallback used {len(top_evidences)} previous evidences.")

        # -------------------- 7. Build Multichannel Context --------------------
        channels = build_multichannel_context_for_level(
            level=level,
            top_evidences=top_evidences,
            previous_levels_map=previous_levels_map,
            max_main_context_tokens=3000,
            max_summary_sentences=3
        )
        
        # -------------------- 7a. Direct / Aux files (Used for logging/debug) --------------------
        direct_files_set: Set[str] = set()
        aux_files_set: Set[str] = set()
        for d in top_evidences:
            chunk_uuid = d.get('chunk_uuid')
            doc_id = d.get('doc_id')
            valid_ids = [cid for cid in [chunk_uuid, doc_id] if cid and not str(cid).startswith('HASH-')]
            filename = d.get('source_filename') or d.get('source') or d.get('filename') or 'Unknown'
            # Check if this evidence is also a priority chunk from previous levels
            map_filename = next((previous_levels_map.get(cid) for cid in valid_ids if cid in previous_levels_map), None)
            is_priority_chunk = map_filename is not None
            
            if is_priority_chunk and map_filename:
                # Use the mapped filename if it's a priority chunk
                filename = map_filename
                direct_files_set.add(filename)
                continue
            
            pdca_tag = d.get('pdca_tag', '').capitalize() or 'Other'
            
            # Existing logic for file tagging based on PDCA
            if filename != 'Unknown':
                if (level >= 3 and pdca_tag in ['C', 'A']) or pdca_tag in ['Plan', 'Do']:
                    direct_files_set.add(filename)
                elif pdca_tag in ['Check', 'Act']:
                    aux_files_set.add(filename)
                elif (pdca_tag in ['Other', ''] and level <= 2):
                    direct_files_set.add(filename)

        direct_files = sorted(list(direct_files_set))
        aux_files = sorted(list(aux_files_set))
        baseline_count = channels['debug_meta']['baseline_count'] if 'debug_meta' in channels else 0

        logger.info(
            f"  > Context channels built: {{'direct_count': {len(direct_files)}, 'aux_count': {len(aux_files)}, "
            f"'baseline_count': {baseline_count}, 'direct_files': {direct_files}, 'aux_files': {aux_files}}}"
        )

        # -------------------- 8. LLM Evaluation (Build Context) --------------------
        context_parts = []
        if channels.get('direct_context'):
            context_parts.append(f"--- DIRECT EVIDENCE (FROM L{level} RAG) ---\n{channels['direct_context']}")
        if channels.get('aux_summary'):
            context_parts.append(f"--- AUXILIARY EVIDENCE ---\n{channels['aux_summary']}")
        if channels.get('baseline_summary'):
            context_parts.append(f"--- BASELINE EVIDENCE (FROM PREVIOUS LEVELS) ---\n{channels['baseline_summary']}")
            
        final_llm_context = "\n\n".join(context_parts)

        llm_start = time.time()
        llm_result = llm_evaluator_to_use(
            context=final_llm_context,
            sub_criteria_name=sub_criteria_name,
            level=level,
            statement_text=statement_text,
            sub_id=sub_id,
            pdca_phase=pdca_phase,
            level_constraint=level_constraint,
            contextual_rules=contextual_rules_prompt,
            llm_executor=self.llm,
            # Placeholder for backward compatibility 
            baseline_summary="",
            aux_summary="",
            check_evidence="",
            act_evidence=""
        )
        llm_duration = time.time() - llm_start

        # -------------------- 9. Context Summary --------------------
        try:
            summary_result = create_context_summary_llm(
                context=aggregated_context,
                sub_criteria_name=sub_criteria_name,
                level=level,
                sub_id=sub_id,
                llm_executor=self.llm
            )
            summary_for_save = summary_result
        except Exception as e:
            logger.error(f"Context summarization failed for {sub_id} L{level}: {e}")
            summary_for_save = {"summary": f"ERROR: {e}"}

        # -------------------- 10. PDCA Breakdown & Pass --------------------
        llm_score = llm_result.get('score', 0) if llm_result else 0
        pdca_breakdown, is_passed, raw_pdca_score = calculate_pdca_breakdown_and_pass_status(
            llm_score=llm_score, 
            level=level
        )
        pass_status = "✅ PASS" if is_passed else "❌ FAIL"

        # -------------------- 11. Save Evidence Mapper (UUID + filename) --------------------
        map_key_current = f"{sub_id}.L{level}"
        if is_passed and top_evidences:
            evidence_for_save = []
            for d in top_evidences:
                chunk_id = d.get('doc_id') or d.get('chunk_uuid')
                actual_filename = d.get('source_filename') or d.get('source') or d.get('filename')
                # Use map to find filename if not found in RAG result (Fall-back to stored filename)
                if (not actual_filename or actual_filename == 'Unknown') and chunk_id:
                    actual_filename = previous_levels_map.get(chunk_id)
                    
                if chunk_id and actual_filename and actual_filename != 'Unknown':
                    evidence_for_save.append({
                        "doc_id": chunk_id,
                        "filename": actual_filename,
                        "mapper_type": "AI_GENERATED",
                        "timestamp": datetime.now().isoformat()
                    })
            if evidence_for_save:
                self.temp_map_for_save[map_key_current] = evidence_for_save
                logger.info(f"  > Saved {len(evidence_for_save)} evidence items (UUID + filename) for {map_key_current} (PASS only).")
        elif is_passed:
            logger.warning(
                f"  > ⚠️ L{level} Passed, but no top_evidences available. Evidence Map not saved."
            )
        # -------------------- 11.5 DEBUG METADATA CHECK --------------------
        logger.critical("--- DEBUG METADATA CHECK FOR L%s (Total Chunks: %s) ---", level, len(top_evidences))
        for i, d in enumerate(top_evidences):
            # Log the metadata fields that are supposed to contain the filename
            logger.critical("Chunk %s -> source_filename: '%s' | source: '%s' | doc_id: '%s'", 
                            i + 1, 
                            d.get("source_filename"), 
                            d.get("source"), 
                            d.get("doc_id") or d.get("chunk_uuid"))
        logger.critical("-------------------------------------------------")

        # -------------------- 12. Return Full Result --------------------
        final_result = {
            "sub_criteria_id": sub_id,
            "statement_id": statement_id,
            "level": level,
            "statement": statement_text,
            "pdca_phase": pdca_phase,
            "llm_score": llm_score,
            "pdca_score_required": raw_pdca_score,
            "pdca_breakdown": pdca_breakdown,
            "is_passed": is_passed,
            "status": "PASS" if is_passed else "FAIL",
            "score": llm_score,
            "llm_result_full": llm_result,
            "context_summary": summary_for_save,
            "retrieval_duration_s": retrieval_duration,
            "llm_duration_s": llm_duration,
            "top_evidences_ref": [
                {"doc_id": d.get("doc_id") or d.get("chunk_uuid"), 
                 "filename": (
                     # 1. พยายามใช้ source_filename หรือ source ถ้าไม่เป็น 'Unknown'
                     (d.get("source_filename") if d.get("source_filename") != 'Unknown' else None) or 
                     (d.get("source") if d.get("source") != 'Unknown' else None) or
                     # 2. Fallback ไปใช้ชื่อไฟล์ที่บันทึกไว้จาก Level ก่อนหน้า
                     previous_levels_map.get(d.get("doc_id") or d.get("chunk_uuid"))
                 )}
                for d in top_evidences
            ],
            "used_chunk_uuids": used_chunk_uuids
        }

        logger.info(f"  > Assessment {sub_id} L{level} completed. Status: {pass_status} (Score: {llm_score:.2f})")
        return final_result