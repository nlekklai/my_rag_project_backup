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
from functools import partial
from core.llm_data_utils import enhance_query_for_statement
import pathlib, uuid
from langchain_core.documents import Document as LcDocument
from core.retry_policy import RetryPolicy, RetryResult
from copy import deepcopy
import tempfile
import shutil
# from json_extractor import _robust_extract_json
from .json_extractor import _robust_extract_json
from filelock import FileLock  # ต้องติดตั้ง: pip install filelock


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


# 📌 แก้ไข Type Hint และ Arguments ของ Tuple ให้รวม LLM parameters (7 elements)
def _static_worker_process(worker_input_tuple: Tuple[Dict[str, Any], str, int, str, str, str, float]) -> Dict[str, Any]:
    """
    Static worker function for multiprocessing pool. 
    It reconstructs SeamAssessment in the new process and executes the assessment 
    for a single sub-criteria.
    
    Args:
        worker_input_tuple: (sub_criteria_data, enabler: str, target_level: int, mock_mode: str, evidence_map_path: str, model_name: str, temperature: float) 

    Returns:
        Dict[str, Any]: Final result of the sub-criteria assessment.
    """
    
    # 🟢 NEW FIX: PATH SETUP สำหรับ Worker Process
    # ทำให้ Worker Process รู้จัก Root Directory ของโปรเจกต์ เพื่อ Import modules ภายในได้ (เช่น models.llm)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.append(project_root)
        
    # NOTE: logger ควรถูกสร้างใหม่ใน Worker process
    worker_logger = logging.getLogger(__name__)

    try:
        # 🟢 FIX: Unpack ค่า Primitives ทั้ง 7 ตัว (รวม LLM parameters)
        sub_criteria_data, enabler, target_level, mock_mode, evidence_map_path, model_name, temperature = worker_input_tuple
    except ValueError as e:
        worker_logger.critical(f"Worker input tuple unpack failed: {e}")
        return {"error": f"Invalid worker input: {e}"}
        
    # 1. Reconstruct Config โดยการสร้างใหม่ด้วยค่า Primitives (The Robust Fix)
    try:
        # 🟢 FIX: สร้าง AssessmentConfig ใหม่ใน Worker Process พร้อมส่ง LLM parameters
        # (AssessmentConfig ต้องมี field model_name และ temperature แล้ว)
        worker_config = AssessmentConfig(
            enabler=enabler,
            target_level=target_level,
            mock_mode=mock_mode,
            model_name=model_name, 
            temperature=temperature
            # force_sequential ไม่จำเป็นใน worker
        )
    except Exception as e:
        worker_logger.critical(f"Failed to reconstruct AssessmentConfig in worker: {e}")
        # Return ผลลัพธ์ที่ผิดพลาด
        return {
            "sub_criteria_id": sub_criteria_data.get('sub_id', 'UNKNOWN'),
            "error": f"Config reconstruction failed: {e}"
        }

    # 2. Re-instantiate SeamAssessment (LLM และ VSM จะถูกสร้างใหม่ใน Worker)
    try:
        # Worker Instance จะเรียก _initialize_llm_if_none() เพื่อสร้าง LLM/VSM ใหม่
        # (ต้องมั่นใจว่า _initialize_llm_if_none ถูกแก้ให้ Import จาก models.llm แล้ว)
        worker_instance = SEAMPDCAEngine(
            config=worker_config, 
            evidence_map_path=evidence_map_path, 
            llm_instance=None, # ให้ Worker สร้างใหม่
            vectorstore_manager=None, # ให้ Worker สร้างใหม่
            logger_instance=worker_logger
        )
    except Exception as e:
        worker_logger.critical(f"FATAL: SEAMPDCAEngine instantiation failed in worker: {e}")
        return {
            "sub_criteria_id": sub_criteria_data.get('sub_id', 'UNKNOWN'),
            "error": f"Engine initialization failed: {e}"
        }
    
    # 3. Execute the worker logic
    # เมธอดนี้จะรัน Logic L1-L5 สำหรับ Sub-Criteria เดี่ยว
    return worker_instance._run_sub_criteria_assessment_worker(sub_criteria_data)


def merge_evidence_mappings(results_list: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    รวม evidence_mapping dictionaries ที่ได้จาก Worker ทุกตัว 
    """
    merged_mapping = defaultdict(list)
    for result in results_list:
        # ตรวจสอบว่าผลลัพธ์จาก Worker มี Key 'evidence_mapping' หรือไม่
        if 'evidence_mapping' in result and isinstance(result['evidence_mapping'], dict):
            # วนลูปผ่าน Key/Value ของ Worker แต่ละตัว
            for level_key, evidence_list in result['evidence_mapping'].items():
                # ใช้ .extend() เพื่อผนวกรายการหลักฐานทั้งหมดอย่างปลอดภัย
                if isinstance(evidence_list, list):
                    merged_mapping[level_key].extend(evidence_list)
    
    # แปลง defaultdict กลับเป็น dict ธรรมดา
    return dict(merged_mapping)

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
    # 🟢 FIX: เพิ่ม LLM Configuration Fields เข้าไปใน Dataclass
    model_name: str = "llama3.1:8b" # ใช้ค่า default ตามที่คุณใช้
    temperature: float = 0.0


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
        # 🟢 FIX #1: เพิ่ม doc_type 
        doc_type: str = EVIDENCE_DOC_TYPES, 
        # 🟢 FIX #2: เพิ่ม vectorstore_manager
        vectorstore_manager: Optional['VectorStoreManager'] = None,
        # 📌 FIX #3 (ใหม่): เพิ่ม evidence_map_path เพื่อรับค่าจาก Worker Process
        evidence_map_path: Optional[str] = None 
    ):

            self.config = config
            self.enabler_id = config.enabler
            self.target_level = config.target_level
            self.rubric = self._load_rubric()
            
            # 🟢 กำหนดค่า VSM และ doc_type
            self.vectorstore_manager = vectorstore_manager
            self.doc_type = doc_type

            self.FINAL_K_RERANKED = FINAL_K_RERANKED
            self.PRIORITY_CHUNK_LIMIT = PRIORITY_CHUNK_LIMIT

            # 🟢 กำหนดค่า LLM และ Logger
            self.llm = llm_instance           
            self.logger = logger_instance if logger_instance is not None else logging.getLogger(__name__)

            # 🟢 Disable Strict Filter
            self.initial_evidence_ids: Set[str] = self._load_initial_evidence_info()
            all_statements = self._flatten_rubric_to_statements()
            initial_count = len(all_statements)

            self.logger.info(f"DEBUG: Statements found: {initial_count}. Strict Filter is **DISABLED**.")

            self.statements_to_assess = all_statements
            self.logger.info(f"DEBUG: Statements selected for assessment: {len(self.statements_to_assess)} (Skipped: {initial_count - len(self.statements_to_assess)})")

            # Assessment results storage
            self.raw_llm_results: List[Dict[str, Any]] = []
            self.final_subcriteria_results: List[Dict[str, Any]] = []
            self.total_stats: Dict[str, Any] = {}

            self.is_sequential = False  

            self.retry_policy = RetryPolicy(
                max_attempts=3,            
                base_delay=2.0,            
                jitter=True,               
                escalate_context=True,     
                shorten_prompt_on_fail=True,  
                exponential_backoff=True,  
            )

            # 📌 Persistent Mapping Configuration
            
            # 1. กำหนด Evidence Map Path
            # ใช้ค่าที่ส่งมาจาก Worker (ถ้ามี) หรือคำนวณค่า Default
            if evidence_map_path:
                self.evidence_map_path = evidence_map_path
            else:
                map_filename = f"{self.enabler_id.lower()}{EVIDENCE_MAPPING_FILENAME_SUFFIX}"
                # 🔹 ใช้ Absolute Path เพื่อป้องกันปัญหา CWD ไม่คงที่
                self.evidence_map_path = os.path.join(PROJECT_ROOT, RUBRIC_CONFIG_DIR, map_filename)

            
            # 2. เตรียม Attribute สำหรับ Persistent Mapping
            self.evidence_map: Dict[str, List[str]] = {}
            self.temp_map_for_save: Dict[str, List[str]] = {}

            self.contextual_rules_map: Dict[str, Dict[str, str]] = self._load_contextual_rules_map()
            
            # 3. โหลดแผนที่ 
            self.evidence_map = self._load_evidence_map()

            self.logger.info(f"Persistent Map Path set to: {self.evidence_map_path}")
            self.logger.info(f"Loaded {len(self.evidence_map)} existing evidence entries into self.evidence_map.")
            
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

            # 📌 โหลด LLM และ VSM หากยังไม่มี
            if self.llm is None: self._initialize_llm_if_none()
            if self.vectorstore_manager is None: self._initialize_vsm_if_none()
            
            self.logger.info(f"Engine initialized for Enabler: {self.enabler_id}, Mock Mode: {config.mock_mode}")

    def _initialize_llm_if_none(self):
        """Initializes LLM instance if self.llm is None."""
        if self.llm is None:
            self.logger.warning("⚠️ Initializing LLM: model=%s, temperature=%s", 
                                self.config.model_name, self.config.temperature)
            try:
                # 🟢 FIX: Import และใช้ create_llm_instance
                from models.llm import create_llm_instance 
                self.llm = create_llm_instance( 
                    model_name=self.config.model_name,
                    temperature=self.config.temperature
                )
                self.logger.info("✅ LLM Instance created successfully: %s (Temp: %s)", 
                                 self.config.model_name, self.config.temperature)
            except Exception as e:
                self.logger.error(f"FATAL: Could not initialize LLM: {e}")
                raise


    def _initialize_vsm_if_none(self):
        """Initializes VectorStoreManager if self.vectorstore_manager is None."""
        if self.vectorstore_manager is None:
            self.logger.info("Loading central evidence vectorstore(s)...")
            try:
                self.vectorstore_manager = load_all_vectorstores(
                    doc_types=[EVIDENCE_DOC_TYPES], 
                    evidence_enabler=self.enabler_id
                )
                
                # 📌 FINAL FIX: เข้าถึง MultiDocRetriever (Private Attribute) 
                # และตามด้วย _all_retrievers (Private Attribute)
                len_retrievers = len(
                    self.vectorstore_manager._multi_doc_retriever._all_retrievers
                )
                
                self.logger.info("✅ MultiDocRetriever loaded with %s collections and cached in VSM.", 
                                 len_retrievers) 
            except Exception as e:
                self.logger.error(f"FATAL: Could not initialize VectorStoreManager: {e}")
                raise
        
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


    def _collect_previous_level_evidences(self, sub_id: str, current_level: int, vectorstore_manager: Optional['VectorStoreManager'] = None) -> Dict[str, List[Dict]]:
        """
        ดึงหลักฐาน (Metadata/DocIDs) ที่ผ่านจาก Level ก่อนหน้าทั้งหมด 
        เพื่อนำมาใช้เป็น Context (Baseline) สำหรับ Level ปัจจุบัน
        
        มีการทำ Hydration (ดึง Text) จาก VSM ในโหมด Parallel (เมื่อเรียกจาก Main Process เพื่อเตรียม Context)
        
        Args:
            sub_id (str): Sub-Criteria ID ปัจจุบัน (เช่น "3.1").
            current_level (int): Level ปัจจุบันที่กำลังประเมิน (เช่น L2).
            vectorstore_manager (Optional['VectorStoreManager']): VSM สำหรับดึงเนื้อหา (Text) 
                                                                 ในโหมด Parallel/Worker (ถ้า Main Process เตรียม Context)
        Returns:
            Dict[str, List[Dict]]: Dictionary ของหลักฐานที่รวบรวมได้, โดย Key คือ "sub_id.L#".
                                   อาจมี Text รวมอยู่ด้วยหากมีการทำ Hydration.
        """
        collected = {}

        # 1. กำหนดแหล่งข้อมูลหลัก
        # ใน Sequential Mode หรือ Parallel Main Process เราใช้ self.evidence_map 
        # (ซึ่งใน Sequential มันจะถูกอัปเดตไปเรื่อยๆ)
        source_map = self.evidence_map
        source_name = "evidence_map (SEQ/PAR Main)"

        # 2. กรองและรวบรวม Metadata (DocID/Filename)
        for key, evidence_list in source_map.items():
            # กรองเฉพาะ Key ที่ขึ้นต้นด้วย sub_id และเป็น Level
            if key.startswith(f"{sub_id}.L") and isinstance(evidence_list, list):
                try:
                    level_num = int(key.split(".L")[-1])
                    
                    # <<< จุดสำคัญ: ตรวจสอบว่า Level ก่อนหน้าหรือไม่
                    if level_num < current_level: 
                        collected[key] = evidence_list 
                except:
                    continue
        
        # 3. HYDRATION (การดึง Text): ในโหมด Parallel (is_sequential=False) 
        #    ถ้ามีการส่ง vectorstore_manager มา (หมายถึง Main Process กำลังเตรียม Context) 
        #    เราจะดึงเนื้อหาเต็มๆ เพื่อส่งไปให้ Worker ใช้เป็น Baseline Context
        
        is_worker_context_preparation = not self.is_sequential and vectorstore_manager is not None

        if is_worker_context_preparation and collected and vectorstore_manager:
            all_uuids = [ev['doc_id'] for ev_list in collected.values() for ev in ev_list if 'doc_id' in ev]
            
            try:
                # ดึง Chunks เต็มๆ จาก VSM
                full_chunks = vectorstore_manager.retrieve_chunks_by_uuids(all_uuids) 
            except Exception as e:
                self.logger.error(f"Failed to retrieve full chunks for baseline in PAR mode: {e}")
                full_chunks = []

            full_chunk_map = {c.get('doc_id') or c.get('chunk_uuid'): c for c in full_chunks}
            
            hydrated_collected = {}
            for key, ev_list in collected.items():
                hydrated_list = []
                for ev_metadata in ev_list:
                    doc_id = ev_metadata.get('doc_id')
                    full_chunk = full_chunk_map.get(doc_id)
                    
                    if full_chunk and full_chunk.get('text'):
                        # รวม Metadata เดิมเข้ากับ Text ที่ดึงมา
                        chunk_to_add = full_chunk.copy()
                        chunk_to_add.update(ev_metadata) 
                        hydrated_list.append(chunk_to_add)
                
                if hydrated_list:
                    hydrated_collected[key] = hydrated_list
            
            collected = hydrated_collected # ใช้ข้อมูลที่มี Text เพื่อเป็น Baseline Context
        
        # 4. Debug Log
        total_files = sum(len(v) for v in collected.values())
        self.logger.info(
            f"BASELINE LOADED → Mode: {'SEQ' if self.is_sequential else 'PAR'} | "
            f"Source: {source_name} | "
            f"Found {len(collected)} levels | "
            f"Keys: {sorted(collected.keys())} | "
            f"Total files: {total_files}"
        )

        return collected

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
    
    # -------------------- Helper Function for Map Processing --------------------
    def _get_level_order_value(self, level_str: str) -> int:
        """Converts Level string ('L1', 'L5') to an integer for comparison."""
        try:
            return int(level_str.upper().replace('L', ''))
        except:
            return 0

    # -------------------- Persistent Mapping Handlers (FIXED) --------------------
    def _process_temp_map_to_final_map(self, temp_map: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Converts the temporary map into the final map format for saving, 
        and filters out temporary/unresolvable evidence IDs.
        """
        working_map = temp_map or self.temp_map_for_save or {}
        final_map_for_save = {}
        total_cleaned_items = 0

        for sub_level_key, evidence_list in working_map.items():
            if isinstance(evidence_list, dict):
                evidence_list = [evidence_list]
            elif not isinstance(evidence_list, list):
                logger.warning(f"[EVIDENCE] Skipping {sub_level_key}: not a list or dict")
                continue

            clean_list = []
            seen_ids = set()
            for ev in evidence_list:
                doc_id = ev.get("doc_id")
                
                if not doc_id:
                    continue
                
                # 1. FIX: กรอง ID ชั่วคราว (TEMP-) ออก
                if doc_id.startswith("TEMP-"):
                    # ID ที่ไม่สามารถแปลงกลับเป็น Stable Document ID ได้ ถือว่าใช้ไม่ได้
                    logger.debug(f"[EVIDENCE] Filtering out unresolvable TEMP- ID: {doc_id} for {sub_level_key}.")
                    continue 
                
                # 2. Logic เดิม: กรอง HASH- (Placeholder) และรายการซ้ำ
                if doc_id.startswith("HASH-") or doc_id in seen_ids:
                    continue
                    
                seen_ids.add(doc_id)
                clean_list.append(ev)
                total_cleaned_items += 1 

            if clean_list:
                final_map_for_save[sub_level_key] = clean_list

        logger.info(f"[EVIDENCE] Processed {len(final_map_for_save)} sub-level keys with total {total_cleaned_items} evidence items")
        return final_map_for_save

    def _clean_map_for_json(self, data: Union[Dict, List, Set, Any]) -> Union[Dict, List, Any]:
        """Recursively converts objects that cannot be serialized (like sets) into lists."""
        if isinstance(data, dict):
            return {k: self._clean_map_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._clean_map_for_json(v) for v in data]
        elif isinstance(data, set):
            return [self._clean_map_for_json(v) for v in data]
        return data

    def _clean_temp_entries(self, evidence_map: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """
        กรอง TEMP-, HASH-, และ Unknown ออกจาก evidence map ทั้งหมด
        ใช้ทั้งตอน merge และก่อน save เพื่อความสะอาด 100%
        """
        if not evidence_map:
            return {}

        cleaned_map = {}
        total_removed = 0
        total_unknown_fixed = 0

        for key, entries in evidence_map.items():
            valid_entries = []
            for entry in entries:
                doc_id = entry.get("doc_id", "")

                # 1. กรอง TEMP- และ HASH- ออกเด็ดขาด
                if str(doc_id).startswith("TEMP-") or str(doc_id).startswith("HASH-"):
                    total_removed += 1
                    continue

                # 2. ถ้า doc_id ว่าง หรือไม่มีเลย → ทิ้ง
                if not doc_id or doc_id == "Unknown":
                    total_removed += 1
                    continue

                # 3. แก้ filename ที่เป็น Unknown / None / ว่าง
                filename = entry.get("filename", "").strip()
                if not filename or filename == "Unknown" or filename.lower() == "unknown_file.pdf":
                    # ใช้ doc_id สั้น ๆ ตั้งชื่อชั่วคราว (ดูดีกว่าว่างเปล่า)
                    short_id = doc_id[:8]
                    entry["filename"] = f"เอกสารอ้างอิง_{short_id}.pdf"
                    total_unknown_fixed += 1
                else:
                    # เอา path ออก ให้เหลือแค่ชื่อไฟล์
                    entry["filename"] = os.path.basename(filename)

                valid_entries.append(entry)

            if valid_entries:
                cleaned_map[key] = valid_entries
            else:
                logger.debug(f"[CLEAN] Key {key} กลายเป็นว่างหลังกรอง → ถูกลบออก")

        logger.info(f"[CLEANUP] ลบ TEMP-/HASH- ออก {total_removed} รายการ | "
                    f"แก้ Unknown filename {total_unknown_fixed} รายการ | "
                    f"เหลือ {len(cleaned_map)} keys")

        return cleaned_map


    def _save_evidence_map(self, map_to_save: Optional[Dict[str, List[Dict[str, Any]]]] = None):
        """
        Saves the evidence map to a persistent JSON file using atomic write + FileLock.
        """
        map_file_path = self.evidence_map_path
        lock_path = map_file_path + ".lock"
        tmp_path = None

        logger.info(f"[EVIDENCE] Evidence map target path: {map_file_path}")

        try:
            # 1. ใช้ FileLock ป้องกันการเขียนพร้อมกัน
            logger.debug(f"[EVIDENCE] Acquiring file lock: {lock_path}")
            with FileLock(lock_path, timeout=60):
                logger.debug("[EVIDENCE] Lock acquired. Proceeding with save...")

                # === เริ่ม FIX LOGIC MERGE & FILTER ===
                if map_to_save is not None:
                    # กรณีมีการส่ง map เข้ามาโดยตรง (ใช้กรณีพิเศษ)
                    final_map_to_write = map_to_save
                    logger.debug("[EVIDENCE] Using passed map_to_save. Skipping deep merge/filter logic.")
                else:
                    # 1. โหลด Map ที่มีอยู่เดิมจาก Disk (ตอนนี้คือ 3.1.L1-L5)
                    existing_map_from_disk = self._load_evidence_map(is_for_merge=True) or {}
                    
                    # 2. Map ที่เพิ่งรันเสร็จในหน่วยความจำ (Worker Process เพิ่งอัปเดต 3.1.L1-L5)
                    map_from_runtime = deepcopy(self.evidence_map)
                    
                    # 3. [FIXED] ผสาน (Merge): เริ่มต้นด้วย Map เก่า และรวม Map ปัจจุบัน เข้าไป
                    # final_map_to_write จะมีข้อมูลทั้งหมด (3.1 + 1.1)
                    final_map_to_write = existing_map_from_disk
                    final_map_to_write.update(map_from_runtime) # 👈 การแก้ไขหลัก: รวม Map ทั้งหมด

                    # 4. [FIXED] กรอง TEMP- ID จาก Map ที่ถูกรวมแล้ว
                    final_map_to_write = self._process_temp_map_to_final_map(final_map_to_write)
                    # หลังจาก merge เสร็จ
                    final_map_to_write = self._clean_temp_entries(final_map_to_write)
                    
                    logger.debug(f"[DEBUG] Final Map keys count: {len(final_map_to_write.keys())}") # 👈 Log ยืนยัน
                # === สิ้นสุด FIX LOGIC MERGE & FILTER ===
                
                if not final_map_to_write:
                    logger.warning("[EVIDENCE] final_map_to_write is empty. Skipping save.")
                    return

                # เตรียม directory
                target_dir = os.path.dirname(map_file_path)
                os.makedirs(target_dir, exist_ok=True)

                # เขียนไฟล์ชั่วคราวก่อน (Atomic Write)
                with tempfile.NamedTemporaryFile(
                    mode='w', delete=False, encoding="utf-8", dir=target_dir
                ) as tmp_file:
                    map_to_write_cleaned = self._clean_map_for_json(final_map_to_write)
                    json.dump(map_to_write_cleaned, tmp_file, indent=4, ensure_ascii=False)
                    tmp_path = tmp_file.name

                # ย้ายไฟล์จริง (atomic)
                shutil.move(tmp_path, map_file_path)
                tmp_path = None

                logger.info(f"[EVIDENCE] Evidence map saved successfully to: {map_file_path}")
                logger.info(f"[DEBUG] Final file size: {os.path.getsize(map_file_path)} bytes")
                
                items_count = sum(len(v) for v in final_map_to_write.values())
                logger.info(f"Persisted final evidence map | Keys: {len(final_map_to_write.keys())} | Items: {items_count} | Size: ~{(os.path.getsize(map_file_path) / 1024):.1f} KB")

        except TimeoutError:
            logger.critical(f"[EVIDENCE] Could not acquire lock within 60s: {lock_path}")
            logger.critical("[EVIDENCE] Another process is holding the lock — possible stuck process!")
            raise
        except Exception as e:
            logger.critical("FATAL FILE WRITE ERROR - CHECK LOG TRACE")
            logger.exception(f"[EVIDENCE] Failed to save map: {e}")
            raise
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                    logger.debug(f"[EVIDENCE] Cleaned up temp file: {tmp_path}")
                except Exception:
                    pass
            logger.debug(f"[EVIDENCE] File lock released: {lock_path}")

    def _load_evidence_map(self, is_for_merge: bool = False):
        """
        Safe load of persistent evidence map. Always returns dict.
        is_for_merge: If True, suppresses "No existing evidence map" INFO log.
        """
        path = self.evidence_map_path

        if not os.path.exists(path):
            if not is_for_merge:
                self.logger.info("[EVIDENCE] No existing evidence map – starting empty.")
            return {}

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not is_for_merge:
                self.logger.info(f"[EVIDENCE] Loaded evidence map: {len(data)} entries")
            return data

        except Exception as e:
            self.logger.error(f"[EVIDENCE] Failed to load evidence map from {path}: {e}")
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
        # กำหนดเฟส PDCA
        pdca_phases_th = ["วางแผน", "ปฏิบัติ", "ตรวจสอบ", "ปรับปรุง"]
        
        # 1. System Prompt ภาษาไทย: บังคับให้ตอบ JSON 100%
        system_prompt = (
            "คุณคือผู้เชี่ยวชาญด้าน PDCA Cycle\n"
            "ภารกิจของคุณคือวิเคราะห์ข้อความหลักฐาน แล้วจัดประเภทว่าเน้นขั้นตอนใดของ PDCA\n"
            f"ต้องเลือกเพียงหนึ่งใน: {', '.join(pdca_phases_th)} หรือ 'อื่นๆ'\n\n"
            "ตอบกลับด้วย **JSON Object ที่ถูกต้องเท่านั้น** รูปแบบ:\n"
            "{\"phase\": \"วางแผน\"}\n"
            "หรือ {\"phase\": \"อื่นๆ\"}\n"
            "ห้ามมีข้อความนอก JSON เด็ดขาด"
        )

        # 2. User Prompt: ให้บริบทชัดเจน + ตัวอย่าง
        user_prompt = (
            f"ข้อความหลักฐาน:\n\"\"\"\n{chunk_text.strip()}\n\"\"\"\n\n"
            "คำนิยามแต่ละเฟส:\n"
            "- วางแผน: นโยบาย, กลยุทธ์, เป้าหมาย, แผนงาน, คณะกรรมการ\n"
            "- ปฏิบัติ: ดำเนินการ, ฝึกอบรม, สื่อสาร, พัฒนาระบบ, ใช้งานจริง\n"
            "- ตรวจสอบ: ติดตามผล, วัดผล, รายงาน, ตรวจสอบภายใน, วิเคราะห์ข้อมูล\n"
            "- ปรับปรุง: แก้ไข, ปรับปรุง, มาตรฐานใหม่, Lesson Learned, ปิดช่องว่าง\n\n"
            "ตอบเฉพาะ JSON:"
        )
        
        try:
            raw_response = _fetch_llm_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,  # สำคัญมาก! ต้องแน่นอน
                max_retries=2,
                llm_executor=self.llm
            )

            if not raw_response:
                return "Other"

            # ดึง JSON ออกมาอย่างแข็งแกร่ง (ใช้ _robust_extract_json ที่คุณมีอยู่แล้ว!)
            parsed = _robust_extract_json(raw_response)
            
            # ถ้า _robust_extract_json ไม่ได้ ให้ fallback ด้วยวิธีเบสิก
            if not parsed or not isinstance(parsed, dict):
                # ลองดึงด้วย regex ง่าย ๆ
                import re
                match = re.search(r'"phase"\s*:\s*"([^"]+)"', raw_response, re.IGNORECASE)
                if match:
                    phase_th = match.group(1).strip()
                else:
                    phase_th = "อื่นๆ"
            else:
                phase_th = parsed.get("phase", parsed.get("classification", "อื่นๆ"))
                phase_th = str(phase_th).strip()

            # แปลงเป็น Literal ที่ต้องการ
            mapping = {
                "วางแผน": "Plan",
                "ปฏิบัติ": "Do",
                "ตรวจสอบ": "Check",
                "ปรับปรุง": "Act",
                "อื่นๆ": "Other",
                "อื่น": "Other",
                "other": "Other"
            }
            result = mapping.get(phase_th, "Other")
            
            self.logger.debug(f"PDCA Classification: '{phase_th}' → {result}")
            return result

        except Exception as e:
            self.logger.error(f"PDCA Classification failed: {e}\nRaw: {raw_response[:200]}")
            return "Other"

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
            1. Gathers all PASSED Stable Chunk UUIDs (doc_id) from L1 up to L[level-1]. 
            2. Fetches limited priority RAG chunks (Hybrid Retrieval) 
            based on the gathered Chunk UUIDs.
            
            Returns: (mapped_chunk_uuids: list[str], priority_docs: list[dict])
            """
            
            all_priority_items: List[Dict[str, Any]] = [] 
            
            # 📌 DEBUG: ตรวจสอบสถานะของ evidence_map ก่อนเริ่มดึง
            logger.info(f"DEBUG: EVIDENCE MAP KEYS BEFORE PRIORITY SEARCH (L{level}): {sorted(self.evidence_map.keys())}")
            
            # 1. วนซ้ำเพื่อดึงหลักฐานที่ PASS จาก Level 1 จนถึง Level ก่อนหน้า (L1 -> L[level - 1])
            for prev_level in range(1, level): 
                prev_map_key = f"{sub_id}.L{prev_level}"
                
                # 🎯 ดึงจาก self.evidence_map (แหล่งข้อมูลหลักที่ถูกอัปเดตใน Sequential)
                items_from_map = self.evidence_map.get(prev_map_key, [])
                all_priority_items.extend(items_from_map)
                
                # [การปรับปรุง]: ลบการดึงจาก self.temp_map_for_save ออกในโหมด Sequential เพื่อความชัดเจน
                # แต่ถ้าเป็นโหมด Parallel, Main Process ต้องรวมผลลัพธ์จาก self.temp_map_for_save 
                # เข้ามาใน all_priority_items ด้วย (ซึ่ง Logic นี้ควรอยู่ใน Main Loop ของ run_assessment)
                # เราจะคงไว้แค่การดึงจาก evidence_map เพื่อให้สอดคล้องกับ Sequential mode ที่กำลังรันอยู่

            
            # 2. แปลงรายการทั้งหมดให้เป็น Chunk UUID (String) และ Dedup
            doc_ids_for_dedup: List[str] = [
                item.get('doc_id') 
                for item in all_priority_items
                if isinstance(item, dict) and item.get('doc_id')
            ]

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
                            
                            # สร้าง LcDocument list สำหรับ Rerank (ต้องนำเข้า LcDocument)
                            from langchain_core.documents import Document as LcDocument

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
        
    # -------------------- Multiprocessing Worker Method --------------------
    def _assess_single_sub_criteria_worker(self, args) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
        """
        Worker function for multiprocessing. 
        ฟังก์ชันนี้ถูกออกแบบมาเพื่อรันใน Worker Process แยกต่างหาก (True Parallel) 
        ดังนั้นจึงต้องรับ Baseline Evidence มาจาก Main Process โดยตรง
        
        Args:
            args: tuple(statement_data, engine_config_dict)
        Returns:
            - raw_results_for_sub: list of final results for each level
            - final_sub_result: summary of sub-criteria evaluation
            - level_evidences: dict of evidences to merge later in main process (เฉพาะ Metadata)
                Format: { "1.1.L1": [ { "doc_id": ..., "filename": ... }, ... ] }
        """
        statement_data, engine_config_dict = args

        # Unpack engine config
        llm_executor = engine_config_dict['llm_executor']
        enabler = engine_config_dict['enabler']
        doc_type = engine_config_dict['doc_type']
        vectorstore_manager = engine_config_dict['vectorstore_manager']
        mapped_uuids = engine_config_dict.get('mapped_uuids')
        priority_docs_input = engine_config_dict.get('priority_docs_input')
        contextual_rules_prompt = engine_config_dict.get('contextual_rules_prompt', "")
        
        # <<< จุดสำคัญ: ดึง Baseline Context ที่ถูก Hydrate (มี Text) จาก Main Process
        # Main Process ต้องดึงหลักฐาน L1-L(n-1) ใส่เข้ามาใน Dict นี้แล้ว
        previous_levels_evidence_full = engine_config_dict.get('previous_levels_evidence_full', []) 
        # >>>

        # Statement metadata
        level = int(statement_data.get("level", 0))
        statement_text = statement_data.get("statement", "")
        sub_criteria_name = statement_data.get("sub_criteria_name", "")
        pdca_phase = statement_data.get("pdca_phase", "")
        level_constraint = statement_data.get("level_constraint", "")
        sub_id = statement_data.get("sub_criteria_id", statement_data.get("sub_id", ""))

        # Determine retrieval/evaluation functions
        if level <= 2:
            retrieval_func = retrieve_context_for_low_levels
            evaluation_func = evaluate_with_llm_low_level
            top_k = 5
        else:
            retrieval_func = retrieve_context_with_filter
            evaluation_func = evaluate_with_llm
            top_k = 10

        # Build enhanced query for RAG
        rag_query_list = enhance_query_for_statement(
            statement_text=statement_text,
            sub_id=sub_id,
            statement_id=statement_data.get('statement_id', sub_id),
            level=level,
            enabler_id=enabler,
            focus_hint=level_constraint,
            llm_executor=llm_executor
        )
        rag_query = rag_query_list[0] if rag_query_list else statement_text

        # Retrieval
        retrieval_result = retrieval_func(
            query=rag_query,
            doc_type=doc_type,
            enabler=enabler,
            vectorstore_manager=vectorstore_manager,
            top_k=top_k,
            mapped_uuids=mapped_uuids,
            priority_docs_input=priority_docs_input,
            sub_id=sub_id,
            level=level
        )

        top_evidences = retrieval_result.get("top_evidences", [])
        aggregated_context = retrieval_result.get("aggregated_context", "")

        # Collect previous level evidences (ถูกลบออกแล้ว)
        # เนื่องจาก worker function นี้ไม่ควรเรียกเมธอดของ Engine class โดยตรง

        # Build multichannel context
        # <<< ใช้ previous_levels_evidence_full ที่รับมาจาก args โดยตรง
        channels = build_multichannel_context_for_level(
            level=level, 
            top_evidences=top_evidences, 
            previous_levels_evidence=previous_levels_evidence_full # <<< ส่งหลักฐานเต็มๆ เข้าไป
        )
        # >>>

        # Evaluate statement
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

        # Summarize context for report
        summary_result = create_context_summary_llm(
            context=channels.get("direct_context", "") or aggregated_context,
            sub_criteria_name=sub_criteria_name,
            level=level,
            sub_id=sub_id,
            llm_executor=llm_executor
        )

        # Prepare evidences to return (for main process to merge)
        # <<< ส่งกลับเฉพาะ Metadata เพื่อลดภาระการส่งผ่าน Process
        level_key = f"{sub_id}.L{level}"
        level_evidences = {
            level_key: [
                {
                    "doc_id": ev.get("doc_id"),
                    "filename": ev.get("filename"),
                    "mapper_type": "AI_GENERATED", 
                    "timestamp": datetime.now().isoformat() # เพิ่ม timestamp เพื่อความสมบูรณ์
                }
                for ev in top_evidences if ev.get("doc_id")
            ]
        }
        # >>>

        # Final result dict
        final_sub_result = {
            "sub_criteria_id": sub_id,
            "sub_criteria_name": sub_criteria_name,
            "level": level,
            "statement": statement_text,
            "pdca_phase": pdca_phase,
            "llm_result": evaluation_result,
            "used_doc_ids": [d.get("doc_id") for d in top_evidences if d.get("doc_id")],
            "channels_debug": channels.get("debug_meta", {}),
            "summary": summary_result
        }

        raw_results_for_sub = [final_sub_result]

        return raw_results_for_sub, final_sub_result, level_evidences

    def _run_sub_criteria_assessment_worker(
        self,
        sub_criteria: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
        """
        รันการประเมิน L1-L5 แบบ sequential สำหรับ sub-criteria หนึ่งตัว
        และส่ง evidence map กลับไปให้ main process รวม
        """
        sub_id = sub_criteria['sub_id']
        sub_criteria_name = sub_criteria['sub_criteria_name']
        sub_weight = sub_criteria.get('weight', 0)

        MAX_L1_ATTEMPTS = 2
        highest_full_level = 0
        is_passed_current_level = True
        raw_results_for_sub_seq: List[Dict[str, Any]] = []

        self.logger.info(f"[WORKER START] Assessing Sub-Criteria: {sub_id} - {sub_criteria_name} (Weight: {sub_weight})")

        # รีเซ็ต temp_map_for_save เฉพาะ worker นี้ (สำคัญมากสำหรับ Parallel!)
        self.temp_map_for_save = {}

        # 1. Loop ผ่านทุก Level (L1 → L5)
        for statement_data in sub_criteria.get('levels', []):
            level = statement_data.get('level')
            if level is None or level > self.config.target_level:
                continue

            # Dependency check: ถ้า level ก่อนหน้า fail → cap ที่นี่
            dependency_failed = level > 1 and not is_passed_current_level
            previous_level = level - 1
            persistence_key = f"{sub_id}.L{previous_level}"
            sequential_chunk_uuids = self.evidence_map.get(persistence_key, [])

            level_result = {}
            level_temp_map: List[Dict[str, Any]] = []

            # --- เรียก _run_single_assessment (รับ 2 ค่า: result, temp_map) ---
            if level >= 3:
                # L3-L5: ใช้ RetryPolicy
                wrapper = self.retry_policy.run(
                    fn=lambda attempt: self._run_single_assessment(
                        sub_criteria=sub_criteria,
                        statement_data=statement_data,
                        vectorstore_manager=self.vectorstore_manager,
                        sequential_chunk_uuids=sequential_chunk_uuids
                    ),
                    level=level,
                    statement=statement_data.get('statement', ''),
                    context_blocks={"sequential_chunk_uuids": sequential_chunk_uuids},
                    logger=self.logger
                )

                # สำคัญ: wrapper.result ตอนนี้เป็น tuple (result, temp_map)
                if isinstance(wrapper, RetryResult) and wrapper.result is not None:
                    level_result = wrapper.result
                    level_temp_map = level_result.get("temp_map_for_level", []) # <-- ดึง List Evidence ออกมา
                else:
                    level_result = {}
                    level_temp_map = []

            else:
                # L1-L2: ลองสูงสุด 2 ครั้ง
                for attempt in range(MAX_L1_ATTEMPTS):
                    level_result = self._run_single_assessment(
                        sub_criteria=sub_criteria,
                        statement_data=statement_data,
                        vectorstore_manager=self.vectorstore_manager,
                        sequential_chunk_uuids=sequential_chunk_uuids
                    )
                    level_temp_map = level_result.get("temp_map_for_level", []) # <-- ดึง List Evidence ออกมา
                    if level_result.get('is_passed', False):
                        break

            # ใช้ result ที่ได้มา
            result_to_process = level_result or {}
            result_to_process.setdefault("used_chunk_uuids", [])

            # ตัดสิน pass/fail สุดท้าย (รวม dependency cap)
            is_passed_llm = result_to_process.get('is_passed', False)
            is_passed_final = is_passed_llm and not dependency_failed

            result_to_process['is_passed'] = is_passed_final
            result_to_process['is_capped'] = is_passed_llm and not is_passed_final
            result_to_process['pdca_score_required'] = get_correct_pdca_required_score(level)

            # บันทึก evidence ลง temp_map_for_save เฉพาะเมื่อ PASS จริง
            if is_passed_final and level_temp_map and isinstance(level_temp_map, list):
                current_key = f"{sub_id}.L{level}"
                self.temp_map_for_save[current_key] = level_temp_map
                self.logger.info(f"[EVIDENCE SAVED] {current_key} → {len(level_temp_map)} chunks")

            # อัปเดตสถานะสำหรับ level ถัดไป
            is_passed_current_level = is_passed_final

            # เพิ่มลง raw results
            result_to_process.setdefault("level", level)
            result_to_process["execution_index"] = len(raw_results_for_sub_seq)
            raw_results_for_sub_seq.append(result_to_process)

            # อัปเดต highest level
            if is_passed_final:
                highest_full_level = level
            else:
                self.logger.info(f"[WORKER STOP] {sub_id} failed at L{level}. Highest achieved: L{highest_full_level}")
                break  # หยุดทันทีเมื่อ fail

        # สรุปผล sub-criteria
        weighted_score = self._calculate_weighted_score(highest_full_level, sub_weight)
        num_passed = sum(1 for r in raw_results_for_sub_seq if r.get("is_passed", False))

        sub_summary = {
            "num_statements": len(raw_results_for_sub_seq),
            "num_passed": num_passed,
            "num_failed": len(raw_results_for_sub_seq) - num_passed,
            "pass_rate": round(num_passed / len(raw_results_for_sub_seq), 4) if raw_results_for_sub_seq else 0.0
        }

        final_sub_result = {
            "sub_criteria_id": sub_id,
            "sub_criteria_name": sub_criteria_name,
            "highest_full_level": highest_full_level,
            "weight": sub_weight,
            "target_level_achieved": highest_full_level >= self.config.target_level,
            "weighted_score": weighted_score,
            "action_plan": [],
            "raw_results_ref": raw_results_for_sub_seq,
            "sub_summary": sub_summary,
        }

        # final_temp_map = self.temp_map_for_save  # ส่งกลับทั้ง dict
        # เป็น
        final_temp_map = {}
        if self.is_sequential:
            # ใน sequential เราใช้ self.evidence_map โดยตรงอยู่แล้ว
            # แต่ส่ง snapshot กลับไปเพื่อความปลอดภัย
            for key in self.evidence_map:
                if key.startswith(sub_criteria['sub_id'] + "."):
                    final_temp_map[key] = self.evidence_map[key]
        else:
            final_temp_map = self.temp_map_for_save.copy()

        self.logger.info(f"[WORKER END] {sub_id} | Highest: L{highest_full_level} | Evidence keys: {len(final_temp_map)}")
        self.logger.debug(f"Evidence keys returned: {list(final_temp_map.keys())}")

        return final_sub_result, final_temp_map

    def run_assessment(
            self,
            target_sub_id: str = "all",
            export: bool = False,
            vectorstore_manager: Optional['VectorStoreManager'] = None,
            sequential: bool = False
        ) -> Dict[str, Any]:
        """
        Main runner ของ Assessment Engine
        รองรับทั้ง Parallel และ Sequential 100%
        และรับประกันว่า evidence_map ครบทุกกรณี
        """

        # if export:
        #     logger.info("EXPORT DETECTED → FORCING SEQUENTIAL MODE...")
        #     sequential = True
        #     run_parallel = False
            
        start_ts = time.time()
        self.is_sequential = sequential

        # ============================== 1. Filter Rubric ==============================
        if target_sub_id.lower() == "all":
            sub_criteria_list = self.rubric
        else:
            sub_criteria_list = [
                s for s in self.rubric if s.get('sub_id') == target_sub_id
            ]
            if not sub_criteria_list:
                logger.error(f"Sub-Criteria ID '{target_sub_id}' not found in rubric.")
                return {"error": f"Sub-Criteria ID '{target_sub_id}' not found."}

        # Reset states
        self.raw_llm_results = []
        self.final_subcriteria_results = []
        # self.evidence_map.clear()  # เคลียร์ทุกครั้งก่อนเริ่มใหม่

        # แทนที่ด้วย:
        if os.path.exists(self.evidence_map_path):
            loaded = self._load_evidence_map()
            if loaded:
                self.evidence_map = loaded
                self.logger.info(f"Resumed from existing evidence map: {len(self.evidence_map)} keys")
            else:
                self.evidence_map = {}
        else:
            self.evidence_map = {}

        if not sequential:
            self.logger.info("[PARALLEL MODE] Starting parallel assessment...")

        # run_parallel = (target_sub_id.lower() == "all" and not self.config.force_sequential)
        run_parallel = (target_sub_id.lower() == "all") and not (sequential or export)

        # ============================== 2. Run Assessment ==============================
        if run_parallel:
            # --------------------- PARALLEL MODE ---------------------
            logger.info("Starting Parallel Assessment with Multiprocessing...")

            worker_args = [(
                sub_data,
                self.config.enabler,
                self.config.target_level,
                self.config.mock_mode,
                self.evidence_map_path,
                self.config.model_name,
                self.config.temperature
            ) for sub_data in sub_criteria_list]

            try:
                pool_ctx = multiprocessing.get_context('spawn')
                with pool_ctx.Pool(processes=max(1, os.cpu_count() - 1)) as pool:
                    results_list = pool.map(_static_worker_process, worker_args)
            except Exception as e:
                logger.critical(f"Multiprocessing failed: {e}")
                raise

            # รวมผลลัพธ์จากทุก worker
            for sub_result, temp_map_from_worker in results_list:
                # รวม Evidence Map
                if isinstance(temp_map_from_worker, dict):
                    for level_key, evidence_list in temp_map_from_worker.items():
                        if isinstance(evidence_list, list) and evidence_list:
                            current_list = self.evidence_map.setdefault(level_key, [])
                            current_list.extend(evidence_list)
                            self.logger.info(f"AGGREGATED: +{len(evidence_list)} → {level_key} "
                                           f"(total: {len(current_list)})")

                # รวมผลลัพธ์อื่น ๆ
                raw_refs = sub_result.get("raw_results_ref", [])
                self.raw_llm_results.extend(raw_refs if isinstance(raw_refs, list) else [])
                self.final_subcriteria_results.append(sub_result)

        else:
            # --------------------- SEQUENTIAL MODE ---------------------
            mode_desc = target_sub_id if target_sub_id != "all" else "All Sub-Criteria (Sequential)"
            self.logger.info(f"Starting Sequential Assessment: {mode_desc}")

            # สำคัญที่สุด: อย่าสร้าง temp_map_for_save เลยใน Sequential
            # และอย่ารับ temp_map_from_worker มาทำอะไรทั้งนั้น!

            local_vsm = vectorstore_manager or (
                load_all_vectorstores(doc_types=[EVIDENCE_DOC_TYPES], evidence_enabler=self.config.enabler)
                if self.config.mock_mode == "none" else None
            )
            self.vectorstore_manager = local_vsm

            for sub_criteria in sub_criteria_list:
                # เรียก worker แต่ไม่สนใจ temp_map_from_worker เพราะ Sequential บันทึกตรงใน evidence_map แล้ว
                sub_result, _ = self._run_sub_criteria_assessment_worker(sub_criteria)

                # รวมผลลัพธ์ปกติ
                self.raw_llm_results.extend(sub_result.get("raw_results_ref", []))
                self.final_subcriteria_results.append(sub_result)

        # ============================== 3. บันทึก Evidence Map ==============================
        if self.evidence_map:
            self._save_evidence_map(map_to_save=self.evidence_map)
            total_items = sum(len(v) for v in self.evidence_map.values())
            self.logger.info(f"Persisted final evidence map | Keys: {len(self.evidence_map)} | "
                            f"Items: {total_items} | Size: ~{total_items * 0.35:.1f} KB")

        # ============================== 4. สรุปผล & Export ==============================
        self._calculate_overall_stats(target_sub_id)

        final_results = {
            "summary": self.total_stats,
            "sub_criteria_results": self.final_subcriteria_results,
            "raw_llm_results": self.raw_llm_results,
            "run_time_seconds": round(time.time() - start_ts, 2),
            "timestamp": datetime.now().isoformat(),
        }

        if export:
            export_path = self._export_results(
                results=final_results,
                enabler=self.config.enabler,
                sub_criteria_id=target_sub_id if target_sub_id != "all" else "ALL",
                target_level=self.config.target_level
            )
            # สำคัญ: รวม evidence_map ทุกกรณี (ทั้ง sequential และ parallel)
            final_results["export_path_used"] = export_path
            final_results["evidence_map"] = deepcopy(self.evidence_map)
            self.logger.info(f"Exported full results → {export_path}")

        return final_results

    def _run_single_assessment(
        self,
        sub_criteria: Dict[str, Any],
        statement_data: Dict[str, Any],
        vectorstore_manager: Optional['VectorStoreManager'],
        sequential_chunk_uuids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        รันการประเมิน Level เดียว (L1-L5) อย่างสมบูรณ์
        - ใช้หลักฐานจาก Level ก่อนหน้าทั้งหมด (baseline)
        - บันทึก evidence map ครบทุก Level
        - รองรับ Sequential & Parallel 100%
        - ไม่มี TEMP-, HASH-, Unknown อีกต่อไป
        """

        sub_id = sub_criteria['sub_id']
        level = statement_data['level']
        statement_text = statement_data['statement']
        sub_criteria_name = sub_criteria['sub_criteria_name']
        statement_id = statement_data.get('statement_id', sub_id)

        logger.info(f"  > Starting assessment for {sub_id} L{level}...")

        # ==================== 1. PDCA & Level Prompt ====================
        pdca_phase = self._get_pdca_phase(level)
        level_constraint = self._get_level_constraint_prompt(level)
        contextual_rules_prompt = self._get_contextual_rules_prompt(sub_id, level)
        full_focus_hint = level_constraint + contextual_rules_prompt

        # ==================== 2. Hybrid Retrieval Setup ====================
        mapped_stable_doc_ids, priority_docs = self._get_mapped_uuids_and_priority_chunks(
            sub_id=sub_id,
            level=level,
            statement_text=statement_text,
            level_constraint=level_constraint,
            vectorstore_manager=vectorstore_manager
        )

        # ==================== 3. Enhance Query ====================
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

        # ==================== 4. LLM Evaluator Setup ====================
        llm_evaluator_to_use = self.llm_evaluator
        if level <= 2:
            llm_evaluator_to_use = evaluate_with_llm_low_level

        # ==================== 5. RAG Retrieval ====================
        retrieval_start = time.time()
        try:
            retrieval_result = self.rag_retriever(
                query=rag_query_list,
                doc_type=EVIDENCE_DOC_TYPES,
                enabler=self.enabler_id,
                sub_id=sub_id,
                level=level,
                vectorstore_manager=vectorstore_manager,
                mapped_uuids=mapped_stable_doc_ids,
                priority_docs_input=priority_docs,
                sequential_chunk_uuids=sequential_chunk_uuids
            )
        except Exception as e:
            logger.error(f"RAG retrieval failed for {sub_id} L{level}: {e}")
            retrieval_result = {"top_evidences": [], "aggregated_context": "ERROR: RAG failure.", "used_chunk_uuids": []}

        retrieval_duration = time.time() - retrieval_start
        top_evidences = retrieval_result.get("top_evidences", [])
        used_chunk_uuids = retrieval_result.get("used_chunk_uuids", [])

        # ==================== 6. ดึงหลักฐานจาก Level ก่อนหน้า ====================
        try:
            previous_levels_raw = self._collect_previous_level_evidences(
                sub_id, current_level=level, vectorstore_manager=vectorstore_manager
            )
        except Exception as e:
            logger.error(f"Failed to collect previous evidences: {e}")
            previous_levels_raw = {}

        previous_levels_evidence_full = []
        previous_levels_filename_map = {}

        for ev_list in previous_levels_raw.values():
            for ev in ev_list:
                doc_id = ev.get("doc_id") or ev.get("chunk_uuid")
                if not doc_id or str(doc_id).startswith("HASH-"):
                    continue
                previous_levels_evidence_full.append(ev)
                filename = ev.get("source_filename") or ev.get("source") or ev.get("filename") or "UNKNOWN"
                previous_levels_filename_map[doc_id] = filename

        # ==================== 6a. Sequential fallback ====================
        if level > 1 and self.is_sequential:
            current_ids = {d.get("doc_id") or d.get("chunk_uuid") for d in top_evidences}
            for ev in previous_levels_evidence_full:
                ev_id = ev.get("doc_id") or ev.get("chunk_uuid")
                if ev_id not in current_ids:
                    fallback_ev = ev.copy()
                    fallback_ev["pdca_tag"] = "Baseline"
                    top_evidences.append(fallback_ev)

        # ==================== 7. สร้าง Multi-Channel Context ====================
        channels = build_multichannel_context_for_level(
            level=level,
            top_evidences=top_evidences,
            previous_levels_evidence=previous_levels_evidence_full,
            max_main_context_tokens=3000,
            max_summary_sentences=4
        )

        debug = channels.get("debug_meta", {})
        logger.info(
            f"  > Context built → Direct: {debug.get('direct_count',0)}, "
            f"Aux: {debug.get('aux_count',0)}, "
            f"Baseline: {len(previous_levels_evidence_full)} files "
            f"from {len(previous_levels_raw)} previous levels"
        )

        # ==================== 8. LLM Evaluation ====================
        context_parts = [
            f"--- DIRECT EVIDENCE (L{level}) ---\n{channels.get('direct_context','')}",
            f"--- AUXILIARY EVIDENCE ---\n{channels.get('aux_summary','')}",
            f"--- BASELINE FROM PREVIOUS LEVELS ---\n{channels.get('baseline_summary','ไม่มี')}"
        ]
        final_llm_context = "\n\n".join([p for p in context_parts if p.strip()])

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
            llm_executor=self.llm
        )
        llm_duration = time.time() - llm_start

        # ==================== 9-10. Scoring & Pass/Fail ====================
        llm_score = llm_result.get('score', 0) if llm_result else 0
        pdca_breakdown, is_passed, _ = calculate_pdca_breakdown_and_pass_status(llm_score, level)
        status = "PASS" if is_passed else "FAIL"

        # ==================== 11. บันทึก Evidence Map (เฉพาะ PASS) ====================
        temp_map_for_level = None
        evidence_entries = []  # ใช้ตัวนี้ทั้งบันทึกและแสดงผล
        logger.critical(f"🧭 DEBUG: Entering Evidence Save Logic for {sub_id}.L{level}. Passed: {is_passed}, Top Evidences: {len(top_evidences)}")

        if is_passed and top_evidences:
            seen = set()
            discarded_ids = []

            for ev in top_evidences:
                doc_id = ev.get("doc_id") or ev.get("chunk_uuid")
                if not doc_id or str(doc_id).startswith("TEMP-") or str(doc_id).startswith("HASH-"):
                    discarded_ids.append(f"Skipped: {doc_id}")
                    continue
                if doc_id in seen:
                    continue
                seen.add(doc_id)

                filename = (
                    ev.get("source_filename") or
                    ev.get("source") or
                    ev.get("filename") or
                    previous_levels_filename_map.get(doc_id) or
                    f"เอกสารอ้างอิง_{doc_id[:8]}.pdf"
                )

                entry = {
                    "doc_id": doc_id,
                    "filename": os.path.basename(filename),
                    "mapper_type": "AI_GENERATED",
                    "timestamp": datetime.now().isoformat()
                }
                evidence_entries.append(entry)

            logger.critical(f"🧭 DEBUG: Discarded {len(discarded_ids)} invalid entries. "
                            f"Valid entries: {len(evidence_entries)}")

            if evidence_entries:
                key = f"{sub_id}.L{level}"
                if self.is_sequential:
                    current_list = self.evidence_map.setdefault(key, [])
                    existing_ids = {item["doc_id"] for item in current_list}
                    new_entries = [e for e in evidence_entries if e["doc_id"] not in existing_ids]
                    current_list.extend(new_entries)
                    logger.info(f"DIRECT SAVE evidence_map[{key}] +{len(new_entries)} files → total {len(current_list)}")
                else:
                    if not hasattr(self, "temp_map_for_save"):
                        self.temp_map_for_save = {}
                    self.temp_map_for_save[key] = evidence_entries

                logger.info(f"  > [EVIDENCE SAVED] {key} → {len(evidence_entries)} files")
                temp_map_for_level = evidence_entries if not self.is_sequential else None

        # ==================== 12. Evidence Strength & Confidence ====================
        direct_count = len([d for d in top_evidences if d.get("pdca_tag") in ["P", "D", "C", "A"]])
        total_chunks = len(top_evidences)
        pdca_coverage = len({d.get("pdca_tag") for d in top_evidences if d.get("pdca_tag")})

        evidence_strength = min(10.0, 
            (direct_count * 1.8) + 
            (2.0 if total_chunks >= 20 else 1.0 if total_chunks >= 10 else 0.0) +
            (pdca_coverage * 1.5)
        )

        ai_confidence = "HIGH" if evidence_strength >= 8.0 and is_passed else \
                        "MEDIUM" if evidence_strength >= 5.5 else "LOW"

        evidence_count_for_level = len(evidence_entries)

        # ==================== 13. สร้างผลลัพธ์สุดท้าย ====================
        final_result = {
            "sub_criteria_id": sub_id,
            "statement_id": statement_id,
            "level": level,
            "statement": statement_text,
            "pdca_phase": pdca_phase,
            "llm_score": llm_score,
            "pdca_breakdown": pdca_breakdown,
            "is_passed": is_passed,
            "status": status,
            "score": llm_score,
            "llm_result_full": llm_result,
            "retrieval_duration_s": round(retrieval_duration, 2),
            "llm_duration_s": round(llm_duration, 2),
            "top_evidences_ref": evidence_entries,  # ใช้ตัวเดียวกัน → ตรงกัน 100%
            "temp_map_for_level": temp_map_for_level,
            "evidence_strength": round(evidence_strength, 1),
            "ai_confidence": ai_confidence,
            "evidence_count": evidence_count_for_level,
            "pdca_coverage": pdca_coverage,
            "direct_evidence_count": direct_count
        }

        logger.info(f"  > Assessment {sub_id} L{level} completed → {status} (Score: {llm_score:.1f})")
        return final_result