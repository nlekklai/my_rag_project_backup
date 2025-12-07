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
import re


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
        PRIORITY_CHUNK_LIMIT,
        DEFAULT_TENANT,
        DEFAULT_YEAR,
        RERANK_THRESHOLD,
        MAX_EVI_STR_CAP
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
    RUBRIC_FILENAME_PATTERN = "{tenant}_{enabler}_rubric.json"
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


# 📌 แก้ไข Type Hint และ Arguments ของ Tuple ให้รวม document_map (8 elements)
def _static_worker_process(worker_input_tuple: Tuple[Dict[str, Any], str, int, str, str, str, float, Optional[Dict[str, str]]]) -> Dict[str, Any]:
    """
    Static worker function for multiprocessing pool. 
    It reconstructs SeamAssessment in the new process and executes the assessment 
    for a single sub-criteria.
    
    Args:
        worker_input_tuple: (sub_criteria_data, enabler: str, target_level: int, mock_mode: str, 
                             evidence_map_path: str, model_name: str, temperature: float, 
                             document_map: Optional[Dict[str, str]]) 

    Returns:
        Dict[str, Any]: Final result of the sub-criteria assessment.
    """
    
    # 🟢 NEW FIX: PATH SETUP สำหรับ Worker Process
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.append(project_root)
        
    worker_logger = logging.getLogger(__name__)

    try:
        # 🟢 FIX: Unpack ค่า Primitives ทั้ง 8 ตัว (รวม document_map)
        sub_criteria_data, enabler, target_level, mock_mode, evidence_map_path, model_name, temperature, document_map = worker_input_tuple
    except ValueError as e:
        worker_logger.critical(f"Worker input tuple unpack failed (expected 8 elements): {e}")
        return {"error": f"Invalid worker input: {e}"}
        
    # 1. Reconstruct Config 
    try:
        # 🟢 FIX: สร้าง AssessmentConfig ใหม่ใน Worker Process
        worker_config = AssessmentConfig(
            enabler=enabler,
            target_level=target_level,
            mock_mode=mock_mode,
            model_name=model_name, 
            temperature=temperature
        )
    except Exception as e:
        worker_logger.critical(f"Failed to reconstruct AssessmentConfig in worker: {e}")
        return {
            "sub_criteria_id": sub_criteria_data.get('sub_id', 'UNKNOWN'),
            "error": f"Config reconstruction failed: {e}"
        }

    # 2. Re-instantiate SeamAssessment 
    try:
        # 🟢 FIX (สำคัญ): ส่ง document_map เข้าไปใน SEAMPDCAEngine
        worker_instance = SEAMPDCAEngine(
            config=worker_config, 
            evidence_map_path=evidence_map_path, 
            llm_instance=None, 
            vectorstore_manager=None, 
            logger_instance=worker_logger,
            document_map=document_map # ⬅️ ส่ง document_map ที่เพิ่ง Unpack เข้ามา
        )
    except Exception as e:
        worker_logger.critical(f"FATAL: SEAMPDCAEngine instantiation failed in worker: {e}")
        return {
            "sub_criteria_id": sub_criteria_data.get('sub_id', 'UNKNOWN'),
            "error": f"Engine initialization failed: {e}"
        }
    
    # 3. Execute the worker logic
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
    # 🟢 FIX: เพิ่ม Tenant และ Year
    tenant: str = DEFAULT_TENANT
    year: int = DEFAULT_YEAR


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
        config: AssessmentConfig,
        llm_instance: Any = None, 
        logger_instance: logging.Logger = None,
        rag_retriever_instance: Any = None,
        # 🟢 FIX #1: เพิ่ม doc_type 
        doc_type: str = EVIDENCE_DOC_TYPES, 
        # 🟢 FIX #2: เพิ่ม vectorstore_manager
        vectorstore_manager: Optional['VectorStoreManager'] = None,
        # 📌 FIX #3 (ใหม่): เพิ่ม evidence_map_path เพื่อรับค่าจาก Worker Process
        evidence_map_path: Optional[str] = None,
        document_map: Optional[Dict[str, str]] = None 
    ):

            # =======================================================
            # 🎯 FIX 1: ย้ายและแก้ไขการกำหนดค่า Logger เป็นอันดับแรก
            # =======================================================
            if logger_instance is not None:
                 self.logger = logger_instance
            else:
                 # สร้าง Child Logger เพื่อให้ Log มี Context ของ Tenant/Year
                 self.logger = logging.getLogger(__name__).getChild(f"Engine|{config.enabler}|{config.tenant}/{config.year}")
            
            self.logger.info(f"Initializing SEAMPDCAEngine for {config.enabler} ({config.tenant}/{config.year}).")

            # =======================================================
            # 🟢 เริ่มกำหนด Attribute และเรียกเมธอดที่ใช้ self.logger
            # =======================================================
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
            # 🎯 FIX 2: ปรับปรุงการกำหนด logger
            if logger_instance is None:
                # ใช้ logger ที่กำหนดไว้ตั้งแต่ต้น
                self.logger.warning("Re-setting logger instance using the pre-initialized one.")
            
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

            self.RERANK_THRESHOLD: Final[float] = RERANK_THRESHOLD
            self.MAX_EVI_STR_CAP: Final[float] = MAX_EVI_STR_CAP
            # 📌 Persistent Mapping Configuration
            
            # 1. กำหนด Evidence Map Path
            # ใช้ค่าที่ส่งมาจาก Worker (ถ้ามี) หรือคำนวณค่า Default
            if evidence_map_path:
                self.evidence_map_path = evidence_map_path
            else:
                base_dir = os.path.join(
                    PROJECT_ROOT, 
                    RUBRIC_CONFIG_DIR, # <-- ตอนนี้คือ config/gov_tenants
                    self.config.tenant, 
                    str(self.config.year)
                )

                # 🟢 FIX #2: สร้าง Filename: tenant_year_enabler_evidence_mapping.json
                map_filename = (
                    f"{self.config.tenant}_{self.config.year}_{self.enabler_id.lower()}"
                    f"{EVIDENCE_MAPPING_FILENAME_SUFFIX}"
                )

                # 🟢 FIX #3: รวม Path
                self.evidence_map_path = os.path.join(base_dir, map_filename)

            
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
                self.logger.info("Enabling global LLM data utils mock control mode.")
                set_llm_data_mock_mode(True)
            elif config.mock_mode == "random":
                self.logger.warning("Mock mode 'random' is not fully implemented. Using 'control' logic if available.")
                if hasattr(sys.modules.get('seam_mocking'), 'set_mock_control_mode'):
                    sys.modules.get('seam_mocking').set_mock_control_mode(False)
                    set_llm_data_mock_mode(False)

            # 📌 โหลด LLM และ VSM หากยังไม่มี
            if self.llm is None: self._initialize_llm_if_none()
            if self.vectorstore_manager is None: self._initialize_vsm_if_none()
            
            # =======================================================
            # 🎯 FIX #4: โหลด Document Map หากไม่ได้ส่งเข้ามา หรือเป็น Dictionary ว่าง
            # (เพื่อแก้ปัญหา Filename Resolution Failed)
            # =======================================================
            map_to_use: Dict[str, str] = document_map if document_map is not None else {}

            if not map_to_use:
                # คำนวณ Path ของไฟล์ Doc ID Mapping ที่ VSM ใช้สำเร็จ
                mapping_path = os.path.join(
                    PROJECT_ROOT, 
                    RUBRIC_CONFIG_DIR, 
                    self.config.tenant, 
                    str(self.config.year), 
                    # 🟢 FIX: ต้องใช้ _doc_id_mapping.json เท่านั้น (ใช้ชื่อไฟล์ตรงๆ)
                    f"{self.config.tenant}_{self.config.year}_{self.enabler_id.lower()}_doc_id_mapping.json"
                )
                
                self.logger.info(f"Attempting to load document_map from file: {mapping_path}")

                try:
                    with open(mapping_path, 'r', encoding='utf-8') as f:
                        doc_map_raw = json.load(f)
                    
                    # แปลงโครงสร้าง Doc ID -> {file_name: X, ...} เป็น Doc ID -> file_name
                    map_to_use = {
                        doc_id: data.get("file_name", doc_id) # ใช้ doc_id เป็น fallback
                        for doc_id, data in doc_map_raw.items()
                    }
                    self.logger.info(f"Successfully loaded {len(map_to_use)} document mappings from file.")
                    
                except FileNotFoundError:
                    self.logger.warning(f"Document mapping file not found at: {mapping_path}. Using empty map.")
                except Exception as e:
                    self.logger.error(f"Error loading document map from file: {e}")

            # จัดเก็บ Document Map
            self.doc_id_to_filename_map: Dict[str, str] = map_to_use
            self.document_map: Dict[str, str] = self.doc_id_to_filename_map # เพื่อความเข้ากันได้
            
            self.logger.info(f"Loaded {len(self.doc_id_to_filename_map)} document mappings.")
            if not self.doc_id_to_filename_map:
                self.logger.warning("Document ID to Filename Map is empty. Filename resolution might be limited.")

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
        """
        Initializes VectorStoreManager if self.vectorstore_manager is None.
        Handles multi-tenant/multi-year vector store loading.
        """
        if self.vectorstore_manager is None:
            self.logger.info("Loading central evidence vectorstore(s)...")
            try:
                # 🎯 FIX: ส่ง tenant และ year จาก config เข้าไปใน load_all_vectorstores()
                self.vectorstore_manager = load_all_vectorstores(
                    doc_types=[EVIDENCE_DOC_TYPES], 
                    evidence_enabler=self.enabler_id,
                    tenant=self.config.tenant,  # <--- NEW: เพิ่ม Argument นี้
                    year=self.config.year       # <--- NEW: เพิ่ม Argument นี้
                )
                
                # 📌 FINAL FIX: เข้าถึง MultiDocRetriever (Private Attribute) 
                # และตามด้วย _all_retrievers (Private Attribute)
                # ตรวจสอบว่า VSM ถูกสร้างสำเร็จก่อน
                if self.vectorstore_manager and hasattr(self.vectorstore_manager, '_multi_doc_retriever'):
                     len_retrievers = len(
                        self.vectorstore_manager._multi_doc_retriever._all_retrievers
                    )
                     self.logger.info("✅ MultiDocRetriever loaded with %s collections and cached in VSM.", 
                                 len_retrievers) 
                else:
                    self.logger.warning("VectorStoreManager loaded but _multi_doc_retriever structure is missing or unexpected.")

            except Exception as e:
                self.logger.error(f"FATAL: Could not initialize VectorStoreManager: {e}")
                raise # Re-raise the exception to หยุดโปรแกรม
        
    def _resolve_evidence_filenames(self, evidence_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """
            ฟังก์ชันสำหรับแก้ไขชื่อไฟล์ในรายการหลักฐานอ้างอิง
            1. จัดการหลักฐานที่ doc_id ขึ้นต้นด้วย 'UNKNOWN-' (หลักฐานภายใน/ที่ไม่ใช่เอกสาร)
            2. แปลง doc_id (ที่เป็น Hash/UUID) ให้เป็นชื่อไฟล์ที่มนุษย์อ่านได้ โดยใช้ doc_id_to_filename_map
            """
            # ต้องมั่นใจว่า re และ deepcopy ถูก import ไว้ที่ส่วนหัวของไฟล์แล้ว
            from copy import deepcopy
            import re
            
            resolved_entries = []
            for entry in evidence_entries:
                # ใช้ deepcopy เพื่อป้องกันการแก้ไขข้อมูลต้นฉบับ
                resolved_entry = deepcopy(entry)
                # ใช้ doc_id เป็นค่าหลักในการค้นหาชื่อไฟล์
                doc_id = resolved_entry.get("doc_id", "")
                # ตรวจสอบชื่อไฟล์ปัจจุบัน (จาก metadata ของ vectorstore ถ้ามี)
                current_filename = resolved_entry.get("filename", "")
                
                # --- 1. จัดการกรณี UNKNOWN- (AI-GENERATED or Lost Source) ---
                if doc_id.startswith("UNKNOWN-"):
                    # ให้ใช้ชื่อไฟล์ที่สื่อสารชัดเจนว่าไม่ใช่ไฟล์เอกสารจริง
                    # เช่น "UNKNOWN-2fac2f11" --> "AI-GENERATED-REF-2fac2f11"
                    resolved_entry["filename"] = f"AI-GENERATED-REF-{doc_id.split('-')[-1]}"
                    resolved_entries.append(resolved_entry)
                    continue

                # --- 2. จัดการกรณี Doc ID (Hash/UUID) ที่ถูกต้อง ---
                if doc_id:
                    # A. ลองค้นหาชื่อไฟล์จาก Map
                    if doc_id in self.doc_id_to_filename_map:
                        resolved_entry["filename"] = self.doc_id_to_filename_map[doc_id]
                        # ชื่อถูกต้องแล้ว
                        resolved_entries.append(resolved_entry)
                        continue

                    # B. ถ้าค้นหาไม่เจอ (Map Fail)
                    else:
                        # ตรวจสอบว่าชื่อไฟล์ปัจจุบันเป็นชื่อที่ไม่เหมาะสม (เช่น "Unknown" หรือ Hash.pdf)
                        is_generic_name = (
                            current_filename.lower() == "unknown" or
                            # ✅ รองรับ Hash/UUID 64 ตัวอักษรอย่างเดียว หรือตามด้วยนามสกุล
                            re.match(r"^[0-9a-f]{64}(\.pdf|\.txt)?$", current_filename, re.IGNORECASE)
                        )
                        
                        if is_generic_name:
                            # ใช้ชื่อไฟล์ Fallback ที่สื่อว่า Map ไม่สำเร็จ แต่มี ID
                            resolved_entry["filename"] = f"MAPPING-FAILED-{doc_id[:8]}..."
                            self.logger.warning(f"Failed to map doc_id {doc_id[:8]}... to filename. Using fallback.")
                        # else: หากชื่อไฟล์ปัจจุบันที่มาจาก metadata ไม่ใช่ Generic Name 
                        # (เช่นเป็นชื่อไฟล์ที่ดีอยู่แล้ว) จะใช้ชื่อไฟล์นั้นต่อไปโดยปริยาย

                # --- 3. กรณีไม่มี Doc ID หรือเข้าถึงชื่อไฟล์ไม่ได้เลย (เหลือเป็น Unknown) ---
                elif not doc_id:
                    # ถ้าไม่มี doc_id และ filename เป็น Unknown
                    if current_filename.lower() == "unknown":
                        resolved_entry["filename"] = "MISSING-SOURCE-METADATA"
                        self.logger.error("Evidence found with no doc_id and generic filename.")
                
                # เพิ่ม entry เข้าไป (ไม่ว่าจะได้รับการแก้ไขหรือไม่)
                resolved_entries.append(resolved_entry)

            return resolved_entries

    
    # -------------------- Contextual Rules Handlers (FIXED) --------------------
    def _load_contextual_rules_map(self) -> Dict[str, Dict[str, str]]:
        """
        Loads the contextual rules JSON file from the Tenant-Only path: 
        config/mapping/{tenant}/{tenant}_{enabler}_contextual_rules.json
        """
        
        # 🎯 สร้างชื่อไฟล์ตาม Pattern
        rules_filename = f"{self.config.tenant.lower()}_{self.enabler_id.lower()}_contextual_rules.json"

        # 1. TARGET PATH (Tenant-Only): config/mapping/{tenant}/
        base_dir_tenant_only = os.path.join(
            RUBRIC_CONFIG_DIR, 
            self.config.tenant.lower() # <--- ใช้ TENANT เท่านั้น
        )
        filepath_tenant_only = os.path.join(PROJECT_ROOT, base_dir_tenant_only, rules_filename)
        
        filepath = None
        if os.path.exists(filepath_tenant_only):
            filepath = filepath_tenant_only
            self.logger.info(f"✅ Contextual Rules loaded from Tenant-Only path: {filepath_tenant_only}")

        
        if not filepath:
            self.logger.warning(
                f"⚠️ Contextual Rules map not found at expected path. Using empty map. "
                f"(Expected File: {rules_filename} in {base_dir_tenant_only})"
            )
            return {}

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.logger.info(f"✅ Loaded Contextual Rules from {filepath}. ({len(data)} sub-criteria rules)")
                return data
        except json.JSONDecodeError as e:
            self.logger.error(f"❌ Failed to parse Contextual Rules JSON from {filepath}: {e}")
            return {}
        except Exception as e:
            self.logger.error(f"❌ Failed to load Contextual Rules from {filepath}: {e}")
            return {}

    def _collect_previous_level_evidences(self, sub_id: str, current_level: int) -> Dict[str, List[Dict]]:
            """
            ดึงหลักฐาน (Metadata + Text) ที่ผ่านจาก Level ก่อนหน้าทั้งหมด 
            เพื่อใช้เป็น Baseline Context ใน Level ปัจจุบัน (Sequential Mode)
            
            🎯 FIX: เพิ่ม Logic การ Mapping Stable Doc ID เป็น Chunk UUIDs เพื่อแก้ปัญหา Hydration Fail
            """
            collected = {}
            # ดึงค่าคงที่ที่จำเป็น (สมมติว่ามีการ import EVIDENCE_DOC_TYPES มาแล้ว)
            try:
                from config.global_vars import EVIDENCE_DOC_TYPES
            except ImportError:
                EVIDENCE_DOC_TYPES = "evidence"

            # 1. ใช้ evidence_map ของ engine ตัวเอง (shared ระหว่าง Level)
            source_map = self.evidence_map
            source_name = "evidence_map (SEQ/PAR Main)"

            # 2. รวบรวม Metadata จาก Level ก่อนหน้า
            for key, evidence_list in source_map.items():
                if key.startswith(f"{sub_id}.L") and isinstance(evidence_list, list):
                    try:
                        level_num = int(key.split(".L")[-1])
                        if level_num < current_level:
                            collected[key] = evidence_list
                    except (ValueError, IndexError):
                        continue

            # 3. HYDRATION: ดึง Text เต็มจาก Chroma ด้วย chunk_uuid
            vectorstore_manager = self.vectorstore_manager
            is_hydration_needed = vectorstore_manager is not None and collected

            if is_hydration_needed:
                # รวบรวม Stable Doc/Chunk IDs ทั้งหมดที่ถูกเลือกใน Level ก่อนหน้า
                all_uuids_raw = [
                    # ใช้ chunk_uuid หรือ doc_id ซึ่งในตอนนี้คือ Stable Doc ID (64 char)
                    str(
                        ev.get('chunk_uuid') or ev.get('stable_doc_uuid') or ev.get('doc_id')
                    ).strip()
                    for ev_list in collected.values()
                    for ev in ev_list
                    # กรอง ID ที่เป็น None/ว่าง หรือเป็นตัวชั่วคราว/Hash ที่ไม่สมบูรณ์ตั้งแต่ต้น
                    if (ev.get('chunk_uuid') or ev.get('stable_doc_uuid') or ev.get('doc_id')) 
                    and not str(ev.get('chunk_uuid') or ev.get('stable_doc_uuid') or ev.get('doc_id')).startswith(("TEMP-", "HASH-", "Unknown"))
                ]
                
                # แบ่ง ID เป็น Stable Doc ID (64 ตัว) และ Chunk ID อื่นๆ (ที่มี _index หรืออื่นๆ)
                all_uuids_stable_doc_only = list(set([uid for uid in all_uuids_raw if len(uid) == 64]))
                all_uuids_non_stable_doc = list(set([uid for uid in all_uuids_raw if len(uid) > 64])) # Chunk ID ที่ถูกต้องควรยาวกว่า 64

                # 🎯 NEW FIX: ทำการแปลง Stable Doc ID (64-char) เป็น Chunk UUIDs (64-char_index)
                chunk_uuids_for_chroma = []
                mapped_count = 0
                
                if not hasattr(vectorstore_manager, 'doc_id_map') or not vectorstore_manager.doc_id_map:
                    self.logger.warning("VSM Doc ID Map is missing! Using raw IDs for ChromaDB (may fail).")
                    chunk_uuids_for_chroma = all_uuids_raw # Fallback
                else:
                    # 1. แปลง Stable Doc IDs
                    for input_id in all_uuids_stable_doc_only:
                        # 🎯 แปลง: ใช้ Stable Doc ID ค้นหา Chunk UUIDs ที่ถูกต้อง
                        mapped_info = vectorstore_manager.doc_id_map.get(input_id, {})
                        full_chunk_list = mapped_info.get('chunk_uuids', [])
                        chunk_uuids_for_chroma.extend(full_chunk_list)
                        mapped_count += 1
                    
                    # 2. เพิ่ม Chunk IDs ที่อาจถูกบันทึกมาอย่างถูกต้อง (ยาวกว่า 64 ตัว)
                    chunk_uuids_for_chroma.extend(all_uuids_non_stable_doc)

                    if mapped_count > 0:
                        self.logger.info(f"VSM: Successfully mapped {mapped_count}/{len(all_uuids_stable_doc_only)} Stable Doc IDs to {len(chunk_uuids_for_chroma)} potential Chunk UUIDs.")
                    
                    # ลบซ้ำและกรอง ID ที่ว่างเปล่า/สั้นเกินไป
                    chunk_uuids_for_chroma = list(set([uid for uid in chunk_uuids_for_chroma if len(uid) > 10]))

                
                # 🎯 FIX A: เพิ่ม Log เพื่อยืนยันจำนวน ID ที่รวบรวมได้ทั้งหมด
                total_metadata_chunks = sum(len(v) for v in collected.values())
                self.logger.info(f"DEBUG HYDRATION: Total evidence entries found in metadata (evidence_map): {total_metadata_chunks} items.")
                
                # ใช้ chunk_uuids_for_chroma แทน all_uuids
                if not chunk_uuids_for_chroma:
                    self.logger.warning(
                        f"No valid Chunk UUIDs found for hydration. Raw IDs found: {len(all_uuids_raw)}. "
                        "Skipping hydration for previous levels."
                    )
                    full_chunks = []
                else:
                    collection_name = f"{EVIDENCE_DOC_TYPES.lower()}_{self.enabler_id.lower()}"

                    try:
                        self.logger.info(
                            f"🚨 DEBUG: Attempting to HYDRATE {len(chunk_uuids_for_chroma)} unique chunks from Collection: '{collection_name}'. "
                            f"First 3 IDs: {chunk_uuids_for_chroma[:3]}" # แสดงตัวอย่าง ID เพื่อตรวจสอบ (ตอนนี้ควรมี _index)
                        )
                        # 🚨 การเรียกใช้ ChromaDB ด้วย Chunk UUIDs ที่ถูกต้อง
                        retrieved_lc_docs = vectorstore_manager.retrieve_by_chunk_uuids(chunk_uuids_for_chroma, collection_name)

                        full_chunks = []
                        for doc in retrieved_lc_docs:
                            chunk_dict = doc.metadata.copy()
                            chunk_dict["text"] = doc.page_content
                            # รับประกันว่า chunk_uuid มีค่า
                            chunk_dict["chunk_uuid"] = (
                                doc.metadata.get("chunk_uuid")
                                or doc.metadata.get("id")
                                or doc.metadata.get("_id")
                            )
                            # เพิ่ม Stable Doc ID เพื่อใช้ในการค้นหาในขั้นตอนถัดไป
                            chunk_dict["stable_doc_uuid"] = doc.metadata.get("stable_doc_uuid") or doc.metadata.get("doc_id")
                            
                            full_chunks.append(chunk_dict)

                        self.logger.info(f"Successfully hydrated {len(full_chunks)} chunks from previous levels")
                        
                        # 🎯 FIX B: ตรวจสอบ ID ที่หายไป (Critical Debug)
                        retrieved_uuids = {c.get('chunk_uuid') for c in full_chunks if c.get('chunk_uuid')}
                        missing_uuids = set(chunk_uuids_for_chroma) - retrieved_uuids
                        
                        if missing_uuids:
                            self.logger.error(
                                f"❌ FATAL HYDRATION ISSUE: {len(missing_uuids)} chunks were requested but NOT FOUND in ChromaDB. "
                                f"Example missing IDs (3): {list(missing_uuids)[:3]}"
                            )
                            self.logger.error(f"Please verify: 1. ChromaDB is loaded from the correct base_path. 2. These IDs were INGESTED correctly.")
                            
                    except Exception as e:
                        self.logger.error(f"Failed to retrieve full chunks for baseline: {e}")
                        full_chunks = []

                # 4. สร้าง map เพื่อรวม Text เข้ากับ metadata เดิม
                
                # 🟢 FIX: เปลี่ยนการทำ map keying เพื่อให้ค้นหาจาก Stable Doc ID หรือ Chunk ID ได้
                # Map retrieved full chunks by their full chunk ID (Primary Key)
                full_chunk_map_by_chunk_uuid = {c.get('chunk_uuid'): c for c in full_chunks if c.get('chunk_uuid')}
                # Map retrieved full chunks by their Stable Doc ID (Hash 64)
                full_chunk_map_by_stable_doc_id = {}
                for c in full_chunks:
                    s_doc_id = c.get('stable_doc_uuid') or c.get('doc_id')
                    if s_doc_id and s_doc_id not in full_chunk_map_by_stable_doc_id:
                        # เลือกเก็บ chunk แรก (เพื่อป้องกันการซ้ำซ้อนในการค้นหาด้วย Stable Doc ID)
                        full_chunk_map_by_stable_doc_id[s_doc_id] = c 

                hydrated_collected = {}
                for key, ev_list in collected.items():
                    hydrated_list = []
                    for ev_metadata in ev_list:
                        # 5. ใช้ ID ที่ถูกบันทึกใน metadata (Stable Doc ID) ในการค้นหา
                        uuid_key = ev_metadata.get('chunk_uuid') or ev_metadata.get('stable_doc_uuid') or ev_metadata.get('doc_id')
                        
                        full_chunk = None

                        # 5.1 ลองค้นหาด้วย Full Chunk UUID (หาก L1 บันทึกมาถูกต้อง)
                        if len(uuid_key) > 64:
                            full_chunk = full_chunk_map_by_chunk_uuid.get(uuid_key)
                        
                        # 5.2 ถ้าหาไม่เจอ หรือ ID ที่บันทึกมาเป็น Stable Doc ID (64 ตัว) ให้ใช้ Stable Doc ID ค้นหา
                        if full_chunk is None and len(uuid_key) == 64:
                            full_chunk = full_chunk_map_by_stable_doc_id.get(uuid_key)

                        if full_chunk and full_chunk.get('text'):
                            combined = full_chunk.copy()
                            # 🟢 FIX: ใช้ content/text ของ chunk ที่ดึงมา (Full Text) และใช้ metadata เดิมที่ LLM เลือก (Short Text)
                            combined['text'] = full_chunk['text'] # Full Text
                            
                            # นำ metadata ส่วนอื่น ๆ จาก ev_metadata มาอัพเดต (เช่น score, reason จาก L1)
                            # ยกเว้น 'text' เพื่อไม่ให้ Text สั้นจาก L1 ทับ Text เต็ม
                            metadata_to_update = {k:v for k,v in ev_metadata.items() if k != 'text'}
                            combined.update(metadata_to_update)
                            
                            hydrated_list.append(combined)
                        else:
                            # ถ้าดึงไม่ได้ ยังใส่ metadata เปล่าไว้ (ไม่ให้หาย)
                            hydrated_list.append(ev_metadata)

                    if hydrated_list:
                        hydrated_collected[key] = hydrated_list

                collected = hydrated_collected

            else:
                if collected:
                    self.logger.info("Hydration skipped: vectorstore_manager not ready")

            # 6. Debug Log
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
        Retrieves the specific Contextual Rule prompt for a given Sub-Criteria and Level,
        รวมถึงการ Inject กฎ L5 พิเศษ หาก Level == 5
        """
        sub_id_rules = self.contextual_rules_map.get(sub_id)
        rule_text = ""
        
        # 1. ดึงกฎเฉพาะเกณฑ์ย่อย (ถ้ามี)
        if sub_id_rules:
            level_key = f"L{level}"
            specific_rule = sub_id_rules.get(level_key)
            if specific_rule:
                rule_text += f"\n--- กฎเฉพาะเกณฑ์ย่อย ({sub_id} L{level}) ---\nหลักฐานที่เกี่ยวข้องควรแสดงความสอดคล้องกับข้อกำหนดต่อไปนี้: {specific_rule}\n"
        
        # 2. **INJECT L5 SPECIAL RULE (Safe Injection)**
        # ใส่กฎ Bonus 2.0 เมื่อประเมิน L5 เท่านั้น เพื่อป้องกันการรบกวน L3/L4
        if level == 5:
            l5_bonus_rule = """
            \n--- L5 SPECIAL RULE (Innovation & Sustainability) ---
            * **L5 PASS Condition (บังคับ):** หาก Level นี้คือ **L5** ท่านต้องให้คะแนนรวม PDCA (P+D+C+A) ตามหลักฐาน L3/L4 ที่ค้นพบก่อน (สูงสุด 8.0)
            * **เงื่อนไข Bonus 2.0:** หากคะแนนรวม PDCA ได้ **≥ 7.0** **และ** ท่านพบหลักฐานที่ชัดเจนเพียงพอ **อย่างน้อย 1 ชิ้น** ใน Context ที่แสดงถึง:
                * (a) รางวัล KM / นวัตกรรม (Innovation Award)
                * (b) ตัวเลขผลลัพธ์เชิงธุรกิจที่วัดได้ (ROI, Productivity, Cost Saving)
                * (c) การเผยแพร่/การยอมรับจากภายนอก (External Recognition/Publication)
            * **ให้พิจารณาให้ Bonus Score 2.0 ทันที** เพื่อให้คะแนนรวมสุดท้าย **Score ≥ 9.0** และตั้ง **is_passed=true** เพื่อสะท้อนความเป็นเลิศ (ป้องกันการ Reset คะแนน)
            """
            rule_text += l5_bonus_rule
            
        return rule_text
    

    def _load_rubric(self) -> Dict[str, Any]:
        """
        Loads the SEAM rubric JSON file from the Tenant-Only path: 
        config/mapping/{tenant}/{tenant}_{enabler}_rubric.json
        """
        # 1. สร้างชื่อไฟล์ตาม Pattern
        try:
            rubric_filename = RUBRIC_FILENAME_PATTERN.format(
                tenant=self.config.tenant.lower(), 
                enabler=self.enabler_id.lower()
            )
        except KeyError:
             # Fallback
            rubric_filename = f"{self.enabler_id.lower()}_rubric.json"
            self.logger.warning(f"RUBRIC_FILENAME_PATTERN is malformed. Defaulting to {rubric_filename}.")

        # 2. TARGET PATH (Tenant-Only): config/mapping/{tenant}/
        base_dir_tenant_only = os.path.join(
            RUBRIC_CONFIG_DIR, 
            self.config.tenant.lower() # <--- ใช้ TENANT เท่านั้น
        )
        filepath_tenant_only = os.path.join(PROJECT_ROOT, base_dir_tenant_only, rubric_filename)
        
        # 3. ตรวจสอบและโหลดไฟล์
        filepath = None
        if os.path.exists(filepath_tenant_only):
            filepath = filepath_tenant_only
        
        if not filepath:
            self.logger.error(
                f"Rubric file not found for {self.enabler_id} ({self.config.tenant}) "
                f"at expected path: {filepath_tenant_only}"
            )
            raise FileNotFoundError(f"Rubric not found at expected path: {filepath_tenant_only}")

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.logger.info(f"✅ Loaded Rubric for {self.enabler_id} ({self.config.tenant}/{self.config.year}) from {filepath}")
            return data
        except json.JSONDecodeError as e:
            self.logger.error(f"❌ Failed to parse Rubric JSON from {filepath}: {e}")
            raise ValueError(f"Invalid Rubric JSON format in {filepath}: {e}")
        except Exception as e:
            self.logger.error(f"❌ Failed to load Rubric from {filepath}: {e}")
            raise
    
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
            บันทึก evidence map อย่างปลอดภัย 100% - Atomic + Lock + Clean + Sort + Score
            """
            map_file_path = self.evidence_map_path
            lock_path = map_file_path + ".lock"
            tmp_path = None

            logger.info(f"[EVIDENCE] Saving evidence map → {map_file_path}")

            try:
                with FileLock(lock_path, timeout=60):
                    logger.debug("[EVIDENCE] Lock acquired.")

                    if map_to_save is not None:
                        final_map_to_write = map_to_save
                    else:
                        # 1. โหลดของเก่าจากดิสก์
                        existing_map = self._load_evidence_map(is_for_merge=True) or {}
                        runtime_map = deepcopy(self.evidence_map)

                        # 2. Merge: เก่า + ใหม่ (ไม่ทับ แต่รวม)
                        final_map_to_write = existing_map
                        for key, entries in runtime_map.items():
                            if key not in final_map_to_write:
                                final_map_to_write[key] = []
                                
                            # 🟢 FIX 1: Deduplicate โดยใช้ Chunk UUID (หรือ doc_id เป็น Fallback)
                            existing_ids = {
                                e.get("chunk_uuid", e.get("doc_id", "N/A")) 
                                for e in final_map_to_write[key]
                            }
                            
                            # 🟢 FIX 1: ใช้ Logic ID เดียวกันในการตรวจสอบ
                            new_entries = [
                                e for e in entries 
                                if e.get("chunk_uuid", e.get("doc_id", "N/A")) not in existing_ids
                            ]
                            final_map_to_write[key].extend(new_entries)

                        # 3. ทำความสะอาดสุดท้าย (TEMP-, HASH-, Unknown)
                        final_map_to_write = self._clean_temp_entries(final_map_to_write)

                        # 4. เรียงลำดับในแต่ละ key จาก relevance_score สูง → ต่ำ (ถ้ามี)
                        for key, entries in final_map_to_write.items():
                            if entries and "relevance_score" in entries[0]:
                                entries.sort(
                                    key=lambda x: x.get("relevance_score", 0.0),
                                    reverse=True
                                )

                    if not final_map_to_write:
                        logger.warning("[EVIDENCE] Nothing to save.")
                        return

                    # เตรียมโฟลเดอร์
                    os.makedirs(os.path.dirname(map_file_path), exist_ok=True)

                    # Atomic write
                    with tempfile.NamedTemporaryFile(
                        mode='w', delete=False, encoding="utf-8", dir=os.path.dirname(map_file_path)
                    ) as tmp_file:
                        cleaned_for_json = self._clean_map_for_json(final_map_to_write)
                        json.dump(cleaned_for_json, tmp_file, indent=4, ensure_ascii=False)
                        tmp_path = tmp_file.name

                    shutil.move(tmp_path, map_file_path)
                    tmp_path = None

                    # สรุปสถิติสุดท้าย (สวยมาก)
                    total_keys = len(final_map_to_write)
                    total_items = sum(len(v) for v in final_map_to_write.values())
                    file_size_kb = os.path.getsize(map_file_path) / 1024

                    logger.info(f"[EVIDENCE] Evidence map saved successfully!")
                    logger.info(f"   Keys: {total_keys} | Items: {total_items} | Size: ~{file_size_kb:.1f} KB")

                    # โชว์ Top 1 ของแต่ละ sub-criteria (สุดยอดมาก)
                    preview = []
                    for key in sorted(final_map_to_write.keys())[:5]:  # แสดงแค่ 5 อันแรก
                        entries = final_map_to_write[key]
                        if entries:
                            top = entries[0]
                            score = top.get("relevance_score", "-")
                            preview.append(f"{key}: {top['filename'][:50]} ({score})")
                    if preview:
                        logger.info(f"   Top evidence preview → {', '.join(preview[:3])}{'...' if len(preview)>3 else ''}")

            except TimeoutError:
                logger.critical(f"[EVIDENCE] Lock timeout! Another process may be stuck: {lock_path}")
                raise
            except Exception as e:
                logger.critical("[EVIDENCE] FATAL SAVE ERROR")
                logger.exception(e)
                raise
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try: os.unlink(tmp_path)
                    except: pass
                logger.debug(f"[EVIDENCE] Lock released: {lock_path}")


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
        """
        Transforms the hierarchical rubric structure loaded in self.rubric
        into a flat list of statements ready for assessment.
        """
        if not self.rubric:
            self.logger.warning("Cannot flatten rubric: self.rubric is empty.")
            return []
            
        data = deepcopy(self.rubric)
        extracted_list = []
        
        if not isinstance(data, dict):
             self.logger.error("Rubric data structure is invalid (expected dict of criteria).")
             return []
             
        for criteria_id, criteria_data in data.items():
            # 🎯 FIX 1: ตรวจสอบ Criteria Data
            if not isinstance(criteria_data, dict):
                self.logger.warning(f"Skipping malformed criteria entry: {criteria_id} (not a dict).")
                continue
                
            sub_criteria_map = criteria_data.get('subcriteria', {})
            criteria_name = criteria_data.get('name')
            
            # 🎯 FIX 2: ตรวจสอบ Sub-criteria Map
            if not isinstance(sub_criteria_map, dict):
                 self.logger.warning(f"Skipping criteria {criteria_id}: 'subcriteria' is not a dictionary.")
                 continue

            for sub_id, sub_data in sub_criteria_map.items():
                
                # 🎯 FIX 3: ตรวจสอบว่า sub_data เป็น Dictionary ก่อนใช้งาน (ป้องกัน TypeError)
                if not isinstance(sub_data, dict):
                    self.logger.warning(
                        f"Skipping malformed sub-criteria entry: {criteria_id}.{sub_id} "
                        f"is not a dictionary (found type: {type(sub_data).__name__})."
                    )
                    continue
                
                # กำหนดค่า Metadata ที่จำเป็น
                sub_data['criteria_id'] = criteria_id
                sub_data['criteria_name'] = criteria_name
                sub_data['sub_id'] = sub_id 
                sub_data['sub_criteria_name'] = sub_data.get('name', criteria_name + ' sub')
                if 'weight' not in sub_data:
                    sub_data['weight'] = criteria_data.get('weight', 0)
                extracted_list.append(sub_data)

        # Re-check and re-sort levels
        final_list = []
        for sub_criteria in extracted_list: 
            
            # 🎯 FIX 4: ตรวจสอบและแปลง Level จาก Dict ที่มี Key เป็น String เป็น List
            if "levels" in sub_criteria and isinstance(sub_criteria["levels"], dict):
                levels_list = []
                for level_str, statement in sub_criteria["levels"].items():
                    try:
                        # พยายามแปลง Level Key ให้เป็น Integer
                        level_int = int(level_str)
                        if isinstance(statement, str):
                            levels_list.append({"level": level_int, "statement": statement})
                        else:
                            self.logger.warning(f"Level {level_str} statement in {sub_criteria.get('sub_id')} is not a string.")
                    except ValueError:
                        self.logger.error(f"Invalid level key '{level_str}' found in {sub_criteria.get('sub_id', 'UNKNOWN_ID')}. Skipping.")
                        continue
                        
                sub_criteria["levels"] = levels_list
            
            # จัดเรียงระดับ
            if "levels" in sub_criteria and isinstance(sub_criteria["levels"], list):
                 sub_criteria["levels"].sort(key=lambda x: x.get("level", 0))
                 final_list.append(sub_criteria)
            else:
                 self.logger.warning(f"Sub-criteria {sub_criteria.get('sub_id', 'UNKNOWN_ID')} missing 'levels' list.")


        return final_list

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

        
        # 2. แปลงรายการทั้งหมดให้เป็น Chunk UUID (String) และ Dedup
        # 🟢 FIX 2: เน้นใช้ Chunk UUID หรือ Stable Doc UUID เป็น ID หลัก
        doc_ids_for_dedup: List[str] = [
            (
                item.get('chunk_uuid') 
                or item.get('stable_doc_uuid') # <-- เน้น Stable Doc UUID
                or item.get('doc_id')
            )
            for item in all_priority_items
            if isinstance(item, dict) and (
                item.get('chunk_uuid') or item.get('stable_doc_uuid') or item.get('doc_id')
            )
        ]

        mapped_chunk_uuids: List[str] = list(set([uid for uid in doc_ids_for_dedup if uid is not None])) # กรอง None ออก
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
                        doc_uuids=mapped_chunk_uuids, # <-- ใช้ Chunk/Stable Doc UUIDs
                        doc_type=doc_type,
                        enabler=self.enabler_id,
                        vectorstore_manager=vectorstore_manager
                    )
                    
                    initial_priority_chunks: List[Dict[str, Any]] = retrieved_docs_result.get("top_evidences", [])
                    
                    if initial_priority_chunks:
                        # Rerank เพื่อเลือก Chunk ที่เกี่ยวข้องที่สุด
                        reranker = get_global_reranker() 
                        rerank_query = rag_queries_for_vsm[0] 
                        
                        # สร้าง LcDocument list สำหรับ Rerank (ต้องนำเข้า LcDocument)
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

                            # === วิชามารสุดท้ายที่ฆ่า 0.0000 ตลอดกาล ===
                            # เขียน relevance_score กลับลง metadata ก่อน

                            # 🛑 [แก้ไข V3] แทนที่จะพึ่ง reranker.scores เราจะวนลูปผ่าน reranked_docs 
                            # และพยายามดึง score จาก metadata ของมัน
                            
                            scores = []
                            # ตรวจสอบว่า Reranker ได้เขียน score เข้าไปใน Document.metadata หรือไม่
                            for doc in reranked_docs:
                                # score ที่มาจาก reranker อาจจะอยู่ใน metadata ภายใต้ key 'relevance_score' หรือ 'score'
                                score = doc.metadata.get('relevance_score') or doc.metadata.get('score', 0.0)
                                scores.append(float(score))
                                # บังคับเขียน relevance_score กลับเข้าไปใน metadata
                                doc.metadata["relevance_score"] = float(score)

                            if scores: # ตรวจสอบว่ามี Scores ที่รวบรวมได้
                                
                                # 🟢 [แก้ไข] 1. Log สรุปด้วย INFO และ CRITICAL
                                num_reranked = len(reranked_docs)
                                logger.info(f"✨ Reranking success ({sub_id} L{level}) → Prioritized {num_reranked} chunks. Logging top scores:")
                                # 🎯 Log 2 บรรทัดที่ท่านต้องการ!
                                logger.critical(f"✨ RERANK SCORE LOG (PRIORITY CHUNKS) ({sub_id} L{level}) → Logging top {min(5, num_reranked)} scores:")
                                
                                for i in range(len(reranked_docs)):
                                    doc = reranked_docs[i]
                                    score = scores[i] # ใช้ score ที่ดึงมาแล้ว
                                    
                                    # 🟢 [แก้ไข] 2. เพิ่มเงื่อนไขและ Log รายละเอียดด้วย CRITICAL
                                    if i < 5: 
                                        filename = doc.metadata.get('filename', doc.metadata.get('source_filename', 'N/A'))
                                        doc_id_full = doc.metadata.get('doc_id', doc.metadata.get('chunk_uuid', 'N/A'))
                                        
                                        # ตัด Chunk ID ให้สั้นลง
                                        if len(doc_id_full) > 8 and '_' in doc_id_full:
                                            doc_id_short = doc_id_full.split('_')[0][:8]
                                        else:
                                            doc_id_short = doc_id_full[:8]
                                            
                                        logger.critical(f"  > Rerank #{i+1}: {doc_id_short} ({filename}) | Score: {float(score):.4f}")

                            # แปลงกลับเป็น dict และให้ 'score' ทับค่าเก่าอย่างแน่นอน
                            priority_docs = []
                            for d in reranked_docs:
                                # เริ่มจาก metadata เดิม
                                item = dict(d.metadata)
                                # อัปเดตข้อมูลที่จำเป็น
                                item.update({
                                    'content': d.page_content,
                                    'text': d.page_content,
                                    # สำคัญที่สุด: score ต้องมาท้ายสุด และทับแน่นอน
                                    'score': float(d.metadata.get('relevance_score', 0.0)),
                                    'relevance_score': float(d.metadata.get('relevance_score', 0.0))
                                })
                                priority_docs.append(item)
                            # ========================================

                            logger.critical(f"DEBUG: Limited and prioritized {len(priority_docs)} chunks from {num_historical_chunks} mapped UUIDs.")
                        else:
                            # fallback กรณีไม่มี reranker
                            priority_docs = initial_priority_chunks[:self.PRIORITY_CHUNK_LIMIT]
                            # แม้ fallback ก็ยังใส่ score ให้ครบ
                            for item in priority_docs:
                                if 'score' not in item:
                                    item['score'] = 0.0

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
        ประเมินเกณฑ์ย่อยในระดับที่กำหนด โดยดึงบริบท (Evidence) และให้ LLM ประเมิน
        
        Args:
            args: tuple(statement_data, engine_config_dict)
        Returns:
            - raw_results_for_sub: list of final results for each level
            - final_sub_result: summary of sub-criteria evaluation
            - level_evidences: dict of evidences to merge later in main process (เฉพาะ Metadata)
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
        
        # 🟢 NEW: Unpack Rerank Gate Constants
        # NOTE: RERANK_THRESHOLD และ MAX_EVI_STR_CAP จะถูกใช้ใน _calculate_evidence_strength_cap()
        
        # ดึง Baseline Context ที่ถูก Hydrate (มี Text) จาก Main Process
        previous_levels_evidence_full = engine_config_dict.get('previous_levels_evidence_full', []) 

        # Statement metadata
        level = int(statement_data.get("level", 0))
        statement_text = statement_data.get("statement", "")
        sub_criteria_name = statement_data.get("sub_criteria_name", "")
        pdca_phase = statement_data.get("pdca_phase", "")
        level_constraint = statement_data.get("level_constraint", "")
        sub_id = statement_data.get("sub_criteria_id", statement_data.get("sub_id", ""))

        # Determine retrieval/evaluation functions (Assuming existence of these helpers)
        if level <= 2:
            retrieval_func = self.retrieve_context_for_low_levels 
            evaluation_func = self.evaluate_with_llm_low_level
            top_k = 5
        else:
            retrieval_func = self.retrieve_context_with_filter
            evaluation_func = self.evaluate_with_llm
            top_k = 10

        # Build enhanced query for RAG (Assuming existence of this helper)
        rag_query_list = self.enhance_query_for_statement(
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
        # NOTE: Assuming vectorstore_manager is part of 'self' or passed correctly to the helper
        self.logger.info(f"   > Starting assessment for {sub_id} L{level} (Attempt: 1)...") # <-- Re-added the missing log
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

        # 🟢 DEBUG: บังคับแสดง Key ทั้งหมดของ retrieval_result (สำคัญมาก!)
        self.logger.critical(f"🔑 DEBUG L{level}: RETRIEVAL KEYS (All): {list(retrieval_result.keys())}") 
        if top_evidences: 
            self.logger.critical(
                f"DEBUG L{level}: Inspecting first document (Type: {type(top_evidences[0])})"
                f" | EVIDENCE KEYS: {list(top_evidences[0].keys())}"
            )
        # 🟢 END DEBUG

        # ------------------ 🟢 Relevant Score Gate Logic (NEW) ------------------
        # 🎯 FIX: ใช้ Helper Function ที่สร้างใหม่เพื่อค้นหาคะแนนสูงสุดและกำหนด Cap
        evi_cap_data = self._calculate_evidence_strength_cap(
            top_evidences=top_evidences,
            level=level,
        )
        
        # ดึงค่าที่คำนวณได้จาก Helper มาใช้ต่อ
        highest_rerank_score = evi_cap_data['highest_rerank_score']
        max_score_source = evi_cap_data['max_score_source']
        is_capped = evi_cap_data['is_capped']
        max_evi_str_for_prompt = evi_cap_data['max_evi_str_for_prompt']
        
        # NOTE: การ Log การ Cap ได้ย้ายไปอยู่ใน _calculate_evidence_strength_cap แล้ว

        # ------------------ END: Relevant Score Gate Logic ------------------
        
        # Build multichannel context (Assuming existence of this helper)
        channels = self.build_multichannel_context_for_level(
            level=level, 
            top_evidences=top_evidences, 
            previous_levels_evidence=previous_levels_evidence_full 
        )

        # Evaluate statement (Assuming existence of these helpers)
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
            aux_summary=channels.get("aux_summary", ""),
            # 🟢 Pass max_evidence_strength
            max_evidence_strength=max_evi_str_for_prompt
        )

        # Summarize context for report (Assuming existence of this helper)
        summary_result = self.create_context_summary_llm(
            context=channels.get("direct_context", "") or aggregated_context,
            sub_criteria_name=sub_criteria_name,
            level=level,
            sub_id=sub_id,
            llm_executor=llm_executor
        )

        # Prepare evidences to return (for main process to merge)
        level_key = f"{sub_id}.L{level}"
        level_evidences = {
            level_key: [
                {
                    "doc_id": ev.get("doc_id"),
                    "filename": ev.get("source_filename") or ev.get("source") or ev.get("filename"),
                    "relevance_score": ev.get("rerank_score", ev.get("score", 0.0)), # ใช้ rerank_score ที่มาจาก top_evidences โดยตรง (ซึ่งถูก patch มาจาก retrieval_func แล้ว)
                    "mapper_type": "AI_GENERATED", 
                    "timestamp": datetime.now().isoformat()
                }
                for ev in top_evidences if ev.get("doc_id")
            ]
        }

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
            "summary": summary_result,
            
            # 🟢 Relevant Score Gate Metadata
            "max_relevant_score": evi_cap_data['highest_rerank_score'],
            "max_relevant_source": evi_cap_data['max_score_source'],
            "is_evidence_strength_capped": evi_cap_data['is_capped'],
            "max_evidence_strength_used": evi_cap_data['max_evi_str_for_prompt'],
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
                
                    # 🟢 FIX 1: ใช้วิธี Lookup Filename จาก doc_id_to_filename_map ถ้า filename เป็น 'Unknown' หรือขาดหาย
                    resolved_temp_map = []
                    for ev in level_temp_map:
                        filename = ev.get("filename")
                        doc_id = ev.get("doc_id")
                        
                        # หาก filename ไม่ถูกต้อง (e.g., 'Unknown' หรือ None) ให้พยายาม lookup
                        if not filename or filename == "Unknown":
                            # ใช้แผนที่หลักของ Engine ในการค้นหาชื่อไฟล์จาก doc_id
                            # NOTE: self.doc_id_to_filename_map คือ map ที่โหลดไว้ตอน Engine Init
                            resolved_filename = self.doc_id_to_filename_map.get(doc_id) 
                            if resolved_filename:
                                ev['filename'] = resolved_filename
                                self.logger.debug(f"Resolved 'Unknown' filename for {doc_id} to {resolved_filename}")
                            else:
                                self.logger.warning(f"Could not find filename for doc_id: {doc_id} in mapping. Keeping filename: {filename}")
                        
                        resolved_temp_map.append(ev)
                        
                    current_key = f"{sub_id}.L{level}"
                    self.temp_map_for_save[current_key] = resolved_temp_map # ใช้ resolved_temp_map ที่แก้ไขแล้ว
                    self.logger.info(f"[EVIDENCE SAVED] {current_key} → {len(resolved_temp_map)} chunks")

                    # 🎯 FIX SEQUENTIAL DEPENDENCY: อัปเดต self.evidence_map ทันทีใน Sequential Mode
                    if self.is_sequential:
                        self.evidence_map[current_key] = resolved_temp_map # ใช้ resolved_temp_map ที่แก้ไขแล้ว
                        self.logger.info(f"[SEQUENTIAL UPDATE] {current_key} added to engine's main evidence_map for L{level+1} dependency.")
                    # END FIX

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
                    # break  # หยุดทันทีเมื่อ fail 
                    pass

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

    def _calculate_evidence_strength_cap(
        self,
        top_evidences: List[Union[Dict[str, Any], Any]],  # รองรับทั้ง dict และ LcDocument
        level: int,
    ) -> Dict[str, Any]:
        """
        Relevant Score Gate เวอร์ชัน DEBUG FINAL: ดึงคะแนนจาก metadata, top-level key/attribute, และ Regex fallback ที่ครอบคลุม
        """

        highest_rerank_score = 0.0
        max_score_source = "N/A"

        score_keys = [
            "relevance_score", "rerank_score", "score", 
            "_rerank_score_force", "_rerank_score", 
            "Score", "RelevanceScore"
        ]
        
        # 💡 ดึงค่า Threshold และ Cap จาก Attribute ของ Class
        threshold = getattr(self, "RERANK_THRESHOLD", 0.5) 
        cap_value = getattr(self, "MAX_EVI_STR_CAP", 3.0)
        
        # 💡 Fallback: ถ้ายังไม่ได้ตั้งค่า Attribute ให้ดึงจาก config/global_vars
        if not isinstance(threshold, (int, float)):
            from config.global_vars import RERANK_THRESHOLD as G_RERANK_THRESHOLD
            from config.global_vars import MAX_EVI_STR_CAP as G_MAX_EVI_STR_CAP
            threshold = G_RERANK_THRESHOLD
            cap_value = G_MAX_EVI_STR_CAP


        for doc in top_evidences:
            
            # # -------------------- DEBUGGING BLOCK (START) --------------------
            # if doc is top_evidences[0]:
            #     self.logger.critical(f"DEBUG L{level}: Inspecting first document (Type: {type(doc)})")
                
            #     if isinstance(doc, dict):
            #         content = doc.get("text", "")
            #         tail_content = content[-200:] if len(content) > 200 else content
            #         self.logger.critical(f"DEBUG L{level}: Dict keys: {list(doc.keys())}")
            #         self.logger.critical(f"DEBUG L{level}: END OF 'text' content (last 200 chars): \n***\n{tail_content}\n***")
            #     else:
            #         try:
            #             doc_attrs = [attr for attr in dir(doc) if not attr.startswith('_') and not callable(getattr(doc, attr))]
            #             self.logger.critical(f"DEBUG L{level}: Doc public attributes (potential score location): {doc_attrs}")
            #         except:
            #             self.logger.critical(f"DEBUG L{level}: Cannot inspect attributes of this object type.")
            # # -------------------- DEBUGGING BLOCK (END) --------------------
            
            page_content = ""
            metadata = {}
            
            # ─── 1. แปลงเป็น metadata + content เดียวกัน (รองรับทุกโครงสร้าง) ───
            if isinstance(doc, dict):
                metadata = doc.get("metadata", {}) 
                page_content = doc.get("page_content", "") or doc.get("text", "") or doc.get("content", "")
            else:
                metadata = getattr(doc, "metadata", {})
                page_content = getattr(doc, "page_content", "") or getattr(doc, "text", "")

            # ─── 2. ค้นหาคะแนน (ตรวจสอบ top-level key/attribute และ metadata) ───
            current_score = 0.0
            
            for key in score_keys:
                score_val = None
                
                if key in metadata:
                    score_val = metadata[key]
                
                if score_val is None:
                    if isinstance(doc, dict):
                        score_val = doc.get(key)
                    else:
                        score_val = getattr(doc, key, None)

                if score_val is not None:
                    try:
                        current_score = float(score_val)
                        if current_score > 0:
                            break
                    except (ValueError, TypeError):
                        continue
            
            # ─── 3. Fallback: ดึงจากท้าย content (Aggressive Regex) ───
            if current_score == 0.0 and page_content and isinstance(page_content, str):
                # ตรวจสอบว่ามี re import หรือไม่ก่อนเรียกใช้
                try:
                    import re
                    tail = page_content[-1000:]
                    patterns = [
                        r"Relevance[ :]+([0-9]*\.?[0-9]+)",
                        r"Score[ :]+([0-9]*\.?[0-9]+)",
                        r"Re:[ ]*([0-9]*\.?[0-9]+)",
                        r"\[Relevance: ([0-9]*\.?[0-9]+)\]",
                        r"\[Score: ([0-9]*\.?[0-9]+)\]",
                        r"rerank_score['\"]?\s*:\s*([0-9]*\.?[0-9]+)",
                        r"\|\s*([0-9]*\.?[0-9]+)\s*\|",
                        r"\s+([0-9]\.[0-9]+)$",
                    ]
                    for pat in patterns:
                        m = re.search(pat, tail, re.IGNORECASE)
                        if m:
                            try:
                                current_score = float(m.group(1))
                                break
                            except:
                                continue
                except ImportError:
                    # ถ้า re ไม่ถูก import, ข้ามไป
                    pass


            # ─── 4. ดึง source ที่ดีที่สุด ───
            source = (
                metadata.get("source_filename") or metadata.get("filename") or
                doc.get("source_filename") or doc.get("filename") or 
                doc.get("source") or doc.get("doc_id") or
                "N/A"
            )

            # ─── 5. อัปเดตคะแนนสูงสุด ───
            if current_score > highest_rerank_score:
                highest_rerank_score = current_score
                max_score_source = source

        # ─── 6. Relevant Score Gate + Log ───
        
        # NOTE: ใช้ threshold และ cap_value ที่ดึงมาอย่างถูกต้อง
        if highest_rerank_score < threshold:
            max_evi_str_for_prompt = cap_value
            is_capped = True
            self.logger.warning(
                f"🚨 Evi Str CAPPED L{level}: "
                f"Rerank {highest_rerank_score:.4f} (จาก '{max_score_source}') "
                f"< {threshold} → จำกัดที่ {cap_value}"
            )
        else:
            max_evi_str_for_prompt = 10.0
            is_capped = False
            self.logger.info(
                f"✅ Evi Str FULL L{level}: "
                f"Rerank {highest_rerank_score:.4f} (จาก '{max_score_source}') "
                f">= {threshold} → ปล่อยเต็ม 10.0"
            )

        return {
            "is_capped": is_capped,
            "max_evi_str_for_prompt": max_evi_str_for_prompt,
            "highest_rerank_score": round(float(highest_rerank_score), 4),
            "max_score_source": max_score_source,
        }
        

    def run_assessment(
            self,
            target_sub_id: str = "all",
            export: bool = False,
            vectorstore_manager: Optional['VectorStoreManager'] = None,
            sequential: bool = False,
            document_map: Optional[Dict[str, str]] = None, # 🟢 FIX: รับ document_map
        ) -> Dict[str, Any]:
        """
        Main runner ของ Assessment Engine
        รองรับทั้ง Parallel และ Sequential 100%
        และรับประกันว่า evidence_map ครบทุกกรณี
        """

        start_ts = time.time()
        self.is_sequential = sequential

        # ============================== 1. Filter Rubric ==============================
        if target_sub_id.lower() == "all":
            sub_criteria_list = self._flatten_rubric_to_statements() # 🟢 NOTE: ใช้ _flatten_rubric_to_statements ที่แก้ไขแล้ว
        else:
            # 🟢 NOTE: ใช้ _flatten_rubric_to_statements เพื่อให้ได้โครงสร้าง List ก่อน
            all_statements = self._flatten_rubric_to_statements()
            sub_criteria_list = [
                s for s in all_statements if s.get('sub_id') == target_sub_id
            ]
            if not sub_criteria_list:
                self.logger.error(f"Sub-Criteria ID '{target_sub_id}' not found in rubric.")
                return {"error": f"Sub-Criteria ID '{target_sub_id}' not found."}

        # Reset states
        self.raw_llm_results = []
        self.final_subcriteria_results = []

        # โหลด evidence map ที่มีอยู่แล้ว (ไม่ clear!)
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

        run_parallel = (target_sub_id.lower() == "all") and not (sequential or export)

        # ============================== 2. Run Assessment ==============================
        if run_parallel:
            # --------------------- PARALLEL MODE ---------------------
            self.logger.info("Starting Parallel Assessment with Multiprocessing...")
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
                # ใช้ self.logger แทน logger (ถ้า logger เป็น global)
                self.logger.info(f"Using {max(1, os.cpu_count() - 1)} processes...")
                pool_ctx = multiprocessing.get_context('spawn')
                with pool_ctx.Pool(processes=max(1, os.cpu_count() - 1)) as pool:
                    # NOTE: _static_worker_process จะคืนค่า (sub_result, temp_map_from_worker)
                    results_list = pool.map(_static_worker_process, worker_args)
            except Exception as e:
                self.logger.critical(f"Multiprocessing failed: {e}")
                raise

            # รวมผลลัพธ์จากทุก worker
            for result_tuple in results_list:
                # ตรวจสอบโครงสร้างผลลัพธ์จาก worker ก่อน unpack
                if not isinstance(result_tuple, tuple) or len(result_tuple) != 2:
                    self.logger.error(f"Worker returned invalid result structure: {result_tuple}")
                    continue
                
                sub_result, temp_map_from_worker = result_tuple

                if isinstance(temp_map_from_worker, dict):
                    # รวม Evidence Map ที่ได้จาก Worker
                    for level_key, evidence_list in temp_map_from_worker.items():
                        if isinstance(evidence_list, list) and evidence_list:
                            current_list = self.evidence_map.setdefault(level_key, [])
                            current_list.extend(evidence_list)
                            self.logger.info(f"AGGREGATED: +{len(evidence_list)} → {level_key} "
                                        f"(total: {len(current_list)})")

                # รวมผลลัพธ์การประเมิน
                raw_refs = sub_result.get("raw_results_ref", [])
                self.raw_llm_results.extend(raw_refs if isinstance(raw_refs, list) else [])
                self.final_subcriteria_results.append(sub_result)


        else:
            # --------------------- SEQUENTIAL MODE ---------------------
            mode_desc = target_sub_id if target_sub_id != "all" else "All Sub-Criteria (Sequential)"
            self.logger.info(f"Starting Sequential Assessment: {mode_desc}")

            # 🎯 FIX: แก้ไข load_all_vectorstores ให้มี tenant และ year
            local_vsm = vectorstore_manager or (
                load_all_vectorstores(
                    doc_types=[EVIDENCE_DOC_TYPES], 
                    evidence_enabler=self.config.enabler,
                    tenant=self.config.tenant,  # <--- NEW: Argument ที่หายไป
                    year=self.config.year       # <--- NEW: Argument ที่หายไป
                )
                if self.config.mock_mode == "none" else None
            )
            self.vectorstore_manager = local_vsm

            for sub_criteria in sub_criteria_list:
                sub_result, final_temp_map = self._run_sub_criteria_assessment_worker(sub_criteria)
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
            final_results["export_path_used"] = export_path
            final_results["evidence_map"] = deepcopy(self.evidence_map)
            self.logger.info(f"Exported full results → {export_path}")

        return final_results

    def _run_single_assessment(
        self,
        sub_criteria: Dict[str, Any],
        statement_data: Dict[str, Any],
        vectorstore_manager: Optional['VectorStoreManager'],
        sequential_chunk_uuids: Optional[List[str]] = None,
        attempt: int = 1 # เพิ่ม attempt สำหรับ RetryPolicy ใน L3-L5
    ) -> Dict[str, Any]:
        """
        รันการประเมิน Level เดียว (L1-L5) อย่างสมบูรณ์
        - แก้ไขปัญหา LLM คืนค่าเป็น 'int' object โดยการแปลงเป็น dict ทันที
        - 🟢 NEW: Implement Relevant Score Gate
        """

        start_time = time.time() # เพิ่ม start_time เพื่อคำนวณ duration
        sub_id = sub_criteria['sub_id']
        level = statement_data['level']
        statement_text = statement_data['statement']
        sub_criteria_name = sub_criteria['sub_criteria_name']
        statement_id = statement_data.get('statement_id', sub_id)
        
        # กำหนดค่าเริ่มต้นของ duration (ถ้าเกิด error ก่อน llm call)
        retrieval_duration = 0.0 
        llm_duration = 0.0
        rag_query = statement_text # กำหนดค่าเริ่มต้น

        self.logger.info(f"  > Starting assessment for {sub_id} L{level} (Attempt: {attempt})...")

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
            self.logger.error(f"RAG retrieval failed for {sub_id} L{level}: {e}")
            return self._create_error_result(level, 'RAG Retrieval Error', start_time, 0.0)

        retrieval_duration = time.time() - retrieval_start
        top_evidences = retrieval_result.get("top_evidences", [])
        used_chunk_uuids = retrieval_result.get("used_chunk_uuids", [])

        # ==================== 6. ดึงหลักฐานจาก Level ก่อนหน้า ====================
        try:
            previous_levels_raw = self._collect_previous_level_evidences(sub_id, current_level=level)
        except Exception as e:
            self.logger.error(f"Failed to collect previous evidences: {e}")
            previous_levels_raw = {}

        previous_levels_evidence_full = []
        for ev_list in previous_levels_raw.values():
            for ev in ev_list:
                doc_id = ev.get("doc_id") or ev.get("chunk_uuid")
                if not doc_id or str(doc_id).startswith("HASH-"):
                    continue
                previous_levels_evidence_full.append(ev)

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
        self.logger.info(
            f"  > Context built → Direct: {debug.get('direct_count',0)}, "
            f"Aux: {debug.get('aux_count',0)}, "
            f"Baseline: {len(previous_levels_evidence_full)} files "
            f"from {len(previous_levels_raw)} previous levels"
        )

        # ==================== 8. LLM Evaluation ====================
        
        # 🟢 NEW: 8.1. Relevant Score Gate - Calculate Max Evidence Strength
        # self.logger.critical(f"FINAL DEBUG L{level}: DUMPING RAW top_evidences[0] JSON:")
        try:
            # ใช้ json.dumps เพื่อแปลง Object/Dict ทั้งหมดเป็น String
            # เราใช้ getattr() เพื่อดึงค่าที่เป็นไปได้ทั้งหมด
            if isinstance(top_evidences[0], dict):
                raw_doc_data = top_evidences[0]
            else:
                raw_doc_data = {'page_content': getattr(top_evidences[0], 'page_content', 'N/A'),
                                'metadata': getattr(top_evidences[0], 'metadata', {}),
                                'score': getattr(top_evidences[0], 'score', 'N/A')}
            
            # self.logger.critical(json.dumps(raw_doc_data, indent=2, ensure_ascii=False))
        except Exception as e:
             self.logger.critical(f"FINAL DEBUG L{level}: FAILED TO DUMP RAW DOC: {e}")

        evi_cap_data = self._calculate_evidence_strength_cap(top_evidences, level)
        max_evi_str_for_prompt = evi_cap_data['max_evi_str_for_prompt']
        
        context_parts = [
            f"--- DIRECT EVIDENCE (L{level})---\n{channels.get('direct_context','')}",
            f"--- AUXILIARY EVIDENCE ---\n{channels.get('aux_summary','')}",
            f"--- BASELINE FROM PREVIOUS LEVELS ---\n{channels.get('baseline_summary','ไม่มี')}"
        ]
        final_llm_context = "\n\n".join([p for p in context_parts if p.strip()])

        llm_start = time.time()
        try:
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
                # 🟢 NEW: ส่งค่า Max Evi Str Cap เข้าไปใน LLM Evaluator
                max_evidence_strength=max_evi_str_for_prompt # <--- ต้องรับใน evaluate_with_llm
            )
        except Exception as e:
            self.logger.error(f"LLM Call failed for {sub_id} L{level}: {e}")
            llm_result = {}
        
        llm_duration = time.time() - llm_start

        # =====================================================================================
        # 🎯 FINAL FIX: จัดการกับผลลัพธ์ที่เป็นตัวเลข (int) ก่อน RETURN 
        # (ใช้ calculate_pdca_breakdown_and_pass_status เพื่อคง Business Logic ไว้)
        # =====================================================================================
        
        is_numeric_result = isinstance(llm_result, (int, float)) or \
                            (isinstance(llm_result, str) and str(llm_result).strip().isdigit())
                            
        if is_numeric_result:
            level_num = int(llm_result)
            self.logger.warning(
                f"🚨 L{level} LLM returned ONLY number {level_num} (Type: {type(llm_result).__name__}). "
                f"Converting to standardized dict format to prevent RetryPolicy crash."
            )
            
            # *** FIX: เรียกฟังก์ชันที่ถูกต้องเพื่อคง Logic การ PASS/FAIL และ PDCA Breakdown ***
            try:
                # ใช้ calculate_pdca_breakdown_and_pass_status เพื่อให้ใช้ Logic L5/L4 >= 4 ได้ถูกต้อง
                # NOTE: เนื่องจากฟังก์ชันนี้ไม่ได้รับ max_evi_str_cap_for_llm เราจะใช้ค่า Evidence Strength Default
                pdca_breakdown_data, is_passed_num, _ = calculate_pdca_breakdown_and_pass_status(level_num, level) 
            except NameError:
                self.logger.error("calculate_pdca_breakdown_and_pass_status function is missing from scope.")
                is_passed_num = level_num >= level
                pdca_breakdown_data = {}

            status_num = "PASS" if is_passed_num else "FAIL"
            # *******************************************************************************

            return {
                "sub_criteria_id": sub_id,
                "statement_id": statement_id,
                "level": level,
                "statement": statement_text,
                "pdca_phase": pdca_phase,
                "llm_score": level_num,
                "pdca_breakdown": pdca_breakdown_data, # ใช้ผลลัพธ์จากฟังก์ชัน
                "is_passed": is_passed_num,             # ใช้ผลลัพธ์จากฟังก์ชัน
                "status": status_num,                   # ใช้ผลลัพธ์จากฟังก์ชัน
                "score": level_num,
                "llm_result_full": {"raw_number": level_num, "raw_type": type(llm_result).__name__},
                "retrieval_duration_s": round(retrieval_duration, 2),
                "llm_duration_s": round(llm_duration, 2),
                "top_evidences_ref": [],
                "temp_map_for_level": [],
                "evidence_strength": self.MAX_EVI_STR_CAP if is_passed_num else 0.0, # ใช้ Evi Str Cap เป็น Default
                "ai_confidence": "HIGH" if is_passed_num else "LOW",
                "evidence_count": 0,
                "pdca_coverage": 0.0,
                "direct_evidence_count": 0,
                "rag_query": rag_query,
                "full_context_meta": debug,
                # 🟢 NEW: Relevant Score Gate Metadata (ใช้ค่าจาก capiing)
                "max_relevant_score": evi_cap_data['highest_rerank_score'],
                "max_relevant_source": evi_cap_data['max_score_source'],
                "is_evidence_strength_capped": evi_cap_data['is_capped'],
                "max_evidence_strength_used": max_evi_str_for_prompt,
            }
        # =====================================================================================

        # ==================== 9-10. Scoring & Pass/Fail ====================
        
        if not isinstance(llm_result, dict):
            self.logger.error(
                f"🚨 LLM parsing failed for {statement_id} L{level}. Received unexpected type: {type(llm_result).__name__}. Setting FAIL defaults."
            )
            llm_result = {}
        
        llm_score = llm_result.get('score', 0)
        # ใช้ฟังก์ชันนี้จาก Global Scope หรือ Scope ที่กำหนดไว้
        pdca_breakdown, is_passed, _ = calculate_pdca_breakdown_and_pass_status(llm_score, level)
        status = "PASS" if is_passed else "FAIL"

        # -------------------- 11. SAVE EVIDENCE MAP (PASS ONLY) --------------------
        temp_map_for_level = None
        evidence_entries = []
        
        if is_passed and top_evidences:
            seen = set()
            
            def safe_float(val, default=0.0):
                """Convert val to float safely, fallback to default if fails"""
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return default

            for ev in top_evidences:
                doc_id = ev.get("doc_id") or ev.get("chunk_uuid")
                
                # กรอง ID ที่ไม่ถูกต้องและ Chunk ที่ซ้ำ
                if not doc_id or str(doc_id).startswith(("TEMP-", "HASH-")) or doc_id in seen:
                    continue

                # --- START: SCORE EXTRACTION REVISED ---
                score = 0.0
                metadata = ev.get("metadata", {}) or {}
                filename_to_use = ev.get("source_filename") or metadata.get("source_filename") or ""

                # 🎯 Priority 1: ดึงจาก 'relevance_score' และ 'score' ที่ Reranker/Retriever เขียนไว้
                score_sources = [
                    ev.get("relevance_score"), ev.get("score"),
                    metadata.get("relevance_score"), metadata.get("score"),
                    ev.get("rerank_score"), metadata.get("rerank_score"),
                    metadata.get("_rerank_score_force") # ค่าที่ถูกบังคับใส่
                ]
                
                for s in score_sources:
                    score = max(score, safe_float(s))
                    
                # 🎯 Priority 2: ตรวจสอบ distance (ChromaDB Similarity)
                distance = metadata.get("distance") or ev.get("distance")
                if distance is not None:
                    try:
                        distance_val = safe_float(distance)
                        similarity = round(1.0 - distance_val, 4)
                        score = max(score, similarity)
                    except (TypeError, ValueError):
                        pass

                # 🎯 Priority 3: ตรวจสอบจากชื่อไฟล์ (โค้ดเดิม)
                if "|SCORE:" in filename_to_use:
                    try:
                        score_str = filename_to_use.split("|SCORE:")[1].split("|")[0]
                        filename_score = safe_float(score_str)
                        score = max(score, filename_score)
                        filename_to_use = filename_to_use.split("|SCORE:")[0] 
                    except Exception:
                        pass
                
                # 💡 FIX: ถ้า score ยังเป็น 0.0 และไม่ใช่ Baseline Chunk ให้กำหนดเป็น 0.5 (Evidence Default)
                if score == 0.0 and (ev.get("pdca_tag") != "Baseline"):
                    score = 0.5 
                    
                score = round(score, 4)
                # --- END: SCORE EXTRACTION REVISED ---
                
                if not filename_to_use:
                    filename_to_use = ev.get("source_filename") or ev.get("source") or ev.get("filename") or metadata.get("source_filename") or metadata.get("filename") or "UNKNOWN_FILE"

                evidence_entries.append({
                    "doc_id": doc_id,
                    "filename": filename_to_use,
                    "mapper_type": "AI_GENERATED", 
                    "timestamp": datetime.now().isoformat(), 
                    "relevance_score": score, # 💡 FIX 2: ใช้ score ที่ถูกคำนวณแล้ว
                    "chunk_uuid": doc_id,
                })
                seen.add(doc_id) # เพิ่มใน seen
            
            # -------------------- 12. Calculate PDCA Coverage & Strength --------------------
            direct_count = channels.get("debug_meta", {}).get("direct_count", 0)
            
            avg_score = sum(entry.get("relevance_score", 0.0) for entry in evidence_entries) / len(evidence_entries) if evidence_entries else 0.0
            
            pdca_coverage = sum(1 for score in pdca_breakdown.values() if score > 0) / 4.0 # ใช้ Logic เดิม

            # 💡 FIX: ปรับตัวคูณจาก 1.5 เป็น 2.0 เพื่อให้ Evidence Strength มีค่าสูงขึ้น
            # 💡 FIX 2: บังคับ Evidence Strength ไม่ให้เกินค่า max_evi_str_for_prompt (จาก Capping)
            evidence_strength_raw = (avg_score * 10.0) * (pdca_coverage * 2.0)
            
            evidence_strength = min(
                max_evi_str_for_prompt, # <--- 🟢 ใช้ค่า Capped เป็นตัวควบคุมสูงสุด
                evidence_strength_raw
            )

            ai_confidence = "HIGH" if evidence_strength >= 8.0 and is_passed else \
                            "MEDIUM" if evidence_strength >= 5.5 else "LOW"

            evidence_count_for_level = len(evidence_entries)
            
            # -------------------- 13. Prepare temp_map_for_level and Finalize Evidence --------------------
            evidence_entries.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)

            final_k_reranked = self.config.final_k_reranked if hasattr(self.config, 'final_k_reranked') else 5
            evidence_entries = evidence_entries[:final_k_reranked]

            temp_map_for_level = [
                {
                    "doc_id": entry["doc_id"],
                    "filename": entry["filename"],
                    "mapper_type": entry["mapper_type"],
                    "timestamp": entry["timestamp"],
                    "relevance_score": entry["relevance_score"],
                    "chunk_uuid": entry["chunk_uuid"],
                }
                for entry in evidence_entries
            ]
        
        else:
            evidence_entries = []
            temp_map_for_level = []
            evidence_strength = 0.0
            ai_confidence = "LOW"
            pdca_coverage = 0.0
            direct_count = 0
            evidence_count_for_level = 0
        
        # ==================== 14. สร้างผลลัพธ์สุดท้าย ====================
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
            "top_evidences_ref": self._resolve_evidence_filenames(evidence_entries), 
            "temp_map_for_level": temp_map_for_level,
            "evidence_strength": round(evidence_strength, 1),
            "ai_confidence": ai_confidence,
            "evidence_count": evidence_count_for_level,
            "pdca_coverage": round(pdca_coverage, 4), 
            "direct_evidence_count": direct_count,
            "rag_query": rag_query,
            "full_context_meta": debug,
            
            # 🟢 NEW: Relevant Score Gate Metadata
            "max_relevant_score": evi_cap_data['highest_rerank_score'],
            "max_relevant_source": evi_cap_data['max_score_source'],
            "is_evidence_strength_capped": evi_cap_data['is_capped'],
            "max_evidence_strength_used": max_evi_str_for_prompt,
        }

        # 🟢 [แก้ไข] 1. สร้างตัวแปร icon_status
        icon_status = "✅" if status == "PASS" else "❌"

        # 🟢 [แก้ไข] 2. นำ icon_status ไปใส่ใน Log
        self.logger.info(f"  > Assessment {sub_id} L{level} completed → {icon_status} {status} (Score: {llm_score:.1f} | Evi Str: {final_result['evidence_strength']:.1f} | Conf: {ai_confidence})")

        return final_result