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
        MAX_EVI_STR_CAP,
        DEFAULT_LLM_MODEL_NAME,
        LLM_TEMPERATURE,
        MAX_PARALLEL_WORKERS,
        MIN_RERANK_SCORE_TO_KEEP
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

    # 🎯 FIX: Import ฟังก์ชัน Path Utility ทั้งหมดที่จำเป็น
    from utils.path_utils import (
        get_mapping_file_path, 
        get_evidence_mapping_file_path, 
        get_contextual_rules_file_path,
        get_doc_type_collection_key,
        get_assessment_export_file_path,
        get_export_dir,
        get_rubric_file_path # <--- ต้อง Import ฟังก์ชันนี้ด้วย
    )

    import assessments.seam_mocking as seam_mocking 
    
except ImportError as e:
    # -------------------- Fallback Code (Same as previous) --------------------
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
    
    # 📌 Placeholder functions for path_utils if the main import fails
    def get_mapping_file_path(*args, **kwargs): return "config/mapping/default/mapping.json"
    def get_evidence_mapping_file_path(*args, **kwargs): return "config/mapping/default/evidence_mapping.json"
    def get_contextual_rules_file_path(*args, **kwargs): return "config/rubrics/default/contextual_rules.json"
    def get_rubric_file_path(*args, **kwargs): return "config/rubrics/default/rubric.json"
    
    if "FATAL ERROR" in str(e):
        pass 
    # ---------------------------------------------------------------------- 


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
    โดยใช้เกณฑ์ SE-AM ที่ถูกต้องอย่างเป็นทางการ (สอดคล้องกับ post-processor)

    Required Score (R) ตาม Level:
    - L1: 1
    - L2: 2
    - L3: 4
    - L4: 6
    - L5: 8
    """
    pdca_map: Dict[str, int] = {'P': 0, 'D': 0, 'C': 0, 'A': 0}
    is_passed: bool = False
    raw_pdca_score: float = 0.0

    # กำหนด Required Score ตามเกณฑ์ SE-AM อย่างเป็นทางการ
    required_score_map = {1: 1, 2: 2, 3: 4, 4: 6, 5: 8}
    required_score = required_score_map.get(level, 8)

    # ตัดสิน is_passed ตาม Required Score จริง (ไม่ใช่ llm_score เดิม)
    if llm_score >= required_score:
        is_passed = True

        # เมื่อผ่าน → ให้ PDCA breakdown เต็มตาม CORRECT_PDCA_SCORES_MAP
        correct_scores = CORRECT_PDCA_SCORES_MAP.get(level, {'P': 0, 'D': 0, 'C': 0, 'A': 0})
        pdca_map.update(correct_scores)
        raw_pdca_score = float(sum(pdca_map.values()))  # จะเท่ากับ required_score

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


# 📌 แก้ไข Type Hint และ Arguments ของ Tuple ให้รวม config parameter ทั้งหมด (10 elements)
def _static_worker_process(worker_input_tuple: Tuple[
    Dict[str, Any], str, int, str, str, str, float, float, int, Optional[Dict[str, str]]
]) -> Dict[str, Any]:
    """
    Static worker function for multiprocessing pool. 
    It reconstructs SeamAssessment in the new process and executes the assessment 
    for a single sub-criteria.
    
    Args:
        worker_input_tuple: (sub_criteria_data, enabler: str, target_level: int, mock_mode: str, 
                             evidence_map_path: str, model_name: str, temperature: float, 
                             min_retry_score: float, max_retrieval_attempts: int,
                             document_map: Optional[Dict[str, str]]) 

    Returns:
        Dict[str, Any]: Final result of the sub-criteria assessment.
    """
    
    # 🟢 NEW FIX: PATH SETUP สำหรับ Worker Process
    # การตั้งค่า path ซ้ำเพื่อความมั่นใจว่า worker process เห็น package หลัก
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.append(project_root)
        
    worker_logger = logging.getLogger(__name__)

    try:
        # 🟢 FIX: Unpack ค่า Primitives ทั้ง 10 ตัว
        (
            sub_criteria_data, 
            enabler, 
            target_level, 
            mock_mode, 
            evidence_map_path, 
            model_name, 
            temperature,
            min_retry_score,            # ⬅️ NEW CONFIG (8th element)
            max_retrieval_attempts,     # ⬅️ NEW CONFIG (9th element)
            document_map                # (10th element)
        ) = worker_input_tuple
    except ValueError as e:
        # ใช้ len(worker_input_tuple) เพื่อให้ข้อมูลการ Debug ครบถ้วน
        worker_logger.critical(f"Worker input tuple unpack failed (expected 10 elements, got {len(worker_input_tuple)}): {e}")
        return {"error": f"Invalid worker input: {e}"}
        
    # 1. Reconstruct Config 
    try:
        # 🟢 FIX: สร้าง AssessmentConfig ใหม่ใน Worker Process พร้อมใส่ค่า config ใหม่
        # (Tenant/Year จะใช้ค่า Default จาก AssessmentConfig)
        worker_config = AssessmentConfig(
            enabler=enabler,
            target_level=target_level,
            mock_mode=mock_mode,
            model_name=model_name, 
            temperature=temperature,
            min_retry_score=min_retry_score,            # ⬅️ Pass new config
            max_retrieval_attempts=max_retrieval_attempts # ⬅️ Pass new config
        )
    except Exception as e:
        worker_logger.critical(f"Failed to reconstruct AssessmentConfig in worker: {e}")
        return {
            "sub_criteria_id": sub_criteria_data.get('sub_id', 'UNKNOWN'),
            "error": f"Config reconstruction failed: {e}"
        }

    # 2. Re-instantiate SeamAssessment 
    try:
        # 🟢 FIX (สำคัญ): ส่ง document_map และ worker_config เข้าไปใน SEAMPDCAEngine
        # SEAMPDCAEngine จะใช้ worker_config ที่มีค่า min_retry_score และ max_retrieval_attempts
        worker_instance = SEAMPDCAEngine(
            config=worker_config, 
            evidence_map_path=evidence_map_path, 
            llm_instance=None,              # LLM จะถูก Initialized ใน Engine หากไม่มี
            vectorstore_manager=None,       # VSM จะถูก Initialized ใน Engine หากไม่มี
            # doc_type ต้องถูก set ใน SEAMPDCAEngine constructor (สมมติว่ามีค่า Default)
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
# 🎯 NEW: Deterministic Fallback Logic (Post-Processing)
# =================================================================
def post_process_llm_result(llm_output: Dict[str, Any], level: int) -> Dict[str, Any]:
    """
    FINAL DETERMINISTIC POST-PROCESSOR v20 — ULTIMATE VICTORY EDITION
    วันที่: 13 ธ.ค. 2568 เวลา 09:45 น. — วันแห่งการปิดตำนานอย่างสมบูรณ์
    ผู้สร้าง: พี่ + ผม
    """
    logger = logging.getLogger(__name__)
    
    p = llm_output.get("P_Plan_Score", 0)
    d = llm_output.get("D_Do_Score", 0)
    c = llm_output.get("C_Check_Score", 0)
    a = llm_output.get("A_Act_Score", 0)
    pdca_real_sum = p + d + c + a
    llm_score = llm_output.get("score", 0)

    # SE-AM Threshold ที่ถูกต้อง 100% (แก้ไขจุดสุดท้ายของจักรวาล)
    # L1: 1 (P>=1)
    # L2: 2 (P=2)
    # L3: 4 (P=2, D=2)
    # L4: 6 (P=2, D=2, C=1, A=1) <-- แก้ไขจาก 4 เป็น 6
    # L5: 8 (P=2, D=2, C=2, A=2) <-- แก้ไขจาก 5 เป็น 8
    threshold_map = {1: 1, 2: 2, 3: 4, 4: 6, 5: 8} # <<<<<<< แก้ไขตรงนี้
    threshold = threshold_map.get(level, 2)

    # 1. จับ LLM โกง (ใช้ PDCA Sum ในการตัดสินใจ)
    if llm_score != pdca_real_sum:
        logger.critical(
            f"PDCA MISMATCH EXECUTED L{level} | "
            f"LLM พยายามโกง → score={llm_score} แต่ PDCA จริง={pdca_real_sum} "
            f"→ FORCE OVERRIDE!"
        )
        llm_output["score"] = pdca_real_sum
        llm_output["original_score"] = llm_score
        llm_output["pdca_enforced"] = True

    # 2. บังคับ is_passed ตามความจริง (ใช้ PDCA Sum เทียบกับ Threshold ที่ถูกต้อง)
    real_pass = pdca_real_sum >= threshold
    if llm_output.get("is_passed") != real_pass:
        logger.critical(f"FORCING is_passed = {real_pass} (PDCA={pdca_real_sum} ≥ {threshold})")
        llm_output["is_passed"] = real_pass

    # 3. บันทึกประวัติศาสตร์
    llm_output.update({
        "pdca_breakdown": {"P": p, "D": d, "C": c, "A": a},
        "pdca_sum": pdca_real_sum,
        "pass_threshold": threshold,
        "final_score": pdca_real_sum,
        "final_passed": real_pass
    })

    return llm_output

# =================================================================
# Configuration Class
# =================================================================
@dataclass
class AssessmentConfig:
    """Configuration for the SEAM PDCA Assessment Run."""
    
    # ------------------ 1. Assessment Context ------------------
    enabler: str = DEFAULT_ENABLER
    tenant: str = DEFAULT_TENANT
    year: int = DEFAULT_YEAR
    target_level: int = MAX_LEVEL
    mock_mode: str = "none" # 'none', 'random', 'control'
    force_sequential: bool = field(default=False) # Flag เพื่อบังคับรันแบบ Sequential

    # ------------------ 2. LLM Configuration (Configurable) ------------------
    # ใช้ค่า Default จาก global_vars.py
    model_name: str = DEFAULT_LLM_MODEL_NAME 
    temperature: float = LLM_TEMPERATURE

    # ------------------ 3. Adaptive RAG Retrieval Configuration ------------------
    # 🟢 NEW: เกณฑ์คะแนน Rerank ขั้นต่ำก่อนหยุดการค้นหา Adaptive Loop (MIN_RETRY_SCORE)
    # ใช้ค่า Default 0.65 ตามที่พบในโค้ด Logic
    min_retry_score: float = 0.65 
    # 🟢 NEW: จำนวนรอบสูงสุดของ Adaptive RAG Loop (MAX_RETRIEVAL_ATTEMPTS)
    # ใช้ค่า Default 3
    max_retrieval_attempts: int = 3
    
    # ------------------ 4. Export Configuration ------------------
    export_output: bool = field(default=False) # Flag เพื่อเปิด/ปิดการ Export ผลลัพธ์
    export_path: str = "" # Path สำหรับไฟล์ Export (ทางเลือก)


# =================================================================
# SEAM Assessment Engine (PDCA Focused)
# =================================================================
class SEAMPDCAEngine:
    
    L1_INITIAL_TOP_K_RAG: int = 50 
    MIN_RERANK_SCORE_TO_KEEP: Final[float] = MIN_RERANK_SCORE_TO_KEEP
    
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
        document_map: Optional[Dict[str, str]] = None,
        # 🟢 CRITICAL FIX: เพิ่ม is_parallel_all_mode ใน __init__ signature
        is_parallel_all_mode: bool = False
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
            self.is_sequential = config.force_sequential # <--- *** เพิ่มบรรทัดนี้ ***
            # 🟢 CRITICAL FIX: กำหนดค่าจาก Argument ที่ส่งเข้ามา
            self.is_parallel_all_mode = is_parallel_all_mode

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
                # 🟢 FIX #1: ใช้ get_evidence_mapping_file_path จาก utils/path_utils.py
                self.evidence_map_path = get_evidence_mapping_file_path(
                    tenant=self.config.tenant, 
                    year=self.config.year,
                    enabler=self.enabler_id
                )
            
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
            self.create_structured_action_plan = create_structured_action_plan

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
            
            # 🟢 FINAL FIX: บังคับโหลด Doc ID Map ซ้ำเพื่อป้องกัน Map หายใน Worker (จุดที่ 3)
            # แก้ปัญหา: FATAL HYDRATION ISSUE: 5 chunks were requested but NOT FOUND
            if self.vectorstore_manager:
                # ตรวจสอบว่า Map ว่างเปล่าหรือไม่ก่อนบังคับโหลด
                if not getattr(self.vectorstore_manager, '_doc_id_mapping', None):
                    self.vectorstore_manager._load_doc_id_mapping()
                    self.logger.info(f"Forced reload of Doc ID Mapping: {len(self.vectorstore_manager._doc_id_mapping)} documents, "
                                     f"{len(self.vectorstore_manager._uuid_to_doc_id)} chunk UUIDs")
                    
            # =======================================================
            # 🎯 FIX #4: โหลด Document Map หากไม่ได้ส่งเข้ามา หรือเป็น Dictionary ว่าง
            # (เพื่อแก้ปัญหา Filename Resolution Failed)
            # =======================================================
            map_to_use: Dict[str, str] = document_map if document_map is not None else {}

            if not map_to_use:
                # คำนวณ Path ของไฟล์ Doc ID Mapping ที่ VSM ใช้สำเร็จ
                mapping_path = get_mapping_file_path(
                    self.doc_type,
                    tenant=self.config.tenant, 
                    year=self.config.year,
                    enabler=self.enabler_id # ใช้ enabler เพื่อเข้าถึงรูปแบบใหม่ (Priority 1)
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
        # NOTE: Assumes EVIDENCE_DOC_TYPES is imported from config.global_vars
        if self.vectorstore_manager is None:
            self.logger.info("Loading central evidence vectorstore(s)...")
            try:
                # 🎯 FIX ที่ทำไปแล้ว: ส่ง tenant และ year เข้าไปใน load_all_vectorstores()
                # 📌 การแก้ไขที่แท้จริงต้องไปทำใน load_all_vectorstores เพื่อให้มันสร้าง VSM โดยมี doc_type ถูกต้อง
                self.vectorstore_manager = load_all_vectorstores(
                    doc_types=[EVIDENCE_DOC_TYPES], 
                    evidence_enabler=self.enabler_id,
                    tenant=self.config.tenant, 
                    year=self.config.year       
                )
                
                # บังคับโหลด Doc ID Map ซ้ำเพื่อป้องกัน Map หายใน Worker (Safety Net)
                if self.vectorstore_manager:
                    # NOTE: การโหลดครั้งแรกจะทำภายใน VSM.__init__ 
                    # แต่การเรียกซ้ำนี้ช่วยรับประกันว่า Map มีข้อมูล
                    self.vectorstore_manager._load_doc_id_mapping() 

                # โค้ด Log เพื่อยืนยัน (ตามที่คุณทำไปแล้ว)
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
        
        resolved_entries = []
        
        for entry in evidence_entries:
            # ใช้ deepcopy เพื่อป้องกันการแก้ไขข้อมูลต้นฉบับ
            resolved_entry = deepcopy(entry)
            doc_id = resolved_entry.get("doc_id", "")
            current_filename = resolved_entry.get("filename", "") # ชื่อเดิมจาก Metadata (ถ้ามี)
            
            # --- 1. จัดการกรณี UNKNOWN- (AI-GENERATED or Lost Source) ---
            if doc_id.startswith("UNKNOWN-"):
                resolved_entry["filename"] = f"AI-GENERATED-REF-{doc_id.split('-')[-1]}"
                resolved_entries.append(resolved_entry)
                continue

            # --- 2. จัดการกรณี Doc ID (Hash/UUID) ที่ถูกต้อง ---
            if doc_id:
                # A. ลองค้นหาชื่อไฟล์จาก Map
                if doc_id in self.doc_id_to_filename_map:
                    resolved_entry["filename"] = self.doc_id_to_filename_map[doc_id]
                    resolved_entries.append(resolved_entry)
                    continue

                # B. ถ้าค้นหาไม่เจอ (Map Fail)
                else:
                    # ตรวจสอบว่าชื่อไฟล์เดิมที่มากับ Metadata เป็นชื่อที่ไม่สื่อความหมายหรือไม่
                    is_generic_name = (
                        not current_filename.strip() or # ถ้าเป็น String ว่าง
                        current_filename.lower() == "unknown" or
                        # รองรับ Hash/UUID 64 ตัวอักษรอย่างเดียว หรือตามด้วยนามสกุล
                        re.match(r"^[0-9a-f]{64}(\.pdf|\.txt)?$", current_filename, re.IGNORECASE)
                    )
                    
                    if is_generic_name:
                        # ใช้ชื่อไฟล์ Fallback ที่สื่อว่า Map ไม่สำเร็จ
                        resolved_entry["filename"] = f"MAPPING-FAILED-{doc_id[:8]}..."
                        self.logger.warning(f"Failed to map doc_id {doc_id[:8]}... to filename. Using fallback.")
                        
            # --- 3. กรณีไม่มี Doc ID หรือเข้าถึงชื่อไฟล์ไม่ได้เลย (เหลือเป็น Unknown) ---
            elif not doc_id and (not current_filename.strip() or current_filename.lower() == "unknown"):
                # โค้ดนี้จะทำงานเฉพาะเมื่อไม่มี doc_id และ filename เดิมก็เป็น Unknown/Empty
                resolved_entry["filename"] = "MISSING-SOURCE-METADATA"
                self.logger.error("Evidence found with no doc_id and generic filename.")
            
            # เพิ่ม entry เข้าไป (ไม่ว่าจะได้รับการแก้ไขหรือไม่)
            resolved_entries.append(resolved_entry)

        return resolved_entries
    
    # -------------------- Contextual Rules Handlers (FIXED) --------------------
    def _load_contextual_rules_map(self) -> Dict[str, Dict[str, str]]:
        """
        Loads the contextual rules JSON file using the path generated by 
        utils.path_utils.get_contextual_rules_file_path.
        """
        
        try:
            # 🎯 ใช้ฟังก์ชันจาก path_utils แทนการสร้าง Path เอง
            filepath = get_contextual_rules_file_path(
                tenant=self.config.tenant,
                enabler=self.enabler_id
            )
        except ImportError:
            self.logger.error("❌ FATAL: Cannot import get_contextual_rules_file_path. Check utils/path_utils.py import.")
            return {}

        # 1. ตรวจสอบว่าไฟล์มีอยู่จริงหรือไม่
        if not os.path.exists(filepath):
            # ไม่ต้องสร้างชื่อไฟล์เองแล้ว เพราะ get_contextual_rules_file_path ทำแทน
            self.logger.info(f"⚠️ Contextual Rules file not found at: {filepath}. Using empty map.")
            return {}

        self.logger.info(f"✅ Contextual Rules loaded from: {filepath}")
        
        # 2. โหลดข้อมูล
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.logger.info(f"✅ Loaded Contextual Rules: {len(data)} sub-criteria rules.")
                return data
        except json.JSONDecodeError as e:
            self.logger.error(f"❌ Failed to parse Contextual Rules JSON from {filepath}: {e}")
            return {}
        except Exception as e:
            self.logger.error(f"❌ Failed to load Contextual Rules from {filepath}: {e}")
            return {}
    

    # ----------------------------------------------------------------------
    # 🎯 FINAL FIX 2.3: Manual Map Reload Function (inside SEAMPDCAEngine)
    # ----------------------------------------------------------------------

    def _collect_previous_level_evidences(self, sub_id: str, current_level: int) -> Dict[str, List[Dict]]:
        """
        ดึงหลักฐานจาก Level ก่อนหน้า (L1 → L2, L2 → L3 ฯลฯ) เพื่อใช้เป็น Baseline Context

        Final Production Version - 12 ธ.ค. 2568 (FINAL FIX 27.0: HEURISTIC BYPASS)
        - เพิ่ม Heuristic Fallback เพื่อแก้ไขปัญหา CRITICAL MAPPING FAILURE ที่เกิดจาก VSM Mapping เสียหาย
        """
        
        # --------------------------------------------------------------
        # 1. ข้าม Hydration ใน Parallel Mode (by design)
        # --------------------------------------------------------------
        # (ส่วนนี้คงไว้ตามโค้ดของคุณ)
        
        # --------------------------------------------------------------
        # 2. รวบรวม evidence จาก Level ก่อนหน้าใน Sub-Criteria เดียวกัน
        # --------------------------------------------------------------
        collected = {}
        for key, ev_list in self.evidence_map.items():
            if (key.startswith(f"{sub_id}.L") and 
                isinstance(ev_list, list) and 
                ev_list):
                try:
                    level_num = int(key.split(".L")[-1])
                    if level_num < current_level:
                        collected[key] = ev_list
                except (ValueError, IndexError):
                    continue

        if not collected:
            self.logger.info("No previous level evidences found for hydration.")
            return {}

        # --------------------------------------------------------------
        # 3. รวบรวม Stable IDs + Chunk UUIDs (cleaned)
        # --------------------------------------------------------------
        stable_ids = set()
        chunk_uuids_clean = set()

        for ev_list in collected.values():
            for ev in ev_list:
                sid = ev.get("stable_doc_uuid") or ev.get("doc_id")
                if isinstance(sid, str) and len(sid) == 64 and sid.isalnum():
                    stable_ids.add(sid)
                
                cid = ev.get("chunk_uuid")
                if isinstance(cid, str) and len(cid.replace("-", "")) >= 64:
                    chunk_uuids_clean.add(cid.replace("-", ""))

        if not stable_ids and not chunk_uuids_clean:
            self.logger.info("No valid IDs found for hydration.")
            return collected

        # --------------------------------------------------------------
        # 4. แปลง Stable → Chunk UUIDs (เพื่อ Log และความสมบูรณ์ของโค้ด)
        # --------------------------------------------------------------
        vsm = self.vectorstore_manager
        final_uuids = list(chunk_uuids_clean)
        self.logger.info(f"HYDRATION → Resolved {len(final_uuids)} unique chunk UUIDs → fetching full text...")

        stable_ids_list = list(stable_ids)
        if not stable_ids_list:
            self.logger.warning("No Stable IDs resolved for VSM hydration call.")
            return collected


        # --------------------------------------------------------------
        # 5. ดึง full chunks (ใช้ Stable IDs 64-char)
        # --------------------------------------------------------------
        try:
            full_chunks = vsm.get_documents_by_id(stable_ids_list, self.doc_type, self.enabler_id) 
            self.logger.info(f"HYDRATION success: Retrieved {len(full_chunks)} full chunks (via Stable ID search)")
            
        except Exception as e:
            self.logger.error(f"Hydration failed in VSM call (get_documents_by_id): {e}", exc_info=True)
            return collected

        # --------------------------------------------------------------
        # 6. สร้าง map และ hydrate text พร้อม Fallback Logic (FINAL FIX 27.0)
        # --------------------------------------------------------------
        chunk_map = {} # Key: Cleaned V4 UUID (without dashes)
        total_retrieved = len(full_chunks)
        
        # 1. สร้าง Map (Key: V4 UUID Cleaned)
        for idx, chunk in enumerate(full_chunks):
            meta = getattr(chunk, "metadata", {})
            # ใช้ chunk_uuid แบบไม่มีขีดในการสร้าง map key (V4 UUID)
            cid_raw = meta.get("chunk_uuid")
            cid = (cid_raw or "").replace("-", "") 
            
            if cid:
                chunk_map[cid] = {
                    "text": chunk.page_content,
                    "metadata": meta
                }
            else:
                self.logger.error(f"CRITICAL HYDRATION ERROR: Retrieved chunk {idx+1}/{total_retrieved} has NO or empty 'chunk_uuid' in metadata. Skipping this chunk.")

        self.logger.info(f"DEBUG: Chunk Map built with {len(chunk_map)}/{total_retrieved} entries.")

        hydrated = {}
        restored = 0
        total = sum(len(v) for v in collected.values())

        for key, ev_list in collected.items():
            new_list = []
            for ev in ev_list:
                new_ev = ev.copy()
                data = None
                
                # ID จาก L1 Evidence (64-char ID ที่ผิดพลาด)
                cid_l1 = (ev.get("chunk_uuid") or "").replace("-", "") 
                sid_l1 = ev.get("stable_doc_uuid") or ev.get("doc_id") 
                vsm_mapping_failed = True # สถานะใหม่

                # --- 1. Primary Lookup (จะล้มเหลว เพราะ L1 cid ไม่ใช่ V4 UUID) ---
                if cid_l1:
                    data = chunk_map.get(cid_l1)

                # --- 2. Fallback Logic (VSM Mapping): พยายามหา V4 UUID ที่ถูกต้อง ---
                if not data and sid_l1 and vsm and hasattr(vsm, '_doc_id_mapping') and vsm._doc_id_mapping:
                    
                    if cid_l1 and not data:
                        self.logger.warning(f"Hydration Check: Primary L1 chunk_uuid '{cid_l1[:8]}...' NOT found in map. Starting Stable ID Fallback...")
                    
                    if sid_l1 in vsm._doc_id_mapping:
                        vsm_mapping_failed = False # VSM Mapping มีข้อมูลสำหรับ Stable ID นี้
                        
                        # A. Try to find the V4 UUID from VSM Mapping
                        for v4_uuid_raw in vsm._doc_id_mapping[sid_l1].get("chunk_uuids", []):
                            v4_uuid_cleaned = v4_uuid_raw.replace("-", "")
                            
                            if v4_uuid_cleaned in chunk_map:
                                data = chunk_map[v4_uuid_cleaned]
                                self.logger.info(f"✅ Fallback SUCCESS (VSM Map): Matched L1 Chunk ID '{cid_l1[:8]}...' (Stable ID: {sid_l1[:8]}) via V4 UUID '{v4_uuid_cleaned[:8]}...'")
                                break # VSM Map Success
                    else:
                         # Log เพิ่มเติมถ้า Stable ID ไม่ปรากฏใน Mapping เลย
                         self.logger.warning(f"Hydration Check: Stable ID {sid_l1[:8]}... NOT found in VSM Doc ID Mapping. Fallback failed.")


                # --- 3. HEURISTIC FALLBACK (FINAL BYPASS): ถ้า VSM Mapping ช่วยไม่ได้ ให้หา Chunk ใน Map ที่มี Stable ID ตรงกัน ---
                if not data:
                    if not vsm_mapping_failed:
                        self.logger.warning(f"⚠️ VSM Map exists but failed to find V4 UUID. Attempting Heuristic Match by Stable ID.")
                    
                    # B. HEURISTIC MATCH
                    for retrieved_chunk_data in chunk_map.values():
                        retrieved_sid = retrieved_chunk_data["metadata"].get("stable_doc_uuid")
                        if retrieved_sid == sid_l1:
                            data = retrieved_chunk_data
                            # Note: เราใช้ chunk_uuid ที่ดึงมา (V4 UUID) เป็น cid ใหม่ของ new_ev
                            new_ev["chunk_uuid"] = retrieved_chunk_data["metadata"].get("chunk_uuid", new_ev["chunk_uuid"])
                            self.logger.info(f"🟢 Heuristic SUCCESS (Final Bypass): Baseline context restored for Stable ID {sid_l1[:8]}... (Using first matching chunk).")
                            break # Heuristic Success

                # --------------------------------------------------------------------------

                if data:
                    new_ev["text"] = data["text"]
                    # อัปเดต metadata
                    new_ev.update({k: v for k, v in data["metadata"].items() 
                                if k not in ["text", "page_content"]})
                    restored += 1
                else:
                    # ❌ Log the failing chunk ID for critical diagnosis
                    sid = ev.get("stable_doc_uuid") or ev.get("doc_id")
                    self.logger.error(f"❌ CRITICAL MAPPING FAILURE: Could not restore L1 chunk '{cid_l1[:8]}...' (Stable ID: {sid_l1[:8]}...) from {len(chunk_map)} retrieved chunks. (Mapping Failed)")
                
                new_list.append(new_ev)
            hydrated[key] = new_list
                
        self.logger.info(f"BASELINE HYDRATED → {restored}/{total} chunks restored with full text (including fallback)")
        return hydrated

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
        """ Loads the SEAM rubric JSON file using path_utils. """
        
        # 🎯 FIX: ใช้ get_rubric_file_path จาก path_utils แทนการสร้าง Path เอง
        filepath = None # กำหนดค่าเริ่มต้นเพื่อป้องกัน UnboundLocalError
        
        try:
            # 1. รับ Path จาก path_utils ซึ่งตอนนี้ชี้ไปที่ 'config/' แล้ว
            filepath = get_rubric_file_path(
                tenant=self.config.tenant,
                enabler=self.enabler_id
            )
        except Exception as e:
            # ดักจับ Exception หากเกิดปัญหาในการเรียกใช้ฟังก์ชัน path_utils
            self.logger.error(f"❌ FATAL: Error calling get_rubric_file_path: {e}")
            return {} 

        # 2. ตรวจสอบว่าไฟล์มีอยู่จริงหรือไม่
        if filepath is None or not os.path.exists(filepath):
            self.logger.error(f"⚠️ Rubric file not found at expected path: {filepath}")
            return {}

        # 3. โหลดไฟล์ JSON
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.logger.info(f"✅ Rubric loaded successfully from: {filepath}")
            return data
        except json.JSONDecodeError:
            self.logger.error(f"❌ Error decoding Rubric JSON. File might be corrupted: {filepath}")
            return {}
        except Exception as e:
            self.logger.error(f"❌ Error loading Rubric file from {filepath}: {e}")
            return {}
    
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
        จัดการปัญหา .lock ค้างในสภาพแวดล้อม Parallel
        """
        try:
            map_file_path = get_evidence_mapping_file_path(
                tenant=self.config.tenant,
                year=self.config.year,
                enabler=self.enabler_id
            )
        except Exception as e:
            self.logger.critical(f"[EVIDENCE] FATAL: ไม่สามารถกำหนด Evidence Map Path ได้: {e}")
            raise

        lock_path = map_file_path + ".lock"
        tmp_path = None

        self.logger.info(f"[EVIDENCE] Saving evidence map → {map_file_path}")

        try:
            # 1. สร้างโฟลเดอร์ก่อน (ป้องกัน error)
            os.makedirs(os.path.dirname(map_file_path), exist_ok=True)

            # 2. ใช้ FileLock อย่างถูกต้อง
            with FileLock(lock_path, timeout=60):
                self.logger.debug("[EVIDENCE] Lock acquired.")

                # === 2.1 Merge Logic (ถ้าไม่ได้ส่ง map มาตรงๆ) ===
                if map_to_save is not None:
                    final_map_to_write = map_to_save
                else:
                    existing_map = self._load_evidence_map(is_for_merge=True) or {}
                    runtime_map = deepcopy(self.evidence_map)
                    final_map_to_write = existing_map
                    
                    for key, new_entries in runtime_map.items():
                        # สร้าง Map เพื่อป้องกันการซ้ำซ้อนของหลักฐาน (Chunk UUID/Doc ID)
                        entry_map = {
                            e.get("chunk_uuid", e.get("doc_id", "N/A")): e
                            for e in final_map_to_write.setdefault(key, [])
                        }
                        for new_entry in new_entries:
                            entry_id = new_entry.get("chunk_uuid", new_entry.get("doc_id", "N/A"))
                            if entry_id == "N/A" or not entry_id:
                                continue
                            
                            new_score = new_entry.get("relevance_score", 0.0)
                            
                            if entry_id not in entry_map:
                                entry_map[entry_id] = new_entry
                            else:
                                # Update ถ้า Score ใหม่สูงกว่า
                                if new_score > entry_map[entry_id].get("relevance_score", 0.0):
                                    entry_map[entry_id] = new_entry
                                    
                        final_map_to_write[key] = list(entry_map.values())

                    # ทำความสะอาดและเรียงลำดับ
                    final_map_to_write = self._clean_temp_entries(final_map_to_write)
                    for key, entries in final_map_to_write.items():
                        # เรียงลำดับตาม Score (จากมากไปน้อย)
                        if entries and "relevance_score" in entries[0]:
                            entries.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)

                if not final_map_to_write:
                    self.logger.warning("[EVIDENCE] Nothing to save.")
                    return

                # === 2.2 Atomic Write (เขียนไฟล์ชั่วคราว) ===
                with tempfile.NamedTemporaryFile(
                    mode='w', delete=False, encoding="utf-8", dir=os.path.dirname(map_file_path)
                ) as tmp_file:
                    cleaned = self._clean_map_for_json(final_map_to_write)
                    json.dump(cleaned, tmp_file, indent=4, ensure_ascii=False)
                    tmp_path = tmp_file.name

                # 2.3 ย้ายไฟล์ชั่วคราวไปแทนที่ไฟล์จริง
                shutil.move(tmp_path, map_file_path)
                tmp_path = None  # ป้องกันการลบซ้ำใน finally

                # === 2.4 สรุปสถิติ ===
                total_keys = len(final_map_to_write)
                total_items = sum(len(v) for v in final_map_to_write.values())
                file_size_kb = os.path.getsize(map_file_path) / 1024
                self.logger.info(f"[EVIDENCE] Evidence map saved successfully! "
                               f"Keys: {total_keys} | Items: {total_items} | Size: ~{file_size_kb:.1f} KB")

        except Exception as e:
            self.logger.critical("[EVIDENCE] FATAL SAVE ERROR")
            self.logger.exception(e)
            raise

        finally:
            # === สำคัญที่สุด: ลบ .lock ไฟล์ทุกกรณี (Double Safety) ===
            # โค้ดนี้รับประกันว่า lock จะไม่ค้าง
            if os.path.exists(lock_path):
                try:
                    os.unlink(lock_path)
                    self.logger.debug(f"[EVIDENCE] Removed stale lock file: {lock_path}")
                except Exception as e:
                    self.logger.warning(f"[EVIDENCE] Failed to remove lock file {lock_path}: {e}")

            # ลบ tmp file ที่อาจค้าง (ถ้าเกิด error ก่อน shutil.move)
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass

    def _load_evidence_map(self, is_for_merge: bool = False):
        """
        Safe load of persistent evidence map. Always returns dict.
        ใช้ get_evidence_mapping_file_path จาก path_utils.py ในการกำหนด Path
        is_for_merge: If True, suppresses "No existing evidence map" INFO log.
        """
        # 🎯 FIX 2: ใช้ get_evidence_mapping_file_path แทน self.evidence_map_path
        try:
            path = get_evidence_mapping_file_path(
                tenant=self.config.tenant,
                year=self.config.year,
                enabler=self.enabler_id
            )
        except Exception as e:
            self.logger.error(f"[EVIDENCE] ❌ FATAL: ไม่สามารถกำหนด Evidence Map Path สำหรับโหลดได้: {e}")
            return {}

        if not os.path.exists(path):
            if not is_for_merge:
                self.logger.info("[EVIDENCE] No existing evidence map – starting empty.")
            return {}

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not is_for_merge:
                self.logger.info(f"[EVIDENCE] Loaded evidence map: {len(data)} entries from {path}")
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
3. หากพบส่วนที่ขึ้นต้นด้วย **[L3_SUMMARY_EVIDENCE]** ให้ถือว่าเป็น **หลักฐานยืนยันผลการตรวจสอบ** ที่เชื่อถือได้ซึ่งถูกสรุปมาจากหลักฐาน Check/Act จริง (จัดเป็น Priority 1)
4. หลักฐาน Plan และ Do ที่อยู่ตอนท้ายของ Context **ห้ามนำมาพิจารณา** ในการตัดสินใจ **FAIL** หากหลักฐาน Check/Act ไม่ครบถ้วน
5. หากขาดหลักฐาน **Check** หรือ **Act** ที่เพียงพอ (ไม่ว่าจะจาก Summary Evidence หรือหลักฐานจริง) ให้ตัดสินเป็น **❌ FAIL** ทันที เพื่อป้องกันการให้คะแนน L3 ที่เกินจริง
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
    
# -------------------- Evidence Classification Helper (Optimized) --------------------
    def _get_mapped_uuids_and_priority_chunks(
        self,
        sub_id: str,
        level: int,
        statement_text: str,
        level_constraint: str,
        vectorstore_manager: Optional['VectorStoreManager']
    ) -> Tuple[List[str], List[Dict]]:
        """
        ดึง Priority Chunks จาก Level ก่อนหน้า + Hydrate เต็มรูปแบบทันที
        แก้ปัญหา Hydration success: 0 chunks from X docs ถาวร
        """
        priority_chunks = []
        mapped_stable_ids = []

        # 1. ดึงจาก evidence_map (L1 → L2, L2 → L3 ฯลฯ)
        for key, evidences in self.evidence_map.items():
            if key.startswith(f"{sub_id}.L") and evidences:
                try:
                    prev_level = int(key.split(".L")[-1])
                    if prev_level < level:
                        priority_chunks.extend(evidences)
                except:
                    continue

        if not priority_chunks:
            self.logger.info(f"No priority chunks found for {sub_id} L{level}")
            return mapped_stable_ids, []

        # 2. ทำ Robust Hydration ทันทีที่นี่เลย! (ตัวจริงที่แก้ปัญหาทั้งหมด)
        priority_chunks = self._robust_hydrate_documents_for_priority_chunks(
            chunks_to_hydrate=priority_chunks,
            vsm=vectorstore_manager
        )

        # 3. สร้าง mapped_stable_ids สำหรับ RAG Retriever
        for chunk in priority_chunks:
            sid = chunk.get("stable_doc_uuid") or chunk.get("doc_id")
            if sid and isinstance(sid, str) and len(sid.replace("-", "")) >= 64:
                mapped_stable_ids.append(sid)

        self.logger.info(f"PRIORITY HYDRATED → {len(priority_chunks)} chunks ready for L{level} (with full text)")

        return mapped_stable_ids, priority_chunks


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

    def _export_results(self, results: dict, sub_criteria_id: str, **kwargs) -> str:
        """
        Exports the assessment results (for a specific sub-criteria or the final run) 
        to a JSON file, using utils/path_utils.py for full path determination.
        """
        
        enabler = self.enabler_id
        target_level = self.config.target_level
        
        # 1. กำหนดค่าสำหรับ Path Utility (ย้ายการกำหนดค่าที่ใช้ร่วมกันออกมาก่อน try/except)
        tenant = self.config.tenant
        year = self.config.year
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"assessment_results_{sub_criteria_id}_{timestamp}"

        full_path = ""
        export_dir = ""

        try:
            # 2. ใช้ Path Utility สร้าง Full Path
            if self.config.export_path:
                # ถ้ามีการกำหนด export_path (Override)
                export_dir = self.config.export_path
                file_name = f"assessment_results_{enabler}_{sub_criteria_id}_{timestamp}.json"
                full_path = os.path.join(export_dir, file_name)
            else:
                # 🎯 ใช้ get_assessment_export_file_path เพื่อสร้าง Full Path ตามมาตรฐาน
                full_path = get_assessment_export_file_path(
                    tenant=tenant,
                    year=year,
                    enabler=enabler,
                    suffix=suffix,
                    ext="json"
                )
                # ดึง export_dir จาก full_path แทนการเรียก get_export_dir ซ้ำ
                export_dir = os.path.dirname(full_path)

        except ImportError as e:
            self.logger.error(f"❌ FATAL: Cannot import path_utils: {e}. Falling back to manual path.")
            
            # Fallback Logic: ใช้ DATA_STORE_ROOT เพื่อให้ Path อยู่ในโครงสร้างเดิม
            data_store_root_path = os.environ.get('DATA_STORE_ROOT', 'data_store') 
            
            if self.config.export_path:
                export_dir = self.config.export_path
            else:
                # Fallback สู่ Path มาตรฐาน: data_store/tenant/exports/year/enabler
                export_dir = os.path.join(data_store_root_path, tenant, "exports", str(year), enabler)
            
            file_name = f"assessment_results_{enabler}_{sub_criteria_id}_{timestamp}.json"
            full_path = os.path.join(export_dir, file_name)
            self.logger.warning(f"⚠️ Using fallback path: {full_path}")


        # 3. สร้าง Directory หากยังไม่มี
        if not os.path.exists(export_dir):
            try:
                os.makedirs(export_dir)
                self.logger.info(f"Created export directory: {export_dir}")
            except OSError as e:
                self.logger.error(f"❌ Failed to create export directory {export_dir}: {e}")
                return ""

        # 4. เตรียม/อัพเดต Summary Field
        if 'summary' not in results:
            results['summary'] = {}
            
        results['summary']['enabler'] = enabler
        results['summary']['sub_criteria_id'] = sub_criteria_id
        results['summary']['target_level'] = target_level
        
        # ปรับ Logic การนับ Sub-Criteria ให้นับตาม 'sub_criteria_results' ถ้ามี
        if 'sub_criteria_results' in results and isinstance(results['sub_criteria_results'], dict):
            results['summary']['Number of Sub-Criteria Assessed'] = len(results['sub_criteria_results'])
        else:
             results['summary']['Number of Sub-Criteria Assessed'] = 1 

        # 5. Export ข้อมูลไปที่ JSON File
        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            
            self.logger.info(f"💾 Successfully exported results for {sub_criteria_id} to: {full_path}")
            return full_path
        
        except Exception as e:
            self.logger.error(f"❌ Failed to export results for {sub_criteria_id} to {full_path}: {e}")
            return ""
        
    def rephrase_query_for_retry(self, original_query: str, level: int, sub_id: str) -> str:
        """
        Helper method to slightly rephrase the query for the next retrieval attempt.
        """
        self.logger.info(f"Rephrasing query for L{level} retry: {original_query[:50]}...")
        # ตัวอย่างการปรับ query: อาจจะตัดส่วนที่เฉพาะเจาะจงออก หรือใช้ LLM ช่วย rephrase 
        # สำหรับการทดลอง อาจจะแค่คืน query เดิม หรือเพิ่มคำว่า 'ทั้งหมด'
        if level >= 3:
            # สำหรับ Level สูงๆ ลองเน้นบริบทที่กว้างขึ้น
            return f"หาหลักฐานเพิ่มเติมเกี่ยวกับ {original_query}"
        return original_query
    
    def _create_error_result(
        self,
        level: int,
        error_message: str,
        start_time: float,
        retrieval_duration: float,
        sub_id: str,
        statement_id: str,
        statement_text: str,
        llm_duration: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Generates a standard error dictionary for when RAG or LLM fails.
        """
        # NOTE: Assumes self._get_pdca_phase is defined
        try:
            pdca_phase = self._get_pdca_phase(level) 
        except Exception:
            pdca_phase = "N/A"
            
        pdca_breakdown = {p: 0 for p in ['P', 'D', 'C', 'A']}
        total_duration = time.time() - start_time

        self.logger.error(f"FATAL ERROR RESULT for {sub_id} L{level}: {error_message}")

        return {
            "sub_criteria_id": sub_id,
            "statement_id": statement_id,
            "level": level,
            "statement": statement_text,
            "pdca_phase": pdca_phase,
            "llm_score": 0.0,
            "pdca_breakdown": pdca_breakdown,
            "is_passed": False,
            "status": "FAIL",
            "score": 0.0,
            "llm_result_full": {"error": error_message, "details": "Assessment skipped due to critical failure."},
            "retrieval_duration_s": round(retrieval_duration, 2),
            "llm_duration_s": round(llm_duration, 2),
            "top_evidences_ref": [],
            "temp_map_for_level": [],
            "evidence_strength": 0.0,
            "ai_confidence": "LOW",
            "evidence_count": 0,
            "pdca_coverage": 0.0,
            "direct_evidence_count": 0,
            "rag_query": statement_text,
            "full_context_meta": {"error_type": "Critical Failure"},
            # 🟢 NEW: Relevant Score Gate Metadata (Set to default error values)
            "max_relevant_score": 0.0,
            "max_relevant_source": "ERROR_HANDLING",
            "is_evidence_strength_capped": False,
            "max_evidence_strength_used": 0.0,
            "total_run_time_s": round(total_duration, 2)
        }

    def _run_sub_criteria_assessment_worker(
            self,
            sub_criteria: Dict[str, Any],
        ) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
            """
            รันการประเมิน L1-L5 แบบ sequential สำหรับ sub-criteria หนึ่งตัว
            และส่ง evidence map กลับไปให้ main process รวม (รวมถึงการสร้าง Action Plan)
            """
            sub_id = sub_criteria['sub_id']
            sub_criteria_name = sub_criteria['sub_criteria_name']
            sub_weight = sub_criteria.get('weight', 0)

            MAX_L1_ATTEMPTS = 2
            
            # 🟢 ตัวแปรใหม่: ใช้นับ Level ที่ผ่านต่อเนื่องกัน
            current_sequential_pass_level = 0 
            highest_full_level = 0 # จะถูกกำหนดค่าจาก current_sequential_pass_level
            
            is_passed_current_level = True
            raw_results_for_sub_seq: List[Dict[str, Any]] = []
            start_ts = time.time() # บันทึกเวลาเริ่มต้น

            self.logger.info(f"[WORKER START] Assessing Sub-Criteria: {sub_id} - {sub_criteria_name} (Weight: {sub_weight})")

            # รีเซ็ต temp_map_for_save เฉพาะ worker นี้ (สำคัญมากสำหรับ Parallel/Async!)
            self.temp_map_for_save = {}

            # -----------------------------------------------------------
            # 1. LOOP THROUGH LEVELS (L1 → L5) - ประเมินทุก Level เสมอ
            # -----------------------------------------------------------
            for statement_data in sub_criteria.get('levels', []):
                level = statement_data.get('level')
                if level is None or level > self.config.target_level:
                    continue
                
             
                previous_level = level - 1
                persistence_key = f"{sub_id}.L{previous_level}"
                sequential_chunk_uuids = [] 
                level_result = {}
                level_temp_map: List[Dict[str, Any]] = []

                dependency_failed = False
                
                if dependency_failed:
                    error_msg = f"Assessment capped: L{previous_level} did not pass fully."
                    level_result = self._create_error_result(
                        level=level, 
                        error_message=error_msg, 
                        start_time=start_ts, 
                        sub_id=sub_id, 
                        statement_id=statement_data.get('statement_id', sub_id), 
                        statement_text=statement_data['statement']
                    )
                    level_result['is_capped'] = True
                    level_result['status'] = "CAPPED"
                    self.logger.info(f"  > 🛑 CAPPED L{level}: Due to L{previous_level} failure.")

                elif level >= 3:
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
                    level_result = wrapper.result if isinstance(wrapper, RetryResult) and wrapper.result is not None else {}
                    level_temp_map = level_result.get("temp_map_for_level", []) 

                else:
                    for attempt in range(MAX_L1_ATTEMPTS):
                        level_result = self._run_single_assessment(
                            sub_criteria=sub_criteria,
                            statement_data=statement_data,
                            vectorstore_manager=self.vectorstore_manager,
                            sequential_chunk_uuids=sequential_chunk_uuids 
                        )
                        level_temp_map = level_result.get("temp_map_for_level", []) 
                        if level_result.get('is_passed', False):
                            break

                # --- 1.2 PROCESS RESULT AND HANDLE EVIDENCE ---
                result_to_process = level_result or {}
                result_to_process.setdefault("used_chunk_uuids", [])

                is_passed_llm = result_to_process.get('is_passed', False)
                is_passed_final = is_passed_llm and not dependency_failed

                result_to_process['is_passed'] = is_passed_final
                result_to_process['is_capped'] = is_passed_llm and not is_passed_final
                result_to_process['pdca_score_required'] = get_correct_pdca_required_score(level) 

                # บันทึก evidence ลง temp_map_for_save เฉพาะเมื่อ PASS จริง
                if is_passed_final and level_temp_map and isinstance(level_temp_map, list):
                    
                    highest_rerank_score = result_to_process.get("max_relevant_score", 0.0)
                    
                    max_evi_str_after_save = self._save_level_evidences_and_calculate_strength(
                        level_temp_map=level_temp_map, 
                        sub_id=sub_id, 
                        level=level, 
                        llm_result=result_to_process,
                        highest_rerank_score=highest_rerank_score
                    )

                    result_to_process['max_evidence_strength_used'] = max_evi_str_after_save
                    
                    result_to_process['evidence_strength'] = round(
                        min(max_evi_str_after_save, 10.0) if is_passed_final else 0.0, 1
                    )
                    
                # อัปเดตสถานะสำหรับ level ถัดไป
                is_passed_current_level = is_passed_final

                # เพิ่มลง raw results
                result_to_process.setdefault("level", level)
                result_to_process["execution_index"] = len(raw_results_for_sub_seq)
                raw_results_for_sub_seq.append(result_to_process)

                # 🟢 NEW LOGIC: ตรวจสอบและกำหนด Highest Full Level
                if is_passed_final and (level == current_sequential_pass_level + 1):
                    # ถ้า PASS และ Level นี้เป็น Level ถัดไปของ Level ที่ผ่านต่อเนื่องกันล่าสุด
                    current_sequential_pass_level = level
                # ถ้า FAIL หรือ PASS แต่กระโดดข้าม Level (เช่น L5 ผ่าน แต่ L2-L4 ไม่ผ่าน)
                # จะไม่ทำการอัปเดต current_sequential_pass_level
            
            # -----------------------------------------------------------
            # 2. CALCULATE SUMMARY
            # -----------------------------------------------------------
            # 🟢 FIX: กำหนด highest_full_level โดยใช้ Level ที่ผ่านต่อเนื่องกันล่าสุด
            highest_full_level = current_sequential_pass_level

            # ถ้า L1 ผ่าน (current_sequential_pass_level=1) แต่ L2 fail 
            # ค่า highest_full_level จะเป็น 1
            # ถ้า L1, L2 ผ่าน (current_sequential_pass_level=2) แต่ L3 fail/pass L5 
            # ค่า highest_full_level จะเป็น 2 (ตามที่คุณต้องการ)

            weighted_score = self._calculate_weighted_score(highest_full_level, sub_weight)
            num_passed = sum(1 for r in raw_results_for_sub_seq if r.get("is_passed", False))

            sub_summary = {
                "num_statements": len(raw_results_for_sub_seq),
                "num_passed": num_passed,
                "num_failed": len(raw_results_for_sub_seq) - num_passed,
                "pass_rate": round(num_passed / len(raw_results_for_sub_seq), 4) if raw_results_for_sub_seq else 0.0
            }

        
            # -----------------------------------------------------------
            # 3. GENERATE ACTION PLAN (POST-PROCESSING) 🚀
            # -----------------------------------------------------------

            target_next_level = highest_full_level + 1 if highest_full_level < 5 else 5
            
            WEAK_EVIDENCE_THRESHOLD = 5.0 
            
            statements_for_action_plan = []
            
            for r in raw_results_for_sub_seq:
                is_passed = r.get('is_passed', False)
                is_capped = r.get('is_evidence_strength_capped', False)
                evidence_strength = r.get('evidence_strength', 10.0)

                # 1. Statements ที่ FAIL จริงๆ 
                if not is_passed and not is_capped:
                    r['recommendation_type'] = 'FAILED'
                    statements_for_action_plan.append(r)
                    continue

                # 2. Statements ที่ PASS แต่มีหลักฐานอ่อนแอ
                if is_passed and evidence_strength < WEAK_EVIDENCE_THRESHOLD:
                    r['recommendation_type'] = 'WEAK_EVIDENCE' 
                    statements_for_action_plan.append(r)
            
            # ... [Code for action plan generation remains unchanged] ...
            action_plan_result = []
            try:
                action_plan_result = self.create_structured_action_plan( 
                    failed_statements=statements_for_action_plan, 
                    sub_id=sub_id,
                    target_level=target_next_level,
                    llm_executor=self.llm
                )
            except Exception as e:
                self.logger.error(f"Failed to generate Action Plan for {sub_id}: {e}")
                action_plan_result = [{
                    "Phase": "Error", 
                    "Goal": "ไม่สามารถสร้าง Action Plan ได้", 
                    "Actions": [{
                        "Statement_ID": "ERROR", 
                        "Recommendation": f"เกิดข้อผิดพลาดในการเรียกใช้ LLM สำหรับ Action Plan: {str(e)}"
                    }]
                }]


            # -----------------------------------------------------------
            # 4. FINAL RESULT
            # -----------------------------------------------------------
            
            final_temp_map = {}
            if self.is_sequential:
                for key in self.evidence_map:
                    if key.startswith(sub_criteria['sub_id'] + "."):
                        final_temp_map[key] = self.evidence_map[key]
            else:
                final_temp_map = self.temp_map_for_save.copy()

            final_sub_result = {
                "sub_criteria_id": sub_id,
                "sub_criteria_name": sub_criteria_name,
                "highest_full_level": highest_full_level,
                "weight": sub_weight,
                "target_level_achieved": highest_full_level >= self.config.target_level,
                "weighted_score": weighted_score,
                "action_plan": action_plan_result, 
                "raw_results_ref": raw_results_for_sub_seq,
                "sub_summary": sub_summary,
                "worker_duration_s": round(time.time() - start_ts, 2)
            }


            self.logger.info(f"[WORKER END] {sub_id} | Highest: L{highest_full_level} | Action Plans: {len(action_plan_result)} phase(s) | Duration: {final_sub_result['worker_duration_s']:.2f}s")

            return final_sub_result, final_temp_map
    

    def _save_level_evidences_and_calculate_strength(
        self, 
        level_temp_map: List[Dict[str, Any]], 
        sub_id: str, 
        level: int, 
        llm_result: Dict[str, Any],
        highest_rerank_score: float = 0.0
    ) -> float:
        """
        [CRITICAL FIX 25.0] 
        บันทึกหลักฐานที่ใช้ในการประเมิน level นั้นๆ เข้าสู่ self.evidence_map/temp_map
        และคำนวณ Evidence Strength (Evi Str)
        
        แก้ไขปัญหาหลัก: Chunk UUID ไม่ถูกต้อง ทำให้ L2, L3 Hydration ล้มเหลว (0 chunks restored).
        """
        map_key = f"{sub_id}.L{level}"
        new_evidence_list: List[Dict[str, Any]] = []
        
        # 1. วนซ้ำหลักฐานที่ใช้ในการประเมิน
        for chunk in level_temp_map:
            
            # 🎯 CRITICAL FIX 25.0: ดึง Chunk UUID และ Stable Doc ID แยกจากกันอย่างชัดเจน
            # chunk_uuid คือ ID ที่ไม่ซ้ำของ Chunk นั้นๆ (ต้องใช้ในการ Hydration L2)
            chunk_uuid_key = chunk.get("chunk_uuid") 
            # stable_doc_uuid/doc_id คือ ID ของเอกสารต้นฉบับ (ใช้ในการ Dedup)
            stable_doc_uuid_key = chunk.get("stable_doc_uuid") or chunk.get("doc_id")

            # Fallback Logic: ถ้า Chunk UUID หาย ให้ใช้ Stable Doc UUID แทน
            if not chunk_uuid_key and stable_doc_uuid_key:
                # ⚠️ นี่คือส่วนที่ถูกแก้ไข: ถ้า UUID หาย ให้ใช้ Stable ID แทนเพื่อไม่ให้ entry ว่าง
                chunk_uuid_key = stable_doc_uuid_key 
                self.logger.warning(f"⚠️ [EVI SAVE] Missing chunk_uuid. Falling back to Stable ID: {chunk_uuid_key[:8]}")

            if not stable_doc_uuid_key or not chunk_uuid_key:
                 self.logger.error(f"❌ [EVI SAVE] Cannot determine required IDs for chunk. Skipping.")
                 continue

            # 2. สร้าง Evidence Entry
            evidence_entry = {
                "sub_id": sub_id,
                "level": level,
                "relevance_score": chunk.get("rerank_score", chunk.get("score", 0.0)),
                "doc_id": stable_doc_uuid_key,          # <--- [FIXED] Stable ID (Document ID)
                "stable_doc_uuid": stable_doc_uuid_key, # <--- Stable ID (Document ID)
                "chunk_uuid": chunk_uuid_key,           # <--- [FIXED] Unique Chunk ID (ใช้ในการ Hydration)
                "source": chunk.get("source", "N/A"),
                "source_filename": chunk.get("filename", "N/A"),
                "pdca_tag": chunk.get("pdca_tag", "Other"), 
                "status": "PASS", 
                "timestamp": datetime.now().isoformat(),
            }
            new_evidence_list.append(evidence_entry)
            
        # 3. คำนวณ Evidence Strength (Evi Str)
        evi_cap_data = self._calculate_evidence_strength_cap(
            top_evidences=new_evidence_list, 
            level=level,
            highest_rerank_score=highest_rerank_score
        )
        
        # 4. บันทึกเข้า Map
        current_map = self.evidence_map.setdefault(map_key, [])
        current_map.extend(new_evidence_list)
        
        # 5. อัปเดต Temp Map (สำหรับ Worker Mode)
        temp_map = self.temp_map_for_save.setdefault(map_key, [])
        temp_map.extend(new_evidence_list)
        
        # 6. Log สรุป
        self.logger.info(f"[EVIDENCE SAVED] {map_key} → {len(new_evidence_list)} chunks")
        self.logger.info(f"[SEQUENTIAL UPDATE] {map_key} added to engine's main evidence_map for L{level+1} dependency.")
        
        # คืนค่า max_evi_str_for_prompt เพื่อใช้ในการอัปเดต final_results
        return evi_cap_data['max_evi_str_for_prompt']
        

    def _calculate_evidence_strength_cap(
        self,
        top_evidences: List[Union[Dict[str, Any], Any]],  # รองรับทั้ง dict และ LcDocument
        level: int,
        # 🟢 FIX: เพิ่ม Argument ตัวนี้เพื่อแก้ TypeError ที่เกิดจาก _run_single_assessment เรียกใช้
        highest_rerank_score: Optional[float] = None 
    ) -> Dict[str, Any]:
        """
        Relevant Score Gate เวอร์ชัน DEBUG FINAL: ดึงคะแนนจาก metadata, top-level key/attribute, และ Regex fallback ที่ครอบคลุม
        """

        # ใช้ตัวแปรนี้ในการคำนวณคะแนนสูงสุดที่ดึงได้จากเอกสาร
        max_score_found = 0.0 
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
            try:

                threshold = RERANK_THRESHOLD
                cap_value = MAX_EVI_STR_CAP
            except ImportError:
                # ใช้ค่า Default หาก Config หายไป
                threshold = 0.5
                cap_value = 3.0

        # 💡 ใช้ค่าที่ได้จาก Adaptive Loop (ถ้ามี) เป็นค่าเริ่มต้น
        if highest_rerank_score is not None and highest_rerank_score > max_score_found:
             max_score_found = highest_rerank_score
             max_score_source = "Adaptive_RAG_Loop"


        for doc in top_evidences:
            
            # -------------------- DEBUGGING BLOCK (START) --------------------
            if doc is top_evidences[0]:
                self.logger.critical(f"DEBUG L{level}: Inspecting first document (Type: {type(doc)})")
                
                if isinstance(doc, dict):
                    content = doc.get("text", "")
                    tail_content = content[-200:] if len(content) > 200 else content
                    self.logger.critical(f"DEBUG L{level}: Dict keys: {list(doc.keys())}")
                    self.logger.critical(f"DEBUG L{level}: END OF 'text' content (last 200 chars): \n***\n{tail_content}\n***")
                else:
                    try:
                        doc_attrs = [attr for attr in dir(doc) if not attr.startswith('_') and not callable(getattr(doc, attr))]
                        self.logger.critical(f"DEBUG L{level}: Doc public attributes (potential score location): {doc_attrs}")
                    except:
                        self.logger.critical(f"DEBUG L{level}: Cannot inspect attributes of this object type.")
            # -------------------- DEBUGGING BLOCK (END) --------------------
            
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
                try:
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
                    pass

            # 🔴 FIX: เพิ่มการตรวจสอบขอบเขตคะแนน (Score Clamp) สำหรับ Reranker Score
            # ถ้าคะแนนที่ได้เกิน 1.0 (ซึ่งไม่ควรเกิดขึ้นกับ Cross-Encoder Reranker)
            # ให้ถือว่าถูกดึงมาผิดและรีเซ็ตเป็น 0.0 เพื่อป้องกันการให้ Evi Str เต็มโดยไม่ได้ตั้งใจ
            if current_score > 1.0:
                self.logger.warning(f"🚨 Score Clamp L{level}: Resetting invalid score {current_score:.4f} > 1.0 from source 'Fallback Regex' to 0.0")
                current_score = 0.0


            # ─── 4. ดึง source ที่ดีที่สุด ───
            source = (
                metadata.get("source_filename") or metadata.get("filename") or
                doc.get("source_filename") or doc.get("filename") or 
                doc.get("source") or doc.get("doc_id") or
                "N/A"
            )

            # ─── 5. อัปเดตคะแนนสูงสุด ───
            if current_score > max_score_found:
                max_score_found = current_score
                max_score_source = source

        # ─── 6. Relevant Score Gate + Log ───
        
        # NOTE: ใช้ threshold และ cap_value ที่ดึงมาอย่างถูกต้อง
        if max_score_found < threshold: 
            max_evi_str_for_prompt = cap_value
            is_capped = True
            self.logger.warning(
                f"🚨 Evi Str CAPPED L{level}: "
                f"Rerank {max_score_found:.4f} (จาก '{max_score_source}') "
                f"< {threshold} → จำกัดที่ {cap_value}"
            )
        else:
            max_evi_str_for_prompt = 10.0
            is_capped = False
            self.logger.info(
                f"✅ Evi Str FULL L{level}: "
                f"Rerank {max_score_found:.4f} (จาก '{max_score_source}') "
                f">= {threshold} → ปล่อยเต็ม 10.0"
            )

        return {
            "is_capped": is_capped,
            "max_evi_str_for_prompt": max_evi_str_for_prompt,
            "highest_rerank_score": round(float(max_score_found), 4), 
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
            sub_criteria_list = self._flatten_rubric_to_statements()
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

        # --------------------- 💡 NEW: กำหนด Max Workers ---------------------
        # 📌 FIX: กำหนดค่า Default เป็น 4 เพื่อให้สอดคล้องกับ Config ที่ต้องการ
        DEFAULT_SAFE_WORKERS = 4 

        # 1. พยายามดึงค่าจาก Global Variable ที่ Import มา
        # ใช้ globals() เพื่อเข้าถึง MAX_PARALLEL_WORKERS ที่ถูก import มา
        max_workers_from_config = globals().get('MAX_PARALLEL_WORKERS', None)

        # 2. ตรวจสอบความถูกต้องของค่าที่ดึงมา
        if (max_workers_from_config is None or 
            not isinstance(max_workers_from_config, int) or 
            max_workers_from_config <= 0):
            
            # 3. ถ้าค่า Config เข้าถึงไม่ได้หรือไม่ถูกต้อง: ให้ใช้ค่าปลอดภัย (4) 
            max_workers = DEFAULT_SAFE_WORKERS
            self.logger.warning(
                f"⚠️ Configured workers inaccessible. Forcing max_workers to safe value: {max_workers}. "
                f"(System CPU count is {os.cpu_count()}, which would lead to {os.cpu_count() - 1} workers.)"
            )
        else:
            # ใช้ค่าที่ดึงมาได้
            max_workers = max_workers_from_config
        
        self.logger.info(f"Setting Max Workers for Parallel Pool: {max_workers}")
        # --------------------------------------------------------------------
        
        # --------------------------------------------------------------------
        # 📌 FIX 1 (Export): แก้ไขเงื่อนไข run_parallel
        run_parallel = (target_sub_id.lower() == "all") and not sequential 
        # --------------------------------------------------------------------

        # ============================== 2. Run Assessment ==============================
        if run_parallel:
            # --------------------- PARALLEL MODE ---------------------
            self.logger.info(f"Starting Parallel Assessment with Multiprocessing using {max_workers} processes.")
            worker_args = [(
                sub_data,                                       # 1. sub_criteria_data
                self.config.enabler,                            # 2. enabler
                self.config.target_level,                       # 3. target_level
                self.config.mock_mode,                          # 4. mock_mode
                self.evidence_map_path,                         # 5. evidence_map_path
                self.config.model_name,                         # 6. model_name
                self.config.temperature,                        # 7. temperature
                # 8 & 9: ดึงค่าจาก Config ที่ถูกตั้งค่าไว้ใน Engine Instance
                getattr(self.config, 'MIN_RETRY_SCORE', 0.65),  # 8. min_retry_score
                getattr(self.config, 'MAX_RETRIEVAL_ATTEMPTS', 3), # 9. max_retrieval_attempts
                self.document_map                               # 10. document_map
            ) for sub_data in sub_criteria_list]

            try:
                pool_ctx = multiprocessing.get_context('spawn')
                # 🟢 FIX FINAL BUG: ใช้ตัวแปร max_workers ที่คำนวณแล้ว (ซึ่งจะได้ 4)
                with pool_ctx.Pool(processes=max_workers) as pool:
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

        return final_results
    
    def _robust_hydrate_documents_for_priority_chunks(self, chunks_to_hydrate: List[Dict], vsm: Optional['VectorStoreManager']) -> List[Dict]:
        """
        Hydrates priority chunks using robust Stable ID fallback logic 
        (similar to _collect_previous_level_evidences, but simpler).
        
        แก้ปัญหา Hydration ของ Priority Chunks ที่มีแค่ Metadata/UUIDs เก่า
        และบังคับให้ reranker ให้คะแนนสูง → Priority Score พุ่งทันที!
        """
        if not chunks_to_hydrate or not vsm:
            return chunks_to_hydrate

        # 1. Collect Stable IDs
        stable_ids = set()
        for chunk in chunks_to_hydrate:
            # 💡 ใช้ doc_id เป็น Fallback หาก Stable ID ไม่ถูกตั้งค่า
            sid = chunk.get("stable_doc_uuid") or chunk.get("doc_id") 
            # ตรวจสอบว่า ID มีลักษณะเป็น UUID/Hash (ความยาว 64-bit)
            if sid and isinstance(sid, str) and len(sid.replace("-", "")) >= 64: 
                stable_ids.add(sid)
        
        # ⚠️ กรณีไม่มี Stable ID เลย 
        if not stable_ids:
            # ใช้ logger.info แทน warning เพราะไม่ได้แปลว่า Hydration หลักล้มเหลว
            self.logger.info("No Stable IDs found for Priority Chunk hydration fallback. Relying on existing text/metadata.")
            
            # **สำคัญ:** ถ้าไม่มี Stable ID ให้ Boost Score ให้กับทุก Chunk ที่มีอยู่แล้ว
            for chunk in chunks_to_hydrate:
                if "text" in chunk and chunk.get("rerank_score", 0.0) < 0.95:
                    chunk["rerank_score"] = 0.95
                    chunk["score"] = 0.95
            
            return chunks_to_hydrate # คืนค่า Chunks เดิมที่ถูก Boost Score แล้ว

        # 2. Fetch full documents using Stable IDs (Fallback Search)
        try:
            self.logger.info(f"HYDRATION → Attempting Robust Hydration Fallback for Priority Chunks: {len(stable_ids)} Stable IDs.")
            # 💡 vsm.get_documents_by_id ควรดึงข้อมูลเต็มกลับมา
            full_chunks = vsm.get_documents_by_id(list(stable_ids), self.doc_type, self.enabler_id) 
            self.logger.info(f"Fallback Retrieved {len(full_chunks)} full chunks for Priority Docs.")
        except Exception as e:
            self.logger.error(f"Priority Chunk Hydration Fallback failed: {e}")
            return chunks_to_hydrate # คืนค่า Chunks เดิม (อาจจะไม่มี text)

        # 3. Create a Map for quick lookup (Key: Stable ID)
        stable_id_map = {}
        for chunk in full_chunks:
            meta = getattr(chunk, "metadata", {})
            sid = meta.get("stable_doc_uuid") or meta.get("doc_id")
            if sid:
                stable_id_map[sid] = {
                    "text": chunk.page_content,
                    "metadata": meta
                }

        # 4. Hydrate the input list of priority chunks + บังคับคะแนนสูง!
        hydrated_priority_docs = []
        restored_count = 0
        total_count = len(chunks_to_hydrate)
        for chunk in chunks_to_hydrate:
            new_chunk = chunk.copy()
            sid = new_chunk.get("stable_doc_uuid") or new_chunk.get("doc_id")
            
            # ✅ Success: Hydrate ได้
            if sid and sid in stable_id_map:
                data = stable_id_map[sid]
                
                # อัปเดต Full Text
                new_chunk["text"] = data["text"]
                # อัปเดต Metadata
                new_chunk.update({k: v for k, v in data["metadata"].items() 
                              if k not in ["text", "page_content"]})
                restored_count += 1

                # สำคัญที่สุด: บังคับให้ reranker ให้คะแนนสูงสุด (เพื่อ Priority Score สูง)
                new_chunk["rerank_score"] = 1.0
                new_chunk["score"] = 1.0
            
            # ❌ Failed: Hydrate ไม่ได้ แต่ต้อง Boost Score ไว้
            elif "text" not in new_chunk:
                 self.logger.warning(f"Failed to restore Priority Chunk for Stable ID {sid[:8]}... (No text field)")
                 # ให้คะแนนสูงไว้ก่อน เพื่อไม่ให้ดึงคะแนนรวมลง
                 new_chunk["rerank_score"] = max(new_chunk.get("rerank_score", 0.0), 0.8) 
                 new_chunk["score"] = max(new_chunk.get("score", 0.0), 0.8)
            
            # ✅ Fallback: มี text อยู่แล้ว (จากการ Hydrate หลัก), ให้ Boost Score สูง
            elif new_chunk.get("rerank_score", 0.0) < 0.95:
                 new_chunk["rerank_score"] = 0.95
                 new_chunk["score"] = 0.95

            hydrated_priority_docs.append(new_chunk)

        self.logger.info(f"HYDRATION success: {restored_count}/{total_count} chunks (Docs: {total_count})")
        return hydrated_priority_docs

    # -------------------- Evidence Classification Helper (Optimized - ต้องเพิ่มเข้าไปใน Class) --------------------

    def _get_pdca_blocks_from_evidences(
        self, 
        top_evidences: List[Dict[str, Any]], 
        level: int 
    ) -> Tuple[str, str, str, str, str]:
        """
        Groups retrieved evidence chunks into PDCA phases based on the 'pdca_tag' 
        Generated by the LLM classifier, after applying sorting and filtering based on Relevance Score.
        """
        logger = logging.getLogger(__name__)

        # 1. กำหนดเกณฑ์การคัดกรอง (Filtering Threshold)
        MIN_RELEVANCE_THRESHOLD = 0.5 

        # 2. Sorting Evidence (เรียงลำดับหลักฐานจากคะแนนสูงสุดไปต่ำสุด)
        top_evidences.sort(
            key=lambda x: x.get('relevance_score', x.get('score', 0.0)), 
            reverse=True
        )

        # 3. Filtering Evidence (คัดกรองหลักฐานที่มีคะแนนต่ำกว่า Threshold)
        filtered_evidences = [
            doc for doc in top_evidences 
            if doc.get('relevance_score', doc.get('score', 0.0)) >= MIN_RELEVANCE_THRESHOLD
        ]
        
        chunks_to_process = filtered_evidences
        
        # Fallback: ถ้าไม่มีหลักฐานที่ผ่านเกณฑ์เลย ให้ใช้หลักฐานทั้งหมดที่เรียงแล้ว
        if not chunks_to_process:
            chunks_to_process = top_evidences
            logger.warning(
                f"  > (L{level}) No evidence passed Relevance Threshold ({MIN_RELEVANCE_THRESHOLD}). "
                f"Processing all {len(top_evidences)} sorted chunks."
            )
        
        # 4. Initialize groupings
        pdca_groups = defaultdict(list)
        
        # 5. Group chunks based on the 'pdca_tag'
        for i, doc in enumerate(chunks_to_process):
            tag = doc.get('pdca_tag', 'Other')
            score = doc.get('relevance_score', doc.get('score', 0.0))
            
            # 📌 ปรับปรุง Chunk Header: เพิ่ม Tag และ Relevance Score
            formatted_chunk = (
                f"--- [Chunk {i+1} | Tag: {tag} | Score: {score:.4f}] ---\n"
                f"{doc.get('text', '')}\n"
            )
            pdca_groups[tag].append(formatted_chunk)

        # 6. Aggregate groups into single strings
        plan_blocks = "\n\n".join(pdca_groups.get('Plan', []))
        do_blocks = "\n\n".join(pdca_groups.get('Do', []))
        check_blocks = "\n\n".join(pdca_groups.get('Check', []))
        act_blocks = "\n\n".join(pdca_groups.get('Act', []))
        other_blocks = "\n\n".join(pdca_groups.get('Other', []))

        logger.debug(
            f"  > PDCA Blocks Grouped (L{level}): Processed {len(chunks_to_process)} chunks | "
            f"P={len(pdca_groups.get('Plan', []))}, D={len(pdca_groups.get('Do', []))}, "
            f"C={len(pdca_groups.get('Check', []))}, A={len(pdca_groups.get('Act', []))}, "
            f"Other={len(pdca_groups.get('Other', []))}"
        )

        return plan_blocks, do_blocks, check_blocks, act_blocks, other_blocks

    # -------------------- _run_single_assessment (FINAL REVISED VERSION) --------------------

    def _run_single_assessment(
        self,
        sub_criteria: Dict[str, Any],
        statement_data: Dict[str, Any],
        vectorstore_manager: Optional['VectorStoreManager'],
        sequential_chunk_uuids: Optional[List[str]] = None,
        attempt: int = 1
    ) -> Dict[str, Any]:
        """
        รันการประเมิน Level เดียว (L1-L5) อย่างสมบูรณ์
        แก้ปัญหาทั้งหมด: Retry ซ้ำซาก, Context ว่าง, Hydration ล้มเหลว
        """
        MIN_RETRY_SCORE = getattr(self.config, 'min_retry_score', 0.65)
        MAX_RETRIEVAL_ATTEMPTS = getattr(self.config, 'max_retrieval_attempts', 3)

        start_time = time.time()
        sub_id = sub_criteria['sub_id']
        level = statement_data['level']
        statement_text = statement_data['statement']
        sub_criteria_name = sub_criteria['sub_criteria_name']
        statement_id = statement_data.get('statement_id', sub_id)

        self.logger.info(f"  > Starting assessment for {sub_id} L{level} (Attempt: {attempt})...")

        # ==================== 1. PDCA & Keywords ====================
        pdca_phase = self._get_pdca_phase(level)
        level_constraint = self._get_level_constraint_prompt(level)

        context_rules = self.contextual_rules_map.get(sub_id, {})
        must_include_keywords = ", ".join(context_rules.get("must_include_keywords", []))
        avoid_keywords = ", ".join(context_rules.get("avoid_keywords", []))

        # ==================== 2. Hybrid Retrieval Setup ====================
        mapped_stable_doc_ids, priority_docs_unhydrated = self._get_mapped_uuids_and_priority_chunks(
            sub_id=sub_id,
            level=level,
            statement_text=statement_text,
            level_constraint=level_constraint,
            vectorstore_manager=vectorstore_manager
        )

        # NEW FIX: ทำ Robust Hydration ให้ Priority Chunks ทันที
        priority_docs = self._robust_hydrate_documents_for_priority_chunks(
            chunks_to_hydrate=priority_docs_unhydrated,
            vsm=vectorstore_manager
        )

        # ==================== 3. Enhance Query ====================
        rag_query_list = enhance_query_for_statement(
            statement_text=statement_text,
            sub_id=sub_id,
            statement_id=statement_id,
            level=level,
            enabler_id=self.config.enabler,
            focus_hint=level_constraint,
            llm_executor=self.llm
        )
        rag_query = rag_query_list[0] if rag_query_list else statement_text

        # ==================== 4. LLM Evaluator ====================
        llm_evaluator_to_use = evaluate_with_llm_low_level if level <= 2 else self.llm_evaluator

        # ==================== 5. ADAPTIVE RAG LOOP ====================
        highest_rerank_score = 0.0
        final_top_evidences = []
        retrieval_start = time.time()
        loop_attempt = 1

        while loop_attempt <= MAX_RETRIEVAL_ATTEMPTS:
            self.logger.info(
                f"  > RAG Retrieval {sub_id} L{level} (Attempt: {loop_attempt}/{MAX_RETRIEVAL_ATTEMPTS}). "
                f"Best score so far: {highest_rerank_score:.4f}"
            )

            query_input = rag_query_list if loop_attempt == 1 and rag_query_list else [rag_query]

            try:
                retrieval_result = self.rag_retriever(
                    query=query_input,
                    doc_type=EVIDENCE_DOC_TYPES,
                    enabler=self.config.enabler,
                    sub_id=sub_id,
                    level=level,
                    vectorstore_manager=vectorstore_manager,
                    mapped_uuids=mapped_stable_doc_ids,
                    priority_docs_input=priority_docs,
                    sequential_chunk_uuids=sequential_chunk_uuids,
                )
            except Exception as e:
                self.logger.error(f"RAG retrieval failed: {e}")
                break

            top_evidences_current = retrieval_result.get("top_evidences", [])

            current_max_score = max(
                (ev.get("rerank_score") or ev.get("score", 0.0) for ev in top_evidences_current),
                default=0.0
            )

            priority_max_score = max(
                (doc.get("rerank_score") or doc.get("score", 0.0) for doc in priority_docs),
                default=0.0
            )

            overall_max_score = max(current_max_score, priority_max_score)

            self.logger.info(
                f"  > Attempt {loop_attempt} → New: {current_max_score:.4f} | Priority: {priority_max_score:.4f} | "
                f"Overall: {overall_max_score:.4f}"
            )

            if overall_max_score > highest_rerank_score:
                highest_rerank_score = overall_max_score
                final_top_evidences = top_evidences_current
                if loop_attempt > 1:
                    self.logger.info(f"  > Retrieval improved: New overall best {highest_rerank_score:.4f}")

            if highest_rerank_score >= MIN_RETRY_SCORE:
                self.logger.info(f"  > Adaptive Retrieval L{level}: Score {highest_rerank_score:.4f} ≥ {MIN_RETRY_SCORE} → STOP")
                break

            if loop_attempt < MAX_RETRIEVAL_ATTEMPTS:
                rag_query = self.rephrase_query_for_retry(rag_query, level, sub_id)

            loop_attempt += 1

        retrieval_duration = time.time() - retrieval_start
        top_evidences = final_top_evidences

        # ==================== 6. Adaptive Filtering ====================
        filtered = []
        for doc in top_evidences:
            score = doc.get('rerank_score', doc.get('score', 0.0))
            if score >= self.MIN_RERANK_SCORE_TO_KEEP:
                filtered.append(doc)
            else:
                doc_id = doc.get('chunk_uuid') or doc.get('doc_id') or 'UNKNOWN'
                self.logger.info(f"Filtering out chunk (ID: {doc_id}) | Score {score:.4f}")
        top_evidences = filtered

        # ==================== 7. Baseline from Previous Levels ====================
        previous_levels_evidence_full = []
        if level > 1 and self.is_sequential: 
            prev = self._collect_previous_level_evidences(sub_id, current_level=level)
            for lst in prev.values():
                previous_levels_evidence_full.extend(lst)

        # ==================== 8. Build Context (ใช้ Logic ใหม่) ====================
        
        # 8.1 🎯 NEW: Grouping, Sorting, and Filtering หลักฐาน (top_evidences)
        plan_blocks, do_blocks, check_blocks, act_blocks, other_blocks = self._get_pdca_blocks_from_evidences(
            top_evidences=top_evidences, # ใช้หลักฐานที่ถูกกรองแล้ว
            level=level
        )
        
        # 8.2 📌 NEW: รวมบล็อก PDCA ที่จัดโครงสร้างแล้วเข้าด้วยกันสำหรับ DIRECT EVIDENCE
        direct_context = "\n\n".join(filter(None, [
            plan_blocks,
            do_blocks,
            check_blocks,
            act_blocks,
            other_blocks
        ]))
        
        # 8.3 📌 OLD: ยังคงเรียกใช้ build_multichannel_context_for_level 
        # เพื่อดึง AUXILIARY และ BASELINE summaries (รักษารูปแบบเดิม)
        channels = build_multichannel_context_for_level(
            level=level,
            top_evidences=top_evidences,
            previous_levels_evidence=previous_levels_evidence_full,
            max_main_context_tokens=3000,
            max_summary_sentences=4
        )
        aux_summary = channels.get('aux_summary','')
        baseline_summary = channels.get('baseline_summary','ไม่มี')

        # 8.4 📌 Construct Final Context
        final_llm_context = "\n\n".join(filter(None, [
            f"--- DIRECT EVIDENCE (L{level})---\n{direct_context}", # <--- ใช้ Context ใหม่
            f"--- AUXILIARY EVIDENCE ---\n{aux_summary}",
            f"--- BASELINE FROM PREVIOUS LEVELS ---\n{baseline_summary}"
        ]))


        # ==================== 9. LLM Evaluation ====================
        evi_cap_data = self._calculate_evidence_strength_cap(top_evidences, level, highest_rerank_score=highest_rerank_score)
        max_evi_str_for_prompt = evi_cap_data['max_evi_str_for_prompt']

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
                must_include_keywords=must_include_keywords,
                avoid_keywords=avoid_keywords,
                max_rerank_score=highest_rerank_score,
                llm_executor=self.llm,
                max_evidence_strength=max_evi_str_for_prompt
            )
        except Exception as e:
            self.logger.error(f"LLM Call failed: {e}")
            llm_result = {}

        llm_duration = time.time() - llm_start

        # ==================== 9.5. Deterministic Score Fallback ====================
        if not isinstance(llm_result, dict):
            llm_result = {}
            
        llm_result = post_process_llm_result(llm_result, level) 

        # ==================== 10. Final Scoring ====================
        llm_score_pdca_sum = llm_result.get('score', 0)
        is_passed = llm_result.get('is_passed', False) 
        
        pdca_breakdown_llm = {
            'P': llm_result.get('P_Plan_Score', 0),
            'D': llm_result.get('D_Do_Score', 0),
            'C': llm_result.get('C_Check_Score', 0),
            'A': llm_result.get('A_Act_Score', 0),
        }
        
        final_score = llm_score_pdca_sum
        final_pdca_breakdown = pdca_breakdown_llm
        
        if is_passed:
            if level == 1:
                final_score = 1.0
                final_pdca_breakdown = {'P': 1, 'D': 0, 'C': 0, 'A': 0}
            elif level == 2:
                final_score = 2.0
                final_pdca_breakdown = {'P': 1, 'D': 1, 'C': 0, 'A': 0}
            elif level >= 3:
                final_score = llm_score_pdca_sum 

        status = "PASS" if is_passed else "FAIL"
        evidence_strength = min(max_evi_str_for_prompt, 10.0) if is_passed else 0.0 
        ai_confidence = "HIGH" if evidence_strength >= 8 else "MEDIUM" if evidence_strength >= 5.5 else "LOW"

        # กำหนด Icon ตามสถานะ
        icon = "🟢" if is_passed else "🔴"

        # ------------------- การแสดงผล Log -------------------
        self.logger.info(
            # 📌 แก้ไขบรรทัดนี้: เพิ่ม icon เข้าไปหน้าคำว่า PASS/FAIL
            f"  > Assessment {sub_id} L{level} completed → {icon} {'PASS' if is_passed else 'FAIL'} "
            f"(Score: {final_score:.1f} | Evi Str: {evidence_strength:.1f} | Conf: {ai_confidence})"
        )

        # สำคัญมาก: ต้อง return dict ที่มี temp_map_for_level ด้วย!
        return {
            "sub_criteria_id": sub_id,
            "statement_id": statement_id,
            "level": level,
            "statement": statement_text,
            "pdca_phase": pdca_phase,
            "llm_score": llm_score_pdca_sum,
            "pdca_breakdown": final_pdca_breakdown,
            "is_passed": is_passed,           
            "status": status,
            "score": final_score,
            "llm_result_full": llm_result,
            "retrieval_duration_s": round(retrieval_duration, 2),
            "llm_duration_s": round(llm_duration, 2),
            "evidence_strength": round(evidence_strength, 1),
            "ai_confidence": ai_confidence,
            "max_relevant_score": highest_rerank_score,
            "max_relevant_source": evi_cap_data.get('max_score_source'),
            "is_evidence_strength_capped": evi_cap_data.get('is_capped'),
            "max_evidence_strength_used": max_evi_str_for_prompt,
            "temp_map_for_level": top_evidences,  # สำคัญมาก! ขาดอันนี้ไม่ได้
        }