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
import hashlib


# -------------------- PATH SETUP & IMPORTS --------------------
try:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)

    from config.global_vars import (
        # เดิมที่มีอยู่แล้ว
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
        MIN_RERANK_SCORE_TO_KEEP,
        MIN_RETRY_SCORE,
        MIN_RELEVANCE_THRESHOLD,

        # === เพิ่มเหล่านี้ ===
        CONTEXT_CAP_L3_PLUS,                 # ใช้ใน _run_single_assessment สำหรับ cap context L3+
        CRITICAL_CA_THRESHOLD,               # ใช้ใน critical C/A evidence summary และ override logic
        MAX_RETRIEVAL_ATTEMPTS,              # ใช้ใน adaptive RAG loop
        HYBRID_VECTOR_WEIGHT,                # ถ้าใช้ใน rag_retriever หรือ hybrid setup
        HYBRID_BM25_WEIGHT,                  # เช่นเดียวกัน
        CHUNK_SIZE,                          # ถ้าใช้ใน hydration หรือ chunking
        CHUNK_OVERLAP,                       # เช่นเดียวกัน
        REQUIRED_PDCA,                 # <--- เพิ่ม
        CORRECT_PDCA_SCORES_MAP,       # <--- เพิ่ม
        PDCA_PRIORITY_ORDER,           # <--- เพิ่ม
        BASE_PDCA_KEYWORDS,            # <--- เพิ่ม
        PDCA_LEVEL_SYNONYMS,
        ENABLE_HARD_FAIL_LOGIC,
        ENABLE_CONTEXTUAL_RULE_OVERRIDE
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
    from core.action_plan_schema import ActionPlanActions  # เพิ่มด้านบน

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



# -----------------------------------------------------------------
# 🎯 FUNCTION: classify_by_keyword (Heuristic PDCA Classification)
# -----------------------------------------------------------------
def classify_by_keyword(
    text: str, 
    sub_id: str = None, 
    level: int = None,
    contextual_rules_map: dict = None
) -> str:
    """Heuristic PDCA Classification v13 – Custom P>D>C>A priority + Fallback"""
    if not text:
        return 'Other'
    
    text_lower = text.lower()
    
    # === ขั้น 1: รวบรวม Custom Keywords จาก contextual_rules.json (ครบ 4 phase) ===
    custom_keywords = defaultdict(list)
    
    if contextual_rules_map and sub_id:
        rules = contextual_rules_map.get(sub_id, {})
        
        # รวบรวม keywords จากทุก Level ที่มี (L1-L5)
        for l_key, l_rules in rules.items():
            if l_key.startswith("L"):
                # Plan Keywords (L1)
                planning_kw = l_rules.get("plan_keywords", "")
                if planning_kw:
                    custom_keywords['Plan'].extend([kw.strip().lower() for kw in planning_kw.split(",")])
                
                # Do Keywords (L2, L3)
                do_kw = l_rules.get("do_keywords", "")
                if do_kw:
                    custom_keywords['Do'].extend([kw.strip().lower() for kw in do_kw.split(",")])

                # Check Keywords (L3, L4, L5)
                check_kw = l_rules.get("check_keywords", "")
                if check_kw:
                    custom_keywords['Check'].extend([kw.strip().lower() for kw in check_kw.split(",")])
                
                # Act Keywords (L4, L5)
                act_kw = l_rules.get("act_keywords", "")
                if act_kw:
                    custom_keywords['Act'].extend([kw.strip().lower() for kw in act_kw.split(",")])

        # กรองให้เหลือ Keyword ที่ไม่ซ้ำกัน
        for key in custom_keywords:
            custom_keywords[key] = list(set(filter(None, custom_keywords[key])))

    # === ขั้น 2: ตรวจสอบ Custom Keywords (ลำดับใหม่: Plan > Do > Check > Act) ===
    
    # 📌 FIX: ตรวจสอบ P ก่อน A เพื่อให้หลักฐาน L1 P/D (เช่น แผน) ไม่ถูก Tag เป็น A/C ง่ายเกินไป
    
    # Plan
    if custom_keywords['Plan']:
        if any(kw in text_lower for kw in custom_keywords['Plan'] if kw):
            return 'Plan'
    # Do
    if custom_keywords['Do']:
        if any(kw in text_lower for kw in custom_keywords['Do'] if kw):
            return 'Do'
            
    # Check
    if custom_keywords['Check']:
        if any(kw in text_lower for kw in custom_keywords['Check'] if kw):
            return 'Check'
    # Act
    if custom_keywords['Act']:
        if any(kw in text_lower for kw in custom_keywords['Act'] if kw):
            return 'Act'

    
    # === ขั้น 3: Fallback ด้วย BASE_PDCA_KEYWORDS (ใช้ Act ก่อน Check ก่อน Do ก่อน Plan ตาม PDCA_PRIORITY_ORDER) ===
    # Fallback ยังคงใช้ลำดับที่เน้น A/C เพราะเป็น Logic ที่มาจาก BASE_PDCA_KEYWORDS
    for tag in PDCA_PRIORITY_ORDER: 
        for pattern in BASE_PDCA_KEYWORDS[tag]:
            if re.search(pattern, text, re.IGNORECASE):
                return tag
    
    return 'Other'

def get_correct_pdca_required_score(level: int) -> int:
    """กำหนดคะแนนรวมขั้นต่ำที่ต้องทำได้เพื่อถือว่าผ่าน Level นั้น ๆ"""
    if level == 1:
        return 1
    elif level == 2:
        return 2
    # หมายเหตุ: L3, L4, L5 ต้องมีคะแนนรวม PDCA ครบตามที่กำหนดในเกณฑ์ 
    # (P, D, C, A = 2 คะแนนต่อแกน)
    elif level == 3: # ต้องมี P, D, C อย่างน้อย 1.0/2.0 คะแนนรวม 4 (ถ้า L3 เน้น C ต้องได้ P, D, C)
        return 4
    elif level == 4: # ต้องมี P, D, C, A คะแนนรวม 6 (ถ้า L4 เน้น A ต้องได้ P, D, C, A)
        return 6
    elif level == 5: # ต้องมีคะแนนเต็มทุกแกน คะแนนรวม 8
        return 8
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
            document_map,                # (10th element)
            action_plan_model
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
            document_map=document_map, # ⬅️ ส่ง document_map ที่เพิ่ง Unpack เข้ามา
            ActionPlanActions=action_plan_model
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
    วันที่: 15 ธ.ค. 2568 เวลา 08:30 น. — แก้ไข PDCA Logic และ Floating Point Precision
    ผู้สร้าง: พี่ + ผม + Gemini
    """
    logger = logging.getLogger(__name__)
    
    # 📌 FIX 1: Ensure scores are float for calculations and round them immediately for output (Floating Point Precision Fix)
    p = round(float(llm_output.get("P_Plan_Score", 0)), 1)
    d = round(float(llm_output.get("D_Do_Score", 0)), 1)
    c = round(float(llm_output.get("C_Check_Score", 0)), 1)
    a = round(float(llm_output.get("A_Act_Score", 0)), 1)
    pdca_real_sum = p + d + c + a
    llm_score = llm_output.get("score", 0)

    # SE-AM Threshold ที่ถูกต้อง 100% (แก้ไขตามที่ user ให้มา)
    # L1: 1 (P>=1)
    # L2: 2 (P=2)
    # L3: 4 (P=2, D=2)
    # L4: 6 (P=2, D=2, C=1, A=1) 
    # L5: 8 (P=2, D=2, C=2, A=2)
    threshold_map = {1: 1, 2: 2, 3: 4, 4: 6, 5: 8} 
    threshold = threshold_map.get(level, 2)

    # 1. จับ LLM โกง (ใช้ PDCA Sum ในการตัดสินใจ)
    if float(llm_score) != pdca_real_sum: 
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
    
    # 🎯 CRITICAL FIX 2: บังคับ FAIL หากขาดหลักฐานสำคัญ (C/A) แม้ PDCA Sum จะถึงเกณฑ์
    # (Bug ที่ทำให้ L3/L4/L5 ผ่าน ทั้งที่ C=0 หรือ A=0)
    if real_pass:
        if level == 3:
            # L3 ต้องมี C > 0 (อย่างน้อย 1.0 คะแนน)
            if c <= 0.0:
                logger.warning(f"🚨 L3 FAIL OVERRIDE: C_Check_Score is {c:.1f} (Must be > 0.0).")
                real_pass = False
        elif level == 4:
            # L4 ต้องมี A > 0 (อย่างน้อย 1.0 คะแนน)
            if a <= 0.0:
                logger.warning(f"🚨 L4 FAIL OVERRIDE: A_Act_Score is {a:.1f} (Must be > 0.0).")
                real_pass = False
        elif level == 5:
            # L5 ต้องมี C >= 2.0 และ A >= 2.0
            if c < 2.0 or a < 2.0:
                logger.warning(f"🚨 L5 FAIL OVERRIDE: L5 requires C={c:.1f} and A={a:.1f} (Must be >= 2.0 each).")
                real_pass = False

    if llm_output.get("is_passed") != real_pass:
        logger.critical(f"FORCING is_passed = {real_pass} (PDCA={pdca_real_sum} ≥ {threshold}) [Post-Logic Check]")
        llm_output["is_passed"] = real_pass

    # 3. บันทึกประวัติศาสตร์ และใช้ค่าที่ถูก Round แล้ว
    llm_output.update({
        "P_Plan_Score": p, # <--- ใช้ค่าที่ Round แล้ว
        "D_Do_Score": d,
        "C_Check_Score": c,
        "A_Act_Score": a,
        "pdca_breakdown": {"P": p, "D": d, "C": c, "A": a},
        "pdca_sum": pdca_real_sum,
        "pass_threshold": threshold,
        "final_score": round(pdca_real_sum, 2), # <--- Round final_score
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
    min_retry_score: float = MIN_RETRY_SCORE
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
        doc_type: str = EVIDENCE_DOC_TYPES, 
        vectorstore_manager: Optional['VectorStoreManager'] = None,
        evidence_map_path: Optional[str] = None,
        document_map: Optional[Dict[str, str]] = None,
        is_parallel_all_mode: bool = False,
        sub_id: str = 'all'
    ):
        # =======================================================
        # 🎯 Logger Setup (อันดับแรกสุด)
        # =======================================================
        if logger_instance is not None:
            self.logger = logger_instance
        else:
            self.logger = logging.getLogger(__name__).getChild(
                f"Engine|{config.enabler}|{config.tenant}/{config.year}"
            )
        
        self.logger.info(f"Initializing SEAMPDCAEngine for {config.enabler} ({config.tenant}/{config.year})")

        # =======================================================
        # Core Configuration
        # =======================================================
        self.config = config
        self.enabler_id = config.enabler
        self.target_level = config.target_level
        self.sub_id = sub_id

        # Load rubric
        self.rubric = self._load_rubric()

        # Vectorstore & Doc Type
        self.vectorstore_manager = vectorstore_manager
        self.doc_type = doc_type

        # Constants
        self.FINAL_K_RERANKED = FINAL_K_RERANKED
        self.PRIORITY_CHUNK_LIMIT = PRIORITY_CHUNK_LIMIT

        # LLM
        self.llm = llm_instance

        # Mode flags
        self.is_sequential = config.force_sequential
        self.is_parallel_all_mode = is_parallel_all_mode
        self.logger.info(
            f"Engine mode: {'FULL PARALLEL (stateless)' if is_parallel_all_mode else 'SEQUENTIAL/MIXED (with hydration)'}"
        )

        self.required_pdca_map = REQUIRED_PDCA
        self.base_pdca_keywords = BASE_PDCA_KEYWORDS

        # Retry policy
        self.retry_policy = RetryPolicy(
            max_attempts=3,
            base_delay=2.0,
            jitter=True,
            escalate_context=True,
            shorten_prompt_on_fail=True,
            exponential_backoff=True,
        )

        # Thresholds
        self.RERANK_THRESHOLD: float = RERANK_THRESHOLD
        self.MAX_EVI_STR_CAP: float = MAX_EVI_STR_CAP

        # =======================================================
        # Persistent Evidence Mapping
        # =======================================================
        if evidence_map_path:
            self.evidence_map_path = evidence_map_path
        else:
            self.evidence_map_path = get_evidence_mapping_file_path(
                tenant=self.config.tenant,
                year=self.config.year,
                enabler=self.enabler_id
            )

        self.evidence_map: Dict[str, List[Dict]] = {}
        self.temp_map_for_save: Dict[str, List[Dict]] = {}

        # self.RERANK_THRESHOLD = 0.5
        # self.MAX_EVI_STR_CAP = 3.0

        # Load contextual rules and existing evidence map
        self.contextual_rules_map: Dict[str, Dict[str, Any]] = self._load_contextual_rules_map()
        self.evidence_map = self._load_evidence_map()

        self.logger.info(f"Persistent Map Path: {self.evidence_map_path}")
        self.logger.info(f"Loaded {len(self.evidence_map)} existing evidence entries.")

        # =======================================================
        # Function Pointers (with mocking support)
        # =======================================================
        self.llm_evaluator = evaluate_with_llm
        self.rag_retriever = retrieve_context_with_filter
        self.create_structured_action_plan = create_structured_action_plan

        if config.mock_mode in ["random", "control"]:
            self._set_mock_handlers(config.mock_mode)

        if config.mock_mode == "control":
            self.logger.info("Enabling global LLM data utils mock control mode.")
            set_llm_data_mock_mode(True)
        elif config.mock_mode == "random":
            self.logger.warning("Mock mode 'random' not fully implemented. Using default behavior.")

        # =======================================================
        # Lazy Initialization
        # =======================================================
        if self.llm is None:
            self._initialize_llm_if_none()
        if self.vectorstore_manager is None:
            self._initialize_vsm_if_none()

        # Force reload doc mapping to prevent hydration issues in workers
        if self.vectorstore_manager and not getattr(self.vectorstore_manager, '_doc_id_mapping', None):
            self.vectorstore_manager._load_doc_id_mapping()
            self.logger.info(
                f"Forced reload Doc ID Mapping: {len(self.vectorstore_manager._doc_id_mapping)} docs, "
                f"{len(self.vectorstore_manager._uuid_to_doc_id)} chunks"
            )

        # =======================================================
        # Document Map Loading (Filename Resolution)
        # =======================================================
        map_to_use: Dict[str, str] = document_map or {}

        if not map_to_use:
            mapping_path = get_mapping_file_path(
                self.doc_type,
                tenant=self.config.tenant,
                year=self.config.year,
                enabler=self.enabler_id
            )
            self.logger.info(f"Loading document_map from: {mapping_path}")

            try:
                with open(mapping_path, 'r', encoding='utf-8') as f:
                    doc_map_raw = json.load(f)
                map_to_use = {
                    doc_id: data.get("file_name", doc_id)
                    for doc_id, data in doc_map_raw.items()
                }
                self.logger.info(f"Loaded {len(map_to_use)} document mappings.")
            except FileNotFoundError:
                self.logger.warning(f"Document mapping file not found: {mapping_path}")
            except Exception as e:
                self.logger.error(f"Failed to load document map: {e}")

        self.doc_id_to_filename_map: Dict[str, str] = map_to_use
        self.document_map: Dict[str, str] = self.doc_id_to_filename_map

        if not self.doc_id_to_filename_map:
            self.logger.warning("Document ID → Filename map is empty. Filename resolution limited.")

        self.ActionPlanActions = ActionPlanActions  # เพิ่มบรรทัดนี้
        if self.ActionPlanActions is None:
            self.ActionPlanActions = ActionPlanActions
            
        self.logger.info(f"Engine initialized: Enabler={self.enabler_id}, Mock={config.mock_mode}")
    
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
                # 🎯 FIX: เปลี่ยน evidence_enabler เป็น enabler_filter ให้ถูกต้องตาม load_all_vectorstores
                self.vectorstore_manager = load_all_vectorstores(
                    doc_types=[EVIDENCE_DOC_TYPES], 
                    enabler_filter=self.enabler_id, # <--- **แก้ไขตรงนี้!**
                    tenant=self.config.tenant, 
                    year=self.config.year       
                )
                
                # บังคับโหลด Doc ID Map ซ้ำเพื่อป้องกัน Map หายใน Worker (Safety Net)
                if self.vectorstore_manager:
                    # NOTE: การโหลดครั้งแรกจะทำภายใน VSM.__init__ 
                    # แต่การเรียกซ้ำนี้ช่วยรับประกันว่า Map มีข้อมูล
                    self.vectorstore_manager._load_doc_id_mapping() 

                # โค้ด Log เพื่อยืนยัน
                len_retrievers = 0
                if self.vectorstore_manager and hasattr(self.vectorstore_manager, '_multi_doc_retriever') and self.vectorstore_manager._multi_doc_retriever:
                     # 💡 การเข้าถึง _all_retrievers ต้องทำผ่าน self.vectorstore_manager._multi_doc_retriever._all_retrievers
                     len_retrievers = len(
                        self.vectorstore_manager._multi_doc_retriever._all_retrievers
                    )
                     self.logger.info("✅ MultiDocRetriever loaded with %s collections and cached in VSM.", 
                                 len_retrievers) 
                else:
                    self.logger.warning("VectorStoreManager loaded but MultiDocRetriever is None or missing expected attributes.")
                
                if len_retrievers == 0:
                    self.logger.error("FATAL: VectorStoreManager initialized but loaded 0 vector store collections. Check data path.")
                    raise ValueError("0 vector store collections loaded. Cannot proceed with assessment.")


            except Exception as e:
                # 📌 Log เดิม: ERROR - FATAL: Could not initialize VectorStoreManager: load_all_vectorstores() got an unexpected keyword argument 'evidence_enabler'
                # 📌 หลังจากแก้ไขแล้ว: จะเจอข้อความ Error ที่แท้จริง (เช่น No collections found)
                self.logger.error(f"FATAL: Could not initialize VectorStoreManager: {e}")
                raise # Re-raise the exception to หยุดโปรแกรม

    def _get_applicable_contextual_rule(self, sub_id: str, level: int) -> Optional[Dict[str, Any]]:
        """
        ค้นหา Contextual Rule ที่มี target_sub_criteria และ target_level ตรงกัน
        """
        # self.contextual_rules_map คือ dict ที่โหลดจาก pea_km_contextual_rules.json
        for rule_name, rule_data in self.contextual_rules_map.items():
            if (rule_data.get('target_sub_criteria') == sub_id and 
                rule_data.get('target_level') == level):
                
                # เพิ่มชื่อ Rule เข้าไปใน Data เพื่อการตรวจสอบภายหลัง
                rule_data['name'] = rule_name 
                return rule_data
        return None

    def _check_contextual_rule_condition(
        self, 
        condition: Dict[str, Any], 
        sub_id: str, 
        level: int, 
        previous_levels_evidence_dict: Dict[str, List[Dict[str, Any]]], 
        top_evidences: List[Dict[str, Any]]
    ) -> bool:
        """
        ตรวจสอบเงื่อนไขทั้งหมดใน Rule Condition (รองรับ 'and' และเงื่อนไขย่อย)
        """
        
        # ตรวจสอบเงื่อนไข 'and' (ปัจจุบัน Contextual Rule ของเราใช้แค่ 'and')
        if 'and' in condition:
            for sub_condition in condition['and']:
                
                # 1. check_passed_levels (ตรวจสอบ L1, L2 ต้อง PASS)
                if 'check_passed_levels' in sub_condition:
                    required_levels = sub_condition['check_passed_levels']
                    for required_level in required_levels:
                        if not self._is_previous_level_passed(sub_id, required_level):
                            self.logger.debug(f"Rule Condition Failed: {required_level} not passed.")
                            return False # ถ้า L1/L2 ไม่ผ่าน ถือว่าเงื่อนไข Fail
                
                # 2. check_missing_pdca (ตรวจสอบว่าเกิด Evidence Gap ใน D/C หรือไม่)
                if 'check_missing_pdca' in sub_condition:
                    # Logic นี้ถูกตรวจสอบภายนอกใน _run_single_assessment (missing_tags) 
                    # แต่เราสามารถตรวจสอบซ้ำได้ แต่สำหรับตอนนี้ ให้ถือว่าถ้ามาถึงขั้นนี้แล้ว missing_tags ถูกตรวจสอบแล้ว
                    pass
                
                # 3. check_evidence_exists (สำคัญที่สุด: ตรวจสอบหลักฐาน C/A ที่มี Rerank Score สูง)
                if 'check_evidence_exists' in sub_condition:
                    required_phases = sub_condition['check_evidence_exists'].get('phase', [])
                    min_rerank = sub_condition['check_evidence_exists'].get('min_rerank', globals().get('CRITICAL_CA_THRESHOLD', 0.65))
                    min_count = sub_condition['check_evidence_exists'].get('min_count', 1)
                    
                    found_count = 0
                    
                    # ตรวจสอบในหลักฐานที่ค้นพบใน Level ปัจจุบัน (top_evidences)
                    for doc in top_evidences:
                        if doc.get('pdca_tag') in required_phases and doc.get('rerank_score', 0.0) >= min_rerank:
                            found_count += 1
                            
                    if found_count < min_count:
                        self.logger.debug(f"Rule Condition Failed: Found only {found_count} of required {min_count} {required_phases} with Rerank >= {min_rerank}.")
                        return False # ถ้าหาหลักฐาน C/A Critical ไม่เจอ ถือว่าเงื่อนไข Fail

            # ถ้าเงื่อนไขย่อยทั้งหมดใน 'and' เป็นจริง
            return True 
        
        return False # ถ้า Rule Format ไม่ถูกต้อง

    def _is_previous_level_passed(self, sub_id: str, level: int) -> bool:
        """
        Helper: ตรวจสอบสถานะการผ่านของ Level ก่อนหน้าจาก self.assessment_results_map
        (ใช้สำหรับ check_passed_levels ใน Contextual Rule)
        """
        # Key ใน assessment_results_map มักจะเป็น '3.1.L1', '3.1.L2'
        key = f"{sub_id}.L{level}"
        
        # ตรวจสอบว่าผลลัพธ์ของ Level นั้นมีอยู่ และมี is_passed เป็น True
        result = self.assessment_results_map.get(key)
        
        return result is not None and result.get('is_passed', False)

    def enhance_query_for_statement(
        self,
        statement_text: str,
        sub_id: str,
        statement_id: str,
        level: int,
        focus_hint: str,
    ) -> List[str]:
        """
        สร้าง Query ที่ฉลาด แม่นยำ และครอบคลุม PDCA ทุกด้าน
        โดยใช้ Contextual Map ที่ Engine ถือครอง (self.contextual_rules_map)
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Generating queries for {self.enabler_id} - {sub_id} L{level}")

        # --- 1. PDCA Synonyms ตาม Level (ดึงจาก Global Constant) ---
        current_synonyms = PDCA_LEVEL_SYNONYMS.get(level, "")

        # --- 2. ดึง Keyword จาก Contextual Map ที่ Engine ถือครอง ---
        custom_keywords_list = []
        
        contextual_map = self.contextual_rules_map
        enabler_id = self.enabler_id

        # เข้าถึง Map เฉพาะ Sub-Criteria ที่กำลังประเมิน
        if contextual_map and sub_id in contextual_map:
            sub_map = contextual_map[sub_id]

            # Logic การดึง Keywords ที่เราปรับแก้สำหรับ L1-L5 (เน้น Cross-Phase Keyword)
            if level == 1:
                custom_keywords_list.append(sub_map.get('L1', {}).get('plan_keywords', ''))
            elif level == 2:
                custom_keywords_list.append(sub_map.get('L2', {}).get('do_keywords', ''))
            elif level == 3: # L3: Check & Do
                # ดึง Keywords ของ C และ D มารวมกันเพื่อช่วย RAG/Reranker
                custom_keywords_list.append(sub_map.get('L3', {}).get('check_keywords', ''))
                custom_keywords_list.append(sub_map.get('L3', {}).get('do_keywords', ''))
            elif level == 4: # L4: Check & Act
                custom_keywords_list.append(sub_map.get('L4', {}).get('act_keywords', ''))
                custom_keywords_list.append(sub_map.get('L4', {}).get('check_keywords', ''))
            elif level == 5: # L5: Act & Check
                custom_keywords_list.append(sub_map.get('L5', {}).get('act_keywords', ''))
                custom_keywords_list.append(sub_map.get('L5', {}).get('check_keywords', ''))

        # รวม Custom Keywords เข้ากับ Synonyms
        custom_keywords_str = ", ".join(filter(None, custom_keywords_list))
        if custom_keywords_str:
            current_synonyms += f", {custom_keywords_str}"

        # --- 3. Base Query (หลัก) ---
        base_query = f"**{statement_text}** {sub_id} L{level} {enabler_id} คำสำคัญ: {current_synonyms}"
        queries = [base_query]

        # --- 4. Dedicated Queries สำหรับ L3+ ---
        if level >= 3:
            # Query 1: เน้น C (การวัดผล/รายงาน)
            queries.append(
                f"รายงานผล การวัดผล KPI Audit การประเมิน {statement_text} {sub_id} รายงานประจำปี การวิเคราะห์ช่องว่าง"
            )
            # Query 2: เน้น A (การปรับปรุงแก้ไข)
            queries.append(
                f"การปรับปรุง แก้ไข Corrective Action บทเรียนที่ได้รับ {statement_text} {sub_id} ตามผลการประเมิน"
            )
            # Query 3: รวม PDCA 4 ด้าน (D, C, A) เพื่อหาหลักฐานครบวงจร
            queries.append(
                f"การดำเนินการ การวัดผล การปรับปรุง {statement_text} (PDCA) {sub_id} {enabler_id}"
            )

        # --- 5. L5 Special ---
        if level == 5:
            # Query 4: เน้นผลลัพธ์และความยั่งยืน (Optimization)
            queries.append(
                f"นวัตกรรม ความยั่งยืน Best Practice รางวัล {statement_text} {sub_id} {enabler_id}"
            )
            # Query 5: เน้นบทบาทผู้บริหารในการทบทวน (สำคัญต่อ 1.1)
            queries.append(
                f"ผู้บริหารระดับสูงติดตาม ดูแล และสนับสนุน {statement_text} การทบทวนวิสัยทัศน์"
            )

        # --- 6. จำกัดจำนวน + Log ---
        queries = queries[:6]
        logger.info(f"Generated {len(queries)} queries for {enabler_id} - {sub_id} L{level}")
        return queries
    

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

        Final Production Version - 14 ธ.ค. 2568 (ULTIMATE VICTORY EDITION)
        - เพิ่ม is_parallel_all_mode เพื่อข้าม hydration ใน full parallel mode (by design)
        - รักษา Heuristic Fallback เพื่อแก้ CRITICAL MAPPING FAILURE
        """
        
        # --------------------------------------------------------------
        # 1. ข้าม Hydration ทั้งหมดใน Full Parallel Mode (by design)
        # --------------------------------------------------------------
        if self.is_parallel_all_mode:
            self.logger.info("FULL PARALLEL MODE: Skipping previous level evidence hydration (stateless by design)")
            return {}  # คืน map ว่าง → ไม่ hydrate priority chunks จาก level ก่อนหน้า

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
        chunk_map = {}  # Key: Cleaned V4 UUID (without dashes)
        total_retrieved = len(full_chunks)
        
        # 1. สร้าง Map (Key: V4 UUID Cleaned)
        for idx, chunk in enumerate(full_chunks):
            meta = getattr(chunk, "metadata", {})
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
                
                # ID จาก evidence level ก่อนหน้า
                cid_l1 = (ev.get("chunk_uuid") or "").replace("-", "") 
                sid_l1 = ev.get("stable_doc_uuid") or ev.get("doc_id") 
                vsm_mapping_failed = True

                # --- 1. Primary Lookup ---
                if cid_l1:
                    data = chunk_map.get(cid_l1)

                # --- 2. Fallback: VSM Mapping ---
                if not data and sid_l1 and vsm and hasattr(vsm, '_doc_id_mapping') and vsm._doc_id_mapping:
                    if cid_l1 and not data:
                        self.logger.warning(f"Hydration Check: Primary L1 chunk_uuid '{cid_l1[:8]}...' NOT found in map. Starting Stable ID Fallback...")
                    
                    if sid_l1 in vsm._doc_id_mapping:
                        vsm_mapping_failed = False
                        for v4_uuid_raw in vsm._doc_id_mapping[sid_l1].get("chunk_uuids", []):
                            v4_uuid_cleaned = v4_uuid_raw.replace("-", "")
                            if v4_uuid_cleaned in chunk_map:
                                data = chunk_map[v4_uuid_cleaned]
                                self.logger.info(f"✅ Fallback SUCCESS (VSM Map): Matched via V4 UUID '{v4_uuid_cleaned[:8]}...'")
                                break
                    else:
                        self.logger.warning(f"Hydration Check: Stable ID {sid_l1[:8]}... NOT found in VSM Doc ID Mapping.")

                # --- 3. HEURISTIC FALLBACK (Final Bypass) ---
                if not data:
                    if not vsm_mapping_failed:
                        self.logger.warning(f"⚠️ VSM Map exists but failed. Attempting Heuristic Match by Stable ID.")
                    
                    for retrieved_chunk_data in chunk_map.values():
                        retrieved_sid = retrieved_chunk_data["metadata"].get("stable_doc_uuid")
                        if retrieved_sid == sid_l1:
                            data = retrieved_chunk_data
                            new_ev["chunk_uuid"] = retrieved_chunk_data["metadata"].get("chunk_uuid", new_ev["chunk_uuid"])
                            self.logger.info(f"🟢 Heuristic SUCCESS: Restored using matching Stable ID {sid_l1[:8]}...")
                            break

                # --------------------------------------------------------------------------
                if data:
                    new_ev["text"] = data["text"]
                    new_ev.update({k: v for k, v in data["metadata"].items() 
                                if k not in ["text", "page_content"]})
                    restored += 1
                    
                    # 🎯 CRITICAL FIX: เพิ่ม Flag is_baseline
                    new_ev["is_baseline"] = True 
                    
                else:
                    sid_l1 = ev.get("stable_doc_uuid") or ev.get("doc_id")
                    self.logger.error(f"❌ CRITICAL MAPPING FAILURE: Could not restore chunk (Stable ID: {sid_l1[:8] if sid_l1 else 'N/A'}...) from {len(chunk_map)} retrieved chunks.")
                    new_ev["is_baseline"] = False # กำหนดเป็น False หากล้มเหลวในการ hydrate
                
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
        บันทึก evidence map อย่างปลอดภัย 100% - Atomic Write + FileLock + Cleanup + Raw LLM Data
        รองรับการบันทึก raw_llm_pdca เพื่อความโปร่งใสและการตรวจสอบย้อนกลับ
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

        self.logger.info(f"[EVIDENCE] Preparing to save evidence map → {map_file_path}")

        try:
            os.makedirs(os.path.dirname(map_file_path), exist_ok=True)

            with FileLock(lock_path, timeout=60):
                self.logger.debug("[EVIDENCE] File lock acquired.")

                # === Merge Logic ===
                if map_to_save is not None:
                    final_map_to_write = map_to_save
                else:
                    existing_map = self._load_evidence_map(is_for_merge=True) or {}
                    runtime_map = deepcopy(self.evidence_map)
                    final_map_to_write = existing_map

                    for key, new_entries in runtime_map.items():
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
                                # Update ถ้า score สูงกว่า หรือถ้ามี raw_llm_pdca ที่ละเอียดกว่า
                                old_entry = entry_map[entry_id]
                                old_score = old_entry.get("relevance_score", 0.0)
                                if new_score > old_score:
                                    entry_map[entry_id] = new_entry

                        final_map_to_write[key] = list(entry_map.values())

                if not final_map_to_write:
                    self.logger.warning("[EVIDENCE] Nothing to save (empty map).")
                    return

                # === Cleanup + Sort ===
                final_map_to_write = self._clean_temp_entries(final_map_to_write)
                for key, entries in final_map_to_write.items():
                    entries.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)

                # === Atomic Write ===
                with tempfile.NamedTemporaryFile(
                    mode='w', delete=False, encoding="utf-8", dir=os.path.dirname(map_file_path)
                ) as tmp_file:
                    cleaned_data = self._clean_map_for_json(final_map_to_write)
                    json.dump(cleaned_data, tmp_file, indent=4, ensure_ascii=False)
                    tmp_path = tmp_file.name

                shutil.move(tmp_path, map_file_path)
                tmp_path = None

                # === Stats ===
                total_keys = len(final_map_to_write)
                total_items = sum(len(v) for v in final_map_to_write.values())
                file_size_kb = os.path.getsize(map_file_path) / 1024
                self.logger.info(
                    f"[EVIDENCE] SAVED SUCCESSFULLY! "
                    f"Keys: {total_keys} | Items: {total_items} | Size: ~{file_size_kb:.1f} KB | Path: {map_file_path}"
                )

        except Exception as e:
            self.logger.critical("[EVIDENCE] FATAL ERROR DURING SAVE")
            self.logger.exception(e)
            raise

        finally:
            # === Cleanup lock & temp file (Double Safety) ===
            if os.path.exists(lock_path):
                try:
                    os.unlink(lock_path)
                    self.logger.debug(f"[EVIDENCE] Removed lock file: {lock_path}")
                except Exception as e:
                    self.logger.warning(f"[EVIDENCE] Failed to remove lock: {e}")

            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass

    def _load_evidence_map(self, is_for_merge: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        """
        โหลด evidence map อย่างปลอดภัย
        is_for_merge = True → ไม่ log "No existing map" (ใช้ใน save)
        """
        try:
            path = get_evidence_mapping_file_path(
                tenant=self.config.tenant,
                year=self.config.year,
                enabler=self.enabler_id
            )
        except Exception as e:
            self.logger.error(f"[EVIDENCE] FATAL: ไม่สามารถกำหนด Path สำหรับโหลดได้: {e}")
            return {}

        if not os.path.exists(path):
            if not is_for_merge:
                self.logger.info("[EVIDENCE] No existing evidence map found – starting fresh.")
            return {}

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not is_for_merge:
                total_items = sum(len(v) for v in data.values() if isinstance(v, list))
                self.logger.info(f"[EVIDENCE] Loaded evidence map: {len(data)} keys, {total_items} items from {path}")
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
        🟢 FIX: เพิ่ม Pre-boost เพื่อป้องกัน Score 0.0000
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

        # 🟢 FIX 1: PRE-BOOST BEFORE HYDRATION
        # ป้องกัน Score 0.0000 ที่จะถูกกรองออกภายหลัง
        for chunk in priority_chunks:
            # ถ้าไม่มี Score หรือ Score = 0 → Set ค่า Default 0.80
            if "rerank_score" not in chunk or chunk.get("rerank_score", 0.0) == 0.0:
                chunk["rerank_score"] = 0.80
            if "score" not in chunk or chunk.get("score", 0.0) == 0.0:
                chunk["score"] = 0.80
            
            # ⭐ CRITICAL: ตั้ง Flag is_baseline = True ตั้งแต่ตอนนี้
            chunk["is_baseline"] = True

        self.logger.info(f"Pre-boosted {len(priority_chunks)} priority chunks (Score set to 0.80, is_baseline=True)")

        # 2. ทำ Robust Hydration ทันทีที่นี่เลย!
        priority_chunks = self._robust_hydrate_documents_for_priority_chunks(
            chunks_to_hydrate=priority_chunks,
            vsm=vectorstore_manager
        )

        # 3. สร้าง mapped_stable_ids สำหรับ RAG Retriever
        for chunk in priority_chunks:
            sid = chunk.get("stable_doc_uuid") or chunk.get("doc_id")
            if sid and isinstance(sid, str) and len(sid.replace("-", "")) >= 64:
                mapped_stable_ids.append(sid)

        self.logger.info(f"PRIORITY HYDRATED → {len(priority_chunks)} chunks ready for L{level} (with full text + baseline flag)")

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

    def _calculate_overall_stats(self, target_sub_id: str):
        """
        Calculates overall statistics from sub-criteria results (self.final_subcriteria_results)
        and stores them in self.total_stats.
        """
        results = self.final_subcriteria_results
        if not results:
            self.total_stats = {
                "Overall Maturity Score (Avg.)": 0.0,
                "Overall Maturity Level (Weighted)": "L0",
                "Number of Sub-Criteria Assessed": 0,
                "Total Weighted Score Achieved": 0.0,
                "Total Possible Weight": 0.0,
                "Overall Progress Percentage (0.0 - 1.0)": 0.0,
                "percentage_achieved_run": 0.0,
                "total_subcriteria": len(self._flatten_rubric_to_statements()),
                "target_level": self.config.target_level,
                "enabler": self.config.enabler,
                "sub_criteria_id": target_sub_id,
            }
            return

        # 1. Calculate Sums
        total_weighted_score_achieved = sum(r.get('weighted_score', 0) for r in results)
        total_possible_weight = sum(r.get('weight', 0) for r in results)

        # 2. Overall Maturity Score (Avg.)
        overall_avg_score = 0.0
        if total_possible_weight > 0:
            overall_avg_score = total_weighted_score_achieved / total_possible_weight
            # 🟢 FIX: ROUNDING for clean output (e.g., 1.999... -> 2.0)
            overall_avg_score = round(overall_avg_score, 2) 
        
        # 3. Overall Progress Percentage (0.0 - 1.0)
        overall_progress_percentage = 0.0
        # Assume MAX_LEVEL is 5 (หรือดึงจาก self.config หรือ global_vars)
        MAX_LEVEL = getattr(globals(), 'MAX_LEVEL', 5) 
        if total_possible_weight > 0 and MAX_LEVEL > 0:
            max_possible_score = total_possible_weight * MAX_LEVEL
            overall_progress_percentage = total_weighted_score_achieved / max_possible_score
            # 🟢 FIX: ROUNDING for clean output (4 ตำแหน่งสำหรับเปอร์เซ็นต์)
            overall_progress_percentage = round(overall_progress_percentage, 4)

        # 4. Overall Maturity Level (Weighted)
        # ปัดเศษค่าเฉลี่ยที่คำนวณได้ เพื่อกำหนด Level (เช่น 1.2 -> L1, 1.5 -> L2)
        highest_level_achieved = round(overall_avg_score)
        final_level = min(max(int(highest_level_achieved), 0), MAX_LEVEL)
        overall_level_label = f"L{final_level}"
        
        # 5. Final Percentage Achieved (0-100%)
        percentage_achieved_run = overall_progress_percentage * 100
        # 🟢 FIX: ROUNDING for clean output (1 ตำแหน่งสำหรับ 0-100%)
        percentage_achieved_run = round(percentage_achieved_run, 1)


        self.total_stats = {
            "Overall Maturity Score (Avg.)": overall_avg_score, # <--- FIXED
            "Overall Maturity Level (Weighted)": overall_level_label,
            "Number of Sub-Criteria Assessed": len(results),
            "Total Weighted Score Achieved": round(total_weighted_score_achieved, 2), # <--- FIXED
            "Total Possible Weight": total_possible_weight,
            "Overall Progress Percentage (0.0 - 1.0)": overall_progress_percentage, # <--- FIXED
            "percentage_achieved_run": percentage_achieved_run, # <--- FIXED
            "total_subcriteria": len(self._flatten_rubric_to_statements()),
            "target_level": self.config.target_level,
            "enabler": self.config.enabler,
            "sub_criteria_id": target_sub_id,
        }
        
        self.logger.info(f"OVERALL STATS: Avg Score={overall_avg_score}, Level={overall_level_label}")

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
        รันการประเมิน L1-L5 แบบ sequential (หรือแบบเต็มในโหมด Mixed/Parallel) สำหรับ sub-criteria หนึ่งตัว
        และส่ง evidence map กลับไปให้ main process รวม (รวมถึงการสร้าง Action Plan)
        """
        # 📌 ใช้ Global/Class Constant ที่ประกาศไว้ใน Header ของ core/seam_assessment.py
        REQUIRED_PDCA: Final[Dict[int, Set[str]]] = globals().get('REQUIRED_PDCA', {1: {"P"}, 2: {"P", "D"}, 3: {"P", "D", "C"}, 4: {"P", "D", "C", "A"}, 5: {"P", "D", "C", "A"}})
        MAX_L1_ATTEMPTS = globals().get('MAX_L1_ATTEMPTS', 2)
        WEAK_EVIDENCE_THRESHOLD = globals().get('WEAK_EVIDENCE_THRESHOLD', 5.0)

        sub_id = sub_criteria['sub_id']
        sub_criteria_name = sub_criteria['sub_criteria_name']
        sub_weight = sub_criteria.get('weight', 0)

        
        current_sequential_pass_level = 0 
        
        # 🟢 NEW: Local State Variables for Sequential Logic (Patch 3)
        first_failed_level_local = None 
        
        is_passed_previous_level = True 
        raw_results_for_sub_seq: List[Dict[str, Any]] = []
        start_ts = time.time() 

        self.logger.info(f"[WORKER START] Assessing Sub-Criteria: {sub_id} - {sub_criteria_name} (Weight: {sub_weight})")

        # รีเซ็ต temp_map_for_save เฉพาะ worker นี้
        self.temp_map_for_save = {}

        # -----------------------------------------------------------
        # 1. LOOP THROUGH LEVELS (L1 → L5) - ประเมินทุก Level เสมอ
        # -----------------------------------------------------------
        for statement_data in sub_criteria.get('levels', []):
            level = statement_data.get('level')
            if level is None or level > self.config.target_level:
                continue
            
            # 🛑 [REVISED LOGIC]: ลบการ Capping ก่อนเรียก LLM เพื่อบังคับรันทุก Level
            
            sequential_chunk_uuids = [] 
            level_result = {}
            level_temp_map: List[Dict[str, Any]] = []

            # (โค้ดเรียก _run_single_assessment ด้วย Retry Policy)
            # L3+ ใช้ RetryPolicy, L1/L2 ใช้ Manual Retry (MAX_L1_ATTEMPTS)
            if level >= 3:
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
                # (โค้ดเรียก _run_single_assessment สำหรับ L1/L2)
                for attempt_num in range(1, MAX_L1_ATTEMPTS + 1):
                    self.logger.info(f"  > Starting assessment for {sub_id} L{level} (Attempt: {attempt_num}/{MAX_L1_ATTEMPTS})...")
                    level_result = self._run_single_assessment(
                        sub_criteria=sub_criteria,
                        statement_data=statement_data,
                        vectorstore_manager=self.vectorstore_manager,
                        sequential_chunk_uuids=sequential_chunk_uuids,
                        attempt=attempt_num
                    )
                    level_temp_map = level_result.get("temp_map_for_level", []) 
                    if level_result.get('is_passed', False):
                        self.logger.info(f"  > L{level} passed on attempt {attempt_num}.")
                        break
                    elif attempt_num < MAX_L1_ATTEMPTS:
                        self.logger.warning(f"  > L{level} failed on attempt {attempt_num}. Retrying...")
                    else:
                        self.logger.error(f"  > L{level} failed all {MAX_L1_ATTEMPTS} attempts.")


            # --- 1.2 PROCESS RESULT AND HANDLE EVIDENCE ---
            result_to_process = level_result or {}
            result_to_process.setdefault("level", level) 
            result_to_process.setdefault("used_chunk_uuids", [])

            is_passed_llm = result_to_process.get('is_passed', False)
            # 🛑 [REVISED LOGIC]: is_passed_final ไม่ควรถูก cap ที่นี่แล้ว
            is_passed_final = is_passed_llm 

            result_to_process['is_passed'] = is_passed_final
            result_to_process['is_capped'] = False # ตั้งเป็น False เสมอ
            result_to_process['is_counted'] = True # ตั้งเป็น True ก่อน แล้วจะถูกเปลี่ยนเป็น False ใน Logic Capping

            # บันทึก evidence ลง temp_map_for_save เฉพาะเมื่อ PASS จริง
            if is_passed_final and level_temp_map and isinstance(level_temp_map, list):
                
                # 📌 Logic การบันทึกหลักฐานและการคำนวณ Strength หลังการประเมิน
                # --- START OF FIX ---
                # 1. ดึงค่า highest_rerank_score
                #    (ค่านี้ถูกส่งมาจาก _run_single_assessment ภายใต้คีย์ 'max_relevant_score')
                highest_rerank = result_to_process.get('max_relevant_score', 0.0)

                # 2. เรียกใช้ฟังก์ชันด้วยชื่อ Argument ที่ถูกต้องตาม Definition:
                max_evi_str_after_save = self._save_level_evidences_and_calculate_strength(
                    level_temp_map=level_temp_map,
                    sub_id=sub_id,
                    level=level,
                    llm_result=result_to_process, 
                    highest_rerank_score=highest_rerank 
                )
                                
                result_to_process['max_evidence_strength_used'] = max_evi_str_after_save
                
                result_to_process['evidence_strength'] = round(
                    min(max_evi_str_after_save, 10.0) if is_passed_final else 0.0, 1
                )
                
            # 🟢 NEW LOGIC: Update Sequential State (Patch 3 Logic)
            if first_failed_level_local is not None:
                # 💡 Level ที่ตามมาทั้งหมดเป็น GAP_ONLY
                result_to_process["evaluation_mode"] = "GAP_ONLY"
                result_to_process["is_counted"] = False
                result_to_process["is_passed"] = False # ถือว่าไม่ผ่านสำหรับคะแนนรวม
                result_to_process["cap_reason"] = (
                    f"Gap analysis after sequential fail at L{first_failed_level_local}"
                )
                self.logger.info(f"  > L{level} marked as GAP_ONLY (Fail at L{first_failed_level_local}).")
            
            elif not is_passed_final and first_failed_level_local is None:
                # 1. ตรวจสอบการ Fail ครั้งแรก (ถ้ายังไม่เคย Fail)
                first_failed_level_local = level
                # 💡 Level นี้ถือว่า Fail และเป็นจุดเริ่มต้นของการ Capping
                self.logger.info(f"  > 🛑 First Sequential FAIL detected at L{level}. (Setting first_failed_level_local={level})")
            
            elif is_passed_final:
                # 2. อัปเดต Level ที่ผ่านสูงสุด (ถ้าเป็น Level ถัดไป)
                if level == current_sequential_pass_level + 1:
                    current_sequential_pass_level = level
                    self.logger.info(f"  > Sequential PASS L{level}. Current Highest Pass: L{current_sequential_pass_level}")
                else:
                    # กรณีที่เกิดการข้าม Level (เช่น L1 ผ่าน, L3 ถูกรัน แต่ L2 ถูก Skip)
                    # 🛑 [REVISED LOGIC]: ใช้ Logic นี้เฉพาะในโหมด SEQUENTIAL เท่านั้น 
                    # ในโหมด MIXED/PARALLEL ปล่อยให้มันผ่านไป
                    if self.is_sequential: # ตรวจสอบว่าเป็นโหมด sequential จริงๆ
                        self.logger.warning(f"  > Sequential BREAK detected at L{level}. Capping at {current_sequential_pass_level}")
                        first_failed_level_local = current_sequential_pass_level + 1 # Force cap ที่ Level ที่ถูกข้าม/ไม่ผ่าน
                        result_to_process["is_counted"] = False
                        result_to_process["is_passed"] = False


            # เพิ่มลง raw results
            result_to_process["execution_index"] = len(raw_results_for_sub_seq)
            raw_results_for_sub_seq.append(result_to_process)
        
        # -----------------------------------------------------------
        # 2. CALCULATE SUMMARY
        # -----------------------------------------------------------
        # 📌 highest_full_level คือ Level ที่ผ่านต่อเนื่องสูงสุด (current_sequential_pass_level)
        highest_full_level = current_sequential_pass_level

        # (โค้ด _calculate_weighted_score)
        weighted_score = self._calculate_weighted_score(highest_full_level, sub_weight)
        weighted_score = round(weighted_score, 2)

        # 📌 num_passed ต้องนับเฉพาะ Level ที่ "is_counted" ไม่ใช่ False (หรือ is_passed เป็น True ใน Logic เดิม)
        num_passed = sum(1 for r in raw_results_for_sub_seq if r.get("is_passed", False) and r.get("is_counted", True))

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
        
        statements_for_action_plan = []
        
        for r in raw_results_for_sub_seq:
            is_passed = r.get('is_passed', False)
            evidence_strength = r.get('evidence_strength', 10.0)

            # 1. Fail ที่ไม่ได้ถูก Cap (คือ Fail แรก)
            if not is_passed and r.get('evaluation_mode') != "GAP_ONLY":
                r['recommendation_type'] = 'FAILED'
                statements_for_action_plan.append(r)
                continue
            
            # 1.1 Fail ที่เป็น GAP_ONLY (Level ที่ตามมา)
            if r.get('evaluation_mode') == "GAP_ONLY":
                r['recommendation_type'] = 'GAP_ANALYSIS' # ใช้ Recommendation type ใหม่สำหรับ Gap
                statements_for_action_plan.append(r)
                continue

            # 2. ผ่านแต่หลักฐานอ่อน
            if is_passed and evidence_strength < WEAK_EVIDENCE_THRESHOLD: # สมมติ WEAK_EVIDENCE_THRESHOLD = 5.0
                r['recommendation_type'] = 'WEAK_EVIDENCE' 
                statements_for_action_plan.append(r) # ✅ Statement ที่ Pass แต่หลักฐานอ่อน ถูกรวบรวมแล้ว!
        
        action_plan_result = []

        try:
            # 🔥 การเรียกใช้ที่ถูกต้อง:
            action_plan_result = self.create_structured_action_plan( # ใช้ self.create_structured_action_plan
                recommendation_statements=statements_for_action_plan, 
                sub_id=sub_id,
                sub_criteria_name=sub_criteria_name, 
                target_level=target_next_level, 
                llm_executor=self.llm,
                ActionPlanActions=self.ActionPlanActions
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
        # 💡 Logic การดึงหลักฐานเพื่อส่งคืน Main Process
        if self.is_sequential or self.is_parallel_all_mode: # ในโหมด Parallel ก็ต้องส่ง map ที่ใช้กลับไปด้วย
            for key in self.evidence_map:
                # ดึงเฉพาะหลักฐานที่ตรงกับ Sub-Criteria นี้เท่านั้น
                if key.startswith(sub_criteria['sub_id'] + "."):
                    final_temp_map[key] = self.evidence_map[key]
        else:
            # โหมด Mixed (หรือโหมดเก่า)
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
        # 📌 Map Key ที่ใช้สำหรับเก็บ Evidence ใน self.evidence_map และ self.temp_map_for_save
        map_key = f"{sub_id}.L{level}"
        new_evidence_list: List[Dict[str, Any]] = []
        
        # 1. วนซ้ำหลักฐานที่ใช้ในการประเมิน
        for chunk in level_temp_map:
            
            # 🎯 CRITICAL FIX 25.0: ดึง Chunk UUID และ Stable Doc ID แยกจากกันอย่างชัดเจน
            chunk_uuid_key = chunk.get("chunk_uuid") 
            stable_doc_uuid_key = chunk.get("stable_doc_uuid") or chunk.get("doc_id")

            # Fallback Logic: ถ้า Chunk UUID หาย ให้ใช้ Stable Doc UUID แทน (เพื่อไม่ให้ entry ว่าง)
            if not chunk_uuid_key and stable_doc_uuid_key:
                chunk_uuid_key = stable_doc_uuid_key 
                self.logger.warning(f"⚠️ [EVI SAVE] Missing chunk_uuid. Falling back to Stable ID: {chunk_uuid_key[:8]}...")

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
        # 📌 ใช้ self._calculate_evidence_strength_cap เพื่อกำหนดเพดานคะแนน (Cap)
        evi_cap_data = self._calculate_evidence_strength_cap(
            top_evidences=new_evidence_list, # ใช้ List ของ Evidence ที่ถูกกรองและจัดรูปแบบแล้ว
            level=level,
            highest_rerank_score=highest_rerank_score
        )
        
        max_evi_str_for_prompt = evi_cap_data['max_evi_str_for_prompt']

        # 4. บันทึกเข้า Map (ใช้ setdefault เพื่อให้แน่ใจว่า List ถูกสร้างขึ้น)
        # *Note: ในโหมด Worker/Parallel, self.evidence_map จะถูกรวมใน Process หลัก*
        current_map = self.evidence_map.setdefault(map_key, [])
        current_map.extend(new_evidence_list)
        
        # 5. อัปเดต Temp Map (สำหรับ Worker Mode: ใช้สำหรับการส่งผลลัพธ์กลับใน Process)
        temp_map = self.temp_map_for_save.setdefault(map_key, [])
        temp_map.extend(new_evidence_list)
        
        # 6. Log สรุป
        self.logger.info(f"[EVIDENCE SAVED] {map_key} → {len(new_evidence_list)} chunks")
        self.logger.info(f"[SEQUENTIAL UPDATE] {map_key} added to engine's main evidence_map for L{level+1} dependency.")
        
        # คืนค่า max_evi_str_for_prompt เพื่อใช้ในการอัปเดต final_results
        return evi_cap_data['max_evi_str_for_prompt']
        
    def _calculate_evidence_strength_cap(
        self,
        top_evidences: List[Union[Dict[str, Any], Any]],
        level: int,
        highest_rerank_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Relevant Score Gate เวอร์ชัน FINAL: ดึงคะแนนจาก metadata, top-level key/attribute, และ Regex fallback ที่ครอบคลุม
        ฟังก์ชันนี้มีหน้าที่หลักในการกำหนด 'max_evi_str_for_prompt' โดยอิงจาก Rerank Score สูงสุด
        """

        score_keys = [
            "rerank_score", "score", "relevance_score", # จัด rerank_score มาก่อน
            "_rerank_score_force", "_rerank_score", 
            "Score", "RelevanceScore"
        ]
        
        # ─── 1. ดึงค่า Threshold และ Cap จาก Attribute/Fallback (ปรับปรุงการเริ่มต้น) ───
        # 💡 ใช้ Attribute ของ Class เป็นหลัก (ซึ่งถูกตั้งค่าจาก global_vars ใน __init__)
        threshold = getattr(self, "RERANK_THRESHOLD", 0.5) 
        cap_value = getattr(self, "MAX_EVI_STR_CAP", 3.0)
        
        # 💡 Fallback จาก globals() (ในกรณีที่ __init__ ไม่ได้ตั้งค่า)
        if threshold == 0.5:
            threshold = globals().get('RERANK_THRESHOLD', 0.5)
        if cap_value == 3.0:
            cap_value = globals().get('MAX_EVI_STR_CAP', 3.0)

        # 🟢 FIX 1: ใช้ highest_rerank_score เป็นค่าตั้งต้นที่น่าเชื่อถือที่สุด
        max_score_found = highest_rerank_score if highest_rerank_score is not None else 0.0
        max_score_source = "Adaptive_RAG_Loop" if highest_rerank_score is not None else "N/A"


        for doc in top_evidences:
            
            page_content = ""
            metadata = {}
            current_score = 0.0 # รีเซ็ตคะแนนสำหรับแต่ละเอกสาร

            # ─── 2. แปลงเป็น metadata + content เดียวกัน (รองรับทุกโครงสร้าง) ───
            if isinstance(doc, dict):
                # ใช้ doc.get() เพื่อป้องกัน KeyError/AttributeError
                metadata = doc.get("metadata", {}) 
                page_content = doc.get("page_content", "") or doc.get("text", "") or doc.get("content", "")
            else:
                # รองรับ Langchain Document หรือ Object อื่นๆ
                metadata = getattr(doc, "metadata", {})
                page_content = getattr(doc, "page_content", "") or getattr(doc, "text", "")

            # ─── 3. ค้นหาคะแนน (ตรวจสอบ top-level key/attribute และ metadata) ───
            for key in score_keys:
                score_val = None
                
                # ตรวจสอบใน metadata
                score_val = metadata.get(key)
                
                # ตรวจสอบใน doc object/dict
                if score_val is None:
                    if isinstance(doc, dict):
                        score_val = doc.get(key)
                    else:
                        score_val = getattr(doc, key, None)
                
                # แปลงเป็น float
                if score_val is not None:
                    try:
                        temp_score = float(score_val)
                        # ตรวจสอบว่าค่าที่ดึงมาสมเหตุสมผลหรือไม่ (ควรอยู่ระหว่าง 0 ถึง 1.0)
                        if 0.0 < temp_score <= 1.0: 
                            if temp_score > current_score:
                                current_score = temp_score
                                break # หยุดเมื่อพบคะแนนที่ใช้ได้
                    except (ValueError, TypeError):
                        continue
            
            # ─── 4. Fallback: ดึงจากท้าย content (Aggressive Regex) ───
            if current_score == 0.0 and page_content and isinstance(page_content, str):
                try:
                    # 📌 ใช้ re.search ที่ถูก import ใน header
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
                    import re # Ensure re is imported if it's not a class method
                    for pat in patterns:
                        m = re.search(pat, tail, re.IGNORECASE)
                        if m:
                            try:
                                temp_score = float(m.group(1))
                                if 0.0 < temp_score <= 1.0: # ตรวจสอบขอบเขต
                                    if temp_score > current_score:
                                        current_score = temp_score
                                        break
                            except:
                                continue
                except Exception:
                    # กรณี re ไม่ถูก import หรือ error อื่นๆ
                    pass 

            # 🔴 FIX: เพิ่มการตรวจสอบขอบเขตคะแนน (Score Clamp) 
            # (กรณีดึงมาเป็น > 1.0 จาก metadata/top-level key ซึ่งอาจเป็นคะแนนที่ไม่ใช่ 0-1)
            if current_score > 1.0:
                self.logger.warning(f"🚨 Score Clamp L{level}: Detected score {current_score:.4f} > 1.0. Assuming score not in 0-1 range and ignoring.")
                current_score = 0.0 # รีเซ็ต ถ้าเกิน 1.0 ถือว่าน่าจะเป็นคะแนนที่ไม่ใช่ Rerank/Relevance 0-1

            # ─── 5. ดึง source ───
            source = (
                metadata.get("source_filename") or metadata.get("filename") or
                doc.get("source_filename") or doc.get("filename") or 
                doc.get("source") or doc.get("doc_id") or
                "N/A"
            )

            # ─── 6. อัปเดตคะแนนสูงสุด (พร้อม Log เมื่อมีการ Override) ───
            if current_score > max_score_found:
                # 🟢 FIX 2: เพิ่ม Log เมื่อมีการ Override คะแนนจาก Loop
                if highest_rerank_score is not None and current_score > highest_rerank_score:
                    self.logger.critical(f"⚠️ Score Override: Found hidden score {current_score:.4f} > Loop score {highest_rerank_score:.4f} from source: {source}")

                max_score_found = current_score
                max_score_source = source

        # ─── 7. Relevant Score Gate + Log ───
        
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
        

    def _robust_hydrate_documents_for_priority_chunks(
        self, 
        chunks_to_hydrate: List[Dict], 
        vsm: Optional['VectorStoreManager'],
        current_sub_id: Optional[str] = None
    ) -> List[Dict]:
        
        """
        Hydrates priority chunks using robust Stable ID fallback logic
        and guarantees the 'text' key is present in all returned chunks.
        Preserves and reinforces the 'is_baseline' flag.
        """
        
        # 1. กำหนด active_sub_id (Fix 5)
        # ใช้ current_sub_id ก่อน, ตามด้วย self.sub_id, และ Fallback ไปที่ config
        active_sub_id = current_sub_id or getattr(self, 'sub_id', None)
        if not active_sub_id:
            self.logger.error("Hydration failed: Missing sub_id.")
            active_sub_id = getattr(self, 'config', {}).get('sub_id') or 'unknown'
            
        if active_sub_id == 'unknown':
            self.logger.error("FATAL: Hydration cannot proceed without sub_id. Returning unhydrated chunks.")
            return chunks_to_hydrate 
        
        if not chunks_to_hydrate or not vsm:
            return chunks_to_hydrate

        # 1. Collect Stable/Chunk IDs
        stable_ids = set()
        for chunk in chunks_to_hydrate:
            # ใช้ Stable ID หรือ Chunk UUID เป็น ID สำหรับการ Hydration
            sid = chunk.get("stable_doc_uuid") or chunk.get("doc_id") or chunk.get("chunk_uuid") 
            if sid and isinstance(sid, str) and len(sid.replace("-", "")) >= 32:
                stable_ids.add(sid)

        hydrated_priority_docs = []
        restored_count = 0
        total_count = len(chunks_to_hydrate)

        # -------------------- FALLBACK LOGIC 1 (No Valid IDs) --------------------
        if not stable_ids:
            self.logger.info("No Stable IDs found for Priority Chunk hydration. Boosting existing chunks.")
            for chunk in chunks_to_hydrate:
                new_chunk = chunk.copy()
                new_chunk["is_baseline"] = True # Fix 3
                if "text" in new_chunk and new_chunk["text"].strip():
                    # Re-tag PDCA (Fix 5: ใช้ active_sub_id)
                    new_tag = classify_by_keyword(
                        new_chunk["text"],
                        sub_id=active_sub_id, 
                        contextual_rules_map=self.contextual_rules_map
                    )
                    if new_tag != 'Other':
                        new_chunk["pdca_tag"] = new_tag
                    new_chunk["rerank_score"] = max(new_chunk.get("rerank_score", 0.0), 0.95)
                    new_chunk["score"] = max(new_chunk.get("score", 0.0), 0.95)
                hydrated_priority_docs.append(new_chunk)
            
            self.logger.info(f"PRIORITY HYDRATION → Fallback boost used. Final chunks: {len(hydrated_priority_docs)} (all baseline).")
            
            # 🟢 FIX 7: มั่นใจว่ามีคีย์ 'text' ก่อน return
            return self._guarantee_text_key(hydrated_priority_docs)


        # -------------------- HYDRATION LOGIC --------------------
        self.logger.info(f"Attempting to hydrate {len(stable_ids)} documents...")
        stable_id_map = defaultdict(list)
        
        try:
            # 🟢 FIX 6.1: แก้ชื่อฟังก์ชัน VSM
            # get_documents_by_id ควรคืนค่าเป็น Map หรือ List[LcDocument] ที่เราสามารถแปลงเป็น Map ได้
            retrieved_docs = vsm.get_documents_by_id(list(stable_ids), doc_type=self.doc_type, enabler=self.config.enabler)
            
            # แปลงผลลัพธ์จาก LcDocument เป็น Map เพื่อให้เข้ากับ Logic เดิม
            # เราอาจจะต้องกำหนดวิธีการแปลง LcDocument เป็น Dict ที่นี่
            for doc in retrieved_docs:
                sid = doc.metadata.get("stable_doc_uuid") or doc.metadata.get("doc_id")
                if sid:
                    # Store as Dict with 'text' key for consistency
                    stable_id_map[sid].append({
                        "text": doc.page_content,
                        "metadata": doc.metadata
                    })
            
        except Exception as e:
            self.logger.error(f"VSM failed to get documents by ID: {e}. Falling back to boosting only.")
            
            # 🟢 FIX 6.2: VSM Fallback Logic (ใช้โค้ด Fallback 1 ซ้ำ)
            hydrated_priority_docs_fallback = []
            self.logger.info("PRIORITY HYDRATION → VSM Fallback. Boosting existing chunks.")
            for chunk in chunks_to_hydrate:
                new_chunk = chunk.copy()
                new_chunk["is_baseline"] = True
                if "text" in new_chunk and new_chunk["text"].strip():
                    new_tag = classify_by_keyword(
                        new_chunk["text"],
                        sub_id=active_sub_id, 
                        contextual_rules_map=self.contextual_rules_map
                    )
                    if new_tag != 'Other':
                        new_chunk["pdca_tag"] = new_tag
                    new_chunk["rerank_score"] = max(new_chunk.get("rerank_score", 0.0), 0.95)
                    new_chunk["score"] = max(new_chunk.get("score", 0.0), 0.95)
                hydrated_priority_docs_fallback.append(new_chunk)
            
            self.logger.info(f"PRIORITY HYDRATION → Fallback boost used. Final chunks: {len(hydrated_priority_docs_fallback)} (all baseline).")
            
            # 🟢 FIX 7: มั่นใจว่ามีคีย์ 'text' ก่อน return
            return self._guarantee_text_key(hydrated_priority_docs_fallback)
        
        
        # 3. Hydrate + Boost + Re-tag PDCA 
        seen_signatures = set()
        for chunk in chunks_to_hydrate:
            new_chunk = chunk.copy()
            new_chunk["is_baseline"] = True # Fix 3
            
            sid = new_chunk.get("stable_doc_uuid") or new_chunk.get("doc_id")

            hydrated = False
            if sid and sid in stable_id_map and stable_id_map[sid]:
                best_match = stable_id_map[sid][0]
                
                # ดึง Text
                new_chunk["text"] = best_match["text"] 
                # อัปเดต Metadata
                new_chunk.update({k: v for k, v in best_match["metadata"].items()
                                if k not in ["text", "page_content"]})
                
                # รักษา chunk_uuid เดิมไว้ถ้ามี
                original_uuid = new_chunk.get("chunk_uuid")
                if original_uuid:
                    new_chunk["chunk_uuid"] = original_uuid

                restored_count += 1
                hydrated = True
                new_chunk["rerank_score"] = 1.0
                new_chunk["score"] = 1.0

            # Re-tag PDCA from full text
            if "text" in new_chunk and new_chunk["text"].strip():
                new_tag = classify_by_keyword(
                    new_chunk["text"], 
                    sub_id=active_sub_id, # Fix 5
                    contextual_rules_map=self.contextual_rules_map
                )
                if new_tag != 'Other':
                    old_tag = new_chunk.get("pdca_tag", "Other")
                    new_chunk["pdca_tag"] = new_tag
                    if old_tag == 'Other':
                        self.logger.debug(f"Re-tagged priority chunk as '{new_tag}' (from 'Other')")

            # Dedup
            signature = (sid, new_chunk.get("text", "")[:200])
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)

            # Boost ถ้าไม่ hydrated แต่มี text
            if not hydrated and "text" in new_chunk:
                new_chunk["rerank_score"] = max(new_chunk.get("rerank_score", 0.0), 0.95)
                new_chunk["score"] = max(new_chunk.get("score", 0.0), 0.95)

            # Boost น้อยถ้าไม่มี text (หรือ Hydration ล้มเหลว)
            if "text" not in new_chunk:
                self.logger.warning(f"Priority chunk (SID: {sid[:8] if sid else 'N/A'}...) has no text after hydration attempt.")
                new_chunk["rerank_score"] = max(new_chunk.get("rerank_score", 0.0), 0.80)
                new_chunk["score"] = max(new_chunk.get("score", 0.0), 0.80)

            hydrated_priority_docs.append(new_chunk)

        # -------------------- FINAL GUARANTEE (Fix 7) --------------------
        return self._guarantee_text_key(hydrated_priority_docs, total_count=total_count, restored_count=restored_count)

    def _guarantee_text_key(self, chunks: List[Dict], total_count: int = 0, restored_count: int = 0) -> List[Dict]:
        """Ensures every chunk has a 'text' key to prevent downstream KeyError."""
        final_chunks_guaranteed = []
        guaranteed_count = 0
        for chunk in chunks:
            if 'text' not in chunk:
                chunk['text'] = ""
                guaranteed_count += 1
                self.logger.debug(f"Guaranteed 'text' key for chunk (ID: {chunk.get('chunk_uuid', 'N/A')[:8]})")
            final_chunks_guaranteed.append(chunk)

        baseline_count = len([d for d in final_chunks_guaranteed if d.get('is_baseline', False)])
        
        # ถ้าถูกเรียกจาก Hydration Loop จะแสดง Log สรุป
        if total_count > 0:
            self.logger.info(
                f"PRIORITY HYDRATION SUMMARY → Restored {restored_count}/{total_count} chunks with full text. "
                f"Final priority chunks: {len(final_chunks_guaranteed)} ({baseline_count} marked as baseline). Guaranteed 'text' key for {guaranteed_count} chunks."
            )
        
        return final_chunks_guaranteed
    
    def _get_keywords_for_phase(self, sub_id: str, level: int, phase: str = "Plan") -> str:
        """
        ดึง keywords สำหรับ phase (Plan, Do, Check, Act) จาก contextual_rules.json
        ลำดับความสำคัญ: level-specific → sub-specific → enabler default → fallback
        """
        keywords = []
        sub_rules = self.contextual_rules_map.get(sub_id, {})
        
        phase_key = f"{phase.lower()}_keywords"
        
        # 1. Level-specific (L1, L2, ...)
        level_key = f"L{level}"
        level_rules = sub_rules.get(level_key, {})
        if phase_key in level_rules:
            keywords = level_rules[phase_key]
        
        # 2. Sub-specific fallback
        elif phase_key in sub_rules:
            keywords = sub_rules[phase_key]
        
        # 3. Enabler default
        elif "_enabler_defaults" in self.contextual_rules_map:
            default_keywords = self.contextual_rules_map["_enabler_defaults"].get(phase_key)
            if default_keywords:
                keywords = default_keywords
        
        # 4. Global fallback (ถ้าไม่มีเลย)
        if not keywords:
            fallback_map = {
                "plan": ["วิสัยทัศน์", "นโยบาย", "ทิศทาง", "เป้าหมาย", "ยุทธศาสตร์", "แผน"],
                "do": ["ดำเนินการ", "ปฏิบัติ", "จัดกิจกรรม", "โครงการ", "ขับเคลื่อน"],
                "check": ["ตรวจสอบ", "ประเมิน", "ติดตาม", "วัดผล", "ตรวจสอบ"],
                "act": ["ปรับปรุง", "พัฒนา", "แก้ไข", "นำไปใช้", "ปรับเปลี่ยน"]
            }
            keywords = fallback_map.get(phase.lower(), [])
        
        return ", ".join(keywords)


    def _get_pdca_blocks_from_evidences(
            self, 
            evidences: List[Dict], 
            baseline_evidences: Dict[str, List[Dict]], 
            level: int,
            sub_id: str,
            contextual_rules_map: Dict[str, Any]
        ) -> Tuple[str, str, str, str, str]:
            """
            Groups evidences by PDCA tags and formats them into context blocks
            for the LLM prompt.

            Args:
                evidences: The list of all relevant evidence chunks (new and previous levels).
                baseline_evidences: Dictionary of baseline evidences collected from previous levels.
                level: The current assessment level (L1-L5).
                sub_id: The ID of the sub-criteria being assessed.
                contextual_rules_map: Rules for keyword classification.

            Returns:
                A tuple containing (plan_blocks, do_blocks, check_blocks, act_blocks, other_blocks)
                as formatted strings.
            """
            
            # 🟢 FIX 9: รับประกันคีย์ 'text' ทันทีที่เข้าเมธอด 
            # (สมมติว่า self._guarantee_text_key มีอยู่ในคลาสหลัก)
            evidences = self._guarantee_text_key(evidences)
            
            # 🟢 NEW FIX 12: กรอง Chunk ที่ไม่มี Text ออกจากรายการ Evidences ทันที
            original_count = len(evidences)
            evidences = [
                chunk for chunk in evidences 
                if chunk.get("text", "").strip() # เก็บเฉพาะ chunk ที่ text ไม่ใช่ string ว่าง/space
            ]
            if len(evidences) < original_count:
                # 💡 ใช้ self.logger.debug แทน self.logger.info
                self.logger.debug(f"PDCA Pre-Filter: Removed {original_count - len(evidences)} chunks with empty text.")

            # ------------------ Local Helper Function: _create_block ------------------
            def _create_block(tag: str, chunks: List[Dict]) -> str:
                """Helper to format a list of chunks into a single text block."""
                if not chunks:
                    return ""

                block_content = []
                sorted_chunks = sorted(chunks, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
                
                for i, chunk in enumerate(sorted_chunks):
                    # 🎯 รวม Header และ Content ใน Block เดียวกัน
                    header = f"### [{tag} {i+1}/{len(sorted_chunks)}]"
                    content = chunk['text'].strip() # ใช้ strip() เพื่อความสะอาด
                    
                    block_content.append(f"{header}\n{content}")
                
                # ใช้ \n---\n เป็นตัวแบ่ง Block
                return "\n---\n".join(block_content)
            # --------------------------------------------------------------------------

            # 1. Group evidences by PDCA tag (using the consolidated list)
            pdca_groups = defaultdict(list)
            
            # 2. Tag/Re-tag and Group the evidences
            for chunk in evidences:
                # ใช้ classify_by_keyword เพื่อ re-tag หาก LLM ไม่ได้ให้ tag มา หรือต้องการยืนยัน
                # (สมมติว่า classify_by_keyword ถูก Import หรือเป็น method ของ self)
                if 'pdca_tag' not in chunk or not chunk['pdca_tag'].strip() or chunk['pdca_tag'] == 'Other':
                    new_tag = classify_by_keyword(
                        chunk.get("text", ""), 
                        sub_id=sub_id, 
                        contextual_rules_map=contextual_rules_map
                    )
                    chunk['pdca_tag'] = new_tag
                
                tag = chunk['pdca_tag']
                
                # ⭐ CRITICAL: รวม Baseline Evidences จาก level ก่อนหน้าเข้าไปในกลุ่ม PDCA 
                # (พวกนี้ควรถูก re-tag แล้วใน _collect_previous_level_evidences)
                
                if tag in ["Plan", "Do", "Check", "Act", "Other"]:
                    pdca_groups[tag].append(chunk)

            # 3. Create blocks from grouped evidences
            plan_blocks = _create_block("Plan", pdca_groups["Plan"])
            do_blocks = _create_block("Do", pdca_groups["Do"])
            check_blocks = _create_block("Check", pdca_groups["Check"])
            act_blocks = _create_block("Act", pdca_groups["Act"])
            other_blocks = _create_block("Other", pdca_groups["Other"])

            return plan_blocks, do_blocks, check_blocks, act_blocks, other_blocks
                
    def run_assessment(
        self,
        target_sub_id: str = "all",
        export: bool = False,
        vectorstore_manager: Optional['VectorStoreManager'] = None,
        sequential: bool = False,
        document_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Main runner ของ Assessment Engine
        รองรับทั้ง Parallel และ Sequential 100%
        บันทึก Evidence Map เสมอ (แม้รัน sub เดียว)
        """
        start_ts = time.time()
        self.is_sequential = sequential

        # ============================== 1. Filter Rubric ==============================
        if target_sub_id.lower() == "all":
            sub_criteria_list = self._flatten_rubric_to_statements()
        else:
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

        # --------------------- กำหนด Max Workers ---------------------
        max_workers = globals().get('MAX_PARALLEL_WORKERS', 4)
        if not isinstance(max_workers, int) or max_workers <= 0:
            max_workers = 4
            self.logger.warning(f"Invalid MAX_PARALLEL_WORKERS in config → using safe default: {max_workers}")
        self.logger.info(f"Using max_workers = {max_workers}")

        run_parallel = (target_sub_id.lower() == "all") and not sequential

        # ============================== 2. Run Assessment ==============================
        if run_parallel:
            self.logger.info(f"Starting Parallel Assessment with {max_workers} processes")
            worker_args = [(
                sub_data,
                self.config.enabler,
                self.config.target_level,
                self.config.mock_mode,
                self.evidence_map_path,
                self.config.model_name,
                self.config.temperature,
                getattr(self.config, 'MIN_RETRY_SCORE', 0.50),
                getattr(self.config, 'MAX_RETRIEVAL_ATTEMPTS', 3),
                document_map or self.document_map,
                ActionPlanActions  # <--- เพิ่มตัวนี้เป็น element สุดท้าย
            ) for sub_data in sub_criteria_list]

            try:
                with multiprocessing.get_context('spawn').Pool(processes=max_workers) as pool:
                    results_list = pool.map(_static_worker_process, worker_args)
            except Exception as e:
                self.logger.critical(f"Multiprocessing failed: {e}")
                raise

            for result_tuple in results_list:
                if not isinstance(result_tuple, tuple) or len(result_tuple) != 2:
                    continue
                sub_result, temp_map_from_worker = result_tuple

                if isinstance(temp_map_from_worker, dict):
                    for level_key, evidence_list in temp_map_from_worker.items():
                        current_list = self.evidence_map.setdefault(level_key, [])
                        current_list.extend(evidence_list)

                raw_refs = sub_result.get("raw_results_ref", [])
                self.raw_llm_results.extend(raw_refs if isinstance(raw_refs, list) else [])
                self.final_subcriteria_results.append(sub_result)

        else:
            # --------------------- SEQUENTIAL MODE ---------------------
            mode_desc = target_sub_id if target_sub_id != "all" else "All Sub-Criteria"
            self.logger.info(f"Starting Sequential Assessment: {mode_desc}")

            local_vsm = vectorstore_manager or (
                load_all_vectorstores(
                    doc_types=[EVIDENCE_DOC_TYPES],
                    enabler_filter=self.config.enabler,
                    tenant=self.config.tenant,
                    year=self.config.year,
                ) if self.config.mock_mode == "none" else None
            )
            self.vectorstore_manager = local_vsm

            if self.vectorstore_manager:
                self.vectorstore_manager.logger = self.logger
                self.logger.info("Assigned Engine logger to VectorStoreManager")

            for sub_criteria in sub_criteria_list:
                sub_result, final_temp_map = self._run_sub_criteria_assessment_worker(sub_criteria)
                self.raw_llm_results.extend(sub_result.get("raw_results_ref", []))
                self.final_subcriteria_results.append(sub_result)

        # ============================== 3. บันทึก Evidence Map (บังคับบันทึกเสมอ) ==============================
        if self.evidence_map:
            self.logger.info(f"[EVIDENCE] Attempting to persist Evidence Map → {self.evidence_map_path}")
            try:
                self._save_evidence_map(map_to_save=self.evidence_map)
                total_items = sum(len(v) for v in self.evidence_map.values() if isinstance(v, list))
                self.logger.info(
                    f"Evidence Map SAVED SUCCESSFULLY | Keys: {len(self.evidence_map)} | "
                    f"Items: {total_items} | Path: {self.evidence_map_path}"
                )
            except Exception as e:
                self.logger.critical(f"FAILED TO SAVE Evidence Map → {self.evidence_map_path}")
                self.logger.exception(e)
        else:
            self.logger.warning("[EVIDENCE] Evidence Map is EMPTY → Nothing to save")

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
            final_results["evidence_map_snapshot"] = deepcopy(self.evidence_map)

        return final_results

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
        *FIXED: Adaptive Filtering Fallback Logic, PDCA Gate & Weighted Strength*
        """
    
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

        planning_keywords = self._get_keywords_for_phase(sub_id, level, "Plan")
        # 📌 NEW: ตรวจสอบ Contextual Rule ที่เกี่ยวข้อง
        # ดึง Rule ที่มี level และ sub_id ตรงกัน (ถ้ามี)
        contextual_rule = self._get_applicable_contextual_rule(sub_id, level)

        # ==================== 2. Hybrid Retrieval Setup ====================
        mapped_stable_doc_ids, priority_docs_unhydrated = self._get_mapped_uuids_and_priority_chunks(
            sub_id=sub_id,
            level=level,
            statement_text=statement_text,
            level_constraint=level_constraint,
            vectorstore_manager=vectorstore_manager
        )

        priority_docs = self._robust_hydrate_documents_for_priority_chunks(
            chunks_to_hydrate=priority_docs_unhydrated,
            vsm=vectorstore_manager,
            current_sub_id=sub_id
        )

        # ==================== 3. Enhance Query ====================
        rag_query_list = self.enhance_query_for_statement( 
            statement_text=statement_text,
            sub_id=sub_id,
            statement_id=statement_id,
            level=level,
            focus_hint=level_constraint,
            # ไม่ต้องส่ง enabler_id และ contextual_map เพราะถูกเข้าถึงผ่าน self แล้ว
        )
        rag_query = rag_query_list[0] if rag_query_list else statement_text

        # ==================== 4. LLM Evaluator ====================
        llm_evaluator_to_use = evaluate_with_llm_low_level if level <= 2 else self.llm_evaluator

        # ==================== 5. ADAPTIVE RAG LOOP (FIXED: Filter Score 0.0000) ====================
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
            
            # 🟢 FIX: ถ้ามีการค้นพบ Rerank Score ที่ไม่ใช่ 0.0 ให้กรอง Chunks ที่ Score 0.0000 ออก (มาจาก Priority Chunks ที่ถูกรวมมา)
            if current_max_score > 0.0001: 
                original_count = len(top_evidences_current)
                top_evidences_current = [
                    ev for ev in top_evidences_current 
                    if (ev.get("rerank_score") or ev.get("score", 0.0)) > 0.0001
                ]
                if len(top_evidences_current) < original_count:
                    # ✅ แก้ไข: เปลี่ยน INFO เป็น DEBUG
                    self.logger.debug(f"Filtered out {original_count - len(top_evidences_current)} chunks with Score 0.0000 (assumed Priority Chunks).")
            # -----------------------------------------------------------------------------------------------------------------------------------


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
                rag_query = f"หลักฐานเพิ่มเติมสำหรับ {statement_text} ในบริบท {level_constraint}"

            loop_attempt += 1

        retrieval_duration = time.time() - retrieval_start
        top_evidences = final_top_evidences 

        # ==================== 6. Adaptive Filtering ====================
        filtered = []
        original_top_evidences = top_evidences 

        for doc in original_top_evidences:
            score = doc.get('rerank_score', doc.get('score', 0.0))
            is_baseline = doc.get('is_baseline', False)  # ⭐ Check Baseline Flag
            doc_id = doc.get('chunk_uuid') or doc.get('doc_id') or 'UNKNOWN'
            
            # 🟢 FIX 2: ถ้าเป็น Baseline Evidence → ผ่านทันที (ไม่เช็ค Threshold)
            if is_baseline:
                filtered.append(doc)
                self.logger.debug(f"✅ Baseline chunk kept (ID: {doc_id[:8]}...) | Score {score:.4f}")
                continue
            
            # ถ้าไม่ใช่ Baseline → เช็ค Score ตาม Threshold ปกติ
            if score >= globals()['MIN_RERANK_SCORE_TO_KEEP']:
                filtered.append(doc)
            else:
                # ✅ แก้ไข: เปลี่ยน INFO เป็น DEBUG
                self.logger.debug(f"Filtering out chunk (ID: {doc_id[:8]}...) | Score {score:.4f}")

        # Fallback Logic ถ้ากรองหมด
        if not filtered and original_top_evidences:
            # 🟢 เก็บไว้เป็น WARNING เนื่องจากเป็นเหตุการณ์ที่ควรแจ้งเตือน
            self.logger.warning(
                f"  > (L{level}) Adaptive Filtering removed all chunks. "
                f"Using all {len(original_top_evidences)} original chunks for PDCA grouping (Fallback)."
            )
            top_evidences = original_top_evidences
        elif not filtered and not original_top_evidences:
            top_evidences = [] 
        else:
            top_evidences = filtered 

        # ✅ แก้ไข: เปลี่ยน INFO เป็น DEBUG
        self.logger.debug(f"Adaptive Filter L{level}: Kept {len(top_evidences)}/{len(original_top_evidences)} chunks "
                        f"({len([d for d in top_evidences if d.get('is_baseline')])} baseline)")

        # ==================== 6.5. 🟢 FIX 4: Robust Hydration for Filtered Chunks ====================
        if top_evidences and vectorstore_manager:
            # ✅ แก้ไข: เปลี่ยน INFO เป็น DEBUG
            self.logger.debug(f"FIX 4: Running Robust Hydration for {len(top_evidences)} filtered chunks (New + Baseline)...")
            
            # **CRITICAL FIX**: ใช้ฟังก์ชันที่มีอยู่เพื่อ Hydrate
            top_evidences = self._robust_hydrate_documents_for_priority_chunks(
                chunks_to_hydrate=top_evidences,
                vsm=vectorstore_manager
            )
            
            # ✅ แก้ไข: เปลี่ยน INFO เป็น DEBUG
            self.logger.debug(f"FIX 4: Hydration complete. Final chunks with text: {len([c for c in top_evidences if 'text' in c and c['text'].strip()])}")

        # ==================== 7. Baseline from Previous Levels ====================
        # 🎯 FIX 1: Initial Value ต้องเป็น Dictionary เปล่า {} เสมอ
        previous_levels_evidence_dict = {} 
        # List สำหรับการส่งเข้า build_multichannel_context_for_level
        previous_levels_evidence_list = [] 

        if level > 1 and not self.is_parallel_all_mode:
            # _collect_previous_level_evidences คืนค่าเป็น Dict[str, List[Dict]]
            previous_levels_evidence_dict = self._collect_previous_level_evidences(sub_id, current_level=level)
        
        # รวบรวมหลักฐานทั้งหมดจาก Dictionary เข้า List สำหรับ Summary Context
        for lst in previous_levels_evidence_dict.values():
            previous_levels_evidence_list.extend(lst)

        # ==================== 8. Build Context ====================
        # 🟢 FIX 10.1: รวมหลักฐานใหม่ (top_evidences) กับหลักฐาน Level ก่อนหน้า
        # NOTE: top_evidences มาจากผลลัพธ์ของ _robust_hydrate_documents_for_priority_chunks
        total_evidences = top_evidences + previous_levels_evidence_list 

        # 🟢 FIX 10.2: รับประกันคีย์ 'text' สำหรับหลักฐานทั้งหมดที่ถูกรวม (รวมทั้งหลักฐานเก่า L1)
        total_evidences = self._guarantee_text_key(total_evidences) 
        
        plan_blocks, do_blocks, check_blocks, act_blocks, other_blocks = self._get_pdca_blocks_from_evidences(
            total_evidences,
            # 🎯 FIX 2: ส่ง Dictionary เข้าไปเพื่อให้ .values() ในฟังก์ชันถัดไปไม่เกิด Error
            baseline_evidences=previous_levels_evidence_dict, 
            level=level,
            sub_id=sub_id,
            contextual_rules_map=self.contextual_rules_map
        )

        direct_context = "\n\n".join(filter(None, [
            plan_blocks,
            do_blocks,
            check_blocks,
            act_blocks,
            other_blocks
        ]))

        # กำหนด Context Length Limit สำหรับ L3 ขึ้นไป
        max_context_length = None
        if level >= 3:
            max_context_length = CONTEXT_CAP_L3_PLUS 
            self.logger.info(f"Context Cap set for L{level}: {max_context_length} characters.")

        # --- CRITICAL C/A EVIDENCE SUMMARY ---
        critical_evidence_summary = ""
        if level >= 2:
            # CRITICAL_SCORE_THRESHOLD = globals()['CRITICAL_CA_THRESHOLD'] 
            
            critical_chunks = [
                doc for doc in top_evidences 
                if doc.get('pdca_tag') in ['Check', 'Act'] and doc.get('rerank_score', 0.0) >= CRITICAL_CA_THRESHOLD
            ]
            
            if critical_chunks:
                self.logger.critical(f"Found {len(critical_chunks)} CRITICAL C/A chunks (Score >= {CRITICAL_CA_THRESHOLD}) for L{level}.")
                summary_text = "\n".join([
                    f"- [{doc['pdca_tag']} | Score: {doc.get('rerank_score'):.4f}] {doc['text'][:180].strip()}..." 
                    for doc in critical_chunks
                ])
                critical_evidence_summary = f"--- CRITICAL C/A EVIDENCE (SCORE > {CRITICAL_CA_THRESHOLD}) ---\n{summary_text}"
            else:
                self.logger.info(f"No CRITICAL C/A chunks found (Score < {CRITICAL_CA_THRESHOLD}) for L{level}.")
                
        # 🟢 ตัวแปร aux_summary และ baseline_summary ถูกกำหนดที่นี่ 
        channels = build_multichannel_context_for_level(
            level=level,
            top_evidences=top_evidences,
            # 🎯 FIX 3: ส่ง List เข้าไปใน build_multichannel_context_for_level
            previous_levels_evidence=previous_levels_evidence_list, 
            max_main_context_tokens=3000,
            max_summary_sentences=4,
            max_context_length=max_context_length
        )
        aux_summary = channels.get('aux_summary', 'ไม่มีหลักฐานรอง')
        baseline_summary = channels.get('baseline_summary', 'ไม่มี')

        # 🟢 ตัวแปร final_llm_context ถูกกำหนดที่นี่
        final_llm_context = "\n\n".join(filter(None, [
            f"--- DIRECT EVIDENCE (L{level} | PDCA Structured)---\n{direct_context}",
            critical_evidence_summary, 
            f"--- AUXILIARY EVIDENCE SUMMARY ---\n{aux_summary}",
            f"--- BASELINE FROM PREVIOUS LEVELS SUMMARY ---\n{baseline_summary}"
        ]))

        if not final_llm_context.strip():
            final_llm_context = "--- ไม่พบหลักฐานที่เกี่ยวข้อง ---"
            self.logger.warning(f"No context generated for {sub_id} L{level}")
        elif max_context_length and len(final_llm_context) > max_context_length:
            self.logger.warning(f"Final LLM Context for L{level} still exceeded Cap. Length: {len(final_llm_context)} (Cap: {max_context_length})")

        self.logger.debug(f"--- LLM CONTEXT (L{level}) --- \n{final_llm_context}")

        # ==================== 9. Evidence Strength Calculation & PDCA Gate ====================
        
        # --- PDCA Completeness Check --- (Patch 1: PDCA Gate)
        available_tags = set()
        if plan_blocks.strip(): available_tags.add("P")
        if do_blocks.strip(): available_tags.add("D")
        if check_blocks.strip(): available_tags.add("C")
        if act_blocks.strip(): available_tags.add("A")

        missing_tags = REQUIRED_PDCA.get(level, set()) - available_tags
        
        # ตั้งค่าเริ่มต้น
        ai_confidence = "HIGH"
        is_hard_fail_pdca = False
        
        # (เรียกใช้ Logic เดิม เพื่อให้ได้ max_evi_str_for_prompt เบื้องต้นจาก Rerank)
        evi_cap_data = self._calculate_evidence_strength_cap(
            top_evidences=top_evidences,
            level=level,
            highest_rerank_score=highest_rerank_score
        )
        max_evi_str_for_prompt = evi_cap_data['max_evi_str_for_prompt']

        # ----------------------------------------------------------------------------------------------------
        # 🟢 NEW: CONTEXTUAL RULE HARD FAIL OVERRIDE CHECK
        # ----------------------------------------------------------------------------------------------------
        is_contextual_override_active = False
        
        # 🎯 Check 1: ตรวจสอบว่ามี Contextual Rule ที่เกี่ยวข้องหรือไม่
        if contextual_rule and globals().get('ENABLE_CONTEXTUAL_RULE_OVERRIDE', False):
            # 🎯 Check 2: ตรวจสอบว่า Rule นี้เป็นประเภท OVERRIDE_ON_PDCA_GAP หรือไม่
            if contextual_rule['rule_type'] == 'SCORE_OVERRIDE_ON_PDCA_GAP':
                
                # 🎯 Check 3: ตรวจสอบเงื่อนไขของ Rule ด้วยหลักฐานปัจจุบัน
                if self._check_contextual_rule_condition(contextual_rule['condition'], sub_id, level, previous_levels_evidence_dict, top_evidences):
                    
                    self.logger.critical(f"  > CONTEXTUAL OVERRIDE L{level}: Rule '{contextual_rule['name']}' matched conditions (PDCA Gap Bypass).")
                    is_contextual_override_active = True
        
        # ----------------------------------------------------------------------------------------------------

        if missing_tags:
            self.logger.warning(
                f"  > PDCA INCOMPLETE for L{level} | Missing: {sorted(missing_tags)} | "
                f"Forcing PARTIAL evidence strength & confidence"
            )
            # Force partial strength cap
            max_evi_str_for_prompt = min(max_evi_str_for_prompt, 3.0)
            ai_confidence = "LOW"
            
            # Hard rule for L3+ (strict SE-AM)
            if level >= 3 and (("C" in missing_tags) or ("A" in missing_tags)):
                if not is_contextual_override_active:
                    # 🔴 ทำ Hard Fail ถ้า Rule ไม่ Override
                    self.logger.critical(f"  > HARD FAIL L{level}: Missing critical closed-loop PDCA phase(s): {missing_tags} - Skipping LLM call.")
                    is_hard_fail_pdca = True
                else:
                    # 🟢 Bypass Hard Fail แต่ใช้ Logic การให้คะแนนจาก Rule แทน
                    self.logger.warning(f"  > HARD FAIL AVOIDED: Contextual Rule Bypassed PDCA Hard Fail Logic.")
                    # ไม่ตั้ง is_hard_fail_pdca = True เพื่อให้โค้ดรันต่อได้ แต่เราจะใช้ Rule Score แทน
                
        else:
            # ✅ แก้ไข: เปลี่ยน INFO เป็น DEBUG
            self.logger.debug(f"  > PDCA COMPLETE for L{level}: {sorted(available_tags)}")


        # --- Weighted Evidence Strength --- (Patch 2: Weighted Strength)
        if level in REQUIRED_PDCA and available_tags:
            pdca_completeness_ratio = len(available_tags) / len(REQUIRED_PDCA.get(level))
            
            # น้ำหนักของ PDCA (0.6 สำหรับ L3+ และ 0.3 สำหรับ L1/L2)
            weight_pdca = 0.6 if level >= 3 else 0.3
            
            # highest_rerank_score ต้องเป็นตัวแปรที่มีค่า Rerank Score สูงสุด
            adjusted_score = (
                (1 - weight_pdca) * highest_rerank_score 
                + weight_pdca * pdca_completeness_ratio
            )
            
            # ใช้ adjusted_score ในการกำหนด Evidence Strength ขั้นสุดท้ายที่ส่งเข้า Prompt (ไม่เกิน 10.0)
            # และใช้ min กับ max_evi_str_for_prompt ที่อาจถูก cap 3.0 จาก Hard Fail ก่อนหน้า
            final_strength_from_weighted = min(adjusted_score * 10.0, 10.0)
            max_evi_str_for_prompt = min(max_evi_str_for_prompt, final_strength_from_weighted)
            
            # ✅ แก้ไข: เปลี่ยน INFO เป็น DEBUG
            self.logger.debug(
                f"  > Weighted Evidence Score L{level}: Rerank={highest_rerank_score:.2f} "
                f"| PDCA Ratio={pdca_completeness_ratio:.2f} | Final Strength={max_evi_str_for_prompt:.1f}"
            )
        
        # ==================== 10. LLM Evaluation ====================
        llm_start = time.time()
        llm_result = {}
        is_passed_llm = False # ตั้งค่าเริ่มต้น

        if not is_hard_fail_pdca:
            try:
                llm_kwargs = {
                    "context": final_llm_context,
                    "sub_criteria_name": sub_criteria_name,
                    "level": level,
                    "statement_text": statement_text,
                    "sub_id": sub_id,
                    "pdca_phase": pdca_phase,
                    "level_constraint": level_constraint,
                    "must_include_keywords": must_include_keywords,
                    "avoid_keywords": avoid_keywords,
                    "max_rerank_score": highest_rerank_score,
                    "max_evidence_strength": max_evi_str_for_prompt, # ใช้ค่าที่ถูกปรับแล้ว
                    "llm_executor": self.llm,
                    "contextual_rules_map": self.contextual_rules_map,
                    "enabler_id": self.config.enabler,
                    "ai_confidence": ai_confidence # ใช้ค่าที่ถูกปรับแล้ว
                }

                if level <= 2:
                    llm_kwargs["planning_keywords"] = planning_keywords

                llm_result = llm_evaluator_to_use(**llm_kwargs)
                is_passed_llm = llm_result.get('is_passed', False)
            except Exception as e:
                self.logger.error(f"LLM Call failed: {e}")
                llm_result = {}
                is_passed_llm = False
        else:
            self.logger.info(f"  > Skipping LLM call for L{level} due to PDCA Hard Fail.")

        llm_duration = time.time() - llm_start

        # ==================== 11. Post-Processing & Scoring (STRICT PDCA OVERRIDE APPLIED) ====================
        if not isinstance(llm_result, dict):
            llm_result = {}

        llm_result = post_process_llm_result(llm_result, level)

        # 📌 ปรับ Final Correction ให้รองรับ Hard Fail
        if is_hard_fail_pdca:
            # ไม่มีการเรียก LLM, score ต้องเป็น 0 ทันที
            final_score = 0.0
            is_passed = False
            final_pdca_breakdown = {'P': 0.0, 'D': 0.0, 'C': 0.0, 'A': 0.0}
            self.logger.warning(f"  > L{level} Final Score is 0.0 due to PDCA HARD FAIL.")
        
        # 🟢 NEW: CONTEXTUAL OVERRIDE SCORING
        elif is_contextual_override_active:
            action = contextual_rule['action']
            if action.get('bypass_hard_fail', False):
                
                # ใช้คะแนนที่ถูกกำหนดไว้ใน Rule แทน LLM
                final_pdca_breakdown = action.get('force_llm_input_scores', {'P': 0.0, 'D': 0.0, 'C': 0.0, 'A': 0.0})
                
                # กำหนด Reason/Extraction จาก Rule Comment
                llm_result['reason'] = action.get('comment', 'Bypassed by Contextual Rule due to PDCA Gap.')
                llm_result['extraction_c'] = f"OVERRIDE: {final_pdca_breakdown.get('C', 0.0)}"
                llm_result['extraction_a'] = f"OVERRIDE: {final_pdca_breakdown.get('A', 0.0)}"

                final_score = sum(final_pdca_breakdown.values())
                required_score_for_level = get_correct_pdca_required_score(level)
                is_passed = final_score >= required_score_for_level

                self.logger.critical(f"  > L{level} Contextual Override Success. Score: {final_score:.1f} (Rule: {final_pdca_breakdown})")

        # -------------------------------------------------------------------------------------------

        else:
            # ใช้ Logic เดิม (Post-process และ Override C/A Score จาก LLM)
            final_pdca_breakdown = llm_result.get('pdca_breakdown', {})
            
            C_KEYWORDS = BASE_PDCA_KEYWORDS['Check']
            A_KEYWORDS = BASE_PDCA_KEYWORDS['Act']
            
            # --- C/A SCORE OVERRIDE (FORCE UP) ---
            
            # 1. การตรวจสอบและบังคับคะแนน C (Check) สำหรับ L3, L4, L5
            if level >= 3 and final_pdca_breakdown.get('C', 0) < 2:
                
                is_c_evidence_found = any(
                    chunk.get('pdca_tag') == 'Check' or 
                    any(k in chunk.get('text', '') for k in C_KEYWORDS)
                    for chunk in top_evidences
                )
                
                is_p_d_ok = final_pdca_breakdown.get('P', 0) >= 1 and final_pdca_breakdown.get('D', 0) >= 1
                
                if is_c_evidence_found and is_p_d_ok:
                    final_pdca_breakdown['C'] = 2.0
                    self.logger.warning(f"  > L{level} C Score OVERRIDE: Forced to 2.0 due to evidence/keywords 'Check' found.")
                elif level == 3 and is_c_evidence_found:
                    final_pdca_breakdown['C'] = 2.0
                    self.logger.warning(f"  > L3 C Score OVERRIDE: Forced to 2.0 (L3 Focus) due to evidence/keywords 'Check' found.")


            # 2. การตรวจสอบและบังคับคะแนน A (Act) สำหรับ L4, L5
            if level >= 4 and final_pdca_breakdown.get('A', 0) < 2:
                
                is_a_evidence_found = any(
                    chunk.get('pdca_tag') == 'Act' or 
                    any(k in chunk.get('text', '') for k in A_KEYWORDS)
                    for chunk in top_evidences
                )
                
                if is_a_evidence_found and final_pdca_breakdown.get('C', 0) == 2.0: 
                    final_pdca_breakdown['A'] = 2.0
                    self.logger.warning(f"  > L{level} A Score OVERRIDE: Forced to 2.0 due to evidence/keywords 'Act' found and C is 2.0.")

            # 3. Final Correction & Calculation after Override UP
            
            if level == 3:
                if final_pdca_breakdown.get('A', 0) > 0:
                    self.logger.warning(f"  > L3 PDCA Correction: A_Act_Score must be 0. Correcting.")
                    final_pdca_breakdown['A'] = 0.0

            final_score = sum(final_pdca_breakdown.values())
            
            required_score_for_level = get_correct_pdca_required_score(level)
            is_passed = final_score >= required_score_for_level


        # -------------------------------------------------------------
        
        status = "PASS" if is_passed else "FAIL"
        # 📌 Final Evidence Strength ต้องใช้ค่าที่ถูก Cap แล้ว (max_evi_str_for_prompt)
        evidence_strength = max_evi_str_for_prompt if is_passed else 0.0
        
        # 📌 Final AI Confidence ต้องใช้ค่าที่ถูกปรับจาก Patch 1 หรือคำนวณจาก Strength ใหม่
        if is_passed:
            # ใช้ Logic เดิมสำหรับ Confidence ถ้า PASS และไม่ Hard Fail
            # 🔴 FIX: แก้ไข Syntax Error ตรงนี้
            ai_confidence = "HIGH" if evidence_strength >= 8 else "MEDIUM" if evidence_strength >= 5.5 else "LOW"
        elif is_hard_fail_pdca:
            ai_confidence = "LOW"
            
        icon = "🟢" if is_passed else "🔴"

        self.logger.info(
            f"  > Assessment {sub_id} L{level} completed → {icon} {status} "
            f"(Score: {final_score:.1f} | Evi Str: {evidence_strength:.1f} | Conf: {ai_confidence})"
        )

        # ✅ แก้ไข: เปลี่ยน INFO เป็น DEBUG
        self.logger.debug(
            f"  > Context Built L{level}: Direct chunks={len(top_evidences)}, "
            f"Aux={'มี' if aux_summary != 'ไม่มีหลักฐานรอง' else 'ไม่มี'}, "
            f"Baseline={'มี' if 'ไม่มีหลักฐานจาก Level ก่อนหน้า' not in baseline_summary else 'ไม่มี'}"
        )

        # === แก้ pdca_tag ใน temp_map_for_level ให้ตรงกับที่ re-tag แล้ว ===
        for chunk in top_evidences:
            if "text" in chunk and chunk["text"].strip():
                # 📌 ต้องแน่ใจว่า classify_by_keyword ถูก Import หรือเป็น method ของ class
                new_tag = classify_by_keyword(chunk["text"], sub_id=sub_id, contextual_rules_map=self.contextual_rules_map)
                if new_tag != 'Other':
                    chunk["pdca_tag"] = new_tag
        # ==================================================================================

        return {
            "sub_criteria_id": sub_id,
            "level": level,
            "statement_id": statement_id,
            "statement_text": statement_text,
            "is_passed": is_passed,
            "score": round(final_score, 1),
            "pdca_breakdown": final_pdca_breakdown,
            "reason": llm_result.get('reason', "No reason provided"),
            "extraction_c": llm_result.get('extraction_c', "-"),
            "extraction_a": llm_result.get('extraction_a', "-"),
            "evidence_strength": evidence_strength,
            "ai_confidence": ai_confidence,
            "max_relevant_score": highest_rerank_score,
            "temp_map_for_level": top_evidences,
            "duration": time.time() - start_time,
            "retrieval_duration": retrieval_duration,
            "llm_duration": llm_duration,
        }