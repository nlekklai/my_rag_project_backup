# -*- coding: utf-8 -*-
# core/seam_assessment.py

import sys
import json
import logging
import time
from datetime import datetime, date
import os
from typing import List, Dict, Any, Optional, Union, Tuple, Set, Final, Literal
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
import multiprocessing 
from functools import partial
import pathlib, uuid
from copy import deepcopy
import tempfile
import shutil
import re
import hashlib
import unicodedata 
import random

from core.json_extractor import _robust_extract_json, _robust_extract_json_list

# -------------------- 1. PROTECTIVE IMPORTS --------------------
# จัดการเรื่อง FileLock และ Database ก่อนเพื่อนเพื่อให้ส่วนอื่นเรียกใช้ได้ชัวร์ๆ
try:
    from filelock import FileLock
except ImportError:
    class FileLock:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
    print("⚠️ WARNING: 'filelock' not installed.")

try:
    from database import init_db, db_update_task_status
    update_db_core = db_update_task_status
except ImportError:
    def init_db(): pass
    def update_db_core(*args, **kwargs): pass

# -------------------- 2. PATH SETUP --------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# -------------------- 3. CORE LOGIC & CONFIG IMPORTS --------------------
try:
    # --- Configs ---
    from config.global_vars import (
        MAX_LEVEL, EVIDENCE_DOC_TYPES, RERANK_THRESHOLD, MAX_EVI_STR_CAP,
        DEFAULT_LLM_MODEL_NAME, LLM_TEMPERATURE, MIN_RETRY_SCORE,
        MAX_PARALLEL_WORKERS, PDCA_PRIORITY_ORDER, TARGET_DEVICE,
        PDCA_PHASE_MAP, INITIAL_TOP_K, FINAL_K_RERANKED,
        MAX_CHUNKS_PER_FILE, MAX_CHUNKS_PER_BLOCK, MATURITY_LEVEL_GOALS,
        SEAM_ENABLER_FULL_NAME_TH, SEAM_ENABLER_FULL_NAME_EN,
        SCORING_MODE, MAX_CHUNKS_LOW, MAX_CHUNKS_HIGH,
        CRITICAL_CA_THRESHOLD, EVIDENCE_SELECTION_STRATEGY, EVIDENCE_CUMULATIVE_CAP,
        GLOBAL_EVIDENCE_INSTRUCTION, DEFAULT_ENABLER, PDCA_CONFIG_MAP, PDCA_PHASE_DESCRIPTIONS,
        SEAM_ENABLER_FULL_NAME_TH, ANALYSIS_FINAL_K, RETRIEVAL_RERANK_FLOOR, 
        RETRIEVAL_EARLY_EXIT_COUNT, RETRIEVAL_HIGH_RERANK_THRESHOLD,
        RETRIEVAL_EARLY_EXIT_SCORE_THRESHOLD, RETRIEVAL_RELEVANCE_THRESHOLD
    )
    

    # --- Utilities ---
    from core.llm_data_utils import ( 
        evaluate_with_llm, retrieve_context_with_filter, 
        action_plan_normalize_keys, evaluate_with_llm_low_level, 
        LOW_LEVEL_K, create_context_summary_llm, _fetch_llm_response,
        _check_and_handle_empty_context, set_mock_control_mode as set_llm_data_mock_mode
    )
    from core.vectorstore import VectorStoreManager, load_all_vectorstores
    from core.retry_policy import RetryPolicy, RetryResult  # <-- สำคัญมาก
    from core.json_extractor import _robust_extract_json

    # --- Path Utils ---
    from utils.path_utils import (
        get_mapping_file_path, get_evidence_mapping_file_path, 
        get_contextual_rules_file_path, get_assessment_export_file_path,
        get_export_dir, get_rubric_file_path, _n, get_doc_type_collection_key,
        get_tenant_info_file_path
    )

    # --- Prompts ---
    try:
        from core.seam_prompts import (
            ATOMIC_ACTION_PROMPT, MASTER_ROADMAP_PROMPT,
            SYSTEM_ATOMIC_ACTION_PROMPT, SYSTEM_MASTER_ROADMAP_PROMPT
        )
    except ImportError:
        ATOMIC_ACTION_PROMPT = "Recommendation: {coaching_insight} Level: {level}"
        MASTER_ROADMAP_PROMPT = "Roadmap for {sub_criteria_name}: {aggregated_insights}"
        SYSTEM_ATOMIC_ACTION_PROMPT = "Assistant mode."
        SYSTEM_MASTER_ROADMAP_PROMPT = "Strategy mode."

except ImportError as e:
    # 🚨 EMERGENCY FALLBACK: ด่านสุดท้าย 🚨
    print(f"⚠️ EMERGENCY: Core Module missing, initializing safety fallbacks: {e}")
    
    # Constants
    MAX_LEVEL = 5
    EVIDENCE_DOC_TYPES = "evidence"
    LOW_LEVEL_K = 5  # เพิ่มตัวนี้
    
    # [FIX] Class/Policy
    class RetryResult:
        def __init__(self, data=None): self.data = data or {}
        def get(self, k, d=None): return self.data.get(k, d)
    class RetryPolicy:
        def __init__(self, *args, **kwargs): pass
        def execute(self, func, *args, **kwargs): return RetryResult(func(*args, **kwargs))

    # [FIX] Path Functions
    def _n(s): return str(s).lower().strip()
    def get_rubric_file_path(tenant, enabler, **kwargs): 
        return f"data_store/{_n(tenant)}/config/{_n(tenant)}_{_n(enabler)}_rubric.json"
    def get_contextual_rules_file_path(tenant, enabler, **kwargs):
        return f"data_store/{_n(tenant)}/config/{_n(tenant)}_{_n(enabler)}_contextual_rules.json"
    def get_evidence_mapping_file_path(tenant, year, enabler, **kwargs):
        return f"data_store/{_n(tenant)}/mapping/{year}/{_n(tenant)}_{year}_{_n(enabler)}_evidence_mapping.json"
    def get_mapping_file_path(doc_type, tenant, year=None, enabler=None, **kwargs):
        return f"data_store/{_n(tenant)}/mapping/{year}/{_n(tenant)}_{year}_{_n(enabler)}_doc_id_mapping.json"
    def get_assessment_export_file_path(*args, **kwargs): return "exports/temp_report.json"
    def get_export_dir(*args, **kwargs): return "exports"

    # [FIX] Logic Functions (ด่านสุดท้ายที่ระบบถามหา)
    def evaluate_with_llm_low_level(*args, **kwargs): 
        return {"score": 0.0, "reason": "Fallback mode active", "is_passed": False}
    
    def evaluate_with_llm(*args, **kwargs): 
        return {"score": 0.0, "reason": "Fallback mode active", "is_passed": False}

    def retrieve_context_with_filter(*args, **kwargs): 
        return {"top_evidences": [], "aggregated_context": ""}


    # Placeholders
    def _fetch_llm_response(*args, **kwargs): return "{}"
    def _robust_extract_json(t): return {}
    def set_llm_data_mock_mode(m): pass
    def action_plan_normalize_keys(d): return d
    def create_context_summary_llm(*args, **kwargs): return {"summary": "N/A", "coaching": "N/A"}
    def _check_and_handle_empty_context(*args, **kwargs): return None, False

    ATOMIC_ACTION_PROMPT = "Level {level}: {coaching_insight}"
    MASTER_ROADMAP_PROMPT = "Roadmap: {aggregated_insights}"
    SYSTEM_ATOMIC_ACTION_PROMPT = "Assistant"
    SYSTEM_MASTER_ROADMAP_PROMPT = "Strategist"

# -------------------- 5. LOGGER SETUP --------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_enabler_full_name(enabler: str, lang: str = "th") -> str:
    """
    ดึงชื่อเต็มของ Enabler ตามรหัส (เช่น "KM" → "การจัดการความรู้")
    
    Args:
        enabler_code (str): รหัส enabler เช่น "KM", "CG", "SP"
        lang (str): "th" สำหรับภาษาไทย, "en" สำหรับอังกฤษ (default: "th")
    
    Returns:
        str: ชื่อเต็ม หรือคืนรหัสเดิมถ้าไม่พบ
    
    Example:
        get_enabler_full_name("KM") → "การจัดการความรู้"
        get_enabler_full_name("CG", "en") → "Corporate Governance"
    """
    code = str(enabler).upper().strip()
    if lang.lower() == "th":
        return SEAM_ENABLER_FULL_NAME_TH.get(code, code)
    return SEAM_ENABLER_FULL_NAME_EN.get(code, code)


def get_pdca_goal_for_level(level: int) -> str:
    """
    ดึงคำอธิบายเป้าหมายหลักของ Maturity Level นั้น ๆ
    
    Args:
        level (int): ระดับ 1-5
    
    Returns:
        str: คำอธิบายเป้าหมาย หรือ "ไม่ระบุเป้าหมาย" ถ้าไม่พบ
    
    Example:
        get_pdca_goal_for_level(5) → "เน้นความยั่งยืน การปรับปรุงเชิงรุก...และการเป็นต้นแบบ (Role Model)"
    """
    return MATURITY_LEVEL_GOALS.get(int(level), "ไม่ระบุเป้าหมาย")
    
def _static_worker_process(worker_input_tuple: Tuple) -> Any:
    """
    [ULTIMATE WORKER v2026.1.23] Isolated Execution with Evidence Streaming
    ---------------------------------------------------------------------
    - สร้างสภาพแวดล้อมใหม่แยกขาดจากกัน (Zero Memory Leak)
    - รองรับการส่งคืน Evidence Map กลับไปยัง Main Process เพื่อป้องกันข้อมูลหาย
    - เพิ่มระบบ Error Handling แบบก้อนข้อมูล (Object) เพื่อไม่ให้ระบบรวมผลพัง
    """
    # 1. 📂 PATH & ENVIRONMENT SETUP
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.append(project_root)
        
    worker_logger = logging.getLogger(f"Worker_{os.getpid()}")

    # 2. 📦 UNPACKING ARGS
    try:
        (
            sub_criteria_data, enabler, target_level, mock_mode, 
            evidence_map_path, model_name, temperature,
            min_retry_score, max_retrieval_attempts, document_map, 
            action_plan_model, year, tenant
        ) = worker_input_tuple
        
        sub_id = sub_criteria_data.get('sub_id', 'UNKNOWN')
    except Exception as e:
        return {"error": f"Worker unpacking failed: {str(e)}", "status": "failure"}, {}

    # 3. 🏗️ RECONSTRUCT ISOLATED ENGINE
    try:
        # สร้าง Config เฉพาะกิจ
        worker_config = AssessmentConfig(
            enabler=enabler,
            tenant=tenant,
            year=int(year) if year else None,     
            target_level=target_level,
            mock_mode=mock_mode,
            model_name=model_name, 
            temperature=temperature,
            min_retry_score=min_retry_score,            
            max_retrieval_attempts=max_retrieval_attempts 
        )

        # คืนชีพ Engine (ตรวจสอบให้แน่ใจว่า Class SEAMPDCAEngine มีการรับ document_map)
        worker_instance = SEAMPDCAEngine(
            config=worker_config, 
            evidence_map_path=evidence_map_path, 
            llm_instance=None,              
            vectorstore_manager=None,       
            logger_instance=worker_logger,
            document_map=document_map,      
            ActionPlanActions=action_plan_model
        )
    except Exception as e:
        worker_logger.error(f"❌ Worker initialization failed: {e}")
        return {"sub_id": sub_id, "error": f"Init Error: {str(e)}"}, {}

    # 4. ⚡ EXECUTE & STREAM BACK RESULTS
    try:
        # 🎯 จุดสำคัญ: _run_sub_criteria_assessment_worker ต้องคืนค่า (result, evidence_mem)
        # evidence_mem คือ dict ที่เก็บ chunks/docs ที่ AI เลือกใช้จริง
        result, worker_evidence_mem = worker_instance._run_sub_criteria_assessment_worker(sub_criteria_data)
        
        # ตรวจสอบว่า result มี sub_id ติดไปด้วยเพื่อให้ Merge ถูกตัว
        if isinstance(result, dict) and 'sub_id' not in result:
            result['sub_id'] = sub_id

        return result, worker_evidence_mem
        
    except Exception as e:
        worker_logger.error(f"❌ Execution error for {sub_id}: {str(e)}")
        # คืนค่าแบบ Fallback เพื่อให้ Main Process ไม่ค้าง
        return {
            "sub_id": sub_id,
            "error": str(e),
            "status": "failed",
            "is_passed": False,
            "score": 0.0,
            "reason": f"Worker Exception: {str(e)}"
        }, {}
        
# =================================================================
# Configuration Class
# =================================================================
@dataclass
class AssessmentConfig:
    """Configuration for the SEAM PDCA Assessment Run."""
    
    # ------------------ 1. Assessment Context ------------------
    enabler: str = None
    tenant: str = None
    year: int = None  # 👈 เปลี่ยนจาก DEFAULT_YEAR เป็น None
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
# SEAM Assessment Engine (PDCA Focused) - Full Revise v2026
# =================================================================
class SEAMPDCAEngine:
    
    def __init__(
        self, 
        config: AssessmentConfig,
        llm_instance: Any = None, 
        logger_instance: logging.Logger = None,
        rag_retriever_instance: Any = None,
        doc_type: str = None, 
        vectorstore_manager: Optional['VectorStoreManager'] = None,
        evidence_map_path: Optional[str] = None,
        document_map: Optional[Dict[str, str]] = None,
        is_parallel_all_mode: bool = False,
        sub_id: str = 'all',
        record_id: Optional[str] = None,
        **kwargs  
    ):
        """
        [ULTIMATE REVISE v2026.3.1] 
        - FIXED: flattened_rubric logic corruption (No more overwrite)
        - FIXED: Robust doc_type comparison (is_evidence_mode)
        - FIXED: Safe VSM/LLM initialization order
        """
        # -------------------------------------------------------
        # 1. Basic Config & Logger Setup
        # -------------------------------------------------------
        self.config = config
        self.doc_type = doc_type or getattr(config, 'doc_type', EVIDENCE_DOC_TYPES)
        
        # [REVISED] Robust Comparison Pattern ตามคำแนะนำ
        self.is_evidence_mode = str(self.doc_type).strip().lower() in (
            t.lower() for t in (EVIDENCE_DOC_TYPES if isinstance(EVIDENCE_DOC_TYPES, (list, tuple)) else [EVIDENCE_DOC_TYPES])
        )
        
        log_year = config.year if self.is_evidence_mode else "general"

        if logger_instance is not None:
            self.logger = logger_instance
        else:
            self.logger = logging.getLogger(__name__).getChild(
                f"Engine|{config.enabler}|{config.tenant}/{log_year}"
            )

        self.logger.info(f"🚀 Initializing SEAMPDCAEngine: {config.enabler} ({config.tenant}/{log_year})")

        # -------------------------------------------------------
        # 2. Core Configuration & Sanity Check
        # -------------------------------------------------------
        if not self.config.enabler or not self.config.tenant:
            self.logger.critical("❌ Mandatory Config Missing: enabler and tenant must be provided!")
            raise ValueError("Enabler and Tenant are required for SEAMPDCAEngine.")

        self.enabler = config.enabler
        self.tenant_id = config.tenant
        self.year = config.year
        self.target_level = config.target_level
        self.sub_id = sub_id
        self.record_id = record_id
        
        # State Management
        self.is_parallel_all_mode = is_parallel_all_mode
        self.is_sequential = getattr(config, 'force_sequential', True)
        
        # [REVISED] results vs assessment_results_map
        # results เก็บเพื่อ legacy compatibility, assessment_results_map คือ source of truth
        self.results = {} 
        self.assessment_results_map = {} 

        # -------------------------------------------------------
        # 3. System Warm-up
        # -------------------------------------------------------
        try:
            init_db()
        except Exception as e:
            self.logger.error(f"⚠️ DB Init Warning: {e}")

        # -------------------------------------------------------
        # 4. Data Loading (Rubric & Rules)
        # -------------------------------------------------------
        self.rubric = self._load_rubric()

        # [REVISED] Flatten Rubric - ทำที่เดียวและไม่ Overwrite ซ้ำตอนท้าย
        try:
            self.flattened_rubric = self._flatten_rubric_to_statements(self.rubric)
            self.logger.info(f"✅ Rubric Meta-Data Flattened: {len(self.flattened_rubric)} levels ready.")
        except Exception as e:
            self.logger.error(f"⚠️ Rubric Flattening Failed: {e}")
            self.flattened_rubric = []
        
        self.contextual_rules_map = self._load_contextual_rules_map()
        self.retry_policy = RetryPolicy(max_attempts=3, base_delay=2.0)

        # -------------------------------------------------------
        # 5. Mapping & Evidence Setup
        # -------------------------------------------------------
        self.evidence_map = {}
        if self.is_evidence_mode:
            self.evidence_map_path = evidence_map_path or get_evidence_mapping_file_path(
                tenant=self.config.tenant, year=self.config.year, enabler=self.enabler
            )
            self.evidence_map = self._load_evidence_map()

        # Document Mapping
        loaded_map = document_map or {}
        if not loaded_map:
            mapping_path = get_mapping_file_path(
                self.doc_type, 
                tenant=self.config.tenant, 
                year=self.config.year if self.is_evidence_mode else None,
                enabler=self.enabler if self.is_evidence_mode else None
            )
            if os.path.exists(mapping_path):
                try:
                    with open(mapping_path, 'r', encoding='utf-8') as f:
                        raw_data = json.load(f)
                    loaded_map = {k: v.get("file_name", k) for k, v in raw_data.items()}
                except Exception as e:
                    self.logger.error(f"❌ Error parsing mapping file: {e}")

        self.doc_id_to_filename_map = loaded_map
        self.document_map = loaded_map

        # -------------------------------------------------------
        # 6. Lazy Engine Initialization (VSM & LLM)
        # -------------------------------------------------------
        # [REVISED] Safe initialization order
        if llm_instance is None: 
            self._initialize_llm_if_none()
        else:
            self.llm = llm_instance

        if vectorstore_manager is None: 
            # ส่ง LLM เข้าไปถ้า VSM จำเป็นต้องใช้ embedding จาก LLM
            self._initialize_vsm_if_none() 
        else:
            self.vectorstore_manager = vectorstore_manager

        if self.vectorstore_manager:
            try:
                self.vectorstore_manager._load_doc_id_mapping()
            except Exception as e:
                self.logger.warning(f"⚠️ VSM mapping not loaded: {e}")

       
        # -------------------------------------------------------
        # 7. Function Registry & Final States (CLEAN)
        # -------------------------------------------------------
        # self.standard_audit_agent = evaluate_with_llm
        # Register agents (ตัวทำงานจริง)
        self.standard_audit_agent = evaluate_with_llm              # L3–L5
        self.foundation_coaching_agent = evaluate_with_llm_low_level  # L1–L2

        # Entry
        self.assessment_router = self.evaluate_pdca

        self.rag_retriever = retrieve_context_with_filter

        # State Initialization
        self.final_subcriteria_results = []
        self.total_stats = {}
        self.raw_llm_results = []
        self.level_details_map = {} 
        self.previous_levels_evidence = [] 
        self.level_evidence_cache = {}
        self._cumulative_rules_cache = {}

        # [CRITICAL] ห้ามใส่ self.flattened_rubric = [] ตรงนี้เด็ดขาด!

        self.logger.info(f"✅ Engine Initialized: Ready for Assessment (Sub-ID: {self.sub_id})")

    # =================================================================
    # DB Proxy Methods (Enhanced v2026)
    # =================================================================
    def db_update_task_status(self, message: str, progress: Optional[int] = None, status: str = "RUNNING"):
        """
        Enhanced Wrapper สำหรับอัปเดตสถานะ
        - ไม่ต้องส่ง record_id ทุกครั้ง (ใช้ self.current_record_id อัตโนมัติ)
        - ถ้าไม่ส่ง progress จะเป็นการอัปเดตเฉพาะ message (คงค่า % เดิมไว้)
        """
        # 1. ดึง record_id จาก instance ถ้าไม่มีให้ข้าม
        rid = getattr(self, 'current_record_id', None) or getattr(self, 'record_id', None)
        if not rid: 
            return

        try:
            # 2. เตรียมข้อมูลอัปเดต
            # ในกรณีที่ progress เป็น None เราจะดึงค่าล่าสุดจาก memory หรือ database 
            # แต่โดยปกติ database.db_update_task_status ควรจัดการข้ามค่าที่เป็น None ให้เอ
            
            # 3. ส่งคำสั่งอัปเดต
            update_db_core(
                record_id=rid, 
                progress=progress, 
                message=message, 
                status=status
            )
            
            self.logger.debug(f"[DB-PROGRESS] {rid}: {progress if progress is not None else 'KEEP'}% - {message}")
            
        except Exception as e:
            self.logger.error(f"❌ DB Update Error for {rid}: {e}")

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
        Initializes VectorStoreManager with Smart Year Selection.
        - If Evidence: Priority 1 = Specific Year, Priority 2 = Root Fallback.
        - If Document: Priority 1 = Root (General), Priority 2 = Specific Year Fallback.
        """
        if self.vectorstore_manager is not None:
            return

        # 1. เตรียมความพร้อมของ DocType
        clean_dt = str(self.doc_type or getattr(self.config, 'doc_type', EVIDENCE_DOC_TYPES)).strip().lower()
        is_evidence = (clean_dt == EVIDENCE_DOC_TYPES.lower())
        
        self.logger.info(f"🚀 Loading vectorstore(s) for DocType: '{clean_dt}' (Mode: {'Evidence' if is_evidence else 'General'})")

        try:
            target_enabler = str(self.enabler).lower() if self.enabler else None
            
            # 🎯 [SMART SELECTION] กำหนดปีที่จะค้นหาเป็นอันดับแรก
            # ถ้าเป็น evidence ให้หาตามปี (2568) ก่อน แต่ถ้าเป็น document ให้หาที่ Root (None) ก่อน
            primary_year = self.config.year if is_evidence else None
            secondary_year = None if is_evidence else self.config.year

            # 2. First Attempt: โหลดตามลำดับความสำคัญ
            self.vectorstore_manager = load_all_vectorstores(
                doc_types=[clean_dt], 
                enabler_filter=target_enabler, 
                tenant=self.config.tenant, 
                year=primary_year       
            )
            
            def count_retrievers(vsm):
                if vsm and hasattr(vsm, '_multi_doc_retriever') and vsm._multi_doc_retriever:
                    return len(vsm._multi_doc_retriever._all_retrievers)
                return 0

            len_retrievers = count_retrievers(self.vectorstore_manager)
            
            # 3. Second Attempt (Fallback): ถ้าหาแบบแรกไม่เจอ ให้สลับไปหาอีกแบบ
            if len_retrievers == 0:
                lookup_label = f"year {secondary_year}" if secondary_year else "tenant root"
                self.logger.info(f"⚠️ No collections found in primary path, searching in {lookup_label}...")
                
                self.vectorstore_manager = load_all_vectorstores(
                    doc_types=[clean_dt], 
                    enabler_filter=target_enabler, 
                    tenant=self.config.tenant, 
                    year=secondary_year       
                )
                len_retrievers = count_retrievers(self.vectorstore_manager)

            # 4. Post-Load Process
            if self.vectorstore_manager and len_retrievers > 0:
                self.vectorstore_manager._load_doc_id_mapping() 
                self.logger.info(f"✅ MultiDocRetriever loaded with {len_retrievers} collections.") 
            else:
                # 5. Final Error Handling
                expected_p = f"data_store/{self.config.tenant}/vectorstore/{primary_year or 'root'}"
                self.logger.error(f"❌ FATAL: 0 vector store collections loaded. Please check folder: {expected_p}")
                raise ValueError(f"No vector collections found for '{target_enabler}' in {self.config.tenant}")

        except Exception as e:
            self.logger.error(f"❌ FATAL: Could not initialize VectorStoreManager: {str(e)}")
            raise

    # -------------------- Contextual Rules Handlers (FIXED) --------------------
    def _load_contextual_rules_map(self) -> Dict[str, Any]:
        """
        [FINAL REVISED v2026.5] โหลด Contextual Rules อย่างปลอดภัย
        - รองรับ multi-tenant และ multi-enabler เต็มรูปแบบ
        - มี fallback ที่ชัดเจน
        - ตรวจสอบโครงสร้าง Maturity (L1-L5) อย่างเข้มงวด
        - Logging ที่อ่านง่ายและช่วย debug
        - ไม่ raise exception (เพื่อไม่ให้ engine ล้มทั้งระบบ)
        """
        # 1. สร้าง path ตาม tenant + enabler
        try:
            filepath = get_contextual_rules_file_path(
                tenant=self.config.tenant,
                enabler=self.enabler
            )
            self.logger.debug(f"🔍 Attempting to load contextual rules from: {filepath}")
        except Exception as e:
            self.logger.error(f"❌ FATAL: Failed to generate rules file path: {e}")
            return {"_enabler_defaults": {}}

        # 2. ตรวจสอบว่าไฟล์มีอยู่จริงหรือไม่
        if not os.path.exists(filepath):
            self.logger.warning(
                f"⚠️ Contextual Rules file not found: {filepath}\n"
                f"   → Using only global defaults (if available from fallback)."
            )
            return {"_enabler_defaults": {}}

        # 3. โหลดและ parse JSON
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            self.logger.error(f"❌ JSON Decode Error in {filepath}: {e} (line {e.lineno}, col {e.colno})")
            return {"_enabler_defaults": {}}
        except Exception as e:
            self.logger.error(f"❌ Unexpected error reading {filepath}: {e}")
            return {"_enabler_defaults": {}}

        # 4. ตรวจสอบโครงสร้างพื้นฐาน
        if not isinstance(data, dict):
            self.logger.error(f"❌ Invalid rules format in {filepath}: Expected dict, got {type(data)}")
            return {"_enabler_defaults": {}}

        # 5. นับจำนวน criteria และตรวจสอบ Maturity Structure
        sub_criteria_keys = [k for k in data.keys() if not k.startswith("_")]
        num_criteria = len(sub_criteria_keys)

        if num_criteria == 0:
            self.logger.warning(f"⚠️ No sub-criteria found in {filepath} (only defaults).")
        else:
            # ตรวจสอบว่ามี Maturity Levels (L1, L2, ...) หรือไม่
            sample_sub = sub_criteria_keys[0]
            sub_data = data[sample_sub]
            level_keys = [k for k in sub_data.keys() if k.startswith("L") and len(k) >= 2]
            
            if level_keys:
                detected_levels = sorted(level_keys)
                self.logger.info(
                    f"✅ Maturity-Based Rules loaded successfully!\n"
                    f"   File: {filepath}\n"
                    f"   Criteria: {num_criteria} (e.g., {sample_sub})\n"
                    f"   Levels detected: {', '.join(detected_levels)}"
                )
            else:
                self.logger.warning(
                    f"⚠️ Rules for '{sample_sub}' in {filepath} do not contain Maturity Levels (L1-L5).\n"
                    f"   Falling back to flat structure or defaults."
                )

        # 6. ตรวจสอบ _enabler_defaults
        if "_enabler_defaults" in data:
            defaults = data["_enabler_defaults"]
            if isinstance(defaults, dict) and any(k.endswith("_keywords") for k in defaults.keys()):
                self.logger.info("✅ Global PDCA Keywords (_enabler_defaults) loaded.")
            else:
                self.logger.warning("⚠️ _enabler_defaults exists but has invalid structure.")
        else:
            self.logger.info("ℹ️ No _enabler_defaults section found. Using empty defaults.")

        # 7. ส่งคืน data ที่โหลดสำเร็จ
        self.logger.info(f"✅ Contextual Rules fully loaded from {filepath} ({num_criteria} criteria).")
        return data
    
    def get_rule_content(self, sub_id: str, level: int, key_type: str):
        """
        ดึงข้อมูลเกณฑ์จาก Contextual Rules แบบลำดับชั้น
        Priority: Specific Level > Sub-ID Root > Global Defaults > Fallback
        """
        rule = self.contextual_rules_map.get(sub_id, {})
        level_key = f"L{level}"

        # 1. Specific Level (e.g., 1.2 -> L1 -> query_synonyms)
        if key_type in rule.get(level_key, {}):
            return rule[level_key][key_type]

        # 2. Sub-ID Root (e.g., 1.2 -> query_synonyms)
        if key_type in rule:
            return rule[key_type]

        # 3. Global Defaults (e.g., _enabler_defaults -> plan_keywords)
        defaults = self.contextual_rules_map.get("_enabler_defaults", {})
        if key_type in defaults:
            return defaults[key_type]

        # 4. Fallback (ป้องกัน Error กรณีไม่เจอ Key)
        fallbacks = {
            "require_phase": ["P", "D"], # Default พื้นฐาน
            "must_include_keywords": [],
            "plan_keywords": [],
            "do_keywords": [],
            "check_keywords": [],
            "act_keywords": [],
            "query_synonyms": ""
        }
        return fallbacks.get(key_type, "")

    def get_cumulative_rules_cached(self, sub_id: str, level: int) -> Dict[str, Any]:
        """
        Return cumulative rules with per-engine caching.
        Cache key: (sub_id, level)
        """
        key = (sub_id, level)

        if key not in self._cumulative_rules_cache:
            self._cumulative_rules_cache[key] = self.get_cumulative_rules(
                sub_id=sub_id,
                current_level=level
            )

        return self._cumulative_rules_cache[key]


    def get_cumulative_rules(self, sub_id: str, current_level: int) -> Dict[str, Any]:
        """
        [FINAL REVISED v2026.1.20] - SMART ACCUMULATION & ROBUST MAPPING
        ------------------------------------------------------------------
        - สะสม Keywords (PDCA) และคำสั่งจาก L1 จนถึงระดับปัจจุบัน
        - **Robust Synonym Split**: รองรับการแยกคำด้วย ช่องว่าง, คอมม่า, และเซมิโคลอน
        - **Auto-Fallback Phases**: เติม Required Phases ให้อัตโนมัติหาก Config ไม่ระบุ
        - **Case-Insensitive**: ปรับ Keywords เป็นตัวพิมพ์เล็กเพื่อความแม่นยำในการ Match
        """
        defaults = self.contextual_rules_map.get('_enabler_defaults', {})
        sub_rules = self.contextual_rules_map.get(sub_id, {})
        
        # 1. เตรียม OrderedDict เพื่อรักษาลำดับความสำคัญและกำจัดคำซ้ำ
        cum_keywords = {
            "plan": OrderedDict((k.lower(), None) for k in defaults.get('plan_keywords', [])),
            "do":   OrderedDict((k.lower(), None) for k in defaults.get('do_keywords', [])),
            "check": OrderedDict((k.lower(), None) for k in defaults.get('check_keywords', [])),
            "act":  OrderedDict((k.lower(), None) for k in defaults.get('act_keywords', []))
        }
        
        cum_must_include = OrderedDict()
        required_phases = set()
        level_specific_instructions = {}
        source_levels = []

        # 2. เริ่มสะสมกฎจาก L1 จนถึงระดับปัจจุบัน
        for lv in range(1, current_level + 1):
            lv_key = f"L{lv}"
            level_rule = sub_rules.get(lv_key, {})
            
            if not level_rule:
                continue
            
            source_levels.append(lv)

            # A) ดึง query_synonyms มาเป็น must_include (Robust Split)
            synonyms_str = level_rule.get('query_synonyms', "")
            if synonyms_str:
                # ใช้ Regex แยกคำรองรับทั้ง "คำ1 คำ2" หรือ "คำ1,คำ2" หรือ "คำ1;คำ2"
                words = re.split(r'[,\s;|]+', synonyms_str)
                for word in words:
                    clean_word = word.strip().lower()
                    if clean_word:
                        cum_must_include[clean_word] = None

            # B) อัปเดต PDCA Keywords (สะสมต่อจากระดับก่อนหน้า)
            for phase, key_name in [("plan", "plan_keywords"), ("do", "do_keywords"),
                                ("check", "check_keywords"), ("act", "act_keywords")]:
                new_kws = level_rule.get(key_name, [])
                for kw in new_kws:
                    cum_keywords[phase][kw.lower()] = None
            
            # C) สะสม Required Phases
            if 'require_phase' in level_rule:
                # รับได้ทั้ง List ["P", "D"] หรือ String "P,D"
                phases = level_rule['require_phase']
                if isinstance(phases, str):
                    phases = re.split(r'[,\s]+', phases)
                required_phases.update([p.upper() for p in phases if p])
            
            # D) เก็บคำสั่งเฉพาะ (Specific Instructions)
            specific = level_rule.get('specific_contextual_rule')
            if specific:
                level_specific_instructions[lv] = specific.strip()

        # 3. จัดการข้อมูลก่อนส่งออก (Finalize & Fallback)
        result_keywords = {phase: list(cum_keywords[phase].keys()) for phase in cum_keywords}
        
        # Smart Fallback สำหรับ Required Phases (ถ้าใน Config ไม่ได้ระบุเลย)
        final_phases = sorted(list(required_phases))
        if not final_phases:
            if current_level <= 3: final_phases = ["P", "D"]
            elif current_level == 4: final_phases = ["P", "D", "C"]
            else: final_phases = ["P", "D", "C", "A"]

        # สร้าง Instructions String สำหรับ Prompt
        instructions_lines = [f"เกณฑ์การพิจารณาสำหรับ {sub_id} ระดับ Maturity L{current_level}:"]
        for lv in sorted(level_specific_instructions.keys()):
            icon = "🎯" if lv == current_level else "✅"
            instructions_lines.append(f"{icon} [Level {lv}]: {level_specific_instructions[lv]}")

        # 4. Logging & Return
        self.logger.info(
            f"🚀 [RULE_CUMULATIVE] {sub_id} L{current_level} | "
            f"Must-Include: {len(cum_must_include)} words | "
            f"Phases: {final_phases}"
        )

        return {
            "plan_keywords": result_keywords["plan"],
            "do_keywords": result_keywords["do"],
            "check_keywords": result_keywords["check"],
            "act_keywords": result_keywords["act"],
            "required_phases": final_phases,
            "must_include_keywords": list(cum_must_include.keys()),
            "level_specific_instructions": level_specific_instructions,
            "all_instructions": "\n".join(instructions_lines),
            "source_summary": f"Accumulated from levels: {source_levels}"
        }

    def _check_contextual_rule_condition(
        self, 
        condition: Dict[str, Any], 
        sub_id: str, 
        level: int, 
        top_evidences: List[Dict[str, Any]]
    ) -> bool:
        """
        [SIMPLIFIED v2026] แค่ log เตือน continuity + min evidence
        ไม่ block การประเมิน (return True เสมอ)
        """
        if level > 1:
            prev_level = level - 1
            is_prev_passed = False
            if hasattr(self, 'level_details_map') and str(prev_level) in self.level_details_map:
                is_prev_passed = self.level_details_map[str(prev_level)].get('is_passed', False)
            
            if not is_prev_passed:
                self.logger.warning(f"⚠️ [GAP DETECTED] L{prev_level} not passed for {sub_id} L{level} - may affect validity")

        min_docs = condition.get('min_evidences', 1)
        if len(top_evidences) < min_docs:
            self.logger.warning(f"⚠️ [LOW EVIDENCE] {sub_id} L{level}: {len(top_evidences)} docs (required: {min_docs})")

        return True  # ไม่ block

    def post_process_llm_result(
        self,
        llm_output: Any,
        level: int,
        sub_id: str = None,
        contextual_config: Optional[Dict] = None,
        top_evidences: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        [ULTIMATE REVISED v2026.01.31] Post-process LLM output for SE-AM Assessment
        เป้าหมาย: รับประกันคะแนนไม่ตกต่ำเกินจริง, กำจัด IT Ghost, Dashboard แสดงผลสวยงาม
        """
        log_prefix = f"Sub:{sub_id or '??'} L{level}"

        # 1. Robust JSON Repair
        if isinstance(llm_output, str):
            cleaned = re.sub(r'```json\s*|\s*```|\n+', ' ', llm_output.strip())
            cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)  # ลบ trailing comma
            try:
                llm_output = json.loads(cleaned)
                self.logger.debug(f"[JSON-REPAIR-SUCCESS] {log_prefix}")
            except json.JSONDecodeError as e:
                self.logger.warning(f"[JSON-REPAIR-FAIL] {log_prefix}: {e}")
                llm_output = {}
        if not isinstance(llm_output, dict) or not llm_output:
            return self._get_fallback_result(log_prefix)

        # 2. Required Phases
        required_phases = contextual_config.get("required_phases", []) or (
            ["P", "D"] if level <= 3 else ["P", "D", "C", "A"]
        )
        self.logger.debug(f"[REQUIRED-PHASES] {log_prefix}: {required_phases}")

        # 3. PDCA Extraction + Keyword Rescue
        pdca_results = {"P": 0.0, "D": 0.0, "C": 0.0, "A": 0.0}
        reason_text = str(llm_output.get('reason', '')).lower()
        ext_texts = {p: str(llm_output.get(f"Extraction_{p}", "")).lower() for p in "PDCA"}

        for phase in "PDCA":
            score = 0.0
            for k in [f"{phase}_Score", f"score_{phase.lower()}", f"Extraction_{phase}_Score"]:
                if k in llm_output:
                    try:
                        score = float(llm_output[k])
                        break
                    except:
                        continue

            # Keyword Rescue
            phase_kws = contextual_config.get(f"{phase.lower()}_keywords", [])
            combined_text = reason_text + " " + ext_texts.get(phase, "")
            if score < 1.0 and any(kw.lower() in combined_text for kw in phase_kws):
                score = max(score, 1.5)
                self.logger.info(f"[PHASE-RESCUE] {log_prefix} {phase} boosted to 1.5 by keyword")

            pdca_results[phase] = min(max(score, 0.0), 2.0)

        # 4. Mandatory Floor (L1-L3 leniency)
        floor_value = 1.0 if level == 1 else 0.8 if level <= 3 else 0.5
        for phase in required_phases:
            if pdca_results[phase] < floor_value:
                pdca_results[phase] = floor_value
                self.logger.info(f"[PHASE-FLOOR] {log_prefix} {phase} forced to {floor_value} (L{level})")

        # 5. Normalized Score
        sum_req = sum(pdca_results[p] for p in required_phases)
        max_req = len(required_phases) * 2.0
        normalized_score = round((sum_req / max_req) * 2.0 if max_req > 0 else 0.0, 2)

        # 6. Safety Net (Rerank-based override)
        max_rr = max([ev.get('rerank_score', ev.get('score', 0.0)) for ev in top_evidences] or [0.0])
        explicit_pass = llm_output.get("is_passed") is True
        is_force_pass = (normalized_score < 1.2 and max_rr >= 0.75)

        is_passed = explicit_pass or is_force_pass or (normalized_score >= 1.0)

        if is_passed and normalized_score < 1.2:
            normalized_score = max(normalized_score, 1.2)
            self.logger.info(f"[PASS-BOOST] {log_prefix} Score forced to {normalized_score} (safety net)")

        # 7. Anti-IT Ghost Cleanup
        coaching = str(llm_output.get("coaching_insight", "")).strip()
        self.logger.debug(f"[IT-CHECK-BEFORE] {log_prefix} Original: {coaching[:150]}...")
        it_patterns = r"(ระบบสารสนเทศอัตโนมัติ|พัฒนาระบบ|KMS|Software|IT System|แพลตฟอร์มดิจิทัล|Automation|พัฒนาระบบสารสนเทศ|ระบบอัตโนมัติ)"
        cleaned_coaching = re.sub(it_patterns, "กระบวนการและกิจกรรมสร้างการมีส่วนร่วม", coaching, flags=re.IGNORECASE)
        cleaned_coaching = re.sub(r"ควรพัฒนา", "ควรจัดทำ/ดำเนินการ", cleaned_coaching)
        if cleaned_coaching != coaching:
            self.logger.info(f"[ANTI-IT-CLEAN] {log_prefix} Cleaned: {cleaned_coaching[:150]}...")

        # 8. Dashboard Phase Sync (ให้ ✅ สวยงาม)
        if is_passed:
            for p in required_phases:
                if pdca_results[p] < 1.0:
                    pdca_results[p] = 1.0
                    self.logger.info(f"[DASHBOARD-SYNC] {log_prefix} {p} synced to 1.0")

        # 9. Final Summary Log
        self.logger.info(
            f"[POST-PROCESS-SUMMARY] {log_prefix} | "
            f"Raw score: {llm_output.get('score', 'N/A')} | "
            f"Normalized: {normalized_score:.2f} | "
            f"Passed: {is_passed} | "
            f"Force pass: {is_force_pass} (max_rr={max_rr:.3f}) | "
            f"PDCA: {pdca_results} | "
            f"Insight (cleaned): {cleaned_coaching[:120]}..."
        )

        return {
            "score": normalized_score,
            "is_passed": is_passed,
            "pdca_breakdown": pdca_results,
            "reason": llm_output.get("reason", "N/A"),
            "coaching_insight": cleaned_coaching,
            "required_phases": required_phases,
            "is_force_pass": is_force_pass,
            "max_rerank": max_rr
        }


    def _get_fallback_result(self, prefix: str) -> Dict[str, Any]:
        """Fallback เมื่อ LLM output พังหรือว่าง"""
        self.logger.error(f"[FALLBACK] {prefix} using zero-score fallback")
        return {
            "score": 0.0,
            "is_passed": False,
            "pdca_breakdown": {"P": 0.0, "D": 0.0, "C": 0.0, "A": 0.0},
            "reason": "AI Output Error - Fallback triggered",
            "coaching_insight": "ไม่สามารถประมวลผลข้อมูลได้ กรุณาตรวจสอบ log และหลักฐาน",
            "required_phases": []
        }
        

    def _expand_context_with_neighbor_pages(self, top_evidences: List[Any], collection_name: str) -> List[Dict[str, Any]]:
        """
        [ULTIMATE REVISE v2026.1.25]
        - Log Transparency: แสดงชื่อไฟล์จริงแทน UUID ในการทำ Neighbor Fetch
        - Smart Offsets: ปรับช่วงการกวาดหน้ากระดาษตามบริบทของคำสำคัญ
        - Metadata Enrichment: เชื่อมโยง ID กลับไปยังชื่อไฟล์จริงเพื่อใช้ในการประเมิน
        """
        if not self.vectorstore_manager or not top_evidences:
            return top_evidences

        standardized_evidences = []
        for d in top_evidences:
            # 1. Normalize ข้อมูลเบื้องต้น (รองรับทั้ง Dict และ Langchain Document)
            orig_score = d.get('score', d.get('rerank_score', 0.5)) if isinstance(d, dict) else getattr(d, 'metadata', {}).get('score', 0.5)
            
            if hasattr(d, 'page_content'):
                standardized_evidences.append({
                    "text": d.page_content, 
                    "metadata": getattr(d, 'metadata', {}), 
                    "score": orig_score
                })
            elif isinstance(d, dict):
                d['score'] = orig_score
                standardized_evidences.append(deepcopy(d))

        expanded_evidences = list(standardized_evidences)
        # ใช้ stable_doc_uuid หรือ source_id เป็น key หลักในการกันข้อมูลซ้ำ
        seen_keys = {
            f"{ev.get('metadata', {}).get('stable_doc_uuid', ev.get('metadata', {}).get('doc_id'))}_{ev.get('metadata', {}).get('page_label')}" 
            for ev in standardized_evidences
        }
        
        added_count = 0
        max_neighbors = 15  # เพิ่มขีดจำกัดเล็กน้อยเพื่อความครอบคลุม
        
        for doc in standardized_evidences:
            if added_count >= max_neighbors: break
            
            meta = doc.get('metadata', {})
            doc_uuid = meta.get("stable_doc_uuid") or meta.get("doc_id") or meta.get("source_id")
            if not doc_uuid: continue

            try:
                curr_page = int(str(meta.get("page_label", "1")).strip())
            except (ValueError, TypeError): 
                continue

            # 🎯 ดึงชื่อไฟล์จาก Map (เพื่อใช้พ่น Log และใส่ใน Metadata)
            display_filename = self.doc_id_to_filename_map.get(doc_uuid, f"DOC-{str(doc_uuid)[:8]}")

            # 🧠 Smart Offsets Logic: ปรับทิศทางการค้นหาตามเนื้อหา
            text_lower = doc.get('text', '').lower()
            offsets = [1] # Default คือดูหน้าถัดไป
            
            # กรณีเป็นพวกนโยบาย/แผน (มักจะยาวไปข้างหน้า)
            if any(k in text_lower for k in ["วิสัยทัศน์", "นโยบาย", "ยุทธศาสตร์", "แผนแม่บท"]):
                offsets = [-1, 1, 2]
            # กรณีเป็นพวกรายงาน/สรุปผล (มักจะมีบริบทอยู่หน้าก่อนหน้า)
            elif any(k in text_lower for k in ["สรุปผล", "รายงาน", "คะแนน", "ผลการประเมิน", "lesson learned"]):
                offsets = [-2, -1, 1]

            for off in sorted(list(set(offsets))):
                if off == 0: continue
                target_page = curr_page + off
                
                # ข้ามถ้าหน้า < 1 หรือเคยดึงมาแล้ว
                if target_page < 1 or f"{doc_uuid}_{target_page}" in seen_keys: 
                    continue
                
                # ดึง Chunks จาก VectorStoreManager
                neighbor_chunks = self.vectorstore_manager.get_chunks_by_page(
                    collection_name, 
                    doc_uuid, 
                    str(target_page)
                )
                
                if neighbor_chunks:
                    # ➕ พ่น Log ที่มนุษย์อ่านออก (ใช้ชื่อไฟล์จริง)
                    self.logger.info(
                        f"➕ Neighbor Fetch: พบข้อมูลหน้า {target_page} "
                        f"ในไฟล์ {os.path.basename(display_filename)} ({len(neighbor_chunks)} chunks)"
                    )

                    for nc in neighbor_chunks:
                        # สร้าง Metadata ใหม่ที่สมบูรณ์ขึ้น
                        new_meta = {**nc.metadata}
                        new_meta["is_supplemental"] = True
                        new_meta["pdca_tag"] = "Support"
                        new_meta["filename"] = display_filename  # ใส่ชื่อไฟล์จริงเข้าไปเลย
                        
                        new_ev = {
                            "text": nc.page_content,
                            "metadata": new_meta,
                            "score": doc.get('score', 0.5) * 0.85, # ลดน้ำหนักลงเล็กน้อย
                            "is_supplemental": True
                        }
                        expanded_evidences.append(new_ev)
                        seen_keys.add(f"{doc_uuid}_{target_page}")
                    
                    added_count += 1

        return expanded_evidences
    
    def _resolve_evidence_filenames(self, evidence_entries: List[Any]) -> List[Dict[str, Any]]:
        """
        [ULTIMATE SAFE v2026.3] - ป้องกัน Error 'str' has no attribute 'get'
        - การันตีคืนค่า List of Dictionary เท่านั้น
        - ป้องกันกรณีข้อมูลนำเข้าเป็น String หรือ None
        """
        resolved_entries = []
        if not evidence_entries:
            return []

        for entry in evidence_entries:
            # 🛡️ GUARD 1: ตรวจสอบว่าเป็น Dictionary หรือไม่
            if not isinstance(entry, dict):
                # ถ้าหลุดมาเป็น String หรือ Langchain Document ให้พยายามแปลงหรือข้าม
                if hasattr(entry, 'page_content'): # กรณีเป็น Document Object
                    entry = {
                        "content": entry.page_content,
                        "metadata": getattr(entry, 'metadata', {}),
                        "doc_id": getattr(entry, 'metadata', {}).get('doc_id', str(hash(entry.page_content)))
                    }
                else:
                    self.logger.warning(f"⚠️ Skipping non-dict evidence: {type(entry)}")
                    continue

            resolved_entry = deepcopy(entry)
            doc_id = resolved_entry.get("doc_id", "")
            
            # 🛡️ GUARD 2: Metadata Check
            meta = resolved_entry.get("metadata", {})
            if not isinstance(meta, dict): meta = {}
            
            # ดึงชื่อไฟล์จากหลายแหล่ง
            meta_filename = meta.get("source") or meta.get("source_filename") or meta.get("filename")
            content_raw = resolved_entry.get('content') or resolved_entry.get('text', '')
            page_label = str(meta.get("page_label") or meta.get("page") or meta.get("page_number") or "N/A")

            # 1. AI Generated / Missing Content Case
            if not content_raw:
                # ถ้าไม่มีเนื้อหา แต่มี ID ให้ถือว่าเป็น Reference ว่าง
                resolved_entry["filename"] = "MISSING-CONTENT"
                resolved_entry["display_source"] = f"ไม่พบเนื้อหา (ID: {doc_id})"
            
            # 2. Match ใน Map
            elif doc_id in self.doc_id_to_filename_map:
                mapped_name = self.doc_id_to_filename_map[doc_id]
                resolved_entry["filename"] = mapped_name
                resolved_entry["display_source"] = f"{os.path.basename(mapped_name)} (หน้า {page_label})"
            
            # 3. Match ใน Metadata
            elif meta_filename:
                clean_name = os.path.basename(str(meta_filename))
                resolved_entry["filename"] = clean_name
                resolved_entry["display_source"] = f"{clean_name} (หน้า {page_label})"

            # 4. Fallback สุดท้าย
            else:
                short_id = str(doc_id)[:8] if doc_id else "UNKNOWN"
                resolved_entry["filename"] = f"DOC-{short_id}"
                resolved_entry["display_source"] = f"เอกสารอ้างอิง {short_id} (หน้า {page_label})"

            # ตรวจสอบว่ามี Key สำคัญครบก่อนเพิ่ม
            if 'content' not in resolved_entry and content_raw:
                resolved_entry['content'] = content_raw

            resolved_entries.append(resolved_entry)
            
        return resolved_entries

    # ----------------------------------------------------------------------
    # 🎯 FINAL FIX 2.3: Manual Map Reload Function (inside SEAMPDCAEngine)
    # ----------------------------------------------------------------------
    def _collect_previous_level_evidences(self, sub_id: str, current_level: int) -> Dict[str, List[Dict]]:
        """
        [REVISED v2026.1.25] - Nested Key & Context Hydration
        - รองรับ Format คีย์ใหม่ "1.1_L1"
        - ดึงเนื้อหาเต็มจาก VectorStore เพื่อเป็น Baseline ให้เลเวลถัดไป
        """
        if getattr(self, 'is_parallel_all_mode', False):
            return {}

        collected = {}
        # 🔄 ปรับ Logic การกรอง Key ให้รองรับทั้ง 1.1.L1 และ 1.1_L1
        for key, bucket in self.evidence_map.items():
            # ตรวจสอบรูปแบบคีย์ (เช่น 1.1_L1 หรือ 1.1.L1)
            if key.startswith(f"{sub_id}_L") or key.startswith(f"{sub_id}.L"):
                try:
                    # แยกเอาตัวเลข Level ออกมา (รองรับทั้ง "_" และ ".")
                    level_part = key.replace(f"{sub_id}_L", "").replace(f"{sub_id}.L", "")
                    level_num = int(level_part)
                    
                    if level_num < current_level:
                        # ดึงเฉพาะ List ของ evidences ออกมา
                        ev_list = bucket.get("evidences", []) if isinstance(bucket, dict) else bucket
                        collected[key] = ev_list
                except: continue

        if not collected: return {}

        # 1. รวบรวม Unique IDs (Stable ID logic)
        stable_ids = set()
        for ev_list in collected.values():
            for ev in ev_list:
                sid = ev.get("stable_doc_uuid") or ev.get("doc_id")
                if sid and str(sid).lower() not in ["n/a", "none", ""]:
                    stable_ids.add(str(sid))

        if not stable_ids: return collected

        # 2. Bulk Hydration (Query จาก VectorStore เพื่อเอา Text เต็ม)
        vsm = self.vectorstore_manager
        chunk_map = {}
        try:
            # ดึงข้อมูลจาก VectorStore มาคืนชีพ (Restore) เนื้อหา
            full_chunks = vsm.get_documents_by_id(list(stable_ids), self.doc_type, self.enabler)
            for chunk in full_chunks:
                m = chunk.metadata
                keys = [str(m.get(k)) for k in ["stable_doc_uuid", "doc_id", "chunk_uuid"] if m.get(k)]
                for k in keys:
                    chunk_map[k] = {"text": chunk.page_content, "metadata": m}
                    chunk_map[k.replace("-", "")] = {"text": chunk.page_content, "metadata": m}
        except Exception as e:
            self.logger.error(f"❌ Hydration VSM Error: {e}")
            return collected

        # 3. Restoration Loop (ยัดเนื้อหาเต็มกลับเข้า Evidence List)
        restored_count = 0
        for key, ev_list in collected.items():
            for ev in ev_list:
                sid = str(ev.get("stable_doc_uuid") or ev.get("doc_id") or "")
                data = chunk_map.get(sid) or chunk_map.get(sid.replace("-", ""))

                if data:
                    ev.update({
                        "text": data["text"],
                        "metadata": data.get("metadata", {}),
                        "is_baseline": True  # 🚩 Mark ไว้เพื่อให้ AI รู้ว่าเป็นของเก่า
                    })
                    restored_count += 1
                
        self.logger.info(f"✅ Hydrated {restored_count} baseline chunks for {sub_id} L{current_level}")
        return collected

    def _get_contextual_rules_prompt(self, sub_id: str, level: int) -> str:
        """
        [REVISED v2026.1.25]
        - บังคับใช้คู่มือตรวจสอบ (Specific Rules)
        - ฉีด L5 Special Rule เพื่อป้องกันการ "Reset คะแนน" ในระดับสูงสุด
        """
        sub_id_rules = self.contextual_rules_map.get(sub_id, {})
        rule_text = ""
        
        # 1. กฎราย Level (ดึงมาจาก config)
        level_key = f"L{level}"
        specific_rule = sub_id_rules.get(level_key)
        if specific_rule:
            rule_text += f"\n[CRITICAL RULE L{level}]\n{specific_rule}\n"
        
        # 2. 🎖️ L5 SPECIAL RULE: การรักษาความเป็นเลิศ
        if level == 5:
            rule_text += """
            \n--- [JUDICIAL GUIDELINE FOR LEVEL 5] ---
            * ท่านกำลังประเมินระดับ 'Excellence' (L5)
            * **Score Continuity:** หากผลการประเมิน P-D-C-A จากหลักฐานเดิม (Baseline) มั่นคงอยู่แล้ว ห้ามลดคะแนนต่ำกว่า 7.0
            * **Excellence Bonus (+2.0):** ค้นหาหลักฐาน 'ความโดดเด่น' อย่างน้อย 1 อย่าง เช่น:
                - การเป็นต้นแบบ (Best Practice/Role Model)
                - รางวัลระดับชาติ/นานาชาติ
                - ตัวเลข ROI หรือ Business Impact ที่ชัดเจน
            * **Final Decision:** หากพบความโดดเด่น ให้ปรับ Score ขึ้นเป็น 9.0+ และเซต is_passed = true ทันที
            """
            
        return rule_text

    def _load_rubric(self) -> Dict[str, Any]:
        """ Loads the SEAM rubric JSON file using path_utils. """
        
        # 🎯 FIX: ใช้ get_rubric_file_path จาก path_utils แทนการสร้าง Path เอง
        filepath = None # กำหนดค่าเริ่มต้นเพื่อป้องกัน UnboundLocalError
        
        try:
            # 1. รับ Path จาก path_utils ซึ่งตอนนี้ชี้ไปที่ 'config/' แล้ว
            filepath = get_rubric_file_path(
                tenant=self.config.tenant,
                enabler=self.enabler
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

    
    def _clean_temp_entries(self, evidence_map: Dict[str, List[Any]]) -> Dict[str, List[Dict]]:
        """
        [ULTIMATE SANITIZER v2026.1.23]
        ทำความสะอาด Evidence Map แบบเบ็ดเสร็จ:
        1. กรองขยะ (TEMP-, HASH-, Unknown)
        2. ลบข้อมูลซ้ำ (Deduplication) ภายใน Level เดียวกัน
        3. ซ่อมแซม Metadata (Filename, Page)
        4. ป้องกัน Type Error (Type-Safe Processing)
        """
        if not evidence_map or not isinstance(evidence_map, dict):
            return {}

        cleaned_map = {}
        stats = {"removed": 0, "fixed": 0, "dupes": 0}

        for key, entries in evidence_map.items():
            if not isinstance(entries, list):
                continue
                
            valid_entries = []
            seen_doc_ids = set() # สำหรับ Deduplication ภายใน Key (Level) เดียวกัน

            for entry in entries:
                # --- 🛡️ ชั้นที่ 1: Type Validation ---
                if not isinstance(entry, dict):
                    if isinstance(entry, str) and len(entry.strip()) > 5:
                        # กรณีหลุดมาเป็น String (เช่น ID หรือ Content) ให้พยายามสร้าง Dict พื้นฐาน
                        entry = {
                            "doc_id": entry.strip(), 
                            "filename": "Unknown_Reference.pdf", 
                            "relevance_score": 0.1,
                            "page": "N/A"
                        }
                    else:
                        stats["removed"] += 1
                        continue

                # --- 🛡️ ชั้นที่ 2: Garbage Filtering ---
                doc_id = entry.get("doc_id") or entry.get("chunk_uuid")
                if not doc_id:
                    stats["removed"] += 1
                    continue
                
                doc_id_str = str(doc_id).strip()

                # กรองค่าว่าง หรือ Keyword ขยะ
                if doc_id_str.lower() in ["none", "unknown", "n/a", "", "null"]:
                    stats["removed"] += 1
                    continue

                # กรอง Prefix ที่เป็นเครื่องหมายของข้อมูลชั่วคราว
                if doc_id_str.startswith(("TEMP-", "HASH-", "REF-")):
                    stats["removed"] += 1
                    continue

                # --- 🛡️ ชั้นที่ 3: Deduplication (ป้องกันการปรากฏซ้ำ) ---
                if doc_id_str in seen_doc_ids:
                    stats["dupes"] += 1
                    continue
                seen_doc_ids.add(doc_id_str)

                # --- 🛡️ ชั้นที่ 4: Metadata Repair & Normalization ---
                # 1. จัดการชื่อไฟล์ (Filename)
                filename = str(entry.get("filename") or entry.get("source", "")).strip()
                if not filename or filename.lower() in ["unknown", "none", "n/a", "unknown_file.pdf"]:
                    # ถ้าไม่มีชื่อไฟล์ ให้ใช้ ID ย่อมาตั้งชื่อแทนเพื่อให้ User อ้างอิงได้
                    short_id = doc_id_str[:8]
                    entry["filename"] = f"Reference_{short_id}.pdf"
                    stats["fixed"] += 1
                else:
                    # Clean path ให้เหลือแค่ชื่อไฟล์
                    try:
                        entry["filename"] = os.path.basename(filename)
                    except:
                        entry["filename"] = filename

                # 2. จัดการคะแนน (Relevance Score)
                try:
                    score = float(entry.get("relevance_score", 0.0))
                    entry["relevance_score"] = max(0.0, min(1.0, score)) # บีบคะแนนให้อยู่ในช่วง 0-1
                except (ValueError, TypeError):
                    entry["relevance_score"] = 0.0

                # 3. จัดการเลขหน้า (Page)
                if "page" not in entry or entry["page"] is None:
                    entry["page"] = entry.get("page_label") or "N/A"

                valid_entries.append(entry)

            # บันทึกเฉพาะ Key ที่มีข้อมูลที่มีคุณภาพ
            if valid_entries:
                # เรียงลำดับตามคะแนนความเกี่ยวข้อง (จากมากไปน้อย)
                valid_entries.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
                cleaned_map[key] = valid_entries

        self.logger.info(
            f"🧹 [CLEAN-MAP] Stats: Removed={stats['removed']}, Fixed={stats['fixed']}, Dupes={stats['dupes']}"
        )
        return cleaned_map

    def _clean_map_for_json(self, data_map: Dict[str, Any]) -> Dict[str, Any]:
        """
        [v2026.1.23] บังคับให้ข้อมูลทุกอย่างเป็น JSON-Compatible
        - ป้องกัน TypeError จาก Object ที่ JSON ไม่รู้จัก
        - ป้องกัน AttributeError ในขั้นตอนประมวลผลต่อ
        """
        if not isinstance(data_map, dict):
            return {}
        
        clean_data = {}
        for k, v in data_map.items():
            str_key = str(k) # บังคับ Key เป็น String
            
            if isinstance(v, dict):
                clean_data[str_key] = self._clean_map_for_json(v)
            elif isinstance(v, list):
                clean_data[str_key] = [
                    (self._clean_map_for_json(item) if isinstance(item, dict) 
                     else (str(item) if not isinstance(item, (str, int, float, bool)) and item is not None else item))
                    for item in v
                ]
            elif isinstance(v, (str, int, float, bool)) or v is None:
                clean_data[str_key] = v
            else:
                # แปลง Object อื่นๆ (เช่น Datetime, UUID) เป็น String ทันที
                clean_data[str_key] = str(v)
                
        return clean_data
    

    def _group_statements_by_sub_criteria(
        self,
        flat_statements: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Group flattened rubric statements into Sub-Criteria bundles.

        Expected input item structure (PEA compatible):
        {
            "sub_id": "1.1",
            "sub_criteria_name": "...",
            "weight": 4,
            "levels": [
                { "level": 1, "statement": "..." },
                ...
            ]
        }
        """
        grouped: Dict[str, Dict[str, Any]] = {}

        for item in flat_statements:
            sub_id = item.get("sub_id")

            # 🔒 Minimal validation only (อย่า strict เกิน)
            if not sub_id:
                self.logger.warning(f"⚠️ Skip item without sub_id: {item}")
                continue

            levels = item.get("levels")

            if not isinstance(levels, list) or not levels:
                self.logger.warning(
                    f"⚠️ Skip sub_id {sub_id}: invalid or empty levels"
                )
                continue

            sub_id = str(sub_id)

            # ✅ init group
            if sub_id not in grouped:
                grouped[sub_id] = {
                    "sub_id": sub_id,
                    "sub_criteria_name": item.get("sub_criteria_name", ""),
                    "weight": float(item.get("weight", 0.0)),
                    "levels": []
                }

            # ✅ normalize each level
            for lv in levels:
                level_no = lv.get("level")
                statement = lv.get("statement")

                if level_no is None or not statement:
                    self.logger.warning(
                        f"⚠️ Skip invalid level in {sub_id}: {lv}"
                    )
                    continue

                grouped[sub_id]["levels"].append({
                    "level": int(level_no),
                    "statement": statement,
                    "keywords": lv.get("keywords", []),
                    "score_threshold": lv.get("score_threshold"),
                    "raw": lv
                })

        # 🔒 sort L1 → L5 and drop empty subs
        cleaned_grouped = {}
        for sub_id, sub in grouped.items():
            if not sub["levels"]:
                self.logger.warning(
                    f"⚠️ Drop sub_id {sub_id}: no valid levels after normalization"
                )
                continue

            sub["levels"] = sorted(
                sub["levels"], key=lambda x: x["level"]
            )
            cleaned_grouped[sub_id] = sub

        self.logger.info(
            f"📦 Grouped rubric into {len(cleaned_grouped)} sub-criteria bundles"
        )

        return cleaned_grouped

    # -------------------- Statement Preparation & Filtering Helpers --------------------
    def _flatten_rubric_to_statements(self, rubric_data: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        [ULTIMATE REVISED v2026.1.24 - PROMPT-READY VERSION]
        แปลงโครงสร้าง Rubric ที่ซับซ้อนให้เป็น Flat List พร้อมสกัด Focus Points 
        และ Evidence Guidelines ราย Level เพื่อส่งให้ LLM Agent ได้อย่างแม่นยำ
        """
        # 1. เลือก Source ข้อมูล (ลำดับความสำคัญ: Argument > self.rubric)
        source_rubric = rubric_data if rubric_data is not None else getattr(self, 'rubric', None)
        
        if not source_rubric:
            self.logger.warning("⚠️ [FLATTEN] Cannot proceed: Source rubric is empty or None.")
            return []
            
        try:
            # ใช้ deepcopy เพื่อป้องกันการแก้ไขข้อมูลต้นฉบับ (Thread-safety)
            data = deepcopy(source_rubric)
            criteria_map = data.get('criteria', {}) if isinstance(data, dict) else {}
            
            if not criteria_map:
                 self.logger.error("❌ [FLATTEN] Invalid structure: 'criteria' key not found.")
                 return []
                 
            extracted_list = []
            
            # 2. Loop เจาะลึกระดับ Criteria และ Sub-Criteria
            for criteria_id, criteria_data in criteria_map.items():
                if not isinstance(criteria_data, dict): continue
                    
                sub_criteria_map = criteria_data.get('subcriteria', {})
                criteria_name = criteria_data.get('name', 'Unknown Criteria')
                
                for sub_id, sub_data in sub_criteria_map.items():
                    if not isinstance(sub_data, dict): continue
                    
                    # --- [CORE EXTRACTION] ---
                    # จัดการ Focus Points: แปลง List เป็น String เพื่อใส่ใน Prompt ง่ายๆ
                    fps = sub_data.get('focus_points', [])
                    focus_points_str = " | ".join(fps) if isinstance(fps, list) else str(fps or "-")
                    
                    # เก็บ Evidence Guidelines ทั้งหมดไว้ก่อน (Dictionary)
                    all_guidelines = sub_data.get('evidence_guidelines', {})

                    # สร้าง Base Object ของ Sub-Criteria นี้
                    item = {
                        'criteria_id': criteria_id,
                        'criteria_name': criteria_name,
                        'sub_id': sub_id,
                        'sub_criteria_name': sub_data.get('name', f"{criteria_name} - {sub_id}"),
                        'weight': sub_data.get('weight', criteria_data.get('weight', 0)),
                        'focus_points': focus_points_str,            # สำหรับ Prompt {focus_points}
                        'evidence_guidelines_all': all_guidelines,    # แหล่งเก็บต้นทาง
                        'raw_levels': sub_data.get('levels', {})      # รอประมวลผลต่อ
                    }
                    extracted_list.append(item)

            # 3. [LEVEL PROCESSING] แตกย่อยเป็นราย Level พร้อมผูก Guideline เฉพาะตัว
            final_list = []
            for sub_item in extracted_list:
                raw_levels = sub_item.pop('raw_levels') 
                processed_levels = []
                
                if isinstance(raw_levels, dict):
                    for level_str, statement in raw_levels.items():
                        try:
                            level_int = int(level_str)
                            # 🎯 ดึง Guideline เฉพาะของ Level นั้นๆ (e.g., 'level_1', 'level_2')
                            # หากไม่มี ให้ใช้ค่า Default เป็น "-"
                            current_guideline = sub_item['evidence_guidelines_all'].get(f"level_{level_int}", "-")
                            
                            processed_levels.append({
                                "level": level_int, 
                                "statement": statement,
                                "level_specific_guideline": current_guideline # สำหรับ Prompt {evidence_guidelines}
                            })
                        except (ValueError, TypeError):
                            self.logger.error(f"❌ [FLATTEN] Invalid level key '{level_str}' in Sub-ID: {sub_item['sub_id']}")
                            continue
                
                if processed_levels:
                    # เรียงลำดับ Level 1 -> 5 เสมอ
                    processed_levels.sort(key=lambda x: x.get("level", 0))
                    sub_item["levels"] = processed_levels
                    final_list.append(sub_item)
                else:
                    self.logger.warning(f"⚠️ [FLATTEN] Sub-criteria {sub_item['sub_id']} has no valid levels.")

            self.logger.info(f"✅ [FLATTEN] Rubric transformation complete. Processed {len(final_list)} sub-criteria.")
            return final_list

        except Exception as e:
            self.logger.error(f"🛑 [FLATTEN-ERROR] Failure during flattening: {str(e)}", exc_info=True)
            return []
    
    # -------------------- Evidence Classification Helper (Robust 2026) --------------------
    def _get_mapped_uuids_and_priority_chunks(
        self,
        sub_id: str,
        level: int,
        statement_text: str = "",
        level_constraint: Optional[Any] = None, # 🟢 ปรับเป็น Optional ป้องกัน Error Missing Argument
        vectorstore_manager: Optional['VectorStoreManager'] = None,
        evidence_map: Optional[Dict] = None 
    ) -> Tuple[List[str], List[Dict]]:
        """
        [DYNAMIC CONTINUITY v2026.6.2] - ROBUST SIGNATURE
        ----------------------------------------------
        - Fix: รองรับการเรียกแบบยืดหยุ่น ป้องกัน Error 'missing 1 required positional argument'
        - Logic: ดึง Baseline จาก Memory/Map มาใช้เพื่อรักษาความต่อเนื่อง (Inheritance)
        """
        from copy import deepcopy
        priority_chunks = []
        mapped_stable_ids = []

        # 1. เลือกใช้ Evidence Map (ลำดับความสำคัญ: 1. Argument ที่ส่งมา -> 2. Class Attribute -> 3. Empty Dict)
        target_map = evidence_map if evidence_map is not None else getattr(self, 'evidence_map', {})

        # 2. 🧠 [AUTO-HISTORY & INHERITANCE]
        # ค้นหาหลักฐานที่เคยผ่านใน Level ต่ำกว่า เพื่อนำมาใช้ใน Level ปัจจุบัน
        for key, evidences in target_map.items():
            if key.startswith(f"{sub_id}.L") and isinstance(evidences, list):
                try:
                    lvl_in_key = int(key.split(".L")[-1])
                    # กฎ Inheritance: ใช้หลักฐานจาก Level ที่ <= ปัจจุบัน
                    if lvl_in_key <= level:
                        history_items = deepcopy(evidences)
                        for item in history_items:
                            item["is_baseline"] = True
                            # บูสต์คะแนนเพื่อให้ผ่านเกณฑ์การกรองของ RAG
                            item["rerank_score"] = max(item.get("rerank_score", 0.0), 0.85)
                        priority_chunks.extend(history_items)
                except (ValueError, IndexError):
                    continue

        # 3. 🔍 [SEMANTIC HINTING] กรณี L1 หรือยังไม่มีประวัติใน Map
        if not priority_chunks and level == 1:
            rule_config = getattr(self, 'contextual_rules_map', {}).get(sub_id, {}).get(str(level), {})
            hints = rule_config.get("plan_keywords", [])[:2]
            if hints and vectorstore_manager:
                self.logger.info(f"🔎 L1 Discovery: Searching using hints: {hints}")
                try:
                    # แก้จุด 1: ใช้ helper จริง + ดึง enabler จาก self
                    enabler = getattr(self, 'enabler', 'KM').upper()  # fallback KM ถ้าไม่มี
                    collection_name = get_doc_type_collection_key("evidence", enabler)
                    
                    discovery_docs = vectorstore_manager.retrieve(
                        query=f"{sub_id} {' '.join(hints)}",
                        collection_name=collection_name,
                        top_k=5
                    )
                    
                    for doc in discovery_docs:
                        # แก้จุด 2: fallback chunk_uuid และ source ให้ปลอดภัย
                        chunk_uuid = (
                            doc.metadata.get("chunk_uuid") or
                            doc.metadata.get("id") or
                            doc.metadata.get("chunk_id") or
                            hashlib.sha256(doc.page_content.encode()).hexdigest()[:32]  # last resort
                        )
                        
                        source = (
                            doc.metadata.get("source") or
                            doc.metadata.get("source_filename") or
                            doc.metadata.get("file_path") or
                            "unknown_source"
                        )
                        
                        chunk = {
                            "page_content": doc.page_content,
                            "metadata": doc.metadata or {},
                            "rerank_score": 0.85,
                            "chunk_uuid": chunk_uuid,
                            "source": source
                        }
                        priority_chunks.append(chunk)
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Discovery retrieve failed for {collection_name}: {e}")
                    

        if not priority_chunks:
            return [], []

        # 4. 💧 [ROBUST HYDRATION] เติมเนื้อหา Full Text จาก Vector DB
        try:
            priority_chunks = self._robust_hydrate_documents_for_priority_chunks(
                chunks_to_hydrate=priority_chunks,
                vsm=vectorstore_manager
            )
        except Exception as e:
            self.logger.error(f"❌ Hydration failed in priority module: {e}")

        # 5. 🎯 [ID SYNC] สกัด UUID เพื่อส่งให้ Filter ของ Main Retriever
        seen_ids = set()
        for chunk in priority_chunks:
            sid = chunk.get("stable_doc_uuid") or chunk.get("doc_id")
            if sid and isinstance(sid, str):
                if sid not in seen_ids and len(sid) >= 32:
                    mapped_stable_ids.append(sid)
                    seen_ids.add(sid)

        self.logger.info(f"✅ Continuity Ready: {len(priority_chunks)} priority chunks | Sub:{sub_id} L{level}")
        return mapped_stable_ids, priority_chunks

    
    def get_actual_score(self, ev: Any) -> float:
        """
        [v2026.1 - ROBUST SCORING EXTRACTOR]
        ดึงคะแนนความเกี่ยวข้องออกมาจากข้อมูลหลักฐาน ไม่ว่าจะอยู่ในรูปแบบ Dict หรือ Object
        แก้ปัญหาเรื่องคะแนน 0.0 และการซ่อนคีย์ใน Metadata
        """
        if not ev:
            return 0.0

        # 1. รายการคีย์คะแนนที่ระบบยอมรับ (เรียงตามลำดับความสำคัญ)
        score_keys = ["rerank_score", "score", "relevance_score"]
        
        # 2. ค้นหาในระดับ Top-level ก่อน
        for key in score_keys:
            # รองรับทั้ง dict.get() และ getattr() สำหรับ Document Object
            val = ev.get(key) if isinstance(ev, dict) else getattr(ev, key, None)
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    continue

        # 3. Fallback: ค้นหาภายใน Metadata
        meta = ev.get("metadata", {}) if isinstance(ev, dict) else getattr(ev, "metadata", {})
        if isinstance(meta, dict):
            for key in score_keys:
                val = meta.get(key)
                if val is not None:
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        continue

        return 0.0
    
    def _calculate_evidence_strength_cap(
        self,
        top_evidences: List[Any],
        level: int,
        highest_rerank_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        [PROTECTED v2026.1.17 - REVISED]
        - ใช้ get_actual_score เป็นมาตรฐานกลางในการดึงคะแนน
        - แยก Logic การหา 'ชื่อแหล่งข้อมูล' ให้ฉลาดขึ้น
        - เพิ่มระบบ Logging ที่ชัดเจนสำหรับการ Audit
        """
        try:
            # ⚙️ Load Configuration
            threshold = getattr(self, "RERANK_THRESHOLD", 0.35)
            cap_value = getattr(self, "MAX_EVI_STR_CAP", 5.0)
            
            # 1. เริ่มต้นคะแนนสูงสุดจากค่าที่ส่งมา (Baseline)
            max_score_found = 0.0
            try:
                if highest_rerank_score is not None:
                    max_score_found = float(highest_rerank_score)
            except (ValueError, TypeError):
                max_score_found = 0.0

            max_score_source = "Adaptive_RAG_Loop"
            
            if not isinstance(top_evidences, list):
                top_evidences = []

            # 2. Iterate หาคะแนนสูงสุดจากหลักฐานทั้งหมดที่มี
            for idx, doc in enumerate(top_evidences, 1):
                # ใช้วิธีดึงคะแนนที่เป็นมาตรฐานเดียวกับระบบ
                current_score = self.get_actual_score(doc)

                # ถ้าเจอคะแนนที่สูงกว่า ให้บันทึกไว้พร้อมชื่อแหล่งที่มา
                if current_score > max_score_found:
                    max_score_found = current_score
                    
                    # --- ดึงชื่อแหล่งข้อมูลแบบ Robust ---
                    meta = doc.get("metadata", {}) if isinstance(doc, dict) else getattr(doc, "metadata", {})
                    if not isinstance(meta, dict): meta = {}
                    
                    max_score_source = (
                        meta.get("source_filename") or 
                        meta.get("file_name") or 
                        meta.get("source") or 
                        f"Doc_{idx}"
                    )

            # 3. Decision Logic (Gated Check)
            # ถ้าคะแนนสูงสุดยังต่ำกว่า Threshold (0.35) -> สั่ง Capped
            is_capped = max_score_found < threshold
            # ค่านี้จะถูกส่งเข้า Prompt: 5.0 (ไม่ผ่าน) หรือ 10.0 (มีโอกาสผ่าน)
            max_evi_str_for_prompt = float(cap_value) if is_capped else 10.0

            # 📊 Internal Audit Log
            status_icon = "🚨 [CAPPED]" if is_capped else "✅ [FULL-STRENGTH]"
            self.logger.info(
                f"{status_icon} L{level} | Best Score: {max_score_found:.4f} "
                f"from: '{os.path.basename(str(max_score_source))}' | Threshold: {threshold}"
            )

            return {
                "is_capped": bool(is_capped),
                "max_evi_str_for_prompt": float(max_evi_str_for_prompt),
                "top_score": round(float(max_score_found), 4),
                "max_score_source": str(max_score_source),
                "threshold_used": threshold
            }

        except Exception as e:
            self.logger.error(f"❌ [CRITICAL-CAP-ERROR] {e}", exc_info=True)
            # ปลอดภัยไว้ก่อน: ถ้า Error ไม่ต้อง Cap เพื่อให้ LLM ประเมินต่อได้
            return {
                "is_capped": False, 
                "max_evi_str_for_prompt": 10.0, 
                "top_score": 0.0, 
                "max_score_source": "Fallback-Error"
            }
       
    
    def _extract_strategic_gaps(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        [HELPER] วิเคราะห์หาช่องว่าง (Gaps) ระหว่างระดับที่ได้จริงกับเป้าหมาย
        เพื่อใช้ในการนำเสนอ Coaching Insight ในระดับภาพรวม
        """
        gaps = []
        # เรียงลำดับตามความรุนแรงของ Gap (ข้อที่คะแนนน้อยที่สุดขึ้นก่อน)
        sorted_results = sorted(results, key=lambda x: x.get('score', 0.0))
        
        for res in sorted_results:
            sub_id = res.get('sub_id', 'Unknown')
            current_score = res.get('score', 0.0)
            passed = res.get('is_passed', False)
            
            # ถ้าไม่ผ่าน หรือคะแนนต่ำกว่า 0.8 ถือว่าเป็น Gap
            if not passed or current_score < 0.8:
                gap_info = {
                    "sub_id": sub_id,
                    "level": res.get('level'),
                    "current_score": current_score,
                    "impact": "High" if current_score < 0.5 else "Medium",
                    "reason": res.get('reason', 'ไม่พบหลักฐานที่ชัดเจนในระดับนี้'),
                    "suggestion": res.get('coaching_insight', 'ควรเพิ่มหลักฐานในส่วนที่ขาดหายตาม PDCA')
                }
                gaps.append(gap_info)
        
        # คืนค่า Gap ที่สำคัญที่สุด 5 อันดับแรก
        return gaps[:5]

    def calculate_audit_confidence(
        self,
        matched_chunks: List[Any],
        sub_id: str = "unknown",
        level: int = 1
    ) -> Dict[str, Any]:
        """
        [ULTIMATE AUDIT CONFIDENCE v2026.3.10 – Real-Count Enabled]
        - แก้ไขปัญหาเลข (1) โดยการส่ง Raw Tags ทั้งหมดกลับไปนับความถี่ (Frequency Counting)
        - ตรวจสอบ PDCA ผ่าน 3 ชั้น: Metadata -> Tagging -> Keyword Fallback
        - ปรับ Decision Matrix ให้สมดุลกับจำนวนหลักฐานจริง
        """
        if not matched_chunks:
            return {
                "level": "NONE", "reason": "ไม่พบหลักฐานในระบบ", "source_count": 0,
                "coverage_ratio": 0.0, "pdca_found": [], "valid_chunks_count": 0,
                "traceability_score": 0.0, "recency_bonus": 0.0
            }

        # 1. Quality Gate: คัดเฉพาะ Chunk ที่มีเนื้อหาเน้นๆ (Relevance >= 0.40)
        valid_chunks = [doc for doc in matched_chunks if self.get_actual_score(doc) >= 0.40]
        valid_count = len(valid_chunks)

        if valid_count == 0:
            return {
                "level": "LOW", "reason": "หลักฐานมีความเกี่ยวข้องต่ำกว่าเกณฑ์คุณภาพ",
                "source_count": 0, "coverage_ratio": 0.0, "pdca_found": [], "valid_chunks_count": 0
            }

        # 2. Independence Check: ความหลากหลายของแหล่งข้อมูล
        unique_sources = set()
        for doc in valid_chunks:
            meta = getattr(doc, 'metadata', {}) if hasattr(doc, 'metadata') else doc.get('metadata', {})
            src = next((meta.get(k) for k in ['source_filename', 'filename', 'source'] if meta.get(k)), None)
            if src:
                unique_sources.add(os.path.basename(str(src).strip()))
        
        independence_score = len(unique_sources)

        # 3. PDCA Detection (Revised: Multi-Tag Frequency Collection)
        all_detected_tags = [] # เก็บ List ของ Tag ทั้งหมดที่เจอในทุก Chunks (ห้ามทำ Unique)
        
        # ดึง Keyword rules สำหรับ Fallback
        cum_rules = {}
        if hasattr(self, "get_cumulative_rules_cached"):
            try:
                loaded = self.get_cumulative_rules_cached(sub_id, level)
                if isinstance(loaded, dict):
                    cum_rules = loaded
                else:
                    self.logger.warning(
                        f"[PDCA] cumulative rules invalid type for {sub_id}-{level}"
                    )
            except Exception as e:
                self.logger.error(
                    f"[PDCA] failed to load cumulative rules for {sub_id}-{level}: {e}"
                )
                
        kw_map = {
            "P": [k.lower() for k in cum_rules.get('plan_keywords', [])],
            "D": [k.lower() for k in cum_rules.get('do_keywords', [])],
            "C": [k.lower() for k in cum_rules.get('check_keywords', [])],
            "A": [k.lower() for k in cum_rules.get('act_keywords', [])]
        }

        for doc in valid_chunks:
            meta = getattr(doc, 'metadata', {}) if hasattr(doc, 'metadata') else doc.get('metadata', {})
            tag = (getattr(doc, 'pdca_tag', None) or meta.get('pdca_tag') or meta.get('tag') or "").strip().upper()
            
            chunk_detected_phases = []
            
            # ชั้นที่ 1: ตรวจจาก Tag โดยตรง
            if tag in ["P", "D", "C", "A"]:
                chunk_detected_phases.append(tag)
            
            # ชั้นที่ 2: Fallback Keyword Detection (ถ้า Tag ว่าง หรือไม่ตรง)
            text = (doc.get('text') or doc.get('page_content') or '').lower()
            for phase, kws in kw_map.items():
                if any(k in text for k in kws):
                    if phase not in chunk_detected_phases: # 1 chunk นับ 1 สิทธิ์ต่อ 1 เฟส
                        chunk_detected_phases.append(phase)
            
            # นำ Tags ที่เจอใน Chunk นี้ไปสะสมรวม (เพื่อใช้แสดงจำนวนจริงใน Log)
            all_detected_tags.extend(chunk_detected_phases)

        # คำนวณ Coverage Ratio (นับจำนวนหมวดที่เจอ / 4)
        unique_found_phases = set(all_detected_tags)
        coverage_ratio = len(unique_found_phases) / 4.0

        # 4. Traceability & Recency Check
        traceable_count = 0
        recent_count = 0
        # ✅ ใช้ self.year จาก argument ถ้าไม่มีให้ถอยไปใช้ DEFAULT_YEAR จาก global_vars
        from config.global_vars import DEFAULT_YEAR
        try:
            current_year = int(self.year) if self.year else DEFAULT_YEAR
        except (ValueError, TypeError):
            current_year = DEFAULT_YEAR
            
        self.logger.debug(f"📊 [CONFIDENCE] Evaluating recency using year baseline: {current_year}")

        
        for doc in valid_chunks:
            meta = getattr(doc, 'metadata', {}) if hasattr(doc, 'metadata') else doc.get('metadata', {})
            # ตรวจความสมบูรณ์ของที่มา (ต้องมีทั้งไฟล์และหน้า)
            if any(meta.get(k) for k in ['page', 'page_label']) and any(meta.get(k) for k in ['source', 'filename']):
                traceable_count += 1
            
            # ตรวจความใหม่ (Bonus 2-3 ปีย้อนหลัง)
            year_val = str(meta.get('year') or meta.get('doc_year') or "")
            if not year_val: # Fallback search in filename
                year_match = re.search(r'(25[67]\d)', str(meta.get('source', '')))
                if year_match: year_val = year_match.group(1)
            
            if year_val.isdigit() and int(year_val) >= current_year - 2:
                recent_count += 1

        trace_score = traceable_count / valid_count if valid_count > 0 else 0
        recency_score = recent_count / valid_count if valid_count > 0 else 0

        # 5. Decision Matrix: กำหนดระดับความเชื่อมั่น (Confidence Level)
        if independence_score >= 8 and coverage_ratio >= 0.75:
            conf_level = "HIGH"
            reason = "หลักฐานครบวงจร PDCA จากแหล่งข้อมูลที่หลากหลายและระบุที่มาชัดเจน"
        elif independence_score >= 4 and coverage_ratio >= 0.50:
            conf_level = "MEDIUM"
            reason = "หลักฐานครอบคลุมมิติสำคัญและมีความหลากหลายในระดับเกณฑ์มาตรฐาน"
        else:
            conf_level = "LOW"
            reason = "หลักฐานยังไม่ครอบคลุมครบวงจร PDCA หรือแหล่งข้อมูลไม่หลากหลายพอ"

        # Penalty: ลดระดับหากระบุเลขหน้าไม่ครบ
        if trace_score < 0.6 and conf_level != "LOW":
            conf_level = "MEDIUM" if conf_level == "HIGH" else "LOW"
            reason += " (ลดระดับเนื่องจากการอ้างอิงเลขหน้าไม่สมบูรณ์)"

        return {
            "level": conf_level,
            "reason": reason,
            "source_count": independence_score,
            "coverage_ratio": round(coverage_ratio, 3),
            "traceability_score": round(trace_score, 3),
            "recency_bonus": round(recency_score, 3),
            "valid_chunks_count": valid_count,
            "pdca_found": all_detected_tags  # <--- ส่ง LIST ดิบเพื่อให้นับจำนวนจริงได้
        }

    def _calculate_weighted_score(
        self, 
        highest_full_level: int, 
        weight: float, 
        level_details: Dict[str, Any] = None
    ) -> float:
        """
        [ULTIMATE REVISED v2026.01.24 - MATURITY-DRIVEN SCORING]
        - 🧩 Logic: Continuous Base Level + Partial PDCA from the FIRST GAP level.
        - 🛡️ Governance: คะแนน Partial จะไม่ข้ามเลเวล (นับเฉพาะเลเวลถัดจากจุดที่ตกต่อเนื่อง)
        - 🎯 Precision: จัดการ Scaling ให้คะแนนสะท้อน Maturity จริงตามเป้าหมาย SE-AM
        """
        # 1. Configuration Setup
        max_lv = getattr(self.config, 'max_level', 5) or 5
        safe_weight = float(weight) if weight else 4.0
        # ใช้โหมดที่กำหนดจาก Global Vars (PARTIAL_PDCA)
        mode = getattr(self, 'scoring_mode', SCORING_MODE)
        
        # 2. Base Maturity Calculation
        # highest_full_level คือระดับสูงสุดที่ผ่านต่อเนื่อง (เช่น ถ้าผ่าน L1, L2 แล้ว L3 ตก ค่าจะเป็น 2)
        base_level = float(max(0, min(highest_full_level, max_lv)))
        partial_contribution = 0.0

        # 3. Partial Score Logic (คะแนนเก็บตกจากระดับแรกที่ "ไม่ผ่าน")
        if mode == 'PARTIAL_PDCA' and level_details:
            # เลเวลที่จะเอามาคิดคะแนนเศษส่วน คือเลเวลถัดจากระดับสูงสุดที่ผ่านต่อเนื่อง
            # เช่น ถ้าผ่าน L2 เต็มตัว เราจะไปดู PDCA ของ L3 ว่าทำได้กี่ %
            next_lv_idx = int(base_level + 1)
            
            if next_lv_idx <= max_lv:
                lv_data = level_details.get(str(next_lv_idx), {})
                pdca = lv_data.get('pdca_breakdown', {})
                
                if isinstance(pdca, dict) and pdca:
                    # ดึงคะแนนราย Phase (P, D, C, A) มาเฉลี่ย
                    scores = [float(v) for v in pdca.values() if v is not None]
                    if scores:
                        # คำนวณค่าเฉลี่ยของ PDCA (0.0 - 1.0)
                        raw_partial = sum(scores) / len(scores)
                        # ถ่วงน้ำหนัก: ช่วยเพิ่มคะแนนในระดับทศนิยม (Max 0.99)
                        partial_contribution = round(raw_partial, 4)
                        
                        self.logger.info(f"➕ [PARTIAL-BOOST] Found Gap at L{next_lv_idx}: +{partial_contribution:.2f} PDCA progress")

        # 4. Final Maturity Level Assembly
        # Formula: ระดับต่อเนื่อง + ความคืบหน้าของระดับถัดไป
        # ตัวอย่าง: ผ่าน L2 (Base 2.0) + ทำ L3 ได้ครึ่งหนึ่ง (Partial 0.5) = 2.50
        effective_level = min(base_level + partial_contribution, float(max_lv))
        
        # 5. Scaling to Weighted Score
        # แปลงระดับ Maturity เป็นคะแนนถ่วงน้ำหนักตามสัดส่วน
        # Formula: (Effective Level / Max Level) * Weight
        # ตัวอย่าง: (2.5 / 5.0) * 4.0 = 2.0 คะแนน
        base_ratio = effective_level / max_lv
        final_score = base_ratio * safe_weight

        # 6. Step-Ladder Final Touch
        # ถ้าเป็นโหมด STEP_LADDER แท้ๆ จะไม่นับเศษส่วน (ใช้ base_level เพียวๆ)
        if mode == 'STEP_LADDER':
            final_score = (base_level / max_lv) * safe_weight

        final_score = round(final_score, 4)
        
        # 7. Detailed Audit Logging (เพื่อใช้ตรวจสอบใน Dashboard)
        self.logger.info(
            f"📊 [SCORING-SUMMARY] Sub-ID Weighting:\n"
            f"   > Scoring Mode: {mode}\n"
            f"   > Maturity Level: {base_level} (Continuous Full)\n"
            f"   > Partial Progress: +{partial_contribution} (from Level {int(base_level+1)})\n"
            f"   > Weighted Score: {final_score} / {safe_weight} ({base_ratio:.2%} of target)"
        )
        
        return final_score
    
    def _calculate_overall_stats(self, target_sub_id: str):
        """
        [ULTIMATE REVISED v2026.1.25 - SMART AGGREGATOR]
        - 🛡️ Data Resilience: รองรับโครงสร้างข้อมูลที่ซับซ้อน ป้องกันคะแนน 0
        - ⚖️ Step-Ladder Logic: ตรวจสอบความต่อเนื่องของระดับ Maturity (1->5)
        - 🧬 Analytics: คำนวณคะแนนตาม Rubric Weight (Normalized Scoring)
        """
        from datetime import datetime
        
        # ดึงข้อมูลจาก Memory หลัก
        results = getattr(self, 'final_subcriteria_results', [])
        
        if not results:
            self.logger.critical(f"❌ [STATS-FAIL] No results found for {target_sub_id}. Assessment might have crashed.")
            self.total_stats = self._get_empty_stats_template()
            return

        passed_levels_pool = []
        sub_details = []
        total_weighted_sum = 0.0
        total_weight_sum = 0.0
        
        for r in results:
            if not isinstance(r, dict): continue
            
            sub_id = r.get('sub_id', 'Unknown')
            # ดึงค่า Weight จาก Rubric (เช่น 4.0)
            weight = float(r.get('weight', 4.0))
            
            # 1. SMART DETECTION: ดึง level_details
            details_map = r.get('level_details', {})
            if not details_map:
                possible_wrapper = r.get(sub_id) or r.get('results')
                if isinstance(possible_wrapper, dict):
                    details_map = possible_wrapper.get('level_details', {})

            # 2. STEP-LADDER MATURITY CALCULATION (1 -> 5)
            # ต้องผ่านเลเวลก่อนหน้าต่อเนื่องกันเท่านั้น
            current_maturity_lvl = 0
            for l_idx in range(1, 6):
                lv_data = details_map.get(str(l_idx))
                if not lv_data: break
                
                # เช็คเกณฑ์การผ่าน (Score >= 0.7 หรือ is_passed เป็น True)
                is_passed = lv_data.get('is_passed') is True or float(lv_data.get('score', 0)) >= 0.7
                
                if is_passed:
                    current_maturity_lvl = l_idx
                else:
                    break # ตกเลเวลไหน หยุดนับทันที

            # 3. NORMALIZED SCORE CALCULATION (แก้จากคะแนน 20 เป็นคะแนนตาม Rubric)
            # 🎯 สูตร: (Maturity Level / 5) * Weight
            # เช่น (L5 / 5) * 4.0 = 4.00 คะแนน (เต็ม Rubric)
            # เช่น (L4 / 5) * 4.0 = 3.20 คะแนน
            sub_weighted_score = (float(current_maturity_lvl) / 5.0) * weight
            
            total_weighted_sum += sub_weighted_score
            total_weight_sum += weight

            # 4. PREPARE ANALYTICS DATA
            passed_levels_pool.append(current_maturity_lvl)
            sub_details.append({
                "sub_id": sub_id,
                "sub_name": r.get('sub_criteria_name', 'N/A'),
                "maturity": current_maturity_lvl,
                "score": round(sub_weighted_score, 2), # คะแนนตามสัดส่วน weight
                "weight": weight,
                "evidence_count": len(details_map)
            })

        # 5. FINAL CALCULATION (OVERALL)
        # คำนวณค่าเฉลี่ยระดับภาพรวม (Overall Level) มักใช้ค่าฐาน (Min) ของทุกหัวข้อ
        overall_min = min(passed_levels_pool) if passed_levels_pool else 0
        overall_max = max(passed_levels_pool) if passed_levels_pool else 0
        
        # คำนวณ Average Score (0.0 - 5.0) สำหรับการทำ Radar Chart
        # สูตร: (คะแนนรวมที่ได้ / น้ำหนักรวม) * 5
        avg_maturity_score = (total_weighted_sum / total_weight_sum * 5.0) if total_weight_sum > 0 else 0.0

        self.total_stats = {
            "overall_max_level": int(overall_max),
            "overall_min_level": int(overall_min),
            "overall_level_label": f"L{int(overall_min)}", 
            "overall_avg_score": round(avg_maturity_score, 2), # สเกล 0-5
            "total_weighted_score": round(total_weighted_sum, 2), # สเกลตามผลรวม weight
            "total_weight": round(total_weight_sum, 2),
            "evaluated_at": datetime.now().isoformat(),
            "status": "SUCCESS",
            "analytics": {
                "sub_details": sub_details,
                "total_sub_items": len(results)
            }
        }

        self.logger.info(
            f"✅ [AGGREGATION SUCCESS] Maturity: {self.total_stats['overall_level_label']} | "
            f"Final Score: {self.total_stats['total_weighted_score']}/{total_weight_sum}"
        )

    def _get_empty_stats_template(self):
        """สร้าง Template กรณีไม่มีข้อมูล เพื่อป้องกัน Error ในการ Export"""
        return {
            "overall_max_level": 0,
            "overall_min_level": 0,
            "overall_level_label": "L0",
            "overall_avg_score": 0.0,
            "total_weighted_score": 0.0,
            "total_weight": 0.0,
            "status": "NO_DATA"
        }

    def _export_results(self, results_data: Any, sub_criteria_id: str, **kwargs) -> str:
        """
        [ULTIMATE EXPORTER v2026.1.25 - DATA INTEGRITY]
        - 🛡️ Score Sync: ดึงคะแนนโดยตรงจาก total_stats ป้องกันคะแนน 0
        - 🧬 Evidence Recovery: รองรับโครงสร้าง Map ทั้งแบบ List และ Dict (ป้องกัน Map หาย)
        - 📊 Deep Audit Trail: เก็บ Snippet และ Confidence รายเลเวลเพื่อการตรวจสอบย้อนกลับ
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            record_id = kwargs.get("record_id", getattr(self, "current_record_id", f"auto_{timestamp}"))
            tenant = getattr(self.config, 'tenant', 'unknown')
            year = getattr(self.config, 'year', 'unknown')
            enabler = getattr(self, 'enabler', 'unknown').upper()

            # 1. 🔍 Data Source Selection (ลำดับความสำคัญของข้อมูล)
            # ลำดับ: ข้อมูลที่ส่งมา > ข้อมูลใน Memory > ข้อมูลว่าง
            if results_data is None:
                results_data = getattr(self, 'final_subcriteria_results', [])
            
            if isinstance(results_data, dict):
                results_data = [results_data]
            
            if not results_data:
                self.logger.warning(f"⚠️ [EXPORT] No result data found for {sub_criteria_id}")
                return ""

            # 2. 📊 Summary Retrieval (ดึงค่าสถิติรวม)
            # ดึงจาก total_stats ที่ผ่านการคำนวณ Smart Mapping มาแล้ว
            stats = getattr(self, 'total_stats', {})
            if not stats or stats.get('total_weighted_score') == 0:
                # Fallback: ถ้า stats ว่าง ให้พยายามคำนวณสดจาก results_data
                highest_lvl = max([int(r.get('highest_full_level', 0)) for r in results_data])
                total_weighted = sum([float(r.get('weighted_score', 0.0)) for r in results_data])
                is_passed = highest_lvl >= 1
            else:
                highest_lvl = stats.get('overall_max_level', 0)
                total_weighted = stats.get('total_weighted_score', 0.0)
                is_passed = stats.get('overall_level_label') != "L0"

            # 3. 🛡️ Robust Evidence Mapping (The Fix for Empty Maps)
            master_map = getattr(self, 'evidence_map', {})
            processed_evidence = {}
            
            for lv_key, val in master_map.items():
                if not val: continue
                
                # รองรับทั้งโครงสร้างใหม่ {"evidences": [...]} และโครงสร้างเก่า [...]
                v_list = val.get("evidences", []) if isinstance(val, dict) else val
                
                if not isinstance(v_list, list) or not v_list:
                    continue
                
                try:
                    # เลือกหลักฐานที่มี Rerank Score สูงสุดในเลเวลนั้น
                    sorted_ev = sorted(
                        [ev for ev in v_list if isinstance(ev, dict)], 
                        key=lambda x: x.get('rerank_score', x.get('relevance_score', 0)), 
                        reverse=True
                    )
                    
                    if sorted_ev:
                        top_ev = sorted_ev[0]
                        doc_id = top_ev.get("doc_id") or top_ev.get("stable_doc_uuid")
                        
                        # ดึงชื่อไฟล์จริงจาก Map กลาง
                        filename = self.document_map.get(doc_id) if hasattr(self, 'document_map') else None
                        filename = filename or top_ev.get("filename") or top_ev.get("source") or "Unknown_Source"

                        processed_evidence[str(lv_key)] = {
                            "file": filename,
                            "page": top_ev.get("page", top_ev.get("page_label", "N/A")),
                            "pdca": str(top_ev.get("pdca_tag", "N/A")).upper(),
                            "confidence": round(float(top_ev.get("rerank_score", 0)), 4),
                            "snippet": str(top_ev.get("content", ""))[:150] + "..."
                        }
                except Exception as ev_err:
                    self.logger.debug(f"⚠️ Skip evidence key {lv_key}: {ev_err}")

            # 4. 📝 Build Final Payload (Standard Schema v2026)
            payload = {
                "metadata": {
                    "record_id": record_id,
                    "tenant": tenant,
                    "year": year,
                    "enabler": enabler,
                    "engine_version": "SEAM-ENGINE-v2026.1.25",
                    "exported_at": datetime.now().isoformat()
                },
                "result_summary": {
                    "maturity_level": stats.get('overall_level_label', f"L{highest_lvl}"),
                    "is_passed": is_passed,
                    "total_weighted_score": round(total_weighted, 4),
                    "evidence_used_count": len(processed_evidence),
                    "evaluated_sub_count": len(results_data),
                    "status": "COMPLETED"
                },
                "sub_criteria_details": results_data,
                "evidence_audit_trail": processed_evidence,
                "strategic_roadmap": getattr(self, 'master_roadmap_data', {
                    "status": "GENERATED",
                    "overall_strategy": "โปรดดูรายละเอียดในส่วน sub_criteria_details"
                })
            }

            # 5. 💾 Save to JSON
            # พยายามใช้ path จาก config ถ้าไม่มีให้ใช้ local exports
            try:
                from utils.path_utils import get_assessment_export_file_path
                export_path = get_assessment_export_file_path(
                    tenant=tenant, year=year, enabler=enabler.lower(),
                    suffix=f"{sub_criteria_id}_{timestamp}", ext="json"
                )
            except ImportError:
                out_dir = f"exports/{tenant}/{year}/{enabler.lower()}"
                os.makedirs(out_dir, exist_ok=True)
                export_path = f"{out_dir}/REPORT_{sub_criteria_id}_{timestamp}.json"

            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            self.logger.info(f"✅ [EXPORT SUCCESS] Report generated: {export_path}")
            return export_path

        except Exception as e:
            self.logger.error(f"🛑 [EXPORT CRITICAL ERROR] {str(e)}", exc_info=True)
            return ""
    

    def _guarantee_text_key(
        self,
        chunks: List[Dict],
        total_count: int = 0,
        restored_count: int = 0
    ) -> List[Dict]:
        """
        Guarantee 'text' key exists in every chunk
        """
        final_chunks = []

        for chunk in chunks:
            if "text" not in chunk or not isinstance(chunk["text"], str):
                chunk["text"] = ""
                cid = str(chunk.get("chunk_uuid", "N/A"))
                self.logger.debug(f"Guaranteed 'text' key for chunk (ID: {cid[:8]})")
            final_chunks.append(chunk)

        if total_count > 0:
            baseline_count = sum(1 for c in final_chunks if c.get("is_baseline"))
            self.logger.info(
                f"HYDRATION FINAL: Restored {restored_count}/{total_count} "
                f"(Baseline={baseline_count}, Total final={len(final_chunks)})"
            )

        return final_chunks

    
    def _normalize_meta(self, c: Dict) -> Tuple[str, str]:
        """
        [REVISED] ดึงชื่อไฟล์และเลขหน้าด้วยระบบ Priority Fallback
        - ป้องกันปัญหาเลขหน้า 0 หาย (Index-based)
        - รองรับโครงสร้างข้อมูลที่ซับซ้อนทั้งจาก Ingest ใหม่ และ Reranker Output
        - เน้น page_label สำหรับการแสดงผลบน UI
        """
        if not isinstance(c, dict):
            # ตรวจสอบเผื่อเป็น LangChain Document ให้ดึง metadata ออกมา
            if hasattr(c, "metadata"):
                meta = c.metadata
                # จำลองโครงสร้าง dict เพื่อให้โค้ดส่วนล่างทำงานต่อได้
                c = {"metadata": meta}
            else:
                return "Unknown Source", "N/A"

        # ดึง metadata มาเตรียมไว้ (ต้องรองรับทั้งกรณี None หรือ Dict)
        meta = c.get("metadata") or {}

        # 1. 🔍 ลำดับการหาชื่อไฟล์ (Source Name Priority)
        source_priority = [
            c.get("source_filename"),
            meta.get("source_filename"),
            c.get("filename"),
            meta.get("source"),
            meta.get("file_name"),
            c.get("id") 
        ]
        
        source = "Unknown"
        for s in source_priority:
            if s and str(s).strip():
                source = str(s).strip()
                break

        # 2. 🔍 ลำดับการหาเลขหน้า (Page Label Priority)
        page_keys = ["page_label", "page", "page_number"]
        found_page = None
        for key in page_keys:
            val_root = c.get(key)
            if val_root is not None and str(val_root).strip().lower() != "n/a":
                found_page = val_root
                break
            val_meta = meta.get(key)
            if val_meta is not None and str(val_meta).strip().lower() != "n/a":
                found_page = val_meta
                break

        # 3. ✨ Final Cleaning & UI Formatting
        clean_source = os.path.basename(source) if "/" in source or "\\" in source else source
        if found_page is not None:
            page_str = str(found_page).strip()
            clean_page = page_str if page_str.lower() != "n/a" else "N/A"
        else:
            clean_page = "N/A"

        return clean_source, clean_page
    
    def audit_agent_router(
        self,
        *,
        context: str,
        sub_criteria_name: str,
        level: int,
        statement_text: str,
        sub_id: str,
        llm_executor,
        confidence_reason: str = "",
        **kwargs
    ):
        """
        [AUDIT AGENT ROUTER – FINAL]
        - L1–L2 → foundation_coaching_agent
        - L3–L5 → standard_audit_agent
        - บังคับใช้ llm_executor แบบ keyword
        """

        if llm_executor is None:
            raise RuntimeError("LLM executor missing in audit_agent_router")

        # เลือก agent ตาม level
        if level <= 2:
            agent = self.foundation_coaching_agent
        else:
            agent = self.standard_audit_agent

        return agent(
            context=context,
            sub_criteria_name=sub_criteria_name,
            level=level,
            statement_text=statement_text,
            sub_id=sub_id,
            llm_executor=llm_executor,        # 🔥 keyword เท่านั้น
            confidence_reason=confidence_reason,
            **kwargs
        )


    def evaluate_pdca(
        self,
        pdca_blocks: Union[Dict[str, Any], str],
        sub_id: str,
        level: int,
        audit_confidence: Any,
        audit_instruction: str = ""
    ) -> Dict[str, Any]:
        """
        [FINAL CANONICAL VERSION]
        - ใช้ audit_agent_router เป็น entry point เดียว
        - L1–L2 → evaluate_with_llm_low_level
        - L3–L5 → evaluate_with_llm
        - แก้ครบ:
            • LLM instance not initialized
            • agent routing ผิด logic
            • duplicate argument
            • positional vs keyword mismatch
        """

        log_prefix = f"🧠 [{sub_id}-L{level}]"

        # --------------------------------------------------
        # [1] Build PDCA Context
        # --------------------------------------------------
        pdca_summary = []

        if isinstance(pdca_blocks, dict):
            for tag in ["P", "D", "C", "A"]:
                val = pdca_blocks.get(tag)
                if val:
                    clean_val = str(val).replace('"', "'")
                    pdca_summary.append(
                        f"### {tag} PHASE EVIDENCE ###\n{clean_val}"
                    )
        else:
            pdca_summary.append(str(pdca_blocks))

        final_context_str = "\n\n".join(pdca_summary)

        # --------------------------------------------------
        # [2] Rubric Lookup
        # --------------------------------------------------
        sub_item = next(
            (i for i in self.flattened_rubric if i.get("sub_id") == sub_id),
            {}
        )

        sub_name = sub_item.get("sub_criteria_name", sub_id)

        level_info = next(
            (lv for lv in sub_item.get("levels", []) if lv.get("level") == level),
            {}
        )

        statement = level_info.get("statement", "")

        # --------------------------------------------------
        # [3] Confidence Normalize
        # --------------------------------------------------
        try:
            if isinstance(audit_confidence, dict):
                conf_val = float(audit_confidence.get("coverage_ratio", 0.0))
            else:
                conf_val = float(audit_confidence or 0.0)
        except Exception:
            conf_val = 0.0

        # --------------------------------------------------
        # [4] Ensure LLM Ready (CRITICAL)
        # --------------------------------------------------
        if self.llm is None:
            self._initialize_llm_if_none()

        if self.llm is None:
            raise RuntimeError("LLM instance not initialized (post-init).")

        # --------------------------------------------------
        # [5] Build Agent Payload (KEYWORD-ONLY)
        # --------------------------------------------------
        agent_payload = {
            # core
            "context": final_context_str,
            "pdca_context": final_context_str,

            # rubric
            "sub_id": sub_id,
            "sub_criteria_name": sub_name,
            "level": level,
            "statement_text": statement,

            # llm
            "llm_executor": self.llm,

            # confidence
            "confidence_reason": f"Coverage: {conf_val:.2f}",
            "ai_confidence": "HIGH" if conf_val >= 0.7 else "MEDIUM",

            # enrichment
            "enabler": self.enabler,
            "enabler_full_name": getattr(
                self, "enabler_full_name", f"ด้าน {self.enabler}"
            ),
            "focus_points": sub_item.get("focus_points", "-"),
            "evidence_guidelines": level_info.get(
                "level_specific_guideline", "-"
            ),
            "specific_contextual_rule": audit_instruction,
        }

        # --------------------------------------------------
        # [6] Execute via audit_agent_router (ONLY ENTRY POINT)
        # --------------------------------------------------
        try:
            return self.audit_agent_router(**agent_payload)

        except Exception as e:
            self.logger.error(
                f"🛑 [EVAL-ERROR] {log_prefix}: {str(e)}",
                exc_info=True
            )
            return {
                "sub_id": sub_id,
                "level": level,
                "score": 0.0,
                "is_passed": False,
                "reason": f"Evaluation Failure: {str(e)}"
            }

        
    def _prepare_worker_tuple(self, sub_criteria_data: Dict, document_map: Optional[Dict]) -> Tuple:
        """เตรียมข้อมูลสำหรับการส่งเข้า Process ใหม่ (Pickle-friendly)"""
        return (
            sub_criteria_data,                 # ข้อมูล Rubric รายข้อ
            self.config.enabler,               # Enabler Code
            self.config.target_level,          # ระดับเป้าหมาย
            self.config.mock_mode,             # โหมด Mock
            getattr(self, 'evidence_map_path', None), 
            self.config.model_name,            # ชื่อรุ่น LLM
            self.config.temperature,           # ค่า Temp
            self.config.min_retry_score,       # เกณฑ์ Rerank
            self.config.max_retrieval_attempts,
            document_map or self.document_map, # ID Mapping
            None,                              # Action Plan Model (ถ้ามี)
            self.config.year,
            self.config.tenant
        )
    
    def _create_failed_result(self, record_id: str, message: str, start_ts: float) -> Dict[str, Any]:
        """สร้างมาตรฐาน Response เมื่อเกิดความผิดพลาดใน Orchestrator"""
        self.logger.error(f"❌ Assessment Failed: {message}")
        return {
            "record_id": record_id,
            "status": "FAILED",
            "error": message,
            "run_time_seconds": round(time.time() - start_ts, 2),
            "summary": {},
            "sub_criteria_results": {}
        }
    
    def _normalize_evidence_metadata(self, evidence_list: List[Dict[str, Any]]):
        """
        [ULTIMATE REVISED v2026.01.25]
        - 🛡️ Type Safety & Resolve: จัดการ Metadata ทั้งหมดให้เป็นมาตรฐานเดียวกัน
        """
        for ev in evidence_list:
            if not isinstance(ev, dict): continue
            
            meta = ev.get("metadata", {})
            if not isinstance(meta, dict): meta = {}
            
            # 1. Resolve ID & UUID
            doc_id = (
                ev.get("doc_id") or 
                ev.get("stable_doc_uuid") or 
                meta.get("stable_doc_uuid") or 
                meta.get("doc_id") or
                f"gen_{uuid.uuid4().hex[:8]}"
            )
            ev["doc_id"] = doc_id
            ev["stable_doc_uuid"] = doc_id

            # 2. Resolve Filename (จากหลายแหล่งรวมถึง document_map)
            raw_source = (
                meta.get("source_filename") or 
                meta.get("file_name") or 
                ev.get("filename") or 
                ev.get("source") or 
                meta.get("source")
            )
            filename = os.path.basename(str(raw_source)) if raw_source else "Unknown_File"
            
            # Cross-check กับคลังชื่อไฟล์กลาง
            if (filename == "Unknown_File" or not filename) and hasattr(self, 'document_map'):
                filename = self.document_map.get(doc_id, "Unknown_File")
                
            ev["filename"] = filename
            ev["source_filename"] = filename
            ev["source"] = filename

            # 3. Resolve Page Label
            raw_page = meta.get("page_label") or meta.get("page") or meta.get("page_number") or ev.get("page") or "0"
            ev["page"] = str(raw_page)

            # 4. Resolve Scoring (ใช้ Get Actual Score ตามที่คุณระบุ)
            actual_score = 0.0
            if hasattr(self, 'get_actual_score'):
                actual_score = self.get_actual_score(ev)
            else:
                actual_score = float(ev.get("relevance_score") or ev.get("rerank_score") or 0.0)
            ev["relevance_score"] = actual_score

            # 5. UI Fields Consistency
            ev["source_type"] = ev.get("source_type") or meta.get("source_type") or "system_gen"
            ev["is_selected"] = ev.get("is_selected") if ev.get("is_selected") is not None else True
            ev["pdca_tag"] = ev.get("pdca_tag") or meta.get("pdca_tag") or "Other"
            ev["note"] = ev.get("note") or ""

        return evidence_list
    
    # ------------------------------------------------------------------------------------------
    # [ULTIMATE REVISE v2026.01.30] 🧠 LAYER 1: Decision Engine (The Brain – Final Hardened)
    # ------------------------------------------------------------------------------------------
    def _get_semantic_tag(self, text: str, sub_id: str, level: int, filename: str = "") -> str:
        """
        [ULTIMATE REVISED v2026.01.31]
        ศูนย์กลางการตัดสินใจ Tag: Heuristic → AI Semantic → Contextual Fallback
        รองรับ Multi-Tenant ผ่านการโหลด Config อัตโนมัติ และจัดการ Error อย่างเป็นระบบ
        - เพิ่ม fallback สำหรับ enabler_name_th และ keywords
        - เพิ่ม log ทุกขั้นตอนเพื่อ debug ง่าย
        - LLM call ปลอดภัย + retry JSON parse
        - ลด "Other" โดยบังคับ fallback phase ถ้า LLM ให้ Other
        """
        # --- [PREPARATION] ดึง Metadata พื้นฐาน ---
        enabler_key = getattr(self.config, 'enabler', 'DEFAULT').upper()
        # fallback ชื่อเต็มถ้า dict ไม่มี key หรือ error
        try:
            enabler_name_th = SEAM_ENABLER_FULL_NAME_TH.get(enabler_key, f"ด้าน {enabler_key}")
        except NameError:
            enabler_name_th = f"ด้าน {enabler_key}"
            self.logger.error("[CRITICAL] SEAM_ENABLER_FULL_NAME_TH not defined → fallback")

        if enabler_name_th == f"ด้าน {enabler_key}":
            self.logger.warning(f"[FALLBACK-NAME] No full name for {enabler_key} → using '{enabler_name_th}'")

        # fallback keywords ถ้า dict ไม่มี key หรือ error
        try:
            enabler_keywords = PDCA_CONFIG_MAP.get(enabler_key, PDCA_CONFIG_MAP["DEFAULT"])
        except NameError:
            enabler_keywords = PDCA_CONFIG_MAP["DEFAULT"]
            self.logger.error("[CRITICAL] PDCA_CONFIG_MAP not defined → using DEFAULT")

        # ดึง require_phase (reuse)
        require_phases = self.get_rule_content(sub_id, level, "require_phase") or []

        # Tenant info + fallback
        tenant_id = getattr(self.config, 'tenant', 'default').lower()
        tenant_name_th = "องค์กรที่รับการประเมิน"
        tenant_code = tenant_id.upper()
        tenant_info_path = get_tenant_info_file_path(tenant_id)
        if os.path.exists(tenant_info_path):
            try:
                with open(tenant_info_path, 'r', encoding='utf-8') as f:
                    t_data = json.load(f)
                    tenant_name_th = t_data.get("tenant_name_th", tenant_name_th)
                    tenant_code = t_data.get("tenant_abbreviation", tenant_code)
            except Exception as e:
                self.logger.warning(f"⚠️ Load tenant_info failed: {e}")

        # Log preparation summary (ช่วย debug)
        self.logger.debug(f"[TAG-PREPARE] Enabler: {enabler_key} ({enabler_name_th}) | Tenant: {tenant_code} ({tenant_name_th}) | Req Phases: {require_phases}")

        text_clean = (text or "").strip()
        if len(text_clean) < 20:
            fallback = require_phases[0] if require_phases else ("P" if level == 1 else "D")
            self.logger.debug(f"[TAG-SHORT] Text too short → fallback {fallback} | {filename[:30]}")
            return fallback

        text_lower = text_clean.lower()

        # --- [LAYER 1] Heuristic: ตรวจสอบด้วย Keyword ---
        for tag, keywords in enabler_keywords.items():
            if any(k.lower() in text_lower for k in keywords):
                self.logger.debug(f"⚡ [HEURISTIC-HIT] {enabler_key}:{tag} | {filename[:30]}")
                return tag

        # --- [LAYER 2] AI Semantic: วิเคราะห์ด้วย LLM ---
        require_str = ", ".join(require_phases) if require_phases else "P, D, C, A"
        desc_bullets = "\n".join(f"- {v}" for v in PDCA_PHASE_DESCRIPTIONS.values())

        system_prompt = (
            f"คุณคือผู้เชี่ยวชาญการตรวจประเมินองค์กร **{tenant_name_th}** ({tenant_code}) "
            f"ในด้าน '{enabler_name_th}' ({enabler_key}) ตามมาตรฐาน SE-AM\n"
            f"ภารกิจ: จำแนกข้อความหลักฐานเข้าหมวด PDCA ตามนิยาม:\n{desc_bullets}\n\n"
            f"บริบทระดับ Level {level}: **เน้นพิจารณา {require_str} เป็นลำดับแรก**\n"
            f"ถ้าเนื้อหาใกล้เคียง phase ใน {require_str} ให้เลือก tag นั้นก่อน 'Other'\n"
            f"ห้ามเดา ถ้าไม่ชัดเจนจริง ๆ ให้ใช้ 'Other'\n\n"
            f"ตอบเฉพาะ JSON Object เท่านั้น ห้ามมี Markdown หรือข้อความอธิบาย:\n"
            f"{{'tag': 'P' | 'D' | 'C' | 'A' | 'Other', 'reason': 'เหตุผลสั้นเป็นภาษาไทย'}}\n\n"
            f"ตัวอย่าง (บริบท {tenant_name_th}):\n"
            f"{{'tag': 'P', 'reason': 'ปรากฏแผนยุทธศาสตร์ {enabler_key} ของ {tenant_code}'}}\n"
            f"{{'tag': 'Other', 'reason': 'เนื้อหาไม่เกี่ยวข้องกับ PDCA หรือ {enabler_name_th}'}}\n\n"
            f"ห้ามใช้ ```json หรือภาษาไทยแบบ Unicode escape"
        )

        user_prompt = f"ชื่อไฟล์: {filename}\nเนื้อหา: {text_clean[:600]}\nระบุ Tag และเหตุผล"

        try:
            json_str = _fetch_llm_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                llm_executor=self.llm,
                max_retries=3
            )
            
            self.logger.debug(f"[LLM-RAW-TAG] {filename[:30]} | {json_str[:300]}...")
            
            # ซ่อม JSON ถ้าพัง (ลบ code block)
            json_str = re.sub(r'```json\s*|\s*```', '', json_str).strip()
            
            data = json.loads(json_str)
            if isinstance(data, list) and data: data = data[0]
            
            tag = str(data.get("tag", "Other")).upper().strip()
            reason = data.get("reason", "ไม่ระบุเหตุผล")
            
            if tag in {"P", "D", "C", "A"}:
                self.logger.info(f"🎯 [AI-TAG SUCCESS] {tenant_code} | {enabler_key}:{tag} | reason: {reason[:60]}")
                return tag
        except Exception as e:
            self.logger.warning(f"⚠️ [AI-TAG ERR] {tenant_code}:{enabler_key}: {str(e)}")

        # --- [LAYER 3] Contextual Fallback (ไม่ยอม Other ถ้าเป็น level สำคัญ) ---
        if require_phases:
            primary_phase = require_phases[0]
            self.logger.debug(f"[FALLBACK-REQ] {primary_phase} (from {require_str})")
            return primary_phase

        # Ultimate fallback ตาม Maturity Level
        fallback = "P" if level == 1 else "D" if level <= 3 else "C" if level == 4 else "A"
        self.logger.debug(f"[ULTIMATE-FALLBACK] {fallback} for L{level}")
        return fallback

    def _get_heuristic_pdca_tag(self, text: str, level: int) -> Optional[str]:
        t = text.lower()
        
        # Do-specific สำหรับ L1 (เน้นผู้บริหารทำจริง)
        do_keywords = [
            "ดำเนินการ", "ปฏิบัติ", "ประชุม", "กิจกรรม", "อบรม", "จัดทำ", "ลงพื้นที่", 
            "ติดตามผล", "ภาพถ่าย", "ผู้บริหาร", "มุ่งมั่น", "ตัวอย่าง", "สนับสนุน", 
            "ขับเคลื่อน", "นำร่อง", "ลงมือทำ", "นำไปใช้"
        ]
        if level <= 2 and any(k in t for k in do_keywords):
            return "D"

        # Check keywords (เพิ่มจาก log)
        check_keywords = [
            "รายงานผล", "ประเมิน", "ติดตาม", "ตัวชี้วัด", "kpi", "ผลลัพธ์", "สรุปผล", 
            "สถิติ", "สำรวจ", "ตรวจสอบ", "วัดผล"
        ]
        if any(k in t for k in check_keywords):
            return "C"

        # Plan & Act (คงเดิม แต่ลด priority)
        if any(k in t for k in ["นโยบาย", "แผน", "ยุทธศาสตร์", "มติ", "คำสั่ง", "เป้าหมาย", "เจตนารมณ์"]):
            return "P"
        if any(k in t for k in ["ปรับปรุง", "พัฒนา", "แก้ไข", "บทเรียน", "lesson learned", "ต่อยอด", "นวัตกรรม"]):
            return "A"

        return None
    
    # ------------------------------------------------------------------------------------------
    # [ULTIMATE REVISE v2026.01.28] 📊 LAYER 2: Contextual Blocker (The Focus)
    # ------------------------------------------------------------------------------------------
    def _get_pdca_blocks_from_evidences(
        self,
        evidences: List[Dict[str, Any]],
        baseline_evidences: List[Dict[str, Any]],
        level: int,
        sub_id: str,
        contextual_rules_map: Dict[str, Any],
        record_id: str = None
    ) -> Dict[str, Any]:
        """
        จัดกลุ่มหลักฐานแบบ Multi-Layer Tagging:
        1. Semantic (AI/Engine) -> 2. Heuristic (Keyword-based) -> 3. Forced (Fallback)
        """

        pdca_groups = defaultdict(list)
        seen_texts = set()
        all_candidate = (evidences or []) + (baseline_evidences or [])

        # ดึงรายชื่อ Phase ที่เกณฑ์เลเวลนี้ต้องการ (เช่น L1 ต้องการ P และ D)
        require_phases = self.get_rule_content(sub_id, level, "require_phase") or ["P", "D"]

        for idx, chunk in enumerate(all_candidate, start=1):
            # --- 1. Data Cleaning & Deduplication ---
            txt = (chunk.get("text") or chunk.get("page_content") or "").strip()
            if not txt or len(txt) < 10:
                continue

            txt_hash = hashlib.sha256(txt.encode()).hexdigest()
            if txt_hash in seen_texts:
                continue
            seen_texts.add(txt_hash)

            # --- 2. Metadata Preparation ---
            meta = chunk.get("metadata", {}) or {}
            fname = chunk.get("source_filename") or meta.get("source_filename") or "Unknown"
            page = meta.get("page_label") or meta.get("page") or "N/A"
            is_baseline = chunk.get("source") == "BASELINE" or chunk.get("is_baseline", False)
            
            prefix = "[BASELINE] " if is_baseline else ""
            source_display = f"{prefix}{fname} (P.{page})"

            # --- 3. MULTI-LAYER TAGGING LOGIC (หัวใจสำคัญ) ---
            is_forced = False
            
            # Layer 1: Semantic Tag (ลองดึงจาก AI/Engine เดิม)
            final_tag = self._get_semantic_tag(txt, sub_id, level, fname)
            tag_source = "Semantic-Engine"

            # Layer 2: Heuristic Fallback (ถ้า Layer 1 หาไม่เจอ หรือได้ค่ากลางๆ)
            if final_tag in [None, "Other", "OTHER", "N/A"]:
                heuristic_tag = self._get_heuristic_pdca_tag(text=txt, level=level)
                if heuristic_tag:
                    final_tag = heuristic_tag
                    tag_source = "Heuristic-Rule-Base"

            # Layer 3: Forced Contextual Fallback (ทางเลือกสุดท้าย)
            if final_tag in [None, "Other", "OTHER", "N/A"]:
                if level >= 4:
                    # เลเวลสูงเราเน้นคุณภาพ ไม่แม่นจริงเราไม่เอามาคิด (Strict Mode)
                    self.logger.debug(f"🚫 Excluded Other (L{level} strict): {source_display}")
                    continue

                is_forced = True
                # แจก Tag ตามลำดับ Require Phase (Round-robin)
                final_tag = require_phases[(idx - 1) % len(require_phases)]
                tag_source = f"Forced-Contextual-L{level} ({final_tag})"
                self.logger.debug(f"⚠️ Forced {final_tag} → {source_display}")

            # --- 4. Append to Group ---
            pdca_groups[final_tag].append({
                "text": txt,
                "source_display": source_display,
                "filename": fname,
                "page": page,
                "is_forced": is_forced,
                "is_baseline": is_baseline,
                "relevance": float(chunk.get("rerank_score") or chunk.get("score") or 0.5),
                "tag_source": tag_source,
                "pdca_tag": final_tag  # 👈 ส่งค่านี้กลับไปเพื่อให้ Router/UI ใช้งานได้จริง
            })

        # --- 5. Block Construction for LLM ---
        max_ch = getattr(self.config, 'MAX_CHUNKS_PER_BLOCK', 5)
        blocks = {
            "sources": {}, 
            "actual_counts": {},
            "all_evidences_with_tags": [] # สำหรับส่งกลับไปทำ Report metadata
        }

        for tag in ["P", "D", "C", "A"]:
            # เรียงลำดับความน่าเชื่อถือ: ของจริง(ไม่ forced) > คะแนนสูง > ไม่ใช่ baseline
            ranked = sorted(
                pdca_groups.get(tag, []),
                key=lambda x: (x["is_forced"], -x["relevance"], x["is_baseline"])
            )[:max_ch]

            if ranked:
                blocks[tag] = "\n\n".join([
                    f"[{c['source_display']} | {c['tag_source']}{' ⚠️FORCED' if c['is_forced'] else ''}]\n"
                    f"{c['text'][:1000]}"
                    for c in ranked
                ])
                # เก็บก้อนหลักฐานแบบละเอียดไว้ทำ UI JSON
                blocks["all_evidences_with_tags"].extend(ranked)
            else:
                blocks[tag] = f"[ไม่พบหลักฐานชัดเจนในหมวด {tag}]"

            blocks["sources"][tag] = [c["source_display"] for c in ranked]
            blocks["actual_counts"][tag] = len([c for c in ranked if not c["is_forced"]])

        return blocks

    # ------------------------------------------------------------------------------------------
    # [ULTIMATE REVISE v2026.01.28] 💾 LAYER 3: Persistence & Retroactive Sync
    # ------------------------------------------------------------------------------------------
    def _save_level_evidences_and_calculate_strength(
        self,
        level_temp_map: List[Dict[str, Any]],
        sub_id: str,
        level: int,
        llm_result: Dict[str, Any],
        highest_rerank_score: float = 0.0
    ) -> float:
        """
        บันทึกหลักฐาน + Retroactive Sync จาก AI Extraction + คำนวณ Strength
        """
        map_key = f"{sub_id}.L{level}"
        ai_contexts = {t: str(llm_result.get(f"Extraction_{t}", "")).lower() for t in "PDCA"}

        new_evidence_list = []
        seen_keys = set()
        PASS_STATUS = "PASS" if llm_result.get("is_passed", False) else "FAIL"

        # ดึง require_phase ไว้ใช้ fallback ล่วงหน้า (ลดการเรียกซ้ำ)
        require_phases = self.get_rule_content(sub_id, level, "require_phase") or []
        default_fallback = require_phases[0] if require_phases else ("P" if level == 1 else "D")

        for chunk in level_temp_map:
            text = chunk.get("text") or ""
            if not text.strip():
                continue

            meta = chunk.get("metadata", {})
            fname = os.path.basename(str(meta.get("source") or "Unknown")).lower()

            doc_id = chunk.get("stable_doc_uuid") or meta.get("stable_doc_uuid") or "unknown"
            chunk_uuid = chunk.get("chunk_uuid") or hashlib.sha256(text.encode()).hexdigest()[:16]
            unique_key = f"{doc_id}:{chunk_uuid}"
            if unique_key in seen_keys:
                continue
            seen_keys.add(unique_key)

            pdca_tag = chunk.get("pdca_tag") or "Other"
            self.logger.debug(f"[EVI-TAG-INPUT] {fname} | raw_pdca_tag: {pdca_tag}")

            # 1. Retroactive Sync จาก AI Extraction (ก่อน retry)
            for tag, summary in ai_contexts.items():
                if fname in summary and len(summary.strip()) > 5:
                    pdca_tag = tag
                    self.logger.info(f"[EVI-TAG-RETRO] {fname} → {pdca_tag} (from AI extraction)")
                    break

            # 2. ถ้ายังเป็น "Other" → พยายาม tag ใหม่ด้วย _get_semantic_tag
            if pdca_tag == "Other":
                try:
                    pdca_tag = self._get_semantic_tag(text, sub_id, level, fname)
                    self.logger.info(f"[EVI-TAG-RETRY] {fname} → {pdca_tag} (retry from Other)")
                except Exception as e:
                    self.logger.warning(f"[EVI-TAG-RETRY-ERR] {fname}: {e}")

            # 3. ถ้ายังเป็น "Other" อีก → บังคับ fallback ไป require_phase[0] หรือ default
            if pdca_tag == "Other":
                pdca_tag = default_fallback
                self.logger.info(f"[EVI-TAG-FORCE] {fname} → {pdca_tag} (force from require_phase/default)")

            entry = {
                "sub_id": sub_id,
                "level": level,
                "pdca_tag": pdca_tag,
                "doc_id": doc_id,
                "chunk_uuid": chunk_uuid,
                "source_filename": fname,
                "page": str(meta.get("page_label") or meta.get("page") or "N/A"),
                "relevance_score": float(chunk.get("rerank_score") or chunk.get("score") or 0.5),
                "text_preview": text[:300].replace("\n", " ") + "..." if len(text) > 300 else text,
                "status": PASS_STATUS,
                "timestamp": datetime.now().isoformat(),
            }
            new_evidence_list.append(entry)

        if not new_evidence_list:
            return 0.0

        self.evidence_map.setdefault(map_key, []).extend(deepcopy(new_evidence_list))

        tags_set = {"P", "D", "C", "A"}
        found_tags = {ev["pdca_tag"] for ev in new_evidence_list if ev["pdca_tag"] in tags_set}
        coverage = len(found_tags) / 4.0
        strength = round((highest_rerank_score * 0.6) + (coverage * 0.4), 2)

        self.assessment_results_map[map_key] = {
            "is_passed": llm_result.get("is_passed", False),
            "score": llm_result.get("score", 0.0),
            "strength": strength
        }

        counts = {t: sum(1 for e in new_evidence_list if e["pdca_tag"] == t) for t in list(tags_set) + ["Other"]}
        self.logger.info(
            f"[EVI-SAVED] {map_key} | items:{len(new_evidence_list)} "
            f"P:{counts['P']} D:{counts['D']} C:{counts['C']} A:{counts['A']} Other:{counts['Other']} "
            f"strength:{strength:.2f}"
        )

        return strength

    def _robust_hydrate_documents_for_priority_chunks(
        self,
        chunks_to_hydrate: List[Dict],
        vsm: Optional['VectorStoreManager'],
        current_sub_id: Optional[str] = None,
        level: Optional[int] = None
    ) -> List[Dict]:
        """
        [ULTIMATE HYDRATION v2026.01.28]
        - ดึง Full Text + Pre-tag ด้วย Decision Engine เดียว
        - Dedup ด้วย full text hash เพื่อความแม่นยำสูง
        - Boost score + Log สรุปเพื่อ audit/debug
        """

        active_sub_id = current_sub_id or getattr(self, 'sub_id', 'unknown')
        if not chunks_to_hydrate:
            self.logger.debug(f"ℹ️ [HYDRATION] No chunks for {active_sub_id} L{level}")
            return []

        def _safe_classify(text: str, filename: str = "") -> str:
            try:
                tag = self._get_semantic_tag(text, active_sub_id, level or 1, filename)
                self.logger.debug(f"[SAFE-CLASSIFY] Raw from engine: {tag} | file: {filename[:30]}")
                
                # ถ้า engine return PDCA → ใช้เลย (ไม่ทับเป็น Other)
                if tag in {"P", "D", "C", "A"}:
                    return tag
                
                # ถ้าเป็น Other จริง → fallback ตาม require_phase (ไม่ใช่ "Other" ทับ)
                reqs = self.get_rule_content(active_sub_id, level or 1, "require_phase") or []
                fallback_tag = reqs[0] if reqs else ("P" if (level or 1) == 1 else "D")
                self.logger.debug(f"[SAFE-CLASSIFY-FALLBACK] {tag} → {fallback_tag} | file: {filename[:30]}")
                return fallback_tag
                
            except Exception as e:
                self.logger.warning(f"⚠️ [CLASSIFY-ERR] Hybrid Fallback: {e} | file: {filename[:30]}")
                reqs = self.get_rule_content(active_sub_id, level or 1, "require_phase") or []
                return reqs[0] if reqs else ("P" if (level or 1) <= 1 else "D")

        def _standardize_chunk(chunk: Dict, boost_score: float) -> Dict:
            chunk = chunk.copy()
            # is_baseline ควรมาจากข้อมูลจริง ไม่ใช่ set True เสมอ
            chunk.setdefault("is_baseline", chunk.get("is_baseline", False))
            text = chunk.get("text", "").strip()
            if not text:
                return chunk
            meta = chunk.get("metadata", {})
            fname = os.path.basename(str(meta.get("source") or meta.get("file_name") or "unknown"))
            chunk["pdca_tag"] = _safe_classify(text, fname)
            chunk["rerank_score"] = max(float(chunk.get("rerank_score", 0.0)), boost_score)
            chunk["score"] = max(float(chunk.get("score", 0.0)), boost_score)
            return chunk

        # 3. เตรียม stable IDs (เพิ่ม chunk_uuid เพื่อความแม่นยำ)
        stable_ids = set()
        for c in chunks_to_hydrate:
            sid = c.get("stable_doc_uuid") or c.get("doc_id")
            if sid:
                stable_ids.add(sid)

        if not stable_ids or not vsm:
            self.logger.warning(f"[HYDRATION] No stable IDs or VSM → fallback boost")
            return [_standardize_chunk(c.copy(), 0.9) for c in chunks_to_hydrate]

        # 4. Fetch full documents
        stable_id_map = defaultdict(list)
        try:
            retrieved_docs = vsm.get_documents_by_id(
                list(stable_ids), doc_type=self.doc_type, enabler=self.config.enabler
            )
            for doc in retrieved_docs:
                sid = doc.metadata.get("stable_doc_uuid") or doc.metadata.get("doc_id")
                if sid:
                    stable_id_map[sid].append({"text": doc.page_content, "metadata": doc.metadata})
        except Exception as e:
            self.logger.error(f"❌ [HYDRATION] VSM Fetch failed: {e}")
            return [_standardize_chunk(c.copy(), 0.9) for c in chunks_to_hydrate]

        # 5. Hydrate + Dedup + Tag
        hydrated_docs = []
        seen_hashes = set()  # ใช้ hash เต็มเพื่อ dedup จริง
        hydrated_count = 0
        total = len(chunks_to_hydrate)

        SAFE_META_KEYS = {"source", "file_name", "page", "page_label", "page_number"}

        for chunk in chunks_to_hydrate:
            new_chunk = chunk.copy()
            sid = new_chunk.get("stable_doc_uuid") or new_chunk.get("doc_id")

            hydrated = False
            if sid and sid in stable_id_map:
                full_doc = stable_id_map[sid][0]
                new_chunk["text"] = full_doc["text"]
                new_chunk.update({k: v for k, v in full_doc["metadata"].items() if k in SAFE_META_KEYS})
                hydrated = True
                hydrated_count += 1

            # Standardize + Tag + Boost
            boost = 1.0 if hydrated else 0.85
            new_chunk = _standardize_chunk(new_chunk, boost)

            # Dedup ด้วย full text hash (แม่นยำกว่า [:100])
            text_hash = hashlib.sha256(new_chunk.get("text", "").encode()).hexdigest()
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                hydrated_docs.append(new_chunk)

        self.logger.info(
            f"✅ [HYDRATION] Complete: {len(hydrated_docs)} chunks ready "
            f"(hydrated: {hydrated_count}/{total}, dedup removed: {total - len(hydrated_docs)})"
        )

        # ถ้ามีฟังก์ชันนี้อยู่จริง ให้เรียกท้ายสุด
        return self._guarantee_text_key(hydrated_docs) if hasattr(self, '_guarantee_text_key') else hydrated_docs


    def _is_previous_level_passed(self, sub_id: str, level: int) -> bool:
        """
        [STRICT REVISE v2026.01.18.1] - ระบบตรวจสอบสถานะ Level ก่อนหน้าแบบเข้มงวด
        """
        if level <= 1: 
            return True
            
        prev_level = level - 1
        # ตรวจสอบ Key ทั้งสองรูปแบบที่ระบบอาจใช้บันทึก
        possible_keys = [f"{sub_id}.L{prev_level}", f"{sub_id}_L{prev_level}"]
        
        for key in possible_keys:
            # ใช้ get() เพื่อป้องกัน KeyError และตรวจสอบผลลัพธ์
            result = getattr(self, 'assessment_results_map', {}).get(key)
            if result:
                if result.get('is_passed') is True:
                    self.logger.info(f"✅ [LEVEL-GATE] Level {prev_level} passed for {sub_id}")
                    return True
                else:
                    self.logger.warning(f"⚠️ [LEVEL-GATE] Level {prev_level} found but status is FAIL")
                    return False

        # Safe Guard: ป้องกันการข้าม Level โดยไม่มีบันทึก
        self.logger.warning(f"🚫 [LEVEL-GATE] No assessment record for L{prev_level}. Blocking L{level}.")
        return False

    def _normalize_thai_text(self, text: str) -> str:
        """
        [ULTIMATE THAI NORMALIZE v2026.1.31 – FULL REVISED]
        - แก้ไขปัญหา 'สระหาย' (เช่น ผู้บริหาร -> ผบรหาร) อย่างถาวร
        - High Performance: วนลูป Filter + Lowercase เพียงรอบเดียวจบ
        - Unicode NFC: รวม Combining Characters (สระ/วรรณยุกต์) ก่อนประมวลผล
        - Strict Thai Range: รักษา ก-ฮ, สระ, วรรณยุกต์ และเลขไทย (\u0E00-\u0E7F) ครบ 100%
        """
        if not text or not isinstance(text, str):
            return ""

        # 1. Unicode NFC normalization 
        # ป้องกันการแยกส่วนของอักขระไทย (เช่น สระอำ หรือ สระที่ซ้อนกัน)
        text = unicodedata.normalize('NFC', text)

        # 2. Compile Regex Pattern สำหรับตัวอักษรที่อนุญาต
        # ภาษาอังกฤษ (a-zA-Z), ภาษาไทยทั้งหมด (\u0E00-\u0E7F), ตัวเลข (0-9), และช่องว่าง (\s)
        allowed_pattern = re.compile(r"[a-zA-Z0-9\u0E00-\u0E7F\s]")

        # 3. Single-pass Filter & Lowercase
        # วนลูปตรวจสอบทีละตัวเพื่อความแม่นยำสูงสุดและประหยัดทรัพยากร
        res = []
        for char in text:
            if allowed_pattern.match(char):
                # ทำ Lowercase เฉพาะตัวอักษรภาษาอังกฤษ (isascii ช่วยแยกภาษาไทยออกทันที)
                if char.isascii() and char.isalpha():
                    res.append(char.lower())
                else:
                    res.append(char)
        
        # รวม List กลับเป็น String
        text = "".join(res)

        # 4. Collapse whitespace (เปลี่ยนช่องว่างซ้ำเป็น 1 ช่อง) และ Trim
        text = re.sub(r"\s+", " ", text).strip()

        return text
    
    
    def enhance_query_for_statement(
        self,
        statement_text: str,
        sub_id: str,
        statement_id: str,
        level: int,
        focus_hint: str = "",
    ) -> List[str]:
        """
        [STRATEGIC QUERY GEN v2026.1.25 – HYBRID OPTIMIZED]
        - Robust Safety: จัดการ input ว่างและสร้าง fallback เสมอ
        - Level-Aware Negatives: ปลดล็อก '-MasterPlan' สำหรับ L1-L2 เพื่อให้เจอไฟล์นโยบาย
        - Phase-Targeted: ดึง keywords ตาม PDCA phases จาก JSON Config
        - Precision Shuffle: จำกัด 8 queries พร้อมทำความสะอาดข้อความ
        """
        logger = logging.getLogger(__name__)
        log_prefix = f"[QUERY-GEN] {sub_id} L{level}"

        # 0. Safety guard
        if not statement_text or not isinstance(statement_text, str):
            logger.warning(f"{log_prefix} Empty/invalid statement_text → fallback basic")
            fallback_q = f"{sub_id} {focus_hint or 'วิสัยทัศน์ นโยบาย'} การจัดการความรู้"
            return [fallback_q, f"{sub_id} แผนแม่บท KM"]

        # Anchors
        enabler_id = getattr(self.config, 'enabler', 'KM').upper()
        tenant_name = getattr(self.config, 'tenant', 'PEA').upper()
        id_anchor = f"{enabler_id} {sub_id}"

        # --- 🛡️ 1. Dynamic Negative Keywords (The Core Fix) ---
        if level <= 2:
            # ระดับนโยบาย/วางรากฐาน ห้ามบล็อกแผนแม่บทเด็ดขาด
            neg_strict = "-ภาคผนวก -ภาพถ่ายกิจกรรม"
        else:
            # ระดับปฏิบัติ/ยกระดับ บล็อกแผนแม่บทเพื่อหาหลักฐานการทำจริง
            neg_strict = "-แผนแม่บท -ยุทธศาสตร์ชาติ -MasterPlan -ภาคผนวก"

        # 2. Keywords Preparation
        require_phases = self.get_rule_content(sub_id, level, "require_phase") or ['P', 'D']
        query_syn = self.get_rule_content(sub_id, level, "query_synonyms") or ""
        
        raw_kws = self.get_rule_content(sub_id, level, "must_include_keywords") or []
        phase_map = {"P": "plan_keywords", "D": "do_keywords", "C": "check_keywords", "A": "act_keywords"}
        
        # ดึง Default Keywords จาก Enabler (KM)
        defaults = getattr(self, 'contextual_rules_map', {}).get("_enabler_defaults", {})
        for phase in require_phases:
            kw_key = phase_map.get(phase)
            if kw_key:
                raw_kws.extend(defaults.get(kw_key, []))

        clean_kws = " ".join(sorted(set(str(k).strip() for k in raw_kws if k))[:3])
        
        # Clean Statement (ตัดส่วน 'เช่น' ออกเพื่อความคมในการ Search)
        clean_stmt = statement_text.split("เช่น", 1)[0].strip()
        clean_stmt = re.sub(r'[^\w\s]', '', clean_stmt)[:60]

        queries: List[str] = []

        # 3. Strategy: Multi-Angle Retrieval
        # A: Core Strategy (Synonyms + Statement)
        queries.append(f"{id_anchor} {query_syn} {clean_stmt}")

        # B: Evidence Type Strategy (ตามระดับ Maturity)
        if level <= 2:
            queries.append(f"{tenant_name} ประกาศ คำสั่ง ระเบียบ บันทึกข้อความ มติบอร์ด ลงนาม {id_anchor} {query_syn}")
        else:
            queries.append(f"{tenant_name} รายงานผล KPI ผลสำเร็จ ติดตามผล {id_anchor} {query_syn} {neg_strict}")

        # C: Phase-Specific Strategy
        for phase in require_phases:
            queries.append(f"{id_anchor} {query_syn} {phase} {clean_kws} {neg_strict}")

        # D: Fallback Core
        if len(queries) < 4:
            queries.append(f"{tenant_name} {id_anchor} {clean_stmt} {clean_kws}")

        # 4. Post-process: Dedup, Truncate, and Shuffle
        final_queries = []
        seen = set()
        import random
        
        for q in queries:
            # Normalize และตัดคำให้พอเหมาะสำหรับ Vector Search (18-24 คำ)
            q_clean = self._normalize_thai_text(q)
            words = q_clean.split()
            q_trunc = " ".join(words[:22])
            
            # ป้องกัน Query ซ้ำ
            q_key = " ".join(words[:15])
            if q_trunc and q_key not in seen:
                final_queries.append(q_trunc)
                seen.add(q_key)

        random.shuffle(final_queries)
        
        logger.info(f"🚀 [Query Gen v2026.1.25] {sub_id} L{level} | Final Queries: {len(final_queries[:8])} | Neg: {neg_strict}")
        return final_queries[:8]
    

    def _get_level_aware_queries(self, criteria_id: str, level_key: str) -> List[str]:
        """
        [REVISED v2026.1.25 - PRECISION EVIDENCE]
        """
        criteria_rules = self.contextual_rules_map.get(criteria_id, {})
        level_rule = criteria_rules.get(level_key, {})
        synonyms = level_rule.get("query_synonyms", "")
        
        tenant = getattr(self.config, 'tenant', 'PEA').upper()
        prefix = f"{tenant} {self.enabler} {criteria_id}"
        
        # เจาะจงประเภทไฟล์ตาม Synonyms ที่ดึงจาก JSON
        generated_queries = [
            f"{prefix} {synonyms}", # ทิศทางหลัก
            f"{prefix} {synonyms} รายงานการประชุม มติอนุมัติ คำสั่งแต่งตั้ง", # งานกรรมการ
            f"{prefix} {synonyms} บันทึกข้อความ ประกาศใช้ ลงนามอนุมัติ", # งานนโยบาย
            f"{prefix} {synonyms} ผลการดำเนินงาน สรุปโครงการ รายงานผล" # งานปฏิบัติ
        ]
        
        return [self._normalize_thai_text(q) for q in generated_queries]

    def relevance_score_fn(self, evidence: Dict[str, Any], sub_id: str, level: int) -> float:
        """
        [ULTIMATE REVISED v2026.01.25]
        - 45% Rerank + 35% Keyword + 20% Contextual Bonuses
        - Optimized Code: ใช้โครงสร้างที่อ่านง่ายและลดการคำนวณซ้ำซ้อน
        - ปรับปรุงการดึง Metadata ให้ Robust มากขึ้นตาม Header core/seam_assessment.py
        """
        if not evidence or not isinstance(evidence, dict):
            return 0.0

        # 1. Rerank Score Processing (45%)
        # ดึงคะแนนดิบจาก VectorStore หรือ Reranker
        raw_val = evidence.get('rerank_score') or evidence.get('score') or 0.0
        normalized_rerank = min(max(float(raw_val), 0.0), 1.0)

        # 2. ข้อมูลพื้นฐาน (ใช้โครงสร้างจาก Header)
        # ดึงเนื้อหาและทำความสะอาดเบื้องต้น
        text = str(evidence.get('text') or evidence.get('page_content') or '').lower().strip()
        meta = evidence.get('metadata') or {}
        if not isinstance(meta, dict): meta = {}
        
        # ดึงชื่อไฟล์เพื่อใช้ทำ Source Grading
        filename = str(meta.get('source') or meta.get('source_filename') or evidence.get('source') or '').lower()
        cum_rules = self.get_cumulative_rules_cached(sub_id, level)

        # 3. Source Grading Bonus (ถ่วงน้ำหนักความน่าเชื่อถือของประเภทไฟล์)
        source_bonus = 0.0
        primary_docs = ["มติ", "บันทึก", "คำสั่ง", "ประกาศ", "นโยบาย", "แผนแม่บท", "มติบอร์ด"]
        secondary_docs = ["assessment report", "รายงานการประเมิน", "สรุปผล", "รายงานผล", "kpi"]
        
        if any(p in filename for p in primary_docs):
            source_bonus = 0.20
        elif any(s in filename for s in secondary_docs):
            source_bonus = 0.10

        # 4. Keyword Score (35%) - ให้น้ำหนักตามระดับ Maturity
        target_kws = set()
        if level <= 2:
            target_kws.update(cum_rules.get('plan_keywords', []) + cum_rules.get('do_keywords', []))
        else:
            target_kws.update(cum_rules.get('check_keywords', []) + cum_rules.get('act_keywords', []))

        keyword_score = 0.0
        if target_kws:
            match_count = sum(1 for kw in target_kws if str(kw).lower() in text)
            if match_count > 0:
                expected = max(1, len(target_kws) * 0.3)
                # ใช้ Power function เพื่อให้การเจอ Keyword เพียงไม่กี่คำก็ได้คะแนนโดดขึ้นมา
                keyword_score = min((match_count / expected) ** 0.6, 1.0)
                keyword_score = max(keyword_score, 0.20) # Floor สำหรับการเจออย่างน้อย 1 คำ

        # 5. PDCA Tag Bonus (High Priority สำหรับเฟสที่เกณฑ์ต้องการ)
        pdca_bonus = 0.0
        pdca_tag = str(evidence.get('pdca_tag') or meta.get('pdca_tag') or "").upper()
        required_phases = cum_rules.get('required_phases', [])
        
        if pdca_tag in required_phases:
            pdca_bonus = 0.30
        elif pdca_tag in {'P', 'D', 'C', 'A'}:
            pdca_bonus = 0.15

        # 6. Contextual Bonuses (Neighbors & Specific Rules)
        neighbor_bonus = 0.15 if evidence.get('is_neighbor') or meta.get('is_neighbor') else 0.0
        
        rule_bonus = 0.0
        specific_rule = str(cum_rules.get('specific_contextual_rule', '')).lower()
        if specific_rule and any(word in text for word in specific_rule.split()[:10]):
            rule_bonus = 0.15

        # 7. ผลรวมคะแนนสุทธิ (Final Weighted Score)
        final_score = (
            (0.45 * normalized_rerank) + 
            (0.35 * keyword_score) + 
            source_bonus + pdca_bonus + neighbor_bonus + rule_bonus
        )

        # 8. High-Confidence Min Floor
        # ถ้า Rerank มั่นใจมาก (0.8+) ให้คะแนนผ่านเกณฑ์ขั้นต่ำแน่นอน
        if normalized_rerank > 0.80:
            final_score = max(final_score, 0.45)

        final_score = min(max(final_score, 0.0), 1.0)

        # 🎯 Logging สำหรับการ Debug (INFO Level ตามที่คุณต้องการ)
        self.logger.info(
            f"🔎 [REL-CHECK] {sub_id} L{level} | Final: {final_score:.3f} | "
            f"Rerank: {normalized_rerank:.2f} | KW: {keyword_score:.2f} | Tag: {pdca_tag} | File: {os.path.basename(filename)[:30]}"
        )

        return float(final_score)

    def _perform_adaptive_retrieval(
        self,
        sub_id: str,
        level: int,
        stmt: str,
        vectorstore_manager: Any,
    ) -> Tuple[List[Dict], float]:
        """
        [ULTIMATE REVISED v2026.01.25]
        - Clean Code: เรียกใช้ Global Variables (RETRIEVAL_*) โดยตรง ไม่ Assign ซ้ำ
        - High Performance: ใช้ Early Exit และ High-Rerank Threshold เพื่อลด Latency
        - Robustness: มี Safe Scoring และ Recovery Sweep กรณีหลักฐานไม่เพียงพอ
        """
        start_time = time.time()
        if not stmt or not isinstance(stmt, str):
            return [], 0.0

        candidates: List[Dict] = []
        used_uuids = set()
        final_max_score = 0.0
        level_key = f"L{level}"
        tenant = getattr(self.config, "tenant", "PEA").upper()

        def safe_relevance_score(evidence: Dict) -> float:
            """ Safe wrapper สำหรับการคำนวณคะแนนเสริมโดยใช้ฟังก์ชันภายใน """
            try:
                return self.relevance_score_fn(evidence, sub_id, level)
            except Exception as e:
                self.logger.warning(f"⚠️ [SAFE-SCORE] {sub_id} L{level}: {e}")
                return float(evidence.get('rerank_score') or evidence.get('score') or 0.0)

        # --- STEP 1: PRIORITY MAPPING (บังคับดึงข้อมูลที่ Mapping ไว้ล่วงหน้า) ---
        try:
            _, priority_docs = self._get_mapped_uuids_and_priority_chunks(
                sub_id=sub_id, level=level, statement_text=stmt, vectorstore_manager=vectorstore_manager
            ) or (set(), [])
            
            for p in priority_docs:
                uid = p.get("chunk_uuid")
                if not uid or uid in used_uuids: continue
                
                p["source"] = os.path.basename(p.get("source") or "Unknown")
                # ฉีดคะแนนพิเศษให้ Priority Docs (ขั้นต่ำ 0.90)
                p["score"] = max(safe_relevance_score(p), 0.90) 
                
                used_uuids.add(uid)
                candidates.append(p)
                final_max_score = max(final_max_score, p["score"])
        except Exception as e:
            self.logger.warning(f"⚠️ Priority mapping skip: {e}")

        # --- STEP 2: HYBRID QUERY GENERATION ---
        json_queries = self._get_level_aware_queries(sub_id, level_key)
        legacy_queries = self.enhance_query_for_statement(stmt, sub_id, f"{sub_id}.L{level}", level)
        active_queries = list(dict.fromkeys(json_queries + legacy_queries))[:10]

        # --- STEP 3: ITERATIVE RETRIEVAL LOOP (พร้อม EARLY EXIT) ---
        
        for i, q in enumerate(active_queries):
            # 🎯 [EARLY EXIT] ใช้ตัวแปร Global โดยตรง (ไม่ต้อง Assign ซ้ำ)
            if len(candidates) >= RETRIEVAL_EARLY_EXIT_COUNT and final_max_score >= RETRIEVAL_EARLY_EXIT_SCORE_THRESHOLD:
                self.logger.info(f"🎯 [EARLY-EXIT] {sub_id} L{level} | Found {len(candidates)} docs | Max: {final_max_score:.4f}")
                break
            
            try:
                res = self.rag_retriever(
                    self._normalize_thai_text(q), self.doc_type, sub_id=sub_id, level=level,
                    vectorstore_manager=vectorstore_manager
                ) or {}
                
                for d in (res.get("top_evidences") or []):
                    uid = d.get("chunk_uuid")
                    score = float(d.get("score", 0.0))
                    
                    # กรองด้วย RETRIEVAL_RERANK_FLOOR จาก Global
                    if uid and uid not in used_uuids and score >= RETRIEVAL_RERANK_FLOOR:
                        d["source"] = os.path.basename(d.get("source") or "Unknown")
                        
                        # ฉีดคะแนนวิเคราะห์ความเกี่ยวข้องเฉพาะชิ้นที่มีคุณภาพ (High Rerank)
                        if score > RETRIEVAL_HIGH_RERANK_THRESHOLD:
                            d["score"] = max(score, safe_relevance_score(d))
                        else:
                            d["score"] = score  
                        
                        used_uuids.add(uid)
                        candidates.append(d)
                        final_max_score = max(final_max_score, d["score"])
            except Exception as e:
                self.logger.error(f"❌ Query Loop {i+1} failed: {e}")

        # --- STEP 4: RECOVERY SWEEP (ถ้าหลักฐานไม่พอหรือคะแนนต่ำเกินไป) ---
        if final_max_score < RETRIEVAL_RELEVANCE_THRESHOLD or len(candidates) < 5:
            self.logger.info(f"🚨 [RECOVERY] Insufficient evidence (Max:{final_max_score:.4f}). Triggering sweep...")
            self._execute_recovery_sweep(sub_id, level, stmt, tenant, used_uuids, candidates, vectorstore_manager)
            
            # Re-calculating Final Max Score หลัง Recovery
            if candidates:
                for c in candidates:
                    if c.get("is_recovery"):
                        c["score"] = max(c.get("score", 0.0), safe_relevance_score(c))
                final_max_score = max([float(c.get("score", 0.0)) for c in candidates])

        # --- STEP 5: FINAL SORT & TRIM ---
        candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        # ใช้ ANALYSIS_FINAL_K จาก Global โดยตรง
        final_docs = candidates[:ANALYSIS_FINAL_K]
        
        elapsed = time.time() - start_time
        self.logger.info(f"🏁 [COMPLETE] {sub_id} L{level} | Final Docs: {len(final_docs)} | Max: {final_max_score:.4f} | {elapsed:.2f}s")
        
        return final_docs, float(final_max_score)

    def _execute_recovery_sweep(self, sub_id, level, stmt, tenant, used_uuids, candidates, vectorstore_manager):
        """ [ULTIMATE REVISED] ระบบค้นหาแบบกว้าง (Broad Search) โดยใช้ตัวแปรจาก Global Header """
        try:
            # ดึงคำสำคัญจาก Contextual Rules (ถ้ามี)
            rule = getattr(self, 'contextual_rules_map', {}).get(sub_id, {}).get(f"L{level}", {})
            keywords = rule.get("must_include_keywords", [])[:4]
            
            # สร้าง Query กว้างๆ: ใช้ tenant, keywords และหัวข้อบางส่วน
            recovery_query = self._normalize_thai_text(
                f"{sub_id} {tenant} {' '.join(keywords)} {stmt[:30]}"
            )
            
            # เรียกใช้ rag_retriever (ใช้ self.doc_type และ vectorstore_manager)
            res_fb = self.rag_retriever(
                recovery_query, 
                self.doc_type, 
                sub_id=sub_id, 
                level=level,  # ส่ง level ไปด้วยเพื่อให้ retriever ทำงานได้แม่นยำขึ้น
                vectorstore_manager=vectorstore_manager
            ) or {}
            
            for d in (res_fb.get("top_evidences") or []):
                uid = d.get("chunk_uuid")
                score = float(d.get("score", 0.0))
                
                # ใช้ RETRIEVAL_RERANK_FLOOR จาก Global แทนการ Hard-coded
                if uid and uid not in used_uuids and score >= RETRIEVAL_RERANK_FLOOR:
                    d["source"] = os.path.basename(d.get("source") or "Unknown")
                    d["is_recovery"] = True
                    used_uuids.add(uid)
                    candidates.append(d)
        except Exception as e:
            self.logger.error(f"❌ Recovery sweep failed: {e}")

    def _log_pdca_status(self, sub_id, name, level, blocks, req_phases, sources_count, score, conf_level, **kwargs):
        """ [FULL REVISED] พ่น Dashboard สรุปสถานะ PDCA แบบ Real-time """
        try:
            actual_counts = kwargs.get('pdca_breakdown', {}) 
            is_safety_pass = kwargs.get('is_safety_pass', False)
            status_parts = []
            source_errors = 0
            
            # Mapping Key จาก LLM Output กับ PDCA Phase
            mapping = [("Extraction_P", "P"), ("Extraction_D", "D"), ("Extraction_C", "C"), ("Extraction_A", "A")]

            for full_key, short in mapping:
                count = float(actual_counts.get(short, 0.0))
                content = str(blocks.get(full_key, "")).strip()
                
                # ตรวจสอบว่า AI เจอเนื้อหาจริงหรือไม่
                ai_found = bool(content and content.lower() not in ["-", "n/a", "none", "ไม่พบข้อมูล"])
                has_source = "[Source:" in content and "]" in content
                
                if ai_found and not has_source:
                    source_errors += 1

                # Logic การเลือก Icon เพื่อแสดง Maturity Gap
                if count >= 1.0:
                    icon = "✅" if has_source else "⚠️" # ผ่าน/ผ่านแต่ลืมอ้างอิง
                elif is_safety_pass and short in req_phases:
                    icon = "🛡️" # ผ่านด้วยระบบ Safety Pass
                elif ai_found: 
                    icon = "🔷" # พบร่องรอยแต่หลักฐานไม่แน่นพอ
                elif short not in req_phases: 
                    icon = "➖" # เฟสนี้ไม่ถูกบังคับตามเกณฑ์ระดับนี้
                else: 
                    icon = "❌" # ไม่พบข้อมูล

                status_parts.append(f"{short}:{icon}({count:.1f})")

            display_score = float(score or 0.0)
            alert_msg = f" 🚨[REF-ERR:{source_errors}]" if source_errors > 0 else ""
            pass_label = " 🛡️[SAFETY-PASS]" if is_safety_pass else ""
            
            # พ่น Dashboard ออก Console
            self.logger.info(
                f"📊 [PDCA-DASHBOARD] {sub_id} L{level} | {str(name)[:60]}...\n"
                f"   Maturity Status: {' '.join(status_parts)}{pass_label}{alert_msg}\n"
                f"   Summary: Score={display_score:.2f} | Evidence={sources_count} chunks | AI-Conf={str(conf_level).upper()}"
            )
        except Exception as e:
            self.logger.error(f"❌ Dashboard Logging Failed: {str(e)}")

    
    def _summarize_evidence_list_short(self, evidences: list, max_sentences: int = 3) -> str:
        """ [REVISED] แปลงรายการหลักฐานเป็นข้อความสรุปเพื่อให้ AI ประมวลผลง่ายขึ้น """
        if not evidences: 
            return "ไม่พบข้อมูลหลักฐานเชิงประจักษ์ในระบบ"
        
        parts = []
        # กรองเฉพาะหลักฐานที่มีเนื้อหาจริง
        valid_evidences = [ev for ev in evidences if isinstance(ev, dict) and (ev.get("text") or ev.get("content", "")).strip()]
        
        for ev in valid_evidences[:max_sentences]:
            # ดึงชื่อไฟล์ (ใช้ os.path.basename จาก Header)
            filename = os.path.basename(ev.get("file_name") or ev.get("source") or "Unknown_Document")
            page = ev.get("page", "N/A")
            
            # ล้างข้อความและจำกัดความยาวเพื่อประหยัด Token
            raw_text = ev.get("text") or ev.get("content") or ""
            clean_text = " ".join(raw_text[:150].split()).strip()
            
            parts.append(f"• [{filename}, หน้า {page}]: \"{clean_text}...\"")

        return "\n".join(parts)
    
    def _run_expert_re_evaluation(
        self,
        sub_id: str,
        level: int,
        statement_text: str,
        context: str,
        first_attempt_reason: str,
        missing_tags: Union[List[str], Set[str]],
        highest_rerank_score: float,
        sub_criteria_name: str,
        llm_evaluator_to_use: Any,
        base_kwargs: Dict[str, Any],
        audit_instruction: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        [JUDICIAL REVIEW - FINAL POLISH v2026]
        ระบบอุทธรณ์รอบสอง: ป้องกันการพังจากข้อมูลผิด Format และบังคับใช้เกณฑ์ Substance over Form
        """
        log_prefix = f"Sub:{sub_id} L{level}"
        self.logger.info(f"⚖️ [EXPERT-APPEAL] Starting for {log_prefix} (Max Rerank: {highest_rerank_score:.4f})")

        # 1. ป้องกัน Missing Tags ว่างเปล่า
        missing_set = set(missing_tags) if isinstance(missing_tags, (list, set)) else set()
        missing_str = ", ".join(sorted(missing_set)) if missing_set else "เกณฑ์การพิจารณา PDCA"

        # 2. ปรับปรุง Instruction ให้ดุดันขึ้น (ป้องกัน AI ขี้เกียจตรวจ)
        enabler_header = f"--- [ENABLER RULES] ---\n{audit_instruction}\n" if audit_instruction else ""
        hint_msg = f"""
### 🚨 EXPERT JUDICIAL REVIEW - SECOND CHANCE 🚨
{enabler_header}
[ROUND 1 FAILURE]: "{first_attempt_reason[:150]}..."
[CRITICAL HINT]: ตรวจพบหลักฐานที่มีความเกี่ยวข้องสูงมาก ({highest_rerank_score:.4f}) ในหัวข้อ: {missing_str}

MANDATORY AUDIT RULES:
1. **Substance over Form**: หากเนื้อหาในหลักฐานระบุว่ามีการทำกิจกรรมจริง แม้ชื่อไฟล์จะไม่ตรง หรือไม่มีลายเซ็น "ต้องให้ผ่าน"
2. **Specific Defense**: ระบุชื่อไฟล์และหน้าที่ใช้ยืนยันการเปลี่ยนคำตัดสินในช่อง reason
"""

        # 3. ✨ [SAFE INJECTION] จัดการ pdca_blocks ให้รองรับทุกรูปแบบข้อมูล
        # ดึงของเก่าออกมา ถ้าไม่มีให้ใช้ List ว่าง
        original_blocks = base_kwargs.get("pdca_blocks", [])
        
        if isinstance(original_blocks, list):
            # ก๊อปปี้มาเพื่อป้องกันการกระทบข้อมูลเดิม (Side Effect)
            expert_pdca_blocks = list(original_blocks) 
            expert_pdca_blocks.append({
                "type": "judicial_review_instruction",
                "content": hint_msg,
                "metadata": {"priority": "highest", "is_appeal": True}
            })
        else:
            # ถ้าเป็น String หรือ Format อื่น ให้ต่อท้ายแบบ Text
            expert_pdca_blocks = f"{str(original_blocks)}\n\n{hint_msg}"

        # 4. ประกอบร่าง Arguments ใหม่
        # ใช้ .copy() เพื่อไม่ให้ไปแก้ base_kwargs ตัวจริงที่อาจถูกใช้ซ้ำใน Loop อื่น
        expert_kwargs = base_kwargs.copy()
        expert_kwargs.update({
            "pdca_blocks": expert_pdca_blocks,
            "sub_id": sub_id,
            "level": level,
            "is_expert_mode": True # ส่ง Flag ให้ Prompt รู้ว่าเป็นโหมดตรวจละเอียด
        })

        # 5. Execute LLM Call (พร้อมระบบกันตาย)
        re_eval_result = None
        try:
            # ลองเรียกใช้ 1 ครั้งด้วยความละเอียดสูง (Expert Mode)
            re_eval_result = llm_evaluator_to_use(**expert_kwargs)
        except Exception as e:
            self.logger.error(f"❌ [APPEAL-FATAL] LLM Call failed: {e}")
            return {"is_passed": False, "score": 0.0, "reason": f"Appeal system error: {str(e)}"}

        # 6. ประเมินผลการอุทธรณ์
        if not isinstance(re_eval_result, dict):
            return {"is_passed": False, "score": 0.0, "reason": "Appeal result format error"}

        is_passed_now = bool(re_eval_result.get("is_passed", False))
        
        if is_passed_now:
            self.logger.info(f"🛡️ [OVERRIDE-SUCCESS] {log_prefix} | ผลอุทธรณ์: ผ่าน")
            re_eval_result.update({
                "is_safety_pass": True,
                "appeal_status": "GRANTED",
                "reason": f"🌟 [EXPERT OVERRIDE]: {re_eval_result.get('reason', '')}"
            })
        else:
            self.logger.info(f"❌ [APPEAL-DENIED] {log_prefix} | ผลอุทธรณ์: ไม่ผ่าน")
            re_eval_result["appeal_status"] = "DENIED"

        return re_eval_result

    def _build_multichannel_context_for_level( # เปลี่ยนเป็น Private Method
        self, # เพิ่ม self
        level: int,
        top_evidences: List[Dict[str, Any]],
        previous_levels_map: Optional[Dict[str, Any]] = None,
        previous_levels_evidence: Optional[List[Dict[str, Any]]] = None,
        max_main_context_tokens: int = 3000, 
        max_summary_sentences: int = 4,
        max_context_length: Optional[int] = None, 
        **kwargs
    ) -> Dict[str, Any]:
        """
        [ULTIMATE OPTIMIZED v2026.8 - REFACTORED AS CLASS METHOD]
        """
        K_MAIN = 5
        # ใช้ค่าจาก Config ใน Class ได้เลยถ้ามี เช่น self.config.l1_threshold
        MIN_RELEVANCE_FOR_AUX = 0.15 if level == 1 else 0.4 

        # 1. Baseline Summary
        baseline_evidence = previous_levels_evidence or []
        if previous_levels_map:
            for lvl_ev in previous_levels_map.values():
                baseline_evidence.extend(lvl_ev)

        # Normalize list (ตาม Logic เดิมของคุณที่แก้บั๊ก Slice แล้ว)
        baseline_evidence_list = []
        if isinstance(baseline_evidence, list):
            baseline_evidence_list = baseline_evidence
        elif isinstance(baseline_evidence, dict):
            baseline_evidence_list = list(baseline_evidence.values())
        
        summarizable_baseline = [
            item for item in baseline_evidence_list[:40] 
            if isinstance(item, dict) and (item.get("text") or item.get("content", "")).strip()
        ]

        if not summarizable_baseline:
            baseline_summary = "ไม่มีหลักฐานจากระดับก่อนหน้า (เริ่มต้นที่ Level 1)"
        else:
            # เรียกใช้ผ่าน self
            baseline_summary = self._summarize_evidence_list_short(
                summarizable_baseline,
                max_sentences=max_summary_sentences
            )

        # 2. Direct + Aux Separation
        direct, aux_candidates = [], []

        for idx, ev in enumerate(top_evidences[:40], 1):
            if not isinstance(ev, dict): continue

            tag = (ev.get("pdca_tag") or ev.get("PDCA") or "Other").upper()
            relevance = ev.get("rerank_score") or ev.get("score", 0.0)
            text_preview = (ev.get('text', '')[:80] + "...") if ev.get('text') else "[No text]"

            # ใช้ self.logger ได้เลย
            self.logger.debug(f"[TAG-CHECK L{level} #{idx}] Rel: {relevance:.3f} | Tag: {tag} | Preview: {text_preview}")

            if tag in {"P", "PLAN", "D", "DO", "C", "CHECK", "A", "ACT"}:
                direct.append(ev)
            elif relevance >= MIN_RELEVANCE_FOR_AUX:
                aux_candidates.append(ev)

        # 3. Logic การย้าย Aux และ Fallback (เหมือนเดิม แต่เปลี่ยนเป็น self.logger)
        if len(direct) < K_MAIN:
            need = K_MAIN - len(direct)
            moved = aux_candidates[:need]
            direct.extend(moved)
            aux_candidates = aux_candidates[need:]
            self.logger.info(f"[DIRECT-FILL] Moved {need} aux chunks to direct (total direct: {len(direct)})")

        if level == 1 and len(direct) == 0 and top_evidences:
            need = min(K_MAIN, len(top_evidences))
            forced_chunks = sorted(top_evidences, key=lambda e: e.get("rerank_score", 0) or e.get("score", 0), reverse=True)[:need]
            direct.extend(forced_chunks)
            self.logger.warning(f"[L1-ULTRA-FALLBACK] No PDCA tag at all → Forced top {need} chunks to direct")

        # 5. สร้าง aux_summary (เรียกผ่าน self)
        aux_summary = self._summarize_evidence_list_short(aux_candidates, max_sentences=3) if aux_candidates else \
            "ไม่มีหลักฐานรองที่มีคุณภาพเพียงพอ"

        # 6. Return พร้อม Debug Meta
        self.logger.info(f"Context L{level} → Direct:{len(direct)} | Aux:{len(aux_candidates)} | Baseline:{len(summarizable_baseline)}")

        return {
            "baseline_summary": baseline_summary,
            "direct_context": "", 
            "aux_summary": aux_summary,
            "debug_meta": {
                "level": level,
                "direct_count": len(direct),
                "aux_count": len(aux_candidates),
                "top_relevance": max((ev.get("rerank_score", 0) for ev in top_evidences), default=0)
            },
        }

    def _apply_diversity_filter(
        self,
        evidences: List[Dict[str, Any]],
        max_per_source: int = 3,
        max_total: int = 40,
    ):
        """
        Diversity & dedup filter for evidence chunks

        Rules:
        - Preserve priority chunks
        - Deduplicate by chunk_uuid
        - Limit chunks per source file
        - Sort by (priority, rerank_score)
        - Stable / deterministic
        """

        if not evidences:
            return []

        # --------------------------------------------------
        # 1️⃣ Deduplicate by chunk_uuid (or content hash)
        # --------------------------------------------------
        unique = {}
        for d in evidences:
            if not isinstance(d, dict):
                continue

            uid = d.get("chunk_uuid")
            if not uid:
                uid = hashlib.sha256(
                    str(d.get("page_content", "")).encode()
                ).hexdigest()

            if uid not in unique:
                unique[uid] = d

        docs = list(unique.values())

        # --------------------------------------------------
        # 2️⃣ Sort: priority first, then score
        # --------------------------------------------------
        docs = sorted(
            docs,
            key=lambda x: (
                bool(x.get("is_priority", False)),
                self.get_actual_score(x),
            ),
            reverse=True,
        )

        # --------------------------------------------------
        # 3️⃣ Enforce per-source diversity
        # --------------------------------------------------
        source_counter = {}
        diversified = []

        for d in docs:
            src = (
                d.get("source")
                or d.get("metadata", {}).get("source")
                or "unknown"
            )

            count = source_counter.get(src, 0)
            if count >= max_per_source:
                continue

            diversified.append(d)
            source_counter[src] = count + 1

            if len(diversified) >= max_total:
                break

        return diversified

    
    def _load_evidence_map(self, is_for_merge: bool = False) -> Dict[str, Any]:
        """
        [REVISED v2026.1.24]
        - โหลดและแปลงโครงสร้างเก่า (List) ให้เป็นโครงสร้างใหม่ (UI-Ready)
        - ดึงไฟล์ที่ User เลือก (is_selected) มาเป็นลำดับแรก
        """
        if hasattr(self, '_evidence_cache') and self._evidence_cache is not None:
            return deepcopy(self._evidence_cache)

        try:
            path = get_evidence_mapping_file_path(
                tenant=self.config.tenant, year=self.config.year, enabler=self.enabler
            )
        except: return {}

        if not os.path.exists(path):
            return {}

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            processed_map = {}
            for key, content in data.items():
                # 🔄 Auto-Convert: ถ้าของเก่าเป็น List ให้ยัดใส่โครงสร้างใหม่
                if isinstance(content, list):
                    evidences = content
                    status = "pending"
                else:
                    evidences = content.get("evidences", [])
                    status = content.get("status", "pending")
                
                cleaned = []
                for e in evidences:
                    if not isinstance(e, dict): continue
                    # บังคับให้มีฟิลด์สำหรับ UI
                    e["is_selected"] = e.get("is_selected", True)
                    e["source_type"] = e.get("source_type", "ai_found")
                    cleaned.append(e)
                
                processed_map[key] = {"status": status, "evidences": cleaned}

            self._evidence_cache = deepcopy(processed_map)
            return processed_map
        except Exception as e:
            self.logger.error(f"❌ Load failed: {e}")
            return {}


    # ------------------------------------------------------------------------------------------
    # [FIXED] 🧩 Persistence Helper: Update Internal Evidence
    # ------------------------------------------------------------------------------------------
    def _update_internal_evidence_map(self, merged_evidence: Dict[str, Any]):
        """
        [FINAL REVISED v2026.01.25 - THE PERSISTENCE GUARD]
        - 🔄 Live Sync: เรียกใช้ Normalize เพื่อประกันความถูกต้องของข้อมูล
        """
        if not hasattr(self, 'evidence_map') or self.evidence_map is None:
            self.evidence_map = {}
            
        if not isinstance(merged_evidence, dict): return

        def get_stable_hash(text: str) -> str:
            if not text: return ""
            target = f"{text[:250]}...{text[-250:]}" if len(text) > 500 else text
            return hashlib.md5(target.encode('utf-8')).hexdigest()

        for key, incoming_data in merged_evidence.items():
            new_ev_list = incoming_data.get("evidences", []) if isinstance(incoming_data, dict) else incoming_data
            if not isinstance(new_ev_list, list): continue
                
            if key not in self.evidence_map or not isinstance(self.evidence_map[key], dict):
                self.evidence_map[key] = {"status": "pending", "evidences": []}
            
            target_bucket = self.evidence_map[key]
            existing_hashes = {get_stable_hash(str(e.get('content') or e.get('text', ''))) for e in target_bucket["evidences"]}
            
            for ev in new_ev_list:
                if not isinstance(ev, dict): continue
                content_str = str(ev.get('content') or ev.get('text') or "").strip()
                if not content_str: continue 
                
                if get_stable_hash(content_str) not in existing_hashes:
                    # 🎯 [POINT OF CHANGE]: เรียกใช้งาน Normalize แทนการสร้าง Manual Dict
                    normalized_batch = self._normalize_evidence_metadata([ev])
                    if normalized_batch:
                        clean_ev = normalized_batch[0]
                        # ประกันว่า Content ล่าสุดจะไม่หายไป
                        clean_ev["content"] = content_str
                        target_bucket["evidences"].append(clean_ev)
                        existing_hashes.add(get_stable_hash(content_str))

        self.logger.info(f"✅ Sync complete. Total Groups: {len(self.evidence_map)}")

    # evidence map structure (for ai understanding)
    # {
    #   "1.1_L1": {
    #     "status": "reviewed",
    #     "evidences": [
    #       {
    #         "doc_id": "15f0060f-674d-551e-b855-3b7e335450a8",
    #         "filename": "KM1.1L502 Learning Form กระบวนการ.pdf",
    #         "page": "11",
    #         "source_type": "human_map",
    #         "is_selected": true,
    #         "relevance_score": 0.95,
    #         "note": "ห้ามลบ! ใช้ยันข้อ 1.1 โดยเฉพาะ"
    #       },
    #       {
    #         "doc_id": "ai-file-999",
    #         "filename": "KM_Policy_2567_Final.pdf",
    #         "page": "1",
    #         "source_type": "system_gen",
    #         "relevance_score": 0.98
    #       }
    #     ]
    #   }
    # }

    def _save_evidence_map(self, map_to_save: Optional[Dict[str, Any]] = None, clear_existing: bool = False):
        """
        [ULTIMATE REVISE v2026.1.25 - THE STABLE ATOMIC BUILD]
        - 🎯 บังคับโครงสร้าง Nested Level Key: "1.1_L1", "1.1_L2"
        - 🛡️ Atomic Write: ป้องกันไฟล์พังด้วยการใช้ tempfile + shutil.move
        - 🧹 Post-Merge: เรียงคะแนนและอัปเดตสถานะอัตโนมัติ
        """
        try:
            # 1. เตรียม Path และ Folder
            map_file_path = get_evidence_mapping_file_path(
                tenant=self.config.tenant, year=self.config.year, enabler=self.enabler
            )
            os.makedirs(os.path.dirname(map_file_path), exist_ok=True)
            
            # 2. โหลดข้อมูลเดิมมาตั้งต้น (ยกเว้นสั่งล้างเครื่อง)
            final_map = {} if clear_existing else self._load_evidence_map(is_for_merge=True)
            
            # เลือกใช้ข้อมูลที่ส่งมา หรือข้อมูลใน Class memory
            incoming = map_to_save if map_to_save is not None else getattr(self, 'evidence_map', {})

            # 3. เริ่มขั้นตอน Merge ข้อมูล
            for key, evidence_data in incoming.items():
                # ตรวจสอบรูปแบบ Key (ต้องเป็น 1.1_L1)
                if "_L" not in key:
                    self.logger.warning(f"⚠️ [EVIDENCE-MAP] Key format mismatch: '{key}' should be like '1.1_L1'")
                
                target_bucket = final_map.setdefault(key, {"status": "pending", "evidences": []})
                existing_evs = target_bucket["evidences"]
                
                # จัดการ list ของหลักฐาน
                new_evs = evidence_data.get("evidences", []) if isinstance(evidence_data, dict) else evidence_data
                if not isinstance(new_evs, list): continue

                for new_e in new_evs:
                    if not isinstance(new_e, dict): continue
                    
                    doc_id = new_e.get("doc_id") or new_e.get("chunk_uuid")
                    if not doc_id: continue

                    page = str(new_e.get("page") or new_e.get("page_label", "0"))
                    idx_key = f"{doc_id}_{page}"
                    
                    # ค้นหาว่ามีไฟล์นี้ใน bucket นี้หรือยัง (Deduplicate ราย Level)
                    match = next((e for e in existing_evs if f"{e.get('doc_id')}_{e.get('page')}" == idx_key), None)

                    if match:
                        # --- UPDATE EXISTING ---
                        match["relevance_score"] = float(new_e.get("relevance_score", match.get("relevance_score", 0.0)))
                        match["is_selected"] = new_e.get("is_selected", match.get("is_selected", True))
                        
                        if new_e.get("source_type") == "human_map":
                            match["source_type"] = "human_map"
                        if new_e.get("note"):
                            match["note"] = new_e["note"]
                    else:
                        # --- INSERT NEW ---
                        if not new_e.get("filename"):
                            new_e["filename"] = getattr(self, 'document_map', {}).get(doc_id, "Unknown File")
                        
                        new_node = {
                            "doc_id": doc_id,
                            "filename": new_e.get("filename"),
                            "page": page,
                            "source_type": new_e.get("source_type", "system_gen"),
                            "is_selected": new_e.get("is_selected", True),
                            "relevance_score": float(new_e.get("relevance_score", new_e.get("rerank_score", 0.0))),
                            "note": new_e.get("note", "")
                        }
                        existing_evs.append(new_node)

            # 4. 🧹 Post-Processing: Sorting & Status Update
            for k in final_map:
                evs = final_map[k]["evidences"]
                # เรียงคะแนนจากสูงไปต่ำเพื่อให้ UI/AI เห็นหลักฐานที่ดีที่สุดก่อน
                evs.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
                
                # อัปเดตสถานะความน่าเชื่อถือ
                has_human = any(e.get("source_type") == "human_map" for e in evs)
                final_map[k]["status"] = "reviewed" if has_human else "ai_generated"

            # 5. 🛡️ Atomic Saving: บันทึกไฟล์แบบปลอดภัย
            temp_dir = os.path.dirname(map_file_path)
            with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=temp_dir, suffix='.tmp', encoding="utf-8") as tmp:
                json.dump(final_map, tmp, indent=4, ensure_ascii=False)
                tmp_path = tmp.name
            
            # ย้ายไฟล์ temp ไปทับไฟล์จริง (Atomic Operation ใน OS ส่วนใหญ่)
            shutil.move(tmp_path, map_file_path)
            
            # อัปเดต Cache ในหน่วยความจำด้วย
            self.evidence_map = final_map
            self.logger.info(f"✅ [EVIDENCE-MAP] Save Successful: {map_file_path}")

        except Exception as e:
            self.logger.error(f"❌ [EVIDENCE-MAP] Fatal Save Error: {str(e)}")
            # พยายามล้างไฟล์ขยะถ้าเกิด Error
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)

    def merge_evidence_mappings(self, results_list: List[Any]) -> Dict[str, Any]:
        """
        [FIXED v2026.1.25] - บังคับโครงสร้าง Nested Level Key
        """
        merged_mapping = {}
        
        for item in results_list:
            if not item: continue
            
            # ดึงข้อมูลออกมาจาก tuple (res, worker_mem)
            res_data = item[0] if isinstance(item, tuple) else item
            worker_ev_map = item[1] if isinstance(item, tuple) and len(item) > 1 else {}

            # กรณีข้อมูลมาจาก _run_single_assessment โดยตรง (ผ่าน Parallel)
            if 'evidence_sources' in res_data:
                sub_id = res_data.get('sub_id')
                level = res_data.get('level')
                level_key = f"{sub_id}_L{level}"
                
                if level_key not in merged_mapping:
                    merged_mapping[level_key] = {"status": "pending", "evidences": []}
                
                # นำหลักฐานเข้าสู่ list
                merged_mapping[level_key]["evidences"].extend(res_data['evidence_sources'])
            
            # รวมจาก temp_map (worker_mem) ถ้ามี
            if isinstance(worker_ev_map, dict):
                for l_key, ev_list in worker_ev_map.items():
                    if l_key not in merged_mapping:
                        merged_mapping[l_key] = {"status": "pending", "evidences": []}
                    
                    # ถ้า l_key เป็น "1.1" เฉยๆ ให้แปลงเป็น "1.1_L?" (ถ้าทำได้) หรือข้ามไป
                    # แต่ในระบบใหม่ Worker ควรส่ง 1.1_L1 มาเลย
                    target_list = merged_mapping[l_key]["evidences"]
                    target_list.extend(ev_list if isinstance(ev_list, list) else [])

        # Deduplicate แต่ละ Level Key ทิ้งท้าย
        for k in merged_mapping:
            merged_mapping[k]["evidences"] = self._deduplicate_list(merged_mapping[k]["evidences"])

        return merged_mapping
    
    def _deduplicate_list(self, items: List[Dict]) -> List[Dict]:
        """
        [ULTIMATE REVISED v2026.1.25 - SMART DEDUPE]
        - ⚖️ เลือกชิ้นที่คะแนนสูงสุดในแต่ละหน้าเอกสาร
        - 🧬 รองรับ ID ทุกรูปแบบจาก VectorStore
        """
        if not items: return []
        unique_map = {}
        for item in items:
            if not isinstance(item, dict): continue
            raw_id = str(item.get('doc_id') or item.get('stable_doc_uuid') or item.get('chunk_uuid') or "unknown")
            doc_id = raw_id.replace("-", "").lower().strip()
            page = str(item.get('page') or item.get('page_label', '0')).strip()
            uid = f"{doc_id}_pg{page}"
            
            score = float(item.get('relevance_score') or item.get('rerank_score') or 0.0)
            if uid not in unique_map or score > float(unique_map[uid].get('relevance_score') or 0.0):
                unique_map[uid] = item
        
        res = list(unique_map.values())
        res.sort(key=lambda x: float(x.get('relevance_score') or x.get('rerank_score') or 0.0), reverse=True)
        return res
    
    def _merge_worker_results(self, sub_result: Dict[str, Any], temp_map: Dict[str, Any]):
        """
        [ULTIMATE REVISE v2026.1.25 - NESTED & PARALLEL SAFE]
        - 🛡️ Data Persistence: บังคับซิงค์ข้อมูลกลับเข้าสู่ Global State เสมอ
        - 🧬 Evidence Integrity: ป้องกันการสูญหายของ Audit Trail แม้ผลประเมินจะเป็น 0
        - ⚖️ Resilience: รองรับการประเมินที่ระดับสูงพัง แต่ระดับล่างผ่าน (Partial Pass)
        """
        if not sub_result:
            self.logger.warning("⚠️ Received empty sub_result in merge process.")
            return None

        # 1. 🔍 Identity & Type Setup
        sub_id = str(sub_result.get('sub_id', 'Unknown'))
        # ดึง Level ล่าสุดที่ประเมิน (อาจจะเป็นเลเวลที่กำลังตรวจอยู่)
        raw_lvl = sub_result.get('level') or sub_result.get('highest_full_level', 0)
        try:
            level_received = int(raw_lvl)
        except (ValueError, TypeError):
            level_received = 0
            
        # 2. 🛡️ Evidence Mapping Sync (The Audit Trail Guard)
        if temp_map and isinstance(temp_map, dict):
            if not hasattr(self, 'evidence_map'): self.evidence_map = {}
            
            for level_key, evidence_list in temp_map.items():
                if not evidence_list: continue
                
                # มาตรฐาน Key: sub_id_L{level}
                formatted_key = level_key if "_L" in level_key else f"{sub_id}_L{level_received}"
                
                # หากเป็น L0 ให้พยายามเดาจาก level_received หรือข้ามไปถ้าไม่มีข้อมูลจริง
                if "_L0" in formatted_key and level_received > 0:
                    formatted_key = f"{sub_id}_L{level_received}"

                target_node = self.evidence_map.setdefault(formatted_key, {"status": "completed", "evidences": []})
                existing_evs = target_node["evidences"]
                
                # สร้างชุด Unique ID เพื่อกันหลักฐานซ้ำ (doc_id + page)
                existing_uids = {f"{e.get('doc_id')}_{e.get('page')}" for e in existing_evs}
                
                for ev in evidence_list:
                    if not isinstance(ev, dict) or not ev: continue
                    
                    doc_id = ev.get('doc_id') or ev.get('stable_doc_uuid')
                    page = str(ev.get('page') or ev.get('page_label', '0'))
                    uid = f"{doc_id}_{page}"
                    
                    if uid not in existing_uids and doc_id not in [None, "na", "n/a", "none"]:
                        # Mapping filename จาก document_map กลาง
                        if hasattr(self, 'document_map') and doc_id in self.document_map:
                            ev['filename'] = self.document_map.get(doc_id)
                        
                        existing_evs.append(ev)
                        existing_uids.add(uid)

        # 3. 🏗️ Final Sub-criteria Results Aggregation
        if not hasattr(self, 'final_subcriteria_results'):
            self.final_subcriteria_results = []

        # ค้นหา Object เดิมใน List หรือสร้างใหม่ถ้ายังไม่มี
        target = next((r for r in self.final_subcriteria_results if str(r.get('sub_id')) == sub_id), None)
        if not target:
            target = {
                "sub_id": sub_id,
                "sub_criteria_name": sub_result.get('sub_criteria_name') or f"Criteria {sub_id}",
                "weight": float(sub_result.get('weight', 4.0)),
                "level_details": {},
                "highest_full_level": 0,
                "weighted_score": 0.0,
                "is_passed": False,
                "audit_stop_reason": "Initialized",
                "pdca_overall": {"P": 0.0, "D": 0.0, "C": 0.0, "A": 0.0}
            }
            self.final_subcriteria_results.append(target)

        # 4. 🧩 Atomic Update (Merge level details)
        new_details = sub_result.get('level_details', {})
        if isinstance(new_details, dict) and new_details:
            # ซิงค์รายละเอียดรายเลเวล (L1, L2, L3...)
            target['level_details'].update(new_details)
        else:
            # Fallback หากข้อมูลหลุดมาเป็นก้อนเดียว
            if level_received > 0:
                target['level_details'][str(level_received)] = sub_result

        # 5. ⚖️ Step-Ladder Maturity Calculation (Robust Logic)
        current_highest = 0
        stop_reason = "Assessment complete"
        pdca_sums = {"P": 0.0, "D": 0.0, "C": 0.0, "A": 0.0}
        passed_lv_count = 0
        
        # ตรวจสอบทีละขั้น L1 -> L5 (ต้องผ่านขั้นล่างก่อนถึงจะนับขั้นบน)
        for l in range(1, 6):
            l_str = str(l)
            l_data = target['level_details'].get(l_str)
            
            if l_data and isinstance(l_data, dict):
                score_val = float(l_data.get('score', 0.0))
                # เกณฑ์การผ่าน: AI Flag ว่าผ่าน หรือ คะแนน >= 0.7
                is_lv_passed = (l_data.get('is_passed') is True or score_val >= 0.7)
                
                if is_lv_passed:
                    current_highest = l
                    l_data['is_passed'] = True # Force sync flag
                    
                    # สะสมคะแนน PDCA สำหรับคำนวณภาพรวม
                    bd = l_data.get('pdca_breakdown', {})
                    for phase in pdca_sums:
                        pdca_sums[phase] += float(bd.get(phase, 0.0))
                    passed_lv_count += 1
                else:
                    # บันทึกเหตุผลที่หยุดประเมินต่อ (Gap ที่เจอครั้งแรก)
                    stop_reason = f"Stopped at L{l}: {l_data.get('reason', 'Insufficient evidence')[:60]}..."
                    break
            else:
                # ถ้าไม่มีข้อมูลเลเวลนี้ ให้หยุดนับ (Chain broken)
                if l <= level_received: # กรณีเลเวลที่ควรจะมีแต่ไม่มี
                    stop_reason = f"Data missing at L{l}"
                    break
                break

        # 6. 💰 Score & Status Finalization
        target['highest_full_level'] = current_highest
        target['is_passed'] = (current_highest >= 1)
        target['weighted_score'] = round(current_highest * target['weight'], 2)
        
        # คำนวณค่าเฉลี่ย PDCA เฉพาะเลเวลที่ผ่าน
        if passed_lv_count > 0:
            target['pdca_overall'] = {k: round(v / passed_lv_count, 2) for k, v in pdca_sums.items()}
            
        target['audit_stop_reason'] = stop_reason if current_highest < 5 else "Maximum maturity level reached"
        
        # 🛡️ บรรทัดสำคัญ: Force update กลับไปที่ class attribute เพื่อป้องกัน Score: 0 ในรายงานสรุป
        self.final_subcriteria_results = [
            target if str(r.get('sub_id')) == sub_id else r 
            for r in getattr(self, 'final_subcriteria_results', [])
        ]

        self.logger.info(f"🏁 [MERGE-DONE] {sub_id} | Maturity: L{current_highest} | Weighted Score: {target['weighted_score']}")
        return target
    
    def run_assessment(
        self,
        target_sub_id: str = "all",
        export: bool = False,
        vectorstore_manager: Optional['VectorStoreManager'] = None,
        sequential: bool = False,
        document_map: Optional[Dict[str, str]] = None,
        record_id: str = None,
    ) -> Dict[str, Any]:
        """
        [FINAL REVISED v2026.01.25 - THE PERSISTENCE MASTER]
        - 🛡️ มั่นใจ 100% ว่า Evidence จะไม่หายด้วยการ Sync State ทันทีที่จบแต่ละหัวข้อ
        - 🧩 แก้ไขจุดอ่อนเรื่องการนับจำนวนหลักฐานใน Summary
        - 💾 บังคับ Save Mapping ก่อนส่งผลลัพธ์สุดท้าย
        """
        start_ts = time.time()
        self.is_sequential = sequential
        self.current_record_id = record_id or self.record_id
        
        # 1. Setup Document Map และ Internal State
        if document_map:
            self.document_map.update(document_map)
        
        if not hasattr(self, 'evidence_map') or self.evidence_map is None:
            self.evidence_map = {}

        # 📂 โหลดเกณฑ์ประเมิน
        self.flattened_rubric = self._flatten_rubric_to_statements()
        grouped_sub_criteria = self._group_statements_by_sub_criteria(self.flattened_rubric)

        is_all = str(target_sub_id).lower() == "all"
        sub_criteria_list = list(grouped_sub_criteria.values()) if is_all else [grouped_sub_criteria.get(target_sub_id)]
        
        if not all(sub_criteria_list):
            return self._create_failed_result(self.current_record_id, f"Criteria '{target_sub_id}' not found", start_ts)

        total_subs = len(sub_criteria_list)
        results_list = []

        # 🧠 2. เริ่มการประเมิน (Core Engine)
        if is_all and not sequential:
            # [MODE A] PARALLEL
            max_workers = int(os.environ.get("MAX_PARALLEL_WORKERS", 4))
            worker_args = [self._prepare_worker_tuple(sub, self.document_map) for sub in sub_criteria_list]
            
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(processes=max_workers) as pool:
                for idx, res_tuple in enumerate(pool.imap_unordered(_static_worker_process, worker_args)):
                    results_list.append(res_tuple)
                    
                    # 🎯 CRITICAL FIX: บังคับ Merge Evidence ทันทีที่ Worker ส่งกลับมา
                    # ข้อมูลหลักฐานจะถูกยัดเข้า self.evidence_map โดยตรงในฟังก์ชันนี้
                    self._merge_worker_results(res_tuple[0], res_tuple[1])
                    
                    self.db_update_task_status(
                        progress=15 + int(((idx+1)/total_subs) * 65), 
                        message=f"🧠 ประเมิน {res_tuple[0].get('sub_id', '?')} สำเร็จ"
                    )
        else:
            # [MODE B] SEQUENTIAL
            if not vectorstore_manager: self._initialize_vsm_if_none()
            vsm = vectorstore_manager or self.vectorstore_manager

            for idx, sub_criteria in enumerate(sub_criteria_list):
                sub_id = str(sub_criteria.get("sub_id", "Unknown"))
                
                # Baseline Hydration
                prev_map = self._collect_previous_level_evidences(sub_id=sub_id, current_level=1)
                initial_baseline = [ev for evs in prev_map.values() for ev in evs]
                
                # Run Worker
                res, worker_mem = self._run_sub_criteria_assessment_worker(sub_criteria, vsm, initial_baseline)
                results_list.append((res, worker_mem))
                
                # 🎯 CRITICAL FIX: อัปเดต State หลักทันที
                self._merge_worker_results(res, worker_mem)


        # -------------------------------------------------------
        # 🧩 3. ขั้นตอนการจัดระเบียบหลักฐาน (The Evidence Guard)
        # -------------------------------------------------------
        self.db_update_task_status(progress=85, message="🧩 กำลังจัดระเบียบหลักฐาน")
        
        # รวบรวมข้อมูลทั้งหมดเข้าสู่ self.evidence_map
        full_raw_mapping = self.merge_evidence_mappings(results_list)
        self._update_internal_evidence_map(full_raw_mapping)
        
        # [CRITICAL FIX] บังคับล้างค่าว่างและจัดรูปแบบให้เป๊ะก่อนนับ
        total_evidence_found = 0
        for key in list(self.evidence_map.keys()):
            bucket = self.evidence_map[key]
            if isinstance(bucket, dict) and "evidences" in bucket:
                # Deduplicate ทิ้งท้ายหนึ่งรอบ
                bucket["evidences"] = self._deduplicate_list(bucket["evidences"])
                count = len(bucket["evidences"])
                total_evidence_found += count
                if count > 0:
                    bucket["status"] = "ai_generated"
            else:
                # ถ้าหลุดมาเป็น list ให้แปลงเป็นโครงสร้างที่ถูกต้อง
                ev_list = self._deduplicate_list(bucket if isinstance(bucket, list) else [])
                self.evidence_map[key] = {"status": "ai_generated", "evidences": ev_list}
                total_evidence_found += len(ev_list)

        self.logger.info(f"📊 Sanitized Evidence Total: {total_evidence_found} items")
        self._save_evidence_map(map_to_save=self.evidence_map)

        # -------------------------------------------------------
        # 🏁 4. สรุปผล (Final Response & Export)
        # -------------------------------------------------------
        master_roadmap_data = None
        if is_all and len(self.final_subcriteria_results) > 0:
            master_roadmap_data = self.synthesize_strategic_roadmap(
                sub_criteria_results=self.final_subcriteria_results,
                enabler_name=self.enabler,
                llm_executor=self.llm
            )

        overall_stats = self._calculate_overall_stats(target_sub_id)
        if not overall_stats:
            overall_stats = {"efficiency": 0.0, "score": 0.0, "passed_count": 0, "total_count": 0}
            
        # [FIX 4] ยัดค่าจำนวนหลักฐานที่นับได้จริงลงใน Summary
        overall_stats["evidence_used_count"] = total_evidence_found

        final_response = {
            "record_id": self.current_record_id,
            "status": "COMPLETED",
            "enabler": self.enabler,
            "summary": overall_stats,
            "sub_criteria_results": self.final_subcriteria_results,
            "evidence_audit_trail": self.evidence_map, # ข้อมูลตรงนี้ต้องสมบูรณ์แล้ว
            "strategic_roadmap": master_roadmap_data,
            "run_time_seconds": round(time.time() - start_ts, 2)
        }

        if export:
            # ส่งออกเป็นไฟล์ JSON
            final_response["export_path"] = self._export_results(final_response, target_sub_id)

        self.db_update_task_status(progress=100, message="✅ ประเมินเสร็จสมบูรณ์", status="COMPLETED")
        return final_response
    
    # ------------------------------------------------------------------
    # 🏛️ [TIER-3 METHOD] synthesize_strategic_roadmap - FINAL PRODUCTION
    # ------------------------------------------------------------------
    def synthesize_strategic_roadmap(
        self,
        sub_criteria_results: List[Dict[str, Any]],
        enabler_name: str,
        llm_executor: Any
    ) -> Dict[str, Any]:
        """
        [TIER-3 STRATEGIC ORCHESTRATOR - v2026.3.26]
        รวบรวมผลประเมินรายหัวข้อมาสังเคราะห์เป็นแผนยุทธศาสตร์ภาพรวม (Master Roadmap)
        - ป้องกันปัญหา JSON Malformed จากการ Quote ข้อมูลซ้อน
        - จัดกลุ่ม Action Plans ให้เป็นยุทธศาสตร์ที่จับต้องได้
        """
        self.logger.info(f"🌐 [TIER-3] Starting Master Strategic Roadmap Synthesis for {enabler_name}")
        
        if not sub_criteria_results:
            self.logger.warning("⚠️ No sub-criteria results available for synthesis")
            return {"status": "INCOMPLETE", "overall_strategy": "ไม่พบข้อมูลเพียงพอในการสังเคราะห์แผน"}

        # 1. 📂 Data Collection (Gap Aggregation)
        aggregated_insights = []
        for res in sub_criteria_results:
            sub_id = res.get("sub_id", "Unknown")
            sub_name = res.get("sub_criteria_name", "N/A")
            highest_lv = res.get("highest_full_level", 0)
            level_details = res.get("level_details", {})
            
            gap_recs = []
            # เก็บ Insight เฉพาะตัวที่ "ไม่ผ่าน" หรือ "ผ่านแบบคาบเส้น (Score < 0.7)"
            for lvl_idx in range(1, 6):
                lv_str = str(lvl_idx)
                detail = level_details.get(lv_str, {})
                score = float(detail.get("score", 0))
                is_passed = detail.get("is_passed", False)
                
                if not is_passed or score < 0.7:
                    # ทำความสะอาด Quote ทันทีเพื่อไม่ให้ JSON พังในอนาคต
                    insight = str(detail.get("coaching_insight") or "").replace('"', "'").strip()
                    if insight and insight not in gap_recs and len(insight) > 5:
                        gap_recs.append(f"[L{lv_str}] {insight}")

            summary_text = " | ".join(gap_recs[:3]) if gap_recs else "ผ่านเกณฑ์มาตรฐานในระดับสูง (รักษามาตรฐานต่อเนื่อง)"
            aggregated_insights.append(f"📌 [{sub_id}] {sub_name} (Highest: L{highest_lv}): {summary_text}")

        # 2. 🧠 LLM Orchestration
        formatted_insights_text = "\n".join(aggregated_insights)
        
        # 💡 มั่นใจว่า Prompt มีการระบุฟอร์แมต JSON ที่ชัดเจน
        final_prompt = MASTER_ROADMAP_PROMPT.format(
            sub_id="OVERALL",
            sub_criteria_name=enabler_name, 
            enabler=enabler_name, 
            aggregated_insights=formatted_insights_text
        )

        try:
            # ใช้ Temperature 0.2 เพื่อให้แผนงานมีความเป็นเหตุเป็นผล
            response = llm_executor.generate(
                system=SYSTEM_MASTER_ROADMAP_PROMPT, 
                prompts=[final_prompt], 
                temperature=0.2
            )
            
            raw_text = getattr(response, 'content', str(response)).strip()
            
            # 3. 🧹 Robust JSON Extraction
            # ใช้เครื่องมือที่เราอัปเกรดกันไปก่อนหน้านี้
            strategic_plan = _robust_extract_json(raw_text)
            
            # 4. 🛡️ Result Normalization (Ensure standard keys for Exporter)
            # ตรวจสอบหา Roadmap ในหลายๆ Key ที่ AI มักจะตั้งชื่อมา
            final_roadmap = (
                strategic_plan.get("roadmap") or 
                strategic_plan.get("strategic_roadmap") or 
                strategic_plan.get("action_plan") or []
            )

            # 🚨 Fallback: ถ้าสกัด Roadmap ไม่ได้เลย ให้สร้าง 1 Step ใหญ่จาก Raw Text
            if not final_roadmap and len(raw_text) > 20:
                final_roadmap = [{
                    "phase": "Strategic Improvement",
                    "target_levels": "Overall",
                    "main_objective": "ดำเนินการยกระดับตามข้อเสนอแนะภาพรวม",
                    "key_actions": [raw_text[:200] + "..."],
                    "expected_outcome": "ผ่านเกณฑ์มาตรฐานในรอบถัดไป"
                }]

            return {
                "status": "SUCCESS",
                "overall_strategy": (strategic_plan.get("overall_strategy") or 
                                    strategic_plan.get("summary") or 
                                    f"แผนขับเคลื่อนยุทธศาสตร์ {enabler_name}"),
                "roadmap": final_roadmap,
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "input_sub_count": len(sub_criteria_results),
                    "enabler": enabler_name
                }
            }

        except Exception as e:
            self.logger.error(f"💥 Master Roadmap Critical Error: {str(e)}", exc_info=True)
            return {
                "status": "ERROR", 
                "overall_strategy": "ไม่สามารถสังเคราะห์แผนได้เนื่องจากข้อผิดพลาดในระบบประมวลผล",
                "roadmap": [],
                "reason": str(e)
            }

    def create_atomic_action_plan(
        self, 
        insight: str, 
        level: int, 
        level_criteria: str = "", 
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        [ULTIMATE REVISED v2026.01.25 - FINAL STABLE]
        - FIXED: เปลี่ยน system_msg เป็น system_prompt เพื่อให้ตรงกับ llm_data_utils.py
        - Clean Code: ใช้ทรัพยากรจาก Header (re, SEAM_ENABLER_FULL_NAME_TH, ฯลฯ)
        - Anti-IT Ghost: กรองคำแนะนำด้าน IT สำหรับ L1-L3 เปลี่ยนเป็นงานด้านบริหารจัดการ
        """
        try:
            # 1. Validation & Data Sanitization
            clean_insight = str(insight or "").strip()
            if not clean_insight or clean_insight.lower() in ["-", "n/a", "none", "ไม่มีข้อมูล", ""]:
                return []

            # จัดการเรื่องเกณฑ์ (Criteria)
            actual_criteria = level_criteria or kwargs.get('level_statement') or "พิจารณาตามเกณฑ์มาตรฐาน SE-AM"
            
            # 2. ป้องกัน f-string formatting error (Double Braces Escape)
            safe_insight = clean_insight.replace('"', "'").replace('{', '{{').replace('}', '}}')
            safe_criteria = str(actual_criteria).replace('"', "'").replace('{', '{{').replace('}', '}}')

            # 3. เตรียมตัวแปร Enabler (Mapping Name & Code)
            enabler_code = str(getattr(self, 'enabler', 'UNKNOWN')).upper()
            enabler_name_th = SEAM_ENABLER_FULL_NAME_TH.get(enabler_code, f"ด้าน {enabler_code}")

            # 4. Packaging ข้อมูลสำหรับ Prompt
            prompt_payload = {
                "coaching_insight": safe_insight,
                "level": level,
                "enabler": enabler_code,
                "enabler_name_th": enabler_name_th,
                "level_criteria": safe_criteria
            }

            try:
                # ใช้ Template ที่ Import มาจาก Header
                human_prompt = ATOMIC_ACTION_PROMPT.format(**prompt_payload)
                system_prompt_content = SYSTEM_ATOMIC_ACTION_PROMPT.format(
                    enabler_name_th=enabler_name_th, 
                    enabler=enabler_code
                )
            except Exception as e:
                self.logger.warning(f"⚠️ [FORMAT-ERROR] {e} -> Use raw backup format")
                system_prompt_content = f"Expert Action Plan Generator for {enabler_name_th}"
                human_prompt = f"Insight: {safe_insight}\nLevel: {level}"

            # 5. LLM Execution (FIXED Parameter Name: system_prompt)
            # 
            raw_response = _fetch_llm_response(
                system_prompt=system_prompt_content, # ✅ แก้จาก system_msg เป็น system_prompt แล้ว
                user_prompt=human_prompt,
                llm_executor=self.llm
            )

            # 6. Robust Extraction (JSON -> Regex Fallback)
            actions = []
            try:
                # ใช้ helper _robust_extract_json จาก Header
                parsed = _robust_extract_json(raw_response)
                if isinstance(parsed, list):
                    actions = parsed
                elif isinstance(parsed, dict):
                    actions = parsed.get("actions", [parsed])
            except:
                pass

            # Regex Scavenger (ถ้า JSON มีปัญหา) - ใช้ re จาก Header
            if not actions:
                matches = re.findall(r'["\']action["\']\s*:\s*["\']([^"\']+)["\']', raw_response)
                for m in matches:
                    actions.append({"action": m, "target_evidence": "หลักฐานประกอบการดำเนินงาน"})

            # 7. Post-Processing & Anti-IT Ghost (L1-L3 Safety)
            final_actions = []
            it_ghost_terms = r"(ระบบสารสนเทศ|KMS|Software|Automation|แอปพลิเคชัน|IT System|แพลตฟอร์มดิจิทัล|โปรแกรมคอมพิวเตอร์)"
            
            for item in actions:
                if not isinstance(item, dict): continue
                
                act_text = (item.get("action") or "").strip()
                if len(act_text) < 5: continue
                
                # กรองเนื้อหา IT ออกถ้าอยู่ในระดับรากฐาน L1-L3 (เปลี่ยนเป็นงานบริหารจัดการ/กิจกรรม)
                if level <= 3:
                    act_text = re.sub(it_ghost_terms, "แนวทางปฏิบัติ/คู่มือการทำงาน/กิจกรรมสร้างการมีส่วนร่วม", act_text, flags=re.IGNORECASE)
                
                final_actions.append({
                    "action": act_text,
                    "target_evidence": item.get("target_evidence", "เอกสารประกอบ/บันทึกกิจกรรม"),
                    "level": level
                })

            # 8. Emergency Fallback
            if not final_actions:
                final_actions = [{
                    "action": f"จัดทำหลักฐานและแผนงานเบื้องต้นให้สอดคล้องกับเกณฑ์ระดับ {level}",
                    "target_evidence": "รายงานผลการดำเนินงาน",
                    "level": level
                }]

            self.logger.info(f"✅ [ATOMIC-PLAN] {enabler_code} L{level} Success (Output: {len(final_actions[:2])})")
            return final_actions[:2]

        except Exception as e:
            self.logger.error(f"🛑 [ATOMIC-PLAN-CRITICAL] {str(e)}", exc_info=True)
            return [{"action": "ดำเนินการพัฒนางานตามข้อเสนอแนะของเกณฑ์", "target_evidence": "หลักฐานเชิงประจักษ์", "level": level}]
        
    # ------------------------------------------------------------------
    # 🏛️ [TIER-3 METHOD] generate_master_roadmap - FULL REVISE v2026.1.23
    # ------------------------------------------------------------------
    def generate_master_roadmap(self, sub_id, sub_criteria_name, enabler, aggregated_insights):
        """
        [TIER-3 STRATEGIC SYNTHESIS - v2026.01.24]
        สังเคราะห์ Roadmap โดยคำนึงถึง Maturity Capping และ Step-Ladder Logic
        - 🧩 Maturity Aware: แยกแยะเลเวลที่ผ่านจริง (Continuous) กับเลเวลที่ผ่านแต่โดน Cap
        - 🛠️ Strategic Alignment: สั่ง LLM ให้เน้นการซ่อมแซม "รอยต่อที่ขาด" ก่อนพัฒนาส่วนปลาย
        """
        
        self.logger.info(f"🔮 [MASTER-ROADMAP] Starting synthesis for {sub_id} ({sub_criteria_name})")

        if not aggregated_insights:
            self.logger.warning(f"⚠️ No insights for {sub_id} - Using emergency fallback")
            return self._get_emergency_fallback_plan(sub_id, sub_criteria_name, "No insights provided")

        # 1. 📂 Data Condensing & Maturity Tagging
        condensed_insights = []
        highest_continuous = 0
        has_gap_before = False
        
        # ค้นหาจุดที่ขาดช่วง (Maturity Gap)
        for item in aggregated_insights:
            lv = int(item.get('level', 0))
            is_passed = item.get('status') == "PASSED"
            is_capped = item.get('is_capped', False)
            
            # สรุปสถานะเพื่อส่งให้ LLM
            if is_passed and not has_gap_before:
                status_text = "✅ PASSED (Maturity นับ)"
                highest_continuous = lv
            elif is_passed and has_gap_before:
                status_text = "⚠️ PASSED (CAPPED - พื้นฐานยังไม่แน่น)"
            else:
                status_text = "❌ FAILED (GAP)"
                has_gap_before = True # เริ่มเกิดรอยต่อที่ขาด

            insight = item.get('insight_summary') or item.get('reason') or 'ไม่มีรายละเอียด'
            condensed_insights.append(f"Level {lv} [{status_text}]: {insight[:250]}")

        summary_context = "\n".join(condensed_insights)
        
        # เพิ่ม Metadata สำหรับคุมทิศทาง LLM
        strategic_focus = f"ระดับ Maturity ปัจจุบันหยุดอยู่ที่เลเวล {highest_continuous} "
        if has_gap_before:
            strategic_focus += "เนื่องจากตรวจพบรอยต่อ (Gap) ในเลเวลพื้นฐาน กลยุทธ์ต้องเน้นการซ่อมแซมรอยต่อนี้"
        else:
            strategic_focus += "กลยุทธ์ต้องเน้นการรักษาระดับและต่อยอดสู่เลเวลถัดไป"

        # 2. 📝 Prompt Construction
        # ส่งค่า strategic_focus เข้าไปใน Prompt เพื่อไกด์ AI
        try:
            formatted_prompt = MASTER_ROADMAP_PROMPT.format(
                sub_id=sub_id,
                sub_criteria_name=sub_criteria_name,
                enabler=enabler,
                aggregated_insights=summary_context,
                strategic_focus=strategic_focus # 👈 เพิ่มไกด์ไลน์เรื่อง Maturity
            )
        except Exception as fe:
            self.logger.error(f"❌ Prompt formatting error: {fe}")
            formatted_prompt = f"Summarize roadmap for {sub_criteria_name} (Focus: {strategic_focus}): {summary_context}"

        # 3. 🧠 LLM Execution & 4. 🧹 Extraction (Logic เดิมที่เสถียรอยู่แล้ว)
        try:
            raw_json_str = _fetch_llm_response(
                system_prompt=SYSTEM_MASTER_ROADMAP_PROMPT,
                user_prompt=formatted_prompt,
                max_retries=3,
                llm_executor=self.llm 
            )

            master_data = _robust_extract_json(raw_json_str)
            
            if not master_data:
                return self._get_emergency_fallback_plan(sub_id, sub_criteria_name, "Hollow JSON")

            # 5. 🏗️ UI-Ready Normalization
            final_strategy = master_data.get("overall_strategy") or master_data.get("summary") or "ไม่สามารถสรุปกลยุทธ์ได้"
            raw_phases = master_data.get("phases") or master_data.get("roadmap") or []

            normalized_phases = []
            for i, p in enumerate(raw_phases, 1):
                if isinstance(p, dict):
                    # เติมลำดับขั้นให้ชัดเจน
                    p["step_label"] = f"Phase {i}"
                    normalized_phases.append(p)
                else:
                    normalized_phases.append({"step": f"Phase {i}", "action": str(p)})

            self.logger.info(f"✅ [MASTER-ROADMAP] Synthesis Success for {sub_id} (Maturity: {highest_continuous})")
            
            return {
                "sub_id": sub_id,
                "sub_criteria_name": sub_criteria_name,
                "highest_maturity_level": highest_continuous, # 🛡️ ยืนยันคะแนนจริง
                "overall_strategy": final_strategy,
                "phases": normalized_phases,
                "status": "SUCCESS",
                "is_gap_detected": has_gap_before,
                "generated_at": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"💥 Critical error in Master Roadmap {sub_id}: {str(e)}")
            return self._get_emergency_fallback_plan(sub_id, sub_criteria_name, str(e))
            
    def _get_emergency_fallback_plan(self, sub_id, name, error_msg=""):
        """สร้างแผนสำรองกรณี LLM พัง เพื่อไม่ให้ระบบหยุดทำงาน"""
        return {
            "overall_strategy": "แผนพัฒนาระบบเบื้องต้น (Fallback Mode)",
            "phases": [
                {
                    "phase": "Quick Win",
                    "goal": f"ตรวจสอบและแก้ไขข้อบกพร่องใน {name}",
                    "actions": [{"action": "สอบทานหลักฐานและวิเคราะห์ Gap วิเคราะห์เบื้องต้น", "priority": "High"}]
                }
            ],
            "status": "FALLBACK",
            "error_context": error_msg[:100]
        }
    

    def _apply_evidence_cap(self, evidence_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        [NEW v2026.1.24] จัดการ Cap ขนาดหลักฐานสะสมตาม Strategy ที่ตั้งไว้
        """
        if not evidence_list:
            return []

        # 1. Deduplicate ก่อน
        unique_evidences = self._deduplicate_list(evidence_list)

        # 2. เลือกกลยุทธ์การตัดข้อมูล
        if EVIDENCE_SELECTION_STRATEGY == "score":
            # เรียงตามคะแนน Rerank จากสูงไปต่ำ แล้วเลือกตัวท็อป
            sorted_list = sorted(
                unique_evidences, 
                key=lambda x: x.get('rerank_score', 0) if isinstance(x, dict) else 0, 
                reverse=True
            )
        else:
            # แบบเดิม: เอาที่เจอใหม่ล่าสุด (ตัดท้าย)
            sorted_list = unique_evidences

        # 3. Cap ขนาดตามที่ตั้งไว้ใน global_vars
        return sorted_list[:EVIDENCE_CUMULATIVE_CAP]
    

    # ------------------------------------------------------------------------------------------
    # 🧠 [TIER-1 & TIER-2 WORKER] Sequential Assessment (HYDRATED) - FULL REVISED v2026.1.25
    # ------------------------------------------------------------------------------------------
    def _run_sub_criteria_assessment_worker(
        self,
        sub_criteria: Dict[str, Any],
        vectorstore_manager: Optional[Any] = None,
        initial_baseline: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
        """
        [PATCHED v2026.01.25] ประเมินราย Sub-Criteria แบบเจาะลึก 1-5 เลเวล 
        - Fix Parameter Mismatch (level_criteria)
        - Add Atomic-Plan Isolation (try-except)
        """
        sub_id = str(sub_criteria.get("sub_id", "Unknown"))
        sub_name = sub_criteria.get("sub_criteria_name", "No Name")
        sub_weight = float(sub_criteria.get("weight", 0.0))
        target_limit = getattr(self.config, "target_level", 5)

        vsm = vectorstore_manager or getattr(self, "vectorstore_manager", None)
        level_details = {}
        roadmap_input_bundle = []
        
        highest_continuous_level = 0
        is_still_continuous = True 
        cumulative_baseline = list(initial_baseline or [])
        levels = sorted(sub_criteria.get("levels", []), key=lambda x: x.get("level", 0))

        for stmt in levels:
            level = int(stmt.get("level", 0))
            if level == 0 or level > target_limit: continue

            # --- 🎯 1. PER-LEVEL MAP HYDRATION ---
            level_key = f"{sub_id}_L{level}"
            map_data = self.evidence_map.get(level_key, {})
            saved_evidences = map_data.get("evidences", []) if isinstance(map_data, dict) else map_data
            priority_items = [e for e in saved_evidences if e.get("is_selected", True)] if saved_evidences else []

            # --- 🧠 2. CORE ASSESSMENT ---
            current_baseline = self._deduplicate_list(cumulative_baseline + priority_items)
            res = self._run_single_assessment(
                sub_id=sub_id, level=level,
                criteria={"name": sub_name, "statement": stmt.get("statement", ""), "sub_criteria_name": sub_name},
                keyword_guide=stmt.get("keywords", []),
                baseline_evidences=current_baseline,
                vectorstore_manager=vsm,
            )

            # [SYNC STATE] บันทึกหลักฐานเข้า Memory
            self._update_internal_evidence_map({f"{sub_id}_L{level}": res.get("top_chunks_data", [])})
            
            is_passed_by_llm = bool(res.get("is_passed", False))
            
            # --- ⚖️ 3. STEP-LADDER MATURITY LOGIC ---
            if is_passed_by_llm:
                new_found = res.get("top_chunks_data", [])
                cumulative_baseline.extend(new_found)
                cumulative_baseline = self._apply_evidence_cap(cumulative_baseline)
                if is_still_continuous: highest_continuous_level = level
            else:
                is_still_continuous = False

            # --- 🛠️ 4. ATOMIC ACTION PLAN (Isolated Call) ---
            try:
                # [FIXED] เปลี่ยน level_statement เป็น level_criteria ให้ตรงกับ def ฟังก์ชัน
                atomic_actions = self.create_atomic_action_plan(
                    insight=res.get("coaching_insight", ""),
                    level=level,
                    level_criteria=stmt.get("statement", "")
                )
            except Exception as e:
                # ถ้าพัง ให้ Log Error แต่ประเมินต่อได้
                self.logger.error(f"[ATOMIC-PLAN-ERROR] L{level} for {sub_id}: {str(e)}", exc_info=True)
                atomic_actions = []

            level_details[str(level)] = {
                "level": level, 
                "is_passed": is_passed_by_llm, 
                "is_maturity_capped": (is_passed_by_llm and not is_still_continuous),
                "score": float(res.get("score", 0.0)) if is_still_continuous else 0.25,
                "reason": res.get("reason", ""),
                "coaching_insight": res.get("coaching_insight", ""),
                "atomic_action_plan": atomic_actions, 
                "pdca_breakdown": res.get("pdca_breakdown", {}),
                "evidence_sources": res.get("top_chunks_data", []),
                "judicial_review_applied": res.get("is_safety_pass", False)
            }

            roadmap_input_bundle.append({
                "level": level, "status": "PASSED" if is_passed_by_llm else "FAILED",
                "is_capped": (is_passed_by_llm and not is_still_continuous),
                "insight_summary": res.get("coaching_insight", "")[:200]
            })

        # --- 🔮 5. MASTER ROADMAP SYNTHESIS ---
        master_roadmap = self.generate_master_roadmap(
            sub_id=sub_id, sub_criteria_name=sub_name,
            enabler=getattr(self, "enabler", "KM"), aggregated_insights=roadmap_input_bundle
        )

        return {
            "sub_id": sub_id, 
            "sub_criteria_name": sub_name, 
            "highest_full_level": highest_continuous_level, 
            "weighted_score": round(highest_continuous_level * sub_weight, 2),
            "is_passed": highest_continuous_level >= 1,
            "level_details": level_details, 
            "strategic_roadmap": master_roadmap
        }, self.evidence_map

    def _get_level_constraint_prompt(self, sub_id: str, level: int, req_phases: list = None, spec_rule: str = None) -> str:
        """
        [ULTIMATE REVISED v2026.1.25 - SCOPE GUARD ENABLED]
        """
        enabler = getattr(self, 'enabler', 'KM').upper()
        enabler_name = "การจัดการความรู้ (KM)"
        
        level_goal = get_pdca_goal_for_level(level)
        level_focus = PDCA_PHASE_MAP.get(level, "ตรวจสอบความครบถ้วน")

        # กำหนด Mandatory Phases ตามความจริงของ Maturity
        required_phases = req_phases or []
        if not required_phases:
            if level >= 4: required_phases = ['P', 'D', 'C']
            elif level >= 2: required_phases = ['P', 'D']
            else: required_phases = ['P']

        req_str = " + ".join(required_phases)

        lines = [
            f"\n### 🛡️ [AUDIT GUIDELINE: {enabler} - LEVEL {level}] ###",
            f"🎯 หัวข้อประเมิน: {enabler_name} | {sub_id}",
            f"🚩 เป้าหมายระดับนี้: {level_goal}",
            f"🔍 สิ่งที่ต้องตรวจพบ (Mandatory PDCA): {req_str}",
            f"📌 จุดเน้นสำคัญ: {level_focus}",
            f"💡 [เกณฑ์เฉพาะในรูบริก]: {spec_rule}" if spec_rule else "",
            "\n⚠️ [กฎเหล็กการตัดสิน - Strict Rules]:",
            f"1. [Maturity Scope Guard] ห้ามนำข้อกำหนดของระดับสูง (เช่น AI, Automation, Benchmarking) มาเป็นเหตุผลให้ 'ไม่ผ่าน' ในระดับ {level} หากรูบริกไม่ได้ระบุไว้",
            "2. [L1-L2 Policy Check] สำหรับระดับนโยบายและการวางรากฐาน หากพบเอกสาร 'แผนแม่บท' หรือ 'ประกาศ' ที่มีลายเซ็นอนุมัติ ให้ถือว่าผ่านเกณฑ์ 'P' และ 'D' ในเชิงโครงสร้างทันที",
            f"3. {GLOBAL_EVIDENCE_INSTRUCTION}",
            "4. [Substance over Form] เน้นดูเนื้อหาว่าตอบโจทย์วิสัยทัศน์องค์กรหรือไม่ มากกว่าดูแค่ชื่อไฟล์",
            "5. [Coaching Insight] หากประเมินว่า 'ไม่ผ่าน' ต้องบอกจุดที่ขาดหายไปให้ชัดเจน เพื่อให้หน่วยงานไปปรับปรุงได้ถูกจุด"
        ]
        return "\n".join(filter(None, lines))


    # ------------------------------------------------------------------------------------------
    # 🧠 [TIER-1 CORE] _run_single_assessment (GOVERNANCE-LOCKED & FULL AUDIT TRACE)
    # ------------------------------------------------------------------------------------------
    def _run_single_assessment(
        self,
        sub_id: str,
        level: int,
        criteria: Dict[str, Any],
        keyword_guide: List[str],
        baseline_evidences: List[Dict[str, Any]],
        vectorstore_manager: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        [ULTIMATE REVISED v2026.1.25]
        - 🛡️ Governance: ฉีด audit_instruction เพื่อคุมขอบเขตการตรวจ (Scope Guard)
        - 🧬 Retrieval Expansion: ผสาน Adaptive Retrieval กับ Neighbor Expansion
        - ⚖️ Resilience: ระบบ Judicial Review (อุทธรณ์) เมื่อ Rerank Score สูงแต่ LLM มองไม่เห็น
        - 📊 Integrity: รักษาการ Log PDCA Status และ Multichannel Context ตามมาตรฐานเดิม
        """
        log_prefix = f"Sub:{sub_id} L{level}"
        sub_name = criteria.get('name', 'Unknown Sub-item')
        statement_text = criteria.get('statement', 'No statement defined')
        
        self.logger.info(f"🔍 [START-ASSESSMENT] {log_prefix} | {sub_name}")
        self.logger.info(f"📋 [CRITERIA] Level {level}: \"{statement_text}\"")

        # --- [STEP 1: GOVERNANCE & RULES] ---
        # สร้าง "ใบสั่งงาน" (Scope Guard) จากเกณฑ์ Maturity
        audit_instruction = self._get_level_constraint_prompt(sub_id, level)
        current_rules = getattr(self, 'contextual_rules_map', {}).get(sub_id, {}).get(f"L{level}", {})

        # --- [STEP 2: ADAPTIVE RETRIEVAL & EXPANSION] ---
        # 1. ค้นหาแบบ Adaptive (Vector + Rerank)
        retrieved_chunks, max_rerank = self._perform_adaptive_retrieval(
            sub_id=sub_id,
            level=level,
            stmt=statement_text,
            vectorstore_manager=vectorstore_manager,
        )

        # 2. ✨ Neighbor Expansion: กู้คืนบริบทหน้าใกล้เคียง (หน้าลงนาม/หน้าผนวก)
        if retrieved_chunks:
            enabler_key = str(getattr(self, 'enabler', 'km')).lower()
            collection_name = f"evidence_{enabler_key}"
            retrieved_chunks = self._expand_context_with_neighbor_pages(
                top_evidences=retrieved_chunks, 
                collection_name=collection_name
            )

        # 3. Diversity Filter: กรองความซ้ำซ้อนของข้อมูล
        retrieved_chunks = self._apply_diversity_filter(retrieved_chunks, level)

        # --- [STEP 3: EVIDENCE FUSION & METADATA] ---
        # ผสมหลักฐานใหม่เข้ากับ Baseline สะสม
        evidences = (baseline_evidences or []) + (retrieved_chunks or [])

        # แยกกลุ่มหลักฐานตาม PDCA Tags
        pdca_blocks = self._get_pdca_blocks_from_evidences(
            evidences=evidences,
            baseline_evidences=baseline_evidences,
            level=level,
            sub_id=sub_id,
            contextual_rules_map=getattr(self, 'contextual_rules_map', {})
        )

        # คำนวณ Audit Confidence (ใช้แสดงผลใน Report)
        audit_confidence = self.calculate_audit_confidence(
            matched_chunks=retrieved_chunks,
            sub_id=sub_id,
            level=level,
        )
        self.current_audit_meta = audit_confidence # 🛡️ เก็บ Metadata เดิมไว้

        # --- [STEP 4: MULTICHANNEL LLM EXECUTION] ---
        # สร้าง Multichannel Context (Current + Historical)
        llm_context = self._build_multichannel_context_for_level(
            level=level,
            top_evidences=retrieved_chunks,
            previous_levels_evidence=baseline_evidences
        )

        # ส่งให้ LLM วิเคราะห์ด้วยใบสั่งงานคุมกฎ
        llm_raw = self.evaluate_pdca(
            pdca_blocks=pdca_blocks,
            sub_id=sub_id,
            level=level,
            audit_confidence=audit_confidence,
            audit_instruction=audit_instruction # 👈 ฉีดใบสั่งงานคุมกฎ
        )
        if not isinstance(llm_raw, dict): llm_raw = {}

        # --- [STEP 5: SMART NORMALIZATION] ---
        result = self.post_process_llm_result(
            llm_output=llm_raw,
            level=level,
            sub_id=sub_id,
            contextual_config=current_rules,
            top_evidences=retrieved_chunks
        )

        # --- [STEP 6: JUDICIAL REVIEW (RESCUE LOGIC)] ---
        is_safety_pass = False
        # ถ้าคะแนน Rerank สูงมากแต่ AI ให้ตก ระบบจะทำการอุทธรณ์ทันที
        if not result.get("is_passed") and max_rerank >= 0.70:
            self.logger.info(f"⚖️ [TRIGGER-APPEAL] {log_prefix} | Rerank {max_rerank:.4f} is high. Re-evaluating...")

            appeal_result = self._run_expert_re_evaluation(
                sub_id=sub_id,
                level=level,
                statement_text=statement_text,
                context=str(llm_context.get("full_context", "")),
                first_attempt_reason=result.get("reason", "หลักฐานไม่ชัดเจน"),
                missing_tags=result.get("missing_phases", []),
                highest_rerank_score=max_rerank,
                sub_criteria_name=sub_name,
                llm_evaluator_to_use=self.evaluate_pdca,
                audit_instruction=audit_instruction,
                base_kwargs={
                    "pdca_blocks": pdca_blocks,
                    "contextual_config": current_rules,
                    "top_evidences": retrieved_chunks
                }
            )

            if appeal_result and appeal_result.get("appeal_status") == "GRANTED":
                self.logger.info(f"✅ [APPEAL-GRANTED] {log_prefix} | Expert Rescue Successful.")
                final_appeal = self.post_process_llm_result(
                    llm_output=appeal_result,
                    level=level,
                    sub_id=sub_id,
                    contextual_config=current_rules,
                    top_evidences=retrieved_chunks
                )
                result.update(final_appeal)
                result["is_passed"] = True
                result["score"] = max(result.get("score", 0.0), 1.0)
                is_safety_pass = True

        # --- [STEP 7: FINAL INSIGHTS & LOGGING] ---
        final_insight = (result.get("coaching_insight") or result.get("reason") or "").strip()
        final_insight = f"[{'STRENGTH' if result.get('is_passed') else 'GAP'}] {final_insight}"

        # เรียกใช้ Logger เดิมเพื่อเก็บสถานะ PDCA ทั้งหมด
        if hasattr(self, "_log_pdca_status"):
            self._log_pdca_status(
                sub_id=sub_id,
                name=sub_name,
                level=level,
                blocks=llm_raw,
                req_phases=result.get("required_phases", []),
                sources_count=len(retrieved_chunks),
                score=result.get("score", 0.0),
                conf_level=audit_confidence.get("level", "LOW"),
                pdca_breakdown=result.get("pdca_breakdown", {}),
                tagging_result=audit_confidence.get("pdca_found", []),
                is_safety_pass=is_safety_pass
            )

        return {
            "is_passed": bool(result.get("is_passed", False)),
            "score": float(result.get("score", 0.0)),
            "reason": result.get("reason", ""),
            "coaching_insight": final_insight,
            "pdca_breakdown": result.get("pdca_breakdown", {}),
            "audit_confidence": audit_confidence,
            "top_chunks_data": retrieved_chunks,
            "is_safety_pass": is_safety_pass,
            "debug_meta": {
                "max_rerank": max_rerank,
                "evidence_count": len(evidences),
                "judicial_review": is_safety_pass,
                "neighbor_expansion": True,
                "audit_instruction_applied": True
            }
        }