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
        CRITICAL_CA_THRESHOLD
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
        get_export_dir, get_rubric_file_path, _n
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

def get_enabler_full_name(enabler_code: str, lang: str = "th") -> str:
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
    code = str(enabler_code).upper().strip()
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
        record_id: Optional[str] = None, # 👈 เพิ่มตรงนี้
        **kwargs  
    ):
        """
        [ULTIMATE REVISE v2026.3] SEAM Assessment Engine Constructor
        เน้นความทนทาน (Resilience), ความปลอดภัย (Sanity Check) และความเร็วในการดึงข้อมูล
        """
        # -------------------------------------------------------
        # 1. Logger Setup (สำคัญอันดับ 1 เพื่อการ Trace Error)
        # -------------------------------------------------------
        self.doc_type = doc_type or getattr(config, 'doc_type', EVIDENCE_DOC_TYPES)
        clean_dt = str(self.doc_type).strip().lower()
        log_year = config.year if clean_dt == EVIDENCE_DOC_TYPES.lower() else "general"

        if logger_instance is not None:
            self.logger = logger_instance
        else:
            self.logger = logging.getLogger(__name__).getChild(
                f"Engine|{config.enabler}|{config.tenant}/{log_year}"
            )

        self.logger.info(f"🚀 Initializing SEAMPDCAEngine: {config.enabler} ({config.tenant}/{log_year})")

        # -------------------------------------------------------
        # 2. Patch: Sanity Check & Core Configuration
        # -------------------------------------------------------
        self.config = config
        
        # [CRITICAL PATCH] ตรวจสอบข้อมูลบังคับก่อนเริ่มงาน
        if not self.config.enabler or not self.config.tenant:
            self.logger.critical("❌ Mandatory Config Missing: enabler and tenant must be provided!")
            raise ValueError("Enabler and Tenant are required for SEAMPDCAEngine.")

        self.enabler = config.enabler
        self.tenant_id = config.tenant  # 👈 เพิ่มบรรทัดนี้: เพื่อให้ _get_semantic_tag เรียกใช้ได้
        self.year = config.year        # 👈 เพิ่มบรรทัดนี้: เพื่อความครบถ้วน
        self.target_level = config.target_level
        self.sub_id = sub_id
        self.llm = llm_instance
        self.vectorstore_manager = vectorstore_manager
        self.is_parallel_all_mode = is_parallel_all_mode
        self.is_sequential = getattr(config, 'force_sequential', True)
        self.results = {}

        # -------------------------------------------------------
        # 3. Database & System Warm-up
        # -------------------------------------------------------
        try:
            init_db()  # ป้องกัน 'no such table' หรือ Schema mismatch
            self.logger.info("📂 Database Schema verified/initialized.")
        except Exception as e:
            self.logger.error(f"⚠️ DB Init Warning: {e} (Check if tables already exist)")

        self.record_id = record_id # 👈 บันทึกเก็บไว้ใน instance

        # -------------------------------------------------------
        # 4. Data Loading (Rubric, Rules & Policies)
        # -------------------------------------------------------
        # โหลด Rubric ทันทีเพื่อป้องกัน AttributeError ในขั้นตอนประเมิน
        self.rubric = self._load_rubric()
        
        # โหลดกฎ Contextual Rules สำหรับ PDCA Logic
        self.contextual_rules_map = self._load_contextual_rules_map()
        
        self.retry_policy = RetryPolicy(
            max_attempts=3,
            base_delay=2.0,
            jitter=True,
            exponential_backoff=True,
        )

        # ค่าคงที่สำหรับการประเมิน (Global Constants)
        self.RERANK_THRESHOLD = RERANK_THRESHOLD
        self.MAX_EVI_STR_CAP = MAX_EVI_STR_CAP

        # -------------------------------------------------------
        # 5. Mapping & Evidence Persistence Setup
        # -------------------------------------------------------
        # 5.1 Evidence Mapping (สำหรับสืบค้นหลักฐานต่อเนื่อง)
        self.evidence_map = {}
        if clean_dt == EVIDENCE_DOC_TYPES.lower():
            self.evidence_map_path = evidence_map_path or get_evidence_mapping_file_path(
                tenant=self.config.tenant, 
                year=self.config.year, 
                enabler=self.enabler
            )
            self.evidence_map = self._load_evidence_map()
            self.logger.info(f"📊 Evidence Mapping: Loaded {len(self.evidence_map)} keys.")

        # 5.2 Document Mapping (ID -> Filename สำหรับทำ Report/Audit)
        loaded_map = document_map or {}
        if not loaded_map:
            is_evi_mode = (clean_dt == EVIDENCE_DOC_TYPES.lower())
            mapping_path = get_mapping_file_path(
                self.doc_type, 
                tenant=self.config.tenant, 
                year=self.config.year if is_evi_mode else None,
                enabler=self.enabler if is_evi_mode else None
            )

            if os.path.exists(mapping_path):
                try:
                    with open(mapping_path, 'r', encoding='utf-8') as f:
                        raw_data = json.load(f)
                    loaded_map = {k: v.get("file_name", k) for k, v in raw_data.items()}
                    self.logger.info(f"🎯 Document Mapping: Loaded {len(loaded_map)} entries.")
                except Exception as e:
                    self.logger.error(f"❌ Error parsing mapping file: {e}")

        self.doc_id_to_filename_map = loaded_map
        self.document_map = loaded_map
        self.temp_map_for_save = {}

        # -------------------------------------------------------
        # 6. Lazy Engine Initialization (VSM & LLM)
        # -------------------------------------------------------
        if self.llm is None: self._initialize_llm_if_none()
        if self.vectorstore_manager is None: self._initialize_vsm_if_none()

        # เชื่อมต่อ ID Mapping ภายใน VectorStore (ถ้าจำเป็น)
        if self.vectorstore_manager:
            try:
                self.vectorstore_manager._load_doc_id_mapping()
            except: pass

        # -------------------------------------------------------
        # 7. Function Registry (Pointers)
        # -------------------------------------------------------
        self.standard_audit_agent = evaluate_with_llm        # สำหรับ L3-L5 (เข้มงวด)
        self.foundation_coaching_agent = evaluate_with_llm_low_level # สำหรับ L1-L2 (แนะนำ)

        # ผูกฟังก์ชันหลักในการตัดสินใจ (The Router)
        self.assessment_router = self.evaluate_pdca
        
        # Registry อื่นๆ ตามเดิม
        self.rag_retriever = retrieve_context_with_filter
        # --- [PATCH v2026.1.17] State Management Initialization ---
        self.final_subcriteria_results = []
        self.total_stats = {}
        self.raw_llm_results = []
        self.level_details_map = {} # สำหรับเก็บสถานะ L1-L5 เพื่อทำ Gap Analysis

        # --- [CRITICAL PATCH v2026.2.20] Core Assessment States ---
        # ตัวแปรนี้สำคัญที่สุด เพราะ _is_previous_level_passed จะมาอ่านค่าจากที่นี่
        self.assessment_results_map = {} 
        
        # ตัวแปรเก็บหลักฐานสะสม เพื่อใช้ใน Step 2: Baseline Hydration
        self.previous_levels_evidence = [] 
        
        # เก็บ Mapping ของแต่ละ Level แยกตาม Sub-ID
        self.level_evidence_cache = {}
        self._cumulative_rules_cache: Dict[Tuple[str, int], Dict[str, Any]] = {}

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
        contextual_config: Dict = {},
        top_evidences: List[Dict[str, Any]] = []
    ) -> Dict[str, Any]:
        """
        [FULL REVISED v2026.1.24 - OPTIMIZED & ROBUST]
        - Robust PDCA key extraction (handle _Do_Score, _Check_Score, etc.)
        - Explicit 'is_passed' boost to 1.2 if LLM confirms pass
        - Rerank Safety Net threshold reduced to 0.82 for better rescue
        - Enhanced debug logging for key matching & rescue
        """
        log_prefix = f"Sub:{sub_id or '??'} L{level}"

        # 1. JSON Repair
        if isinstance(llm_output, tuple):
            llm_output = llm_output[0] if len(llm_output) > 0 else {}
        
        if isinstance(llm_output, str):
            try:
                cleaned = re.sub(r'```json\s*|\s*```', '', llm_output)
                cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned.strip())
                cleaned = cleaned.encode('utf-8', 'ignore').decode('utf-8')
                llm_output = json.loads(cleaned)
            except Exception as e:
                self.logger.error(f"❌ [JSON FAILED] {log_prefix}: {str(e)}")
                return {"is_passed": False, "score": 0.0, "reason": "AI Response Format Error"}

        if not isinstance(llm_output, dict):
            return {"is_passed": False, "score": 0.0, "reason": "Invalid Output Format"}

        # 2. Required Phases Setup
        required_phases = contextual_config.get("required_phases", [])
        if not required_phases:
            if level <= 3: required_phases = ["P", "D"]
            elif level == 4: required_phases = ["P", "D", "C"]
            else: required_phases = ["P", "D", "C", "A"]

        must_include_keywords = contextual_config.get("must_include_keywords", [])

        # 3. Robust PDCA Extraction + Smart Rescue
        pdca_results = {"P": 0.0, "D": 0.0, "C": 0.0, "A": 0.0}
        reason_content = str(llm_output.get('reason', '')).lower()

        for phase in ["P", "D", "C", "A"]:
            possible_keys = [
                f"{phase}_Plan_Score",
                f"{phase}_Do_Score",
                f"{phase}_Check_Score",
                f"{phase}_Act_Score",
                f"Extraction_{phase}_Score",
                f"score_{phase.lower()}",
                f"{phase}_Score"
            ]
            
            val = 0.0
            for k in possible_keys:
                if k in llm_output:
                    try:
                        val = float(llm_output[k])
                        self.logger.debug(f"🟢 [KEY-FOUND] {log_prefix} Phase {phase}: {k} = {val}")
                        break
                    except ValueError:
                        continue
            
            score = min(val, 2.0)

            # Smart Rescue by keywords
            phase_kws = contextual_config.get(f"{phase.lower()}_keywords", [])
            all_critical = list(set(phase_kws + must_include_keywords))
            extraction_text = str(llm_output.get(f"Extraction_{phase}", "")).lower()
            
            if score < 1.0 and any(kw.lower() in (reason_content + extraction_text) for kw in all_critical):
                old_score = score
                score = 1.5
                self.logger.info(f"🛡️ [RESCUE] {log_prefix} Phase {phase} boosted from {old_score} to {score} by keyword match.")

            pdca_results[phase] = score

        # 4. Adaptive Normalization + Explicit Pass Boost
        sum_required = sum(pdca_results[p] for p in required_phases)
        max_required = len(required_phases) * 2.0
        normalized_score = round((sum_required / max_required) * 2.0 if max_required > 0 else 0.0, 2)

        # 🟢 [ADJUSTED] Respect explicit 'is_passed' → force to 1.2 if LLM confirms pass
        explicit_pass = llm_output.get("is_passed", False)
        if explicit_pass and normalized_score < 1.2:
            normalized_score = 1.2
            self.logger.info(f"🛡️ [EXPLICIT-PASS BOOST] {log_prefix}: LLM says pass → set score to 1.2")

        # 5. Rerank Safety Net (Adjusted threshold)
        max_rerank = max([ev.get('relevance_score', 0.0) for ev in top_evidences]) if top_evidences else 0.0
        # 🟢 [ADJUSTED] Lower threshold from 0.88 → 0.82 for better rescue
        is_conflict = (normalized_score < 1.2) and (max_rerank > 0.82)

        if is_conflict:
            normalized_score = 1.2
            llm_output["is_force_pass"] = True
            self.logger.warning(f"🚨 [CONFLICT RESOLVED] {log_prefix} Force Passed due to high Rerank ({max_rerank:.2f})")

        # 6. Final Packaging
        is_passed = normalized_score >= 1.2
        missing_phases = [p for p in required_phases if pdca_results[p] < 1.0]
        
        coaching = str(llm_output.get("coaching_insight") or llm_output.get("ข้อเสนอแนะ") or "").strip()
        if missing_phases:
            coaching = f"⚠️ ขาดหลักฐานในเฟส: {', '.join(missing_phases)}. {coaching}"
        if is_conflict:
            coaching += " (หมายเหตุ: ผ่านด้วยเกณฑ์ความเกี่ยวข้องของเอกสารสูงเป็นพิเศษ)"

        return {
            "score": normalized_score,
            "is_passed": is_passed,
            "pdca_breakdown": pdca_results,
            "reason": llm_output.get("reason", "ไม่ระบุเหตุผล"),
            "summary_thai": llm_output.get("summary_thai") or llm_output.get("บทสรุป") or "",
            "coaching_insight": coaching,
            "required_phases": required_phases,
            "missing_phases": missing_phases,
            "needs_human_review": is_conflict or llm_output.get("consistency_check") == False,
            "is_force_pass": is_conflict
        }

    def _expand_context_with_neighbor_pages(self, top_evidences: List[Any], collection_name: str) -> List[Dict[str, Any]]:
        """
        [REVISED v2026.3.5] - ป้องกัน Type Mismatch & Standardize Output
        - แปลงทุกอย่าง (ทั้งของเก่าและของใหม่) ให้เป็น Dictionary มาตรฐาน
        - รักษา PDCA Rescue Tagging ให้แม่นยำเพื่อส่งต่อให้ _get_pdca_blocks_from_evidences
        """
        if not self.vectorstore_manager or not top_evidences:
            return top_evidences

        # 🛡️ STEP A: Standardize ข้อมูลดั้งเดิมให้เป็น Dict ทั้งหมดก่อน
        standardized_evidences = []
        for d in top_evidences:
            if hasattr(d, 'page_content'):
                standardized_evidences.append({
                    "text": d.page_content,
                    "page_content": d.page_content,
                    "metadata": getattr(d, 'metadata', {}),
                    "rerank_score": getattr(d, 'metadata', {}).get('rerank_score', 0.5) # Fallback score
                })
            elif isinstance(d, dict):
                standardized_evidences.append(deepcopy(d))
            else:
                continue

        expanded_evidences = list(standardized_evidences)
        seen_keys = set()
        added_pages = 0
        MAX_PAGES_PER_SUB = 12
        
        # Triggers... (คงเดิม)
        strategic_triggers = ["วิสัยทัศน์", "นโยบาย", "ทิศทาง", "เป้าหมายหลัก", "ยุทธศาสตร์", "สารจาก", "คำนำ"]
        check_triggers = ["ความพึงพอใจ", "คะแนน", "สรุปผล", "ตัวชี้วัด", "ผลประเมิน", "kpi", "score", "สรุปการดำเนินงาน"]
        action_triggers = ["ดำเนินการ", "จัดกิจกรรม", "อบรม", "จัดทำ", "ประชุมเมื่อวันที่", "บันทึกข้อความที่", "ประกาศฉบับที่"]

        for doc in standardized_evidences: # วนลูปจากตัวที่ Standardized แล้ว
            if added_pages >= MAX_PAGES_PER_SUB: break

            meta = doc.get('metadata', {})
            text = (doc.get('text', '') or "").lower()
            
            filename = meta.get("source") or meta.get("source_filename") or "Unknown File"
            doc_uuid = meta.get("stable_doc_uuid") or meta.get("doc_id")
            if not doc_uuid: continue

            # คำนวณเลขหน้า (คงเดิม)
            try:
                current_page_str = str(meta.get("page_label", meta.get("page", "1")))
                current_page = int("".join(filter(str.isdigit, current_page_str)))
            except: continue

            # Advanced Offset Strategy... (คงเดิม)
            offsets = []
            if any(k in text for k in strategic_triggers): offsets.extend([-1, 1, 2])
            if any(k in text for k in check_triggers): offsets.extend([-2, -1, 1, 2, 3])
            if any(k in text for k in action_triggers): offsets.extend([-1, 1])
            if not offsets: offsets = [1]

            for offset in sorted(list(set(offsets))):
                target_page = current_page + offset
                if target_page < 1 or target_page == current_page: continue
                
                cache_key = f"{doc_uuid}_{target_page}"
                if cache_key in seen_keys: continue
                seen_keys.add(cache_key)

                neighbor_chunks = self.vectorstore_manager.get_chunks_by_page(
                    collection_name=collection_name,
                    stable_doc_uuid=doc_uuid,
                    page_label=str(target_page)
                )

                if neighbor_chunks:
                    for nc in neighbor_chunks:
                        nc_text = nc.page_content.lower()
                        assigned_tag = "Support" if offset < 0 else "Detail"
                        
                        # 🏷️ Smart PDCA Rescue Tagging
                        if any(k in nc_text for k in check_triggers): assigned_tag = "Check" # เปลี่ยน Act/Check เป็น Check เพื่อให้ตรงกับมาตรฐาน
                        elif any(k in nc_text for k in action_triggers): assigned_tag = "Do"
                        elif any(k in nc_text for k in strategic_triggers): assigned_tag = "Plan"

                        # 🛡️ สร้าง Dict ที่สมบูรณ์แบบ
                        fixed_metadata = nc.metadata.copy() if hasattr(nc, 'metadata') else {}
                        fixed_metadata.update({
                            "stable_doc_uuid": doc_uuid,
                            "page_label": str(target_page),
                            "source": filename,
                            "is_supplemental": True,
                            "pdca_tag": assigned_tag 
                        })

                        expanded_evidences.append({
                            "text": nc.page_content, # เก็บเฉพาะเนื้อหาเพียวๆ (ไม่เติม Prefix เพื่อให้ LLM อ่านง่าย)
                            "page_content": nc.page_content,
                            "metadata": fixed_metadata,
                            "pdca_tag": assigned_tag,
                            "is_supplemental": True,
                            "rerank_score": float(doc.get('rerank_score', 0.5)) * 0.9,
                            "source": filename # เพิ่มคีย์ source ตรงๆ เพื่อความไว
                        })
                    added_pages += 1

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
        [REVISED v2026.1.18] - Robust Context Hydration
        - ดึงเนื้อหาเต็มจาก VectorStore เพื่อเป็น Baseline ให้เลเวลถัดไป
        - ปรับปรุงการ Match UUID ให้ครอบคลุมทุก Format (Strip dashes)
        """
        if getattr(self, 'is_parallel_all_mode', False):
            return {}

        collected = {}
        for key, ev_list in self.evidence_map.items():
            # กรองเฉพาะเลเวลที่ต่ำกว่าใน Sub-Criteria เดียวกัน
            if key.startswith(f"{sub_id}.L"):
                try:
                    level_num = int(key.split(".L")[-1])
                    if level_num < current_level:
                        collected[key] = ev_list
                except: continue

        if not collected: return {}

        # 1. รวบรวม Unique IDs (คัดกรองขยะ)
        stable_ids = set()
        for ev_list in collected.values():
            for ev in ev_list:
                sid = ev.get("stable_doc_uuid") or ev.get("doc_id")
                if sid and str(sid).lower() not in ["n/a", "none", ""]:
                    stable_ids.add(str(sid))

        if not stable_ids: return collected

        # 2. Bulk Hydration (Query ทีเดียวเพื่อประสิทธิภาพ)
        vsm = self.vectorstore_manager
        chunk_map = {}
        try:
            full_chunks = vsm.get_documents_by_id(list(stable_ids), self.doc_type, self.enabler)
            for chunk in full_chunks:
                m = chunk.metadata
                # เก็บ Map ทั้งแบบมีขีดและไม่มีขีดเพื่อความชัวร์
                keys = [str(m.get(k)) for k in ["stable_doc_uuid", "doc_id", "chunk_uuid"] if m.get(k)]
                for k in keys:
                    chunk_map[k] = {"text": chunk.page_content, "metadata": m}
                    chunk_map[k.replace("-", "")] = {"text": chunk.page_content, "metadata": m}
        except Exception as e:
            self.logger.error(f"❌ Hydration VSM Error: {e}")
            return collected

        # 3. Restoration Loop
        restored_count = 0
        for key, ev_list in collected.items():
            for ev in ev_list:
                sid = str(ev.get("stable_doc_uuid") or ev.get("doc_id") or "")
                data = chunk_map.get(sid) or chunk_map.get(sid.replace("-", ""))

                if data:
                    ev.update({
                        "text": data["text"],
                        "metadata": data["metadata"],
                        "is_baseline": True
                    })
                    restored_count += 1
                else:
                    ev["is_baseline"] = False
                
        self.logger.info(f"✅ Hydrated {restored_count} baseline chunks for {sub_id} L{current_level}")
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
        กรอง TEMP-, HASH-, และ Unknown ออกจาก evidence map ทั้งหมด
        พร้อมระบบป้องกัน 'str' object has no attribute 'get' และการซ่อมแซมข้อมูล
        """
        if not evidence_map or not isinstance(evidence_map, dict):
            return {}

        cleaned_map = {}
        total_removed = 0
        total_unknown_fixed = 0
        total_invalid_type = 0

        for key, entries in evidence_map.items():
            if not isinstance(entries, list):
                continue
                
            valid_entries = []
            for entry in entries:
                # 🛡️ Defense: ตรวจสอบประเภทข้อมูล
                if not isinstance(entry, dict):
                    if isinstance(entry, str) and entry.strip():
                        # พยายามกู้คืนข้อมูลถ้าเป็น string
                        entry = {"doc_id": entry, "filename": "Unknown", "relevance_score": 0.0}
                    else:
                        total_invalid_type += 1
                        continue

                doc_id = entry.get("doc_id")
                if doc_id is None:
                    total_removed += 1
                    continue
                
                doc_id_str = str(doc_id)

                # 1. กรอง TEMP- และ HASH-
                if doc_id_str.startswith("TEMP-") or doc_id_str.startswith("HASH-"):
                    total_removed += 1
                    continue

                # 2. กรอง Unknown
                if not doc_id_str or doc_id_str.lower() == "unknown":
                    total_removed += 1
                    continue

                # 3. จัดการ Filename
                filename = str(entry.get("filename", "")).strip()
                if not filename or filename.lower() in ["unknown", "none", "unknown_file.pdf", "n/a"]:
                    short_id = doc_id_str[:8]
                    entry["filename"] = f"เอกสารอ้างอิง_{short_id}.pdf"
                    total_unknown_fixed += 1
                else:
                    try:
                        entry["filename"] = os.path.basename(filename)
                    except:
                        entry["filename"] = filename

                valid_entries.append(entry)

            if valid_entries:
                cleaned_map[key] = valid_entries

        return cleaned_map

    
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

    def _save_evidence_map(self, map_to_save: Optional[Dict[str, Any]] = None):
        """
        [IRONCLAD FINAL v2026.1.18 — Ultra Safe Edition]
        - Load-Merge-Save Pattern (ไม่ overwrite ข้อมูลเก่าเด็ดขาด)
        - Atomic Write + FileLock + Tempfile
        - Backup (.bak) ก่อน save ทุกครั้ง
        - Validate + Clean ID เข้มงวด ป้องกันขยะ
        - Log การเปลี่ยนแปลงชัดเจน (จำนวน merge/อัปเดต)
        - Skip ถ้าไม่มีข้อมูลใหม่จริง ๆ
        """
        try:
            map_file_path = get_evidence_mapping_file_path(
                tenant=self.config.tenant,
                year=self.config.year,
                enabler=self.enabler
            )
        except Exception as e:
            self.logger.critical(f"[EVIDENCE] FATAL: ไม่สามารถกำหนด Path ได้: {e}")
            raise

        lock_path = map_file_path + ".lock"
        tmp_path = None
        backup_path = map_file_path + ".bak"  # Backup ก่อน save

        self.logger.info(f"[EVIDENCE] Preparing atomic save → {map_file_path}")

        try:
            os.makedirs(os.path.dirname(map_file_path), exist_ok=True)

            # Backup ไฟล์เดิมก่อน (สำคัญมาก!)
            if os.path.exists(map_file_path):
                try:
                    shutil.copy2(map_file_path, backup_path)
                    self.logger.debug(f"[EVIDENCE] Backup created: {backup_path}")
                except Exception as be:
                    self.logger.warning(f"[EVIDENCE] Backup failed (non-critical): {be}")

            with FileLock(lock_path, timeout=60):
                # STEP 1: โหลดข้อมูลเก่าจาก Disk (Base) — ต้องมีเสมอ
                final_map = self._load_evidence_map(is_for_merge=True) or {}
                self.logger.debug(f"[EVIDENCE] Loaded existing map: {len(final_map)} keys")

                # STEP 2: เตรียมข้อมูลใหม่ (Incoming)
                incoming = {}
                if map_to_save is not None:
                    # รองรับทั้ง Payload {"evidence_map": ...} และ Dict ตรง ๆ
                    if isinstance(map_to_save, dict) and "evidence_map" in map_to_save:
                        incoming = map_to_save["evidence_map"]
                    else:
                        incoming = map_to_save
                else:
                    # ถ้าไม่ส่งมา → ใช้จาก memory ของ Engine
                    incoming = getattr(self, 'evidence_map', {}) or {}

                # Skip ถ้าไม่มีข้อมูลใหม่จริง ๆ
                if not incoming:
                    self.logger.info("[EVIDENCE] No new data incoming. Skipping write.")
                    return

                # STEP 3: Merge + Validate + Clean
                merged_new = 0
                updated_existing = 0

                for key, new_entries in incoming.items():
                    if not isinstance(new_entries, list) or not new_entries:
                        continue

                    # ดึง entries เดิม (ถ้าไม่มี → สร้างใหม่)
                    current = final_map.setdefault(key, [])

                    # Index เดิมด้วย Clean ID
                    entry_index = {}
                    for e in current:
                        if not isinstance(e, dict):
                            continue
                        raw_id = e.get("chunk_uuid") or e.get("doc_id") or "N/A"
                        clean_id = str(raw_id).replace("-", "").lower()
                        if clean_id not in ["na", "n/a", "fallback", "none", ""]:
                            entry_index[clean_id] = e

                    # นำข้อมูลใหม่เข้า Merge
                    for new_e in new_entries:
                        if not isinstance(new_e, dict):
                            continue

                        raw_new_id = new_e.get("chunk_uuid") or new_e.get("doc_id") or "N/A"
                        clean_new_id = str(raw_new_id).replace("-", "").lower()

                        # Skip ขยะ
                        if clean_new_id in ["na", "n/a", "fallback", "none", ""]:
                            continue

                        new_score = new_e.get("relevance_score", 0.0)

                        if clean_new_id not in entry_index:
                            entry_index[clean_new_id] = new_e
                            merged_new += 1
                        else:
                            old_e = entry_index[clean_new_id]
                            old_score = old_e.get("relevance_score", 0.0)

                            # รักษา metadata เดิมถ้าขาด
                            if "page" not in new_e or new_e["page"] in ["N/A", None]:
                                new_e["page"] = old_e.get("page")
                            if "page_label" not in new_e:
                                new_e["page_label"] = old_e.get("page_label")

                            if new_score >= old_score:
                                entry_index[clean_new_id] = new_e
                                updated_existing += 1

                    # อัปเดตกลับ
                    final_map[key] = list(entry_index.values())

                # ถ้าไม่มีอะไร merge จริง ๆ → skip
                if merged_new == 0 and updated_existing == 0:
                    self.logger.info("[EVIDENCE] No unique new/updated entries. Skipping write.")
                    return

                # STEP 4: Clean + Sort
                final_map = self._clean_temp_entries(final_map)
                for key, entries in final_map.items():
                    entries.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)

                # STEP 5: Atomic Write
                with tempfile.NamedTemporaryFile(
                    mode='w', delete=False, encoding="utf-8", dir=os.path.dirname(map_file_path)
                ) as tmp_file:
                    cleaned_data = self._clean_map_for_json(final_map)
                    json.dump(cleaned_data, tmp_file, indent=4, ensure_ascii=False)
                    tmp_path = tmp_file.name

                shutil.move(tmp_path, map_file_path)
                tmp_path = None

                total_keys = len(final_map)
                total_items = sum(len(v) for v in final_map.values())
                self.logger.info(
                    f"✅ [EVIDENCE] SAVED SUCCESSFULLY! "
                    f"Keys: {total_keys} | Items: {total_items} | "
                    f"New: {merged_new} | Updated: {updated_existing}"
                )

        except Exception as e:
            self.logger.critical("[EVIDENCE] FATAL ERROR DURING ATOMIC SAVE")
            self.logger.exception(e)
            raise
        finally:
            # Cleanup
            if os.path.exists(lock_path):
                try:
                    os.unlink(lock_path)
                except:
                    pass
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass

    def merge_evidence_mappings(self, results_list: List[Any]) -> Dict[str, List[Dict]]:
        """
        [ULTIMATE STABLE v2026.1.18 — Key Mismatch Fix]
        - รองรับคีย์ 'evidence_sources' จากฟังก์ชันประเมินตัวใหม่
        - ทำ Deduplication โดยใช้ chunk_uuid/doc_id
        """
        merged_mapping = {}
        
        self.logger.info(f"🧬 Starting to merge evidence mappings from {len(results_list)} levels...")

        for item in results_list:
            if not item: continue
            
            temp_map = {}
            
            # 1. การดึงข้อมูล Mapping ออกมาตามโครงสร้างต่างๆ
            if isinstance(item, tuple) and len(item) == 2:
                temp_map = item[1]
            elif isinstance(item, dict):
                # [FIX] เพิ่มการเช็ค 'evidence_sources' เพื่อให้ดึงข้อมูลจาก _run_single_assessment ได้
                if 'evidence_sources' in item:
                    level_key = f"{item.get('sub_id', 'Unknown')}_L{item.get('level', 0)}"
                    temp_map = {level_key: item['evidence_sources']}
                elif 'temp_map_for_level' in item:
                    level_key = f"{item.get('sub_id', 'Unknown')}_L{item.get('level', 0)}"
                    data = item.get('temp_map_for_level', [])
                    temp_map = {level_key: data} if isinstance(data, list) else {}
                elif 'evidence_mapping' in item:
                    temp_map = item['evidence_mapping']
                else:
                    # ถ้าไม่มีคีย์พิเศษ ให้ถือว่าเป็น dict ของ mapping โดยตรง
                    temp_map = item

            if not temp_map or not isinstance(temp_map, dict):
                continue

            # 2. วน Loop รวมหลักฐานเข้ากับก้อนหลัก
            for level_key, evidence_list in temp_map.items():
                actual_list = []
                if isinstance(evidence_list, list):
                    actual_list = evidence_list
                elif isinstance(evidence_list, dict) and 'evidences' in evidence_list:
                    actual_list = evidence_list['evidences']
                else:
                    continue 
                
                if level_key not in merged_mapping:
                    merged_mapping[level_key] = []
                
                # เตรียม Set เพื่อตัดข้อมูลซ้ำ (Deduplication)
                existing_ids = set()
                for e in merged_mapping[level_key]:
                    eid = str(e.get('chunk_uuid') or e.get('doc_id') or "N/A").replace("-", "").lower()
                    existing_ids.add(eid)
                
                for new_ev in actual_list:
                    if not isinstance(new_ev, dict): continue
                    
                    raw_new_id = new_ev.get('chunk_uuid') or new_ev.get('doc_id') or "N/A"
                    clean_new_id = str(raw_new_id).replace("-", "").lower()

                    if clean_new_id in ["na", "n/a", "fallback", "none", "", "unknown"]:
                        continue

                    if clean_new_id not in existing_ids:
                        merged_mapping[level_key].append(new_ev)
                        existing_ids.add(clean_new_id)
        
        total_items = sum(len(v) for v in merged_mapping.values())
        self.logger.info(f"✅ Merging completed. Levels: {len(merged_mapping)} | Total items: {total_items}")
        
        return merged_mapping

    def _load_evidence_map(self, is_for_merge: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        """
        [REVISED v2026.1.16]
        - เพิ่ม cache ใน memory เพื่อลด I/O
        - Clean ข้อมูลที่โหลดมา (ลบ fallback/na)
        """
        if hasattr(self, '_evidence_cache') and self._evidence_cache is not None:
            return deepcopy(self._evidence_cache)

        try:
            path = get_evidence_mapping_file_path(
                tenant=self.config.tenant,
                year=self.config.year,
                enabler=self.enabler
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

            # Clean ข้อมูลที่โหลดมา
            for key in list(data.keys()):
                entries = data[key]
                cleaned = []
                for e in entries:
                    raw_id = e.get("chunk_uuid") or e.get("doc_id") or "N/A"
                    clean_id = str(raw_id).replace("-", "").lower()
                    if clean_id not in ["na", "n/a", "fallback", "none", ""]:
                        cleaned.append(e)
                data[key] = cleaned

            # Cache ใน memory
            self._evidence_cache = deepcopy(data)

            if not is_for_merge:
                total_items = sum(len(v) for v in data.values())
                self.logger.info(f"[EVIDENCE] Loaded evidence map: {len(data)} keys, {total_items} items from {path}")

            return data
        except Exception as e:
            self.logger.error(f"[EVIDENCE] Failed to load evidence map from {path}: {e}")
            return {}

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
             
        # for criteria_id, criteria_data in data.items():
        for criteria_id, criteria_data in data.get('criteria', {}).items():
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
                    discovery_result = vectorstore_manager.quick_search(
                        query=f"{sub_id} {' '.join(hints)}",
                        top_k=5
                    )
                    for chunk in discovery_result:
                        chunk["rerank_score"] = 0.85 # บูสต์ไฟล์ที่เจอจาก Keyword Rule
                        priority_chunks.append(chunk)
                except Exception as e:
                    self.logger.warning(f"⚠️ Quick search failed: {e}")

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

    def _save_level_evidences_and_calculate_strength(
        self,
        level_temp_map: List[Dict[str, Any]],
        sub_id: str,
        level: int,
        llm_result: Dict[str, Any],
        highest_rerank_score: float = 0.0
    ) -> float:
        """ 
        [IRONCLAD REVISE v2026.01.18] - ระบบบันทึกหลักฐานและคำนวณความเชื่อมั่น (Strength)
        - เพิ่มระบบ Deduplication ด้วย unique_key (Doc UUID + Chunk UUID)
        - ระบบ Normalize PDCA Tag อัตโนมัติ (Plan -> P, Detail -> D)
        - คำนวณ Strength Score โดยให้น้ำหนัก Rerank Score (60%) และ PDCA Coverage (40%)
        - รองรับการ Retagging กรณี Tag ไม่ชัดเจน (Semantic Fallback)
        """

        map_key = f"{sub_id}.L{level}"
        new_evidence_list: List[Dict[str, Any]] = []
        seen_ids = set()

        self.logger.info(f"💾 [EVI SAVE] Starting persist for {map_key} | Incoming: {len(level_temp_map)}")

        # 🚩 Configuration
        STANDARD_TAGS = {"P", "D", "C", "A"}
        PASS_STATUS = "PASS" if llm_result.get("is_passed", False) else "FAIL"

        for chunk in level_temp_map:
            if not chunk: continue

            # 1. ข้อมูลพื้นฐาน (รองรับทั้ง Dictionary และ LangChain Document)
            meta = chunk.get("metadata", {}) if isinstance(chunk, dict) else getattr(chunk, "metadata", {})
            text = chunk.get("text") or chunk.get("page_content") if isinstance(chunk, dict) else getattr(chunk, "page_content", "")
            
            if not text or not str(text).strip():
                continue

            # 2. Stable ID Generation (ป้องกัน Duplicate Chunks)
            # ใช้ SHA-256 จากข้อความหากไม่มี ID ประจำตัว เพื่อความแน่นอน
            c_uuid = str(chunk.get("chunk_uuid") or meta.get("chunk_uuid") or hashlib.sha256(text.encode()).hexdigest()[:16])
            d_uuid = str(chunk.get("stable_doc_uuid") or meta.get("stable_doc_uuid") or "doc-unknown")
            unique_key = f"{d_uuid}:{c_uuid}"
            
            if unique_key in seen_ids:
                continue
            seen_ids.add(unique_key)

            # 3. ✨ PDCA TAG NORMALIZATION & GUARD
            # แปลงค่าจากคำเต็มหรือค่าเสริมให้เป็น P, D, C, A มาตรฐาน
            raw_tag = chunk.get("pdca_tag") or meta.get("pdca_tag") or "Other"
            
            if isinstance(raw_tag, str):
                u_tag = raw_tag.strip().upper()
                if u_tag.startswith("P") or "PLAN" in u_tag: pdca_tag = "P"
                elif u_tag.startswith("D") or "DETAIL" in u_tag or "SUPPORT" in u_tag: pdca_tag = "D"
                elif u_tag.startswith("C") or "CHECK" in u_tag: pdca_tag = "C"
                elif u_tag.startswith("A") or "ACT" in u_tag: pdca_tag = "A"
                else: pdca_tag = "Other"
            else:
                pdca_tag = "Other"

            # Semantic Fallback: ถ้า Tag ยังเป็น Other ให้ลองใช้ Logic วิเคราะห์เนื้อหา (ถ้ามี)
            if pdca_tag == "Other" and hasattr(self, '_get_semantic_tag'):
                pdca_tag = self._get_semantic_tag(text, sub_id, level)

            # 4. สร้างหน่วยข้อมูล (Evidence Entry)
            source_raw = meta.get("source") or meta.get("source_filename") or "Unknown"
            entry = {
                "sub_id": sub_id,
                "level": level,
                "relevance_score": float(chunk.get("rerank_score") or chunk.get("score") or 0.5),
                "doc_id": d_uuid,
                "stable_doc_uuid": d_uuid,
                "chunk_uuid": c_uuid,
                "source": source_raw,
                "source_filename": os.path.basename(str(source_raw)),
                "page": str(meta.get("page_label") or meta.get("page") or "N/A"),
                "pdca_tag": pdca_tag,
                "text_preview": str(text)[:300].replace("\n", " ") + "...",
                "status": PASS_STATUS,
                "timestamp": datetime.now().isoformat(),
            }
            new_evidence_list.append(entry)

        # 5. การบันทึกลงหน่วยความจำและคำนวณคะแนนความแข็งแกร่ง (Strength)
        if new_evidence_list:
            # อัปเดต Memory Map (หลักฐานสะสม)
            if not hasattr(self, 'evidence_map'): self.evidence_map = {}
            self.evidence_map.setdefault(map_key, []).extend(deepcopy(new_evidence_list))
            
            # บันทึกสถานะการประเมินลงใน Map (เพื่อให้ฟังก์ชันตรวจสอบ Level อื่นๆ เรียกใช้ได้)
            if not hasattr(self, 'assessment_results_map'): self.assessment_results_map = {}
            self.assessment_results_map[map_key] = {
                "is_passed": llm_result.get("is_passed", False),
                "score": llm_result.get("score", 0.0),
                "strength": 0.0 # จะอัปเดตด้านล่าง
            }

            # 📊 STRENGTH CALCULATION LOGIC
            # 1. ตรวจสอบความครอบคลุม (Coverage): พบกี่หมวดใน P, D, C, A
            found_tags = {ev['pdca_tag'] for ev in new_evidence_list if ev['pdca_tag'] in STANDARD_TAGS}
            coverage_score = len(found_tags) / 4.0  # (0.0 - 1.0)
            
            # 2. ผสมคะแนน: Rerank Max (ความแม่นจาก Vector) 60% + PDCA Coverage (ความครบถ้วน) 40%
            final_strength = round((float(highest_rerank_score) * 0.6) + (coverage_score * 0.4), 2)
            
            # เก็บค่าความแข็งแกร่งกลับลงใน Result Map
            self.assessment_results_map[map_key]["strength"] = final_strength

            # สรุปผลทาง Log
            counts = {t: sum(1 for e in new_evidence_list if e['pdca_tag'] == t) for t in (list(STANDARD_TAGS) + ["Other"])}
            self.logger.info(
                f"✅ [SAVED] {map_key}: {len(new_evidence_list)} items | "
                f"P:{counts['P']} D:{counts['D']} C:{counts['C']} A:{counts['A']} | "
                f"Final Strength: {final_strength:.2f}"
            )
            return final_strength
            
        return 0.0
    
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
        current_year = 2568 
        
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

    def _get_level_constraint_prompt(self, sub_id: str, level: int, req_phases: list = None, spec_rule: str = None) -> str:
        """
        [ADAPTIVE AUDIT GUIDELINE v2026.1.19 - Concise & Stronger]
        - เน้น PDCA ชัดเจน + Substance over Form
        - ใช้ fallback ถ้าไม่มี req_phases
        - สั้นแต่ครบถ้วนสำหรับ LLM
        """
        required_phases = req_phases or self.get_rule_content(sub_id, level, "require_phase") or []
        specific_rule = spec_rule or self.get_rule_content(sub_id, level, "specific_contextual_rule") or ""

        phase_map = {
            "P": "Plan - วางแผนและกำหนดเป้าหมาย",
            "D": "Do - ปฏิบัติและขับเคลื่อนจริง",
            "C": "Check - ตรวจสอบ ติดตาม วัดผล",
            "A": "Act - ปรับปรุง พัฒนา ยกระดับ"
        }
        req_str = ", ".join(phase_map.get(p, p) for p in required_phases) if required_phases else "ตรวจสอบตามเกณฑ์มาตรฐาน"

        lines = [
            f"\n### แนวทางประเมิน {sub_id} Level {level} ###",
            f"เป้าหมายหลัก: {MATURITY_LEVEL_GOALS.get(level, 'ระดับที่เหมาะสม')}",
            f"มิติ PDCA ที่ต้องครบ: {req_str}",
            f"กฎเฉพาะ: {specific_rule}" if specific_rule else "",
            "\nกฎสำคัญ (ต้องยึดถือ):",
            "- Substance over Form: ให้ความสำคัญกับเนื้อหาและการกระทำจริงมากกว่าคำเฉพาะเจาะจง",
            "- Positive First: เน้นหาจุดแข็งและความสอดคล้องก่อน แล้วจึงระบุช่องว่าง",
            "- Coaching Mindset: ถ้าไม่ผ่าน ต้องระบุชัดเจนว่าขาดอะไร + แนะนำวิธีแก้ไขที่ทำได้จริง",
            "- Continuity: ระดับสูงต้องต่อยอดจากระดับต่ำกว่า (ถ้าพื้นฐานยังไม่แน่น ต้องระบุ)"
        ]

        return "\n".join(filter(None, lines))

    def _calculate_weighted_score(
        self, 
        highest_full_level: int, 
        weight: float, 
        level_details: Dict[str, Any] = None
    ) -> float:
        """
        [FIXED & ROBUST] คำนวณคะแนนถ่วงน้ำหนัก ป้องกันการหารด้วยศูนย์และค่าว่าง
        """
        # ดึงค่า MAX_LEVEL จาก config หรือใช้ Default เป็น 5
        max_lv = getattr(self.config, 'max_level', 5) 
        if max_lv <= 0: max_lv = 5 
        
        # ป้องกัน weight เป็น None หรือ 0
        safe_weight = float(weight) if weight else 4.0

        # 1. คำนวณ Base Level
        base_level = float(max(0, min(highest_full_level, max_lv)))
        
        # 2. คำนวณ Partial Score (โหมดสะสมคะแนนจาก Level ที่กำลังทำอยู่)
        partial_contribution = 0.0
        # ใช้ค่าจาก global_vars หรือ config
        mode = getattr(self, 'scoring_mode', 'PARTIAL_PDCA') 

        if mode == 'PARTIAL_PDCA' and level_details:
            next_lv_idx = int(base_level + 1)
            if next_lv_idx <= max_lv:
                next_level_str = str(next_lv_idx)
                lv_data = level_details.get(next_level_str, {})
                
                # ตรวจสอบ pdca_breakdown
                pdca = lv_data.get('pdca_breakdown')
                if isinstance(pdca, dict) and pdca:
                    pdca_values = [float(v) for v in pdca.values() if v is not None]
                    if pdca_values:
                        # (P+D+C+A)/4 -> ให้คะแนนเพิ่มสูงสุดไม่เกิน 1.0 ระดับ
                        partial_contribution = sum(pdca_values) / len(pdca_values)
                        self.logger.debug(f"➕ [PARTIAL] L{next_level_str} adds {partial_contribution:.2f}")

        # 3. รวมคะแนน Maturity (Effective Level)
        effective_level = base_level + partial_contribution
        
        # 4. คำนวณ Ratio (คะแนนที่ได้ / คะแนนเต็ม)
        base_ratio = effective_level / max_lv
        
        # 5. สรุปคะแนนถ่วงน้ำหนัก
        scaled_score = base_ratio * safe_weight
        
        # Boost Logic (ถ้าต้องการ)
        if mode == 'STEP_LADDER' and base_level >= max_lv - 1:
            scaled_score = min(scaled_score * 1.1, safe_weight)
        
        final_score = round(scaled_score, 4)
        
        self.logger.info(f"📊 [WEIGHT CALC] Mode: {mode} | Eff: {effective_level:.2f}/{max_lv} | Score: {final_score}/{safe_weight}")
        
        return final_score
    
    def _calculate_overall_stats(self, target_sub_id: str):
        """
        [AUDIT-READY v2026.1.24] — Bottleneck + Weighted + Force-Max Safe
        - Crash-proof: รองรับ results ว่าง/None ไม่ให้ .get() บน None
        - Audit-friendly: log ชัด + reject ถ้า evidence น้อยเกิน (no dummy accept)
        - Force-max: ใช้ max ถ้ามี force-pass หรือผ่าน L3+ ≥50%
        - Weighted avg จริง + analytics เพิ่ม evidence coverage
        """
        results = self.final_subcriteria_results or []

        if not results:
            self.logger.critical("[AUDIT CRITICAL] No subcriteria results found - Setting L0 with warning")
            self.total_stats = {
                "overall_max_level": 0,
                "overall_min_level": 0,
                "overall_level_label": "L0",
                "overall_avg_score": 0.0,
                "total_weighted_score": 0.0,
                "total_weight": 0.0,
                "force_pass_count": 0,
                "high_level_pass_count": 0,
                "use_max_override": False,
                "audit_note": "No valid subcriteria results - Possible retrieval failure",
                "analytics": {"sub_details": []}
            }
            return

        passed_levels = []
        sub_details = []
        total_weighted_sum = 0.0
        total_weight = 0.0
        force_pass_count = 0
        high_level_pass_count = 0  # นับ sub ที่ผ่าน L3+
        use_max_override = False

        for r in results:
            sub_id = r.get('sub_id', 'Unknown')
            
            # 1. Flexible Level Details Access
            details_map = r.get('level_details', {})
            if not details_map and '0' in r:
                details_map = r.get('0', {}).get('level_details', {})

            # 2. Step-Ladder Maturity Scan
            current_maturity_lvl = 0
            for l_idx in range(1, 6):
                lv_data = details_map.get(str(l_idx), {})
                is_passed = lv_data.get('is_passed', False)
                is_force = lv_data.get('is_force_pass', False)
                
                if is_passed or is_force:
                    current_maturity_lvl = l_idx
                    if is_force:
                        force_pass_count += 1
                    if current_maturity_lvl >= 3:
                        high_level_pass_count += 1
                else:
                    break

            # 3. Weighted Score
            weight = float(r.get('weight', 4.0))
            total_weight += weight
            
            if hasattr(self, '_calculate_weighted_score'):
                sub_score = self._calculate_weighted_score(
                    highest_full_level=current_maturity_lvl,
                    weight=weight,
                    level_details=details_map
                )
            else:
                sub_score = float(current_maturity_lvl) * (weight / 5.0 if weight > 0 else 0)

            total_weighted_sum += sub_score

            # Update back to result
            r['highest_full_level'] = current_maturity_lvl
            r['weighted_score'] = round(sub_score, 2)
            r['is_passed'] = (current_maturity_lvl >= 1)

            passed_levels.append(current_maturity_lvl)
            
            sub_details.append({
                "sub_id": sub_id,
                "maturity": current_maturity_lvl,
                "score": round(sub_score, 2),
                "weight": weight,
                "is_force_pass": any(lv_data.get('is_force_pass', False) for lv_data in details_map.values()),
                "evidence_count": len(details_map)  # เพิ่ม audit info
            })

        # 4. Final Aggregation
        num_subs = len(results)
        avg_score = total_weighted_sum / total_weight if total_weight > 0 else 0.0
        
        overall_min_maturity = min(passed_levels) if passed_levels else 0
        overall_max_maturity = max(passed_levels) if passed_levels else 0

        # Decide label: min default, max ถ้ามี force-pass หรือผ่าน L3+ ≥50%
        final_label_level = overall_min_maturity
        use_max_override = force_pass_count > 0 or (high_level_pass_count / num_subs >= 0.5 if num_subs > 0 else False)
        if use_max_override:
            final_label_level = overall_max_maturity
            self.logger.info(f"[STATS OVERRIDE] Using MAX level L{final_label_level} (force-pass: {force_pass_count}, high-level pass: {high_level_pass_count}/{num_subs})")

        # Audit Note: ถ้า evidence น้อยเกิน ให้เตือน
        audit_note = "All subcriteria processed" 
        if any(d["evidence_count"] < 3 for d in sub_details):
            audit_note += " - Warning: Some subcriteria have low evidence count (<3) - Audit may require manual review"

        self.total_stats = {
            "overall_max_level": int(overall_max_maturity),
            "overall_min_level": int(overall_min_maturity),
            "overall_level_label": f"L{int(final_label_level)}",
            "overall_avg_score": round(avg_score, 2),
            "total_weighted_score": round(total_weighted_sum, 2),
            "total_weight": round(total_weight, 2),
            "force_pass_count": force_pass_count,
            "high_level_pass_count": high_level_pass_count,
            "use_max_override": use_max_override,
            "audit_note": audit_note,
            
            "total_sub_assessed": num_subs,
            "analytics": {
                "sub_details": sub_details,
                "passed_levels_map": passed_levels,
                "assessed_at": datetime.now().isoformat()
            }
        }

        self.logger.info(
            f"✅ [STATS SUCCESS] Overall: {self.total_stats['overall_level_label']} | "
            f"Avg Score: {self.total_stats['overall_avg_score']} | "
            f"Max/Min: L{overall_max_maturity}/L{overall_min_maturity} | "
            f"Force-Pass: {force_pass_count} | High-Level Pass: {high_level_pass_count}/{num_subs} | "
            f"Audit Note: {audit_note}"
        )

    def _export_results(self, results_data: Any, sub_criteria_id: str, **kwargs) -> str:
        """
        [ULTIMATE EXPORTER v2026.EXPORT.5 - STRATEGIC INTEGRATED]
        ฟังก์ชันส่งออกผลลัพธ์ที่รวมคะแนน, หลักฐาน และแผนยุทธศาสตร์ (Tier-3) ไว้ในที่เดียว
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            record_id = kwargs.get("record_id", getattr(self, "current_record_id", f"auto_{timestamp}"))
            tenant = getattr(self.config, 'tenant', 'unknown')
            year = getattr(self.config, 'year', 'unknown')
            enabler = getattr(self, 'enabler', 'unknown').upper()

            # 1. 🧩 Data Consolidation (รวมและตรวจสอบความถูกต้องของข้อมูล)
            if results_data is None:
                results_data = getattr(self, 'final_subcriteria_results', [])
            
            # กรณีส่งมาเป็น Dictionary ตัวเดียว ให้ห่อเป็น List
            if isinstance(results_data, dict):
                results_data = [results_data]
            
            if not results_data:
                self.logger.warning(f"⚠️ [EXPORT] ไม่พบข้อมูลสำหรับ {sub_criteria_id} - ยกเลิกการ Export")
                return ""

            # 2. 📊 Summary Calculation (สรุปผลคะแนนและความสำเร็จ)
            valid_results = [r for r in results_data if isinstance(r, dict)]
            
            # คำนวณ Maturity Level สูงสุดและคะแนนรวมถ่วงน้ำหนัก
            highest_lvl = max([int(r.get('highest_full_level', 0)) for r in valid_results]) if valid_results else 0
            total_weighted = sum([float(r.get('weighted_score', 0.0)) for r in valid_results])

            # 3. 🛡️ Robust Evidence Mapping (Audit Trail สำหรับผู้ตรวจสอบ)
            # ดึง Master Map ที่สะสมมาจาก Worker ทุกตัว
            master_map = getattr(self, 'evidence_map', {})
            processed_evidence = {}
            
            for lv_key, v_list in master_map.items():
                if not v_list or not isinstance(v_list, list): 
                    continue
                
                try:
                    # เลือกหลักฐานที่มีค่า Rerank Score สูงสุดมาเป็นตัวแทนของ Level นั้นๆ
                    sorted_ev = sorted(
                        [ev for ev in v_list if isinstance(ev, dict)], 
                        key=lambda x: x.get('relevance_score', x.get('rerank_score', 0)), 
                        reverse=True
                    )
                    top_ev = sorted_ev[0] if sorted_ev else None
                    
                    if top_ev:
                        # ตรวจสอบชื่อไฟล์จาก document_map เพื่อความแม่นยำ
                        doc_id = top_ev.get("doc_id")
                        filename = self.document_map.get(doc_id) if hasattr(self, 'document_map') else None
                        filename = filename or top_ev.get("source") or top_ev.get("file_name") or "Unknown_Source"

                        processed_evidence[str(lv_key)] = {
                            "file": filename,
                            "page": top_ev.get("page", "N/A"),
                            "tag": str(top_ev.get("pdca_tag", "OTHER")).upper(),
                            "confidence": round(float(top_ev.get("relevance_score", top_ev.get("rerank_score", 0))), 4),
                            "content_snippet": str(top_ev.get("content", ""))[:200] + "..." # เก็บตัวอย่างเนื้อหา
                        }
                except Exception as ev_err:
                    self.logger.debug(f"Skip processing evidence level {lv_key}: {ev_err}")

            # 4. 📝 Construct Final Payload (โครงสร้างข้อมูลระดับโปรดักชัน)
            payload = {
                "metadata": {
                    "record_id": record_id,
                    "tenant": tenant,
                    "year": year,
                    "enabler": enabler,
                    "engine_version": "SEAM-ENGINE-v2026.3.26",
                    "exported_at": datetime.now().isoformat(),
                    "scoring_mode": "Step-Ladder Maturity"
                },
                "result_summary": {
                    "maturity_level": f"L{highest_lvl}",
                    "is_passed": highest_lvl >= 1,
                    "total_weighted_score": round(total_weighted, 4),
                    "evidence_used_count": len(processed_evidence),
                    "evaluated_sub_count": len(valid_results),
                    "status": "COMPLETED"
                },
                "sub_criteria_details": valid_results,  # ข้อมูลวิเคราะห์ราย Level และ Tier-2 Action Plans
                "evidence_audit_trail": processed_evidence, # ร่องรอยหลักฐานสำหรับการตรวจสอบย้อนกลับ
                "strategic_synthesis": getattr(self, 'master_roadmap_data', {
                    "status": "PENDING",
                    "overall_strategy": "แผนภาพรวมกำลังถูกประมวลผลหรือข้อมูลไม่เพียงพอ"
                }) # 🎯 ข้อมูล Tier-3 Strategic Roadmap
            }

            # 5. 💾 Smart Path & Persistence (ระบบบันทึกไฟล์อัจฉริยะ)
            filename = f"REPORT_{enabler}_{sub_criteria_id}_{timestamp}.json"
            
            try:
                # พยายามบันทึกลงใน Path มาตรฐานตามปีและหัวข้อ
                export_path = get_assessment_export_file_path(
                    tenant=tenant, year=year, enabler=enabler.lower(),
                    suffix=f"{sub_criteria_id}_{timestamp}", ext="json"
                )
            except:
                # Fallback: หากระบบ Path มีปัญหา ให้สร้างโฟลเดอร์ชั่วคราว
                out_dir = os.path.join("exports", str(tenant), str(year), enabler.lower())
                os.makedirs(out_dir, exist_ok=True)
                export_path = os.path.join(out_dir, filename)

            # 6. 🖊️ Write JSON File
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            self.logger.info(f"💾 [EXPORT SUCCESS] รายงานฉบับสมบูรณ์ถูกสร้างแล้วที่: {export_path}")
            return export_path

        except Exception as e:
            self.logger.error(f"❌ [EXPORT FAILED] เกิดข้อผิดพลาดวิกฤต: {str(e)}", exc_info=True)
            return ""
        
    def _robust_hydrate_documents_for_priority_chunks(
        self,
        chunks_to_hydrate: List[Dict],
        vsm: Optional['VectorStoreManager'],
        current_sub_id: Optional[str] = None,
        level: Optional[int] = None
    ) -> List[Dict]:
        """
        [ULTIMATE HYDRATION v2026.12]
        - ดึง Full Text สำหรับ Priority Chunks เพื่อให้ LLM เห็นบริบทครบถ้วน
        - เปลี่ยนไปใช้ LLM-based Tagging เพื่อความแม่นยำสูงสุดในทุกระดับ (L1-L5)
        - เพิ่มระบบ Boost Score เพื่อให้ AI ให้ความสำคัญกับหลักฐานกลุ่มนี้
        """
        from collections import defaultdict
        
        active_sub_id = current_sub_id or getattr(self, 'sub_id', 'unknown')
        if not chunks_to_hydrate:
            self.logger.debug(f"ℹ️ [HYDRATION] No chunks to hydrate for {active_sub_id} L{level}")
            return []

        # 1. 🏷️ Helper: จัดหมวดหมู่ด้วย LLM (ใช้ตัวเดียวกับ Core Assessment)
        def _safe_classify(text: str, filename: str = "") -> str:
            try:
                # เรียกใช้ LLM Tagging ที่เราปรับจูนไว้ (Self-contained within class)
                tag = self._get_semantic_tag(
                    text=text, 
                    sub_id=active_sub_id, 
                    level=level or 1,
                    filename=filename
                )
                return tag if tag in {"P", "D", "C", "A"} else "Other"
            except Exception as e:
                self.logger.warning(f"⚠️ PDCA classify failed in hydration: {e}")
                return "Other"

        # 2. 📏 Helper: มาตรฐานข้อมูลและ Scoring Boost
        def _standardize_chunk(chunk: Dict, score: float):
            chunk.setdefault("is_baseline", True)
            text = chunk.get("text", "").strip()
            meta = chunk.get("metadata", {})
            
            if text:
                fname = os.path.basename(str(meta.get("source") or meta.get("file_name") or "Unknown"))
                # ✨ ใช้ LLM Tagging ตรงนี้เลย
                chunk["pdca_tag"] = _safe_classify(text, filename=fname)
                
                # บังคับ Boost คะแนนให้สูง เพื่อให้ชนะ Chunk ทั่วไปในขั้นตอนการเรียงลำดับ
                chunk["rerank_score"] = max(float(chunk.get("rerank_score", 0.0)), score)
                chunk["score"] = max(float(chunk.get("score", 0.0)), score)
            return chunk

        # 3. 🔑 เตรียมความพร้อม IDs
        stable_ids = {
            sid for c in chunks_to_hydrate
            if (sid := (c.get("stable_doc_uuid") or c.get("doc_id") or c.get("chunk_uuid")))
        }

        if not stable_ids or not vsm:
            boosted = [_standardize_chunk(c.copy(), 0.9) for c in chunks_to_hydrate]
            return self._guarantee_text_key(boosted)

        # 4. 🛰️ Fetch Full Documents (Hydration Process)
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
            self.logger.error(f"❌ [HYDRATION] VSM Fetch Error: {e}")
            return self._guarantee_text_key([_standardize_chunk(c.copy(), 0.9) for c in chunks_to_hydrate])

        # 5. 💧 Hydrate & Dedup
        hydrated_priority_docs = []
        seen_signatures = set()
        SAFE_META_KEYS = {"source", "file_name", "page", "page_label", "page_number"}

        for chunk in chunks_to_hydrate:
            new_chunk = chunk.copy()
            sid = new_chunk.get("stable_doc_uuid") or new_chunk.get("doc_id")

            hydrated = False
            if sid and sid in stable_id_map:
                # ดึง Full Text มาทับ Snippet สั้นๆ
                best_match = stable_id_map[sid][0]
                new_chunk["text"] = best_match["text"]
                meta = best_match.get("metadata", {})
                new_chunk.update({k: v for k, v in meta.items() if k in SAFE_META_KEYS})
                hydrated = True

            # ทำ Standardize + Tagging (Score 1.0 ถ้าดึงเต็มสำเร็จ / 0.85 ถ้าใช้ของเดิม)
            new_chunk = _standardize_chunk(new_chunk, score=1.0 if hydrated else 0.85)

            # Check Signature กันซ้ำ
            sig = (sid, new_chunk.get("chunk_uuid"), new_chunk.get("text", "")[:100])
            if sig not in seen_signatures:
                seen_signatures.add(sig)
                hydrated_priority_docs.append(new_chunk)

        self.logger.info(f"✅ [HYDRATION] Complete: {len(hydrated_priority_docs)} priority chunks ready.")
        return self._guarantee_text_key(hydrated_priority_docs)

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
    
    def evaluate_pdca(
        self,
        pdca_blocks: Dict[str, Any],
        sub_id: str,
        level: int,
        audit_confidence: Any
    ) -> Dict[str, Any]:
        """
        [ULTIMATE ROUTING ENGINE v2026.3.26]
        ระบบตัดสินใจเลือก LLM Agent ตามระดับความสำคัญของ Maturity (Coaching vs Audit)
        และจัดการโครงสร้างข้อมูลให้เหมาะสมก่อนส่งให้ LLM
        """
        
        # 1. ดึงข้อมูลพื้นฐานจาก Rubric เพื่อสร้างบริบทให้ AI
        criteria_info = self.rubric.get(sub_id, {})
        sub_name = criteria_info.get("name", sub_id)
        statement = criteria_info.get("statement", "ไม่มีรายละเอียดเกณฑ์การประเมิน")
        log_prefix = f"🧠 [{sub_id}-L{level}]"

        # 2. จัดการ Audit Confidence (Type Guard)
        conf_score = 0.0
        if isinstance(audit_confidence, dict):
            conf_score = float(audit_confidence.get("coverage_ratio", 0.0))
        else:
            conf_score = float(audit_confidence or 0.0)

        # 3. เตรียม PDCA Context (แยกหมวดหมู่ชัดเจนเพื่อช่วย AI สกัดหลักฐาน) 
        pdca_summary_list = []
        for tag in ["P", "D", "C", "A"]:
            content = pdca_blocks.get(tag, "")
            if content:
                pdca_summary_list.append(f"--- {tag} PHASE EVIDENCE ---\n{content}")
            else:
                pdca_summary_list.append(f"--- {tag} PHASE EVIDENCE ---\n(ไม่พบหลักฐานในหมวดนี้)")
        
        pdca_string_context = "\n\n".join(pdca_summary_list)

        # 4. ดึงกฎสะสม (Cumulative Rules) ที่คำนวณไว้แล้ว
        rules = self.get_cumulative_rules_cached(sub_id, level)

        # 5. รวบรวม Parameters ส่วนเกินที่จะส่งเป็น Keyword Arguments (**kwargs)
        # หมายเหตุ: เราจะไม่ใส่ sub_id, level, sub_name, statement ในนี้ 
        # เพราะจะส่งเป็น Positional Arguments ในขั้นตอนถัดไป
        extra_kwargs = {
            "pdca_context": pdca_string_context, 
            "context": str(pdca_blocks),         # ส่งข้อมูลดิบเผื่อไว้
            "required_phases": rules.get("required_phases", []),
            "specific_contextual_rule": rules.get("all_instructions", "พิจารณาตามเกณฑ์มาตรฐาน"),
            "llm_executor": self.llm,
            "enabler_full_name": self.config.enabler,
            "enabler_code": self.enabler,
            "plan_keywords": rules.get("plan_keywords", []),
            "confidence_reason": f"Coverage Score: {conf_score:.2f}",
            "ai_confidence": "HIGH" if conf_score >= 0.7 else "MEDIUM"
        }

        # ---------------------------------------------------------------------
        # 6. ROUTING LOGIC (Strategic Separation)
        # ---------------------------------------------------------------------
        
        
        try:
            # 🎯 CASE A: [STRATEGIC LEVEL] Level 3 ขึ้นไป (Audit Mode)
            if level >= 3:
                self.logger.info(f"{log_prefix} ROUTE → Standard Audit Agent (Strict Mode)")
                return self.standard_audit_agent(
                    sub_criteria_name=sub_name, # arg 1
                    level=level,                # arg 2
                    statement_text=statement,   # arg 3
                    sub_id=sub_id,              # arg 4
                    **extra_kwargs
                )

            # 🎯 CASE B: [FOUNDATION LEVEL] Level 1-2 (Coaching Mode)
            else:
                self.logger.info(f"{log_prefix} ROUTE → Foundation Coaching Agent (Helpful Mode)")
                # ในโหมด Coaching เราอาจปรับ AI Confidence ให้ยืดหยุ่นขึ้น
                extra_kwargs["ai_confidence"] = "MEDIUM" 
                
                return self.foundation_coaching_agent(
                    sub_criteria_name=sub_name, # arg 1
                    level=level,                # arg 2
                    statement_text=statement,   # arg 3
                    sub_id=sub_id,              # arg 4
                    **extra_kwargs
                )
                
        except Exception as e:
            self.logger.error(f"🛑 [ROUTING-ERROR] {log_prefix} Failure: {str(e)}")
            return {
                "is_passed": False,
                "score": 0.0,
                "reason": f"Routing System Error: {str(e)}",
                "is_error": True
            }

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
        [ULTIMATE HYDRATED v2026.3.26 - AUDIT SAFE]

        Responsibilities:
        - Merge new + baseline evidences
        - Classify into PDCA with multi-layer guards
        - Prevent Zero-Chunks illusion
        - Preserve full audit trace (baseline / forced / relevance)

        Design Principles:
        - L1–L3 : Evidence existence > perfection (allow forced)
        - L4–L5 : Substance over form (no forced guessing)
        """

        pdca_groups = defaultdict(list)
        seen_texts = set()

        # ------------------------------------------------------------------
        # STEP 1: Merge evidences (Foundation Flex)
        # ------------------------------------------------------------------
        all_candidate_evidences = (evidences or []) + (baseline_evidences or [])
        self.logger.info(
            f"🏷️ [PDCA] Candidates: {len(all_candidate_evidences)} "
            f"(new={len(evidences or [])}, baseline={len(baseline_evidences or [])}, L{level})"
        )

        # ------------------------------------------------------------------
        # STEP 2: Iterate & classify
        # ------------------------------------------------------------------
        for idx, chunk in enumerate(all_candidate_evidences, start=1):
            txt = (chunk.get("text") or "").strip()
            if not txt:
                continue

            txt_key = txt.lower()
            if txt_key in seen_texts:
                continue
            seen_texts.add(txt_key)

            # ---- Metadata recovery (Audit Trace) ----
            meta = chunk.get("metadata", {}) or {}
            filename = (
                chunk.get("source_filename")
                or meta.get("source_filename")
                or "Unknown_File"
            )
            page = meta.get("page_label") or meta.get("page") or "N/A"

            is_baseline = chunk.get("source") == "BASELINE" or chunk.get("is_baseline", False)
            baseline_level = chunk.get("baseline_level") or (level - 1 if is_baseline else level)

            prefix = f"[BASELINE-L{baseline_level}] " if is_baseline else ""
            source_display = f"{prefix}{filename} (P.{page})"

            # ---- STEP 3: PDCA classification ----
            final_tag = None
            tag_source = None
            is_forced = False

            # 3.1 Heuristic
            if hasattr(self, "_get_heuristic_pdca_tag"):
                final_tag = self._get_heuristic_pdca_tag(txt, level)
                if final_tag:
                    tag_source = "Heuristic"

            # 3.2 Keyword
            if not final_tag:
                txt_lower = txt.lower()
                p_kws = ['แผน', 'นโยบาย', 'ยุทธศาสตร์', 'เป้าหมาย', 'master plan', 'policy', 'แผนปฏิบัติการ']
                d_kws = ['ดำเนินการ', 'กิจกรรม', 'ปฏิบัติ', 'ประชุม', 'บันทึก', 'implement', 'การดำเนินงาน']
                c_kws = ['ติดตาม', 'ประเมิน', 'รายงาน', 'ผล', 'audit', 'kpi', 'ตัวชี้วัด', 'monitor', 'สรุปผล']
                a_kws = ['ปรับปรุง', 'ทบทวน', 'lesson learned', 'พัฒนา', 'นวัตกรรม', 'improve', 'review', 'บทเรียน']

                if any(k in txt_lower for k in p_kws):
                    final_tag, tag_source = "P", "Keyword"
                elif any(k in txt_lower for k in d_kws):
                    final_tag, tag_source = "D", "Keyword"
                elif any(k in txt_lower for k in c_kws):
                    final_tag, tag_source = "C", "Keyword"
                elif any(k in txt_lower for k in a_kws):
                    final_tag, tag_source = "A", "Keyword"

            # 3.3 Semantic (AI)
            if not final_tag:
                try:
                    tag = self._get_semantic_tag(txt, sub_id, level, filename)
                    if tag in {"P", "D", "C", "A"}:
                        final_tag, tag_source = tag, "AI-Semantic"
                    else:
                        final_tag = "Other"
                except Exception:
                    final_tag, tag_source = "Other", "AI-Fail"

            # ---- STEP 4: Upper Guard & Forced Logic ----
            if final_tag == "Other":
                if level >= 4:
                    # 🚩 L4–L5: ห้ามเดา → ตัดทิ้งอย่างโปร่งใส
                    self.logger.debug(
                        f"🚫 [PDCA] Excluded insufficient evidence (L{level}): {source_display}"
                    )
                    continue
                else:
                    # L1–L3: อนุญาต Forced เพื่อหา existence
                    is_forced = True
                    if level <= 2:
                        final_tag = "P" if idx % 2 == 0 else "D"
                    else:  # level == 3
                        final_tag = "C" if idx % 3 == 0 else "D"
                    tag_source = f"Forced-L{level}"

            # ---- STEP 5: Collect chunk ----
            pdca_groups[final_tag].append({
                "text": txt,
                "pdca_tag": final_tag,
                "tag_source": tag_source,
                "source_display": source_display,
                "is_baseline": is_baseline,
                "is_forced": is_forced,
                "relevance": chunk.get("final_relevance_score") or 0
            })

        # ------------------------------------------------------------------
        # STEP 6: Build final PDCA blocks (consistent ranking)
        # ------------------------------------------------------------------
        blocks: Dict[str, Any] = {"sources": {}, "actual_counts": {}}

        for tag in ["P", "D", "C", "A"]:
            ranked_chunks = sorted(
                pdca_groups.get(tag, []),
                key=lambda x: (
                    x["is_baseline"],   # new before baseline
                    x["is_forced"],     # real before forced
                    -x["relevance"]     # higher relevance first
                )
            )[:5]

            if ranked_chunks:
                blocks[tag] = "\n\n".join([
                    f"[{c['source_display']} | {c['tag_source']}{' ⚠️FORCED' if c['is_forced'] else ''}]\n"
                    f"{c['text'][:500]}"
                    for c in ranked_chunks
                ])
            else:
                blocks[tag] = f"[ไม่พบหลักฐานชัดเจนในหมวด {tag}]"

            # ---- Persistence & Audit ----
            blocks["sources"][tag] = [c["source_display"] for c in ranked_chunks]
            blocks["actual_counts"][tag] = len([
                c for c in ranked_chunks if not c["is_forced"]
            ])

        return blocks

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
        [REVISED v2026] ปรับแต่ง Metadata ให้พร้อมสำหรับการประเมินและ Export
        - แก้ไขปัญหาคีย์กระจัดกระจาย (Flattened vs Nested)
        - ประกันความปลอดภัยของชนิดข้อมูล (Type Safety)
        """
        for ev in evidence_list:
            if not isinstance(ev, dict):
                continue
                
            # 1. เข้าถึง Metadata (รองรับทั้ง Langchain Doc และ Dict)
            meta = ev.get("metadata", {})
            if not isinstance(meta, dict): meta = {}
            
            # 2. ปรับ Source (ดึงจากจุดที่ลึกที่สุดก่อน)
            raw_source = (
                meta.get("source_filename") or 
                meta.get("file_name") or 
                ev.get("source") or 
                meta.get("source") or 
                "Unknown_File"
            )
            ev["source"] = os.path.basename(str(raw_source))
            ev["source_filename"] = ev["source"] # Sync ทั้งสองคีย์
            
            # 3. ปรับ Page (ป้องกันปัญหาเลข 0 หรือ None)
            raw_page = (
                meta.get("page_label") or 
                meta.get("page") or 
                meta.get("page_number") or 
                ev.get("page") or 
                "N/A"
            )
            ev["page"] = str(raw_page)
            
            # 4. ปรับ Relevance Score โดยใช้มาตรฐานกลาง
            ev["relevance_score"] = self.get_actual_score(ev)
            
            # 5. จัดการ ID สำหรับการทำ Audit Trail
            if not ev.get("stable_doc_uuid"):
                ev["stable_doc_uuid"] = (
                    meta.get("stable_doc_uuid") or 
                    ev.get("doc_id") or 
                    meta.get("doc_id") or 
                    f"id_{uuid.uuid4().hex[:8]}"
                )
            
            # 6. ตรวจสอบ PDCA Tag (ถ้าไม่มีให้เป็น Other)
            if not ev.get("pdca_tag"):
                ev["pdca_tag"] = meta.get("pdca_tag") or "Other"

        return evidence_list
    
    # ------------------------------------------------------------------------------------------
    # [FIXED] 🧩 Persistence Helper: Update Internal Evidence
    # ------------------------------------------------------------------------------------------
    def _update_internal_evidence_map(self, merged_evidence: Dict[str, Any]):
        """
        รวบรวมและบันทึกความเชื่อมโยงของหลักฐานจากผลลัพธ์ที่ Merge แล้ว
        """
        if not hasattr(self, 'evidence_map'):
            self.evidence_map = {}
            
        self.logger.info("💾 Syncing merged evidence to internal storage...")
        
        if isinstance(merged_evidence, dict):
            for key, ev_list in merged_evidence.items():
                if not isinstance(ev_list, list): continue
                if key not in self.evidence_map:
                    self.evidence_map[key] = []
                
                # Deduplicate content เพื่อประหยัดพื้นที่และลดความซ้ำซ้อน
                existing_hashes = {hash(str(e.get('content'))[:100]) for e in self.evidence_map[key]}
                for ev in ev_list:
                    ev_hash = hash(str(ev.get('content'))[:100])
                    if ev_hash not in existing_hashes:
                        self.evidence_map[key].append(ev)
                        existing_hashes.add(ev_hash)
        
        self.logger.info(f"✅ Evidence mapping persistence ready. Total groups: {len(self.evidence_map)}")

    # ------------------------------------------------------------------------------------------
    # [ULTIMATE REVISE v2026.1.23] 🧩 Merge Worker Results (The "Zero-Score" Antidote)
    # ------------------------------------------------------------------------------------------
    def _merge_worker_results(self, sub_result: Dict[str, Any], temp_map: Dict[str, Any]):
        """
        รวมผลลัพธ์จาก Worker เข้าสู่โครงสร้างหลัก พร้อมระบบเชื่อมต่อ Evidence 
        ที่ป้องกันข้อมูลหลักฐานหายในโหมด Parallel
        """
        if not sub_result:
            return None

        # 1. 🔍 Identity & Metadata Setup
        sub_id = str(sub_result.get('sub_id', 'Unknown'))
        # ดึง Level ที่ประเมิน (รองรับทั้ง single และ batch result)
        level_received = int(sub_result.get('level') or sub_result.get('highest_full_level', 0))
            
        # 2. 🛡️ Evidence Mapping Sync (จุดที่แก้ปัญหา Evidence หาย)
        # temp_map มักจะส่งมาในรูปแบบ { "1": [chunks], "2": [chunks] }
        if temp_map and isinstance(temp_map, dict):
            for level_key, evidence_list in temp_map.items():
                if level_key not in self.evidence_map:
                    self.evidence_map[level_key] = []
                
                # สร้าง Set ของ ID ที่มีอยู่แล้วเพื่อป้องกันข้อมูลซ้ำ
                existing_ids = {
                    str(e.get('stable_doc_uuid') or e.get('doc_id') or e.get('source')) 
                    for e in self.evidence_map[level_key] if isinstance(e, dict)
                }
                
                for ev in evidence_list:
                    if not ev or ev in ["na", "n/a"]: continue
                    
                    # ตรวจสอบความซ้ำซ้อน
                    ev_id = str(ev.get('stable_doc_uuid') or ev.get('doc_id') or ev.get('source')) if isinstance(ev, dict) else str(ev)
                    
                    if ev_id not in existing_ids:
                        # Normalize ข้อมูลหลักฐานให้เป็น Dict มาตรฐาน
                        if not isinstance(ev, dict):
                            ev = {"content": str(ev), "source": "Manual Evidence", "page": "N/A"}
                        
                        # เพิ่มชื่อไฟล์จริงจาก document_map (ถ้ามี)
                        if 'doc_id' in ev and self.document_map:
                            ev['filename'] = self.document_map.get(ev['doc_id'], ev.get('source', 'Unknown'))
                        
                        self.evidence_map[level_key].append(ev)
                        existing_ids.add(ev_id)

        # 3. 🏗️ Manage Target Container (เตรียมก้อนผลลัพธ์ของหัวข้อนั้นๆ)
        if not hasattr(self, 'final_subcriteria_results'):
            self.final_subcriteria_results = []

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
                "audit_stop_reason": "Initiating...",
                "pdca_overall": {"P": 0.0, "D": 0.0, "C": 0.0, "A": 0.0}
            }
            self.final_subcriteria_results.append(target)

        # 4. 🧩 Atomic Update (อัปเดตข้อมูลราย Level)
        if 'level_details' in sub_result and isinstance(sub_result['level_details'], dict):
            target['level_details'].update(sub_result['level_details'])
        else:
            target['level_details'][str(level_received)] = sub_result

        # 5. ⚖️ Step-Ladder Maturity Calculation (จุดป้องกัน Score 0.0)
        current_highest = 0
        stop_reason = ""
        total_p, total_d, total_c, total_a, count_lv = 0, 0, 0, 0, 0
        
        for l in range(1, 6):
            l_str = str(l)
            l_data = target['level_details'].get(l_str)
            
            if l_data and isinstance(l_data, dict):
                # 🔍 เช็คความผ่านแบบยืดหยุ่น (Score >= 0.7 ถือว่า Pass เพื่อไม่ให้ Chain ขาด)
                score_val = float(l_data.get('score', 0))
                is_lv_passed = (
                    l_data.get('is_passed') is True or 
                    l_data.get('is_safety_pass') is True or
                    score_val >= 0.7
                )
                
                if is_lv_passed:
                    current_highest = l
                    l_data['is_passed'] = True # บันทึกสถานะกลับ
                    
                    # สะสมคะแนน PDCA เพื่อหาค่าเฉลี่ยภาพรวม
                    pdca = l_data.get('pdca_breakdown', {})
                    total_p += float(pdca.get('P', 0))
                    total_d += float(pdca.get('D', 0))
                    total_c += float(pdca.get('C', 0))
                    total_a += float(pdca.get('A', 0))
                    count_lv += 1
                else:
                    stop_reason = f"Stopped at L{l}: {str(l_data.get('reason', 'Insufficient evidence'))[:50]}..."
                    break
            else:
                stop_reason = f"No data for L{l}"
                break

        # 6. 💰 Final Summary Integration
        target['highest_full_level'] = current_highest
        target['is_passed'] = (current_highest >= 1)
        
        # คำนวณคะแนนถ่วงน้ำหนัก
        target['weighted_score'] = self._calculate_weighted_score(
            highest_full_level=current_highest,
            weight=target['weight'],
            level_details=target['level_details']
        )
        
        # คำนวณ PDCA เฉลี่ยของหัวข้อนี้
        if count_lv > 0:
            target['pdca_overall'] = {
                "P": round(total_p / count_lv, 2),
                "D": round(total_d / count_lv, 2),
                "C": round(total_c / count_lv, 2),
                "A": round(total_a / count_lv, 2)
            }
            
        target['audit_stop_reason'] = stop_reason if current_highest < 5 else "Target level achieved"
        
        self.logger.info(f"✅ [MERGE DONE] Sub {sub_id} -> Level {current_highest} (Score: {target['weighted_score']:.2f})")
        return target

           
    def _get_semantic_tag(self, text: str, sub_id: str, level: int, filename: str = "") -> str:
        """
        [ULTIMATE REVISE v2026.25 – Required Phase Aware]
        - Follow required_phase จาก contextual_rules ของ enabler นั้น ๆ
        - Prompt ส่ง require_phase ให้ LLM รู้ว่าต้อง prefer phase ไหน
        - Fallback เลือก phase จาก require_phase (ถ้ามีหลาย → random ตาม priority)
        - Heuristic ครอบคลุมทุก phase + JSON Clean ทนทาน
        """
        tenant = getattr(self.config, 'tenant', 'Unknown').upper()
        enabler = getattr(self.config, 'enabler', 'Unknown').upper()
        text_lower = text.lower().strip()

        # ดึง required_phase จาก rules (สำคัญที่สุด!)
        require_phases = self.get_rule_content(sub_id, level, "require_phase") or []
        require_str = ", ".join(require_phases) if require_phases else "P,D,C,A"

        # 1. Enhanced Heuristic (ครอบคลุมทุก phase – ขยายคำจาก log + defaults)
        if any(k in text_lower for k in [
            "นโยบาย", "แผน", "ยุทธศาสตร์", "มติ", "คำสั่ง", "เป้าหมาย", "เจตนารมณ์",
            "วิสัยทัศน์", "master plan", "roadmap", "km policy", "จัดทำแผน"
        ]):
            return "P"

        if any(k in text_lower for k in [
            "ดำเนินการ", "ปฏิบัติ", "กิจกรรม", "อบรม", "ประชุม", "จัดทำ", "ภาพถ่าย",
            "ลงพื้นที่", "ติดตามผล", "สรุปผล", "ขับเคลื่อน", "ลงนาม", "มติบอร์ด",
            "เห็นชอบ", "deployment", "จัดกิจกรรม", "ถ่ายทอด", "สื่อสาร", "คณะทำงาน"
        ]):
            return "D"

        if any(k in text_lower for k in [
            "รายงานผล", "ประเมิน", "ติดตาม", "ตัวชี้วัด", "kpi", "ผลลัพธ์", "สำรวจ",
            "สถิติ", "รายงาน", "monitoring", "review", "audit", "benchmarking",
            "feedback", "แบบสำรวจ", "วัดผล"
        ]):
            return "C"

        if any(k in text_lower for k in [
            "ปรับปรุง", "พัฒนา", "แก้ไข", "บทเรียน", "lesson learned", "ต่อยอด",
            "นวัตกรรม", "ดีขึ้น", "ยกระดับ", "agile", "นำไปใช้", "ทบทวน",
            "feedback loop"
        ]):
            return "A"

        # 2. LLM tagging (prompt ปรับตาม required_phase)
        system_prompt = (
            f"Auditor for {tenant} {enabler}. Classify text to ONE PDCA tag only. "
            "P=Plan (นโยบาย/แผน/เป้าหมาย), D=Do (กิจกรรม/อบรม/ประชุม/ภาพถ่าย/ดำเนินการ), "
            "C=Check (รายงาน/KPI/ประเมิน/ติดตาม), A=Act (ปรับปรุง/บทเรียน/นวัตกรรม). "
            f"Required phases for this level: {require_str}. Prefer one of these phases if ambiguous. "
            "JSON only: {\"tag\":\"P/D/C/A/Other\",\"reason\":\"สั้น ๆ ไทย\"}"
        )

        user_prompt = f"File: {filename}\nText (first 500 chars):\n{text[:500]}\n→ JSON"

        try:
            response = self.llm.invoke(system_prompt + "\n" + user_prompt)
            # Clean ทนทานสุด
            cleaned = response.strip()
            cleaned = re.sub(r'^.*?\{', '{', cleaned, flags=re.DOTALL | re.IGNORECASE)
            cleaned = re.sub(r'\}.*?$', '}', cleaned, flags=re.DOTALL | re.IGNORECASE)
            cleaned = cleaned.replace("```json", "").replace("```", "").replace("\n", " ").strip()
            cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)
            data = json.loads(cleaned)
            if isinstance(data, list) and data:
                data = data[0]
            tag = str(data.get("tag", "Other")).strip().upper()
            if tag in {"P", "D", "C", "A"}:
                reason = data.get('reason', '')
                self.logger.debug(f"[TAG LLM] {tag} | {reason} | {filename[:30]}")
                return tag
        except Exception as e:
            self.logger.warning(f"[TAG LLM FAIL] {sub_id} L{level} {filename[:30]} → {str(e)} | Raw: {response[:100]}...")

        # 3. Final fallback (ตาม required_phase – ถ้ามีหลาย phase เลือกตาม priority ของ level)
        if require_phases:
            # Priority: เน้น phase แรกที่ต้องการ (เช่น L1 → D ถ้ามี, L4 → C, L5 → A)
            for phase in require_phases:
                if phase in ["P", "D", "C", "A"]:
                    self.logger.debug(f"[TAG FALLBACK] L{level} → {phase} (from required phases)")
                    return phase
        # ถ้าไม่มี require_phase → fallback ทั่วไปตาม level
        if level <= 3:
            return "D"
        elif level == 4:
            return "C"
        else:
            return "A"
    
    def _build_pdca_context(self, blocks: Dict[str, str]) -> str:
        """
        [REVISED v2026.2]
        - เพิ่มการจัดการกรณี content ว่าง/สั้นเกิน
        - ใช้ XML-like แต่ตัดส่วนที่ไม่จำเป็นออก
        """
        tags = ["Plan", "Do", "Check", "Act", "Other"]
        parts = []

        for t in tags:
            content = blocks.get(t, "").strip()
            if not content or len(content) < 10:
                content = "[ไม่พบข้อมูลที่ชัดเจน]"
            # จำกัดความยาวเพื่อป้องกัน prompt ยาวเกิน
            content = content[:800]
            parts.append(f"<{t}>{content}</{t}>")

        return "\n".join(parts)
    

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
        [STRATEGIC QUERY GEN v2026.2.22 – FULL REVISION & OPTIMIZED]
        - Robust: handle empty/invalid input, fallback queries เสมอ
        - Phase-aware: weight queries ตาม required_phase + level maturity
        - Negative: เฉพาะเจาะจง ไม่บล็อกคำสำคัญ (เช่น ยังหา "แผนปฏิบัติการ" ได้)
        - Debug: log keywords + sample queries + total count
        - Post-process: dedup fuzzy + truncate 18-24 คำ + shuffle + max 8
        - Fallback: มี query อย่างน้อย 2 เสมอ (tenant + stmt core)
        - Speed: จำกัด queries สูงสุด 8 + truncate สั้นลง
        """
        logger = logging.getLogger(__name__)
        log_prefix = f"[QUERY-GEN] {sub_id} L{level}"

        # 0. Safety guard
        if not statement_text or not isinstance(statement_text, str):
            logger.warning(f"{log_prefix} Empty/invalid statement_text → fallback basic")
            fallback_q = f"{sub_id} {focus_hint or 'วิสัยทัศน์ นโยบาย'} การจัดการความรู้"
            return [fallback_q, f"{sub_id} แผนปฏิบัติการ"]

        # Anchors
        enabler_id = getattr(self.config, 'enabler', 'Unknown').upper()
        tenant_name = getattr(self.config, 'tenant', 'Unknown').upper()
        id_anchor = f"{enabler_id} {sub_id}"

        # 1. Required phases + keywords
        require_phases = self.get_rule_content(sub_id, level, "require_phase") or []
        require_str = ", ".join(require_phases) if require_phases else "P,D"

        raw_kws = self.get_rule_content(sub_id, level, "must_include_keywords") or []
        phase_map = {"P": "plan_keywords", "D": "do_keywords", "C": "check_keywords", "A": "act_keywords"}

        for phase in require_phases:
            kw_key = phase_map.get(phase)
            if kw_key:
                raw_kws.extend(self.get_rule_content(sub_id, level, kw_key) or [])

        # Fallback phases if none
        if not require_phases:
            if level <= 3:
                raw_kws.extend(self.get_rule_content(sub_id, 1, "plan_keywords") or [])
                raw_kws.extend(self.get_rule_content(sub_id, 2, "do_keywords") or [])
            else:
                raw_kws.extend(self.get_rule_content(sub_id, 2, "do_keywords") or [])
                raw_kws.extend(self.get_rule_content(sub_id, 3, "check_keywords") or [])

        clean_kws = sorted(set(str(k).strip() for k in raw_kws if k))
        keywords_str = " ".join(clean_kws[:5])
        short_keywords = " ".join(clean_kws[:3])

        logger.debug(f"{log_prefix} Keywords ({len(clean_kws)}): {keywords_str[:100]}...")

        # Clean stmt
        clean_stmt = statement_text.split("เช่น", 1)[0].strip()
        clean_stmt = re.sub(r'[^\w\s]', '', clean_stmt)[:70]

        queries: List[str] = []

        # 2. Negative + Core Queries
        neg_strict = "-แผนแม่บท -ยุทธศาสตร์ชาติ -MasterPlan -รายงานสรุป -ภาคผนวก"

        queries.append(f"{id_anchor} {clean_stmt} {keywords_str}")
        queries.append(f"{id_anchor} {clean_stmt}")

        if level <= 3:
            queries.append(f"{tenant_name} ประกาศ คำสั่ง ระเบียบ บันทึกข้อความ {id_anchor} {short_keywords}")
            queries.append(f"{id_anchor} (ผู้บริหาร OR ลงนาม OR มุ่งมั่น OR ขับเคลื่อน) {neg_strict}")
        else:
            queries.append(f"{tenant_name} รายงานผล KPI ผลสำเร็จ {id_anchor} {short_keywords}")
            queries.append(f"{id_anchor} (รายงานผล OR ประเมิน OR ติดตาม OR ปรับปรุง) {neg_strict}")

        # 3. Source Bias (P/D)
        if "P" in require_phases or "D" in require_phases:
            queries.append(f"{id_anchor} มติที่ประชุม รายงานการประชุม ประกาศ คำสั่ง ลงนาม {short_keywords}")

        # 4. Priority 1: query_synonyms
        query_syn = self.get_rule_content(sub_id, level, "query_synonyms") or ""
        if query_syn:
            queries.append(f"{id_anchor} {query_syn} {short_keywords}")

        # 5. Priority 2: specific_contextual_rule
        if not query_syn:
            specific_rule = self.get_rule_content(sub_id, level, "specific_contextual_rule") or ""
            if specific_rule:
                rule_words = [w.strip() for w in specific_rule.split() if len(w.strip()) >= 4]
                rule_synonyms = " ".join(list(dict.fromkeys(rule_words))[:8])
                if rule_synonyms:
                    queries.append(f"{id_anchor} {rule_synonyms} {short_keywords}")

        # 6. Priority 3: Fallback PDCA
        fallback_synonyms = {
            "P": "แผนปฏิบัติการ เป้าหมาย นโยบาย เจตนารมณ์ มุ่งมั่น ลงนาม",
            "D": "ปฏิบัติ ดำเนินการ ขับเคลื่อน กิจกรรม อบรม ประชุม บันทึกข้อความ",
            "C": "ประเมิน ติดตาม ตรวจสอบ รายงานผล KPI วัดผล สรุปผล",
            "A": "ปรับปรุง พัฒนา แก้ไข ต่อยอด นวัตกรรม ยกระดับ Best Practice"
        }

        for phase in require_phases:
            fallback = fallback_synonyms.get(phase, "")
            if fallback:
                queries.append(f"{id_anchor} {fallback} {short_keywords}")

        # 7. KM Specific
        if level <= 3 and enabler_id == "KM" and "D" in require_phases:
            queries.append(f"{id_anchor} (ประชุม OR อบรม OR กิจกรรม OR วิทยากร OR ถ่ายทอดความรู้) {neg_strict}")

        # 8. Advanced/Focus + tenant fallback
        if level >= 4 or focus_hint:
            adv = "นวัตกรรม Best Practice Lesson Learned ผลลัพธ์"
            queries.append(f"{id_anchor} {adv} {focus_hint or ''} {tenant_name}")

        # 9. Fallback core (ถ้า queries น้อยเกิน)
        if len(queries) < 3:
            queries.append(f"{tenant_name} {clean_stmt} {short_keywords}")
            queries.append(f"{id_anchor} แผนปฏิบัติการ การจัดการความรู้")

        # 10. Post-process: Dedup + Truncate + Shuffle + Limit
        final_queries = []
        seen = set()
        import random
        for q in queries:
            words = q.split()
            trunc_len = random.randint(18, 24)
            q_trunc = " ".join(words[:trunc_len])
            q_norm = " ".join(words[:16])
            if q_trunc and q_norm not in seen:
                final_queries.append(q_trunc)
                seen.add(q_norm)

        random.shuffle(final_queries)
        final_queries = final_queries[:8]  # Max 8

        logger.info(f"🚀 [Query Gen v2026.2.22] {sub_id} L{level} | Generated {len(final_queries)} queries (Phases: {require_str}) | Neg: {neg_strict}")
        if final_queries:
            logger.debug(f"{log_prefix} Final queries (top 3): {final_queries[:3]}")

        return final_queries
 
    
    def _get_level_aware_queries(self, criteria_id: str, level_key: str) -> List[str]:
        """
        ดึงคำค้นหาจาก JSON Rules (query_synonyms) มาผสมกับ PDCA Keywords
        """
        # 1. ดึงกฎจาก JSON ที่โหลดไว้ใน self.contextual_rules_map
        criteria_rules = self.contextual_rules_map.get(criteria_id, {})
        level_rule = criteria_rules.get(level_key, {})
        
        # 2. ดึง Synonyms ที่คุณเขียนไว้ (เช่น "คณะทำงาน กลไกขับเคลื่อน...")
        synonyms = level_rule.get("query_synonyms", "")
        
        # 3. ดึงกลุ่ม Phase ที่ต้องเน้น (P, D, C, A)
        required_phases = level_rule.get("require_phase", ["P", "D"])
        
        # 4. ดึงคำพื้นฐานของแต่ละ Phase จาก _enabler_defaults
        defaults = self.contextual_rules_map.get("_enabler_defaults", {})
        
        generated_queries = []
        
        # Query หลัก: เน้นตามเกณฑ์ + Synonyms
        main_q = f"{self.enabler} {criteria_id} {synonyms}"
        generated_queries.append(self._normalize_thai_text(main_q))
        
        # Query เสริม: แยกตาม Phase เพื่อกวาดหลักฐานให้ครบ PDCA
        for phase in required_phases:
            phase_key = f"{phase.lower()}_keywords"
            phase_words = " ".join(defaults.get(phase_key, [])[:4]) # เอามาแค่ 4 คำกันยาวเกิน
            combined_q = f"{self.enabler} {criteria_id} {synonyms} {phase_words}"
            generated_queries.append(self._normalize_thai_text(combined_q))
            
        return list(set(generated_queries)) # ลบตัวซ้ำ

    def _perform_adaptive_retrieval(
        self,
        sub_id: str,
        level: int,
        stmt: str,
        vectorstore_manager: Any,
    ) -> tuple[List[Dict], float]:
        """
        [HYBRID REVISED v2026.1.23] - Optimized for Mac & Thai Language
        - Hybrid Strategy: ใช้ JSON Rules เป็นหลัก และใช้ Legacy Enhance เป็นตัวเสริม
        - Dynamic Threshold: ปรับค่าการคัดออกให้เหมาะกับภาษาไทย
        - Smart Early Exit: หยุดเมื่อเจอหลักฐานที่ 'ดีพอ' ไม่ใช่ 'สมบูรณ์แบบ' (ป้องกัน Loop นาน)
        """
        if not stmt or not isinstance(stmt, str):
            return [], 0.0

        # --- 1. Configuration & Local Tuning ---
        level_key = f"L{level}"
        current_tenant = getattr(self.config, "tenant", "PEA").upper()
        
        # ปรับค่าเหล่านี้ให้เหมาะสมกับ BGE-M3 บน Mac
        EXIT_SCORE_THRESHOLD = CRITICAL_CA_THRESHOLD
        LOCAL_RERANK_FLOOR = RERANK_THRESHOLD      # ยอมรับคะแนนขั้นต่ำที่ 0.20 (สัมพันธ์กับ .env)
        MAX_LOOP_QUERIES = 6           # รันสูงสุด 6 loops เพื่อไม่ให้ช้าเกินไป
        
        candidates: List[Dict] = []
        final_max_rerank = 0.0
        used_uuids = set()

        # --- 2. Step 1: Priority Document Mapping ---
        # ดึงเอกสารที่ถูกระบุไว้ใน Mapping (ถ้ามี) มาเป็นฐานก่อน
        try:
            mapped_ids, priority_docs = self._get_mapped_uuids_and_priority_chunks(
                sub_id=sub_id, level=level, statement_text=stmt, vectorstore_manager=vectorstore_manager
            ) or (set(), [])
            for p in priority_docs:
                if p.get("chunk_uuid"): used_uuids.add(p.get("chunk_uuid"))
        except Exception as e:
            self.logger.error(f"❌ Priority loading failed: {e}")
            mapped_ids, priority_docs = set(), []

        # --- 3. Step 2: Hybrid Query Generation ---
        # รวมพลัง JSON Rules (แม่นยำ) + Enhance Query (ครอบคลุม)
        all_queries = []
        
        # (A) ตัวหลัก: ดึงจาก JSON
        json_queries = self._get_level_aware_queries(sub_id, level_key)
        all_queries.extend(json_queries)
        
        # (B) ตัวเสริม: ถ้า JSON มีน้อย ให้ดึงจาก Legacy มาช่วยเหวี่ยงแห
        if len(all_queries) < 3:
            legacy_queries = self.enhance_query_for_statement(stmt, sub_id, f"{sub_id}.L{level}", level)
            for q in legacy_queries:
                if q not in all_queries: all_queries.append(q)

        # กรองและจำกัดจำนวน
        active_queries = [q for q in all_queries if len(q.strip()) > 5][:MAX_LOOP_QUERIES]
        self.logger.info(f"🚀 [HYBRID-QUERY] {sub_id} {level_key} | Total: {len(active_queries)} (JSON + Fallback)")

        # --- 4. Step 3: Iterative Retrieval Loop ---
        for i, q in enumerate(active_queries):
            q_norm = self._normalize_thai_text(q)
            
            try:
                res = self.rag_retriever(
                    query=q_norm,
                    doc_type=self.doc_type,
                    sub_id=sub_id,
                    level=level,
                    vectorstore_manager=vectorstore_manager,
                    stable_doc_ids=mapped_ids,
                ) or {}
            except Exception as e:
                self.logger.error(f"❌ Retrieval error @ loop {i+1}: {e}")
                continue

            loop_docs = res.get("top_evidences") or []
            if not loop_docs: continue

            # วัดคุณภาพ Loop นี้
            current_max = max([d.get("score", 0.0) for d in loop_docs])
            final_max_rerank = max(final_max_rerank, current_max)

            # เก็บเฉพาะ Chunk ใหม่ที่คะแนนไม่น่าเกลียด (>= 0.15)
            new_found = 0
            for d in loop_docs:
                uid = d.get("chunk_uuid")
                score = d.get("score", 0.0)
                if uid and uid not in used_uuids and score >= 0.15:
                    used_uuids.add(uid)
                    candidates.append(d)
                    new_found += 1

            self.logger.info(
                f"🔍 [LOOP {i+1}] Query: {q_norm[:40]}... | New: {new_found} | Max Score: {current_max:.4f}"
            )

            # --- SMART EXIT ---
            # ถ้าเจอหลักฐานที่ 'ดีพอ' และจำนวน 'เยอะพอ' ให้หยุดทันทีเพื่อความเร็ว
            if current_max >= EXIT_SCORE_THRESHOLD and len(candidates) >= 12:
                self.logger.info(f"🎯 [SMART EXIT] Found high-quality match ({current_max:.4f}).")
                break

        # --- 5. Step 4: Recovery (กรณีคะแนนต่ำมาก) ---
        if final_max_rerank < LOCAL_RERANK_FLOOR and len(candidates) < 5:
            self.logger.warning(f"⚠️ [LOW-RESULT] Final score {final_max_rerank:.4f} is too low. Trying core recovery...")
            recovery_q = self._normalize_thai_text(f"{sub_id} {current_tenant} {stmt[:40]}")
            res_fb = self.rag_retriever(query=recovery_q, doc_type=self.doc_type, vectorstore_manager=vectorstore_manager)
            for d in (res_fb.get("top_evidences") or []):
                if d.get("chunk_uuid") not in used_uuids:
                    candidates.append(d)

        # --- 6. Step 5: Final Assembly ---
        all_results = priority_docs + candidates
        
        # เรียงลำดับตามคะแนน Rerank
        all_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        # ส่งให้ LLM ตามจำนวนที่กำหนดใน .env (ANALYSIS_FINAL_K)
        final_limit = int(os.environ.get("ANALYSIS_FINAL_K", "15"))
        final_docs = all_results[:final_limit]

        self.logger.info(
            f"🏁 [DONE] {sub_id} L{level} | Final Chunks: {len(final_docs)} | Max Rerank Score: {final_max_rerank:.4f}"
        )

        return final_docs, float(final_max_rerank)

    def _log_pdca_status(self, sub_id, name, level, blocks, req_phases, sources_count, score, conf_level, **kwargs):
        """
        [THE AUDITOR DASHBOARD v2026.3.10 - FULL REVISED]
        🧩 ระบบแสดงผลสถานะ PDCA แบบ Real-Count Dashboard
        - แก้ไขปัญหาตัวเลข Maturity Gap ไม่ตรงกับจำนวนที่ Save จริง
        - ใช้ระบบ Double-Check (Payload Count + Tagging List)
        """
        try:
            # 1. ดึงข้อมูลการนับจาก Payload หลักที่ฉีดเข้ามา (Single Source of Truth)
            # ดึงจาก pdca_breakdown ที่เราส่งมาจาก _run_single_assessment
            actual_counts = kwargs.get('pdca_breakdown', {}) 
            raw_tagging = kwargs.get('tagging_result') or []
            is_safety_pass = kwargs.get('is_safety_pass', False)
            
            status_parts = []
            extract_parts = []
            
            # Mapping ระหว่าง Key ใน JSON Response และตัวย่อ Phase
            mapping = [
                ("Extraction_P", "P"), 
                ("Extraction_D", "D"), 
                ("Extraction_C", "C"), 
                ("Extraction_A", "A")
            ]

            # 2. เริ่มการวิเคราะห์สถานะราย Phase
            for full_key, short in mapping:
                # --- [REVISED COUNTING LOGIC] ---
                # ลำดับความสำคัญ: 1. ดูจาก actual_counts | 2. นับจาก raw_tagging list
                if actual_counts and short in actual_counts:
                    count = actual_counts[short]
                elif isinstance(raw_tagging, list):
                    count = raw_tagging.count(short)
                else:
                    count = 0
                
                # ตรวจสอบว่า LLM สกัดเนื้อหา (Extraction) ออกมาได้จริงหรือไม่
                # กรองคำที่เป็นค่าว่างหรือ N/A ออกเพื่อให้ Icon แสดงผลแม่นยำ
                content = str(blocks.get(full_key, "")).strip()
                ai_found = bool(content and content.lower() not in [
                    "-", "n/a", "none", "null", "ไม่พบข้อมูล", "ไม่พบหลักฐาน", "ไม่ระบุ"
                ])
                
                # --- [ICON LOGIC v2026.3.10] ---
                # ✅: พบหลักฐานเชิงประจักษ์ (Count > 0)
                # 🔷: ระบบ Force Pass หรือ AI วิเคราะห์เจอเองแต่ RAG Tagging ไม่ติด
                # ➖: เฟสนี้ไม่ได้ถูกบังคับ (Not in req_phases)
                # ❌: เฟสที่บังคับแต่หาหลักฐานไม่เจอเลย (Count=0 และ AI ไม่เจอ)
                
                if count > 0: 
                    icon = "✅" 
                elif ai_found or (is_safety_pass and short in req_phases): 
                    icon = "🔷"
                elif short not in req_phases: 
                    icon = "➖"
                else: 
                    icon = "❌"
                
                # ประกอบข้อความสถานะ เช่น P:✅(15)
                status_parts.append(f"{short}:{icon}({count})")
                
                # เก็บตัวอย่างการสกัดข้อมูลสั้นๆ เพื่อทำ Trace Log (2 ชิ้นแรก)
                if ai_found and len(extract_parts) < 2:
                    clean_content = content.replace("\n", " ")
                    extract_parts.append(f"[{short}: {clean_content[:60]}...]")

            # 3. จัดการเรื่องคะแนนและการแสดงผล
            display_score = float(score) if score is not None else 0.0
            
            # 4. [DASHBOARD OUTPUT] พิมพ์ Log หลักที่แสดงผลหน้าจอ
            # แสดงค่าที่ตรงกับข้อมูลที่บันทึกลงฐานข้อมูลจริง
            self.logger.info(
                f"📊 [PDCA-STATUS] {sub_id} L{level} | {str(name)[:60]}...\n"
                f"   Maturity Gap: {' '.join(status_parts)}{' 🛡️[SAFETY-PASS]' if is_safety_pass else ''}\n"
                f"   Summary: Score={display_score:.2f} | Evidence={sources_count} chunks | Conf={conf_level.upper()}"
            )
            
            # 5. พิมพ์ Log ยืนยันร่องรอยหลักฐาน (Traceability)
            if extract_parts:
                self.logger.info(f"🔍 [EXTRACT-TRACE] {' | '.join(extract_parts)}")

        except Exception as e:
            # ป้องกันระบบพังหากเกิด Error ในขั้นตอนการทำ Log Dashboard
            self.logger.error(f"❌ Critical Error in _log_pdca_status: {str(e)}")

    def _summarize_evidence_list_short(self, evidences: list, max_sentences: int = 3) -> str:
        """
        [REVISED v2026.SUMMARY.4]
        - เปลี่ยนเป็น Method เพื่อใช้ self.logger
        - เน้นดึง Source และ Page เพื่อสร้าง Audit Traceability
        - ปรับ Formatting ให้เป็น Bullet points (LLM อ่านง่ายกว่า Pipe '|')
        """
        if not evidences:
            return "ไม่พบข้อมูลหลักฐานเพิ่มเติม"
        
        parts = []
        # เลือกเฉพาะหลักฐานที่มีคุณภาพ (มีเนื้อหา) และจำกัดจำนวนตาม max_sentences
        valid_evidences = [
            ev for ev in evidences 
            if isinstance(ev, dict) and (ev.get("text") or ev.get("content", "")).strip()
        ]
        
        # ตัดจำนวนตามที่ต้องการ (ไม่เกินจำนวนที่มีจริง)
        target_count = max(1, min(len(valid_evidences), max_sentences))
        
        for ev in valid_evidences[:target_count]:
            # 1. สกัดข้อมูลแหล่งที่มา (Source Mapping)
            filename = (ev.get("file_name") or ev.get("source") or 
                        ev.get("source_filename") or "ไม่ระบุชื่อไฟล์")
            page = ev.get("page", "-")
            
            # 2. ทำความสะอาดเนื้อหา (Data Cleaning)
            raw_text = ev.get("text") or ev.get("content") or ""
            # ลบการขึ้นบรรทัดใหม่ที่เกินจำเป็น และตัดเอาเฉพาะส่วนต้น
            clean_text = " ".join(raw_text.split()).strip()
            text_preview = clean_text[:150] # เพิ่มความยาวเป็น 150 เพื่อให้ได้บริบทที่ชัดขึ้น
            
            # 3. ประกอบร่าง (Formatting)
            if text_preview:
                parts.append(f"• [{filename}, หน้า {page}]: \"{text_preview}...\"")
            else:
                parts.append(f"• [{filename}, หน้า {page}]: (พบความเกี่ยวข้องแต่ไม่สามารถสรุปเนื้อหาได้)")

        # เชื่อมด้วยการขึ้นบรรทัดใหม่เพื่อให้ AI แยกแยะแต่ละชิ้นได้ง่าย
        return "\n".join(parts) 
    
    def relevance_score_fn(self, evidence: Dict[str, Any], sub_id: str, level: int) -> float:
        """
        [REVISED RELEVANCE SCORE v2026.3.1 - Balanced & Robust]
        - 45% Rerank + 35% Keyword + 20% Bonuses
        - PDCA tag bonus สูงขึ้นถ้าตรงกับ required phases
        - Source grading ปรับให้สมดุล (primary + secondary)
        - Min floor สำหรับ rerank สูง
        - Logging ชัดเจนขึ้น
        """
        if not evidence:
            return 0.0

        # 1. Rerank (45%)
        rerank_raw = evidence.get('rerank_score') or evidence.get('score') or 0.0
        rerank_score = float(rerank_raw) if isinstance(rerank_raw, (int, float)) else 0.0
        normalized_rerank = min(max(rerank_score, 0.0), 1.0)

        # 2. ข้อมูลพื้นฐาน
        text = (evidence.get('text') or evidence.get('page_content') or '').lower().strip()
        meta = evidence.get('metadata', {}) if isinstance(evidence.get('metadata'), dict) else {}
        filename = (meta.get('source') or meta.get('source_filename') or '').lower()

        # 3. Cumulative rules
        cum_rules = self.get_cumulative_rules_cached(sub_id, level)

        # 4. Source Grading (ปรับสมดุล)
        source_bonus = 0.0
        primary = ["มติ", "บันทึก", "คำสั่ง", "ประกาศ", "นโยบาย", "แผนแม่บท", "มติบอร์ด"]
        secondary = ["assessment report", "รายงานการประเมิน", "สรุปผล", "รายงานผล", "kpi"]
        if any(p in filename for p in primary):
            source_bonus += 0.20
        if any(p in filename for p in secondary):
            source_bonus += 0.10  # ไม่ลบ แต่ลดน้ำหนัก

        # 5. Keyword Score (35%)
        target_kws = set()
        if level <= 2:
            target_kws.update(cum_rules.get('plan_keywords', []) + cum_rules.get('do_keywords', []))
        else:
            target_kws.update(cum_rules.get('check_keywords', []) + cum_rules.get('act_keywords', []))

        match_count = sum(1 for kw in target_kws if kw.lower() in text)
        expected = max(1, len(target_kws) * 0.3)  # ปรับ threshold ลงนิด
        keyword_score = min((match_count / expected) ** 0.6, 1.0)  # ^0.6 เพื่อไม่ให้ต่ำเกิน
        keyword_score = max(keyword_score, 0.20 if match_count >= 1 else 0.0)

        # 6. PDCA Tag Bonus (0.30 ถ้าตรง required)
        pdca_bonus = 0.0
        pdca_tag = evidence.get('pdca_tag') or meta.get('pdca_tag') or ""
        required_phases = cum_rules.get('required_phases', [])
        if pdca_tag and str(pdca_tag).upper() in required_phases:
            pdca_bonus = 0.30
        elif pdca_tag and str(pdca_tag).upper() in {'P', 'D', 'C', 'A'}:
            pdca_bonus = 0.15  # bonus เล็กถ้าตรง PDCA แต่ไม่อยู่ใน required

        # 7. Neighbor Bonus
        neighbor_bonus = 0.15 if evidence.get('is_neighbor', False) or meta.get('is_neighbor', False) else 0.0

        # 8. Specific Rule Bonus (เพิ่มถ้าข้อความตรงกับ rule เฉพาะ)
        specific_rule = cum_rules.get('specific_contextual_rule', '').lower()
        rule_bonus = 0.15 if specific_rule and any(word in text for word in specific_rule.split()[:10]) else 0.0

        # 9. รวมคะแนน (45% Rerank + 35% Keyword + 20% Bonuses)
        final_score = (
            0.45 * normalized_rerank +
            0.35 * keyword_score +
            source_bonus + pdca_bonus + neighbor_bonus + rule_bonus
        )

        # 10. Min floor สำหรับ rerank สูง
        if normalized_rerank > 0.80:
            final_score = max(final_score, 0.45)

        final_score = min(max(final_score, 0.0), 1.0)

        # 11. Logging (info สำหรับ high score, debug สำหรับทุกตัว)
        if normalized_rerank > 0.75 or final_score > 0.60:
            self.logger.info(
                f"[HIGH-RELEVANCE] {sub_id} L{level} | "
                f"final={final_score:.4f} | rerank={normalized_rerank:.4f} | "
                f"kw={keyword_score:.4f} | pdca_bonus={pdca_bonus:.3f} | "
                f"tag={pdca_tag} | source_bonus={source_bonus:.3f}"
            )

        self.logger.debug(
            f"[{sub_id} L{level}] RelScore: {final_score:.4f} | Rerank: {normalized_rerank:.4f} | "
            f"KW: {keyword_score:.4f} | PDCA: {pdca_bonus:.3f} | Src: {source_bonus:.3f}"
        )

        return final_score

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
        base_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        [JUDICIAL REVIEW MODULE - FULL REVISED v2026.3.28]
        - Inject hint_msg เข้า pdca_blocks โดยตรง (แก้ปัญหา evaluate_pdca ไม่รับ context)
        - มี retry mechanism (max 2 ครั้ง) + basic timeout handling
        - Validation เข้มงวด + fallback ที่ปลอดภัย
        - Log ละเอียดเพื่อ traceability และ debug
        - รองรับ missing_tags ทั้ง list และ set
        """
        log_prefix = f"Sub:{sub_id} L{level}"
        self.logger.info(f"⚖️ [EXPERT-APPEAL START] {log_prefix} | Max Rerank: {highest_rerank_score:.4f}")

        # 1. เตรียม missing string (รองรับทั้ง list และ set)
        missing_set = set(missing_tags) if isinstance(missing_tags, (list, set)) else set()
        missing_str = ", ".join(sorted(missing_set)) if missing_set else "พฤติกรรมตามเกณฑ์ PDCA"

        # 2. สร้าง Expert Instruction (Hint) ที่ชัดเจนและกระชับ
        hint_msg = f"""
    ### 🚨 EXPERT JUDICIAL REVIEW - SECOND CHANCE (APPEAL) 🚨
    CONTEXT จากรอบแรก: ไม่ผ่านเพราะ "{first_attempt_reason[:120]}..."
    หลักฐานที่มี: ความเกี่ยวข้องสูงมาก (rerank {highest_rerank_score:.4f}) ซึ่งอาจมีข้อมูลเกี่ยวกับ {missing_str}

    MANDATE (กฎบังคับ):
    - ใช้หลัก 'Substance over Form' → หากมีร่องรอยการปฏิบัติจริง แม้เอกสารไม่สมบูรณ์แบบ ให้ถือว่าผ่าน (is_passed: true)
    - อย่าตัดสินเข้มงวดเกินไปกับรูปแบบเอกสาร ถ้ามี substance พอสมควร
    - ถ้าคิดว่าผ่าน ให้ระบุเหตุผลชัดเจนว่ามีหลักฐานอะไรสนับสนุน

    DO NOT reject just because of missing formal signature if practice is evident.
    """

        # 3. Inject hint เข้า pdca_blocks (วิธีหลักที่ evaluate_pdca จะเห็นแน่นอน)
        expert_pdca_blocks = base_kwargs.get("pdca_blocks", []).copy()

        if isinstance(expert_pdca_blocks, list):
            # เพิ่มเป็น block พิเศษท้ายสุด (priority สูง)
            expert_pdca_blocks.append({
                "type": "judicial_review_instruction",
                "content": hint_msg,
                "metadata": {
                    "priority": "highest",
                    "source": "appeal_system",
                    "rerank_score": highest_rerank_score
                }
            })
            self.logger.debug(f"[APPEAL-INJECT] Added hint block to pdca_blocks (total blocks: {len(expert_pdca_blocks)})")
        else:
            # ถ้า pdca_blocks เป็น string หรือ dict อื่น → concat
            expert_pdca_blocks = f"{expert_pdca_blocks}\n\n--- APPEAL INSTRUCTION ---\n{hint_msg}"
            self.logger.debug("[APPEAL-INJECT] Concatenated hint to pdca_blocks (string mode)")

        # 4. เตรียม kwargs สำหรับ LLM evaluator (เฉพาะที่มันรับจริง)
        expert_kwargs = {
            "pdca_blocks": expert_pdca_blocks,
            "sub_id": sub_id,
            "level": level,
            "audit_confidence": getattr(self, "current_audit_meta", {"level": "HIGH", "score": 1.0})
        }

        self.logger.info(f"[APPEAL-SEND] {log_prefix} | Sending pdca_blocks with appeal hint | Confidence: {expert_kwargs['audit_confidence'].get('level')}")

        # 5. Call LLM ด้วย retry (max 2 ครั้ง)
        re_eval_result = None
        max_attempts = 2
        for attempt in range(1, max_attempts + 1):
            try:
                re_eval_result = llm_evaluator_to_use(**expert_kwargs)
                if re_eval_result is not None:
                    self.logger.debug(f"[APPEAL-SUCCESS] Attempt {attempt}: Received result")
                    break
                self.logger.warning(f"[APPEAL-RETRY] Attempt {attempt}: No result returned")
            except Exception as e:
                self.logger.warning(f"[APPEAL-ERROR] Attempt {attempt}: {str(e)}")
                if attempt == max_attempts:
                    return {
                        "is_passed": False,
                        "score": 0.0,
                        "reason": f"Appeal failed after {max_attempts} attempts: {str(e)}",
                        "appeal_status": "FAILED"
                    }

        # 6. Validation เข้มงวด
        if not isinstance(re_eval_result, dict):
            self.logger.error(f"❌ [APPEAL-INVALID] {log_prefix}: Result is not dict → {type(re_eval_result)}")
            return {
                "is_passed": False,
                "score": 0.0,
                "reason": "Expert System: Invalid response format",
                "appeal_status": "INVALID"
            }

        # 7. จัดการผลลัพธ์ + เพิ่ม traceability
        is_passed = bool(re_eval_result.get("is_passed", False))

        if is_passed:
            self.logger.info(f"🛡️ [OVERRIDE-SUCCESS] {log_prefix} | Appeal Granted")
            re_eval_result["is_safety_pass"] = True
            re_eval_result["appeal_status"] = "GRANTED"
            re_eval_result["reason"] = f"🌟 [EXPERT OVERRIDE]: {re_eval_result.get('reason', 'ผ่านจากการอุทธรณ์')}"
        else:
            self.logger.info(f"❌ [APPEAL-DENIED] {log_prefix}")
            re_eval_result["appeal_status"] = "DENIED"
            re_eval_result["reason"] = re_eval_result.get("reason", "อุทธรณ์ไม่สำเร็จ - ยังไม่ผ่านเกณฑ์")

        # เพิ่มข้อมูล traceability เพิ่มเติม
        re_eval_result.update({
            "appeal_rerank_score": highest_rerank_score,
            "appeal_missing_tags": missing_str,
            "appeal_attempt": attempt,
            "appeal_timestamp": datetime.now().isoformat()
        })

        return re_eval_result

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
        [ULTIMATE REVISED v2026.1.23 - FULL EVIDENCE TRACEABILITY] 
        Main entry point สำหรับการประเมิน SE-AM
        - แก้ไข: การเชื่อมต่อหลักฐาน (Evidence) จาก Worker เข้าสู่ Master Map
        - แก้ไข: ระบบสะสมคะแนน PDCA ให้ครบถ้วนทุกระดับ
        """
        start_ts = time.time()
        self.is_sequential = sequential
        self.current_record_id = record_id or self.record_id
        
        # มั่นใจว่ามี document_map สำหรับแปลง ID เป็นชื่อไฟล์
        if document_map:
            self.document_map.update(document_map)

        # 1. 📂 โหลดและจัดกลุ่มเกณฑ์ประเมิน
        flat_statements = self._flatten_rubric_to_statements()
        grouped_sub_criteria = self._group_statements_by_sub_criteria(flat_statements)

        is_all = str(target_sub_id).lower() == "all"
        sub_criteria_list = list(grouped_sub_criteria.values()) if is_all else [grouped_sub_criteria.get(target_sub_id)]
        
        if not all(sub_criteria_list):
            return self._create_failed_result(self.current_record_id, f"Criteria '{target_sub_id}' not found", start_ts)

        total_subs = len(sub_criteria_list)
        self.db_update_task_status(progress=5, message=f"📊 เตรียมความพร้อม: ประเมิน {total_subs} หัวข้อ")

        # 2. 🧠 เริ่มการประเมิน (Tier-1 & Tier-2)
        results_list = []
        
        if is_all and not sequential:
            # [MODE A] PARALLEL
            max_workers = int(os.environ.get("MAX_PARALLEL_WORKERS", 4))
            worker_args = [self._prepare_worker_tuple(sub, self.document_map) for sub in sub_criteria_list]
            
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(processes=max_workers) as pool:
                for idx, res_tuple in enumerate(pool.imap_unordered(_static_worker_process, worker_args)):
                    # res_tuple: (worker_result_dict, worker_evidence_mem)
                    results_list.append(res_tuple)
                    
                    # 🎯 [CRITICAL FIX] Merge ข้อมูลและ 'หลักฐาน' เข้าสู่ Memory หลักทันที
                    self._merge_worker_results(res_tuple[0], res_tuple[1])
                    
                    # อัปเดตความคืบหน้า
                    sub_id_now = res_tuple[0].get('sub_id', '?')
                    self.db_update_task_status(
                        progress=15 + int(((idx+1)/total_subs) * 65), 
                        message=f"🧠 เสร็จสิ้นหัวข้อ {sub_id_now} ({idx+1}/{total_subs})"
                    )
        else:
            # [MODE B] SEQUENTIAL
            if not vectorstore_manager: self._initialize_vsm_if_none()
            vsm = vectorstore_manager or self.vectorstore_manager

            for idx, sub_criteria in enumerate(sub_criteria_list):
                sub_id = str(sub_criteria.get("sub_id", "Unknown"))
                self.db_update_task_status(progress=15 + int((idx/total_subs)*65), message=f"🧠 กำลังประเมิน {sub_id}")
                
                # รวบรวมหลักฐานจาก Level ก่อนหน้า (ถ้ามี)
                prev_map = self._collect_previous_level_evidences(sub_id=sub_id, current_level=1)
                initial_baseline = [ev for evs in prev_map.values() for ev in evs]
                
                # รันการประเมิน
                res, worker_mem = self._run_sub_criteria_assessment_worker(sub_criteria, vsm, initial_baseline)
                results_list.append((res, worker_mem))
                
                # 🎯 [CRITICAL FIX] Merge ทันทีเพื่อให้คะแนนสะสม (Cumulative) ไม่หาย
                self._merge_worker_results(res, worker_mem)

        # 3. 🧩 รวมผลและสังเคราะห์แผนยุทธศาสตร์ (Tier-3)
        self.db_update_task_status(progress=85, message="🧩 กำลังสังเคราะห์แผนยุทธศาสตร์ภาพรวม")
        
        # ตรวจสอบความสมบูรณ์ของข้อมูลหลักฐาน (Evidence Trail)
        total_evidence_found = len(self.evidence_map)
        self.logger.info(f"📊 Total Evidence Found: {total_evidence_found} files")

        # --- 💾 [IRONCLAD SAVE POINT] ---
        try:
            self.logger.info("💾 [EVIDENCE] Initiating ironclad persistence...")
            self._save_evidence_map() # บันทึกไฟล์ mapping ลง disk กันเหนียว
        except Exception as e:
            self.logger.error(f"⚠️ [EVIDENCE] Auto-save failed: {e}")

        # สังเคราะห์ Master Strategic Roadmap (สำหรับทุกหัวข้อ)
        master_roadmap_data = None
        if is_all and len(self.final_subcriteria_results) > 0:
            master_roadmap_data = self.synthesize_strategic_roadmap(
                sub_criteria_results=self.final_subcriteria_results,
                enabler_name=self.enabler,
                llm_executor=self.llm
            )

        # 4. 🏁 บันทึกและสรุปผล
        overall_stats = self._calculate_overall_stats(target_sub_id)
        # เพิ่มจำนวนหลักฐานที่ใช้จริงเข้าไปใน stats
        overall_stats["evidence_used_count"] = total_evidence_found
        
        final_response = {
            "record_id": self.current_record_id,
            "status": "COMPLETED",
            "enabler": self.enabler,
            "summary": overall_stats,
            "sub_criteria_results": self.final_subcriteria_results,
            "evidence_audit_trail": self.evidence_map, # 🎯 ส่ง Mapping ไฟล์ทั้งหมดออกไปด้วย
            "strategic_roadmap": master_roadmap_data,
            "run_time_seconds": round(time.time() - start_ts, 2)
        }

        self.master_roadmap_data = master_roadmap_data 
        if export:
            # ส่งข้อมูลทั้งหมด (รวมถึง Roadmap) ไปยังฟังก์ชัน Export
            final_response["export_path"] = self._export_results(
                results_data=final_response, # ส่งก้อนใหญ่ไปเลย
                sub_criteria_id=target_sub_id
            )

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
    
    def create_atomic_action_plan(self, insight: str, level: int) -> List[Dict[str, Any]]:
        """
        [FULL REVISED v2026.1.25 - ROBUST ATOMIC PLAN GENERATION]
        - Prompt ชัดเจน + ตัวอย่าง JSON เพื่อลด truncation
        - Log raw LLM response เต็มเพื่อ debug
        - Regex scavenger ยืดหยุ่น (รองรับ key อื่น ๆ)
        - Validation + cleanup ก่อน return
        - Fallback ที่สมเหตุสมผลกว่าเดิม
        """
        try:
            # 1. Validation
            if not insight or str(insight).lower().strip() in ["-", "n/a", "none", "ไม่มีข้อมูล", "ไม่พบหลักฐาน", ""]:
                self.logger.debug(f"[ATOMIC-SKIP] Insight ว่างหรือไม่เกี่ยวข้องสำหรับ L{level}")
                return []

            # 2. Preparation
            clean_insight = str(insight).replace('"', "'").strip()
            if len(clean_insight) > 800:
                clean_insight = clean_insight[:800] + "... (ตัดเพื่อความปลอดภัย)"

            # Prompt ที่แรงขึ้น + ตัวอย่างชัดเจน
            human_prompt = ATOMIC_ACTION_PROMPT.format(
                coaching_insight=clean_insight,
                level=level
            )

            # 3. LLM Generation (เพิ่ม log raw)
            raw_text = _fetch_llm_response(
                system_prompt=(
                    "You are a strict JSON generator. "
                    "Return ONLY a valid JSON array of objects. "
                    "NO extra text, NO explanation, NO markdown. "
                    "Format MUST be: "
                    '[{"action": "ข้อความภาษาไทยชัดเจน", "target_evidence": "เอกสาร/หลักฐานอ้างอิง"}]'
                    "\nExample for level 1: "
                    '[{"action": "จัดทำคำสั่งแต่งตั้งคณะทำงาน KM", "target_evidence": "ประกาศองค์กร ฉบับที่..."}]'
                ),
                user_prompt=human_prompt,
                llm_executor=self.llm
            )

            # Log raw response เพื่อ debug (สำคัญมาก!)
            self.logger.debug(f"[ATOMIC-RAW-L{level}] Raw LLM response (length={len(raw_text)}):\n{raw_text[:1000]}...")

            # 4. Hybrid Extraction Logic (เพิ่มด่าน)
            actions = []

            # ด่าน 1: Robust JSON extract
            try:
                actions = _robust_extract_json_list(raw_text)
                if actions and isinstance(actions, list):
                    self.logger.debug(f"[ATOMIC-JSON-SUCCESS] Extracted {len(actions)} actions from JSON")
            except Exception as e:
                self.logger.warning(f"[ATOMIC-JSON-FAIL] {str(e)}")

            # ด่าน 2: Enhanced Scavenger Regex (รองรับ key หลากหลาย)
            if not actions:
                # รองรับ "action", "step", "recommendation", "task"
                patterns = [
                    r'"action"\s*:\s*"([^"]+)"',
                    r'"step"\s*:\s*"([^"]+)"',
                    r'"recommendation"\s*:\s*"([^"]+)"',
                    r'"task"\s*:\s*"([^"]+)"',
                    r'"activity"\s*:\s*"([^"]+)"'
                ]
                found_actions = []
                for pat in patterns:
                    found_actions.extend(re.findall(pat, raw_text))

                # ดึง target_evidence ถ้ามี
                found_evidences = re.findall(r'"(target_evidence|evidence|reference)"\s*:\s*"([^"]+)"', raw_text)
                found_evidences = [ev[1] for ev in found_evidences]  # เอาแค่ value

                for i, act in enumerate(found_actions):
                    evid = found_evidences[i] if i < len(found_evidences) else "เอกสารประกอบ/หลักฐานตามข้อเสนอแนะ"
                    actions.append({"action": act, "target_evidence": evid})

                if actions:
                    self.logger.debug(f"[ATOMIC-SCAVENGER] Found {len(actions)} actions via regex")

            # ด่าน 3: Normalization + Validation
            final_actions = []
            if isinstance(actions, list):
                for item in actions:
                    if not isinstance(item, dict):
                        continue

                    # ดึง action จาก key ที่เป็นไปได้
                    act_val = (
                        item.get("action") or
                        item.get("step") or
                        item.get("recommendation") or
                        item.get("task") or
                        item.get("activity") or
                        ""
                    ).strip()

                    if not act_val or len(act_val) < 5:
                        continue  # ข้ามรายการที่ไม่สมบูรณ์

                    evid = (
                        item.get("target_evidence") or
                        item.get("evidence") or
                        item.get("reference") or
                        "หลักฐานอ้างอิงตามข้อเสนอแนะ"
                    ).strip()

                    final_actions.append({
                        "action": act_val,
                        "target_evidence": evid,
                        "level": int(level)
                    })

            # 4. Emergency Fallback (ปรับให้สมจริงขึ้น)
            if not final_actions:
                self.logger.warning(f"⚠️ [ATOMIC-FALLBACK] Salvaging from insight for L{level}")
                # ใช้ประโยคแรก + ทำให้เป็น action ที่ชัดเจน
                first_sent = clean_insight.split(".")[0].strip() if "." in clean_insight else clean_insight
                if len(first_sent) > 10:
                    final_actions.append({
                        "action": f"ดำเนินการปรับปรุงตามข้อเสนอแนะหลัก: {first_sent}",
                        "target_evidence": "รายงานผลการดำเนินงาน / เอกสารประกอบ",
                        "level": int(level)
                    })
                else:
                    final_actions.append({
                        "action": f"ยกระดับการจัดการความรู้ตามระดับ {level}",
                        "target_evidence": "หลักฐานเชิงประจักษ์จากองค์กร",
                        "level": int(level)
                    })

            # จำกัดสูงสุด 3 รายการ + log สรุป
            processed_actions = final_actions[:3]
            if processed_actions:
                self.logger.info(f"✅ [Atomic-Plan] L{level} Success with {len(processed_actions)} actions")
                for i, act in enumerate(processed_actions, 1):
                    self.logger.debug(f"  Action {i}: {act['action'][:80]}... | Evidence: {act['target_evidence'][:60]}...")
            else:
                self.logger.warning(f"[Atomic-Plan] L{level} No valid actions after all stages")

            return processed_actions

        except Exception as e:
            self.logger.warning(f"⚠️ [Atomic-Plan] Critical fallback at L{level}: {str(e)}")
            return [{
                "action": f"ดำเนินงานตามแนวทางระดับ {level} (fallback จากข้อผิดพลาด)",
                "target_evidence": "ตรวจสอบ log และปรับปรุงระบบ",
                "level": int(level)
            }]
    
    # ------------------------------------------------------------------
    # 🏛️ [TIER-3 METHOD] generate_master_roadmap - FULL REVISE v2026.1.23
    # ------------------------------------------------------------------
    def generate_master_roadmap(self, sub_id, sub_criteria_name, enabler, aggregated_insights):
        """
        [TIER-3 STRATEGIC SYNTHESIS - v2026.1.23]
        สังเคราะห์ Roadmap ภาพรวมราย Sub-Criteria โดยเชื่อมต่อกับ Ironclad Fetcher
        - ใช้ _fetch_llm_response เพื่อความนิ่งของ JSON และระบบ Retry
        - มีระบบ Normalization เพื่อให้ข้อมูลพร้อมใช้งานใน Report และ Dashboard
        - จัดการภาษาไทยและอักขระพิเศษอย่างเป็นระบบ
        """
        
        self.logger.info(f"🔮 [MASTER-ROADMAP] Starting synthesis for {sub_id} ({sub_criteria_name})")

        # 1. 📂 Data Condensing: บีบอัด Insight เพื่อประหยัด Token และป้องกัน Context Overflow
        if not aggregated_insights:
            self.logger.warning(f"⚠️ No insights for {sub_id} - Using emergency fallback")
            return self._get_emergency_fallback_plan(sub_id, sub_criteria_name, "No insights provided")

        condensed_insights = []
        for item in aggregated_insights:
            status = "✅ PASSED" if item.get('is_passed') or item.get('status') == "PASSED" else "❌ FAILED"
            lv = item.get('level', '?')
            # ดึงบทสรุปเพียงส่วนสำคัญเพื่อลด Token Input
            insight = item.get('insight_summary') or item.get('reason') or 'ไม่มีรายละเอียด'
            condensed_insights.append(f"Level {lv} [{status}]: {insight[:200]}")

        summary_context = "\n".join(condensed_insights)

        # 2. 📝 Prompt Construction
        try:
            formatted_prompt = MASTER_ROADMAP_PROMPT.format(
                sub_id=sub_id,
                sub_criteria_name=sub_criteria_name,
                enabler=enabler,
                aggregated_insights=summary_context
            )
        except Exception as fe:
            self.logger.error(f"❌ Prompt formatting error: {fe}")
            formatted_prompt = f"Summarize roadmap for {sub_criteria_name}: {summary_context}"

        # 3. 🧠 LLM Execution via Ironclad Fetcher
        try:
            # ใช้ Fetcher ตัวใหม่ที่เราคุยกัน ซึ่งมีระบบล้าง JSON และ Retry ในตัว
            raw_json_str = _fetch_llm_response(
                system_prompt=SYSTEM_MASTER_ROADMAP_PROMPT,
                user_prompt=formatted_prompt,
                max_retries=3,
                llm_executor=self.llm  # มั่นใจว่าตัวนี้คือ ChatOllama/Ollama Instance
            )

            # 4. 🧹 Double-Check Extraction & Normalization
            master_data = _robust_extract_json(raw_json_str)
            
            if not master_data or (not master_data.get("overall_strategy") and not master_data.get("phases")):
                self.logger.warning(f"⚠️ Synthesis result is hollow for {sub_id} - Using fallback")
                return self._get_emergency_fallback_plan(sub_id, sub_criteria_name, "Hollow JSON response")

            # 5. 🏗️ UI-Ready Normalization
            # ปรับจูนโครงสร้างข้อมูลให้เป็นมาตรฐานเดียวกัน (Standard Schema)
            final_strategy = master_data.get("overall_strategy") or master_data.get("summary") or "ไม่สามารถสรุปกลยุทธ์ได้"
            raw_phases = master_data.get("phases") or master_data.get("roadmap") or master_data.get("atomic_action_plan") or []

            # ทำให้มั่นใจว่า Phases เป็น List of Dict เสมอ
            normalized_phases = []
            if isinstance(raw_phases, list):
                for i, p in enumerate(raw_phases, 1):
                    if isinstance(p, dict):
                        normalized_phases.append(p)
                    else:
                        normalized_phases.append({"step": f"Phase {i}", "action": str(p)})
            elif isinstance(raw_phases, str):
                normalized_phases.append({"step": "General Action", "action": raw_phases})

            self.logger.info(f"✅ [MASTER-ROADMAP] Synthesis Success for {sub_id}")
            
            return {
                "sub_id": sub_id,
                "sub_criteria_name": sub_criteria_name,
                "overall_strategy": final_strategy,
                "phases": normalized_phases,
                "status": "SUCCESS",
                "generated_at": datetime.now().isoformat(),
                "source_insights_count": len(aggregated_insights),
                "maturity_score": master_data.get("score") # ถ้ามีใน Prompt
            }

        except Exception as e:
            self.logger.error(f"💥 Critical error in Master Roadmap {sub_id}: {str(e)}", exc_info=True)
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
    
    # ------------------------------------------------------------------------------------------
    # 🧠 [TIER-1 & TIER-2 WORKER] Sequential Assessment (HYDRATED) - FULL REVISED
    # ------------------------------------------------------------------------------------------
    def _run_sub_criteria_assessment_worker(
        self,
        sub_criteria: Dict[str, Any],
        vectorstore_manager: Optional[Any] = None,
        initial_baseline: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
        """
        [PRODUCTION READY - v2026.3.28]
        แก้ไข Bug: ส่ง keyword_guide ให้ Tier-1 Assessment
        แก้ไข Flow: มั่นใจว่าข้อมูลไหลเข้าสู่ Tier-2 (Atomic) และ Tier-3 (Master) อย่างถูกต้อง
        """
        # 1. ข้อมูลพื้นฐาน
        sub_id = str(sub_criteria.get("sub_id", "Unknown"))
        sub_name = sub_criteria.get("sub_criteria_name", "No Name")
        sub_weight = float(sub_criteria.get("weight", 0.0))
        
        # ป้องกัน AttributeError ถ้า self.config ไม่มี target_level
        target_limit = getattr(self.config, "target_level", 5) if hasattr(self, 'config') else 5
        enabler = getattr(self, "enabler", "KM")

        vsm = vectorstore_manager or getattr(self, "vectorstore_manager", None)
        current_highest_level = 0
        level_details = {}
        roadmap_input_bundle = []

        # 2. Evidence Hydration Memory
        baseline_memory = {sub_id: list(initial_baseline or [])}
        levels = sorted(sub_criteria.get("levels", []), key=lambda x: x.get("level", 0))

        self.logger.info(f"🚀 [START-SUB] {sub_id} | Target Level: {target_limit}")

        for stmt in levels:
            level = int(stmt.get("level", 0))
            if level == 0 or level > target_limit: 
                continue

            # --- 🔥 STEP 1: Core Assessment (Tier-1) ---
            # แก้ไข Bug: เพิ่ม keyword_guide ให้ตรงตาม Parameter requirements
            res = self._run_single_assessment(
                sub_id=sub_id, 
                level=level,
                criteria={
                    "name": sub_name, 
                    "statement": stmt.get("statement", ""), 
                    "sub_criteria_name": sub_name
                },
                keyword_guide=stmt.get("keywords", []), # ✅ FIXED: ส่ง Keywords จาก rubric
                baseline_evidences=baseline_memory.get(sub_id, []),
                vectorstore_manager=vsm,
            )

            is_passed = bool(res.get("is_passed", False))
            
            # 🔄 Evidence Hydration (ส่งต่อหลักฐาน)
            if is_passed:
                current_highest_level = max(current_highest_level, level)
                new_chunks = res.get("top_chunks_data", [])
                if new_chunks:
                    baseline_memory[sub_id].extend(new_chunks)
                    # เก็บเฉพาะ 5 Chunks ล่าสุดเพื่อคุม Token
                    baseline_memory[sub_id] = baseline_memory[sub_id][-5:]

            # --- 🔥 STEP 2: Atomic Action Plan (Tier-2 ราย Level) ---
            # สร้าง Feedback ทันทีเพื่อให้ User เห็นแผนพัฒนารายระดับ
            self.logger.info(f"🛠️ [ATOMIC] Level {level} for {sub_id}")
            
            atomic_actions = self.create_atomic_action_plan(
                insight=res.get("coaching_insight", ""),
                level=level
            )

            # เก็บรายละเอียดราย Level ลงโครงสร้างข้อมูลหลัก
            level_details[str(level)] = {
                "level": level, 
                "is_passed": is_passed, 
                "score": float(res.get("score", 0.0)),
                "reason": res.get("reason", ""),
                "coaching_insight": res.get("coaching_insight", ""),
                "atomic_action_plan": atomic_actions, 
                "pdca_breakdown": res.get("pdca_breakdown", {}),
                "audit_confidence": res.get("audit_confidence", {})
            }

            # 📦 สะสมข้อมูลส่งต่อให้ Master Roadmap (Tier-3)
            roadmap_input_bundle.append({
                "level": level,
                "status": "PASSED" if is_passed else "FAILED",
                "insight_summary": res.get("coaching_insight", "")[:200]
            })

        # --- 🔥 STEP 3: Strategic Master Roadmap (Tier-3 สังเคราะห์ภาพรวม) ---
        # รวบรวม Insights ทั้งหมดมาสร้าง Phase พัฒนาในระยะยาว
        self.logger.info(f"🔮 [MASTER] Synthesis for {sub_id}")
        
        master_roadmap = self.generate_master_roadmap(
            sub_id=sub_id,
            sub_criteria_name=sub_name,
            enabler=enabler,
            aggregated_insights=roadmap_input_bundle
        )

        # 4. Final Output Assembly
        # ข้อมูลชุดนี้จะถูกส่งไปที่ Transformer และ UI React
        return {
            "sub_id": sub_id, 
            "sub_criteria_name": sub_name, 
            "highest_full_level": current_highest_level,
            "weighted_score": round(current_highest_level * sub_weight, 2),
            "is_passed": current_highest_level >= 1, # <--- เพิ่มตัวนี้
            "level_details": level_details, 
            "master_roadmap": master_roadmap 
        }, baseline_memory

    # ------------------------------------------------------------------------------------------
    # 🧠 [TIER-1 CORE] _run_single_assessment (GOVERNANCE-LOCKED) - REVISED v2026.1.22
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
        [ULTIMATE VERSION - FULL REVISED v2026.1.25]
        - Robust retrieval + evidence fusion
        - Multi-channel context + LLM evaluation
        - Smart rescue in post-process
        - Judicial Review (appeal) with safety net force-pass
        - Enhanced logging & traceability
        """
        log_prefix = f"Sub:{sub_id} L{level}"
        self.logger.info(f"🔍 [START-ASSESSMENT] {log_prefix} | {criteria.get('name', '')[:50]}...")

        # ------------------------------------------------------------------
        # STEP 1-2: Adaptive Retrieval & Evidence Fusion
        # ------------------------------------------------------------------
        retrieved_chunks, max_rerank = self._perform_adaptive_retrieval(
            sub_id=sub_id,
            level=level,
            stmt=criteria.get("statement", ""),
            vectorstore_manager=vectorstore_manager,
        )

        # Diversity Filter
        retrieved_chunks = self._apply_diversity_filter(retrieved_chunks, level)

        # Log preview ของ chunks ที่ได้ (ช่วย debug)
        chunk_count = len(retrieved_chunks)
        top_preview = retrieved_chunks[0].get('text', '')[:80] + "..." if retrieved_chunks else "No chunks"
        self.logger.debug(f"[RETRIEVAL] {log_prefix} | Chunks: {chunk_count} | Max Rerank: {max_rerank:.4f} | Top: {top_preview}")

        # Evidence Fusion (hydration)
        evidences = []
        evidences.extend(baseline_evidences or [])
        evidences.extend(retrieved_chunks or [])

        # ------------------------------------------------------------------
        # STEP 3-5: Metadata & Audit Preparation
        # ------------------------------------------------------------------
        pdca_blocks = self._get_pdca_blocks_from_evidences(
            evidences=evidences,
            baseline_evidences=baseline_evidences,
            level=level,
            sub_id=sub_id,
            contextual_rules_map=self.contextual_rules_map
        )

        audit_confidence = self.calculate_audit_confidence(
            matched_chunks=retrieved_chunks,
            sub_id=sub_id,
            level=level,
        )
        self.current_audit_meta = audit_confidence

        # ------------------------------------------------------------------
        # STEP 6-8: Multi-channel LLM Execution
        # ------------------------------------------------------------------
        llm_context = self._build_multichannel_context_for_level(
            level=level,
            top_evidences=retrieved_chunks,
            previous_levels_evidence=baseline_evidences
        )

        # Standard Evaluation (LLM ครั้งแรก)
        llm_raw = self.evaluate_pdca(
            pdca_blocks=pdca_blocks,
            sub_id=sub_id,
            level=level,
            audit_confidence=audit_confidence
        )
        if not isinstance(llm_raw, dict):
            self.logger.warning(f"[LLM-RAW] {log_prefix} | Invalid LLM output → fallback empty dict")
            llm_raw = {}

        # ------------------------------------------------------------------
        # STEP 9: Smart Rescue & Normalization
        # ------------------------------------------------------------------
        current_rules = self.contextual_rules_map.get(sub_id, {}).get(f"L{level}", {})
        
        result = self.post_process_llm_result(
            llm_output=llm_raw,
            level=level,
            sub_id=sub_id,
            contextual_config=current_rules,
            top_evidences=retrieved_chunks
        )

        # ------------------------------------------------------------------
        # STEP 10: Expert Re-evaluation (Judicial Review) + SAFETY NET
        # ------------------------------------------------------------------
        is_safety_pass = False
        if not result.get("is_passed") and max_rerank >= 0.70:  # 🟢 [ADJUSTED] ลด threshold เพื่อ trigger ง่ายขึ้น
            self.logger.info(f"⚖️ [TRIGGER-APPEAL] {log_prefix} | Rerank {max_rerank:.4f} ≥ 0.70 → Starting Judicial Review")

            base_kwargs = {
                "pdca_blocks": pdca_blocks,
                "contextual_config": current_rules,
                "top_evidences": retrieved_chunks
            }

            appeal_result = self._run_expert_re_evaluation(
                sub_id=sub_id,
                level=level,
                statement_text=criteria.get("statement", ""),
                context=str(llm_context.get("full_context", "")),
                first_attempt_reason=result.get("reason", "หลักฐานไม่ชัดเจน"),
                missing_tags=result.get("missing_phases", []),
                highest_rerank_score=max_rerank,
                sub_criteria_name=criteria.get("name", sub_id),
                llm_evaluator_to_use=self.evaluate_pdca,
                base_kwargs=base_kwargs
            )

            # 🚨 [SAFETY NET] Force pass ถ้า appeal granted
            if appeal_result and appeal_result.get("appeal_status") == "GRANTED":
                self.logger.info(f"⚖️ [APPEAL-FORCE-PASS] {log_prefix} | Judicial Review granted → Force score ≥ 1.2")
                
                appeal_result["score"] = max(appeal_result.get("score", 0.0), 1.2)
                appeal_result["is_passed"] = True
                appeal_result["is_safety_pass"] = True
                appeal_result["is_force_pass"] = True
                appeal_result["reason"] = f"{appeal_result.get('reason', '')} [ผ่านจากการอุทธรณ์โดย Judicial Review]"

                # อัปเดต coaching insight ให้สะท้อน appeal
                if "coaching_insight" in appeal_result:
                    appeal_result["coaching_insight"] += " (ผ่านด้วยการพิจารณา substance over form)"

            # ถ้า appeal ให้ผลผ่าน → update result ด้วย post-process อีกครั้ง
            if appeal_result and appeal_result.get("is_passed"):
                final_appeal = self.post_process_llm_result(
                    llm_output=appeal_result,
                    level=level,
                    sub_id=sub_id,
                    contextual_config=current_rules,
                    top_evidences=retrieved_chunks
                )
                result.update(final_appeal)
                is_safety_pass = True
                self.logger.info(f"✅ [APPEAL-SUCCESS] {log_prefix} passed via Judicial Review | Final score: {result.get('score', 0.0):.2f}")

        # ------------------------------------------------------------------
        # STEP 11: Final Insight Refinement
        # ------------------------------------------------------------------
        final_insight = (
            result.get("coaching_insight") or
            result.get("reason") or
            "ไม่มีข้อเสนอแนะเพิ่มเติมจากระบบ"
        ).strip()

        if result.get("is_passed"):
            final_insight = f"[STRENGTH] {final_insight}"
        else:
            final_insight = f"[GAP] {final_insight}"

        # ------------------------------------------------------------------
        # STEP 12: Logging & Final Assembly
        # ------------------------------------------------------------------
        if hasattr(self, "_log_pdca_status"):
            self._log_pdca_status(
                sub_id=sub_id,
                name=criteria.get("name", "Unknown"),
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

        # Final return with enhanced debug_meta
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
                "appeal_triggered": max_rerank >= 0.70,
                "retrieval_chunks": chunk_count,
                "top_chunk_preview": top_preview
            }
        }