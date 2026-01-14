# core/seam_assessment.py

import sys
import json
import logging
import time
import os
from typing import List, Dict, Any, Optional, Union, Tuple, Set, Final, Literal
from collections import defaultdict, OrderedDict
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
from .json_extractor import _robust_extract_json
from filelock import FileLock  # ต้องติดตั้ง: pip install filelock
import re
import hashlib
import copy
from database import init_db
from database import db_update_task_status as update_db
from pydantic import BaseModel
import random  # Added for shuffle

# -------------------- PATH SETUP & IMPORTS --------------------
try:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    # 1. Import Constants จาก global_vars
    from config.global_vars import (
        MAX_LEVEL,
        EVIDENCE_DOC_TYPES,
        RERANK_THRESHOLD,
        MAX_EVI_STR_CAP,
        DEFAULT_LLM_MODEL_NAME,
        LLM_TEMPERATURE,
        MIN_RETRY_SCORE,
        MAX_PARALLEL_WORKERS,
        PDCA_PRIORITY_ORDER,
        TARGET_DEVICE,
        PDCA_PHASE_MAP,
        INITIAL_TOP_K,
        FINAL_K_RERANKED,
        MAX_CHUNKS_PER_FILE,
        MAX_CHUNKS_PER_BLOCK,
        MATURITY_LEVEL_GOALS
    )
    
    # 2. Import Logic Functions
    from core.llm_data_utils import ( 
        create_structured_action_plan, evaluate_with_llm,
        retrieve_context_with_filter, retrieve_context_for_low_levels,
        evaluate_with_llm_low_level, LOW_LEVEL_K, 
        set_mock_control_mode as set_llm_data_mock_mode,
        create_context_summary_llm,
        retrieve_context_by_doc_ids,
        _fetch_llm_response,
        build_multichannel_context_for_level,
        _get_emergency_fallback_plan,
        _check_and_handle_empty_context
    )
    from core.vectorstore import VectorStoreManager, load_all_vectorstores, get_global_reranker 
    from core.action_plan_schema import ActionPlanActions, ActionPlanResult

    # 3. 🎯 Import Path Utilities
    from utils.path_utils import (
        get_mapping_file_path, 
        get_evidence_mapping_file_path, 
        get_contextual_rules_file_path,
        get_doc_type_collection_key,
        get_assessment_export_file_path,
        get_export_dir,
        get_rubric_file_path,
        load_evidence_mapping,
        _n
    )

    import assessments.seam_mocking as seam_mocking 
    
except ImportError as e:
    # -------------------- Modernized Fallback Code --------------------
    print(f"⚠️ WARNING: Import failed, using dynamic fallback for Mac. Error: {e}", file=sys.stderr)
    
    # Fallback Constants
    EXPORTS_DIR = "exports"
    MAX_LEVEL = 5
    INITIAL_LEVEL = 1
    QA_FINAL_K = 3
    RUBRIC_FILENAME_PATTERN = "{tenant}_{enabler}_rubric.json"
    DEFAULT_ENABLER = "KM"
    EVIDENCE_DOC_TYPES = "evidence"
    INITIAL_TOP_K = 10

    # 📌 Placeholder functions for path_utils (ชี้เข้า data_store ตรงๆ)
    def _n(s): return str(s).lower().strip()

    def get_mapping_file_path(doc_type, tenant, year=None, enabler=None):
        t = _n(tenant)
        if _n(doc_type) == "evidence":
            return f"data_store/{t}/mapping/{year}/{t}_{year}_{_n(enabler)}_doc_id_mapping.json"
        return f"data_store/{t}/mapping/{t}_{_n(doc_type)}_doc_id_mapping.json"

    def get_evidence_mapping_file_path(tenant, year, enabler):
        t = _n(tenant)
        return f"data_store/{t}/mapping/{year}/{t}_{year}_{_n(enabler)}_evidence_mapping.json"

    def get_contextual_rules_file_path(tenant, enabler):
        t = _n(tenant)
        return f"data_store/{t}/config/{t}_{_n(enabler)}_contextual_rules.json"

    def get_rubric_file_path(tenant, enabler):
        t = _n(tenant)
        return f"data_store/{t}/config/{t}_{_n(enabler)}_rubric.json"

    def load_evidence_mapping(tenant="pea", year=2568, enabler="KM"):
        path = get_evidence_mapping_file_path(tenant, year, enabler)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f: return json.load(f)
            except: return {}
        return {}

    # Mock Logic Functions
    def create_structured_action_plan(*args, **kwargs): return []
    def evaluate_with_llm(*args, **kwargs): return {"score": 0, "reason": "Import Error Fallback", "is_passed": False}
    def retrieve_context_with_filter(*args, **kwargs): return {"top_evidences": [], "aggregated_context": ""}
    def retrieve_context_for_low_levels(*args, **kwargs): return {"top_evidences": [], "aggregated_context": ""}
    def evaluate_with_llm_low_level(*args, **kwargs): return {"score": 0, "is_passed": False}
    def set_llm_data_mock_mode(mode): pass
    def build_multichannel_context_for_level(*args, **kwargs): return ""
    
    class VectorStoreManager: pass
    def load_all_vectorstores(*args, **kwargs): return None
    
    class ActionPlanActions:
        @staticmethod
        def generate(*args, **kwargs): return []
    
    class ActionPlanResult:
        def __init__(self): self.success = False

    PDCA_PHASE_MAP = {
        1: "Plan (การกำหนดเป้าหมายและนโยบาย)",
        2: "Do (การนำแผนไปปฏิบัติและขับเคลื่อน)",
        3: "Check (การติดตามและประเมินผล)",
        4: "Act (การปรับปรุงและสร้างนวัตกรรม)",
        5: "Sustainability (ความยั่งยืนและต้นแบบที่ดี)"
    }

    MATURITY_LEVEL_GOALS = {
        1: "เน้นการเริ่มต้น มีนโยบาย หรือมีแนวทางปฏิบัติเบื้องต้น",
        2: "เน้นการนำไปใช้อย่างเป็นระบบ มีคณะทำงาน",
        3: "เน้นการปฏิบัติอย่างต่อเนื่อง และเห็นผลลัพธ์ชัดเจน",
        4: "เน้นการวิเคราะห์ข้อมูลเชิงสถิติ หรือสร้างนวัตกรรม",
        5: "เน้นความยั่งยืนและการเป็นต้นแบบ (Role Model)"
    }

    def _get_emergency_fallback_plan(sub_id, name, level, *args, **kwargs):
        return {"summary": f"Fallback plan for {sub_id}", "steps": []}
    
    class seam_mocking:
        @staticmethod
        def set_mock_control_mode(mode): pass
    
    def create_context_summary_llm(*args, **kwargs): 
        return {"summary": "ไม่สามารถสรุปได้เนื่องจากระบบโหลด Module พัง", "coaching": "โปรดตรวจสอบการ Import"}
    
    def _fetch_llm_response(*args, **kwargs): 
        return "{}"

    # เพิ่มตัวแปรเหล่านี้ด้วยครับ เพราะใน _run_single_assessment มีการเรียกใช้
    MAX_EVI_STR_CAP = 10.0
    RERANK_THRESHOLD = 0.35

    if "FATAL ERROR" in str(e):
        pass 
# ----------------------------------------------------------------------

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

def get_actual_score(ev: dict) -> float:
    """
    [v2026.1 - ROBUST SCORING]
    - แก้ไขปัญหา '0.0 or score' ที่ทำให้ Logic เพี้ยน
    - รองรับทั้ง Dict และ Langchain Document Object
    """
    if not ev:
        return 0.0

    # 1. รวบรวมค่าจากระดับ Top-level
    # ใช้ next(...) เพื่อหาค่าแรกที่ไม่ใช่ None
    score_keys = ["relevance_score", "rerank_score", "score"]
    
    # ดึงค่าจาก Dict หรือ Object attribute
    val = None
    for key in score_keys:
        val = ev.get(key) if isinstance(ev, dict) else getattr(ev, key, None)
        if val is not None:
            return float(val)

    # 2. Fallback ไปที่ Metadata (เผื่อเก็บไว้ใน ChromaDB)
    meta = ev.get("metadata", {}) if isinstance(ev, dict) else getattr(ev, "metadata", {})
    if meta:
        for key in score_keys:
            val = meta.get(key)
            if val is not None:
                return float(val)

    return 0.0

def _static_worker_process(worker_input_tuple: Tuple) -> Any:
    """
    [ULTIMATE WORKER v2026.3] Isolated Execution for Parallel Assessment
    ---------------------------------------------------------------------
    - สร้างสภาพแวดล้อมใหม่ทั้งหมดในแต่ละ Process (Zero Memory Leak)
    - บังคับใช้ Context จากแม่เพื่อให้ผลการประเมินถูกต้องแม่นยำ
    - ป้องกันระบบพังด้วย Fallback Dictionary
    """

    # 1. 📂 PATH SETUP
    # ตรวจสอบว่ามองเห็นโมดูลหลัก เพื่อให้ Import AssessmentConfig และ Engine ได้
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.append(project_root)
        
    worker_logger = logging.getLogger(f"Worker_{os.getpid()}")

    # 2. 📦 ROBUST UNPACKING
    try:
        (
            sub_criteria_data, enabler, target_level, mock_mode, 
            evidence_map_path, model_name, temperature,
            min_retry_score, max_retrieval_attempts, document_map, 
            action_plan_model, year, tenant
        ) = worker_input_tuple
        
        sub_id = sub_criteria_data.get('sub_id', 'UNKNOWN')
        worker_logger.info(f"⚙️ PID:{os.getpid()} | Starting: {sub_id} ({tenant}/{year})")
        
    except Exception as e:
        return {"error": f"Worker unpacking failed: {str(e)}", "status": "critical_failure"}
        
    # 3. 🏗️ RECONSTRUCT ISOLATED ENGINE
    try:
        # สร้าง Config เฉพาะสำหรับ Worker ตัวนี้
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

        # คืนชีพ Engine (จะเข้าสู่ __init__ ที่เราเพิ่ง Patch ความปลอดภัยไป)
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
        worker_logger.error(f"❌ Worker initialization failed for {sub_id}: {e}")
        return {"sub_id": sub_id, "error": f"Init Error: {str(e)}", "status": "failed"}

    # 4. ⚡ EXECUTE & TIME TRACKING
    try:
        start_time = time.time()
        
        # รันการประเมินรายข้อ (Core Logic)
        result = worker_instance._run_sub_criteria_assessment_worker(sub_criteria_data)
        
        elapsed = time.time() - start_time
        worker_logger.info(f"✅ PID:{os.getpid()} | Finished: {sub_id} in {elapsed:.2f}s")
        
        return result
        
    except Exception as e:
        worker_logger.error(f"❌ Execution error for {sub_id}: {str(e)}")
        return {
            "sub_id": sub_id,
            "error": str(e),
            "status": "failed",
            "execution_time": 0
        }
    
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

        self.enabler_id = config.enabler
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
                enabler=self.enabler_id
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
                enabler=self.enabler_id if is_evi_mode else None
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
        self.llm_evaluator = evaluate_with_llm
        self.rag_retriever = retrieve_context_with_filter
        self.create_structured_action_plan = create_structured_action_plan
        self.ActionPlanActions = ActionPlanActions
        self.action_plan_model = ActionPlanResult

        self.logger.info(f"✅ Engine Initialized: Ready for Assessment (Sub-ID: {self.sub_id})")
    

    # =================================================================
    # DB Proxy Methods
    # =================================================================
    def db_update_task_status(self, record_id: str, progress: int, message: str, status: str = "RUNNING"):
        """
        Wrapper สำหรับอัปเดตสถานะผ่าน Database Module
        - record_id: ID ของการประเมินในรอบนั้น
        - progress: 0-100
        - message: ข้อความแสดงสถานะ
        """
        if not record_id: return
        try:
            # ใช้ update_db ที่ alias มาจาก database.db_update_task_status
            update_db(record_id, progress, message, status=status)
            self.logger.debug(f"[DB-PROGRESS] {record_id}: {progress}% - {message}")
        except Exception as e:
            self.logger.error(f"❌ DB Update Error: {e}")


    def get_rule_content(self, sub_id: str, level: int, key_type: str):
        """
        [ULTIMATE RULE ENGINE v2026.3]
        ดึงข้อมูลเกณฑ์การประเมินจาก Contextual Rules แบบลำดับชั้น
        - รองรับ Priority: Specific Level > Sub-ID Root > Global Defaults
        """
        # ดึงข้อมูลกฎของ Sub-ID นั้นๆ
        rule = self.contextual_rules_map.get(sub_id, {})
        level_key = f"L{level}"
        
        # 1. 🥇 Priority 1: ข้อมูลระดับ Level เจาะจง (e.g., 1.1 -> L3)
        level_data = rule.get(level_key, {})
        if key_type in level_data:
            return level_data[key_type]
        
        # 2. 🥈 Priority 2: ข้อมูลระดับ Sub-ID Root (กฎที่ใช้ร่วมกันทุก Level ในข้อนั้น)
        if key_type in rule:
            return rule[key_type]
        
        # 3. 🥉 Priority 3: Global Defaults (ค่ากลางของ Enabler)
        # เน้นกลุ่ม keywords และ required_phases มาตรฐาน
        defaults = self.contextual_rules_map.get("_enabler_defaults", {})
        if key_type in defaults:
            return defaults[key_type]
            
        # 4. 🛡️ Fallback: คืนค่าว่างตามประเภทข้อมูล
        if key_type == "require_phase":
            return None # เพื่อให้ caller เช็ค if phase is not None ได้
        if "keywords" in key_type:
            return [] # คืน list ว่างป้องกัน loop พัง
            
        return ""


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
            target_enabler = str(self.enabler_id).lower() if self.enabler_id else None
            
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

    def get_cumulative_rules(self, sub_id: str, current_level: int) -> Dict[str, Any]:
        """
        [REVISED CUMULATIVE RULES ENGINE v2026.8 - PRIORITY & SMART ACCUMULATION]
        ----------------------------------------------------------------------------
        - รวบรวมกฎสะสมจาก L1 → current_level โดยให้ความสำคัญกับ level สูงกว่า
        - ใช้ OrderedDict เพื่อรักษาลำดับ (specific > default)
        - Required phases คำนวณจาก level สูงสุด (maturity-driven)
        - แยก instructions เป็น dict {level: rule} + string รวมทั้งหมด
        - Logging แยก level: info สำหรับ summary, debug สำหรับ detail
        - Fallback ปลอดภัยถ้าไม่มี rules สำหรับ sub_id นั้น

        Args:
            sub_id (str): รหัส sub-criteria เช่น "1.2"
            current_level (int): ระดับ maturity ปัจจุบัน (1-5)

        Returns:
            Dict[str, Any]: {
                "plan_keywords": List[str],
                "do_keywords": List[str],
                "check_keywords": List[str],
                "act_keywords": List[str],
                "required_phases": List[str],          # sorted, unique
                "level_specific_instructions": Dict[int, str],
                "all_instructions": str,               # รวมทั้งหมดสำหรับ prompt
                "source_summary": str                  # สรุปว่ามาจาก level ไหนบ้าง
            }
        """
        # 1. ดึง defaults เป็นฐาน (fallback ถ้าไม่มี rules เฉพาะ)
        defaults = self.contextual_rules_map.get('_enabler_defaults', {})
        
        # ใช้ OrderedDict เพื่อรักษาลำดับ: default → L1 → L2 → ... → current_level
        cum_keywords = {
            "plan": OrderedDict((k, None) for k in defaults.get('plan_keywords', [])),
            "do":   OrderedDict((k, None) for k in defaults.get('do_keywords', [])),
            "check": OrderedDict((k, None) for k in defaults.get('check_keywords', [])),
            "act":  OrderedDict((k, None) for k in defaults.get('act_keywords', []))
        }

        required_phases: Set[str] = set()
        level_specific_instructions: Dict[int, str] = {}
        source_levels: List[int] = []

        # 2. ดึง rules เฉพาะของ sub_id
        sub_rules = self.contextual_rules_map.get(sub_id, {})
        if not sub_rules:
            logger.warning(f"[RULE_CUMULATIVE] No specific rules for {sub_id} → using defaults only")
        
        # 3. สะสมจาก L1 ถึง current_level (ให้ level สูงกว่าทับลำดับ)
        for lv in range(1, current_level + 1):
            lv_key = f"L{lv}"
            level_rule = sub_rules.get(lv_key, {})
            
            if not level_rule:
                continue  # ข้าม level ที่ไม่มี rules
            
            source_levels.append(lv)
            
            # อัปเดต keywords (level สูงกว่าทับ default และ level ต่ำกว่า)
            for phase, key_name in [("plan", "plan_keywords"), ("do", "do_keywords"),
                                   ("check", "check_keywords"), ("act", "act_keywords")]:
                new_kws = level_rule.get(key_name, [])
                for kw in new_kws:
                    cum_keywords[phase][kw] = None  # OrderedDict จะรักษาลำดับ
            
            # อัปเดต required phases (union ทุก level)
            if 'require_phase' in level_rule:
                required_phases.update(level_rule['require_phase'])
            
            # เก็บ specific rule ตาม level
            specific = level_rule.get('specific_contextual_rule')
            if specific:
                level_specific_instructions[lv] = specific.strip()

        # 4. แปลง OrderedDict → list (รักษาลำดับ)
        result_keywords = {phase: list(cum_keywords[phase].keys()) for phase in cum_keywords}

        # 5. Required phases: ใช้จาก level สูงสุดเป็นหลัก (ถ้า L5 ต้องการ P,D,C,A → ใช้ทั้งหมด)
        # แต่ยังคง union เพื่อความปลอดภัย
        final_required = sorted(list(required_phases)) if required_phases else []

        # 6. รวม instructions เป็น string (เพิ่มหัวข้อเพื่อให้ LLM โฟกัสถูกจุด)
        instructions_lines = ["ข้อกำหนดเฉพาะสำหรับการพิจารณาระดับ Maturity:"]
        for lv in sorted(level_specific_instructions.keys()):
            prefix = "🎯 " if lv == current_level else "✅ " # เน้นเลเวลปัจจุบัน
            instructions_lines.append(f"{prefix}ระดับ L{lv}: {level_specific_instructions[lv]}")
        all_instructions = "\n".join(instructions_lines)

        # 7. สรุป source สำหรับ debug
        source_str = f"from levels {source_levels}" if source_levels else "defaults only"

        # 8. Logging (info สำหรับ production monitoring, debug สำหรับ trace)
        logger.info(
            f"[RULE_CUMULATIVE] {sub_id} L{current_level} | "
            f"Keywords (dedup): P={len(result_keywords['plan'])} | "
            f"D={len(result_keywords['do'])} | C={len(result_keywords['check'])} | "
            f"A={len(result_keywords['act'])} | Required={final_required} | "
            f"Instructions={len(level_specific_instructions)} | Source={source_str}"
        )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"[RULE_DETAIL] {sub_id} L{current_level} | "
                f"Plan keywords sample: {result_keywords['plan'][:5]}... | "
                f"Required phases accumulated from levels: {sorted(source_levels)}"
            )

        # 9. Return structure ที่สมบูรณ์และใช้งานง่าย
        return {
            "plan_keywords": result_keywords["plan"],
            "do_keywords": result_keywords["do"],
            "check_keywords": result_keywords["check"],
            "act_keywords": result_keywords["act"],
            "required_phases": final_required,
            "level_specific_instructions": level_specific_instructions,  # Dict[int, str]
            "all_instructions": all_instructions,                       # string รวม
            "source_summary": source_str,
            "accumulated_levels": sorted(source_levels)
        }


    def validate_accumulative_pass(self, llm_result: Dict[str, Any], target_phases: List[str]) -> Tuple[bool, str]:
        """
        ตรวจสอบว่า Phase ที่ AI ตรวจเจอ ครอบคลุมสิ่งที่ Level นั้นต้องการสะสมหรือไม่
        """
        # ดึงเฉพาะ Phase ที่ AI ให้คะแนน > 0 (คือเจอจริง)
        pdca_breakdown = llm_result.get('pdca_breakdown', {})
        found_phases = {phase for phase, score in pdca_breakdown.items() if score > 0}
        required_set = set(target_phases)

        # ตรวจสอบความเป็น Subset (Required ต้องอยู่ใน Found)
        is_subset = required_set.issubset(found_phases)
        
        if not is_subset:
            missing = required_set - found_phases
            error_msg = f"ขาดหลักฐานในเฟสสำคัญ: {', '.join(missing)} (ตามเกณฑ์สะสม Maturity)"
            return False, error_msg
        
        return True, ""
    

    def _check_maturity_consistency(
        self, 
        sub_id: str, 
        current_level: int, 
        top_evidences: List[Dict[str, Any]],
        llm_pdca_breakdown: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        ตรวจสอบความสอดคล้องเชิง Maturity:
        1. Phase ต้องครบตาม JSON require_phase (สะสม)
        2. คุณภาพหลักฐาน (Rerank) ของ Phase บังคับต้องถึงเกณฑ์
        """
        # ดึงกฎสะสมจาก JSON
        cum_rules = self.get_cumulative_rules(sub_id, current_level)
        required_phases = set(cum_rules['phases'])
        
        # ดึง Phase ที่ AI ตรวจเจอจริง (score > 0)
        found_phases = {p for p, score in llm_pdca_breakdown.items() if score > 0}
        
        # 1. ตรวจสอบ Phase Gap
        missing_phases = required_phases - found_phases
        
        # 2. ตรวจสอบคุณภาพหลักฐานราย Phase (Critical Evidence Check)
        # เช่น ถ้า L3 require 'D' ต้องมีหลักฐาน D ที่ Rerank Score สูงพอ
        low_quality_phases = []
        for phase in required_phases:
            max_score = max([
                doc.get('rerank_score', 0.0) 
                for doc in top_evidences 
                if doc.get('pdca_tag') == phase
            ], default=0.0)
            
            if max_score < 0.4: # Threshold ขั้นต่ำสำหรับหลักฐานที่ "เชื่อถือได้"
                low_quality_phases.append(phase)

        return {
            "is_consistent": len(missing_phases) == 0 and len(low_quality_phases) == 0,
            "missing_phases": list(missing_phases),
            "low_quality_phases": low_quality_phases,
            "required_phases": list(required_phases)
        }

    def post_process_llm_result(
        self,
        llm_output: Any,
        level: int,
        sub_id: str = None
    ) -> Dict[str, Any]:
        """
        [POST-PROCESS v2026.Expert - FULL INTEGRATION]
        - FIXED: คะแนน PDCA เป็น 0 (เพิ่ม Force Mapping จาก Extraction)
        - FIXED: Floor Rescue Mapping (ปรับคะแนนให้ถึงเกณฑ์ถ้าโดนสั่งผ่าน)
        - FEATURE: Maturity-based Threshold Validation
        """
        log_prefix = f"{sub_id or 'Unknown'} L{level}"
        
        # 1. 🛠️ JSON Repair & Unpacking
        # รองรับทั้ง String JSON และ Dict (ป้องกัน Tuple Error จากต้นทาง)
        if isinstance(llm_output, tuple):
            llm_output = llm_output[0] if len(llm_output) > 0 else {}

        if isinstance(llm_output, str):
            try:
                # Clean up problematic symbols common in LLM outputs
                cleaned_str = re.sub(r'(\d+\.?\d*)\s*[\+\-]\s*(\d+\.?\d*)\s*=\s*(\d+\.?\d*)', r'\3', llm_output)
                cleaned_str = cleaned_str.strip().replace(",\n}", "\n}").replace(",}", "}")
                llm_output = json.loads(cleaned_str)
            except Exception as e:
                self.logger.error(f"❌ JSON Repair failed for {log_prefix}: {str(e)}")
                return {"is_passed": False, "score": 0.0, "reason": "Hard JSON Parsing Error"}

        if not isinstance(llm_output, dict):
            return {"is_passed": False, "score": 0.0, "reason": "Invalid Output Format"}

        # 2. 🛡️ Floor Rescue Awareness
        # ตรวจสอบว่า Engine ส่วนหน้า (Single Assessment) สั่ง Override ให้ผ่านหรือไม่
        is_overridden = llm_output.get('is_passed', False)

        # 3. 📊 Score & PDCA Extraction (Heuristic Recovery)
        # แมปชื่อ Key ที่ AI อาจจะตอบมาผิดเพี้ยนให้กลับเข้าสู่มาตรฐาน P-D-C-A
        extraction_map = {
            "P": ["P_Plan_Score", "score_p", "Plan_Score"],
            "D": ["D_Do_Score", "score_d", "Do_Score"],
            "C": ["C_Check_Score", "score_c", "Check_Score"],
            "A": ["A_Act_Score", "score_a", "Act_Score"]
        }

        pdca_results = {"P": 0.0, "D": 0.0, "C": 0.0, "A": 0.0}

        for phase, possible_keys in extraction_map.items():
            # พยายามดึงคะแนนจากหลายๆ Key ที่ AI มักจะชอบใช้
            score = 0.0
            for key in possible_keys:
                val = llm_output.get(key)
                if val is not None:
                    try:
                        score = float(val)
                        break
                    except: continue
            
            # 🛡️ Protection: ถ้าคะแนนเป็น 0 แต่มีข้อความ Extraction ยาวๆ (แปลว่าเจอหลักฐานแต่ AI ลืมให้คะแนน)
            # เราจะกู้คืนให้ 0.5 คะแนน (เฉพาะ L1-L2 หรือ Overridden)
            ext_text = str(llm_output.get(f"Extraction_{phase}", "")).strip()
            if score == 0.0 and len(ext_text) > 15 and "ไม่พบ" not in ext_text:
                if is_overridden or level <= 2:
                    self.logger.info(f"🛡️ [RECOVERY] Found evidence text for {phase} in {log_prefix}. Assigning 0.5")
                    score = 0.5
            
            pdca_results[phase] = score

        # 4. ⚖️ Maturity Threshold Calculation
        p, d, c, a = pdca_results["P"], pdca_results["D"], pdca_results["C"], pdca_results["A"]
        pdca_sum = round(p + d + c + a, 2)
        
        # เกณฑ์คะแนนตามระดับความยาก (Threshold)
        threshold_map = {1: 1.0, 2: 2.0, 3: 4.0, 4: 6.0, 5: 8.0}
        threshold = threshold_map.get(level, 2.0)

        # 5. 🏁 Final Decision Logic
        is_passed = pdca_sum >= threshold
        fail_reason = llm_output.get('reason') or llm_output.get('fail_reason') or ""

        # กฎผ่อนปรน L1-L2 หรือกรณี Floor Rescue
        if is_overridden or (level <= 2 and p > 0 and d > 0):
            is_passed = True
            # ถ้าสถานะ "ผ่าน" แต่คะแนนรวมไม่ถึงเกณฑ์ ให้ปัดคะแนนรวมขึ้นให้เท่าเกณฑ์ (เพื่อให้ UI โชว์เขียว)
            if pdca_sum < threshold:
                pdca_sum = threshold

        # เข้มงวด L3+ (ต้องมี Check/Act)
        if not is_overridden:
            if level >= 3 and c <= 0:
                is_passed = False
                fail_reason = f"Level 3+ requires Check (C). Current C: {c}"
            if level >= 4 and a <= 0:
                is_passed = False
                fail_reason = f"Level 4+ requires Act (A). Current A: {a}"

        # 6. 📦 Sync & Return Object
        # รวมทุกอย่างกลับเข้าก้อนเดิมที่ Router/UI ต้องการ
        llm_output.update({
            "score": round(pdca_sum, 2),
            "is_passed": is_passed,
            "reason": fail_reason,
            "pdca_breakdown": pdca_results, # ✨ ตัวนี้แหละที่ UI จะเอาไปวาดกราฟ
            "status": "PASSED" if is_passed else "FAILED"
        })

        self.logger.info(
            f"🎯 [POST-PROCESS] {log_prefix} | Final: {llm_output['score']} | "
            f"P:{p} D:{d} C:{c} A:{a} | Passed: {is_passed} | Overridden: {is_overridden}"
        )

        return llm_output

    def _check_contextual_rule_condition(
        self, 
        condition: Dict[str, Any], 
        sub_id: str, 
        level: int, 
        top_evidences: List[Dict[str, Any]]
    ) -> bool:
        """
        [ADAPTIVE GATE v2026] 
        - เปลี่ยนจาก 'สั่งตก' เป็น 'บันทึกคำเตือน'
        - เพื่อให้เห็นผล Gap Analysis ครบทุกเลเวล
        """
        self.logger.info(f"🔍 [VALIDATION GATE] Analyzing L{level} for {sub_id}")
        
        # 1. เช็คความต่อเนื่อง (Maturity Check)
        if level > 1:
            prev_level = level - 1
            is_prev_passed = False
            
            # ดึงจาก Memory ที่เราบันทึกไว้ใน _run_single_assessment
            if hasattr(self, 'level_details_map') and str(prev_level) in self.level_details_map:
                is_prev_passed = self.level_details_map[str(prev_level)].get('is_passed', False)
            
            if not is_prev_passed:
                # 💡 เปลี่ยนจาก return False เป็นการฉีด Warning เข้าไปใน Context แทน
                self.logger.warning(f"⚠️ [GAP DETECTED] L{prev_level} is not passed. L{level} might be considered invalid by auditor.")
                # เราให้ True เพื่อให้ LLM ได้อ่านหลักฐาน L2 ต่อไปก่อน
        
        # 2. เช็คจำนวนหลักฐานขั้นต่ำ
        min_docs = condition.get('min_evidences', 1)
        if len(top_evidences) < min_docs:
            self.logger.warning(f"⚠️ [LOW EVIDENCE] Found only {len(top_evidences)} docs. Required: {min_docs}")

        return True # บังคับผ่านเพื่อให้รันไปจนถึง L5    

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

    def _expand_context_with_neighbor_pages(self, top_evidences: List[Any], collection_name: str) -> List[Any]:
        """
        [SMART EXPAND v3 - Low Log] ดึงหน้าถัดไป (และย้อนหลัง 1 หน้า) หากพบ Check
        - ลด log ให้เหลือน้อยที่สุด แต่ยัง debug ได้
        - รวม summary log ตอนจบ
        """
        if not self.vectorstore_manager or not top_evidences:
            return top_evidences

        expanded_evidences = list(top_evidences)
        seen_keys = set()
        added_pages = 0
        added_chunks = 0
        failed_pages = set()  # เก็บไฟล์+หน้า ที่ไม่เจอ เพื่อไม่ log ซ้ำ

        check_triggers = [
            "ความพึงพอใจ", "คะแนน", "สรุปผล", "ผลการดำเนินงาน", "score", "kpi", "3.41",
            "ประเมินผล", "รายงานผล", "ตัวชี้วัด", "ผลลัพธ์", "สรุปการทำงาน", "ผลประเมิน"
        ]

        for doc in top_evidences:
            text = (doc.get('text') or doc.get('page_content') or "").lower()
            meta = doc.metadata if hasattr(doc, 'metadata') else doc.get('metadata', {})
            page_label = meta.get("page_label")
            doc_uuid = meta.get("stable_doc_uuid")

            try:
                current_page = int("".join(filter(str.isdigit, str(page_label))))
            except (ValueError, TypeError):
                continue

            if not any(k in text for k in check_triggers):
                continue

            # ดึง ±1 และ +2 หน้า (ย้อนหลัง + ถัดไป)
            for offset in [-1, 1, 2]:
                target_page = current_page + offset
                if target_page < 1:
                    continue  # ข้ามหน้าติดลบ
                cache_key = f"{doc_uuid}_{target_page}"

                if cache_key in seen_keys:
                    continue

                neighbor_chunks = self.vectorstore_manager.get_chunks_by_page(
                    collection_name=collection_name,
                    stable_doc_uuid=doc_uuid,
                    page_label=str(target_page)
                )

                if neighbor_chunks:
                    for nc in neighbor_chunks:
                        new_doc = {
                            "text": f"[Act Context - Page {target_page} (จาก Check ที่หน้า {current_page})]:\n{nc.page_content}",
                            "page_content": nc.page_content,
                            "metadata": nc.metadata,
                            "pdca_tag": "Act",
                            "is_supplemental": True,
                            "rerank_score": doc.get('rerank_score', 0.0)
                        }
                        expanded_evidences.append(new_doc)
                    seen_keys.add(cache_key)
                    added_pages += 1
                    added_chunks += len(neighbor_chunks)
                else:
                    fail_key = f"{doc_uuid}_{target_page}"
                    if fail_key not in failed_pages:
                        self.logger.debug(f"⚠️ ไม่พบหน้า {target_page} ในไฟล์ {doc_uuid}")
                        failed_pages.add(fail_key)

        # Summary log แค่ครั้งเดียว
        if added_pages > 0:
            self.logger.info(f"Act-Hook completed: Added {added_pages} pages ({added_chunks} chunks) from Check triggers")
        else:
            self.logger.debug("Act-Hook: No additional neighbor pages found")

        return expanded_evidences
        
    def _resolve_evidence_filenames(self, evidence_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        [ULTIMATE FIX] ผสมความเสถียรของเวอร์ชันเดิม + ความฉลาดในการ Trace ของเวอร์ชันใหม่
        """
        resolved_entries = []
        for entry in evidence_entries:
            resolved_entry = deepcopy(entry)
            doc_id = resolved_entry.get("doc_id", "")
            content_raw = resolved_entry.get('content')
            level_origin = resolved_entry.get('level', 'N/A')
            
            # ดึงเลขหน้าให้ครอบคลุม (Page Label จาก Ingest หรือเลขหน้าดิบ)
            page_label = resolved_entry.get("page_label") or resolved_entry.get("page") or "N/A"

            # 1. AI Generated Reference (มโนมา)
            if str(doc_id).startswith("UNKNOWN-"):
                resolved_entry["filename"] = "AI-GENERATED-REF"
                resolved_entry["page"] = "N/A"
                resolved_entries.append(resolved_entry)
                continue

            # 2. เคสปกติ: มี doc_id และอยู่ใน Map
            if doc_id and doc_id in self.doc_id_to_filename_map:
                mapped_name = self.doc_id_to_filename_map[doc_id]
                resolved_entry["filename"] = mapped_name
                resolved_entry["display_source"] = f"{mapped_name} (หน้า {page_label})"
            
            # 3. เคสหาไม่เจอใน Map (Fallback)
            elif doc_id:
                resolved_entry["filename"] = f"DOC-{str(doc_id)[:8]}"
                resolved_entry["display_source"] = f"รหัส {str(doc_id)[:8]} (หน้า {page_label})"

            # 4. เคสข้อมูลหาย (The "Skipping" Case)
            else:
                if not content_raw:
                    # 🟢 ระบุ Level เพื่อให้รู้ว่าควรกลับไปแก้ PDF ที่ข้อไหน
                    self.logger.warning(f"⚠️ [Data Gap] Level {level_origin}: Entry ว่างเปล่า (Skipped)")
                    continue 
                
                resolved_entry["filename"] = "UNMAPPED-DOCUMENT"
                preview = str(content_raw)[:30].replace('\n', ' ')
                self.logger.debug(f"🔍 Unmapped Content Preview: {preview}...")

            resolved_entries.append(resolved_entry)
        return resolved_entries
        
    
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
                enabler=self.enabler_id
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

    # ----------------------------------------------------------------------
    # 🎯 FINAL FIX 2.3: Manual Map Reload Function (inside SEAMPDCAEngine)
    # ----------------------------------------------------------------------
    def _collect_previous_level_evidences(self, sub_id: str, current_level: int) -> Dict[str, List[Dict]]:
        """
        ดึงหลักฐานจาก Level ก่อนหน้า (L1 → L2, L2 → L3 ฯลฯ) เพื่อใช้เป็น Baseline Context
        
        [FINAL REVISED 2026] - Robust Hydration & UUID Normalization
        - รองรับ UUID ทั้งแบบมีขีด (-) และไม่มีขีด
        - กรองข้อมูลขยะ (fallback_doc_id, N/A) ออกก่อนเริ่มกระบวนการ
        - รักษาความสอดคล้องของ Metadata (PDCA Tag, Source, Page)
        """
        
        # 1. ข้าม Hydration ใน Full Parallel Mode เพื่อความเร็ว
        if getattr(self, 'is_parallel_all_mode', False):
            self.logger.info("FULL PARALLEL MODE: Skipping hydration")
            return {}

        # 2. คัดเลือกเฉพาะหลักฐานของ Level ที่ต่ำกว่าปัจจุบันใน Sub-Criteria เดียวกัน
        collected = {}
        for key, ev_list in self.evidence_map.items():
            if key.startswith(f"{sub_id}.L") and isinstance(ev_list, list) and ev_list:
                try:
                    level_num = int(key.split(".L")[-1])
                    if level_num < current_level:
                        collected[key] = ev_list
                except (ValueError, IndexError):
                    continue

        if not collected:
            self.logger.info(f"No previous level evidences found for {sub_id} L{current_level}")
            return {}

        # 3. รวบรวม IDs ที่ใช้งานได้จริง (Valid IDs only)
        stable_ids = set()
        for ev_list in collected.values():
            for ev in ev_list:
                sid = ev.get("stable_doc_uuid") or ev.get("doc_id")
                # กรองค่าขยะหรือค่าว่างที่อาจหลุดมาจาก LLM Response ในรอบก่อนหน้า
                if sid and isinstance(sid, str) and sid not in ["N/A", "fallback_doc_id", "None", ""]:
                    stable_ids.add(sid)

        if not stable_ids:
            self.logger.warning(f"⚠️ No valid IDs to hydrate for {sub_id} L{current_level}")
            return collected

        # 4. ดึง Full Text Chunks จาก Vector Store (Bulk Hydration)
        vsm = self.vectorstore_manager
        try:
            # ดึงเอกสารทั้งหมดที่เกี่ยวข้องมาไว้ในหน่วยความจำเพื่อทำ Mapping ครั้งเดียว
            full_chunks = vsm.get_documents_by_id(list(stable_ids), self.doc_type, self.enabler_id) 
            self.logger.info(f"HYDRATION: Retrieved {len(full_chunks)} chunks from VSM for mapping")
        except Exception as e:
            self.logger.error(f"Hydration failed in VSM call: {e}")
            return collected

        # 5. สร้าง Chunk Map (Key Optimization: เก็บทั้งแบบมีและไม่มีขีด)
        chunk_map = {}
        for chunk in full_chunks:
            meta = getattr(chunk, "metadata", {})
            # เก็บ UUID ทุกลูกเล่นที่เป็นไปได้เพื่อให้ Match เจอ 100%
            potential_ids = [
                meta.get("chunk_uuid"),
                meta.get("stable_doc_uuid"),
                meta.get("doc_id")
            ]
            for pid in potential_ids:
                if pid and isinstance(pid, str):
                    chunk_map[pid] = {"text": chunk.page_content, "metadata": meta}
                    chunk_map[pid.replace("-", "")] = {"text": chunk.page_content, "metadata": meta}

        # 6. กระบวนการฟื้นฟูเนื้อหา (Hydration Loop)
        hydrated = {}
        restored_count = 0
        total_items = sum(len(v) for v in collected.values())

        for key, ev_list in collected.items():
            new_list = []
            for ev in ev_list:
                new_ev = ev.copy()
                data = None
                
                # เตรียม IDs สำหรับใช้ค้นหาใน Map
                sid_raw = ev.get("stable_doc_uuid") or ev.get("doc_id") or ""
                sid_clean = sid_raw.replace("-", "")
                cid_raw = ev.get("chunk_uuid") or ""
                cid_clean = cid_raw.replace("-", "")

                # TRY 1: Match ด้วย Chunk UUID (แม่นยำที่สุด)
                data = chunk_map.get(cid_raw) or chunk_map.get(cid_clean)

                # TRY 2: Match ด้วย Stable Doc UUID / Doc ID (Fallback)
                if not data:
                    data = chunk_map.get(sid_raw) or chunk_map.get(sid_clean)

                if data:
                    new_ev["text"] = data["text"]
                    # Merge Metadata สำคัญกลับเข้าไป
                    merged_meta = data["metadata"].copy()
                    merged_meta.update(new_ev) # ให้ค่าเดิมใน Evidence เก่งกว่า
                    new_ev = merged_meta
                    
                    new_ev["is_baseline"] = True 
                    restored_count += 1
                else:
                    # รายงาน Error เฉพาะกรณีที่เป็น ID จริงแต่หาไม่เจอ
                    if sid_raw not in ["N/A", "fallback_doc_id", ""]:
                        self.logger.error(f"❌ HYDRATION FAILURE: {sid_raw[:8]}... (File: {ev.get('source_filename', 'Unknown')})")
                    new_ev["is_baseline"] = False
                    new_ev["page_label"] = ev.get("page_label") or ev.get("page") or "N/A"
                
                new_list.append(new_ev)
            hydrated[key] = new_list
                
        self.logger.info(f"✅ BASELINE HYDRATED: {restored_count}/{total_items} chunks restored successfully")
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

    def _save_evidence_map(self, map_to_save: Optional[Dict[str, List[Dict[str, Any]]]] = None):
        """
        บันทึก evidence map อย่างปลอดภัย 100% 
        [REVISED 2026] - รองรับ UUID v5, ป้องกัน ID 'fallback', และทำ Atomic Write
        """
        try:
            map_file_path = get_evidence_mapping_file_path(
                tenant=self.config.tenant,
                year=self.config.year,
                enabler=self.enabler_id
            )
        except Exception as e:
            self.logger.critical(f"[EVIDENCE] FATAL: ไม่สามารถกำหนด Path ได้: {e}")
            raise

        lock_path = map_file_path + ".lock"
        tmp_path = None

        self.logger.info(f"[EVIDENCE] Preparing atomic save → {map_file_path}")

        try:
            os.makedirs(os.path.dirname(map_file_path), exist_ok=True)

            with FileLock(lock_path, timeout=60):
                # 1. การเตรียมข้อมูลที่จะบันทึก (Merge Logic)
                if map_to_save is not None:
                    final_map_to_write = map_to_save
                else:
                    # โหลดข้อมูลเก่าจาก Disk มา Merge กับข้อมูลใหม่ใน Memory
                    existing_map = self._load_evidence_map(is_for_merge=True) or {}
                    runtime_map = deepcopy(self.evidence_map)
                    final_map_to_write = existing_map

                    for key, new_entries in runtime_map.items():
                        current_entries = final_map_to_write.setdefault(key, [])
                        
                        # สร้าง Index เพื่อเช็คความซ้ำซ้อน (Key: Cleaned UUID)
                        entry_map = {}
                        for e in current_entries:
                            if isinstance(e, dict):
                                # 🟢 [FIX] ทำความสะอาด ID เพื่อใช้เป็น Key ในการเปรียบเทียบ
                                raw_id = e.get("chunk_uuid") or e.get("doc_id") or "N/A"
                                clean_id = str(raw_id).replace("-", "").lower()
                                entry_map[clean_id] = e
                        
                        for new_entry in new_entries:
                            if not isinstance(new_entry, dict): continue
                            
                            # ดึง ID ของหลักฐานใหม่
                            raw_new_id = new_entry.get("chunk_uuid") or new_entry.get("doc_id") or "N/A"
                            clean_new_id = str(raw_new_id).replace("-", "").lower()

                            # 🟢 [FIX] ป้องกันข้อมูลขยะหรือ 'fallback' หลุดเข้าฐานข้อมูล
                            if clean_new_id in ["na", "n/a", "fallback", "none", ""]:
                                continue

                            new_score = new_entry.get("relevance_score", 0.0)

                            # ถ้ายังไม่มี ID นี้ใน Database หรือตัวใหม่ได้คะแนนสูงกว่า -> ให้บันทึก
                            if clean_new_id not in entry_map:
                                entry_map[clean_new_id] = new_entry
                            else:
                                old_entry = entry_map[clean_new_id]
                                old_score = old_entry.get("relevance_score", 0.0)
                                
                                # รักษาข้อมูล Metadata สำคัญหากตัวใหม่ไม่มี (เช่น เลขหน้า)
                                if "page" not in new_entry or new_entry["page"] in ["N/A", None]:
                                    new_entry["page"] = old_entry.get("page")
                                if "page_label" not in new_entry:
                                    new_entry["page_label"] = old_entry.get("page_label")
                                        
                                if new_score >= old_score:
                                    entry_map[clean_new_id] = new_entry

                        final_map_to_write[key] = list(entry_map.values())

                if not final_map_to_write:
                    self.logger.warning("[EVIDENCE] Nothing to save (empty map).")
                    return

                # 2. ทำความสะอาดข้อมูลก่อนเขียนลงไฟล์
                final_map_to_write = self._clean_temp_entries(final_map_to_write)
                for key, entries in final_map_to_write.items():
                    # เรียงลำดับตามคะแนนความเกี่ยวข้อง (สูงสุดอยู่บน)
                    entries.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)

                # 3. 🛡️ ATOMIC WRITE (เขียนลงไฟล์ชั่วคราวก่อนเพื่อป้องกันไฟล์พังขณะเขียน)
                with tempfile.NamedTemporaryFile(
                    mode='w', delete=False, encoding="utf-8", dir=os.path.dirname(map_file_path)
                ) as tmp_file:
                    cleaned_data = self._clean_map_for_json(final_map_to_write)
                    json.dump(cleaned_data, tmp_file, indent=4, ensure_ascii=False)
                    tmp_path = tmp_file.name

                # ย้ายไฟล์ชั่วคราวมาทับไฟล์จริง (เป็น Atomic Operation ในระดับ OS)
                shutil.move(tmp_path, map_file_path)
                tmp_path = None

                self.logger.info(f"[EVIDENCE] SAVED SUCCESSFULLY! Total Keys: {len(final_map_to_write)}")

        except Exception as e:
            self.logger.critical("[EVIDENCE] FATAL ERROR DURING ATOMIC SAVE")
            self.logger.exception(e)
            raise
        finally:
            # เก็บกวาดไฟล์ Lock และไฟล์ Temp ที่ค้างอยู่
            if os.path.exists(lock_path):
                try: os.unlink(lock_path)
                except: pass
            if tmp_path and os.path.exists(tmp_path):
                try: os.unlink(tmp_path)
                except: pass

    def merge_evidence_mappings(self, results_list: List[Any]) -> Dict[str, List[Dict]]:
        """
        [REVISED METHOD v2026.1.15]
        - ย้ายเข้าเป็น Method ของ Class SEAMPDCAEngine เรียบร้อย
        - ใช้ Logic การแกะ Tuple (level_id, evidence_map) ของพี่ 100%
        - ป้องกันข้อมูลซ้ำด้วย Indexing ของ doc_id/chunk_uuid
        """
        merged_mapping = {}
        
        self.logger.info(f"🧬 Starting to merge evidence mappings from {len(results_list)} levels...")

        for item in results_list:
            # 🎯 ดึงเฉพาะส่วนที่เป็น map ออกมาจาก Tuple (level_id, evidence_map)
            # โครงสร้าง item ปกติจะเป็น: (1, {"L1": [...], "L2": [...]})
            temp_map = item[1] if isinstance(item, tuple) and len(item) == 2 else {}
            
            # กรณีที่ item ไม่ใช่ tuple แต่เป็น dict อยู่แล้ว (กันพลาด)
            if not temp_map and isinstance(item, dict):
                temp_map = item

            if not temp_map: 
                continue

            for level_key, evidence_list in temp_map.items():
                if level_key not in merged_mapping:
                    merged_mapping[level_key] = []
                
                # 🛡️ สร้าง Index เพื่อกันข้อมูลซ้ำ (Unique ID)
                existing_ids = {
                    str(e.get('doc_id') or e.get('chunk_uuid')) 
                    for e in merged_mapping[level_key]
                }
                
                for new_ev in evidence_list:
                    new_id = str(new_ev.get('doc_id') or new_ev.get('chunk_uuid'))
                    if new_id not in existing_ids:
                        merged_mapping[level_key].append(new_ev)
                        existing_ids.add(new_id)
        
        self.logger.info(f"✅ Merging completed. Levels detected: {list(merged_mapping.keys())}")
        return merged_mapping

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

    def _get_level_constraint_prompt(self, sub_id: str, level: int) -> str:
        """
        [ADAPTIVE AUDIT GUIDELINE v2026] 
        - เปลี่ยนจาก 'กฎเหล็ก' เป็น 'แนวทางการพิจารณา'
        - ปรับให้ AI เน้นการหาความสอดคล้อง (Alignment) มากกว่าการจับผิด
        """
        required_phases = self.get_rule_content(sub_id, level, "require_phase") or []
        specific_rule = self.get_rule_content(sub_id, level, "specific_contextual_rule") or ""
        
        level_name = PDCA_PHASE_MAP.get(level, f"Level {level}")
        level_goal = MATURITY_LEVEL_GOALS.get(level, "")

        prompt_lines = [
            f"\n### [แนวทางการประเมิน: {sub_id} | {level_name}] ###",
            f"🎯 เป้าหมายสำคัญ: {level_goal}",
            "---"
        ]
        
        # 📌 1. การพิจารณาเฟส (เปลี่ยนจาก 'ตัดสินไม่ผ่าน' เป็น 'ประเมินความสอดคล้อง')
        if required_phases:
            phase_labels = {"P": "วางแผน (Plan)", "D": "ปฏิบัติ (Do)", "C": "ตรวจสอบ (Check)", "A": "ปรับปรุง (Act)"}
            readable = [phase_labels.get(p, p) for p in required_phases]
            prompt_lines.append(f"🔍 องค์ประกอบที่ควรพบ: {', '.join(readable)}")
            prompt_lines.append("   - โปรดวิเคราะห์ว่าหลักฐานที่พบสะท้อนกิจกรรมในเฟสเหล่านี้เพียงพอหรือไม่")
            prompt_lines.append("   - หากไม่พบหลักฐานในเฟสสำคัญ ให้ระบุสิ่งที่ขาดหายในช่องข้อเสนอแนะเพื่อการพัฒนา")

        # 🛑 2. กฎเฉพาะข้อ (เปลี่ยนจาก 'กฎเหล็ก' เป็น 'เกณฑ์การพิจารณาพิเศษ')
        if specific_rule:
            prompt_lines.append(f"💡 เกณฑ์พิจารณาพิเศษสำหรับข้อนี้: \"{specific_rule}\"")
            prompt_lines.append("   - ใช้เกณฑ์นี้เป็นตัวชี้วัดความสมบูรณ์ของเนื้อหา")
            prompt_lines.append("   - หากหลักฐานใกล้เคียงหรือสอดคล้องตามเจตนารมณ์ของเกณฑ์ ให้คะแนนตามระดับความเหมาะสม")

        # ⚖️ 3. เพิ่ม Instruction เพื่อความเป็นธรรม (The Fair Guard)
        prompt_lines.append("\n⚖️ [หลักการตัดสิน]")
        prompt_lines.append("- ประเมินตามเนื้อหาจริง (Substance over Form) อย่าปัดตกเพียงเพราะไม่เจอ Keyword ตรงตัว")
        prompt_lines.append("- กรณีที่หลักฐานก้ำกึ่ง ให้ความสำคัญกับ 'ความพยายามในการดำเนินการ' และให้ Coaching Insight ที่ชัดเจน")
        
        return "\n".join(prompt_lines)

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
    
    # -------------------- Calculation Helpers (ADDED) --------------------
    def _calculate_weighted_score(self, highest_full_level: int, weight: int) -> float:
        """
        Calculates the weighted score based on the highest full level achieved.
        Score is calculated by: (Level / MAX_LEVEL) * Weight
        """
        # 🎯 เปลี่ยนจาก MAX_LEVEL_CALC เป็น MAX_LEVEL ที่ดึงมาจาก global_vars
        from config.global_vars import MAX_LEVEL  
        
        if highest_full_level <= 0:
            return 0.0
        
        # ป้องกันกรณีคะแนนเกิน (เช่น ถ้ามี data ผิดพลาด)
        level_for_calc = min(highest_full_level, MAX_LEVEL)
        
        # คำนวณคะแนนถ่วงน้ำหนักตามเพดานเลเวลสูงสุด
        score = (level_for_calc / MAX_LEVEL) * weight
        return score

    def _calculate_overall_stats(self, target_sub_id: str):
        """
        [ULTIMATE STATS v2026.4] Weighted Maturity & Coaching Analytics
        ------------------------------------------------------
        - คำนวณ Maturity Score และ Progress %
        - เพิ่มการนับจำนวน Soft Gaps และ Strength Points สำหรับ Coaching Report
        """
        from config.global_vars import MAX_LEVEL
        results = self.final_subcriteria_results
        
        # 1. 🛡️ Safety Guard
        if not results:
            self.total_stats = {
                "overall_avg_score": 0.0,
                "overall_level_label": "L0",
                "record_id": self.current_record_id,
                "status": "No Data"
            }
            return

        # 2. ⚖️ คะแนนถ่วงน้ำหนัก
        total_weighted_score_achieved = sum(r.get('weighted_score', 0.0) for r in results)
        total_possible_weight = sum(r.get('weight', 0.0) for r in results)

        # 3. 📊 Maturity Score & Progress
        overall_avg_score = 0.0
        if total_possible_weight > 0:
            overall_avg_score = round((total_weighted_score_achieved / total_possible_weight) * MAX_LEVEL, 2)
        
        max_possible_points = total_possible_weight * MAX_LEVEL
        progress_percent = 0.0
        if max_possible_points > 0:
            progress_percent = round((total_weighted_score_achieved / max_possible_points) * 100, 2)

        # 4. 🏷️ Maturity Level Label (Audit Logic)
        avg_full_level = sum(r.get('highest_full_level', 0) for r in results) / len(results)
        final_level = int(avg_full_level) 
        overall_level_label = f"L{min(max(final_level, 0), MAX_LEVEL)}"

        # 5. 💡 [NEW] Coaching Metrics (นับจำนวนจุดแข็งและจุดที่ต้องพัฒนา)
        total_strengths = 0
        total_coaching_needs = 0
        
        for r in results:
            # ตรวจสอบจากรายละเอียดเลเวล
            details = r.get('level_details', {})
            for lvl_data in details.values():
                insight = lvl_data.get('coaching_insight', "")
                if "จุดแข็ง" in insight or "🌟" in insight:
                    total_strengths += 1
                if "ข้อแนะนำ" in insight or "💡" in insight:
                    total_coaching_needs += 1

        # 6. 📝 บันทึกผลสรุป
        self.total_stats = {
            "overall_avg_score": min(overall_avg_score, float(MAX_LEVEL)),
            "overall_level_label": overall_level_label,
            "total_weighted_score": round(total_weighted_score_achieved, 2),
            "total_possible_weight": total_possible_weight,
            "progress_percent": progress_percent,
            "gap_to_full": round(total_possible_weight - total_weighted_score_achieved, 2),
            "assessed_count": len(results),
            # เพิ่มมิติการ Coaching ใน Stats
            "coaching_metrics": {
                "total_strength_points": total_strengths,
                "total_improvement_areas": total_coaching_needs
            },
            "enabler": self.config.enabler,
            "record_id": self.current_record_id,
            "assessed_at": datetime.now().isoformat()
        }

        # 7. 📢 Logging
        self.logger.info(f"🏆 Overall: {overall_level_label} ({overall_avg_score}/{MAX_LEVEL})")
        self.logger.info(f"💡 Analytics: Strengths[{total_strengths}] | Needs Improvement[{total_coaching_needs}]")


    def _export_results(self, results: dict, sub_criteria_id: str, **kwargs) -> str:
        """
        [ULTIMATE EXPORTER v2026.4 - PRODUCTION READY]
        ---------------------------
        - 🛡️ JSON Safe: รองรับ Pydantic objects และ ActionPlan
        - 📂 Hierarchical Storage: เก็บแยก Tenant/Year/Enabler
        - 💡 Coaching Summary: รวม Insight ทุก Level ไว้ที่เดียว
        - 📊 Audit Summary: สรุปสถิติคะแนนและสถานะเป้าหมาย
        """

        # 1. ⚙️ เตรียมข้อมูลพื้นฐาน
        record_id = kwargs.get("record_id", getattr(self, "current_record_id", "no_id"))
        enabler = self.config.enabler
        tenant = self.config.tenant
        year = str(self.config.year)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 2. 📁 การจัดการ Path
        export_dir = os.path.join("data_store", tenant, "exports", year, enabler)
        try:
            os.makedirs(export_dir, exist_ok=True)
            file_name = f"assessment_{enabler}_{record_id}_{sub_criteria_id}_{timestamp}.json"
            full_path = os.path.join(export_dir, file_name)
        except Exception as e:
            self.logger.error(f"❌ Directory creation failed: {e}")
            full_path = f"assessment_fallback_{record_id}.json"

        # 3. 🛠️ JSON Serialization Helper (จุดสำคัญที่แก้ไข)
        def json_serializable(obj):
            """แปลง Object พิเศษให้เป็น Type ที่ JSON รองรับ"""
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, BaseModel): # รองรับ Pydantic (ActionPlan)
                return obj.model_dump()
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            return str(obj)

        # 4. 📊 ตรวจสอบและสร้างโครงสร้าง Summary
        if 'summary' not in results:
            results['summary'] = {}
        
        summary = results['summary']
        sub_res_list = results.get('sub_criteria_results', [])

        # 5. 💡 สกัด Coaching Insights จากรายเลเวล (รวมศูนย์)
        all_coaching_insights = []
        for sub_res in sub_res_list:
            details = sub_res.get('level_details', {})
            for lvl, data in details.items():
                insight = data.get('coaching_insight', "")
                if insight:
                    all_coaching_insights.append({
                        "id": sub_res.get('sub_id'),
                        "level": lvl,
                        "insight": insight
                    })
        
        summary['coaching_summary'] = all_coaching_insights

        # 6. 📑 ฝัง Identity Metadata
        results['metadata'] = {
            "record_id": record_id,
            "tenant": tenant,
            "year": year,
            "enabler": enabler,
            "model_used": getattr(self.config, "model_name", "unknown"),
            "target_level": self.config.target_level,
            "export_at": datetime.now().isoformat()
        }

        # 7. 📈 คำนวณสถิติคะแนน (Audit Summary)
        if str(sub_criteria_id).lower() != "all" and len(sub_res_list) > 0:
            # กรณีประเมินรายข้อ (เช่น 1.1)
            main_res = sub_res_list[0]
            summary.update({
                "highest_pass_level": main_res.get('highest_full_level', 0),
                "achieved_weight": round(main_res.get('weighted_score', 0.0), 2),
                "total_weight": main_res.get('weight', 0.0),
                "is_target_achieved": main_res.get('target_level_achieved', False)
            })
        else:
            # กรณีประเมินภาพรวม (All)
            all_pass_levels = [r.get('highest_full_level', 0) for r in sub_res_list]
            total_achieved = sum(r.get('weighted_score', 0.0) for r in sub_res_list)
            total_possible = sum(r.get('weight', 0.0) for r in sub_res_list)
            
            summary.update({
                "highest_pass_level_overall": max(all_pass_levels) if all_pass_levels else 0,
                "total_achieved_weight": round(total_achieved, 2),
                "total_possible_weight": round(total_possible, 2),
                "overall_percentage": round((total_achieved / total_possible * 100), 2) if total_possible > 0 else 0.0,
                "total_subcriteria_assessed": len(sub_res_list)
            })

        # นับจำนวน Action Plan Items ทั้งหมด
        summary['total_action_plan_items'] = sum(
            len(r.get('action_plan', [])) 
            for r in sub_res_list 
            if isinstance(r.get('action_plan'), list)
        )

        # 8. 💾 บันทึกไฟล์ด้วย Safety Serializer
        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                # ใช้ default=json_serializable เพื่อป้องกัน Error
                json.dump(results, f, ensure_ascii=False, indent=4, default=json_serializable)
            
            self.logger.info(f"💾 EXPORT SUCCESS: {full_path}")
            return full_path
            
        except Exception as e:
            self.logger.error(f"❌ Export failed at write stage: {str(e)}")
            # Fallback: พยายามบันทึกข้อมูลแบบดิบหาก JSON พัง
            try:
                fallback_path = full_path.replace(".json", "_emergency_dump.txt")
                with open(fallback_path, 'w', encoding='utf-8') as f:
                    f.write(str(results))
                return fallback_path
            except:
                return ""
        
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

    def _save_level_evidences_and_calculate_strength(
        self,
        level_temp_map: List[Dict[str, Any]],
        sub_id: str,
        level: int,
        llm_result: Dict[str, Any],
        highest_rerank_score: float = 0.0
    ) -> float:
        """ [REVISED v2026.12] - ใช้ LLM Tagging ในการบันทึกหลักฐานถาวร """
        import hashlib
        import os
        from datetime import datetime

        map_key = f"{sub_id}.L{level}"
        new_evidence_list: List[Dict[str, Any]] = []
        seen_ids = set()

        self.logger.info(f"💾 [EVI SAVE] Processing {map_key} | Count: {len(level_temp_map)}")

        for chunk in level_temp_map:
            # 1. จัดเตรียมข้อมูล (Handle both dict & LangChain Doc)
            meta = chunk.get("metadata", {}) if isinstance(chunk, dict) else getattr(chunk, "metadata", {})
            text = chunk.get("text") if isinstance(chunk, dict) else getattr(chunk, "page_content", "")
            
            if not text.strip(): continue

            # 2. Stable ID Generation (ป้องกันข้อมูลซ้ำ)
            c_uuid = str(chunk.get("chunk_uuid") or meta.get("chunk_uuid") or hashlib.sha256(text.encode()).hexdigest()[:16])
            d_uuid = str(chunk.get("stable_doc_uuid") or meta.get("stable_doc_uuid") or "doc-unknown")
            unique_key = f"{d_uuid}:{c_uuid}"
            if unique_key in seen_ids: continue
            seen_ids.add(unique_key)

            # 3. ✨ CRITICAL FIX: การระบุ PDCA Tag (ใช้ LLM แทน Regex)
            # ดึง Tag เดิมจาก Metadata ถ้ามี (จากขั้นตอนก่อนหน้าใน Assessment)
            pdca_tag = chunk.get("pdca_tag") or meta.get("pdca_tag")
            
            # ถ้ายังไม่มี Tag หรือเป็น Other ให้ Re-tag ด้วย LLM (เพื่อความเป๊ะก่อนลง DB)
            if not pdca_tag or pdca_tag == "Other":
                fname = os.path.basename(str(meta.get("source") or "Unknown"))
                # เรียกใช้ตัวเก่งที่เราจูนไว้
                pdca_tag = self._get_semantic_tag(text, sub_id, level, filename=fname)

            # 4. Normalize ข้อมูลสำหรับรายงาน
            source_raw = meta.get("source") or "Unknown"
            
            evidence_entry = {
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
                "text_preview": text[:300].replace("\n", " ") + "...",
                "status": "PASS" if llm_result.get("is_passed", False) else "FAIL",
                "timestamp": datetime.now().isoformat(),
            }
            new_evidence_list.append(evidence_entry)

        # 5. การบันทึกและคำนวณ Strength
        if new_evidence_list:
            # บันทึกลง Memory Map ของระบบ
            self.evidence_map.setdefault(map_key, []).extend(new_evidence_list)
            self.temp_map_for_save.setdefault(map_key, []).extend(new_evidence_list)

            # นับจำนวน Tag เพื่อทำ Log Summary
            counts = {"P": 0, "D": 0, "C": 0, "A": 0, "Other": 0}
            for ev in new_evidence_list: counts[ev['pdca_tag']] += 1

            # 📊 Strength Calculation ( rerank + pdca_richness )
            # ให้ความสำคัญกับ Rerank Score 60% และความหลากหลายของ PDCA 40%
            unique_tags = {ev['pdca_tag'] for ev in new_evidence_list if ev['pdca_tag'] in "PDCA"}
            coverage_score = len(unique_tags) / 4.0 # พบกี่หมวดจาก 4 หมวด
            
            final_strength = round((highest_rerank_score * 0.6) + (coverage_score * 0.4), 2)

            self.logger.info(
                f"✅ [SAVED] {map_key}: {len(new_evidence_list)} chunks | "
                f"P:{counts['P']} D:{counts['D']} C:{counts['C']} A:{counts['A']} | "
                f"Strength: {final_strength:.2f}"
            )
            return final_strength
            
        return 0.0
    
    def _calculate_evidence_strength_cap(
        self,
        top_evidences: List[Any],
        level: int,
        highest_rerank_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        [PROTECTED v2026] คำนวณขีดจำกัดคะแนนและตรวจสอบคุณภาพหลักฐาน
        - คืนค่าเป็น Dictionary เสมอเพื่อป้องกัน Error ในฟังก์ชันเรียกใช้
        """
        try:
            # ⚙️ Configuration
            threshold = getattr(self, "RERANK_THRESHOLD", 0.35)
            cap_value = getattr(self, "MAX_EVI_STR_CAP", 5.0)
            
            # ป้องกันค่า None หรือ String ในการคำนวณเลข
            try:
                baseline_score = float(highest_rerank_score) if highest_rerank_score is not None else 0.0
            except (ValueError, TypeError):
                baseline_score = 0.0

            max_score_found = baseline_score
            max_score_source = "Adaptive_RAG_Loop"
            
            if not isinstance(top_evidences, list):
                top_evidences = []

            score_keys = ["rerank_score", "score", "relevance_score", "_rerank_score_force"]

            # 🔍 Scan Metadata
            for idx, doc in enumerate(top_evidences, 1):
                current_score = 0.0
                if isinstance(doc, dict):
                    metadata = doc.get("metadata") or {}
                    current_doc_source = metadata.get("file_name") or metadata.get("source") or f"Doc_{idx}"
                else:
                    metadata = getattr(doc, "metadata", {}) or {}
                    current_doc_source = getattr(doc, "source", f"Doc_{idx}")

                for key in score_keys:
                    val = metadata.get(key) if isinstance(metadata, dict) else None
                    if val is None and isinstance(doc, dict): val = doc.get(key)
                    
                    if val is not None:
                        try:
                            temp_s = float(val)
                            if 0.0 < temp_s <= 1.0:
                                current_score = temp_s
                                break
                        except: continue

                if current_score > max_score_found:
                    max_score_found = current_score
                    max_score_source = str(current_doc_source)

            is_capped = max_score_found < threshold
            max_evi_str_for_prompt = float(cap_value) if is_capped else 10.0

            # 📊 Internal Log
            status_icon = "🚨" if is_capped else "✅"
            self.logger.info(
                f"{status_icon} Evi Str {'CAPPED' if is_capped else 'FULL'} L{level}: "
                f"Best {max_score_found:.4f} from '{max_score_source}' (Threshold: {threshold})"
            )

            return {
                "is_capped": bool(is_capped),
                "max_evi_str_for_prompt": float(max_evi_str_for_prompt),
                "top_score": round(float(max_score_found), 4),
                "max_score_source": str(max_score_source)
            }

        except Exception as e:
            self.logger.error(f"❌ Critical Fallback in _calculate_evidence_strength_cap: {e}")
            return {"is_capped": False, "max_evi_str_for_prompt": 10.0, "top_score": 0.0, "max_score_source": "Fallback"}
    
    def calculate_audit_confidence(self, matched_chunks: List[Any]) -> Dict[str, Any]:
        """
        [ULTIMATE AUDIT CONFIDENCE v2026.3]
        - Quality Gate: กรองหลักฐานที่ Score ต่ำทิ้ง
        - Independence: เช็คความหลากหลายของแหล่งข้อมูล (ป้องกันการใช้ไฟล์เดียวตอบทุกอย่าง)
        - Coverage: เช็คความครบถ้วนของวงจร PDCA
        - Traceability: เช็คความพร้อมของ Metadata (ชื่อไฟล์/เลขหน้า)
        """
        if not matched_chunks:
            return {
                "level": "NONE", "reason": "ไม่พบหลักฐานที่เกี่ยวข้องในระบบ",
                "source_count": 0, "coverage_ratio": 0, "traceability_score": 0
            }

        # 0. 🛡️ Quality Filter (ใช้ Threshold 0.40 ตามมาตรฐาน SE-AM)
        valid_chunks = []
        for doc in matched_chunks:
            # รองรับทั้ง Dict และ Langchain Document
            meta = doc.metadata if hasattr(doc, 'metadata') else doc.get('metadata', {})
            score = meta.get('rerank_score') or meta.get('score') or 1.0
            
            if score >= 0.40:
                valid_chunks.append(doc)
        
        if not valid_chunks:
            return {
                "level": "LOW", "reason": "พบข้อมูลแต่คะแนนความเกี่ยวข้อง (Relevance) ต่ำเกินกว่าจะเชื่อถือได้",
                "source_count": 0, "coverage_ratio": 0, "traceability_score": 0
            }

        # 1. 📂 Source Independence (นับไฟล์ที่ไม่ซ้ำกัน)
        unique_sources = set()
        for doc in valid_chunks:
            meta = doc.metadata if hasattr(doc, 'metadata') else doc.get('metadata', {})
            # ใช้ Priority ในการดึงชื่อไฟล์เหมือนฟังก์ชัน _normalize_meta
            src = (meta.get('source_filename') or meta.get('filename') or 
                   meta.get('file_name') or meta.get('source'))
            if src:
                unique_sources.add(os.path.basename(str(src)))
        
        independence_score = len(unique_sources)
        
        # 2. 🧩 PDCA Coverage (เช็คมิติความครบถ้วน)
        pdca_map = {"P": False, "D": False, "C": False, "A": False}
        for doc in valid_chunks:
            meta = doc.metadata if hasattr(doc, 'metadata') else doc.get('metadata', {})
            tag = str(meta.get('pdca_tag', '')).upper()
            if tag in pdca_map:
                pdca_map[tag] = True
        
        found_tags_count = sum(pdca_map.values())
        coverage_ratio = found_tags_count / 4
        
        # 3. 🔍 Traceability (ความโปร่งใสของแหล่งอ้างอิง)
        traceable_count = 0
        for doc in valid_chunks:
            meta = doc.metadata if hasattr(doc, 'metadata') else doc.get('metadata', {})
            # ตรวจสอบเลขหน้าอย่างละเอียด (ป้องกันหน้า 0 หาย)
            has_page = any([
                meta.get('page_label') is not None,
                meta.get('page') is not None,
                meta.get('page_number') is not None
            ])
            has_file = any([meta.get('source_filename'), meta.get('filename'), meta.get('file_name')])
            
            if has_page and has_file:
                traceable_count += 1
        
        traceability_score = traceable_count / len(valid_chunks) if valid_chunks else 0

        # 4. ⚖️ Decision Matrix (Gated Audit Logic)
        if coverage_ratio <= 0.25 or independence_score < 1:
            confidence = "LOW"
            desc = "ความเชื่อมั่นต่ำ: ขาดความหลากหลายของแหล่งข้อมูล หรือมีมิติ PDCA ไม่เพียงพอ"
        elif independence_score < 3 or coverage_ratio < 0.75:
            confidence = "MEDIUM"
            desc = "ความเชื่อมั่นปานกลาง: พบหลักฐานสนับสนุนจากหลายแหล่ง แต่ยังขาดความครบถ้วนในมิติ PDCA"
        else:
            confidence = "HIGH"
            desc = "ความเชื่อมั่นสูง: หลักฐานครบวงจร PDCA และมีการยืนยันจากแหล่งข้อมูลที่หลากหลาย (Cross-Check)"

        # 🚨 Penalty: ลดระดับความมั่นใจหากระบุตำแหน่ง (เลขหน้า) ไม่ชัดเจน
        if traceability_score < 0.50 and confidence != "LOW":
            confidence = "MEDIUM" if confidence == "HIGH" else "LOW"
            desc += " (คุณภาพการอ้างอิงตำแหน่งเอกสารต่ำ)"

        return {
            "level": confidence, 
            "reason": desc, 
            "source_count": independence_score,
            "coverage_ratio": coverage_ratio,
            "traceability_score": round(traceability_score, 2),
            "pdca_found": [k for k, v in pdca_map.items() if v]
        }

        

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
    
    
    def _get_pdca_blocks_from_evidences(self, evidences, baseline_evidences, level, sub_id, contextual_rules_map, record_id=None) -> Dict[str, str]:
        """ รวมข้อมูลและ Tagging ให้ไอคอน ✅ แสดงผล """
        pdca_groups = defaultdict(list)
        seen_texts = set()

        for chunk in (evidences or []):
            txt = chunk.get("text", "").strip()
            if not txt or txt in seen_texts: continue
            seen_texts.add(txt)

            # เรียก LLM Tagging
            tag = self._get_semantic_tag(text=txt, sub_id=sub_id, level=level, filename=chunk.get("source_filename", "Unknown"))
            final_tag = tag if tag in {"P", "D", "C", "A"} else ("P" if level == 1 else "Other")
            
            chunk["pdca_tag"] = final_tag # บันทึกเพื่อให้ระบบอื่นรู้
            pdca_groups[final_tag].append(chunk)

        # สร้าง Dictionary ของ Blocks
        return {
            "Plan": self._create_text_block_from_chunks("Plan", pdca_groups["P"]),
            "Do": self._create_text_block_from_chunks("Do", pdca_groups["D"]),
            "Check": self._create_text_block_from_chunks("Check", pdca_groups["C"]),
            "Act": self._create_text_block_from_chunks("Act", pdca_groups["A"]),
            "Other": self._create_text_block_from_chunks("Other", pdca_groups["Other"])
        }

    def _create_text_block_from_chunks(self, tag, chunks):
        """ ตัวช่วยสร้างเนื้อหาจาก chunk """
        if not chunks: return ""
        parts = [f"### [{tag} Evidence]\n{c.get('text','')}" for c in chunks[:5]]
        return "\n\n".join(parts)


    def _generate_action_plan_safe(
        self, 
        sub_id: str, 
        name: str, 
        enabler: str, 
        results: List[Dict]
    ) -> Any:
        """
        [ULTIMATE REVISE 2026.5] - STRATEGIC ACTION PLAN GENERATOR
        -----------------------------------------------------------
        - บูรณาการ Coaching Insight และ Soft Gap เข้าสู่ Roadmap
        - แยก Mode การทำงานตามความเข้มข้นของผลการประเมิน (Sustain / Refinement / Gap)
        - ใช้ Logic ซ่อมรากฐาน (Foundation Repair) ก่อนยกระดับสู่เป้าหมาย
        """
        try:
            self.logger.info(f"🛠️ Preparing Strategic Action Plan for {sub_id} - {name}")
            
            # 1. คัดกรองและเตรียมข้อมูลสำหรับคำแนะนำ (Data Enrichment)
            to_recommend = []
            has_major_gap = False
            
            for r in results:
                is_passed = r.get('is_passed', False)
                strength = r.get('score', 0.0) # ใช้ score รายด้านมาพิจารณา
                coaching = r.get('coaching_insight', '').strip()
                reason = r.get('reason', '').strip()
                level = r.get('level', 0)

                # รวม Coaching Insight เข้าไปในเนื้อหาที่จะส่งให้ AI
                enhanced_reason = reason
                if coaching:
                    enhanced_reason += f"\n[Coaching Insight & Soft Gap]: {coaching}"

                # เงื่อนไขการเลือกเข้าสู่ Action Plan:
                # - ไม่ผ่าน (FAILED/GAP)
                # - ผ่านแต่คะแนนต่ำ (Weak Evidence)
                # - มี Coaching Insight (มีจุดที่ควรปรับปรุงหรือต่อยอด)
                if not is_passed:
                    has_major_gap = True
                    to_recommend.append({
                        "level": level,
                        "reason": enhanced_reason,
                        "recommendation_type": "FAILED"
                    })
                elif is_passed and (strength < 1.0 or coaching):
                    to_recommend.append({
                        "level": level,
                        "reason": enhanced_reason,
                        "recommendation_type": "QUALITY_REFINEMENT" if strength < 1.0 else "SUSTAIN_ADVICE"
                    })

            # 2. กรณีผ่านหมดและหลักฐานดีมาก (No recommendation needed)
            if not to_recommend:
                return {
                    "status": "EXCELLENT", 
                    "message": "ไม่ต้องมีแผนปรับปรุง เนื่องจากระบบงานและหลักฐานมีความเข้มแข็งครอบคลุมทุกวงจร PDCA แล้ว",
                    "coaching_summary": "เน้นการรักษามาตรฐานและการเป็นต้นแบบ (Best Practice)"
                }

            # 3. ตัดสินโหมดการรัน Action Plan
            # ถ้ามีข้อที่ไม่ผ่าน (FAILED) ให้ใช้ ACTION_PLAN_PROMPT (Remediation Mode)
            # ถ้าผ่านหมดแต่มีจุดเติมคุณภาพ ให้ใช้ QUALITY_REFINEMENT_PROMPT
            # ถ้าเป็น Level 5 หมดแล้ว ให้ใช้ EXCELLENCE_ADVICE_PROMPT
            
            target_level = self.config.target_level if hasattr(self, 'config') else 5
            
            # เตรียม Argument หลัก
            action_plan_args = {
                "recommendation_statements": to_recommend,
                "sub_id": sub_id,
                "sub_criteria_name": name,
                "enabler": enabler,
                "target_level": target_level,
                "llm_executor": self.llm,
                "logger": self.logger
            }

            # 4. เรียกใช้เครื่องยนต์สร้างแผนงาน (Ref. create_structured_action_plan ที่คุณมี)
            # หมายเหตุ: ใน create_structured_action_plan จะมีการเลือกใช้ PromptTemplate 
            # ตาม Mode ที่วิเคราะห์จาก recommendation_statements
            
            roadmap = create_structured_action_plan(**action_plan_args)

            # 5. สรุปผลการสร้าง
            if isinstance(roadmap, list) and len(roadmap) > 0:
                self.logger.info(f"✅ Action Plan generated with {len(roadmap)} phases")
                return roadmap
            else:
                return _get_emergency_fallback_plan(sub_id, name, target_level, not has_major_gap, False, enabler)

        except Exception as e:
            self.logger.error(f"⚠️ Action Plan Generation Failed: {str(e)}", exc_info=True)
            return {
                "status": "ERROR",
                "message": f"เกิดข้อผิดพลาดในการสร้างแผนงานอัตโนมัติ: {str(e)}",
                "fallback_plan": _get_emergency_fallback_plan(sub_id, name, 5, False, False, enabler)
            }
    
    def _prepare_worker_tuple(self, sub_data: Dict, document_map: Optional[Dict]) -> Tuple:
        return (
            sub_data,                          # ข้อมูลเกณฑ์ (1.1, 1.2...)
            self.config.enabler,               # KM / IT / ...
            self.config.target_level,          # เป้าหมาย L5
            self.config.mock_mode,             # none/random
            self.evidence_map_path,            # ที่อยู่ไฟล์เก็บหลักฐาน
            self.config.model_name,            # llama3.1:70b หรือ 8b
            self.config.temperature,           # 0.0
            getattr(self.config, 'min_retry_score', 0.65),
            getattr(self.config, 'max_retrieval_attempts', 3),
            document_map or self.document_map, # แผนผังเอกสาร
            getattr(self, 'ActionPlanActions', None),
            self.config.year,                  # 2567
            self.config.tenant                 # pea
        )

    def _integrate_worker_results(self, sub_result: Dict, temp_map: Dict):
        # 1. จัดการแผนผังหลักฐาน (Evidence Map)
        if isinstance(temp_map, dict):
            for level_key, evidence_list in temp_map.items():
                # แปลง UUID เป็นชื่อไฟล์จริง และจัดการ Metadata
                resolved_list = self._resolve_evidence_filenames(evidence_list)
                self._normalize_evidence_metadata(resolved_list)
                
                # ยัดเข้ากระเป๋าหลักของ Main Process
                if level_key not in self.evidence_map:
                    self.evidence_map[level_key] = []
                self.evidence_map[level_key].extend(resolved_list)
        
        # 2. เก็บผลประเมินดิบจาก LLM
        if sub_result:
            self.raw_llm_results.extend(sub_result.get("raw_results_ref", []))
            self.final_subcriteria_results.append(sub_result)

    # ----------------------------------------------------------------------
    # 🚀 CORE WORKER: Assessment Execution (REVISED v2026.1.14 - FINAL STABLE)
    # ----------------------------------------------------------------------
    def _run_sub_criteria_assessment_worker(
        self,
        sub_criteria: Dict[str, Any],
        vectorstore_manager: Optional['VectorStoreManager'] = None
    ) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
        """
        [ADVANCED AUDITOR MODE v2026.1.14]
        - FIXED: 'tuple' object has no attribute 'get' (Robust Result Handling)
        - IMPROVED: PDCA Breakdown mapping directly from LLM response
        - OPTIMIZED: Prevention of redundant audit starts during retries
        """
        # 1. INITIALIZATION
        MAX_RETRY_ATTEMPTS = 2
        sub_id = sub_criteria['sub_id']
        sub_criteria_name = sub_criteria['sub_criteria_name']
        sub_weight = sub_criteria.get('weight', 0)
        
        current_enabler = getattr(self.config, 'enabler', 'KM')
        vsm = vectorstore_manager or getattr(self, 'vectorstore_manager', None)
        
        current_sequential_pass_level = 0 
        first_failure_level = None 
        raw_results_for_sub_seq: List[Dict[str, Any]] = []
        level_details_map = {} 
        start_ts = time.time() 

        self.logger.info(f"🧵 [WORKER START] {sub_id} | Mode: Phase-Based Sequential")
        all_rules_for_sub = getattr(self, 'contextual_rules_map', {}).get(sub_id, {})

        # -----------------------------------------------------------
        # 2. EVALUATION LOOP (L1 → Target Level)
        # -----------------------------------------------------------
        levels_to_assess = sorted(sub_criteria.get('levels', []), key=lambda x: x.get('level', 0))

        for statement_data in levels_to_assess:
            level = statement_data.get('level')
            if level is None or level > self.config.target_level:
                continue
            
            # --- [EXECUTION with RETRY & SAFE UNPACKING] ---
            level_result = {}
            for attempt_num in range(1, MAX_RETRY_ATTEMPTS + 1):
                try:
                    # 🔍 เรียกประเมินรายเลเวล
                    raw_res = self._run_single_assessment(
                        sub_criteria=sub_criteria,
                        statement_data=statement_data,
                        vectorstore_manager=vsm,
                        attempt=attempt_num,
                        record_id=self.current_record_id,
                        evidence_map=self.evidence_map,
                        **all_rules_for_sub.get(str(level), {})
                    )

                    # ✨ [CRITICAL FIX] ป้องกัน Error Tuple แบบเบ็ดเสร็จ
                    if isinstance(raw_res, tuple):
                        # ถ้าส่งมาเป็น (dict, map) หรือ (dict,) ให้แกะเอาตัวแรก
                        level_result = raw_res[0] if len(raw_res) > 0 else {}
                    elif isinstance(raw_res, dict):
                        level_result = raw_res
                    else:
                        self.logger.warning(f"⚠️ Unknown response format from L{level}: {type(raw_res)}")
                        level_result = {}

                    # เช็คว่าผลลัพธ์ใช้ได้หรือไม่
                    if level_result and "is_passed" in level_result:
                        break
                        
                except Exception as e:
                    self.logger.error(f"❌ {sub_id} L{level} Attempt {attempt_num} failed: {str(e)}")
                    if attempt_num == MAX_RETRY_ATTEMPTS:
                        level_result = {
                            "level": level, 
                            "is_passed": False, 
                            "reason": f"System Error: {str(e)}", 
                            "score": 0.0
                        }

            # Fallback level in case of failure
            if 'level' not in level_result: level_result['level'] = level

            # --- [SEQUENTIAL & GAP LOGIC] ---
            is_passed_llm = level_result.get('is_passed', False)
            level_result['raw_is_passed'] = is_passed_llm 

            if not is_passed_llm and first_failure_level is None:
                first_failure_level = level
                level_result.update({"display_status": "FAILED", "gap_type": "PRIMARY_GAP"})
            elif is_passed_llm and first_failure_level is not None:
                level_result.update({"display_status": "PASSED (CAPPED)", "gap_type": "SEQUENTIAL_GAP", "is_passed": False})
            elif not is_passed_llm and first_failure_level is not None:
                level_result.update({"display_status": "FAILED (GAP)", "gap_type": "COMPOUND_GAP"})
            else:
                current_sequential_pass_level = level
                level_result.update({"display_status": "PASSED", "gap_type": "NONE"})

            # --- [DATA MAPPING for UI & DASHBOARD] ---
            # สกัดค่า PDCA อย่างละเอียดเพื่อส่งให้หน้า Dashboard (Router)
            pdca_raw = level_result.get('pdca_breakdown', {})
            pdca_final = {
                "P": float(pdca_raw.get('P', 0.0)),
                "D": float(pdca_raw.get('D', 0.0)),
                "C": float(pdca_raw.get('C', 0.0)),
                "A": float(pdca_raw.get('A', 0.0))
            }

            level_details_map[str(level)] = {
                "level": level,
                "is_passed": level_result.get('is_passed', False),
                "raw_is_passed": level_result.get('raw_is_passed', False),
                "score": level_result.get('score', 0.0),
                "pdca_breakdown": pdca_final, # สำหรับวาดกราฟ PDCA
                "reason": level_result.get('reason', ""),
                "summary_thai": level_result.get('summary_thai', ""),
                "coaching_insight": level_result.get('coaching_insight', ""),
                "display_status": level_result.get("display_status", "UNKNOWN"),
                "gap_type": level_result.get("gap_type", "NONE"),
                "evidences": level_result.get('temp_map_for_level', []), # รายชื่อไฟล์ PDF/PNG
                "audit_confidence": level_result.get('audit_confidence', {})
            }
            raw_results_for_sub_seq.append(level_result)

        # -----------------------------------------------------------
        # 3. FINAL SYNTHESIS
        # -----------------------------------------------------------
        action_plan_result = self._generate_action_plan_safe(
            sub_id, sub_criteria_name, current_enabler, raw_results_for_sub_seq
        )
        
        weighted_score = round(self._calculate_weighted_score(current_sequential_pass_level, sub_weight), 2)
        current_sub_map = {k: v for k, v in self.evidence_map.items() if k.startswith(f"{sub_id}.")}

        final_output = {
            "sub_id": sub_id,
            "sub_criteria_id": sub_id,
            "sub_criteria_name": sub_criteria_name,
            "highest_pass_level": current_sequential_pass_level, 
            "level_details": level_details_map,
            "weight": sub_weight,
            "weighted_score": weighted_score,
            "target_level_achieved": current_sequential_pass_level >= self.config.target_level,
            "action_plan": action_plan_result, 
            "raw_results_ref": raw_results_for_sub_seq,
            "worker_duration_s": round(time.time() - start_ts, 2)
        }

        return final_output, current_sub_map

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
        [ULTIMATE ASSEMBLY v2026.6 - INTEGRATED]
        ระบบ Orchestrator ที่ผสานความเสถียรจาก Main และความฉลาดจาก PDCA Branch
        """
        start_ts = time.time()
        self.is_sequential = sequential
        self.current_record_id = record_id 

        # ============================== 1. กรองเกณฑ์การประเมิน ==============================
        all_statements = self._flatten_rubric_to_statements()
        is_all = str(target_sub_id).lower() == "all"
        sub_criteria_list = all_statements if is_all else [
            s for s in all_statements if str(s.get('sub_id')).lower() == str(target_sub_id).lower()
        ]

        if not sub_criteria_list:
            return self._create_failed_result(record_id, f"Criteria '{target_sub_id}' not found", start_ts)

        self.logger.info(f"🎯 Target: {target_sub_id} | Record ID: {record_id}")

        # ============================== 2. ระบบ Resumption (Load Baseline) ==============================
        # ดึงไฟล์หลักฐานที่เคยหาเจอ (Evidence Map) มาเป็นตัวช่วยตั้งต้นให้ RAG
        self.evidence_map = {}
        loaded_data = self._load_evidence_map()
        if loaded_data:
            # ตรวจสอบ record_id เพื่อความปลอดภัย (Origin Main Logic)
            if isinstance(loaded_data, dict) and loaded_data.get("record_id") == record_id:
                self.evidence_map = loaded_data.get("evidence_map", {})
                self.logger.info(f"🔄 Resumed Evidence Map: {len(self.evidence_map)} keys loaded")

        # ============================== 3. ตั้งค่าการรัน (Parallel vs Sequential) ==============================
        max_workers = int(os.environ.get('MAX_PARALLEL_WORKERS', 4))
        run_parallel = is_all and not sequential
        
        self.raw_llm_results = []
        self.final_subcriteria_results = []
        results_list = []

        # ============================== 4. Execution Phase ==============================
        if run_parallel:
            # --- โหมดรันขนาน (Parallel) ---
            self.logger.info(f"🚀 Starting Parallel Assessment (Workers: {max_workers})")
            # เตรียม Argument ให้ Worker (ต้องมี record_id ติดไปด้วย)
            worker_args = [self._prepare_worker_tuple(s, document_map) for s in sub_criteria_list]
            try:
                ctx = multiprocessing.get_context('spawn')
                with ctx.Pool(processes=max_workers) as pool:
                    results_list = pool.map(_static_worker_process, worker_args)
            except Exception as e:
                self.logger.critical(f"❌ Parallel execution failed: {e}")
                raise
        else:
            # --- โหมดรันทีละขั้นตอน (Sequential) ---
            self.logger.info(f"🧵 Starting Sequential Assessment: {target_sub_id}")
            vsm = vectorstore_manager or self._init_local_vsm()
            for sub_criteria in sub_criteria_list:
                # ส่ง record_id เข้าไปเพื่อให้ _run_single_assessment ทำงานได้
                res = self._run_sub_criteria_assessment_worker(sub_criteria, vsm)
                results_list.append(res)

        # ============================== 5. Integration (Merge & Normalize) ==============================
        # รวมผลลัพธ์จาก Worker ทั้งหมด และจัดการเรื่อง Metadata
        new_merged_map = self.merge_evidence_mappings(results_list)
        
        for key, evidences in new_merged_map.items():
            # ✨ Normalize Metadata ก่อนบันทึก (New PDCA Branch Logic)
            self._normalize_evidence_metadata(evidences)
            
            if key not in self.evidence_map:
                self.evidence_map[key] = evidences
            else:
                # กันข้อมูลซ้ำ (Unique by doc_id/chunk_uuid)
                existing_ids = {str(e.get('doc_id') or e.get('chunk_uuid')) for e in self.evidence_map[key]}
                for ev in evidences:
                    if str(ev.get('doc_id') or ev.get('chunk_uuid')) not in existing_ids:
                        self.evidence_map[key].append(ev)

        # แยกผลลัพธ์ LLM ออกมาทำรายงาน
        for res in results_list:
            if isinstance(res, tuple) and len(res) == 2:
                sub_result, _ = res
                self.raw_llm_results.extend(sub_result.get("raw_results_ref", []))
                self.final_subcriteria_results.append(sub_result)

        # ============================== 6. Persistence (Save Baseline) ==============================
        if self.evidence_map:
            try:
                save_payload = {
                    "record_id": record_id,
                    "evidence_map": self.evidence_map,
                    "timestamp": datetime.now().isoformat()
                }
                self._save_evidence_map(map_to_save=save_payload)
                self.logger.info(f"✅ Baseline Saved for Record: {record_id}")
            except Exception as e:
                self.logger.error(f"❌ Persistence failed: {e}")

        # ============================== 7. Final Summary & Analytics ==============================
        self._calculate_overall_stats(target_sub_id)
        
        # รวบรวม Coaching Insights สำหรับเล่มรายงาน (New Branch Feature)
        all_insights = []
        for res in self.final_subcriteria_results:
            details = res.get('level_details', {})
            for lvl, data in details.items():
                if data.get('coaching_insight'):
                    all_insights.append({
                        "sub_id": res.get('sub_id'),
                        "level": int(lvl),
                        "text": data['coaching_insight'],
                        "status": "passed" if data.get('is_passed') else "failed"
                    })
        self.total_stats['global_coaching_brief'] = all_insights

        return {
            "record_id": record_id,
            "summary": self.total_stats,
            "sub_criteria_results": self.final_subcriteria_results,
            "run_time_seconds": round(time.time() - start_ts, 2),
            "timestamp": datetime.now().isoformat(),
            "export_path": self._export_results(self.final_subcriteria_results, target_sub_id, record_id) if export else None
        }

    def _merge_worker_results(self, sub_result: Dict[str, Any], temp_map: Dict[str, List[Dict]]):
        """
        [INTERNAL HELPER] รวมผลลัพธ์จาก Worker เข้าสู่ตัวแปรหลักของ Engine
        - ทำการ Normalize Metadata เพื่อให้ชื่อไฟล์และหน้าถูกต้อง
        - รวมผลลัพธ์ LLM และความเชื่อมโยงของหลักฐาน (Evidence Map)
        """
        if not sub_result:
            return

        # 1. จัดการ Evidence Map (ข้อมูลหลักฐานที่ค้นพบ)
        if temp_map and isinstance(temp_map, dict):
            for level_key, evidence_list in temp_map.items():
                if not evidence_list:
                    continue
                
                # Normalize Metadata ทันที (ตรวจสอบชื่อไฟล์/หน้า) เพื่อให้สอดคล้องกับ seam_prompts.py
                self._normalize_evidence_metadata(evidence_list)
                
                # นำไปรวมใน map หลักของ Engine
                if level_key not in self.evidence_map:
                    self.evidence_map[level_key] = []
                self.evidence_map[level_key].extend(evidence_list)

        # 2. เก็บผลลัพธ์ดิบจาก LLM (สำหรับ Ref. และ Debug)
        if "raw_results_ref" in sub_result:
            self.raw_llm_results.extend(sub_result["raw_results_ref"])

        # 3. เก็บผลการประเมินรายเกณฑ์ (Final Results per Sub-ID)
        self.final_subcriteria_results.append(sub_result)
        
        self.logger.debug(f"✅ Merged results for {sub_result.get('sub_criteria_id', 'Unknown')}")
    
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
        [EXPERT LOOP v2026.45] 
        - ระบบอุทธรณ์พิเศษเมื่อ Rerank score สูง (>0.75) แต่ LLM ให้ตก
        - ใช้ 'Substance over Form' กระตุ้นให้ AI มองหาพฤติกรรมจริงแทน Keyword
        """
        self.logger.info(f"🔍 [EXPERT-RE-EVAL] Triggered for {sub_id} L{level} | Rerank: {highest_rerank_score:.3f}")

        missing_str = ", ".join(sorted(set(missing_tags))) if missing_tags else "PDCA ที่ครบถ้วน"

        hint_msg = f"""
        --- ⚠️ [EXPERT RE-ASSESSMENT NOTICE] ---
        ผลการประเมินรอบแรก: "ไม่ผ่าน" (เหตุผล: {first_attempt_reason[:150]}...)
        
        ข้อมูลจากระบบวิเคราะห์เชิงลึก:
        - พบหลักฐานที่สอดคล้องสูงมาก (Highest Rerank: {highest_rerank_score:.4f})
        - จุดที่ตรวจไม่พบในรอบแรก: {missing_str}
        
        คำสั่งสำหรับผู้เชี่ยวชาญ:
        โปรดวิเคราะห์เนื้อหาแบบ 'Substance over Form' อีกครั้ง 
        หากบริบทของเอกสารแสดงถึงการปฏิบัติจริง (Do) หรือการติดตาม (Check) แม้จะไม่ใช้คำศัพท์ตาม Keyword เป๊ะ 
        ท่านสามารถพิจารณาให้คะแนนตามระดับ Maturity จริงที่ปรากฏในหลักฐานได้
        """

        expert_kwargs = base_kwargs.copy()
        expert_kwargs["context"] = f"{context}\n\n{hint_msg}"
        expert_kwargs["sub_criteria_name"] = f"{sub_criteria_name} (Expert Re-assessment)"
        
        # รอบสองใช้ temperature 0 เพื่อความนิ่งที่สุด
        expert_kwargs["temperature"] = 0.0

        try:
            re_eval_result = llm_evaluator_to_use(**expert_kwargs)
            
            # บันทึกสถานะว่าผ่านการอุทธรณ์
            re_eval_result["is_expert_evaluated"] = True
            re_eval_result["original_fail_reason"] = first_attempt_reason
            
            if re_eval_result.get("is_passed", False):
                self.logger.info(f"🛡️ [EXPERT-OVERRIDE] {sub_id} L{level} REVERSED to PASSED!")
                re_eval_result["reason"] = f"[Expert Pass]: {re_eval_result.get('reason', '')}"
            
            return re_eval_result
        except Exception as e:
            self.logger.error(f"🛑 Expert Eval Error: {str(e)}")
            return {"is_passed": False, "score": 0.0, "reason": f"Expert Eval Failure: {str(e)}"}
    
    def _apply_diversity_filter(self, evidences: List[Dict], level: int) -> List[Dict]:
        if not evidences:
            return []

        sorted_evidences = sorted(
            evidences,
            key=lambda x: x.get('rerank_score', 0) or x.get('priority_score', 0),
            reverse=True
        )

        if level <= 2:
            return sorted_evidences[:20]  # เพิ่มจาก 15 เป็น 20 เพื่อให้มีโอกาสเห็น D

        diverse_results = []
        file_counts = defaultdict(int)
        per_file_limit = 5  # เพิ่มจาก 4 เป็น 5
        max_total = 30  # เพิ่มจาก 25 เป็น 30

        for ev in sorted_evidences:
            source = ev.get('metadata', {}).get('source_filename') or 'Unknown'
            source = os.path.basename(str(source))
            file_counts[source] += 1

            if file_counts[source] <= per_file_limit:
                diverse_results.append(ev)

            if len(diverse_results) >= max_total:
                break

        return diverse_results
    
    def _normalize_evidence_metadata(self, evidence_list: List[Dict[str, Any]]):
        """
        ปรับปรุงให้รองรับทั้ง Flattened และ Nested Metadata 
        เพื่อให้ระบบรายงาน (Export) ดึงค่าไปใช้ได้โดยไม่ Error
        """
        for ev in evidence_list:
            if not isinstance(ev, dict):
                continue
                
            # 1. ดึง Metadata หลัก (ถ้ามี)
            meta = ev.get("metadata", {}) if isinstance(ev.get("metadata"), dict) else {}
            
            # 2. ปรับ Source (เช็คทั้งนอกและใน meta)
            raw_source = ev.get("source") or meta.get("source") or ev.get("source_filename") or meta.get("source_filename")
            ev["source"] = os.path.basename(str(raw_source)) if raw_source else "Unknown_File"
            
            # 3. ปรับ Page (ให้เป็น String เสมอเพื่อป้องกัน JSON Error)
            raw_page = ev.get("page") or meta.get("page") or meta.get("page_label") or "N/A"
            ev["page"] = str(raw_page)
            
            # 4. ปรับ Relevance Score (รองรับหลายชื่อ Key)
            raw_score = ev.get("relevance_score") or ev.get("score") or meta.get("rerank_score") or 0.0
            try:
                ev["relevance_score"] = float(raw_score)
            except (ValueError, TypeError):
                ev["relevance_score"] = 0.0
            
            # 5. ตรวจสอบ ID สำคัญ (สำหรับโยงใย Database)
            if not ev.get("stable_doc_uuid"):
                ev["stable_doc_uuid"] = ev.get("doc_id") or meta.get("stable_doc_uuid") or "unknown_uuid"
            
            # 6. เพิ่มฟิลด์แสดงชื่อไฟล์แบบสวยๆ (Optional - สำหรับโชว์ใน Log)
            ev["source_filename"] = ev["source"]

        return evidence_list

    def relevance_score_fn(self, evidence: Dict[str, Any], sub_id: str, level: int) -> float:
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
        cum_rules = self.get_cumulative_rules(sub_id, level)

        # 4. Source Grading
        source_bonus = 0.0
        primary = ["มติ", "บันทึก", "คำสั่ง", "ประกาศ", "นโยบาย", "แผนแม่บท", "มติบอร์ด"]
        secondary = ["assessment report", "รายงานการประเมิน", "สรุปผล", "รายงานผล"]
        if any(p in filename for p in primary):
            source_bonus += 0.25
        if any(p in filename for p in secondary):
            source_bonus -= 0.15

        # 5. Keyword Score (35%)
        target_kws = set()
        if level <= 2:
            target_kws.update(cum_rules.get('plan_keywords', []) + cum_rules.get('do_keywords', []))
        else:
            target_kws.update(cum_rules.get('check_keywords', []) + cum_rules.get('act_keywords', []))

        match_count = sum(1 for kw in target_kws if kw.lower() in text)
        expected = max(1, len(target_kws) * 0.25)
        keyword_score = min((match_count / expected) ** 0.5, 1.0)
        keyword_score = max(keyword_score, 0.15 if match_count >= 1 else 0.0)

        # 6. PDCA Tag Bonus (สมดุล 0.30)
        pdca_bonus = 0.0
        pdca_tag = evidence.get('pdca_tag') or meta.get('pdca_tag')
        if pdca_tag and str(pdca_tag).upper() in {'P', 'D', 'C', 'A'}:
            pdca_bonus = 0.30

        # 7. Neighbor Bonus
        neighbor_bonus = 0.20 if evidence.get('is_neighbor', False) or meta.get('is_neighbor', False) else 0.0

        # 8. Specific Rule Bonus
        specific_rule = cum_rules.get('specific_contextual_rule', '').lower()
        rule_bonus = 0.20 if specific_rule and specific_rule in text else 0.0

        # 9. รวมคะแนน (45% Rerank + 35% Keyword + 20% Bonuses)
        final_score = (
            0.45 * normalized_rerank +
            0.35 * keyword_score +
            source_bonus + pdca_bonus + neighbor_bonus + rule_bonus
        )

        # 10. Min floor สำหรับ rerank สูง (ป้องกัน miss good evidence)
        if normalized_rerank > 0.80:
            final_score = max(final_score, 0.40)

        final_score = min(max(final_score, 0.0), 1.0)

        # 11. Logging (ทั่วไป ไม่เฉพาะ 1.2)
        if normalized_rerank > 0.75:
            self.logger.info(
                f"[HIGH-RERANK] {sub_id} L{level} | "
                f"rerank={rerank_score:.4f} | kw={keyword_score:.4f} | "
                f"pdca_bonus={pdca_bonus:.3f} | final={final_score:.4f} | "
                f"tag={pdca_tag} | text={text[:100]}..."
            )

        self.logger.debug(
            f"[{sub_id} L{level}] RelScore: {final_score:.4f} | Rerank: {normalized_rerank:.4f} | "
            f"KW: {keyword_score:.4f} | Src: {source_bonus:.3f} | PDCA: {pdca_bonus:.3f}"
        )

        return final_score

    def enhance_query_for_statement(
        self,
        statement_text: str,
        sub_id: str,
        statement_id: str,
        level: int,
        focus_hint: str = "",
    ) -> List[str]:
        """
        [REVISED STRATEGIC v2026] 
        เน้นแก้ปัญหา Bias หน้าแรก และเพิ่มความคมชัดของหลักฐานในระดับ Maturity สูง
        """
        logger = logging.getLogger(__name__)
        
        # 1. เตรียมอัตลักษณ์พื้นฐาน (Anchors)
        enabler_id = getattr(self.config, 'enabler', 'KM').upper()
        id_anchor = f"{enabler_id} {sub_id}"
        tenant_name = getattr(self.config, 'tenant', 'PEA').upper()
        
        # 2. รวบรวมวัตถุดิบ Keywords จาก Rules แบบ Cumulative
        raw_kws = []
        
        # ดึงกฎเหล็กประจำข้อ
        must_list = self.get_rule_content(sub_id, level, "must_include_keywords")
        if isinstance(must_list, list): raw_kws.extend(must_list)
        
        # 🟢 Strategic Selection: แยกคำค้นหาตามระดับเพื่อหนีเอกสารหน้า 1-2
        if level <= 2:
            # L1-L2: เน้นแผนและนโยบาย
            raw_kws.extend(self.get_rule_content(sub_id, 1, "plan_keywords") or [])
            raw_kws.extend(self.get_rule_content(sub_id, 2, "do_keywords") or [])
        else:
            # L3-L5: เน้นการปฏิบัติและผลลัพธ์ (❌ จงใจไม่เอา plan_keywords)
            raw_kws.extend(self.get_rule_content(sub_id, 2, "do_keywords") or [])
            raw_kws.extend(self.get_rule_content(sub_id, 3, "check_keywords") or [])
            if level >= 4:
                raw_kws.extend(self.get_rule_content(sub_id, 4, "act_keywords") or [])
            if level >= 5:
                raw_kws.extend(self.get_rule_content(sub_id, 5, "act_keywords") or [])

        # ล้างคำซ้ำและจำกัดจำนวนคำเพื่อไม่ให้ Query กระจัดกระจาย
        clean_kws_list = sorted(list(set(str(k).strip() for k in raw_kws if k)))
        keywords_str = " ".join(clean_kws_list[:12])

        # 3. สร้างชุด Queries แบบ Diversified
        queries = []
        # ตัดส่วนขยาย 'เช่น' ออกเพื่อให้ Embedding จับใจความสำคัญได้ดีขึ้น
        clean_stmt = statement_text.split("เช่น")[0].strip()

        # Query 1: Core Precision (Anchor + Statement + Keywords)
        queries.append(f"{id_anchor} {clean_stmt} {keywords_str}")

        # Query 2: Evidence Type Targeting (ระบุประเภทไฟล์หลักฐาน)
        if level <= 2:
            queries.append(f"ประกาศ คำสั่ง แนวทาง ระเบียบปฏิบัติ {id_anchor} {keywords_str}")
        else:
            # บังคับหาภาคผนวกและรายงานผล
            queries.append(f"รายงานสรุปผล KPI สถิติ ภาคผนวก รายละเอียดแนบท้าย {id_anchor} {keywords_str}")

        # Query 3: Organization Context (Tenant Specific)
        queries.append(f"{tenant_name} {id_anchor} {clean_stmt}")

        # Query 4: PDCA Synonyms (จาก Global Vars)
        synonyms = ""
        try:
            from config.global_vars import PDCA_LEVEL_SYNONYMS
            synonyms = PDCA_LEVEL_SYNONYMS.get(level, "")
        except ImportError:
            fallback = {1: "แผน", 2: "ปฏิบัติ", 3: "ประเมิน", 4: "ปรับปรุง", 5: "ยั่งยืน"}
            synonyms = fallback.get(level, "")
        
        if synonyms:
            queries.append(f"{id_anchor} {synonyms} {keywords_str}")

        # Query 5: Advanced Maturity (L4-L5 เท่านั้น)
        if level >= 4:
            queries.append(f"นวัตกรรม Best Practice บทเรียนที่ได้รับ Lesson Learned {id_anchor}")

        # 4. Final Processing & Truncation
        final_queries = []
        seen = set()
        for q in queries:
            q_norm = " ".join(q.split()[:25]) # จำกัด 25 คำเพื่อความคมของ Rerank
            if q_norm and q_norm not in seen:
                final_queries.append(q_norm)
                seen.add(q_norm)
        
        logger.info(f"🚀 [Query Gen] {sub_id} L{level} | Generated {len(final_queries[:5])} refined queries.")
        return final_queries[:5]
    
    def _get_semantic_tag(self, text: str, sub_id: str, level: int, filename: str = "") -> str:
        """
        [ULTIMATE REVISE v2026.12] 
        - แก้ไขปัญหา "ดูงาน/อบรม" หลุดไปเป็น P โดยใช้ Filename + Linguistic Analysis
        - ใช้ Strict Zero-Tolerance สำหรับการแยกแยะ Plan และ Do
        """
        # 1. นิยามที่เน้นความแตกต่างระหว่าง 'เจตนา' (P) และ 'หลักฐานประจักษ์' (D)
        system_prompt = """
        You are a KM Audit Specialist for PEA. Classify the text into P, D, C, or A.
        
        STRICT RULES:
        - P (Plan): "สารตั้งต้น" - แผนปฏิบัติการ, คำสั่งแต่งตั้ง, ยุทธศาสตร์, งบประมาณ, แนวทางที่ยังไม่ได้ทำ.
        - D (Do): "หลักฐานการทำจริง" - **ถ้ามีคำว่า 'สรุปผล', 'ภาพถ่าย', 'รายงานการประชุม', 'รายชื่อผู้เข้าร่วม', 'ดูงาน' หรือ 'กิจกรรมที่เสร็จสิ้นแล้ว' ให้ตอบ D เท่านั้น**
        - C (Check): "การประเมิน" - รายงาน KPI, ผลประเมินความพึงพอใจ, การติดตามความคืบหน้า.
        - A (Act): "การปรับปรุง" - บทเรียนที่ได้รับ (AAR), การปรับปรุงระบบจากข้อเสนอแนะ.
        """
        
        user_prompt = f"""
        Analyze this KM Evidence:
        ---
        Source Filename: "{filename}"
        Text: "{text[:800]}"
        ---
        CRITICAL CHECK:
        - หากชื่อไฟล์หรือเนื้อหามีคำว่า "ดูงาน", "ภาพกิจกรรม", "สรุปผลการ..." หรือเห็นเป็นรูปภาพ (.png, .jpg) -> **ตอบ "D" ทันที**
        - หากเป็น "คำสั่ง" หรือ "แผน" -> **ตอบ "P"**
        
        Return ONLY JSON: {{"tag": "P/D/C/A/Other", "reason": "thai_reason"}}
        """
            
        try:
            # ใช้ temperature=0 เพื่อความแม่นยำสูงสุด
            response_json_str = _fetch_llm_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                llm_executor=self.llm,
                max_retries=2
            )
            
            import json
            data = json.loads(response_json_str)
            tag = data.get('tag', 'Other').strip().upper()
            
            # Validation logic
            valid_tags = ['P', 'D', 'C', 'A']
            return tag if tag in valid_tags else 'Other'
            
        except Exception as e:
            self.logger.error(f"[SEMANTIC-TAG-ERROR] {sub_id} L{level}: {str(e)}")
            return 'Other'
        
    def _build_pdca_context(self, blocks: Dict[str, str]) -> str:
        """
        [REVISE] รวม PDCA Blocks จาก Dictionary เป็น XML Context
        รองรับ Key: Plan, Do, Check, Act, Other
        """
        tags = ["Plan", "Do", "Check", "Act", "Other"]
        xml_parts = []
        for t in tags:
            content = blocks.get(t, "N/A")
            xml_parts.append(f"<{t}>\n{content}\n</{t}>")
        return "\n".join(xml_parts)

    def _log_pdca_status(self, sub_id, name, level, blocks, req_phases, sources_count, score, conf_level, **kwargs):
        """
        [ULTIMATE LOG v2026.1.14] 
        - มั่นใจว่าแสดง Level Name (ผู้บริหารแสดงความมุ่งมั่น...) แน่นอน
        - รองรับ Argument ส่วนเกินด้วย **kwargs (ป้องกัน Crash)
        - จัด Format ไอคอน PDCA ให้สแกนด้วยสายตาง่าย
        """
        try:
            # 1. 🛡️ Guard & Format Level Name
            # รับค่า name (ซึ่งเราส่ง stmt มาจากฟังก์ชันหลัก)
            raw_name = str(name) if name else "No Level Statement Defined"
            # ตัดคำถ้าเกิน 60 ตัวอักษรเพื่อให้ Log ไม่ยาวเกินหน้าจอ
            display_name = (raw_name[:57] + "...") if len(raw_name) > 60 else raw_name

            # 2. 🛡️ Guard Blocks (PDCA Results)
            if not isinstance(blocks, dict):
                blocks = {}

            # 3. 🛡️ Guard & Format Required Phases
            if not isinstance(req_phases, list):
                req_phases = [str(req_phases)]
            req_str = f"[{','.join(map(str, req_phases))}]"

            # 4. ⚙️ Build PDCA Icons (P D C A)
            mapping = [("Plan", "P"), ("Do", "D"), ("Check", "C"), ("Act", "A")]
            icons_list = []
            for full_phase, short_phase in mapping:
                content = blocks.get(full_phase, "")
                # เช็คว่ามีข้อมูลและไม่ใช่ "N/A"
                is_valid = content and str(content).strip().upper() != "N/A"
                status_icon = "✅" if is_valid else "❌"
                icons_list.append(f"{short_phase}:{status_icon}")
            
            icons_str = " ".join(icons_list)

            # 5. 🏷️ Extra Metadata (เช่น ชื่อหัวข้อหลัก 1.2)
            rubric_title = kwargs.get('rubric_name', '')
            extra_info = f" | {rubric_title}" if rubric_title else ""

            # 📊 [FINAL OUTPUT] พ่น Log สรุปความลัดเป๊ะๆ
            self.logger.info(
                f"📊 [PDCA-STATUS] {sub_id} L{level} | {display_name}{extra_info} | "
                f"Req:{req_str} | Res:{icons_str} | "
                f"Docs:{sources_count} | Score:{score:.4f} | Conf:{conf_level}"
            )

        except Exception as e:
            # ป้องกันไม่ให้การ Log ทำระบบพัง แต่ให้แจ้ง Error ไว้
            self.logger.error(f"❌ Critical Error in _log_pdca_status: {e}")

    def _perform_adaptive_retrieval(self, sub_id, level, stmt, vectorstore_manager):
        """ 
        [STRATEGIC REVISE v2026.Expert] 
        - รองรับ Priority Documents (User-Specified) แบบ 100%
        - ผสานระบบ Discovery เพื่อหาหลักฐานมาเติมเต็ม PDCA ที่ขาดหาย
        - เพิ่มระบบ Early Exit และ Metadata Reinforcement
        """
        # 1. ดึงเอกสารที่ User ระบุมา (Priority) หรือที่มีการ Mapping ไว้ล่วงหน้า
        # mapped_ids: สำหรับใช้กรองใน VectorStore, priority_docs: ก้อนเอกสารพร้อมประเมิน
        mapped_ids, priority_docs = self._get_mapped_uuids_and_priority_chunks(
            sub_id, level, stmt, vectorstore_manager
        )
        
        candidates = []
        final_max_rerank = 0.0
        
        # ใส่เครื่องหมายแสดงใน Log เพื่อให้รู้ว่ามีเอกสาร "สั่งตรวจ" มา
        if priority_docs:
            self.logger.info(f"📌 [TARGETED-AUDIT] Found {len(priority_docs)} priority chunks for {sub_id} L{level}")
            # คำนวณค่า Rerank พื้นฐานจาก Priority Docs ก่อน
            if any(p.get('rerank_score') for p in priority_docs):
                final_max_rerank = max((float(p.get('rerank_score', 0)) for p in priority_docs), default=0.0)

        # 2. สร้าง Queries เพื่อ "ค้นหาหลักฐานส่วนเพิ่ม" (Discovery Mode)
        # แม้ User จะให้เอกสารมาแล้ว แต่ระบบจะหาเพิ่มเพื่อเช็คว่ามีอะไรที่ "ดีกว่า" หรือ "มาเติมเต็ม" หรือไม่
        queries = self.enhance_query_for_statement(stmt, sub_id, f"{sub_id}.L{level}", level)
        
        # 3. Retrieval Loop (Adaptive 3-Loop)
        for i, q in enumerate(queries[:3]):
            # ค้นหาในฐานข้อมูลโดยเปิดกว้าง (แต่ถ้ามี mapped_ids ระบบจะสนใจส่วนนั้นเป็นพิเศษ)
            res = self.rag_retriever(
                query=q, 
                doc_type=self.doc_type, 
                sub_id=sub_id, 
                level=level,
                vectorstore_manager=vectorstore_manager, 
                stable_doc_ids=mapped_ids 
            )
            
            loop_docs = res.get("top_evidences", [])
            
            if loop_docs:
                # อัปเดตคะแนนความสอดคล้องสูงสุด
                current_max = max((float(c.get('rerank_score', 0)) for c in loop_docs), default=0.0)
                final_max_rerank = max(final_max_rerank, current_max)
                
                # กรองเอาเฉพาะเอกสารที่ไม่ซ้ำกับ Priority ที่มีอยู่แล้ว
                new_docs = [
                    d for d in loop_docs 
                    if d.get('chunk_uuid') not in [p.get('chunk_uuid') for p in priority_docs]
                ]
                candidates.extend(new_docs)
            
            # ✨ Early Exit Logic: ถ้าคะแนนสูงเกิน 0.88 และได้เอกสารมากพอแล้ว ให้หยุดค้นหาเพิ่ม
            if final_max_rerank >= 0.88 and len(candidates) >= 10:
                self.logger.info(f"🎯 High relevance found ({final_max_rerank:.4f}). Optimizing speed by stopping loop.")
                break

        # 4. Final Integration & Scoring Reinforcement
        # นำเอกสารที่ User สั่ง (Priority) รวมกับที่ AI หามาได้ (Discovery)
        all_retrieved = priority_docs + candidates
        
        # เสริม Metadata ให้พร้อมแสดงผลบน UI (เช่น ชื่อไฟล์, หน้า)
        self._normalize_evidence_metadata(all_retrieved)
        
        # 5. Safety Net: ตรวจสอบว่า Priority Docs ต้องอยู่ครบ (ไม่ถูก Diversify ทิ้งในขั้นตอนถัดไป)
        for p in priority_docs:
            p['is_priority'] = True # ทำ Tag พิเศษไว้
            p['rerank_score'] = max(p.get('rerank_score', 0), 0.70) # บังคับ Floor Score ให้เอกสารที่ตั้งใจเลือกมา

        self.logger.info(f"🏁 Retrieval Finished: Total {len(all_retrieved)} units (Priority: {len(priority_docs)}, Discovered: {len(candidates)})")

        return all_retrieved, final_max_rerank

    def _run_single_assessment(
        self, 
        sub_criteria: Dict[str, Any], 
        statement_data: Dict[str, Any], 
        vectorstore_manager: Optional['VectorStoreManager'], 
        **kwargs
    ) -> Dict[str, Any]:
        """ [ULTIMATE FINISH - v2026.1.14] """
        start_time = time.time()
        
        # --- 🛡️ 1. เตรียมค่าและดึงประโยคเกณฑ์ ---
        sub_id = str(sub_criteria.get('sub_id', 'Unknown'))
        level = statement_data.get('level', 1)
        level_idx = str(level)
        name = str(sub_criteria.get('name', sub_criteria.get('sub_criteria_name', 'No Title')))

        levels_map = sub_criteria.get('levels', {})
        target_val = levels_map.get(level_idx, "") if isinstance(levels_map, dict) else ""
        if isinstance(target_val, dict):
            stmt = str(target_val.get('statement', ''))
        else:
            stmt = str(target_val) or f"เกณฑ์ระดับ {level}"

        self.logger.info(f"🚀 [AUDIT START] {sub_id} L{level} | {name}")
        self.logger.info(f"📌 เกณฑ์ระดับนี้: {stmt}") 

        # --- 🛡️ 2. Retrieval & Rules ---
        try:
            all_candidates, raw_max_score = self._perform_adaptive_retrieval(sub_id, level, stmt, vectorstore_manager)
        except:
            all_candidates, raw_max_score = [], 0.0

        rules_map = getattr(self, 'contextual_rules_map', {})
        current_rules = rules_map.get(sub_id, {}).get(level_idx, {}) if isinstance(rules_map, dict) else {}

        # --- 🛡️ 3. กรองและรวบรวม Blocks (จุดสำคัญที่ทำให้ Res: ✅) ---
        diverse_docs = self._apply_diversity_filter(all_candidates, level)
        
        # เรียกใช้ฟังก์ชันที่เราปรับใหม่ (คืนค่าเป็น Dict)
        blocks = self._get_pdca_blocks_from_evidences(diverse_docs, None, level, sub_id, rules_map)

        # --- 🛡️ 4. สรุปผล Log ---
        req_phases = current_rules.get('require_phase') or (['P','D'] if level <= 2 else ['P','D','C'])
        display_score = raw_max_score if raw_max_score > 0 else (0.85 if diverse_docs else 0.0)

        if hasattr(self, '_log_pdca_status'):
            self._log_pdca_status(
                sub_id=sub_id, name=stmt, level=level, blocks=blocks, 
                req_phases=req_phases, sources_count=len(diverse_docs), 
                score=display_score, conf_level="High", rubric_name=name
            )

        # --- 🛡️ 5. LLM Evaluation ---
        ctx = self._build_pdca_context(blocks) 
        eval_fn = evaluate_with_llm_low_level if level <= 2 else evaluate_with_llm
        res = eval_fn(
            context=f"{ctx}\n\n{self._get_level_constraint_prompt(sub_id, level)}", 
            sub_criteria_name=name, level=level, statement_text=stmt, 
            sub_id=sub_id, llm_executor=self.llm, require_phase=req_phases
        )
        
        # Final Guard
        res = self.post_process_llm_result(res, level, sub_id=sub_id)
        if not hasattr(self, 'level_details_map'): self.level_details_map = {}
        self.level_details_map[str(level)] = res

        return {
            "sub_criteria_id": sub_id, "level": level, "score": res.get('score', display_score),
            "is_passed": res.get('is_passed', False), "reason": res.get('reason', ""),
            "duration": round(time.time() - start_time, 2)
        }