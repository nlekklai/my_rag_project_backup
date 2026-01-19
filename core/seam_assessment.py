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
import psutil
import time

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

    # Mock Logic Functions
    def create_structured_action_plan(*args, **kwargs): return []
    def evaluate_with_llm(*args, **kwargs): return {"score": 0, "reason": "Import Error Fallback", "is_passed": False}
    def retrieve_context_with_filter(*args, **kwargs): return {"top_evidences": [], "aggregated_context": ""}
    def retrieve_context_for_low_levels(*args, **kwargs): return {"top_evidences": [], "aggregated_context": ""}
    def evaluate_with_llm_low_level(*args, **kwargs): return {"score": 0, "is_passed": False}
    def set_llm_data_mock_mode(mode): pass
    
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
        self.llm_evaluator = evaluate_with_llm
        self.rag_retriever = retrieve_context_with_filter
        self.create_structured_action_plan = create_structured_action_plan
        self.ActionPlanActions = ActionPlanActions
        self.action_plan_model = ActionPlanResult

        # --- [PATCH v2026.1.17] State Management Initialization ---
        self.final_subcriteria_results = []
        self.total_stats = {}
        self.raw_llm_results = []
        self.level_details_map = {} # สำหรับเก็บสถานะ L1-L5 เพื่อทำ Gap Analysis

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
        cum_rules = self.get_cumulative_rules(sub_id, level)

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

        # 1. Specific Level
        if key_type in rule.get(level_key, {}):
            return rule[level_key][key_type]

        # 2. Sub-ID Root
        if key_type in rule:
            return rule[key_type]

        # 3. Global Defaults
        defaults = self.contextual_rules_map.get("_enabler_defaults", {})
        if key_type in defaults:
            return defaults[key_type]

        # 4. Fallback ตามประเภท (ชัดเจน ไม่ใช้ string check)
        fallbacks = {
            "require_phase": None,
            "must_include_keywords": [],
            "plan_keywords": [],
            "do_keywords": [],
            "check_keywords": [],
            "act_keywords": [],
            # เพิ่ม key อื่น ๆ ที่เป็น list ได้ที่นี่
        }
        return fallbacks.get(key_type, "")  # ถ้าไม่รู้จัก key_type → คืน string ว่าง    

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

    def post_process_llm_result(
        self,
        llm_output: Any,
        level: int,
        sub_id: str = None,
        contextual_config: Dict = {},
        top_evidences: List[Dict[str, Any]] = []
    ) -> Dict[str, Any]:
        """
        [POST-PROCESS v2026.1.19 — Production Stable]
        - JSON Repair ทนทานสูงสุด (markdown, trailing comma, calculation text, encoding)
        - Rescue Logic แข็งแรง (keyword + rerank conflict)
        - PDCA Normalization สมดุล (required phases เป็นหลัก)
        - Safety Net ฉลาด (force 1.2 + ฉีด PDCA breakdown)
        - Coaching + Missing phases ชัดเจน
        """
        log_prefix = f"{sub_id or 'Unknown'} L{level}"

        # 1. JSON Repair & Unpacking (Robust v2)
        if isinstance(llm_output, tuple):
            llm_output = llm_output[0] if len(llm_output) > 0 else {}
        if isinstance(llm_output, str):
            try:
                # ล้าง markdown, calculation, trailing comma
                cleaned = re.sub(r'```json\s*|\s*```', '', llm_output)
                cleaned = re.sub(r'(\d+\.?\d*)\s*[\+\-]\s*(\d+\.?\d*)\s*=\s*(\d+\.?\d*)', r'\3', cleaned)
                cleaned = cleaned.strip().replace(",\n}", "\n}").replace(",}", "}")
                cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)  # ลบ trailing comma
                cleaned = cleaned.encode('utf-8', 'ignore').decode('utf-8')  # ลบอักขระพิเศษ
                llm_output = json.loads(cleaned)
            except Exception as e:
                self.logger.error(f"❌ [JSON REPAIR FAILED] {log_prefix}: {str(e)}")
                return {"is_passed": False, "score": 0.0, "reason": "JSON Parsing Error"}

        if not isinstance(llm_output, dict):
            self.logger.error(f"❌ [INVALID OUTPUT] {log_prefix}: Not a dict")
            return {"is_passed": False, "score": 0.0, "reason": "Invalid LLM Output Format"}

        # 2. Required Phases (จาก config หรือ default)
        rubric = contextual_config.get(sub_id, {}).get(str(level), {})
        if level <= 3:
            default_phases = ["P", "D"]
        elif level == 4:
            default_phases = ["P", "D", "C"]
        else:
            default_phases = ["P", "D", "C", "A"]
        required_phases = rubric.get("require_phase", default_phases)

        # 3. PDCA Score Extraction + Smart Rescue
        pdca_results = {"P": 0.0, "D": 0.0, "C": 0.0, "A": 0.0}
        reason_raw = str(llm_output.get('reason', '')).lower()
        extraction_raw = {p: str(llm_output.get(f"Extraction_{p}", "")).lower() for p in ["P", "D", "C", "A"]}

        for phase in ["P", "D", "C", "A"]:
            val = float(llm_output.get(f"{phase}_Plan_Score") or 
                        llm_output.get(f"score_{phase.lower()}") or 
                        llm_output.get(f"{phase}_Score") or 0.0)
            score = min(val, 2.0)

            # Rescue 1: Keyword in reason/extraction
            phase_keywords = rubric.get(f"{phase.lower()}_keywords", [])
            if score < 1.0 and any(kw.lower() in (reason_raw + extraction_raw[phase]) for kw in phase_keywords):
                score = max(score, 1.5)
                self.logger.info(f"🛡️ [RESCUE: KEYWORD] {log_prefix} {phase} boosted to {score}")

            pdca_results[phase] = score

        # 4. Adaptive Normalization
        raw_total_required = sum(pdca_results[p] for p in required_phases)
        max_possible_required = len(required_phases) * 2.0
        normalized_score = (raw_total_required / max_possible_required) * 2.0 if max_possible_required > 0 else 0.0
        normalized_score = round(normalized_score, 2)

        # 5. Rerank Safety Net (แก้ L0 Gap)
        max_rerank = max(ev.get('relevance_score', 0.0) for ev in top_evidences) if top_evidences else 0.0
        is_conflict = (normalized_score < 1.2) and (max_rerank > 0.85)

        if is_conflict:
            normalized_score = 1.2  # Force Pass Threshold
            llm_output["needs_human_review"] = True
            self.logger.info(f"🛡️ [RERANK-SAFETY] Force Passed (1.2) | Rerank: {max_rerank:.2f} | Original: {normalized_score}")

            # ฉีดคะแนนกระจายให้เฟสที่จำเป็น (สมดุล PDCA breakdown)
            min_per_phase = 1.2 / len(required_phases)
            for phase in required_phases:
                if pdca_results[phase] < min_per_phase:
                    pdca_results[phase] = round(min_per_phase + 0.1, 2)
            llm_output["raw_pdca_sum"] = round(sum(pdca_results.values()), 2)

        # 6. Threshold & Decision
        is_passed = normalized_score >= 1.2

        # 7. Missing Phases + Enhanced Coaching
        missing_phases = [p for p in required_phases if pdca_results[p] < 1.0]
        coaching = llm_output.get("coaching_insight", "").strip()
        if missing_phases:
            missing_str = ", ".join(missing_phases)
            coaching = f"⚠️ ขาดหลักฐานชัดเจนในส่วน: {missing_str}. {coaching}"
            if is_conflict:
                coaching += " (ผ่านด้วย rerank สูง แต่ควรเสริมหลักฐานในเฟสที่ขาด)"

        # 8. Final Packaging
        llm_output.update({
            "score": normalized_score,
            "pdca_breakdown": pdca_results,
            "is_passed": is_passed,
            "required_phases": required_phases,
            "coaching_insight": coaching,
            "status": "PASSED" if is_passed else "FAILED",
            "missing_phases": missing_phases
        })

        return llm_output

    def _is_previous_level_passed(self, sub_id: str, level: int) -> bool:
        """
        [STRICT REVISE v2026.01.18.1] - ระบบตรวจสอบสถานะ Level ก่อนหน้าแบบเข้มงวด
        - แก้ไขปัญหา Fallback ผิดพลาดทำให้ข้าม Level ไม่สมบูรณ์
        - เชื่อถือเฉพาะ assessment_results_map ที่ผ่านกระบวนการตัดสินแล้วเท่านั้น
        """
        if level <= 1: 
            return True
            
        prev_level = level - 1
        possible_keys = [f"{sub_id}.L{prev_level}", f"{sub_id}_L{prev_level}"]
        
        # 1. ตรวจสอบจากผลลัพธ์การประเมินโดยตรง (แหล่งข้อมูลที่เชื่อถือได้ที่สุด)
        for key in possible_keys:
            result = self.assessment_results_map.get(key)
            if result:
                # ต้องเช็คว่าเป็น bool และต้องเป็น True เท่านั้น
                if result.get('is_passed') is True:
                    self.logger.info(f"✅ [LEVEL-GATE] Level {prev_level} passed for {sub_id}")
                    return True
                else:
                    self.logger.warning(f"⚠️ [LEVEL-GATE] Level {prev_level} found but status is FAIL")
                    return False

        # 2. 🛡️ Safe Guard: ถ้าไม่มีข้อมูลใน Map เลย (เช่น โปรแกรมเพิ่งเริ่มหรือข้ามมา) 
        # ให้ถือว่า "ไม่ผ่าน" ไว้ก่อน เพื่อความปลอดภัยตามมาตรฐาน SE-AM
        self.logger.warning(f"🚫 [LEVEL-GATE] No assessment record for L{prev_level}. Blocking L{level}.")
        return False

    def _expand_context_with_neighbor_pages(self, top_evidences: List[Any], collection_name: str) -> List[Any]:
        """
        [REVISE v6] Optimized Context Expansion
        - เพิ่มระบบ Duplicate Prevention ก่อน Query DB
        - ปรับปรุง PDCA Tagging ตาม Offset (Before=Support, After=Detail)
        - จำกัด Max Expansion เพื่อป้องกัน Token Overload
        """
        if not self.vectorstore_manager or not top_evidences:
            return top_evidences

        expanded_evidences = list(top_evidences)
        seen_keys = set()
        added_pages = 0
        MAX_PAGES_PER_SUB = 10 # 🚩 ป้องกันบริบทบวมเกินไป
        
        strategic_triggers = ["วิสัยทัศน์", "นโยบาย", "ทิศทาง", "เป้าหมายหลัก", "ยุทธศาสตร์", "สารจาก", "คำนำ"]
        check_triggers = ["ความพึงพอใจ", "คะแนน", "สรุปผล", "ตัวชี้วัด", "ผลประเมิน", "kpi", "score", "สรุปการดำเนินงาน"]

        for doc in top_evidences:
            if added_pages >= MAX_PAGES_PER_SUB: break

            meta = doc.metadata if hasattr(doc, 'metadata') else doc.get('metadata', {})
            text = (doc.get('text') or doc.get('page_content') or "").lower()
            
            filename = meta.get("source") or meta.get("source_filename") or "Unknown File"
            doc_uuid = meta.get("stable_doc_uuid") or meta.get("doc_id")
            if not doc_uuid: continue

            try:
                current_page_str = str(meta.get("page_label", meta.get("page", "1")))
                current_page = int("".join(filter(str.isdigit, current_page_str)))
            except: continue

            # 🎯 Offset Strategy: ปรับตามพฤติกรรม Auditor
            offsets = []
            if any(k in text for k in strategic_triggers): offsets.extend([-1, 1, 2])
            if any(k in text for k in check_triggers): offsets.extend([-1, 1, 2, 3])
            
            for offset in sorted(list(set(offsets))):
                target_page = current_page + offset
                if target_page < 1 or target_page == current_page: continue
                
                cache_key = f"{doc_uuid}_{target_page}"
                if cache_key in seen_keys: continue
                seen_keys.add(cache_key) # 🚩 ล็อก Key ทันทีเพื่อลดภาระ DB

                neighbor_chunks = self.vectorstore_manager.get_chunks_by_page(
                    collection_name=collection_name,
                    stable_doc_uuid=doc_uuid,
                    page_label=str(target_page)
                )

                if neighbor_chunks:
                    self.logger.info(f"➕ Neighbor Fetch: Page {target_page} in {filename}")
                    
                    for nc in neighbor_chunks:
                        fixed_metadata = (nc.metadata.copy() if hasattr(nc, 'metadata') else {}).copy()
                        fixed_metadata.update({
                            "stable_doc_uuid": doc_uuid,
                            "page_label": str(target_page),
                            "source": filename,
                            "is_supplemental": True
                        })

                        # 🏷️ Smart Tagging: หน้าก่อนมักเป็น Context, หน้าหลังมักเป็น Detail
                        assigned_tag = "Support" if offset < 0 else "Detail"
                        if any(k in nc.page_content.lower() for k in check_triggers):
                            assigned_tag = "Act/Check"

                        expanded_evidences.append({
                            "text": f"[Supplemental Context - Page {target_page}]:\n{nc.page_content}",
                            "page_content": nc.page_content,
                            "metadata": fixed_metadata,
                            "pdca_tag": assigned_tag,
                            "is_supplemental": True,
                            "rerank_score": doc.get('rerank_score', 0.0) if isinstance(doc, dict) else 0.0
                        })
                    added_pages += 1

        return expanded_evidences
        

    def _resolve_evidence_filenames(self, evidence_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        [REVISED v2026.1.18] - ปรับปรุงการสืบค้นชื่อไฟล์ให้แม่นยำขึ้น
        - เพิ่มระบบตรวจสอบ Metadata โดยตรง (Direct Metadata Check)
        - ปรับปรุง Display Source ให้สวยงามและอ่านง่ายสำหรับรายงาน
        """
        resolved_entries = []
        for entry in evidence_entries:
            resolved_entry = deepcopy(entry)
            doc_id = resolved_entry.get("doc_id", "")
            # ดึงชื่อไฟล์จาก metadata มาเป็นตัวสำรอง (Fallback)
            meta = resolved_entry.get("metadata", {}) if isinstance(resolved_entry.get("metadata"), dict) else {}
            meta_filename = meta.get("source") or meta.get("source_filename")
            
            content_raw = resolved_entry.get('content') or resolved_entry.get('text', '')
            level_origin = resolved_entry.get('level', 'N/A')
            page_label = resolved_entry.get("page_label") or resolved_entry.get("page") or "N/A"

            # 1. AI Generated Reference (ป้องกัน AI มโนรหัสเอกสาร)
            if str(doc_id).startswith("UNKNOWN-") or not doc_id:
                if not content_raw:
                    continue # ข้ามข้อมูลขยะ
                resolved_entry["filename"] = "AI-GENERATED-REF"
                resolved_entry["display_source"] = f"Reference (หน้า {page_label})"
            
            # 2. เคสปกติ: ค้นหาใน Map หรือ Metadata
            elif doc_id in self.doc_id_to_filename_map:
                mapped_name = self.doc_id_to_filename_map[doc_id]
                resolved_entry["filename"] = mapped_name
                resolved_entry["display_source"] = f"{mapped_name} (หน้า {page_label})"
            
            elif meta_filename:
                # 🚩 [NEW] ถ้าใน Map ไม่มีแต่ใน Metadata มีชื่อไฟล์ ให้ใช้ตัวนั้นเลย
                resolved_entry["filename"] = meta_filename
                resolved_entry["display_source"] = f"{meta_filename} (หน้า {page_label})"

            # 3. Fallback: กรณีรหัสผ่านแต่หาชื่อไฟล์ไม่เจอจริงๆ
            else:
                short_id = str(doc_id)[:8]
                resolved_entry["filename"] = f"DOC-{short_id}"
                resolved_entry["display_source"] = f"รหัสเอกสาร {short_id} (หน้า {page_label})"

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

    
    def _clean_map_for_json(self, data: Union[Dict, List, Set, Any]) -> Union[Dict, List, Any]:
        """Recursively converts objects that cannot be serialized (like sets) into lists."""
        if isinstance(data, dict):
            return {k: self._clean_map_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._clean_map_for_json(v) for v in data]
        elif isinstance(data, set):
            return [self._clean_map_for_json(v) for v in data]
        return data

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
        import hashlib
        import os
        from datetime import datetime
        from copy import deepcopy

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
        sub_id: str = "unknown",  # เพิ่ม argument นี้ (optional)
        level: int = 1           # เพิ่ม level เพื่อใช้ใน fallback
    ) -> Dict[str, Any]:
        """
        [ULTIMATE AUDIT CONFIDENCE v2026.3.4 – Final Production Stable]
        - แก้ NameError โดยรับ sub_id เป็น argument (fallback "unknown")
        - PDCA Detection แข็งแรงสุด (tag + metadata + fallback text + keywords จาก rules)
        - Decision Matrix เข้มงวดขึ้น (MEDIUM ต้อง coverage ≥ 0.5 + independence ≥ 5)
        - Recency Bonus fallback จาก filename
        - Guard ครบทุกจุด + Log debug ชัดเจน
        """
        if not matched_chunks:
            return {
                "level": "NONE",
                "reason": "ไม่พบหลักฐานที่เกี่ยวข้องในระบบ",
                "source_count": 0,
                "coverage_ratio": 0.0,
                "traceability_score": 0.0,
                "recency_bonus": 0.0,
                "valid_chunks_count": 0,
                "pdca_found": []
            }

        # 0. Quality Gate
        valid_chunks = [doc for doc in matched_chunks if self.get_actual_score(doc) >= 0.40]
        valid_count = len(valid_chunks)

        if valid_count == 0:
            return {
                "level": "LOW",
                "reason": "หลักฐานทั้งหมดมีคะแนน relevance ต่ำกว่า 0.40",
                "source_count": 0,
                "coverage_ratio": 0.0,
                "traceability_score": 0.0,
                "recency_bonus": 0.0,
                "valid_chunks_count": 0,
                "pdca_found": []
            }

        # 1. Independence (unique sources - robust)
        unique_sources = set()
        for doc in valid_chunks:
            meta = getattr(doc, 'metadata', {}) if hasattr(doc, 'metadata') else doc.get('metadata', {})
            src_keys = ['source_filename', 'filename', 'file_name', 'source', 'file_path']
            src = next((meta.get(k) for k in src_keys if meta.get(k)), None)
            if src:
                unique_sources.add(os.path.basename(str(src).strip()))

        independence_score = len(unique_sources)

        # 2. PDCA Coverage (enhanced multi-layer detection)
        pdca_map = {"P": False, "D": False, "C": False, "A": False}
        
        # ดึง keywords จาก rules เพื่อ fallback (ใช้ sub_id ที่รับมา)
        cum_rules = self.get_cumulative_rules(sub_id, level) if hasattr(self, 'get_cumulative_rules') else {}
        do_kws = [k.lower() for k in cum_rules.get('do_keywords', [])]
        check_kws = [k.lower() for k in cum_rules.get('check_keywords', [])]
        plan_kws = [k.lower() for k in cum_rules.get('plan_keywords', [])]
        act_kws = [k.lower() for k in cum_rules.get('act_keywords', [])]

        for doc in valid_chunks:
            meta = getattr(doc, 'metadata', {}) if hasattr(doc, 'metadata') else doc.get('metadata', {})
            tag = (
                getattr(doc, 'pdca_tag', None) or
                meta.get('pdca_tag') or
                meta.get('tag') or
                ""  # fallback ว่าง
            )
            tag = str(tag).strip().upper()

            if tag in pdca_map:
                pdca_map[tag] = True
            else:
                # Fallback 1: Text keyword detection
                text = (doc.get('text') or doc.get('page_content') or '').lower()
                if any(k in text for k in plan_kws):
                    pdca_map["P"] = True
                if any(k in text for k in do_kws):
                    pdca_map["D"] = True
                if any(k in text for k in check_kws):
                    pdca_map["C"] = True
                if any(k in text for k in act_kws):
                    pdca_map["A"] = True

        found_tags = [k for k, v in pdca_map.items() if v]
        coverage_ratio = len(found_tags) / 4.0

        # Debug PDCA detection
        self.logger.info(f"[PDCA DETECTION DEBUG] {sub_id} L{level} | "
                         f"Detected tags: {found_tags} | Coverage: {coverage_ratio:.2f} | "
                         f"Total chunks checked: {valid_count} | "
                         f"Plan kws sample: {plan_kws[:5] if plan_kws else 'N/A'}...")

        # 3. Traceability
        traceable_count = 0
        for doc in valid_chunks:
            meta = getattr(doc, 'metadata', {}) if hasattr(doc, 'metadata') else doc.get('metadata', {})
            page_keys = ['page_label', 'page', 'page_number', 'page_idx']
            has_page = any(meta.get(k) is not None for k in page_keys)
            has_file = any(meta.get(k) for k in ['source_filename', 'filename', 'file_name', 'source'])
            if has_page and has_file:
                traceable_count += 1

        traceability_score = traceable_count / valid_count if valid_count > 0 else 0.0

        # 4. Recency Bonus (enhanced fallback from filename)
        recency_bonus = 0.0
        current_year = 2568
        recent_count = 0
        for doc in valid_chunks:
            meta = getattr(doc, 'metadata', {}) if hasattr(doc, 'metadata') else doc.get('metadata', {})
            year_str = meta.get('year') or meta.get('doc_year')
            if not year_str:
                # Fallback จาก filename (เช่น ปี 2567 ในชื่อไฟล์)
                filename = str(meta.get('source_filename') or meta.get('source') or "")
                year_match = re.search(r'(25[67]\d)', filename)
                if year_match:
                    year_str = year_match.group(1)
            if year_str and str(year_str).isdigit():
                doc_year = int(year_str)
                if doc_year >= current_year - 2:
                    recent_count += 1
        if valid_count > 0:
            recency_bonus = round(recent_count / valid_count, 3)

        # 5. Decision Matrix (เข้มงวดขึ้น + เกณฑ์ใหม่)
        if independence_score <= 1 or coverage_ratio <= 0.25:
            level = "LOW"
            reason = "ความเชื่อมั่นต่ำมาก: แหล่งข้อมูลน้อยเกินไป หรือขาดมิติ PDCA อย่างน้อย 3 ด้าน"
        elif independence_score <= 4 or coverage_ratio < 0.50:
            level = "LOW"
            reason = "ความเชื่อมั่นต่ำ: หลักฐานยังไม่หลากหลายหรือครอบคลุม PDCA ไม่ถึง 50%"
        elif independence_score <= 7 or coverage_ratio < 0.75:
            level = "MEDIUM"
            reason = "ความเชื่อมั่นปานกลาง: หลักฐานเริ่มมีน้ำหนัก แต่ยังไม่ครบถ้วนสมบูรณ์ (ต้องการ 2+ เฟส PDCA)"
        else:
            level = "HIGH"
            reason = "ความเชื่อมั่นสูง: หลักฐานครบวงจร PDCA จากแหล่งข้อมูลหลากหลาย"

        # Penalty System
        if traceability_score < 0.60:
            if level == "HIGH":
                level = "MEDIUM"
                reason += " (การอ้างอิงหน้าเอกสารยังไม่ครบถ้วน)"
            elif level == "MEDIUM":
                level = "LOW"
                reason += " (การอ้างอิงหน้าเอกสารต่ำมาก)"

        return {
            "level": level,
            "reason": reason,
            "source_count": independence_score,
            "coverage_ratio": round(coverage_ratio, 3),
            "traceability_score": round(traceability_score, 3),
            "recency_bonus": round(recency_bonus, 3),
            "valid_chunks_count": valid_count,
            "pdca_found": found_tags
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

    def _calculate_weighted_score(self, highest_full_level: int, weight: float) -> float:
        """
        [SAFE WEIGHTED SCORE v2026.1.18]
        """
        MAX_LEVEL = getattr(self, 'MAX_LEVEL', 5)  # ใช้ attribute ก่อน fallback
        try:
            from config.global_vars import MAX_LEVEL as GLOBAL_MAX
            MAX_LEVEL = GLOBAL_MAX
        except:
            self.logger.debug("Using fallback MAX_LEVEL=5")

        level = max(0, min(float(highest_full_level), float(MAX_LEVEL)))
        if MAX_LEVEL <= 0:
            return 0.0

        return round((level / MAX_LEVEL) * float(weight), 4)

    def _export_results(self, results_data: Any, sub_criteria_id: str, **kwargs) -> str:
        """
        [ULTIMATE EXPORTER v2026.EXPORT.3 - FIXED & FULL REPORT]
        - แก้ไข AttributeError: 'list' object has no attribute 'get'
        - รวมผลลัพธ์ PDCA จากทุกระดับ (L1-L5) เข้าสู่ Report เดียว
        - เพิ่มระบบตรวจสอบความถูกต้องของคะแนน (Sanity Check) ก่อนเขียนไฟล์
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            record_id = kwargs.get("record_id", getattr(self, "current_record_id", f"auto_{timestamp}"))
            tenant = getattr(self.config, 'tenant', 'unknown')
            year = getattr(self.config, 'year', 'unknown')
            enabler = getattr(self, 'enabler', 'unknown').upper()

            # 1. จัดระเบียบข้อมูล Input (รองรับทั้ง List และ Dict)
            if results_data is None:
                results_data = self.final_subcriteria_results if hasattr(self, 'final_subcriteria_results') else []
            
            if isinstance(results_data, dict):
                results_data = [results_data]
            
            if not results_data:
                self.logger.warning(f"⚠️ [EXPORT] No results found for {sub_criteria_id}. Generating empty report.")

            # 2. ประมวลผลคะแนนสรุป (Summary Calculation)
            valid_results = [r for r in results_data if isinstance(r, dict)]
            highest_lvl = 0
            total_weighted = 0.0

            for res in valid_results:
                # ดึงระดับสูงสุดที่ผ่านจริง (เช็ค is_passed)
                lvl = int(res.get('level', 0)) if res.get('is_passed') else 0
                if lvl > highest_lvl:
                    highest_lvl = lvl
                
                # คำนวณถ่วงน้ำหนัก (ถ้ามี)
                weight = float(res.get('weight', 4.0))
                res['weighted_score'] = (lvl / 5.0) * weight
                total_weighted += res['weighted_score']

            # 3. แก้ไขบั๊ก Evidence Mapping (FIXED: List handling)
            master_map = self._load_evidence_map() or {}
            processed_evidence = {}
            
            for k, v in master_map.items():
                if not v: continue
                
                # 🛡️ FIX: ถ้า v เป็น list ให้เอาตัวแรกมาโชว์ ถ้าเป็น dict ให้ใช้ได้เลย
                item = v[0] if isinstance(v, list) and len(v) > 0 else v
                
                if isinstance(item, dict):
                    processed_evidence[k] = {
                        "file": item.get("filename") or item.get("file_name") or "unknown",
                        "page": item.get("page", "-"),
                        "pdca": (item.get("pdca_tag") or item.get("phase") or "Other").upper(),
                        "score": item.get("rerank_score", item.get("score", 0))
                    }

            # 4. ประกอบร่าง Payload (Full Report Structure)
            payload = {
                "record_id": record_id,
                "assessment_info": {
                    "tenant": tenant,
                    "year": year,
                    "enabler": enabler,
                    "sub_id": sub_criteria_id,
                    "engine_version": "SEAM-PDCA-v2026.1.20",
                    "exported_at": datetime.now().isoformat()
                },
                "summary": {
                    "maturity_level": f"L{highest_lvl}",
                    "is_passed": highest_lvl >= 1,
                    "total_weighted_score": round(total_weighted, 2),
                    "evidence_count": len(processed_evidence)
                },
                "detailed_results": valid_results,
                "evidence_mapping": processed_evidence,
                "action_plan": getattr(self, 'last_action_plan', {}) # เก็บ Roadmap ไว้ในไฟล์เดียวเลย
            }

            # 5. บันทึกไฟล์ (Smart Path Selection)
            try:
                export_path = get_assessment_export_file_path(
                    tenant=tenant, year=year, enabler=enabler.lower(),
                    suffix=f"{record_id}_{sub_criteria_id}_{timestamp}", ext="json"
                )
            except:
                # Fallback กรณีฟังก์ชันสร้าง Path มีปัญหา
                base_dir = f"exports/{tenant}/{year}"
                os.makedirs(base_dir, exist_ok=True)
                export_path = f"{base_dir}/{enabler}_{sub_criteria_id}_{timestamp}.json"

            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            self.logger.info(f"✅ [FULL EXPORT SUCCESS] Path: {export_path}")
            return export_path

        except Exception as e:
            self.logger.critical(f"❌ [EXPORT CRASH] Error: {str(e)}", exc_info=True)
            return ""
        
    def _calculate_overall_stats(self, target_sub_id: str):
        """
        [REVISED v2026.PDCA.FINAL]
        - สรุปผล Maturity Level โดยเจาะ Path Nested '0' ให้ถูกต้อง
        - ป้องกัน L0 ด้วยการ Scan ผลการประเมินรายเลเวลซ้ำ (Rescue Scan)
        """
        from datetime import datetime
        results = self.final_subcriteria_results
        if not results: return

        passed_levels = []
        for r in results:
            # 🚩 [FIX]: เจาะ Path ข้อมูลเข้าหา Level 1-5 ที่เก็บใน Key '0'
            level_zero = r.get('level_details', {}).get('0', {})
            details_map = level_zero.get('level_details', {})
            
            # Rescue Scan: ตรวจหาเลเวลสูงสุดที่ผ่านต่อเนื่องจริงจากรายละเอียด
            lvl = 0
            for l_idx in range(1, 6):
                lv_data = details_map.get(str(l_idx))
                if lv_data and lv_data.get('is_passed') is True:
                    lvl = l_idx
                else:
                    break # กฎความต่อเนื่อง SE-AM
            
            # อัปเดตผลระดับและคะแนนถ่วงน้ำหนัก
            r['highest_full_level'] = lvl 
            weight = float(r.get('weight', 4.0))
            if lvl > 0:
                new_score = self._calculate_weighted_score(lvl, weight)
                r['weighted_score'] = new_score
                r['is_passed'] = True
            else:
                r['weighted_score'] = 0.0
                r['is_passed'] = False

            passed_levels.append(lvl)

        # สรุปสถิติภาพรวม
        avg_level = sum(passed_levels) / len(results) if results else 0
        total_score = sum(float(r.get('weighted_score', 0.0)) for r in results)
        total_weight = sum(float(r.get('weight', 0.0)) for r in results)

        self.total_stats = {
            "overall_avg_score": round(total_score / total_weight, 4) if total_weight > 0 else 0,
            "overall_level_label": f"L{int(avg_level)}",
            "total_weighted_score": round(total_score, 2),
            "analytics": {
                "passed_levels_map": passed_levels,
                "strategic_gaps": self._extract_strategic_gaps(results)
            },
            "assessed_at": datetime.now().isoformat(),
            "highest_pass_level": int(avg_level)
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
    
    def _get_pdca_blocks_from_evidences(
        self, 
        evidences: List[Dict[str, Any]], 
        baseline_evidences: Any, 
        level: int, 
        sub_id: str, 
        contextual_rules_map: Dict[str, Any], 
        record_id: str = None
    ) -> Dict[str, str]:
        """
        [ULTIMATE REVISE v2026.1.18 - Fixed PDCA Coverage Bias]
        - Heuristic tag แข็งแรงขึ้นมาก (ช่วยดึง D/C/A สำหรับ L1)
        - Fallback tag ตาม level (L1 → 50% P/D ถ้า Other)
        - ใส่ source + page ใน block ทุก chunk
        - จำกัด 5 chunks ต่อ tag (ป้องกัน context ล้น)
        """
        pdca_groups = defaultdict(list)
        seen_texts = set()

        total_chunks = len(evidences or [])
        self.logger.info(f"🏷️ [TAGGING START] Processing {total_chunks} chunks for {sub_id} L{level}")

        for idx, chunk in enumerate(evidences or [], start=1):
            txt = chunk.get("text", "").strip()
            if not txt or txt in seen_texts:
                continue
            seen_texts.add(txt)

            # --- Metadata Recovery (robust) ---
            meta = chunk.get("metadata", {})
            filename = (
                chunk.get("source_filename") or 
                meta.get("source_filename") or 
                meta.get("source") or 
                meta.get("file_name") or 
                "Unknown_File"
            )
            page = meta.get("page_label") or meta.get("page") or meta.get("page_number") or "N/A"
            display_name = f"{filename} (P.{page})"

            # --- 1. Enhanced Heuristic Tag (Priority) ---
            heuristic_tag = self._get_heuristic_pdca_tag(txt, level)
            final_tag = heuristic_tag

            if final_tag:
                self.logger.info(f"🏷️  [{idx}/{total_chunks}] {final_tag} | (Heuristic) | {display_name}")
            else:
                # --- 2. AI Tag (เฉพาะเมื่อ heuristic ไม่ชัด) ---
                try:
                    tag = self._get_semantic_tag(txt, sub_id, level, filename)
                    final_tag = tag if tag in {"P", "D", "C", "A"} else "Other"
                    self.logger.info(f"🏷️  [{idx}/{total_chunks}] {final_tag} | (AI) | {display_name}")
                except Exception as e:
                    self.logger.warning(f"⚠️ AI Tag failed {display_name}: {e}")
                    final_tag = "Other"

            # --- 3. Level-specific Fallback (แก้ bias L1 มีแต่ P) ---
            if final_tag == "Other":
                if level <= 2:
                    # สุ่ม P หรือ D 50/50 สำหรับ L1-L2
                    final_tag = "P" if idx % 2 == 0 else "D"
                elif level == 3:
                    final_tag = "C"  # L3 เน้น Check
                else:
                    final_tag = "A"  # L4+ เน้น Act

            chunk["pdca_tag"] = final_tag
            chunk["source_display"] = display_name

            pdca_groups[final_tag].append(chunk)

        # Summary Log
        tag_counts = {t: len(pdca_groups[t]) for t in ["P", "D", "C", "A", "Other"]}
        self.logger.info(
            f"📊 [TAGGING RESULT] {sub_id} L{level} -> "
            f"P:{tag_counts['P']} | D:{tag_counts['D']} | C:{tag_counts['C']} | A:{tag_counts['A']} | Other:{tag_counts['Other']}"
        )

        # 4. Build Blocks (จำกัด 5 chunks/tag)
        blocks = {}
        for tag in ["P", "D", "C", "A", "Other"]:
            chunks = pdca_groups[tag][:5]  # จำกัด 5
            if chunks:
                parts = []
                for c in chunks:
                    source = c.get("source_display", "Unknown")
                    parts.append(f"[{source}]\n{c.get('text', '').strip()[:400]}...")
                blocks[tag] = "\n\n".join(parts)
            else:
                blocks[tag] = "[ไม่พบหลักฐานชัดเจนในหมวดนี้]"

        # เพิ่ม sources สำหรับ traceability
        blocks["sources"] = {
            tag: [c.get("source_display") for c in pdca_groups[tag][:5]]
            for tag in ["Plan", "Do", "Check", "Act"]
        }

        return blocks

    def _generate_action_plan_safe(
        self, 
        sub_id: str, 
        name: str, 
        enabler: str, 
        results: List[Dict]
    ) -> Any:
        """
        [ULTIMATE STRATEGIC REVISE v2026.1.18 - Production Ready]
        - Strength Awareness: ดึง summary_thai เป็นจุดแข็ง
        - Missing Phases: คำนวณจาก pdca_breakdown จริง
        - Recommendation Type: แยก Remediation / Refinement / Excellence
        - Emergency Fallback: มี PDCA สั้น ๆ พร้อม coaching
        """
        try:
            self.logger.info(f"🚀 [ACTION PLAN] Generating for {sub_id} - {name}")

            to_recommend = []
            has_major_gap = False

            sorted_results = sorted(results, key=lambda x: x.get('level', 0))

            for r in sorted_results:
                level = r.get('level', 0)
                is_passed = r.get('is_passed', False)
                score = float(r.get('score', 0.0))
                pdca_raw = r.get('pdca_breakdown', {})

                # Missing phases (ต่ำกว่า 0.5)
                missing = [p for p in ['P', 'D', 'C', 'A'] if float(pdca_raw.get(p, 0.0)) < 0.5]

                coaching = r.get('coaching_insight', '').strip()
                strength = r.get('summary_thai', r.get('reason', '')).strip() if is_passed and score >= 0.8 else ""

                payload = {
                    "level": level,
                    "is_passed": is_passed,
                    "score": score,
                    "missing_phases": missing,
                    "coaching_insight": coaching,
                    "strength_context": strength,
                    "recommendation_type": "FAILED_REMEDIATION" if not is_passed else
                                          "QUALITY_REFINEMENT" if missing or score < 1.0 else
                                          "EXCELLENCE_MAINTENANCE"
                }

                if not is_passed or missing:
                    has_major_gap = True
                to_recommend.append(payload)

            # กรณีผ่านหมด → Excellence
            if not has_major_gap:
                self.logger.info(f"🌟 {sub_id} EXCELLENT - No major gaps")
                return {
                    "status": "EXCELLENT",
                    "message": "บรรลุเกณฑ์ทั้งหมดอย่างสมบูรณ์",
                    "coaching_summary": "รักษามาตรฐานและพัฒนาเป็น Best Practice ต่อไป"
                }

            # ส่งไปสร้าง roadmap
            action_plan_args = {
                "recommendation_statements": to_recommend,
                "sub_id": sub_id,
                "sub_criteria_name": name,
                "enabler": enabler,
                "target_level": getattr(self.config, 'target_level', 5),
                "llm_executor": self.llm,
                "logger": self.logger
            }

            self.logger.info(f"[ACTION PLAN] Invoking engine with {len(to_recommend)} items")
            return create_structured_action_plan(**action_plan_args)

        except Exception as e:
            self.logger.error(f"❌ Action Plan Failed: {str(e)}")
            return _get_emergency_fallback_plan(
                sub_id, name, 
                getattr(self.config, 'target_level', 5), 
                has_major_gap, False, enabler
            )

    def _create_text_block_from_chunks(self, tag, chunks):
        """ ตัวช่วยสร้างเนื้อหาจาก chunk """
        if not chunks: return ""
        parts = [f"### [{tag} Evidence]\n{c.get('text','')}" for c in chunks[:5]]
        return "\n\n".join(parts)
    
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
    

    # ----------------------------------------------------------------------
    # 🚀 CORE WORKER: Assessment Execution (REVISED v2026.1.18 - FULL RECOVERY)
    # ----------------------------------------------------------------------
    def _run_sub_criteria_assessment_worker(
        self,
        sub_criteria: Dict[str, Any],
        vectorstore_manager: Optional['VectorStoreManager'] = None
    ) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
        """
        [FIXED & FINAL PRODUCTION v2026.1.18]
        - แก้ไขปัญหา L0: ใช้ is_passed ที่ผ่านการ Rescue/Rerank มาคำนวณ Sequential Logic
        - ป้องกันสะพานขาด: หาก L1-L3 ผ่านด้วย Safety Net ระบบจะนับต่อเนื่องให้ตามเกณฑ์ SE-AM
        - จัดโครงสร้าง Nested '0' ให้สอดคล้องกับตัวคำนวณสถิติภาพรวม
        """
        MAX_RETRY_ATTEMPTS = 2
        sub_id = sub_criteria['sub_id']
        sub_criteria_name = sub_criteria['sub_criteria_name']
        sub_weight = sub_criteria.get('weight', 0)
        
        current_enabler = getattr(self.config, 'enabler', 'Unknown')
        vsm = vectorstore_manager or getattr(self, 'vectorstore_manager', None)
        
        # --- [TRACKING STATES] ---
        current_sequential_pass_level = 0 
        found_primary_gap = False  
        
        raw_results_for_sub_seq: List[Dict[str, Any]] = []
        level_details_map = {} 
        current_worker_evidence_map = {} 
        start_ts = time.time() 

        all_rules_for_sub = getattr(self, 'contextual_rules_map', {}).get(sub_id, {})
        levels_to_assess = sorted(sub_criteria.get('levels', []), key=lambda x: x.get('level', 0))

        # -----------------------------------------------------------
        # EVALUATION LOOP
        # -----------------------------------------------------------
        for statement_data in levels_to_assess:
            level = statement_data.get('level')
            if level is None or level > getattr(self.config, 'target_level', 5):
                continue
            
            level_result = {}
            for attempt_num in range(1, MAX_RETRY_ATTEMPTS + 1):
                try:
                    raw_res = self._run_single_assessment(
                        sub_criteria=sub_criteria,
                        statement_data=statement_data,
                        vectorstore_manager=vsm,
                        attempt=attempt_num,
                        record_id=self.current_record_id,
                        evidence_map=self.evidence_map,
                        **all_rules_for_sub.get(str(level), {})
                    )

                    if isinstance(raw_res, tuple):
                        level_result = raw_res[0] if len(raw_res) > 0 else {}
                    else:
                        level_result = raw_res if isinstance(raw_res, dict) else {}

                    if level_result and "is_passed" in level_result:
                        break
                except Exception as e:
                    self.logger.error(f"❌ [L{level}] Error: {str(e)}")
                    level_result = {"level": level, "is_passed": False, "score": 0.0}

            # --- [SEQUENTIAL GAP LOGIC - FIXED] ---
            # 🚩 สำคัญ: ใช้ is_passed ที่ผ่าน Post-process มาแล้ว (รวม Rerank Safety)
            is_passed_final = level_result.get('is_passed', False)

            if is_passed_final and not found_primary_gap:
                current_sequential_pass_level = level
                level_result.update({"display_status": "PASSED", "gap_type": "NONE", "is_passed": True})
            elif not is_passed_final and not found_primary_gap:
                found_primary_gap = True
                level_result.update({"display_status": "FAILED", "gap_type": "PRIMARY_GAP", "is_passed": False})
            elif is_passed_final and found_primary_gap:
                level_result.update({"display_status": "PASSED (CAPPED)", "gap_type": "SEQUENTIAL_GAP", "is_passed": False})
            else:
                level_result.update({"display_status": "FAILED (GAP)", "gap_type": "COMPOUND_GAP", "is_passed": False})

            # บันทึกข้อมูลรายเลเวล
            level_details_map[str(level)] = {
                "level": level,
                "is_passed": level_result.get('is_passed', False),
                "score": float(level_result.get('score', 0.0)),
                "pdca_breakdown": level_result.get('pdca_breakdown', {}),
                "reason": level_result.get('reason', ""),
                "display_status": level_result.get("display_status"),
                "gap_type": level_result.get("gap_type")
            }
            raw_results_for_sub_seq.append(level_result)

        # -----------------------------------------------------------
        # FINAL SYNTHESIS
        # -----------------------------------------------------------
        weighted_score = round(self._calculate_weighted_score(current_sequential_pass_level, sub_weight), 2)
        action_plan_result = self._generate_action_plan_safe(sub_id, sub_criteria_name, current_enabler, raw_results_for_sub_seq)

        return {
            "sub_id": sub_id,
            "sub_criteria_name": sub_criteria_name,
            "highest_full_level": current_sequential_pass_level,
            "weighted_score": weighted_score,
            "weight": sub_weight,
            "is_passed": current_sequential_pass_level > 0,
            "level_details": {
                "0": { # ห่อด้วย "0" เพื่อให้ตรงกับโครงสร้าง Aggregator
                    "sub_id": sub_id,
                    "highest_pass_level": current_sequential_pass_level,
                    "weighted_score": weighted_score,
                    "level_details": level_details_map,
                    "action_plan": action_plan_result
                }
            }
        }, current_worker_evidence_map
    
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
        [PRODUCTION FINAL v2026.1.17 — Ultimate Stability Fix]
        - แก้ไขปัญหาคะแนน L0 โดยการปรับปรุงการ Merge Results และ Evidence
        - รองรับโครงสร้างข้อมูลแบบ Nested จาก Worker ทั้ง Parallel และ Sequential
        - ปรับปรุงการ Deduplicate หลักฐานเพื่อให้ Total Items สะท้อนความเป็นจริง
        """
        start_ts = time.time()
        self.is_sequential = sequential
        self.current_record_id = record_id 

        # 1. 🎯 กรองเกณฑ์การประเมิน
        all_statements = self._flatten_rubric_to_statements()
        is_all = str(target_sub_id).lower() == "all"
        sub_criteria_list = all_statements if is_all else [
            s for s in all_statements if str(s.get('sub_id')).lower() == str(target_sub_id).lower()
        ]

        if not sub_criteria_list:
            return self._create_failed_result(record_id, f"Criteria '{target_sub_id}' not found", start_ts)

        self.logger.info(f"🎯 Assessment Start | Target: {target_sub_id} | Record ID: {record_id} | Sub-items: {len(sub_criteria_list)}")

        # 2. 🔄 Resumption: โหลด Evidence Map เดิมจาก Disk
        existing_data = self._load_evidence_map()
        if isinstance(existing_data, dict):
            # ตรวจสอบ Key ที่อาจเกิดขึ้นจากการ Save ต่างเวอร์ชัน
            self.evidence_map = existing_data.get("evidence_map", existing_data)
            self.logger.info(f"🔄 Resumed Evidence Map: {len(self.evidence_map)} keys from disk")
        else:
            self.evidence_map = {}
            self.logger.info("🆕 Starting with fresh Evidence Map")

        # 3. ⚙️ ตั้งค่า Execution Mode
        max_workers = int(os.environ.get('MAX_PARALLEL_WORKERS', 4))
        run_parallel = is_all and not sequential
        
        results_list = []
        execution_start = time.time()

        # 4. 🚀 Execution Phase
        if run_parallel:
            self.logger.info(f"🚀 Running Parallel Assessment (Workers: {max_workers})")
            worker_args = [self._prepare_worker_tuple(s, document_map) for s in sub_criteria_list]
            try:
                # ใช้ 'spawn' เพื่อความปลอดภัยของหน่วยความจำในระบบ RAG
                ctx = multiprocessing.get_context('spawn')
                with ctx.Pool(processes=max_workers) as pool:
                    results_list = pool.map(_static_worker_process, worker_args)
            except Exception as e:
                self.logger.critical(f"❌ Parallel execution failed: {e}")
                raise
        else:
            self.logger.info(f"🧵 Running Sequential Assessment: {target_sub_id}")
            vsm = vectorstore_manager or self._init_local_vsm()
            for sub_criteria in sub_criteria_list:
                res = self._run_sub_criteria_assessment_worker(sub_criteria, vsm)
                results_list.append(res)

        execution_time = time.time() - execution_start
        self.logger.info(f"[EXECUTION] Completed | Time: {execution_time:.2f}s | Results: {len(results_list)}")

        # 5. 🧩 Integration Phase (Merge Results + Evidence Mapping)
        integration_start = time.time()
        self.logger.info(f"🧩 Integrating {len(results_list)} results...")

        if not results_list:
            self.logger.warning("[INTEGRATION] No results to merge.")
        else:
            # --- 5.1 Merge Scores & Details ---
            for res in results_list:
                # รองรับทั้ง (data, map) จาก Parallel และ dict จาก Sequential
                worker_data = res[0] if isinstance(res, tuple) else res
                worker_map = res[1] if isinstance(res, tuple) else res.get('temp_map_for_level', {})
                
                # หาก temp_map_for_level ส่งมาเป็น List ตรงๆ ให้แปลงเป็น Dict Key
                if isinstance(worker_map, list):
                    key = f"{worker_data.get('sub_id')}_L{worker_data.get('level')}"
                    worker_map = {key: worker_map}
                
                self._merge_worker_results(worker_data, worker_map)

            # --- 5.2 Merge Evidence Mapping (Fix Total Items: 0) ---
            merged_evidence = self.merge_evidence_mappings(results_list)
            merge_stats = {"new_levels": 0, "added_items": 0, "updated_levels": 0}
            
            if merged_evidence:
                for key, items in merged_evidence.items():
                    if key not in self.evidence_map:
                        self.evidence_map[key] = items
                        merge_stats["new_levels"] += 1
                        merge_stats["added_items"] += len(items)
                    else:
                        # Deduplicate ID ป้องกันข้อมูลบวม
                        existing_ids = {
                            str(e.get('chunk_uuid') or e.get('doc_id') or "N/A").replace("-", "").lower() 
                            for e in self.evidence_map[key]
                        }
                        
                        new_to_add = []
                        for item in items:
                            clean_id = str(item.get('chunk_uuid') or item.get('doc_id') or "N/A").replace("-", "").lower()
                            if clean_id not in existing_ids and clean_id not in ["na", "n/a", ""]:
                                new_to_add.append(item)
                                existing_ids.add(clean_id)
                        
                        self.evidence_map[key].extend(new_to_add)
                        merge_stats["added_items"] += len(new_to_add)
                        merge_stats["updated_levels"] += 1
                
                self.logger.info(f"🧬 [MERGE-EVIDENCE] New: {merge_stats['new_levels']} | Updated: {merge_stats['updated_levels']} | Total Items: {sum(len(v) for v in self.evidence_map.values())}")
            else:
                self.logger.warning("[MERGE-EVIDENCE] No valid evidence found in results_list")

        integration_time = time.time() - integration_start

        # 6. 💾 Persistence
        persistence_start = time.time()
        try:
            save_payload = {
                "record_id": record_id,
                "evidence_map": self.evidence_map,
                "timestamp": datetime.now().isoformat()
            }
            self._save_evidence_map(save_payload)
            self.logger.info("✅ Final merged evidence map persisted to disk")
        except Exception as e:
            self.logger.error(f"❌ Save Failed: {e}")

        # 7. 📊 Final Summary
        self._calculate_overall_stats(target_sub_id)
        
        final_response = {
            "record_id": record_id,
            "summary": self.total_stats,
            "sub_criteria_results": self.final_subcriteria_results,
            "run_time_seconds": round(time.time() - start_ts, 2),
            "evidence_mapping_summary": {
                "total_levels": len(self.evidence_map),
                "total_items": sum(len(v) for v in self.evidence_map.values())
            }
        }

        if export:
            final_response["export_path"] = self._export_results(
                self.final_subcriteria_results, 
                target_sub_id, 
                record_id=record_id  # 👈 ใส่ชื่อ parameter ให้ชัดเจนเพื่อให้มันเข้าไปอยู่ใน **kwargs
            )

        return final_response
    
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
    
    def _merge_worker_results(self, sub_result: Dict[str, Any], temp_map: Dict[str, List[Dict]]):
        """
        [ULTIMATE REVISE v2026.PDCA.FINAL - THE CHAIN RESTORER]
        - FIXED: คะแนน L0 ทั้งที่ระดับสูงผ่าน (ด้วยระบบ Smart Chain Validation)
        - IMPROVED: รองรับการ Merge ข้อมูลจาก Parallel Workers ที่มาไม่พร้อมกัน
        - ADDED: ระบบตรวจสอบสถานะ 'is_passed' ที่ยืดหยุ่น (รองรับ Safety Net และ Manual Boost)
        """
        if not sub_result:
            return

        sub_id = str(sub_result.get('sub_id', 'Unknown'))
        # มั่นใจว่าระดับเป็น int
        try:
            level = int(sub_result.get('level', 0))
        except:
            level = 0
            
        # 1. 🛡️ Evidence Mapping Integration (Deduplicated)
        if temp_map and isinstance(temp_map, dict):
            for level_key, evidence_list in temp_map.items():
                if not isinstance(evidence_list, list): continue
                
                if level_key not in self.evidence_map:
                    self.evidence_map[level_key] = []
                
                # สร้าง set ของ ID ปัจจุบันเพื่อความเร็วในการเช็ค
                existing_ids = {
                    str(e.get('chunk_uuid') or e.get('doc_id') or id(e)) 
                    for e in self.evidence_map[level_key]
                }
                
                for ev in evidence_list:
                    ev_id = str(ev.get('chunk_uuid') or ev.get('doc_id') or id(ev))
                    if ev_id not in existing_ids and ev_id not in ["na", "n/a", ""]:
                        self.evidence_map[level_key].append(ev)
                        existing_ids.add(ev_id)

        # 2. 🔍 ค้นหาหรือสร้าง Container สำหรับ Sub-Criteria นี้
        # ตรวจสอบว่ามี List ผลลัพธ์หรือยัง
        if not hasattr(self, 'final_subcriteria_results'):
            self.final_subcriteria_results = []

        target = next((r for r in self.final_subcriteria_results if str(r.get('sub_id')) == sub_id), None)
        
        if not target:
            target = {
                "sub_id": sub_id,
                "sub_criteria_name": sub_result.get('sub_criteria_name') or sub_id,
                "weight": float(sub_result.get('weight', 4.0)),
                "level_details": {},  # เก็บผลลัพธ์แยกแต่ละ Level { "1": {...}, "2": {...} }
                "highest_full_level": 0,
                "weighted_score": 0.0,
                "is_passed": False,
                "audit_stop_reason": "Starting integration..."
            }
            self.final_subcriteria_results.append(target)

        # 3. 🧩 ผสานข้อมูลระดับล่าสุดเข้าไปใน level_details
        # ใช้ String Key เพื่อป้องกันปัญหาตอน JSON Export
        target['level_details'][str(level)] = sub_result
        
        # 4. ⚖️ คำนวณ Sequential Maturity (ระบบตรวจสอบความต่อเนื่อง)
        # เราต้องเช็คตั้งแต่ L1 ขึ้นไป ถ้า L ไหนพัง Chain จะหยุดทันที
        current_highest = 0
        stop_reason = ""
        
        # ตรวจสอบ L1 ถึง L5
        for l in range(1, 6):
            l_str = str(l)
            l_data = target['level_details'].get(l_str)
            
            if l_data:
                # เช็คผ่านทั้งจาก AI ปกติ หรือ Safety Net (is_passed)
                is_passed = l_data.get('is_passed', False)
                
                if is_passed:
                    current_highest = l
                else:
                    # ถ้าประเมินแล้วแต่ไม่ผ่าน
                    stop_reason = f"Stopped at L{l} because: {l_data.get('reason', 'ไม่ผ่านเกณฑ์')[:60]}..."
                    break
            else:
                # ถ้ายังไม่มีข้อมูลระดับนี้ (อาจจะยังประเมินไม่ถึงในโหมด Parallel หรือถูก Block)
                if level > l:
                    # ถ้าปัจจุบันเรากำลัง Merge L ที่สูงกว่า แต่ L ต่ำกว่าไม่มีข้อมูล 
                    # แสดงว่า L ต่ำกว่าอาจจะล้มเหลว (Critical Error) หรือถูกข้าม
                    stop_reason = f"Chain broken: L{l} data is missing"
                else:
                    stop_reason = f"Assessment pending for L{l} and above"
                break 
        
        # อัปเดตค่าสรุปสุดท้าย
        target['highest_full_level'] = current_highest
        target['audit_stop_reason'] = stop_reason
        target['is_passed'] = (current_highest >= 1)
        
        # 5. 💰 คำนวณ Weighted Score (Maturity / 5.0 * Weight)
        target['weighted_score'] = round((current_highest / 5.0) * target['weight'], 2)
        
        # 📊 Logging เพื่อการ Trace
        log_status = "CONTINUE" if current_highest == level else "CEASED/PENDING"
        self.logger.info(
            f"✅ [MERGE] {sub_id} L{level} -> Current Maturity: L{current_highest} | "
            f"Score: {target['weighted_score']} | Status: {log_status}"
        )
        if stop_reason and level >= current_highest:
             self.logger.debug(f"ℹ️ [REASON] {stop_reason}")

        return target
    
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
        [REVISED v2026.1.18 - JUDICIAL AUDITOR EDITION]
        - ยกระดับการอุทธรณ์ผลการประเมินด้วยหลัก 'Evidence-Based Reasoning'
        - เพิ่มความโปร่งใสของกระบวนการอุทธรณ์ (Explainable AI)
        - บังคับการค้นหา 'Golden Sentence' เพื่อใช้ยืนยันการ Override คะแนน
        """
        self.logger.info(f"⚖️ [EXPERT-APPEAL] Processing {sub_id} L{level} | Evidence Strength: {highest_rerank_score:.4f}")

        # 1. วิเคราะห์ช่องว่างจากรอบแรก (Gap Identification)
        missing_str = ", ".join(sorted(set(missing_tags))) if missing_tags else "พฤติกรรมตามเกณฑ์ PDCA"
        
        # 2. ปรับปรุง Hint Message ให้เป็นเชิงรุก (Proactive Instruction)
        # สั่งให้ AI เจาะจงหาหลักฐานที่ 'เกือบผ่าน' แต่ถูกมองข้าม
        hint_msg = f"""
        ### 🚨 EXPERT AUDIT INSTRUCTION: DO NOT MISS KEY SUBSTANCE 🚨
        
        [CONTEXT]: การประเมินรอบแรกให้ "ไม่ผ่าน" เนื่องจาก: "{first_attempt_reason[:150]}..."
        [OPPORTUNITY]: ระบบตรวจสอบพบข้อมูลที่มีความเกี่ยวข้องสูง ({highest_rerank_score:.4f}) 
        ซึ่งอาจมีเนื้อหาที่สะท้อนถึง: {missing_str}
        
        [TASK]: ในฐานะ 'ผู้ตรวจสอบอาวุโส' ให้คุณมองข้ามเรื่อง Keyword ที่ไม่ตรงเป๊ะ (Form) 
        และมุ่งเน้นที่ 'การกระทำจริง' (Substance) หากพบหลักฐานแม้เพียงจุดเดียวที่สะท้อนถึงเกณฑ์ 
        "{base_kwargs.get('specific_contextual_rule', 'เกณฑ์มาตรฐาน')}" ให้ตัดสินว่า "ผ่าน"
        
        [REQUIRED]: หากตัดสินให้ 'ผ่าน' คุณต้องระบุประโยคอ้างอิงจาก Context มาอย่างน้อย 1 ประโยค
        """

        expert_kwargs = base_kwargs.copy()
        expert_kwargs["context"] = f"{context}\n\n{hint_msg}"
        expert_kwargs["sub_criteria_name"] = f"{sub_criteria_name} (Expert Re-assessment)"
        
        # ตั้งค่า Confidence เป็นสูงสุดเพื่อบังคับให้ AI ใช้สมาธิในการวิเคราะห์เนื้อหา
        expert_kwargs["ai_confidence"] = "MAX" 

        try:
            # 3. รันการประเมินรอบที่สอง (The Appeal)
            re_eval_result = llm_evaluator_to_use(**expert_kwargs)
            
            re_eval_result["is_expert_evaluated"] = True
            re_eval_result["appeal_status"] = "PENDING"

            if re_eval_result.get("is_passed", False):
                # ✅ กรณีอุทธรณ์สำเร็จ (Override)
                self.logger.info(f"🛡️ [OVERRIDE-SUCCESS] {sub_id} L{level} | Appeal Granted by Expert Reasoning")
                re_eval_result["appeal_status"] = "GRANTED"
                
                # ตกแต่งเหตุผลให้ Auditor มนุษย์อ่านแล้วประทับใจ
                orig_reason = re_eval_result.get('reason', '')
                re_eval_result["reason"] = f"🌟 [EXPERT OVERRIDE]: {orig_reason}"
            else:
                # ❌ กรณีอุทธรณ์ไม่สำเร็จ (Confirm Failure)
                self.logger.warning(f"⚖️ [APPEAL-DENIED] {sub_id} L{level} | Evidence insufficient even under expert review.")
                re_eval_result["appeal_status"] = "DENIED"
                re_eval_result["reason"] = f"Confirmed Fail: {re_eval_result.get('reason', '')}"

            return re_eval_result

        except Exception as e:
            self.logger.error(f"🛑 [EXPERT-ERROR] {sub_id} Appeal Process Crashed: {str(e)}")
            return {
                "is_passed": False, 
                "score": 0.0, 
                "reason": f"System error during Expert Appeal: {str(e)}",
                "is_expert_evaluated": True,
                "appeal_status": "ERROR"
            }
        
    def _apply_diversity_filter(self, evidences: List[Dict], level: int) -> List[Dict]:
        """
        [REVISED v2026] กรองหลักฐานให้มีความหลากหลายของแหล่งข้อมูล
        - ใช้ get_actual_score เพื่อการจัดลำดับที่แม่นยำ
        - ปรับ Limit เพื่อให้ครอบคลุมมิติ PDCA (โดยเฉพาะมิติ D และ C)
        """
        if not evidences:
            return []

        # 1. จัดลำดับตามคะแนนจริงของระบบ
        sorted_evidences = sorted(
            evidences,
            key=lambda x: self.get_actual_score(x),
            reverse=True
        )

        # 2. สำหรับ Level ต่ำ (1-2) เน้นปริมาณเพื่อหาพื้นฐาน (P/D)
        if level <= 2:
            return sorted_evidences[:20] 

        # 3. สำหรับ Level สูง (3+) ใช้ระบบ Diversity Guard
        diverse_results = []
        file_counts = defaultdict(int)
        per_file_limit = 5  # จำกัดไม่เกิน 5 chunks ต่อ 1 ไฟล์
        max_total = 30      # รวมทั้งหมดไม่เกิน 30 chunks เพื่อไม่ให้ Prompt ยาวเกินไป

        for ev in sorted_evidences:
            # ดึงชื่อไฟล์มาเป็น Key ในการนับ
            meta = ev.get('metadata', {}) if isinstance(ev, dict) else getattr(ev, 'metadata', {})
            source = meta.get('source_filename') or meta.get('file_name') or ev.get('source') or 'Unknown'
            source_key = os.path.basename(str(source))
            
            file_counts[source_key] += 1

            # ถ้าไฟล์นี้ยังส่งหลักฐานไม่เกินโควตา ให้รับเพิ่ม
            if file_counts[source_key] <= per_file_limit:
                diverse_results.append(ev)

            # ถ้าได้จำนวนรวมตามเป้าแล้วให้หยุด
            if len(diverse_results) >= max_total:
                break

        return diverse_results

    def enhance_query_for_statement(
        self,
        statement_text: str,
        sub_id: str,
        statement_id: str,
        level: int,
        focus_hint: str = "",
    ) -> List[str]:
        """
        [REVISED STRATEGIC v2026.2.12 – Balanced & Required Phase Optimized]
        - Follow required_phase จาก contextual_rules อย่างเคร่งครัด (keywords + fallback + bias)
        - Priority: query_synonyms (json) > specific_contextual_rule (แยกคำ) > fallback PDCA
        - Keywords จาก _enabler_defaults + required_phase (สมดุลทุก phase)
        - General phase-focused query ตาม required_phase (ไม่ hard-code เฉพาะ 1.2)
        - Negative ปรับตาม level (L1–L3 ลบ Plan เยอะ, L4+ ลบน้อยลง)
        - log ชัดเจน + post-process เดิม
        """
        import random
        from typing import List

        logger = logging.getLogger(__name__)

        # 1. Anchors
        enabler_id = getattr(self.config, 'enabler', 'Unknown').upper()
        tenant_name = getattr(self.config, 'tenant', 'Unknown').upper()
        id_anchor = f"{enabler_id} {sub_id}"

        # ดึง required_phase (สำคัญที่สุด – ใช้กำหนดทุกอย่าง)
        require_phases = self.get_rule_content(sub_id, level, "require_phase") or []
        require_str = ", ".join(require_phases) if require_phases else "P,D"

        # 2. Keywords จาก _enabler_defaults + required_phase (สมดุลตามเกณฑ์จริง)
        raw_kws = []
        must_list = self.get_rule_content(sub_id, level, "must_include_keywords")
        if isinstance(must_list, list):
            raw_kws.extend(must_list)

        phase_keywords_map = {
            "P": "plan_keywords",
            "D": "do_keywords",
            "C": "check_keywords",
            "A": "act_keywords"
        }

        # ดึง keywords เฉพาะ phase ที่ required
        for phase in require_phases:
            kw_key = phase_keywords_map.get(phase)
            if kw_key:
                raw_kws.extend(self.get_rule_content(sub_id, level, kw_key) or [])

        # ถ้าไม่มี require_phase → fallback ทั่วไปตาม level
        if not require_phases:
            if level <= 3:
                raw_kws.extend(self.get_rule_content(sub_id, 1, "plan_keywords") or [])
                raw_kws.extend(self.get_rule_content(sub_id, 2, "do_keywords") or [])
            elif level == 4:
                raw_kws.extend(self.get_rule_content(sub_id, 2, "do_keywords") or [])
                raw_kws.extend(self.get_rule_content(sub_id, 3, "check_keywords") or [])
            else:
                raw_kws.extend(self.get_rule_content(sub_id, 2, "do_keywords") or [])
                raw_kws.extend(self.get_rule_content(sub_id, 3, "check_keywords") or [])
                raw_kws.extend(self.get_rule_content(sub_id, 4, "act_keywords") or [])

        clean_kws = sorted(set(str(k).strip() for k in raw_kws if k))
        keywords_str = " ".join(clean_kws[:5])
        short_keywords = " ".join(clean_kws[:3])

        clean_stmt = statement_text.split("เช่น", 1)[0].strip()
        clean_stmt = re.sub(r'[^\w\s]', '', clean_stmt)[:70]

        queries: List[str] = []

        # 3. Queries พื้นฐาน (ปรับตาม level)
        queries.append(f"{id_anchor} {clean_stmt} {keywords_str}")
        queries.append(f"{id_anchor} {clean_stmt}")

        if level <= 3:
            queries.append(f"{tenant_name} ประกาศ คำสั่ง ระเบียบ {id_anchor} {short_keywords}")
        else:
            queries.append(f"{tenant_name} รายงานผล KPI ภาคผนวก {id_anchor} {short_keywords}")

        # Negative ตาม level (L1–L3 ลบ Plan เยอะ, L4+ ลบน้อยลงเพื่อให้ C/A เข้ามา)
        if level <= 3:
            queries.append(f"{id_anchor} (ผู้บริหาร OR มุ่งมั่น OR ตัวอย่าง OR ขับเคลื่อน) -แผน -นโยบาย -ยุทธศาสตร์")
        else:
            queries.append(f"{id_anchor} (รายงานผล OR ประเมิน OR ติดตาม OR ปรับปรุง OR นวัตกรรม) -แผน -นโยบาย")

        # 4. Priority 1: query_synonyms จาก json
        query_syn = self.get_rule_content(sub_id, level, "query_synonyms") or ""
        if query_syn:
            queries.append(f"{id_anchor} {query_syn} {short_keywords}")
            logger.info(f"[QUERY_SYNONYMS] Used {len(query_syn.split())} words from json for {sub_id} L{level}: {query_syn[:80]}...")

        # 5. Priority 2: Rule-based synonyms จาก specific_contextual_rule
        specific_rule = ""  # default
        if not query_syn:
            specific_rule = self.get_rule_content(sub_id, level, "specific_contextual_rule") or ""
            if specific_rule:
                rule_words = [w.strip() for w in specific_rule.split() if len(w.strip()) >= 4]
                unique_rule_words = list(dict.fromkeys(rule_words))[:8]
                rule_synonyms = " ".join(unique_rule_words)
                if rule_synonyms:
                    queries.append(f"{id_anchor} {rule_synonyms} {short_keywords}")
                    logger.info(f"[RULE SYNONYMS] Used {len(unique_rule_words)} words from specific_contextual_rule for {sub_id} L{level}")

        # 6. Priority 3: Fallback PDCA synonyms (ปรับตาม required_phase)
        fallback_synonyms = {
            "P": "แผน เป้าหมาย นโยบาย เจตนารมณ์ มุ่งมั่น ตัวอย่าง",
            "D": "ปฏิบัติ ดำเนินการ ขับเคลื่อน กิจกรรม อบรม ประชุม",
            "C": "ประเมิน ติดตาม ตรวจสอบ รายงานผล KPI วัดผล สรุปผล",
            "A": "ปรับปรุง พัฒนา แก้ไข ต่อยอด นวัตกรรม ยกระดับ Best Practice"
        }

        for phase in require_phases:
            fallback = fallback_synonyms.get(phase, "")
            if fallback:
                queries.append(f"{id_anchor} {fallback} {short_keywords}")
                logger.info(f"[FALLBACK {phase}] Used for {sub_id} L{level}")

        # 7. General phase-focused สำหรับ L1–L3 (เฉพาะ KM + ตาม required_phase)
        if level <= 3 and enabler_id == "KM" and "D" in require_phases:
            queries.append(f"{id_anchor} (ประชุม OR อบรม OR กิจกรรม OR ตัวอย่าง OR มุ่งมั่น OR สนับสนุน OR ขับเคลื่อน) -แผน -นโยบาย")

        # 8. Advanced/Focus hint (L4+)
        if level >= 4 or focus_hint:
            adv = "นวัตกรรม Best Practice Lesson Learned"
            queries.append(f"{id_anchor} {adv} {focus_hint or ''}")

        # Post-process
        final_queries = []
        seen = set()
        for q in queries:
            words = q.split()
            trunc_len = random.randint(22, 28)
            q_trunc = " ".join(words[:trunc_len])
            q_norm = " ".join(words[:18])
            if q_trunc and q_norm not in seen:
                final_queries.append(q_trunc)
                seen.add(q_norm)

        random.shuffle(final_queries)

        logger.info(f"🚀 [Query Gen] {sub_id} L{level} | Generated {len(final_queries)} diverse queries "
                    f"(required phases: {require_str}) "
                    f"(used query_synonyms: {bool(query_syn)}) "
                    f"(used rule synonyms: {bool(specific_rule and not query_syn)})")
        return final_queries[:7]
    
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

    def _log_pdca_status(self, sub_id, name, level, blocks, req_phases, sources_count, score, conf_level, **kwargs):
        """
        [THE AUDITOR DASHBOARD v2026.1.18]
        - Sync ระหว่าง Tagging (RAG) และ Extraction (LLM)
        """
        try:
            display_name = (str(name)[:50] + "...") if len(str(name)) > 50 else str(name)
            tagging_result = kwargs.get('tagging_result') or {}
            is_safety_pass = kwargs.get('is_safety_pass', False)

            status_parts = []
            extract_parts = []
            
            # Mapping ให้ตรงกับ LLM Result Key
            mapping = [("Extraction_P", "P"), ("Extraction_D", "D"), ("Extraction_C", "C"), ("Extraction_A", "A")]

            for full_key, short in mapping:
                evidence_count = tagging_result.get(short, 0)
                content = str(blocks.get(full_key, "")).strip()
                
                # AI พบข้อมูลถ้ามีเนื้อหาและไม่ใช่สัญลักษณ์ว่าง
                ai_found = bool(content and content not in ["-", "N/A", "ไม่พบข้อมูล"])
                
                # ตัดสิน Icon: ผ่านถ้า AI เจอ OR RAG เจอ OR Force Pass
                if ai_found or evidence_count > 0 or (is_safety_pass and short in req_phases):
                    icon = "✅"
                else:
                    icon = "❌"
                
                status_parts.append(f"{short}:{icon}({evidence_count})")
                
                if ai_found:
                    snippet = content[:45].replace('\n', ' ') + "..."
                    extract_parts.append(f"[{short}: {snippet}]")

            safety_tag = " 🛡️[FORCE-PASS]" if is_safety_pass else ""
            
            # Log 1: สถานะ PDCA
            self.logger.info(
                f"📊 [PDCA-STATUS] {sub_id} L{level} | {display_name} | "
                f"Req:{','.join(req_phases)} | Res: {' '.join(status_parts)}{safety_tag} | "
                f"Docs:{sources_count} | Score:{score:.2f} | Conf:{conf_level}"
            )
            
            # Log 2: คำสกัดจากหลักฐาน (Extraction Trace)
            if extract_parts:
                self.logger.info(f"🔍 [EXTRACT-TRACE] {sub_id} L{level} | {' | '.join(extract_parts[:2])}")

        except Exception as e:
            self.logger.error(f"❌ Error in _log_pdca_status: {str(e)}")   

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

    def _perform_adaptive_retrieval(self, sub_id: str, level: int, stmt: str, vectorstore_manager: Any):
        """
        [ULTIMATE RETRIEVAL v2026.1.22 – FULL & STABLE]
        - Early Exit: New >= 5 + Max >= 0.88 + >= 2 queries
        - Fallback: Multi-stage (Phase-based & Minimal) เมื่อไม่พบข้อมูลใหม่
        - Smart Priority Floor: 0.70 สำหรับ Priority Chunks
        """
        import hashlib
        
        # 0. Configuration & Parameters
        MAX_TOTAL_CHUNKS = 45
        MIN_QUALITY_FOR_EXIT = 0.88
        MIN_NEW_FOR_EXIT = 5
        MIN_QUERIES_FOR_EXIT = 2
        FORCE_EXTRA_LOOP_THRESHOLD = 3
        
        # ดึงชื่อ Tenant ป้องกัน NameError 'tenant_name'
        current_tenant = getattr(self.config, 'tenant', 'องค์กร')
        
        # 1. Priority Docs (ดึงจากระบบ Mapping/History)
        mapped_ids, priority_docs = self._get_mapped_uuids_and_priority_chunks(
            sub_id, level, stmt, vectorstore_manager
        ) or (set(), [])

        candidates = []
        final_max_rerank = 0.0
        all_scores = []
        used_queries = 0
        forced_continue = False
        new_counts = []
        priority_uuids = {p.get('chunk_uuid') for p in priority_docs if p and p.get('chunk_uuid')}

        if priority_docs:
            p_scores = [self.get_actual_score(p) for p in priority_docs if p]
            if p_scores:
                final_max_rerank = max(p_scores)
                all_scores.extend(p_scores)
            self.logger.info(f"📌 [PRIORITY] {sub_id} L{level} | {len(priority_docs)} chunks | Max: {final_max_rerank:.4f}")
        else:
            self.logger.warning(f"[PRIORITY] No priority chunks found for {sub_id}")

        # 2. Adaptive Search Loop
        # สร้าง Queries ที่หลากหลายจาก Statement
        queries = self.enhance_query_for_statement(stmt, sub_id, f"{sub_id}.L{level}", level)
        queries = queries[:5]  # จำกัดสูงสุด 5 queries เพื่อประหยัดทรัพยากร

        for i, q in enumerate(queries):
            used_queries += 1
            res = self.rag_retriever(
                query=q, doc_type=self.doc_type, sub_id=sub_id, level=level,
                vectorstore_manager=vectorstore_manager, stable_doc_ids=mapped_ids
            ) or {"top_evidences": []}

            loop_docs = res.get("top_evidences", [])
            if not loop_docs:
                self.logger.debug(f"[LOOP {i+1}] No docs returned for query: {q[:50]}...")
                new_counts.append(0)
                continue

            loop_scores = [self.get_actual_score(d) for d in loop_docs if d]
            if not loop_scores:
                new_counts.append(0)
                continue

            current_max = max(loop_scores)
            final_max_rerank = max(final_max_rerank, current_max)
            all_scores.extend(loop_scores)

            # Deduplication: ตรองเฉพาะข้อมูลใหม่ที่ไม่ซ้ำกับ Priority
            new_docs = [d for d in loop_docs if d.get('chunk_uuid') not in priority_uuids]
            
            # อัปเดต UUIDs เพื่อป้องกันซ้ำใน Loop ถัดไป
            for d in new_docs:
                if d.get('chunk_uuid'):
                    priority_uuids.add(d.get('chunk_uuid'))

            candidates.extend(new_docs)
            new_counts.append(len(new_docs))

            log_msg = f"🔍 [LOOP {i+1}] Query: {q[:50]}... | New: {len(new_docs)} | Max: {current_max:.4f}"
            self.logger.info(log_msg)

            # Force continue Logic
            if i == 0 and len(new_docs) == 0:
                forced_continue = True
                self.logger.info("[FORCE-CONTINUE] Loop 1 New=0 → บังคับรัน loop 2 ต่อ")

            # Enhanced Early Exit Check
            total_current = len(priority_docs) + len(candidates)
            if (final_max_rerank >= MIN_QUALITY_FOR_EXIT and
                total_current >= 12 and
                len(new_docs) >= MIN_NEW_FOR_EXIT and
                used_queries >= MIN_QUERIES_FOR_EXIT and
                not forced_continue):
                self.logger.info(f"🎯 [SMART EXIT] Loop {i+1} บรรลุเป้าหมายคุณภาพและปริมาณ")
                break

            # Force extra loop ถ้าข้อมูลยังน้อยเกินไป
            if i == 1 and sum(new_counts[:2]) < FORCE_EXTRA_LOOP_THRESHOLD:
                forced_continue = True
                self.logger.info(f"[FORCE EXTRA] New docs ({sum(new_counts[:2])}) ต่ำกว่าเกณฑ์ → บังคับ loop 3")
            else:
                forced_continue = False

        # 3. Fallback Mechanism (กรณีหาข้อมูลใหม่ไม่ได้เลย)
        if used_queries >= 3 and sum(new_counts) == 0:
            self.logger.warning("[FALLBACK] ข้อมูลใหม่เป็น 0 ใน 3 ลูปหลัก → เริ่มแผนสำรอง")
            
            # Fallback 1: Phase-based (P, D, C, A)
            require_phases = self.get_rule_content(sub_id, level, "require_phase") or ["P", "D"]
            fallback_q = f"{sub_id} {' OR '.join(require_phases)} {current_tenant}"
            
            res = self.rag_retriever(
                query=fallback_q, doc_type=self.doc_type, sub_id=sub_id, level=level,
                vectorstore_manager=vectorstore_manager, stable_doc_ids=mapped_ids
            ) or {"top_evidences": []}
            
            fb_docs = res.get("top_evidences", [])
            fb_new_count = 0
            if fb_docs:
                fb_new = [d for d in fb_docs if d.get('chunk_uuid') not in priority_uuids]
                candidates.extend(fb_new)
                fb_new_count = len(fb_new)
                self.logger.info(f"[FALLBACK-PHASE] เพิ่มได้ {fb_new_count} chunks")

            # Fallback 2: Minimal query (กวาดกว้างที่สุด)
            if fb_new_count == 0:
                minimal_q = f"{sub_id} {current_tenant}"
                res = self.rag_retriever(
                    query=minimal_q, doc_type=self.doc_type, sub_id=sub_id, level=level,
                    vectorstore_manager=vectorstore_manager, stable_doc_ids=mapped_ids
                ) or {"top_evidences": []}
                min_docs = res.get("top_evidences", [])
                if min_docs:
                    min_new = [d for d in min_docs if d.get('chunk_uuid') not in priority_uuids]
                    candidates.extend(min_new)
                    self.logger.info(f"[FALLBACK-MINIMAL] เพิ่มได้ {len(min_new)} chunks")

        # 4. Deduplication & Final Hard Cap
        unique_docs = {}
        for doc in (priority_docs + candidates):
            if not doc: continue
            uid = doc.get('chunk_uuid')
            if not uid:
                content = doc.get('page_content') or doc.get('text') or str(doc)
                uid = hashlib.sha256(content.encode('utf-8')).hexdigest()
            
            if uid not in unique_docs:
                unique_docs[uid] = doc

        final_docs = list(unique_docs.values())

        # ตัดจำนวนถ้าเกิน Hard Cap (เรียงตามคะแนนสูงสุด)
        if len(final_docs) > MAX_TOTAL_CHUNKS:
            final_docs = sorted(final_docs, key=lambda x: self.get_actual_score(x), reverse=True)[:MAX_TOTAL_CHUNKS]
            self.logger.warning(f"[HARD CAP] ตัดจำนวน Chunks จาก {len(unique_docs)} เหลือ {MAX_TOTAL_CHUNKS}")

        self._normalize_evidence_metadata(final_docs)

        # 5. Safety Net Floor & Priority Tagging
        for p in priority_docs:
            if isinstance(p, dict):
                p['is_priority'] = True
                # บังคับพื้นฐานคะแนนที่ 0.70 เพื่อให้ผ่าน Filter ในขั้นตอนถัดไป
                p['rerank_score'] = max(self.get_actual_score(p), 0.70)

        # สรุปผลการ Retrieval เข้า Log
        self.logger.info(f"📊 [RETRIEVAL SUMMARY] {sub_id} L{level} | Final Chunks: {len(final_docs)} | Max Rerank: {final_max_rerank:.4f} | Loops: {used_queries}")

        return final_docs, final_max_rerank
    
    # ------------------------------------------------------------------------------------------
    # [CRITICAL: SYSTEM CORE ARCHITECTURE v2026.1.19 - FULL TRACEABILITY]
    # 🚩 คำเตือน: ห้ามตัดต่อหรือลดทอนกระบวนการประเมิน 10 ขั้นตอนนี้เด็ดขาด เพื่อรักษามาตรฐาน Audit Traceability
    # 1. Dependency Gate: (_is_previous_level_passed) ตรวจสอบเงื่อนไข Level Skip
    # 2. Baseline Hydration: (_collect_previous_level_evidences) ดึงมรดกหลักฐานจากระดับก่อนหน้า
    # 3. Adaptive Retrieval: (_perform_adaptive_retrieval) Multi-loop RAG ค้นหาข้อมูลเชิงลึก
    # 4. Quality Gate & Diversity: (_apply_diversity_filter) กรองความซ้ำซ้อนของเนื้อหา
    # 5. Neighbor Expansion: (_expand_context_with_neighbor_pages) เติมเต็มบริบทจากหน้าข้างเคียง
    # 6. Multichannel Context: (_build_multichannel_context_for_level) แยกหมวดหมู่ P,D,C,A & Baseline
    # 7. PDCA Blocks Construction: (_get_pdca_blocks_from_evidences & _build_pdca_context)
    # 8. Dual-Round Evaluation: (evaluate_with_llm -> _build_audit_result_object) ประเมิน & ประกอบร่าง
    # 9. Expert Safety Net: (High Rerank Boost) ตรวจสอบความขัดแย้งระหว่าง Score และ Evidence
    # 10. Persistence & Traceability: (_save_level_evidences & _log_pdca_status) บันทึกและแสดง Log
    # ------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------
    # [CRITICAL: SYSTEM CORE ARCHITECTURE v2026.1.19 - FULL TRACEABILITY]
    # 🚩 คำเตือน: ห้ามตัดต่อหรือลดทอนกระบวนการประเมิน 10 ขั้นตอนนี้เด็ดขาด เพื่อรักษามาตรฐาน Audit Traceability
    # ------------------------------------------------------------------------------------------

    def _run_single_assessment(
        self,
        sub_criteria: Dict[str, Any],
        statement_data: Dict[str, Any],
        vectorstore_manager: Optional[Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        [ULTIMATE MASTER REVISE v2026.1.20]
        - FIXED: Baseline slice KeyError by flatten in build_multichannel...
        - IMPROVED: Prioritize top-ranked chunks for LLM context (fix AI blindness)
        - ADDED: Debug log for sent context preview
        """
        import time
        import json

        start_time = time.time()
        sub_id = str(sub_criteria.get('sub_id', 'Unknown'))
        level = int(statement_data.get('level', 1))
        name = str(sub_criteria.get('name', sub_criteria.get('sub_criteria_name', 'No Title')))
        stmt = str(statement_data.get('statement', f"เกณฑ์ระดับ {level}"))
        
        diverse_docs = []
        res = {"is_passed": False, "score": 0.0, "reason": "เริ่มการประเมิน"}
        raw_res = {} 

        try:
            # --- STEP 1-2: GATE & BASELINE HYDRATION ---
            if not self._is_previous_level_passed(sub_id, level):
                return {"sub_id": sub_id, "level": level, "score": 0.0, "is_passed": False, "status": "SKIPPED"}

            previous_evidences = self._collect_previous_level_evidences(sub_id, level)

            # --- STEP 3-5: ADAPTIVE RETRIEVAL & EXPANSION ---
            all_evidences, raw_max_score = self._perform_adaptive_retrieval(sub_id, level, stmt, vectorstore_manager) or ([], 0.0)
            diverse_docs = self._apply_diversity_filter(all_evidences, level) or []
            
            if hasattr(self, '_expand_context_with_neighbor_pages') and vectorstore_manager and raw_max_score > 0.35:
                diverse_docs = self._expand_context_with_neighbor_pages(diverse_docs, f"evidence_{self.enabler.lower()}")

            # --- STEP 6: MULTICHANNEL CONTEXT BUILDING ---
            multichannel_data = self._build_multichannel_context_for_level( # เพิ่ม self. และ _
                level=level,
                top_evidences=diverse_docs,
                previous_levels_evidence=previous_evidences
            )
            
            baseline_summary = str(multichannel_data.get("baseline_summary", ""))
            current_tagging = multichannel_data.get("debug_meta", {}).get("tagging_result")
            
            if not current_tagging:
                current_tagging = {p: len([d for d in diverse_docs if str(d.get('pdca_tag') or d.get('phase', '')).upper() == p]) for p in ["P", "D", "C", "A"]}

            # --- STEP 7: PDCA BLOCKS & RULE PREPARATION ---
            rules_map = getattr(self, 'contextual_rules_map', {})
            current_rules = rules_map.get(sub_id, {}).get(str(level), {})
            defaults = rules_map.get("_enabler_defaults", {})

            req_phases = self.get_rule_content(sub_id, level, "require_phase") or (['P', 'D'] if level <= 2 else ['P', 'D', 'C'])
            
            raw_plan_kws = current_rules.get("plan_keywords") or defaults.get("plan_keywords", ["นโยบาย", "แผนงาน"])
            plan_kws_str = ", ".join(raw_plan_kws) if isinstance(raw_plan_kws, list) else str(raw_plan_kws)

            blocks = self._get_pdca_blocks_from_evidences(diverse_docs, previous_evidences, level, sub_id, rules_map)
            full_context_text = self._build_pdca_context(blocks) or "ไม่พบข้อมูลหลักฐานชัดเจน"

            # === IMPROVEMENT: Prioritize top-ranked chunks for LLM context ===
            # เรียง diverse_docs ตาม rerank_score จากสูงไปต่ำ
            if diverse_docs:
                sorted_diverse_docs = sorted(
                    diverse_docs,
                    key=lambda d: float(d.get('rerank_score', 0) or d.get('score', 0)),
                    reverse=True
                )
            else:
                sorted_diverse_docs = []

            # เลือก top chunks (ปรับจำนวนได้ตาม level)
            max_chunks = 40 if level <= 2 else 25
            top_chunks = sorted_diverse_docs[:max_chunks]

            # สร้าง context จาก top chunks (แทน full_context_text เดิม)
            context_parts = []
            for idx, chunk in enumerate(top_chunks, 1):
                score = chunk.get('rerank_score', chunk.get('score', 'N/A'))
                source = chunk.get('source', chunk.get('file_name', 'ไม่ระบุ'))
                page = chunk.get('page', '-')
                text = chunk.get('text', '').strip()
                if text:
                    context_parts.append(
                        f"[หลักฐานอันดับ {idx} | Score: {score} | {source} หน้า {page}]\n"
                        f"{text}\n"
                        f"{'-'*80}\n"
                    )

            prioritized_context = "".join(context_parts) or "ไม่พบข้อมูลหลักฐานที่เกี่ยวข้อง"

            if baseline_summary:
                prioritized_context += f"\n\n=== ข้อมูลจากระดับก่อนหน้า (Baseline) ===\n{baseline_summary}\n"

            # ตัดความยาวถ้ายาวเกิน (แต่ไม่ตัดกลาง chunk)
            max_context_len = 15000 if level <= 2 else 10000
            if len(prioritized_context) > max_context_len:
                prioritized_context = prioritized_context[:max_context_len].rsplit('\n', 1)[0] + "\n... (ตัดเพื่อความเหมาะสม)"

            # Debug: Preview context ที่จะส่งจริง
            self.logger.info(f"[CONTEXT PRIORITY L{level}] Length: {len(prioritized_context)} | First 300 chars: {prioritized_context[:300]}...")

            # --- STEP 8: EVALUATION ---
            eval_fn = evaluate_with_llm_low_level if level <= 2 else evaluate_with_llm
            audit_conf = self.calculate_audit_confidence(diverse_docs, sub_id=sub_id, level=level)

            llm_params = {
                "context": prioritized_context,  # ← ใช้ตัวใหม่นี้แทน full_context_text
                "sub_criteria_name": name,
                "level": level,
                "statement_text": stmt,
                "sub_id": sub_id,
                "llm_executor": self.llm,
                "required_phases": req_phases,
                "baseline_summary": baseline_summary,
                "plan_keywords": plan_kws_str,
                "ai_confidence": str(audit_conf.get('level', "MEDIUM")),
                "specific_contextual_rule": str(current_rules.get("rule_text", current_rules.get("specific_contextual_rule", "พิจารณาตามเกณฑ์มาตรฐาน")))
            }

            try:
                res = eval_fn(**llm_params)
                if res is None:
                    raise ValueError("LLM Evaluation returned None")
                
                raw_res = res.copy() 
                # self.logger.critical(f"LLM RAW RESPONSE (DEBUG): {json.dumps(raw_res, ensure_ascii=False)}")
                
            except Exception as eval_err:
                self.logger.error(f"🛑 Evaluation Error at {sub_id} L{level}: {str(eval_err)}")
                res = {
                    "is_passed": False, "score": 0.0, 
                    "reason": f"System Evaluation Error: {str(eval_err)}",
                    "summary_thai": "ไม่สามารถประเมินได้เนื่องจากระบบขัดข้อง",
                    "P_Plan_Score": 0.0, "D_Do_Score": 0.0
                }

            # --- STEP 9: EXPERT SAFETY NET ---
            is_force_pass = False
            if not res.get("is_passed", False) and raw_max_score >= 0.85:
                self.logger.info(f"🛡️ [SAFETY-NET] {sub_id} L{level} | High Rerank {raw_max_score:.2f} -> Boosted.")
                res.update({
                    "is_passed": True, 
                    "score": max(res.get("score", 0.0), 1.2),
                    "reason": str(res.get("reason", "")) + " (Safety Net Pass: พบหลักฐานความเกี่ยวข้องสูงมาก)"
                })
                is_force_pass = True

            # --- STEP 10: PERSISTENCE & LOGGING ---
            final_strength = self._save_level_evidences_and_calculate_strength(diverse_docs, sub_id, level, res, raw_max_score)
            evidence_sources = self._resolve_evidence_filenames(diverse_docs)

            self._log_pdca_status(
                sub_id=sub_id, name=name, level=level, 
                blocks=raw_res, 
                req_phases=req_phases, 
                sources_count=len(evidence_sources),
                score=float(res.get('score', 0.0)), 
                conf_level=str(audit_conf.get('level', 'N/A')),
                tagging_result=current_tagging,
                is_safety_pass=is_force_pass
            )

            return {
                "sub_id": sub_id, "level": level, 
                "score": float(res.get('score', 0.0)),
                "is_passed": bool(res.get('is_passed', False)), 
                "reason": str(res.get('reason', "")),
                "summary_thai": str(res.get('summary_thai', "")), 
                "coaching_insight": str(res.get('coaching_insight', "ไม่มีคำแนะนำเพิ่มเติม")),
                "P_Plan_Score": float(res.get("P_Plan_Score", 0.0)), 
                "D_Do_Score": float(res.get("D_Do_Score", 0.0)),
                "evidence_sources": evidence_sources,
                "evidence_strength": final_strength, 
                "duration": round(time.time() - start_time, 2)
            }

        except Exception as e:
            self.logger.critical(f"🛑 CRITICAL FAILURE {sub_id} L{level}: {str(e)}", exc_info=True)
            return {
                "sub_id": sub_id, "level": level, "score": 0.0, "is_passed": False, 
                "reason": f"Critical Error: {str(e)}", "summary_thai": "เกิดข้อผิดพลาดร้ายแรงในระบบ"
            }