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
import copy
from database import init_db
from database import db_update_task_status as update_db

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
        BASE_PDCA_KEYWORDS,
        MAX_PARALLEL_WORKERS,
        PDCA_PRIORITY_ORDER,
        TARGET_DEVICE,
        PDCA_PHASE_MAP,
        INITIAL_TOP_K,
        FINAL_K_RERANKED,
        MAX_CHUNKS_PER_FILE,
        MAX_CHUNKS_PER_BLOCK
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
        build_multichannel_context_for_level
    )
    from core.vectorstore import VectorStoreManager, load_all_vectorstores, get_global_reranker 
    from core.action_plan_schema import ActionPlanActions

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
    
    PDCA_PHASE_MAP = {1: "Plan", 2: "Do", 3: "Check", 4: "Act", 5: "Sustainability"}

    class seam_mocking:
        @staticmethod
        def set_mock_control_mode(mode): pass

    if "FATAL ERROR" in str(e):
        pass 
# ----------------------------------------------------------------------

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

def classify_by_keyword(
    text: str, 
    sub_id: str = None, 
    level: int = None,
    contextual_rules_map: dict = None,
    chunk_metadata: dict = None
) -> str:
    """
    [ULTIMATE PDCA CLASSIFIER v2026.4 - FULL REVISED FOR NEW BRANCH]
    --------------------------------------------------
    - Metadata & Filename Awareness: ค้นหาข้อมูลจากชื่อไฟล์เพื่อความแม่นยำสูงสุด
    - JSON v2 Compatibility: รองรับโครงสร้าง 'require_phase' และ 'specific_contextual_rule'
    - L1/L2 High-Pass: ยอมรับมติบอร์ด (D) ในระดับนโยบายโดยไม่ถูกบล็อกด้วย Must-include
    """
    if not text: return 'Other'

    # --- 0. METADATA OVERRIDE (ลำดับความสำคัญสูงสุด) ---
    if chunk_metadata:
        meta_tag = chunk_metadata.get("pdca_tag") or chunk_metadata.get("PDCA")
        if meta_tag and str(meta_tag).upper() in {"P", "D", "C", "A"}:
            return str(meta_tag).upper()

    if not contextual_rules_map: return 'Other'
    text_lower = text.lower()

    def keyword_match(text_to_search: str, keywords_input, anchor: str = None) -> bool:
        """
        หัวใจสำคัญ: ค้นหา Keywords ทั้งใน 'เนื้อหา' และ 'ชื่อไฟล์'
        """
        filename = ""
        if chunk_metadata and isinstance(chunk_metadata, dict):
            # ดึงจาก 'source' หรือ 'filename' (ChromaDB มักเก็บใน source)
            filename = chunk_metadata.get("source", "") or chunk_metadata.get("filename", "")
            filename = os.path.basename(filename)

        # รวมข้อความที่จะใช้ค้นหา (ช่วยให้ KM1.1L106.pdf กลายเป็นหลักฐานชั้นดี)
        search_scope = (text_to_search + " " + filename).lower()

        # 1. ANCHOR CHECK (ระดับความแม่นยำรายหัวข้อ)
        if anchor:
            # ใช้ regex เพื่อหาเลขข้อ (เช่น 1.1) โดยไม่ให้ติดเลขอื่น (เช่น 1.11)
            anchor_pattern = rf'(?<!\d){re.escape(anchor)}(?!\d)'
            if not re.search(anchor_pattern, search_scope):
                return False

        # 2. KEYWORD NORMALIZATION
        kws = keywords_input if isinstance(keywords_input, list) else \
              [k.strip() for k in str(keywords_input).split(",") if k.strip()]
        
        # 3. REGEX MATCHING
        for kw in kws:
            kw_clean = str(kw).strip().lower()
            if not kw_clean: continue
            
            is_thai = any("\u0e00" <= c <= "\u0e7f" for c in kw_clean)
            if is_thai:
                pattern = re.escape(kw_clean)
            else:
                pattern = r'\b{}\b'.format(re.escape(kw_clean))
                
            if re.search(pattern, search_scope, re.IGNORECASE):
                return True
                
        return False

    def check_pdca_sequence(rules_dict: dict, anchor: str = None) -> Optional[str]:
        mapping = {
            "plan_keywords": "P", 
            "do_keywords": "D", 
            "check_keywords": "C", 
            "act_keywords": "A"
        }
        for json_key, tag in mapping.items():
            if keyword_match(text_lower, rules_dict.get(json_key, []), anchor=anchor):
                return tag
        return None

    # --- EXECUTION STEPS ---
    
    # Step 1: Specific Level Rules (จาก JSON ใหม่ เช่น 1.1 -> L1)
    rules = contextual_rules_map.get(sub_id, {})
    current_l_rules = rules.get(f"L{level}", {})
    
    if isinstance(current_l_rules, dict) and current_l_rules:
        # ดึง Phase ที่ JSON ระบุว่าจำเป็น (เช่น ["P", "D"])
        required_phases = current_l_rules.get("require_phase", [])
        
        # ค้นหา Tag เบื้องต้น
        tag = check_pdca_sequence(current_l_rules, anchor=None)
        
        if tag:
            # [LOGIC OVERRIDE] สำหรับ Level 1-2 ถ้าเจอ 'D' (มติ/อนุมัติ) 
            # ให้ผ่านทันทีเพื่อรองรับ "นโยบายที่ถูกนำไปปฏิบัติจริงในขั้นต้น"
            if tag == "D" and level <= 2:
                return "D"

            # Must-include / Avoid Guard สำหรับกรณีทั่วไป
            if rules.get("must_include_keywords"):
                if not keyword_match(text_lower, rules["must_include_keywords"]):
                    return 'Other'
            
            if rules.get("avoid_keywords"):
                if keyword_match(text_lower, rules["avoid_keywords"]):
                    return 'Other'
                    
            return tag

    # Step 2: JSON Sub-ID Fallback (ค้นหา Keywords รวมของหัวข้อนั้นๆ)
    # ใช้ Anchor=sub_id เพื่อกรองเนื้อหาที่ตรงข้อจริง
    tag = check_pdca_sequence(rules, anchor=sub_id)
    if tag: return tag

    # Step 3: Default Enabler Rules (เช่น KM กลาง)
    defaults = contextual_rules_map.get("_enabler_defaults", {})
    tag = check_pdca_sequence(defaults, anchor=sub_id)
    if tag: return tag

    # Step 4: Global System Fallback (ถ้ายังไม่เจออะไรเลย)
    try:
        from config.global_vars import PDCA_PRIORITY_ORDER, BASE_PDCA_KEYWORDS
        tag_map = {"Plan": "P", "Do": "D", "Check": "C", "Act": "A"}
        for full_tag in PDCA_PRIORITY_ORDER:
            if keyword_match(text_lower, BASE_PDCA_KEYWORDS.get(full_tag, []), anchor=sub_id):
                return tag_map.get(full_tag, 'Other')
    except ImportError:
        pass

    return 'Other'

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

def merge_evidence_mappings(results_list: List[Union[Tuple, Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    รวม evidence_mapping ที่ได้จาก Worker (ซึ่งคืนค่าเป็น Tuple) เข้าด้วยกัน
    """
    merged_mapping = defaultdict(list)
    
    for item in results_list:
        # 1. กรณี Worker คืนค่าเป็น Tuple (Standard Engine Return)
        # item[0] คือ sub_result, item[1] คือ temp_map (evidence)
        if isinstance(item, tuple) and len(item) == 2:
            worker_evidence_map = item[1]
            if isinstance(worker_evidence_map, dict):
                for level_key, evidence_list in worker_evidence_map.items():
                    if isinstance(evidence_list, list):
                        merged_mapping[level_key].extend(evidence_list)
        
        # 2. กรณี Worker คืนค่าเป็น Dict (Fallback หรือ Error case)
        elif isinstance(item, dict) and 'evidence_mapping' in item:
            worker_evidence_map = item['evidence_mapping']
            for level_key, evidence_list in worker_evidence_map.items():
                merged_mapping[level_key].extend(evidence_list)
                
    return dict(merged_mapping)

def get_pdca_keywords_str(phase: str) -> str:
    """
    ดึง Keywords จาก Global Vars และทำความสะอาด Regex 
    เพื่อให้ LLM นำไปใช้เป็นตัวอย่างในการ Extraction
    """
    # ดึง list ตาม phase (Plan, Do, Check, Act)
    raw_keywords = BASE_PDCA_KEYWORDS.get(phase, [])
    
    # ล้างอักขระพิเศษของ Regex ออก (เช่น r"", \, ^, $)
    clean_keywords = []
    for kw in raw_keywords:
        # ลบ escape characters และสัญลักษณ์ regex พื้นฐาน
        k = re.sub(r'[\\^$r"\']', '', kw)
        if k not in clean_keywords:
            clean_keywords.append(k)
            
    # ส่งคืนเป็น string ขั้นด้วยจุลภาค (เอาแค่ 10 คำแรกเพื่อประหยัด Token)
    return ", ".join(clean_keywords[:10])

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
        self.base_pdca_keywords = BASE_PDCA_KEYWORDS
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

    def enhance_query_for_statement(
        self,
        statement_text: str,
        sub_id: str,
        statement_id: str,
        level: int,
        focus_hint: str,
    ) -> List[str]:
        """
        [Revised 2026 - STABLE ANCHOR VERSION]
        เน้นการใช้ Sub-ID เป็นตัวสมอ (Anchor) เพื่อป้องกันการดึงข้อมูลข้ามข้อ
        """
        logger = logging.getLogger(__name__)
        enabler_id = self.enabler_id
        cum_rules = self.get_cumulative_rules(sub_id, level)
        
        # 🟢 [ANCHOR] สร้างตัวระบุข้อที่แข็งแรง
        # เช่น "KM 1.1" หรือ "1.1 KM"
        id_anchor = f"{enabler_id} {sub_id}"
        
        # รวบรวม Keywords
        plan_kws = cum_rules.get('plan_keywords', [])
        do_kws = cum_rules.get('do_keywords', [])
        check_kws = cum_rules.get('check_keywords', [])
        act_kws = cum_rules.get('act_keywords', [])
        all_kws = list(set(plan_kws + do_kws + check_kws + act_kws))
        keywords_str = " ".join(all_kws[:8]) 

        queries = []

        # Query 1: Direct Matching (เน้นข้อ + ใจความเกณฑ์) - ต้องอยู่หน้าสุด
        # ตัวอย่าง: "KM 1.1 ผู้บริหารกำหนดวิสัยทัศน์ ทิศทาง"
        queries.append(f"{id_anchor} {statement_text}")

        # Query 2: Document Type Anchor (เน้นหาชนิดไฟล์ในข้อนั้นๆ)
        if level <= 2:
            # เน้นรากฐาน: ประกาศ นโยบาย แผน
            queries.append(f"{id_anchor} ประกาศ นโยบาย แผนปฏิบัติการ คำสั่งแต่งตั้ง {keywords_str}")
        else:
            # เน้นหลักฐานเชิงประจักษ์: รายงาน สรุปผล
            queries.append(f"{id_anchor} รายงานผล สรุปกิจกรรม บันทึกข้อความ {keywords_str}")

        # Query 3: Maturity Specific (เจาะเฟส PDCA)
        # เพิ่ม Query ที่เจาะจง Maturity Rules ที่ดึงมา
        specific_rule = cum_rules.get('instruction', '')
        if specific_rule:
            queries.append(f"{id_anchor} {specific_rule}")

        # Query 4: Cross-Check (ใช้ Keyword ล้วนๆ แต่ต้องมี ID Anchor นำหน้า)
        # เพื่อให้ BM25 ทำงานได้แม่นยำที่สุด
        if level >= 3:
            ca_kws = " ".join(list(set(check_kws + act_kws))[:5])
            queries.append(f"{id_anchor} ผลประเมิน ติดตาม ตรวจสอบ ปรับปรุง {ca_kws}")

        # --- กรองและส่งออก ---
        final_queries = []
        seen = set()
        for q in queries:
            q_strip = q.strip()
            if q_strip and q_strip not in seen:
                final_queries.append(q_strip)
                seen.add(q_strip)

        return final_queries[:5]

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
        [FINAL POLISHED v2026.7] - Cumulative Maturity Rules Engine
        ------------------------------------------------------------
        - รวบรวมกฎสะสมจาก L1 ถึง current_level (Maturity Core)
        - ผสม Generic Defaults จาก _enabler_defaults ของโครงการ
        - ใช้ set ป้องกัน duplication ของ keywords เพื่อลด token usage
        - แยก instructions เป็น list ตาม level เพื่อส่งต่อให้ LLM ประมวลผลได้แม่นยำ
        - Logging ครบถ้วนสำหรับการ debug ในระบบ Production
        """
        import logging
        logger = logging.getLogger("AssessmentApp")

        # 1. เริ่มต้นด้วย Defaults (Global PDCA Keywords จากไฟล์ JSON)
        defaults = self.contextual_rules_map.get('_enabler_defaults', {})
        
        # ใช้ set เพื่อจัดการความซ้ำซ้อนของคำค้นหาโดยอัตโนมัติ
        cum_plan = set(defaults.get('plan_keywords', []))
        cum_do   = set(defaults.get('do_keywords', []))
        cum_check = set(defaults.get('check_keywords', []))
        cum_act  = set(defaults.get('act_keywords', []))

        required_phases = set()
        instructions = []

        # 2. สะสมกฎจาก Level 1 จนถึง current_level (แนวคิด Maturity สะสมคะแนน)
        sub_rules = self.contextual_rules_map.get(sub_id, {})
        
        for lv in range(1, current_level + 1):
            lv_key = f"L{lv}"
            level_rule = sub_rules.get(lv_key, {})
            
            # ข้ามเลเวลที่ไม่มีการนิยามกฎไว้ (Graceful Handling)
            if not level_rule:
                continue

            # อัปเดต Keywords เฉพาะของเลเวลนั้นๆ เข้าสู่กลุ่มสะสม
            cum_plan.update(level_rule.get('plan_keywords', []))
            cum_do.update(level_rule.get('do_keywords', []))
            cum_check.update(level_rule.get('check_keywords', []))
            cum_act.update(level_rule.get('act_keywords', []))

            # อัปเดต Phase ที่จำเป็น (เช่น L1 อาจต้องการแค่ P, D แต่ L4 ต้องการ P, D, C, A)
            if 'require_phase' in level_rule:
                required_phases.update(level_rule.get('require_phase', []))

            # เก็บคำสั่งพิเศษรายเลเวล (เช่น "L1: มติรับทราบ = ผ่าน")
            specific = level_rule.get('specific_contextual_rule')
            if specific:
                instructions.append(f"L{lv}: {specific.strip()}")

        # 3. สรุปผลลัพธ์และเตรียมโครงสร้างสำหรับส่งให้ Engine ส่วนถัดไป
        result = {
            "plan_keywords": list(cum_plan),
            "do_keywords": list(cum_do),
            "check_keywords": list(cum_check),
            "act_keywords": list(cum_act),
            "required_phases": sorted(list(required_phases)),
            "instructions": instructions,  # ส่งแบบ List เพื่อให้ LLM แยกแยะได้ง่าย
            "cumulative_instruction": "\n".join(instructions) if instructions else ""
        }

        # 4. Logging สำหรับการตรวจสอบการทำงาน (Monitoring)
        logger.debug(
            f"[RULE_CUMULATIVE] {sub_id} L{current_level} | "
            f"Keywords: P={len(result['plan_keywords'])} | "
            f"D={len(result['do_keywords'])} | "
            f"C={len(result['check_keywords'])} | "
            f"A={len(result['act_keywords'])} | "
            f"Phases={result['required_phases']} | "
            f"Instructions={len(instructions)}"
        )

        return result


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


    def post_process_llm_result(self, llm_output: Dict[str, Any], level: int) -> Dict[str, Any]:
        """
        [REVISED v2026.3] - ผสาน Hard-Fail Logic และ Heuristic จากเวอร์ชันคุณ
        เข้ากับ Class Engine ปัจจุบัน
        """
        if not llm_output:
            return {"is_passed": False, "score": 0.0, "reason": "No Output from LLM"}

        # 1. นิยามความสัมพันธ์ (Mapping Keys)
        # ปรับให้รองรับทั้งชื่อเต็มและชื่อย่อที่ LLM อาจจะคืนมา
        extraction_map = {
            "Extraction_P": ["P_Plan_Score", "score_p"],
            "Extraction_D": ["D_Do_Score", "score_d"],
            "Extraction_C": ["C_Check_Score", "score_c"],
            "Extraction_A": ["A_Act_Score", "score_a"]
        }
        phase_map = {"Extraction_P": "Plan", "Extraction_D": "Do", "Extraction_C": "Check", "Extraction_A": "Act"}
        reason_text = llm_output.get("reason", "")
        is_consistent = llm_output.get("consistency_check", True)

        for ext_key, score_keys in extraction_map.items():
            val = llm_output.get(ext_key, "-") or "-"
            raw_val = str(val).strip()
            
            # --- ก. ทำความสะอาดข้อมูล ---
            content_only = re.sub(r'[^\u0e00-\u0e7fa-zA-Z0-9]', '', raw_val)
            is_negative = raw_val in ["-", "N/A", "n/a", "ไม่พบ", "ไม่มี", "ไม่ปรากฏ", "ไม่ระบุ"]
            is_empty = (not content_only) or is_negative
            
            # ดึงคะแนนปัจจุบัน (ลองดูทุก key ที่เป็นไปได้)
            current_score = 0.0
            target_key = score_keys[0]
            for sk in score_keys:
                if sk in llm_output:
                    current_score = float(llm_output.get(sk, 0))
                    target_key = sk
                    break

            if is_empty and current_score > 0:
                # --- ข. [DYNAMIC HEURISTIC OVERRIDE] ---
                # ดึง Keywords จาก self.global_pdca_keywords (หรือตัวแปรที่คุณมีในเครื่อง)
                phase_name = phase_map.get(ext_key)
                # ดึง keywords มาเช็คซ้ำใน reason
                raw_keywords = getattr(self, 'global_pdca_keywords', {}).get(phase_name, [])
                
                found_keyword = any(kw in reason_text for kw in raw_keywords if len(kw) > 1)
                
                if found_keyword:
                    self.logger.info(f" 🛡️ [Heuristic Pass] L{level}: {ext_key} empty but found keyword in Reason.")
                    continue 
                    
                # ริบคะแนนหากไม่มีหลักฐานจริงใน Extraction field
                self.logger.warning(f" 🚨 [Revoke] L{level}: {target_key} revoked. No evidence in {ext_key}")
                llm_output[target_key] = 0.0

        # 2. Normalize คะแนนรายเฟส (Max 2.0 ต่อเฟส)
        def get_sc(keys):
            for k in keys:
                if k in llm_output: return float(llm_output[k])
            return 0.0

        p = round(min(get_sc(["P_Plan_Score", "score_p"]), 2.0), 1)
        d = round(min(get_sc(["D_Do_Score", "score_d"]), 2.0), 1)
        c = round(min(get_sc(["C_Check_Score", "score_c"]), 2.0), 1)
        a = round(min(get_sc(["A_Act_Score", "score_a"]), 2.0), 1)
        pdca_sum = round(p + d + c + a, 1)
        
        # 3. SE-AM Threshold & Hard-Fail Logic
        threshold_map = {1: 1, 2: 2, 3: 4, 4: 6, 5: 8}
        threshold = threshold_map.get(level, 2)
        is_passed = pdca_sum >= threshold

        # กฎบังคับตก (Hard-Fail)
        fail_reason = ""
        if is_passed:
            if not is_consistent:
                is_passed = False
                fail_reason = "พบความขัดแย้งของข้อมูลหลักฐาน (Consistency Fail)"
            elif level == 3 and c <= 0:
                is_passed = False
                fail_reason = "ระดับ 3 บังคับต้องมีผลการวัด (C > 0)"
            elif level == 4 and a <= 0:
                is_passed = False
                fail_reason = "ระดับ 4 บังคับต้องมีการปรับปรุง (A > 0)"
            elif level == 5 and (c < 2.0 or a < 2.0):
                is_passed = False
                fail_reason = "ระดับ 5 ต้องการคะแนน C และ A เต็ม (2.0)"

        # 4. Final Object Update
        llm_output.update({
            "score": pdca_sum,
            "pdca_breakdown": {"P": p, "D": d, "C": c, "A": a},
            "is_passed": is_passed,
            "fail_reason": fail_reason,
            "consistency_check": is_consistent,
            "pass_threshold": threshold
        })

        return llm_output

    def _check_contextual_rule_condition(
        self, 
        condition: Dict[str, Any], 
        sub_id: str, 
        level: int, 
        top_evidences: List[Dict[str, Any]]
    ) -> bool:
        """
        [Revised 2026] ตรวจสอบเงื่อนไขตามหลัก Maturity Accumulation
        - ตรวจสอบความต่อเนื่องของ Level (Sequential Pass)
        - ตรวจสอบคุณภาพหลักฐาน (Rerank Quality) ตาม Phase บังคับใน JSON
        """
        self.logger.info(f"🔍 [Maturity Check] Verifying rules for {sub_id} L{level}...")
        
        # 1. ดึงกฎสะสมจาก JSON ใหม่ของเรา
        cum_rules = self.get_cumulative_rules(sub_id, level)
        required_phases = cum_rules.get('phases', [])

        # 2. Sequential Check: ถ้าไม่ใช่ L1 ต้องเช็คว่าเลเวลก่อนหน้า "ผ่านจริง" หรือไม่
        if level > 1:
            for prev_lv in range(1, level):
                if not self._is_previous_level_passed(sub_id, prev_lv):
                    self.logger.warning(f"❌ Maturity Gap: {sub_id} L{prev_lv} must pass before L{level}.")
                    return False

        # 3. Evidence Quality Check: ตรวจสอบ 'require_phase' ที่บังคับใน JSON
        # เราจะเข้มงวดกับ Phase ที่ระบุไว้ใน JSON มากกว่า Phase ทั่วไป
        for phase_to_check in required_phases:
            # ดึงเกณฑ์คะแนน (ถ้าไม่มีใน JSON ให้ใช้ Global Threshold)
            # เราสามารถขยาย JSON ให้ใส่ min_rerank ในแต่ละเฟสได้ในอนาคต
            threshold = globals().get('CRITICAL_CA_THRESHOLD', 0.60) if phase_to_check in ['C', 'A'] else 0.40
            
            found_valid_evidence = any(
                doc.get('pdca_tag') == phase_to_check and 
                doc.get('rerank_score', 0.0) >= threshold
                for doc in top_evidences
            )
            
            if not found_valid_evidence:
                self.logger.warning(f"❌ Evidence Gap: Required Phase '{phase_to_check}' not found or quality too low (Threshold: {threshold}).")
                return False

        # 4. ตรวจสอบเงื่อนไข 'and' เพิ่มเติม (หากใน JSON ยังมี Logic พิเศษหลงเหลืออยู่)
        if 'and' in condition:
            # ... (รักษา Logic การวน Loop เช็ค 'and' เดิมของคุณไว้เพื่อความ Backward Compatible) ...
            pass

        self.logger.info(f"✅ [Maturity Check] {sub_id} L{level} passed all rule conditions.")
        return True

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
        สร้าง Prompt Constraint แบบ Dynamic ตามเกณฑ์ที่ระบุใน Contextual Rules
        [REVISED 2026] - ยืดหยุ่นตาม Sub-ID และ Level จริง
        """
        # 1. ดึงกฎเฉพาะของเลเวลนั้นจาก JSON
        required_phases = self.get_rule_content(sub_id, level, "require_phase") or []
        specific_rule = self.get_rule_content(sub_id, level, "specific_contextual_rule") or ""
        
        # แปลงตัวย่อ P, D, C, A เป็นชื่อเต็มเพื่อบอก AI
        phase_map = {"P": "วางแผน (Plan)", "D": "ปฏิบัติ (Do)", "C": "ตรวจสอบ (Check)", "A": "ปรับปรุง (Act)"}
        full_phase_names = [phase_map.get(p, p) for p in required_phases]

        # 2. สร้าง Prompt พื้นฐาน
        constraint_msg = f"--- ข้อจำกัดการประเมินระดับวุฒิภาวะ (Level {level}) ---\n"
        
        if full_phase_names:
            constraint_msg += f"🎯 บังคับตรวจสอบเฟส: {', '.join(full_phase_names)} เป็นหลัก\n"
        
        # 3. ใส่กฎพิเศษ (ถ้ามี)
        if specific_rule:
            constraint_msg += f"⚠️ กฎเฉพาะข้อนี้: {specific_rule}\n"
        
        # 4. ใส่ Logic มาตรฐานสำหรับ Maturity Level
        if level >= 3:
            constraint_msg += (
                "💡 หมายเหตุสำหรับ L3 ขึ้นไป: หลักฐานต้องแสดงให้เห็นถึงความต่อเนื่อง "
                "และการนำผลการประเมินมาปรับปรุงจริง ไม่ใช่เพียงแค่มีแผนงาน\n"
            )
            
        return constraint_msg
        

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

    
    # -------------------- Evidence Classification Helper (Full Fixed 2026) --------------------
    def _get_mapped_uuids_and_priority_chunks(
        self,
        sub_id: str,
        level: int,
        statement_text: str,
        level_constraint: str,
        vectorstore_manager: Optional['VectorStoreManager']
    ) -> Tuple[List[str], List[Dict]]:
        """
        [DYNAMIC CONTINUITY v2026.5] - No Manual UUIDs
        ----------------------------------------------
        - Auto-History: ดึงหลักฐานที่ AI เคยเลือกไว้ใน Level ก่อนหน้ามาเป็นฐาน (Baseline)
        - Semantic Hinting: ใช้ Keywords จาก Rules ไปช่วยสกัดไฟล์สำคัญล่วงหน้า
        - Zero Manual Input: ไม่ต้องกรอก UUID เอง ระบบจัดการ ID ใน Memory ทั้งหมด
        """
        from copy import deepcopy
        priority_chunks = []
        mapped_stable_ids = []

        # 1. 🧠 [AUTO-HISTORY] ดึงความจำจาก Level ก่อนหน้า (ถ้ามี)
        # ระบบจะจำ UUID ที่มันเคยใช้ตรวจผ่านใน L1 มาส่งต่อให้ L2 เองอัตโนมัติ
        for key, evidences in self.evidence_map.items():
            if key.startswith(f"{sub_id}.L") and isinstance(evidences, list):
                try:
                    prev_level = int(key.split(".L")[-1])
                    if prev_level < level:
                        history_items = deepcopy(evidences)
                        for item in history_items:
                            item["is_baseline"] = True
                            item["rerank_score"] = max(item.get("rerank_score", 0.0), 0.85)
                        priority_chunks.extend(history_items)
                except (ValueError, IndexError):
                    continue

        # 2. 🔍 [SEMANTIC HINTING] สำหรับ L1 หรือกรณีที่ต้องการพยุงคะแนน
        # ถ้าไม่มีประวัติ (เช่น L1) ให้ใช้คำสำคัญจาก Rule ไปทำ 'Pre-Search' สั้นๆ เพื่อหาไฟล์หลัก
        if not priority_chunks:
            rule_config = self.contextual_rules_map.get(sub_id, {}).get(str(level), {})
            # ดึงคำสำคัญจาก plan_keywords หรือ do_keywords ใน JSON มาเป็นคำค้น
            hints = rule_config.get("plan_keywords", [])[:2] # เอาแค่ 2 คำหลัก
            if hints and vectorstore_manager:
                self.logger.info(f"🔎 L1 Discovery: Searching for anchors using hints: {hints}")
                discovery_result = vectorstore_manager.quick_search(
                    query=f"{sub_id} {' '.join(hints)}",
                    top_k=5 # ดึงมาแค่ 5 อันดับแรกเพื่อเป็นสมอเรือ
                )
                for chunk in discovery_result:
                    chunk["rerank_score"] = 0.85 # บูสต์คะแนนให้ลอยขึ้นมา
                    priority_chunks.append(chunk)

        if not priority_chunks:
            return [], []

        # 3. 💧 [ROBUST HYDRATION] เติมเนื้อหาเต็ม
        try:
            priority_chunks = self._robust_hydrate_documents_for_priority_chunks(
                chunks_to_hydrate=priority_chunks,
                vsm=vectorstore_manager
            )
        except Exception as e:
            self.logger.error(f"❌ Hydration failed: {e}")

        # 4. 🎯 [ID SYNC] รวบรวม IDs ส่งกลับไปเป็น Filter ให้ Main RAG
        seen_ids = set()
        for chunk in priority_chunks:
            sid = chunk.get("stable_doc_uuid") or chunk.get("doc_id")
            if sid and isinstance(sid, str):
                if sid not in seen_ids and len(sid) >= 32:
                    mapped_stable_ids.append(sid)
                    seen_ids.add(sid)

        self.logger.info(f"✅ Continuity Ready: {len(priority_chunks)} priority chunks | {len(mapped_stable_ids)} IDs")
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
        [ULTIMATE STATS v2026.3] Weighted Maturity Calculation
        ------------------------------------------------------
        - คำนวณ Maturity Score (0-5) ตามน้ำหนักความสำคัญของแต่ละเกณฑ์
        - คำนวณ Progress % เทียบกับการผ่านระดับ L5 ทั้งหมด
        - กำหนด Maturity Level Label โดยอิงจากระดับที่ผ่านเกณฑ์จริง (Conservative Approach)
        """
        from config.global_vars import MAX_LEVEL
        results = self.final_subcriteria_results
        
        # 1. 🛡️ Safety Guard: กรณีไม่มีข้อมูลผลลัพธ์
        if not results:
            self.total_stats = {
                "overall_avg_score": 0.0,
                "overall_level_label": "L0",
                "progress_percent": 0.0,
                "total_weighted_score": 0.0,
                "total_possible_weight": 0.0,
                "record_id": self.current_record_id,
                "status": "No Data"
            }
            return

        # 2. ⚖️ คะแนนถ่วงน้ำหนัก (Weighted Summation)
        # weighted_score = (Level / MAX_LEVEL) * weight ของข้อนั้นๆ
        total_weighted_score_achieved = sum(r.get('weighted_score', 0.0) for r in results)
        total_possible_weight = sum(r.get('weight', 0.0) for r in results)

        # 3. 📊 คำนวณ Maturity Score (Scale 0.0 - 5.0)
        overall_avg_score = 0.0
        if total_possible_weight > 0:
            # สูตร: (คะแนนรวมที่ได้ / น้ำหนักรวม) * 5
            overall_avg_score = round((total_weighted_score_achieved / total_possible_weight) * MAX_LEVEL, 2)
        
        # 4. 📈 คำนวณ Progress Percentage (0-100%)
        # เทียบกับคะแนนเต็มสูงสุดที่เป็นไปได้ (น้ำหนักรวม * 5)
        max_possible_points = total_possible_weight * MAX_LEVEL
        progress_percent = 0.0
        if max_possible_points > 0:
            progress_percent = round((total_weighted_score_achieved / max_possible_points) * 100, 2)

        # 5. 🏷️ กำหนด Maturity Level Label (Audit Logic)
        # ใช้ค่าเฉลี่ยของระดับที่ผ่าน "ครบถ้วน" (Highest Full Level) เพื่อความแม่นยำตามเกณฑ์ SE-AM
        avg_full_level = sum(r.get('highest_full_level', 0) for r in results) / len(results)
        final_level = int(avg_full_level) # ใช้ Floor (ตัดเศษ) เพื่อความ Conservative หรือใช้ round() ตามความเหมาะสม
        overall_level_label = f"L{min(max(final_level, 0), MAX_LEVEL)}"
        
        # 6. 📝 บันทึกผลสรุป (Comprehensive Object)
        self.total_stats = {
            "overall_avg_score": min(overall_avg_score, float(MAX_LEVEL)),
            "overall_level_label": overall_level_label,
            "total_weighted_score": round(total_weighted_score_achieved, 2),
            "total_possible_weight": total_possible_weight,
            "progress_percent": progress_percent,
            "gap_to_full": round(total_possible_weight - total_weighted_score_achieved, 2),
            "assessed_count": len(results),
            "total_subcriteria_in_rubric": len(self._flatten_rubric_to_statements()),
            "enabler": self.config.enabler,
            "target_sub_id": target_sub_id,
            "record_id": self.current_record_id,
            "assessed_at": datetime.now().isoformat()
        }

        # 7. 📢 Logging สรุปผลภาพรวม
        self.logger.info(f"--- 🏁 ASSESSMENT COMPLETE (ID: {self.current_record_id}) ---")
        self.logger.info(f"🏆 Overall Level: {overall_level_label} | Score: {overall_avg_score}/{MAX_LEVEL}")
        self.logger.info(f"📊 Progress: {progress_percent}% | Weighted Score: {self.total_stats['total_weighted_score']}/{total_possible_weight}")
        self.logger.info(f"----------------------------------------------------------")
       
    def _export_results(self, results: dict, sub_criteria_id: str, **kwargs) -> str:
        """
        [ULTIMATE EXPORTER v2026.3]
        ---------------------------
        - 📂 Hierarchical Storage: เก็บไฟล์แยกตาม Tenant/Year/Enabler อย่างเป็นระเบียบ
        - 📑 Self-Contained Metadata: ฝังข้อมูลบริบท (Model, Target, ID) ลงใน JSON
        - 📊 Audit Summary: สรุปผลคะแนนเบื้องต้นลงในไฟล์เพื่อให้ระบบอื่นอ่านง่าย
        - 🛡️ Path Resilience: ระบบจัดการ Path ที่ป้องกันความผิดพลาดจากการสร้าง Folder
        """
        import os
        import json
        from datetime import datetime

        # 1. ⚙️ เตรียมข้อมูลพื้นฐาน
        record_id = kwargs.get("record_id", getattr(self, "current_record_id", "no_id"))
        enabler = self.config.enabler
        tenant = self.config.tenant
        year = str(self.config.year)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 2. 📁 การจัดการ Path (Enterprise Directory Structure)
        # โครงสร้าง: data_store/{tenant}/exports/{year}/{enabler}/
        export_dir = os.path.join("data_store", tenant, "exports", year, enabler)
        
        try:
            os.makedirs(export_dir, exist_ok=True)
            file_name = f"assessment_{enabler}_{record_id}_{sub_criteria_id}_{timestamp}.json"
            full_path = os.path.join(export_dir, file_name)
        except Exception as e:
            self.logger.error(f"❌ Directory creation failed: {e}")
            # Fallback ไปยัง root data_store กรณี Permission มีปัญหา
            full_path = f"assessment_fallback_{record_id}.json"

        # 3. 📊 เสริมข้อมูล Summary (ดึง Logic จาก Origin มาทำให้ครบถ้วน)
        if 'summary' not in results:
            results['summary'] = {}
        
        summary = results['summary']
        sub_res_list = results.get('sub_criteria_results', [])

        # ฝังข้อมูลระบุตัวตน (Identity Metadata)
        results['metadata'] = {
            "record_id": record_id,
            "tenant": tenant,
            "year": year,
            "enabler": enabler,
            "model_used": getattr(self.config, "model_name", "unknown"),
            "target_level": self.config.target_level,
            "export_at": datetime.now().isoformat()
        }

        # สรุปผลสถิติตามประเภทการรัน (Single vs All)
        if str(sub_criteria_id).lower() != "all" and len(sub_res_list) > 0:
            main_res = sub_res_list[0]
            summary.update({
                "highest_pass_level": main_res.get('highest_full_level', 0),
                "achieved_weight": round(main_res.get('weighted_score', 0.0), 2),
                "total_weight": main_res.get('weight', 0.0),
                "is_target_achieved": main_res.get('target_level_achieved', False)
            })
        else:
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

        # นับจำนวน Action Plan ทั้งหมด
        summary['total_action_plan_items'] = sum(len(r.get('action_plan', [])) for r in sub_res_list if isinstance(r.get('action_plan'), list))

        # 4. 💾 การเขียนไฟล์ (Writing Process)
        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            
            self.logger.info(f"💾 EXPORT SUCCESS: {full_path}")
            
            # สรุปบรรทัดสุดท้ายใน Log
            final_lvl = summary.get('highest_pass_level', summary.get('highest_pass_level_overall', 0))
            self.logger.info(f"📊 [FINAL REPORT] Record: {record_id} | Level: L{final_lvl} | Score: {summary.get('total_achieved_weight', summary.get('achieved_weight', 0))}")
            
            return full_path
            
        except Exception as e:
            self.logger.error(f"❌ Export failed: {str(e)}")
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
        """
        [ULTIMATE SAVER v2026.3] Evidence Persistence & Strength Calculation
        --------------------------------------------------------------------
        - 🛡️ Robust Extraction: รองรับทั้ง Dict และ LangChain Document
        - 🔗 Unique Identification: ระบบ Fallback ID ด้วย Hash (ป้องกัน Data Loss)
        - 📊 Precision Scoring: คำนวณความมั่นใจของหลักฐานผ่าน Scoring Gate
        - 🧩 UI Ready: จัดเตรียม Metadata สำหรับแสดงผลเลขหน้าและชื่อไฟล์ในรายงาน
        """
        import hashlib
        map_key = f"{sub_id}.L{level}"
        new_evidence_list: List[Dict[str, Any]] = []
        
        self.logger.info(f"💾 [EVI SAVE] Starting save for {map_key} ({len(level_temp_map)} potential chunks)")

        for chunk in level_temp_map:
            # 🎯 1. 📦 การสกัดข้อมูลเบื้องต้น (รองรับโครงสร้างข้อมูลที่หลากหลาย)
            if isinstance(chunk, dict):
                meta = chunk.get("metadata", {}) or {}
                # ลำดับความสำคัญ: Direct Key > Metadata > ID
                chunk_uuid = chunk.get("chunk_uuid") or chunk.get("id") or meta.get("chunk_uuid")
                stable_doc_uuid = (chunk.get("stable_doc_uuid") or chunk.get("doc_id") or 
                                  meta.get("stable_doc_uuid") or meta.get("doc_id"))
                source = chunk.get("source") or meta.get("source") or "N/A"
                text_content = chunk.get("page_content") or chunk.get("text") or ""
            else:
                # กรณีเป็น LangChain Document Object
                meta = getattr(chunk, "metadata", {}) or {}
                chunk_uuid = meta.get("chunk_uuid") or getattr(chunk, "id", None)
                stable_doc_uuid = meta.get("stable_doc_uuid") or meta.get("doc_id")
                source = meta.get("source", "N/A")
                text_content = getattr(chunk, "page_content", "")

            # 🎯 2. 🛡️ Validation & Fallback (ระบบป้องกัน Error "ID Missing")
            if not stable_doc_uuid or not chunk_uuid:
                if text_content and len(str(text_content)) > 10:
                    # สร้าง Hash จากเนื้อหาเพื่อให้ได้ ID ที่คงที่ (Deterministic)
                    content_hash = hashlib.md5(str(text_content).encode()).hexdigest()
                    chunk_uuid = chunk_uuid or f"hash-{content_hash[:16]}"
                    stable_doc_uuid = stable_doc_uuid or f"doc-{content_hash[16:32]}"
                    self.logger.warning(f"⚠️ [EVI SAVE] Generated Hash-ID for source: {source}")
                else:
                    self.logger.error(f"❌ [EVI SAVE] Critical ID Missing & No Content! Skipping source: {source}")
                    continue

            # 🎯 3. 📄 ดึงข้อมูลเลขหน้า (รองรับระบบ Label ของ PDF)
            page_val = (
                meta.get("page_label") or 
                meta.get("page") or 
                meta.get("page_number") or 
                (chunk.get("page") if isinstance(chunk, dict) else "N/A")
            )

            # 🎯 4. 📝 สร้าง Evidence Entry มาตรฐาน (สำหรับนำไปใช้ใน JSON Report)
            evidence_entry = {
                "sub_id": sub_id,
                "level": level,
                "relevance_score": float(chunk.get("rerank_score") or chunk.get("score") or 0.5 if isinstance(chunk, dict) else 0.5),
                "doc_id": str(stable_doc_uuid),
                "stable_doc_uuid": str(stable_doc_uuid),
                "chunk_uuid": str(chunk_uuid),
                "source": source,
                "source_filename": meta.get("source_filename") or meta.get("file_name") or os.path.basename(str(source)),
                "page": str(page_val),
                "pdca_tag": (chunk.get("pdca_tag") if isinstance(chunk, dict) else meta.get("pdca_tag")) or "Other", 
                "status": "PASS" if llm_result.get("is_passed") else "FAIL", 
                "timestamp": datetime.now().isoformat(),
            }
            new_evidence_list.append(evidence_entry)

        # 🎯 5. ⚖️ คำนวณความเข้มข้นของหลักฐาน (Evidence Strength)
        # ส่งหลักฐานที่กรองแล้วเข้า Score Gate เพื่อดูว่าจะ Cap คะแนน AI หรือไม่
        evi_cap_data = self._calculate_evidence_strength_cap(
            top_evidences=new_evidence_list,
            level=level,
            highest_rerank_score=highest_rerank_score
        )
        final_evi_str = evi_cap_data.get('max_evi_str_for_prompt', 0.0)

        # 🎯 6. 💾 บันทึกลง Memory Map (รองรับทั้ง Main Process และ Worker)
        if new_evidence_list:
            self.evidence_map.setdefault(map_key, []).extend(new_evidence_list)
            self.temp_map_for_save.setdefault(map_key, []).extend(new_evidence_list)
            self.logger.info(f"✅ [EVIDENCE SAVED] {map_key}: {len(new_evidence_list)} chunks | Strength: {final_evi_str}")
        else:
            self.logger.warning(f"⚠️ [EVI SAVE] No valid evidence entries for {map_key}")
            final_evi_str = 0.0
        
        return final_evi_str
    
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

    def _calculate_evidence_strength_cap(
        self,
        top_evidences: List[Union[Dict[str, Any], Any]],
        level: int,
        highest_rerank_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        [ULTIMATE REVISE v2026.3] Relevant Score Gate
        --------------------------------------------
        - Fixes: UnboundLocalError & Metadata Extraction
        - Scoring: Metadata (P1) > Adaptive Loop (P2) > Regex Tail (P3)
        - Logic: หากคะแนนสูงสุด < Threshold จะทำการ Cap คะแนน 'ความมั่นใจ' ของ AI
        """
        import re
        score_keys = [
            "rerank_score", "score", "relevance_score",
            "_rerank_score_force", "_rerank_score",
            "Score", "RelevanceScore"
        ]

        # 1. ⚙️ Configuration (ดึงจาก Class หรือ Global Vars)
        threshold = getattr(self, "RERANK_THRESHOLD", 0.35) 
        cap_value = getattr(self, "MAX_EVI_STR_CAP", 5.0)

        # 2. 📍 Baseline Score (จาก Adaptive RAG Loop ที่เราวนหามา)
        max_score_found = highest_rerank_score if highest_rerank_score is not None else 0.0
        max_score_source = "Adaptive_RAG_Loop" if highest_rerank_score is not None else "N/A"

        for idx, doc in enumerate(top_evidences, 1):
            current_doc_source = "Unknown_Source"
            current_score = 0.0
            page_content = ""
            metadata = {}

            # 3. 📦 Object Extraction (รองรับทั้ง Dict และ Langchain Document)
            if isinstance(doc, dict):
                metadata = doc.get("metadata", {}) or {}
                page_content = doc.get("page_content", "") or doc.get("text", "") or doc.get("content", "")
                current_doc_source = (
                    metadata.get("file_name") or metadata.get("source_filename") or 
                    metadata.get("filename") or metadata.get("source") or 
                    doc.get("source") or f"Doc_ID_{idx}"
                )
            else:
                metadata = getattr(doc, "metadata", {}) or {}
                page_content = getattr(doc, "page_content", "") or getattr(doc, "text", "")
                current_doc_source = (
                    metadata.get("file_name") or metadata.get("source_filename") or 
                    getattr(doc, "source", "Unknown_Document")
                )

            # 4. 🔍 Priority 1: ค้นหาคะแนนจาก Metadata/Top-level
            for key in score_keys:
                score_val = metadata.get(key)
                if score_val is None and isinstance(doc, dict):
                    score_val = doc.get(key)

                if score_val is not None:
                    try:
                        temp_score = float(score_val)
                        if 0.0 < temp_score <= 1.0:
                            current_score = temp_score
                            break
                    except (ValueError, TypeError): continue

            # 5. 📑 Priority 2: Regex Fallback (กรณีคะแนนถูกฝังใน Text)
            if current_score <= 0.0 and page_content and isinstance(page_content, str):
                tail = page_content[-1200:].replace('\n', ' ')
                patterns = [
                    r"Relevance[ :]+([0-9]*\.?[0-9]+)",
                    r"Score[ :]+([0-9]*\.?[0-9]+)",
                    r"\[Score: ([0-9]*\.?[0-9]+)\]",
                    r"rerank_score['\"]?[\s:=]+([0-9]*\.?[0-9]+)",
                    r"\|\s*([0-9]\.[0-9]+)\s*\|" # ดักจับ Markdown Table Score
                ]
                for pat in patterns:
                    m = re.search(pat, tail, re.IGNORECASE)
                    if m:
                        try:
                            ts = float(m.group(1))
                            if 0.0 < ts <= 1.0:
                                current_score = ts
                                break
                        except: continue

            # 6. 🛡️ Clamp & Protection
            if current_score > 1.0:
                self.logger.warning(f"🚨 Score Clamp L{level}: {current_score} > 1.0 จาก '{current_doc_source}' (Scaled to 0.0)")
                current_score = 0.0

            # 7. 🏆 คัดเลือกคะแนนสูงสุด
            if current_score > max_score_found:
                max_score_found = current_score
                max_score_source = current_doc_source

        # 8. ⚖️ Decision Gate (Capping Logic)
        # ถ้าคะแนนดีที่สุดยังต่ำกว่าเกณฑ์ (Threshold) -> AI จะถูกสั่งให้ "ระมัดระวัง" ในการให้ผ่าน
        is_capped = max_score_found < threshold
        max_evi_str_for_prompt = cap_value if is_capped else 10.0

        status_icon = "🚨" if is_capped else "✅"
        self.logger.info(
            f"{status_icon} Evi Str {'CAPPED' if is_capped else 'FULL'} L{level}: "
            f"Best {max_score_found:.4f} from '{max_score_source}' (Threshold: {threshold})"
        )

        return {
            "is_capped": is_capped,
            "max_evi_str_for_prompt": max_evi_str_for_prompt,
            "highest_rerank_score": round(float(max_score_found), 4),
            "max_score_source": str(max_score_source),
        }

    def _robust_hydrate_documents_for_priority_chunks(
        self,
        chunks_to_hydrate: List[Dict],
        vsm: Optional['VectorStoreManager'],
        current_sub_id: Optional[str] = None,
        level: Optional[int] = None
    ) -> List[Dict]:
        """
        [ULTIMATE HYDRATION v2026.3]
        - ดึงเนื้อหาเต็ม (Full Text) ของ Priority Chunks เพื่อให้ AI เห็นบริบททั้งหมด
        - ระบบ Fallback Scoring (1.0 สำหรับตัวที่ดึงสำเร็จ, 0.85 สำหรับตัวที่ล้มเหลว)
        - เพิ่มระบบ Deduplication เพื่อป้องกันเนื้อหาซ้ำซ้อนใน Prompt
        """
        active_sub_id = current_sub_id or getattr(self, 'sub_id', 'unknown')
        if not chunks_to_hydrate:
            self.logger.debug(f"ℹ️ [HYDRATION] No chunks to hydrate for L{level} {active_sub_id}")
            return []

        TAG_ABBREV = {
            "PLAN": "P", "DO": "D", "CHECK": "C", "ACT": "A",
            "P": "P", "D": "D", "C": "C", "A": "A"
        }

        # 1. 🏷️ Helper: จัดหมวดหมู่ PDCA ทันทีที่เนื้อหาถูกเติม
        def _safe_classify(text: str) -> str:
            try:
                raw = classify_by_keyword(
                    text=text, sub_id=active_sub_id, level=level,
                    contextual_rules_map=self.contextual_rules_map
                )
                if not raw: return "Other"
                return TAG_ABBREV.get(str(raw).upper(), "Other")
            except Exception as e:
                self.logger.warning(f"⚠️ PDCA classify failed in hydration: {e}")
                return "Other"

        # 2. 📏 Helper: ปรับค่ามาตรฐานให้ Chunk
        def _standardize_chunk(chunk: Dict, score: float):
            chunk.setdefault("is_baseline", True)
            text = chunk.get("text", "").strip()
            if text:
                chunk["pdca_tag"] = _safe_classify(text)
                # Boost คะแนนให้หลักฐานกลุ่ม Priority เพื่อให้ AI ให้ความสำคัญ
                chunk["rerank_score"] = max(chunk.get("rerank_score", 0.0), score)
                chunk["score"] = max(chunk.get("score", 0.0), score)
            return chunk

        # 3. 🔑 Extract IDs สำหรับดึงข้อมูลจาก Database
        stable_ids = {
            sid for c in chunks_to_hydrate
            if (sid := (c.get("stable_doc_uuid") or c.get("doc_id") or c.get("chunk_uuid")))
        }

        if not stable_ids or not vsm:
            self.logger.warning(f"⚠️ [HYDRATION] No IDs or VSM available → Using partial content")
            boosted = [_standardize_chunk(c.copy(), 0.9) for c in chunks_to_hydrate]
            return self._guarantee_text_key(boosted)

        # 4. 🛰️ Fetch Full Documents from VSM
        stable_id_map = defaultdict(list)
        try:
            retrieved_docs = vsm.get_documents_by_id(
                list(stable_ids), doc_type=self.doc_type, enabler=self.config.enabler
            )
            self.logger.info(f"🛰️ [HYDRATION] Retrieved {len(retrieved_docs)} full docs from VSM")

            for doc in retrieved_docs:
                sid = doc.metadata.get("stable_doc_uuid") or doc.metadata.get("doc_id")
                if sid:
                    stable_id_map[sid].append({"text": doc.page_content, "metadata": doc.metadata})
        except Exception as e:
            self.logger.error(f"❌ [HYDRATION] VSM Fetch Error: {e}")
            fallback = [_standardize_chunk(c.copy(), 0.9) for c in chunks_to_hydrate]
            return self._guarantee_text_key(fallback)

        # 5. 💧 Hydrate & Deduplicate
        hydrated_priority_docs = []
        restored_count = 0
        seen_signatures = set()
        SAFE_META_KEYS = {"source", "file_name", "page", "page_label", "page_number", 
                          "enabler", "tenant", "year", "sub_topic"}

        for chunk in chunks_to_hydrate:
            new_chunk = chunk.copy()
            sid = new_chunk.get("stable_doc_uuid") or new_chunk.get("doc_id")

            hydrated = False
            if sid and sid in stable_id_map:
                # ใช้เนื้อหาตัวเต็มจาก VSM แทน Snippet เดิม
                best_match = stable_id_map[sid][0]
                new_chunk["text"] = best_match["text"]
                # อัปเดต Metadata ที่สำคัญ
                meta = best_match.get("metadata", {})
                new_chunk.update({k: v for k, v in meta.items() if k in SAFE_META_KEYS})
                hydrated = True
                restored_count += 1

            new_chunk = _standardize_chunk(new_chunk, score=1.0 if hydrated else 0.85)

            # ตรวจสอบการซ้ำซ้อน (Dedup)
            signature = (sid, new_chunk.get("chunk_uuid"), new_chunk.get("text", "")[:200])
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            hydrated_priority_docs.append(new_chunk)

        self.logger.info(
            f"✅ [HYDRATION SUMMARY] Restored {restored_count}/{len(chunks_to_hydrate)} chunks | "
            f"Unique: {len(hydrated_priority_docs)}"
        )

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
    
    def _get_pdca_blocks_from_evidences(
        self,
        evidences: List[Dict],
        baseline_evidences: Dict[str, List[Dict]],
        level: int,
        sub_id: str,
        contextual_rules_map: Dict[str, Any],
        record_id: str = None
    ) -> Tuple[str, str, str, str, str]:

        # 1. 🧹 Data Integration (คงเดิม)
        all_chunks = []
        for c in (evidences or []):
            if isinstance(c, dict) and c.get("text", "").strip():
                chunk = c.copy()
                chunk["is_baseline"] = False
                all_chunks.append(chunk)

        # ปรับการดึง Baseline ให้ยืดหยุ่นขึ้น (รองรับทั้ง Key ที่เป็น Level ตรงๆ หรือ ID เต็ม)
        target_baseline = baseline_evidences.get(str(level)) or baseline_evidences.get(f"{sub_id}.L{level}") or []
        for b in target_baseline:
            if isinstance(b, dict) and b.get("text", "").strip():
                b_copy = b.copy()
                b_copy["is_baseline"] = True
                all_chunks.append(b_copy)

        if not all_chunks:
            return "", "", "", "", ""

        # 2. 🏷️ Classification & Smart Sorting
        pdca_groups = defaultdict(list)
        for chunk in all_chunks:
            tag = classify_by_keyword(
                text=chunk["text"], sub_id=sub_id, level=level,
                contextual_rules_map=contextual_rules_map,
                chunk_metadata=chunk.get('metadata') # ส่ง Meta ไปช่วยจัดประเภท
            )
            final_tag = tag if tag in {"P", "D", "C", "A"} else "Other"
            
            # L1 Logic: เน้นสร้างโครงสร้างพื้นฐาน
            if level == 1 and final_tag == "Other":
                final_tag = "P"
            
            # [NEW] คำนวณ Priority Score เพื่อใช้ในการ Sorting
            # ให้ความสำคัญกับ: 1. คะแนน Relevancy ใหม่  2. ไฟล์ประเภทหลักฐานจริง
            rel_score = float(chunk.get("relevance_score_custom") or 0.0)
            rerank_score = float(chunk.get("rerank_score") or chunk.get("score") or 0.0)
            chunk["priority_score"] = (rel_score * 0.7) + (rerank_score * 0.3)

            label = {"P":"Plan","D":"Do","C":"Check","A":"Act"}.get(final_tag, "Other")
            pdca_groups[label].append(chunk)

        # 3. 🎭 Diverse Block Generator (Enhanced)
        def _create_block(tag: str, chunks: List[Dict]) -> str:
            if not chunks: return ""
            
            # เรียงตาม priority_score ที่รวมเรื่อง Source Grading ไว้แล้ว
            sorted_chunks = sorted(chunks, key=lambda x: x.get("priority_score", 0), reverse=True)
            
            diverse_list = []
            file_counts = {}
            for doc in sorted_chunks:
                # สกัดชื่อไฟล์เพื่อทำ Diversity Check
                meta = doc.get('metadata') or {}
                raw_source = doc.get("source_filename") or meta.get("source_filename") or meta.get("source") or "Unknown"
                fname = os.path.basename(str(raw_source))
                
                # จำกัด Chunks ต่อไฟล์ เพื่อให้ AI เห็นหลักฐานจากหลายแหล่ง
                if file_counts.get(fname, 0) < MAX_CHUNKS_PER_FILE:
                    diverse_list.append(doc)
                    file_counts[fname] = file_counts.get(fname, 0) + 1
                
                if len(diverse_list) >= MAX_CHUNKS_PER_BLOCK: # จำกัดไม่ให้ Block ยาวเกินไป (LLM Context Optimization)
                    break

            # Build Formatted String
            parts = []
            for i, c in enumerate(diverse_list, start=1):
                meta = c.get('metadata') or {}
                fname = os.path.basename(str(c.get("source_filename") or meta.get("source") or "Unknown"))
                pnum = c.get("page_label") or meta.get("page_label") or "N/A"
                
                baseline_tag = " [📜 ข้อมูลอ้างอิงระดับเดิม]" if c.get("is_baseline") else ""
                parts.append(
                    f"### [{tag} Evidence {i}/{len(diverse_list)}]{baseline_tag}\n"
                    f"{c['text'].strip()}\n"
                    f"[อ้างอิง: {fname}, หน้า: {pnum}, Relevancy: {c.get('priority_score', 0):.4f}]"
                )
            
            return "\n\n---\n\n".join(parts)

        # 4. Construct Final Outputs
        p_text = _create_block("Plan",  pdca_groups["Plan"])
        d_text = _create_block("Do",    pdca_groups["Do"])
        c_text = _create_block("Check", pdca_groups["Check"])
        a_text = _create_block("Act",   pdca_groups["Act"])
        o_text = _create_block("Other", pdca_groups["Other"])

        # Logging สรุปสถานะ
        status = " | ".join([f"{k}:{'✅' if v else '❌'}" for k, v in 
                            {"P":p_text, "D":d_text, "C":c_text, "A":a_text}.items()])
        self.logger.info(f"📊 [PDCA Blocks Ready] {sub_id} L{level} -> {status}")

        return p_text, d_text, c_text, a_text, o_text
    
    # ----------------------------------------------------------------------
    # 🚀 CORE WORKER: Assessment Execution (REVISED v2026.3)
    # ----------------------------------------------------------------------
    def _run_sub_criteria_assessment_worker(
        self,
        sub_criteria: Dict[str, Any],
        vectorstore_manager: Optional['VectorStoreManager'] = None
    ) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
        """
        [ADVANCED AUDITOR MODE v2026.5]
        - ประเมินครบ L1-L5 เพื่อทำ Gap Analysis และเก็บ Evidence Roadmap
        - Sequential Integrity: กรองคะแนนตามความต่อเนื่องของ SE-AM
        - Gap Type Classification: แยกแยะจุดตกจริง (Primary) กับจุดติด Sequential (Gap)
        """

        # 1. INITIALIZATION
        MAX_RETRY_ATTEMPTS = 2
        sub_id = sub_criteria['sub_id']
        sub_criteria_name = sub_criteria['sub_criteria_name']
        sub_weight = sub_criteria.get('weight', 0)
        
        current_enabler = getattr(self.config, 'enabler', 'KM')
        vsm = vectorstore_manager or getattr(self, 'vectorstore_manager', None)
        
        current_sequential_pass_level = 0 
        first_failure_level = None # ตัวแปรชี้จุดที่ "ตกจริง" ครั้งแรก
        raw_results_for_sub_seq: List[Dict[str, Any]] = []
        start_ts = time.time() 

        self.logger.info(f"🧵 [WORKER START] Full Gap-Analysis Mode | {sub_id}")
        
        # ดึงกฎที่ฉีดไว้ใน Class
        all_rules_for_sub = getattr(self, 'contextual_rules_map', {}).get(sub_id, {})

        # -----------------------------------------------------------
        # 2. EVALUATION LOOP (L1 → Target Level)
        # -----------------------------------------------------------
        for statement_data in sub_criteria.get('levels', []):
            level = statement_data.get('level')
            if level is None or level > self.config.target_level:
                continue
            
            # --- [PREPARATION] ---
            level_key = f"L{level}"
            specific_rules = all_rules_for_sub.get(level_key, {})
            
            # ฉีด Keywords และ Rules
            extra_keys = set(specific_rules.get('plan_keywords', []) + specific_rules.get('do_keywords', []))
            focus_hint = specific_rules.get('specific_contextual_rule', "")
            
            enhanced_statement = statement_data.get('statement', '')
            if extra_keys:
                enhanced_statement += f" (Keywords: {', '.join(extra_keys)})"
            
            # 🎯 [BASELINE SYNC]
            past_summaries = [f"L{p['level']}: {'PASS' if p.get('raw_is_passed') else 'FAIL'} - {p.get('reason', '')[:100]}..." for p in raw_results_for_sub_seq]
            baseline_context = "\n".join(past_summaries) if past_summaries else "ยังไม่มีข้อมูลระดับก่อนหน้า"

            # ฟังก์ชันประเมิน
            def assessment_fn(attempt):
                return self._run_single_assessment(
                    sub_criteria=sub_criteria,
                    statement_data={**statement_data, 'statement': enhanced_statement, 'focus_hint': focus_hint},
                    vectorstore_manager=vsm,
                    attempt=attempt,
                    doc_type=self.doc_type,      
                    top_k=INITIAL_TOP_K,          
                    baseline_summary=baseline_context,
                    **specific_rules 
                )

            # --- [EXECUTION] ---
            level_result = {}
            for attempt_num in range(1, MAX_RETRY_ATTEMPTS + 1):
                level_result = assessment_fn(attempt_num)
                if level_result.get('is_passed', False): 
                    break

            # --- [SEQUENTIAL & GAP LOGIC] ---
            # เก็บค่าผลตรวจดิบจาก LLM ไว้ก่อน
            is_passed_llm = level_result.get('is_passed', False)
            level_result['raw_is_passed'] = is_passed_llm # จำไว้ว่าจริงๆ แล้วเนื้อหาผ่านไหม

            if not is_passed_llm and first_failure_level is None:
                # จุดตกจริงจุดแรก (Primary Gap)
                first_failure_level = level
                level_result["display_status"] = "FAILED"
                level_result["gap_type"] = "PRIMARY_GAP"
            
            elif is_passed_llm and first_failure_level is not None:
                # ผ่านเนื้อหาแต่ "ติด Sequential" (Capped)
                level_result["display_status"] = "PASSED (CAPPED)"
                level_result["gap_type"] = "SEQUENTIAL_GAP"
                level_result["is_passed"] = False # ในทางคะแนนไม่ให้นับผ่าน
            
            elif not is_passed_llm and first_failure_level is not None:
                # ตกซ้ำซ้อน
                level_result["display_status"] = "FAILED (GAP)"
                level_result["gap_type"] = "COMPOUND_GAP"
            else:
                # ผ่านแบบต่อเนื่องปกติ
                current_sequential_pass_level = level
                level_result["display_status"] = "PASSED"
                level_result["gap_type"] = "NONE"

            # บันทึกหลักฐาน (Evidence) ทุกลำดับ
            highest_rerank = level_result.get('max_relevant_score', 0.0)
            self._save_level_evidences_and_calculate_strength(
                level_temp_map=level_result.get("temp_map_for_level", []),
                sub_id=sub_id, level=level, llm_result=level_result, highest_rerank_score=highest_rerank 
            )

            raw_results_for_sub_seq.append(level_result)
            self.logger.info(f"✅ L{level} Done | Status: {level_result['display_status']} | Gap: {level_result['gap_type']}")

        # -----------------------------------------------------------
        # 3. FINAL SYNTHESIS
        # -----------------------------------------------------------
        # สร้าง Action Plan โดยใช้ข้อมูล Gap Type ไปช่วยเขียนคำแนะนำ
        action_plan_result = self._generate_action_plan_safe(
            sub_id, sub_criteria_name, current_enabler, raw_results_for_sub_seq
        )

        # แปลงเป็นชื่อไฟล์สำหรับ UI
        for r in raw_results_for_sub_seq:
            if "temp_map_for_level" in r:
                r["temp_map_for_level"] = self._resolve_evidence_filenames(r["temp_map_for_level"])

        # คะแนนถ่วงน้ำหนักตาม Level สูงสุดที่ผ่านต่อเนื่องจริง
        weighted_score = round(self._calculate_weighted_score(current_sequential_pass_level, sub_weight), 2)
        final_temp_map = {k: v for k, v in self.evidence_map.items() if k.startswith(f"{sub_id}.")}

        return {
            "sub_criteria_id": sub_id,
            "sub_criteria_name": sub_criteria_name,
            "highest_full_level": current_sequential_pass_level,
            "weight": sub_weight,
            "weighted_score": weighted_score,
            "display_status": "PASSED" if current_sequential_pass_level >= self.config.target_level else "FAILED",
            "action_plan": action_plan_result, 
            "raw_results_ref": raw_results_for_sub_seq,
            "worker_duration_s": round(time.time() - start_ts, 2)
        }, final_temp_map

    def _generate_action_plan_safe(
        self, 
        sub_id: str, 
        name: str, 
        enabler: str, 
        results: List[Dict]
    ) -> Any:
        """
        Wrapper เพื่อเรียกใช้ระบบ Action Plan อย่างปลอดภัย
        """
        try:
            # 1. นำเข้าฟังก์ชันหลัก (Import ภายในเพื่อลด Circular Dependency)
            # 2. กรองข้อมูลเฉพาะเกณฑ์ที่ต้องการคำแนะนำ (FAILED, WEAK_EVIDENCE, PDCA_INCOMPLETE)
            # กฎ: ถ้าผ่านระดับ target แล้ว และ strength > 3.0 อาจจะไม่ต้องสร้าง Action Plan
            to_recommend = []
            for r in results:
                is_passed = r.get('is_passed', False)
                strength = r.get('evidence_strength', 10.0)
                
                if not is_passed or strength < 3.0:
                    to_recommend.append(r)

            if not to_recommend:
                return {"status": "EXCELLENT", "message": "ไม่ต้องมีแผนปรับปรุง เนื่องจากผ่านเกณฑ์ในระดับดีมาก"}

            # 3. เรียกใช้ฟังก์ชันหลัก (ตัวที่คุณส่งมาในคำถาม)
            return create_structured_action_plan(
                recommendation_statements=to_recommend,
                sub_id=sub_id,
                sub_criteria_name=name,
                enabler=enabler,
                target_level=self.config.target_level,
                llm_executor=self.llm,  # ส่งตัว LLM ไปรัน Prompt
                logger=self.logger
            )

        except Exception as e:
            self.logger.error(f"⚠️ Action Plan Generation Failed: {str(e)}")
            return {
                "status": "ERROR",
                "message": "เกิดข้อผิดพลาดในการสร้างแผนงานอัตโนมัติ",
                "error_detail": str(e)
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
        [ULTIMATE ASSEMBLY v2026.3] — Server L40 & Mac Optimized
        ------------------------------------------------------
        ทำหน้าที่เป็นตัวควบคุม (Orchestrator) ในการกระจายงานประเมิน
        """
        start_ts = time.time()
        self.current_record_id = record_id 

        # 1. 📋 เตรียมเกณฑ์ (Rubric Filtering)
        all_statements = self._flatten_rubric_to_statements()
        is_all = str(target_sub_id).lower() == "all"
        sub_criteria_list = all_statements if is_all else [
            s for s in all_statements if str(s.get('sub_id')).lower() == str(target_sub_id).lower()
        ]

        if not sub_criteria_list:
            return self._create_failed_result(record_id, f"Criteria '{target_sub_id}' not found", start_ts)

        # 2. ⚙️ การตั้งค่า Parallel / Sequential (L40 vs Mac Detection)
        # ดึงจำนวน Worker จาก .env (บน L40 คือ 4, บน Mac ของคุณคือ 1 ตาม Log)
        env_workers = os.environ.get('NUM_WORKERS') or os.environ.get('MAX_PARALLEL_WORKERS')
        num_workers = 1 if (sequential or not is_all) else int(env_workers or 1)
        
        run_parallel = (num_workers > 1)

        # ตรวจสอบ Hardware เพื่อพ่น Log ที่ถูกต้อง
        import torch
        device_info = "NVIDIA CUDA (L40)" if torch.cuda.is_available() else "Apple Silicon (MPS)" if torch.backends.mps.is_available() else "CPU"

        self.logger.info(f"🎯 Target: {target_sub_id} | Mode: {'Parallel' if run_parallel else 'Sequential'}")
        self.logger.info(f"🚀 Device Config: {device_info} | Active Workers: {num_workers}")

        self.raw_llm_results = []
        self.final_subcriteria_results = []
        self.evidence_map = {}

        # 3. 🔥 Execution Phase
        if run_parallel:
            # 📌 การรันขนาน (สำหรับ L40)
            worker_args = [self._prepare_worker_tuple(s, document_map) for s in sub_criteria_list]
            try:
                # ใช้ 'spawn' เพื่อล้าง CUDA memory context ทุกครั้งที่สร้าง process ใหม่
                ctx = multiprocessing.get_context('spawn')
                with ctx.Pool(processes=num_workers) as pool:
                    results_list = pool.map(_static_worker_process, worker_args)
                
                for res in results_list:
                    if isinstance(res, tuple) and len(res) == 2:
                        sub_result, temp_map = res
                        self._merge_worker_results(sub_result, temp_map)
            except Exception as e:
                self.logger.critical(f"❌ Parallel execution failed: {e}")
                raise
        else:
            # 🧵 การรันทีละขั้นตอน (สำหรับ Mac/Specific Sub-ID)
            vsm = vectorstore_manager or self._init_local_vsm()
            for sub_criteria in sub_criteria_list:
                # เรียก Assessment Core (_run_single_assessment ภายใน)
                sub_result, final_temp_map = self._run_sub_criteria_assessment_worker(sub_criteria, vsm)
                self._merge_worker_results(sub_result, final_temp_map)

        # 4. 🏁 สรุปผลและส่งออก (Calculate & Export)
        self._calculate_overall_stats(target_sub_id)
        
        final_results = {
            "record_id": record_id,
            "summary": self.total_stats,
            "sub_criteria_results": self.final_subcriteria_results,
            "run_time_seconds": round(time.time() - start_ts, 2),
            "timestamp": datetime.now().isoformat(),
            "device_used": device_info
        }

        if export:
            self._export_results(results=final_results, sub_criteria_id=target_sub_id, record_id=record_id)

        return final_results

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
    
    # ฟังก์ชันนี้เปรียบเสมือน "ศาลอุทธรณ์" หาก AI ตรวจรอบแรกแล้วให้ตก 
    # แต่ระบบพบว่ามีหลักฐานคุณภาพสูงอยู่ ระบบจะบังคับให้ LLM ตรวจซ้ำด้วยมุมมองที่กว้างขึ้น (Diversity Focus)
    def _run_expert_re_evaluation(
        self,
        sub_id: str,
        level: int,
        statement_text: str,
        context: str,
        first_attempt_reason: str,
        missing_tags: Set[str],
        highest_rerank_score: float,
        sub_criteria_name: str,
        llm_evaluator_to_use: Any,
        base_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        [EXPERT LOOP - DIVERSITY FORCED] บังคับให้ LLM พิจารณาหลักฐานที่หลากหลายขึ้น
        - ใช้เมื่อผลประเมินรอบแรกตกแต่คะแนน Rerank สูง
        - แทรกคำสั่งพิเศษ (Expert Hint) เพื่อลดอคติของโมเดล
        """
        self.logger.info(f"🕵️ [EXPERT RE-EVAL] Starting second pass for {sub_id} L{level} (Rerank: {highest_rerank_score:.4f})")

        # 🟢 สร้าง Expert Hint เพื่อกระตุ้นการวิเคราะห์เชิงลึก
        missing_info = f"โดยเฉพาะมิติ {', '.join(missing_tags)}" if missing_tags else "ในภาพรวม"
        
        hint_msg = f"""
--- 🚨 EXPERT AUDITOR OVERRIDE (รอบพิจารณาพิเศษ) ---
ผลการตรวจรอบแรก: ไม่ผ่าน (เหตุผล: {first_attempt_reason})

ข้อตรวจพบเพิ่มเติมจากระบบ Search:
- ระบบพบหลักฐานที่มีความเกี่ยวข้องสูง (Score: {highest_rerank_score:.4f}) แต่คุณอาจมองข้ามไป
- คำสั่งพิเศษ: โปรดพิจารณาไฟล์อื่นๆ ใน Context นอกเหนือจากไฟล์หลักที่อ่านรอบแรก
- หากพบ "ร่องรอย" (Implicit Evidence) ของกระบวนการ {missing_info} แม้ Keyword จะไม่ตรง 100% แต่ถ้ามีการปฏิบัติจริง ให้ใช้ดุลยพินิจ (Professional Discretion) ในการให้คะแนน
- พิจารณาความต่อเนื่องจากระดับก่อนหน้า (Baseline) หากพื้นฐานดีและมีหลักฐานใหม่เสริม ให้คะแนนอย่างเป็นธรรม
"""

        # คัดลอกพารามิเตอร์เดิมและอัปเดต Context ใหม่
        expert_kwargs = base_kwargs.copy()
        expert_kwargs["context"] = f"{context}\n\n{hint_msg}"
        expert_kwargs["sub_criteria_name"] = f"{sub_criteria_name} (Expert Re-assessment Mode)"
        
        # เพิ่มระดับความมั่นใจจำลองเพื่อให้ LLM กล้าตัดสินใจมากขึ้น
        expert_kwargs["ai_confidence"] = "High (Expert Override)"

        try:
            # รันการประเมินซ้ำด้วย Evaluator ตัวเดิม (แต่ Prompt มี Hint ใหม่)
            re_eval_result = llm_evaluator_to_use(**expert_kwargs)
            
            # บันทึกสถานะว่าผ่านการทำ Expert Re-eval
            re_eval_result["is_expert_evaluated"] = True
            re_eval_result["original_fail_reason"] = first_attempt_reason
            
            if re_eval_result.get("is_passed"):
                self.logger.info(f"✨ [EXPERT SUCCESS] {sub_id} L{level} has been reversed to PASSED")
            else:
                self.logger.info(f"❌ [EXPERT CONFIRMED] {sub_id} L{level} is still FAILED after re-evaluation")
                
            return re_eval_result
            
        except Exception as e:
            self.logger.error(f"❌ Expert Re-evaluation crashed: {e}")
            # กรณีพัง ให้คืนค่าผลการตรวจรอบแรก
            return {"is_passed": False, "reason": f"Expert Loop Error: {str(e)}", "score": 0.0}
    
    def _apply_diversity_filter(self, evidences: List[Dict], level: int) -> List[Dict]:
        """
        กรองหลักฐานให้มีความหลากหลาย (Diversity) และไม่เยอะจนเกินไป (Context Window Management)
        - Level 1-2: เน้นความแม่นยำ (Top 15 chunks)
        - Level 3 ขึ้นไป: จำกัด chunks ต่อไฟล์เพื่อลด Bias
        """
        if not evidences:
            return []

        # เรียงลำดับตาม Score ล่าสุด (Rerank score)
        sorted_evidences = sorted(evidences, key=lambda x: x.get('rerank_score', 0), reverse=True)

        if level <= 2:
            # เลเวลพื้นฐาน เอา Top 15 ที่ดีที่สุด
            return sorted_evidences[:15]
        
        # เลเวลสูง: จำกัดให้ไม่เกิน 3-4 chunks ต่อหนึ่งชื่อไฟล์ เพื่อให้ AI เห็นไฟล์ที่หลากหลาย
        diverse_results = []
        file_counts = {}
        per_file_limit = 4

        for ev in sorted_evidences:
            source = ev.get('metadata', {}).get('source_filename', 'Unknown')
            file_counts[source] = file_counts.get(source, 0) + 1
            
            if file_counts[source] <= per_file_limit:
                diverse_results.append(ev)
            
            # ป้องกัน Context ยาวเกินไป (จำกัดรวมไม่เกิน 20-25 chunks)
            if len(diverse_results) >= 20:
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
        """
        [REVISED v2026.4.8] 
        - เพิ่ม Source Grading: ให้คะแนนไฟล์ 'บันทึก/คำสั่ง/มติ' สูงกว่า 'รายงานประเมิน'
        - ปรับ Keyword Saturation: เน้นคุณภาพการ Match มากกว่าปริมาณ
        - ปรับปรุง Neighbor Context Handling ให้เสถียรขึ้น
        """
        if not evidence:
            return 0.0

        # 1. Rerank Score (Weight 50%)
        rerank_score = evidence.get('rerank_score', evidence.get('score', 0.0))
        normalized_rerank = min(max(rerank_score, 0.0), 1.0)

        # 2. เตรียมข้อมูลพื้นฐาน
        text = (evidence.get('text', '') or evidence.get('page_content', '')).lower()
        meta = evidence.get('metadata', {})
        filename = (meta.get('source', '') or meta.get('source_filename', '') or '').lower()

        # 3. ดึง Rules ประจำ Level
        cum_rules = self.get_cumulative_rules(sub_id, level)
        
        # 4. Source Grading Logic (แก้ปัญหาดึง Assessment Report)
        source_bonus = 0.0
        # รายชื่อคำที่บ่งบอกว่าเป็นหลักฐานชั้นต้น (High Priority)
        primary_evidence_patterns = ["มติ", "บันทึก", "คำสั่ง", "ประกาศ", "นโยบาย", "แผนแม่บท"]
        # รายชื่อคำที่บ่งบอกว่าเป็นเอกสารสรุป/รายงานประเมิน (Lower Priority ในขั้นตอนนี้)
        secondary_report_patterns = ["assessment report", "รายงานการประเมิน", "สรุปผล"]

        if any(p in filename for p in primary_evidence_patterns):
            source_bonus += 0.20  # Boost ให้หลักฐานตัวจริง
        if any(p in filename for p in secondary_report_patterns):
            source_bonus -= 0.15  # Penalty ให้ไฟล์รายงานเพื่อไม่ให้เบียดไฟล์จริง

        # 5. Keyword Score (Weight 50%) - Saturation 
        target_kws = set()
        if level <= 2:
            target_kws.update(cum_rules.get('plan_keywords', []) + cum_rules.get('do_keywords', []))
        else:
            target_kws.update(cum_rules.get('check_keywords', []) + cum_rules.get('act_keywords', []))
        
        # เน้นเฉพาะคำที่ Match จริงๆ
        match_count = sum(1 for kw in target_kws if kw.lower() in text)
        # Saturation: เจอ 2-3 คำที่สำคัญมากพอแล้ว (สำหรับเอกสารราชการที่ข้อความไม่ยาวมาก)
        keyword_score = min(match_count / 2.5, 1.0) 

        # 6. Act-Hook Bonus (Neighbor Boost)
        neighbor_bonus = 0.0
        is_neighbor = evidence.get('is_neighbor', False) or meta.get('is_neighbor', False)
        if is_neighbor:
            # เพิ่มแต้มเพื่อให้หน้าบริบทไม่ถูกกรองทิ้ง (Threshold มักอยู่ที่ 0.35)
            neighbor_bonus += 0.25 

        # 7. Specific Context Phrase Match
        specific_rule = cum_rules.get('specific_contextual_rule', '').lower()
        rule_bonus = 0.25 if specific_rule and specific_rule in text else 0.0

        # 8. รวมคะแนน (50/50 Ratio + Bonuses)
        # เราใช้ 0.5/0.5 เพื่อให้ Keyword Match มีผลเท่ากับ Vector Search ในการตัดสินความเกี่ยวข้อง
        final_score = (0.5 * normalized_rerank) + (0.5 * keyword_score) + source_bonus + neighbor_bonus + rule_bonus
        final_score = min(max(final_score, 0.0), 1.0)

        self.logger.debug(f"[{sub_id} L{level}] RelScore: {final_score:.4f} | Rerank: {normalized_rerank:.4f} | KW: {keyword_score:.4f} | Src: {source_bonus}")
        return final_score

    def enhance_query_for_statement(
        self,
        statement_text: str,
        sub_id: str,
        statement_id: str,
        level: int,
        focus_hint: str = ""
    ) -> List[str]:
        """
        [ULTIMATE QUERY ENHANCER v2026.6 - Enhanced for all PDCA phases]
        - เพิ่ม keyword และ query สำหรับ Check & Act
        - ขยาย synonym ภาษาไทยให้ครอบคลุมเอกสารรัฐวิสาหกิจ
        - เน้น Do สำหรับ L1-L2, Check & Act สำหรับ L3+
        - จำกัดสูงสุด 8 queries
        """
        logger = logging.getLogger(__name__)

        # 1. Anchor พื้นฐาน
        enabler_id = str(self.enabler_id).upper()  # e.g., "KM", "CG"
        id_anchor = f"{enabler_id} {sub_id}"      # e.g., "KM 1.1"

        # 2. ดึง cumulative rules
        cum_rules = self.get_cumulative_rules(sub_id, level)

        # 3. รวบรวม keywords จาก rules
        plan_kws = cum_rules.get('plan_keywords', [])
        do_kws = cum_rules.get('do_keywords', [])
        check_kws = cum_rules.get('check_keywords', [])
        act_kws = cum_rules.get('act_keywords', [])

        # 4. ขยาย Synonym ภาษาไทย (เพิ่ม Check & Act)
        do_synonyms = [
            "มติบอร์ด", "มติคณะกรรมการ", "มติที่ประชุม", "อนุมัติ", "เห็นชอบ",
            "รับทราบ", "พิจารณาแล้ว", "ประกาศใช้", "ลงมติ", "มีมติ",
            "คำสั่งแต่งตั้ง", "อนุมัติหลักการ", "ขออนุมัติ", "ผ่านการอนุมัติ",
            "มติบริหาร", "มติ กฟภ", "มติคณะทำงาน"
        ]

        check_synonyms = [
            "รายงานผล", "ผลการดำเนินงาน", "ตัวชี้วัด", "KPI", "ประเมินผล",
            "ติดตามผล", "ตรวจสอบ", "วัดผล", "สรุปผล", "ผลลัพธ์",
            "ความก้าวหน้า", "เปรียบเทียบผล", "รายงานไตรมาส"
        ]

        act_synonyms = [
            "ปรับปรุง", "พัฒนาต่อเนื่อง", "ถอดบทเรียน", "บทเรียนที่ได้",
            "แก้ไข", "ป้องกัน", "นวัตกรรม", "ปรับเปลี่ยน", "ยกระดับ",
            "แผนปฏิบัติการใหม่", "ข้อเสนอแนะ", "มาตรการแก้ไข"
        ]

        # รวม keywords ทั้งหมด (จำกัดจำนวนเพื่อไม่ให้ query ยาวเกิน)
        all_kws = list(set(
            plan_kws + do_kws + check_kws + act_kws +
            do_synonyms + check_synonyms + act_synonyms
        ))
        keywords_str = " ".join(all_kws[:15])  # เพิ่มจาก 12 เป็น 15

        queries = []

        # Query 1: Direct + Anchor (ความแม่นยำสูงสุด)
        queries.append(f"{id_anchor} {statement_text}")

        # Query 2: Maturity + Keywords
        if level <= 2:
            queries.append(f"{id_anchor} นโยบาย วิสัยทัศน์ มติบอร์ด อนุมัติ คำสั่งแต่งตั้ง {keywords_str}")
        else:
            queries.append(f"{id_anchor} รายงานผล ติดตามผล ปรับปรุง ถอดบทเรียน {keywords_str}")

        # Query 3: Specific Rule (ถ้ามี)
        specific_rule = cum_rules.get('specific_contextual_rule', '')
        if specific_rule:
            queries.append(f"{id_anchor} {specific_rule[:80]}")  # ตัดสั้นลงอีกนิด

        # Query 4: Do-Focused (L1-L2)
        if level <= 2:
            do_focus = f"มติบอร์ด อนุมัติ เห็นชอบ รับทราบ คำสั่งแต่งตั้ง {sub_id} {keywords_str}"
            queries.append(do_focus)

        # Query 5: Check-Focused (L3+)
        if level >= 3:
            check_focus = f"รายงานผล ตัวชี้วัด KPI สรุปผล ติดตามผล {id_anchor} {keywords_str}"
            queries.append(check_focus)

        # Query 6: Act-Focused (L4+)
        if level >= 4:
            act_focus = f"ปรับปรุง ถอดบทเรียน แก้ไข นวัตกรรม {id_anchor} {keywords_str}"
            queries.append(act_focus)

        # Query 7: Broad/Global (ไม่ติด anchor เต็มรูปแบบ)
        broad_query = f"{enabler_id} {statement_text} {keywords_str}"
        queries.append(broad_query)

        # Query 8: Tenant + Enabler Context
        tenant_context = f"{self.config.tenant} {enabler_id} {statement_text}"
        queries.append(tenant_context)

        # Final Cleaning & Deduplication
        final_queries = []
        seen = set()
        for q in queries:
            q_strip = q.strip()
            if q_strip and q_strip not in seen and len(q_strip) > 5:
                # ตัดให้เหมาะสมกับ embedding model (30 คำ)
                clean_q = " ".join(q_strip.split()[:30])
                final_queries.append(clean_q)
                seen.add(clean_q)

        # จำกัดสูงสุด 8
        final_queries = final_queries[:8]

        logger.debug(f"Generated {len(final_queries)} enhanced queries for {id_anchor} L{level}")
        return final_queries
    
    def _get_baseline_summary_text(self, sub_id: str, level: int) -> str:
        """ สรุปผลประเมินเลเวลก่อนหน้าให้ LLM (Fix AttributeError) """
        if level <= 1:
            return "ระดับเริ่มต้น (Starting Level): ไม่มีการประเมินก่อนหน้า"
        
        prev_res = self.results.get(f"{sub_id}.L{level-1}", {})
        if prev_res:
            status = "ผ่าน ✅" if prev_res.get('is_passed') else "ไม่ผ่าน ❌"
            return f"ผลประเมิน L{level-1}: {status} | เหตุผล: {prev_res.get('reason', 'N/A')[:200]}..."
        
        return f"ไม่พบข้อมูล L{level-1}"
        
    def _run_single_assessment(
        self,
        sub_criteria: Dict[str, Any],
        statement_data: Dict[str, Any],
        vectorstore_manager: Optional['VectorStoreManager'],
        sequential_chunk_uuids: Optional[List[str]] = None,
        record_id: str = None,
        attempt: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        [ULTIMATE PRODUCTION v2026.4.5] - Precision & Data-Driven Logic
        - ใช้ top_k=INITIAL_TOP_K เพื่อรักษา Precision
        - ใช้ Dynamic Query Expansion จาก Rules แทนการ Hardcode ชื่อไฟล์
        - ระบบ Relevance Filter + PDCA Tagging ก่อนส่ง LLM
        """
        start_time = time.time()
        sub_id = sub_criteria['sub_id']
        level = statement_data['level']
        statement_text = statement_data['statement']
        sub_criteria_name = sub_criteria['sub_criteria_name']
        
        MIN_RETRY_SC = 0.70
        self.logger.info(f"🔍 [ASSESSMENT] {sub_id} L{level} | Attempt: {attempt} (Server Mode)")

        # 1. เตรียม Contextual Rules และ Level Constraints
        # ดึงกฎที่สอดคล้องกับ Rubrics และ Keywords ที่คุณตั้งไว้ใน JSON
        cum_rules = self.get_cumulative_rules(sub_id, level)
        level_constraint = self._get_level_constraint_prompt(sub_id, level)

        # 2. ADAPTIVE RAG LOOP + Dynamic Query Expansion
        mapped_stable_ids, priority_chunks = self._get_mapped_uuids_and_priority_chunks(
            sub_id=sub_id, level=level, statement_text=statement_text,
            level_constraint=level_constraint, vectorstore_manager=vectorstore_manager
        )

        # สร้าง Query List อัตโนมัติ: รวมคำค้นหาเดิม + คำสำคัญจาก Rules
        rag_query_list = [statement_text]
        important_kws = cum_rules.get('plan_keywords', []) + cum_rules.get('do_keywords', [])
        if important_kws:
            # ดึงคำสำคัญมาสร้าง Query เสริม (ไม่เกิน 3-5 คำเพื่อไม่ให้ Query ยาวเกินไป)
            rag_query_list.append(" OR ".join(important_kws[:3]))

        highest_rerank_score = -1.0
        final_top_evidences = []

        # Adaptive Loop เพื่อหาเอกสารที่ดีที่สุด (สูงสุด 4 รอบ)
        for loop_attempt in range(1, 5):
            # รอบแรกใช้ Full Query, รอบถัดๆ ไปใช้เป้าหมายของเกณฑ์เป็นตัวนำ
            query_input = rag_query_list if loop_attempt == 1 else [statement_text]
            
            retrieval_result = self.rag_retriever(
                query=query_input,
                doc_type=self.doc_type,
                sub_id=sub_id, level=level,
                vectorstore_manager=vectorstore_manager,
                stable_doc_ids=mapped_stable_ids,
                top_k=INITIAL_TOP_K  # รักษาค่าเดิมตามคำแนะนำของคุณ
            )
            
            current_evidences = retrieval_result.get("top_evidences", [])
            all_candidates = current_evidences + priority_chunks
            
            if not all_candidates:
                continue

            current_max = max(
                (ev.get('rerank_score', ev.get('score', 0)) for ev in all_candidates),
                default=0.0
            )

            if current_max >= highest_rerank_score:
                highest_rerank_score = current_max
                final_top_evidences = all_candidates

            # ถ้าได้เอกสารที่ Rerank score สูงพอแล้ว ให้หยุด Loop
            if highest_rerank_score >= MIN_RETRY_SC:
                break

        self.logger.info(f"Adaptive RAG completed | Highest score: {highest_rerank_score:.4f} | Raw evidences: {len(final_top_evidences)}")

        # 3. ACT-HOOK: CONTEXT EXPANSION (ดึงหน้าข้างเคียงมาเสริมบริบท)
        expanded_evidences = self._expand_context_with_neighbor_pages(
            top_evidences=final_top_evidences,
            collection_name=f"evidence_{self.enabler_id.lower()}"
        )

        # 4. ROBUST PDCA TAGGING (ติดป้ายกำกับประเภทหลักฐาน)
        for doc in expanded_evidences:
            doc["pdca_tag"] = classify_by_keyword(
                text=doc.get("text", ""),
                sub_id=sub_id, level=level,
                contextual_rules_map=self.contextual_rules_map,
                chunk_metadata=doc.get('metadata')
            )

        # 5. DIVERSITY FILTER (กรองไฟล์ซ้ำซ้อน)
        sorted_evidences = sorted(
            expanded_evidences,
            key=lambda x: x.get('rerank_score', 0),
            reverse=True
        )
        
        diverse_filtered = []
        file_counts = {}
        for doc in sorted_evidences:
            meta = doc.get('metadata', {})
            fname = os.path.basename(meta.get('source', 'unknown'))
            if file_counts.get(fname, 0) < 5:  # จำกัดจำนวน Chunks ต่อไฟล์
                diverse_filtered.append(doc)
                file_counts[fname] = file_counts.get(fname, 0) + 1
            if len(diverse_filtered) >= 30: # Max Context Chunks
                break

        self.logger.info(f"Diversity filter applied | Before relevance filter: {len(diverse_filtered)}")

        # 6. RELEVANCE FILTER (กรอง Noise โดยอ้างอิงจาก Rules ใน JSON)
        relevant_filtered = []
        # ใช้เกณฑ์ความเข้มงวดตาม Maturity Level
        rel_threshold = 0.35 if level <= 2 else 0.45 
        
        for doc in diverse_filtered:
            rel_score = self.relevance_score_fn(doc, sub_id, level) # ฟังก์ชันที่คุณต้องการ
            doc['relevance_score_custom'] = rel_score
            if rel_score >= rel_threshold:
                relevant_filtered.append(doc)

        # Fallback Mechanism: หากกรองแล้วไม่เหลือหลักฐานเลย ให้ใช้ Top 5 จาก Rerank
        if not relevant_filtered and diverse_filtered:
            relevant_filtered = sorted(diverse_filtered, key=lambda x: x.get('rerank_score', 0), reverse=True)[:5]
            self.logger.warning(f"Relevance filter too strict for {sub_id} L{level} → fallback to top 5")

        self.logger.info(f"After relevance filter: {len(relevant_filtered)} evidences sent to LLM")

        # 7. PDCA SYNTHESIS (เตรียมข้อมูลสำหรับ LLM)
        previous_evidence = self._collect_previous_level_evidences(sub_id, level) if level > 1 else {}
        formatted_baseline_evi = {k.split(".L")[-1]: v for k, v in previous_evidence.items()}
        
        plan_b, do_b, check_b, act_b, other_b = self._get_pdca_blocks_from_evidences(
            evidences=relevant_filtered,
            baseline_evidences=formatted_baseline_evi,
            level=level, sub_id=sub_id,
            contextual_rules_map=self.contextual_rules_map,
            record_id=record_id
        )

        # 8. EVALUATION WITH LLM
        processed_lc_docs = [LcDocument(page_content=d['text'], metadata=d.get('metadata', d)) for d in relevant_filtered]
        confidence_result = self.calculate_audit_confidence(processed_lc_docs)
        
        synthesized_context = (
            f"--- EVIDENCE BLOCKS SYNTHESIS ---\n"
            f"<Plan_Evidence>\n{plan_b or 'ไม่พบหลักฐานส่วนการวางแผน'}\n</Plan_Evidence>\n"
            f"<Do_Evidence>\n{do_b or 'ไม่พบหลักฐานส่วนการดำเนินงาน'}\n</Do_Evidence>\n"
            f"<Check_Evidence>\n{check_b or 'ไม่พบหลักฐานส่วนการตรวจสอบ'}\n</Check_Evidence>\n"
            f"<Act_Evidence>\n{act_b or 'ไม่พบหลักฐานส่วนการปรับปรุง'}\n</Act_Evidence>\n"
            f"<General_Context>\n{other_b}\n</General_Context>"
        )

        llm_result = evaluate_with_llm(
            context=synthesized_context,
            baseline_summary=self._get_baseline_summary_text(sub_id, level),
            sub_id=sub_id,
            level=level,
            statement_text=statement_text,
            sub_criteria_name=sub_criteria_name,
            ai_confidence=confidence_result["level"],
            confidence_reason=confidence_result.get("reason", "N/A"),
            max_rerank_score=highest_rerank_score,
            llm_executor=self.llm
        )

        # 9. POST-PROCESSING & SAVE
        llm_result = self.post_process_llm_result(llm_result, level)
        max_evi_str = self._save_level_evidences_and_calculate_strength(
            level_temp_map=relevant_filtered,
            sub_id=sub_id,
            level=level,
            llm_result=llm_result,
            highest_rerank_score=highest_rerank_score
        )

        self.logger.info(f"📊 [RESULT] {sub_id} L{level} -> Score: {llm_result.get('score', 0.0):.4f} | Pass: {llm_result.get('is_passed')}")

        return {
            "sub_criteria_id": sub_id,
            "level": level,
            "is_passed": llm_result.get('is_passed', False),
            "score": float(llm_result.get('score', 0.0)),
            "audit_confidence": confidence_result,
            "pdca_breakdown": {
                "P": llm_result.get("P_Plan_Score", 0),
                "D": llm_result.get("D_Do_Score", 0),
                "C": llm_result.get("C_Check_Score", 0),
                "A": llm_result.get("A_Act_Score", 0)
            },
            "reason": llm_result.get('reason', "วิเคราะห์ไม่สำเร็จ"),
            "max_relevant_score": highest_rerank_score,
            "evidence_strength": max_evi_str,
            "temp_map_for_level": relevant_filtered,
            "duration": round(time.time() - start_time, 2)
        }