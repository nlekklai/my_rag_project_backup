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


# -------------------- 5. LOGGER SETUP --------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


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
        RETRIEVAL_EARLY_EXIT_SCORE_THRESHOLD, RETRIEVAL_RELEVANCE_THRESHOLD,
        DEFAULT_TENANT
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
            ATOMIC_ACTION_PROMPT, SUB_ROADMAP_PROMPT,
            SYSTEM_ATOMIC_ACTION_PROMPT, SYSTEM_SUB_ROADMAP_PROMPT,
            STRATEGIC_OVERALL_PROMPT, SYSTEM_OVERALL_STRATEGIC_PROMPT
        )
    except ImportError:
        ATOMIC_ACTION_PROMPT = "Recommendation: {coaching_insight} Level: {level}"
        SUB_ROADMAP_PROMPT = "Roadmap: {aggregated_insights}"
        STRATEGIC_OVERALL_PROMPT = "Overall: {aggregated_context}"
        SYSTEM_ATOMIC_ACTION_PROMPT = "Assistant mode."
        SYSTEM_SUB_ROADMAP_PROMPT = "Strategy mode."

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
    SUB_ROADMAP_PROMPT = "Roadmap: {aggregated_insights}"
    SYSTEM_ATOMIC_ACTION_PROMPT = "Assistant"
    SYSTEM_SUB_ROADMAP_PROMPT = "Strategist"

# try:
#     from core.seam_prompts import SUB_ROADMAP_TEMPLATE
#     logger.info("[PROMPT] Loaded SUB_ROADMAP_TEMPLATE from core.seam_prompts (real version)")
# except ImportError as e:
#     logger.warning(f"[PROMPT] Import failed ({str(e)}), using fallback SUB_ROADMAP_TEMPLATE")
#     SUB_ROADMAP_TEMPLATE = """
# ### [Strategic Context]
# - หัวข้อ: {sub_criteria_name} ({sub_id}) | Enabler: {enabler}
# - ทิศทางเชิงกลยุทธ์: {strategic_focus}

# ### [Input Data: Assets & Gaps - ใช้เฉพาะข้อมูลนี้ ห้ามมโนเพิ่ม]
# {aggregated_insights}

# ---
# สร้าง Master Roadmap ตามกฎเข้มงวดข้างต้นอย่างเคร่งครัดที่สุด:
# - ทุก action ต้องเจาะจง + อ้างชื่อไฟล์จริง + หน้า/ส่วน (ถ้ามี) + verb ปฏิบัติได้ทันที
# - ห้ามใช้ verb ต้องห้ามเด็ดขาด (รวมใน goal และ overall_strategy)
# - หากผ่าน L5 และไม่มี gap ให้ Phase 1 = "Reinforce & Sustain" และ Phase 2 = Standardization / Automation / ขยายผลต้นแบบ
# - ห้ามมี Phase เดียวถ้าเป็น L5

# ตัวอย่าง action ที่ถูกต้องเท่านั้น (ใช้เป็นแนวทางเท่านั้น ไม่ใช่ copy ตรง ๆ):
# - "ประกาศใช้ KMS Policy ที่ผู้บริหารลงนามจากหน้า 12 ของไฟล์ KM6.1L301 KM_6_3_PEA_Assessment Report.pdf เป็นมาตรฐานองค์กร พร้อมกำหนดการสื่อสารไตรมาสละ 1 ครั้งผ่าน KM-Si"
# - "สถาปนา dashboard อัตโนมัติสำหรับติดตามผลการประเมิน KM จากโครงสร้างในหน้า 7 ของไฟล์ KM2.1L405 PEA KM Master Plan_...13Dec24_edit.pdf โดยบูรณาการเข้ากับระบบ KM-Survey"
# - "ขยายผลนโยบายเร่งด่วน 12 ด้านจากหน้า 48 ของไฟล์ KM1.2L301 แผนแม่บท ปรับปรุงครั้งที่ 4 ย่อ.pdf มาจัดทำโปรแกรมอบรมผู้บริหารทุกระดับเรื่องการขับเคลื่อน KM"

# {{
#   "status": "SUCCESS",
#   "overall_strategy": "ใช้ความสำเร็จจากไฟล์ A หน้า X มาสร้างระบบยั่งยืนและขยายผลข้ามหน่วยงาน (ต้องอ้างไฟล์จริงจาก input)",
#   "phases": [
#     {{
#       "phase": "Phase 1: Quick Win (Reinforce & Sustain หรือ Remediation)",
#       "goal": "เสริมความแข็งแกร่งหรือปิดช่องว่างโดยอ้างอิงหลักฐานจริง",
#       "key_actions": [
#         {{
#           "action": "ระบุ action เฉพาะเจาะจง + อ้างชื่อไฟล์ + หน้า/ส่วน",
#           "priority": "High"
#         }}
#       ]
#     }},
#     {{
#       "phase": "Phase 2: Level-Up Excellence",
#       "goal": "ยกระดับด้วย standardization, automation หรือขยายผลต้นแบบ",
#       "key_actions": [
#         {{
#           "action": "ระบุแผนงานเชิงสถาปัตยกรรม + อ้างไฟล์และส่วนที่เกี่ยวข้อง",
#           "priority": "Medium"
#         }}
#       ]
#     }}
#   ],
#   "strategic_focus_applied": "{strategic_focus}"
# }}
# """

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
    [ULTIMATE WORKER v2026.1.26 - SELF-HEALING & VSM-READY]
    ---------------------------------------------------------------------
    - 🛡️ Isolated Execution: รองรับระบบ Spawn สำหรับ Server 8-GPU เต็มสูบ
    - 🔫 Pre-loaded VSM: โหลด VectorStore ทันทีภายในร่างแยก ป้องกัน AttributeError
    - 🧬 Evidence Streaming: ส่งคืน Memory กลับสู่ Main Process แบบ Real-time
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

    # 3. 🏗️ RECONSTRUCT ISOLATED ENGINE (With Self-Healing LLM & VSM)
    try:
        # 🎯 CRITICAL: ต้องนำเข้าใหม่ภายใน Worker เพื่อให้มองเห็น Module ในโหมด Spawn
        from models.llm import create_llm_instance
        from core.seam_assessment import SEAMPDCAEngine, AssessmentConfig
        from core.vectorstore import load_all_vectorstores
        from config.global_vars import EVIDENCE_DOC_TYPES

        # สร้าง Config เฉพาะกิจสำหรับ Worker นี้
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

        # 🚀 3.1 สร้าง LLM Instance (ร่างแยก)
        worker_llm = None
        if mock_mode == "none":
            worker_llm = create_llm_instance(
                model_name=model_name, 
                temperature=temperature
            )

        # 🔫 3.2 โหลด VectorStoreManager (VSM) ล่วงหน้า
        # แก้ปัญหา: 'SEAMPDCAEngine' object has no attribute 'vectorstore_manager'
        worker_vsm = None
        try:
            worker_vsm = load_all_vectorstores(
                doc_types=[EVIDENCE_DOC_TYPES], 
                enabler_filter=enabler, 
                tenant=tenant, 
                year=str(year)
            )
        except Exception as v_err:
            worker_logger.warning(f"⚠️ VSM Load Warning for {sub_id}: {v_err}")

        # 🛠️ 3.3 คืนชีพ Engine พร้อม "อาวุธ" ครบมือ
        worker_instance = SEAMPDCAEngine(
            config=worker_config, 
            evidence_map_path=evidence_map_path, 
            llm_instance=worker_llm,
            vectorstore_manager=worker_vsm,  # ✅ ส่งเข้าหูขวาไปเลย!
            logger_instance=worker_logger,
            document_map=document_map,      
            ActionPlanActions=action_plan_model
        )
    except Exception as e:
        worker_logger.error(f"❌ Worker initialization failed for {sub_id}: {e}")
        return {
            "sub_id": sub_id, 
            "error": f"Init Error: {str(e)}",
            "status": "failed"
        }, {}

    # 4. ⚡ EXECUTE & STREAM BACK RESULTS
    try:
        # เรียกใช้ฟังก์ชันประเมินตัวเก่งที่คุณพี่แก้ไว้ v2026.01.25
        result, worker_evidence_mem = worker_instance._run_sub_criteria_assessment_worker(sub_criteria_data)
        
        if isinstance(result, dict):
            if 'sub_id' not in result: result['sub_id'] = sub_id
            result['status'] = "success"

        return result, worker_evidence_mem
        
    except Exception as e:
        worker_logger.error(f"❌ Execution error for {sub_id}: {str(e)}")
        return {
            "sub_id": sub_id,
            "error": str(e),
            "status": "failed",
            "is_passed": False,
            "score": 0.0,
            "reason": f"Worker Runtime Exception: {str(e)}"
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
        # [REGISTER AGENTS] - ตัวทำงานหลักในการวิเคราะห์ด้วย LLM
        self.standard_audit_agent = evaluate_with_llm              # สำหรับ L3–L5 (เน้น Audit)
        self.foundation_coaching_agent = evaluate_with_llm_low_level  # สำหรับ L1–L2 (เน้น Coaching)
        
        # [ROUTING & RETRIEVAL]
        self.assessment_router = self.evaluate_pdca
        self.rag_retriever = retrieve_context_with_filter

        # [STATE: CORE RESULTS] - ตะกร้าเก็บผลลัพธ์หลัก
        self.final_subcriteria_results = []      # ผลการประเมินรายข้อ (พร้อม sub_roadmap)
        self.total_stats = self._get_empty_stats_template()  # สรุปคะแนนภาพรวม
        
        # [STATE: STRATEGIC ROADMAPS] - แผนงานยุทธศาสตร์
        self.enabler_roadmap_data = {}           # แผนภาพรวม Enabler (Tier-3 Synthesis)
        self.sub_roadmap_data = {}               # แผนภาพรวมรายข้อ (ชั่วคราว)
        
        # [STATE: EVIDENCE & CACHE] - ระบบจัดการหลักฐานและประสิทธิภาพ
        self.level_details_map = {} 
        self.level_evidence_cache = {}
        self.previous_levels_evidence = [] 
        self._cumulative_rules_cache = {}
        
        # [STATE: LOGGING & DEBUG]
        self.raw_llm_results = []

        # 🎯 [CRITICAL] ห้ามใส่ self.flattened_rubric = [] ตรงนี้เด็ดขาด 
        # เพราะจะไป Overwrite ข้อมูลที่โหลดมาจากขั้นตอนที่ 4
        
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
        [ULTIMATE REVISED v2026.02.05-final]
        - SYNC: แก้ปัญหา UI Matrix กรอบประโดยส่งคะแนนที่สัมพันธ์กับ Tag
        - VALIDATION: เพิ่ม Keyword Cross-check เพื่อดึงคะแนนที่ AI ลืมให้
        - GOVERNANCE: บังคับใช้เกณฑ์การผ่าน (Gatekeeper) ตาม SE-AM Standard
        """
        contextual_config = contextual_config or {}
        top_evidences = top_evidences or []
        
        # 1. จัดการ Metadata และ Enabler Context
        meta = contextual_config.get("_metadata", {})
        tenant = meta.get("tenant", "PEA").upper()
        enabler = meta.get("enabler", "KM").upper()
        replacement_term = contextual_config.get("replacement_term", "กระบวนการจัดการองค์กร")
        log_prefix = f"[{tenant}-{enabler}] {sub_id} L{level}"

        # 2. Robust JSON Repair
        if isinstance(llm_output, str):
            try:
                cleaned = re.sub(r'```json\s*|\s*```|\n+', ' ', llm_output.strip())
                llm_output = json.loads(cleaned)
            except:
                return self._get_fallback_result(log_prefix)

        # 3. เตรียม Keyword Lookup สำหรับตรวจสอบความสอดคล้อง (Consistency Check)
        enabler_defaults = contextual_config.get("_enabler_defaults", {})
        global_enabler_map = PDCA_CONFIG_MAP.get(enabler, PDCA_CONFIG_MAP["DEFAULT"])
        
        phase_kws_lookup = {
            "P": contextual_config.get(sub_id, {}).get(f"L{level}", {}).get("query_synonyms", "").split() or 
                enabler_defaults.get("plan_keywords", []) or global_enabler_map["P"],
            "D": enabler_defaults.get("do_keywords", []) or global_enabler_map["D"],
            "C": enabler_defaults.get("check_keywords", []) or global_enabler_map["C"],
            "A": enabler_defaults.get("act_keywords", []) or global_enabler_map["A"]
        }

        # 4. Extract PDCA Scores (รองรับ Multiple Key Formats)
        pdca_raw = {"P": 0.0, "D": 0.0, "C": 0.0, "A": 0.0}
        norm_out = {str(k).lower().strip(): v for k, v in llm_output.items() if isinstance(k, str)}
        reason_text = str(norm_out.get("reason", "")).lower()

        search_keys = {
            "P": ["p_score", "p_plan_score", "score_p", "plan_score"],
            "D": ["d_score", "d_do_score", "score_d", "do_score"],
            "C": ["c_score", "c_check_score", "score_c", "check_score"],
            "A": ["a_score", "a_act_score", "score_a", "act_score"]
        }

        for phase, keys in search_keys.items():
            score = 0.0
            for k in keys:
                if k in norm_out:
                    try: score = float(norm_out[k]); break
                    except: continue
            
            # ป้องกัน AI "ตาถั่ว": ถ้าบอกว่าเจอใน Reason แต่ลืมให้คะแนน ให้คะแนนพื้นฐาน
            if score < 0.4 and any(kw.lower() in reason_text for kw in phase_kws_lookup.get(phase, [])):
                score = 0.75
                
            pdca_raw[phase] = min(max(score, 0.0), 2.0)

        # 5. Scoring Logic & Floor Injection (เพื่อให้ UI Matrix ขึ้นสีเขียว Solid)
        current_cfg = contextual_config.get(sub_id, {}).get(f"L{level}", {})
        required_phases = current_cfg.get("require_phase") or current_cfg.get("required_phases") or ["P"]
        
        pdca_scored = pdca_raw.copy()
        # Floor Score: ถ้ามีร่องรอย (Score > 0) ให้ดันขึ้นระดับพื้นฐานตาม Level
        floor = 1.0 if level == 1 else 0.8 if level <= 3 else 0.5
        for p in required_phases:
            if pdca_raw[p] > 0.1: # มีหลักฐานบ้าง
                pdca_scored[p] = max(pdca_raw[p], floor)

        # คำนวณคะแนนสุทธิ (0.0 - 2.0)
        sum_req = sum(pdca_scored[p] for p in required_phases)
        max_req = len(required_phases) * 2.0
        normalized_score = round((sum_req / max_req) * 2.0, 2) if max_req else 0.0

        # 6. Gatekeeper Decision (กฎการผ่าน)
        max_rr = max([float(ev.get("rerank_score", ev.get("score", 0))) for ev in top_evidences] or [0.0])
        is_passed = bool(norm_out.get("is_passed", False))
        
        # Auto-Pass based on score
        if normalized_score >= 1.2: is_passed = True
        
        # กฎเหล็กระดับสูง (L4-L5)
        if level >= 4:
            if max_rr < 0.65: is_passed = False # Retrieval ต้องแม่นจริง
            # ต้องมี Check/Act เป็นรูปธรรม
            for critical in [p for p in ["C", "A"] if p in required_phases]:
                if pdca_raw[critical] < 0.4: is_passed = False

        # 7. Coaching Insight & Anti-IT Ghost Pattern
        coaching = str(norm_out.get("coaching_insight", norm_out.get("reason", "")))
        it_patterns = r"(ระบบสารสนเทศอัตโนมัติ|KMS|Software|แอปพลิเคชัน|IT System|พัฒนาระบบ|Digital Platform)"
        cleaned_coaching = re.sub(it_patterns, replacement_term, coaching, flags=re.IGNORECASE)

        # 8. ผลลัพธ์สุดท้ายที่ส่งไปยัง UI
        return {
            "score": normalized_score if is_passed else min(normalized_score, 0.9),
            "is_passed": is_passed,
            "pdca_breakdown": pdca_scored, # ตัวนี้จะทำให้ Matrix ขึ้นสีเขียว
            "pdca_raw": pdca_raw,
            "required_phases": required_phases,
            "reason": norm_out.get("reason", "ประเมินตามเกณฑ์มาตรฐาน SE-AM"),
            "coaching_insight": f"[{'STRENGTH' if is_passed else 'GAP'}] {cleaned_coaching}",
            "max_rerank": max_rr,
            "metadata": {"tenant": tenant, "enabler": enabler}
        }
        
    def _get_fallback_result(self, prefix: str) -> Dict[str, Any]:
        """Fallback เมื่อโครงสร้าง JSON เสียหาย"""
        return {
            "score": 0.0,
            "is_passed": False,
            "pdca_breakdown": {"P": 0.0, "D": 0.0, "C": 0.0, "A": 0.0},
            "reason": "AI Output Parse Error - ระบบป้องกันข้อมูลผิดพลาดทำงาน",
            "coaching_insight": "ไม่สามารถประมวลผลข้อสรุปได้ กรุณาตรวจสอบหลักฐานไฟล์แนบอีกครั้ง",
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

    def _export_results(self, results_data: Any, sub_criteria_id: str, record_id: str = None, **kwargs) -> str:
        """
        [FINAL EXPORTER v2026.01.28 — SEMANTIC SAFE & POSITION-AWARE]
        - ✅ แก้ไข TypeError โดยรองรับ record_id เป็น positional arg ตัวที่ 3
        - ✅ ป้องกันปัญหาค่า null หรือ Empty List ในการหา Max Level
        - ✅ รักษาค่า PDCA และ Confidence ดั้งเดิม (ไม่ Force/ไม่ Fabricate)
        """

        try:
            # --------------------------------------------------
            # 0. Metadata (Priority: Arg > Kwargs > Self)
            # --------------------------------------------------
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # แก้ไขจุดนี้เพื่อรองรับค่าที่ส่งมาจาก run_assessment
            final_record_id = (
                record_id 
                or kwargs.get("record_id") 
                or getattr(self, "current_record_id", f"auto_{timestamp}")
            )

            tenant = getattr(self.config, "tenant", "unknown")
            year = getattr(self.config, "year", "unknown")
            enabler = getattr(self, "enabler", "unknown").upper()

            # --------------------------------------------------
            # 1. Result data normalization
            # --------------------------------------------------
            if results_data is None:
                results_data = getattr(self, "final_subcriteria_results", [])

            if isinstance(results_data, dict):
                results_data = [results_data]

            if not results_data:
                self.logger.warning(f"⚠️ [EXPORT] No result data for {sub_criteria_id}")
                return ""

            # --------------------------------------------------
            # 2. Summary stats (Safety Logic)
            # --------------------------------------------------
            stats = getattr(self, "total_stats", {}) or {}

            # ใช้ default=0 เพื่อป้องกัน ValueError กรณี list ว่าง
            highest_lvl = stats.get(
                "overall_max_level",
                max((int(r.get("highest_full_level", 0)) for r in results_data), default=0)
            )

            total_weighted = stats.get(
                "total_weighted_score",
                sum(float(r.get("weighted_score", 0.0)) for r in results_data)
            )

            # --------------------------------------------------
            # 3. Evidence Audit Trail (Semantic Safe)
            # --------------------------------------------------
            master_map = getattr(self, "evidence_map", {}) or {}
            processed_evidence = {}

            for level_key, bucket in master_map.items():
                if not bucket: continue

                # Normalization
                ev_list = bucket.get("evidences", []) if isinstance(bucket, dict) else bucket
                if not isinstance(ev_list, list) or not ev_list: continue

                # Sort by highest confidence (rerank_score)
                valid_evs = [ev for ev in ev_list if isinstance(ev, dict)]
                if not valid_evs: continue

                sorted_ev = sorted(
                    valid_evs,
                    key=lambda x: float(x.get("rerank_score") or -1.0),
                    reverse=True
                )
                top_ev = sorted_ev[0]

                # Resolve Filename
                doc_id = top_ev.get("doc_id") or top_ev.get("stable_doc_uuid")
                filename = (
                    top_ev.get("filename")
                    or (self.document_map.get(doc_id) if hasattr(self, "document_map") else None)
                    or top_ev.get("source")
                    or "Unknown_Source"
                )

                # PDCA (No Guesses)
                pdca = top_ev.get("pdca_tag") or top_ev.get("pdca_phase")
                pdca = pdca.upper() if isinstance(pdca, str) else None

                # Confidence (No Force 0)
                confidence = top_ev.get("rerank_score")
                if not isinstance(confidence, (int, float)) or confidence < 0:
                    confidence = None

                processed_evidence[str(level_key)] = {
                    "file": filename,
                    "page": top_ev.get("page") or top_ev.get("page_label", "N/A"),
                    "pdca": pdca,
                    "confidence": confidence,
                    "snippet": (top_ev.get("content") or top_ev.get("text") or "")[:160].strip()
                }

            # --------------------------------------------------
            # 4. Final payload (STABLE SCHEMA)
            # --------------------------------------------------
            payload = {
                "metadata": {
                    "record_id": final_record_id,
                    "tenant": tenant,
                    "year": year,
                    "enabler": enabler,
                    "engine_version": "SEAM-ENGINE-v2026.01.28",
                    "exported_at": datetime.now().isoformat()
                },
                "result_summary": {
                    "maturity_level": f"L{highest_lvl}",
                    "total_weighted_score": round(total_weighted, 4),
                    "status": "COMPLETED"
                },
                "sub_criteria_details": results_data,
                "evidence_audit_trail": processed_evidence,
                "enabler_roadmap": getattr(self, "enabler_roadmap_data", {})
            }

            # --------------------------------------------------
            # 5. Export
            # --------------------------------------------------
            export_path = get_assessment_export_file_path(
                tenant=tenant, year=year, enabler=enabler.lower(),
                suffix=f"{sub_criteria_id}_{timestamp}", ext="json"
            )

            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            self.logger.info(f"✅ [EXPORT OK] {export_path}")
            return export_path

        except Exception as e:
            self.logger.error(f"🛑 [EXPORT FAILED] {str(e)}", exc_info=True)
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
        confidence_reason: str = "N/A", # ✅ รับค่าที่คำนวณจาก evaluate_pdca
        **kwargs
    ):
        """
        [AUDIT AGENT ROUTER – v2026.01.27]
        - FIXED: บังคับส่งต่อ confidence_reason อย่างเป็นระบบ
        - ROUTE: L1–L2 → foundation_coaching_agent (Low Level)
        - ROUTE: L3–L5 → standard_audit_agent (High Level)
        """

        if llm_executor is None:
            raise RuntimeError(f"🛑 [ROUTER ERROR] LLM executor missing for {sub_id}")

        # 1. 🔍 เลือก Agent ตามระดับความยากของเกณฑ์
        # L1-L2 เน้นเรื่องระเบียบ/แผน (Low level logic)
        # L3-L5 เน้นเรื่องการบูรณาการ/ผลลัพธ์ (High level logic)
        if level <= 2:
            agent = self.foundation_coaching_agent
        else:
            agent = self.standard_audit_agent

        # 2. 📡ส่งต่อ Payload ไปยัง Agent ที่เลือก
        # หมายเหตุ: **kwargs จะรวมตัวแปรเสริม เช่น enabler_name_th, ai_confidence, focus_points
        return agent(
            context=context,
            sub_criteria_name=sub_criteria_name,
            level=level,
            statement_text=statement_text,
            sub_id=sub_id,
            llm_executor=llm_executor,
            confidence_reason=confidence_reason, # ✅ ส่งต่อชื่อตัวแปรให้ชัดเจน
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
        [ULTIMATE REVISED v2026.01.27] - Optimized for SE-AM v36.9
        - FIXED: ส่ง enabler_full_name และ enabler_code เพื่อรองรับ Result Builder
        - FIXED: ป้องกันปัญหา Multiple Values ใน Router โดยการจัดการ payload ให้คลีน
        - STABLE: รองรับการทำ Audit Trail ด้วย Metadata ที่ครบถ้วน
        """
        log_prefix = f"🧠 [{sub_id}-L{level}]"

        # 1. [PREPARE CONTEXT] ล้างอักขระพิเศษเพื่อความปลอดภัยของ JSON
        pdca_summary = []
        if isinstance(pdca_blocks, dict):
            for tag in ["P", "D", "C", "A"]:
                val = pdca_blocks.get(tag)
                if val:
                    clean_val = str(val).replace('"', "'")
                    pdca_summary.append(f"### {tag} PHASE EVIDENCE ###\n{clean_val}")
        else:
            final_context_str = str(pdca_blocks).replace('"', "'")
        
        final_context_str = "\n\n".join(pdca_summary) if pdca_summary else str(pdca_blocks)

        # 2. [LOOKUP RUBRIC]
        sub_item = next((i for i in self.flattened_rubric if i.get("sub_id") == sub_id), {})
        sub_name = sub_item.get("sub_criteria_name", sub_id)
        
        level_info = next((lv for lv in sub_item.get("levels", []) if lv.get("level") == level), {})
        statement = level_info.get("statement", "")

        # 3. [NORMALIZE CONFIDENCE]
        try:
            if isinstance(audit_confidence, dict):
                conf_val = float(
                    audit_confidence.get("coverage_ratio") or 
                    audit_confidence.get("rerank_score") or 
                    audit_confidence.get("score") or 0.0
                )
            else:
                conf_val = float(audit_confidence or 0.0)
        except:
            conf_val = 0.5

        # 4. [ENABLER RESOLUTION]
        if self.llm is None: self._initialize_llm_if_none()
        
        e_code = str(getattr(self, 'enabler', 'UNKNOWN')).upper()
        # ดึงชื่อเต็มจากตัวแปร Global หรือ Mapping
        e_name_th = SEAM_ENABLER_FULL_NAME_TH.get(e_code, f"ด้าน {e_code}")

        # 5. [BUILD AGENT PAYLOAD] 
        # รวมทุกอย่างที่ Prompt ต้องการ และทุกอย่างที่ Result Builder ต้องการ
        agent_payload = {
            "context": final_context_str,
            "pdca_context": final_context_str,
            "sub_id": sub_id,
            "sub_criteria_name": sub_name,
            "level": level,
            "statement_text": statement,
            "llm_executor": self.llm,
            
            # สำหรับ Prompt Mapping
            "confidence_reason": f"System Confidence (Rerank/Coverage): {conf_val:.4f}",
            "ai_confidence": "HIGH" if conf_val >= 0.7 else "MEDIUM",
            "enabler_name_th": e_name_th,
            
            # สำหรับ Result Builder (_build_audit_result_object)
            "enabler_full_name": e_name_th,
            "enabler_code": e_code,
            
            # Extra Guidelines
            "focus_points": sub_item.get("focus_points", "-"),
            "evidence_guidelines": level_info.get("level_specific_guideline", "-"),
            "specific_contextual_rule": audit_instruction,
        }

        # 6. [EXECUTE ROUTER]
        try:
            # กระจาย payload เข้าสู่ router
            return self.audit_agent_router(**agent_payload)

        except Exception as e:
            self.logger.error(f"🛑 [EVAL-ERROR] {log_prefix}: {str(e)}", exc_info=True)
            # Fallback ที่มี Enabler Info ครบเพื่อให้ UI ไม่แสดง Error เปล่าๆ
            return {
                "sub_id": sub_id,
                "level": level,
                "score": 0.0,
                "is_passed": False,
                "reason": f"Evaluation Failure: {str(e)}",
                "enabler_full_name": e_name_th,
                "enabler_code": e_code
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
        [ULTIMATE REVISED v2026.01.27] - FIXED CONFIDENCE 0.0 ISSUE
        - 🛡️ Score Sync: บังคับอัปเดต confidence ให้เท่ากับ relevance_score
        - 🔍 Deep Resolve: ค้นหาคะแนนจากทุกคีย์ที่เป็นไปได้ (score, rerank, relevance)
        - 🏗️ UUID Alignment: ตรวจสอบความสอดคล้องของ ID เพื่อการทำ Merge Mapping
        """
        import uuid
        import os

        for ev in evidence_list:
            if not isinstance(ev, dict): continue
            
            meta = ev.get("metadata", {})
            if not isinstance(meta, dict): meta = {}
            
            # 1. Resolve ID & UUID (หัวใจสำคัญของการทำ Mapping)
            doc_id = (
                ev.get("doc_id") or 
                ev.get("stable_doc_uuid") or 
                meta.get("stable_doc_uuid") or 
                meta.get("doc_id") or
                f"gen_{uuid.uuid4().hex[:8]}"
            )
            ev["doc_id"] = doc_id
            ev["stable_doc_uuid"] = doc_id

            # 2. Resolve Filename (ล้างเครื่องหมายคำพูดและ Metadata ส่วนเกิน)
            raw_source = (
                meta.get("source_filename") or 
                meta.get("file_name") or 
                ev.get("filename") or 
                ev.get("source") or 
                meta.get("source")
            )
            # ล้างเอาเฉพาะชื่อไฟล์ (ตัดพวก |SCORE:0.xxxx ออกถ้ามี)
            filename_raw = str(raw_source).split('|')[0] if raw_source else "Unknown_File"
            filename = os.path.basename(filename_raw)
            
            # Cross-check กับคลังชื่อไฟล์กลาง
            if (filename == "Unknown_File" or not filename) and hasattr(self, 'document_map'):
                filename = self.document_map.get(doc_id, "Unknown_File")
                
            ev["filename"] = filename
            ev["source_filename"] = filename
            ev["source"] = filename

            # 3. Resolve Page Label
            raw_page = meta.get("page_label") or meta.get("page") or meta.get("page_number") or ev.get("page") or "0"
            ev["page"] = str(raw_page)

            # 4. 🎯 [FIXED] Resolve Scoring & Confidence (จุดที่พบปัญหา)
            actual_score = 0.0
            if hasattr(self, 'get_actual_score'):
                actual_score = self.get_actual_score(ev)
            else:
                # ดึงจากทุกแหล่งที่ JSON ของคุณอาจพ่นออกมา
                actual_score = float(
                    ev.get("relevance_score") or 
                    ev.get("score") or 
                    meta.get("rerank_score") or 
                    meta.get("score") or 0.0
                )
            
            # Sync ทุกฟิลด์ที่ UI เรียกใช้ให้ตรงกัน
            ev["relevance_score"] = actual_score
            ev["score"] = actual_score
            ev["confidence"] = actual_score # ✅ ปลดล็อคค่า 0.0 ให้เป็นคะแนนจริง

            # 5. UI Fields Consistency
            ev["source_type"] = ev.get("source_type") or meta.get("source_type") or "system_gen"
            ev["is_selected"] = ev.get("is_selected") if ev.get("is_selected") is not None else True
            
            # ดึง PDCA Tag จาก Metadata (ถ้ามี)
            ev["pdca_tag"] = ev.get("pdca_tag") or meta.get("pdca_tag") or "Other"
            ev["note"] = ev.get("note") or ""

        return evidence_list
        
    # ------------------------------------------------------------------------------------------
    # [ULTIMATE REVISE v2026.01.30] 🧠 LAYER 1: Decision Engine (The Brain – Final Hardened)
    # ------------------------------------------------------------------------------------------
    def _get_semantic_tag(
        self,
        text: str,
        sub_id: str,
        level: int,
        filename: str = ""
    ) -> str:
        """
        [FINAL STABLE v2026.02]
        Decision Engine สำหรับ PDCA Tag
        Priority:
        1) Keyword Heuristic (Enabler-based)
        2) Behavioral Heuristic (_get_heuristic_pdca_tag)
        3) LLM Semantic Classification
        4) Contextual Fallback (LIMITED)
        5) Ultimate Maturity Fallback
        """

        # ------------------ Preparation ------------------
        text_clean = (text or "").strip()
        text_lower = text_clean.lower()

        # Enabler name + keywords (safe fallback)
        enabler_key = getattr(self.config, "enabler", "DEFAULT").upper()
        enabler_keywords = PDCA_CONFIG_MAP.get(enabler_key, PDCA_CONFIG_MAP.get("DEFAULT", {}))

        # Required phases from contextual rules
        require_phases = self.get_rule_content(sub_id, level, "require_phase") or []

        self.logger.debug(
            f"[TAG-PREP] {sub_id} L{level} | "
            f"ReqPhases={require_phases} | File={filename[:40]}"
        )

        # ------------------ Short Text Guard ------------------
        if len(text_clean) < 20:
            fallback = (
                require_phases[0]
                if require_phases and level <= 2
                else ("P" if level == 1 else "D")
            )
            self.logger.debug(f"[TAG-SHORT] → {fallback}")
            return fallback

        # ------------------ LAYER 1: Enabler Keyword Heuristic ------------------
        for tag, keywords in enabler_keywords.items():
            if any(k.lower() in text_lower for k in keywords):
                self.logger.debug(
                    f"⚡ [KEYWORD-HIT] {tag} | {filename[:30]}"
                )
                return tag

        # ------------------ LAYER 1.5: Behavioral Heuristic (CRITICAL FIX) ------------------
        heuristic_pdca = self._get_heuristic_pdca_tag(text_clean, level)
        if heuristic_pdca:
            self.logger.debug(
                f"🧠 [BEHAVIOR-HIT] {heuristic_pdca} | {filename[:30]}"
            )
            return heuristic_pdca

        # ------------------ LAYER 2: LLM Semantic Classification ------------------
        require_str = ", ".join(require_phases) if require_phases else "P, D, C, A"
        desc_bullets = "\n".join(
            f"- {v}" for v in PDCA_PHASE_DESCRIPTIONS.values()
        )

        system_prompt = (
            "คุณคือผู้เชี่ยวชาญการตรวจประเมินตามมาตรฐาน SE-AM\n"
            f"หน้าที่: จำแนกหลักฐานเข้าหมวด PDCA\n\n"
            f"{desc_bullets}\n\n"
            f"บริบท Level {level}: ให้พิจารณา {require_str} ก่อน\n"
            "ถ้าใกล้เคียง phase ที่จำเป็น ให้เลือก phase นั้น\n"
            "ถ้าไม่เกี่ยวข้องจริง ๆ ให้ใช้ Other\n\n"
            "ตอบเป็น JSON เท่านั้น:\n"
            "{'tag':'P|D|C|A|Other','reason':'เหตุผลสั้น'}"
        )

        user_prompt = (
            f"ชื่อไฟล์: {filename}\n"
            f"เนื้อหา:\n{text_clean[:700]}\n\n"
            "ระบุ tag"
        )

        try:
            raw = _fetch_llm_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                llm_executor=self.llm,
                max_retries=3
            )

            self.logger.debug(f"[LLM-TAG-RAW] {raw[:200]}")

            raw = re.sub(r"```json|```", "", raw).strip()
            data = json.loads(raw)
            if isinstance(data, list) and data:
                data = data[0]

            tag = str(data.get("tag", "Other")).upper().strip()
            if tag in {"P", "D", "C", "A"}:
                self.logger.info(
                    f"🎯 [AI-TAG] {tag} | {filename[:30]}"
                )
                return tag

        except Exception as e:
            self.logger.warning(f"[AI-TAG-ERROR] {filename[:30]} | {e}")

        # ------------------ LAYER 3: Contextual Fallback (LIMITED) ------------------
        # ใช้เฉพาะ L1–L2 เพื่อกัน PDCA ปลอมในระดับสูง
        if require_phases and level <= 2:
            fallback = require_phases[0]
            self.logger.debug(
                f"[CTX-FALLBACK-LIMITED] {fallback}"
            )
            return fallback

        # ------------------ Ultimate Fallback by Maturity ------------------
        if level == 1:
            fallback = "P"
        elif level <= 3:
            fallback = "D"
        elif level == 4:
            fallback = "C"
        else:
            fallback = "A"

        self.logger.debug(
            f"[ULTIMATE-FALLBACK] {fallback} for L{level}"
        )
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
        [FINAL LOCKED v2026.01.28.1]
        - 🔒 Forced PDCA ≠ Real PDCA
        - 🔒 ส่ง pdca_confidence downstream
        """

        pdca_groups = defaultdict(list)
        seen_texts = set()
        all_candidate = (evidences or []) + (baseline_evidences or [])

        require_phases = self.get_rule_content(sub_id, level, "require_phase") or ["P", "D"]

        for idx, chunk in enumerate(all_candidate, start=1):
            txt = (chunk.get("text") or chunk.get("page_content") or "").strip()
            if not txt or len(txt) < 10:
                continue

            txt_hash = hashlib.sha256(txt.encode()).hexdigest()
            if txt_hash in seen_texts:
                continue
            seen_texts.add(txt_hash)

            meta = chunk.get("metadata", {}) or {}
            fname = chunk.get("source_filename") or meta.get("source_filename") or "Unknown"
            page = meta.get("page_label") or meta.get("page") or "N/A"
            is_baseline = chunk.get("is_baseline", False)

            source_display = f"{'[BASELINE] ' if is_baseline else ''}{fname} (P.{page})"

            # ---------- MULTI-LAYER TAGGING ----------
            is_forced = False
            tag_source = "Semantic-Engine"
            final_tag = self._get_semantic_tag(txt, sub_id, level, fname)

            if final_tag not in {"P", "D", "C", "A"}:
                heuristic = self._get_heuristic_pdca_tag(txt, level)
                if heuristic in {"P", "D", "C", "A"}:
                    final_tag = heuristic
                    tag_source = "Heuristic-Rule-Base"

            if final_tag not in {"P", "D", "C", "A"}:
                if level >= 4:
                    continue  # STRICT MODE
                is_forced = True
                final_tag = require_phases[(idx - 1) % len(require_phases)]
                tag_source = f"Forced-L{level}"

            pdca_confidence = (
                0.9 if tag_source == "Semantic-Engine"
                else 0.7 if tag_source == "Heuristic-Rule-Base"
                else 0.4
            )

            pdca_groups[final_tag].append({
                "text": txt,
                "filename": fname,
                "page": page,
                "source_display": source_display,
                "pdca_tag": final_tag,
                "pdca_confidence": pdca_confidence,
                "is_forced": is_forced,
                "is_baseline": is_baseline,
                "relevance": float(chunk.get("rerank_score") or chunk.get("score") or 0.5),
                "tag_source": tag_source
            })

        # ---------- BLOCK OUTPUT ----------
        max_ch = getattr(self.config, "MAX_CHUNKS_PER_BLOCK", 5)
        blocks = {
            "sources": {},
            "actual_counts": {},
            "all_evidences_with_tags": []
        }

        for tag in ["P", "D", "C", "A"]:
            real_only = [c for c in pdca_groups.get(tag, []) if not c["is_forced"]]

            ranked = sorted(
                real_only,
                key=lambda x: (-x["pdca_confidence"], -x["relevance"])
            )[:max_ch]

            blocks["actual_counts"][tag] = len(real_only)
            blocks["sources"][tag] = [c["source_display"] for c in ranked]

            if ranked:
                blocks[tag] = "\n\n".join(
                    f"[{c['source_display']} | {c['tag_source']}]\n{c['text'][:1000]}"
                    for c in ranked
                )
                blocks["all_evidences_with_tags"].extend(ranked)
            else:
                blocks[tag] = f"[ไม่พบหลักฐาน PDCA จริงในหมวด {tag}]"

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
        [FINAL STRENGTH v2026.01.28.1]
        - 🔒 Forced PDCA NEVER counted
        """

        map_key = f"{sub_id}_L{level}"
        new_evidence_list = []
        seen = set()

        for ev in level_temp_map:
            if ev.get("is_forced"):
                continue  # 🔥 DROP FORCED

            tag = ev.get("pdca_tag")
            if tag not in {"P", "D", "C", "A"}:
                continue

            uid = f"{ev.get('doc_id')}:{ev.get('page')}:{tag}"
            if uid in seen:
                continue
            seen.add(uid)

            new_evidence_list.append({
                "sub_id": sub_id,
                "level": level,
                "pdca_tag": tag,
                "doc_id": ev.get("doc_id"),
                "filename": ev.get("filename"),
                "page": ev.get("page"),
                "relevance_score": ev.get("relevance", 0.0),
                "pdca_confidence": ev.get("pdca_confidence", 0.5),
                "timestamp": datetime.now().isoformat()
            })

        if not new_evidence_list:
            return 0.0

        self.evidence_map.setdefault(map_key, []).extend(new_evidence_list)

        # ---------- REAL COVERAGE ONLY ----------
        found = {e["pdca_tag"] for e in new_evidence_list}
        coverage = len(found) / 4.0

        # ---------- STRENGTH ----------
        strength = round(
            (highest_rerank_score * 0.6) +
            (coverage * 0.4),
            2
        )

        self.assessment_results_map[map_key] = {
            "is_passed": llm_result.get("is_passed", False),
            "strength": strength,
            "coverage": coverage
        }

        return strength

    def _robust_hydrate_documents_for_priority_chunks(
        self,
        chunks_to_hydrate: List[Dict],
        vsm: Optional['VectorStoreManager'],
        current_sub_id: Optional[str] = None,
        level: Optional[int] = None
    ) -> List[Dict]:
        """
        [ULTIMATE HYDRATION – FINAL LOCKED VERSION]
        - Hydrate full text from stable_doc_uuid
        - Preserve ingest metadata (page / page_label / chunk_uuid)
        - Correct PDCA tagging (no OTHER leakage)
        - Full-text hash dedup
        """

        active_sub_id = current_sub_id or getattr(self, "sub_id", "unknown")
        level = level or 1

        if not chunks_to_hydrate:
            self.logger.debug(f"ℹ️ [HYDRATION] No chunks for {active_sub_id} L{level}")
            return []

        # --------------------------------------------------
        # 1. Safe PDCA classification
        # --------------------------------------------------
        def _safe_classify(text: str, filename: str = "") -> str:
            try:
                tag = self._get_semantic_tag(text, active_sub_id, level, filename)

                if tag in {"P", "D", "C", "A"}:
                    return tag

                # fallback using rule-based required phase
                reqs = self.get_rule_content(active_sub_id, level, "require_phase") or []
                return reqs[0] if reqs else ("P" if level <= 1 else "D")

            except Exception as e:
                self.logger.warning(f"⚠️ [PDCA-CLASSIFY-ERR] {e} | file={filename}")
                reqs = self.get_rule_content(active_sub_id, level, "require_phase") or []
                return reqs[0] if reqs else ("P" if level <= 1 else "D")

        # --------------------------------------------------
        # 2. Standardize chunk without destroying ingest metadata
        # --------------------------------------------------
        def _standardize_chunk(chunk: Dict, boost: float) -> Dict:
            c = chunk.copy()

            text = (c.get("text") or "").strip()
            meta = c.get("metadata") or {}

            filename = meta.get("source") or meta.get("source_filename") or "unknown"

            # PDCA fix: re-classify ONLY if invalid
            if c.get("pdca_tag") in [None, "", "OTHER", "Other"]:
                if text:
                    c["pdca_tag"] = _safe_classify(text, filename)

            # Preserve is_baseline
            c["is_baseline"] = bool(c.get("is_baseline", False))

            # Score boost (non-destructive)
            c["score"] = max(float(c.get("score", 0.0)), boost)
            c["rerank_score"] = max(float(c.get("rerank_score", 0.0)), boost)

            return c

        # --------------------------------------------------
        # 3. Collect stable_doc_uuid
        # --------------------------------------------------
        stable_ids = {
            c.get("stable_doc_uuid") or c.get("doc_id")
            for c in chunks_to_hydrate
            if c.get("stable_doc_uuid") or c.get("doc_id")
        }

        if not stable_ids or not vsm:
            self.logger.warning("⚠️ [HYDRATION] Missing stable IDs or VSM → fallback mode")
            return [
                _standardize_chunk(c, 0.85)
                for c in chunks_to_hydrate
            ]

        # --------------------------------------------------
        # 4. Fetch full documents from VSM
        # --------------------------------------------------
        stable_doc_map = defaultdict(list)

        try:
            docs = vsm.get_documents_by_id(
                list(stable_ids),
                doc_type=self.doc_type,
                enabler=self.config.enabler
            )
            for d in docs:
                sid = d.metadata.get("stable_doc_uuid") or d.metadata.get("doc_id")
                if sid:
                    stable_doc_map[sid].append({
                        "text": d.page_content,
                        "metadata": d.metadata
                    })
        except Exception as e:
            self.logger.error(f"❌ [HYDRATION] VSM fetch failed: {e}")
            return [
                _standardize_chunk(c, 0.85)
                for c in chunks_to_hydrate
            ]

        # --------------------------------------------------
        # 5. Hydrate + Dedup
        # --------------------------------------------------
        hydrated_docs = []
        seen_hashes = set()
        hydrated_count = 0

        SAFE_META_KEYS = {
            "source",
            "source_filename",
            "page",
            "page_label",
            "page_number",
            "chunk_uuid",
            "chunk_index",
            "doc_id",
            "stable_doc_uuid",
            "version",
            "year",
            "enabler",
            "subject",
            "sub_topic",
        }

        for chunk in chunks_to_hydrate:
            new_chunk = chunk.copy()
            meta = new_chunk.get("metadata") or {}

            sid = meta.get("stable_doc_uuid") or meta.get("doc_id")
            hydrated = False

            if sid and sid in stable_doc_map:
                # choose the most complete text
                full_doc = max(
                    stable_doc_map[sid],
                    key=lambda d: len(d.get("text", ""))
                )

                new_chunk["text"] = full_doc["text"]

                # merge metadata carefully (ingest is source of truth)
                merged_meta = meta.copy()
                for k in SAFE_META_KEYS:
                    if k in full_doc["metadata"] and k not in merged_meta:
                        merged_meta[k] = full_doc["metadata"][k]

                new_chunk["metadata"] = merged_meta
                hydrated = True
                hydrated_count += 1

            boost = 1.0 if hydrated else 0.85
            new_chunk = _standardize_chunk(new_chunk, boost)

            raw_text = (new_chunk.get("text") or "").strip()
            if not raw_text:
                hydrated_docs.append(new_chunk)
                continue

            text_hash = hashlib.sha256(raw_text.encode()).hexdigest()
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                hydrated_docs.append(new_chunk)

        self.logger.info(
            f"✅ [HYDRATION] Done: {len(hydrated_docs)} chunks | "
            f"hydrated={hydrated_count}/{len(chunks_to_hydrate)} | "
            f"dedup={len(chunks_to_hydrate) - len(hydrated_docs)}"
        )

        return (
            self._guarantee_text_key(hydrated_docs)
            if hasattr(self, "_guarantee_text_key")
            else hydrated_docs
        )
    

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
        [ULTIMATE PRECISION v2026.02.01]
        - 40% Rerank (Base Confidence)
        - 30% Level-Aware Keywords (Semantic & Partial Match)
        - 20% Temporal & Source Alignment (Fiscal Year & Doc Type)
        - 10% Structural Quality (Density & Penalty)
        - Contextual Bonuses: Neighbor / Specific Rule
        """
        if not evidence or not isinstance(evidence, dict):
            return 0.0

        # --- 1. PREPARE DATA & METADATA ---
        text = str(evidence.get("text") or evidence.get("page_content") or "").lower().strip()
        meta = evidence.get("metadata") or {}
        
        # ดึงปีงบประมาณปัจจุบันจาก Config (สมมติเป็น 2569 ตามที่ระบุใน Header)
        target_year = str(getattr(self.config, 'year', '2569'))
        
        # --- 2. BASE RERANK SCORE (40%) ---
        raw_val = evidence.get("rerank_score") or evidence.get("score") or 0.0
        normalized_rerank = min(max(float(raw_val), 0.0), 1.0)

        # --- 3. KEYWORD MATCHING (30%) - PARTIAL MATCH LOGIC ---
        cum_rules = self.get_cumulative_rules_cached(sub_id, level)
        target_kws = set()
        if level <= 2:
            target_kws.update(cum_rules.get("plan_keywords", []) + cum_rules.get("do_keywords", []))
        else:
            target_kws.update(cum_rules.get("check_keywords", []) + cum_rules.get("act_keywords", []))

        keyword_score = 0.0
        if target_kws and text:
            # ใช้ Partial Match (ไทย): ถ้า Keyword มีความยาว > 4 ให้เช็คว่าอยู่ใน Text ไหม
            matches = [kw for kw in target_kws if str(kw).lower() in text]
            if matches:
                # คำนวณความหลากหลาย (Diversity) ของ Keyword ที่พบ
                match_ratio = len(matches) / max(1, int(len(target_kws) * 0.4))
                keyword_score = min(match_ratio ** 0.7, 1.0)
                keyword_score = max(keyword_score, 0.20) # Floor 0.20 ถ้าเจออย่างน้อย 1 คำ

        # --- 4. TEMPORAL & SOURCE ALIGNMENT (20%) ---
        alignment_bonus = 0.0
        
        # 📅 Year Check: ให้คะแนนเพิ่มถ้าเจอปีงบประมาณปัจจุบัน / หักคะแนนถ้าเก่าเกินไป
        if target_year in text:
            alignment_bonus += 0.10
        elif any(old_yr in text for old_yr in ["2566", "2567"]):
            alignment_bonus -= 0.10 # Penalty เอกสารล้าสมัย

        # 📄 Source Check: ตรวจสอบความน่าเชื่อถือของประเภทเอกสาร
        filename = str(meta.get("source") or evidence.get("source") or "").lower()
        primary_docs = ["มติ", "บันทึก", "คำสั่ง", "ประกาศ", "นโยบาย", "แผนแม่บท", "ยุทธศาสตร์"]
        if any(p in filename for p in primary_docs):
            alignment_bonus += 0.10

        # --- 5. STRUCTURAL QUALITY & PENALTY (10%) ---
        quality_score = 0.0
        # ❌ Penalty สำหรับ Chunks ที่สั้นเกินไปหรือเป็นแค่สารบัญ
        if len(text) < 100:
            quality_score -= 0.15
        if "สารบัญ" in text or "ภาคผนวก" in text:
            quality_score -= 0.20
        
        # ✅ Bonus สำหรับ Chunk ที่มีความหนาแน่นของข้อมูลสูง (เนื้อหายาวพอเหมาะ)
        if 500 < len(text) < 2000:
            quality_score += 0.05

        # --- 6. CONTEXTUAL BONUSES ---
        neighbor_bonus = 0.15 if (evidence.get("is_neighbor") or meta.get("is_neighbor")) else 0.0
        
        rule_bonus = 0.0
        specific_rule = str(cum_rules.get("specific_contextual_rule", "")).lower()
        if specific_rule and any(w in text for w in specific_rule.split()[:5]):
            rule_bonus = 0.15

        # --- 7. FINAL AGGREGATION ---
        # (0.40 * Rerank) + (0.30 * Keyword) + Alignment + Quality + Bonuses
        final_score = (
            (0.40 * normalized_rerank) + 
            (0.30 * keyword_score) + 
            alignment_bonus + 
            quality_score + 
            neighbor_bonus + 
            rule_bonus
        )

        # 8. HIGH-CONFIDENCE OVERRIDE
        # ถ้า Rerank มาสูงมาก (0.85+) บังคับให้ผ่าน Threshold พื้นฐานเสมอ
        if normalized_rerank >= 0.85:
            final_score = max(final_score, 0.50)

        final_score = min(max(final_score, 0.0), 1.0)

        # 9. LOGGING (PDCA-FREE DEBUG)
        try:
            if final_score > 0.30: # Log เฉพาะตัวที่น่าสนใจ
                self.logger.info(
                    f"🔎 [REL] {sub_id} L{level} | Score:{final_score:.3f} | "
                    f"R:{normalized_rerank:.2f} KW:{keyword_score:.2f} "
                    f"Align:{alignment_bonus:+.2f} Q:{quality_score:+.2f} "
                    f"Src:{os.path.basename(filename)[:20]}"
                )
        except: pass

        return float(final_score)
    
    def _perform_adaptive_retrieval(
        self,
        sub_id: str,
        level: int,
        stmt: str,
        vectorstore_manager: Any,
    ) -> Tuple[List[Dict], float]:
        """
        [ULTIMATE REVISED v2026.02.01]
        - Adaptive Exit: ใช้เกณฑ์คะแนนที่ยืดหยุ่นตามระดับความยาก (L4-L5 ยอมรับ 0.35)
        - Forced Injection: บังคับคำนวณ Relevance Score สำหรับ Top 5 เสมอ
        - Clean Recovery: ลดการเกิด Recovery Sweep ซ้ำซ้อนเพื่อประหยัดเวลา
        """
        start_time = time.time()
        if not stmt or not isinstance(stmt, str):
            return [], 0.0

        candidates: List[Dict] = []
        used_uuids = set()
        final_max_score = 0.0
        level_key = f"L{level}"
        tenant = getattr(self.config, "tenant", "PEA").upper()

        # 🎯 STRATEGY 1: Dynamic Threshold (แก้ปัญหา L4 ที่มักจะได้คะแนน Rerank ต่ำ)
        # ถ้าเป็นระดับ 4-5 ที่เน้นการเชื่อมโยงยุทธศาสตร์ คะแนน 0.35 ถือว่ามีนัยสำคัญแล้ว
        effective_threshold = RETRIEVAL_RELEVANCE_THRESHOLD if level < 4 else 0.35

        def safe_relevance_score(evidence: Dict) -> float:
            try:
                return self.relevance_score_fn(evidence, sub_id, level)
            except Exception as e:
                self.logger.warning(f"⚠️ [SAFE-SCORE] {sub_id} L{level}: {e}")
                return float(evidence.get('rerank_score') or evidence.get('score') or 0.0)

        # --- STEP 1: PRIORITY MAPPING (บังคับดึงข้อมูลที่ Mapping ไว้) ---
        try:
            _, priority_docs = self._get_mapped_uuids_and_priority_chunks(
                sub_id=sub_id, level=level, statement_text=stmt, vectorstore_manager=vectorstore_manager
            ) or (set(), [])
            
            for p in priority_docs:
                uid = p.get("chunk_uuid")
                if not uid or uid in used_uuids: continue
                p["source"] = os.path.basename(p.get("source") or "Unknown")
                p["score"] = max(safe_relevance_score(p), 0.90) # บังคับ High Score สำหรับ Priority
                used_uuids.add(uid)
                candidates.append(p)
                final_max_score = max(final_max_score, p["score"])
        except Exception as e:
            self.logger.warning(f"⚠️ Priority mapping skip: {e}")

        # --- STEP 2: HYBRID QUERY GENERATION ---
        json_queries = self._get_level_aware_queries(sub_id, level_key)
        legacy_queries = self.enhance_query_for_statement(stmt, sub_id, f"{sub_id}.L{level}", level)
        active_queries = list(dict.fromkeys(json_queries + legacy_queries))[:10]

        # --- STEP 3: ITERATIVE RETRIEVAL (พร้อม Early Exit ใหม่) ---
        for i, q in enumerate(active_queries):
            # 🎯 [EARLY EXIT] ใช้เกณฑ์ที่ปรับตาม Level แล้ว
            if len(candidates) >= RETRIEVAL_EARLY_EXIT_COUNT and final_max_score >= effective_threshold:
                self.logger.info(f"🎯 [EARLY-EXIT] {sub_id} L{level} | Found {len(candidates)} | Score {final_max_score:.4f} >= {effective_threshold}")
                break
            
            try:
                res = self.rag_retriever(
                    self._normalize_thai_text(q), self.doc_type, sub_id=sub_id, level=level,
                    vectorstore_manager=vectorstore_manager
                ) or {}
                
                for d in (res.get("top_evidences") or []):
                    uid = d.get("chunk_uuid")
                    score = float(d.get("score", 0.0))
                    
                    if uid and uid not in used_uuids and score >= RETRIEVAL_RERANK_FLOOR:
                        d["source"] = os.path.basename(d.get("source") or "Unknown")
                        
                        # 🎯 [THE CORE FIX]: บังคับฉีดคะแนนคุณภาพสำหรับ 5 ชิ้นแรกเสมอ 
                        # เพื่อไม่ให้คะแนน Rerank ที่ต่ำ (0.36) มาหยุดการทำงานของ AI
                        if len(candidates) < 5 or score > RETRIEVAL_HIGH_RERANK_THRESHOLD:
                            d["score"] = max(score, safe_relevance_score(d))
                        else:
                            d["score"] = score  
                        
                        used_uuids.add(uid)
                        candidates.append(d)
                        final_max_score = max(final_max_score, d["score"])
            except Exception as e:
                self.logger.error(f"❌ Query Loop {i+1} failed: {e}")

        # --- STEP 4: RECOVERY SWEEP (รันต่อเมื่อหลักฐานน้อยหรือคะแนนยังไม่พ้นเกณฑ์) ---
        if final_max_score < effective_threshold or len(candidates) < 3:
            self.logger.info(f"🚨 [RECOVERY] L{level} Max:{final_max_score:.4f} < {effective_threshold}. Sweep triggered.")
            self._execute_recovery_sweep(sub_id, level, stmt, tenant, used_uuids, candidates, vectorstore_manager)
            
            if candidates:
                for c in candidates:
                    if c.get("is_recovery"):
                        # ใช้คะแนนถ่วงน้ำหนักสำหรับข้อมูลที่กวาดมาใหม่
                        c["score"] = max(c.get("score", 0.0), 
                                         0.7 * safe_relevance_score(c) + 0.3 * float(c.get("rerank_score", 0.0)))
                final_max_score = max([float(c.get("score", 0.0)) for c in candidates])

        # --- STEP 5: FINAL SORT & TRIM ---
        candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        final_docs = candidates[:ANALYSIS_FINAL_K]
        
        elapsed = time.time() - start_time
        self.logger.info(f"🏁 [COMPLETE] {sub_id} L{level} | Docs: {len(final_docs)} | Max Score: {final_max_score:.4f} | Time: {elapsed:.2f}s")
        
        return final_docs, float(final_max_score)

    def _execute_recovery_sweep(self, sub_id, level, stmt, tenant, used_uuids, candidates, vectorstore_manager):
        """ 
        [REVISED v2026.02.01] 
        - เพิ่ม Multi-Query Fallback 
        - ปรับ Floor ตามระดับความยาก
        """
        try:
            # 🎯 1. เตรียม Query แบบกว้าง (ตัด Noise)
            base_stmt = stmt.split('(')[0].split('เช่น')[0].strip()
            
            # 🎯 2. สร้าง Backup Query จาก Keywords (ถ้ามี)
            cum_rules = self.get_cumulative_rules_cached(sub_id, level)
            important_kws = " ".join((cum_rules.get("check_keywords") or [])[:3])
            
            # ผสม Query: Tenant + ID + Statement + Keywords สำคัญ
            recovery_query = f"{tenant} {sub_id} {base_stmt} {important_kws}".strip()
            
            self.logger.info(f"🔍 [RECOVERY-START] Query: {recovery_query[:60]}...")

            res_fb = self.rag_retriever(
                self._normalize_thai_text(recovery_query), 
                self.doc_type, 
                sub_id=sub_id, 
                level=level,
                vectorstore_manager=vectorstore_manager,
                enable_neighbor=False # เน้นความเร็ว
            ) or {}
            
            new_found = 0
            # 🎯 3. ปรับ Recovery Floor ตาม Level (L4-L5 ยอมให้ต่ำลงอีก)
            recovery_floor = RETRIEVAL_RERANK_FLOOR * (0.7 if level >= 4 else 0.8)
            
            for d in (res_fb.get("top_evidences") or []):
                uid = d.get("chunk_uuid")
                score = float(d.get("score", 0.0))
                
                if uid and uid not in used_uuids and score >= recovery_floor:
                    d["source"] = os.path.basename(d.get("source") or "Unknown")
                    d["is_recovery"] = True
                    
                    # 🎯 4. Re-calculate score ทันทีเพื่อให้คะแนนสะท้อนคุณภาพจริง
                    # ไม่รอให้ loop นอกทำ เพื่อให้มั่นใจว่าของดีจะไม่ถูกทิ้ง
                    d["score"] = self.relevance_score_fn(d, sub_id, level)
                    
                    used_uuids.add(uid)
                    candidates.append(d)
                    new_found += 1
            
            if new_found > 0:
                self.logger.info(f"✅ [RECOVERY-SUCCESS] Found {new_found} chunks with floor {recovery_floor:.3f}")
                
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
        [JUDICIAL REVIEW - ENHANCED v2026.02.05]
        - เพิ่ม log ละเอียด + appeal hint ที่คมชัดขึ้น
        - ใช้ highest_rerank_score เป็น weight ในการ override
        - เพิ่ม fallback JSON structure ถ้า LLM ตอบไม่ครบ
        - ไม่เปลี่ยน signature หรือ input เดิมเลย
        """
        log_prefix = f"⚖️ [EXPERT-APPEAL] {sub_id} L{level}"
        self.logger.info(f"{log_prefix} | Starting rescue (Max Rerank: {highest_rerank_score:.4f} | Missing: {missing_tags})")

        # 1. [PREPARE APPEAL HINT] – ดุดัน + เน้น substance over form
        missing_str = ", ".join(sorted(set(missing_tags))) if missing_tags else "PDCA Core Criteria (P/D/C/A)"
        
        appeal_instruction = f"""
    ### 🚨 EXPERT JUDICIAL REVIEW - SECOND CHANCE (OVERRIDE MODE) 🚨
    [CONTEXT]: หลักฐานมีความเกี่ยวข้องสูงมาก (Rerank {highest_rerank_score:.4f}) แต่รอบแรกตัดสินว่าไม่ผ่านเพราะ "{first_attempt_reason[:150]}..."
    [CRITICAL FOCUS]: พิจารณาเนื้อหาเชิงสาระ (Substance over Form) ในประเด็นที่ขาด: {missing_str}
    [MANDATORY RULE]: 
    - หากพบร่องรอยแม้เพียงบางส่วน ให้ใช้ "ดุลยพินิจเชิงบวก" (Expert Positive Override)
    - อย่าติดกับรูปแบบ แต่ดูความหมายจริงของหลักฐาน
    - ถ้าผ่าน ให้เพิ่ม reason ว่า "ผ่านจากการอุทธรณ์เชิงผู้เชี่ยวชาญ"
    """

        # 2. [PDCA BLOCKS INJECTION] – ฉีด hint เข้าไปด้านบนสุด
        original_blocks = base_kwargs.get("pdca_blocks", [])
        expert_pdca_blocks = []
        if isinstance(original_blocks, list):
            expert_pdca_blocks = [{"tag": "SYSTEM-APPEAL", "content": appeal_instruction}] + list(original_blocks)
        else:
            expert_pdca_blocks = f"{appeal_instruction}\n\n{str(original_blocks)}"

        # 3. [SAFE KWARGS CONSTRUCTION] – กรองเฉพาะที่ llm_evaluator_to_use รับได้
        safe_kwargs = {
            "pdca_blocks": expert_pdca_blocks,
            "sub_id": sub_id,
            "level": level,
            "audit_confidence": highest_rerank_score,  # ใช้ rerank เป็น confidence หลัก
        }
        
        # เพิ่ม audit_instruction ถ้ามี (fallback ถ้า base_kwargs ไม่มี)
        if audit_instruction:
            safe_kwargs["audit_instruction"] = audit_instruction
        elif "audit_instruction" in base_kwargs:
            safe_kwargs["audit_instruction"] = base_kwargs["audit_instruction"]

        # 4. [EXECUTE RE-EVALUATION]
        try:
            self.logger.debug(f"{log_prefix} | Calling evaluator with safe_kwargs: {safe_kwargs.keys()}")
            re_eval_result = llm_evaluator_to_use(**safe_kwargs)
            
            if not isinstance(re_eval_result, dict):
                self.logger.warning(f"{log_prefix} | Evaluator returned non-dict → fallback")
                re_eval_result = {"is_passed": False, "score": 0.0, "reason": "Evaluator returned invalid format"}

            # 5. [EVALUATE & OVERRIDE]
            is_passed_now = bool(re_eval_result.get("is_passed", False))
            
            if is_passed_now:
                self.logger.info(f"🛡️ [OVERRIDE-SUCCESS] {log_prefix} | อุทธรณ์สำเร็จ! (New Score: {re_eval_result.get('score', 'N/A')})")
                re_eval_result.update({
                    "is_safety_pass": True,
                    "appeal_status": "GRANTED",
                    "reason": f"🌟 [EXPERT OVERRIDE]: {re_eval_result.get('reason', 'ผ่านจากการอุทธรณ์')}"
                })
            else:
                self.logger.info(f"❌ [APPEAL-DENIED] {log_prefix} | ผลอุทธรณ์: ไม่ผ่าน")
                re_eval_result["appeal_status"] = "DENIED"

            # Fallback ถ้า JSON ไม่ครบ field
            required_keys = ["is_passed", "score", "reason"]
            for key in required_keys:
                if key not in re_eval_result:
                    re_eval_result[key] = False if key == "is_passed" else 0.0 if key == "score" else "Missing key after appeal"

            return re_eval_result

        except Exception as e:
            self.logger.error(f"🛑 [APPEAL-CRASH] {log_prefix} failed: {str(e)}", exc_info=True)
            return {
                "is_passed": False,
                "score": 0.0,
                "appeal_status": "FATAL_ERROR",
                "reason": f"Appeal system error: {str(e)[:200]}",
                "is_safety_pass": False
            }

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


    def _update_internal_evidence_map(self, merged_evidence: Dict[str, Any]):
        """
        [FINAL REVISED v2026.01.27 - THE PERSISTENCE GUARD]
        - 🛡️ Atomic Sync: บังคับ Normalize Metadata ก่อนบันทึกเข้า State เสมอ
        - 🧠 Hash Intelligence: ป้องกันการบันทึกหลักฐานซ้ำ (Deduplication)
        - ⚡ Data Alignment: ประกันว่าฟิลด์ 'content' และ 'confidence' จะถูก Sync ให้ตรงกัน
        """
        if not hasattr(self, 'evidence_map') or self.evidence_map is None:
            self.evidence_map = {}
            
        if not isinstance(merged_evidence, dict): 
            return

        def get_stable_hash(text: str) -> str:
            """สร้าง Fingerprint สำหรับตรวจสอบข้อความซ้ำ"""
            if not text: return ""
            # ใช้หน้า-หลังของ Content เพื่อความเร็วและแม่นยำ
            target = f"{text[:250]}...{text[-250:]}" if len(text) > 500 else text
            return hashlib.md5(target.encode('utf-8')).hexdigest()

        for key, incoming_data in merged_evidence.items():
            # รองรับทั้งโครงสร้าง {"evidences": []} และ list โดยตรง
            new_ev_list = incoming_data.get("evidences", []) if isinstance(incoming_data, dict) else incoming_data
            if not isinstance(new_ev_list, list): 
                continue
                
            # เตรียม Bucket สำหรับ Level Key นั้นๆ (เช่น 1.1_L1)
            if key not in self.evidence_map or not isinstance(self.evidence_map[key], dict):
                self.evidence_map[key] = {"status": "completed", "evidences": []}
            
            target_bucket = self.evidence_map[key]
            
            # ดึง Hash ของหลักฐานที่มีอยู่แล้วมาเช็ค
            existing_hashes = {
                get_stable_hash(str(e.get('content') or e.get('text', ''))) 
                for e in target_bucket["evidences"]
            }
            
            for ev in new_ev_list:
                if not isinstance(ev, dict): continue
                
                # 1. สกัดเนื้อหาหลัก
                content_str = str(ev.get('content') or ev.get('text') or "").strip()
                if not content_str: 
                    continue 
                
                # 2. ตรวจสอบความซ้ำซ้อน (Deduplication)
                ev_hash = get_stable_hash(content_str)
                if ev_hash not in existing_hashes:
                    
                    # 🎯 [THE CORE FIX]: ส่งเข้า Normalize เพื่อแก้ปัญหา Confidence 0.0
                    # ขั้นตอนนี้จะทำให้มั่นใจว่า Metadata ทุกอย่างถูกสกัดมาลงที่ Root Object
                    normalized_batch = self._normalize_evidence_metadata([ev])
                    
                    if normalized_batch:
                        clean_ev = normalized_batch[0]
                        
                        # 3. Final Content Restoration
                        # ประกันว่า Content ล่าสุดจะไม่หายไปและถูกเรียกใช้ในชื่อ 'content' เสมอ
                        clean_ev["content"] = content_str
                        
                        # 4. Persistence
                        target_bucket["evidences"].append(clean_ev)
                        existing_hashes.add(ev_hash)

        self.logger.info(f"✅ State Sync complete. Level Groups: {len(self.evidence_map)}")

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

    def _save_evidence_map(
        self,
        map_to_save: Optional[Dict[str, Any]] = None,
        clear_existing: bool = False
    ):
        """
        [ULTIMATE HARDENED BUILD v2026.01.27.1700]
        - ✅ ป้องกัน NoneType Crash 100% ด้วย _safe_float และ _safe_str
        - ✅ บังคับให้ข้อมูลทุกโหนดผ่านการ Normalize ก่อนการ Sort
        - ✅ แก้ไขปัญหา 'float() argument must be a string or a real number, not NoneType'
        - 🛡️ Atomic write (tempfile + move) เพื่อป้องกันไฟล์พังหากระบบดับ
        """

        # --- INTERNAL HELPERS ---
        def _safe_float(val, default=0.0) -> float:
            try:
                if val is None: return float(default)
                return float(val)
            except (TypeError, ValueError):
                return float(default)

        def _safe_str(val, default="0") -> str:
            if val is None: return str(default)
            return str(val).strip()

        def _normalize_evidence_node(e: Dict[str, Any]) -> Dict[str, Any]:
            """แปลง Raw Data ให้เป็นคลีนโหนดที่ปลอดภัยสำหรับ UI และการคำนวณ"""
            return {
                "doc_id": _safe_str(e.get("doc_id") or e.get("chunk_uuid"), "unknown"),
                "filename": _safe_str(e.get("filename"), "Unknown File"),
                "page": _safe_str(e.get("page") or e.get("page_label"), "0"),
                "source_type": _safe_str(e.get("source_type"), "system_gen"),
                "is_selected": bool(e.get("is_selected", True)),
                "relevance_score": _safe_float(
                    e.get("relevance_score"), 
                    _safe_float(e.get("rerank_score", 0.0))
                ),
                "note": _safe_str(e.get("note"), "")
            }

        try:
            # 1. เตรียม Path และตรวจสอบความพร้อมของ Folder
            map_file_path = get_evidence_mapping_file_path(
                tenant=self.config.tenant,
                year=self.config.year,
                enabler=self.enabler
            )
            os.makedirs(os.path.dirname(map_file_path), exist_ok=True)

            # 2. โหลดแมพเดิมที่มีอยู่มาตั้งต้น (Merge Logic)
            final_map = {} if clear_existing else self._load_evidence_map(is_for_merge=True)

            # ตรวจสอบ Incoming Data
            incoming = map_to_save if map_to_save is not None else getattr(self, "evidence_map", {})
            if not isinstance(incoming, dict):
                self.logger.warning("⚠️ [EVIDENCE-MAP] Incoming map is not dict, skip save")
                return

            # 3. เริ่มขั้นตอนการ Merge และ Normalize
            for key, evidence_data in incoming.items():
                if not key or not isinstance(key, str): continue

                # สร้าง Bucket สำหรับแต่ละหัวข้อประเมิน (e.g., 1.1_L1)
                bucket = final_map.setdefault(
                    key, {"status": "pending", "evidences": []}
                )
                existing_evs = bucket["evidences"]

                # ดึง List ของหลักฐานออกมา
                new_evs = (
                    evidence_data.get("evidences", [])
                    if isinstance(evidence_data, dict)
                    else (evidence_data if isinstance(evidence_data, list) else [])
                )

                for raw_e in new_evs:
                    if not isinstance(raw_e, dict): continue

                    # Normalize ข้อมูลก่อนนำไปใช้งาน
                    normalized = _normalize_evidence_node(raw_e)
                    
                    # สร้าง Key สำหรับเช็คความซ้ำ (doc_id + หน้า)
                    idx_key = f"{normalized['doc_id']}_{normalized['page']}"

                    match = next(
                        (e for e in existing_evs if f"{e.get('doc_id')}_{e.get('page')}" == idx_key),
                        None
                    )

                    if match:
                        # อัปเดตข้อมูลเดิมที่มีอยู่ (Update)
                        match.update(normalized)
                        # ถ้าของเดิมหรือของใหม่ระบุเป็น human_map ให้คงสถานะไว้
                        if normalized.get("source_type") == "human_map":
                            match["source_type"] = "human_map"
                    else:
                        # เพิ่มข้อมูลใหม่ (Insert)
                        existing_evs.append(normalized)

            # 4. Post-processing (เรียงลำดับความเกี่ยวข้อง และอัปเดตสถานะ)
            for bucket in final_map.values():
                evs = bucket.get("evidences", [])

                # ประกันความปลอดภัยชั้นสุดท้ายก่อนการ Sort (ป้องกัน NoneType รั่วไหล)
                for e in evs:
                    e["relevance_score"] = _safe_float(e.get("relevance_score"))
                    e["page"] = _safe_str(e.get("page"))

                # เรียงลำดับตามความเกี่ยวข้อง (สูงสุดไปต่ำสุด)
                evs.sort(key=lambda x: x["relevance_score"], reverse=True)

                # อัปเดตสถานะ (ถ้ามีชิ้นไหน User เลือกเอง ให้ถือว่าเป็น reviewed)
                bucket["status"] = (
                    "reviewed"
                    if any(e.get("source_type") == "human_map" for e in evs)
                    else "ai_generated"
                )

            # 5. การบันทึกไฟล์แบบ Atomic (เขียนไฟล์ชั่วคราวก่อนแล้วค่อยย้าย)
            temp_path = None
            temp_dir = os.path.dirname(map_file_path)
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", delete=False, dir=temp_dir, suffix=".tmp", encoding="utf-8"
                ) as tmp:
                    json.dump(final_map, tmp, indent=4, ensure_ascii=False)
                    temp_path = tmp.name

                shutil.move(temp_path, map_file_path)
                
                # Sync กลับเข้าสู่ Instance Memory
                self.evidence_map = final_map
                self.logger.info(f"✅ [EVIDENCE-MAP] Save Successful: {map_file_path}")

            except Exception as io_err:
                self.logger.error(f"❌ [EVIDENCE-MAP] File System Error: {str(io_err)}")
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

        except Exception as e:
            # 🧯 กฎเหล็ก: ห้ามหยุดกระบวนการประเมินหลักแม้การเซฟจะล้มเหลว
            self.logger.error(f"❌ [EVIDENCE-MAP] Fatal Internal Error: {str(e)}")
            self.logger.warning("🧯 [EVIDENCE-MAP] Skipped saving this round, assessment continues")

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
        [ULTIMATE REVISED MERGE v2026.01.31-final-stable]
        - 🛡️ Metadata Sync: เชื่อม Key ให้ตรงกับระบบ Ingestion ล่าสุด (source_filename, filename, source, metadata.source)
        - 🧬 Traceability Guard: ใช้ UID (doc_id + page) ป้องกันนับหลักฐานซ้ำ
        - ⚖️ Maturity Lock: หยุดนับระดับทันทีถ้า level ก่อนหน้าไม่ผ่านหรือ capped
        - ⚖️ Weighted Score: highest_level * weight / 5 (ตาม SE-AM logic)
        - Human-in-the-loop Guard: ไม่ทับข้อมูล source_type = "human_map"
        - Logging ละเอียดเพื่อ debug metadata sync และ maturity lock
        """
        if not sub_result:
            return None

        sub_id = str(sub_result.get("sub_id", "Unknown"))
        incoming_levels = sub_result.get("level_details", {}) or {}
        
        # --------------------------------------------------
        # 1. Evidence Merge & Audit Trail Sync
        # --------------------------------------------------
        if temp_map and isinstance(temp_map, dict):
            if not hasattr(self, "evidence_map"):
                self.evidence_map = {}

            for level_key, ev_list in temp_map.items():
                if not isinstance(ev_list, list) or not ev_list:
                    continue
                if "_L" not in level_key:
                    continue

                lv_num = level_key.split("_L")[-1]
                lv_data = incoming_levels.get(lv_num, {})
                
                node = self.evidence_map.setdefault(level_key, {
                    "status": "pending",
                    "evidences": [],
                    "pdca": None,
                    "confidence": None,
                    "snippet": "",
                    "file": "Unknown File",
                    "page": "N/A"
                })
                
                existing_evs = node["evidences"]
                # ใช้ UID ป้องกัน duplicate (doc_id + page)
                seen_uids = {f"{e.get('doc_id')}_{e.get('page')}": i for i, e in enumerate(existing_evs)}

                for ev in ev_list:
                    doc_id = ev.get("doc_id") or ev.get("stable_doc_uuid") or "unknown_id"
                    page = str(ev.get("page_label") or ev.get("page") or "0")
                    uid = f"{doc_id}_{page}"

                    # Guard: ไม่ทับข้อมูลที่มนุษย์แก้ (human_map)
                    if uid in seen_uids:
                        idx = seen_uids[uid]
                        if existing_evs[idx].get("source_type") == "human_map":
                            self.logger.debug(f"[MERGE-GUARD] Skip overwrite for human_map UID: {uid}")
                            continue
                        existing_evs[idx].update(ev)
                    else:
                        existing_evs.append(ev)
                        seen_uids[uid] = len(existing_evs) - 1

                # Sync Metadata สำหรับ UI/Export (เลือก top evidence)
                if existing_evs:
                    # เรียงลำดับหาหลักฐานที่น่าเชื่อถือที่สุด (rerank_score สูงสุด)
                    existing_evs.sort(key=lambda x: float(x.get("rerank_score") or x.get("relevance_score") or 0.0), reverse=True)
                    top_ev = existing_evs[0]
                    
                    # ดึงชื่อไฟล์จาก key ที่ ingestion สร้าง (priority order)
                    file_keys = ["source_filename", "filename", "source", "metadata.source"]
                    res_file = next((top_ev.get(k) for k in file_keys if top_ev.get(k)), "Unknown File")
                    
                    raw_text = top_ev.get("text") or top_ev.get("content") or ""
                    clean_snippet = raw_text[:350].replace("\n", " ").strip() + "..." if raw_text else ""
                    
                    node.update({
                        "pdca": top_ev.get("pdca_tag") or lv_data.get("pdca_breakdown", {}).get("top_phase") or "P",
                        "confidence": round(float(top_ev.get("rerank_score") or top_ev.get("relevance_score") or 0.0), 2),
                        "snippet": clean_snippet,
                        "file": res_file,
                        "page": str(top_ev.get("page_label") or top_ev.get("page") or "N/A"),
                        "status": "completed"
                    })

                    self.logger.debug(f"[MERGE-SYNC] {sub_id}_L{lv_num} | Top File: {res_file} | Confidence: {node['confidence']}")

        # --------------------------------------------------
        # 2. Results Integration (The Bridge)
        # --------------------------------------------------
        if not hasattr(self, "final_subcriteria_results"):
            self.final_subcriteria_results = []

        target = next((r for r in self.final_subcriteria_results if str(r.get("sub_id")) == sub_id), None)

        if not target:
            target = {
                "sub_id": sub_id,
                "sub_criteria_name": sub_result.get("sub_criteria_name", f"Criteria {sub_id}"),
                "weight": float(sub_result.get("weight", 3.0)),
                "level_details": {},
                "highest_full_level": 0,
                "weighted_score": 0.0,
                "pdca_overall": {"P": 0.0, "D": 0.0, "C": 0.0, "A": 0.0},
                "sub_roadmap": sub_result.get("sub_roadmap") or {},
                "strategic_focus": sub_result.get("strategic_focus") or ""
            }
            self.final_subcriteria_results.append(target)

        if isinstance(incoming_levels, dict):
            for lv, lv_data in incoming_levels.items():
                target["level_details"][str(lv)] = lv_data

        # --------------------------------------------------
        # 3. Maturity Step-Ladder & Score Calculation
        # --------------------------------------------------
        highest = 0
        pdca_sum = {"P": 0.0, "D": 0.0, "C": 0.0, "A": 0.0}
        passed_count = 0

        # ตรวจสอบทีละระดับ 1-5 (หยุดทันทีถ้าเจอไม่ผ่านหรือ capped)
        for l in range(1, 6):
            data = target["level_details"].get(str(l))
            if not data or not data.get("is_passed") or data.get("is_maturity_capped"):
                self.logger.debug(f"[MATURITY-STOP] {sub_id} หยุดที่ L{l} (ไม่ผ่านหรือ capped)")
                break
            
            highest = l
            bd = data.get("pdca_breakdown", {}) or {}
            for k in pdca_sum:
                pdca_sum[k] += float(bd.get(k, 0.0))
            passed_count += 1

        target["highest_full_level"] = highest
        
        # Weighted Score ตาม SE-AM (ระดับสูงสุดที่ผ่านต่อเนื่อง * น้ำหนัก / 5)
        target["weighted_score"] = round(highest * (float(target["weight"]) / 5), 2)
        
        if passed_count > 0:
            target["pdca_overall"] = {k: round(v / passed_count, 2) for k, v in pdca_sum.items()}

        self.logger.info(f"[MERGE-FINAL] {sub_id} | Highest: L{highest} | Weighted Score: {target['weighted_score']}")

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
        [FINAL SHIELDED BUILD v2026.01.28 - PDCA SYNC ENABLED]
        - 🛡️ Shield Pattern: Score คำนวณก่อน IO
        - 🔄 Evidence Sync: ดึงข้อมูลจาก Worker เข้าสู่ Audit Trail โดยตรง
        - ✅ แก้ไขปัญหาค่า null ในไฟล์ JSON
        """

        start_ts = time.time()
        self.is_sequential = sequential
        self.current_record_id = record_id or self.record_id

        # -------------------------------
        # 0. Init State (CRITICAL)
        # -------------------------------
        self.final_subcriteria_results = []
        self.sub_roadmap_data = None
        self.enabler_roadmap_data = None

        if document_map:
            self.document_map.update(document_map)

        if not getattr(self, "evidence_map", None):
            self.evidence_map = {}

        # -------------------------------
        # 1. Load Rubric
        # -------------------------------
        self.flattened_rubric = self._flatten_rubric_to_statements()
        grouped_sub_criteria = self._group_statements_by_sub_criteria(self.flattened_rubric)

        is_all = str(target_sub_id).lower() == "all"
        sub_criteria_list = (
            list(grouped_sub_criteria.values())
            if is_all
            else [grouped_sub_criteria.get(target_sub_id)]
        )

        if not all(sub_criteria_list):
            return self._create_failed_result(
                self.current_record_id,
                f"Criteria '{target_sub_id}' not found",
                start_ts
            )

        total_subs = len(sub_criteria_list)
        results_list = []

        # -------------------------------
        # 2. Core Assessment
        # -------------------------------
        if is_all and not sequential:
            # MODE A: PARALLEL
            max_workers = int(os.environ.get("MAX_PARALLEL_WORKERS", 4))
            worker_args = [
                self._prepare_worker_tuple(sub, self.document_map)
                for sub in sub_criteria_list
            ]

            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(processes=max_workers) as pool:
                for idx, (res, worker_mem) in enumerate(
                    pool.imap_unordered(_static_worker_process, worker_args)
                ):
                    results_list.append((res, worker_mem))
                    self._merge_worker_results(res, worker_mem)

                    self.db_update_task_status(
                        progress=15 + int(((idx + 1) / total_subs) * 65),
                        message=f"🧠 ประเมิน {res.get('sub_id', '?')} สำเร็จ"
                    )
        else:
            # MODE B: SEQUENTIAL
            if not vectorstore_manager:
                self._initialize_vsm_if_none()
            vsm = vectorstore_manager or self.vectorstore_manager

            for idx, sub_criteria in enumerate(sub_criteria_list):
                sub_id = str(sub_criteria.get("sub_id", "Unknown"))
                res, worker_mem = self._run_sub_criteria_assessment_worker(
                    sub_criteria, vsm, []
                )

                results_list.append((res, worker_mem))
                self._merge_worker_results(res, worker_mem)

                self.db_update_task_status(
                    progress=15 + int(((idx + 1) / total_subs) * 65),
                    message=f"🧠 ประเมิน {sub_id} สำเร็จ (Sequential)"
                )

        # -------------------------------
        # 3. Evidence Guard (REVISED & PATCHED)
        # -------------------------------
        self.db_update_task_status(progress=85, message="🧩 กำลังจัดระเบียบหลักฐานและวิเคราะห์ PDCA")

        full_raw_mapping = self.merge_evidence_mappings(results_list)
        self._update_internal_evidence_map(full_raw_mapping)

        total_evidence_found = 0
        for key in list(self.evidence_map.keys()):
            bucket = self.evidence_map[key]
            
            # ตรวจสอบโครงสร้าง bucket ให้พร้อมใช้งาน
            if not isinstance(bucket, dict):
                bucket = {"status": "ai_generated", "evidences": bucket}
                self.evidence_map[key] = bucket

            # 🎯 [FIX START]: เติมข้อมูลที่ขาดหาย (PDCA, Confidence, Snippet) จากหลักฐานที่ดีที่สุด
            ev_list = self._deduplicate_list(bucket.get("evidences", []))
            if ev_list:
                # 1. เรียงลำดับเอาหลักฐานที่ AI มั่นใจสูงสุดมาเป็นตัวแทนระดับ Level
                ev_list.sort(key=lambda x: float(x.get("rerank_score") or x.get("relevance_score") or 0.0), reverse=True)
                top_ev = ev_list[0]
                
                # 2. สร้าง Snippet ข้อความสั้นเพื่อแสดงบน Dashboard
                raw_content = top_ev.get("text") or top_ev.get("content") or top_ev.get("page_content") or ""
                clean_snippet = raw_content[:350].replace("\n", " ").strip() + "..." if raw_content else ""
                
                # 3. Sync ข้อมูลจาก Top Evidence ขึ้นสู่ระดับ Node (Audit Trail)
                bucket.update({
                    "pdca": top_ev.get("pdca_tag") or "P",
                    "confidence": round(float(top_ev.get("rerank_score") or top_ev.get("relevance_score") or 0.0), 2),
                    "snippet": clean_snippet,
                    "file": top_ev.get("filename") or top_ev.get("source_filename") or top_ev.get("source"),
                    "page": str(top_ev.get("page") or top_ev.get("page_label") or "N/A")
                })
                
                bucket["evidences"] = ev_list
                total_evidence_found += len(ev_list)
                bucket["status"] = "completed"
            # 🎯 [FIX END]

        # -------------------------------
        # 4. 🛡️ SCORE FREEZE (CRITICAL)
        # -------------------------------
        overall_stats = self._calculate_overall_stats(target_sub_id) or {}
        overall_stats["evidence_used_count"] = total_evidence_found

        # -------------------------------
        # 5. IO RISK ZONE (SAFE NOW)
        # -------------------------------
        try:
            self._save_evidence_map(self.evidence_map)
        except Exception as e:
            self.logger.warning(f"🧯 [EVIDENCE-MAP] Save failed: {e}")

        # -------------------------------
        # 6. Enabler Roadmap
        # -------------------------------
        if self.final_subcriteria_results:
            self.enabler_roadmap_data = self.synthesize_enabler_roadmap(
                sub_criteria_results=self.final_subcriteria_results,
                enabler_name=self.enabler,
                llm_executor=self.llm
            )

        # -------------------------------
        # 7. Final Response
        # -------------------------------
        final_response = {
            "record_id": self.current_record_id,
            "status": "COMPLETED",
            "enabler": self.enabler,
            "summary": overall_stats,
            "sub_criteria_results": self.final_subcriteria_results,
            "evidence_audit_trail": self.evidence_map,
            "enabler_roadmap": self.enabler_roadmap_data,
            "run_time_seconds": round(time.time() - start_ts, 2)
        }

        if export:
            self._export_results(self.final_subcriteria_results, target_sub_id, self.current_record_id)

        self.db_update_task_status(progress=100, message="✅ ประเมินเสร็จสมบูรณ์", status="COMPLETED")
        return final_response
    

    # ------------------------------------------------------------------
    # 🏛️ [ULTIMATE REVISED] generate_sub_roadmap - USING UB_ROADMAP_PROMPT
    # ------------------------------------------------------------------
    def generate_sub_roadmap(
        self,
        sub_id: str,
        sub_criteria_name: str,
        enabler: str,
        aggregated_insights: List[Dict[str, Any]],
        strategic_focus: str = ""
    ) -> Dict[str, Any]:
        """
        - เปลี่ยนมาใช้ UB_ROADMAP_PROMPT ที่รวม System Rules และ Template ไว้ด้วยกัน
        - ใช้ .format() หรือ .invoke() ตามมาตรฐาน PromptTemplate
        """
        if not aggregated_insights:
            return self._get_emergency_fallback_plan(sub_id, sub_criteria_name, "No insights provided")

        self.logger.info(f"🚀 [ROADMAP] Generating Professional Coach Plan via UB_PROMPT for {sub_id}")

        # --- [STEP 1: PREPARE DATA - เหมือนเดิมแต่เน้นความสะอาด] ---
        best_practice_assets = []
        top_asset_name = None
        highest_continuous = 0
        has_gap = False
        noise_filenames = ["Unknown File", "เอกสารอ้างอิง", "N/A", "Reference Document", "SCORE:", ""]
        evidence_map = getattr(self, "evidence_map", {})

        for item in aggregated_insights:
            lv = int(item.get("level", 0))
            passed = (item.get("status") == "PASSED")
            ev_key = f"{sub_id}_L{lv}"
            ev_node = evidence_map.get(ev_key, {})
            
            raw_filename = ev_node.get("file", "Unknown File")
            clean_filename = raw_filename if raw_filename not in noise_filenames else None
            
            if passed:
                if not has_gap: highest_continuous = lv
                if not top_asset_name and clean_filename: top_asset_name = clean_filename
                best_practice_assets.append(f"- Level {lv}: {clean_filename if clean_filename else 'แนวทางปฏิบัติเดิม'}")
            else:
                has_gap = True

        # --- [STEP 2: PREPARE ENRICHED CONTEXT] ---
        # เตรียมข้อมูลสำหรับตัวแปร aggregated_insights ใน Prompt
        enriched_context = (
            f"💎 EXISTING STRATEGIC ASSETS:\n" +
            ("\n".join(best_practice_assets) if best_practice_assets else "- ไม่พบหลักฐานหลัก") +
            "\n\n🚨 GAP ANALYSIS:\n" +
            "\n".join([f"- L{i.get('level')}: {i.get('insight_summary')}" for i in aggregated_insights])
        )

        if not strategic_focus:
            strategic_focus = f"ยกระดับจาก L{highest_continuous} มุ่งสู่มาตรฐาน Excellence"

        # --- [STEP 3: EXECUTE UB_ROADMAP_PROMPT] ---
        try:
            # ใช้ UB_ROADMAP_PROMPT ในการสร้าง Prompt String
            # หมายเหตุ: ถ้าใช้ LangChain สามารถใช้ .format() ได้เลย
            final_prompt_string = SUB_ROADMAP_PROMPT.format(
                sub_id=sub_id,
                sub_criteria_name=sub_criteria_name,
                enabler=enabler,
                aggregated_insights=enriched_context,
                strategic_focus=strategic_focus
            )

            # ส่งเข้า LLM (ส่งเป็น User Prompt เพราะ System Rules ถูกรวมเข้าไปใน String แล้ว)
            # หรือถ้า _fetch_llm_response รองรับแยก System สามารถตัดแบ่งได้
            raw = _fetch_llm_response(
                system_prompt="คุณคือที่ปรึกษาเชิงยุทธศาสตร์ด้าน SE-AM", 
                user_prompt=final_prompt_string, 
                llm_executor=self.llm
            )

            data = _robust_extract_json(raw) or {}
            raw_phases = data.get("phases") or []

            # --- [STEP 4: NORMALIZE & INJECT EVIDENCE] ---
            final_phases = []
            for i, p in enumerate(raw_phases, 1):
                actions = p.get("key_actions") or []
                
                # Injection Logic: ถ้าไม่มีชื่อไฟล์ใน Action ให้แทรกชื่อไฟล์จริงเข้าไป
                if top_asset_name and not any(top_asset_name in str(a) for a in actions):
                    actions.insert(0, {
                        "action": f"ต่อยอดมาตรฐานจาก {top_asset_name} เพื่อทำ Standardization",
                        "priority": "High"
                    })

                final_phases.append({
                    "phase": p.get("phase", f"Phase {i}"),
                    "target_levels": p.get("target_levels") or [min(highest_continuous + i, 5)],
                    "main_objective": p.get("main_objective") or "ยกระดับระบบงาน",
                    "key_actions": actions,
                    "expected_outcome": p.get("expected_outcome") or "เพิ่ม Traceability Score > 85%"
                })

            # บังคับ L5 Sustainability เสมอ
            if highest_continuous == 5 and len(final_phases) < 2:
                final_phases.append({
                    "phase": "Phase 2: Sustainability & Learning Culture",
                    "main_objective": "รักษามาตรฐาน Excellence และสร้างระบบ Knowledge Governance",
                    "key_actions": [
                        {"action": f"ถอดบทเรียนจาก {top_asset_name} เป็น Best Practice องค์กร", "priority": "High"}
                    ],
                    "expected_outcome": "Traceability 100%"
                })

            return {
                "scope": "SUB_CRITERIA",
                "sub_id": sub_id,
                "highest_maturity_level": highest_continuous,
                "overall_strategy": data.get("overall_strategy", strategic_focus),
                "phases": final_phases,
                "is_gap_detected": has_gap,
                "status": "SUCCESS"
            }

        except Exception as e:
            self.logger.error(f"🛑 UB_PROMPT Error: {str(e)}")
            return self._get_emergency_fallback_plan(sub_id, sub_criteria_name, str(e))
        
    def synthesize_enabler_roadmap(
        self,
        sub_criteria_results: List[Dict[str, Any]],
        enabler_name: str,
        llm_executor: Any
    ) -> Dict[str, Any]:
        """
        [TIER-3 STRATEGIC ORCHESTRATOR - FULL REVISE v2026.01.28]
        - 🧩 Macro-Synthesis: สังเคราะห์แผนภาพรวมจากทุก Sub-id
        - 🛡️ KeyError Shield: ใช้ STRATEGIC_OVERALL_PROMPT ที่แยกตัวแปรชัดเจน
        - 🚀 Performance: ปรับลดความซับซ้อนของ Context เพื่อให้ L40S ตอบสนองไวขึ้น
        """

        self.logger.info(f"🌐 [TIER-3] Synthesizing Strategic Master Plan for {enabler_name}")

        if not sub_criteria_results:
            return {
                "status": "INCOMPLETE",
                "overall_strategy": "ไม่พบข้อมูลเพียงพอในการสังเคราะห์แผนภาพรวม",
                "phases": []
            }

        # --- [STEP 1: DETERMINING GLOBAL MATURITY & FOCUS] ---
        # หาเลเวลต่ำสุดที่ทุกข้อผ่าน (Baseline)
        global_maturity = min(
            [int(r.get("highest_full_level", 0)) for r in sub_criteria_results if r],
            default=0
        )

        # กำหนด Strategic Focus สำหรับภาพรวมองค์กร
        if global_maturity < 3:
            global_focus = f"Focus: Foundational Integrity (การสถาปนารากฐานระบบ {enabler_name} และปิดช่องว่างมาตรฐาน)"
        elif 3 <= global_maturity < 5:
            global_focus = f"Focus: Strategic Integration (การเชื่อมโยงระบบ {enabler_name} เข้ากับเป้าหมายยุทธศาสตร์องค์กร)"
        else:
            global_focus = f"Focus: Excellence & Innovation (การสร้างนวัตกรรมและเป็นต้นแบบระดับสากล)"

        # --- [STEP 2: AGGREGATING MULTI-SUB GAPS] ---
        blocking_gaps = []
        key_strengths = []

        for res in sub_criteria_results:
            sid = res.get("sub_id", "N/A")
            sname = res.get("sub_criteria_name", "N/A")
            
            # เก็บจุดแข็ง (เลเวลสูงสุดที่ผ่าน)
            if res.get("highest_full_level", 0) > 0:
                key_strengths.append(f"- [{sid}] ผ่านระดับ L{res.get('highest_full_level')}: {sname}")

            # ดึงเฉพาะ Coaching Insight ของเลเวลที่ติดขัด (Next Target Level)
            next_lv = str(res.get("highest_full_level", 0) + 1)
            details = res.get("level_details", {}).get(next_lv)
            
            if details and not details.get("is_passed"):
                insight = details.get("coaching_insight", "").strip()
                if insight:
                    blocking_gaps.append(f"🔴 [{sid} L{next_lv}]: {insight[:200]}")

        # จำกัดปริมาณข้อมูลไม่ให้ LLM สับสน
        aggregated_context = (
            "💎 KEY STRENGTHS (จุดแข็งที่เป็นต้นทุน):\n" +
            ("\n".join(key_strengths[:5]) if key_strengths else "- อยู่ระหว่างการเริ่มต้น") +
            "\n\n🚨 CRITICAL BLOCKING GAPS (ช่องว่างรวมที่ต้องเร่งปิด):\n" +
            ("\n".join(blocking_gaps[:10]) if blocking_gaps else "- ไม่พบช่องว่างวิกฤต")
        )

        # --- [STEP 3: PROMPT ORCHESTRATION (THE FIX)] ---
        # 🛡️ ใช้ Prompt ตัวใหม่ที่รับ 3 ตัวแปร (ตรงกับ seam_prompts.py)
        final_prompt = STRATEGIC_OVERALL_PROMPT.format(
            enabler_name=enabler_name,
            aggregated_context=aggregated_context,
            strategic_focus=global_focus
        )

        try:
            # ใช้ helper _fetch_llm_response หรือเรียก llm_executor โดยตรง
            raw = _fetch_llm_response(
                system_prompt=SYSTEM_OVERALL_STRATEGIC_PROMPT,
                user_prompt=final_prompt,
                llm_executor=llm_executor
            )

            data = _robust_extract_json(raw) or {}
            raw_phases = data.get("phases") or data.get("roadmap") or []

            # --- [STEP 4: NORMALIZE PHASES] ---
            final_phases = []
            for i, p in enumerate(raw_phases, 1):
                if isinstance(p, dict):
                    final_phases.append({
                        "phase": p.get("phase") or f"Phase {i}: การยกระดับภาพรวม",
                        "target_levels": p.get("target_levels") or f"L{global_maturity + 1}",
                        "main_objective": p.get("main_objective") or p.get("target_objectives") or "ปิดช่องว่างเชิงยุทธศาสตร์",
                        "key_actions": p.get("key_actions") or p.get("strategic_actions") or [],
                        "expected_outcome": p.get("expected_outcome") or p.get("key_performance_indicator") or "ผ่านเกณฑ์มาตรฐานเพิ่มขึ้น"
                    })

            if not final_phases:
                final_phases = self._get_emergency_fallback_plan("OVERALL", enabler_name).get("phases", [])

            return {
                "status": "SUCCESS",
                "overall_strategy": data.get("overall_strategy") or global_focus,
                "phases": final_phases,
                "metadata": {
                    "global_maturity_baseline": global_maturity,
                    "applied_strategic_focus": global_focus,
                    "total_sub_evaluated": len(sub_criteria_results),
                    "generated_at": datetime.now().isoformat()
                }
            }

        except Exception as e:
            self.logger.error(f"🛑 [TIER-3-CRITICAL] Global Roadmap Error: {e}", exc_info=True)
            return {
                "status": "ERROR",
                "overall_strategy": "เกิดข้อผิดพลาดในการสังเคราะห์แผนยุทธศาสตร์",
                "reason": str(e),
                "phases": []
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
                "level_criteria": safe_criteria,
                "focus_points": kwargs.get('focus_points', 'พิจารณาตามเกณฑ์มาตรฐาน') # 👈 เพิ่มบรรทัดนี้
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
                # ดึงส่วนหนึ่งของ insight มาสร้าง action แบบฉุกเฉิน
                short_insight = clean_insight[:50] + "..."
                final_actions = [{
                    "action": f"ดำเนินการปิดช่องว่างตามข้อเสนอแนะ: {short_insight}",
                    "target_evidence": "เอกสาร/รายงานการดำเนินงานที่เกี่ยวข้อง",
                    "level": level
                }]

            self.logger.info(f"✅ [ATOMIC-PLAN] {enabler_code} L{level} Success (Output: {len(final_actions[:2])})")
            return final_actions[:2]

        except Exception as e:
            self.logger.error(f"🛑 [ATOMIC-PLAN-CRITICAL] {str(e)}", exc_info=True)
            return [{"action": "ดำเนินการพัฒนางานตามข้อเสนอแนะของเกณฑ์", "target_evidence": "หลักฐานเชิงประจักษ์", "level": level}]
        
    
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
        [REFINED] เพิ่มการเช็คความถูกต้องของข้อมูลก่อน Sort
        """
        if not evidence_list:
            return []

        # กรองเฉพาะที่เป็น dict และมีข้อมูลสำคัญ
        valid_evidences = [ev for ev in evidence_list if isinstance(ev, dict) and (ev.get('text') or ev.get('content'))]
        
        # 1. Deduplicate โดยใช้ Hash ของเนื้อหา (ถ้าทำได้) หรือ Source+Page
        unique_evidences = self._deduplicate_list(valid_evidences)

        # 2. Strategy-based Selection
        if EVIDENCE_SELECTION_STRATEGY == "score":
            # ให้ความสำคัญกับ rerank_score ก่อน ถ้าไม่มีให้ใช้ relevance_score
            sorted_list = sorted(
                unique_evidences, 
                key=lambda x: x.get('rerank_score') or x.get('relevance_score') or 0, 
                reverse=True
            )
        else:
            sorted_list = unique_evidences

        # 3. Apply Cap
        return sorted_list[:EVIDENCE_CUMULATIVE_CAP]
    
    def _run_sub_criteria_assessment_worker(
        self,
        sub_criteria: Dict[str, Any],
        vectorstore_manager: Optional[Any] = None,
        initial_baseline: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
        """
        [ULTIMATE REVISED v2026.02.01 - FULL UI SYNC & PDCA STATUS]
        - เพิ่ม rubric_statement เข้า level_details เพื่อ tooltip แสดงเกณฑ์จริง
        - เพิ่ม pdca_status (actual/missing phases, coverage %, is_full_coverage, status_label)
        - Logic PDCA แม่นยำขึ้น (ใช้ set + upper case + กรอง tag ถูกต้อง)
        - Logging เพิ่มเพื่อ debug statement + pdca_status
        - รักษา Maturity Ladder Lock + Gap Traceability เดิม
        """
        sub_id = str(sub_criteria.get("sub_id", "Unknown"))
        sub_name = sub_criteria.get("sub_criteria_name", "No Name")
        sub_weight = float(sub_criteria.get("weight", 0.0))
        target_limit = getattr(self.config, "target_level", 5)

        vsm = vectorstore_manager or self.vectorstore_manager
        level_details: Dict[str, Any] = {}
        roadmap_input_bundle = []

        highest_continuous_level = 0
        is_still_continuous = True  # ตัวแปรคุมบันได Maturity (ถ้าติด gap → False ตลอด)
        cumulative_baseline = list(initial_baseline or [])
        evidence_delta: Dict[str, List[Dict[str, Any]]] = {}

        levels = sorted(sub_criteria.get("levels", []), key=lambda x: x.get("level", 0))

        for stmt in levels:
            level = int(stmt.get("level", 0))
            if level == 0 or level > target_limit:
                continue

            level_key = f"{sub_id}_L{level}"
            required_pdca = self.get_rule_content(sub_id, level, "require_phase") or ["P"]

            # --- [STEP 1: EVIDENCE HYDRATION] ---
            saved = self.evidence_map.get(level_key, {})
            saved_evs = saved.get("evidences", []) if isinstance(saved, dict) else []
            priority_items = [e for e in saved_evs if e.get("is_selected", True)]
            current_baseline = self._deduplicate_list(cumulative_baseline + priority_items)

            # --- [STEP 2: CORE ASSESSMENT] ---
            res = self._run_single_assessment(
                sub_id=sub_id,
                level=level,
                criteria={"name": sub_name, "statement": stmt.get("statement", "")},
                keyword_guide=stmt.get("keywords", []),
                baseline_evidences=current_baseline,
                vectorstore_manager=vsm,
            )

            # --- [STEP 3: MATURITY STEP-LADDER LOGIC] ---
            llm_passed = bool(res.get("is_passed", False))
            actual_passed = llm_passed and is_still_continuous
            is_capped = llm_passed and not is_still_continuous

            if is_capped:
                self.logger.warning(f"[MATURITY-LOCK] {sub_id}_L{level} CAPPED: Previous level has unresolved GAP")

            if actual_passed:
                highest_continuous_level = level
                top_chunks = res.get("top_chunks_data", []) or []
                cumulative_baseline.extend(top_chunks)
                cumulative_baseline = self._apply_evidence_cap(cumulative_baseline)
            else:
                is_still_continuous = False

            # --- [STEP 4: SCORE & DATA ENRICHMENT] ---
            effective_score = float(res.get("score", 0.0)) if actual_passed else 0.0

            enriched_chunks = res.get("top_chunks_data", []) or []
            pdca_results = res.get("pdca_breakdown", {})

            try:
                atomic_actions = self.create_atomic_action_plan(
                    insight=res.get("coaching_insight", ""),
                    level=level,
                    level_criteria=stmt.get("statement", ""),
                    focus_points=sub_criteria.get("focus_points", "-")
                )
            except Exception as e:
                self.logger.warning(f"Atomic action failed for {sub_id}_L{level}: {e}")
                atomic_actions = []

            # --- [STEP 5: RESULT COMPILATION - REVISED FOR UI SYNC] ---
            # 1. รวบรวม PDCA Phases ที่พบจริงจาก evidence (แม่นยำที่สุด)
            actual_found_phases = set(
                chunk.get("pdca_tag", "").upper()
                for chunk in enriched_chunks
                if chunk.get("pdca_tag", "").upper() in ["P", "D", "C", "A"]
            )

            # 2. คำนวณ missing + coverage + status
            missing_phases = [p for p in required_pdca if p not in actual_found_phases]
            coverage_percentage = round(
                (len(actual_found_phases) / len(required_pdca)) * 100 if required_pdca else 0, 1
            )
            is_full_coverage = len(missing_phases) == 0

            level_details[str(level)] = {
                "level": level,
                "is_passed": actual_passed,
                "is_maturity_capped": is_capped,
                "score": round(effective_score, 2),
                "reason": res.get("reason", "ไม่มีเหตุผลระบุ"),
                "coaching_insight": res.get("coaching_insight", ""),
                "atomic_action_plan": atomic_actions,
                "evidence_sources": enriched_chunks,
                "pdca_breakdown": pdca_results,
                "required_pdca_phases": required_pdca,
                "rubric_statement": stmt.get("statement", "").strip() or "ไม่พบเกณฑ์ระดับนี้",

                # ข้อมูล PDCA Status สำหรับ UI (PDCA Matrix + Tooltip)
                "pdca_status": {
                    "actual_phases": list(actual_found_phases),
                    "missing_phases": missing_phases,
                    "coverage_percentage": coverage_percentage,
                    "is_full_coverage": is_full_coverage,
                    "status_label": "PASS" if is_full_coverage else "GAP"
                }
            }

            # Log เพื่อ debug PDCA + statement (ช่วยหาปัญหาได้เร็ว)
            self.logger.debug(
                f"[PDCA-STATUS] {sub_id}_L{level} | "
                f"Required: {required_pdca} | "
                f"Found: {actual_found_phases} | "
                f"Missing: {missing_phases} | "
                f"Coverage: {coverage_percentage}% | "
                f"Statement: {level_details[str(level)]['rubric_statement'][:80]}..."
            )

            # --- [STEP 5.1: ROADMAP BUNDLE - SIMPLE & TRACEABLE] ---
            if enriched_chunks:
                top_ev = max(enriched_chunks, key=lambda x: float(x.get("rerank_score") or x.get("relevance_score") or 0.0))
                top_file = top_ev.get("source_filename") or top_ev.get("filename") or "N/A"
                top_page = top_ev.get("page_label") or top_ev.get("page", "N/A")
                top_score = f"{float(top_ev.get('rerank_score') or top_ev.get('relevance_score') or 0.0):.4f}"
                top_pdca = top_ev.get("pdca_tag", "N/A")
                snippet = (top_ev.get("content") or top_ev.get("text") or "")[:150].replace("\n", " ").strip() + "..."

                insight_summary = (
                    f"L{level}: {'PASSED' if actual_passed else 'FAILED'} | "
                    f"Top: {top_file} (p.{top_page}) | Score: {top_score} | PDCA: {top_pdca} | "
                    f"Insight: {res.get('coaching_insight', 'N/A')[:200]}... | Snippet: {snippet}"
                )
            else:
                insight_summary = (
                    f"L{level}: {'PASSED' if actual_passed else 'FAILED'} | "
                    f"No evidence | Statement: {stmt.get('statement', '')[:200]} | "
                    f"Insight: {res.get('coaching_insight', 'N/A')[:150]}..."
                )

            roadmap_input_bundle.append({
                "level": level,
                "status": "PASSED" if actual_passed else ("CAPPED" if is_capped else "FAILED"),
                "statement": stmt.get("statement", "")[:400],
                "insight_summary": insight_summary
            })

            self.logger.debug(f"[ROADMAP-BUNDLE] {sub_id}_L{level} | {insight_summary[:400]}...")

        # --- [STEP 6: STRATEGIC FOCUS SELECTION] ---
        has_gap = any(not ld["is_passed"] for ld in level_details.values())

        if highest_continuous_level < 3:
            strategic_focus = "Focus: Stabilization (เน้นสถาปนามาตรฐานและปิด Gap ระดับฐาน)"
        elif highest_continuous_level < 5:
            strategic_focus = "Focus: Scaling & Integration (เน้นการเชื่อมโยงและขยายผล)"
        else:
            strategic_focus = (
                "Focus: Strategic Excellence & Sustainability (เน้นนวัตกรรมและความยั่งยืน)"
                if not has_gap else
                "Focus: Strategic Excellence (เน้นปิด gap ที่เหลือและยกระดับสู่ต้นแบบ)"
            )

        self.logger.info(f"[STRATEGIC-FOCUS] {sub_id} | Highest: L{highest_continuous_level} | Gap: {has_gap} | {strategic_focus}")

        sub_roadmap = self.generate_sub_roadmap(
            sub_id=sub_id,
            sub_criteria_name=sub_name,
            enabler=getattr(self, "enabler", "KM"),
            aggregated_insights=roadmap_input_bundle,
            strategic_focus=strategic_focus
        )

        return {
            "sub_id": sub_id,
            "sub_criteria_name": sub_name,
            "weight": sub_weight,
            "highest_full_level": highest_continuous_level,
            "weighted_score": round(highest_continuous_level * (sub_weight / 5), 2),
            "is_passed": highest_continuous_level >= 1,
            "level_details": level_details,
            "sub_roadmap": sub_roadmap,
            "strategic_focus": strategic_focus
        }, evidence_delta
            
    def _get_level_constraint_prompt(
        self,
        sub_id: str,
        level: int,
        req_phases: list | None = None,
        spec_rule: str | None = None
    ) -> str:
        """
        [FINAL LOCKED VERSION v2026.01.27]
        - PDCA Scope Guard (ไม่ over-require maturity)
        - Align กับ relevance_score_fn / cumulative rules
        - ป้องกัน LLM hallucinate SUPPORT / OTHER
        """

        enabler = getattr(self, "enabler", "KM").upper()
        enabler_name = "การจัดการความรู้ (KM)"

        # -------------------------------
        # 1. Level Goal & Focus
        # -------------------------------
        level_goal = get_pdca_goal_for_level(level)
        level_focus = PDCA_PHASE_MAP.get(level, "ตรวจสอบความครบถ้วนของระบบ")

        # -------------------------------
        # 2. Mandatory PDCA (Truth-based)
        # -------------------------------
        if req_phases:
            required_phases = list(dict.fromkeys(req_phases))
        else:
            if level >= 4:
                required_phases = ["P", "D", "C"]
            elif level >= 2:
                required_phases = ["P", "D"]
            else:
                required_phases = ["P"]

        req_str = " + ".join(required_phases)

        # -------------------------------
        # 3. Guardrails (Hard Rules)
        # -------------------------------
        strict_rules = [
            f"1. [PDCA Scope Lock] ให้พิจารณาเฉพาะ PDCA = {req_str} เท่านั้น ห้ามสร้างหมวดอื่น เช่น SUPPORT, OTHER, CONTEXT",
            f"2. [Maturity Guard] ห้ามใช้ข้อกำหนดของระดับสูงกว่า Level {level} (เช่น Automation, AI, Benchmarking) เป็นเหตุให้ 'ไม่ผ่าน' หากรูบริกไม่ได้ระบุ",
            "3. [Evidence Integrity] ห้ามสรุป PDCA Phase หากไม่สามารถอ้างอิงเอกสาร หลักฐาน หรือหน้า (page) ได้ชัดเจน",
            "4. [Policy Shortcut] หากพบเอกสารเชิงนโยบาย/แผนแม่บทที่ได้รับอนุมัติ ให้ถือว่าผ่าน Phase 'P' (และ 'D' หากมีแผนปฏิบัติ)",
            f"5. {GLOBAL_EVIDENCE_INSTRUCTION}",
            "6. [Substance First] ให้ดูเนื้อหาว่าตอบโจทย์เป้าหมายระดับนี้หรือไม่ มากกว่าชื่อไฟล์",
            "7. [Coaching Output] หากประเมินว่า 'ไม่ผ่าน' ต้องระบุสิ่งที่ขาดอย่างเป็นรูปธรรม เพื่อให้หน่วยงานปรับปรุงได้"
        ]

        # -------------------------------
        # 4. Prompt Assembly
        # -------------------------------
        lines = [
            f"\n### 🛡️ AUDIT GUIDELINE : {enabler} | LEVEL {level} ###",
            f"🎯 หัวข้อประเมิน: {enabler_name} ({sub_id})",
            f"📈 เป้าหมายระดับ: {level_goal}",
            f"🔍 PDCA ที่ต้องพบ (Mandatory): {req_str}",
            f"📌 จุดเน้นระดับนี้: {level_focus}",
            f"💡 เกณฑ์เฉพาะในรูบริก: {spec_rule}" if spec_rule else "",
            "\n⚠️ STRICT DECISION RULES:",
            *strict_rules
        ]

        return "\n".join(filter(None, lines))

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
        [ULTIMATE REVISED v2026.02.05-enhanced]
        - Governance: ฉีด audit_instruction คุมขอบเขต
        - Retrieval: Adaptive + Neighbor Expansion + Diversity Filter
        - Resilience: Judicial Review (อุทธรณ์) เมื่อ Rerank สูงแต่ LLM มองไม่เห็น
        - Logging: เพิ่ม debug appeal trigger/success + fallback JSON
        - ไม่เปลี่ยน signature หรือ flow เดิม
        """
        log_prefix = f"Sub:{sub_id} L{level}"
        sub_name = criteria.get('name', 'Unknown Sub-item')
        statement_text = criteria.get('statement', 'No statement defined')
        
        self.logger.info(f"🔍 [START-ASSESSMENT] {log_prefix} | {sub_name}")
        self.logger.info(f"📋 [CRITERIA] Level {level}: \"{statement_text}\"")

        # --- [STEP 1: GOVERNANCE & RULES] ---
        audit_instruction = self._get_level_constraint_prompt(sub_id, level)
        current_rules = getattr(self, 'contextual_rules_map', {}).get(sub_id, {}).get(f"L{level}", {})

        # --- [STEP 2: ADAPTIVE RETRIEVAL & EXPANSION] ---
        retrieved_chunks, max_rerank = self._perform_adaptive_retrieval(
            sub_id=sub_id,
            level=level,
            stmt=statement_text,
            vectorstore_manager=vectorstore_manager,
        )

        if retrieved_chunks:
            enabler_key = str(getattr(self, 'enabler', 'km')).lower()
            collection_name = f"evidence_{enabler_key}"
            retrieved_chunks = self._expand_context_with_neighbor_pages(
                top_evidences=retrieved_chunks, 
                collection_name=collection_name
            )

        retrieved_chunks = self._apply_diversity_filter(retrieved_chunks, level)

        # --- [STEP 3: EVIDENCE FUSION & METADATA] ---
        evidences = (baseline_evidences or []) + (retrieved_chunks or [])
        pdca_blocks = self._get_pdca_blocks_from_evidences(
            evidences=evidences,
            baseline_evidences=baseline_evidences,
            level=level,
            sub_id=sub_id,
            contextual_rules_map=getattr(self, 'contextual_rules_map', {})
        )

        audit_confidence = self.calculate_audit_confidence(
            matched_chunks=retrieved_chunks,
            sub_id=sub_id,
            level=level,
        )
        self.current_audit_meta = audit_confidence

        # --- [STEP 4: MULTICHANNEL LLM EXECUTION] ---
        llm_context = self._build_multichannel_context_for_level(
            level=level,
            top_evidences=retrieved_chunks,
            previous_levels_evidence=baseline_evidences
        )

        llm_raw = self.evaluate_pdca(
            pdca_blocks=pdca_blocks,
            sub_id=sub_id,
            level=level,
            audit_confidence=audit_confidence,
            audit_instruction=audit_instruction
        )
        if not isinstance(llm_raw, dict): 
            llm_raw = {}

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
        if not result.get("is_passed") and max_rerank >= 0.70:
            self.logger.info(f"⚖️ [TRIGGER-APPEAL] {log_prefix} | Rerank {max_rerank:.4f} HIGH → Re-evaluating... (Missing: {result.get('missing_phases', [])})")

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
                    "audit_instruction": audit_instruction,
                    "audit_confidence": audit_confidence
                }
            )

            if appeal_result and appeal_result.get("appeal_status") == "GRANTED":
                self.logger.info(f"✅ [APPEAL-GRANTED] {log_prefix} | Expert Rescue OK! (New Score: {appeal_result.get('score', 'N/A')})")
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
            else:
                self.logger.info(f"❌ [APPEAL-DENIED/ERROR] {log_prefix} | ไม่ผ่านอุทธรณ์")

        # --- [STEP 7: FINAL INSIGHTS & LOGGING] ---
        final_insight = (result.get("coaching_insight") or result.get("reason") or "").strip()
        final_insight = f"[{'STRENGTH' if result.get('is_passed') else 'GAP'}] {final_insight}"

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