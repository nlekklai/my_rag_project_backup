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


# -------------------- PATH SETUP & IMPORTS --------------------
try:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    # 1. Import Constants จาก global_vars
    from config.global_vars import (
        EXPORTS_DIR, MAX_LEVEL, INITIAL_LEVEL, QA_FINAL_K,
        RUBRIC_FILENAME_PATTERN, DEFAULT_ENABLER,
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
        OLLAMA_MAX_RETRIES,
        CONTEXT_CAP_L3_PLUS,
        CRITICAL_CA_THRESHOLD,
        MAX_RETRIEVAL_ATTEMPTS,
        HYBRID_VECTOR_WEIGHT,
        HYBRID_BM25_WEIGHT,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        REQUIRED_PDCA,
        CORRECT_PDCA_SCORES_MAP,
        PDCA_PHASE_MAP,        # ✅ ย้ายมาไว้ที่นี่ตามที่มีใน global_vars.py
        PDCA_PRIORITY_ORDER,
        BASE_PDCA_KEYWORDS,
        PDCA_LEVEL_SYNONYMS,
        ENABLE_HARD_FAIL_LOGIC,
        ENABLE_CONTEXTUAL_RULE_OVERRIDE,
        TARGET_SCORE_THRESHOLD_MAP
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
    
    # ❌ ลบบรรทัดเดิมที่เป็นสาเหตุของ ImportError ออก:
    # from core.seam_prompts import PDCA_PHASE_MAP 
    
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
    contextual_rules_map: dict = None
) -> str:
    """
    Heuristic PDCA Classification v18 (Supports Array/List from JSON)
    --------------------------------------------------
    - รองรับ Keywords ทั้งรูปแบบ String และ List
    - แก้ไข Error: 'list' object has no attribute 'split'
    - ปรับปรุงการเข้าถึง Defaults ให้แม่นยำขึ้น
    """
    if not text or not contextual_rules_map:
        return 'Other'

    text_lower = text.lower()

    def keyword_match(text_to_search: str, keywords_input) -> bool:
        """
        ฟังก์ชันช่วยเช็คการ Match คำ รองรับทั้ง String และ List
        """
        # แปลง input ให้เป็น list เสมอ
        keywords_list = []
        if isinstance(keywords_input, list):
            keywords_list = keywords_input
        elif isinstance(keywords_input, str):
            keywords_list = [k.strip() for k in keywords_input.split(",") if k.strip()]
        
        for kw in keywords_list:
            kw_clean = str(kw).strip().lower()
            if not kw_clean:
                continue
            
            # ตรวจสอบว่าเป็นภาษาไทยหรือไม่
            is_thai = any("\u0e00" <= c <= "\u0e7f" for c in kw_clean)
            
            if is_thai:
                pattern = re.escape(kw_clean)
            else:
                pattern = r'\b{}\b'.format(re.escape(kw_clean))
                
            if re.search(pattern, text_to_search, re.IGNORECASE):
                return True
        return False

    def check_level_keywords(l_rules: dict) -> str:
        """ตรวจหา P, D, C, A จากชุดกฎ โดยรองรับทั้ง List และ String"""
        mapping = {
            "plan_keywords": "P",
            "do_keywords": "D",
            "check_keywords": "C",
            "act_keywords": "A"
        }
        for json_key, tag in mapping.items():
            kw_data = l_rules.get(json_key)
            if kw_data:
                # 🎯 จุดที่แก้ไข: ใช้ keyword_match ที่รองรับ List แทนการ split เอง
                if keyword_match(text_lower, kw_data):
                    return tag
        return None

    # --- Step 1: ตรวจสอบตาม Level ปัจจุบัน ---
    if sub_id and level:
        rules = contextual_rules_map.get(sub_id, {})
        current_level_rules = rules.get(f"L{level}", {})
        if isinstance(current_level_rules, dict):
            tag = check_level_keywords(current_level_rules)
            if tag:
                # ตรวจสอบเงื่อนไขบังคับ
                must_include = rules.get("must_include_keywords", [])
                avoid_kw = rules.get("avoid_keywords", [])
                if must_include and not keyword_match(text_lower, must_include):
                    return 'Other'
                if avoid_kw and keyword_match(text_lower, avoid_kw):
                    return 'Other'
                return tag

    # --- Step 2: วนหาจากทุก Level ใน Sub-ID ---
    if sub_id:
        rules = contextual_rules_map.get(sub_id, {})
        for l_key, l_rules in rules.items():
            if l_key.startswith("L") and isinstance(l_rules, dict):
                tag = check_level_keywords(l_rules)
                if tag:
                    must_include = rules.get("must_include_keywords", [])
                    avoid_kw = rules.get("avoid_keywords", [])
                    if must_include and not keyword_match(text_lower, must_include):
                        continue
                    if avoid_kw and keyword_match(text_lower, avoid_kw):
                        continue
                    return tag

    # --- Step 3: ตรวจสอบจากค่าเริ่มต้น (_enabler_defaults) ---
    defaults = contextual_rules_map.get("_enabler_defaults", {})
    mapping_defaults = {
        "plan_keywords": "P", # แก้จาก plann_keywords
        "do_keywords": "D", 
        "check_keywords": "C", 
        "act_keywords": "A"
    }
    for json_key, tag in mapping_defaults.items():
        kw_data = defaults.get(json_key)
        if kw_data:
            if keyword_match(text_lower, kw_data):
                return tag

    # --- Step 4: Fallback สุดท้าย ---
    try:
        from config.global_vars import PDCA_PRIORITY_ORDER, BASE_PDCA_KEYWORDS
        tag_map = {"Plan": "P", "Do": "D", "Check": "C", "Act": "A"}
        for full_tag in PDCA_PRIORITY_ORDER: 
            patterns = BASE_PDCA_KEYWORDS.get(full_tag, [])
            if patterns and keyword_match(text_lower, patterns):
                return tag_map.get(full_tag, 'Other')
    except Exception as e:
        if 'logger' in globals():
            logger.error(f"Error in classify_by_keyword fallback: {e}")

    return 'Other'


def get_actual_score(ev: dict) -> float:
    """
    Unified score resolver (ENGINE SOURCE OF TRUTH)
    Priority:
    1) relevance_score
    2) rerank_score
    3) score
    (fallback to metadata)
    """
    if not ev:
        return 0.0

    score = ev.get("relevance_score") or ev.get("rerank_score") or ev.get("score")
    if score is not None:
        return float(score)

    meta = ev.get("metadata", {}) or {}
    return float(
        meta.get("relevance_score")
        or meta.get("rerank_score")
        or meta.get("score")
        or 0.0
    )

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


def post_process_llm_result(llm_output: Dict[str, Any], level: int) -> Dict[str, Any]:
    """
    POST-PROCESSOR v21.9.3 — REGEX FALLBACK & PDCA SAFEGUARD
    """

    extraction_map = {"Extraction_P": "P_Plan_Score", "Extraction_D": "D_Do_Score", 
                      "Extraction_C": "C_Check_Score", "Extraction_A": "A_Act_Score"}
    phase_map = {"Extraction_P": "Plan", "Extraction_D": "Do", "Extraction_C": "Check", "Extraction_A": "Act"}

    reason_text = llm_output.get("reason", "") or ""

    for ext_key, score_key in extraction_map.items():
        val = llm_output.get(ext_key)
        
        # 1. Fallback: ถ้าค่าว่าง ให้ใช้ Regex ดึงจาก Reason (รองรับทั้ง "หลักฐาน P:" และ "หลักฐาน Plan:")
        if val is None or str(val).strip() in ["", "-"]:
            p_full = phase_map[ext_key]
            p_char = p_full[0]
            
            # Regex กวาดข้อมูลจนกว่าจะเจอ Marker ตัวถัดไป หรือคำสรุป
            pattern = rf"หลักฐาน\s*({p_full}|{p_char})\s*:\s*(.*?)(?=หลักฐาน|สรุปภาพรวม|is_passed|score|\Z)"
            match = re.search(pattern, reason_text, re.DOTALL | re.IGNORECASE)
            if match:
                val = match.group(2).strip()
                logger.info(f" 🛡️ [Regex Match] Recovered {p_full} from reason text")

        raw_val = str(val or "-").strip()
        current_score = float(llm_output.get(score_key, 0))
        
        logger.info(f" 📦 [Content Check] Phase: {phase_map[ext_key]} | Score: {current_score} | Raw: '{raw_val[:80]}...'")

        # 2. Validation Logic
        has_file_ref = any(x in raw_val.lower() for x in [".pdf", ".docx", "source", "["])
        content_stripped = re.sub(r'[\[\]\(\)\-\|\:\s\_\#\*\"]', '', raw_val)
        is_too_short = len(content_stripped) < 10

        # 3. Smart Penalty: มีไฟล์แต่เนื้อหาสั้นไป (ปรับเหลือ 0.5)
        if has_file_ref and is_too_short and current_score > 0.5:
            llm_output[score_key] = 0.5
            current_score = 0.5

        # 4. Safeguard: ถ้าไม่มีหลักฐานเลยแต่ได้คะแนน -> ริบคะแนนเป็น 0
        if not has_file_ref and is_too_short and current_score > 0:
            # ตรวจสอบ Keyword อีกครั้งใน reason (Heuristic Rescue)
            if not any(kw.lower() in reason_text.lower() for kw in ["พบ", "สอดคล้อง", "มีแผน"]):
                logger.warning(f" 🚨 [Safeguard] L{level}: {score_key} revoked (No evidence found)")
                llm_output[score_key] = 0.0

    # 5. Final Score Calculation (SE-AM Logic)
    scores = { k: round(min(max(float(llm_output.get(k, 0)), 0.0), 2.0), 1) 
              for k in ["P_Plan_Score", "D_Do_Score", "C_Check_Score", "A_Act_Score"] }
    
    total_score = sum(scores.values())
    threshold = {1: 1, 2: 2, 3: 4, 4: 6, 5: 8}.get(level, 2)
    is_passed = total_score >= threshold

    # Hard-fail rules
    if is_passed:
        if level == 3 and scores["C_Check_Score"] <= 0: is_passed = False
        elif level == 4 and scores["A_Act_Score"] <= 0: is_passed = False

    llm_output.update({
        "score": total_score, **scores, "is_passed": is_passed, "normalized": True
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
        # =======================================================
        # 1. Logger & ActionPlan Setup
        # =======================================================
        # ดึงค่า doc_type มาเช็คก่อนเพื่อตั้งชื่อ Logger
        clean_dt = str(doc_type or getattr(config, 'doc_type', EVIDENCE_DOC_TYPES)).strip().lower()
        log_year = config.year if clean_dt == EVIDENCE_DOC_TYPES.lower() else "general"

        if logger_instance is not None:
            self.logger = logger_instance
        else:
            # เปลี่ยน config.year เป็น log_year เพื่อความชัดเจน
            self.logger = logging.getLogger(__name__).getChild(
                f"Engine|{config.enabler}|{config.tenant}/{log_year}"
            )

        # ปรับ Log ตอนเริ่มต้นด้วย
        self.logger.info(f"Initializing SEAMPDCAEngine for {config.enabler} ({config.tenant}/{log_year})")

        # =======================================================
        # 2. Core Configuration & Safety First
        # =======================================================
        self.config = config
        self.enabler_id = config.enabler
        self.target_level = config.target_level
        self.sub_id = sub_id
        self.llm = llm_instance
        self.vectorstore_manager = vectorstore_manager

        # ✅ [CRITICAL FIX] ประกาศ doc_type ทันทีตรงนี้ เพื่อให้ Worker มองเห็นแน่นอน
        # ไม่ว่าจะรัน Parallel หรือมี document_map ส่งเข้ามาหรือไม่ก็ตาม
        self.doc_type = doc_type or getattr(config, 'doc_type', EVIDENCE_DOC_TYPES)

        # --- [CRITICAL LOADING] ---
        self.rubric = self._load_rubric()
        self.retry_policy = RetryPolicy(
            max_attempts=3,
            base_delay=2.0,
            jitter=True,
            escalate_context=True,
            shorten_prompt_on_fail=True,
            exponential_backoff=True,
        )

        self.is_sequential = getattr(config, 'force_sequential', True)
        self.is_parallel_all_mode = is_parallel_all_mode
        self.required_pdca_map = REQUIRED_PDCA
        self.base_pdca_keywords = BASE_PDCA_KEYWORDS
        self.RERANK_THRESHOLD: float = RERANK_THRESHOLD
        self.MAX_EVI_STR_CAP: float = MAX_EVI_STR_CAP

        # =======================================================
        # 3. Persistent Evidence Mapping (ฉบับปรับปรุง)
        # =======================================================
        
        clean_dt = str(self.doc_type).strip().lower()
        self.evidence_map = {}
        self.evidence_map_path = None # ตั้งค่า Default เป็น None ไว้ก่อน

        if clean_dt == EVIDENCE_DOC_TYPES.lower():
            # 🎯 สร้าง Path เฉพาะเมื่อต้องใช้งาน Evidence Mode เท่านั้น
            self.evidence_map_path = evidence_map_path or get_evidence_mapping_file_path(
                tenant=self.config.tenant, 
                year=self.config.year, 
                enabler=self.enabler_id
            )
            self.evidence_map = self._load_evidence_map()
            self.logger.info(f"📊 Evidence mode: Loaded {len(self.evidence_map)} mapping keys.")
        else:
            # 🎯 โหมด Document จะไม่เรียกฟังก์ชันที่ต้องมุดลงไปใน folder ปี
            self.logger.info(f"📄 Document mode: Skipping heavy evidence mapping load (Speed Optimized).")

        self.contextual_rules_map = self._load_contextual_rules_map()
        self.temp_map_for_save = {}

        # =======================================================
        # 4. Document Map Loading (Dynamic Logic)
        # =======================================================
        map_to_use: Dict[str, str] = document_map or {}

        if not map_to_use:
            # ใช้ self.doc_type ที่เราประกาศไว้ชัวร์ๆ ด้านบน
            clean_dt = str(self.doc_type).strip().lower()
            
            if clean_dt == EVIDENCE_DOC_TYPES.lower():
                mapping_path = get_mapping_file_path(self.doc_type, tenant=self.config.tenant, year=self.config.year, enabler=self.enabler_id)
            else:
                mapping_path = get_mapping_file_path(self.doc_type, tenant=self.config.tenant)

            self.logger.info(f"🎯 Loading {clean_dt} mapping from: {mapping_path}")

            try:
                if os.path.exists(mapping_path):
                    with open(mapping_path, 'r', encoding='utf-8') as f:
                        doc_map_raw = json.load(f)
                    map_to_use = {doc_id: data.get("file_name", doc_id) for doc_id, data in doc_map_raw.items()}
                    self.logger.info(f"Loaded {len(map_to_use)} mappings.")
                else:
                    self.logger.warning(f"File not found: {mapping_path}")
            except Exception as e:
                self.logger.error(f"Failed to load document map: {e}")

        self.doc_id_to_filename_map = map_to_use
        self.document_map = map_to_use

        # =======================================================
        # 5. Lazy Initialization (VSM & LLM)
        # =======================================================
        # ตอนนี้ _initialize_vsm_if_none จะทำงานได้ราบรื่นเพราะมี self.doc_type แล้ว
        if self.llm is None: self._initialize_llm_if_none()
        if self.vectorstore_manager is None: self._initialize_vsm_if_none()

        if self.vectorstore_manager and not getattr(self.vectorstore_manager, '_doc_id_mapping', None):
            self.vectorstore_manager._load_doc_id_mapping()

        # =======================================================
        # 6. Function Pointers
        # =======================================================
        self.llm_evaluator = evaluate_with_llm
        self.rag_retriever = retrieve_context_with_filter
        self.create_structured_action_plan = create_structured_action_plan
        self.ActionPlanActions = ActionPlanActions

        self.logger.info(f"✅ Engine initialized: Enabler={self.enabler_id}, DocType={self.doc_type}")

    def get_rule_content(self, sub_id: str, level: int, key_type: str):
        """
        [NEW] ดึงข้อมูลจาก Contextual Rules รองรับโครงสร้าง Nested L1-L5
        key_type: 'plan_keywords', 'do_keywords', 'specific_contextual_rule', 'must_include_keywords'
        """
        rule = self.contextual_rules_map.get(sub_id, {})
        level_key = f"L{level}"
        
        # 1. ค้นหาใน Level เจาะจงก่อน (L1, L2...)
        level_data = rule.get(level_key, {})
        if key_type in level_data:
            return level_data[key_type]
        
        # 2. Fallback ไปที่ Root ของข้อนั้นๆ
        if key_type in rule:
            return rule[key_type]
        
        # 3. Fallback ไปที่ Defaults กลาง (ดึงจาก _enabler_defaults ใน JSON)
        if "keywords" in key_type:
            return self.contextual_rules_map.get("_enabler_defaults", {}).get(key_type, [])
            
        return ""

    def enhance_query_for_statement(
        self,
        statement_text: str,
        sub_id: str,
        statement_id: str,
        level: int,
        focus_hint: str,
        additional_context: Dict[str, Any] = None
    ) -> List[str]:
        """
        [REVISED v21.4 - FULL]
        สร้างชุดคำค้นหา (Search Queries) โดยดึงวัตถุดิบจาก Focus Points และ Guidelines 
        เพื่อเพิ่มความแม่นยำในการเข้าถึงเอกสารหลักฐานจริง
        """
        logger = logging.getLogger(__name__)
        enabler_id = self.enabler_id
        additional_context = additional_context or {}
        
        # --- 1. เตรียมวัตถุดิบ (Keywords & Context) ---
        # ดึง Focus Points และ Guideline จาก Context
        focus_points = additional_context.get('focus_points', [])
        guideline = additional_context.get('guideline', "")
        
        # ดึง Keywords ตามมาตรฐาน PDCA (สะสมตาม Level)
        raw_keywords = []
        # ต้องมี (Must include)
        must_list = self.get_rule_content(sub_id, level, "must_include_keywords")
        if isinstance(must_list, list): raw_keywords.extend(must_list)
        
        # คำสำคัญตามระดับ Maturity (Plan/Do/Check/Act)
        if level >= 1: raw_keywords.extend(self.get_rule_content(sub_id, 1, "plan_keywords") or [])
        if level >= 3: raw_keywords.extend(self.get_rule_content(sub_id, 3, "check_keywords") or [])
        
        # รวม Focus Points เข้าไปในก้อน Keywords
        if isinstance(focus_points, list):
            raw_keywords.extend(focus_points)

        # Clean & Deduplicate
        clean_keywords = list(dict.fromkeys([str(k).strip() for k in raw_keywords if k]))
        keywords_str = " ".join(clean_keywords[:12]) # ใช้สูงสุด 12 คำเพื่อไม่ให้ Query ยาวเกินไป

        # --- 2. สร้างชุด Queries แบบมุ่งเป้า (Targeted Queries) ---
        queries = []

        # Query 1: Focus-Core Query (ผสม Enabler + Sub-ID + Focus Points)
        # ตัวอย่าง: "KM 2.2 การจัดสรรงบประมาณ ระบบ IT Infrastructure"
        queries.append(f"{enabler_id} {sub_id} {keywords_str}")

        # Query 2: Evidence-Type Query (เน้นตาม Guideline หรือ Maturity)
        # ตัวอย่าง: "รายงานการประชุม คณะทำงาน KM" หรือ "ระเบียบปฏิบัติ นโยบาย"
        if guideline:
            queries.append(f"{guideline} {sub_id} {enabler_id} {keywords_str}")
        else:
            if level <= 2:
                queries.append(f"ประกาศ คำสั่ง ระเบียบปฏิบัติ แผนงาน หลักเกณฑ์ {sub_id} {keywords_str}")
            else:
                queries.append(f"รายงานผลการดำเนินงาน สรุปผลการประเมิน บันทึกข้อความ {sub_id} {keywords_str}")

        # Query 3: Statement-Based (ใช้ข้อความเกณฑ์โดยตรง + Focus Hint)
        # ตัวอย่าง: "กำหนดหลักเกณฑ์การจัดสรรงบประมาณ L2 KM"
        queries.append(f"{statement_text} {sub_id} {focus_hint}")

        # Query 4: PDCA & Result (สำหรับระดับสูง L4-L5)
        if level >= 4:
            queries.append(f"การปรับปรุง พัฒนาต่อเนื่อง นวัตกรรม Best Practice {sub_id} {keywords_str}")

        # --- 3. Finalize & Limit ---
        # ลบ Query ที่ซ้ำกันและลบช่องว่าง
        final_queries = []
        seen = set()
        for q in queries:
            q_clean = " ".join(q.split())
            if q_clean and q_clean not in seen:
                final_queries.append(q_clean)
                seen.add(q_clean)

        logger.info(f"🚀 Enhanced {len(final_queries[:5])} queries for {sub_id} L{level} (Context: {len(clean_keywords)} keywords)")
        return final_queries[:5] # ส่งออกสูงสุด 5 Queries

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

    def _expand_context_with_neighbor_pages(self, top_evidences: List[Any], collection_name: str) -> List[Any]:
        """
        [NEW] ตรวจสอบหลักฐาน หากพบ Keyword ของ 'Check' 
        จะทำการดึงหน้าถัดไป (Act) มาเสริมให้อัตโนมัติ
        """
        if not self.vectorstore_manager or not top_evidences:
            return top_evidences

        expanded_evidences = list(top_evidences)
        seen_pages = set()
        
        # Keywords สำหรับ Trigger (ถ้าเจอคำพวกนี้ ให้ไปดึงหน้าถัดไปทันที)
        check_triggers = ["ความพึงพอใจ", "คะแนน", "สรุปผล", "ผลการดำเนินงาน", "score", "kpi", "3.41"]

        for doc in top_evidences:
            # ดึงเนื้อหา (Engine นี้ใช้ dict ที่มี key 'text' หลังผ่าน Hydration)
            text = (doc.get('text') or doc.get('page_content') or "").lower()
            
            # ดึง Metadata
            meta = doc.metadata if hasattr(doc, 'metadata') else doc.get('metadata', {})
            page_label = meta.get("page_label")
            doc_uuid = meta.get("stable_doc_uuid")

            # เงื่อนไข: เจอ Keyword + มีเลขหน้า + ยังไม่เคยดึงหน้านี้มาเสริมในรอบนี้
            if any(k in text for k in check_triggers) and str(page_label).isdigit():
                next_page = str(int(page_label) + 1)
                cache_key = f"{doc_uuid}_{next_page}"

                if cache_key not in seen_pages:
                    self.logger.info(f"🔍 Act-Hook: พบข้อมูล Check ที่หน้า {page_label}, กำลังดึงหน้า {next_page}...")
                    
                    # เรียก Method จาก Step 1 (ใน vectorstore.py)
                    neighbor_chunks = self.vectorstore_manager.get_chunks_by_page(
                        collection_name=collection_name,
                        stable_doc_uuid=doc_uuid,
                        page_label=next_page
                    )

                    if neighbor_chunks:
                        for nc in neighbor_chunks:
                            # สร้าง dict ใหม่ให้โครงสร้างเหมือนกับ top_evidences เดิม
                            new_doc = {
                                "text": f"[Supplemental Context - Next Page {next_page} for Act analysis]:\n{nc.page_content}",
                                "page_content": nc.page_content,
                                "metadata": nc.metadata,
                                "pdca_tag": "Act", # บังคับเป็น Act เพื่อให้ถูกจัดกลุ่มลงใน act_blocks
                                "is_supplemental": True,
                                "rerank_score": doc.get('rerank_score', 0.0) # ใช้ score เดิมเพื่อให้ผ่าน Filter
                            }
                            expanded_evidences.append(new_doc)
                        seen_pages.add(cache_key)

        return expanded_evidences
    
    
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
                    # ถ้ามีเลขหน้าเดิมอยู่แล้ว ให้ถือว่ายังเป็นประโยชน์แม้ไม่มี Text เต็ม
                    new_ev["is_baseline"] = False
                    if "page" not in new_ev:
                         new_ev["page"] = ev.get("page") # Fallback ไปที่ค่าเดิมใน evidence_map
                
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

                                if "page" not in new_entry or new_entry["page"] in ["N/A", None]:
                                    if "page" in old_entry:
                                        new_entry["page"] = old_entry["page"]
                                        
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
        Calculates overall statistics from sub-criteria results 
        and stores them in self.total_stats.
        """
        from config.global_vars import MAX_LEVEL
        
        results = self.final_subcriteria_results
        
        # ---------------------------------------------------------
        # 1. กรณีไม่มีผลลัพธ์ (Safety Guard)
        # ---------------------------------------------------------
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
                "status": "No Data"
            }
            return

        # ---------------------------------------------------------
        # 2. คำนวณผลรวมคะแนน (Summation)
        # ---------------------------------------------------------
        # weighted_score คือ (Level / 5) * Weight ของข้อนั้นๆ
        total_weighted_score_achieved = sum(r.get('weighted_score', 0.0) for r in results)
        
        # total_possible_weight คือ ผลรวมน้ำหนักของ Sub-criteria ที่ถูกประเมินในรอบนี้
        # เช่น ถ้าประเมินแค่ 1.2 จะได้ 4.0 แต่ถ้าประเมินทั้ง Enabler จะได้ 40.0
        total_possible_weight = sum(r.get('weight', 0.0) for r in results)

        # ---------------------------------------------------------
        # 3. คำนวณ Maturity Score เฉลี่ย (0.0 - 5.0)
        # ---------------------------------------------------------
        overall_avg_score = 0.0
        if total_possible_weight > 0:
            # สูตร: คะแนนรวมที่ได้ / น้ำหนักรวม = ค่าเฉลี่ยเลเวล (1-5)
            overall_avg_score = (total_weighted_score_achieved / total_possible_weight) * MAX_LEVEL
            overall_avg_score = round(overall_avg_score, 2) 
        
        # ---------------------------------------------------------
        # 4. คำนวณ Progress (%) เทียบกับคะแนนเต็ม (Max Possible)
        # ---------------------------------------------------------
        overall_progress_percentage = 0.0
        # เพดานคะแนนสูงสุดที่ควรจะได้ (Weight รวม * 5)
        max_possible_points = total_possible_weight * MAX_LEVEL
        
        if max_possible_points > 0:
            overall_progress_percentage = total_weighted_score_achieved / max_possible_points
            overall_progress_percentage = round(overall_progress_percentage, 4)

        # ---------------------------------------------------------
        # 5. กำหนด Label ของ Maturity Level (L1 - L5)
        # ---------------------------------------------------------
        # ใช้เกณฑ์การปัดเศษ (Round) เพื่อหาค่า Level ที่ใกล้เคียงที่สุด
        highest_level_achieved = round(overall_avg_score)
        final_level = min(max(int(highest_level_achieved), 0), MAX_LEVEL)
        overall_level_label = f"L{final_level}"
        
        # ---------------------------------------------------------
        # 6. สรุปเปอร์เซ็นต์ความสำเร็จ (0-100%)
        # ---------------------------------------------------------
        percentage_achieved_run = round(overall_progress_percentage * 100, 1)

        # ---------------------------------------------------------
        # 7. บันทึกค่าลงใน stats object
        # ---------------------------------------------------------
        self.total_stats = {
            "Overall Maturity Score (Avg.)": overall_avg_score,
            "Overall Maturity Level (Weighted)": overall_level_label,
            "Number of Sub-Criteria Assessed": len(results),
            "Total Weighted Score Achieved": round(total_weighted_score_achieved, 2),
            "Total Possible Weight": total_possible_weight,
            "Overall Progress Percentage (0.0 - 1.0)": overall_progress_percentage,
            "percentage_achieved_run": percentage_achieved_run,
            "total_subcriteria": len(self._flatten_rubric_to_statements()),
            "target_level": self.config.target_level,
            "enabler": self.config.enabler,
            "sub_criteria_id": target_sub_id,
            "gap_to_full_score": round(total_possible_weight - total_weighted_score_achieved, 2)
        }
        
        self.logger.info(f"--- ASSESSMENT SUMMARY ---")
        self.logger.info(f"Enabler: {self.config.enabler} | Sub: {target_sub_id}")
        self.logger.info(f"Maturity: {overall_level_label} (Avg Score: {overall_avg_score})")
        self.logger.info(f"Score: {total_weighted_score_achieved}/{total_possible_weight} ({percentage_achieved_run}%)")
        self.logger.info(f"---------------------------")

    def _export_results(self, results: dict, sub_criteria_id: str, **kwargs) -> str:
        """
        Exports the assessment results to a JSON file.
        Includes enhanced summary stats and persistence support with record_id.
        """
        # --- 0. ดึง record_id จาก kwargs (ส่งมาจาก run_assessment) ---
        record_id = kwargs.get("record_id", "no_id")

        enabler = self.enabler_id
        target_level = self.config.target_level
        tenant = self.config.tenant
        year = self.config.year
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # --- 1. การจัดการ Path และชื่อไฟล์ (เพิ่ม record_id เพื่อให้ Router หาเจอ) ---
        # ปรับ Suffix ให้มี ID: assessment_results_b0deb65560b3_1.1_20251218...
        suffix = f"results_{record_id}_{sub_criteria_id}_{timestamp}"

        full_path = ""
        export_dir = ""

        try:
            if self.config.export_path:
                export_dir = self.config.export_path
                file_name = f"assessment_{enabler}_{record_id}_{sub_criteria_id}_{timestamp}.json"
                full_path = os.path.join(export_dir, file_name)
            else:
                # ใช้ utility ช่วยสร้าง path โดยใส่ suffix ที่มี record_id
                full_path = get_assessment_export_file_path(
                    tenant=tenant,
                    year=year,
                    enabler=enabler,
                    suffix=suffix,
                    ext="json"
                )
                export_dir = os.path.dirname(full_path)

        except Exception as e:
            self.logger.warning(f"⚠️ Path utility failed, using fallback: {e}")
            export_dir = os.path.join("data_store", tenant, "exports", str(year), enabler)
            file_name = f"assessment_{enabler}_{record_id}_{sub_criteria_id}_{timestamp}.json"
            full_path = os.path.join(export_dir, file_name)

        if not os.path.exists(export_dir):
            os.makedirs(export_dir, exist_ok=True)

        # --- 2. จัดการ/คำนวณ Summary Field ---
        if 'summary' not in results:
            results['summary'] = {}
        
        summary = results['summary']
        summary['record_id'] = record_id  # ฝัง ID ลงในไฟล์ JSON ด้วย
        summary['enabler'] = enabler
        summary['sub_criteria_id'] = sub_criteria_id
        summary['target_level'] = target_level
        summary['tenant'] = tenant
        summary['year'] = year
        summary['export_timestamp'] = timestamp

        # ดึงข้อมูลจาก sub_criteria_results เพื่อมาทำ summary
        sub_res_list = results.get('sub_criteria_results', [])
        
        if sub_criteria_id.lower() != "all" and len(sub_res_list) > 0:
            # กรณีรัน Single Sub-Criteria
            main_res = sub_res_list[0]
            summary['highest_pass_level'] = main_res.get('highest_full_level', 0)
            summary['achieved_weight'] = main_res.get('weighted_score', 0.0)
            summary['total_weight'] = main_res.get('weight', 0.0)
            summary['is_target_achieved'] = main_res.get('target_level_achieved', False)
            summary['total_subcriteria_assessed'] = 1
        else:
            # กรณีรัน All Sub-Criteria
            all_pass_levels = [r.get('highest_full_level', 0) for r in sub_res_list]
            total_achieved = sum(r.get('weighted_score', 0.0) for r in sub_res_list)
            total_possible = sum(r.get('weight', 0.0) for r in sub_res_list)
            
            summary['highest_pass_level_overall'] = max(all_pass_levels) if all_pass_levels else 0
            summary['total_achieved_weight'] = round(total_achieved, 2)
            summary['total_possible_weight'] = round(total_possible, 2)
            summary['total_subcriteria_assessed'] = len(sub_res_list)
            
            if total_possible > 0:
                summary['overall_percentage'] = round((total_achieved / total_possible) * 100, 2)
            else:
                summary['overall_percentage'] = 0.0

        # --- 3. ตรวจสอบ Action Plan Status ---
        total_action_plans = 0
        for res in sub_res_list:
            ap = res.get('action_plan', [])
            if isinstance(ap, list):
                total_action_plans += len(ap)
        summary['total_action_plan_phases'] = total_action_plans

        # --- 4. เขียนไฟล์ JSON ---
        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            
            self.logger.info(f"💾 Exported Results to: {full_path}")
            
            # แสดง Log Summary
            final_lvl = summary.get('highest_pass_level', summary.get('highest_pass_level_overall', 0))
            final_score = summary.get('achieved_weight', summary.get('total_achieved_weight', 0.0))
            total_score = summary.get('total_weight', summary.get('total_possible_weight', 0.0))
            
            self.logger.info(
                f"📊 [SUMMARY] ID: {record_id} | Sub: {sub_criteria_id} | "
                f"Level: L{final_lvl} | Score: {final_score}/{total_score}"
            )
            return full_path
        
        except Exception as e:
            self.logger.error(f"❌ Export failed: {e}")
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

    def _save_level_evidences_and_calculate_strength(
        self, 
        level_temp_map: List[Dict[str, Any]], 
        sub_id: str, 
        level: int, 
        llm_result: Dict[str, Any],
        highest_rerank_score: float = 0.0
    ) -> float:
        """
        [FULL REFINED VERSION 25.1]
        บันทึกหลักฐานการประเมินและคำนวณ Evidence Strength 
        รองรับ Metadata แบบ Robust ที่ดึงข้อมูลได้ทั้งจาก Direct Key และ Nested Metadata
        สอดคล้องกับโครงสร้างจาก ingest.py (deterministic UUIDs)
        """
        map_key = f"{sub_id}.L{level}"
        new_evidence_list: List[Dict[str, Any]] = []
        
        # ดึงค่าเริ่มต้นเพื่อใช้ในการ Debug หากเกิดปัญหา
        self.logger.info(f"💾 [EVI SAVE] Starting save for {map_key} with {len(level_temp_map)} potential chunks.")

        for chunk in level_temp_map:
            # 🎯 1. ดึง Metadata และ Identifiers (Robust Extraction)
            # รองรับทั้ง Dictionary, Object ที่มี metadata field หรือโครงสร้างแบนราบ (Flattened)
            meta = chunk.get("metadata", {}) if isinstance(chunk.get("metadata"), dict) else {}
            
            # ค้นหา Chunk UUID (ลำดับความสำคัญ: Direct -> Metadata -> id)
            chunk_uuid_key = (
                chunk.get("chunk_uuid") or 
                meta.get("chunk_uuid") or 
                chunk.get("id")
            )
            
            # ค้นหา Stable Doc UUID (ลำดับความสำคัญ: Direct -> Metadata -> doc_id)
            stable_doc_uuid_key = (
                chunk.get("stable_doc_uuid") or 
                chunk.get("doc_id") or 
                meta.get("stable_doc_uuid") or 
                meta.get("doc_id")
            )

            # 🎯 2. Fallback Logic: ถ้ายังหา ID ไม่เจอ (เพื่อป้องกัน Data Loss)
            if not chunk_uuid_key and stable_doc_uuid_key:
                # ใช้ Stable Doc ID + fake suffix ถ้าหา chunk uuid ไม่เจอจริงๆ
                chunk_uuid_key = f"{stable_doc_uuid_key}_missing_uuid"
                self.logger.warning(f"⚠️ [EVI SAVE] Missing chunk_uuid for doc {stable_doc_uuid_key[:8]}. Using Fallback.")

            # 🎯 3. Validation: ถ้าไม่มี ID สำคัญเลย ให้ Skip เพื่อความปลอดภัยของ Database
            if not stable_doc_uuid_key or not chunk_uuid_key:
                self.logger.error(f"❌ [EVI SAVE] Critical ID Missing! Skipping chunk from source: {chunk.get('source', 'Unknown')}")
                continue

            # 🎯 4. สร้าง Evidence Entry (อิงตามมาตรฐาน ingest.py)
            # ดึงหน้ากระดาษโดยให้ความสำคัญกับ page_label ก่อน (เพื่อ UI ที่สวยงาม)
            page_val = (
                meta.get("page_label") or 
                chunk.get("page") or 
                meta.get("page") or 
                chunk.get("page_number") or 
                "N/A"
            )

            evidence_entry = {
                "sub_id": sub_id,
                "level": level,
                "relevance_score": float(chunk.get("rerank_score", chunk.get("score", 0.0))),
                "doc_id": str(stable_doc_uuid_key),
                "stable_doc_uuid": str(stable_doc_uuid_key),
                "chunk_uuid": str(chunk_uuid_key),
                "source": chunk.get("source") or meta.get("source") or "N/A",
                "source_filename": chunk.get("filename") or meta.get("source_filename") or "N/A",
                "page": str(page_val),
                "pdca_tag": chunk.get("pdca_tag") or meta.get("pdca_tag") or "Other", 
                "status": "PASS" if llm_result.get("is_passed") else "FAIL", 
                "timestamp": datetime.now().isoformat(),
            }
            new_evidence_list.append(evidence_entry)

        # 🎯 5. คำนวณ Evidence Strength (Evi Str)
        # ใช้ logic การคำนวณ Cap ที่คุณออกแบบไว้
        evi_cap_data = self._calculate_evidence_strength_cap(
            top_evidences=new_evidence_list,
            level=level,
            highest_rerank_score=highest_rerank_score
        )
        
        final_evi_str = evi_cap_data.get('max_evi_str_for_prompt', 0.0)

        # 🎯 6. บันทึกเข้า Memory Maps (ทั้งหลักและสำรองสำหรับ Worker)
        self.evidence_map.setdefault(map_key, []).extend(new_evidence_list)
        self.temp_map_for_save.setdefault(map_key, []).extend(new_evidence_list)
        
        # 🎯 7. Log สรุปผล
        self.logger.info(f"✅ [EVIDENCE SAVED] {map_key}: {len(new_evidence_list)} chunks | Strength: {final_evi_str}")
        
        return final_evi_str
        
    def _calculate_evidence_strength_cap(
        self,
        top_evidences: List[Union[Dict[str, Any], Any]],
        level: int,
        highest_rerank_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Relevant Score Gate เวอร์ชัน FINAL
        
        คำนวณ Evidence Strength โดยอิงจาก Rerank Score สูงสุดที่พบ
        - ดึงคะแนนจาก metadata, top-level key, และ regex fallback
        - ยึดตาม global_vars:
            • RERANK_THRESHOLD = 0.5
            • MAX_EVI_STR_CAP = 10.0
        
        Returns:
            dict ประกอบด้วย is_capped, max_evi_str_for_prompt, highest_rerank_score, max_score_source
        """

        score_keys = [
            "rerank_score", "score", "relevance_score",
            "_rerank_score_force", "_rerank_score",
            "Score", "RelevanceScore"
        ]

        # ─── 1. ดึงค่า config จาก class attribute ก่อน → fallback ไป global_vars ───
        threshold = getattr(self, "RERANK_THRESHOLD", 0.5)
        cap_value = getattr(self, "MAX_EVI_STR_CAP", 10.0)

        # Fallback จาก global_vars โดยตรง (หลังจาก import global_vars แล้ว)
        threshold = threshold if threshold != 0.5 else RERANK_THRESHOLD
        cap_value = cap_value if cap_value != 10.0 else MAX_EVI_STR_CAP

        # ─── 2. เริ่มต้นด้วย highest_rerank_score จาก Adaptive Loop (ค่าที่น่าเชื่อถือที่สุด) ───
        max_score_found = highest_rerank_score if highest_rerank_score is not None else 0.0
        max_score_source = "Adaptive_RAG_Loop" if highest_rerank_score is not None else "N/A"

        for doc in top_evidences:
            page_content = ""
            metadata = {}
            current_score = 0.0

            # ─── แปลง document ให้รองรับทั้ง dict และ Langchain Document ───
            if isinstance(doc, dict):
                metadata = doc.get("metadata", {})
                page_content = doc.get("page_content", "") or doc.get("text", "") or doc.get("content", "")
            else:
                metadata = getattr(doc, "metadata", {})
                page_content = getattr(doc, "page_content", "") or getattr(doc, "text", "")

            # ─── ค้นหาคะแนนจาก metadata และ top-level keys ───
            for key in score_keys:
                score_val = metadata.get(key)
                if score_val is None:
                    if isinstance(doc, dict):
                        score_val = doc.get(key)
                    else:
                        score_val = getattr(doc, key, None)

                if score_val is not None:
                    try:
                        temp_score = float(score_val)
                        if 0.0 < temp_score <= 1.0:
                            if temp_score > current_score:
                                current_score = temp_score
                                break
                    except (ValueError, TypeError):
                        continue

            # ─── Fallback: ดึงคะแนนจากท้าย content ด้วย regex (aggressive) ───
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
                                temp_score = float(m.group(1))
                                if 0.0 < temp_score <= 1.0:
                                    if temp_score > current_score:
                                        current_score = temp_score
                                        break
                            except:
                                continue
                except Exception as e:
                    self.logger.debug(f"Regex fallback failed at L{level}: {e}")

            # ─── Score Clamp: ถ้าคะแนน > 1.0 ถือว่าไม่ใช่ relevance scale 0-1 → ignore ───
            if current_score > 1.0:
                source = (
                    metadata.get("source_filename") or metadata.get("filename") or
                    doc.get("source_filename") or doc.get("filename") or
                    doc.get("source") or doc.get("doc_id") or "N/A"
                )
                self.logger.warning(
                    f"🚨 Score Clamp L{level}: Score {current_score:.4f} > 1.0 from '{source}'. Ignoring."
                )
                current_score = 0.0

            # ─── ดึง source สำหรับ log ───
            source = (
                metadata.get("source_filename") or metadata.get("filename") or
                doc.get("source_filename") or doc.get("filename") or
                doc.get("source") or doc.get("doc_id") or "N/A"
            )

            # ─── อัปเดตคะแนนสูงสุด พร้อม log override ───
            if current_score > max_score_found:
                if highest_rerank_score is not None and current_score > highest_rerank_score:
                    self.logger.critical(
                        f"⚠️ Score Override L{level}: Hidden score {current_score:.4f} > Loop score {highest_rerank_score:.4f} "
                        f"from source: {source}"
                    )
                max_score_found = current_score
                max_score_source = source

        # ─── Relevant Score Gate: ตัดสินใจ cap หรือ full ───
        if max_score_found < threshold:
            max_evi_str_for_prompt = cap_value  # ใช้ค่า cap จาก config (10.0)
            is_capped = True
            self.logger.warning(
                f"🚨 Evi Str CAPPED L{level}: Rerank {max_score_found:.4f} (from '{max_score_source}') "
                f"< {threshold} → จำกัดที่ {cap_value}"
            )
        else:
            max_evi_str_for_prompt = 10.0
            is_capped = False
            self.logger.info(
                f"✅ Evi Str FULL L{level}: Rerank {max_score_found:.4f} (from '{max_score_source}') "
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
        current_sub_id: Optional[str] = None,
        level: Optional[int] = None
    ) -> List[Dict]:

        active_sub_id = current_sub_id or getattr(self, 'sub_id', 'unknown')
        if not chunks_to_hydrate:
            return []

        TAG_ABBREV = {
            "PLAN": "P", "DO": "D", "CHECK": "C", "ACT": "A",
            "P": "P", "D": "D", "C": "C", "A": "A"
        }

        def _safe_classify(text: str) -> str:
            try:
                raw = classify_by_keyword(
                    text=text,
                    sub_id=active_sub_id,
                    level=level,
                    contextual_rules_map=self.contextual_rules_map
                )
                if not raw:
                    return "Other"
                return TAG_ABBREV.get(str(raw).upper(), "Other")
            except Exception as e:
                self.logger.warning(f"PDCA classify failed → Other | {e}")
                return "Other"

        def _standardize_chunk(chunk: Dict, score: float):
            chunk.setdefault("is_baseline", True)

            text = chunk.get("text", "").strip()
            if text:
                chunk["pdca_tag"] = _safe_classify(text)

                # ป้องกัน baseline score inflate
                chunk["rerank_score"] = max(chunk.get("rerank_score", 0.0), score)
                chunk["score"] = max(chunk.get("score", 0.0), score)

            return chunk

        stable_ids = {
            sid for c in chunks_to_hydrate
            if (sid := (c.get("stable_doc_uuid") or c.get("doc_id") or c.get("chunk_uuid")))
        }

        if not stable_ids or not vsm:
            boosted = [_standardize_chunk(c.copy(), 0.9) for c in chunks_to_hydrate]
            return self._guarantee_text_key(boosted)

        stable_id_map = defaultdict(list)

        try:
            retrieved_docs = vsm.get_documents_by_id(
                list(stable_ids),
                doc_type=self.doc_type,
                enabler=self.config.enabler
            )

            for doc in retrieved_docs:
                sid = doc.metadata.get("stable_doc_uuid") or doc.metadata.get("doc_id")
                if sid:
                    stable_id_map[sid].append({
                        "text": doc.page_content,
                        "metadata": doc.metadata
                    })

        except Exception as e:
            self.logger.error(f"VSM Hydration failed: {e}")
            fallback = [_standardize_chunk(c.copy(), 0.9) for c in chunks_to_hydrate]
            return self._guarantee_text_key(fallback)

        hydrated_priority_docs = []
        restored_count = 0
        seen_signatures = set()

        SAFE_META_KEYS = {
            "source", "file_name", "page", "page_label",
            "page_number", "enabler", "tenant", "year"
        }

        for chunk in chunks_to_hydrate:
            new_chunk = chunk.copy()
            sid = new_chunk.get("stable_doc_uuid") or new_chunk.get("doc_id")

            hydrated = False
            if sid and sid in stable_id_map:
                best_match = stable_id_map[sid][0]
                new_chunk["text"] = best_match["text"]

                meta = best_match.get("metadata", {})
                new_chunk.update({k: v for k, v in meta.items() if k in SAFE_META_KEYS})

                hydrated = True
                restored_count += 1

            new_chunk = _standardize_chunk(
                new_chunk,
                score=1.0 if hydrated else 0.85
            )

            signature = (
                sid,
                new_chunk.get("chunk_uuid"),
                new_chunk.get("text", "")[:200]
            )

            if signature in seen_signatures:
                continue

            seen_signatures.add(signature)
            hydrated_priority_docs.append(new_chunk)

        return self._guarantee_text_key(
            hydrated_priority_docs,
            total_count=len(chunks_to_hydrate),
            restored_count=restored_count
        )


    def _guarantee_text_key(
        self,
        chunks: List[Dict],
        total_count: int = 0,
        restored_count: int = 0
    ) -> List[Dict]:

        final_chunks = []

        for chunk in chunks:
            if "text" not in chunk:
                chunk["text"] = ""
                cid = str(chunk.get("chunk_uuid", "N/A"))
                self.logger.debug(f"Guaranteed 'text' key for chunk (ID: {cid[:8]})")
            final_chunks.append(chunk)

        if total_count > 0:
            baseline_count = sum(1 for c in final_chunks if c.get("is_baseline"))
            self.logger.info(
                f"HYDRATION SUMMARY: Restored {restored_count}/{total_count} "
                f"(Baseline={baseline_count})"
            )

        return final_chunks

    
    def _get_keywords_for_phase(self, sub_id: str, level: int, phase: str = "Plan") -> str:
        """
        ดึง keywords สำหรับ phase (Plan, Do, Check, Act) จาก contextual_rules.json
        
        ความสามารถ:
        1. รองรับทั้งรูปแบบ Array [..] และ String ".." จาก JSON
        2. จัดการเรื่องช่องว่าง (Whitespace) ให้อัตโนมัติ
        3. มีระบบ Fallback 4 ชั้น (Level -> Sub -> Enabler -> Global)
        4. แก้ปัญหาการส่งค่า String แล้วโดน join แยกตัวอักษร
        """
        sub_rules = self.contextual_rules_map.get(sub_id, {})
        phase_key = f"{phase.lower()}_keywords"
        
        raw_keywords = None

        # 1. Level-specific (L1, L2, ...)
        level_key = f"L{level}"
        level_rules = sub_rules.get(level_key, {})
        if phase_key in level_rules:
            raw_keywords = level_rules[phase_key]
        
        # 2. Sub-specific fallback (ถ้าไม่มีใน Level นั้นๆ)
        elif phase_key in sub_rules:
            raw_keywords = sub_rules[phase_key]
        
        # 3. Enabler default (ใช้ค่ากลางของโครงการ)
        elif "_enabler_defaults" in self.contextual_rules_map:
            raw_keywords = self.contextual_rules_map["_enabler_defaults"].get(phase_key)

        # ---------------------------------------------------------
        # 🎯 จุดประมวลผล Keywords (Data Cleaning)
        # ---------------------------------------------------------
        keywords_list = []
        if raw_keywords:
            if isinstance(raw_keywords, list):
                # กรณีเป็น Array: ล้างช่องว่างแต่ละคำ
                keywords_list = [str(k).strip() for k in raw_keywords if k]
            elif isinstance(raw_keywords, str):
                # กรณีเป็น String: แยกด้วย comma และล้างช่องว่าง
                keywords_list = [k.strip() for k in raw_keywords.split(",") if k.strip()]
        
        # 4. Global fallback (กรณีสุดท้ายจริงๆ ถ้าในไฟล์ JSON ว่างเปล่า)
        if not keywords_list:
            fallback_map = {
                "plan": ["วิสัยทัศน์", "นโยบาย", "ทิศทาง", "เป้าหมาย", "ยุทธศาสตร์", "แผน"],
                "do": ["ดำเนินการ", "ปฏิบัติ", "จัดกิจกรรม", "โครงการ", "ขับเคลื่อน"],
                "check": ["ตรวจสอบ", "ประเมิน", "ติดตาม", "วัดผล"],
                "act": ["ปรับปรุง", "พัฒนา", "แก้ไข", "นำไปใช้", "ทบทวน"]
            }
            keywords_list = fallback_map.get(phase.lower(), [])

        # ---------------------------------------------------------
        # ✅ จุดสุดท้าย: คืนค่าเป็น String ที่ปลอดภัย
        # ---------------------------------------------------------
        # ป้องกันกรณี keywords_list ไม่ใช่ list (ซึ่งไม่ควรเกิดขึ้นแต่เช็คไว้เพื่อความชัวร์)
        if isinstance(keywords_list, str):
            return keywords_list
            
        return ", ".join(keywords_list)
    
    def _get_pdca_blocks_from_evidences(
        self,
        evidences: List[Dict],
        baseline_evidences: Dict[str, List[Dict]],
        level: int,
        sub_id: str,
        contextual_rules_map: Dict[str, Any]
    ) -> Tuple[str, str, str, str, str]:
        """
        Build PDCA context blocks from evidences.
        Logic: Re-classify chunks, force L1 Plan, group, and render to text.
        """
        import copy
        from collections import defaultdict

        # ------------------------------------------------------------------
        # 1) เตรียม Chunks ทั้งหมด (รวม Baseline)
        # ------------------------------------------------------------------
        all_chunks: List[Dict] = []
        evidences = self._guarantee_text_key(evidences or [])
        for c in evidences:
            all_chunks.append(copy.deepcopy(c))

        level_baseline = baseline_evidences.get(str(level), []) or []
        for b in level_baseline:
            b_copy = copy.deepcopy(b)
            b_copy["is_baseline"] = True
            all_chunks.append(b_copy)

        all_chunks = [c for c in all_chunks if isinstance(c, dict) and c.get("text", "").strip()]
        if not all_chunks:
            return "", "", "", "", ""

        # ------------------------------------------------------------------
        # 2) Re-classify PDCA Tags ตาม Keyword ปัจจุบัน
        # ------------------------------------------------------------------
        for chunk in all_chunks:
            try:
                new_tag = classify_by_keyword(
                    text=chunk["text"],
                    sub_id=sub_id,
                    level=level,
                    contextual_rules_map=contextual_rules_map
                )
                chunk["pdca_tag"] = new_tag if new_tag in {"P", "D", "C", "A"} else "Other"
            except Exception as e:
                self.logger.warning(f"PDCA classify failed → fallback Other | {e}")
                chunk["pdca_tag"] = "Other"

        # ------------------------------------------------------------------
        # 3) 🔥 KM L1 Logic: ถ้าเป็น Level 1 ให้ถือว่าหลักฐานพื้นฐานคือ Plan (P)
        # ------------------------------------------------------------------
        if level == 1:
            forced_count = 0
            for chunk in all_chunks:
                if chunk.get("pdca_tag") == "Other":
                    chunk["pdca_tag"] = "P"
                    forced_count += 1
            if forced_count > 0:
                self.logger.info(f"💡 [L1 Domain Logic] Forced {forced_count} 'Other' chunks to 'P' (Plan) for level 1")

        # ------------------------------------------------------------------
        # 4) จัดกลุ่มตาม Label เพื่อสร้าง Text Blocks
        # ------------------------------------------------------------------
        TAG_FULL = {"P": "Plan", "D": "Do", "C": "Check", "A": "Act", "Other": "Other"}
        pdca_groups_full: Dict[str, List[Dict]] = defaultdict(list)
        for c in all_chunks:
            tag_abbr = c.get("pdca_tag", "Other")
            full_label = TAG_FULL.get(tag_abbr, "Other")
            pdca_groups_full[full_label].append(c)

        # ------------------------------------------------------------------
        # 5) Helpers สำหรับการสร้างข้อความ
        # ------------------------------------------------------------------
        def _normalize_meta(c: Dict) -> Tuple[str, str]:
            """
            ดึงชื่อไฟล์และเลขหน้าจาก Chunk โดยใช้ Fallback Logic 
            เพื่อให้ตรงกับระบบ Ingest ใหม่และเก่า (source_filename, page_label)
            รองรับกรณีค่าเป็น 0 หรือ Metadata กระจัดกระจาย
            """
            # ดึง metadata มาไว้ก่อน เผื่อต้องใช้หาหลายรอบ
            meta = c.get("metadata", {}) or {}
            
            # 1. 🔍 ลำดับการหาชื่อไฟล์ (Source)
            # พยายามหาจากชั้นนอกสุดก่อน แล้วค่อยไล่เข้าไปใน metadata
            source = (
                c.get("source_filename") or 
                c.get("filename") or 
                meta.get("source_filename") or 
                meta.get("source") or 
                meta.get("file_name") or 
                None  # ใช้ None เพื่อไปเช็คต่อที่ clean_source
            )
            
            # 2. 🔍 ลำดับการหาเลขหน้า (Page)
            # สำคัญมาก: ต้องแยกเช็คเลข 0 เพราะ Python มอง 0 เป็น False
            page = None
            page_keys = ["page_label", "page", "page_number"]
            
            # ลองหาจาก Root ของ Chunk ก่อน
            for key in page_keys:
                val = c.get(key)
                if val is not None:
                    page = val
                    break
                    
            # ถ้ายังไม่เจอ ลองหาใน metadata
            if page is None:
                for key in page_keys:
                    val = meta.get(key)
                    if val is not None:
                        page = val
                        break

            # 3. ✨ ทำความสะอาดข้อมูล (Cleaning)
            # จัดการชื่อไฟล์
            clean_source = str(source).strip() if source is not None else "Unknown"
            
            # จัดการเลขหน้า: ป้องกันเลข 0 หาย และจัดการพวก 'n/a'
            if page is not None:
                page_str = str(page).strip()
                clean_page = page_str if page_str.lower() != "n/a" else "N/A"
            else:
                clean_page = "N/A"
            
            return clean_source, clean_page

        def _create_block(tag: str, chunks: List[Dict]) -> str:
            if not chunks: return ""
            
            # 1. เรียงตามคะแนน Re-rank จากมากไปน้อย
            chunks = sorted(chunks, key=get_actual_score, reverse=True)
            
            # 2. 🛡️ ขีดเส้นใต้ 7 ชิ้นที่เทพที่สุด (ใช้ร่วมกันทั้ง Mac และ Server)
            top_chunks = chunks[:7]
            
            total = len(top_chunks)
            blocks: List[str] = []
            seen_texts = set() # สำหรับป้องกันเนื้อหาซ้ำ (Deduplication)

            for i, c in enumerate(top_chunks, start=1):
                body = c["text"].strip()
                
                # Check ซ้ำเบื้องต้น (เผื่อเจอ Chunk ที่เนื้อหาเหมือนกันเป๊ะ)
                if body[:100] in seen_texts: 
                    continue
                seen_texts.add(body[:100])

                source, page = _normalize_meta(c)
                score = get_actual_score(c)
                baseline_mark = " [📜 BASELINE/REFERENCE]" if c.get("is_baseline") else ""
                
                header = f"### [{tag} Evidence {i}/{total}]{baseline_mark}"
                footer = f"\n[อ้างอิง: {source}, หน้า: {page}, Score: {score:.4f}]"
                blocks.append(f"{header}\n{body}{footer}")
                
            return "\n---\n".join(blocks)

        # ------------------------------------------------------------------
        # 6) สร้าง Output และบันทึก Log สรุป
        # ------------------------------------------------------------------
        plan_text = _create_block("Plan", pdca_groups_full.get("Plan", []))
        do_text = _create_block("Do", pdca_groups_full.get("Do", []))
        check_text = _create_block("Check", pdca_groups_full.get("Check", []))
        act_text = _create_block("Act", pdca_groups_full.get("Act", []))
        other_text = _create_block("Other", pdca_groups_full.get("Other", []))

        # สรุปสถานะการมีอยู่ของข้อมูลแต่ละ Phase
        pdca_status = [f"{t}:{'✅' if txt else '❌'}" for t, txt in [("P", plan_text), ("D", do_text), ("C", check_text), ("A", act_text)]]
        self.logger.info(f"📊 [PDCA Block Output] {sub_id} L{level} | {' | '.join(pdca_status)} | Other:{'✅' if other_text else '❌'}")

        return (plan_text, do_text, check_text, act_text, other_text)
    
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
        [FIXED VERSION] 
        - แก้ไข AttributeError โดยการเรียกชื่อ _initialize_vsm_if_none ให้ถูกต้อง
        - คงความ Robust ของการ Merge Evidence และ Persistence ไว้ครบถ้วน
        """
        start_ts = time.time()
        self.is_sequential = sequential
        self.current_record_id = record_id 

        # ============================== 1. Filter Rubric ==============================
        all_statements = self._flatten_rubric_to_statements()
        
        if target_sub_id.lower() == "all":
            sub_criteria_list = all_statements
            self.logger.info(f"📋 Assessing ALL criteria ({len(sub_criteria_list)} items)")
        else:
            sub_criteria_list = [
                s for s in all_statements 
                if str(s.get('sub_id')).strip().lower() == str(target_sub_id).strip().lower()
            ]

        if not sub_criteria_list:
            error_msg = f"ไม่พบรหัสเกณฑ์ '{target_sub_id}' ในไฟล์ Rubric"
            self.logger.error(f"❌ {error_msg}")
            return {
                "record_id": record_id, "status": "FAILED", "error_message": error_msg,
                "summary": {"score": 0.0, "level": "L0", "total_weighted_score": 0.0, "max_weight": 0.0},
                "sub_criteria_results": [], "run_time_seconds": round(time.time() - start_ts, 2),
                "timestamp": datetime.now().isoformat()
            }

        # โหลด Evidence Map เดิม
        if os.path.exists(self.evidence_map_path):
            try:
                loaded = self._load_evidence_map()
                self.evidence_map = loaded if loaded else {}
            except Exception: self.evidence_map = {}
        else:
            self.evidence_map = {}

        self.raw_llm_results = []
        self.final_subcriteria_results = []

        max_workers = globals().get('MAX_PARALLEL_WORKERS', 4)
        run_parallel = (target_sub_id.lower() == "all") and not sequential

        # ============================== 2. Run Assessment ==============================
        if run_parallel:
            # (Parallel logic เหมือนเดิม...)
            self.logger.info(f"🚀 Starting Parallel Assessment with {max_workers} processes")
            worker_args = [(
                sub_data, self.config.enabler, self.config.target_level, self.config.mock_mode,
                self.evidence_map_path, self.config.model_name, self.config.temperature,
                getattr(self.config, 'MIN_RETRY_SCORE', 0.50), getattr(self.config, 'MAX_RETRIEVAL_ATTEMPTS', 3),
                document_map or self.document_map, self.ActionPlanActions 
            ) for sub_data in sub_criteria_list]

            try:
                with multiprocessing.get_context('spawn').Pool(processes=max_workers) as pool:
                    results_list = pool.map(_static_worker_process, worker_args)
            except Exception as e:
                self.logger.critical(f"Multiprocessing failed: {e}")
                raise

            for result_tuple in results_list:
                if not isinstance(result_tuple, tuple) or len(result_tuple) != 2: continue
                sub_result, temp_map_from_worker = result_tuple
                if isinstance(temp_map_from_worker, dict):
                    for k, v in temp_map_from_worker.items():
                        for ev in v:
                            if "page" not in ev: ev["page"] = ev.get("metadata", {}).get("page", "N/A")
                        current_list = self.evidence_map.setdefault(k, [])
                        current_list.extend(v)
                self.raw_llm_results.extend(sub_result.get("raw_results_ref", []))
                self.final_subcriteria_results.append(sub_result)

        else:
            # --------------------- SEQUENTIAL MODE ---------------------
            self.logger.info(f"Starting Sequential Assessment: {target_sub_id}")
            
            # ✅ [CRITICAL FIX] เปลี่ยนชื่อเรียกให้ตรงกับ Class Method ของคุณ
            if vectorstore_manager:
                self.vectorstore_manager = vectorstore_manager
            else:
                self._initialize_vsm_if_none() # เรียกฟังก์ชันที่คุณมีอยู่จริง
            
            for sub_criteria in sub_criteria_list:
                sub_result, final_temp_map = self._run_sub_criteria_assessment_worker(sub_criteria)
                
                if final_temp_map:
                    for k, v in final_temp_map.items():
                        for ev in v:
                            if "page" not in ev: ev["page"] = ev.get("metadata", {}).get("page", "N/A")
                        current_list = self.evidence_map.setdefault(k, [])
                        current_list.extend(v)

                self.raw_llm_results.extend(sub_result.get("raw_results_ref", []))
                self.final_subcriteria_results.append(sub_result)

        # ============================== 3. Final Persistence & Summary ==============================
        if self.evidence_map:
            try:
                self._save_evidence_map(map_to_save=self.evidence_map)
                self.logger.info(f"✅ Evidence Map Saved (Total: {len(self.evidence_map)} keys)")
            except Exception as e:
                self.logger.error(f"❌ Failed to save evidence map: {e}")

        self._calculate_overall_stats(target_sub_id)

        final_results = {
            "record_id": record_id,
            "summary": self.total_stats,
            "sub_criteria_results": self.final_subcriteria_results,
            "raw_llm_results": self.raw_llm_results,
            "run_time_seconds": round(time.time() - start_ts, 2),
            "timestamp": datetime.now().isoformat(),
        }

        if export:
            export_path = self._export_results(
                results=final_results,
                sub_criteria_id=target_sub_id if target_sub_id != "all" else "ALL",
                record_id=record_id
            )
            final_results["export_path_used"] = export_path

        return final_results
    
    def _run_sub_criteria_assessment_worker(
        self,
        sub_criteria: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
        """
        [v21.9.15 - STABLE VERSION]
        - แก้ปัญหา Source of Evidence หายในหน้า UI
        - แก้ปัญหา PDCA Opaque (สีเขียวจาง) ไม่ขึ้นในช่อง GAP_ONLY
        - แก้ Bug TypeError (dict + int) ใน Action Plan
        """
        # 📌 โหลด Global Constants
        REQUIRED_PDCA: Final[Dict[int, Set[str]]] = globals().get('REQUIRED_PDCA', {1: {"P"}, 2: {"P", "D"}, 3: {"P", "D", "C"}, 4: {"P", "D", "C", "A"}, 5: {"P", "D", "C", "A"}})
        MAX_L1_ATTEMPTS = globals().get('MAX_L1_ATTEMPTS', 2)
        MIN_KEEP_SC = globals().get('MIN_RERANK_SCORE_TO_KEEP', 0.15)

        sub_id = sub_criteria['sub_id']
        sub_criteria_name = sub_criteria['sub_criteria_name']
        sub_weight = sub_criteria.get('weight', 0)
        
        current_sequential_pass_level = 0 
        first_failed_level_local = None 
        raw_results_for_sub_seq: List[Dict[str, Any]] = []
        start_ts = time.time() 

        self.logger.info(f"[WORKER START] Assessing: {sub_id}")
        self.temp_map_for_save = {}

        # -----------------------------------------------------------
        # 1. LOOP THROUGH LEVELS (L1 → L5)
        # -----------------------------------------------------------
        for statement_data in sub_criteria.get('levels', []):
            level = statement_data.get('level')
            if level is None or level > self.config.target_level:
                continue
            
            sequential_chunk_uuids = [] 
            level_result = {}

            # --- ส่วนการประเมิน (คงเดิมจาก Original) ---
            if level >= 3:
                wrapper = self.retry_policy.run(
                    fn=lambda attempt: self._run_single_assessment(
                        sub_criteria=sub_criteria,
                        statement_data=statement_data,
                        vectorstore_manager=self.vectorstore_manager,
                        sequential_chunk_uuids=sequential_chunk_uuids,
                        attempt=attempt
                    ),
                    level=level,
                    statement=statement_data.get('statement', ''),
                    context_blocks={"sequential_chunk_uuids": sequential_chunk_uuids},
                    logger=self.logger
                )
                level_result = wrapper.result if isinstance(wrapper, RetryResult) and wrapper.result is not None else {}
            else:
                for attempt_num in range(1, MAX_L1_ATTEMPTS + 1):
                    level_result = self._run_single_assessment(
                        sub_criteria=sub_criteria,
                        statement_data=statement_data,
                        vectorstore_manager=self.vectorstore_manager,
                        sequential_chunk_uuids=sequential_chunk_uuids,
                        attempt=attempt_num
                    )
                    if level_result.get('is_passed', False): break

            # --- 1.2 PROCESS RESULT AND HANDLE EVIDENCE ---
            result_to_process = level_result or {"level": level, "is_passed": False}
            is_passed_llm = result_to_process.get('is_passed', False)
            
            # 🟢 [REPAIR] PDCA Repair Logic (ฉีดเพิ่มเพื่อแก้ปัญหาสีขาว)
            pdca_val = result_to_process.get('pdca_breakdown', {})
            if is_passed_llm and (not pdca_val or all(v == 0 for v in pdca_val.values())):
                repaired_pdca = {"P": 1, "D": (1 if level >= 2 else 0), "C": (1 if level >= 3 else 0), "A": (1 if level >= 4 else 0)}
                result_to_process['pdca_breakdown'] = repaired_pdca

            result_to_process.setdefault("is_counted", True)
            result_to_process.setdefault("is_capped", False)

            # 🟢 [SOURCE] บันทึก Evidence ลงระบบ (ฉีดเพิ่มเพื่อให้ Source ไม่หาย)
            level_temp_map = result_to_process.get("temp_map_for_level", [])
            if level_temp_map:
                highest_rerank = result_to_process.get('max_relevant_score', 0.0)
                # บันทึกไฟล์เข้าสู่ระบบ Mapping
                max_evi_str = self._save_level_evidences_and_calculate_strength(
                    level_temp_map=level_temp_map,
                    sub_id=sub_id,
                    level=level,
                    llm_result=result_to_process, 
                    highest_rerank_score=highest_rerank 
                )
                # เก็บ Strength เฉพาะข้อที่ผ่าน Sequential จริง
                if is_passed_llm and first_failed_level_local is None:
                    result_to_process['evidence_strength'] = round(min(max_evi_str, 10.0), 1)
                else:
                    result_to_process['evidence_strength'] = 0.0
                
            # --- 🟡 [SAFE UPDATE] Sequential State (ปรับปรุงจาก Original เพื่อรักษา Metadata) ---
            if first_failed_level_local is not None:
                # บังคับจำค่า Metadata ที่ AI เจอจริงก่อนเปลี่ยนสถานะเป็น GAP_ONLY
                act_pdca = result_to_process.get('pdca_breakdown', {"P": 0, "D": 0, "C": 0, "A": 0})
                act_src = result_to_process.get('temp_map_for_level', [])
                
                result_to_process.update({
                    "evaluation_mode": "GAP_ONLY",
                    "is_counted": False,
                    "is_passed": False,
                    "pdca_breakdown": act_pdca,      # <--- ช่วยให้สีเขียวจางขึ้น
                    "temp_map_for_level": act_src,   # <--- ช่วยให้ปุ่ม Source ขึ้น
                    "cap_reason": f"Gap analysis after sequential fail at L{first_failed_level_local}"
                })
            elif not is_passed_llm:
                first_failed_level_local = level
            else:
                if level == current_sequential_pass_level + 1:
                    current_sequential_pass_level = level
                else:
                    if self.is_sequential:
                        first_failed_level_local = current_sequential_pass_level + 1
                        result_to_process["is_counted"] = False
                        result_to_process["is_passed"] = False

            result_to_process["execution_index"] = len(raw_results_for_sub_seq)
            raw_results_for_sub_seq.append(result_to_process)
        
        # -----------------------------------------------------------
        # 2. CALCULATE SUMMARY (คงเดิม)
        # -----------------------------------------------------------
        highest_full_level = current_sequential_pass_level
        weighted_score = round(self._calculate_weighted_score(highest_full_level, sub_weight), 2)
        num_passed = sum(1 for r in raw_results_for_sub_seq if r.get("is_passed", False) and r.get("is_counted", True))

        sub_summary = {
            "num_statements": len(raw_results_for_sub_seq),
            "num_passed": num_passed,
            "num_failed": len(raw_results_for_sub_seq) - num_passed,
            "pass_rate": round(num_passed / len(raw_results_for_sub_seq), 4) if raw_results_for_sub_seq else 0.0
        }

        # -----------------------------------------------------------
        # 3. GENERATE ACTION PLAN (คงเดิม แต่แก้ Bug Parameter)
        # -----------------------------------------------------------
        roadmap_target_level = self.config.target_level if hasattr(self.config, 'target_level') else 5
        statements_for_ap = []
        level_statements_map = {l.get('level'): l.get('statement', '') for l in sub_criteria.get('levels', [])}
        
        for r in raw_results_for_sub_seq:
            res_item = r.copy()
            res_item['statement_text'] = level_statements_map.get(res_item.get('level'), "")
            if not res_item.get('is_passed', False):
                res_item['recommendation_type'] = 'FAILED' if res_item.get('evaluation_mode') != "GAP_ONLY" else 'GAP_ANALYSIS'
                statements_for_ap.append(res_item)
            else:
                pdca = res_item.get('pdca_breakdown', {})
                if any(v == 0 for v in pdca.values()):
                    res_item['recommendation_type'] = 'PDCA_INCOMPLETE'
                    statements_for_ap.append(res_item)

        # ✅ FIX: ระบุชื่อ parameter เพื่อป้องกัน TypeError (dict + int)
        action_plan_result = create_structured_action_plan(
            recommendation_statements=statements_for_ap,
            sub_id=sub_id,
            sub_criteria_name=sub_criteria_name,
            target_level=roadmap_target_level, 
            llm_executor=self.llm,
            logger=self.logger,
            enabler_rules=self.contextual_rules_map 
        )

        # -----------------------------------------------------------
        # 4. FINAL RETURN
        # -----------------------------------------------------------
        final_temp_map = {}
        # รวบรวม Source ของทุุก Level ที่รันผ่านมา
        for res in raw_results_for_sub_seq:
            lvl = res.get('level')
            for evi in res.get("temp_map_for_level", []):
                f_id = evi.get("file_id") or evi.get("uuid")
                if f_id:
                    final_temp_map[f"{sub_id}.{lvl}.{f_id}"] = evi

        final_sub_result = {
            "sub_criteria_id": sub_id,
            "sub_criteria_name": sub_criteria_name,
            "highest_full_level": highest_full_level,
            "weight": sub_weight,
            "target_level_achieved": highest_full_level >= roadmap_target_level,
            "weighted_score": weighted_score,
            "action_plan": action_plan_result, 
            "raw_results_ref": raw_results_for_sub_seq,
            "sub_summary": sub_summary,
            "worker_duration_s": round(time.time() - start_ts, 2)
        }

        return final_sub_result, final_temp_map
    
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
        [EXPERT LOOP v21.9.3] Fixed Context Delivery & Expert Hint
        """
        # 1. ป้องกัน Infinite Loop (Max 2 times)
        if not hasattr(self, "_expert_re_eval_count"):
            self._expert_re_eval_count = 0
        self._expert_re_eval_count += 1
        
        if self._expert_re_eval_count > 2:
            self.logger.warning(f"🛑 Expert Re-evaluation limit reached for {sub_id} L{level}")
            return {
                "score": 0.0, "is_passed": False, 
                "reason": f"Limit exceeded. Original error: {first_attempt_reason}",
                "P_Plan_Score": 0.0, "D_Do_Score": 0.0, "C_Check_Score": 0.0, "A_Act_Score": 0.0
            }

        self.logger.info(f"🔍 [EXPERT RE-EVAL #{self._expert_re_eval_count}] Starting second pass for {sub_id} L{level}...")

        # 2. เตรียม Hint Message ให้ AI
        hint_msg = f"""
        --- ⚠️ EXPERT RE-EVALUATION MODE (Attempt #{self._expert_re_eval_count}) ---
        ท่านคือผู้เชี่ยวชาญการตรวจประเมิน SE-AM ระดับสูง
        
        ข้อมูลสำคัญ:
        - พบหลักฐานคุณภาพสูงมาก (Rerank Score: {highest_rerank_score:.4f}) แต่รอบแรก LLM ให้ตก
        - เหตุผลที่ตกในรอบแรก: {first_attempt_reason}
        - Tag ที่ตรวจไม่พบ: {', '.join(missing_tags) if missing_tags else 'N/A'}

        คำสั่งพิเศษ:
        โปรดอ่าน Context อย่างละเอียดอีกครั้ง หากพบร่องรอยการทำงานหรือหลักฐานที่ "สะท้อนถึง" วงจร PDCA 
        แม้จะไม่มี Keyword ตรงตัว (เช่น พบตารางเวลาทำงานที่อนุมานได้ว่าเป็นแผนงาน) 
        จงใช้ Expert Judgment ให้คะแนนตามความเหมาะสม อย่าติดอยู่กับรูปแบบเดิม
        """

        # 3. เตรียม Context (ป้องกันค่าว่าง)
        final_context = f"{context}\n\n{hint_msg}" if context else hint_msg
        
        # 4. เตรียม Kwargs และลบค่าที่อาจซ้ำซ้อนออก
        expert_kwargs = base_kwargs.copy()
        expert_kwargs.pop("context", None)
        expert_kwargs.pop("context_str", None)
        expert_kwargs.pop("final_llm_context", None)

        # 5. เรียก Evaluator แบบ Explicit (ระบุชื่อตัวแปรชัดเจน)
        result = llm_evaluator_to_use(
            context=final_context, # ส่งเข้า positional 'context' ของ evaluator
            sub_id=sub_id,
            level=level,
            statement_text=statement_text,
            sub_criteria_name=f"{sub_criteria_name} (Expert Mode)",
            max_rerank_score=highest_rerank_score,
            **expert_kwargs
        )
        
        self.logger.info(f"✅ [EXPERT RE-EVAL #{self._expert_re_eval_count}] Completed for {sub_id} L{level}")
        return result

    def _run_single_assessment(
        self,
        sub_criteria: Dict[str, Any],
        statement_data: Dict[str, Any],
        vectorstore_manager: Optional['VectorStoreManager'],
        sequential_chunk_uuids: Optional[List[str]] = None,
        attempt: int = 1
    ) -> Dict[str, Any]:
        """
        [REVISED v21.9.11 - PRODUCTION READY]
        - การันตีการส่ง temp_map_for_level กลับไปเสมอแม้ประเมินไม่ผ่าน
        - ปรับปรุงโครงสร้าง Metadata เพื่อให้ UI แสดงผล Source of Evidence ได้แม่นยำ
        - บูรณาการ Focus Points และ Evidence Guidelines เข้ากับระบบ RAG
        """
        start_time = time.time()
        sub_id = sub_criteria['sub_id']
        level = statement_data['level']
        statement_text = statement_data['statement']
        sub_criteria_name = sub_criteria['sub_criteria_name']
        statement_id = statement_data.get('statement_id', sub_id)

        # ⚙️ Configuration Setup
        MAX_RETRI_ATTEMPTS = globals().get('MAX_RETRIEVAL_ATTEMPTS', 3)
        MIN_RETRY_SC = globals().get('MIN_RETRY_SCORE', 0.7)
        MIN_KEEP_SC = globals().get('MIN_RERANK_SCORE_TO_KEEP', 0.15)
        TARGET_SCORE_THRESHOLD_MAP = globals().get('TARGET_SCORE_THRESHOLD_MAP', {1:2, 2:2, 3:2, 4:2, 5:2})

        self.logger.info(f"  > Starting assessment for {sub_id} L{level} (Attempt: {attempt})...")

        # ==================== 1. PDCA & Keywords Setup ====================
        pdca_phase = self._get_pdca_phase(level)
        level_constraint = self._get_level_constraint_prompt(level)
        
        must_list = self.get_rule_content(sub_id, level, "must_include_keywords")
        must_include_keywords = ", ".join(must_list) if isinstance(must_list, list) else (must_list or "")
        
        avoid_list = self.get_rule_content(sub_id, level, "avoid_keywords")
        avoid_keywords = ", ".join(avoid_list) if isinstance(avoid_list, list) else (avoid_list or "")
        
        plan_keywords = self.get_rule_content(sub_id, level, "plan_keywords")

        # ==================== 2. Hybrid Retrieval Setup ====================
        mapped_ids, priority_unhydrated = self._get_mapped_uuids_and_priority_chunks(
            sub_id=sub_id, level=level, statement_text=statement_text,
            level_constraint=level_constraint, vectorstore_manager=vectorstore_manager
        )
        priority_docs = self._robust_hydrate_documents_for_priority_chunks(
            chunks_to_hydrate=priority_unhydrated, vsm=vectorstore_manager, current_sub_id=sub_id
        )

        # ==================== 3. Enhanced Query with Focus & Guidelines ====================
        focus_points = sub_criteria.get('focus_points', [])
        guideline = sub_criteria.get('evidence_guidelines', {}).get(f'level_{level}', "")

        rag_query_list = self.enhance_query_for_statement(
            statement_text=statement_text, 
            sub_id=sub_id, 
            statement_id=statement_id,
            level=level, 
            focus_hint=level_constraint,
            additional_context={
                "focus_points": focus_points,
                "guideline": guideline
            }
        )

        # ==================== 4. LLM Evaluator Selection ====================
        llm_evaluator_to_use = evaluate_with_llm_low_level if level <= 2 else self.llm_evaluator

        # ==================== 5. ADAPTIVE RAG LOOP ====================
        highest_rerank_score = -1.0
        final_top_evidences = []

        for loop_attempt in range(1, MAX_RETRI_ATTEMPTS + 1):
            query_input = rag_query_list if loop_attempt == 1 and rag_query_list else [statement_text]
            try:
                retrieval_result = self.rag_retriever(
                    query=query_input, doc_type=self.doc_type, enabler=self.config.enabler,
                    sub_id=sub_id, level=level, vectorstore_manager=vectorstore_manager,
                    mapped_uuids=mapped_ids, priority_docs_input=priority_docs,
                    sequential_chunk_uuids=sequential_chunk_uuids,
                )
                top_evidences_current = retrieval_result.get("top_evidences", [])
                
                # หาคะแนนสูงสุด
                current_max = max((ev.get('metadata', {}).get('rerank_score', 0.0) for ev in top_evidences_current), default=0.0)
                priority_max = max((doc.get('metadata', {}).get('rerank_score', 0.0) for doc in priority_docs), default=0.0)
                overall_max = max(current_max, priority_max)

                if overall_max >= highest_rerank_score:
                    highest_rerank_score = overall_max
                    final_top_evidences = top_evidences_current + priority_docs

                if highest_rerank_score >= MIN_RETRY_SC:
                    break 
            except Exception as e:
                self.logger.error(f"RAG retrieval failed at loop {loop_attempt}: {e}")
                break

        # Filter & Expand Context
        top_evidences = [doc for doc in final_top_evidences if doc.get('metadata', {}).get('rerank_score', 0) >= MIN_KEEP_SC or doc.get('is_baseline', False)]
        
        if not top_evidences and level <= 2:
            top_evidences = sorted(final_top_evidences, key=lambda x: x.get('metadata', {}).get('rerank_score', 0), reverse=True)[:10]

        if top_evidences and vectorstore_manager:
            top_evidences = self._robust_hydrate_documents_for_priority_chunks(top_evidences, vectorstore_manager)
            top_evidences = self._expand_context_with_neighbor_pages(top_evidences, f"evidence_{self.config.enabler.lower()}")

        # ==================== 6. Context Building ====================
        previous_evidence = self._collect_previous_level_evidences(sub_id, level) if level > 1 else {}
        flat_previous = [item for sublist in previous_evidence.values() for item in sublist]

        plan_blocks, do_blocks, check_blocks, act_blocks, other_blocks = self._get_pdca_blocks_from_evidences(
            top_evidences + flat_previous,
            baseline_evidences=previous_evidence,
            level=level,
            sub_id=sub_id,
            contextual_rules_map=self.contextual_rules_map
        )

        channels = build_multichannel_context_for_level(level, top_evidences, flat_previous)
        final_llm_context = "\n\n".join(filter(None, [
            f"--- DIRECT EVIDENCE (L{level} | PDCA Structured)---\n{plan_blocks}\n{do_blocks}\n{check_blocks}\n{act_blocks}\n{other_blocks}",
            f"--- AUXILIARY SUMMARY ---\n{channels.get('aux_summary')}",
            f"--- BASELINE SUMMARY ---\n{channels.get('baseline_summary')}"
        ]))

        # ==================== 7. Evidence Strength & Tags ====================
        available_tags = {tag for tag, block in zip(['P', 'D', 'C', 'A'], [plan_blocks, do_blocks, check_blocks, act_blocks]) if block.strip()}
        evi_cap_data = self._calculate_evidence_strength_cap(top_evidences, level, highest_rerank_score)
        max_evi_str_for_prompt = evi_cap_data['max_evi_str_for_prompt']

        # ==================== 8. Contextual Rule Logic ====================
        rule_instruction = self.get_rule_content(sub_id, level, "specific_contextual_rule")
        if not rule_instruction:
            rule_instruction = f"เน้นพิจารณาหลักฐานตามหัวข้อ: {', '.join(focus_points)}" if focus_points else "พิจารณาตามเกณฑ์ SE-AM มาตรฐาน"

        # ==================== 9. EVALUATION EXECUTION ====================
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
            "specific_contextual_rule": rule_instruction,
            "max_rerank_score": highest_rerank_score,
            "max_evidence_strength": max_evi_str_for_prompt,
            "llm_executor": self.llm,
            "ai_confidence": "HIGH" if len(available_tags) >= 2 else "MEDIUM",
            "target_score_threshold": TARGET_SCORE_THRESHOLD_MAP.get(level, 2),
            "planning_keywords": plan_keywords if level <= 2 else "N/A"
        }

        llm_result = llm_evaluator_to_use(**llm_kwargs)

        # Expert Re-evaluation Fallback
        if not llm_result.get('is_passed', False) and highest_rerank_score >= 0.6:
            try:
                llm_result = self._run_expert_re_evaluation(
                    sub_id=sub_id, level=level, statement_text=statement_text,
                    context=final_llm_context, first_attempt_reason=llm_result.get('reason', 'N/A'),
                    missing_tags=set(), highest_rerank_score=highest_rerank_score,
                    sub_criteria_name=sub_criteria_name, llm_evaluator_to_use=llm_evaluator_to_use,
                    base_kwargs=llm_kwargs
                )
            except Exception: pass

        # ==================== 10. Metadata Mapping for Save (THE FIX) ====================
        # สกัดหลักฐานทั้งหมดที่ใช้ในระดับนี้เพื่อส่งกลับ แม้ประเมินจะไม่ผ่าน
        temp_map_for_level = []
        for doc in top_evidences:
            meta = doc.get('metadata', {})
            chunk_id = meta.get('id') or meta.get('uuid') or meta.get('chunk_id')
            if chunk_id:
                temp_map_for_level.append({
                    "id": chunk_id,
                    "file_id": meta.get('file_id') or meta.get('uuid'),
                    "file_name": meta.get('file_name', 'Unknown'),
                    "page": meta.get('page', 'N/A'),
                    "rerank_score": meta.get('rerank_score', 0.0),
                    "content": doc.page_content if hasattr(doc, 'page_content') else ""
                })

        # ==================== 11. Final Output Mapping ====================
        llm_result = post_process_llm_result(llm_result, level)
        thai_summary_data = create_context_summary_llm(final_llm_context, sub_criteria_name, level, sub_id, self.llm)

        return {
            "sub_criteria_id": sub_id,
            "level": level,
            "is_passed": llm_result.get('is_passed', False),
            "score": llm_result.get('score', 0.0),
            "pdca_breakdown": llm_result.get('pdca_breakdown', {'P': 0, 'D': 0, 'C': 0, 'A': 0}),
            "reason": llm_result.get('reason', "No reason provided"),
            "evidence_strength": max_evi_str_for_prompt if llm_result.get('is_passed', False) else 0.0,
            "max_relevant_score": highest_rerank_score,
            "summary_thai": thai_summary_data.get("summary"),
            "temp_map_for_level": temp_map_for_level, # ส่งกลับไปให้ worker ตลอดเวลา
            "duration": time.time() - start_time
        }