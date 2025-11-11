# enabler_assessment.py (Full Script)
import os
import sys
import json
import logging
import re
import time
from typing import List, Dict, Any, Optional, Union, Tuple
from core.action_plan_schema import ActionPlanActions, ActionItem, StepDetail
from pydantic import ValidationError


# -------------------- PATH SETUP --------------------
try:
    # Ensure project root is in sys.path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.append(project_root)

    # -------------------- IMPORT CONFIG --------------------
    from config.global_vars import (
        FINAL_K_RERANKED,
        FINAL_K_NON_RERANKED,
        EVIDENCE_DOC_TYPES
    )

    # -------------------- IMPORT CORE LOGIC --------------------
    from core.retrieval_utils import (
        evaluate_with_llm, 
        retrieve_context_with_filter, 
        set_mock_control_mode, 
        summarize_context_with_llm,
        retrieve_context_by_doc_ids,
        extract_uuids_from_llm_response
    )
    import core.retrieval_utils

except ImportError as e:
    print(f"FATAL ERROR: Failed to import required modules. Check sys.path and file structure. Error: {e}", file=sys.stderr)
    sys.exit(1)

# -------------------- LOGGING --------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Standardize logging configuration
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# -------------------- CONFIG & FALLBACKS --------------------

# Level Fractions สำหรับ Linear Interpolation
DEFAULT_LEVEL_FRACTIONS = {"0": 0.0, "1": 0.1, "2": 0.3, "3": 0.6, "4": 0.85, "5": 1.0, "MAX_LEVEL_FRACTION": 1.0}


# -------------------- EnablerAssessment Class --------------------

class EnablerAssessment:

    def __init__(self,
                 enabler_abbr: str, 
                 evidence_data: Optional[List] = None,
                 rubric_data: Optional[Dict] = None,
                 level_fractions: Optional[Dict] = None,
                 evidence_mapping_data: Optional[Dict] = None, 
                 vectorstore_retriever=None,
                 use_mapping_filter: bool = True,
                 target_sub_id: Optional[str] = None,
                 mock_llm_eval_func=None,
                 mock_llm_summarize_func=None,
                 mock_llm_action_plan_func=None,
                 disable_semantic_filter: bool = False,
                 allow_fallback: bool = False):
        
        # --- กำหนดค่า K-Values และ Context Length สำหรับใช้ภายในคลาส ---
        # 🟢 FIX: เพิ่ม Logger Instance
        self.logger = logging.getLogger(__name__) # 🎯 ต้องเพิ่มบรรทัดนี้
        self.MAX_CONTEXT_LENGTH = 35000 
        self.MAX_SNIPPET_LENGTH = 300
        self.FINAL_K_RERANKED = FINAL_K_RERANKED 
        self.FINAL_K_NON_RERANKED = FINAL_K_NON_RERANKED 
        self.enabler_rubric_key = f"{enabler_abbr.upper()}_Maturity_Rubric" 

        # --- การกำหนดค่า Attributes (สำคัญมาก) ---
        
        # 1. Attributes พื้นฐาน
        self.enabler_abbr = enabler_abbr.upper()
        
        self.BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "evidence_checklist"))
        self.EVIDENCE_FILE = os.path.join(self.BASE_DIR, f"{enabler_abbr.lower()}_evidence_statements_checklist.json")
        self.RUBRIC_FILE = os.path.join(self.BASE_DIR, f"{enabler_abbr.lower()}_rating_criteria_rubric.json")
        self.LEVEL_FRACTIONS_FILE = os.path.join(self.BASE_DIR, f"{enabler_abbr.lower()}_scoring_level_fractions.json")
        # self.MAPPING_FILE = os.path.join(self.BASE_DIR, f"{enabler_abbr.lower()}_evidence_mapping_new.json")
        self.MAPPING_FILE = os.path.join(self.BASE_DIR, f"{enabler_abbr.lower()}_evidence_mapping_new.json")

        self.evidence_data = evidence_data or self._load_json_fallback(self.EVIDENCE_FILE, default=[])
        default_rubric = {self.enabler_rubric_key: {"levels": []}}
        self.rubric_data = rubric_data or self._load_json_fallback(self.RUBRIC_FILE, default=default_rubric)
        self.level_fractions = level_fractions or self._load_json_fallback(self.LEVEL_FRACTIONS_FILE, default=DEFAULT_LEVEL_FRACTIONS)
        self.evidence_mapping_data = evidence_mapping_data or self._load_json_fallback(self.MAPPING_FILE, default={}) 
        
        self.vectorstore_retriever = vectorstore_retriever
        
        # 2. Attributes สำหรับ Filter และ Control
        self.use_mapping_filter = use_mapping_filter
        self.target_sub_id = target_sub_id
        
        self.disable_semantic_filter = disable_semantic_filter 
        self.allow_fallback = allow_fallback 
        
        # 3. Attributes สำหรับ Mocking
        self.mock_llm_eval_func = mock_llm_eval_func
        self.mock_llm_summarize_func = mock_llm_summarize_func
        self.mock_llm_action_plan_func = mock_llm_action_plan_func

        self.raw_llm_results: List[Dict] = []
        self.final_subcriteria_results: List[Dict] = []
        
        self.global_rubric_map: Dict[int, Dict[str, str]] = self._prepare_rubric_map()

    def _get_all_failed_statements(self) -> List[Dict]:
        """
        ดึง Statement ทั้งหมดที่ไม่ผ่าน (pass_status = False) จาก self.raw_llm_results
        โดยใช้คำจำกัดความของ Evidence Statement และ Rubric Standard
        """
        failed_statements = []
        for r in self.raw_llm_results:
            # ตรวจสอบว่ามี pass_status และเป็น False
            if not r.get('pass_status', False) and r.get('level', 0) >= 1:
                
                # 🟢 Mapping Key ตามคำจำกัดความใหม่ของคุณ
                failed_statements.append({
                    'sub_id': r.get('sub_criteria_id'),
                    'statement_id': r.get('statement_id'),
                    'level': r.get('level'),
                    'statement_number': r.get('statement_number'),
                    
                    # 1. EVIDENCE STATEMENT (ES) - ข้อความ Rubric เฉพาะ (Statement ใน raw_llm_results)
                    'evidence_statement_text': r.get('statement', 'N/A'), 
                    
                    # 2. RUBRIC STANDARD (RS) - ข้อความ Rubric ทั่วไป (Standard ใน raw_llm_results)
                    'rubric_standard_text': r.get('standard', 'N/A'), 
                    
                    # 3. ข้อมูลสนับสนุนอื่น ๆ
                    'llm_reasoning': r.get('reason', 'N/A'), 
                    'retrieved_context': r.get('context_retrieved_snippet', 'N/A'),
                    'context_uuids': r.get('context_uuids', []), 
                })
        return failed_statements

    def _load_json_fallback(self, path: str, default: Any):
        """Loads JSON หากไฟล์ไม่มี ให้ใช้ default"""
        if not os.path.isfile(path):
            return default
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            return default


    def _prepare_rubric_map(self) -> Dict[int, Dict[str, str]]:
        """
        แปลง Global Rubric เป็น Map ง่ายต่อการเรียกใช้ โดยใช้คีย์ Enabler ปัจจุบัน
        """
        rubric_map = {}
        
        if self.rubric_data is None:
            logger.warning("⚠️ Rubric data is None. Cannot prepare rubric map. Using default/fallback structure.")
            
        rubric_data_entry = self.rubric_data.get(self.enabler_rubric_key)
        
        if isinstance(rubric_data_entry, dict) and 'levels' in rubric_data_entry:
            for level_entry in rubric_data_entry.get("levels", []):
                level_num = level_entry.get("level")
                if level_num:
                    rubric_map[level_num] = level_entry.get("criteria", {})
        else:
            logger.warning(f"⚠️ Rubric key '{self.enabler_rubric_key}' not found or is invalid in loaded rubric data. Falling back to internal defaults.")
            
            default_levels = [int(lvl) for lvl in self.level_fractions.keys() if lvl.isdigit()]
            for level in default_levels:
                rubric_map[level] = {f"subtopic_{i+1}": f"Default standard L{level} S{i+1}" for i in range(5)} 

        return rubric_map

    def _compute_subcriteria_score(self, level_pass_ratios: Dict[str, float], sub_criteria_weight: float) -> Dict[str, Any]:
        """
        คำนวณคะแนนละเอียดตาม Linear Interpolation Logic
        """
        highest_full_level = 0
        progress_score = 0.0
        
        # 1. หา Highest Fully Passed Level (1.0 ratio)
        for level in range(1, 6):
            level_str = str(level)
            if level_pass_ratios.get(level_str, 0.0) < 1.0: 
                highest_full_level = level - 1 
                if highest_full_level < 0:
                    highest_full_level = 0
                break 
            else:
                highest_full_level = level 
        
        # 2. คำนวณ Progress Score (Linear Interpolation)
        max_fraction = self.level_fractions.get("MAX_LEVEL_FRACTION", 1.0)
        
        if highest_full_level == 5:
            progress_score = max_fraction * sub_criteria_weight
        else:
            base_fraction = self.level_fractions.get(str(highest_full_level) if highest_full_level > 0 else "0", 0.0)
            gap_level = highest_full_level + 1 
            gap_fraction = self.level_fractions.get(str(gap_level), 0.0)
            progress_ratio = level_pass_ratios.get(str(gap_level), 0.0)
            fraction_increase = (gap_fraction - base_fraction) * progress_ratio
            total_fraction = base_fraction + fraction_increase
            progress_score = total_fraction * sub_criteria_weight
        
        # 3. จัด Gap Analysis และ Action Item
        action_item = ""
        development_gap = False
        target_gap_level = highest_full_level + 1
        
        if target_gap_level <= 5: 
             development_gap = True
             ratio = level_pass_ratios.get(str(target_gap_level), 0.0)
             action_item = f"ดำเนินการเพื่อบรรลุหลักฐานทั้งหมดใน Level {target_gap_level} โดยเฉพาะรายการที่ยังไม่สอดคล้อง (Pass Ratio: {ratio})"
        
        # 4. Return Results
        return {
            "highest_full_level": highest_full_level,
            "progress_score": round(progress_score, 2),
            "development_gap": development_gap,
            "action_item": action_item,
            "weight": sub_criteria_weight
        }

    def _get_collection_name(self) -> str:
        """
        Constructs the specific collection name based on the enabler abbreviation.
        """
        return f"{EVIDENCE_DOC_TYPES}_{self.enabler_abbr.lower()}" 

    def _get_doc_uuid_filter(self, sub_id: str, level: int) -> Optional[List[str]]:
        """
        Generates a list of document UUIDs to filter the RAG retrieval.
        NOTE: Logic is mainly executed in _retrieve_context for convenience/debugging
        """
        if not self.evidence_mapping_data: 
            return None

        key = f"{sub_id}_L{level}"
        if key not in self.evidence_mapping_data: 
            return None

        try:
            key_data = self.evidence_mapping_data[key]
            
            if 'evidences' in key_data and isinstance(key_data['evidences'], list):
                # 🟢 FIX: ใช้ stable_doc_uuid เป็นหลัก (ตามการแก้ไขหลัก)
                doc_uuids = [
                    d.get('stable_doc_uuid') or d.get('doc_id')
                    for d in key_data['evidences'] 
                    if isinstance(d, dict) and (d.get('stable_doc_uuid') or d.get('doc_id'))
                ]
                return doc_uuids
            return None

        except Exception as e:
            logger.error(f"Error parsing evidence mapping for key {key}: {e}", exc_info=True)
            return None


    def _get_level_constraint_prompt(self, level: int) -> str:
        """
        สร้าง Prompt Constraint เพื่อจำกัดขอบเขตของหลักฐานให้เหมาะสมกับระดับวุฒิภาวะที่กำลังประเมิน
        """
        if level == 1:
            return "ข้อจำกัด: หลักฐานต้องเกี่ยวกับ 'การเริ่มต้น', 'การมีอยู่', หรือ 'การวางแผน' เท่านั้น ห้ามเกี่ยวข้องกับ 'การปรับปรุงอย่างต่อเนื่อง', 'การบูรณาการ', 'นวัตกรรม', หรือ 'การวัดผลลัพธ์ระยะยาว' (L1-Filter)"
        elif level == 2:
            return "ข้อจำกัด: หลักฐานต้องเกี่ยวกับ 'การปฏิบัติ', 'การทำให้เป็นมาตรฐาน', หรือ 'การประเมินเบื้องต้น' ห้ามเกี่ยวข้องกับ 'การบูรณาการ', 'นวัตกรรม', หรือ 'การวัดผลลัพธ์ระยะยาว' (L2-Filter)"
        elif level == 3:
            return "ข้อจำกัด: หลักฐานควรเกี่ยวกับ 'การควบคุม', 'การกำกับดูแล', หรือ 'การวัดผลเบื้องต้น' ห้ามเกี่ยวข้องกับ 'นวัตกรรม', หรือ 'การสร้างคุณค่าทางธุรกิจระยะยาว' (L3-Filter)"
        elif level == 4:
            return "ข้อจำกัด: หลักฐานควรแสดงถึง 'การบูรณาการ' หรือ 'การปรับปรุงอย่างต่อเนื่อง' ห้ามเกี่ยวข้องกับการพิสูจน์ 'คุณค่าทางธุรกิจระยะยาว' (L4-Filter)"
        elif level == 5:
            return "ข้อจำกัด: หลักฐานควรแสดงถึง 'นวัตกรรม', 'การสร้างคุณค่าทางธุรกิจ', หรือ 'ผลลัพธ์ระยะยาว' โดยชัดเจน (L5-Focus)"
        else:
            return "กรุณาหาหลักฐานที่สอดคล้องกับเกณฑ์ที่ต้องการ"


    def _retrieve_context(
                self, 
                query: str, 
                sub_criteria_id: str, 
                level: int, 
                mapping_data: Optional[Dict[str, Any]] = None,
                statement_number: int = 1 
            ) -> Union[Dict[str, Any], List[str]]:
                """
                Retrieves relevant context based on the query and current sub-criteria/level.
                """
                
                doc_ids_to_filter = []
                final_mapping_key = None 
                data_found = None
                
                if self.use_mapping_filter:
                    mapping_data_to_use = self.evidence_mapping_data
                    
                    if mapping_data is not None:
                        mapping_data_to_use = mapping_data 
                    
                    # 1. สร้างและใช้ Key แบบ Level-only เท่านั้น เช่น "1.1_L1"
                    key_level = f"{sub_criteria_id}_L{level}"
                    data_found = mapping_data_to_use.get(key_level)
                    final_mapping_key = key_level
                    
                    if data_found:
                        logger.info(f"RAG Filter Check: Found data using Level-only key: {key_level}")
                    else:
                        logger.info(f"RAG Filter Check: Level-only key {key_level} not found.")

                    # 2. ดึง Doc IDs จาก Structure ที่พบ
                    if data_found and isinstance(data_found, dict):
                        evidences = data_found.get("evidences", [])
                        
                        # 🟢 FIX: ดึง Stable UUID จาก Mapping File โดยมี doc_id เป็น Fallback
                        doc_ids_to_filter = [
                            d.get('stable_doc_uuid') or d.get('doc_id')
                            for d in evidences 
                            if isinstance(d, dict) and (d.get('stable_doc_uuid') or d.get('doc_id'))
                        ]
                    
                    logger.info(f"RAG Filter Check: Final Key used: {final_mapping_key} - Found {len(doc_ids_to_filter)} doc_ids for Hard Filter.")
                
                # เรียกใช้ retrieve_context_with_filter
                result = retrieve_context_with_filter(
                    query=query,
                    doc_type=EVIDENCE_DOC_TYPES,
                    enabler=self.enabler_abbr,
                    stable_doc_ids=doc_ids_to_filter, 
                    top_k_reranked=self.FINAL_K_RERANKED,
                    disable_semantic_filter=self.disable_semantic_filter
                )
                
                return result
        
    def _process_subcriteria_results(self):
        """
        จัดกลุ่มผลลัพธ์ LLM ตาม Sub-Criteria และคำนวณคะแนนสุดท้าย 
        """
        grouped_results: Dict[str, Dict] = {}
        for r in self.raw_llm_results:
            key = r['sub_criteria_id'] 
            if key not in grouped_results:
                enabler_data = next((e for e in self.evidence_data 
                                     if e.get("Sub_Criteria_ID") == r['sub_criteria_id']), {}) 
                
                grouped_results[key] = {
                    "enabler_id": self.enabler_abbr.upper(),
                    "sub_criteria_id": r['sub_criteria_id'],
                    "sub_criteria_name": enabler_data.get("Sub_Criteria_Name_TH", "N/A"),
                    "weight": enabler_data.get("Weight", 1.0),
                    "raw_llm_scores": [],
                    "level_pass_ratios": {}, 
                    "num_statements_per_level": {} 
                }
            grouped_results[key]["raw_llm_scores"].append(r)

        for key, data in grouped_results.items():
            level_statements: Dict[int, List[Dict]] = {} 
            for r in data["raw_llm_scores"]:
                level = r["level"]
                if level not in level_statements:
                    level_statements[level] = []
                level_statements[level].append(r) 
            
            for level, results in level_statements.items():
                total_statements = len(results)
                
                passed_statements = sum(r.get("llm_score", 0) for r in results) 
                
                level_str = str(level)
                
                data["level_pass_ratios"][level_str] = round(passed_statements / total_statements, 3)
                data["num_statements_per_level"][level_str] = total_statements

        self.final_subcriteria_results = []
        for key, data in grouped_results.items():
            scoring_results = self._compute_subcriteria_score(
                level_pass_ratios=data["level_pass_ratios"],
                sub_criteria_weight=data["weight"]
            )
            data.update(scoring_results)
            self.final_subcriteria_results.append(data)

    def _get_source_name_for_display(self, doc_id: str, metadata: Dict[str, Any]) -> str:
        """Helper to determine the source name for display, prioritizing common metadata keys safely."""
        if not metadata or not isinstance(metadata, dict):
            return "N/A"

        if metadata.get("source_name_for_display"):
            return metadata["source_name_for_display"]

        for key in ["source_file", "file_name", "source", "document_name", "title"]:
            val = metadata.get(key)
            if val and isinstance(val, str) and val.strip():
                return val.strip()

        if doc_id and doc_id != "N/A":
            return f"Document ({doc_id[:8]}...)"

        return "N/A"

    def summarize_results(self) -> Dict[str, Dict]:
        """
        สรุปคะแนนรวมจาก final_subcriteria_results (ตามโครงสร้างที่ผู้ใช้กำหนด)
        """
        if not self.final_subcriteria_results:
             return {
                 "Overall": {
                    "enabler": self.enabler_abbr.upper(),
                    "total_weighted_score": 0.0,
                    "total_possible_weight": 0.0,
                    "overall_progress_percent": 0.0,
                    "overall_maturity_score": 0.0
                 },
                 "SubCriteria_Breakdown": {}
             }
        
        total_weight = sum(r["weight"] for r in self.final_subcriteria_results)
        total_score = sum(r["progress_score"] for r in self.final_subcriteria_results)
        
        # NOTE: การคำนวณ Overall Maturity Level ตาม Level Fractions ถูกละไว้ในโค้ดนี้ 
        # เพื่อคงไว้ตามโครงสร้างที่ผู้ใช้ระบุล่าสุด
        
        return {
            "Overall": {
                "enabler": self.enabler_abbr.upper(),
                "total_weighted_score": round(total_score, 2),
                "total_possible_weight": round(total_weight, 2),
                "overall_progress_percent": round((total_score / total_weight) * 100, 2) if total_weight > 0 else 0.0,
                "overall_maturity_score": round(total_score / total_weight, 2) if total_weight > 0 else 0.0
            },
            "SubCriteria_Breakdown": {
                r["sub_criteria_id"]: {
                    "topic": r.get("sub_criteria_name", "N/A"),
                    "score": r["progress_score"],
                    "weight": r["weight"],
                    "highest_full_level": r["highest_full_level"],
                    "pass_ratios": r["level_pass_ratios"], 
                    "development_gap": r["development_gap"],
                    "action_item": r["action_item"]
                } for r in self.final_subcriteria_results
            }
        }


    def run_assessment(self, target_doc_ids_or_filter_status: Union[List[str], str] = 'none') -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Run assessment across all levels & subtopics
        """
        self.raw_llm_results = [] 
        self.final_subcriteria_results = []
        
        is_mock_mode = self.mock_llm_eval_func is not None
        mapping_data_for_mock = self.evidence_mapping_data if is_mock_mode else None 
        
        llm_eval_func = self.mock_llm_eval_func if self.mock_llm_eval_func else evaluate_with_llm

        for enabler in self.evidence_data:
            enabler_id = enabler.get("Enabler_ID")
            sub_criteria_id = enabler.get("Sub_Criteria_ID")
            sub_criteria_name = enabler.get("Sub_Criteria_Name_TH", "N/A")

            if self.target_sub_id and self.target_sub_id != sub_criteria_id:
                continue

            for level in range(1, 6):
                level_key = f"Level_{level}_Statements"
                statements: List[str] = enabler.get(level_key, [])
                
                # 🟢 NEW: ใช้ Augmented Statements ถ้ามี
                augmented_statements = enabler.get(f"Augmented_Level_{level}_Statements", [])
                if augmented_statements:
                    statements = augmented_statements

                if not statements:
                    continue 
                
                rubric_criteria = self.global_rubric_map.get(level, {})

                # 🟢 NEW: เตรียม constraint สำหรับ level นี้
                constraint_prompt = self._get_level_constraint_prompt(level)
                
                for i, statement in enumerate(statements):
                    subtopic_key = f"subtopic_{i+1}"
                    standard = rubric_criteria.get(subtopic_key, f"Default standard L{level} S{i+1}")

                    # 🟢 NEW: รวม sub_criteria_name + constraint_prompt ใน query
                    query_string = (
                        f"หัวข้อ: {sub_criteria_name}\n"
                        f"ระดับวุฒิภาวะที่ประเมิน: {level}\n"
                        f"ข้อความเกณฑ์: {statement}\n\n"
                        f"{constraint_prompt}\n"
                        "ให้ค้นหาเฉพาะหลักฐานจากเอกสารจริงขององค์กร เช่น แผนยุทธศาสตร์ รายงานผลการดำเนินงาน หรือบันทึกการประชุม."
                    )
                    
                    logger.info(
                        f"DEBUG QUERY: SubCriteria={sub_criteria_id} Level={level} "
                        f"Statement={i+1} Subtopic={subtopic_key}\nQuery:\n{query_string}"
                    )

                    retrieval_result = self._retrieve_context(
                        query=query_string,
                        sub_criteria_id=sub_criteria_id, 
                        level=level,
                        mapping_data=mapping_data_for_mock, 
                        statement_number=i + 1
                    )

                    context_list = []
                    context_length = 0
                    retrieved_sources_list = [] 
                    context = "" 
                    
                    if isinstance(retrieval_result, dict):
                        top_evidence = retrieval_result.get("top_evidences", [])
                        
                        logger.info(f"DEBUG RAG: {sub_criteria_id}_L{level}_S{i+1} Retrieved {len(top_evidence)} raw evidences. Filter: {self.use_mapping_filter}")
                        
                        k_to_use = self.FINAL_K_RERANKED
                        if self.disable_semantic_filter:
                            k_to_use = self.FINAL_K_NON_RERANKED

                        for idx, doc in enumerate(top_evidence[:k_to_use]): 
                            doc_content = doc.get("content", "")
                            metadata = doc.get("metadata", {}) 
                            
                            doc_id = metadata.get("stable_doc_uuid", metadata.get("doc_id", "N/A")) 
                            source_name = self._get_source_name_for_display(doc_id, metadata)
                            location_value = str(metadata.get("page_label") or metadata.get("page", "N/A"))
                            
                            if location_value in ("N/A", "None") and doc_id != "N/A":
                                chunk_idx = metadata.get("chunk_index")
                                location_value = f"Chunk {chunk_idx}" if chunk_idx else "N/A"
                                
                            logger.info(f"DEBUG RAG Evidence (Top {idx + 1}): UUID={doc_id[:7]}... Source={source_name[:35]}... Location={location_value}")
                
                            snippet = doc_content[:self.MAX_SNIPPET_LENGTH]

                            retrieved_sources_list.append({
                                "source_name": source_name,
                                "doc_id": doc_id,
                                "location": location_value, 
                                "snippet_for_display": snippet, 
                            })
                            
                            if context_length + len(doc_content) <= self.MAX_CONTEXT_LENGTH:
                                context_list.append(doc_content)
                                context_length += len(doc_content)
                            else:
                                remaining_len = self.MAX_CONTEXT_LENGTH - context_length
                                if remaining_len > 0:
                                    context_list.append(doc_content[:remaining_len])
                                context_length = self.MAX_CONTEXT_LENGTH
                                break
                                
                        context = "\n---\n".join(context_list)
                    
                    llm_kwargs = {
                        "level": level, 
                        "sub_criteria_id": sub_criteria_id,
                        "statement_number": i + 1
                    }
                    
                    raw_llm_response_content = ""
                    llm_result_dict = {}

                    if is_mock_mode:
                        llm_result_dict = llm_eval_func(
                            statement=statement, context=context, standard=standard, **llm_kwargs
                        )
                        try:
                            raw_llm_response_content = json.dumps(llm_result_dict, ensure_ascii=False, indent=2)
                        except:
                            raw_llm_response_content = str(llm_result_dict)
                    else:
                        llm_output = llm_eval_func(
                            statement=statement,
                            context=context,
                            standard=standard,
                            enabler_name=self.enabler_abbr,  
                            **llm_kwargs
                        )
                        if isinstance(llm_output, tuple) and len(llm_output) == 2:
                            llm_result_dict, raw_llm_response_content = llm_output
                        elif isinstance(llm_output, dict):
                            llm_result_dict = llm_output
                            try:
                                raw_llm_response_content = json.dumps(llm_result_dict, ensure_ascii=False, indent=2)
                            except:
                                raw_llm_response_content = str(llm_result_dict)
                        else:
                            logger.error(f"Unexpected return type from LLM evaluation: {type(llm_output)}")
                            llm_result_dict = {}

                    unique_sources = []
                    seen = set()
                    
                    if is_mock_mode:
                        final_score = llm_result_dict.get("llm_score", 0)
                        final_reason = llm_result_dict.get("reason", "")
                        final_sources = llm_result_dict.get("retrieved_sources_list", []) 
                        final_context_snippet = llm_result_dict.get("context_retrieved_snippet", "") 
                        final_pass_status = llm_result_dict.get("pass_status", False)
                        final_status_th = llm_result_dict.get("status_th", "ไม่ผ่าน")
                    else:
                        for src in retrieved_sources_list:
                            key = (src['doc_id'], src['location']) 
                            if key not in seen:
                                seen.add(key)
                                unique_sources.append(src)

                        final_score = llm_result_dict.get("score", 0) 
                        final_reason = llm_result_dict.get("reason", "")
                        final_sources = unique_sources 
                        final_context_snippet = context[:240] + "..." if context else ""
                        final_pass_status = final_score == 1
                        final_status_th = "ผ่าน" if final_pass_status else "ไม่ผ่าน"

                    self.raw_llm_results.append({
                        "enabler_id": self.enabler_abbr.upper(),
                        "sub_criteria_id": sub_criteria_id,
                        "sub_criteria_name": sub_criteria_name, 
                        "level": level,
                        "statement_number": i + 1, 
                        "statement": statement,
                        "subtopic": subtopic_key,
                        "standard": standard,
                        "llm_score": final_score, 
                        "reason": final_reason,
                        "retrieved_sources_list": final_sources, 
                        "context_retrieved_snippet": final_context_snippet, 
                        "pass_status": final_pass_status,
                        "status_th": final_status_th,
                        "statement_id": f"{sub_criteria_id}_L{level}_S{i+1}",
                        "llm_result": llm_result_dict, 
                        "llm_raw_response_content": raw_llm_response_content 
                    })
        
        # 🟢 Post-processing phase
        self._process_subcriteria_results()
        final_summary = self.summarize_results()
        action_plan = {}
        return {"summary": final_summary, "action_plan": action_plan}, {f"{r['statement_id']}": r for r in self.raw_llm_results}

    def generate_evidence_summary_for_level(self, sub_criteria_id: str, level: int) -> Dict[str, Any]: 
        """
        รวมบริบทจากทุก Statement ใน Sub-Criteria/Level ที่กำหนด และให้ LLM สร้างคำอธิบาย
        """
        # 1. เช็คข้อมูล Enabler
        enabler_data = next((e for e in self.evidence_data 
                            if e.get("Sub_Criteria_ID") == sub_criteria_id), None)
        
        if not enabler_data:
            self.logger.error(f"Sub-Criteria ID {sub_criteria_id} not found in evidence data.")
            # ปรับปรุง: คืนค่าเป็น Dict
            return {"summary": "ไม่พบข้อมูลเกณฑ์ย่อยที่กำหนด", "suggestion_for_next_level": "N/A"}

        level_key = f"Level_{level}_Statements"
        statements: List[str] = enabler_data.get(level_key, [])
        sub_criteria_name = enabler_data.get("Sub_Criteria_Name_TH", "N/A")

        # 2. เช็ค Statements
        if not statements:
            # ปรับปรุง: คืนค่าเป็น Dict
            return {"summary": f"ไม่พบ Statements ใน Level {level} สำหรับเกณฑ์ {sub_criteria_id}", "suggestion_for_next_level": "N/A"}

        aggregated_context_list = []
        total_context_length = 0
        
        is_mock_mode = self.mock_llm_summarize_func is not None
        mapping_data_for_mock = self.evidence_mapping_data if is_mock_mode else None 
        
        k_to_use = self.FINAL_K_RERANKED 
        if self.disable_semantic_filter:
            k_to_use = self.FINAL_K_NON_RERANKED 
        
        # 3. วนลูป Retrieval และรวบรวม Context
        for i, statement in enumerate(statements):
            query_string = f"{statement} ({sub_criteria_name})"
            
            retrieval_result = self._retrieve_context(
                query=query_string,
                sub_criteria_id=sub_criteria_id,
                level=level,
                mapping_data=mapping_data_for_mock,
                statement_number=i + 1
            )
            
            if isinstance(retrieval_result, dict):
                top_evidence = retrieval_result.get("top_evidences", [])
                
                for doc in top_evidence[:k_to_use]: 
                    doc_content = doc.get("content", "")
                    
                    if total_context_length + len(doc_content) <= self.MAX_CONTEXT_LENGTH:
                        aggregated_context_list.append(doc_content)
                        total_context_length += len(doc_content)
                    else:
                        # ปรับปรุง: ใช้การตัดสตริงที่ง่ายขึ้นเมื่อเกินขีดจำกัด
                        remaining_content = doc_content[:self.MAX_CONTEXT_LENGTH - total_context_length]
                        if remaining_content:
                            aggregated_context_list.append(remaining_content)
                        total_context_length = self.MAX_CONTEXT_LENGTH
                        break 
        
        # 4. เช็ค Context ที่รวบรวมได้
        if not aggregated_context_list:
            return {"summary": f"ไม่พบหลักฐานที่เกี่ยวข้องใน Vector Store สำหรับเกณฑ์ {sub_criteria_id} Level {level}", "suggestion_for_next_level": "N/A"}
        
        # Deduplicate context while preserving order
        final_context = "\n---\n".join(list(dict.fromkeys(aggregated_context_list)))
        
        # 5. สรุปด้วย LLM
        try:
            summarize_func = self.mock_llm_summarize_func if self.mock_llm_summarize_func else summarize_context_with_llm
            
            summary_result = summarize_func(
                context=final_context,
                sub_criteria_name=sub_criteria_name,
                level=level,
                sub_id=sub_criteria_id,
                schema=None 
            )
            
            if isinstance(summary_result, dict):
                return summary_result
            else:
                # ตรวจสอบและคืนค่าในรูปแบบ Dict เสมอ
                summary_text = summary_result if isinstance(summary_result, str) else "LLM return type ไม่ถูกต้อง"
                return {"summary": summary_text, "suggestion_for_next_level": "N/A"}
            
        except Exception as e:
            self.logger.error(f"Failed to generate summary with LLM: {e}")
            return {"summary": f"เกิดข้อผิดพลาดในการเรียกใช้ LLM เพื่อสรุปข้อมูล: {e}", "suggestion_for_next_level": "Error"}

            
    def _integrate_evidence_summaries(self, results: Dict) -> Dict:
        """
        Generates and integrates L5 Summary and Highest Full Level Summary into the results dictionary.
        This uses the class's internal generate_evidence_summary_for_level method.
        (Refactored from core/run_assessment.py)
        """
        updated_breakdown = {}
        sub_criteria_breakdown = results.get("SubCriteria_Breakdown", {})
        
        for sub_id, sub_data in sub_criteria_breakdown.items():
            try:
                if isinstance(sub_data, str):
                    sub_data = {"name": sub_data}
                
                # 🟢 ดึง Highest Full Level
                highest_full_level = sub_data.get('highest_full_level', 0)
                
                # 1. --- Generate L5 Summary ---
                l5_summary_result = {"summary": "ไม่พบหลักฐานที่เกี่ยวข้องสำหรับ Level 5 ในฐานข้อมูล RAG.", "suggestion_for_next_level": "N/A"}
                try:
                    # 🛑 ใช้ generate_evidence_summary_for_level() ของ EnablerAssessment สำหรับ Level 5
                    l5_context_info = self.generate_evidence_summary_for_level(sub_id, 5)
                    l5_summary_result = l5_context_info

                    if isinstance(l5_summary_result, str):
                        if "ไม่พบหลักฐาน" in l5_summary_result:
                            l5_summary_result = {"summary": l5_summary_result, "suggestion_for_next_level": "N/A (No Evidence Found)"}
                        else:
                            l5_summary_result = {"summary": l5_summary_result, "suggestion_for_next_level": "N/A"}
                    
                    if not l5_summary_result.get("summary", "").strip():
                         l5_summary_result = {"summary": "ไม่พบหลักฐานที่เกี่ยวข้องสำหรับ Level 5 ในฐานข้อมูล RAG.", "suggestion_for_next_level": "N/A"}

                except Exception as e:
                    logger.error(f"[ERROR] L5 Summary generation failed for {sub_id}: {e}", exc_info=True)
                    l5_summary_result = {"summary": f"Error during L5 summary: {e}", "suggestion_for_next_level": "N/A"}


                # ผสานรวม L5 Summary
                sub_data["evidence_summary_L5"] = l5_summary_result


                # 2. --- Generate Highest Full Level Summary ---
                highest_summary_result = {"summary": "N/A (Level 0 or Error)", "suggestion_for_next_level": "N/A"}
                
                if highest_full_level > 0 and highest_full_level <= 5:
                    if highest_full_level == 5:
                        # ถ้า Highest Full Level คือ 5 ให้ใช้ผลลัพธ์เดียวกับ L5 (ลดการเรียก LLM ซ้ำ)
                        highest_summary_result = l5_summary_result
                    else:
                        try:
                            # 🛑 ใช้ generate_evidence_summary_for_level() ของ EnablerAssessment สำหรับ Level ที่ผ่านสูงสุด
                            highest_context_info = self.generate_evidence_summary_for_level(sub_id, highest_full_level)
                        except Exception:
                            highest_context_info = None

                        # 🛑 การจัดการผลลัพธ์ Highest Level
                        highest_summary_result = highest_context_info

                        if isinstance(highest_summary_result, str):
                            if "ไม่พบหลักฐาน" in highest_summary_result:
                                highest_summary_result = {"summary": highest_summary_result, "suggestion_for_next_level": "N/A (No Evidence Found)"}
                            else:
                                highest_summary_result = {"summary": highest_summary_result, "suggestion_for_next_level": "N/A"}
                        
                        if not highest_summary_result.get("summary", "").strip():
                            highest_summary_result = {"summary": f"ไม่พบหลักฐานที่เกี่ยวข้องสำหรับ Level {highest_full_level} ในฐานข้อมูล RAG.", "suggestion_for_next_level": "N/A"}

                # 🟢 ผสานรวม Highest Full Level Summary โดยใช้คีย์ที่สร้างแบบ Dynamic
                highest_summary_key = f"evidence_summary_L{highest_full_level}"
                sub_data[highest_summary_key] = highest_summary_result

                updated_breakdown[sub_id] = sub_data
                
            except Exception as e_outer:
                # Handle error gracefully
                logger.error(f"[ERROR] during summary integration for {sub_id}: {e_outer}", exc_info=True)
                
                error_summary = {"summary": f"Error: {e_outer}", "suggestion_for_next_level": "N/A"}
                sub_data["evidence_summary_L5"] = error_summary
                
                # Ensure the highest_summary key is also updated in case of error
                highest_summary_key = f"evidence_summary_L{sub_data.get('highest_full_level', 0)}"
                sub_data[highest_summary_key] = {"summary": f"Error: {e_outer} (Highest Level Summary)", "suggestion_for_next_level": "N/A"}
                updated_breakdown[sub_id] = sub_data
                
        results["SubCriteria_Breakdown"] = updated_breakdown
        return results


    def generate_action_plan_sub(
        self, 
        sub_id: str, 
        enabler: str, 
        summary_data: Optional[dict] = None, 
        full_summary: Optional[dict] = None
    ) -> list[ActionPlanActions]:
        """
        ✅ Generate Action Plan per Sub-Criteria using LLM (Robust + Schema Validation)
        """
        summary_data = next(
            (r for r in self.final_subcriteria_results if r['sub_criteria_id'] == sub_id),
            {}
        )
        highest_full_level = summary_data.get('highest_full_level', 0)
        target_level = highest_full_level + 1

        # Base Cases
        if not summary_data.get('development_gap', False):
            self.logger.info(f"[Action Plan] {sub_id}: Passed all levels (L{highest_full_level}). No action needed.")
            return [
                ActionPlanActions(
                    Phase="No Action Needed",
                    Goal=f"Sub-Criteria {sub_id} ผ่านครบถึง Level {highest_full_level}",
                    Actions=[]
                )
            ]

        if target_level > 5:
            self.logger.info(f"[Action Plan] {sub_id}: Already at L5. Recommend maintenance only.")
            return [
                ActionPlanActions(
                    Phase="L5 Maintenance",
                    Goal="รักษาความสม่ำเสมอของกระบวนการและผลลัพธ์ในระดับ 5",
                    Actions=[]
                )
            ]

        # ดึง failed statements
        all_failed_statements = self._get_all_failed_statements()
        failed_statements_for_sub = [s for s in all_failed_statements if s['sub_id'] == sub_id]
        self.logger.debug(f"[Action Plan] Found {len(failed_statements_for_sub)} failed statements for {sub_id}.")

        # Prepare Target Criteria
        target_criteria_statements = []
        enabler_entry = next(
            (e for e in self.evidence_data if e.get('Sub_Criteria_ID') == sub_id),
            None
        )
        if enabler_entry:
            criteria_key = f"Level_{target_level}_Statements"
            target_criteria_statements = enabler_entry.get(criteria_key, [])

        # Generate Action Plan
        llm_action_plan_result: list[dict] = []

        if not failed_statements_for_sub:
            self.logger.warning(f"[Action Plan] {sub_id}: No failed statements found. Using general evidence plan.")
            llm_action_plan_result = [{
                "Phase": f"{target_level}. General Evidence Collection",
                "Goal": f"รวบรวมหลักฐานเพิ่มเติมเพื่อผ่านเกณฑ์ Level {target_level}",
                "Actions": [{
                    "Statement_ID": f"ALL_L{target_level}",
                    "Failed_Level": target_level,
                    "Recommendation": f"ไม่พบ Statement ที่ล้มเหลว โปรดตรวจสอบหลักฐานเพิ่มเติมของ Level {target_level}",
                    "Target_Evidence_Type": "Policy, Record, Report",
                    "Key_Metric": "ผ่านครบทุก Statement ในการประเมินรอบถัดไป",
                    "Steps": []
                }]
            }]
        else:
            try:
                if self.mock_llm_action_plan_func:
                    self.logger.info(f"[Action Plan] MOCK mode → {sub_id} (Target L{target_level})")
                    llm_action_plan_result = self.mock_llm_action_plan_func(
                        failed_statements_data=failed_statements_for_sub,
                        sub_id=sub_id,
                        enabler=enabler,
                        target_level=target_level,
                    )
                else:
                    self.logger.info(f"[Action Plan] REAL LLM mode → {sub_id} (Target L{target_level})")
                    import core.retrieval_utils
                    llm_action_plan_result = core.retrieval_utils.create_structured_action_plan(
                        failed_statements_data=failed_statements_for_sub,
                        sub_id=sub_id,
                        enabler=enabler,
                        target_level=target_level,
                        max_retries=3
                    )
            except Exception as e:
                self.logger.error(f"[ERROR] Action Plan generation failed for {sub_id}: {e}", exc_info=True)
                llm_action_plan_result = [{
                    "Phase": "Error",
                    "Goal": f"สร้าง Action Plan ไม่สำเร็จ (Target L{target_level})",
                    "Actions": [{
                        "Statement_ID": "LLM_ERROR",
                        "Failed_Level": target_level,
                        "Recommendation": f"System Error: {str(e)} → ตรวจสอบหลักฐานและรันใหม่",
                        "Target_Evidence_Type": "Manual Collection",
                        "Key_Metric": "LLM Process Fixed",
                        "Steps": []
                    }]
                }]

        # Normalize & validate
        validated_action_plans: list[ActionPlanActions] = []

        if isinstance(llm_action_plan_result, dict):
            llm_action_plan_result = [llm_action_plan_result]

        for plan in llm_action_plan_result:
            # normalize Steps types
            actions = plan.get("Actions", [])
            for action in actions:
                # Steps must be list
                steps = action.get("Steps", [])
                if steps is None:
                    action["Steps"] = []
                elif isinstance(steps, dict):
                    action["Steps"] = [steps]
                else:
                    # ensure Step is str
                    for step in steps:
                        step["Step"] = str(step.get("Step", "")).strip()
                # Normalize Statement_ID
                sid = action.get("Statement_ID", "")
                action["Statement_ID"] = str(sid).replace(" ", "_").replace("-", "_").upper()

            try:
                validated_plan = ActionPlanActions(**plan)
                validated_action_plans.append(validated_plan)
            except Exception as e:
                self.logger.error(f"[ERROR] Validation failed for LLM plan: {plan}\n{e}", exc_info=True)
                validated_action_plans.append(
                    ActionPlanActions(
                        Phase="Validation Error",
                        Goal=f"LLM output invalid, Target L{target_level}",
                        Actions=[]
                    )
                )

        # Append AI Validation Maintenance
        validated_action_plans.append(
            ActionPlanActions(
                Phase="2. AI Validation & Maintenance",
                Goal=f"ยืนยัน Level-Up และรักษาความต่อเนื่องของหลักฐานใน L{target_level}",
                Actions=[ActionItem(
                    Statement_ID=f"ALL_L{target_level}",
                    Failed_Level=target_level,
                    Recommendation=f"รวบรวมหลักฐานใหม่สำหรับ Level {target_level} แล้วนำเข้าระบบ AI Assessment เพื่อยืนยันผล",
                    Target_Evidence_Type="Assessment Re-run",
                    Key_Metric=f"Highest Full Level ของ {sub_id} เพิ่มเป็น L{target_level}",
                    Steps=[]
                )]
            )
        )

        self.logger.info(f"[Action Plan READY] {sub_id}: {len(failed_statements_for_sub)} fails at levels {[s.get('level', 'N/A') for s in failed_statements_for_sub]}")

        return validated_action_plans



    def _add_pass_status_and_extract_uuids(self):
        """
        Internal method to process raw results: 
        1. Determine pass status (score >= 1).
        2. Extract chunk UUIDs mentioned by the LLM for post-retrieval.
        Updates self.raw_llm_results in place.
        """
        if not self.raw_llm_results:
            return

        for item in self.raw_llm_results:
            llm_res = item.get('llm_result', {})
            score = item.get('llm_score') or item.get('score')
            passed = False
            is_passed_from_sub = llm_res.get('is_passed')
            
            # 1. Determine Pass Status
            if isinstance(is_passed_from_sub, bool):
                passed = is_passed_from_sub
            elif score is not None:
                try:
                    if int(float(score)) >= 1:
                        passed = True
                except (ValueError, TypeError):
                    pass # treat invalid score as fail/pass based on default 

            item['pass_status'] = passed
            item['status_th'] = "ผ่าน" if passed else "ไม่ผ่าน"
            item['sub_criteria_id'] = item.get('sub_criteria_id', 'N/A')
            item['level'] = item.get('level', 0)
            item['statement_id'] = item.get('statement_id', 'N/A')

            # 2. Extract UUIDs
            llm_raw_response = item.get('llm_raw_response_content', '')
            if llm_raw_response:
                extracted_uuids = extract_uuids_from_llm_response(
                    llm_raw_response, 
                    key_hint=["chunk_uuids", "evidence_uuids", "doc_uuids"] 
                )
                item['llm_extracted_chunk_uuids'] = extracted_uuids
            else:
                item['llm_extracted_chunk_uuids'] = []


    def _retrieve_full_source_info_and_update(self):
        """
        Internal method to collect unique chunk UUIDs, retrieve full document details,
        and merge the results back into self.raw_llm_results.
        """
        all_uuids = set()
        for r in self.raw_llm_results:
            uuids = r.get('llm_extracted_chunk_uuids')
            if uuids:
                all_uuids.update(uuids)

        if not all_uuids:
            logger.info("ℹ️ No Chunk UUIDs extracted from LLM results for post-retrieval.")
            return
        
        # 🛑 CONSTRUCT COLLECTION NAME
        collection_name = f"{EVIDENCE_DOC_TYPES}_{self.enabler_abbr.lower()}"
        logger.info(f"Retrieving source info from collection: {collection_name} for {len(all_uuids)} chunks.")
        
        try:
            # 🛑 Call the new function from retrieval_utils
            # (ต้องมั่นใจว่า retrieve_context_by_doc_ids ถูก Import ใน enabler_assessment.py)
            retrieval_result = retrieve_context_by_doc_ids(list(all_uuids), collection_name) 
        except Exception as e:
            logger.error(f"Failed to retrieve full source info by doc ids: {e}", exc_info=True)
            return
        
        full_docs_map: Dict[str, Dict[str, Any]] = {
            doc.get("chunk_uuid"): {
                "chunk_uuid": doc.get("chunk_uuid"),
                "doc_id": doc.get("doc_id"),
                "source_name": doc.get("source"), 
                "location": doc.get("doc_type"), 
                "content_snippet": (doc.get("content", "")[:100] + "...") if doc.get("content") else ""
            } for doc in retrieval_result.get("top_evidences", []) if doc.get("chunk_uuid")
        }
        
        # Merge results back into raw_llm_results
        for r in self.raw_llm_results:
            chunk_uuids = r.get('llm_extracted_chunk_uuids', [])
            source_info_list = []
            for uuid in chunk_uuids:
                if uuid in full_docs_map:
                    source_info_list.append(full_docs_map[uuid])
            
            # Store the full source info list
            r['retrieved_full_source_info'] = source_info_list


    def print_detailed_results(self, target_sub_id: str):
        """
        แสดงผลลัพธ์การประเมิน LLM ราย Statement พร้อม Source File
        โดยใช้ข้อมูลที่ประมวลผลแล้ว (self.raw_llm_results)
        """
        if not self.raw_llm_results:
            logger.info("\n[Detailed Results] No raw LLM results found to display.")
            return

        raw_llm_results = self.raw_llm_results
        
        grouped: Dict[str, Dict[int, List[Dict]]] = {}
        for r in raw_llm_results:
            sub_id = r.get('sub_criteria_id')
            level = r.get('level')
            if sub_id and level is not None:
                 grouped.setdefault(sub_id, {}).setdefault(level, []).append(r)
            
        for sub_id in sorted(grouped.keys()):
            if target_sub_id != "all" and sub_id != target_sub_id:
                 continue
            
            print(f"\n--- Sub-Criteria ID: {sub_id} ---")
            for level in sorted(grouped[sub_id].keys()):
                level_results = grouped[sub_id][level]
                total_statements = len(level_results)
                # ใช้ 'llm_score'
                # passed_statements = sum(r.get('llm_score', 0) for r in level_results) 
                passed_statements = sum(1 for r in level_results if r.get('llm_score', 0) >= 1)
                pass_ratio = passed_statements / total_statements if total_statements > 0 else 0.0
                print(f"  > Level {level} ({passed_statements}/{total_statements}, Pass Ratio: {pass_ratio:.3f})")
                
                for r in level_results:
                    llm_score = r.get('llm_score', 0)
                    score_text = f"✅ PASS | Score: {llm_score}" if llm_score == 1 else f"❌ FAIL | Score: {llm_score}"
                    fail_status = "" if llm_score == 1 else "❌ FAIL |"
                    statement_num = r.get('statement_number', 'N/A')
                    print(f"    - S{statement_num}: {fail_status} Statement: {r.get('statement', 'N/A')[:100]}...")
                    print(f"      [Score]: {score_text}")
                    print(f"      [Reason]: {r.get('reason', 'N/A')[:120]}...")

                    # 🟢 ใช้ sources จาก retrieved_full_source_info
                    sources = r.get('retrieved_full_source_info', []) 
                    if sources:
                         print("      [SOURCE FILES]:")
                         for src in sources:
                             source_name = src.get('source_name', 'Unknown File')
                             location = src.get('location', 'N/A')
                             chunk_uuid = src.get('chunk_uuid', 'N/A') 
                             uuid_short = chunk_uuid[:8] + "..." if chunk_uuid else "N/A"
                             print(f"        -> {source_name} (Location: {location}, UUID: {uuid_short})")
                    print(f"      [Context]: {r.get('context_retrieved_snippet', 'N/A')}")