import os
import json
import logging
import sys
import re 
from typing import List, Dict, Any, Optional, Union
import time 

# --- PATH SETUP (Must be executed first for imports to work) ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    from core.vectorstore import load_all_vectorstores, FINAL_K_RERANKED 
    # NOTE: ต้องมั่นใจว่า evaluate_with_llm และ summarize_context_with_llm ถูกเรียกใช้ใน Core/Retrieval_utils
    # 🛑 FIX: ต้อง import evaluate_with_llm_with_raw_response แทน evaluate_with_llm เดิม
    from core.retrieval_utils import (
        evaluate_with_llm, 
        retrieve_context_with_filter, 
        set_mock_control_mode, 
        summarize_context_with_llm,
        # 🟢 NEW IMPORT: Assuming retrieval_utils now has a function that returns raw response
        # If not, we call evaluate_with_llm and assume it returns a tuple (dict, str) in real mode
    )
    from core.assessment_schema import EvidenceSummary
    from core.action_plan_schema import ActionPlanActions

except ImportError as e:
    print(f"FATAL ERROR: Failed to import required modules. Check sys.path and file structure. Error: {e}", file=sys.stderr)
    sys.exit(1)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Level Fractions สำหรับ Linear Interpolation
DEFAULT_LEVEL_FRACTIONS = {
    "1": 0.16667,
    "2": 0.33333,
    "3": 0.50000,
    "4": 0.66667,
    "5": 0.83333,
    "MAX_LEVEL_FRACTION": 1.00000, 
    "0": 0.0 
}

# Default fallback rubric structure
DEFAULT_RUBRIC_STRUCTURE = {
    "Default_Maturity_Rubric": { 
        "levels": [
            {
                "level": i,
                "name": "Default",
                "criteria": {
                    f"subtopic_{j}": f"Default criteria for level {i}, subtopic {j}"
                    for j in range(1, 4)
                }
            }
            for i in range(1, 6)
        ]
    }
}


class EnablerAssessment:
    """
    Automated Enabler Maturity Assessment System
    ประเมินระดับวุฒิภาวะของ Enabler ใดๆ (KM, SCM, DT ฯลฯ)
    """

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "evidence_checklist"))
    
    # Context length limit สำหรับการประเมินแต่ละ Statement
    MAX_CONTEXT_LENGTH = 2500 


    def __init__(self,
                 enabler_abbr: str, 
                 evidence_data: Optional[List] = None,
                 rubric_data: Optional[Dict] = None,
                 level_fractions: Optional[Dict] = None,
                 evidence_mapping_data: Optional[Dict] = None, # สำหรับ Mapping File
                 vectorstore_retriever=None,
                 # Argument สำหรับควบคุม Filter 
                 use_retrieval_filter: bool = False, 
                 target_sub_id: Optional[str] = None, # e.g., '1.1'
                 # Mock/Control LLM Function Override - 🛑 FIX: เพิ่ม 2 Argument ใหม่
                 mock_llm_eval_func=None,
                 mock_llm_summarize_func=None,
                 mock_llm_action_plan_func=None):
        
        self.enabler_abbr = enabler_abbr.lower()
        self.enabler_rubric_key = f"{self.enabler_abbr.upper()}_Maturity_Rubric"
        self.vectorstore_retriever = vectorstore_retriever
        
        # DYNAMIC FILENAMES
        self.EVIDENCE_FILE = os.path.join(self.BASE_DIR, f"{self.enabler_abbr}_evidence_statements_checklist.json")
        self.RUBRIC_FILE = os.path.join(self.BASE_DIR, f"{self.enabler_abbr}_rating_criteria_rubric.json")
        self.LEVEL_FRACTIONS_FILE = os.path.join(self.BASE_DIR, f"{self.enabler_abbr}_scoring_level_fractions.json")
        self.MAPPING_FILE = os.path.join(self.BASE_DIR, f"{self.enabler_abbr}_evidence_mapping.json")
        # self.MAPPING_FILE = os.path.join(self.BASE_DIR, f"{self.enabler_abbr}_mapping_by_level.json")

        # LOAD DATA
        self.evidence_data = evidence_data or self._load_json_fallback(self.EVIDENCE_FILE, default=[])
        default_rubric = {self.enabler_rubric_key: DEFAULT_RUBRIC_STRUCTURE["Default_Maturity_Rubric"]}
        self.rubric_data = rubric_data or self._load_json_fallback(self.RUBRIC_FILE, default=default_rubric)
        self.level_fractions = level_fractions or self._load_json_fallback(self.LEVEL_FRACTIONS_FILE, default=DEFAULT_LEVEL_FRACTIONS)
        self.evidence_mapping_data = evidence_mapping_data or self._load_json_fallback(self.MAPPING_FILE, default={})
        
        # เก็บสถานะ Filter
        self.use_retrieval_filter = use_retrieval_filter
        self.target_sub_id = target_sub_id
        
        # เก็บ Mock Function
        self.mock_llm_eval_func = mock_llm_eval_func 
        self.mock_llm_summarize_func = mock_llm_summarize_func # 🛑 NEW
        self.mock_llm_action_plan_func = mock_llm_action_plan_func # 🛑 NEW

        self.raw_llm_results: List[Dict] = []
        self.final_subcriteria_results: List[Dict] = []
        
        self.global_rubric_map: Dict[int, Dict[str, str]] = self._prepare_rubric_map()


    def _load_json_fallback(self, path: str, default: Any):
        """Loads JSON หากไฟล์ไม่มี ให้ใช้ default"""
        if not os.path.isfile(path):
            logger.warning(f"[Warning] JSON file not found for enabler '{self.enabler_abbr}': {path}, using default fallback.")
            return default
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load JSON {path}: {e}")
            return default

    def _prepare_rubric_map(self) -> Dict[int, Dict[str, str]]:
        """
        แปลง Global Rubric เป็น Map ง่ายต่อการเรียกใช้ โดยใช้คีย์ Enabler ปัจจุบัน
        """
        rubric_map = {}
        rubric_data_entry = self.rubric_data.get(self.enabler_rubric_key)
        
        if self.rubric_data and isinstance(rubric_data_entry, dict):
            for level_entry in rubric_data_entry.get("levels", []):
                level_num = level_entry.get("level")
                if level_num:
                    rubric_map[level_num] = level_entry.get("criteria", {})
        else:
             logger.warning(f"⚠️ Rubric key '{self.enabler_rubric_key}' not found in loaded rubric data. Using default/fallback structure.")
             if self.enabler_rubric_key not in self.rubric_data:
                default_data = DEFAULT_RUBRIC_STRUCTURE["Default_Maturity_Rubric"]
                for level_entry in default_data.get("levels", []):
                     level_num = level_entry.get("level")
                     if level_num:
                         rubric_map[level_num] = level_entry.get("criteria", {})

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
        if highest_full_level == 5:
            progress_score = self.level_fractions.get("MAX_LEVEL_FRACTION", 1.0) * sub_criteria_weight
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


    def _get_metadata_filter(self) -> Optional[Dict]:
        """Return None เสมอ เพราะ Filter Logic ถูกย้ายไปใช้กับ document ID"""
        return None 
    
    # 🌟 NEW HELPER METHOD: สำหรับสร้างข้อจำกัดให้กับ Query ตาม Level
    def _get_level_constraint_prompt(self, level: int) -> str:
        """
        สร้าง Prompt Constraint เพื่อบอก LLM/Vector Search ให้กรองหลักฐาน L3/L4/L5 ออก
        เพื่อให้ RAG ดึงหลักฐานที่เหมาะสมกับระดับวุฒิภาวะ
        """
        # หลักการ: ห้ามดึงหลักฐานที่มีระดับสูงกว่าระดับที่กำลังประเมิน
        if level == 1:
            # L1: ห้ามดึงหลักฐาน L3-L5 (การบูรณาการ, นวัตกรรม, การวัดผลระยะยาว)
            return "ข้อจำกัด: หลักฐานต้องเกี่ยวกับ 'การเริ่มต้น', 'การมีอยู่', หรือ 'การวางแผน' เท่านั้น ห้ามเกี่ยวข้องกับ 'การปรับปรุงอย่างต่อเนื่อง', 'การบูรณาการ', 'นวัตกรรม', หรือ 'การวัดผลลัพธ์ระยะยาว' (L1-Filter)"
        elif level == 2:
            # L2: อนุญาต L1/L2 เท่านั้น ห้ามดึงหลักฐาน L4-L5
            return "ข้อจำกัด: หลักฐานต้องเกี่ยวกับ 'การปฏิบัติ', 'การทำให้เป็นมาตรฐาน', หรือ 'การประเมินเบื้องต้น' ห้ามเกี่ยวข้องกับ 'การบูรณาการ', 'นวัตกรรม', หรือ 'การวัดผลลัพธ์ระยะยาว' (L2-Filter)"
        elif level == 3:
            # L3: มุ่งเน้นไปที่การควบคุมและการวัดผลระยะสั้น ห้าม L5
            return "ข้อจำกัด: หลักฐานควรเกี่ยวกับ 'การควบคุม', 'การกำกับดูแล', หรือ 'การวัดผลเบื้องต้น' ห้ามเกี่ยวข้องกับ 'นวัตกรรม', หรือ 'การสร้างคุณค่าทางธุรกิจระยะยาว' (L3-Filter)"
        elif level == 4:
            # L4: อนุญาตทุกอย่างยกเว้น L5 (เน้นการบูรณาการและการปรับปรุง)
            return "ข้อจำกัด: หลักฐานควรแสดงถึง 'การบูรณาการ' หรือ 'การปรับปรุงอย่างต่อเนื่อง' ห้ามเกี่ยวข้องกับการพิสูจน์ 'คุณค่าทางธุรกิจระยะยาว' (L4-Filter)"
        elif level == 5:
            # L5: ไม่จำกัด แต่เน้นเฉพาะคำสำคัญของ L5
            return "ข้อจำกัด: หลักฐานควรแสดงถึง 'นวัตกรรม', 'การสร้างคุณค่าทางธุรกิจ', หรือ 'ผลลัพธ์ระยะยาว' โดยชัดเจน (L5-Focus)"
        else:
            return "กรุณาหาหลักฐานที่สอดคล้องกับเกณฑ์ที่ต้องการ"


    def _retrieve_context(self, query: str, sub_criteria_id: str, level: int, mapping_data: Optional[Dict] = None, statement_number: int = 0) -> Dict[str, Any]:
        """
        ดึง Context โดยใช้ Filter จาก evidence mapping และ Metadata Filter ตาม Sub ID ที่ส่งมา
        
        🛑 อัพเดทตรรกะ: ใช้ Constrained Query เสมอ เพื่อเป็นเกราะป้องกันการ Mapping ที่ผิดพลาด
        """
        effective_mapping_data = mapping_data if mapping_data is not None else self.evidence_mapping_data
        
        if not self.vectorstore_retriever and mapping_data is None:
            logger.warning("Vectorstore retriever is None and not in Mock Mode. Skipping RAG retrieval.")
            return {"top_evidences": []} 

        # 1. สร้างคีย์สำหรับ Mapping: "1.1_L1", "1.1_L2", ...
        mapping_key = f"{sub_criteria_id}_L{level}"
        
        # 2. ดึง Filter IDs (Hard Filter - อาจจะว่างเปล่า)
        filter_ids: List[str] = effective_mapping_data.get(mapping_key, {}).get("filter_ids", [])
        
        # 3. 🌟 (NEW) สร้าง Constrained Query เสมอ
        constraint_instruction = self._get_level_constraint_prompt(level)
        effective_query = f"{query}. {constraint_instruction}"
        
        logger.info(f"🔑 Constrained Query (L{level}, Filtered={bool(filter_ids)}): {effective_query}")
        
        # --- LOGIC สำหรับ REAL MODE (mapping_data is None) ---
        if mapping_data is None: 

            # 4. เรียกใช้ RAG Retrieval 
            # - query: ใช้ effective_query (มี Level Constraint เสมอ)
            # - metadata_filter: ใช้ filter_ids (จะเป็น Hard Filter ถ้ามี ID, หรือเป็น Full-Scope ถ้าว่างเปล่า)
            result = retrieve_context_with_filter(
                query=effective_query, 
                retriever=self.vectorstore_retriever, 
                metadata_filter=filter_ids 
            )
            return result

        # --- LOGIC สำหรับ MOCK MODE ---
        return {"top_evidences": []} # ใน Mock Mode การเรียกนี้ควรถูก Patch ใน run_assessment2.py


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

        # คำนวณ Pass Ratio
        for key, data in grouped_results.items():
            level_statements: Dict[int, List[Dict]] = {} 
            for r in data["raw_llm_scores"]:
                level = r["level"]
                if level not in level_statements:
                    level_statements[level] = []
                level_statements[level].append(r) 
            
            for level, results in level_statements.items():
                total_statements = len(results)
                
                # ใช้ 'llm_score' ที่เป็น 0/1 
                passed_statements = sum(r.get("llm_score", 0) for r in results) 
                
                level_str = str(level)
                
                data["level_pass_ratios"][level_str] = round(passed_statements / total_statements, 3)
                data["num_statements_per_level"][level_str] = total_statements

        # คำนวณ Final Score (Detailed Score)
        self.final_subcriteria_results = []
        for key, data in grouped_results.items():
            scoring_results = self._compute_subcriteria_score(
                level_pass_ratios=data["level_pass_ratios"],
                sub_criteria_weight=data["weight"]
            )
            data.update(scoring_results)
            self.final_subcriteria_results.append(data)


    def run_assessment(self) -> List[Dict]:
        """
        Run assessment across all levels & subtopics
        """
        self.raw_llm_results = [] 
        self.final_subcriteria_results = []
        
        # ตรวจสอบว่ามีการ Patch ฟังก์ชันหรือไม่ 
        is_mock_mode = self.mock_llm_eval_func is not None
        mapping_data_for_mock = self.evidence_mapping_data if is_mock_mode else None
        
        # เลือกใช้ LLM Evaluation Function (Mock หรือ Real)
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
                
                if not statements:
                    continue 
                
                rubric_criteria = self.global_rubric_map.get(level, {})
                
                for i, statement in enumerate(statements):
                    subtopic_key = f"subtopic_{i+1}"
                    standard = rubric_criteria.get(subtopic_key, f"Default standard L{level} S{i+1}")
                    
                    # 1. สร้าง Query String
                    query_string = f"{statement} ({sub_criteria_name})"
                    
                    # 2. เรียก retrieval_result
                    retrieval_result = self._retrieve_context(
                        query=query_string,
                        sub_criteria_id=sub_criteria_id, 
                        level=level,
                        mapping_data=mapping_data_for_mock, 
                        statement_number=i + 1
                    )
                    
                    # 3. ขยาย Context String
                    context_list = []
                    context_length = 0
                    retrieved_sources_list = [] 
                    context = "" 
                    
                    if isinstance(retrieval_result, dict):
                        top_evidence = retrieval_result.get("top_evidences", [])
                        
                        for doc in top_evidence[:FINAL_K_RERANKED]: 
                            doc_content = doc.get("content", "")
                            source_name = doc.get("source", "N/A (No Source Tag)")
                            location = doc.get("metadata", {}).get("page_number", doc.get("doc_id", "N/A"))
                            location_str = f"Page {location}" if isinstance(location, int) else location
                            doc_id = doc.get("doc_id", "N/A")
                            retrieved_sources_list.append({
                                "source_name": source_name,
                                "doc_id": doc_id,
                                "location": location_str
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
                    
                    # 4. Call the selected evaluation function
                    llm_kwargs = {
                        "level": level, 
                        "sub_criteria_id": sub_criteria_id,
                        "statement_number": i + 1
                    }
                    
                    # 🟢 FIX: Call LLM และตรวจสอบผลลัพธ์เพื่อแยก Dict และ Raw Text
                    raw_llm_response_content = ""
                    llm_result_dict = {}

                    if is_mock_mode:
                        # Mock Mode: mock_llm_eval_func ถูกตั้งให้ return dict ที่มีข้อมูลที่ต้องการ
                        llm_result_dict = llm_eval_func(
                            statement=statement, context=context, standard=standard, **llm_kwargs
                        )
                        # ใน Mock Mode เราจะสมมติว่า raw response คือ JSON string ของ dict ที่ return มา
                        try:
                            raw_llm_response_content = json.dumps(llm_result_dict, ensure_ascii=False, indent=2)
                        except:
                            raw_llm_response_content = str(llm_result_dict)
                    else:
                        # Real Mode: เราต้องสมมติว่า evaluate_with_llm ถูกแก้ไขให้ return (Dict, Str)
                        # หรือถ้าไม่ เราก็เรียกใช้ evaluate_with_llm เดิม และสมมติว่ามีการแก้ไขใน retrieval_utils
                        # ในโค้ดนี้ เราจะปรับให้รองรับการรับค่า return เป็น tuple (Dict, Str)
                        llm_output = llm_eval_func(
                            statement=statement, context=context, standard=standard, **llm_kwargs
                        )
                        if isinstance(llm_output, tuple) and len(llm_output) == 2:
                            llm_result_dict, raw_llm_response_content = llm_output
                        elif isinstance(llm_output, dict):
                            llm_result_dict = llm_output
                            # หาก LLM Function เดิมไม่คืนค่า Raw Response ให้ใช้ JSON ของ Dict เป็น Fallback (แต่ควรแก้ไข evaluate_with_llm ใน retrieval_utils)
                            try:
                                raw_llm_response_content = json.dumps(llm_result_dict, ensure_ascii=False, indent=2)
                            except:
                                raw_llm_response_content = str(llm_result_dict)
                        else:
                            logger.error(f"Unexpected return type from LLM evaluation: {type(llm_output)}")
                            llm_result_dict = {}

                    
                    # 5. Deduplicate sources & Final Data Structuring 
                    unique_sources = []
                    seen = set()
                    
                    if is_mock_mode:
                        # ใน Mock Mode: ใช้ค่า Context/Sources/Status/Score ที่มาจาก Mock Function โดยตรง
                        final_score = llm_result_dict.get("llm_score", 0)
                        final_reason = llm_result_dict.get("reason", "")
                        final_sources = llm_result_dict.get("retrieved_sources_list", []) # 👈 ใช้ Mock Sources
                        final_context_snippet = llm_result_dict.get("context_retrieved_snippet", "") # 👈 ใช้ Mock Context
                        final_pass_status = llm_result_dict.get("pass_status", False)
                        final_status_th = llm_result_dict.get("status_th", "ไม่ผ่าน")
                    else:
                        # ใน Real Mode: ใช้ Context/Sources ที่ได้จาก RAG
                        for src in retrieved_sources_list:
                            # 🟢 FIXED: ใช้ doc_id และ location เป็นคีย์ในการ Deduplicate
                            key = (src['doc_id'], src['location']) 
                            if key not in seen:
                                seen.add(key)
                                unique_sources.append(src)

                        final_score = llm_result_dict.get("score", 0) # ใช้ score จาก LLM จริง
                        final_reason = llm_result_dict.get("reason", "")
                        final_sources = unique_sources # จาก RAG
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
                        "statement_id": "N/A", 
                        "llm_result": llm_result_dict, # 🟢 เก็บ Dict Result เดิมไว้
                        "llm_raw_response_content": raw_llm_response_content # 🟢 NEW: เก็บ Raw Response สำหรับดึง UUIDs
                    })
        
        self._process_subcriteria_results()
        
        return self.final_subcriteria_results

    # ----------------------------------------------------
    # 🌟 NEW FEATURE: Generate Evidence Summary
    # ----------------------------------------------------
    def generate_evidence_summary_for_level(self, sub_criteria_id: str, level: int) -> Union[str, Dict]: # 🟢 อัพเดท Type Hint
        """
        รวมบริบทจากทุก Statement ใน Sub-Criteria/Level ที่กำหนด และให้ LLM สร้างคำอธิบาย
        """
        # 1. ค้นหา Statement Data
        enabler_data = next((e for e in self.evidence_data 
                             if e.get("Sub_Criteria_ID") == sub_criteria_id), None)
        
        if not enabler_data:
            logger.error(f"Sub-Criteria ID {sub_criteria_id} not found in evidence data.")
            return "ไม่พบข้อมูลเกณฑ์ย่อยที่กำหนด"

        level_key = f"Level_{level}_Statements"
        statements: List[str] = enabler_data.get(level_key, [])
        sub_criteria_name = enabler_data.get("Sub_Criteria_Name_TH", "N/A")

        if not statements:
            return f"ไม่พบ Statements ใน Level {level} สำหรับเกณฑ์ {sub_criteria_id}"

        aggregated_context_list = []
        total_context_length = 0
        
        # 2. วนลูปดึง Context
        for i, statement in enumerate(statements):
            query_string = f"{statement} ({sub_criteria_name})"
            
            # NOTE: การเรียก _retrieve_context ที่นี่จะใช้ Logic Constrained Query ที่ปรับปรุงใหม่แล้ว
            retrieval_result = self._retrieve_context(
                query=query_string,
                sub_criteria_id=sub_criteria_id,
                level=level,
                statement_number=i + 1
            )
            
            if isinstance(retrieval_result, dict):
                top_evidence = retrieval_result.get("top_evidences", [])
                
                for doc in top_evidence[:FINAL_K_RERANKED]: 
                    doc_content = doc.get("content", "")
                    
                    if total_context_length + len(doc_content) <= self.MAX_CONTEXT_LENGTH:
                        aggregated_context_list.append(doc_content)
                        total_context_length += len(doc_content)
                    else:
                        remaining_len = self.MAX_CONTEXT_LENGTH - total_context_length
                        if remaining_len > 0:
                            aggregated_context_list.append(doc_content[:remaining_len])
                        total_context_length = self.MAX_CONTEXT_LENGTH
                        break 
        
        if not aggregated_context_list:
            return f"ไม่พบหลักฐานที่เกี่ยวข้องใน Vector Store สำหรับเกณฑ์ {sub_criteria_id} Level {level}"
        
        final_context = "\n---\n".join(list(dict.fromkeys(aggregated_context_list)))
        
        # 3. เรียก LLM เพื่อสร้างคำอธิบายสรุป
        try:
            # 🛑 FIX: เลือกใช้ Mock/Real Function (ถ้ามีการส่งเข้า __init__ โดยตรง)
            # NOTE: ใน run_assessment2.py เราใช้ patch() ครอบอีกที ซึ่งจะ override ตรงนี้ แต่เราเขียนโค้ดเผื่อไว้
            summarize_func = self.mock_llm_summarize_func if self.mock_llm_summarize_func else summarize_context_with_llm
            
            # NOTE: summarize_context_with_llm ควรถูกแก้ไขให้ return Dict (EvidenceSummary) หรือ Str (Fallback)
            summary_result = summarize_func(
                context=final_context,
                sub_criteria_name=sub_criteria_name,
                level=level,
                sub_id=sub_criteria_id,
                schema=EvidenceSummary
            )
            
            # 🟢 FIXED: Return Dict เพื่อให้ run_assessment2.py จัดการได้ง่ายขึ้น
            if isinstance(summary_result, dict):
                return summary_result
            elif isinstance(summary_result, str):
                return {"summary": summary_result, "suggestion_for_next_level": "N/A"}
            else:
                return {"summary": "LLM return type ไม่ถูกต้อง", "suggestion_for_next_level": "N/A"}
            
        except Exception as e:
            logger.error(f"Failed to generate summary with LLM: {e}")
            return {"summary": f"เกิดข้อผิดพลาดในการเรียกใช้ LLM เพื่อสรุปข้อมูล: {e}", "suggestion_for_next_level": "Error"}
    # ----------------------------------------------------
    
    # ----------------------------------------------------
    # 🌟 NEW FEATURE: Generate Action Plan (เมธอดนี้ถูกใช้โดย generate_action_plan_for_sub ใน run_assessment2.py)
    # ----------------------------------------------------
    # NOTE: เมธอดนี้อาจไม่ได้ถูกเรียกใช้โดยตรงเพราะ run_assessment2.py เรียก generate_action_plan_for_sub()
    # แต่ถ้าคุณต้องการให้คลาสนี้มีเมธอด Action Plan ด้วย (เช่น เพื่อใช้กับฟังก์ชันจริงที่ไม่ได้ถูก patch)
    def generate_action_plan(self, sub_criteria_id: str) -> List[Dict]:
        """
        สร้าง Action Plan สำหรับ Sub-Criteria ที่กำหนด (เมธอดที่คลาสควรมี แต่ถูก Patch ใน run_assessment2.py)
        """
        # NOTE: การเรียกใช้ LLM จริงควรถูกเรียกจาก core.retrieval_utils 
        # ดังนั้นเมธอดนี้จะ Return ค่าว่าง เนื่องจาก Logic การสร้าง Action Plan ถูกรวมไว้ใน run_assessment2.py
        
        if self.mock_llm_action_plan_func:
            # ใช้ Mock Function ที่ถูกส่งเข้ามา (ถ้ามีการส่ง)
            return self.mock_llm_action_plan_func(
                failed_statements_data=[], # ต้องดึงข้อมูลจาก raw_llm_results
                sub_id=sub_criteria_id,
                target_level=0 # ต้องคำนวณ Target Level
            )
        
        logger.warning(f"generate_action_plan is stubbed. Action Plan logic is handled in run_assessment2.py.")
        return []
    # ----------------------------------------------------

    def summarize_results(self) -> Dict[str, Dict]:
        """
        สรุปคะแนนรวมจาก final_subcriteria_results
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
                    "name": r.get("sub_criteria_name", "N/A"),
                    "score": r["progress_score"],
                    "weight": r["weight"],
                    "highest_full_level": r["highest_full_level"],
                    "pass_ratios": r["level_pass_ratios"], 
                    "development_gap": r["development_gap"],
                    "action_item": r["action_item"]
                } for r in self.final_subcriteria_results
            }
        }