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
    
    # 🚨 FIX: ต้อง Import FINAL_K_RERANKED จาก core.vectorstore
    from core.vectorstore import load_all_vectorstores, FINAL_K_RERANKED 
    # 🚨 FIX: ต้อง Import summarize_context_with_llm สำหรับฟังก์ชันใหม่
    from core.retrieval_utils import evaluate_with_llm, retrieve_context_with_filter, set_mock_control_mode, summarize_context_with_llm

    from core.assessment_schema import EvidenceSummary
    from core.action_plan_schema import ActionPlanActions

except ImportError as e:
    # หากรันจากที่ที่ไม่ใช่ root project directory อาจจะต้องปรับ path เพิ่ม
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
                 # Argument สำหรับควบคุม Filter (ปัจจุบันไม่ถูกใช้งานใน _get_metadata_filter แต่คงไว้เพื่อความยืดหยุ่น)
                 use_retrieval_filter: bool = False, 
                 target_sub_id: Optional[str] = None, # e.g., '1.1'
                 # Mock/Control LLM Function Override
                 mock_llm_eval_func=None): # Default: core.retrieval_utils.evaluate_with_llm
        
        self.enabler_abbr = enabler_abbr.lower()
        self.enabler_rubric_key = f"{self.enabler_abbr.upper()}_Maturity_Rubric"
        self.vectorstore_retriever = vectorstore_retriever
        
        # DYNAMIC FILENAMES
        self.EVIDENCE_FILE = os.path.join(self.BASE_DIR, f"{self.enabler_abbr}_evidence_statements_checklist.json")
        self.RUBRIC_FILE = os.path.join(self.BASE_DIR, f"{self.enabler_abbr}_rating_criteria_rubric.json")
        self.LEVEL_FRACTIONS_FILE = os.path.join(self.BASE_DIR, f"{self.enabler_abbr}_scoring_level_fractions.json")
        self.MAPPING_FILE = os.path.join(self.BASE_DIR, f"{self.enabler_abbr}_evidence_mapping.json")

        # LOAD DATA
        self.evidence_data = evidence_data or self._load_json_fallback(self.EVIDENCE_FILE, default=[])
        # ตรวจสอบว่า rubric_data มีคีย์ Enabler ปัจจุบันหรือไม่ หากไม่มีจะใช้ Default Fallback
        default_rubric = {self.enabler_rubric_key: DEFAULT_RUBRIC_STRUCTURE["Default_Maturity_Rubric"]}
        self.rubric_data = rubric_data or self._load_json_fallback(self.RUBRIC_FILE, default=default_rubric)
        self.level_fractions = level_fractions or self._load_json_fallback(self.LEVEL_FRACTIONS_FILE, default=DEFAULT_LEVEL_FRACTIONS)
        self.evidence_mapping_data = evidence_mapping_data or self._load_json_fallback(self.MAPPING_FILE, default={})
        
        # เก็บสถานะ Filter
        self.use_retrieval_filter = use_retrieval_filter
        self.target_sub_id = target_sub_id
        
        # เก็บ Mock Function
        self.mock_llm_eval_func = mock_llm_eval_func 

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
             # การแจ้งเตือนนี้อาจเกิดขึ้นได้หากไฟล์ rubric_data ไม่มีคีย์ Enabler ที่ถูกต้อง แต่ใช้ Default Rubric แทน
             logger.warning(f"⚠️ Rubric key '{self.enabler_rubric_key}' not found in loaded rubric data. Using default/fallback structure.")
             
             # สร้าง Rubric Map จาก DEFAULT_RUBRIC_STRUCTURE หากไม่มีข้อมูลที่ถูกต้อง
             if self.enabler_rubric_key not in self.rubric_data:
                # ลองใช้ default key หากไม่มีคีย์เฉพาะของ Enabler
                default_data = DEFAULT_RUBRIC_STRUCTURE["Default_Maturity_Rubric"]
                for level_entry in default_data.get("levels", []):
                     level_num = level_entry.get("level")
                     if level_num:
                         rubric_map[level_num] = level_entry.get("criteria", {})


        return rubric_map


    def _compute_subcriteria_score(self, level_pass_ratios: Dict[str, float], sub_criteria_weight: float) -> Dict[str, Any]:
        """
        คำนวณคะแนนละเอียดตาม Linear Interpolation Logic (คง Logic Maturity Model ไว้)
        """
        highest_full_level = 0
        progress_score = 0.0
        
        # 1. หา Highest Fully Passed Level (1.0 ratio) - Logic Maturity Model
        for level in range(1, 6):
            level_str = str(level)
            # 🚨 FIX 1: ตรวจสอบ Ratio โดยใช้ Key String
            # ถ้า Level ปัจจุบันไม่ผ่านครบ 100%
            if level_pass_ratios.get(level_str, 0.0) < 1.0: 
                highest_full_level = level - 1 
                if highest_full_level < 0:
                    highest_full_level = 0
                break # หยุดทันทีเมื่อเจอ Level แรกที่ไม่ผ่าน
            else:
                highest_full_level = level # ถ้า Level ผ่านครบ ให้ตั้งเป็น Level นี้แล้ววนลูปต่อ
        
        # 2. คำนวณ Progress Score (Linear Interpolation)
        if highest_full_level == 5:
            progress_score = self.level_fractions.get("MAX_LEVEL_FRACTION", 1.0) * sub_criteria_weight
        else: # highest_full_level < 5
            # Level ที่ผ่านสมบูรณ์แล้ว (ฐาน)
            base_fraction = self.level_fractions.get(str(highest_full_level) if highest_full_level > 0 else "0", 0.0)
            
            # Level ที่กำลังมี Gap คือ Level ต่อไป
            gap_level = highest_full_level + 1 
            gap_fraction = self.level_fractions.get(str(gap_level), 0.0)
            
            # 🚨 FIX 2: ดึง progress_ratio จาก Level ที่มี Gap
            progress_ratio = level_pass_ratios.get(str(gap_level), 0.0)
            
            # เพิ่มคะแนนจากความคืบหน้าของ Level ที่มี Gap
            fraction_increase = (gap_fraction - base_fraction) * progress_ratio
            total_fraction = base_fraction + fraction_increase
            progress_score = total_fraction * sub_criteria_weight
        
        # 3. จัด Gap Analysis และ Action Item (ปรับปรุงให้ชัดเจน)
        action_item = ""
        development_gap = False
        target_gap_level = highest_full_level + 1
        
        if target_gap_level <= 5: 
             development_gap = True
             ratio = level_pass_ratios.get(str(target_gap_level), 0.0)
             
             # Action Item จะโฟกัสไปที่ Gap แรกสุด
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
        """
        ฟังก์ชันนี้ถูกทิ้งไว้แต่จะ return None เสมอ เพราะ Filter Logic ถูกย้ายไปใช้กับ document ID
        ที่ได้จาก Mapping File ใน _retrieve_context แทน
        """
        return None 


    def _retrieve_context(self, query: str, sub_criteria_id: str, level: int, mapping_data: Optional[Dict] = None, statement_number: int = 0) -> Dict[str, Any]:
        """
        ดึง Context โดยใช้ Filter จาก evidence mapping และ Metadata Filter ตาม Sub ID ที่ส่งมา
        """
        effective_mapping_data = mapping_data if mapping_data is not None else self.evidence_mapping_data
        
        if not self.vectorstore_retriever and mapping_data is None:
            logger.warning("Vectorstore retriever is None and not in Mock Mode. Skipping RAG retrieval.")
            # Return empty structure if RAG is skipped
            return {"top_evidences": []} 

        # 1. สร้างคีย์สำหรับ Mapping: "1.1_L1", "1.1_L2", ...
        mapping_key = f"{sub_criteria_id}_L{level}"
        
        # 2. ดึง Filter IDs (รายชื่อไฟล์ที่ถูก Clean แล้ว) จาก effective_mapping_data
        filter_ids: List[str] = effective_mapping_data.get(mapping_key, {}).get("filter_ids", [])
        
        
        # --- LOGIC สำหรับ REAL MODE (mapping_data is None) ---
        if mapping_data is None: 
            if not filter_ids:
                logger.warning(f"No filter IDs found for {mapping_key}. Retrieving context without doc_id restriction.")

            # 4. เรียกใช้ RAG Retrieval (ส่ง filter_ids ที่ได้จาก mapping ไปที่ retrieve_context_with_filter)
            result = retrieve_context_with_filter(
                query=query, 
                retriever=self.vectorstore_retriever, 
                metadata_filter=filter_ids # ส่ง filter_ids (doc_id) ไปที่ RAG
            )
            
            # 5. ส่งผลลัพธ์ที่เป็น Dictionary กลับไป
            return result

        # --- LOGIC สำหรับ MOCK MODE ---
        # ถ้าอยู่ใน Mock Mode แต่ฟังก์ชันนี้ไม่ได้ถูก Patch จะ return ค่าว่าง
        return {"top_evidences": []} 


    def _process_subcriteria_results(self):
        """
        จัดกลุ่มผลลัพธ์ LLM ตาม Sub-Criteria และคำนวณคะแนนสุดท้าย (แก้ไข Key ใน Grouping)
        """
        grouped_results: Dict[str, Dict] = {}
        for r in self.raw_llm_results:
            # ใช้ Sub-Criteria ID เป็น Key หลักในการจัดกลุ่ม
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
            level_statements: Dict[int, List[Dict]] = {} # 🚨 FIX 3: เก็บทั้ง Dict เพื่อเข้าถึง pass_status
            for r in data["raw_llm_scores"]:
                level = r["level"]
                if level not in level_statements:
                    level_statements[level] = []
                level_statements[level].append(r) # 🚨 FIX 4: เก็บผลลัพธ์ LLM ทั้งก้อน
            
            for level, results in level_statements.items():
                total_statements = len(results)
                
                # 🚨 FIX 5: นับ 'ผ่าน' จาก 'llm_score' ที่ตอนนี้ถูกกำหนดให้เป็น 1 เมื่อ 'pass_status' เป็น True
                # (ตาม logic ที่แก้ไขใน retrieval_utils.py)
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
        
        # ตรวจสอบว่ามีการ Patch ฟังก์ชันหรือไม่ (เพื่อส่ง Mapping Data ไปให้ Mock)
        is_mock_mode = getattr(self._retrieve_context, '__name__', 'N/A') == 'retrieve_context_MOCK'
        mapping_data_for_mock = self.evidence_mapping_data if is_mock_mode else None
        
        # เลือกใช้ LLM Evaluation Function (Mock หรือ Real)
        llm_eval_func = self.mock_llm_eval_func if self.mock_llm_eval_func else evaluate_with_llm

        
        for enabler in self.evidence_data:
            enabler_id = enabler.get("Enabler_ID")
            sub_criteria_id = enabler.get("Sub_Criteria_ID")
            sub_criteria_name = enabler.get("Sub_Criteria_Name_TH", "N/A")

            # ตรวจสอบว่ามีการกรองเฉพาะ Sub ID หรือไม่
            if self.target_sub_id and self.target_sub_id != sub_criteria_id:
                continue

            for level in range(1, 6):
                level_key = f"Level_{level}_Statements"
                statements: List[str] = enabler.get(level_key, [])
                
                if not statements:
                    continue 
                
                rubric_criteria = self.global_rubric_map.get(level, {})
                
                for i, statement in enumerate(statements):
                    # 💡 NOTE: subtopic key ใช้สำหรับดึงมาตรฐานจาก rubric_criteria
                    subtopic_key = f"subtopic_{i+1}"
                    # ดึงมาตรฐาน: ใช้ criteria ที่ i+1 หรือใช้ default fallback
                    standard = rubric_criteria.get(subtopic_key, f"Default standard L{level} S{i+1}")
                    
                    # 🚨 FIX 1: สร้าง Query String ที่รวม Statement และ Sub Criteria Name เพื่อให้ RAG ทำงานได้ดีขึ้น
                    query_string = f"{statement} ({sub_criteria_name})"
                    
                    # 1. เรียก retrieval_result
                    retrieval_result = self._retrieve_context(
                        query=query_string, # ส่ง Query String ใหม่
                        sub_criteria_id=sub_criteria_id, 
                        level=level,
                        mapping_data=mapping_data_for_mock, 
                        statement_number=i + 1
                    )
                    
                    # 2. ขยาย Context String โดยการรวม Content จาก Top N Reranked Documents
                    context_list = []
                    context_length = 0
                    # 💡 NEW: Initialize list to store source/location data
                    retrieved_sources_list = [] 
                    context = "" # Initialize context string
                    
                    if isinstance(retrieval_result, dict):
                        top_evidence = retrieval_result.get("top_evidences", [])
                        
                        # ใช้ FINAL_K_RERANKED เป็นตัวกำหนดจำนวนเอกสารสูงสุดที่จะรวม
                        for doc in top_evidence[:FINAL_K_RERANKED]: 
                            doc_content = doc.get("content", "")
                            
                            # 💡 NEW: Extract Source Information
                            source_name = doc.get("source", "N/A (No Source Tag)")
                            # Assume 'page_number' is stored in metadata or directly. Use 'doc_id' as fallback.
                            location = doc.get("metadata", {}).get("page_number", doc.get("doc_id", "N/A"))
                            # Format location string
                            location_str = f"Page {location}" if isinstance(location, int) else location
                            
                            # 💡 NEW: Store Source Data for traceability
                            # ใช้ Doc ID และ Page Number เพื่อระบุแหล่งที่มาอย่างชัดเจน
                            doc_id = doc.get("doc_id", "N/A")
                            retrieved_sources_list.append({
                                "source_name": source_name,
                                "doc_id": doc_id,
                                "location": location_str
                            })
                            
                            # ตรวจสอบว่าความยาวเกิน MAX_CONTEXT_LENGTH หรือไม่
                            if context_length + len(doc_content) <= self.MAX_CONTEXT_LENGTH:
                                context_list.append(doc_content)
                                context_length += len(doc_content)
                            else:
                                # ถ้าเกิน ให้ตัดส่วนที่เหลือของ Content ชิ้นสุดท้ายออก
                                remaining_len = self.MAX_CONTEXT_LENGTH - context_length
                                if remaining_len > 0:
                                    context_list.append(doc_content[:remaining_len])
                                context_length = self.MAX_CONTEXT_LENGTH
                                break # หยุดการรวมเมื่อ Context เต็ม
                                
                        # รวมบริบทที่ถูกตัดให้ได้ความยาวตามต้องการ
                        context = "\n---\n".join(context_list)
                    
                    # 3. Call the selected evaluation function
                    result = llm_eval_func(
                        statement=statement,
                        context=context, # ส่ง String Context ที่ถูกขยาย
                        standard=standard,
                        #  🚨 FIX 3: ส่ง level, sub_criteria_id และ statement_number เข้าไปใน kwargs
                        # level=level, 
                        # sub_criteria_id=sub_criteria_id,
                        # statement_number=i + 1
                    )
                    
                    # 4. Deduplicate sources before saving (เพื่อไม่ให้แสดงซ้ำในรายงาน)
                    unique_sources = []
                    seen = set()
                    for src in retrieved_sources_list:
                        # ใช้ doc_id และ location เป็นคีย์ในการตรวจสอบความซ้ำ
                        # ในทางปฏิบัติ: Location ควรเป็น Page Number/Section
                        key = (src['doc_id'], src['location']) 
                        if key not in seen:
                            seen.add(key)
                            unique_sources.append(src)
                    
            # 5. บันทึกผลลัพธ์
                    
                    # 🚨 FIX LOGIC: กำหนดค่า Context/Sources/Status ตาม Mode
                    if is_mock_mode:
                        # ใน Mock Mode: ใช้ค่า Context/Sources/Status/Score ที่มาจาก Mock Function โดยตรง
                        final_score = result.get("llm_score", 0) # ใช้ llm_score ที่เป็น 0/1
                        final_reason = result.get("reason", "")
                        final_sources = result.get("retrieved_sources_list", []) # 👈 ใช้ Mock Sources
                        final_context_snippet = result.get("context_retrieved_snippet", "") # 👈 ใช้ Mock Context
                        final_pass_status = result.get("pass_status", False)
                        final_status_th = result.get("status_th", "ไม่ผ่าน")
                    else:
                        # ใน Real Mode: ใช้ Context/Sources ที่ได้จาก RAG (Step 2)
                        final_score = result.get("score", 0) # ใช้ score จาก LLM จริง
                        final_reason = result.get("reason", "")
                        final_sources = unique_sources # จาก Step 4 (RAG)
                        final_context_snippet = context[:120] + "..." if context else ""
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
                        # ใช้ค่าที่ถูกกำหนดตาม Mode
                        "llm_score": final_score, 
                        "reason": final_reason,
                        "retrieved_sources_list": final_sources, # 👈 ใช้ final_sources
                        "context_retrieved_snippet": final_context_snippet, # 👈 ใช้ final_context_snippet
                        "pass_status": final_pass_status,
                        "status_th": final_status_th,
                        "statement_id": "N/A" # เพิ่ม placeholder สำหรับ ID ที่ขาดไป
                    })
        
        self._process_subcriteria_results()
        
        return self.final_subcriteria_results
    
    # ----------------------------------------------------
    # 🌟 NEW FEATURE: Generate Evidence Summary
    # ----------------------------------------------------
    def generate_evidence_summary_for_level(self, sub_criteria_id: str, level: int) -> str:
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
        
        # 2. วนลูปดึง Context สำหรับทุก Statement ใน Level นั้น (รวมหลักฐานทั้งหมด)
        for i, statement in enumerate(statements):
            # สร้าง Query โดยรวม Statement และ Sub Criteria Name
            query_string = f"{statement} ({sub_criteria_name})"
            
            # ดึง Context โดยใช้ Filter เฉพาะ Sub-Criteria/Level นี้
            retrieval_result = self._retrieve_context(
                query=query_string,
                sub_criteria_id=sub_criteria_id,
                level=level,
                statement_number=i + 1
            )
            
            if isinstance(retrieval_result, dict):
                top_evidence = retrieval_result.get("top_evidences", [])
                
                # รวม Context จาก Top N evidences (เหมือนใน run_assessment)
                for doc in top_evidence[:FINAL_K_RERANKED]: 
                    doc_content = doc.get("content", "")
                    
                    if total_context_length + len(doc_content) <= self.MAX_CONTEXT_LENGTH:
                        aggregated_context_list.append(doc_content)
                        total_context_length += len(doc_content)
                    else:
                        remaining_len = self.MAX_CONTEXT_LENGTH - total_context_length
                        if remaining_len > 0:
                            # ตัดส่วนที่เหลือของ Content ชิ้นสุดท้ายออก
                            aggregated_context_list.append(doc_content[:remaining_len])
                        total_context_length = self.MAX_CONTEXT_LENGTH
                        break 
        
        if not aggregated_context_list:
            return f"ไม่พบหลักฐานที่เกี่ยวข้องใน Vector Store สำหรับเกณฑ์ {sub_criteria_id} Level {level}"
        
        # เชื่อม Context ที่รวบรวมได้ทั้งหมด (ใช้ dict.fromkeys เพื่อลบรายการที่ซ้ำกันก่อน join)
        final_context = "\n---\n".join(list(dict.fromkeys(aggregated_context_list)))
        
        # 3. เรียก LLM เพื่อสร้างคำอธิบายสรุป
        try:
            summary_result = summarize_context_with_llm(
                context=final_context,
                sub_criteria_name=sub_criteria_name,
                level=level,
                sub_id=sub_criteria_id,
                schema=EvidenceSummary
            )
            # ตรวจสอบ return type
            if isinstance(summary_result, dict):
                return summary_result.get("summary", "LLM ไม่สามารถสร้างคำอธิบายได้")
            elif isinstance(summary_result, str):
                return summary_result
            else:
                return "LLM return type ไม่ถูกต้อง"
            
        except Exception as e:
            logger.error(f"Failed to generate summary with LLM: {e}")
            return f"เกิดข้อผิดพลาดในการเรียกใช้ LLM เพื่อสรุปข้อมูล: {e}"
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
