# -------------------- assessments/km_assessment.py (FINAL FULL VERSION WITH RUBRIC FIX) --------------------
import os
import json
import logging
import sys
from typing import List, Dict, Any, Optional, Union

# ต้องแก้ไข sys.path ใน __main__ เพื่อให้ Import core ได้ (สมมติว่าจัดการแล้ว)
# NOTE: สมมติว่าไฟล์เหล่านี้มีอยู่จริง
from core.vectorstore import load_all_vectorstores
from core.retrieval_utils import evaluate_with_llm, retrieve_context, set_mock_control_mode 

logger = logging.getLogger(__name__)
# NOTE: ปรับ level logging ที่นี่เป็น INFO
logging.basicConfig(level=logging.INFO)


# Level Fractions สำหรับ Linear Interpolation (ค่าประมาณของ 1/6, 2/6, 3/6, 4/6, 5/6)
DEFAULT_LEVEL_FRACTIONS = {
    "1": 0.16667,
    "2": 0.33333,
    "3": 0.50000,
    "4": 0.66667,
    "5": 0.83333,
    "MAX_LEVEL_FRACTION": 1.00000, 
    "0": 0.0 
}

# Default fallback rubric structure (ใช้คีย์ที่ถูกต้อง)
DEFAULT_RUBRIC_STRUCTURE = {
    "KM_Maturity_Rubric": { 
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


class KMAssessment:
    """
    Automated KM Maturity Assessment System
    ประเมินระดับวุฒิภาวะ KM ขององค์กร
    """

    # BASE_DIR ที่ถูกต้องสำหรับการรันจาก Root (.. คือ Root Directory)
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "evidence_checklist"))
    EVIDENCE_FILE = os.path.join(BASE_DIR, "km_evidence_statements_checklist.json")
    RUBRIC_FILE = os.path.join(BASE_DIR, "km_rating_criteria_rubric.json")
    LEVEL_FRACTIONS_FILE = os.path.join(BASE_DIR, "km_scoring_level_fractions.json")

    def __init__(self,
                 evidence_data: Optional[List] = None,
                 rubric_data: Optional[Dict] = None,
                 level_fractions: Optional[Dict] = None,
                 vectorstore_retriever=None):
        
        self.evidence_data = evidence_data or self._load_json_fallback(self.EVIDENCE_FILE, default=[])
        self.rubric_data = rubric_data or self._load_json_fallback(self.RUBRIC_FILE, default=DEFAULT_RUBRIC_STRUCTURE)
        self.level_fractions = level_fractions or self._load_json_fallback(self.LEVEL_FRACTIONS_FILE, default=DEFAULT_LEVEL_FRACTIONS)
        self.vectorstore_retriever = vectorstore_retriever
        
        self.raw_llm_results: List[Dict] = []
        self.final_subcriteria_results: List[Dict] = []
        
        # เตรียม Global Rubric Map
        self.global_rubric_map: Dict[int, Dict[str, str]] = self._prepare_rubric_map()


    def _load_json_fallback(self, path: str, default: Any):
        """โหลด JSON หากไฟล์ไม่มี ให้ใช้ default"""
        if not os.path.isfile(path):
            logger.warning(f"[Warning] ไฟล์ JSON ไม่พบ: {path}, ใช้ default fallback")
            return default
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load JSON {path}: {e}")
            return default

    def _prepare_rubric_map(self) -> Dict[int, Dict[str, str]]:
        """
        แปลง Global Rubric เป็น Map ง่ายต่อการเรียกใช้
        FIX: ใช้คีย์ 'KM_Maturity_Rubric' แทน 'Global_Maturity_Rubric'
        """
        rubric_map = {}
        
        # 🟢 FIX HERE: ใช้คีย์ที่ถูกต้องตามไฟล์ JSON ของคุณ
        rubric_data_entry = self.rubric_data.get("KM_Maturity_Rubric")
        
        if self.rubric_data and isinstance(rubric_data_entry, dict):
            # ใช้ rubric_data_entry ในการวนซ้ำ
            for level_entry in rubric_data_entry.get("levels", []):
                level_num = level_entry.get("level")
                if level_num:
                    rubric_map[level_num] = level_entry.get("criteria", {})
        return rubric_map

    def _retrieve_context(self, statement: str) -> str:
        """ใช้ retrieve_context และส่ง LLM/Context กลับไปเป็น String"""
        result = retrieve_context(statement, self.vectorstore_retriever)
        contexts = [e["content"] for e in result.get("top_evidences", [])]
        return "\n".join(contexts)

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
        elif highest_full_level < 5:
            base_fraction = self.level_fractions.get(str(highest_full_level) if highest_full_level > 0 else "0", 0.0)
            next_level = highest_full_level + 1 
            next_fraction = self.level_fractions.get(str(next_level), 0.0)
            progress_ratio = level_pass_ratios.get(str(next_level), 0.0)
            
            fraction_increase = (next_fraction - base_fraction) * progress_ratio
            total_fraction = base_fraction + fraction_increase
            progress_score = total_fraction * sub_criteria_weight
        
        # 3. จัด Gap Analysis
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


    def _process_subcriteria_results(self):
        """
        จัดกลุ่มผลลัพธ์ LLM ตาม Sub-Criteria และคำนวณคะแนนสุดท้าย
        """
        grouped_results: Dict[str, Dict] = {}
        for r in self.raw_llm_results:
            key = f"{r['enabler_id']}-{r['sub_criteria_id']}"
            if key not in grouped_results:
                enabler_data = next((e for e in self.evidence_data 
                                     if e.get("Enabler_ID") == r['enabler_id'] and 
                                        e.get("Sub_Criteria_ID") == r['sub_criteria_id']), {})
                
                grouped_results[key] = {
                    "enabler_id": r['enabler_id'],
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
            level_scores: Dict[int, List[int]] = {}
            for r in data["raw_llm_scores"]:
                level = r["level"]
                if level not in level_scores:
                    level_scores[level] = []
                level_scores[level].append(r["llm_score"])
            
            for level, scores in level_scores.items():
                total_statements = len(scores)
                passed_statements = sum(scores)
                
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

        from core.retrieval_utils import _MOCK_CONTROL_FLAG
        if _MOCK_CONTROL_FLAG:
             set_mock_control_mode(True) 
        
        for enabler in self.evidence_data:
            enabler_id = enabler.get("Enabler_ID")
            sub_criteria_id = enabler.get("Sub_Criteria_ID")
            sub_criteria_name = enabler.get("Sub_Criteria_Name_TH", "N/A")

            for level in range(1, 6):
                level_key = f"Level_{level}_Statements"
                statements: List[str] = enabler.get(level_key, [])
                
                if not statements:
                    continue 
                
                rubric_criteria = self.global_rubric_map.get(level, {})
                
                for i, statement in enumerate(statements):
                    subtopic_key = f"subtopic_{i+1}"
                    standard = rubric_criteria.get(subtopic_key, f"Default standard L{level} S{i+1}")
                    
                    context = self._retrieve_context(statement)
                    
                    result = evaluate_with_llm(
                        statement=statement,
                        context=context,
                        standard=standard
                    )
                    
                    self.raw_llm_results.append({
                        "enabler_id": enabler_id,
                        "sub_criteria_id": sub_criteria_id,
                        "sub_criteria_name": sub_criteria_name, 
                        "level": level,
                        "statement": statement,
                        "subtopic": subtopic_key,
                        "standard": standard,
                        "llm_score": result.get("score", 0), 
                        "reason": result.get("reason", ""),
                        "context_retrieved_snippet": context[:100] + "..." 
                    })
        
        self._process_subcriteria_results()
        
        return self.final_subcriteria_results

    def summarize_results(self) -> Dict[str, Dict]:
        """
        สรุปคะแนนรวมจาก final_subcriteria_results
        """
        total_weight = sum(r["weight"] for r in self.final_subcriteria_results)
        total_score = sum(r["progress_score"] for r in self.final_subcriteria_results)
        num_subcriteria = len(self.final_subcriteria_results)
        
        return {
            "Overall": {
                "total_weighted_score": round(total_score, 2),
                "total_possible_weight": round(total_weight, 2),
                "overall_progress_percent": round((total_score / total_weight) * 100, 2) if total_weight > 0 else 0.0,
                "overall_maturity_score": round(total_score / num_subcriteria, 2) if num_subcriteria > 0 else 0.0
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