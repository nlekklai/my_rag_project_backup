import json
import argparse
import os
import sys
import logging
from typing import Dict, Any, List
from pathlib import Path

# 1. Setup Logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -------------------- Module Import Logic --------------------
# 1. Import NARRATIVE PROMPT
NARRATIVE_REPORT_PROMPT = None
SYSTEM_NARRATIVE_PROMPT = None
try:
    # 📝 IMPORT ทั้ง PROMPT Template และ System Instruction
    from core.rag_prompts import NARRATIVE_REPORT_PROMPT, SYSTEM_NARRATIVE_PROMPT 
    logger.info("Successfully imported NARRATIVE_REPORT_PROMPT and SYSTEM_NARRATIVE_PROMPT from core.rag_prompts.")
    NARRATIVE_REPORT_TEMPLATE_TEXT = NARRATIVE_REPORT_PROMPT.template 
except ImportError:
    NARRATIVE_REPORT_PROMPT = None
    SYSTEM_NARRATIVE_PROMPT = None
    NARRATIVE_REPORT_TEMPLATE_TEXT = "[ERROR: Failed to import NARRATIVE_REPORT_PROMPT. Check your core.rag_prompts.py and Python path.]"
    logger.error("Could not import NARRATIVE_REPORT_PROMPT.")

# 2. Import REAL LLM Function
_LLM_NARRATIVE_ENABLED = False
try:
    # 🚨 NEW: Import REAL LLM function from retrieval_utils 🚨
    from core.retrieval_utils import generate_narrative_report_via_llm_real 
    _LLM_NARRATIVE_ENABLED = True
    logger.info("Successfully imported REAL LLM function for narrative report.")
except ImportError:
    logger.warning("Could not import REAL LLM function (generate_narrative_report_via_llm_real). Narrative report will use MOCK FALLBACK.")


# --- Utility Functions for Data Synthesis (NEW LOGIC) ---

def calculate_overall_level(score: float) -> int:
    """Converts the overall maturity score (0.0 to 5.0) to a Maturity Level (L0 to L4)."""
    if 0.00 <= score < 1.00:
        return 0
    elif 1.00 <= score < 2.00:
        return 1
    elif 2.00 <= score < 3.00:
        return 2
    elif 3.00 <= score < 4.00:
        return 3
    elif 4.00 <= score <= 5.00:
        return 4
    return 0

def find_top_bottom_criteria(breakdown: Dict[str, Any]) -> dict:
    """
    Finds the Sub-Criteria with the highest and lowest scores, 
    and extracts relevant contextual data for the LLM.
    """
    
    # Filter out any criteria with non-numeric scores
    criteria_list = [
        {
            'score': data['score'],
            'id': key,
            'name': data['name'],
            'action_item': data.get('action_item', 'N/A'),
            'highest_full_level': data.get('highest_full_level', 0)
        }
        for key, data in breakdown.items() if isinstance(data.get('score'), (int, float))
    ]
    
    if not criteria_list:
        return {
            'top_criteria': {},
            'bottom_criteria': {}
        }

    # Sort criteria by score
    criteria_list.sort(key=lambda x: x['score'], reverse=True)

    top_criteria = criteria_list[0]
    bottom_criteria = criteria_list[-1]
    
    return {
        'top_criteria': top_criteria,
        'bottom_criteria': bottom_criteria
    }

def extract_top_goals(data: Dict[str, Any], limit: int = 3) -> List[str]:
    """Extracts unique goals from the action plans across all criteria."""
    
    unique_goals = set()
    
    # Iterate through each sub-criteria's action plans
    for sub_data in data.get("SubCriteria_Breakdown", {}).values():
        # Check if the action_plan_json exists and is a dictionary
        if "action_plan_json" in sub_data and isinstance(sub_data["action_plan_json"], dict):
            plan_json = sub_data["action_plan_json"]
            
            # Action plan JSON is structured as an array of phases
            for phase in plan_json.get("phases", []):
                goal = phase.get("Goal")
                if goal and isinstance(goal, str):
                    unique_goals.add(goal.strip())
                    
    # Convert set back to list for final output, taking the top N
    return list(unique_goals)[:limit]


# -------------------- NARRATIVE LLM FUNCTION (Updated for Real LLM Integration) --------------------

def generate_narrative_report_via_llm(summary_data: Dict[str, Any]) -> str:
    """
    ใช้ LLM สังเคราะห์ข้อมูลทั้งหมดใน summary_data ให้เป็นรายงานเชิงบรรยาย 4 ส่วนสำหรับ CEO 
    โดยเรียกใช้ฟังก์ชัน LLM จริงจาก core/retrieval_utils.py (ถ้าเปิดใช้งาน)
    """
    
    enabler_name = summary_data.get('Overall', {}).get('enabler', 'UNKNOWN ENABLER')

    # 🚨 FALLBACK LOGIC 🚨
    if not _LLM_NARRATIVE_ENABLED or NARRATIVE_REPORT_PROMPT is None or SYSTEM_NARRATIVE_PROMPT is None:
        # ใช้ MOCK FALLBACK หาก LLM Function, Prompt Template หรือ System Prompt ไม่พร้อมใช้งาน
        overall_progress = summary_data.get('Overall', {}).get('overall_progress_percent', 0.0)
        
        fallback_error_info = ""
        if not _LLM_NARRATIVE_ENABLED:
            fallback_error_info += " (LLM Function Not Imported)"
        if NARRATIVE_REPORT_PROMPT is None or SYSTEM_NARRATIVE_PROMPT is None:
            fallback_error_info += " (Prompt Template Missing)"
        
        return f"""
[🚨 รายงานเชิงบรรยายที่ถูกสร้างโดย MOCK FALLBACK {fallback_error_info} สำหรับ {enabler_name} 🚨]

## 1. ภาพรวมระดับองค์กร (Overall Maturity Summary)

เนื่องจากระบบ LLM ไม่พร้อมใช้งานในขณะนี้ รายงานจะแสดงค่า Summary หลัก: วุฒิภาวะ {enabler_name} โดยรวม: **Overall Progress ที่ {overall_progress:.2f}%** ซึ่งเป็นการสรุปผลโดยตรงจากตารางคะแนน (ไม่ผ่านการสังเคราะห์)

## 2. ข้อเสนอแนะเชิงกลยุทธ์สำหรับผู้บริหาร (Strategic Recommendations)

โปรดรันคำสั่งโดยไม่มี Flag `--narrative` เพื่อดูรายงานแบบตาราง (Structured Markdown) และ Action Plan โดยละเอียดซึ่งเป็นข้อมูลที่แม่นยำที่สุด

---
[System Info: Real LLM call was disabled or failed to initialize.]
"""
    
    # -------------------- REAL LLM CALL LOGIC --------------------
    try:
        # --- NEW: Data Pre-processing ---
        overall_data = summary_data.get("Overall", {})
        overall_score = overall_data.get("overall_maturity_score", 0.0)
        progress = overall_data.get("overall_progress_percent", 0.0)

        # a. Calculate overall level and progress
        overall_level = calculate_overall_level(overall_score)
        overall_progress = progress # already a float percentage

        # b. Find top/bottom criteria for Section 2
        sub_breakdown = summary_data.get("SubCriteria_Breakdown", {})
        criteria_analysis = find_top_bottom_criteria(sub_breakdown)
        top_c = criteria_analysis['top_criteria']
        bottom_c = criteria_analysis['bottom_criteria']
        
        # c. Extract top strategic goals for Section 4
        top_goals = extract_top_goals(summary_data, limit=3)
        
        # Format goals for better prompt injection (e.g., list format)
        formatted_goals = "\n- " + "\n- ".join(top_goals) if top_goals else "ไม่มีเป้าหมายเชิงกลยุทธ์ที่ชัดเจนจาก Action Plan."
        
        # Dumping JSON to pass to LLM (for context)
        data_json_string = json.dumps(summary_data, indent=2, ensure_ascii=False)
        # --- END NEW: Data Pre-processing ---
        
        # 1. สร้าง Prompt ฉบับสมบูรณ์ (Human Message)
        final_prompt_text = NARRATIVE_REPORT_PROMPT.format(
            summary_data=data_json_string,
            enabler_name=enabler_name,
            overall_level=overall_level,
            overall_progress=overall_progress,
            overall_maturity_score=overall_score,
            # เพิ่มรายละเอียดเชิงลึกของจุดแข็ง/จุดอ่อน
            top_criteria_id=top_c.get('id', 'N/A'),
            top_criteria_name=top_c.get('name', 'N/A'),
            top_criteria_score=top_c.get('score', 0.0),
            top_criteria_summary=top_c.get('action_item', 'N/A'), # ใช้ action_item เป็น summary/description
            top_criteria_level=top_c.get('highest_full_level', 0),
            
            bottom_criteria_id=bottom_c.get('id', 'N/A'),
            bottom_criteria_name=bottom_c.get('name', 'N/A'),
            bottom_criteria_score=bottom_c.get('score', 0.0),
            bottom_criteria_summary=bottom_c.get('action_item', 'N/A'), # ใช้ action_item เป็น summary/description
            bottom_criteria_level=bottom_c.get('highest_full_level', 0),
            
            top_goals=formatted_goals
        )
        
        # 2. CALL REAL LLM FUNCTION (sending System Instruction)
        real_llm_response_text = generate_narrative_report_via_llm_real(
            prompt_text=final_prompt_text,
            system_instruction=SYSTEM_NARRATIVE_PROMPT 
        )
        
        # 3. ส่งคืนข้อความที่มาจาก LLM จริง
        # Remove any unwanted triple backticks or leading/trailing whitespace
        final_report = real_llm_response_text.strip().replace("```markdown", "").replace("```", "")
        return final_report.strip()

    except Exception as e:
        logger.error(f"Error during LLM narrative generation: {e}", exc_info=True)
        return f"ERROR: เกิดข้อผิดพลาดในการสร้างรายงานเชิงบรรยาย: {e}"

# -------------------- STRUCTURED MARKDOWN FUNCTION --------------------
def generate_feedback_report(summary: Dict[str, Any]) -> str:
    """สร้างรายงานในรูปแบบตาราง Markdown สำหรับผู้ปฏิบัติงาน (ไม่ใช้ LLM)"""
    report_parts = []
    
    sub_breakdown = summary.get('SubCriteria_Breakdown', {})
    action_plans_raw = summary.get('Action_Plans', {})
    
    # Flatten Action Plans for easy lookup by Sub_ID
    action_plans = {}
    for sub_list in action_plans_raw.values():
        for plan in sub_list:
            # Try to extract the Sub ID from the first action's Statement_ID
            first_statement_id = plan.get('Actions', [{}])[0].get('Statement_ID', 'N/A')
            parts = first_statement_id.split(' ')
            sub_id = parts[0] if '.' in parts[0] else None 

            if sub_id:
                action_plans[sub_id] = plan
            elif plan.get('Actions') and plan['Actions'][0].get('Failed_Level'):
                for key, value in action_plans_raw.items():
                    if value == sub_list:
                        action_plans[key] = plan
                        break


    passed_subs = {}
    failed_subs = {}

    for sub_id, data in sub_breakdown.items():
        highest_full_level = data.get('highest_full_level', 0)
        
        if highest_full_level >= 1:
            passed_subs[sub_id] = data
        else:
            failed_subs[sub_id] = data

    report_parts.append("\n## 1. กลุ่มหัวข้อที่ผ่านเกณฑ์พื้นฐาน (Highest Full Level >= 1)")
    report_parts.append("หัวข้อเหล่านี้ได้สร้างรากฐานการดำเนินงานในระดับเริ่มต้นแล้ว")
    report_parts.append("| รหัส | ชื่อหัวข้อ | คะแนน (Score) | Level ที่ผ่านเต็ม | คำแนะนำสำหรับ Level ถัดไป |")
    report_parts.append("| :---: | :--- | :---: | :---: | :--- |")
    
    for sub_id, data in passed_subs.items():
        highest_full = data.get('highest_full_level', 0)
        score = data.get('score', 0.0)
        action_detail = "ดูแผนปฏิบัติการโดยละเอียดในส่วน Action Plans"
        
        # Lookup the relevant Action Plan goal
        if action_plans.get(sub_id):
            action_detail = action_plans[sub_id].get('Goal', action_detail)
            
        action_detail = action_detail.replace('MOCK:', '').strip()
        if len(action_detail) > 100:
            action_detail = action_detail[:97] + "..."
            
        report_parts.append(f"| {sub_id} | {data.get('name')} | {score:.2f} | L{highest_full} | {action_detail} |")

    report_parts.append("\n\n## 2. กลุ่มหัวข้อที่ต้องเร่งพัฒนา (Highest Full Level = 0)")
    report_parts.append("หัวข้อเหล่านี้ยังไม่ผ่านเกณฑ์ Level 1 ต้องเร่งสร้างหลักฐานและระบบพื้นฐาน")
    report_parts.append("| รหัส | ชื่อหัวข้อ | คะแนน (Score) | คำแนะนำเฉพาะสำหรับการบรรลุ Level 1 |")
    report_parts.append("| :---: | :--- | :---: | :--- |")
    
    for sub_id, data in failed_subs.items():
        score = data.get('score', 0.0)
        recommendation = data.get('action_item', 'ไม่พบคำแนะนำสรุป.')
        
        # Lookup the first specific action for Level 1
        if action_plans.get(sub_id) and action_plans[sub_id].get('Actions'):
            first_action = action_plans[sub_id]['Actions'][0]
            # Use the detailed recommendation from the Action Plan
            rec_text = first_action.get('Recommendation', recommendation)
            recommendation = f"สร้างหลักฐานสำหรับ Statement {first_action.get('Statement_ID')} เพื่อ: {rec_text}"
            
        recommendation = recommendation.replace('MOCK:', '').strip()
        if len(recommendation) > 120:
            recommendation = recommendation[:117] + "..."
            
        report_parts.append(f"| {sub_id} | {data.get('name')} | {score:.2f} | {recommendation} |")

    return "\n".join(report_parts)


# -------------------- CLI Entry Point --------------------

def run_report_from_json(file_path: str, use_narrative_llm: bool, output_file: str = None):
    """ฟังก์ชันหลักสำหรับรันรายงานโดยรับ path ของไฟล์ JSON ผลลัพธ์และบันทึกลงไฟล์ (ถ้ามี)"""
    try:
        if not os.path.exists(file_path):
             raise FileNotFoundError(f"ไม่พบไฟล์ที่ Path: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
            
        print(f"\n✅ โหลดข้อมูลผลการประเมินจากไฟล์: {file_path}")
        
        # พิมพ์สรุปผลรวมเบื้องต้น
        overall = summary_data.get('Overall', {})
        if overall:
            print("=====================================================")
            print("        OVERALL ASSESSMENT SUMMARY")
            print(f"Enabler: {overall.get('enabler', 'N/A')}")
            print(f"Overall Maturity Score (Avg.): {overall.get('overall_maturity_score', 0.0):.2f}")
            print(f"Total Score (Weighted): {overall.get('total_weighted_score', 0.0):.2f}/{overall.get('total_possible_weight', 0.0):.2f}")
            print(f"Progress: {overall.get('overall_progress_percent', 0.0):.2f}%")
            print("=====================================================")
        
        # เลือกโหมดการสร้างรายงาน
        if use_narrative_llm:
            logger.info("Generating narrative report using LLM...")
            # 🚨 LLM Call ถูกย้ายมาอยู่ที่นี่ 🚨
            final_report_text = generate_narrative_report_via_llm(summary_data) 
            report_type = "LLM NARRATIVE (CEO Format)"
        else:
            logger.info("Generating structured markdown report...")
            final_report_text = generate_feedback_report(summary_data)
            report_type = "STRUCTURED MARKDOWN (Operator Format)"
            
        # สร้างและพิมพ์ Feedback Report ลง console
        print(f"\n\n=====================================================")
        print(f"      FINAL FEEDBACK REPORT ({report_type})")
        print("=====================================================")
        print(final_report_text)
        print("\n=====================================================")

        # บันทึกรายงานลงไฟล์
        if output_file:
            # สร้าง directory ถ้ายังไม่มี
            os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as outfile:
                outfile.write(final_report_text)
            print(f"\n✅ SUCCESS: บันทึกรายงาน ({report_type}) ลงในไฟล์: {output_file}")


    except FileNotFoundError as e:
        print(f"❌ ERROR: {e}", file=sys.stderr)
    except json.JSONDecodeError:
        print(f"❌ ERROR: ไฟล์ {file_path} ไม่ใช่รูปแบบ JSON ที่ถูกต้อง", file=sys.stderr)
    except Exception as e:
        print(f"❌ ERROR: เกิดข้อผิดพลาดในการสร้างรายงาน: {e}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a Feedback Report from a saved Assessment JSON file.")
    parser.add_argument("--file", 
                        type=str, 
                        required=True,
                        help="Path to the assessment result JSON file.")
    
    parser.add_argument("--narrative", 
                        action="store_true",
                        help="Generate a human-readable, narrative summary using LLM for CEO.")
    
    parser.add_argument("--output", 
                        type=str,
                        default=None,
                        help="Optional path to save the generated report to a file (e.g., reports/final_summary.txt).")
    
    args = parser.parse_args()
    run_report_from_json(args.file, args.narrative, args.output)
