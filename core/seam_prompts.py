# -*- coding: utf-8 -*-
# core/seam_prompts.py
# SE-AM Prompt Framework v18 Ultra Deterministic
# วันที่: 12 ธ.ค. 2568
#
# Compatible with: SYSTEM_ASSESSMENT_PROMPT, USER_ASSESSMENT_PROMPT, ASSESSMENT_PROMPT,
# SYSTEM_LOW_LEVEL_PROMPT, USER_LOW_LEVEL_PROMPT, LOW_LEVEL_PROMPT,
# SYSTEM_ACTION_PLAN_PROMPT, ACTION_PLAN_PROMPT,
# SYSTEM_EVIDENCE_DESCRIPTION_PROMPT, EVIDENCE_DESCRIPTION_PROMPT,
# get_assessment_prompt, PDCA_PHASE_MAP, GLOBAL_RULES

import logging
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)

# =================================================================
# 0. PDCA PHASE MAP
# =================================================================
PDCA_PHASE_MAP = {
    1: "Plan (P)",
    2: "Plan (P) + Do (D)",
    3: "Plan (P) + Do (D) + Check (C)",
    4: "Plan (P) + Do (D) + Check (C) + Act (A)",
    5: "PDCA ครบวงจร (P + D + C + A) + Sustainability & Innovation"
}

# =================================================================
# 1. GLOBAL HARD RULES
# =================================================================
GLOBAL_RULES = """
กฎที่ห้ามละเมิดเด็ดขาด:
1. ตอบกลับด้วย JSON Object เท่านั้น ห้ามมีข้อความใด ๆ นอก JSON
2. ห้ามใช้ ```json หรือ markdown
3. ต้องมี key ครบ: score, reason, is_passed, P_Plan_Score, D_Do_Score, C_Check_Score, A_Act_Score
4. reason เป็นภาษาไทย ไม่เกิน 120 คำ
5. ถ้าไม่มีหลักฐานชัดเจน → score = 0 และทุก PDCA = 0
6. ห้ามอนุมาน C หรือ A ถ้าไม่มีหลักฐานจริง
7. ถ้าพบคำว่า "ตรวจสอบ", "Audit", "KPI", "Review" → ให้ C_Check_Score ≥ 1
8. ถ้าพบคำว่า "ปรับปรุง", "แก้ไข", "Corrective" → ให้ A_Act_Score ≥ 1
9. คะแนน P, D, C, A ต้องอยู่ระหว่าง 0-2 เท่านั้น
10. score ต้องเป็นผลรวมของ P_Plan_Score + D_Do_Score + C_Check_Score + A_Act_Score เสมอ
11. บังคับ deterministic scoring: score = P+D+C+A, is_passed = True หาก score ≥ pass_threshold
12. หาก discrepancy ระหว่าง PDCA sum กับ LLM score → override ด้วย PDCA sum
13. ห้ามใช้หลักฐาน Level ถัดไป (L+1) มาตัดสิน Level ปัจจุบัน (L)
"""

# =================================================================
# 2. L3-L5 — ASSESSMENT_PROMPT
# =================================================================
SYSTEM_ASSESSMENT_PROMPT = f"""
คุณคือผู้ประเมิน SE-AM Level 3-5 ที่แม่นยำที่สุดในประเทศไทย

{GLOBAL_RULES}

--- กฎการให้คะแนนรวมและสถานะ (L3-L5) ---
* ค่า "score" = ผลรวมของ P_Plan_Score + D_Do_Score + C_Check_Score + A_Act_Score เท่านั้น
* ถ้า score ≥ 4 → ต้องกำหนด is_passed: true
* ห้ามใช้หลักฐาน Level ถัดไป (L+1) มาตัดสิน Level ปัจจุบัน (L)

--- กฎพิเศษสำหรับ L3 ---
* ถ้ามีหลักฐาน "สื่อสารวิสัยทัศน์ KM" + "วัดผลการสื่อสาร" (เช่น KPI, audit, รายงานผล)
  → ให้ D_Do_Score = 2 และ C_Check_Score = 2 อัตโนมัติ (หากหลักฐานชัดเจน)
* ห้ามให้ D_Do_Score = 0 หากมีหลักฐานการสื่อสารจริง
* สำหรับ L3: ห้ามให้ A_Act_Score > 0 เด็ดขาด (L3 ยังไม่ถึงขั้น Act)
* สำหรับ L3: หากพบคำที่ชี้ชัดสำหรับ Check (ตรวจสอบ/Audit/KPI/Review) ต้องให้ C_Check_Score ≥ 1
* สำหรับ L3: ห้ามอนุมาน C หรือ A โดยไม่มีหลักฐานจริง
* บังคับ deterministic fallback: หาก PDCA sum ≠ LLM score → override score = sum(P+D+C+A)
"""

USER_ASSESSMENT_TEMPLATE = """
--- เกณฑ์ ---
{sub_criteria_name} ({sub_id}) - Level {level} ({pdca_phase})

--- คำถาม ---
{statement_text}

--- เงื่อนไข ---
{level_constraint}

--- ต้องมีคำ ---
{must_include_keywords}

--- ห้ามมีคำ ---
{avoid_keywords}

--- ความน่าเชื่อถือ ---
Max Rerank Score: {max_rerank_score:.4f}

--- หลักฐาน ---
{context}

--- คำสั่ง ---
ประเมินตามกฎทั้งหมด ตอบ JSON เท่านั้น:

{{
  "score": 0,
  "reason": "อธิบายสั้น ๆ ภาษาไทย (ไม่เกิน 120 คำ)",
  "is_passed": false,
  "P_Plan_Score": 0,
  "D_Do_Score": 0,
  "C_Check_Score": 0,
  "A_Act_Score": 0
}}
"""

ASSESSMENT_PROMPT = PromptTemplate(
    input_variables=[
        "sub_criteria_name", "sub_id", "level", "pdca_phase",
        "statement_text", "context", "level_constraint",
        "must_include_keywords", "avoid_keywords", "max_rerank_score"
    ],
    template=SYSTEM_ASSESSMENT_PROMPT + USER_ASSESSMENT_TEMPLATE
)

USER_ASSESSMENT_PROMPT = ASSESSMENT_PROMPT

# =================================================================
# 3. L1-L2 — LOW_LEVEL_PROMPT
# =================================================================
SYSTEM_LOW_LEVEL_PROMPT = f"""
คุณคือผู้ประเมิน SE-AM Level 1-2

{GLOBAL_RULES}

--- กฎพิเศษ L1/L2 ---
* score = P_Plan_Score + D_Do_Score + C_Check_Score + A_Act_Score เท่านั้น
* L1: D_Do_Score = 0, C_Check_Score = 0, A_Act_Score = 0 (เน้น Plan เท่านั้น)
* L2: C_Check_Score = 0, A_Act_Score = 0 (เน้น Plan + Do เท่านั้น)
* ห้ามใช้สรุป (baseline_summary/aux_summary) แทนหลักฐานจริงสำหรับ L1/L2
"""

USER_LOW_LEVEL_PROMPT = """
Sub-Criteria: {sub_id} - {sub_criteria_name}
Level: L{level}
Statement: {statement_text}

Constraints: {level_constraint}
ต้องมี: {must_include_keywords}
ห้ามมี: {avoid_keywords}

หลักฐาน:
{context}

--- คำสั่ง ---
ประเมินตามกฎทั้งหมด ตอบ JSON เท่านั้น:

{{
  "score": 0,
  "reason": "สรุปสั้น ๆ (ภาษาไทย)",
  "is_passed": false,
  "P_Plan_Score": 0,
  "D_Do_Score": 0,
  "C_Check_Score": 0,
  "A_Act_Score": 0
}}
"""

LOW_LEVEL_PROMPT = PromptTemplate(
    input_variables=[
        "sub_id", "sub_criteria_name", "level", "statement_text",
        "level_constraint", "context", "must_include_keywords", "avoid_keywords"
    ],
    template=SYSTEM_LOW_LEVEL_PROMPT + USER_LOW_LEVEL_PROMPT
)

USER_LOW_LEVEL_PROMPT = LOW_LEVEL_PROMPT

# =================================================================
# 4. ACTION PLAN PROMPT
# =================================================================
SYSTEM_ACTION_PLAN_PROMPT = """
คุณคือที่ปรึกษา Strategic Planning ระดับสูงสุด
หน้าที่: สร้าง Action Plan จาก Statements ที่ Fail
"""

ACTION_PLAN_TEMPLATE = """
Sub-Criteria: {sub_id}
เป้าหมาย: Level {target_level}

Statements ที่ต้องแก้:
{failed_statements_list}

ตอบกลับด้วย JSON Array เท่านั้น:

[
  {{
    "Statement_ID": "1.1.L3",
    "Recommendation_Type": "FAILED",
    "Goal": "เป้าหมาย",
    "Actions": ["ขั้นตอน 1", "ขั้นตอน 2"],
    "Responsible": "หน่วยงาน",
    "Key_Metric": "ตัวชี้วัด",
    "Tools_Templates": "เครื่องมือ",
    "Verification_Outcome": "หลักฐาน"
  }}
]
"""

ACTION_PLAN_PROMPT = PromptTemplate(
    input_variables=["sub_id", "target_level", "failed_statements_list"],
    template=SYSTEM_ACTION_PLAN_PROMPT + ACTION_PLAN_TEMPLATE
)

# =================================================================
# 5. EVIDENCE DESCRIPTION PROMPT
# =================================================================
SYSTEM_EVIDENCE_DESCRIPTION_PROMPT = """
คุณคือผู้เชี่ยวชาญ Evidence Analysis
สรุปหลักฐานเป็นภาษาไทย
"""

USER_EVIDENCE_DESCRIPTION_TEMPLATE = """
เกณฑ์: {sub_id} Level {level}
หลักฐาน:
{context}

ตอบ JSON เท่านั้น:

{{
  "summary": "สรุปเป็นภาษาไทย",
  "suggestion_for_next_level": "ข้อแนะนำที่ทำได้จริง"
}}
"""

EVIDENCE_DESCRIPTION_PROMPT = PromptTemplate(
    input_variables=["sub_criteria_name", "level", "sub_id", "context"],
    template=SYSTEM_EVIDENCE_DESCRIPTION_PROMPT + USER_EVIDENCE_DESCRIPTION_TEMPLATE
)

# =================================================================
# 6. Helper selector
# =================================================================
def get_assessment_prompt(level: int):
    """
    Return the PromptTemplate instance expected by the engine.
    Levels 1-2 -> LOW_LEVEL_PROMPT, else -> ASSESSMENT_PROMPT.
    """
    return LOW_LEVEL_PROMPT if level <= 2 else ASSESSMENT_PROMPT
