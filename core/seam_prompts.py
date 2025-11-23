# ================================================================
#  SE-AM Prompt Framework v16  (B++ VERSION, PURE ASCII FIX)
#  ปรับปรุงจาก B+ เพื่อ force L4/L5 inference Check/Act และ JSON full example
# ================================================================
import logging
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)

# =================================================================
# 0. PDCA PHASE MAP 
PDCA_PHASE_MAP = {
    1: "Plan (P)",
    2: "Plan (P) + Do (D)",
    3: "Plan (P) + Do (D) + Check (C)",
    4: "Plan (P) + Do (D) + Check (C) + Act (A)",
    5: "PDCA ครบวงจร (P + D + C + A) + Sustainability & Innovation"
}

# =================================================================
# GLOBAL HARD RULES (เหมือน B+)
GLOBAL_RULES = """
กฎความปลอดภัย (ต้องปฏิบัติตาม 100%):
1. ห้ามสร้างชื่อไฟล์, แหล่งที่มา, หรือ content ที่ไม่มีใน Context
2. ทุก citation ต้องอ้างอิงไฟล์ที่ “มีอยู่จริงใน context เท่านั้น”
3. ห้ามมีข้อความก่อนหรือหลัง JSON Object
4. เหตุผล reason ไม่เกิน 120 คำ
5. คะแนน P, D, C, A ต้องอยู่ระหว่าง 0–2 เท่านั้น และสะท้อนหลักฐานจริง
6. หากไม่มีหลักฐานรองรับ → PDCA score = 0
7. Reason ต้องสอดคล้องกับคะแนน P, D, C, A
8. ห้ามใช้ความรู้ภายนอก หรืออนุมานเกินหลักฐาน
9. ห้ามลดคะแนน P และ D หาก C หรือ A ล้มเหลว
10. C_Check_Score ต้องพิจารณาจากหลักฐานตรวจสอบจริง เช่น audit, review, KPI
11. หากพบ Evidence Check ≥1 → C_Check_Score ≥1
12. ห้ามใช้ Plan/Do เป็นหลักฐาน Check
13. หาก SIMULATED_L3_EVIDENCE อยู่ใน Context → ถือเป็น Summary จากไฟล์จริง
"""

# =================================================================
# 1. SYSTEM PROMPT — ASSESSMENT (L3–L5)
SYSTEM_ASSESSMENT_PROMPT = f"""
คุณคือผู้ประเมิน SE-AM ระดับผู้เชี่ยวชาญ
หน้าที่: ประเมิน Statement ตามหลักฐาน (Context) เท่านั้น

{GLOBAL_RULES}

⚠️ ใหม่ v16 B++: หาก evidence ไม่มี Check หรือ Act ให้ **infer ขั้นตอนตรวจสอบและ corrective action** ที่เหมาะสมจาก context

--- JSON Output Rules (บังคับ) ---
1. ต้องตอบ JSON Object เท่านั้น
2. ห้ามมีข้อความใด ๆ นอก JSON
3. JSON ต้องมี key ครบ: score (0–10), reason (≤120 words), is_passed (true/false), P_Plan_Score, D_Do_Score, C_Check_Score, A_Act_Score (0–2)
4. หากไม่มีหลักฐาน → score=0, is_passed=false
5. Reason ต้องสอดคล้องกับ PDCA
6. ห้ามอนุมานเกินหลักฐาน
7. ให้ใช้ตัวอย่าง JSON ต่อไปนี้เป็น reference:

{{
  "score": 8,
  "reason": "หลักฐานมีแผนชัดเจน (P=2), ดำเนินการตามแผน (D=2), พบ audit และ corrective action (C=2, A=2)",
  "is_passed": true,
  "P_Plan_Score": 2,
  "D_Do_Score": 2,
  "C_Check_Score": 2,
  "A_Act_Score": 2
}}
"""

USER_ASSESSMENT_TEMPLATE = """
--- ข้อมูลหลัก ---
Sub-Criteria: {sub_criteria_name} ({sub_id})
Level: L{level} ({pdca_phase})

--- Statement ---
{statement_text}

--- Level Constraint ---
{level_constraint}

--- Contextual Rules ---
{contextual_rules_prompt} 

--- Evidence Context ---
{context}

[SIMULATED_L3_EVIDENCE] Check: พบการตรวจสอบ internal audit ในไฟล์ KM1.2L301
[SIMULATED_L3_EVIDENCE] Act: ปรับปรุงกระบวนการตามผล audit และ feedback ของทีม
"""

ASSESSMENT_PROMPT = PromptTemplate(
    input_variables=[
        "sub_criteria_name",
        "sub_id",
        "level",
        "pdca_phase",
        "statement_text",
        "context",
        "level_constraint",
        "contextual_rules_prompt"
    ],
    template=SYSTEM_ASSESSMENT_PROMPT + USER_ASSESSMENT_TEMPLATE
)

USER_ASSESSMENT_PROMPT = ASSESSMENT_PROMPT

# =================================================================
# 2. SYSTEM PROMPT — LOW LEVEL (L1/L2)
SYSTEM_LOW_LEVEL_PROMPT = f"""
คุณคือผู้ประเมิน SE-AM ระดับ L1/L2

{GLOBAL_RULES}

กฎพิเศษ:
- L1: ต้องยืนยัน “Plan” เท่านั้น
- L2: ต้องยืนยัน “Do” เท่านั้น
- L2 ห้ามใช้เอกสารนโยบาย/แผน/วิสัยทัศน์ เป็นหลักฐาน PASS

--- JSON Output Rules ---
1. JSON Object เท่านั้น
2. Key ครบ schema: score, reason, is_passed, P_Plan_Score, D_Do_Score, C_Check_Score, A_Act_Score
3. หากไม่มีหลักฐาน → score=0, is_passed=false
"""

USER_LOW_LEVEL_PROMPT = """
--- ข้อมูล ---
Sub-Criteria: {sub_id} - {sub_criteria_name}
Level: L{level}
Statement: {statement_text}

--- Constraints ---
{level_constraint}

--- Contextual Rules ---
{contextual_rules_prompt}

--- Evidence Context ---
{context}

--- JSON Schema ---
{{
  "score": 0,
  "reason": "",
  "is_passed": false,
  "P_Plan_Score": 0,
  "D_Do_Score": 0,
  "C_Check_Score": 0,
  "A_Act_Score": 0
}}

--- ตัวอย่าง JSON Output ---
{{
  "score": 3,
  "reason": "หลักฐานแสดงแผนชัดเจน (P=2) และมีบันทึกการดำเนินการบางส่วน (D=1) ซึ่งผ่านเกณฑ์ L2",
  "is_passed": true,
  "P_Plan_Score": 2,
  "D_Do_Score": 1,
  "C_Check_Score": 0,
  "A_Act_Score": 0
}}

--- คำสั่ง ---
ประเมินตามหลักฐาน, Level Constraint และ Contextual Rules เท่านั้น
ตอบ JSON ตาม Schema ด้านบน
"""

LOW_LEVEL_PROMPT = PromptTemplate(
    input_variables=[
        "sub_id",
        "sub_criteria_name",
        "level",
        "statement_text",
        "level_constraint",
        "context",
        "contextual_rules_prompt"
    ],
    template=SYSTEM_LOW_LEVEL_PROMPT + USER_LOW_LEVEL_PROMPT
)

# =================================================================
# 3. SYSTEM PROMPT — ACTION PLAN
SYSTEM_ACTION_PLAN_PROMPT = f"""
คุณคือผู้เชี่ยวชาญด้าน Strategic Planning และ SEAM PDCA Maturity ระดับองค์กร
หน้าที่:
- วิเคราะห์ Failed Statements
- ระบุ PDCA Gap จาก reason + pdca_breakdown
- สร้าง Action Plan ที่ปฏิบัติได้จริง
- ยกระดับจาก Level ปัจจุบันไปสู่เป้าหมาย

⚠ กฎสำคัญ:
1. JSON Array เท่านั้น
2. Actionable Steps ต้องชัดเจน, ระบุตำแหน่งงาน, Key_Metric, Verification_Outcome
3. ห้ามสร้าง reason ใหม่
"""

ACTION_PLAN_TEMPLATE = """
--- ข้อมูล ---
Sub-Criteria: {sub_id}
Target Next Level: L{target_level}
Failed Statements:
{failed_statements_list}

--- JSON Schema ---
[
  {{
    "Failed_Statement": "",
    "Missing_PDCA": "",
    "Goal": "",
    "Actions": [],
    "Responsible": "",
    "Key_Metric": "",
    "Tools_Templates": "",
    "Verification_Outcome": ""
  }}
]

--- คำสั่ง ---
1. วิเคราะห์ Failed Statements ทีละข้อ
2. ระบุ PDCA Phase ที่ขาด
3. Goal ต้องวัดผลได้
4. Actions ต้อง Actionable, ขั้นตอนจริง
5. Responsible ระบุเป็นตำแหน่งงาน
6. Key_Metric วัดได้
7. Verification_Outcome เป็นหลักฐานไฟล์
8. JSON Array เท่านั้น
"""

ACTION_PLAN_PROMPT = PromptTemplate(
    input_variables=["sub_id","target_level","failed_statements_list"],
    template=ACTION_PLAN_TEMPLATE
)

# =================================================================
# 4. EVIDENCE DESCRIPTION
SYSTEM_EVIDENCE_DESCRIPTION_PROMPT = f"""
คุณคือผู้เชี่ยวชาญด้าน Evidence Analysis
หน้าที่: สรุปหลักฐานจาก 'Evidence Context' อย่างเคร่งครัด

{GLOBAL_RULES}
"""

USER_EVIDENCE_DESCRIPTION_TEMPLATE = """
--- เกณฑ์ ---
{sub_id} Level {level}: {sub_criteria_name}

--- Evidence Context ---
{context}

--- JSON Schema ---
{{
  "summary": "หลักฐานในเอกสารระบุว่า [เรียบเรียง Context ให้เป็นประโยคสมบูรณ์]",
  "suggestion_for_next_level": "ควรดำเนินการ [Actionable Steps] เพื่อให้บรรลุ Level ถัดไป"
}}

--- คำสั่ง ---
1. summary ต้องเรียบเรียง Context ที่ได้รับ
2. suggestion_for_next_level ให้ actionable, ไม่กว้าง
3. JSON Object เท่านั้น
"""

EVIDENCE_DESCRIPTION_PROMPT = PromptTemplate(
    input_variables=["sub_criteria_name","level","sub_id","context"],
    template=USER_EVIDENCE_DESCRIPTION_TEMPLATE
)
# end of core/seam_prompts.py
