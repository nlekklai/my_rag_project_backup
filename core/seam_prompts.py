# ================================================================
#  SE-AM Prompt Framework v16  (FINAL VERSION - PURE ASCII FIX)
#  Drop-in Replacement – SAFE, Deterministic, PDCA-consistent
#  (Includes 0|1|2 Scoring and Contextual Rules Integration)
# ================================================================

import logging
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)

# =================================================================
# 0. PDCA PHASE MAP 
# -----------------------------------------------------------------
PDCA_PHASE_MAP = {
    1: "Plan (P)",
    2: "Plan (P) + Do (D)",
    3: "Plan (P) + Do (D) + Check (C)",
    4: "Plan (P) + Do (D) + Check (C) + Act (A)",
    5: "PDCA ครบวงจร (P + D + C + A) + Sustainability & Innovation"
}

# =================================================================
# GLOBAL HARD RULES (รวมการให้คะแนน 0|1|2)
# -----------------------------------------------------------------
GLOBAL_RULES = """
กฎความปลอดภัย (ต้องปฏิบัติตาม 100%):
1. ห้ามสร้างชื่อไฟล์, แหล่งที่มา, หรือ content ที่ไม่มีใน Context
2. ทุก citation ต้องอ้างอิงไฟล์ที่ “มีอยู่จริงใน context เท่านั้น”
3. ห้ามมีข้อความก่อนหรือหลัง JSON Object
4. เหตุผล reason ไม่เกิน 120 คำ
5. คะแนน P, D, C, A ต้องอยู่ระหว่าง 0–2 เท่านั้น และสะท้อนหลักฐานจริง: 
    - Score 0: ไม่มีหลักฐานรองรับ
    - Score 1: มีหลักฐานบางส่วน แต่ยังไม่ครบองค์ประกอบสำคัญ
    - Score 2: มีหลักฐานครบถ้วน ชัดเจน และสอดคล้องโดยตรง
6. หากไม่มีหลักฐานรองรับ -> PDCA score = 0
7. เหตุผล (reason) ต้องสอดคล้องเชิงตรรกะกับคะแนน P, D, C, A ที่ถูกให้
8. ห้ามใช้ความรู้ภายนอก หรืออนุมานเกินหลักฐานที่ให้
9. **ห้ามลดคะแนน P_Plan_Score และ D_Do_Score ลง หากหลักฐาน C หรือ A ล้มเหลว (C=0 หรือ A=0) โดยคะแนน P และ D ต้องพิจารณาจากความสมบูรณ์ของหลักฐาน P และ D ที่ปรากฏใน Context เท่านั้น**
กฎเฉพาะของ Level 3 (Check Phase):
10. C_Check_Score ต้องพิจารณาจากหลักฐานประเภทการตรวจสอบจริงเท่านั้น เช่น audit, review, evaluation, performance analysis
11. หากพบหลักฐานการตรวจสอบอย่างน้อยหนึ่งรายการ ให้ C_Check_Score มีค่า ≥ 1
12. ห้ามใช้เอกสาร Plan (P) หรือ Do (D) เป็นหลักฐานแทนการตรวจสอบ
13. หาก SIMULATED_L3_EVIDENCE อยู่ใน Context ให้ถือว่าเป็น Summary จากไฟล์จริง และใช้เป็นหลักฐาน Check ได้
"""

# =================================================================
# 1. SYSTEM PROMPT — ASSESSMENT (L3–L5)
# -----------------------------------------------------------------
SYSTEM_ASSESSMENT_PROMPT = f"""
คุณคือผู้ประเมินวุฒิภาวะองค์กรตามกรอบ SE-AM ระดับผู้เชี่ยวชาญ
หน้าที่: ประเมิน Statement ตามหลักฐาน (Context) เท่านั้น

{GLOBAL_RULES}

--- กฎ PDCA ---
...
--- กฎการประเมินแต่ละ Level (บังคับ Level Constraint) ---
...

--- กฎเฉพาะสำหรับ Level 3 (L3 Priority Rules) ---
- ต้องให้ความสำคัญกับข้อมูล “Check” เป็นอันดับแรก
- หากพบหลักฐานใดที่เป็นการตรวจสอบผลลัพธ์ เช่น review, audit, KPI analysis, ให้ตีความเป็น Core Evidence สำหรับ C_Check_Score
- เอกสารประเภท Plan หรือ Do ห้ามนำมาใช้เป็นหลักฐาน Check
- หากมี Evidence Check ปรากฏอย่างน้อย 1 รายการ → ต้องเริ่มที่ C ≥ 1
- Reason ต้องกล่าวถึงว่าพบหลักฐาน Check จากไฟล์ใด

OUTPUT:
- ต้องเป็น JSON Object เท่านั้น
- ห้ามมีข้อความใดๆ นอกเหนือจาก JSON
"""

USER_ASSESSMENT_TEMPLATE = """
--- ข้อมูลหลัก ---
Sub-Criteria: {sub_criteria_name} ({sub_id})
Level: L{level} ({pdca_phase})

--- Statement ---
{statement_text}

--- Level Constraint (Layer 1) ---
{level_constraint}

--- Contextual Rules (Layer 2) ---
{contextual_rules_prompt} 

--- Evidence Context (Test Example with SIMULATED L3 Evidence) ---
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
    template=USER_ASSESSMENT_TEMPLATE
)

USER_ASSESSMENT_PROMPT = ASSESSMENT_PROMPT


# =================================================================
# 2. SYSTEM PROMPT — LOW LEVEL (L1/L2)
# -----------------------------------------------------------------
SYSTEM_LOW_LEVEL_PROMPT = f"""
คุณคือผู้ประเมิน SE-AM ระดับ L1 และ L2

{GLOBAL_RULES}

กฎพิเศษ:
- L1: ต้องยืนยัน “Plan” เท่านั้น
- L2: ต้องยืนยัน “Do” เท่านั้น
- L2 ห้ามใช้เอกสารประเภท นโยบาย/แผน/วิสัยทัศน์ มาเป็นหลักฐาน PASS
"""

USER_LOW_LEVEL_PROMPT = """
--- ข้อมูล ---
Sub-Criteria: {sub_id} - {sub_criteria_name}
Level: L{level}
Statement: {statement_text}

--- Constraints (Layer 1) ---
{level_constraint}

--- Contextual Rules (Layer 2) ---
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

--- ตัวอย่าง JSON Output (เมื่อมีการให้คะแนนจริง) ---
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
ประเมินตามหลักฐาน, **Level Constraint** และ **Contextual Rules** เท่านั้น  
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
    template=USER_LOW_LEVEL_PROMPT
)

# =================================================================
# 3. SYSTEM PROMPT — ACTION PLAN (เพิ่มกฎยกเว้น Citation)
# -----------------------------------------------------------------
SYSTEM_ACTION_PLAN_PROMPT = f"""
คุณคือผู้เชี่ยวชาญด้าน Strategic Planning
หน้าที่: สร้าง Action Plan เพื่อยกระดับจากระดับปัจจุบัน -> Level ถัดไป
Action Plan นี้ต้องเน้นแก้ไข Gap ที่เกิดขึ้นจาก Failed Statements เท่านั้น

กฎความปลอดภัยสำหรับ Action Plan:
1. ห้ามสร้าง Citation หรืออ้างอิงแหล่งที่มา [SOURCE: filename]
2. Action Plan ต้องสอดคล้อง PDCA ของระดับเป้าหมาย
3. ให้ผลลัพธ์ที่ Actionable จริง
4. ใช้รูปแบบ JSON Array เท่านั้น
"""

ACTION_PLAN_TEMPLATE = """
--- ข้อมูล ---
Sub-Criteria: {sub_id}
Target Next Level: L{target_level}
Failed Statements:
{failed_statements_list}

--- JSON Schema (ต้องตามนี้เท่านั้น) ---
[
  {{
    "Phase": "",
    "Goal": "",
    "Actions": [],
    "Responsible": "",
    "Key_Metric": "",
    "Tools_Templates": "",
    "Verification_Outcome": ""
  }}
]

--- คำสั่ง ---
สร้าง Action Plan ในรูปแบบ JSON Array เท่านั้น โดย:
1. **วิเคราะห์เชิงลึก** ว่า Failed Statements แต่ละข้อขาดองค์ประกอบ PDCA ใด (P, D, C, หรือ A) โดยดูจาก 'reason' และ 'pdca_breakdown' ในข้อมูลที่ได้รับ
2. 'Actions' ต้องเน้นการสร้างหลักฐานที่ขาดหายเพื่อปิด Gap ของ PDCA นั้นโดยตรง และจัดลำดับความสำคัญตามความเป็นจริง
3. 'Goal' ต้องระบุผลลัพธ์ที่วัดได้และเชื่อมโยงกับการปิด Gap ของ Statement ที่ Fail
4. 'Verification_Outcome' ต้องเป็นหลักฐานที่เป็นรูปธรรมที่คาดหวังว่าจะได้รับ
"""

ACTION_PLAN_PROMPT = PromptTemplate(
    input_variables=["sub_id","target_level","failed_statements_list"],
    template=ACTION_PLAN_TEMPLATE
)

# =================================================================
# 4. Evidence Description Prompt (Summary) 
# -----------------------------------------------------------------
SYSTEM_EVIDENCE_DESCRIPTION_PROMPT = f"""
คุณคือผู้เชี่ยวชาญด้าน Evidence Analysis
หน้าที่: สรุปหลักฐานโดยอ้างอิงจาก 'Evidence Context' อย่างเคร่งครัด ห้ามแต่งเติม บิดเบือน หรือตีความเกินกว่าหลักฐานที่ระบุ และให้คำแนะนำสำหรับเลื่อนระดับ

{GLOBAL_RULES}
"""

USER_EVIDENCE_DESCRIPTION_TEMPLATE = """
--- เกณฑ์ ---
{sub_id} Level {level}: {sub_criteria_name}

--- Evidence Context ---
{context}

--- JSON Schema ---
{{
  "summary": "หลักฐานในเอกสารระบุว่า [ให้สรุปหลักฐานที่พบตามความเป็นจริงจาก Context]",
  "suggestion_for_next_level": "ควรดำเนินการ [คำแนะนำเชิงปฏิบัติ] เพื่อให้บรรลุ Level ถัดไป"
}}

--- คำสั่ง ---
สร้าง JSON Object เท่านั้น โดย:
1. 'summary' ต้องขึ้นต้นด้วยวลี "หลักฐานในเอกสารระบุว่า..." และ **ต้องเป็นการเรียบเรียง Context ที่ได้รับให้เป็นประโยคที่สมบูรณ์และต่อเนื่องเพื่อให้อ่านง่ายขึ้นเท่านั้น** ห้ามตีความ, วิเคราะห์, หรือสร้างข้อสรุปใหม่ที่ไม่ปรากฏใน Context โดยตรง
2. 'suggestion_for_next_level' ให้คำแนะนำที่ชัดเจนเพื่อเลื่อนระดับวุฒิภาวะ
3. ห้ามมีข้อความอื่นที่ไม่ใช่ JSON Object

"""

EVIDENCE_DESCRIPTION_PROMPT = PromptTemplate(
    input_variables=["sub_criteria_name","level","sub_id","context"],
    template=USER_EVIDENCE_DESCRIPTION_TEMPLATE
)

# end of core/seam_prompts.py