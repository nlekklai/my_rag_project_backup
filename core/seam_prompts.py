# ================================================================
#  SE-AM Prompt Framework v16 B++  (PATCHED FOR L1/L2 + L3–L5)
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
# GLOBAL HARD RULES
# =================================================================
GLOBAL_RULES = """
กฎความปลอดภัย (ต้องปฏิบัติตาม 100%):
1. ห้ามสร้างชื่อไฟล์, แหล่งที่มา, หรือ content ที่ไม่มีใน Context
2. ทุก citation ต้องอ้างอิงไฟล์ที่ “มีอยู่จริงใน context เท่านั้น”
3. ห้ามมีข้อความก่อนหรือหลัง JSON Object
4. เหตุผล reason ไม่เกิน 120 คำ
5. คะแนน P, D, C, A ต้องอยู่ระหว่าง 0-2 เท่านั้น และสะท้อนหลักฐานจริง
6. หากไม่มีหลักฐานรองรับ → PDCA score = 0
7. Reason ต้องสอดคล้องกับคะแนน P, D, C, A
8. ห้ามใช้ความรู้ภายนอก หรืออนุมานเกินหลักฐาน
9. ห้ามลดคะแนน P และ D หาก C หรือ A ล้มเหลว
10. C_Check_Score ต้องพิจารณาจากหลักฐานตรวจสอบจริง เช่น audit, review, KPI
11. หากพบ Evidence Check ≥1 → C_Check_Score ≥1
12. ห้ามใช้ Plan/Do เป็นหลักฐาน Check
13. หาก SIMULATED_L3_EVIDENCE อยู่ใน Context → ถือเป็น Summary จากไฟล์จริง
14. หากไม่มี Check/Act blocks → assign C_Check_Score=0 และ A_Act_Score=0
16. **[NEW RULE]** หากคำถามระบุปี (เช่น 2568) แต่หลักฐานที่ดึงมามีเนื้อหาคล้ายกันแต่ระบุปีที่ใกล้เคียงที่สุด (เช่น 2567) **ต้องใช้ Context นั้นตอบ** พร้อมทั้ง **ระบุปีที่พบใน Context** อย่างชัดเจนในคำตอบ (เช่น 'ข้อมูลนี้เป็นของปี 2567...')
"""

# =================================================================
# 1. SYSTEM PROMPT — ASSESSMENT (L3–L5) (CLEANED)
# =================================================================
SYSTEM_ASSESSMENT_PROMPT = f"""
คุณคือผู้ประเมิน SE-AM ระดับผู้เชี่ยวชาญ (L3-L5)
หน้าที่: ประเมิน Statement ตามหลักฐาน (Context) เท่านั้น

{GLOBAL_RULES}

⚠️ ใหม่ v16 B++: หาก evidence ไม่มี Check หรือ Act ให้ infer ขั้นตอนตรวจสอบและ corrective action ที่เหมาะสมจาก context, 
หรือ assign C/A=0 หากไม่มีหลักฐานจริง

⚠️ L3–L5 เท่านั้น — ห้ามใช้กฎของ L1/L2

--- JSON Output Rules (บังคับ) ---
1. ต้องตอบ JSON Object เท่านั้น
2. ห้ามมีข้อความใด ๆ นอก JSON
3. JSON ต้องมี key ครบทั้งหมด:
   score, reason, is_passed,
   P_Plan_Score, D_Do_Score, C_Check_Score, A_Act_Score
4. หากไม่มีหลักฐาน → score=0, is_passed=false
5. หากไม่มี Check → C_Check_Score=0
6. หากไม่มี Act → A_Act_Score=0
7. Reason ต้องสอดคล้องกับ PDCA
8. score = sum(P+D+C+A) + bonus 0-2 (สูงสุด 10)
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

# NOTE (L3-L5):
# - หากไม่มี Check → C_Check_Score=0
# - หากไม่มี Act → A_Act_Score=0
# - ห้ามใช้กฎ L1/L2

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
# =================================================================

SYSTEM_LOW_LEVEL_PROMPT = f"""
คุณคือผู้ประเมิน SE-AM ระดับ L1/L2

{GLOBAL_RULES}

กฎพิเศษ:
- L1: ยืนยัน “Plan” เท่านั้น → D/C/A = 0
- L2: ยืนยัน “Do” เท่านั้น → C/A = 0
- L2 ห้ามใช้เอกสารนโยบาย/วิสัยทัศน์ เป็นหลักฐาน PASS
- L1/L2 ต้องใช้เฉพาะ Evidence Context เท่านั้น
- L1/L2 ห้ามใช้ baseline_summary, aux_summary หรือสรุปอื่นๆ ทั้งหมด
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

# NOTE:
# - ห้ามใช้ baseline_summary หรือ aux_summary
# - L1: P=1-2, D/C/A=0
# - L2: P=1-2, D=1-2, C/A=0
# - หากไม่มีหลักฐาน → score=0, is_passed=false

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

--- คำสั่ง ---
ประเมินตาม Evidence Context, Level Constraints เท่านั้น
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
# =================================================================
SYSTEM_ACTION_PLAN_PROMPT = f"""
คุณคือผู้เชี่ยวชาญด้าน Strategic Planning และ SEAM PDCA Maturity ระดับองค์กร
หน้าที่:
- วิเคราะห์ Failed Statements
- ระบุ PDCA Gap จาก reason + pdca_breakdown
- สร้าง Action Plan แบบ Actionable

กฎ:
1. JSON Array เท่านั้น
2. ห้ามปรับ reason เดิม
3. ต้องระบุ Responsible, Key Metric, Verification Evidence
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
- วิเคราะห์ Failed Statements ทีละข้อ
- ระบุ Gap PDCA
- เขียน Action Plan ที่ปฏิบัติได้จริง
"""

ACTION_PLAN_PROMPT = PromptTemplate(
    input_variables=["sub_id","target_level","failed_statements_list"],
    template=ACTION_PLAN_TEMPLATE
)


# =================================================================
# 4. EVIDENCE DESCRIPTION PROMPT
# =================================================================
SYSTEM_EVIDENCE_DESCRIPTION_PROMPT = f"""
คุณคือผู้เชี่ยวชาญด้าน Evidence Analysis
หน้าที่: สรุป Evidence Context อย่างเคร่งครัด
ผลลัพธ์ทุกส่วนต้องเป็นภาษาไทยเท่านั้น

{GLOBAL_RULES}
"""

USER_EVIDENCE_DESCRIPTION_TEMPLATE = """
--- เกณฑ์ ---
{sub_id} Level {level}: {sub_criteria_name}

--- Evidence Context ---
{context}

--- JSON Schema ---
{{
  "summary": "",
  "suggestion_for_next_level": ""
}}

--- คำสั่ง ---
1. summary ต้องเรียบเรียง context เป็นภาษาไทยเท่านั้น
2. suggestion_for_next_level ต้อง actionable และต้องเขียนเป็นภาษาไทยเท่านั้น
3. ต้องคืนค่าเป็น JSON Object ตรงตาม schema เท่านั้น ห้ามเพิ่ม field อื่น
"""

EVIDENCE_DESCRIPTION_PROMPT = PromptTemplate(
    input_variables=["sub_criteria_name","level","sub_id","context"],
    template=USER_EVIDENCE_DESCRIPTION_TEMPLATE
)

# =================================================================
# END OF PATCHED v16 B++ PROMPTS
# =================================================================