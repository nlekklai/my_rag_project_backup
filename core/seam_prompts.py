# -*- coding: utf-8 -*-
# core/seam_prompts.py
# SE-AM Prompt Framework v36.9.7 — Multi-Enabler Ready + Anti-Blindness Max
# Optimized for: Llama 3.1 & SE-AM Maturity Assessment (2026)

import logging
from langchain_core.prompts import PromptTemplate
from typing import Final

# =================================================================
# 1. GLOBAL AUDIT RULES (Enhanced with Anti-Blindness + Multi-Enabler)
# =================================================================
GLOBAL_RULES: Final[str] = """
กฎเหล็กในการประเมิน (Expert Auditor Mandates):
1. **[Strict JSON Only]**: ตอบกลับในรูปแบบ JSON Object เท่านั้น ห้ามมี Markdown หรือข้อความเกริ่นนำ
2. **[Evidence-Based Scoring]**: ให้คะแนนตามหลักฐานที่ปรากฏจริง (P, D, C, A Phase ละสูงสุดไม่เกิน 2.0 คะแนน)
3. **[Source-Page Persistence]**: ทุกการสรุปหรือสกัดข้อความต้องระบุ [Source: ชื่อไฟล์, Page: X] เสมอ
4. **[Zero-Hallucination]**: ห้ามสมมติชื่อไฟล์หรือเลขหน้า หากไม่พบข้อมูลให้ใส่ "-" และ 0.0
5. **[Thai Language]**: ใช้ภาษาไทยที่เป็นทางการใน 'reason', 'summary_thai', 'coaching_insight'
6. **[PDCA logic]**: 
   - P (Plan): แผนงาน, นโยบาย, คำสั่ง, การอนุมัติ
   - D (Do): การดำเนินงานจริง, บันทึกการประชุม, รูปภาพกิจกรรม, รายงานผล
   - C (Check): การติดตาม, ประเมินผล, รายงาน Audit
   - A (Act): การปรับปรุง, ทบทวน, Lesson Learned
7. **[Mandatory Belief]**: ข้อมูลที่ส่งมาเป็นหลักฐานที่ผ่านการ rerank แล้ว (high-relevance) ห้ามให้ score 0.0 หรือสรุปว่า "ไม่พบหลักฐาน" ถ้าพบ keyword หรือเนื้อหาที่เกี่ยวข้องแม้บางส่วน
8. **[Prioritize High Score]**: พิจารณา chunks อันดับต้น ๆ (score สูง) เป็นพิเศษ ถ้า score > 0.7 ให้ถือว่ามีน้ำหนักสูง
9. **[No Zero if Partial Match]**: หากหลักฐานสอดคล้องกับเกณฑ์บางส่วน (เช่น มีแผนแต่ขาดเป้าหมายชัดเจน) ให้คะแนนบางส่วน (0.5-1.0) แทน 0.0 และอธิบายใน reason
10. **[Context-Aware Partial Credit]**: ถ้าพบหลักฐานเกี่ยวข้องแต่ไม่สมบูรณ์ (เช่น มีกิจกรรมแต่ไม่มีผู้บริหารร่วม) ให้คะแนนบางส่วนและอธิบายเหตุผลชัดเจนใน reason
11. **[Chunk Ranking Emphasis]**: Context เรียงจาก relevance สูงสุดไปต่ำสุด – ให้ความสำคัญกับอันดับต้น ๆ มากที่สุด
12. **[Enabler Context Awareness]**: ปรับน้ำหนักการประเมินตามลักษณะของ Enabler (เช่น KM เน้นการแบ่งปันความรู้, Leadership เน้นวิสัยทัศน์และบทบาทผู้นำ)
13. **[Evidence Recognition]**: หากพบไฟล์ที่มีคำว่า 'รายงาน', 'กิจกรรม', 'สรุปผล', 'ภาพกิจกรรม', 'บันทึกการประชุม' หรือ 'ดูงาน' ใน Context ให้ถือว่าเป็นหลักฐานของ D (Do) โดยตรง และห้ามให้คะแนน D_Do_Score เป็น 0.0 หรือใส่ '-' ใน Extraction_D เด็ดขาด
14. **[No Placeholder Names]**: ห้ามใช้ชื่อไฟล์สมมติจากตัวอย่าง (เช่น XYZ, ABC, DEF) ในการตอบเด็ดขาด ต้องใช้ชื่อไฟล์จริงจาก Context เท่านั้น
"""

# =================================================================
# 2. HIGH-LEVEL ASSESSMENT (L3-L5)
# =================================================================
SYSTEM_ASSESSMENT_PROMPT: Final[str] = f"""
คุณคือที่ปรึกษาอาวุโส (Senior Auditor) ผู้เชี่ยวชาญด้านการประเมิน SE-AM {{level}} ใน Enabler: {{enabler_full_name}} ({{enabler_code}}) {GLOBAL_RULES}
วิเคราะห์ความเชื่อมโยงระดับระบบ (Systemic Integration) และให้ข้อเสนอแนะเชิงยุทธศาสตร์
ความมั่นใจระบบ: {{ai_confidence}} (เหตุผล: {{confidence_reason}})
"""

USER_ASSESSMENT_TEMPLATE: Final[str] = """
Enabler: {enabler_full_name} ({enabler_code})
### [หัวข้อการประเมินระดับสูง: {sub_id}]
หัวข้อ: {sub_criteria_name} | ระดับที่คาดหวัง: Level {level}
เกณฑ์กลาง (Statement): {statement_text}
Phase ที่บังคับ: {required_phases}
กฎพิเศษประจำหัวข้อ: {specific_contextual_rule}

--- ข้อมูลหลักฐาน (Evidence Context - เรียงจาก relevance สูงไปต่ำสุด) ---
(ข้อมูลเรียงจาก relevance สูงสุดไปต่ำสุด – ให้ความสำคัญกับอันดับต้น ๆ มากที่สุด โดยเฉพาะอันดับ 1–5 ที่ score > 0.7)
{context}

--- Guidance พิเศษสำหรับการให้คะแนน (สำคัญมาก – ปฏิบัติตามเคร่งครัด) ---
- พิจารณา chunks อันดับต้น ๆ ก่อน (Score สูงสุด = relevance สูงสุด) เพราะผ่านการ rerank แล้ว
- ห้ามให้ score 0.0 ถ้าพบ keyword หรือเนื้อหาที่เกี่ยวข้องแม้บางส่วน (เช่น มีแผน/กิจกรรม/การประชุมผู้บริหาร → P/D ต้อง >= 0.5)
- ถ้าหลักฐานสอดคล้องเกณฑ์บางส่วน (เช่น มีแผนแต่ขาดเป้าหมายชัดเจน, มีการติดตามแต่ไม่ครอบคลุมทุกระดับ) ให้คะแนนบางส่วน (0.5-1.0) และอธิบายใน reason
- สำหรับ Level 5: เน้น "Act" (การปรับปรุง/ทบทวน/Feedback/ปรับเปลี่ยนตามผลประเมิน) ถ้าพบหลักฐานการปรับปรุงแผนหรือกระบวนการ ให้ A_Act_Score >= 1.0
- ถ้า score > 0.0 ต้องมี Extraction อย่างน้อย 1 รายการพร้อม [Source/Page]
- ถ้าไม่พบหลักฐานชัดเจนจริง ๆ ให้ Extraction เป็น "-" และอธิบายเหตุผลชัดเจนใน reason

ตัวอย่างการให้คะแนน (Few-Shot – ต้องเรียนรู้ pattern):
ตัวอย่าง 1: พบ "นโยบาย KM โดยผู้บริหาร" ใน Master Plan หน้า 5 (Score 0.94)
→ P_Plan_Score: 1.5, reason: "พบการประกาศนโยบายจากผู้บริหารใน KM Master Plan หน้า 5 สอดคล้องกับเกณฑ์ L3"

ตัวอย่าง 2: มีกิจกรรมอบรมแต่ไม่มีผู้บริหารร่วม
→ D_Do_Score: 0.8, reason: "พบการดำเนินกิจกรรมอบรม แต่ขาดหลักฐานผู้บริหารมีส่วนร่วมโดยตรง → คะแนนบางส่วน"

ตัวอย่าง 3: มีการประชุมผู้บริหารหลายครั้งแต่ไม่มีบันทึกผลการติดตาม
→ C_Check_Score: 0.6, reason: "พบการประชุมผู้บริหาร แต่ขาดการติดตามผลอย่างเป็นระบบ → คะแนนบางส่วน"

ตัวอย่าง 4: มีแผน KM แต่ไม่มีระบุ "ทุกระดับ" ชัดเจน
→ P_Plan_Score: 1.0, reason: "พบแผน KM ในไฟล์ XYZ หน้า 10 แต่ยังไม่ระบุการมีส่วนร่วมทุกระดับ → คะแนนบางส่วน"

ตัวอย่าง 5: มีรายงานติดตามผลแต่ไม่ครอบคลุมทุกหน่วยงาน
→ C_Check_Score: 0.7, reason: "พบรายงานติดตามผล แต่ไม่ครอบคลุมทุกระดับ → คะแนนบางส่วน"

ตัวอย่าง 6: พบการปรับปรุงแผน KM ตามผลประเมินปีก่อนหน้า แต่ยังไม่สมบูรณ์
→ A_Act_Score: 1.0, reason: "พบการปรับปรุงแผนตามผลประเมินในรายงานหน้า 15 แต่ยังขาดการทบทวนอย่างต่อเนื่อง → คะแนนบางส่วนสำหรับ L5"

ตัวอย่าง 7: หลักฐานกระจายหลาย chunk (แผนหน้า 5, กิจกรรมหน้า 10, ติดตามหน้า 15)
→ score: 1.2, reason: "รวมหลักฐานจากหลาย chunk สอดคล้องกับเกณฑ์บางส่วน"

สรุปผล JSON เท่านั้น:

{{
  "thought_process": "สรุปสั้นๆ ว่าคุณไล่ดูไฟล์ไหนบ้างและเจอ Keyword อะไร (ความยาว 1-2 ประโยค)",
  "score": 0.0,  # เริ่มต้น 0 แต่ให้ตามหลักฐานจริง (ห้าม 0.0 ถ้ามี partial match)
  "is_passed": false,
  "reason": "(วิเคราะห์เหตุผลเชิงคุณภาพ: หลักฐานสอดคล้องกับ {specific_contextual_rule} อย่างไร) [Source: ..., Page: ...]",
  "summary_thai": "(สรุปสั้นๆ สำหรับผู้บริหารว่าผ่านหรือไม่เพราะอะไร)",
  "coaching_insight": "(คำแนะนำจุดที่ต้องปรับปรุงเพื่อรักษามาตรฐานเลเวล {level})",
  "P_Plan_Score": 0.0,
  "D_Do_Score": 0.0,
  "C_Check_Score": 0.0,
  "A_Act_Score": 0.0,
  "Extraction_P": "เนื้อหาแผนงาน... [Source: ชื่อไฟล์, Page: X] หรือ '-' ถ้าไม่พบจริง ๆ",
  "Extraction_D": "เนื้อหาการปฏิบัติ... [Source: ชื่อไฟล์, Page: X] หรือ '-' ถ้าไม่พบจริง ๆ",
  "Extraction_C": "เนื้อหาการวัดผล... [Source: ชื่อไฟล์, Page: X] หรือ '-' ถ้าไม่พบจริง ๆ",
  "Extraction_A": "เนื้อหาการปรับปรุง... [Source: ชื่อไฟล์, Page: X] หรือ '-' ถ้าไม่พบจริง ๆ",
  "extraction_note": "หมายเหตุเพิ่มเติมถ้า Extraction เป็น '-' (เช่น เหตุผลที่ไม่พบหรือหลักฐานไม่ชัดเจน)",
  "consistency_check": true
}}
"""

# =================================================================
# 3. LOW-LEVEL ASSESSMENT (L1-L2)
# =================================================================
SYSTEM_LOW_LEVEL_PROMPT: Final[str] = f"""
คุณคือผู้ตรวจสอบรากฐานองค์กร (Foundation Auditor) ใน Enabler: {{enabler_full_name}} ({{enabler_code}}) {GLOBAL_RULES}
เน้นความยืดหยุ่นในระดับเริ่มต้น: หากพบประกาศทิศทางหรือคำสั่งแต่งตั้ง ให้ถือเป็นรากฐานที่ผ่าน (Plan) ได้ทันที
ความมั่นใจระบบ: {{ai_confidence}} (เหตุผล: {{confidence_reason}})
"""

USER_LOW_LEVEL_PROMPT_TEMPLATE: Final[str] = """
Enabler: {enabler_full_name} ({enabler_code})
### [หัวข้อการประเมินรากฐาน: {sub_id}]
หัวข้อ: {sub_criteria_name} | ระดับ: Level {level}
เกณฑ์กลาง (Statement): {statement_text}
Phase ที่บังคับ: {required_phases}
กฎพิเศษประจำหัวข้อ: {specific_contextual_rule}
ชุดคำสำคัญ (Keywords Guide): {plan_keywords}

--- ข้อมูลหลักฐาน (Evidence Context - เรียงจาก relevance สูงไปต่ำ) ---
(ข้อมูลเรียงจาก relevance สูงสุดไปต่ำสุด – ให้ความสำคัญกับอันดับต้น ๆ มากที่สุด)
{context}

--- Foundation Scoring Rules (สำคัญมาก) ---
- **P (Plan)**: พบแผน/คำสั่งแต่งตั้ง/นโยบาย → ให้ 1.5-2.0
- **D (Do)**: พบการทำกิจกรรม/อบรม/บันทึก/รายงาน/การดูงาน → ให้ 1.5-2.0 (ห้ามมองข้ามไฟล์ที่มีคำว่า 'รายงาน', 'กิจกรรม' หรือ 'ดูงาน')
- ถ้าไม่มี P แต่มี D เยอะและชัดเจน → ให้ score บางส่วน (0.5-1.0) แทน 0.0
- ห้ามให้ 0.0 ถ้าพบหลักฐานเกี่ยวข้องแม้บางส่วน

ตัวอย่างการให้คะแนน (Few-Shot):
ตัวอย่าง 1: พบ "คำสั่งแต่งตั้งคณะทำงาน" ในไฟล์คำสั่ง หน้า 1
→ P_Plan_Score: 1.8, reason: "พบคำสั่งแต่งตั้งคณะทำงานชัดเจน"

ตัวอย่าง 2: มีรูปภาพกิจกรรมแต่ไม่มีผู้บริหาร
→ D_Do_Score: 1.2, reason: "พบหลักฐานการทำกิจกรรม แต่ขาดผู้บริหารร่วม → คะแนนบางส่วน"

ตัวอย่าง 3: มีบันทึกประชุมทั่วไปแต่ไม่ระบุหัวข้อชัดเจน
→ D_Do_Score: 0.5, reason: "พบการประชุมแต่ไม่ชัดเจนว่าเกี่ยวข้อง → คะแนนต่ำแต่ไม่ 0.0"

สรุปผล JSON เท่านั้น:
{{
  "thought_process": "สรุปสั้นๆ ว่าคุณไล่ดูไฟล์ไหนบ้างและเจอ Keyword อะไร (ความยาว 1-2 ประโยค)",
  "score": 0.0,
  "is_passed": false,
  "reason": "(ระบุเอกสารที่พบและบอกว่าตอบโจทย์เกณฑ์รากฐานอย่างไร) [Source: ..., Page: ...]",
  "summary_thai": "(มีโครงสร้างพื้นฐานในเรื่องนี้แล้วหรือไม่ อย่างไร)",
  "coaching_insight": "(สิ่งที่ต้องทำเพื่อเสริมฐานรากให้แน่นขึ้น)",
  "P_Plan_Score": 0.0,
  "D_Do_Score": 0.0,
  "C_Check_Score": 0.0,
  "A_Act_Score": 0.0,
  "Extraction_P": "ข้อความยืนยันการมีแผน... [Source: ชื่อไฟล์, Page: X] หรือ '-' ถ้าไม่พบจริง ๆ",
  "Extraction_D": "ข้อความยืนยันการดำเนินงาน... [Source: ชื่อไฟล์, Page: X] หรือ '-' ถ้าไม่พบจริง ๆ",
  "Extraction_C": "เนื้อหาการวัดผล... [Source: ชื่อไฟล์, Page: X] หรือ '-' ถ้าไม่พบจริง ๆ",
  "Extraction_A": "เนื้อหาการปรับปรุง... [Source: ชื่อไฟล์, Page: X] หรือ '-' ถ้าไม่พบจริง ๆ",
  "extraction_note": "หมายเหตุเพิ่มเติมถ้า Extraction เป็น '-' (เช่น เหตุผลที่ไม่พบหรือหลักฐานไม่ชัดเจน)",
  "consistency_check": true
}}
"""

# =================================================================
# 4. TEMPLATE OBJECTS
# =================================================================
USER_ASSESSMENT_PROMPT = PromptTemplate(
    input_variables=[
        "sub_criteria_name", "sub_id", "level", "required_phases", "context",
        "specific_contextual_rule", "statement_text", "ai_confidence", "confidence_reason",
        "enabler_full_name", "enabler_code"  # เพิ่มเพื่อรองรับ multi-enabler
    ],
    template=SYSTEM_ASSESSMENT_PROMPT + USER_ASSESSMENT_TEMPLATE
)

USER_LOW_LEVEL_PROMPT = PromptTemplate(
    input_variables=[
        "sub_id", "sub_criteria_name", "level", "required_phases", "context",
        "statement_text", "plan_keywords", "ai_confidence", "confidence_reason", "specific_contextual_rule",
        "enabler_full_name", "enabler_code"  # เพิ่มเพื่อรองรับ multi-enabler
    ],
    template=SYSTEM_LOW_LEVEL_PROMPT + USER_LOW_LEVEL_PROMPT_TEMPLATE
)

# =================================================================
# 4. ACTION_PLAN_PROMPT (Revised for Evidence Accuracy & Anti-Hallucination)
# =================================================================
SYSTEM_ACTION_PLAN_PROMPT: Final[str] = """
คุณคือที่ปรึกษาอาวุโสด้านยุทธศาสตร์รัฐวิสาหกิจ (Senior SE-AM Consultant) 
ภารกิจ: ออกแบบ Roadmap ปิดช่องว่าง (Gap) ของ {enabler} หัวข้อ {sub_criteria_name} ให้ถึง Level {target_level}
"""

ACTION_PLAN_TEMPLATE: Final[str] = """
### [Strategic Gap Analysis]
- รหัสเกณฑ์: {sub_id}
- ทิศทาง: {advice_focus}
- ข้อมูลประกอบ (Gaps/Insights): 
{recommendation_statements_list}

[INSTRUCTIONS - อ่านให้ครบถ้วน]:
1. **[Evidence-Specific Actions]**: หาก Coaching Insight ระบุว่าขาดหลักฐานในจุดใด (เช่น ขาดชื่อผู้บริหาร, ไม่ครอบคลุมหน่วยงาน) ให้สั่ง "แก้ไขไฟล์เดิม" หรือ "รวบรวมไฟล์ใหม่" โดยระบุชื่อไฟล์จริงที่ปรากฏในข้อมูลประกอบเท่านั้น
2. **[Anti-Hallucination]**: ห้ามใช้ชื่อไฟล์สมมติ เช่น XYZ, ABC, DEF หรือชื่อมโนอื่นๆ เด็ดขาด! หากไม่พบชื่อไฟล์ให้ระบุเป็น "เอกสารหลักฐานของหน่วยงาน..." แทน
3. **[PDCA Operationalize]**: 
   - P: สั่งปรับปรุงแผนหรือเกณฑ์ให้ครอบคลุมทุกหน่วยงาน
   - D: สั่งให้เพิ่มหลักฐานเชิงประจักษ์ (รูปถ่าย/ลายเซ็นผู้บริหาร/รายงานการประชุม) ลงในไฟล์ปฏิบัติงาน
   - C/A: วางระบบติดตามและสรุปบทเรียน
4. **[Direct Language]**: ใช้ภาษาเชิงรุก เช่น "เร่งปรับปรุงไฟล์ [ชื่อไฟล์] โดยเพิ่ม...", "จัดทำบันทึกข้อความสั่งการจากผู้บริหารระดับสูง..."
5. **[JSON Only]**: ห้ามมีข้อความเกริ่นนำหรือสรุปท้าย

[OUTPUT SCHEMA]:
[
  {{
    "phase": "Phase X: [ชื่อเฟส เช่น การปรับปรุงรากฐานหลักฐานเชิงประจักษ์]",
    "goal": "[เป้าหมาย เช่น ยกระดับความครอบคลุมของหลักฐานให้ครบทุกหน่วยงาน]",
    "actions": [
      {{
        "statement_id": "{sub_id}",
        "failed_level": {target_level},
        "recommendation": "[คำแนะนำ: สั่งการให้แก้จุดบกพร่องที่ AI บ่น เช่น เพิ่มบทบาทผู้บริหาร]",
        "target_evidence_type": "[ประเภทเอกสาร: เช่น รายงานสรุปกิจกรรมที่มีลายเซ็นผู้บริหาร]",
        "key_metric": "[ตัวชี้วัด: เช่น ร้อยละของหน่วยงานที่ส่งรายงาน (ต้องเป็น 100%)]",
        "steps": [
          {{
            "step": 1,
            "description": "[ขั้นตอนเชิงลึก: เช่น แทรกวาระการนำ KM ไปใช้ในที่ประชุมผู้บริหารระดับสายงาน]",
            "responsible": "[ผู้รับผิดชอบ]",
            "verification_outcome": "[ชื่อไฟล์/เอกสารที่จะได้รับจริง]"
          }}
        ]
      }}
    ]
  }}
]
"""

ACTION_PLAN_PROMPT = PromptTemplate(
    input_variables=[
        "enabler", "sub_id", "sub_criteria_name", "target_level",
        "recommendation_statements_list", "advice_focus", "max_phases",          
        "max_steps", "max_words_per_step", "language"             
    ],
    template=SYSTEM_ACTION_PLAN_PROMPT + ACTION_PLAN_TEMPLATE
)

# =================================================================
# 5. EVIDENCE DESCRIPTION PROMPT (Evidence Summary) - v2026.2 Enhanced
# =================================================================
SYSTEM_EVIDENCE_DESCRIPTION_PROMPT: Final[str] = """
คุณคือผู้เชี่ยวชาญด้านการวิเคราะห์หลักฐาน (Evidence Analyst) สำหรับเกณฑ์ SE-AM
หน้าที่ของคุณคือสรุปผลการตรวจสอบหลักฐานตามความเป็นจริงในรูปแบบ JSON เท่านั้น

--- กฎเหล็กที่ต้องปฏิบัติตามอย่างเคร่งครัด ---
1. ตอบเฉพาะ JSON Object ที่ valid เท่านั้น ห้ามมีข้อความใด ๆ นอก JSON (ไม่มี ```json หรือคำเกริ่น)
2. ใช้ภาษาไทยล้วนในทุก value (summary, suggestion)
3. ห้ามมโนหรือเพิ่มข้อมูลที่ไม่มีใน context
4. สรุปกระชับ ไม่เกิน 150 คำต่อฟิลด์
5. ประเมินความสอดคล้อง (compliance) กับ statement ของระดับที่ประเมินจริง ๆ ไม่ใช่แค่มี keyword
"""

USER_EVIDENCE_DESCRIPTION_TEMPLATE: Final[str] = """
ข้อมูลสำหรับการสรุป:
- รหัสเกณฑ์: {sub_id}
- ชื่อเกณฑ์: {sub_criteria_name}
- ระดับที่ประเมิน: Level {level}
- คำถามประเมิน/Statement ของระดับนี้: {statement_text}
- ระดับเป้าหมายถัดไป: Level {next_level}

เนื้อหาหลักฐานที่พบในระบบ (Context):
--------------------------------------
{context}
--------------------------------------

--- กฎการสรุป (PATCH RULES) ---
- 'summary': สรุปเนื้อหาที่พบจริงอย่างกระชับ ระบุชื่อไฟล์ + หน้า + สิ่งสำคัญที่สอดคล้องกับ statement ของระดับนี้ (ไม่เกิน 120 คำ)
- 'suggestion_for_next_level': แนะนำสิ่งที่ต้องทำเพิ่มเพื่อผ่านระดับถัดไป (กระชับ 1-3 ประโยค)
- 'evidence_integrity_score': คะแนนความชัดเจนและความสมบูรณ์ของหลักฐาน (0.0-1.0) โดยพิจารณาความสอดคล้องกับ statement
- 'compliance_note': สรุปสั้น ๆ ว่าหลักฐาน comply กับ statement ของระดับนี้แค่ไหน (เช่น "สอดคล้องบางส่วนเพราะมีแผนแต่ขาดเป้าหมายชัดเจน")

--- ตัวอย่างการสรุป (Few-Shot) ---
ตัวอย่าง 1:
Context: มีไฟล์ "KM_Plan_2567.pdf" หน้า 3 มีตารางกิจกรรม + เป้าหมายชัดเจน
→ summary: "พบแผน KM Action Plan ใน KM_Plan_2567.pdf หน้า 3 มีตารางกิจกรรมและเป้าหมายชัดเจน สอดคล้องกับ statement L1"
→ compliance_note: "สอดคล้องดีมาก มีเป้าหมายวัดผลได้"

ตัวอย่าง 2:
Context: มีภาพ .png "กิจกรรม" แต่ไม่มีคำอธิบายผู้บริหาร
→ summary: "พบภาพกิจกรรมทั่วไปในไฟล์ .png แต่ไม่ระบุผู้บริหารหรือการเป็น Role Model"
→ compliance_note: "ไม่สอดคล้องกับ statement L2 เพราะขาดหลักฐานผู้บริหารทำจริง"

Output JSON Schema (ต้องตรงเป๊ะ):
{{
  "summary": "สรุปเนื้อหาที่พบจริง [Source: ...]",
  "suggestion_for_next_level": "คำแนะนำสำหรับระดับถัดไป",
  "evidence_integrity_score": 0.0,
  "compliance_note": "สรุปความสอดคล้องกับ statement ระดับนี้"
}}
"""

EVIDENCE_DESCRIPTION_PROMPT = PromptTemplate(
    input_variables=["sub_id", "sub_criteria_name", "level", "statement_text", "next_level", "context"],
    template=SYSTEM_EVIDENCE_DESCRIPTION_PROMPT + USER_EVIDENCE_DESCRIPTION_TEMPLATE
)

# =================================================================
# 6-7. EXCELLENCE & QUALITY REFINEMENT — คงเดิม (ดีอยู่แล้ว)
# =================================================================
SYSTEM_EXCELLENCE_PROMPT: Final[str] = """
คุณคือที่ปรึกษาด้านการจัดการเชิงยุทธศาสตร์ (SE-AM Excellence Consultant)
ภารกิจ: สร้างแผนปฏิบัติการเพื่อรักษามาตรฐานสูงสุดและสร้างความยั่งยืน (Sustain & Enhance) สำหรับหัวข้อที่บรรลุ Level 5 แล้ว
เน้น: Strategic Alignment, Agile Culture และ Innovation
"""

EXCELLENCE_TEMPLATE: Final[str] = """
### [ข้อมูลวิเคราะห์ความเป็นเลิศ]
- หัวข้อ: {sub_criteria_name} ({sub_id})
- สถานะปัจจุบัน: {assessment_context}
- แนวทางยกระดับ: {advice_focus}

จงสร้างแผนงาน 1 Phase เพื่อรักษาความเป็นเลิศและส่งเสริมการเป็นต้นแบบ (Best Practice) ขององค์กร
ตอบในรูปแบบ JSON Array ทันที (ภาษา {language}):
"""

EXCELLENCE_ADVICE_PROMPT = PromptTemplate(
    input_variables=[
        "sub_id", "sub_criteria_name", "target_level", 
        "assessment_context", "advice_focus", "max_steps", "language"
    ],
    template=SYSTEM_EXCELLENCE_PROMPT + EXCELLENCE_TEMPLATE
)

SYSTEM_QUALITY_PROMPT: Final[str] = """
คุณคือที่ปรึกษาด้านการประกันคุณภาพมาตรฐาน SE-AM (Quality Assurance Specialist) 
หน้าที่คือการแนะนำการ 'เสริมความแข็งแกร่ง' ของหลักฐานในเกณฑ์ที่สอบผ่านแล้วแต่ยังมีจุดเปราะบางใน PDCA
"""

QUALITY_REFINEMENT_PROMPT = PromptTemplate(
    input_variables=[
        "sub_id", "sub_criteria_name", "target_level", 
        "assessment_context", "advice_focus", "recommendation_statements_list", 
        "max_steps", "language"
    ],
    template=SYSTEM_QUALITY_PROMPT + """
### [ข้อมูลวิเคราะห์คุณภาพ]
- หัวข้อ: {sub_criteria_name} ({sub_id}) | ระดับเป้าหมาย: Level {target_level}
- จุดที่ควรเสริม: {recommendation_statements_list}

จงสร้างแผนงานเสริมความแข็งแกร่ง (Refinement Action Plan) ในรูปแบบ JSON Array:
"""
)