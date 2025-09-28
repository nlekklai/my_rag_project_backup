#core/rag_prompts.py
from langchain.prompts import PromptTemplate

# -------------------- QA Prompt for Step 4 --------------------
# ใช้สำหรับให้ LLM สรุปผลการประเมินโดยอ้างอิงจาก Context ที่ได้จาก Step 3
QA_TEMPLATE = """
คุณคือผู้ประเมินคุณภาพที่เป็นกลางและแม่นยำ 
ภารกิจของคุณคือการสรุปผลการประเมินจากหลักฐานที่ให้มา (Context) เพื่อตอบคำถามที่เกี่ยวข้อง 

[Context จาก Rubrics, Evidence, Feedback ที่เกี่ยวข้อง]:
---
{context}
---

[คำถามประเมิน]:
{question}

กรุณาตอบคำถามด้านบนโดย:
1. อ้างอิงเฉพาะข้อมูลที่อยู่ใน Context เท่านั้น
2. ระบุว่าหลักฐานที่พบมีความแข็งแกร่ง (Strong) หรือมีช่องโหว่ (Gap) อย่างไร
3. สรุปผลการประเมินเป็นภาษาไทยที่กระชับ ไม่เกิน 3 ย่อหน้า
"""

QA_PROMPT = PromptTemplate(
    template=QA_TEMPLATE, input_variables=["context", "question"]
)


# -------------------- Compare Prompt for /compare (FIXED) --------------------
COMPARE_TEMPLATE = """
คุณคือผู้ช่วยวิเคราะห์เชิงลึก ภารกิจของคุณคือการเปรียบเทียบเนื้อหาของเอกสาร 2 ชิ้นที่เกี่ยวข้อง ({doc_names}) โดยใช้ข้อมูลบริบทที่ให้มา

[บริบท (Context) ที่รวมข้อมูลจากทั้งสองเอกสาร]:
---
{context}  <--- แก้ไขให้รับ context รวมทั้งหมด
---

[คำสั่ง]:
1. สรุปเนื้อหาสำคัญของเอกสารแต่ละฉบับที่เกี่ยวข้องกับคำถามเปรียบเทียบ
2. ระบุประเด็นหลักที่แตกต่างกัน (Differences) ในด้านเป้าหมาย ตัวชี้วัด หรือระยะเวลาดำเนินการ
3. สรุปผลการเปรียบเทียบโดยรวมเป็นภาษาไทยที่กระชับ

**คำถามเปรียบเทียบ:** {query}
"""

COMPARE_PROMPT = PromptTemplate(
    # *** FIX: เปลี่ยน input_variables เป็น "context", "query", และ "doc_names" ***
    template=COMPARE_TEMPLATE, input_variables=["context", "query", "doc_names"]
)


COMPARE_PROMPT = PromptTemplate(
    template=COMPARE_TEMPLATE, input_variables=["context", "query", "doc_names"]
)

# -------------------- Semantic Mapping Prompt for Step 3 --------------------
SEMANTIC_MAPPING_TEMPLATE = """
คุณคือผู้ช่วยที่เชี่ยวชาญด้านการจับคู่ข้อมูลเชิงความหมาย (Semantic Mapping)
ภารกิจของคุณคือการวิเคราะห์คำถามประเมินและหา context ที่เกี่ยวข้องที่สุดจากเอกสารหลายประเภท ได้แก่ Rubrics, Evidence และ Feedback

[คำถาม]:
{question}

[เอกสารสำหรับจับคู่]:
---
{documents}
---

กรุณาสร้าง Mapping ดังนี้:
1. Mapped Evidence: เลือก chunks ของ Evidence ที่เกี่ยวข้อง
2. Mapped Rubric: เลือก Rubric ที่ตรงกับคำถาม
3. Suggested Action: ข้อเสนอแนะถ้ามี Gap
4. ให้คะแนนความเกี่ยวข้อง (Relevance Score) เป็นตัวเลข 0-1

ผลลัพธ์ให้อยู่ในรูปแบบ JSON ดังนี้:
{{
    "mapped_evidence": [...],
    "mapped_rubric": [...],
    "suggested_action": "...",
    "relevance_score": 0.0
}}
"""

SEMANTIC_MAPPING_PROMPT = PromptTemplate(
    template=SEMANTIC_MAPPING_TEMPLATE, input_variables=["question", "documents"]
)
