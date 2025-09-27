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


# -------------------- Compare Prompt for /compare --------------------
# ใช้สำหรับเปรียบเทียบเอกสาร 2 ชิ้น
COMPARE_TEMPLATE = """
คุณคือผู้ช่วยวิเคราะห์เชิงลึก ภารกิจของคุณคือการเปรียบเทียบเนื้อหาของเอกสาร 2 ชิ้นที่ได้รับ

[เอกสารที่ 1]:
---
{doc_a}
---

[เอกสารที่ 2]:
---
{doc_b}
---

คำสั่ง:
1. ระบุประเด็นหลักที่เหมือนกัน (Similarities)
2. ระบุประเด็นหลักที่แตกต่างกัน (Differences)
3. สรุปผลการเปรียบเทียบโดยรวมเป็นภาษาไทยที่กระชับ
"""

COMPARE_PROMPT = PromptTemplate(
    template=COMPARE_TEMPLATE, input_variables=["doc_a", "doc_b"]
)
