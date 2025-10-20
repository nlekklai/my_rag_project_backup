#core/assessment_schema.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any

# ====================================================================
# 1. LLM Assessment Schema (สำหรับ evaluate_with_llm)
# บังคับให้ LLM ตอบกลับมาในรูปแบบที่กำหนดเพื่อการประมวลผลต่อ
# ====================================================================

class StatementAssessment(BaseModel):
    """
    Schema สำหรับผลลัพธ์การประเมิน Statement โดย LLM 
    """
    # บังคับให้ LLM ให้คะแนนเป็น 1 (Passed) หรือ 0 (Failed) เท่านั้น
    score: int = Field(
        ..., 
        description="Must be 1 if the evidence context clearly supports the statement based on the standard, or 0 if it does not."
    )
    # คำอธิบายว่าทำไมถึงให้คะแนนนี้
    reason: str = Field(
        ..., 
        description="Detailed reason, in Thai, explaining why the statement received the given score, referencing specific details from the context."
    )

# ====================================================================
# 2. LLM Summary Schema (สำหรับ summarize_context_with_llm)
# ====================================================================

class EvidenceSummary(BaseModel):
    """
    Schema สำหรับผลลัพธ์การสรุปหลักฐานที่รวบรวมได้ของ Level นั้นๆ โดย LLM 
    """
    # สรุปหลักฐาน (Evidence Description) ที่ Level นั้นๆ ทำได้แล้ว 
    summary: str = Field(
        ..., 
        description="A concise, high-level summary, in Thai, of the key evidences and achievements demonstrated in the provided context for the specified maturity level."
    )
    # ข้อเสนอแนะเพิ่มเติม (ถ้ามี)
    suggestion_for_next_level: str = Field(
        ..., 
        description="A brief suggestion, in Thai, on what should be the focus for development towards the next maturity level."
    )
