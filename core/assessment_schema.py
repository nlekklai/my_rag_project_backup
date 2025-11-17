from pydantic import BaseModel, Field
from typing import List, Dict, Any

# ====================================================================
# 1. Base Assessment Schema
# ====================================================================

class BaseAssessment(BaseModel):
    """
    Base Schema for LLM assessment result.
    """
    # คะแนนรวมของ Level (0-4)
    score: int = Field(
        ..., 
        description="The calculated overall maturity score (0-4) for the level being assessed."
    )
    # ผลลัพธ์ Pass/Fail 
    is_passed: bool = Field(
        ...,
        description="Boolean indicating if the statement is fully passed (true) or failed (false) based on the evidence."
    )
    # คำอธิบายและเหตุผล
    reason: str = Field(
        ..., 
        description="Detailed reason, in Thai, explaining the scoring and pass/fail decision, referencing specific details and citations from the context (e.g., [SOURCE: filename])."
    )

# ====================================================================
# 2. Combined Assessment Schema (CRITICAL FIX: Includes PDCA Breakdown)
# **นี่คือคลาส CombinedAssessment ที่ถูกเรียกใช้และเคยหายไป**
# ====================================================================
class CombinedAssessment(BaseAssessment):
    """
    Schema สำหรับผลลัพธ์การประเมินสุดท้าย ที่รวมคะแนน PDCA breakdown
    """
    P_Plan_Score: int = Field(
        ..., 
        description="Plan score (0, 1, or 2). 0=No evidence, 1=Partial/Incomplete, 2=Sufficient/Full."
    )
    D_Do_Score: int = Field(
        ..., 
        description="Do score (0, 1, or 2). 0=No evidence, 1=Partial/Incomplete, 2=Sufficient/Full."
    )
    C_Check_Score: int = Field(
        ..., 
        description="Check score (0, 1, or 2). 0=No evidence, 1=Partial/Incomplete, 2=Sufficient/Full."
    )
    A_Act_Score: int = Field(
        ..., 
        description="Act score (0, 1, or 2). 0=No evidence, 1=Partial/Incomplete, 2=Sufficient/Full."
    )

# ====================================================================
# 3. LLM Summary Schema (สำหรับ summarize_context_with_llm)
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