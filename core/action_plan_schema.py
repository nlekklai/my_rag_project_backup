# core/action_plan_schema.py
from pydantic import BaseModel, Field, field_validator
from typing import List

# -----------------------------------------------------------------------------
# 📘 Action Plan Schema for Structured LLM Output
# -----------------------------------------------------------------------------
# ใช้ในขั้นตอนที่ LLM สร้างแผนปฏิบัติการ (Action Plan) เพื่อตอบสนองต่อผลการประเมิน maturity
# -----------------------------------------------------------------------------

class ActionItem(BaseModel):
    """📌 Schema สำหรับ Action แต่ละรายการใน Action Plan"""
    Statement_ID: str = Field(
        ..., 
        description="ระบุ ID ของ Statement ที่ล้มเหลวที่ Action นี้มุ่งแก้ไข (เช่น 'L2 S3', 'L1 S1', ฯลฯ)"
    )
    Failed_Level: int = Field(
        ..., 
        description="Level ที่ Statement นี้ล้มเหลว (เช่น 1, 2, 3)"
    )
    Recommendation: str = Field(
        ..., 
        description="ข้อแนะนำเฉพาะเจาะจงเพื่อแก้ไข Gap ที่พบใน Statement นี้"
    )
    Target_Evidence_Type: str = Field(
        ..., 
        description="ประเภทของหลักฐานที่จำเป็นต้องสร้างหรือปรับปรุง (เช่น Policy Document, Signed Meeting Minutes, Training Record, ฯลฯ)"
    )
    Key_Metric: str = Field(
        ..., 
        description="ตัวชี้วัดความสำเร็จของ Action นี้ (เช่น Document approved by CXO, 90% staff trained)"
    )


class ActionPlanActions(BaseModel):
    """
    🎯 Schema หลักสำหรับผลลัพธ์ JSON ของ Action Plan
    
    ใช้ validate JSON ที่ได้จาก LLM ให้ตรงตาม format ที่ระบบต้องการ
    """
    Phase: str = Field(
        ..., 
        description="ชื่อ Phase ของแผนปฏิบัติการ เช่น '1. Foundational Gap Closure'"
    )
    Goal: str = Field(
        ..., 
        description="เป้าหมายหลักของ Phase นี้"
    )
    Actions: List[ActionItem] = Field(
        ..., 
        description="รายการ Actions ที่ต้องดำเนินการ",
        min_length=1
    )

    # -------------------------------------------------------------------------
    # 🧩 Validation Helper: รองรับกรณีที่ LLM คืนค่า 'actions' (ตัวเล็ก)
    # -------------------------------------------------------------------------
    @field_validator("Actions", mode="before")
    @classmethod
    def handle_lowercase_key(cls, v):
        """
        แก้ปัญหากรณี LLM ส่ง key 'actions' แทน 'Actions'
        """
        if isinstance(v, dict) and "actions" in v:
            return v["actions"]
        return v
