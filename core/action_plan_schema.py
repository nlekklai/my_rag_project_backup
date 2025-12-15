# action_plan_models.py
from pydantic import BaseModel, Field, field_validator
from typing import List, Any
import re

# -----------------------------
# 1️⃣ Step Detail (ขั้นตอนย่อย)
# -----------------------------
class StepDetail(BaseModel):
    Step: str = Field(..., description="ลำดับของขั้นตอนย่อย เช่น '1', '2', '3'")
    Description: str = Field(..., description="รายละเอียดของสิ่งที่ต้องทำในขั้นตอนนี้")
    Responsible: str = Field(..., description="ชื่อตำแหน่ง หรือหน่วยงานที่รับผิดชอบ")
    Tools_Templates: str = Field(..., description="เครื่องมือ/Template/เอกสารที่ต้องใช้")
    Verification_Outcome: str = Field(..., description="สิ่งที่ต้องตรวจสอบหรือผลลัพธ์ที่ต้องเกิดขึ้นเมื่อเสร็จสิ้น")

    @field_validator("Step", mode="before")
    @classmethod
    def ensure_step_is_str(cls, v: Any) -> str:
        return str(v).strip() if v is not None else "1"

    @field_validator("Description", "Responsible", "Tools_Templates", "Verification_Outcome", mode="before")
    @classmethod
    def sanitize_text(cls, v: Any) -> str:
        if v is None:
            return ""
        v = re.sub(r'[\n\r\t\u200b\u200c\u200d\uFEFF]+', ' ', str(v))
        return v.strip()

# -----------------------------
# 2️⃣ Action Item (รายการปฏิบัติการ)
# -----------------------------
class ActionItem(BaseModel):
    Statement_ID: str = Field(..., description="ระบุ ID ของ Statement ที่ล้มเหลว เช่น 'L2_S3', 'L1_S1'")
    Failed_Level: int = Field(..., description="Level ที่ Statement นี้ล้มเหลว (เช่น 1, 2, 3)")
    Recommendation: str = Field(..., description="ข้อแนะนำเฉพาะเจาะจงเพื่อแก้ไข Gap ที่พบใน Statement นี้")
    Target_Evidence_Type: str = Field(..., description="ประเภทของหลักฐานที่ต้องสร้างหรือปรับปรุง")
    Key_Metric: str = Field(..., description="ตัวชี้วัดความสำเร็จของ Action นี้")
    Steps: List[StepDetail] = Field(default_factory=list, description="รายการขั้นตอนปฏิบัติย่อย")

    @field_validator("Statement_ID", mode="before")
    @classmethod
    def normalize_statement_id(cls, v: Any) -> str:
        if isinstance(v, str):
            v = v.strip().replace(' ', '_').replace('-', '_').upper()
            if not v.startswith('L'):
                v = 'L' + v
            return v
        return str(v)

    @field_validator("Recommendation", "Target_Evidence_Type", "Key_Metric", mode="before")
    @classmethod
    def sanitize_text(cls, v: Any) -> str:
        if v is None:
            return ""
        v = re.sub(r'[\n\r\t\u200b\u200c\u200d\uFEFF]+', ' ', str(v))
        return v.strip()

    @field_validator("Steps", mode="before")
    @classmethod
    def ensure_steps_is_list(cls, v: Any) -> List[Any]:
        if v is None:
            return []
        if isinstance(v, dict):
            return [v]
        if not isinstance(v, list):
            return []
        return v

# -----------------------------
# 3️⃣ Action Plan Actions (Schema หลัก - แก้ไข Phase และ Goal)
# -----------------------------
class ActionPlanActions(BaseModel):
    # แก้ไข: กำหนดค่าเริ่มต้นเป็น String ว่าง ("") ทำให้ฟิลด์เป็น Optional
    Phase: str = Field("", description="ชื่อ Phase ของแผนปฏิบัติการ เช่น 'Foundational Gap Closure'")
    Goal: str = Field("", description="เป้าหมายหลักของ Phase นี้")
    Actions: List[ActionItem] = Field(default_factory=list, description="รายการ Actions ที่ต้องดำเนินการ")

    @field_validator("Actions", mode="before")
    @classmethod
    def handle_case_insensitive_actions(cls, v: Any) -> Any:
        if isinstance(v, dict):
            if "actions" in v:
                return v["actions"]
            if "Actions" in v:
                return v["Actions"]
        return v

    @field_validator("Phase", "Goal", mode="before")
    @classmethod
    def sanitize_text(cls, v: Any) -> str:
        if v is None:
            return ""
        v = re.sub(r'[\n\r\t\u200b\u200c\u200d\uFEFF]+', ' ', str(v))
        return v.strip()