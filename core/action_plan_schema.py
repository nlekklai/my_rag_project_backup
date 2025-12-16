from pydantic import BaseModel, Field, field_validator, RootModel
from typing import List, Any, Dict
import re
import json

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
        # ลบอักขระที่ไม่ใช่ข้อความที่มองเห็นได้
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
            # ถ้า ID ไม่ได้ขึ้นต้นด้วยตัวอักษรตามด้วยตัวเลข ให้ใส่ L (Level) นำหน้า
            if not re.match(r'[A-Z]+\d+', v): 
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
            # หาก LLM ตอบเป็น object เดี่ยว แทนที่จะเป็น list
            return [v]
        if not isinstance(v, list):
            # หากไม่ใช่ list หรือ dict ให้ return list ว่าง
            return []
        return v

# -----------------------------
# 3️⃣ Action Plan Actions (Schema หลัก - กลุ่ม Phase)
# -----------------------------
class ActionPlanActions(BaseModel):
    Phase: str = Field("", description="ชื่อ Phase ของแผนปฏิบัติการ เช่น 'Foundational Gap Closure'")
    Goal: str = Field("", description="เป้าหมายหลักของ Phase นี้")
    Actions: List[ActionItem] = Field(default_factory=list, description="รายการ Actions ที่ต้องดำเนินการ")

    @field_validator("Actions", mode="before")
    @classmethod
    def handle_case_insensitive_actions(cls, v: Any) -> Any:
        # จัดการกรณีที่ LLM ใช้ 'actions' ตัวเล็ก
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

# สำหรับการ Validate ผลลัพธ์รวมที่เป็น JSON Array
class ActionPlanResult(RootModel): 
    # ใช้ root: List[ActionPlanActions] เพื่อ validate ว่า output เป็น JSON Array ของ ActionPlanActions objects
    root: List[ActionPlanActions] = Field(..., description="The complete list of action plan phases and their items.")

# =================================================================
# 4️⃣ HELPER FUNCTION: Clean JSON Schema Generator (FIX for $defs leakage)
# =================================================================
def get_clean_action_plan_schema() -> Dict[str, Any]:
    """
    Generates a clean Pydantic JSON Schema for ActionPlanResult 
    by removing $defs and $schema keys which cause LLM errors.
    """
    try:
        # Get the full schema for the RootModel (which is a list)
        schema_dict = ActionPlanResult.model_json_schema(by_alias=True)
        
        # We need to remove $defs and $schema keys from the root dictionary
        # The schema of the array contents is defined in 'items' key
        schema_dict.pop('$defs', None) 
        schema_dict.pop('$schema', None) 
        schema_dict.pop('title', None) 
        
        # Return the cleaned dictionary
        return schema_dict
        
    except Exception as e:
        # Should not happen if pydantic is correctly installed
        # In case of failure, return a simplified structure
        return {
            "type": "array",
            "description": "The complete list of action plan phases and their items.",
            "items": {
                "type": "object",
                "properties": {
                    "Phase": {"type": "string"},
                    "Goal": {"type": "string"},
                    "Actions": {"type": "array"}
                }
            }
        }