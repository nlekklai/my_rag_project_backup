# core/action_plan_schema.py

from pydantic import BaseModel, Field, field_validator, RootModel, ConfigDict
from typing import List, Any, Dict, Optional
import re
import json

# -----------------------------
# 1. Step Detail: ขั้นตอนย่อย
# -----------------------------
class StepDetail(BaseModel):
    # เปลี่ยนจาก str เป็น int เพื่อความ Strict
    Step: int = Field(..., description="ลำดับที่ (1, 2, 3...)")
    Description: str = Field(..., description="กิจกรรมที่ต้องทำ")
    Responsible: str = Field(..., description="หน่วยงาน/ตำแหน่งที่รับผิดชอบ")
    Tools_Templates: str = Field(..., description="ชื่อไฟล์ Template หรือระบบที่ใช้")
    Verification_Outcome: str = Field(..., description="ผลลัพธ์หรือหลักฐานที่ได้ (Evidence)")

    @field_validator("Step", mode="before")
    @classmethod
    def ensure_int(cls, v: Any) -> int:
        if isinstance(v, int): return v
        # ดักจับกรณีหลุดมาเป็น "Step 1" หรือ "1"
        nums = re.findall(r'\d+', str(v))
        return int(nums[0]) if nums else 0

    @field_validator("Description", "Responsible", "Tools_Templates", "Verification_Outcome", mode="before")
    @classmethod
    def sanitize_text(cls, v: Any) -> str:
        if v is None: return ""
        # ล้างพวก Newline หรือช่องว่างประหลาด
        v = re.sub(r'[\n\r\t\u200b\u200c\u200d\uFEFF]+', ' ', str(v))
        return v.strip()

# -----------------------------
# 2. Action Item: รายการแผนงาน
# -----------------------------
class ActionItem(BaseModel):
    Statement_ID: str = Field(..., alias="statement_id")
    Failed_Level: int = Field(..., alias="failed_level")
    Recommendation: str = Field(..., alias="recommendation")
    Target_Evidence_Type: str = Field(..., alias="target_evidence_type")
    Key_Metric: str = Field(..., alias="key_metric")
    Steps: List[StepDetail] = Field(default_factory=list, alias="steps")

    model_config = ConfigDict(populate_by_name=True, coerce_numbers_to_str=False)

    @field_validator("Failed_Level", mode="before")
    @classmethod
    def clean_level(cls, v: Any) -> int:
        if isinstance(v, int): return v
        # ดักจับ "Level 3" -> 3
        nums = re.findall(r'\d+', str(v))
        return int(nums[0]) if nums else 0

# -----------------------------
# 3. ActionPlanActions: ระยะ (Phase)
# -----------------------------
class ActionPlanActions(BaseModel):
    Phase: str = Field(..., alias="phase")
    Goal: str = Field(..., alias="goal")
    Actions: List[ActionItem] = Field(default_factory=list, alias="actions")

    model_config = ConfigDict(populate_by_name=True)

# -----------------------------
# 4. Root Model & Helper
# -----------------------------
class ActionPlanResult(RootModel):
    root: List[ActionPlanActions]

def get_clean_action_plan_schema() -> Dict[str, Any]:
    # ดึง Schema โดยใช้ alias เพื่อให้ LLM เห็นเป็นตัวพิมพ์เล็กตามที่ตกลงกัน
    full_schema = ActionPlanResult.model_json_schema(by_alias=True)
    
    # ดึงโครงสร้างหลักของ item ใน array ออกมา
    # ป้องกันปัญหา $defs โดยการดึงเฉพาะ properties ของ ActionPlanActions
    if "$defs" in full_schema:
        # วิธีการที่สะอาดที่สุดคือระบุโครงสร้างแบบ Manual หรือใช้ Ref Resolution
        # แต่เพื่อความรวดเร็วและแม่นยำสำหรับ LLM เราจะใช้ Schema สั้นๆ:
        return {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "phase": {"type": "string"},
                    "goal": {"type": "string"},
                    "actions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "statement_id": {"type": "string"},
                                "failed_level": {"type": "integer"},
                                "recommendation": {"type": "string"},
                                "target_evidence_type": {"type": "string"},
                                "key_metric": {"type": "string"},
                                "steps": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "Step": {"type": "integer"},
                                            "Description": {"type": "string"},
                                            "Responsible": {"type": "string"},
                                            "Tools_Templates": {"type": "string"},
                                            "Verification_Outcome": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    return full_schema