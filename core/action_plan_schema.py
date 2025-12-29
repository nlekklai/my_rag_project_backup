# core/action_plan_schema.py

from pydantic import BaseModel, Field, field_validator, RootModel, ConfigDict
from typing import List, Any, Dict, Optional
import re
import json

# -----------------------------
# 1. Step Detail: ขั้นตอนย่อย
# -----------------------------
class StepDetail(BaseModel):
    # ปรับให้ใช้ alias เป็น snake_case เพื่อความสะดวกของ LLM
    Step: int = Field(..., alias="step", description="ลำดับที่ (1, 2, 3...)")
    Description: str = Field(..., alias="description", description="กิจกรรมที่ต้องทำ")
    Responsible: str = Field(..., alias="responsible", description="หน่วยงาน/ตำแหน่งที่รับผิดชอบ")
    Tools_Templates: str = Field(..., alias="tools_templates", description="ชื่อไฟล์ Template หรือระบบที่ใช้")
    Verification_Outcome: str = Field(..., alias="verification_outcome", description="ผลลัพธ์หรือหลักฐานที่ได้ (Evidence)")

    # อนุญาตให้ใช้ทั้งชื่อจริง (Step) และ alias (step) ในการสร้าง Object
    model_config = ConfigDict(populate_by_name=True)

    @field_validator("Step", mode="before")
    @classmethod
    def ensure_int(cls, v: Any) -> int:
        if isinstance(v, int): return v
        nums = re.findall(r'\d+', str(v))
        return int(nums[0]) if nums else 0

    @field_validator("Description", "Responsible", "Tools_Templates", "Verification_Outcome", mode="before")
    @classmethod
    def sanitize_text(cls, v: Any) -> str:
        if v is None: return ""
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
    """ดึง JSON Schema ที่ใช้ Alias (ตัวพิมพ์เล็ก) ทั้งหมด เพื่อส่งให้ LLM"""
    full_schema = ActionPlanResult.model_json_schema(by_alias=True)
    
    # หาก Pydantic เจน $defs มา ให้ดึงโครงสร้างแบบแบน (Flat) เพื่อให้ LLM ไม่งง
    if "$defs" in full_schema:
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
                                            "step": {"type": "integer"},
                                            "description": {"type": "string"},
                                            "responsible": {"type": "string"},
                                            "tools_templates": {"type": "string"},
                                            "verification_outcome": {"type": "string"}
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