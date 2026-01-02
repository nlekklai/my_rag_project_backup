# core/action_plan_schema.py

from pydantic import BaseModel, Field, field_validator, RootModel, ConfigDict
from typing import List, Any, Dict, Optional
import re
import json

# -----------------------------
# 1. Step Detail: ขั้นตอนย่อย
# -----------------------------
class StepDetail(BaseModel):
    Step: int = Field(default=1, alias="step")
    Description: str = Field(default="", alias="description")
    Responsible: str = Field(default="หน่วยงานที่เกี่ยวข้อง", alias="responsible")
    Tools_Templates: str = Field(default="-", alias="tools_templates")
    Verification_Outcome: str = Field(default="หลักฐานการดำเนินงาน", alias="verification_outcome")

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("Step", mode="before")
    @classmethod
    def ensure_int(cls, v: Any) -> int:
        if isinstance(v, int): return v
        nums = re.findall(r'\d+', str(v))
        return int(nums[0]) if nums else 1

    # ✅ เพิ่มจุดนี้: ถ้า AI ส่งแค่ String มาตรงๆ ให้แปลงเป็น Description ของ Step นั้น
    @field_validator("Description", mode="before")
    @classmethod
    def wrap_string_to_desc(cls, v: Any) -> str:
        if isinstance(v, dict):
            return v.get("Description") or v.get("description") or ""
        return str(v)

# -----------------------------
# 2. Action Item: รายการแผนงาน
# -----------------------------
class ActionItem(BaseModel):
    Statement_ID: str = Field(..., alias="statement_id")
    Failed_Level: int = Field(..., alias="failed_level")
    Recommendation: str = Field(default="ปรับปรุงตามเกณฑ์มาตรฐาน", alias="recommendation")
    Target_Evidence_Type: str = Field(default="เอกสารประกอบการดำเนินงาน", alias="target_evidence_type")
    Key_Metric: str = Field(default="ระดับความสำเร็จตามแผน", alias="key_metric")
    Steps: List[StepDetail] = Field(default_factory=list, alias="steps")

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("Steps", mode="before")
    @classmethod
    def ensure_list_of_objects(cls, v: Any) -> Any:
        # กรณี 1: มาเป็น String ก้อนเดียว (เช่น "ทำ A, B, C")
        if isinstance(v, str):
            return [{"step": 1, "description": v}]
        
        # กรณี 2: มาเป็น List ของ String (เช่น ["ขั้นตอน 1", "ขั้นตอน 2"]) **<-- จุดที่คุณเจอ Error**
        if isinstance(v, list):
            new_steps = []
            for i, item in enumerate(v):
                if isinstance(item, str):
                    new_steps.append({"step": i + 1, "description": item})
                else:
                    new_steps.append(item)
            return new_steps
            
        return v

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