# core/action_plan_schema.py

from pydantic import BaseModel, Field, field_validator, RootModel, ConfigDict
from typing import List, Any, Dict, Optional
import re

# -----------------------------
# 1. Step Detail: รายละเอียดขั้นตอนย่อย
# -----------------------------
class StepDetail(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True, # สำคัญ: ยอมรับทั้ง "step" และ "Step"
        extra="ignore"         # ป้องกัน Error หาก LLM พ่นฟิลด์ขยะมา
    )

    # ใช้ alias เป็นตัวใหญ่ตามมาตรฐานเดิม แต่ชื่อตัวแปรเป็นตัวเล็กตาม Prompt ใหม่
    step: int = Field(..., alias="Step")
    description: str = Field(..., alias="Description")
    responsible: str = Field("หน่วยงานที่เกี่ยวข้อง", alias="Responsible")
    tools_templates: str = Field("-", alias="Tools_Templates")
    verification_outcome: str = Field("หลักฐานการดำเนินงาน", alias="Verification_Outcome")

    @field_validator("step", mode="before")
    @classmethod
    def ensure_int(cls, v: Any) -> int:
        if isinstance(v, int): return v
        try:
            nums = re.findall(r'\d+', str(v))
            return int(nums[0]) if nums else 1
        except: return 1

    @field_validator("description", "responsible", "tools_templates", "verification_outcome", mode="before")
    @classmethod
    def ensure_string(cls, v: Any) -> str:
        if v is None: return "-"
        return str(v).strip() or "-"

# -----------------------------
# 2. Action Item: รายการแผนงาน (Statement level)
# -----------------------------
class ActionItem(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="ignore"
    )

    statement_id: str = Field(..., alias="Statement_ID")
    failed_level: int = Field(..., alias="Failed_Level")
    recommendation: str = Field("ปรับปรุงตามเกณฑ์มาตรฐาน", alias="Recommendation")
    target_evidence_type: str = Field("เอกสารประกอบการดำเนินงาน", alias="Target_Evidence_Type")
    key_metric: str = Field("ระดับความสำเร็จตามแผน", alias="Key_Metric")
    steps: List[StepDetail] = Field(default_factory=list, alias="Steps")

    @field_validator("steps", mode="before")
    @classmethod
    def normalize_steps(cls, v: Any) -> List[Dict]:
        """ยืดหยุ่นต่อรูปแบบ Steps ที่ LLM ส่งมา"""
        if isinstance(v, str):
            return [{"Step": 1, "Description": v.strip()}]
        
        if not isinstance(v, list):
            v = [v] if v else []

        normalized = []
        for i, item in enumerate(v):
            if isinstance(item, str):
                normalized.append({"Step": i + 1, "Description": item.strip()})
            elif isinstance(item, dict):
                # แมป key ตัวเล็กจาก Prompt ให้เข้ากับ Alias ตัวใหญ่ของ StepDetail
                mapping = {
                    "step": "Step", 
                    "description": "Description", 
                    "responsible": "Responsible", 
                    "tools_templates": "Tools_Templates", 
                    "verification_outcome": "Verification_Outcome"
                }
                norm_item = {mapping.get(k.lower().replace(" ", ""), k): val for k, val in item.items()}
                if "Step" not in norm_item: norm_item["Step"] = i + 1
                normalized.append(norm_item)
        return normalized

# -----------------------------
# 3. Phase: กลุ่มของ Action แบ่งตามช่วงเวลา (Roadmap)
# -----------------------------
class ActionPlanActions(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="ignore"
    )

    # เพิ่ม Alias เพื่อรองรับ Prompt ใหม่ที่ใช้ตัวเล็ก (phase) และของเก่า (Phase)
    phase: str = Field(..., alias="Phase")
    goal: str = Field(..., alias="Goal")
    actions: List[ActionItem] = Field(default_factory=list, alias="Actions")

# -----------------------------
# 4. Root Result: ตัวรับข้อมูลหลัก (ฉบับแก้ไข Error)
# -----------------------------
class ActionPlanResult(RootModel):
    """
    ใช้ RootModel เพื่อรับ JSON Array [{}, {}] จาก LLM ได้โดยตรง
    หมายเหตุ: RootModel ใน Pydantic V2 ไม่รองรับ model_config extra
    """
    root: List[ActionPlanActions]

    # ฟังก์ชันเสริมยังคงไว้ได้
    def __iter__(self): return iter(self.root)
    def __getitem__(self, item): return self.root[item]
    def __len__(self): return len(self.root)

# -----------------------------
# 5. Helper: สำหรับ Prompt Engineering
# -----------------------------
def get_clean_action_plan_schema() -> Dict[str, Any]:
    """JSON Schema แบบ Flat สำหรับใส่ใน Prompt"""
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