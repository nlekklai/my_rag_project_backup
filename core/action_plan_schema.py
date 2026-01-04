from pydantic import BaseModel, Field, field_validator, RootModel, ConfigDict, ValidationError
from typing import List, Any, Dict, Optional, Union
import re
import logging

logger = logging.getLogger(__name__)

# -----------------------------
# 1. Step Detail: รายละเอียดขั้นตอนย่อย
# -----------------------------
class StepDetail(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="ignore"
    )

    # รองรับทั้ง "step" (จาก Prompt) และ "Step" (จาก JSON Output เดิม)
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
# 2. Action Item: รายการแผนงาน
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
        if isinstance(v, str):
            return [{"Step": 1, "Description": v.strip()}]
        if not isinstance(v, list):
            v = [v] if v else []
        
        normalized = []
        for i, item in enumerate(v):
            if isinstance(item, str):
                normalized.append({"Step": i + 1, "Description": item.strip()})
            elif isinstance(item, dict):
                # Map keys ให้ตรงกับ Alias ของ StepDetail
                mapping = {
                    "step": "Step", "description": "Description", 
                    "responsible": "Responsible", "tools_templates": "Tools_Templates", 
                    "verification_outcome": "Verification_Outcome"
                }
                norm_item = {mapping.get(k.lower().replace(" ", ""), k): val for k, val in item.items()}
                if "Step" not in norm_item: norm_item["Step"] = i + 1
                normalized.append(norm_item)
        return normalized

# -----------------------------
# 3. Phase: Roadmap Grouping
# -----------------------------
class ActionPlanActions(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="ignore"
    )

    phase: str = Field(..., alias="Phase")
    goal: str = Field(..., alias="Goal")
    actions: List[ActionItem] = Field(default_factory=list, alias="Actions")

# -----------------------------
# 4. Root Result: แก้ไขให้รับข้อมูลได้ทุกลักษณะ
# -----------------------------
class ActionPlanResult(RootModel):
    """
    RootModel สำหรับรับ List ของ ActionPlanActions
    """
    root: List[ActionPlanActions]

    @classmethod
    def validate_flexible(cls, data: Any) -> "ActionPlanResult":
        """
        ฟังก์ชันอัจฉริยะสำหรับแกะข้อมูล JSON จาก LLM 
        รองรับทั้ง:
        1. [ {...}, {...} ] (List ตรงๆ)
        2. { "root": [...] } (Dict ที่หุ้มด้วย root)
        3. { "Phase": [...] } (กรณีพ่นออกมาเป็น Dict ชั้นเดียว)
        """
        # กรณี 1: ถ้ามาเป็น Dictionary
        if isinstance(data, dict):
            # ถ้ามี key 'root' ให้ดึงเฉพาะข้างในมาตรวจ
            if "root" in data and isinstance(data["root"], list):
                return cls.model_validate(data["root"])
            # ถ้าไม่มี 'root' แต่เป็นข้อมูล Phase เดียวที่ถูกห่อมา
            if "phase" in [k.lower() for k in data.keys()]:
                return cls.model_validate([data])
        
        # กรณี 2: ถ้ามาเป็น List ตรงๆ (Best Case)
        if isinstance(data, list):
            return cls.model_validate(data)
            
        raise ValueError(f"Unsupported data format for ActionPlanResult: {type(data)}")

    def __iter__(self): return iter(self.root)
    def __getitem__(self, item): return self.root[item]
    def __len__(self): return len(self.root)

# -----------------------------
# 5. Helper สำหรับ Prompt
# -----------------------------
def get_clean_action_plan_schema() -> Dict[str, Any]:
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