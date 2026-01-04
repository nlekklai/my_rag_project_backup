# core/action_plan_schema.py

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Any, Dict
import re

# -----------------------------
# 1. Step Detail
# -----------------------------
class StepDetail(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid"
    )

    step: int = Field(..., alias="Step")
    description: str = Field(..., alias="Description")
    responsible: str = Field("หน่วยงานที่เกี่ยวข้อง", alias="Responsible")
    tools_templates: str = Field("-", alias="Tools_Templates")
    verification_outcome: str = Field("หลักฐานการดำเนินงาน", alias="Verification_Outcome")

    @field_validator("step", mode="before")
    @classmethod
    def ensure_int(cls, v: Any) -> int:
        if isinstance(v, int):
            return v
        try:
            nums = re.findall(r'\d+', str(v))
            return int(nums[0]) if nums else 1
        except:
            return 1

    @field_validator("description", "responsible", "tools_templates", "verification_outcome", mode="before")
    @classmethod
    def ensure_string(cls, v: Any) -> str:
        if isinstance(v, str):
            return v.strip() or "-"
        return str(v).strip() or "-"

    @field_validator("description", mode="before")
    @classmethod
    def extract_from_dict(cls, v: Any) -> str:
        if isinstance(v, dict):
            return (
                v.get("Description") or
                v.get("description") or
                v.get("Desc") or
                v.get("desc") or
                ""
            )
        return str(v)

# -----------------------------
# 2. Action Item
# -----------------------------
class ActionItem(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid"
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
            return [{"step": 1, "description": v.strip()}]

        if not isinstance(v, list):
            v = [v] if v else []

        normalized = []
        for i, item in enumerate(v):
            if isinstance(item, str):
                normalized.append({"step": i + 1, "description": item.strip()})
            elif isinstance(item, dict):
                norm_item = {}
                for k, val in item.items():
                    k_low = k.lower().replace(" ", "").replace("_", "")
                    mapping = {
                        "step": "step",
                        "description": "description",
                        "responsible": "responsible",
                        "toolstemplates": "tools_templates",
                        "verificationoutcome": "verification_outcome"
                    }
                    target_key = mapping.get(k_low, k_low)
                    norm_item[target_key] = val

                norm_item.setdefault("step", i + 1)
                norm_item.setdefault("description", "")
                normalized.append(norm_item)
            else:
                normalized.append({"step": i + 1, "description": str(item)})
        return normalized

# -----------------------------
# 3. Phase — ใช้ชื่อ ActionPlanActions เพื่อให้ไฟล์อื่น import ได้
# -----------------------------
class ActionPlanActions(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid"
    )

    phase: str = Field(..., alias="Phase")
    goal: str = Field(..., alias="Goal")
    actions: List[ActionItem] = Field(default_factory=list, alias="Actions")

# -----------------------------
# 4. Root Result — ใช้ BaseModel แทน RootModel
# -----------------------------
class ActionPlanResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True
    )

    root: List[ActionPlanActions]

    # ทำให้ serialize ออกมาเป็น list โดยตรง (เหมือน RootModel)
    def model_dump(self, **kwargs) -> List[Dict]:
        return [phase.model_dump(**kwargs) for phase in self.root]

    def model_dump_json(self, **kwargs) -> str:
        import json
        return json.dumps(self.model_dump(**kwargs), ensure_ascii=False, indent=2, **kwargs)

    # ทำให้ใช้งานเหมือน list ได้
    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, item):
        return self.root[item]

    def __len__(self):
        return len(self.root)

# -----------------------------
# 5. Helper: ดึง Schema แบบ Flat สำหรับ LLM
# -----------------------------
def get_clean_action_plan_schema() -> Dict[str, Any]:
    return {
        "type": "array",
        "description": "Roadmap แบ่งเป็นหลาย Phase",
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
                                    },
                                    "required": ["step", "description"]
                                }
                            }
                        },
                        "required": ["statement_id", "failed_level", "recommendation", "steps"]
                    }
                }
            },
            "required": ["phase", "goal", "actions"]
        }
    }