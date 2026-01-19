from pydantic import BaseModel, Field, field_validator, RootModel, ConfigDict, ValidationError, AliasChoices
from typing import List, Any, Dict, Optional, Union
import re
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# 1. StepDetail: รายละเอียดขั้นตอนปฏิบัติ (Tactical Level)
# ---------------------------------------------------------
class StepDetail(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="ignore"
    )

    # ใช้ AliasChoices เพื่อให้รับได้ทั้ง "Step", "step", "STEP"
    step: int = Field(..., validation_alias=AliasChoices("Step", "step", "Step_No"))
    description: str = Field(..., validation_alias=AliasChoices("Description", "description", "desc"))
    responsible: str = Field("หน่วยงานที่เกี่ยวข้อง", validation_alias=AliasChoices("Responsible", "responsible", "owner"))
    tools_templates: str = Field("-", validation_alias=AliasChoices("Tools_Templates", "tools_templates", "tools"))
    verification_outcome: str = Field("หลักฐานเชิงประจักษ์ตามแผน", validation_alias=AliasChoices("Verification_Outcome", "verification_outcome", "outcome"))

    @field_validator("step", mode="before")
    @classmethod
    def parse_step_number(cls, v: Any) -> int:
        if isinstance(v, (int, float)): return int(v)
        found = re.search(r'\d+', str(v))
        return int(found.group()) if found else 1

    @field_validator("description", "responsible", "tools_templates", "verification_outcome", mode="before")
    @classmethod
    def sanitize_text(cls, v: Any) -> str:
        if v is None: return "-"
        text = str(v).strip()
        return re.sub(r'[*_#]', '', text) or "-"

# ---------------------------------------------------------
# 2. ActionItem: แผนงานรายข้อ (Operational Level)
# ---------------------------------------------------------
class ActionItem(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="ignore"
    )

    # แก้จุดตายที่ Error ใน Log: ดักจับทั้ง Statement_ID, statement_id, statementid
    statement_id: str = Field(..., validation_alias=AliasChoices("Statement_ID", "statement_id", "statementid"))
    failed_level: int = Field(..., validation_alias=AliasChoices("Failed_Level", "failed_level", "failedlevel"))
    recommendation: str = Field("ดำเนินการพัฒนาตามเกณฑ์มาตรฐาน", validation_alias=AliasChoices("Recommendation", "recommendation"))
    target_evidence_type: str = Field("Evidence Package / Report", validation_alias=AliasChoices("Target_Evidence_Type", "target_evidence_type"))
    key_metric: str = Field("ความสำเร็จตามตัวชี้วัดที่กำหนด (100%)", validation_alias=AliasChoices("Key_Metric", "key_metric"))
    steps: List[StepDetail] = Field(default_factory=list, validation_alias=AliasChoices("Steps", "steps"))

    @field_validator("failed_level", mode="before")
    @classmethod
    def parse_failed_level(cls, v: Any) -> int:
        if isinstance(v, (int, float)): return int(v)
        # กรณี AI พ่น "Level 1" หรือ "L1"
        found = re.search(r'\d+', str(v))
        return int(found.group()) if found else 0

    @field_validator("steps", mode="before")
    @classmethod
    def handle_variadic_steps(cls, v: Any) -> List[Dict]:
        if not v: return []
        if isinstance(v, str): 
            return [{"step": 1, "description": v.strip()}]
        
        raw_list = v if isinstance(v, list) else [v]
        normalized = []
        for i, item in enumerate(raw_list):
            if isinstance(item, str):
                normalized.append({"step": i + 1, "description": item})
            elif isinstance(item, dict):
                # ทำให้ keys เป็นตัวเล็กและไม่มี underscore เพื่อให้ตรงกับ Model
                new_item = {k.lower().replace("_", ""): val for k, val in item.items()}
                normalized.append(new_item)
        return normalized

# ---------------------------------------------------------
# 3. ActionPlanActions: การจัดกลุ่มเฟสงาน (Strategic Level)
# ---------------------------------------------------------
class ActionPlanActions(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    phase: str = Field(..., validation_alias=AliasChoices("Phase", "phase"))
    goal: str = Field(..., validation_alias=AliasChoices("Goal", "goal"))
    actions: List[ActionItem] = Field(default_factory=list, validation_alias=AliasChoices("Actions", "actions"))

# ---------------------------------------------------------
# 4. ActionPlanResult: ตัวเชื่อมต่อระดับ Root
# ---------------------------------------------------------
class ActionPlanResult(RootModel):
    root: List[ActionPlanActions]

    @classmethod
    def validate_flexible(cls, data: Any) -> "ActionPlanResult":
        """
        ปรับปรุง Logic การแกะ List ให้ทนทานต่อ Dict ที่ AI มโน Key ขึ้นมาหุ้ม
        """
        if not data: return cls.model_validate([])

        if isinstance(data, list):
            return cls.model_validate(data)

        if isinstance(data, dict):
            # ตรวจสอบหา List ของ Actions ภายใน Dict
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 0:
                    # ถ้าเจอ List ที่มีสมาชิกตัวแรกมีแววว่าเป็น Phase
                    if isinstance(value[0], dict) and any(k.lower() == 'phase' for k in value[0].keys()):
                        return cls.model_validate(value)
            
            # ถ้าเป็น Dict ชั้นเดียวที่มีข้อมูลเฟสงาน
            if any(k.lower() == 'phase' for k in data.keys()):
                return cls.model_validate([data])

        return cls.model_validate(data)

# ---------------------------------------------------------
# 5. Helper: ปรับ Schema ให้ LLM ไม่สับสนเรื่อง Case
# ---------------------------------------------------------
def get_clean_action_plan_schema() -> Dict[str, Any]:
    return {
        "format": "array",
        "description": "MUST be a valid JSON array of strategic phases.",
        "example_item": {
            "phase": "Foundation Build",
            "goal": "Establish KM Policy",
            "actions": [
                {
                    "statement_id": "1.1",
                    "failed_level": 1,
                    "recommendation": "Draft KM Policy for Board approval",
                    "steps": [
                        {"step": 1, "description": "Gather requirements", "responsible": "KM Unit"}
                    ]
                }
            ]
        }
    }