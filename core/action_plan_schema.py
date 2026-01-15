from pydantic import BaseModel, Field, field_validator, RootModel, ConfigDict, ValidationError
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

    step: int = Field(..., alias="Step")
    description: str = Field(..., alias="Description")
    responsible: str = Field("หน่วยงานที่เกี่ยวข้อง", alias="Responsible")
    tools_templates: str = Field("-", alias="Tools_Templates")
    verification_outcome: str = Field("หลักฐานเชิงประจักษ์ตามแผน", alias="Verification_Outcome")

    @field_validator("step", mode="before")
    @classmethod
    def parse_step_number(cls, v: Any) -> int:
        if isinstance(v, (int, float)): return int(v)
        # ดึงตัวเลขแรกที่เจอ เช่น "Step 01" -> 1
        found = re.search(r'\d+', str(v))
        return int(found.group()) if found else 1

    @field_validator("description", "responsible", "tools_templates", "verification_outcome", mode="before")
    @classmethod
    def sanitize_text(cls, v: Any) -> str:
        if v is None: return "-"
        text = str(v).strip()
        # ลบ Markdown characters ที่อาจหลงมา เช่น "**" หรือ "_"
        return re.sub(r'[*_#]', '', text) or "-"

# ---------------------------------------------------------
# 2. ActionItem: แผนงานรายข้อ (Operational Level)
# ---------------------------------------------------------
class ActionItem(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="ignore"
    )

    statement_id: str = Field(..., alias="Statement_ID")
    failed_level: int = Field(..., alias="Failed_Level")
    recommendation: str = Field("ดำเนินการพัฒนาตามเกณฑ์มาตรฐาน", alias="Recommendation")
    target_evidence_type: str = Field("Evidence Package / Report", alias="Target_Evidence_Type")
    key_metric: str = Field("ความสำเร็จตามตัวชี้วัดที่กำหนด (100%)", alias="Key_Metric")
    steps: List[StepDetail] = Field(default_factory=list, alias="Steps")

    @field_validator("steps", mode="before")
    @classmethod
    def handle_variadic_steps(cls, v: Any) -> List[Dict]:
        if not v: return []
        if isinstance(v, str): # กรณี AI พ่นมาเป็นประโยคเดียว
            return [{"Step": 1, "Description": v.strip()}]
        
        raw_list = v if isinstance(v, list) else [v]
        normalized = []
        for i, item in enumerate(raw_list):
            if isinstance(item, str):
                normalized.append({"Step": i + 1, "Description": item})
            elif isinstance(item, dict):
                # เทคนิค Case-Insensitive Mapping
                mapping = {
                    "step": "Step", "description": "Description", 
                    "responsible": "Responsible", "tools_templates": "Tools_Templates", 
                    "verification_outcome": "Verification_Outcome",
                    "tools": "Tools_Templates", "outcome": "Verification_Outcome"
                }
                new_item = {mapping.get(k.lower().replace("_", ""), k): val for k, val in item.items()}
                if "Step" not in new_item: new_item["Step"] = i + 1
                normalized.append(new_item)
        return normalized

# ---------------------------------------------------------
# 3. ActionPlanActions: การจัดกลุ่มเฟสงาน (Strategic Level)
# ---------------------------------------------------------
class ActionPlanActions(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    phase: str = Field(..., alias="Phase")
    goal: str = Field(..., alias="Goal")
    actions: List[ActionItem] = Field(default_factory=list, alias="Actions")

# ---------------------------------------------------------
# 4. ActionPlanResult: ตัวเชื่อมต่อระดับ Root
# ---------------------------------------------------------
class ActionPlanResult(RootModel):
    root: List[ActionPlanActions]

    @classmethod
    def validate_flexible(cls, data: Any) -> "ActionPlanResult":
        """
        ระบบอัจฉริยะสำหรับกู้คืนข้อมูล JSON จาก LLM ในทุกรูปแบบ
        """
        if not data: return cls.model_validate([])

        # เตรียมข้อมูลเบื้องต้น (Normalize keys ให้เป็นตัวเล็กเพื่อการเช็ค)
        if isinstance(data, dict):
            lowered_data = {k.lower(): v for k, v in data.items()}
            
            # กรณี 1: ข้อมูลหุ้มด้วย Key มาตรฐาน
            for key in ["root", "phases", "actionplan", "roadmap"]:
                if key in lowered_data and isinstance(lowered_data[key], list):
                    return cls.model_validate(lowered_data[key])
            
            # กรณี 2: ข้อมูลเป็นเฟสเดียวแต่ถูกส่งมาเป็น Dict ชั้นเดียว
            if "phase" in lowered_data:
                return cls.model_validate([data])

        # กรณี 3: ข้อมูลเป็น List ตรงๆ (Best Practice)
        if isinstance(data, list):
            return cls.model_validate(data)
            
        return cls.model_validate([data])

    def __iter__(self): return iter(self.root)
    def __len__(self): return len(self.root)
    def __getitem__(self, idx): return self.root[idx]

# ---------------------------------------------------------
# 5. Helper: สำหรับดึง Schema ไปใส่ใน Prompt
# ---------------------------------------------------------
def get_clean_action_plan_schema() -> Dict[str, Any]:
    """ คืนค่า Schema ที่ Clean ที่สุดเพื่อให้ LLM เข้าใจง่าย """
    return {
        "format": "array",
        "items_structure": {
            "phase": "string (ชื่อเฟสเชิงกลยุทธ์)",
            "goal": "string (เป้าหมาย)",
            "actions": [
                {
                    "statement_id": "string",
                    "failed_level": "integer",
                    "recommendation": "string",
                    "target_evidence_type": "string",
                    "steps": [
                        {"step": "int", "description": "string", "responsible": "string"}
                    ]
                }
            ]
        }
    }