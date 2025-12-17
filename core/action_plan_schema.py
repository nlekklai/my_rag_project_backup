# -*- coding: utf-8 -*-
# core/action_plan_schema.py
# SE-AM Action Plan Pydantic Schema v2.0 — Full & Fixed (2025-12-17)
# FIX: รองรับ LLM ที่ตอบ key ตัวพิมพ์เล็ก (phase, goal, actions)
# FIX: Clean schema without $defs leakage

from pydantic import BaseModel, Field, field_validator, RootModel
from typing import List, Any, Dict
import re
import json

# -----------------------------
# 1. Step Detail: ขั้นตอนย่อย
# -----------------------------
class StepDetail(BaseModel):
    Step: str = Field(..., description="ลำดับที่ (1, 2, 3...)")
    Description: str = Field(..., description="กิจกรรมที่ต้องทำ")
    Responsible: str = Field(..., description="หน่วยงาน/ตำแหน่งที่รับผิดชอบ")
    Tools_Templates: str = Field(..., description="ชื่อไฟล์ Template หรือระบบที่ใช้")
    Verification_Outcome: str = Field(..., description="ผลลัพธ์หรือหลักฐานที่ได้ (Evidence)")

    @field_validator("*", mode="before")
    @classmethod
    def sanitize(cls, v: Any) -> str:
        if v is None:
            return ""
        v = re.sub(r'[\n\r\t\u200b\u200c\u200d\uFEFF]+', ' ', str(v))
        return v.strip()

# -----------------------------
# 2. Action Item: รายการแผนงาน
# -----------------------------
class ActionItem(BaseModel):
    Statement_ID: str = Field(..., alias="statement_id", description="ID ของเกณฑ์ เช่น 1.2.L4")
    Failed_Level: int = Field(..., alias="failed_level", description="ระดับที่ต้องการแก้ไข (1-5)")
    Recommendation: str = Field(..., alias="recommendation", description="ข้อแนะนำเชิงกลยุทธ์")
    Target_Evidence_Type: str = Field(..., alias="target_evidence_type", description="ประเภทหลักฐาน (P/D/C/A)")
    Key_Metric: str = Field(..., alias="key_metric", description="ตัวชี้วัดความสำเร็จ")
    Steps: List[StepDetail] = Field(default_factory=list, alias="steps")

    model_config = {"populate_by_name": True}  # สำคัญ! อนุญาตให้ใช้ alias (ตัวพิมพ์เล็ก)

    @field_validator("Statement_ID", mode="before")
    @classmethod
    def normalize_id(cls, v: Any) -> str:
        if not v:
            return "L0_S0"
        v = str(v).strip().upper().replace('-', '_').replace(' ', '_')
        return v if v.startswith('L') else f"L{v}"

    @field_validator("Steps", mode="before")
    @classmethod
    def ensure_list(cls, v: Any) -> List:
        if isinstance(v, list):
            return v
        if isinstance(v, dict):
            return [v]
        return []

# -----------------------------
# 3. ActionPlanActions: ระยะ (Phase)
# -----------------------------
class ActionPlanActions(BaseModel):
    Phase: str = Field(..., alias="phase", description="ชื่อระยะ เช่น Phase 1: Quick Wins")
    Goal: str = Field(..., alias="goal", description="เป้าหมายของระยะนี้")
    Actions: List[ActionItem] = Field(default_factory=list, alias="actions")

    model_config = {"populate_by_name": True}  # สำคัญ! รองรับ key ตัวพิมพ์เล็กจาก LLM

# -----------------------------
# 4. Root Model สำหรับ JSON Array
# -----------------------------
class ActionPlanResult(RootModel):
    root: List[ActionPlanActions] = Field(..., description="รายการ Phase ทั้งหมดของ Action Plan")

    def to_json(self) -> str:
        return json.dumps(self.model_dump(by_alias=True, exclude_none=True), ensure_ascii=False, indent=2)

# =================================================================
# HELPER: Clean JSON Schema (ป้องกัน $defs leakage)
# =================================================================
def get_clean_action_plan_schema() -> Dict[str, Any]:
    """
    สร้าง JSON Schema ที่สะอาดสำหรับส่งให้ LLM
    """
    try:
        full_schema = ActionPlanResult.model_json_schema(by_alias=True)
        
        clean_schema = {
            "type": "array",
            "description": "รายการ Phase ของ Action Plan",
            "items": full_schema.get("items", {})
        }
        
        # ลบคีย์ที่ทำให้ LLM สับสน
        for key in ["$defs", "$schema", "title", "definitions"]:
            clean_schema.pop(key, None)
            if "items" in clean_schema:
                clean_schema["items"].pop(key, None)
        
        return clean_schema
    except Exception as e:
        print(f"Schema generation error: {e}")
        return {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "phase": {"type": "string"},
                    "goal": {"type": "string"},
                    "actions": {"type": "array"}
                }
            }
        }