# -*- coding: utf-8 -*-

"""
ISO 30401:2018 Mapping Logic for SE-AM 2024 (Revised)
This mapping links KM, IM, and HCM subcriteria to ISO clauses.
"""

ISO_30401_MAPPING = {
    "Clause 4.4: KM System (KMS)": {
        "primary_source": {"enabler": "KM", "sub": "4.1"},  # กระบวนการ KM ที่เป็นระบบ
        "support_source": {"enabler": "IM", "sub": "1.2"},  # โครงสร้างสารสนเทศ
        "iso_requirement": "องค์กรต้องจัดตั้งและรักษาไว้ซึ่งระบบการจัดการความรู้",
        "key_evidence": ["KMS Architecture", "Workflow KM", "Knowledge Maps"]
    },
    
    "Clause 5.1: Leadership & Commitment": {
        "primary_source": {"enabler": "KM", "sub": "1.1"},  # วิสัยทัศน์/นโยบาย KM
        "support_source": {"enabler": "HCM", "sub": "1.1"}, # ยุทธศาสตร์ทุนมนุษย์
        "iso_requirement": "ผู้นำต้องแสดงความมุ่งมั่นต่อนโยบายและการสื่อสาร",
        "key_evidence": ["Board Minutes", "Policy Statement", "HR Strategy Alignment"]
    },

    "Clause 6.1: Actions to address risks/opportunities": {
        "primary_source": {"enabler": "KM", "sub": "5.2"},  # Knowledge Risk
        "support_source": {"enabler": "IM", "sub": "3.1"},  # บริหารความเสี่ยงข้อมูล
        "iso_requirement": "ต้องระบุและจัดการความเสี่ยงที่ส่งผลต่อความรู้ขององค์กร",
        "key_evidence": ["Knowledge Risk Register", "Critical Steps Analysis", "ERM Integration"]
    },

    "Clause 7.2: Competence": {
        "primary_source": {"enabler": "HCM", "sub": "3.1"}, # IDP / Reskill / Upskill
        "support_source": {"enabler": "KM", "sub": "3.3"},  # ทีมงาน KM
        "iso_requirement": "พนักงานต้องมีทักษะที่จำเป็นในการจัดการความรู้",
        "key_evidence": ["Skill Matrix", "Individual Development Plans", "Training Records"]
    },

    "Clause 7.5: Documented information": {
        "primary_source": {"enabler": "IM", "sub": "2.1"},  # การจัดการวงจรชีวิตข้อมูล
        "support_source": {"enabler": "KM", "sub": "4.1"},  # การจัดเก็บ/เข้าถึงความรู้
        "iso_requirement": "ต้องมีการควบคุมการสร้าง การปรับปรุง และการใช้ข้อมูลสารสนเทศ",
        "key_evidence": ["Version Control", "Access Control List (ACL)", "DMS Workflow"]
    },

    "Clause 8: Operation": {
        "primary_source": {"enabler": "KM", "sub": "5.1"},  # การทำงานโดยใช้ความรู้เป็นฐาน
        "support_source": {"enabler": "IM", "sub": "3.1"},  # บูรณาการข้อมูล
        "iso_requirement": "การนำกระบวนการ KM ไปใช้จริงในการทำงานหน้างาน",
        "key_evidence": ["Best Practices Applied", "Standard Operating Procedures (SOP)", "Lessons Learned"]
    },

    "Clause 9: Performance evaluation": {
        "primary_source": {"enabler": "KM", "sub": "2.1"},  # ติดตามประเมินผล KM
        "support_source": {"enabler": "HCM", "sub": "2.1"}, # ระบบประเมิน PMS
        "iso_requirement": "ต้องมีการวัดผลสัมฤทธิ์ของระบบ KM อย่างสม่ำเสมอ",
        "key_evidence": ["KM KPI Reports", "Employee Engagement Surveys", "Performance Reviews"]
    }
}

# Helper function สำหรับดึงข้อมูล Mapping
def get_iso_gap_points(clause_key):
    return ISO_30401_MAPPING.get(clause_key, {}).get("key_evidence", [])