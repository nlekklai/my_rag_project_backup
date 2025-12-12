# utils/enabler_keyword_map.py
# SE-AM Engine Multi-Enabler Keyword Boost Map
# รองรับทุก Enabler ใน PEA — ขยายได้ไม่จำกัด
# เวอร์ชัน Production Final — 12 ธันวาคม 2568

ENABLER_KEYWORD_MAP = {
    "KM": {
        "1.1": "วิสัยทัศน์ KM, นโยบาย KM, KM Policy, KM Master Plan, ผู้บริหารกำหนดทิศทาง KM",
        "1.2": "ผู้บริหารเป็นแบบอย่าง KM, Role Model KM, การมีส่วนร่วมของผู้บริหาร KM",
        "2.1": "แผน KM, KM Roadmap, การติดตาม KM, KPI KM, การประเมินผล KM",
        "2.2": "งบประมาณ KM, บุคลากร KM, ทีม KM, ระบบ KMS, เทคโนโลยีสารสนเทศ KM, โครงสร้างพื้นฐาน KM, การจัดสรรทรัพยากร KM"
    },
    "HR": {
        "1.1": "นโยบายพัฒนาบุคลากร, HR Strategy, Talent Management Policy, Competency Framework",
        "1.2": "ผู้บริหารเป็นแบบอย่างด้าน HR, HR Role Model, การสนับสนุนจากผู้บริหารด้านการพัฒนาคน",
        "2.1": "แผนพัฒนาบุคลากร, HR Plan, Succession Plan, Training Roadmap",
        "2.2": "งบอบรม, งบพัฒนาบุคลากร, LMS, Learning Management System, Trainer, HR Development Fund, กองทุนพัฒนาพนักงาน"
    },
    "IT": {
        "1.1": "นโยบายดิจิทัล, Digital Transformation Policy, IT Governance, CIO Strategy",
        "1.2": "ผู้บริหารขับเคลื่อนดิจิทัล, Digital Leadership, CIO เป็น Role Model",
        "2.1": "แผน IT, IT Roadmap, Digital Roadmap, Cyber Security Plan",
        "2.2": "งบไอที, Server, Cloud, Cybersecurity, Data Center, Digital Infrastructure, Network, ERP System"
    },
    "PM": {
        "1.1": "นโยบายบริหารโครงการ, Project Governance, PM Strategy, PMO Policy",
        "1.2": "ผู้บริหารสนับสนุนโครงการ, Project Sponsorship, การมีส่วนร่วมของผู้บริหารโครงการ",
        "2.1": "แผนบริหารโครงการ, Project Portfolio, PM Roadmap",
        "2.2": "งบโครงการ, Project Budget, PMO Resources, Project Management Tools, Microsoft Project, Jira"
    },
    "Finance": {
        "1.1": "นโยบายการเงิน, Financial Policy, Treasury Strategy, Risk Management Policy",
        "1.2": "ผู้บริหารการเงินเป็นแบบอย่าง, CFO Leadership, Financial Governance",
        "2.1": "แผนการเงิน, Budget Plan, Financial Roadmap",
        "2.2": "งบการเงิน, ERP, SAP, Oracle Financial, Cashflow Management, ระบบบัญชี"
    },
    "Safety": {
        "1.1": "นโยบายความปลอดภัย, Safety Policy, Zero Accident Goal, Safety Culture",
        "1.2": "ผู้บริหารเป็นแบบอย่างด้านความปลอดภัย, Safety Leadership, Safety Walk",
        "2.1": "แผนความปลอดภัย, Safety Plan, Risk Assessment Plan",
        "2.2": "งบความปลอดภัย, PPE, อุปกรณ์ป้องกันภัย, Safety Equipment, Safety Training Budget"
    },
    "Customer": {
        "1.1": "นโยบายลูกค้าเป็นศูนย์กลาง, Customer Centric Policy, Voice of Customer, Customer Experience Strategy",
        "1.2": "ผู้บริหารเน้นลูกค้า, Customer Focus Leadership",
        "2.1": "แผนบริการลูกค้า, Customer Journey Map, Service Blueprint",
        "2.2": "งบลูกค้า, CRM System, Call Center, Customer Experience Platform, NPS Tools"
    },
    "Innovation": {
        "1.1": "นโยบายนวัตกรรม, Innovation Policy, Open Innovation, Creative Culture",
        "1.2": "ผู้บริหารสนับสนุนนวัตกรรม, Innovation Champion, CEO เป็น Innovator",
        "2.1": "แผนนวัตกรรม, Innovation Roadmap, R&D Plan",
        "2.2": "งบนวัตกรรม, R&D Budget, Innovation Lab, Maker Space, Patent Fund, Startup Collaboration"
    },
    # เพิ่ม Enabler อื่น ๆ ได้ที่นี่ ไม่จำกัดจำนวน
}

# Optional: Default fallback สำหรับ Enabler ที่ยังไม่มีใน map
DEFAULT_KEYWORDS = {
    "2.2": "งบประมาณ, การเงิน, บุคลากร, ทรัพยากร, โครงสร้างพื้นฐาน, ระบบสนับสนุน, การจัดสรรทรัพยากร"
}