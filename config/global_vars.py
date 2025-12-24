# config/global_vars.py
import os
import uuid
from typing import List, Dict, Set, Final

# ================================================================
# Project & Namespace
# ================================================================
PROJECT_NAMESPACE_UUID: Final[uuid.UUID] = uuid.UUID(
    "f77c38c0-f213-4318-ae38-e69c73e97022"
)

PROJECT_ROOT: Final[str] = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_STORE_ROOT: Final[str] = os.path.join(PROJECT_ROOT, "data_store")

# ================================================================
# Tenant & Default Configuration
# ================================================================
DEFAULT_TENANT: Final[str] = "pea"
DEFAULT_YEAR: Final[int] = 2568
DEFAULT_ENABLER: Final[str] = "KM"

# ================================================================
# Ollama / LLM Request Control
# ================================================================
OLLAMA_REQUEST_TIMEOUT: Final[int] = 300  # seconds
OLLAMA_MAX_RETRIES: Final[int] = 3

# ================================================================
# Run Mode & LLM Configuration
# ================================================================
RAG_RUN_MODE: Final[str] = os.environ.get("RAG_RUN_MODE", "LOCAL_OLLAMA")

# เลือก Model Name ตามโหมด
if RAG_RUN_MODE == "CLOUD":
    DEFAULT_LLM_MODEL_NAME: Final[str] = os.environ.get("NVIDIA_MODEL_NAME", "meta/llama-3.1-70b-instruct")
    LLM_CONTEXT_WINDOW: Final[int] = int(os.environ.get("LLM_CONTEXT_WINDOW", "32768"))
else:
    DEFAULT_LLM_MODEL_NAME: Final[str] = "llama3:8b"
    LLM_CONTEXT_WINDOW: Final[int] = 8192

LLM_TEMPERATURE: Final[float] = 0.0

# ================================================================
# Embedding & Reranker Models
# ================================================================
EMBEDDING_MODEL_NAME: Final[str] = "BAAI/bge-m3"
RERANKER_MODEL_NAME: Final[str] = "BAAI/bge-reranker-base"

# ================================================================
# Hybrid Search Configuration
# ================================================================
USE_HYBRID_SEARCH: Final[bool] = True
HYBRID_VECTOR_WEIGHT: Final[float] = 0.7
HYBRID_BM25_WEIGHT: Final[float] = 0.3

# ================================================================
# Retrieval & Ranking Parameters
# ================================================================
INITIAL_TOP_K: Final[int] = 75
FINAL_K_RERANKED: Final[int] = 15
FINAL_K_NON_RERANKED: Final[int] = 7

RERANK_THRESHOLD: Final[float] = 0.5
MIN_RETRY_SCORE: Final[float] = 0.50
MAX_RETRIEVAL_ATTEMPTS: Final[int] = 3

MIN_RERANK_SCORE_TO_KEEP: Final[float] = 0.10
MIN_RELEVANCE_THRESHOLD: Final[float] = 0.3

CRITICAL_CA_THRESHOLD: Final[float] = 0.65

# ================================================================
# Hard Fail & Context Control Flags
# ================================================================
ENABLE_HARD_FAIL_LOGIC: Final[bool] = False
ENABLE_CONTEXTUAL_RULE_OVERRIDE: Final[bool] = True

MAX_EVI_STR_CAP: Final[float] = 10.0
CONTEXT_CAP_L3_PLUS: Final[int] = 60000

# ================================================================
# Chunking Configuration
# ================================================================
CHUNK_SIZE: Final[int] = 1500
CHUNK_OVERLAP: Final[int] = 250
STANDARD_K: Final[int] = 5

QUERY_INITIAL_K: Final[int] = 20
QUERY_FINAL_K: Final[int] = 5

# ================================================================
# Priority & Parallel Processing
# ================================================================
LIMIT_CHUNKS_PER_PRIORITY_DOC: Final[int] = 5
PRIORITY_CHUNK_LIMIT: Final[int] = 30
MAX_PARALLEL_WORKERS: Final[int] = 2

# ================================================================
# Logging & Context Control
# ================================================================
IS_LOG_L3_CONTEXT: Final[bool] = True
MAX_EVAL_CONTEXT_LENGTH: Final[int] = 3000

# ================================================================
# Supported File & Document Types
# ================================================================
SUPPORTED_TYPES: Final[List[str]] = [
    ".pdf", ".docx", ".txt", ".xlsx", ".pptx", ".md", ".csv",
    ".jpg", ".jpeg", ".png",
]

SUPPORTED_DOC_TYPES: Final[List[str]] = [
    "document", "policy", "report", "statement", "evidence",
    "feedback", "faq", "seam",
]

EVIDENCE_DOC_TYPES: Final[str] = "evidence"
DEFAULT_DOC_TYPES: Final[str] = "document"

# ================================================================
# Enabler & Assessment Constants
# ================================================================
SUPPORTED_ENABLERS: Final[List[str]] = [
    "CG", "SP", "RM&IC", "SCM", "DT", "HCM", "KM", "IM", "IA",
]

MAX_LEVEL: Final[int] = 5
INITIAL_LEVEL: Final[int] = 1

# ================================================================
# SE-AM Enabler Mapping
# ================================================================
SEAM_ENABLER_MAP: Final[Dict[str, str]] = {
    "CG": "1 การกำกับดูแลที่ดีและการนำองค์กร",
    "SP": "2 การวางแผนเชิงยุทธศาสตร์",
    "RM&IC": "3 การบริหารความเสี่ยงและการควบคุมภายใน",
    "SCM": "4 การมุ่งเน้นผู้มีส่วนได้ส่วนเสีย และลูกค้า",
    "DT": "5 การพัฒนาเทคโนโลยีดิจิทัล",
    "HCM": "6 การบริหารทุนมนุษย์",
    "KM": "7-1 การจัดการความรู้",
    "IM": "7-2 การจัดการนวัตกรรม",
    "IA": "8 การตรวจสอบภายใน",
}

# ------------------------------------------------------------------
# SE-AM Sub-topic Mapping (จากหน้า 3-15 ของ SE-AM Manual Book 2566)
# ------------------------------------------------------------------
SEAM_SUBTOPIC_MAP = {
    # CG
    "1.1": "CG-1.1", "1-1": "CG-1.1",
    # SP
    "2.1": "SP-2.1", "2-1": "SP-2.1",
    # RM&IC
    "3.1": "RMIC-3.1", "3-1": "RMIC-3.1",
    # SCM
    "4.1": "SCM-4.1", "4-1": "SCM-4.1",
    # DT
    "5.1": "DT-5.1", "5-1": "DT-5.1",
    # HCM
    "6.1": "HCM-6.1", "6-1": "HCM-6.1", "6.2": "HCM-6.2", "6.3": "HCM-6.3", "6.4": "HCM-6.4",
    "6.5": "HCM-6.5", "6.6": "HCM-6.6", "6.7": "HCM-6.7",
    # KM & IM
    "7.1": "KM-7.1", "7-1": "KM-7.1",
    "7.20": "IM-7.20", "7-20": "IM-7.20",
    # IA
    "8.1": "IA-8.1", "8-1": "IA-8.1",
}

# ================================================================
# Paths & Export Configuration
# ================================================================
RUBRIC_FILENAME_PATTERN: Final[str] = "{tenant}_{enabler}_rubric.json"
EXPORTS_DIR: Final[str] = os.path.join(PROJECT_ROOT, "exports")

DOCUMENT_ID_MAPPING_FILENAME_SUFFIX: Final[str] = "_doc_id_mapping.json"
EVIDENCE_MAPPING_FILENAME_SUFFIX: Final[str] = "_evidence_mapping.json"

# ================================================================
# Action Plan Generation Control
# ================================================================
MAX_ACTION_PLAN_PHASES: Final[int] = 3
MAX_STEPS_PER_ACTION: Final[int] = 2
ACTION_PLAN_STEP_MAX_WORDS: Final[int] = 15
ACTION_PLAN_LANGUAGE: Final[str] = "th"  # "th" or "en"

# ================================================================
# PDCA Rules & Scoring
# ================================================================

# แก้ในส่วนต้นของไฟล์ core/seam_assessment.py หรือที่นิยามตัวแปรนี้
REQUIRED_PDCA: Final[Dict[int, Set[str]]] = {
    1: {"P"},
    2: {"P", "D"},
    3: {"P", "D", "C"},
    4: {"P", "D", "C", "A"},
    5: {"P", "D", "C", "A"},
}

CORRECT_PDCA_SCORES_MAP: Final[Dict[int, Dict[str, int]]] = {
    1: {"Plan": 1, "Do": 0, "Check": 0, "Act": 0},
    2: {"Plan": 1, "Do": 1, "Check": 0, "Act": 0},
    3: {"Plan": 1, "Do": 1, "Check": 1, "Act": 1},
    4: {"Plan": 2, "Do": 2, "Check": 1, "Act": 1},
    5: {"Plan": 2, "Do": 2, "Check": 2, "Act": 2},
}

PDCA_PHASE_MAP: Final[Dict[int, str]] = {
    1: "Plan (P)",
    2: "Plan (P) + Do (D)",
    3: "Plan (P) + Do (D) + Check (C)",
    4: "Plan (P) + Do (D) + Check (C) + Act (A)",
    5: "PDCA ครบวงจร (P + D + C + A) + Sustainability & Innovation",
}

# ================================================================
# PDCA Heuristic & NLP Helpers
# ================================================================
PDCA_PRIORITY_ORDER: Final[List[str]] = ["Act", "Check", "Do", "Plan"]

# =================================================================
# BASE_PDCA_KEYWORDS (Generic & Professional Version)
# เน้น "ประเภทเอกสาร" และ "คำกริยาแสดงสถานะ" เพื่อลดความซับซ้อนของ Logic
# =================================================================

BASE_PDCA_KEYWORDS: Final[Dict[str, List[str]]] = {
    "Plan": [
        # เน้นเอกสารเชิงนโยบายและโครงสร้าง
        r"นโยบาย", r"แผนแม่บท", r"ยุทธศาสตร์", r"กลยุทธ์", r"วิสัยทัศน์",
        r"เป้าหมาย", r"ทิศทางองค์กร", r"ประกาศ", r"สาส์นจากผู้บริหาร",
        r"คำแถลง", r"กรอบแนวทาง", r"ข้อกำหนด", r"หลักเกณฑ์",
        r"master plan", r"roadmap", r"policy", r"executive message",
        r"การอนุมัติหลักการ", r"คำสั่งแต่งตั้ง", r"โครงสร้างการบริหาร"
    ],
    "Do": [
        # เน้นการปฏิบัติ กิจกรรม และการบันทึกงาน
        r"การดำเนินงาน", r"การปฏิบัติ", r"การจัดทำ", r"การฝึกอบรม",
        r"การประชุม", r"บันทึกข้อความ", r"ระบบงาน", r"จัดกิจกรรม",
        r"workshop", r"cop", r"community of practice", r"การถ่ายทอด",
        r"สื่อสาร", r"ประชาสัมพันธ์", r"การนำไปใช้", r"action plan",
        r"ขับเคลื่อน", r"คณะทำงาน", r"ภาพถ่ายกิจกรรม"
    ],
    "Check": [
        # เน้นการวัดผล การเปรียบเทียบ และการตรวจสอบ
        r"การวัดผล", r"kpi", r"การประเมิน", r"รายงานผล", r"ตรวจสอบ",
        r"ติดตามผล", r"วิเคราะห์", r"ช่องว่าง", r"audit", r"review",
        r"dashboard", r"ผลลัพธ์", r"ตัวชี้วัด", r"สรุปผล",
        r"gap analysis", r"evaluation", r"performance", r"การติดตามความคืบหน้า"
    ],
    "Act": [
        # เน้นการแก้ไข การเรียนรู้ และการพัฒนาต่อยอด
        r"การปรับปรุง", r"การแก้ไข", r"บทเรียนที่ได้รับ", r"lesson learned",
        r"corrective action", r"มาตรการแก้ไข", r"ข้อเสนอแนะ", r"การขยายผล",
        r"best practice", r"นวัตกรรม", r"innovation", r"แนวทางพัฒนา",
        r"การเปลี่ยนแปลง", r"ความยั่งยืน", r"sustainability", r"มาตรฐานใหม่"
    ],
}

PDCA_LEVEL_SYNONYMS: Final[Dict[int, str]] = {
    1: "นโยบาย, แผนแม่บท, ยุทธศาสตร์, วิสัยทัศน์, การกำหนดเป้าหมาย, ผู้บริหารระดับสูง, การมอบนโยบาย, การอนุมัติแผนงาน",
    2: "คณะทำงาน, โครงสร้างองค์กร, การขับเคลื่อน, การดำเนินการ, ความรับผิดชอบ, แผนปฏิบัติการ, การสื่อสารนโยบาย, การอบรม",
    3: "การวัดผล, การประเมินผล, KPI, รายงานผล, Audit, การตรวจสอบ, การทบทวน, การติดตามผล, การนำข้อมูลป้อนกลับไปใช้, ช่องว่าง (Gap)",
    4: "การปรับปรุง, Corrective Action, Preventive Action, บทเรียนที่ได้รับ, การแก้ไข, ปรับแผน, การสร้างนวัตกรรม, การจัดการข้อเสนอแนะ",
    5: "นวัตกรรม, ความยั่งยืน, Best Practice, การขยายผล, ผลกระทบระยะยาว, รางวัล, External Recognition, การทบทวนวิสัยทัศน์องค์กร"
}

# ================================================================
# Intent & Analysis Signals
# ================================================================
PDCA_ANALYSIS_SIGNALS: Final[List[str]] = [
    "วิเคราะห์", "ตรวจสอบ", "มี pdca ไหม", "ครบไหม",
    "ประเมินหลักฐาน", "ความสมบูรณ์", "p-d-c-a", "analyze",
    "ขาดอะไร", "เช็คหลักฐาน",
]

ANALYSIS_FRAMEWORK: Final[str] = "PDCA (Plan-Do-Check-Act)"

INSUFFICIENT_DATA_MSG: Final[str] = (
    "ข้อมูลในเอกสารไม่เพียงพอต่อการวิเคราะห์ครบวงจร PDCA"
)
