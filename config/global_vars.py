# config/global_vars.py
import os
import uuid
from typing import List, Dict, Set, Final

# ==================== Project & Namespace ====================
PROJECT_NAMESPACE_UUID: Final[uuid.UUID] = uuid.UUID('f77c38c0-f213-4318-ae38-e69c73e97022')

PROJECT_ROOT: Final[str] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_STORE_ROOT: Final[str] = os.path.join(PROJECT_ROOT, "data_store")


# ==================== Tenant & Default Configuration ====================
DEFAULT_TENANT: Final[str] = "pea"
DEFAULT_YEAR: Final[int] = 2568
DEFAULT_ENABLER: Final[str] = "KM"

# ==================== Ollama / LLM Request Control ====================
OLLAMA_REQUEST_TIMEOUT: Final[int] = 300  # seconds
OLLAMA_MAX_RETRIES: Final[int] = 3

# ==================== Run Mode & LLM Configuration ====================
RAG_RUN_MODE: Final[str] = "LOCAL_OLLAMA"

DEFAULT_LLM_MODEL_NAME: Final[str] = "llama3:8b"
LLM_TEMPERATURE: Final[float] = 0
LLM_CONTEXT_WINDOW: Final[int] = 8192


# ==================== Embedding & Reranker Models ====================
EMBEDDING_MODEL_NAME: Final[str] = "BAAI/bge-m3"
RERANKER_MODEL_NAME: Final[str] = "BAAI/bge-reranker-base"


# ==================== Hybrid Search Configuration ====================
USE_HYBRID_SEARCH: Final[bool] = True
HYBRID_VECTOR_WEIGHT: Final[float] = 0.7
HYBRID_BM25_WEIGHT: Final[float] = 0.3


# ==================== Retrieval & Ranking Parameters ====================
INITIAL_TOP_K: Final[int] = 75
FINAL_K_RERANKED: Final[int] = 15
FINAL_K_NON_RERANKED: Final[int] = 7

RERANK_THRESHOLD: Final[float] = 0.5
MIN_RETRY_SCORE: Final[float] = 0.50
MAX_RETRIEVAL_ATTEMPTS: Final[int] = 3

MIN_RERANK_SCORE_TO_KEEP: Final[float] = 0.10
MIN_RELEVANCE_THRESHOLD: Final[float] = 0.3

CRITICAL_CA_THRESHOLD: Final[float] = 0.65

# 📌 NEW HARD FAIL CONTROL FLAGS
ENABLE_HARD_FAIL_LOGIC: Final[bool] = False
ENABLE_CONTEXTUAL_RULE_OVERRIDE: Final[bool] = True

MAX_EVI_STR_CAP: Final[float] = 10.0
CONTEXT_CAP_L3_PLUS: Final[int] = 60000


# ==================== Chunking Configuration ====================
CHUNK_SIZE: Final[int] = 1500
CHUNK_OVERLAP: Final[int] = 250
STANDARD_K: Final[int] = 5

QUERY_INITIAL_K: Final[int] = 20
QUERY_FINAL_K: Final[int] = 5


# ==================== Priority & Parallel Processing ====================
LIMIT_CHUNKS_PER_PRIORITY_DOC: Final[int] = 5
PRIORITY_CHUNK_LIMIT: Final[int] = 30
MAX_PARALLEL_WORKERS: Final[int] = 4


# ==================== Logging & Context Control ====================
IS_LOG_L3_CONTEXT: Final[bool] = True
MAX_EVAL_CONTEXT_LENGTH: Final[int] = 3000


# ==================== Supported File & Document Types ====================
SUPPORTED_TYPES: Final[List[str]] = [
    ".pdf", ".docx", ".txt", ".xlsx", ".pptx", ".md", ".csv",
    ".jpg", ".jpeg", ".png"
]

SUPPORTED_DOC_TYPES: Final[List[str]] = [
    "document", "policy", "report", "statement", "evidence",
    "feedback", "faq", "seam"
]

EVIDENCE_DOC_TYPES: Final[str] = "evidence"
DEFAULT_DOC_TYPES: Final[str] = "document"


# ==================== Enabler & Assessment Constants ====================
SUPPORTED_ENABLERS: Final[List[str]] = [
    "CG", "SP", "RM&IC", "SCM", "DT", "HCM", "KM", "IM", "IA"
]

MAX_LEVEL: Final[int] = 5
INITIAL_LEVEL: Final[int] = 1


# ==================== SE-AM Enabler Mapping ====================
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


# ==================== Paths & Export Configuration ====================
RUBRIC_FILENAME_PATTERN: Final[str] = "{tenant}_{enabler}_rubric.json"
RUBRIC_CONFIG_DIR: Final[str] = os.path.join(PROJECT_ROOT, "config", "mapping")
EXPORTS_DIR: Final[str] = os.path.join(PROJECT_ROOT, "exports")

DOCUMENT_ID_MAPPING_FILENAME_SUFFIX: Final[str] = "_doc_id_mapping.json"
EVIDENCE_MAPPING_FILENAME_SUFFIX: Final[str] = "_evidence_mapping.json"


# ==================== Action Plan Generation Control ====================
MAX_ACTION_PLAN_PHASES: Final[int] = 3
MAX_STEPS_PER_ACTION: Final[int] = 2
ACTION_PLAN_STEP_MAX_WORDS: Final[int] = 15
ACTION_PLAN_LANGUAGE: Final[str] = "th"  # "th" หรือ "en"

# =================================================================
# 🟢 Helper Function for PDCA Calculation (Priority 1 Part 2 & Priority 2)
# =================================================================

# 📌 NEW: REQUIRED_PDCA Global Constant (Patch 1 Dependency)
REQUIRED_PDCA: Final[Dict[int, Set[str]]] = {
    1: {"P"},
    2: {"P", "D"},
    3: {"P", "D", "C"},
    4: {"P", "D", "C", "A"},
    5: {"P", "D", "C", "A"},
}

# ----------------------------------------------------------------------
CORRECT_PDCA_SCORES_MAP: Final[Dict[int, Dict[str, int]]] = {
    1: {'P': 1, 'D': 0, 'C': 0, 'A': 0},
    2: {'P': 1, 'D': 1, 'C': 0, 'A': 0},
    3: {'P': 1, 'D': 1, 'C': 1, 'A': 1},
    4: {'P': 2, 'D': 2, 'C': 1, 'A': 1},
    5: {'P': 2, 'D': 2, 'C': 2, 'A': 2},
}

PDCA_PHASE_MAP: Final[Dict[int, str]] = {
    1: "Plan (P)",
    2: "Plan (P) + Do (D)",
    3: "Plan (P) + Do (D) + Check (C)",
    4: "Plan (P) + Do (D) + Check (C) + Act (A)",
    5: "PDCA ครบวงจร (P + D + C + A) + Sustainability & Innovation"
}

# =================================================================
# Heuristic Classification Helpers - ULTIMATE VERSION (ใช้ contextual_rules.json)
# =================================================================
PDCA_PRIORITY_ORDER = ['Act', 'Check', 'Do', 'Plan']

# Keyword พื้นฐาน (fallback)
BASE_PDCA_KEYWORDS: Final[Dict[str, List[str]]] = {
    'Plan': [
        r'นโยบาย', r'แผน', r'กลยุทธ์', r'กรอบแนวทาง', r'วิสัยทัศน์', r'เป้าหมาย', r'กำหนด',
        r'ยุทธศาสตร์', r'แผนแม่บท', r'master plan', r'roadmap', r'กำหนดทิศทาง'
    ],
    'Do': [
        r'การดำเนินงาน', r'การจัดทำ', r'การฝึกอบรม', r'การปฏิบัติ', r'ระบบ', r'ดำเนินการ', r'จัดกิจกรรม',
        r'แต่งตั้ง', r'คณะทำงาน', r'ถ่ายทอด', r'action plan', r'ขับเคลื่อน', r'จัดตั้ง'
    ],
    'Check': [
        r'การวัดผล', r'kpi', r'การประเมิน', r'รายงานผล', r'การวิเคราะห์ช่องว่าง', r'ตรวจสอบ', r'ผลลัพธ์', r'ติดตาม',
        r'ตัวชี้วัด', r'audit', r'review', r'ประเมินผล',
        r'ความคืบหน้า', r'ปัญหาและอุปสรรค'  # <-- เพิ่ม
    ],
    'Act': [
        r'การปรับปรุง', r'การแก้ไข', r'บทเรียนที่ได้รับ', r'corrective action', r'เปลี่ยนแปลงวิธีการ', r'มาตรการ',
        r'ปรับปรุงอย่างต่อเนื่อง', r'lesson learned', r'นำมาปรับปรุง',
        r'ข้อเสนอแนะ', r'แนวทางแก้ไข' # <-- เพิ่ม
    ]
}

PDCA_LEVEL_SYNONYMS: Final[Dict[int, str]] = {
    1: "นโยบาย, แผนแม่บท, ยุทธศาสตร์, วิสัยทัศน์, การกำหนดเป้าหมาย, ผู้บริหารระดับสูง",
    2: "คณะทำงาน, โครงสร้างองค์กร, การขับเคลื่อน, การดำเนินการ, ความรับผิดชอบ, แผนปฏิบัติการ, การสื่อสารนโยบาย",
    3: "การวัดผล, การประเมินผล, KPI, รายงานผล, Audit, การตรวจสอบ, การทบทวน, การติดตามผล, การนำข้อมูลป้อนกลับไปใช้",
    4: "การปรับปรุง, Corrective Action, Preventive Action, บทเรียนที่ได้รับ, การแก้ไข, ปรับแผน, การสร้างนวัตกรรม",
    5: "นวัตกรรม, ความยั่งยืน, Best Practice, การขยายผล, ผลกระทบระยะยาว, รางวัล, External Recognition, การทบทวนวิสัยทัศน์",
}

# --- Intent & Analysis Settings ---

# เพิ่ม signals สำหรับตรวจจับคำถามประเภทวิเคราะห์ PDCA
PDCA_ANALYSIS_SIGNALS: Final[List[str]] = [
    "วิเคราะห์", "ตรวจสอบ", "มี pdca ไหม", "ครบไหม", 
    "ประเมินหลักฐาน", "ความสมบูรณ์", "p-d-c-a", "analyze",
    "ขาดอะไร", "เช็คหลักฐาน"
]

# กำหนดหัวข้อหลักของการวิเคราะห์ (ป้องกัน LLM ออกนอกลู่นอกทาง)
ANALYSIS_FRAMEWORK: Final[str] = "PDCA (Plan-Do-Check-Act)"

# ข้อความแจ้งเตือนกรณีข้อมูลไม่เพียงพอต่อการวิเคราะห์
INSUFFICIENT_DATA_MSG: Final[str] = "ข้อมูลในเอกสารไม่เพียงพอต่อการวิเคราะห์ครบวงจร PDCA"