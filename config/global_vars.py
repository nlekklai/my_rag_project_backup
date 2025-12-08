# config/global_vars.py
import os
from typing import Final, List


# -------------------- Tenant / Context Configuration (NEW) --------------------
DEFAULT_TENANT: Final[str] = "pea" 
DEFAULT_YEAR: Final[int] = 2568    

# ==================== Project Paths (CORRECTED for Clean Multi-Tenant) ====================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
VECTORSTORE_DIR = os.path.join(PROJECT_ROOT, "vectorstore")

MAPPING_BASE_DIR = os.path.join(PROJECT_ROOT, "config", "mapping") 

# RAG_RUN_MODE = "ollama"  # หรือ "local" หรือ "cloud"
RAG_RUN_MODE: Final[str] = "LOCAL_OLLAMA"

# =================================================================
# 🟢 New: EMBEDDING & RERANKER MODEL CONFIGURATION
# =================================================================

# Embedding Model (ใช้สำหรับสร้าง Vector ในการ Ingest และ Retrieval ครั้งแรก)
# BAAI/bge-m3 เป็น SOTA Multilingual Model ขนาด 2.27 GB
EMBEDDING_MODEL_NAME: Final[str] = "BAAI/bge-m3"
#previous version intfloat/multilingual-e5-base

# Reranker Model (ใช้สำหรับคัดกรองความเกี่ยวข้องของ Chunk ที่ดึงมา)
# BAAI/bge-reranker-base เป็น Reranker ที่เหมาะสมกับ BGE-M3 และ Multilingual
RERANKER_MODEL_NAME: Final[str] = "BAAI/bge-reranker-base" 
#previous version "intfloat/multilingual-e5-base" and cross-encoder/ms-marco-MiniLM-L-6-v2


# -------------------- Retrieval / Evaluation --------------------
# L1–L2 ต้องการ context กว้าง → ให้ top_k สูงขึ้น
INITIAL_TOP_K: Final[int] = 100             # ใช้สำหรับ retrieval ก่อน rerank
FINAL_K_RERANKED: Final[int] = 12          # สำหรับ L1–L2; L3–L5 ใช้ 5
FINAL_K_NON_RERANKED: Final[int] = 7       # สำหรับ non-reranked
RERANK_THRESHOLD: Final[float] = 0.5  # เกณฑ์คุณภาพหลักฐาน (Threshold)
MAX_EVI_STR_CAP: Final[float] = 3.0   # Evidence Strength สูงสุดเมื่อหลักฐานคุณภาพต่ำ

CHUNK_SIZE: Final[int] = 1500
CHUNK_OVERLAP: Final[int] = 250
STANDARD_K: Final[int] = 5

QUERY_INITIAL_K = 20
QUERY_FINAL_K = 5

IS_LOG_L3_CONTEXT = True

# LLM Model (ใช้ตัวเดียว)
LLM_MODEL_NAME = "llama3.1:8b"
# LLM_MODEL_NAME = "llama3:8b-instruct-q4_0"
LLM_TEMPERATURE: Final[float] = 0.0
LLM_CONTEXT_WINDOW: Final[int] = 4096


# ==================== Supported File & Document Types ====================
SUPPORTED_TYPES: Final[List[str]] = [
    ".pdf", ".docx", ".txt", ".xlsx", ".pptx", ".md", ".csv", ".jpg", ".jpeg", ".png"
]

SUPPORTED_DOC_TYPES: Final[List[str]] = [
    "document", "policy", "report", "statement", "evidence", "feedback", "faq", "seam"
]

EVIDENCE_DOC_TYPES: Final[str] = "evidence"
DEFAULT_DOC_TYPES: Final[str] = "document"

# ==================== Enabler Configuration ====================
DEFAULT_ENABLER: Final[str] = "KM"
SUPPORTED_ENABLERS: Final[List[str]] = ["CG", "SP", "RM&IC", "SCM", "DT", "HCM", "KM", "IM", "IA"]

# ------------------------------------------------------------------
# SE-AM Reference Document Mapping (Updated from latest ingestion)
# ------------------------------------------------------------------
SEAM_ENABLER_MAP: Final[dict] = {
    "CG": "1 การกำกับดูแลที่ดีและการนำองค์กร",
    "SP": "2 การวางแผนเชิงยุทธศาสตร์",
    "RM&IC": "3 การบริหารความเสี่ยงและการควบคุมภายใน",
    "SCM": "4 การมุ่งเน้นผู้มีส่วนได้ส่วนเสีย และลูกค้า",
    "DT": "5 การพัฒนาเทคโนโลยีดิจิทัล",
    "HCM": "6 การบริหารทุนมนุษย์",
    "KM": "7 การจัดการความรู้และนวัตกรรม",
    "IM": "7 การจัดการความรู้และนวัตกรรม",  # ชี้ไปที่เดียวกับ KM
    "IA": "8 การตรวจสอบภายใน",
}

# --- Assessment Constants ---
MAX_LEVEL: Final[int] = 5 
INITIAL_LEVEL: Final[int] = 1
MAX_PARALLEL_WORKERS: Final[int] = 4   # แนะนำ 4 สำหรับ Mac
LIMIT_CHUNKS_PER_PRIORITY_DOC = 5
MAX_EVAL_CONTEXT_LENGTH = 4500
PRIORITY_CHUNK_LIMIT: Final[int] = 30

# 💡 Rubric / Export Paths
RUBRIC_FILENAME_PATTERN: Final[str] = "{tenant}_{enabler}_rubric.json"
# RUBRIC_CONFIG_DIR: Final[str] = "config"
RUBRIC_CONFIG_DIR: Final[str] = MAPPING_BASE_DIR
EXPORTS_DIR: Final[str] = os.path.join(PROJECT_ROOT, "exports")
# KM_EVIDENCE_STATEMENTS_FILE: Final[str] = os.path.join(RUBRIC_CONFIG_DIR, "km_evidence_statements.json")
DOCUMENT_ID_MAPPING_FILENAME_SUFFIX: Final[str] = "_doc_id_mapping.json"
EVIDENCE_MAPPING_FILENAME_SUFFIX = "_evidence_mapping.json"