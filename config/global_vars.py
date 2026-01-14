# config/global_vars.py
import os
import uuid
from typing import List, Dict, Set, Final
import torch
from dotenv import load_dotenv

load_dotenv()

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
# Device & Hardware Acceleration
# ================================================================
# ระบบจะเลือก cuda (Server), mps (Mac M1/M2), หรือ cpu (ทั่วไป) เอง
if torch.cuda.is_available():
    TARGET_DEVICE: Final[str] = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    TARGET_DEVICE: Final[str] = "mps"
else:
    TARGET_DEVICE: Final[str] = "cpu"

# กำหนด Batch Size ตามความแรงของ Device
# L40S (cuda) ใช้ 32, Mac (mps) ใช้ 8, CPU ใช้ 4
if TARGET_DEVICE == "cuda":
    DEFAULT_EMBED_BATCH_SIZE: Final[int] = 32
elif TARGET_DEVICE == "mps":
    DEFAULT_EMBED_BATCH_SIZE: Final[int] = 8
else:
    DEFAULT_EMBED_BATCH_SIZE: Final[int] = 4

# ================================================================
# Ollama / LLM Request Control
# ================================================================
OLLAMA_REQUEST_TIMEOUT: Final[int] = 300  # seconds
OLLAMA_MAX_RETRIES: Final[int] = 3

# ================================================================
# Run Mode & LLM Configuration
# ================================================================
# อ่านโหมด (ซึ่งของคุณจะเป็น LOCAL_OLLAMA เสมอ)
RAG_RUN_MODE: Final[str] = os.environ.get("RAG_RUN_MODE", "LOCAL_OLLAMA")

# ดึงชื่อ Model และ Context จาก .env
# ถ้าไม่มีใน .env ให้ใช้ 8b เป็นค่าเริ่มต้นสำหรับ Mac
DEFAULT_LLM_MODEL_NAME: Final[str] = os.environ.get("OLLAMA_MODEL_NAME", "llama3:8b")
LLM_CONTEXT_WINDOW: Final[int] = int(os.environ.get("LLM_CONTEXT_WINDOW", "8192"))

# ดึง URL ของ Ollama (เผื่อในอนาคต Mac อยากชี้ไปหา Server)
OLLAMA_BASE_URL: Final[str] = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# จำนวน Workers (Mac อาจใช้ 2, Server L40S อาจใช้ 8)
MAX_PARALLEL_WORKERS: Final[int] = int(os.environ.get("MAX_PARALLEL_WORKERS", "2"))

LLM_TEMPERATURE: Final[float] = 0.0

# ================================================================
# Embedding & Reranker Models
# ================================================================
EMBEDDING_MODEL_NAME: Final[str] = "BAAI/bge-m3"
RERANKER_MODEL_NAME: Final[str] = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-base")

# 🎯 เพิ่มส่วนนี้เข้าไปครับ
EMBEDDING_MODEL_KWARGS: Final[Dict] = {
    "device": TARGET_DEVICE,
    "trust_remote_code": True  # แก้ปัญหาเรื่องความปลอดภัยและ Meta Tensor
}

EMBEDDING_ENCODE_KWARGS: Final[Dict] = {
    "normalize_embeddings": True,
    "batch_size": DEFAULT_EMBED_BATCH_SIZE # ใช้ค่าที่คำนวณตาม Device ด้านบน
}

# ================================================================
# Hybrid Search Configuration
# ================================================================
USE_HYBRID_SEARCH: Final[bool] = True
HYBRID_VECTOR_WEIGHT: Final[float] = 0.6
HYBRID_BM25_WEIGHT: Final[float] = 0.4


# ================================================================
# Retrieval & Ranking Parameters
# ================================================================
# 🎯 [NEW] ปรับเปลี่ยนตามเครื่องที่รัน (Mac vs Server) ผ่าน .env
# INITIAL_TOP_K: จำนวนที่ดึงจาก ChromaDB เบื้องต้น
INITIAL_TOP_K: Final[int] = int(os.environ.get("INITIAL_TOP_K", "75"))

# RETRIEVAL_TOP_K: จำนวนที่ดึงมาสำหรับงาน Analysis/Consultant (เน้นความครอบคลุม)
# แนะนำ: Mac=150-200, Server=300-500
RETRIEVAL_TOP_K: Final[int] = int(os.environ.get("RETRIEVAL_TOP_K", "500"))

# ANALYSIS_FINAL_K: จำนวน Chunk สุดท้ายที่จะส่งให้ LLM วิเคราะห์ (หลัง Rerank)
# แนะนำ: Mac=12-15 (ประหยัด RAM), Server=25-35 (เน้นความละเอียด)
ANALYSIS_FINAL_K: Final[int] = int(os.environ.get("ANALYSIS_FINAL_K", "15"))
# 🎯 จำนวน Chunk สุดท้ายสำหรับคำถามทั่วไป (General QA)
QA_FINAL_K: Final[int] = int(os.environ.get("QA_FINAL_K", "30"))

# ส่วนคงเดิมสำหรับ General QA
FINAL_K_RERANKED: Final[int] = int(os.environ.get("FINAL_K_RERANKED", "15"))
FINAL_K_NON_RERANKED: Final[int] = 7


RERANK_THRESHOLD: Final[float] = 0.35
MIN_RETRY_SCORE: Final[float] = 0.50
MAX_RETRIEVAL_ATTEMPTS: Final[int] = 3

MIN_RERANK_SCORE_TO_KEEP: Final[float] = 0.10
MIN_RELEVANCE_THRESHOLD: Final[float] = 0.3

CRITICAL_CA_THRESHOLD: Final[float] = 0.65

# ================================================================
# Context Control & LLM Optimization (Mac vs Server)
# ================================================================
# 🎯 [NEW] จำกัดจำนวน Chunk ต่อไฟล์ เพื่อป้องกันไฟล์เดียวครอง Block
# แนะนำ: Mac=3, Server=5
MAX_CHUNKS_PER_FILE: Final[int] = int(os.environ.get("RAG_MAX_CHUNKS_PER_FILE", "3"))

# 🎯 [NEW] จำนวน Chunk สูงสุดที่จะส่งให้ LLM ต่อ 1 PDCA Block
# สำหรับเครื่อง Dev (Mac) แนะนำ 5-7 เพื่อป้องกัน Context Overflow
# สำหรับเครื่อง Server (L40S) แนะนำ 12-15 เพื่อความละเอียดสูงสุด
MAX_CHUNKS_PER_BLOCK: Final[int] = int(os.environ.get("RAG_MAX_CHUNKS_PER_BLOCK", "7"))

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

# ================================================================
# Priority & Parallel Processing
# ================================================================
LIMIT_CHUNKS_PER_PRIORITY_DOC: Final[int] = 5
PRIORITY_CHUNK_LIMIT: Final[int] = 30


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

# ================================================================
# PDCA Phase Mapping per Level
# ================================================================
PDCA_PHASE_MAP: Final[Dict[int, str]] = {
    1: "Plan (การกำหนดเป้าหมายและนโยบาย)",
    2: "Do (การนำแผนไปปฏิบัติและขับเคลื่อน)",
    3: "Check (การติดตามและประเมินผล)",
    4: "Act (การปรับปรุงและสร้างนวัตกรรม)",
    5: "Sustainability (ความยั่งยืนและต้นแบบที่ดี)"
}

# ================================================================
# Maturity Level Core Goals (สำหรับ Audit Guidance)
# ================================================================
MATURITY_LEVEL_GOALS: Final[Dict[int, str]] = {
    1: "เน้นการเริ่มต้น มีนโยบาย หรือมีแนวทางปฏิบัติเบื้องต้นที่เป็นลายลักษณ์อักษร",
    2: "เน้นการนำไปใช้อย่างเป็นระบบ มีคณะทำงาน หรือมีการประกาศใช้ครอบคลุมหน่วยงานหลัก",
    3: "เน้นการปฏิบัติอย่างต่อเนื่องทั่วทั้งองค์กร และมีรายงานสรุปผลลัพธ์ที่เป็นรูปธรรม",
    4: "เน้นการวิเคราะห์ด้วยข้อมูลเชิงสถิติ (KPI) หรือมีการสร้างนวัตกรรม/Best Practice",
    5: "เน้นความยั่งยืน การปรับปรุงเชิงรุกตามสภาวะที่เปลี่ยนไป และการเป็นต้นแบบ (Role Model)"
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
PDCA_PRIORITY_ORDER: Final[List[str]] = ["Act", "Check", "Do", "Plan"]