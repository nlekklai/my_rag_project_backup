#config/globa_vars.py
import os

# ==================== Project Paths ====================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
VECTORSTORE_DIR = os.path.join(PROJECT_ROOT, "vectorstore")
MAPPING_FILE_PATH = os.path.join(DATA_DIR, "doc_id_mapping.json")
INITIAL_TOP_K = 25
FINAL_K_RERANKED = 5
FINAL_K_NON_RERANKED = 7
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 250


# ==================== Supported File & Document Types ====================
SUPPORTED_TYPES = [
    ".pdf", ".docx", ".txt", ".xlsx", ".pptx", ".md", ".csv", ".jpg", ".jpeg", ".png"
]

SUPPORTED_DOC_TYPES = [
    "document", "policy", "report", "statement", "evidence", "feedback", "faq", "seam"
]

EVIDENCE_DOC_TYPES = "evidence"
DEFAULT_DOC_TYPES = "document"


# ==================== Enabler Configuration ====================
DEFAULT_ENABLER = "KM"
SUPPORTED_ENABLERS = ["CG", "SP", "RM&IC", "SCM", "DT", "HCM", "KM", "IM", "IA"]

# ------------------------------------------------------------------
# SE-AM Reference Document Mapping (Updated from latest ingestion)
# ------------------------------------------------------------------

SEAM_ENABLER_MAP = {
    "CG": "1 การกำกับดูแลที่ดีและการนำองค์กร (Corporate Governance & Leadership)",
    "SP": "2 การวางแผนเชิงยุทธศาสตร์ (Strategic Planning)",
    "RM&IC": "3 การบริหารความเสี่ยงและการควบคุมภายใน (Risk Management & Internal Control)",
    "SM": "4.1 การมุ่งเน้นผู้มีส่วนได้ส่วนเสีย (Stakeholder Management)",
    "CM": "4.2 การมุ่งเน้นลูกค้า (Customer Management)",
    "DT": "5 การพัฒนาเทคโนโลยีดิจิทัล (Digital Technology)",
    "HCM": "6 การบริหารทุนมนุษย์ (Human Capital Management)",
    "KM": "7.1 การจัดการความรู้ (Knowledge Management)",
    "IM": "7.2 นวัตกรรม (Innovation Management)",
    "IA": "8 การตรวจสอบภายใน (Internal Audit)"
}

