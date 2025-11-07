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

DEFAULT_SEAM_REFERENCE_DOC_ID = "f9925584be20a9c8b2bdce0ebce3ba07790cfe3380b172002a2844afccef47ce"  # SE-AM Manual Book 2566 ฉบับสมบูรณ์

SEAM_DOC_ID_MAP = {
    "CG":    "31aa43addf8c4cbd0f6c8fe94ae8c4848cf3bf52279e6f26af083e2bfac6d829",  # SE-AM_CG (57 chunks)
    "CM":    "04cbf9c9292b390b3378230e3e94fa8111f674db629d34b2a7cbd0b4115836ae",  # SE-AM_CM (28 chunks)
    "DT":    "e3c230930f10585b6b625ec06a5ca5d9eee78345eb8ab209992d9e631ba17dc3",  # SE-AM_DT (48 chunks)
    "HCM":   "3f5fe87c7788af2206dd35c78c009f367921340e00b28854be0dfd1a757546d2",  # SE-AM_HCM (55 chunks)
    "IA":    "84aa9ea37c308b2d6bf8fadd48359abdf679d958516429b730d93901aacafa08",  # SE-AM_IA (42 chunks)
    "IM":    "83651ad6c7c33c7129ff41490443cef6af0ec6110e7c58f9b5f1f41444375537",  # SE-AM_IM (27 chunks)
    "KM":    "68b256a5b6def286425e90c3f75c3b54c70b1f5dd32d0522b2fa235b07b9d454",  # SE-AM_KM (33 chunks)
    "RM&IC": "01f8566eeb9e9da319e11383b4d91a1d4168ac3c4a849db1d0d155d3d1b5c25c",  # SE-AM_RM&IC (62 chunks)
    "SM":    "b78dc9bdebdb83d25b975988c7162d23962ac871e93078b349eb132bb46762f8",  # SE-AM_SM (32 chunks)
    "SP":    "38ad222c392281d64d544dd8d5185dc6b7fc09de99c745d9c902a987fd78aad4",  # SE-AM_SP (54 chunks)
}


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

