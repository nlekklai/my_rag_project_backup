import os

# ==================== Project Paths ====================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
VECTORSTORE_DIR = os.path.join(PROJECT_ROOT, "vectorstore")
MAPPING_FILE_PATH = os.path.join(DATA_DIR, "doc_id_mapping.json")
INITIAL_TOP_K = 15
FINAL_K_RERANKED = 5
FINAL_K_NON_RERANKED = 7

# ==================== Supported File & Document Types ====================
SUPPORTED_TYPES = [
    ".pdf", ".docx", ".txt", ".xlsx", ".pptx", ".md", ".csv", ".jpg", ".jpeg", ".png"
]

SUPPORTED_DOC_TYPES = [
    "document", "policy", "report", "statement", "evidence", "feedback", "faq", "seam"
]

# ==================== Enabler Configuration ====================
DEFAULT_ENABLER = "KM"
SUPPORTED_ENABLERS = ["CG", "SP", "RM&IC", "SCM", "DT", "HCM", "KM", "IM", "IA"]

# ==================== SEAM Reference Documents ====================
DEFAULT_SEAM_REFERENCE_DOC_ID = "f9925584be20a9c8b2bdce0ebce3ba07790cfe3380b172002a2844afccef47ce"

SEAM_DOC_ID_MAP = {
    "CG": "d98fd87335e05887388b57e7d31f3b47814849f3e3075c911ddbf4cac6d93294",
    "SP": "09d85e227d12568c300f2dd4463e7b575910ef97c5b02613f8660701b3eefb07",
    "RM&IC": "5425eeffc6f7319b9346075726c247bb6bd22afee967852ad109a1bcbdd3f535",
    "SCM": "c2039bac6099a6536c8fd572e5e34769f6697dd1d32cd400d4a4c6b3646b4172",
    "DT": "ad673b036b771497fdd201c46d0211d1d7dabe3b4432e4a0c4d3c59fddc27942",
    "HCM": "29c3d2b88c054792d64eac35a04deeed76766feeb246fd7d81f06ef4aebcea15",
    "KM": "f8b7a390dd21584fefd8747c758eea44e024f44aa4174329a7a82d04ed499cad",
    "IM": "f9e52688d46e0f4ecca8b5e7f31506d49d58954d229d8fd0de7f8fbdece129e7",
    "IA": "1dc1122ea571cd7f812bd301d5bdacd89e695fec7203389b343e95a374525eb8",
}

SEAM_ENABLER_MAP = {
    "CG": "1 การกำกับดูแลที่ดีและการนำองค์กร (Corporate Governance & Leadership)",
    "SP": "2 การวางแผนเชิงยุทธศาสตร์ (Strategic Planning)",
    "RM": "3 การบริหารความเสี่ยงและการควบคุมภายใน (Risk Management & Internal Control)",
    "SM": "4.1 การมุ่งเน้นผู้มีส่วนได้ส่วนเสีย (Stakeholder Management)",
    "CM": "4.2 การมุ่งเน้นลูกค้า (Customer Management)",
    "DT": "5 การพัฒนาเทคโนโลยีดิจิทัล (Digital Technology)",
    "HCM": "6 การบริหารทุนมนุษย์ (Human Capital Management)",
    "KM": "7.1 การจัดการความรู้ (Knowledge Management)",
    "IM": "7.2 นวัตกรรม (Innovation Management)",
    "IA": "8 การตรวจสอบภายใน (Internal Audit)"
}

