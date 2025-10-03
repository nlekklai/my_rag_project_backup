import os
import logging
from core.ingest import process_document, DATA_DIR, SUPPORTED_TYPES  # ใช้ SUPPORTED_TYPES แทน

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ingest_all_documents():
    """
    Sequential ingestion ของไฟล์ทั้งหมดใน DATA_DIR
    - รองรับ pdf, docx, txt, csv (ตาม SUPPORTED_TYPES)
    - ใช้ process_document จาก core/ingest.py
    """
    logger.info("🚀 Starting ingestion for all documents in %s", DATA_DIR)

    if not os.path.exists(DATA_DIR):
        logger.warning("⚠️ DATA_DIR %s does not exist", DATA_DIR)
        return

    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_TYPES:  # แก้ตรงนี้
                filepath = os.path.join(root, f)
                logger.info("📄 Ingesting %s", filepath)
                try:
                    process_document(filepath, f)  # เรียกด้วยชื่อไฟล์ด้วย
                except Exception as e:
                    logger.error("❌ Error ingesting %s: %s", filepath, str(e))

    logger.info("✅ Ingestion completed.")

if __name__ == "__main__":
    ingest_all_documents()
