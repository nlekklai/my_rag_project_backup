# utils/ingest_runner.py (ปรับปรุง)
import os, sys
import logging
# เพิ่ม root project ให้ Python หา core ได้
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.ingest import process_document, DATA_DIR, SUPPORTED_TYPES, VECTORSTORE_DIR # เพิ่ม VECTORSTORE_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ingest_all_documents_recursive():
    """
    Sequential ingestion ของไฟล์ทั้งหมดใน DATA_DIR
    - รองรับโครงสร้างโฟลเดอร์ย่อย (ใช้ชื่อโฟลเดอร์ย่อยเป็น doc_type)
    - ใช้ process_document จาก core/ingest.py
    """
    logger.info("🚀 Starting recursive ingestion for all documents in %s", DATA_DIR)

    if not os.path.exists(DATA_DIR):
        logger.warning("⚠️ DATA_DIR %s does not exist", DATA_DIR)
        return

    # สแกนทุกไฟล์ในโฟลเดอร์ย่อย (Recursive)
    for root, _, files in os.walk(DATA_DIR):
        # 1. กำหนด doc_type จากโครงสร้างโฟลเดอร์
        # ถ้า root คือ DATA_DIR ให้ถือว่าเป็น doc_type หลัก "document" (หรือ None)
        if root == DATA_DIR:
            doc_type_folder = None # หรืออาจใช้ "document" เป็นค่า default
        else:
            # ใช้ชื่อโฟลเดอร์ย่อยที่อยู่ติดกับ DATA_DIR เป็น doc_type
            doc_type_folder = os.path.basename(root)
            # ตัวอย่าง: data/rubrics -> doc_type_folder = rubrics

        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_TYPES:
                filepath = os.path.join(root, f)
                logger.info("📄 Ingesting %s (Type: %s)", filepath, doc_type_folder or "default")
                
                try:
                    # process_document จะจัดเก็บ vectorstore ใน vectorstore/<doc_type_folder>/<doc_id>
                    process_document(
                        file_path=filepath, 
                        file_name=f, 
                        doc_type=doc_type_folder # ส่ง doc_type ที่ดึงมา
                    )
                except Exception as e:
                    logger.error("❌ Error ingesting %s: %s", filepath, str(e))

    logger.info("✅ Recursive ingestion completed.")

if __name__ == "__main__":
    # แนะนำให้ลบ vectorstore เก่าทิ้งก่อนรัน เพื่อความสะอาด
    import shutil
    if os.path.exists(VECTORSTORE_DIR):
        logger.warning("🗑️ Deleting existing vectorstore folder: %s", VECTORSTORE_DIR)
        shutil.rmtree(VECTORSTORE_DIR)
    
    ingest_all_documents_recursive()