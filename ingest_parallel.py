# core/ingest_parallel.py
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from core.ingest import process_document, DATA_DIR, SUPPORTED_TYPES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ingest_all_documents_parallel(max_workers: int = 4):
    """
    Parallel ingestion ของไฟล์ทั้งหมดใน DATA_DIR
    - ใช้ ThreadPoolExecutor
    - รองรับไฟล์ตาม SUPPORTED_TYPES
    """
    logger.info("🚀 Starting parallel ingestion for all documents in %s", DATA_DIR)

    if not os.path.exists(DATA_DIR):
        logger.warning("⚠️ DATA_DIR %s does not exist", DATA_DIR)
        return

    # เตรียม list ของไฟล์ที่จะ ingest
    files_to_process = []
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_TYPES:
                filepath = os.path.join(root, f)
                files_to_process.append(filepath)

    if not files_to_process:
        logger.info("⚠️ No files found for ingestion in %s", DATA_DIR)
        return

    results = []

    # ใช้ ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(process_document, f, os.path.basename(f)): f
            for f in files_to_process
        }

        for future in as_completed(future_to_file):
            f = future_to_file[future]
            try:
                doc_id = future.result()
                results.append({"file": f, "doc_id": doc_id, "status": "processed"})
                logger.info("✅ Successfully ingested %s -> %s", f, doc_id)
            except Exception as e:
                results.append({"file": f, "doc_id": None, "status": "failed", "error": str(e)})
                logger.error("❌ Error ingesting %s: %s", f, str(e))

    logger.info("✅ Parallel ingestion completed for %d files", len(results))
    return results


if __name__ == "__main__":
    ingest_all_documents_parallel(max_workers=4)
