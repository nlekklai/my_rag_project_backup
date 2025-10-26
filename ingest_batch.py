import os
import sys
import logging
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict, Any

# 🚨 ปิด Warning Tokenizer Parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- PATH SETUP ---
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Imports ---
from core.vectorstore import get_vectorstore
from core.ingest import process_document 

# -------------------- Config --------------------
DATA_DIR = "data"
VECTORSTORE_DIR = "vectorstore"
SEQUENTIAL = True  # True = process one by one, False = multi-thread
SUPPORTED_DOC_TYPES = ["evidence", "faq", "document"]
DEFAULT_DOC_TYPES_TO_RUN = ["document", "faq"]

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# -------------------- Helper --------------------
def get_all_files(data_dir: str = DATA_DIR) -> List[Dict[str, str]]:
    files = []
    if not os.path.isdir(data_dir):
        logger.warning(f"Data directory '{data_dir}' not found. Creating it now.")
        os.makedirs(data_dir, exist_ok=True)
        return files

    for root, _, filenames in os.walk(data_dir):
        folder_name = os.path.relpath(root, data_dir)
        for f in filenames:
            if f.startswith('.'):
                continue
            full_path = os.path.join(root, f)
            doc_type = folder_name if folder_name in SUPPORTED_DOC_TYPES else "default"
            files.append({
                "file_path": full_path,
                "file_name": f,
                "doc_type": doc_type
            })
    return files

# -------------------- Status Update --------------------
def mark_ingested(doc_type: str, doc_id: str):
    """
    อัปเดตสถานะไฟล์เป็น Ingested ใน JSON metadata (uploads/<doc_type>_uploads.json)
    """
    metadata_dir = "uploads"
    os.makedirs(metadata_dir, exist_ok=True)
    metadata_path = os.path.join(metadata_dir, f"{doc_type}_uploads.json")

    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = []

    updated = False
    for doc in data:
        if doc.get('doc_id') == doc_id:
            doc['status'] = 'Ingested'
            updated = True
            break

    if not updated:
        # ถ้าไฟล์นี้ยังไม่เคยอยู่ใน metadata ให้เพิ่มเข้าไป
        data.append({
            "doc_id": doc_id,
            "filename": doc_id,  # หรือใส่ชื่อจริงของไฟล์ถ้ามี
            "file_type": os.path.splitext(doc_id)[1] if '.' in doc_id else '',
            "status": "Ingested"
        })

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# -------------------- Ingestion --------------------
def ingest_all_files(
    sequential: bool = SEQUENTIAL,
    data_dir: str = DATA_DIR,
    base_path: str = VECTORSTORE_DIR,
    doc_types_to_process: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    
    vector_service = get_vectorstore()
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(base_path, exist_ok=True)

    files_to_process = get_all_files(data_dir)
    results = []

    # Filter files by doc_type
    if doc_types_to_process:
        initial_count = len(files_to_process)
        files_to_process = [f for f in files_to_process if f["doc_type"] in doc_types_to_process]
        logger.info(
            f"Filtering enabled. Kept {len(files_to_process)} files out of {initial_count} "
            f"matching types: {doc_types_to_process}"
        )

    if not files_to_process:
        logger.warning("No files found to process after filtering. Exiting ingestion.")
        return results

    def _process_file(file_info):
        return process_document(
            file_path=file_info["file_path"],
            file_name=file_info["file_name"],
            doc_type=file_info["doc_type"],
            base_path=base_path
        )

    if sequential:
        logger.info("Running ingestion in SEQUENTIAL mode.")
        for f in files_to_process:
            try:
                doc_id = _process_file(f)
                mark_ingested(f["doc_type"], doc_id)  # ✅ อัปเดต status
                results.append({
                    "file": f["file_name"],
                    "doc_id": doc_id,
                    "doc_type": f["doc_type"],
                    "status": "processed"
                })
            except Exception as e:
                logger.error(f"Error processing {f['file_name']}: {e}", exc_info=True)
                results.append({
                    "file": f["file_name"],
                    "doc_id": None,
                    "doc_type": f["doc_type"],
                    "status": "failed",
                    "error": str(e)
                })
    else:
        logger.info("Running ingestion in MULTI-THREAD mode.")
        with vector_service.executor as executor:
            future_to_file = {executor.submit(_process_file, f): f for f in files_to_process}
            for future in as_completed(future_to_file):
                f = future_to_file[future]
                try:
                    doc_id = future.result()
                    mark_ingested(f["doc_type"], doc_id)
                    results.append({
                        "file": f["file_name"],
                        "doc_id": doc_id,
                        "doc_type": f["doc_type"],
                        "status": "processed"
                    })
                except Exception as e:
                    logger.error(f"Error processing {f['file_name']}: {e}", exc_info=True)
                    results.append({
                        "file": f["file_name"],
                        "doc_id": None,
                        "doc_type": f["doc_type"],
                        "status": "failed",
                        "error": str(e)
                    })
    return results

# -------------------- Main --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG Batch Ingestion.")
    parser.add_argument(
        "doc_types",
        nargs='*',
        default=DEFAULT_DOC_TYPES_TO_RUN,
        help="Specify document types to ingest (e.g., 'document faq'). Defaults to: " + str(DEFAULT_DOC_TYPES_TO_RUN)
    )
    args = parser.parse_args()

    # Validate doc types
    if args.doc_types and all(dt in SUPPORTED_DOC_TYPES for dt in args.doc_types):
        DOC_TYPES_TO_RUN = args.doc_types
    else:
        DOC_TYPES_TO_RUN = DEFAULT_DOC_TYPES_TO_RUN

    if not all(dt in SUPPORTED_DOC_TYPES for dt in args.doc_types):
        invalid_types = [dt for dt in args.doc_types if dt not in SUPPORTED_DOC_TYPES]
        if invalid_types:
            logger.error(f"Invalid document types specified: {invalid_types}. Using default: {DEFAULT_DOC_TYPES_TO_RUN}")

    logger.info(f"--- Starting Ingestion Batch Process (Target types: {DOC_TYPES_TO_RUN}) ---")
    vector_service_instance = None

    try:
        vector_service_instance = get_vectorstore()
        res = ingest_all_files(sequential=SEQUENTIAL, doc_types_to_process=DOC_TYPES_TO_RUN)

        print("\n--- Ingestion Summary Results ---")
        for r in res:
            print(f"[{r['status'].upper():<9}] Type: {r['doc_type']:<10} | File: {r['file']} | Doc ID: {r['doc_id']}")
            if r['status'] == 'failed':
                print(f"   -> ERROR: {r['error']}")

    except Exception as e:
        logger.critical(f"A critical error occurred in main execution: {e}", exc_info=True)
    finally:
        if vector_service_instance:
            vector_service_instance.close()
        logger.info("--- Ingestion Batch Process Finished ---")
