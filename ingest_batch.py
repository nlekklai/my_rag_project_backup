# ingest_batch.py
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict, Any
from core.ingest import process_document

# -------------------- Config --------------------
DATA_DIR = "data"
VECTORSTORE_DIR = "vectorstore"
SEQUENTIAL = True  # True = process one by one, False = multi-thread
SUPPORTED_DOC_TYPES = ["evidence"]  # ตัวอย่าง folder/doc_type

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------- Helper --------------------
def get_all_files(data_dir: str = DATA_DIR) -> List[Dict[str, str]]:
    """
    Walk through all subfolders and return files with their doc_type (folder name)
    """
    files = []
    for root, _, filenames in os.walk(data_dir):
        for f in filenames:
            if f.startswith('.'):  # skip hidden files
                continue
            full_path = os.path.join(root, f)
            folder_name = os.path.relpath(root, data_dir)
            doc_type = folder_name if folder_name in SUPPORTED_DOC_TYPES else "default"
            files.append({
                "file_path": full_path,
                "file_name": f,
                "doc_type": doc_type
            })
    return files

# -------------------- Ingestion --------------------
def ingest_all_files(
    sequential: bool = SEQUENTIAL,
    data_dir: str = DATA_DIR,
    base_path: str = VECTORSTORE_DIR,
) -> List[Dict[str, Any]]:
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(base_path, exist_ok=True)
    files_to_process = get_all_files(data_dir)
    results = []

    def _process_file(file_info):
        return process_document(
            file_path=file_info["file_path"],
            file_name=file_info["file_name"],
            doc_type=file_info["doc_type"],
            base_path=base_path
        )

    if sequential:
        for f in files_to_process:
            try:
                doc_id = _process_file(f)
                results.append({
                    "file": f["file_name"],
                    "doc_id": doc_id,
                    "doc_type": f["doc_type"],
                    "status": "processed"
                })
            except Exception as e:
                logger.error(f"Error processing {f['file_name']}: {e}")
                results.append({
                    "file": f["file_name"],
                    "doc_id": None,
                    "doc_type": f["doc_type"],
                    "status": "failed",
                    "error": str(e)
                })
    else:
        with ThreadPoolExecutor() as executor:
            future_to_file = {executor.submit(_process_file, f): f for f in files_to_process}
            for future in as_completed(future_to_file):
                f = future_to_file[future]
                try:
                    doc_id = future.result()
                    results.append({
                        "file": f["file_name"],
                        "doc_id": doc_id,
                        "doc_type": f["doc_type"],
                        "status": "processed"
                    })
                except Exception as e:
                    logger.error(f"Error processing {f['file_name']}: {e}")
                    results.append({
                        "file": f["file_name"],
                        "doc_id": None,
                        "doc_type": f["doc_type"],
                        "status": "failed",
                        "error": str(e)
                    })
    return results

# -------------------- Manual run --------------------
if __name__ == "__main__":
    res = ingest_all_files(sequential=True)
    print("Ingestion results:")
    for r in res:
        print(r)
