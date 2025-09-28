import os
import shutil
from typing import Set, List
# FIXED: เปลี่ยนจาก ingest_all_files เป็น process_document เพื่อการรันแบบเรียงลำดับ
from core.ingest import process_document, delete_document, DATA_DIR 

# กำหนดโฟลเดอร์ที่ต้องการยกเว้น
EXCLUDED_FOLDERS: Set[str] = {'evidence', 'qa', 'rubrics', 'feedback'}

def get_top_level_files(data_dir: str, exclude_dirs: Set[str]) -> List[str]:
    """ดึงรายชื่อ Doc ID (ชื่อไฟล์หลัก) ของไฟล์ที่อยู่ในโฟลเดอร์ data/ และไม่ถูกยกเว้น"""
    file_ids = []
    for item in os.listdir(data_dir):
        path = os.path.join(data_dir, item)
        # ข้ามโฟลเดอร์ที่ถูกยกเว้น
        if os.path.isdir(path) and item in exclude_dirs:
            continue
        
        # รวบรวมเฉพาะไฟล์เท่านั้น และใช้ชื่อไฟล์หลักเป็น Doc ID
        if os.path.isfile(path):
            file_id = os.path.splitext(item)[0]
            file_ids.append(file_id)
    return file_ids

def run_batch_ingestion():
    """
    Ingests all files directly in the data/ folder, excluding specific subdirectories.
    **Now running sequentially to avoid segmentation faults on M-series chips (MPS concurrency issues).**
    """
    print("--- Starting Batch Ingestion of Top-Level Documents (Sequential Mode) ---")
    print(f"Skipping documents in subdirectories: {EXCLUDED_FOLDERS}")
    
    files_to_clean_ids = get_top_level_files(DATA_DIR, EXCLUDED_FOLDERS)

    # 1. CRITICAL: Clean up old vectorstores for top-level documents first
    print("\n🧹 Cleaning up old Vector Stores for top-level documents...")
    cleaned_count = 0
    for doc_id in files_to_clean_ids:
        vectorstore_path = os.path.join("vectorstore", doc_id)
        if os.path.exists(vectorstore_path):
            shutil.rmtree(vectorstore_path)
            print(f"  - Deleted old vector store folder: {doc_id}")
            cleaned_count += 1
    
    if cleaned_count == 0:
        print("  - No existing vector stores found to clean.")
    print("🧹 Cleanup complete.")
    
    # 2. Sequential Ingestion
    print("\n🚀 Starting sequential document processing...")
    results = []
    
    for doc_id in files_to_clean_ids:
        # ค้นหาชื่อไฟล์เต็มจาก Doc ID
        full_filename = next(
            (item for item in os.listdir(DATA_DIR) 
             if os.path.isfile(os.path.join(DATA_DIR, item)) and os.path.splitext(item)[0] == doc_id),
            None
        )
        
        if full_filename:
            file_path = os.path.join(DATA_DIR, full_filename)
            print(f"  - Ingesting '{full_filename}'...")
            try:
                # เรียก process_document โดยตรง (Single-Threaded)
                processed_id = process_document(
                    file_path, 
                    full_filename, 
                    collection_id=doc_id # Explicitly pass the correct ID
                )
                results.append({"file": full_filename, "doc_id": processed_id, "status": "processed"})
            except Exception as e:
                print(f"  ❌ Error processing {full_filename}: {e}")
                results.append({"file": full_filename, "doc_id": doc_id, "status": "failed"})


    # 3. Summary
    print("\n--- Ingestion Summary ---")
    for result in results:
        status_icon = "✅" if result['status'] == 'processed' else "❌"
        print(f"{status_icon} {result['file']} -> Doc ID: {result['doc_id']} ({result['status']})")
        
    if not results:
        print("ℹ️ No top-level files found to process.")
        
    print("\n--- Batch Ingestion Complete ---")


if __name__ == "__main__":
    run_batch_ingestion()
