import os
import sys
# เพิ่ม root project ให้ Python หา core ได้
# Ensure the root project path is included for module discovery
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary functions from core.ingest and core.vectorstore
# 💡 เพิ่ม VECTORSTORE_DIR และ vectorstore_exists เพื่อใช้ในการตรวจสอบ
from core.ingest import ingest_all_files, DATA_DIR, VECTORSTORE_DIR 
from core.vectorstore import vectorstore_exists 

def main():
    # 1. Define the intended document type (and source data subfolder)
    folder_type = "document" 
    folder_path = os.path.join(DATA_DIR, folder_type)
    
    if not os.path.exists(folder_path):
        print(f"⚠️ Folder not found: {folder_path}")
        return
        
    print(f"📂 Ingesting files from source data folder: {folder_path}")
    
    # 2. IMPORTANT: Pass the folder_type as the doc_type argument
    # This ensures the vectorstore is saved to vectorstore/faq/...
    results = ingest_all_files(
        data_dir=folder_path,
        doc_type=folder_type, # <-- FIX: Pass the document type
        exclude_dirs=set(),
        sequential=True,
        version="v1"
    )
    
    print("\n✅ Ingestion results:")
    for r in results:
        doc_id = r.get('doc_id')
        
        if r['status'] == 'processed' and doc_id:
            # New check: Verify existence and print exact path
            is_created = vectorstore_exists(doc_id, doc_type=folder_type)
            vectorstore_path = os.path.join(VECTORSTORE_DIR, folder_type, doc_id)
            
            if is_created:
                 # แสดงพาธที่สร้างสำเร็จ
                 print(f"   -> File: {r['file']} | Doc ID: {doc_id} | Status: {r['status']} | Location: {vectorstore_path}")
            else:
                 # แจ้งเตือนหาก Ingestion สำเร็จ แต่ไม่พบไฟล์ Vectorstore
                 print(f"   -> File: {r['file']} | Doc ID: {doc_id} | Status: {r['status']} | ⚠️ ERROR: Vectorstore not found at {vectorstore_path}")

        else:
            print(f"   -> File: {r['file']} | Status: {r['status']} | Error: {r.get('error', 'N/A')}")
            
if __name__ == "__main__":
    main()
