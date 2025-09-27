import os
from ingest import process_document, delete_document

DATA_DIR = "data"

files = [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]

for file in files:
    file_path = os.path.join(DATA_DIR, file)
    doc_id = os.path.splitext(file)[0]  # ใช้ชื่อไฟล์เป็น doc_id
    
    # ลบ vectorstore เดิมก่อน (ถ้ามี)
    delete_document(doc_id)
    
    try:
        process_document(file_path)
        print(f"✅ Processed {file} -> doc_id: {doc_id}")
    except Exception as e:
        print(f"⚠️ Error processing {file}: {e}")
