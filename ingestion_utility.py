import os
import shutil
# สมมติว่า core.ingest มีฟังก์ชัน process_document และ delete_document
from core.ingest import process_document, delete_document

# กำหนดโฟลเดอร์หลักสำหรับเอกสารประเมินผล (ต้องตรงกับโครงสร้างหลักของโปรเจกต์)
ASSESSMENT_DIR = "assessment_data"

# โฟลเดอร์สำหรับเอกสาร 4 ประเภทหลัก (ชื่อ Key คือชื่อ Vectorstore Collection ID)
FOLDERS = {
    "rubrics": os.path.join(ASSESSMENT_DIR, "rubrics"), 
    "qa": os.path.join(ASSESSMENT_DIR, "qa"),           
    "feedback": os.path.join(ASSESSMENT_DIR, "feedback"), 
    "evidence": os.path.join(ASSESSMENT_DIR, "evidence"), 
}
VECTORSTORE_DIR = "vectorstore" # โฟลเดอร์เก็บ Vectorstore

# -------------------------------------------------------------
print(f"--- Starting Document Ingestion for Assessment Workflow ({ASSESSMENT_DIR}) ---")

# 1. ลบ Vectorstore เก่าทั้งหมดตามชื่อโฟลเดอร์หลัก (เพื่อความมั่นใจว่า Ingestion ใหม่)
print("\n🧹 Cleaning up old vector stores...")
# ใช้ชื่อโฟลเดอร์ (e.g., 'rubrics', 'qa') เป็น doc_id ที่ต้องการลบ
for folder_name in FOLDERS.keys():
    # การลบต้องใช้ doc_id ซึ่งในกรณีนี้คือชื่อ Collection (e.g., 'rubrics')
    try:
        delete_document(folder_name) 
        print(f"  - Deleted old vector store for collection: {folder_name}")
    except Exception:
        # หากลบไม่ได้ (อาจไม่เคยมีอยู่) ให้ข้ามไป
        pass
            
print("🧹 Cleanup complete.")
# -------------------------------------------------------------

files_processed_count = 0
for folder_key, folder_path in FOLDERS.items():
    # ตรวจสอบว่าโฟลเดอร์มีอยู่
    if not os.path.exists(folder_path):
        print(f"⚠️ Warning: Directory '{folder_path}' not found. Skipping.")
        continue
        
    print(f"\nProcessing Folder: '{folder_key}' (Vectorstore ID: {folder_key})...")
    
    # ค้นหาไฟล์ทั้งหมดในโฟลเดอร์ย่อย
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and not f.startswith('.')]
    
    if not files:
        print(f"  ℹ️ No files found in the {folder_key} directory.")
        
    for file in files:
        file_path = os.path.join(folder_path, file)
        
        print(f"  - Ingesting '{file}'...")
        
        # 2. ประมวลผลเอกสารโดยใช้ Logic ใหม่
        try:
            # process_document จะใช้ folder_key ('rubrics') เป็น vectorstore ID
            # และส่งชื่อไฟล์ ('file') เข้าไปด้วยตามที่ core.ingest ต้องการ
            process_document(file_path, file) 
            print(f"  ✅ Successfully indexed '{file}' into collection '{folder_key}'")
            files_processed_count += 1
        except Exception as e:
            print(f"  ❌ Critical Error processing {file}: {e}")

print(f"\n--- Document Ingestion Complete. Total files indexed: {files_processed_count} ---")
