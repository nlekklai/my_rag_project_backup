# core/statement_ingester.py

import os
import sys
import json
import logging
from typing import List, Dict, Any

# --- PATH SETUP (Must be executed first for imports to work) ---
try:
    # project_root คือ my_rag_project/
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    # Import Core Functions (ใช้ Relative Import)
    from .vectorstore import save_to_vectorstore, vectorstore_exists, VECTORSTORE_DIR 
    from .ingest import clean_text # สันนิษฐานว่า clean_text อยู่ใน core/ingest.py
    
except ImportError as e:
    print(f"FATAL ERROR: Failed to import required modules. Check sys.path and file structure. Error: {e}", file=sys.stderr)
    sys.exit(1)

# --- CONFIGURATION ---
# 1. Source File Path (แก้ไขให้ชี้ไปที่ my_rag_project/evidence_checklist)
EVIDENCE_CHECKLIST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "evidence_checklist"))
SOURCE_FILE_NAME = "km_evidence_statements_checklist.json"
SOURCE_FILE_PATH = os.path.join(EVIDENCE_CHECKLIST_DIR, SOURCE_FILE_NAME)

# 2. Target Vectorstore Config 
TARGET_DOC_TYPE = "statement"       # ชื่อโฟลเดอร์/doc_type ที่ run_mapping_generator.py คาดหวัง
TARGET_DOC_ID = "km_statements"     # ID ที่ run_mapping_generator.py คาดหวัง
TARGET_VECTORSTORE_PATH = os.path.join(VECTORSTORE_DIR, TARGET_DOC_TYPE, TARGET_DOC_ID)

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# --- MAIN INGESTION FUNCTION ---

def ingest_km_statements(force_recreate: bool = False):
    """
    อ่านไฟล์ km_evidence_statements_checklist.json, 
    ประมวลผล (Flatten), และบันทึกลงใน Vector Store
    ที่ vectorstore/statement/km_statements
    """
    
    # 1. ตรวจสอบไฟล์ต้นทาง
    if not os.path.exists(SOURCE_FILE_PATH):
        logger.error(f"Source file not found: {SOURCE_FILE_PATH}")
        print(f"❌ Error: ไม่พบไฟล์ Statement Checklist ที่พาธ: {SOURCE_FILE_PATH}")
        return

    # 2. ตรวจสอบว่า Vectorstore ถูกสร้างไปแล้วหรือยัง
    if vectorstore_exists(TARGET_DOC_ID, doc_type=TARGET_DOC_TYPE) and not force_recreate:
        logger.warning(f"Vectorstore already exists at {TARGET_VECTORSTORE_PATH}. Skipping ingestion.")
        print(f"ℹ️ Vectorstore สำหรับ {TARGET_DOC_ID} มีอยู่แล้ว. (ใช้ --force เพื่อ re-index)")
        return

    logger.info(f"Loading statements from {SOURCE_FILE_PATH}...")
    try:
        with open(SOURCE_FILE_PATH, 'r', encoding='utf-8') as f:
            all_criteria = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read or parse JSON file: {e}")
        return

    # 3. Flatten Data
    all_statement_texts: List[str] = []
    all_metadata: List[Dict[str, Any]] = []

    logger.info("Flattening JSON data and creating metadata...")
    for criteria in all_criteria:
        enabler_id = criteria.get("KM_Enabler_ID") # เช่น 1
        sub_id = criteria.get("Sub_Criteria_ID")   # เช่น 1.1
        sub_name = criteria.get("Sub_Criteria_Name_TH")
        criteria_weight = criteria.get("Weight") 

        for level in range(1, 6): # วน Level 1-5
            level_key = f"Level_{level}_Statements"
            statements = criteria.get(level_key, [])
            
            for i, statement_text in enumerate(statements):
                if not statement_text or not statement_text.strip():
                    continue

                statement_num = i + 1
                cleaned_text = clean_text(statement_text) 
                
                # 🚨 การตั้งชื่อ Key สำคัญมากสำหรับ RAG/Generator Logic
                metadata = {
                    # --- Keys ที่ run_mapping_generator.py และ RAG คาดหวัง ---
                    "statement_key": f"{sub_id}_L{level}_{statement_num}", # e.g., 1.1_L1_1
                    "sub_id": sub_id,                                      # e.g., 1.1
                    "sub_name": sub_name,                                  # (สำหรับ Justification)
                    "level": level,
                    "statement_number": statement_num,
                    
                    # --- RAG Standard Keys ---
                    "doc_type": TARGET_DOC_TYPE, 
                    "doc_id": TARGET_DOC_ID,     
                    "source": SOURCE_FILE_NAME,  
                    "enabler_id": f"KM-{enabler_id}",
                    "criteria_weight": criteria_weight, 
                    # --------------------------------------------------------
                }
                
                all_statement_texts.append(cleaned_text)
                all_metadata.append(metadata)

    if not all_statement_texts:
        logger.warning("No statements were extracted from the JSON.")
        print("⚠️ ไม่พบ Statement ที่สามารถประมวลผลได้ในไฟล์ JSON")
        return

    # 4. บันทึกลง Vector Store
    try:
        logger.info(f"Extracted {len(all_statement_texts)} statements. Saving to vectorstore at {TARGET_VECTORSTORE_PATH}...")
        
        # 🛑 แก้ไข: ลบ base_path ออก
        save_to_vectorstore(
            doc_id=TARGET_DOC_ID,
            texts=all_statement_texts,
            metadatas=all_metadata, 
            doc_type=TARGET_DOC_TYPE # 'statement'
        )
        
        logger.info(f"✅ Successfully ingested statements into {TARGET_VECTORSTORE_PATH}")
        print(f"✅ สร้าง Vector Store สำเร็จ! ({len(all_statement_texts)} statements)")

    except Exception as e:
        logger.error(f"Failed to save vectorstore for {TARGET_DOC_ID}: {e}", exc_info=True)
        print(f"❌ เกิดข้อผิดพลาดระหว่างบันทึก Vector Store: {e}")

# -------------------- CLI Entry Point --------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest KM Statements Checklist into Vector Store.")
    parser.add_argument("--force", action="store_true", help="Force re-ingestion, overwriting existing vectorstore.")
    args = parser.parse_args()
    
    ingest_km_statements(force_recreate=args.force)