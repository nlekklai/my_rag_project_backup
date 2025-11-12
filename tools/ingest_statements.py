# tools/ingest_statements.py
# บทบาท: นำเข้า Statements (เกณฑ์การประเมิน) ทั้งหมดสำหรับ Enabler ที่กำหนด
# เข้าสู่ Vector Store (Chroma) ใน Collection เฉพาะ เช่น statement_KM

import os
import sys
import argparse
import logging
import uuid
import time
from typing import Dict, Any, List

# -------------------- PATH SETUP --------------------
# กำหนดพาธรูทของโปรเจกต์
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# -------------------- Global Vars (จำเป็นต้องใช้) --------------------
try:
    # SUPPORTED_DOC_TYPES อาจจำเป็นสำหรับการสร้าง Collection Name ใน _get_collection_name
    from config.global_vars import SUPPORTED_DOC_TYPES 
except ImportError as e:
    print(f"FATAL ERROR: Cannot import global_vars: {e}", file=sys.stderr)
    sys.exit(1)
    
# -------------------- Core & Assessment Imports --------------------
try:
    # นำเข้า EnablerAssessment ซึ่งมีเมธอด get_statements() ที่แก้ไขแล้ว
    from assessments.enabler_assessment import EnablerAssessment
    # นำเข้า VectorStoreManager และฟังก์ชันสำหรับชื่อ Collection
    from core.vectorstore import VectorStoreManager, _get_collection_name 
except ImportError as e:
    # แจ้งเตือนเมื่อเกิด ImportError
    print(f"FATAL ERROR: Failed to import required modules (EnablerAssessment/VectorStore): {e}", file=sys.stderr)
    sys.exit(1)

# -------------------- Logging --------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 🎯 Constant สำหรับ Statements
STATEMENT_DOC_TYPE = "statement"

def ingest_all_statements(enabler_abbr: str):
    """
    ดึง Statements ทั้งหมดของ Enabler ที่กำหนด, ลบ Collection เก่า, และ Ingest เข้า Vector Store ใหม่
    ใช้ Dynamic Collection Name: statement_<enabler_abbr>
    """
    start_time = time.perf_counter()
    enabler_abbr = enabler_abbr.upper()
    logger.info(f"--- Starting Statement Ingestion for Enabler: {enabler_abbr} ---")
    
    # 1. 🎯 กำหนดชื่อ Collection แบบ Dynamic
    collection_name = _get_collection_name(doc_type=STATEMENT_DOC_TYPE, enabler=enabler_abbr)
    
    try:
        # 2. โหลด Statements ทั้งหมด
        # สร้าง Assessor โดยไม่จำเป็นต้องโหลด Vector Store เข้ามา (vectorstore_retriever=None)
        # Assessor จะโหลดพาธไฟล์ Statement JSON ที่จำเป็นเอง
        assessor = EnablerAssessment(enabler_abbr=enabler_abbr, vectorstore_retriever=None)
        
        # 📌 ใช้เมธอด get_statements() เพื่อดึงข้อมูล Statements ทั้งหมด
        all_statements_data: List[Dict[str, Any]] = assessor.get_statements()
            
        # 2.1. ตรวจสอบข้อมูลที่โหลด
        if not all_statements_data:
            logger.error(f"❌ Found 0 statements for Enabler {enabler_abbr}. Check your JSON data files.")
            return
            
        logger.info(f"✅ Loaded {len(all_statements_data)} statements from {enabler_abbr} data.")

        # 3. เตรียมข้อมูลสำหรับ Vector Store
        texts = []
        metadatas = []
        
        for statement in all_statements_data:
            # ตรวจสอบว่ามี Statement_Text หรือไม่
            statement_text = statement.get("Statement_Text", "").strip()
            if not statement_text:
                logger.warning(f"Skipping statement (ID: {statement.get('Statement_ID', 'N/A')}) because 'Statement_Text' is missing or empty.")
                continue

            texts.append(statement_text)
            
            # เตรียม Metadata ที่จำเป็นสำหรับการค้นหา
            metadata = {
                "Statement_ID": statement.get("Statement_ID"),
                "Sub_Criteria_ID": statement.get("Sub_Criteria_ID"),
                "Level": statement.get("Level"),
                "Enabler_Abbr": statement.get("Enabler_Abbr", enabler_abbr),
                "doc_type": STATEMENT_DOC_TYPE, # ใช้กรองประเภทเอกสาร
                "enabler": enabler_abbr,        # ใช้กรอง Enabler
            }
            metadatas.append(metadata)

        # 4. Ingest Logic (ใช้ VectorStoreManager Public Methods)
        logger.info(f"Starting ingestion process into dynamic collection: {collection_name}...")
        
        if not texts:
            logger.warning("No valid texts provided for statement ingestion. Skipping.")
            return

        # 4.1. Initialize VSM and Delete the old collection
        try:
            logger.info("CHECKPOINT 1: Initializing VectorStoreManager and deleting old collection...") 
            vsm = VectorStoreManager()
            
            # 🎯 ลบ Collection เก่าเพื่อทำ Fresh Ingest
            if vsm.delete_collection(collection_name):
                 logger.info(f"🧹 Successfully deleted existing collection: {collection_name} for fresh ingest.")
            else:
                 logger.warning(f"Could not delete collection {collection_name} (likely did not exist). Proceeding.")

        except Exception as e:
            logger.error(f"❌ Could not initialize VSM or delete collection: {e}. Aborting ingestion.", exc_info=True) 
            return

        # 4.2. Get the LangChain Chroma instance 
        vectorstore = vsm.get_chroma_instance(collection_name) 
        
        if not vectorstore:
             logger.error(f"❌ Could not get/create Chroma instance for collection: {collection_name}. Aborting ingestion.")
             return

        # 4.3. Add new statements
        ids = [str(uuid.uuid4()) for _ in texts] # สร้าง ID ใหม่สำหรับแต่ละ Statement

        try:
            # 🟢 ใช้ vectorstore.add_texts() เพื่อ Ingest และสร้าง Embeddings
            vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            
            end_time = time.perf_counter()
            runtime = round(end_time - start_time, 2)
            
            logger.info(f"✅ Indexed {len(ids)} new statements into collection: {collection_name}. Persist finished.")
            logger.info(f"🎉 Statement Ingestion for {enabler_abbr} completed successfully in {runtime}s into {collection_name}!")
        except Exception as e:
            logger.error(f"❌ Error during Chroma indexing for {collection_name}: {e}", exc_info=True)
            return
        
    except Exception as e:
        logger.error(f"❌ FATAL Error during statement ingestion for {enabler_abbr}: {e}", exc_info=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ingest Statements into Vector Store for Mapping Suggestion Tool.")
    parser.add_argument('--enabler', 
                        type=str, 
                        required=True, 
                        choices=["CG", "L", "SP", "RM&IC", "SCM", "DT", "HCM", "KM", "IM", "IA"],
                        help="Enabler abbreviation (e.g., KM, LDR, SUC).")
    
    args = parser.parse_args()
    
    ingest_all_statements(args.enabler)