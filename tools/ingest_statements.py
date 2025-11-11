# tools/ingest_statements.py (Final Code - Logic Self-Contained with Debugging)
import os
import sys
import argparse
import logging
from typing import Dict, Any, List

# --- Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
    
# --- Imports ---
try:
    from assessments.enabler_assessment import EnablerAssessment
    from config.global_vars import STATEMENT_COLLECTION_NAME 
    
    # NEW: Import dependencies for Chroma management and UUID
    import uuid 
    from langchain_chroma import Chroma 
    
    # 📌 Import Logic from Core Files
    from core.ingest import get_vectorstore 
    from core.vectorstore import VectorStoreManager 
except ImportError as e:
    print(f"FATAL ERROR: Failed to import required modules. Check sys.path and file structure. Error: {e}", file=sys.stderr)
    sys.exit(1)

# --- Logging ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def ingest_all_statements(enabler_abbr: str):
    """
    ดึง Statements ทั้งหมดของ Enabler ที่กำหนด, ลบ Collection เก่า, และ Ingest เข้า Vector Store ใหม่
    """
    logger.info(f"--- Starting Statement Ingestion for Enabler: {enabler_abbr.upper()} ---")
    
    try:
        # 1. โหลด Statements ทั้งหมด
        assessor = EnablerAssessment(enabler_abbr=enabler_abbr, vectorstore_retriever=None)
        all_statements_data: List[Dict[str, Any]] = assessor.get_all_statements()
        
        if not all_statements_data:
            logger.error(f"❌ Found 0 statements for Enabler {enabler_abbr}. Check your JSON data files.")
            return
            
        logger.info(f"✅ Loaded {len(all_statements_data)} statements from {enabler_abbr.upper()} data.")

        # 2. เตรียมข้อมูลสำหรับ Vector Store
        texts = []
        metadatas = []
        
        for statement in all_statements_data:
            texts.append(statement["Statement_Text"])
            metadata = {
                "Statement_ID": statement["Statement_ID"],
                "Sub_Criteria_ID": statement["Sub_Criteria_ID"],
                "Level": statement["Level"],
                "Enabler_Abbr": statement["Enabler_Abbr"],
            }
            metadatas.append(metadata)

        # 3. Ingest Logic (Self-Contained in this tool script)
        logger.info(f"Starting ingestion process into collection: {STATEMENT_COLLECTION_NAME}...")
        
        if not texts:
            logger.warning("No texts provided for statement ingestion. Skipping.")
            return

        # 3.1. Access raw client for safe deletion
        client = None
        try:
            # 🟢 DEBUG CHECKPOINT 1: เพื่อระบุว่าโค้ดถึงจุดนี้แล้ว
            logger.info("CHECKPOINT 1: Initializing VectorStoreManager to access Chroma client...") 
            vsm = VectorStoreManager()
            client = vsm.client 
            logger.info("CHECKPOINT 1.1: VectorStoreManager initialized successfully.") 
        except Exception as e:
            # 📌 FIX: ใช้ exc_info=True เพื่อแสดง Stack Trace ของข้อผิดพลาด
            logger.error(f"❌ Could not initialize VSM or access client for deletion: {e}. Skipping collection deletion.", exc_info=True) 
            client = None

        # 3.2. Delete the old collection using raw client
        if client:
            try:
                # 🛑 ใช้ Raw Client ในการลบ Collection
                client.delete_collection(name=STATEMENT_COLLECTION_NAME)
                logger.info(f"🧹 Successfully deleted existing collection: {STATEMENT_COLLECTION_NAME} for fresh ingest.")
            except Exception as e:
                logger.warning(f"Could not delete collection {STATEMENT_COLLECTION_NAME} (likely did not exist): {e}")

        # 3.3. Get the LangChain Chroma instance 
        vectorstore: Chroma = get_vectorstore(STATEMENT_COLLECTION_NAME)
        
        # 3.4. Add new statements
        ids = [str(uuid.uuid4()) for _ in texts] # สร้าง ID ใหม่สำหรับแต่ละ Statement

        try:
            vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            logger.info(f"✅ Indexed {len(ids)} new statements into collection: {STATEMENT_COLLECTION_NAME}. Persist finished.")
        except Exception as e:
            logger.error(f"❌ Error during Chroma indexing for {STATEMENT_COLLECTION_NAME}: {e}", exc_info=True)
            return
        
        logger.info(f"🎉 Statement Ingestion for {enabler_abbr.upper()} completed successfully!")

    except Exception as e:
        logger.error(f"❌ FATAL Error during statement ingestion for {enabler_abbr.upper()}: {e}", exc_info=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ingest Statements into Vector Store for Mapping Suggestion Tool.")
    parser.add_argument('--enabler', type=str, required=True, help="Enabler abbreviation (e.g., KM, LDR, SUC).")
    
    args = parser.parse_args()
    
    ingest_all_statements(args.enabler)