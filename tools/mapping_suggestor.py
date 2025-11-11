# mapping_suggestor.py
import os
import sys
import logging
from typing import Dict, Any
# 💡 แก้ไข: ย้าย SystemMessage และ HumanMessage ไปที่ langchain_core.messages
from langchain_core.messages import SystemMessage, HumanMessage 
from langchain_community.retrievers import ContextualCompressionRetriever


# -------------------- PATH SETUP --------------------
# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# -------------------- CORE & CONFIG IMPORTS --------------------
try:
    from config.global_vars import (
        INITIAL_TOP_K, 
        STATEMENT_COLLECTION_NAME, # 🟢 ต้องมั่นใจว่า STATEMENT_COLLECTION_NAME อยู่ใน global_vars.py
    )
    from core.vectorstore import (
        VectorStoreManager,
        get_reranking_compressor, # 🟢 ต้องมั่นใจว่า get_reranking_compressor อยู่ใน core/vectorstore.py
    )
except ImportError as e:
    print(f"FATAL ERROR: Failed to import required core modules. Error: {e}", file=sys.stderr)
    sys.exit(1)

# -------------------- LOGGING --------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ------------------------------------------------------------------
# Statement Retrieval for Mapping Suggestion (Pure RAG) 🟢 NEW FUNCTION
# ------------------------------------------------------------------

def find_statements_by_document_content(
    document_content_as_query: str, 
    enabler_abbr: str,
    top_k_reranked: int = 10,
) -> Dict[str, Any]:
    """
    ใช้เนื้อหาเอกสารใหม่เป็น Query เพื่อค้นหา Statements ที่เกี่ยวข้องที่สุดจาก 
    STATEMENT_COLLECTION_NAME (Pure Semantic Search + Reranking)
    """
    
    # 📌 ตรวจสอบความพร้อมของ VectorStoreManager
    if 'VectorStoreManager' not in globals() and 'VectorStoreManager' not in locals():
        logger.error("❌ VectorStoreManager class is not available.")
        return {"suggested_statements": []}
        
    global INITIAL_TOP_K
    # 📌 FIX: ตรวจสอบและกำหนดค่าเริ่มต้นที่ปลอดภัย
    initial_k = INITIAL_TOP_K if isinstance(INITIAL_TOP_K, int) and INITIAL_TOP_K > 0 else 15
        
    try:
        manager = VectorStoreManager()
        
        # 1. สร้าง Filter (Optional: กรองตาม Enabler)
        # ใช้ Metadata 'Enabler_Abbr' ที่จะถูกใส่ตอน Ingest
        where_clause = {"Enabler_Abbr": enabler_abbr.upper()}
        
        # 2. สร้าง Base Retriever
        search_kwargs = {"k": initial_k}
        if where_clause:
            search_kwargs["filter"] = where_clause
            
        # 🛑 โหลด Vector Store สำหรับ Statements
        # 📌 ASSUMPTION: VectorStoreManager มีเมธอด _load_chroma_instance
        vectorstore = manager._load_chroma_instance(STATEMENT_COLLECTION_NAME) 
        if vectorstore is None:
             logger.error(f"❌ Vectorstore '{STATEMENT_COLLECTION_NAME}' not found or failed to load.")
             return {"suggested_statements": []}
             
        base_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs=search_kwargs)
        
        # 3. ใช้ Reranker/Compression
        compressor = get_reranking_compressor(top_n=top_k_reranked) 
        
        compressed_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=base_retriever
        )
        
        # 4. Invoke Compressed Retriever (ใช้เนื้อหาเอกสารใหม่เป็น Query ตรงๆ)
        documents = compressed_retriever.invoke(document_content_as_query)
        logger.info(f"RAG Statement Retrieval found {len(documents)} suggested statements (k={initial_k}->{top_k_reranked}).")

        # 5. จัดรูปแบบผลลัพธ์
        suggested_statements = []
        for doc in documents:
            metadata = doc.metadata or {}
            
            # ดึง relevance_score จาก metadata ที่ Reranker ใส่เข้ามา
            # 📌 REVISED: ใช้ .get() เพื่อความปลอดภัยในการเข้าถึง
            relevance_score = doc.metadata.get("relevance_score", 0.0) 
            
            suggested_statements.append({
                "statement_id": metadata.get("Statement_ID", "N/A"),
                "statement_text": doc.page_content,
                "sub_criteria_id": metadata.get("Sub_Criteria_ID", "N/A"),
                "level": metadata.get("Level", "N/A"),
                "relevance_score": relevance_score, 
            })

        return {"suggested_statements": suggested_statements}
        
    except Exception as e:
        logger.error(f"Error during statement retrieval: {e}", exc_info=True)
        return {"suggested_statements": []}