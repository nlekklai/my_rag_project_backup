# test_retrieval.py
import os
import sys
import logging
from typing import List
from langchain.schema import Document

# -------------------- Logging Setup --------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout # บังคับให้ Log ออก Console
)
logger = logging.getLogger(__name__)

# -------------------- Import Core Modules --------------------
# 📌 สำคัญ: ต้องมั่นใจว่าโค้ดถูกรันจาก Root Directory และสามารถ Import ได้
try:
    # สมมติว่า core.vectorstore ถูก Import ได้
    from core.vectorstore import VectorStoreManager, load_all_vectorstores 
except ImportError as e:
    logger.error(f"❌ ไม่สามารถ Import core.vectorstore ได้: {e}")
    logger.error("กรุณาตรวจสอบว่าไฟล์นี้ถูกรันจาก Root Directory ของโปรเจกต์")
    sys.exit(1)

# -------------------- Test Parameters --------------------
TEST_DOC_TYPE = "evidence"
# ID ที่คุณส่งมา: aad986f9-8269-4820-9b79-e3d6b4f55f70 คาดว่าเป็น Chunk UUID
# TEST_CHUNK_UUID = "aad986f9-8269-4820-9b79-e3d6b4f55f70" 
TEST_QUESTION = "สรุปเอกสาร"
TEST_STABLE_DOC_ID = "d9d7ba9b-35af-4bfe-9e8c-7d84467801d8"

def test_vectorstore_retrieval():
    print("\n" + "=" * 60)
    print("--- 1. ตรวจสอบ Current Working Directory (CWD) ---")
    print(f"CWD: {os.getcwd()}")
    print("ต้องเป็นโฟลเดอร์ที่มี 'vectorstore/' อยู่")
    print("=" * 60)

    # ------------------ A. ทดสอบโหลด Vector Store (ที่ทำให้เกิด Error) ------------------
    print("\n" + "-" * 50)
    print("--- 2. ทดสอบ load_all_vectorstores (ฟังก์ชันที่ล้มเหลวใน API) ---")
    print("-" * 50)
    
    try:
        # ฟังก์ชันนี้จะเรียก VectorStoreManager และ vectorstore_exists
        retriever_wrapper = load_all_vectorstores(doc_types=[TEST_DOC_TYPE])
        
        print(f"✅ สำเร็จ! MultiDocRetriever ถูกโหลดด้วย {len(retriever_wrapper._retrievers_list)} Retriever(s) จาก Collection '{TEST_DOC_TYPE}'")
        
        # ------------------ B. ทดสอบ Query ผ่าน Retriever ------------------
        print("\n" + "-" * 50)
        print("--- 3. ทดสอบเรียก Query ผ่าน Retriever (Simulate RAG) ---")
        print("-" * 50)

        # การเรียก invoke จะทำการค้นหาและ Rerank (ถ้ามี)
        results: List[Document] = retriever_wrapper.invoke(TEST_QUESTION)
        
        if results:
            print(f"✅ สำเร็จ! พบ {len(results)} Chunk ที่เกี่ยวข้องกับคำถาม")
            print(f"   - Chunk ที่เกี่ยวข้องที่สุด (Relevance Score): {results[0].metadata.get('relevance_score', 'N/A')}")
            print(f"   - เนื้อหาเริ่มต้น: '{results[0].page_content[:150]}...'")
        else:
            print("⚠️ คำเตือน: Query ไม่พบเอกสารที่เกี่ยวข้อง (อาจเป็นเพราะคำถามไม่ตรง หรือ Embeddings ล้มเหลว)")

    except ValueError as e:
        print(f"❌ ล้มเหลว (load_all_vectorstores): {e}")
        print("⚠️ ปัญหานี้ยืนยันว่าโค้ด 'vectorstore_exists' ใน 'core/vectorstore' ล้มเหลว")
        
    # ------------------ C. ทดสอบดึงเอกสารด้วย UUID โดยตรง ------------------
    print("\n" + "-" * 50)
    print("--- 4. ทดสอบดึงเอกสารด้วย Chunk UUID โดยตรง ---")
    print("-" * 50)
    
    manager = VectorStoreManager() # โหลด Singleton Manager
    try:
        documents = manager.get_chunks_from_doc_ids( # 📌 ใช้ฟังก์ชันนี้แทน
            stable_doc_ids=[TEST_STABLE_DOC_ID], 
            doc_type=TEST_DOC_TYPE
        )
        
        if documents:
            print(f"✅ สำเร็จ! ดึง {len(documents)} Chunk ด้วย Stable ID '{TEST_STABLE_DOC_ID}'")
            print(f"   - เนื้อหาเริ่มต้น: '{documents[0].page_content[:150]}...'")
        else:
            print(f"❌ ล้มเหลว: ดึงเอกสารด้วย Stable ID '{TEST_STABLE_DOC_ID}' ได้ 0 ผลลัพธ์")
            # 📌 ให้ตรวจสอบไฟล์ doc_id_mapping.json ว่ามี ID นี้อยู่จริงหรือไม่
            print(f"   (โปรดตรวจสอบว่า '{TEST_STABLE_DOC_ID}' อยู่ใน {manager._doc_id_mapping.keys()})")
        
    except Exception as e:
        print(f"❌ ข้อผิดพลาดร้ายแรงระหว่างการดึงด้วย Stable ID: {e}")

if __name__ == "__main__":
    test_vectorstore_retrieval()