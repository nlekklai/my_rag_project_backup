import os
import sys
import logging
import argparse 
from typing import List, Optional, Any, Tuple
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever 
import glob # ADDED: สำหรับการนับไฟล์ต้นฉบับ

# -------------------- Logging Setup --------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout # บังคับให้ Log ออก Console
)
logger = logging.getLogger(__name__)

# -------------------- Import Core Modules --------------------
try:
    # สมมติว่า core.vectorstore ถูก Import ได้ และมี MultiDocRetriever, NamedRetriever อยู่
    from core.vectorstore import VectorStoreManager, load_all_vectorstores, MultiDocRetriever, NamedRetriever 
    from core.ingest import SUPPORTED_DOC_TYPES 
except ImportError as e:
    logger.error(f"❌ ไม่สามารถ Import core.vectorstore ได้: {e}")
    logger.error("กรุณาตรวจสอบว่าไฟล์นี้ถูกรันจาก Root Directory ของโปรเจกต์")
    sys.exit(1)

# -------------------- Argument Parsing --------------------
def parse_arguments():
    """Parses command line arguments for retrieval testing."""
    parser = argparse.ArgumentParser(description="Test RAG Retrieval from Vector Stores.")
    
    parser.add_argument(
        "doc_type",
        nargs='?', 
        default="document", 
        help=f"Document type to test (default: document, supported: {SUPPORTED_DOC_TYPES})",
    )
    
    parser.add_argument(
        "--enabler",
        type=str,
        default=None, 
        help="Specific enabler code for 'evidence' type (e.g., KM, L).",
    )
    
    parser.add_argument(
        "--doc_id",
        type=str,
        default=None,
        help="Stable Document UUID to test direct chunk retrieval (optional).",
    )
    
    parser.add_argument(
        "--query",
        type=str,
        default="วิสัยทัศน์ของการไฟฟ้าส่วนภูมิภาคคืออะไร",
        help="The query/question to use for the similarity search test.",
    )
    
    return parser.parse_args()

def test_count_vectors(retriever_wrapper: MultiDocRetriever, collection_name: str):
    """
    Test the total count of vectors (chunks) and unique documents 
    in the specified collection โดยใช้ retriever_wrapper ที่โหลดมาแล้ว
    """
    print("\n" + "-" * 50)
    print(f"--- 3. ทดสอบนับจำนวน Vector (Chunks) และ Unique Documents ใน Collection '{collection_name}' ---")
    print("-" * 50)
    
    try:
        if not retriever_wrapper._retrievers_list:
             print("⚠️ คำเตือน: MultiDocRetriever ไม่พบ Retriever ที่ถูกโหลด")
             return

        # 1. เข้าถึง NamedRetriever ตัวแรก (ซึ่งควรเป็นตัวเดียวที่ถูกโหลดในกรณีนี้)
        named_retriever_instance: NamedRetriever = retriever_wrapper._retrievers_list[0]

        # 2. โหลด BaseRetriever (ContextualCompressionRetriever หรือ ChromaRetriever)
        retriever_instance: BaseRetriever = named_retriever_instance.load_instance()
        
        # 3. ดึง Vector Store Instance (Chroma Object)
        base_retriever = getattr(retriever_instance, 'base_retriever', None)
        vectorstore = getattr(base_retriever, 'vectorstore', None) 
        
        # Fallback: ถ้าเป็น BaseRetriever ธรรมดา (ไม่ได้ใช้ Reranker)
        if not vectorstore:
            vectorstore = getattr(retriever_instance, 'vectorstore', None)


        if vectorstore and hasattr(vectorstore, 'get'):
            
            # 📌 FIX: แก้ปัญหา "Expected where to have exactly one operator, got {} in get."
            query_where = {
                "doc_id": {
                    "$ne": "non_existent_doc_id_placeholder_to_force_query"
                }
            }
            
            # 🚩 MODIFIED: Request metadatas เพื่อนำมานับ Unique Documents
            count_result = vectorstore.get(
                where=query_where,  # ใช้เงื่อนไขที่ถูกต้องตาม Chroma
                limit=None, 
                include=['metadatas'] # MODIFIED: ขอ metadatas เพื่อดึง doc_id
            )
            
            total_count = len(count_result['ids'])
            
            # --- NEW: Count Unique Documents ---
            # ดึงเฉพาะค่า doc_id จาก metadatas ที่ดึงมาทั้งหมด
            all_doc_ids = [m.get('doc_id') for m in count_result.get('metadatas', []) if m.get('doc_id')]
            
            # นับจำนวน doc_id ที่ไม่ซ้ำกัน (เท่ากับจำนวนเอกสารต้นฉบับที่ถูก Ingest สำเร็จ)
            unique_doc_ids = set(all_doc_ids)
            unique_doc_count = len(unique_doc_ids)
            # --- END NEW ---

            print(f"✅ สำเร็จ! พบทั้งหมด {total_count} Vector (Chunks) ใน Collection '{collection_name}'")
            print(f"   - พบ {unique_doc_count} Source Document ที่ถูก Ingest สำเร็จ (Unique Doc IDs)")
        else:
            print(f"❌ ล้มเหลว (Count Vectors): ไม่สามารถเข้าถึง Vector Store Instance ที่สามารถนับได้สำหรับ Collection '{collection_name}'")

    except Exception as e:
        print(f"❌ ล้มเหลว (Count Vectors): {e}")

def count_source_files(collection_name: str):
    """
    Counts and reports the number of source files (.pdf, .docx, .png, .jpg)
    in the corresponding data directory (e.g., data/evidence_km).
    """
    source_dir = os.path.join('data', collection_name)
    print("\n" + "-" * 50)
    print(f"--- 3.5. ทดสอบนับจำนวน Source File ในโฟลเดอร์ '{source_dir}' ---")
    print("-" * 50)
    
    file_types = {
        '.pdf': 0,
        '.docx': 0,
        '.png': 0,
        '.jpg': 0,
    }
    
    total_files = 0
    
    if not os.path.isdir(source_dir):
        print(f"⚠️ คำเตือน: ไม่พบ Source Data Directory: {source_dir}. ข้ามการนับไฟล์ต้นฉบับ.")
        return

    try:
        # ค้นหาไฟล์ทุกประเภทที่รองรับในโฟลเดอร์และโฟลเดอร์ย่อย
        for file_ext in file_types.keys():
            # ใช้ glob.glob เพื่อค้นหาไฟล์แบบ recursive
            pattern = os.path.join(source_dir, '**', f"*{file_ext}")
            files = glob.glob(pattern, recursive=True)
            file_types[file_ext] = len(files)
            total_files += len(files)

        if total_files > 0:
            print(f"✅ สำเร็จ! พบไฟล์ต้นฉบับทั้งหมด {total_files} ไฟล์:")
            # รายงานจำนวนแยกตามประเภทไฟล์
            for ft, count in file_types.items():
                if count > 0:
                    print(f"   - {ft} : {count} ไฟล์")
        else:
            print("⚠️ คำเตือน: ไม่พบไฟล์ต้นฉบับที่รองรับ (.pdf, .docx, .png, .jpg) ในโฟลเดอร์นี้.")

    except Exception as e:
        print(f"❌ ล้มเหลว (Count Source Files): {e}")

def test_vectorstore_retrieval(
    doc_type: str, 
    enabler: Optional[str], 
    doc_id: Optional[str], 
    query: str
):
    doc_type_lower = doc_type.lower()
    retriever_wrapper: Optional[MultiDocRetriever] = None
    
    print("\n" + "=" * 60)
    print("--- 1. ข้อมูลการทดสอบ ---")
    print(f"Collection: **{doc_type_lower}** (Enabler: {enabler or '-'})")
    print(f"Query: **{query}**")
    print(f"Stable Doc ID (Direct Test): **{doc_id or 'Skip'}**")
    print("=" * 60)

    # ------------------ A. ทดสอบโหลด Vector Store สำหรับ Query ------------------
    print("\n" + "-" * 50)
    print("--- 2. ทดสอบ load_all_vectorstores และ Query ---")
    print("-" * 50)
    
    collection_to_load = f"{doc_type_lower}_{enabler.lower()}" if doc_type_lower == 'evidence' and enabler else doc_type_lower
    
    try:
        # ใช้ evidence_enabler ตาม signature ใน core/vectorstore.py
        retriever_wrapper = load_all_vectorstores(
            doc_types=[doc_type], 
            evidence_enabler=enabler 
        )
        
        # 📌 อัปเดต: ใช้ ._retrievers_list
        print(f"✅ สำเร็จ! MultiDocRetriever ถูกโหลดด้วย {len(retriever_wrapper._retrievers_list)} Retriever(s) (Collection: '{collection_to_load}')")
        
        # ------------------ B. ทดสอบ Query ผ่าน Retriever ------------------
        # MultiDocRetriever ควรมี method invoke
        results: List[Document] = retriever_wrapper.invoke(query)
        
        if results:
            print(f"✅ สำเร็จ! พบ {len(results)} Chunk ที่เกี่ยวข้องกับคำถาม")
            print(f"   - Source File: {results[0].metadata.get('source_file', 'N/A')}")
            print(f"   - เนื้อหาเริ่มต้น: '{results[0].page_content[:150]}...'")
        else:
            print("⚠️ คำเตือน: Query ไม่พบเอกสารที่เกี่ยวข้อง")

    except Exception as e:
        print(f"❌ ล้มเหลว (load_all_vectorstores/Query): {e}")
        
    # ------------------ C. ทดสอบนับ Vector ------------------
    if retriever_wrapper:
        # 🚩 NEW: เรียกใช้ฟังก์ชันนับ Vector โดยใช้ retriever_wrapper ที่โหลดแล้ว
        test_count_vectors(retriever_wrapper, collection_to_load)
    else:
        print("❌ ข้ามการนับ Vector: load_all_vectorstores ล้มเหลวในส่วนที่ 2.")

    # ------------------ C.5. ทดสอบนับ Source Files ------------------
    # 🚩 NEW: เรียกใช้ฟังก์ชันนับไฟล์ต้นฉบับตามที่ผู้ใช้ร้องขอ
    count_source_files(collection_to_load)

    # ------------------ D. ทดสอบดึงเอกสารด้วย Stable ID โดยตรง ------------------
    if doc_id:
        print("\n" + "-" * 50)
        print(f"--- 4. ทดสอบดึงเอกสารด้วย Stable ID '{doc_id}' โดยตรง ---")
        print("-" * 50)
        
        manager = VectorStoreManager() # โหลด Singleton Manager
        try:
            documents = manager.get_chunks_from_doc_ids( 
                stable_doc_ids=[doc_id], 
                doc_type=doc_type_lower,
                enabler=enabler 
            )
            
            if documents:
                print(f"✅ สำเร็จ! ดึง {len(documents)} Chunk ด้วย Stable ID '{doc_id}'")
                print(f"   - เนื้อหาเริ่มต้น: '{documents[0].page_content[:150]}...'")
            else:
                print(f"❌ ล้มเหลว: ดึงเอกสารด้วย Stable ID '{doc_id}' ได้ 0 ผลลัพธ์")
            
        except Exception as e:
            print(f"❌ ข้อผิดพลาดร้ายแรงระหว่างการดึงด้วย Stable ID: {e}")

if __name__ == "__main__":
    args = parse_arguments()
    
    # 📌 ตรวจสอบ Enabler สำหรับ Evidence
    if args.doc_type.lower() == 'evidence' and not args.enabler:
         logger.error("❌ ต้องระบุ --enabler เมื่อ doc_type คือ 'evidence'")
         sys.exit(1)
         
    test_vectorstore_retrieval(args.doc_type, args.enabler, args.doc_id, args.query)
