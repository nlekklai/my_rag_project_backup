# test_retrieval_stableid.py
import os
import sys
import logging
import argparse
from typing import List, Optional, Any
import glob

from langchain_core.documents import Document

# -------------------- Logging Setup --------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# -------------------- Import Core Modules --------------------
try:
    from core.vectorstore import VectorStoreManager, load_all_vectorstores, MultiDocRetriever, NamedRetriever
    from core.ingest import SUPPORTED_DOC_TYPES
except ImportError as e:
    logger.error(f"❌ ไม่สามารถ Import core.vectorstore ได้: {e}")
    sys.exit(1)

# -------------------- Argument Parsing --------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="Test RAG Retrieval & Report Stable Doc UUIDs")
    
    parser.add_argument(
        "--doc_type",
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
        "--query",
        type=str,
        default=None,
        help="Optional query to test retrieval (similarity search).",
    )
    
    return parser.parse_args()

# -------------------- Core Functions --------------------
def scan_all_chunks(retriever_wrapper: MultiDocRetriever):
    """
    Scan all chunks in loaded retrievers and report their stable_doc_uuid / doc_id
    """
    print("\n" + "="*60)
    print("--- Scanning all chunks for stable_doc_uuid / doc_id ---")
    print("="*60)
    
    all_chunks_info = []
    
    if not retriever_wrapper._retrievers_list:
        print("⚠️ Warning: No retrievers loaded in MultiDocRetriever")
        return
    
    for named_retriever in retriever_wrapper._retrievers_list:
        retriever_instance = named_retriever.load_instance()
        
        # ใช้ fallback สำหรับชื่อ retriever
        retriever_label = getattr(named_retriever, 'name', str(named_retriever))
        
        # Try to get the underlying vectorstore object
        base_retriever = getattr(retriever_instance, 'base_retriever', None)
        vectorstore = getattr(base_retriever, 'vectorstore', None)
        if not vectorstore:
            vectorstore = getattr(retriever_instance, 'vectorstore', None)
        
        if not vectorstore or not hasattr(vectorstore, 'get'):
            print(f"❌ Cannot access vectorstore for retriever '{retriever_label}'")
            continue
        
        try:
            result = vectorstore.get(
                where={"doc_id": {"$ne": "non_exist_id"}},
                limit=None,
                include=['metadatas', 'documents']
            )
            
            for doc_content, meta in zip(result.get('documents', []), result.get('metadatas', [])):
                stable_uuid = meta.get('stable_doc_uuid', meta.get('doc_id', 'N/A'))
                all_chunks_info.append({
                    'stable_doc_uuid': stable_uuid,
                    'source_file': meta.get('source') or meta.get('source_file') or 'Unknown',
                    'chunk_index': meta.get('chunk_index') or meta.get('chunk_id', 'N/A'),
                    'page_label': meta.get('page_label') or meta.get('page', 'N/A'),
                    'content_preview': doc_content[:100].replace("\n", " ")
                })
            
            print(f"✅ Retriever '{retriever_label}': Found {len(result.get('ids', []))} chunks")
            
        except Exception as e:
            print(f"❌ Failed to scan retriever '{retriever_label}': {e}")
    
    print("\n--- Summary of scanned chunks ---")
    for info in all_chunks_info:
        print(f"{info['stable_doc_uuid']} | {info['source_file']} | chunk_index={info['chunk_index']} | page={info['page_label']} | preview='{info['content_preview']}'")
    
    print(f"\nTotal chunks scanned: {len(all_chunks_info)}")
    return all_chunks_info


def test_vectorstore(doc_type: str, enabler: Optional[str], query: Optional[str]):
    doc_type_lower = doc_type.lower()
    
    # For evidence type, enabler is required
    if doc_type_lower == 'evidence' and not enabler:
        logger.error("❌ Must specify --enabler when doc_type is 'evidence'")
        sys.exit(1)
    
    # Load MultiDocRetriever
    try:
        retriever_wrapper: MultiDocRetriever = load_all_vectorstores(
            doc_types=[doc_type_lower],
            evidence_enabler=enabler
        )
        print(f"✅ MultiDocRetriever loaded with {len(retriever_wrapper._retrievers_list)} retriever(s)")
    except Exception as e:
        print(f"❌ Failed to load MultiDocRetriever: {e}")
        return
    
    # Scan all chunks
    all_chunks_info = scan_all_chunks(retriever_wrapper)
    
    # Optional: run a query if provided
    if query:
        print("\n" + "-"*50)
        print(f"--- Running similarity query: '{query}' ---")
        print("-"*50)
        try:
            results: List[Document] = retriever_wrapper.invoke(query)
            if results:
                print(f"✅ Found {len(results)} relevant chunks for query")
                for doc in results[:5]:
                    meta = doc.metadata or {}
                    print(f"{meta.get('stable_doc_uuid', meta.get('doc_id', 'N/A'))} | {meta.get('source', meta.get('source_file','Unknown'))} | preview='{doc.page_content[:100]}...'")
            else:
                print("⚠️ No relevant chunks found for query")
        except Exception as e:
            print(f"❌ Query failed: {e}")

# -------------------- Main --------------------
if __name__ == "__main__":
    args = parse_arguments()
    test_vectorstore(args.doc_type, args.enabler, args.query)
