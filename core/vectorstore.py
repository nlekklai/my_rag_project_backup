# -------------------- core/vectorstore.py (โค้ดสุดท้ายที่สมบูรณ์) --------------------
import os
from typing import List, Optional, Union
from langchain.schema import Document, BaseRetriever
from concurrent.futures import ThreadPoolExecutor
from pydantic import PrivateAttr
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
# CRITICAL FIX: เปลี่ยนพาธ Import ให้ถูกต้องตาม LangChain v0.2/0.3+
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from flashrank import Ranker # FIX 1: Import Ranker class


# === CRITICAL FIX 2: Rebuild model (แก้ไข Pydantic V2) ===
FlashrankRerank.model_rebuild()

VECTORSTORE_DIR = "vectorstore"

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class VectorStoreManager:
    """
    Manager สำหรับสร้างและโหลด Chroma vectorstore
    """
    _instance: Optional["VectorStoreManager"] = None
    _vectorstore: Optional[Chroma] = PrivateAttr(default=None)

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, persist_dir: str = "vectorstore/document"):
        self.persist_dir = persist_dir
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                                model_kwargs={"device": "mps"})

    def load_vectorstore(self):
        if not os.path.exists(self.persist_dir):
            logger.warning(f"Vectorstore path not found: {self.persist_dir}")
            return None
        try:
            self._vectorstore = Chroma(persist_directory=self.persist_dir,
                                       embedding_function=self.embeddings)
            logger.info(f"✅ Loaded vectorstore from {self.persist_dir}")
        except Exception as e:
            logger.error(f"Failed to load vectorstore: {e}")
            self._vectorstore = None
        return self._vectorstore

    def get_retriever(self, k: int = 5):
        if self._vectorstore is None:
            self.load_vectorstore()
        if self._vectorstore is None:
            logger.error("Vectorstore not available for retrieval.")
            return None
        return self._vectorstore.as_retriever(search_kwargs={"k": k})


# =================================================================
# --- MultiDoc Retriever ---
# =================================================================
class NamedRetriever:
    """Wrapper around a retriever to store doc_id and doc_type"""
    def __init__(self, retriever: BaseRetriever, doc_id: str, doc_type: str):
        self.retriever = retriever
        self.doc_id = doc_id
        self.doc_type = doc_type

    def get_relevant_documents(self, query: str, **kwargs):
        return self.retriever.invoke(query, **kwargs)

class MultiDocRetriever(BaseRetriever):
    """
    Combine multiple NamedRetrievers into one, deduplicating results
    """
    _retrievers_list: list  = PrivateAttr()
    _k_per_doc: int = PrivateAttr()

    def __init__(self, retrievers_list: list, k_per_doc: int = 5):
        super().__init__()
        self._retrievers_list = retrievers_list
        self._k_per_doc = k_per_doc

    @property
    def retrievers_list(self):
        return self._retrievers_list

    def _get_relevant_documents(self, query: str, *, run_manager=None):
        docs = []    
        
        def retrieve(named_r):
            r = named_r.retriever
            # FINAL FIX: Use the public and stable 'invoke' method for parallel execution.
            # This is more robust against internal LangChain callback issues (run_manager).
            return r.invoke(query) # Reranker handles the final count (final_k=5)

        with ThreadPoolExecutor() as executor:
            # results will be a list of lists of Documents (one list per retriever)
            results = list(executor.map(retrieve, self._retrievers_list))

        seen = set()
        unique_docs = []
        # Flatten the list of lists and handle deduplication/metadata
        for dlist, named_r in zip(results, self._retrievers_list):
            if dlist is None: # Should not happen if inner calls are fixed, but safe check
                continue
            for d in dlist:
                # Deduplication logic (based on source file, chunk index, and doc type)
                key = f"{d.metadata.get('source')}_{d.metadata.get('chunk_index')}_{named_r.doc_type}_{d.page_content[:50]}"
                if key not in seen:
                    seen.add(key)
                    d.metadata["doc_type"] = named_r.doc_type
                    d.metadata["doc_id"] = named_r.doc_id
                    unique_docs.append(d)

        print(f"📝 Query='{query}' found {len(unique_docs)} unique docs across all retrieved lists.")
        for d in unique_docs:
            print(f" - source={d.metadata.get('doc_id')}, chunk={d.metadata.get('chunk_index')}, doc_type={d.metadata.get('doc_type')}, score={d.metadata.get('relevance_score', 'N/A')}")
            
        return unique_docs


# -------------------- Embeddings --------------------
def get_hf_embeddings():
    device = "mps"
    print(f"⚡ Using device: {device} for embeddings (M3 acceleration)")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )

def get_vectorstore_path(doc_type: Optional[str] = None):
    if doc_type:
        path = os.path.join(VECTORSTORE_DIR, doc_type)
        os.makedirs(path, exist_ok=True)
        return path
    return VECTORSTORE_DIR

# -------------------- Vectorstore management --------------------
def list_vectorstore_folders(base_path: str = VECTORSTORE_DIR, doc_type: Optional[str] = None) -> List[str]:
    target_path = os.path.join(base_path, doc_type) if doc_type else base_path
    if not os.path.exists(target_path):
        return []
    # If doc_type is provided, list subfolders in that doc_type folder.
    # Otherwise, list top-level folders in VECTORSTORE_DIR.
    if doc_type:
        full_path = os.path.join(base_path, doc_type)
        if os.path.isdir(full_path):
            return [f for f in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, f))]
        return []
    return [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

def vectorstore_exists(doc_id: str, base_path: str = VECTORSTORE_DIR, doc_type: Optional[str] = None) -> bool:
    """
    ตรวจสอบว่า vectorstore สำหรับ doc_id มีอยู่หรือไม่
    """
    if doc_type:
        path = os.path.join(base_path, doc_type, doc_id)
    else:
        path = os.path.join(base_path, doc_id)

    if not os.path.isdir(path):
        return False
    # Check for Chroma specific files
    if os.path.isfile(os.path.join(path, "chroma.sqlite3")):
        return True
    return False # Assume Chroma uses chroma.sqlite3 for persistence

def save_to_vectorstore(
    doc_id: str,
    text_chunks: List[str],
    metadata: Optional[dict] = None,
    doc_type: str = "document",
    base_path: str = VECTORSTORE_DIR
):
    """
    Save document chunks to a Chroma vectorstore.
    """
    # Build Document objects
    docs = [
        Document(
            page_content=t,
            metadata={**(metadata or {}), "source": doc_id, "chunk_index": i + 1} # ใช้ chunk_index
        )
        for i, t in enumerate(text_chunks)
    ]

    # Get embeddings
    embeddings = get_hf_embeddings()

    # Ensure vectorstore folder exists
    doc_dir = os.path.join(base_path, doc_type, doc_id)
    os.makedirs(doc_dir, exist_ok=True)

    # Save to Chroma
    vectordb = Chroma.from_documents(
        docs,             # documents
        embeddings,       # embeddings function
        persist_directory=doc_dir
    )

    print(f"📄 Saved {len(docs)} chunks for doc_id={doc_id} into {doc_dir}")
    return vectordb

def load_vectorstore(
    doc_id: str,
    top_k: int = 15, 
    final_k: int = 5, 
    doc_types: list[str] | str = "document",
    base_path: str = VECTORSTORE_DIR,
):
    """
    Load a vectorstore retriever พร้อมเพิ่ม Contextual Compression (Reranker)
    """
    if isinstance(doc_types, str):
        doc_types = [doc_types]

    embeddings = get_hf_embeddings()
    retriever = None

    for dtype in doc_types:
        path = os.path.join(base_path, dtype, doc_id)
        if os.path.isdir(path) and vectorstore_exists(doc_id, base_path, dtype):
            # 1. สร้าง Base Retriever
            base_retriever = Chroma(
                persist_directory=path,
                embedding_function=embeddings
            ).as_retriever(search_kwargs={"k": top_k}) # ดึงมา top_k ชิ้น
            
            # 2. สร้าง Compressor (Reranker)
            compressor = FlashrankRerank(top_n=final_k, model="ms-marco-MiniLM-L-12-v2") 
            
            # 3. สร้าง Contextual Compression Retriever
            retriever = ContextualCompressionRetriever(
                base_compressor=compressor, 
                base_retriever=base_retriever
            )

            print(f"✅ Loaded Reranking Retriever for doc_id={doc_id} (doc_type={dtype}) with k={top_k}->{final_k}")
            break  # ใช้ตัวแรกที่เจอ

    if retriever is None:
        raise ValueError(f"Vectorstore for doc_id '{doc_id}' not found in any of {doc_types}")

    return retriever

# -------------------- Load multiple vectorstores --------------------
def load_all_vectorstores(doc_ids: Optional[Union[List[str], str]] = None,
                          top_k: int = 15,
                          final_k: int = 5,
                          doc_type: Optional[Union[str, List[str]]] = None,
                          base_path: str = VECTORSTORE_DIR) -> MultiDocRetriever:
    """
    Load vectorstores หลาย doc_id + หลาย doc_type พร้อมกัน
    """
    if isinstance(doc_ids, str):
        doc_ids = [doc_ids]
    if doc_type is None:
        doc_types = ["document"]
    elif isinstance(doc_type, str):
        doc_types = [doc_type]
    else:
        doc_types = doc_type

    all_retrievers = []

    for dt in doc_types:
        folders = list_vectorstore_folders(base_path=base_path, doc_type=dt)
        for folder in folders:
            if doc_ids and folder not in doc_ids:
                continue
            try:
                # เราใช้ load_vectorstore ที่มีการตรวจสอบ vectorstore_exists ภายใน
                retriever = load_vectorstore(folder, top_k=top_k, final_k=final_k, doc_types=[dt], base_path=base_path) 
                all_retrievers.append(NamedRetriever(retriever, doc_id=folder, doc_type=dt))
            except ValueError:
                 # ValueError occurs if vectorstore_exists is False
                 continue
            except Exception as e:
                print(f"⚠️ Skipping folder '{folder}' ({dt}): {e}")

    if not all_retrievers:
        raise ValueError(f"No vectorstores found for doc_ids={doc_ids} in doc_types={doc_types}")

    return MultiDocRetriever(retrievers_list=all_retrievers, k_per_doc=top_k)