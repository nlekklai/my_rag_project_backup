# -------------------- core/vectorstore.py --------------------
import os
from typing import List, Optional
from langchain.schema import Document, BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from concurrent.futures import ThreadPoolExecutor

VECTORSTORE_DIR = "vectorstore"

# -------------------- Embeddings --------------------
def get_hf_embeddings():
    """
    คืนค่า HuggingFace Embeddings สำหรับ M3/Mac Silicon โดยใช้ device 'mps'
    หาก device ไม่สามารถใช้ได้จะ fallback ไป CPU
    """
    device = "mps"
    print(f"⚡ Using device: {device} for embeddings (M3 acceleration)")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )

# -------------------- Vectorstore management --------------------
def list_vectorstore_folders() -> List[str]:
    """List all vectorstore folders under VECTORSTORE_DIR (ignore files like .DS_Store)"""
    if not os.path.exists(VECTORSTORE_DIR):
        return []
    return [
        f for f in os.listdir(VECTORSTORE_DIR)
        if os.path.isdir(os.path.join(VECTORSTORE_DIR, f))
    ]

def vectorstore_exists(doc_id: str) -> bool:
    """Check if vectorstore for a given doc_id exists (must be directory with content)"""
    path = os.path.join(VECTORSTORE_DIR, doc_id)
    return os.path.isdir(path) and bool(os.listdir(path))

def save_to_vectorstore(doc_id: str, text_chunks: List[str], metadata: dict = None):
    """
    Save list of text chunks into a Chroma vectorstore
    """
    docs = [
        Document(
            page_content=t,
            metadata={**(metadata or {}), "source": doc_id, "chunk": i+1}
        )
        for i, t in enumerate(text_chunks)
    ]

    embeddings = get_hf_embeddings()
    doc_dir = os.path.join(VECTORSTORE_DIR, doc_id)
    os.makedirs(doc_dir, exist_ok=True)

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=doc_dir
    )
    print(f"📄 Saved {len(docs)} chunks for doc_id={doc_id} into {doc_dir}")
    return vectordb

def load_vectorstore(doc_id: str, top_k: int = 15):
    """
    Load a vectorstore retriever for a specific doc_id
    top_k: number of passages to retrieve per query
    """
    embeddings = get_hf_embeddings()
    path = os.path.join(VECTORSTORE_DIR, doc_id)
    if not os.path.isdir(path):
        raise ValueError(f"Vectorstore for doc_id '{doc_id}' not found")
    retriever = Chroma(
        persist_directory=path,
        embedding_function=embeddings
    ).as_retriever(search_kwargs={"k": top_k})
    print(f"✅ Loaded retriever for doc_id={doc_id} with top_k={top_k}")
    return retriever

# -------------------- MultiDoc Retriever --------------------
class MultiDocRetriever(BaseRetriever):
    """
    Combine multiple retrievers into one, deduplicating results
    """
    def __init__(self, retrievers_list: List[BaseRetriever], k_per_doc: int = 5):
        super().__init__()
        self._retrievers = retrievers_list
        self._k_per_doc = k_per_doc

    def _get_relevant_documents(self, query: str, *, run_manager=None):
        docs = []

        def retrieve(r):
            # รองรับ retriever แบบ custom (มี _get_relevant_documents) และแบบมาตรฐาน
            if hasattr(r, "_get_relevant_documents"):
                return r._get_relevant_documents(query, run_manager=run_manager)[:self._k_per_doc]
            return r.get_relevant_documents(query)[:self._k_per_doc]

        with ThreadPoolExecutor() as executor:
            results = executor.map(retrieve, self._retrievers)

        # Deduplicate by source + chunk
        seen = set()
        unique_docs = []
        for dlist in results:
            for d in dlist:
                key = f"{d.metadata.get('source')}_{d.metadata.get('chunk')}"
                if key not in seen:
                    seen.add(key)
                    unique_docs.append(d)

        print(f"📝 Query='{query}' found {len(unique_docs)} unique docs")
        for d in unique_docs:
            print(f" - source={d.metadata.get('source')} chunk={d.metadata.get('chunk')}")

        return unique_docs

# -------------------- Load multiple vectorstores --------------------
def load_all_vectorstores(doc_ids: Optional[List[str]] = None, top_k: int = 10) -> MultiDocRetriever:
    """
    Load retrievers for multiple doc_ids (or all folders if doc_ids=None)
    Returns a MultiDocRetriever combining them
    """
    all_retrievers = []
    for folder in list_vectorstore_folders():
        if doc_ids and folder not in doc_ids:
            continue
        retriever = load_vectorstore(folder, top_k=top_k)
        all_retrievers.append(retriever)

    if not all_retrievers:
        raise ValueError("No vectorstores found for the given doc_ids")

    return MultiDocRetriever(retrievers_list=all_retrievers, k_per_doc=top_k)
