# -------------------- vectorstore.py (ปรับปรุง) --------------------
import os
from typing import List, Optional
from langchain.schema import Document, BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from concurrent.futures import ThreadPoolExecutor
from pydantic import PrivateAttr

VECTORSTORE_DIR = "vectorstore"

# -------------------- Embeddings --------------------
def get_hf_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

# -------------------- Vectorstore management --------------------
def list_vectorstore_folders() -> List[str]:
    if not os.path.exists(VECTORSTORE_DIR):
        return []
    return [f for f in os.listdir(VECTORSTORE_DIR)
            if os.path.isdir(os.path.join(VECTORSTORE_DIR, f))]

def vectorstore_exists(doc_id: str) -> bool:
    path = os.path.join(VECTORSTORE_DIR, doc_id)
    return os.path.exists(path) and bool(os.listdir(path))

def save_to_vectorstore(doc_id: str, text_chunks: List[str]):
    docs = [Document(page_content=t, metadata={"source": doc_id, "chunk": i+1}) 
            for i, t in enumerate(text_chunks)]
    embeddings = get_hf_embeddings()
    doc_dir = os.path.join(VECTORSTORE_DIR, doc_id)
    os.makedirs(doc_dir, exist_ok=True)

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=doc_dir
    )
    vectordb.persist()
    print(f"📄 Saved {len(docs)} chunks for doc_id={doc_id} into {doc_dir}")
    return vectordb

def load_vectorstore(doc_id: str):
    embeddings = get_hf_embeddings()
    path = os.path.join(VECTORSTORE_DIR, doc_id)
    if not os.path.exists(path):
        raise ValueError(f"Vectorstore for doc_id '{doc_id}' not found")
    return Chroma(
        persist_directory=path,
        embedding_function=embeddings
    ).as_retriever(search_kwargs={"k": 3})

# -------------------- MultiDoc Retriever --------------------
class MultiDocRetriever(BaseRetriever):
    _retrievers: list[BaseRetriever]
    _k: int = PrivateAttr(default=3)  # k เป็น private attribute

    def __init__(self, retrievers_list: list[BaseRetriever], k: int = 3):
        super().__init__()
        self._retrievers = retrievers_list
        self._k = k

    def _get_relevant_documents(self, query: str, *, run_manager=None):
        docs = []

        def retrieve(r: BaseRetriever):
            if hasattr(r, "_get_relevant_documents"):
                return r._get_relevant_documents(query, run_manager=run_manager)[:self._k]
            return r.get_relevant_documents(query)[:self._k]

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            results = executor.map(retrieve, self._retrievers)

        for res in results:
            docs.extend(res)

        # Deduplicate
        seen = set()
        unique_docs = []
        for d in docs:
            key = f"{d.metadata.get('source')}_{d.metadata.get('chunk')}"
            if key not in seen:
                seen.add(key)
                unique_docs.append(d)

        return unique_docs

# -------------------- Load multiple vectorstores --------------------
def load_all_vectorstores(doc_ids: Optional[List[str]] = None) -> MultiDocRetriever:
    all_retrievers = []
    for folder in list_vectorstore_folders():
        if doc_ids and folder not in doc_ids:
            continue
        retriever = load_vectorstore(folder)
        all_retrievers.append(retriever)
    return MultiDocRetriever(retrievers_list=all_retrievers, k=3)
