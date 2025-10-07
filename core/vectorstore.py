# -------------------- core/vectorstore.py --------------------
import os
from typing import List, Optional, Union
from langchain.schema import Document, BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from concurrent.futures import ThreadPoolExecutor
from pydantic import PrivateAttr

VECTORSTORE_DIR = "vectorstore"

# -------------------- Embeddings --------------------
def get_hf_embeddings():
    device = "mps"
    print(f"⚡ Using device: {device} for embeddings (M3 acceleration)")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )

# -------------------- Vectorstore management --------------------
def list_vectorstore_folders(base_path: str = VECTORSTORE_DIR, doc_type: Optional[str] = None) -> List[str]:
    target_path = os.path.join(base_path, doc_type) if doc_type else base_path
    if not os.path.exists(target_path):
        return []
    return [f for f in os.listdir(target_path) if os.path.isdir(os.path.join(target_path, f))]

def vectorstore_exists(doc_id: str, base_path: str = VECTORSTORE_DIR, doc_type: Optional[str] = None) -> bool:
    """
    ตรวจสอบว่า vectorstore สำหรับ doc_id มีอยู่หรือไม่
    - doc_type: 'document', 'faq' หรือ None (default)
    """
    if doc_type:
        path = os.path.join(base_path, doc_type, doc_id)
    else:
        path = os.path.join(base_path, doc_id)

    if not os.path.isdir(path):
        return False
    if os.path.isfile(os.path.join(path, "chroma.sqlite3")):
        return True
    for f in os.listdir(path):
        subpath = os.path.join(path, f)
        if os.path.isdir(subpath) and any(fname.endswith(".bin") for fname in os.listdir(subpath)):
            return True
    return False

from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from typing import List, Optional
import os

def save_to_vectorstore(
    doc_id: str,
    text_chunks: List[str],
    metadata: Optional[dict] = None,
    doc_type: str = "document",
    base_path: str = VECTORSTORE_DIR
):
    """
    Save document chunks to a Chroma vectorstore.

    Args:
        doc_id: unique ID for the document.
        text_chunks: list of text chunks.
        metadata: optional metadata dict.
        doc_type: folder type (e.g., "document", "faq").
        base_path: root folder for vectorstores.
    """
    # Build Document objects
    docs = [
        Document(
            page_content=t,
            metadata={**(metadata or {}), "source": doc_id, "chunk": i + 1}
        )
        for i, t in enumerate(text_chunks)
    ]

    # Get embeddings
    embeddings = get_hf_embeddings()

    # Ensure vectorstore folder exists
    doc_dir = os.path.join(base_path, doc_type, doc_id)
    os.makedirs(doc_dir, exist_ok=True)

    # Save to Chroma (embedding_function is positional now)
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
    doc_types: list[str] | str = "document",
    base_path: str = VECTORSTORE_DIR,
):
    """
    Load a vectorstore retriever สำหรับ doc_id
    - doc_types: สามารถเป็น list เช่น ['document','faq'] หรือ string
    """
    if isinstance(doc_types, str):
        doc_types = [doc_types]

    embeddings = get_hf_embeddings()
    retriever = None

    for dtype in doc_types:
        path = os.path.join(base_path, dtype, doc_id)
        if os.path.isdir(path):
            retriever = Chroma(
                persist_directory=path,
                embedding_function=embeddings
            ).as_retriever(search_kwargs={"k": top_k})
            print(f"✅ Loaded retriever for doc_id={doc_id} (doc_type={dtype}) with top_k={top_k}")
            break  # ใช้ตัวแรกที่เจอ

    if retriever is None:
        raise ValueError(f"Vectorstore for doc_id '{doc_id}' not found in any of {doc_types}")

    return retriever

# -------------------- MultiDoc Retriever --------------------
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
            if hasattr(r, "_get_relevant_documents"):
                return r._get_relevant_documents(query, run_manager=run_manager)[:self._k_per_doc]
            return r.invoke(query)[:self._k_per_doc]

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(retrieve, self._retrievers_list))

        seen = set()
        unique_docs = []
        for dlist, named_r in zip(results, self._retrievers_list):
            for d in dlist:
                key = f"{d.metadata.get('source')}_{d.metadata.get('chunk')}_{named_r.doc_type}"
                if key not in seen:
                    seen.add(key)
                    d.metadata["doc_type"] = named_r.doc_type
                    unique_docs.append(d)

        print(f"📝 Query='{query}' found {len(unique_docs)} unique docs")
        for d in unique_docs:
            print(f" - source={d.metadata.get('source')}, chunk={d.metadata.get('chunk')}, doc_type={d.metadata.get('doc_type')}")
        return unique_docs

# -------------------- Load multiple vectorstores --------------------
def load_all_vectorstores(doc_ids: Optional[Union[List[str], str]] = None,
                          top_k: int = 10,
                          doc_type: Optional[Union[str, List[str]]] = None,
                          base_path: str = VECTORSTORE_DIR) -> MultiDocRetriever:
    """
    Load vectorstores หลาย doc_id + หลาย doc_type พร้อมกัน
    - doc_type: default ['document'], หรือ ['document','faq']
    - doc_ids: list ของ doc_id ที่ต้องการโหลด
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
            if vectorstore_exists(folder, base_path=base_path, doc_type=dt):
                try:
                    retriever = load_vectorstore(folder, top_k=top_k, doc_types=[dt], base_path=base_path)
                    all_retrievers.append(NamedRetriever(retriever, doc_id=folder, doc_type=dt))
                    print(f"✅ Loaded retriever for doc_id={folder} ({dt}) with top_k={top_k}")
                except Exception as e:
                    print(f"⚠️ Skipping folder '{folder}' ({dt}): {e}")

    if not all_retrievers:
        raise ValueError(f"No vectorstores found for doc_ids={doc_ids} in doc_types={doc_types}")

    return MultiDocRetriever(retrievers_list=all_retrievers, k_per_doc=top_k)
