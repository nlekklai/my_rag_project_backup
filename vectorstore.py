from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document, BaseRetriever
from langchain.vectorstores.base import VectorStoreRetriever
import os
from typing import List

VECTORSTORE_DIR = "vectorstore"

def list_vectorstore_folders() -> list[str]:
    """คืน list ของชื่อ folder ที่เป็น vectorstore"""
    if not os.path.exists(VECTORSTORE_DIR):
        return []
    return [f for f in os.listdir(VECTORSTORE_DIR)
            if os.path.isdir(os.path.join(VECTORSTORE_DIR, f))]

def get_hf_embeddings():
    """คืนค่า embeddings model ของ HuggingFace"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

def vectorstore_exists(doc_id: str) -> bool:
    """เช็คว่า vectorstore ของ doc_id มีหรือไม่"""
    path = os.path.join(VECTORSTORE_DIR, doc_id)
    return os.path.exists(path) and bool(os.listdir(path))

def save_to_vectorstore(doc_id: str, text_chunks: list[str]):
    """สร้างและบันทึก vectorstore ของเอกสาร"""
    docs = [Document(page_content=t, metadata={"source": doc_id, "chunk": i+1}) 
            for i, t in enumerate(text_chunks)]
    embeddings = get_hf_embeddings()
    doc_dir = os.path.join(VECTORSTORE_DIR, doc_id)
    os.makedirs(doc_dir, exist_ok=True)

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,  # embedding_function
        persist_directory=doc_dir
    )
    vectordb.persist()
    print(f"📄 Saved {len(docs)} chunks for doc_id={doc_id} into {doc_dir}")
    return vectordb

def load_vectorstore(doc_id: str) -> VectorStoreRetriever:
    """โหลด vectorstore เฉพาะ doc_id"""
    embeddings = get_hf_embeddings()
    path = os.path.join(VECTORSTORE_DIR, doc_id)
    if not os.path.exists(path):
        raise ValueError(f"Vectorstore for doc_id '{doc_id}' not found")
    return Chroma(
        persist_directory=path,
        embedding_function=embeddings
    ).as_retriever(search_kwargs={"k": 3})

def load_all_vectorstores(doc_ids: list[str] | None = None):
    all_retrievers = []
    for folder in os.listdir("vectorstore"):
        if doc_ids and folder not in doc_ids:
            continue
        retriever = load_vectorstore(folder)
        all_retrievers.append(retriever)
    return MultiDocRetriever(retrievers_list=all_retrievers)




class MultiDocRetriever(BaseRetriever):
    _retrievers: list[BaseRetriever]

    def __init__(self, retrievers_list: list[BaseRetriever]):
        super().__init__()
        self._retrievers = retrievers_list

    def _get_relevant_documents(
        self, query: str, *, run_manager=None
    ) -> list[Document]:
        docs = []
        for r in self._retrievers:
            docs.extend(r._get_relevant_documents(query, run_manager=run_manager))
        return docs
