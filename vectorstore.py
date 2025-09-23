# vectorstore.py
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import os

VECTORSTORE_DIR = "vectorstore"

def get_hf_embeddings():
    """
    คืนค่า embedding function ของ HuggingFace
    ใช้ multilingual model รองรับภาษาไทย
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

def save_to_vectorstore(doc_id: str, text_chunks: list[str]):
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain.schema import Document
    import os

    docs = [Document(page_content=t, metadata={"source": doc_id, "chunk": i+1}) for i, t in enumerate(text_chunks)]
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    doc_dir = os.path.join(VECTORSTORE_DIR, doc_id)
    os.makedirs(doc_dir, exist_ok=True)
    vectordb = Chroma.from_documents(docs, embeddings, persist_directory=doc_dir)
    vectordb.persist()
    return vectordb


def load_vectorstore(doc_id: str):
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    import os

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    doc_dir = os.path.join(VECTORSTORE_DIR, doc_id)
    if not os.path.exists(doc_dir):
        raise ValueError(f"Vectorstore for doc_id '{doc_id}' not found")
    return Chroma(persist_directory=doc_dir, embedding_function=embeddings)
