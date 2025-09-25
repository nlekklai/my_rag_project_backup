from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
import os

VECTORSTORE_DIR = "vectorstore"

def get_hf_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

def vectorstore_exists(doc_id: str) -> bool:
    path = os.path.join(VECTORSTORE_DIR, doc_id)
    return os.path.exists(path) and bool(os.listdir(path))


def save_to_vectorstore(doc_id: str, text_chunks: list[str]):
    docs = [Document(page_content=t, metadata={"source": doc_id, "chunk": i+1}) for i, t in enumerate(text_chunks)]
    embeddings = get_hf_embeddings()
    doc_dir = os.path.join(VECTORSTORE_DIR, doc_id)
    os.makedirs(doc_dir, exist_ok=True)
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,        # ใช้ embedding แทน embedding_function
        persist_directory=doc_dir
    )

    # vectordb.persist()

    print(f"📄 Saving {len(docs)} chunks for doc_id={doc_id} into {doc_dir}")

    return vectordb

def load_vectorstore(doc_id: str):
    embeddings = get_hf_embeddings()
    doc_dir = os.path.join(VECTORSTORE_DIR, doc_id)
    if not os.path.exists(doc_dir):
        raise ValueError(f"Vectorstore for doc_id '{doc_id}' not found")
    
    return Chroma(
        persist_directory=doc_dir,
        embedding_function=embeddings
    )


def load_multiple_vectorstores(doc_ids: list[str]):
    """
    โหลดหลาย vectorstore แล้วรวมเป็น MultiRetriever
    """
    embeddings = get_hf_embeddings()
    chromas = []
    for doc_id in doc_ids:
        doc_dir = os.path.join(VECTORSTORE_DIR, doc_id)
        if os.path.exists(doc_dir):
            chromas.append(Chroma(persist_directory=doc_dir, embedding_function=embeddings))
    if not chromas:
        raise ValueError("No vectorstores found for given doc_ids")

    class MultiRetriever:
        def __init__(self, chromas):
            self.chromas = chromas

        def get_relevant_documents(self, query):
            results = []
            for c in self.chromas:
                results.extend(c.as_retriever(search_kwargs={"k":3}).get_relevant_documents(query))
            return results

    return MultiRetriever(chromas)

def load_all_vectorstores():
    """
    โหลด vectorstore ทุก doc ใน folder
    """
    doc_ids = [d for d in os.listdir(VECTORSTORE_DIR) if os.path.isdir(os.path.join(VECTORSTORE_DIR, d))]
    return load_multiple_vectorstores(doc_ids)
