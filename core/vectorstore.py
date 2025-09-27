import os
from typing import List, Optional
from langchain.schema import Document, BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from concurrent.futures import ThreadPoolExecutor
import logging

# ตั้งค่า Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

VECTORSTORE_DIR = "vectorstore"

# -------------------- Embeddings --------------------
def get_hf_embeddings():
    """
    คืนค่า HuggingFace Embeddings สำหรับ M3/Mac Silicon โดยใช้ device 'mps'
    หาก device ไม่สามารถใช้ได้จะ fallback ไป CPU
    """
    # Note: Using 'cpu' as a safe default in case 'mps' is not available in the environment
    # In a real environment, you might use 'cuda' or 'mps' if supported.
    device = "cpu" 
    logging.info(f"Using device: {device} for embeddings")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )

# -------------------- Vectorstore management --------------------
def list_vectorstore_folders() -> List[str]:
    """List all vectorstore folders under VECTORSTORE_DIR"""
    if not os.path.exists(VECTORSTORE_DIR):
        return []
    return [f for f in os.listdir(VECTORSTORE_DIR)
            if os.path.isdir(os.path.join(VECTORSTORE_DIR, f))]

def vectorstore_exists(doc_id: str) -> bool:
    """Check if vectorstore for a given doc_id exists"""
    path = os.path.join(VECTORSTORE_DIR, doc_id)
    # Check if directory exists and contains files (Chroma creates files on persistence)
    return os.path.exists(path) and bool(os.listdir(path))

def save_to_vectorstore(doc_id: str, text_chunks: list[str], metadata: dict = None):
    """
    Save list of text chunks into a Chroma vectorstore
    """
    # Prepare Document objects from text chunks
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

    # Use Chroma.from_documents to save the chunks and embeddings to disk
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=doc_dir
    )
    logging.info(f"Saved {len(docs)} chunks for doc_id={doc_id} into {doc_dir}")
    return vectordb

def load_vectorstore(doc_id: str, top_k: int = 5):
    """
    Load a vectorstore retriever for a specific doc_id
    top_k: number of passages to retrieve per query
    """
    embeddings = get_hf_embeddings()
    path = os.path.join(VECTORSTORE_DIR, doc_id)
    if not os.path.exists(path) or not os.listdir(path):
        raise ValueError(f"Vectorstore for doc_id '{doc_id}' not found or empty.")
    
    # Load the persisted Chroma store and convert it to a retriever
    retriever = Chroma(
        persist_directory=path,
        embedding_function=embeddings
    ).as_retriever(search_kwargs={"k": top_k})
    logging.info(f"Loaded retriever for doc_id={doc_id} with top_k={top_k}")
    return retriever

# -------------------- MultiDoc Retriever --------------------
class MultiDocRetriever(BaseRetriever):
    """
    Combine multiple retrievers into one, performing parallel retrieval and deduplicating results.
    """
    def __init__(self, retrievers_list: List[BaseRetriever], k_per_doc: int = 5):
        super().__init__()
        self._retrievers = retrievers_list
        self._k_per_doc = k_per_doc

    def _get_relevant_documents(self, query: str, *, run_manager=None):
        docs = []

        def retrieve(r):
            # Attempt to retrieve documents from each retriever in parallel
            try:
                # Use the appropriate retrieval method and limit results to k_per_doc
                retrieved_docs = r.get_relevant_documents(query)
                return retrieved_docs[:self._k_per_doc]
            except Exception as e:
                logging.error(f"Error during retrieval for a document: {e}")
                return []

        # Execute retrieval in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            # Map the retrieve function to all retriever objects
            results = executor.map(retrieve, self._retrievers)

        # Deduplicate results based on source and chunk metadata
        seen = set()
        unique_docs = []
        for dlist in results:
            for d in dlist:
                # Create a unique key using 'source' (doc_id) and 'chunk' number
                key = f"{d.metadata.get('source')}_{d.metadata.get('chunk')}"
                if key not in seen:
                    seen.add(key)
                    unique_docs.append(d)

        logging.info(f"Query='{query}' found {len(unique_docs)} unique documents across all sources.")
        return unique_docs

# -------------------- Load multiple vectorstores --------------------
def load_all_vectorstores(doc_ids: Optional[List[str]] = None, top_k: int = 5) -> MultiDocRetriever:
    """
    Load retrievers for multiple doc_ids (or all folders if doc_ids=None)
    Returns a MultiDocRetriever combining them
    """
    all_retrievers = []
    
    # Determine which folders to load
    folders_to_load = list_vectorstore_folders()
    if doc_ids:
        folders_to_load = [f for f in folders_to_load if f in doc_ids]

    for folder in folders_to_load:
        try:
            retriever = load_vectorstore(folder, top_k=top_k)
            all_retrievers.append(retriever)
        except ValueError as e:
            logging.warning(e) # Log warning if a specified vectorstore is missing

    if not all_retrievers:
        # Raise error only if no vectorstores were successfully loaded
        raise ValueError("No vectorstores found for the given doc_ids or the vectorstore directory is empty.")

    return MultiDocRetriever(retrievers_list=all_retrievers, k_per_doc=top_k)
