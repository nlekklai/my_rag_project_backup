#core/ingest.py
import os
import re
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
    CSVLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .vectorstore import save_to_vectorstore, vectorstore_exists # แก้ไข: ใช้ Relative Import

DATA_DIR = "data"
VECTORSTORE_DIR = "vectorstore"
SUPPORTED_TYPES = [".pdf", ".docx", ".txt", ".xlsx", ".pptx", ".md", ".csv"]

# -------------------- Logging --------------------
logging.basicConfig(
    filename="ingest.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# -------------------- Text Cleaning --------------------
def clean_text(text: str) -> str:
    """
    Performs advanced text cleaning, crucial for high-quality embeddings. 
    Fixes common spacing errors in Thai/English text (e.g., 'พ . ศ . 2567').
    """
    text = text.replace('\xa0', ' ')
    # Patterns to remove excessive spacing between adjacent characters/numbers/Thai letters
    patterns = [
        (r'([ก-๙])\s+([ก-๙])', r'\1\2'),
        (r'([A-Za-z])\s+([A-Za-z])', r'\1\2'),
        (r'(\d)\s+(\d)', r'\1\2'),
        (r'([ก-๙])\s+([A-Za-z0-9])', r'\1\2'),
        (r'([A-Za-z0-9])\s+([ก-๙])', r'\1\2')
    ]
    for pattern, repl in patterns:
        for _ in range(10):  # Loop to catch multiple spaces/patterns recursively
            if re.search(pattern, text):
                text = re.sub(pattern, repl, text)
            else:
                break
    
    # Clean up punctuation spacing
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+\)', ')', text)
    text = re.sub(r'\[\s+', '[', text)
    text = re.sub(r'\s+\]', ']', text)
    
    # Specific cleanup for common date formats (e.g., พ.ศ. 2567)
    text = re.sub(r'พ\s*\.\s*ศ\s*\.\s*(\d+)', r'พ.ศ. \1', text) 
    text = re.sub(r'\s{2,}', ' ', text) # Replace multiple spaces with a single space
    return text.strip()

# -------------------- Loaders --------------------
def load_txt(path):
    """Load text files with encoding fallback."""
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            return TextLoader(path, encoding=enc).load()
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot read text file {path} with utf-8 / latin-1 / cp1252")

def load_pdf(path):
    """Load PDF, using OCR fallback if PyPDFLoader finds minimal content."""
    try:
        loader = PyPDFLoader(path)
        docs = loader.load()
        # Fallback to UnstructuredPDFLoader with OCR if initial load is too sparse
        text_content = " ".join([d.page_content for d in docs]).strip()
        if len(text_content) < 50: 
            logging.warning(f"PDF {path} returned sparse content. Trying UnstructuredOCR.")
            loader = UnstructuredPDFLoader(path, mode="elements", ocr=True)
            docs = loader.load()
        
        for i, doc in enumerate(docs, start=1):
            doc.page_content = clean_text(doc.page_content)
            # Ensure metadata exists for MultiDocRetriever to identify source
            doc.metadata["source_file"] = os.path.basename(path) 
            doc.metadata["page"] = i
        return docs
    except Exception as e:
        logging.error(f"Failed to load PDF {path}: {e}")
        return []

def load_docx(path):
    return UnstructuredWordDocumentLoader(path).load()

def load_xlsx(path):
    return UnstructuredExcelLoader(path).load()

def load_pptx(path):
    return UnstructuredPowerPointLoader(path).load()

def load_md(path):
    return TextLoader(path, encoding="utf-8").load()

def load_csv(path):
    return CSVLoader(path).load()

FILE_LOADER_MAP = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".txt": load_txt,
    ".xlsx": load_xlsx,
    ".pptx": load_pptx,
    ".md": load_md,
    ".csv": load_csv,
}

# -------------------- Process single document --------------------
def process_document(file_path, file_name: str, year: Optional[int] = None, version: str = "v1", metadata: dict = None):
    """
    Loads, cleans, chunks, and saves a single document to its dedicated vectorstore.
    This is called in Step 2 of the main workflow.
    """
    ext = os.path.splitext(file_name)[1].lower()
    if ext not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported file type: {file_name}")
    
    # Create doc_id based on file name only (folder name used as source context)
    doc_id = os.path.splitext(file_name)[0] 
    if year:
        doc_id = f"{year}-{doc_id}-{version}"

    # Use the folder name (e.g., 'rubrics', 'evidence') as the actual doc_id for vectorstore directory
    # This assumes file_path is something like 'assessment_data/rubrics/file.txt'
    # We use the parent directory name as the unique vectorstore identifier for MultiDocRetriever
    vectorstore_doc_id = os.path.basename(os.path.dirname(file_path)) 

    # We need to skip ingestion if the whole folder vectorstore already exists for the workflow
    # However, since the workflow indexes per file but stores in a shared folder (e.g., 'rubrics'), 
    # we proceed with indexing to ensure all files are included in the 'rubrics' vectorstore. 
    # NOTE: The current vectorstore logic assumes one vectorstore per file, not per folder.
    # We will use the file name as the doc_id to create an individual vectorstore per file for safety.

    final_doc_id = vectorstore_doc_id # Use folder name as the collection name (e.g., 'rubrics')
    
    if vectorstore_exists(final_doc_id):
        # We assume the user wants to re-index all files in the folder if the folder exists,
        # unless we implement more complex versioning per file inside the folder.
        # For simplicity, we proceed and overwrite the vectorstore or skip if robustly checked.
        # For now, let's skip the existence check here since the workflow should handle it.
        pass


    loader_func = FILE_LOADER_MAP[ext]
    docs = loader_func(file_path)
    if not docs:
        logging.warning(f"No content loaded from {file_name}")
        return final_doc_id # Still return the ID even if no content

    # Add metadata
    for doc in docs:
        # Use the file name as the source_file metadata for better tracking
        doc.metadata.setdefault("source_file", file_name) 
        if metadata:
            doc.metadata.update(metadata)
        if year:
            doc.metadata["year"] = year
        doc.metadata["version"] = version

    # Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    for c in chunks:
        c.page_content = clean_text(c.page_content)

    chunk_texts = [c.page_content for c in chunks]
    # Pass the file name as metadata for better tracking in the vectorstore
    save_to_vectorstore(final_doc_id, chunk_texts, metadata={"source_file": file_name} or {})
    logging.info(f"Processed {file_name} -> doc_id: {final_doc_id} ({len(chunks)} chunks)")
    return final_doc_id

# -------------------- Batch ingestion --------------------
def ingest_all_files(data_dir: str = DATA_DIR, year: Optional[int] = None, version: str = "v1"):
    """
    Ingest all files in a given directory using parallel execution.
    NOTE: This function is primarily for the older '/ingest' endpoint, 
    but the main workflow uses a loop calling process_document directly.
    """
    os.makedirs(data_dir, exist_ok=True)
    files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    results = []

    with ThreadPoolExecutor() as executor:
        future_to_file = {
            executor.submit(process_document, os.path.join(data_dir, f), f, year, version): f
            for f in files
        }
        for future in as_completed(future_to_file):
            f = future_to_file[future]
            try:
                doc_id = future.result()
                results.append({"file": f, "doc_id": doc_id, "status": "processed"})
            except Exception as e:
                logging.error(f"Error processing {f}: {e}")
                results.append({"file": f, "doc_id": None, "status": "failed"})
    return results

# -------------------- List & Delete --------------------
def list_documents():
    """List documents in the main DATA_DIR."""
    files = []
    for f in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, f)
        if not os.path.isfile(path):
            continue
            
        stat = os.stat(path)
        doc_id = os.path.splitext(f)[0]
        files.append({
            "id": doc_id,
            "filename": f,
            "file_type": os.path.splitext(f)[1].lower(),
            "upload_date": datetime.utcfromtimestamp(stat.st_mtime).isoformat(),
            # NOTE: vectorstore_exists check here needs refinement based on RAG structure
            "status": "processed" if vectorstore_exists(doc_id) else "pending"
        })
    return files

def delete_document(doc_id):
    """Delete the original file and its corresponding vectorstore."""
    # This logic assumes doc_id is the original file name without extension
    # This might need adjustment based on how the assessment pipeline handles deletion
    
    # 1. Try to find the file in DATA_DIR
    file_found = False
    for f in os.listdir(DATA_DIR):
        if os.path.splitext(f)[0] == doc_id:
            path = os.path.join(DATA_DIR, f)
            if os.path.exists(path):
                os.remove(path)
                file_found = True
                break
    
    # 2. Delete Vector Store
    vectordir = os.path.join(VECTORSTORE_DIR, doc_id)
    if os.path.exists(vectordir):
        import shutil
        shutil.rmtree(vectordir)
        logging.info(f"Deleted vectorstore: {doc_id}")
        
    if file_found:
        logging.info(f"Deleted source file: {doc_id}")
    else:
        logging.warning(f"File not found for deletion: {doc_id}")

def list_vectorstore_folders() -> list[str]:
    """
    Returns a list of vectorstore folder names currently existing in VECTORSTORE_DIR.
    These folder names are used as doc_ids in RAG queries.
    """
    if not os.path.exists(VECTORSTORE_DIR):
        return []
    
    folders = [
        name for name in os.listdir(VECTORSTORE_DIR)
        if os.path.isdir(os.path.join(VECTORSTORE_DIR, name))
    ]
    return folders
