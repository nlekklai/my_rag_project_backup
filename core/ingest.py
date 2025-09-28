import os
import re
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Set # ADDED: Set for directory exclusion
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
from .vectorstore import save_to_vectorstore, vectorstore_exists 

DATA_DIR = "data"
VECTORSTORE_DIR = "vectorstore"
SUPPORTED_TYPES = [".pdf", ".docx", ".txt", ".xlsx", ".pptx", ".md", ".csv"]

# -------------------- Logging --------------------
logging.basicConfig(
    filename="ingest.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# -------------------- Text Cleaning (IMPROVED with OCR Replacements) --------------------
def clean_text(text: str) -> str:
    """
    Clean Thai/English text for vectorstore ingestion.
    - NEW: Fix common OCR word errors (e.g., สำนักงน -> สำนักงาน).
    - Remove garbled/non-printable characters, control codes, and common decoding noise.
    - Normalize spacing and punctuation.
    - Fix common OCR errors by re-inserting spaces where necessary.
    """
    import re

    # --- Step 1: Character Normalization and Removal ---
    # Remove non-standard spacing, zero-width characters, soft hyphens, control codes, and decoding failure artifacts (U+FFFD).
    text = text.replace('\xa0', ' ').replace('\u200b', ' ').replace('\u00ad', '')
    text = re.sub(r'[\uFFFD\u2000-\u200F\u2028-\u202F\u2060-\u206F\uFEFF]', '', text) 

    # --- Step 2: Aggressive OCR Word Replacement (BEFORE other cleaning) ---
    # Targets specific, frequent OCR errors observed in documents like 'seam'.
    ocr_replacements = {
        "สำนักงน": "สำนักงาน",
        "คณะกรรมกร": "คณะกรรมการ",
        "รัฐวิสหกิจ": "รัฐวิสาหกิจ",
        "นโยบย": "นโยบาย",
        "ดาน": "ด้าน",
        "การดาเนนงาน": "การดำเนินงาน",
        "การดาเนน": "การดำเนิน",
    }
    for wrong, correct in ocr_replacements.items():
        # Replace common OCR word fragments.
        text = text = text.replace(wrong, correct)


    # --- Step 3: Punctuation and Spacing Cleanup ---
    # 3a. Remove all residual non-printable/garbled characters outside the main range.
    text = re.sub(r'[^\x20-\x7E\sก-๙]', '', text) 
    
    # 3b. Clean up punctuation spacing and remove space before punctuation
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+\)', ')', text)
    text = re.sub(r'\[\s+', '[', text)
    text = re.sub(r'\s+\]', ']', text)
    text = re.sub(r'\s+([,.:;?!])', r'\1', text) 
    
    # 3c. Correct Thai date format (พ.ศ.)
    text = re.sub(r'พ\s*\.\s*ศ\s*\.\s*(\d+)', r'พ.ศ. \1', text)

    # --- Step 4: Fix OCR Glueing Errors ---
    # Re-insert a space where words were glued together (e.g., ระดับ1รัฐวิสาหกิจ)
    text = re.sub(r'([ก-๙])([a-zA-Z0-9])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z0-9])([ก-๙])', r'\1 \2', text)
    
    # --- Step 5: Final Normalization ---
    # Normalize multiple spaces → single space and strip
    text = re.sub(r'\s{2,}', ' ', text)

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
def process_document(file_path, file_name: str, 
                     collection_id: Optional[str] = None, 
                     year: Optional[int] = None, version: str = "v1", metadata: dict = None):
    """
    Loads, cleans, chunks, and saves a single document to its dedicated vectorstore.
    Uses collection_id if provided, otherwise infers from folder path.
    """
    ext = os.path.splitext(file_name)[1].lower()
    if ext not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported file type: {file_name}")
    
    # 1. Determine the final collection ID
    if collection_id:
        final_doc_id = collection_id
    else:
        # Fallback logic (uses parent directory, e.g., 'data' or 'rubrics')
        vectorstore_doc_id = os.path.basename(os.path.dirname(file_path)) 
        final_doc_id = vectorstore_doc_id
        
    if vectorstore_exists(final_doc_id):
        pass


    loader_func = FILE_LOADER_MAP[ext]
    docs = loader_func(file_path)
    if not docs:
        logging.warning(f"No content loaded from {file_name}")
        return final_doc_id

    # Add metadata
    for doc in docs:
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
        # This cleaning step ensures consistency for all file types
        c.page_content = clean_text(c.page_content) 

    chunk_texts = [c.page_content for c in chunks]
    # Pass the file name as metadata for better tracking in the vectorstore
    save_to_vectorstore(final_doc_id, chunk_texts, metadata={"source_file": file_name} or {})
    logging.info(f"Processed {file_name} -> doc_id: {final_doc_id} ({len(chunks)} chunks)")
    return final_doc_id

# -------------------- Batch ingestion --------------------
def ingest_all_files(data_dir: str = DATA_DIR, 
                     exclude_dirs: Set[str] = set(), # NEW: Optional set of directories to exclude
                     year: Optional[int] = None, 
                     version: str = "v1"):
    """
    Ingest all files found directly in the given directory using parallel execution.
    
    This function processes only top-level files and skips specified subdirectories.
    It also corrects the doc_id to be the filename (e.g., 'seam') rather than the 
    folder name ('data') for top-level files.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    files_to_process = []
    
    # Iterate through all items (files/dirs) in the data directory
    for item in os.listdir(data_dir):
        path = os.path.join(data_dir, item)
        
        # 1. Skip excluded directories (for clarity and future robustness)
        if os.path.isdir(path) and item in exclude_dirs:
            logging.info(f"Skipping excluded directory: {item}")
            continue
            
        # 2. Only process files (non-recursive)
        if os.path.isfile(path):
            files_to_process.append(item)

    results = []

    with ThreadPoolExecutor() as executor:
        future_to_file = {}
        for f in files_to_process:
            # CRITICAL FIX: Explicitly set collection_id to file name (e.g., 'seam') 
            # for correct vector store path, rather than letting it default to 'data'.
            file_id = os.path.splitext(f)[0]
            
            future = executor.submit(
                process_document, 
                os.path.join(data_dir, f), 
                f, 
                collection_id=file_id, # Explicitly use file name as collection ID
                year=year, 
                version=version
            )
            future_to_file[future] = f

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
            "status": "processed" if vectorstore_exists(doc_id) else "pending"
        })
    return files

def delete_document(doc_id):
    """Delete the original file and its corresponding vectorstore."""
    # This logic assumes doc_id is the original file name without extension
    
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
