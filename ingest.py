# -------------------- ingest.py (optimized) --------------------
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
from vectorstore import save_to_vectorstore, vectorstore_exists

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
    text = text.replace('\xa0', ' ')
    patterns = [
        (r'([ก-๙])\s+([ก-๙])', r'\1\2'),
        (r'([A-Za-z])\s+([A-Za-z])', r'\1\2'),
        (r'(\d)\s+(\d)', r'\1\2'),
        (r'([ก-๙])\s+([A-Za-z0-9])', r'\1\2'),
        (r'([A-Za-z0-9])\s+([ก-๙])', r'\1\2')
    ]
    for pattern, repl in patterns:
        for _ in range(10):  # limit loop iterations to avoid infinite loops
            if re.search(pattern, text):
                text = re.sub(pattern, repl, text)
            else:
                break
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+\)', ')', text)
    text = re.sub(r'\[\s+', '[', text)
    text = re.sub(r'\s+\]', ']', text)
    text = re.sub(r'พ\s*\.\s*ศ\s*\.\s*(\d+)', r'พ. ศ. \1', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

# -------------------- Loaders --------------------
def load_txt(path):
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            return TextLoader(path, encoding=enc).load()
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot read text file {path} with utf-8 / latin-1 / cp1252")

def load_pdf(path):
    try:
        loader = PyPDFLoader(path)
        docs = loader.load()
        text_content = " ".join([d.page_content for d in docs]).strip()
        if len(text_content) < 50:  # fallback OCR if too little text
            loader = UnstructuredPDFLoader(path, mode="elements", ocr=True)
            docs = loader.load()
        for i, doc in enumerate(docs, start=1):
            doc.page_content = clean_text(doc.page_content)
            doc.metadata["source"] = os.path.basename(path)
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
def process_document(file_path, year: Optional[int] = None, version: str = "v1", metadata: dict = None):
    file_name = os.path.basename(file_path)
    ext = os.path.splitext(file_name)[1].lower()
    if ext not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported file type: {file_name}")
    
    doc_id = os.path.splitext(file_name)[0]
    if year:
        doc_id = f"{year}-{doc_id}-{version}"

    if vectorstore_exists(doc_id):
        logging.info(f"Vectorstore exists for {doc_id}, skipping ingestion")
        return doc_id

    loader_func = FILE_LOADER_MAP[ext]
    docs = loader_func(file_path)
    if not docs:
        logging.warning(f"No content loaded from {file_name}")
        return doc_id

    # Add metadata
    for doc in docs:
        doc.metadata.setdefault("source", file_name)
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
    save_to_vectorstore(doc_id, chunk_texts, metadata=metadata or {})
    logging.info(f"Processed {file_name} -> doc_id: {doc_id} ({len(chunks)} chunks)")
    return doc_id

# -------------------- Batch ingestion --------------------
def ingest_all_files(data_dir: str = DATA_DIR, year: Optional[int] = None, version: str = "v1"):
    os.makedirs(data_dir, exist_ok=True)
    files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    results = []

    with ThreadPoolExecutor() as executor:
        future_to_file = {
            executor.submit(process_document, os.path.join(data_dir, f), year, version): f
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
    files = []
    for f in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, f)
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
    path = os.path.join(DATA_DIR, doc_id)
    if os.path.exists(path):
        os.remove(path)
    vectordir = os.path.join(VECTORSTORE_DIR, doc_id)
    if os.path.exists(vectordir):
        import shutil
        shutil.rmtree(vectordir)
    logging.info(f"Deleted document and vectorstore: {doc_id}")
