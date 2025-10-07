# core/ingest.py
import os
import re
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Set, Iterable, Dict, Any

# Document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
    CSVLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# vectorstore helpers
from .vectorstore import save_to_vectorstore, vectorstore_exists

# Optional OCR
try:
    from pdf2image import convert_from_path
    import pytesseract
    _HAS_PDF2IMAGE = True
except Exception:
    convert_from_path = None
    pytesseract = None
    _HAS_PDF2IMAGE = False

# -------------------- Config --------------------
DATA_DIR = "data"
VECTORSTORE_DIR = "vectorstore"
SUPPORTED_TYPES = [".pdf", ".docx", ".txt", ".xlsx", ".pptx", ".md", ".csv"]

# Logging
logging.basicConfig(
    filename="ingest.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------- Text Cleaning --------------------
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace('\xa0', ' ').replace('\u200b', '').replace('\u00ad', '')
    text = re.sub(r'[\uFFFD\u2000-\u200F\u2028-\u202F\u2060-\u206F\uFEFF]', '', text)
    ocr_replacements = {
        "สำนักงน": "สำนักงาน",
        "คณะกรรมกร": "คณะกรรมการ",
        "รัฐวิสหกิจ": "รัฐวิสาหกิจ",
        "นโยบย": "นโยบาย",
        "ดาน": "ด้าน",
        "การดาเนนงาน": "การดำเนินงาน",
        "การดาเนน": "การดำเนิน",
        "ท\"": "ที่",
    }
    for bad, good in ocr_replacements.items():
        text = text.replace(bad, good)
    text = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\u0E00-\u0E7F]', '', text)
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+\)', ')', text)
    text = re.sub(r'\[\s+', '[', text)
    text = re.sub(r'\s+\]', ']', text)
    text = re.sub(r'\s+([,.:;?!])', r'\1', text)
    text = re.sub(r'พ\s*\.\s*ศ\s*\.\s*(\d+)', r'พ.ศ. \1', text)
    text = re.sub(r'([ก-๙])([A-Za-z0-9])', r'\1 \2', text)
    text = re.sub(r'([A-Za-z0-9])([ก-๙])', r'\1 \2', text)
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

# -------------------- Loaders --------------------
def load_txt(path: str) -> List[Document]:
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            loader = TextLoader(path, encoding=enc)
            return loader.load()
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot read text file {path} with utf-8 / latin-1 / cp1252")

def _ocr_page_from_pdf(pdf_path: str, page_number: int, lang: str = "tha+eng") -> str:
    if not _HAS_PDF2IMAGE:
        logger.warning("pdf2image/pytesseract not available; OCR skipped.")
        return ""
    try:
        images = convert_from_path(pdf_path, first_page=page_number, last_page=page_number, dpi=300)
        if not images:
            return ""
        img = images[0]
        return (pytesseract.image_to_string(img, lang=lang) or "").strip()
    except Exception as e:
        logger.warning(f"OCR error {os.path.basename(pdf_path)} page {page_number}: {e}")
        return ""

def load_pdf(path: str, ocr_pages: Optional[Iterable[int]] = None) -> List[Document]:
    docs: List[Document] = []
    try:
        loader = PyPDFLoader(path)
        pages = loader.load_and_split()
    except Exception as e:
        logger.warning(f"PyPDFLoader failed for {path}: {e}.")
        pages = []
    total_pages = len(pages)
    if total_pages == 0 and _HAS_PDF2IMAGE:
        try:
            images = convert_from_path(path, dpi=50)
            total_pages = len(images)
        except Exception as e:
            logger.error(f"Cannot determine pages for {path}: {e}")
            total_pages = 0
    for i in range(1, total_pages + 1):
        text = (pages[i-1].page_content.strip() if i <= len(pages) else "")
        force_ocr = ocr_pages and i in set(ocr_pages)
        if (not text or len(text) < 50) or force_ocr:
            if _HAS_PDF2IMAGE:
                ocr_text = _ocr_page_from_pdf(path, i)
                if ocr_text:
                    text = ocr_text
        if text:
            docs.append(Document(page_content=text, metadata={"source_file": os.path.basename(path), "page": i}))
    # Deduplicate
    seen = set()
    deduped = []
    for d in docs:
        key = d.page_content.strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(d)
    return deduped

def load_docx(path: str) -> List[Document]:
    try: return UnstructuredWordDocumentLoader(path).load()
    except Exception as e: logger.error(f"Failed to load .docx {path}: {e}"); return []

def load_xlsx(path: str) -> List[Document]:
    try: return UnstructuredExcelLoader(path).load()
    except Exception as e: logger.error(f"Failed to load .xlsx {path}: {e}"); return []

def load_pptx(path: str) -> List[Document]:
    try: return UnstructuredPowerPointLoader(path).load()
    except Exception as e: logger.error(f"Failed to load .pptx {path}: {e}"); return []

def load_md(path: str) -> List[Document]:
    try: return TextLoader(path, encoding="utf-8").load()
    except Exception as e: logger.error(f"Failed to load .md {path}: {e}"); return []

def load_csv(path: str) -> List[Document]:
    try: return CSVLoader(path).load()
    except Exception as e: logger.error(f"Failed to load .csv {path}: {e}"); return []

FILE_LOADER_MAP = {
    ".pdf": load_pdf, ".docx": load_docx, ".txt": load_txt,
    ".xlsx": load_xlsx, ".pptx": load_pptx, ".md": load_md, ".csv": load_csv
}

# -------------------- Process single document --------------------
def process_document(
    file_path: str,
    file_name: str,
    collection_id: Optional[str] = None,
    doc_type: Optional[str] = None,
    base_path: str = VECTORSTORE_DIR,
    year: Optional[int] = None,
    version: str = "v1",
    metadata: dict = None,
    ocr_pages: Optional[Iterable[int]] = None
) -> str:
    final_doc_id = collection_id if collection_id else os.path.splitext(file_name)[0]
    ext = os.path.splitext(file_name)[1].lower()
    if ext not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported file type: {file_name}")
    loader_func = FILE_LOADER_MAP.get(ext)
    docs = loader_func(file_path) if ext != ".pdf" else load_pdf(file_path, ocr_pages=ocr_pages)
    if not docs:
        logger.warning(f"No content loaded from {file_name}")
        return final_doc_id
    for doc in docs:
        doc.metadata.setdefault("source_file", file_name)
        if metadata: doc.metadata.update(metadata)
        if year: doc.metadata["year"] = year
        doc.metadata["version"] = version
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    for idx, c in enumerate(chunks, start=1):
        c.page_content = clean_text(c.page_content)
        c.metadata["chunk_index"] = idx
    chunk_texts = [c.page_content for c in chunks]
    # Save with doc_type subfolder
    try:
        save_to_vectorstore(
            final_doc_id,
            chunk_texts,
            metadata={"source_file": file_name},
            doc_type=doc_type,  # ✅ correct
            base_path=base_path
        )

        logger.info(f"Processed {file_name} -> doc_id: {final_doc_id} ({len(chunks)} chunks) in doc_type={doc_type}")
    except Exception as e:
        logger.error(f"Failed to save vectorstore for {final_doc_id}: {e}")
        raise
    return final_doc_id

# -------------------- Batch ingestion --------------------
def ingest_all_files(
    data_dir: str = DATA_DIR,
    doc_type: Optional[str] = None,
    base_path: str = VECTORSTORE_DIR,
    exclude_dirs: Set[str] = set(),
    year: Optional[int] = None,
    version: str = "v1",
    sequential: bool = True
) -> List[Dict[str, Any]]:
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(base_path, exist_ok=True)
    files_to_process = [
        f for f in os.listdir(data_dir)
        if os.path.isfile(os.path.join(data_dir, f))
    ]
    results = []
    def _process_file(f):
        file_path = os.path.join(data_dir, f)
        doc_id = os.path.splitext(f)[0]
        return process_document(
            file_path=file_path,
            file_name=f,
            collection_id=doc_id,
            doc_type=doc_type,
            base_path=base_path,
            year=year,
            version=version
        )
    if sequential:
        for f in files_to_process:
            try:
                doc_id = _process_file(f)
                results.append({"file": f, "doc_id": doc_id, "status": "processed"})
            except Exception as e:
                logger.error(f"Error processing {f}: {e}")
                results.append({"file": f, "doc_id": None, "status": "failed", "error": str(e)})
    else:
        with ThreadPoolExecutor() as executor:
            future_to_file = {executor.submit(_process_file, f): f for f in files_to_process}
            for future in as_completed(future_to_file):
                f = future_to_file[future]
                try:
                    doc_id = future.result()
                    results.append({"file": f, "doc_id": doc_id, "status": "processed"})
                except Exception as e:
                    logger.error(f"Error processing {f}: {e}")
                    results.append({"file": f, "doc_id": None, "status": "failed", "error": str(e)})
    return results

# -------------------- List & Delete --------------------
def list_documents(doc_type: Optional[str] = None) -> List[dict]:
    """
    List files in DATA_DIR with vectorstore status.
    Status values:
      - 'Ingested'  -> vectorstore exists
      - 'Pending'   -> vectorstore not exists
      - 'Error'     -> file exists but vectorstore folder missing after failed processing
    """
    files = []
    for f in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, f)
        if not os.path.isfile(path):
            continue
        stat = os.stat(path)
        doc_id = os.path.splitext(f)[0]

        # Determine vectorstore folder
        vectordir = None
        if doc_type:
            vectordir = os.path.join(VECTORSTORE_DIR, doc_type, doc_id)
        else:
            # Scan all subfolders
            if os.path.exists(VECTORSTORE_DIR):
                for sub in os.listdir(VECTORSTORE_DIR):
                    candidate = os.path.join(VECTORSTORE_DIR, sub, doc_id)
                    if os.path.isdir(candidate):
                        vectordir = candidate
                        break
            if vectordir is None:
                vectordir = os.path.join(VECTORSTORE_DIR, doc_id)

        # Map status
        if os.path.exists(vectordir):
            status = "Ingested"
        else:
            status = "Pending"  # Could extend to "Error" if needed

        files.append({
            "id": doc_id,
            "filename": f,
            "file_type": os.path.splitext(f)[1].lower(),
            "upload_date": datetime.utcfromtimestamp(stat.st_mtime).isoformat(),
            "status": status
        })
    return files


def delete_document(doc_id: str, doc_type: Optional[str] = None):
    """Delete original file and vectorstore folder (supports doc_type subfolder)."""
    # Delete source file
    file_found = False
    for f in os.listdir(DATA_DIR):
        if os.path.splitext(f)[0] == doc_id:
            path = os.path.join(DATA_DIR, f)
            try:
                os.remove(path)
                file_found = True
                logger.info(f"Deleted source file: {path}")
            except Exception as e:
                logger.error(f"Failed to delete source file {path}: {e}")
            break
    # Delete vectorstore folder
    vectordirs = []
    if doc_type:
        vectordirs.append(os.path.join(VECTORSTORE_DIR, doc_type, doc_id))
    else:
        # Search all subfolders
        if os.path.exists(VECTORSTORE_DIR):
            for sub in os.listdir(VECTORSTORE_DIR):
                candidate = os.path.join(VECTORSTORE_DIR, sub, doc_id)
                if os.path.isdir(candidate):
                    vectordirs.append(candidate)
            # Also check root
            root_candidate = os.path.join(VECTORSTORE_DIR, doc_id)
            if os.path.isdir(root_candidate):
                vectordirs.append(root_candidate)
    import shutil
    for vd in vectordirs:
        try:
            shutil.rmtree(vd)
            logger.info(f"Deleted vectorstore: {vd}")
        except Exception as e:
            logger.error(f"Failed to delete vectorstore {vd}: {e}")
    if not file_found:
        logger.warning(f"File not found for deletion: {doc_id}")

def list_vectorstore_folders() -> List[str]:
    """List vectorstore folders as doc_ids (include subfolders)."""
    if not os.path.exists(VECTORSTORE_DIR):
        return []
    folders = []
    for root, dirs, _ in os.walk(VECTORSTORE_DIR):
        for d in dirs:
            folders.append(d)
    return folders

# -------------------- Manual run --------------------
if __name__ == "__main__":
    res = ingest_all_files(data_dir=DATA_DIR, sequential=True)
    print("Ingestion results:")
    for r in res:
        print(r)
