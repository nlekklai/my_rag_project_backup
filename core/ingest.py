import os
import re
import logging
import unicodedata 
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Set, Iterable, Dict, Any, Union, Tuple
import pandas as pd

# Document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
    CSVLoader
)
try:
    # 🚨 FIX: ใช้ UnstructuredFileLoader จาก langchain_community
    from langchain_community.document_loaders import UnstructuredFileLoader 
except ImportError:
    from langchain.document_loaders import UnstructuredFileLoader 
    
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# vectorstore helpers - สมมติว่าไฟล์ vectorstore.py อยู่ใน .
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

# Try to import helper for filtering metadata
try:
    from langchain_community.vectorstores.utils import filter_complex_metadata as _imported_filter_complex_metadata
except Exception:
    _imported_filter_complex_metadata = None

# -------------------- Config --------------------
DATA_DIR = "data"
VECTORSTORE_DIR = "vectorstore"
SUPPORTED_TYPES = [".pdf", ".docx", ".txt", ".xlsx", ".pptx", ".md", ".csv", ".jpg", ".jpeg", ".png"]

# Logging
logging.basicConfig(
    filename="ingest.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------- Helper: safe metadata filter --------------------
def _safe_filter_complex_metadata(meta: Any) -> Dict[str, Any]:
    """
    Ensure metadata is serializable and safe for Chroma / storage.
    Prefer the imported `filter_complex_metadata` if available; otherwise fallback.
    """
    if _imported_filter_complex_metadata:
        try:
            if hasattr(meta, "items"):
                return _imported_filter_complex_metadata(meta)
            elif isinstance(meta, dict):
                return _imported_filter_complex_metadata(meta)
            else:
                return _imported_filter_complex_metadata(dict(meta))
        except Exception as e:
            logger.debug(f"_imported_filter_complex_metadata failed: {e}")
            # fallback to local
    # Local safe fallback: keep primitives and stringify complex types
    if not isinstance(meta, dict):
        return {}
    clean = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        elif isinstance(v, (list, tuple)):
            try:
                clean[k] = [str(x) for x in v]
            except Exception:
                clean[k] = str(v)
        elif isinstance(v, dict):
            try:
                clean[k] = {str(kk): str(vv) for kk, vv in v.items()}
            except Exception:
                clean[k] = str(v)
        else:
            try:
                clean[k] = str(v)
            except Exception:
                # skip if cannot serialize
                continue
    return clean

# -------------------- Normalization utility (FINAL REVISION) --------------------

def _normalize_doc_id(text: str) -> str:
    """
    Final Logic: doc_id is the filename with the last extension removed, 
    retaining all characters (including Thai and spaces). 
    This function primarily ensures no leading/trailing spaces remain.
    """
    if not text:
        return "default_doc"
    # 🚨 โค้ดเหลือแค่ .strip()
    doc_id = text.strip() 
    
    if not doc_id:
        return "default_doc"
        
    return doc_id


# -------------------- Text Cleaning --------------------
def clean_text(text: str) -> str:
    """Clean Thai/English text including typical OCR mistakes."""
    if not text:
        return ""
    text = text.replace('\xa0', ' ').replace('\u200b', '').replace('\u00ad', '')
    text = re.sub(r'[\uFFFD\u2000-\u200F\u2028-\u202F\u2060-\u206F\uFEFF]', '', text)
    # Collapse spaces between Thai characters that were incorrectly split
    text = re.sub(r'([ก-๙])\s{1,3}(?=[ก-๙])', r'\1', text)
    ocr_replacements = {
        "สำนักงน": "สำนักงาน", "คณะกรรมกร": "คณะกรรมการ", "รัฐวิสหกิจ": "รัฐวิสาหกิจ",
        "นโยบย": "นโยบาย", "ดาน": "ด้าน", "การดาเนนงาน": "การดำเนินงาน",
        "การดาเนน": "การดำเนิน", "ท\"": "ที่",
    }
    for bad, good in ocr_replacements.items():
        text = text.replace(bad, good)
    # remove chars outside ASCII and Thai ranges
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

def load_unstructured(path: str) -> List[Document]:
    """
    Loads complex/image files using UnstructuredFileLoader with enhanced error handling.
    """
    try:
        # 1. ลองโหลดด้วย mode="elements"
        try:
            loader = UnstructuredFileLoader(path, mode="elements")
            docs = loader.load()
            return docs
        except Exception as inner_e:
            error_message = str(inner_e)
            
            # ดักจับ Error เฉพาะเจาะจง
            if "NoneType" in error_message or "__str__ returned non-string" in error_message or "Unsupported file type" in error_message:
                logger.warning(
                    f"⚠️ Fallback: Unstructured mode='elements' failed for {os.path.basename(path)} (Error type: {error_message[:50]}). "
                    f"Attempting load without 'elements' mode."
                )
                
                # 2. Fallback: ลองโหลดแบบปกติ
                try:
                    loader_fallback = UnstructuredFileLoader(path)
                    docs_fallback = loader_fallback.load()
                    return docs_fallback
                except Exception as fallback_e:
                    # ถ้า Fallback ล้มเหลวอีกครั้ง ให้ Log เป็น Error และ Return เป็น List ว่าง
                    logger.error(f"❌ Unstructured final load failed for {os.path.basename(path)}: {fallback_e}")
                    return []
            
            # ถ้าเป็น Error อื่นๆ ที่ไม่รู้จัก ให้ส่งต่อไป
            raise inner_e 
            
    except Exception as e:
        # ดักจับ Error ที่มาจาก raise inner_e หรือ Error อื่นๆ ที่เกิดขึ้นในขั้นตอนการโหลด
        logger.error(f"❌ Failed to load unstructured file {path}: {e}")
        return []


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
        # pytesseract.image_to_string should return str; wrap in try/except to avoid breaking
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
    # if loader didn't provide pages, try to get page count via pdf2image
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
            # Metadata: source_file is used to retain original filename
            docs.append(Document(page_content=text, metadata={"source_file": os.path.basename(path), "page": i}))
    # Deduplicate by content
    seen = set()
    deduped = []
    for d in docs:
        key = d.page_content.strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(d)
    return deduped

def load_docx(path: str) -> List[Document]:
    try:
        return UnstructuredWordDocumentLoader(path).load()
    except Exception as e:
        logger.error(f"Failed to load .docx {path}: {e}")
        return []

def load_xlsx_generic_structured(path: str) -> List[Document]:
    try:
        xls = pd.ExcelFile(path)
    except Exception as e:
        logger.error(f"Failed to open Excel file {path}: {e}")
        # fallback to unstructured loader if available
        try:
            return UnstructuredExcelLoader(path).load()
        except Exception as e2:
            logger.error(f"UnstructuredExcelLoader fallback failed for {path}: {e2}")
            return []

    all_docs: List[Document] = []
    score_keywords = ["น้ำหนัก", "คะแนน", "score", "weight"]
    topic_keywords = ["หัวข้อ", "topic", "criteria_id"]
    category_keywords = ["หมวด", "category", "area"]
    detail_keywords = ["รายละเอียด", "เกณฑ์", "detail", "content", "criterion"]

    for sheet_name in xls.sheet_names:
        sheet_docs: List[Document] = []
        try:
            df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
            df.columns = [str(i) for i in range(len(df.columns))]
            header_row_index = -1
            best_match_count = 0
            for i in range(min(10, len(df))):
                row_str = " ".join(df.iloc[i].astype(str).str.lower().fillna('')).strip()
                score_match = any(k in row_str for k in score_keywords)
                detail_match = any(k in row_str for k in detail_keywords)
                current_match_count = score_match + detail_match + any(k in row_str for k in topic_keywords) + any(k in row_str for k in category_keywords)
                if current_match_count >= 2 and current_match_count > best_match_count:
                    header_row_index = i
                    best_match_count = current_match_count
            if header_row_index != -1:
                df_structured = pd.read_excel(xls, sheet_name=sheet_name, header=header_row_index)
                df_structured.columns = df_structured.columns.astype(str).str.strip().str.lower()
                def find_column(keywords):
                    for k in keywords:
                        matching_cols = [c for c in df_structured.columns if k.lower() in c]
                        if matching_cols:
                            return matching_cols[0]
                    return None
                score_col = find_column(score_keywords)
                topic_col = find_column(topic_keywords)
                category_col = find_column(category_keywords)
                detail_col = find_column(detail_keywords)
                if not detail_col:
                    logger.warning(f"Skipping structured load for sheet '{sheet_name}' - Missing Detail column.")
                    raise ValueError("Missing essential Detail column.")
                sheet_content_seen = set()
                for idx, row in df_structured.iterrows():
                    page_content_raw = str(row[detail_col]).strip()
                    if len(page_content_raw) < 10 or page_content_raw in sheet_content_seen:
                        continue
                    sheet_content_seen.add(page_content_raw)
                    metadata = {
                        "source_file": os.path.basename(path),
                        "sheet_name": sheet_name,
                        "row": idx + header_row_index + 2,
                        "score_or_weight": None,
                        "category": None,
                        "topic": None
                    }
                    if score_col:
                        val = row.get(score_col, None)
                        if pd.notna(val):
                            metadata["score_or_weight"] = float(val) if isinstance(val, (int,float)) else str(val).strip()
                    if topic_col:
                        metadata["topic"] = str(row.get(topic_col, "")).strip()
                    if category_col:
                        metadata["category"] = str(row.get(category_col, "")).strip()
                    context_prefix = []
                    if metadata["category"]:
                        context_prefix.append(f"หมวด: {metadata['category']}")
                    if metadata["topic"]:
                        context_prefix.append(f"หัวข้อ: {metadata['topic']}")
                    if metadata["score_or_weight"]:
                        context_prefix.append(f"เกณฑ์ระดับ/คะแนน: {metadata['score_or_weight']}")
                    page_content_final = f"{' | '.join(context_prefix)} | คำอธิบายเกณฑ์: {page_content_raw}" if context_prefix else page_content_raw
                    sheet_docs.append(Document(page_content=page_content_final, metadata=metadata))
                all_docs.extend(sheet_docs)
            else:
                logger.warning(f"No clear structured header in sheet '{sheet_name}'. Skipping structured load.")
        except Exception as e:
            logger.error(f"Error processing sheet '{sheet_name}' of {os.path.basename(path)}: {e}")
    if not all_docs:
        logger.warning(f"Structured load failed for {os.path.basename(path)}. Falling back to UnstructuredExcelLoader.")
        try:
            return UnstructuredExcelLoader(path).load()
        except Exception as e:
            logger.error(f"UnstructuredExcelLoader fallback failed for {path}: {e}")
            return []
    # Deduplicate final
    final_seen = set()
    final_docs = []
    for d in all_docs:
        key = d.page_content.strip()
        if key and key not in final_seen and len(key) >= 10:
            final_seen.add(key)
            final_docs.append(d)
    if not final_docs:
        logger.warning(f"Structured load resulted in zero documents after filtering. Falling back to UnstructuredExcelLoader.")
        try:
            return UnstructuredExcelLoader(path).load()
        except Exception as e:
            logger.error(f"UnstructuredExcelLoader fallback failed for {path}: {e}")
            return []
    return final_docs

def load_xlsx(path: str) -> List[Document]:
    return load_xlsx_generic_structured(path)

def load_pptx(path: str) -> List[Document]:
    try:
        return UnstructuredPowerPointLoader(path).load()
    except Exception as e:
        logger.error(f"Failed to load .pptx {path}: {e}")
        return []

def load_md(path: str) -> List[Document]:
    try:
        return TextLoader(path, encoding="utf-8").load()
    except Exception as e:
        logger.error(f"Failed to load .md {path}: {e}")
        return []

def load_csv(path: str) -> List[Document]:
    try:
        return CSVLoader(path).load()
    except Exception as e:
        logger.error(f"Failed to load .csv {path}: {e}")
        return []

FILE_LOADER_MAP = {
    ".pdf": load_pdf, ".docx": load_docx, ".txt": load_txt,
    ".xlsx": load_xlsx, ".pptx": load_pptx, ".md": load_md, ".csv": load_csv,
    ".jpg": load_unstructured, ".jpeg": load_unstructured, ".png": load_unstructured
}

# -------------------- Normalization utility --------------------
def normalize_loaded_documents(raw_docs: List[Any], source_path: Optional[str] = None) -> List[Document]:
    """
    Normalize loader outputs to a list of langchain.schema.Document
    """
    normalized: List[Document] = []
    for idx, item in enumerate(raw_docs):
        try:
            if isinstance(item, Document):
                doc = item
            elif isinstance(item, str):
                doc = Document(page_content=item, metadata={})
            elif isinstance(item, dict):
                content = item.get("page_content") or item.get("text") or item.get("content") or ""
                meta = item.get("metadata") or item.get("meta") or {}
                doc = Document(page_content=content, metadata=meta)
            else:
                # try common attributes
                text = None
                if hasattr(item, "page_content"):
                    text = getattr(item, "page_content")
                elif hasattr(item, "text"):
                    text = getattr(item, "text")
                elif hasattr(item, "get_text"):
                    try:
                        text = item.get_text()
                    except Exception:
                        text = str(item)
                else:
                    text = str(item)
                doc = Document(page_content=text, metadata={})
            # ensure metadata is a dict
            if not isinstance(doc.metadata, dict):
                doc.metadata = {"_raw_meta": str(doc.metadata)}
            # add source file if provided and not exists (used as source name for display/lookup)
            if source_path:
                doc.metadata.setdefault("source_file", os.path.basename(source_path))
            # sanitize metadata
            try:
                doc.metadata = _safe_filter_complex_metadata(doc.metadata)
            except Exception:
                doc.metadata = {}
            normalized.append(doc)
        except Exception as e:
            logger.warning(f"normalize_loaded_documents: skipping item #{idx} due to error: {e}")
            continue
    return normalized

# -------------------- Load & Chunk Document (for Retrieval/Mapping) --------------------
def load_and_chunk_document(
    file_path: str,
    doc_id: str,
    year: Optional[int] = None,
    version: str = "v1",
    metadata: Optional[Dict[str, Any]] = None,
    ocr_pages: Optional[Iterable[int]] = None
) -> List[Document]:
    """
    Load, clean, chunk — return list of Documents (not saving to vectorstore)
    """
    file_name = os.path.basename(file_path)
    ext = os.path.splitext(file_name)[1].lower()

    if ext not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported file type: {file_name}")

    loader_func = FILE_LOADER_MAP.get(ext)
    if not loader_func:
        logger.error(f"No loader for extension {ext}")
        return []

    # Load raw
    try:
        raw_docs = loader_func(file_path) if ext != ".pdf" else load_pdf(file_path, ocr_pages=ocr_pages)
    except Exception as e:
        logger.error(f"Loader {loader_func} raised for {file_path}: {e}")
        raw_docs = []

    if not raw_docs:
        logger.warning(f"No content loaded from {file_name}")
        return []

    # Normalize to Document objects
    docs = normalize_loaded_documents(raw_docs, source_path=file_path)

    # Update metadata with provided values
    for d in docs:
        if metadata:
            try:
                d.metadata.update(metadata)
            except Exception:
                d.metadata["injected_metadata"] = str(metadata)
        if year:
            d.metadata["year"] = year
        d.metadata["version"] = version
        d.metadata["doc_id"] = doc_id
        # d.metadata["source_file"] is set in normalize_loaded_documents, use it as 'source'
        d.metadata["source"] = d.metadata.get("source_file", file_name) 

    # Determine structured vs text
    is_structured_data = ext in [".xlsx", ".csv", ".jpg", ".jpeg", ".png"]

    if not is_structured_data:
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        try:
            chunks = splitter.split_documents(docs)
        except Exception as e:
            logger.warning(f"Text splitter failed on {file_name}: {e}. Falling back to using whole documents as chunks.")
            chunks = docs
    else:
        chunks = docs

    # Clean text and annotate chunk_index
    for idx, c in enumerate(chunks, start=1):
        c.page_content = clean_text(c.page_content)
        c.metadata["chunk_index"] = idx

    logger.info(f"Loaded and chunked {file_name} -> {len(chunks)} chunks.")
    return chunks

# -------------------- Process single document --------------------
def process_document(
    file_path: str,
    file_name: str,
    doc_id: Optional[str] = None,
    doc_type: Optional[str] = None,
    base_path: str = VECTORSTORE_DIR, 
    year: Optional[int] = None,
    version: str = "v1",
    metadata: dict = None,
    source_name_for_display: Optional[str] = None,
    ocr_pages: Optional[Iterable[int]] = None
) -> str:
    """
    Load -> chunk -> save to vectorstore. Returns doc_id used.
    """
    
    # 1. Normalize doc_id: ตัด Extension ออกก่อน แล้วใช้ _normalize_doc_id
    raw_doc_id_input = doc_id if doc_id else os.path.splitext(file_name)[0]
    final_doc_id = _normalize_doc_id(raw_doc_id_input) 
    
    if not final_doc_id or len(final_doc_id) < 3:
        raise ValueError(f"Validation error: Cannot generate a valid Doc ID from '{raw_doc_id_input}'. Got: {final_doc_id}")
        
    doc_type = doc_type or "default"
    ext = os.path.splitext(file_name)[1].lower()
    final_source_name = source_name_for_display or file_name
    
    if ext not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported file type: {file_name}")
    loader_func = FILE_LOADER_MAP.get(ext)
    if not loader_func:
        logger.error(f"No loader for extension {ext}")
        return final_doc_id

    # Load raw and normalize
    try:
        raw_docs = loader_func(file_path) if ext != ".pdf" else load_pdf(file_path, ocr_pages=ocr_pages)
    except Exception as e:
        logger.error(f"Loader {loader_func} raised for {file_path}: {e}")
        raw_docs = []

    if not raw_docs:
        logger.warning(f"No content loaded from {file_name}")
        return final_doc_id

    docs = normalize_loaded_documents(raw_docs, source_path=file_path)

    # ensure base metadata keys
    for d in docs:
        d.metadata.setdefault("source_file", file_name)
        if metadata:
            try:
                d.metadata.update(metadata)
            except Exception:
                d.metadata["injected_metadata"] = str(metadata)
        if year:
            d.metadata["year"] = year
        d.metadata["version"] = version
        d.metadata["source"] = final_source_name 
        d.metadata["doc_id"] = final_doc_id

    is_structured_data = ext in [".xlsx", ".csv", ".jpg", ".jpeg", ".png"]

    if not is_structured_data:
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        try:
            chunks = splitter.split_documents(docs)
        except Exception as e:
            logger.warning(f"Text splitter failed on {file_name}: {e}. Using original docs as chunks.")
            chunks = docs
    else:
        chunks = docs

    # Clean chunk content and metadata
    for idx, c in enumerate(chunks, start=1):
        c.page_content = clean_text(c.page_content)
        c.metadata["chunk_index"] = idx

    chunk_texts = [c.page_content for c in chunks]
    chunk_metadatas = [c.metadata for c in chunks]

    try:
        save_to_vectorstore(
            texts=chunk_texts,         
            metadatas=chunk_metadatas, 
            doc_id=final_doc_id,       
            doc_type=doc_type          
        )
        logger.info(f"Processed {file_name} -> doc_id: {final_doc_id} ({len(chunks)} chunks) in doc_type={doc_type}")
    except Exception as e:
        logger.error(f"Failed to save vectorstore for {final_doc_id}: {e}", exc_info=True)
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
    """Process all files in a folder (ThreadPoolExecutor if not sequential)"""
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(base_path, exist_ok=True)
    files_to_process = [
        f for f in os.listdir(data_dir)
        if os.path.isfile(os.path.join(data_dir, f))
    ]
    results = []

    def _process_file(f):
        file_path = os.path.join(data_dir, f)
        return process_document(
            file_path=file_path,
            file_name=f,
            doc_type=doc_type or "default",
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
                raw_doc_id = os.path.splitext(f)[0]
                error_doc_id = _normalize_doc_id(raw_doc_id)
                logger.error(f"Error processing {f}: {e}")
                results.append({"file": f, "doc_id": error_doc_id, "status": "failed", "error": str(e)})
    else:
        max_workers = min(8, (os.cpu_count() or 4))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(_process_file, f): f for f in files_to_process}
            for future in as_completed(future_to_file):
                f = future_to_file[future]
                try:
                    doc_id = future.result()
                    results.append({"file": f, "doc_id": doc_id, "status": "processed"})
                except Exception as e:
                    raw_doc_id = os.path.splitext(f)[0]
                    error_doc_id = _normalize_doc_id(raw_doc_id)
                    logger.error(f"Error processing {f}: {e}")
                    results.append({"file": f, "doc_id": error_doc_id, "status": "failed", "error": str(e)})
    return results

# -------------------- List & Delete (REVISED FINAL) --------------------
def list_documents(doc_types: Optional[List[str]] = None) -> List[dict]:
    """
    แสดงรายการไฟล์ที่ถูกอัปโหลดทั้งหมด โดยค้นหาไฟล์ต้นฉบับทั้งหมดภายใต้ DATA_DIR
    และตรวจสอบสถานะ Ingested จาก Vectorstore paths (ใช้ชื่อไฟล์ดิบที่ตัด ext เป็น doc_id)
    """
    files = []
    processed_doc_ids: Dict[str, str] = {} # doc_id -> filename
    
    for root, _, filenames in os.walk(DATA_DIR):
        for f in filenames:
            if f.startswith('.'):
                continue
            
            file_extension = os.path.splitext(f)[1].lower()
            if file_extension not in SUPPORTED_TYPES:
                continue

            path = os.path.join(root, f)
            stat = os.stat(path)
            
            # 1. ตัด Extension สุดท้ายออก (รองรับ double extension)
            raw_doc_id_no_ext = os.path.splitext(f)[0]
            # 2. ใช้ _normalize_doc_id (strip เท่านั้น)
            doc_id = _normalize_doc_id(raw_doc_id_no_ext)
            
            if doc_id in processed_doc_ids:
                logger.warning(f"Skipping duplicate file with same doc_id '{doc_id}': {f} (original: {processed_doc_ids[doc_id]})")
                continue
            
            processed_doc_ids[doc_id] = f

            # 3. ตรวจสอบสถานะ Ingested
            doc_type_candidates = ['document', 'faq'] 
            relative_path = os.path.relpath(root, DATA_DIR)
            current_folder_name = relative_path.split(os.sep)[0] if relative_path != '.' else 'document'
            
            if current_folder_name != 'document' and current_folder_name not in doc_type_candidates:
                 doc_type_candidates.insert(0, current_folder_name)
                 
            if doc_types:
                doc_type_candidates = [dt for dt in doc_type_candidates if dt in doc_types]
            
            is_ingested = False
            found_doc_type = None

            for dt in doc_type_candidates:
                # 🚨 ใช้ doc_id ที่เป็นชื่อไฟล์ดิบ (ตัด ext) ในการตรวจสอบ
                if vectorstore_exists(doc_id, doc_type=dt, base_path=VECTORSTORE_DIR):
                    is_ingested = True
                    found_doc_type = dt
                    break
            
            # 4. สร้างรายการผลลัพธ์
            file_entry = {
                "doc_id": doc_id, 
                "filename": f,
                "doc_type": found_doc_type or current_folder_name, 
                "file_type": file_extension,
                "upload_date": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "status": "Ingested" if is_ingested else "Not Ingested",
            }
            files.append(file_entry)
            
    return files

def delete_document(doc_id: str, doc_type: str = "document", base_path: str = VECTORSTORE_DIR) -> bool:
    """
    ลบไฟล์ต้นฉบับใน DATA_DIR และ vectorstore ที่เกี่ยวข้องกับ doc_id
    
    Returns True ถ้าลบสำเร็จ, False ถ้าไฟล์หรือ vectorstore ไม่พบ
    """
    # 1️⃣ ลบไฟล์ต้นฉบับใน DATA_DIR
    deleted_any = False
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            raw_doc_id_no_ext = os.path.splitext(f)[0]
            normalized_id = _normalize_doc_id(raw_doc_id_no_ext)
            if normalized_id == doc_id:
                path = os.path.join(root, f)
                try:
                    os.remove(path)
                    logger.info(f"Deleted original file: {path}")
                    deleted_any = True
                except Exception as e:
                    logger.error(f"Failed to delete file {path}: {e}")

    # 2️⃣ ลบ vectorstore folder ถ้ามี
    vs_path = os.path.join(base_path, doc_type, doc_id)
    if os.path.exists(vs_path):
        try:
            import shutil
            shutil.rmtree(vs_path)
            logger.info(f"Deleted vectorstore for doc_id={doc_id}, doc_type={doc_type}")
            deleted_any = True
        except Exception as e:
            logger.error(f"Failed to delete vectorstore {vs_path}: {e}")
    
    if not deleted_any:
        logger.warning(f"No file or vectorstore found for doc_id={doc_id}")
    
    return deleted_any
