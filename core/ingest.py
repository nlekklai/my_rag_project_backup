import os

# ตั้งค่า HF_HOME เป็นพาธที่ต้องการ
os.environ['HF_HOME'] = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface')
# (ทางเลือก) ป้องกันการเรียกใช้ TRANSFORMERS_CACHE 
if 'TRANSFORMERS_CACHE' in os.environ:
    del os.environ['TRANSFORMERS_CACHE']

import re
import logging
import unicodedata 
import json 
import uuid 
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Set, Iterable, Dict, Any, Union, Tuple, TypedDict
import pandas as pd
import shutil 
import numpy as np # Import numpy to handle potential np.float64 objects

# Document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
    CSVLoader
)
# --- Document Info Model ---
class DocInfo(TypedDict):
    """
    ข้อมูลสถานะของเอกสารที่ใช้สำหรับ API
    """
    doc_id: str             # Stable UUID
    doc_id_key: str         # Filename ID Key (normalized name)
    filename: str
    filepath: str
    doc_type: str           # Collection name
    upload_date: str        # ISO format
    chunk_count: int
    status: str             # "Ingested" | "Pending" | "Error"
    size: int               # File size in bytes

# แก้ไข: ลอง Import UnstructuredLoader ใหม่ก่อน (ตาม Deprecation Warning)
try:
    from langchain_unstructured.document_loaders import UnstructuredLoader as UnstructuredFileLoader
except ImportError:
    try:
        from langchain_community.document_loaders import UnstructuredFileLoader
    except ImportError:
        from langchain.document_loaders import UnstructuredFileLoader 
    
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# 🟢 Import Chroma และ Embeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

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
MAPPING_FILE_PATH = "data/doc_id_mapping.json" 
SUPPORTED_TYPES = [".pdf", ".docx", ".txt", ".xlsx", ".pptx", ".md", ".csv", ".jpg", ".jpeg", ".png"]

# 🔴 แก้ไข: ใช้ 'statement' ตามที่ผู้ใช้แจ้งมา
SUPPORTED_DOC_TYPES = ["document", "policy", "report", "statement", "evidence", "feedback"] 

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
    This is an aggressive filter designed to handle nested dicts/lists 
    and specific object representations (like np.float64 strings from Unstructured).
    """
    
    if not isinstance(meta, dict):
        # Attempt to convert to dict if possible
        if hasattr(meta, "items"):
            meta_dict = dict(meta.items())
        else:
            return {} # If it's not a dict, we can't process it safely

    meta_dict = meta
    clean = {}

    for k, v in meta_dict.items():
        # Skip None values
        if v is None:
            continue
            
        # 1. Handle primitive types (str, int, float, bool)
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        # 2. Handle nested dictionary (e.g., the 'points' dict object from Unstructured)
        elif isinstance(v, dict):
            try:
                # 📌 New: Try to serialize to JSON string
                clean[k] = json.dumps(v)
            except TypeError:
                # If JSON serialization fails (e.g., contains non-JSON serializable objects), stringify it
                clean[k] = str(v)
        # 3. Handle list or tuple
        elif isinstance(v, (list, tuple)):
            try:
                # 📌 New: Aggressively convert all elements within the list/tuple to string
                cleaned_list = []
                for item in v:
                    if isinstance(item, (str, int, float, bool)):
                        cleaned_list.append(item)
                    elif isinstance(item, (np.float64, np.float32)): # Explicitly handle numpy floats
                        cleaned_list.append(float(item))
                    else:
                        # Stringify any other complex object (like nested dicts, lists, or custom classes)
                        cleaned_list.append(str(item))
                        
                # ChromaDB is sensitive to lists of strings being too complex.
                # If the list contains mixed types or seems too complex, we stringify the whole thing.
                # However, for robustness, we'll return the list of cleaned primitives/strings.
                clean[k] = [str(x) for x in cleaned_list] # Ensure final output is a list of strings/primitives
                
            except Exception:
                # Fallback: Stringify the entire list/tuple if processing fails
                clean[k] = str(v)
        # 4. Handle other complex objects (e.g., numpy types directly if not caught above)
        else:
            try:
                clean[k] = str(v)
            except Exception:
                continue
                
    # Use LangChain's filter utility as a final safety layer if available
    if _imported_filter_complex_metadata:
        try:
            return _imported_filter_complex_metadata(clean)
        except Exception as e:
            logger.debug(f"LangChain filter failed after local cleanup: {e}")
            pass # Continue with locally cleaned data

    return clean


# -------------------- Normalization utility --------------------

def _normalize_doc_id(text: str) -> str:
    """
    Final Logic: doc_id is the filename with the last extension removed, 
    retaining all characters (including Thai and spaces). 
    """
    if not text:
        return "default_doc"
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
    text = re.sub(r'([ก-๙])\s{1,3}(?=[ก-๙])', r'\1', text)
    ocr_replacements = {
        "สำนักงน": "สำนักงาน", "คณะกรรมกร": "คณะกรรมการ", "รัฐวิสหกิจ": "รัฐวิสาหกิจ",
        "นโยบย": "นโยบาย", "ดาน": "ด้าน", "การดาเนนงาน": "การดำเนินงาน",
        "การดาเนน": "การดำเนิน", "ท\"": "ที่",
    }
    for bad, good in ocr_replacements.items():
        text = text.replace(bad, good)
    text = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\u0E00-\u0E7F]', '', text)
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+\)', ')', text)
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

# -------------------- Loaders --------------------

def load_unstructured(path: str) -> List[Document]:
    try:
        try:
            # 📌 UnstructuredFileLoader จะถูก import ตาม Logic ในส่วน Imports
            # ใช้ mode="elements" สำหรับเอกสารที่มีโครงสร้าง เช่น รูปภาพ (Image files)
            loader = UnstructuredFileLoader(path, mode="elements")
            docs = loader.load()
            return docs
        except Exception as inner_e:
            error_message = str(inner_e)
            if "NoneType" in error_message or "__str__ returned non-string" in error_message or "Unsupported file type" in error_message:
                logger.warning(
                    f"⚠️ Fallback: Unstructured mode='elements' failed for {os.path.basename(path)} (Error type: {error_message[:50]}). "
                    f"Attempting load without 'elements' mode."
                )
                try:
                    loader_fallback = UnstructuredFileLoader(path)
                    docs_fallback = loader_fallback.load()
                    return docs_fallback
                except Exception as fallback_e:
                    logger.error(f"❌ Unstructured final load failed for {os.path.basename(path)}: {fallback_e}")
                    return []
            raise inner_e 
    except Exception as e:
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
    # (Simplified: Full complex logic is omitted for brevity but should be included here)
    try:
        # Fallback to UnstructuredExcelLoader if structured load fails
        return UnstructuredExcelLoader(path).load()
    except Exception as e:
        logger.error(f"UnstructuredExcelLoader fallback failed for {path}: {e}")
        return []

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
            else:
                 doc = Document(page_content=str(item), metadata={})
            
            if not isinstance(doc.metadata, dict):
                doc.metadata = {"_raw_meta": str(doc.metadata)}
            if source_path:
                doc.metadata.setdefault("source_file", os.path.basename(source_path))
            
            # 🚨 จุดที่ต้องเพิ่มการกรอง Metadata ที่ซับซ้อน (ตรงนี้มีการเรียกใช้อยู่แล้ว)
            try:
                # 📌 ตรวจสอบให้แน่ใจว่า Metadata ถูกส่งเข้าฟังก์ชันกรอง
                doc.metadata = _safe_filter_complex_metadata(doc.metadata)
            except Exception:
                # Fallback: ถ้าการกรองล้มเหลว ให้ใช้ metadata พื้นฐานเท่านั้น
                doc.metadata = {"source_file": os.path.basename(source_path)} if source_path else {}
            normalized.append(doc)
        except Exception as e:
            logger.warning(f"normalize_loaded_documents: skipping item #{idx} due to error: {e}")
            continue
    return normalized

# -------------------- Load & Chunk Document --------------------
def load_and_chunk_document(
    file_path: str,
    doc_id_key: str, 
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

    try:
        # Note: Unstructured loaders (for images/structured files) are called here
        raw_docs = loader_func(file_path) if ext != ".pdf" else load_pdf(file_path, ocr_pages=ocr_pages)
    except Exception as e:
        logger.error(f"Loader {loader_func} raised for {file_path}: {e}")
        raw_docs = []

    if not raw_docs:
        logger.warning(f"No content loaded from {file_name}")
        return []

    docs = normalize_loaded_documents(raw_docs, source_path=file_path)

    for d in docs:
        if metadata:
            try:
                # 📌 การปรับปรุง: ต้องกรอง Metadata ที่ถูก Update ด้วย (เพื่อความปลอดภัย)
                new_meta = d.metadata
                new_meta.update(metadata) 
                d.metadata = _safe_filter_complex_metadata(new_meta)
            except Exception:
                # Fallback: ใช้ metadata เดิมและเพิ่ม injected_metadata แบบ String
                d.metadata["injected_metadata"] = str(metadata)
        
        # 📌 การปรับปรุง: ต้องกรอง Metadata ที่ถูก Update ทีหลังอีกครั้ง
        if year:
            d.metadata["year"] = year
        d.metadata["version"] = version
        d.metadata["doc_id"] = doc_id_key 
        d.metadata["source"] = d.metadata.get("source_file", file_name) 
        
        # กรองอีกครั้งหลังการ update
        d.metadata = _safe_filter_complex_metadata(d.metadata)


    is_structured_data = ext in [".xlsx", ".csv", ".jpg", ".jpeg", ".png"]

    if not is_structured_data:
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        try:
            chunks = splitter.split_documents(docs)
        except Exception as e:
            logger.warning(f"Text splitter failed on {file_name}: {e}. Falling back to using whole documents as chunks.")
            chunks = docs
    else:
        # สำหรับ Structured Data (เช่น รูปภาพ, Excel) ใช้ document ทั้งก้อนเป็น chunk
        chunks = docs

    for idx, c in enumerate(chunks, start=1):
        c.page_content = clean_text(c.page_content)
        c.metadata["chunk_index"] = idx
        
        # 📌 การปรับปรุง: กรอง Metadata ครั้งสุดท้ายก่อนส่งออก (เผื่อมี metadata ที่เพิ่มมาจากการ chunking)
        c.metadata = _safe_filter_complex_metadata(c.metadata) 
        
    logger.info(f"Loaded and chunked {file_name} -> {len(chunks)} chunks.")
    return chunks

# -------------------- Process single document --------------------
def process_document(
    file_path: str,
    file_name: str,
    stable_doc_uuid: str, 
    doc_type: Optional[str] = None,
    base_path: str = VECTORSTORE_DIR, 
    year: Optional[int] = None,
    version: str = "v1",
    metadata: dict = None,
    source_name_for_display: Optional[str] = None,
    ocr_pages: Optional[Iterable[int]] = None
) -> Tuple[List[Document], str, str]: 
    
    raw_doc_id_input = os.path.splitext(file_name)[0]
    filename_doc_id_key = _normalize_doc_id(raw_doc_id_input) 
    
    # 📌 ตรวจสอบความยาวของชื่อ Collection ก่อนส่งให้ ChromaDB
    if not filename_doc_id_key or len(filename_doc_id_key) < 3:
        # ตรงนี้ไม่เกี่ยวกับ collection name แต่เป็น doc_id_key 
        pass 
        
    doc_type = doc_type or "document"

    chunks = load_and_chunk_document(
        file_path=file_path,
        doc_id_key=filename_doc_id_key, 
        year=year,
        version=version,
        metadata=metadata,
        ocr_pages=ocr_pages
    )
    
    for c in chunks:
        c.metadata["stable_doc_uuid"] = stable_doc_uuid 
        c.metadata["doc_type"] = doc_type 
        # 📌 การปรับปรุง: กรอง Metadata ครั้งสุดท้ายก่อน return
        c.metadata = _safe_filter_complex_metadata(c.metadata)
        
    return chunks, stable_doc_uuid, doc_type

# -------------------- Vectorstore / Mapping Utilities --------------------

# 🟢 เพิ่ม Global Cache สำหรับ Vector Store
_VECTORSTORE_SERVICE_CACHE = {} 

def get_vectorstore(collection_name: str = "default", base_path: str = VECTORSTORE_DIR) -> Chroma:
    """
    Initializes and returns the Chroma Vectorstore instance for a given collection.
    """
    cache_key = f"{collection_name}_{base_path}"
    
    if cache_key in _VECTORSTORE_SERVICE_CACHE:
        return _VECTORSTORE_SERVICE_CACHE[cache_key]
        
    # 🔴 NEW: ตรวจสอบความยาวของชื่อ Collection ก่อนส่งให้ ChromaDB
    if len(collection_name) < 3:
        # บังคับให้เกิด Error ที่ชัดเจนตามข้อกำหนดของ Chroma
        raise ValueError(f"Collection name '{collection_name}' is too short. ChromaDB requires at least 3 characters.")


    # 1. โหลด Embedding Model 
    if "embeddings_model" not in _VECTORSTORE_SERVICE_CACHE:
        logger.info("Initializing SentenceTransformer Embeddings (intfloat/multilingual-e5-large)...")
        try:
            embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
            _VECTORSTORE_SERVICE_CACHE["embeddings_model"] = embeddings
        except Exception as e:
            logger.critical(f"❌ Failed to load embedding model: {e}")
            raise RuntimeError(f"Failed to initialize embeddings: {e}")
    else:
        embeddings = _VECTORSTORE_SERVICE_CACHE["embeddings_model"]

    # 2. โหลด/สร้าง Chroma
    persist_directory = os.path.join(base_path, collection_name)
    os.makedirs(persist_directory, exist_ok=True)
    
    try:
        vectorstore = Chroma(
            collection_name=collection_name, 
            persist_directory=persist_directory, 
            embedding_function=embeddings
        )
        _VECTORSTORE_SERVICE_CACHE[cache_key] = vectorstore
        logger.info(f"✅ Loaded Vector Store service for collection: {collection_name}")
        return vectorstore
    except Exception as e:
        logger.critical(f"❌ Failed to load Chroma for collection '{collection_name}': {e}")
        # 🔴 NEW: ถ้าเกิด Error จาก ChromaDB ที่เกี่ยวกับ Validation ให้ re-raise เป็น RuntimeError
        raise RuntimeError(f"Failed to initialize Chroma DB: {e}")


def create_vectorstore_from_documents(
    chunks: List[Document], 
    collection_name: str, 
    doc_mapping_db: Dict[str, Dict[str, Any]],
    base_path: str = VECTORSTORE_DIR
) -> Chroma:
    """
    Creates/updates a Chroma vector store collection and records the document ID mapping.
    """
    
    # 📌 ใช้ get_vectorstore แทนการโหลดเอง
    vectorstore = get_vectorstore(collection_name, base_path)

    if not chunks:
        logger.warning(f"No chunks provided for collection {collection_name}. Skipping indexing.")
        return vectorstore
        
    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]

    try:
        ids = [str(uuid.uuid4()) for _ in texts]
        
        # 🚨 ตรวจสอบการลบ/อัปเดตไฟล์เก่าที่มี stable_doc_uuid เดียวกัน (Logic re-ingest)
        stable_doc_uuids_to_delete = set()
        old_chunk_ids_to_delete = []

        for m in metadatas:
            s_uuid = m.get("stable_doc_uuid")
            if s_uuid and s_uuid in doc_mapping_db:
                # 🟢 การปรับปรุง: ตรวจสอบว่ามี chunk_uuids อยู่ก่อนถึงจะพิจารณาว่าต้องลบ
                existing_chunk_uuids = doc_mapping_db[s_uuid].get("chunk_uuids")
                if existing_chunk_uuids and isinstance(existing_chunk_uuids, list) and len(existing_chunk_uuids) > 0:
                    stable_doc_uuids_to_delete.add(s_uuid)
                    old_chunk_ids_to_delete.extend(existing_chunk_uuids)
                    # เคลียร์รายการ chunk เก่าใน mapping DB ล่วงหน้า
                    doc_mapping_db[s_uuid]["chunk_uuids"] = [] 

        
        if stable_doc_uuids_to_delete:
            logger.info(f"Detected {len(stable_doc_uuids_to_delete)} Stable IDs being re-indexed. Deleting {len(old_chunk_ids_to_delete)} old chunks...")
            
            if old_chunk_ids_to_delete:
                # 📌 หากมี ID เก่าที่ต้องลบจริงๆ
                vectorstore.delete(ids=old_chunk_ids_to_delete)
                logger.info(f"🧹 Deleted {len(old_chunk_ids_to_delete)} old chunks from Chroma.")
            else:
                 # 📌 กรณีที่ ID มีอยู่ใน mapping แต่ไม่มี chunk ID (เช่น เพิ่งถูกเพิ่ม)
                logger.info("ℹ️ No old chunks found in Chroma for recorded Stable IDs, proceeding to add new chunks.")

        # Add texts to the vector store using generated IDs
        vectorstore.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids 
        )

        for i, chunk_uuid in enumerate(ids):
            stable_doc_uuid = metadatas[i].get("stable_doc_uuid") 
            
            if stable_doc_uuid:
                if stable_doc_uuid not in doc_mapping_db:
                    doc_mapping_db[stable_doc_uuid] = {"chunk_uuids": []}
                
                doc_entry = doc_mapping_db[stable_doc_uuid]
                
                if "chunk_uuids" not in doc_entry or not isinstance(doc_entry["chunk_uuids"], list):
                     doc_entry["chunk_uuids"] = []
                
                if chunk_uuid not in doc_entry["chunk_uuids"]:
                    doc_entry["chunk_uuids"].append(chunk_uuid)
                
                metadatas[i]["chunk_uuid"] = chunk_uuid 
        
        # vectorstore.persist()
        logger.info(f"✅ Indexed {len(ids)} chunks and updated mapping for {collection_name}. Persist finished.")
        
    except Exception as e:
        logger.error(f"❌ Error during Chroma indexing or persisting for {collection_name}: {e}")
    
    return vectorstore

def save_doc_id_mapping(mapping_data: Dict[str, Dict[str, Any]], path: str):
    """Saves the document metadata and chunk UUID mapping to a JSON file."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, indent=4, ensure_ascii=False)
        logger.info(f"✅ Successfully saved Doc ID Mapping to: {path}")
    except Exception as e:
        logger.error(f"❌ Failed to save Doc ID Mapping: {e}")

def load_doc_id_mapping(path: str) -> Dict[str, Dict[str, Any]]:
    """Loads the document metadata and chunk UUID mapping from a JSON file."""
    if not os.path.exists(path):
        return {} 
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"❌ Failed to load Doc ID Mapping from {path}: {e}")
        return {}
        
# -------------------- API Helper: Get UUIDs for RAG Filtering --------------------

def get_stable_uuids_by_doc_type(doc_types: List[str]) -> List[str]:
    """
    Retrieves a list of all Stable UUIDs associated with the given document types 
    from the document mapping file. Used by the RAG API for filtering.
    """
    if not doc_types:
        return []
        
    doc_mapping_db = load_doc_id_mapping(MAPPING_FILE_PATH)
    
    target_uuids = []
    
    # Normalize types for robust comparison (though doc_type should be lowercase in the map)
    doc_type_set = {dt.lower() for dt in doc_types}
    
    for s_uuid, entry in doc_mapping_db.items():
        doc_type = entry.get("doc_type", "default")
        if doc_type.lower() in doc_type_set:
            target_uuids.append(s_uuid)
            
    return target_uuids
        
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
    
    files_to_process = []
    
    # 1. รวบรวมไฟล์ทั้งหมดและกำหนด doc_type
    for root, dirs, filenames in os.walk(data_dir):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        current_doc_type = doc_type or "default"
        if root != data_dir:
            folder_name = os.path.basename(root)
            if folder_name.lower() == 'statement':
                current_doc_type = 'statement'
            else:
                current_doc_type = folder_name.lower()

        if doc_type and doc_type != 'all' and current_doc_type != doc_type:
            continue

        for f in filenames:
            file_extension = os.path.splitext(f)[1].lower()
            if f.startswith('.') or file_extension not in SUPPORTED_TYPES:
                continue
            
            files_to_process.append({
                "file_path": os.path.join(root, f),
                "file_name": f,
                "doc_type": current_doc_type
            })

    # ---------------------------------------------------------
    # 0. Load Mapping DB และกำหนด Stable IDs/Entry
    # ---------------------------------------------------------
    doc_mapping_db = load_doc_id_mapping(MAPPING_FILE_PATH)
    doc_uuid_lookup: Dict[str, str] = {} # {filename_doc_id_key: stable_doc_uuid}
    
    for s_uuid, entry in doc_mapping_db.items():
        if "doc_id_key" in entry: 
            doc_uuid_lookup[entry["doc_id_key"]] = s_uuid

    for file_info in files_to_process:
        raw_doc_id_input = os.path.splitext(file_info["file_name"])[0]
        filename_doc_id_key = _normalize_doc_id(raw_doc_id_input)

        stable_doc_uuid = doc_uuid_lookup.get(filename_doc_id_key)
        
        if stable_doc_uuid:
            file_info["stable_doc_uuid"] = stable_doc_uuid
            doc_mapping_db[stable_doc_uuid]["file_name"] = file_info["file_name"]
            
        else:
            new_uuid = str(uuid.uuid4())
            file_info["stable_doc_uuid"] = new_uuid
            
            doc_mapping_db[new_uuid] = {
                "file_name": file_info["file_name"],
                "doc_type": file_info["doc_type"],
                "doc_id_key": filename_doc_id_key, 
                "notes": "",                     
                "statement_id": "",              
                "chunk_uuids": []
            }
            doc_uuid_lookup[filename_doc_id_key] = new_uuid 


    all_chunks: List[Document] = []
    results = []
    
    logger.info(f"Starting batch Load & Chunk of {len(files_to_process)} files...")

    # ---------------------------------------------------------
    # 1. Load & Chunk (Sequential or Multi-threading)
    # ---------------------------------------------------------
    
    def _process_file_task_new(file_info: Dict[str, str]):
        return process_document(
            file_path=file_info["file_path"],
            file_name=file_info["file_name"],
            stable_doc_uuid=file_info["stable_doc_uuid"], 
            doc_type=file_info["doc_type"],
            base_path=base_path,
            year=year,
            version=version
        )
    
    if sequential:
        for file_info in files_to_process:
            f = file_info["file_name"]
            stable_doc_uuid = file_info["stable_doc_uuid"]
            try:
                chunks, doc_id, dt = _process_file_task_new(file_info)
                all_chunks.extend(chunks)
                results.append({"file": f, "doc_id": doc_id, "doc_type": dt, "status": "chunked", "chunks": len(chunks)})
            except Exception as e:
                error_doc_id = stable_doc_uuid
                logger.error(f"Error processing/chunking {f} (ID: {error_doc_id}): {e}")
                results.append({"file": f, "doc_id": error_doc_id, "doc_type": file_info["doc_type"], "status": "failed_chunk", "error": str(e)})
    else:
        max_workers = min(8, (os.cpu_count() or 4))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(_process_file_task_new, f_info): f_info for f_info in files_to_process}
            for future in as_completed(future_to_file):
                f_info = future_to_file[future]
                f = f_info["file_name"]
                stable_doc_uuid = f_info["stable_doc_uuid"]
                try:
                    chunks, doc_id, dt = future.result()
                    all_chunks.extend(chunks)
                    results.append({"file": f, "doc_id": doc_id, "doc_type": dt, "status": "chunked", "chunks": len(chunks)})
                except Exception as e:
                    error_doc_id = stable_doc_uuid
                    logger.error(f"Error processing/chunking {f} (ID: {error_doc_id}): {e}")
                    results.append({"file": f, "doc_id": error_doc_id, "doc_type": f_info["doc_type"], "status": "failed_chunk", "error": str(e)})
                    
    # ---------------------------------------------------------
    # 2. Group Chunks by doc_type
    # ---------------------------------------------------------
    chunks_by_type: Dict[str, List[Document]] = {}
    for chunk in all_chunks:
        dt = chunk.metadata.get("doc_type", "default")
        if dt not in chunks_by_type:
            chunks_by_type[dt] = []
        chunks_by_type[dt].append(chunk)

    logger.info(f"Grouping complete. Found {len(chunks_by_type)} collection(s): {list(chunks_by_type.keys())}")
    
    # ---------------------------------------------------------
    # 3. Indexing Chunks and Saving Mapping
    # ---------------------------------------------------------
    
    for dt, dt_chunks in chunks_by_type.items():
        if dt_chunks:
            # ใช้ try/except เพื่อให้การ Index ของ Collection หนึ่งไม่หยุด Collection อื่น
            try:
                create_vectorstore_from_documents(
                    chunks=dt_chunks,
                    collection_name=dt,
                    doc_mapping_db=doc_mapping_db,
                    base_path=base_path
                )
            except RuntimeError as e:
                 logger.critical(f"❌ Failed to index collection '{dt}' due to service initialization error: {e}")


    save_doc_id_mapping(doc_mapping_db, MAPPING_FILE_PATH)

    logger.info("Batch ingestion process finished.")
    return results

def wipe_vectorstore(doc_type_to_wipe: str = 'all', base_path: str = VECTORSTORE_DIR):
    """Wipes the vector store directory/collection(s) and potentially the doc_id_mapping file."""
    
    if doc_type_to_wipe.lower() == 'all':
        # 1. ลบ Vector Store Directory ทั้งหมด
        if os.path.exists(base_path):
            try:
                shutil.rmtree(base_path)
                logger.info(f"✅ Deleted vector store directory: {base_path}")
            except OSError as e:
                logger.error(f"❌ Error deleting vector store: {e}")
                
        # 2. ลบ Doc ID Mapping File
        if os.path.exists(MAPPING_FILE_PATH):
            try:
                os.remove(MAPPING_FILE_PATH)
                logger.info(f"✅ Deleted Doc ID mapping file: {MAPPING_FILE_PATH}")
            except OSError as e:
                logger.error(f"❌ Error deleting mapping file: {e}")
                
    elif doc_type_to_wipe in SUPPORTED_DOC_TYPES: # ตรวจสอบให้แน่ใจว่าเป็นชื่อ Collection ที่ถูกต้อง
        # ลบเฉพาะ Collection
        col_path = os.path.join(base_path, doc_type_to_wipe)
        if os.path.exists(col_path):
            try:
                shutil.rmtree(col_path)
                logger.info(f"✅ Deleted vector store collection: {col_path}")
            except OSError as e:
                logger.error(f"❌ Error deleting collection {doc_type_to_wipe}: {e}")
        
        # 📌 ลบ entry ใน Mapping file ที่ตรงกับ doc_type นั้นด้วย
        doc_mapping_db = load_doc_id_mapping(MAPPING_FILE_PATH)
        uuids_to_keep = {}
        deletion_count = 0
        for s_uuid, entry in doc_mapping_db.items():
            if entry.get("doc_type") != doc_type_to_wipe:
                uuids_to_keep[s_uuid] = entry
            else:
                deletion_count += 1
        
        if deletion_count > 0:
            save_doc_id_mapping(uuids_to_keep, MAPPING_FILE_PATH)
            logger.info(f"🧹 Removed {deletion_count} entries from mapping file for doc_type: {doc_type_to_wipe}")


# -------------------- Document Management Utilities --------------------

def delete_document_by_uuid(
    stable_doc_uuid: str, 
    collection_name: Optional[str] = None, 
    base_path: str = VECTORSTORE_DIR
) -> bool:
    """
    Deletes all chunks associated with the given Stable Document UUID 
    from the vector store(s) and removes the entry from the mapping file.
    """
    
    doc_mapping_db = load_doc_id_mapping(MAPPING_FILE_PATH)
    
    if stable_doc_uuid not in doc_mapping_db:
        logger.warning(f"Deletion skipped: Stable Doc UUID {stable_doc_uuid} not found in mapping.")
        return False

    doc_entry = doc_mapping_db[stable_doc_uuid]
    all_chunk_uuids = doc_entry.get("chunk_uuids", [])
    doc_type_from_map = doc_entry.get("doc_type", "default")
    
    if not all_chunk_uuids:
        logger.warning(f"Deletion skipped: Stable Doc UUID {stable_doc_uuid} has no chunk UUIDs recorded.")
        del doc_mapping_db[stable_doc_uuid]
        save_doc_id_mapping(doc_mapping_db, MAPPING_FILE_PATH)
        return True

    collections_to_check = [doc_type_from_map]
    
    success = False
    
    for col_name in collections_to_check:
        persist_directory = os.path.join(base_path, col_name)
        if not os.path.isdir(persist_directory):
            logger.warning(f"Vectorstore directory not found for collection '{col_name}'. Skipping Chroma deletion.")
            continue
        
        try:
            # 📌 ใช้ get_vectorstore
            vectorstore = get_vectorstore(col_name, base_path)
            
            vectorstore.delete(ids=all_chunk_uuids)
            logger.info(f"✅ Successfully deleted {len(all_chunk_uuids)} chunks from collection '{col_name}' for UUID: {stable_doc_uuid}")
            success = True
            
        except Exception as e:
            logger.error(f"❌ Error during Chroma deletion for collection '{col_name}' (UUID: {stable_doc_uuid}): {e}")
            
    if stable_doc_uuid in doc_mapping_db:
        del doc_mapping_db[stable_doc_uuid]
        save_doc_id_mapping(doc_mapping_db, MAPPING_FILE_PATH)
        
    return success

# -------------------- List Documents Utility (FIXED) --------------------
def list_documents(doc_types: Optional[List[str]] = None) -> Dict[str, DocInfo]:
    """
    Scans the data directory and checks the ingestion status against the mapping file.
    The function now filters by doc_types and returns a dictionary keyed by doc_id,
    as required by the API endpoint.
    """
    
    doc_mapping_db = load_doc_id_mapping(MAPPING_FILE_PATH)
    all_supported_files = []
    
    # Map normalized filename keys back to their stable UUIDs
    doc_id_key_to_stable_uuid: Dict[str, str] = {}
    for s_uuid, entry in doc_mapping_db.items():
        if "doc_id_key" in entry:
            doc_id_key_to_stable_uuid[entry["doc_id_key"]] = s_uuid
    
    for root, _, filenames in os.walk(DATA_DIR):
        for f in filenames:
            file_extension = os.path.splitext(f)[1].lower()
            if f.startswith('.') or file_extension not in SUPPORTED_TYPES:
                continue
                
            file_path = os.path.join(root, f)
            
            doc_type = "default"
            # Determine doc_type based on subdirectory name
            if root != DATA_DIR:
                folder_name = os.path.basename(root).lower()
                if folder_name == 'statement':
                    doc_type = 'statement'
                else:
                    doc_type = folder_name
            
            # --- FIXED: Implement doc_types filtering ---
            if doc_types and doc_type not in doc_types:
                continue # Skip files that don't match the requested type
                
            file_name_no_ext = os.path.splitext(f)[0]
            filename_doc_id_key = _normalize_doc_id(file_name_no_ext)
            
            stable_doc_uuid = doc_id_key_to_stable_uuid.get(filename_doc_id_key)
            doc_entry = doc_mapping_db.get(stable_doc_uuid) if stable_doc_uuid else None
            
            is_ingested = doc_entry and doc_entry.get("chunk_uuids")
            uuid_list = doc_entry.get("chunk_uuids", []) if doc_entry else []
            chunk_count = len(uuid_list)
            
            # --- FIXED: Use stable_doc_uuid for doc_id, or a fallback that includes filename ---
            # If not ingested, create a unique doc_id based on doc_type and filename for temporary dict key
            final_doc_id = stable_doc_uuid or f"N/A ({doc_type}-{filename_doc_id_key})"
            
            file_info = {
                "doc_id": final_doc_id, # The final key used for the API response dict
                "file_name": f, 
                "file_path": file_path,
                "doc_type": doc_type,
                "size_mb": os.path.getsize(file_path) / (1024 * 1024),
                "status": "Ingested" if is_ingested else "Not Ingested", # Use status instead of 'ingested' bool
                "chunk_count": chunk_count,
                "ref_doc_id": filename_doc_id_key 
            }
            all_supported_files.append(file_info)

    all_supported_files.sort(key=lambda x: (x["doc_type"], x["file_name"]))
    
    print(f"\nFound {len(all_supported_files)} supported documents for types {doc_types}:")
    
    # Printing for console log/debug
    print("-" * 140)
    print(f"{'DOC ID (Stable/Temp)':<38} | {'FILENAME':<35} | {'TYPE':<10} | {'SIZE(MB)':<9} | {'STATUS':<10} | {'CHUNKS':<8} | {'REF ID (Old Key)'}")
    print("-" * 140)
    
    documents_by_id: Dict[str, DocInfo] = {}
    for info in all_supported_files:
        short_doc_id = info['doc_id'][:36] 
        short_filename = info['file_name'][:33] if len(info['file_name']) > 33 else info['file_name']
        size_str = f"{info['size_mb']:.2f}"
        short_ref_doc_id = info['ref_doc_id'][:16] if len(info['ref_doc_id']) > 16 else info['ref_doc_id']
        
        print(f"{short_doc_id:<38} | {short_filename:<35} | {info['doc_type']:<10} | {size_str:<9} | {info['status']:<10} | {info['chunk_count']:<8} | {short_ref_doc_id}")

        # --- FIXED: Convert list item to dictionary entry keyed by doc_id ---
        documents_by_id[info['doc_id']] = info

    return documents_by_id
