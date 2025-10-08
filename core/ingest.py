# core/ingest.py
import os
import re
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Set, Iterable, Dict, Any
import pandas as pd # <-- Import pandas

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

def load_xlsx_generic_structured(path: str) -> List[Document]:
    """
    Loads XLSX files by attempting structured parsing using Pandas first.
    It identifies essential columns (Score/Weight, Detail) across all sheets.
    If structured parsing fails for all sheets, it falls back to UnstructuredExcelLoader.
    """
    try:
        # Step 1: Attempt to load all sheets of the Excel file
        xls = pd.ExcelFile(path)
    except Exception as e:
        logger.error(f"Failed to open Excel file {path}: {e}")
        # Fallback if opening fails completely
        return UnstructuredExcelLoader(path).load()

    all_docs: List[Document] = []
    
    # Define keywords for structural columns (Thai and English)
    score_keywords = ["น้ำหนัก", "คะแนน", "score", "weight"]
    topic_keywords = ["หัวข้อ", "topic", "criteria_id"]
    category_keywords = ["หมวด", "category", "area"]
    detail_keywords = ["รายละเอียด", "เกณฑ์", "detail", "content", "criterion"]
    
    # Load all sheets into a dictionary of DataFrames
    sheet_names = xls.sheet_names
    
    for sheet_name in sheet_names:
        sheet_docs: List[Document] = []
        try:
            # Read the entire sheet, ignoring headers initially
            df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
            df.columns = [str(i) for i in range(len(df.columns))] # Ensure columns are strings
            
            # Identify the header row based on keyword presence
            header_row_index = -1
            best_match_count = 0

            # Scan the first 10 rows to find a good header row
            for i in range(min(10, len(df))):
                row_str = " ".join(df.iloc[i].astype(str).str.lower().fillna('')).strip()
                
                # Count matches for essential columns (score, detail)
                score_match = any(k in row_str for k in score_keywords)
                detail_match = any(k in row_str for k in detail_keywords)
                
                # If essential columns are present, this is a potential header
                current_match_count = score_match + detail_match + \
                                      any(k in row_str for k in topic_keywords) + \
                                      any(k in row_str for k in category_keywords)

                if current_match_count >= 2 and current_match_count > best_match_count:
                    header_row_index = i
                    best_match_count = current_match_count

            # --- Structured Processing ---
            if header_row_index != -1:
                logger.info(f"Structured header found in sheet '{sheet_name}' at row {header_row_index} in {os.path.basename(path)}")
                
                # Re-read the DataFrame with the detected header row
                df_structured = pd.read_excel(xls, sheet_name=sheet_name, header=header_row_index)
                
                # Clean column names (strip spaces, lower case)
                df_structured.columns = df_structured.columns.astype(str).str.strip().str.lower()
                
                # Map column names based on keywords (prioritize Thai/specific terms)
                col_mapping = {}
                
                def find_column(keywords):
                    for k in keywords:
                        # Check for exact match or contains (for cases like 'KM Score (คะแนน)')
                        matching_cols = [c for c in df_structured.columns if k.lower() in c]
                        if matching_cols:
                            return matching_cols[0]
                    return None

                score_col = find_column(score_keywords)
                topic_col = find_column(topic_keywords)
                category_col = find_column(category_keywords)
                detail_col = find_column(detail_keywords)

                # Ensure essential columns exist
                if not detail_col:
                    logger.warning(f"Skipping structured load for sheet '{sheet_name}' - Missing Detail column.")
                    raise ValueError("Missing essential Detail column.")
                    
                # Ensure Score/Weight exists if we want structured data
                if not score_col:
                    logger.warning(f"Skipping structured load for sheet '{sheet_name}' - Missing Score/Weight column. Proceeding without score metadata.")
                    pass 
                
                # Start processing rows
                sheet_content_seen = set()
                for idx, row in df_structured.iterrows():
                    # Content to be embedded (the detail/description)
                    page_content_raw = str(row[detail_col]).strip()
                    
                    # 💡 FIX: Check content length and deduplicate within sheet
                    if len(page_content_raw) < 10 or page_content_raw in sheet_content_seen: continue
                    sheet_content_seen.add(page_content_raw)

                    metadata = {
                        "source_file": os.path.basename(path),
                        "sheet_name": sheet_name,
                        "row": idx + header_row_index + 2, # Estimate original row number
                        "score_or_weight": None,
                        "category": None,
                        "topic": None
                    }

                    # Populate structured metadata
                    if score_col:
                        try:
                            # Try converting to float/int, fall back to string
                            val = row[score_col]
                            if pd.isna(val):
                                metadata["score_or_weight"] = None
                            else:
                                metadata["score_or_weight"] = float(val) if isinstance(val, (int, float)) else str(val).strip()
                        except (ValueError, TypeError, KeyError):
                            metadata["score_or_weight"] = str(row.get(score_col, "")).strip()
                    
                    if topic_col:
                        metadata["topic"] = str(row[topic_col]).strip()
                        
                    if category_col:
                        metadata["category"] = str(row[category_col]).strip()

                    
                    # === START OF CRITICAL FIX: Pre-pend context into page_content ===
                    context_prefix = []
                    
                    if metadata["category"]:
                        context_prefix.append(f"หมวด: {metadata['category']}")
                        
                    if metadata["topic"]:
                        context_prefix.append(f"หัวข้อ: {metadata['topic']}")
                        
                    if metadata["score_or_weight"]:
                        # ใช้ score/weight เป็นตัวบอกระดับเกณฑ์ (เช่น ระดับ 5, ระดับ 4)
                        context_prefix.append(f"เกณฑ์ระดับ/คะแนน: {metadata['score_or_weight']}")
                        
                    # สร้าง Content ใหม่: [บริบท 1, บริบท 2] + เนื้อหา
                    if context_prefix:
                        # ใส่ | เป็นตัวแยกที่ชัดเจน
                        full_prefix = " | ".join(context_prefix) 
                        page_content_final = f"{full_prefix} | คำอธิบายเกณฑ์: {page_content_raw}"
                    else:
                        page_content_final = page_content_raw # Fallback
                    # === END OF CRITICAL FIX ===


                    sheet_docs.append(Document(page_content=page_content_final, metadata=metadata))
                
                logger.info(f"Successfully processed {len(sheet_docs)} valid documents from sheet '{sheet_name}'.")
                all_docs.extend(sheet_docs)

            # --- Fallback to Unstructured Loader ---
            else:
                logger.warning(f"No clear structured header found in sheet '{sheet_name}'. Skipping structured load for this sheet.")
                # We skip structured load for this sheet and rely on the final check below.
                pass 
                
        except Exception as e:
            # This handles errors in specific sheets during structured parsing
            logger.error(f"Error processing sheet '{sheet_name}' of {os.path.basename(path)}: {e}. Skipping structured load for this sheet.")
            
    # Final check: If structured loading failed for ALL sheets, use the generic loader.
    if not all_docs:
        logger.warning(f"No documents were generated through structured loading for {os.path.basename(path)}. Falling back to UnstructuredExcelLoader.")
        # Only call the generic loader if the structured approach failed entirely
        return UnstructuredExcelLoader(path).load()
    
    # Final filtering and deduplication across all sheets before return
    final_seen = set()
    final_docs = []
    for d in all_docs:
        key = d.page_content.strip()
        if key and key not in final_seen and len(key) >= 10:
             final_seen.add(key)
             final_docs.append(d)
             
    if not final_docs:
        logger.warning(f"Structured load resulted in zero documents after filtering. Falling back to UnstructuredExcelLoader.")
        return UnstructuredExcelLoader(path).load()
        
    return final_docs

def load_xlsx(path: str) -> List[Document]:
    # Use the generic structured loader for all .xlsx files
    return load_xlsx_generic_structured(path)

# def load_xlsx(path: str) -> List[Document]:
#     try: return UnstructuredExcelLoader(path).load()
#     except Exception as e: logger.error(f"Failed to load .xlsx {path}: {e}"); return []

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


    # --- CRITICAL FIX START: Skip splitting for structured data ---
    is_structured_data = ext == ".xlsx" # ตรวจสอบว่าเป็นไฟล์ Excel ที่ถูกจัดการโครงสร้างแล้ว
    
    if not is_structured_data:
        # ใช้ RecursiveCharacterTextSplitter สำหรับไฟล์ทั่วไป (PDF, DOCX, TXT)
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        chunks = splitter.split_documents(docs)
    else:
        # หากเป็น Structured Data (Excel) แต่ละ Document คือ 1 Chunk ที่สมบูรณ์แล้ว
        chunks = docs
    # --- CRITICAL FIX END ---


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
def list_documents(doc_types: Optional[List[str]] = None) -> List[dict]:
    files = []
    for f in os.listdir(DATA_DIR):
        if f.startswith('.'):  # skip hidden/system files
            continue
        path = os.path.join(DATA_DIR, f)
        if not os.path.isfile(path):
            continue
        stat = os.stat(path)
        doc_id = os.path.splitext(f)[0]

        # Check doc_type
        if doc_types:
            valid = False
            for dt in doc_types:
                vectordir = os.path.join(VECTORSTORE_DIR, dt, doc_id)
                if os.path.exists(vectordir):
                    valid = True
                    break
            if not valid:
                continue

        files.append({
            "doc_id": doc_id,
            "filename": f,
            "file_type": os.path.splitext(f)[1].lower(),
            "upload_date": datetime.utcfromtimestamp(stat.st_mtime).isoformat(),
            "status": "Ingested" if os.path.exists(os.path.join(VECTORSTORE_DIR, doc_id)) else "Pending"
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
