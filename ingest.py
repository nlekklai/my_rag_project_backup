import os
from datetime import datetime
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

# ---------------------
# Document loaders
# ---------------------
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
        text_content = " ".join([doc.page_content for doc in docs]).strip()
        
        # ถ้าอ่าน text layer ไม่ออก → ใช้ OCR
        if not text_content:
            loader = UnstructuredPDFLoader(path, mode="elements", ocr=True)
            docs = loader.load()

        for i, doc in enumerate(docs, start=1):
            doc.metadata["source"] = os.path.basename(path)
            doc.metadata["page"] = i
        return docs
    except Exception as e:
        print(f"❌ โหลดไฟล์ {path} ไม่สำเร็จ: {e}")
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

# ---------------------
# Main functions
# ---------------------
def process_document(file_path, file_name=None):
    if not file_name:
        file_name = os.path.basename(file_path)
    ext = os.path.splitext(file_name)[1].lower()
    if ext not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported file: {file_name}")

    loader_func = FILE_LOADER_MAP[ext]
    docs = loader_func(file_path)
    if not docs:
        raise ValueError(f"Failed to load document: {file_name}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len
    )
    chunks = splitter.split_documents(docs)

    doc_id = os.path.splitext(file_name)[0]

    # Save vectorstore
    chunk_texts = [c.page_content for c in chunks]
    save_to_vectorstore(doc_id, chunk_texts)

    print(f"✅ Document '{file_name}' processed as doc_id '{doc_id}', chunks: {len(chunks)}")
    return doc_id

def list_documents():
    os.makedirs(DATA_DIR, exist_ok=True)
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
