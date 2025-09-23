# file_loaders.py
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    TextLoader,
)
import os

# ---- loader functions ----
def load_txt(path):
    return TextLoader(path, encoding="utf-8", errors="ignore").load()

def load_pdf(path):
    loader = PyPDFLoader(path)
    return loader.load()

def load_docx(path):
    loader = UnstructuredWordDocumentLoader(path)
    return loader.load()

def load_xlsx(path):
    loader = UnstructuredExcelLoader(path)
    return loader.load()

# ---- file extension map ----
FILE_LOADER_MAP = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".txt": load_txt,
    ".xlsx": load_xlsx,
}
