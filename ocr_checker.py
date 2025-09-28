# ocr_checker.py
import os
import json
import logging
from core.ingest import load_pdf, DATA_DIR

logging.basicConfig(
    filename="ocr_checker.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def check_ocr_usage(data_dir=DATA_DIR):
    """
    Check OCR usage for all PDFs in DATA_DIR.
    Returns a list of dicts with:
    - filename
    - total_chars
    - chars_from_ocr
    - ocr_percentage
    """
    results = []

    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("No PDF files found in", data_dir)
        return results

    for pdf_file in pdf_files:
        path = os.path.join(data_dir, pdf_file)
        try:
            # Load PDF first normally (PyPDFLoader)
            docs_normal = load_pdf(path)  # load_pdf in ingest.py already tries OCR fallback
            # Count total characters
            total_chars = sum(len(d.page_content) for d in docs_normal)

            # Check how much text came specifically from OCR fallback
            # Here, assume if PyPDFLoader returns <50 chars, load_pdf uses OCR
            ocr_chars = 0
            loader = None
            from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
            py_loader = PyPDFLoader(path)
            docs_py = py_loader.load()
            text_py = " ".join([d.page_content for d in docs_py]).strip()
            if len(text_py) < 50:
                # OCR was triggered inside load_pdf
                ocr_chars = total_chars
            else:
                ocr_chars = 0  # No OCR needed

            ocr_percentage = (ocr_chars / total_chars * 100) if total_chars else 0

            results.append({
                "filename": pdf_file,
                "total_chars": total_chars,
                "chars_from_ocr": ocr_chars,
                "ocr_percentage": round(ocr_percentage, 2)
            })
        except Exception as e:
            logging.error(f"Error checking OCR for {pdf_file}: {e}")
            results.append({
                "filename": pdf_file,
                "total_chars": 0,
                "chars_from_ocr": 0,
                "ocr_percentage": 0,
                "error": str(e)
            })

    return results

if __name__ == "__main__":
    report = check_ocr_usage()
    print(json.dumps(report, indent=2, ensure_ascii=False))
