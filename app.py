import logging
import os
from langchain.schema import Document
from datetime import datetime, timezone
from typing import List, Dict, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel
from contextlib import asynccontextmanager

# --- Core Imports ---
from core.rag_prompts import QA_PROMPT, COMPARE_PROMPT
from core.ingest import process_document, list_documents, delete_document, DATA_DIR, SUPPORTED_TYPES
from core.vectorstore import load_vectorstore, vectorstore_exists, load_all_vectorstores, get_vectorstore_path
from langchain.chains import RetrievalQA

# ❗❗ FIXED: Import LLM Instance Explicitly to resolve AttributeError
from models.llm import llm as llm_instance 


# -----------------------------
# --- Logging Setup ---
# -----------------------------
logger = logging.getLogger("ingest")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -----------------------------
# --- Global Constants ---
# -----------------------------
VECTORSTORE_DIR = "vectorstore"

# -----------------------------
# --- Lifespan (แทน on_event) ---
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager แทน @app.on_event สำหรับ startup/shutdown"""
    # --- Startup ---
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    logging.info(f"✅ Data directory '{DATA_DIR}' and vectorstore '{VECTORSTORE_DIR}' ensured.")

    yield  # <-- Application runs here

    # --- Shutdown ---
    logging.info("🛑 Application shutdown complete.")

# -----------------------------
# --- FastAPI Initialization ---
# -----------------------------
app = FastAPI(
    title="Assessment RAG API",
    description="API for RAG-based document assessment and analysis.",
    lifespan=lifespan
)

# -----------------------------
# --- CORS ---
# -----------------------------
origins = ["http://localhost:5173", "http://127.0.0.1:5173", "*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# --- Pydantic Models ---
# -----------------------------
class UploadResponse(BaseModel):
    status: str
    doc_id: str
    filename: str
    file_type: str
    upload_date: str


# -----------------------------
# --- Document Endpoints ---
# -----------------------------
@app.get("/api/documents", response_model=List[UploadResponse])
async def get_documents():
    return list_documents(doc_types=['document', 'faq'])

@app.delete("/api/documents/{doc_id}")
async def remove_document(doc_id: str):
    try:
        delete_document(doc_id)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# -----------------------------
# --- Upload Endpoints ---
# -----------------------------
@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile):
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    doc_id = process_document(file_path, file.filename)
    return UploadResponse(
        status="processed",
        doc_id=doc_id,
        filename=file.filename,
        file_type=os.path.splitext(file.filename)[1],
        upload_date=datetime.now(timezone.utc).isoformat()
    )

@app.post("/upload/{doc_type}", response_model=UploadResponse)
async def upload_file_type(doc_type: str, file: UploadFile = File(...)):
    folder = os.path.join(DATA_DIR, doc_type)
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        doc_id = process_document(file_path=file_path, file_name=file.filename, doc_type=doc_type)
    except Exception as e:
        logger.error(f"Failed to process {file.filename} as {doc_type}: {e}")
        return UploadResponse(
            status="failed",
            doc_id=os.path.splitext(file.filename)[0],
            filename=file.filename,
            file_type=os.path.splitext(file.filename)[1],
            upload_date=datetime.now(timezone.utc).isoformat()
        )

    vector_path = os.path.join(VECTORSTORE_DIR, doc_type)
    status = "Ingested" if vectorstore_exists(doc_id, base_path=vector_path) else "Pending"

    return UploadResponse(
        status=status,
        doc_id=doc_id,
        filename=file.filename,
        file_type=os.path.splitext(file.filename)[1],
        upload_date=datetime.now(timezone.utc).isoformat()
    )

@app.get("/api/uploads/{doc_type}", response_model=List[UploadResponse])
async def list_uploads_by_type(doc_type: str):
    folder = os.path.join(DATA_DIR, doc_type)
    os.makedirs(folder, exist_ok=True)
    uploads = []
    vector_path = get_vectorstore_path(doc_type)
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if not os.path.isfile(file_path):
            continue
        doc_id = os.path.splitext(filename)[0]
        status = "Ingested" if vectorstore_exists(doc_id, base_path=vector_path) else "Pending"
        uploads.append(UploadResponse(
            status=status,
            doc_id=doc_id,
            filename=filename,
            file_type=os.path.splitext(filename)[1],
            upload_date=datetime.fromtimestamp(os.path.getmtime(file_path), tz=timezone.utc).isoformat()
        ))
    return uploads

@app.delete("/upload/{doc_type}/{file_id}")
async def delete_upload(doc_type: str, file_id: str):
    folder = os.path.join(DATA_DIR, doc_type)
    filepath = None
    for f in os.listdir(folder):
        if os.path.splitext(f)[0] == file_id:
            filepath = os.path.join(folder, f)
            break
    if not filepath or not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Delete file
    os.remove(filepath)

    # Delete vectorstore folder
    try:
        delete_document(doc_id=file_id, doc_type=doc_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete vectorstore: {e}")

    return {"status": "deleted", "file_id": file_id}

@app.get("/upload/{doc_type}/{file_id}")
async def download_upload(doc_type: str, file_id: str):
    folder = os.path.join(DATA_DIR, doc_type)
    filepath = None
    for f in os.listdir(folder):
        if os.path.splitext(f)[0] == file_id:
            filepath = os.path.join(folder, f)
            break
    if not filepath or not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(filepath, filename=os.path.basename(filepath))


# -----------------------------
# --- Ingest Endpoint ---
# -----------------------------
class IngestRequest(BaseModel):
    doc_ids: List[str]
    doc_type: Optional[str] = "document"

@app.post("/ingest")
async def ingest_documents(request: IngestRequest):
    results = []
    for doc_id in request.doc_ids:
        folder = os.path.join(DATA_DIR, request.doc_type)
        matched_files = [f for f in os.listdir(folder) if os.path.splitext(f)[0] == doc_id]

        if not matched_files:
            results.append({"doc_id": doc_id, "result": "failed", "error": "File not found"})
            continue

        file_name = matched_files[0]
        file_path = os.path.join(folder, file_name)

        try:
            process_document(file_path=file_path, file_name=file_name, doc_type=request.doc_type)
        except Exception as e:
            logger.warning(f"Warning while processing {doc_id}: {e}")

        if vectorstore_exists(doc_id, doc_type=request.doc_type):
            results.append({"doc_id": doc_id, "result": "success"})
        else:
            results.append({"doc_id": doc_id, "result": "failed", "error": "Vectorstore not found after processing"})

    return {"status": "completed", "results": results}


# -----------------------------
# --- Query Endpoint (Fixed) ---
# -----------------------------
@app.post("/query")
async def query_endpoint(
    question: str = Form(...),
    doc_ids: Optional[str] = Form(None),
    doc_types: Optional[str] = Form(None)
):
    doc_id_list = doc_ids.split(",") if doc_ids else None
    doc_type_list = doc_types.split(",") if doc_types else None
    skipped = []

    try:
        multi_retriever = load_all_vectorstores(
            doc_ids=doc_id_list,
            doc_type=doc_type_list,
            top_k=15,
            final_k=5
        )
    except ValueError as e:
        return {"error": str(e), "skipped": skipped}

    loaded_doc_ids = [r.doc_id for r in multi_retriever.retrievers_list]

    def get_all_docs_text(query_text):
        docs = multi_retriever._get_relevant_documents(query_text)
        return "\n\n".join([d.page_content for d in docs if hasattr(d, "page_content")])

    context_text = await run_in_threadpool(lambda: get_all_docs_text(question))

    prompt_text = QA_PROMPT.format(context=context_text, question=question)

    def call_llm_safe(prompt_text):
        # 🟢 FIXED: ใช้ llm_instance แทน llm
        res = llm_instance.invoke(prompt_text)
        if isinstance(res, dict) and "result" in res:
            return res["result"]
        elif isinstance(res, str):
            return res
        return str(res)

    answer = await run_in_threadpool(lambda: call_llm_safe(prompt_text))

    return {
        "question": question,
        "doc_ids": loaded_doc_ids,
        "doc_types": doc_type_list if doc_type_list else ["document", "faq"],
        "answer": answer,
        "skipped": skipped
    }


# -----------------------------
# --- Compare Endpoint (Fixed) ---
# -----------------------------
@app.post("/compare")
async def compare(
    doc1: str = Form(...),
    doc2: str = Form(...),
    query: str = Form(...),
    doc_types: Optional[str] = Form(None)
):
    # NOTE: ลบ Import ซ้ำซ้อนของ vectorstore_exists และ load_all_vectorstores ออก
    from core.vectorstore import vectorstore_exists # นำกลับมาเผื่อโค้ดส่วนล่างจำเป็นต้องใช้ local import
    
    doc_type_list = [dt.strip() for dt in doc_types.split(",")] if doc_types else ["document", "faq"]

    doc_ids = [doc1, doc2]
    valid_docs, skipped = [], []

    for dt in doc_type_list:
        base_path = os.path.join(VECTORSTORE_DIR, dt)
        for doc_id in doc_ids:
            if vectorstore_exists(doc_id, base_path=base_path):
                valid_docs.append((doc_id, dt))
            else:
                skipped.append(f"{dt}/{doc_id}")

    if not valid_docs:
        raise HTTPException(
            status_code=404,
            detail=f"No valid vectorstores found for docs: {', '.join(doc_ids)} in doc_types: {doc_type_list}"
        )

    multi_retriever = load_all_vectorstores(doc_ids=doc_ids, top_k=5, doc_type=doc_type_list)

    def get_docs_text(query_text):
        docs = multi_retriever._get_relevant_documents(query_text)
        doc_text_map = {doc1: "", doc2: ""}
        
        # ปรับปรุง Logic การดึง source เพื่อให้ตรงกับ doc_id ที่รับมาจาก Form
        for d in docs:
            # สมมติว่า d.metadata.get("doc_id") หรือ d.metadata.get("source") ใช้ได้
            doc_key = d.metadata.get("doc_id") or os.path.splitext(os.path.basename(d.metadata.get("source", "")))[0]

            if doc_key == doc1:
                doc_text_map[doc1] += (d.page_content + "\n")
            elif doc_key == doc2:
                doc_text_map[doc2] += (d.page_content + "\n")
        
        doc1_text = doc_text_map.get(doc1, "[No content found for Doc 1]")
        doc2_text = doc_text_map.get(doc2, "[No content found for Doc 2]")
        
        return doc1_text, doc2_text

    doc1_text, doc2_text = await run_in_threadpool(lambda: get_docs_text(query))
    context_text = f"Document 1 Content:\n{doc1_text}\n\nDocument 2 Content:\n{doc2_text}"


    prompt_text = COMPARE_PROMPT.format(context=context_text, query=query, doc_names=f"{doc1} และ {doc2}")
    # 🟢 FIXED: ใช้ llm_instance แทน llm
    delta_answer = await run_in_threadpool(lambda: llm_instance.invoke(prompt_text))

    return {
        "result": {
            "metrics": [
                {
                    "metric": "Key comparison",
                    "doc1": doc1_text,
                    "doc2": doc2_text,
                    "delta": delta_answer,
                    "remark": f"Comparison generated from doc_types: {', '.join(doc_type_list)}"
                }
            ]
        },
        "skipped": skipped
    }


# -----------------------------
# --- Health & Status ---
# -----------------------------
@app.get("/api/status")
async def api_status():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}

@app.get("/api/health")
async def health_check():
    data_ready = os.path.exists(DATA_DIR)
    vector_ready = os.path.exists(VECTORSTORE_DIR)
    return {
        "status": "ok" if data_ready and vector_ready else "error",
        "data_dir_exists": data_ready,
        "vectorstore_exists": vector_ready,
        "time": datetime.now(timezone.utc).isoformat()
    }