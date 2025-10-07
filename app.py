# app.py
import os
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel
from core.rag_chain import (
    load_multi_retriever,
    MultiDocRetriever,
    create_rag_chain,
    llm,
    COMPARE_PROMPT,
    QA_PROMPT,
    run_assessment_workflow,
    get_workflow_status,
    get_workflow_results,
    # *** เพิ่ม answer_question_rag ที่ย้ายมา ***
    answer_question_rag
)

# -----------------------------
# --- Core Imports ---
# -----------------------------
from core.ingest import (
    process_document,
    list_documents,
    delete_document,
    DATA_DIR,
    SUPPORTED_TYPES
)
from core.vectorstore import load_vectorstore, vectorstore_exists, load_all_vectorstores

from langchain.chains import RetrievalQA
import logging

logger = logging.getLogger("ingest")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)
# -----------------------------
# --- FastAPI Initialization ---
# -----------------------------
app = FastAPI(title="Assessment RAG API", description="API for RAG-based document assessment and analysis.")

# CORS
origins = ["http://localhost:5173", "http://127.0.0.1:5173", "*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

VECTORSTORE_DIR = "vectorstore"

# -----------------------------
# --- Pydantic Models ---
# -----------------------------
class UploadResponse(BaseModel):
    status: str
    doc_id: str
    filename: str
    file_type: str
    upload_date: str 

class CompareResultMetrics(BaseModel):
    metric: str
    doc1: str 
    doc2: str
    delta: str
    remark: str

class CompareResponse(BaseModel):
    result: Dict[str, List[CompareResultMetrics]] 

class QueryResponse(BaseModel):
    answer: str
    conversation_id: Optional[str] = None

class ProcessStep(BaseModel):
    id: int
    name: str
    status: str
    progress: Optional[int] = None

class ProcessStatus(BaseModel):
    isRunning: bool
    currentStep: int
    steps: List[ProcessStep]

class ResultSummary(BaseModel):
    totalQuestions: int
    averageScore: float
    evidenceCount: int
    gapCount: int



# -----------------------------
# --- Startup Event ---
# -----------------------------
@app.on_event("startup")
async def startup_event():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    logging.info(f"Data directory '{DATA_DIR}' and vectorstore '{VECTORSTORE_DIR}' ensured.")

# -----------------------------
# --- Helper Functions ---
# -----------------------------
def format_docs(docs):
    """Format documents into a string for prompts."""
    formatted = []
    for doc in docs:
        formatted.append(doc.page_content if hasattr(doc, "page_content") else str(doc))
    return "\n\n".join(formatted)

def get_vectorstore_path(doc_type: Optional[str] = None):
    if doc_type:
        path = os.path.join(VECTORSTORE_DIR, doc_type)
        os.makedirs(path, exist_ok=True)
        return path
    return VECTORSTORE_DIR

# -----------------------------
# --- Document Endpoints ---
# -----------------------------
@app.get("/api/documents", response_model=List[UploadResponse])
async def get_documents():
    return list_documents(doc_type='document')

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
    """
    Upload a file to a specific doc_type folder (e.g., 'document' or 'faq')
    and process it immediately into a vectorstore.
    """
    folder = os.path.join(DATA_DIR, doc_type)
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, file.filename)

    # Save uploaded file
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Process document with correct doc_type
    try:
        doc_id = process_document(
            file_path=file_path,
            file_name=file.filename,
            doc_type=doc_type  # ✅ ensure vectorstore saved in correct folder
        )
    except Exception as e:
        logger.error(f"Failed to process {file.filename} as {doc_type}: {e}")
        return UploadResponse(
            status="failed",
            doc_id=os.path.splitext(file.filename)[0],
            filename=file.filename,
            file_type=os.path.splitext(file.filename)[1],
            upload_date=datetime.now(timezone.utc).isoformat()
        )

    # Check if vectorstore exists
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
    os.remove(filepath)
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
# --- /ingest API ---
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

        # ตรวจสอบ vectorstore จริง
        if vectorstore_exists(doc_id, doc_type=request.doc_type):
            results.append({"doc_id": doc_id, "result": "success"})
        else:
            results.append({"doc_id": doc_id, "result": "failed", "error": "Vectorstore not found after processing"})

    return {"status": "completed", "results": results}


# -----------------------------
# --- Assessment Workflow Endpoints ---
# -----------------------------
@app.post("/process/start")
async def start_assessment_process(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_assessment_workflow)
   

# -----------------------------
# --- RAG Query Endpoint ---
# -----------------------------
# --- /query API (QA_PROMPT version, safe LLM) ---
# -----------------------------
from fastapi import Form
from core.rag_chain import load_all_vectorstores, create_rag_chain, QA_PROMPT, llm
from starlette.concurrency import run_in_threadpool

@app.post("/query")
async def query_endpoint(
    question: str = Form(...),
    doc_ids: Optional[str] = Form(None),
    doc_types: Optional[str] = Form(None)
):
    """
    Query multiple vectorstores using QA_PROMPT
    - doc_ids: comma-separated list of doc_ids
    - doc_types: comma-separated list of doc types ('document', 'faq')
    - ถ้าไม่ส่ง doc_ids → โหลด doc_ids ทั้งหมดใน doc_types
    """

    doc_id_list = doc_ids.split(",") if doc_ids else []
    doc_type_list = doc_types.split(",") if doc_types else ["document", "faq"]

    # ถ้าไม่ส่ง doc_ids → โหลด doc_ids ทั้งหมดจาก folder
    if not doc_id_list:
        doc_id_list = []
        for dt in doc_type_list:
            folder_path = os.path.join("vectorstore", dt)
            if os.path.exists(folder_path):
                doc_id_list.extend([
                    d for d in os.listdir(folder_path)
                    if os.path.isdir(os.path.join(folder_path, d))
                ])

    skipped = []

    try:
        multi_retriever = load_all_vectorstores(
            doc_ids=doc_id_list,
            doc_type=doc_type_list,
            top_k=5
        )
    except ValueError as e:
        return {"error": str(e), "skipped": skipped}

    # -----------------------------
    # --- ดึงทุก chunk ของเอกสาร ---
    # -----------------------------
    def get_all_docs_text(query_text):
        docs = multi_retriever._get_relevant_documents(query_text)
        text_blocks = [d.page_content for d in docs if hasattr(d, "page_content")]
        return "\n\n".join(text_blocks)

    context_text = await run_in_threadpool(lambda: get_all_docs_text(question))

    # -----------------------------
    # --- เตรียม prompt QA_PROMPT ---
    # -----------------------------
    prompt_text = QA_PROMPT.format(
        context=context_text,
        question=question
    )

    # -----------------------------
    # --- Safe LLM call ---
    # -----------------------------
    def call_llm_safe(prompt_text):
        res = llm.invoke(prompt_text)
        if isinstance(res, dict) and "result" in res:
            return res["result"]
        elif isinstance(res, str):
            return res
        else:
            return str(res)

    answer = await run_in_threadpool(lambda: call_llm_safe(prompt_text))

    return {
        "question": question,
        "doc_ids": doc_id_list,
        "doc_types": doc_type_list,
        "answer": answer,
        "skipped": skipped
    }



# -----------------------------
# --- /compare API (ปรับปรุง) ---
# -----------------------------
@app.post("/compare")
async def compare(
    doc1: str = Form(...),
    doc2: str = Form(...),
    query: str = Form(...),
    doc_types: Optional[str] = Form(None)  # optional
):
    """
    Compare documents using MultiDocRetriever.
    - รองรับหลาย doc_type (comma-separated)
    - ถ้าไม่ส่ง doc_types → ใช้ ["document", "faq"]
    """
    from core.vectorstore import vectorstore_exists, load_all_vectorstores

    # default doc_types
    if doc_types:
        doc_type_list = [dt.strip() for dt in doc_types.split(",") if dt.strip()]
    else:
        doc_type_list = ["document", "faq"]

    doc_ids = [doc1, doc2]

    # --- ตรวจสอบ vectorstore ---
    valid_docs = []
    skipped = []
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

    # --- Load retrievers using MultiDocRetriever ---
    try:
        multi_retriever = load_all_vectorstores(
            doc_ids=doc_ids,
            top_k=5,
            doc_type=doc_type_list
        )

        # --- Fetch document texts ---
        def get_docs_text(query_text):
            docs = multi_retriever._get_relevant_documents(query_text)
            doc_text_map = {doc1: "", doc2: ""}
            for d in docs:
                source = d.metadata.get("source")
                if source in doc_text_map:
                    doc_text_map[source] += (d.page_content + "\n")
            # ถ้าไม่เจอ content → ใส่ placeholder
            for key in doc_text_map:
                if not doc_text_map[key]:
                    doc_text_map[key] = f"[No content for {key}]"
            return doc_text_map[doc1], doc_text_map[doc2]

        doc1_text, doc2_text = await run_in_threadpool(lambda: get_docs_text(query))
        context_text = f"{doc1_text}\n\n{doc2_text}"

        # --- Generate comparison ---
        prompt_text = COMPARE_PROMPT.format(
            context=context_text,
            query=query,
            doc_names=f"{doc1} และ {doc2}"
        )

        delta_answer = await run_in_threadpool(lambda: llm.invoke(prompt_text))

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

    except Exception as e:
        logging.error(f"Compare failed: {e}")
        raise HTTPException(status_code=500, detail=f"Compare failed: {e}")




