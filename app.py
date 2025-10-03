import os
import logging
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional
from operator import itemgetter
from starlette.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from core.vectorstore import load_vectorstore, vectorstore_exists
import datetime

# -----------------------------
# --- Pydantic Models ---
# -----------------------------
from pydantic import BaseModel

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
# --- Core Imports ---
# -----------------------------
from core.ingest import process_document, list_documents, delete_document, DATA_DIR
from core.rag_analysis_utils import answer_question_rag, match_doc_ids_from_question
from core.vectorstore import load_vectorstore
from core.rag_chain import (
    MultiDocRetriever,
    create_rag_chain,
    llm,
    COMPARE_PROMPT,
    QA_PROMPT,
    run_assessment_workflow,
    get_workflow_status,
    get_workflow_results
)

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

# -----------------------------
# --- Startup Event ---
# -----------------------------
@app.on_event("startup")
async def startup_event():
    os.makedirs(DATA_DIR, exist_ok=True)
    logging.info(f"Data directory '{DATA_DIR}' ensured.")

# -----------------------------
# --- Helper Functions ---
# -----------------------------
def format_docs(docs):
    """Format documents into a string for prompts."""
    formatted = []
    for doc in docs:
        formatted.append(doc.page_content if hasattr(doc, "page_content") else str(doc))
    return "\n\n".join(formatted)

# -----------------------------
# --- Document Endpoints ---
# -----------------------------
@app.get("/api/documents")
async def get_documents():
    """Return all documents from vectorstore (existing logic)"""
    return list_documents()

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
        upload_date=datetime.utcnow().isoformat()
    )

@app.post("/upload/faq", response_model=List[UploadResponse])
async def upload_faq_files(file: List[UploadFile] = File(...)):
    responses = []
    faq_folder = os.path.join(DATA_DIR, "faq")
    os.makedirs(faq_folder, exist_ok=True)
    for f in file:
        path = os.path.join(faq_folder, f.filename)
        content = await f.read()
        with open(path, "wb") as out:
            out.write(content)
        doc_id = process_document(path, f.filename)
        responses.append(UploadResponse(
            status="processed",
            doc_id=doc_id,
            filename=f.filename,
            file_type=os.path.splitext(f.filename)[1],
            upload_date=datetime.utcnow().isoformat()
        ))
    return responses

@app.post("/upload/{type}", response_model=UploadResponse)
async def upload_assessment_file(type: str, file: UploadFile = File(...)):
    if type not in ['rubrics', 'qa', 'evidence', 'feedback']:
        raise HTTPException(status_code=400, detail="Invalid upload type")
    folder = os.path.join(DATA_DIR, type)
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    doc_id = process_document(file_path, file.filename)
    return UploadResponse(
        status="processed",
        doc_id=doc_id,
        filename=file.filename,
        file_type=os.path.splitext(file.filename)[1],
        upload_date=datetime.utcnow().isoformat()
    )

@app.get("/api/uploads/{type}", response_model=List[UploadResponse])
async def list_uploads_by_type(type: str):
    """List uploaded files by type"""
    if type not in ['rubrics', 'qa', 'evidence', 'feedback']:
        raise HTTPException(status_code=400, detail="Invalid type")
    
    folder = os.path.join(DATA_DIR, type)
    os.makedirs(folder, exist_ok=True)
    uploads = []
    
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if os.path.isfile(path):
            doc_id = os.path.splitext(filename)[0]
            uploads.append(UploadResponse(
                status="Ingested",
                doc_id=doc_id,
                filename=filename,
                file_type=os.path.splitext(filename)[1],
                upload_date=datetime.utcfromtimestamp(os.path.getmtime(path)).isoformat()
            ))
    return uploads

from fastapi.responses import FileResponse

# -----------------------------
# --- Upload Management APIs (match frontend)
# -----------------------------
@app.get("/upload/{type}/list", response_model=List[UploadResponse])
async def list_uploads_by_type_for_ui(type: str):
    """
    List uploaded files by type for frontend, with dynamic status:
    - Ingested
    - Pending
    - Error
    """
    if type not in ['rubrics', 'qa', 'evidence', 'feedback']:
        raise HTTPException(status_code=400, detail="Invalid type")
    
    folder = os.path.join(DATA_DIR, type)
    os.makedirs(folder, exist_ok=True)

    uploads = []
    for filename in os.listdir(folder):
        if not os.path.isfile(os.path.join(folder, filename)):
            continue
        doc_id = os.path.splitext(filename)[0]

        # ตรวจสอบ vectorstore ว่ามี doc_id หรือไม่
        try:
            if vectorstore_exists(doc_id):  # ฟังก์ชันเช็คว่ามี vectorstore หรือไม่
                status = "Ingested"
            else:
                status = "Pending"
        except Exception as e:
            status = "Error"

        uploads.append(UploadResponse(
            status=status,
            doc_id=doc_id,
            filename=filename,
            file_type=os.path.splitext(filename)[1],
            upload_date=datetime.utcfromtimestamp(os.path.getmtime(os.path.join(folder, filename))).isoformat()
        ))
    
    return uploads


@app.delete("/upload/{type}/{file_id}")
async def delete_upload(type: str, file_id: str):
    """
    ลบไฟล์ตาม type และ file_id
    """
    folder = os.path.join(DATA_DIR, type)
    filepath = os.path.join(folder, f"{file_id}")
    
    # handle ทั้งกรณีมีนามสกุล/ไม่มีนามสกุล
    target = None
    if os.path.exists(filepath):
        target = filepath
    else:
        # หาไฟล์ที่ prefix ตรง file_id
        for f in os.listdir(folder):
            if os.path.splitext(f)[0] == file_id:
                target = os.path.join(folder, f)
                break
    
    if not target or not os.path.exists(target):
        raise HTTPException(status_code=404, detail="File not found")

    os.remove(target)
    return {"status": "deleted", "file_id": file_id}


@app.get("/upload/{type}/{file_id}")
async def download_upload(type: str, file_id: str):
    """
    ดาวน์โหลดไฟล์ตาม type และ file_id
    """
    folder = os.path.join(DATA_DIR, type)
    filepath = None
    for f in os.listdir(folder):
        if os.path.splitext(f)[0] == file_id:
            filepath = os.path.join(folder, f)
            break
    
    if not filepath or not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(filepath, filename=os.path.basename(filepath))



# -----------------------------
# --- RAG Query Endpoint ---
# -----------------------------
@app.post("/query", response_model=QueryResponse)
async def query_endpoint(question: str = Form(...), doc_ids: Optional[str] = Form(None)):
    try:
        docs_to_use = [d.strip() for d in doc_ids.split(",")] if doc_ids else match_doc_ids_from_question(question)
        if not docs_to_use:
            raise HTTPException(status_code=404, detail="No matching documents found")
        def summarize_multi_docs():
            results = []
            for doc_id in docs_to_use:
                try:
                    summary = answer_question_rag(question, doc_id)
                except ValueError:
                    summary = f"Error: Vectorstore not found for {doc_id}"
                results.append({"doc_id": doc_id, "summary": summary})
            return results
        summaries = await run_in_threadpool(summarize_multi_docs)
        combined_answer = "\n\n".join([f"**{r['doc_id']}**:\n{r['summary']}" for r in summaries])
        return QueryResponse(answer=combined_answer)
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"RAG processing error: {e}")
        raise HTTPException(status_code=500, detail=f"RAG processing failed: {e}")

# -----------------------------
# --- Compare Endpoint ---
# -----------------------------
@app.post("/compare")
async def compare(
    doc1: str = Form(...),
    doc2: str = Form(...),
    query: str = Form(...),
):
    try:
        # โหลด retrievers
        retrievers = [load_vectorstore(doc1), load_vectorstore(doc2)]

        async def get_docs_text(retrievers, query_text):
            texts = []
            for retriever in retrievers:
                try:
                    # ใช้ get_relevant_documents
                    docs = await run_in_threadpool(lambda: retriever.get_relevant_documents(query_text))
                    # แปลงเป็น text
                    doc_text = "\n".join([d.page_content if hasattr(d, "page_content") else str(d) for d in docs])
                    texts.append(doc_text if doc_text else f"[No content for {doc1}]")
                except Exception as e:
                    logging.error(f"Error fetching documents: {e}")
                    texts.append(f"[Error fetching documents: {e}]")
            return texts

        context_texts = await get_docs_text(retrievers, query)
        context_text = "\n\n".join(context_texts)
        doc1_text = context_texts[0] if len(context_texts) > 0 else "[No content for doc1]"
        doc2_text = context_texts[1] if len(context_texts) > 1 else "[No content for doc2]"

        # สร้าง prompt string
        prompt_text = COMPARE_PROMPT.format(
            context=context_text,
            query=query,
            doc_names=f"{doc1} และ {doc2}"
        )

        # เรียก LLM
        try:
            delta_answer = await run_in_threadpool(lambda: llm.predict(prompt_text))
        except Exception as e:
            logging.error(f"Error generating comparison: {e}")
            delta_answer = f"[Error generating comparison: {e}]"

        return {
            "metrics": [
                {
                    "metric": "Key comparison",
                    "doc1": doc1_text,
                    "doc2": doc2_text,
                    "delta": delta_answer,
                    "remark": "Comparison generated from MultiDocRetriever"
                }
            ]
        }

    except Exception as e:
        logging.error(f"Compare failed: {e}")
        raise HTTPException(status_code=500, detail=f"Compare failed: {e}")

# -----------------------------
# --- Assessment Workflow Endpoints ---
# -----------------------------
@app.post("/process/start")
async def start_assessment_process(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_assessment_workflow)
    return {"status": "Assessment process started in background"}

@app.get("/process/status", response_model=ProcessStatus)
async def get_process_status_endpoint():
    return get_workflow_status()

@app.get("/result/summary", response_model=ResultSummary)
async def get_result_summary_endpoint():
    results = get_workflow_results()
    evidence_count = sum(1 for r in results if "Strong" in r.get("summary", ""))
    gap_count = sum(1 for r in results if "Gap" in r.get("summary", ""))
    return ResultSummary(
        totalQuestions=len(results),
        averageScore=85.0,
        evidenceCount=evidence_count,
        gapCount=gap_count
    )

# -----------------------------
# --- FAQ File Endpoints ---
# -----------------------------
@app.get("/api/faq/files")
async def list_faq_files():
    folder = os.path.join(DATA_DIR, "faq")
    os.makedirs(folder, exist_ok=True)
    return [{"filename": f, "upload_date": datetime.utcfromtimestamp(os.stat(os.path.join(folder, f)).st_mtime).isoformat()} for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

@app.get("/api/faq/files/{filename}")
async def download_faq_file(filename: str):
    path = os.path.join(DATA_DIR, "faq", filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)

@app.delete("/api/faq/files/{filename}")
async def delete_faq_file(filename: str):
    path = os.path.join(DATA_DIR, "faq", filename)
    if os.path.exists(path):
        os.remove(path)
        delete_document(os.path.splitext(filename)[0])
        return {"status": "ok", "filename": filename}
    raise HTTPException(status_code=404, detail="File not found")

# -----------------------------
# API Status
# -----------------------------
@app.get("/api/status")
async def api_status():
    """
    เช็คสถานะ API เบื้องต้น
    """
    return {
        "status": "ok",
        "timestamp": datetime.datetime.now().isoformat(),
        "message": "API is running"
    }

# -----------------------------
# API Health Check
# -----------------------------
@app.get("/api/health")
async def api_health():
    """
    เช็คสุขภาพระบบ: database, retrievers, LLM connection (mock example)
    """
    try:
        # ตัวอย่างตรวจสอบ retriever/LLM connection
        retriever_status = "ok"  # เปลี่ยนเป็น check จริงถ้ามี
        llm_status = "ok"        # เปลี่ยนเป็น check จริงถ้ามี

        health = {
            "status": "healthy" if retriever_status == "ok" and llm_status == "ok" else "degraded",
            "components": {
                "retriever": retriever_status,
                "LLM": llm_status
            },
            "timestamp": datetime.datetime.now().isoformat()
        }
        return health
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        }

