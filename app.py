import os
import logging
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional

# --- Pydantic Import ---
from pydantic import BaseModel

# --- CORE IMPORTS (ใช้ Relative Import เพื่อให้ทำงานร่วมกันใน project structure ได้) ---
# การ import นี้ทำให้เราสามารถเรียกใช้ฟังก์ชัน compare_documents จริงได้
from core.ingest import process_document, list_documents, delete_document, DATA_DIR
from core.rag_analysis_utils import answer_question_rag, compare_documents
from core.rag_chain import run_assessment_workflow, get_workflow_status, get_workflow_results


# --- Pydantic/Response Models (FIXED) ---
# ใช้ Pydantic BaseModel เพื่อให้ FastAPI response_model ใช้งานได้
class UploadResponse(BaseModel):
    status: str
    doc_id: str
    filename: str
    file_type: str
    upload_date: str 

class CompareResultMetrics(BaseModel):
    metric: str
    # เพิ่ม doc1 และ doc2 เพื่อให้ตรงกับโครงสร้าง JSON ที่คุณต้องการในผลลัพธ์
    doc1: str 
    doc2: str
    delta: str
    remark: str

class CompareResponse(BaseModel):
    # Adjusted structure: {"result": {"metrics": [...]}}
    result: Dict[str, List[CompareResultMetrics]] 

class QueryResponse(BaseModel):
    answer: str
    conversation_id: Optional[str] = None

class StatusResponseMetrics(BaseModel):
    processed: int
    pending: int
    errors: int

class StatusResponseActivity(BaseModel):
    id: str
    action: str
    document: str
    timestamp: str

class StatusResponse(BaseModel):
    compliance_score: int
    total_documents: int
    recent_activities: List[StatusResponseActivity]
    metrics: StatusResponseMetrics

class Document(BaseModel):
    id: str
    filename: str
    file_type: str
    upload_date: str
    status: str

class ProcessStep(BaseModel):
    id: int
    name: str
    status: str # Should be 'waiting' | 'running' | 'done'
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

# --- FastAPI Initialization ---
app = FastAPI(
    title="Assessment RAG API",
    description="API for RAG-based document assessment and analysis."
)

# --- CORS Configuration ---
origins = [
    "http://localhost:5173",  # Vite/React default development port
    "http://127.0.0.1:5173",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Initialization on Startup ---
@app.on_event("startup")
async def startup_event():
    # DATA_DIR ต้องมีการกำหนดใน core/ingest.py
    temp_data_dir = os.environ.get("DATA_DIR", "./data")
    os.makedirs(temp_data_dir, exist_ok=True)
    logging.info(f"Data directory '{temp_data_dir}' ensured.")

# -----------------------------------------------------------
# 1. RAG Query Endpoint
# -----------------------------------------------------------

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(
    query_text: str = Form(...), 
    doc_ids: str = Form(...),
    conversation_id: Optional[str] = Form(None)
):
    """
    Handles a RAG query against the specified document(s).
    """
    logging.info(f"Received query: '{query_text}' for doc(s): {doc_ids}")
    
    try:
        # ใช้ฟังก์ชันจริงจาก rag_analysis_utils
        answer = answer_question_rag(question=query_text, doc_id=doc_ids)
        return QueryResponse(answer=answer)
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail="Error during RAG processing")


# -----------------------------------------------------------
# 2. Document Management Endpoints (Stubbed/Simplified)
# -----------------------------------------------------------

@app.post("/upload/{type}", response_model=UploadResponse)
async def upload_assessment_file(type: str, file: UploadFile = File(...)):
    if type not in ['rubrics', 'qa', 'evidence']:
        raise HTTPException(status_code=400, detail="Invalid upload type")

    # ใช้ DATA_DIR จาก core/ingest.py 
    temp_data_dir = os.environ.get("DATA_DIR", "./data")
    file_path = os.path.join(temp_data_dir, type, file.filename)
    os.makedirs(os.path.join(temp_data_dir, type), exist_ok=True)
    
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        # doc_id = process_document(file_path, file.filename, metadata={"type": type}) 
        doc_id = f"doc-id-{file.filename}"
        
        return UploadResponse(
            status="processed",
            doc_id=doc_id,
            filename=file.filename,
            file_type=os.path.splitext(file.filename)[1],
            upload_date=datetime.now().isoformat()
        )
    except Exception as e:
        logging.error(f"Upload/Ingest failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload/Ingest failed: {e}")

@app.post("/compare", response_model=CompareResponse)
async def compare_documents_endpoint(
    doc1: str = Form(...), 
    doc2: str = Form(...),
    # เพิ่ม Form parameter สำหรับ query ซึ่งเป็น optional
    query: Optional[str] = Form(None, description="Specific question or topic for comparison")
):
    """Compares two documents using RAG analysis and structured output."""
    logging.info(f"Comparison requested for {doc1} vs {doc2} (Query: {query})")
    
    try:
        # 1. CALL THE ACTUAL COMPARISON FUNCTION
        comparison_result = compare_documents(doc_a_id=doc1, doc_b_id=doc2, query=query)
        
        # 2. Check for error from the utility function
        if "error" in comparison_result:
             raise HTTPException(status_code=500, detail=comparison_result["error"])
        
        # 3. Format metrics list for the Pydantic model
        formatted_metrics = []
        for metric_dict in comparison_result.get("metrics", []):
            formatted_metrics.append(CompareResultMetrics(
                metric=metric_dict.get("metric", "N/A"),
                doc1=doc1, # Inject doc1 ID
                doc2=doc2, # Inject doc2 ID
                delta=metric_dict.get("delta", "N/A"),
                remark=metric_dict.get("remark", "No detailed remark.")
            ))

        # 4. Return the result in the required format: {"result": {"metrics": [...]}}
        return CompareResponse(
            result={
                "metrics": formatted_metrics
            }
        )
    except HTTPException:
        # Re-raise explicit HTTP exceptions
        raise
    except Exception as e:
        logging.error(f"Error during comparison: {e}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {e}")

# -----------------------------------------------------------
# 3. Assessment Workflow Endpoints
# -----------------------------------------------------------

@app.post("/process/start")
async def start_assessment_process(background_tasks: BackgroundTasks):
    """Starts the 5-step assessment workflow in the background."""
    background_tasks.add_task(run_assessment_workflow)
    return {"status": "Assessment process started in background"}

@app.get("/process/status", response_model=ProcessStatus)
async def get_process_status_endpoint():
    """Returns the current status and progress of the assessment workflow."""
    return get_workflow_status()

@app.get("/result/summary", response_model=ResultSummary)
async def get_result_summary_endpoint():
    """Returns a summary of assessment results."""
    # Placeholder implementation
    results = get_workflow_results()
    
    # Simple mock calculation based on mock results
    evidence_count = sum(1 for r in results if "Strong" in r.get("summary", ""))
    gap_count = sum(1 for r in results if "Gap" in r.get("summary", ""))
    
    return ResultSummary(
        totalQuestions=len(results),
        averageScore=85.0, 
        evidenceCount=evidence_count,
        gapCount=gap_count,
    )
