import os
import logging
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional
from operator import itemgetter
from starlette.concurrency import run_in_threadpool
import traceback

# --- Pydantic ---
from pydantic import BaseModel

# --- Core Imports ---
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

# --- LangChain / LCEL ---
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# -----------------------------
# --- FastAPI Initialization ---
# -----------------------------
app = FastAPI(
    title="Assessment RAG API",
    description="API for RAG-based document assessment and analysis."
)

# --- CORS ---
origins = [
    "http://localhost:5173",
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

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
# --- Helper Functions ---
# -----------------------------
def format_docs(docs):
    """
    Formats retrieved documents into a single string for the prompt context.
    Supports both list of Document objects (with .page_content) or list of strings.
    """
    formatted = []
    for doc in docs:
        if hasattr(doc, "page_content"):
            formatted.append(doc.page_content)
        else:
            formatted.append(str(doc))
    return "\n\n".join(formatted)

# -----------------------------
# --- Startup Event ---
# -----------------------------
@app.on_event("startup")
async def startup_event():
    temp_data_dir = os.environ.get("DATA_DIR", "./data")
    os.makedirs(temp_data_dir, exist_ok=True)
    logging.info(f"Data directory '{temp_data_dir}' ensured.")

# -----------------------------
# --- RAG Query Endpoint ---
# -----------------------------
@app.post("/query", response_model=QueryResponse)
async def query_endpoint(
    question: str = Form(...), 
    doc_ids: Optional[str] = Form(None),
    conversation_id: Optional[str] = Form(None)
):
    try:
        logging.info(f"Received query: '{question}' doc_ids: {doc_ids}")
        docs_to_use = [d.strip() for d in doc_ids.split(",") if d.strip()] if doc_ids else match_doc_ids_from_question(question)

        if not docs_to_use:
            raise HTTPException(status_code=404, detail="No matching documents found for this query")

        answer = answer_question_rag(question=question, doc_id=",".join(docs_to_use))
        return QueryResponse(answer=answer, conversation_id=conversation_id)
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error during RAG processing: {e}")

# -----------------------------
# --- Document Upload Endpoint ---
# -----------------------------
@app.post("/upload/{type}", response_model=UploadResponse)
async def upload_assessment_file(type: str, file: UploadFile = File(...)):
    if type not in ['rubrics', 'qa', 'evidence']:
        raise HTTPException(status_code=400, detail="Invalid upload type")

    temp_data_dir = os.environ.get("DATA_DIR", "./data")
    file_path = os.path.join(temp_data_dir, type, file.filename)
    os.makedirs(os.path.join(temp_data_dir, type), exist_ok=True)
    
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
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

# -----------------------------
# --- Compare Endpoint ---
# -----------------------------
@app.post("/compare")
async def compare(doc1: str = Form(...), doc2: str = Form(...)):
    if llm is None:
        raise HTTPException(status_code=500, detail="LLM not initialized. Check core/rag_chain.py setup.")
        
    try:
        # โหลด retriever ของทั้งสองเอกสาร
        retrievers = [load_vectorstore(doc1), load_vectorstore(doc2)]
        multi_retriever = MultiDocRetriever(retrievers_list=retrievers)

        # Prompt สำหรับเปรียบเทียบ
        prompt = COMPARE_PROMPT

        # LCEL Chain: แปลง output ของ retriever ให้เป็น Document objects
        def docs_lambda(inputs):
            docs = multi_retriever.get_relevant_documents(inputs["query"])
            # ถ้า docs เป็น string list ให้แปลงเป็น Document
            from langchain.schema import Document
            document_list = []
            for idx, d in enumerate(docs):
                if isinstance(d, str):
                    document_list.append(Document(page_content=d, metadata={"source": f"doc_{idx}"}))
                else:
                    document_list.append(d)  # ถ้าเป็น Document อยู่แล้ว
            return document_list

        # สร้าง stuff_documents_chain
        combine_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=prompt,
            document_variable_name="context"
        )

        # สร้าง LCEL Comparison Chain
        from langchain_core.runnables import RunnablePassthrough, RunnableLambda
        from langchain_core.output_parsers import StrOutputParser
        from operator import itemgetter

        question_text = (
            f"เปรียบเทียบเอกสาร {doc1} กับ {doc2} อย่างละเอียด "
            "โดยสรุปเฉพาะความแตกต่างในด้านเป้าหมาย ตัวชี้วัด และระยะเวลาดำเนินการ"
        )

        input_data = {
            "query": question_text,
            "doc_names": f"{doc1} และ {doc2}"
        }

        comparison_chain = (
            RunnablePassthrough.assign(
                query=itemgetter("query"),
                doc_names=itemgetter("doc_names"),
                context=RunnableLambda(docs_lambda)
            )
            | combine_chain
            | StrOutputParser()
        )

        # เรียก invoke ผ่าน run_in_threadpool
        delta_answer_str = await run_in_threadpool(lambda: comparison_chain.invoke(input_data))
        delta_answer = str(delta_answer_str)

        # สรุปเอกสารทั้งสองโดยใช้ RetrievalQA
        def summarize_docs():
            from langchain.chains import RetrievalQA
            # Doc1
            retriever1 = load_vectorstore(doc1)
            qa_chain1 = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever1,
                chain_type="stuff",
                chain_type_kwargs={"prompt": QA_PROMPT}
            )
            result1 = qa_chain1.invoke({"query": f"สรุปหัวข้อหลักและเนื้อหาสำคัญของเอกสาร {doc1}"})
            summary1 = result1.get("result", "") if isinstance(result1, dict) else str(result1)

            # Doc2
            retriever2 = load_vectorstore(doc2)
            qa_chain2 = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever2,
                chain_type="stuff",
                chain_type_kwargs={"prompt": QA_PROMPT}
            )
            result2 = qa_chain2.invoke({"query": f"สรุปหัวข้อหลักและเนื้อหาสำคัญของเอกสาร {doc2}"})
            summary2 = result2.get("result", "") if isinstance(result2, dict) else str(result2)

            return summary1, summary2

        summary1, summary2 = await run_in_threadpool(summarize_docs)

        return {
            "result": {
                "metrics": [{
                    "metric": "หัวข้อหลักและเนื้อหาสำคัญและการเปรียบเทียบ",
                    "doc1": summary1,
                    "doc2": summary2,
                    "delta": delta_answer,
                    "remark": "ผลการเปรียบเทียบได้จาก Multi-Document Retrieval ที่เน้นความแตกต่าง"
                }]
            }
        }

    except Exception as e:
        import traceback
        logging.error(f"Comparison failed: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"การเปรียบเทียบไม่สำเร็จ: {e}")


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
