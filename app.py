import logging
import os
import json
from langchain.schema import Document, SystemMessage, HumanMessage 
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel
from contextlib import asynccontextmanager

# --- Core Imports ---
from core.rag_prompts import QA_PROMPT, COMPARE_PROMPT, SYSTEM_QA_INSTRUCTION 
from core.ingest import process_document, list_documents, delete_document, DATA_DIR, SUPPORTED_TYPES
from core.vectorstore import load_vectorstore, vectorstore_exists, load_all_vectorstores, get_vectorstore_path
from langchain.chains import RetrievalQA

# ❗❗ FIXED: Import LLM Instance Explicitly to resolve AttributeError
from models.llm import llm as llm_instance 

# --- NEW: Import the assessment function ---
from core.run_assessment import run_assessment_process 
# -------------------------------------------


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
REF_DATA_DIR = "ref_data"  # 🟢 NEW: Global Constant สำหรับ Reference Data Directory

# -----------------------------
# --- Helper Functions (สำหรับจัดการ JSON Files) ---
# -----------------------------

def get_ref_data_path(enabler: str, data_type: str) -> str:
    """สร้าง path เต็มของไฟล์ JSON สำหรับ Enabler และ Data Type ที่ระบุ"""
    enabler = enabler.lower()
    
    # กำหนดชื่อไฟล์ตาม Data Type ที่เรารู้จัก
    if data_type == 'statements':
        filename = f"{enabler}_evidence_statements_checklist.json"
    elif data_type == 'rubrics':
        filename = f"{enabler}_rating_criteria_rubric.json"
    elif data_type == 'mapping':
        filename = f"{enabler}_evidence_mapping.json"
    elif data_type == 'weighting':
        filename = f"{enabler}_scoring_level_fractions.json"
    else:
        # NOTE: ถ้าเป็น Path Parameter ใน FastAPI จะถูกดักจับก่อน
        raise ValueError(f"Invalid data_type: {data_type}")
        
    # สมมติว่าไฟล์ JSON อยู่ในโฟลเดอร์หลักของ REF_DATA_DIR
    return os.path.join(REF_DATA_DIR, filename) 

def load_ref_data_file(filepath: str) -> Any:
    """อ่านและโหลดข้อมูล JSON จากไฟล์ที่กำหนด"""
    if not os.path.exists(filepath):
        # คืนค่า default ถ้าไฟล์ยังไม่มี
        if any(keyword in filepath for keyword in ['statements', 'mapping', 'rubric']):
            return []
        return {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON from {filepath}")
        # NOTE: ใช้ HTTPException ใน Helper Function เพราะมันถูกเรียกใน endpoint
        raise HTTPException(status_code=500, detail=f"Invalid JSON format in {filepath}")
    except Exception as e:
        logger.error(f"Error loading file {filepath}: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading file: {filepath}")

def save_ref_data_file(filepath: str, data: Any):
    """บันทึกข้อมูล JSON กลับลงในไฟล์"""
    # ตรวจสอบ/สร้าง โฟลเดอร์หลัก (REF_DATA_DIR) อีกครั้งเพื่อความปลอดภัย
    os.makedirs(os.path.dirname(filepath), exist_ok=True) 
    with open(filepath, 'w', encoding='utf-8') as f:
        # ใช้ indent 2 เพื่อให้อ่านง่าย
        json.dump(data, f, ensure_ascii=False, indent=2)


# -----------------------------
# --- Lifespan (แทน on_event) ---
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager แทน @app.on_event สำหรับ startup/shutdown"""
    # --- Startup ---
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    os.makedirs(REF_DATA_DIR, exist_ok=True) # 🟢 NEW: สร้างโฟลเดอร์ Reference Data
    logging.info(f"✅ Data directory '{DATA_DIR}', vectorstore '{VECTORSTORE_DIR}', and ref_data '{REF_DATA_DIR}' ensured.")

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
    
class AssessmentRequest(BaseModel):
    enabler: str = "KM"
    sub_criteria_id: str = "all"
    mode: str = "real"
    filter_mode: bool = False
    export_results: bool = False
    
class AssessmentRecord(BaseModel):
    record_id: str
    enabler: str
    sub_criteria_id: str
    mode: str
    timestamp: str
    
    # 🟢 UPDATE: เพิ่มสถานะ
    status: str = "RUNNING" # PENDING, RUNNING, COMPLETED, FAILED

    # 🟢 UPDATE: ทำให้คะแนนเป็น Optional
    overall_score: Optional[float] = None
    highest_full_level: Optional[int] = None
    export_path: Optional[str] = None

# 🟢 NEW: Pydantic Model สำหรับ Reference Data Payload
class RefDataPayload(BaseModel):
    data: Dict | List 
    
# Global List of Assessment Records (in-memory for demo/simple environment)
ASSESSMENT_HISTORY: List[AssessmentRecord] = []

# -----------------------------
# --- Assessment Endpoints ---
# -----------------------------
@app.post("/api/assess")
async def run_assessment_task(request: AssessmentRequest, background_tasks: BackgroundTasks):
    record_id = os.urandom(8).hex()
    
    # 1. สร้างและบันทึก Record สถานะ RUNNING เข้า History ทันที
    initial_record = AssessmentRecord(
        record_id=record_id,
        enabler=request.enabler.upper(),
        sub_criteria_id=request.sub_criteria_id,
        mode=request.mode,
        timestamp=datetime.now(timezone.utc).isoformat(),
        status="RUNNING" 
    )
    ASSESSMENT_HISTORY.append(initial_record)
    
    # 2. ส่ง Task ไปรัน Background
    background_tasks.add_task(_background_assessment_runner, record_id, request)
    
    # 3. ตอบกลับ client
    return {"status": "accepted", "record_id": record_id, "message": "Assessment started in background. Check /api/assess/history for status."}

# -----------------------------
# --- Assessment History Endpoint (UPDATED: A1) ---
# -----------------------------
@app.get("/api/assess/history", response_model=List[AssessmentRecord])
async def get_assessment_history(enabler: Optional[str] = None): # ⬅️ รับ Query Parameter 'enabler'
    
    filtered_history = ASSESSMENT_HISTORY
    
    # 🟢 LOGIC: ทำการ Filter ถ้ามี Enabler ถูกส่งมา
    if enabler:
        enabler_upper = enabler.upper()
        
        filtered_history = [
            record for record in ASSESSMENT_HISTORY 
            if record.enabler.upper() == enabler_upper
        ]
        
    # 3. เรียงลำดับตามเวลาและส่งกลับ
    return sorted(filtered_history, key=lambda r: r.timestamp, reverse=True)


@app.get("/api/assess/results/{record_id}")
async def get_assessment_results(record_id: str):
    record = next((r for r in ASSESSMENT_HISTORY if r.record_id == record_id), None)
    if not record:
        raise HTTPException(status_code=404, detail="Assessment record not found.")

    if record.export_path and os.path.exists(record.export_path):
        return FileResponse(record.export_path, media_type="application/json", filename=os.path.basename(record.export_path))
    
    raise HTTPException(status_code=404, detail="Full assessment data not available for this record.")


# -----------------------------
# --- Reference Data Endpoints (NEW: R1, R2, R3) ---
# -----------------------------

# R1: GET /api/ref_data/{enabler}
@app.get("/api/ref_data/{enabler}")
async def get_all_reference_data(enabler: str):
    """
    R1: ดึงข้อมูล Reference Data ทั้ง 4 ชนิด (Statements, Rubrics, Mapping, Weighting) 
    ของ Enabler ที่ระบุในครั้งเดียว
    """
    enabler = enabler.lower()
    data = {}
    
    def load_data_safe(data_type: str):
        filepath = get_ref_data_path(enabler, data_type)
        return load_ref_data_file(filepath)

    try:
        # โหลดข้อมูลทั้งหมด
        data['statements'] = await run_in_threadpool(lambda: load_data_safe('statements'))
        data['rubrics'] = await run_in_threadpool(lambda: load_data_safe('rubrics'))
        data['mapping'] = await run_in_threadpool(lambda: load_data_safe('mapping'))
        data['weighting'] = await run_in_threadpool(lambda: load_data_safe('weighting'))
        
        data['enabler'] = enabler.upper()

        return data
    except HTTPException:
        raise # ส่ง HTTPException 500 ที่มาจาก load_ref_data_file ต่อ
    except Exception as e:
        logger.error(f"Error loading all ref data for {enabler}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load all reference data for {enabler}")


# R2: POST /api/ref_data/{enabler}/{data_type}
@app.post("/api/ref_data/{enabler}/{data_type}")
async def save_reference_data(enabler: str, data_type: str, payload: RefDataPayload):
    """
    R2: บันทึกข้อมูล Reference Data (Statements, Rubrics, Mapping, หรือ Weighting) 
    """
    enabler = enabler.lower()
    
    # 1. ตรวจสอบ data_type ที่ถูกต้อง
    if data_type not in ['statements', 'rubrics', 'mapping', 'weighting']:
        raise HTTPException(status_code=400, detail="Invalid data_type. Must be one of: statements, rubrics, mapping, weighting.")
        
    # 2. สร้าง Path และบันทึก
    filepath = get_ref_data_path(enabler, data_type)
    
    try:
        await run_in_threadpool(lambda: save_ref_data_file(filepath, payload.data))
        logger.info(f"Saved {data_type} for {enabler} to {filepath}")
        return {"status": "success", "enabler": enabler.upper(), "data_type": data_type}
    except Exception as e:
        logger.error(f"Error saving {data_type} for {enabler}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save {data_type} data.")


# R3: POST /api/ref_data/auto_map/{enabler}
@app.post("/api/ref_data/auto_map/{enabler}")
async def trigger_auto_mapping(enabler: str, background_tasks: BackgroundTasks):
    """
    R3: Trigger Background Task สำหรับการทำ Auto Mapping/LLM Generation 
    """
    enabler = enabler.lower()
    
    # 🟢 ส่ง Task ไปรัน Background
    background_tasks.add_task(_background_auto_mapper, enabler)
    
    return {"status": "accepted", "enabler": enabler.upper(), "message": "Auto Mapping process started in background."}


# -----------------------------
# --- Background Runner Logic ---
# -----------------------------

def _background_assessment_runner(record_id: str, request: AssessmentRequest):
    logger.info(f"Processing background assessment for {record_id}...")
    
    record = next((r for r in ASSESSMENT_HISTORY if r.record_id == record_id), None)
    if not record:
        logger.error(f"FATAL: Initial record {record_id} not found in history list. Exiting runner.")
        return 
        
    try:
        # 2. CALL THE NEW FUNCTION (ใช้เวลา)
        final_summary = run_assessment_process(
            enabler=request.enabler,
            sub_criteria_id=request.sub_criteria_id,
            mode=request.mode,
            filter_mode=request.filter_mode,
            export=True # Force export in background task to get a stable file for /results
        )
        
        # 3. 🟢 อัปเดตผลลัพธ์และสถานะ (Completed)
        record.overall_score = final_summary['Overall']['overall_maturity_score']
        
        # Logic สำหรับดึง highest_full_level (ตามโค้ดเดิม)
        sub_id_for_level = request.sub_criteria_id if request.sub_criteria_id != 'all' else list(final_summary['SubCriteria_Breakdown'].keys())[0] if final_summary['SubCriteria_Breakdown'] else None
        record.highest_full_level = final_summary['SubCriteria_Breakdown'].get(sub_id_for_level, {}).get('highest_full_level', 0) if sub_id_for_level else 0
        
        record.export_path = final_summary.get("export_path_used")
        record.status = "COMPLETED" # 🟢 สถานะเสร็จสมบูรณ์
        
        record.timestamp = datetime.now(timezone.utc).isoformat()
        
        logger.info(f"Assessment {record_id} completed successfully. Score: {record.overall_score:.2f}")

    except Exception as e:
        logger.error(f"Assessment task {record_id} failed: {e}")
        # 5. 🟢 อัปเดต Record สถานะล้มเหลว
        record.overall_score = -1.0
        record.highest_full_level = -1
        record.status = "FAILED"
        record.timestamp = datetime.now(timezone.utc).isoformat()

# -----------------------------
# --- Auto Mapping Background Runner (R3 Logic) ---
# -----------------------------
def _background_auto_mapper(enabler: str):
    logger.info(f"Starting Auto Mapping for {enabler}...")
    
    # **********************************************
    # *** Backend Team ต้องเพิ่ม Logic LLM/Generation ที่นี่ ***
    # **********************************************
    
    try:
        # NOTE: นี่คือการจำลองการทำงานที่กินเวลานาน
        import time
        time.sleep(10) 
        
        # 🟢 เมื่อเสร็จแล้ว ควรเรียกใช้ Logic การสร้างไฟล์และการบันทึก:
        #
        # 1. GENERATE DATA (LLM/Custom Logic)
        # new_mapping_data = generate_mapping_data(enabler) 
        
        # 2. SAVE TO FILE 
        # filepath = get_ref_data_path(enabler, 'mapping')
        # save_ref_data_file(filepath, new_mapping_data) 
        
        logger.info(f"Auto Mapping for {enabler} completed and saved successfully (Simulated).")

    except Exception as e:
        logger.error(f"Auto Mapping task for {enabler} failed: {e}")
        
        
# -----------------------------
# --- Document Endpoints (โค้ดเดิม) ---
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
# --- Upload Endpoints (โค้ดเดิม) ---
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
# --- Ingest Endpoint (โค้ดเดิม) ---
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
# --- Query Endpoint (ปรับปรุงแล้ว) ---
# -----------------------------
@app.post("/query")
async def query_endpoint(
    question: str = Form(...),
    doc_ids: Optional[str] = Form(None),
    doc_types: Optional[str] = Form(None)
):
    """
    RAG Endpoint สำหรับถามเอกสารหลายฉบับ
    Output เป็น string สำหรับ UI
    """
    # แปลง doc_ids/doc_types เป็น list
    doc_id_list = doc_ids.split(",") if doc_ids else None
    doc_type_list = doc_types.split(",") if doc_types else None
    skipped = []

    # โหลด vectorstores (เฉพาะ doc_ids ที่เลือก)
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

    # ฟอร์แมต context: แยกเอกสาร + ชื่อจริง/metadata
    def format_context_for_multiple_docs(docs):
        context_sections = []
        for i, d in enumerate(docs, 1):
            doc_name = d.metadata.get("name", f"Document {i}")
            context_sections.append(f"[{doc_name}]\n{d.page_content}")
        return "\n\n".join(context_sections)

    # ดึงเอกสาร rerank แล้ว กรองตาม doc_id ที่เลือก
    def get_all_docs_text(query_text):
        docs = multi_retriever._get_relevant_documents(query_text)
        if doc_id_list:
            docs = [d for d in docs if d.metadata.get("doc_id") in doc_id_list]
        return format_context_for_multiple_docs(docs)

    context_text = await run_in_threadpool(lambda: get_all_docs_text(question))

    # สร้าง Prompt
    human_message_content = QA_PROMPT.format(context=context_text, question=question)
    messages = [
        SystemMessage(content=SYSTEM_QA_INSTRUCTION),
        HumanMessage(content=human_message_content)
    ]

    # เรียก LLM แบบ safe
    def call_llm_safe(messages_list: List[Any]) -> str:
        res = llm_instance.invoke(messages_list)
        if isinstance(res, dict) and "result" in res:
            return res["result"]
        elif hasattr(res, 'content'):
            return res.content.strip()
        elif isinstance(res, str):
            return res.strip()
        return str(res).strip()

    answer_text = await run_in_threadpool(lambda: call_llm_safe(messages))

    # แปลง LLM output เป็น string สำหรับ UI
    output = {
        "question": question,
        "doc_ids": loaded_doc_ids,
        "doc_types": doc_type_list if doc_type_list else ["document", "faq"],
        "answer": "",
        "skipped": skipped
    }

    try:
        import json
        llm_json = json.loads(answer_text)
        flattened_answer = []

        if 'summary' in llm_json and llm_json['summary']:
            flattened_answer.append("📌 Summary:\n" + llm_json['summary'])

        if 'details' in llm_json and llm_json['details']:
            for d in llm_json['details']:
                flattened_answer.append(f"📄 {d.get('doc_name', '')}: {d.get('text', '')}")

        if 'comparison' in llm_json and llm_json['comparison']:
            flattened_answer.append("⚖️ Comparison:")
            for k,v in llm_json['comparison'].items():
                flattened_answer.append(f"{k}: {v}")

        if 'search_results' in llm_json and llm_json['search_results']:
            flattened_answer.append("🔍 Search Results:")
            for r in llm_json['search_results']:
                flattened_answer.append(f"{r.get('doc_name','')}: {r.get('text','')}")

        output['answer'] = "\n\n".join(flattened_answer) if flattened_answer else answer_text

    except Exception:
        output['answer'] = answer_text

    return output


# -----------------------------
# --- Compare Endpoint (ปรับปรุงใหม่) ---
# -----------------------------
@app.post("/compare")
async def compare(
    doc1: str = Form(...),
    doc2: str = Form(...),
    query: str = Form(...),
    doc_types: Optional[str] = Form(None)
):
    from core.vectorstore import vectorstore_exists 
    
    doc_type_list = [dt.strip() for dt in doc_types.split(",")] if doc_types else ["document"]

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
        
        for d in docs:
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

    # 🟢 NEW RAG PROMPT STRUCTURE IMPLEMENTATION for Compare
    
    # 1. Format the Human Message content (Context + Question)
    human_message_content = COMPARE_PROMPT.format(
        context=context_text, 
        query=query, 
        doc_names=f"{doc1} และ {doc2}"
    )

    # 2. Create the message list (System Instruction + Human Message)
    messages = [
        SystemMessage(content=SYSTEM_QA_INSTRUCTION),
        HumanMessage(content=human_message_content)
    ]
    
    def call_llm_safe(messages_list: List[Any]) -> str:
        """Helper to invoke LLM with message list and extract content (reused from /query)."""
        res = llm_instance.invoke(messages_list) 
        if isinstance(res, dict) and "result" in res:
            return res["result"]
        elif hasattr(res, 'content'): 
            return res.content.strip()
        elif isinstance(res, str):
            return res.strip()
        return str(res).strip()

    # 3. Call LLM with the new messages list
    delta_answer = await run_in_threadpool(lambda: call_llm_safe(messages))
    # 🟢 END NEW RAG PROMPT STRUCTURE IMPLEMENTATION

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
# --- Health & Status (โค้ดเดิม) ---
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
