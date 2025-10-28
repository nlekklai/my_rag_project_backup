# app.py (Full Code - Fixed stable_doc_uuid argument in /ingest)
from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import os
from datetime import datetime, timezone
import time
import logging
import json
from langchain.schema import Document, SystemMessage, HumanMessage 
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
from contextlib import asynccontextmanager

# --- Core Imports ---
# 🟢 NOTE: Mock logic is present, using Bracket Notation in upload functions
# (สมมติว่า core.ingest, core.vectorstore, models.llm, core.run_assessment, core.evidence_mapping_generator มีอยู่จริง)
try:
    from core.rag_prompts import QA_PROMPT, COMPARE_PROMPT, SYSTEM_QA_INSTRUCTION 
    # NOTE: doc_info ในที่นี้คือ Dict ตามที่ถูกกำหนดใน ingest.py
    from core.ingest import process_document, list_documents, delete_document_by_uuid, DATA_DIR, SUPPORTED_TYPES, DocInfo
    from core.vectorstore import load_vectorstore, vectorstore_exists, load_all_vectorstores, get_vectorstore_path, VectorStoreManager
    from langchain.chains import RetrievalQA
    from models.llm import llm as llm_instance 
    from core.run_assessment import run_assessment_process 
    from core.evidence_mapping_generator import EvidenceMappingGenerator
except ImportError as e:
    # สำหรับการรันแบบ Standalone ใน Canvas (ไม่มีไฟล์ Core อื่นๆ)
    print(f"Warning: Core module import failed. Using Mock/Local definitions. Error: {e}")
    
    # Mock definitions for DocInfo and required functions if core files are missing
    class DocInfo(BaseModel):
        filename: str = Field(..., description="ชื่อไฟล์เดิม")
        filepath: str = Field(..., description="Path ของไฟล์บนระบบ (ใช้สำหรับหา upload_date)")
        doc_type: str = Field(..., description="ประเภทเอกสาร (e.g., 'document', 'policy')")
        chunk_count: int = Field(0, description="จำนวน chunk ที่ถูก ingest แล้ว")
        mock_upload_timestamp: float = Field(0.0)

    def list_documents(doc_types: List[str]) -> Dict[str, DocInfo]:
        current_time = time.time()
        mock_data = {
            "uuid-doc-123": DocInfo(filename="Annual_Report_2024.pdf", filepath="/tmp/annual.pdf", doc_type="document", chunk_count=50, mock_upload_timestamp=current_time - 86400 * 5),
            "uuid-pol-456": DocInfo(filename="HR_Policy_V1.docx", filepath="/tmp/hr_policy.docx", doc_type="policy", chunk_count=0, mock_upload_timestamp=current_time - 86400 * 2),
        }
        if not doc_types:
             return mock_data 
        return {uuid: info for uuid, info in mock_data.items() if info.doc_type in doc_types}
        
    def delete_document_by_uuid(doc_id: str, doc_type: str): pass
    def process_document(**kwargs): return "mock-doc-id"
    def vectorstore_exists(doc_id: str, doc_type: str, base_path: str): return True 
    def load_all_vectorstores(**kwargs): raise NotImplementedError("Mock function: load_all_vectorstores")
    
    DATA_DIR = "data"
    SUPPORTED_TYPES = [".pdf", ".docx", ".txt"]
    VECTORSTORE_DIR = "vectorstore"
    QA_PROMPT = "{context}\n\nQuestion: {question}"
    COMPARE_PROMPT = "Doc Names: {doc_names}\n\n{context}\n\nQuery: {query}"
    SYSTEM_QA_INSTRUCTION = "You are a helpful assistant."


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
REF_DATA_DIR = "ref_data" 

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
        raise ValueError(f"Invalid data_type: {data_type}")
        
    # สมมติว่าไฟล์ JSON อยู่ในโฟลเดอร์หลักของ REF_DATA_DIR
    return os.path.join(REF_DATA_DIR, filename) 

def load_ref_data_file(filepath: str) -> Any:
    """อ่านและโหลดข้อมูล JSON จากไฟล์ที่กำหนด"""
    if not os.path.exists(filepath):
        if any(keyword in filepath for keyword in ['statements', 'mapping', 'rubric']):
            return []
        return {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON from {filepath}")
        raise HTTPException(status_code=500, detail=f"Invalid JSON format in {filepath}")
    except Exception as e:
        logger.error(f"Error loading file {filepath}: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading file: {filepath}")

def save_ref_data_file(filepath: str, data: Any):
    """บันทึกข้อมูล JSON กลับลงในไฟล์"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True) 
    with open(filepath, 'w', encoding='utf-8') as f:
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
    os.makedirs(REF_DATA_DIR, exist_ok=True) 
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
    doc_id: str
    filename: str
    file_type: str
    status: str
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
    
    status: str = "RUNNING" # PENDING, RUNNING, COMPLETED, FAILED

    overall_score: Optional[float] = None
    highest_full_level: Optional[int] = None
    export_path: Optional[str] = None

class RefDataPayload(BaseModel):
    data: Dict | List 
    
# Global List of Assessment Records (in-memory for demo/simple environment)
ASSESSMENT_HISTORY: List[AssessmentRecord] = []

# -----------------------------
# --- Assessment Endpoints ---
# -----------------------------

@app.get("/list-collections/")
async def debug_list_collections():
    """Returns a list of collection names that the server can see in the vectorstore directory."""
    try:
        # Initialize VectorStoreManager 
        manager = VectorStoreManager() 
        # ฟังก์ชันนี้จะเรียกใช้ list_vectorstore_folders() และแสดงชื่อโฟลเดอร์ทั้งหมดใน vectorstore
        collections = manager.get_all_collection_names() 
        return {"available_collections": collections, "status": "Success", "vectorstore_dir": manager._base_path}
    except Exception as e:
        return {"available_collections": [], "status": f"Error: Failed to initialize VectorStoreManager or access path: {str(e)}"}

@app.post("/api/assess")
async def run_assessment_task(request: AssessmentRequest, background_tasks: BackgroundTasks):
    record_id = os.urandom(8).hex()
    
    initial_record = AssessmentRecord(
        record_id=record_id,
        enabler=request.enabler.upper(),
        sub_criteria_id=request.sub_criteria_id,
        mode=request.mode,
        timestamp=datetime.now(timezone.utc).isoformat(),
        status="RUNNING" 
    )
    ASSESSMENT_HISTORY.append(initial_record)
    
    background_tasks.add_task(_background_assessment_runner, record_id, request)
    
    return {"status": "accepted", "record_id": record_id, "message": "Assessment started in background. Check /api/assess/history for status."}

# -----------------------------
# --- Assessment History Endpoint ---
# -----------------------------
@app.get("/api/assess/history", response_model=List[AssessmentRecord])
async def get_assessment_history(enabler: Optional[str] = None): 
    
    filtered_history = ASSESSMENT_HISTORY
    
    if enabler:
        enabler_upper = enabler.upper()
        
        filtered_history = [
            record for record in ASSESSMENT_HISTORY 
            if record.enabler.upper() == enabler_upper
        ]
        
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
# --- Reference Data Endpoints ---
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
        await run_in_threadpool(lambda: load_data_safe('statements'))
        await run_in_threadpool(lambda: load_data_safe('rubrics'))
        await run_in_threadpool(lambda: load_data_safe('mapping'))
        await run_in_threadpool(lambda: load_data_safe('weighting'))
        
        data['enabler'] = enabler.upper()

        return data
    except HTTPException:
        raise 
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
    
    if data_type not in ['statements', 'rubrics', 'mapping', 'weighting']:
        raise HTTPException(status_code=400, detail="Invalid data_type. Must be one of: statements, rubrics, mapping, weighting.")
        
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
        final_summary = run_assessment_process(
            enabler=request.enabler,
            sub_criteria_id=request.sub_criteria_id,
            mode=request.mode,
            filter_mode=request.filter_mode,
            export=True 
        )
        
        record.overall_score = final_summary['Overall']['overall_maturity_score']
        
        sub_id_for_level = request.sub_criteria_id if request.sub_criteria_id != 'all' else list(final_summary['SubCriteria_Breakdown'].keys())[0] if final_summary['SubCriteria_Breakdown'] else None
        record.highest_full_level = final_summary['SubCriteria_Breakdown'].get(sub_id_for_level, {}).get('highest_full_level', 0) if sub_id_for_level else 0
        
        record.export_path = final_summary.get("export_path_used")
        record.status = "COMPLETED" 
        
        record.timestamp = datetime.now(timezone.utc).isoformat()
        
        logger.info(f"Assessment {record_id} completed successfully. Score: {record.overall_score:.2f}")

    except Exception as e:
        logger.error(f"Assessment task {record_id} failed: {e}")
        record.overall_score = -1.0
        record.highest_full_level = -1
        record.status = "FAILED"
        record.timestamp = datetime.now(timezone.utc).isoformat()

# -----------------------------
# --- Auto Mapping Background Runner ---
# -----------------------------
def _background_auto_mapper(enabler: str):
    logger.info(f"Starting Auto Mapping for {enabler}...")
    
    try:
        generator = EvidenceMappingGenerator(enabler_id=enabler.upper())
        new_mapping_data = generator.generate_full_mapping_data() 
        
        filepath = get_ref_data_path(enabler, 'mapping')
        save_ref_data_file(filepath, new_mapping_data) 
        
        logger.info(f"Auto Mapping for {enabler} completed and saved successfully.")

    except Exception as e:
        logger.error(f"Auto Mapping task for {enabler} failed: {e}")
        
        
# -----------------------------
# --- Uploads & Document Endpoints (Using Bracket Notation for DocInfo) ---
# -----------------------------
@app.get("/api/uploads/document", response_model=List[UploadResponse])
async def list_uploads_document_only():
    """
    Endpoint เฉพาะสำหรับ GET /api/uploads/document 
    """
    return await list_uploads_by_type("document") 


@app.get("/api/uploads/{doc_type}", response_model=List[UploadResponse]) 
async def list_uploads_by_type(doc_type: str):
    """
    ดึงรายการเอกสารทั้งหมดใน doc_type ที่กำหนด (ใช้ DocInfo Pydantic Model/Dict)
    """
    
    doc_data: Dict[str, DocInfo] = list_documents(doc_types=[doc_type])
    
    uploads: List[UploadResponse] = []
    
    if not isinstance(doc_data, dict):
        logger.error(
            f"API Error: list_documents for doc_type='{doc_type}' returned {type(doc_data).__name__}. Expected dict."
        )
        return uploads

    for uuid, doc_info in doc_data.items():
        
        if doc_info['doc_type'] != doc_type: # ใช้ Bracket Notation
            continue
            
        try:
            # ใช้ Bracket Notation
            timestamp = doc_info['mock_upload_timestamp'] if 'mock_upload_timestamp' in doc_info else os.path.getmtime(doc_info['file_path']) 
            upload_datetime = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            upload_date_iso = upload_datetime.isoformat()
            
        except Exception as e:
            logger.warning(f"Failed to get modification time for {doc_info.get('file_name')} ({doc_info.get('file_path')}). Error: {e}")
            upload_date_iso = datetime.now(timezone.utc).isoformat()
            
        # ใช้ Bracket Notation
        uploads.append(UploadResponse(
            doc_id=uuid,
            filename=doc_info['file_name'],
            file_type=os.path.splitext(doc_info['file_name'])[1], 
            status="Ingested" if doc_info['chunk_count'] > 0 else "Pending",
            upload_date=upload_date_iso
        ))
        
    uploads.sort(key=lambda x: x.filename)
    
    return uploads

@app.get("/api/documents", response_model=List[UploadResponse])
async def get_documents():
    return await list_all_uploads() 

@app.get("/api/uploads/list", response_model=List[UploadResponse])
async def list_all_uploads():
    """
    แสดงรายการไฟล์ที่ถูกอัปโหลดทั้งหมด โดยไม่จำกัด doc_type (ใช้ DocInfo Pydantic Model/Dict)
    """
    doc_data: Dict[str, DocInfo] = list_documents(doc_types=None)
    
    uploads: List[UploadResponse] = []
    
    if not isinstance(doc_data, dict):
        logger.error(f"API Error: list_documents returned {type(doc_data).__name__}. Expected dict.")
        return uploads

    for uuid, doc_info in doc_data.items():
        try:
            # ใช้ Bracket Notation
            timestamp = doc_info['mock_upload_timestamp'] if 'mock_upload_timestamp' in doc_info else os.path.getmtime(doc_info['file_path']) 
            upload_datetime = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            upload_date_iso = upload_datetime.isoformat()
            
        except Exception as e:
            logger.warning(f"Failed to get modification time for {doc_info.get('file_name', 'Unknown')} ({doc_info.get('file_path', 'Unknown')}). Error: {e}") 
            upload_date_iso = datetime.now(timezone.utc).isoformat()
            
        uploads.append(UploadResponse(
            doc_id=uuid,
            filename=doc_info['file_name'], 
            file_type=os.path.splitext(doc_info['file_name'])[1], 
            status="Ingested" if doc_info['chunk_count'] > 0 else "Pending",
            upload_date=upload_date_iso
        ))
        
    uploads.sort(key=lambda x: x.filename)
    return uploads

@app.delete("/api/documents/{doc_id}")
async def remove_document(doc_id: str, doc_type: str = Query("document", description="Document type collection name")):
    try:
        await run_in_threadpool(lambda: delete_document_by_uuid(doc_id=doc_id, doc_type=doc_type))
        return {"status": "ok", "doc_id": doc_id, "doc_type": doc_type}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# -----------------------------
# --- Upload Endpoints ---
# -----------------------------
@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...), source_name: Optional[str] = Form(None)):
    os.makedirs(DATA_DIR, exist_ok=True)
    folder = os.path.join(DATA_DIR, "document")
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, file.filename)
    
    with open(file_path, "wb") as f:
        f.write(await file.read())
        
    doc_id = await run_in_threadpool(lambda: process_document(file_path=file_path, file_name=file.filename, doc_type="document")) 
    
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
        doc_id = await run_in_threadpool(lambda: process_document(file_path=file_path, file_name=file.filename, doc_type=doc_type))
        
    except Exception as e:
        logger.error(f"Failed to process {file.filename} as {doc_type}: {e}")
        raise HTTPException(status_code=500, detail=f"File processing failed: {e}")

    # 🟢 ใช้ doc_id และ doc_type ในการตรวจสอบ vectorstore
    status = "Ingested" if await run_in_threadpool(lambda: vectorstore_exists(doc_id=doc_id, doc_type=doc_type, base_path=VECTORSTORE_DIR)) else "Pending"

    return UploadResponse(
        status=status,
        doc_id=doc_id,
        filename=file.filename,
        file_type=os.path.splitext(file.filename)[1],
        upload_date=datetime.now(timezone.utc).isoformat()
    )

# -----------------------------
# --- Upload File Deletion, Download (Fixed logic to use DocInfo) ---
# -----------------------------
@app.delete("/upload/{doc_type}/{file_id}")
async def delete_upload(doc_type: str, file_id: str):
    
    # 1. ค้นหาชื่อไฟล์จริงและ path จาก UUID (DocInfo mapping)
    doc_data: Dict[str, DocInfo] = list_documents(doc_types=[doc_type])
    filepath = None
    
    for uuid, info in doc_data.items():
        if uuid == file_id:
            # 🟢 ใช้ Bracket Notation
            filepath = info['file_path'] 
            break
            
    if filepath and os.path.exists(filepath):
        # 2. ลบไฟล์ต้นฉบับ
        await run_in_threadpool(lambda: os.remove(filepath))
        logger.info(f"Original file deleted: {filepath}")
    else:
        logger.warning(f"Original file for doc_id '{file_id}' not found. Proceeding with vectorstore deletion.")
        
    try:
        # 3. ลบ Vector Store และข้อมูล Mapping
        await run_in_threadpool(lambda: delete_document_by_uuid(doc_id=file_id, doc_type=doc_type))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete vectorstore/mapping for {file_id}: {e}")

    return {"status": "deleted", "doc_id": file_id}

@app.get("/upload/{doc_type}/{file_id}")
async def download_upload(doc_type: str, file_id: str):
    
    doc_data: Dict[str, DocInfo] = list_documents(doc_types=[doc_type])
    file_name_to_download = None
    filepath = None
    
    for uuid, info in doc_data.items():
        if uuid == file_id:
            # 🟢 ใช้ Bracket Notation
            file_name_to_download = info['file_name'] 
            filepath = info['file_path'] 
            break
            
    if not file_name_to_download:
        raise HTTPException(status_code=404, detail="Document ID not found in mapping.")
        
    if not filepath or not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found on disk.")
        
    return FileResponse(filepath, filename=file_name_to_download)


# -----------------------------
# --- Ingest Endpoint (FIXED: Added stable_doc_uuid argument) ---
# -----------------------------
class IngestRequest(BaseModel):
    doc_ids: List[str]
    doc_type: Optional[str] = "document"

@app.post("/ingest")
async def ingest_documents(request: IngestRequest):
    """
    ประมวลผลเอกสารที่มีอยู่แล้วตาม doc_ids (Stable UUIDs)
    """
    results = []
    
    folder = os.path.join(DATA_DIR, request.doc_type)
    if not os.path.isdir(folder):
         return {"status": "failed", "error": f"Document type folder not found: {folder}"}

    # 1. 🟢 FIX: ใช้ list_documents เพื่อดึง DocInfo Mapping (UUID -> File Path/Name)
    doc_data: Dict[str, DocInfo] = await run_in_threadpool(lambda: list_documents(doc_types=[request.doc_type]))

    for doc_id in request.doc_ids:
        
        info = doc_data.get(doc_id)
        if not info:
             results.append({"doc_id": doc_id, "result": "failed", "error": f"Document ID '{doc_id}' not found in DocInfo mapping. Was the file uploaded via /upload first?"})
             continue
        
        # 2. 🟢 FIX: ใช้ file_name และ file_path จาก DocInfo (ใช้ Bracket Notation)
        file_name = info['file_name']
        file_path = info['file_path']

        file_extension = os.path.splitext(file_name)[1].lower()
        if file_extension not in SUPPORTED_TYPES:
            results.append({"doc_id": doc_id, "result": "failed", "error": f"Unsupported file type: {file_extension}. Supported types are: {', '.join(SUPPORTED_TYPES)}"})
            continue
        
        if not os.path.exists(file_path):
            results.append({"doc_id": doc_id, "result": "failed", "error": f"File path not found on disk: {file_path}. The file may have been manually deleted."})
            continue

        logger.info(f"Attempting to re-ingest file: {file_path}")

        try:
            # 3. 🔴 FIX: เพิ่ม stable_doc_uuid=doc_id ตามที่ Error ร้องขอ
            await run_in_threadpool(
                process_document,
                file_path=file_path, 
                file_name=file_name, 
                doc_type=request.doc_type, 
                base_path=VECTORSTORE_DIR,
                stable_doc_uuid=doc_id # <--- ARGUMENT ที่ถูกเพิ่มกลับเข้ามา
            )
            
        except Exception as e:
            logger.error(f"Error while processing document {doc_id}: {e}", exc_info=True)
            results.append({"doc_id": doc_id, "result": "failed", "error": str(e)})
            continue
        
        # 4. 🟢 FIX: ห่อหุ้ม vectorstore_exists ด้วย await run_in_threadpool
        if await run_in_threadpool(lambda: vectorstore_exists(doc_id=doc_id, doc_type=request.doc_type, base_path=VECTORSTORE_DIR)):
            results.append({"doc_id": doc_id, "result": "success"})
        else:
            logger.warning(f"Vectorstore not found for {doc_id} after processing.")
            results.append({"doc_id": doc_id, "result": "failed", "error": "Vectorstore not found after processing"})

    return {"status": "completed", "results": results}

# -----------------------------
# --- Query Endpoint (Full Multi Doc/Type Support) ---
# -----------------------------
@app.post("/query")
async def query_endpoint(
    question: str = Form(...),
    doc_ids: Optional[List[str]] = Query(None),
    doc_types: Optional[str] = Form(None)  # รับเป็น comma-separated string
):
    """
    RAG Endpoint:
    - doc_ids: list of Stable IDs
    - doc_types: comma-separated string หรือ single type
    """

    import json
    skipped = []
    output = {
        "question": question,
        "doc_ids": [],
        "doc_types": [],
        "answer": "",
        "skipped": skipped
    }

    # -----------------------------
    # 1️⃣ Parse doc_types
    # -----------------------------
    if doc_types:
        doc_type_list = [dt.strip() for dt in doc_types.split(",") if dt.strip()]
    else:
        doc_type_list = ["document"]
    output['doc_types'] = doc_type_list

    # -----------------------------
    # 2️⃣ Parse doc_ids
    # -----------------------------
    uuid_list = [uid.strip() for uid in doc_ids if uid] if doc_ids else []

    # -----------------------------
    # Helper: format context
    # -----------------------------
    def format_context(docs):
        context_sections = []
        for i, d in enumerate(docs, 1):
            doc_name = d.metadata.get("doc_id", f"Document {i}")
            doc_type = d.metadata.get("doc_type", "N/A")
            context_sections.append(f"[{doc_name} ({doc_type})]\n{d.page_content}")
        return "\n\n".join(context_sections)

    # -----------------------------
    # Helper: LLM call
    # -----------------------------
    def call_llm_safe(messages_list):
        res = llm_instance.invoke(messages_list)
        if isinstance(res, dict) and "result" in res:
            return res["result"]
        elif hasattr(res, "content"):
            return res.content.strip()
        elif isinstance(res, str):
            return res.strip()
        return str(res).strip()

    try:
        if not doc_type_list:
            raise ValueError("Must specify at least one document type for RAG.")

        # -----------------------------
        # Load MultiRetriever
        # -----------------------------
        multi_retriever = await run_in_threadpool(
            load_all_vectorstores,
            doc_types=doc_type_list,
            top_k=5,
            final_k=3
        )

        # -----------------------------
        # Perform Retrieval
        # -----------------------------
        all_docs = await run_in_threadpool(lambda: multi_retriever.invoke(question))

        # -----------------------------
        # Filter by UUID if provided
        # -----------------------------
        if uuid_list:
            docs_for_question = [d for d in all_docs if d.metadata.get("doc_id") in uuid_list]
            skipped = [uid for uid in uuid_list if uid not in [d.metadata.get("doc_id") for d in docs_for_question]]
        else:
            docs_for_question = all_docs

        if not docs_for_question:
            raise ValueError("No relevant content could be retrieved from the selected documents or collections.")

        output['doc_ids'] = list({d.metadata.get("doc_id") for d in docs_for_question if d.metadata.get("doc_id")})
        output['skipped'] = skipped

        # -----------------------------
        # Build context + prompt
        # -----------------------------
        context_text = format_context(docs_for_question)
        human_message_content = QA_PROMPT.format(context=context_text, question=question)
        messages = [
            SystemMessage(content=SYSTEM_QA_INSTRUCTION),
            HumanMessage(content=human_message_content)
        ]

        # -----------------------------
        # Call LLM
        # -----------------------------
        answer_text = await run_in_threadpool(lambda: call_llm_safe(messages))
        output['answer'] = answer_text

    except ValueError as e:
        output['answer'] = f"เกิดข้อผิดพลาดในการโหลดแหล่งข้อมูล (Vector Store): {str(e)}"
        output['error'] = str(e)
    except Exception as e:
        output['answer'] = f"เกิดข้อผิดพลาดที่ไม่คาดคิดในระหว่างการประมวลผล RAG: {str(e)}"
        output['error'] = str(e)

    # -----------------------------
    # Flatten JSON output from LLM if possible
    # -----------------------------
    if answer_text:
        answer_text = answer_text.strip()
        if answer_text.startswith("{") and answer_text.endswith("}"):
            try:
                llm_json = json.loads(answer_text)
                flattened_answer = []

                if 'summary' in llm_json and llm_json['summary']:
                    flattened_answer.append("📌 Summary:\n" + llm_json['summary'])
                if 'details' in llm_json and llm_json['details']:
                    for d in llm_json['details']:
                        flattened_answer.append(f"📄 {d.get('doc_name','')}: {d.get('text','')}")
                if 'comparison' in llm_json and llm_json['comparison']:
                    flattened_answer.append("⚖️ Comparison:")
                    for k,v in llm_json['comparison'].items():
                        flattened_answer.append(f"{k}: {v}")
                if 'search_results' in llm_json and llm_json['search_results']:
                    flattened_answer.append("🔍 Search Results:")
                    for r in llm_json['search_results']:
                        flattened_answer.append(f"{r.get('doc_name','')}: {r.get('text','')}")

                if flattened_answer:
                    output['answer'] = "\n\n".join(flattened_answer)

            except Exception as e:
                if 'error' not in output:
                    output['error'] = f"JSON Parsing Error: {str(e)}"

    return output


# -----------------------------
# --- Compare Endpoint ---
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

    # 1. ตรวจสอบ Vectorstores ที่มีอยู่ (ต้องห่อหุ้มด้วย run_in_threadpool)
    def check_vectorstore_existence(doc_ids_list, doc_type_list):
        valid, skipped_list = [], []
        for dt in doc_type_list:
            base_path = os.path.join(VECTORSTORE_DIR, dt)
            for doc_id in doc_ids_list:
                if vectorstore_exists(doc_id, doc_type=dt, base_path=base_path):
                    valid.append((doc_id, dt))
                else:
                    skipped_list.append(f"{dt}/{doc_id}")
        return valid, skipped_list
        
    valid_docs, skipped = await run_in_threadpool(lambda: check_vectorstore_existence(doc_ids, doc_type_list))

    if not valid_docs:
        raise HTTPException(
            status_code=404,
            detail=f"No valid vectorstores found for docs: {', '.join(doc_ids)} in doc_types: {doc_type_list}"
        )

    # 2. โหลด MultiRetriever (ต้องห่อหุ้มด้วย run_in_threadpool)
    multi_retriever = await run_in_threadpool(
        load_all_vectorstores, 
        doc_ids=doc_ids, 
        doc_types=doc_type_list, 
        top_k=5, 
        final_k=3
    )

    # 3. ฟังก์ชันค้นหาและกรอง (รันใน Threadpool)
    def get_docs_text(query_text):
        docs = multi_retriever.invoke(query_text)
        doc_text_map = {doc1: "", doc2: ""}
        
        for d in docs:
            # 🟢 FIX: ใช้ doc_id จาก metadata
            doc_key = d.metadata.get("doc_id") 

            if doc_key == doc1:
                doc_text_map[doc1] += (d.page_content + "\n")
            elif doc_key == doc2:
                doc_text_map[doc2] += (d.page_content + "\n")
        
        doc1_text = doc_text_map.get(doc1, "[No content found for Doc 1]")
        doc2_text = doc_text_map.get(doc2, "[No content found for Doc 2]")
        
        return doc1_text, doc2_text

    # 4. เรียกใช้ใน Threadpool 
    doc1_text, doc2_text = await run_in_threadpool(lambda: get_docs_text(query))
    context_text = f"Document 1 Content:\n{doc1_text}\n\nDocument 2 Content:\n{doc2_text}"

    # 5. LLM Call
    human_message_content = COMPARE_PROMPT.format(
        context=context_text, 
        query=query, 
        doc_names=f"{doc1} และ {doc2}"
    )

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

    # 6. Call LLM with the new messages list
    delta_answer = await run_in_threadpool(lambda: call_llm_safe(messages))

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
# --- Evidence Mapping Endpoint (Completed) ---
# -----------------------------
@app.post("/map-evidence/")
async def map_evidence(file: UploadFile, enabler_id):
    raise HTTPException(status_code=501, detail="Not Implemented")