# app.py (Full Code - Fixed robustness for missing 'file_path')

from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks, Query, Path
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Tuple, Union
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
from typing import Tuple
from core.retrieval_utils import retrieve_context_by_doc_ids, extract_uuids_from_llm_response, parse_llm_json_response

# --- Core Imports (ต้องมีไฟล์เหล่านี้ในโครงสร้างโปรเจกต์) ---
try:
    from core.rag_prompts import QA_PROMPT, COMPARE_PROMPT, SYSTEM_QA_INSTRUCTION, SYSTEM_COMPARE_INSTRUCTION 
    
    # 🟢 FIX: นำเข้า SUPPORTED_DOC_TYPES, DocInfo, และอื่นๆ จาก core.ingest
    from core.ingest import (
        process_document, 
        list_documents, 
        delete_document_by_uuid, 
        DATA_DIR, 
        SUPPORTED_TYPES, 
        DocInfo, 
        SUPPORTED_ENABLERS,
        SUPPORTED_DOC_TYPES # <--- นำเข้าจาก ingest อย่างถูกต้อง
    )
    
    from core.vectorstore import (
        vectorstore_exists, 
        get_vectorstore_path, 
        VectorStoreManager, 
        MultiDocRetriever, 
        NamedRetriever, 
        _get_collection_name,
        INITIAL_TOP_K, 
        FINAL_K_RERANKED
    )
    
    from langchain.chains import RetrievalQA
    from models.llm import llm as llm_instance 
    from core.run_assessment import run_assessment_process 
    from core.evidence_mapping_generator import EvidenceMappingGenerator
    
except ImportError as e:
    # 🔴 หากเกิด ImportError แสดงว่าขาดไฟล์อื่นนอกเหนือจากที่แก้ไขแล้ว
    print(f"❌ FATAL ERROR: Core module import failed. Missing file: {e}")
    class TempImportError(Exception): pass 
    raise TempImportError(f"CRITICAL: Missing core files for non-mock operation. Check your imports and project structure. Error: {e}") 


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

# (Add this near other Pydantic model definitions in app.py)
from langchain_core.pydantic_v1 import BaseModel as LangchainBaseModel # Import Langchain Pydantic v1
from langchain_core.output_parsers import JsonOutputParser

class LLMComparisonMetric(LangchainBaseModel):
    """Schema for a single comparison point returned by the LLM."""
    metric: str = Field(..., description="The key metric or area being compared (e.g., 'Required Documents', 'Fee Structure', 'Eligibility')")
    doc1: str = Field(..., description="The value or description of this metric in Document 1")
    doc2: str = Field(..., description="The value or description of this metric in Document 2")
    delta: float = Field(..., description="The quantitative change between doc1 and doc2 (e.g., 5.0, -2.5, 0). Use 0 if not quantitative.")
    remark: Optional[str] = Field(None, description="Any additional brief remark or explanation.")
    
# Update CompareRequest to include 'query'
class CompareRequest(BaseModel):
    # ✅ FIX: ตั้งค่า Default เป็น None อย่างเดียว เพื่อให้ Pydantic ยอมรับ null/omitted field
    doc1_uuid: Optional[str] = None
    doc2_uuid: Optional[str] = None
    
    # ✅ FIX: ตั้งค่า Default เป็น None
    doc_type_list: Optional[List[str]] = None
    
    # ✅ FIX: ตั้งค่า Default เป็น None
    query: Optional[str] = None
# -----------------------------
# --- Helper Functions (สำหรับจัดการ JSON Files) ---
# -----------------------------


def get_ref_data_path(enabler: str, data_type: str) -> str:
    """สร้าง path เต็มของไฟล์ JSON สำหรับ Enabler และ Data Type ที่ระบุ"""
    enabler = enabler.lower()
    
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
    doc_id: str = Field(..., description="Unique ID of the ingested document.") # Changed from uuid to doc_id
    status: str = Field(..., description="Status of the upload and processing.")
    filename: str = Field(..., description="Original name of the uploaded file.")
    doc_type: str = Field(..., description="Document type.")
    upload_date: Optional[str] = Field(None, description="ISO formatted upload date.") # Added for list endpoint consistency
    message: Optional[str] = Field(None, description="Optional message/error.")
    
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
    
    status: str = "RUNNING" 

    overall_score: Optional[float] = None
    highest_full_level: Optional[int] = None
    export_path: Optional[str] = None

class RefDataPayload(BaseModel):
    data: Dict | List 

class QueryResponse(BaseModel):
    answer: str
    conversation_id: Optional[str] = None
    
# Global List of Assessment Records (in-memory for demo/simple environment)
ASSESSMENT_HISTORY: List[AssessmentRecord] = []

# -----------------------------
# --- Helper: Setup MultiDocRetriever ---
# -----------------------------
def _setup_multi_retriever(doc_type_list: List[str], enabler: Optional[str] = None, filter_doc_ids: Optional[List[str]] = None) -> MultiDocRetriever:
    """
    สร้างและตั้งค่า MultiDocRetriever โดยรองรับ Enabler และ Doc Type หลายชนิด
    """
    # VectorStoreManager ถูกใช้สำหรับ path/config แต่ Retriver ถูกสร้างใหม่เสมอ
    manager = VectorStoreManager()
    retrievers_list: List[NamedRetriever] = []
    
    for doc_type in doc_type_list:
        doc_type_lower = doc_type.lower()
        
        # Logic สำหรับ Evidence (ต้องใช้ Enabler)
        if doc_type_lower == "evidence":
            if not enabler:
                logger.warning("⚠️ Skipping 'evidence': No enabler specified.")
                continue
            collection_name = _get_collection_name(doc_type_lower, enabler)
        
        # Logic สำหรับ Doc Type ทั่วไป
        elif doc_type_lower in SUPPORTED_DOC_TYPES: # ใช้ SUPPORTED_DOC_TYPES จาก core.ingest
            collection_name = _get_collection_name(doc_type_lower, None)
        else:
             logger.warning(f"⚠️ Skipping unsupported doc_type: {doc_type_lower}")
             continue
             
        # ตรวจสอบว่า Collection มีอยู่จริงก่อนเพิ่ม
        # NOTE: การเรียก vectorstore_exists ต้องระบุ enabler สำหรับ evidence
        if doc_type_lower == "evidence":
             exists = vectorstore_exists(doc_id="N/A", doc_type=doc_type_lower, enabler=enabler, base_path=VECTORSTORE_DIR)
        else:
             exists = vectorstore_exists(doc_id="N/A", doc_type=doc_type_lower, enabler=None, base_path=VECTORSTORE_DIR)
             
        if not exists:
             logger.warning(f"⚠️ Vectorstore collection '{collection_name}' not found on disk. Skipping.")
             continue
             
        retrievers_list.append(
            NamedRetriever(
                doc_id=doc_type_lower, 
                doc_type=collection_name, 
                top_k=INITIAL_TOP_K,
                final_k=FINAL_K_RERANKED
            )
        )
        logger.info(f"Adding RAG source: {collection_name}")


    if not retrievers_list:
        raise ValueError("No valid document sources configured for RAG based on input types/enabler.")

    # 2. สร้าง MultiDocRetriever
    multidoc_retriever = MultiDocRetriever(
        retrievers_list=retrievers_list,
        k_per_doc=INITIAL_TOP_K,
        doc_ids_filter=filter_doc_ids 
    )
    return multidoc_retriever


# -----------------------------
# --- Assessment Endpoints ---
# -----------------------------

@app.get("/list-collections/")
async def debug_list_collections():
    """Returns a list of collection names that the server can see in the vectorstore directory."""
    try:
        manager = VectorStoreManager() 
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
        data[data_type] = load_ref_data_file(filepath)

    try:
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


@app.get("/api/documents", response_model=List[UploadResponse])
async def get_documents():
    return await list_all_uploads() 

@app.get("/api/uploads/list", response_model=List[UploadResponse])
async def list_all_uploads():
    """
    แสดงรายการไฟล์ที่ถูกอัปโหลดทั้งหมด โดยไม่จำกัด doc_type (ใช้ DocInfo Dict)
    """
    doc_data: Dict[str, DocInfo] | List[DocInfo] = await run_in_threadpool(lambda: list_documents(doc_types=None))
    
    uploads: List[UploadResponse] = []
    
    if not isinstance(doc_data, dict):
        logger.error(f"API Error: list_documents returned {type(doc_data).__name__}. Expected dict.")
        
        # 🟢 FIX: ถ้าเป็น list ให้แปลงเป็น dict โดยใช้ 'doc_id' เป็น key
        if isinstance(doc_data, list):
            doc_data = {item['doc_id']: item for item in doc_data if isinstance(item, dict) and 'doc_id' in item}
        else:
            return uploads # ถ้าไม่ใช่ทั้ง dict และ list ก็คืนค่าว่าง

    if not isinstance(doc_data, dict):
        # หากการแปลงล้มเหลว หรือได้ค่าว่าง
        return uploads

    for uuid, doc_info in doc_data.items():
        
        # 🟢 FIX: ดึงข้อมูลอย่างปลอดภัยและลองเข้าถึงระบบไฟล์ใน try/except ที่แยกกัน
        file_name = doc_info.get('filename', 'Unknown')
        file_path = doc_info.get('file_path')
        upload_date_iso = datetime.now(timezone.utc).isoformat() # Default

        if file_path:
            try:
                timestamp = os.path.getmtime(file_path) 
                upload_datetime = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                upload_date_iso = upload_datetime.isoformat()
            except Exception as e:
                # Log warning when the actual file is missing or inaccessible
                logger.warning(f"Failed to get modification time for {file_name} ({file_path}). Error: {e}")
        else:
             # Log warning when 'file_path' metadata is missing 
             logger.warning(f"Metadata missing 'file_path' for document {uuid} ({file_name}). Using current timestamp.")
            
        uploads.append(UploadResponse(
            doc_id=uuid,
            # 🟢 FIX: ใช้ file_name ที่ดึงมาอย่างถูกต้อง
            filename=file_name,
            file_type=os.path.splitext(file_name)[1], 
            status="Ingested" if doc_info['chunk_count'] > 0 else "Pending",
            upload_date=upload_date_iso
        ))
        
    uploads.sort(key=lambda x: x.filename)
    return uploads

@app.delete("/api/documents/{doc_id}")
async def remove_document(doc_id: str, doc_type: str = Query("document", description="Document type collection name"), enabler: Optional[str] = Query(None)):
    """ลบเอกสารออกจาก Vectorstore และ Mapping โดยใช้ UUID, doc_type และ enabler (ถ้าเป็น evidence)"""
    try:
        # NOTE: delete_document_by_uuid ใน core/ingest.py ต้องรองรับ stable_doc_uuid, doc_type, และ enabler
        await run_in_threadpool(lambda: delete_document_by_uuid(stable_doc_uuid=doc_id, doc_type=doc_type, enabler=enabler))
        return {"status": "ok", "doc_id": doc_id, "doc_type": doc_type, "enabler": enabler}
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
    
    # ตรวจสอบสถานะการ Ingest (ใช้ doc_id และ doc_type)
    status = "Ingested" if await run_in_threadpool(lambda: vectorstore_exists(doc_id=doc_id, doc_type="document", enabler=None, base_path=VECTORSTORE_DIR)) else "Pending"
    
    return UploadResponse(
        status=status,
        doc_id=doc_id,
        filename=file.filename,
        file_type=os.path.splitext(file.filename)[1],
        upload_date=datetime.now(timezone.utc).isoformat()
    )

@app.post("/upload/{doc_type}", response_model=UploadResponse)
async def upload_document(
    doc_type: str = Path(..., description=f"The type of document to upload. Must be one of: {SUPPORTED_DOC_TYPES}"),
    file: UploadFile = File(..., description="The document file to upload."),
    background_tasks: BackgroundTasks = None,
) -> UploadResponse:
    """
    Uploads a document for ingestion into the RAG system. Processing is done in the background.
    """
    if doc_type not in SUPPORTED_DOC_TYPES:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid doc_type: {doc_type}. Must be one of {SUPPORTED_DOC_TYPES}"
        )

    # 1. Save the file and start process (Mocking the UUID generation in process_document)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    sanitized_filename = "".join(c for c in file.filename if c.isalnum() or c in ('.', '_', '-')).rstrip()
    if not sanitized_filename:
        sanitized_filename = f"uploaded_file_{timestamp}"
    
    temp_file_path = os.path.join(DATA_DIR, f"{timestamp}_{sanitized_filename}")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # A mock doc_id is needed for the immediate response, the actual one will be saved by the background task
    mock_doc_id = f"temp-{timestamp}-{os.urandom(4).hex()}" 

    try:
        contents = await file.read()
        await run_in_threadpool(lambda: open(temp_file_path, "wb").write(contents))
        
        # 2. Add the document processing to background tasks
        # NOTE: The process_document implementation must handle saving the final DocInfo with the correct UUID/Doc_ID.
        # 🛑 FIX: Updated to pass 'file_name' and 'stable_doc_uuid' to match the core/ingest.py signature.
        background_tasks.add_task(
            process_document, 
            file_path=temp_file_path, 
            file_name=file.filename, # Re-introducing file name under the correct parameter name
            stable_doc_uuid=mock_doc_id, # Use mock ID as placeholder for the required 64-char hash
            doc_type=doc_type
        )
        
        # 3. Respond immediately
        return UploadResponse(
            doc_id=mock_doc_id, # Return a mock/temp ID
            status="pending", # Use 'pending' for consistency with background process
            filename=file.filename,
            doc_type=doc_type,
            upload_date=datetime.now(timezone.utc).isoformat(),
            message="Document accepted for background processing."
        )

    except Exception as e:
        logger.error(f"Error during file upload process: {e}")
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Failed to process upload: {e}")

# ----------------------------------------------------
# --- List Documents Endpoint (RESTORED) ---
# ----------------------------------------------------
@app.get("/api/uploads/{doc_type}", response_model=List[UploadResponse]) 
async def list_uploads_by_type(doc_type: str) -> List[UploadResponse]:
    """
    ดึงรายการเอกสารทั้งหมดใน doc_type ที่กำหนด หรือ 'all'
    """
    
    doc_type_to_fetch = [doc_type] if doc_type.lower() != 'all' else SUPPORTED_DOC_TYPES
    
    # Fetch document metadata asynchronously
    doc_data: Union[Dict[str, DocInfo], List[DocInfo]] = await run_in_threadpool(
        lambda: list_documents(doc_types=doc_type_to_fetch)
    )
    
    uploads: List[UploadResponse] = []
    
    doc_map = {}
    
    # 🟢 Handle both dict and list return types from list_documents
    if isinstance(doc_data, dict):
        doc_map = doc_data
    elif isinstance(doc_data, list):
         # Convert list of DocInfo (dicts) to a map keyed by doc_id (the stable UUID)
         doc_map = {item.get('doc_id') or item.get('stable_doc_uuid'): item for item in doc_data if isinstance(item, dict) and ('doc_id' in item or 'stable_doc_uuid' in item)}
    else:
        # Handle unexpected type gracefully (should return empty but log error)
        logger.error(f"API Error: list_documents returned unexpected type: {type(doc_data).__name__}. Returning empty list.")
        return uploads

    requested_doc_type = doc_type.lower()
    
    for doc_id, doc_info in doc_map.items():
        
        actual_doc_type = doc_info.get('doc_type', '').lower()
        
        # กรองเอกสาร
        if requested_doc_type != 'all' and actual_doc_type != requested_doc_type: 
            continue
            
        file_name = doc_info.get('filename', 'Unknown')
        file_path = doc_info.get('file_path')
        upload_date_iso = datetime.now(timezone.utc).isoformat() # Default

        # Determine Upload Date from File Metadata
        if file_path and os.path.exists(file_path):
            try:
                timestamp = os.path.getmtime(file_path) 
                upload_datetime = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                upload_date_iso = upload_datetime.isoformat()
            except Exception as e:
                logger.warning(f"Failed to get modification time for {file_name}. Error: {e}")
        
        # Determine Status
        status = "Pending"
        chunk_count = doc_info.get('chunk_count', 0)
        
        # 🟢 การปรับปรุงตรรกะใหม่: เน้นการมีอยู่ของ Chunk เป็นหลัก
        ingestion_status = doc_info.get('ingestion_status', '').lower()
        
        if ingestion_status == 'failed':
             status = "Failed"
        elif ingestion_status == 'processing':
             status = "Processing"
        elif chunk_count > 0:
            # 💡 ถ้ามี chunks และไม่ได้อยู่ในสถานะ Failed/Processing ให้ถือว่าเป็น Ingested
            status = "Ingested"
        # หาก status_lower เป็นค่าว่าง (เช่น ไม่มีข้อมูล status) และ chunk_count เป็น 0 จะคงค่าเป็น "Pending" ไว้
             
        uploads.append(UploadResponse(
            doc_id=doc_id,
            filename=file_name,
            doc_type=actual_doc_type, 
            status=status,
            upload_date=upload_date_iso,
            message=doc_info.get('error_message') # Include error message if failed
        ))
        
    uploads.sort(key=lambda x: x.filename)
    
    return uploads

# -----------------------------
# --- Upload File Deletion, Download ---
# -----------------------------
@app.delete("/upload/{doc_type}/{file_id}")
async def delete_upload_by_id(doc_type: str, file_id: str):
    """
    ลบไฟล์ที่อัปโหลดและ Chunks ออกจาก Vector Store โดยใช้ ID/UUID
    """
    
    target_doc_type = doc_type
        
    try:
        await run_in_threadpool(lambda: delete_document_by_uuid(doc_id=file_id, doc_type=target_doc_type))
        logger.info(f"Successfully initiated deletion for Doc ID: {file_id}")
        return {"status": f"Document {file_id} deletion initiated."}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found for ID: {file_id}")
    except Exception as e:
        logger.error(f"Error deleting file {file_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {e}")

@app.get("/upload/{doc_type}/{file_id}")
async def download_upload(doc_type: str, file_id: str):
    
    # 📌 NOTE: list_documents ถูกเรียกที่นี่ด้วย และอาจคืนค่าเป็น list
    doc_data: Dict[str, DocInfo] | List[DocInfo] = await run_in_threadpool(lambda: list_documents(doc_types=[doc_type]))
    
    # 🟢 FIX: ต้องแปลงเป็น dict ก่อนหา filepath
    if not isinstance(doc_data, dict) and isinstance(doc_data, list):
         doc_data = {item['doc_id']: item for item in doc_data if isinstance(item, dict) and 'doc_id' in item}
         
    file_name_to_download = None
    filepath = None
    
    # ตรวจสอบว่า doc_data เป็น dict ก่อนเรียก .items()
    if isinstance(doc_data, dict):
        for uuid, info in doc_data.items():
            if uuid == file_id:
                # 🟢 FIX: เปลี่ยน info['file_name'] เป็น info['filename']
                file_name_to_download = info.get('filename') 
                filepath = info.get('file_path') # ใช้ .get() เพื่อความปลอดภัย
                break
            
    if not file_name_to_download:
        raise HTTPException(status_code=404, detail="Document ID not found in mapping.")
        
    if not filepath or not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found on disk.")
        
    return FileResponse(filepath, filename=file_name_to_download)


# -----------------------------
# --- Ingest Endpoint ---
# -----------------------------
class IngestRequest(BaseModel):
    doc_ids: List[str]
    doc_type: Optional[str] = "document"
    enabler: Optional[str] = None # NEW: เพิ่ม enabler

@app.post("/ingest")
async def ingest_documents(request: IngestRequest):
    """
    ประมวลผลเอกสารที่มีอยู่แล้วตาม doc_ids (Stable UUIDs)
    """
    results = []
    
    # Note: Folder check removed as files are stored in DATA_DIR regardless of doc_type sub-folder structure.
    # folder = os.path.join(DATA_DIR, request.doc_type)
    # if not os.path.isdir(folder):
    #      return {"status": "failed", "error": f"Document type folder not found: {folder}"}

    # 📌 NOTE: list_documents ถูกเรียกที่นี่ด้วย และอาจคืนค่าเป็น list
    doc_data: Dict[str, DocInfo] | List[DocInfo] = await run_in_threadpool(lambda: list_documents(doc_types=[request.doc_type]))
    
    # 🟢 FIX: ต้องแปลงเป็น dict ก่อนเข้าถึงด้วย doc_id
    # Ensure doc_data is a dictionary mapping doc_id to DocInfo
    if isinstance(doc_data, list):
         doc_data = {item.get('doc_id') or item.get('stable_doc_uuid'): item for item in doc_data if isinstance(item, dict) and ('doc_id' in item or 'stable_doc_uuid' in item)}
         
    if not isinstance(doc_data, dict):
        return {"status": "failed", "error": "Failed to retrieve document list from DocInfo."}

    for doc_id in request.doc_ids:
        
        info = doc_data.get(doc_id)
        if not info:
             results.append({"doc_id": doc_id, "result": "failed", "error": f"Document ID '{doc_id}' not found in DocInfo mapping. Was the file uploaded via /upload first?"})
             continue
        
        # 🟢 FIX: เปลี่ยน info['file_name'] เป็น info['filename'] (ตาม UploadResponse model)
        file_name = info.get('filename')
        file_path = info.get('file_path')
        
        if not file_name or not file_path:
             results.append({"doc_id": doc_id, "result": "failed", "error": f"Document ID '{doc_id}' metadata missing filename or file_path."})
             continue


        file_extension = os.path.splitext(file_name)[1].lower()
        if file_extension not in SUPPORTED_TYPES:
            results.append({"doc_id": doc_id, "result": "failed", "error": f"Unsupported file type: {file_extension}. Supported types are: {', '.join(SUPPORTED_TYPES)}"})
            continue
        
        if not os.path.exists(file_path):
            results.append({"doc_id": doc_id, "result": "failed", "error": f"File path not found on disk: {file_path}. The file may have been manually deleted."})
            continue

        logger.info(f"Attempting to re-ingest file: {file_path}")

        try:
            # NOTE: process_document ใน core/ingest.py ต้องรองรับ enabler
            await run_in_threadpool(
                process_document,
                file_path=file_path, 
                file_name=file_name, 
                doc_type=request.doc_type, 
                base_path=VECTORSTORE_DIR,
                stable_doc_uuid=doc_id, 
                enabler=request.enabler # <--- ส่ง enabler
            )
            
        except Exception as e:
            logger.error(f"Error while processing document {doc_id}: {e}", exc_info=True)
            results.append({"doc_id": doc_id, "result": "failed", "error": str(e)})
            continue
        
        # Check for vectorstore existence after processing
        if await run_in_threadpool(lambda: vectorstore_exists(doc_id=doc_id, doc_type=request.doc_type, enabler=request.enabler, base_path=VECTORSTORE_DIR)):
            results.append({"doc_id": doc_id, "result": "success"})
        else:
            logger.warning(f"Vectorstore not found for {doc_id} after processing.")
            results.append({"doc_id": doc_id, "result": "failed", "error": "Vectorstore not found after processing"})

    return {"status": "completed", "results": results}



# -----------------------------
# --- Query Endpoint (Full Multi Doc/Type Support) ---
# -----------------------------
class QueryRequest(BaseModel):
    query: str
    doc_ids: List[str]
    
@app.post("/query")
async def query_llm(query: QueryRequest):
    """
    รับ Query จากผู้ใช้และใช้ RAG เพื่อดึงข้อมูลและสร้างคำตอบ
    """
    
    # 1. Input validation
    if not query.query or not query.doc_ids:
        raise HTTPException(status_code=400, detail="Query and document IDs are required.")

    # 2. Check if documents were selected
    selected_doc_ids = query.doc_ids
    if not selected_doc_ids:
        # Handle case where no documents are selected
        return {"result": {"answer": "กรุณาเลือกเอกสารที่ต้องการค้นหาข้อมูลก่อน"}, "sources": []}

    # 3. Retrieve context for the query from the selected documents
    logger.info(f"Retrieving context for query: {query.query} from documents: {selected_doc_ids}")
    
    # 🟢 FIX: ใช้ document_ids เพื่อแก้ไข TypeError จากการเรียก retrieve_context_by_doc_ids
    context_docs = await run_in_threadpool(lambda: retrieve_context_by_doc_ids(
        query=query.query,
        document_ids=selected_doc_ids, 
        top_k=INITIAL_TOP_K 
    ))

    # 4. Format context for LLM
    context_text = "\n\n---\n\n".join([doc.page_content for doc in context_docs])
    
    # 5. Construct the final prompt
    final_prompt = QA_PROMPT.format(context=context_text, question=query.query)
    
    messages = [
        SystemMessage(content=SYSTEM_QA_INSTRUCTION),
        HumanMessage(content=final_prompt)
    ]
    
    # Define LLM call helper (assumes llm_instance is available globally)
    def call_llm_safe(messages_list: List[Any]) -> str:
        """Helper to invoke LLM and return raw content."""
        if 'llm_instance' not in globals():
             # MOCK LLM (Temporary until global llm_instance is properly defined/imported)
             return "Error: LLM model instance is not initialized."
             
        # Add basic try/except for LLM invocation
        try:
            # Assumes llm_instance is an object with an invoke method
            res = llm_instance.invoke(messages_list) 
            if hasattr(res, 'content'): 
                return res.content.strip()
            return str(res).strip()
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            return f"Error: Failed to generate response from the model. Details: {e}"


    # 6. Call LLM to get the answer
    logger.info(f"Calling LLM for query: {query.query}")
    answer_text = await run_in_threadpool(lambda: call_llm_safe(messages))
    
    # 7. Format sources for response
    sources = []
    for doc in context_docs:
        doc_id = doc.metadata.get('doc_id')
        filename = doc.metadata.get('filename')
        page = doc.metadata.get('page')
        
        source_entry = {}
        if doc_id: source_entry['doc_id'] = doc_id
        if filename: source_entry['filename'] = filename
        if page is not None: source_entry['page'] = page
        
        if source_entry:
            sources.append(source_entry)
            
    # Remove duplicate sources (based on the combination of fields)
    unique_sources = list({json.dumps(d, sort_keys=True): d for d in sources}.values())
            

    # 8. Final response structure
    return {
        "result": {
            "answer": answer_text
        },
        "sources": unique_sources
    }

# ----------------------------------------------------
# --- Compare Endpoint ---
# ----------------------------------------------------

@app.post("/compare")
async def compare_documents(
    doc1: str = Form(...),
    doc2: str = Form(...),
    query: str = Form(...),
    doc_types: List[str] = Form(None)
):
    """
    เปรียบเทียบเอกสารสองฉบับตามเกณฑ์ที่กำหนดโดยใช้ LLM
    """
    all_doc_ids = [doc1, doc2]
    final_doc_type_list = doc_types if doc_types else SUPPORTED_DOC_TYPES
    try:
        context_docs = await run_in_threadpool(lambda: retrieve_context_by_doc_ids(
            doc_ids=all_doc_ids, 
            k=99999, 
            doc_types=final_doc_type_list
        ))
    except Exception as e:
        logger.error(f"Retrieval error during comparison: {e}")
        raise HTTPException(status_code=500, detail="RAG retrieval failed during comparison setup.")

    doc1_text = "\n".join([d.page_content for d in context_docs if d.metadata.get('doc_id') == doc1])
    doc2_text = "\n".join([d.page_content for d in context_docs if d.metadata.get('doc_id') == doc2])
    
    skipped = False
    if not doc1_text or not doc2_text:
        skipped = True
        logger.warning(f"Skipping comparison: Missing content for Doc1 ({len(doc1_text)} chars) or Doc2 ({len(doc2_text)} chars).")
        return {
            "result": {"metrics": []}, 
            "skipped": True
        }

    final_query = query if query else "Key differences and similarities."
    
    human_message_content = COMPARE_PROMPT.format(
        doc1_content=doc1_text[:20000], 
        doc2_content=doc2_text[:20000], 
        query=final_query
    )
    
    messages = [
        SystemMessage(content=SYSTEM_COMPARE_INSTRUCTION),
        HumanMessage(content=human_message_content)
    ]
    
    def call_llm_safe(messages_list: List[Any]) -> str:
        """Helper to invoke LLM and return raw content (expected to be JSON string)."""
        res = llm_instance.invoke(messages_list) 
        if hasattr(res, 'content'): 
            return res.content.strip()
        return str(res).strip()

    logger.info(f"Calling LLM for comparison with query: {final_query}")
    json_delta_string = await run_in_threadpool(lambda: call_llm_safe(messages))
    
    # ✅ FIX: เรียกใช้ parse_llm_json_response จาก core.retrieval_utils
    try:
        metrics_data = parse_llm_json_response(json_delta_string, List[LLMComparisonMetric])
    except Exception as e:
        logger.error(f"Failed to parse LLM comparison JSON: {e}. Raw response: {json_delta_string[:500]}...")
        metrics_data = []
        
    # 6. Final response structure
    return {
        "result": {
            "metrics": metrics_data
        },
        "skipped": skipped
    }

class HealthCheckResponse(BaseModel):
    status: str
    message: str
    timestamp: str

@app.get("/health", response_model=HealthCheckResponse)
def health_check():
    return HealthCheckResponse(
        status="ok",
        message="RAG service is operational.",
        timestamp=datetime.now(timezone.utc).isoformat()
    )
