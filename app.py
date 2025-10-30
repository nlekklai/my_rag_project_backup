# app.py (Full Code - Fixed robustness for missing 'file_path')

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
from typing import Tuple

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
    
    status: str = "RUNNING" 

    overall_score: Optional[float] = None
    highest_full_level: Optional[int] = None
    export_path: Optional[str] = None

class RefDataPayload(BaseModel):
    data: Dict | List 
    
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


@app.get("/api/uploads/{doc_type}", response_model=List[UploadResponse]) 
async def list_uploads_by_type(doc_type: str):
    """
    ดึงรายการเอกสารทั้งหมดใน doc_type ที่กำหนด (ใช้ DocInfo Dict)
    """
    
    doc_data: Dict[str, DocInfo] | List[DocInfo] = await run_in_threadpool(lambda: list_documents(doc_types=[doc_type]))
    
    uploads: List[UploadResponse] = []
    
    if not isinstance(doc_data, dict):
        logger.error(
            f"API Error: list_documents for doc_type='{doc_type}' returned {type(doc_data).__name__}. Expected dict."
        )
        # 🟢 FIX: ถ้าเป็น list ให้แปลงเป็น dict โดยใช้ 'doc_id' เป็น key
        if isinstance(doc_data, list):
            doc_data = {item['doc_id']: item for item in doc_data if isinstance(item, dict) and 'doc_id' in item}
        else:
            return uploads # ถ้าไม่ใช่ทั้ง dict และ list ก็คืนค่าว่าง
        
    if not isinstance(doc_data, dict):
        # หากการแปลงล้มเหลว หรือได้ค่าว่าง
        return uploads

    for uuid, doc_info in doc_data.items():
        
        if doc_info['doc_type'] != doc_type: 
            continue
            
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
             # Log warning when 'file_path' metadata is missing (the root cause of the previous error)
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
async def upload_file_type(doc_type: str, file: UploadFile = File(...), enabler: Optional[str] = Form(None)):
    folder = os.path.join(DATA_DIR, doc_type)
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        # NOTE: process_document ใน core/ingest.py ต้องรองรับ enabler 
        doc_id = await run_in_threadpool(lambda: process_document(file_path=file_path, file_name=file.filename, doc_type=doc_type, enabler=enabler))
        
    except Exception as e:
        logger.error(f"Failed to process {file.filename} as {doc_type}: {e}")
        raise HTTPException(status_code=500, detail=f"File processing failed: {e}")

    # ใช้ doc_id และ doc_type/enabler ในการตรวจสอบ vectorstore
    status = "Ingested" if await run_in_threadpool(lambda: vectorstore_exists(doc_id=doc_id, doc_type=doc_type, enabler=enabler, base_path=VECTORSTORE_DIR)) else "Pending"

    return UploadResponse(
        status=status,
        doc_id=doc_id,
        filename=file.filename,
        file_type=os.path.splitext(file.filename)[1],
        upload_date=datetime.now(timezone.utc).isoformat()
    )

# -----------------------------
# --- Upload File Deletion, Download ---
# -----------------------------
@app.delete("/upload/{doc_type}/{file_id}")
async def delete_upload(doc_type: str, file_id: str, enabler: Optional[str] = Query(None)):
    
    # 📌 NOTE: list_documents ถูกเรียกที่นี่ด้วย และอาจคืนค่าเป็น list
    doc_data: Dict[str, DocInfo] | List[DocInfo] = await run_in_threadpool(lambda: list_documents(doc_types=[doc_type]))
    
    # 🟢 FIX: ต้องแปลงเป็น dict ก่อนหา filepath
    if not isinstance(doc_data, dict) and isinstance(doc_data, list):
         doc_data = {item['doc_id']: item for item in doc_data if isinstance(item, dict) and 'doc_id' in item}
    
    filepath = None
    
    # ตรวจสอบว่า doc_data เป็น dict ก่อนเรียก .items()
    if isinstance(doc_data, dict):
        for uuid, info in doc_data.items():
            if uuid == file_id:
                filepath = info.get('file_path') # ใช้ .get() เพื่อความปลอดภัย
                break
    
    if filepath and os.path.exists(filepath):
        await run_in_threadpool(lambda: os.remove(filepath))
        logger.info(f"Original file deleted: {filepath}")
    else:
        logger.warning(f"Original file for doc_id '{file_id}' not found. Proceeding with vectorstore deletion.")
        
    try:
        # NOTE: delete_document_by_uuid ใน core/ingest.py ต้องรองรับ enabler
        await run_in_threadpool(lambda: delete_document_by_uuid(stable_doc_uuid=file_id, doc_type=doc_type, enabler=enabler))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete vectorstore/mapping for {file_id}: {e}")

    return {"status": "deleted", "doc_id": file_id}

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
    
    folder = os.path.join(DATA_DIR, request.doc_type)
    if not os.path.isdir(folder):
         return {"status": "failed", "error": f"Document type folder not found: {folder}"}

    # 📌 NOTE: list_documents ถูกเรียกที่นี่ด้วย และอาจคืนค่าเป็น list
    doc_data: Dict[str, DocInfo] | List[DocInfo] = await run_in_threadpool(lambda: list_documents(doc_types=[request.doc_type]))
    
    # 🟢 FIX: ต้องแปลงเป็น dict ก่อนเข้าถึงด้วย doc_id
    if not isinstance(doc_data, dict) and isinstance(doc_data, list):
         doc_data = {item['doc_id']: item for item in doc_data if isinstance(item, dict) and 'doc_id' in item}
         
    if not isinstance(doc_data, dict):
        return {"status": "failed", "error": "Failed to retrieve document list from DocInfo."}

    for doc_id in request.doc_ids:
        
        info = doc_data.get(doc_id)
        if not info:
             results.append({"doc_id": doc_id, "result": "failed", "error": f"Document ID '{doc_id}' not found in DocInfo mapping. Was the file uploaded via /upload first?"})
             continue
        
        # 🟢 FIX: เปลี่ยน info['file_name'] เป็น info['filename']
        file_name = info['filename']
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
        
        if await run_in_threadpool(lambda: vectorstore_exists(doc_id=doc_id, doc_type=request.doc_type, enabler=request.enabler, base_path=VECTORSTORE_DIR)):
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
    doc_types: Optional[str] = Form(None), 
    enabler: Optional[str] = Form(None) # NEW: รับ enabler
):
    
    import json
    skipped = []
    output = {
        "question": question,
        "doc_ids": [],
        "doc_types": [],
        "answer": "",
        "skipped": skipped,
        "enabler": enabler.upper() if enabler else None
    }

    # 1️⃣ Parse doc_types
    if doc_types:
        doc_type_list = [dt.strip() for dt in doc_types.split(",") if dt.strip()]
    else:
        doc_type_list = ["document", "evidence"]
    output['doc_types'] = doc_type_list

    # 2️⃣ Parse doc_ids
    uuid_list = [uid.strip() for uid in doc_ids if uid] if doc_ids else None

    # Helper: format context
    def format_context(docs):
        context_sections = []
        for i, d in enumerate(docs, 1):
            doc_name = d.metadata.get("doc_id", f"Document {i}")
            doc_type = d.metadata.get("doc_type", "N/A")
            # ใช้ metadata.get("retriever_source") ซึ่งคือ Collection Name 
            source = d.metadata.get("retriever_source", doc_type) 
            context_sections.append(f"[{doc_name} ({source})]\n{d.page_content}")
        return "\n\n".join(context_sections)

    # Helper: LLM call
    def call_llm_safe(messages_list: List[Any]) -> str:
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
        # Load MultiRetriever (ใช้ Helper ใหม่)
        # -----------------------------
        multi_retriever = await run_in_threadpool(
            _setup_multi_retriever,
            doc_type_list=doc_type_list,
            enabler=enabler,
            filter_doc_ids=uuid_list 
        )

        # -----------------------------
        # Perform Retrieval
        # -----------------------------
        all_docs = await run_in_threadpool(lambda: multi_retriever.invoke(question))

        if not all_docs:
            raise ValueError("No relevant content could be retrieved from the selected documents or collections.")

        docs_for_question = all_docs 

        output['doc_ids'] = list({d.metadata.get("doc_id") for d in docs_for_question if d.metadata.get("doc_id")})

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
        output['answer'] = f"เกิดข้อผิดพลาดในการโหลดแหล่งข้อมูล: {str(e)}"
        output['error'] = str(e)
    except Exception as e:
        output['answer'] = f"เกิดข้อผิดพลาดที่ไม่คาดคิดในระหว่างการประมวลผล RAG: {str(e)}"
        output['error'] = str(e)

    # Flatten JSON output from LLM if possible (เหมือนเดิม)
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
                        # NOTE: doc_name ใน LLM output มักจะเป็น UUID หรือชื่อไฟล์
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

# -----------------------------------------------------------------------------
# 📌 API Endpoint
# -----------------------------------------------------------------------------
@app.post("/compare")
async def compare_two_documents(request: CompareRequest):
    
    # 📌 การแก้ไข: ตรวจสอบ UUIDs ที่จำเป็นด้วยตัวเอง หลัง Pydantic ผ่าน
    if not request.doc1_uuid or not request.doc2_uuid:
        logger.error(f"Comparison failed: Missing required UUIDs. Doc1: {request.doc1_uuid}, Doc2: {request.doc2_uuid}")
        raise HTTPException(status_code=400, detail="Both doc1_uuid and doc2_uuid must be selected and provided.")

    doc1_uuid = request.doc1_uuid
    doc2_uuid = request.doc2_uuid
    
    # ✅ การแก้ไข: กำหนดค่า final_doc_type_list โดยใช้ค่า Default หากเป็น None หรือ Empty List
    final_doc_type_list = request.doc_type_list if request.doc_type_list and len(request.doc_type_list) > 0 else ['document']
    
    # ✅ การแก้ไข: กำหนดค่า final_query โดยใช้ค่า Default หาก request.query เป็น None
    default_query = "เปรียบเทียบความแตกต่างและความเหมือนของเอกสารทั้งสองฉบับนี้"
    final_query = request.query if request.query else default_query
    
    # 1. Load VectorStoreManager
    try:
        # NOTE: manager ต้องถูก initialize จาก doc_type_list ที่กรองแล้ว
        # 💡 ปรับปรุงการเรียก: ใช้ doc_type_list เป็นชื่อ parameter ที่ชัดเจน
        manager = VectorStoreManager(doc_type_list=final_doc_type_list) 
    except Exception as e:
        logger.error(f"Failed to initialize VectorStoreManager for comparison: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize RAG system: {e}")

    # 2. Retrieve Documents by UUID
    
    # 2.1 Fetch document 1
    doc1_info = await run_in_threadpool(lambda: manager.get_doc_info_by_uuid(doc1_uuid))
    if not doc1_info:
        raise HTTPException(status_code=404, detail=f"Document 1 with UUID {doc1_uuid} not found.")
    doc1_chunks, doc1_text, skipped1 = await run_in_threadpool(
        lambda: manager.retrieve_all_text_for_uuid(doc1_uuid, doc1_info['doc_type'], doc1_info.get('enabler'))
    )
    doc1_name = doc1_info['filename']

    # 2.2 Fetch document 2
    doc2_info = await run_in_threadpool(lambda: manager.get_doc_info_by_uuid(doc2_uuid))
    if not doc2_info:
        raise HTTPException(status_code=404, detail=f"Document 2 with UUID {doc2_uuid} not found.")
    doc2_chunks, doc2_text, skipped2 = await run_in_threadpool(
        lambda: manager.retrieve_all_text_for_uuid(doc2_uuid, doc2_info['doc_type'], doc2_info.get('enabler'))
    )
    doc2_name = doc2_info['filename']
    
    skipped = skipped1 + skipped2

    # 3. Combine Text and setup structured output
    doc_names_formatted = f"{doc1_name} และ {doc2_name}"
    context_text = f"--- Document 1: {doc1_name} ---\n{doc1_text}\n\n--- Document 2: {doc2_name} ---\n{doc2_text}"
    
    # Use JsonOutputParser to get the format instructions for the LLM
    # NOTE: List[LLMComparisonMetric] ต้องถูก Import และเป็น Pydantic Model ที่ถูกต้อง
    parser = JsonOutputParser(pydantic_object=List[LLMComparisonMetric]) 
    json_format_instruction = parser.get_format_instructions()
    
    # 4. Format Prompt (COMPARE_PROMPT must accept {json_format_instruction})
    human_message_content = COMPARE_PROMPT.format(
        context=context_text, 
        query=final_query, # ใช้งาน final_query ที่ถูกตรวจสอบแล้ว
        doc_names=doc_names_formatted,
        json_format_instruction=json_format_instruction # Pass JSON format instruction
    )

    # 5. Setup Messages
    messages = [
        SystemMessage(content=SYSTEM_COMPARE_INSTRUCTION), # ✅ แก้ไข: ใช้ SYSTEM_COMPARE_INSTRUCTION ที่เราปรับแล้ว
        HumanMessage(content=human_message_content)
    ]
    
    def call_llm_safe(messages_list: List[Any]) -> str:
        """Helper to invoke LLM and return raw content (expected to be JSON string)."""
        res = llm_instance.invoke(messages_list) 
        if hasattr(res, 'content'): 
            return res.content.strip()
        return str(res).strip()

    # 6. Call LLM to get the structured JSON string
    logger.info(f"Calling LLM for comparison with query: {final_query}")
    json_delta_string = await run_in_threadpool(lambda: call_llm_safe(messages))

    # 7. Final response structure
    return {
        "result": {
            "metrics": [
                {
                    "metric": final_query, # ใช้งาน final_query ที่ถูกตรวจสอบแล้ว
                    "doc1": doc1_text, 
                    "doc2": doc2_text, 
                    "delta": json_delta_string, # JSON string of List[LLMComparisonMetric]
                    "remark": f"Comparison generated from doc_types: {', '.join(final_doc_type_list)}. LLM Query: {final_query}"
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