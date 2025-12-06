# routers/assessment_router.py

import os
import uuid
import logging
import json
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Path, Depends, status
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field

# ------------------- Core & LLM -------------------
from core.seam_assessment import SEAMPDCAEngine, AssessmentConfig
from models.llm import create_llm_instance

# Import Global Variables ที่จำเป็นทั้งหมดสำหรับ Pre-Check และ Path
from config.global_vars import (
    LLM_MODEL_NAME, DATA_DIR, EVIDENCE_DOC_TYPES,
    MAPPING_BASE_DIR, DOCUMENT_ID_MAPPING_FILENAME_SUFFIX 
)
from routers.auth_router import UserMe, get_current_user 

# NOTE: ต้อง Import logic สำหรับ VectorStore และ DocStore (ถูกยกเว้นไว้)
# from core.vectorstore import get_evidence_content_by_id 

logger = logging.getLogger(__name__)

assessment_router = APIRouter(prefix="/api/assess", tags=["Assessment"])

# ------------------- Pydantic Models -------------------
class StartAssessmentRequest(BaseModel):
    enabler: str = Field(..., example="KM")
    # 💥 แก้ไข 1: เปลี่ยนชื่อเพื่อให้ตรงกับ Frontend payload (sub_criteria)
    sub_criteria: Optional[str] = Field(None, example="1.2") 
    # 💥 แก้ไข 2: เปลี่ยนชื่อเพื่อให้ตรงกับ Frontend payload (sequential_mode)
    sequential_mode: bool = Field(True, description="แนะนำเปิด") 
    
    tenant: str = Field(..., example="pea", description="รหัสองค์กร")
    year: int = Field(..., example=2568, description="ปีงบประมาณ")

class AssessmentStatus(BaseModel):
    record_id: str
    enabler: str
    sub_criteria_id: str # NOTE: Field นี้ยังคงใช้ 'sub_criteria_id' เพื่อความสอดคล้องของผลลัพธ์
    sequential: bool # NOTE: Field นี้ยังคงใช้ 'sequential' เพื่อความสอดคล้องของผลลัพธ์
    status: str
    started_at: str
    tenant: str 
    year: int 
    finished_at: Optional[str] = None
    overall_score: Optional[float] = None
    highest_level: Optional[int] = None
    export_path: Optional[str] = None
    message: str = "Assessment in progress..."

# ------------------- In-memory Store -------------------
# NOTE: ใน Production ควรใช้ Database เช่น PostgreSQL/MongoDB
ASSESSMENT_RECORDS: Dict[str, AssessmentStatus] = {}

# ------------------- Helper Functions for Data Extraction -------------------
def _load_assessment_data(record_id: str, current_user: UserMe) -> Dict[str, Any]:
    """Handles record validation, tenant isolation, and loads the full JSON file."""
    record = ASSESSMENT_RECORDS.get(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
        
    # Tenant Isolation Check
    if record.tenant.lower() != current_user.tenant.lower() or record.year != current_user.year:
        raise HTTPException(status_code=403, detail="Access denied to this assessment record.")

    if record.status != "COMPLETED":
        raise HTTPException(status_code=425, detail=f"Result not ready yet. Status: {record.status}")
    if not record.export_path or not os.path.exists(record.export_path):
        raise HTTPException(status_code=404, detail="Result file not found or path is invalid.")

    try:
        with open(record.export_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON from {record.export_path}")
        raise HTTPException(status_code=500, detail="Error reading assessment result file.")

def _get_summary_data(full_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts LIGHT payload by stripping large fields from sub_criteria_results."""
    summary = full_data.get("summary", {})
    sub_criteria_results_lite = []
    
    # Define fields to be stripped from sub_criteria_results
    FIELDS_TO_EXCLUDE = [
        "raw_results_ref", "llm_result_full", "top_evidences_ref", 
        "full_context_meta", "temp_map_for_level",
    ]

    for sub_result in full_data.get("sub_criteria_results", []):
        # สร้าง Dict ใหม่โดยเลือกเฉพาะ Field ที่ต้องการแสดงผลสรุป
        lite_result = {k: v for k, v in sub_result.items() if k not in FIELDS_TO_EXCLUDE}
        sub_criteria_results_lite.append(lite_result)
        
    return {
        "summary": summary,
        "sub_criteria_results": sub_criteria_results_lite
    }

def _get_sub_criteria_detail(full_data: Dict[str, Any], sub_criteria_id: str) -> Dict[str, Any]:
    """Extracts FULL detail for a specific sub-criteria, including raw_results_ref."""
    for sub_result in full_data.get("sub_criteria_results", []):
        if sub_result.get("sub_criteria_id") == sub_criteria_id:
            # Return the full sub_criteria result including 'raw_results_ref'
            return sub_result
    raise HTTPException(status_code=404, detail=f"Sub-criteria ID '{sub_criteria_id}' not found in results.")

def _get_evidence_content(record: AssessmentStatus, evidence_ref_id: str) -> Dict[str, Any]:
    """
    NOTE: ฟังก์ชันนี้ต้องติดต่อกับ Vector Store เพื่อดึงเนื้อหา Chunk/Document ตาม ID ที่ได้รับ
    """
    # 🚨 ข้อความแจ้งเตือนสำหรับทีม Dev
    raise HTTPException(
        status_code=501, 
        detail=(
            f"Endpoint Not Implemented Yet (501): การดึงเนื้อหาหลักฐาน ID '{evidence_ref_id}' "
            f"ต้องเรียกใช้ Vector Store (Chroma/etc.) โดยใช้ Context (Tenant:{record.tenant}, Year:{record.year}, Enabler:{record.enabler}) "
            f"เพื่อดึง Chunk Content ที่แท้จริง"
        )
    )

# เพิ่มพารามิเตอร์ enabler และใช้ Global Variable ที่ถูกต้อง
def _get_document_file_path(document_id: str, current_user: UserMe, enabler: str) -> str:
    """
    NOTE: ฟังก์ชันนี้ต้องแปลง document_id (จาก mapping) ไปเป็น path ของไฟล์จริง
    (ในโลกความเป็นจริงอาจต้องเรียก S3/Google Drive API)
    Path Structure: DATA_DIR / tenant / year / evidence / enabler / document_id
    """
    
    # ใช้ DATA_DIR + โครงสร้างจริงตามที่ Ingest ใช้
    BASE_DOCUMENT_STORE = os.path.join(
        DATA_DIR, 
        current_user.tenant.lower(), 
        str(current_user.year),
        EVIDENCE_DOC_TYPES.lower(), # 'evidence'
        enabler.lower()             # 'km', 'cg', etc.
    )
    
    # สมมติว่า document_id คือชื่อไฟล์จริง (เช่น 'Policy-QMS-2024.pdf')
    file_path = os.path.join(BASE_DOCUMENT_STORE, document_id) 

    if not os.path.exists(file_path):
         # 🚨 ข้อความแจ้งเตือนสำหรับทีม Dev
         raise HTTPException(
            status_code=501, 
            detail=(
                f"Endpoint Not Implemented Yet (501): การดึงไฟล์ต้นฉบับ ID '{document_id}' "
                f"ต้องเชื่อมต่อกับ Document Storage (Local/S3/Drive) โดยอ้างอิงจาก Doc ID Mapping"
            )
        )
    
    # หากมีการ Implement และพบไฟล์จริง:
    # return file_path 
    
    # ปัจจุบันแจ้ง 501 เพราะยังไม่มีการเชื่อมต่อกับ Document Storage จริง
    raise HTTPException(
        status_code=501, 
        detail=(
            f"Endpoint Not Implemented Yet (501): การดึงไฟล์ต้นฉบับ ID '{document_id}' "
            f"ต้องเชื่อมต่อกับ Document Storage (Local/S3/Drive) โดยอ้างอิงจาก Doc ID Mapping"
        )
    )

# ------------------- Pre-Check Helper -------------------
def _check_ingestion_status(tenant: str, year: int, enabler: str):
    """
    ตรวจสอบว่ามีไฟล์ Doc ID Mapping อยู่หรือไม่
    (บ่งชี้ว่าได้ Ingest ข้อมูลหลักฐานเสร็จสมบูรณ์แล้ว)
    """
    mapping_filename = f"{tenant.lower()}_{year}_{enabler.lower()}{DOCUMENT_ID_MAPPING_FILENAME_SUFFIX}"
    
    # โครงสร้าง Path: MAPPING_BASE_DIR / tenant / year / filename
    doc_id_mapping_path = os.path.join(
        MAPPING_BASE_DIR, 
        tenant.lower(), 
        str(year), 
        mapping_filename
    )
    
    if not os.path.exists(doc_id_mapping_path):
        logger.error(f"Ingestion check failed: Mapping file not found at {doc_id_mapping_path}")
        raise HTTPException(
            status_code=status.HTTP_412_PRECONDITION_FAILED,
            detail=(
                f"🚨 ไม่สามารถเริ่มการประเมินได้: ไม่พบข้อมูลหลักฐานสำหรับ {enabler.upper()} "
                f"ของ {tenant.upper()} ปี {year} ในระบบ "
                f"(ขาดไฟล์ {mapping_filename}). "
                f"โปรดตรวจสอบว่าได้ Ingest ข้อมูลเอกสารต้นฉบับเสร็จสมบูรณ์แล้ว"
            )
        )
# ------------------- END NEW Helper -------------------


# ------------------- Background Runner -------------------
async def _run_assessment_background(record_id: str, request: StartAssessmentRequest):
    record = ASSESSMENT_RECORDS[record_id]
    try:
        logger.info(
            f"Assessment STARTED → {record_id} | {request.enabler} | Tenant/Year: {request.tenant}/{request.year} | Seq: {request.sequential_mode}"
        )

        # สร้าง config
        config = AssessmentConfig(
            enabler=request.enabler.upper(),
            target_level=5,
            mock_mode="none",
            force_sequential=False,
            model_name=LLM_MODEL_NAME,
            temperature=0.0,
            tenant=request.tenant,
            year=request.year      
        )

        # สร้าง engine
        engine = SEAMPDCAEngine(
            config=config,
            llm_instance=create_llm_instance(model_name=LLM_MODEL_NAME, temperature=0.0)
        )

        # 💥 ใช้ชื่อ field ใหม่: request.sub_criteria
        target_id_to_use = (
            request.sub_criteria.strip() 
            if request.sub_criteria and request.sub_criteria.strip()
            else "all"
        )
        
        # 💥 ใช้ชื่อ field ใหม่: request.sequential_mode
        result = engine.run_assessment(
            target_sub_id=target_id_to_use,
            export=True,
            sequential=request.sequential_mode # ส่งค่าไปที่ engine ด้วยชื่อเดิม (sequential)
        )

        # อัปเดต record
        export_path = result.get("export_path_used")
        if not export_path or not os.path.exists(export_path):
            raise Exception("Export file was not created or path is invalid.")

        overall = result.get("Overall", {}) or {}
        record.status = "COMPLETED"
        record.finished_at = datetime.now(timezone.utc).isoformat()
        record.overall_score = overall.get("overall_maturity_score", 0.0)
        record.highest_level = overall.get("overall_maturity_level", 0)
        record.export_path = export_path
        record.message = f"Assessment completed successfully (L{record.highest_level})"
        record.sequential = request.sequential_mode # อัปเดต field sequential ใน record

        logger.info(f"Assessment COMPLETED → {record_id}")

    except Exception as e:
        logger.exception(f"Assessment FAILED → {record_id}")
        record.status = "FAILED"
        record.finished_at = datetime.now(timezone.utc).isoformat()
        record.message = f"Error: {str(e)}"

# ------------------- API Endpoints -------------------
@assessment_router.post("/start", response_model=AssessmentStatus)
async def start_assessment(
    request: StartAssessmentRequest, 
    background_tasks: BackgroundTasks,
    current_user: UserMe = Depends(get_current_user) 
):
    # ⚠️ ตรวจสอบ Tenant/Year ใน Request ต้องตรงกับ User Context
    if request.tenant.lower() != current_user.tenant.lower() or request.year != current_user.year:
         raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot start assessment for another tenant or year."
        )

    llm = create_llm_instance(model_name=LLM_MODEL_NAME, temperature=0.0)
    if not llm:
        raise HTTPException(status_code=503, detail="LLM service unavailable")
    
    # Pre-check for Ingested Data
    _check_ingestion_status(
        tenant=request.tenant,
        year=request.year,
        enabler=request.enabler
    )
    # --------------------------------------------------------------------------

    record_id = uuid.uuid4().hex[:12]
    os.makedirs("exports", exist_ok=True)

    # 💥 ใช้ชื่อ field ใหม่: request.sub_criteria
    sub_id_for_record = (
        request.sub_criteria.strip() 
        if request.sub_criteria and request.sub_criteria.strip()
        else "all"
    )

    record = AssessmentStatus(
        record_id=record_id,
        enabler=request.enabler.upper(),
        sub_criteria_id=sub_id_for_record, # ใช้ชื่อ field เดิมในการบันทึก
        sequential=request.sequential_mode, # 💥 ใช้ชื่อ field ใหม่จาก Request
        tenant=request.tenant,
        year=request.year,
        status="RUNNING",
        started_at=datetime.now(timezone.utc).isoformat(),
        message=f"กำลังวิเคราะห์เอกสารของ {request.tenant} ปี {request.year} ด้วย AI..."
    )
    ASSESSMENT_RECORDS[record_id] = record

    background_tasks.add_task(_run_assessment_background, record_id, request)

    return record

@assessment_router.get("/status/{record_id}", response_model=AssessmentStatus)
async def get_status(
    record_id: str = Path(..., description="Record ID จาก /start"),
    current_user: UserMe = Depends(get_current_user) 
):
    record = ASSESSMENT_RECORDS.get(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
        
    # ⚠️ ตรวจสอบ Tenant Isolation
    if record.tenant.lower() != current_user.tenant.lower() or record.year != current_user.year:
        raise HTTPException(status_code=403, detail="Access denied to this assessment record.")

    return record

# ------------------- OPTIMIZED ENDPOINTS FOR UI -------------------

@assessment_router.get("/results/{record_id}/summary", response_model=Dict[str, Any], summary="1. Get Assessment Summary (Optimized for UI)")
async def get_assessment_summary(
    record_id: str = Path(..., description="Record ID จาก /start"),
    current_user: UserMe = Depends(get_current_user)
):
    """
    ดึงผลลัพธ์การประเมินแบบสรุป (Light Payload) สำหรับโหลดหน้าจอหลัก AssessmentResults.tsx อย่างรวดเร็ว
    """
    full_data = _load_assessment_data(record_id, current_user)
    summary_data = _get_summary_data(full_data)
    return JSONResponse(content=summary_data)


@assessment_router.get("/results/{record_id}/sub_criteria/{sub_criteria_id}/detail", response_model=Dict[str, Any], summary="2. Get Full Detail for a Specific Sub-Criteria")
async def get_sub_criteria_detail(
    record_id: str = Path(..., description="Record ID จาก /start"),
    sub_criteria_id: str = Path(..., description="Sub-Criteria ID (e.g., '1.2')"),
    current_user: UserMe = Depends(get_current_user)
):
    """
    ดึงข้อมูลเชิงลึกทั้งหมดของ Sub-Criteria ที่ระบุ
    """
    full_data = _load_assessment_data(record_id, current_user)
    detail_data = _get_sub_criteria_detail(full_data, sub_criteria_id)
    return JSONResponse(content=detail_data)


@assessment_router.get("/results/{record_id}/evidence/{evidence_ref_id}/content", summary="3. Get Evidence Content (Requires Vector Store)")
async def get_evidence_content(
    record_id: str = Path(..., description="Record ID จาก /start"),
    evidence_ref_id: str = Path(..., description="Unique ID ของ Chunk/Document ที่ใช้เป็นหลักฐาน"),
    current_user: UserMe = Depends(get_current_user)
):
    """
    ดึงเนื้อหาข้อความฉบับเต็มของหลักฐานที่ใช้ในการประเมิน โดยใช้ ID อ้างอิง
    🚨 NOTE: ฟังก์ชันนี้ถูกตั้งค่าให้ส่งข้อความแจ้งเตือน 501 เพราะต้องมีการเรียกใช้ Vector Store จริง
    """
    record = ASSESSMENT_RECORDS.get(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
        
    # Tenant Isolation Check 
    if record.tenant.lower() != current_user.tenant.lower() or record.year != current_user.year:
        raise HTTPException(status_code=403, detail="Access denied to this assessment record.")

    if record.status != "COMPLETED":
        raise HTTPException(status_code=425, detail=f"Result not ready yet. Status: {record.status}")
        
    return JSONResponse(content=_get_evidence_content(record, evidence_ref_id))


# เพิ่ม {enabler} ใน Path เพื่อให้สามารถค้นหาไฟล์ตามโครงสร้างจริงได้
@assessment_router.get("/documents/{enabler}/{document_id}/download", summary="4. Download Original Source Document File")
async def download_original_document(
    enabler: str = Path(..., description="Enabler type (e.g., 'KM')"),
    document_id: str = Path(..., description="Original Document ID (e.g., 'Policy-2024.pdf')"),
    current_user: UserMe = Depends(get_current_user)
):
    """
    ดึงไฟล์เอกสารต้นฉบับ (PDF, DOCX, ฯลฯ) ที่ใช้เป็นหลักฐานในการประเมิน
    🚨 NOTE: ฟังก์ชันนี้ถูกตั้งค่าให้ส่งข้อความแจ้งเตือน 501 เพราะต้องมีการ Implement การดึงไฟล์จาก Storage จริง
    """
    # ส่ง enabler เข้าไปใน Helper Function เพื่อสร้าง Path ที่ถูกต้อง
    _get_document_file_path(document_id, current_user, enabler)


# ------------------- LEGACY ENDPOINTS -------------------

@assessment_router.get("/results/{record_id}", summary="5. Get ALL Assessment Results (Unoptimized Full Payload)")
async def get_results_json(
    record_id: str = Path(..., description="Record ID from /start"),
    current_user: UserMe = Depends(get_current_user) 
):
    """
    🚨 Endpoint นี้คืนค่าผลลัพธ์การประเมินทั้งหมดในรูปแบบ JSON (Unoptimized/Large Payload)
    แนะนำให้ใช้ /summary หรือ /sub_criteria/{sub_criteria_id}/detail แทน
    """
    try:
        data = _load_assessment_data(record_id, current_user)
        return JSONResponse(content=data)
    except HTTPException as e:
        raise e


@assessment_router.get("/download/{record_id}", summary="6. Download Full Assessment Result JSON File")
async def download_result_file(
    record_id: str = Path(...),
    current_user: UserMe = Depends(get_current_user) 
):
    record = ASSESSMENT_RECORDS.get(record_id)
    if not record or record.status != "COMPLETED" or not record.export_path:
        raise HTTPException(status_code=404, detail="Result not ready")
        
    # ⚠️ ตรวจสอบ Tenant Isolation
    if record.tenant.lower() != current_user.tenant.lower() or record.year != current_user.year:
        raise HTTPException(status_code=403, detail="Access denied to this assessment record.")

    return FileResponse(
        path=record.export_path,
        media_type="application/json",
        filename=os.path.basename(record.export_path)
    )

@assessment_router.get("/history", response_model=List[AssessmentStatus])
async def get_assessment_history(
    enabler: Optional[str] = None,
    tenant: Optional[str] = None,
    year: Optional[int] = None,
    current_user: UserMe = Depends(get_current_user) 
):
    items = list(ASSESSMENT_RECORDS.values())
    
    # ⚠️ Tenant Isolation: กรองตาม Tenant/Year ของ User ที่ Login ก่อนเสมอ
    items = [
        i for i in items 
        if i.tenant.lower() == current_user.tenant.lower() and i.year == current_user.year
    ]
    
    # Apply Optional Filters (กรองภายในกลุ่ม Tenant/Year ของตัวเอง)
    if enabler:
        items = [i for i in items if i.enabler == enabler.upper()]
        
    return sorted(items, key=lambda x: x.started_at, reverse=True)