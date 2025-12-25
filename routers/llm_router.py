# -*- coding: utf-8 -*-
# routers/llm_router.py - Enterprise RAG (Query + Compare + PDCA Analysis + Summary)
# ULTIMATE FINAL PRODUCTION VERSION - 22 ธันวาคม 2568
# รองรับ: Multi-year evidence via wrapper, UUID v5, Clean fallback, Full intent routing

import logging
import uuid
import asyncio
from typing import List, Optional, Set, Dict, Any
from collections import defaultdict

from fastapi import APIRouter, Form, HTTPException, Depends
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document as LcDocument

from core.history_utils import async_save_message, get_recent_history
from core.llm_data_utils import retrieve_context_for_endpoint
from core.vectorstore import get_vectorstore_manager
from core.seam_assessment import SEAMPDCAEngine, AssessmentConfig
from core.llm_guardrails import enforce_thai_primary_language, detect_intent
from config.global_vars import (
    EVIDENCE_DOC_TYPES,
    DEFAULT_ENABLER,
    DEFAULT_LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    QUERY_FINAL_K,
    DEFAULT_DOC_TYPES  # เช่น ["document"]
)
from models.llm import create_llm_instance
from routers.auth_router import UserMe, get_current_user
from core.rag_prompts import (
    SYSTEM_QA_INSTRUCTION,
    SYSTEM_ANALYSIS_INSTRUCTION,
    SYSTEM_COMPARE_INSTRUCTION,
    QA_PROMPT_TEMPLATE,
    COMPARE_PROMPT_TEMPLATE,
    ANALYSIS_PROMPT_TEMPLATE,
    SUMMARY_PROMPT_TEMPLATE
)
from utils.path_utils import get_rubric_file_path, get_doc_type_collection_key
import json, os
import time

logger = logging.getLogger(__name__)
llm_router = APIRouter(prefix="/api", tags=["LLM"])

# =====================================================================
# Response Models
# =====================================================================

class QuerySource(BaseModel):
    source_id: str
    file_name: str
    chunk_text: str
    chunk_id: Optional[str] = None
    score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[QuerySource] = Field(default_factory=list)
    conversation_id: str
    result: Optional[Dict[str, Any]] = None


# =====================================================================
# Helper: _map_sources
# =====================================================================
def _map_sources(chunks: List[LcDocument]) -> List[QuerySource]:
    return [
        QuerySource(
            source_id=str(c.metadata.get("doc_id", "unknown")),
            file_name=c.metadata.get("source", "Unknown"),
            chunk_text=c.page_content[:500],
            chunk_id=c.metadata.get("chunk_uuid"),
            score=float(c.metadata.get("score", 0)),
        )
        for c in chunks
    ]


# =====================================================================
# Helper: load_all_chunks_by_doc_ids (สำหรับ /compare)
# =====================================================================
def load_all_chunks_by_doc_ids(
    vectorstore_manager,
    collection_name: str,
    stable_doc_ids: Set[str] | List[str]
) -> List[LcDocument]:
    chroma = vectorstore_manager._load_chroma_instance(collection_name)
    if not chroma:
        logger.warning(f"Chroma collection not found: {collection_name}")
        return []
    where_filter = {"stable_doc_uuid": {"$in": list(stable_doc_ids)}}
    docs = chroma.similarity_search(query="*", k=9999, filter=where_filter)
    return [d for d in docs if getattr(d, "page_content", "").strip()]

# =====================================================================
# 1. /query — Smart General RAG with Intent Detection & Auto-Routing
# =====================================================================
@llm_router.post("/query", response_model=QueryResponse)
async def query_llm(
    question: str = Form(...),
    conversation_id: Optional[str] = Form(None),
    doc_types: List[str] = Form(default=[]),  
    doc_ids: List[str] = Form(default=[]),     
    enabler: Optional[str] = Form(None),
    subject: Optional[str] = Form(None),
    year: Optional[str] = Form(None),
    current_user: UserMe = Depends(get_current_user),
):
    # 🎯 0. Setup พื้นฐาน
    llm = create_llm_instance(model_name=DEFAULT_LLM_MODEL_NAME, temperature=LLM_TEMPERATURE)
    conv_id = conversation_id or str(uuid.uuid4())
    effective_year = year or str(current_user.year)
    
    logger.info(f"📩 Query received: '{question}' from user {current_user.id}")

    # 🎯 1. ตรวจสอบประวัติและเจตนา (Intent Detection)
    history = await get_recent_history(current_user.id, conv_id, limit=6)
    intent = detect_intent(question, user_context=history)

    # 🔍 [SMART ROUTE OVERRIDE] 
    # ตรวจสอบ Keyword พิเศษเพิ่มเติม เพื่อบังคับไป /analysis หาก Intent Detection ปกติพลาด
    analysis_keywords = ["pdca", "comply", "compliance", "เกณฑ์", "ผ่านระดับ", "level", "จุดแข็ง", "ช่องว่าง", "ประเมิน", "สอดคล้อง"]
    is_forcing_analysis = any(kw in question.lower() for kw in analysis_keywords)

    # --- [BRANCH 1] Greeting & Capabilities ---
    if intent.get("is_greeting") or intent.get("is_capabilities"):
        if intent.get("is_greeting"):
            answer = "สวัสดีครับ! 😊 ผมคือ Digital Knowledge Assistant ของ PEA พร้อมช่วยตอบคำถามเกี่ยวกับเอกสาร KM หรือวิเคราะห์ SE-AM ให้ครับ มีอะไรให้ช่วยไหมครับ?"
        else:
            answer = (
                "ผมช่วยคุณได้ดังนี้ครับ:\n"
                "1. **ค้นหาและตอบคำถาม** จากเอกสาร/นโยบาย/ระเบียบ\n"
                "2. **สรุปเอกสาร** สาระสำคัญสำหรับผู้บริหาร\n"
                "3. **เปรียบเทียบเอกสาร** 2 ฉบับขึ้นไป\n"
                "4. **วิเคราะห์ตามเกณฑ์ SE-AM** (PDCA, จุดแข็ง, ช่องว่าง)"
            )
        await async_save_message(current_user.id, conv_id, "user", question)
        await async_save_message(current_user.id, conv_id, "ai", answer)
        return QueryResponse(answer=answer, sources=[], conversation_id=conv_id)

    # --- [BRANCH 2] Comparison (Redirect to compare_llm) ---
    if intent.get("is_comparison"):
        logger.info("🔀 Route -> Comparison")
        return await compare_llm(
            question=question, doc_ids=doc_ids or [],
            doc_types=doc_types, enabler=enabler, current_user=current_user
        )

    # --- [BRANCH 3] SE-AM Analysis (Redirect to analysis_llm) ---
    # บังคับ Route ถ้าเจอ Keyword หรือ Intent บอกว่าเป็น Analysis
    if intent.get("is_analysis") or intent.get("is_criteria_query") or is_forcing_analysis:
        if doc_ids:
            logger.info(f"🚀 Route -> Analysis (Forced: {is_forcing_analysis})")
            return await analysis_llm(
                question=question, doc_ids=doc_ids, doc_types=doc_types,
                enabler=enabler, subject=subject, conversation_id=conv_id,
                current_user=current_user, year=effective_year,
            )
        else:
            # กรณีถามแนววิเคราะห์แต่ลืมเลือกไฟล์
            if is_forcing_analysis:
                answer = "🔍 ผมเห็นว่าคุณต้องการให้วิเคราะห์เกณฑ์ SE-AM/PDCA รบกวนช่วยเลือกไฟล์เอกสารที่ต้องการให้ผมตรวจสอบที่ด้านข้างก่อนนะครับ"
                await async_save_message(current_user.id, conv_id, "user", question)
                await async_save_message(current_user.id, conv_id, "ai", answer)
                return QueryResponse(answer=answer, sources=[], conversation_id=conv_id)

    # --- [BRANCH 4] RAG Flow (General QA & Summary) ---
    logger.info("📖 Executing: General RAG Flow")
    used_doc_types = doc_types or DEFAULT_DOC_TYPES
    used_enabler = enabler or DEFAULT_ENABLER
    vsm = get_vectorstore_manager(tenant=current_user.tenant)
    stable_doc_ids = {str(idx).strip() for idx in doc_ids if str(idx).strip()} if doc_ids else None

    all_chunks = []
    for dt in used_doc_types:
        res = await asyncio.to_thread(
            retrieve_context_for_endpoint,
            vectorstore_manager=vsm, query=question, doc_type=dt,
            enabler=used_enabler, stable_doc_ids=stable_doc_ids,
            tenant=current_user.tenant, year=effective_year, subject=subject,
        )
        if isinstance(res, dict) and "top_evidences" in res:
            for ev in res.get("top_evidences", []):
                # Robust Metadata Extraction
                f_name = ev.get('source_filename') or ev.get('source') or 'Unknown'
                p_val = ev.get('page_label') or ev.get('page_number') or ev.get('page')
                p_display = str(p_val).strip() if p_val and str(p_val).lower() != 'n/a' else "N/A"

                from langchain.schema import Document as LcDocument
                all_chunks.append(
                    LcDocument(
                        page_content=ev["text"],
                        metadata={
                            "score": ev.get("rerank_score") or ev.get("score") or 0.0,
                            "doc_id": ev.get("doc_id"),
                            "source": f_name,
                            "page": p_display,
                            "chunk_uuid": ev.get("chunk_uuid"),
                        }
                    )
                )

    all_chunks.sort(key=lambda c: c.metadata.get("score", 0), reverse=True)
    final_chunks = all_chunks[:QUERY_FINAL_K]

    if not final_chunks:
        answer = "ขออภัยครับ ไม่พบเนื้อหาที่เกี่ยวข้องในเอกสารที่เลือกครับ"
        return QueryResponse(answer=answer, sources=[], conversation_id=conv_id)

    # Context with Metadata for AI
    context_text = "\n\n".join([
        f"[ไฟล์: {c.metadata['source']}, หน้า: {c.metadata['page']}]\n{c.page_content}" 
        for c in final_chunks
    ])

    # Intent-based Message Selection
    if intent.get("is_summary"):
        sys_msg = "คุณคือผู้เชี่ยวชาญด้าน KM ของ PEA ตอบเป็นภาษาไทย สรุปสาระสำคัญสำหรับผู้บริหาร และระบุเลขหน้าอ้างอิงเสมอ"
        prompt_text = SUMMARY_PROMPT_TEMPLATE.format(context=context_text)
    else:
        sys_msg = "ALWAYS ANSWER IN THAI.\n" + SYSTEM_QA_INSTRUCTION
        prompt_text = QA_PROMPT_TEMPLATE.format(context=context_text, question=question)

    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": prompt_text}
    ]
    
    # Invoke LLM (ใช้ wrapper ที่รองรับ invoke)
    raw = await asyncio.to_thread(llm.invoke, messages)
    answer = enforce_thai_primary_language(raw.content if hasattr(raw, "content") else str(raw))

    # Save conversation
    await async_save_message(current_user.id, conv_id, "user", question)
    await async_save_message(current_user.id, conv_id, "ai", answer)

    # Prepare Sources for Frontend
    sources = [
        QuerySource(
            source_id=str(c.metadata["doc_id"]),
            file_name=f"{c.metadata['source']} (หน้า {c.metadata['page']})",
            chunk_text=c.page_content[:500],
            chunk_id=c.metadata["chunk_uuid"],
            score=float(c.metadata["score"]),
        )
        for c in final_chunks
    ]

    return QueryResponse(answer=answer.strip(), sources=sources, conversation_id=conv_id)

# =====================================================================
# 2. /compare — Document Comparison
# =====================================================================
@llm_router.post("/compare", response_model=QueryResponse)
async def compare_llm(
    question: str = Form(...),
    doc_ids: List[str] = Form(...),
    doc_types: Optional[List[str]] = Form(None),
    enabler: Optional[str] = Form(None),
    current_user: UserMe = Depends(get_current_user),
):
    if len(doc_ids) < 2:
        raise HTTPException(400, "ต้องเลือกอย่างน้อย 2 เอกสารเพื่อเปรียบเทียบ")

    used_doc_types = doc_types or ["document"]
    is_evidence = any(dt.lower() == EVIDENCE_DOC_TYPES.lower() for dt in used_doc_types)
    used_enabler = enabler or (DEFAULT_ENABLER if is_evidence else None)

    if is_evidence and not used_enabler:
        raise HTTPException(400, "สำหรับ compare เอกสาร evidence ต้องระบุ enabler")

    collection_name = get_doc_type_collection_key(used_doc_types[0], used_enabler)
    vsm = get_vectorstore_manager(tenant=current_user.tenant)
    llm = create_llm_instance(model_name=DEFAULT_LLM_MODEL_NAME, temperature=LLM_TEMPERATURE)

    all_chunks = load_all_chunks_by_doc_ids(vsm, collection_name, set(doc_ids))
    if not all_chunks:
        raise HTTPException(400, "ไม่พบข้อมูลในเอกสารที่เลือก")

    doc_groups = defaultdict(list)
    for d in all_chunks:
        doc_key = str(d.metadata.get("stable_doc_uuid") or d.metadata.get("doc_id"))
        doc_groups[doc_key].append(d)

    doc_blocks = []
    for idx, doc_id in enumerate(doc_ids, start=1):
        chunks = doc_groups.get(str(doc_id), [])
        if not chunks:
            block = f"### เอกสารที่ {idx}\n(ไม่พบข้อมูลในเอกสารนี้)"
        else:
            fname = chunks[0].metadata.get("source", f"ID:{doc_id}")
            body = "\n".join(f"- {c.page_content}" for c in chunks)
            block = f"### เอกสารที่ {idx}: {fname}\n{body}"
        doc_blocks.append(block)

    prompt_text = COMPARE_PROMPT_TEMPLATE.format(documents_content="\n\n".join(doc_blocks), query=question)
    messages = [
        SystemMessage(content=SYSTEM_COMPARE_INSTRUCTION),
        HumanMessage(content=prompt_text),
    ]

    raw = await asyncio.to_thread(llm.invoke, messages)
    answer = enforce_thai_primary_language(raw.content if hasattr(raw, "content") else str(raw))

    conv_id = str(uuid.uuid4())
    await async_save_message(current_user.id, conv_id, "user", question)
    await async_save_message(current_user.id, conv_id, "ai", answer)

    return QueryResponse(answer=answer.strip(), sources=_map_sources(all_chunks[:10]), conversation_id=conv_id)


def enhance_analysis_query(question: str, subject_id: str, rubric_data: dict) -> str:
    """
    ขยายความคำถามโดยอ้างอิงจากชื่อหัวข้อใน Rubric เพื่อให้ RAG ค้นหาได้แม่นยำขึ้น
    """
    # 1. พยายามหาชื่อหัวข้อจาก Rubric JSON
    criteria_name = ""
    target_rubric = rubric_data.get(subject_id, {})
    if target_rubric:
        criteria_name = target_rubric.get("name", "")
    
    # 2. สร้างคำถามที่รวม Keywords สำคัญ
    enhanced = f"วิเคราะห์ข้อมูลและประเมินตามเกณฑ์ SE-AM หัวข้อ {subject_id} {criteria_name}: {question} "
    enhanced += "โดยเน้นการค้นหาหลักฐานด้าน แผนงาน (Plan), การลงมือปฏิบัติ (Do), การวัดผลและตรวจสอบ (Check), และการปรับปรุงแก้ไข (Act)"
    
    return enhanced

# =====================================================================
# 3. /analysis — PDCA-focused SE-AM analysis (Mac & Server Standard)
# =====================================================================
@llm_router.post("/analysis", response_model=QueryResponse)
async def analysis_llm(
    question: str = Form(...),
    doc_ids: Any = Form(None),      # รองรับทั้ง List จาก Frontend หรือ String คั่นด้วย comma
    doc_types: Any = Form(None),    
    enabler: Optional[str] = Form(None),
    subject: Optional[str] = Form(None), # subject คือ sub_id เช่น '1.1'
    conversation_id: Optional[str] = Form(None),
    year: Optional[str] = Form(None),
    current_user: UserMe = Depends(get_current_user),
):
    start_time = time.time()
    conv_id = conversation_id or str(uuid.uuid4())
    effective_year = year or str(current_user.year)

    # 🛠️ 1. Data Type Normalization (ป้องกัน AttributeError กรณีข้อมูลมาคนละ Format)
    stable_doc_ids = None
    if doc_ids:
        if isinstance(doc_ids, list):
            stable_doc_ids = {str(idx).strip() for idx in doc_ids if str(idx).strip()}
        elif isinstance(doc_ids, str):
            stable_doc_ids = {idx.strip() for idx in doc_ids.split(",") if idx.strip()}

    if not doc_types:
        used_doc_types = [EVIDENCE_DOC_TYPES]
    elif isinstance(doc_types, list):
        used_doc_types = doc_types
    elif isinstance(doc_types, str):
        used_doc_types = [dt.strip() for dt in doc_types.split(",") if dt.strip()]
    else:
        used_doc_types = [EVIDENCE_DOC_TYPES]

    # ตรวจสอบ Enabler สำหรับ Evidence
    is_evidence = any(dt.lower() == EVIDENCE_DOC_TYPES.lower() for dt in used_doc_types)
    used_enabler = enabler or (DEFAULT_ENABLER if is_evidence else None)

    if is_evidence and not used_enabler:
        raise HTTPException(400, "สำหรับ analysis เอกสาร evidence ต้องระบุ enabler")

    vsm = get_vectorstore_manager(tenant=current_user.tenant)
    llm = create_llm_instance(model_name=DEFAULT_LLM_MODEL_NAME, temperature=LLM_TEMPERATURE)

    # 🎯 2. Load Rubric JSON สำหรับใช้ใน Prompt และ Enhancement
    rubric_data = {}
    rubric_json_str = "{}"
    try:
        rubric_path = get_rubric_file_path(current_user.tenant, used_enabler)
        if os.path.exists(rubric_path):
            with open(rubric_path, 'r', encoding='utf-8') as f:
                rubric_data = json.load(f)
                rubric_json_str = json.dumps(rubric_data, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to load rubric JSON: {e}")

    # 🎯 3. Query Enhancement (ขยายความคำถามตามเกณฑ์ SE-AM PDCA)
    search_query = question
    if subject:
        search_query = enhance_analysis_query(question, subject, rubric_data)

    # 🎯 4. Retrieval (ดึงข้อมูลพร้อม Filter ที่คุณปรับปรุง)
    from core.llm_data_utils import retrieve_context_with_rubric
    
    all_evidences = []
    for dt in used_doc_types:
        retrieval_res = await asyncio.to_thread(
            retrieve_context_with_rubric,
            vectorstore_manager=vsm,
            query=search_query,
            doc_type=dt,
            enabler=used_enabler,
            stable_doc_ids=stable_doc_ids,
            tenant=current_user.tenant,
            year=effective_year,
            subject=subject,
            strict_filter=True # บังคับใช้ Filter ที่คุณแก้ใน _create_where_filter
        )
        if retrieval_res and "top_evidences" in retrieval_res:
            all_evidences.extend(retrieval_res["top_evidences"])

    # เรียงลำดับตามคะแนนความเกี่ยวข้อง
    all_evidences.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
    final_evidences = all_evidences[:QUERY_FINAL_K]

    if not final_evidences:
        answer = "ไม่พบเนื้อหาที่เกี่ยวข้องในเอกสารที่เลือกเพื่อนำมาวิเคราะห์ครับ"
        return QueryResponse(answer=answer, sources=[], conversation_id=conv_id)

    # 🎯 5. SE-AM Assessment Engine (จัดกลุ่มข้อมูลตาม PDCA)
    engine_config = AssessmentConfig(
        tenant=current_user.tenant,
        year=int(effective_year) if effective_year.isdigit() else current_user.year,
        enabler=used_enabler,
        target_level=5
    )
    
    engine = SEAMPDCAEngine(
        config=engine_config, 
        llm_instance=llm, 
        vectorstore_manager=vsm, 
        doc_type=used_doc_types[0]
    )

    # ดึง Blocks ข้อมูลแยกตาม P D C A
    plan_blocks, do_blocks, check_blocks, act_blocks, other_blocks = engine._get_pdca_blocks_from_evidences(
        evidences=final_evidences,
        baseline_evidences={},
        level=5,
        sub_id=subject or "all",
        contextual_rules_map=engine.contextual_rules_map
    )
    pdca_context = "\n\n".join(filter(None, [plan_blocks, do_blocks, check_blocks, act_blocks, other_blocks]))

    # 🎯 6. Inference (ส่งข้อมูลให้ LLM วิเคราะห์)
    prompt_text = ANALYSIS_PROMPT_TEMPLATE.format(
        rubric_json=rubric_json_str,
        documents_content=pdca_context,
        question=question 
    )

    messages = [
        SystemMessage(content="ALWAYS ANSWER IN THAI.\n" + SYSTEM_ANALYSIS_INSTRUCTION),
        HumanMessage(content=prompt_text),
    ]

    raw_response = await asyncio.to_thread(llm.invoke, messages)
    answer = enforce_thai_primary_language(raw_response.content if hasattr(raw_response, "content") else str(raw_response))

    # 🎯 7. Source Mapping (แก้ไขให้ดึง Metadata ใหม่ที่ Ingest มา)
    sources = []
    for ev in final_evidences[:10]:
        # 🟢 ลำดับความสำคัญของชื่อไฟล์: source_filename (ใหม่) -> source (เดิม)
        f_name = ev.get('source_filename') or ev.get('source') or 'Unknown'
        
        # 🟢 ลำดับความสำคัญของหน้า: page_label (UI-Ready) -> page_number -> page
        p_val = ev.get('page_label') or ev.get('page_number') or ev.get('page')
        
        # ตรวจสอบความสะอาดของข้อมูลหน้า
        p_display = str(p_val).strip() if p_val and str(p_val).lower() != 'n/a' else "N/A"

        sources.append(
            QuerySource(
                source_id=str(ev.get("doc_id", "unknown")),
                file_name=f"{f_name} (หน้า {p_display})",
                chunk_text=ev.get("text", "")[:500],
                chunk_id=ev.get("chunk_uuid"),
                # ใช้ rerank_score (จาก Mac/GPU) ถ้าไม่มีให้ใช้ score ปกติ
                score=float(ev.get("rerank_score") or ev.get("score") or 0.0),
            )
        )

    # บันทึกประวัติสนทนา
    await async_save_message(current_user.id, conv_id, "user", question)
    await async_save_message(current_user.id, conv_id, "ai", answer)

    return QueryResponse(
        answer=answer.strip(), 
        sources=sources, 
        conversation_id=conv_id,
        result={"process_time": round(time.time() - start_time, 2)}
    )