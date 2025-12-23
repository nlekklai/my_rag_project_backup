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
# 1. /query — Smart General RAG with Intent Detection & Auto-Routing (Complete Revised)
# =====================================================================
@llm_router.post("/query", response_model=QueryResponse)
async def query_llm(
    question: str = Form(...),
    conversation_id: Optional[str] = Form(None),
    doc_types: Optional[List[str]] = Form(None),
    doc_ids: Optional[List[str]] = Form(None),
    enabler: Optional[str] = Form(None),
    subject: Optional[str] = Form(None),
    year: Optional[str] = Form(None),
    current_user: UserMe = Depends(get_current_user),
):
    # 🎯 0. Setup พื้นฐาน
    llm = create_llm_instance(model_name=DEFAULT_LLM_MODEL_NAME, temperature=LLM_TEMPERATURE)
    conv_id = conversation_id or str(uuid.uuid4())
    effective_year = year or str(current_user.year)

    # 🎯 1. ตรวจสอบประวัติและเจตนา (Intent Detection)
    history = await get_recent_history(current_user.id, conv_id, limit=6)
    intent = detect_intent(question, user_context=history)

    # --- [BRANCH 1] Greeting & Capabilities (Static Response) ---
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
        return await compare_llm(
            question=question, doc_ids=doc_ids or [],
            doc_types=doc_types, enabler=enabler, current_user=current_user
        )

    # --- [BRANCH 3] SE-AM Analysis (Redirect to analysis_llm) ---
    if intent.get("is_analysis") or intent.get("is_criteria_query"):
        if doc_ids:
            return await analysis_llm(
                question=question, doc_ids=doc_ids, doc_types=doc_types,
                enabler=enabler, subject=subject, conversation_id=conv_id,
                current_user=current_user, year=year,
            )
        else:
            answer = "🔍 กรุณาเลือกเอกสารที่ต้องการวิเคราะห์ก่อน แล้วผมจะบอกได้ทันทีว่าผ่าน Level ไหน พร้อมจุดแข็งและข้อเสนอแนะครับ"
            await async_save_message(current_user.id, conv_id, "user", question)
            await async_save_message(current_user.id, conv_id, "ai", answer)
            return QueryResponse(answer=answer, sources=[], conversation_id=conv_id)

    # --- [BRANCH 4] RAG Flow (Summary & General QA) ---
    # เตรียมพารามิเตอร์สำหรับการดึงข้อมูล
    used_doc_types = doc_types or DEFAULT_DOC_TYPES
    used_enabler = enabler or DEFAULT_ENABLER
    vsm = get_vectorstore_manager(tenant=current_user.tenant)
    stable_doc_ids = set(doc_ids) if doc_ids else None

    all_chunks = []
    # วนลูปดึงข้อมูลตาม doc_types ที่เลือก
    for dt in used_doc_types:
        res = await asyncio.to_thread(
            retrieve_context_for_endpoint,
            vectorstore_manager=vsm, query=question, doc_type=dt,
            enabler=used_enabler, stable_doc_ids=stable_doc_ids,
            tenant=current_user.tenant, year=effective_year, subject=subject,
        )
        if isinstance(res, dict):
            for ev in res.get("top_evidences", []):
                all_chunks.append(
                    LcDocument(
                        page_content=ev["text"],
                        metadata={
                            "score": ev.get("score", 0),
                            "doc_id": ev.get("doc_id"),
                            "source": ev.get("source"),
                            # ⭐ จุดสำคัญ: ดึงเลขหน้ามาเก็บไว้ ป้องกัน N/A
                            "page": str(ev.get("page") or ev.get("page_number") or "N/A"),
                            "chunk_uuid": ev.get("chunk_uuid"),
                        }
                    )
                )

    # จัดลำดับและจำกัดจำนวน chunks
    all_chunks.sort(key=lambda c: c.metadata.get("score", 0), reverse=True)
    final_chunks = all_chunks[:QUERY_FINAL_K]

    # กรณีไม่พบข้อมูล
    if not final_chunks:
        answer = "ขออภัยครับ ไม่พบเนื้อหาที่เกี่ยวข้องในเอกสารที่เลือกครับ"
        return QueryResponse(answer=answer, sources=[], conversation_id=conv_id)

    # สร้างบริบท (Context) พร้อมระบุแหล่งที่มาและเลขหน้าในตัวเนื้อหาเพื่อให้ AI เห็น
    context_text = "\n\n".join([
        f"[เอกสาร: {c.metadata['source']}, หน้า: {c.metadata['page']}]\n{c.page_content}" 
        for c in final_chunks
    ])

    # เลือก Prompt และ System Message ตามเจตนา
    if intent.get("is_summary"):
        sys_msg = (
            "คุณคือที่ปรึกษาอาวุโสด้าน KM ของ PEA ตอบเป็นภาษาไทยเท่านั้น "
            "ห้ามใช้คำว่า Executive Summary ให้ใช้ 'สรุปสาระสำคัญสำหรับผู้บริหาร' แทน "
            "และต้องระบุเลขหน้าอ้างอิงทุกครั้ง"
        )
        prompt_text = SUMMARY_PROMPT_TEMPLATE.format(context=context_text)
    else:
        sys_msg = "ALWAYS ANSWER IN THAI.\n" + SYSTEM_QA_INSTRUCTION
        prompt_text = QA_PROMPT_TEMPLATE.format(context=context_text, question=question)

    # เรียก LLM
    messages = [SystemMessage(content=sys_msg), HumanMessage(content=prompt_text)]
    raw = await asyncio.to_thread(llm.invoke, messages)
    answer = enforce_thai_primary_language(raw.content if hasattr(raw, "content") else str(raw))

    # บันทึกประวัติ
    await async_save_message(current_user.id, conv_id, "user", question)
    await async_save_message(current_user.id, conv_id, "ai", answer)

    # สร้าง Source List ส่งกลับ UI พร้อมข้อมูลเลขหน้า
    sources = [
        QuerySource(
            source_id=str(c.metadata["doc_id"]),
            file_name=f"{c.metadata['source']} (หน้า {c.metadata.get('page_label') or c.metadata.get('page_number') or c.metadata.get('page') or 'N/A'})",
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
# 3. /analysis — PDCA-focused SE-AM analysis with Query Enhancement
# =====================================================================
# =====================================================================
# 3. /analysis — PDCA-focused SE-AM analysis with Query Enhancement
# =====================================================================
@llm_router.post("/analysis", response_model=QueryResponse)
async def analysis_llm(
    question: str = Form(...),
    doc_ids: Optional[List[str]] = Form(None),
    doc_types: Optional[List[str]] = Form(None),
    enabler: Optional[str] = Form(None),
    subject: Optional[str] = Form(None), # subject คือ sub_id เช่น '1.1'
    conversation_id: Optional[str] = Form(None),
    year: Optional[str] = Form(None),
    current_user: UserMe = Depends(get_current_user),
):
    conv_id = conversation_id or str(uuid.uuid4())
    effective_year = year or str(current_user.year)

    # 1. จัดการ Enabler และ Doc Types
    used_doc_types = doc_types or [EVIDENCE_DOC_TYPES]
    is_evidence = any(dt.lower() == EVIDENCE_DOC_TYPES.lower() for dt in used_doc_types)
    used_enabler = enabler or (DEFAULT_ENABLER if is_evidence else None)

    if is_evidence and not used_enabler:
        raise HTTPException(400, "สำหรับ analysis เอกสาร evidence ต้องระบุ enabler")

    vsm = get_vectorstore_manager(tenant=current_user.tenant)
    stable_doc_ids = set(doc_ids) if doc_ids else None

    # 🎯 2. Load Rubric Data สำหรับ Query Enhancement
    rubric_data = {}
    rubric_json_str = "{}"
    try:
        rubric_path = get_rubric_file_path(current_user.tenant, used_enabler)
        if os.path.exists(rubric_path):
            with open(rubric_path, 'r', encoding='utf-8') as f:
                rubric_data = json.load(f)
                rubric_json_str = json.dumps(rubric_data, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to load rubric: {e}")

    # 🎯 3. Enhance Question: ขยายความคำถามเพื่อให้ RAG ค้นหาได้แม่นยำ (Plan, Do, Check, Act)
    search_query = question
    if subject:
        search_query = enhance_analysis_query(question, subject, rubric_data)

    # 🎯 4. Retrieval (ดึงข้อมูลพร้อมเลขหน้า)
    all_chunks = []
    for dt in used_doc_types:
        res = await asyncio.to_thread(
            retrieve_context_for_endpoint,
            vectorstore_manager=vsm,
            query=search_query,
            doc_type=dt,
            enabler=used_enabler,
            stable_doc_ids=stable_doc_ids,
            tenant=current_user.tenant,
            year=effective_year,
            subject=subject,
        )
        if isinstance(res, dict):
            for ev in res.get("top_evidences", []):
                # สำคัญ: เก็บ 'page' จาก metadata เพื่อใช้ใน prompt
                all_chunks.append(
                    LcDocument(
                        page_content=ev["text"],
                        metadata={
                            "score": ev.get("score", 0),
                            "doc_id": ev.get("doc_id"),
                            "source": ev.get("source"),
                            "page": str(ev.get("page") or ev.get("page_number") or "N/A"),
                            "chunk_uuid": ev.get("chunk_uuid"),
                            "pdca_tag": ev.get("pdca_tag", "Other"),
                        },
                    )
                )

    all_chunks.sort(key=lambda c: c.metadata.get("score", 0), reverse=True)
    all_chunks = all_chunks[:QUERY_FINAL_K]

    if not all_chunks:
        raise HTTPException(400, "ไม่พบข้อมูลหลักฐานสำหรับวิเคราะห์")

    # แปลงเป็น dict format เพื่อส่งให้ Engine
    evidences = [
        {
            "text": c.page_content,
            "source": c.metadata.get("source"),
            "page": c.metadata.get("page", "N/A"),
            "doc_id": c.metadata.get("doc_id"),
            "chunk_uuid": c.metadata.get("chunk_uuid"),
            "rerank_score": c.metadata.get("score", 0.0),
            "pdca_tag": c.metadata.get("pdca_tag", "Other")
        }
        for c in all_chunks
    ]

    # 🎯 5. Initialize Engine
    primary_doc_type = used_doc_types[0] if used_doc_types else EVIDENCE_DOC_TYPES
    engine_config = AssessmentConfig(
        tenant=current_user.tenant,
        year=current_user.year,
        enabler=used_enabler,
        target_level=5
    )

    llm = create_llm_instance(model_name=DEFAULT_LLM_MODEL_NAME, temperature=LLM_TEMPERATURE)
    engine = SEAMPDCAEngine(
        config=engine_config,
        llm_instance=llm,
        vectorstore_manager=vsm,
        doc_type=primary_doc_type
    )

    # 🎯 6. PDCA Context Preparation
    # ฟังก์ชันนี้จะนำ 'page' ใน evidences มาประกอบร่างเป็น [Source: ..., หน้า: ...]
    plan_blocks, do_blocks, check_blocks, act_blocks, other_blocks = engine._get_pdca_blocks_from_evidences(
        evidences=evidences,
        baseline_evidences={},
        level=5,
        sub_id=subject or "all",
        contextual_rules_map=engine.contextual_rules_map
    )
    pdca_context = "\n\n".join(filter(None, [plan_blocks, do_blocks, check_blocks, act_blocks, other_blocks]))

    # 🎯 7. Inference
    prompt_text = ANALYSIS_PROMPT_TEMPLATE.format(
        rubric_json=rubric_json_str,
        documents_content=pdca_context,
        question=question 
    )

    messages = [
        SystemMessage(content="ALWAYS ANSWER IN THAI.\n" + SYSTEM_ANALYSIS_INSTRUCTION),
        HumanMessage(content=prompt_text),
    ]

    raw = await asyncio.to_thread(llm.invoke, messages)
    answer = enforce_thai_primary_language(raw.content if hasattr(raw, "content") else str(raw))

    # บันทึกประวัติการสนทนา
    await async_save_message(current_user.id, conv_id, "user", question)
    await async_save_message(current_user.id, conv_id, "ai", answer)

    # 🎯 8. Return Response with Metadata-enriched Sources
    sources = [
        QuerySource(
            source_id=str(c.get("doc_id", "unknown")),
            file_name=f"{c.metadata['source']} (หน้า {c.metadata.get('page_label') or c.metadata.get('page_number') or c.metadata.get('page') or 'N/A'})",
            chunk_text=c.get("text", "")[:500],
            chunk_id=c.get("chunk_uuid"),
            score=float(c.get("rerank_score", 0)),
        )
        for c in evidences[:10]
    ]

    return QueryResponse(answer=answer.strip(), sources=sources, conversation_id=conv_id)