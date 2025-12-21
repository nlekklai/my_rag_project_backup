# -*- coding: utf-8 -*-
# routers/llm_router.py - Enterprise RAG (Query + Compare + PDCA Analysis + Summary)
# ULTIMATE REVISED VERSION - 21 ธันวาคม 2568
# เพิ่ม: Intent detection + Auto-route + User guidance + Summary support

import logging
import uuid
import asyncio
from typing import List, Optional, Any, Dict, Union, Set
from collections import defaultdict

from fastapi import APIRouter, Form, HTTPException, Depends
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document as LcDocument

from core.history_utils import async_save_message, get_recent_history
from core.llm_data_utils import retrieve_context_for_endpoint, _create_where_filter
from core.vectorstore import get_vectorstore_manager
from core.seam_assessment import SEAMPDCAEngine
from core.llm_guardrails import enforce_thai_primary_language, detect_intent, build_prompt
from config.global_vars import (
    EVIDENCE_DOC_TYPES,
    DEFAULT_ENABLER,
    DEFAULT_LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    QUERY_FINAL_K,
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
    SUMMARY_PROMPT,
    SUMMARY_PROMPT_TEMPLATE
)

from utils.path_utils import _n, get_doc_type_collection_key
from utils.path_utils import get_document_file_path

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
# Helpers
# =====================================================================
async def _get_context_chunks(
    question: str,
    doc_types: List[str],
    stable_doc_ids: Optional[Set[str]],
    enabler: Optional[str],
    subject: Optional[str],
    vsm,
    user: UserMe,
):
    tasks = [
        asyncio.to_thread(
            retrieve_context_for_endpoint,
            vectorstore_manager=vsm,
            query=question,
            doc_type=dt,
            enabler=enabler,
            stable_doc_ids=stable_doc_ids,
            tenant=user.tenant,
            year=user.year,
            subject=subject,
        )
        for dt in doc_types
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    chunks: List[LcDocument] = []
    for res in results:
        if isinstance(res, dict):
            for ev in res.get("top_evidences", []):
                chunks.append(
                    LcDocument(
                        page_content=ev["text"],
                        metadata={
                            "score": ev.get("score", 0),
                            "doc_id": ev.get("doc_id"),
                            "source": ev.get("source"),
                            "chunk_uuid": ev.get("chunk_uuid"),
                            "pdca_tag": ev.get("pdca_tag", "Other"),
                        },
                    )
                )
    chunks.sort(key=lambda c: c.metadata.get("score", 0), reverse=True)
    return chunks[:QUERY_FINAL_K]


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


def load_all_chunks_by_doc_ids(
    vectorstore_manager,
    collection_name: str,
    stable_doc_ids: Union[Set[str], List[str]]
) -> List[LcDocument]:
    chroma = vectorstore_manager._load_chroma_instance(collection_name)
    if not chroma:
        logger.warning(f"Chroma collection not found: {collection_name}")
        return []
    where_filter = _create_where_filter(stable_doc_ids=set(stable_doc_ids))
    docs = chroma.similarity_search(query="*", k=9999, filter=where_filter)
    return [d for d in docs if getattr(d, "page_content", "").strip()]


# =====================================================================
# 1. /query — Smart General RAG with Intent Detection & Auto-Routing
# =====================================================================
@llm_router.post("/query", response_model=QueryResponse)
async def query_llm(
    question: str = Form(...),
    doc_types: Optional[List[str]] = Form(None),
    doc_ids: Optional[List[str]] = Form(None),
    enabler: Optional[str] = Form(None),
    subject: Optional[str] = Form(None),
    conversation_id: Optional[str] = Form(None),
    current_user: UserMe = Depends(get_current_user),
):
    llm = create_llm_instance(model_name=DEFAULT_LLM_MODEL_NAME, temperature=LLM_TEMPERATURE)
    conv_id = conversation_id or str(uuid.uuid4())

    # ดึงประวัติการสนทนาเพื่อใช้ใน intent detection
    history = await get_recent_history(current_user.id, conv_id, limit=6)

    # วิเคราะห์ intent
    intent = detect_intent(question, user_context=history)

    # === Greeting Intent (ตรวจก่อนอย่างอื่น เพราะเป็นคำทักทายธรรมดา) ===
    if intent.get("is_greeting"):
        answer = (
            "สวัสดีครับ! 😊\n"
            "ผมคือ Digital Knowledge Assistant ของ PEA\n"
            "พร้อมช่วยตอบคำถามเกี่ยวกับเอกสาร นโยบาย การจัดการความรู้ (KM)\n"
            "หรือวิเคราะห์เกณฑ์ SE-AM ให้ครับ\n\n"
            "มีอะไรให้ช่วยวันนี้บ้างครับ?"
        )
        await async_save_message(current_user.id, conv_id, "user", question)
        await async_save_message(current_user.id, conv_id, "ai", answer)
        return QueryResponse(answer=answer, sources=[], conversation_id=conv_id)
    
    # Auto-route ตาม intent
    if intent.get("is_comparison"):
        return await compare_llm(
            question=question,
            doc_ids=doc_ids or [],
            doc_types=doc_types,
            enabler=enabler,
            current_user=current_user
        )

    if intent.get("is_analysis") or intent.get("is_criteria_query"):
        if doc_ids:
            return await analysis_llm(
                question=question,
                doc_ids=doc_ids,
                doc_types=doc_types,
                enabler=enabler,
                subject=subject,
                conversation_id=conv_id,
                current_user=current_user
            )
        else:
            answer = (
                "🔍 คำถามของคุณเกี่ยวกับการวิเคราะห์หลักฐานตามเกณฑ์ SE-AM\n\n"
                "เพื่อผลลัพธ์ที่แม่นยำที่สุด กรุณาเลือกเอกสารที่ต้องการวิเคราะห์ก่อน\n"
                "แล้วใช้คำสั่ง **วิเคราะห์** หรือกดปุ่ม 'วิเคราะห์หลักฐาน'\n\n"
                "ระบบจะบอกได้ชัดเจนว่า:\n"
                "- ผ่าน Level ไหนบ้าง\n"
                "- จุดแข็งและช่องว่าง\n"
                "- ข้อเสนอแนะเพื่อยกระดับ"
            )
            await async_save_message(current_user.id, conv_id, "user", question)
            await async_save_message(current_user.id, conv_id, "ai", answer)
            return QueryResponse(answer=answer, sources=[], conversation_id=conv_id)

    if intent.get("is_summary"):
        used_doc_types = doc_types or [EVIDENCE_DOC_TYPES]
        used_enabler = enabler or DEFAULT_ENABLER

        vsm = get_vectorstore_manager(tenant=current_user.tenant)
        stable_doc_ids = set(doc_ids) if doc_ids else None

        all_chunks = await _get_context_chunks(
            question=question,
            doc_types=used_doc_types,
            stable_doc_ids=stable_doc_ids,
            enabler=used_enabler,
            subject=subject,
            vsm=vsm,
            user=current_user,
        )

        if not all_chunks:
            raise HTTPException(400, "ไม่พบข้อมูลสำหรับสรุป")

        context_text = "\n\n".join(f"[{c.metadata.get('source')}]\n{c.page_content}" for c in all_chunks)

        # แก้ตรงนี้: ใช้ template.format()
        prompt_text = SUMMARY_PROMPT_TEMPLATE.format(context=context_text)

        messages = [
            SystemMessage(content=(
                "คุณคือที่ปรึกษาอาวุโสด้านการจัดการความรู้ของการไฟฟ้าส่วนภูมิภาค\n"
                "ตอบเป็นภาษาไทยเท่านั้น ห้ามใช้ภาษาอังกฤษแม้แต่คำเดียว\n"
                "ยกเว้นชื่อเอกสารหรือคำย่อที่ปรากฏในเอกสารจริง เช่น KM, KMS, PEA\n"
                "ห้ามใช้คำว่า Executive Summary ให้ใช้ 'สรุปสาระสำคัญสำหรับผู้บริหาร' แทน"
            )),
            HumanMessage(content=prompt_text),
        ]

        raw = await asyncio.to_thread(llm.invoke, messages)
        answer = enforce_thai_primary_language(raw.content if hasattr(raw, "content") else str(raw))

        await async_save_message(current_user.id, conv_id, "user", question)
        await async_save_message(current_user.id, conv_id, "ai", answer)

        return QueryResponse(answer=answer.strip(), sources=_map_sources(all_chunks), conversation_id=conv_id)
    
    # === General QA (fallback) ===
    used_doc_types = doc_types or [EVIDENCE_DOC_TYPES]
    is_evidence = any(_n(dt) == _n(EVIDENCE_DOC_TYPES) for dt in used_doc_types)
    used_enabler = enabler or (DEFAULT_ENABLER if is_evidence else None)

    if is_evidence and not used_enabler:
        raise HTTPException(status_code=400, detail="สำหรับเอกสาร evidence ต้องระบุ enabler (เช่น KM, IM)")

    vsm = get_vectorstore_manager(tenant=current_user.tenant)
    stable_doc_ids = set(doc_ids) if doc_ids else None

    all_chunks = await _get_context_chunks(
        question=question,
        doc_types=used_doc_types,
        stable_doc_ids=stable_doc_ids,
        enabler=used_enabler,
        subject=subject,
        vsm=vsm,
        user=current_user,
    )

    if not all_chunks and doc_ids:
        raise HTTPException(status_code=400, detail="ไม่พบข้อมูลในเอกสารที่เลือก")

    context_text = "\n\n".join(f"[{c.metadata.get('source')}]\n{c.page_content}" for c in all_chunks)
    prompt_text = QA_PROMPT_TEMPLATE.format(context=context_text, question=question)

    messages = [
        SystemMessage(content="ALWAYS ANSWER IN THAI.\n" + SYSTEM_QA_INSTRUCTION),
        HumanMessage(content=prompt_text),
    ]

    raw = await asyncio.to_thread(llm.invoke, messages)
    answer = enforce_thai_primary_language(raw.content if hasattr(raw, "content") else str(raw))

    await async_save_message(current_user.id, conv_id, "user", question)
    await async_save_message(current_user.id, conv_id, "ai", answer)

    return QueryResponse(answer=answer.strip(), sources=_map_sources(all_chunks), conversation_id=conv_id)


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

    used_doc_types = doc_types or [EVIDENCE_DOC_TYPES]
    is_evidence = any(_n(dt) == _n(EVIDENCE_DOC_TYPES) for dt in used_doc_types)
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


# =====================================================================
# 3. /analysis — PDCA-focused SE-AM analysis
# =====================================================================
@llm_router.post("/analysis", response_model=QueryResponse)
async def analysis_llm(
    question: str = Form(...),
    doc_ids: Optional[List[str]] = Form(None),
    doc_types: Optional[List[str]] = Form(None),
    enabler: Optional[str] = Form(None),
    subject: Optional[str] = Form(None),
    conversation_id: Optional[str] = Form(None),
    current_user: UserMe = Depends(get_current_user),
):
    conv_id = conversation_id or str(uuid.uuid4())

    used_doc_types = doc_types or [EVIDENCE_DOC_TYPES]
    is_evidence = any(_n(dt) == _n(EVIDENCE_DOC_TYPES) for dt in used_doc_types)
    used_enabler = enabler or (DEFAULT_ENABLER if is_evidence else None)

    if is_evidence and not used_enabler:
        raise HTTPException(400, "สำหรับ analysis เอกสาร evidence ต้องระบุ enabler")

    vsm = get_vectorstore_manager(tenant=current_user.tenant)
    llm = create_llm_instance(model_name=DEFAULT_LLM_MODEL_NAME, temperature=LLM_TEMPERATURE)

    stable_doc_ids = set(doc_ids) if doc_ids else None

    all_chunks = await _get_context_chunks(
        question=question,
        doc_types=used_doc_types,
        stable_doc_ids=stable_doc_ids,
        enabler=used_enabler,
        subject=subject,
        vsm=vsm,
        user=current_user,
    )

    if not all_chunks:
        raise HTTPException(400, "ไม่พบข้อมูลหลักฐานสำหรับวิเคราะห์")

    evidences = [
        {
            "text": c.page_content,
            "source": c.metadata.get("source"),
            "doc_id": c.metadata.get("doc_id"),
            "chunk_uuid": c.metadata.get("chunk_uuid"),
            "rerank_score": c.metadata.get("score", 0.0),
            "pdca_tag": c.metadata.get("pdca_tag", "Other")
        }
        for c in all_chunks
    ]

    engine = SEAMPDCAEngine(
        config=dict(
            tenant=current_user.tenant,
            year=current_user.year,
            enabler=used_enabler,
            target_level=5,
        ),
        llm_instance=llm,
        vectorstore_manager=vsm
    )

    plan_blocks, do_blocks, check_blocks, act_blocks, other_blocks = engine._get_pdca_blocks_from_evidences(
        evidences=evidences, baseline_evidences={}, level=5, sub_id=subject or "all",
        contextual_rules_map=engine.contextual_rules_map
    )
    pdca_context = "\n\n".join(filter(None, [plan_blocks, do_blocks, check_blocks, act_blocks, other_blocks]))
    prompt_text = ANALYSIS_PROMPT_TEMPLATE.format(documents_content=pdca_context, question=question)

    messages = [
        SystemMessage(content="ALWAYS ANSWER IN THAI.\n" + SYSTEM_ANALYSIS_INSTRUCTION),
        HumanMessage(content=prompt_text),
    ]

    raw = await asyncio.to_thread(llm.invoke, messages)
    answer = enforce_thai_primary_language(raw.content if hasattr(raw, "content") else str(raw))

    await async_save_message(current_user.id, conv_id, "user", question)
    await async_save_message(current_user.id, conv_id, "ai", answer)

    sources = [
        QuerySource(
            source_id=str(c.get("doc_id", "unknown")),
            file_name=c.get("source", "Unknown"),
            chunk_text=c.get("text", "")[:500],
            chunk_id=c.get("chunk_uuid"),
            score=float(c.get("rerank_score", 0)),
        )
        for c in evidences[:10]
    ]

    return QueryResponse(answer=answer.strip(), sources=sources, conversation_id=conv_id)