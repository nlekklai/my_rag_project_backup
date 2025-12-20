# -*- coding: utf-8 -*-
# routers/llm_router.py - Enterprise RAG (Query + Compare + PDCA Analysis)
# FINAL STABLE VERSION - FIXED AttributeError: 'dict' object has no attribute 'metadata'

import logging
import uuid
import asyncio
from typing import List, Optional, Any, Dict, Union, Set
from collections import defaultdict

from fastapi import APIRouter, Form, HTTPException, Depends
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document as LcDocument

from core.history_utils import async_save_message
from core.llm_data_utils import retrieve_context_with_rubric, _create_where_filter, retrieve_context_for_endpoint
from core.vectorstore import get_vectorstore_manager
from core.seam_assessment import SEAMPDCAEngine
from core.llm_guardrails import enforce_thai_primary_language
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
    ANALYSIS_PROMPT  # ใช้ ANALYSIS_PROMPT ฉบับใหม่ที่รองรับ rubric_json
)
from utils.path_utils import _n, get_doc_type_collection_key, get_rubric_file_path
import os, json
# routers/llm_router.py
from core.history_utils import async_save_message, get_recent_history

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
# Helpers (Revised for Type Safety)
# =====================================================================
async def _get_context_chunks(
    question: str,
    doc_types: List[str],
    stable_doc_ids: Optional[Set[str]],
    enabler: Optional[str],
    subject: Optional[str],
    vsm,
    user: UserMe,
    rubric_vectorstore_name: Optional[str] = None,
):
    tasks = []
    for dt in doc_types:
        tasks.append(
            asyncio.to_thread(
                retrieve_context_with_rubric,
                vectorstore_manager=vsm,
                query=question,
                doc_type=dt,
                enabler=enabler,
                stable_doc_ids=stable_doc_ids,
                tenant=user.tenant,
                year=user.year,
                subject=subject,
                rubric_vectorstore_name=rubric_vectorstore_name,
                top_k=QUERY_FINAL_K,
            )
        )
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    all_retrieved: List[Union[LcDocument, dict]] = []
    
    for res in results:
        if isinstance(res, list):
            all_retrieved.extend(res)
        elif isinstance(res, Exception):
            logger.error(f"❌ Retrieval Task Failed: {res}")

    def extract_score(item):
        if isinstance(item, dict):
            return float(item.get("rerank_score") or item.get("score") or 0.1)
        return float(item.metadata.get("_rerank_score_force") or item.metadata.get("score") or 0.1)

    all_retrieved.sort(key=extract_score, reverse=True)
    return all_retrieved[:QUERY_FINAL_K]


def _map_sources(chunks: List[Union[LcDocument, dict]]) -> List[QuerySource]:
    sources = []
    for c in chunks:
        if isinstance(c, dict):
            sources.append(QuerySource(
                source_id=str(c.get("doc_id", "unknown")),
                file_name=c.get("source", "Unknown"),
                chunk_text=c.get("text", "")[:500],
                chunk_id=c.get("chunk_uuid"),
                score=float(c.get("rerank_score") or 0.0),
            ))
        else:
            m = c.metadata or {}
            sources.append(QuerySource(
                source_id=str(m.get("doc_id", m.get("stable_doc_uuid", "unknown"))),
                file_name=m.get("source", "Unknown"),
                chunk_text=c.page_content[:500],
                chunk_id=m.get("chunk_uuid"),
                score=float(m.get("_rerank_score_force") or m.get("score") or 0.0),
            ))
    return sources

# เพิ่มฟังก์ชันเหล่านี้ลงใน routers/llm_router.py

def ensure_deterministic_id(doc_id: str) -> str:
    if not doc_id: return doc_id
    try:
        import uuid
        uuid.UUID(doc_id)
        return str(doc_id)
    except ValueError:
        import uuid
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id))

def _parse_doc_ids(doc_ids: Any) -> List[str]:
    """
    ฟังก์ชันสำหรับแปลงข้อมูล doc_ids ที่ส่งมาจาก Frontend (ซึ่งอาจเป็น String หรือ List)
    ให้กลายเป็น List[str] ที่สะอาดพร้อมใช้งาน
    """
    import re
    if not doc_ids: return []
    if isinstance(doc_ids, str):
        # ล้างเครื่องหมาย [], '', "" และช่องว่าง แล้วแยกด้วย comma
        clean = re.sub(r"[\[\]'\" ]", "", doc_ids)
        return [d for d in clean.split(",") if d]
    if isinstance(doc_ids, list):
        return [str(d) for d in doc_ids if d]
    return []

# =====================================================================
# 1. /query — General RAG QA (Updated to use retrieve_context_for_endpoint)
# =====================================================================
@llm_router.post("/query", response_model=QueryResponse)
async def query_llm(
    question: str = Form(...),
    doc_types: Optional[List[str]] = Form(None),
    doc_ids: Optional[Any] = Form(None), # รับเป็น Any เพื่อใช้ _parse_doc_ids
    enabler: Optional[str] = Form(None),
    subject: Optional[str] = Form(None),
    conversation_id: Optional[str] = Form(None),
    current_user: UserMe = Depends(get_current_user),
):
    # 1. เตรียมข้อมูลพื้นฐาน
    actual_doc_ids = _parse_doc_ids(doc_ids)
    conv_id = conversation_id or str(uuid.uuid4())
    history = await get_recent_history(current_user.id, conv_id, limit=4)
    
    # 2. ตรวจสอบ Intent (ใช้ logic เดิมของคุณหรือ detect_intent)
    # ในที่นี้ใช้ logic keywords ตามที่คุณเขียนมา
    q_lower = question.lower()
    analysis_keywords = ["วิเคราะห์", "pdca", "จุดแข็ง", "ช่องว่าง", "ผ่าน level", "ประเมินหลักฐาน"]
    
    if any(kw in q_lower for kw in analysis_keywords) and actual_doc_ids:
        return await analysis_llm(question, actual_doc_ids, doc_types, enabler, subject, conv_id, current_user)

    # 3. เรียกใช้ retrieve_context_for_endpoint (หัวใจสำคัญที่เปลี่ยน)
    vsm = get_vectorstore_manager(tenant=current_user.tenant)
    
    # ดึงข้อมูลเพียงครั้งเดียวด้วย Endpoint Logic
    retrieval_result = await asyncio.to_thread(
        retrieve_context_for_endpoint,
        vectorstore_manager=vsm,
        query=question,
        tenant=current_user.tenant,
        stable_doc_ids=set(actual_doc_ids) if actual_doc_ids else None,
        doc_type=doc_types[0] if doc_types else EVIDENCE_DOC_TYPES,
        enabler=enabler or DEFAULT_ENABLER,
        subject=subject,
        k_to_retrieve=25,
        k_to_rerank=QUERY_FINAL_K # ใช้ค่าจาก global_vars
    )

    context_text = retrieval_result.get("aggregated_context", "ไม่พบข้อมูลที่เกี่ยวข้อง")
    top_evidences = retrieval_result.get("top_evidences", [])

    if not top_evidences:
        raise HTTPException(status_code=400, detail="ไม่พบข้อมูลที่เกี่ยวข้องในเอกสารที่เลือก")

    # 4. เตรียม Prompt และเรียก LLM
    llm = create_llm_instance(model_name=DEFAULT_LLM_MODEL_NAME, temperature=LLM_TEMPERATURE)
    prompt_text = QA_PROMPT_TEMPLATE.format(context=context_text, question=question)

    messages = [
        SystemMessage(content="ALWAYS ANSWER IN THAI.\n" + SYSTEM_QA_INSTRUCTION),
        HumanMessage(content=prompt_text),
    ]

    raw = await asyncio.to_thread(llm.invoke, messages)
    answer = enforce_thai_primary_language(raw.content if hasattr(raw, "content") else str(raw))

    # 5. บันทึกประวัติและส่งคำตอบ
    await async_save_message(current_user.id, conv_id, "user", question)
    await async_save_message(current_user.id, conv_id, "ai", answer)

    # แปลงรูปแบบ sources ให้เข้ากับ QuerySource model
    sources = [
        QuerySource(
            source_id=str(e["doc_id"]),
            file_name=e["source"],
            chunk_text=e["text"][:500],
            chunk_id=e["chunk_uuid"],
            score=float(e["score"])
        ) for e in top_evidences
    ]

    return QueryResponse(
        answer=answer.strip(), 
        sources=sources, 
        conversation_id=conv_id
    )

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
    used_enabler = enabler or DEFAULT_ENABLER
    conv_id = str(uuid.uuid4())

    vsm = get_vectorstore_manager(tenant=current_user.tenant)
    llm = create_llm_instance(model_name=DEFAULT_LLM_MODEL_NAME, temperature=LLM_TEMPERATURE)

    # Retrieval for all selected IDs
    all_chunks = await _get_context_chunks(
        question=question,
        doc_types=used_doc_types,
        stable_doc_ids=set(doc_ids),
        enabler=used_enabler,
        subject=None,
        vsm=vsm,
        user=current_user,
    )

    doc_groups = defaultdict(list)
    for c in all_chunks:
        if isinstance(c, dict):
            did = str(c.get("doc_id"))
            txt = c.get("text", "")
            src = c.get("source", "Unknown")
        else:
            did = str(c.metadata.get("stable_doc_uuid") or c.metadata.get("doc_id"))
            txt = c.page_content
            src = c.metadata.get("source", "Unknown")
        doc_groups[did].append({"text": txt, "source": src})

    doc_blocks = []
    for idx, did in enumerate(doc_ids, start=1):
        items = doc_groups.get(str(did), [])
        if not items:
            block = f"### เอกสารที่ {idx}\n(ไม่พบข้อมูลในเอกสารนี้)"
        else:
            body = "\n".join(f"- {i['text']}" for i in items[:10])
            block = f"### เอกสารที่ {idx}: {items[0]['source']}\n{body}"
        doc_blocks.append(block)

    prompt_text = COMPARE_PROMPT_TEMPLATE.format(documents_content="\n\n".join(doc_blocks), query=question)
    messages = [
        SystemMessage(content=SYSTEM_COMPARE_INSTRUCTION),
        HumanMessage(content=prompt_text),
    ]

    raw = await asyncio.to_thread(llm.invoke, messages)
    answer = enforce_thai_primary_language(raw.content if hasattr(raw, "content") else str(raw))

    await async_save_message(current_user.id, conv_id, "user", question)
    await async_save_message(current_user.id, conv_id, "ai", answer)

    return QueryResponse(answer=answer.strip(), sources=_map_sources(all_chunks[:10]), conversation_id=conv_id)


# =====================================================================
# 3. /analysis — SE-AM Analysis with JSON Rubrics
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
    used_enabler = enabler or DEFAULT_ENABLER
    stable_doc_ids = set(doc_ids) if doc_ids else None

    vsm = get_vectorstore_manager(tenant=current_user.tenant)
    llm = create_llm_instance(model_name=DEFAULT_LLM_MODEL_NAME, temperature=LLM_TEMPERATURE)

    # 1. Load JSON Rubrics (หัวใจของการฟันธง Level)
    rubric_json_str = "{}"
    try:
        rubric_path = get_rubric_file_path(current_user.tenant, used_enabler)
        if os.path.exists(rubric_path):
            with open(rubric_path, 'r', encoding='utf-8') as f:
                rubric_data = json.load(f)
                rubric_json_str = json.dumps(rubric_data, ensure_ascii=False)
                logger.info(f"✅ Loaded Rubric JSON for {used_enabler}")
    except Exception as e:
        logger.error(f"❌ Failed to load Rubric JSON: {e}")

    # 2. Retrieval Phase
    all_chunks = await _get_context_chunks(
        question=question,
        doc_types=used_doc_types,
        stable_doc_ids=stable_doc_ids,
        enabler=used_enabler,
        subject=subject,
        vsm=vsm,
        user=current_user,
        rubric_vectorstore_name="seam", # ใช้คอลเลกชันกลางช่วยค้นหาเกณฑ์
    )

    if not all_chunks:
        raise HTTPException(400, "ไม่พบข้อมูลสำหรับวิเคราะห์")

    # 3. Normalize Data for Engine
    evidences = []
    for c in all_chunks:
        if isinstance(c, dict):
            evidences.append(c)
        else:
            m = c.metadata or {}
            evidences.append({
                "text": c.page_content,
                "source": m.get("source") or "Unknown",
                "doc_id": m.get("doc_id") or m.get("stable_doc_uuid"),
                "pdca_tag": m.get("pdca_tag", "Other"),
            })

    # 4. PDCA Engine Grouping
    class ConfigObj:
        def __init__(self, **entries): self.__dict__.update(entries)
        def __getattr__(self, name): return None

    engine_config = ConfigObj(
        tenant=current_user.tenant, year=current_user.year,
        enabler=used_enabler, target_level=5
    )
    engine = SEAMPDCAEngine(config=engine_config, llm_instance=llm, vectorstore_manager=vsm)

    p_b, d_b, c_b, a_b, o_b = engine._get_pdca_blocks_from_evidences(
        evidences=evidences, baseline_evidences={}, level=5,
        sub_id=subject or "all", contextual_rules_map=engine.contextual_rules_map
    )

    # 5. Final LLM Prompting with Rubric JSON
    pdca_context = "\n\n".join(filter(None, [p_b, d_b, c_b, a_b, o_b]))
    final_prompt = ANALYSIS_PROMPT.format(
        rubric_json=rubric_json_str, 
        documents_content=pdca_context, 
        question=question
    )

    messages = [
        SystemMessage(content=SYSTEM_ANALYSIS_INSTRUCTION),
        HumanMessage(content=final_prompt),
    ]

    raw = await asyncio.to_thread(llm.invoke, messages)
    answer = enforce_thai_primary_language(raw.content if hasattr(raw, "content") else str(raw))

    await async_save_message(current_user.id, conv_id, "user", question)
    await async_save_message(current_user.id, conv_id, "ai", answer)

    return QueryResponse(answer=answer.strip(), sources=_map_sources(all_chunks[:10]), conversation_id=conv_id)