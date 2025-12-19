# -*- coding: utf-8 -*-
# routers/llm_router.py - Enterprise RAG (Query + Compare)

import logging
import uuid
import asyncio
from typing import List, Optional, Any, Dict, Union, Set
from collections import defaultdict

from fastapi import APIRouter, Form, HTTPException, Depends
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document as LcDocument

# ================= Core Imports =================
from core.history_utils import async_save_message
from core.llm_data_utils import retrieve_context_for_endpoint, _create_where_filter
from core.vectorstore import get_vectorstore_manager
from core.llm_guardrails import (
    detect_intent,
    build_prompt,
    enforce_thai_primary_language,   # ใช้เฉพาะ /query
)
from core.rag_prompts import (
    SYSTEM_QA_INSTRUCTION,
    SYSTEM_ANALYSIS_INSTRUCTION,
    COMPARE_PROMPT,
)

# ================= Models & Config =================
from models.llm import create_llm_instance
from routers.auth_router import UserMe, get_current_user
from config.global_vars import (
    EVIDENCE_DOC_TYPES,
    DEFAULT_ENABLER,
    DEFAULT_LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    QUERY_FINAL_K,
)

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
# Helper: Load ALL chunks by doc_ids (NO semantic search)
# =====================================================================

def load_all_chunks_by_doc_ids(
    vectorstore_manager,
    collection_name: str,
    stable_doc_ids: Union[Set[str], List[str]],
) -> List[LcDocument]:
    """
    ใช้เฉพาะ /compare
    - โหลดทุก chunk ของเอกสารที่เลือก
    - ใช้ metadata filter เท่านั้น
    - ไม่พึ่ง semantic similarity
    """

    chroma = vectorstore_manager._load_chroma_instance(collection_name)
    if not chroma:
        logger.warning("Chroma collection not found: %s", collection_name)
        return []

    where_filter = _create_where_filter(
        stable_doc_ids=set(stable_doc_ids)
    )

    docs = chroma.similarity_search(
        query="*",
        k=9999,
        filter=where_filter,
    )

    return [
        d for d in docs
        if getattr(d, "page_content", "").strip()
    ]


# =====================================================================
# 1. /query — General RAG QA (ENFORCE THAI PRIMARY)
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
    llm = create_llm_instance(
        model_name=DEFAULT_LLM_MODEL_NAME,
        temperature=LLM_TEMPERATURE,
    )

    conv_id = conversation_id or str(uuid.uuid4())
    q_lower = question.lower()

    # ---------- Auto route to /compare ----------
    if any(w in q_lower for w in ["compare", "เปรียบเทียบ", "vs", "ความแตกต่าง"]):
        return await compare_llm(
            question=question,
            doc_ids=doc_ids,
            doc_types=doc_types,
            current_user=current_user,
        )

    vsm = get_vectorstore_manager(tenant=current_user.tenant)

    all_chunks = await _get_context_chunks(
        question=question,
        doc_types=doc_types or [EVIDENCE_DOC_TYPES],
        stable_doc_ids=set(doc_ids) if doc_ids else None,
        enabler=enabler or DEFAULT_ENABLER,
        subject=subject,
        vsm=vsm,
        user=current_user,
    )

    context_text = "\n\n".join(
        f"[{c.metadata.get('source')}]\n{c.page_content}"
        for c in all_chunks
    )

    intent = detect_intent(question)
    base_instruction = (
        SYSTEM_ANALYSIS_INSTRUCTION
        if intent.get("is_analysis")
        else SYSTEM_QA_INSTRUCTION
    )

    messages = [
        SystemMessage(content="ALWAYS ANSWER IN THAI.\n" + base_instruction),
        HumanMessage(content=build_prompt(context_text, question, intent)),
    ]

    raw = await asyncio.to_thread(llm.invoke, messages)
    answer = raw.content if hasattr(raw, "content") else str(raw)

    # ✅ ENFORCE ภาษาไทยเป็นหลัก (เฉพาะ /query)
    answer = enforce_thai_primary_language(answer)

    await async_save_message(conv_id, "user", question)
    await async_save_message(conv_id, "ai", answer)

    return QueryResponse(
        answer=answer.strip(),
        sources=_map_sources(all_chunks),
        conversation_id=conv_id,
    )


# =====================================================================
# 2. /compare — Deterministic Document Comparison
# ❗ NO LANGUAGE BLOCK (table-driven, evidence-based)
# =====================================================================

@llm_router.post("/compare", response_model=QueryResponse)
async def compare_llm(
    question: str = Form(...),
    doc_ids: List[str] = Form(...),
    doc_types: Optional[List[str]] = Form(None),
    current_user: UserMe = Depends(get_current_user),
):
    if not doc_ids or len(doc_ids) < 2:
        raise HTTPException(400, "ต้องเลือกอย่างน้อย 2 เอกสาร")

    vsm = get_vectorstore_manager(tenant=current_user.tenant)
    llm = create_llm_instance(
        model_name=DEFAULT_LLM_MODEL_NAME,
        temperature=LLM_TEMPERATURE,
    )

    collection_name = doc_types[0] if doc_types else EVIDENCE_DOC_TYPES

    all_chunks = load_all_chunks_by_doc_ids(
        vectorstore_manager=vsm,
        collection_name=collection_name,
        stable_doc_ids=doc_ids,
    )

    if not all_chunks:
        raise HTTPException(
            400,
            "ไม่พบ chunk ใด ๆ ในเอกสารที่เลือก (metadata filter ไม่ match)",
        )

    # ---------- Group by document ----------
    doc_groups = defaultdict(list)
    for d in all_chunks:
        doc_key = (
            str(d.metadata.get("stable_doc_uuid"))
            or str(d.metadata.get("doc_id"))
        )
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

    user_prompt = COMPARE_PROMPT.format(
        documents_content="\n\n".join(doc_blocks),
        query=question,
    )

    messages = [
        SystemMessage(
            content=(
                "คุณเป็นผู้ประเมิน SE-AM ระดับผู้เชี่ยวชาญ\n"
                "ให้สรุปผลเป็นตาราง Markdown\n"
                "อธิบายเป็นภาษาไทยเป็นหลัก\n"
                "อนุญาตคำอังกฤษเชิงโครงสร้าง เช่น PDCA, KPI, Plan, Do\n"
                "ห้ามสรุปนอกเหนือจากเอกสาร"
            )
        ),
        HumanMessage(content=user_prompt),
    ]

    raw = await asyncio.to_thread(llm.invoke, messages)
    answer = raw.content if hasattr(raw, "content") else str(raw)

    # ❌ ไม่ enforce ภาษา สำหรับ /compare (ตั้งใจ)
    return QueryResponse(
        answer=answer.strip(),
        sources=_map_sources(all_chunks[:10]),
        conversation_id=str(uuid.uuid4()),
    )


# =====================================================================
# Helpers
# =====================================================================

async def _get_context_chunks(
    question,
    doc_types,
    stable_doc_ids,
    enabler,
    subject,
    vsm,
    user,
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
