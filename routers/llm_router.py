# routers/llm_router.py
# Production Ready – 11 ธันวาคม 2568
# รองรับ Form 100% (สำหรับ UI ปัจจุบันของคุณ) + ไม่ error อีกต่อไป

import logging
import uuid
import asyncio
from typing import List, Optional

from fastapi import APIRouter, Form, HTTPException, Depends
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

# LangChain
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document as LcDocument

# Project imports
from core.history_utils import async_save_message, async_load_conversation_history
from core.llm_data_utils import retrieve_context_for_endpoint
from core.vectorstore import get_vectorstore_manager
from core.rag_prompts import SYSTEM_QA_INSTRUCTION
from core.llm_guardrails import detect_intent, build_prompt
from models.llm import create_llm_instance
from routers.auth_router import UserMe, get_current_user

from config.global_vars import (
    DEFAULT_ENABLER,
    EVIDENCE_DOC_TYPES,
    FINAL_K_RERANKED,
    QUERY_INITIAL_K,
    QUERY_FINAL_K,
    DEFAULT_LLM_MODEL_NAME
)

# Logger
logger = logging.getLogger(__name__)

# Router
llm_router = APIRouter(prefix="/api", tags=["LLM"])


# =============================
#    Pydantic Models
# =============================
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


# =============================
#    /query → รับ Form 100% (เหมาะกับ UI ปัจจุบันของคุณ)
# =============================
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
    llm = create_llm_instance(model_name=DEFAULT_LLM_MODEL_NAME, temperature=0.0)
    if not llm:
        raise HTTPException(status_code=503, detail="LLM service unavailable")

    tenant_context = current_user.tenant
    year_context = current_user.year

    # สร้าง conversation ID ถ้ายังไม่มี
    conv_id = conversation_id or str(uuid.uuid4())

    # ค่า default
    enabler = enabler or DEFAULT_ENABLER
    doc_types = doc_types or [EVIDENCE_DOC_TYPES]
    doc_ids = doc_ids or []

    logger.info(f"RAG Query (Form) | User:{current_user.id} | Q:{question[:50]}... | Docs:{doc_types} | Enabler:{enabler}")

    # ใช้ VSM ที่ถูกต้อง
    vsm = get_vectorstore_manager(
        # doc_type="all",
        tenant=tenant_context,
        # year=year_context
    )

    # บันทึกข้อความผู้ใช้
    await async_save_message(conv_id, "user", question)
    history_messages = await async_load_conversation_history(conv_id)

    # ดึง context
    all_chunks: List[LcDocument] = []
    final_doc_set = set(doc_ids)

    tasks = [
        asyncio.to_thread(
            retrieve_context_for_endpoint,
            query=question,
            doc_type=d_type,
            enabler=enabler,
            subject=subject,
            vectorstore_manager=vsm,
            stable_doc_ids=final_doc_set,
            k_to_retrieve=QUERY_INITIAL_K,
            k_to_rerank=QUERY_FINAL_K,
            tenant=tenant_context,
            year=year_context
        )
        for d_type in doc_types
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            logger.warning(f"Retrieval error: {result}")
            continue
        for ev in result.get("top_evidences", []):
            all_chunks.append(LcDocument(
                page_content=ev["text"],
                metadata={
                    "score": float(ev.get("score", 1.0)),
                    "stable_doc_uuid": ev.get("doc_id"),
                    "chunk_uuid": ev.get("chunk_uuid"),
                    "file_name": ev.get("source", "Unknown Document"),
                    "doc_type": ev.get("doc_type"),
                }
            ))

    # ถ้าไม่มี context → ตอบด้วย LLM ล้วน
    if not all_chunks:
        messages = [
            SystemMessage(content=SYSTEM_QA_INSTRUCTION),
            *history_messages,
            HumanMessage(content=question)
        ]
        response = llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)
        await async_save_message(conv_id, "ai", answer)
        return QueryResponse(answer=answer.strip(), sources=[], conversation_id=conv_id)

    # RAG Mode
    top_chunks = sorted(all_chunks, key=lambda x: x.metadata.get("score", 0), reverse=True)[:FINAL_K_RERANKED]

    context = "\n\n---\n\n".join([
        f"Source [{doc.metadata['file_name']} | Score: {doc.metadata['score']:.3f}]:\n{doc.page_content[:3500]}"
        for doc in top_chunks
    ])

    intent = detect_intent(question)
    user_prompt = build_prompt(context, question, intent)

    messages = [
        SystemMessage(content=SYSTEM_QA_INSTRUCTION),
        *history_messages,
        HumanMessage(content=user_prompt)
    ]

    response = llm.invoke(messages)
    answer = response.content if hasattr(response, "content") else str(response)
    await async_save_message(conv_id, "ai", answer)

    sources = [
        QuerySource(
            source_id=doc.metadata.get("stable_doc_uuid", "unknown"),
            file_name=doc.metadata.get("file_name", "Unknown"),
            chunk_text=doc.page_content,
            chunk_id=doc.metadata.get("chunk_uuid"),
            score=doc.metadata.get("score", 0.0)
        )
        for doc in top_chunks
    ]

    logger.info(f"RAG Success | conv:{conv_id[:8]} | chunks:{len(top_chunks)}")
    return QueryResponse(answer=answer.strip(), sources=sources, conversation_id=conv_id)