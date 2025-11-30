# routers/llm_router.py
import logging
import uuid
import asyncio
from typing import List, Optional

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

# LangChain
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document as LcDocument
from langchain_core.output_parsers import PydanticOutputParser

# Project imports (ใช้เวอร์ชันล่าสุดที่เราปรับแล้ว)
from core.history_utils import async_save_message, async_load_conversation_history
from core.llm_data_utils import retrieve_context_with_filter, retrieve_context_by_doc_ids
from core.vectorstore import get_vectorstore_manager
from core.rag_prompts import (
    SYSTEM_QA_INSTRUCTION,
    QA_PROMPT,
    SYSTEM_COMPARE_INSTRUCTION,
    COMPARE_PROMPT
)
from core.llm_guardrails import detect_intent, build_prompt
from models.llm import create_llm_instance

from config.global_vars import (
    DEFAULT_ENABLER,
    EVIDENCE_DOC_TYPES,
    FINAL_K_RERANKED,
    QUERY_INITIAL_K,
    QUERY_FINAL_K,
    LLM_MODEL_NAME
)

logger = logging.getLogger(__name__)
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


# สำหรับ /compare (ใช้ Pydantic Parser → แม่น 100%)
class ComparisonItem(BaseModel):
    metric: str
    doc1: str
    doc2: str
    delta: str
    remark: Optional[str] = ""

class ComparisonOutput(BaseModel):
    metrics: List[ComparisonItem]
    overall_summary: str

class CompareResponse(BaseModel):
    result: ComparisonOutput
    status: str = "success"


# =============================
#    /query → RAG สุดยอด (เร็ว + แม่น + ปลอดภัย)
# =============================
@llm_router.post("/query", response_model=QueryResponse)
async def query_llm(
    request: Request,
    question: str = Form(...),
    doc_ids: Optional[List[str]] = Form(None),
    doc_types: Optional[List[str]] = Form(None),
    enabler: Optional[str] = Form(None),
    conversation_id: Optional[str] = Form(None),
):
    llm = create_llm_instance(model_name=LLM_MODEL_NAME, temperature=0.0)
    if not llm:
        raise HTTPException(status_code=503, detail="LLM service unavailable")

    conversation_id = conversation_id or str(uuid.uuid4())
    enabler = enabler or DEFAULT_ENABLER
    doc_types = doc_types or EVIDENCE_DOC_TYPES
    doc_ids = doc_ids or []

    vsm = get_vectorstore_manager()

    # บันทึก + โหลด history แบบ async (ไม่แข่งกันอีกต่อไป)
    await async_save_message(conversation_id, "user", question)
    history_messages = await async_load_conversation_history(conversation_id)

    # ใช้ guardrails ล่าสุดที่เราปรับให้ฉลาดสุด ๆ
    intent = detect_intent(question)

    # ดึงข้อมูลแบบ parallel → เร็วสุดในสามโลก
    all_chunks: List[LcDocument] = []
    if vsm:
        tasks = [
            run_in_threadpool(
                retrieve_context_with_filter,
                query=question,
                doc_type=d_type,
                enabler=enabler,
                vectorstore_manager=vsm,
                stable_doc_ids=doc_ids,
                # ลบ top_k และ initial_k ออก เพราะ retrieve_context_with_filter ใช้ Global Var โดยตรงแล้ว
                # top_k=QUERY_FINAL_K,
                # initial_k=QUERY_INITIAL_K
            )
            for d_type in doc_types
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Retrieval failed for a doc_type: {result}")
                continue
            for ev in result.get("top_evidences", []):
                all_chunks.append(LcDocument(
                    page_content=ev["text"],
                    metadata={
                        "score": float(ev.get("score", 0.0)),
                        "stable_doc_uuid": ev.get("doc_id"),
                        "chunk_uuid": ev.get("chunk_uuid"),
                        "file_name": ev.get("source", "Unknown Document"),
                        "doc_type": ev.get("doc_type"),
                    }
                ))

    # Fallback: Pure LLM (ถ้าไม่มี vectorstore)
    if not all_chunks:
        messages = [
            SystemMessage(content=SYSTEM_QA_INSTRUCTION),
            *history_messages,
            HumanMessage(content=question)
        ]
        response = await run_in_threadpool(llm.invoke, messages)
        answer = getattr(response, "content", str(response)).strip()
        await async_save_message(conversation_id, "ai", answer)
        return QueryResponse(answer=answer, sources=[], conversation_id=conversation_id)

    # RAG Mode → ใช้ prompt ล่าสุดที่เราปรับให้ผู้บริหารรัก
    top_chunks = sorted(all_chunks, key=lambda x: x.metadata.get("score", 0), reverse=True)[:FINAL_K_RERANKED]

    context = "\n\n---\n\n".join([
        f"Source [{doc.metadata['file_name']} | Score: {doc.metadata['score']:.3f}]:\n{doc.page_content[:3500]}"
        for doc in top_chunks
    ])

    user_prompt = build_prompt(context, question, intent)
    messages = [
        SystemMessage(content=SYSTEM_QA_INSTRUCTION),
        *history_messages,
        HumanMessage(content=user_prompt)
    ]

    response = await run_in_threadpool(llm.invoke, messages)
    answer = getattr(response, "content", str(response)).strip()
    await async_save_message(conversation_id, "ai", answer)

    sources = [
        QuerySource(
            source_id=doc.metadata.get("stable_doc_uuid", "unknown"),
            file_name=doc.metadata.get("file_name", "Unknown Document"),
            chunk_text=doc.page_content,
            chunk_id=doc.metadata.get("chunk_uuid"),
            score=doc.metadata.get("score", 0.0)
        )
        for doc in top_chunks
    ]

    logger.info(f"RAG Query Success | conv:{conversation_id[:8]} | chunks:{len(top_chunks)} | intent:{intent}")
    return QueryResponse(answer=answer, sources=sources, conversation_id=conversation_id)


# =============================
#    /compare → ใช้ Pydantic Parser → ไม่พังอีกต่อไป!
# =============================
@llm_router.post("/compare", response_model=CompareResponse)
async def compare_documents(
    doc1_id: str = Form(...),
    doc2_id: str = Form(...),
    final_query: str = Form("เปรียบเทียบเอกสารทั้งสองฉบับอย่างละเอียด"),
    doc_type: str = Form("document"),
    enabler: str = Form("KM")
):
    llm = create_llm_instance(model_name=LLM_MODEL_NAME, temperature=0.0)
    vsm = get_vectorstore_manager()
    if not vsm:
        raise HTTPException(503, "Vector store not available")

    docs = await run_in_threadpool(
        retrieve_context_by_doc_ids,
        doc_uuids=[doc1_id, doc2_id],
        doc_type=doc_type,
        enabler=enabler,
        vectorstore_manager=vsm
    )

    evidences = docs.get("top_evidences", [])
    if len(evidences) < 2:
        raise HTTPException(404, "One or both documents not found")

    doc_map = {}
    for ev in evidences:
        doc_map.setdefault(ev["doc_id"], []).append(ev["content"])

    doc1_text = "\n\n".join(doc_map.get(doc1_id, []))[:18000]
    doc2_text = "\n\n".join(doc_map.get(doc2_id, []))[:18000]

    if not doc1_text or not doc2_text:
        raise HTTPException(404, "Document content is empty")

    # ใช้ Pydantic Parser → แม่น 100%
    parser = PydanticOutputParser(pydantic_object=ComparisonOutput)
    format_instructions = parser.get_format_instructions()

    prompt = COMPARE_PROMPT.format(
        doc1_content=doc1_text,
        doc2_content=doc2_text,
        query=final_query
    ) + "\n\n" + format_instructions

    messages = [
        SystemMessage(content=SYSTEM_COMPARE_INSTRUCTION),
        HumanMessage(content=prompt)
    ]

    response = await run_in_threadpool(llm.invoke, messages)
    raw_output = getattr(response, "content", str(response)).strip()

    try:
        parsed = parser.parse(raw_output)
    except Exception as e:
        logger.error(f"Comparison parser failed:\n{raw_output}\nError: {e}")
        raise HTTPException(500, "Failed to parse comparison result from LLM")

    return CompareResponse(result=parsed)