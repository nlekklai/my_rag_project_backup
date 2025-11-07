# llm_router.py
import logging
from typing import List, Optional, Dict
from fastapi import APIRouter, Form, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from langchain.schema import Document as LcDocument, SystemMessage, HumanMessage

from core.retrieval_utils import retrieve_context_with_filter, retrieve_context_by_doc_ids
from core.llm_guardrails import augment_seam_query, detect_intent, build_prompt
from core.rag_prompts import QA_PROMPT, COMPARE_PROMPT, SYSTEM_QA_INSTRUCTION, SYSTEM_COMPARE_INSTRUCTION
from models.llm import llm as llm_instance
from config.global_vars import DEFAULT_ENABLER, EVIDENCE_DOC_TYPES, FINAL_K_RERANKED

logger = logging.getLogger(__name__)
llm_router = APIRouter(prefix="/api", tags=["LLM"])

# -----------------------------
# --- Pydantic Models ---
# -----------------------------
class QuerySource(BaseModel):
    source_id: str = Field(..., example="doc-uuid-123")
    file_name: str = Field(..., example="SE-AM_IM.pdf")
    chunk_text: str
    chunk_id: str
    score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[QuerySource]

# -----------------------------
# --- /query Endpoint ---
# -----------------------------
@llm_router.post("/query", response_model=QueryResponse)
async def query_llm(
    question: str = Form(...),
    doc_ids: Optional[List[str]] = Form(None),
    doc_types: Optional[List[str]] = Form(None),
    enabler: Optional[str] = Form(None),
):
    if llm_instance is None:
        raise HTTPException(status_code=503, detail="LLM service unavailable")

    enabler = enabler or DEFAULT_ENABLER
    doc_ids = doc_ids or []
    doc_types = doc_types or EVIDENCE_DOC_TYPES

    logger.info(f"/query received | question='{question[:60]}...' | doc_ids={doc_ids or 'All'}")

    # --- Augment question for SEAM codes ---
    augmented_question = augment_seam_query(question)
    intent = detect_intent(augmented_question)

    # --- Retrieve relevant chunks ---
    all_chunks_raw: List[LcDocument] = []
    for d_type in doc_types:
        try:
            result = await run_in_threadpool(lambda: retrieve_context_with_filter(
                query=augmented_question,
                doc_type=d_type,
                enabler=enabler,
                stable_doc_ids=doc_ids,
                top_k_reranked=FINAL_K_RERANKED
            ))
            evidences = result.get("top_evidences", [])
            for e in evidences:
                metadata = e.get("metadata", {})
                metadata["score"] = e.get("score", 0.0)
                all_chunks_raw.append(LcDocument(page_content=e["content"], metadata=metadata))
        except Exception as e:
            logger.error(f"Retrieval error for {d_type}: {e}")

    # --- Fallback if no context ---
    if not all_chunks_raw:
        logger.warning("No relevant documents found, using fallback LLM generation")
        messages = [
            SystemMessage(content=SYSTEM_QA_INSTRUCTION),
            HumanMessage(content=augmented_question)
        ]
        llm_answer = await run_in_threadpool(lambda: getattr(llm_instance.invoke(messages), "content", str(llm_instance.invoke(messages))))
        return QueryResponse(answer=llm_answer, sources=[])

    # --- Build context for LLM ---
    context_text = "\n\n---\n\n".join([f"Source {i+1}: {doc.page_content}" for i, doc in enumerate(all_chunks_raw)])
    prompt_text = build_prompt(context_text, augmented_question, intent)

    messages = [
        SystemMessage(content=SYSTEM_QA_INSTRUCTION),
        HumanMessage(content=prompt_text)
    ]

    try:
        llm_answer_object = await run_in_threadpool(lambda: llm_instance.invoke(messages))
        llm_answer = getattr(llm_answer_object, "content", str(llm_answer_object)).strip()
    except Exception as e:
        logger.error(f"LLM error: {e}")
        llm_answer = "เกิดข้อผิดพลาดในการสร้างคำตอบ"

    # --- Format structured sources for frontend ---
    final_sources = [
        QuerySource(
            source_id=doc.metadata.get("stable_doc_uuid", "N/A"),
            file_name=doc.metadata.get("file_name", "N/A"),
            chunk_text=doc.page_content,
            chunk_id=doc.metadata.get("chunk_uuid", "N/A"),
            score=doc.metadata.get("score", 0.0)
        )
        for doc in all_chunks_raw
    ]

    # --- Return answer and structured sources separately ---
    return QueryResponse(answer=llm_answer, sources=final_sources)

# -----------------------------
# --- /compare Endpoint ---
# -----------------------------
@llm_router.post("/compare")
async def compare_documents(
    doc1_id: str = Form(...),
    doc2_id: str = Form(...),
    final_query: str = Form(...),
    doc_type: str = Form(...),
    enabler: Optional[str] = Form(None)
):
    enabler = enabler or DEFAULT_ENABLER

    if llm_instance is None:
        raise HTTPException(status_code=503, detail="LLM service unavailable")

    logger.info(f"/compare | doc1={doc1_id} | doc2={doc2_id} | enabler={enabler} | doc_type={doc_type}")

    try:
        context_docs = await run_in_threadpool(lambda: retrieve_context_by_doc_ids(
            doc_uuids=[doc1_id, doc2_id],
            collection_name=doc_type
        ))
    except Exception as e:
        logger.error(f"Retrieval failed during comparison: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"RAG retrieval failed: {e}")

    if not context_docs or not context_docs.get("top_evidences"):
        raise HTTPException(status_code=404, detail="Documents not found")

    doc1_text = next((d["content"] for d in context_docs["top_evidences"] if d["doc_id"] == doc1_id), "")
    doc2_text = next((d["content"] for d in context_docs["top_evidences"] if d["doc_id"] == doc2_id), "")

    human_msg = build_prompt(
        context=f"Doc1:\n{doc1_text[:20000]}\n\nDoc2:\n{doc2_text[:20000]}",
        question=final_query,
        intent={"is_synthesis": True, "is_faq": False}  # Always synthesis for compare
    )

    messages = [
        SystemMessage(content=SYSTEM_COMPARE_INSTRUCTION),
        HumanMessage(content=human_msg)
    ]

    def call_llm_safe(msgs):
        res = llm_instance.invoke(msgs)
        return getattr(res, "content", str(res)).strip()

    json_text = await run_in_threadpool(lambda: call_llm_safe(messages))

    try:
        import json
        result = json.loads(json_text)
        return {"result": result}
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON returned by LLM: {json_text[:200]}...")
        raise HTTPException(
            status_code=500,
            detail="LLM returned invalid JSON. Raw response: " + json_text[:500]
        )
