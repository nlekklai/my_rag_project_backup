from fastapi import APIRouter, HTTPException, Form, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import List, Optional
import os, json, logging
from langchain.schema import Document as LcDocument, SystemMessage, HumanMessage
from datetime import datetime

from core.retrieval_utils import retrieve_context_with_filter, retrieve_context_by_doc_ids
from core.rag_prompts import QA_PROMPT, COMPARE_PROMPT, SYSTEM_QA_INSTRUCTION, SYSTEM_COMPARE_INSTRUCTION
from core.vectorstore import FINAL_K_RERANKED
from core.ingest import DEFAULT_ENABLER
from models.llm import llm as llm_instance

# -----------------------------
# --- Config & Router Setup ---
# -----------------------------
logger = logging.getLogger(__name__)
llm_router = APIRouter(prefix="/api", tags=["LLM"])


# -----------------------------
# --- Models ---
# -----------------------------
class QuerySource(BaseModel):
    source_id: str
    file_name: str
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
    background_tasks: BackgroundTasks,
    question: str = Form(...),
    conversation_id: Optional[str] = Form(None),
    doc_ids: Optional[List[str]] = Form(None),
    doc_types: Optional[List[str]] = Form(None),
    enabler: Optional[str] = Form(None)
):
    """RAG query endpoint"""
    # --- Validate Inputs ---
    enabler = enabler or DEFAULT_ENABLER
    doc_ids = doc_ids or []
    doc_types = doc_types or ["document"]

    if not question:
        raise HTTPException(status_code=400, detail="Missing question")

    if llm_instance is None:
        raise HTTPException(status_code=503, detail="LLM service unavailable")

    # --- Log incoming query ---
    logger.info(f"🧠 /query received | question='{question[:60]}...' | doc_ids={doc_ids or 'All'} | doc_types={doc_types} | enabler={enabler}")

    # --- Retrieve relevant chunks ---
    all_chunks_raw: List[LcDocument] = []

    for d_type in doc_types:
        try:
            collection_name = f"{d_type}_{enabler.lower()}" if d_type == "evidence" else d_type.lower()
            vectorstore_path = os.path.join("vectorstore", collection_name)

            if not os.path.exists(vectorstore_path):
                logger.warning(f"⚠️ Skipping {collection_name} — vectorstore not found")
                continue

            result = await run_in_threadpool(lambda: retrieve_context_with_filter(
                query=question,
                doc_type=d_type,
                enabler=enabler,
                stable_doc_ids=doc_ids,
                top_k_reranked=FINAL_K_RERANKED
            ))

            evidences = result.get("top_evidences", [])
            logger.info(f"📄 Retrieved {len(evidences)} chunks from {collection_name}")

            for e in evidences:
                all_chunks_raw.append(LcDocument(page_content=e["content"], metadata=e["metadata"]))
        except Exception as e:
            logger.error(f"❌ Retrieval error for {d_type}: {e}", exc_info=True)

    # --- Handle no context case ---
    if not all_chunks_raw:
        logger.warning("⚠️ No relevant documents found for this query.")
        return QueryResponse(answer="ไม่พบข้อมูลที่เกี่ยวข้อง", sources=[])

    # --- Build context ---
    context_text = "\n\n---\n\n".join([
        f"Source {i+1}: {doc.page_content}" for i, doc in enumerate(all_chunks_raw)
    ])
    human_prompt = QA_PROMPT.format(context=context_text, question=question)
    messages = [
        SystemMessage(content=SYSTEM_QA_INSTRUCTION),
        HumanMessage(content=human_prompt)
    ]

    # --- Generate answer via LLM ---
    try:
        logger.info("🚀 Invoking LLM for answer generation...")
        llm_answer = await run_in_threadpool(lambda: llm_instance.invoke(messages))
        llm_answer = getattr(llm_answer, "content", str(llm_answer)).strip()
        logger.info("✅ LLM generation complete.")
    except Exception as e:
        logger.error(f"❌ LLM error: {e}", exc_info=True)
        llm_answer = "เกิดข้อผิดพลาดในการสร้างคำตอบ"

    # --- Prepare sources ---
    final_sources = [
        QuerySource(
            source_id=doc.metadata.get("doc_id", "N/A"),
            file_name=doc.metadata.get("file_name", "N/A"),
            chunk_text=doc.page_content,
            chunk_id=doc.metadata.get("chunk_uuid", "N/A"),
            score=doc.metadata.get("score", 0.0)
        )
        for doc in all_chunks_raw
    ]
    final_sources.sort(key=lambda x: x.score, reverse=True)

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
    """Compare two documents and let the LLM summarize differences"""
    enabler = enabler or DEFAULT_ENABLER

    if llm_instance is None:
        raise HTTPException(status_code=503, detail="LLM service unavailable")

    logger.info(f"🔍 /compare | doc1={doc1_id} | doc2={doc2_id} | enabler={enabler} | doc_type={doc_type}")

    # --- Retrieve documents for comparison ---
    try:
        context_docs = await run_in_threadpool(lambda: retrieve_context_by_doc_ids(
            doc_ids=[doc1_id, doc2_id],
            doc_type=doc_type,
            enabler=enabler
        ))
    except Exception as e:
        logger.error(f"❌ Retrieval failed during comparison: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"RAG retrieval failed: {e}")

    if not context_docs:
        raise HTTPException(status_code=404, detail="Documents not found")

    doc1_text = next((d.page_content for d in context_docs if d.metadata.get("doc_id") == doc1_id), "")
    doc2_text = next((d.page_content for d in context_docs if d.metadata.get("doc_id") == doc2_id), "")

    human_msg = COMPARE_PROMPT.format(
        doc1_content=doc1_text[:20000],
        doc2_content=doc2_text[:20000],
        query=final_query
    )
    messages = [
        SystemMessage(content=SYSTEM_COMPARE_INSTRUCTION),
        HumanMessage(content=human_msg)
    ]

    def call_llm_safe(msgs):
        res = llm_instance.invoke(msgs)
        return getattr(res, "content", str(res)).strip()

    logger.info("⚖️ Comparing documents via LLM...")
    json_text = await run_in_threadpool(lambda: call_llm_safe(messages))

    try:
        result = json.loads(json_text)
        logger.info("✅ Comparison complete.")
        return {"result": result}
    except json.JSONDecodeError:
        logger.error(f"❌ Invalid JSON returned by LLM: {json_text[:200]}...")
        raise HTTPException(status_code=500, detail="LLM returned invalid JSON")
