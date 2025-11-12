# llm_router.py (Optimized Version with History & Metadata Fix)

import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Form, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
import uuid
import json # ต้อง import json ในระดับบนสุดเพื่อใช้ใน /compare

# Langchain Imports
from langchain_core.documents import Document as LcDocument
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage

# *** นำเข้า History Utils ***
# สมมติว่าไฟล์เหล่านี้อยู่ในโครงสร้างโปรเจกต์ของคุณ
from core.history_utils import load_conversation_history, save_message 
# **********************

# สมมติว่าไฟล์เหล่านี้ถูก import ได้อย่างถูกต้อง
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
    file_name: str 
    chunk_text: str
    chunk_id: Optional[str] = None
    score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[QuerySource]
    conversation_id: str = Field(..., example="conv-uuid-456")

class MetricResult(BaseModel):
    metric: str
    doc1: str | None = None
    doc2: str | None = None
    delta: str | List[dict] | None = None
    remark: str | None = None

class CompareResults(BaseModel):
    metrics: List[MetricResult] = Field(default_factory=list)
    overall_summary: str | None = None

class CompareResponse(BaseModel):
    result: CompareResults
    status: str = "success"

# -----------------------------
# --- /query Endpoint ---
# -----------------------------
@llm_router.post("/query", response_model=QueryResponse)
async def query_llm(
    question: str = Form(...),
    doc_ids: Optional[List[str]] = Form(None),
    doc_types: Optional[List[str]] = Form(None),
    enabler: Optional[str] = Form(None),
    conversation_id: Optional[str] = Form(None), 
):
    if llm_instance is None:
        raise HTTPException(status_code=503, detail="LLM service unavailable")

    # 1. การกำหนดค่าเริ่มต้น
    enabler = enabler or DEFAULT_ENABLER
    doc_ids = doc_ids or []
    doc_types = doc_types or EVIDENCE_DOC_TYPES

    # 2. ตรรกะ Conversation ID และ History
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
    
    try:
        await run_in_threadpool(lambda: save_message(conversation_id, 'user', question))
        history_messages = await run_in_threadpool(lambda: load_conversation_history(conversation_id))
    except Exception as e:
        logger.error(f"History operation failed: {e}")
        history_messages = []

    # 3. Guardrails & Intent
    augmented_question = augment_seam_query(question)
    intent = detect_intent(augmented_question)

    # 4. Retrieve relevant chunks
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
            logger.error(f"Retrieval error for {d_type}: {e}", exc_info=True)
            
    # 5. Fallback ถ้าไม่มี context
    if not all_chunks_raw:
        # Build messages including history for pure LLM call
        messages = [SystemMessage(content=SYSTEM_QA_INSTRUCTION)] + history_messages + [HumanMessage(content=augmented_question)]
        llm_obj = await run_in_threadpool(lambda: llm_instance.invoke(messages))
        llm_answer = getattr(llm_obj, "content", str(llm_obj)).strip()
        await run_in_threadpool(lambda: save_message(conversation_id, 'ai', llm_answer))
        return QueryResponse(answer=llm_answer, sources=[], conversation_id=conversation_id)

    # 6. Use RAG context & Build Messages
    top_chunks = sorted(all_chunks_raw, key=lambda d: d.metadata.get("score", 0), reverse=True)[:FINAL_K_RERANKED]
    context_text = "\n\n---\n\n".join([f"Source {i+1}: {doc.page_content[:3000]}" for i, doc in enumerate(top_chunks)])
    prompt_text = build_prompt(context_text, augmented_question, intent)

    # รวม History เข้าไปในการเรียก LLM
    messages = [SystemMessage(content=SYSTEM_QA_INSTRUCTION)] + history_messages + [HumanMessage(content=prompt_text)]
    
    # 7. เรียก LLM และบันทึกข้อความ AI
    try:
        llm_answer_obj = await run_in_threadpool(lambda: llm_instance.invoke(messages))
        llm_answer = getattr(llm_answer_obj, "content", str(llm_answer_obj)).strip()
    except Exception as e:
        logger.error(f"LLM error: {e}", exc_info=True)
        llm_answer = "เกิดข้อผิดพลาดในการสร้างคำตอบ"
    
    await run_in_threadpool(lambda: save_message(conversation_id, 'ai', llm_answer))

    # 8. Format structured sources for frontend
    final_sources = [
        QuerySource(
            source_id=doc.metadata.get("stable_doc_uuid", "unknown"),
            file_name=doc.metadata.get("file_name", "Unknown Document"), 
            chunk_text=doc.page_content,
            chunk_id=doc.metadata.get("chunk_uuid"),
            score=doc.metadata.get("score", 0.0)
        )
        for doc in top_chunks
    ]

    return QueryResponse(answer=llm_answer, sources=final_sources, conversation_id=conversation_id)

# -----------------------------
# --- /compare Endpoint ---
# -----------------------------
@llm_router.post(
    "/compare",
    response_model=CompareResponse,
    status_code=status.HTTP_200_OK,
    tags=["LLM Operations"]
)
async def compare_documents(
    doc1_id: str = Form(..., description="ID of the first document (Required)."),
    doc2_id: str = Form(..., description="ID of the second document (Required)."),
    final_query: str = Form(..., description="The specific question/prompt for comparison (Required)."),
    
    # [จุดสำคัญ]: กำหนดให้ doc_type เป็น Optional และมีค่า Default 'document'
    doc_type: Optional[str] = Form('document', description="The type/collection of documents (Default: 'document')."), 
    
    enabler: Optional[str] = Form(None, description="Optional enabler/filter ID."),
):
    """
    Compares two documents using RAG retrieval and an LLM, accepting data via Form Data.
    """
    
    # 1. การจัดการ Enabler และ Doc Type
    enabler = enabler or DEFAULT_ENABLER
    doc_type = doc_type or 'document' # ยืนยันค่า doc_type อีกครั้ง
    
    if llm_instance is None:
        logger.error("LLM service is not initialized.")
        raise HTTPException(status_code=503, detail="LLM service unavailable")

    logger.info(f"/compare | doc1={doc1_id} | doc2={doc2_id} | enabler={enabler} | doc_type={doc_type}")

    # 2. RAG Retrieval Logic
    try:
        # ใช้ doc_type เป็น collection_name ในการเรียก RAG
        context_docs = await run_in_threadpool(lambda: retrieve_context_by_doc_ids(
            doc_uuids=[doc1_id, doc2_id],
            collection_name=doc_type # ใช้ doc_type ที่มีค่า default แล้ว
        ))
    except Exception as e:
        logger.error(f"Retrieval failed during comparison: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"RAG retrieval failed: {e}")

    evidences = context_docs.get("top_evidences", [])
    if not evidences:
        raise HTTPException(status_code=404, detail="Documents not found in RAG collection.")

    # 3. เตรียม Context และ Message สำหรับ LLM
    doc1_text = next((d["content"][:20000] for d in evidences if d.get("doc_id") == doc1_id), "")
    doc2_text = next((d["content"][:20000] for d in evidences if d.get("doc_id") == doc2_id), "")
    
    if not doc1_text or not doc2_text:
        raise HTTPException(status_code=404, detail="One or both document contents could not be retrieved.")

    human_msg = build_prompt(
        context=f"Doc1:\n{doc1_text}\n\nDoc2:\n{doc2_text}",
        question=final_query,
        intent={"is_synthesis": True, "is_faq": False}
    )

    messages: List[BaseMessage] = [
        SystemMessage(content=SYSTEM_COMPARE_INSTRUCTION),
        HumanMessage(content=human_msg)
    ]

    # 4. เรียก LLM
    def call_llm_safe(msgs: List[BaseMessage]) -> str:
        res = llm_instance.invoke(msgs)
        return getattr(res, "content", str(res)).strip()

    json_text = await run_in_threadpool(lambda: call_llm_safe(messages))

    # 5. ประมวลผล JSON Response
    try:
        result_dict = json.loads(json_text)
        
        # คืนค่าตาม CompareResponse Model
        return {"result": result_dict, "status": "success"}
    
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON returned by LLM: {json_text[:200]}...")
        raise HTTPException(
            status_code=500,
            detail="LLM returned invalid JSON. Raw response: " + json_text[:500]
        )
    except Exception as e:
        logger.error(f"Error processing LLM result: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error processing LLM result: {e}")