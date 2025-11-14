#router/llm_router.py
import logging
from typing import List, Optional, Dict, Any, Tuple
from fastapi import APIRouter, Form, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
import uuid
import json 
import re 

# Langchain Imports
from langchain_core.documents import Document as LcDocument
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_core.prompts import PromptTemplate

# *** นำเข้า History Utils ***
from core.history_utils import load_conversation_history, save_message 
# **********************

# Project Imports
from core.retrieval_utils import retrieve_context_with_filter, retrieve_context_by_doc_ids, normalize_stable_ids
from core.llm_guardrails import augment_seam_query, detect_intent, build_prompt
from core.rag_prompts import (
    QA_PROMPT, 
    COMPARE_PROMPT, 
    SYSTEM_QA_INSTRUCTION, 
    SYSTEM_COMPARE_INSTRUCTION
)
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
# --- COMPARE UTILITY FUNCTION ---
# -----------------------------

def get_summary_for_comparison(doc1_text: str, doc2_text: str, final_query: str) -> Tuple[str, str]:
    """
    สร้าง Prompt สำหรับสั่งให้ LLM เปรียบเทียบสองเอกสาร โดยใช้ Prompts จาก rag_prompts.py
    """
    # 🟢 System Prompt: ใช้ SYSTEM_COMPARE_INSTRUCTION โดยตรง
    system_prompt = SYSTEM_COMPARE_INSTRUCTION

    # 🟢 User Prompt: ใช้ COMPARE_PROMPT (LangChain Template)
    compare_template = PromptTemplate.from_template(COMPARE_PROMPT)
    
    user_prompt = compare_template.format(
        doc1_content=doc1_text,
        doc2_content=doc2_text,
        query=final_query
    )
    
    return system_prompt, user_prompt

# -----------------------------
# --- /query Endpoint (ไม่มีการแก้ไข) ---
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
    
    # 💡 ใช้ QA_PROMPT ในการ build prompt
    context_text = "\n\n---\n\n".join([f"Source {i+1}: {doc.page_content[:3000]}" for i, doc in enumerate(top_chunks)])
    
    # แทนที่ build_prompt ด้วยการใช้ QA_PROMPT template
    qa_template = PromptTemplate.from_template(QA_PROMPT)
    prompt_text = qa_template.format(context=context_text, question=augmented_question)

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

# ---------------------------------------------------------------------
# --- /compare Endpoint (Revised & Robust) ---
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# --- /compare Endpoint (Full, Metric List Version) ---
# ---------------------------------------------------------------------
@llm_router.post("/compare", response_model=CompareResponse)
async def compare_documents(
    doc1_id: str = Form(...),
    doc2_id: str = Form(...),
    final_query: str = Form("เปรียบเทียบ 2 เอกสาร"),
    doc_type: str = Form("document"), 
    enabler: str = Form("KM")         
):
    if llm_instance is None:
        raise HTTPException(status_code=503, detail="LLM service unavailable")
        
    # 1️⃣ Retrieve document contents
    try:
        context_docs = await run_in_threadpool(lambda: retrieve_context_by_doc_ids(
            doc_uuids=[doc1_id, doc2_id],
            doc_type=doc_type,
            enabler=enabler
        ))
    except Exception as e:
        logger.error(f"Retrieval failed for /compare: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve context from RAG system.")

    evidences = context_docs.get("top_evidences", [])
    if not evidences:
        raise HTTPException(status_code=404, detail="Documents not found in RAG collection.")

    # 2️⃣ Aggregate document text
    UUID_KEY = "doc_id"
    logger.critical(f"🧭 DEBUG COMPARE IDS (RAW): Doc1={doc1_id}, Doc2={doc2_id}")

    doc1_chunks = [d.get("content", "") for d in evidences if d.get(UUID_KEY) == doc1_id]
    doc2_chunks = [d.get("content", "") for d in evidences if d.get(UUID_KEY) == doc2_id]
    doc1_text = "\n\n".join(doc1_chunks)[:20000]
    doc2_text = "\n\n".join(doc2_chunks)[:20000]

    if not doc1_text or not doc2_text:
        logger.error(f"Document contents missing. Doc1 found: {bool(doc1_text)}, Doc2 found: {bool(doc2_text)}")
        raise HTTPException(status_code=404, detail="One or both document contents could not be retrieved.")

    # 3️⃣ Build LLM prompt and Call LLM
    try:
        system_prompt, user_prompt = get_summary_for_comparison(
            doc1_text=doc1_text,
            doc2_text=doc2_text,
            final_query=final_query
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        llm_obj = await run_in_threadpool(lambda: llm_instance.invoke(
            messages,
            format="json" 
        ))
        
        llm_response_str = getattr(llm_obj, "content", str(llm_obj)).strip()
        logger.critical(f"🧭 LLM RAW JSON OUTPUT: {llm_response_str[:500]}")

        # 4️⃣ Parse Result & Normalize JSON
        try:
            llm_json = json.loads(llm_response_str)
        except json.JSONDecodeError:
            logger.error(f"LLM output is not valid JSON: {llm_response_str[:200]}")
            raise HTTPException(status_code=500, detail="LLM output is not a valid JSON structure.")

        metrics_list: List[MetricResult] = []

        # 🟢 รองรับ list of metrics objects
        metrics_data = llm_json.get("metrics") or llm_json.get("metric") or llm_json.get("comparison")

        if isinstance(metrics_data, list):
            for item in metrics_data:
                if not isinstance(item, dict):
                    logger.warning(f"Skipping non-dict metric item: {item}")
                    continue
                metrics_list.append(MetricResult(
                    metric=item.get("metric", "N/A"),
                    doc1=item.get("doc1", "N/A"),
                    doc2=item.get("doc2", "N/A"),
                    delta=item.get("delta"),
                    remark=item.get("remark")
                ))
        elif isinstance(metrics_data, dict):
            # Scenario: dict of dicts {"หัวข้อ": {"doc1": "...", "doc2": "..."}}
            for metric_name, details in metrics_data.items():
                if isinstance(details, dict):
                    metrics_list.append(MetricResult(
                        metric=metric_name,
                        doc1=details.get("doc1", "N/A"),
                        doc2=details.get("doc2", "N/A"),
                        delta=details.get("delta"),
                        remark=details.get("remark")
                    ))
                else:
                    # simple key-value
                    metrics_list.append(MetricResult(
                        metric=metric_name,
                        doc1=str(details),
                        doc2="N/A"
                    ))

        # 5️⃣ Handle overall summary
        overall_summary_raw = llm_json.get("overall_summary")
        summary_text = None
        if isinstance(overall_summary_raw, str):
            summary_text = overall_summary_raw
        elif isinstance(overall_summary_raw, dict):
            summary_text = " | ".join(f"{k}: {v}" for k, v in overall_summary_raw.items())
        elif isinstance(overall_summary_raw, list) and overall_summary_raw:
            first = overall_summary_raw[0]
            if isinstance(first, dict):
                summary_text = " | ".join(f"{k}: {v}" for k, v in first.items())
            else:
                summary_text = str(first)

        final_result = CompareResults(
            metrics=metrics_list,
            overall_summary=summary_text
        )

        logger.critical(f"🧭 FINAL METRICS COUNT: {len(metrics_list)}")
        return {"status": "success", "result": final_result}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"LLM Processing failed for /compare: {e}", exc_info=True)
        if "validation errors for CompareResults" in str(e):
            raise HTTPException(
                status_code=500, 
                detail="LLM processing failed: Pydantic Validation Error. Check LLM output schema mapping."
            )
        else:
            raise HTTPException(status_code=500, detail=f"LLM processing failed: {e}")
