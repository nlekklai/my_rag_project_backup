# -*- coding: utf-8 -*-
# routers/llm_router.py

import logging
import uuid
import asyncio
import json
import re
from typing import List, Optional, Any, Dict

from fastapi import APIRouter, Form, HTTPException, Depends
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document as LcDocument

# Core Imports
from core.history_utils import async_save_message
from core.llm_data_utils import retrieve_context_for_endpoint
from core.vectorstore import get_vectorstore_manager

# Import Prompts ที่ผ่านการ Revise (รวมกฎเหล็กเดิม + PDCA + General Chat)
from core.rag_prompts import (
    SYSTEM_QA_INSTRUCTION, 
    SYSTEM_ANALYSIS_INSTRUCTION,
    SYSTEM_GENERAL_CHAT_INSTRUCTION,
    QA_PROMPT,
    SYSTEM_COMPARE_INSTRUCTION,
    COMPARE_PROMPT
)
from core.llm_guardrails import detect_intent, build_prompt 

from models.llm import create_llm_instance
from routers.auth_router import UserMe, get_current_user

# Global Config
from config.global_vars import (
    DEFAULT_ENABLER,
    EVIDENCE_DOC_TYPES,
    FINAL_K_RERANKED,
    QUERY_INITIAL_K,
    QUERY_FINAL_K,
    DEFAULT_LLM_MODEL_NAME,
    LLM_TEMPERATURE
)

logger = logging.getLogger(__name__)
llm_router = APIRouter(prefix="/api", tags=["LLM"])

# --- Schema สำหรับ Response ---
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

# ===================================================================
# 1. /query - รองรับ FAQ, KM Chat, General Chat, และ PDCA Analysis
# ===================================================================
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
    
    # [1] Detect Intent เพื่อแยกประเภทคำถาม (FAQ / Analysis / Chat)
    intent = detect_intent(question)
    detected_sub_topic = intent.get("sub_topic")

    # [2] Retrieval Context (ใช้ Sub-topic ช่วยกรองถ้ามี)
    vsm = get_vectorstore_manager(tenant=current_user.tenant)
    all_chunks = await _get_context_chunks(
        question, doc_types or [EVIDENCE_DOC_TYPES], doc_ids or [], 
        enabler or DEFAULT_ENABLER, subject, vsm, current_user,
        sub_topic=detected_sub_topic
    )

    # [3] เตรียม Context Text
    context_text = "\n\n---\n\n".join([
        f"Source [{d.metadata.get('source', 'Unknown')}]:\n{d.page_content}" 
        for d in all_chunks
    ])
    
    # [4] **Intent Switching Logic** - เลือก System Instruction ตามประเภทคำถาม
    if intent.get("is_analysis"):
        # โหมดวิเคราะห์หลักฐาน/ประเมิน (PDCA)
        base_system_instruction = SYSTEM_ANALYSIS_INSTRUCTION
    elif intent.get("is_faq") or intent.get("is_evidence") or intent.get("sub_topic"):
        # โหมด KM / SE-AM / ข้อมูลเชิงลึกจากเอกสาร (ใช้กฎเหล็ก Original)
        base_system_instruction = SYSTEM_QA_INSTRUCTION
    else:
        # โหมด Chat ทั่วไป / FAQ สวัสดิการพนักงาน
        base_system_instruction = SYSTEM_GENERAL_CHAT_INSTRUCTION

    # 🟢 บังคับภาษาไทย (Strict Thai)
    strict_thai_instruction = (
        "ALWAYS ANSWER IN THAI LANGUAGE. คุณต้องตอบเป็นภาษาไทยเท่านั้น\n" 
        + base_system_instruction
    )

    # ใช้ build_prompt เพื่อประกอบ User Message (ใส่กฎเหล็กเรื่อง Subtopic และ Source)
    user_prompt_content = build_prompt(context_text, question, intent) 

    messages = [
        SystemMessage(content=strict_thai_instruction),
        HumanMessage(content=user_prompt_content)
    ]

    # [5] เรียก LLM
    response = llm.invoke(messages)
    answer = response if isinstance(response, str) else getattr(response, 'content', str(response))
    
    # [6] บันทึกประวัติ
    await async_save_message(conv_id, "user", question)
    await async_save_message(conv_id, "ai", answer)

    return QueryResponse(
        answer=answer.strip(),
        sources=_map_sources(all_chunks),
        conversation_id=conv_id
    )

# ===================================================================
# 2. /compare - เปรียบเทียบเอกสาร (JSON Output 100%)
# ===================================================================
@llm_router.post("/compare", response_model=QueryResponse)
async def compare_llm(
    question: str = Form(...),
    doc_ids: List[str] = Form(...),
    doc_types: Optional[List[str]] = Form(None),
    current_user: UserMe = Depends(get_current_user),
):
    if len(doc_ids) < 2:
        raise HTTPException(status_code=400, detail="กรุณาเลือกอย่างน้อย 2 เอกสารเพื่อเปรียบเทียบ")

    llm = create_llm_instance(model_name=DEFAULT_LLM_MODEL_NAME, temperature=LLM_TEMPERATURE)
    vsm = get_vectorstore_manager(tenant=current_user.tenant)

    async def fetch_single_doc_context(d_id):
        res = await asyncio.to_thread(
            retrieve_context_for_endpoint,
            query=question,
            doc_type=doc_types[0] if doc_types else EVIDENCE_DOC_TYPES,
            vectorstore_manager=vsm,
            stable_doc_ids={d_id},
            tenant=current_user.tenant,
            year=current_user.year,
            k_to_retrieve=QUERY_INITIAL_K,
            k_to_rerank=QUERY_FINAL_K
        )
        return "\n".join([ev["text"] for ev in res.get("top_evidences", [])])

    # ดึงข้อมูลจากทั้ง 2 เอกสารพร้อมกัน
    doc1_content, doc2_content = await asyncio.gather(
        fetch_single_doc_context(doc_ids[0]),
        fetch_single_doc_context(doc_ids[1])
    )

    user_compare_content = COMPARE_PROMPT.format(
        doc1_content=doc1_content or "ไม่พบข้อมูล",
        doc2_content=doc2_content or "ไม่พบข้อมูล",
        query=question
    )

    # 🟢 บังคับ JSON + ภาษาไทย (ตาม Original)
    messages = [
        SystemMessage(content="RESPONSE MUST BE IN THAI. OUTPUT MUST BE VALID JSON ONLY.\n" + SYSTEM_COMPARE_INSTRUCTION),
        HumanMessage(content=user_compare_content)
    ]

    raw_response_obj = llm.invoke(messages)
    raw_response = getattr(raw_response_obj, 'content', str(raw_response_obj))
    
    # Clean JSON
    json_str = re.sub(r'^```json\s*|```$', '', raw_response.strip(), flags=re.MULTILINE)
    try:
        result_data = json.loads(json_str)
        summary = result_data.get("overall_summary", "วิเคราะห์เปรียบเทียบเรียบร้อยแล้ว")
    except Exception as e:
        logger.error(f"Compare JSON Parse Error: {e}")
        result_data = {"error": "Invalid JSON from AI", "raw": raw_response}
        summary = "เกิดข้อผิดพลาดในการประมวลผลข้อมูลเปรียบเทียบ"

    return QueryResponse(
        answer=summary,
        sources=[],
        conversation_id=str(uuid.uuid4()),
        result=result_data
    )

# ===================================================================
# --- Helper Functions ---
# ===================================================================

async def _get_context_chunks(question, d_types, d_ids, enabler, subject, vsm, user, sub_topic=None):
    tasks = [
        asyncio.to_thread(
            retrieve_context_for_endpoint,
            query=question, doc_type=dt, enabler=enabler, subject=subject,
            vectorstore_manager=vsm, stable_doc_ids=set(d_ids),
            tenant=user.tenant, year=user.year,
            sub_topic=sub_topic,
            k_to_retrieve=QUERY_INITIAL_K, 
            k_to_rerank=QUERY_FINAL_K
        ) for dt in d_types
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    all_chunks = []
    for res in results:
        if isinstance(res, dict):
            for ev in res.get("top_evidences", []):
                all_chunks.append(LcDocument(
                    page_content=ev["text"],
                    metadata={
                        "score": ev.get("score", 0), 
                        "doc_id": ev.get("doc_id"), 
                        "source": ev.get("source"),
                        "chunk_uuid": ev.get("chunk_uuid")
                    }
                ))
    # จัดลำดับตามความเกี่ยวข้อง
    return sorted(all_chunks, key=lambda x: x.metadata["score"], reverse=True)[:FINAL_K_RERANKED]

def _map_sources(chunks):
    return [QuerySource(
        source_id=str(c.metadata.get("doc_id", "unknown")),
        file_name=c.metadata.get("source", "Unknown"),
        chunk_text=c.page_content,
        chunk_id=c.metadata.get("chunk_uuid"),
        score=float(c.metadata.get("score", 0))
    ) for c in chunks]