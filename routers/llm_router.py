# -*- coding: utf-8 -*-
# routers/llm_router.py - Enterprise Enhanced (Markdown Focus)

import os
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

# --- Core & Utils Imports ---
from core.history_utils import async_save_message
from core.llm_data_utils import retrieve_context_for_endpoint
from core.vectorstore import get_vectorstore_manager
# ยังคง import ไว้สำหรับส่วน /query ที่อาจต้องการใช้ score/reason
from core.json_extractor import _robust_extract_json 

# --- Prompts ---
from core.rag_prompts import (
    SYSTEM_QA_INSTRUCTION, 
    SYSTEM_ANALYSIS_INSTRUCTION,
    SYSTEM_COMPARE_INSTRUCTION,
    COMPARE_PROMPT # มั่นใจว่าใน rag_prompts.py แก้เป็น Markdown Table แล้ว
)
from core.llm_guardrails import detect_intent, build_prompt 

# --- Models & Config ---
from models.llm import create_llm_instance
from routers.auth_router import UserMe, get_current_user
from config.global_vars import (
    EVIDENCE_DOC_TYPES,
    DEFAULT_ENABLER,
    DEFAULT_LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    QUERY_FINAL_K
)

logger = logging.getLogger(__name__)
llm_router = APIRouter(prefix="/api", tags=["LLM"])

# ===================================================================
# Models สำหรับ Request และ Response
# ===================================================================
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
# 1. /query (General QA)
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
    
    q_lower = question.lower()
    is_compare = any(word in q_lower for word in ["compare", "เปรียบเทียบ", "ต่างกัน", "ความแตกต่าง", "vs"])
    
    if is_compare:
        return await compare_llm(question=question, doc_ids=doc_ids, doc_types=doc_types, current_user=current_user)

    vsm = get_vectorstore_manager(tenant=current_user.tenant)
    all_chunks = await _get_context_chunks(question, doc_types or [EVIDENCE_DOC_TYPES], set(doc_ids) if doc_ids else None, enabler or DEFAULT_ENABLER, subject, vsm, current_user)
    
    context_text = "\n\n".join([f"Source [{c.metadata.get('source')}]: {c.page_content}" for c in all_chunks])
    intent = detect_intent(question)
    base_instruction = SYSTEM_ANALYSIS_INSTRUCTION if intent.get("is_analysis") else SYSTEM_QA_INSTRUCTION
    
    messages = [
        SystemMessage(content=f"ALWAYS ANSWER IN THAI.\n{base_instruction}"),
        HumanMessage(content=build_prompt(context_text, question, intent))
    ]

    raw_res = await asyncio.to_thread(llm.invoke, messages)
    answer = raw_res.content if hasattr(raw_res, 'content') else str(raw_res)
    
    await async_save_message(conv_id, "user", question)
    await async_save_message(conv_id, "ai", answer)

    return QueryResponse(answer=answer.strip(), sources=_map_sources(all_chunks), conversation_id=conv_id)

# ===================================================================
# 2. /compare (Markdown Table Version - Optimized for Enterprise)
# ===================================================================
@llm_router.post("/compare", response_model=QueryResponse)
async def compare_llm(
    question: str = Form(...),
    doc_ids: List[str] = Form(...),
    doc_types: Optional[List[str]] = Form(None),
    current_user: UserMe = Depends(get_current_user),
):
    # 1. Validation: ตรวจสอบจำนวนเอกสาร
    if not doc_ids or len(doc_ids) < 2:
        raise HTTPException(
            status_code=400, 
            detail="กรุณาเลือกอย่างน้อย 2 เอกสารเพื่อทำการเปรียบเทียบข้อมูล"
        )

    llm = create_llm_instance(model_name=DEFAULT_LLM_MODEL_NAME, temperature=LLM_TEMPERATURE)
    vsm = get_vectorstore_manager(tenant=current_user.tenant)
    
    # 2. ฟังก์ชันดึงข้อมูลแบบ Strict Filter (ดึงเฉพาะไฟล์ที่ระบุเท่านั้น)
    async def fetch_doc(d_id, index):
        # ขยาย Query ให้สื่อความหมายมากขึ้นในกรณี User พิมพ์มาสั้นๆ (เช่นพิมพ์แค่ "KM")
        enhanced_query = question
        if len(question.strip()) < 10:
            enhanced_query = f"รายละเอียดและเนื้อหาสำคัญเกี่ยวกับ {question} ในเอกสารนี้"

        res = await asyncio.to_thread(
            retrieve_context_for_endpoint,
            vectorstore_manager=vsm,
            query=enhanced_query, # ใช้ query ที่ขยายความแล้ว
            tenant=current_user.tenant,
            year=current_user.year,
            stable_doc_ids={d_id}, 
            doc_type=doc_types[0] if doc_types else EVIDENCE_DOC_TYPES,
            enabler=DEFAULT_ENABLER,
            k_to_retrieve=40, # 🎯 ดึงเพิ่มขึ้นจาก 15 เป็น 40 เพื่อให้ครอบคลุมเนื้อหาทั้งไฟล์
            # ลบ strict_filter ออกชั่วคราวเพื่อป้องกัน Error 
            # เพราะใน llm_data_utils.py เรามี Double-Gate Filter รองรับอยู่แล้ว
        )
        
        evidences = res.get("top_evidences", []) if isinstance(res, dict) else []
        
        # ดึงชื่อไฟล์จริงจาก Metadata (ถ้าหาไม่เจอให้ใช้ ID แทน)
        file_name = "Unknown File"
        if evidences:
            first_meta = evidences[0]
            file_name = first_meta.get("source") or first_meta.get("file_name") or f"ID: {d_id}"
        
        # สร้าง Context Block ที่มีขอบเขตชัดเจน
        content_text = "\n".join([f"- {e['text']}" for e in evidences])
        
        formatted_content = f"### [เอกสารที่ {index}]: {file_name}\n"
        if not content_text:
            formatted_content += "(ไม่พบเนื้อหาที่เกี่ยวข้องในฐานข้อมูลสำหรับไฟล์นี้)\n"
        else:
            formatted_content += f"{content_text}\n"
            
        return {
            "formatted_content": formatted_content,
            "evidences": evidences
        }

    # 3. ดึงข้อมูลจากทุกไฟล์พร้อมกันแบบ Parallel
    fetch_results = await asyncio.gather(*[fetch_doc(d_id, i+1) for i, d_id in enumerate(doc_ids)])
    
    doc_contents = [r["formatted_content"] for r in fetch_results]
    comparison_sources = []
    for r in fetch_results:
        comparison_sources.extend(r["evidences"])

    # 4. เตรียม Prompt โดยใส่ Context ที่แยกส่วนชัดเจน
    # มั่นใจว่า COMPARE_PROMPT ใน rag_prompts.py มี 'กฎเหล็ก' ห้ามใช้ความรู้ภายนอก
    user_compare_content = COMPARE_PROMPT.format(
        documents_content="\n\n".join(doc_contents), 
        query=question
    )

    # 5. สั่งงาน LLM
    messages = [
        SystemMessage(content=(
            "คุณคือนักวิเคราะห์ข้อมูล SE-AM มืออาชีพ "
            "จงตอบข้อมูลในรูปแบบ Markdown Table เท่านั้น "
            "ห้ามเดาข้อมูลนอกเหนือจาก Context ที่ให้ไว้ "
            "หากข้อมูลไม่พอให้ระบุว่า 'ไม่ปรากฏข้อมูล' ในช่องนั้นๆ"
        )),
        HumanMessage(content=user_compare_content)
    ]

    try:
        # เรียกใช้ invoke และจัดการ response
        raw_res_obj = await asyncio.to_thread(llm.invoke, messages)
        answer = raw_res_obj.content if hasattr(raw_res_obj, 'content') else str(raw_res_obj)
        
        # 6. บันทึกประวัติการสนทนา
        conv_id = str(uuid.uuid4())
        await async_save_message(conv_id, "user", f"[เปรียบเทียบเอกสาร] {question}")
        await async_save_message(conv_id, "ai", answer)

        # 7. ส่งค่ากลับ (Markdown Table จะเรนเดอร์อัตโนมัติที่หน้าบ้าน)
        return QueryResponse(
            answer=answer.strip(),
            sources=_map_sources_from_list(comparison_sources),
            conversation_id=conv_id,
            result=None # ปิด JSON Parse ชั่วคราวเพื่อความเสถียร
        )

    except Exception as e:
        logger.error(f"Error in compare_llm: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="เกิดข้อผิดพลาดในการประมวลผลการเปรียบเทียบ")

# ===================================================================
# Helpers
# ===================================================================

def _map_sources_from_list(evidences):
    # กำจัดความซ้ำซ้อนของ source ในการเปรียบเทียบ
    seen = set()
    unique_sources = []
    for e in evidences:
        if e.get("doc_id") not in seen:
            unique_sources.append(e)
            seen.add(e.get("doc_id"))
    
    return [QuerySource(
        source_id=str(e.get("doc_id", "unknown")),
        file_name=e.get("source", "Unknown"),
        chunk_text=e.get("text", "")[:200] + "...", # ย่อ text เพื่อลดขนาด response
        chunk_id=e.get("chunk_uuid"),
        score=float(e.get("score", 0))
    ) for e in unique_sources][:10]

async def _get_context_chunks(question, doc_types, stable_doc_ids, enabler, subject, vsm, user):
    tasks = [
        asyncio.to_thread(
            retrieve_context_for_endpoint,
            vectorstore_manager=vsm, query=question, doc_type=dt,
            enabler=enabler, stable_doc_ids=stable_doc_ids,
            tenant=user.tenant, year=user.year, subject=subject
        ) for dt in doc_types
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    all_chunks = []
    for res in results:
        if isinstance(res, dict) and "top_evidences" in res:
            for ev in res["top_evidences"]:
                all_chunks.append(LcDocument(
                    page_content=ev["text"],
                    metadata={
                        "score": ev.get("score", 0),
                        "doc_id": ev.get("doc_id"),
                        "source": ev.get("source"),
                        "chunk_uuid": ev.get("chunk_uuid")
                    }
                ))
    
    all_chunks.sort(key=lambda x: x.metadata.get("score", 0), reverse=True)
    return all_chunks[:QUERY_FINAL_K]

def _map_sources(chunks):
    return [QuerySource(
        source_id=str(c.metadata.get("doc_id", "unknown")),
        file_name=c.metadata.get("source", "Unknown"),
        chunk_text=c.page_content,
        chunk_id=c.metadata.get("chunk_uuid"),
        score=float(c.metadata.get("score", 0))
    ) for c in chunks]