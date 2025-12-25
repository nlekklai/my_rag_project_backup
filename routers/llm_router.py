# -*- coding: utf-8 -*-
# routers/llm_router.py - Enterprise RAG (Query + Compare + PDCA Analysis + Summary)
# ULTIMATE FINAL PRODUCTION VERSION - 22 ธันวาคม 2568
# รองรับ: Multi-year evidence via wrapper, UUID v5, Clean fallback, Full intent routing

import logging
import uuid
import asyncio
from typing import List, Optional, Set, Dict, Any
from collections import defaultdict

from fastapi import APIRouter, Form, HTTPException, Depends
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document as LcDocument

from core.history_utils import async_save_message, get_recent_history
from core.llm_data_utils import retrieve_context_for_endpoint
from core.vectorstore import get_vectorstore_manager
from core.seam_assessment import SEAMPDCAEngine, AssessmentConfig
from core.llm_guardrails import enforce_thai_primary_language, detect_intent
from config.global_vars import (
    EVIDENCE_DOC_TYPES,
    DEFAULT_ENABLER,
    DEFAULT_LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    QUERY_FINAL_K,
    DEFAULT_DOC_TYPES,  # เช่น ["document"]
    RETRIEVAL_TOP_K,      # 🎯 ดึงจาก .env (Mac: 150, Server: 500)
    ANALYSIS_FINAL_K,     # 🎯 ดึงจาก .env (Mac: 12, Server: 30)
    QA_FINAL_K
)
from models.llm import create_llm_instance
from routers.auth_router import UserMe, get_current_user
from core.rag_prompts import (
    SYSTEM_QA_INSTRUCTION,
    SYSTEM_ANALYSIS_INSTRUCTION,
    SYSTEM_COMPARE_INSTRUCTION,
    SYSTEM_CONSULTANT_INSTRUCTION,      # <--- เพิ่มอันนี้
    QA_PROMPT_TEMPLATE,
    COMPARE_PROMPT_TEMPLATE,
    ANALYSIS_PROMPT_TEMPLATE,
    REVERSE_MAPPING_PROMPT_TEMPLATE,     # <--- เพิ่มอันนี้
    SUMMARY_PROMPT_TEMPLATE
)
from utils.path_utils import get_rubric_file_path, get_doc_type_collection_key
import json, os
import time

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
# Helper: _map_sources
# =====================================================================
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


# =====================================================================
# Helper: load_all_chunks_by_doc_ids (สำหรับ /compare)
# =====================================================================
def load_all_chunks_by_doc_ids(
    vectorstore_manager,
    collection_name: str,
    stable_doc_ids: Set[str] | List[str]
) -> List[LcDocument]:
    chroma = vectorstore_manager._load_chroma_instance(collection_name)
    if not chroma:
        logger.warning(f"Chroma collection not found: {collection_name}")
        return []
    where_filter = {"stable_doc_uuid": {"$in": list(stable_doc_ids)}}
    docs = chroma.similarity_search(query="*", k=9999, filter=where_filter)
    return [d for d in docs if getattr(d, "page_content", "").strip()]

# =====================================================================
# 1. /query — Smart General RAG with Intent Detection & Auto-Routing
# =====================================================================
# =====================================================================
# 1. /query — Smart General RAG with Intent Detection & Auto-Routing
# =====================================================================
@llm_router.post("/query", response_model=QueryResponse)
async def query_llm(
    question: str = Form(...),
    conversation_id: Optional[str] = Form(None),
    doc_types: List[str] = Form(default=[]),  
    doc_ids: List[str] = Form(default=[]),     
    enabler: Optional[str] = Form(None),
    subject: Optional[str] = Form(None),
    year: Optional[str] = Form(None),
    current_user: UserMe = Depends(get_current_user),
):
    # 🎯 0. Setup พื้นฐาน
    llm = create_llm_instance(model_name=DEFAULT_LLM_MODEL_NAME, temperature=LLM_TEMPERATURE)
    conv_id = conversation_id or str(uuid.uuid4())
    effective_year = year or str(current_user.year)
    
    # ดึงค่า Config จาก .env (Mac=8, Server=20)
    from config.global_vars import QA_FINAL_K, DEFAULT_DOC_TYPES, DEFAULT_ENABLER

    logger.info(f"📩 Query received: '{question}' from user {current_user.id}")

    # 🎯 1. ตรวจสอบประวัติและเจตนา (Intent Detection)
    history = await get_recent_history(current_user.id, conv_id, limit=6)
    intent = detect_intent(question, user_context=history)

    # 🔍 [SMART ROUTE OVERRIDE] 
    # บังคับไป /analysis หากเจอ Keyword เกี่ยวกับเกณฑ์ประเมิน
    analysis_keywords = ["pdca", "comply", "compliance", "เกณฑ์", "ผ่านระดับ", "level", "จุดแข็ง", "ช่องว่าง", "ประเมิน", "สอดคล้อง"]
    is_forcing_analysis = any(kw in question.lower() for kw in analysis_keywords)

    # --- [BRANCH 1] Greeting & Capabilities ---
    if intent.get("is_greeting") or intent.get("is_capabilities"):
        if intent.get("is_greeting"):
            answer = "สวัสดีครับ! 😊 ผมคือ Digital Knowledge Assistant ของ PEA พร้อมช่วยตอบคำถามเกี่ยวกับเอกสาร KM หรือวิเคราะห์ SE-AM ให้ครับ มีอะไรให้ช่วยไหมครับ?"
        else:
            answer = (
                "ผมช่วยคุณได้ดังนี้ครับ:\n"
                "1. **ค้นหาและตอบคำถาม** จากเอกสาร/นโยบาย/ระเบียบ\n"
                "2. **สรุปเอกสาร** สาระสำคัญสำหรับผู้บริหาร\n"
                "3. **เปรียบเทียบเอกสาร** 2 ฉบับขึ้นไป\n"
                "4. **วิเคราะห์ตามเกณฑ์ SE-AM** (PDCA, จุดแข็ง, ช่องว่าง)"
            )
        await async_save_message(current_user.id, conv_id, "user", question)
        await async_save_message(current_user.id, conv_id, "ai", answer)
        return QueryResponse(answer=answer, sources=[], conversation_id=conv_id)

    # --- [BRANCH 2] Comparison (Redirect) ---
    if intent.get("is_comparison"):
        logger.info("🔀 Route -> Comparison")
        return await compare_llm(
            question=question, doc_ids=doc_ids or [],
            doc_types=doc_types, enabler=enabler, current_user=current_user
        )

    # --- [BRANCH 3] SE-AM Analysis (Redirect) ---
    if intent.get("is_analysis") or intent.get("is_criteria_query") or is_forcing_analysis:
        if doc_ids:
            logger.info(f"🚀 Route -> Analysis (Forced: {is_forcing_analysis})")
            return await analysis_llm(
                question=question, doc_ids=doc_ids, doc_types=doc_types,
                enabler=enabler, subject=subject, conversation_id=conv_id,
                current_user=current_user, year=effective_year,
            )
        else:
            if is_forcing_analysis:
                answer = "🔍 ผมเห็นว่าคุณต้องการให้วิเคราะห์เกณฑ์ SE-AM/PDCA รบกวนช่วยเลือกไฟล์เอกสารที่ต้องการให้ผมตรวจสอบที่ด้านข้างก่อนนะครับ"
                await async_save_message(current_user.id, conv_id, "user", question)
                await async_save_message(current_user.id, conv_id, "ai", answer)
                return QueryResponse(answer=answer, sources=[], conversation_id=conv_id)

    # --- [BRANCH 4] RAG Flow (General QA & Summary) ---
    logger.info(f"📖 Executing: General RAG Flow (K={QA_FINAL_K})")
    used_doc_types = doc_types or DEFAULT_DOC_TYPES
    used_enabler = enabler or DEFAULT_ENABLER
    vsm = get_vectorstore_manager(tenant=current_user.tenant)
    stable_doc_ids = {str(idx).strip() for idx in doc_ids if str(idx).strip()} if doc_ids else None

    all_chunks = []
    for dt in used_doc_types:
        res = await asyncio.to_thread(
            retrieve_context_for_endpoint,
            vectorstore_manager=vsm, query=question, doc_type=dt,
            enabler=used_enabler, stable_doc_ids=stable_doc_ids,
            tenant=current_user.tenant, year=effective_year, subject=subject,
        )
        if isinstance(res, dict) and "top_evidences" in res:
            for ev in res.get("top_evidences", []):
                # 🟢 Robust Metadata Extraction (Sync กับ ingest.py และ _normalize_meta)
                f_name = ev.get('source_filename') or ev.get('source') or 'Unknown'
                
                # ลำดับความสำคัญ: page_label (จาก UI) -> page_number -> page
                p_val = ev.get('page_label') or ev.get('page_number') or ev.get('page')
                p_display = str(p_val).strip() if p_val and str(p_val).lower() != 'n/a' else "N/A"

                all_chunks.append(
                    LcDocument(
                        page_content=ev["text"],
                        metadata={
                            "score": ev.get("rerank_score") or ev.get("score") or 0.0,
                            "doc_id": ev.get("doc_id"),
                            "source": f_name,
                            "page": p_display,
                            "chunk_uuid": ev.get("chunk_uuid"),
                        }
                    )
                )

    # 🎯 คัดเลือก "หัวกะทิ" ตามจำนวนที่ตั้งไว้ใน .env
    all_chunks.sort(key=lambda c: c.metadata.get("score", 0), reverse=True)
    final_chunks = all_chunks[:QA_FINAL_K]

    if not final_chunks:
        answer = "ขออภัยครับ ไม่พบเนื้อหาที่เกี่ยวข้องในเอกสารที่เลือกครับ"
        return QueryResponse(answer=answer, sources=[], conversation_id=conv_id)

    # Context with Metadata for AI
    context_text = "\n\n".join([
        f"[ไฟล์: {c.metadata['source']}, หน้า: {c.metadata['page']}]\n{c.page_content}" 
        for c in final_chunks
    ])

    # 🎯 Intent-based Prompting
    if intent.get("is_summary"):
        sys_msg = "คุณคือผู้เชี่ยวชาญด้าน KM ของ PEA ตอบเป็นภาษาไทย สรุปสาระสำคัญสำหรับผู้บริหาร และระบุเลขหน้าอ้างอิงเสมอ"
        prompt_text = SUMMARY_PROMPT_TEMPLATE.format(context=context_text)
    else:
        sys_msg = "ALWAYS ANSWER IN THAI.\n" + SYSTEM_QA_INSTRUCTION
        prompt_text = QA_PROMPT_TEMPLATE.format(context=context_text, question=question)

    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": prompt_text}
    ]
    
    # Inference
    raw = await asyncio.to_thread(llm.invoke, messages)
    answer = enforce_thai_primary_language(raw.content if hasattr(raw, "content") else str(raw))

    # บันทึกประวัติการสนทนา
    await async_save_message(current_user.id, conv_id, "user", question)
    await async_save_message(current_user.id, conv_id, "ai", answer)

    # 🎯 เตรียม Sources สำหรับ UI (แสดงชื่อไฟล์ + เลขหน้า)
    sources = [
        QuerySource(
            source_id=str(c.metadata["doc_id"]),
            file_name=f"{c.metadata['source']} (หน้า {c.metadata['page']})",
            chunk_text=c.page_content[:500],
            chunk_id=c.metadata["chunk_uuid"],
            score=float(c.metadata["score"]),
        )
        for c in final_chunks
    ]

    return QueryResponse(answer=answer.strip(), sources=sources, conversation_id=conv_id)

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

    used_doc_types = doc_types or ["document"]
    is_evidence = any(dt.lower() == EVIDENCE_DOC_TYPES.lower() for dt in used_doc_types)
    used_enabler = enabler or (DEFAULT_ENABLER if is_evidence else None)

    if is_evidence and not used_enabler:
        raise HTTPException(400, "สำหรับ compare เอกสาร evidence ต้องระบุ enabler")

    collection_name = get_doc_type_collection_key(used_doc_types[0], used_enabler)
    vsm = get_vectorstore_manager(tenant=current_user.tenant)
    llm = create_llm_instance(model_name=DEFAULT_LLM_MODEL_NAME, temperature=LLM_TEMPERATURE)

    all_chunks = load_all_chunks_by_doc_ids(vsm, collection_name, set(doc_ids))
    if not all_chunks:
        raise HTTPException(400, "ไม่พบข้อมูลในเอกสารที่เลือก")

    doc_groups = defaultdict(list)
    for d in all_chunks:
        doc_key = str(d.metadata.get("stable_doc_uuid") or d.metadata.get("doc_id"))
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

    prompt_text = COMPARE_PROMPT_TEMPLATE.format(documents_content="\n\n".join(doc_blocks), query=question)
    messages = [
        SystemMessage(content=SYSTEM_COMPARE_INSTRUCTION),
        HumanMessage(content=prompt_text),
    ]

    raw = await asyncio.to_thread(llm.invoke, messages)
    answer = enforce_thai_primary_language(raw.content if hasattr(raw, "content") else str(raw))

    conv_id = str(uuid.uuid4())
    await async_save_message(current_user.id, conv_id, "user", question)
    await async_save_message(current_user.id, conv_id, "ai", answer)

    return QueryResponse(answer=answer.strip(), sources=_map_sources(all_chunks[:10]), conversation_id=conv_id)


def enhance_analysis_query(question: str, subject_id: str, rubric_data: dict) -> str:
    """
    ขยายความคำถามโดยอ้างอิงจากชื่อหัวข้อใน Rubric เพื่อให้ RAG ค้นหาได้แม่นยำขึ้น
    """
    # 1. พยายามหาชื่อหัวข้อจาก Rubric JSON
    criteria_name = ""
    target_rubric = rubric_data.get(subject_id, {})
    if target_rubric:
        criteria_name = target_rubric.get("name", "")
    
    # 2. สร้างคำถามที่รวม Keywords สำคัญ
    enhanced = f"วิเคราะห์ข้อมูลและประเมินตามเกณฑ์ SE-AM หัวข้อ {subject_id} {criteria_name}: {question} "
    enhanced += "โดยเน้นการค้นหาหลักฐานด้าน แผนงาน (Plan), การลงมือปฏิบัติ (Do), การวัดผลและตรวจสอบ (Check), และการปรับปรุงแก้ไข (Act)"
    
    return enhanced

# =====================================================================
# 3. /analysis — PDCA-focused SE-AM analysis (Mac & Server Standard)
# =====================================================================
# =====================================================================
# 3. /analysis — PDCA-focused SE-AM analysis (Revise Standard)
# =====================================================================
@llm_router.post("/analysis", response_model=QueryResponse)
async def analysis_llm(
    question: str = Form(...),
    doc_ids: Any = Form(None),      
    doc_types: Any = Form(None),    
    enabler: Optional[str] = Form(None),
    subject: Optional[str] = Form(None), 
    conversation_id: Optional[str] = Form(None),
    year: Optional[str] = Form(None),
    current_user: UserMe = Depends(get_current_user),
):
    start_time = time.time()
    conv_id = conversation_id or str(uuid.uuid4())
    effective_year = year or str(current_user.year)

    # 🛠️ 1. Data Type Normalization (IDs & Types)
    stable_doc_ids = []
    if doc_ids:
        if isinstance(doc_ids, list):
            stable_doc_ids = [str(idx).strip() for idx in doc_ids if str(idx).strip()]
        elif isinstance(doc_ids, str):
            stable_doc_ids = [idx.strip() for idx in doc_ids.split(",") if idx.strip()]

    if not doc_types:
        used_doc_types = [EVIDENCE_DOC_TYPES]
    elif isinstance(doc_types, list):
        used_doc_types = doc_types
    elif isinstance(doc_types, str):
        used_doc_types = [dt.strip() for dt in doc_types.split(",") if dt.strip()]
    else:
        used_doc_types = [EVIDENCE_DOC_TYPES]

    is_evidence = any(dt.lower() == EVIDENCE_DOC_TYPES.lower() for dt in used_doc_types)
    used_enabler = enabler or (DEFAULT_ENABLER if is_evidence else None)

    if is_evidence and not used_enabler:
        raise HTTPException(400, "สำหรับ analysis เอกสาร evidence ต้องระบุ enabler")

    # Initialize Engine & LLM
    vsm = get_vectorstore_manager(tenant=current_user.tenant)
    llm = create_llm_instance(model_name=DEFAULT_LLM_MODEL_NAME, temperature=LLM_TEMPERATURE)

    # 🎯 2. Load Rubric JSON (for logic & prompts)
    rubric_data = {}
    rubric_json_str = "{}"
    try:
        rubric_path = get_rubric_file_path(current_user.tenant, used_enabler)
        if os.path.exists(rubric_path):
            with open(rubric_path, 'r', encoding='utf-8') as f:
                rubric_data = json.load(f)
                rubric_json_str = json.dumps(rubric_data, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to load rubric JSON: {e}")

    # 🎯 3. Determine Analysis Mode
    consultant_keywords = ["เหมาะ", "หลักฐาน", "ประเมินหรือไม่", "สอดคล้อง", "หัวข้อไหน", "เกณฑ์ไหน", "ระดับไหน"]
    is_consultant_mode = any(kw in question.lower() for kw in consultant_keywords) or (not subject and len(stable_doc_ids) <= 2)

    search_query = question
    if subject:
        search_query = enhance_analysis_query(question, subject, rubric_data)
    elif is_consultant_mode:
        # ปรับ Query ให้ค้นหาเชิง "ความพึงพอใจ" หรือ "ผลการดำเนินงาน" เมื่อมีคำใบ้เรื่องหน้า/ข้อมูลสรุป
        search_query = f"ผลการดำเนินงาน ตัวชี้วัด เป้าหมาย KM PDCA สรุปผู้บริหาร {question}"

    # 🎯 4. Smart Retrieval (Environment Sensitive)
    from core.llm_data_utils import retrieve_context_with_rubric
    all_evidences = []
    for dt in used_doc_types:
        retrieval_res = await asyncio.to_thread(
            retrieve_context_with_rubric,
            vectorstore_manager=vsm,
            query=search_query,
            doc_type=dt,
            enabler=used_enabler,
            stable_doc_ids=stable_doc_ids,
            tenant=current_user.tenant,
            year=effective_year,
            subject=subject,
            strict_filter=True,
            top_k=RETRIEVAL_TOP_K  # 🚀 ดึงจาก .env (Mac: 150, Server: 500)
        )
        if retrieval_res and "top_evidences" in retrieval_res:
            all_evidences.extend(retrieval_res["top_evidences"])

    # Reranking & Context Selection
    all_evidences.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
    final_evidences = all_evidences[:ANALYSIS_FINAL_K]  # 🚀 คัดตามความจุ LLM (.env)

    if not final_evidences:
        answer = "ไม่พบเนื้อหาที่เกี่ยวข้องในเอกสารที่เลือกเพื่อนำมาวิเคราะห์ครับ"
        return QueryResponse(answer=answer, sources=[], conversation_id=conv_id)

    # 🎯 5. PDCA Assessment Engine
    engine_config = AssessmentConfig(
        tenant=current_user.tenant,
        year=int(effective_year) if effective_year.isdigit() else current_user.year,
        enabler=used_enabler,
        target_level=5
    )
    engine = SEAMPDCAEngine(config=engine_config, llm_instance=llm, vectorstore_manager=vsm, doc_type=used_doc_types[0])

    # แยก Blocks ข้อมูลเข้าโครงสร้าง PDCA
    plan_blocks, do_blocks, check_blocks, act_blocks, other_blocks = engine._get_pdca_blocks_from_evidences(
        evidences=final_evidences, baseline_evidences={}, level=5, sub_id=subject or "all", contextual_rules_map=engine.contextual_rules_map
    )
    pdca_context = "\n\n".join(filter(None, [plan_blocks, do_blocks, check_blocks, act_blocks, other_blocks]))

    # 🎯 6. Inference Flow
    if is_consultant_mode:
        sys_msg_content = "ALWAYS ANSWER IN THAI.\n" + SYSTEM_CONSULTANT_INSTRUCTION
        prompt_text = REVERSE_MAPPING_PROMPT_TEMPLATE.format(
            rubric_json=rubric_json_str,
            documents_content=pdca_context
        )
        mode_label = "Consultant"
    else:
        sys_msg_content = "ALWAYS ANSWER IN THAI.\n" + SYSTEM_ANALYSIS_INSTRUCTION
        prompt_text = ANALYSIS_PROMPT_TEMPLATE.format(
            rubric_json=rubric_json_str,
            documents_content=pdca_context,
            question=question 
        )
        mode_label = "Auditor"

    logger.info(f"🚀 Analysis Mode: {mode_label} | Retrieval K: {RETRIEVAL_TOP_K} | Final K: {ANALYSIS_FINAL_K}")

    messages = [
        SystemMessage(content=sys_msg_content),
        HumanMessage(content=prompt_text),
    ]

    raw_response = await asyncio.to_thread(llm.invoke, messages)
    answer = enforce_thai_primary_language(raw_response.content if hasattr(raw_response, "content") else str(raw_response))

    # 🎯 7. Map Sources for UI
    sources = []
    for ev in final_evidences[:10]: # แสดงแหล่งที่มา 10 อันดับแรก
        f_name = ev.get('source_filename') or ev.get('source') or 'Unknown'
        p_val = ev.get('page_label') or ev.get('page_number') or ev.get('page')
        p_display = str(p_val).strip() if p_val and str(p_val).lower() != 'n/a' else "N/A"

        sources.append(
            QuerySource(
                source_id=str(ev.get("doc_id", "unknown")),
                file_name=f"{f_name} (หน้า {p_display})",
                chunk_text=ev.get("text", "")[:500],
                chunk_id=ev.get("chunk_uuid"),
                score=float(ev.get("rerank_score") or ev.get("score") or 0.0),
            )
        )

    await async_save_message(current_user.id, conv_id, "user", question)
    await async_save_message(current_user.id, conv_id, "ai", answer)

    return QueryResponse(
        answer=answer.strip(), 
        sources=sources, 
        conversation_id=conv_id,
        result={
            "process_time": round(time.time() - start_time, 2), 
            "mode": mode_label.lower(),
            "config": {"retrieval_k": RETRIEVAL_TOP_K, "final_k": ANALYSIS_FINAL_K}
        }
    )