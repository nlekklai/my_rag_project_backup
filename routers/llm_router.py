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
from core.llm_data_utils import retrieve_context_for_endpoint, retrieve_context_with_rubric
from core.vectorstore import get_vectorstore_manager
from core.seam_assessment import SEAMPDCAEngine, AssessmentConfig
from core.llm_guardrails import enforce_thai_primary_language, detect_intent, build_prompt
from config.global_vars import (
    EVIDENCE_DOC_TYPES,
    DEFAULT_ENABLER,
    DEFAULT_LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    DEFAULT_DOC_TYPES,  # เช่น ["document"]
    RETRIEVAL_TOP_K,      # 🎯 ดึงจาก .env (Mac: 150, Server: 500)
    ANALYSIS_FINAL_K,     # 🎯 ดึงจาก .env (Mac: 12, Server: 30)
    QA_FINAL_K,
    DEFAULT_YEAR
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
from utils.path_utils import get_rubric_file_path, get_doc_type_collection_key, get_document_file_path
import json, os
import time
from fastapi.responses import FileResponse

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
    document_uuid: Optional[str] = None
    page_number: Optional[int] = 1
    page_display: Optional[str] = None
    url: Optional[str] = None # มั่นใจว่ามีบรรทัดนี้

class QueryResponse(BaseModel):
    answer: str
    sources: List[QuerySource] = Field(default_factory=list)
    conversation_id: str
    result: Optional[Dict[str, Any]] = None


def generate_source_url(
    doc_id: str, 
    page: int, 
    doc_type: str, 
    tenant: str, 
    year: str, 
    enabler: Optional[str] = None
) -> str:
    if not doc_id or doc_id == "unknown":
        return ""

    # 🔥 FIX จุดเดียว: ชี้ไปที่ Backend โดยตรง
    BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://localhost:8000")
    url = f"{BACKEND_BASE_URL}/api/files/view/{doc_id}"

    params = [
        f"page={page}",
        f"doc_type={doc_type.lower()}",
        f"tenant={tenant}",
    ]

    if doc_type.lower() == EVIDENCE_DOC_TYPES.lower():
        safe_year = (
            year if year and str(year) not in ("None", "undefined")
            else str(DEFAULT_YEAR)
        )
        params.append(f"year={safe_year}")

        if enabler:
            params.append(f"enabler={enabler}")
        elif DEFAULT_ENABLER:
            params.append(f"enabler={DEFAULT_ENABLER}")

    final_url = url + "?" + "&".join(params)
    logger.info(f"🔗 Generated source URL: {final_url}")

    return final_url


# =====================================================================
# Revised Helper: _map_sources
# =====================================================================
def _map_sources(
    chunks: List[LcDocument], 
    tenant: str, 
    doc_type: str, 
    year: str = None, 
    enabler: str = None
) -> List[QuerySource]:
    """รับพารามิเตอร์เพิ่มเพื่อสร้าง URL ที่ถูกต้องตามบริบทของ Chunk นั้นๆ"""
    return [
        QuerySource(
            source_id=str(c.metadata.get("doc_id", "unknown")),
            file_name=c.metadata.get("source", "Unknown"),
            chunk_text=c.page_content[:500],
            chunk_id=c.metadata.get("chunk_uuid"),
            score=float(c.metadata.get("score", 0)),
            document_uuid=str(c.metadata.get("stable_doc_uuid") or c.metadata.get("doc_id")),
            page_number=int(c.metadata.get("page", 1)) if str(c.metadata.get("page")).isdigit() else 1,
            page_display=f"p. {c.metadata.get('page', '1')}",
            # 🔥 สร้าง URL สำเร็จรูปจากตรงนี้เลย
            url=generate_source_url(
                doc_id=str(c.metadata.get("stable_doc_uuid") or c.metadata.get("doc_id")),
                page=int(c.metadata.get("page", 1)) if str(c.metadata.get("page")).isdigit() else 1,
                doc_type=doc_type,
                tenant=tenant,
                year=year,
                enabler=enabler
            )
        )
        for c in chunks
    ]

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
# 1. /query — Smart General RAG (Revise Sources with URL)
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
    llm = create_llm_instance(model_name=DEFAULT_LLM_MODEL_NAME, temperature=LLM_TEMPERATURE)
    conv_id = conversation_id or str(uuid.uuid4())
   
    if year and year.strip() and year != "undefined":
        effective_year = year
    else:
        effective_year = str(DEFAULT_YEAR) 

    history = await get_recent_history(current_user.id, conv_id, limit=6)
    intent = detect_intent(question, user_context=history)

    # Smart Routing Logic
    analysis_keywords = ["pdca", "เกณฑ์", "ระดับ", "วิเคราะห์", "พิกัด", "เลขหน้า", "หลักฐาน"]
    is_forcing_analysis = any(kw in question.lower() for kw in analysis_keywords) or subject is not None
    
    if intent.get("is_analysis") or is_forcing_analysis:
        if doc_ids:
            return await analysis_llm(
                question=question, doc_ids=doc_ids, doc_types=doc_types,
                enabler=enabler, subject=subject, conversation_id=conv_id,
                current_user=current_user, year=effective_year,
            )

    # General RAG Flow
    used_doc_types = doc_types if doc_types else DEFAULT_DOC_TYPES
    is_evidence_search = any(dt.lower() == "evidence" for dt in used_doc_types)
    used_enabler = enabler if enabler else (DEFAULT_ENABLER if is_evidence_search else None)

    vsm = get_vectorstore_manager(tenant=current_user.tenant, year=int(effective_year))
    stable_doc_ids = {str(idx).strip() for idx in doc_ids if str(idx).strip()} if doc_ids else None

    all_chunks = []
    for dt in used_doc_types:
        res = await asyncio.to_thread(
            retrieve_context_for_endpoint,
            vectorstore_manager=vsm, query=question, doc_type=dt,
            enabler=used_enabler, stable_doc_ids=stable_doc_ids,
            tenant=current_user.tenant, year=effective_year, subject=subject,
            k_to_retrieve=RETRIEVAL_TOP_K, k_to_rerank=QA_FINAL_K
        )
        
        if isinstance(res, dict) and "top_evidences" in res:
            for ev in res.get("top_evidences", []):
                # 🎯 แก้ไขการดึงเลขหน้าที่แม่นยำขึ้น
                p_val = ev.get('page_label') or ev.get('page') or "1"
                d_uuid = ev.get("doc_id") or ev.get("stable_doc_uuid")
                
                all_chunks.append(
                    LcDocument(
                        page_content=ev["text"],
                        metadata={
                            "score": ev.get("rerank_score") or ev.get("score") or 0.0,
                            "doc_id": d_uuid,
                            "source": ev.get('source_filename') or ev.get('source') or 'Unknown',
                            "page": p_val,
                            "chunk_uuid": ev.get("chunk_uuid"),
                            "doc_type": dt
                        }
                    )
                )

    all_chunks.sort(key=lambda c: c.metadata.get("score", 0), reverse=True)
    final_chunks = all_chunks[:QA_FINAL_K]

    if not final_chunks:
        return QueryResponse(answer="ไม่พบเนื้อหาที่เกี่ยวข้องครับ", sources=[], conversation_id=conv_id)

    # Inference
    context_text = "\n\n".join([f"[ไฟล์: {c.metadata['source']}, หน้า: {c.metadata['page']}]\n{c.page_content}" for c in final_chunks])
    full_prompt = build_prompt(context=context_text, question=question, intent=intent, user_context=history)
    raw = await asyncio.to_thread(llm.invoke, [{"role": "user", "content": full_prompt}])
    answer = enforce_thai_primary_language(raw.content if hasattr(raw, "content") else str(raw))

    # 🎯 สร้าง Sources พร้อม URL (FIXED)
    sources = []
    for c in final_chunks:
        p_num = int(c.metadata["page"]) if str(c.metadata["page"]).isdigit() else 1
        sources.append(QuerySource(
            source_id=str(c.metadata["doc_id"]),
            file_name=c.metadata['source'],
            chunk_text=c.page_content[:500],
            chunk_id=c.metadata["chunk_uuid"],
            score=float(c.metadata["score"]),
            document_uuid=str(c.metadata["doc_id"]),
            page_number=p_num,
            page_display=f"p. {c.metadata['page']}",
            url=generate_source_url(
                doc_id=str(c.metadata["doc_id"]),
                page=p_num,
                doc_type=c.metadata["doc_type"],
                tenant=current_user.tenant,
                year=effective_year,
                enabler=used_enabler
            )
        ))

    await async_save_message(current_user.id, conv_id, "user", question)
    await async_save_message(current_user.id, conv_id, "ai", answer)
    return QueryResponse(answer=answer.strip(), sources=sources, conversation_id=conv_id)

# =====================================================================
# 2. /compare — Document Comparison (Revised for Llama 3:70B)
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
    
    # สำหรับ Comparison แนะนำ Temperature ต่ำ (0.1) เพื่อลดอาการหลุดภาษาอังกฤษของ Llama 3
    llm = create_llm_instance(model_name=DEFAULT_LLM_MODEL_NAME, temperature=0.1)

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
            # Llama 3:70B รับ context ได้เยอะ แต่จำกัด 15 chunks เพื่อความคมของเนื้อหา
            body = "\n".join(f"- {c.page_content}" for c in chunks[:15]) 
            block = f"### เอกสารที่ {idx}: {fname}\n{body}"
        doc_blocks.append(block)

    # --- [Llama 3 Language Enforcement Strategy] ---
    # บังคับ Prompt ให้ดุขึ้นและระบุภาษาชัดเจนในระดับ Message
    thai_enforcement = "\n\n(IMPORTANT: สรุปเป็นภาษาไทยสละสลวยเท่านั้น ห้ามตอบเป็นภาษาอังกฤษเด็ดขาด)"
    full_query = f"{question}{thai_enforcement}"
    
    prompt_text = COMPARE_PROMPT_TEMPLATE.format(
        documents_content="\n\n".join(doc_blocks), 
        query=full_query
    )

    messages = [
        SystemMessage(content=SYSTEM_COMPARE_INSTRUCTION),
        # เพิ่ม HumanMessage ตัวที่สองเพื่อย้ำคำสั่ง (Llama 3 จะให้ความสำคัญกับข้อความท้ายๆ)
        HumanMessage(content=prompt_text),
        HumanMessage(content="จงเปรียบเทียบข้อมูลด้านบนและตอบกลับในรูปแบบตารางภาษาไทยเท่านั้น")
    ]

    # เรียกใช้งาน LLM
    raw = await asyncio.to_thread(llm.invoke, messages)
    raw_content = raw.content if hasattr(raw, "content") else str(raw)
    
    # ตรวจสอบความเรียบร้อยของภาษาผ่าน Guardrails
    answer = enforce_thai_primary_language(raw_content)

    conv_id = str(uuid.uuid4())
    await async_save_message(current_user.id, conv_id, "user", question)
    await async_save_message(current_user.id, conv_id, "ai", answer)

    # 🎯 ปรับการส่ง Sources: ใช้ Helper ที่เจน URL ให้เสร็จสรรพ
    return QueryResponse(
        answer=answer.strip(), 
        sources=_map_sources(
            chunks=all_chunks[:10],
            tenant=current_user.tenant,
            doc_type=used_doc_types[0],
            year=str(DEFAULT_YEAR), # หรือดึงจาก context
            enabler=used_enabler
        ), 
        conversation_id=conv_id
    )


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

    # แก้ไข Logic การเลือกปีใหม่ (ลบ current_user.year ออก)
    if year and year.strip() and year != "undefined":
        effective_year = year
    else:
        # หากไม่มีการส่งมาจาก UI ให้ใช้ค่ากลางของระบบทันที
        effective_year = str(DEFAULT_YEAR) 

    logger.info(f"📅 [System] Active Year: {effective_year} (Override by UI: {year is not None})")
        
    logger.info(f"📅 [Query] User Selected: {year} | Final Decision: {effective_year}")
    logger.info(f"📩 Query received: '{question}' from user {current_user.id}")

    # 🛠️ 1. Data Type Normalization
    stable_doc_ids = []
    if doc_ids:
        if isinstance(doc_ids, list):
            stable_doc_ids = [str(idx).strip() for idx in doc_ids if str(idx).strip()]
        elif isinstance(doc_ids, str):
            stable_doc_ids = [idx.strip() for idx in doc_ids.split(",") if idx.strip()]

    # กำหนด Doc Type และ Enabler
    if not doc_types:
        used_doc_types = [EVIDENCE_DOC_TYPES]
    else:
        used_doc_types = [doc_types] if isinstance(doc_types, str) else doc_types

    is_evidence = any(dt.lower() == EVIDENCE_DOC_TYPES.lower() for dt in used_doc_types)
    used_enabler = enabler or (DEFAULT_ENABLER if is_evidence else None)

    # Initialize Manager & LLM
    vsm = get_vectorstore_manager(tenant=current_user.tenant)
    llm = create_llm_instance(model_name=DEFAULT_LLM_MODEL_NAME, temperature=LLM_TEMPERATURE)

    # 🎯 2. Load Rubric JSON (โครงสร้างเกณฑ์จากไฟล์ Config)
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

    # 🎯 3. Determine Mode & Search Query
    consultant_keywords = [
        "เหมาะ", "หลักฐาน", "ประเมินหรือไม่", "สอดคล้อง", "หัวข้อไหน", 
        "เกณฑ์ไหน", "ระดับไหน", "ขาดอะไร", "พิกัด", "เลขหน้า" # <--- เพิ่มตรงนี้
    ]
    is_consultant_mode = any(kw in question.lower() for kw in consultant_keywords) or (not subject and len(stable_doc_ids) <= 2)

    search_query = question
    if subject:
        search_query = enhance_analysis_query(question, subject, rubric_data)
    elif is_consultant_mode:
        search_query = f"ผลการดำเนินงาน ตัวชี้วัด เป้าหมาย {used_enabler} PDCA {question}"

    # 🎯 4. Hybrid Retrieval (ดึงทั้งคู่มือเกณฑ์ และ หลักฐาน)
    all_evidences = []
    all_rubric_chunks = []
    
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
            top_k=RETRIEVAL_TOP_K,
            k_to_rerank=QA_FINAL_K
        )
        
        # เก็บเนื้อหาจากคลังความรู้ (SE-AM Manual/Guideline)
        if "rubric_context" in retrieval_res:
            all_rubric_chunks.extend(retrieval_res["rubric_context"])

        # ตรวจสอบหลักฐาน (Evidence) - ถ้าไม่พอให้ Retry ด้วยคำถามเดิม
        ev_list = retrieval_res.get("top_evidences", [])
        if len(ev_list) < 5:
            logger.info("♻️ Enhanced Query yields low results. Retrying with original question...")
            retry_res = await asyncio.to_thread(
                retrieve_context_with_rubric,
                vectorstore_manager=vsm,
                query=question,
                doc_type=dt,
                enabler=used_enabler,
                stable_doc_ids=stable_doc_ids,
                tenant=current_user.tenant,
                year=effective_year,
                subject=subject,
                top_k=RETRIEVAL_TOP_K
            )
            ev_list = retry_res.get("top_evidences", [])

        all_evidences.extend(ev_list)

    # ขจัดตัวซ้ำและคัดเลือกตัวที่ดีที่สุด
    unique_evidences = {ev['text']: ev for ev in all_evidences}.values()
    final_evidences = sorted(unique_evidences, key=lambda x: x.get("rerank_score", 0), reverse=True)[:ANALYSIS_FINAL_K]

    if not final_evidences:
        return QueryResponse(answer="ไม่พบเนื้อหาที่เกี่ยวข้องในเอกสารที่เลือกเพื่อนำมาวิเคราะห์ครับ", sources=[], conversation_id=conv_id)

    # 🎯 5. PDCA Assessment Engine (จัดกลุ่มข้อมูลเข้า P-D-C-A)
    engine_config = AssessmentConfig(
        tenant=current_user.tenant,
        year=int(effective_year) if effective_year.isdigit() else DEFAULT_YEAR,
        enabler=used_enabler,
        target_level=5
    )
    engine = SEAMPDCAEngine(config=engine_config, llm_instance=llm, vectorstore_manager=vsm, doc_type=used_doc_types[0])

    # กรอง Blocks ข้อมูลตามโครงสร้าง PDCA
    plan_blocks, do_blocks, check_blocks, act_blocks, other_blocks = engine._get_pdca_blocks_from_evidences(
        evidences=final_evidences, baseline_evidences={}, level=5, sub_id=subject or "all", contextual_rules_map=engine.contextual_rules_map
    )
    pdca_context = "\n\n".join(filter(None, [plan_blocks, do_blocks, check_blocks, act_blocks, other_blocks]))
    
    # รวมเนื้อหาเกณฑ์จากคลังความรู้ (Manual/Rubrics) เพื่อส่งให้ AI
    rubric_manual_context = "\n".join([r['text'] for r in all_rubric_chunks])

    # 🎯 6. Final Inference
    if is_consultant_mode:
        sys_msg_content = "ALWAYS ANSWER IN THAI.\n" + SYSTEM_CONSULTANT_INSTRUCTION
        # ส่งทั้งเกณฑ์ใน JSON และเนื้อหาจากคู่มือ PDF (Rubric Manual)
        prompt_text = REVERSE_MAPPING_PROMPT_TEMPLATE.format(
            rubric_json=rubric_json_str,
            rubric_manual=rubric_manual_context,
            documents_content=pdca_context
        )
        mode_label = "Consultant"
    else:
        sys_msg_content = "ALWAYS ANSWER IN THAI.\n" + SYSTEM_ANALYSIS_INSTRUCTION
        prompt_text = ANALYSIS_PROMPT_TEMPLATE.format(
            rubric_json=rubric_json_str,
            rubric_manual=rubric_manual_context,
            documents_content=pdca_context,
            question=question 
        )
        mode_label = "Auditor"

    logger.info(f"🚀 Analysis Mode: {mode_label} | Rubrics: {len(all_rubric_chunks)} chks | Evidence: {len(final_evidences)} chks")

    messages = [SystemMessage(content=sys_msg_content), HumanMessage(content=prompt_text)]
    raw_response = await asyncio.to_thread(llm.invoke, messages)
    answer = enforce_thai_primary_language(raw_response.content if hasattr(raw_response, "content") else str(raw_response))

    # 🎯 7. Map Sources for UI (REVISED & FIXED)
    sources = []
    for ev in final_evidences[:10]:
        # ดึง UUID ให้ชัวร์
        d_uuid = str(ev.get("doc_id") or ev.get("stable_doc_uuid"))
        
        # ดึงเลขหน้ามาทำหน้าแสดงผล (Page Display)
        p_val = ev.get('page_label') or ev.get('page') or "1"
        p_num = int(p_val) if str(p_val).isdigit() else 1
        
        # ระบุประเภทเอกสารเพื่อสร้าง URL
        current_dt = used_doc_types[0] if used_doc_types else "evidence_doc"

        sources.append(QuerySource(
            source_id=d_uuid,
            file_name=ev.get('source_filename') or ev.get('source') or 'Document',
            chunk_text=ev.get("text", "")[:500],
            score=float(ev.get("rerank_score") or 0.0),
            document_uuid=d_uuid,
            page_number=p_num,
            page_display=f"p. {p_val}", # 🟢 จุดนี้จะทำให้ N/A หายไป
            url=generate_source_url(    # 🟢 จุดนี้จะทำให้มีลิงก์กดได้
                doc_id=d_uuid,
                page=p_num,
                doc_type=current_dt,
                tenant=current_user.tenant,
                year=effective_year,
                enabler=used_enabler
            )
        ))

    await async_save_message(current_user.id, conv_id, "user", question)
    await async_save_message(current_user.id, conv_id, "ai", answer)

    return QueryResponse(
        answer=answer.strip(), 
        sources=sources, 
        conversation_id=conv_id,
        result={"process_time": round(time.time() - start_time, 2)}
    )

# =====================================================================
# 4. /files/view — PDF File Viewer Endpoint (เพิ่มส่วนนี้เข้าไป)
# =====================================================================
@llm_router.get("/files/view/{document_uuid}")
async def view_document_llm(
    document_uuid: str,
    tenant: str = "pea",
    year: Optional[str] = None,
    enabler: Optional[str] = None,
    doc_type: str = "document",
    page: int = 1
):
    file_info = get_document_file_path(
        document_uuid=document_uuid,
        tenant=tenant,
        year=year,
        enabler=enabler,
        doc_type_name=doc_type
    )

    if not file_info:
        raise HTTPException(status_code=404, detail="ไม่พบข้อมูลไฟล์")

    file_path = file_info["file_path"]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="ไม่พบไฟล์บนเซิร์ฟเวอร์")

    # 🎯 หัวใจสำคัญสำหรับ Mac:
    # 1. ห้ามใส่ filename= ใน FileResponse (เพราะมันจะเติม 'attachment' ให้ทันที)
    # 2. ใส่ Content-Disposition: inline เพียวๆ ใน headers
    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        headers={
            "Content-Disposition": "inline"
        }
    )