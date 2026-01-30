# -*- coding: utf-8 -*-
# routers/llm_router.py - Enterprise RAG (Query + Compare + PDCA Analysis + Summary)
# ULTIMATE FINAL PRODUCTION VERSION - 22 ธันวาคม 2568
# รองรับ: Multi-year evidence via wrapper, UUID v5, Clean fallback, Full intent routing

import logging
import uuid
import asyncio
from typing import List, Optional, Set, Dict, Any
from collections import defaultdict

from fastapi import APIRouter, Form, HTTPException, Depends, Request, Query
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
from utils.path_utils import (
    get_rubric_file_path, 
    get_doc_type_collection_key, 
    get_document_file_path,
    _n
)

import json, os
import time
from fastapi.responses import FileResponse
from urllib.parse import quote
import unicodedata, mimetypes



PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL")

logger = logging.getLogger(__name__)
llm_router = APIRouter(prefix="/api/llm", tags=["LLM"]) # เปลี่ยนจาก /api เป็น /api/llm

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


# =====================================================================
# Revised Helper: _map_sources
# =====================================================================
def _map_sources(
    request: Request,
    chunks: List[LcDocument], 
    tenant: str, 
    doc_type: str, 
    year: str = None, 
    enabler: str = None
) -> List[QuerySource]:
    return [
        QuerySource(
            source_id=str(c.metadata.get("doc_id", "unknown")),
            file_name=c.metadata.get("source", "Unknown"),
            chunk_text=c.page_content[:500],
            chunk_id=c.metadata.get("chunk_uuid"),
            score=float(c.metadata.get("score", 0)),
            document_uuid=str(
                c.metadata.get("stable_doc_uuid") or c.metadata.get("doc_id")
            ),
            page_number=(
                int(c.metadata.get("page", 1))
                if str(c.metadata.get("page")).isdigit()
                else 1
            ),
            page_display=f"p. {c.metadata.get('page', '1')}",
            url=generate_source_url(
                request=request,   # ✅ สำคัญที่สุด
                doc_id=str(
                    c.metadata.get("stable_doc_uuid") or c.metadata.get("doc_id")
                ),
                page=(
                    int(c.metadata.get("page", 1))
                    if str(c.metadata.get("page")).isdigit()
                    else 1
                ),
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
    request: Request,  # 👈 เพิ่มบรรทัดนี้
    question: str = Form(...),
    conversation_id: Optional[str] = Form(None),
    doc_types: List[str] = Form(default=[]),  
    doc_ids: List[str] = Form(default=[]),     
    enabler: Optional[str] = Form(None),
    subject: Optional[str] = Form(None),
    year: Optional[str] = Form(None),
    current_user: UserMe = Depends(get_current_user),
):
    
    logger.info(f"🚨 [DEBUG Identity] User Data: {current_user}")

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
                request=request, 
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
                request=request,   # 👈 จุดตายของระบบ
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
# 2. /compare — Document Comparison (Revised & Fixed Version)
# =====================================================================
@llm_router.post("/compare", response_model=QueryResponse)
async def compare_llm(
    request: Request,
    question: str = Form(...),
    doc_ids: Any = Form(...),           # รับได้ทั้ง List และ Comma-separated string
    doc_types: Optional[Any] = Form(None),
    enabler: Optional[str] = Form(None),
    year: Optional[str] = Form(None),    # 🎯 เพิ่มการรับปีจาก Frontend
    current_user: UserMe = Depends(get_current_user),
):
    conv_id = str(uuid.uuid4())
    
    logger.info(f"🚨 [DEBUG Identity] User Data: {current_user}")
    
    # 1. 🎯 จัดการเรื่อง "ปี" ให้แม่นยำ (หัวใจสำคัญที่ทำให้หาไฟล์เจอ)
    if year and year.strip() and year != "undefined":
        effective_year = year
    else:
        effective_year = str(DEFAULT_YEAR)

    # 2. Normalize doc_ids (รองรับทั้ง array และ string จาก form)
    stable_doc_ids = []
    if isinstance(doc_ids, list):
        stable_doc_ids = [str(idx).strip() for idx in doc_ids if str(idx).strip()]
    elif isinstance(doc_ids, str):
        stable_doc_ids = [idx.strip() for idx in doc_ids.split(",") if idx.strip()]

    if len(stable_doc_ids) < 2:
        raise HTTPException(400, "ต้องเลือกอย่างน้อย 2 เอกสารเพื่อเปรียบเทียบ")

    # 3. กำหนด Doc Type และ Enabler
    if not doc_types:
        used_doc_types = ["document"]
    else:
        used_doc_types = [doc_types] if isinstance(doc_types, str) else doc_types

    is_evidence = any(dt.lower() == EVIDENCE_DOC_TYPES.lower() for dt in used_doc_types)
    used_enabler = enabler or (DEFAULT_ENABLER if is_evidence else None)

    if is_evidence and not used_enabler:
        raise HTTPException(400, "สำหรับเปรียบเทียบเอกสารหลักฐาน (Evidence) ต้องระบุ Enabler ด้วยครับ")

    # 4. 🎯 เรียก Vectorstore Manager โดยล็อค Tenant และ Year ให้ตรงกัน
    # เพิ่มการส่ง year เข้าไปเพื่อให้ VSM หา Path ของ PEA/2568 เจอ
    vsm = get_vectorstore_manager(tenant=current_user.tenant, year=int(effective_year))
    collection_name = get_doc_type_collection_key(used_doc_types[0], used_enabler)
    
    logger.info(f"📊 [Compare] Tenant: {current_user.tenant} | Year: {effective_year} | Coll: {collection_name}")

    # 5. Load chunks จากเอกสารที่เลือก
    all_chunks = load_all_chunks_by_doc_ids(vsm, collection_name, set(stable_doc_ids))
    if not all_chunks:
        return QueryResponse(answer="ไม่พบข้อมูลในเอกสารที่เลือกเพื่อนำมาเปรียบเทียบครับ", sources=[], conversation_id=conv_id)

    # จัดกลุ่ม chunks ตามเอกสาร
    doc_groups = defaultdict(list)
    for d in all_chunks:
        doc_key = str(d.metadata.get("stable_doc_uuid") or d.metadata.get("doc_id"))
        doc_groups[doc_key].append(d)

    doc_blocks = []
    for idx, d_id in enumerate(stable_doc_ids, start=1):
        chunks = doc_groups.get(str(d_id), [])
        if not chunks:
            block = f"### เอกสารที่ {idx}\n(ไม่พบข้อมูลเนื้อหาในฐานข้อมูล)"
        else:
            fname = chunks[0].metadata.get("source", f"ID:{d_id}")
            # จำกัด 15 chunks เพื่อความแม่นยำของ Llama 3:70B
            body = "\n".join(f"- {c.page_content}" for c in chunks[:15]) 
            block = f"### เอกสารที่ {idx}: {fname}\n{body}"
        doc_blocks.append(block)

    # 6. Prepare LLM & Inference
    llm = create_llm_instance(model_name=DEFAULT_LLM_MODEL_NAME, temperature=0.1)
    
    # บังคับ Prompt ให้ตอบไทยสละสลวย (Llama 3 Friendly)
    thai_enforcement = "\n\n(ย้ำ: สรุปผลการเปรียบเทียบเป็นภาษาไทยสละสลวยเท่านั้น ห้ามตอบเป็นภาษาอังกฤษ)"
    full_query = f"{question}{thai_enforcement}"
    
    prompt_text = COMPARE_PROMPT_TEMPLATE.format(
        documents_content="\n\n".join(doc_blocks), 
        query=full_query
    )

    messages = [
        SystemMessage(content=SYSTEM_COMPARE_INSTRUCTION),
        HumanMessage(content=prompt_text),
        HumanMessage(content="จงสรุปความแตกต่างและเปรียบเทียบข้อมูลด้านบนในรูปแบบตารางภาษาไทยที่อ่านง่าย")
    ]

    raw = await asyncio.to_thread(llm.invoke, messages)
    answer = enforce_thai_primary_language(raw.content if hasattr(raw, "content") else str(raw))

    # 7. 🎯 สร้าง Sources พร้อม URL ที่ถูกต้อง (ล็อค Year และ Tenant)
    sources = _map_sources(
        request=request,
        chunks=all_chunks[:10],
        tenant=current_user.tenant,
        doc_type=used_doc_types[0],
        year=effective_year,    # ✅ ใช้ปีที่เลือกจริง ไม่ใช้ค่า Default
        enabler=used_enabler
    )

    await async_save_message(current_user.id, conv_id, "user", question)
    await async_save_message(current_user.id, conv_id, "ai", answer)

    return QueryResponse(answer=answer.strip(), sources=sources, conversation_id=conv_id)


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
# 3. /analysis — PDCA-focused SE-AM analysis (v2026.02.04-final-stable-ultimate)
# =====================================================================
@llm_router.post("/analysis", response_model=QueryResponse)
async def analysis_llm(
    request: Request,
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

    # 1. Year Management
    effective_year = str(DEFAULT_YEAR)
    if year and str(year).strip().lower() not in ("undefined", "none", ""):
        try:
            effective_year = str(int(year.strip()))
        except ValueError:
            logger.warning(f"Invalid year format: {year} → fallback to {DEFAULT_YEAR}")

    logger.info(f"🔍 [Analysis] User: {current_user.id} | Tenant: {current_user.tenant} | Year: {effective_year} | Q: {question[:80]}...")

    # 2. Normalize Inputs
    stable_doc_ids = []
    if doc_ids:
        if isinstance(doc_ids, list):
            stable_doc_ids = [str(idx).strip() for idx in doc_ids if str(idx).strip()]
        elif isinstance(doc_ids, str):
            stable_doc_ids = [idx.strip() for idx in doc_ids.split(",") if idx.strip()]

    used_doc_types = [EVIDENCE_DOC_TYPES] if not doc_types else (
        [doc_types] if isinstance(doc_types, str) else doc_types
    )

    is_evidence = any(dt.lower() == EVIDENCE_DOC_TYPES.lower() for dt in used_doc_types)
    used_enabler = enabler or (DEFAULT_ENABLER if is_evidence else None)

    # 3. Initialize Resources
    try:
        vsm = get_vectorstore_manager(tenant=current_user.tenant, year=int(effective_year))
    except Exception:
        logger.warning("Vectorstore fallback to DEFAULT_YEAR")
        vsm = get_vectorstore_manager(tenant=current_user.tenant, year=DEFAULT_YEAR)

    llm = create_llm_instance(model_name=DEFAULT_LLM_MODEL_NAME, temperature=LLM_TEMPERATURE)

    # 4. Load Rubric (fallback safe)
    rubric_json_str = "{}"
    try:
        rubric_path = get_rubric_file_path(current_user.tenant, used_enabler)
        if os.path.exists(rubric_path):
            with open(rubric_path, 'r', encoding='utf-8') as f:
                rubric_json_str = f.read()  # ใช้ raw string เพื่อความเร็ว
    except Exception as e:
        logger.error(f"Rubric load failed: {e}")

    # 5. Mode & Search Query
    consultant_keywords = ["เหมาะ", "หลักฐาน", "ประเมิน", "สอดคล้อง", "หัวข้อไหน", "เกณฑ์ไหน", "ระดับไหน", "ขาดอะไร"]
    is_consultant_mode = any(kw in question.lower() for kw in consultant_keywords) or (not subject and len(stable_doc_ids) <= 2)

    search_query = question
    if subject:
        search_query = enhance_analysis_query(question, subject, rubric_json_str)
    elif is_consultant_mode:
        search_query = f"ผลการดำเนินงาน ตัวชี้วัด {used_enabler} {question}"

    # 6. Retrieval
    all_evidences = []
    all_rubric_chunks = []
    for dt in used_doc_types:
        try:
            res = await asyncio.to_thread(
                retrieve_context_with_rubric,
                vectorstore_manager=vsm, query=search_query, doc_type=dt,
                enabler=used_enabler, stable_doc_ids=stable_doc_ids,
                tenant=current_user.tenant, year=effective_year,
                subject=subject, top_k=RETRIEVAL_TOP_K, k_to_rerank=QA_FINAL_K
            )
            all_rubric_chunks.extend(res.get("rubric_context", []))
            ev_list = res.get("top_evidences", [])
            if len(ev_list) < 5:
                logger.info("Low evidence → retry original query")
                retry_res = await asyncio.to_thread(
                    retrieve_context_with_rubric,
                    vectorstore_manager=vsm, query=question, doc_type=dt,
                    enabler=used_enabler, stable_doc_ids=stable_doc_ids,
                    tenant=current_user.tenant, year=effective_year,
                    top_k=RETRIEVAL_TOP_K
                )
                ev_list = retry_res.get("top_evidences", [])
            all_evidences.extend(ev_list)
        except Exception as e:
            logger.error(f"Retrieval failed ({dt}): {e}")

    # Deduplicate & Sort
    unique_evidences = {ev.get("text", ""): ev for ev in all_evidences if ev.get("text")}.values()
    final_evidences = sorted(unique_evidences, key=lambda x: x.get("rerank_score", 0), reverse=True)[:ANALYSIS_FINAL_K]

    if not final_evidences:
        return QueryResponse(answer="ไม่พบหลักฐานที่เกี่ยวข้อง", sources=[], conversation_id=conv_id)

    # 7. PDCA Engine
    engine = SEAMPDCAEngine(
        config=AssessmentConfig(
            tenant=current_user.tenant, 
            year=int(effective_year), 
            enabler=used_enabler
        ),
        llm_instance=llm,
        vectorstore_manager=vsm, # ระบุชื่อชัดเจน จะไม่อ่านผิดเป็น logger อีกต่อไป
        doc_type=used_doc_types[0]
    )


    pdca_res = engine._get_pdca_blocks_from_evidences(
        final_evidences, {}, 5, subject or "all", engine.contextual_rules_map
    )
    valid_blocks = []
    counts = pdca_res.get("actual_counts", {})
    for tag in ["P", "D", "C", "A"]:
        content = pdca_res.get(tag, "")
        if content and "[ไม่พบหลักฐาน" not in content:
            valid_blocks.append(f"### หมวด {tag}:\n{content}")

    pdca_context = f"### สรุปหลักฐานจริง: P={counts.get('P',0)}, D={counts.get('D',0)}, C={counts.get('C',0)}, A={counts.get('A',0)}\n\n"
    pdca_context += "\n\n".join(valid_blocks) if valid_blocks else "ไม่พบหลักฐาน PDCA"

    rubric_manual = "\n".join([r['text'] for r in all_rubric_chunks])

    # -------------------------------------------------------
    # 8. Final Inference (v2026 Stable Replacement)
    # -------------------------------------------------------
    # -------------------------------------------------------
    # 8. Final Inference (v2026.01.30 - Anti-Crash & Auto-Extraction)
    # -------------------------------------------------------
    sys_msg = SYSTEM_CONSULTANT_INSTRUCTION if is_consultant_mode else SYSTEM_ANALYSIS_INSTRUCTION
    raw_template = REVERSE_MAPPING_PROMPT_TEMPLATE if is_consultant_mode else ANALYSIS_PROMPT_TEMPLATE

    try:
        # [FIX] ดึง string จาก LangChain PromptTemplate หรือใช้ string ตรงๆ
        template_str = raw_template.template if hasattr(raw_template, 'template') else str(raw_template)

        # [SAFE REPLACE] ใช้ .replace แทน .format เพื่อป้องกัน JSON Braces Error
        prompt_text = (
            template_str.replace("{rubric_json}", rubric_json_str)
                        .replace("{rubric_manual}", rubric_manual or "ไม่มีข้อมูลคู่มือเพิ่มเติม")
                        .replace("{documents_content}", pdca_context)
                        .replace("{question}", question)
        )
        
        # ล้างปีกกาคู่ {{ }} ให้เหลือ { } (กรณี template เดิมทำไว้เพื่อหนี .format)
        prompt_text = prompt_text.replace("{{", "{").replace("}}", "}")
        
        logger.info(f"✅ Prompt prepared successfully (Length: {len(prompt_text)})")
        
    except Exception as fmt_err:
        logger.error(f"❌ Critical Prompt Error: {fmt_err}")
        prompt_text = f"คำถาม: {question}\nบริบท: {pdca_context[:2000]}"

    messages = [
        SystemMessage(content=f"ALWAYS ANSWER IN THAI.\n{sys_msg}"),
        HumanMessage(content=prompt_text)
    ]

    raw_response = None
    try:
        raw_response = await asyncio.to_thread(llm.invoke, messages)
    except Exception as e:
        logger.error(f"❌ LLM invocation failed: {e}")
        # สร้าง Mock object เพื่อให้ code รันต่อได้ไม่ crash
        raw_response = type('obj', (object,), {'content': json.dumps({"text": "เกิดข้อผิดพลาดในการเชื่อมต่อ LLM", "score": 0.0})})

    # 9. Robust Extraction & Cleaning
    from core.json_extractor import _robust_extract_json

    raw_content = ""
    if raw_response:
        raw_content = raw_response.content if hasattr(raw_response, "content") else str(raw_response)
        logger.debug(f"[RAW-RESPONSE] Length: {len(raw_content)} | Preview: {raw_content[:400]}...")

    structured_data = {}
    if raw_content:
        try:
            structured_data = _robust_extract_json(raw_content) or {}
            logger.debug(f"[PARSED] Keys: {list(structured_data.keys())} | Score: {structured_data.get('score')}")
        except Exception as e:
            logger.error(f"JSON extract failed: {e}")
            structured_data = {"text": raw_content[:1000], "reason": "JSON Parse Error"}

    display_answer = (
        structured_data.get("text") or
        structured_data.get("คำตอบ") or
        structured_data.get("reason") or
        raw_content or
        "ไม่สามารถวิเคราะห์ได้"
    )

    # 10. Final Clean
    final_answer = display_answer.replace("```json", "").replace("```markdown", "").replace("```", "").strip()
    final_answer = final_answer.replace("\\n", "\n").replace('\\"', '"').replace("\\\"", "\"")
    if final_answer.startswith('"') and final_answer.endswith('"'):
        final_answer = final_answer[1:-1].strip()

    final_answer = enforce_thai_primary_language(final_answer)

    # 11. Sources Mapping
    sources = []
    for ev in final_evidences[:10]:
        d_uuid = str(ev.get("doc_id") or ev.get("stable_doc_uuid", "N/A"))
        p_val = ev.get("page_label") or ev.get("page") or "1"
        p_num = int(p_val) if str(p_val).isdigit() else 1

        sources.append(QuerySource(
            source_id=d_uuid,
            file_name=ev.get("source_filename") or "Document",
            chunk_text=ev.get("text", "")[:500],
            score=float(ev.get("rerank_score") or 0.0),
            document_uuid=d_uuid,
            page_number=p_num,
            page_display=f"p. {p_val}",
            url=generate_source_url(
                request=request,
                doc_id=d_uuid,
                page=p_num,
                doc_type=used_doc_types[0],
                tenant=current_user.tenant,
                year=effective_year,
                enabler=used_enabler
            )
        ))

    # 12. Save & Return
    await async_save_message(current_user.id, conv_id, "user", question)
    await async_save_message(current_user.id, conv_id, "ai", final_answer)

    return QueryResponse(
        answer=final_answer,
        sources=sources,
        conversation_id=conv_id,
        result={
            "process_time": round(time.time() - start_time, 2),
            "structured": structured_data
        }
    )
    
# =====================================================================
# Revised Helper: generate_source_url (หัวใจของการเชื่อมโยงไฟล์)
# =====================================================================
def generate_source_url(
    request: Request,
    doc_id: str, 
    page: int, 
    doc_type: str, 
    tenant: str, 
    year: str, 
    enabler: Optional[str] = None
) -> str:
    if not doc_id or doc_id == "unknown":
        return ""

    auth_header = request.headers.get("Authorization", "")
    token = auth_header.replace("Bearer ", "") if auth_header else ""

    if PUBLIC_BASE_URL:
        base_url = PUBLIC_BASE_URL.rstrip("/")
    else:
        base_url = f"{request.url.scheme}://{request.url.netloc}"

    # 🎯 ต้องใส่ /llm เพิ่มเข้าไปให้ตรงกับ Prefix ของ APIRouter
    endpoint_path = f"/api/llm/files/view/{doc_id}"
    
    url = f"{base_url}{endpoint_path}"

    p_num = max(1, int(page) if str(page).isdigit() else 1)
    params = [
        f"page={p_num}", 
        f"doc_type={doc_type.lower()}", 
        f"tenant={tenant}",
        f"year={year}",
        f"token={token}"
    ]

    if doc_type.lower() == EVIDENCE_DOC_TYPES.lower() and enabler:
        params.append(f"enabler={enabler}")

    return f"{url}?{'&'.join(params)}"

# =====================================================================
# 4. /files/view — PDF File Viewer Endpoint (Revised เพื่อความแม่นยำ)
# =====================================================================
@llm_router.get("/files/view/{document_uuid}")
async def view_document_llm(
    document_uuid: str,
    tenant: str,               
    year: Optional[str] = None, 
    enabler: Optional[str] = None,
    doc_type: str = "document",
    page: int = 1,
    # 🟢 เพิ่ม token ตรงนี้เพื่อรับจาก URL (เพราะ window.open ส่ง header ไม่ได้)
    token: Optional[str] = Query(None) 
):
    """
    เวอร์ชันปรับปรุงตาม upload_router.py ที่ใช้งานได้จริง
    """
    
    # 🕵️ หมายเหตุ: หากระบบคุณเข้มงวดเรื่อง Security 
    # คุณควรนำ token ไปตรวจสอบความถูกต้องก่อนตรงนี้
    # if not token: raise HTTPException(status_code=401)

    dt_clean = _n(doc_type)
    
    # 1. กำหนดปีที่จะค้นหา (เลียนแบบ Logic ใน upload_router)
    from config.global_vars import EVIDENCE_DOC_TYPES, DEFAULT_YEAR
    if dt_clean != _n(EVIDENCE_DOC_TYPES):
        search_year = None
    else:
        # พยายามแปลง year เป็น int เหมือนใน upload_router
        try:
            search_year = int(year) if year and year != "undefined" else DEFAULT_YEAR
        except:
            search_year = DEFAULT_YEAR

    # 2. ค้นหา Path (ใช้ get_document_file_path เหมือนกัน)
    resolved = get_document_file_path(document_uuid, tenant, search_year, enabler, doc_type)
    
    if not resolved:
         logger.error(f"❌ View failed: Mapping not found for {document_uuid}")
         raise HTTPException(status_code=404, detail="ไม่พบรหัสไฟล์ในฐานข้อมูล")

    # 3. จัดการ Path ภาษาไทย (Logic เดียวกับ upload_router เป๊ะๆ)
    target_path = resolved["file_path"]
    normalized_path = unicodedata.normalize('NFC', target_path)
    
    if not os.path.exists(normalized_path):
        normalized_path = unicodedata.normalize('NFD', target_path)
        if not os.path.exists(normalized_path):
            logger.error(f"❌ File missing on disk: {target_path}")
            raise HTTPException(status_code=404, detail="ไม่พบไฟล์จริงบนดิสก์")

    # 4. ระบุ MIME Type
    m_type, _ = mimetypes.guess_type(normalized_path)
    file_ext = normalized_path.lower()
    if not m_type:
        if file_ext.endswith('.pdf'): m_type = 'application/pdf'
        elif file_ext.endswith('.png'): m_type = 'image/png'
        else: m_type = 'application/octet-stream'

    # 5. สร้าง Headers เหมือน upload_router
    filename = resolved["original_filename"]
    encoded_filename = quote(filename)
    
    headers = {
        "Content-Disposition": f"inline; filename=\"{encoded_filename}\"; filename*=UTF-8''{encoded_filename}",
        "Cache-Control": "no-cache"
    }

    logger.info(f"✅ [Chat View] Serving: {filename} as {m_type}")

    return FileResponse(
        path=normalized_path,
        media_type=m_type,
        headers=headers
    )