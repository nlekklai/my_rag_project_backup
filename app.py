# app.py
import os
import re
import shutil
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from typing import List, Optional

# --- Local imports ---
from rag_chain import compare_documents, answer_question, create_rag_chain, run_assessment_workflow, QA_PROMPT, COMPARE_PROMPT
from vectorstore import (
    load_vectorstore,
    load_all_vectorstores,
    MultiDocRetriever,
    list_vectorstore_folders,
)
from ingest import process_document, list_documents, delete_document
from assessment_state import process_state
from models.llm import get_llm

from langchain.chains import RetrievalQA

# -------------------- Setup --------------------
app = FastAPI()
ASSESSMENT_DIR = "assessment_data"
RESULTS_DIR = "results"
os.makedirs(ASSESSMENT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = get_llm()  # single instance


# -------------------- Helper Functions --------------------
def clear_assessment_folder():
    """ลบไฟล์และโฟลเดอร์ทั้งหมดใน assessment_data"""
    if os.path.exists(ASSESSMENT_DIR):
        for item in os.listdir(ASSESSMENT_DIR):
            path = os.path.join(ASSESSMENT_DIR, item)
            try:
                if os.path.isfile(path) or os.path.islink(path):
                    os.unlink(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)
            except Exception as e:
                print(f"❌ Cannot delete {path}: {e}")


def match_doc_ids_from_question(question: str):
    all_docs = list_vectorstore_folders()
    question_lower = question.lower()
    matched = []
    
    # 1. ตรวจจับคำถามเปรียบเทียบ (เทียบ, แตกต่าง, compare)
    is_comparison_query = any(keyword in question_lower for keyword in ["เทียบ", "แตกต่าง", "compare"])
    
    # 2. พยายามดึงชื่อเอกสารจากคำถาม (ใช้ Regex ที่ยืดหยุ่นกว่า)
    # เช่น "เทียบ 2566-PEA กับ 2567-PEA" -> ['2566-PEA', '2567-PEA']
    # หรือ "Compare the plan A with plan B"
    
    # ดึงคำที่คล้าย doc_id/ชื่อเฉพาะ (เช่นมีขีด/ตัวเลขติดกัน)
    potential_doc_names = re.findall(r'(\w+-\w+|\w+_\w+|\d{4})', question_lower)
    
    # 3. สร้าง list ของ doc_id ที่ตรงกับคำที่ตรวจพบ
    explicit_matches = [
        doc_id for doc_id in all_docs 
        if any(p_name in doc_id.lower() for p_name in potential_doc_names)
    ]
    
    if explicit_matches:
        matched = explicit_matches
    elif is_comparison_query:
        # หากเป็นการเปรียบเทียบ แต่ไม่ระบุชื่อเอกสารชัดเจน ให้ใช้เอกสารทั้งหมด
        matched = all_docs
    else:
        # หากเป็นการ Query ทั่วไป และไม่มีการระบุชื่อเอกสารชัดเจน ก็ใช้เอกสารทั้งหมด
        matched = all_docs
        
    # 4. สำหรับการเปรียบเทียบ 2566/2567 ที่เคย hardcode (เพื่อรักษา feature เดิม)
    # หากมีการใช้ปี 2566, 2567 และเป็นคำถามเปรียบเทียบ ให้ใช้เอกสาร PEA สองตัวนั้น
    if is_comparison_query and any(y in question_lower for y in ["2566", "2567"]):
        # Note: ยังคงต้องมี hardcode logic เล็กน้อยสำหรับ use case นี้
        # แต่ควรมี doc_id ที่ถูกตั้งชื่อโดย user แล้ว 
        # เช่น ถ้ามี ['doc-a', '2566-PEA', '2567-PEA'] ใน all_docs 
        matched = [d for d in all_docs if any(y in d for y in ["2566","2567"])]

    # หากสุดท้ายแล้วไม่มีอะไร match ให้ใช้เอกสารทั้งหมด
    if not matched:
        matched = all_docs

    # การป้องกัน: หาก match ได้ 1 doc แต่คำถามต้องการ compare ให้กลับไปใช้ all_docs
    if is_comparison_query and len(matched) < 2 and len(all_docs) >= 2:
         return all_docs # บังคับใช้ทั้งหมดเพื่อให้ LLM เปรียบเทียบได้

    return matched


def get_latest_doc_id():
    docs = list_documents()
    if not docs:
        raise HTTPException(status_code=404, detail="No documents found")
    latest = docs[-1]
    return latest.get("id") if isinstance(latest, dict) else latest


# -------------------- API Endpoints --------------------
@app.post("/upload")
async def upload_file(file: UploadFile):
    os.makedirs("data", exist_ok=True)
    file_path = os.path.join("data", file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    doc_id = process_document(file_path, file.filename)
    return {
        "status": "ok",
        "doc_id": doc_id,
        "filename": file.filename,
        "file_type": os.path.splitext(file.filename)[1].lower(),
        "upload_date": datetime.utcnow().isoformat(),
    }

# app.py (Modified /compare endpoint)

@app.post("/compare")
async def compare(
    doc1: str = Form(...),
    doc2: str = Form(...),
    # 💡 NOTE: The question parameter from Form(...) is now IGNORED for the RAG chain
    # We use a hard-coded question to guarantee results based on the most critical differences.
    question: str = Form(
        "เปรียบเทียบเอกสาร {doc1} กับ {doc2} โดยสรุป **เฉพาะความแตกต่าง** "
        "ที่สำคัญเป็นข้อๆ (bullet points) ในด้าน **เป้าหมาย ตัวชี้วัด และระยะเวลาดำเนินการ**"
    ),
):
    """
    เปรียบเทียบเอกสาร 2 ฉบับโดยใช้ MultiDocRetriever และ COMPARE_PROMPT โดยตรง
    """
    try:
        doc_ids_list = [doc1, doc2]
        retrievers = [load_vectorstore(d) for d in doc_ids_list]
        retriever = MultiDocRetriever(retrievers_list=retrievers)

        # -------------------------------------------------------------
        # 💡 FIX: Hard-code the RAG question to focus on the confirmed differences.
        # This question will retrieve the chunks that contain the timeline/KPI changes.
        RAG_COMPARISON_QUESTION = (
            f"เปรียบเทียบเอกสาร {doc1} กับ {doc2} อย่างละเอียด โดยสรุปเป็นข้อ (bullet points) "
            f"**เฉพาะความแตกต่าง** ในด้าน **เป้าหมายระยะยาว (KPIs) และระยะเวลาดำเนินการ (Timeline)** เท่านั้น"
        )
        # -------------------------------------------------------------
        
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": COMPARE_PROMPT, "document_variable_name": "context"},
        )
        
        # 4. รัน Chain เพื่อหาความแตกต่าง (delta) โดยตรง โดยใช้ RAG_COMPARISON_QUESTION
        delta_answer = await run_in_threadpool(lambda: chain.invoke({"query": RAG_COMPARISON_QUESTION})["result"])

        # 5. สรุป Doc1/Doc2 (ใช้ QA_PROMPT เดิม)
        def summarize_docs():
            summary_query = "สรุปหัวข้อหลักและเนื้อหาสำคัญของเอกสารนี้"
            chain1 = create_rag_chain(doc1, QA_PROMPT) 
            chain2 = create_rag_chain(doc2, QA_PROMPT)
            summary1 = chain1.invoke({"query": summary_query}).get("result", "")
            summary2 = chain2.invoke({"query": summary_query}).get("result", "")
            return summary1, summary2

        summary1, summary2 = await run_in_threadpool(summarize_docs)

        # 6. จัดรูปแบบผลลัพธ์ JSON
        return {
            "result": {
                "metrics": [{
                    "metric": "หัวข้อหลักและเนื้อหาสำคัญ",
                    "doc1": summary1,
                    "doc2": summary2,
                    "delta": delta_answer, 
                    "remark": "ผลการเปรียบเทียบได้จาก Multi-Document Retrieval ที่เน้นความแตกต่าง"
                }]
            }
        }
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"การเปรียบเทียบไม่สำเร็จ: {e}")


@app.post("/ask")
async def ask(question: str = Form(...)):
    doc_id = get_latest_doc_id()
    # 💡 ปรับให้ใช้คำถามที่ผู้ใช้ส่งมาแทนคำถามตายตัว Later
    question = "สรุปหัวข้อหลักและเนื้อหาสำคัญสั้น ๆ ของเอกสารนี้ โดยแยกบรรทัดและใส่ bullet (*) สำหรับแต่ละหัวข้อย่อย"
    result = await run_in_threadpool(answer_question, question, doc_id)
    return {"answer": result}

@app.post("/query")
async def query_documents(
    question: str = Form(...),
    doc_ids: Optional[str] = Form(None)  # comma-separated string จาก UI
):
    # กำหนดค่า k ที่ต้องการ: 
    # 💡 เพื่อหา '8 Enablers' ที่มีข้อมูลกระจัดกระจาย ลองเพิ่ม k เป็น 8 หรือ 10
    K_VALUE = 8 
    
    # 1. แปลง doc_ids เป็น list
    if doc_ids:
        doc_ids_list = [d.strip() for d in doc_ids.split(",") if d.strip()]
    else:
        # fallback: match จากคำถาม
        doc_ids_list = match_doc_ids_from_question(question)

    # 2. โหลด retriever
    if not doc_ids_list:
        retriever = load_all_vectorstores()
        doc_ids_list = list_vectorstore_folders() # update list
    elif len(doc_ids_list) == 1:
        # ✅ FIX: เปลี่ยน k=K_VALUE เป็น top_k=K_VALUE เพื่อให้ตรงกับ vectorstore.py
        retriever = load_vectorstore(doc_ids_list[0], top_k=K_VALUE) 
    else:
        # Multi-document retrieval
        # ✅ FIX: เปลี่ยน k=K_VALUE เป็น top_k=K_VALUE เพื่อให้ตรงกับ vectorstore.py
        retrievers = [load_vectorstore(d, top_k=K_VALUE) for d in doc_ids_list] 
        retriever = MultiDocRetriever(retrievers_list=retrievers)

    # 3. เลือก Prompt ตามประเภท Query
    # 💡 ใช้ COMPARE_PROMPT เมื่อมีการใช้ MultiDocRetriever สำหรับการเปรียบเทียบ
    if len(doc_ids_list) > 1 and ("เทียบ" in question or "แตกต่าง" in question):
        prompt_to_use = COMPARE_PROMPT
    else:
        prompt_to_use = QA_PROMPT

    # 4. ทำ QA
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_to_use, "document_variable_name": "context"},
    )

    answer = await run_in_threadpool(lambda: chain.invoke({"query": question})["result"])
    return {"answer": answer, "doc_ids": doc_ids_list}


# -------------------- Document Management --------------------
@app.get("/api/documents")
async def get_documents():
    return list_documents()


@app.delete("/api/documents/{doc_id}")
async def remove_document(doc_id: str):
    try:
        delete_document(doc_id)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# -------------------- Assessment Workflow --------------------
@app.post("/process/start")
async def start_process(background_tasks: BackgroundTasks):
    for step in process_state["steps"]:
        step["status"] = "waiting"

    async def workflow_with_cleanup():
        await run_assessment_workflow()
        clear_assessment_folder()

    background_tasks.add_task(workflow_with_cleanup)
    return {"message": "Processing started"}


@app.get("/process/status")
async def get_status():
    return process_state

@app.get("/api/status")
def status():
    return {"status": "ok"}


@app.get("/result/{type}")
async def download_result(type: str):
    file_map = {
        "score": os.path.join(RESULTS_DIR, "score.xlsx"),
        "evidence": os.path.join(RESULTS_DIR, "evidence.xlsx"),
        "gap": os.path.join(RESULTS_DIR, "gap.xlsx"),
    }
    if type not in file_map:
        raise HTTPException(status_code=400, detail="Invalid type")
    return FileResponse(file_map[type], media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


@app.get("/result/summary")
async def get_summary():
    return {
        "total_questions": 120,
        "avg_score": 82.5,
        "evidence_found": 95,
        "gaps": 25,
    }


@app.post("/upload/{type}")
async def upload_assessment_file(type: str, file: UploadFile = File(...)):
    if type not in ["rubrics", "qa", "evidence"]:
        raise HTTPException(status_code=400, detail="Invalid type")

    folder = os.path.join(ASSESSMENT_DIR, type)
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    return {
        "status": "ok",
        "type": type,
        "filename": file.filename,
        "file_type": os.path.splitext(file.filename)[1].lower(),
        "upload_date": datetime.utcnow().isoformat(),
    }
