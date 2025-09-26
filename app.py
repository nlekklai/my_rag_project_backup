# app.py
import os
import re
import shutil
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

# --- Local imports ---
from rag_chain import compare_documents, answer_question, create_rag_chain, run_assessment_workflow, QA_PROMPT
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

    # filter metadata ตามคำถาม
    if "2566" in question_lower or "2567" in question_lower:
        matched = [d for d in all_docs if "pea" in d.lower() and any(y in d for y in ["2566","2567"])]
    elif "seam" in question_lower:
        matched = [d for d in all_docs if "seam" in d.lower()]
    else:
        matched = all_docs
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


@app.post("/compare")
async def compare(
    doc1: str = Form(...),
    doc2: str = Form(...),
    question: str = Form("เปรียบเทียบหัวข้อหลักและเนื้อหาสำคัญของเอกสารทั้งสอง"),
):
    try:
        # สรุป doc1/doc2
        def summarize_docs():
            chain1 = create_rag_chain(doc1)
            chain2 = create_rag_chain(doc2)
            summary1 = chain1.invoke({"query": "สรุปหัวข้อหลักและเนื้อหาสำคัญของเอกสารนี้"}).get("result", "")
            summary2 = chain2.invoke({"query": "สรุปหัวข้อหลักและเนื้อหาสำคัญของเอกสารนี้"}).get("result", "")
            return summary1, summary2

        summary1, summary2 = await run_in_threadpool(summarize_docs)

        # เปรียบเทียบ
        metrics = await run_in_threadpool(lambda: compare_documents(summary1, summary2, question))
        return {"result": metrics}
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
async def ask(question: str = Form(...)):
    doc_id = get_latest_doc_id()
    question = "สรุปหัวข้อหลักและเนื้อหาสำคัญสั้น ๆ ของเอกสารนี้ โดยแยกบรรทัดและใส่ bullet (*) สำหรับแต่ละหัวข้อย่อย"
    result = await run_in_threadpool(answer_question, question, doc_id)
    return {"answer": result}


@app.post("/query")
async def query_documents(question: str = Form(...)):
    try:
        doc_ids = match_doc_ids_from_question(question)
        if not doc_ids:
            retriever = load_all_vectorstores()
        elif len(doc_ids) == 1:
            retriever = load_vectorstore(doc_ids[0])
        else:
            retrievers = [load_vectorstore(d) for d in doc_ids]
            retriever = MultiDocRetriever(retrievers_list=retrievers)

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": QA_PROMPT, "document_variable_name": "context"},
        )

        answer = await run_in_threadpool(lambda: chain.invoke({"query": question})["result"])
        return {"answer": answer, "doc_ids": doc_ids}
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


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
