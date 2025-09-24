from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from ingest import process_document, list_documents, delete_document
from rag_chain import compare_documents, answer_question
from models.llm import get_llm
from vectorstore import load_vectorstore
from langchain.chains import RetrievalQA
import os
from datetime import datetime
import re
from assessment_state import process_state
from rag_chain import run_assessment_workflow
import shutil


llm = get_llm()
app = FastAPI()

ASSESSMENT_DIR = "assessment_data"
os.makedirs(ASSESSMENT_DIR, exist_ok=True)

def clear_assessment_folder():
    """ลบไฟล์และโฟลเดอร์ย่อยทั้งหมดใน assessment_data"""
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
    else:
        os.makedirs(ASSESSMENT_DIR, exist_ok=True)


# ---- CORS ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def extract_doc_id_from_question(question: str):
    match = re.search(r"(?:เอกสาร\s+)?([\d]{4}-[A-Z]+)(?:\.pdf)?", question, re.IGNORECASE)
    if match:
        return match.group(1)
    return None

# ---- Helper ----
def get_latest_doc_id():
    docs = list_documents()
    if not docs:
        raise HTTPException(status_code=404, detail="No documents found")
    latest = docs[-1]

    if isinstance(latest, dict):
        return latest.get("id")   # ✅ ตอนนี้ได้ doc_id ตรง ๆ แล้ว
    return latest


# ---- Endpoints ----
@app.post("/upload")
async def upload_file(file: UploadFile):
    try:
        os.makedirs("data", exist_ok=True)
        file_path = os.path.join("data", file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Process document -> return doc_id ตรงกับ vectorstore folder
        doc_id = process_document(file_path, file.filename)

        return {
            "status": "ok",
            "doc_id": doc_id,
            "filename": file.filename,
            "file_type": os.path.splitext(file.filename)[1].lower(),
            "upload_date": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/compare")
async def compare(
    doc1: str = Form(...),
    doc2: str = Form(...),
    question: str = Form("สรุปหัวข้อหลักและเนื้อหาสำคัญของเอกสารทั้งสองฉบับ")
):
    result = compare_documents(doc1, doc2, question)  # ไม่ส่ง question เข้าไป
    return {"result": result}



@app.post("/ask")
async def ask(question: str = Form(...)):
    doc_id = get_latest_doc_id()
    question = "สรุปหัวข้อหลักและเนื้อหาสำคัญสั้น ๆ ของเอกสารนี้ โดยแยกบรรทัดและใส่ bullet (*) สำหรับแต่ละหัวข้อย่อย"
    result = await run_in_threadpool(answer_question, question, doc_id)
    return {"answer": result}

@app.get("/status")
async def status():
    return {
        "compliance_score": 85,
        "total_documents": len(list_documents()),
        "recent_activities": [],
        "metrics": {"processed": 10, "pending": 2, "errors": 0}
    }

@app.post("/query")
async def query_documents(question: str = Form(...), doc_id: str = Form(None)):
    try:
        if not doc_id:
            doc_id = extract_doc_id_from_question(question)
        if not doc_id:
            doc_id = get_latest_doc_id()

        # --- สร้าง chain พร้อม prompt ภาษาไทย ---
        from rag_chain import QA_PROMPT, get_llm
        from vectorstore import load_vectorstore
        from langchain.chains import RetrievalQA

        vs = load_vectorstore(doc_id)
        retriever = vs.as_retriever(search_kwargs={"k": 3})

        chain = RetrievalQA.from_chain_type(
            llm=get_llm(),
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": QA_PROMPT, "document_variable_name": "context"}
        )

        # --- เรียก RAG chain ---
        result = chain.run(question)
        return {"answer": result, "doc_id": doc_id}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---- Document management endpoints ----
@app.get("/api/documents")
async def get_documents():
    docs = list_documents()
    return docs

@app.delete("/api/documents/{doc_id}")
async def remove_document(doc_id: str):
    try:
        delete_document(doc_id)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

#Asssessment API to support lovable.dev assessment ui page

@app.post("/process/start")
async def start_process(background_tasks: BackgroundTasks):
    # รีเซ็ต status ก่อนเริ่ม
    for step in process_state["steps"]:
        step["status"] = "waiting"

    async def workflow_with_cleanup():
        # 1️⃣ Run assessment workflow (process files → vectorstore)
        await run_assessment_workflow()
        
        # 2️⃣ ลบไฟล์เก่าใน assessment_data หลังสร้าง vectorstore เสร็จ
        clear_assessment_folder()

    background_tasks.add_task(workflow_with_cleanup)
    return {"message": "Processing started"}


@app.get("/process/status")
async def get_status():
    return process_state

@app.get("/result/{type}")
async def download_result(type: str):
    file_map = {
        "score": "results/score.xlsx",
        "evidence": "results/evidence.xlsx",
        "gap": "results/gap.xlsx"
    }
    if type not in file_map:
        raise HTTPException(status_code=400, detail="Invalid type")
    return FileResponse(file_map[type], media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

@app.get("/result/summary")
async def get_summary():
    # โหลด summary.json หรือ generate จาก rag_chain
    return {
        "total_questions": 120,
        "avg_score": 82.5,
        "evidence_found": 95,
        "gaps": 25
    }

@app.post("/upload/{type}")
async def upload_assessment_file(type: str, file: UploadFile = File(...)):
    """
    Upload assessment file by type (rubrics | qa | evidence)
    - rubrics: อัปโหลด Rubrics (ไฟล์เดียว)
    - qa: อัปโหลด Q&A (ไฟล์เดียว)
    - evidence: อัปโหลด Evidence (หลายไฟล์)
    """
    if type not in ["rubrics", "qa", "evidence"]:
        raise HTTPException(status_code=400, detail="Invalid type")

    # โฟลเดอร์แยกตามประเภท
    folder = os.path.join(ASSESSMENT_DIR, type)
    os.makedirs(folder, exist_ok=True)

    file_path = os.path.join(folder, file.filename)

    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot save file: {e}")

    return {
        "status": "ok",
        "type": type,
        "filename": file.filename,
        "file_type": os.path.splitext(file.filename)[1].lower(),
        "upload_date": datetime.utcnow().isoformat()
    }
