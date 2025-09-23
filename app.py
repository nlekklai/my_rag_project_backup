from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from ingest import process_document, list_documents, delete_document
from rag_chain import compare_documents, answer_question
from models.llm import get_llm
from vectorstore import load_vectorstore
from langchain.chains import RetrievalQA
import os
from datetime import datetime
import re

llm = get_llm()
app = FastAPI()

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
        doc_id = latest.get("id")
        # ถ้ามี .pdf หรือ .docx ให้ตัดนามสกุลออก
        if doc_id and "." in doc_id:
            doc_id = os.path.splitext(doc_id)[0]
        return doc_id
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
    result = compare_documents(doc1, doc2)  # ไม่ส่ง question เข้าไป
    return {"result": result}



@app.post("/ask")
async def ask(question: str = Form(...)):
    result = await run_in_threadpool(answer_question, question)
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
