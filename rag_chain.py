# rag_chain.py
import os
import asyncio
import re
import pandas as pd
from datetime import datetime
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from vectorstore import save_to_vectorstore, load_vectorstore, vectorstore_exists
from models.llm import get_llm
from ingest import process_document, load_txt
from assessment_state import process_state
from file_loaders import FILE_LOADER_MAP
import json
from typing import List, Dict

ASSESSMENT_DIR = "assessment_data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---- Prompt ภาษาไทย ----
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
คุณคือผู้ช่วยวิเคราะห์เอกสารและสรุปเนื้อหา
- อ่าน context อย่างละเอียด
- ตอบคำถามโดยสรุปหัวข้อหลักและเนื้อหาสำคัญ
- แยกหัวข้อด้วย bullet points หรือ numbered list

Context:
{context}

คำถาม:
{question}

คำตอบ:
"""
)

# ---------------- RAG Chain ----------------
def create_rag_chain(doc_id: str):
    """สร้าง RetrievalQA chain สำหรับ doc_id"""
    retriever = load_vectorstore(doc_id)
    return RetrievalQA.from_chain_type(
        llm=get_llm(),
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": QA_PROMPT, "document_variable_name": "context"}
    )

def answer_question(question: str, doc_id: str):
    """ตอบคำถามจากเอกสาร doc_id"""
    chain = create_rag_chain(doc_id)
    return chain.run(question)

# ---------------- Helper ----------------
def clean_diff_markers(text: str) -> str:
    """ลบ marker ของ diff/git เช่น --- +++ @@ - +"""
    return re.sub(r'^(---|\+\+\+|@@|\+|-).*$', '', text, flags=re.MULTILINE).strip()

def compare_documents(doc1: str, doc2: str, question: str = None):
    """
    เปรียบเทียบเอกสาร 2 ฉบับ → คืนค่า JSON schema
    - delta: สรุปความแตกต่าง qualitative แบบอ่านง่าย
    """
    if question is None:
        question = (
            "สรุปความแตกต่างระหว่างเอกสารสองฉบับนี้ในลักษณะ qualitative "
            "โดยไม่ซ้ำเนื้อหาของ doc1/doc2 และให้สรุปเป็น bullet points สั้น ๆ"
        )

    combined_prompt = f"""
เอกสารที่ 1:
{doc1}

เอกสารที่ 2:
{doc2}

{question}

ผลลัพธ์ delta:
"""

    llm = get_llm()
    prompt = PromptTemplate(input_variables=["query"], template="{query}")
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    result_text = llm_chain.invoke({"query": combined_prompt}).get("text", "").strip()

    # fallback ถ้า LLM ไม่ส่ง JSON → wrap ใน schema
    return {
        "metrics": [
            {
                "metric": "หัวข้อหลักและเนื้อหาสำคัญ",
                "doc1": doc1,
                "doc2": doc2,
                "delta": result_text or "LLM ไม่สามารถสรุป delta ได้",
                "remark": "delta แสดงเฉพาะความแตกต่างเชิง qualitative"
            }
        ]
    }


# ---------------- Assessment Workflow ----------------
async def run_assessment_workflow():
    """
    5-step Assessment Workflow
    1) Ingestion & Metadata
    2) Chunking & Indexing
    3) Evidence Mapping
    4) RAG Answering
    5) Scoring Engine
    """
    steps = process_state["steps"]
    rubrics_folder = os.path.join(ASSESSMENT_DIR, "rubrics")
    qa_folder = os.path.join(ASSESSMENT_DIR, "qa")
    evidence_folder = os.path.join(ASSESSMENT_DIR, "evidence")

    # ---- Step 1: Ingestion & Metadata ----
    steps[0]["status"] = "running"
    await asyncio.sleep(1)
    steps[0]["status"] = "done"

    # ---- Step 2: Chunking & Indexing ----
    steps[1]["status"] = "running"
    for folder in [rubrics_folder, qa_folder, evidence_folder]:
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                doc_id = os.path.splitext(filename)[0]
                if vectorstore_exists(doc_id):
                    print(f"✅ {doc_id} already processed, skipping...")
                    continue
                try:
                    process_document(os.path.join(folder, filename), filename)
                except Exception as e:
                    print(f"❌ Cannot process {filename}: {e}")
    await asyncio.sleep(1)
    steps[1]["status"] = "done"

    # ---- Step 3: Evidence Mapping ----
    steps[2]["status"] = "running"
    evidence_map = []
    if os.path.exists(qa_folder) and os.path.exists(evidence_folder):
        for qfile in os.listdir(qa_folder):
            loader_func = FILE_LOADER_MAP.get(os.path.splitext(qfile)[1].lower())
            if not loader_func:
                print(f"❌ Unsupported QA file type: {qfile}")
                continue
            try:
                questions = [doc.page_content.strip() for doc in loader_func(os.path.join(qa_folder, qfile)) if doc.page_content.strip()]
                for q in questions:
                    ev_file = os.listdir(evidence_folder)[0] if os.listdir(evidence_folder) else ""
                    evidence_map.append({"question": q, "evidence": ev_file})
            except Exception as e:
                print(f"❌ Failed to load {qfile}: {e}")
    await asyncio.sleep(1)
    steps[2]["status"] = "done"

    # ---- Step 4: RAG Answering ----
    steps[3]["status"] = "running"
    rag_answers = []
    if os.path.exists(qa_folder):
        for qfile in os.listdir(qa_folder):
            with open(os.path.join(qa_folder, qfile), "r", encoding="utf-8") as f:
                questions = [line.strip() for line in f if line.strip()]
            for q in questions:
                latest_evidence = os.listdir(evidence_folder)[0] if os.listdir(evidence_folder) else None
                if latest_evidence:
                    doc_id = os.path.splitext(latest_evidence)[0]
                    if not vectorstore_exists(doc_id):
                        process_document(os.path.join(evidence_folder, latest_evidence), latest_evidence)
                    answer = await asyncio.to_thread(answer_question, q, doc_id)
                    rag_answers.append({"question": q, "answer": answer})
    await asyncio.sleep(1)
    steps[3]["status"] = "done"

    # ---- Step 5: Scoring Engine ----
    steps[4]["status"] = "running"
    pd.DataFrame([{"question": r["question"], "score": 80} for r in rag_answers]).to_excel(os.path.join(RESULTS_DIR, "score.xlsx"), index=False)
    pd.DataFrame(evidence_map).to_excel(os.path.join(RESULTS_DIR, "evidence.xlsx"), index=False)
    pd.DataFrame([{"question": r["question"]} for r in rag_answers if not r["answer"]]).to_excel(os.path.join(RESULTS_DIR, "gap.xlsx"), index=False)
    await asyncio.sleep(1)
    steps[4]["status"] = "done"

    print("✅ Assessment workflow completed")
