# rag_chain.py
import os
import asyncio
import difflib
from datetime import datetime
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from vectorstore import save_to_vectorstore, load_vectorstore
from models.llm import get_llm
from ingest import process_document
from assessment_state import process_state
import pandas as pd
from ingest import load_txt
from file_loaders import FILE_LOADER_MAP

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

# ---- สร้าง RAG chain สำหรับ doc_id ใดก็ได้ ----
def create_rag_chain(doc_id: str):
    """
    สร้าง RetrievalQA chain สำหรับเอกสาร doc_id
    """
    retriever = load_vectorstore(doc_id)  # VectorStoreRetriever อยู่แล้ว

    chain = RetrievalQA.from_chain_type(
        llm=get_llm(),
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": QA_PROMPT, "document_variable_name": "context"}
    )
    return chain

# ---- ตอบคำถามจากเอกสาร ----
def answer_question(question: str, doc_id: str):
    """
    ตอบคำถามจากเอกสาร doc_id
    """
    chain = create_rag_chain(doc_id)
    return chain.run(question)

# ---- เปรียบเทียบเอกสาร 2 ฉบับ ----
def compare_documents(doc1: str, doc2: str, question: str):
    """
    เปรียบเทียบหัวข้อหลักของสองเอกสาร
    """
    chain1 = create_rag_chain(doc1)
    chain2 = create_rag_chain(doc2)

    # ดึงหัวข้อและเนื้อหาหลักจากเอกสาร
    content1 = chain1.run(question)
    content2 = chain1.run(question)

    # แยกเป็นบรรทัดเพื่อเปรียบเทียบง่าย
    lines1 = content1.splitlines()
    lines2 = content2.splitlines()

    # สร้าง diff
    diff = difflib.unified_diff(lines1, lines2, lineterm='', n=0)
    diff_text = '\n'.join(diff)

    return {
        "metrics": [
            {
                "metric": "หัวข้อหลักและเนื้อหาสำคัญ",
                "doc1": content1,
                "doc2": content2,
                "delta": diff_text,
                "remark": "ความแตกต่าง, การเพิ่มหรือลบ จะแสดงใน delta"
            }
        ]
    }


# ---- Assessment workflow ----
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

    # ---------------- Step 1: Ingestion & Metadata ----------------
    steps[0]["status"] = "running"
    await asyncio.sleep(1)  # mock
    steps[0]["status"] = "done"

    # ---------------- Step 2: Chunking & Indexing ----------------
    steps[1]["status"] = "running"
    for folder in [rubrics_folder, qa_folder, evidence_folder]:
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                doc_id = os.path.splitext(filename)[0]  # ใช้ชื่อไฟล์เป็น doc_id
                try:
                    # เช็คว่ามี vectorstore ของ doc_id แล้วหรือยัง
                    from vectorstore import vectorstore_exists
                    if vectorstore_exists(doc_id):
                        print(f"✅ {doc_id} already processed, skipping...")
                        continue

                    process_document(file_path, filename)
                except Exception as e:
                    print(f"❌ Cannot process {filename}: {e}")
    await asyncio.sleep(1)
    steps[1]["status"] = "done"

    # ---------------- Step 3: Evidence Mapping ----------------
    steps[2]["status"] = "running"
    evidence_map = []  # list of {question, matched_evidence}
    if os.path.exists(qa_folder) and os.path.exists(evidence_folder):
        for qfile in os.listdir(qa_folder):
            q_path = os.path.join(qa_folder, qfile)
            ext = os.path.splitext(qfile)[1].lower()
            loader_func = FILE_LOADER_MAP.get(ext)
            if not loader_func:
                print(f"❌ Unsupported QA file type: {qfile}")
                continue

            try:
                docs = loader_func(q_path)
                questions = [doc.page_content.strip() for doc in docs if doc.page_content.strip()]
                for q in questions:
                    # mock: match first evidence file
                    evidence_file = os.listdir(evidence_folder)[0] if os.listdir(evidence_folder) else ""
                    evidence_map.append({"question": q, "evidence": evidence_file})
            except Exception as e:
                print(f"❌ Failed to load {qfile}: {e}")
    await asyncio.sleep(1)
    steps[2]["status"] = "done"

    # ---------------- Step 4: RAG Answering ----------------
    steps[3]["status"] = "running"
    rag_answers = []
    if os.path.exists(qa_folder):
        for qfile in os.listdir(qa_folder):
            q_path = os.path.join(qa_folder, qfile)
            questions = []
            with open(q_path, "r", encoding="utf-8") as f:
                questions = [line.strip() for line in f if line.strip()]
            for q in questions:
                latest_evidence = os.listdir(evidence_folder)[0] if os.listdir(evidence_folder) else None
                if latest_evidence:
                    doc_id = os.path.splitext(latest_evidence)[0]

                    # ✅ ensure vectorstore exists
                    try:
                        vs = load_vectorstore(doc_id)
                    except ValueError:
                        file_path = os.path.join(evidence_folder, latest_evidence)
                        process_document(file_path, latest_evidence)
                        vs = load_vectorstore(doc_id)

                    answer = await asyncio.to_thread(answer_question, q, doc_id)
                    rag_answers.append({"question": q, "answer": answer})


    await asyncio.sleep(1)
    steps[3]["status"] = "done"

    # ---------------- Step 5: Scoring Engine ----------------
    steps[4]["status"] = "running"
    # สร้าง Excel results
    score_file = os.path.join(RESULTS_DIR, "score.xlsx")
    evidence_file = os.path.join(RESULTS_DIR, "evidence.xlsx")
    gap_file = os.path.join(RESULTS_DIR, "gap.xlsx")

    # Score: mock
    df_score = pd.DataFrame([{"question": r["question"], "score": 80} for r in rag_answers])
    df_score.to_excel(score_file, index=False)

    # Evidence map
    df_evidence = pd.DataFrame(evidence_map)
    df_evidence.to_excel(evidence_file, index=False)

    # Gap: questions with no answer
    df_gap = pd.DataFrame([{"question": r["question"]} for r in rag_answers if not r["answer"]])
    df_gap.to_excel(gap_file, index=False)

    await asyncio.sleep(1)
    steps[4]["status"] = "done"

    print("✅ Assessment workflow completed")