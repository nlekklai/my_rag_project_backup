import os
import asyncio
import re
import pandas as pd
from datetime import datetime
import time
import random
from typing import List, Dict, Any

from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from vectorstore import save_to_vectorstore, load_vectorstore, vectorstore_exists
from models.llm import get_llm
from ingest import process_document, load_txt
from assessment_state import process_state
from file_loaders import FILE_LOADER_MAP
import json


# ------------------- GLOBAL SETUP -------------------

ASSESSMENT_DIR = "assessment_data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# โฟลเดอร์สำหรับเอกสาร 4 ประเภทหลัก (รวม Feedback แล้ว)
FOLDERS = {
    "rubrics": os.path.join(ASSESSMENT_DIR, "rubrics"), # เกณฑ์การให้คะแนน
    "qa": os.path.join(ASSESSMENT_DIR, "qa"),           # คำถามการประเมิน
    "feedback": os.path.join(ASSESSMENT_DIR, "feedback"), # Feedback Report (NEW)
    "evidence": os.path.join(ASSESSMENT_DIR, "evidence"), # Evidence Files
}
# ตรวจสอบและสร้างโฟลเดอร์สำหรับเอกสาร 4 ประเภทใหม่
for folder_name in FOLDERS:
    os.makedirs(FOLDERS[folder_name], exist_ok=True)

# ------------------- MOCK RAG/SCORING FUNCTIONS -------------------

def get_documents_from_file(filepath: str) -> List[str]:
    """ฟังก์ชันจำลองการโหลดเอกสารจากไฟล์ประเภทต่างๆ (สำหรับ Step 3)"""
    if filepath.endswith(".txt"):
        with open(filepath, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    # Mock content if file type is complex (e.g., PDF)
    return ["Mock Document Content: Question 1", "Mock Document Content: Question 2"]

def semantic_search_and_map(question: str) -> Dict[str, Any]:
    """
    ฟังก์ชันจำลอง Semantic Search สำหรับทำ Mapping โดยอัตโนมัติ
    (TODO: ต้องแทนที่ด้วย Multi-Document Retriever ที่ค้นหา Rubrics, Evidence, Feedback พร้อมกัน)
    """
    # 25% chance of being a gap (Relevance score < 0.7)
    is_gap = random.random() < 0.25 
    
    if is_gap:
        return {
            "evidence": "Not Found (ไม่พบหลักฐานที่ชัดเจนใน Evidence Files)",
            "rubric": "Not Matched (เกณฑ์การให้คะแนนไม่ครอบคลุม)",
            "relevance_score": random.uniform(0.1, 0.69),
            "suggested_action": "ต้องการเอกสารหลักฐานเพิ่มเติมเกี่ยวกับ 'การประเมินความเสี่ยง'",
            "enabler": "Enabler C"
        }
    else:
        # Mock Context string that includes information from all mapped documents
        mapped_context = (
            f"Evidence Found: A key paragraph from project_report.pdf states the organization follows standard XYZ. "
            f"Rubric Matched: Criteria 4.1 requires documented policy. "
            f"Feedback Note: The latest feedback mentions a high score in documentation quality. "
        )
        return {
            "evidence": mapped_context, # ส่ง context ทั้งหมดไปใน key 'evidence' เพื่อให้ Step 4 นำไปใช้
            "rubric": "Matched: Criteria: Must meet standard 4.1.",
            "relevance_score": random.uniform(0.7, 0.99),
            "suggested_action": "จุดแข็ง: ระบบบันทึกผลการดำเนินงานชัดเจน (อ้างอิงจาก Feedback)",
            "enabler": f"Enabler {random.choice(['A', 'B'])}"
        }

def answer_question_rag(question: str, context: str) -> str:
    """
    RAG Answering (REAL LLM LOGIC) - ใช้ LLM เพื่อสรุปผลการประเมินโดยอ้างอิงจาก Context
    """
    if "Not Found" in context:
        return "" # ข้าม LLM call ถ้าหลักฐานขาดหาย
    
    # 1. โหลด LLM (ใช้ get_llm ที่มีอยู่แล้ว)
    llm = get_llm()
    
    # 2. เตรียม Prompt โดยใส่ Context และ Question ที่ได้จากการ Mapping
    prompt_text = QA_PROMPT.format(context=context, question=question)
    
    # 3. สร้าง LLM Chain แบบง่ายเพื่อรัน Text Generation
    # (ใช้ PromptTemplate ที่มีแค่ input_variables=["query"] เพื่อส่ง prompt_text ทั้งก้อน)
    llm_chain = LLMChain(llm=llm, prompt=PromptTemplate(input_variables=["query"], template="{query}"))
    
    try:
        # 4. รัน LLM Inference จริง
        result = llm_chain.invoke({"query": prompt_text})
        return result.get("text", "ไม่สามารถสร้างคำตอบจาก LLM ได้").strip()
    except Exception as e:
        print(f"❌ Error during LLM Inference for question '{question}': {e}")
        return "เกิดข้อผิดพลาดในการสรุปผลการประเมิน"


# ------------------- PROMPTS & RAG Chain UTILITIES (ส่วนเดิม) -------------------

# ---- Prompt ภาษาไทย (QA ทั่วไป) ----
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
คุณคือผู้ช่วยวิเคราะห์เอกสารและสรุปเนื้อหา
- อ่าน context อย่างละเอียด
- ตอบคำถามโดยสรุปหัวข้อหลักและเนื้อหาสำคัญ
- **ไม่ว่าคำถามจะเป็นภาษาใดก็ตาม (เช่น ภาษาอังกฤษ) ให้ตอบเป็นภาษาไทยเท่านั้น**
- แยกหัวข้อด้วย bullet points หรือ numbered list

Context:
{context}

คำถาม:
{question}

คำตอบ:
"""
)

# ---- Prompt ภาษาไทย (สำหรับการเปรียบเทียบหลายเอกสาร) ----
COMPARE_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
คุณคือผู้ช่วยวิเคราะห์และเปรียบเทียบเอกสารหลายฉบับ
- Context ได้ถูกรวบรวมมาจากเอกสารหลายฉบับ (เช่น 2566-PEA และ 2567-PEA)
- **งานของคุณคือการเปรียบเทียบเนื้อหาที่เกี่ยวข้องกับคำถาม**
- **ให้เน้นสรุปเฉพาะความแตกต่างที่สำคัญ** หรือการเปลี่ยนแปลงระหว่างเอกสารที่ระบุในคำถาม
- หากไม่พบความแตกต่างที่ชัดเจน ให้ระบุว่าเนื้อหาส่วนใหญ่เหมือนกันและให้ยกตัวอย่างที่เหมือนกันมาประกอบ
- **ไม่ว่าคำถามจะเป็นภาษาใดก็ตาม (เช่น ภาษาอังกฤษ) ให้ตอบเป็นภาษาไทยเท่านั้น**

Context:
{context}

คำถาม: {question} (เช่น 'สรุปเทียบ 2566-PEA กับ 2567-PEA')

สรุปผลการเปรียบเทียบ (เน้นความแตกต่าง):
"""
)

def create_rag_chain(doc_id: str, prompt_template: PromptTemplate = QA_PROMPT):
    """สร้าง RetrievalQA chain สำหรับ doc_id"""
    retriever = load_vectorstore(doc_id)
    return RetrievalQA.from_chain_type(
        llm=get_llm(),
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template, "document_variable_name": "context"}
    )

def answer_question(question: str, doc_id: str, prompt_template: PromptTemplate = QA_PROMPT):
    """ตอบคำถามจากเอกสาร doc_id"""
    chain = create_rag_chain(doc_id, prompt_template)
    return chain.run(question)

def clean_diff_markers(text: str) -> str:
    """ลบ marker ของ diff/git เช่น --- +++ @@ - +"""
    return re.sub(r'^(---|\+\+\+|@@|\+|-).*$', '', text, flags=re.MULTILINE).strip()

def compare_documents(doc1: str, doc2: str, question: str = None):
    # ... (Logic for compare_documents remains the same, assuming it works with LangChain) ...
    # This function is usually not called directly from the assessment workflow.
    if question is None:
        question = (
            "สรุปความแตกต่างระหว่างเอกสารสองฉบับนี้ในลักษณะ qualitative "
            "โดยไม่ซ้ำเนื้อหาของ doc1/doc2 และให้สรุปเป็น bullet points สั้น ๆ"
        )

    combined_prompt = f"""
เอกสารที่ 1 (Summary):
{doc1}

เอกสารที่ 2 (Summary):
{doc2}

{question}

ผลลัพธ์ delta:
"""
    llm = get_llm()
    prompt = PromptTemplate(input_variables=["query"], template="{query}")
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    result_text = llm_chain.invoke({"query": combined_prompt}).get("text", "").strip()

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

# ---------------- Assessment Workflow (NEW, ADVANCED LOGIC) ----------------

async def run_assessment_workflow():
    """
    5-step Assessment Workflow: A complete pipeline for multi-document RAG assessment
    with Semantic Mapping, Gap Analysis, and 3-part Report Generation.
    """
    steps = process_state["steps"]
    process_state["isRunning"] = True
    print("\n--- Starting Full Assessment Workflow (SEAM Assessment Logic) ---")

    # Helper function to simulate granular progress
    async def simulate_progress(step_index, duration=2.0):
        start_time = time.time()
        while time.time() - start_time < duration:
            elapsed = time.time() - start_time
            progress = int((elapsed / duration) * 100)
            steps[step_index]["progress"] = progress
            await asyncio.sleep(0.05) 
        steps[step_index]["progress"] = 100

    # ---- Step 1: Ingestion & Metadata ----
    steps[0]["status"] = "running"
    print("➡️ Step 1: Ingestion & Metadata...")
    await simulate_progress(0, duration=0.8)
    steps[0]["status"] = "done"

    # ---- Step 2: Chunking & Indexing (4 types) ----
    steps[1]["status"] = "running"
    print("➡️ Step 2: Chunking & Indexing (Building Vector Stores)...")
    
    total_files = sum(len(os.listdir(folder)) for folder in FOLDERS.values())
    files_processed = 0
    
    for folder_key, folder_path in FOLDERS.items():
        for filename in os.listdir(folder_path):
            filepath = os.path.join(folder_path, filename)
            # Simulate real indexing process
            # Note: process_document needs to be updated to handle different file types (if not already done)
            await asyncio.to_thread(process_document, filepath, filename)
            
            files_processed += 1
            steps[1]["progress"] = int((files_processed / max(1, total_files)) * 100)
            await asyncio.sleep(0.1)

    steps[1]["progress"] = 100
    steps[1]["status"] = "done"

    # ---- Step 3: Semantic Mapping (Q, R, F, E) ----
    steps[2]["status"] = "running"
    print("➡️ Step 3: Semantic Mapping (Matching Qs to Context)...")
    
    # 1. Load questions (ใช้ Folder "qa")
    qa_file = next(iter(os.listdir(FOLDERS["qa"])), None)
    questions = get_documents_from_file(os.path.join(FOLDERS["qa"], qa_file)) if qa_file else []

    # 2. Perform mapping
    assessment_results = []
    
    for i, q in enumerate(questions):
        # ใช้ Mock Function สำหรับ Semantic Mapping (เพื่อป้องกันการ crash จาก Multi-Doc Retrieval ที่ขาดไป)
        # TODO: เปลี่ยน semantic_search_and_map ให้ใช้ Multi-Document Retriever จริงๆ
        mapping_data = await asyncio.to_thread(semantic_search_and_map, q)
        
        # Store initial mapping result
        assessment_results.append({
            "question": q,
            # Note: mapped_evidence ตอนนี้เก็บ context string ที่จะใช้ใน Step 4
            "mapped_evidence": mapping_data["evidence"], 
            "mapped_rubric": mapping_data["rubric"],
            "relevance_score": mapping_data["relevance_score"],
            "suggested_action": mapping_data["suggested_action"],
            "enabler": mapping_data["enabler"],
            "final_score": 0, # Placeholder
            "rag_answer": "", # Placeholder
            "is_gap": False  # Placeholder
        })
        
        steps[2]["progress"] = int(((i + 1) / max(1, len(questions))) * 100)
        await asyncio.sleep(0.1)
        
    steps[2]["status"] = "done"

    # ---- Step 4: RAG Answering & Gap Analysis (REAL LLM INFERENCE) ----
    steps[3]["status"] = "running"
    print("➡️ Step 4: RAG Answering & Final Scoring (REAL LLM CALL)...")
    
    for i, res in enumerate(assessment_results):
        
        # Determine RAG context (currently stored in mapped_evidence from the mock)
        context = res['mapped_evidence']
        
        # *** ใช้ Logic การเรียก LLM Inference จริง ***
        rag_answer = await asyncio.to_thread(answer_question_rag, res["question"], context)
        
        assessment_results[i]["rag_answer"] = rag_answer
        
        # Gap Analysis based on relevance score or empty answer (if LLM returned nothing)
        if res["relevance_score"] < 0.7 or not rag_answer or "ไม่สามารถสร้างคำตอบ" in rag_answer:
             assessment_results[i]["is_gap"] = True
             assessment_results[i]["final_score"] = 40 # คะแนนต่ำเมื่อมี Gap
        else:
             assessment_results[i]["is_gap"] = False
             # จำลองการให้คะแนนตามเกณฑ์ (สามารถแทนที่ด้วย LLM call เพื่อให้คะแนนจริงได้)
             assessment_results[i]["final_score"] = 85 + random.randint(0, 10) # คะแนนสูงเมื่อไม่มี Gap

        steps[3]["progress"] = int(((i + 1) / max(1, len(assessment_results))) * 100)
        await asyncio.sleep(0.2)
        
    steps[3]["status"] = "done"

    # ---- Step 5: Scoring Engine & Report Generation ----
    steps[4]["status"] = "running"
    print("➡️ Step 5: Reporting (Generating 3 Excel files)...")
    await simulate_progress(4, duration=1.0)
    
    df_full = pd.DataFrame(assessment_results)
    
    # 1. Score Summary Report
    df_score = df_full[["enabler", "question", "final_score", "rag_answer", "suggested_action"]]
    df_score.rename(columns={"rag_answer": "Summary/Conclusion", "suggested_action": "General Recommendation"}, inplace=True)
    df_score.to_excel(os.path.join(RESULTS_DIR, "score_summary.xlsx"), index=False)
    
    # 2. Evidence Mapping Report
    # Note: mapped_evidence column currently holds the combined context string.
    df_evidence = df_full[["question", "mapped_evidence", "mapped_rubric", "relevance_score"]]
    df_evidence.to_excel(os.path.join(RESULTS_DIR, "evidence_map.xlsx"), index=False)
    
    # 3. Gap Analysis Report
    df_gap = df_full[df_full["is_gap"]].copy() 
    df_gap = df_gap[["enabler", "question", "mapped_evidence", "suggested_action"]]
    df_gap.rename(columns={"mapped_evidence": "Missing Evidence Status", "suggested_action": "Recommendation for Additional Documents"}, inplace=True)
    df_gap["Status"] = "Evidence Insufficient (Gap Found)"
    df_gap.to_excel(os.path.join(RESULTS_DIR, "gap_analysis.xlsx"), index=False)
    
    steps[4]["status"] = "done"

    print("\n--- ✅ Assessment workflow completed. ---")
    process_state["isRunning"] = False
    
    return True