#core/rag_chain.py
import logging
from typing import List, Dict, Any
import json

# --- จำเป็นต้อง import ---
from core.vectorstore import load_vectorstore
from core.rag_analysis_utils import get_llm  # get_llm ต้อง return LLM instance
from core.rag_prompts import QA_PROMPT, COMPARE_PROMPT, SEMANTIC_MAPPING_PROMPT
from langchain.chains import LLMChain

# -----------------------------
# Logging: console + file
# -----------------------------
# -----------------------------
# Logging Formatter
# -----------------------------
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# -----------------------------
# Logger
# -----------------------------
logger = logging.getLogger("workflow")
logger.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

# File handler
file_handler = logging.FileHandler("workflow.log", mode="a", encoding="utf-8")
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

# Example usage
logger.info("Workflow logger initialized")

# -----------------------------
# Workflow State
# -----------------------------
workflow_status = {
    "isRunning": False,
    "currentStep": 0,
    "steps": [
        {"id": 1, "name": "Load Rubrics & QA", "status": "waiting", "progress": 0},
        {"id": 2, "name": "Ingest Evidence", "status": "waiting", "progress": 0},
        {"id": 3, "name": "Retrieve Contexts (RAG)", "status": "waiting", "progress": 0},
        {"id": 4, "name": "Generate Assessment Summary (LLM)", "status": "waiting", "progress": 0},
        {"id": 5, "name": "Finalize Results", "status": "waiting", "progress": 0},
    ]
}

# Mock storage for final assessment results
assessment_results = []

# -----------------------------
# 3. Semantic Mapping Function (TF-IDF mock)
# -----------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def semantic_search_and_map(question: str, chunks: List[Dict], threshold: float = 0.3) -> Dict:
    """
    ทำ Semantic Mapping ระหว่างคำถามและ chunks
    ใช้ TF-IDF + cosine similarity mock แทน embeddings จริง
    """
    texts = [question] + [c['text'] for c in chunks]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    sim_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    mapped_evidence = []
    mapped_rubric = []
    mapped_feedback = []

    for i, score in enumerate(sim_scores):
        if score >= threshold:
            c = chunks[i]
            c_copy = c.copy()
            c_copy['relevance_score'] = float(score)
            if c['type'] == 'evidence':
                mapped_evidence.append(c_copy)
            elif c['type'] == 'rubric':
                mapped_rubric.append(c_copy)
            elif c['type'] == 'feedback':
                mapped_feedback.append(c_copy)

    relevance_score = max(sim_scores) if len(sim_scores) > 0 else 0.0
    suggested_action = ""
    if relevance_score < 0.5:
        suggested_action = "ควรเพิ่มเอกสารเพิ่มเติมเพื่อสนับสนุนคำถามนี้"

    return {
        "mapped_evidence": mapped_evidence,
        "mapped_rubric": mapped_rubric,
        "mapped_feedback": mapped_feedback,
        "suggested_action": suggested_action,
        "relevance_score": relevance_score
    }

# -----------------------------
# Utility Function
# -----------------------------
def _update_step_status(step_id: int, status: str, progress: int):
    """Internal helper to update the status and progress."""
    global workflow_status
    workflow_status["currentStep"] = step_id
    for step in workflow_status["steps"]:
        if step["id"] == step_id:
            step["status"] = status
            step["progress"] = progress
            break

# -----------------------------
# LLM instance
# -----------------------------
llm = get_llm()

def create_rag_chain(doc_id: str, prompt_template=QA_PROMPT) -> LLMChain:
    """
    สร้าง LLMChain สำหรับ RAG query กับเอกสารเดียว
    """
    retriever = load_vectorstore(doc_id).as_retriever()
    chain = LLMChain(llm=llm, prompt=prompt_template)
    return chain

# -----------------------------
# Main Workflow Runner (Mock-ready + Real-ready)
# Full run_assessment_workflow with logger
# -----------------------------
# core/rag_chain.py (ปรับ run_assessment_workflow)
from fastapi import BackgroundTasks

# -----------------------------
# Main Workflow Runner (Background-friendly)
# -----------------------------
def run_assessment_workflow(use_llm_mapping: bool = False, mock_mode: bool = True):
    """
    รัน workflow การประเมิน 5 ขั้นตอน
    รองรับการรันจาก FastAPI BackgroundTasks
    """
    # ป้องกัน workflow ซ้ำ
    if workflow_status.get("isRunning"):
        logger.warning("Assessment workflow is already running.")
        return

    workflow_status["isRunning"] = True
    workflow_status["currentStep"] = 0
    assessment_results.clear()  # reset previous results

    try:
        # ---------------------------
        # Step 1: Load Rubrics & QA
        # ---------------------------
        _update_step_status(1, "running", 20)
        logger.info(f"Step 1: {workflow_status['steps'][0]['name']} - running (20%)")

        # Mock mode
        if mock_mode:
            questions = [
                "องค์กรปฏิบัติตามมาตรฐาน SEAM mock หรือไม่?",
                "มีขั้นตอนตรวจสอบความสอดคล้องกับ SEAM mock หรือไม่?",
                "พบช่องว่าง mock ที่ควรแก้ไขหรือไม่?"
            ]
            for q in questions:
                assessment_results.append({
                    "question": q,
                    "context_used": "Sample context for testing purposes.",
                    "mapped_evidence": [{"id": "e1", "text": "Sample evidence", "relevance_score": 0.9}],
                    "mapped_rubric": [{"id": "r1", "text": "Sample rubric", "relevance_score": 0.9}],
                    "mapped_feedback": [{"id": "f1", "text": "Sample feedback", "relevance_score": 0.9}],
                    "suggested_action": "Sample suggested action",
                    "relevance_score": 0.9,
                    "summary": "Sample summary generated for mock mode"
                })

            _update_step_status(5, "done", 100)
            logger.info("Mock workflow completed successfully.")
            return  # ไม่ต้อง return result, API จะเรียก get_workflow_results()

        # ---------------------------
        # Real workflow
        # ---------------------------
        questions = [
            "องค์กรปฏิบัติตามมาตรฐาน SEAM อย่างครบถ้วนหรือไม่?",
            "มีขั้นตอนตรวจสอบความสอดคล้องกับ SEAM ในทุกหน่วยงานหรือไม่?",
            "พบช่องว่างในการปฏิบัติตาม SEAM guideline ที่ควรแก้ไขหรือไม่?"
        ]
        logger.info("Step 1: Rubrics and QAs loaded successfully.")

        # Step 2: Ingest Evidence
        _update_step_status(2, "running", 40)
        logger.info(f"Step 2: {workflow_status['steps'][1]['name']} - running (40%)")

        # โหลด retrievers (สมมติว่า load_vectorstore() return retriever object)
        rubric_retriever = load_vectorstore("rubrics").as_retriever(search_kwargs={"k":5})
        evidence_retriever = load_vectorstore("evidence").as_retriever(search_kwargs={"k":10})
        feedback_retriever = load_vectorstore("feedback").as_retriever(search_kwargs={"k":5})

        seam_retrievers = [
            load_vectorstore("seam").as_retriever(search_kwargs={"k":5}),
            load_vectorstore("seam2").as_retriever(search_kwargs={"k":5}),
            load_vectorstore("seam2567").as_retriever(search_kwargs={"k":5})
        ]

        multi_retriever = MultiDocRetriever([rubric_retriever, evidence_retriever, feedback_retriever, *seam_retrievers])

        vectorstore_chunks = []
        for q in questions:
            docs = multi_retriever.get_relevant_documents(q)
            for i, d in enumerate(docs):
                chunk_type = getattr(d, "metadata", {}).get("type", "evidence")
                vectorstore_chunks.append({
                    "id": f"{q[:10]}_cand_{i}",
                    "type": chunk_type,
                    "text": getattr(d, "page_content", str(d)),
                    "source": getattr(d, "metadata", {}).get("source", "")
                })
        logger.info(f"Step 2: Ingested {len(vectorstore_chunks)} chunks")

        # Step 3: Semantic Mapping
        _update_step_status(3, "running", 60)
        logger.info(f"Step 3: {workflow_status['steps'][2]['name']} - running (60%)")

        for q in questions:
            if use_llm_mapping:
                chain = LLMChain(llm=llm, prompt=SEMANTIC_MAPPING_PROMPT)
                mapping_json_str = chain.run(
                    question=q,
                    documents="\n".join([c['text'] for c in vectorstore_chunks])
                )
                try:
                    mapping = json.loads(mapping_json_str)
                except json.JSONDecodeError:
                    logger.warning(f"JSON parse error, fallback to TF-IDF for question: {q}")
                    mapping = semantic_search_and_map(q, vectorstore_chunks)
            else:
                mapping = semantic_search_and_map(q, vectorstore_chunks)

            # normalize mapping
            for key in ["mapped_evidence", "mapped_rubric", "mapped_feedback"]:
                items = mapping.get(key, [])
                if isinstance(items, str):
                    mapping[key] = [{"text": items}]
                elif not isinstance(items, list):
                    mapping[key] = []

            context_text = "\n".join([c.get('text', '') for c in mapping["mapped_evidence"] +
                                                           mapping["mapped_rubric"] +
                                                           mapping["mapped_feedback"]])

            assessment_results.append({
                "question": q,
                "context_used": context_text,
                "mapped_evidence": mapping["mapped_evidence"],
                "mapped_rubric": mapping["mapped_rubric"],
                "mapped_feedback": mapping["mapped_feedback"],
                "suggested_action": mapping.get("suggested_action", ""),
                "relevance_score": mapping.get("relevance_score", 0.0)
            })
        logger.info("Step 3: Semantic Mapping completed")

        # Step 4: LLM Summary
        _update_step_status(4, "running", 80)
        logger.info(f"Step 4: {workflow_status['steps'][3]['name']} - running (80%)")

        for r in assessment_results:
            chain = LLMChain(llm=llm, prompt=QA_PROMPT)
            r['summary'] = chain.run(context=r['context_used'], question=r['question'])
        logger.info("Step 4: LLM summarization completed")

        # Step 5: Finalize
        _update_step_status(5, "done", 100)
        logger.info(f"Step 5: {workflow_status['steps'][4]['name']} - done (100%)")
        logger.info("Assessment workflow completed successfully.")

    except Exception as e:
        logger.error(f"Assessment workflow failed: {e}")
        _update_step_status(workflow_status["currentStep"], "error", 0)
    finally:
        workflow_status["isRunning"] = False


# -----------------------------
# Status and Results Getters
# -----------------------------
def get_workflow_status() -> Dict[str, Any]:
    return workflow_status

def get_workflow_results() -> List[Dict[str, Any]]:
    return assessment_results

# -----------------------------
# MultiDocRetriever
# -----------------------------
class MultiDocRetriever:
    """
    Combine multiple retrievers into one interface.
    """
    def __init__(self, retrievers_list: List):
        self.retrievers = retrievers_list

    def get_relevant_documents(self, query: str):
        all_docs = []
        for retriever in self.retrievers:
            if hasattr(retriever, "get_relevant_documents"):
                all_docs.extend(retriever.get_relevant_documents(query))
        return all_docs
