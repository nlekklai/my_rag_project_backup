import logging
import os
import json
import re
from typing import List, Dict, Any, Optional

# --- Core Imports ---
# ต้องเพิ่ม imports ที่ย้ายมาจาก rag_analysis_utils
from core.vectorstore import load_vectorstore, load_all_vectorstores, vectorstore_exists
from core.rag_analysis_utils import get_llm # get_llm ต้อง return LLM instance
from core.rag_prompts import QA_PROMPT, COMPARE_PROMPT, SEMANTIC_MAPPING_PROMPT, ASSESSMENT_PROMPT
from core.ingest import list_vectorstore_folders, DATA_DIR, VECTORSTORE_DIR 

# LangChain Imports
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document, BaseRetriever
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from starlette.concurrency import run_in_threadpool


# -----------------------------
# Logging: console + file
# -----------------------------
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger("workflow")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler("workflow.log", mode="a", encoding="utf-8")
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

logger.info("Workflow logger initialized")
# -----------------------------

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
assessment_results = []
# -----------------------------

# -----------------------------
# LLM instance
# -----------------------------
llm = get_llm()

# -----------------------------
# Helper Functions (ย้ายมาจาก rag_analysis_utils)
# -----------------------------
def _format_docs(docs: list[Document]) -> str:
    """
    Convert a list of Document objects into a single readable string.
    """
    formatted = []
    for doc in docs:
        content = doc.page_content
        if isinstance(content, list):
            content = " ".join(content)
        content = str(content).replace("\n", " ").strip()
        formatted.append(content)
    return "\n\n---\n\n".join(formatted)

def normalize_doc_id_for_api(doc_id: str, doc_type: Optional[str] = None) -> str:
    """Normalize doc_id to include type prefix for API endpoints."""
    return f"{doc_type}_{doc_id}" if doc_type else doc_id

def match_doc_ids_from_question(question: str) -> List[str]:
    """Match doc_ids from question string using simple heuristics."""
    all_docs = list_vectorstore_folders()
    question_lower = question.lower()
    matched = []
    for doc_id in all_docs:
        if doc_id.lower() in question_lower:
            matched.append(doc_id)
    return sorted(list(set(matched)))

# -----------------------------
# 3. Semantic Mapping Function (TF-IDF mock)
# -----------------------------
def semantic_search_and_map(question: str, chunks: List[Dict], threshold: float = 0.3) -> Dict:
    # ... (Logic ภายใน semantic_search_and_map เหมือนเดิม) ...
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

def create_rag_chain(
    retriever, 
    llm_instance=llm, 
    prompt_template=QA_PROMPT
):
    """
    สร้าง RAG chain จาก retriever object
    retriever: BaseRetriever หรือ MultiDocRetriever
    """
    if retriever is None:
        raise ValueError("retriever cannot be None")

    # ใช้ RetrievalQA ของ LangChain
    chain = RetrievalQA.from_chain_type(
        llm=llm_instance,
        chain_type="stuff",  # or "map_reduce" ตาม requirement
        retriever=retriever,
        return_source_documents=True
    )
    return chain

# -----------------------------
# Main Workflow Runner (Background-friendly)
# -----------------------------
def run_assessment_workflow(use_llm_mapping: bool = False, mock_mode: bool = True):
    # ... (Logic ภายใน run_assessment_workflow เหมือนเดิม) ...
    """
    รัน workflow การประเมิน 5 ขั้นตอน
    รองรับการรันจาก FastAPI BackgroundTasks
    """
    if workflow_status.get("isRunning"):
        logger.warning("Assessment workflow is already running.")
        return

    workflow_status["isRunning"] = True
    workflow_status["currentStep"] = 0
    assessment_results.clear()

    try:
        # Step 1: Load Rubrics & QA
        _update_step_status(1, "running", 20)
        logger.info(f"Step 1: {workflow_status['steps'][0]['name']} - running (20%)")

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
            return

        # Real workflow
        questions = [
            "องค์กรปฏิบัติตามมาตรฐาน SEAM อย่างครบถ้วนหรือไม่?",
            "มีขั้นตอนตรวจสอบความสอดคล้องกับ SEAM ในทุกหน่วยงานหรือไม่?",
            "พบช่องว่างในการปฏิบัติตาม SEAM guideline ที่ควรแก้ไขหรือไม่?"
        ]
        logger.info("Step 1: Rubrics and QAs loaded successfully.")

        # Step 2: Ingest Evidence
        _update_step_status(2, "running", 40)
        logger.info(f"Step 2: {workflow_status['steps'][1]['name']} - running (40%)")

        rubric_retriever = load_vectorstore("rubrics", base_path=os.path.join(VECTORSTORE_DIR, "rubrics")).as_retriever(search_kwargs={"k":5})
        evidence_retriever = load_vectorstore("evidence", base_path=os.path.join(VECTORSTORE_DIR, "evidence")).as_retriever(search_kwargs={"k":10})
        feedback_retriever = load_vectorstore("feedback", base_path=os.path.join(VECTORSTORE_DIR, "feedback")).as_retriever(search_kwargs={"k":5})

        # Example: Loading multiple seam docs - need explicit paths
        seam_retrievers = [
            load_vectorstore("seam", base_path=os.path.join(VECTORSTORE_DIR, "seam")).as_retriever(search_kwargs={"k":5}),
            # Assuming "seam2" and "seam2567" are doc_ids in the 'seam' doc_type
            # load_vectorstore("seam2", base_path=os.path.join(VECTORSTORE_DIR, "seam")).as_retriever(search_kwargs={"k":5}),
            # load_vectorstore("seam2567", base_path=os.path.join(VECTORSTORE_DIR, "seam")).as_retriever(search_kwargs={"k":5})
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
            # Logic for mapping (LLM or TF-IDF)
            if use_llm_mapping:
                # ... (LLM mapping logic) ...
                pass
            else:
                mapping = semantic_search_and_map(q, vectorstore_chunks)

            # normalize mapping
            # ... (Normalization logic) ...
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
            chain = LLMChain(llm=llm, prompt=ASSESSMENT_PROMPT)
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
class MultiDocRetriever(BaseRetriever):
    """
    Combine multiple retrievers into one interface.
    ไม่ต้องส่งค่าใด ๆ ให้ BaseRetriever
    """
    def __init__(self, retrievers_list: List[BaseRetriever]):
        self._retrievers = retrievers_list  # ใช้ private attribute ป้องกันชน field ของ BaseRetriever

    def get_relevant_documents(self, query: str) -> List[Document]:
        all_docs = []
        for retriever in self._retrievers:
            if hasattr(retriever, "get_relevant_documents"):
                all_docs.extend(retriever.get_relevant_documents(query))
        return all_docs

    def invoke(self, query: str):
        return self.get_relevant_documents(query)


# -----------------------------
# Safe MultiDocRetriever Loader
# -----------------------------
def load_multi_retriever(doc_ids_list: List[str], doc_types_list: List[str] = None) -> MultiDocRetriever | None:
    """
    Load multiple retrievers safely. Skip missing vectorstores.
    Returns MultiDocRetriever instance or None.
    """
    if doc_types_list is None:
        doc_types_list = ["document", "faq"]

    retrievers = []

    for doc_id in doc_ids_list:
        found = False
        for dtype in doc_types_list:
            base_path = os.path.join(VECTORSTORE_DIR, dtype)
            if not os.path.isdir(base_path):
                logging.warning(f"Skipped missing doc_type folder: {dtype}")
                continue

            if vectorstore_exists(doc_id, base_path=base_path):
                try:
                    retriever = load_vectorstore(doc_id, base_path=base_path).as_retriever(search_kwargs={"k":5})
                    retrievers.append(retriever)
                    found = True
                    logging.info(f"Loaded vectorstore: {base_path}/{doc_id}")
                    break
                except Exception as e:
                    logging.error(f"Failed to load vectorstore {doc_id}/{dtype}: {e}")
                    continue

        if not found:
            logging.warning(f"No vectorstore found for doc_id {doc_id} in any doc_type")

    if not retrievers:
        logging.error("No valid vectorstores found for the given IDs")
        return None

    return MultiDocRetriever(retrievers)



# -----------------------------
# Safe RAG Question Answering
# -----------------------------
async def answer_question_rag(question: str, doc_ids: List[str], doc_types: Optional[List[str]] = None) -> Dict:
    """
    Safely load retrievers and run RAG.
    Returns JSON-friendly dict: {"answer": str, "sources": List[str]}
    """
    if doc_types is None:
        doc_types = ["document", "faq"]

    multi_retriever = load_multi_retriever(doc_ids, doc_types)
    if multi_retriever is None:
        return {"answer": "Error: No valid vectorstores found.", "sources": []}

    try:
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,  # ใช้ LLM global instance
            chain_type="stuff",
            retriever=multi_retriever,
            return_source_documents=True
        )

        def run_rag_sync():
            return rag_chain({"query": question})

        result = await run_in_threadpool(run_rag_sync)
        answer_text = result.get("result") or result.get("answer") or ""

        sources = []
        for doc in result.get("source_documents", []):
            src = getattr(doc, "metadata", {}).get("source") or getattr(doc, "metadata", {}).get("doc_id")
            if src:
                sources.append(src)

        return {"answer": answer_text, "sources": list(set(sources))}

    except Exception as e:
        logging.error(f"RAG execution failed: {e}")
        return {"answer": f"RAG execution failed: {str(e)}", "sources": []}
