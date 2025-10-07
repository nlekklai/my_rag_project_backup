# -------------------- core/rag_analysis_utils.py --------------------
import logging
import json
import os
import re
from typing import List, Dict, Any, Optional
from core.ingest import list_vectorstore_folders, DATA_DIR, VECTORSTORE_DIR

# LangChain Imports
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# -------------------- Logging Setup --------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -------------------- Mock Mode Fallback --------------------
IS_MOCKING_ACTIVE = False

try:
    from models.llm import get_llm
    from .rag_prompts import QA_PROMPT, COMPARE_PROMPT
    from .vectorstore import load_vectorstore, load_all_vectorstores
except ImportError as e:
    logging.warning(f"Missing internal module imports: {e}. Using Mock Mode.")
    IS_MOCKING_ACTIVE = True

    class MockLLM:
        def invoke(self, input):
            return {"text": "[Mock LLM Response]"}

    def get_llm(): return MockLLM()
    QA_PROMPT = PromptTemplate(input_variables=["context", "question"], template="Context: {context}\nQuestion: {question}\nAnswer:")
    COMPARE_PROMPT = PromptTemplate(input_variables=["doc_a", "doc_b"], template="Compare Doc A: {doc_a}\nDoc B: {doc_b}")

    class MockRetriever:
        def get_relevant_documents(self, query):
            return [Document(page_content=f"Mock content for query '{query}'")]

    class MockVectorStore:
        def as_retriever(self, **kwargs): return MockRetriever()
        def get_relevant_documents(self, query): return MockRetriever().get_relevant_documents(query)

    def load_vectorstore(doc_id): 
        logging.warning(f"Mocking vectorstore load for {doc_id}.")
        return MockVectorStore()

    def load_all_vectorstores(): pass

# -------------------- Helper Functions --------------------
def _format_docs(docs: list[Document]) -> str:
    """
    Convert a list of Document objects into a single readable string.
    Avoid splitting each character; replace newlines with space.
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

    is_comparison_query = any(keyword in question_lower for keyword in ["เทียบ", "แตกต่าง", "compare"])
    potential_doc_names = re.findall(r'(\w+-\w+|\w+_\w+|\d{4})', question_lower)

    explicit_matches = [
        doc_id for doc_id in all_docs
        if any(p_name in doc_id.lower() for p_name in potential_doc_names)
    ]
    if explicit_matches:
        matched = explicit_matches
    elif is_comparison_query:
        matched = all_docs
    else:
        matched = all_docs

    # Year-specific comparison
    if is_comparison_query and any(y in question_lower for y in ["2566", "2567"]):
        matched = [d for d in all_docs if any(y in d for y in ["2566", "2567"])]

    if is_comparison_query and len(matched) < 2 and len(all_docs) >= 2:
        matched = all_docs

    return matched

def get_document_text(doc_id: str, type_folder: Optional[str] = None) -> str:
    """Read content from DATA_DIR / type_folder / doc_id.*"""
    folder = os.path.join(DATA_DIR, type_folder) if type_folder else DATA_DIR
    if not os.path.exists(folder): return ""
    for f in os.listdir(folder):
        if os.path.splitext(f)[0] == doc_id:
            try:
                with open(os.path.join(folder, f), "r", encoding="utf-8") as file:
                    return file.read()
            except Exception as e:
                logging.warning(f"Failed reading file {f}: {e}")
    return ""

# -------------------- RAG Answer Generation --------------------
MAX_CONTEXT_CHARS = 15000

def answer_question_rag(
    question: str,
    doc_ids: Optional[List[str]] = None,
    doc_types: Optional[List[str]] = None,
    context: Optional[str] = None,
    prompt_template: PromptTemplate = QA_PROMPT
) -> str:
    """Generate answer using RAG with multi-doc support."""
    if IS_MOCKING_ACTIVE:
        logging.warning(f"Mock RAG response for question: {question}")
        return f"[Mock Answer] Question: {question}, Docs: {doc_ids}"

    llm_instance = get_llm()

    if context:
        prompt_text = prompt_template.format(context=context, question=question)
        llm_chain = LLMChain(
            llm=llm_instance,
            prompt=PromptTemplate(input_variables=["query"], template="{query}")
        )
        result = llm_chain.invoke({"query": prompt_text})
        return result.get("text", str(result)).strip()

    if not doc_ids:
        raise ValueError("doc_ids or context must be provided")

    combined_docs = []
    for idx, doc_id in enumerate(doc_ids):
        dtype = doc_types[idx] if doc_types and idx < len(doc_types) else "document"
        try:
            vectorstore = load_vectorstore(doc_id, base_path=os.path.join(VECTORSTORE_DIR, dtype))
            retriever = getattr(vectorstore, "as_retriever", lambda: vectorstore)()
            docs = retriever.get_relevant_documents(question)
            combined_docs.extend(docs)
        except ValueError as e:
            logging.warning(f"Vectorstore not found for {doc_id}: {e}")
        except Exception as e:
            logging.error(f"Error loading vectorstore {doc_id}: {e}")

    if not combined_docs:
        return "No documents found for RAG search"

    context_text = _format_docs(combined_docs)[:MAX_CONTEXT_CHARS]
    prompt_text = prompt_template.format(context=context_text, question=question)

    llm_chain = LLMChain(
        llm=llm_instance,
        prompt=PromptTemplate(input_variables=["query"], template="{query}")
    )
    try:
        result = llm_chain.invoke({"query": prompt_text})
        if isinstance(result, dict):
            return result.get("text", "LLM failed to generate answer").strip()
        return str(result).strip()
    except Exception as e:
        logging.error(f"Error during LLM inference: {e}")
        return "Error generating RAG answer"

# -------------------- Document Comparison --------------------
# -------------------- core/rag_analysis_utils.py --------------------
def compare_documents(doc_a_id, doc_b_id, doc_type=None, query=None):
    """Compare two documents via 2-step LLM workflow."""

    llm = get_llm()
    
    # ---------------- Step 1: Summarize each doc ----------------
    def summarize(doc_id):
        vectorstore = load_vectorstore(doc_id)
        retriever = getattr(vectorstore, "as_retriever", lambda: vectorstore)()
        docs = retriever.get_relevant_documents(f"สรุปเนื้อหาหลักของ {doc_id}")
        context_text = _format_docs(docs)
        prompt_text = QA_PROMPT.format(context=context_text, question=f"สรุปเนื้อหาหลักของ {doc_id}")
        chain = LLMChain(llm=llm, prompt=QA_PROMPT)
        result = chain.invoke({"query": prompt_text})
        return result.get("text", "").strip() if isinstance(result, dict) else str(result).strip()

    summary_a, summary_b = summarize(doc_a_id), summarize(doc_b_id)

    # ---------------- Step 2: Compare summaries ----------------
    compare_chain = LLMChain(llm=llm, prompt=COMPARE_PROMPT)
    compare_prompt_text = COMPARE_PROMPT.format(
        context=f"Doc1 Summary:\n{summary_a}\n\nDoc2 Summary:\n{summary_b}",
        query=f"เปรียบเทียบเอกสาร {doc_a_id} กับ {doc_b_id} โดยสรุปความแตกต่างสำคัญเป็น bullet points",
        doc_names=f"{doc_a_id}, {doc_b_id}"
    )
    delta_result = compare_chain.invoke({"query": compare_prompt_text})
    delta_text = delta_result.get("text", "").strip() if isinstance(delta_result, dict) else str(delta_result).strip()

    # ---------------- Return ----------------
    return {
        "doc_a_id": doc_a_id,
        "doc_b_id": doc_b_id,
        "metrics": [{
            "metric": "สรุปและเปรียบเทียบเอกสาร",
            "doc1": summary_a,
            "doc2": summary_b,
            "delta": delta_text,
            "remark": "2-step LLM workflow applied"
        }]
    }

