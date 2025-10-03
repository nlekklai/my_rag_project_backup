# core/rag_analysis_utils.py
import logging
import json
from typing import List, Dict, Any, Optional
from core.ingest import list_vectorstore_folders
import re

# LangChain Imports
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document 

# External/Local Module Imports
IS_MOCKING_ACTIVE = False

# -------------------- Internal Module Imports --------------------
try:
    from models.llm import get_llm
    from .rag_prompts import QA_PROMPT, COMPARE_PROMPT 
    from .vectorstore import load_vectorstore, load_all_vectorstores 
    
    def _format_docs(docs: list[Document]) -> str:
        return "\n\n---\n\n".join(doc.page_content for doc in docs)
    
except ImportError as e:
    logging.warning(f"Missing internal module imports: {e}. Using mock components.")
    IS_MOCKING_ACTIVE = True
    
    class MockLLM:
        def invoke(self, input): 
            if "TASK: Based only on the provided context, generate a structured JSON object" in input:
                return {"text": json.dumps({
                    "metrics": [
                        {"metric": "Overall Similarity", "delta": "High", "remark": "Mock comparison based on RAG context A and B."},
                        {"metric": "Risk Assessment Changes", "delta": "Low", "remark": "Risk profile remained stable year-over-year."},
                    ]
                })}
            return {"text": "Mock LLM Response"}
    
    def get_llm(): return MockLLM()
    
    QA_PROMPT = PromptTemplate(input_variables=["context", "question"], template="Context: {context}\nQuestion: {question}\nAnswer:")
    COMPARE_PROMPT = PromptTemplate(input_variables=["doc_a", "doc_b"], template="Compare Doc A Context: {doc_a} and Doc B Context: {doc_b}")

    class MockRetriever:
        def get_relevant_documents(self, query):
            return [Document(page_content=f"Mock content for query '{query}' from Mock Doc.")]
        
    class MockVectorStore:
        def as_retriever(self, **kwargs):
            return MockRetriever()
        def get_relevant_documents(self, query):
            return MockRetriever().get_relevant_documents(query)
        
    def load_vectorstore(doc_id): 
        logging.warning(f"Mocking vectorstore load for {doc_id}.")
        return MockVectorStore()
        
    def load_all_vectorstores(): pass
    
    def _format_docs(docs: list[Document]) -> str:
        return "\n\n---\n\n".join(doc.page_content for doc in docs)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -------------------- Helper Functions --------------------
def normalize_doc_id_for_api(doc_id: str, doc_type: Optional[str] = None) -> str:
    """Normalize doc_id to include type prefix for API endpoints."""
    if doc_type:
        return f"{doc_type}_{doc_id}"
    return doc_id

def match_doc_ids_from_question(question: str) -> List[str]:
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

    if is_comparison_query and any(y in question_lower for y in ["2566", "2567"]):
        matched = [d for d in all_docs if any(y in d for y in ["2566", "2567"])]

    if not matched:
        matched = all_docs

    if is_comparison_query and len(matched) < 2 and len(all_docs) >= 2:
        return all_docs

    return matched

# -------------------- RAG Answer Generation --------------------
MAX_CONTEXT_CHARS = 15000

def answer_question_rag(
    question: str,
    doc_id: Optional[str] = None,
    doc_type: Optional[str] = None,
    context: Optional[str] = None,
    prompt_template: PromptTemplate = QA_PROMPT
) -> str:
    """
    Generate answer using RAG.
    - doc_type: optional, used for API endpoint mapping
    """
    if IS_MOCKING_ACTIVE:
        if doc_id:
             return (f"คำตอบจำลองสำหรับเอกสาร {doc_id}: ระบบทำงานในโหมดจำลอง (Mock Mode). "
                     f"คำถามคือ: '{question}'.")
    
    llm = get_llm()
    if context:
        logging.info("Using pre-mapped context for RAG answering.")
        prompt_text = prompt_template.format(context=context, question=question)
        llm_chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate(input_variables=["query"], template="{query}")
        )
        try:
            result = llm_chain.invoke({"query": prompt_text})
            if isinstance(result, dict):
                return result.get("text", "ไม่สามารถสร้างคำตอบจาก LLM ได้").strip()
            return str(result).strip()
        except Exception as e:
            logging.error(f"Error during LLM inference: {e}")
            return "เกิดข้อผิดพลาดในการสรุปผลการประเมิน"

    elif doc_id:
        final_doc_id = normalize_doc_id_for_api(doc_id, doc_type)
        logging.info(f"Using RetrievalQA for doc_id: {final_doc_id}")
        try:
            vectorstore = load_vectorstore(final_doc_id)
            retriever = getattr(vectorstore, "as_retriever", lambda: vectorstore)()
            rag_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff",
                chain_type_kwargs={"prompt": prompt_template, "document_variable_name": "context"}
            )
            result = rag_chain.invoke({"query": question})
            return result.get("result", "")
        except ValueError as e:
            logging.error(f"Error loading vectorstore for {final_doc_id}: {e}")
            return f"ข้อผิดพลาด: ไม่พบ Vectorstore สำหรับ {final_doc_id}"
        except Exception as e:
            logging.error(f"Unexpected error in RetrievalQA chain: {e}")
            return f"เกิดข้อผิดพลาดในการเรียกใช้ RetrievalQA Chain: ({e})"
    
    return "ไม่สามารถดำเนินการ RAG ได้: ขาด Context หรือ Doc ID"

# -------------------- Document Comparison --------------------
def compare_documents(doc_a_id: str, doc_b_id: str, doc_type: Optional[str] = None, query: str = None) -> Dict[str, Any]:
    """
    Compare two documents using RAG retrieval and generate structured JSON.
    - doc_type: optional, used to normalize doc_id for API endpoints
    """
    if IS_MOCKING_ACTIVE:
        return {
            "error": f"ระบบทำงานในโหมดจำลอง (Mock Mode) สำหรับคำถาม '{query}'",
            "doc_a_id": doc_a_id,
            "doc_b_id": doc_b_id,
            "metrics": []
        }
    
    llm = get_llm()
    comparison_query = query if query else "เปรียบเทียบข้อค้นพบหลัก การประเมินความเสี่ยง และแผนการดำเนินงานของเอกสารนี้"
    logging.info(f"Comparison Query: {comparison_query}")

    final_doc_a_id = normalize_doc_id_for_api(doc_a_id, doc_type)
    final_doc_b_id = normalize_doc_id_for_api(doc_b_id, doc_type)

    context_a, context_b = "ไม่พบ Context", "ไม่พบ Context"

    try:
        vectorstore_a = load_vectorstore(final_doc_a_id)
        retriever_a = vectorstore_a.as_retriever() if hasattr(vectorstore_a, 'as_retriever') else vectorstore_a
        docs_a = retriever_a.get_relevant_documents(comparison_query)
        context_a = _format_docs(docs_a)[:MAX_CONTEXT_CHARS]

        vectorstore_b = load_vectorstore(final_doc_b_id)
        retriever_b = vectorstore_b.as_retriever() if hasattr(vectorstore_b, 'as_retriever') else vectorstore_b
        docs_b = retriever_b.get_relevant_documents(comparison_query)
        context_b = _format_docs(docs_b)[:MAX_CONTEXT_CHARS]

    except ValueError as e:
        logging.error(f"Vectorstore loading error: {e}")
        return {"error": f"เกิดข้อผิดพลาด: ไม่พบ Vectorstore. {e}"}
    except Exception as e:
        logging.error(f"Unexpected error during context retrieval: {e}")
        return {"error": "เกิดข้อผิดพลาดที่ไม่คาดคิดในการดึงข้อมูลเพื่อเปรียบเทียบ"}

    JSON_SCHEMA_HINT = """
{
  "metrics": [
    {"metric": "string (e.g., Overall Similarity, Risk Assessment)", "delta": "string (e.g., High, Low, Neutral)", "remark": "string (Detailed finding based on RAG context)"}
  ]
}
"""
    prompt_text = (
        "TASK: Based only on the provided context, generate a structured JSON object "
        "following this schema:\n"
        f"{JSON_SCHEMA_HINT.strip()}\n"
        "The comparison should focus on the requested query, detailing key findings, "
        "risk assessment changes, and differences in action plans. Return ONLY the JSON object.\n\n"
        "CONTEXT:\n"
        f"{COMPARE_PROMPT.format(doc_a=f'เอกสาร A ({final_doc_a_id}):\n{context_a}', doc_b=f'เอกสาร B ({final_doc_b_id}):\n{context_b}')}"
    )

    llm_chain = LLMChain(llm=llm, prompt=PromptTemplate(input_variables=["query"], template="{query}"))
    try:
        result = llm_chain.invoke({"query": prompt_text})
        raw_text = result.get("text", "")

        if raw_text.strip().startswith("```json"):
            raw_text = raw_text.strip().replace("```json\n", "").replace("\n```", "")
        
        try:
            parsed_json = json.loads(raw_text)
            return {
                "doc_a_id": final_doc_a_id,
                "doc_b_id": final_doc_b_id,
                "metrics": parsed_json.get("metrics", []),
                "comparison_report": raw_text.strip()
            }
        except json.JSONDecodeError:
            logging.error(f"LLM response not valid JSON: {raw_text[:100]}...")
            return {
                "doc_a_id": final_doc_a_id,
                "doc_b_id": final_doc_b_id,
                "metrics": [],
                "comparison_report": raw_text.strip() or "LLM failed to generate structured metrics."
            }
    except Exception as e:
        logging.error(f"Error during comparison LLM Inference: {e}")
        return {"error": "เกิดข้อผิดพลาดในการเปรียบเทียบเอกสาร"}
