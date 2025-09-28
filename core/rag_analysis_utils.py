#core/rag_analysis_utils.py
import logging
import json # [ADDED] Import json for parsing structured output
from typing import List, Dict, Any
from core.ingest import list_vectorstore_folders
import re

# LangChain Imports
# IMPORTANT: These imports must match your environment (LangChain 0.1+)
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document 

# External/Local Module Imports (These need to exist in your project structure)
# NOTE: Using relative import (.) as specified in the provided code snippet
try:
    from models.llm import get_llm
    from .rag_prompts import QA_PROMPT, COMPARE_PROMPT 
    from .vectorstore import load_vectorstore, load_all_vectorstores 
    
    # [NEW HELPER FUNCTION]
    def _format_docs(docs: list[Document]) -> str:
        """Joins retrieved documents into a single context string."""
        return "\n\n---\n\n".join(doc.page_content for doc in docs)
    
except ImportError as e:
    # Fallback/Mock for environment where internal modules are not defined
    logging.warning(f"Missing internal module imports (e.g., rag_prompts/vectorstore/models.llm): {e}. Using mock components.")
    
    # Define Mock objects to allow the file to be runnable/loadable
    class MockLLM:
        def invoke(self, input): 
            # [UPDATED MOCK] Mock LLM to return structured JSON string for comparison
            if "TASK: Based only on the provided context, generate a structured JSON object" in input:
                 return {"text": json.dumps({
                     "metrics": [
                         {"metric": "Overall Similarity", "delta": "High", "remark": "Mock comparison based on RAG context A and B."},
                         {"metric": "Risk Assessment Changes", "delta": "Low", "remark": "Risk profile remained stable year-over-year."},
                     ]
                 })}
            return {"text": "Mock LLM Response"}
    
    def get_llm(): return MockLLM()
    
    # Define Mock PromptTemplates 
    QA_PROMPT = PromptTemplate(input_variables=["context", "question"], template="Context: {context}\nQuestion: {question}\nAnswer:")
    # Updated Mock Prompt to reflect how comparison prompt is formatted
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
        """Joins retrieved documents into a single context string (used by mock)."""
        return "\n\n---\n\n".join(doc.page_content for doc in docs)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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

# -------------------- RAG Answer Generation (Main Workflow Step 4) --------------------

def answer_question_rag(
    question: str, 
    doc_id: str = None, 
    context: str = None,
    prompt_template: PromptTemplate = QA_PROMPT
) -> str:
    """
    RAG Answering using a specific document or a pre-retrieved context.

    Note: The main 5-step workflow (in rag_chain.py) uses the 'context' branch 
    because Step 3 handles the multi-document retrieval and mapping.
    """
    llm = get_llm()

    if context:
        # Scenario 1: Context is already mapped and provided (used in the 5-step workflow)
        logging.info("Using pre-mapped context for RAG answering.")
        
        # Prepare the full prompt text using the context and question
        prompt_text = prompt_template.format(context=context, question=question)
        
        # Use a simple LLMChain for text generation
        llm_chain = LLMChain(llm=llm, prompt=PromptTemplate(input_variables=["query"], template="{query}"))
        try:
            # We pass the fully formatted prompt text to the LLMChain
            result = llm_chain.invoke({"query": prompt_text})
            # LangChain 0.1 returns a dictionary, and the output text is usually under "text"
            return result.get("text", "ไม่สามารถสร้างคำตอบจาก LLM ได้").strip()
        except Exception as e:
            logging.error(f"Error during LLM Inference for question '{question}': {e}")
            return "เกิดข้อผิดพลาดในการสรุปผลการประเมิน"
    
    elif doc_id:
        # Scenario 2: Standard RetrievalQA from a single vectorstore (if needed for API endpoints)
        logging.info(f"Using RetrievalQA for doc_id: {doc_id}")
        try:
            # load_vectorstore should return a Retriever or a Vectorstore object that can act as a Retriever
            vectorstore = load_vectorstore(doc_id)
            retriever = vectorstore.as_retriever() if hasattr(vectorstore, 'as_retriever') else vectorstore
            
            # Use RetrievalQA chain to handle retrieval and answering
            rag_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff",
                chain_type_kwargs={"prompt": prompt_template, "document_variable_name": "context"}
            )
            return rag_chain.run(question)
        except ValueError as e:
            logging.error(f"Error loading vectorstore for {doc_id}: {e}")
            return f"ข้อผิดพลาด: ไม่พบ Vectorstore สำหรับ {doc_id}"
        except Exception as e:
            logging.error(f"Unexpected error in RetrievalQA chain: {e}")
            return "เกิดข้อผิดพลาดในการเรียกใช้ RetrievalQA Chain"
    
    return "ไม่สามารถดำเนินการ RAG ได้: ขาด Context หรือ Doc ID"


# -------------------- Document Comparison --------------------

def compare_documents(doc_a_id: str, doc_b_id: str, query: str = None) -> Dict[str, Any]:
    """
    Compares two documents/vectorstores summaries by retrieving context from both 
    vectorstores based on the query before passing to LLM for comparison inference.
    (Updated to use RAG retrieval before inference and structured JSON output)
    """
    llm = get_llm()
    
    # 1. Define the Comparison Query
    comparison_query = query if query else "เปรียบเทียบข้อค้นพบหลัก การประเมินความเสี่ยง และแผนการดำเนินงานของเอกสารนี้"
    logging.info(f"Comparison Query: {comparison_query}")
    
    context_a = "ไม่พบ Context"
    context_b = "ไม่พบ Context"
    
    try:
        # 2. Retrieve Context from Doc A
        logging.info(f"Retrieving context from document A: {doc_a_id}")
        vectorstore_a = load_vectorstore(doc_a_id)
        retriever_a = vectorstore_a.as_retriever() if hasattr(vectorstore_a, 'as_retriever') else vectorstore_a
        docs_a = retriever_a.get_relevant_documents(comparison_query)
        context_a = _format_docs(docs_a)
        
        # 3. Retrieve Context from Doc B
        logging.info(f"Retrieving context from document B: {doc_b_id}")
        vectorstore_b = load_vectorstore(doc_b_id)
        retriever_b = vectorstore_b.as_retriever() if hasattr(vectorstore_b, 'as_retriever') else vectorstore_b
        docs_b = retriever_b.get_relevant_documents(comparison_query)
        context_b = _format_docs(docs_b)
        
    except ValueError as e:
        logging.error(f"Error loading vectorstore during comparison: {e}")
        return {"error": f"เกิดข้อผิดพลาด: ไม่พบ Vectorstore สำหรับเอกสารที่ต้องการเปรียบเทียบ. {e}"}
    except Exception as e:
        logging.error(f"Unexpected error during context retrieval for comparison: {e}")
        return {"error": "เกิดข้อผิดพลาดที่ไม่คาดคิดในการดึงข้อมูลเพื่อเปรียบเทียบ"}

    # 4. Format the comparison prompt with JSON schema instructions
    
    # Define the expected JSON structure for the LLM
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
        f"{COMPARE_PROMPT.format(doc_a=f'เอกสาร A ({doc_a_id}):\n{context_a}', doc_b=f'เอกสาร B ({doc_b_id}):\n{context_b}')}"
    )
    
    # 5. Use a simple LLMChain for the comparison generation
    llm_chain = LLMChain(llm=llm, prompt=PromptTemplate(input_variables=["query"], template="{query}"))
    
    try:
        result = llm_chain.invoke({"query": prompt_text})
        raw_text = result.get("text", "")
        
        # Attempt to parse the LLM's response as JSON
        try:
            # Clean up the output if the LLM wrapped it in a markdown code block
            if raw_text.strip().startswith("```json"):
                raw_text = raw_text.strip().replace("```json\n", "").replace("\n```", "")
            
            parsed_json = json.loads(raw_text)
            
            # Final structured return (matching the desired output structure's core data)
            return {
                "doc_a_id": doc_a_id,
                "doc_b_id": doc_b_id,
                "metrics": parsed_json.get("metrics", []),
                # Keep comparison_report for debugging/logging, now contains the raw JSON string
                "comparison_report": raw_text.strip()
            }
        except json.JSONDecodeError:
            logging.error(f"LLM response was not valid JSON: {raw_text[:100]}...")
            return {
                "doc_a_id": doc_a_id,
                "doc_b_id": doc_b_id,
                "metrics": [],
                "comparison_report": raw_text.strip() or "LLM failed to generate structured metrics."
            }
            
    except Exception as e:
        logging.error(f"Error during comparison LLM Inference: {e}")
        return {"error": "เกิดข้อผิดพลาดในการเปรียบเทียบเอกสาร"}
