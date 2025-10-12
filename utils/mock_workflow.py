# mock_workflow.py
import logging
import json
import pandas as pd
from core.rag_chain import llm, semantic_search_and_map
from core.rag_prompts import QA_PROMPT, SEMANTIC_MAPPING_PROMPT
from core.vectorstore import load_vectorstore

# -----------------------------
# Logger
# -----------------------------
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mock_workflow")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)
logger.info("Workflow logger initialized")

# -----------------------------
# Mock Questions
# -----------------------------
mock_questions = [
    "องค์กรมีการบริหารความเสี่ยงอย่างครบถ้วนหรือไม่?",
    "องค์กรมีขั้นตอนตรวจสอบความสอดคล้องกับ SEAM หรือไม่?",
    "พบช่องว่างในกระบวนการ SEAM ที่ควรแก้ไขหรือไม่?",
    "องค์กรมีการติดตามและประเมินผลการดำเนินงานอย่างไร?",
    "มีการใช้ข้อมูลเพื่อปรับปรุงกระบวนการหรือไม่?",
    "องค์กรมีการบริหารความรู้ (KM) อย่างครบถ้วนหรือไม่?",
    "องค์กรมีการบริหารทรัพยากรและบุคลากรอย่างเหมาะสมหรือไม่?",
    "องค์กรมีการสื่อสารแนวทางการบริหารความเสี่ยงภายในหรือไม่?",
    "องค์กรมีการประเมินประสิทธิผลของทุกขั้นตอนหรือไม่?",
    "องค์กรมีการปรับปรุงกระบวนการตามผลการประเมินหรือไม่?"
]

# -----------------------------
# Helper: Normalize mapping
# -----------------------------
def normalize_mapping(mapping):
    """
    Ensure mapping dict has correct structure for RAG workflow
    """
    for key in ["mapped_evidence", "mapped_rubric", "mapped_feedback"]:
        value = mapping.get(key, [])
        if isinstance(value, str):
            mapping[key] = [{"text": value}]
        elif not isinstance(value, list):
            mapping[key] = []
        else:
            mapping[key] = [v if isinstance(v, dict) else {"text": str(v)} for v in value]

    if "relevance_score" not in mapping:
        mapping["relevance_score"] = 0.0
    if "suggested_action" not in mapping:
        mapping["suggested_action"] = ""

    return mapping

# -----------------------------
# Run Mock Assessment
# -----------------------------
def run_mock_assessment_with_real_evidence(top_k=15):
    results = []

    # Load retriever
    logger.info("⚡ Using device for embeddings (M3 acceleration)")
    retriever = load_vectorstore("seam2567")
    logger.info(f"✅ Loaded retriever for doc_id=seam2567 with top_k={top_k}")

    for idx, q in enumerate(mock_questions, 1):
        print(f"\n--- Processing Question {idx}/{len(mock_questions)} ---")
        print(f"Question: {q}")

        # -------------------------
        # Retrieve top_k documents
        # -------------------------
        try:
            retrieved_docs = retriever.invoke(q)
            docs = retrieved_docs[:top_k]
        except Exception as e:
            logger.error(f"Retriever failed: {e}")
            docs = []

        context_text = "\n".join([getattr(d, 'page_content', str(d)) for d in docs])

        # -------------------------
        # Semantic Mapping with LLM
        # -------------------------
        prompt_str = SEMANTIC_MAPPING_PROMPT.format(
            question=q,
            documents=context_text
        )

        try:
            mapping_output = llm.invoke(prompt_str)

            if isinstance(mapping_output, str):
                try:
                    mapping = json.loads(mapping_output)
                except json.JSONDecodeError:
                    logger.warning("JSON parse failed, fallback to TF-IDF mock")
                    mapping = semantic_search_and_map(q, [{"id": f"d{i}", "text": getattr(d, 'page_content', str(d)), "type": "evidence"} for i, d in enumerate(docs)])
            elif isinstance(mapping_output, dict):
                mapping = mapping_output
            else:
                mapping = semantic_search_and_map(q, [{"id": f"d{i}", "text": getattr(d, 'page_content', str(d)), "type": "evidence"} for i, d in enumerate(docs)])
        except Exception as e:
            logger.error(f"LLM semantic mapping failed: {e}")
            mapping = semantic_search_and_map(q, [{"id": f"d{i}", "text": getattr(d, 'page_content', str(d)), "type": "evidence"} for i, d in enumerate(docs)])

        # Normalize mapping
        mapping = normalize_mapping(mapping)

        # -------------------------
        # QA Summary with LLM
        # -------------------------
        context_summary = "\n".join([c.get('text', '') for c in mapping["mapped_evidence"] +
                                                           mapping["mapped_rubric"] +
                                                           mapping["mapped_feedback"]])

        summary_prompt = QA_PROMPT.format(
            context=context_summary,
            question=q
        )

        try:
            summary = llm.invoke(summary_prompt)
        except Exception as e:
            logger.error(f"LLM QA summary failed: {e}")
            summary = "ข้อมูลไม่เพียงพอ"

        # -------------------------
        # Show result in terminal
        # -------------------------
        print(f"Relevance Score: {mapping.get('relevance_score', 0.0):.2f}")
        print(f"Suggested Action: {mapping.get('suggested_action', '')}")
        print(f"Summary:\n{summary}\n")

        results.append({
            "question": q,
            "mapped_evidence": mapping["mapped_evidence"],
            "mapped_rubric": mapping["mapped_rubric"],
            "mapped_feedback": mapping["mapped_feedback"],
            "suggested_action": mapping.get("suggested_action", ""),
            "relevance_score": mapping.get("relevance_score", 0.0),
            "summary": summary
        })

    # -----------------------------
    # Export to Excel
    # -----------------------------
    df = pd.DataFrame([{
        "Question": r["question"],
        "Summary": r["summary"],
        "Relevance": r["relevance_score"],
        "SuggestedAction": r["suggested_action"],
        "MappedEvidence": "; ".join([e.get('text', '') for e in r["mapped_evidence"]])
    } for r in results])

    df.to_excel("mock_assessment_results.xlsx", index=False)
    logger.info("✅ Results exported to mock_assessment_results.xlsx")

    return results

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    run_mock_assessment_with_real_evidence()
