# llm_guardrails.py
import re
from typing import Dict

# -----------------------------
# --- Intent Detection ---
# -----------------------------
def detect_intent(question: str, doc_type: str = "document") -> Dict[str, bool]:
    """
    Detect user intent for LLM query.
    Intent types:
      - FAQ: general questions or FAQs
      - Synthesis: compare / summarize multiple sources
      - Evidence: refer to documents or provide evidence-based answer
    """
    question_lower = question.lower()
    intent = {"is_faq": False, "is_synthesis": False, "is_evidence": False}

    # 1️⃣ Base on doc_type
    if doc_type in ["faq", "qa"]:
        intent["is_faq"] = True
    elif doc_type in ["document", "rubrics", "feedback", "evidence"]:
        intent["is_evidence"] = True

    # 2️⃣ Override / refine based on question keywords
    synthesis_keywords = ["compare", "difference", "highlight", "สรุป", "เปรียบเทียบ"]
    faq_keywords = ["what", "who", "when", "how", "อะไร", "ใคร", "เมื่อไร", "อย่างไร", "faq"]
    evidence_keywords = ["document", "evidence", "เอกสาร", "หลักฐาน", "source", "reference"]

    if any(k in question_lower for k in synthesis_keywords):
        intent["is_synthesis"] = True

    if any(k in question_lower for k in faq_keywords):
        intent["is_faq"] = True

    if any(k in question_lower for k in evidence_keywords):
        intent["is_evidence"] = True

    # Synthesis dominates for compare questions
    if intent["is_synthesis"]:
        intent["is_faq"] = False

    return intent


# -----------------------------
# --- SEAM Query Augmentation ---
# -----------------------------
SEAM_CODE_PATTERN = r"\b(SEAM\s*\d{1,3})\b"

def augment_seam_query(question: str) -> str:
    """
    Add SEAM code references to question if found.
    For example: "How to handle risk? SEAM 101" -> add context to guide LLM
    """
    matches = re.findall(SEAM_CODE_PATTERN, question, flags=re.IGNORECASE)
    if matches:
        seam_context = "Please consider the following SEAM codes: " + ", ".join(matches)
        return f"{seam_context}\n\n{question}"
    return question


# -----------------------------
# --- Prompt Builder ---
# -----------------------------
def build_prompt(context: str, question: str, intent: Dict[str, bool]) -> str:
    """
    Build LLM prompt based on context, question, and detected intent
    """
    sections = []

    # 1️⃣ Instruction based on intent
    if intent.get("is_synthesis"):
        sections.append("You are asked to compare the provided documents and synthesize key differences and insights.")
    elif intent.get("is_faq"):
        sections.append("You are asked to answer a FAQ-style question concisely, using the context if needed.")
    elif intent.get("is_evidence"):
        sections.append("You are asked to answer based on the provided evidence and documents, citing sources where applicable.")

    # 2️⃣ Add context if available
    if context:
        sections.append(f"Context from documents:\n{context}")

    # 3️⃣ Add question
    sections.append(f"User question:\n{question}")

    # 4️⃣ Guidance for structured answer
    if intent.get("is_synthesis"):
        sections.append(
            "Provide a structured comparison if relevant. Highlight differences, similarities, and references to sources."
        )
    elif intent.get("is_evidence"):
        sections.append(
            "If applicable, include sources with filename and chunk reference for any information you use."
        )

    return "\n\n".join(sections)


# -----------------------------
# --- Example Usage ---
# -----------------------------
if __name__ == "__main__":
    q = "เปรียบเทียบเอกสาร SEAM 101 กับ SEAM 102 เรื่องการจัดสรรทรัพยากร"
    doc_type = "document"
    augmented_q = augment_seam_query(q)
    intent = detect_intent(augmented_q, doc_type)
    context = "Source 1: ...\nSource 2: ..."
    prompt = build_prompt(context, augmented_q, intent)
    print("Intent:", intent)
    print("Prompt:\n", prompt)
