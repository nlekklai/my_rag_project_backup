# models/llm.py
import logging
from typing import Optional

# ✅ ใช้เวอร์ชันใหม่ของ langchain-ollama
from langchain_ollama import OllamaLLM

logger = logging.getLogger(__name__)

# --- CONFIGURATION CONSTANTS ---
LLM_MODEL = "llama3.1:8b"  # สามารถเปลี่ยนเป็น "qwen3:latest" ได้
LLM_TEMPERATURE = 0.0       # ตั้งเป็น 0.0 เพื่อความเสถียร
LLM_CONTEXT_WINDOW = 4096   # เวอร์ชันใหม่ใช้ context_window แทน num_ctx

# -----------------------------------------------------
# --- Global LLM Instance ---
# -----------------------------------------------------
llm: Optional[OllamaLLM] = None

try:
    logger.warning(f"⚠️ Initializing LLM: model={LLM_MODEL}, temperature={LLM_TEMPERATURE}")

    # สร้าง LLM instance
    llm = OllamaLLM(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        context_window=LLM_CONTEXT_WINDOW
    )

    if llm:
        logger.info(f"✅ LLM Instance created successfully: {llm.model} (Temp: {llm.temperature})")
    else:
        logger.error("❌ OllamaLLM failed to initialize but no exception was raised.")

except Exception as e:
    logger.error(f"❌ Failed to initialize Ollama LLM: {e}")
    llm = None
