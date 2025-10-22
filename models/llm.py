# models/llm.py
from langchain_ollama import OllamaLLM # <<< ใช้คลาสและแพ็คเกจใหม่
import logging

logger = logging.getLogger(__name__)

# --- CONFIGURATION CONSTANTS ---
LLM_MODEL = "llama3.1:8b"
# LLM_MODEL = "qwen3:latest"
LLM_TEMPERATURE = 0.0 # <<< แก้ไข: ต้องเป็น 0.0 เพื่อความเสถียรในการประเมิน
LLM_NUM_CTX = 4096

# -----------------------------------------------------
# --- Global LLM Instance ---
# -----------------------------------------------------
try:
    logger.warning(f"⚠️ Initializing LLM for EVALUATION: Setting temperature={LLM_TEMPERATURE} for stability.")
    
    # llm ถูกสร้างเป็น Instance โดยตรง
    llm = OllamaLLM( 
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        num_ctx=LLM_NUM_CTX
    )
    logger.info(f"✅ LLM Instance created successfully: {llm.model} (Temp: {llm.temperature})")

except Exception as e:
    logger.error(f"❌ Failed to initialize Ollama LLM: {e}")
    llm = None