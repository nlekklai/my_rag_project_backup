# models/llm.py
import logging
from typing import Optional, Final
from langchain_ollama import OllamaLLM
from langchain_core.language_models.llms import BaseLLM

logger = logging.getLogger(__name__)

# --- CONFIGURATION CONSTANTS ---
LLM_MODEL: Final[str] = "llama3.1:8b"
LLM_TEMPERATURE: Final[float] = 0.0
LLM_CONTEXT_WINDOW: Final[int] = 4096

# -----------------------------------------------------
# 🎯 Global LLM Instance (เพื่อให้ Legacy Code ยังคง Import ได้)
# -----------------------------------------------------
llm: Optional[BaseLLM] = None 

def create_llm_instance(
    model_name: str = LLM_MODEL,
    temperature: float = LLM_TEMPERATURE,
    context_window: int = LLM_CONTEXT_WINDOW
) -> Optional[BaseLLM]:
    """
    Initializes and returns a new Ollama LLM instance. 
    It also sets the global 'llm' variable if it's currently None (for compatibility).
    """
    global llm # ⬅️ เข้าถึงตัวแปร Global
    
    # 1. สร้าง Instance ใหม่
    try:
        logger.warning(f"⚠️ Initializing LLM: model={model_name}, temperature={temperature}")
        llm_instance = OllamaLLM(
            model=model_name,
            temperature=temperature,
            context_window=context_window
        )
        logger.info(f"✅ LLM Instance created successfully: {model_name} (Temp: {temperature})")
        
        # 2. 🟢 รักษาสภาพแวดล้อม Global: กำหนดค่าให้ตัวแปร Global ถ้ายังว่างอยู่
        if llm is None:
            llm = llm_instance
            logger.debug("Global 'llm' variable set for backward compatibility.")
            
        return llm_instance

    except Exception as e:
        logger.error(f"❌ Failed to initialize Ollama LLM: {e}")
        return None

