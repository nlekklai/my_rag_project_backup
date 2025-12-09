# models/llm.py
import logging
import os
from typing import Optional, Final
from langchain_core.language_models.llms import BaseLLM

# --- Conditional Imports for Specific Backend ---
# Local Ollama Backend
try:
    from langchain_ollama import OllamaLLM
except ImportError:
    OllamaLLM = None

# Cloud Backend (Placeholder/OpenAI)
try:
    import openai
except ImportError:
    openai = None

logger = logging.getLogger(__name__)

# --- CONFIGURATION CONSTANTS (ดึงจาก Global Config และ Environment) ---
from config.global_vars import (
    DEFAULT_LLM_MODEL_NAME, 
    LLM_CONTEXT_WINDOW, 
    LLM_TEMPERATURE,
    RAG_RUN_MODE # ใช้ค่าที่ถูกตั้งค่าใน global_vars.py
)

# การตั้งค่าสำหรับ CLOUD Mode (ถ้ามี)
CLOUD_LLM_MODEL: Final[str] = "gpt-4o" 
OPENAI_API_KEY: Final[str] = os.environ.get("OPENAI_API_KEY", "")


# -----------------------------------------------------
# 🎯 Global LLM Instance (สำหรับ Backward Compatibility)
# -----------------------------------------------------
llm: Optional[BaseLLM] = None 

# -----------------------------------------------------
# 🧩 Placeholder Class สำหรับ Cloud (จำลองการทำงาน)
# -----------------------------------------------------
class CloudLLMPlaceholder(BaseLLM):
    """Placeholder for the Cloud LLM connector (e.g., OpenAI)."""
    model_name: str = CLOUD_LLM_MODEL

    def _generate(self, prompts: list[str], stop: Optional[list[str]] = None, **kwargs) -> str:
        logger.warning(f"--- Simulating CLOUD call to {self.model_name} ---")
        # จำลอง JSON output ที่ Assessment Engine คาดหวัง
        return '{"summary": "Simulation result", "score": 5, "explanation": "This is a cloud simulation.", "evidence_map": []}' 

    @property
    def _llm_type(self) -> str:
        return "cloud-placeholder"
    
    def _call(self, prompt: str, stop: Optional[list[str]] = None, **kwargs) -> str:
        return self._generate([prompt], stop=stop, **kwargs)


# -----------------------------------------------------
# 🛠️ Factory Logic: สร้าง Instance ตามโหมด
# -----------------------------------------------------

def create_llm_instance(
    model_name: str = DEFAULT_LLM_MODEL_NAME,
    temperature: float = LLM_TEMPERATURE,
    context_window: int = LLM_CONTEXT_WINDOW
) -> Optional[BaseLLM]:
    """
    Initializes and returns the appropriate LLM instance based on RAG_RUN_MODE.
    """
    global llm 
    
    selected_model = CLOUD_LLM_MODEL if RAG_RUN_MODE == "CLOUD" else model_name
    
    logger.warning(f"⚠️ Initializing LLM in {RAG_RUN_MODE} mode with model: {selected_model}")

    try:
        llm_instance = None
        
        # --- 1. CLOUD MODE ---
        if RAG_RUN_MODE == "CLOUD":
            if not OPENAI_API_KEY and not os.environ.get('OPENAI_API_KEY'):
                logger.error("❌ CLOUD mode requires OPENAI_API_KEY to be set.")
                return None
            
            llm_instance = CloudLLMPlaceholder(
                model_name=selected_model,
                temperature=temperature
            )
        
        # --- 2. LOCAL_OLLAMA MODE ---
        elif RAG_RUN_MODE == "LOCAL_OLLAMA":
            if OllamaLLM is None:
                raise ImportError("langchain-ollama is required for LOCAL_OLLAMA mode.")

            llm_instance = OllamaLLM(
                model=selected_model,
                temperature=temperature,
                context_window=context_window
            )
        
        else:
            raise ValueError(f"Unknown RAG_RUN_MODE: {RAG_RUN_MODE}. Must be 'LOCAL_OLLAMA' or 'CLOUD'.")


        logger.info(f"✅ LLM Instance created successfully: {selected_model} (Mode: {RAG_RUN_MODE})")
        
        # 3. กำหนดค่าให้ตัวแปร Global
        if llm is None:
            llm = llm_instance
            logger.debug("Global 'llm' variable set for backward compatibility.")
            
        return llm_instance

    except Exception as e:
        logger.error(f"❌ Failed to initialize LLM in {RAG_RUN_MODE} mode: {e}")
        return None
    
