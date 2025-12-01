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
# เราจะ import RAG_RUN_MODE, LLM_MODEL_NAME, ฯลฯ มาจาก global_vars
from config.global_vars import (
    LLM_MODEL_NAME, 
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
    model_name: str = LLM_MODEL_NAME,
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
    
    
# import logging
# from typing import Optional, Final
# from langchain_ollama import OllamaLLM
# from langchain_core.language_models.llms import BaseLLM

# logger = logging.getLogger(__name__)

# # --- CONFIGURATION CONSTANTS ---
# # LLM_MODEL: Final[str] = "mistral:latest"
# # LLM_MODEL: Final[str] = LLM_MODEL_NAME
# from config.global_vars import LLM_MODEL_NAME, LLM_CONTEXT_WINDOW, LLM_TEMPERATURE

# # -----------------------------------------------------
# # 🎯 Global LLM Instance (เพื่อให้ Legacy Code ยังคง Import ได้)
# # -----------------------------------------------------
# llm: Optional[BaseLLM] = None 

# def create_llm_instance(
#     model_name: str = LLM_MODEL_NAME,
#     temperature: float = LLM_TEMPERATURE,
#     context_window: int = LLM_CONTEXT_WINDOW
# ) -> Optional[BaseLLM]:
#     """
#     Initializes and returns a new Ollama LLM instance. 
#     It also sets the global 'llm' variable if it's currently None (for compatibility).
#     """
#     global llm # ⬅️ เข้าถึงตัวแปร Global
    
#     # 1. สร้าง Instance ใหม่
#     try:
#         logger.warning(f"⚠️ Initializing LLM: model={model_name}, temperature={temperature}")
#         llm_instance = OllamaLLM(
#             model=model_name,
#             temperature=temperature,
#             context_window=context_window
#         )
#         logger.info(f"✅ LLM Instance created successfully: {model_name} (Temp: {temperature})")
        
#         # 2. 🟢 รักษาสภาพแวดล้อม Global: กำหนดค่าให้ตัวแปร Global ถ้ายังว่างอยู่
#         if llm is None:
#             llm = llm_instance
#             logger.debug("Global 'llm' variable set for backward compatibility.")
            
#         return llm_instance

#     except Exception as e:
#         logger.error(f"❌ Failed to initialize Ollama LLM: {e}")
#         return None

# models/llm.py

# import logging
# from typing import Optional, Final
# import os
# from langchain_core.language_models.llms import BaseLLM

# # --- Conditional Imports for Specific Backend ---
# # เราจะนำเข้า Dependencies เฉพาะเมื่อจำเป็นเท่านั้น

# # สำหรับ LOCAL Mode (LlamaCpp)
# try:
#     from langchain_community.llms import LlamaCpp
# except ImportError:
#     LlamaCpp = None

# # สำหรับ CLOUD Mode (OpenAI/Placeholder)
# try:
#     import openai
# except ImportError:
#     openai = None


# logger = logging.getLogger(__name__)

# # --- CONFIGURATION CONSTANTS (ดึงจาก ENV หรือตั้งค่า Default) ---
# # ตัวสลับโหมดหลัก: 'LOCAL' หรือ 'CLOUD'
# RUN_MODE: Final[str] = os.environ.get("RAG_RUN_MODE", "LOCAL") 

# # การตั้งค่าสำหรับ LOCAL Mode
# LLM_MODEL: Final[str] = "llama3.1:8b" # โมเดลที่ต้องการรัน
# # 🛑 ต้องแทนที่ด้วย Path ไฟล์ GGUF ที่ถูกต้อง (4.9G) ที่คุณพบใน ~/.ollama/models/blobs/
# GGUF_FILE_PATH: Final[str] = "/Users/oddnaphat/.ollama/models/blobs/sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29" 
# N_GPU_LAYERS: Final[int] = -1 # สำหรับใช้ GPU/MPS

# # การตั้งค่าสำหรับ CLOUD Mode
# CLOUD_LLM_MODEL: Final[str] = "gpt-4o"  
# OPENAI_API_KEY: Final[str] = os.environ.get("OPENAI_API_KEY", "")

# LLM_TEMPERATURE: Final[float] = 0.0
# LLM_CONTEXT_WINDOW: Final[int] = 8192

# # -----------------------------------------------------
# # 🎯 Global LLM Instance (สำหรับ Backward Compatibility)
# # -----------------------------------------------------
# llm: Optional[BaseLLM] = None 

# # -----------------------------------------------------
# # 🧩 Placeholder Class สำหรับ Cloud (จำลองการทำงาน)
# # -----------------------------------------------------
# class CloudLLMPlaceholder(BaseLLM):
#     """
#     Placeholder class for the OpenAI/Cloud LLM connector. 
#     It mimics BaseLLM behavior but calls the OpenAI API internally.
#     """
#     model_name: str = CLOUD_LLM_MODEL

#     def _call(self, prompt: str, stop: Optional[list[str]] = None) -> str:
#         # NOTE: ใน Production, โค้ดส่วนนี้จะเรียกใช้ openai.chat.completions.create
#         logger.warning(f"--- Simulating CLOUD call to {self.model_name} ---")
#         return '{"summary": "Simulation result", "score": 5, "evidence_map": []}' # จำลอง JSON output

#     @property
#     def _llm_type(self) -> str:
#         return "cloud-placeholder"

# # -----------------------------------------------------
# # 🛠️ Factory Logic: สร้าง Instance ตามโหมด
# # -----------------------------------------------------

# def create_llm_instance(
#     model_name: str = LLM_MODEL,
#     temperature: float = LLM_TEMPERATURE,
#     context_window: int = LLM_CONTEXT_WINDOW
# ) -> Optional[BaseLLM]:
#     """
#     Initializes and returns the appropriate LLM instance based on RUN_MODE (Factory Pattern).
#     """
#     global llm 
    
#     selected_model = CLOUD_LLM_MODEL if RUN_MODE == "CLOUD" else model_name
    
#     logger.warning(f"⚠️ Initializing LLM in {RUN_MODE} mode with model: {selected_model}")

#     try:
#         llm_instance = None
        
#         # --- 1. CLOUD MODE (GPT-4o / Production Test) ---
#         if RUN_MODE == "CLOUD":
#             if not OPENAI_API_KEY and not os.environ.get('OPENAI_API_KEY'):
#                 logger.error("❌ CLOUD mode requires OPENAI_API_KEY to be set.")
#                 return None
            
#             llm_instance = CloudLLMPlaceholder(
#                 model_name=selected_model,
#                 temperature=temperature
#             )
        
#         # --- 2. LOCAL MODE (LlamaCpp / Local Dev) ---
#         elif RUN_MODE == "LOCAL":
#             if LlamaCpp is None:
#                 raise ImportError("llama-cpp-python is required for LOCAL mode (GGUF backend).")

#             logger.info(f"💾 Loading GGUF model from: {GGUF_FILE_PATH}")
#             llm_instance = LlamaCpp(
#                 model_path=GGUF_FILE_PATH,
#                 n_gpu_layers=N_GPU_LAYERS,
#                 n_ctx=context_window,
#                 temperature=temperature,
#                 n_threads=0,
#                 verbose=False
#             )
        
#         else:
#             raise ValueError(f"Unknown RUN_MODE: {RUN_MODE}. Must be 'LOCAL' or 'CLOUD'.")


#         logger.info(f"✅ LLM Instance created successfully: {selected_model} (Mode: {RUN_MODE})")
        
#         # 3. 🟢 กำหนดค่าให้ตัวแปร Global
#         if llm is None:
#             llm = llm_instance
#             logger.debug("Global 'llm' variable set for backward compatibility.")
            
#         return llm_instance

#     except Exception as e:
#         logger.error(f"❌ Failed to initialize LLM in {RUN_MODE} mode: {e}")
#         return None