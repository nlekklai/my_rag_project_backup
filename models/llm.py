# # models/llm.py
# from langchain_ollama import OllamaLLM

# def get_llm():
#     """
#     คืนค่า Ollama LLM ที่ใช้ Qwen2.5:7b
#     """
#     return OllamaLLM(
#         # model="qwen3:latest",   # ชื่อโมเดลต้องตรงกับ `ollama list`
#         # model="qwen2.5:7b",   # ชื่อโมเดลต้องตรงกับ `ollama list`
#         # model="mistral:latest",  
#         # model="deepseek-llm:7b-chat",
#         # model="deepseek-llm:7b",
#         model="llama3.1:8b",
#         # model="phi3:latest",
#         temperature=0.3,
#         num_ctx=4096
#     )


# -----------------------------------------------------
# --- Global LLM Instance ---
# -----------------------------------------------------
# models/llm.py
from langchain_ollama import OllamaLLM # <<< ใช้คลาสและแพ็คเกจใหม่
import logging

logger = logging.getLogger(__name__)

try:
    # llm ถูกสร้างเป็น Instance โดยตรง
    llm = OllamaLLM( 
        model="llama3.1:8b",
        temperature=0.3,
        num_ctx=4096
    )
    logger.info(f"✅ LLM Instance created successfully: {llm.model}")

except Exception as e:
    logger.error(f"❌ Failed to initialize Ollama LLM: {e}")
    llm = None