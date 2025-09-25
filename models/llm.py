# models/llm.py
from langchain_ollama import OllamaLLM

def get_llm():
    """
    คืนค่า Ollama LLM ที่ใช้ Qwen2.5:7b
    """
    return OllamaLLM(
        # model="qwen3:latest",   # ชื่อโมเดลต้องตรงกับ `ollama list`
        # model="qwen2.5:7b",   # ชื่อโมเดลต้องตรงกับ `ollama list`
        # model="mistral:latest",  
        # model="deepseek-llm:7b-chat",
        # model="deepseek-llm:7b",
        model="llama3.1:8b",
        # model="phi3:latest",
        temperature=0.3,
        num_ctx=4096
    )

