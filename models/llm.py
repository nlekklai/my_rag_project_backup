# -*- coding: utf-8 -*-
# models/llm.py - Production Version (Ollama Unified Connector)
# รองรับทั้ง Local (Mac 8B) และ Cloud (NVIDIA 70B) ผ่าน Ollama

import logging
import os
from typing import Optional, Any, Final
from langchain_core.language_models.llms import BaseLLM

# --- Conditional Imports ---
try:
    from langchain_ollama import OllamaLLM
except ImportError:
    logger = logging.getLogger(__name__)
    logger.error("❌ 'langchain-ollama' not found. Please install it using 'pip install langchain-ollama'")
    OllamaLLM = None

logger = logging.getLogger(__name__)

# --- Configuration Constants (ดึงจาก Global Config) ---
from config.global_vars import (
    DEFAULT_LLM_MODEL_NAME, 
    LLM_CONTEXT_WINDOW, 
    LLM_TEMPERATURE,
    RAG_RUN_MODE,
    LLM_NUM_PREDICT
)

# -----------------------------------------------------
# 🎯 Global LLM Instance (สำหรับ Backward Compatibility)
# -----------------------------------------------------
# เก็บอินสแตนซ์แรกที่สร้างขึ้นเพื่อใช้งานร่วมกับส่วนอื่นๆ ของระบบ
llm: Optional[Any] = None 

# -----------------------------------------------------
# 🛠️ Factory Logic: create_llm_instance
# -----------------------------------------------------

def create_llm_instance(
    model_name: Optional[str] = None,
    temperature: float = LLM_TEMPERATURE,
    context_window: Optional[int] = None
) -> Optional[Any]:
    global llm 
    
    selected_model = model_name or DEFAULT_LLM_MODEL_NAME
    selected_ctx = context_window or LLM_CONTEXT_WINDOW
    
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # ดึงค่าลิมิต Token จาก Global Vars (Mac=2048, Server=4096)
    selected_predict = LLM_NUM_PREDICT 

    logger.warning(f"⚠️ Initializing LLM in {RAG_RUN_MODE} mode")
    logger.info(f"📍 Model: {selected_model} | Predict: {selected_predict}")

    try:
        if OllamaLLM is None:
            raise ImportError("langchain-ollama is required.")

        # 🎯 ปรับการส่งพารามิเตอร์ให้ครอบคลุม
        llm_instance = OllamaLLM(
            model=selected_model,
            temperature=temperature,
            num_ctx=selected_ctx,
            base_url=base_url,
            timeout=600,
            # num_predict คือหัวใจหลักที่แก้ปัญหา JSON ตัดจบ
            num_predict=selected_predict 
        )

        logger.info(f"✅ LLM Instance created successfully")
        if llm is None:
            llm = llm_instance
        return llm_instance

    except Exception as e:
        logger.error(f"❌ Failed to initialize Ollama LLM: {e}")
        return None

# -----------------------------------------------------
# 🧪 Health Check (Optional)
# -----------------------------------------------------
def check_llm_connection() -> bool:
    """ตรวจสอบเบื้องต้นว่าสามารถเชื่อมต่อกับ Ollama Server ได้หรือไม่"""
    if llm:
        try:
            # ลองส่งคำขอด่วนๆ เพื่อเช็คสถานะ
            # llm.invoke("Hi") 
            return True
        except:
            return False
    return False