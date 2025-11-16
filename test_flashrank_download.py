# test_flashrank_download_minilm.py
import logging
from flashrank import Ranker 

logging.basicConfig(level=logging.INFO)

print("--- Start FlashRank Test (Trying MiniLM) ---")
try:
    reranker = Ranker(model_name="ms-marco-MiniLM-L-12")
    print("\n✅ Success: Flashrank Ranker model loaded successfully!")

except Exception as e:
    print(f"\n❌ ERROR: Failed to load MiniLM model: {e}")
    print("   (หากยัง Error แสดงว่าปัญหาอยู่ที่สิทธิ์/Path ของ ONNXRuntime)")

print("--- End FlashRank Test ---")