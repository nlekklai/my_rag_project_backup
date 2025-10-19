import os
import logging
from typing import List, Optional, Union, Sequence, Any
from concurrent.futures import ThreadPoolExecutor

# LangChain and Core Imports
from langchain.schema import Document as LcDocument, BaseRetriever
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever

# External Libraries
from pydantic import PrivateAttr, ConfigDict, BaseModel 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from flashrank import Ranker 

import chromadb
from chromadb.config import Settings 

# 🟢 FIX: ใช้ configure() โดยส่งพารามิเตอร์ของ Settings เข้าไปตรงๆ
# เพื่อเลี่ยง Pydantic validation error ของ Field 'settings'
try:
    chromadb.configure(anonymized_telemetry=False)
except AttributeError:
    # Fallback if chromadb.configure is not available
    chromadb.settings = Settings(anonymized_telemetry=False)

# --- CONFIGURATION CONSTANTS (Optimized for Speed and Precision) ---
INITIAL_TOP_K = 15  # จำนวนเอกสารที่ดึงมาค้นหาเบื้องต้น (Recall: ลดจาก 30 เพื่อเพิ่มความเร็ว)
FINAL_K_RERANKED = 7  # จำนวนเอกสารสุดท้ายหลังผ่านการ Rerank (Precision: คมชัด)

VECTORSTORE_DIR = "vectorstore"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 🚨 GLOBAL CACHE & PATHS:
CACHED_RANKER = None
CUSTOM_CACHE_DIR = os.path.expanduser("~/.hf_cache_dir/flashrank_models")


# -------------------- Custom Pydantic Model for Flashrank Input --------------------
class FlashrankRequest(BaseModel):
    """Mimics flashrank.Data.RankRequest to provide object attributes (query, passages, top_n)."""
    query: str
    passages: list[dict[str, Any]]
    top_n: int


# -------------------- Custom Compressor (ใช้ Flashrank) --------------------
class CustomFlashrankCompressor(BaseDocumentCompressor):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    ranker: Ranker
    top_n: int = FINAL_K_RERANKED # ใช้ค่าคงที่

    def compress_documents(
        self, 
        documents: Sequence[LcDocument], 
        query: str,
        **kwargs 
    ) -> Sequence[LcDocument]:
        
        if not documents:
            return []

        # 1. จัดรูปแบบเอกสารให้ Flashrank
        doc_list_for_rerank = [
            {"id": i, "text": doc.page_content, "meta": doc.metadata}
            for i, doc in enumerate(documents)
        ]

        run_input = FlashrankRequest( 
            query=query,
            passages=doc_list_for_rerank,
            top_n=self.top_n
        )

        # 2. ทำการ Rerank
        ranked_results = self.ranker.rerank(run_input) 

        # 3. จัดรูปแบบผลลัพธ์กลับเป็น LangChain Document
        reranked_docs = []
        for result in ranked_results:
            original_doc = documents[result['id']]
            reranked_doc = LcDocument(
                page_content=original_doc.page_content,
                metadata={**original_doc.metadata, 'relevance_score': result['score']}
            )
            reranked_docs.append(reranked_doc)
            
        return reranked_docs


# -------------------- Preload/Cache Logic (แก้ไข Flashrank Argument) --------------------
def preload_flashrank_model(model_name: str = "ms-marco-MiniLM-L-12-v2"):
    """
    พยายามบังคับดาวน์โหลด/โหลด Ranker instance จาก Cache Path ถาวร
    """
    global CACHED_RANKER
    if CACHED_RANKER:
        return CACHED_RANKER
        
    try:
        print(f"📦 Attempting to preload/cache Flashrank model to: {CUSTOM_CACHE_DIR}")
        
        # 🔴 CRITICAL FIX: ลบ device="cpu" ออกเพื่อเลี่ยง Argument Error
        CACHED_RANKER = Ranker(model_name=model_name, cache_dir=CUSTOM_CACHE_DIR) 
        
        print(f"✅ Flashrank model '{model_name}' is loaded from cache at {CUSTOM_CACHE_DIR} on CPU.")
        return CACHED_RANKER
    except Exception as e:
        print(f"⚠️ Flashrank model preload/load failed. Error: {e}")
        return None


# =================================================================
# --- Utility Functions ---
# =================================================================
def get_hf_embeddings():
    # 🟢 FIX: ยืนยันการใช้ "cpu" เพื่อแก้ปัญหาหน่วยความจำ (Bypassing MPS)
    device = "cpu"
    print(f"⚡ Using device: {device} for embeddings (Bypassing memory error)")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )

def get_vectorstore_path(doc_type: Optional[str] = None):
    if doc_type:
        path = os.path.join(VECTORSTORE_DIR, doc_type)
        os.makedirs(path, exist_ok=True)
        return path
    return VECTORSTORE_DIR

def list_vectorstore_folders(base_path: str = VECTORSTORE_DIR, doc_type: Optional[str] = None) -> List[str]:
    target_path = os.path.join(base_path, doc_type) if doc_type else base_path
    if not os.path.exists(target_path):
        return []
    if doc_type:
        full_path = os.path.join(base_path, doc_type)
        if os.path.isdir(full_path):
            return [f for f in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, f))]
        return []
    return [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

def vectorstore_exists(doc_id: str, base_path: str = VECTORSTORE_DIR, doc_type: Optional[str] = None) -> bool:
    if doc_type:
        path = os.path.join(base_path, doc_type, doc_id)
    else:
        path = os.path.join(base_path, doc_id)
    if not os.path.isdir(path):
        return False
    if os.path.isfile(os.path.join(path, "chroma.sqlite3")):
        return True
    return False

def save_to_vectorstore(
    doc_id: str,
    text_chunks: List[str],
    metadata: Optional[dict] = None,
    doc_type: str = "document",
    base_path: str = VECTORSTORE_DIR
):
    docs = [
        LcDocument( 
            page_content=t,
            metadata={**(metadata or {}), "source": doc_id, "chunk_index": i + 1}
        )
        for i, t in enumerate(text_chunks)
    ]
    embeddings = get_hf_embeddings()
    doc_dir = os.path.join(base_path, doc_type, doc_id)
    os.makedirs(doc_dir, exist_ok=True)
    vectordb = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=doc_dir
    )
    print(f"📄 Saved {len(docs)} chunks for doc_id={doc_id} into {doc_dir}")
    return vectordb

# -------------------- Load Vectorstore with CACHED Ranker Logic --------------------
def load_vectorstore(
    doc_id: str,
    top_k: int = INITIAL_TOP_K, 
    final_k: int = FINAL_K_RERANKED, 
    doc_types: list[str] | str = "document",
    base_path: str = VECTORSTORE_DIR,
):
    """
    Load a vectorstore retriever พร้อม Contextual Compression (Reranker) และ Fallback Logic
    """
    if isinstance(doc_types, str):
        doc_types = [doc_types]

    embeddings = get_hf_embeddings()
    retriever = None

    # 🚨 NEW: พยายามโหลด Ranker ที่ถูก Cache ไว้
    ranker_instance = preload_flashrank_model()

    for dtype in doc_types:
        path = os.path.join(base_path, dtype, doc_id)
        if os.path.isdir(path) and vectorstore_exists(doc_id, base_path, dtype):
            
            # 1. สร้าง Base Retriever
            base_retriever = Chroma(
                persist_directory=path,
                embedding_function=embeddings
            ).as_retriever(search_kwargs={"k": top_k})

            if ranker_instance:
                # 2. ถ้า Ranker โหลดสำเร็จ ให้สร้าง Custom Compressor
                try:
                    compressor = CustomFlashrankCompressor(
                        ranker=ranker_instance, 
                        top_n=final_k
                    ) 
                    
                    retriever = ContextualCompressionRetriever(
                        base_compressor=compressor, 
                        base_retriever=base_retriever
                    )
                    print(f"✅ Loaded Reranking Retriever for doc_id={doc_id} (doc_type={dtype}) with k={top_k}->{final_k} (Using Custom Ranker)")
                    
                except Exception as e:
                    logger.error(f"⚠️ CustomFlashrankCompressor failed. Error: {e}")
                    logger.warning("⚙️ Falling back to Base Retriever (Similarity Search Only).")
                    retriever = base_retriever
                    print(f"ℹ️ Using Base Retriever only for doc_id={doc_id} with k={top_k}")
            else:
                logger.warning("⚙️ Flashrank model failed to load. Falling back to Base Retriever.")
                retriever = base_retriever
                print(f"ℹ️ Using Base Retriever only for doc_id={doc_id} with k={top_k}")

            break # ใช้ตัวแรกที่เจอ

    if retriever is None:
        raise ValueError(f"❌ Vectorstore for doc_id '{doc_id}' not found in any of {doc_types}")

    return retriever


# =================================================================
# --- MultiDoc Retriever ---
# =================================================================
class NamedRetriever:
    """Wrapper around a retriever to store doc_id and doc_type"""
    def __init__(self, retriever: BaseRetriever, doc_id: str, doc_type: str):
        self.retriever = retriever
        self.doc_id = doc_id
        self.doc_type = doc_type

    def get_relevant_documents(self, query: str, **kwargs):
        # ใช้ invoke() ที่ถูกกำหนดใน NamedRetriever
        return self.retriever.invoke(query, **kwargs)

class MultiDocRetriever(BaseRetriever):
    """Combine multiple NamedRetrievers into one, deduplicating results"""
    _retrievers_list: list  = PrivateAttr()
    _k_per_doc: int = PrivateAttr()

    def __init__(self, retrievers_list: list, k_per_doc: int = INITIAL_TOP_K):
        super().__init__()
        self._retrievers_list = retrievers_list
        self._k_per_doc = k_per_doc

    @property
    def retrievers_list(self):
        return self._retrievers_list

    def _get_relevant_documents(self, query: str, *, run_manager=None):
        docs = []    
        
        def retrieve(named_r):
            # แต่ละ retriever (ซึ่งเป็น ContextualCompressionRetriever) จะทำการ Rerank ภายในตัวมันเองแล้ว
            return named_r.retriever.invoke(query) 

        # ใช้ Threading เพื่อดึงเอกสารจากทุกแหล่งพร้อมกัน
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(retrieve, self._retrievers_list))

        seen = set()
        unique_docs = []
        
        # รวมและล้างข้อมูลซ้ำซ้อน
        for dlist, named_r in zip(results, self._retrievers_list):
            if dlist is None:
                continue
            for d in dlist:
                # สร้าง key สำหรับ deduplication
                key = f"{d.metadata.get('source')}_{d.metadata.get('chunk_index')}_{named_r.doc_type}_{d.page_content[:50]}"
                if key not in seen:
                    seen.add(key)
                    # 🚨 สำคัญ: ตั้งค่า metadata keys ที่จำเป็นสำหรับ filter/audit
                    d.metadata["doc_type"] = named_r.doc_type
                    d.metadata["doc_id"] = named_r.doc_id 
                    d.metadata["doc_source"] = d.metadata.get("source")
                    unique_docs.append(d)

        print(f"📝 Query='{query[:50]}...' found {len(unique_docs)} unique docs across all retrieved lists.")
        # สำหรับ Debug: แสดงเฉพาะเอกสารที่มี score (Reranked)
        for d in unique_docs:
            if 'relevance_score' in d.metadata:
                 print(f" - [Reranked] Source={d.metadata.get('doc_source')}, Score={d.metadata.get('relevance_score'):.4f}, Type={d.metadata.get('doc_type')}, Content='{d.page_content[:80]}...'")
        
        # คืนเอกสารที่ไม่ซ้ำกันทั้งหมด (ซึ่งถูก Rerank มาจากแหล่งกำเนิดแต่ละอันแล้ว)
        return unique_docs

# -------------------- Load multiple vectorstores (ใช้ค่าคงที่ใหม่) --------------------
def load_all_vectorstores(doc_ids: Optional[Union[List[str], str]] = None,
                          top_k: int =  INITIAL_TOP_K,
                          final_k: int = FINAL_K_RERANKED,
                          doc_type: Optional[Union[str, List[str]]] = None,
                          base_path: str = VECTORSTORE_DIR) -> MultiDocRetriever:
    
    if isinstance(doc_ids, str):
        doc_ids = [doc_ids]
    # 🟢 FIX: ตั้งค่า default เป็น "evidence" (ตามที่ต้องการ)
    if doc_type is None:
        doc_types = ["evidence"]
    elif isinstance(doc_type, str):
        doc_types = [doc_type]
    else:
        doc_types = doc_type

    all_retrievers = []

    for dt in doc_types:
        folders = list_vectorstore_folders(base_path=base_path, doc_type=dt)
        for folder in folders:
            if doc_ids and folder not in doc_ids:
                continue
            try:
                # load_vectorstore จะใช้ doc_types เป็น list[dt] เสมอ
                retriever = load_vectorstore(folder, top_k=top_k, final_k=final_k, doc_types=[dt], base_path=base_path) 
                all_retrievers.append(NamedRetriever(retriever, doc_id=folder, doc_type=dt))
            except ValueError:
                 continue
            except Exception as e:
                print(f"⚠️ Skipping folder '{folder}' ({dt}): {e}")

    if not all_retrievers:
        raise ValueError(f"No vectorstores found for doc_ids={doc_ids} in doc_types={doc_types}")

    return MultiDocRetriever(retrievers_list=all_retrievers, k_per_doc=top_k)


# =================================================================
# --- VectorStoreManager (สำหรับงานสร้าง/จัดการ) ---
# =================================================================
class VectorStoreManager:
    """Manager สำหรับสร้างและโหลด Chroma vectorstore"""
    _instance: Optional["VectorStoreManager"] = None
    _vectorstore: Optional[Chroma] = PrivateAttr(default=None)

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, persist_dir: str = "vectorstore/document"):
        self.persist_dir = persist_dir
        # 🟢 FIX: ยืนยันการใช้ "cpu" เพื่อแก้ปัญหาหน่วยความจำ
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                                model_kwargs={"device": "cpu"})

    def load_vectorstore(self):
        if not os.path.exists(self.persist_dir):
            logger.warning(f"Vectorstore path not found: {self.persist_dir}")
            return None
        try:
            self._vectorstore = Chroma(persist_directory=self.persist_dir,
                                       embedding_function=self.embeddings)
            logger.info(f"✅ Loaded vectorstore from {self.persist_dir}")
        except Exception as e:
            logger.error(f"Failed to load vectorstore: {e}")
            self._vectorstore = None
        return self._vectorstore

    def get_retriever(self, k: int = 5):
        if self._vectorstore is None:
            self.load_vectorstore()
        if self._vectorstore is None:
            logger.error("Vectorstore not available for retrieval.")
            return None
        return self._vectorstore.as_retriever(search_kwargs={"k": k})
