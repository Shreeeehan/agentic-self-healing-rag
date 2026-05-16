import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from core.ingestion import DocumentChunk

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "rag_documents")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K_RETRIEVAL", 5))

def get_embedding_model() -> HuggingFaceEmbeddings:
    print(f"[VectorStore] Loading embedding model: {EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

class VectorStore:
    def __init__(self):
        self.embedding_model = get_embedding_model()
        self.vectorstore: Optional[Chroma] = None
        self._initialize()

    def _initialize(self):
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        self.vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embedding_model,
            persist_directory=CHROMA_PERSIST_DIR
        )
        count = self.vectorstore._collection.count()
        print(f"[VectorStore] Connected to ChromaDB. Documents in store: {count}")

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        if not chunks:
            print("[VectorStore] No chunks to add.")
            return
        texts = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [f"{chunk.source}_chunk_{chunk.chunk_index}" for chunk in chunks]
        self.vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        print(f"[VectorStore] Added {len(chunks)} chunks to ChromaDB.")

    def search(self, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        if not query.strip():
            return []
        results = self.vectorstore.similarity_search_with_relevance_scores(
            query=query, k=top_k
        )
        formatted = []
        for doc, score in results:
            formatted.append({
                "text": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", 0),
                "chunk_index": doc.metadata.get("chunk_index", 0),
                "score": round(score, 4)
            })
        print(f"[VectorStore] Query: '{query[:60]}' → {len(formatted)} results")
        return formatted

    def search_with_filter(self, query: str, source_filter: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        results = self.vectorstore.similarity_search_with_relevance_scores(
            query=query, k=top_k, filter={"source": source_filter}
        )
        return [
            {
                "text": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", 0),
                "score": round(score, 4)
            }
            for doc, score in results
        ]

    def get_collection_stats(self) -> Dict[str, Any]:
        count = self.vectorstore._collection.count()
        return {
            "total_chunks": count,
            "collection_name": COLLECTION_NAME,
            "persist_dir": CHROMA_PERSIST_DIR
        }

    def delete_collection(self) -> None:
        self.vectorstore.delete_collection()
        print("[VectorStore] Collection deleted. Re-initializing...")
        self._initialize()

_vectorstore_instance: Optional[VectorStore] = None

def get_vectorstore() -> VectorStore:
    global _vectorstore_instance
    if _vectorstore_instance is None:
        _vectorstore_instance = VectorStore()
    return _vectorstore_instance