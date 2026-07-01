"""
Vector Store — manages embedding generation and vector database
operations for RAG pipeline using Pinecone and FAISS.
"""

from typing import List, Optional
from dataclasses import dataclass
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS, Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
import logging

logger = logging.getLogger(__name__)


@dataclass
class VectorStoreConfig:
    store_type: str = "faiss"
    pinecone_index: Optional[str] = None
    pinecone_env: Optional[str] = None
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5


class VectorStore:
    """
    Manages vector database operations for RAG pipelines.
    Supports both FAISS (local) and Pinecone (cloud) backends.
    """

    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.embeddings = OpenAIEmbeddings()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        self.store = None

    def build_from_documents(self, documents: List[str]):
        """Build vector store from raw document strings."""
        chunks = []
        for doc in documents:
            chunks.extend(self.splitter.split_text(doc))
        logger.info(f"Built {len(chunks)} chunks from {len(documents)} documents")

        if self.config.store_type == "faiss":
            self.store = FAISS.from_texts(chunks, self.embeddings)
            logger.info("FAISS vector store built successfully")

        elif self.config.store_type == "pinecone":
            pinecone.init(
                api_key="your-pinecone-api-key",
                environment=self.config.pinecone_env
            )
            self.store = Pinecone.from_texts(
                chunks,
                self.embeddings,
                index_name=self.config.pinecone_index
            )
            logger.info(f"Pinecone vector store built: {self.config.pinecone_index}")

    def similarity_search(self, query: str) -> List[str]:
        """Retrieve top-k most relevant chunks for a query."""
        if not self.store:
            raise ValueError("Vector store not initialized. Call build_from_documents first.")
        docs = self.store.similarity_search(query, k=self.config.top_k)
        return [doc.page_content for doc in docs]

    def save_local(self, path: str):
        """Save FAISS index to disk."""
        if self.config.store_type == "faiss" and self.store:
            self.store.save_local(path)
            logger.info(f"FAISS index saved to {path}")

    def load_local(self, path: str):
        """Load FAISS index from disk."""
        self.store = FAISS.load_local(path, self.embeddings)
        logger.info(f"FAISS index loaded from {path}")


if __name__ == "__main__":
    config = VectorStoreConfig(store_type="faiss", top_k=5)
    vs = VectorStore(config)
    docs = [
        "BGP routing protocol manages inter-domain routing across the internet.",
        "OSPF is a link-state routing protocol used within autonomous systems.",
        "SNMP provides a framework for network device management and monitoring.",
    ]
    vs.build_from_documents(docs)
    results = vs.similarity_search("network routing protocol")
    for r in results:
        print(r)
