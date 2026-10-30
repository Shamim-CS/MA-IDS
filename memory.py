"""
Experience Library: FAISS vector store for storing and retrieving learned rules.

Author: Md Shamimul Islam
Project: MA-IDS: Multi-Agent RAG Framework for IoT Network Intrusion Detection with an Experience Library
"""
import os
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from logger import IDSLogger


class MemoryManager:
    """Manages the FAISS-based memory system for storing learned rules"""
    
    def __init__(
        self, 
        brain_path: str,
        embedding_model: str,
        embedding_device: str,
        logger: IDSLogger
    ):
        self.brain_path = brain_path
        self.logger = logger
        self.embeddings = self._initialize_embeddings(embedding_model, embedding_device)
        self.vectorstore = self._load_or_create_brain()
    
    def _initialize_embeddings(
        self, 
        model_name: str, 
        device: str
    ) -> HuggingFaceEmbeddings:
        """Initialize HuggingFace embeddings"""
        self.logger.info(f"Loading Local Embeddings ({model_name}) on {device.upper()}...")
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device}
        )
    
    def _load_or_create_brain(self) -> FAISS:
        """Load existing FAISS index or create new one"""
        if os.path.exists(self.brain_path):
            self.logger.info(f"Loading existing brain from disk: {self.brain_path}...")
            return FAISS.load_local(
                self.brain_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        else:
            self.logger.info("Creating new brain (Empty)...")
            dummy_doc = Document(
                page_content="init", 
                metadata={"strategy_signature": "Initial Memory"}
            )
            return FAISS.from_documents([dummy_doc], self.embeddings)
    
    def save(self):
        """Save the FAISS index to disk"""
        self.vectorstore.save_local(self.brain_path)
        self.logger.info("Brain saved to disk.")
    
    def retrieve_similar_rules(
        self, 
        query_str: str, 
        top_k: int = 1
    ) -> str:
        """
        Retrieve similar rules from memory
        
        Args:
            query_str: Query string to search for
            top_k: Number of top results to retrieve
            
        Returns:
            String containing retrieved rules
        """
        try:
            docs = self.vectorstore.similarity_search(query_str, k=top_k)
            valid_rules = [
                d.metadata['strategy_signature'] 
                for d in docs 
                if 'strategy_signature' in d.metadata
            ]
            
            if valid_rules and valid_rules[0] != "Initial Memory":
                self.logger.info(f"   Retrieved Rules: {valid_rules}")
            else:
                self.logger.info("   No specific past rules found (or only initial memory).")
            
            return "\n".join(valid_rules) if valid_rules else "No specific past rules found."
        
        except Exception as e:
            self.logger.error(f"Error retrieving rules: {e}")
            return "No specific past rules found."
    
    def add_rule(self, flow_features: str, rule_metadata: dict):
        """
        Add a new rule to memory
        
        Args:
            flow_features: JSON string of flow features
            rule_metadata: Dictionary containing strategy_signature
        """
        if not rule_metadata or "strategy_signature" not in rule_metadata:
            self.logger.warning("Invalid rule metadata, skipping addition to memory")
            return
        
        doc = Document(
            page_content=flow_features,
            metadata=rule_metadata
        )
        self.vectorstore.add_documents([doc])
        self.logger.info("   Brain updated with new rule.")
    
    def get_memory_stats(self) -> dict:
        """Get statistics about the memory store"""
        try:
            # FAISS doesn't have a direct count method, estimate from index
            return {
                "status": "active",
                "path": self.brain_path
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
