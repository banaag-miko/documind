from typing import List
from langchain_core.documents.base import Document
from models.document import collection


class ChromaService:
    """Service for ChromaDB operations"""
    
    def add_documents(self, doc_id: int, chunks: List[Document], filename: str):
        """Add document chunks to ChromaDB"""
        ids = [f"doc_{doc_id}_chunk_{i}" for i in range(len(chunks))]
        
        metadatas = [{
            "document_id": doc_id,
            "filename": filename,
            "chunk_index": i,
            **chunk.metadata
        } for i, chunk in enumerate(chunks)]
        
        documents = [chunk.page_content for chunk in chunks]
        
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        
        return len(chunks)