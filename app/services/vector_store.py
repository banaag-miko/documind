import os
import shutil
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents.base import Document 
from app.config import Config


class VectorStoreService:
    """Manages vector embeddings and similarity search using FAISS"""
    
    def __init__(self, persist_directory: str = None):
        """
        Initialize the vector store service
        
        Args:
            persist_directory: Directory to store FAISS indexes
        """
        self.persist_directory = persist_directory or Config.VECTOR_STORE_PATH
        
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=Config.OPENAI_API_KEY,
            model="text-embedding-3-small"  # Cost-effective and fast
        )
        
        # Ensure directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
    
    def _get_collection_path(self, collection_name: str) -> str:
        """Get the file path for a collection"""
        return os.path.join(self.persist_directory, collection_name)
    
    def create_and_save_vectorstore(self, collection_name: str, 
                                    documents: List[Document]) -> FAISS:
        """
        Create a new FAISS vectorstore and save it
        
        Args:
            collection_name: Name for this collection
            documents: List of documents to embed
            
        Returns:
            FAISS vectorstore instance
        """
        # Create vectorstore from documents
        vectorstore = FAISS.from_documents(documents, self.embeddings)
        
        # Save to disk
        vectorstore.save_local(self._get_collection_path(collection_name))
        
        return vectorstore
    
    def add_documents(self, collection_name: str, documents: List[Document]) -> List[str]:
        """
        Add documents to vector store
        
        Args:
            collection_name: Name for the collection
            documents: Documents to add
            
        Returns:
            List of document IDs
        """
        self.create_and_save_vectorstore(collection_name, documents)
        return [f"{collection_name}_{i}" for i in range(len(documents))]
    
    def load_vectorstore(self, collection_name: str) -> Optional[FAISS]:
        """
        Load an existing vectorstore
        
        Args:
            collection_name: Name of collection to load
            
        Returns:
            FAISS vectorstore or None if not found
        """
        path = self._get_collection_path(collection_name)
        
        if not os.path.exists(path):
            return None
        
        try:
            vectorstore = FAISS.load_local(
                path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            return vectorstore
        except Exception as e:
            print(f"Error loading vectorstore: {e}")
            return None
    
    def similarity_search(self, collection_name: str, query: str, 
                         k: int = 4) -> List[Document]:
        """
        Perform similarity search
        
        Args:
            collection_name: Name of collection to search
            query: Search query
            k: Number of results to return
            
        Returns:
            List of most similar documents
        """
        vectorstore = self.load_vectorstore(collection_name)
        
        if vectorstore is None:
            return []
        
        # Get similar documents with scores
        docs_with_scores = vectorstore.similarity_search_with_score(query, k=k)
        
        # Add scores to metadata
        for doc, score in docs_with_scores:
            doc.metadata['similarity_score'] = float(score)
        
        return [doc for doc, _ in docs_with_scores]
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection
        
        Args:
            collection_name: Name of collection to delete
            
        Returns:
            True if successful, False otherwise
        """
        path = self._get_collection_path(collection_name)
        
        try:
            if os.path.exists(path):
                shutil.rmtree(path)
            return True
        except Exception as e:
            print(f"Error deleting collection: {e}")
            return False