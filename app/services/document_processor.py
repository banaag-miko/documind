from typing import List, Tuple
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter   
from langchain_core.documents.base import Document as LangChainDocument


class DocumentProcessor:
    """Handles document extraction and chunking"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document processor
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks to maintain context
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def extract_text(self, file_path: str, file_type: str) -> str:
        """
        Extract text from different file types
        
        Args:
            file_path: Path to the file
            file_type: Type of file (pdf, txt, md)
            
        Returns:
            Extracted text content
        """
        if file_type == 'pdf':
            return self._extract_from_pdf(file_path)
        elif file_type in ['txt', 'md']:
            return self._extract_from_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF with page markers"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n\n--- Page {page_num} ---\n\n{page_text}"
        except Exception as e:
            raise Exception(f"Error extracting PDF: {str(e)}")
        
        if not text.strip():
            raise ValueError("No text could be extracted from PDF")
        
        return text
    
    def _extract_from_text(self, file_path: str) -> str:
        """Extract text from plain text files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try with different encoding if UTF-8 fails
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()
    
    def chunk_text(self, text: str, metadata: dict = None) -> List[LangChainDocument]:
        """
        Split text into chunks for embedding
        
        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk
            
        Returns:
            List of LangChain Document objects
        """
        # Create documents with metadata
        chunks = self.text_splitter.create_documents(
            texts=[text],
            metadatas=[metadata] if metadata else [{}]
        )
        return chunks
    
    def process_document(self, file_path: str, file_type: str, doc_id: int) -> Tuple[List[LangChainDocument], int]:
        """Complete document processing pipeline
        
        Args:
            file_path: Path to document
            file_type: Type of document
            doc_id: Database ID of document
            
        Returns:
            Tuple of (chunks, chunk_count)
        """
        # Extract text
        text = self.extract_text(file_path, file_type)
        
        # Prepare metadata
        metadata = {
            'document_id': doc_id,
            'source': file_path,
            'file_type': file_type
        }
        
        # Chunk text
        chunks = self.chunk_text(text, metadata)
        
        # Add chunk-specific metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['total_chunks'] = len(chunks)
        
        return chunks, len(chunks)