import re
import time
from typing import Dict, List

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langsmith import traceable

from app.config import Config
from models.schemas import SourceChunk
from models.document import collection  # ChromaDB collection


class RAGService:
    """Retrieval-Augmented Generation service with multiple LLM providers"""
    
    def __init__(self):
        """Initialize RAG service with LLM providers"""
        
        # Initialize different LLM providers
        self.llms = {}
        
        # OpenAI
        if Config.OPENAI_API_KEY:
            self.llms['openai'] = ChatOpenAI(
                model="gpt-4o-2024-08-06",
                temperature=0.3,
                max_completion_tokens=512,
                openai_api_key=Config.OPENAI_API_KEY
            )
        
        # Custom prompt template for structured responses
        self.prompt_template = """You are an AI assistant analyzing a document. Use the following pieces of context to answer the question at the end.
        If you don't know the answer based on the context, just say that you don't know. Don't try to make up an answer.
        Provide a confidence level (high, medium, or low) based on how well the context supports your answer.
        
        Also suggest 2-3 relevant follow-up questions the user might want to ask.

        Context:
        {context}

        Question: {question}

        Please structure your response as follows:
        1. Answer: [Your detailed answer]
        2. Confidence: [high/medium/low]
        3. Follow-up Questions:
           - [Question 1]
           - [Question 2]
           - [Question 3]

        Answer:"""

        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
    
    
    @traceable(name="retrieve_documents")
    def retrieve_from_chromadb(self, question: str, doc_id: int = None, k: int = 4) -> List[Dict]:
        """
        Retrieve relevant documents from ChromaDB
        
        Args:
            question: Query text
            doc_id: Optional document ID to filter by
            k: Number of results to return
            
        Returns:
            List of retrieved documents with metadata
        """
        query_params = {
            "query_texts": [question],
            "n_results": k
        }
        
        if doc_id is not None:
            query_params["where"] = {"document_id": doc_id}
        
        query_results = collection.query(**query_params)
        
        # Format results
        retrieved_docs = []
        if query_results['documents'][0]:
            for doc_text, metadata, distance in zip(
                query_results['documents'][0],
                query_results['metadatas'][0],
                query_results['distances'][0]
            ):
                retrieved_docs.append({
                    'content': doc_text,
                    'metadata': metadata,
                    'distance': distance,
                    'similarity_score': 1 - distance
                })
        
        return retrieved_docs
    
    def format_docs(self, docs: List[Dict]) -> str:
        """Format retrieved documents into a single context string"""
        return "\n\n".join(doc['content'] for doc in docs)
    
    @traceable(name="query_document")
    def query_document(self, collection_name: str, question: str, 
                      llm_provider: str = 'openai') -> Dict:
        """
        Query a document using RAG with ChromaDB and LCEL
        
        Args:
            collection_name: Name of the vector store collection (e.g., "doc_1" or "document_chunks")
            question: User's question
            llm_provider: LLM provider to use (openai, anthropic, google)
            
        Returns:
            Dictionary containing answer, sources, and metadata
        """
        start_time = time.time()
        
        # Step 1: Check if LLM provider is available
        if llm_provider not in self.llms:
            raise ValueError(f"LLM provider '{llm_provider}' not configured. Check your API keys.")
        
        # Step 2: Extract document_id from collection_name
        doc_id = None
        if collection_name.startswith('doc_'):
            try:
                doc_id = int(collection_name.split('_')[1])
            except (IndexError, ValueError):
                pass
        
        # Step 3: Create retriever function that works with LCEL
        def retriever(query: str) -> List[Dict]:
            """Retriever function for LCEL chain"""
            return self.retrieve_from_chromadb(query, doc_id=doc_id, k=4)
        
        # Step 4: Get LLM
        llm = self.llms[llm_provider]
        
        # Step 5: Create RAG chain using LCEL
        rag_chain = (
            {
                "context": lambda x: self.format_docs(retriever(x)),
                "question": RunnablePassthrough()
            }
            | self.prompt
            | llm
            | StrOutputParser()
        )
        
        # Step 6: Execute chain and get source documents
        try:
            # Get the answer using the chain
            answer_text = rag_chain.invoke(question)
            
            # Get source documents separately for metadata
            source_documents = retriever(question)
            
        except Exception as e:
            raise Exception(f"Error querying LLM: {str(e)}")
        
        # Step 7: Check if we got results
        if not source_documents:
            return {
                'answer': "I couldn't find any relevant information in the document to answer your question.",
                'confidence': 'low',
                'follow_up_questions': [],
                'sources': [],
                'llm_provider': llm_provider,
                'response_time': time.time() - start_time
            }
        
        # Step 8: Parse response
        parsed_response = self._parse_llm_response(answer_text, source_documents)
        
        # Calculate response time
        response_time = time.time() - start_time
        
        return {
            'answer': parsed_response['answer'],
            'confidence': parsed_response['confidence'],
            'follow_up_questions': parsed_response['follow_up_questions'],
            'sources': parsed_response['sources'],
            'llm_provider': llm_provider,
            'response_time': response_time
        }
    
    def _parse_llm_response(self, answer_text: str, source_docs: List[Dict]) -> Dict:
        """
        Parse the LLM response into structured format
        
        Args:
            answer_text: Raw text response from LLM
            source_docs: Source documents used
            
        Returns:
            Parsed response dictionary
        """
        lines = answer_text.split('\n')
        
        answer = ""
        confidence = "medium"
        follow_ups = []
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect sections
            if 'answer:' in line.lower():
                current_section = 'answer'
                # Extract answer if on same line
                if ':' in line:
                    answer = line.split(':', 1)[1].strip()
            elif 'confidence:' in line.lower():
                current_section = 'confidence'
                # Extract confidence level
                conf_text = line.split(':', 1)[1].strip().lower() if ':' in line else "medium"
                if conf_text in ['high', 'medium', 'low']:
                    confidence = conf_text
            elif 'follow-up' in line.lower():
                current_section = 'followup'
            elif current_section == 'answer' and not any(x in line.lower() for x in ['confidence:', 'follow-up']):
                answer += " " + line
            elif current_section == 'followup' and line.startswith(('-', '*', '•')):
                question = line.lstrip('-*• ').strip()
                if question:
                    follow_ups.append(question)
        
        # Fallback: if parsing failed, use entire text as answer
        if not answer.strip():
            answer = answer_text
        
        # Create source chunks with metadata
        sources = []
        for doc in source_docs:
            # Truncate content for display
            content = doc['content']
            if len(content) > 300:
                content = content[:300] + "..."
            
            sources.append(SourceChunk(
                content=content,
                page=self._extract_page_number(doc['content']),
                score=doc['similarity_score']
            ))
        
        return {
            'answer': answer.strip(),
            'confidence': confidence,
            'follow_up_questions': follow_ups[:3],  # Limit to 3
            'sources': sources
        }
    
    def _extract_page_number(self, text: str) -> int:
        """Extract page number from text if available"""
        match = re.search(r'--- Page (\d+) ---', text)
        if match:
            return int(match.group(1))
        return None