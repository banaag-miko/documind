from pydantic import BaseModel, Field, validator
from typing import List, Optional, Literal


class DocumentUploadRequest(BaseModel):
    """Validates document upload requests"""
    file_type: str = Field(..., description="Type of file being uploaded")
    
    @validator('file_type')
    def validate_file_type(cls, v):
        allowed = ['pdf', 'txt', 'md']
        if v.lower() not in allowed:
            raise ValueError(f'File type must be one of {allowed}')
        return v.lower()


class QueryRequest(BaseModel):
    """Validates query requests"""
    document_id: int = Field(..., gt=0, description="ID of the document to query")
    question: str = Field(..., min_length=3, max_length=1000, description="Question to ask")
    llm_provider: Literal['openai', 'anthropic', 'google'] = Field(
        default='openai',
        description="LLM provider to use"
    )
    
    @validator('question')
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError('Question cannot be empty')
        return v.strip()


class SourceChunk(BaseModel):
    """Represents a source chunk from the document"""
    content: str = Field(..., description="Content of the chunk")
    page: Optional[int] = Field(None, description="Page number if available")
    score: float = Field(..., description="Relevance score")
    
    class Config:
        # Allow conversion to dict
        from_attributes = True


class QueryResponse(BaseModel):
    """Structured response from the LLM"""
    answer: str = Field(..., description="Answer to the question")
    sources: List[SourceChunk] = Field(default_factory=list, description="Source chunks used")
    confidence: Literal['high', 'medium', 'low'] = Field(
        default='medium',
        description="Confidence level of the answer"
    )
    follow_up_questions: List[str] = Field(
        default_factory=list,
        description="Suggested follow-up questions"
    )