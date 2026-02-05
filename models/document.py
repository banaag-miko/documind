from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
import chromadb
from chromadb.config import Settings

db = SQLAlchemy()

# Initialize ChromaDB client (will be configured in app factory)
chroma_client = None
collection = None


def init_chromadb(app):
    """Initialize ChromaDB with app config"""
    global chroma_client, collection
    
    # Use cloud client if credentials are provided
    if app.config.get('CHROMA_API_KEY'):
        chroma_client = chromadb.CloudClient(
            api_key=app.config['CHROMA_API_KEY'],
            tenant=app.config['CHROMA_TENANT'],
            database=app.config['CHROMA_DATABASE']
        )
    else:
        # Fallback to local persistent client
        chroma_client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
    
    # Create or get collection
    collection = chroma_client.get_or_create_collection(
        name="document_chunks",
        metadata={"hnsw:space": "cosine"}
    )
    
    return chroma_client, collection


class Document(db.Model):
    """Document metadata stored in SQLite"""
    __tablename__ = 'documents'
    
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    file_size = db.Column(db.Integer)
    file_type = db.Column(db.String(50))
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    processed = db.Column(db.Boolean, default=False)
    chunk_count = db.Column(db.Integer, default=0)
    vector_store_id = db.Column(db.String(100))  # ✅ ADD THIS LINE
    
    queries = db.relationship('Query', backref='document', lazy=True, cascade='all, delete-orphan')
    
    def add_to_chromadb(self, chunks, embeddings=None):
        """Add document chunks to ChromaDB"""
        ids = [f"doc_{self.id}_chunk_{i}" for i in range(len(chunks))]
        
        metadatas = [{
            "document_id": self.id,
            "filename": self.original_filename,
            "chunk_index": i,
            "upload_date": self.upload_date.isoformat()
        } for i in range(len(chunks))]
        
        if embeddings:
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas
            )
        else:
            collection.add(
                ids=ids,
                documents=chunks,
                metadatas=metadatas
            )
        
        self.chunk_count = len(chunks)
        self.processed = True
        self.vector_store_id = f"doc_{self.id}_collection"  # ✅ SET THE VALUE
    
    def query_chromadb(self, query_text, n_results=5):
        """Query ChromaDB for relevant chunks"""
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where={"document_id": self.id}
        )
        return results
    
    def delete_from_chromadb(self):
        """Delete all chunks for this document"""
        results = collection.get(where={"document_id": self.id})
        if results['ids']:
            collection.delete(ids=results['ids'])
    
    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.original_filename,
            'file_size': self.file_size,
            'file_type': self.file_type,
            'upload_date': self.upload_date.isoformat(),
            'processed': self.processed,
            'chunk_count': self.chunk_count,
            'vector_store_id': self.vector_store_id  # ✅ ADD TO DICT
        }
    
    def __repr__(self):
        return f'<Document {self.original_filename}>'


class Query(db.Model):
    """Query history stored in SQLite"""
    __tablename__ = 'queries'
    
    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey('documents.id'), nullable=False)
    question = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text)
    llm_provider = db.Column(db.String(50))
    query_date = db.Column(db.DateTime, default=datetime.utcnow)
    response_time = db.Column(db.Float)
    confidence = db.Column(db.String(20))
    chunks_used = db.Column(db.Integer)
    
    def to_dict(self):
        return {
            'id': self.id,
            'document_id': self.document_id,
            'question': self.question,
            'answer': self.answer,
            'llm_provider': self.llm_provider,
            'query_date': self.query_date.isoformat(),
            'response_time': self.response_time,
            'confidence': self.confidence,
            'chunks_used': self.chunks_used
        }
    
    def __repr__(self):
        return f'<Query {self.id}: {self.question[:50]}...>'