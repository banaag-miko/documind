import os
import uuid
from flask import Blueprint, request, jsonify, render_template
from werkzeug.utils import secure_filename
from pydantic import ValidationError

from models.document import db, Document, Query
from models.schemas import QueryRequest
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import VectorStoreService
from app.services.rag_service import RAGService
from app.config import Config

# Create Blueprint
api_bp = Blueprint('api', __name__)

# Initialize services
doc_processor = DocumentProcessor()
vector_service = VectorStoreService()
rag_service = RAGService()


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


@api_bp.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@api_bp.route('/api/upload', methods=['POST'])
def upload_document():
    """Upload and process a document"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'File type not allowed. Allowed types: {", ".join(Config.ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Generate unique filename
        file_ext = file.filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4()}.{file_ext}"
        file_path = os.path.join(Config.UPLOAD_FOLDER, unique_filename)
        
        # Save file
        file.save(file_path)
        file_size = os.path.getsize(file_path)
        
        # Create database record
        document = Document(
            filename=unique_filename,
            original_filename=file.filename,
            file_path=file_path,
            file_size=file_size,
            file_type=file_ext
        )
        db.session.add(document)
        db.session.commit()
        
        # Process document
        try:
            # Extract and chunk
            chunks, chunk_count = doc_processor.process_document(
                file_path, file_ext, document.id
            )
            
            # Create vector store collection
            collection_name = f"doc_{document.id}"
            vector_service.add_documents(collection_name, chunks)
            
            # Update document record
            document.processed = True
            document.chunk_count = chunk_count
            document.vector_store_id = collection_name
            db.session.commit()
            
            return jsonify({
                'message': 'Document uploaded and processed successfully',
                'document': document.to_dict()
            }), 201
            
        except Exception as e:
            # Clean up on processing error
            document.processed = False
            db.session.commit()
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/api/query', methods=['POST'])
def query_document():
    """Query a document using RAG"""
    try:
        # Validate request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        query_req = QueryRequest(**data)
        
        # Check if document exists
        document = Document.query.get(query_req.document_id)
        if not document:
            return jsonify({'error': 'Document not found'}), 404
        
        if not document.processed:
            return jsonify({'error': 'Document not yet processed'}), 400
        
        # Perform RAG query
        result = rag_service.query_document(
            collection_name=document.vector_store_id,
            question=query_req.question,
            llm_provider=query_req.llm_provider
        )
        
        # Save query to database
        query_record = Query(
            document_id=document.id,
            question=query_req.question,
            answer=result['answer'],
            llm_provider=result['llm_provider'],
            response_time=result['response_time'],
            confidence=result['confidence']
        )
        db.session.add(query_record)
        db.session.commit()
        
        return jsonify({
            'query_id': query_record.id,
            'answer': result['answer'],
            'confidence': result['confidence'],
            'follow_up_questions': result['follow_up_questions'],
            'sources': [s.model_dump() for s in result['sources']],
            'llm_provider': result['llm_provider'],
            'response_time': round(result['response_time'], 2)
        }), 200
        
    except ValidationError as e:
        return jsonify({'error': 'Validation error', 'details': e.errors()}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/api/documents', methods=['GET'])
def list_documents():
    """List all documents"""
    try:
        documents = Document.query.order_by(Document.upload_date.desc()).all()
        return jsonify({
            'documents': [doc.to_dict() for doc in documents]
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/api/documents/<int:doc_id>', methods=['GET'])
def get_document(doc_id):
    """Get document details"""
    try:
        document = Document.query.get(doc_id)
        if not document:
            return jsonify({'error': 'Document not found'}), 404
        
        return jsonify({'document': document.to_dict()}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/api/documents/<int:doc_id>/history', methods=['GET'])
def get_query_history(doc_id):
    """Get query history for a document"""
    try:
        document = Document.query.get(doc_id)
        if not document:
            return jsonify({'error': 'Document not found'}), 404
        
        queries = Query.query.filter_by(document_id=doc_id)\
                            .order_by(Query.query_date.desc()).all()
        
        return jsonify({
            'document_id': doc_id,
            'queries': [q.to_dict() for q in queries]
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/api/documents/<int:doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    """Delete a document and its associated data"""
    try:
        document = Document.query.get(doc_id)
        if not document:
            return jsonify({'error': 'Document not found'}), 404
        
        # Delete physical file
        if os.path.exists(document.file_path):
            os.remove(document.file_path)
        
        # Delete vector store collection
        if document.vector_store_id:
            vector_service.delete_collection(document.vector_store_id)
        
        # Delete from database (cascades to queries)
        db.session.delete(document)
        db.session.commit()
        
        return jsonify({'message': 'Document deleted successfully'}), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@api_bp.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    # Check which LLM providers are configured
    configured_providers = []
    if Config.OPENAI_API_KEY:
        configured_providers.append('openai')
    if Config.ANTHROPIC_API_KEY:
        configured_providers.append('anthropic')
    if Config.GOOGLE_API_KEY:
        configured_providers.append('google')
    
    return jsonify({
        'status': 'healthy',
        'configured_providers': configured_providers
    }), 200