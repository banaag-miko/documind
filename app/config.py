import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Application configuration class"""
    
    # Flask Config
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.getenv('FLASK_ENV') == 'development'
    
    # File Upload Config
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_FILE_SIZE', 16 * 1024 * 1024))  # 16MB default
    ALLOWED_EXTENSIONS = {'pdf', 'txt', 'md'}
    
    # Database Config - SQLite for development (easy to get started)
    BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    SQLALCHEMY_DATABASE_URI = f'sqlite:///{os.path.join(BASE_DIR, "documind.db")}'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Vector Store Config
    VECTOR_STORE_PATH = os.getenv('VECTOR_STORE_PATH', 'vector_store')
    
    # LLM API Keys
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    
    
    # LangChain Tracing
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    os.environ['LANGSMITH_PROJECT'] = "documind-v2"
    LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY")
    LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY")
    
     # ChromaDB Cloud settings
    CHROMA_API_KEY = os.environ.get('CHROMA_API_KEY')
    CHROMA_TENANT = os.environ.get('CHROMA_TENANT')
    CHROMA_DATABASE = os.environ.get('CHROMA_DATABASE')
    CHROMA_HOST = os.environ.get('CHROMA_HOST', 'api.trychroma.com')
    
    @staticmethod
    def init_app(app):
        """Initialize application"""
        # Ensure required directories exist
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.VECTOR_STORE_PATH, exist_ok=True)