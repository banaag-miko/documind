import os
from flask import Flask
from flask_cors import CORS
from app.config import Config
from models.document import db, init_chromadb


def create_app(config_class=Config):
    """
    Application factory pattern
    
    Args:
        config_class: Configuration class to use
        
    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Initialize configuration
    config_class.init_app(app)
    
    # Enable CORS for all routes
    CORS(app)
    
    # Initialize database
    db.init_app(app)
    
    # Register blueprints
    from app.routes.api import api_bp
    app.register_blueprint(api_bp)
    
    # Create database tables
    with app.app_context():
        init_chromadb(app)
        db.create_all()
        print("✓ Database tables created")
    
    return app