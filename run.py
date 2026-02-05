import os

from flask import render_template
from app import create_app

# Create the Flask application
app = create_app()

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # Get configuration from environment
    debug = os.environ.get('FLASK_DEBUG', 'True') == 'True'
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5000))
    
    print(f"🚀 Starting Documind server...")
    print(f"   Running on http://{host}:{port}")
    print(f"   Debug mode: {debug}")
    
    app.run(
        host=host,
        port=port,
        debug=debug
    )