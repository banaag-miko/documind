from flask import Flask, render_template
from datetime import datetime


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    print("=" * 60)
    print("🧠 DocuMind - AI Document Intelligence Platform")
    print("=" * 60)
    print("Starting server...")
    print("Access the application at: http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)

# Runner and Debugger
if __name__ == '__main__':
    with app.app_context():
        app.run(debug=True)