from flask import Flask
from src.web.routes import main_routes
from src.web.views import view_routes
from src.web.settings_routes import settings_routes
from src.web.rag_routes import rag_routes
import os

def create_app():
    # Tell Flask where to find the templates folder
    app = Flask(__name__, template_folder='web/templates')
    
    # Set up configuration
    app.config['CONFIG_DIR'] = 'config'
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.config['CHROMA_DIR'] = os.path.join(app.config['CONFIG_DIR'], 'chroma_db')
    
    # Ensure required directories exist
    os.makedirs(app.config['CONFIG_DIR'], exist_ok=True)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['CHROMA_DIR'], exist_ok=True)
    
    # Register blueprints
    app.register_blueprint(view_routes)  # No URL prefix - handles main views
    app.register_blueprint(main_routes)  # API endpoints
    app.register_blueprint(settings_routes, url_prefix='/settings')  # Settings routes
    app.register_blueprint(rag_routes)  # RAG routes
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)