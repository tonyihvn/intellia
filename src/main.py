from flask import Flask
from src.web.routes import main_routes
from src.web.views import view_routes
from src.web.settings_routes import settings_routes
from src.web.rag_routes import rag_routes
from src.web.schedules_routes import schedules_routes
from src.scheduler.scheduler import get_scheduler
from src.db.connection import get_db_connection
from src.rag.manager import RAGManager
import os
import logging

def check_database_connection():
    """Check if database connection can be established"""
    try: 
        conn = get_db_connection()
        if conn:
            conn.close()
            logging.info("Database connection successful")
            return True
        return False
    except Exception as e:
        logging.error(f"Database connection failed: {str(e)}")
        return False

def create_app():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create Flask app with template and static folders in the web directory
    app = Flask(__name__, 
                template_folder='web/templates',
                static_folder='web/static')
    
    # Set up configuration
    app.config['CONFIG_DIR'] = 'config'
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.config['VECTOR_DIR'] = os.path.join(app.config['CONFIG_DIR'], 'vector_store')
    
    # Ensure required directories exist
    os.makedirs(app.config['CONFIG_DIR'], exist_ok=True)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['VECTOR_DIR'], exist_ok=True)
    
    # Register blueprints
    app.register_blueprint(view_routes)  # Main views (/, /settings, /settings/db, /rag)
    app.register_blueprint(main_routes)  # API endpoints
    app.register_blueprint(settings_routes)  # Settings API routes
    app.register_blueprint(rag_routes)  # RAG API routes
    app.register_blueprint(schedules_routes)  # Scheduler API routes

    # Initialize scheduler manager (runs in background)
    try:
        get_scheduler()
        logging.info("Scheduler initialized")
    except Exception as e:
        logging.warning(f"Failed to initialize scheduler: {e}")
    
    # Initialize RAG with database context
    try:
        if check_database_connection():
            logging.info("Successfully connected to database")
            conn = get_db_connection()
            if conn:
                rm = RAGManager()
                # Set database context
                rm.set_db_context(conn)
                
                # Bootstrap if empty
                if rm.is_empty():
                    ok = rm.bootstrap_from_db(conn)
                    if ok:
                        logging.info("RAG knowledge bootstrapped from database schema")
                conn.close()
    except Exception as e:
        logging.error(f"Error initializing RAG with database context: {str(e)}")
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)