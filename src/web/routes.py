from flask import Blueprint, request, jsonify, current_app
from ..core.query_handler import QueryHandler
from ..llm.client import LLMClient
from ..rag.enhancer import QueryEnhancer
from ..rag.manager import RAGManager
from ..config import Config
from .history_manager import HistoryManager
import os
import logging
import mysql.connector
from datetime import datetime
import uuid

main_routes = Blueprint('main', __name__, url_prefix='/api')

def get_query_handler():
    """
    Create and return a properly initialized QueryHandler instance
    """
    try:
        # Get LLM configuration
        llm_config = Config.get_llm_config()
        
        # Initialize LLM client with provider and settings
        llm_client = None
        providers = llm_config.get('providers', {})
        
        # Try each provider in order of preference (OpenAI first)
        for provider in ['openai', 'google', 'anthropic']:
            settings = providers.get(provider, {})
            if settings.get('api_key'):
                try:
                    llm_client = LLMClient(provider=provider, settings=settings)
                    logging.info(f"Successfully initialized LLM client with {provider}")
                    break
                except Exception as e:
                    logging.warning(f"Failed to initialize {provider} client: {e}")
                    continue
                
        if not llm_client:
            logging.info("No cloud providers configured, using local provider")
            llm_client = LLMClient()  # Uses default local provider
            
        # Initialize RAG manager and query enhancer
        rag_manager = RAGManager()
        query_enhancer = QueryEnhancer(rag_manager)
        
        # Create and return QueryHandler instance
        query_handler = QueryHandler(llm_client, query_enhancer)
        logging.info("Successfully initialized QueryHandler")
        return query_handler
        
    except Exception as e:
        logging.error(f"Error initializing QueryHandler: {e}")
        raise

@main_routes.route('/config/db', methods=['GET', 'POST'])
def handle_db_config():
    if request.method == 'GET':
        return jsonify(Config.get_db_config())
    
    if request.method == 'POST':
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 415
            
        new_config = request.get_json()
        required_fields = ['host', 'user', 'password', 'database', 'port']
        
        if not all(field in new_config for field in required_fields):
            return jsonify({'error': 'Missing required database configuration fields'}), 400
            
        # Test the connection before saving
        try:
            conn = mysql.connector.connect(**new_config)
            conn.close()
        except mysql.connector.Error as e:
            return jsonify({'error': f'Database connection failed: {str(e)}'}), 400
            
        Config.save_db_config(new_config)
        return jsonify({'message': 'Database configuration updated successfully'})

@main_routes.route('/history', methods=['GET', 'POST', 'DELETE'])
def handle_history():
    if request.method == 'GET':
        try:
            history = HistoryManager.load_history()
            return jsonify(history)
        except Exception as e:
            logging.error(f"Error getting query history: {str(e)}")
            return jsonify({'error': 'Failed to load query history'}), 500
            
    elif request.method == 'POST':
        try:
            data = request.get_json()
            if not data or 'question' not in data:
                return jsonify({'error': 'Missing required fields'}), 400
                
            new_item = HistoryManager.add_item(
                question=data['question'],
                sql=data.get('sql', ''),
                result=data.get('result', ''),
                status=data.get('status', 'success')
            )
            
            if new_item:
                return jsonify(new_item)
            return jsonify({'error': 'Failed to save history item'}), 500
            
        except Exception as e:
            logging.error(f"Error saving to history: {str(e)}")
            return jsonify({'error': 'Failed to save to history'}), 500
            
    elif request.method == 'DELETE':
        try:
            if HistoryManager.clear_history():
                return jsonify({'message': 'Query history cleared successfully'})
            return jsonify({'error': 'Failed to clear history'}), 500
            
        except Exception as e:
            logging.error(f"Error clearing history: {str(e)}")
            return jsonify({'error': 'Failed to clear history'}), 500
            
@main_routes.route('/history/<string:id>', methods=['GET', 'DELETE'])
def handle_history_item(id):
    try:
        if request.method == 'GET':
            item = HistoryManager.get_item(id)
            if item:
                return jsonify(item)
            return jsonify({'error': 'History item not found'}), 404
            
        elif request.method == 'DELETE':
            if HistoryManager.delete_item(id):
                return jsonify({'message': 'History item deleted successfully'})
            return jsonify({'error': 'History item not found'}), 404
            
    except Exception as e:
        logging.error(f"Error handling history item: {str(e)}")
        return jsonify({'error': f'Failed to handle history item: {str(e)}'}), 500

@main_routes.route('/query', methods=['POST'])
def handle_query():
    """Handle SQL query generation and execution."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 415

    data = request.get_json()
    question = data.get('question')
    action = data.get('action', 'execute')  # 'generate' or 'execute'
    
    logging.info(f"Handling query request: action={action}, question={question}")
    
    if not question:
        return jsonify({'error': 'Please provide a question to generate SQL for'}), 400

    try:
        # Initialize query handler
        query_handler = get_query_handler()
        
        if action == 'generate':
            # Generate SQL without executing
            result = query_handler.handle_query(question, execute=False)
            
            if not result or 'sql' not in result:
                return jsonify({'error': 'Failed to generate SQL query'}), 500
            
            # Update history
            history = Config.get_query_history()
            history_item = {
                'id': str(uuid.uuid4()),
                'timestamp': datetime.now().isoformat(),
                'question': question,
                'sql': sql,
                'status': 'success'
            }
            history.insert(0, history_item)
            Config.save_query_history(history[:50])  # Keep last 50 queries
            
            return jsonify({'prompt': prompt, 'sql': sql})
            
        else:  # action == 'execute'
            # Generate SQL and execute
            result = query_handler.handle_query(question, execute=True)
            
            # Log the query to history
            history_entry = {
                'id': str(uuid.uuid4()),
                'timestamp': datetime.now().isoformat(),
                'question': question,
                'sql': result.get('sql', ''),
                'explanation': result.get('explanation', ''),
                'status': 'executed' if result.get('results') else 'error',
                'error': result.get('error', '')
            }
            
            history = Config.get_query_history()
            history.insert(0, history_entry)
            Config.save_query_history(history[:50])  # Keep last 50 queries
            
            return jsonify(result)
            
    except Exception as e:
        error_message = str(e)
        logging.error(f"Error handling query: {error_message}")
        
        # Log failed query to history
        history_entry = {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'status': 'error',
            'error': error_message
        }
        
        history = Config.get_query_history()
        history.insert(0, history_entry)
        Config.save_query_history(history[:50])
        
        return jsonify({'error': error_message}), 500

@main_routes.route('/execute', methods=['POST'])
def execute_sql():
    """Execute a raw SQL query sent from the client."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 415

    data = request.get_json()
    sql = data.get('sql')
    if not sql:
        return jsonify({'error': 'SQL is required'}), 400

    try:
        query_handler = get_query_handler()
        results = query_handler.execute_sql(sql)
        return jsonify(results)
    except Exception as e:
        logging.error(f"Error executing SQL: {str(e)}")
        return jsonify({'error': str(e)}), 500