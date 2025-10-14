from flask import Blueprint, request, jsonify
from src.core.query_handler import QueryHandler
from config import Config
import logging
import mysql.connector
from datetime import datetime

main_routes = Blueprint('main', __name__, url_prefix='/api')

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

@main_routes.route('/history', methods=['GET'])
def get_history():
    return jsonify(Config.get_query_history())

@main_routes.route('/query', methods=['POST'])
def handle_query():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 415

    data = request.get_json()
    question = data.get('question')
    action = data.get('action', 'execute')  # 'generate' or 'execute'
    
    if not question:
        return jsonify({'error': 'JSON body must contain a non-empty "question" key'}), 400

    try:
        handler = QueryHandler()
        if action == 'generate':
            sql_query = handler.generate_sql_query(question)
            return jsonify({'sql_query': sql_query})
        else:
            result = handler.handle_query(question)
            
            # Log the query to history
            history_entry = {
                'timestamp': datetime.now().isoformat(),
                'question': question,
                'sql_query': result.get('sql_query', ''),
                'status': 'success' if result.get('data') else 'error',
                'error': result.get('error', '')
            }
            Config.add_to_query_history(history_entry)
            
            return jsonify(result)
    except Exception as e:
        logging.error(f"Error handling query: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
    try:
        query_handler = QueryHandler()
        
        if action == 'generate':
            # Only generate the SQL query
            sql = query_handler.generate_sql(question)
            
            # Update history
            history = Config.get_query_history()
            history.insert(0, {
                'question': question,
                'sql': sql,
                'timestamp': datetime.now().isoformat()
            })
            Config.save_query_history(history[:50])  # Keep last 50 queries
            
            return jsonify({'sql': sql})
        
        elif action == 'execute':
            # Execute the provided SQL and return results
            sql = data.get('sql')
            if not sql:
                return jsonify({'error': 'SQL query is required for execution'}), 400
                
            result = query_handler.execute_sql(sql)
            return jsonify({'result': result})
        
        else:
            return jsonify({'error': 'Invalid action specified'}), 400
            
    except Exception as e:
        logging.error(f"An error occurred while handling the query: {e}")
        return jsonify({'error': str(e)}), 500