from flask import Blueprint, request, jsonify, current_app
from ..core.query_handler import QueryHandler
from ..core.command_handler import CommandHandler
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
from ..db.connection import get_db_connection

main_routes = Blueprint('main', __name__, url_prefix='/api')

def get_query_handler():
    """
    Create and return a properly initialized QueryHandler instance
    """
    try:
        # Get LLM configuration and return a ready QueryHandler
        llm_config = Config.get_llm_config()
        rag_manager = RAGManager()

        # Try to set DB context for RAG so schema snippets are available to QueryEnhancer
        try:
            conn = get_db_connection()
            if conn:
                try:
                    rag_manager.set_db_context(conn)
                    # Bootstrap RAG knowledge if empty
                    if rag_manager.is_empty():
                        try:
                            rag_manager.bootstrap_from_db(conn)
                        except Exception:
                            logging.warning('Failed to bootstrap RAG from DB during handler init')
                finally:
                    try:
                        conn.close()
                    except Exception:
                        pass
        except Exception:
            logging.warning('Could not initialize DB context for RAGManager in request')

        query_enhancer = QueryEnhancer(rag_manager)
        query_handler = QueryHandler(LLMClient(), query_enhancer)
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
    """Handle queries and commands including immediate actions."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 415

    data = request.get_json()
    command = data.get('question') or data.get('command')
    
    logging.info(f"Handling command/query: {command}")
    
    if not command:
        return jsonify({'error': 'Please provide a question or command'}), 400

    try:
        # Initialize command handler
        llm_client = LLMClient()
        query_handler = get_query_handler()
        command_handler = CommandHandler(llm_client, query_handler)
        
        result = command_handler.handle_command(command)

        # Determine status for history: previews are stored as 'preview'
        status = 'error' if result.get('error') else ('preview' if result.get('type') in ('query_preview', 'action_preview') else 'success')

        # Log to history (store preview entries so confirm can reference them)
        history_entry = {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'command': command,
            'type': result.get('type'),
            'sql': result.get('sql', ''),
            'explanation': result.get('explanation', ''),
            'status': status,
            'error': result.get('error', '')
        }

        history = Config.get_query_history()
        history.insert(0, history_entry)
        Config.save_query_history(history[:50])  # Keep last 50 queries

        # Return result plus history id so clients can confirm against it
        result['_history_id'] = history_entry['id']
        return jsonify(result)
            
    except Exception as e:
        error_message = str(e)
        logging.error(f"Error handling command: {error_message}")
        
        # Log failed command to history
        history_entry = {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'command': command,
            'status': 'error',
            'error': error_message
        }
        
        history = Config.get_query_history()
        history.insert(0, history_entry)
        Config.save_query_history(history[:50])
        
        return jsonify({'error': error_message}), 500


@main_routes.route('/confirm', methods=['POST'])
def confirm_action():
    """Execute a confirmed action or SQL (user clicked Confirm)."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 415

    data = request.get_json()
    command = data.get('command')
    sql = data.get('sql')
    action = data.get('action')
    history_id = data.get('history_id')

    if not history_id:
        return jsonify({'error': 'history_id is required to confirm a preview'}), 400

    try:
        # Load history and verify the preview exists and is in preview state
        history = Config.get_query_history()
        preview_entry = next((h for h in history if h.get('id') == history_id), None)
        if not preview_entry or preview_entry.get('status') != 'preview':
            return jsonify({'error': 'Invalid or expired preview id'}), 400

        # Proceed to execute, with server-side SQL validation to avoid executing queries
        query_handler = get_query_handler()
        command_handler = CommandHandler(None, query_handler)

        # Mark history as executing
        preview_entry['status'] = 'executing'
        Config.save_query_history(history[:50])

        # If action and sql provided (user edited SQL), prefer using provided sql
        if action and sql:
            # validate provided sql first
            valid = query_handler.validate_sql_against_schema(sql)
            if not valid.get('ok'):
                analysis = query_handler.analyze_error(sql, f"Validation failed: {valid}")
                preview_entry['status'] = 'error'
                preview_entry['error'] = f"SQL validation failed: {valid}"
                Config.save_query_history(history[:50])
                return jsonify({'error': 'SQL validation failed', 'details': valid, 'analysis': analysis}), 400
            res = command_handler._execute_action_with_optional_sql(command or '', action, sql)
        elif sql:
            # raw SQL provided — validate before executing
            valid = query_handler.validate_sql_against_schema(sql)
            if not valid.get('ok'):
                analysis = query_handler.analyze_error(sql, f"Validation failed: {valid}")
                preview_entry['status'] = 'error'
                preview_entry['error'] = f"SQL validation failed: {valid}"
                Config.save_query_history(history[:50])
                return jsonify({'error': 'SQL validation failed', 'details': valid, 'analysis': analysis}), 400
            res = {'type': 'sql_executed', 'results': query_handler.execute_sql(sql)}
        else:
            # Fallback: produce a preview first, validate generated SQL, then execute
            preview = command_handler.handle_command(command, execute=False)
            generated_sql = preview.get('sql')
            if generated_sql:
                valid = query_handler.validate_sql_against_schema(generated_sql)
                if not valid.get('ok'):
                    analysis = query_handler.analyze_error(generated_sql, f"Validation failed: {valid}")
                    preview_entry['status'] = 'error'
                    preview_entry['error'] = f"SQL validation failed: {valid}"
                    Config.save_query_history(history[:50])
                    return jsonify({'error': 'Generated SQL validation failed', 'details': valid, 'analysis': analysis}), 400

            # Now execute for real
            res = command_handler.handle_command(command, execute=True)

        # Update history entry with execution result
        preview_entry['status'] = 'success' if not res.get('error') else 'error'
        preview_entry['result'] = res
        Config.save_query_history(history[:50])

        return jsonify(res)

    except Exception as e:
        logging.error(f"Error executing confirmed action: {e}")
        # Update history as error
        try:
            history = Config.get_query_history()
            preview_entry = next((h for h in history if h.get('id') == history_id), None)
            if preview_entry:
                preview_entry['status'] = 'error'
                preview_entry['error'] = str(e)
                Config.save_query_history(history[:50])
        except Exception:
            pass
        return jsonify({'error': str(e)}), 500

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

@main_routes.route('/query/preview', methods=['POST'])
def preview_query():
    data = request.get_json()
    question = data.get('question')
    if not question:
        return jsonify({'error': 'question is required'}), 400
    try:
        qh = get_query_handler()
        preview = qh.preview_query(question)
        return jsonify(preview)
    except Exception as e:
        logging.error(f"Error generating preview: {e}")
        return jsonify({'error': str(e)}), 500

@main_routes.route('/query/history', methods=['GET'])
def get_query_history():
    try:
        history = Config.get_query_history()
        return jsonify(history)
    except Exception as e:
        logging.error(f"Error fetching history: {e}")
        return jsonify({'error': 'Failed to load history'}), 500

@main_routes.route('/query/history/<int:query_id>', methods=['GET', 'PUT', 'DELETE'])
def manage_query_history(query_id):
    history = Config.get_query_history()
    if request.method == 'GET':
        item = next((h for h in history if str(h.get('id')) == str(query_id)), None)
        return jsonify(item) if item else ('', 404)

    elif request.method == 'PUT':
        updates = request.get_json()
        for i, h in enumerate(history):
            if str(h.get('id')) == str(query_id):
                history[i].update(updates)
                Config.save_query_history(history[:50])
                return ('', 200)
        return ('', 404)

    elif request.method == 'DELETE':
        new_history = [h for h in history if not (str(h.get('id')) == str(query_id))]
        Config.save_query_history(new_history[:50])
        return ('', 204)