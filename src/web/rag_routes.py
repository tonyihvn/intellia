import os
import json
import logging
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from ..rag.manager import RAGManager
from ..config import Config
from ..db.connection import get_db_connection

rag_routes = Blueprint('rag', __name__)
rag_manager = RAGManager()

@rag_routes.route('/api/rag/schema', methods=['GET', 'POST', 'DELETE'])
def handle_schema_knowledge():
    """Endpoint to manage schema-related knowledge."""
    if request.method == 'GET':
        knowledge = rag_manager.get_all_knowledge()
        return jsonify({'schema': knowledge['schema']})
    
    elif request.method == 'POST':
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 415
            
        data = request.get_json()
        descriptions = data.get('descriptions', [])
        
        if not descriptions:
            return jsonify({'error': 'No descriptions provided'}), 400
            
        success = rag_manager.add_schema_knowledge(descriptions)
        return jsonify({'success': success})
        
    elif request.method == 'DELETE':
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 415
            
        data = request.get_json()
        item_text = data.get('id')  # The text content serves as the ID
        
        if not item_text:
            return jsonify({'error': 'No item ID provided'}), 400
            
        success = rag_manager.delete_by_text('schema', item_text)
        if success:
            return jsonify({'success': True})
        return jsonify({'error': 'Failed to delete item'}), 500

@rag_routes.route('/api/rag/business-rules', methods=['GET', 'POST', 'DELETE'])
def handle_business_rules():
    """Endpoint to manage business rule knowledge."""
    if request.method == 'GET':
        knowledge = rag_manager.get_all_knowledge()
        return jsonify({'business_rules': knowledge['business_rules']})
    
    elif request.method == 'POST':
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 415
            
        data = request.get_json()
        descriptions = data.get('descriptions', [])
        
        if not descriptions:
            return jsonify({'error': 'No descriptions provided'}), 400
            
        success = rag_manager.add_business_rule(descriptions)
        return jsonify({'success': success})
        
    elif request.method == 'DELETE':
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 415
            
        data = request.get_json()
        item_text = data.get('id')  # The text content serves as the ID
        
        if not item_text:
            return jsonify({'error': 'No item ID provided'}), 400
            
        success = rag_manager.delete_by_text('business_rules', item_text)
        if success:
            return jsonify({'success': True})
        return jsonify({'error': 'Failed to delete item'}), 500

@rag_routes.route('/api/rag/sources', methods=['GET', 'POST', 'DELETE'])
def manage_sources():
    """Endpoint to manage external context sources."""
    sources_path = os.path.join(current_app.config.get('CONFIG_DIR', Config.CONFIG_DIR), 'sources.json')
    
    if request.method == 'GET':
        try:
            with open(sources_path, 'r') as f:
                data = json.load(f)
                return jsonify({'sources': data.get('sources', [])})
        except FileNotFoundError:
            return jsonify({'sources': []})
            
    elif request.method == 'POST':
        # Accept either JSON with a URL or form-data file upload
        new_entry = None
        # Handle file upload (multipart/form-data)
        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename:
                filename = secure_filename(file.filename)
                upload_dir = os.path.join(current_app.static_folder, 'uploads', 'context_sources')
                os.makedirs(upload_dir, exist_ok=True)
                file_path = os.path.join(upload_dir, filename)
                try:
                    file.save(file_path)
                except Exception as e:
                    return jsonify({'error': f'Failed to save uploaded file: {str(e)}'}), 500
                new_entry = {'filename': filename, 'path': file_path, 'type': 'file'}

        else:
            # JSON payload with URL
            if not request.is_json:
                return jsonify({"error": "Request must be JSON or multipart/form-data"}), 415
            data = request.get_json()
            new_source = data.get('url')
            if not new_source:
                return jsonify({'error': 'No URL provided'}), 400
            new_entry = {'url': new_source, 'type': 'url'}

        # Load existing sources
        try:
            with open(sources_path, 'r') as f:
                existing_data = json.load(f)
        except FileNotFoundError:
            existing_data = {'sources': []}

        # Avoid duplicates by url or filename
        exists = False
        for s in existing_data.get('sources', []):
            if new_entry.get('type') == 'url' and s.get('url') == new_entry.get('url'):
                exists = True
                break
            if new_entry.get('type') == 'file' and s.get('filename') == new_entry.get('filename'):
                exists = True
                break

        if not exists:
            existing_data.setdefault('sources', []).append(new_entry)
            os.makedirs(os.path.dirname(sources_path), exist_ok=True)
            with open(sources_path, 'w') as f:
                json.dump(existing_data, f, indent=4)

        return jsonify({'success': True, 'source': new_entry})

    elif request.method == 'DELETE':
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 415
            
        data = request.get_json()
        source_to_remove = data.get('source')
        
        if not source_to_remove:
            return jsonify({'error': 'No source identifier provided'}), 400

        try:
            with open(sources_path, 'r') as f:
                existing_data = json.load(f)
        except FileNotFoundError:
            return jsonify({'error': 'Sources file not found'}), 404

        sources = existing_data.get('sources', [])
        source_found = None
        
        # Find the source to remove
        for source in sources:
            if source_to_remove in (source.get('url'), source.get('filename')):
                source_found = source
                break
                
        if not source_found:
            return jsonify({'error': 'Source not found'}), 404

        # Remove source from list
        sources.remove(source_found)

        # Delete the file if it exists
        if 'path' in source_found:
            try:
                file_path = source_found['path']
                if os.path.exists(file_path):
                    os.remove(file_path)
            except OSError as e:
                # Log error but continue since we still want to remove from sources list
                logging.warning(f"Error removing source file: {e}")

        # Save updated sources list
        try:
            with open(sources_path, 'w') as f:
                json.dump({'sources': sources}, f, indent=4)
            return jsonify({'success': True})
        except Exception as e:
            return jsonify({'error': f'Failed to update sources file: {str(e)}'}), 500

@rag_routes.route('/api/rag/knowledge', methods=['GET'])
def get_all_knowledge():
    """Endpoint to get all stored knowledge."""
    try:
        knowledge = rag_manager.get_all_knowledge()
        # Ensure keys exist
        knowledge = knowledge or {}
        return jsonify({
            'schema': knowledge.get('schema', []),
            'business_rules': knowledge.get('business_rules', []),
            'examples': knowledge.get('examples', []),
        })
    except Exception as e:
        # Return empty lists rather than an error status so the UI can show empty state
        logging.warning(f"Failed to load RAG knowledge: {e}")
        return jsonify({'schema': [], 'business_rules': [], 'examples': [], 'error': str(e)})


@rag_routes.route('/api/rag/clarify', methods=['POST'])
def clarify_selection():
    """Accept clarifier selections from the UI and return a compact context summary.

    Expected JSON: { question: str, selected_tables: [str] }
    Returns: same shape as RAGManager.get_compact_context
    """
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 415

    data = request.get_json()
    question = data.get('question') or ''
    selected = data.get('selected_tables') or []

    # Try to set DB context so schema snippets are available
    try:
        conn = get_db_connection()
        if conn:
            try:
                rag_manager.set_db_context(conn)
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
    except Exception:
        # proceed even if DB context couldn't be set
        pass

    # If user selected tables, bias the question to include them so get_compact_context will use them
    if selected and isinstance(selected, (list, tuple)):
        augmented = f"{question}. Use tables: {', '.join(selected)}"
    else:
        augmented = question

    compact = rag_manager.get_compact_context(augmented)
    return jsonify(compact)


@rag_routes.route('/api/rag/summarize', methods=['POST'])
def summarize_compact_context():
    """Return the compact context summary for a question (optionally biased by selected_tables)."""
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 415
    data = request.get_json()
    question = data.get('question') or ''
    selected = data.get('selected_tables') or []

    if selected and isinstance(selected, (list, tuple)):
        question = f"{question}. Use tables: {', '.join(selected)}"

    try:
        compact = rag_manager.get_compact_context(question)
        return jsonify({'summary': compact.get('summary', ''), 'tables': compact.get('tables', []), 'rules': compact.get('rules', [])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500