import os
import json
from flask import Blueprint, request, jsonify, current_app
from ..rag.manager import RAGManager
from ..config import Config

rag_routes = Blueprint('rag', __name__)
rag_manager = RAGManager()

@rag_routes.route('/api/rag/schema', methods=['GET', 'POST'])
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

@rag_routes.route('/api/rag/business-rules', methods=['GET', 'POST'])
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

@rag_routes.route('/api/rag/sources', methods=['GET', 'POST'])
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
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 415
            
        data = request.get_json()
        new_source = data.get('url')
        
        if not new_source:
            return jsonify({'error': 'No URL provided'}), 400
            
        # Load existing sources
        try:
            with open(sources_path, 'r') as f:
                existing_data = json.load(f)
        except FileNotFoundError:
            existing_data = {'sources': []}
            
        # Add new source if not already present
        if new_source not in [s['url'] for s in existing_data['sources']]:
            existing_data['sources'].append({'url': new_source})
            
            # Save updated sources
            os.makedirs(os.path.dirname(sources_path), exist_ok=True)
            with open(sources_path, 'w') as f:
                json.dump(existing_data, f, indent=4)
                
        return jsonify({'success': True})

@rag_routes.route('/api/rag/knowledge', methods=['GET'])
def get_all_knowledge():
    """Endpoint to get all stored knowledge."""
    knowledge = rag_manager.get_all_knowledge()
    return jsonify(knowledge)