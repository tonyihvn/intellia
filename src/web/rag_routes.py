from flask import Blueprint, request, jsonify
from ..rag.manager import RAGManager

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

@rag_routes.route('/api/rag/knowledge', methods=['GET'])
def get_all_knowledge():
    """Endpoint to get all stored knowledge."""
    knowledge = rag_manager.get_all_knowledge()
    return jsonify(knowledge)