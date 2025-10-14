from flask import Blueprint, render_template, request, jsonify, current_app
import json
import os
from werkzeug.utils import secure_filename

settings_routes = Blueprint('settings', __name__)

def get_config_path(config_type):
    """Get the path to a configuration file"""
    config_dir = current_app.config.get('CONFIG_DIR', 'config')
    return os.path.join(config_dir, f'{config_type}.json')

def load_config(config_type):
    """Load configuration from a JSON file"""
    config_path = get_config_path(config_type)
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_config(config_type, data):
    """Save configuration to a JSON file"""
    config_path = get_config_path(config_type)
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(data, f, indent=4)

@settings_routes.route('/')
def settings_page():
    return render_template('settings.html')

@settings_routes.route('/api/settings/guiders', methods=['GET', 'POST', 'DELETE'])
def handle_guiders():
    config = load_config('guiders')
    
    if request.method == 'GET':
        return jsonify(config.get('guiders', []))
    
    elif request.method == 'POST':
        data = request.get_json()
        guiders = config.get('guiders', [])
        if data.get('guider') not in guiders:
            guiders.append(data['guider'])
            save_config('guiders', {'guiders': guiders})
        return jsonify({'success': True})
    
    elif request.method == 'DELETE':
        data = request.get_json()
        guiders = config.get('guiders', [])
        if data.get('guider') in guiders:
            guiders.remove(data['guider'])
            save_config('guiders', {'guiders': guiders})
        return jsonify({'success': True})

@settings_routes.route('/api/settings/sources', methods=['GET', 'POST', 'DELETE'])
def handle_sources():
    config = load_config('sources')
    
    if request.method == 'GET':
        return jsonify(config.get('sources', []))
    
    elif request.method == 'POST':
        sources = config.get('sources', [])
        source_info = {}
        
        if 'url' in request.form:
            source_info['url'] = request.form['url']
        
        if 'file' in request.files:
            file = request.files['file']
            filename = secure_filename(file.filename)
            upload_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], 'context_sources')
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, filename)
            file.save(file_path)
            source_info['filename'] = filename
            source_info['path'] = file_path
        
        if source_info and source_info not in sources:
            sources.append(source_info)
            save_config('sources', {'sources': sources})
            
        return jsonify({'success': True, 'source': source_info})
    
    elif request.method == 'DELETE':
        data = request.get_json()
        sources = config.get('sources', [])
        source_to_remove = None
        
        for source in sources:
            if data.get('source') in (source.get('url'), source.get('filename')):
                source_to_remove = source
                break
                
        if source_to_remove:
            sources.remove(source_to_remove)
            # Delete the file if it exists
            if 'path' in source_to_remove:
                try:
                    os.remove(source_to_remove['path'])
                except OSError:
                    pass
            save_config('sources', {'sources': sources})
            
        return jsonify({'success': True})

@settings_routes.route('/api/settings/visualization', methods=['GET', 'POST'])
def handle_visualization():
    config = load_config('visualization')
    
    if request.method == 'GET':
        return jsonify({
            'preferred_charts': config.get('preferred_charts', []),
            'default_format': config.get('default_format', 'html')
        })
    
    elif request.method == 'POST':
        data = request.get_json()
        save_config('visualization', {
            'preferred_charts': data.get('preferred_charts', []),
            'default_format': data.get('default_format', 'html')
        })
        return jsonify({'success': True})