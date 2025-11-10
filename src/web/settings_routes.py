from flask import Blueprint, render_template, request, jsonify, current_app
import json
import os
import requests
from dotenv import load_dotenv, find_dotenv, set_key
from werkzeug.utils import secure_filename
from ..config import Config
from ..llm.client import LLMClient
from ..rag.manager import RAGManager

# Single RAG manager instance for settings operations (uses on-disk vector store)
rag_manager = RAGManager()

# Load environment variables
env_path = find_dotenv()
if not env_path:
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(env_path)

settings_routes = Blueprint('settings', __name__, url_prefix='/api')

def get_config_path(config_type):
    """Get the path to a configuration file"""
    config_dir = current_app.config.get('CONFIG_DIR', Config.CONFIG_DIR)
    os.makedirs(config_dir, exist_ok=True)
    return os.path.join(config_dir, f'{config_type}.json')

def load_config(config_type):
    """Load configuration from a JSON file"""
    config_path = get_config_path(config_type)
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback: look in framework default config dir for legacy files
        legacy_path = os.path.join(Config.CONFIG_DIR, f'{config_type}.json')
        try:
            with open(legacy_path, 'r') as f:
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
    config = Config.get_llm_config()
    return render_template('settings.html', config=config)

@settings_routes.route('/db')
def db_config_page():
    config = Config.get_db_config()
    return render_template('db_config.html', config=config)

@settings_routes.route('/llm/config', methods=['GET', 'POST'])
def handle_llm_config():
    if request.method == 'GET':
        config = Config.get_llm_config()
        return jsonify(config), 200
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        provider = data.get('provider')
        # Accept any known provider keys from Config
        allowed = list(Config.LLM_PROVIDERS.keys())
        if not provider or provider not in allowed:
            return jsonify({'error': 'Invalid provider'}), 400
        
        # Get current config
        config = Config.get_llm_config()
        
        # Ensure providers dict exists
        if 'providers' not in config:
            config['providers'] = {}
            
        # Ensure provider dict exists
        if provider not in config['providers']:
            config['providers'][provider] = {}
        
        # Update the specific provider's settings
        provider_config = data.get('config', {})
        if not provider_config:
            return jsonify({'error': 'No configuration provided'}), 400
        
        config['providers'][provider].update(provider_config)
        
        # Save API key to environment file if provided
        if 'api_key' in provider_config and provider_config['api_key']:
            env_var_name = f"{provider.upper()}_API_KEY"
            try:
                # Ensure .env file exists
                if not os.path.exists(env_path):
                    with open(env_path, 'w') as f:
                        f.write('# Environment Variables\n')
                        
                # Update .env file
                set_key(env_path, env_var_name, provider_config['api_key'])
                os.environ[env_var_name] = provider_config['api_key']  # Update current environment
                print(f"Updated {env_var_name} in .env file")
            except Exception as e:
                print(f"Error updating environment variable: {str(e)}")
        
        # Save the updated config
        if Config.save_llm_config(config):
            return jsonify({
                'success': True, 
                'message': f'{provider.title()} settings saved successfully'
            }), 200
        else:
            return jsonify({'error': 'Failed to save configuration'}), 500
            
    except Exception as e:
        print(f"Error in handle_llm_config: {str(e)}")
        payload = Config.client_error_payload(str(e), e)
        return jsonify(payload), 500

@settings_routes.route('/llm/providers', methods=['GET', 'POST'])
def manage_providers():
    """Get or set provider enable flags and priority order."""
    if request.method == 'GET':
        # Ensure environment variables are loaded
        load_dotenv()
        
        cfg = Config.get_llm_config()
        providers = cfg.get('providers', {})
        # Normalize shape with enabled and priority
        out = {}
        for name, p in providers.items():
            # Check environment variable first for API key
            env_api_key = os.getenv(f"{name.upper()}_API_KEY", '')
            api_key = env_api_key or p.get('api_key', '')
            
            out[name] = {
                'enabled': p.get('enabled', True if api_key else False),
                'priority': p.get('priority', 99),
                'model': p.get('model', ''),
                'api_url': p.get('api_url', ''),
                'api_key': api_key
            }
        # Augment with local Ollama model information when available
        try:
            # Only add detailed model list for the 'local' provider
            if 'local' in providers:
                # Reuse the existing route logic to detect installed models
                resp = requests.get('http://localhost:11434/api/tags', timeout=3)
                installed_tags = []
                if resp.status_code == 200:
                    installed_tags = [m.get('name') for m in resp.json().get('models', []) if m.get('name')]

                installed_basenames = set([name.split(':', 1)[0] for name in installed_tags])

                llm_cfg = Config.get_llm_config()
                priority = llm_cfg.get('providers', {}).get('local', {}).get('models', [])

                combined = []
                seen = set()

                def add_model_entry(model_name):
                    base = model_name.split(':', 1)[0]
                    installed = (model_name in installed_tags) or (base in installed_basenames)
                    key = model_name
                    if key in seen:
                        return
                    seen.add(key)
                    combined.append({'name': model_name, 'installed': installed})

                for name in priority:
                    add_model_entry(name)
                for tag in installed_tags:
                    add_model_entry(tag)

                out['local'].update({'models': combined, 'running': True if installed_tags else False, 'models_priority': priority})
        except Exception:
            # If Ollama isn't reachable, still return provider entry without models
            out.get('local', {}).update({'models': [], 'running': False, 'models_priority': []})

        return jsonify(out)

    # POST to update
    data = request.get_json()
    if not data or 'providers' not in data:
        return jsonify({'error': 'providers object required'}), 400

    cfg = Config.get_llm_config()
    changed = False
    for name, p in data['providers'].items():
        if name not in cfg['providers']:
            continue
        if 'enabled' in p:
            cfg['providers'][name]['enabled'] = bool(p['enabled'])
            changed = True
        if 'priority' in p:
            cfg['providers'][name]['priority'] = int(p['priority'])
            changed = True
        if 'api_key' in p:
            cfg['providers'][name]['api_key'] = p['api_key']
            # Also update environment variable
            env_var_name = f"{name.upper()}_API_KEY"
            if os.path.exists(env_path):
                set_key(env_path, env_var_name, p['api_key'])
                os.environ[env_var_name] = p['api_key']
            changed = True
        if 'model' in p:
            cfg['providers'][name]['model'] = p['model']
            changed = True
        # allow updating local models priority list
        if name == 'local' and 'models' in p and isinstance(p['models'], list):
            # store the list of model names in the local provider config
            cfg['providers'][name]['models'] = p['models']
            changed = True

    ok = Config.save_llm_config(cfg)
    if ok:
        return jsonify({'success': True})
    return jsonify({'error': 'Failed to save provider settings'}), 500

@settings_routes.route('/llm/test', methods=['POST'])
def test_llm_connection():
    data = request.get_json()
    provider = data.get('provider')
    
    if not provider:
        return jsonify({'error': 'Provider not specified'}), 400
        
    try:
        client = LLMClient()
        test_prompt = "Generate a simple SELECT query to test the connection."
        result = client.generate_sql(test_prompt)
        return jsonify({
            'success': True,
            'message': f'Successfully connected to {provider}'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@settings_routes.route('/llm/ollama/status', methods=['GET'])
def check_ollama_status():
    try:
        response = requests.get('http://localhost:11434/api/tags')
        if response.status_code == 200:
            models = response.json().get('models', [])
            return jsonify({
                'running': True,
                'models': [model['name'] for model in models]
            })
    except:
        pass
    
    return jsonify({
        'running': False,
        'models': []
    })

@settings_routes.route('/llm/ollama/models', methods=['GET'])
def list_ollama_models():
    """List local Ollama models as objects { name, installed } and priority order."""
    try:
        # Discover installed models from Ollama
        resp = requests.get('http://localhost:11434/api/tags', timeout=5)
        installed_tags = []
        if resp.status_code == 200:
            installed_tags = [m.get('name') for m in resp.json().get('models', []) if m.get('name')]

        installed_basenames = set([name.split(':', 1)[0] for name in installed_tags])

        # Priority order from config (fallback to defaults)
        llm_cfg = Config.get_llm_config()
        priority = llm_cfg.get('providers', {}).get('local', {}).get('models', [])

        # Build model objects: union of priority and installed
        combined = []
        seen = set()

        def add_model_entry(model_name):
            base = model_name.split(':', 1)[0]
            installed = (model_name in installed_tags) or (base in installed_basenames)
            key = model_name
            if key in seen:
                return
            seen.add(key)
            combined.append({'name': model_name, 'installed': installed})

        for name in priority:
            add_model_entry(name)
        for tag in installed_tags:
            add_model_entry(tag)

        return jsonify({
            'running': True if installed_tags else False,
            'models': combined,
            'priority': priority
        }), 200
    except Exception:
        # Return empty but valid payload to avoid UI errors
        llm_cfg = Config.get_llm_config()
        priority = llm_cfg.get('providers', {}).get('local', {}).get('models', [])
        return jsonify({
            'running': False,
            'models': [],
            'priority': priority
        }), 200

@settings_routes.route('/llm/ollama/pull', methods=['POST'])
def pull_ollama_models():
    """Pull missing priority models from Ollama."""
    try:
        # Current installed
        tags_resp = requests.get('http://localhost:11434/api/tags', timeout=5)
        installed_tags = []
        if tags_resp.status_code == 200:
            installed_tags = [m.get('name') for m in tags_resp.json().get('models', []) if m.get('name')]
        installed_basenames = set([name.split(':', 1)[0] for name in installed_tags])

        # Priority list
        llm_cfg = Config.get_llm_config()
        priority = llm_cfg.get('providers', {}).get('local', {}).get('models', [])

        pulled = []
        skipped = []
        errors = []

        for name in priority:
            base = name.split(':', 1)[0]
            if (name in installed_tags) or (base in installed_basenames):
                skipped.append(name)
                continue
            try:
                resp = requests.post('http://localhost:11434/api/pull', json={'name': name}, timeout=600)
                if resp.status_code == 200:
                    pulled.append(name)
                else:
                    errors.append({'model': name, 'status': resp.status_code})
            except Exception as e:
                errors.append({'model': name, 'error': str(e)})

        success = len(errors) == 0
        return jsonify({'success': success, 'pulled': pulled, 'skipped': skipped, 'errors': errors}), (200 if success else 207)
    except Exception as e:
        payload = Config.client_error_payload(str(e), e)
        return jsonify(payload), 500

@settings_routes.route('/settings/guiders', methods=['GET', 'POST', 'DELETE'])
def handle_guiders():
    config = load_config('guiders')
    
    if request.method == 'GET':
        return jsonify({'guiders': config.get('guiders', [])})
    
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

@settings_routes.route('/settings/sources', methods=['GET', 'POST', 'DELETE'])
def handle_sources():
    config = load_config('sources')
    
    if request.method == 'GET':
        return jsonify({'sources': config.get('sources', [])})
    
    elif request.method == 'POST':
        sources = config.get('sources', [])
        source_info = {}
        
        # Handle both form data and JSON
        if request.is_json:
            data = request.get_json()
            if data.get('url'):
                source_info['url'] = data['url']
                source_info['type'] = 'url'
        else:
            if 'url' in request.form:
                source_info['url'] = request.form['url']
                source_info['type'] = 'url'
            
            if 'file' in request.files:
                file = request.files['file']
                if file.filename:
                    filename = secure_filename(file.filename)
                    upload_dir = os.path.join(current_app.static_folder, 'uploads', 'context_sources')
                    os.makedirs(upload_dir, exist_ok=True)
                    file_path = os.path.join(upload_dir, filename)
                    file.save(file_path)
                    source_info['filename'] = filename
                    source_info['path'] = file_path
                    source_info['type'] = 'file'
        
        # Validate source info
        if not source_info:
            return jsonify({'error': 'No valid source provided'}), 400
            
        # Check for duplicates
        is_duplicate = False
        for existing in sources:
            if (('url' in source_info and source_info['url'] == existing.get('url')) or
                ('filename' in source_info and source_info['filename'] == existing.get('filename'))):
                is_duplicate = True
                break
                
        if not is_duplicate:
            sources.append(source_info)
            try:
                save_config('sources', {'sources': sources})
                return jsonify({'success': True, 'source': source_info})
            except Exception as e:
                payload = Config.client_error_payload(f'Failed to save source: {str(e)}', e)
                return jsonify(payload), 500
        else:
            return jsonify({'error': 'Source already exists'}), 400
    
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

@settings_routes.route('/settings/visualization', methods=['GET', 'POST'])
def handle_visualization():
    config = load_config('visualization')
    
    if request.method == 'GET':
        return jsonify({
            'preferred_charts': config.get('preferred_charts', []),
            'default_format': config.get('default_format', 'html'),
            'rag_enabled': config.get('rag_enabled', True),
            'rag_max_tables': config.get('rag_max_tables', 3),
            'rag_max_rules': config.get('rag_max_rules', 5)
        })
    
    elif request.method == 'POST':
        data = request.get_json()
        save_config('visualization', {
            'preferred_charts': data.get('preferred_charts', []),
            'default_format': data.get('default_format', 'html'),
            'rag_enabled': bool(data.get('rag_enabled', True)),
            'rag_max_tables': int(data.get('rag_max_tables', 3)),
            'rag_max_rules': int(data.get('rag_max_rules', 5))
        })
        return jsonify({'success': True})


@settings_routes.route('/settings/examples', methods=['GET', 'POST', 'DELETE'])
def handle_examples():
    """Get or set example commands shown in the command UI."""
    # Use the RAG examples collection as source of truth
    if request.method == 'GET':
        try:
            docs = rag_manager.get_all_examples()
            examples = [d.get('text') for d in docs]
            return jsonify({'examples': examples})
        except Exception as e:
            payload = Config.client_error_payload(str(e), e)
            # include examples fallback plus error payload
            out = {'examples': []}
            out.update(payload)
            return jsonify(out), 500

    if request.method == 'POST':
        data = request.get_json() or {}
        new_example = data.get('example')
        if not new_example:
            return jsonify({'error': 'example is required'}), 400

        # Persist into vector store with optional metadata (featured / auto_run)
        try:
            featured = bool(data.get('featured', False))
            auto_run = bool(data.get('auto_run', False))
            metadata = {'source': 'examples'}
            if featured:
                metadata['featured'] = True
            if auto_run:
                metadata['auto_run'] = True
            success = rag_manager.add_examples([{'text': new_example, 'metadata': metadata}])
            if success:
                return jsonify({'success': True})
            return jsonify({'error': 'Failed to persist example'}), 500
        except Exception as e:
            return jsonify(Config.client_error_payload(str(e), e)), 500

    if request.method == 'DELETE':
        data = request.get_json() or {}
        example = data.get('example')
        if not example:
            return jsonify({'error': 'example is required'}), 400
        try:
            ok = rag_manager.delete_example(example)
            if ok:
                return jsonify({'success': True})
            return jsonify({'error': 'Failed to delete example'}), 500
        except Exception as e:
            return jsonify(Config.client_error_payload(str(e), e)), 500


@settings_routes.route('/settings/app', methods=['GET', 'POST'])
def handle_app_settings():
    """Get or save simple application settings (auto-send emails, persist password, organization)."""
    if request.method == 'GET':
        try:
            cfg = load_config('app') or {}
            return jsonify(cfg)
        except Exception as e:
            payload = Config.client_error_payload(str(e), e)
            return jsonify(payload), 500

    if request.method == 'POST':
        try:
            data = request.get_json() or {}
            # Only accept simple keys
            allowed_keys = ['auto_send_emails', 'persist_smtp_password', 'organization_name']
            out = {}
            for k in allowed_keys:
                if k in data:
                    out[k] = data[k]
            save_config('app', out)
            return jsonify({'success': True})
        except Exception as e:
            payload = Config.client_error_payload(str(e), e)
            return jsonify(payload), 500