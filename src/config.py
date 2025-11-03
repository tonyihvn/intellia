# config.py

import os
import json
import logging
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Get the configuration directory
    CONFIG_DIR = os.path.join(os.path.dirname(__file__), 'config')
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR)

    # Configuration files
    CONFIG_FILES = {
        'database': 'db_config.json',
        'history': 'query_history.json',
        'guiders': 'guiders.json',
        'datasources': 'datasources.json',
        'visualizations': 'vis_config.json',
        'contexts': 'context_sources.json',
        'llm': 'llm_config.json'
    }
    
    # Initialize config file paths
    DB_CONFIG_FILE = os.path.join(CONFIG_DIR, CONFIG_FILES['database'])
    HISTORY_FILE = os.path.join(CONFIG_DIR, CONFIG_FILES['history'])
    GUIDERS_FILE = os.path.join(CONFIG_DIR, CONFIG_FILES['guiders'])
    DATASOURCES_FILE = os.path.join(CONFIG_DIR, CONFIG_FILES['datasources'])
    VIS_CONFIG_FILE = os.path.join(CONFIG_DIR, CONFIG_FILES['visualizations'])
    CONTEXT_FILE = os.path.join(CONFIG_DIR, CONFIG_FILES['contexts'])

    LLM_CONFIG_FILE = os.path.join(CONFIG_DIR, CONFIG_FILES['llm'])

    # Scheduler file
    SCHEDULES_FILE = os.path.join(CONFIG_DIR, 'schedules.json')

    # Email settings (can be provided via environment variables)
    EMAIL_SETTINGS = {
        'host': os.getenv('SMTP_HOST'),
        'port': int(os.getenv('SMTP_PORT', '587')),
        'username': os.getenv('SMTP_USER'),
        'password': os.getenv('SMTP_PASS'),
        'use_tls': os.getenv('SMTP_TLS', 'true').lower() in ('1','true','yes'),
        'from_address': os.getenv('SMTP_FROM')
    }



    @classmethod
    def get_db_config(cls):
        if os.path.exists(cls.DB_CONFIG_FILE):
            with open(cls.DB_CONFIG_FILE, 'r') as f:
                return json.load(f)
        return {
            'host': 'localhost',
            'user': 'root',
            'password': '',
            'database': 'openmrs',
            'port': 3306
        }

    @classmethod
    def get_query_history(cls):
        """Get the query history from file."""
        try:
            os.makedirs(os.path.dirname(cls.HISTORY_FILE), exist_ok=True)
            if os.path.exists(cls.HISTORY_FILE):
                with open(cls.HISTORY_FILE, 'r') as f:
                    try:
                        history = json.load(f)
                        if not isinstance(history, list):
                            logging.warning("History file corrupted, resetting")
                            return []
                        return history
                    except json.JSONDecodeError:
                        logging.warning("History file corrupted, resetting")
                        return []
            return []
        except Exception as e:
            logging.error(f"Error reading history file: {e}")
            return []

    @classmethod
    def save_query_history(cls, history):
        """Save the query history to file."""
        try:
            if not isinstance(history, list):
                raise ValueError("History must be a list")
            os.makedirs(os.path.dirname(cls.HISTORY_FILE), exist_ok=True)
            with open(cls.HISTORY_FILE, 'w') as f:
                json.dump(history, f, indent=2)
            return True
        except Exception as e:
            logging.error(f"Error saving history file: {e}")
            return False

    @classmethod
    def save_db_config(cls, config):
        with open(cls.DB_CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)

    @classmethod
    def get_llm_config(cls):
        """Get LLM configuration with providers nested under a 'providers' key."""
        # Refresh environment variables
        load_dotenv()
        
        # Initialize config with structure
        config = {'providers': {}}
        
        try:
            # First, get the saved config if it exists
            saved_config = {}
            if os.path.exists(cls.LLM_CONFIG_FILE):
                with open(cls.LLM_CONFIG_FILE, 'r') as f:
                    saved_config = json.load(f)
                    if not isinstance(saved_config, dict) or 'providers' not in saved_config:
                        saved_config = {'providers': {}}
            
            # For each provider in our default config
            for provider, default_settings in cls.LLM_PROVIDERS.items():
                config['providers'][provider] = default_settings.copy()
                
                # Get the environment variable key name
                env_key = f"{provider.upper()}_API_KEY"
                
                # Update with any saved settings
                if provider in saved_config.get('providers', {}):
                    config['providers'][provider].update(saved_config['providers'][provider])
                
                # Override with environment variables if they exist
                env_api_key = os.getenv(env_key)
                if env_api_key:
                    config['providers'][provider]['api_key'] = env_api_key
                    # If we have an API key, ensure the provider is enabled unless explicitly disabled
                    if 'enabled' not in config['providers'][provider]:
                        config['providers'][provider]['enabled'] = True
                
                # Ensure model and API URL are preserved
                config['providers'][provider]['api_url'] = cls.LLM_PROVIDERS[provider]['api_url']
                if not config['providers'][provider].get('model'):
                    config['providers'][provider]['model'] = cls.LLM_PROVIDERS[provider]['model']
                
            return config
            
        except Exception as e:
            print(f"Error loading LLM config: {str(e)}, using defaults")
            return {'providers': cls.LLM_PROVIDERS.copy()}
        
    @classmethod
    def save_llm_config(cls, config):
        """Save LLM configuration to file."""
        try:
            # Ensure the config directory exists
            os.makedirs(os.path.dirname(cls.LLM_CONFIG_FILE), exist_ok=True)
            
            # Ensure we have the right structure
            if not isinstance(config, dict):
                config = {'providers': config}
            elif 'providers' not in config:
                config = {'providers': config}
            
            # Merge with defaults to ensure required structure
            current_config = cls.get_llm_config()
            # Overwrite only explicit fields and preserve api_url defaults unless explicitly provided
            for provider, settings in config['providers'].items():
                if provider not in current_config['providers']:
                    current_config['providers'][provider] = {}
                for key, value in settings.items():
                    if key == 'api_url' and not value:
                        continue
                    current_config['providers'][provider][key] = value
                    
            # Save merged config
            with open(cls.LLM_CONFIG_FILE, 'w') as f:
                json.dump(current_config, f, indent=4)
                
            return True
            
        except Exception as e:
            print(f"Error saving LLM config: {str(e)}")
            return False



    # LLM Providers Configuration
    LLM_PROVIDERS = {
        'openai': {
            'priority': 1,
            # API keys and sensitive values should be provided via environment variables
            'api_key': os.getenv('OPENAI_API_KEY', ''),
            'api_url': 'https://api.openai.com/v1/chat/completions',
            'model': 'gpt-3.5-turbo'
        },
        'google': {
            'priority': 2,
            'api_key': os.getenv('GOOGLE_API_KEY', ''),
            'api_url': 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent',
            'model': 'gemini-2.0-flash'
        },        
        'vertex': {
            'priority': 3,
            'api_key': os.getenv('VERTEX_API_KEY', ''),
            'api_url': os.getenv('VERTEX_API_URL', 'https://us-central1-aiplatform.googleapis.com/v1/projects/PROJECT/locations/LOCATION/publishers/google/models/gemini-1.5-flash:streamGenerateContent'),
            'model': os.getenv('VERTEX_MODEL', 'gemini-1.5-flash')
        },
        'anthropic': {
            'priority': 4,
            'api_key': os.getenv('ANTHROPIC_API_KEY', ''),
            'api_url': 'https://api.anthropic.com/v1/messages',
            'model': 'claude-3-opus-20240229'
        },
        'openrouter': {
            'priority': 5,
            'api_key': os.getenv('OPENROUTER_API_KEY', ''),
            'api_url': 'https://openrouter.ai/api/v1/chat/completions',
            'model': os.getenv('OPENROUTER_MODEL', 'openrouter/auto')
        },
        'cohere': {
            'priority': 6,
            'api_key': os.getenv('COHERE_API_KEY', ''),
            'api_url': 'https://api.cohere.ai/v1/chat',
            'model': os.getenv('COHERE_MODEL', 'command-r')
        },
        'grok': {
            'priority': 7,
            'api_key': os.getenv('GROK_API_KEY', ''),
            'api_url': 'https://api.x.ai/v1/chat/completions',
            'model': os.getenv('GROK_MODEL', 'grok-beta')
        },
        'github': {
            'priority': 8,
            'api_key': os.getenv('GITHUB_MODELS_API_KEY', ''),
            'api_url': 'https://models.inference.ai.azure.com/chat/completions',
            'model': os.getenv('GITHUB_MODEL', 'gpt-4o-mini')
        },
        'local': {
            'priority': 9,
            'models': [os.getenv('LLM_MODEL', 'codellama'), 'llama3', 'deepseek-coder'],
            'api_url': 'http://localhost:11434'
        }
    }

    # Output and Visualization Configuration
    OUTPUT_FORMATS = {
        'document': ['pdf', 'docx', 'txt'],
        'spreadsheet': ['xlsx', 'csv'],
        'visualization': ['png', 'svg', 'html'],
        'data': ['json', 'xml']
    }

    VISUALIZATION_TYPES = {
        'charts': ['bar', 'line', 'pie', 'scatter', 'heatmap'],
        'tables': ['basic', 'pivot', 'summary'],
        'infographics': ['timeline', 'comparison', 'distribution']
    }

    @classmethod
    def get_guiders(cls):
        """Get system guiders/training data"""
        if os.path.exists(cls.GUIDERS_FILE):
            with open(cls.GUIDERS_FILE, 'r') as f:
                return json.load(f)
        return {}

    @classmethod
    def save_guiders(cls, guiders):
        """Save system guiders/training data"""
        with open(cls.GUIDERS_FILE, 'w') as f:
            json.dump(guiders, f, indent=4)

    @classmethod
    def get_context_sources(cls):
        """Get additional context sources (URLs, docs, etc)"""
        if os.path.exists(cls.CONTEXT_FILE):
            with open(cls.CONTEXT_FILE, 'r') as f:
                return json.load(f)
        return {'urls': [], 'documents': []}

    @classmethod
    def save_context_sources(cls, sources):
        """Save context sources"""
        with open(cls.CONTEXT_FILE, 'w') as f:
            json.dump(sources, f, indent=4)

    @classmethod
    def get_visualization_config(cls):
        """Get visualization preferences"""
        if os.path.exists(cls.VIS_CONFIG_FILE):
            with open(cls.VIS_CONFIG_FILE, 'r') as f:
                return json.load(f)
        return {
            'default_format': 'html',
            'preferred_charts': ['bar', 'line'],
            'color_scheme': 'default'
        }

    @classmethod
    def save_visualization_config(cls, config):
        """Save visualization preferences"""
        with open(cls.VIS_CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)

    # Flask configuration
    SECRET_KEY = os.getenv('SECRET_KEY')
    # Instant execution flag (default false). Set env var INSTANT_EXECUTION=1 to enable.
    INSTANT_EXECUTION = os.getenv('INSTANT_EXECUTION', 'false').lower() in ('1', 'true', 'yes')