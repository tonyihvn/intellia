# config.py

import os
import json

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
        'contexts': 'context_sources.json'
    }
    
    # Initialize config file paths
    DB_CONFIG_FILE = os.path.join(CONFIG_DIR, CONFIG_FILES['database'])
    HISTORY_FILE = os.path.join(CONFIG_DIR, CONFIG_FILES['history'])
    GUIDERS_FILE = os.path.join(CONFIG_DIR, CONFIG_FILES['guiders'])
    DATASOURCES_FILE = os.path.join(CONFIG_DIR, CONFIG_FILES['datasources'])
    VIS_CONFIG_FILE = os.path.join(CONFIG_DIR, CONFIG_FILES['visualizations'])
    CONTEXT_FILE = os.path.join(CONFIG_DIR, CONFIG_FILES['contexts'])

    @classmethod
    def get_db_config(cls):
        if os.path.exists(cls.DB_CONFIG_FILE):
            with open(cls.DB_CONFIG_FILE, 'r') as f:
                return json.load(f)
        return {
            'host': 'localhost',
            'user': 'root',
            'password': '',
            'database': 'openmrsmcp',
            'port': 3306
        }

    @classmethod
    def save_db_config(cls, config):
        with open(cls.DB_CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)

    @classmethod
    def get_query_history(cls):
        if os.path.exists(cls.HISTORY_FILE):
            with open(cls.HISTORY_FILE, 'r') as f:
                return json.load(f)
        return []

    @classmethod
    def save_query_history(cls, history):
        with open(cls.HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=4)

    # LLM Providers Configuration
    LLM_PROVIDERS = {
        'google': {
            'priority': 1,
            'api_key': os.getenv('GOOGLE_API_KEY', ''),
            'api_url': 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent',
            'model': 'gemini-2.0-flash'
        },
        'openai': {
            'priority': 2,
            'api_key': os.getenv('OPENAI_API_KEY', ''),
            'api_url': 'https://api.openai.com/v1/chat/completions',
            'model': 'gpt-3.5-turbo'
        },
        'anthropic': {
            'priority': 3,
            'api_key': os.getenv('ANTHROPIC_API_KEY', ''),
            'api_url': 'https://api.anthropic.com/v1/messages',
            'model': 'claude-3-opus-20240229'
        },
        'local': {
            'priority': 4,
            'models': [os.getenv('LLM_MODEL', 'codellama'), 'llama3:latest'],
            'api_url': 'http://localhost:11434/api/generate'
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
    SECRET_KEY = os.getenv('SECRET_KEY', 'your_secret_key_here')