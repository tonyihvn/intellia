import os
import json
import logging
from datetime import datetime
import uuid
from ..config import Config

class HistoryManager:
    @staticmethod
    def load_history():
        """Load and clean the history file."""
        try:
            history = Config.get_query_history()
            if not isinstance(history, list):
                return []
                
            # Clean and validate each entry
            cleaned_history = []
            for item in history:
                if isinstance(item, dict) and 'question' in item:
                    if 'id' not in item:
                        item['id'] = str(uuid.uuid4())
                    if 'timestamp' not in item:
                        item['timestamp'] = datetime.now().isoformat()
                    cleaned_history.append(item)
                    
            # Sort by timestamp descending
            cleaned_history.sort(key=lambda x: x['timestamp'], reverse=True)
            return cleaned_history
            
        except Exception as e:
            logging.error(f"Error loading history: {str(e)}")
            return []
            
    @staticmethod
    def save_history(history):
        """Save history to file with validation."""
        try:
            if not isinstance(history, list):
                return False
                
            # Ensure all items are valid
            cleaned_history = []
            for item in history:
                if isinstance(item, dict) and 'question' in item:
                    if 'id' not in item:
                        item['id'] = str(uuid.uuid4())
                    if 'timestamp' not in item:
                        item['timestamp'] = datetime.now().isoformat()
                    cleaned_history.append(item)
                    
            return Config.save_query_history(cleaned_history[:50])  # Keep last 50
            
        except Exception as e:
            logging.error(f"Error saving history: {str(e)}")
            return False
            
    @staticmethod
    def clear_history():
        """Clear all history."""
        try:
            if os.path.exists(Config.HISTORY_FILE):
                os.remove(Config.HISTORY_FILE)
            return Config.save_query_history([])
        except Exception as e:
            logging.error(f"Error clearing history: {str(e)}")
            return False
            
    @staticmethod
    def add_item(question, sql=None, result=None, status='success'):
        """Add a new item to history."""
        try:
            history = HistoryManager.load_history()
            
            new_item = {
                'id': str(uuid.uuid4()),
                'timestamp': datetime.now().isoformat(),
                'question': question,
                'sql': sql or '',
                'result': result or '',
                'status': status
            }
            
            history.insert(0, new_item)
            if HistoryManager.save_history(history):
                return new_item
            return None
            
        except Exception as e:
            logging.error(f"Error adding history item: {str(e)}")
            return None
            
    @staticmethod
    def get_item(item_id):
        """Get a specific history item by ID."""
        try:
            history = HistoryManager.load_history()
            for item in history:
                if str(item.get('id')) == str(item_id):
                    return item
            return None
        except Exception as e:
            logging.error(f"Error getting history item: {str(e)}")
            return None
            
    @staticmethod
    def delete_item(item_id):
        """Delete a specific history item by ID."""
        try:
            history = HistoryManager.load_history()
            new_history = [item for item in history if str(item.get('id')) != str(item_id)]
            if len(new_history) != len(history):
                return HistoryManager.save_history(new_history)
            return False
        except Exception as e:
            logging.error(f"Error deleting history item: {str(e)}")
            return False