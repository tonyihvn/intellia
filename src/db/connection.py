import mysql.connector
import logging
from ..config import Config

def get_db_connection():
    """
    Establishes a connection to the database using settings from the config.
    """
    try:
        # Get the current database configuration
        config = Config()
        db_config = config.get_db_config()
        
        if not db_config:
            logging.error("No database configuration found")
            return None
            
        connection = mysql.connector.connect(**db_config)
        
        if connection.is_connected():
            logging.info("Successfully connected to the database")
            return connection
        else:
            logging.error("Failed to connect to database - connection not established")
            return None
            
    except mysql.connector.Error as e:
        logging.error(f"Error connecting to MySQL: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error while connecting to database: {e}")
        return None
