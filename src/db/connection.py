import mysql.connector
from ..config import Config

def get_db_connection():
    """
    Establishes a connection to the database using settings from the config.
    """
    try:
        # Get the current database configuration
        db_config = Config.get_db_config()
        connection = mysql.connector.connect(**db_config)
        
        if connection.is_connected():
            return connection
    except mysql.connector.Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None
    return None
