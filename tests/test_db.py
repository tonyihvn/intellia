import pytest
from src.db.connection import get_db_connection
from src.db.schema import get_schema_info

def test_database_connection():
    # Test if the database connection can be established
    connection = get_db_connection()
    assert connection is not None
    connection.close()

def test_schema_info():
    # Test if the schema information can be retrieved
    schema_info = get_schema_info()
    assert isinstance(schema_info, dict)  # Assuming schema info is returned as a dictionary
    assert 'tables' in schema_info  # Check if 'tables' key exists in the schema info