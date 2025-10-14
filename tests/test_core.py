import pytest
from src.core.query_handler import QueryHandler

def test_query_handler_initialization():
    query_handler = QueryHandler()
    assert query_handler is not None

def test_process_natural_language_query():
    query_handler = QueryHandler()
    natural_language_query = "What are the patients' names?"
    sql_query = query_handler.process_natural_language_query(natural_language_query)
    assert sql_query is not None
    assert "SELECT" in sql_query  # Basic check to see if it's a SQL query

def test_query_execution():
    query_handler = QueryHandler()
    natural_language_query = "How many patients are there?"
    sql_query = query_handler.process_natural_language_query(natural_language_query)
    result = query_handler.execute_query(sql_query)
    assert result is not None  # Ensure that the result is not None
    assert isinstance(result, list)  # Assuming the result should be a list of records

def test_invalid_query():
    query_handler = QueryHandler()
    natural_language_query = "This is an invalid query."
    sql_query = query_handler.process_natural_language_query(natural_language_query)
    assert sql_query is None  # Ensure that invalid queries return None

def test_query_handler_with_database():
    query_handler = QueryHandler()
    natural_language_query = "List all patients."
    sql_query = query_handler.process_natural_language_query(natural_language_query)
    result = query_handler.execute_query(sql_query)
    assert len(result) > 0  # Ensure that there are results returned from the database