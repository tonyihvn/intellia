import logging
import json
import os
from datetime import datetime
from pathlib import Path
from ..llm.client import LLMClient
from ..db.connection import get_db_connection
from ..db.schema_fetcher import SchemaFetcher
from ..rag.manager import RAGManager
from ..rag.enhancer import QueryEnhancer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class QueryHistory:
    def __init__(self):
        self.history_file = Path("query_history.json")
        self.load_history()

    def load_history(self):
        if self.history_file.exists():
            with open(self.history_file) as f:
                self.history = json.load(f)
        else:
            self.history = []

    def save_history(self):
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)

    def add_query(self, query_data):
        entry = {
            'id': len(self.history) + 1,
            'timestamp': datetime.now().isoformat(),
            'question': query_data['question'],
            'sql': query_data['sql'],
            'status': query_data['status'],
            'results': query_data.get('results', None)
        }
        self.history.append(entry)
        self.save_history()
        return entry['id']

    def get_query(self, query_id):
        return next((q for q in self.history if q['id'] == query_id), None)

    def update_query(self, query_id, updates):
        for i, query in enumerate(self.history):
            if query['id'] == query_id:
                self.history[i].update(updates)
                self.save_history()
                return True
        return False

    def delete_query(self, query_id):
        self.history = [q for q in self.history if q['id'] != query_id]
        self.save_history()

class QueryHandler:
    """
    Handles natural language questions and converts them to SQL queries
    """

    def __init__(self, llm_client, query_enhancer):
        """
        Initialize QueryHandler with LLMClient instance
        """
        self.llm_client = llm_client 
        self.query_enhancer = query_enhancer
        self.history = QueryHistory()

    def preview_query(self, question):
        """Preview SQL query generation without execution"""
        try:
            response = self.generate_sql(question)
            return {
                'preview': True,
                'question': question,
                'sql': response['sql'],
                'explanation': response['explanation'],
                'needs_confirmation': True
            }
        except Exception as e:
            return {
                'preview': True,
                'error': str(e),
                'needs_confirmation': False
            }

    def analyze_error(self, sql, error):
        """Use LLM to analyze SQL errors and suggest fixes"""
        prompt = f"""
        Analyze this SQL query that failed:
        {sql}
        
        Error message:
        {error}
        
        Please explain:
        1. What caused the error
        2. How to fix it
        3. Provide the corrected SQL query
        """
        
        analysis = self.llm_client.generate(prompt)
        return {
            'original_sql': sql,
            'error': str(error),
            'analysis': analysis,
            'needs_confirmation': True
        }

    def handle_query(self, question, execute=True, preview_mode=True):
        """Enhanced query handling with preview and history"""
        if preview_mode:
            return self.preview_query(question)

        try:
            response = self.generate_sql(question)
            result = {
                'sql': response['sql'],
                'explanation': response.get('explanation', ''),
                'status': 'pending'
            }

            # Execute the SQL if requested
            if execute:
                try:
                    results = self.execute_sql(response['sql'])
                    result.update({
                        'results': results,
                        'status': 'success'
                    })
                except Exception as e:
                    analysis = self.analyze_error(response['sql'], str(e))
                    result.update({
                        'status': 'error',
                        'error_analysis': analysis
                    })

            # Save to history
            history_entry = {
                'question': question,
                **result
            }
            query_id = self.history.add_query(history_entry)
            result['query_id'] = query_id

            return result

        except Exception as e:
            raise Exception(f"Error handling query: {str(e)}")

    def generate_sql(self, question):
        """
        Generates a SQL query from a natural language question.
        """
        db_connection = None
        try:
            db_connection = get_db_connection()
            if not db_connection:
                raise Exception("Could not connect to the database.")

            # Build full prompt
            context = self.build_prompt(question, db_connection)
            logging.info(f"Generated context for LLM:\n{context}")

            # Generate the SQL query
            gen = self.llm_client.generate_sql(context)

            # Normalize generator output to a dict with 'sql', 'explanation', 'full_response'
            sql_text = None
            explanation = None
            full_response = None

            if isinstance(gen, dict):
                full_response = gen.get('full_response') or gen.get('sql') or ''
                sql_text = gen.get('sql') or (full_response and self.llm_client._extract_sql(full_response))
                explanation = gen.get('explanation')
            else:
                # gen may be a plain string
                full_response = str(gen)
                # Try to extract SQL from the free text
                try:
                    sql_text = self.llm_client._extract_sql(full_response)
                except Exception:
                    sql_text = full_response
                explanation = full_response.replace(sql_text, '').strip() if sql_text else ''

            if not sql_text:
                raise Exception("Failed to generate SQL query.")

            return {'sql': sql_text.strip(), 'explanation': explanation, 'full_response': full_response}

        except Exception as e:
            if db_connection and hasattr(db_connection, 'close'):
                try:
                    db_connection.close()
                except:
                    pass
            raise Exception(f"Error generating SQL: {str(e)}")

    def build_prompt(self, question, db_connection=None):
        """Build the prompt with schema summary and RAG-enhanced context."""
        # 1. Use RAG-based enhanced context (includes a targeted schema snippet)
        enhanced = self.query_enhancer.enhance_query_context(question)
        enhanced_prompt = enhanced.get('enhanced_prompt', '')

        # 2. Construct a concise prompt using only the enhanced context to limit tokens
        context = f"""Task: Generate a MySQL query to {question}

Context (targeted):
{enhanced_prompt}

Requirements:
- Use only necessary tables and joins from the provided schema context
- Return accurate count/results
- Handle NULL values appropriately"""
        return context

    def execute_sql(self, sql):
        """
        Execute a SQL query and return the results.
        
        Args:
            sql: The SQL query to execute
            
        Returns:
            List of dictionaries containing the query results
        """
        db_connection = None
        cursor = None
        try:
            db_connection = get_db_connection()
            if not db_connection:
                raise Exception("Could not connect to the database.")
            
            cursor = db_connection.cursor(dictionary=True)
            cursor.execute(sql)
            results = cursor.fetchall()
            
            # Convert decimal and datetime objects to strings for JSON serialization
            sanitized_results = []
            for row in results:
                sanitized_row = {}
                for key, value in row.items():
                    if hasattr(value, 'isoformat'):  # datetime objects
                        sanitized_row[key] = value.isoformat()
                    elif hasattr(value, 'normalize'):  # Decimal objects
                        sanitized_row[key] = str(value)
                    else:
                        sanitized_row[key] = value
                sanitized_results.append(sanitized_row)
            
            return sanitized_results

        except Exception as e:
            if cursor:
                try:
                    cursor.close()
                except:
                    pass
            if db_connection and hasattr(db_connection, 'close'):
                try:
                    db_connection.close()
                except:
                    pass
            raise Exception(f"Error executing query: {str(e)}")

    def generate_sql_query(self, question):
        """
        Generates a SQL query from a natural language question.
        Alias for generate_sql for backward compatibility
        """
        return self.generate_sql(question)

    def validate_sql_against_schema(self, sql: str):
        """Basic validation: check that referenced tables exist in the database schema.

        Returns dict: {'ok': True} or {'ok': False, 'missing_tables': [...]}.
        This is intentionally conservative and uses simple token matching for table names.
        """
        try:
            conn = get_db_connection()
            if not conn:
                return {'ok': False, 'error': 'Could not connect to database for validation'}

            schema_fetcher = SchemaFetcher(conn)
            # Get list of tables in DB
            cursor = conn.cursor()
            cursor.execute("SHOW TABLES")
            tables = [row[0] for row in cursor.fetchall()]
            cursor.close()

            # Simple parser: look for FROM/JOIN `table` or FROM/JOIN table patterns
            import re
            found = set()
            for m in re.finditer(r"(?:from|join)\s+[`']?([a-zA-Z0-9_]+)[`']?", sql, flags=re.I):
                found.add(m.group(1))

            missing = [t for t in found if t not in tables]
            if missing:
                return {'ok': False, 'missing_tables': missing, 'found': list(found), 'tables': tables}

            return {'ok': True}
        except Exception as e:
            return {'ok': False, 'error': str(e)}