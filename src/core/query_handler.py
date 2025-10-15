import logging
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

    def handle_query(self, question):
        """
        Handle a natural language question by generating and executing SQL.
        
        Args:
            question: The natural language question to process
            
        Returns:
            Dictionary containing results and metadata
        """
        logging.info(f"\n{'='*50}\nQuery Handler Processing\n{'='*50}")
        logging.info(f"Original question: {question}")
        try:
            # Generate SQL from natural language
            sql = self.generate_sql(question)
            if not sql:
                raise Exception("Failed to generate SQL query")

            # Execute the generated SQL
            results = self.execute_sql(sql)

            # Ask LLM to summarize/format results per the question intent
            try:
                render_prompt = (
                    "You are a data analyst. Given the user's request and a JSON result set, "
                    "produce a concise, human-friendly summary or formatted output. Keep it short.\n\n"
                    f"Request: {question}\n"
                    f"Result JSON: {results[:20]}"  # limit size to avoid large prompts
                )
                rendered = self.llm_client._try_cloud_generation(render_prompt)
            except Exception:
                rendered = ""

            return {
                "sql": sql,
                "results": results,
                "rendered": rendered
            }

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
            sql_query = self.llm_client.generate_sql(context)
            if not sql_query or "Error:" in sql_query:
                raise Exception(sql_query or "Failed to generate SQL query.")

            return sql_query.strip()

        except Exception as e:
            if db_connection and hasattr(db_connection, 'close'):
                try:
                    db_connection.close()
                except:
                    pass
            raise Exception(f"Error generating SQL: {str(e)}")

    def build_prompt(self, question, db_connection=None):
        """Build the prompt with schema summary and RAG-enhanced context."""
        # 1. Get focused database schema information
        schema_fetcher = SchemaFetcher(db_connection or get_db_connection())
        schema_info = schema_fetcher.get_schema_summary(question)

        # 2. Enhance the question with RAG to get relevant context
        enhanced = self.query_enhancer.enhance_query_context(question)
        enhanced_prompt = enhanced.get('enhanced_prompt', '')

        # 3. Construct a concise prompt
        context = f"""Task: Generate a MySQL query to {question}

Relevant Tables and Fields (schema summary):
{schema_info}

Additional Context from Knowledge Base (RAG):
{enhanced_prompt}

Requirements:
- Use only necessary tables and joins
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