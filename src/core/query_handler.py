from src.llm.client import LLMClient
from src.db.connection import get_db_connection
from src.db.schema_fetcher import SchemaFetcher
from src.rag.manager import RAGManager
from src.rag.enhancer import QueryEnhancer

class QueryHandler:
    def __init__(self):
        self.llm_client = LLMClient()
        self.rag_manager = RAGManager()
        self.query_enhancer = QueryEnhancer(self.rag_manager)

    def generate_sql(self, question):
        """
        Generates a SQL query from a natural language question.
        """
        db_connection = None
        try:
            db_connection = get_db_connection()
            if not db_connection:
                raise Exception("Could not connect to the database.")

            # 1. Get the database schema as a string
            schema_fetcher = SchemaFetcher(db_connection)
            schema_prompt = schema_fetcher.get_schema_prompt()

            # 2. Enhance the question with RAG context
            enhanced = self.query_enhancer.enhance_query_context(question)
            
            # 3. Construct the final prompt
            full_prompt = (
                "You are an expert on the OpenMRS database schema. "
                f"The database schema is as follows:\n{schema_prompt}\n\n"
                f"{enhanced['enhanced_prompt']}\n\n"
                "Only return the SQL query, with no additional text, explanation, or formatting."
            )

            # 4. Generate the SQL query
            sql_query = self.llm_client.generate_sql(full_prompt)
            if not sql_query or "Error:" in sql_query:
                raise Exception(sql_query or "Failed to generate SQL query.")

            return sql_query.strip()

        except Exception as e:
            print(f"Error generating SQL: {e}")
            raise
        finally:
            if db_connection and db_connection.is_connected():
                db_connection.close()

    def execute_sql(self, sql_query):
        """
        Executes a SQL query and returns the results.
        """
        db_connection = None
        try:
            db_connection = get_db_connection()
            if not db_connection:
                raise Exception("Could not connect to the database.")

            cursor = db_connection.cursor(dictionary=True)
            cursor.execute(sql_query)
            result = cursor.fetchall()
            
            return result

        except Exception as e:
            print(f"Error executing SQL: {e}")
            raise
        finally:
            if db_connection and db_connection.is_connected():
                cursor.close()
                db_connection.close()