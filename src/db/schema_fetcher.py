import mysql.connector

class SchemaFetcher:
    def __init__(self, db_connection):
        self.connection = db_connection

    def get_schema_prompt(self):
        """
        Fetches the database schema and formats it as a string for an LLM prompt.
        """
        if not self.connection or not self.connection.is_connected():
            return ""
            
        schema = {}
        cursor = self.connection.cursor()
        
        try:
            # Get all table names
            cursor.execute("SHOW TABLES")
            tables = [table[0] for table in cursor.fetchall()]
            
            # For each table, get its columns
            for table_name in tables:
                cursor.execute(f"DESCRIBE `{table_name}`") # Use backticks for safety
                columns = [row[0] for row in cursor.fetchall()]
                schema[table_name] = columns
        finally:
            cursor.close()

        # Format the schema into a string for the prompt
        prompt_string = "The database schema is as follows:\n"
        for table, columns in schema.items():
            prompt_string += f"Table '{table}' has columns: {', '.join(columns)}\n"
        prompt_string += "\nBased on this schema, "
        
        return prompt_string
