import mysql.connector

class SchemaFetcher:
    def __init__(self, db_connection):
        self.connection = db_connection

    def get_table_columns(self, table_name):
        """Get detailed column information for a table."""
        cursor = self.connection.cursor(dictionary=True)
        try:
            cursor.execute(f"DESCRIBE `{table_name}`")
            return cursor.fetchall()
        finally:
            cursor.close()

    def get_foreign_keys(self, table_name):
        """Get foreign key relationships for a table."""
        cursor = self.connection.cursor()
        try:
            cursor.execute(f"""
                SELECT 
                    COLUMN_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
                FROM 
                    INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                WHERE 
                    TABLE_NAME = %s 
                    AND REFERENCED_TABLE_NAME IS NOT NULL
                    AND TABLE_SCHEMA = DATABASE()
            """, (table_name,))
            return cursor.fetchall()
        finally:
            cursor.close()

    def get_schema_summary(self, query_text=None):
        """
        Get a targeted summary of the database schema focusing on relevant tables.
        """
        if not self.connection or not self.connection.is_connected():
            return "Database connection not available."

        cursor = None
        try:
            cursor = self.connection.cursor(dictionary=True)
            query_lower = query_text.lower() if query_text else ""
            
            # For patient count query, return only patient table info
            if 'total' in query_lower and 'patient' in query_lower:
                table_info = self._get_table_info(cursor, 'patient')
                return table_info if table_info else "Table: patient (not found)"
            
            # For other queries, identify relevant tables
            relevant_tables = self._get_relevant_tables(query_lower)
            
            # Get schema for relevant tables
            schema_info = []
            for table_name in relevant_tables:
                table_info = self._get_table_info(cursor, table_name)
                if table_info:
                    schema_info.append(table_info)
            
            return "\n".join(schema_info) if schema_info else "No relevant tables found"
            
        except Exception as e:
            return f"Error fetching schema: {str(e)}"
        finally:
            if cursor:
                cursor.close()
                
    def _get_relevant_tables(self, query_lower):
        """Helper method to identify relevant tables based on query."""
        relevant_tables = set()
        
        # Keywords to match tables for specific types of queries
        table_keywords = {
            'patient': ['patient', 'person'],
            'visit': ['visit', 'encounter'],
            'provider': ['provider', 'staff'],
            'drug': ['drug', 'medication', 'prescription'],
            'diagnosis': ['diagnosis', 'condition', 'obs'],
            'appointment': ['appointment', 'schedule']
        }
        
        # Match tables based on keywords
        for category, keywords in table_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                if category == 'patient':
                    relevant_tables.update(['patient'])
                elif category == 'visit':
                    relevant_tables.update(['visit'])
                # Add other mappings as needed
                
        # Default to patient table if no matches
        if not relevant_tables:
            relevant_tables.add('patient')
            
        return relevant_tables
        
    def _get_table_info(self, cursor, table_name):
        """Helper method to get concise table information."""
        try:
            # Get primary key and important columns
            cursor.execute(f"DESCRIBE `{table_name}`")
            columns = cursor.fetchall()
            if not columns:
                return None
                
            # Format table information concisely
            important_cols = [col['Field'] for col in columns 
                            if col['Key'] in ['PRI', 'MUL'] or 
                               col['Field'] in ['patient_id', 'person_id', 'name', 'gender', 'birthdate']]
            
            table_info = f"Table: {table_name}\n"
            table_info += f"Key Fields: {', '.join(important_cols)}"
            
            return table_info
            
        except Exception:
            return None

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
            
            # For each table, get its columns and relationships
            for table_name in tables:
                columns = self.get_table_columns(table_name)
                foreign_keys = self.get_foreign_keys(table_name)
                
                schema[table_name] = {
                    'columns': [col['Field'] for col in columns],
                    'relationships': [
                        f"{fk[0]} references {fk[1]}.{fk[2]}"
                        for fk in foreign_keys
                    ]
                }
        finally:
            cursor.close()

        # Format the schema into a string for the prompt
        prompt_string = "Database Schema:\n\n"
        for table, info in schema.items():
            prompt_string += f"Table '{table}':\n"
            prompt_string += f"  Columns: {', '.join(info['columns'])}\n"
            if info['relationships']:
                prompt_string += f"  Relationships:\n    " + "\n    ".join(info['relationships']) + "\n"
        prompt_string += "\nBased on this schema, "
        
        return prompt_string
