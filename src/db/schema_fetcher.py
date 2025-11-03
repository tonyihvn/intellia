import mysql.connector

class SchemaFetcher:
    def __init__(self, db_connection):
        self.connection = db_connection

    def get_tables(self):
        """Return a list of table dicts: [{'table_name': name}, ...]"""
        if not self.connection:
            return []
        cursor = self.connection.cursor()
        try:
            cursor.execute("SHOW TABLES")
            rows = cursor.fetchall()
            # rows may be list of tuples like [(table1,), (table2,)]
            tables = []
            for r in rows:
                name = r[0] if isinstance(r, (list, tuple)) else r
                tables.append({'table_name': name})
            return tables
        finally:
            cursor.close()

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

    # Convenience methods expected by SchemaContextManager
    def get_columns(self):
        """Return a list of columns as dicts with table_name and column metadata."""
        if not self.connection:
            return []
        # Use a non-dictionary cursor for SHOW TABLES to avoid returning dict rows
        # which would make indexing by [0] fail (KeyError: 0).
        cursor = self.connection.cursor()
        try:
            cursor.execute("SHOW TABLES")
            tables = [t[0] for t in cursor.fetchall()]
        finally:
            cursor.close()

        all_columns = []
        for table in tables:
            c = None
            try:
                c = self.connection.cursor(dictionary=True)
                c.execute(f"DESCRIBE `{table}`")
                cols = c.fetchall() or []
                for col in cols:
                    entry = {'table_name': table, 'column_name': col.get('Field'), 'type': col.get('Type'), 'key': col.get('Key')}
                    all_columns.append(entry)
            finally:
                if c:
                    c.close()

        return all_columns

    def get_relationships(self):
        """Return a list of relationship dicts for foreign keys."""
        if not self.connection:
            return []
        cursor = self.connection.cursor()
        try:
            cursor.execute("SHOW TABLES")
            tables = [t[0] for t in cursor.fetchall()]
        finally:
            cursor.close()

        rels = []
        for table in tables:
            c = None
            try:
                c = self.connection.cursor()
                c.execute("""
                    SELECT COLUMN_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
                    FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                    WHERE TABLE_NAME = %s
                      AND REFERENCED_TABLE_NAME IS NOT NULL
                      AND TABLE_SCHEMA = DATABASE()
                """, (table,))
                rows = c.fetchall() or []
                for row in rows:
                    # row may be tuple (col, ref_table, ref_col)
                    rels.append({
                        'table_name': table,
                        'column': row[0],
                        'referenced_table': row[1],
                        'referenced_column': row[2]
                    })
            finally:
                if c:
                    c.close()

        return rels
