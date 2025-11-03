from ..db.schema_fetcher import SchemaFetcher
import logging

class SchemaContextManager:
    def __init__(self, db_connection):
        self.schema_fetcher = SchemaFetcher(db_connection)
        self.schema_context = None
        self.refresh_schema()

    def refresh_schema(self):
        """Refresh the database schema context"""
        try:
            tables = self.schema_fetcher.get_tables()
            columns = self.schema_fetcher.get_columns()
            relationships = self.schema_fetcher.get_relationships()
            
            self.schema_context = {
                'tables': tables,
                'columns': columns,
                'relationships': relationships
            }
            logging.info("Schema context refreshed successfully")
            return True
        except Exception as e:
            logging.error(f"Failed to refresh schema context: {str(e)}")
            return False

    def get_context(self):
        """Get the current schema context"""
        return self.schema_context

    def get_table_info(self, table_name):
        """Get detailed information about a specific table"""
        if not self.schema_context:
            return None
        
        return {
            'columns': [col for col in self.schema_context['columns'] 
                       if col['table_name'] == table_name],
            'relationships': [rel for rel in self.schema_context['relationships']
                            if rel['table_name'] == table_name or rel['referenced_table'] == table_name]
        }