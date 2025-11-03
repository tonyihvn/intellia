from ..db.schema_fetcher import SchemaFetcher
import logging

class SchemaContextManager:
    def __init__(self, db_connection):
        self.schema_fetcher = SchemaFetcher(db_connection)
        self.schema_context = None
        self.refresh_schema()

    def refresh_schema(self):
        """Refresh the database schema context"""
        # Perform stepwise refresh with detailed logging to aid debugging
        try:
            # Check connection availability
            conn = getattr(self.schema_fetcher, 'connection', None)
            if conn is None:
                logging.error("Schema refresh failed: no DB connection provided to SchemaFetcher")
                return False

            is_conn_ok = True
            try:
                # Some connectors expose is_connected(), others do not
                if hasattr(conn, 'is_connected'):
                    is_conn_ok = conn.is_connected()
            except Exception:
                # If checking connection status fails, proceed and let the calls fail with useful errors
                is_conn_ok = True

            if not is_conn_ok:
                logging.error("Schema refresh failed: DB connection appears closed or not connected")
                return False

            try:
                tables = self.schema_fetcher.get_tables()
            except Exception as e:
                logging.error(f"Failed to get tables when refreshing schema context: {e!r}")
                return False

            try:
                columns = self.schema_fetcher.get_columns()
            except Exception as e:
                logging.error(f"Failed to get columns when refreshing schema context: {e!r}")
                return False

            try:
                relationships = self.schema_fetcher.get_relationships()
            except Exception as e:
                logging.error(f"Failed to get relationships when refreshing schema context: {e!r}")
                return False

            self.schema_context = {
                'tables': tables,
                'columns': columns,
                'relationships': relationships
            }
            logging.info("Schema context refreshed successfully")
            return True
        except Exception as e:
            # Final fallback: log repr for easier debugging
            logging.error(f"Failed to refresh schema context: {e!r}")
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