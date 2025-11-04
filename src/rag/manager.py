import os
import logging
import shutil
from typing import List, Dict, Optional
import numpy as np
from pathlib import Path
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from .schema_context import SchemaContextManager

class SimpleVectorStore:
    def __init__(self, name: str):
        """Initialize a simple vector store."""
        self.name = name
        self.vectors = []
        self.documents = []
        self.metadatas = []
        self.storage_dir = Path("vector_store") / name
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.vectorizer = TfidfVectorizer()
        self._load_from_disk()
        
    def clear_data(self):
        """Clear all stored data and remove files from disk."""
        self.vectors = []
        self.documents = []
        self.metadatas = []
        self.vectorizer = TfidfVectorizer()
        
        # Remove all files in storage directory
        if self.storage_dir.exists():
            for file in self.storage_dir.glob("*"):
                try:
                    if file.is_file():
                        file.unlink()
                    elif file.is_dir():
                        shutil.rmtree(str(file))
                except Exception as e:
                    logging.error(f"Error removing file {file}: {e}")
            
            # Remove and recreate the directory
            shutil.rmtree(str(self.storage_dir))
            self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _load_from_disk(self):
        """Load vectors and documents from disk if they exist."""
        vector_file = self.storage_dir / "vectors.npy"
        docs_file = self.storage_dir / "documents.json"
        vectorizer_file = self.storage_dir / "vectorizer.pkl"
        
        if vector_file.exists() and docs_file.exists():
            self.vectors = np.load(str(vector_file)).tolist()
            with open(docs_file, 'r') as f:
                data = json.load(f)
                self.documents = data['documents']
                self.metadatas = data['metadatas']
                
            # If vectorizer was saved, load it
            if vectorizer_file.exists():
                with open(vectorizer_file, 'rb') as f:
                    import pickle
                    self.vectorizer = pickle.load(f)

    def _save_to_disk(self):
        """Save vectors and documents to disk."""
        np.save(str(self.storage_dir / "vectors.npy"), np.array(self.vectors))
        with open(self.storage_dir / "documents.json", 'w') as f:
            json.dump({
                'documents': self.documents,
                'metadatas': self.metadatas
            }, f)
            
        # Save the vectorizer
        with open(self.storage_dir / "vectorizer.pkl", 'wb') as f:
            import pickle
            pickle.dump(self.vectorizer, f)

    def add(self, documents: List[str], metadatas: Optional[List[dict]] = None):
        """Add documents to the store."""
        if metadatas is None:
            metadatas = [{} for _ in documents]
        
        if not documents:
            return
            
        # If we have existing documents, extend the vectorizer's vocabulary
        if self.vectors:
            all_docs = self.documents + documents
            matrix = self.vectorizer.fit_transform(all_docs)
            self.vectors = matrix.toarray().tolist()
        else:
            # First time adding documents
            matrix = self.vectorizer.fit_transform(documents)
            self.vectors = matrix.toarray().tolist()
        
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self._save_to_disk()

    def query(self, query: str, n_results: int = 5) -> dict:
        """Query the store with a text query."""
        if not self.vectors:
            return {'documents': [], 'metadatas': [], 'distances': []}
            
        # Transform query using the same vocabulary
        query_vec = self.vectorizer.transform([query]).toarray()[0]
        
        # Calculate cosine similarities
        similarities = np.dot(self.vectors, query_vec) / (
            np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(query_vec)
        )
        
        # Get top k results
        top_k_idx = np.argsort(similarities)[-n_results:][::-1]
        
        return {
            'documents': [self.documents[i] for i in top_k_idx],
            'metadatas': [self.metadatas[i] for i in top_k_idx],
            'distances': [float(similarities[i]) for i in top_k_idx]
        }

    def get_all(self) -> List[Dict]:
        """Get all documents and their metadata."""
        return [
            {'text': doc, 'metadata': meta}
            for doc, meta in zip(self.documents, self.metadatas)
        ]

    def delete_by_text(self, text: str) -> bool:
        """Delete documents that match the given text exactly and persist changes.

        Returns True if deletion occurred (or nothing to delete), False on error.
        """
        try:
            # Find indexes to keep
            remaining = [(d, m) for d, m in zip(self.documents, self.metadatas) if d != text]
            if len(remaining) == len(self.documents):
                # nothing to delete
                return True

            if not remaining:
                # No documents remain; clear files
                self.documents = []
                self.metadatas = []
                self.vectors = []
                # Overwrite files
                self._save_to_disk()
                return True

            remaining_docs, remaining_metas = zip(*remaining)
            self.documents = list(remaining_docs)
            self.metadatas = list(remaining_metas)

            # Refit vectorizer on remaining documents
            try:
                matrix = self.vectorizer.fit_transform(self.documents)
                self.vectors = matrix.toarray().tolist()
            except Exception:
                # If fitting fails for any reason, clear vectors but keep docs on disk
                self.vectors = []

            self._save_to_disk()
            return True
        except Exception as e:
            logging.error(f"Error deleting document from vector store '{self.name}': {str(e)}")
            return False

class RAGManager:
    def __init__(self):
        """Initialize RAG manager with simple vector stores for schema and business rules."""
        self.schema_collection = SimpleVectorStore("schema")
        self.business_rules_collection = SimpleVectorStore("business_rules")
        # New collection to store user-provided example commands/prompts
        self.examples_collection = SimpleVectorStore("examples")
        self.schema_context = None
        
    def clear_all_data(self):
        """Clear all stored data and caches."""
        # Clear vector stores
        self.schema_collection.clear_data()
        self.business_rules_collection.clear_data()
        self.examples_collection.clear_data()
        
        # Reset schema context
        self.schema_context = None
        
        # Clear any cached data in memory
        self.__init__()
    
    def set_db_context(self, db_connection):
        """Set up database schema context"""
        self.schema_context = SchemaContextManager(db_connection)
        
    def generate_query_context(self, question):
        """Generate context for query generation"""
        context = {
            'question': question,
            'schema': self.schema_context.get_context() if self.schema_context else None,
            'relevant_tables': self._find_relevant_tables(question)
        }
        return context
    
    def _find_relevant_tables(self, question):
        """Find tables relevant to the question"""
        if not self.schema_context:
            return []
            
        schema = self.schema_context.get_context()
        relevant = []
        
        # First try semantic search against the schema collection if available
        try:
            results = self.schema_collection.query(question, n_results=5)
            tables_found = []
            for meta in results.get('metadatas', []):
                if isinstance(meta, dict) and meta.get('table'):
                    tname = meta.get('table')
                    if tname not in tables_found:
                        tables_found.append(tname)
            if tables_found:
                # Build table entries from schema_context for these tables
                for tname in tables_found:
                    tbl = next((t for t in schema['tables'] if t.get('table_name') == tname), None)
                    if tbl:
                        relevant.append(tbl)
                return relevant
        except Exception:
            # fall back to keyword matching
            pass

        # Fallback: Simple keyword matching
        for table in schema.get('tables', []):
            if table.get('table_name', '').lower() in question.lower():
                relevant.append(table)
                # Add related tables through relationships
                for rel in schema.get('relationships', []):
                    if rel.get('table_name') == table.get('table_name'):
                        ref = rel.get('referenced_table')
                        if ref:
                            ref_tbl = next((t for t in schema.get('tables', []) if t.get('table_name') == ref), None)
                            if ref_tbl:
                                relevant.append(ref_tbl)
                        
        return relevant

    def get_schema_snippet_for_question(self, question: str, max_tables: int = 5) -> str:
        """Return a concise schema snippet (only relevant tables and relationships) for the given question.

        This is optimized to keep the LLM prompt small: only include table names, important columns,
        and direct relationships for the most relevant tables.
        """
        if not self.schema_context:
            return ""

        schema = self.schema_context.get_context()
        if not schema:
            return ""

        # Determine relevant tables using semantic search or fallback
        table_objs = self._find_relevant_tables(question)
        # If none found, default to top N tables
        if not table_objs:
            table_names = [t.get('table_name') for t in schema.get('tables', [])][:max_tables]
            table_objs = [t for t in schema.get('tables', []) if t.get('table_name') in table_names]

        # Limit to max_tables
        table_objs = table_objs[:max_tables]

        parts = []
        for tbl in table_objs:
            tname = tbl.get('table_name')
            parts.append(f"Table: {tname}")
            # columns
            cols = [c.get('column_name') or c.get('Field') for c in schema.get('columns', []) if c.get('table_name') == tname]
            if not cols:
                # try to get detailed table info from schema_context
                info = self.schema_context.get_table_info(tname)
                if info and info.get('columns'):
                    cols = [c.get('Field') for c in info.get('columns')]
            if cols:
                parts.append("  Columns: " + ", ".join(cols[:20]))

            # relationships
            rels = [r for r in schema.get('relationships', []) if r.get('table_name') == tname or r.get('referenced_table') == tname]
            if not rels:
                info = self.schema_context.get_table_info(tname)
                if info and info.get('relationships'):
                    rels = info.get('relationships')
            if rels:
                # format relationships concisely
                rel_lines = []
                for r in rels:
                    if isinstance(r, dict):
                        rel_lines.append(f"{r.get('column')} -> {r.get('referenced_table')}.{r.get('referenced_column')}")
                    else:
                        # assume string
                        rel_lines.append(str(r))
                parts.append("  Relationships: " + "; ".join(rel_lines[:10]))

        snippet = "\n".join(parts)
        return snippet

    def add_schema_knowledge(self, descriptions: List[Dict[str, str]]):
        """Add schema-related knowledge to the vector store.
        
        Args:
            descriptions: List of dictionaries with 'text' and 'metadata' keys
        """
        try:
            texts = [d['text'] for d in descriptions]
            metadatas = [d.get('metadata', {}) for d in descriptions]
            
            # Add to the vector store
            self.schema_collection.add(
                documents=texts,
                metadatas=metadatas
            )
            
            return True
        except Exception as e:
            logging.error(f"Error adding schema knowledge: {str(e)}")
            return False
            
    def add_business_rule(self, descriptions: List[Dict[str, str]]):
        """Add business rule knowledge to the vector store.
        
        Args:
            descriptions: List of dictionaries with 'text' and 'metadata' keys
        """
        try:
            texts = [d['text'] for d in descriptions]
            metadatas = [d.get('metadata', {}) for d in descriptions]
            
            # Add to the vector store
            self.business_rules_collection.add(
                documents=texts,
                metadatas=metadatas
            )
            
            return True
        except Exception as e:
            logging.error(f"Error adding business rule: {str(e)}")
            return False

    def add_examples(self, examples: List[Dict[str, str]]):
        """Add example commands/prompts to the examples vector collection.

        Args:
            examples: List of dictionaries with 'text' and optional 'metadata'
        """
        try:
            texts = [d['text'] for d in examples]
            metadatas = [d.get('metadata', {}) for d in examples]
            # Tag metadata as example source for easier filtering
            for m in metadatas:
                m.setdefault('source', 'examples')

            self.examples_collection.add(documents=texts, metadatas=metadatas)
            return True
        except Exception as e:
            logging.error(f"Error adding examples: {str(e)}")
            return False

    def get_all_examples(self) -> List[Dict]:
        """Return all stored examples as list of {'text', 'metadata'}"""
        try:
            return self.examples_collection.get_all()
        except Exception as e:
            logging.error(f"Error getting examples: {str(e)}")
            return []

    def delete_by_text(self, collection: str, text: str) -> bool:
        """Delete an item from the specified collection that matches the given text exactly.
        
        Args:
            collection: The collection to delete from ('schema', 'business_rules', or 'examples')
            text: The exact text to match and delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            if collection == 'schema':
                store = self.schema_collection
            elif collection == 'business_rules':
                store = self.business_rules_collection
            elif collection == 'examples':
                store = self.examples_collection
            else:
                logging.error(f"Invalid collection: {collection}")
                return False
                
            # Use SimpleVectorStore.delete_by_text
            return store.delete_by_text(text)

        except Exception as e:
            logging.error(f"Error deleting from {collection}: {str(e)}")
            return False
            
    def delete_example(self, text: str) -> bool:
        """Delete example(s) that match the given text exactly from the examples collection."""
        return self.delete_by_text('examples', text)
    
    def query_knowledge(self, question: str, n_results: int = 5) -> List[Dict]:
        """Query both schema and business knowledge to find relevant information.
        
        Args:
            question: The user's question
            n_results: Number of results to return
            
        Returns:
            List of relevant pieces of information with their metadata
        """
        try:
            # Query both collections
            schema_results = self.schema_collection.query(
                query=question,
                n_results=n_results
            )
            
            business_results = self.business_rules_collection.query(
                query=question,
                n_results=n_results
            )
            
            combined_results = []
            
            # Add schema results
            for doc, metadata, distance in zip(
                schema_results['documents'],
                schema_results['metadatas'],
                schema_results['distances']
            ):
                combined_results.append({
                    'text': doc,
                    'metadata': metadata,
                    'relevance': distance,  # Use cosine similarity directly
                    'type': 'schema'
                })
            
            # Add business rule results
            for doc, metadata, distance in zip(
                business_results['documents'],
                business_results['metadatas'],
                business_results['distances']
            ):
                combined_results.append({
                    'text': doc,
                    'metadata': metadata,
                    'relevance': distance,
                    'type': 'business_rule'
                })
            
            # Sort by relevance
            combined_results.sort(key=lambda x: x['relevance'], reverse=True)
            
            return combined_results[:n_results]
            
        except Exception as e:
            logging.error(f"Error querying knowledge base: {str(e)}")
            return []
    
    def get_all_knowledge(self) -> Dict[str, List[Dict]]:
        """Retrieve all stored knowledge."""
        try:
            return {
                'schema': self.schema_collection.get_all(),
                'business_rules': self.business_rules_collection.get_all(),
                'examples': self.examples_collection.get_all()
            }
        except Exception as e:
            logging.error(f"Error retrieving knowledge base: {str(e)}")
            return {'schema': [], 'business_rules': [], 'examples': []}

    def is_empty(self) -> bool:
        """Return True if both collections are empty."""
        try:
            return len(self.schema_collection.get_all()) == 0 and len(self.business_rules_collection.get_all()) == 0
        except Exception:
            return True

    def bootstrap_from_db(self, db_connection) -> bool:
        """Populate initial schema knowledge from the connected database.

        Adds one entry per table describing key columns and basic relationships.
        """
        try:
            if not db_connection or not hasattr(db_connection, 'is_connected') or not db_connection.is_connected():
                return False

            cursor = db_connection.cursor()
            cursor.execute("SHOW TABLES")
            tables = [row[0] for row in cursor.fetchall()]
            cursor.close()

            if not tables:
                return False

            schema_descriptions = []
            business_rule_descriptions = []
            for table in tables:
                # Columns and types
                columns_cursor = db_connection.cursor(dictionary=True)
                try:
                    columns_cursor.execute(f"DESCRIBE `{table}`")
                    columns = columns_cursor.fetchall() or []
                finally:
                    columns_cursor.close()

                key_cols = [c['Field'] for c in columns if c.get('Key') in ('PRI', 'MUL')]
                sample_cols = [c['Field'] for c in columns[:10]]
                col_types = [(c['Field'], c['Type']) for c in columns]

                # Sample values for all columns
                sample_values = {}
                try:
                    sample_cursor = db_connection.cursor(dictionary=True)
                    sample_cursor.execute(f"SELECT * FROM `{table}` LIMIT 5")
                    rows = sample_cursor.fetchall() or []
                    for col in [c['Field'] for c in columns]:
                        # Collect up to 3 sample values for each column
                        sample_values[col] = [str(row.get(col)) for row in rows if row.get(col) is not None][:3]
                except Exception:
                    pass
                finally:
                    try:
                        sample_cursor.close()
                    except Exception:
                        pass

                # Relationships
                fk_cursor = db_connection.cursor()
                try:
                    fk_cursor.execute(
                        """
                        SELECT COLUMN_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
                        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                        WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s AND REFERENCED_TABLE_NAME IS NOT NULL
                        """,
                        (table,)
                    )
                    fks = fk_cursor.fetchall() or []
                finally:
                    fk_cursor.close()

                rels = [f"{r[0]} -> {r[1]}.{r[2]}" for r in fks]

                text_parts = [
                    f"Table '{table}' overview:",
                    f"- Key columns: {', '.join(key_cols) if key_cols else 'none detected'}",
                    f"- Sample columns: {', '.join(sample_cols)}",
                ]
                if rels:
                    text_parts.append("- Relationships: " + "; ".join(rels))

                schema_descriptions.append({
                    'text': "\n".join(text_parts),
                    'metadata': {'table': table}
                })

                # Business rule: describe columns, types, and sample values
                br_lines = [f"Business Rule for table '{table}':"]
                for col, typ in col_types:
                    samples = sample_values.get(col, [])
                    sample_str = f"; Sample values: {', '.join(samples)}" if samples else ""
                    br_lines.append(f"- Column '{col}' (type: {typ}){sample_str}")
                if rels:
                    br_lines.append("- Relationships: " + "; ".join(rels))
                business_rule_descriptions.append({
                    'text': "\n".join(br_lines),
                    'metadata': {'table': table}
                })

            # Add both schema and business rules
            ok1 = self.add_schema_knowledge(schema_descriptions) if schema_descriptions else True
            ok2 = self.add_business_rule(business_rule_descriptions) if business_rule_descriptions else True
            return ok1 and ok2
        except Exception as e:
            logging.error(f"Error bootstrapping RAG from DB: {str(e)}")
            return False