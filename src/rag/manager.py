import os
import logging
from typing import List, Dict, Optional
import numpy as np
from pathlib import Path
import json
from sklearn.feature_extraction.text import TfidfVectorizer

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

class RAGManager:
    def __init__(self):
        """Initialize RAG manager with simple vector stores for schema and business rules."""
        self.schema_collection = SimpleVectorStore("schema")
        self.business_rules_collection = SimpleVectorStore("business_rules")
        
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
                'business_rules': self.business_rules_collection.get_all()
            }
        except Exception as e:
            logging.error(f"Error retrieving knowledge base: {str(e)}")
            return {'schema': [], 'business_rules': []}

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
            for table in tables:
                # Columns
                columns_cursor = db_connection.cursor(dictionary=True)
                try:
                    columns_cursor.execute(f"DESCRIBE `{table}`")
                    columns = columns_cursor.fetchall() or []
                finally:
                    columns_cursor.close()

                key_cols = [c['Field'] for c in columns if c.get('Key') in ('PRI', 'MUL')]
                sample_cols = [c['Field'] for c in columns[:10]]

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

            if schema_descriptions:
                return self.add_schema_knowledge(schema_descriptions)
            return False
        except Exception as e:
            logging.error(f"Error bootstrapping RAG from DB: {str(e)}")
            return False