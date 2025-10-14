import chromadb
import os
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import logging

class RAGManager:
    def __init__(self, persist_directory: str = "chroma_db"):
        """Initialize the RAG manager with ChromaDB for vector storage."""
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Create collections for different types of knowledge
        self.schema_collection = self.client.get_or_create_collection(
            name="schema_knowledge",
            metadata={"description": "Database schema and structure knowledge"}
        )
        
        self.business_rules_collection = self.client.get_or_create_collection(
            name="business_rules",
            metadata={"description": "Business rules and domain knowledge"}
        )
        
        # Initialize the sentence transformer model for embeddings
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def add_schema_knowledge(self, descriptions: List[Dict[str, str]]):
        """Add schema-related knowledge to the vector store.
        
        Args:
            descriptions: List of dictionaries with 'text' and 'metadata' keys
        """
        try:
            texts = [d['text'] for d in descriptions]
            ids = [f"schema_{i}" for i in range(len(texts))]
            metadatas = [d.get('metadata', {}) for d in descriptions]
            
            # Generate embeddings using sentence-transformers
            embeddings = self.model.encode(texts).tolist()
            
            # Add to ChromaDB
            self.schema_collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
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
            ids = [f"rule_{i}" for i in range(len(texts))]
            metadatas = [d.get('metadata', {}) for d in descriptions]
            
            # Generate embeddings
            embeddings = self.model.encode(texts).tolist()
            
            # Add to ChromaDB
            self.business_rules_collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
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
            # Generate embedding for the question
            question_embedding = self.model.encode(question).tolist()
            
            # Query both collections
            schema_results = self.schema_collection.query(
                query_embeddings=[question_embedding],
                n_results=n_results
            )
            
            business_results = self.business_rules_collection.query(
                query_embeddings=[question_embedding],
                n_results=n_results
            )
            
            # Combine and format results
            combined_results = []
            
            # Add schema results
            if schema_results['documents']:
                for doc, metadata, distance in zip(
                    schema_results['documents'][0],
                    schema_results['metadatas'][0],
                    schema_results['distances'][0]
                ):
                    combined_results.append({
                        'text': doc,
                        'metadata': metadata,
                        'relevance': 1 - distance,  # Convert distance to similarity score
                        'type': 'schema'
                    })
            
            # Add business rule results
            if business_results['documents']:
                for doc, metadata, distance in zip(
                    business_results['documents'][0],
                    business_results['metadatas'][0],
                    business_results['distances'][0]
                ):
                    combined_results.append({
                        'text': doc,
                        'metadata': metadata,
                        'relevance': 1 - distance,
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
            schema_results = self.schema_collection.get()
            business_results = self.business_rules_collection.get()
            
            return {
                'schema': [
                    {'text': doc, 'metadata': meta}
                    for doc, meta in zip(
                        schema_results['documents'],
                        schema_results['metadatas']
                    )
                ],
                'business_rules': [
                    {'text': doc, 'metadata': meta}
                    for doc, meta in zip(
                        business_results['documents'],
                        business_results['metadatas']
                    )
                ]
            }
        except Exception as e:
            logging.error(f"Error retrieving knowledge base: {str(e)}")
            return {'schema': [], 'business_rules': []}