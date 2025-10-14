from typing import List, Dict
from ..rag.manager import RAGManager
import json
import os

class QueryEnhancer:
    def __init__(self, rag_manager: RAGManager):
        """Initialize the query enhancer with a RAG manager."""
        self.rag_manager = rag_manager
    
    def enhance_query_context(self, question: str) -> Dict:
        """Enhance the query with relevant context from the RAG system.
        
        Args:
            question: The user's natural language question
            
        Returns:
            Dict containing:
                - enhanced_prompt: The enhanced prompt with relevant context
                - relevant_info: List of relevant pieces of information used
        """
        # Get relevant knowledge
        relevant_info = self.rag_manager.query_knowledge(question, n_results=5)
        
        # Build context string from relevant information
        context_parts = []
        
        # Add schema information
        schema_info = [info for info in relevant_info if info['type'] == 'schema']
        if schema_info:
            context_parts.append("Database Schema Context:")
            for info in schema_info:
                context_parts.append(f"- {info['text']}")
        
        # Add business rules
        rule_info = [info for info in relevant_info if info['type'] == 'business_rule']
        if rule_info:
            context_parts.append("\nBusiness Rules:")
            for info in rule_info:
                context_parts.append(f"- {info['text']}")
        
        # Build the enhanced prompt
        context_str = "\n".join(context_parts)
        enhanced_prompt = f"""Use this context to help answer the question:
{context_str}

User Question: {question}

Based on the above context, generate an SQL query that will answer the question."""
        
        return {
            'enhanced_prompt': enhanced_prompt,
            'relevant_info': relevant_info
        }