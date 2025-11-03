from typing import List, Dict
from .manager import RAGManager
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
        # Get a concise schema snippet for the question (limits token usage)
        schema_snippet = self.rag_manager.get_schema_snippet_for_question(question, max_tables=5)

        # Get relevant knowledge (business rules and any schema docs)
        relevant_info = self.rag_manager.query_knowledge(question, n_results=5)

        # Build context string from relevant information
        context_parts = []
        
        # Add the targeted schema snippet first (if any)
        if schema_snippet:
            context_parts.append("Database Schema Context:")
            context_parts.append(schema_snippet)
        else:
            # Fallback to including top schema docs from semantic search
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