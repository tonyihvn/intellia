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
        relevant_info = self.rag_manager.query_knowledge(question, n_results=10)

        # Build context string from relevant information
        context_parts = []

        # Add business rules first - business rules take precedence in deciding
        # how to interpret or filter data. They should be applied before schema
        # instructions when the LLM produces an answer or SQL.
        rule_info = [info for info in relevant_info if info.get('type') == 'business_rule']
        if rule_info:
            context_parts.append("Business Rules (PRIORITY - apply these first):")
            for info in rule_info:
                context_parts.append(f"- {info.get('text')}")

        # Then provide the targeted schema snippet (if any). The schema is
        # authoritative for table and column names. The model MUST use exact
        # table and column names as provided below; do NOT invent or guess names.
        if schema_snippet:
            context_parts.append("\nDatabase Schema Context (use these table and column names exactly):")
            context_parts.append(schema_snippet)
        else:
            # Fallback to including top schema docs from semantic search
            schema_info = [info for info in relevant_info if info.get('type') == 'schema']
            if schema_info:
                context_parts.append("\nDatabase Schema Context (use these table and column names exactly):")
                for info in schema_info:
                    context_parts.append(f"- {info.get('text')}")

        # Safety note: enforce strict behavior in prompt construction
        context_parts.append("\nIMPORTANT: When producing SQL, reference only the table and column names shown in the Database Schema Context above. If a required column is not present, do not guess - ask for clarification instead. Do not generate SQL using inferred or assumed column names.")
        
        # Build the enhanced prompt
        context_str = "\n".join(context_parts)
        enhanced_prompt = f"""Use this context to help answer the question.
Business rules (above) have priority and must be applied before schema-level reasoning.
Schema lines are authoritative for table and column names.

Context:
{context_str}

User Question: {question}

Instructions for the assistant:
- Apply the Business Rules first when deciding filters, joins, or row-level logic.
- Use ONLY the table and column names exactly as provided in the Database Schema Context; do NOT invent or guess column names.
- If necessary columns are missing from the schema, ask for clarification instead of generating SQL.
- When asked to produce SQL, return the SQL and a brief explanation of what it does.
"""

        return {
            'enhanced_prompt': enhanced_prompt,
            'relevant_info': relevant_info
        }