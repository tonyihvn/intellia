from typing import List, Dict
from .manager import RAGManager
import json
import os

class QueryEnhancer:
    def __init__(self, rag_manager: RAGManager):
        """Initialize the query enhancer with a RAG manager."""
        self.rag_manager = rag_manager
    
    def enhance_query_context(self, question: str, selected_tables: list = None) -> Dict:
        """Enhance the query with relevant context from the RAG system.
        
        Args:
            question: The user's natural language question
            
        Returns:
            Dict containing:
                - enhanced_prompt: The enhanced prompt with relevant context
                - relevant_info: List of relevant pieces of information used
        """
        # Use RAG manager to build a compact context (schema + rules) and compress it via LLM
        # If the caller provided an explicit selection of tables, bias the RAG selection
        qry = question
        if selected_tables and isinstance(selected_tables, (list, tuple)) and len(selected_tables) > 0:
            qry = f"{question}. Use tables: {', '.join(selected_tables)}"

        compact = self.rag_manager.get_compact_context(qry, max_tables=5, max_rules=8)

        # If compaction indicates clarification is needed, build a clarifying minimal response
        if compact.get('clarify'):
            candidates = compact.get('candidates', []) or []
            candidate_lines = '\n'.join([f"- {c}" for c in candidates[:8]])
            enhanced_prompt = f"""The user's question appears ambiguous with respect to which table(s) to use.
Candidate tables (from semantic search):
{candidate_lines}

Please choose which of the candidate tables you'd like to target, or provide the exact table name(s)."""
            # Return structured payload so callers can detect clarify and candidates
            return {'enhanced_prompt': enhanced_prompt, 'relevant_info': [], 'clarify': True, 'candidates': candidates}

        # Build the enhanced prompt using the compressed summary
        summary = compact.get('summary', '')
        tables = compact.get('tables', [])
        rules = compact.get('rules', [])

        context_parts = []
        if rules:
            context_parts.append("Business Rules (apply these first):")
            for r in rules[:8]:
                context_parts.append(f"- {r}")

        if summary:
            context_parts.append("Database Schema Summary (concise):")
            context_parts.append(summary)

        context_parts.append("IMPORTANT: Use only the table and column names provided in the Database Schema Summary. If a required column is missing, ask for clarification instead of guessing.")

        context_str = "\n".join(context_parts)
        enhanced_prompt = f"""Use this compact context to help answer the question.
Context:
{context_str}

User Question: {question}

Instructions for the assistant:
- Apply the Business Rules first when deciding filters, joins, or row-level logic.
- Use ONLY the table and column names exactly as provided in the Database Schema Summary; do NOT invent or guess column names.
- If necessary columns are missing from the schema, ask for clarification instead of generating SQL.
- When asked to produce SQL, return the SQL and a brief explanation of what it does.
"""

        return {'enhanced_prompt': enhanced_prompt, 'relevant_info': []}