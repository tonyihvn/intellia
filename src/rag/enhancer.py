from typing import List, Dict
from .manager import RAGManager
import json
import os
from ..config import Config
from ..llm.client import LLMClient
import json

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
        # Check visualization settings to see if RAG is enabled globally
        try:
            vis_cfg_path = os.path.join(Config.CONFIG_DIR, 'visualization.json')
            if os.path.exists(vis_cfg_path):
                try:
                    with open(vis_cfg_path, 'r') as vf:
                        vis = json.load(vf) or {}
                        if vis.get('rag_enabled') is False:
                            # RAG explicitly disabled: return no enhanced prompt so caller falls back to plain generation
                            return {'enhanced_prompt': None, 'relevant_info': [], 'clarify': False}
                except Exception:
                    pass
        except Exception:
            pass

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

    def detect_intent(self, question: str, conv_prefix: str = '') -> dict:
        """Use the enhancer + a lightweight LLM prompt to rapidly classify intent.

        Returns a dict with keys:
          - intent: one of ('send_email','sql','call_api','add_source','text','schedule','unknown')
          - display: optional display hint ('table','chart','csv','analysis','pdf','excel',...)
          - explain: short explanation of the decision
          - suggested_sql: optional SQL string or None
        """
        try:
            llm = LLMClient()
            prompt = f"""
Classify the user's instruction into one of these intents: send_email, sql, call_api, add_source, schedule, text.
Also, if the user clearly requested a particular output presentation, include a display hint (table, chart, csv, analysis, pdf, excel) in the response.

Return ONLY a JSON object with keys: intent, display, explain, suggested_sql (suggested_sql may be null).

User instruction: {question}

If intent is 'sql' and you can produce an immediate suggested SQL, include it in suggested_sql; otherwise set it to null.
"""
            out = llm.generate(prompt)
            txt = out if isinstance(out, str) else str(out)
            # extract JSON substring
            s = txt.find('{')
            e = txt.rfind('}')
            if s != -1 and e != -1:
                try:
                    payload = json.loads(txt[s:e+1])
                    # normalize keys
                    return {
                        'intent': (payload.get('intent') or payload.get('label') or 'unknown').lower(),
                        'display': (payload.get('display') or payload.get('presentation') or None),
                        'explain': payload.get('explain') or payload.get('reason') or '',
                        'suggested_sql': payload.get('suggested_sql') or payload.get('sql') or None
                    }
                except Exception:
                    pass
            # fallback: attempt simple heuristics
            low = (question or '').lower()
            if 'send' in low and 'email' in low:
                return {'intent': 'send_email', 'display': None, 'explain': 'heuristic: contains "send" and "email"', 'suggested_sql': None}
            if 'call api' in low or 'call the api' in low or low.startswith('call api'):
                return {'intent': 'call_api', 'display': None, 'explain': 'heuristic: contains "call api"', 'suggested_sql': None}
            if 'select' in low or 'find' in low or 'show' in low or 'how many' in low or 'list' in low:
                return {'intent': 'sql', 'display': None, 'explain': 'heuristic: likely a SQL/lookup request', 'suggested_sql': None}
            return {'intent': 'unknown', 'display': None, 'explain': 'fallback: unable to classify', 'suggested_sql': None}
        except Exception:
            return {'intent': 'unknown', 'display': None, 'explain': 'error during detection', 'suggested_sql': None}