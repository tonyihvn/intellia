import logging
import json
import uuid
from ..scheduler import parser as schedule_parser
from ..scheduler.actions import send_email, call_api
from ..scheduler.scheduler import get_scheduler
from .query_handler import QueryHandler
from ..llm.client import LLMClient
from ..rag.manager import RAGManager
from ..rag.enhancer import QueryEnhancer
from ..config import Config
import subprocess
import json
import os
import sys
import re
from ..db.connection import get_db_connection

logger = logging.getLogger(__name__)

class CommandHandler:
    """Handles both SQL queries and action commands (preview + confirm/execute)"""

    def __init__(self, llm_client=None, query_handler=None):
        self.llm_client = llm_client or LLMClient()
        # If no QueryHandler provided, attach a QueryEnhancer backed by RAG so
        # the LLM always receives schema/context from the RAG vector store.
        if query_handler:
            self.query_handler = query_handler
        else:
            try:
                rag = RAGManager()
                enhancer = QueryEnhancer(rag)
                self.query_handler = QueryHandler(self.llm_client, enhancer)
            except Exception:
                # Fallback to no enhancer
                self.query_handler = QueryHandler(self.llm_client, None)
        self.scheduler = get_scheduler()

    def handle_command(self, command_text: str, execute: bool = False, conversation: list = None, selected_tables: list = None):
        """
        Process a command which could be a query or an action.

        If execute=False (default) this returns a preview object that includes
        generated SQL (if applicable), explanation and any results for preview.

        If execute=True this will perform the action (run SQL / send email / call API)
        and return the final execution result.
        """
        # Quick heuristic: treat simple greetings as plain text chat replies
        try:
            import re
            greeting_re = re.compile(r"^\s*(hi|hello|hey|hiya|yo|good\s+morning|good\s+afternoon|good\s+evening)\b[\.!]?\s*$", re.I)
            if greeting_re.match(command_text or ""):
                # Prefer an LLM-generated friendly greeting if available, otherwise a default
                try:
                    conv_prefix = ''
                    if conversation and isinstance(conversation, list):
                        conv_prefix = "\n".join([f"{(m.get('role') or 'user').upper()}: {m.get('text') or ''}" if isinstance(m, dict) else str(m) for m in conversation[-10:]]) + "\n\n"
                    friendly = None
                    try:
                        friendly = self.llm_client.generate(conv_prefix + command_text)
                    except Exception:
                        friendly = None

                    return {
                        'type': 'chat_response',
                        'response': friendly or 'Hello! How can I help you today?',
                        'explanation': friendly or 'Greeting detected and handled as a chat message.',
                        'sql': None,
                        'classifier': {'intent': 'text', 'explain': 'greeting detected'},
                        'needs_confirmation': False
                    }
                except Exception:
                    return {
                        'type': 'chat_response',
                        'response': 'Hello! How can I help you today?',
                        'explanation': 'Greeting detected and handled as a chat message.',
                        'sql': None,
                        'classifier': {'intent': 'text', 'explain': 'greeting detected'},
                        'needs_confirmation': False
                    }
        except Exception:
            # on any error fall back to the normal flow below
            pass

        # Try to parse as an action (send_email, call_api, etc.)
        # Rapid intent detection via QueryEnhancer if available (helps route send_email vs sql quickly)
        enhancer = getattr(self.query_handler, 'query_enhancer', None)
        intent_payload = None
        try:
            if enhancer and hasattr(enhancer, 'detect_intent'):
                try:
                    intent_payload = enhancer.detect_intent(command_text)
                except Exception:
                    intent_payload = None
        except Exception:
            intent_payload = None

        try:
            action_spec = schedule_parser.parse(command_text)
        except Exception as e:
            logger.info(f"Action parser error: {e}")
            action_spec = None

        # If intent detector strongly indicates an action (e.g., send_email or call_api) and parser didn't
        # identify one, synthesize a minimal action_spec so preview flow proceeds faster.
        try:
            if not action_spec and intent_payload and isinstance(intent_payload, dict):
                it = (intent_payload.get('intent') or '').lower()
                if it == 'send_email':
                    # Build a minimal action_spec for send_email
                    # Attempt to pull any suggested recipient from heuristics
                    to = None
                    m = re.search(r"send\s+(?:an\s+)?email\s+to\s+([\w.\-+%]+@[\w.\-]+)\b", command_text, flags=re.I)
                    if m:
                        to = m.group(1)
                    action_spec = {'action': {'type': 'send_email', 'to': to, 'parameters': {}}, 'schedule': None}
                elif it == 'call_api':
                    action_spec = {'action': {'type': 'call_api', 'parameters': {}}, 'schedule': None}
        except Exception:
            pass

        # If parser didn't identify an action, attempt a simple heuristic for common actions
        # e.g., "send an email to foo@bar.com ..."
        if not action_spec:
            try:
                import re
                m = re.search(r"send\s+(?:an\s+)?email\s+to\s+([\w.\-+%]+@[\w.\-]+)\b(.*)$", command_text, flags=re.I)
                if m:
                    email = m.group(1).strip()
                    rest = m.group(2).strip()
                    body = rest or 'Please come right now.'
                    subject = 'Urgent: Please Come Now'
                    
                    # If execute is True, send the email
                    if execute:
                        try:
                            send_email(subject=subject, body=body, to=[email])
                            return {
                                'type': 'action_success',
                                'result': f"Email sent successfully to {email}",
                                'explanation': f"Email sent with subject '{subject}'",
                                'sql': None
                            }
                        except Exception as e:
                            return {
                                'type': 'action_failed',
                                'error': str(e),
                                'explanation': "Failed to send email",
                                'sql': None
                            }
                    
                    # Otherwise show preview
                    return {
                        'type': 'action_preview',
                        'action': 'send_email',
                        'details': {
                            'to': [email],
                            'subject': subject,
                            'body': body,
                            'from': Config().EMAIL_SETTINGS.get('from_address')
                        },
                        'explanation': f"Will send email to: {email}\nSubject: {subject}\nBody: {body}",
                        'sql': None,
                        'needs_confirmation': True
                    }
                    
                    # Build a minimal action_spec that mimics parser output
                    action_spec = {
                        'action': {
                            'type': 'send_email',
                            'to': email,
                            'parameters': {
                                'body': body
                            }
                        },
                        'schedule': None
                    }
                else:
                    # Detect simple "call API" commands: e.g. "Call API POST https://... with body {...}"
                    m2 = re.search(r"call\s+(?:the\s+)?api\s+(post|get|put|delete)\s+(https?://\S+)(.*)$", command_text, flags=re.I)
                    if m2:
                        method = m2.group(1).strip().lower()
                        url = m2.group(2).strip()
                        rest = m2.group(3).strip()
                        # try to extract a JSON body if present
                        body = None
                        try:
                            bmatch = re.search(r"\{.*\}", rest, flags=re.S)
                            if bmatch:
                                body = json.loads(bmatch.group(0))
                        except Exception:
                            body = rest or None
                        action_spec = {
                            'action': {
                                'type': 'call_api',
                                'parameters': {
                                    'method': method,
                                    'url': url,
                                    'body': body
                                }
                            },
                            'schedule': None
                        }
                    else:
                        # Detect a simple 'read webpage <url>' instruction to add a source
                        m3 = re.search(r"read\s+(?:the\s+)?webpage\s+(https?://\S+)", command_text, flags=re.I)
                        if m3:
                            url = m3.group(1).strip()
                            action_spec = {
                                'action': {
                                    'type': 'add_source',
                                    'parameters': {
                                        'source_type': 'webpage',
                                        'url': url
                                    }
                                },
                                'schedule': None
                            }
            except Exception:
                action_spec = None

        # If we detected an action, prepare a preview first
        if action_spec and action_spec.get('action'):
            action = action_spec.get('action')
            schedule = action_spec.get('schedule')

            # Determine whether this should be treated as immediate: schedule present means don't send now
            is_immediate_hint = ('now' in command_text.lower() or 'immediately' in command_text.lower())

            # If the schedule contains a date in the past, treat as immediate preview
            if schedule and schedule.get('date'):
                try:
                    from datetime import datetime, timezone
                    dt = datetime.fromisoformat(schedule.get('date').replace('Z', '+00:00'))
                    if dt < datetime.now(timezone.utc):
                        is_immediate_hint = True
                except Exception:
                    pass

            # For send_email actions, prefer to have the LLM compose the email subject/body
            # rather than attempting to generate SQL. SQL generation can be used only when
            # the action explicitly requests data from the DB (not the default).
            sql = None
            sql_explanation = None
            sql_results = None
            sql_error = None

            if isinstance(action, dict):
                if action.get('type') == 'send_email':
                    # Compose email via LLM into structured JSON: {subject, body}
                    try:
                        prompt = (
                            "You are an assistant that composes professional emails.\n"
                            "User instruction: \"" + command_text + "\"\n\n"
                            "Produce a JSON object with keys: subject, body.\n"
                            "- subject: a short email subject (max 80 chars).\n"
                            "- body: the full email body. Include a polite greeting, a clear message, and a closing salutation/signature.\n"
                            "Return ONLY valid JSON. Do NOT include any SQL or extra commentary.\n"
                        )
                        llm_out = None
                        try:
                            llm_out = self.llm_client.generate(prompt)
                        except Exception:
                            llm_out = self.llm_client.generate(command_text)

                        subj = None
                        body = None
                        try:
                            import json as _json
                            if isinstance(llm_out, dict):
                                subj = llm_out.get('subject')
                                body = llm_out.get('body')
                            else:
                                txt = str(llm_out)
                                s = txt.find('{')
                                e = txt.rfind('}')
                                if s != -1 and e != -1:
                                    j = _json.loads(txt[s:e+1])
                                    subj = j.get('subject')
                                    body = j.get('body')
                        except Exception:
                            try:
                                body = str(llm_out)
                                subj = (body.split('\n')[0] or '')[:80]
                            except Exception:
                                subj = 'Message'
                                body = str(command_text)

                        if 'parameters' not in action or not isinstance(action.get('parameters'), dict):
                            action['parameters'] = {}
                            if subj and not action['parameters'].get('subject'):
                                action['parameters']['subject'] = subj
                            if body and not action['parameters'].get('body'):
                                action['parameters']['body'] = body
                    except Exception as e:
                        sql_error = f"Email composition failed: {e}"
                elif action.get('type') == 'add_source':
                    # For web reading, do not generate/validate SQL
                    sql = None
                    sql_explanation = None
                    sql_results = None
                    sql_error = None
                else:
                    # For other actions, optionally generate SQL for recipient resolution
                    try:
                        gen = self.query_handler.generate_sql(command_text)
                        if isinstance(gen, dict):
                            sql = gen.get('sql') or gen.get('query') or None
                            sql_explanation = gen.get('explanation')
                        else:
                            sql = gen
                        if sql:
                            try:
                                val = self.query_handler.validate_sql_against_schema(sql)
                                if not val.get('ok'):
                                    sql_error = f"SQL validation failed: {val}"
                                    sql = None
                            except Exception as e:
                                sql_error = f"SQL validation error: {e}"
                    except Exception as e:
                        sql_error = str(e)

            # If we have an explanation for generated SQL, include it as a SQL comment
            sql_with_comment = None
            try:
                if sql:
                    if sql_explanation:
                        safe_expl = str(sql_explanation).strip().replace('*/', '')
                        sql_with_comment = f"/* {safe_expl} */\n{sql}"
                    else:
                        sql_with_comment = sql
            except Exception:
                sql_with_comment = sql

            # Attempt to extract any presentation hint from the SQL generator's full response
            presentation_hint_act = None
            try:
                full = gen.get('full_response') if isinstance(gen, dict) else (str(gen) if gen else '')
                if full and isinstance(full, str):
                    s = full.find('{')
                    e = full.rfind('}')
                    if s != -1 and e != -1:
                        try:
                            j = json.loads(full[s:e+1])
                            if isinstance(j, dict) and 'presentation' in j:
                                presentation_hint_act = j.get('presentation')
                        except Exception:
                            presentation_hint_act = None
            except Exception:
                presentation_hint_act = None

            preview = {
                'type': 'action_preview',
                'action': action,
                'schedule': schedule,
                'is_immediate_hint': is_immediate_hint,
                'sql': sql_with_comment or sql,
                'sql_explanation': sql_explanation,
                'sql_results': sql_results,
                'sql_error': sql_error,
                'needs_confirmation': True,
                'presentation_hint': presentation_hint_act,
                'format': (presentation_hint_act.get('display') if isinstance(presentation_hint_act, dict) else None) if presentation_hint_act else None
            }

            # Immediate send: consult app settings. If auto_send_emails is enabled, send when an explicit
            # recipient is present and no schedule is provided. Otherwise present a preview for confirmation.
            try:
                app_cfg = {}
                try:
                    from ..config import Config as _Config
                    app_cfg = _Config.get_app_settings() or {}
                except Exception:
                    app_cfg = {}

                auto_send = bool(app_cfg.get('auto_send_emails', False))

                if isinstance(action, dict) and action.get('type') == 'send_email':
                    to_field = action.get('to')
                    has_explicit_email = False
                    if isinstance(to_field, str) and '@' in to_field:
                        has_explicit_email = True
                    elif isinstance(to_field, list) and any(isinstance(t, str) and '@' in t for t in to_field):
                        has_explicit_email = True

                    # If auto_send is enabled and recipient is explicit and no schedule, execute immediately
                    if auto_send and has_explicit_email and not schedule:
                        exec_res = self._execute_action_with_optional_sql(command_text, action, sql=None)
                        return exec_res
            except Exception:
                # Fall back to preview if immediate execution fails
                pass

            # Attempt to resolve recipients from SQL results if not explicitly provided
            recipients = action.get('to')
            resolved_recipients = []
            if not recipients and sql_results:
                # collect any values that look like email addresses from the sample results
                for row in sql_results:
                    if isinstance(row, dict):
                        for v in row.values():
                            try:
                                if isinstance(v, str) and '@' in v:
                                    resolved_recipients.append(v.strip())
                            except Exception:
                                continue
            elif isinstance(recipients, str):
                if '@' in recipients:
                    resolved_recipients = [recipients]
                else:
                    # may be an id or username; leave unresolved for confirm step
                    resolved_recipients = []
            elif isinstance(recipients, list):
                resolved_recipients = [r for r in recipients if isinstance(r, str) and '@' in r]

            # Build a human-friendly planned action explanation for preview
            planned_parts = []
            if resolved_recipients:
                planned_parts.append(f"Recipients: {', '.join(resolved_recipients)}")
            else:
                planned_parts.append("Recipients: (will be determined from database or prompt on confirmation)")

            subj = action.get('parameters', {}).get('subject') or action.get('subject') or ''
            if subj:
                planned_parts.append(f"Subject: {subj}")

            body_preview = action.get('parameters', {}).get('body') or action.get('body') or ''
            if body_preview:
                # truncate long bodies for preview
                short_body = (body_preview[:300] + '...') if len(body_preview) > 300 else body_preview
                planned_parts.append(f"Body (preview): {short_body}")

            if schedule:
                planned_parts.append(f"Schedule: {schedule}")

            preview['planned_action'] = "; ".join(planned_parts)

            # If caller explicitly requested execution, perform the action now
            if execute:
                # For send_email, always send immediately if no schedule, or schedule if schedule is present
                if isinstance(action, dict) and action.get('type') == 'send_email':
                    if schedule:
                        # Schedule the email for later
                        try:
                            scheduler = get_scheduler()
                            job_id = f"email_{uuid.uuid4()}"
                            scheduler.add_job(
                                send_email,
                                'date',
                                run_date=schedule.get('date'),
                                args=[
                                    action.get('parameters', {}).get('subject') or action.get('subject') or 'No Subject',
                                    action.get('parameters', {}).get('body') or action.get('body') or '',
                                    action.get('to') or [],
                                ],
                                id=job_id
                            )
                            return {'type': 'action_completed', 'action': 'send_email', 'message': f'Email scheduled for {schedule.get('date')}', 'job_id': job_id}
                        except Exception as e:
                            return {'type': 'action_failed', 'error': f'Failed to schedule email: {e}'}
                    else:
                        # Send immediately
                        return self._execute_action_with_optional_sql(command_text, action, sql)
                else:
                    return self._execute_action_with_optional_sql(command_text, action, sql)

            # Do not auto-execute otherwise; user must confirm after reviewing the planned_action
            return preview

        # Not an action => treat as a database query / general question
        try:
            # For queries we return preview (sql + sample results) unless execute==True
            if not execute:
                # First, ask the LLM to classify whether this prompt needs SQL or can be answered as text
                try:
                    enhancer = getattr(self.query_handler, 'query_enhancer', None)
                    enhanced_ctx = None
                    if enhancer:
                        try:
                            # Pass through any selected_tables if provided (from clarifier UI)
                            enhanced = enhancer.enhance_query_context(command_text, selected_tables=selected_tables)
                            # If enhancer requests clarification, return a structured clarify response
                            if isinstance(enhanced, dict) and enhanced.get('clarify'):
                                return {
                                    'type': 'clarify',
                                    'prompt': enhanced.get('enhanced_prompt'),
                                    'candidates': enhanced.get('candidates', []),
                                    'sql': None,
                                    'needs_confirmation': False
                                }
                            enhanced_ctx = enhanced.get('enhanced_prompt')
                        except Exception:
                            enhanced_ctx = None

                    # Build a conversation prefix if provided so the LLM has context from prior chat turns
                    conv_prefix = ''
                    if conversation and isinstance(conversation, list):
                        try:
                            conv_lines = []
                            # limit to last 20 turns to avoid overly long prompts
                            for m in conversation[-20:]:
                                if isinstance(m, dict):
                                    role = m.get('role', 'user')
                                    text = m.get('text') or m.get('message') or ''
                                else:
                                    # fallback: treat as user text
                                    role = 'user'
                                    text = str(m)
                                conv_lines.append(f"{role.upper()}: {text}")
                            conv_prefix = "\n".join(conv_lines) + "\n\n"
                        except Exception:
                            conv_prefix = ''

                    classify_input = (enhanced_ctx + "\n\n" + conv_prefix + command_text) if enhanced_ctx else (conv_prefix + command_text)
                    cls = self.llm_client.classify(classify_input)
                except Exception as e:
                    cls = {'intent': 'unknown', 'explain': str(e), 'suggested_sql': None}

                intent = (cls.get('intent') or 'unknown').lower()
                # If intent is text, return an immediate chat-like response
                if intent == 'text':
                    try:
                        # Include conversation context when generating chat-style answers
                        gen_input = (conv_prefix + command_text) if 'conv_prefix' in locals() else command_text
                        answer = self.llm_client.generate(gen_input)
                    except Exception:
                        answer = cls.get('explain') or 'Unable to generate text response.'

                    # If the classifier nevertheless provided a suggested_sql, include it so the UI
                    # can show the generated SQL in the SQL editor (but do NOT execute without confirm).
                    suggested_sql = cls.get('suggested_sql') if isinstance(cls, dict) else None

                    # Validate any suggested SQL before returning it to the client so we
                    # don't populate the SQL editor with invalid references.
                    suggested_sql_validation = None
                    if suggested_sql:
                        try:
                            val = self.query_handler.validate_sql_against_schema(suggested_sql)
                            if not val.get('ok'):
                                suggested_sql_validation = val
                                suggested_sql = None
                        except Exception as e:
                            suggested_sql_validation = {'ok': False, 'error': str(e)}

                    return {
                        'type': 'chat_response',
                        'response': answer,
                        'explanation': answer,
                        'sql': suggested_sql or None,
                        'sql_validation': suggested_sql_validation,
                        'classifier': cls,
                        # Chat responses should not force a confirmation flow. If a suggested SQL
                        # is provided, include it for user visibility but do not require confirm.
                        'needs_confirmation': False
                    }

                # Otherwise, fall back to SQL preview flow (intent == 'sql' or unknown)
                # When generating SQL, include conversation context if available to help refine prompts
                gen_prompt = (conv_prefix + command_text) if 'conv_prefix' in locals() else command_text
                gen = self.query_handler.generate_sql(gen_prompt)
                sql = gen if isinstance(gen, str) else gen.get('sql') if isinstance(gen, dict) else None
                explanation = gen.get('explanation') if isinstance(gen, dict) else None

                # IMPORTANT: do NOT execute SQL during preview. Validate SQL against schema
                # and only return it for editing/execution if it passes validation.
                sql_error = None
                if sql:
                    try:
                        val = self.query_handler.validate_sql_against_schema(sql)
                        if not val.get('ok'):
                            sql_error = f"SQL validation failed: {val}"
                            sql = None
                    except Exception as e:
                        sql_error = f"SQL validation error: {e}"

                # If we have an explanation, include it as a SQL comment above the SQL
                sql_with_comment = None
                try:
                    if sql:
                        if explanation:
                            safe_expl = str(explanation).strip().replace('*/', '')
                            sql_with_comment = f"/* {safe_expl} */\n{sql}"
                        else:
                            sql_with_comment = sql
                except Exception:
                    sql_with_comment = sql

                # Attempt to parse any presentation hint JSON from the LLM full response
                presentation_hint = None
                try:
                    full = gen.get('full_response') if isinstance(gen, dict) else (str(gen) if gen else '')
                    if full and isinstance(full, str):
                        s = full.find('{')
                        e = full.rfind('}')
                        if s != -1 and e != -1:
                            try:
                                j = json.loads(full[s:e+1])
                                # if JSON contains presentation key, surface it
                                if isinstance(j, dict) and 'presentation' in j:
                                    presentation_hint = j.get('presentation')
                            except Exception:
                                # best-effort: ignore parse errors
                                presentation_hint = None
                except Exception:
                    presentation_hint = None

                return {
                    'type': 'query_preview',
                    'sql': sql_with_comment or sql,
                    'explanation': explanation,
                    'results': None,
                    'error': sql_error,
                    'needs_confirmation': True,
                    'classifier': cls,
                    'presentation_hint': presentation_hint,
                    'format': (presentation_hint.get('display') if isinstance(presentation_hint, dict) else None) if presentation_hint else None
                }

            # execute True => actually run the query
            result = self.query_handler.handle_query(command_text, execute=True)
            # Build a response that includes executed results or a presentation generated from sample rows
            resp = {
                'type': 'query_result',
                'sql': result.get('sql'),
                'results': result.get('results'),
                'explanation': result.get('explanation')
            }
            # Include presentation synthesized by the LLM when execution failed but a presentation exists
            if result.get('presentation'):
                resp['presentation'] = result.get('presentation')
            if result.get('sample_rows'):
                resp['sample_rows'] = result.get('sample_rows')
            if result.get('error_analysis'):
                resp['error_analysis'] = result.get('error_analysis')
            return resp

        except Exception as e:
            logger.error(f"Failed to handle query/command: {str(e)}")
            return {'type': 'error', 'error': str(e)}

    def _execute_action_with_optional_sql(self, command_text, action, sql=None, conversation=None, result_data=None):
        """Execute the given action. If `sql` is provided, it will be executed to produce data
        that can be embedded into the action (for example, a report used in an email body)."""
        try:
            # --- Business Rule Summarization Helper ---
            def summarize_and_store_business_rule():
                try:
                    rag = RAGManager()
                    # Compose a summary from the conversation, command, and result
                    summary_parts = []
                    if conversation and isinstance(conversation, list):
                        summary_parts.append("Conversation:\n" + "\n".join([
                            f"{(m.get('role') or 'user').capitalize()}: {m.get('text') or ''}" if isinstance(m, dict) else str(m)
                            for m in conversation[-10:]
                        ]))
                    summary_parts.append(f"Command: {command_text}")
                    if action:
                        summary_parts.append(f"Action: {json.dumps(action, default=str)[:500]}")
                    if result_data:
                        # Try to summarize the structure/nature of the result
                        if isinstance(result_data, dict) and 'results' in result_data and isinstance(result_data['results'], list):
                            rows = result_data['results']
                            if rows:
                                keys = list(rows[0].keys()) if isinstance(rows[0], dict) else []
                                summary_parts.append(f"Result columns: {keys}")
                                summary_parts.append(f"Sample row: {rows[0]}")
                                summary_parts.append(f"Rows returned: {len(rows)}")
                        elif isinstance(result_data, dict):
                            summary_parts.append(f"Result: {json.dumps(result_data, default=str)[:300]}")
                        else:
                            summary_parts.append(f"Result: {str(result_data)[:300]}")
                    # Compose the business rule text
                    rule_text = "\n".join(summary_parts)
                    rag.add_business_rule([{'text': rule_text, 'metadata': {'source': 'conversation', 'timestamp': str(uuid.uuid1())}}])
                except Exception as e:
                    logger.warning(f"Failed to summarize/store business rule: {e}")
            data_for_action = None
            # validate SQL against schema before executing to avoid accidental DDL/unknown-table errors
            if sql:
                try:
                    valid = self.query_handler.validate_sql_against_schema(sql)
                    if not valid.get('ok'):
                        return {'type': 'action_failed', 'error': 'SQL validation failed', 'details': valid}
                except Exception as e:
                    return {'type': 'action_failed', 'error': f'SQL validation error: {e}'}
            if sql:
                try:
                    data_for_action = self.query_handler.execute_sql(sql)
                except Exception as e:
                    logger.warning(f"Failed to execute SQL for action: {e}")

            # Resolve recipients: supports direct emails or patient id extraction
            recipients = action.get('to')
            if isinstance(recipients, str):
                # simple email
                if '@' in recipients:
                    recipients = [recipients]
                else:
                    # try to extract patient id and lookup email
                    m = re.search(r'(?:patient[_\s]*id[:=\s]*)(\d+)', command_text, re.I)
                    if m:
                        pid = m.group(1)
                        email = self._lookup_email_for_patient(pid)
                        recipients = [email] if email else []
                    else:
                        # treat as a contact name; attempt DB lookup for matching emails
                        names_found = self._lookup_email_by_name(recipients)
                        if names_found:
                            recipients = names_found
                        else:
                            # fallback: keep original string (may be resolved later)
                            recipients = [recipients]

            result_payload = None
            if action.get('type') == 'send_email':
                subject = action.get('parameters', {}).get('subject') or action.get('subject') or 'No Subject'
                body = action.get('parameters', {}).get('body') or action.get('body') or ''

                # If there is data_for_action, append a short report summary to the body
                if data_for_action:
                    try:
                        # produce small textual summary
                        if isinstance(data_for_action, list):
                            if len(data_for_action) == 1 and isinstance(data_for_action[0], dict):
                                vals = list(data_for_action[0].values())
                                body += '\n\nReport:\n' + str(vals[0])
                            else:
                                body += '\n\nReport (first rows):\n' + str(data_for_action[:5])
                    except Exception:
                        pass

            # Handle 'add_source' action for reading and summarizing a website
            if action.get('type') == 'add_source':
                url = action.get('parameters', {}).get('url') or action.get('url')
                if url:
                    try:
                        from ..web import fetch_webpage
                        page_content = fetch_webpage(url)
                        llm = LLMClient()
                        summary_prompt = f"Summarize the world news from the following webpage content:\n\n{page_content}"
                        summary = llm.generate(summary_prompt)
                        return {'type': 'action_completed', 'action': 'read_webpage', 'url': url, 'summary': summary}
                    except Exception as e:
                        return {'type': 'action_failed', 'error': f'Failed to read or summarize website: {e}'}
                else:
                    return {'type': 'action_failed', 'error': 'No URL provided for website reading.'}

                try:
                    send_email(subject=subject, body=body, to=recipients or [], cc=action.get('cc', []))
                    result_payload = {'type': 'action_completed', 'action': 'send_email', 'message': f'Email sent to {recipients}'}
                except Exception as e:
                    # fallback: try invoking the CLI script (scripts/send_email.py) as a subprocess
                    try:
                        payload = {
                            'to': recipients or [],
                            'subject': subject,
                            'body': body,
                            'cc': action.get('cc', []) or []
                        }
                        # Call python CLI script with JSON via stdin
                        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'scripts', 'send_email.py')
                        proc = subprocess.run([
                            sys.executable,
                            script_path,
                            '--stdin'
                        ], input=json.dumps(payload), text=True, capture_output=True, timeout=120)
                        out = proc.stdout.strip()
                        if proc.returncode == 0:
                            result_payload = {'type': 'action_completed', 'action': 'send_email', 'message': f'Email sent to {recipients}', 'cli_output': out}
                        else:
                            result_payload = {'type': 'action_failed', 'error': f'CLI fallback failed: {proc.stderr or out}'}
                    except Exception as e2:
                        result_payload = {'type': 'action_failed', 'error': f'Failed to send email: {e} ; fallback error: {e2}'}
                summarize_and_store_business_rule()
                return result_payload

            elif action.get('type') == 'call_api':
                result = call_api(
                    method=action.get('parameters', {}).get('method', action.get('method', 'post')),
                    url=action.get('parameters', {}).get('url', action.get('url')),
                    headers=action.get('parameters', {}).get('headers', action.get('headers')),
                    json_body=action.get('parameters', {}).get('body', action.get('body'))
                )
                summarize_and_store_business_rule()
                return {'type': 'action_completed', 'action': 'call_api', 'response': result}

            else:
                summarize_and_store_business_rule()
                return {'type': 'action_failed', 'error': 'Unsupported action type'}

        except Exception as e:
            logger.error(f"Failed to execute action: {e}")
            try:
                summarize_and_store_business_rule()
            except Exception:
                pass
            return {'type': 'action_failed', 'error': str(e)}

    def _lookup_email_for_patient(self, patient_id):
        """Try common table/column patterns to find an email for a given patient id."""
        try:
            # Try a few common queries - tolerant to schema differences
            candidates = [
                f"SELECT email FROM patient WHERE patient_id = {patient_id} LIMIT 1",
                f"SELECT email FROM person WHERE person_id = {patient_id} LIMIT 1",
                f"SELECT email FROM users WHERE id = {patient_id} LIMIT 1"
            ]
            for q in candidates:
                try:
                    res = self.query_handler.execute_sql(q)
                    if res and isinstance(res, list) and len(res) and list(res[0].values())[0]:
                        return list(res[0].values())[0]
                except Exception:
                    continue
        except Exception:
            pass
        return ''

    def _lookup_email_by_name(self, name: str):
        """Try to find one or more email addresses for a person/contact by name.

        This method is tolerant to common schemas. It tries several likely tables
        and columns using case-insensitive LIKE searches. Returns a list of
        matching email addresses (may be empty).
        """
        emails = []
        try:
            q = name or ''
            q = q.strip()
            if not q:
                return []

            # split into tokens to allow partial matching
            tokens = [t for t in q.replace(',', ' ').split() if t]
            like_clause = '%' + '%'.join(tokens) + '%'

            conn = None
            try:
                conn = get_db_connection()
                if not conn:
                    return []

                cur = conn.cursor()
                # Try a sequence of common tables/columns
                candidates = [
                    ("contacts", "email", ["full_name", "name", "contact_name"]),
                    ("people", "email", ["full_name", "name", "first_name", "last_name"]),
                    ("users", "email", ["full_name", "username", "name", "first_name", "last_name"]),
                    ("patients", "email", ["full_name", "first_name", "last_name"]),
                ]

                for table, email_col, name_cols in candidates:
                    for col in name_cols:
                        try:
                            # very light sanitization for identifiers
                            if not re.match(r'^[A-Za-z0-9_]+$', col) or not re.match(r'^[A-Za-z0-9_]+$', table) or not re.match(r'^[A-Za-z0-9_]+$', email_col):
                                continue
                            sql = f"SELECT `{email_col}` FROM `{table}` WHERE LOWER(`{col}`) LIKE %s LIMIT 10"
                            cur.execute(sql, (like_clause.lower(),))
                            rows = cur.fetchall() or []
                            for r in rows:
                                try:
                                    val = r[0]
                                    if val and isinstance(val, (str,)) and '@' in val:
                                        emails.append(val.strip())
                                except Exception:
                                    continue
                            if emails:
                                # stop early if found
                                break
                        except Exception:
                            continue
                    if emails:
                        break
            finally:
                try:
                    if conn:
                        conn.close()
                except Exception:
                    pass
        except Exception:
            pass

        # Deduplicate and return
        out = []
        for e in emails:
            if e not in out:
                out.append(e)
        return out