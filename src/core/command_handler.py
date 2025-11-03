import logging
import json
from ..scheduler import parser as schedule_parser
from ..scheduler.actions import send_email, call_api
from ..scheduler.scheduler import get_scheduler
from .query_handler import QueryHandler
from ..llm.client import LLMClient

logger = logging.getLogger(__name__)

class CommandHandler:
    """Handles both SQL queries and action commands (preview + confirm/execute)"""

    def __init__(self, llm_client=None, query_handler=None):
        self.llm_client = llm_client or LLMClient()
        self.query_handler = query_handler or QueryHandler(self.llm_client, None)
        self.scheduler = get_scheduler()

    def handle_command(self, command_text: str, execute: bool = False):
        """
        Process a command which could be a query or an action.

        If execute=False (default) this returns a preview object that includes
        generated SQL (if applicable), explanation and any results for preview.

        If execute=True this will perform the action (run SQL / send email / call API)
        and return the final execution result.
        """
        # Try to parse as an action (send_email, call_api, etc.)
        try:
            action_spec = schedule_parser.parse(command_text)
        except Exception as e:
            logger.info(f"Action parser error: {e}")
            action_spec = None

        # If we detected an action, prepare a preview first
        if action_spec and action_spec.get('action'):
            action = action_spec.get('action')
            schedule = action_spec.get('schedule')

            # Determine whether this should be treated as immediate (now/immediately or schedule absent)
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

            # Try to also generate SQL (if the command references data) for preview
            sql = None
            sql_explanation = None
            sql_results = None
            sql_error = None
            try:
                gen = self.query_handler.generate_sql(command_text)
                # generate_sql may return a string or dict depending on implementation; normalize
                if isinstance(gen, dict):
                    sql = gen.get('sql') or gen.get('query') or None
                    sql_explanation = gen.get('explanation')
                else:
                    sql = gen
                if sql:
                    # For preview only, execute readonly SQL to show results (safe for SELECTs)
                    try:
                        sql_results = self.query_handler.execute_sql(sql)
                    except Exception as e:
                        sql_error = str(e)
            except Exception as e:
                sql_error = str(e)

            preview = {
                'type': 'action_preview',
                'action': action,
                'schedule': schedule,
                'is_immediate_hint': is_immediate_hint,
                'sql': sql,
                'sql_explanation': sql_explanation,
                'sql_results': sql_results,
                'sql_error': sql_error
            }

            # If caller requested execution, perform the action now
            if execute:
                return self._execute_action_with_optional_sql(command_text, action, sql)

            return preview

        # Not an action => treat as a database query / general question
        try:
            # For queries we return preview (sql + sample results) unless execute==True
            if not execute:
                gen = self.query_handler.generate_sql(command_text)
                sql = gen if isinstance(gen, str) else gen.get('sql') if isinstance(gen, dict) else None
                explanation = gen.get('explanation') if isinstance(gen, dict) else None
                sample_results = None
                sample_error = None
                try:
                    if sql:
                        sample_results = self.query_handler.execute_sql(sql)
                except Exception as e:
                    sample_error = str(e)

                return {
                    'type': 'query_preview',
                    'sql': sql,
                    'explanation': explanation,
                    'results': sample_results,
                    'error': sample_error
                }

            # execute True => actually run the query
            result = self.query_handler.handle_query(command_text, execute=True)
            return {
                'type': 'query_result',
                'sql': result.get('sql'),
                'results': result.get('results'),
                'explanation': result.get('explanation')
            }

        except Exception as e:
            logger.error(f"Failed to handle query/command: {str(e)}")
            return {'type': 'error', 'error': str(e)}

    def _execute_action_with_optional_sql(self, command_text, action, sql=None):
        """Execute the given action. If `sql` is provided, it will be executed to produce data
        that can be embedded into the action (for example, a report used in an email body)."""
        try:
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
                    import re
                    m = re.search(r'(?:patient[_\s]*id[:=\s]*)(\d+)', command_text, re.I)
                    if m:
                        pid = m.group(1)
                        recipients = [self._lookup_email_for_patient(pid)]
                    else:
                        recipients = [recipients]

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

                send_email(subject=subject, body=body, to=recipients or [], cc=action.get('cc', []))
                return {'type': 'action_completed', 'action': 'send_email', 'message': f'Email sent to {recipients}'}

            elif action.get('type') == 'call_api':
                result = call_api(
                    method=action.get('parameters', {}).get('method', action.get('method', 'post')),
                    url=action.get('parameters', {}).get('url', action.get('url')),
                    headers=action.get('parameters', {}).get('headers', action.get('headers')),
                    json_body=action.get('parameters', {}).get('body', action.get('body'))
                )
                return {'type': 'action_completed', 'action': 'call_api', 'response': result}

            else:
                return {'type': 'action_failed', 'error': 'Unsupported action type'}

        except Exception as e:
            logger.error(f"Failed to execute action: {e}")
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