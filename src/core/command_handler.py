import logging
import json
from ..scheduler import parser as schedule_parser
from ..scheduler.actions import send_email, call_api
from ..scheduler.scheduler import get_scheduler
from .query_handler import QueryHandler
from ..llm.client import LLMClient
import subprocess
import json
import os
import sys

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
                # Do NOT execute any SQL during preview. Execution only occurs on explicit confirm.
                sql_results = None
                sql_error = None
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
                'sql_error': sql_error,
                'needs_confirmation': True
            }

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
                return self._execute_action_with_optional_sql(command_text, action, sql)

            # Do not auto-execute otherwise; user must confirm after reviewing the planned_action
            return preview

        # Not an action => treat as a database query / general question
        try:
            # For queries we return preview (sql + sample results) unless execute==True
            if not execute:
                gen = self.query_handler.generate_sql(command_text)
                sql = gen if isinstance(gen, str) else gen.get('sql') if isinstance(gen, dict) else None
                explanation = gen.get('explanation') if isinstance(gen, dict) else None

                # IMPORTANT: do NOT execute SQL during preview. Return SQL and explanation only.
                return {
                    'type': 'query_preview',
                    'sql': sql,
                    'explanation': explanation,
                    'results': None,
                    'error': None,
                    'needs_confirmation': True
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

                try:
                    send_email(subject=subject, body=body, to=recipients or [], cc=action.get('cc', []))
                    return {'type': 'action_completed', 'action': 'send_email', 'message': f'Email sent to {recipients}'}
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
                            return {'type': 'action_completed', 'action': 'send_email', 'message': f'Email sent to {recipients}', 'cli_output': out}
                        else:
                            return {'type': 'action_failed', 'error': f'CLI fallback failed: {proc.stderr or out}'}
                    except Exception as e2:
                        return {'type': 'action_failed', 'error': f'Failed to send email: {e} ; fallback error: {e2}'}

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