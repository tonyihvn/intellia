import re
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)

# Very small deterministic parser for simple NL schedule instructions.
# Supports patterns like:
# - "every Monday at 09:00 send an email to x@example.com subject 'Hi' body '...'
# - "every day at 08:00"
# - "on 2025-10-21 14:00 call api POST https://... with body {...}"

WEEKDAYS = {
    'monday': 'mon', 'tuesday': 'tue', 'wednesday': 'wed', 'thursday': 'thu', 'friday': 'fri', 'saturday': 'sat', 'sunday': 'sun'
}


def parse(prompt: str):
    p = prompt.strip().lower()

    # Check for email instruction
    email_match = re.search(r"send an email to ([\w@.\-,\s]+) (?:with subject '([^']+)')?(?: body '([^']+)')?", p)
    if email_match:
        to_raw = email_match.group(1).strip()
        to = [x.strip() for x in re.split(r'[;,\s]+', to_raw) if x.strip()]
        subject = email_match.group(2) or 'No subject'
        body = email_match.group(3) or ''
    else:
        to = None
        subject = None
        body = None

    # Check schedule frequency: every <weekday> at hh:mm
    every_match = re.search(r"every\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|day)\s+at\s+(\d{1,2}:\d{2})", p)
    if every_match:
        day = every_match.group(1)
        time_str = every_match.group(2)
        if day == 'day':
            cron = f"{int(time_str.split(':')[1])} {int(time_str.split(':')[0])} * * *"
        else:
            dow = WEEKDAYS.get(day)
            cron = f"{int(time_str.split(':')[1])} {int(time_str.split(':')[0])} * * {dow}"
        schedule = {'cron': cron}
    else:
        # On specific date/time
        date_match = re.search(r"on\s+(\d{4}-\d{2}-\d{2})(?:[ T](\d{1,2}:\d{2}))?", p)
        if date_match:
            date = date_match.group(1)
            time_str = date_match.group(2) or '00:00'
            dt = f"{date}T{time_str}:00"
            schedule = {'date': dt}
        else:
            # interval e.g., every 10 minutes
            interval_match = re.search(r"every\s+(\d+)\s+(minute|minutes|hour|hours|second|seconds)", p)
            if interval_match:
                val = int(interval_match.group(1))
                unit = interval_match.group(2)
                seconds = val * (60 if 'minute' in unit else (3600 if 'hour' in unit else 1))
                schedule = {'interval': {'seconds': seconds}}
            else:
                schedule = None

    action = None
    if to:
        action = {
            'type': 'send_email',
            'to': to,
            'subject': subject,
            'body': body
        }
    else:
        # Check for API call
        api_match = re.search(r"call api (get|post|put|delete) (https?://\S+)(?: with body (\{.*\}))?", prompt, re.I)
        if api_match:
            method = api_match.group(1)
            url = api_match.group(2)
            body = None
            try:
                if api_match.group(3):
                    body = json.loads(api_match.group(3))
            except Exception:
                body = None
            action = {'type': 'call_api', 'method': method, 'url': url, 'body': body}

    result = {'action': action, 'schedule': schedule}
    logger.info(f"Parsed NL schedule: {result}")
    return result
