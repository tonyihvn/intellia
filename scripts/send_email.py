#!/usr/bin/env python3
"""
Simple CLI to send an email using the app's Config EMAIL_SETTINGS or explicit parameters.
Usage:
  python scripts/send_email.py --to alice@example.com,bob@example.com --subject "Hi" --body "Hello"
Or pass JSON via stdin:
  echo '{"to": ["a@b.com"], "subject": "Hi", "body": "Hello"}' | python scripts/send_email.py --stdin

This script intentionally doesn't store or require credentials in code. It will read SMTP settings from
environment variables (SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_TLS, SMTP_FROM) or fall back to
Config in the repository if run inside the project.
"""
import os
import sys
import json
import argparse
import smtplib
from email.mime.text import MIMEText

try:
    # try relative import of Config if running inside project
    from src.config import Config
except Exception:
    Config = None


def send_email_cli(to, subject, body, cc=None):
    smtp_cfg = {
        'host': os.getenv('SMTP_HOST'),
        'port': int(os.getenv('SMTP_PORT', '587')) if os.getenv('SMTP_PORT') else None,
        'username': os.getenv('SMTP_USER'),
        'password': os.getenv('SMTP_PASS'),
        'use_tls': os.getenv('SMTP_TLS', 'true').lower() in ('1','true','yes'),
        'from_address': os.getenv('SMTP_FROM')
    }

    # If some fields missing, try Config
    if (not smtp_cfg['host'] or not smtp_cfg['username'] or not smtp_cfg['password']) and Config:
        cfg = Config.EMAIL_SETTINGS or {}
        smtp_cfg['host'] = smtp_cfg['host'] or cfg.get('host')
        smtp_cfg['port'] = smtp_cfg['port'] or cfg.get('port') or 587
        smtp_cfg['username'] = smtp_cfg['username'] or cfg.get('username')
        smtp_cfg['password'] = smtp_cfg['password'] or cfg.get('password')
        smtp_cfg['use_tls'] = smtp_cfg['use_tls'] if smtp_cfg['use_tls'] is not None else cfg.get('use_tls', True)
        smtp_cfg['from_address'] = smtp_cfg['from_address'] or cfg.get('from_address') or smtp_cfg['username']

    if not smtp_cfg['host'] or not smtp_cfg['username'] or not smtp_cfg['password']:
        print('SMTP configuration incomplete. Set SMTP_HOST, SMTP_USER, SMTP_PASS (or configure in Config).', file=sys.stderr)
        sys.exit(2)

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = smtp_cfg.get('from_address') or smtp_cfg['username']
    msg['To'] = ', '.join(to)
    if cc:
        msg['Cc'] = ', '.join(cc)

    try:
        if smtp_cfg.get('use_tls', True):
            server = smtplib.SMTP(smtp_cfg['host'], smtp_cfg.get('port', 587))
            server.starttls()
        else:
            server = smtplib.SMTP_SSL(smtp_cfg['host'], smtp_cfg.get('port', 465))

        server.login(smtp_cfg['username'], smtp_cfg['password'])
        server.sendmail(msg['From'], to + (cc or []), msg.as_string())
        server.quit()
        print(json.dumps({'success': True, 'message': f'Email sent to {to}'}))
        return 0
    except Exception as e:
        print(json.dumps({'success': False, 'error': str(e)}))
        return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--to', help='Comma-separated recipient emails')
    parser.add_argument('--subject', help='Email subject', default='No subject')
    parser.add_argument('--body', help='Email body', default='')
    parser.add_argument('--cc', help='Comma-separated cc emails', default='')
    parser.add_argument('--stdin', action='store_true', help='Read JSON from stdin')

    args = parser.parse_args()

    if args.stdin:
        try:
            payload = json.load(sys.stdin)
            to = payload.get('to') or payload.get('recipients') or []
            if isinstance(to, str):
                to = [x.strip() for x in to.split(',') if x.strip()]
            subject = payload.get('subject', args.subject)
            body = payload.get('body', args.body)
            cc = payload.get('cc') or []
            if isinstance(cc, str) and cc:
                cc = [x.strip() for x in cc.split(',') if x.strip()]
        except Exception as e:
            print('Failed to read JSON from stdin: ' + str(e), file=sys.stderr)
            sys.exit(3)
    else:
        if not args.to:
            print('Recipient (--to) is required when not using --stdin', file=sys.stderr)
            sys.exit(2)
        to = [x.strip() for x in args.to.split(',') if x.strip()]
        subject = args.subject
        body = args.body
        cc = [x.strip() for x in args.cc.split(',') if x.strip()] if args.cc else []

    code = send_email_cli(to, subject, body, cc)
    sys.exit(code)


if __name__ == '__main__':
    sys.exit(main())
