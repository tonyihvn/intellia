import smtplib
import logging
import requests
from email.mime.text import MIMEText
from ..config import Config

logger = logging.getLogger(__name__)


def send_email(subject: str, body: str, to: list, cc: list = None):
    """Send an email using SMTP settings in Config.

    Config must provide SMTP settings under Config.EMAIL_SETTINGS as dict with keys:
    - host, port, username, password, use_tls (bool)
    """
    config = Config()
    smtp_cfg = config.EMAIL_SETTINGS
    if not smtp_cfg:
        raise Exception("SMTP configuration not set")

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = smtp_cfg.get('from_address', smtp_cfg.get('username'))
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
        logger.info(f"Email sent to {to}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email: {str(e)}")
        raise


def call_api(method: str, url: str, headers: dict = None, json_body: dict = None):
    """Perform an HTTP call and return response content/status."""
    method = method.lower()
    try:
        if method == 'get':
            r = requests.get(url, headers=headers, timeout=30)
        elif method == 'post':
            r = requests.post(url, headers=headers, json=json_body, timeout=30)
        else:
            r = requests.request(method, url, headers=headers, json=json_body, timeout=30)

        r.raise_for_status()
        logger.info(f"API call to {url} succeeded with status {r.status_code}")
        return {'status': r.status_code, 'body': r.json() if r.headers.get('content-type','').startswith('application/json') else r.text}
    except Exception as e:
        logger.error(f"API call failed: {str(e)}")
        raise