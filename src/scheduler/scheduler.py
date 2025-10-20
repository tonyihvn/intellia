import os
import json
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime
from typing import Dict, Any
from ..config import Config
from .actions import send_email, call_api

logger = logging.getLogger(__name__)

class SchedulerManager:
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.scheduler.start()
        self.config_dir = Config.CONFIG_DIR
        os.makedirs(self.config_dir, exist_ok=True)
        self.jobs_file = os.path.join(self.config_dir, 'schedules.json')
        self._load_jobs()

    def _load_jobs(self):
        if os.path.exists(self.jobs_file):
            try:
                with open(self.jobs_file, 'r') as f:
                    data = json.load(f)
                    for job in data.get('jobs', []):
                        try:
                            self._schedule_job(job, save=False)
                        except Exception as e:
                            logger.error(f"Failed to reschedule job {job.get('id')}: {e}")
            except Exception as e:
                logger.error(f"Failed to read schedules file: {e}")

    def _save_jobs(self):
        # Dump currently known jobs to disk (source is jobs scheduled via this manager)
        jobs = []
        for job in self.scheduler.get_jobs():
            job_data = job.kwargs.get('meta') if job.kwargs else None
            if job_data:
                jobs.append(job_data)
        try:
            with open(self.jobs_file, 'w') as f:
                json.dump({'jobs': jobs}, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save jobs: {e}")

    def _schedule_job(self, job_spec: Dict[str, Any], save=True):
        """Internal: schedule a job from job_spec"""
        job_id = job_spec.get('id') or f"job_{int(datetime.utcnow().timestamp())}"
        action = job_spec.get('action')
        schedule = job_spec.get('schedule')
        meta = job_spec.copy()
        meta['id'] = job_id

        # Determine trigger
        trigger = None
        if schedule.get('cron'):
            trigger = CronTrigger.from_crontab(schedule['cron'])
        elif schedule.get('date'):
            trigger = DateTrigger(run_date=datetime.fromisoformat(schedule['date']))
        elif schedule.get('interval'):
            iv = schedule['interval']
            trigger = IntervalTrigger(seconds=iv.get('seconds'), minutes=iv.get('minutes'), hours=iv.get('hours'))
        else:
            raise Exception('Invalid schedule format')

        # Decide the callable and args
        if action.get('type') == 'send_email':
            subject = action.get('subject')
            body = action.get('body')
            to = action.get('to', [])
            cc = action.get('cc', [])
            func = lambda meta=meta: send_email(subject, body, to, cc)
        elif action.get('type') == 'call_api':
            method = action.get('method', 'post')
            url = action.get('url')
            headers = action.get('headers')
            json_body = action.get('body')
            func = lambda meta=meta: call_api(method, url, headers, json_body)
        else:
            raise Exception('Unsupported action type')

        # Add job
        self.scheduler.add_job(func, trigger, id=job_id, kwargs={'meta': meta}, replace_existing=True)
        logger.info(f"Scheduled job {job_id} action={action.get('type')} schedule={schedule}")

        if save:
            self._save_jobs()

        return job_id

    def add_job(self, job_spec: Dict[str, Any]):
        return self._schedule_job(job_spec)

    def list_jobs(self):
        jobs = []
        for job in self.scheduler.get_jobs():
            md = job.kwargs.get('meta') if job.kwargs else {}
            jobs.append(md)
        return jobs

    def remove_job(self, job_id: str):
        try:
            self.scheduler.remove_job(job_id)
            self._save_jobs()
            return True
        except Exception as e:
            logger.error(f"Failed to remove job {job_id}: {e}")
            return False

# Single global manager instance
_manager = None

def get_scheduler():
    global _manager
    if _manager is None:
        _manager = SchedulerManager()
    return _manager