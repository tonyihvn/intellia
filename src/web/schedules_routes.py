import json
import logging
from flask import Blueprint, request, jsonify
from ..llm.client import LLMClient
from ..scheduler.scheduler import get_scheduler
from ..scheduler import parser as nlparser

logger = logging.getLogger(__name__)

schedules_routes = Blueprint('schedules', __name__, url_prefix='/api')


@schedules_routes.route('/schedules/parse', methods=['POST'])
def parse_schedule():
    """Parse natural language schedule into structured job spec using LLM.

    Request JSON: {"prompt": "send an email every Monday at 9am to team@example.com about weekly report"}
    Response: {"job_spec": {...}}
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 415
    data = request.get_json()
    prompt = data.get('prompt')
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    # First try deterministic parser
    try:
        parsed = nlparser.parse(prompt)
        if parsed and parsed.get('action') and parsed.get('schedule'):
            return jsonify({'job_spec': {'action': parsed['action'], 'schedule': parsed['schedule']}})
    except Exception as e:
        logger.info(f"Deterministic parser failed or returned nothing: {e}")

    # Fallback to LLM-based parse
    client = LLMClient()
    parse_prompt = (
        "Parse the following user instruction into strict JSON with keys: action and schedule.\n"
        "action should be an object with 'type' (send_email or call_api) and parameters.\n"
        "schedule should be an object with either 'cron' (cron string), 'date' (ISO datetime), or 'interval' (seconds/minutes/hours).\n"
        "Respond ONLY with valid JSON and nothing else.\n\n"
        f"Instruction: {prompt}"
    )

    try:
        res = client.generate_sql(parse_prompt)  # Using generate_sql to call LLM; returns dict
        # The LLM may return 'full_response' containing JSON; extract text
        text = res.get('full_response') if isinstance(res, dict) else str(res)
        # Attempt to find JSON substring
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1:
            return jsonify({"error": "LLM did not return JSON"}), 500
        json_text = text[start:end+1]
        job_spec = json.loads(json_text)
        return jsonify({'job_spec': job_spec})
    except Exception as e:
        logger.error(f"Failed to parse schedule with LLM: {e}")
        return jsonify({"error": str(e)}), 500

@schedules_routes.route('/schedules', methods=['POST'])
def create_schedule():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 415
    data = request.get_json()
    job_spec = data.get('job_spec')
    if not job_spec:
        return jsonify({"error": "job_spec required"}), 400

    scheduler = get_scheduler()
    try:
        job_id = scheduler.add_job(job_spec)
        return jsonify({'success': True, 'job_id': job_id})
    except Exception as e:
        logger.error(f"Failed to create schedule: {e}")
        return jsonify({'error': str(e)}), 500

@schedules_routes.route('/schedules', methods=['GET'])
def list_schedules():
    scheduler = get_scheduler()
    jobs = scheduler.list_jobs()
    return jsonify({'jobs': jobs})

@schedules_routes.route('/schedules/<job_id>', methods=['DELETE'])
def delete_schedule(job_id):
    scheduler = get_scheduler()
    ok = scheduler.remove_job(job_id)
    return jsonify({'success': ok})
