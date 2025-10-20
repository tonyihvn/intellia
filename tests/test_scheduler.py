from src.scheduler import parser


def test_parse_every_monday_email():
    prompt = "Send an email every Monday at 09:00 to team@example.com with subject 'Weekly' body 'Report attached'"
    parsed = parser.parse(prompt)
    assert parsed['action']['type'] == 'send_email'
    assert any('team@example.com' in t for t in parsed['action']['to'])
    assert 'cron' in parsed['schedule']


def test_parse_on_date_call_api():
    prompt = "On 2025-10-21 14:00 call api POST https://httpbin.org/post with body {\"foo\":\"bar\"}"
    parsed = parser.parse(prompt)
    assert parsed['action']['type'] == 'call_api'
    assert parsed['schedule']['date'].startswith('2025-10-21')