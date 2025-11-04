import json
from src.main import create_app


def test_rag_summarize_requires_json():
    app = create_app()
    client = app.test_client()
    resp = client.post('/api/rag/summarize')
    assert resp.status_code == 415


def test_rag_summarize_returns_structure():
    app = create_app()
    client = app.test_client()
    payload = {'question': 'Count laptops in Nasarawa'}
    resp = client.post('/api/rag/summarize', data=json.dumps(payload), content_type='application/json')
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data, dict)
    assert 'summary' in data and 'tables' in data and 'rules' in data


def test_presentation_word_requires_json():
    app = create_app()
    client = app.test_client()
    resp = client.post('/api/presentation/word')
    assert resp.status_code == 415


def test_presentation_word_generates_docx(monkeypatch):
    app = create_app()
    client = app.test_client()

    # Monkeypatch Document to a minimal fake that supports add_heading/add_paragraph/add_table/save
    class DummyTable:
        def __init__(self, rows, cols):
            self.rows = [[None]*cols]
        def add_row(self):
            self.rows.append([None]*len(self.rows[0]))
            class Row:
                pass
            return self.rows[-1]

    class DummyDoc:
        def __init__(self):
            self._content = []
        def add_heading(self, t, level=1):
            self._content.append(('h', t))
        def add_paragraph(self, t):
            self._content.append(('p', t))
        def add_table(self, rows, cols):
            return DummyTable(rows, cols)
        def save(self, fp):
            # write a tiny placeholder
            try:
                fp.write(b'PK\x03\x04')
            except Exception:
                pass

    # Patch Document in the routes module
    import src.web.routes as routes_mod
    monkeypatch.setattr(routes_mod, 'Document', DummyDoc)

    payload = {
        'title': 'Test Export',
        'summary': 'A short summary',
        'table': {
            'headers': ['col1', 'col2'],
            'rows': [[1, 'a'], [2, 'b']]
        }
    }
    resp = client.post('/api/presentation/word', data=json.dumps(payload), content_type='application/json')
    assert resp.status_code == 200
    # Content-type should be docx
    assert 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' in resp.content_type
