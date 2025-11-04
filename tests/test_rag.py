import os
import json
from src.rag.manager import RAGManager


def test_get_compact_context_requires_schema_context():
    rag = RAGManager()
    # Ensure schema_context is None
    rag.schema_context = None
    res = rag.get_compact_context('show latest records')
    assert isinstance(res, dict)
    # When no schema_context, enhancer should request clarification
    assert res.get('clarify') is True


def test_get_compact_context_defaults_structure(tmp_path):
    rag = RAGManager()
    # Provide a dummy schema_context object with minimal API used by RAGManager
    class DummySchema:
        def get_context(self):
            return {'tables': [], 'relationships': [], 'columns': []}

        def get_table_info(self, name):
            return None

    rag.schema_context = DummySchema()
    res = rag.get_compact_context('who are the users?')
    assert isinstance(res, dict)
    # Even with empty schema context, the function should return the expected keys
    assert set(['summary', 'tables', 'rules', 'clarify', 'candidates']).issubset(set(res.keys()))
