import sys
from pathlib import Path

# Add src/ to Python path for tests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from unittest.mock import patch, MagicMock

@pytest.fixture(scope="session", autouse=True)
def mock_llm_dependencies():
    """Patch Ollama/LangChain BEFORE emma_v1 is imported by pytest."""
    with patch("langchain_ollama.ChatOllama") as MockLLM, \
            patch("langchain_ollama.OllamaEmbeddings") as MockEmbed:

        # Simulate successful initialization & tool binding
        mock_llm_instance = MagicMock()
        mock_llm_instance.bind_tools.return_value.invoke = MagicMock()
        MockLLM.return_value = mock_llm_instance

        mock_embed_instance = MagicMock()
        mock_embed_instance.embed_documents.return_value = [[0.1] * 768 for _ in range(2)]
        MockEmbed.return_value = mock_embed_instance

        yield

@pytest.fixture(autouse=True)
def reset_emma_state(monkeypatch, tmp_path):
    """Reset module-level globals & state before each test."""
    import emma_v1 as emma

    # Reset thread store & KB
    monkeypatch.setattr(emma, "thread_store", {})
    monkeypatch.setattr(emma, "knowledge_base_docs", [])
    monkeypatch.setattr(emma, "knowledge_base_embeddings", [])

    # Safe temp DB path
    db_path = str(tmp_path / "test_inventory.db")
    Path(db_path).touch()
    monkeypatch.setattr(emma, "db_path", db_path)

    yield

@pytest.fixture
def mock_ollama_tool_loop(monkeypatch):
    """Simulate LLM tool-calling -> final response loop."""
    import emma_v1 as emma
    call_count = [0]

    def fake_invoke(messages):
        call_count[0] += 1
        res = MagicMock()

        if call_count[0] == 1:
            # First turn: tool call
            res.content = None
            res.tool_calls = [{
                "id": "call_abc123",
                "name": "query_database",
                "args": {"query": "SELECT * FROM products LIMIT 1;"}
            }]
        else:
            # Second turn: final answer
            res.content = "✅ We have 42 units of A4 paper in stock."
            res.tool_calls = []
        return res

    monkeypatch.setattr(emma, "llm_with_tools", MagicMock(invoke=fake_invoke))
    yield emma.llm_with_tools
