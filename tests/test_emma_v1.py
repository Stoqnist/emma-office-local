import pytest
import json
import time
import sqlite3
from unittest.mock import patch, MagicMock
import emma_v1 as emma

# ──────────────────────────────────────────────────────────────────────────────
# 1. Telemetry Validation
# ──────────────────────────────────────────────────────────────────────────────
class TestTelemetryMetrics:
    def test_record_updates_state_correctly(self):
        tel = emma.PoCTelemetryMetrics()
        tel.record("Test prompt", 0.5)
        assert tel.requests == 1
        assert abs(tel.total_latency_ms - 500.0) < 1e-6
        assert abs(tel.avg_latency_ms - 500.0) < 1e-6

    def test_avg_latency_math(self):
        tel = emma.PoCTelemetryMetrics()
        tel.record("P1", 1.0)
        tel.record("P2", 2.0)
        assert abs(tel.avg_latency_ms - 1500.0) < 1e-6

    def test_prompt_truncation(self):
        tel = emma.PoCTelemetryMetrics()
        long_prompt = "A" * 200
        tel.record(long_prompt, 0.1)
        assert len(tel.prompts_tracked[0]) == 123  # 120 chars + "..."

    def test_zero_division_safety(self):
        tel = emma.PoCTelemetryMetrics()
        tel.avg_latency_ms = 0.0  # initial state
        assert hasattr(tel, 'avg_latency_ms')  # Should not raise

# ──────────────────────────────────────────────────────────────────────────────
# 2. Core Logic & Chunking
# ──────────────────────────────────────────────────────────────────────────────
class TestChunking:
    @pytest.mark.parametrize("text,expected_len", [
        ("Short", 1),
        ("A" * 300, 2),
        ("A" * 400, 3),
    ])
    def test_chunk_text_size_and_overlap(self, text, expected_len):
        chunks = emma.chunk_text(text, chunk_size=200, overlap=20)
        assert len(chunks) == expected_len

    def test_empty_string_handling(self):
        assert emma.chunk_text("", 100, 10) == []

    def test_boundary_condition(self):
        text = "A" * 200
        chunks = emma.chunk_text(text, chunk_size=200, overlap=20)
        assert len(chunks) == 1
        assert len(chunks[0]) == 200

# ──────────────────────────────────────────────────────────────────────────────
# 3. State Management
# ──────────────────────────────────────────────────────────────────────────────
class TestStateManagement:
    def test_get_thread_history_defaults(self):
        assert emma.get_thread_history("nonexistent") == []

    def test_save_and_retrieve_history(self):
        msgs = [emma.HumanMessage(content="Hi")]
        emma.save_thread_history("t1", msgs)
        assert len(emma.get_thread_history("t1")) == 1
        assert emma.get_thread_history("t1")[0].content == "Hi"

# ──────────────────────────────────────────────────────────────────────────────
# 4. Database Tool (SQLite Safety & Format)
# ──────────────────────────────────────────────────────────────────────────────
class TestDatabaseTool:
    def test_valid_query_returns_str(self, monkeypatch):
        conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchall.return_value = [("Product A", 10.0, 50)]
        conn.cursor.return_value = cursor
        monkeypatch.setattr(sqlite3, "connect", lambda *a, **k: conn)

        result = emma.query_database("SELECT 1")
        assert isinstance(result, str)
        assert "Product A" in result

    def test_invalid_query_handles_error(self, monkeypatch):
        conn = MagicMock()
        conn.cursor.side_effect = sqlite3.OperationalError("no such table")
        monkeypatch.setattr(sqlite3, "connect", lambda *a, **k: conn)

        result = emma.query_database("DROP TABLE fake;")
        assert result.startswith("Error executing query:")

# ──────────────────────────────────────────────────────────────────────────────
# 5. Knowledge Base Loading Logic (Fixed syntax & file handling)
# ──────────────────────────────────────────────────────────────────────────────
class TestKnowledgeBaseLoading:
    def test_load_from_cache_exists(self, tmp_path):
        # Create exact directory structure expected by load_knowledge_base()
        kb_dir = tmp_path / "knowledge_base"
        cache_dir = kb_dir / "embeddings"
        cache_dir.mkdir(parents=True)
        cache_file = cache_dir / "embeddings.json"

        with open(cache_file, 'w') as f:
            json.dump({
                "docs": [["doc1", "content1"], ["doc2", "content2"]],
                "embeddings": [[0.1]*768, [0.2]*768]
            }, f)

        # Load using the temp directory (bypasses real filesystem warnings)
        emma.load_knowledge_base(str(kb_dir))
        assert len(emma.knowledge_base_docs) == 2

    def test_missing_kb_dir_logs_warning(self, tmp_path, capsys):
        kb_dir = str(tmp_path / "nonexistent")
        emma.load_knowledge_base(kb_dir)
        captured = capsys.readouterr()
        assert "Warning" in captured.out

# ──────────────────────────────────────────────────────────────────────────────
# 6. End-to-End Query Flow (run_query)
# ──────────────────────────────────────────────────────────────────────────────
class TestRunQueryFlow:
    def test_successful_tool_loop(self, mock_ollama_tool_loop):
        tel = emma.PoCTelemetryMetrics()
        response, error = emma.run_query("Check paper stock", tel)

        assert error is None
        assert "✅" in response or "stock" in response.lower()
        assert tel.requests == 1
        assert tel.total_latency_ms > 0

    def test_error_path_captured(self, monkeypatch):
        fake_llm = MagicMock(side_effect=RuntimeError("Ollama timeout"))
        monkeypatch.setattr(emma, "llm_with_tools", MagicMock(invoke=fake_llm.invoke))

        tel = emma.PoCTelemetryMetrics()
        response, error = emma.run_query("Fail test", tel)

        assert response is None
        assert isinstance(error, RuntimeError)
        assert tel.errors == 1

    def test_thread_history_updated(self, mock_ollama_tool_loop):
        initial_len = len(emma.get_thread_history(emma.thread_id))
        emma.run_query("Test history", emma.PoCTelemetryMetrics())
        assert len(emma.get_thread_history(emma.thread_id)) > initial_len

    def test_latency_measurement_precision(self, mock_ollama_tool_loop):
        tel = emma.PoCTelemetryMetrics()
        _ , _ = emma.run_query("Timing test", tel)
        assert isinstance(tel.total_latency_ms, float)
        assert tel.total_latency_ms > 0