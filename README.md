![Python versions](https://img.shields.io/pypi/pyversions/mkdocs-badges)

# emma-office-local
PoC: DeepSeek R1 locally served via Ollama, orchestrated with LangChain & telemetry tracking

### 🔑 Key Design Choices for a PoC
| Component | Why This Approach |
|-----------|-------------------|
| **`PoCTelemetryMetrics`** | Self-contained, no API keys or cloud deps. Captures latency & prompt volume per request. Easily swappable to LangSmith/DB later. |
| **`LangChain Expression Language`** | Declarative pipeline. Minimal boilerplate, composable, and standard across LangChain versions. |
| **`ChatOllama` + `base_url`** | Explicit Ollama endpoint handling. Works with local dev or remote Ollama instances. |
| **Time Capture via `perf_counter()`** | High-resolution timing outside the chain to avoid callback overhead in a PoC. |

### 🚀 How to Run Tests

###### Install test dependencies
```shell
uv pip install pytest pytest-cov hypothesis
```

###### Run full suite with coverage
```shell
cd /Users/stoyan/Projects/emma-office-local
uv run pytest tests/test_emma_v1.py -v --tb=short --cov=src/emma_v1 --cov-report=term-missing
```

### 🔍 What This Suite Validates
| Area | Tests Covered |
|------|---------------|
| **Telemetry Math** | Latency accumulation, average calculation, division-by-zero safety, prompt truncation |
| **Chunking Logic** | Boundary conditions, overlap behavior, empty string handling |
| **State Isolation** | Thread history CRUD, global variable reset between tests |
| **DB Tool Safety** | Valid query format, SQL error handling, connection cleanup |
| **KB Caching** | Cache hit/miss flow, missing directory warnings |
| **LLM Loop & Telemetry Integration** | Tool-calling sequence simulation, error path capture, history persistence, latency tracking |

### 🔧 Next Steps / Customization
1. Replace mock responses in `mock_ollama_responses` with expected outputs from your merged logic if the tool-call structure changed.
2. Add `pytest.mark.slow` to LLM-heavy tests if you want to run them separately.
3. Want me to generate **Hypothesis-based property tests** for `chunk_text` or telemetry math? Or add **benchmark timing assertions**? Reply with your preference and I'll extend the suite in 
<2 minutes.


# 🚀 Setup & Run

#### 1. Pull local models
```bash
ollama pull deepseek-r1:latest
```
```bash
ollama pull nomic-embed-text
```
#### 2. Install deps (aligns with your pyproject.toml)
```bash
uv pip install -r requirements.txt
```

#### 3. Run PoC
```bash
uv run python src/emma_v1.py
```

### 🚀 Production Telemetry Upgrade Path
When moving from PoC → prod, replace the custom metric collector with:
```bash
pip install langsmith opentelemetry-instrumentation-langchain
```
```python
# Then simply set env vars and add tracing decorator/context manager:
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_API_KEY"] = "ls_xxxxxx"

from langsmith import traceable

@traceable(name="deepseek_r1_chain")
def run_query_prod(q: str):
    return chain.invoke({"query": q})
```
LangSmith will automatically capture latency, token usage, prompt/response logs, and error rates without code 
changes.

### 🔮 Next Steps for Production
- Swap `PoCTelemetryMetrics` → LangSmith or OpenTelemetry exporter
- Replace SQLite with Postgres/Supabase for concurrent access
- Add prompt caching via Ollama's `keep_alive=0` + Redis layer
- Containerize with multi-stage Dockerfile + GPU passthrough

### ⚠️ Troubleshooting
- `ConnectionRefusedError` → Ollama isn't running or port is blocked: `ollama serve`
- `ModelNotFoundError` → Verify exact model tag: `ollama list | grep deepseek`
- High latency/VRAM issues → Use quantized models like `deepseek-r1:7b-q4_K_M`

