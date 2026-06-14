![Python versions](https://img.shields.io/pypi/pyversions/mkdocs-badges)

# emma-office-local
PoC software prototype where LLM is running locally in ollama and is called programatically with python code and langchain telemetrics integration

### 📦 Prerequisites
```bash
# 1. Pull DeepSeek R1 via Ollama
ollama pull deepseek-r1:latest  # or verify with `ollama list`

# 2. Install Python dependencies
pip install langchain-core langchain-ollama langsmith python-dotenv
```

### 🔑 Key Design Choices for a PoC
| Component | Why This Approach |
|-----------|-------------------|
| **`PoCTelemetryMetrics`** | Self-contained, no API keys or cloud deps. Captures latency & prompt volume per 
request. Easily swappable to LangSmith/DB later. |
| **LangChain Expression Language (`|`)** | Declarative pipeline. Minimal boilerplate, composable, and standard 
across LangChain versions. |
| **`ChatOllama` + `base_url`** | Explicit Ollama endpoint handling. Works with local dev or remote Ollama 
instances. |
| **Time Capture via `perf_counter()`** | High-resolution timing outside the chain to avoid callback overhead in a 
PoC. |

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

### ⚠️ Troubleshooting
- `ConnectionRefusedError` → Ollama isn't running or port is blocked: `ollama serve`
- `ModelNotFoundError` → Verify exact model tag: `ollama list | grep deepseek`
- High latency/VRAM issues → Use quantized models like `deepseek-r1:7b-q4_K_M`
