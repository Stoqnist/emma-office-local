import time
import sqlite3
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np
from dotenv import load_dotenv

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# 1. Telemetry Collector (PoC-optimized, no external services required)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class PoCTelemetryMetrics:
    """Self-contained telemetry tracker for prototype validation."""
    requests: int = 0
    total_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    prompts_tracked: List[str] = field(default_factory=list)
    errors: int = 0

    def record(self, prompt: str, latency_sec: float):
        self.requests += 1
        self.total_latency_ms += latency_sec * 1000
        self.avg_latency_ms = self.total_latency_ms / max(1, self.requests)
        # Truncate for PoC logging; store full in production DB/Redis
        self.prompts_tracked.append(prompt[:120] + "..." if len(prompt) > 120 else prompt)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Local LLM & Embedding Configuration
# ──────────────────────────────────────────────────────────────────────────────
# Prerequisites: ollama pull deepseek-r1:latest && ollama pull nomic-embed-text
llm = ChatOllama(
    model="deepseek-r1:latest",
    temperature=0.3,
    base_url="http://localhost:11434"
)

embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")

# ──────────────────────────────────────────────────────────────────────────────
# 3. State & Configuration (PoC-optimized)
# ──────────────────────────────────────────────────────────────────────────────
thread_id = "default"
thread_store: dict[str, list] = {}
knowledge_base_docs: List[Tuple[str, str]] = []
knowledge_base_embeddings: List[List[float]] = []

def get_thread_history(thread_id: str) -> list:
    return thread_store.get(thread_id, [])

def save_thread_history(thread_id: str, messages: list):
    thread_store[thread_id] = messages

# ──────────────────────────────────────────────────────────────────────────────
# 4. Business Logic & System Prompt (Preserved verbatim)
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Emma, a customer support specialist for OfficeFlow Supply Co., a paper and office supplies 
distribution company serving small-to-medium businesses across North America.

ABOUT YOUR ROLE:
You're part of the Customer Experience team and have been with OfficeFlow for 3 years. You're known for being helpful, 
efficient, and genuinely caring about solving customer problems. Your manager emphasizes that every interaction is an 
opportunity to build trust and loyalty.

WHAT YOU CAN HELP WITH:
✓ Product Information - Answer questions about our catalog of office supplies, paper products, writing instruments, 
organizational tools, and desk accessories
✓ Inventory & Availability - Check current stock levels and help customers find what they need
✓ Product Recommendations - Suggest products based on customer needs, usage patterns, and budget
✓ General Inquiries - Handle questions about our company, product lines, and services

WHAT YOU CANNOT DIRECTLY HANDLE:
✗ Order Placement - While you can provide product info, actual ordering is done through our web portal or by contacting 
our sales team at sales@officeflow.com
✗ Order Status & Tracking - Direct customers to check their account portal or contact fulfillment@officeflow.com
✗ Returns & Refunds - These require approval from our Returns Department at returns@officeflow.com
✗ Account Changes - Billing, payment methods, and account settings must go through accounts@officeflow.com
✗ Technical Support - For website issues, direct to support@officeflow.com

YOUR COMMUNICATION STYLE:
- Warm and professional, never robotic or overly formal
- Use natural language - "I'd be happy to help" instead of "I will assist you"
- Show empathy when customers are frustrated
- Be specific and accurate with information
- If you don't know something, be honest and direct them to the right resource
- Use the customer's name if they provide it
- Keep responses concise but thorough

IMPORTANT - CHECK DATABASE FIRST:
When customers ask about products or inventory, ALWAYS check the database FIRST before asking clarifying questions. Give 
them useful information about what you find, rather than asking for more details upfront. For example, if a customer 
asks "do you have any paper?" - check what paper products are in stock and tell them what's available, don't ask "what 
type of paper are you looking for?"

INTERACTION GUIDELINES:
1. Always greet customers warmly and acknowledge their question
2. Ask clarifying questions only if truly necessary AFTER checking available information
3. Provide complete, accurate information about products and availability
4. If recommending products, explain why they're a good fit
5. End conversations by checking if they need anything else
6. When you can't help directly, provide the specific contact or resource they need
7. Never make up information - if you're unsure, say so and offer to connect them with someone who knows

YOUR TOOLS:
You have access to two powerful tools to help customers:

1. query_database - Use this for product-related questions:
   - Product availability and stock levels
   - Product prices and pricing information
   - Product details and specifications
   - Searching for specific items in inventory

2. search_knowledge_base - Use this for company policies and information:
   - Returns and refunds policies
   - Shipping and delivery information
   - Ordering process and payment methods
   - Store locations and contact information
   - Company background and general info
   - Business hours and holiday closures

Choose the right tool based on what the customer is asking about. For questions about specific products, use the 
database. For questions about policies, processes, or company information, use the knowledge base.

EXAMPLE INTERACTIONS:

Customer: "Do you have copy paper?"
You: "Yes, we do! We carry several types of copy paper. Are you looking for standard 8.5x11 inch letter size, or do you 
need a specific weight or finish? I can check what we have in stock."

Customer: "I need to return an order"
You: "I understand you need to process a return. While I can't handle returns directly, our Returns Department will be 
happy to help you. You can reach them at returns@officeflow.com or call 1-800-OFFICE-1 ext. 3. They typically respond 
within 4 business hours. Do you need any other information I can help with?"

Customer: "What's the best pen for signing documents?"
You: "For document signing, I'd recommend a pen with archival-quality ink that won't fade over time. Let me check what 
we have available that would work well for that purpose."

Remember: You represent OfficeFlow's commitment to excellent customer service. Be helpful, honest, and human in every 
interaction."""

db_path = str(Path(__file__).parent / 'inventory' / 'inventory.db')

@tool
def query_database(query: str) -> str:
    """Execute SQL query against the inventory database for products, quantities, and prices."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()
        return str(results)
    except Exception as e:
        return f"Error executing query: {str(e)}"

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 20) -> List[str]:
    """Split text into chunks with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks

def load_knowledge_base(kb_dir: str = "./knowledge_base") -> None:
    """Load knowledge base documents and embeddings from cache or generate them."""
    global knowledge_base_docs, knowledge_base_embeddings

    kb_path = Path(kb_dir) / "documents"
    cache_path = Path(kb_dir) / "embeddings" / "embeddings.json"

    if cache_path.exists():
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
        knowledge_base_docs = [tuple(doc) for doc in cache_data["docs"]]
        knowledge_base_embeddings = cache_data["embeddings"]
        print(f"📚 Knowledge base loaded from cache: {len(knowledge_base_docs)} chunks")
        return

    if not kb_path.exists():
        print(f"⚠️ Warning: Knowledge base directory '{kb_dir}' not found")
        return

    chunks = []
    for file_path in kb_path.glob("*.md"):
        if file_path.name == "CHUNKING_NOTES.md":
            continue
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            file_chunks = chunk_text(content)
            for i, chunk in enumerate(file_chunks):
                chunks.append((f"{file_path.name}:chunk_{i}", chunk))

    if not chunks:
        print(f"⚠️ Warning: No documents found in '{kb_dir}'")
        return

    knowledge_base_docs = chunks
    print(f"🧠 Generating local embeddings for {len(chunks)} chunks...")

    contents = [c[1] for c in chunks]
    knowledge_base_embeddings = embeddings.embed_documents(contents)
    print(f"✅ Knowledge base loaded: {len(chunks)} chunks indexed locally")

@tool
def search_knowledge_base_tool(query: str) -> str:
    """Search company knowledge base for policies, procedures, contact info, and business hours."""
    return search_knowledge_base(query)

tools = [query_database, search_knowledge_base_tool]
llm_with_tools = llm.bind_tools(tools)

# ──────────────────────────────────────────────────────────────────────────────
# 5. Execution & Telemetry Capture (Exact signature from PoC)
# ──────────────────────────────────────────────────────────────────────────────
def run_query(q: str, telemetry: PoCTelemetryMetrics):
    start = time.perf_counter()
    try:
        history = get_thread_history(thread_id)
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + history + [HumanMessage(content=q)]

        current_messages = messages.copy()
        final_content = ""

        # Tool-calling loop (local Ollama native function calling)
        while True:
            response = llm_with_tools.invoke(current_messages)

            if not hasattr(response, 'tool_calls') or not response.tool_calls:
                final_content = response.content or "I'm sorry, I couldn't process that request."
                break

            current_messages.append(response)

            for tc in response.tool_calls:
                func_args = json.loads(tc["args"]) if isinstance(tc["args"], str) else tc["args"]
                query_arg = func_args.get("query", "") if isinstance(func_args, dict) else ""

                if tc["name"] == "query_database":
                    result = query_database.invoke({"query": query_arg})
                elif tc["name"] == "search_knowledge_base_tool":
                    result = search_knowledge_base_tool.invoke({"query": query_arg})
                else:
                    result = f"Error: Unknown tool {tc['name']}"

                current_messages.append(ToolMessage(tool_call_id=tc["id"], content=result, name=tc["name"]))

        save_thread_history(thread_id, current_messages)

        latency = time.perf_counter() - start
        telemetry.record(q, latency)
        return final_content, None

    except Exception as e:
        latency = time.perf_counter() - start
        telemetry.record(q, latency)
        telemetry.errors += 1
        return None, e

# ──────────────────────────────────────────────────────────────────────────────
# 6. Run PoC
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load KB on startup (sync for PoC stability)
    load_knowledge_base()

    telemetry = PoCTelemetryMetrics()

    print("\n🤖 OfficeFlow Support Agent initialized.")
    print("💡 Type 'quit' to exit and view telemetry summary.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        if not user_input:
            continue

        print(f"📥 Query: {user_input}")
        response, error = run_query(user_input, telemetry)

        if error:
            print(f"❌ Error: {error}\n")
        else:
            print(f"✅ Agent Response:\n{response}\n")

    # ──────────────────────────────────────────────────────────────────────
    print("\n📊 PoC Telemetry Summary:")
    print(f"   Total Requests : {telemetry.requests}")
    print(f"   Avg Latency    : {telemetry.avg_latency_ms:.2f} ms")
    print(f"   Errors         : {telemetry.errors}")
    if telemetry.prompts_tracked:
        print(f"   Sample Tracked : {telemetry.prompts_tracked[-1][:80]}...")

    # Optional: Export to CSV/JSON for downstream analysis
    # import json; json.dump(telemetry.__dict__, open("poc_telemetry.json","w"), indent=2)