# AI Agent Toolkit

A step-by-step learning project for building production-grade AI agents on top of **Claude (Anthropic)**, progressing from a bare LLM call all the way to a RAG-powered agent with persistent memory and automated evaluation.

## Learning Path

This toolkit is structured as a progression. Each stage builds on the previous one:

```
Stage 1  →  Stage 2  →  Stage 3  →  Stage 4  →  Stage 5  →  Stage 6
Raw LLM     Tools       LangGraph   Memory      RAG         Evaluation
```

---

## Requirements

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) — fast Python package manager

## Installation

```bash
git clone https://github.com/libaice/ai-agent-toolkit.git
cd ai-agent-toolkit
uv sync
```

## Configuration

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=your_api_key_here
```

> Get your key from [console.anthropic.com](https://console.anthropic.com/).

---

## Stage 1 — Raw LLM Call

**File:** [`src/ai_agent_toolkit/__init__.py`](src/ai_agent_toolkit/__init__.py)

The simplest possible starting point: call the Anthropic API directly with no framework. Implements a **manual tool-use loop** — send message, check if Claude wants to call a tool, execute it, send the result back.

**Concepts covered:**
- `anthropic.Anthropic()` client setup
- Message structure (`role: user/assistant`)
- Tool definitions and `tool_use` content blocks
- Manual agentic loop

```bash
uv run ai-agent-toolkit
```

---

## Stage 2 — Tools

**File:** [`src/ai_agent_toolkit/tools.py`](src/ai_agent_toolkit/tools.py)

Defines three real-world tools that query the **Polymarket Gamma API** for live prediction market data. These tools are reused by all subsequent agents.

| Tool | Description |
|------|-------------|
| `search_markets(query, limit)` | Search active markets by keyword |
| `get_trending_markets(limit)` | Get markets sorted by volume |
| `get_market_detail(market_id)` | Get full details for a specific market |

**Concepts covered:**
- `@tool` decorator from LangChain
- Typed function signatures as tool schemas
- Calling external REST APIs from tools
- Graceful error handling for missing fields

---

## Stage 3 — LangGraph ReAct Agent

**Files:** [`agent.py`](src/ai_agent_toolkit/agent.py) · [`langgraph_agent.py`](src/ai_agent_toolkit/langgraph_agent.py)

Replace the manual loop with a proper **LangGraph StateGraph**. The graph handles the tool-call cycle automatically using `ToolNode`.

```
[Entry] → llm → should_continue?
                    ├── "tools" → ToolNode → llm (loop)
                    └── END
```

**Concepts covered:**
- `StateGraph` and `AgentState` with `TypedDict`
- `add_messages` reducer for message accumulation
- `ToolNode` from `langgraph.prebuilt`
- Conditional edges with `should_continue`
- `llm.bind_tools(tools)` for automatic tool-calling

```bash
uv run src/ai_agent_toolkit/agent.py
```

The streaming variant (`langgraph_agent.py`) uses `app.stream()` to observe each graph node's output step by step.

---

## Stage 4 — Agent with Persistent Memory

**Files:** [`langgraph_memory.py`](src/ai_agent_toolkit/langgraph_memory.py) · [`main.py`](src/ai_agent_toolkit/main.py)

Extends the ReAct agent with **two memory mechanisms**:

1. **Short-term memory** — `MemorySaver` checkpointer stores the full conversation state per `thread_id`, enabling multi-turn conversations
2. **Long-term user profile** — A dedicated `update_memory` node extracts user preferences (preferred assets, risk tolerance, capital) from the conversation after each turn and persists them in the state

```
[Entry] → llm → should_continue?
                    ├── "tools" → ToolNode → llm (loop)
                    └── "update_memory" → END
```

**State structure:**
```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # conversation history
    user_profile: dict                        # extracted preferences
```

**Concepts covered:**
- `MemorySaver` checkpointer for cross-turn persistence
- `thread_id` for per-user session isolation
- Profile extraction with a second LLM call
- Message trimming to prevent token accumulation (rate limit mitigation)
- System prompt injection of user preferences

```bash
uv run src/ai_agent_toolkit/main.py
```

> **Note:** The script adds 35-second delays between rounds to stay within Anthropic's 30K token/minute rate limit. The `MemorySaver` accumulates full conversation history, which can make later rounds token-heavy.

---

## Stage 5 — RAG + Agent (Knowledge-Augmented Agent)

**Files:** [`rag.py`](src/ai_agent_toolkit/rag.py) · [`agent_with_rag.py`](src/ai_agent_toolkit/agent_with_rag.py)

Adds a local **vector knowledge base** so the agent can answer questions from internal documentation, not just live API data.

### Step 5a — Build the Knowledge Base (`rag.py`)

```
Documents → Chunker → Embeddings Model → Chroma Vector Store
```

| Component | Implementation |
|-----------|---------------|
| Documents | GhostGuard guide, Polymarket V2 tech notes, risk management guide |
| Splitter | `RecursiveCharacterTextSplitter` (500 chars, 50 overlap) |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (multilingual, local, ~470MB) |
| Vector Store | Chroma, persisted to `./knowledge_base/` |
| Exposed as | `search_knowledge_base` LangChain `@tool` |

```bash
uv run src/ai_agent_toolkit/rag.py
# Downloads embedding model on first run (~1 min)
# Subsequent runs load from cache instantly
```

### Step 5b — Agent with RAG Tool (`agent_with_rag.py`)

The agent now has **three tools** and decides which to use based on the question type:

| Tool | When to use |
|------|-------------|
| `search_markets` | User asks about active prediction markets |
| `get_trending_markets` | User asks what's popular right now |
| `search_knowledge_base` | User asks about Polymarket V2 tech, GhostGuard, risk management |

```bash
uv run src/ai_agent_toolkit/agent_with_rag.py
```

**Example interaction:**
- *"Polymarket V2 有什么风险？"* → agent calls `search_knowledge_base` → answers from docs
- *"推荐几个安全的市场"* → agent calls `get_trending_markets` → answers from live data

---

## Stage 6 — Agent Evaluation

**File:** [`evaluator.py`](src/ai_agent_toolkit/evaluator.py)

An automated evaluation framework that tests both **tool selection** (did the agent call the right tools?) and **answer quality** (is the response accurate and relevant?).

### Test Cases

| ID | Question | Expected Tool | Reference Answer |
|----|----------|--------------|-----------------|
| `tc_001` | GhostGuard 的月费是多少 | `search_knowledge_base` | $299/月 |
| `tc_002` | 现在 BTC 相关市场有哪些 | `search_markets` | *(live data, no fixed answer)* |
| `tc_003` | V2 的签名风险怎么防 | `search_knowledge_base` | EIP-1271 + GhostGuard |
| `tc_004` | 查一下热门市场，结合风险给建议 | `get_trending_markets` + `search_knowledge_base` | *(combined)* |

### Evaluation Dimensions

**Tool usage score** (rule-based):
- Whether the agent called all expected tools
- Hit / missed / extra tool breakdown

**Answer quality score** (LLM-as-judge using Claude):
- Accuracy (vs. reference answer)
- Completeness
- Hallucination (5 = no fabrication, 1 = severe)

```bash
uv run src/ai_agent_toolkit/evaluator.py
# Outputs pass/fail per test case
# Saves full results to eval_results.json
```

**Latest results:**
```
✅ tc_001  Tool score: 1.0
✅ tc_002  Tool score: 1.0
✅ tc_003  Tool score: 1.0
✅ tc_004  Tool score: 1.0

────────────────────────────────────────
Pass rate: 4/4 (100%)
```

---

## Project Structure

```
ai-agent-toolkit/
├── src/
│   └── ai_agent_toolkit/
│       ├── __init__.py          # Stage 1: Raw Anthropic API + manual tool loop
│       ├── tools.py             # Stage 2: Polymarket API tools (@tool decorated)
│       ├── agent.py             # Stage 3: LangGraph ReAct agent
│       ├── langgraph_agent.py   # Stage 3: Streaming variant with step-by-step output
│       ├── langgraph_memory.py  # Stage 4: Agent with MemorySaver + user profile
│       ├── main.py              # Stage 4: Multi-turn conversation runner
│       ├── rag.py               # Stage 5a: Build vector knowledge base
│       ├── agent_with_rag.py    # Stage 5b: Agent with RAG tool
│       ├── evaluator.py         # Stage 6: Automated tool + answer quality eval
│       └── multi_agent.py       # Bonus: Multi-agent workflow
├── knowledge_base/              # Generated — Chroma vector store (gitignored)
├── eval_results.json            # Generated — evaluation output (gitignored)
├── .env                         # API keys (gitignored)
├── pyproject.toml               # Project metadata & dependencies
└── uv.lock                      # Locked dependency versions
```

---

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `anthropic` | Anthropic Claude API client |
| `langchain-anthropic` | LangChain wrapper for Claude |
| `langgraph` | Agent graph execution framework |
| `langchain-huggingface` | HuggingFace embedding models |
| `sentence-transformers` | Local multilingual embedding model |
| `langchain-chroma` | Chroma vector store integration |
| `langchain-community` | Community tools and utilities |
| `httpx` | HTTP client for Polymarket API |
| `python-dotenv` | `.env` file loading |

---

## Add a Dependency

```bash
uv add <package-name>
```

---

## Architecture Summary

```
User Input
    │
    ▼
┌─────────────────────────────────────────────┐
│              LangGraph Agent                │
│                                             │
│  ┌─────┐    ┌──────────┐    ┌────────────┐ │
│  │ llm │───▶│ToolNode  │───▶│update_mem  │ │
│  │     │◀───│          │    │(profile)   │ │
│  └─────┘    └──────────┘    └────────────┘ │
│      │                                      │
│  [system prompt]                            │
│  [user profile]  ←── MemorySaver           │
└─────────────────────────────────────────────┘
         │              │
         ▼              ▼
  Live API Tools    RAG Knowledge Base
  ─────────────     ─────────────────
  Polymarket        GhostGuard docs
  Gamma API         V2 tech notes
                    Risk management
```

---

## License

MIT
