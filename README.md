# AI Agent Toolkit

A Python toolkit for building AI agents powered by [Anthropic Claude](https://www.anthropic.com/), featuring raw API usage, [LangGraph](https://langchain-ai.github.io/langgraph/)-based single agents, and multi-agent workflows.

## Requirements

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) (package manager)

## Installation

Clone the repo and install dependencies using `uv`:

```bash
git clone https://github.com/libaice/ai-agent-toolkit.git
cd ai-agent-toolkit
uv sync
```

## Configuration

Edit `.env` and fill in your API key:

```env
ANTHROPIC_API_KEY=your_api_key_here
```

> Get your API key from [console.anthropic.com](https://console.anthropic.com/).

## Usage

### Basic LLM Agent (`__init__.py`)

Demonstrates raw Anthropic API usage with a manual tool-use loop:

```bash
uv run ai-agent-toolkit
```

### LangGraph Single Agent (`langgraph_agent.py`)

A ReAct-style agent built with LangGraph + LangChain Anthropic, using `app.stream()` to observe each node's output step by step:

```bash
uv run python src/ai_agent_toolkit/langgraph_agent.py
```

### Multi-Agent Workflow (`multi_agent.py`)

A multi-agent system where specialized sub-agents (market data search, report writing, email sending) collaborate via a LangGraph StateGraph:

```bash
uv run python src/ai_agent_toolkit/multi_agent.py
```

### Add a dependency

```bash
uv add <package-name>
```

## Project Structure

```
ai-agent-toolkit/
├── src/
│   └── ai_agent_toolkit/
│       ├── __init__.py          # Basic Anthropic API + manual tool-use loop
│       ├── langgraph_agent.py   # LangGraph single ReAct agent with streaming
│       └── multi_agent.py       # Multi-agent workflow with specialized sub-agents
├── .env                         # API keys (not committed to git)
├── pyproject.toml               # Project metadata & dependencies
└── uv.lock                      # Locked dependency versions
```

## Development

All project metadata is defined in `pyproject.toml`. The package source lives under `src/ai_agent_toolkit/`.

To add a new dependency:

```bash
uv add <package>
```

## License

Add license information here.
