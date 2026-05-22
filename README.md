# AI Agent Toolkit

A Python toolkit for building AI agents powered by [Anthropic Claude](https://www.anthropic.com/), with support for tool use and real-time data queries.

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

Copy the `.env` template and fill in your API key:

```bash
cp .env .env.local   # optional, or edit .env directly
```

Edit `.env`:

```env
ANTHROPIC_API_KEY=your_api_key_here
```

> Get your API key from [console.anthropic.com](https://console.anthropic.com/).

## Usage

Run the agent with `uv`:

```bash
uv run ai-agent-toolkit
```

Add a new dependency:

```bash
uv add <package-name>
```

Run a one-off script without modifying the project:

```bash
uv run --with <package-name> python your_script.py
```

## Project Structure

```
ai-agent-toolkit/
├── src/
│   └── ai_agent_toolkit/
│       └── __init__.py   # main entry point & agent logic
├── .env                  # API keys (not committed to git)
├── pyproject.toml        # project metadata & dependencies
└── uv.lock               # locked dependency versions
```

## Development

All project metadata is defined in `pyproject.toml`. The package source lives under `src/ai_agent_toolkit/`.

To add a new dependency:

```bash
uv add <package>
```

## License

Add license information here.
