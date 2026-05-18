# AskSwift

A local RAG chat CLI for asking questions about the Swift documentation.

## Setup

Install [Ollama](https://ollama.com/download) first.

Then run:

```bash
./scripts/setup.sh
```

This creates a virtual environment, installs the Python package, and pulls the
local Ollama models:

- `embeddinggemma`
- `qwen2.5-coder:3b`

Start the chat:

```bash
./askswift
```
