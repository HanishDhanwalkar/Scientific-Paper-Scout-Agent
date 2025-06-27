# Kairos Intern Take‑Home: **Scientific‑Paper Scout**

## 0 — Goal

Build an AI "Scientific‑Paper Scout" agent that helps a user discover and summarise recent research papers on any topic. The user interacts with the agent through a **command‑line chat** (`python main.py`). A tiny React front‑end is an **optional stretch goal**.

Everything should run locally on a typical laptop with **Python 3.x**. You may use `venv`, `poetry`, or `conda`—your choice. **Docker is *not* required.**

---

## 1 — Architecture

| Component                                                                                       | Tech                                         | Required behaviour                                                                                                                                                                                                                                                 |
| ----------------------------------------------------------------------------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Agent Host**                                                                                  | Python 3.x · FastAPI (optionally)           | Receives user messages, decides when/with which arguments to call each MCP server, streams partial responses. Logs every tool call (name, args, timestamps, outcome). **Must be model-agnostic**: should work with any LLM provider (OpenAI, Anthropic, Google, etc.) by simply changing model configuration—no code changes required.                                                                                              |
| **`paper_search` MCP server**                                                                   | Python or Typescript MCP server                       | **Input**: `{ "query": "string", "max_results": int }` → Queries the public arXiv API (no key) and returns ≤ `max_results` items.                                                                           |
| **`pdf_summarize` MCP server**                                                                  | Python or Typescript MCP server                      | **Input**: `{ "pdf_url": "string" }` → Downloads PDF, extracts text and summarises using configurable cloud LLM provider (OpenAI, Anthropic, Google, etc.). **Output**: `{ "summary": "string" }`. Model selection should be configurable via environment variables or config file. |
| **Interface**                                                                                   | CLI chat (required) · Vite + React (stretch) | Shows streamed assistant responses **and prints each tool call** (name, args, latency) to the console.                                                                                                                                                             |

## 2 — Model Agnosticism Requirements

The system should support multiple cloud LLM providers through a unified interface:
- **OpenAI** 
- **Anthropic** 
- **Gemini** 

**Configuration should be environment-based** (e.g., `.env` file) with parameters like:
- LLM_PROVIDER (e.g. openai, anthropic, gemini)
- LLM_MODEL (e.g. claude-4-sonnet-20250514, etc)

Switching providers should require **zero code changes**—only configuration updates.

---

Please feel free to ask any clarifying questions.

**Good luck & have fun!**


