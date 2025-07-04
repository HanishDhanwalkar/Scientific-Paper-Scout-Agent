# Scientific-Paper-Scout-Agent

## Installation

Clone the repository and create a virtual environment 

`git clone https://github.com/HanishDhanwalkar/Scientific-Paper-Scout-Agent.git`

`cd scientific-paper-scout-agent`

`python -m venv .venv`

for linux : `source .venv/bin/activate`\
for windows : `.\.venv\Scripts\activate` 

`pip install -r requirements.txt`

Paste your api keys in `.env_EXAMPLE` and Rename the file to `.env`
(Optional: leave as it is for models that won't be used)

Configure LLMs in config.py file\
edit (in config.py):\
`CURRENT_LLM_PROVIDER = "ollama"`  # Change this to switch LLMs,  Options: `"ollama"`, `"openai"`, `"claude"`, `"gemini"`\
`CURRENT_SUMMARIZER_LLM_PROVIDER = "ollama"`  # Change this to switch LLMs,  Options: `"ollama"`, `"openai"`, `"claude"`, `"gemini"`

## Usage
Run MCP server:\
`mcp run server.py`


| For CLI usage         |  For Web UI chatbot       |
| --------------------- | ------------------------- |
| `python main.py`      | `streamlit run app.py`    |


## ScreenShots
## CLI tool:

<img src="./assets/Screenshot 2025-06-28 235621.png" width=800px>
<img src="./assets/Screenshot 2025-06-28 235816.png" width=800px>
<img src="./assets/Screenshot 2025-06-28 235859.png" width=800px>

## Web UI (Using Streamlit)

<img src="./assets/Screenshot 2025-06-30 232050.png" width=800px>
<img src="./assets/Screenshot 2025-06-30 232128.png" width=800px>
<img src="./assets/Screenshot 2025-06-30 232312.png" width=800px>