# Scientific-Paper-Scout-Agent

## Installation

Clone the repository and create a virtual environment 

`git clone https://github.com/HanishDhanwalkar/Scientific-Paper-Scout-Agent.git`

`cd scientific-paper-scout-agent`

`python -m venv .venv`

for linux : `source .venv/bin/activate`\
for windows : `.\.venv\Scripts\activate` 

`pip install -r requirements.txt`

Rename `.env_EXAMPLE` to `.env` and paste your api keys. 
(Optional: leave as it is for models that won't be used)

Configure LLMs in config.py file\
edit (in config.py):\
`CURRENT_LLM_PROVIDER = "ollama"`  # Change this to switch LLMs,  Options: `"ollama"`, `"openai"`, `"claude"`, `"gemini"`\
`CURRENT_SUMMARIZER_LLM_PROVIDER = "ollama"`  # Change this to switch LLMs,  Options: `"ollama"`, `"openai"`, `"claude"`, `"gemini"`

## Usage

For CLI usage run:
`python main.py`

For Web UI chatbot run: 
`streamlit run app.py`


## ScreenShots
### CLI tool:

<img src="./assets/Screenshot 2025-06-28 235621.png" width=1000px>
<img src="./assets/Screenshot 2025-06-28 235816.png" width=1000px>
<img src="./assets/Screenshot 2025-06-28 235859.png" width=1000px>