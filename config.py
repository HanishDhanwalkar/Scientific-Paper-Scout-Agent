import os
from dotenv import load_dotenv

load_dotenv() 

DOWNLOADS_DIR = "./Downloads"

# Set the desired LLM provider and model here.
# Options: "ollama", "openai", "claude", "gemini"
CURRENT_LLM_PROVIDER = "ollama"  # Change this to switch LLMs
CURRENT_SUMMARIZER_LLM_PROVIDER = "ollama"  # Change this to switch LLMs


OLLAMA_MODEL = "llama3.2" # run ollama list to see local models
OPENAI_MODEL = "gpt-4o" # "gpt-4o", "gpt-4.1", "gpt-3.5-turbo"
CLAUDE_MODEL = "claude-opus-4-20250514" # "claude-opus-4-20250514" "claude-sonnet-4-20250514", "claude-3-7-sonnet-20250219", "claude-3-5-haiku-20241022"
GEMINI_MODEL = "gemini-2.5-flash" # "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-pro-preview-03-25", "gemini-2.5-flash-preview-04-17"


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") # For Claude
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # For Gemini
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") # Default for Ollama




####### LIST gemini models #####################
# from google import genai
# client = genai.Client(api_key=GOOGLE_API_KEY)

# print("List of models that support generateContent:\n")
# for m in client.models.list():
#     for action in m.supported_actions:
#         if action == "generateContent":
#             print(m.name)

# print("List of models that support embedContent:\n")
# for m in client.models.list():
#     for action in m.supported_actions:
#         if action == "embedContent":
#             print(m.name)