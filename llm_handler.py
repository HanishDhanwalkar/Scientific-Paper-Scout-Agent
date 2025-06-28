import json
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.tools import tool

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# Import configuration
from config import (
    CURRENT_LLM_PROVIDER,
    CURRENT_SUMMARIZER_LLM_PROVIDER,
    OLLAMA_MODEL,
    OPENAI_MODEL,
    CLAUDE_MODEL,
    GEMINI_MODEL,
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    GOOGLE_API_KEY,
    OLLAMA_BASE_URL
)

def get_llm_client():
    """Initializes and returns the appropriate LLM client based on configuration."""
    if CURRENT_LLM_PROVIDER == "ollama":
        # print(f"Using Ollama model: {OLLAMA_MODEL}")
        return ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    elif CURRENT_LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in environment variables.")
        # print(f"Using OpenAI model: {OPENAI_MODEL}")
        return ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY)
    elif CURRENT_LLM_PROVIDER == "claude":
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not set in environment variables.")
        # print(f"Using Claude model: {CLAUDE_MODEL}")
        return ChatAnthropic(model=CLAUDE_MODEL, anthropic_api_key=ANTHROPIC_API_KEY)
    elif CURRENT_LLM_PROVIDER == "gemini":
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not set in environment variables.")
        # print(f"Using Gemini model: {GEMINI_MODEL}")
        return ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=GOOGLE_API_KEY)
    else:
        raise ValueError(f"Unsupported LLM provider: {CURRENT_LLM_PROVIDER}")
    
def get_summarizer_llm_client():
    """Initializes and returns the appropriate LLM client based on configuration."""
    if CURRENT_SUMMARIZER_LLM_PROVIDER == "ollama":
        # print(f"Using Ollama model: {OLLAMA_MODEL} as summarizer")
        return ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    elif CURRENT_SUMMARIZER_LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in environment variables.")
        # print(f"Using OpenAI model: {OPENAI_MODEL} as summarizer")
        return ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY)
    elif CURRENT_SUMMARIZER_LLM_PROVIDER == "claude":
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not set in environment variables.")
        # print(f"Using Claude model: {CLAUDE_MODEL} as summarizer")
        return ChatAnthropic(model=CLAUDE_MODEL, anthropic_api_key=ANTHROPIC_API_KEY)
    elif CURRENT_SUMMARIZER_LLM_PROVIDER == "gemini":
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not set in environment variables.")
        # print(f"Using Gemini model: {GEMINI_MODEL} as summarizer")
        return ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=GOOGLE_API_KEY)
    else:
        raise ValueError(f"Unsupported LLM provider: {CURRENT_SUMMARIZER_LLM_PROVIDER}")

def call_llm(message_history, functions):
    """
    Calls the configured LLM with the given message history and functions.

    Args:
        message_history (list): A list of LangChain message objects (e.g., HumanMessage, AIMessage).
        functions (list): A list of LangChain tool definitions.

    Returns:
        list: A list of dictionaries, each representing a function to call.
    """
    client = get_llm_client()
    
    print("CALLING LLM")
    try:
        response = client.invoke(
            input=message_history,
            tools=functions,
        )

        functions_to_call = []
        if response.tool_calls:
            for tool_call in response.tool_calls:
                name = tool_call.name
                args = tool_call.args
                functions_to_call.append({"name": name, "args": args})

        return functions_to_call

    except Exception as e:
        print(f"Error calling LLM: {e}")
        return []