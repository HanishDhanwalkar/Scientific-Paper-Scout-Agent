from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from config import (
    CURRENT_LLM_PROVIDER,
    OLLAMA_MODEL,
    OPENAI_MODEL,
    CLAUDE_MODEL,
    GEMINI_MODEL,
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    GOOGLE_API_KEY,
    OLLAMA_BASE_URL
)

def convert_to_llm_tool(tool):
    tool_schema = {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "type": "function",
            "parameters": {
                "type": "object",
                "properties": tool.inputSchema["properties"]
            }
        }
    }
    return tool_schema

def get_llm_client():
    """Initializes and returns the appropriate LLM client based on configuration."""
    if CURRENT_LLM_PROVIDER == "ollama":
        print(f"Using Ollama model: {OLLAMA_MODEL}")
        return ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    elif CURRENT_LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in environment variables.")
        print(f"Using OpenAI model: {OPENAI_MODEL}")
        return ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY)
    elif CURRENT_LLM_PROVIDER == "claude":
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not set in environment variables.")
        print(f"Using Claude model: {CLAUDE_MODEL}")
        return ChatAnthropic(model=CLAUDE_MODEL, anthropic_api_key=ANTHROPIC_API_KEY)
    elif CURRENT_LLM_PROVIDER == "gemini":
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not set in environment variables.")
        print(f"Using Gemini model: {GEMINI_MODEL}")
        return ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=GOOGLE_API_KEY)
    else:
        raise ValueError(f"Unsupported LLM provider: {CURRENT_LLM_PROVIDER}")

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
    
    tool_call_present = False
    try:
        response = client.invoke(
            input=message_history,
            tools=functions,
        )
        
        functions_to_call = []
        if response.tool_calls:
            for tool_call in response.tool_calls:
                name = tool_call['name']
                args = tool_call['args']
                functions_to_call.append({"name": name, "args": args})
            
            tool_call_present = True
                
        else:
            print("No tool calls found in response.")
            return tool_call_present, response.content

        print("LLM calling complete....")
        return tool_call_present, functions_to_call

    except Exception as e:
        return tool_call_present, "Error calling LLM: " + str(e)