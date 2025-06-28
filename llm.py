from langchain_ollama import ChatOllama
import json

# llm
def call_llm(prompt, functions):
    model_name = "llama3.2"
    client = ChatOllama(model=model_name)

    print("CALLING LLM")
    response = client.invoke(
        input=[
            {
            "role": "system",
            "content": "You are a helpful assistant.",
            },
            {
            "role": "user",
            "content": prompt,
            },
        ],
        tools = functions,
        # Optional parameters
    )
    # print(response)
    print("Status: ", response.response_metadata['done'])
    
    functions_to_call = []
    if response.tool_calls:
        for tool_call in response.tool_calls:
            # print("TOOL: ", tool_call)
            name = tool_call['name']
            args = tool_call['args']
            # args = json.loads(tool_call['args'])
            functions_to_call.append({ "name": name, "args": args })
            

    return functions_to_call


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