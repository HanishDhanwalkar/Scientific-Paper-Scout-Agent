from langchain_ollama import ChatOllama
import json

from tools import add

tools = [add]

llm = ChatOllama(model="llama3.2")
llm_with_tools = llm.bind_tools(tools)

# prompt = input("you: ")
prompt = "Add 39 to 94"

# response = llm.invoke("Hello, how are you?")
response = llm_with_tools.invoke(
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
        # tools = functions,
        # Optional parameters
    )

# print(response)
# print(response.content)

if response.tool_calls:
    for tool_call in response.tool_calls:
        print("TOOL: ", tool_call)
        name = tool_call['name']
        args = tool_call['args']
        print(name, args)

# print(response.tool_calls)