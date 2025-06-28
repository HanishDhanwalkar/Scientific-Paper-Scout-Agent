import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from llm import call_llm, convert_to_llm_tool

server_params = StdioServerParameters(
    command="mcp",
    args=["run", "server.py"],
    env=None,
)

async def run():
    message_history = []

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            functions = []
            for tool in tools.tools:
                # print("Tool: ", tool.name)
                # print("Tool", tool.inputSchema["properties"])
                functions.append(convert_to_llm_tool(tool))
                # print("======================")

            print("================= MCP Chatbot Ready! ('exit' | 'q' to quit)=================")

            while True:
                user_input = input("You: ")
                if user_input.lower() in {"exit", "quit", "q"}:
                    print("Exiting.......")
                    break

                message_history.append({"role": "user", "content": user_input})

                tool_call_present, response = call_llm(message_history, functions)

                full_response = ""

                if tool_call_present:
                    for tool_call in response:
                        print(f"Calling tool: {tool_call['name']} with args: {tool_call['args']}")
                        result = await session.call_tool(tool_call["name"], arguments=tool_call["args"])

                        # output_text = result.content.text if hasattr(result.content, "text") else str(result.content)
                        output_text = result.content[0].text
                        print(f"Assistant: {output_text}")
                        full_response += output_text + "\n"
                        
                else:
                    full_response = response
                    print(f"Assistant: {full_response}")
                
                message_history.append({"role": "assistant", "content": full_response.strip()}) # Append assistant response to history

if __name__ == "__main__":
    asyncio.run(run())