import asyncio

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client


from llm import call_llm, convert_to_llm_tool

server_params = StdioServerParameters(
    command="mcp",  # Executable
    args=["run", "server.py"],  # Optional command line arguments
    env=None,
)

async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(
            read, write
        ) as session:
            # Initialize the connection
            await session.initialize()
            
            ## List available resources
            # resources = await session.list_resources()
            # print("LISTING RESOURCES")
            # for resource in resources:
            #     print("Resource: ", resource)

            # List available tools
            tools = await session.list_tools()
            print("LISTING TOOLS=======")
            
            functions = []
            for tool in tools.tools:
                print("Tool: ", tool.name)
                print("Tool", tool.inputSchema["properties"])
                functions.append(convert_to_llm_tool(tool))
                print("======================")
                
            
            # prompt = "Add 38 to 98"
            # prompt = "Summarize the paper : On_the_Equivalence_between_Logic_Programming_and_SETAF"
            prompt = "Summarize http://arxiv.org/pdf/2412.08520v1"
            
            functions_to_call = call_llm(prompt, functions)
            
            for f in functions_to_call:
                result = await session.call_tool(f["name"], arguments=f["args"])
                
                print(f"Calling tool: {f['name']} with args: {f['args']}")
                print("TOOL RESULT: ", result.content)
                
                # print("print TEST:\n",  result.content.text) # not works

if __name__ == "__main__":
    asyncio.run(run())