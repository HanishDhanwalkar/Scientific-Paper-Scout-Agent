import asyncio
import json

import streamlit as st

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from llm import call_llm, convert_to_llm_tool

from contextlib import AsyncExitStack



SERVER_PATH = "./server.py"

server_params = StdioServerParameters(
    command="mcp",
    args=["run", SERVER_PATH],
    env=None,
)


async def run():
    st.set_page_config(page_title="MCP Chatbot", layout="wide")
    st.title("MCP Chatbot")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            functions = []
            for tool in tools.tools:
                # print("Tool: ", tool.name)
                # print("Tool", tool.inputSchema["properties"])
                functions.append(convert_to_llm_tool(tool))

            print("================= MCP Chatbot Ready! ('exit' | 'q' to quit)=================")
            
            with st.expander(f"Functions({len(functions)}:)"):
                total_tools = len(functions)
                for idx, mcp_tool in enumerate(functions):
                    st.markdown(f"**Tool {idx + 1}: `{mcp_tool['function']['name']}`**")
                    st.caption(f"{mcp_tool['function']['description']}")
                    with st.popover("Schema"):
                        st.json(mcp_tool["function"]["parameters"]['properties']) # inputSchema
                    if idx < total_tools - 1:
                        st.divider()
            
            if user_input := st.chat_input("Ask MCP client..."):
                with st.chat_message("user"):
                    st.markdown(user_input)
                st.session_state.messages.append({"role": "user", "content": user_input})

                with st.chat_message("assistant"):
                    status_placeholder = st.status("Processing...", expanded=False)
                    message_placeholder = st.empty()
                    
                tool_call_present, response = call_llm(st.session_state.messages, functions)

                full_response = ""

                if tool_call_present:
                    for tool_call in response:
                        print(f"Calling tool: {tool_call['name']} with args: {tool_call['args']}")
                        status_placeholder.update(label=f"🔧 Calling tool: {tool_call['name']} with args: {tool_call['args']}", state="running")
                        result = await session.call_tool(tool_call["name"], arguments=tool_call["args"])

                        output_text = result.content[0].text
                        print(f"Assistant (tool result): {output_text}")
                        full_response += output_text + "\n"
                        
                        status_placeholder.update(label=f"✅ Done Calling tool : {tool_call['name']} with args: {tool_call['args']}", state="complete")
                        
                else:
                    full_response = response
                    status_placeholder.update(label=f"✅ response generated", state="complete")
                    print(f"Assistant: {full_response}")
                    
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

           
if __name__ == "__main__":
    asyncio.run(run())

        
    