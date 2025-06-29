import asyncio
import json
import os
import re
import sys
from contextlib import AsyncExitStack
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from llm import call_llm, convert_to_llm_tool

import streamlit as st

# --- Asyncio Event Loop Policy for Windows ---
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# --- Add Project Root to Python Path ---
PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)
print(f"Project dir: {PROJECT_ROOT}")

sys.path.insert(0, PROJECT_ROOT)

# # --- Streamlit Logo Configuration ---
# st.logo(
#     os.path.join(PROJECT_ROOT, "assets", "mcp_chatbot_logo.png"),
#     size="large",
# )

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="MCP Chatbot", layout="wide")
st.title("⚙️ MCP Chatbot - Interactive Agent")
st.caption(
    "A chatbot that uses the Model Context Protocol (MCP) to interact with tools."
)

# --- Session State Initialization ---
# Streamlit's session_state is used to preserve data across reruns of the script.
if "messages" not in st.session_state:
    st.session_state.messages = []  # Stores the UI chat messages (user/assistant)

if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = "openai"  # Default LLM provider

# TODO: Use config.py
if "llm_config" not in st.session_state:
    st.session_state.llm_config = {
        "openai_model_name": "gpt-3.5-turbo",
        "openai_api_key": "",
        "openai_base_url": "",
        "ollama_model_name": "llama3.2",
        "ollama_base_url": "http://localhost:11434",
    }

if "mcp_tools_cache" not in st.session_state:
    st.session_state.mcp_tools_cache = {}  # Cache for discovered MCP tools

# State for managing the active MCP client session and its associated configuration hash.
if "mcp_client_session" not in st.session_state:
    st.session_state.mcp_client_session = None
if "session_config_hash" not in st.session_state:
    st.session_state.session_config_hash = None

# Variables to manage the lifecycle of MCP clients within Streamlit's context.
if "active_mcp_clients_raw" not in st.session_state:
    # Stores raw (read, write) pairs or similar handles for direct stdio_client interaction
    st.session_state.active_mcp_clients_raw = []
if "mcp_client_stack" not in st.session_state:
    # AsyncExitStack manages the context of multiple async clients.
    st.session_state.mcp_client_stack = None

if "llm_message_history" not in st.session_state:
    # This stores the message history *specifically for the LLM call*,
    # which includes system prompts, user queries, tool calls, and tool outputs.
    st.session_state.llm_message_history = []


# --- Constants for Workflow Visualization ---
WORKFLOW_ICONS = {
    "USER_QUERY": "👤",
    "LLM_THINKING": "☁️",
    "LLM_RESPONSE": "💬",
    "TOOL_CALL": "🔧",
    "TOOL_EXECUTION": "⚡️",
    "TOOL_RESULT": "📊",
    "FINAL_STATUS": "✅",
    "ERROR": "❌",
    "JSON_TOOL_CALL": "📜",
}

# --- DataClass for Workflow Step ---
@dataclass
class WorkflowStep:
    """
    Represents a single step in the chatbot's interaction workflow.
    This helps in visualizing the internal thought process and tool usage.
    """
    type: str
    content: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Converts the WorkflowStep instance to a dictionary."""
        return asdict(self)


async def get_mcp_tools(force_refresh=False) -> Dict[str, List[Dict[str, Any]]]:
    """
    Retrieves MCP tools from cache or by initializing MCP clients to list them.
    This function now directly uses mcp.ClientSession and stdio_client.

    Args:
        force_refresh (bool): If True, bypass the cache and re-initialize clients.

    Returns:
        Dict[str, List[Dict[str, Any]]]: A dictionary mapping client names to lists of their tool dictionaries.
    """
    if not force_refresh and st.session_state.mcp_tools_cache:
        return st.session_state.mcp_tools_cache

    server_path = os.path.join(PROJECT_ROOT, "server.py")

    # Use a temporary AsyncExitStack for tool discovery
    async with AsyncExitStack() as stack:
        try:
            server_params = StdioServerParameters(
                command="mcp",
                args=["run", server_path],
                env=None,
            )
            
            read, write = await stack.enter_async_context(stdio_client(server_params))
                    
            # Create an MCP ClientSession using the read/write streams
            client_session = await stack.enter_async_context(ClientSession(read, write))
            await client_session.initialize() # Initialize the session

            tools = await client_session.list_tools()
            functions = []
            for tool in tools.tools:
                # tool_name = tool.name
                # tool_input_schema = tool.inputSchema["properties"]
                # # functions.append({
                # #     "name": tool_name,
                # #     "input_schema": tool_input_schema
                # # })
                functions.append(convert_to_llm_tool(tool))
        except Exception as e:
            st.sidebar.error(f"Error fetching tools: {e}")

    st.session_state.mcp_tools_cache = functions
    return functions


def render_sidebar(mcp_tools: Optional[Dict[str, List[Dict[str, Any]]]] = None):
    """
    Renders the Streamlit sidebar with settings for LLM and MCP,
    and control buttons like "Clear Chat" and "Refresh Tools".
    """
    with st.sidebar:
        st.header("Settings")

        # --- Clear Chat Button ---
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.mcp_client_session = None
            st.session_state.session_config_hash = None
            st.session_state.active_mcp_clients_raw = []
            # Ensure the AsyncExitStack is properly closed if it exists from a previous run
            if st.session_state.mcp_client_stack:
                try:
                    # Manually exit the stack to ensure clients are closed.
                    # This is a synchronous call in a synchronous context, but it triggers
                    # the __aexit__ which runs in the current event loop.
                    asyncio.run(st.session_state.mcp_client_stack.__aexit__(None, None, None))
                except Exception as e:
                    print(f"Error during clear chat cleanup: {e}", file=sys.stderr)
            st.session_state.mcp_client_stack = None
            st.session_state.llm_message_history = [] # Clear LLM's internal history
            st.toast("Chat cleared!", icon="🧹")
            st.rerun()

        llm_tab, mcp_tab = st.tabs(["LLM", "MCP"])
        with llm_tab:
            st.session_state.llm_provider = st.radio(
                "LLM Provider:",
                ["openai", "ollama"],
                index=["openai", "ollama"].index(st.session_state.llm_provider),
                key="llm_provider_radio",
            )

            # TODO: Use config.py
            llm_config = st.session_state.llm_config
            if st.session_state.llm_provider == "openai":
                llm_config["openai_model_name"] = st.text_input(
                    "OpenAI Model Name:",
                    value=llm_config.get("openai_model_name", "gpt-3.5-turbo"),
                    placeholder="e.g. gpt-4o",
                    key="openai_model_name",
                )
                llm_config["openai_api_key"] = st.text_input(
                    "OpenAI API Key:",
                    value=llm_config.get("openai_api_key", ""),
                    type="password",
                    key="openai_api_key",
                )
                llm_config["openai_base_url"] = st.text_input(
                    "OpenAI Base URL (optional):",
                    value=llm_config.get("openai_base_url", ""),
                    key="openai_base_url",
                )
            else:  # ollama
                llm_config["ollama_model_name"] = st.text_input(
                    "Ollama Model Name:",
                    value=llm_config.get("ollama_model_name", "llama3"),
                    placeholder="e.g. llama3",
                    key="ollama_model_name",
                )
                llm_config["ollama_base_url"] = st.text_input(
                    "Ollama Base URL:",
                    value=llm_config.get("ollama_base_url", "http://localhost:11434"),
                    key="ollama_base_url",
                )

        with mcp_tab:
            # --- Refresh Tools Button ---
            if st.button("🔄 Refresh Tools", use_container_width=True, type="primary"):
                st.session_state.mcp_tools_cache = {}
                st.session_state.mcp_client_session = None # Reset client session on tool refresh
                st.session_state.session_config_hash = None
                st.session_state.active_mcp_clients_raw = []
                if st.session_state.mcp_client_stack:
                    try:
                        asyncio.run(st.session_state.mcp_client_stack.__aexit__(None, None, None))
                    except Exception as e:
                        print(f"Error during refresh tools cleanup: {e}", file=sys.stderr)
                st.session_state.mcp_client_stack = None
                st.session_state.llm_message_history = []
                st.toast("Tools refreshed and session reset.", icon="🔄")
                st.rerun()

            if not mcp_tools:
                st.info("No MCP tools loaded or configured.")

            with st.expander(f"({len(mcp_tools)} tools)"):
                total_tools = len(mcp_tools)
                for idx, mcp_tool in enumerate(mcp_tools):
                        st.markdown(f"**Tool {idx + 1}: `{mcp_tool['function']['name']}`**")
                        st.caption(f"{mcp_tool['function']['description']}")
                        with st.popover("Schema"):
                            st.json(mcp_tool["function"]["parameters"]['properties']) # inputSchema
                        if idx < total_tools - 1:
                            st.divider()

        # --- About Tabs ---
        en_about_tab, = st.tabs(["About"])
        with en_about_tab:
            st.info(
                "This chatbot uses the Model Context Protocol (MCP) for tool use. "
                "Configure LLM and MCP settings, then ask questions! "
                "Use the 'Clear Chat' button to reset the conversation."
            )

def extract_json_tool_calls(text: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    Extracts potential JSON tool call objects from a given text string.
    This function is generic and does not depend on mcp_chatbot.
    """
    tool_calls = []
    cleaned_text = text
    json_parsed = False

    try:
        data = json.loads(text.strip())
        if isinstance(data, list):
            valid_tools = True
            for item in data:
                if not (
                    isinstance(item, dict) and "tool" in item and "arguments" in item
                ):
                    valid_tools = False
                    break
            if valid_tools:
                tool_calls.extend(data)
                json_parsed = True
        elif isinstance(data, dict) and "tool" in data and "arguments" in data:
            tool_calls.append(data)
            json_parsed = True

        if json_parsed:
            return tool_calls, ""

    except json.JSONDecodeError:
        pass

    json_pattern = r"\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}"
    matches = list(re.finditer(json_pattern, text))
    extracted_indices = set()

    for match in matches:
        start, end = match.span()
        if any(
            start < prev_end and end > prev_start
            for prev_start, prev_end in extracted_indices
        ):
            continue

        json_str = match.group(0)
        try:
            obj = json.loads(json_str)
            if isinstance(obj, dict) and "tool" in obj and "arguments" in obj:
                tool_calls.append(obj)
                extracted_indices.add((start, end))
        except json.JSONDecodeError:
            pass

    if extracted_indices:
        cleaned_parts = []
        last_end = 0
        for start, end in sorted(list(extracted_indices)):
            cleaned_parts.append(text[last_end:start])
            last_end = end
        cleaned_parts.append(text[last_end:])
        cleaned_text = "".join(cleaned_parts).strip()
    else:
        cleaned_text = text

    return tool_calls, cleaned_text

def render_workflow(steps: List[WorkflowStep], container=None):
    """
    Renders the workflow steps in the Streamlit UI, grouping tool call
    sequences into expandable sections for clarity.
    """
    if not steps:
        return

    target = container if container else st

    rendered_indices = set()

    for i, step in enumerate(steps):
        if i in rendered_indices:
            continue

        step_type = step.type

        if step_type == "TOOL_CALL":
            tool_name = step.details.get("tool_name", "Unknown Tool")
            expander_title = f"{WORKFLOW_ICONS['TOOL_CALL']} Tool Call: {tool_name}"
            with target.expander(expander_title, expanded=False):
                st.write("**Arguments:**")
                arguments = step.details.get("arguments", {})
                if isinstance(arguments, str) and arguments == "Pending...":
                    st.info("Preparing arguments...")
                elif isinstance(arguments, dict):
                    if arguments:
                        for key, value in arguments.items():
                            st.write(f"- `{key}`: `{repr(value)}`")
                    else:
                        st.write("_No arguments_")
                else:
                    st.code(str(arguments), language="json")
                rendered_indices.add(i)

                j = i + 1
                while j < len(steps):
                    next_step = steps[j]
                    if next_step.type == "TOOL_EXECUTION":
                        st.write(
                            f"**Status** {WORKFLOW_ICONS['TOOL_EXECUTION']}: "
                            f"{next_step.content}"
                        )
                        rendered_indices.add(j)
                    elif next_step.type == "TOOL_RESULT":
                        st.write(f"**Result** {WORKFLOW_ICONS['TOOL_RESULT']}:")
                        details = next_step.details
                        try:
                            details_dict = json.loads(details)
                            st.json(details_dict)
                        except json.JSONDecodeError:
                            result_str = str(details)
                            st.text(
                                result_str[:500]
                                + ("..." if len(result_str) > 500 else "")
                                or "_Empty result_"
                            )
                        rendered_indices.add(j)
                        break
                    elif next_step.type in ["TOOL_CALL", "JSON_TOOL_CALL"]:
                        break
                    j += 1

        elif step_type == "JSON_TOOL_CALL":
            tool_name = step.details.get("tool_name", "Unknown")
            expander_title = (
                f"{WORKFLOW_ICONS['JSON_TOOL_CALL']} LLM Generated Tool Call: {tool_name}"
            )
            with target.expander(expander_title, expanded=False):
                st.write("**Arguments:**")
                arguments = step.details.get("arguments", {})
                if isinstance(arguments, dict):
                    if arguments:
                        for key, value in arguments.items():
                            st.write(f"- `{key}`: `{value}`")
                    else:
                        st.write("_No arguments_")
                else:
                    st.code(str(arguments), language="json")
            rendered_indices.add(i)

        elif step_type == "ERROR":
            target.error(f"{WORKFLOW_ICONS['ERROR']} {step.content}")
            rendered_indices.add(i)

# TODO: use config.py
def get_config_hash(llm_config: Dict[str, Any], provider: str) -> int:
    """
    Generates a hash based on relevant LLM configuration settings.
    This now uses the directly stored `llm_config` dictionary.
    """
    relevant_config = {
        "provider": provider,
    }
    if provider == "openai":
        relevant_config.update(
            {
                "model": llm_config.get("openai_model_name"),
                "api_key": llm_config.get("openai_api_key"),
                "base_url": llm_config.get("openai_base_url"),
            }
        )
    else:  # ollama
        relevant_config.update(
            {
                "model": llm_config.get("ollama_model_name"),
                "base_url": llm_config.get("ollama_base_url"),
            }
        )
    return hash(json.dumps(relevant_config, sort_keys=True))

async def initialize_mcp_client_session(stack: AsyncExitStack) -> Optional[ClientSession]:
    """
    Initializes a single MCP ClientSession using stdio_client based on servers_config.json.
    This function now directly uses mcp.ClientSession and stdio_client.
    It returns the *first successfully initialized* client session found.
    """
    server_path = os.path.join(PROJECT_ROOT, "server.py")

    # Use a temporary AsyncExitStack for tool discovery
    async with AsyncExitStack() as stack:
        try:
            server_params = StdioServerParameters(
                command="mcp",
                args=["run", server_path],
                env=None,
            )
            
            read, write = await stack.enter_async_context(stdio_client(server_params))
                    
            # Create an MCP ClientSession using the read/write streams
            client_session = await stack.enter_async_context(ClientSession(read, write))
            await client_session.initialize() # Initialize the session

            print("MCP client session initialized")
            st.toast(f"Connected to MCP server: {server_path}", icon="🔌")
            
            return client_session # Return the first successful session
        except Exception as client_ex:
            st.error(f"Failed to initialize MCP client  {client_ex}")
    return None # No client session could be initialized

async def process_chat(user_input: str):
    """
    Handles a user's input, orchestrates the interaction with the LLM and MCP tools,
    and updates the Streamlit UI with the progress and final response.
    This function now directly manages the LLM call and MCP tool calls.
    """

    # 1. Add user message to state and display it
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Prepare Streamlit UI elements for the assistant's response.
    current_workflow_steps = []
    with st.chat_message("assistant"):
        status_placeholder = st.status("Processing...", expanded=False)
        workflow_display_container = st.empty()
        message_placeholder = st.empty()

    llm_config = st.session_state.llm_config
    provider = st.session_state.llm_provider
    current_config_hash = get_config_hash(llm_config, provider)

    try:
        # --- MCP Client Session Management ---
        # If config changed or session doesn't exist, re-initialize
        if (
            st.session_state.mcp_client_session is None
            or current_config_hash != st.session_state.session_config_hash
        ):
            # Clear previous stack if config changed to ensure old clients are closed
            if st.session_state.mcp_client_stack:
                await st.session_state.mcp_client_stack.__aexit__(None, None, None)
                st.session_state.mcp_client_stack = None
                st.session_state.llm_message_history = [] # Clear history on session reset

            st.session_state.mcp_client_stack = AsyncExitStack()
            # initialize_mcp_client_session now returns the mcp.ClientSession directly
            mcp_client_session = await initialize_mcp_client_session(st.session_state.mcp_client_stack)
            st.session_state.mcp_client_session = mcp_client_session

            print(mcp_client_session)

            st.session_state.session_config_hash = current_config_hash
            # Re-initialize LLM's message history if session was reset
            if not st.session_state.llm_message_history:
                st.session_state.llm_message_history = [] # Start fresh


        mcp_client_session = st.session_state.mcp_client_session
        if not mcp_client_session:
            raise RuntimeError("MCP Client Session is not available.")

        # --- Get Tools for LLM ---
        # List tools via the active MCP client session
        tools_list_response = await mcp_client_session.list_tools()
        llm_functions = [convert_to_llm_tool(t) for t in tools_list_response.tools]

        print("Tools:", llm_functions)
        
        # Add user query to workflow steps
        current_workflow_steps.append(
            WorkflowStep(type="USER_QUERY", content=user_input)
        )
        
        print("Workflow steps:", current_workflow_steps)
        
        with workflow_display_container.container():
            render_workflow(current_workflow_steps, container=st)

        status_placeholder.update(
            label="🧠 Processing request...", state="running", expanded=False
        )

        accumulated_response_content = ""
        tool_call_count = 0

        # Add user message to LLM's internal history
        st.session_state.llm_message_history.append({"role": "user", "content": user_input})

        # --- Call LLM and process response ---
        # `call_llm` now includes LLM config details if your llm.py is updated to accept it
        # For this example, we'll pass it as a separate argument if call_llm expects it
        # Otherwise, assume llm.py handles config internally (e.g., via environment vars).
        
        print(st.session_state.llm_message_history)
        
        tool_call_present, llm_response_or_calls = await call_llm(
            st.session_state.llm_message_history, 
            llm_functions,
        )

        if tool_call_present:
            for tool_call in llm_response_or_calls:
                tool_name = tool_call['name']
                arguments = tool_call['args']

                if tool_name and arguments is not None:
                    tool_call_count += 1
                    current_workflow_steps.append(
                        WorkflowStep(
                            type="TOOL_CALL",
                            content=f"Initiating call to: {tool_name}",
                            details={"tool_name": tool_name, "arguments": arguments},
                        )
                    )
                    status_placeholder.update(label=f"🔧 Calling tool: {tool_name}", state="running")
                    with workflow_display_container.container():
                        render_workflow(current_workflow_steps, container=st)

                    try:
                        current_workflow_steps.append(
                            WorkflowStep(type="TOOL_EXECUTION", content=f"Executing {tool_name}...")
                        )
                        status_placeholder.update(label=f"⚡ Executing {tool_name}...", state="running")
                        with workflow_display_container.container():
                            render_workflow(current_workflow_steps, container=st)

                        tool_result_obj = await mcp_client_session.call_tool(tool_name, arguments=arguments)
                        # MCP tool results can be complex. Assuming result.content is a list of parts,
                        # and we want text from the first part if available.
                        tool_output_content = ""
                        if hasattr(tool_result_obj, 'content') and isinstance(tool_result_obj.content, list):
                            if tool_result_obj.content and hasattr(tool_result_obj.content[0], 'text'):
                                tool_output_content = tool_result_obj.content[0].text
                            elif tool_result_obj.content: # Fallback for non-text content
                                try:
                                    tool_output_content = json.dumps([asdict(part) for part in tool_result_obj.content])
                                except TypeError: # If not dataclasses
                                    tool_output_content = str(tool_result_obj.content)
                        else:
                            tool_output_content = str(tool_result_obj) # Fallback if no .content or not a list

                        current_workflow_steps.append(
                            WorkflowStep(
                                type="TOOL_RESULT",
                                content="Received result.",
                                details=tool_output_content, # Store raw output for render_workflow to format
                            )
                        )
                        # Add tool output to LLM's message history for context in next turn
                        st.session_state.llm_message_history.append(
                            {"role": "tool_output", "content": tool_output_content}
                        )
                        
                        status_placeholder.update(
                            label=f"🧠 Processing results from {tool_name}...",
                            state="running",
                        )
                        with workflow_display_container.container():
                            render_workflow(current_workflow_steps, container=st)

                    except Exception as tool_ex:
                        error_msg = f"Error during tool '{tool_name}' execution: {tool_ex}"
                        current_workflow_steps.append(
                            WorkflowStep(type="ERROR", content=error_msg)
                        )
                        st.session_state.llm_message_history.append(
                            {"role": "tool_output", "content": f"ERROR: {error_msg}"}
                        )
                        status_placeholder.update(
                            label=f"❌ Tool Error: {error_msg[:100]}...",
                            state="error", expanded=True
                        )
                        message_placeholder.error(f"Tool error: {error_msg}")
                        with workflow_display_container.container():
                            render_workflow(current_workflow_steps, container=st)
                        # Optionally break or continue based on error handling strategy
                        # For now, we'll let it try other tools or proceed.

            # After all tool calls, potentially call LLM again with updated history
            # This is simplified; a real agent might loop LLM->Tool->LLM
            # For this example, we re-call LLM to get a final natural language response
            # based on new tool outputs.
            status_placeholder.update(label="🧠 Finalizing response...", state="running")
            _, final_llm_response = await call_llm(
                st.session_state.llm_message_history, 
                llm_functions, # Pass tools again just in case LLM needs them for final thought
                llm_config # Pass the llm_config dictionary
            )
            accumulated_response_content = final_llm_response
            
        else: # LLM returned a direct text response
            accumulated_response_content = llm_response_or_calls
        
        # Display the final LLM response
        final_display_content = accumulated_response_content.strip()
        message_placeholder.markdown(final_display_content or "_No text response_")

        if final_display_content:
            current_workflow_steps.append(
                WorkflowStep(
                    type="LLM_RESPONSE",
                    content="Final response generated.",
                    details={"response_text": final_display_content},
                )
            )
        # Append assistant's final response to LLM's history
        st.session_state.llm_message_history.append(
            {"role": "assistant", "content": final_display_content}
        )

        final_status_message = "Completed."
        if tool_call_count > 0:
            final_status_message += f" Processed {tool_call_count} tool call(s)."
        current_workflow_steps.append(
            WorkflowStep(type="FINAL_STATUS", content=final_status_message)
        )

        status_placeholder.update(
            label=f"✅ {final_status_message}", state="complete", expanded=False
        )

        with workflow_display_container.container():
            render_workflow(current_workflow_steps, container=st)

        # --- Store results in session state for UI display ---
        last_user_message_index = -1
        for i in range(len(st.session_state.messages) - 1, -1, -1):
            if st.session_state.messages[i]["role"] == "user":
                last_user_message_index = i
                break

        assistant_message = {
            "role": "assistant",
            "content": final_display_content,
            "workflow_steps": [step.to_dict() for step in current_workflow_steps],
        }
        if last_user_message_index != -1:
            st.session_state.messages.insert(
                last_user_message_index + 1, assistant_message
            )
        else:
            st.session_state.messages.append(assistant_message)

    except Exception as e:
        error_message = f"An unexpected error occurred in process_chat: {str(e)}"
        st.error(error_message)
        current_workflow_steps.append(WorkflowStep(type="ERROR", content=error_message))
        try:
            with workflow_display_container.container():
                render_workflow(current_workflow_steps, container=st)
        except Exception as render_e:
            st.error(f"Additionally, failed to render workflow after error: {render_e}")

        status_placeholder.update(
            label=f"❌ Error: {error_message[:100]}...", state="error", expanded=True
        )
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"Error: {error_message}",
                "workflow_steps": [step.to_dict() for step in current_workflow_steps],
            }
        )
        # Ensure error is also reflected in LLM history for debugging in subsequent turns
        st.session_state.llm_message_history.append(
            {"role": "assistant", "content": f"Error: {error_message}"}
        )

    finally:
        # --- Critical Clean-up for Async Objects in Streamlit ---
        # It's crucial to exit the AsyncExitStack within the same async loop
        # it was entered to properly close all managed resources (like stdio clients).
        try:
            if st.session_state.mcp_client_stack is not None:
                await st.session_state.mcp_client_stack.__aexit__(None, None, None)
        except Exception as cleanup_exc:
            print("MCP clean‑up error during finalizer:", cleanup_exc, file=sys.stderr)
        finally:
            st.session_state.mcp_client_stack = None
            st.session_state.mcp_client_session = None # Ensure new session is created next turn
            st.session_state.active_mcp_clients_raw = []

def display_chat_history():
    """Displays the chat history from st.session_state.messages."""
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "workflow_steps" in message:
                workflow_history_container = st.container()
                workflow_steps = []
                if isinstance(message["workflow_steps"], list):
                    for step_dict in message["workflow_steps"]:
                        if isinstance(step_dict, dict):
                            workflow_steps.append(
                                WorkflowStep(
                                    type=step_dict.get("type", "UNKNOWN"),
                                    content=step_dict.get("content", ""),
                                    details=step_dict.get("details", {}),
                                )
                            )
                if workflow_steps:
                    render_workflow(
                        workflow_steps, container=workflow_history_container
                    )

            st.markdown(message["content"], unsafe_allow_html=True)


async def main():
    """Main application entry point."""
    mcp_tools = await get_mcp_tools()
    render_sidebar(mcp_tools)
    display_chat_history()

    if prompt := st.chat_input(
        "Ask something... (e.g., 'What files are in the root directory?')"
    ):
        await process_chat(prompt)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        # Note: Reliable async cleanup on Streamlit shutdown is still complex.
        pass

