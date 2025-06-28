import streamlit as st
import asyncio
import threading
import queue
import time
import os 
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from llm import call_llm, convert_to_llm_tool 

# --- Streamlit App Configuration ---
st.set_page_config(layout="centered", page_title="MCP Chatbot")

# --- Session State Initialization (More robust) ---
st.session_state.messages = st.session_state.get("messages", [])
st.session_state.mcp_worker_thread_started = st.session_state.get("mcp_worker_thread_started", False)
st.session_state.mcp_initialized_success = st.session_state.get("mcp_initialized_success", False)
st.session_state.mcp_manager = st.session_state.get("mcp_manager", None)

# --- Global Queues for Inter-Thread Communication ---
# These queues facilitate thread-safe communication between the Streamlit main thread and the separate MCP worker thread.
request_queue = queue.Queue()  # Streamlit UI -> MCP Worker Thread (for sending commands like tool calls)
response_queue = queue.Queue() # MCP Worker Thread -> Streamlit UI (for sending back results/errors/status)

# --- Global Flag for MCP Initialization Status ---
mcp_initialized_event = threading.Event()

# --- MCP Client Manager Class (Designed to run in a separate thread) ---
class MCPClientManager:
    def __init__(self):
        self._session = None      # Holds the MCP ClientSession instance
        self._functions = []      # Stores the LLM-compatible tool definitions
        self._event_loop = None   # Each thread needs its own dedicated asyncio event loop

    async def _initialize_mcp_client(self, ui_response_queue):
        """
        Asynchronous initialization method for the MCP client.
        Sends status messages to the UI via the provided queue.
        """
        # Send initial status to the UI
        ui_response_queue.put(("init_status", "INFO", "Establishing stdio client connection..."))

        # Define the parameters for starting the MCP server process.
        # It assumes 'server.py' is runnable via 'mcp run server.py'.
        # Set PYTHONUNBUFFERED to 1 to ensure subprocess output is not buffered,
        # which can help with real-time communication over pipes.
        env_vars = os.environ.copy()
        env_vars["PYTHONUNBUFFERED"] = "1"
        server_params = StdioServerParameters(
            command="mcp",
            args=["run", "server.py"],
            env=env_vars,
        )

        try:
            # Use async with for proper context management.
            # This ensures __aexit__ is called correctly when the blocks are exited.
            async with stdio_client(server_params) as (reader, writer):
                # The reader and writer are managed by stdio_client's context manager.
                # We don't store them directly as self._reader, self._writer if not strictly needed
                # for later direct access, as their lifecycle is tied to this `async with` block.
                # If they were needed after this block, manual __aenter__ and __aexit__ calls
                # would be necessary with careful context management.
                async with ClientSession(reader, writer) as session:
                    self._session = session # Store the session object for subsequent tool calls
                    ui_response_queue.put(("init_status", "INFO", "Initializing MCP session..."))
                    await self._session.initialize()

                    ui_response_queue.put(("init_status", "INFO", "Listing available tools..."))
                    tools_response = await self._session.list_tools()
                    # Convert the MCP tool definitions into a format suitable for the LLM.
                    self._functions = [convert_to_llm_tool(tool) for tool in tools_response.tools]

                    ui_response_queue.put(("init_status", "SUCCESS", "MCP Client Initialized Successfully!"))
                    return True # Indicate success of initialization within this scope

        except Exception as e:
            # Send error status to the UI
            ui_response_queue.put(("init_status", "ERROR", f"Error during MCP client setup: {e}"))
            print(f"MCP Worker Thread: Initialization error: {e}") # Debug print
            return False # Indicate failure

    async def _process_requests(self, request_q, response_q):
        """
        Asynchronous loop to continuously process requests received from the main Streamlit thread.
        This loop runs within the dedicated asyncio event loop of the worker thread.
        """
        while True:
            try:
                # Use get_nowait() to poll the synchronous queue without blocking the asyncio loop.
                # If the queue is empty, a queue.Empty exception is raised.
                request_type, data, request_id = request_q.get_nowait()
            except queue.Empty:
                # If the queue is empty, pause briefly to avoid busy-waiting and yield control.
                await asyncio.sleep(0.01) # Sleep for a very short duration
                continue # Continue to the next iteration of the loop to poll again

            print(f"MCP Worker Thread: Received request: {request_type} (ID: {request_id})")

            if request_type == "CALL_TOOL":
                tool_name, tool_args = data
                try:
                    if not self._session:
                        # This should ideally not happen if initialization was successful,
                        # but as a safeguard.
                        raise RuntimeError("MCP session not available for tool call.")
                    # Execute the tool call via the MCP session.
                    result = await self._session.call_tool(tool_name, arguments=tool_args)
                    # Put the successful result back into the response queue for the Streamlit UI.
                    response_q.put((request_id, "SUCCESS", result))
                except Exception as e:
                    # If a tool call fails, send an error response to the UI.
                    print(f"MCP Worker Thread: Error calling tool {tool_name}: {e}")
                    response_q.put((request_id, "ERROR", str(e)))
            elif request_type == "SHUTDOWN":
                # If a shutdown request is received, break out of the loop to terminate the thread.
                print("MCP Worker Thread: Shutting down gracefully.")
                break
            else:
                # Handle unknown request types.
                print(f"MCP Worker Thread: Unknown request type: {request_type}")
                response_q.put((request_id, "ERROR", "Unknown request type"))

    def start_mcp_listener_thread(self, ui_response_queue):
        """
        This method is the entry point for the dedicated MCP worker thread.
        It sets up its own asyncio event loop and orchestrates the MCP client's lifecycle.
        """
        # Create a new asyncio event loop for this specific thread.
        self._event_loop = asyncio.new_event_loop()
        # Set this new loop as the current event loop for this thread.
        asyncio.set_event_loop(self._event_loop)
        
        try:
            # First, run the asynchronous MCP client initialization.
            # This call blocks this thread until initialization is complete.
            init_success = self._event_loop.run_until_complete(
                self._initialize_mcp_client(ui_response_queue)
            )
            
            # After initialization, signal to the main Streamlit thread about the status.
            if init_success:
                # IMPORTANT: DO NOT set st.session_state here directly from worker thread.
                # Only update global `mcp_initialized_event` and let main thread set session_state.
                mcp_initialized_event.set() # Unblock the main thread's wait()
                print("MCP Worker Thread: Client initialized. Starting request processing loop.")
                # If initialization was successful, start the loop for processing incoming requests.
                self._event_loop.run_until_complete(
                    self._process_requests(request_queue, response_queue)
                )
            else:
                # If initialization failed, still set the event to unblock the main thread.
                # IMPORTANT: DO NOT set st.session_state here directly from worker thread.
                mcp_initialized_event.set()
                print("MCP Worker Thread: Client initialization failed. Exiting.")

        except Exception as e:
            # Catch any unexpected critical errors that occur during the thread's main operation.
            print(f"MCP Worker Thread: Critical error in main loop: {e}")
            # Ensure the event is set to prevent the main thread from getting stuck.
            mcp_initialized_event.set()
            # Also send an error message to the UI for visibility via the queue.
            ui_response_queue.put(("init_status", "ERROR", f"Critical error in worker thread: {e}"))
        finally:
            # Ensure the event loop is closed when the thread is done.
            if self._event_loop and not self._event_loop.is_closed():
                self._event_loop.close()
            print("MCP Worker Thread: Event loop closed. Thread exiting.")


# --- Start the MCP Worker Thread (executed once per Streamlit app run) ---
# This ensures the background thread is only created and started one time.
# The initial state is set to False at the very top of the script.
if not st.session_state.mcp_worker_thread_started:
    st.session_state.mcp_worker_thread_started = True # Mark as started immediately
    print("Main Thread: Starting MCP Worker Thread...")
    
    # Create an instance of the MCPClientManager.
    manager = MCPClientManager()
    # Create and start the new thread.
    # We pass the global `response_queue` to the thread so it can send messages back to the UI.
    mcp_thread = threading.Thread(
        target=manager.start_mcp_listener_thread,
        args=(response_queue,), # Pass the queue as an argument
        daemon=True # Daemon threads exit automatically when the main program exits
    )
    mcp_thread.start() # Start the background thread.

    # Display a loading spinner in the Streamlit UI while waiting for MCP initialization.
    # The main thread actively monitors the response_queue for initialization status.
    with st.spinner("Connecting to MCP server and loading tools..."):
        init_status = "PENDING"
        max_wait_time = 180 # Increased timeout for potentially slower server starts or first-time setup
        start_time = time.time()
        while init_status == "PENDING" and (time.time() - start_time) < max_wait_time:
            try:
                # Poll the response queue for initialization status messages.
                # Use a short timeout to prevent blocking the Streamlit UI indefinitely.
                msg_id, msg_type, msg_data = response_queue.get(timeout=0.1) # Shorter poll interval
                if msg_id == "init_status":
                    if msg_type == "INFO":
                        st.info(msg_data) # Display informative messages
                    elif msg_type == "SUCCESS":
                        st.success(msg_data) # Display success message
                        init_status = "SUCCESS"
                        st.session_state.mcp_initialized_success = True # Update session state in main thread
                        st.session_state.mcp_manager = manager # Store the manager instance for later use
                    elif msg_type == "ERROR":
                        st.error(msg_data) # Display error message
                        init_status = "ERROR"
                        st.session_state.mcp_initialized_success = False # Update session state in main thread
                        break # Exit the loop on error
            except queue.Empty:
                pass # Keep waiting if the queue is empty
        
        # After the loop (either success, error, or timeout)
        if init_status == "PENDING": # If loop exited due to timeout
            st.error(f"MCP client initialization timed out after {max_wait_time} seconds. Please check server.py and your MCP installation.")
            st.session_state.mcp_initialized_success = False
        
    # If initialization was not successful, stop the Streamlit app.
    # This check now relies on the `st.session_state.mcp_initialized_success` set by the main thread.
    if not st.session_state.mcp_initialized_success:
        st.stop()


# --- Main Streamlit App Content ---
st.title("Interactive MCP Chatbot")

# Display all messages from the chat history.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input at the bottom of the chat interface.
user_prompt = st.chat_input("Type your message here...")

if user_prompt:
    # Append the user's message to the chat history.
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    # Display the user's message immediately in the UI.
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Prepare a placeholder for the assistant's response to allow dynamic updates.
    with st.chat_message("assistant"):
        message_placeholder = st.empty() # Creates an empty container for dynamic content
        full_response_content = "" # This string will accumulate the full response content

        # Check if the MCP client was successfully initialized.
        if not st.session_state.mcp_initialized_success:
            full_response_content = "Error: MCP client is not initialized. Please refresh the page."
            message_placeholder.markdown(full_response_content)
        else:
            # Retrieve the LLM-compatible functions from the initialized MCP manager instance.
            mcp_functions = st.session_state.mcp_manager._functions
            
            # Prepare the message history in the format expected by the `call_llm` function.
            llm_message_history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]

            # Call the LLM to get a response (which might be direct text or a tool call).
            with st.spinner("Thinking..."):
                tool_call_present, response = call_llm(llm_message_history, mcp_functions)

            if tool_call_present:
                # If the LLM decided to call one or more tools:
                for tool_call in response:
                    tool_name = tool_call['name']
                    tool_args = tool_call['args']

                    # Display tool call details in an expandable section for clarity.
                    with st.expander(f"Tool Call: `{tool_name}`", expanded=True):
                        st.json(tool_args) # Display tool arguments in a JSON format

                    # Update the message placeholder to show that a tool is being called.
                    full_response_content += f"**Calling tool:** `{tool_name}` with arguments: `{tool_args}`\n"
                    message_placeholder.markdown(full_response_content)

                    try:
                        # Generate a unique request ID for this specific tool call.
                        request_id = str(time.time())

                        # Put the tool call request into the `request_queue` for the worker thread to process.
                        request_queue.put(("CALL_TOOL", (tool_name, tool_args), request_id))

                        with st.spinner(f"Executing tool `{tool_name}`..."):
                            status = "PENDING"
                            result_data = None
                            max_tool_wait_time = 180 # Maximum time to wait for tool execution in seconds
                            start_time = time.time()
                            while status == "PENDING" and (time.time() - start_time) < max_tool_wait_time:
                                try:
                                    # Poll the response queue for the result of *this specific* request.
                                    response_id, status, result_data = response_queue.get(timeout=0.1)
                                    if response_id != request_id:
                                        # If a response for a different request is found, put it back
                                        # and continue waiting for the correct one.
                                        response_queue.put((response_id, status, result_data))
                                        status = "PENDING" # Still waiting for our specific ID
                                except queue.Empty:
                                    pass 

                            # tool execution
                            if status == "PENDING":
                                error_message = f"Tool `{tool_name}` execution timed out after {max_tool_wait_time} seconds."
                                message_placeholder.markdown(f"**Error:** {error_message}")
                                full_response_content += f"**Error:** {error_message}\n"
                            elif status == "SUCCESS":
                                tool_result = result_data
                                output_text = ""
                                if hasattr(tool_result, 'content') and tool_result.content:
                                    if isinstance(tool_result.content, list) and len(tool_result.content) > 0 and hasattr(tool_result.content[0], 'text'):
                                        output_text = tool_result.content[0].text
                                    elif hasattr(tool_result.content, 'text'):
                                        output_text = tool_result.content.text
                                    else:
                                        output_text = str(tool_result.content) # Fallback for other content types
                                else:
                                    output_text = str(tool_result) # Fallback if no 'content' attribute

                                tool_output_display = f"**Tool Output:**\n```\n{output_text}\n```"
                                message_placeholder.markdown(full_response_content + tool_output_display)
                                full_response_content += tool_output_display + "\n"
                            else: # status == "ERROR"
                                error_message = f"Error executing tool `{tool_name}`: {result_data}"
                                message_placeholder.markdown(f"**Error:** {error_message}")
                                full_response_content += f"**Error:** {error_message}\n"

                    except Exception as e:
                        error_message = f"Unexpected error during tool execution: {e}"
                        message_placeholder.markdown(f"**Error:** {error_message}")
                        full_response_content += f"**Error:** {error_message}\n"
            else:
                full_response_content = response
                message_placeholder.markdown(full_response_content)

        st.session_state.messages.append({"role": "assistant", "content": full_response_content.strip()})
    
    st.experimental_rerun()
