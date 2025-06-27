import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any
import yaml
# from pathlib import Path

from llm_provider import LLMProvider
from mcp_client import MCPClient


class AgentHost:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.llm_provider = LLMProvider(self.config['llm'])
        self.mcp_clients = {}
        self.logger = self._setup_logging()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('agent_host.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    async def initialize_mcp_servers(self):
        """Initialize connections to MCP servers"""
        for server_name, server_config in self.config['mcp_servers'].items():
            try:
                client = MCPClient(server_config['transport'])
                await client.connect()
                self.mcp_clients[server_name] = client
                self.logger.info(f"Connected to MCP server: {server_name}")
            except Exception as e:
                self.logger.error(f"Failed to connect to {server_name}: {e}")
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on the appropriate MCP server with logging and timing"""
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        
        self.logger.info(f"TOOL_CALL: {tool_name} | Args: {arguments} | Timestamp: {timestamp}")
        print(f"\n🔧 Tool Call: {tool_name}")
        print(f"📝 Arguments: {json.dumps(arguments, indent=2)}")
        
        try:
            # Determine which MCP server to use based on tool name
            server_name = self._get_server_for_tool(tool_name)
            if server_name not in self.mcp_clients:
                raise ValueError(f"MCP server '{server_name}' not available")
            
            client = self.mcp_clients[server_name]
            result = await client.call_tool(tool_name, arguments)
            
            latency = time.time() - start_time
            self.logger.info(f"TOOL_RESULT: {tool_name} | Latency: {latency:.2f}s | Success: True")
            print(f"⏱️  Latency: {latency:.2f}s")
            print(f"✅ Status: Success")
            
            return result
            
        except Exception as e:
            latency = time.time() - start_time
            self.logger.error(f"TOOL_ERROR: {tool_name} | Latency: {latency:.2f}s | Error: {str(e)}")
            print(f"⏱️  Latency: {latency:.2f}s")
            print(f"❌ Status: Error - {str(e)}")
            raise
    
    def _get_server_for_tool(self, tool_name: str) -> str:
        """Map tool names to MCP server names"""
        tool_mapping = {
            'search_papers': 'paper_search',
            'summarize_pdf': 'pdf_summarize'
        }
        return tool_mapping.get(tool_name, 'paper_search')
    
    async def process_query(self, user_query: str) -> str:
        """Process user query and determine appropriate actions"""
        # Use LLM to analyze query and determine tool calls
        system_prompt = """You are an academic research assistant. Analyze the user's query and determine what tools to call.

Available tools:
1. search_papers - Search for academic papers on arXiv
   - Parameters: {"query": "search terms", "max_results": number}
2. summarize_pdf - Summarize a PDF from URL
   - Parameters: {"pdf_url": "https://..."}

Respond with a JSON array of tool calls in this format:
[
    {
        "tool": "tool_name",
        "arguments": {"param": "value"}
    }
]

If no tools are needed, respond with an empty array []."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
        
        tool_plan = await self.llm_provider.generate(messages)
        
        try:
            tool_calls = json.loads(tool_plan)
            if not isinstance(tool_calls, list):
                tool_calls = []
        except json.JSONDecodeError:
            self.logger.warning(f"Failed to parse tool calls: {tool_plan}")
            tool_calls = []
        
        # Execute tool calls
        results = []
        for tool_call in tool_calls:
            try:
                result = await self.call_tool(tool_call['tool'], tool_call['arguments'])
                results.append(result)
            except Exception as e:
                results.append({"error": str(e)})
        
        # Generate final response using LLM
        if results:
            response_prompt = f"""Based on the user query: "{user_query}"
            
Tool results: {json.dumps(results, indent=2)}

Provide a helpful response to the user based on these results."""
            
            response_messages = [
                {"role": "system", "content": "You are a helpful academic research assistant. Provide clear, informative responses based on the tool results."},
                {"role": "user", "content": response_prompt}
            ]
            
            return await self.llm_provider.generate(response_messages)
        else:
            # Direct LLM response when no tools needed
            messages = [
                {"role": "system", "content": "You are a helpful academic research assistant."},
                {"role": "user", "content": user_query}
            ]
            return await self.llm_provider.generate(messages)
    
    async def stream_response(self, response: str):
        """Stream response back to user with typing effect"""
        words = response.split()
        for i, word in enumerate(words):
            print(word, end=' ', flush=True)
            if i % 10 == 0:  # Add slight delay every 10 words
                await asyncio.sleep(0.1)
        print()  # New line at end
    
    async def run_cli(self):
        """Main CLI interface"""
        print("🔬 Academic Research Agent Host")
        print("=" * 50)
        print("Initializing MCP servers...")
        
        await self.initialize_mcp_servers()
        
        print(f"✅ Connected to {len(self.mcp_clients)} MCP servers")
        print("Type 'quit' to exit, 'help' for commands\n")
        
        while True:
            try:
                user_input = input("\n📚 Research Query: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                elif not user_input:
                    continue
                
                print("\n🤔 Processing your query...")
                response = await self.process_query(user_input)
                
                print("\n📄 Response:")
                print("-" * 30)
                await self.stream_response(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                self.logger.error(f"Error processing query: {e}")
                print(f"❌ Error: {e}")
    
    def _show_help(self):
        """Show help information"""
        print("""
📖 Available Commands:
• Search papers: "Find papers about quantum computing"
• Summarize PDF: "Summarize this paper: https://arxiv.org/pdf/..."
• General questions: "What is machine learning?"
• quit/exit/q: Exit the application
        """)
    
    async def cleanup(self):
        """Cleanup MCP connections"""
        for client in self.mcp_clients.values():
            await client.disconnect()


async def main():
    agent = AgentHost(config_path="agent_host/config.yaml")
    try:
        await agent.run_cli()
    finally:
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main())