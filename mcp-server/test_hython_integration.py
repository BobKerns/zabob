#!/usr/bin/env python3
"""
Test script to verify hython MCP server integration
"""
import asyncio
import json
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

async def test_hython_tools():
    """Test the hython proxy tools"""

    # Create client session
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "zabob.mcp.server"],
        env=None
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the session
            await session.initialize()

            # List all available tools
            tools = await session.list_tools()
            print("Available tools:")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")

            # Test hython_get_scene_info
            print("\n" + "="*50)
            print("Testing hython_get_scene_info...")
            try:
                result = await session.call_tool("hython_get_scene_info", {})
                print("Result:", json.dumps(result.content, indent=2))
            except Exception as e:
                print(f"Error: {e}")

            # Test hython_list_nodes
            print("\n" + "="*50)
            print("Testing hython_list_nodes...")
            try:
                result = await session.call_tool("hython_list_nodes", {})
                print("Result:", json.dumps(result.content, indent=2))
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_hython_tools())
