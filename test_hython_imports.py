#!/usr/bin/env hython
"""Test script to verify hython MCP server imports and setup."""

import sys
import traceback

def test_imports():
    """Test if all imports work correctly."""
    try:
        print("Testing imports...")

        # Test basic imports
        import hou
        print("✅ hou import successful")

        from fastmcp import FastMCP
        print("✅ FastMCP import successful")

        # Test the module import
        sys.path.insert(0, '/Users/rwk/p/zabob/houdini/h20.5/src')
        import zabob.h20_5.hython_mcp_server
        print("✅ hython_mcp_server import successful")

        # Test that the MCP instance is created
        mcp_instance = zabob.h20_5.hython_mcp_server.mcp
        print(f"✅ MCP instance created: {type(mcp_instance)}")

        print("🎉 All imports and setup successful!")
        return True

    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
