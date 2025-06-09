#!/usr/bin/env python3
"""
Test FastMCP import in hython environment.
"""

try:
    import fastmcp
    print("✅ FastMCP import successful!")
    print(f"   FastMCP location: {fastmcp.__file__}")

    from mcp.server.fastmcp import FastMCP
    print("✅ FastMCP.FastMCP import successful!")

    print("✅ All imports working correctly!")

except ImportError as e:
    print(f"❌ Import failed: {e}")

    # Show current Python path for debugging
    import sys
    print("\nCurrent Python path:")
    for i, path in enumerate(sys.path):
        print(f"  {i}: {path}")
