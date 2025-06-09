#!/usr/bin/env python3
"""Test script to verify hython MCP server imports work correctly."""

import subprocess
import sys
from pathlib import Path

# Find the hython wrapper
hython_wrapper_path = Path("/Users/rwk/p/zabob/houdini/zcommon/src/zabob/common/hython.py")

def test_hython_server_import():
    """Test if the hython MCP server can be imported without errors."""
    try:
        # Test import of the hython server module
        result = subprocess.run(
            ["python", str(hython_wrapper_path), "-c",
             "import sys; sys.path.insert(0, '/Users/rwk/p/zabob/houdini/h20.5/src'); import zabob.h20_5.hython_mcp_server; print('✅ Hython MCP server import successful!')"],
            capture_output=True,
            text=True,
            timeout=30
        )

        print("Return code:", result.returncode)
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        if result.returncode == 0:
            print("✅ Import test passed!")
            return True
        else:
            print("❌ Import test failed!")
            return False

    except subprocess.TimeoutExpired:
        print("❌ Import test timed out!")
        return False
    except Exception as e:
        print(f"❌ Import test failed with exception: {e}")
        return False

if __name__ == "__main__":
    print("Testing hython MCP server import...")
    success = test_hython_server_import()
    sys.exit(0 if success else 1)
