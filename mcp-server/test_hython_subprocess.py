#!/usr/bin/env python3
"""
Test script to verify hython subprocess launch pattern.
"""

import subprocess
import sys
import time
from pathlib import Path

# Find the project root
ROOT = Path(__file__).parent.parent

def test_hython_launch():
    """Test launching hython MCP server via subprocess."""

    # Path to the hython MCP server script
    hython_server_path = ROOT / "houdini" / "h20.5" / "src" / "zabob" / "h20_5" / "hython_mcp_server.py"

    if not hython_server_path.exists():
        print(f"❌ Hython server script not found: {hython_server_path}")
        return False

    # Path to the hython.py wrapper script
    hython_wrapper_path = ROOT / "houdini" / "zcommon" / "src" / "zabob" / "common" / "hython.py"

    if not hython_wrapper_path.exists():
        print(f"❌ Hython wrapper script not found: {hython_wrapper_path}")
        return False

    print(f"🚀 Testing hython launch pattern with module...")
    print(f"   Wrapper: {hython_wrapper_path}")
    print(f"   Module:  zabob.h20_5.hython_mcp_server")

    try:
        # Launch the hython server via the wrapper using -m flag
        proc = subprocess.Popen(
            ["python", str(hython_wrapper_path), "-m", "zabob.h20_5.hython_mcp_server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0,
            cwd=str(ROOT)
        )

        print(f"✅ Process started with PID: {proc.pid}")

        # Give it a moment to start
        time.sleep(3)

        # Check if process is still running
        if proc.returncode is not None:
            print(f"❌ Process exited with return code: {proc.returncode}")
            if proc.stderr:
                stderr_output = proc.stderr.read()
                print(f"   Stderr: {stderr_output}")
            if proc.stdout:
                stdout_output = proc.stdout.read()
                print(f"   Stdout: {stdout_output}")
            return False
        else:
            print(f"✅ Process is still running after 3 seconds")

            # Try to communicate with it briefly
            try:
                # Send a simple test message
                test_request = '{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}\n'
                proc.stdin.write(test_request)
                proc.stdin.flush()

                # Wait a bit for response
                time.sleep(1)

                print("✅ Successfully sent test request")

            except Exception as e:
                print(f"⚠️  Warning: Could not send test request: {e}")

            # Terminate the process
            try:
                proc.terminate()
                proc.wait(timeout=5)
                print("✅ Process terminated successfully")
                return True
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                print("⚠️  Process had to be killed")
                return True

    except Exception as e:
        print(f"❌ Failed to launch process: {e}")
        return False

if __name__ == "__main__":
    success = test_hython_launch()
    sys.exit(0 if success else 1)
