#!/usr/bin/env hython
"""Test script to check if hython MCP server imports work."""

import sys
sys.path.insert(0, '/Users/rwk/p/zabob/houdini/h20.5/src')

try:
    import zabob.h20_5.hython_mcp_server
    print('✅ Hython MCP server import successful!')
except Exception as e:
    print(f'❌ Import failed: {e}')
    sys.exit(1)
