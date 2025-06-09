#!/bin/bash
# Start Houdini MCP Server in hython environment using zabob infrastructure

# Set default values
PORT=${HOUDINI_MCP_PORT:-8765}
HOST=${HOUDINI_MCP_HOST:-localhost}
VERSION=${HOUDINI_VERSION:-20.5}

echo "Starting Houdini MCP Server..."
echo "Host: $HOST"
echo "Port: $PORT"  
echo "Houdini Version: $VERSION"
echo

# Change to the project directory to ensure proper module resolution
cd "$(dirname "$0")" || exit 1

# Use the zabob hython command to start the MCP server
# This ensures proper environment setup and hython execution
exec uv run --from "../zcommon" zabob.common.hython \
    --version "$VERSION" \
    --module zabob.h20_5.houdini_mcp_server \
    --host "$HOST" \
    --port "$PORT" \
    "$@"
