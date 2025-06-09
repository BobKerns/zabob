#!/usr/bin/env hython
"""
Hython MCP Server - Runs within hython environment with direct hou access.
This server provides sophisticated Houdini scene analysis and code generation.
"""

import json
from typing import Any

import hou
from fastmcp import FastMCP

# Import our scene analysis functions
from zabob.h20_5.node_loader import analyze_houdini_scene, extract_non_default_parms

mcp = FastMCP("Houdini Scene Analysis Server")

@mcp.tool("analyze_scene")
async def analyze_scene(file_path: str) -> dict[str, Any]:
    """
    Analyze a Houdini scene file and return structured data with generated Python code.

    Args:
        file_path: Path to the Houdini scene file (.hip, .hipnc, .hda)

    Returns:
        Dictionary containing analysis results, statistics, and generated Python code
    """
    try:
        result = analyze_houdini_scene(file_path)
        return result
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to analyze scene: {str(e)}",
            "file_path": file_path
        }

@mcp.tool("get_scene_info")
async def get_scene_info() -> dict[str, Any]:
    """
    Get information about the current Houdini scene.

    Returns:
        Dictionary containing current scene information
    """
    try:
        current_file = hou.hipFile.name()
        all_nodes = hou.node('/').allSubChildren()
        real_nodes = [n for n in all_nodes if n.parent() is not None]

        categories = {}
        for node in real_nodes:
            cat = node.type().category().name()
            if cat not in categories:
                categories[cat] = 0
            categories[cat] += 1

        return {
            "success": True,
            "current_file": current_file,
            "total_nodes": len(real_nodes),
            "categories": categories,
            "modified": hou.hipFile.hasUnsavedChanges()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@mcp.tool("get_node_info")
async def get_node_info(node_path: str) -> dict[str, Any]:
    """
    Get detailed information about a specific node.

    Args:
        node_path: Path to the Houdini node (e.g., "/obj/geo1")

    Returns:
        Dictionary containing node information
    """
    try:
        node = hou.node(node_path)
        if not node:
            return {
                "success": False,
                "error": f"Node not found: {node_path}"
            }

        non_default_parms = extract_non_default_parms(node)

        inputs = []
        if hasattr(node, 'inputs'):
            for i, inp in enumerate(node.inputs()):
                inputs.append({
                    "index": i,
                    "node": inp.path() if inp else None
                })

        outputs = []
        if hasattr(node, 'outputs'):
            for i, out in enumerate(node.outputs()):
                outputs.append({
                    "index": i,
                    "node": out.path() if out else None
                })

        return {
            "success": True,
            "path": node.path(),
            "name": node.name(),
            "type": node.type().name(),
            "category": node.type().category().name(),
            "non_default_parameters": non_default_parms,
            "inputs": inputs,
            "outputs": outputs,
            "creation_time": node.creationTime() if hasattr(node, 'creationTime') else None
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "node_path": node_path
        }

@mcp.tool("list_nodes")
async def list_nodes(parent_path: str = "/") -> dict[str, Any]:
    """
    List all nodes under a given parent path.

    Args:
        parent_path: Parent node path to search under

    Returns:
        Dictionary containing list of nodes
    """
    try:
        parent = hou.node(parent_path)
        if not parent:
            return {
                "success": False,
                "error": f"Parent node not found: {parent_path}"
            }

        children = parent.allSubChildren()
        nodes_info = []

        for node in children:
            nodes_info.append({
                "path": node.path(),
                "name": node.name(),
                "type": node.type().name(),
                "category": node.type().category().name(),
                "parent": node.parent().path() if node.parent() else None
            })

        return {
            "success": True,
            "parent_path": parent_path,
            "total_nodes": len(nodes_info),
            "nodes": nodes_info
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "parent_path": parent_path
        }

@mcp.tool("create_node")
async def create_node(parent_path: str, node_type: str, name: str = "", **kwargs) -> dict[str, Any]:
    """
    Create a new node in the scene.

    Args:
        parent_path: Path to parent node where new node will be created
        node_type: Type of node to create (e.g., "geo", "merge", "transform")
        name: Optional name for the new node
        **kwargs: Optional parameter values to set on the new node

    Returns:
        Dictionary containing information about the created node
    """
    try:
        parent = hou.node(parent_path)
        if not parent:
            return {
                "success": False,
                "error": f"Parent node not found: {parent_path}"
            }

        new_node = parent.createNode(node_type, name)

        # Set any provided parameters
        for parm_name, value in kwargs.items():
            parm = new_node.parm(parm_name)
            if parm:
                parm.set(value)

        return {
            "success": True,
            "node_path": new_node.path(),
            "node_name": new_node.name(),
            "node_type": new_node.type().name(),
            "parent_path": parent_path
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "parent_path": parent_path,
            "node_type": node_type
        }

@mcp.resource("scene://current")
async def get_current_scene() -> str:
    """Get information about the current scene as a text resource."""
    try:
        # Get scene info directly instead of calling the tool function
        current_file = hou.hipFile.name()
        all_nodes = hou.node('/').allSubChildren()
        real_nodes = [n for n in all_nodes if n.parent() is not None]

        categories = {}
        for node in real_nodes:
            cat = node.type().category().name()
            if cat not in categories:
                categories[cat] = 0
            categories[cat] += 1

        scene_info = {
            "success": True,
            "current_file": current_file,
            "total_nodes": len(real_nodes),
            "categories": categories,
            "modified": hou.hipFile.hasUnsavedChanges()
        }

        return json.dumps(scene_info, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    """Main entry point for the hython MCP server."""
    # Run the FastMCP server
    mcp.run()

if __name__ == "__main__":
    main()
