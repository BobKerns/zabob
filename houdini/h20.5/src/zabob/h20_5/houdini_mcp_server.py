#!/usr/bin/env python3
"""
Houdini MCP Server - FastMCP server that runs Houdini analysis in hython environment.

This server provides access to Houdini scene analysis and code generation
functionality through the Model Context Protocol (MCP). It uses the zabob.common.hython
infrastructure to properly execute Houdini code in the hython environment.
"""

import os
import sys
import json
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from fastmcp import FastMCP
from semver import Version

# Import zabob.common.hython for proper hython execution
from zabob.common.hython import run_houdini_script

# Create the FastMCP server
mcp = FastMCP("Houdini Scene Analyzer")


@mcp.tool()
def analyze_houdini_scene(file_path: str) -> Dict[str, Any]:
    """
    Analyze a Houdini scene file and generate Python code to recreate it.

    Args:
        file_path: Path to the Houdini file (.hip, .hipnc, .hda, etc.)

    Returns:
        Dictionary containing analysis results and generated Python code
    """
    try:
        # Ensure file exists
        if not os.path.exists(file_path):
            return {
                "success": False,
                "error": f"File not found: {file_path}"
            }

        # Create a temporary script to run the analysis in hython
        script_content = f'''
import sys
import json
from zabob.h20_5.node_load import analyze_houdini_scene

try:
    result = analyze_houdini_scene("{file_path}")
    print(json.dumps(result, indent=2))
except Exception as e:
    import traceback
    error_result = {{
        "success": False,
        "error": str(e),
        "traceback": traceback.format_exc()
    }}
    print(json.dumps(error_result, indent=2))
'''

        # Write the script to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_file:
            tmp_file.write(script_content)
            tmp_script_path = tmp_file.name

        try:
            # Run the script in hython and capture output
            output = run_houdini_script(
                tmp_script_path,
                capture_output=True
            )
            
            # Parse the JSON output
            result = json.loads(output)
            return result
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_script_path)

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        real_nodes = [
            n for n in nodes
            if n.parent() is not None and n.parent() is not ROOT_NODE
        ]

        if not real_nodes:
            return {
                "success": True,
                "message": "No nodes found in scene",
                "code": "# Empty scene - no nodes to recreate",
                "statistics": {
                    "total_nodes": 0,
                    "chains": 0,
                    "node_references": 0
                }
            }

        # Extract scene information
        types = {n.type() for n in real_nodes}
        categories = {n.type().category() for n in real_nodes}
        parents_used = {n.parent() for n in real_nodes}

        # Enhanced analysis: extract parameters and node references
        all_node_refs = {}
        all_nodes_set = set(real_nodes)

        for node in real_nodes:
            node_refs = find_node_references_in_parms(node, all_nodes_set)
            if node_refs:
                all_node_refs[node] = node_refs

        # Detect linear chains (basic blocks)
        chains = detect_linear_chains(real_nodes)

        # Perform topological sort
        sorted_chains = topological_sort_with_chains(real_nodes, chains)

        # Generate Python code
        code_lines = []

        # Add header
        code_lines.append("#!/usr/bin/env hython")
        code_lines.append("# Generated Houdini scene recreation script")
        code_lines.append(f"# Created from: {file_path}")
        code_lines.append("# Created by Houdini MCP Server")
        code_lines.append("")
        code_lines.append("import hou")
        code_lines.append("from zabob.h20_5.node_load import *")
        code_lines.append("")

        # Generate function definitions for used categories
        for c in categories:
            fn = category_fns.get(c)
            if fn is not None:
                fn_name = fn.__name__
                code_lines.append(f"def {fn_name}(node_type: str, name: str='', **kwargs):")
                code_lines.append(f'    """Create or check a {c.name()} node."""')
                code_lines.append(f"    node = _parent.get().createNode(node_type, name)")
                code_lines.append(f"    for parm_name, value in kwargs.items():")
                code_lines.append(f"        if node.parm(parm_name):")
                code_lines.append(f"            node.parm(parm_name).set(value)")
                code_lines.append(f"    return node")
        code_lines.append("")

        # Generate parent node setup
        for n in parents_used:
            fn = category_fns.get(n.type().category())
            if fn is not None:
                fn_name = fn.__name__
                code_lines.append(f"{good_var(n)} = {fn_name}(hou.node(f'/{n.name()}'))")
        code_lines.append("")

        # Generate chains in topological order
        code_lines.append("# Create nodes in topological order")
        for chain in sorted_chains:
            if not chain:
                continue

            # Check if chain nodes are in a parent container
            parent_node = chain[0].parent()
            if parent_node != ROOT_NODE and parent_node in parents_used:
                code_lines.append(f'with parent({good_var(parent_node)}):')
                chain_code = generate_chain_code(chain, all_node_refs)
                for line in chain_code.split('\n'):
                    if line.strip():
                        code_lines.append(f"    {line}")
            else:
                chain_code = generate_chain_code(chain, all_node_refs)
                code_lines.append(chain_code)
            code_lines.append("")

        # Generate explicit connections section
        code_lines.append("# Scene recreation complete!")
        code_lines.append("# You can now run this script with: hython <script_name>.py")

        generated_code = "\n".join(code_lines)

        return {
            "success": True,
            "file_path": file_path,
            "code": generated_code,
            "statistics": {
                "total_nodes": len(real_nodes),
                "node_types": [t.name() for t in types],
                "categories": [c.name() for c in categories],
                "chains": len(chains),
                "node_references": len(all_node_refs),
                "parents_used": len(parents_used)
            },
            "chains_info": [
                {
                    "chain_index": i,
                    "length": len(chain),
                    "first_node": chain[0].name() if chain else None,
                    "last_node": chain[-1].name() if chain else None,
                    "node_names": [n.name() for n in chain]
                }
                for i, chain in enumerate(chains)
            ]
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@mcp.tool()
def get_current_scene_info() -> Dict[str, Any]:
    """
    Get information about the currently loaded Houdini scene.

    Returns:
        Dictionary containing current scene information
    """
    try:
        current_file = hou.hipFile.name()
        nodes = hou.node('/').allSubChildren()
        real_nodes = [
            n for n in nodes
            if n.parent() is not None and n.parent() is not ROOT_NODE
        ]

        if not real_nodes:
            return {
                "success": True,
                "current_file": current_file,
                "message": "No nodes in current scene"
            }

        types = {n.type() for n in real_nodes}
        categories = {n.type().category() for n in real_nodes}

        return {
            "success": True,
            "current_file": current_file,
            "statistics": {
                "total_nodes": len(real_nodes),
                "node_types": [t.name() for t in types],
                "categories": [c.name() for c in categories]
            },
            "node_list": [
                {
                    "name": n.name(),
                    "type": n.type().name(),
                    "category": n.type().category().name(),
                    "path": n.path()
                }
                for n in real_nodes
            ]
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@mcp.tool()
def list_available_files(directory: str = "/tmp") -> Dict[str, Any]:
    """
    List available Houdini files in a directory.

    Args:
        directory: Directory to search for Houdini files

    Returns:
        List of found Houdini files
    """
    try:
        if not os.path.exists(directory):
            return {
                "success": False,
                "error": f"Directory not found: {directory}"
            }

        houdini_extensions = {'.hip', '.hipnc', '.hda', '.otl', '.hdanc'}
        found_files = []

        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in houdini_extensions):
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, directory)

                    try:
                        stat = os.stat(full_path)
                        found_files.append({
                            "name": file,
                            "path": full_path,
                            "relative_path": relative_path,
                            "size": stat.st_size,
                            "modified": stat.st_mtime
                        })
                    except OSError:
                        continue

        return {
            "success": True,
            "directory": directory,
            "files": found_files,
            "count": len(found_files)
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def main():
    """Main entry point for the Houdini MCP server."""
    import argparse

    parser = argparse.ArgumentParser(description="Houdini MCP Server")
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to run the server on (default: 8765)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to bind the server to (default: localhost)"
    )

    args = parser.parse_args()

    print(f"Starting Houdini MCP Server on {args.host}:{args.port}")
    print("Available tools:")
    print("  - analyze_houdini_scene: Analyze a Houdini file and generate recreation code")
    print("  - get_current_scene_info: Get info about currently loaded scene")
    print("  - list_available_files: List Houdini files in a directory")
    print()

    # Run the server
    mcp.run(
        host=args.host,
        port=args.port,
        transport="sse"  # Server-Sent Events transport
    )


if __name__ == "__main__":
    main()
