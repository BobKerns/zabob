"""
Node reference detection utilities for Houdini node analysis.
Finds references to other nodes within parameter values.
"""

import re
import hou


def find_node_references_in_parms(node):
    """
    Find references to other nodes in parameter values.
    This includes channel references, node path references, etc.

    Args:
        node: Houdini node to analyze

    Returns:
        dict: Parameter name -> list of referenced node paths
    """
    references = {}

    try:
        for parm_tuple in node.parmTuples():
            try:
                parm_refs = []

                for parm in parm_tuple:
                    # Get the raw (unexpanded) parameter value
                    raw_value = parm.unexpandedString()

                    # Find channel references (e.g., ch("../node/parm"))
                    channel_refs = _find_channel_references(raw_value)
                    parm_refs.extend(channel_refs)

                    # Find node path references (e.g., "../node" or "/obj/geo1")
                    path_refs = _find_node_path_references(raw_value)
                    parm_refs.extend(path_refs)

                    # Find other types of references
                    other_refs = _find_other_references(raw_value)
                    parm_refs.extend(other_refs)

                if parm_refs:
                    parm_name = parm_tuple[0].name()
                    references[parm_name] = list(set(parm_refs))  # Remove duplicates

            except Exception as e:
                print(f"Warning: Could not analyze parameter {parm_tuple}: {e}")
                continue

    except Exception as e:
        print(f"Warning: Could not analyze parameters for node {node.name()}: {e}")

    return references


def _find_channel_references(value_str):
    """
    Find channel references in a parameter value string.
    Channel references look like: ch("../node/parm") or ch('../node/parm')

    Args:
        value_str: Parameter value string to search

    Returns:
        list: List of referenced node paths
    """
    if not isinstance(value_str, str):
        return []

    references = []

    # Pattern for channel references: ch("path") or ch('path')
    ch_pattern = r'ch\s*\(\s*["\']([^"\']+)["\']\s*\)'

    matches = re.finditer(ch_pattern, value_str, re.IGNORECASE)
    for match in matches:
        ref_path = match.group(1)

        # Extract node path from the reference (remove parameter name)
        if '/' in ref_path:
            # Split path and parameter
            path_parts = ref_path.split('/')
            if path_parts[-1] and not path_parts[-1].startswith('.'):
                # Last part might be parameter name, remove it
                node_path = '/'.join(path_parts[:-1])
                if node_path:
                    references.append(node_path)
            else:
                references.append(ref_path)

    return references


def _find_node_path_references(value_str):
    """
    Find direct node path references in parameter values.

    Args:
        value_str: Parameter value string to search

    Returns:
        list: List of referenced node paths
    """
    if not isinstance(value_str, str):
        return []

    references = []

    # Pattern for node paths: starts with / or .. and contains /
    # Common patterns: "/obj/geo1", "../merge1", "../../light1"
    path_patterns = [
        r'(?:^|[\s,])(/[a-zA-Z0-9_/]+)(?:[\s,]|$)',  # Absolute paths
        r'(?:^|[\s,])((?:\.\./)+[a-zA-Z0-9_/]+)(?:[\s,]|$)',  # Relative paths with ../
        r'(?:^|[\s,])(\./[a-zA-Z0-9_/]+)(?:[\s,]|$)',  # Relative paths with ./
    ]

    for pattern in path_patterns:
        matches = re.finditer(pattern, value_str)
        for match in matches:
            path = match.group(1).strip()
            if path and len(path) > 1:  # Ignore single characters
                references.append(path)

    return references


def _find_other_references(value_str):
    """
    Find other types of node references that might exist in parameter values.

    Args:
        value_str: Parameter value string to search

    Returns:
        list: List of other referenced paths
    """
    if not isinstance(value_str, str):
        return []

    references = []

    # Pattern for expressions that might reference nodes
    # Example: point("../grid1", 0, "P", 0)
    expr_pattern = r'(?:point|prim|detail|vertex)\s*\(\s*["\']([^"\']+)["\']\s*,'

    matches = re.finditer(expr_pattern, value_str, re.IGNORECASE)
    for match in matches:
        ref_path = match.group(1)
        if ref_path and ref_path != "op:":  # Filter out special tokens
            references.append(ref_path)

    return references


def resolve_node_references(node, references):
    """
    Resolve node reference paths to actual node objects.

    Args:
        node: Source node containing the references
        references: Dict of parameter -> reference paths

    Returns:
        dict: Parameter name -> list of resolved node objects
    """
    resolved = {}

    for parm_name, ref_paths in references.items():
        resolved_nodes = []

        for ref_path in ref_paths:
            try:
                # Try to resolve the path relative to the source node
                if ref_path.startswith('/'):
                    # Absolute path
                    ref_node = hou.node(ref_path)
                else:
                    # Relative path
                    ref_node = node.node(ref_path)

                if ref_node is not None:
                    resolved_nodes.append(ref_node)
                else:
                    print(f"Warning: Could not resolve reference '{ref_path}' from node {node.path()}")

            except Exception as e:
                print(f"Warning: Error resolving reference '{ref_path}': {e}")
                continue

        if resolved_nodes:
            resolved[parm_name] = resolved_nodes

    return resolved
