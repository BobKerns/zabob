"""
Code generation utilities for Houdini node analysis.
Generates Python code to recreate Houdini scenes from analyzed node data.
"""

import hou
from zabob.h20_5.parameter_extraction import format_parm_value


def generate_chain_code(chain, chain_info, non_default_parms=None):
    """
    Generate Python code to create a chain of connected nodes.

    Args:
        chain: List of nodes in the chain
        chain_info: Chain information dict
        non_default_parms: Dict of node -> parameters for the chain nodes

    Returns:
        str: Python code to create the chain
    """
    if not chain:
        return ""

    code_lines = []
    chain_name = chain_info.get('chain_name', 'unknown_chain')

    # Add comment header
    code_lines.append(f"# Create chain: {chain_name}")
    code_lines.append("")

    # Generate code for each node in the chain
    prev_var = None

    for i, node in enumerate(chain):
        node_name = node.name()
        node_type = node.type().name()
        var_name = f"{node_name}_node"

        # Create the node
        if i == 0:
            # First node in chain - create at parent level
            parent_path = node.parent().path() if node.parent() else "/obj"
            code_lines.append(f'{var_name} = hou.node("{parent_path}").createNode("{node_type}", "{node_name}")')
        else:
            # Subsequent nodes - create and connect
            code_lines.append(f'{var_name} = {prev_var}.createOutputNode("{node_type}", "{node_name}")')

        # Set non-default parameters
        if non_default_parms and node in non_default_parms:
            node_parms = non_default_parms[node]
            if node_parms:
                code_lines.append("")
                code_lines.append(f"# Set parameters for {node_name}")
                for parm_name, parm_value in node_parms.items():
                    formatted_value = format_parm_value(parm_value)
                    code_lines.append(f'{var_name}.parm("{parm_name}").set({formatted_value})')

        # Set position (if available)
        try:
            pos = node.position()
            code_lines.append(f'{var_name}.setPosition([{pos[0]}, {pos[1]}])')
        except:
            pass

        code_lines.append("")
        prev_var = var_name

    return "\n".join(code_lines)


def generate_connection_code(connections, chains):
    """
    Generate Python code for explicit connections between chains.

    Args:
        connections: List of connection tuples
        chains: List of node chains

    Returns:
        str: Python code for connections
    """
    if not connections:
        return ""

    code_lines = []
    code_lines.append("# Create explicit connections between chains")
    code_lines.append("")

    for source_chain_idx, target_chain_idx, connection_info in connections:
        source_chain = chains[source_chain_idx]
        target_chain = chains[target_chain_idx]

        source_node = connection_info['source_node']
        target_node = connection_info['target_node']

        source_var = f"{source_node.name()}_node"
        target_var = f"{target_node.name()}_node"

        # Generate connection code
        output_idx = connection_info.get('source_output', 0)
        input_idx = connection_info.get('target_input', 0)

        code_lines.append(f"# Connect {source_node.name()} to {target_node.name()}")
        code_lines.append(f"{target_var}.setInput({input_idx}, {source_var}, {output_idx})")
        code_lines.append("")

    return "\n".join(code_lines)


def generate_node_code(node, var_name=None, non_default_parms=None):
    """
    Generate Python code to create a single node.

    Args:
        node: Houdini node to generate code for
        var_name: Variable name to use (auto-generated if None)
        non_default_parms: Dict of non-default parameters for the node

    Returns:
        str: Python code to create the node
    """
    if var_name is None:
        var_name = f"{node.name()}_node"

    code_lines = []
    node_name = node.name()
    node_type = node.type().name()

    # Create the node
    parent_path = node.parent().path() if node.parent() else "/obj"
    code_lines.append(f'{var_name} = hou.node("{parent_path}").createNode("{node_type}", "{node_name}")')

    # Set non-default parameters
    if non_default_parms:
        code_lines.append("")
        code_lines.append(f"# Set parameters for {node_name}")
        for parm_name, parm_value in non_default_parms.items():
            formatted_value = format_parm_value(parm_value)
            code_lines.append(f'{var_name}.parm("{parm_name}").set({formatted_value})')

    # Set position
    try:
        pos = node.position()
        code_lines.append(f'{var_name}.setPosition([{pos[0]}, {pos[1]}])')
    except:
        pass

    return "\n".join(code_lines)


def generate_network_code(chains, connections, creation_order, all_non_default_parms=None):
    """
    Generate complete Python code to recreate a node network.

    Args:
        chains: List of node chains
        connections: List of connections between chains
        creation_order: Order in which to create chains
        all_non_default_parms: Dict of all non-default parameters by node

    Returns:
        str: Complete Python code to recreate the network
    """
    code_lines = []

    # Add header
    code_lines.append("#!/usr/bin/env hython")
    code_lines.append("# Generated Houdini scene recreation script")
    code_lines.append("# Created by enhanced node_load.py")
    code_lines.append("")
    code_lines.append("import hou")
    code_lines.append("")

    # Generate code for each chain in dependency order
    for chain_idx in creation_order:
        if chain_idx < len(chains):
            chain = chains[chain_idx]
            chain_info = {
                'chain_name': f"chain_{chain_idx}",
                'first_node': chain[0].name() if chain else None,
                'last_node': chain[-1].name() if chain else None
            }

            # Get parameters for nodes in this chain
            chain_parms = {}
            if all_non_default_parms:
                for node in chain:
                    if node in all_non_default_parms:
                        chain_parms[node] = all_non_default_parms[node]

            # Generate chain creation code
            chain_code = generate_chain_code(chain, chain_info, chain_parms)
            if chain_code:
                code_lines.append(chain_code)
                code_lines.append("")

    # Generate connection code
    connection_code = generate_connection_code(connections, chains)
    if connection_code:
        code_lines.append(connection_code)

    # Add footer
    code_lines.append("# Layout nodes for better visualization")
    code_lines.append("# hou.node('/obj').layoutChildren()")
    code_lines.append("")
    code_lines.append("print('Scene recreation complete!')")

    return "\n".join(code_lines)


def generate_summary_comment(chains, all_nodes, analysis_info=None):
    """
    Generate a summary comment describing the analyzed scene.

    Args:
        chains: List of node chains
        all_nodes: List of all nodes
        analysis_info: Optional analysis information

    Returns:
        str: Summary comment text
    """
    lines = []
    lines.append("# Scene Analysis Summary:")
    lines.append(f"# Total nodes: {len(all_nodes)}")
    lines.append(f"# Total chains: {len(chains)}")

    if chains:
        chain_lengths = [len(chain) for chain in chains]
        lines.append(f"# Chain lengths: min={min(chain_lengths)}, max={max(chain_lengths)}, avg={sum(chain_lengths)/len(chain_lengths):.1f}")

    if analysis_info:
        lines.append(f"# Isolated chains: {len(analysis_info.get('isolated_chains', []))}")
        lines.append(f"# Complex dependencies: {len(analysis_info.get('complex_dependencies', []))}")

    lines.append("#")

    return "\n".join(lines)


def format_code_with_style(code):
    """
    Apply basic code formatting and style improvements.

    Args:
        code: Raw Python code string

    Returns:
        str: Formatted code
    """
    lines = code.split('\n')
    formatted_lines = []

    for line in lines:
        # Remove trailing whitespace
        line = line.rstrip()

        # Add proper spacing around operators (basic)
        # This could be expanded with more sophisticated formatting

        formatted_lines.append(line)

    # Ensure proper spacing between sections
    result_lines = []
    prev_was_empty = False

    for line in formatted_lines:
        is_empty = len(line.strip()) == 0

        # Avoid multiple consecutive empty lines
        if is_empty and prev_was_empty:
            continue

        result_lines.append(line)
        prev_was_empty = is_empty

    return '\n'.join(result_lines)
