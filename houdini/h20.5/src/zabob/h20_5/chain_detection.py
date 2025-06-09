"""
Linear chain detection utilities for Houdini node analysis.
Detects linear chains of connected nodes to simplify topology representation.
"""

import hou


def detect_linear_chains(nodes):
    """
    Detect linear chains of connected nodes.
    A chain is a sequence of nodes where each node has exactly one input
    and one output (except for the first and last nodes).

    Args:
        nodes: List of nodes to analyze

    Returns:
        list: List of chains, where each chain is a list of nodes in order
    """
    # Build adjacency information
    node_inputs = {}
    node_outputs = {}

    for node in nodes:
        node_inputs[node] = []
        node_outputs[node] = []

        # Get input connections
        if hasattr(node, 'inputs') and callable(node.inputs):
            for input_node in node.inputs():
                if input_node and input_node in nodes:
                    node_inputs[node].append(input_node)

        # Get output connections
        if hasattr(node, 'outputs') and callable(node.outputs):
            for output_node in node.outputs():
                if output_node and output_node in nodes:
                    node_outputs[node].append(output_node)

    # Find chain start nodes (nodes with 0 or >1 inputs, or >1 outputs)
    visited = set()
    chains = []

    for node in nodes:
        if node in visited:
            continue

        # Check if this could be the start of a chain
        num_inputs = len(node_inputs[node])
        num_outputs = len(node_outputs[node])

        # Chain starts: nodes with 0 inputs, >1 inputs, or >1 outputs
        if num_inputs != 1 or num_outputs > 1:
            chain = _build_chain_from_start(node, node_inputs, node_outputs, visited)
            if len(chain) > 1:  # Only include chains with multiple nodes
                chains.append(chain)
            elif len(chain) == 1:
                # Single node "chain"
                chains.append(chain)

    # Also check for any unvisited nodes (isolated cycles, etc.)
    for node in nodes:
        if node not in visited:
            chain = _build_chain_from_start(node, node_inputs, node_outputs, visited)
            if chain:
                chains.append(chain)

    return chains


def _build_chain_from_start(start_node, node_inputs, node_outputs, visited):
    """
    Build a linear chain starting from a given node.

    Args:
        start_node: Node to start the chain from
        node_inputs: Dict of node -> input nodes
        node_outputs: Dict of node -> output nodes
        visited: Set of already visited nodes

    Returns:
        list: Chain of nodes starting from start_node
    """
    chain = []
    current_node = start_node

    while current_node and current_node not in visited:
        visited.add(current_node)
        chain.append(current_node)

        # Follow the chain if there's exactly one output and that output
        # has exactly one input (this node)
        outputs = node_outputs[current_node]
        if len(outputs) == 1:
            next_node = outputs[0]
            next_inputs = node_inputs[next_node]

            # Continue chain only if next node has exactly one input (this node)
            if len(next_inputs) == 1 and next_inputs[0] == current_node:
                current_node = next_node
            else:
                break
        else:
            break

    return chain


def find_chain_connections(chains, all_nodes):
    """
    Find connections between chains and individual nodes.

    Args:
        chains: List of node chains
        all_nodes: List of all nodes in the network

    Returns:
        list: List of connection tuples (source_chain_idx, target_chain_idx, connection_info)
    """
    connections = []

    # Create mapping from node to chain index
    node_to_chain = {}
    for chain_idx, chain in enumerate(chains):
        for node in chain:
            node_to_chain[node] = chain_idx

    # Find connections between chains
    for chain_idx, chain in enumerate(chains):
        # Check outputs of the last node in the chain
        last_node = chain[-1]

        if hasattr(last_node, 'outputs') and callable(last_node.outputs):
            for output_node in last_node.outputs():
                if output_node in node_to_chain:
                    target_chain_idx = node_to_chain[output_node]
                    if target_chain_idx != chain_idx:
                        # This is a connection between different chains
                        target_chain = chains[target_chain_idx]
                        target_node_idx = target_chain.index(output_node)

                        connection_info = {
                            'source_node': last_node,
                            'target_node': output_node,
                            'target_node_idx': target_node_idx,
                            'source_output': 0,  # Could be enhanced to find actual output index
                            'target_input': 0    # Could be enhanced to find actual input index
                        }

                        connections.append((chain_idx, target_chain_idx, connection_info))

    return connections


def get_chain_info(chain):
    """
    Get information about a chain for code generation.

    Args:
        chain: List of nodes in the chain

    Returns:
        dict: Chain information including names, types, etc.
    """
    if not chain:
        return {}

    return {
        'nodes': [node.name() for node in chain],
        'types': [node.type().name() for node in chain],
        'first_node': chain[0].name(),
        'last_node': chain[-1].name(),
        'length': len(chain),
        'chain_name': f"chain_{chain[0].name()}_to_{chain[-1].name()}" if len(chain) > 1 else chain[0].name()
    }


def is_simple_chain(chain):
    """
    Check if a chain is a simple linear sequence with no branching.

    Args:
        chain: List of nodes in the chain

    Returns:
        bool: True if the chain is simple (no internal branching)
    """
    if len(chain) <= 1:
        return True

    for i, node in enumerate(chain):
        # Check inputs (except for first node)
        if i > 0:
            if hasattr(node, 'inputs') and callable(node.inputs):
                inputs = [inp for inp in node.inputs() if inp is not None]
                if len(inputs) != 1 or inputs[0] != chain[i-1]:
                    return False

        # Check outputs (except for last node)
        if i < len(chain) - 1:
            if hasattr(node, 'outputs') and callable(node.outputs):
                # For internal chain nodes, outputs should only go to next in chain
                # But we allow additional outputs to nodes outside the chain
                outputs = [out for out in node.outputs() if out is not None]
                if chain[i+1] not in outputs:
                    return False

    return True
