"""
Topological sorting utilities for Houdini node analysis.
Provides chain-aware topological sorting for proper dependency ordering.
"""

from collections import deque
import hou


def topological_sort_with_chains(chains, connections):
    """
    Perform topological sort on chains, treating each chain as a single unit.
    This ensures proper dependency ordering for code generation.

    Args:
        chains: List of node chains
        connections: List of connection tuples between chains

    Returns:
        list: Ordered list of chain indices in dependency order
    """
    if not chains:
        return []

    num_chains = len(chains)

    # Build adjacency list and in-degree count
    adj_list = [[] for _ in range(num_chains)]
    in_degree = [0] * num_chains

    for source_idx, target_idx, connection_info in connections:
        if 0 <= source_idx < num_chains and 0 <= target_idx < num_chains:
            adj_list[source_idx].append(target_idx)
            in_degree[target_idx] += 1

    # Initialize queue with chains that have no dependencies
    queue = deque()
    for i in range(num_chains):
        if in_degree[i] == 0:
            queue.append(i)

    # Perform topological sort using Kahn's algorithm
    result = []

    while queue:
        current = queue.popleft()
        result.append(current)

        # Reduce in-degree of adjacent chains
        for neighbor in adj_list[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Check for cycles
    if len(result) != num_chains:
        print("Warning: Cycle detected in node dependencies!")
        # Add remaining chains to result (best effort)
        for i in range(num_chains):
            if i not in result:
                result.append(i)

    return result


def sort_nodes_by_dependency(nodes):
    """
    Sort individual nodes by dependency order (simpler version without chains).

    Args:
        nodes: List of nodes to sort

    Returns:
        list: Nodes sorted in dependency order
    """
    if not nodes:
        return []

    # Build adjacency information
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    adj_list = [[] for _ in range(len(nodes))]
    in_degree = [0] * len(nodes)

    for i, node in enumerate(nodes):
        if hasattr(node, 'inputs') and callable(node.inputs):
            for input_node in node.inputs():
                if input_node and input_node in node_to_idx:
                    source_idx = node_to_idx[input_node]
                    adj_list[source_idx].append(i)
                    in_degree[i] += 1

    # Topological sort
    queue = deque()
    for i in range(len(nodes)):
        if in_degree[i] == 0:
            queue.append(i)

    result_indices = []
    while queue:
        current = queue.popleft()
        result_indices.append(current)

        for neighbor in adj_list[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Check for cycles and handle incomplete sorts
    if len(result_indices) != len(nodes):
        print("Warning: Cycle detected in node dependencies!")
        # Add remaining nodes
        for i in range(len(nodes)):
            if i not in result_indices:
                result_indices.append(i)

    return [nodes[i] for i in result_indices]


def analyze_dependencies(chains, all_nodes):
    """
    Analyze dependency relationships between chains and nodes.

    Args:
        chains: List of node chains
        all_nodes: List of all nodes

    Returns:
        dict: Dependency analysis results
    """
    analysis = {
        'total_chains': len(chains),
        'total_nodes': len(all_nodes),
        'chain_dependencies': [],
        'isolated_chains': [],
        'complex_dependencies': []
    }

    # Create mapping from node to chain
    node_to_chain = {}
    for chain_idx, chain in enumerate(chains):
        for node in chain:
            node_to_chain[node] = chain_idx

    # Analyze each chain's dependencies
    for chain_idx, chain in enumerate(chains):
        deps = {
            'chain_idx': chain_idx,
            'chain_length': len(chain),
            'depends_on': set(),
            'dependents': set(),
            'external_inputs': 0,
            'external_outputs': 0
        }

        # Check first node for external inputs
        first_node = chain[0]
        if hasattr(first_node, 'inputs') and callable(first_node.inputs):
            for input_node in first_node.inputs():
                if input_node and input_node in node_to_chain:
                    source_chain = node_to_chain[input_node]
                    if source_chain != chain_idx:
                        deps['depends_on'].add(source_chain)
                elif input_node:
                    deps['external_inputs'] += 1

        # Check last node for external outputs
        last_node = chain[-1]
        if hasattr(last_node, 'outputs') and callable(last_node.outputs):
            for output_node in last_node.outputs():
                if output_node and output_node in node_to_chain:
                    target_chain = node_to_chain[output_node]
                    if target_chain != chain_idx:
                        deps['dependents'].add(target_chain)
                elif output_node:
                    deps['external_outputs'] += 1

        # Check for complex dependencies (internal chain nodes with external connections)
        complex_deps = 0
        for i, node in enumerate(chain[1:-1], 1):  # Skip first and last
            if hasattr(node, 'inputs') and callable(node.inputs):
                external_inputs = [inp for inp in node.inputs()
                                 if inp and inp not in chain]
                complex_deps += len(external_inputs)

            if hasattr(node, 'outputs') and callable(node.outputs):
                external_outputs = [out for out in node.outputs()
                                  if out and out not in chain]
                complex_deps += len(external_outputs)

        if complex_deps > 0:
            analysis['complex_dependencies'].append(chain_idx)

        if not deps['depends_on'] and not deps['dependents']:
            analysis['isolated_chains'].append(chain_idx)

        analysis['chain_dependencies'].append(deps)

    return analysis


def get_creation_order(chains, connections):
    """
    Get the optimal order for creating chains in generated code.

    Args:
        chains: List of node chains
        connections: List of connections between chains

    Returns:
        dict: Creation order information
    """
    sorted_indices = topological_sort_with_chains(chains, connections)

    return {
        'creation_order': sorted_indices,
        'chains_info': [
            {
                'index': idx,
                'first_node': chains[idx][0].name() if chains[idx] else None,
                'last_node': chains[idx][-1].name() if chains[idx] else None,
                'length': len(chains[idx])
            }
            for idx in sorted_indices
        ]
    }
