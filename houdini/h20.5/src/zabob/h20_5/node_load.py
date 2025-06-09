print("DEBUG: Starting Houdini HDA/Scene analysis script", file=sys.stderr)
#!/usr/bin/env hython
'''
Load Houdini HDA or HIP files and analyze their contents.
Sophisticated scene-to-code transpiler with runtime support.
'''

import re
import json
import sys
from io import StringIO
import click
import hou
from collections import defaultdict, deque
from collections.abc import Sequence
from contextvars import ContextVar
from contextlib import contextmanager
from itertools import count
from typing import Generic, LiteralString, Protocol, TypeVar, cast, Any
import hou

RE_BAD_CHARS = re.compile(r'[^a-zA-Z0-9_]+')

var_names = set(globals().keys())  # Get all global variable names
name_count = count(1)  # Create a counter for unique names
node_vars: dict[hou.Node, str] = {}

def good_var(node: hou.Node) -> str:
    """
    Convert a string to a valid python variable name by replacing invalid characters.
    Remembers the mapping of nodes to variable names to allow multiple references.

    Args:
        name (str): The input string to convert.

    Returns:
        str: A valid python variable name.
    """
    if node in node_vars:
        return node_vars[node]
    name = node.name()
    name = name.lstrip('0123456789')  # Remove leading digits
    name = RE_BAD_CHARS.sub('_', name).strip('_')
    if not name:
        name = node.type().name()
    while name in var_names:
        name = f"{name}_{next(name_count)}"
    var_names.add(name)
    node_vars[node] = name
    return name


_parent = ContextVar('parent_node', default=hou.node('/'))
'''
Context variable to store the parent node for node creation. Defaults to the root node.
'''

@contextmanager
def parent(node: hou.Node|str='/') -> hou.Node:
    """
    Context manager to yield the parent node of a given node.

    Args:
        node (hou.Node or str): The node to set as the parent. If a string is provided, it should be
            the path to an existing node.
    Yields:
        hou.Node: The node set as the parent within the context.
    """
    if isinstance(node, str):
        node = hou.node(node)
    old = _parent.get()
    _parent.set(node)
    try:
       yield node
    finally:
        _parent.set(old)

T = TypeVar('T', bound=hou.Node, covariant=True)
class NodeCreator(Protocol[T]):
    def __call__(self, type: LiteralString|hou.NodeType, name: str='', _inputs: Sequence[hou.Node|None]=(), **kwargs) -> hou.Node:
        """
        Create a Houdini node of the specified type and name, optionally connecting inputs.

        Args:
            type (LiteralString or hou.NodeType): The type of the node to create.
            name (str): The name of the node to create.
            _inputs (Sequence[hou.Node|None]): Optional input nodes to connect to the created node.
            **kwargs: Additional keyword arguments for node creation.

        Returns:
            hou.Node: The created Houdini node.
        """
        ...


def _create_fn(ntype: type[T], name: str)  -> NodeCreator [T]:
    def _create(node_type: LiteralString|hou.NodeType|T, name: str|None='',
            _inputs: Sequence[hou.Node|None]=(),
            **kwargs) -> T:
        """
        Create a Houdini node of the specified type and name, optionally connecting inputs.

        Args:
            type (LiteralString or hou.NodeType): The type of the node to create.
            name (str): The name of the node to create.
            _inputs (Sequence[hou.Node|None]): Optional input nodes to connect to the created node.
                    `None` values set the corresponding input to unconnected.
            **kwargs: Additional keyword arguments for node creation.

        Returns:
            hou.Node: The created Houdini node.
            """
        if isinstance(node_type, ntype):
            return node_type  # Already a node of the correct type
        if isinstance(node_type, hou.Node):
            raise TypeError(f"Expected a node of type {ntype.__name__}, got {node_type.name()}") # type: ignore[return-value]
        n = _parent.get().createNode(node_type, name, **kwargs)
        if _inputs:
            for i, input_node in enumerate(_inputs):
                n.setInput(i, input_node)
        # Remember for chaining
        setattr(n, '_inputs', _inputs)
        return n
    _create.__name__ = name
    return cast(NodeCreator[T], _create)


_CH: NodeCreator[hou.ChopNode] = _create_fn(hou.ChopNode, "_CH")
_CHNet: NodeCreator[hou.OpNode] = _create_fn(hou.OpNode, "_CHNet")
_C: NodeCreator[hou.CopNode] = _create_fn(hou.CopNode, "_C")
_C2: NodeCreator[hou.Cop2Node] = _create_fn(hou.Cop2Node, "_C2")
_Cnet: NodeCreator[hou.OpNode] = _create_fn(hou.OpNode, "_Cnet")
_Data: NodeCreator[hou.OpNode] = _create_fn(hou.OpNode, "_Data")
_Director: NodeCreator[hou.OpNode] = _create_fn(hou.OpNode, "_Director")
_D: NodeCreator[hou.DopNode] = _create_fn(hou.DopNode, "_D")
_L: NodeCreator[hou.LopNode] = _create_fn(hou.LopNode, "_L")
_M: NodeCreator[hou.OpNode] = _create_fn(hou.OpNode, "_M")
_R: NodeCreator[hou.OpNode] = _create_fn(hou.OpNode, "_R")
_O: NodeCreator[hou.ObjNode] = _create_fn(hou.ObjNode, "_O")
_S: NodeCreator[hou.SopNode] = _create_fn(hou.SopNode, "_S")
_SH: NodeCreator[hou.ShopNode] = _create_fn(hou.ShopNode, "_SH")
_T: NodeCreator[hou.TopNode] = _create_fn(hou.TopNode, "_T")
_Tnet: NodeCreator[hou.OpNode] = _create_fn(hou.OpNode, "_Tnet")
_V: NodeCreator[hou.VopNode] = _create_fn(hou.VopNode, "_V")
_Vnet: NodeCreator[hou.VopNetNode] = _create_fn(hou.VopNetNode, "_Vnet")

category_fns= {
    hou.nodeTypeCategories()['Chop']: _CH,
    hou.nodeTypeCategories()['ChopNet']: _CHNet,
    hou.nodeTypeCategories()['Cop']: _C,
    hou.nodeTypeCategories()['Cop2']: _C2,
    hou.nodeTypeCategories()['CopNet']: _Cnet,
    hou.nodeTypeCategories()['Data']: _Data,
    hou.nodeTypeCategories()['Director']: _Director,
    hou.nodeTypeCategories()['Dop']: _D,
    hou.nodeTypeCategories()['Lop']: _L,
    hou.nodeTypeCategories()['Manager']: _M,
    hou.nodeTypeCategories()['Object']: _O,
    hou.nodeTypeCategories()['Driver']: _R,
    hou.nodeTypeCategories()['Sop']: _S,
    hou.nodeTypeCategories()['Shop']: _SH,
    hou.nodeTypeCategories()['Top']: _T,
    hou.nodeTypeCategories()['TopNet']: _Tnet,
    hou.nodeTypeCategories()['Vop']: _V,
    hou.nodeTypeCategories()['VopNet']: _Vnet,
}

ROOT_NODE = hou.node('/')
parent_nodes: set[hou.Node] = set()
parent_nodes.add(ROOT_NODE)  # Add the root node as a parent
parent_nodes.update(ROOT_NODE.children())  # Add all immediate children of the root node

class Chain(Generic[T]):
    """
    A chain of Houdini nodes that can be created and connected together.
    """
    __nodes: tuple[T, ...]
    __by_name: dict[str, T]
    __inputs: tuple[hou.Node|None, ...]=()
    def __init__(self, *nodes: hou.Node):
        self.__nodes = tuple(n
                            for n in nodes
                            if n is not None)
        self.__by_name = {n.name(): n
                          for n in nodes
                          if n is not None
                          if n.name()}

    def __getitem__(self, name: str|int) -> NodeCreator[hou.Node]:
        if isinstance(name, int):
            if 0 <= name < len(self.__nodes):
                return self.__nodes[name]
            raise IndexError(f"Node index {name} out of range.")
        if isinstance(name, str):
            if name in self.__by_name:
                return self.__by_name[name]
            raise KeyError(f"Node with name '{name}' not found in chain.")
        raise TypeError(f"Invalid key type: {type(name)}. Expected str or int.")

    def inputs(self) -> tuple[hou.Node|None,...]:
        """
        Get the inputs of the first node in the chain.

        Returns:
            tuple[hou.Node|None, ...]: A tuple of input nodes for the first node.
        """
        if self.__nodes:
            return getattr(self.__nodes[0], '_inputs', ())
        return self.__inputs

    def input(self, index: int) -> hou.Node|None:
        """
        Get the input node at the specified index for the first node in the chain.

        Args:
            index (int): The index of the input node to retrieve.

        Returns:
            hou.Node|None: The input node at the specified index, or None if not connected.
        """
        inputs = self.inputs()
        if 0 <= index < len(inputs):
            return inputs[index]
        raise IndexError(f"Input index {index} out of range.")

    def setInput(self, index: int, node: hou.Node|None) -> None:
        """
        Set the input node at the specified index for the first node in the chain.

        Args:
            index (int): The index of the input node to set.
            node (hou.Node|None): The node to set as input, or None to disconnect.
        """
        if self.__nodes:
            self.__nodes[0].setInput(index, node)
        else:
            inputs = self.__inputs
            def select_input(i: int) -> hou.Node|None:
                if i < len(inputs):
                    return inputs[i]
                return None
            self.__inputs = tuple(
                node if i == index else select_input(i)
                for i in range(max(len(inputs), index))
            )

    def outputs(self) -> tuple[hou.Node|None, ...]:
        """
        Get the output nodes of the last node in the chain.
        Returns:
            tuple[hou.Node|None, ...]: A tuple of output nodes for the last node.
        """
        if self.__nodes:
            return self.__nodes[-1].outputs()
        return self.__inputs

    def __len__(self) -> int:
        """
        Get the number of nodes in the chain.

        Returns:
            int: The number of nodes in the chain.
        """
        return len(self.__nodes)

    def children(self) -> tuple[T, ...]:
        """
        Get the child nodes of the first node in the chain.

        Returns:
            tuple[T, ...]: A tuple of child nodes for the first node.
        """
        if self.__nodes:
            return self.__nodes[0].children()
        return ()

    def __repr__(self) -> str:
        return f"Chain({', '.join(n.name() for n in self.__nodes)})"

    def __str__(self) -> str:
        return ' | '.join(n.name() for n in self.__nodes)


def analyze_houdini_scene(file_path: str) -> dict[str, Any]:
    """
    Analyze a Houdini scene file and return structured data.

    Args:
        file_path (str): Path to the Houdini file to load.

    Returns:
        Dictionary containing analysis results and generated Python code
    """
    try:
        # Load the Houdini file
        hou.hipFile.load(file_path)

        # Analyze the contents
        nodes = hou.node('/').allSubChildren()
        real_nodes = [
            n
            for n in nodes
            if n.parent() is not None
            and n.parent() is not ROOT_NODE
        ]
        types = {n.type() for n in nodes}
        categories = {
            n.type().category()
            for n in real_nodes
        }
        parents_used = {
            n.parent()
            for n in real_nodes
        }
        links = [
            c
            for n in real_nodes
            for c in n.outputConnections()
        ]
        by_parent: dict[hou.Node, list[hou.Node]] = defaultdict(list)
        for n in real_nodes:
            by_parent[n.parent()].append(n)

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

        # Generate code
        code_lines = []
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
            fn = category_fns[c]
            assert fn is not None, f"Missing function for category {c.name()}"
            fn_name = fn.__name__ # type: ignore[return-value]
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
            fn = category_fns[n.type().category()]
            assert fn is not None, f"Missing function for category {n.type().category().name()}"
            fn_name = fn.__name__ # type: ignore[return-value]
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

        # Generate explicit connections (only for connections not covered by chains)
        code_lines.append("# Explicit connections (reverse/branch connections)")
        explicit_connections = []
        chain_connections = set()

        # Create a mapping from nodes to their chain variables
        node_to_chain_var = {}
        for chain in sorted_chains:
            if chain:
                chain_var = good_var(chain[0])
                if len(chain) == 1:
                    node_to_chain_var[chain[0]] = chain_var
                else:
                    for node in chain:
                        node_to_chain_var[node] = f"{good_var(node)}"  # Individual node vars for chains

        # Track connections that are implicit in chains
        for chain in chains:
            for i in range(len(chain) - 1):
                current = chain[i]
                next_node = chain[i + 1]
                # Check if next_node is actually connected to current
                if hasattr(next_node, 'numInputs') and hasattr(next_node, 'input'):
                    try:
                        for input_idx in range(next_node.numInputs()):
                            if next_node.input(input_idx) == current:
                                chain_connections.add((current, next_node, input_idx))
                    except AttributeError:
                        # Some node types don't support input operations
                        pass

        # Generate only non-chain connections
        for connection in links:
            src = connection.outputNode()
            dst = connection.inputNode()
            input_idx = connection.inputIndex()

            if (src, dst, input_idx) not in chain_connections:
                src_var = node_to_chain_var.get(src, good_var(src))
                dst_var = node_to_chain_var.get(dst, good_var(dst))
                explicit_connections.append(f"{dst_var}.setInput({input_idx}, {src_var})")

        for conn in explicit_connections:
            code_lines.append(conn)

        code_lines.append("")
        code_lines.append("# Scene recreation complete!")

        # Return structured data
        return {
            "success": True,
            "file_path": file_path,
            "generated_code": "\n".join(code_lines),
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
            "file_path": file_path
        }


@click.command()
@click.argument('file_path', type=click.Path(exists=True, dir_okay=False))
def load_houdini_file(file_path):
    """
    Load a Houdini file (HDA or HIP) and analyze its contents.
    Print the results to console (for command-line usage).

    Args:
        file_path (str): Path to the Houdini file to load.
    """
    result = analyze_houdini_scene(file_path)

    if result["success"]:
        print(f"Loaded Houdini file: {file_path}")
        stats = result["statistics"]
        print(f"Total nodes in the scene: {stats['total_nodes']}")
        print(f"Detected {stats['chains']} linear chains")
        print(f"Found {stats['node_references']} nodes with parameter references")
        print("Types in the scene: " + ', '.join(stats['node_types']))
        print("Categories in the scene: " + ', '.join(stats['categories']))
        print()
        print("```python")
        print(result["generated_code"])
        print("```")
        print()
        print("Scene recreation complete!")
    else:
        print(f"Error loading file: {result['error']}")
        sys.exit(1)

def extract_non_default_parms(node: hou.Node) -> dict[str, Any]:
    """
    Extract parameters that differ from their default values.

    Args:
        node: The Houdini node to analyze

    Returns:
        Dictionary of parameter names to their current values
    """
    non_default_parms = {}

    try:
        parm_template_group = node.parmTemplateGroup()
        for parm_template in parm_template_group.parmTemplates():
            if isinstance(parm_template, hou.FolderParmTemplate):
                # Recursively handle folder parameters
                for sub_template in parm_template.parmTemplates():
                    _extract_parm_value(node, sub_template, non_default_parms)
            else:
                _extract_parm_value(node, parm_template, non_default_parms)
    except Exception as e:
        # Could optionally log this to a StringIO buffer if needed
        pass

    return non_default_parms

def _extract_parm_value(node: hou.Node, parm_template: hou.ParmTemplate, result: dict[str, Any]) -> None:
    """Helper function to extract individual parameter values."""
    parm_name = parm_template.name()

    try:
        parm = node.parm(parm_name)
        if parm is None:
            return

        # Skip parameters that are at their default values
        if hasattr(parm_template, 'defaultValue'):
            default_val = parm_template.defaultValue()
            current_val = parm.eval()

            # For different parameter types, handle comparison appropriately
            if isinstance(parm_template, hou.StringParmTemplate):
                if current_val != (default_val[0] if isinstance(default_val, tuple) else default_val):
                    result[parm_name] = current_val
            elif isinstance(parm_template, (hou.IntParmTemplate, hou.FloatParmTemplate)):
                if isinstance(default_val, (list, tuple)) and len(default_val) > 0:
                    default_val = default_val[0]
                if abs(current_val - default_val) > 1e-6:  # Float tolerance
                    result[parm_name] = current_val
            elif isinstance(parm_template, hou.ToggleParmTemplate):
                if current_val != (default_val[0] if isinstance(default_val, tuple) else default_val):
                    result[parm_name] = current_val
    except Exception:
        # If we can't evaluate the parameter, skip it
        pass

def find_node_references_in_parms(node: hou.Node, all_nodes: set[hou.Node]) -> dict[str, list[hou.Node]]:
    """
    Find parameters that reference other nodes in the scene.

    Args:
        node: The node to analyze
        all_nodes: Set of all nodes in the scene for reference lookup

    Returns:
        Dictionary mapping parameter names to lists of referenced nodes
    """
    node_refs = {}
    node_paths = {n.path(): n for n in all_nodes}

    try:
        for parm in node.parms():
            if parm.parmTemplate().type() == hou.parmTemplateType.String:
                parm_val = parm.evalAsString()

                # Look for node path patterns
                referenced_nodes = []

                # Direct node path references (e.g., "/obj/geo1")
                if parm_val.startswith('/') and parm_val in node_paths:
                    referenced_nodes.append(node_paths[parm_val])

                # Relative path references (e.g., "../geo1", "./subnet1/node1")
                elif parm_val.startswith(('./', '../')):
                    try:
                        resolved_node = node.node(parm_val)
                        if resolved_node and resolved_node in all_nodes:
                            referenced_nodes.append(resolved_node)
                    except:
                        pass

                # Channel references (e.g., "ch('../tx')")
                if 'ch(' in parm_val:
                    ch_pattern = r'ch\(["\']([^"\']+)["\']\)'
                    for match in re.finditer(ch_pattern, parm_val):
                        ref_path = match.group(1)
                        try:
                            if ref_path.startswith('/'):
                                # Absolute path
                                ref_node_path = '/'.join(ref_path.split('/')[:-1])
                                if ref_node_path in node_paths:
                                    referenced_nodes.append(node_paths[ref_node_path])
                            else:
                                # Relative path
                                ref_node = node.node('/'.join(ref_path.split('/')[:-1]))
                                if ref_node and ref_node in all_nodes:
                                    referenced_nodes.append(ref_node)
                        except:
                            pass

                if referenced_nodes:
                    node_refs[parm.name()] = referenced_nodes

    except Exception as e:
        # Capture error in StringIO instead of printing
        error_msg = f"Warning: Could not analyze node references for {node.path()}: {e}"
        # Could optionally log this to a StringIO buffer if needed

    return node_refs

def detect_linear_chains(nodes: list[hou.Node]) -> list[list[hou.Node]]:
    """
    Detect linear chains of nodes (basic blocks) where each node has exactly one input
    and one output, forming a simple pipeline.

    Args:
        nodes: List of nodes to analyze

    Returns:
        List of chains, where each chain is a list of connected nodes
    """
    # Build adjacency information
    node_inputs = {}
    node_outputs = defaultdict(list)

    for node in nodes:
        inputs = []
        # Check if node has inputs (not all node types do, e.g. OpNodes)
        if hasattr(node, 'numInputs') and hasattr(node, 'input'):
            try:
                for i in range(node.numInputs()):
                    input_node = node.input(i)
                    if input_node:
                        inputs.append(input_node)
                        node_outputs[input_node].append(node)
            except AttributeError:
                # Some node types don't support input operations
                pass
        node_inputs[node] = inputs

    # Find chain starts (nodes with no inputs or multiple inputs)
    # and chain ends (nodes with no outputs or multiple outputs)
    chain_starts = []
    processed = set()

    for node in nodes:
        if node in processed:
            continue

        # Start a new chain if:
        # - Node has no inputs (source node)
        # - Node has multiple inputs (merge point)
        # - Node's input has multiple outputs (branch point)
        num_inputs = len(node_inputs[node])
        is_chain_start = (
            num_inputs == 0 or  # Source node
            num_inputs > 1 or   # Merge point
            any(len(node_outputs[inp]) > 1 for inp in node_inputs[node])  # Input branches
        )

        if is_chain_start or node not in processed:
            # Follow the linear chain forward
            chain = []
            current = node

            while current and current not in processed:
                chain.append(current)
                processed.add(current)

                # Continue chain if current node has exactly one output
                # and that output has exactly one input
                outputs = node_outputs[current]
                if (len(outputs) == 1 and
                    len(node_inputs[outputs[0]]) == 1):
                    current = outputs[0]
                else:
                    break

            if chain:
                chain_starts.append(chain)

    return chain_starts

def topological_sort_with_chains(nodes: list[hou.Node], chains: list[list[hou.Node]]) -> list[list[hou.Node]]:
    """
    Perform topological sort considering chains as single units.

    Args:
        nodes: All nodes in the scene
        chains: List of detected chains

    Returns:
        Topologically sorted list of chains
    """
    # Create mapping from node to chain
    node_to_chain = {}
    chain_nodes = set()
    for i, chain in enumerate(chains):
        for node in chain:
            node_to_chain[node] = i
            chain_nodes.add(node)

    # Add single-node chains for nodes not in any chain
    single_chains = []
    for node in nodes:
        if node not in chain_nodes:
            single_chains.append([node])
            node_to_chain[node] = len(chains) + len(single_chains) - 1

    all_chains = chains + single_chains

    # Build chain dependency graph
    chain_deps = defaultdict(set)
    chain_in_degree = defaultdict(int)

    for i in range(len(all_chains)):
        chain_in_degree[i] = 0

    for chain_idx, chain in enumerate(all_chains):
        # Find dependencies for this chain
        deps = set()
        for node in chain:
            # Check if node supports inputs
            if hasattr(node, 'numInputs') and hasattr(node, 'input'):
                try:
                    for input_idx in range(node.numInputs()):
                        input_node = node.input(input_idx)
                        if input_node and input_node in node_to_chain:
                            input_chain_idx = node_to_chain[input_node]
                            if input_chain_idx != chain_idx:  # Don't depend on self
                                deps.add(input_chain_idx)
                except AttributeError:
                    # Some node types don't support input operations
                    pass

        for dep_idx in deps:
            if dep_idx not in chain_deps[chain_idx]:
                chain_deps[chain_idx].add(dep_idx)
                chain_in_degree[chain_idx] += 1

    # Kahn's algorithm for topological sort
    queue = deque([i for i in range(len(all_chains)) if chain_in_degree[i] == 0])
    sorted_chains = []

    while queue:
        current_chain_idx = queue.popleft()
        sorted_chains.append(all_chains[current_chain_idx])

        # Update in-degrees of dependent chains
        for chain_idx in range(len(all_chains)):
            if current_chain_idx in chain_deps[chain_idx]:
                chain_deps[chain_idx].remove(current_chain_idx)
                chain_in_degree[chain_idx] -= 1
                if chain_in_degree[chain_idx] == 0:
                    queue.append(chain_idx)

    return sorted_chains

def format_parm_value(value: Any) -> str:
    """Format a parameter value for Python code generation."""
    if isinstance(value, str):
        return repr(value)
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, bool):
        return str(value)
    else:
        return repr(str(value))

def generate_chain_code(chain: list[hou.Node], all_node_refs: dict[hou.Node, dict[str, list[hou.Node]]]) -> str:
    """Generate Python code for a chain of nodes."""
    if not chain:
        return ""

    lines = []
    chain_var = good_var(chain[0])

    if len(chain) == 1:
        # Single node
        node = chain[0]
        fn = category_fns[node.type().category()]
        fn_name = fn.__name__ # type: ignore[return-value]

        # Extract parameters
        parms = extract_non_default_parms(node)
        parm_strs = []
        for parm_name, parm_value in parms.items():
            parm_strs.append(f"{parm_name}={format_parm_value(parm_value)}")

        parm_str = ", ".join(parm_strs)
        if parm_str:
            parm_str = ", " + parm_str

        lines.append(f"{chain_var} = {fn_name}('{node.type().name()}', name='{node.name()}'{parm_str})")
    else:
        # Chain of multiple nodes - use Chain class
        node_exprs = []
        for node in chain:
            fn = category_fns[node.type().category()]
            fn_name = fn.__name__ # type: ignore[return-value]

            # Extract parameters
            parms = extract_non_default_parms(node)
            parm_strs = []
            for parm_name, parm_value in parms.items():
                parm_strs.append(f"{parm_name}={format_parm_value(parm_value)}")

            parm_str = ", ".join(parm_strs)
            if parm_str:
                parm_str = ", " + parm_str

            node_exprs.append(f"{fn_name}('{node.type().name()}', name='{node.name()}'{parm_str})")

        lines.append(f"{chain_var} = Chain({', '.join(node_exprs)})")

    # Add node reference updates
    for node in chain:
        if node in all_node_refs:
            if len(chain) == 1:
                node_var = chain_var
            else:
                # For chains, we need to reference individual nodes
                node_var_name = good_var(node)
                lines.append(f"{node_var_name} = {chain_var}['{node.name()}']")
                node_var = node_var_name

            for parm_name, ref_nodes in all_node_refs[node].items():
                for ref_node in ref_nodes:
                    ref_var = good_var(ref_node)
                    lines.append(f"{node_var}.parm('{parm_name}').set('{ref_var}')")

    return "\n".join(lines)


if __name__ == '__main__':
    load_houdini_file()

