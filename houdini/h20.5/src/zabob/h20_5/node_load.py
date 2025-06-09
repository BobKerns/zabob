#!/usr/bin/env hython
'''
Load Houdini HDA or HIP files and analyze their contents.
'''

from collections import defaultdict, deque
from collections.abc import Sequence
from contextvars import ContextVar
from contextlib import contextmanager
from itertools import count
from typing import Generic, LiteralString, Protocol, TypeVar, cast, Any
import re
import json

import click
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


@click.command()
@click.argument('file_path', type=click.Path(exists=True, dir_okay=False))
def load_houdini_file(file_path):
    """
    Load a Houdini file (HDA or HIP) and analyze its contents.

    Args:
        file_path (str): Path to the Houdini file to load.
    """
    try:
        # Load the Houdini file
        hou.hipFile.load(file_path)
        print(f"Loaded Houdini file: {file_path}")

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

        vars: set[str] = set()
        print(f"Total nodes in the scene: {len(nodes)}")
        print("Types in the scene:" + ', '.join([t.name() for t in types]))
        print("Categories in the scene:" + ', '.join([c.name() for c in categories]))
        print("```python")
        for c in categories:
            fn = category_fns[c]
            assert fn is not None, f"Missing function for category {c.name()}"
            fn_name = fn.__name__ # type: ignore[return-value]
            print(f"def {fn_name}(node_type: str, name: str='', **kwargs):")
            print(f"    \"\"\"Create or check a {c.name()} node.\"\"\"")
            print(f"    return _parent.get().createNode(node_type, name)")
        print()
        for n in parents_used:
            fn = category_fns[n.type().category()]
            assert fn is not None, f"Missing function for category {n.type().category().name()}"
            fn_name = fn.__name__ # type: ignore[return-value]
            print(f"{good_var(n)} = {fn_name}(hou.node(f'/{n.name()}')")
        print()
        for p, children in by_parent.items():
            if p != ROOT_NODE:
                print(f'with parent({good_var(p)}):')
                for c in children:
                    fn = category_fns[c.type().category()]
                    assert fn is not None, f"Missing function for category {c.type().category().name()}"
                    fn_name = fn.__name__ # type: ignore[return-value]
                    print(f"    {good_var(c)} = {fn_name}('{c.type().name()}', name='{c.name()}')")
        print("# Connect the nodes")
        for c in links:
            src = good_var(c.inputNode())
            dst = good_var(c.outputNode())
            print(f"{src}.setInput({c.inputIndex()}, {dst})")
        print("```")

    except Exception as e:
        print(f"Error loading file: {e}")
        raise e

if __name__ == '__main__':
    load_houdini_file()

