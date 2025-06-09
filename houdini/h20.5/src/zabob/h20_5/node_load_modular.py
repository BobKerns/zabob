#!/usr/bin/env hython
"""
Enhanced Houdini scene analysis and code generation tool.
Analyzes .hipnc files and generates Python code to recreate the scene.

This is the main entry point that orchestrates the analysis pipeline using
specialized modules for different aspects of the analysis.
"""

import sys
import os
import hou

# Import our specialized modules
from zabob.h20_5.parameter_extraction import extract_non_default_parms
from zabob.h20_5.node_references import find_node_references_in_parms, resolve_node_references
from zabob.h20_5.chain_detection import detect_linear_chains, find_chain_connections, get_chain_info
from zabob.h20_5.topological_sort import topological_sort_with_chains, analyze_dependencies, get_creation_order
from zabob.h20_5.code_generation import generate_network_code, generate_summary_comment, format_code_with_style


def load_houdini_file(file_path, output_file=None):
    """
    Main function to load and analyze a Houdini scene file.

    Args:
        file_path: Path to the .hipnc file to analyze
        output_file: Optional output file for generated code

    Returns:
        dict: Analysis results and generated code
    """
    print(f"Loading Houdini file: {file_path}")

    try:
        # Load the scene file
        hou.hipFile.load(file_path)
        print("Scene loaded successfully")

        # Get all nodes in the scene (focusing on /obj level for now)
        obj_context = hou.node("/obj")
        if not obj_context:
            print("Error: Could not access /obj context")
            return None

        all_nodes = obj_context.children()
        if not all_nodes:
            print("No nodes found in /obj context")
            return {"nodes": [], "code": "# No nodes found"}

        print(f"Found {len(all_nodes)} nodes")

        # Step 1: Extract non-default parameters for all nodes
        print("Extracting non-default parameters...")
        all_non_default_parms = {}
        for node in all_nodes:
            parms = extract_non_default_parms(node)
            if parms:
                all_non_default_parms[node] = parms
                print(f"  {node.name()}: {len(parms)} non-default parameters")

        # Step 2: Find node references in parameters
        print("Finding node references in parameters...")
        all_node_references = {}
        for node in all_nodes:
            refs = find_node_references_in_parms(node)
            if refs:
                all_node_references[node] = refs
                print(f"  {node.name()}: {len(refs)} parameter references")

        # Step 3: Detect linear chains
        print("Detecting linear chains...")
        chains = detect_linear_chains(all_nodes)
        print(f"Found {len(chains)} chains:")
        for i, chain in enumerate(chains):
            chain_info = get_chain_info(chain)
            print(f"  Chain {i}: {chain_info['length']} nodes ({chain_info['first_node']} -> {chain_info['last_node']})")

        # Step 4: Find connections between chains
        print("Analyzing chain connections...")
        connections = find_chain_connections(chains, all_nodes)
        print(f"Found {len(connections)} inter-chain connections")

        # Step 5: Perform dependency analysis
        print("Analyzing dependencies...")
        dependency_analysis = analyze_dependencies(chains, all_nodes)

        # Step 6: Get creation order using topological sort
        print("Computing creation order...")
        creation_order_info = get_creation_order(chains, connections)
        creation_order = creation_order_info['creation_order']

        # Step 7: Generate Python code
        print("Generating Python code...")
        summary_comment = generate_summary_comment(chains, all_nodes, dependency_analysis)

        network_code = generate_network_code(
            chains,
            connections,
            creation_order,
            all_non_default_parms
        )

        full_code = summary_comment + "\n\n" + network_code
        formatted_code = format_code_with_style(full_code)

        # Step 8: Save or return results
        if output_file:
            with open(output_file, 'w') as f:
                f.write(formatted_code)
            print(f"Generated code saved to: {output_file}")

        # Prepare results
        results = {
            'file_path': file_path,
            'total_nodes': len(all_nodes),
            'chains': [get_chain_info(chain) for chain in chains],
            'connections': len(connections),
            'non_default_parms': {node.name(): parms for node, parms in all_non_default_parms.items()},
            'node_references': {node.name(): refs for node, refs in all_node_references.items()},
            'dependency_analysis': dependency_analysis,
            'creation_order': creation_order_info,
            'generated_code': formatted_code
        }

        print("Analysis complete!")
        return results

    except Exception as e:
        print(f"Error loading Houdini file: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_specific_context(context_path="/obj"):
    """
    Analyze a specific context in the current scene.

    Args:
        context_path: Path to the context to analyze (default: /obj)

    Returns:
        dict: Analysis results for the context
    """
    print(f"Analyzing context: {context_path}")

    try:
        context = hou.node(context_path)
        if not context:
            print(f"Error: Could not access context {context_path}")
            return None

        nodes = context.children()
        if not nodes:
            print(f"No nodes found in context {context_path}")
            return {"nodes": [], "code": "# No nodes found"}

        print(f"Found {len(nodes)} nodes in {context_path}")

        # Run the same analysis pipeline as load_houdini_file
        # but on the current scene context

        # Extract parameters
        all_non_default_parms = {}
        for node in nodes:
            parms = extract_non_default_parms(node)
            if parms:
                all_non_default_parms[node] = parms

        # Find references
        all_node_references = {}
        for node in nodes:
            refs = find_node_references_in_parms(node)
            if refs:
                all_node_references[node] = refs

        # Detect chains
        chains = detect_linear_chains(nodes)
        connections = find_chain_connections(chains, nodes)

        # Analyze and sort
        dependency_analysis = analyze_dependencies(chains, nodes)
        creation_order_info = get_creation_order(chains, connections)

        # Generate code
        summary_comment = generate_summary_comment(chains, nodes, dependency_analysis)
        network_code = generate_network_code(
            chains,
            connections,
            creation_order_info['creation_order'],
            all_non_default_parms
        )

        full_code = summary_comment + "\n\n" + network_code
        formatted_code = format_code_with_style(full_code)

        results = {
            'context_path': context_path,
            'total_nodes': len(nodes),
            'chains': [get_chain_info(chain) for chain in chains],
            'connections': len(connections),
            'generated_code': formatted_code
        }

        print(f"Analysis of {context_path} complete!")
        return results

    except Exception as e:
        print(f"Error analyzing context {context_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """
    Command-line interface for the tool.
    """
    if len(sys.argv) < 2:
        print("Usage: hython node_load.py <scene_file.hipnc> [output_file.py]")
        print("   or: hython node_load.py --analyze-current [context_path]")
        sys.exit(1)

    if sys.argv[1] == "--analyze-current":
        # Analyze current scene
        context_path = sys.argv[2] if len(sys.argv) > 2 else "/obj"
        results = analyze_specific_context(context_path)

        if results:
            print("\n" + "="*60)
            print("GENERATED CODE:")
            print("="*60)
            print(results['generated_code'])
    else:
        # Load and analyze a scene file
        scene_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None

        if not os.path.exists(scene_file):
            print(f"Error: Scene file not found: {scene_file}")
            sys.exit(1)

        results = load_houdini_file(scene_file, output_file)

        if results:
            print("\n" + "="*60)
            print("ANALYSIS SUMMARY:")
            print("="*60)
            print(f"Total nodes: {results['total_nodes']}")
            print(f"Chains detected: {len(results['chains'])}")
            print(f"Inter-chain connections: {results['connections']}")

            if not output_file:
                print("\n" + "="*60)
                print("GENERATED CODE:")
                print("="*60)
                print(results['generated_code'])


if __name__ == "__main__":
    main()
