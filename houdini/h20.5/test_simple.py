#!/usr/bin/env python3
"""
Simple test to check if our node_load modular system can be imported.
"""

try:
    print("Testing imports...")

    # Test basic Houdini import
    import hou
    print("✓ hou module imported successfully")

    # Test our modular imports
    from zabob.h20_5.parameter_extraction import extract_non_default_parms
    print("✓ parameter_extraction module imported")

    from zabob.h20_5.node_references import find_node_references_in_parms
    print("✓ node_references module imported")

    from zabob.h20_5.chain_detection import detect_linear_chains
    print("✓ chain_detection module imported")

    from zabob.h20_5.topological_sort import topological_sort_with_chains
    print("✓ topological_sort module imported")

    from zabob.h20_5.code_generation import generate_network_code
    print("✓ code_generation module imported")

    from zabob.h20_5.node_load_modular import load_houdini_file, main
    print("✓ node_load_modular main functions imported")

    print("\n✓ All imports successful!")
    print("Testing with a simple file check...")

    import sys
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print(f"Attempting to load: {file_path}")
        result = load_houdini_file(file_path)
        if result:
            print("✓ Load successful!")
        else:
            print("✗ Load failed")
    else:
        print("No file provided - import test only")

except ImportError as e:
    print(f"✗ Import error: {e}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
