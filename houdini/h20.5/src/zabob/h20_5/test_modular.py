#!/usr/bin/env hython
"""
Test script for the modular enhanced node_load system.
Tests individual modules and the main integration.
"""

import sys
import os

# Add the current directory to the path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_imports():
    """Test that all modules can be imported without errors."""
    print("Testing module imports...")

    try:
        from parameter_extraction import extract_non_default_parms, format_parm_value
        print("✓ parameter_extraction module imported successfully")
    except Exception as e:
        print(f"✗ parameter_extraction import failed: {e}")
        return False

    try:
        from node_references import find_node_references_in_parms, resolve_node_references
        print("✓ node_references module imported successfully")
    except Exception as e:
        print(f"✗ node_references import failed: {e}")
        return False

    try:
        from chain_detection import detect_linear_chains, find_chain_connections
        print("✓ chain_detection module imported successfully")
    except Exception as e:
        print(f"✗ chain_detection import failed: {e}")
        return False

    try:
        from topological_sort import topological_sort_with_chains, analyze_dependencies
        print("✓ topological_sort module imported successfully")
    except Exception as e:
        print(f"✗ topological_sort import failed: {e}")
        return False

    try:
        from code_generation import generate_network_code, generate_summary_comment
        print("✓ code_generation module imported successfully")
    except Exception as e:
        print(f"✗ code_generation import failed: {e}")
        return False

    try:
        from node_load_modular import load_houdini_file, analyze_specific_context
        print("✓ node_load_modular main module imported successfully")
    except Exception as e:
        print(f"✗ node_load_modular import failed: {e}")
        return False

    try:
        import node_load
        print("✓ main node_load module imported successfully")
    except Exception as e:
        print(f"✗ main node_load import failed: {e}")
        return False

    return True

def test_basic_functionality():
    """Test basic functionality with mock objects."""
    print("\nTesting basic functionality...")

    try:
        import hou
        print("✓ hou module available")

        # Test parameter formatting
        from parameter_extraction import format_parm_value
        test_values = [
            (None, "None"),
            ("test", '"test"'),
            (True, "True"),
            (False, "False"),
            (42, "42"),
            (3.14, "3.14"),
            ([1, 2, 3], "(1, 2, 3)")
        ]

        for value, expected in test_values:
            result = format_parm_value(value)
            if result == expected:
                print(f"✓ format_parm_value({value}) = {result}")
            else:
                print(f"✗ format_parm_value({value}) = {result}, expected {expected}")
                return False

        print("✓ Parameter formatting tests passed")

    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

    return True

def test_current_scene_analysis():
    """Test analysis of the current scene."""
    print("\nTesting current scene analysis...")

    try:
        from node_load_modular import analyze_specific_context

        # Analyze the current scene's /obj context
        results = analyze_specific_context("/obj")

        if results:
            print(f"✓ Current scene analysis successful")
            print(f"  - Found {results['total_nodes']} nodes")
            print(f"  - Detected {len(results['chains'])} chains")
            print(f"  - Found {results['connections']} connections")
            return True
        else:
            print("✗ Current scene analysis returned None")
            return False

    except Exception as e:
        print(f"✗ Current scene analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("MODULAR NODE_LOAD SYSTEM TEST")
    print("=" * 60)

    all_passed = True

    # Test imports
    if not test_imports():
        all_passed = False

    # Test basic functionality
    if not test_basic_functionality():
        all_passed = False

    # Test current scene analysis
    if not test_current_scene_analysis():
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED! ✓")
        print("The modular node_load system is working correctly.")
    else:
        print("SOME TESTS FAILED! ✗")
        print("Check the error messages above for details.")
    print("=" * 60)

if __name__ == "__main__":
    main()
