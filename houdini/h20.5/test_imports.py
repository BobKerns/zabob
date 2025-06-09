#!/usr/bin/env hython
"""Test script to verify all imports work correctly."""

import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test all module imports."""
    try:
        print("Testing imports...")

        # Test main module
        from zabob.h20_5.node_loader import main
        print("✓ Main imports successful")

        # Test parameter extraction
        from zabob.h20_5.parameter_extraction import extract_non_default_parms
        print("✓ Parameter extraction imports successful")

        # Test node references
        from zabob.h20_5.node_references import find_node_references_in_parms
        print("✓ Node references imports successful")

        # Test chain detection
        from zabob.h20_5.chain_detection import detect_linear_chains
        print("✓ Chain detection imports successful")

        # Test topological sort
        from zabob.h20_5.topological_sort import topological_sort_with_chains
        print("✓ Topological sort imports successful")

        # Test code generation
        from zabob.h20_5.code_generation import generate_network_code
        print("✓ Code generation imports successful")

        print("🎉 All imports working correctly!")
        return True

    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
