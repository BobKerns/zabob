#!/usr/bin/env python3
"""
Test script for the enhanced node_load.py functionality
"""

import sys
import os
sys.path.insert(0, 'src')

def test_without_houdini():
    """Test functions that don't require Houdini"""
    try:
        # Test basic imports (excluding hou-dependent parts)
        print("Testing basic utilities...")

        # Import specific functions to test
        from zabob.h20_5.node_loader import good_var, RE_BAD_CHARS, format_parm_value

        # Test good_var function
        class MockNode:
            def __init__(self, name):
                self._name = name
            def name(self):
                return self._name
            def type(self):
                class MockType:
                    def name(self):
                        return "geo"
                return MockType()

        node1 = MockNode("test_node")
        node2 = MockNode("another-node")
        node3 = MockNode("123invalid")

        var1 = good_var(node1)
        var2 = good_var(node2)
        var3 = good_var(node3)

        print(f"good_var('test_node') = '{var1}'")
        print(f"good_var('another-node') = '{var2}'")
        print(f"good_var('123invalid') = '{var3}'")

        # Test format_parm_value
        print(f"format_parm_value('string') = {format_parm_value('string')}")
        print(f"format_parm_value(42) = {format_parm_value(42)}")
        print(f"format_parm_value(3.14) = {format_parm_value(3.14)}")
        print(f"format_parm_value(True) = {format_parm_value(True)}")

        print("✅ Basic utilities test passed!")
        return True

    except Exception as e:
        print(f"❌ Error in basic utilities test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing enhanced node_load.py...")
    test_without_houdini()
