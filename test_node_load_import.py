#!/usr/bin/env hython
"""Test script to check if node_load.py can be imported and analyze_houdini_scene is available."""

try:
    import sys
    sys.path.insert(0, '/Users/rwk/p/zabob/houdini/h20.5/src')

    print("Importing zabob.h20_5.node_load...")
    import zabob.h20_5.node_load as node_load
    print("✅ Module imported successfully!")

    print("Checking for analyze_houdini_scene function...")
    if hasattr(node_load, 'analyze_houdini_scene'):
        print("✅ analyze_houdini_scene function found!")
        print(f"Function: {node_load.analyze_houdini_scene}")
    else:
        print("❌ analyze_houdini_scene function NOT found!")
        print("Available functions in module:")
        for attr in dir(node_load):
            if callable(getattr(node_load, attr)) and not attr.startswith('_'):
                print(f"  - {attr}")

    print("✅ Import test completed successfully!")

except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
