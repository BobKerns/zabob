#!/usr/bin/env hython
import sys
sys.path.insert(0, '/Users/rwk/p/zabob/houdini/h20.5/src')

print("Testing direct import...")
try:
    from zabob.h20_5 import node_load
    print("✅ Module import successful")
    print(f"Module location: {node_load.__file__}")
    print(f"Available functions: {[name for name in dir(node_load) if not name.startswith('_')]}")

    if hasattr(node_load, 'analyze_houdini_scene'):
        print("✅ analyze_houdini_scene found")
    else:
        print("❌ analyze_houdini_scene NOT found")

except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
