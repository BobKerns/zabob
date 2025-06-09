#!/usr/bin/env hython
"""
Enhanced Houdini scene analysis and code generation tool.
Analyzes .hipnc files and generates Python code to recreate the scene.

This version uses a modular architecture with separate files for each major function
to minimize corruption issues and improve maintainability.
"""

# Import the main functionality from the modular implementation
from zabob.h20_5.node_load_modular import load_houdini_file, analyze_specific_context, main

# Re-export the main functions for backward compatibility
__all__ = ['load_houdini_file', 'analyze_specific_context', 'main']

if __name__ == "__main__":
    main()

