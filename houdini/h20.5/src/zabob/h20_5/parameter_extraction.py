"""
Parameter extraction utilities for Houdini node analysis.
Extracts non-default parameters and their values from Houdini nodes.
"""

import hou


def extract_non_default_parms(node):
    """
    Extract parameters that differ from default values.

    Args:
        node: Houdini node to analyze

    Returns:
        dict: Parameter name -> value mapping for non-default parameters
    """
    non_default_parms = {}

    try:
        for parm_tuple in node.parmTuples():
            try:
                parm = parm_tuple[0] if len(parm_tuple) > 0 else None
                if parm is None:
                    continue

                # Skip parameters that are at default values
                if parm.isAtDefault():
                    continue

                # Extract the parameter value
                parm_name = parm.name()
                parm_value = _extract_parm_value(parm_tuple)

                if parm_value is not None:
                    non_default_parms[parm_name] = parm_value

            except Exception as e:
                # Skip parameters that can't be read
                print(f"Warning: Could not read parameter {parm_tuple}: {e}")
                continue

    except Exception as e:
        print(f"Warning: Could not read parameters for node {node.name()}: {e}")

    return non_default_parms


def _extract_parm_value(parm_tuple):
    """
    Extract value from a parameter tuple, handling different parameter types.

    Args:
        parm_tuple: Houdini parameter tuple

    Returns:
        Parameter value in appropriate Python type
    """
    try:
        if len(parm_tuple) == 1:
            # Single parameter
            parm = parm_tuple[0]
            return _get_single_parm_value(parm)
        else:
            # Multi-component parameter (vector, etc.)
            values = []
            for parm in parm_tuple:
                value = _get_single_parm_value(parm)
                values.append(value)
            return tuple(values)

    except Exception as e:
        print(f"Warning: Could not extract value from parameter tuple: {e}")
        return None


def _get_single_parm_value(parm):
    """
    Get value from a single parameter, handling different data types.

    Args:
        parm: Single Houdini parameter

    Returns:
        Parameter value in appropriate Python type
    """
    try:
        parm_type = parm.parmTemplate().type()

        if parm_type == hou.parmTemplateType.String:
            return parm.eval()
        elif parm_type == hou.parmTemplateType.Int:
            return parm.eval()
        elif parm_type == hou.parmTemplateType.Float:
            return parm.eval()
        elif parm_type == hou.parmTemplateType.Toggle:
            return bool(parm.eval())
        elif parm_type == hou.parmTemplateType.Menu:
            return parm.eval()
        else:
            # For other types, try to get the raw value
            return parm.eval()

    except Exception as e:
        print(f"Warning: Could not get value for parameter {parm.name()}: {e}")
        return None


def format_parm_value(value):
    """
    Format a parameter value for inclusion in Python code.

    Args:
        value: Parameter value to format

    Returns:
        str: Formatted value ready for Python code
    """
    if value is None:
        return "None"
    elif isinstance(value, str):
        # Escape quotes and special characters
        escaped = value.replace('\\', '\\\\').replace('"', '\\"')
        return f'"{escaped}"'
    elif isinstance(value, bool):
        return "True" if value else "False"
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, (list, tuple)):
        # Format as tuple for multi-component parameters
        formatted_items = [format_parm_value(item) for item in value]
        return f"({', '.join(formatted_items)})"
    else:
        # Try to convert to string and format as string
        return f'"{str(value)}"'
