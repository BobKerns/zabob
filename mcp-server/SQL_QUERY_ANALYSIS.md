# SQL Query Analysis and MCP Server Fixes

## Overview
This document analyzes the SQL queries used by the Zabob MCP server, their exposure through MCP tools, and the fixes implemented to resolve function search issues.

## Issues Identified

### 1. SQL Query Case Mismatch
**Problem**: Database queries were looking for `type = 'function'` (lowercase) but actual values in database are `'FUNCTION'` (uppercase).

**Impact**: All function search tools returned 0 results despite database containing 48,693 functions.

**Affected Queries**:
- `get_functions_returning_nodes()`
- `search_functions_by_keyword()`
- `get_primitive_related_functions()`
- `get_modules_summary()` (function count)
- `get_database_stats()` (function/class counts)

### 2. Module Aliasing Issue
**Problem**: Functions like `nodeTypeCategories` exist in `_hou` module but users expect them in `hou` module (Python C extension pattern).

**Impact**: Users searching for `hou.nodeTypeCategories()` couldn't find it because it's stored as `_hou.nodeTypeCategories`.

### 3. Tool Description Gaps
**Problem**: MCP tool descriptions didn't clearly indicate they search the `houdini_module_data` table or explain their capabilities.

**Impact**: Reduced discoverability and understanding of tool capabilities.

## Fixes Implemented

### 1. Fixed SQL Query Case Sensitivity

**Files Modified**: `/Users/rwk/p/zabob/mcp-server/src/zabob/mcp/database.py`

**Changes Made**:
```sql
-- BEFORE (incorrect):
WHERE (type = 'function' OR type = '"function"')

-- AFTER (correct):
WHERE type = 'FUNCTION'
```

**Functions Fixed**:
- `get_functions_returning_nodes()`
- `search_functions_by_keyword()`
- `get_primitive_related_functions()`
- `get_modules_summary()`
- `get_database_stats()`

### 2. Implemented Module Aliasing

**New Logic**: Strip leading underscore from module names for display:
```python
# Handle module aliasing: if parent_name starts with '_', also expose without underscore
display_module = row['parent_name']
if display_module.startswith('_') and len(display_module) > 1:
    display_module = display_module[1:]  # _hou -> hou
```

**Result**: `_hou.nodeTypeCategories` now appears as `hou.nodeTypeCategories` in search results.

### 3. Added New MCP Tool

**New Tool**: `search_functions_by_module(module_name, limit)`
- Searches for functions within a specific module
- Handles module aliasing automatically (searches both 'hou' and '_hou')
- Returns both display module and original module names

### 4. Enhanced Tool Descriptions

**Updated Descriptions**:
- `get_functions_returning_nodes`: Added "Searches the houdini_module_data table"
- `search_functions`: Added "Searches across all Houdini modules in the database"
- `get_primitive_functions`: Added "Searches the houdini_module_data table"
- `enhanced_search_functions`: Added "Searches houdini_module_data table and enhances with web search"
- `get_modules_summary`: Added "Uses houdini_modules and houdini_module_data tables"

## Database Schema Context

### Key Tables
1. **`houdini_module_data`**: Contains functions, classes, methods, etc.
   - `type` field uses uppercase values: 'FUNCTION', 'CLASS', 'METHOD', etc.
   - `parent_name` field contains module names like '_hou', 'hou', 'toolutils'

2. **`houdini_modules`**: Contains module metadata
   - Linked to `houdini_module_data` via `parent_name`

3. **`houdini_node_types`**: Contains node type information
4. **`houdini_categories`**: Contains category information
5. **`pdg_registry`**: Contains PDG component information

### Database Statistics (After Fixes)
- **Modules**: 3,685
- **Functions**: 48,693 (was showing 0 before fix)
- **Classes**: 10,716 (was showing 0 before fix)
- **Node Types**: 4,205
- **Categories**: 18
- **PDG Registry Entries**: 131

## Test Results

### Function Search Test
```python
# Search for functions with 'categories' in name
functions = db.search_functions_by_keyword('categories', 10)
# Result: 10 functions found (was 0 before fix)
```

### Module Search Test
```python
# Search for functions in 'hou' module (includes _hou)
functions = db.search_functions_by_module('hou', 20)
# Result: 20 functions found, including nodeTypeCategories
```

### Specific Function Test
```python
# Search for nodeTypeCategories specifically
functions = db.search_functions_by_keyword('nodeTypeCategories', 5)
# Result: nodeTypeCategories in hou (orig: _hou) -> builtin_function_or_method
```

## MCP Tools Affected

### Fixed Tools (Now Working)
1. `f1e_search_functions` - Now returns actual results
2. `f1e_get_functions_returning_nodes` - Now finds node-returning functions
3. `f1e_get_primitive_functions` - Now finds primitive-related functions
4. `f1e_enhanced_search_functions` - Now has base data to enhance
5. `f1e_get_database_stats` - Now shows correct function/class counts
6. `f1e_get_modules_summary` - Now shows correct function counts per module

### New Tool
1. `f1e_search_functions_by_module` - Search functions within specific modules

### Unchanged Tools (Were Already Working)
1. `f1e_search_node_types` - Uses different table
2. `f1e_get_node_types_by_category` - Uses different table
3. `f1e_get_pdg_registry` - Uses different table
4. `f1e_search_pdg_registry` - Uses different table

## Example Usage

### Finding nodeTypeCategories
```python
# Now works correctly:
result = search_functions_by_keyword("nodeTypeCategories")
# Returns: nodeTypeCategories in hou module

# Or search by module:
result = search_functions_by_module("hou")
# Returns: All hou functions including those from _hou
```

### Getting Correct Statistics
```python
# Now shows accurate counts:
stats = get_database_stats()
# Returns: 48,693 functions (not 0)
```

## Architecture Notes

### Module Aliasing Pattern
Python C extensions often use internal modules prefixed with underscore (`_hou`) that are then aliased to public modules (`hou`). Our fix handles this by:

1. **Searching**: When searching for module 'hou', also search '_hou'
2. **Display**: Show '_hou' functions as 'hou' functions in results
3. **Tracking**: Keep original module name for debugging/completeness

### SQL Query Patterns
All function-related queries now use this pattern:
```sql
WHERE type = 'FUNCTION'  -- Uppercase, no JSON variants
```

### Error Handling
Maintained existing error handling patterns while fixing underlying data access issues.

## Next Steps

### Potential Improvements
1. **Comprehensive Module Aliasing**: Extend to other underscore-prefixed modules if needed
2. **Query Optimization**: Add indexes if search performance becomes an issue
3. **Enhanced Descriptions**: Further improve tool descriptions based on user feedback
4. **Documentation**: Update MCP server documentation to reflect fixes

### Monitoring
- Monitor MCP tool usage to ensure fixes are effective
- Watch for any remaining edge cases in module aliasing
- Track performance impact of module aliasing logic
