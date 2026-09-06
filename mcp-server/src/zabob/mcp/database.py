"""
Database query functions for the Zabob MCP server.

This module provides functions to query the Houdini analysis database
and return structured information about modules, functions, node types, etc.
"""

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass
import sys

# Add the paths for zabob modules
ROOT = Path(__file__).parent.parent.parent.parent.parent
CORE_SRC = ROOT / 'zabob-modules/src'
COMMON_SRC = ROOT / 'houdini/zcommon/src'

for p in (CORE_SRC, COMMON_SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from zabob.core.paths import ZABOB_OUT_DIR, ZABOB_HOUDINI_DATA


@dataclass
class FunctionInfo:
    """Information about a Houdini function."""
    name: str
    module: str
    parent_name: str
    parent_type: str
    datatype: str
    docstring: str | None
    returns_nodes: bool = False


@dataclass
class ModuleInfo:
    """Information about a Houdini module."""
    name: str
    directory: str | None
    file: str | None
    status: str
    function_count: int


@dataclass
class NodeTypeInfo:
    """Information about a Houdini node type."""
    name: str
    category: str
    description: str
    min_inputs: int
    max_inputs: int
    max_outputs: int
    is_generator: bool
    is_manager: bool


@dataclass
class PDGRegistryInfo:
    """Information about a PDG registry entry."""
    name: str
    registry: str


class HoudiniDatabase:
    """Interface to the Houdini analysis database."""

    def __init__(self, db_path: Path | None = None):
        """Initialize database connection."""
        if db_path is None:
            # Try to find the database in the standard locations
            db_path = self._find_database()

        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def _find_database(self) -> Path:
        """Find the Houdini database in standard locations."""
        # Try development database first
        dev_db = ZABOB_OUT_DIR / "20.5.584" / "houdini_data_dev.db"
        if dev_db.exists():
            return dev_db

        # Try production database
        prod_db = ZABOB_HOUDINI_DATA / "20.5.584" / "houdini_data.db"
        if prod_db.exists():
            return prod_db

        # Fall back to any database in the dev_out directory
        for db_file in ZABOB_OUT_DIR.rglob("*.db"):
            if "houdini_data" in db_file.name:
                return db_file

        raise FileNotFoundError("Could not find Houdini analysis database")

    def connect(self) -> sqlite3.Connection:
        """Get database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def get_functions_returning_nodes(self) -> List[FunctionInfo]:
        """Find functions that return node types."""
        conn = self.connect()
        cursor = conn.cursor()

        # Look for functions with return types containing 'Node' or specific node types
        query = """
        SELECT name, parent_name, parent_type, datatype, docstring
        FROM houdini_module_data
        WHERE type = 'FUNCTION'
        AND (datatype LIKE '%Node%'
             OR datatype LIKE '%hou.Node%'
             OR datatype LIKE '%GeometryNode%'
             OR datatype LIKE '%SopNode%'
             OR datatype LIKE '%ObjNode%')
        ORDER BY parent_name, name
        """

        cursor.execute(query)
        results = []

        for row in cursor.fetchall():
            # Handle module aliasing: if parent_name starts with '_', also expose without underscore
            display_module = row['parent_name']
            if display_module.startswith('_') and len(display_module) > 1:
                display_module = display_module[1:]

            func_info = FunctionInfo(
                name=row['name'],
                module=display_module,
                parent_name=row['parent_name'],
                parent_type=row['parent_type'],
                datatype=row['datatype'],
                docstring=row['docstring'],
                returns_nodes=True
            )
            results.append(func_info)

        return results

    def search_functions_by_keyword(self, keyword: str, limit: int = 50) -> List[FunctionInfo]:
        """Search for functions by keyword in name or docstring."""
        conn = self.connect()
        cursor = conn.cursor()

        # Search in function names and docstrings
        query = """
        SELECT name, parent_name, parent_type, datatype, docstring
        FROM houdini_module_data
        WHERE type = 'FUNCTION'
        AND (name LIKE ? OR docstring LIKE ?)
        ORDER BY
            CASE WHEN name LIKE ? THEN 1 ELSE 2 END,
            parent_name, name
        LIMIT ?
        """

        keyword_pattern = f"%{keyword}%"
        name_pattern = f"%{keyword}%"

        cursor.execute(query, (keyword_pattern, keyword_pattern, name_pattern, limit))
        results = []

        for row in cursor.fetchall():
            # Handle module aliasing: if parent_name starts with '_', also expose without underscore
            display_module = row['parent_name']
            if display_module.startswith('_') and len(display_module) > 1:
                # For _hou -> hou, _something -> something
                display_module = display_module[1:]

            func_info = FunctionInfo(
                name=row['name'],
                module=display_module,
                parent_name=row['parent_name'],
                parent_type=row['parent_type'],
                datatype=row['datatype'],
                docstring=row['docstring']
            )
            results.append(func_info)

        return results

    def get_primitive_related_functions(self) -> List[FunctionInfo]:
        """Find functions related to primitive operations."""
        conn = self.connect()
        cursor = conn.cursor()

        # Look for functions with 'primitive', 'prim', 'geometry' in name or docstring
        query = """
        SELECT name, parent_name, parent_type, datatype, docstring
        FROM houdini_module_data
        WHERE type = 'FUNCTION'
        AND (name LIKE '%primitive%'
             OR name LIKE '%prim%'
             OR name LIKE '%geometry%'
             OR name LIKE '%geo%'
             OR docstring LIKE '%primitive%'
             OR docstring LIKE '%geometry%'
             OR docstring LIKE '%group%')
        ORDER BY
            CASE
                WHEN name LIKE '%primitive%' THEN 1
                WHEN name LIKE '%prim%' THEN 2
                WHEN name LIKE '%geometry%' THEN 3
                WHEN name LIKE '%geo%' THEN 4
                ELSE 5
            END,
            parent_name, name
        """

        cursor.execute(query)
        results = []

        for row in cursor.fetchall():
            # Handle module aliasing: if parent_name starts with '_', also expose without underscore
            display_module = row['parent_name']
            if display_module.startswith('_') and len(display_module) > 1:
                display_module = display_module[1:]

            func_info = FunctionInfo(
                name=row['name'],
                module=display_module,
                parent_name=row['parent_name'],
                parent_type=row['parent_type'],
                datatype=row['datatype'],
                docstring=row['docstring']
            )
            results.append(func_info)

        return results

    def search_functions_by_module(self, module_name: str, limit: int = 100) -> List[FunctionInfo]:
        """Search for functions by module name, handling module aliasing."""
        conn = self.connect()
        cursor = conn.cursor()

        # Search both the module name and its underscore variant
        # e.g., search for both 'hou' and '_hou'
        search_modules = [module_name]
        if not module_name.startswith('_'):
            search_modules.append(f'_{module_name}')

        placeholders = ','.join(['?'] * len(search_modules))
        query = f"""
        SELECT name, parent_name, parent_type, datatype, docstring
        FROM houdini_module_data
        WHERE type = 'FUNCTION'
        AND parent_name IN ({placeholders})
        ORDER BY name
        LIMIT ?
        """

        cursor.execute(query, search_modules + [limit])
        results = []

        for row in cursor.fetchall():
            # Handle module aliasing: if parent_name starts with '_', also expose without underscore
            display_module = row['parent_name']
            if display_module.startswith('_') and len(display_module) > 1:
                display_module = display_module[1:]

            func_info = FunctionInfo(
                name=row['name'],
                module=display_module,
                parent_name=row['parent_name'],
                parent_type=row['parent_type'],
                datatype=row['datatype'],
                docstring=row['docstring']
            )
            results.append(func_info)

        return results

    def get_modules_summary(self) -> List[ModuleInfo]:
        """Get a summary of all modules."""
        conn = self.connect()
        cursor = conn.cursor()

        # Get module info with function counts
        query = """
        SELECT
            m.name,
            m.directory,
            m.file,
            m.status,
            COUNT(md.name) as function_count
        FROM houdini_modules m
        LEFT JOIN houdini_module_data md ON m.name = md.parent_name
            AND md.type = 'FUNCTION'
        GROUP BY m.name, m.directory, m.file, m.status
        ORDER BY m.name
        """

        cursor.execute(query)
        results = []

        for row in cursor.fetchall():
            module_info = ModuleInfo(
                name=row['name'],
                directory=row['directory'],
                file=row['file'],
                status=row['status'],
                function_count=row['function_count']
            )
            results.append(module_info)

        return results

    def get_node_types_by_category(self, category: str | None = None) -> List[NodeTypeInfo]:
        """Get node types, optionally filtered by category."""
        conn = self.connect()
        cursor = conn.cursor()

        if category:
            # Handle both old JSON format ("Sop") and new plain format (Sop)
            query = """
            SELECT name, category, description, minNumInputs, maxNumInputs,
                   maxNumOutputs, isGenerator, isManager
            FROM houdini_node_types
            WHERE category = ? OR category = ?
            ORDER BY name
            """
            cursor.execute(query, (f'"{category}"', category))
        else:
            query = """
            SELECT name, category, description, minNumInputs, maxNumInputs,
                   maxNumOutputs, isGenerator, isManager
            FROM houdini_node_types
            ORDER BY category, name
            """
            cursor.execute(query)

        results = []

        for row in cursor.fetchall():
            # Handle boolean values that might be stored as strings
            def parse_bool(value):
                if isinstance(value, str):
                    return value.lower() in ('true', '1', 'yes')
                return bool(value)

            node_info = NodeTypeInfo(
                name=row['name'],
                category=row['category'],
                description=row['description'],
                min_inputs=int(row['minNumInputs']),
                max_inputs=int(row['maxNumInputs']),
                max_outputs=int(row['maxNumOutputs']),
                is_generator=parse_bool(row['isGenerator']),
                is_manager=parse_bool(row['isManager'])
            )
            results.append(node_info)

        return results

    def search_node_types(self, keyword: str, limit: int = 50) -> List[NodeTypeInfo]:
        """Search node types by keyword."""
        conn = self.connect()
        cursor = conn.cursor()

        query = """
        SELECT name, category, description, minNumInputs, maxNumInputs,
               maxNumOutputs, isGenerator, isManager
        FROM houdini_node_types
        WHERE name LIKE ? OR description LIKE ?
        ORDER BY
            CASE WHEN name LIKE ? THEN 1 ELSE 2 END,
            category, name
        LIMIT ?
        """

        keyword_pattern = f"%{keyword}%"
        name_pattern = f"%{keyword}%"

        cursor.execute(query, (keyword_pattern, keyword_pattern, name_pattern, limit))
        results = []

        for row in cursor.fetchall():
            # Handle boolean values that might be stored as strings
            def parse_bool(value):
                if isinstance(value, str):
                    return value.lower() in ('true', '1', 'yes')
                return bool(value)

            node_info = NodeTypeInfo(
                name=row['name'],
                category=row['category'],
                description=row['description'],
                min_inputs=int(row['minNumInputs']),
                max_inputs=int(row['maxNumInputs']),
                max_outputs=int(row['maxNumOutputs']),
                is_generator=parse_bool(row['isGenerator']),
                is_manager=parse_bool(row['isManager'])
            )
            results.append(node_info)

        return results

    def get_database_stats(self) -> Dict[str, int]:
        """Get statistics about the database contents."""
        conn = self.connect()
        cursor = conn.cursor()

        stats = {}

        # Count modules
        cursor.execute("SELECT COUNT(*) FROM houdini_modules")
        stats['modules'] = cursor.fetchone()[0]

        # Count functions
        cursor.execute("SELECT COUNT(*) FROM houdini_module_data WHERE type = 'FUNCTION'")
        stats['functions'] = cursor.fetchone()[0]

        # Count classes
        cursor.execute("SELECT COUNT(*) FROM houdini_module_data WHERE type = 'CLASS'")
        stats['classes'] = cursor.fetchone()[0]

        # Count node types
        cursor.execute("SELECT COUNT(*) FROM houdini_node_types")
        stats['node_types'] = cursor.fetchone()[0]

        # Count categories
        cursor.execute("SELECT COUNT(*) FROM houdini_categories")
        stats['categories'] = cursor.fetchone()[0]

        # Count PDG registry entries
        cursor.execute("SELECT COUNT(*) FROM pdg_registry")
        stats['pdg_registry_entries'] = cursor.fetchone()[0]

        return stats

    def get_pdg_registry(self, registry_type: str | None = None) -> List[PDGRegistryInfo]:
        """Get PDG registry entries, optionally filtered by registry type."""
        conn = self.connect()
        cursor = conn.cursor()

        if registry_type:
            query = """
            SELECT name, registry
            FROM pdg_registry
            WHERE registry = ?
            ORDER BY name
            """
            cursor.execute(query, (registry_type,))
        else:
            query = """
            SELECT name, registry
            FROM pdg_registry
            ORDER BY registry, name
            """
            cursor.execute(query)

        results = []
        for row in cursor.fetchall():
            pdg_info = PDGRegistryInfo(
                name=row['name'],
                registry=row['registry']
            )
            results.append(pdg_info)

        return results

    def search_pdg_registry(self, keyword: str, limit: int = 50) -> List[PDGRegistryInfo]:
        """Search PDG registry entries by keyword in name."""
        conn = self.connect()
        cursor = conn.cursor()

        query = """
        SELECT name, registry
        FROM pdg_registry
        WHERE name LIKE ?
        ORDER BY
            CASE
                WHEN name = ? THEN 1
                WHEN name LIKE ? THEN 2
                WHEN name LIKE ? THEN 3
                ELSE 4
            END,
            registry, name
        LIMIT ?
        """

        search_pattern = f"%{keyword}%"
        exact_match = keyword
        starts_with = f"{keyword}%"
        ends_with = f"%{keyword}"

        cursor.execute(query, (search_pattern, exact_match, starts_with, ends_with, limit))

        results = []
        for row in cursor.fetchall():
            pdg_info = PDGRegistryInfo(
                name=row['name'],
                registry=row['registry']
            )
            results.append(pdg_info)

        return results
