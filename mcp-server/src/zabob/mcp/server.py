#!/usr/bin/env uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "aiohttp",
#     "aiopath",
#     "anyio",
#     "click>=8.0.0,<8.2.0",
#     "fastapi",
#     "fastmcp",
#     "httpx",
#     "mcp",
#     "pedantic",
#     "psutil",
#     "semver",
#     "sqlite-vec",
#     "uvicorn",
# ]
# ///
'''
An MCP server for the Zabob project.
'''

from collections.abc import AsyncGenerator, AsyncIterable, Awaitable, AsyncIterator
from doctest import debug
import json
from typing import Any, TypeVar, cast, TypedDict
import asyncio
import sys
import click
import httpx
import logging
import subprocess
from contextlib import asynccontextmanager

from aiopath.path import AsyncPath as Path
from pathlib import Path as SyncPath

from mcp.server.fastmcp import FastMCP


ROOT = SyncPath(__file__).parent.parent.parent.parent.parent
MCP_SRC = ROOT/ 'mcp-server/src'
CORE_SRC = ROOT / 'zabob-modules/src'
COMMON_SRC = ROOT / 'houdini/zcommon/src'


# Check for source directories (but not .venv in Docker)
for p in (MCP_SRC, CORE_SRC, COMMON_SRC):
    if not p.exists():
        print(f"Error: {p} does not exist. Please run 'zabob setup' first.", file=sys.stderr)
        sys.exit(1)
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))  # type: ignore[no-redef]

from zabob.common import DEBUG, INFO, ZABOB_OUT_DIR, Level, get_houdini, spawn, config_logging
from zabob.core import JsonData
from zabob.mcp.database import HoudiniDatabase
from zabob.common.hython import run_houdini_script
from zabob.common.subproc import spawn, check_pid

DEFAULT_LOG = ZABOB_OUT_DIR / 'logs/mcp_server.log'
HYTHON_LOG = DEFAULT_LOG.with_stem('hython_mcp_server')

# TypedDict definitions for better type safety
class SearchResult(TypedDict):
    title: str
    url: str
    snippet: str

class WebSearchResponse(TypedDict):
    query: str
    results: list[SearchResult]
    error: str | None

class NodeTypeInfo(TypedDict):
    name: str
    category: str
    description: str
    inputs: str
    outputs: int
    is_generator: bool

class EnhancedNodeTypeInfo(NodeTypeInfo):
    documentation_search: list[SearchResult]
    official_docs: dict[str, Any] | None

class FunctionInfo(TypedDict):
    name: str
    module: str
    datatype: str
    docstring: str

class EnhancedFunctionInfo(FunctionInfo):
    example_search: list[SearchResult]
    hom_docs: dict[str, Any] | None



RESPONSES_DIR = Path(__file__).parent / "responses"
PROMPTS_DIR = Path(__file__).parent / "prompts"
INSTRUCTIONS_PATH = SyncPath(__file__).parent / "instructions.md"
with open(INSTRUCTIONS_PATH, "r", encoding="utf-8") as f:
    INSTRUCTIONS = f.read()

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[None]:
    """Manage application lifecycle - initialize responses and start hython server on startup."""
    try:
        # Initialize responses and prompts during startup
        # This awaits the listing of responses, but not their loading.
        # That happens at the point of first use, or on exit.
        await load_responses()

        # Start the hython MCP server
        await start_hython_server()

        yield
    finally:
        # Stop the hython server first
        await stop_hython_server()

        # Cancel any remaining tasks for graceful shutdown
        current_task = asyncio.current_task()
        tasks = [task for task in asyncio.all_tasks() if task != current_task and not task.done()]
        if tasks:
            for task in tasks:
                task.cancel()
            # Wait a short time for tasks to cleanup
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=2.0
                )
            except asyncio.TimeoutError:
                # Tasks didn't cleanup in time, but that's okay
                pass

mcp = FastMCP("zabob", instructions=INSTRUCTIONS, lifespan=app_lifespan)

RESPONSES: dict[str, Awaitable[JsonData|str]] =  {}
PROMPTS: dict[str, Awaitable[JsonData|str]] =  {}

# Global variable to store the hython MCP server process
HYTHON_SERVER_PROCESS: subprocess.Popen | None = None



async def load_responses():
    """Load response JSON and Markdown files."""
    async def load_text(f: Path):
        if await f.is_file():
            async with f.open("r", encoding="utf-8") as stream:
                return await stream.read()
    async def load_json(f: Path):
        text = await load_text(f)
        if text:
            return json.loads(text)
    async for f in cast(AsyncIterable[Path], RESPONSES_DIR.glob("*.json")):
        RESPONSES[f.stem] = load_json(f)
    async for f in cast(AsyncIterable[Path], RESPONSES_DIR.glob("*.md")):
        RESPONSES[f.stem] = load_text(f)
    async for f in cast(AsyncIterable[Path], PROMPTS_DIR.glob("*.md")):
        PROMPTS[f.stem] = load_text(f)


async def start_hython_server():
    """Start the hython MCP server as a subprocess using the hython.py wrapper."""
    global HYTHON_SERVER_PROCESS

    if HYTHON_SERVER_PROCESS is not None:
        logging.warning("Hython server already running")
        return

    try:
        houdini = get_houdini()
        # Path to the hython MCP server script
        hython_server_path = ROOT / "houdini" / "h20.5" / "src" / "zabob" / "h20_5" / "hython_mcp_server.py"

        if not hython_server_path.exists():
            logging.error(f"Hython server script not found: {hython_server_path}")
            return

        # Path to the hython.py wrapper script
        hython_wrapper_path = ROOT / "houdini" / "zcommon" / "src" / "zabob" / "common" / "hython.py"

        if not hython_wrapper_path.exists():
            logging.error(f"Hython wrapper script not found: {hython_wrapper_path}")
            return

        logging.info(f"Starting hython MCP server via module: zabob.h20_5.hython_mcp_server")

        if DEBUG:
            debug_flag = ("--debug",)
        else:
            debug_flag = ()
        # Use Python to run the hython.py wrapper with -m flag for the MCP server module
        # This pattern ensures proper hython environment setup:
        # python hython.py -m zabob.h20_5.hython_mcp_server
        HYTHON_SERVER_PROCESS = spawn(
            houdini.hython, hython_wrapper_path, "--module", "zabob.h20_5.hython_mcp_server",
            "--log-file", HYTHON_LOG, *debug_flag,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0,  # Unbuffered for real-time communication
            cwd=str(ROOT)  # Set working directory to project root
        )

        # Give the process a moment to start
        await asyncio.sleep(2)

        if HYTHON_SERVER_PROCESS.returncode is not None:
            logging.error(f"Hython server failed to start with return code: {HYTHON_SERVER_PROCESS.returncode}")
            if HYTHON_SERVER_PROCESS.stderr:
                stderr_output = HYTHON_SERVER_PROCESS.stderr.read()
                logging.error(f"Hython server stderr: {stderr_output}")
            HYTHON_SERVER_PROCESS = None
        else:
            logging.info(f"Hython server started successfully with PID: {HYTHON_SERVER_PROCESS.pid}")
        if HYTHON_SERVER_PROCESS is not None:
            proc = HYTHON_SERVER_PROCESS
            async def monitor():
                global HYTHON_SERVER_PROCESS
                proc.wait()
                if proc.returncode is not None:
                    logging.error(f"MONITOR: Hython server process exited with return code: {proc.returncode}")
                    if proc.stderr:
                        stderr_output = proc.stderr.read()
                        if stderr_output:
                            # Log stderr output if available
                            logging.error(f"MONITOR: Hython server stderr: {stderr_output}")
                    if proc.stdout:
                        stdout_output = proc.stdout.read()
                        if stdout_output:
                            # Log stderr output if available
                            logging.error(f"MONITOR: Hython server stdout: {stdout_output}")
                    HYTHON_SERVER_PROCESS = None
            asyncio.create_task(monitor())

    except Exception as e:
        logging.error(f"Failed to start hython server: {e}")
        HYTHON_SERVER_PROCESS = None


async def stop_hython_server():
    """Stop the hython MCP server subprocess."""
    global HYTHON_SERVER_PROCESS

    if HYTHON_SERVER_PROCESS is None:
        return

    try:
        logging.info(f"Stopping hython server with PID: {HYTHON_SERVER_PROCESS.pid}")

        # Try graceful termination first
        HYTHON_SERVER_PROCESS.terminate()

        # Wait up to 5 seconds for graceful shutdown
        try:
            await asyncio.wait_for(
                asyncio.create_task(asyncio.to_thread(HYTHON_SERVER_PROCESS.wait)),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            # Force kill if graceful termination didn't work
            logging.warning("Hython server didn't terminate gracefully, forcing kill")
            HYTHON_SERVER_PROCESS.kill()
            await asyncio.create_task(asyncio.to_thread(HYTHON_SERVER_PROCESS.wait))

        logging.info("Hython server stopped successfully")

    except Exception as e:
        logging.error(f"Error stopping hython server: {e}")
    finally:
        HYTHON_SERVER_PROCESS = None


# Hython server communication
async def call_hython_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Call a tool on the hython MCP server via subprocess communication."""
    if HYTHON_SERVER_PROCESS is None or HYTHON_SERVER_PROCESS.returncode is not None:
        return {
            "success": False,
            "error": "Hython server is not running"
        }

    if not HYTHON_SERVER_PROCESS.stdin or not HYTHON_SERVER_PROCESS.stdout:
        return {
            "success": False,
            "error": "Hython server pipes not available"
        }

    try:
        # Create a request message for the hython server
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }

        # Send request to hython process stdin
        request_json = json.dumps(request) + "\n"
        HYTHON_SERVER_PROCESS.stdin.write(request_json)
        await asyncio.create_task(asyncio.to_thread(HYTHON_SERVER_PROCESS.stdin.flush))

        # Read response from stdout with timeout
        try:
            response_line = await asyncio.wait_for(
                asyncio.create_task(asyncio.to_thread(HYTHON_SERVER_PROCESS.stdout.readline)),
                timeout=30.0
            )
            if response_line:
                response_json = response_line.strip()
                response_data = json.loads(response_json)

                if "result" in response_data:
                    return response_data["result"]
                elif "error" in response_data:
                    return {
                        "success": False,
                        "error": f"Hython server error: {response_data['error']}"
                    }
                else:
                    return {
                        "success": False,
                        "error": "Invalid response from hython server"
                    }
            else:
                return {
                    "success": False,
                    "error": "No response from hython server"
                }
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": f"Timeout calling hython tool '{tool_name}'"
            }

    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error calling hython tool '{tool_name}': {str(e)}: {HYTHON_SERVER_PROCESS.stderr.read() if HYTHON_SERVER_PROCESS.stderr else 'No stderr available'}"
        }

async def get_hython_resource(uri: str) -> dict[str, Any]:
    """Get a resource from the hython MCP server via subprocess communication."""
    if HYTHON_SERVER_PROCESS is None or HYTHON_SERVER_PROCESS.returncode is not None:
        return {
            "success": False,
            "error": "Hython server is not running"
        }

    if not HYTHON_SERVER_PROCESS.stdin or not HYTHON_SERVER_PROCESS.stdout:
        return {
            "success": False,
            "error": "Hython server pipes not available"
        }

    try:
        # Create a request message for the hython server
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "resources/read",
            "params": {
                "uri": uri
            }
        }

        # Send request to hython process stdin
        request_json = json.dumps(request) + "\n"
        HYTHON_SERVER_PROCESS.stdin.write(request_json)
        await asyncio.create_task(asyncio.to_thread(HYTHON_SERVER_PROCESS.stdin.flush))

        # Read response from stdout with timeout
        try:
            response_line = await asyncio.wait_for(
                asyncio.create_task(asyncio.to_thread(HYTHON_SERVER_PROCESS.stdout.readline)),
                None, #timeout=30.0
            )
            if response_line:
                response_json = response_line.strip()
                response_data = json.loads(response_json)

                if "result" in response_data:
                    return response_data["result"]
                elif "error" in response_data:
                    return {
                        "success": False,
                        "error": f"Hython server error: {response_data['error']}"
                    }
                else:
                    return {
                        "success": False,
                        "error": "Invalid response from hython server"
                    }
            else:
                return {
                    "success": False,
                    "error": "No response from hython server"
                }
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": f"Timeout getting hython resource '{uri}'"
            }

    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error getting hython resource '{uri}': {str(e)}"
        }


T = TypeVar("T")
def awaitable_value(value: T) -> Awaitable[T]:
    async def wrapper() -> AsyncGenerator[T, None]:
        yield  value
    return anext(aiter(wrapper()))

# Initialize database connection
db = HoudiniDatabase()

# Web search integration helpers
async def vscode_websearchforcopilot_webSearch(query: str, num_results: int = 5) -> dict[str, Any]:
    """Perform web search using DuckDuckGo instant answer API."""
    try:
        # Use DuckDuckGo's instant answer API (no API key required)
        async with httpx.AsyncClient() as client:
            # DuckDuckGo instant answer API
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1"
            }

            response = await client.get(
                "https://api.duckduckgo.com/",
                params=params,
                timeout=10.0
            )

            if response.status_code == 200:
                data = response.json()
                results = []

                # Extract abstract/definition if available
                if data.get("Abstract"):
                    results.append({
                        "title": data.get("AbstractText", query),
                        "url": data.get("AbstractURL", ""),
                        "snippet": data.get("Abstract", "")[:300]
                    })

                # Extract related topics
                for topic in data.get("RelatedTopics", [])[:num_results-len(results)]:
                    if isinstance(topic, dict) and "Text" in topic:
                        results.append({
                            "title": topic.get("FirstURL", "").split("/")[-1].replace("_", " "),
                            "url": topic.get("FirstURL", ""),
                            "snippet": topic.get("Text", "")[:200]
                        })

                # If no results, create a basic search result
                if not results:
                    results.append({
                        "title": f"Search results for '{query}'",
                        "url": f"https://duckduckgo.com/?q={query.replace(' ', '+')}",
                        "snippet": f"No direct results found. Try searching on DuckDuckGo for more information about '{query}'."
                    })

                return {
                    "query": query,
                    "results": results[:num_results],
                    "error": None
                }
            else:
                logging.warning(f"DuckDuckGo API returned status {response.status_code}")
                return {
                    "query": query,
                    "results": [{
                        "title": f"Search for '{query}'",
                        "url": f"https://duckduckgo.com/?q={query.replace(' ', '+')}",
                        "snippet": "Search results available on DuckDuckGo"
                    }],
                    "error": None
                }

    except Exception as e:
        logging.error(f"Web search failed: {e}")
        return {
            "query": query,
            "results": [{
                "title": f"Search for '{query}'",
                "url": f"https://duckduckgo.com/?q={query.replace(' ', '+')}",
                "snippet": "Search functionality temporarily unavailable"
            }],
            "error": str(e)
        }

async def fetch_webpage(urls: list[str], query: str) -> str:
    """Fetch content from web pages."""
    try:
        async with httpx.AsyncClient() as client:
            for url in urls:
                try:
                    response = await client.get(url, timeout=10.0)
                    if response.status_code == 200:
                        # Simple text extraction (in practice, you'd want better HTML parsing)
                        content = response.text
                        # Return first 1000 characters as preview
                        return content[:1000] if len(content) > 1000 else content
                except Exception as e:
                    logging.warning(f"Failed to fetch {url}: {e}")
                    continue
        return "No content could be fetched from provided URLs"
    except Exception as e:
        logging.error(f"Webpage fetch failed: {e}")
        return f"Fetch error: {str(e)}"

@mcp.tool("get_functions_returning_nodes")
async def get_functions_returning_nodes():
    """Find functions that return Houdini node objects. Searches the houdini_module_data table for functions with Node-related return types."""
    try:
        with db:
            functions = db.get_functions_returning_nodes()
            return {
                "functions": [
                    {
                        "name": f.name,
                        "module": f.module,
                        "datatype": f.datatype,
                        "docstring": f.docstring[:200] + "..." if f.docstring and len(f.docstring) > 200 else f.docstring
                    }
                    for f in functions
                ],
                "count": len(functions)
            }
    except Exception as e:
        return {"error": f"Database query failed: {str(e)}"}

@mcp.tool("search_functions")
async def search_functions(keyword: str, limit: int = 20):
    """Search for functions by keyword in name or docstring. Searches across all Houdini modules in the database."""
    if not keyword:
        return {"error": "No keyword provided."}

    try:
        with db:
            functions = db.search_functions_by_keyword(keyword, limit)
            return {
                "keyword": keyword,
                "functions": [
                    {
                        "name": f.name,
                        "module": f.module,
                        "datatype": f.datatype,
                        "docstring": f.docstring[:200] + "..." if f.docstring and len(f.docstring) > 200 else f.docstring
                    }
                    for f in functions
                ],
                "count": len(functions)
            }
    except Exception as e:
        return {"error": f"Database query failed: {str(e)}"}

@mcp.tool("search_functions_by_module")
async def search_functions_by_module(module_name: str, limit: int = 50):
    """Search for functions within a specific module (e.g., 'hou', 'toolutils'). Handles module aliasing automatically."""
    if not module_name:
        return {"error": "No module name provided."}

    try:
        with db:
            functions = db.search_functions_by_module(module_name, limit)
            return {
                "module": module_name,
                "functions": [
                    {
                        "name": f.name,
                        "module": f.module,
                        "original_module": f.parent_name,
                        "datatype": f.datatype,
                        "docstring": f.docstring[:200] + "..." if f.docstring and len(f.docstring) > 200 else f.docstring
                    }
                    for f in functions
                ],
                "count": len(functions),
                "note": "Module aliasing applied (e.g., _hou functions shown as hou)"
            }
    except Exception as e:
        return {"error": f"Database query failed: {str(e)}"}

@mcp.tool("get_primitive_functions")
async def get_primitive_functions():
    """Find functions related to primitive operations (selection, manipulation, etc.). Searches the houdini_module_data table."""
    try:
        with db:
            functions = db.get_primitive_related_functions()
            return {
                "functions": [
                    {
                        "name": f.name,
                        "module": f.module,
                        "datatype": f.datatype,
                        "docstring": f.docstring[:200] + "..." if f.docstring and len(f.docstring) > 200 else f.docstring
                    }
                    for f in functions
                ],
                "count": len(functions)
            }
    except Exception as e:
        return {"error": f"Database query failed: {str(e)}"}

@mcp.tool("get_modules_summary")
async def get_modules_summary():
    """Get a summary of all Houdini modules with function counts. Uses houdini_modules and houdini_module_data tables."""
    try:
        with db:
            modules = db.get_modules_summary()
            return {
                "modules": [
                    {
                        "name": m.name,
                        "status": m.status,
                        "function_count": m.function_count,
                        "file": m.file
                    }
                    for m in modules[:50]  # Limit to first 50 for readability
                ],
                "total_count": len(modules)
            }
    except Exception as e:
        return {"error": f"Database query failed: {str(e)}"}

@mcp.tool("search_node_types")
async def search_node_types(keyword: str, limit: int = 20):
    """Search for node types by keyword in name or description."""
    if not keyword:
        return {"error": "No keyword provided."}

    try:
        with db:
            node_types = db.search_node_types(keyword, limit)
            return {
                "keyword": keyword,
                "node_types": [
                    {
                        "name": nt.name,
                        "category": nt.category,
                        "description": nt.description,
                        "inputs": f"{nt.min_inputs}-{nt.max_inputs}",
                        "outputs": nt.max_outputs,
                        "is_generator": nt.is_generator
                    }
                    for nt in node_types
                ],
                "count": len(node_types)
            }
    except Exception as e:
        return {"error": f"Database query failed: {str(e)}"}

@mcp.tool("enhanced_search_node_types")
async def enhanced_search_node_types(keyword: str, include_docs: bool = True, limit: int = 5):
    """Search node types with optional live documentation integration."""
    try:
        with db:
            # Get static database results
            node_types = db.search_node_types(keyword, limit)

            basic_results = []
            for nt in node_types:
                basic_results.append({
                    "name": nt.name,
                    "category": nt.category,
                    "description": nt.description,
                    "inputs": f"{nt.min_inputs}-{nt.max_inputs}",
                    "outputs": nt.max_outputs,
                    "is_generator": nt.is_generator
                })

            result = {
                "keyword": keyword,
                "node_types": basic_results,
                "count": len(node_types)
            }

            if include_docs and node_types:
                # Enhance top 3 results with live documentation
                enhanced_nodes = []

                for node in node_types[:3]:
                    enhanced_node: dict[str, Any] = {
                        "name": node.name,
                        "category": node.category,
                        "description": node.description,
                        "inputs": f"{node.min_inputs}-{node.max_inputs}",
                        "outputs": node.max_outputs,
                        "is_generator": node.is_generator
                    }

                    # Use web search to find documentation
                    search_query = f"Houdini {node.name} {node.category} node documentation examples"
                    search_results = await vscode_websearchforcopilot_webSearch(search_query)
                    enhanced_node["documentation_search"] = search_results.get("results", [])[:3]

                    # Try to fetch SideFX official documentation
                    if node.category.lower() in ['sop', 'top', 'object', 'dop']:
                        doc_url = f"https://www.sidefx.com/docs/houdini/nodes/{node.category.lower()}/{node.name}.html"
                        try:
                            doc_content = await fetch_webpage([doc_url], f"{node.name} node documentation")
                            official_docs: dict[str, Any] = {
                                "url": doc_url,
                                "content_preview": doc_content[:400] + "..." if len(doc_content) > 400 else doc_content
                            }
                            enhanced_node["official_docs"] = official_docs
                        except:
                            failed_docs: dict[str, Any] = {"url": doc_url, "status": "fetch_failed"}
                            enhanced_node["official_docs"] = failed_docs

                    enhanced_nodes.append(enhanced_node)

                result["enhanced_results"] = enhanced_nodes
                result["enhancement_note"] = "Top results enhanced with live documentation"

            return result

    except Exception as e:
        return {"error": f"Enhanced search failed: {str(e)}"}

@mcp.tool("enhanced_search_functions")
async def enhanced_search_functions(keyword: str, include_examples: bool = True, limit: int = 5):
    """Search functions with optional code examples and documentation. Searches houdini_module_data table and enhances with web search."""
    try:
        with db:
            # Get static database results
            functions = db.search_functions_by_keyword(keyword, limit)

            basic_results = []
            for f in functions:
                basic_results.append({
                    "name": f.name,
                    "module": f.module,
                    "datatype": f.datatype,
                    "docstring": f.docstring[:200] + "..." if f.docstring and len(f.docstring) > 200 else f.docstring
                })

            result = {
                "keyword": keyword,
                "functions": basic_results,
                "count": len(functions)
            }

            if include_examples and functions:
                # Enhance top 3 functions with examples
                enhanced_functions = []

                for func in functions[:3]:
                    enhanced_func: dict[str, Any] = {
                        "name": func.name,
                        "module": func.module,
                        "datatype": func.datatype,
                        "docstring": func.docstring[:200] + "..." if func.docstring and len(func.docstring) > 200 else func.docstring
                    }

                    # Search for code examples and tutorials
                    example_query = f"Houdini Python {func.name} code examples tutorial"
                    search_results = await vscode_websearchforcopilot_webSearch(example_query)
                    enhanced_func["example_search"] = search_results.get("results", [])[:3]

                    # Try to fetch official HOM documentation
                    if func.module == "hou":
                        # Build HOM documentation URL
                        doc_url = f"https://www.sidefx.com/docs/houdini/hom/hou/{func.name}.html"
                        try:
                            doc_content = await fetch_webpage([doc_url], f"{func.name} function documentation")
                            hom_docs: dict[str, Any] = {
                                "url": doc_url,
                                "content_preview": doc_content[:400] + "..." if len(doc_content) > 400 else doc_content
                            }
                            enhanced_func["hom_docs"] = hom_docs
                        except:
                            failed_hom_docs: dict[str, Any] = {"url": doc_url, "status": "fetch_failed"}
                            enhanced_func["hom_docs"] = failed_hom_docs

                    enhanced_functions.append(enhanced_func)

                result["enhanced_results"] = enhanced_functions
                result["enhancement_note"] = "Top results enhanced with code examples and documentation"

            return result

    except Exception as e:
        return {"error": f"Enhanced function search failed: {str(e)}"}

@mcp.tool("web_search_houdini")
async def web_search_houdini(query: str, num_results: int = 5):
    """Perform a web search specifically for Houdini-related content."""
    if not query:
        return {"error": "No query provided."}

    try:
        # Enhance query with Houdini context
        enhanced_query = f"Houdini 3D software {query}"
        search_results = await vscode_websearchforcopilot_webSearch(enhanced_query, num_results)

        return {
            "original_query": query,
            "enhanced_query": enhanced_query,
            "results": search_results.get("results", []),
            "count": len(search_results.get("results", []))
        }
    except Exception as e:
        return {"error": f"Web search failed: {str(e)}"}

@mcp.tool("fetch_houdini_docs")
async def fetch_houdini_docs(doc_type: str, node_name: str = "", function_name: str = ""):
    """Fetch official Houdini documentation for nodes or functions."""
    if not doc_type:
        return {"error": "No documentation type provided (node, function, or tutorial)."}

    try:
        urls = []

        if doc_type == "node" and node_name:
            # Try common node categories
            for category in ['sop', 'top', 'object', 'dop', 'chop', 'cop2']:
                urls.append(f"https://www.sidefx.com/docs/houdini/nodes/{category}/{node_name}.html")

        elif doc_type == "function" and function_name:
            urls.append(f"https://www.sidefx.com/docs/houdini/hom/hou/{function_name}.html")

        elif doc_type == "tutorial":
            # Search for tutorials
            tutorial_query = f"Houdini tutorial {node_name or function_name}"
            search_results = await vscode_websearchforcopilot_webSearch(tutorial_query)
            return {
                "doc_type": doc_type,
                "search_query": tutorial_query,
                "tutorial_results": search_results.get("results", [])[:5]
            }

        if urls:
            content = await fetch_webpage(urls, f"{doc_type} documentation")
            return {
                "doc_type": doc_type,
                "target": node_name or function_name,
                "urls_tried": urls,
                "content": content
            }
        else:
            return {"error": "Invalid parameters for documentation fetch."}

    except Exception as e:
        return {"error": f"Documentation fetch failed: {str(e)}"}

@mcp.tool("pdg_workflow_assistant")
async def pdg_workflow_assistant(workflow_description: str):
    """Get PDG components and workflow guidance for a specific task."""
    try:
        with db:
            # Extract keywords and search PDG registry
            keywords = workflow_description.lower().split()
            relevant_entries = []

            for keyword in keywords[:3]:  # Limit to avoid too many queries
                entries = db.search_pdg_registry(keyword, limit=5)
                relevant_entries.extend(entries)

            # Remove duplicates while preserving order
            seen = set()
            unique_entries = []
            for entry in relevant_entries:
                if entry.name not in seen:
                    seen.add(entry.name)
                    unique_entries.append(entry)

            result = {
                "workflow_description": workflow_description,
                "pdg_components": [
                    {
                        "name": entry.name,
                        "registry": entry.registry
                    }
                    for entry in unique_entries[:10]
                ],
                "count": len(unique_entries)
            }

            # Enhance with web search for workflow guidance
            workflow_query = f"Houdini PDG workflow {workflow_description} tutorial"
            search_results = await vscode_websearchforcopilot_webSearch(workflow_query)
            result["workflow_guidance"] = search_results.get("results", [])[:3]

            return result

    except Exception as e:
        return {"error": f"PDG workflow assistance failed: {str(e)}"}

@mcp.tool("get_node_types_by_category")
async def get_node_types_by_category(category: str = ""):
    """Get node types, optionally filtered by category (e.g., 'Sop', 'Object', 'Dop')."""
    try:
        with db:
            if category:
                node_types = db.get_node_types_by_category(category)
            else:
                node_types = db.get_node_types_by_category()[:50]  # Limit if no category

            return {
                "category": category or "all",
                "node_types": [
                    {
                        "name": nt.name,
                        "category": nt.category,
                        "description": nt.description,
                        "inputs": f"{nt.min_inputs}-{nt.max_inputs}",
                        "outputs": nt.max_outputs,
                        "is_generator": nt.is_generator
                    }
                    for nt in node_types
                ],
                "count": len(node_types)
            }
    except Exception as e:
        return {"error": f"Database query failed: {str(e)}"}

@mcp.tool("get_database_stats")
async def get_database_stats():
    """Get statistics about the Houdini database contents."""
    try:
        with db:
            stats = db.get_database_stats()
            return {
                "database_path": str(db.db_path),
                "statistics": stats
            }
    except Exception as e:
        return {"error": f"Database query failed: {str(e)}"}

@mcp.tool("get_pdg_registry")
async def get_pdg_registry(registry_type: str | None = None):
    """Get PDG (TOPs) registry entries, optionally filtered by registry type (Node, Scheduler, Service, etc.)."""
    try:
        with db:
            entries = db.get_pdg_registry(registry_type)
            return {
                "entries": [
                    {
                        "name": entry.name,
                        "registry": entry.registry
                    }
                    for entry in entries
                ],
                "count": len(entries),
                "registry_type": registry_type or "all"
            }
    except Exception as e:
        return {"error": f"Database query failed: {str(e)}"}

@mcp.tool("search_pdg_registry")
async def search_pdg_registry(keyword: str, limit: int = 50):
    """Search PDG registry entries by keyword in name."""
    try:
        with db:
            entries = db.search_pdg_registry(keyword, limit)
            return {
                "entries": [
                    {
                        "name": entry.name,
                        "registry": entry.registry
                    }
                    for entry in entries
                ],
                "count": len(entries),
                "keyword": keyword
            }
    except Exception as e:
        return {"error": f"Database query failed: {str(e)}"}

@mcp.tool("query_response")
async def query_response(query: str):
    """Handle a general query and return a canned response (legacy tool)."""
    return {"response": f'{RESPONSES_DIR}.json'}
    if not query:
        return {"error": "No query provided."}
    return {"response": await RESPONSES.get(query, awaitable_value("No response found."))}


# Hython MCP Server Proxy Tools
@mcp.tool("houdini_analyze_scene")
async def houdini_analyze_scene(file_path: str):
    """
    Analyze a Houdini scene file and return structured data with generated Python code.
    This tool proxies requests to the hython-based MCP server for analysis.

    Args:
        file_path: Path to the Houdini scene file (.hip, .hipnc, .hda)

    Returns:
        Dictionary containing analysis results, statistics, and generated Python code
    """
    return await call_hython_tool("analyze_scene", {"file_path": file_path})

@mcp.tool("houdini_get_scene_info")
async def houdini_get_scene_info():
    """
    Get information about the current Houdini scene.
    This tool proxies requests to the hython-based MCP server.

    Returns:
        Dictionary containing current scene information including node counts and categories
    """
    return await call_hython_tool("get_scene_info", {})

@mcp.tool("houdini_get_node_info")
async def houdini_get_node_info(node_path: str):
    """
    Get detailed information about a specific Houdini node.
    This tool proxies requests to the hython-based MCP server.

    Args:
        node_path: Path to the Houdini node (e.g., "/obj/geo1")

    Returns:
        Dictionary containing node information including parameters, inputs, outputs
    """
    return await call_hython_tool("get_node_info", {"node_path": node_path})

@mcp.tool("houdini_list_nodes")
async def houdini_list_nodes(path: str = "/", node_type: str = "", recursive: bool = True):
    """
    List nodes in the current Houdini scene with optional filtering.
    This tool proxies requests to the hython-based MCP server.

    Args:
        path: Starting path for node search (default: "/")
        node_type: Filter by node type (optional)
        recursive: Whether to search recursively (default: True)

    Returns:
        Dictionary containing list of nodes with their paths and types
    """
    return await call_hython_tool("list_nodes", {
        "path": path,
        "node_type": node_type,
        "recursive": recursive
    })

@mcp.tool("houdini_create_node")
async def houdini_create_node(parent_path: str, node_type: str, name: str = ""):
    """
    Create a new node in the current Houdini scene.
    This tool proxies requests to the hython-based MCP server.

    Args:
        parent_path: Path to the parent node where the new node will be created
        node_type: Type of node to create (e.g., "geo", "merge", "transform")
        name: Optional name for the new node

    Returns:
        Dictionary containing information about the created node
    """
    return await call_hython_tool("create_node", {
        "parent_path": parent_path,
        "node_type": node_type,
        "name": name
    })

@mcp.resource("houdini://current_scene")
async def houdini_current_scene():
    """Get information about the current Houdini scene as a resource."""
    return await get_hython_resource("scene://current")


@mcp.resource("status://status")
async def status() -> dict[str, Any]:
    """Return server status."""
    return {"status": "ok"}

@mcp.resource('sop://info')
def sop_info():
    """Return SOP info."""
    return {
        "name": "SOP",
        "version": "1.0.0",
        "description": "Standard Operating Procedure for AIBridge.",
        "author": "AIBridge Team"
    }

@mcp.prompt("prompt://prompt")
async def prompt(prompt: str, data: dict[str, Any]) -> dict[str, Any]:
    """Handle a prompt and return a canned response."""
    if not prompt:
        return {"error": "No prompt provided."}
    return {"response": await PROMPTS.get(prompt, awaitable_value("No response found."))}

@click.command()
@click.option('--help-tools', is_flag=True, help='Show detailed information about available MCP tools and exit')
@click.option('--debug', is_flag=True, help='Enable debug mode for additional logging')
@click.option('--log-file', type=click.Path(path_type=SyncPath), default=None, help='Path to log file for debug output')
def main(help_tools: bool = False,
         debug: bool = False,
         log_file: SyncPath|None = None):
    """
    Available MCP Tools:
    • get_functions_returning_nodes    - Find functions that return Houdini node objects
    • search_functions                 - Search functions by keyword in name or docstring
    • enhanced_search_functions        - Search functions with optional code examples and documentation
    • get_primitive_functions          - Find functions related to primitive operations
    • get_modules_summary              - Get summary of all Houdini modules with function counts
    • search_node_types               - Search node types by keyword
    • enhanced_search_node_types      - Search node types with optional live documentation integration
    • get_node_types_by_category      - Get node types filtered by category (Sop, Object, Dop, etc.)
    • get_database_stats              - Get statistics about the Houdini database contents
    • get_pdg_registry                - Get PDG registry entries
    • search_pdg_registry             - Search PDG registry entries by keyword
    • pdg_workflow_assistant          - Get PDG components and workflow guidance for tasks
    • web_search_houdini              - Perform web search specifically for Houdini-related content
    • fetch_houdini_docs              - Fetch official Houdini documentation for nodes or functions
    • query_response                  - Handle general queries (legacy tool)

    Hython Scene Analysis Tools (via subprocess):
    • houdini_analyze_scene           - Analyze a Houdini scene file and generate Python code
    • houdini_get_scene_info          - Get information about the current Houdini scene
    • houdini_get_node_info           - Get detailed information about a specific node
    • houdini_list_nodes              - List nodes in the scene with optional filtering
    • houdini_create_node             - Create a new node in the scene

    Database: {db.db_path if hasattr(db, 'db_path') else 'Not initialized'}

    Usage:
    This server starts an MCP (Model Context Protocol) server that provides
    AI agents with access to comprehensive Houdini Python API information.

    The server waits for MCP client connections and responds to tool requests
    with data from the Houdini modules database, enhanced with live web search
    and documentation fetching capabilities.

    Additionally, the server automatically manages a hython-based subprocess
    for direct Houdini scene analysis and manipulation within the hython environment.
    """

    def echo(*args, err: bool=True, **kwargs):
        """
        click.echo, but to stderr by default.
        """
        click.echo(*args, **kwargs, err=err)
    if help_tools:
        echo("🔧 Zabob MCP Server - Available Tools:\n")
        tools = [
            ("get_functions_returning_nodes", "Find functions that return Houdini node objects"),
            ("search_functions", "Search functions by keyword in name/docstring (requires: keyword, optional: limit)"),
            ("search_functions_by_module", "Search functions within a specific module (requires: module_name, optional: limit)"),
            ("enhanced_search_functions", "Search functions with code examples and docs (requires: keyword, optional: include_examples, limit)"),
            ("get_primitive_functions", "Find functions related to primitive operations"),
            ("get_modules_summary", "Get summary of all Houdini modules with function counts"),
            ("search_node_types", "Search node types by keyword (requires: keyword, optional: limit)"),
            ("enhanced_search_node_types", "Search node types with live documentation (requires: keyword, optional: include_docs, limit)"),
            ("get_node_types_by_category", "Get node types by category (optional: category)"),
            ("get_database_stats", "Get statistics about the Houdini database contents"),
            ("get_pdg_registry", "Get PDG registry entries (optional: registry_type)"),
            ("search_pdg_registry", "Search PDG registry entries by keyword (requires: keyword, optional: limit)"),
            ("pdg_workflow_assistant", "Get PDG components and workflow guidance (requires: workflow_description)"),
            ("web_search_houdini", "Perform web search for Houdini content (requires: query, optional: num_results)"),
            ("fetch_houdini_docs", "Fetch official Houdini documentation (requires: doc_type, optional: node_name, function_name)"),
            ("query_response", "Handle general queries (requires: query)"),
            ("", "--- Hython Scene Analysis Tools (via subprocess) ---"),
            ("houdini_analyze_scene", "Analyze a Houdini scene file and generate Python code (requires: file_path)"),
            ("houdini_get_scene_info", "Get information about the current Houdini scene"),
            ("houdini_get_node_info", "Get detailed information about a specific node (requires: node_path)"),
            ("houdini_list_nodes", "List nodes in the scene with filtering (optional: path, node_type, recursive)"),
            ("houdini_create_node", "Create a new node in the scene (requires: parent_path, node_type, optional: name)")
        ]

        for tool_name, description in tools:
            if tool_name:  # Skip separator lines
                echo(f"  {tool_name:30} - {description}")
            else:
                echo(f"\n{description}")

        echo(f"\n📊 Database: {db.db_path if hasattr(db, 'db_path') else 'Not initialized'}")
        echo("\n🚀 To start the MCP server, run without arguments.")
        echo("🔧 Hython subprocess will be automatically managed for scene analysis tools.")
        return

    config_logging(DEBUG if debug else INFO, log_file)
    if log_file:
        global HYTHON_LOG
        HYTHON_LOG = log_file.with_stem("hython_mcp")

    echo("🚀 Starting Zabob MCP Server...")
    echo(f"📊 Database: {db.db_path if hasattr(db, 'db_path') else 'Not initialized'}")
    echo("🔗 Waiting for MCP client connections...")
    mcp.run()

if __name__ == "__main__":
    main()
