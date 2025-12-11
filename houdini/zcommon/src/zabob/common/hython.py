#!/usr/bin/env uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "click",
#     "psutil",
#     "semver",
# ]
# ///
'''
Hython invoker
'''

import os
from pathlib import Path
from collections.abc import Sequence
from subprocess import DEVNULL
import sys
from typing import Literal, overload

import click
from semver import Version

try:
    from zabob.common import (
        environment,
        ZABOB_ZCOMMON_DIR, ZABOB_HOUDINI_DIR,
        ZABOB_PYCACHE_DIR, ZABOB_ROOT,  get_houdini,
        run, capture, exec_cmd, CompletedProcess, OptionalType, SemVerParamType,
    )
except ImportError:
    script = Path(__file__).resolve()
    src = script.parent.parent.parent
    sys.path.insert(0, str(src))
    from zabob.common import (
        environment,
        ZABOB_ZCOMMON_DIR, ZABOB_HOUDINI_DIR,
        ZABOB_PYCACHE_DIR, ZABOB_ROOT, get_houdini,
        run, capture, exec_cmd, CompletedProcess, OptionalType, SemVerParamType,
    )

@overload
def run_houdini_script(script_path: Path|str|None=None,
                      *args: Path|str,
                      module: None=None,
                      version: Version|None=None,
                      capture_output: Literal[False]=False,
                      exec: bool = False,
                      env_vars: dict[str, str]|None = None,
                      **kwargs) -> CompletedProcess: ...
@overload
def run_houdini_script(script_path: Path|str|None=None,
                      *args: Path|str,
                      module: None=None,
                      version: Version|None=None,
                      capture_output: Literal[True],
                      env_vars: dict[str, str]|None = None,
                      **kwargs) -> str: ...
@overload
def run_houdini_script(*args: Path|str,
                       module: str,
                      version: Version|None=None,
                      capture_output: Literal[False]=False,
                      exec: bool = False,
                      env_vars: dict[str, str]|None = None,
                      **kwargs) -> CompletedProcess: ...
@overload
def run_houdini_script(*args: Path|str,
                      module: str,
                      version: Version|None=None,
                      capture_output: Literal[True],
                      env_vars: dict[str, str]|None = None,
                      **kwargs) -> str: ...
@overload
def run_houdini_script(*args: Path|str,
                      module: str|None,
                      version: Version|None=None,
                      capture_output: Literal[True],
                      env_vars: dict[str, str]|None = None,
                      **kwargs) -> str: ...
@overload
def run_houdini_script(script_path: Path|str|None,
                       *args: Path|str,
                      module: str|None,
                      version: Version|None=None,
                      capture_output: Literal[False],
                      env_vars: dict[str, str]|None = None,
                      **kwargs) -> CompletedProcess: ...
def run_houdini_script(script_path: Path|str|None=None,
                      *args: Path|str,
                      module: str|None = None,
                      version: Version|None=None,
                      capture_output: bool = False,
                      exec: bool = False,
                      env_vars: dict[str, str]|None = None,
                      **kwargs) -> CompletedProcess|str:
    """
    Run a script with a specific Houdini version.

    With no positional arguments, enters the interactive REPL.

    Args:
        script_path: Path to the script to run
        *args: Additional arguments to pass to the script
        module: Optional module to run instead of a script
        version: Houdini version to use
        env_vars: Additional environment variables
        capture_output: Whether to capture stdout/stderr
        exec: If True, execute the script via execve. Incompatible with `capture_output`.
        **kwargs: Additional arguments for run/capture
    """
    match script_path, module:
        case None, None:
            script = ()
        case None, str():
            script = ('-m', module)
        case _, None:
            script = (script_path, )
        case _, _:
            script = ('-m', module, script_path)

    match capture_output, exec:
        case True, True:
            raise ValueError("Cannot use capture_output with exec=True.")

    # Get Houdini installation
    houdini = get_houdini(version)

    # Build paths for the environment
    major_minor = f"{houdini.houdini_version.major}.{houdini.houdini_version.minor}"
    paths = [ZABOB_ZCOMMON_DIR / "src", ZABOB_ROOT / "zabob-modules" / "src"]
    version_path = ZABOB_HOUDINI_DIR / f"h{major_minor}" / "src"
    if version_path.exists():
        paths.append(version_path)

    # Add hython world virtual environment site-packages for FastMCP access
    # Both h20.5 and zcommon venvs contain packages needed for the hython MCP server
    h20_5_venv_site_packages = ZABOB_HOUDINI_DIR / f"h{major_minor}" / ".venv" / "lib" / "python3.11" / "site-packages"
    if h20_5_venv_site_packages.exists():
        paths.append(h20_5_venv_site_packages)

    zcommon_venv_site_packages = ZABOB_ZCOMMON_DIR / ".venv" / "lib" / "python3.11" / "site-packages"
    if zcommon_venv_site_packages.exists():
        paths.append(zcommon_venv_site_packages)

    # Setup bytecode cache directory
    pycache_dir = ZABOB_PYCACHE_DIR / f"houdini_{houdini.houdini_version}"
    pycache_dir.mkdir(parents=True, exist_ok=True)

    # Use the environment context manager with direct keyword arguments
    with environment(PYTHONPATH=os.pathsep.join(str(p) for p in paths),
                    PYTHONPYCACHEPREFIX=str(pycache_dir),
                    **(env_vars or {})):
        try:
            if exec:
                return exec_cmd(houdini.hython, *script, *args,
                                **kwargs)
            if capture_output:
                return capture(houdini.hython, *script, *args,
                               **kwargs)
            else:
                return run(houdini.hython, *script, *args,
                           **kwargs)
        except RuntimeError as ex:
            print(ex, file=sys.stderr)
            sys.exit(1)

# We disable --help here so that invoked commands can handle it.
# But if no arguments are given, it will show the help message.
@click.command(
    name='hython',
    help='Run hython with the given arguments.',
    add_help_option=False,
    no_args_is_help=True,
    context_settings=dict(
        ignore_unknown_options=True,
    )
)
@click.argument(
    'arguments',
    nargs=-1,
    type=str,
)
@click.option(
    '--houdini-version',
    type=OptionalType(SemVerParamType(min_parts=2)),
    default=None,
    help='Houdini version to use (e.g., "20.5" or "20.5.584"). If not specified, the latest version will be used.'
)
@click.option(
    '--module', '-m',
    type=OptionalType(str),
    default=None,
)
def hython(arguments: Sequence[str],
           houdini_version: Version|None=None,
           module: str|None = None) -> None:
    """
    Run hython with the given arguments.

    ARGUMENTS:
        SCRIPT_PATH <arguments
        -m MODULE <arguments>

    If `-m MODULE` is given, the positional arguments are passed to the
    module.
    """
    import sys

    # Check if hython is installed
    houdini = get_houdini(houdini_version)
    if houdini is None:
        print("Houdini is not installed or not found.")
        sys.exit(1)
    hython_path = houdini.hython
    try:
        run(hython_path, '--version', stdout=DEVNULL, stderr=DEVNULL)
    except FileNotFoundError:
        print("Hython is not installed. Please install it first.")
        sys.exit(1)

    script_path: Path|None = None
    if not module and len(arguments) > 0:
        # If no module is specified, treat the first argument as the script path
        script_path = Path(arguments[0])
        arguments = arguments[1:]
        if not script_path.exists():
            print(f"Script path '{script_path}' does not exist.")
            sys.exit(1)

    run_houdini_script(
        script_path,
        *arguments,
        module=module,
        version=houdini_version,
        capture_output=False,
    )

if __name__ == '__main__':
    hython()
