'''
Common utility functions for the zabob-modules modules.
These functions are used for logging, running commands,
capturing output, and handling errors. They are designed
to be used in the context of a command line interface (CLI).
'''


from collections.abc import Sequence
from enum import StrEnum
from os import PathLike
from pathlib import Path
from shutil import which
from typing import Final, Literal
from time import sleep

from click import ParamType
from semver import Version

from zabob.core.paths import ZABOB_ROOT
from zabob.common.common_utils import DEBUG, QUIET, VERBOSE, Level


def repo_relative(f: PathLike|str) -> Path:
    '''
    Return a path relative to the repository root.
    '''
    f = ZABOB_ROOT / f
    f = f.resolve()
    f = f.relative_to(ZABOB_ROOT)
    return f

_git: Path|None = None

def find_git() -> Path:
    """
    Find the git executable in the system PATH.
    Returns:
        Path to the git executable.
    """
    global _git
    if _git is None:
        g = which('git')
        if g is None:
            raise RuntimeError("git not found in PATH.")
        _git = Path(g)
    return _git

def same_commit(files: Sequence[PathLike|str]) -> bool:
    '''
    Check if all files are at the same commit.
    Args:
        files (Sequence[PathLike|str]): A sequence of file paths.
    Returns:
        bool: True if all files are at the same commit, False otherwise.
    '''

    from zabob.common.subproc import capture
    git = find_git()
    matches  = capture(git, 'log', '--name-only', '-n', 1, '--format=', '--',
                       *(repo_relative(f) for f in files)).strip()
    return len(files) == matches.count('\n') + 1

def is_clean(file: PathLike|str) -> bool:
    '''
    Check if the file is clean (not modified).
    Args:
        file (PathLike|str): The file path.
    Returns:
        bool: True if the file is clean, False otherwise.
    '''

    from zabob.common.subproc import capture
    git = find_git()
    matches  = capture(git, 'status', '--porcelain', '--', repo_relative(file))
    return not matches.strip()

def needs_update(*files: PathLike|str) -> bool:
    '''
    Check if any of the files need to be updated.
    Args:
        files PathLike|str]: 2 or more file paths.
    Returns:
        bool: True if any of the files need to be updated, False otherwise.
    '''
    if len(files) < 2:
        DEBUG("Nothing to update, only one file.")
        # Nothing to update.
        return False
    # Check 1, if all the files exist.
    if not all((ZABOB_ROOT / f).exists() for f in files):
        DEBUG("Not all files exist, rebuilding")
        return True
    # Check 2, if all the files are at the same commit.
    if not same_commit(files):
        DEBUG("Files are not at the same commit, rebuilding")
        return True
    # Check 3, if all the files are clean.
    if not all(is_clean(f) for f in files):
        DEBUG("Files are not clean, rebuilding")
        return True
    DEBUG("Files are clean, no need to update.")
    return False


def flatten_tree(*dirs: Path):
    """
    Depth-first flattening of the directory tree.

    The only ordering is that children are yielded before parents, and that
    directories given as arguments are processed in the order given.

    Args:
        dirs (Path): The directories to flatten.
    Yields:
        paths to all files and directories in the trees.
    """
    for dir in dirs:
        if not dir.is_dir():
            yield dir
        else:
            for f in dir.iterdir():
                if f.is_dir():
                    yield from flatten_tree(f)
                yield f
            yield dir


def rmdir(*dirs: Path,
          level: Level=VERBOSE,
          dry_run: bool=False,
          retries: int=3) -> None:
    "Remove the given directories (or files) and contents."
    while retries > 0:
        failures: bool = False
        for f in flatten_tree(*dirs):
            try:
                level(f"Removing {f}")
                if not dry_run:
                    if f.is_symlink():
                        f.unlink(missing_ok=True)
                    elif f.is_dir():
                        f.rmdir()
                    else:
                        f.unlink(missing_ok=True)
            except OSError as e:
                failures = True
                QUIET(f"Failed to remove {f}: {e}")
        if not failures:
            break
        retries -= 1
        sleep(0.5)

