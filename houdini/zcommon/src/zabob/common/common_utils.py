'''
Common Utilities.
'''

from collections import defaultdict
from collections.abc import Callable, Generator, Iterable, MutableMapping
from functools import wraps
from logging import Logger
import logging
import os
from contextlib import contextmanager, suppress
from enum import Enum, StrEnum
from pathlib import Path
import sys
from typing import IO, Hashable, Literal, ParamSpec, TypeAlias, TypeVar, TypeVarTuple, Any
import atexit
from weakref import WeakKeyDictionary

from semver import Version


def _version(version: Version|str) -> Version:
    """
    Convert a version to a semver.Version object.

    Args:
        version (Version|str): The version to convert.

    Returns:
        Version: The converted version.
    """
    if isinstance(version, Version):
        return version
    return Version.parse(version, optional_minor_and_patch=True)


class Level(StrEnum):
    """
    Enum for logging levels. Each level acts both as a string and as a callable.

    Calling `INFO("message")` will print the message if the current logging level
    is `INFO` or more verbose. The levels are ordered from most verbose to least
    verbose. The default level is `INFO`. `DEBUG` will print all messages, while
    `SILENT` will suppress all output except for explicit calls to `SILENT`.
    """
    DEBUG = "DEBUG"
    VERBOSE = "VERBOSE"
    INFO = "INFO"
    QUIET = "QUIET"
    SILENT = "SILENT"

    level: 'Level'
    logger: Logger|None

    @property
    def enabled(self) -> bool:
        """
        Check if the current logging level is enabled.
        """
        return LEVELS.index(self) >= LEVELS.index(Level.level)

    def __bool__(self) -> bool:
        """
        Check if the logging level is enabled.
        This allows the level to be used in boolean contexts.
        """
        return self.enabled

    def __call__(self, message: str) -> None:
        """
        Output a message at the specified logging level.
        """
        if self.enabled:
            # Only print the message if the current logging level is
            # at least as verbose as the message level.
            if self.logger is None:
                print(message)
            else:
                # If a logger is set, use it to log the message.
                match self:
                    case Level.DEBUG:
                        self.logger.debug(message)
                    case Level.VERBOSE:
                        self.logger.info(message)
                    case Level.INFO:
                        self.logger.info(message)
                    case Level.QUIET:
                        self.logger.warning(message)
                    case Level.SILENT:
                        self.logger.error(message)


Level.level = Level.INFO
Level.logger = None

LEVELS: tuple[Level, ...] = tuple(Level.__members__.values())
'''
Ordered list of logging levels, fro the most verbose to the least verbose.
'''

DEBUG: Literal[Level.DEBUG]= Level.DEBUG
VERBOSE: Literal[Level.VERBOSE] = Level.VERBOSE
INFO: Literal[Level.INFO] = Level.INFO
QUIET: Literal[Level.QUIET] = Level.QUIET
SILENT: Literal[Level.SILENT] = Level.SILENT


def config_logging(level: Level|str|None=None,
                     log_file: Path|None=None,
                     ) -> None:
    """
    Configure the logging level and logger.

    Args:
        level (Level|str|None): The logging level to set. If `None`, the current level is used.
        logger (Logger|None): The logger to use. If `None`, the default logger is used.
    """
    if isinstance(level, str):
        level = Level(level)
    if level is None:
        level = Level.level
    Level.level = level
    if level in (Level.DEBUG, Level.VERBOSE):
        print(f"Setting output level to {level.value}")
    if log_file:
        print (f"Logging to file: {log_file}")
        log_file.resolve().parent.mkdir(parents=True, exist_ok=True)
        match level:
            case Level.DEBUG:
                log_level = logging.DEBUG
            case Level.VERBOSE:
                log_level = logging.INFO
            case Level.INFO:
                log_level = logging.INFO
            case Level.QUIET:
                log_level = logging.WARNING
            case Level.SILENT:
                log_level = logging.ERROR

        # If a log file is specified, configure the logger to write to it.
        logging.basicConfig(
            handlers=[
                logging.FileHandler(log_file, mode='a', encoding='utf-8')
                ],
            force=True,
            level=level.value,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        Level.logger = logging.getLogger()

@contextmanager
def environment(*remove, **update):
    """
    Temporarily updates the ``os.environ`` dictionary in-place.

    The ``os.environ`` dictionary is updated in-place so that the modification
    is sure to work in all situations.

    :param remove: Environment variables to remove.
    :param update: Dictionary of environment variables and values to add/update.
    """
    env = os.environ
    update = update or {}
    remove = remove or []

    # List of environment variables being updated or removed.
    stomped = (set(update.keys()) | set(remove)) & set(env.keys())
    # Environment variables and values to restore on exit.
    update_after = {k: env[k] for k in stomped}
    # Environment variables and values to remove on exit.
    remove_after = frozenset(k for k in update if k not in env)

    try:
        env.update(update)
        for k in remove:
            with suppress(KeyError):
                del env[k]
        yield
    finally:
        env.update(update_after)
        for k in remove_after:
            with suppress(KeyError):
                del env[k]


@contextmanager
def prevent_exit():
    """Context manager that prevents sys.exit() from terminating the process."""
    original_exit = sys.exit

    def exit_handler(code=0):
        # Instead of exiting, raise a custom exception
        raise RuntimeError(f"Module attempted to exit with code {code}")

    # Replace sys.exit temporarily
    sys.exit = exit_handler
    try:
        yield
    finally:
        # Always restore the original exit function
        sys.exit = original_exit


@contextmanager
def prevent_atexit():
    """
    Temporarily prevent modules from registering atexit handlers.
    """
    original_register = atexit.register
    original_unregister = atexit.unregister
    captured_handlers = []

    def fake_register(func, *args, **kwargs):
        """
        Capture atexit handlers instead of registering them.
        """
        print(f"Prevented atexit registration: {func.__module__}.{func.__name__}")
        captured_handlers.append((func, args, kwargs))
        return func

    def fake_unregister(func):
        """
        Do nothing when unregistering.
        """
        pass

    # Replace the atexit functions
    atexit.register = fake_register
    atexit.unregister = fake_unregister

    try:
        yield
    finally:
        # Restore original functions
        atexit.register = original_register
        atexit.unregister = original_unregister


T = TypeVar('T')
R = TypeVar('R')
def none_or(value: T|None, fn: Callable[[T], R]) -> R|None:
    """
    Call a function with the given value if it is not `None`,
    otherwise return `None`.

    Args:
        value (T|None): The value to pass to the function.
        Callable[T]: The function to call with the value.

    Returns:
        R|None: The result of the function call or None if value is None.
    """
    if value is None:
        return None
    return fn(value)


def not_none(*values: T|None) -> Generator[T, None, None]:
    '''
    Generator that yields the values if they are not `None`.
    Args:
        *values (T|None): The values to yield.
    Yields:
        T: All non-`None` values from the arguments.
    '''
    for v in values:
        if v is not None:
            yield v


def not_none1(value: T|None) -> Generator[T, None, None]:
    '''
    Generator that yields the single value if it is not `None`.

    Args:
        value (T|None): The value to yield.
    Yields:
        T: The value if it is not `None`.
    '''
    if value is not None:
        yield value


def not_none2(value1: T|None, value2: T|None) -> Generator[T, None, None]:
    '''
    Generator that yields the individual values if it they are not `None`.

    Args:
        value1 (T|None): The first value to yield.
        value2 (T|None): The second value to yield.
    Yields:
        T: The first or second value if it is not `None`.
    '''
    if value1 is not None:
        yield value1
    if value2 is not None:
        yield value2


def value(arg: T) -> Generator[T, None, None]:
    """
    Generator that yields the given argument.
    A substitute for a tuple that can be more clear about intent.

    Useful in generators to yield an intermediate value.

    Example:
        >>> def trouble():
        >>> ...     return [
        >>>     (f'{ref}-bug', f'{ref}-{fix}')
        >>>     for id in range(10)
        >>>     for repo in ('repo1', 'repo2')
        >>>     for ref in value(f'{repo}-{id}')
        >>> ]

    Args:
        arg (T|None): The value to yield.

    Yields:
        T: The value from the argument.
    """
    yield arg


def values(*args: T) -> Generator[T, None, None]:
    """
    Generator that yields all the values from the given arguments.
    A substitute for a tuple that can be more clear about intent.

    Args:
        *args (T|None): The values to yield.

    Yields:
        T: All non-`None` values from the arguments.
    """
    for value in args:
        yield value

def if_true(condition: bool, value: T) -> Generator[T, None, None]:
    """
    Yield the value if the condition is `True`, otherwise return `None`.

    Args:
        condition (bool): The condition to check.
        value (T): The value to return if the condition is `True`.

    Yields:
        T: The value if the condition is `True`.
    """
    if condition:
        yield value


def if_false(condition: bool, value: T) -> Generator[T, None, None]:
    """
    Yield the value if the condition is `False`, yield nothing.
        condition (bool): The condition to check.
        value (T): The value to return if the condition is `False`.

    Yields:
        T: The value if the condition is `False`.
    """
    if not condition:
        yield value

_name_counter: MutableMapping[str, int] = defaultdict[str, int](lambda: 0)
_names: MutableMapping[Any, str] = WeakKeyDictionary[Any, str]()
def get_name(d: Any) -> str:
    '''
    Get the name of the given object. If it does not have a name,
    one will be assigned.

    If the object has a `name` or `__name__` attribute, it will
    be used as the name.

    If the object has a method called `name`, `getName`,
    or `get_name`, it will be called to try to get the name.

    Args:
        d (Any): The object to get the name of.
    Returns:
        str: The name of the object.
    '''
    match d:
        case Enum():
            return str(d.name)
        case str() | int() | float() | complex() | bool():
            return str(d)
        case None:
            return "None"
        case Exception():
            # If the object is an Exception, return its class name.
            return d.__class__.__name__
        case _ if hasattr(d, '__name__'):
            return str(d.__name__)
        case _ if hasattr(d, 'name') and isinstance(d.name, str):
            # If the object has a name attribute that is a string, return it.
            return str(d.name)
        case _ if hasattr(d, 'name') and callable(d.name):
            try:
                return str(d.name())
            except Exception:
                pass
        case _ if hasattr(d, 'get_name') and callable(d.get_name):
            try:
                return str(d.get_name())
            except Exception:
                pass
        case _ if hasattr(d, 'getName') and callable(d.getName):
            try:
                return str(d.get_name())
            except Exception:
                pass
        case _ if hasattr(d, 'name'):
            return str(d.name)
        case d if isinstance(d, Hashable):
            n = _names.get(d, None)
            if n is not None:
                return n
            pass
        case _:
            pass
    # If we reach here, we don't have a name, so generate one.
    typename = get_name(type(d))
    _name_counter[typename] += 1
    c = _name_counter[typename]
    n = f"{typename}_{c}"
    try:
        # If the object has a __name__ attribute, set it to the generated name.
        # This is useful for debugging and logging.
        setattr(d, '__name__', n)
        return n
    except AttributeError:
        match d:
            case _ if isinstance(d, Hashable):
                # If the object is hashable, store the name in a weak dictionary.
                try:
                    _names[d] = n
                    return n
                except TypeError:
                    pass
        # If we can't save the name, generate one based on the id.
        return f"{typename}_{id(d):x}"

Condition: TypeAlias = Callable[[T], bool]|bool|None
'''
A condition function that takes an item of type `T` and returns a boolean.
Or `True`, indicating all items meet the condition.
Or `False`, indicating no items meet the condition.
Or `None`, indicating all non-`None` items meet the condition, or if negated,
all `None` items meet the condition.

'''


def trace(v: T,
          label: str|None=None,
          file:IO[str]=sys.stderr,
          condition: Condition=False,
          ) -> T:
    '''
    Like `print`, but returns the value.
    This is useful for debugging, as it allows you to see the value
    while still returning it for further processing.

    Args:
        v (T): The value to trace.
        label (str|None): An optional label to print before the value.
        file (IO[str]): The file to print the trace to (default: sys.stderr).
        condition (Callable[[T], bool]|None): An optional condition to check
            before printing the value.
            If `True`, all values are printed.
            If `False`, no values are printed.
            If `None`, all non-None values are printed.
    Returns:
        T: The value that was traced.
    '''
    if (
        (condition is True)
        or (condition is None and v is not None)
        or (callable(condition) and condition(v))
        ):
        if label is not None:
            print(f"{label}: {v}", file=file)
        else:
            print(v, file=file)
    return v

_trace = trace


def trace_(i: Iterable[T],
           label: str|None=None,
           file:IO[str] = sys.stderr,
           condition: Condition=False,
           ) -> Iterable[T]:
    """
    Iterate over the items in the iterable and trace each item.
    This is useful for debugging, as it allows you to see the items
    while still returning them for further processing.

    Args:
        i (Iterable[T]): The iterable to iterate over.
        label (str|None): An optional label to print before each item.
        file (IO[str]): The file to print the trace to (default: `sys.stderr`).
        condition (Callable[[T], bool]|None): An optional condition to check
            before printing each item. If `None`, all items are printed.
    Yields:
        T: Each item in the iterable after tracing it.

    Returns:
        T: the returned items:
        or R (the default value) if no item meets the condition.
    """
    yield from (trace(item, label, file, condition) for item in i)


def do_all(i: Iterable,
           trace: Condition=False,
           label: str|None=None,
           file: IO[str] = sys.stderr,
          ) -> None:
    """
    Iterate over the items in the `Iterable` (usually a `Generator`)
    and do nothing with them.

    This is useful for iterating over an `Iterable` when you don't care about the
    items, but you want to ensure that the `Iterable` is fully consumed.

    Args:
        i (Iterable): The `Iterable` to iterate over.
        trace (Callable[[T], bool]|bool|None): An optional `Condition` to check
            before tracing each item.
            If `None`, all items are traced.
            If `True`, all items are traced.
            If `False`, no items are traced.
        label (str|None): An optional label to print with the trace.
        file (IO[str]): The file to print the trace to (default: `sys.stderr`).
    """
    for item in i:
        _trace(item, condition=trace, label=label, file=file)
    # This is a no-op, but it ensures that the iterable is fully consumed.


def do_until(i: Iterable[T],
             condition: Condition=None,
             default: R=None,
             trace: Condition=False,
             label: str|None=None,
             file: IO[str] = sys.stderr,
             ) -> T|R:
    """
    Iterate over the items in the `Iterable` until the condition is met.

    If no item meets the condition, return the default value (default=`None`).

    `do_until` and `find_first` are the same function, but `do_until`
    emphasizes its use to perform actions via a generator.

    Args:
        i (Iterable[T]): The `Iterable` to iterate over.
        condition (Callable[[T], bool]|bool|None): The `Condition` to check for each item.
            If `None`, all non-None items are considered to meet the condition.
            If `True`, all items are considered to meet the condition.
            If `False`, no items are considered to meet the condition.
        default (R): The default value to return if no item meets the condition.

    Returns:
        T: The first item that meets the condition,
        or R (the default value) if no item meets the condition.
    """
    match condition:
        case True:
            # If the condition is True, we want to return the first item.
            return next(iter(trace_(i, condition=trace, label=label, file=file)),
                        default)

        case False:
            do_all(i, trace=trace)
            return default
        case None:
            return next((_trace(item, condition=trace, label=label, file=file)
                        for item in i
                        if item is not None),
                    default)
        case _ if callable(condition):
            return next((_trace(item, condition=trace, label=label, file=file)
                         for item in i
                         if condition(item)),
                     default)
        case _:
            raise TypeError(
                f"Condition must be a callable, bool, or None, not {type(condition)}"
            )

find_first = do_until

def do_while(i: Iterable[T],
             condition: Condition=None,
             default: R=None,
             trace: Condition=False,
             label: str|None=None,
             file: IO[str] = sys.stderr,
             ) -> T|R:
    """
    Iterate over the items in the `Iterable` until the condition is not met.
    If all items meet the condition, return the default value (default=`None`).

    `do_while` and `find_first_not` are the same function, but `do_until`
    emphasizes its use to perform actions via a generator.

    Args:
        i (Iterable[T]): The `Iterable` to iterate over.
        condition (Callable[[T], bool]|bool|None): The `Condition` to check for each item.
            If `None`, all None items are considered to meet the condition.
            If `True`, all items are considered to meet the condition.
            If `False`, no items are considered to meet the condition.
        default (R): The default value to return if all items meet the condition.
    Returns:
        T: The first item that does not the condition,
        or R (the default value) if all items meet the condition.
    """
    match condition:
        case True:
            do_all(i, trace=trace, label=label, file=file)
            return default
        case False:
            return default
        case None:
            return next((_trace(item, condition=trace, label=label, file=file)
                         for item in i
                         if item is None),
                        default)
        case _ if callable(condition):
            # If the condition is a callable, we want to find the first item
            # that does not meet the condition.
            return next((_trace(item, condition=trace, label=label, file=file)
                        for item in i
                        if not condition(item)),
                    default)
        case _:
            raise TypeError(
                f"Condition must be a callable, bool, or None, not {type(condition)}"
            )

find_first_not = do_while


Ps = TypeVarTuple('Ps')
Pspec = ParamSpec('Pspec')

def do_yield(fn: Callable[[T, *Ps], None],
             val: T,
             *args: *Ps,
             **kwargs: dict[str, Any] # Can't type this in 3.11 even with a Protocol
             ) -> Generator[T, None, None]:
    """
    Call a function with the given value and yield the value.

    See also:
        `do_yielder` for a version returns a function that takes a value and yields it after calling the fn.
        `call_yield` for a version that calls the fn and yields the result.
        `call_yielder` for a version returns a function that calls the fn and yields the result.

    Args:
        fn (Callable[[T], R]): The function to call with the value.
        val (T): The value to pass to the function.
        *args (Ps): Additional positional arguments to pass to the function.
        **kwargs: Additional keyword arguments to pass to the function.

    Yields:
        T: The value that was passed to the function.
    """
    fn(val, *args, **kwargs)
    yield val

def do_yielder(fn: Callable[[T, *Ps], None],
               *args: *Ps,
               **kwargs: dict[str, Any] # Can't type this in 3.11 even with a Protocol
               ) -> Callable[[T], Generator[T, None, None]]:
    """
    Returns a function that takes the given value and yield the value.

    See also:
        `do_yield` for a version that takes a value and yields it after calling the fn.
        `call_yield` for a version that calls the fn and yields the result.
        `call_yielder` for a version returns a function that calls the fn and yields the result.

    Args:
        fn (Callable[[T], R]): The function to call with the value.
        val (T): The value to pass to the function.
        *args (Ps): Additional positional arguments to pass to the function.
        **kwargs: Additional keyword arguments to pass to the function.

    Yields:
        T: The value that was passed to the function.
    """
    @wraps(fn)
    def _do_yield(val: T) -> Generator[T, None, None]:
        fn(val, *args, **kwargs)
        yield val
    return _do_yield


def call_yield(fn: Callable[Pspec, R],
               *args: Pspec.args,
               **kwargs: Pspec.kwargs,
               ) -> Generator[R, None, None]:
    """
    Call a function with the given arguments and yield the result.

    See also:
        `do_yield` for a version that takes a value and yields it after calling the fn.
        `do_yielder` for a version returns a function that takes a value and yields it after calling the fn.
        `call_yielder` for a version returns a function that calls the fn and yields the result.

    Args:
        fn (Callable[P, R]): The function to call.
        *args (P.args): The arguments to pass to the function.
        **kwargs (P.kwargs): The keyword arguments to pass to the function.

    Yields:
        R: The result of the function call.
    """
    yield fn(*args, **kwargs)


def call_yielder(fn: Callable[[T, *Ps], R],
                *args: *Ps,
                **kwargs: dict[str, Any], # Can't type this in 3..11 even with a Protocol
                ) -> Callable[[T], Generator[R, None, None]]:
    """
    Call a function with the given arguments and yield the result.

    See also:
        `do_yield` for a version that takes a value and yields it after calling the fn.
        `do_yielder` for a version returns a function that takes a value and yields it after calling the fn.
        `call_yield` for a version that calls the fn and yields the result.

    Args:
        fn (Callable[P, R]): The function to call.
        *args (P.args): The arguments to pass to the function.
        **kwargs (P.kwargs): The keyword arguments to pass to the function.

    Yields:
        R: The result of the function call.
    """
    @wraps(fn)
    def _call_yield(val: T) -> Generator[R, None, None]:
        """
        Call the function with the given arguments and yield the result.
        """
        yield fn(val, *args, **kwargs)
    return _call_yield

def last(i: Iterable[T],
          condition: Condition=None,
          default: R=None,
          trace: Condition=False,
          label: str|None=None,
          file: IO[str] = sys.stderr,
          ) -> T|R:
    """
    Get the last item in the `Iterable` that meets the `Condition`.

    If no item meets the `Condition`, return the default value (default=`None`).

    Unlike many implementations, this tests the entire iterable, ensuring
    that any generator is fully consumed before returning a value.
    """
    val: T = last # # type: ignore
    for item in i:
        if (
            (condition is True)
            or (condition is None and item is not None)
            or (callable(condition) and condition(item))
        ):
            val = _trace(item, condition=trace, label=label, file=file)
    return val if val is not last else default
