import functools
import os
import warnings
from typing import Callable, Any, Type, Union, TypeVar

# Type hint helpers for functions and classes
F = TypeVar('F', bound=Callable[..., Any])
C = TypeVar('C', bound=Type[Any])


def _marker(kind: str, category: Type[Warning], env_var: str, default_reason: str
           ) -> Callable[..., Callable[[Union[F, C]], Union[F, C]]]:
    """Build a decorator factory like `experimental`/`deprecated` below.

    stdlib-only (functools/os/warnings), and does NOT use
    `warnings.deprecated` -- that is Python 3.13+ only (PEP 702), and using it
    here would make `import raim` fail outright on any earlier interpreter.
    This hand-rolled version works identically back to any Python 3 this
    project otherwise supports.
    """
    def factory(reason: str = default_reason) -> Callable[[Union[F, C]], Union[F, C]]:
        def decorator(target: Union[F, C]) -> Union[F, C]:
            # Check environment variable dynamically at execution time
            if os.environ.get(env_var) == "1":
                return target

            # Handle class decoration
            if isinstance(target, type):
                original_init = target.__init__

                @functools.wraps(original_init)
                def wrapped_init(self: Any, *args: Any, **kwargs: Any) -> None:
                    warnings.warn(
                        f"Class '{target.__name__}' is {kind}. {reason}",
                        category=category,
                        stacklevel=2
                    )
                    original_init(self, *args, **kwargs)

                target.__init__ = wrapped_init  # type: ignore[assignment]
                return target

            # Handle function/method decoration
            elif callable(target):
                @functools.wraps(target)
                def wrapper(*args: Any, **kwargs: Any) -> Any:
                    warnings.warn(
                        f"Call to {kind} function '{target.__name__}'. {reason}",
                        category=category,
                        stacklevel=2
                    )
                    return target(*args, **kwargs)
                return wrapper  # type: ignore[return-value]

            return target
        return decorator
    return factory


experimental = _marker(
    "experimental", FutureWarning, "DISABLE_EXPERIMENTAL_WARNINGS",
    "This component is under active development and may change.")
"""Marks functions, methods, or classes that are untested, or carry a known
caveat the reason names. Silenced globally by DISABLE_EXPERIMENTAL_WARNINGS=1."""

deprecated = _marker(
    "deprecated", DeprecationWarning, "DISABLE_DEPRECATED_WARNINGS",
    "This component is retired and kept only for reference or as a test target.")
"""Marks functions, methods, or classes as superseded but intentionally kept
(e.g. a retired algorithm a regression test still calls directly). Silenced
globally by DISABLE_DEPRECATED_WARNINGS=1."""
