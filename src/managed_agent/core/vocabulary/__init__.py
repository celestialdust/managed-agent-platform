"""The closed, published set of tenant-facing event types.

Closed means a runtime event with no mapping never reaches a tenant as itself
(ADR-013), and the set a given API version can emit is fixed for that version. It
does not mean the set lives in one file: each family declares its own module and
registers into this registry at import, and this module discovers those modules
rather than importing them by name. Adding a family is therefore a new file, never
an edit to a switch statement here — which is also what keeps two slices from
writing one file.

The registry is frozen after discovery so a late import cannot widen the published set
behind a caller's back.
"""

import pkgutil
from importlib import import_module
from types import MappingProxyType
from typing import Final

_types: dict[str, str] = {}
_sealed = False


def declare(type_name: str, family: str) -> str:
    """Register one tenant-facing event type. Returns it, so a module can assign it."""
    if _sealed:
        raise RuntimeError(f"vocabulary sealed; {type_name} declared too late")
    if type_name in _types:
        raise ValueError(f"duplicate event type {type_name}")
    _types[type_name] = family
    return type_name


def _discover() -> None:
    global _sealed
    for module in pkgutil.iter_modules(__path__):
        import_module(f"{__name__}.{module.name}")
    _sealed = True


_discover()

PUBLISHED: Final[MappingProxyType[str, str]] = MappingProxyType(_types)
"""Every event type this API version may emit, mapped to its family."""


def is_published(type_name: str) -> bool:
    return type_name in PUBLISHED
