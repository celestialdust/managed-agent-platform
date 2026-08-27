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
_webhook_eligible: set[str] = set()
_sealed = False


def declare(type_name: str, family: str, *, webhook: bool = False) -> str:
    """Register one tenant-facing event type. Returns it, so a module can assign it.

    `webhook` marks the type as one a tenant may point a callback registration at. It
    is recorded here rather than listed by whatever delivers callbacks, so a family that
    gains a type decides the question at the declaration instead of leaving a second
    place to be edited and forgotten.

    It defaults to False because the two mistakes do not cost the same. An eligible type
    left unmarked is a subscription a tenant cannot make, and the refusal names it. An
    ineligible type left marked puts a per-token event through a delivery ledger and
    onto somebody's endpoint, which is discovered by the receiver.
    """
    if _sealed:
        raise RuntimeError(f"vocabulary sealed; {type_name} declared too late")
    if type_name in _types:
        raise ValueError(f"duplicate event type {type_name}")
    _types[type_name] = family
    if webhook:
        _webhook_eligible.add(type_name)
    return type_name


def _discover() -> None:
    global _sealed
    for module in pkgutil.iter_modules(__path__):
        import_module(f"{__name__}.{module.name}")
    _sealed = True


_discover()

PUBLISHED: Final[MappingProxyType[str, str]] = MappingProxyType(_types)
"""Every event type this API version may emit, mapped to its family."""

WEBHOOK_ELIGIBLE: Final[frozenset[str]] = frozenset(_webhook_eligible)
"""Every event type a tenant may register a callback for.

Deliberately narrower than `PUBLISHED`. The delivery tail scans a window of the log and
filters by type, so admitting a per-token type here would put one row per token through
a ledger; and a callback carries a sequence rather than content, so a type whose value
is the content it names is answered better by reading the log at that sequence.
"""


def is_published(type_name: str) -> bool:
    return type_name in PUBLISHED
