"""The argument schema a registered tool is advertised under, minus what Scope fills.

A Scope Binding names an argument the platform writes on every outbound call, so the
model has no say in it and no use for it. Offering it anyway costs twice: the model
spends a field it must guess at, and a model that guesses is refused rather than
answered, because a supplied bound argument is a protocol violation on the call path.
Removing the field ahead of time is what keeps that refusal off well-behaved callers.

**Presentation, never enforcement.** Nothing removed here can stop a caller supplying
the argument -- a schema of `{"type": "object", "additionalProperties": true}` has no
property to remove and forbids nothing, and a caller below the advertised layer never
read the schema at all. The boundary is `scope_clamp.narrow`, which refuses the call.
This module only means an honest model never trips it (ADR-034).

**The upstream's schema is somebody else's, and it moves.** A registration is written
once; the server behind it keeps shipping. The argument a binding names can stop being
declared, stop being required, or arrive under a shape no JSON Schema draft would
accept -- and all of that lands here, at listing time, inside a loop where a raised
exception is recorded as a server that could not be reached and takes every other tool
on that server down with it. So a schema this module cannot read is advertised exactly
as it came, and a listing never fails over one.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from managed_agent.core.registration.scope_binding import ScopeBinding


def _is_readable(schema: dict[str, Any]) -> bool:
    """Whether the two keywords this module edits are the shapes JSON Schema gives them.

    Read as a pair rather than one at a time, because the removal is only correct when
    it happens in both places. A schema whose `properties` is a mapping and whose
    `required` is some other shape would otherwise have the argument taken out of the
    declarations and left in the requirements -- a tool that is advertised and cannot
    be called, which is a worse answer than a tool whose bound argument is still shown.
    """
    return all(
        not (keyword in schema and not isinstance(schema[keyword], expected))
        for keyword, expected in (("properties", dict), ("required", list))
    )


def without_bound_arguments(
    schema: dict[str, Any], bindings: Sequence[ScopeBinding]
) -> dict[str, Any]:
    """One tool's argument schema with every Scope-bound argument taken out of it.

    Returns a new mapping and leaves the argument it was given alone: the caller holds
    the upstream's own `Tool`, the MCP client caches it for the life of the connection,
    and this runs again on every listing that connection serves.

    A `required` entry goes with the property it names, which is the half that is easy
    to miss and impossible to recover from -- a schema requiring a property it does not
    declare is one no call can satisfy, so the tool would be advertised and uncallable.
    A `required` list the removal empties is dropped rather than left as `[]`: the
    absent keyword says the same thing, and an empty array is illegal in drafts this
    platform does not get to rule out, since it does not choose the validator that
    reads what it advertises.

    Nothing is removed for a binding whose argument the schema no longer declares, and
    nothing is removed at all from a schema whose `properties` or `required` is not the
    shape JSON Schema gives it. Both cases go out as they came in.
    """
    if not _is_readable(schema):
        return dict(schema)
    bound = {binding.argument for binding in bindings}
    offered: dict[str, Any] = {}
    for keyword, value in schema.items():
        if keyword == "properties":
            offered[keyword] = {
                name: declared for name, declared in value.items() if name not in bound
            }
        elif keyword == "required":
            kept = [name for name in value if name not in bound]
            if kept:
                offered[keyword] = kept
        else:
            offered[keyword] = value
    return offered
