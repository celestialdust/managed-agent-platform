"""Narrow one tool call to the Session's Scope, or refuse the call that cannot be.

A tool registration declares, per tool, which Session Scope dimension binds into which
of the tool's arguments, and a tool that declares none is refused at registration. This
is where that declaration becomes a boundary rather than a promise: on every call the
Scope's value for a bound dimension is written into the named argument, and a caller
that supplied that argument itself is refused rather than answered.

**Refusing, not overwriting.** The bound argument is not in the schema the Gateway
advertises, so a call carrying it is either from below the advertised layer or from a
model ignoring its own schema -- a protocol violation either way, and the field it
names is the one field on the tool the caller has no say in. This replaces an earlier
overwrite, whose argument was that refusing a disagreement would hand the model an
oracle. That argument does not survive a successful call: the overwritten call went out
under the Scope's value and came back with the scoped data, so a model read the Scope
out of the first success rather than out of many refusals. What the overwrite cost was
real and unweighed -- on a write or a delete the platform silently redirected the target
the model named, and "the model asked to delete A, the platform deleted B" is not a
sentence that can be explained to a tenant afterwards (ADR-034).

**The attempt is written to this process's own log, with the value on it.** That value
is the only thing separating a model probing for the Scope from a caller carrying an
argument that has gone stale, and it is tenant data, so it goes where no model and no
tenant reads: a platform log line, never an Event Log entry, which would put it under a
retention clock and in front of the tenant's own webhooks. It is deliberately not
carried on the refusal this function returns -- everything the caller does with that
value flows toward the model, and one echo of it reopens the oracle.

**A mismatch between the Scope and the bindings refuses the call, whichever way it
runs.** A Scope silent on a bound dimension refuses the whole call, including the
bindings it could have narrowed; and a Scope dimension that no binding names refuses it
too. Applying the bindings it can and sending the rest wide is the dangerous
half-measure, and it is dangerous in both directions for one reason: the call looks
narrowed at the point it goes out while a dimension nobody could hold it to runs across
everything the credential can reach. The second direction ran that way silently until
ADR-034 -- a Scope of `{repository, environment}` against a tool binding only
`repository` went out across every environment, fully narrowed to look at.

The cost is deliberate and is the point. A Session scoped on two dimensions can no
longer call a tool that binds one, which may be most tools, so this surfaces as broken
calls before it surfaces as safety and it forces a tenant's registrations to match the
Scopes they create Sessions with. Refusing at call time rather than at Session creation
is what keeps a tool edited later from invalidating a live Session's invariant, and it
still refuses nothing for the tools a Session never calls.

Nothing here reaches a network, a clock or a store, so what a call is narrowed to is a
function of the registration and the Session's creation facts alone -- both of which
are fixed before the Session's first Turn.

The clamp is the enforcement half of ADR-003, whose Status records that it was deferred
and that `tool.out_of_scope` was published with nothing raising it.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from managed_agent.core.registration.scope_binding import RegisteredTool

_log = logging.getLogger(__name__)


class ScopeRefusal(StrEnum):
    """Which way a Scope and a tool's bindings failed to meet.

    Two causes rather than one because the tenant's next move differs: one is fixed by
    the Scope a Session is created with, the other by a binding the registration does
    not declare. They share a refusal type and a published code all the same -- a
    second type would be a second arm at every caller, and a caller that forgets one
    falls through to sending the call.

    A member here obliges a sentence in `error_map.out_of_scope`, which matches on this
    exhaustively, so a third cause cannot be added without deciding what it says.
    """

    DIMENSION_NOT_IN_SCOPE = "dimension_not_in_scope"
    """The tool binds a dimension this Session's Scope does not carry."""

    DIMENSION_NOT_BOUND = "dimension_not_bound"
    """This Session's Scope carries a dimension the tool binds nothing to."""


@dataclass(frozen=True, slots=True)
class OutOfScope:
    """The call could not be held to the Session's Scope, and which way it failed.

    Carries names only. `dimension` is a term the tenant wrote into their own
    registration or their own create call, and `tool_name` is the name the caller just
    used, so neither tells a reader of the refusal anything it did not already have --
    and the Scope's values are deliberately not here, for the reason the module
    docstring gives about oracles.

    `cause` says which of the two mismatches this is. Without it the two would arrive
    as one sentence that is true of one of them and misleading about the other, and a
    tenant reading "this Session's Scope does not carry one" about a dimension their
    Scope plainly carries goes looking at the wrong half of the pair.
    """

    tool_name: str
    dimension: str
    cause: ScopeRefusal


@dataclass(frozen=True, slots=True)
class ArgumentNotOffered:
    """The call supplied an argument the Scope fills, so it was not made.

    Carries names only, on the same terms as `OutOfScope` above and for a sharper
    reason: the value that provoked this refusal is the one thing a probing model
    would learn from. It is written to this module's log at the point of detection and
    goes no further, so nothing downstream can echo back either what was attempted or
    what the Scope holds.

    `argument` is here and `OutOfScope` has no equivalent because the two refusals ask
    for different things. That one is about the Session and the caller can do nothing
    about it; this one is about the call, and the argument's own name -- which the
    caller supplied -- is what it needs in order to stop supplying it.
    """

    tool_name: str
    dimension: str
    argument: str


def narrow(
    tool: RegisteredTool,
    scope: Mapping[str, str],
    arguments: Mapping[str, object],
) -> dict[str, object] | OutOfScope | ArgumentNotOffered:
    """The arguments the outbound call should carry, or why it must not be made.

    Returns a new mapping; `arguments` is left as the caller passed it, so the record
    of what the model asked for stays readable beside what the platform sent.

    Three refusals, resolved in a fixed order, and the order is a decision rather than
    an accident of which loop is written first.

    Every binding is resolved against the Scope before any is applied, so a tool whose
    Scope covers some dimensions and not others refuses rather than going out
    half-narrowed. It runs first of the three, which settles the one case where a
    Session trips both mismatches at once -- a Scope of `{topic}` against a tool binding
    `region` is answered about `region`, the dimension the registration declared. That
    keeps the second rule from re-labelling refusals that already had a settled answer:
    a Session refused before this rule existed is refused afterwards for the same stated
    reason, and the new cause speaks only where the old one had nothing to say.

    Then the Scope's own dimensions, each of which some binding must name. Both of these
    are facts about the Session and the registration, true of every call this Session
    will ever make to this tool, so both are settled before anything about how one call
    was written. A caller told to drop an argument would drop it, call again, and be
    refused for the reason that held all along.

    Last, the arguments the caller supplied. An unbindable Scope therefore records no
    attempt in the log: the arguments were never read, so there is nothing to record,
    and a line here would put a value in the log for a call whose arguments were not the
    problem.

    Which unbound dimension is named does not depend on the order the Scope was
    written in. A Session's Scope is stored as an ordered tuple of pairs, so two
    Sessions bounded to the same thing can carry it either way round, and a refusal
    that varied with that would tell two identically-scoped tenants to fix different
    things.
    """
    bound: dict[str, object] = {}
    for binding in tool.scope_bindings:
        value = scope.get(binding.dimension)
        if value is None:
            return OutOfScope(
                tool_name=tool.name,
                dimension=binding.dimension,
                cause=ScopeRefusal.DIMENSION_NOT_IN_SCOPE,
            )
        bound[binding.argument] = value
    narrowed_by = {binding.dimension for binding in tool.scope_bindings}
    for dimension in sorted(scope):
        if dimension not in narrowed_by:
            return OutOfScope(
                tool_name=tool.name,
                dimension=dimension,
                cause=ScopeRefusal.DIMENSION_NOT_BOUND,
            )
    for binding in tool.scope_bindings:
        if binding.argument in arguments:
            _log.warning(
                "scope clamp refused a supplied bound argument tool=%s dimension=%s "
                "argument=%s attempted=%r",
                tool.name,
                binding.dimension,
                binding.argument,
                arguments[binding.argument],
            )
            return ArgumentNotOffered(
                tool_name=tool.name,
                dimension=binding.dimension,
                argument=binding.argument,
            )
    return {**arguments, **bound}
