"""Which model each agent inside one Session runs on, and the one provider it may reach.

A Session's model is a creation-time choice that has to survive into a running pod
without anything inside the pod being able to change it. Two things carry it: the model
name and provider compiled into the Session's configuration, and one standalone document
per configured agent beyond the root. Both are rendered here so the checks over them are
written once.

Nothing here validates a model name against a catalog. A model name is a routing key,
not a contract -- a name no Routing Entry declares fails at the moment a Turn calls it,
naming the model, and a catalog checked at binding time would go stale in a way
indistinguishable from the failure it was added to prevent (ADR-010).

A slot is a filename component under the runtime's agents directory, so its shape is
parsed rather than trusted: a slot carrying a separator or a parent reference would
put a document somewhere the pod's own layout did not plan for.
"""

import re
import tomllib
from dataclasses import dataclass, replace
from typing import NewType

from managed_agent.core.ids import SessionId
from managed_agent.core.pod.workspace_contract import instructions_for_the_model
from managed_agent.core.registration.definition import AgentDefinition
from managed_agent.core.session.session import SessionRecord
from managed_agent.core.toml_text import toml_string

# The one model provider a Session may name, and it is the platform's rather than the
# tenant's. Every model call leaves a Session pod through the Model Gateway -- that is
# what holds the provider credentials and what the Egress Policy allows -- so a provider
# named by whoever submitted the definition would be a request to an endpoint outside
# it, from inside a pod running model-driven code. The definition supplies the model;
# this supplies where the model is reached.
#
# The spelling is also load-bearing in a way nothing else here would show. The runtime
# decides from a provider's NAME and base URL whether it is talking to first-party
# OpenAI or to Azure, and that decision turns on feature capability -- remote
# compaction, and a diagnostic's model-route probe. Neither is served here. The test
# beside this module asserts the string reads as neither, because the check belongs
# where the string can change: a runtime branch over a module constant has no reachable
# state that would flip it.
MODEL_PROVIDER_ID = "map-model-gateway"

# What the runtime shows a human when it names the provider it is talking to. Required,
# not decorative: a provider table without it is refused at configuration load, measured
# against codex-cli 0.149.0 in the pushed Session image --
# `Error loading config.toml: model_providers.map-model-gateway: provider name must not
# be empty`. So a table carrying only `base_url` does not start a Session either.
MODEL_PROVIDER_NAME = "MAP Model Gateway"

# The single wire protocol the Agent Runtime speaks, and the only value this field
# accepts (ADR-009): the runtime's own deserializer maps `"responses"` and hard-rejects
# `"chat"`. Written out rather than left to the default because the Model Gateway's
# router registers a handler for that wire and refuses the other two with a 502 on
# purpose, so the two ends have to be readable side by side.
MODEL_PROVIDER_WIRE = "responses"

MODEL_PROVIDER_AUTH_HEADER = "authorization"
"""The header the runtime attaches to every model call, carrying the Session token.

A static header in the compiled document rather than an environment variable, which is
what an earlier version of this module wrote (`env_key = "MAP_POD_TOKEN"`). That version
could not work as shipped and the reason is worth keeping: `env_key` names a
variable for whoever starts the pod to fill, and nothing fills it --
`session-pod.yaml`'s environment list is fixed at exactly four names with no
`valueFrom` and no `envFrom`, which is a floor another slice asserts and this one has
no business loosening. So the runtime read an unset variable and, measured against
real `map-dev`, sent the Model Gateway no request at all.

`http_headers` on a provider table is the runtime's own supported spelling for this
and puts the token exactly where the Tool Gateway's already rides: inside the one
document the compiler writes, under `CODEX_HOME`, which this module's profile denies
to every confined command. The runtime builds its header map once at client
construction, so what is written here is what every model call carries for the life
of the pod.

Spelled `authorization: Bearer <token>` rather than the Tool Gateway's `x-map-session`,
because this endpoint is wire-compatible with the Responses API and its inbound
authentication should look like the wire it imitates. What unified is the part that was
actually broken -- one token layout and one signing key across both gateways. ADR-023.
"""


AgentSlot = NewType("AgentSlot", str)
ROOT_SLOT = AgentSlot("root")

_SLOT = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")

_AGENT_DOCUMENT_KEYS = frozenset(
    {"name", "description", "developer_instructions", "model"}
)
"""Every key an agent document may carry, and every key it must carry.

A permit list rather than a deny list, because the key space it draws from is the whole
of the runtime's configuration and grows with the runtime: a deny list needs an edit
every time a new weakening key appeared, and would be silently short until somebody
noticed.
"""


class ModelBindingViolation(Exception):
    """A binding, a selection or a document that would not hold, so it is refused."""


class UnboundAgentSlot(Exception):
    """This Session binds no such slot, and no other slot's answer will do."""

    def __init__(self, session_id: SessionId, slot: str) -> None:
        super().__init__(f"session {session_id} binds no agent slot {slot!r}")
        self.slot = slot


def agent_slot(name: str) -> AgentSlot:
    """Parse an agent name into a slot, or refuse it.

    The pattern is narrow because the value becomes a filename and a TOML bare key: a
    dot, a slash or a space in it would either escape the agents directory or produce a
    document the runtime parses as something other than what was written.
    """
    if _SLOT.match(name) is None:
        raise ModelBindingViolation(
            f"agent slot {name!r} is not a lowercase name of at most 64 characters "
            "drawn from letters, digits, underscore and hyphen"
        )
    return AgentSlot(name)


@dataclass(frozen=True, slots=True)
class AgentBinding:
    """One agent's configuration inside one Session: its model, and what it is told.

    The slot is re-checked here rather than trusted from its type, because NewType is
    erased at runtime and a plain string reaches this constructor unchallenged.
    """

    slot: AgentSlot
    model: str
    instructions: str
    description: str

    def __post_init__(self) -> None:
        if _SLOT.match(self.slot) is None:
            raise ModelBindingViolation(f"agent slot {self.slot!r} is not a slot name")
        for field_name in ("model", "instructions", "description"):
            if not str(getattr(self, field_name)).strip():
                raise ModelBindingViolation(
                    f"agent slot {self.slot} binds an empty {field_name}"
                )


@dataclass(frozen=True, slots=True)
class SessionModelBindings:
    """Every agent configuration one Session holds, and nothing any other Session holds.

    One value per Session, built once and replaced rather than mutated, so there is no
    shared mutable state two Sessions could reach through. `definition_revision` travels
    with it because the models came out of that revision of the definition and no later
    one: whoever asks why a Session ran on a given model has the answer inside the same
    value.

    The agents are a tuple and lookups scan it. A Session holds a handful of agents, and
    an index would have to be built inside a frozen value or cached beside it -- either
    of which costs more than the scan it replaces.
    """

    session_id: SessionId
    definition_revision: str
    multiagent_enabled: bool
    agents: tuple[AgentBinding, ...]

    def __post_init__(self) -> None:
        slots = [agent.slot for agent in self.agents]
        if ROOT_SLOT not in slots:
            raise ModelBindingViolation(
                f"session {self.session_id} has no {ROOT_SLOT} binding, so nothing "
                "names the model its own thread runs on"
            )
        if len(set(slots)) != len(slots):
            raise ModelBindingViolation(
                f"session {self.session_id} binds one slot twice: {sorted(slots)}"
            )

    def binding_for(self, slot: AgentSlot) -> AgentBinding:
        """This slot's binding. Raises rather than answering with another slot's.

        The refusal is the isolation property. A lookup that fell back to the root would
        hand one agent another agent's model and another agent's instructions, and a
        fallback is indistinguishable from a correct answer at the call site.
        """
        for agent in self.agents:
            if agent.slot == slot:
                return agent
        raise UnboundAgentSlot(self.session_id, slot)

    @property
    def root(self) -> AgentBinding:
        return self.binding_for(ROOT_SLOT)

    @property
    def models(self) -> frozenset[str]:
        """Every model an agent here may reach. Nothing widens it at runtime."""
        return frozenset(agent.model for agent in self.agents)

    def permits(self, model: str) -> bool:
        return model in self.models

    def with_agent(self, binding: AgentBinding) -> "SessionModelBindings":
        """The same Session with one more agent bound, as a new value.

        The root cannot be replaced this way. It is fixed when the Session is created,
        and a Session whose root model changed under a running thread would hold a
        Rollout whose earlier Turns were served by a model the Session no longer names.
        """
        if binding.slot == ROOT_SLOT:
            raise ModelBindingViolation(
                f"session {self.session_id} already binds {ROOT_SLOT}"
            )
        return replace(self, agents=(*self.agents, binding))


def bind_agent(slot: AgentSlot, definition: AgentDefinition) -> AgentBinding:
    """Bind one registered definition to one slot.

    `description` is the definition's own name. The runtime requires the field on an
    agent document -- it is what a delegating model reads to decide whether to hand work
    to this agent -- and a definition carries no separate field for it, so the platform
    passes through the one string the tenant wrote rather than authoring prompt text on
    the tenant's behalf.
    """
    return AgentBinding(
        slot=slot,
        model=definition.model,
        instructions=definition.instructions,
        description=definition.name,
    )


def bind_session(
    record: SessionRecord, definition: AgentDefinition
) -> SessionModelBindings:
    """The bindings a Session starts with: its root agent, from the revision it pinned.

    `multiagent_enabled` comes from the definition rather than from a default, and the
    default is off: a Session whose definition never asked for subagents cannot spawn
    one, which is one fewer way for a model nobody named to be reached. The runtime's
    own default for that key is on, so writing it out narrows rather than restates.
    """
    return SessionModelBindings(
        session_id=record.id,
        definition_revision=record.definition_revision,
        multiagent_enabled=definition.multiagent.enabled,
        agents=(bind_agent(ROOT_SLOT, definition),),
    )


def _quoted(value: str) -> str:
    """Quote a value as a TOML basic string, as a `ModelBindingViolation` on refusal.

    The escaping is `core/toml_text.py`'s, which this module's own table used to be a
    copy of. What stays here is the exception type: everything in this module raises
    `ModelBindingViolation`, and a caller catching that should not also have to catch a
    `ValueError` from one line inside it.

    Why it matters at all: a definition's instructions run to many lines, and a raw
    newline in a basic string is a parse error rather than a line break -- so the
    document would be rejected by the runtime at load, after the pod started.
    """
    try:
        return toml_string(value)
    except ValueError as refusal:
        raise ModelBindingViolation(str(refusal)) from refusal


def _check_selection_floor(gateway_base_url: str) -> None:
    """Refuse a base URL that would render correctly and reach nothing.

    The runtime builds each request's URL by joining the configured base URL with the
    literal segment `responses`, trimming one trailing slash from the base and no more:
    `{base}/responses`. So the base has to end at `/v1`, with or without that slash,
    because `/v1/responses` is the only path the Model Gateway serves. Any other tail
    produces a 404 from a service that is running correctly, and a 404 for a path is
    spelled the same way as a 404 for an unknown model.

    The scheme is checked for being one the runtime can address at all, not for being
    TLS. This hop is plaintext inside the cluster today and that is a recorded decision,
    not an oversight -- a check demanding https here would refuse the address the
    platform actually runs on and would be satisfied by nothing.
    """
    if not gateway_base_url.startswith(("http://", "https://")):
        raise ModelBindingViolation(
            f"the model gateway base url {gateway_base_url!r} names no http scheme, so "
            "the runtime has no address to send a model call to"
        )
    if not gateway_base_url.rstrip("/").endswith("/v1"):
        raise ModelBindingViolation(
            f"the model gateway base url {gateway_base_url!r} must end at '/v1': the "
            "runtime appends the literal segment 'responses' to it, and "
            "'/v1/responses' is the only path the Model Gateway serves"
        )


def render_model_selection(
    bindings: SessionModelBindings, *, gateway_base_url: str, session_token: str
) -> str:
    """The configuration fragment naming this Session's model and its one provider.

    Emitted as bare keys followed by tables, and placed first in the document it joins:
    TOML reads a bare key as belonging to whatever table header precedes it, so `model`
    written after a table header would quietly become that table's key.

    `agents.default_subagent_model` is the fail-safe half of the multi-agent posture. A
    spawned subagent resolves its model as explicit-value-at-spawn, then this default,
    then the parent's -- so an agent spawned with no model of its own lands here, and
    pointing it at the root's model makes the worst case a model this Session bound
    rather than whatever the runtime would otherwise pick.

    **Writing that key is also what puts the model catalogue on every spawn path, and
    that is the reason it stays.** The runtime checks a requested subagent model against
    its catalogue and refuses an absent one -- but it skips the check entirely when the
    spawn names no model *and* no default is configured. This key removes that skip: it
    supplies the name, so the catalogue is consulted on every spawn whether the model
    mentions a model or not.

    Until 2026-08-24 that made the key the cause of the defect rather than a fail-safe.
    The catalogue is the runtime's own, it names eight models, and it named none of this
    platform's -- so the default this line writes was refused every time, and a tenant
    who asked for delegation got five refused spawns and an agent that did the whole
    task itself. What changed is not this line but that `config_compiler` now compiles a
    catalogue naming this Session's models, which is what makes the default legal.

    So the two are one mechanism and neither works alone. Deleting this line would
    re-open the runtime's skip and leave the catalogue untested on the common path;
    compiling no catalogue while this line stands is the defect above. The test beside
    this module asserts them together rather than separately, because a reader who
    changes one has to be told about the other.

    **`developer_instructions` is the tenant's own text, and until 2026-08-23 it went
    nowhere.** A definition's `instructions` reached `AgentBinding.instructions`, and
    the only reader was `render_agent_document`, which refuses the root slot outright --
    so a SUBAGENT received its instructions while the root agent -- the thread every
    Session actually wraps -- received none. Measured on a live pod: a definition whose
    instructions said "your codeword is X", asked for its codeword, answered
    `NO CODEWORD`. A tenant wrote their agent's whole persona and the model never saw a
    word of it, with nothing in any log saying so.

    `developer_instructions` and NOT `base_instructions`, which the runtime also accepts
    here. `base_instructions` IS the runtime's built-in system prompt -- the text that
    teaches the model `apply_patch`, the shell conventions and its own tool protocol
    (`codex-rs/core/src/session/mod.rs:1296`, reading
    `prompt_with_apply_patch_instructions.md`). Writing a tenant's paragraph there would
    not add a persona, it would delete the runtime's operating manual and leave a model
    that cannot edit a file. Emitted above the first table header for the same reason
    the egress keys are: a key written after `[agents]` parses as
    `agents.developer_instructions`, a table nothing reads, in a file that loads clean.

    Unconditional, because `AgentDefinition.instructions` has `min_length=1` and so
    cannot be empty. There is deliberately no "if set" branch: a branch here is a path
    on which the model silently loses its instructions again, which is the defect this
    line closes.

    **It carries the platform's workspace contract as well, and that is not where
    that text belongs.** The runtime has an administrator's channel for it,
    `additional_developer_instructions` in `requirements.toml`, which would keep the
    two authorships apart and out of a document a tenant can read. Neither released
    codex has that key: zero occurrences in the 0.149.0 binary this platform pins and
    zero in 0.149.1, against 39 and 43 for this key. A document written that way
    parses, loads, and drops the contract without a word. Measured before it was
    believed -- the key sat correctly escaped in a live pod's
    `/etc/codex/requirements.toml` while an agent asked three questions only the
    contract answers said `UNKNOWN` to all three.

    `core/pod/workspace_contract.py:instructions_for_the_model` composes the two and
    labels both halves. Not concatenated here: the moment this function decides how
    the authorships are marked, the module that owns every other sentence the model
    reads about its workspace stops being the one place that answers for them.

    The Session's token to the Model Gateway rides in the provider table as a static
    header, for the reasons `MODEL_PROVIDER_AUTH_HEADER` gives. What that costs, stated
    rather than left for a reader to find: a shell command the model runs can read the
    configuration file and so can read this token, and with it call the Model Gateway
    outside the Turn the platform is metering. That is already true of the Tool
    Gateway token two tables further down, so this changes the number of credentials
    in that file from one to one -- both are the same token. The bound is the token's
    own expiry and the fact that it proves only which Session is calling; closing it
    properly means handing the runtime a credential the model's own commands cannot
    read, which no configuration key here offers.
    """
    _check_selection_floor(gateway_base_url)
    root = bindings.root
    return "\n".join(
        (
            "# Compiled per Session. The model this Session named, and the one",
            "# provider it may reach. Nothing inside the pod writes this file.",
            f"model = {_quoted(root.model)}",
            f"model_provider = {_quoted(MODEL_PROVIDER_ID)}",
            "developer_instructions = "
            + _quoted(instructions_for_the_model(root.instructions)),
            "",
            "[agents]",
            f"enabled = {str(bindings.multiagent_enabled).lower()}",
            f"default_subagent_model = {_quoted(root.model)}",
            "",
            f"[model_providers.{_quoted(MODEL_PROVIDER_ID)}]",
            f"name = {_quoted(MODEL_PROVIDER_NAME)}",
            f"base_url = {_quoted(gateway_base_url)}",
            f"wire_api = {_quoted(MODEL_PROVIDER_WIRE)}",
            "http_headers = { "
            f"{_quoted(MODEL_PROVIDER_AUTH_HEADER)} = "
            f"{_quoted('Bearer ' + session_token)}"
            " }",
            "",
        )
    )


@dataclass(frozen=True, slots=True)
class AgentDocument:
    """One agent's standalone configuration, named but not yet written anywhere.

    A filename rather than a path: which directory these land in is a property of the
    pod's layout, and this module would be a second place that layout was written down.
    """

    filename: str
    toml: str


def _keys_at_any_depth(node: object) -> set[str]:
    """Every key name at any depth of a parsed document.

    Depth matters: a weakening key is legal inside a nested table as well as at the top,
    so a check reading only the root would pass a document that reopens what the two
    compiled documents were written to close.
    """
    if isinstance(node, dict):
        found: set[str] = set(node)
        for value in node.values():
            found |= _keys_at_any_depth(value)
        return found
    if isinstance(node, list):
        nested: set[str] = set()
        for item in node:
            nested |= _keys_at_any_depth(item)
        return nested
    return set()


def check_agent_document(document: str) -> None:
    """Raise unless the document carries every permitted key and nothing else.

    This is the third configuration layer and the only one nothing else reads. A custom
    agent document may override any configuration key the runtime has, `sandbox_mode`
    and `mcp_servers` among them -- the first names the mechanism the permission profile
    replaces, and the second would put a second MCP server beside the Tool Gateway,
    which is the invariant that makes the Gateway the runtime's only reachable server
    (ADR-005, ADR-014). The floor check over the two compiled documents never sees this
    one, so a document is graded here before it is returned.

    `cwd` is outside the permit list for a quieter reason: with no working directory of
    its own an agent inherits the parent thread's, which is what keeps every agent in
    one Session on one Session VFS.
    """
    parsed = tomllib.loads(document)
    present = _keys_at_any_depth(parsed)
    surplus = present - _AGENT_DOCUMENT_KEYS
    if surplus:
        raise ModelBindingViolation(
            f"an agent document carries {sorted(surplus)}; such a document can "
            "override any configuration key, so only the permitted set is ever written"
        )
    missing = _AGENT_DOCUMENT_KEYS - present
    if missing:
        raise ModelBindingViolation(f"an agent document is missing {sorted(missing)}")


def render_agent_document(binding: AgentBinding) -> AgentDocument:
    """Render one configured agent, and refuse to return one that does not hold.

    The document's `name` is the slot, so the name the runtime spawns by and the name
    the platform binds by are one string and nothing has to map between them.
    """
    if binding.slot == ROOT_SLOT:
        raise ModelBindingViolation(
            "the root agent is the thread the Session wraps and has no agent document"
        )
    document = "\n".join(
        (
            "# Compiled per Session. Nothing inside the pod writes this file.",
            f"name = {_quoted(binding.slot)}",
            f"description = {_quoted(binding.description)}",
            f"model = {_quoted(binding.model)}",
            f"developer_instructions = {_quoted(binding.instructions)}",
            "",
        )
    )
    check_agent_document(document)
    return AgentDocument(filename=f"{binding.slot}.toml", toml=document)
