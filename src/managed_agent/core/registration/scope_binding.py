"""What a Tool registration declares: the name a Grant will name, and how the Tool
Gateway reaches the server behind it.

The tool-name form is the platform's, not the Agent Runtime's. The Agent Runtime
qualifies an MCP tool as `mcp__<server>__<tool>`, rewrites every character outside
`[a-zA-Z0-9_]` to `_`, appends a SHA1-derived twelve-hex suffix when two tools would
sanitize to one name, and truncates the result to fit 128 bytes. A name a Grant was
written against would then be free to change under a runtime upgrade. So a registered
name is already lowercase ASCII, already fits the byte budget with room held back for
the qualification prefix, and is unique per tenant -- under which the sanitizer is the
identity function and the collision suffix has nothing to disambiguate.

A registration names a vault entry and never a secret, and it names one per server
because an enterprise server this platform brokers is an authenticated one. Which
attachment point that entry ends up in is the difference between the two transports,
which is why the transport is a field of the registration rather than a fact discovered
when someone first tries to call the server.

The refusal of a tool whose Scope cannot be expressed lands here too, while the
declaration is being parsed rather than at the store or at the first call: the
declaration carries everything the check needs, so a type that admitted it would let it
reach a Session (ADR-003).
"""

from enum import StrEnum
from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_QUALIFIED_TOOL_NAME_BYTES: Final[int] = 128
"""The Agent Runtime's own ceiling on the qualified name it shows a model."""

QUALIFICATION_RESERVE_BYTES: Final[int] = 32
"""Held back from that ceiling for the `mcp__<server>__` prefix it prepends."""

MAX_TOOL_NAME_BYTES: Final[int] = (
    MAX_QUALIFIED_TOOL_NAME_BYTES - QUALIFICATION_RESERVE_BYTES
)

_TOOL_NAME_PATTERN: Final[str] = rf"^[a-z][a-z0-9_]{{0,{MAX_TOOL_NAME_BYTES - 1}}}$"

ToolName = Annotated[str, Field(pattern=_TOOL_NAME_PATTERN)]
"""The name a Grant names and the Tool Gateway advertises.

Lowercase only, though the sanitizer would preserve uppercase: it keeps `Search` and
`search` apart as two names, so allowing both would let a Grant written against one
silently miss the other. The character class is ASCII, so the pattern's length limit and
the byte limit are the same number and neither needs encoding to check.
"""

ServerName = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]{0,62}$")]
"""The name an agent definition names a server by. Unique within one tenant."""

HeaderName = Annotated[str, Field(pattern=r"^[!#$%&'*+.^_`|~0-9A-Za-z-]{1,64}$")]
"""The name of the request header a credential is attached under.

RFC 7230's `tchar` set exactly, rather than something narrower: this refuses a name no
HTTP request can carry and refuses nothing a server might really want, and a check that
fails a legal registration is a check somebody deletes.

Its sibling `credential_env_var` was a pattern from the start and this was
`min_length=1`, which accepts `"Ok\\nBad"` and `"  "`. Neither reaches the wire -- h11
refuses an illegal header name at serialization and httpx surfaces that as a transport
error -- so what the asymmetry cost was not an injection but a registration that can
never work, failing on every tool call instead of once at the moment it was written.
"""


def qualification_fits(runtime_facing_server_name: str) -> bool:
    """Whether `mcp__<name>__` fits the reserve held back above.

    The Agent Runtime is configured with exactly one MCP server, the Tool Gateway, and
    whatever name the compiled configuration gives it is what consumes this reserve.
    Stating the arithmetic as a function means the compiler checks it against the same
    number the names were bounded by, rather than carrying a copy of that number.
    """
    qualified = f"mcp__{runtime_facing_server_name}__"
    return len(qualified.encode()) <= QUALIFICATION_RESERVE_BYTES


class StdioServer(BaseModel):
    """A server the Tool Gateway reaches by spawning it and speaking over its stdio."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    transport: Literal["stdio"]
    command: str = Field(min_length=1)
    args: tuple[str, ...] = ()
    credential_ref: str = Field(min_length=1)
    credential_env_var: str = Field(pattern=r"^[A-Z][A-Z0-9_]{0,63}$")


class StreamableHttpServer(BaseModel):
    """A server the Tool Gateway reaches over Streamable HTTP."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    transport: Literal["streamable_http"]
    url: str = Field(pattern=r"^https://")
    credential_ref: str | None = None
    credential_header: HeaderName = "Authorization"

    @model_validator(mode="after")
    def _a_header_is_not_named_without_a_credential_to_put_in_it(self) -> Self:
        """Refuse a `credential_header` on a registration that names no credential.

        An omitted `credential_ref` means a server that authenticates nobody, and
        that is a real thing: `https://mcp.deepwiki.com/mcp` answers `initialize`
        over plain HTTPS with no header at all. Naming the header for a credential
        that does not exist is the one combination nothing can honour -- the call
        would carry no header while the registration states which header it
        carries -- so it is refused where it is written, rather than arriving later
        as a 401 from the far end.

        Both sides are read from which fields the caller wrote, not from their
        values. For `credential_header` that is necessary: its default is a real
        header name, which a caller may equally have typed on purpose. For
        `credential_ref` it is a constraint rather than a preference -- composing a
        vault key is reserved to the two modules that put the calling tenant in
        front of it, so reading the ref's value anywhere else, including here in
        the model that declares it, is the thing that must not spread.

        One case therefore passes: an explicit `credential_ref: null` written
        alongside a header. It reaches the far end as an unauthenticated call,
        which that end refuses, and getting there takes writing the contradiction
        out longhand.
        """
        wrote_a_header = "credential_header" in self.model_fields_set
        wrote_a_credential = "credential_ref" in self.model_fields_set
        if wrote_a_header and not wrote_a_credential:
            raise ValueError(
                "credential_header names where a credential goes, and this "
                "registration names no credential_ref to put there"
            )
        return self


ServerEndpoint = Annotated[
    StdioServer | StreamableHttpServer, Field(discriminator="transport")
]
"""One or the other, never a merge: a stdio registration cannot carry a URL and an HTTP
one cannot carry an argv, so whatever attaches the credential has no half-populated case
to decide about."""


class ParameterType(StrEnum):
    """The declared type of one tool parameter. Only STRING can carry a Scope value."""

    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    OBJECT = "object"
    ARRAY = "array"


ScopeDimension = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
ParameterName = Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")]

RegisteredNameKind = Literal["server", "tool"]


class ScopeBinding(BaseModel):
    """One dimension of a Session's Scope, and the tool argument it binds into."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dimension: ScopeDimension
    argument: ParameterName


class ToolRegistration(BaseModel):
    """One tool as a registration declares it.

    `name` is what a Grant names; `remote_name` is what the server behind it calls the
    same tool. Keeping them apart is what lets a server rename its own tool without
    invalidating a Grant, and what keeps the name a person reads out of the hands of
    whatever the Agent Runtime does to names it is handed.

    `parameters` is the tenant's declaration of the tool's arguments, not an
    introspection of them. The Tool Gateway clamps the bound argument by name on the
    outbound call either way, so a declaration that does not match the real tool makes
    the call fail at the server rather than widening the Scope. That asymmetry is the
    reason a wrong declaration is safe to accept and an absent binding is not.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: ToolName
    remote_name: str = Field(min_length=1, max_length=128)
    parameters: dict[ParameterName, ParameterType]
    scope_bindings: tuple[ScopeBinding, ...]

    @model_validator(mode="after")
    def _bindings_are_expressible(self) -> Self:
        """Refuse a tool no Session Scope could narrow, naming the tool and the reason.

        A tool registered without an enforceable narrowing is reachable by every Session
        whose Grant names it, at the full breadth of the tenant's data -- so the refusal
        is the enforcement, and it belongs where the person who wrote the declaration is
        the one who reads it (ADR-003).
        """
        if not self.scope_bindings:
            raise ValueError(
                f"tool {self.name}: no Scope Binding declared, so no Session Scope "
                f"could narrow a call to it"
            )
        seen_dimensions: set[str] = set()
        seen_arguments: set[str] = set()
        for binding in self.scope_bindings:
            declared = self.parameters.get(binding.argument)
            if declared is None:
                raise ValueError(
                    f"tool {self.name}: the Scope Binding for dimension "
                    f"{binding.dimension} names argument {binding.argument}, which "
                    f"this tool does not declare"
                )
            if declared is not ParameterType.STRING:
                raise ValueError(
                    f"tool {self.name}: the Scope Binding for dimension "
                    f"{binding.dimension} names argument {binding.argument}, declared "
                    f"as {declared.value}; a Scope value can only be written into a "
                    f"string"
                )
            if binding.dimension in seen_dimensions:
                raise ValueError(
                    f"tool {self.name}: dimension {binding.dimension} is bound twice"
                )
            if binding.argument in seen_arguments:
                raise ValueError(
                    f"tool {self.name}: argument {binding.argument} is bound twice, so "
                    f"which dimension narrows it would depend on evaluation order"
                )
            seen_dimensions.add(binding.dimension)
            seen_arguments.add(binding.argument)
        return self


class ServerRegistration(BaseModel):
    """One MCP server and every tool it offers, stated once.

    Nothing here refers to an agent definition. A definition names a server by name, so
    one registration serves as many definitions as name it and there is no
    per-definition copy free to drift from this one.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    server_name: ServerName
    endpoint: ServerEndpoint
    tools: tuple[ToolRegistration, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _tool_names_are_unique_within_the_registration(self) -> Self:
        """Refuse a registration that offers one name twice, naming the repeats.

        The store refuses this too, by way of the key on `(tenant, name)`. Both are
        needed and they are not the same check: the store's is what holds against a
        concurrent second registration, and this one is what makes the refusal legible
        -- an integrity error names a constraint, not the tool a person mistyped.
        """
        names = [tool.name for tool in self.tools]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(
                f"tool names repeat within one registration: {', '.join(repeated)}"
            )
        return self


class RegisteredTool(BaseModel):
    """What the registry hands back: one tool, with the server it lives behind.

    Parsed on the way out of the store as well as on the way in, so a row written under
    an older shape reaches the Tool Gateway as a checked value or not at all.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: ToolName
    remote_name: str
    parameters: dict[ParameterName, ParameterType]
    scope_bindings: tuple[ScopeBinding, ...]
    server_name: ServerName
    endpoint: ServerEndpoint


class NameAlreadyRegistered(Exception):
    """A registration would claim a name this tenant has already registered.

    Carries which kind of name and which ones, because what the caller does next
    differs: a taken server name means this registration was already made, while a taken
    tool name means two servers are offering one name to the same Grant.

    Defined here rather than beside the SQL that raises it, because an exception a port
    promises to raise is part of that port's interface -- and because the control plane
    has to catch it while the layer rule forbids the control plane importing an adapter.
    """

    def __init__(self, kind: RegisteredNameKind, names: tuple[str, ...]) -> None:
        super().__init__(f"{kind} name already registered: {', '.join(names)}")
        self.kind: RegisteredNameKind = kind
        self.names: tuple[str, ...] = names


class UnknownTool(Exception):
    """No tool of that name is registered to that tenant.

    One refusal covers both "never registered" and "another tenant's", so holding a
    name confirms nothing about anyone else's catalog.
    """
