"""A named, reusable sandbox shape that a Session is started in.

Everything here is fixed when the shape is registered. That is not tidiness: the
guarantee this type exists to make is that two Sessions naming one id run in the same
shape, and a field that can be rewritten -- or an image reference that can resolve to
different bytes on different days -- breaks that guarantee silently, on the second
Session, with nothing to read afterwards that says the two differed.

The image must therefore be digest-pinned. A tag is a name for whatever was pushed
last; a digest is a name for bytes.

`denied_paths` narrows and can only narrow. Each entry becomes one more deny rule in
the Session's Permission Profile, and the registry that parses this refuses any path
that is not strictly inside the one writable root -- so a shape can carve a hole in
what the agent may write and can never open one. That direction is checked where the
writable root is defined rather than here, because this module is pure domain and the
root is the compiler's contract with the pod manifest.

`allowed_domains` is the one field here that WIDENS, and it is written differently for
that reason. Empty is no network at all, which is the default a caller gets by saying
nothing; a non-empty list turns the sandbox's egress on and confines it to those names.
So the safe value is the absent one, and every entry a caller adds is a capability the
agent did not have. The rules below are correspondingly strict about spelling -- a name
this type accepts is one the sandbox's proxy compares against, and a pattern it would
read more broadly than the caller meant is refused rather than narrowed for them.

Rules run in `__post_init__` rather than in a separate checker, so a constructed
Environment is proof its fields were parsed and nothing downstream re-checks or skips
them.
"""

import re
from dataclasses import dataclass
from typing import NewType
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from managed_agent.core.ids import TenantId

EnvironmentId = NewType("EnvironmentId", UUID)
"""Declared beside the type it identifies rather than in `core/ids.py`.

`NewType` is erased at runtime, so `EnvironmentId(x)` asserts nothing at a call site;
the name is documentation and the parse that gives it meaning is `Environment` below.
"""

_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_GLOB_METACHARACTERS = frozenset("*?[]")

MAX_DENIED_PATHS = 64
"""A bound rather than a limit anyone asked for: every entry becomes a rule the runtime
compiles into a sandbox argv on every launch, so an unbounded list is an unbounded
argv."""

MAX_ALLOWED_DOMAINS = 32
"""Bounded for the same reason and at half the size, because each entry is a rule the
sandbox's proxy evaluates on every connection rather than once at launch."""

_DOMAIN_LABEL = re.compile(r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$")

_CLUSTER_SUFFIXES = (".local", ".internal", ".svc", ".cluster")
"""Suffixes refused outright, and the refusal is defence in depth rather than the guard.

The real guard is the sandbox's proxy, which blocks loopback, link-local and private
destinations by default and blocks them by RESOLVED address -- so a hostname pointing
at a private IP stays blocked whatever it is called, which is what keeps the node's
instance metadata service and this cluster's own unauthenticated control plane out of
reach.

These are refused anyway because a caller writing one of them has misunderstood what
this field is for, and a request that silently does nothing teaches them nothing. A
refusal naming the suffix does.
"""


def new_environment_id() -> EnvironmentId:
    """A fresh Environment id."""
    return EnvironmentId(uuid4())


class CreateEnvironment(BaseModel):
    """What a tenant sends. Shape only -- the invariants belong to Environment below.

    Unknown fields are refused rather than ignored: a caller that misspelled
    `denied_paths` would otherwise believe it had narrowed a shape that denies nothing.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=64)
    runtime_image: str = Field(min_length=1)
    denied_paths: tuple[str, ...] = ()
    allowed_domains: tuple[str, ...] = ()
    """Absent by default, which is no egress. See `Environment.allowed_domains`."""


@dataclass(frozen=True, slots=True)
class Environment:
    """One registered sandbox shape. No field of this is ever rewritten.

    Raises `ValueError` on construction for a name that is only whitespace, an image
    that is not digest-pinned, a denied path that is relative, glob-form, unnormalised
    or repeated, and a list longer than `MAX_DENIED_PATHS`.

    A path is accepted only in the spelling the sandbox compares against.
    `/session/workspace/x/` and `/session/workspace/./x` name one directory and are two
    strings, and a rule written in a spelling nothing else uses is a rule whose target
    the sandbox looks for at a path the pod never created. The same four glob
    metacharacters `FsRule` refuses are refused here, so a path this type accepts is one
    the runtime parses as a concrete path rather than as a pattern to expand.

    Whether a path is one this shape is *allowed* to deny is a different question and is
    not asked here: it depends on where the writable root is, which belongs to whatever
    compiles the pod's configuration rather than to the domain.
    """

    id: EnvironmentId
    tenant_id: TenantId
    name: str
    runtime_image: str
    denied_paths: tuple[str, ...]
    allowed_domains: tuple[str, ...] = ()
    """The names the agent's own commands may reach, or nothing, which is no network.

    Empty is not "unrestricted": the sandbox keeps egress off unless a profile turns it
    on, and this field is what turns it on. So a shape that says nothing here produces
    an agent whose `curl` and `pip` fail to connect, which is the state every Session on
    this platform ran in before this field existed.

    A leading `*.` matches subdomains and nothing else -- `*.example.com` covers
    `a.example.com` and does NOT cover `example.com`, so a caller who wants both writes
    both. That is the sandbox proxy's own reading and it is repeated here because the
    other reading is the one a caller assumes.

    Refused: an entry with a scheme, a path, a port, a userinfo, upper case, an IP
    literal, a bare `*`, a `*` anywhere but a leading label, a name with no dot, and any
    name under `_CLUSTER_SUFFIXES`. Each of those is either a spelling the proxy would
    compare differently from how it reads, or a destination this platform does not offer
    a tenant a way to ask for.
    """

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("an environment needs a name a person can tell it by")
        reference, separator, digest = self.runtime_image.partition("@")
        if not separator or not _DIGEST.match(digest):
            raise ValueError(
                f"runtime_image must be digest-pinned as name@sha256:<64 hex>, "
                f"not {self.runtime_image!r}"
            )
        if not reference or any(character.isspace() for character in reference):
            raise ValueError(f"runtime_image names no image: {self.runtime_image!r}")
        if len(self.denied_paths) > MAX_DENIED_PATHS:
            raise ValueError(f"at most {MAX_DENIED_PATHS} denied paths")
        if len(set(self.denied_paths)) != len(self.denied_paths):
            raise ValueError(f"a path is denied twice: {sorted(self.denied_paths)}")
        for path in self.denied_paths:
            if not path.startswith("/"):
                raise ValueError(f"a denied path must be absolute: {path!r}")
            if _GLOB_METACHARACTERS & set(path):
                raise ValueError(
                    f"a denied path must be a prefix, not a glob: {path!r}"
                )
            if path.endswith("/") or "//" in path:
                raise ValueError(f"a denied path must be normalised: {path!r}")
            if any(part in {".", ".."} for part in path.split("/")):
                raise ValueError(f"a denied path must be normalised: {path!r}")
        if len(self.allowed_domains) > MAX_ALLOWED_DOMAINS:
            raise ValueError(f"at most {MAX_ALLOWED_DOMAINS} allowed domains")
        if len(set(self.allowed_domains)) != len(self.allowed_domains):
            raise ValueError(
                f"a domain is allowed twice: {sorted(self.allowed_domains)}"
            )
        for domain in self.allowed_domains:
            _parse_domain(domain)


def _parse_domain(domain: str) -> None:
    """Raise unless `domain` is a name the sandbox's proxy reads the way it is written.

    Every refusal here is one of two kinds and it is worth telling them apart.

    A SPELLING the proxy compares differently from how a caller reads it: upper case, a
    scheme, a path, a port, userinfo, a trailing dot, an empty label. Each of these is a
    name that either matches nothing or matches something other than what was meant, and
    a field whose whole purpose is to be narrow cannot afford either.

    A DESTINATION this platform does not offer a tenant a way to ask for: an IP
    literal, a bare `*`, a wildcard anywhere but the leading label, a name with no dot,
    and anything under `_CLUSTER_SUFFIXES`. A bare `*` is the whole internet written as
    if it were one entry. A wildcard in the middle -- `a.*.com` -- is not something the
    proxy expands, so it would silently match nothing. A dotless name is either a
    search-domain lookup, whose answer depends on the pod's resolver rather than on this
    list, or `localhost`, which the proxy blocks anyway.

    Nothing here checks that the name RESOLVES. It may not resolve today and may resolve
    tomorrow, and a parse that reached DNS would make constructing an Environment depend
    on the network -- and would then have to decide what to do when the answer changed.
    """
    if not domain:
        raise ValueError("an allowed domain cannot be empty")
    if domain != domain.lower():
        raise ValueError(f"an allowed domain must be lower case: {domain!r}")
    if any(character.isspace() for character in domain):
        raise ValueError(f"an allowed domain cannot contain whitespace: {domain!r}")
    for spelling, what in (
        ("//", "a scheme"),
        (":", "a port or a scheme"),
        ("/", "a path"),
        ("@", "userinfo"),
        ("?", "a query"),
    ):
        if spelling in domain:
            raise ValueError(
                f"an allowed domain is a host name and carries no {what}: {domain!r}"
            )
    if domain.endswith("."):
        raise ValueError(f"an allowed domain must not be fully qualified: {domain!r}")
    for suffix in _CLUSTER_SUFFIXES:
        if domain == suffix.lstrip(".") or domain.endswith(suffix):
            raise ValueError(
                f"{domain!r} names this cluster's own network, which is not a "
                f"destination a Session may be granted"
            )
    labels = domain.split(".")
    if len(labels) < 2:
        raise ValueError(
            f"an allowed domain needs at least two labels, so that what it matches "
            f"does not depend on the pod's resolver: {domain!r}"
        )
    if labels[0] == "*":
        labels = labels[1:]
    for label in labels:
        if not _DOMAIN_LABEL.match(label):
            raise ValueError(
                f"{label!r} is not a domain label, in {domain!r}. A wildcard is "
                f"allowed only as the leading label, written as '*.'"
            )
    if all(character.isdigit() or character == "." for character in domain):
        raise ValueError(
            f"{domain!r} is an address and not a name; the proxy blocks private "
            f"addresses by resolution, so naming one here grants nothing and reads "
            f"as if it did"
        )
