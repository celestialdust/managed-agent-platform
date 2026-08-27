"""A Session's durable filesystem: the lanes a Turn's output is delivered in.

The pod's workspace is a mounted subtree of a volume every Session shares (ADR-035), so
the agent's working files are durable where they sit and cross a Turn boundary without
anything copying them anywhere. This module is for the other thing: what a Turn
*delivers*. An artifact is not a working file that happened to survive -- it is a byte
string the platform promised a tenant under a digest the tenant can check later, and
that promise needs a store which can refuse to overwrite. A filesystem cannot be that
store; its whole contract is that the last writer wins.

So what is here is the namespace the delivery path agrees on, and the rule that a
delivered object is written once. Getting bytes out of the pod is transport and is
deliberately not here.

**Which lanes a developer's own code organises into is that developer's decision,
not this module's.** A platform that shipped a fixed taxonomy would hand a product
opinion to every engineer building on it, and the one who needed a fifth lane, or three,
or different words for the same four, would work around the platform rather than with
it. So the mechanism comes first: a lane is a name refused unless it could be one
directory, and a caller declares the set it wants.

**One lane is named here anyway, because the platform itself writes it.** `ARTIFACTS`
holds what a Turn produced, written by control-plane code in this tree on every Session,
whether or not any developer ever declares a lane of their own. A name the platform
writes has to be one constant: the ship-out that places into it and the route that
serves it are two modules that must agree, and two string literals agree only until one
is edited. That is the platform's one, not a ceiling on anyone else's.

**There is one lane kind, and every lane is sealed.** There was a second -- a mutable
kind, carrying the agent's workspace between Turns, with a `replace` that accepted files
in it and nothing else. That lane is gone: the workspace is a mounted volume now and
nothing copies it across a boundary (ADR-035), which left the mutable kind with no
caller. Keeping it for one that does not exist would leave a way to make an object here
rewritable sitting next to the promise that objects here are not, and the next writer to
reach for it would be reaching past a guarantee rather than around a limitation.

**Why sealed at all.** An agent that can overwrite its own evidence can "correct" the
raw record it is supposed to be judged against, and an artifact whose bytes can be
swapped after the fact makes every recorded digest a claim about bytes that may no
longer exist. Both are silent: the write succeeds, the file looks right, and only an
auditor rehashing an object finds the discrepancy. A boolean on a lane, or a string
compared at each call site, would put that guarantee in every caller's hands -- and the
one caller that forgets is the one that matters. So there is no call here that
overwrites. Rewriting a stored object is not refused; it is not expressible.

The surface is the design and the store is the enforcement. `place` writes conditionally
-- it fails if the key already holds an object -- so a sealed lane's immutability
survives a caller that reaches the adapter by some route this module did not anticipate.
That conditional write is also why nothing here needs a delete: the control plane's
grant has `s3:PutObject` and `s3:GetObject` and deliberately no `s3:DeleteObject`, so an
overwrite is the only way to destroy a stored object at all, and a conditional put is
exactly the thing that closes it.

**Keyed by tenant and then Session, in that order.** The Session id alone would be
enough to separate two Sessions, and it is not enough to make a cross-tenant read
inexpressible: the relative path is text the agent wrote, and a reader composing a key
from it holds every tenant's objects. So the tenant is composed in rather than compared
afterwards -- the same rule, and the same reason, as `core/vault_names.py`, where a
surface that shipped with no composition at all read as though the platform had one.

The digest type is `core.evidence`'s rather than a second one of this module's own.
What it pins -- SHA-256 over the octets as stored and as later served, with the length
part of the identity so a truncated read cannot masquerade as a corrupt object -- is
exactly what a VFS object needs, and a second copy of that rule is a copy that can be
weakened here while the tests over there still pass.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

from managed_agent.core.ids import SessionId, TenantId
from managed_agent.core.vfs.evidence import EvidenceDigest

_LANE_PATTERN: Final = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class LaneNameInvalid(ValueError):
    """A lane was declared under a name that cannot be one directory.

    Carries the name. Unlike a relative path, this is text a developer wrote in a
    declaration rather than anything a tenant's data reveals, so it is safe in a log and
    useless without it -- a refusal that does not say which name it refused sends the
    reader back to guess which entry of a declaration was wrong.
    """

    def __init__(self, directory: str) -> None:
        super().__init__(f"{directory!r} is not a lane directory name")
        self.directory: str = directory


def parse_lane_name(directory: str) -> str:
    """The name itself, once it is one directory.

    Raises `LaneNameInvalid` when it is not.

    Validated at construction because which lanes exist is a caller's decision rather
    than this module's, so a lane name arrives from a declaration nobody here reviewed.
    While the set was four constants in this file, an unvalidated name was safe by
    accident; the moment it is declared elsewhere, it is the leading segments of an
    object key composed from input.

    Two names matter. One containing `/` composes a prefix below a directory nothing
    declared, and one that is or contains `..` composes a prefix above this Session's --
    which would put one tenant's key inside another's. S3 traverses neither, because it
    holds a key as a literal string, and that is exactly why this cannot be left to the
    store: the same objects are reachable through a mounted filesystem, which resolves
    both. `..` is refused as a substring for the reason `parse_relative_path` gives for
    doing the same, and at the same cost -- a lane named `a..b`, which nothing wants.
    """
    if ".." in directory or not _LANE_PATTERN.match(directory):
        raise LaneNameInvalid(directory)
    return directory


@dataclass(frozen=True, slots=True)
class SealedLane:
    """A lane whose objects are written once. Rewriting one is not an operation.

    Carries only its directory name, so two sealed lanes differ in nothing a caller can
    branch on -- which is the point. What distinguishes one sealed lane from another is
    what a reader is promised about it, and none of that is a decision this store makes.
    """

    directory: str

    def __post_init__(self) -> None:
        parse_lane_name(self.directory)


Lane = SealedLane
"""What the rest of the tree annotates a lane parameter with.

One kind today, and the alias is kept rather than inlined so that stays a fact about
this module: a second kind would be added to this line, not to the thirty signatures
that name it.
"""


ARTIFACTS: Final = SealedLane("artifacts")
"""What a Turn produced, kept exactly as it was produced.

Sealed, and that is the delivery promise rather than tidiness. A tenant told an artifact
has digest X downloads it later and can check; if the same key could be rewritten, every
recorded digest would be a claim about bytes that may since have been swapped, and the
swap leaves no trace a reader could find. The conditional write is what makes an
artifact's bytes final, and it is the only thing that does: the Tool Gateway -- the
process that writes this lane -- does hold `s3:DeleteObject` on this bucket, in
`deploy/iam/map-tool-gateway.json` under Sid `EvidenceAndArtifacts`. So the seal is a
property of this code, not one IAM would go on holding if the code stopped asking.
Nothing under `gateway/` calls a delete today, which is what keeps narrowing that grant
a live option rather than a rewrite.

The cost is real and is handled rather than avoided: nothing clears the agent's output
directory between Turns, so a second Turn offers the first Turn's files again and the
store refuses the second write. `control/files/output_shipout.py` reads the stored
digest and treats identical bytes as already delivered -- see the reasoning there.
"""

LANES: Final = (ARTIFACTS,)
"""The lanes the platform itself writes, for a caller sweeping or listing all of them.

One entry, and still a tuple. A second lane -- `working`, holding the agent's tree
between Turns -- was here until the workspace became a mounted volume that needs no
carrying (ADR-035). What is left is a sequence with one member rather than a bare
constant, because the callers that sweep it sweep "the lanes the platform writes", and
that is a set which happens to have one member rather than a fact about `ARTIFACTS`.

A tuple rather than a list because nothing may append to it at run time -- a lane the
platform writes is a lane some module in this tree places into, and one added by
mutation at run time has no such module behind it.
"""


MAX_RELATIVE_LEN: Final[int] = 512
"""Longest relative path a lane will hold.

Bounded because the composed key goes to a store with a key-length limit of its own, and
a path that composes to an over-long key fails at the write -- which reads as an object
store that would not answer rather than as a path to fix.
"""

_RELATIVE_PATTERN: Final[re.Pattern[str]] = re.compile(
    rf"^[A-Za-z0-9][A-Za-z0-9_./-]{{0,{MAX_RELATIVE_LEN - 1}}}$"
)
"""What a relative path may spell. The first character is alphanumeric, so a path can
neither begin with a separator and reach the segment above its lane nor begin with a dot
and address the lane's own prefix."""


class VfsPathInvalid(Exception):
    """The relative path is not a well-formed name, so it composes to no object key.

    Carries the path and never a composed key: the path is text the agent wrote, so
    echoing it discloses nothing, while a composed key carries the tenant's own id and
    this message reaches a service log.
    """

    def __init__(self, relative: str) -> None:
        super().__init__(f"{relative!r} is not a lane-relative path")
        self.relative: str = relative


def parse_relative_path(relative: str) -> str:
    """The path itself, once it is one. Raises `VfsPathInvalid` when it is not.

    Refuses `..` anywhere in the path rather than only at the front, and as a substring
    rather than as a whole segment. S3 holds a key as a literal string and would not
    traverse, so the substring rule is not what stops an escape today; the promise is
    that a composed key cannot leave this Session's lane, and that promise has to
    survive the store being swapped for a path-like one. It costs a path containing
    `a..b`, which nothing has a reason to write.

    An empty segment and a trailing separator are refused too. Both compose to a key
    that names a prefix rather than an object, and a write to one stores bytes under a
    name no read reconstructs from the path that produced it.
    """
    if (
        ".." in relative
        or "//" in relative
        or relative.endswith("/")
        or not _RELATIVE_PATTERN.match(relative)
    ):
        raise VfsPathInvalid(relative)
    return relative


def lane_prefix(tenant_id: TenantId, session_id: SessionId, lane: Lane) -> str:
    """Where one Session's lane begins in the bucket.

    The one place a VFS key's leading segments are composed. Both file types below
    build their key from this rather than each formatting its own, because the guarantee
    is not the format -- it is that the result cannot leave this tenant's segment, and a
    second copy of that is a copy that can be weakened on one path type while a test on
    the other still passes.

    Tenant before Session so a tenant's whole VFS is one prefix. That ordering is what
    lets a tenant-wide sweep be a prefix operation instead of a join against every
    Session id the tenant ever created.
    """
    return f"sessions/{tenant_id}/{session_id}/{lane.directory}/"


@dataclass(frozen=True, slots=True)
class SealedFile:
    """One object in a lane that does not accept a rewrite.

    The relative path is parsed at construction, so holding one of these is proof the
    path composes to a key inside this Session's own lane. Nothing downstream re-checks
    it, which is the point: a check repeated at each use is a check one use will omit.
    """

    tenant_id: TenantId
    session_id: SessionId
    lane: SealedLane
    relative: str

    def __post_init__(self) -> None:
        parse_relative_path(self.relative)

    @property
    def key(self) -> str:
        return lane_prefix(self.tenant_id, self.session_id, self.lane) + self.relative


VfsFile = SealedFile
"""What the rest of the tree annotates a stored-object parameter with.

Kept for the same reason as `Lane`, and it moves with it: one file type today because
there is one lane kind, and a second would be added here rather than at every use.
"""


@dataclass(frozen=True, slots=True)
class SourceRef:
    """One input a stored object was derived from, and the digest it had at the time.

    A path plus the digest read from it, so a later reader can rehash the source and
    learn whether the derivation still describes the bytes it was made from. That check
    is the whole reason the digest is recorded beside the path rather than looked up
    when asked.
    """

    relative: str
    digest: EvidenceDigest


@dataclass(frozen=True, slots=True)
class StoredObject:
    """What a write returns: where the bytes went, and the digest taken over them.

    The digest is computed at the write and returned rather than left for the caller to
    take, so the value that reaches the record is a hash of the octets that were
    actually stored. A caller hashing its own buffer and reporting that instead would be
    right until the two came apart, and nothing would say which had happened.
    """

    key: str
    digest: EvidenceDigest


@dataclass(frozen=True, slots=True)
class LaneEntry:
    """One object found by listing a lane: its path within the lane, and its size.

    No digest. A listing returns sizes and not content hashes, and a field filled by
    reading every object would make listing a lane cost a download of it. A caller that
    needs the digest reads the provenance the write appended, or reads the object.
    """

    relative: str
    byte_length: int


class ObjectAlreadyPresent(Exception):
    """That key already holds an object, so the conditional write stored nothing.

    Raised by `place` in every lane, not only the sealed ones. In a sealed lane it is
    the immutability rule refusing a second write; in the mutable lane it says the
    caller wanted `replace` and reached for the creating call instead.

    Declared here rather than in the adapter that raises it, because an exception a port
    promises to raise is part of that port's interface, and the layer rule forbids the
    control plane from importing an adapter. `SequenceRace` and `UnknownDefinition` in
    `core/ports.py` are here for the same reason.
    """


class VfsUnconfigured(RuntimeError):
    """No object store is wired behind the VFS, so a lane has nowhere to live."""


@runtime_checkable
class LaneBlobs(Protocol):
    """The three object-store operations a lane needs, and no others.

    The only write is conditional. An unconditional `put` sat here while a mutable lane
    existed, and it went with that lane rather than being left as a capability nobody
    calls -- a store that *can* overwrite is one a later writer will overwrite with,
    and the seal is worth exactly as much as the narrowest surface behind it.

    There is no delete either, for a nearer reason: the grant this runs under has none,
    and a lane's expiry is a prefix sweep owned by whatever owns retention -- not by the
    module that writes one file.
    """

    async def put_new(self, key: str, body: bytes) -> None:
        """Store `body` at `key` only if nothing is there. Raises
        `ObjectAlreadyPresent`."""
        ...

    async def get(self, key: str) -> bytes | None:
        """The bytes at `key`, or None when the key holds nothing.

        None rather than raising, because "the key is empty" is a fact the caller has to
        act on -- it means a record and the bucket disagree -- while a store's own error
        type reaching control-plane code would put an infrastructure vocabulary in a
        decision that is about the platform.
        """
        ...

    async def list_prefix(self, prefix: str) -> Sequence[LaneEntry]:
        """Every object under `prefix`, with the prefix stripped from each name."""
        ...


@runtime_checkable
class SessionFiles(Protocol):
    """A Session's durable filesystem, as everything above the adapter sees it.

    One write, and it creates. There is no call here that overwrites, so "may this key
    be written twice" is not a question any caller asks or any store answers -- which is
    the point. A `write(lane, ...)` that decided by inspecting the lane would make every
    caller a place that decision could be got wrong.
    """

    async def place(
        self, file: VfsFile, body: bytes, sources: Sequence[SourceRef] = ()
    ) -> StoredObject:
        """Write a new object, in any lane. Raises `ObjectAlreadyPresent`.

        `sources` is taken at the write because the write is the moment they are known.
        A caller that has nothing to declare passes nothing, and the record then says
        the writer named no sources -- which is deliberately not the same recorded claim
        as an object derived from nothing.
        """
        ...

    async def read(self, file: VfsFile) -> bytes | None:
        """The object's bytes, or None when the lane holds nothing at that path."""
        ...

    async def list_lane(
        self, tenant_id: TenantId, session_id: SessionId, lane: Lane
    ) -> Sequence[LaneEntry]:
        """Everything in one lane of one Session.

        Takes the lane as `Lane`, which is the alias every lane-shaped parameter in this
        tree takes. A caller listing several of a Session's lanes in one loop holds them
        in one variable, and that stays true of whatever `Lane` names later.
        """
        ...
