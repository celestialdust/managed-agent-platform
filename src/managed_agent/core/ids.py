"""Identity and ordering types for everything the platform records.

Ids are opaque to tenants on purpose: ADR-007 keeps the Agent Runtime's own identifiers
off the tenant surface, and an id that carries no runtime substring cannot leak one by
being logged or echoed.

Seq carries its constraint as a pydantic annotation, and where that constraint is
enforced is the thing to be clear about. A Session's sequence is contiguous from 1, so 0
is not a smaller sequence, it is a bug -- usually an index/count confusion. `ge=1` turns
that bug into a parse error wherever a **pydantic model** declares a field of this type,
which is every request body and every response the control plane parses. `strict` widens
that to ``True`` and ``1.0``: both compare equal to 1 and neither is a sequence number,
so admitting them would let an arithmetic mistake travel to the store.

What it does **not** do is validate at a call site. `Seq(0)` is `int(0)`: `Annotated` is
erased at runtime, so there the name is documentation and nothing more. That matters
because it is the opposite of what the syntax looks like: code reading `Seq(cursor + 1)`
looks guarded and is not. Anything that has to refuse a bad sequence outside a model
boundary checks it explicitly and says so -- `PostgresEventLogRange.read` is the worked
example, and the `next_seq >= 2` constraint in migration 0002 is the store's own.
"""

from typing import Annotated, NewType
from uuid import UUID, uuid4

from pydantic import Field

SessionId = NewType("SessionId", UUID)
TenantId = NewType("TenantId", UUID)
TurnId = NewType("TurnId", UUID)
DefinitionId = NewType("DefinitionId", UUID)
SkillId = NewType("SkillId", UUID)
VaultId = NewType("VaultId", UUID)
CredentialId = NewType("CredentialId", UUID)

Seq = Annotated[int, Field(ge=1, strict=True)]
"""A Session's own event sequence number. Contiguous from 1; never global, never 0.

Enforced where pydantic validates a field of this type. Written at a call site it is a
plain `int` and asserts nothing -- see the module docstring.
"""

FIRST_SEQ: Seq = 1


def new_session_id() -> SessionId:
    return SessionId(uuid4())


def new_turn_id() -> TurnId:
    return TurnId(uuid4())


def new_definition_id() -> DefinitionId:
    """A fresh Definition id.

    Here because there was nowhere else for it, and its absence had already produced a
    wrong line: MAP-4's plan snippet minted a Definition id by calling
    `new_session_id()`, which yields a correct value under a name saying it is a
    Session's. `NewType` is erased at runtime, so nothing would have caught it -- the
    two are the same `UUID` once the annotations are gone, and only the name would
    have been lying.
    """
    return DefinitionId(uuid4())


def new_skill_id() -> SkillId:
    """A fresh Skill id.

    Minted by the platform rather than taken from the uploader, for the reason
    `new_definition_id` is: a caller-chosen id could aim an upload at one another
    tenant already holds.
    """
    return SkillId(uuid4())


def new_vault_id() -> VaultId:
    """A fresh Vault id.

    Minted by the platform rather than taken from the caller, for the reason
    `new_skill_id` is: a caller-chosen id could aim a write at one another tenant
    already holds.
    """
    return VaultId(uuid4())


def new_credential_id() -> CredentialId:
    """A fresh Credential id.

    A named minter rather than `uuid4()` at the call site, for the reason this module
    records above: `NewType` is erased at runtime, so an id minted by the wrong
    neighbour's function carries a correct value under a lying name and nothing
    catches it. A credential id and a vault id are both uuids and the routes that
    write them sit in one file.
    """
    return CredentialId(uuid4())
