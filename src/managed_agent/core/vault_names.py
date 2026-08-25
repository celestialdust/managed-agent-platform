"""Turning a tenant's reference to a vault entry into the name of a vault entry.

Two surfaces let a tenant name an entry the platform will read on its behalf: a Tool
registration names one to authenticate an enterprise server, and a webhook registration
names one to sign a callback. In both, the reference is text the tenant wrote, and in
both the reader holds credentials belonging to every tenant. So the reference is not a
key: it is one component of a key whose other component is the tenant it came from.

Both prefixes compose here rather than each surface composing its own. The rule is not
the format -- it is the guarantee that the returned name cannot leave the tenant's
segment, and a second copy of that guarantee is a copy that can be weakened on one
surface while a test on the other still passes. That is not hypothetical: the webhook
path shipped with no composition at all while the Tool path had one, and the code read
as if the platform had a rule.

The prefix is the caller's because the two entries are different kinds of thing. A
webhook signing secret is not a Tool credential, and a shared namespace would let a
tenant sign callbacks under its own vendor token -- within one tenant, so far less
severe than a cross-tenant read, and still a confusion nothing needs.

`..` is refused anywhere in the reference rather than only at the front, and as a
substring rather than as a path segment. Secrets Manager holds a name as a literal
string and would not traverse, so this is not what stops the escalation today; the
promise is that the name returned cannot leave the tenant's segment, and that promise
has to survive the store being swapped for a path-like one. Refusing the substring costs
a reference containing `a..b`, which no registration has a reason to name.
"""

import re
from typing import Final

from managed_agent.core.ids import TenantId

TOOL_CREDENTIAL_PREFIX: Final[str] = "map/tool-credential"
"""The prefix every tool credential is stored under, ahead of the tenant and the ref.

Here rather than beside either user of it, because it stopped having one user. The Tool
Gateway composes this name to *read* a credential on the outbound call, and the control
plane composes the same name to *write* one when a tenant registers it -- two processes,
two IAM roles, one string that must be identical or the write lands where no read looks.
A constant defined in one of them and imported by the other would make one package's
internal detail into the other's contract; a constant copied into both is a value that
can be changed on one side while every test on the other still passes.

That is not hypothetical here. This platform has already shipped a prefix the code
composed and the IAM policy did not grant -- `map/dev/tools/*` was allowed and nothing
ever composed it -- and every authenticated MCP registration failed at AWS while looking
correct in isolation. Concentrating the string is what lets one guard check it against
both the policy and both callers.
"""

MAX_REF_LEN: Final[int] = 128
"""Longest reference a registration may name.

Bounded because the composed name goes to a store with a length limit of its own, and a
reference that composes to an over-long key fails at the fetch -- which reads as a vault
that would not answer rather than as a registration to fix.
"""

_REF_PATTERN: Final[re.Pattern[str]] = re.compile(
    rf"^[A-Za-z0-9][A-Za-z0-9_./-]{{0,{MAX_REF_LEN - 1}}}$"
)
"""What a reference may spell. Alphanumeric first character, so a reference cannot begin
with a separator and reach the segment above the tenant's."""


class VaultRefInvalid(Exception):
    """The reference is not a well-formed name, so it composes to no vault entry.

    Carries the reference and never a composed name: the reference is text the tenant
    wrote, so echoing it discloses nothing, while a composed name carries the tenant's
    own id and this message reaches a service log.
    """

    def __init__(self, credential_ref: str) -> None:
        super().__init__(f"{credential_ref!r} is not a vault reference")
        self.credential_ref: str = credential_ref


def parse_vault_ref(credential_ref: str) -> str:
    """The reference itself, once it is one. Raises `VaultRefInvalid` when it is not.

    Separate from composing a name so a registration surface can refuse a bad reference
    at the moment it is written, where the tenant reads the refusal, without inventing a
    tenant and a prefix it has no use for. `scoped_vault_name` runs the same function,
    so the boundary check and the composition cannot drift into two rules.
    """
    if ".." in credential_ref or not _REF_PATTERN.match(credential_ref):
        raise VaultRefInvalid(credential_ref)
    return credential_ref


def scoped_vault_name(prefix: str, tenant_id: TenantId, credential_ref: str) -> str:
    """The vault entry `credential_ref` means when this tenant is the one asking.

    Raises `VaultRefInvalid` rather than returning a name that would need checking
    afterwards: every caller here is about to read a credential, and a caller that
    forgot the check would read one belonging to somebody else.

    The tenant is composed in rather than compared against, so a reference naming
    another tenant's entry does not resolve to it -- the escalation is inexpressible
    instead of refused, and there is no branch to get wrong.
    """
    return f"{prefix}/{tenant_id}/{parse_vault_ref(credential_ref)}"
