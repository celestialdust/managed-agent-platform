"""Where a tenant's own tool credentials are named, so it can bring one of its own.

The platform could already *resolve* a tool credential and could not *accept* one.
`ToolCredentialBroker` fetches `map/tool-credential/<tenant>/<ref>` and attaches it to
the outbound call, and nothing anywhere let a tenant put a value at that name. So the
only MCP server that ever worked was a public one, and "bring your own credential" --
the thing an enterprise MCP server exists for -- was unreachable from outside.

These two tables hold the *names*. **No column here holds a secret**, and none ever
should: the value goes to Secrets Manager under the name composed from these rows, and
a copy in this database would be a second place to leak it from and a second place for
a rotation to miss. What a reader gets back from these tables is enough to manage a
credential and never enough to use one.

**The credential's ref is `<vault name>/<credential name>`, and that is why both names
are one segment.** `parse_vault_ref` admits `/`, so the two names are joined into the
ref the Tool Gateway already knows how to resolve -- which is what lets this arrive with
no change at all to the resolution path (`api-parity.md` §C5 requires exactly that). The
cost of the separator is the collision it invites: with `/` allowed inside a name, vault
`a/b` credential `c` and vault `a` credential `b/c` compose to one vault key, and one
tenant's credential silently shadows another of its own. The check constraints below
refuse a `/` in either name, so the two components of the ref can always be told apart.
This is the same failure `0028` names for `uuid5` inputs, met again on a different path.

`tenant_id` is carried on the credential as well as on its vault, denormalised on
purpose. Every read of a credential is tenant-scoped, and reading it through a join
means a query that forgets the join returns another tenant's row -- the predicate is
load-bearing for isolation, so it belongs on the table being read rather than one hop
away. The foreign key still points at the vault, and a credential whose tenant differs
from its vault's is refused by the composite reference below rather than by a trigger.

Archived rather than deleted, for a vault and for a credential alike, and no unarchive
-- the archive semantics `gap.md` adopts verbatim. **Archiving revokes nothing.** It
marks a row; the value stays in the vault and every outbound call using that ref goes on
working, because the Tool Gateway composes the key from the ref and never reads these
tables. `DELETE` is what revokes, and the two are separate routes for exactly that
reason. An earlier version of this docstring said an archived credential "stops being
attachable", which no code here or in the Gateway makes true -- the route's own
docstring has always said the opposite, and this is the half that was wrong.
"""

import sqlalchemy as sa
from alembic import op

revision = "0029"
down_revision = "0028"
branch_labels = None
depends_on = None

_ONE_SEGMENT = (
    "{column} <> '' AND {column} NOT LIKE '%/%' AND {column} NOT LIKE '%..%'"
    " AND {column} !~ '^[^A-Za-z0-9]'"
)
"""What a name may be, as SQL, for both tables.

Written once and formatted per column rather than pasted twice: the two names are
joined into one ref, so a rule that held for one and not the other would leave the ref
ambiguous in exactly the direction the docstring above describes.

It restates `core/vault_names._REF_PATTERN`'s guarantees in the database rather than
trusting the route that inserts. The route is where a tenant reads the refusal and this
is where the invariant is kept, and the two are not the same job: a second write door --
a backfill, a migration, a repair script -- reaches this and not the route.
"""


def upgrade() -> None:
    op.create_table(
        "vault",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("archived_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        # One name per tenant, so a ref composed from it names one vault. Two vaults of
        # one name would make `<vault>/<credential>` ambiguous at the point it is
        # resolved, which is inside the Tool Gateway where nothing can ask which was
        # meant.
        sa.UniqueConstraint("tenant_id", "name", name="vault_name_is_one_per_tenant"),
        # The composite target the credential's foreign key needs, so a credential
        # cannot point at a vault belonging to another tenant.
        sa.UniqueConstraint("id", "tenant_id", name="vault_id_carries_its_tenant"),
        sa.CheckConstraint(
            _ONE_SEGMENT.format(column="name"), name="vault_name_is_one_segment"
        ),
    )
    op.create_table(
        "vault_credential",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("vault_id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        # When the value behind this name was last written, which is not when the row
        # was created once a rotation has happened. Read by nothing that decides
        # anything -- it exists so a tenant looking at a credential that stopped working
        # can tell "rotated an hour ago" from "untouched since March" without the
        # platform having to show them a value it deliberately never returns.
        sa.Column(
            "value_written_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("archived_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "vault_id", "name", name="credential_name_is_one_per_vault"
        ),
        # Against `(id, tenant_id)` rather than `(id)`, which is what makes a credential
        # in another tenant's vault unrepresentable instead of merely refused: there is
        # no pair of values that satisfies this reference and crosses a tenant.
        sa.ForeignKeyConstraint(
            ["vault_id", "tenant_id"], ["vault.id", "vault.tenant_id"]
        ),
        sa.CheckConstraint(
            _ONE_SEGMENT.format(column="name"), name="credential_name_is_one_segment"
        ),
        # The closed set, kept in the database as well as in the type that parses the
        # request. A kind is what decides where the value may be attached, so a row
        # holding an unknown one is a credential nothing can place -- and it would be
        # discovered at a tool call, inside the Gateway, rather than at the write.
        sa.CheckConstraint(
            "kind IN ('static_bearer', 'environment_variable')",
            name="credential_kind_is_a_known_one",
        ),
    )
    # Listing a tenant's credentials is the read this serves, and it is the one read
    # that has no id to seek on. Partial on the live rows because a listing shows those
    # by default and an archive that grows without bound must not slow it down.
    op.create_index(
        "vault_credential_live_by_tenant",
        "vault_credential",
        ["tenant_id", "created_at"],
        postgresql_where=sa.text("archived_at IS NULL"),
    )


def downgrade() -> None:
    op.drop_index("vault_credential_live_by_tenant", table_name="vault_credential")
    op.drop_table("vault_credential")
    op.drop_table("vault")
