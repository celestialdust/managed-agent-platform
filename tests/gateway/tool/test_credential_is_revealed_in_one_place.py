"""A credential is unwrapped in one module, composed in one, and never read unscoped.

Tier 1, structural, over the whole of `src/managed_agent`.

Three claims, each about a different way a credential escapes:

`Secret` refuses to render itself, so the only route from a held credential to a string
is `reveal()` -- and the whole value of that is lost if a second module calls it,
because that module can then format the value into a log line.

`credential_ref` is text a tenant wrote. A second module reading one is a second
composer of a vault key, and the tenant scoping would not travel there.

And every `.fetch(` on a keyed store is a read that must know whose key it is reading.
This is the claim that was missing: `control/webhooks/dispatcher.py` passed a tenant's
own `secret_ref` to the vault as a key, which let one tenant have any tenant's
credential used as the HMAC key over a body and to a destination it chose. The scan
reads every site in the package and names the ones it cannot read a tenant at, so no
count is written down here to drift away from what the tree holds.

**Formerly scoped to `gateway/tool/`**, because that read was live, merged and outside
the slice's declared surface -- a guard over all of `src/` would have failed on it and
invited whoever hit the failure to edit a file they may not touch. It is fixed, so the
scoping is no longer honest and is gone.

An AST walk rather than a grep, so a name inside a comment or a docstring cannot trip it
and a name reached through a differently-spelled attribute cannot hide from it. The glob
is recursive: a check that reads `glob("*.py")` is blind to a module one directory down,
which is how six structural checks in this repository were once defeated at once.

`unscoped_fetches` takes parsed modules rather than reading the tree itself, so the test
below drives it over `src/` **and** over synthetic modules whose answer is known. A
structural guard nobody has watched fail is a guard nobody has checked.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final

_PACKAGE: Final[Path] = Path(__file__).resolve().parents[3] / "src" / "managed_agent"

_UNWRAPPER: Final[str] = "gateway/tool/credential_broker.py"
"""The one module that may turn a `Secret` back into a string."""

_COMPOSERS: Final[frozenset[str]] = frozenset({"scoped_vault_name", "vault_name"})
"""Functions whose result is a vault key already scoped to a tenant.

Two rather than one because `gateway/tool/credential_broker.vault_name` wraps
`core.vault_names.scoped_vault_name` to raise the refusal the Tool path needs. Both are
checked below to actually take a tenant, so this set cannot drift into a claim that is
merely spelled.

A **prohibition, not an allowlist**: an argument this scan does not recognise fails the
test rather than passing it, so a third composer arriving fails loudly here and the
failure names the fix. `docs/lessons.md` records what the other direction costs.
"""

_PLATFORM_CREDENTIAL_READS: Final[frozenset[str]] = frozenset(
    {"gateway/model/credential_broker.py"}
)
"""The one vault read that has no tenant to be scoped to.

Every other `.fetch(` in the package reads something a tenant named, and that is the
whole premise of the scan below: such a name has to carry whose it is. This one reads
the opposite kind of thing -- an upstream provider credential named by a ConfigMap the
operator writes, one value shared by every tenant the platform serves, by design.
`deploy/iam/map-model-gateway.json` grants one secret prefix with no tenant component
anywhere in it, so there is no per-tenant entry for a composed name to reach and
nothing a composer could scope to.

Two, until 2026-08-23. The second was `gateway/model/router.py`, reading the
session-token signing key, and it is gone rather than merely moved: the key comes from
the environment now, so there is no fetch to exempt. Worth stating because the entry it
read was one nothing in the tree ever minted against, which made every model call a 401
-- the exemption was covering a read that could not have worked.

Named **modules** and deliberately not a new accepted argument form, because that is the
difference between narrowing and widening. A form would let every future unscoped read
pass; this lets exactly this file pass, and
`test_the_platform_credential_read_is_exactly_the_one_named` holds it there -- the scan
must report exactly this set, and the file must hold exactly one read, so a second read
landing in it does not inherit the justification here.
"""

_REF_COMPOSERS: Final[frozenset[str]] = frozenset({"core/vault_names.py", _UNWRAPPER})
"""The modules that may read a `credential_ref`.

`core/vault_names.py` is where a reference becomes a key. The broker reads one to hand
it to that function and to carry it on its own refusal. A third reader is a third answer
to "what entry does this tenant mean", and the tenant scoping would not travel to it.
"""


def _modules() -> dict[str, ast.Module]:
    """Every module under the package, keyed by its path relative to it."""
    return {
        path.relative_to(_PACKAGE).as_posix(): ast.parse(path.read_text())
        for path in sorted(_PACKAGE.rglob("*.py"))
        if path.name != "__init__.py"
    }


def _attribute_sites(tree: ast.Module, *names: str) -> list[int]:
    """The line of every attribute access whose attribute name is one of `names`."""
    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in names
    )


def _class_named(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is not defined where this guard expects it")


def _composed_locals(scope: ast.AST) -> set[str]:
    """Names bound in this scope, exactly once, to the result of a composer.

    Bound *exactly* once on purpose. A name assigned from a composer and then reassigned
    from something else would otherwise read as scoped at the fetch while holding
    whatever the second assignment put there.
    """
    bound: dict[str, int] = {}
    composed: set[str] = set()
    for node in ast.walk(scope):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            bound[target.id] = bound.get(target.id, 0) + 1
            if (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id in _COMPOSERS
            ):
                composed.add(target.id)
    return {name for name in composed if bound[name] == 1}


def _names_a_tenant_travels_under(call: ast.Call) -> set[str]:
    """Every spelling this call passes, positional and keyword, flattened one level.

    An attribute contributes its own last segment, so `pending.tenant_id` counts and
    `pending.url` does not.

    A leading underscore is stripped from that segment, so a caller holding the tenant
    privately -- `self._tenant_id`, which is how every long-lived object in this package
    keeps it -- reads as carrying the tenant, because it does. Without this the scan
    reported a genuinely scoped read and the fix on offer was to exempt the module,
    which is how an exemption list written for two credential reads grows a third entry
    that is not a credential read at all.
    """
    spellings: set[str] = set()
    for node in [*call.args, *(k.value for k in call.keywords)]:
        if isinstance(node, ast.Name):
            spellings.add(node.id)
        elif isinstance(node, ast.Attribute):
            spellings.add(node.attr)
            spellings.add(node.attr.lstrip("_"))
    spellings.update(k.arg for k in call.keywords if k.arg is not None)
    return spellings


def unscoped_fetches(modules: dict[str, ast.Module]) -> dict[str, list[int]]:
    """Every `.fetch(` call whose scope this scan cannot read, by module and line.

    A call is read as scoped when either
      * some argument is spelled `tenant_id`, positionally, as a keyword, or as the last
        segment of an attribute -- the six control-plane reads take this form; or
      * its argument is a call to a composer, or a local bound once from one -- the two
        credential reads take this form, because a vault key is not a value a caller
        holds, it is one a caller builds from the tenant and the tenant's reference.

    Anything else is reported. That is the point: the fix for a reported call is to pass
    the tenant or to compose the name, and both are readable here.
    """
    found: dict[str, list[int]] = {}
    for name, tree in modules.items():
        for scope in ast.walk(tree):
            if not isinstance(scope, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            composed = _composed_locals(scope)
            for node in ast.walk(scope):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "fetch"
                ):
                    continue
                spellings = _names_a_tenant_travels_under(node)
                if "tenant_id" in spellings or spellings & composed:
                    continue
                if any(
                    isinstance(a, ast.Call)
                    and isinstance(a.func, ast.Name)
                    and a.func.id in _COMPOSERS
                    for a in node.args
                ):
                    continue
                found.setdefault(name, []).append(node.lineno)
    return {name: sorted(lines) for name, lines in found.items()}


# --------------------------------------------------------------------------------
# The scan, held honest before anything leans on it.
# --------------------------------------------------------------------------------


def test_the_unwrapper_is_here_and_does_unwrap_a_credential() -> None:
    """Guard the guard. Both prohibitions below are satisfied by a package that stopped
    holding credentials at all, and this is the only thing that tells the two apart --
    the vacuous-negative shape `docs/lessons.md` records more than once."""
    modules = _modules()

    assert _UNWRAPPER in modules, f"{_UNWRAPPER} is not under {_PACKAGE}"
    assert _attribute_sites(modules[_UNWRAPPER], "reveal", "_value"), (
        f"{_UNWRAPPER} unwraps no credential, so every assertion below passes by "
        "having nothing to find. Either the Secret type moved, or the attribute it is "
        "unwrapped through was renamed and this walk no longer recognises it."
    )
    assert _attribute_sites(modules[_UNWRAPPER], "credential_ref"), (
        f"{_UNWRAPPER} reads no credential_ref, so it is composing no vault key"
    )


def test_every_composer_this_scan_trusts_actually_takes_a_tenant() -> None:
    """The scan reads a composer call as proof the name is scoped. That is only true
    while every composer takes a tenant, which is checked here rather than assumed."""
    defined = {
        node.name: node
        for tree in _modules().values()
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and node.name in _COMPOSERS
    }

    assert set(defined) == set(_COMPOSERS), (
        f"{sorted(set(_COMPOSERS) - set(defined))} is trusted as a composer of vault "
        "keys and is defined nowhere in the package"
    )
    for name, node in defined.items():
        parameters = {a.arg for a in node.args.args} | {
            a.arg for a in node.args.kwonlyargs
        }
        assert "tenant_id" in parameters, (
            f"{name} is trusted to return a name scoped to a tenant and takes no "
            f"tenant_id; it takes {sorted(parameters)}"
        )


def test_the_scan_reports_a_fetch_no_tenant_reaches() -> None:
    """The positive control, in the shape the live defect had: a reference the caller
    was handed, passed to a keyed store as a key."""
    unscoped = ast.parse(
        "async def deliver(self, secret_ref):\n"
        "    return await self._vault.fetch(secret_ref)\n"
    )
    composed_elsewhere = ast.parse(
        "async def deliver(self, tenant_id, secret_ref):\n"
        "    key = build_it(tenant_id, secret_ref)\n"
        "    return await self._vault.fetch(key)\n"
    )
    rebound = ast.parse(
        "async def deliver(self, tenant_id, secret_ref):\n"
        "    key = scoped_vault_name('p', tenant_id, secret_ref)\n"
        "    key = secret_ref\n"
        "    return await self._vault.fetch(key)\n"
    )

    assert unscoped_fetches({"m.py": unscoped}) == {"m.py": [2]}, (
        "the scan does not report the exact call shape the live defect had"
    )
    assert unscoped_fetches({"m.py": composed_elsewhere}) == {"m.py": [3]}, (
        "a name built by something this scan cannot read passed as scoped; an "
        "unrecognised form must fail rather than be trusted"
    )
    assert unscoped_fetches({"m.py": rebound}) == {"m.py": [4]}, (
        "a name composed and then reassigned passed as scoped"
    )

    near_miss = ast.parse(
        "async def read(self, ref):\n    return await self._vault.fetch(self._ref_id)\n"
    )

    assert unscoped_fetches({"m.py": near_miss}) == {"m.py": [2]}, (
        "stripping the leading underscore off an attribute must not turn every "
        "private attribute into a tenant; only `_tenant_id` may read as one"
    )


def test_the_scan_accepts_the_two_forms_the_package_actually_uses() -> None:
    """The negative control. A guard that fails working code is a guard somebody
    deletes, so both shapes in use are pinned as passing."""
    at_the_call = ast.parse(
        "async def read(self, session_id, tenant_id):\n"
        "    return await self._store.fetch(session_id, tenant_id)\n"
    )
    by_keyword = ast.parse(
        "async def read(self, tenant_id, file_id):\n"
        "    return await self._store.fetch(tenant_id=tenant_id, file_id=file_id)\n"
    )
    composed_inline = ast.parse(
        "async def read(self, tenant_id, ref):\n"
        "    return await self._vault.fetch(scoped_vault_name('p', tenant_id, ref))\n"
    )
    composed_local = ast.parse(
        "async def read(self, tenant_id, ref):\n"
        "    name = vault_name(tenant_id, ref)\n"
        "    return await self._vault.fetch(name)\n"
    )
    tenant_held_privately = ast.parse(
        "async def read(self):\n"
        "    return await self._store.fetch(self._session_id, self._tenant_id)\n"
    )

    for label, tree in (
        ("positional tenant", at_the_call),
        ("keyword tenant", by_keyword),
        ("composed inline", composed_inline),
        ("composed into a local", composed_local),
        ("tenant held privately", tenant_held_privately),
    ):
        assert unscoped_fetches({"m.py": tree}) == {}, (
            f"the scan reports {label}, which is a form the package uses correctly"
        )


# --------------------------------------------------------------------------------
# The prohibitions.
# --------------------------------------------------------------------------------


def test_no_read_of_a_keyed_store_is_unscoped() -> None:
    """Every `.fetch(` in the package either carries the tenant or is handed a name
    composed under one. This is the claim `control/webhooks/dispatcher.py` broke."""
    modules = _modules()
    reads = {
        name
        for name, tree in modules.items()
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "fetch"
    }

    assert {
        "gateway/tool/credential_broker.py",
        "control/webhooks/dispatcher.py",
    } <= reads, (
        f"the two credential reads this guard exists for are missing from "
        f"{sorted(reads)}; either they moved or the attribute they read through was "
        "renamed, and this "
        "walk no longer sees them"
    )
    reported = {
        name: lines
        for name, lines in unscoped_fetches(modules).items()
        if name not in _PLATFORM_CREDENTIAL_READS
    }

    assert reported == {}, (
        "these reads pass a key nothing scoped to a tenant. A vault or a keyed store "
        "answers for every tenant, so the name has to carry whose it is -- compose it "
        "with core.vault_names.scoped_vault_name, or pass tenant_id at the call."
    )


def test_the_platform_credential_read_is_exactly_the_one_named() -> None:
    """The control that keeps the exemption above from widening on its own.

    Two properties, and neither is about the *form* of a call. The scan must report
    exactly the exempted modules -- so an exemption that stopped being needed, or a
    further unscoped read anywhere, fails here. And each exempted module must hold
    exactly one `.fetch(` -- so a second read landing inside an already-exempted file
    does not inherit the justification written for the first.

    The first of those did its job on 2026-08-23: retiring the router's vault read made
    this fail, naming the stale entry, rather than leaving an exemption standing for a
    read nothing performs.
    """
    modules = _modules()
    sites = {
        name: [
            node.lineno
            for node in ast.walk(modules[name])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "fetch"
        ]
        for name in sorted(_PLATFORM_CREDENTIAL_READS)
    }

    assert set(unscoped_fetches(modules)) == _PLATFORM_CREDENTIAL_READS, (
        "the exemption no longer names exactly the unscoped reads in the package; "
        f"the scan reports {sorted(unscoped_fetches(modules))}"
    )
    assert {name: len(lines) for name, lines in sites.items()} == dict.fromkeys(
        sorted(_PLATFORM_CREDENTIAL_READS), 1
    ), f"an exempted module grew a second vault read: {sites}"


def test_no_other_module_unwraps_a_secret() -> None:
    offenders = {
        name: sites
        for name, tree in _modules().items()
        if name != _UNWRAPPER and (sites := _attribute_sites(tree, "reveal", "_value"))
    }

    assert offenders == {}, (
        f"{sorted(offenders)} reach inside a Secret. The reveal happens in "
        f"{_UNWRAPPER} and nowhere else, which is what stops a credential being "
        "formatted into a log line by a module that only meant to describe it."
    )


def test_no_other_module_reads_a_credential_ref() -> None:
    """A second reader of the ref is a second composer of a vault key, and the tenant
    scoping would not travel to it -- which is exactly the defect this closes."""
    offenders = {
        name: sites
        for name, tree in _modules().items()
        if name not in _REF_COMPOSERS
        and (sites := _attribute_sites(tree, "credential_ref"))
    }

    assert offenders == {}, (
        f"{sorted(offenders)} read credential_ref. Composing the vault key belongs to "
        f"{sorted(_REF_COMPOSERS)}, which put the calling tenant in front of it; a ref "
        "read anywhere else is text one tenant wrote being used as a key unscoped."
    )


def test_session_upstreams_holds_a_broker_and_not_a_vault() -> None:
    """The fetch must not come back beside the broker. A `_vault` on this class is the
    shape the cross-tenant read had: an object that can read any name, held by an
    object that was never told whose names it may read."""
    tree = _modules()["gateway/tool/mcp_proxy.py"]
    upstreams = _class_named(tree, "SessionUpstreams")

    assigned = {
        target.attr
        for node in ast.walk(upstreams)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Attribute)
    }

    assert "_vault" not in assigned, (
        "SessionUpstreams assigns self._vault. It holds a broker instead, so the "
        "object that composes a vault key is the one that knows whose key it is."
    )
    assert "_broker" in assigned, "SessionUpstreams no longer holds a broker at all"
    assert "_tenant_id" in assigned, (
        "SessionUpstreams no longer holds the tenant, so nothing it asks the broker "
        "for can be scoped"
    )
