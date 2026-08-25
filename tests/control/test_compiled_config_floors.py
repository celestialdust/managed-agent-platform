"""Grades config-compiler's own output against the floors the pod's isolation rests on.

Tier 1 (local, no infrastructure). Realizes MAP-A88 and MAP-A89 at the configuration
layer, and is the check invariants I5, I6, I7 and I15 name.

Two halves, and neither works alone. The negative cases mutate one floor in an
otherwise valid document and assert compilation would have refused it -- a checker
that has never rejected anything is not known to reject anything. The positive cases
assert the document is the real thing, because every absence this file checks is also
satisfied by an empty document: a compiler that emitted nothing names no MCP server,
configures no TCP listener and carries no sandbox key. So each absence has a presence
beside it, driven by the same fixture.

Every document mutation asserts that it changed the text before asserting the
refusal. A mutation that silently fails to apply passes vacuously and looks exactly
like a guard that works; this repository has shipped that mistake from a pattern that
could not match what it was aimed at.

The last group is a drift check between two files that must name the same paths. It
parses the pod manifest rather than searching it: the manifest's own comments name
every path the compiler denies, so a substring search over the text is satisfied by
the comments alone and would keep passing if every mount, volume and command were
deleted.
"""

import re
import tomllib
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
import yaml

from managed_agent.control.pod_config import compiler as config_compiler
from managed_agent.control.pod_config.compiler import (
    CODEX_HOME,
    CONTROL_SOCKET,
    CONTROL_SOCKET_DIR,
    GATEWAY_SERVER_ID,
    PROFILE_NAME,
    SYSTEM_CONFIG_DIR,
    WORKSPACE_ROOT,
    CompiledConfig,
    FloorViolation,
    check_floors,
    compile_session_config,
    session_profile,
)
from managed_agent.control.pod_config.model_binding import (
    MODEL_PROVIDER_AUTH_HEADER,
    MODEL_PROVIDER_ID,
)
from managed_agent.core.ids import TenantId, new_definition_id, new_session_id
from managed_agent.core.pod.permission_profile import (
    FsAccess,
    FsRule,
    PermissionProfile,
    is_strictly_under,
    nested_deny_pairs,
    path_spelling_error,
)
from managed_agent.core.registration.definition import (
    AgentDefinition,
    SkillsRevision,
)
from managed_agent.core.registration.environment import Environment, new_environment_id
from managed_agent.core.session.session import SessionRecord
from managed_agent.core.session.session_token import (
    SESSION_TOKEN_HEADER_NAME,
    InvalidSessionToken,
    verify_session_token,
)

POD_TEMPLATE = (
    Path(__file__).resolve().parents[2] / "deploy" / "k8s" / "session-pod.yaml"
)
GATEWAY_URL = "https://tool-gateway.map.internal/mcp"

# Where a Session pod reaches the Model Gateway. The `/v1` is load-bearing at both ends:
# the Agent Runtime POSTs `{base_url}/responses`, and the Gateway's router mounts
# `POST /v1/responses`.
MODEL_GATEWAY_URL = "http://model-gateway.map-dev.svc.cluster.local/v1"

_TOKEN_KEY = b"a signing key that is thirty-two"

_EXPIRY = 4102444800
"""An absolute second far enough out that no run of this suite reaches it.

A literal rather than `time.time() + n`: a fixture whose expiry depends on when it
ran is a fixture that can expire between two assertions, and the failure would read
as a broken floor rather than as a stale fixture.
"""

# The definition a Session pins, for the one field the compiler reads off it: the model.
# The provider is not here because it is not the definition's to name -- every model
# call leaves a Session pod through the Model Gateway.
A_DEFINITION = AgentDefinition(
    name="slr-reviewer",
    instructions="Extract findings and name the source for each.",
    model="gpt-5-codex",
    skills_repository="git@github.com:acme/skills.git",
    skills_revision=SkillsRevision("0" * 39 + "a"),
)


def _record() -> SessionRecord:
    return SessionRecord(
        id=new_session_id(),
        tenant_id=TenantId(uuid4()),
        definition_id=new_definition_id(),
        definition_revision="rev-1",
        grant=frozenset(),
        scope=(),
        budget_minor_units=10_000,
        budget_currency="USD",
        retention_days=30,
    )


def _compiled() -> CompiledConfig:
    return compile_session_config(
        _record(),
        tool_gateway_url=GATEWAY_URL,
        model_gateway_url=MODEL_GATEWAY_URL,
        definition=A_DEFINITION,
        environment=_a_shape_that_narrows_nothing(),
        session_token_key=_TOKEN_KEY,
        session_token_expiry_epoch_s=_EXPIRY,
    )


def _requirements() -> dict[str, Any]:
    return tomllib.loads(_compiled().requirements_toml)


def _profile_rules() -> dict[str, Any]:
    rules: dict[str, Any] = _requirements()["permissions"][PROFILE_NAME]["filesystem"]
    return rules


# --------------------------------------------------------------------------------------
# The document is the real thing
# --------------------------------------------------------------------------------------


def test_compiling_a_session_yields_a_configuration_that_holds_every_floor() -> None:
    check_floors(_compiled())


def test_both_documents_parse_as_toml_and_are_not_empty() -> None:
    compiled = _compiled()
    assert tomllib.loads(compiled.config_toml)
    assert tomllib.loads(compiled.requirements_toml)


def test_the_compiled_configuration_carries_the_session_it_was_compiled_for() -> None:
    record = _record()
    assert (
        compile_session_config(
            record,
            tool_gateway_url=GATEWAY_URL,
            model_gateway_url=MODEL_GATEWAY_URL,
            definition=A_DEFINITION,
            environment=_a_shape_that_narrows_nothing(),
            session_token_key=_TOKEN_KEY,
            session_token_expiry_epoch_s=_EXPIRY,
        ).session_id
        == record.id
    )


def test_the_gateways_tools_are_approved_so_a_call_to_one_is_not_refused() -> None:
    """The one line that decides whether a granted tool can actually be called.

    Measured on a live Turn before this existed: the model was offered the tool, called
    it, and reported it "blocked by the current approval policy" -- so the failure mode
    is not an error anywhere, it is the agent politely answering from memory instead.
    Nothing else in this suite would notice, because the document stays valid TOML and
    the Session still completes its Turn.

    `codex-mcp/src/mcp/mod.rs:96-105` auto-approves an MCP call only when
    `approval_policy` is never AND the Permission Profile is Disabled, External, or
    Managed with full disk write access. Ours is Managed and narrow, so that third arm
    is false by design; with nobody inside a pod to ask, the call is refused. This
    override is the only door that does not widen the sandbox.
    """
    server = tomllib.loads(_compiled().config_toml)["mcp_servers"][GATEWAY_SERVER_ID]
    assert server["default_tools_approval_mode"] == "approve", (
        f"the Gateway is registered with "
        f"{server.get('default_tools_approval_mode')!r}, so the runtime will offer its "
        "tools and refuse every call to one"
    )


def test_the_gateway_is_still_the_only_server_the_approval_applies_to() -> None:
    """The approval is per-server, so it must reach exactly the one server we vouch for.

    Written as an equality over the table's keys rather than a check on the Gateway's
    entry, because the risk is a second server arriving later and inheriting an approval
    nobody decided for it -- a tool the platform never clamped, called without asking.
    """
    servers = tomllib.loads(_compiled().config_toml)["mcp_servers"]
    assert set(servers) == {GATEWAY_SERVER_ID}, (
        f"{sorted(set(servers) - {GATEWAY_SERVER_ID})} also appear in mcp_servers; the "
        "approval above was decided for the Gateway alone"
    )


def test_the_compiled_configuration_carries_the_definitions_model() -> None:
    """The model is the definition's, and the only thing read off it.

    A Session runs whatever model the definition it pinned names, so this comes from the
    definition rather than from a platform constant -- unlike the provider below.
    """
    assert _compiled().model == A_DEFINITION.model


def test_the_provider_is_the_platforms_gateway_and_not_the_definitions_to_name() -> (
    None
):
    """Every model call leaves a Session pod through the Model Gateway.

    That gateway holds the provider credentials and is what the Egress Policy allows, so
    a provider named by whoever submitted the definition would be a request to an
    endpoint outside it, from a pod running model-driven code. Asserted structurally as
    well as by value: a definition has no field for it at all.
    """
    assert _compiled().model_provider == MODEL_PROVIDER_ID
    assert "model_provider" not in AgentDefinition.model_fields


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_configuration_naming_no_model_or_no_provider_is_refused(blank: str) -> None:
    """The floor for a failure that surfaces three layers from where it is set.

    Measured in `map-dev`: a pod whose shim reads these as empty strings pulls,
    initialises, and brings its runtime container to ready -- and then the shim's
    `thread/start` is answered `-32600`, the shim exits 3, and under
    `restartPolicy: Never` the pod sits at phase `Running` with one container dead. So
    an empty value does not fail where it is set; it fails looking like a pod that is
    slow to start. Refusing at compilation means it cannot be set at all.
    """
    with pytest.raises(FloorViolation, match="names no model,"):
        check_floors(replace(_compiled(), model=blank))
    with pytest.raises(FloorViolation, match="names no model_provider"):
        check_floors(replace(_compiled(), model_provider=blank))


def test_the_compiled_config_declares_the_provider_the_shim_will_name() -> None:
    """The document has to declare the provider, or no Session in it ever starts.

    This is the fact the empty-string floor above could not reach. The shim reads
    `model_provider` out of its environment and hands it to `thread/start`; the runtime
    resolves it against the `[model_providers]` tables of the configuration it loaded.
    Measured against codex-cli 0.149.0 in the pushed Session image, one file varied and
    nothing else: with the table absent the answer is `Model provider
    `map-model-gateway` not found` and the shim exits 3; with it present the runtime
    starts and reports the model.

    All three fields are asserted because the three fail differently, and `base_url`
    is the one the runtime does not enforce -- measured, a table carrying `name` alone
    loads and the runtime then addresses its model calls to its own default endpoint,
    which is a request out of a Session pod to somewhere no Egress Policy allowed.
    """
    compiled = _compiled()
    entry = tomllib.loads(compiled.config_toml)["model_providers"][MODEL_PROVIDER_ID]
    assert entry["base_url"] == MODEL_GATEWAY_URL
    assert entry["wire_api"] == "responses"
    assert entry["name"].strip()
    assert compiled.model_provider == MODEL_PROVIDER_ID


def test_a_provider_the_compiled_document_does_not_declare_is_refused() -> None:
    """The floor that can actually fail, which the empty-string one never could.

    The provider is a module constant, so the only non-empty value the old floor ever
    saw was the one the runtime went on to reject: it was green on 100% of the inputs
    that fail. This drives the two values apart -- the field says one provider, the
    document declares another -- which is the shape of the defect that shipped.
    """
    with pytest.raises(FloorViolation, match="declares no model provider"):
        check_floors(replace(_compiled(), model_provider="some-other-gateway"))


@pytest.mark.parametrize(
    ("field_name", "complaint"),
    [("name", "with no name"), ("base_url", "with no base_url")],
)
def test_a_provider_table_missing_a_required_field_is_refused(
    field_name: str, complaint: str
) -> None:
    """Each field separately, so a floor cannot pass on the strength of the others."""
    compiled = _compiled()
    stripped = "\n".join(
        line
        for line in compiled.config_toml.splitlines()
        if not line.startswith(f"{field_name} = ")
    )
    with pytest.raises(FloorViolation, match=complaint):
        check_floors(replace(compiled, config_toml=stripped))


def test_a_provider_table_on_the_wrong_wire_is_refused() -> None:
    """`responses` is the only wire either end speaks.

    The Agent Runtime's own deserializer accepts that one string and hard-rejects
    `"chat"` naming its removal -- measured in the pushed image, `Error loading
    config.toml: `wire_api = "chat"` is no longer supported.` And the Model Gateway
    registers a handler for that wire alone and answers the other two 502. So a document
    on any other wire is refused here rather than at whichever end reaches it first.
    """
    compiled = _compiled()
    rewired = compiled.config_toml.replace(
        'wire_api = "responses"', 'wire_api = "chat"'
    )
    with pytest.raises(FloorViolation, match="declares wire_api"):
        check_floors(replace(compiled, config_toml=rewired))


def test_the_rendered_rules_are_exactly_the_profiles_rules() -> None:
    """Whole-table equality, not a spot check.

    A rule the profile declares and the document omits is a rule the sandbox never
    enforces, and checking four interesting paths would not see the fifth going missing.
    """
    assert _profile_rules() == {r.path: r.access.value for r in session_profile().rules}


def test_the_managed_deny_read_list_is_exactly_the_profiles_deny_paths() -> None:
    deny_read = _requirements()["permissions"]["filesystem"]["deny_read"]
    assert deny_read == list(session_profile().denied())


def test_the_control_path_is_denied_by_the_rule_that_dominates_it() -> None:
    """The directory, and the socket NOT by name as well.

    Access resolves to the prefix-matching rule with the most path components, so the
    directory rule already denies the socket and a second rule at the leaf changes the
    resolved access for nothing. It did change the compiled argv, and fatally: once
    the socket exists, the pair makes bubblewrap refuse to build any sandbox, so no
    Session could run a confined command. The condition is load-bearing -- with the
    socket absent the same pair builds a sandbox fine, and the runtime binds the
    socket on every start.

    The directory is also the stronger mask: with only the leaf rule the parent is
    world-listable and the socket's own filename is readable out of it. See ADR-012's
    Status.

    Both halves are asserted, and in both lists. "The directory rule is present" alone
    would pass on a configuration that also named the leaf, which is the configuration
    this test exists to keep out; and the leaf absent from the profile table while
    still in the managed deny_read leaves the compiled argv unchanged, because every
    deny_read entry is pushed into the same policy the argv is built from.
    """
    rules = _profile_rules()
    assert rules[CONTROL_SOCKET_DIR] == FsAccess.DENY.value
    assert CONTROL_SOCKET not in rules
    assert CONTROL_SOCKET_DIR in _deny_read()
    assert CONTROL_SOCKET not in _deny_read()


def _deny_read() -> list[str]:
    read: list[str] = _requirements()["permissions"]["filesystem"]["deny_read"]
    return read


def _every_denied_path() -> set[str]:
    """The union of the two lists the sandbox policy is built from.

    The profile table is what a reader sees; `deny_read` is the copy no layer inside
    the pod can weaken. They are equal by construction today and a test above pins
    them so, which is exactly why the union is written out rather than assumed: a
    rendering that stopped keeping them equal would otherwise put half of every check
    below out of reach.
    """
    requirements = _requirements()
    table = requirements["permissions"][PROFILE_NAME]["filesystem"]
    denied = {path for path, access in table.items() if access == FsAccess.DENY.value}
    return denied | set(_deny_read())


def test_no_denied_path_lies_inside_another_denied_path() -> None:
    """Over the union of the profile table and the managed deny_read list.

    Both feed the one policy the sandbox argv is compiled from, so a nested pair in
    either is a Session that cannot run a confined command at all: the ancestor
    becomes a mode-000 tmpfs remounted read-only and the descendant's own operation is
    then attempted inside a filesystem already frozen.
    """
    denied = _every_denied_path()
    assert denied, "the configuration denies nothing, so this asserts nothing"
    assert nested_deny_pairs(sorted(denied)) == ()


def test_every_denied_path_is_spelled_the_one_way_the_check_above_can_compare() -> None:
    """The guard on the guard, and it is not tidiness.

    The nesting check compares path components, so it reads `/etc//codex/deeper` as
    inside `/etc/codex` the way a filesystem does. That makes the comparison right,
    and it would also make a weakened rendering invisible: a document naming a deny
    path in a spelling nothing else in the tree uses would be compared correctly and
    pass without anyone learning that the parse had stopped normalising. This is the
    clause that turns that into a loud failure instead.
    """
    denied = _every_denied_path()
    assert denied, "the configuration denies nothing, so this asserts nothing"
    for path in sorted(denied):
        assert path_spelling_error(path) is None, path


def test_the_nesting_floor_refuses_a_document_that_denies_nothing_at_all() -> None:
    """An empty deny set has no nested pair, so without this the floor would pass it.

    Driven straight at the floor rather than through `check_floors`, and the reason is
    worth writing down instead of leaving a reader to wonder whether the clause is
    dead code: in `check_floors` the control-path floor runs first and fires on the
    same document, so this clause is never the one that refuses a compiled
    configuration. It is reachable here, it is what the floor promises on its own
    terms, and it is what keeps the promise true if the two floors are ever reordered
    or if the control path stops being the thing that guarantees the set is non-empty.
    """
    with pytest.raises(FloorViolation, match="denies no path at all"):
        config_compiler._refuse_a_deny_nested_under_another(
            {
                "permissions": {
                    PROFILE_NAME: {"filesystem": {WORKSPACE_ROOT: "write"}},
                    "filesystem": {"deny_read": []},
                }
            }
        )


def test_the_profile_still_nests_two_denies_under_the_writable_root() -> None:
    """The floor above is deny-under-deny and must never become rule-under-rule.

    This profile carves two holes in its one writable root on purpose, and that
    nesting is measured working -- mask mode `d---------`, with a read of seeded bytes
    and a write both refused. A check that did not distinguish the two would fail here
    on day one, which is why this case sits beside it rather than in another file.
    """
    rules = {rule.path: rule.access for rule in session_profile().rules}
    assert rules[WORKSPACE_ROOT] is FsAccess.WRITE
    for name in (".codex", ".agents"):
        assert rules[f"{WORKSPACE_ROOT}/{name}"] is FsAccess.DENY
        assert is_strictly_under(f"{WORKSPACE_ROOT}/{name}", WORKSPACE_ROOT)


@pytest.mark.parametrize(
    ("child", "parent", "nested"),
    [
        (f"{WORKSPACE_ROOT}/a/b", f"{WORKSPACE_ROOT}/a", True),
        (f"{WORKSPACE_ROOT}/a", f"{WORKSPACE_ROOT}/a", False),
        (f"{WORKSPACE_ROOT}/.agentsX", f"{WORKSPACE_ROOT}/.agents", False),
        ("/run/codexx", "/run/codex", False),
        (CONTROL_SOCKET, CONTROL_SOCKET_DIR, True),
        ("/run/codex", "/", True),
        ("/", "/", False),
        ("/run//codex/x", "/run/codex", True),
        ("/run/./codex/x", "/run/codex", True),
        ("/run/codex/", "/run/codex", False),
        ("/run/codex/x", "/run/codex/", True),
        ("/run/codex/x", "run/codex", False),
    ],
)
def test_the_predicate_reads_the_components_and_not_the_spelling(
    child: str, parent: str, nested: bool
) -> None:
    """Six traps in one table, and the last five are why this compares components.

    A bare `startswith` nests `/run/codexx` under `/run/codex` and nests a path under
    itself. `parent + "/"` spells `"//"` for the root and so makes the root nobody's
    parent -- and `FsRule` accepts the root, so a platform profile can reach that case
    even though no tenant can write it. And the string form that fixes both of those,
    `child.startswith(parent.rstrip("/") + "/")`, still answers three of these rows
    wrongly: it misses `/run//codex/x` and `/run/./codex/x`, each of which names a
    path plainly inside its parent, and it reports `/run/codex/` as nested under
    `/run/codex`, which is the same directory.

    Those three are not hypothetical spellings nobody would write -- they are what a
    rendering bug produces, and the floors grade the rendered document precisely
    because the Python value and the document can disagree. A guard that graded one
    spelling of them is this repository's most expensive recurring defect.
    """
    assert is_strictly_under(child, parent) is nested


def test_the_credential_bearing_config_directory_is_denied() -> None:
    """`$CODEX_HOME/config.toml` carries this Session's bearer token and the provider
    base URL, so an agent that reads it can spend the tenant's budget under its own
    name. The directory is denied rather than the file, because a rule over the
    directory already denies every path beneath it and is the stronger mask."""
    assert _profile_rules()[CODEX_HOME] == FsAccess.DENY.value


def test_the_skills_config_directory_is_readable_and_holds_no_credential() -> None:
    """The asymmetry with the test above, and why it is not a hole.

    `/etc/codex` holds the managed `requirements.toml` and the Session's skills. It is
    readable because a Host skill's catalogue line hands the model a file path and
    expects it to open the file itself, and the tool route that would read it for the
    model is inert without an orchestrator provider our pod does not have -- so while
    this was denied, every skill was delivered, catalogued, named to the model, and
    unreadable by it.

    What makes that safe is the second assertion, not the first: the document the agent
    can now read is these permission rules plus an in-cluster URL, and the kernel
    enforces the rules whether or not the confined process can read them. If a token
    or a key is ever rendered into this document, this test fails and the rule above
    has to come back.
    """
    assert _profile_rules()[SYSTEM_CONFIG_DIR] == FsAccess.READ.value
    document = _compiled().requirements_toml.lower()
    for smell in ("token", "authorization", "bearer", "api_key", "secret", "passwo"):
        assert smell not in document, f"{smell!r} is rendered into a readable document"


def test_workspace_metadata_is_denied_before_it_exists() -> None:
    """A missing deny path inside a writable root is masked at its first absent
    component, so the confined process cannot create the hierarchy at all -- which is
    what makes configuration written into the workspace inert rather than ignored."""
    rules = _profile_rules()
    assert rules[WORKSPACE_ROOT] == FsAccess.WRITE.value
    assert rules[f"{WORKSPACE_ROOT}/.codex"] == FsAccess.DENY.value
    assert rules[f"{WORKSPACE_ROOT}/.agents"] == FsAccess.DENY.value


def test_the_workspace_is_the_only_writable_prefix() -> None:
    assert session_profile().writable() == (WORKSPACE_ROOT,)


def test_the_profile_extends_the_read_only_built_in() -> None:
    """Not ':workspace': what is writable must be the prefix named here, and not
    whatever the runtime counts as a workspace root on the day."""
    assert _requirements()["permissions"][PROFILE_NAME]["extends"] == ":read-only"


def test_profile_mode_is_forced_and_this_profile_is_the_one_selected() -> None:
    """The presence that makes every absence in this file worth checking.

    A stray older sandbox key in any layer of the loaded chain downgrades the whole
    enforcement layer with no error, and a managed allowed_permission_profiles is the
    one thing that overrides that.
    """
    requirements = _requirements()
    assert requirements["default_permissions"] == PROFILE_NAME
    assert requirements["allowed_permission_profiles"] == {PROFILE_NAME: True}


def test_the_workspace_is_declared_and_marked_untrusted() -> None:
    projects = tomllib.loads(_compiled().config_toml)["projects"]
    assert projects == {WORKSPACE_ROOT: {"trust_level": "untrusted"}}


def test_only_the_tool_gateway_is_named_and_it_is_named() -> None:
    compiled = _compiled()
    for document in (compiled.config_toml, compiled.requirements_toml):
        assert tuple(tomllib.loads(document)["mcp_servers"]) == (GATEWAY_SERVER_ID,)


def test_the_gateway_url_reaches_both_documents() -> None:
    compiled = _compiled()
    config = tomllib.loads(compiled.config_toml)["mcp_servers"][GATEWAY_SERVER_ID]
    managed = tomllib.loads(compiled.requirements_toml)["mcp_servers"][
        GATEWAY_SERVER_ID
    ]
    assert config["url"] == GATEWAY_URL
    assert managed["identity"]["url"] == GATEWAY_URL


def test_the_runtime_is_launched_on_a_unix_socket_and_nothing_else() -> None:
    assert _compiled().launch_argv == (
        "codex",
        "app-server",
        "--listen",
        f"unix://{CONTROL_SOCKET}",
    )


def test_neither_document_carries_the_older_sandbox_settings() -> None:
    compiled = _compiled()
    for text in (compiled.config_toml, compiled.requirements_toml):
        assert "sandbox_mode" not in text
        assert "sandbox_workspace_write" not in text


# --------------------------------------------------------------------------------------
# The checker refuses
# --------------------------------------------------------------------------------------


def _drop_socket_dir_deny(text: str) -> str:
    return text.replace(f'"{CONTROL_SOCKET_DIR}" = "deny"\n', "")


def _swap_the_control_path_deny_for_the_leaf_alone(text: str) -> str:
    """The one configuration a relaxed control-path floor would let through.

    Naming only the socket is not a smaller version of naming its directory, it is a
    weaker one: measured in the cluster, with only the leaf rule `/run/codex` is
    `drwxrwsrwx` and world-listable, the socket's exact filename is readable out of
    that listing and lstat-able, and the directory is reachable through `/proc` as
    well. With only the directory rule every one of those is EACCES. So the floor must
    require the DOMINATING rule specifically, and this row is what fails if it is ever
    relaxed to "either control-path rule will do" -- a relaxation no other case in this
    file can see, because it changes what the floor accepts and not what the compiler
    emits.
    """
    swapped = text.replace(f'"{CONTROL_SOCKET_DIR}"', f'"{CONTROL_SOCKET}"')
    assert swapped != text, "the document names the control path directory nowhere"
    return swapped


def _nest_a_deny_in_the_profile_table(text: str) -> str:
    """A second deny rule one segment under one the platform already names.

    In the table only, which is the arm a reader would think of first.
    """
    line = f'"{CODEX_HOME}" = "deny"'
    assert line in text, "the profile table does not name the codex home"
    return text.replace(line, f'{line}\n"{CODEX_HOME}/deeper" = "deny"')


def _nest_a_deny_in_the_managed_deny_read_only(text: str) -> str:
    """The other arm, and the reason the floor reads a union.

    `deny_read` is rendered from the profile's own deny paths and a test above pins the
    two lists equal, so no mutation that goes through `session_profile()` can ever
    reach this arm. It is written as raw text for exactly that reason: without it, half
    of the floor could not fail and nobody would know.
    """
    needle = f'"{CODEX_HOME}"]'
    assert needle in text, "the deny_read list does not end where this expects"
    return text.replace(needle, f'"{CODEX_HOME}", "{CODEX_HOME}/deeper"]')


def _deny_by_an_access_value_this_module_cannot_classify(text: str) -> str:
    """A rule that denies, written in a value the deny-set selection does not recognise.

    Measured against codex-cli 0.149.0 in the cluster: `"none"` is accepted and denies
    -- a confined `ls` of the path carrying it was refused -- while `"Deny"` and
    `"DENY"` make the runtime refuse the document outright. So the dangerous spelling
    is not a case variant; it is a fourth word that means the same thing, and a
    selection asking `== "deny"` would report this row as not-denied and hand every
    check below a deny set with a hole in it.
    """
    line = f'"{CODEX_HOME}" = "deny"'
    assert line in text, "the profile table does not name the codex home"
    return text.replace(line, f'"{CODEX_HOME}" = "none"', 1)


def _nest_a_deny_in_a_spelling_a_string_compare_would_miss(text: str) -> str:
    """The same nesting, written the way a rendering bug would write it.

    `/var/lib/map//codex/deeper` names a path inside `/var/lib/map/codex` and does not
    start with that string plus a slash. The floor refuses it on the spelling, before
    the comparison -- which is the point: the spelling is the fact the floor can be
    certain of, and refusing it is what keeps the comparison's own input to the one
    form the rest of the tree writes.
    """
    line = f'"{CODEX_HOME}" = "deny"'
    assert line in text, "the profile table does not name the codex home"
    head, _, tail = CODEX_HOME.rpartition("/")
    return text.replace(line, f'{line}\n"{head}//{tail}/deeper" = "deny"')


def _empty_deny_read(text: str) -> str:
    return re.sub(r"deny_read = \[[^\]]*\]", "deny_read = []", text)


def _rename_the_profile_filesystem_table(text: str) -> str:
    """Rename rather than truncate: cutting the tail off would also take the MCP server
    table with it, and the run would then be refused for the wrong floor."""
    return text.replace(
        f'[permissions."{PROFILE_NAME}".filesystem]',
        f'[permissions."{PROFILE_NAME}".elsewhere]',
    )


def _drop_the_projects_table(text: str) -> str:
    return text.replace(
        f'[projects."{WORKSPACE_ROOT}"]\ntrust_level = "untrusted"\n', ""
    )


def _workspace_read_only(text: str) -> str:
    return text.replace(f'"{WORKSPACE_ROOT}" = "write"', f'"{WORKSPACE_ROOT}" = "read"')


def _header_line(text: str, key: str = SESSION_TOKEN_HEADER_NAME) -> str:
    """The one `http_headers` line whose table names `key`, in a rendered document.

    Found rather than reconstructed: a helper that rebuilt the expected line would
    silently stop matching the day the renderer changed its spacing, and every mutation
    below would then apply to nothing and pass vacuously.

    Selected by the header it names rather than by being the only one, which it stopped
    being when the model provider table gained a bearer of its own. A helper that still
    asserted uniqueness would fail every mutation below on the wrong document -- and one
    that took the first match would quietly move all of them onto whichever table the
    renderer happened to emit first.
    """
    matches = [
        line
        for line in text.splitlines()
        if line.startswith("http_headers = ") and f'"{key}"' in line
    ]
    assert len(matches) == 1, f"{len(matches)} http_headers lines naming {key!r}"
    return matches[0]


def _drop_the_header(text: str) -> str:
    return text.replace(_header_line(text) + "\n", "")


def _misspell_the_header_key(text: str) -> str:
    """`headers` instead of `http_headers`.

    Measured against codex-cli 0.149.0: the runtime parses this, does nothing with it,
    reports the server `"enabled": true` with no warning, and sends the request with no
    header. Only `--strict-config` names it, and this deployment does not pass that
    flag.
    """
    return text.replace("http_headers = ", "headers = ")


def _empty_the_token(text: str) -> str:
    """Measured: an empty value IS put on the wire, as an empty header. So this is not
    the same failure as a dropped one, and both end in the same fixed 401."""
    line = _header_line(text)
    return text.replace(
        line, f'http_headers = {{ "{SESSION_TOKEN_HEADER_NAME}" = "" }}'
    )


def _put_a_newline_in_the_token(text: str) -> str:
    """A value the runtime cannot put in a header. Measured: it warns and continues, so
    the request goes out with no header at all -- indistinguishable from the dropped
    line above, from anywhere outside the pod."""
    line = _header_line(text)
    return text.replace(
        line,
        f'http_headers = {{ "{SESSION_TOKEN_HEADER_NAME}" = "bad\\nvalue" }}',
    )


def _truncate_the_signature(text: str) -> str:
    line = _header_line(text)
    token = line.split('"')[3]
    return text.replace(line, line.replace(token, token[:-1]))


def _add_a_second_header(text: str) -> str:
    """A second header is a second thing the pod asserts about itself, and nothing in
    this compiler writes one."""
    line = _header_line(text)
    return text.replace(line, line[: -len(" }")] + ', "x-map-extra" = "1" }')


def _name_another_session(text: str) -> str:
    return _rename_token_part(text, 0)


def _name_another_tenant(text: str) -> str:
    """Leave the Session right and swap only the tenant.

    This is the mutation the floor was blind to until 2026-08-23: it graded part 0 and
    never part 1, so a token naming this pod's own Session under somebody else's tenant
    passed every check and would have made the Gateway lend out that tenant's
    credentials. The Session being correct is what makes it dangerous -- nothing else
    downstream disagrees.
    """
    return _rename_token_part(text, 1)


def _rename_token_part(
    text: str, index: int, key: str = SESSION_TOKEN_HEADER_NAME
) -> str:
    line = _header_line(text, key)
    value = line.split('"')[3]
    # `rpartition` and not `partition`: it yields ("", "", value) when there is no
    # space, so the Tool Gateway's bare token needs no branch of its own, and the
    # scheme is put back verbatim so this mutation changes the identity and nothing
    # else. Dropping it here would make both cases fail on the scheme instead.
    prefix, space, token = value.rpartition(" ")
    parts = token.split(".")
    parts[index] = "99999999-9999-9999-9999-999999999999"
    renamed = prefix + space + ".".join(parts)
    return text.replace(line, line.replace(value, renamed))


# --- the same eight mutations, on the model provider's bearer ----------------------
#
# Spelled out rather than generated from the block above, because what differs between
# the two is not a parameter: this header carries a scheme, so a mutation that produces
# a legal token under a broken scheme has no counterpart on the Tool Gateway side, and
# a loop over both would have to grow a branch for it and stop being a loop.


def _model_header_line(text: str) -> str:
    return _header_line(text, MODEL_PROVIDER_AUTH_HEADER)


def _drop_the_model_header(text: str) -> str:
    return text.replace(_model_header_line(text) + "\n", "")


def _misspell_the_model_header_key(text: str) -> str:
    line = _model_header_line(text)
    return text.replace(
        line, line.replace(f'"{MODEL_PROVIDER_AUTH_HEADER}"', '"authorisation"')
    )


def _add_a_second_model_header(text: str) -> str:
    line = _model_header_line(text)
    return text.replace(line, line[: -len(" }")] + ', "x-map-extra" = "1" }')


def _model_bearer(value: str) -> str:
    return f'http_headers = {{ "{MODEL_PROVIDER_AUTH_HEADER}" = "{value}" }}'


def _empty_the_model_token(text: str) -> str:
    return text.replace(_model_header_line(text), _model_bearer("Bearer "))


def _drop_the_bearer_scheme(text: str) -> str:
    """The token alone, correct in every other way.

    Measured against codex-cli 0.149.0: a bare token IS put on the wire as the whole
    `authorization` value, so this is not a dropped header. The Model Gateway parses
    the scheme off before it verifies and finds none, and answers the same fixed 401 it
    answers an unsigned token -- so from outside the pod a scheme this compiler forgot
    to write is indistinguishable from a key mismatch.
    """
    line = _model_header_line(text)
    return text.replace(line, _model_bearer(line.split('"')[3].removeprefix("Bearer ")))


def _use_another_scheme(text: str) -> str:
    """`Token` rather than `Bearer`, which the Gateway does not accept."""
    line = _model_header_line(text)
    token = line.split('"')[3].removeprefix("Bearer ")
    return text.replace(line, _model_bearer(f"Token {token}"))


def _put_a_newline_in_the_model_token(text: str) -> str:
    return text.replace(_model_header_line(text), _model_bearer("Bearer bad\\nvalue"))


def _truncate_the_model_signature(text: str) -> str:
    line = _model_header_line(text)
    value = line.split('"')[3]
    return text.replace(line, line.replace(value, value[:-1]))


def _name_another_session_to_the_model_gateway(text: str) -> str:
    return _rename_token_part(text, 0, MODEL_PROVIDER_AUTH_HEADER)


def _name_another_tenant_to_the_model_gateway(text: str) -> str:
    """The mutation with the most reach, and the reason this whole block exists.

    A token naming this pod's own Session under another tenant is a document that
    works. It verifies, and the Model Gateway resolves the caller and then bills the
    provider credential selected by the tenant the token named. So a compiler that
    rendered the wrong tenant here is unauthorized spend with a valid signature on it,
    and nothing downstream disagrees with it.
    """
    return _rename_token_part(text, 1, MODEL_PROVIDER_AUTH_HEADER)


_DOCUMENT_MUTATIONS: list[tuple[str, Callable[[str], str], str]] = [
    (
        "config_toml",
        lambda t: t.replace('trust_level = "untrusted"', 'trust_level = "trusted"'),
        "not marked untrusted",
    ),
    (
        "config_toml",
        lambda t: t + '\n[projects."/session/other"]\ntrust_level = "trusted"\n',
        "not marked untrusted",
    ),
    (
        "config_toml",
        lambda t: t.replace('trust_level = "untrusted"\n', ""),
        "not marked untrusted",
    ),
    ("config_toml", _drop_the_projects_table, "no projects table"),
    (
        "config_toml",
        lambda t: t + '\n[mcp_servers.extra]\nurl = "https://x/mcp"\n',
        "MCP servers",
    ),
    (
        "config_toml",
        lambda t: t + '\n[profiles.other]\nsandbox_mode = "workspace-write"\n',
        "older sandbox settings",
    ),
    ("requirements_toml", _drop_socket_dir_deny, "no deny rule over"),
    (
        "requirements_toml",
        _swap_the_control_path_deny_for_the_leaf_alone,
        "no deny rule over",
    ),
    ("requirements_toml", _nest_a_deny_in_the_profile_table, "the deny set nests"),
    (
        "requirements_toml",
        _nest_a_deny_in_the_managed_deny_read_only,
        "the deny set nests",
    ),
    (
        "requirements_toml",
        _nest_a_deny_in_a_spelling_a_string_compare_would_miss,
        "the deny set names",
    ),
    (
        "requirements_toml",
        _deny_by_an_access_value_this_module_cannot_classify,
        "which is not one of",
    ),
    ("requirements_toml", _workspace_read_only, "not writable"),
    ("requirements_toml", _empty_deny_read, "deny_read list is empty"),
    ("requirements_toml", _rename_the_profile_filesystem_table, "no permissions"),
    (
        "requirements_toml",
        lambda t: t + '\n[mcp_servers.extra]\nidentity = { url = "https://x/mcp" }\n',
        "MCP servers",
    ),
    (
        "requirements_toml",
        lambda t: t.replace(
            f'[allowed_permission_profiles]\n"{PROFILE_NAME}" = true\n',
            "[allowed_permission_profiles]\n",
        ),
        "nothing forces profile mode",
    ),
    (
        "requirements_toml",
        lambda t: t.replace(
            f'default_permissions = "{PROFILE_NAME}"',
            'default_permissions = ":workspace"',
        ),
        "default_permissions",
    ),
    (
        "requirements_toml",
        lambda t: t + "\n[sandbox_workspace_write]\nnetwork_access = true\n",
        "older sandbox settings",
    ),
    ("config_toml", _drop_the_header, "must name exactly"),
    ("config_toml", _misspell_the_header_key, "must name exactly"),
    ("config_toml", _add_a_second_header, "must name exactly"),
    ("config_toml", _empty_the_token, "not shaped like a session token"),
    ("config_toml", _put_a_newline_in_the_token, "not shaped like a session token"),
    ("config_toml", _truncate_the_signature, "not shaped like a session token"),
    ("config_toml", _name_another_session, "names session"),
    ("config_toml", _name_another_tenant, "names tenant"),
    ("config_toml", _drop_the_model_header, "must name exactly"),
    ("config_toml", _misspell_the_model_header_key, "must name exactly"),
    ("config_toml", _add_a_second_model_header, "must name exactly"),
    ("config_toml", _drop_the_bearer_scheme, "does not begin with 'Bearer '"),
    ("config_toml", _use_another_scheme, "does not begin with 'Bearer '"),
    ("config_toml", _empty_the_model_token, "not shaped like a session token"),
    ("config_toml", _put_a_newline_in_the_model_token, "not shaped like a session"),
    ("config_toml", _truncate_the_model_signature, "not shaped like a session token"),
    ("config_toml", _name_another_session_to_the_model_gateway, "names session"),
    ("config_toml", _name_another_tenant_to_the_model_gateway, "names tenant"),
    ("requirements_toml", lambda t: t + "\nthis is not toml\n", "does not parse"),
]


@pytest.mark.parametrize(("field", "mutate", "refusal"), _DOCUMENT_MUTATIONS)
def test_check_floors_refuses_a_mutated_document(
    field: str, mutate: Callable[[str], str], refusal: str
) -> None:
    compiled = _compiled()
    original: str = getattr(compiled, field)
    mutated = mutate(original)
    mutant = (
        replace(compiled, config_toml=mutated)
        if field == "config_toml"
        else replace(compiled, requirements_toml=mutated)
    )

    assert mutant != compiled, f"the mutation of {field} changed nothing"
    with pytest.raises(FloorViolation, match=re.escape(refusal)):
        check_floors(mutant)


# --------------------------------------------------------------------------------------
# The document carries this Session's token, and the Gateway's own reader accepts it
# --------------------------------------------------------------------------------------


def _token_in(compiled: CompiledConfig) -> str:
    """The token as the runtime would read it: parsed back out of the document."""
    parsed = tomllib.loads(compiled.config_toml)
    headers = parsed["mcp_servers"][GATEWAY_SERVER_ID]["http_headers"]
    assert set(headers) == {SESSION_TOKEN_HEADER_NAME}
    value: str = headers[SESSION_TOKEN_HEADER_NAME]
    return value


def test_the_compiled_config_carries_a_token_for_the_session_it_was_compiled_for() -> (
    None
):
    record = _record()
    compiled = compile_session_config(
        record,
        tool_gateway_url=GATEWAY_URL,
        model_gateway_url=MODEL_GATEWAY_URL,
        definition=A_DEFINITION,
        environment=_a_shape_that_narrows_nothing(),
        session_token_key=_TOKEN_KEY,
        session_token_expiry_epoch_s=_EXPIRY,
    )

    assert _token_in(compiled).split(".")[0] == str(record.id)


def test_the_token_in_the_document_verifies_under_the_key_it_was_minted_with() -> None:
    """The loop closed inside the default gate: the document a pod is started from
    carries a token the Gateway's own reader accepts, naming this Session and this
    tenant.

    `verify_session_token` is the real one the Tool Gateway's middleware calls. A
    re-derivation here would assert that this file agrees with itself.
    """
    record = _record()
    compiled = compile_session_config(
        record,
        tool_gateway_url=GATEWAY_URL,
        model_gateway_url=MODEL_GATEWAY_URL,
        definition=A_DEFINITION,
        environment=_a_shape_that_narrows_nothing(),
        session_token_key=_TOKEN_KEY,
        session_token_expiry_epoch_s=_EXPIRY,
    )

    context = verify_session_token(_token_in(compiled), _TOKEN_KEY, _EXPIRY - 1)
    assert context.session_id == record.id
    assert context.tenant_id == record.tenant_id


def test_two_sessions_of_one_tenant_get_tokens_that_are_not_each_others() -> None:
    """Asserted rather than assumed. Two pods of one tenant holding one token would
    make the Session half of the Gateway's key meaningless, and nothing downstream
    would notice: both tokens verify."""
    tenant = TenantId(uuid4())
    tokens = [
        _token_in(
            compile_session_config(
                replace(_record(), tenant_id=tenant),
                tool_gateway_url=GATEWAY_URL,
                model_gateway_url=MODEL_GATEWAY_URL,
                definition=A_DEFINITION,
                environment=_a_shape_that_narrows_nothing(),
                session_token_key=_TOKEN_KEY,
                session_token_expiry_epoch_s=_EXPIRY,
            )
        )
        for _ in range(2)
    ]

    assert tokens[0] != tokens[1]
    read = [verify_session_token(token, _TOKEN_KEY, _EXPIRY - 1) for token in tokens]
    assert read[0].session_id != read[1].session_id
    assert read[0].tenant_id == read[1].tenant_id == tenant


def test_the_expiry_the_caller_names_is_the_expiry_in_the_document() -> None:
    """The argument is not silently replaced by a constant, and the expiry is absolute
    rather than a lifetime added to some clock inside the compiler."""
    chosen = _EXPIRY - 1_000_000
    compiled = compile_session_config(
        _record(),
        tool_gateway_url=GATEWAY_URL,
        model_gateway_url=MODEL_GATEWAY_URL,
        definition=A_DEFINITION,
        environment=_a_shape_that_narrows_nothing(),
        session_token_key=_TOKEN_KEY,
        session_token_expiry_epoch_s=chosen,
    )

    assert _token_in(compiled).split(".")[2] == str(chosen)


def test_a_token_minted_under_another_key_does_not_verify_under_the_gateways() -> None:
    """The key is the caller's, and the compiler cannot check a signature it just made.
    So this is the case that says a wrong key at the control plane is invisible until
    the Gateway refuses -- which is why the floor checks shape and identity only."""
    compiled = compile_session_config(
        _record(),
        tool_gateway_url=GATEWAY_URL,
        model_gateway_url=MODEL_GATEWAY_URL,
        definition=A_DEFINITION,
        environment=_a_shape_that_narrows_nothing(),
        session_token_key=b"a different key, thirty-two bytes",
        session_token_expiry_epoch_s=_EXPIRY,
    )

    with pytest.raises(InvalidSessionToken):
        verify_session_token(_token_in(compiled), _TOKEN_KEY, _EXPIRY - 1)


_ARGV_MUTATIONS: list[tuple[tuple[str, ...], str]] = [
    (("codex", "app-server", "--listen", "ws://0.0.0.0:8080"), "not on a unix socket"),
    (
        ("codex", "app-server", "--listen", f"unix://{CONTROL_SOCKET}", "--sandbox"),
        "carries",
    ),
    (
        (
            "codex",
            "app-server",
            "--listen",
            f"unix://{CONTROL_SOCKET}",
            "--sandbox=read-only",
        ),
        "carries",
    ),
    (("codex", "app-server"), "exactly once"),
    (
        (
            "codex",
            "app-server",
            "--listen",
            f"unix://{CONTROL_SOCKET}",
            "--listen",
            "tcp://0.0.0.0:8080",
        ),
        "exactly once",
    ),
    (("codex", "app-server", "--listen"), "names no address"),
]


@pytest.mark.parametrize(("argv", "refusal"), _ARGV_MUTATIONS)
def test_check_floors_refuses_a_mutated_argv(
    argv: tuple[str, ...], refusal: str
) -> None:
    with pytest.raises(FloorViolation, match=re.escape(refusal)):
        check_floors(replace(_compiled(), launch_argv=argv))


def test_a_forbidden_key_is_caught_at_any_depth() -> None:
    """Nesting is the point: the key is legal at top level, inside a named config
    profile and inside a table this platform does not write, and a root-only check
    would pass a document that downgrades the whole enforcement layer."""
    compiled = _compiled()
    nested = (
        compiled.config_toml
        + f'\n[projects."{WORKSPACE_ROOT}".nested.deeper]\n'
        + 'sandbox_mode = "read-only"\n'
    )
    with pytest.raises(FloorViolation, match="older sandbox settings"):
        check_floors(replace(compiled, config_toml=nested))


def test_a_profile_missing_the_control_socket_deny_yields_no_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The check runs on the way out of compilation, not only from this file.

    So a compiler that stopped denying the control path fails to produce a configuration
    at all, rather than producing one that is graded later by whoever remembers to.
    """
    weakened = PermissionProfile(
        name=PROFILE_NAME,
        extends=":read-only",
        rules=(FsRule(path=WORKSPACE_ROOT, access=FsAccess.WRITE),),
    )
    monkeypatch.setattr(config_compiler, "session_profile", lambda: weakened)

    with pytest.raises(FloorViolation, match="no deny rule over"):
        compile_session_config(
            _record(),
            tool_gateway_url=GATEWAY_URL,
            model_gateway_url=MODEL_GATEWAY_URL,
            definition=A_DEFINITION,
            environment=_a_shape_that_narrows_nothing(),
            session_token_key=_TOKEN_KEY,
            session_token_expiry_epoch_s=_EXPIRY,
        )


# --------------------------------------------------------------------------------------
# The pod manifest names the same paths
# --------------------------------------------------------------------------------------


def _pod() -> dict[str, Any]:
    loaded: Any = yaml.safe_load(POD_TEMPLATE.read_text())
    assert isinstance(loaded, dict), "the pod manifest did not parse as a mapping"
    return loaded


def _all_containers() -> list[dict[str, Any]]:
    spec = _pod()["spec"]
    containers: list[dict[str, Any]] = [
        *spec.get("initContainers", []),
        *spec.get("containers", []),
    ]
    return containers


def _init_script() -> str:
    script: str = _pod()["spec"]["initContainers"][0]["args"][0]
    return script


_SCRATCH_PATH = "/tmp"
"""The system temporary directory, which is where the sandbox helper builds its
registry of synthetic mount targets before every confined command."""


def _workspace_deny_paths() -> set[str]:
    """Every path the profile denies strictly under the writable root.

    Computed from the profile rather than written out, because the manifest and the
    compiler are two artifacts that must name the same paths and a literal pair here
    would be a third place for them to disagree.
    """
    return {
        rule.path
        for rule in session_profile().rules
        if rule.access is FsAccess.DENY and rule.path.startswith(f"{WORKSPACE_ROOT}/")
    }


def test_the_pod_manifest_parses_and_declares_both_stages() -> None:
    """Guard the guard: every loop below is over something read out of this file, and
    a manifest that failed to parse into the expected shape would make them vacuous."""
    spec = _pod()["spec"]
    # By name and in order rather than by count, because the ORDER is the load-bearing
    # part: init containers run in sequence, and `restore-working-lane` writes into
    # /session/workspace/.codex's parent only after `seed-runtime-home` has created that
    # directory and `.agents` beside it -- the two targets bwrap refuses to build a
    # sandbox without. A count says nothing about which came first.
    #
    # `seed-rollout` is third for a reason of the same kind: it writes into a second
    # volume, and a workspace restore that is going to refuse should refuse before that
    # happens. A refusal stops every container behind it, so the cheaper and likelier
    # failure goes first.
    assert [container["name"] for container in spec["initContainers"]] == [
        "seed-runtime-home",
        "restore-working-lane",
        "seed-rollout",
    ]
    assert [container["name"] for container in spec["containers"]] == [
        "agent-runtime",
        "session-shim",
    ]
    assert len(spec["volumes"]) == 7


def test_every_path_the_profile_names_lives_in_a_volume_the_pod_mounts() -> None:
    """The drift check the two files exist on either side of.

    A deny rule whose target the pod never materialises is a rule the sandbox may
    compile to nothing -- and the compiler cannot see that, because the pod is the
    other artifact.
    """
    mounts = {
        mount["mountPath"]
        for container in _all_containers()
        for mount in container.get("volumeMounts", [])
    }
    assert mounts, "no container mounts anything"

    for rule in session_profile().rules:
        assert any(
            rule.path == mount or is_strictly_under(rule.path, mount)
            for mount in mounts
        ), f"{rule.path} is a {rule.access.value} rule under no mounted volume"


def test_the_init_container_creates_every_directory_the_runtime_needs() -> None:
    made = [
        line.split()[2:]
        for line in _init_script().splitlines()
        if line.strip().startswith("mkdir ")
    ]
    assert made, "the init container creates no directory"
    created = {path for argv in made for path in argv}
    assert {CODEX_HOME, CONTROL_SOCKET_DIR, WORKSPACE_ROOT} <= created


def test_the_sandbox_helper_has_somewhere_to_write_its_mount_registry() -> None:
    """The container that runs bwrap can write under the system temporary directory,
    and no other container can.

    Before every confined command the sandbox helper creates a registry of synthetic
    mount targets there and panics if the mkdir fails, so on a read-only root with
    nothing mounted no tool call runs at all. The helper runs in the container outside
    the sandbox, which is why this grants the confined process nothing -- the profile
    extends `:read-only` and only the workspace is writable, so bwrap ro-binds this
    path like the rest of the filesystem.

    Asserted as an exact set rather than as "agent-runtime has it", because the point
    is as much about the containers that must NOT: an emptyDir carries no sticky bit,
    so a second mounting container is a second uid able to unlink the first's files.
    """
    mounting = {
        container["name"]
        for container in _all_containers()
        for mount in container.get("volumeMounts", [])
        if mount["mountPath"] == _SCRATCH_PATH
    }
    assert mounting == {"agent-runtime"}


def test_no_container_redirects_the_system_temporary_directory() -> None:
    """The other route to the same panic, and the reason the case above is not enough
    on its own.

    The helper writes under whatever `std::env::temp_dir()` resolves to, which is the
    value of TMPDIR when one is set and only otherwise the path the case above asserts
    is mounted. So a TMPDIR pointing anywhere else makes that mount inert while it
    still reads as present -- and the redirect is not hypothetical: it is the remedy
    that looks cheapest, and it was measured on map-dev as worse than the panic. The
    runtime refuses to place its arg0 helper binaries under a temporary directory, logs
    `Refusing to create helper binaries under temporary dir`, and proceeds degraded
    rather than failing.
    """
    for container in _all_containers():
        named = {entry["name"] for entry in container.get("env", [])}
        assert not named & {"TMPDIR", "TMP", "TEMP"}, container["name"]


def test_the_scratch_volume_is_bounded_and_on_the_node_disk() -> None:
    """Bounded because an unbounded emptyDir can reach the node's own eviction
    threshold and take its neighbours down; on disk because a tmpfs would charge a
    directory measured at zero bytes to the pod's memory cgroup, which is the resource
    the pods-per-node figure was measured against.

    The bound is a blast radius and not a capacity estimate: kubelet enforces it by
    polling, so crossing it evicts this pod rather than failing the write.

    Looked up through the mount rather than by volume name, so that removing the mount
    and leaving the volume fails here too instead of passing on a volume nothing uses.
    """
    mount = next(
        mount
        for container in _all_containers()
        for mount in container.get("volumeMounts", [])
        if mount["mountPath"] == _SCRATCH_PATH
    )
    volume = next(
        volume
        for volume in _pod()["spec"]["volumes"]
        if volume["name"] == mount["name"]
    )
    assert "sizeLimit" in volume["emptyDir"]
    assert "medium" not in volume["emptyDir"]


def test_the_init_container_creates_every_workspace_deny_target_as_a_directory() -> (
    None
):
    """Derived from the profile, so a fourth workspace deny rule fails here instead of
    failing every confined command in every Session.

    Why a directory and not an empty file, which is what the same instruction used for
    the control-socket path: the runtime protects `.git`, `.agents` and `.codex` under
    every writable root on its own account, and for a path with one of those basenames
    that does not exist it masks a directory while this platform's deny rule for the
    same path binds an empty file. One target, two shapes, and bwrap dies before
    running anything. A directory on disk makes both operations directory operations.

    The `test -d` half is not decoration: the script runs under `set -eu`, so a mkdir
    that produced something else fails the pod rather than leaving the sandbox to fail
    later with a message about a mask.
    """
    script = _init_script()
    created = {
        path
        for line in script.splitlines()
        if line.strip().startswith("mkdir ")
        for path in line.split()[2:]
    }
    wanted = _workspace_deny_paths()
    assert wanted, "the profile denies nothing under the writable root"
    assert wanted <= created, sorted(wanted - created)
    for path in wanted:
        assert f"test -d {path}" in script, path


def test_no_workspace_deny_target_is_created_as_an_empty_file() -> None:
    """The failure mode this pins is a plausible edit, not a hypothetical: `: > path`
    is what the control-socket instruction used, it reads as equivalent, and at a
    protected basename it makes bwrap fail with a different message about a bind
    mount."""
    script = _init_script()
    for path in _workspace_deny_paths():
        assert f": > {path}" not in script
        assert f"touch {path}" not in script


def test_the_init_container_refuses_a_symlink_on_the_control_socket_path() -> None:
    """Every component, not just the leaf.

    A deny path crossing a symlink is masked at that symlink's current target rather
    than at the logical path, and where the component is writable by the confined
    process it is fatal at argv construction -- bwrap then runs no command at all.
    """
    script = _init_script()
    walked = re.search(r"for c in ([^;]+); do", script)
    assert walked is not None, "the init container walks no path components"
    assert walked.group(1).split() == CONTROL_SOCKET.strip("/").split("/")
    assert "-L" in script and "exit 1" in script


def test_the_runtime_container_is_launched_with_the_compiled_argv() -> None:
    """Element for element. A runtime listening anywhere else is reachable at a path no
    deny rule covers, and the two files are the only place that could disagree."""
    runtime = _pod()["spec"]["containers"][0]
    assert runtime["command"] == list(_compiled().launch_argv)


def test_the_startup_probe_waits_for_the_socket_the_deny_rule_covers() -> None:
    """A pod whose socket never appears never becomes ready, so it never runs a confined
    command against a rule whose target was not there when the argv was compiled.

    Covers rather than names: no rule names this path any more, and the directory rule
    over its parent is what denies it. The wait is unchanged and still matters, because
    whether the leaf exists is what decides which bubblewrap operations the surviving
    rule compiles to."""
    probe = _pod()["spec"]["containers"][0]["startupProbe"]["exec"]["command"]
    assert any(CONTROL_SOCKET in word for word in probe)


def test_the_pod_mounts_no_service_account_token() -> None:
    """On this cluster the projected token is AssumeRoleWithWebIdentity material, so the
    default puts an AWS credential inside the pod that is meant to hold none."""
    assert _pod()["spec"]["automountServiceAccountToken"] is False


def test_every_container_drops_every_capability() -> None:
    """Container-level and not inherited from the pod's securityContext, which is why an
    init container that omits the block runs with the default bounding set while its
    sibling runs with none -- and this manifest is compiled per Session, so the init
    container's body is an interpolation site."""
    containers = _all_containers()
    assert [container["name"] for container in containers] == [
        "seed-runtime-home",
        "restore-working-lane",
        "seed-rollout",
        "agent-runtime",
        "session-shim",
    ]
    for container in containers:
        security = container["securityContext"]
        assert security["allowPrivilegeEscalation"] is False
        assert security["capabilities"]["drop"] == ["ALL"]
        assert security["readOnlyRootFilesystem"] is True


def test_the_compiled_documents_are_mounted_read_only() -> None:
    """The managed layer is what the tenant must not change; a read-write mount into a
    container the agent runs in is the tenant holding the pen on its own constraints."""
    secrets = {
        volume["name"] for volume in _pod()["spec"]["volumes"] if "secret" in volume
    }
    assert secrets, "the compiled documents arrive by some means other than a secret"
    for container in _all_containers():
        for mount in container.get("volumeMounts", []):
            if mount["name"] in secrets:
                assert mount.get("readOnly") is True, mount


def _a_shape_that_narrows_nothing() -> Environment:
    """A registered shape contributing an image and no extra deny rule.

    Compiling now names an environment, and these cases are about the platform's own
    floors rather than about a tenant's narrowing -- so the shape here adds nothing, and
    the compiled profile is exactly what the platform declares.
    """
    return Environment(
        id=new_environment_id(),
        tenant_id=TenantId(uuid4()),
        name="fixture",
        runtime_image="registry.map.internal/session@sha256:" + "a" * 64,
        denied_paths=(),
    )


# --------------------------------------------------------------------------------------
# Whether this Session is continuing a thread travels with the rest of its shape
# --------------------------------------------------------------------------------------


def test_a_compiled_session_is_a_first_placement_unless_it_says_otherwise() -> None:
    """The common path, and the one a wrong default would make silently wrong.

    Defaulting the other way would tell every pod to continue a thread, and the pod
    refuses when it is told to continue and there is nothing to continue from -- so a
    flipped default is a platform that places nothing rather than one that resumes
    wrongly. Asserted anyway, because that refusal is three components away.
    """
    assert _compiled().resuming is False


def test_a_session_compiled_as_resuming_carries_that_to_whatever_starts_its_pod() -> (
    None
):
    """The resume fact rides on the same value as the image and the two documents.

    Here rather than on a second argument to `PodRunner.ensure`, for the reason
    `CompiledConfig` exists at all: a Session's shape and the facts compiled for that
    shape cannot then be handed over separately and turn out to disagree about which
    Session is resuming.
    """
    compiled = compile_session_config(
        _record(),
        tool_gateway_url=GATEWAY_URL,
        model_gateway_url=MODEL_GATEWAY_URL,
        definition=A_DEFINITION,
        environment=_a_shape_that_narrows_nothing(),
        session_token_key=_TOKEN_KEY,
        session_token_expiry_epoch_s=_EXPIRY,
        resuming=True,
    )
    assert compiled.resuming is True
    check_floors(compiled)
