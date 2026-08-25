"""The configuration a Session's pod is started with, and the floors it must hold.

A compiled configuration is a value rather than a file the pod edits: it is rendered
here, mounted, and denied to every command the agent runs. Three of the floors are
load-bearing and none of them announces itself — a TCP listener configured for
convenience, a second MCP server added beside the Tool Gateway, or a deny rule left
out each reopens the runtime's whole un-gated control surface (ADR-012).

Those three are SETTINGS, and the distinction has already been needed once. A
redundant entry removed from a compiled deny set, while the setting it belonged to
still holds, is not one of the three left out: what item 3 of that record requires is
that a deny rule COVER the control path, and access resolves to the prefix-matching
rule with the most path components, so the rule over the directory covers everything
beneath it on its own.

check_floors runs on the way out of compilation and not only from a test, so a
configuration missing a floor is never returned at all. It parses the rendered
documents back before asserting, because what matters is the document the runtime
loads and not the text this module concatenated.

Half the floors are absences and half are presences, and the presences are not
padding. An empty document satisfies every absence: no `sandbox_mode`, no second MCP
server, no TCP listener. So each absence is checked beside the presence that makes it
mean something — the profile is allowlisted and selected, the workspace is writable,
the managed deny list is non-empty — and a compiler that emitted nothing would fail
here rather than pass.

The path constants below are this module's contract with deploy/k8s/session-pod.yaml,
and the two must agree on every path down to the spelling. How badly they must agree
is worth stating precisely, because the repository has been carrying two answers.
Read off the runtime's source, a deny path outside every writable root that does not
exist when the sandbox argv is compiled produces no mask at all and no error.
Measured on the CLI version in hand (codex-cli 0.149.0, MAP-1 probe 4c) it is
synthesized instead: with the socket file deleted and its parent directory still
present, the rule was in the compiled argv and the write was still refused. So the
drop is not what happens today, and nobody has measured the case where the parent
directory is missing too. The ordering the pod manifest keeps is insurance against a
runtime version that behaves as its source reads — not a fix for an observed failure.
"""

import re
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from managed_agent.control.pod_config.model_binding import (
    MODEL_PROVIDER_AUTH_HEADER,
    MODEL_PROVIDER_ID,
    MODEL_PROVIDER_WIRE,
    ModelBindingViolation,
    bind_session,
    render_model_selection,
)
from managed_agent.core.ids import SessionId, TenantId
from managed_agent.core.pod.permission_profile import (
    FsAccess,
    FsRule,
    PermissionProfile,
    nested_deny_pairs,
    path_spelling_error,
)
from managed_agent.core.registration.definition import AgentDefinition
from managed_agent.core.registration.environment import Environment
from managed_agent.core.registration.skill import SkillFile
from managed_agent.core.session.session import SessionRecord
from managed_agent.core.session.session_token import (
    SESSION_TOKEN_HEADER_NAME,
    mint_session_token,
)
from managed_agent.core.toml_text import toml_string

CODEX_HOME = "/var/lib/map/codex"
CONTROL_SOCKET_DIR = "/run/codex"
# One segment deeper than the mount root, and that segment is the whole reason.
#
# `codex app-server` calls `prepare_private_socket_directory()` on the *parent* of
# its `--listen` path before it binds: mkdir(0700), and on EEXIST, lstat, and if the
# mode is not 0700 exactly, chmod(0700). A Kubernetes `emptyDir` mount root is
# created by kubelet as uid 0; `fsGroup` sets the group and the setgid bit and never
# the owner. So with the socket directly in the mount root, that chmod is issued by
# uid 10001 against a root-owned directory, by a process holding no capabilities --
# neither owner nor CAP_FOWNER -- and POSIX refuses it with EPERM.
#
# Measured: the runtime dies with `Error: Operation not permitted (os error 1)`
# before binding anything, identically under `Unconfined` and `RuntimeDefault`,
# because seccomp permits `fchmodat` and the refusal is an ownership check
# downstream of the filter. That is why it read as orthogonal to seccomp: it is.
#
# With the leaf one level down, the parent is a directory the runtime creates
# itself, so the mkdir succeeds and the chmod branch never runs. `CONTROL_SOCKET_DIR`
# stays the mount root and stays separately denied, so the floor is unchanged.
CONTROL_SOCKET = "/run/codex/ctl/app-server-control.sock"
SYSTEM_CONFIG_DIR = "/etc/codex"
WORKSPACE_ROOT = "/session/workspace"
GATEWAY_SERVER_ID = "map-tool-gateway"
PROFILE_NAME = "map-session"

RUNTIME_CATALOG_PATH = "/opt/codex/models.json"
"""Where the pod reads the replacement model catalogue from.

Baked into the runtime image by `tools/bake_model_catalog.py`, not compiled per
Session: the document is identical for every Session on an image, and extracting it
from the binary that ships is what stops it describing a different runtime than the one
that loads it.

**Deliberately not under `SYSTEM_CONFIG_DIR`, and that is why this is spelled out
rather than derived from it.** A Secret volume replaces its mount point's contents
wholesale, and `deploy/k8s/session-pod.yaml` mounts the requirements Secret at
`/etc/codex` in the agent-runtime container -- so a file baked at that path is
invisible to the process that needs it, and invisible in the worst way: the runtime's
own error is that the path does not exist, which reads exactly like an image build that
never ran. Nothing in that container mounts over `/opt`.

The agent can read it, and that is fine -- it holds model metadata and the runtime's
own instruction templates, no credential. What matters is that the agent cannot *edit*
it, and that comes from the profile extending `:read-only` with the workspace as its
only write rule, not from where the file sits.
"""

_CATALOG_REQUIREMENT = f"model_catalog_json = {toml_string(RUNTIME_CATALOG_PATH)}"
"""The requirements line naming the catalogue, emitted and checked as one value.

Two spellings of this line -- one written, one looked for -- would be free to disagree,
and the disagreement would surface as a Session that delegates nothing for a reason no
message names.
"""

_FORBIDDEN_KEYS = frozenset({"sandbox_mode", "sandbox_workspace_write"})
_SANDBOX_FLAG = "--sandbox"
_LISTEN_FLAG = "--listen"
_UNIX_SCHEME = "unix://"

_TOKEN_SHAPE = re.compile(
    r"\A[0-9a-fA-F-]{32,36}\.[0-9a-fA-F-]{32,36}\.[0-9]{1,19}\.[0-9a-f]{64}\Z"
)
"""What a session token looks like, checked without the key that signed it.

There is no key here to verify a signature with -- it belongs to whoever calls this
-- and checking a signature against the key that just made it proves only that HMAC
is deterministic. What is worth checking is the shape, because the shape is what the
runtime silently discards a value for failing: every byte this admits is one an HTTP
header value may carry.
"""


class FloorViolation(Exception):
    """A compiled configuration is missing a floor, so it is not a configuration."""


@dataclass(frozen=True, slots=True)
class CompiledConfig:
    """What one Session's pod is started with: an image, two documents and one argv.

    The image travels here rather than on a second path to whatever starts the pod.
    Placement passes this value through opaquely, so a Session's shape and the documents
    compiled for that shape cannot be handed over separately and disagree.

    `model` and `model_provider` travel here for the same reason and reach the pod the
    same way -- as environment entries whatever starts the pod substitutes, because the
    process that reads them is the shim rather than the runtime, and it reads them
    before it opens the Session's thread. They are not in either document: the runtime
    takes the model per thread, not per configuration file.
    """

    session_id: SessionId
    tenant_id: TenantId
    runtime_image: str
    model: str
    model_provider: str
    config_toml: str
    requirements_toml: str
    launch_argv: tuple[str, ...]
    skill_files: tuple[SkillFile, ...] = ()
    """The skills this Session's agent attached, as files to place beside the documents.

    Here rather than passed alongside, for the reason this class exists: whatever starts
    the pod receives one value, so a Session's shape and the skills compiled for that
    shape cannot be handed over separately and turn out to be about different Sessions.

    Defaulted, because a Session with no skills is the ordinary case and not an error.
    The relative paths are the registry's and carry no root -- which root is mounted is
    the pod manifest's decision, and naming one here would put it in two files free to
    disagree.
    """

    resuming: bool = False
    """Whether this pod continues the thread a previous pod of this Session wrote.

    Here rather than as a second argument to whatever starts the pod, for the reason
    the image and the model are: one value travels, so a Session's shape and the fact
    that it is resuming cannot be handed over separately and turn out to be about
    different Sessions.

    False is the first placement and is the common case. It reaches the pod as an
    environment entry on the init container that seeds the Rollout, and a pod told
    `true` with nothing stored to continue from refuses to start rather than opening a
    fresh thread -- which would replay history the Rollout's compaction checkpoints
    have already folded, bill the tenant for the replay, and report success (ADR-004,
    ADR-031).

    Nothing about this reaches either compiled document. The runtime is not told it is
    resuming; it is handed a Rollout file and asked to continue from it, which is the
    only form of the instruction it takes.
    """


def _keys_anywhere(node: object) -> set[str]:
    """Every key name at any depth of a parsed document.

    Depth matters: the forbidden keys are legal at top level, inside a named config
    profile and inside a table this platform does not write today, so a check that
    looked only at the root would pass a document that downgrades the whole
    enforcement layer.
    """
    if isinstance(node, dict):
        found: set[str] = set(node)
        for value in node.values():
            found |= _keys_anywhere(value)
        return found
    if isinstance(node, list):
        nested: set[str] = set()
        for item in node:
            nested |= _keys_anywhere(item)
        return nested
    return set()


def _parsed(name: str, text: str) -> dict[str, Any]:
    """Parse one rendered document, turning a syntax error into a floor violation.

    A document that does not parse is not a document the runtime can load, which is
    the same outcome as a missing floor and belongs in the same exception rather than
    reaching the caller as a TOMLDecodeError from three frames down.
    """
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as err:
        raise FloorViolation(f"{name} does not parse as TOML: {err}") from err


def _table(document: dict[str, Any], *keys: str) -> dict[str, Any]:
    """Walk to a nested table, refusing a missing or non-table node.

    Every floor below reads some table, and a plain subscript would leave half of
    them raising KeyError instead of FloorViolation — a refusal a caller cannot catch
    by the name this module documents, and one a mutation test cannot tell from a bug.
    """
    node: Any = document
    for depth, key in enumerate(keys, start=1):
        if not isinstance(node, dict) or key not in node:
            raise FloorViolation(f"no {'.'.join(keys[:depth])} table in the document")
        node = node[key]
    if not isinstance(node, dict):
        raise FloorViolation(f"{'.'.join(keys)} is not a table")
    return node


def _refuse_the_older_sandbox_settings(documents: dict[str, dict[str, Any]]) -> None:
    """No document names a key that would silently replace the permission profile."""
    for name, document in documents.items():
        present = _keys_anywhere(document) & _FORBIDDEN_KEYS
        if present:
            raise FloorViolation(
                f"{name} carries the older sandbox settings {sorted(present)}, "
                "which silently replace the permission profile"
            )


def _require_only_the_tool_gateway(documents: dict[str, dict[str, Any]]) -> None:
    """Each document names the Tool Gateway as its one MCP server, and names it."""
    for name, document in documents.items():
        servers = tuple(_table(document, "mcp_servers"))
        if servers != (GATEWAY_SERVER_ID,):
            raise FloorViolation(
                f"{name} names MCP servers {list(servers)}; the Tool Gateway must "
                "be the only one"
            )


def _require_a_unix_listener(argv: tuple[str, ...]) -> None:
    """The runtime is reachable over one unix socket and carries no sandbox flag.

    `--sandbox=read-only` is refused alongside the bare flag. Both spellings reach
    the same argument parser, so a check matching only the bare one is satisfied by
    the spelling an engineer copying from the runtime's own SDK is likely to write.
    """
    forbidden = [
        arg
        for arg in argv
        if arg == _SANDBOX_FLAG or arg.startswith(f"{_SANDBOX_FLAG}=")
    ]
    if forbidden:
        raise FloorViolation(f"the launch argv carries {forbidden}")
    if argv.count(_LISTEN_FLAG) != 1:
        raise FloorViolation(f"the launch argv must name {_LISTEN_FLAG} exactly once")
    at = argv.index(_LISTEN_FLAG) + 1
    if at >= len(argv):
        raise FloorViolation(f"{_LISTEN_FLAG} names no address")
    if not argv[at].startswith(_UNIX_SCHEME):
        raise FloorViolation(
            f"the runtime listens on {argv[at]!r}, not on a unix socket"
        )


def _require_every_project_untrusted(config: dict[str, Any]) -> None:
    """There is a project root, and no project root is trusted.

    Both halves. Marking a project untrusted is what makes the runtime skip its
    project-scoped `.codex` config, hooks and rules; a document naming no project at
    all has nothing marked, and would pass a check that only looked for 'trusted'.
    """
    projects = _table(config, "projects")
    if not projects:
        raise FloorViolation("config.toml declares no project root to mark untrusted")
    trusted = sorted(
        root
        for root, body in projects.items()
        if not isinstance(body, dict) or body.get("trust_level") != "untrusted"
    )
    if trusted:
        raise FloorViolation(f"project roots not marked untrusted: {trusted}")


def _require_profile_mode_forced(requirements: dict[str, Any]) -> None:
    """The managed document selects this profile and allowlists it.

    This is the presence that makes the absence of `sandbox_mode` worth checking. The
    forbidden keys are refused in the two documents this module renders, and a stray
    one in any other layer of the loaded chain would still downgrade the deployment.
    A managed `allowed_permission_profiles` is the single thing that overrides that,
    and it overrides it by being present rather than by what it lists (ADR-005).
    """
    allowed = _table(requirements, "allowed_permission_profiles")
    if not allowed:
        raise FloorViolation(
            "allowed_permission_profiles is empty, so nothing forces profile mode"
        )
    selected = requirements.get("default_permissions")
    if selected != PROFILE_NAME:
        raise FloorViolation(
            f"default_permissions is {selected!r}, not the compiled profile "
            f"{PROFILE_NAME!r}"
        )
    if allowed.get(selected) is not True:
        raise FloorViolation(
            f"the selected profile {selected!r} is not allowlisted, so the runtime "
            "denies it"
        )


def _refuse_egress_the_proxy_does_not_bound(requirements: dict[str, Any]) -> None:
    """Egress is on with the proxy bounding it, or egress is off. No third state.

    **The state this refuses is not "no rules" -- it is rules that look like rules and
    are not.** A profile carrying `network.enabled = true` grants the agent's commands
    outbound access. The allowlist beside it restricts where they may go ONLY while the
    managed proxy is active, and `experimental_network.enabled = true` is what activates
    it: the list alone does not. So a document naming three domains, with the profile's
    network on and the managed proxy off, is a Session with the whole internet -- and it
    is the configuration an auditor reading that document is most likely to approve,
    because the domains are right there.

    `managed_allowed_domains_only` is required for the other half. Without it a config
    layer inside the pod may ADD domains to the ones named here, so the list this
    platform compiled becomes a floor a tenant can raise rather than a ceiling.

    Off is checked as hard as on. A document with the proxy configured and the profile's
    network absent is not dangerous, but it is a shape nothing in this tree emits, and
    admitting it would mean the pair could come apart in one direction unnoticed --
    which is how the other direction eventually gets emitted too.
    """
    managed = requirements.get("experimental_network", {})
    if not isinstance(managed, dict):
        raise FloorViolation("experimental_network is not a table")
    # Read by hand rather than through `_table`, which raises for an absent table --
    # and an absent table is the NORMAL case here: a Session with no granted domain has
    # no network block at all, and a floor that refused every such document would refuse
    # every Session this platform has ever run.
    profiles = requirements.get("permissions", {})
    profile = profiles.get(PROFILE_NAME, {}) if isinstance(profiles, dict) else {}
    network = profile.get("network", {}) if isinstance(profile, dict) else {}
    if not isinstance(network, dict):
        raise FloorViolation(f"permissions.{PROFILE_NAME}.network is not a table")
    granted = network.get("enabled") is True
    proxied = managed.get("enabled") is True
    if granted != proxied:
        raise FloorViolation(
            f"the profile grants network access {granted} and the managed proxy is "
            f"active {proxied}; with the grant and no proxy the agent reaches the "
            f"whole internet while the document reads as if a list bounded it, and "
            f"with a proxy and no grant nothing in this tree emits the document"
        )
    if not granted:
        if managed:
            raise FloorViolation(
                f"no network is granted, so the managed egress keys should be absent "
                f"rather than present and inert: {sorted(managed)}"
            )
        return
    domains = managed.get("allowed_domains")
    if not isinstance(domains, list) or not domains:
        raise FloorViolation(
            "network access is granted with no allowed_domains, which is unbounded "
            "egress written as if it were bounded"
        )
    if managed.get("managed_allowed_domains_only") is not True:
        raise FloorViolation(
            "allowed_domains is not marked managed-only, so a config layer inside the "
            "pod may add to the list this platform compiled"
        )


def _require_the_control_path_denied(requirements: dict[str, Any]) -> None:
    """The control path is denied in both lists, and the workspace is still writable.

    Both because the two mechanisms fail differently. The profile's own rule is what
    the sandbox argv is compiled from; the managed `deny_read` list is non-weakenable
    by any local layer and, while it is non-empty, makes the runtime refuse
    full-access permissions for every profile including ones added later (ADR-005).

    The path required is the DIRECTORY, and specifically it: access resolves to the
    prefix-matching rule with the most path components, so the directory rule denies
    everything beneath it including the socket. This is not relaxed to "either rule
    will do". The leaf alone leaves the directory world-listable and the socket's own
    filename readable out of that listing, so a configuration naming only the leaf is
    a weaker one and must not satisfy this floor. The socket by name AS WELL is
    refused separately, by the nesting floor below, because that pair makes bubblewrap
    refuse to build any sandbox once the socket exists (ADR-012).

    The writable assertion is the positive half: a profile that denied every path
    named here and granted nothing would satisfy every deny check and could run no
    agent, and the two states are indistinguishable from the deny rules alone.
    """
    rules = _table(requirements, "permissions", PROFILE_NAME, "filesystem")
    for path in (CONTROL_SOCKET_DIR,):
        if rules.get(path) != FsAccess.DENY.value:
            raise FloorViolation(f"no deny rule over {path}, the control path")
    if rules.get(WORKSPACE_ROOT) != FsAccess.WRITE.value:
        raise FloorViolation(f"the workspace root {WORKSPACE_ROOT} is not writable")

    deny_read = _table(requirements, "permissions", "filesystem").get("deny_read")
    if not isinstance(deny_read, list) or not deny_read:
        raise FloorViolation(
            "the managed deny_read list is empty, so full-access permissions are "
            "not foreclosed"
        )
    missing = [p for p in (CONTROL_SOCKET_DIR,) if p not in deny_read]
    if missing:
        raise FloorViolation(f"the managed deny_read list omits {missing}")


def _denied_paths(requirements: dict[str, Any]) -> set[str]:
    """The union of the two lists the sandbox's filesystem policy is built from.

    The profile table is what a reader sees and `deny_read` is the copy no layer
    inside the pod can weaken, and every `deny_read` entry is pushed into the same
    policy the argv is compiled from -- so an entry present in only one of them still
    reaches the sandbox. They are rendered from one source today and a test pins them
    equal, which is exactly why the union is taken rather than either list alone: a
    rendering that stopped keeping them equal would otherwise put half of every check
    over this set out of reach.

    Raises rather than guessing when the table uses an access value this module cannot
    classify, and that clause is the whole reason this is a function with a contract
    instead of a set comprehension at the call site. Selecting the denied rows means
    asking "which of these values means denied", and `"deny"` is one spelling of that
    answer rather than the whole of it: measured against codex-cli 0.149.0 on this
    cluster, `"none"` is also accepted and also denies -- a pod whose profile said
    `"/etc/codex" = "none"` refused a confined `ls` of that directory at exit 0. A
    check reading only `== "deny"` would have counted that row as not-denied and
    reported a deny set with a hole in it. `"Deny"` and `"DENY"`, by contrast, make
    the runtime refuse the whole document (`data did not match any variant of untagged
    enum FilesystemPermissionToml`), so case needs nothing here.

    So the recognised set is inverted rather than extended: the three values `FsAccess`
    can render are allowed and everything else is refused, which stays complete when
    the runtime adds a fifth. Nothing this platform can express is lost -- `FsAccess`
    has no member outside those three -- and a value from outside them in a RENDERED
    document is a rendering that has stopped matching what the profile can hold.
    """
    table = _table(requirements, "permissions", PROFILE_NAME, "filesystem")
    known = {access.value for access in FsAccess}
    for path, access in table.items():
        if access not in known:
            raise FloorViolation(
                f"the rule over {path!r} has access {access!r}, which is not one of "
                f"{sorted(known)}; this module cannot tell whether it denies, and the "
                "runtime honours values outside that set"
            )
    denied = {path for path, access in table.items() if access == FsAccess.DENY.value}
    deny_read = _table(requirements, "permissions", "filesystem").get("deny_read")
    if isinstance(deny_read, list):
        denied |= {path for path in deny_read if isinstance(path, str)}
    return denied


def _refuse_a_deny_nested_under_another(requirements: dict[str, Any]) -> None:
    """No denied path may lie strictly inside another denied path.

    Not tidiness, and not about the control path specifically. A nested deny is fatal
    to the whole sandbox the moment the descendant EXISTS when the argv is compiled:
    unreadable roots are applied parent-first, an unreadable directory becomes a
    mode-000 tmpfs remounted read-only, and the descendant's own operation is then
    attempted inside a filesystem already frozen. bubblewrap dies and NO confined
    command runs for that Session.

    The condition is measured rather than assumed, and it is the whole reason this
    refuses a class rather than a case. In one pod, one document, either side of a
    single mkdir: with `/run/codex/deeper` denied beside `/run/codex` and nothing
    having created it, `codex sandbox -- /bin/echo` exits 0; with the directory
    created, the same pod answers `bwrap: Can't mkdir /run/codex/deeper: Read-only
    file system` and exits 1. So a nested pair whose descendant nothing creates is
    harmless today and fatal the instant anything creates it -- which is exactly what
    happened to this platform, whose control socket is a nested deny the runtime binds
    on every start.

    THAT is why the existence condition cannot be part of this floor. This runs in the
    control plane, where the pod does not exist yet, so whether a path will be there
    when the argv is compiled is not decidable here. Refusing every nested pair is the
    decidable approximation, it is the fail-closed direction, and it costs nothing:
    access resolves to the prefix-matching rule with the most path components, so a
    deny strictly under a deny resolves to the answer its ancestor already gives.
    Stated plainly because it is a real choice -- this floor is STRICTER than the
    runtime for a pair the runtime tolerates while the target is absent.

    Deny-under-deny only, never rule-under-rule: this profile carves two denied holes
    in its one writable root on purpose, and that nesting is measured working.

    "Inside" is decided by comparing path components rather than string prefixes, and
    that is measured too rather than argued. `/run//codex/deeper` does not start with
    the string `/run/codex/`, so a string comparator reports the pair un-nested -- and
    the runtime is fatal on it in the same state as the canonical spelling, naming the
    normalised path in its own message. A guard grading the spelling would have passed
    a document that starts no Session.

    The spelling clause runs first and is a second, narrower rule: every deny path
    must be in the one spelling `FsRule` and `Environment` accept. Both parses produce
    only that spelling, so any other in a RENDERED document is evidence the rendering
    has stopped matching the parse, and it is the one thing about a path this module
    can settle without a pod. It also keeps the comparison's input canonical, so which
    comparison form is used stops being a question a reader has to re-derive.
    """
    denied = _denied_paths(requirements)
    if not denied:
        raise FloorViolation(
            "the compiled configuration denies no path at all, so this floor and "
            "every other deny check over the set would pass by never being reached"
        )
    for path in sorted(denied):
        reason = path_spelling_error(path)
        if reason is not None:
            raise FloorViolation(
                f"the deny set names {path!r}, which {reason}; a path spelled a way "
                "nothing else in this tree writes is a rule whose target the sandbox "
                "looks for where the pod never created one"
            )
    nested = nested_deny_pairs(sorted(denied))
    if nested:
        ancestor, descendant = nested[0]
        raise FloorViolation(
            f"the deny set nests {descendant!r} under {ancestor!r}; the ancestor "
            f"already denies everything the descendant would, and bubblewrap refuses "
            f"to build any sandbox for the pair once the descendant exists"
        )


def _require_a_model_and_a_provider(compiled: CompiledConfig) -> None:
    """Neither the model nor the provider may be empty.

    Not a document floor like the others, and it is here for a measured reason. A pod
    started with these as empty strings pulls, initialises, and brings its runtime to
    ready -- and then the shim's `thread/start` is answered `-32600` by the Agent
    Runtime, the shim exits, and the pod sits at phase Running with one container dead.
    So an empty value does not fail where it is set; it fails three layers away, looking
    like a pod that is slow to start. Refusing here means it cannot be set at all.

    Non-empty is half the floor and on its own it was a floor that could not fail: the
    provider is a module constant, so the only non-empty value this ever saw was the one
    the runtime went on to reject. `_require_the_named_provider_is_declared` below is
    the other half, and the two only mean something together.
    """
    for name, value in (
        ("model", compiled.model),
        ("model_provider", compiled.model_provider),
    ):
        if not value.strip():
            raise FloorViolation(
                f"the compiled configuration names no {name}, so the pod's shim would "
                "start a thread the Agent Runtime refuses"
            )


def _require_the_named_provider_is_declared(
    config: dict[str, Any], compiled: CompiledConfig
) -> None:
    """The provider the shim will name must be declared in the document it loads.

    This is the floor the empty-string check could not be. The shim reads
    `model_provider` out of its environment and passes it to `thread/start`; the runtime
    resolves it against the `[model_providers]` tables of the configuration chain it
    loaded and refuses the call if it is not there. The two values come from different
    places -- one from a field of this value, one from a table this module renders -- so
    nothing but a check makes them agree.

    Asserted over the parsed document rather than over the rendered text, because what
    the runtime resolves against is the table it loads, and a `base_url` line that lands
    under the wrong header still greps.

    Each of the three fields is checked, and the three fail differently. Measured
    against codex-cli 0.149.0 in the pushed Session image: a missing table is `Model
    provider `map-model-gateway` not found` and no Session starts; an empty `name` is
    refused at configuration load with `provider name must not be empty`; and
    `wire_api = "chat"` is refused at load naming the removal.

    `base_url` is the one the runtime does *not* refuse, which is why it is checked here
    hardest. Measured: a provider table carrying `name` alone loads, and the runtime
    starts and reports `provider: map-model-gateway` -- and then addresses its model
    calls to the runtime's own default endpoint, which is a request leaving a Session
    pod to somewhere the Egress Policy never allowed. An absent `base_url` is not a call
    with nowhere to go; it is a call to the wrong place, made silently. So this floor is
    stricter than the runtime, on purpose.
    """
    provider = compiled.model_provider
    tables = config.get("model_providers")
    declared = tables if isinstance(tables, dict) else {}
    entry = declared.get(provider)
    if not isinstance(entry, dict):
        raise FloorViolation(
            f"the compiled config.toml declares no model provider {provider!r} -- it "
            f"declares {sorted(declared)} -- so the runtime answers the Session's "
            "thread/start with 'Model provider not found' and the shim exits"
        )
    if not str(entry.get("name", "")).strip():
        raise FloorViolation(
            f"the model provider {provider!r} is declared with no name, which the "
            "runtime refuses at configuration load"
        )
    if not str(entry.get("base_url", "")).strip():
        raise FloorViolation(
            f"the model provider {provider!r} is declared with no base_url, so the "
            "runtime would address its model calls to its own default endpoint rather "
            "than to the Model Gateway"
        )
    if entry.get("wire_api") != MODEL_PROVIDER_WIRE:
        raise FloorViolation(
            f"the model provider {provider!r} declares wire_api "
            f"{entry.get('wire_api')!r}; the Agent Runtime speaks "
            f"{MODEL_PROVIDER_WIRE!r} and the Model Gateway serves only that"
        )


def _require_a_header_carrying_this_sessions_token(
    headers: object,
    *,
    header_name: str,
    where: str,
    session_id: SessionId,
    tenant_id: TenantId,
) -> None:
    """One header table names exactly one token header, and it names this Session.

    Written once and called for both gateways rather than spelled per gateway, because
    what is duplicated here is a *rule* and not merely shape: the same token, verified
    by the same key against the same layout, at two destinations. Spelled twice, the two
    copies get to disagree -- and the way they disagree is the way they already did. The
    Tool Gateway floor graded the Session and never the tenant until 2026-08-23, which
    caught the typo and missed the breach; a second hand-written copy is a second
    chance to make exactly that omission, in a place nobody thinks to re-read.

    Three separate silent failures live behind this, all measured against the pinned
    runtime rather than reasoned about. A misspelled key is accepted by the
    configuration parser and does nothing. A value the runtime cannot put in a header is
    dropped with a warning and the request goes out unauthenticated. An empty value is
    sent as an empty header. In all three the gateway answers the same fixed 401 it
    answers a request that carried nothing, so nothing downstream can tell a
    mis-rendered document from a wrong key or an expired token.

    Both halves of the identity are checked, and the tenant is the half that matters
    most. A token naming another Session or another tenant is a document that *works*:
    it verifies, and the gateway acts under whatever the token names rather than under
    whoever the pod belongs to. The tenant is the field carrying authority -- the
    enterprise credentials one gateway lends out and the provider spend the other
    authorizes are both selected by it -- so a compiled configuration whose token names
    the wrong tenant is cross-tenant access with a valid signature on it.

    The set is exact, not a superset. A second header alongside this one is a second
    thing the pod asserts about itself, and nothing here writes one.

    What this cannot check is the signature. The key belongs to whoever called the
    compiler, and a signature checked against the key that just made it proves only that
    HMAC is deterministic -- so a wrong key is invisible here and shows up as a 401 at
    the gateway. Shape and identity are what is checkable without the key.

    `header_name` is compared case-sensitively against what the renderer emitted, which
    is stricter than HTTP. Header names are case-insensitive on the wire, but the value
    here has not reached the wire yet: it is a TOML key the runtime looks up, and a
    lookup that misses sends no header at all.
    """
    if not isinstance(headers, dict) or set(headers) != {header_name}:
        named = sorted(headers) if isinstance(headers, dict) else []
        raise FloorViolation(
            f"the {where} names http_headers {named}; it must name "
            f"exactly ['{header_name}']"
        )
    value = headers[header_name]
    token = _token_in(value, header_name=header_name)
    if not _TOKEN_SHAPE.fullmatch(token):
        raise FloorViolation(
            f"the {header_name} value is not shaped like a session token, so the "
            "runtime would drop it and send no header at all"
        )
    named_session, named_tenant = token.split(".")[0], token.split(".")[1]
    if named_session != str(session_id):
        raise FloorViolation(
            f"the {header_name} value names session {named_session}, not {session_id}"
        )
    if named_tenant != str(tenant_id):
        raise FloorViolation(
            f"the {header_name} value names tenant {named_tenant}, not {tenant_id}"
        )


def _token_in(value: object, *, header_name: str) -> str:
    """The token out of a header value, whether or not the value carries a scheme.

    The two gateways differ in exactly one way and this is it. The Tool Gateway reads a
    bespoke header whose whole value is the token; the Model Gateway reads
    `authorization`, where the token is preceded by `Bearer `. So the scheme is stripped
    for the header that has one and its absence is a refusal, rather than the shape
    check below being loosened to admit a space -- a `_TOKEN_SHAPE` that tolerated a
    prefix would stop being able to say the value is header-safe, which is the one thing
    it exists to say.
    """
    if not isinstance(value, str):
        raise FloorViolation(
            f"the {header_name} value is not a string, so the runtime would drop it "
            "and send no header at all"
        )
    if header_name != MODEL_PROVIDER_AUTH_HEADER:
        return value
    scheme, space, token = value.partition(" ")
    if not space or scheme != "Bearer":
        raise FloorViolation(
            f"the {header_name} value does not begin with 'Bearer ', so the Model "
            "Gateway reads no bearer token and answers every model call 401"
        )
    return token


def _require_the_agents_instructions_reached(config: dict[str, Any]) -> None:
    """Refuse a compiled config that carries no `developer_instructions` at top level.

    **This floor exists because its absence shipped.** Until 2026-08-23 a definition's
    `instructions` were bound, carried on `AgentBinding`, and written into no document
    the root agent reads -- so a tenant's agent ran with no persona, no conventions and
    no domain knowledge, and every symptom of that looks like the model being unhelpful.
    Nothing failed. A live probe was the only thing that found it.

    Top level and not nested, because the nesting is the failure mode with no other
    symptom. `developer_instructions` written one line lower parses as
    `agents.developer_instructions`, a key the runtime does not read, in a document that
    loads without complaint -- so the tenant's text is present in the file, visible to
    anyone who opens it, and still invisible to the model.

    Emptiness is refused as well as absence. `AgentDefinition.instructions` cannot be
    empty, so an empty value here means something between the definition and this
    document dropped it, which is exactly the class of defect this floor is for.
    """
    carried = config.get("developer_instructions")
    if carried is None:
        raise FloorViolation(
            "the compiled config carries no top-level `developer_instructions`, so "
            "this Session's agent would run with none of the instructions its "
            "definition declared -- and nothing at run time would say so"
        )
    if not isinstance(carried, str) or not carried.strip():
        raise FloorViolation(
            "the compiled config's `developer_instructions` is empty, so some "
            "definition's instructions were lost between binding and rendering: "
            f"{carried!r}"
        )


def check_floors(compiled: CompiledConfig) -> None:
    """Raise FloorViolation unless every floor holds over the parsed documents."""
    _require_a_model_and_a_provider(compiled)
    documents = {
        "config.toml": _parsed("config.toml", compiled.config_toml),
        "requirements.toml": _parsed("requirements.toml", compiled.requirements_toml),
    }
    _require_the_named_provider_is_declared(documents["config.toml"], compiled)
    _refuse_the_older_sandbox_settings(documents)
    _require_only_the_tool_gateway(documents)
    _require_a_unix_listener(compiled.launch_argv)
    _require_every_project_untrusted(documents["config.toml"])
    _require_the_agents_instructions_reached(documents["config.toml"])
    _require_profile_mode_forced(documents["requirements.toml"])
    _require_the_control_path_denied(documents["requirements.toml"])
    _refuse_egress_the_proxy_does_not_bound(documents["requirements.toml"])
    _refuse_a_deny_nested_under_another(documents["requirements.toml"])
    _require_a_header_carrying_this_sessions_token(
        _table(documents["config.toml"], "mcp_servers", GATEWAY_SERVER_ID).get(
            "http_headers"
        ),
        header_name=SESSION_TOKEN_HEADER_NAME,
        where="Tool Gateway block",
        session_id=compiled.session_id,
        tenant_id=compiled.tenant_id,
    )
    _require_a_header_carrying_this_sessions_token(
        _table(documents["config.toml"], "model_providers", MODEL_PROVIDER_ID).get(
            "http_headers"
        ),
        header_name=MODEL_PROVIDER_AUTH_HEADER,
        where=f"model provider {MODEL_PROVIDER_ID!r} block",
        session_id=compiled.session_id,
        tenant_id=compiled.tenant_id,
    )
    _require_the_requirements_name_the_catalogue(compiled)


def _require_the_requirements_name_the_catalogue(compiled: CompiledConfig) -> None:
    """Raise unless the managed document points the runtime at the baked catalogue.

    The only catalogue check this compiler can still make, and the reason is worth
    stating: the document itself is built into the runtime image by
    `tools/bake_model_catalog.py`, which asserts everything about its *contents*
    there -- that every entry carries a slug, an instruction template and the raised
    output cap,
    and that each routable model's entry is api-supported, picker-visible and not
    excluded from the multi-agent backend. Those belong at build time because a
    catalogue the runtime's deserializer refuses is a hard `InvalidData` at load,
    which stops every pod on the image; failing the build is the cheaper place to find
    out.

    What is left here is the half no image can check. The runtime reads the catalogue
    only because this line names it, so a document compiled without it produces a
    Session that loads cleanly, runs, and refuses every spawn -- the defect this change
    exists to remove, returning silently.
    """
    if _CATALOG_REQUIREMENT not in compiled.requirements_toml:
        raise FloorViolation(
            f"session {compiled.session_id} compiled a managed document that does not "
            f"name the model catalogue ({_CATALOG_REQUIREMENT!r}), so the runtime "
            "would never load it and every subagent spawn would be refused"
        )


def _toml_string(value: str) -> str:
    """Quote a value as a TOML basic string.

    Delegated rather than done here, and the delegation is the point. This module emits
    paths, URLs and profile names, none of which contains a newline -- so the escaping
    it needed for years was backslash and quote, and that is what it did. Then it had to
    emit `additional_developer_instructions`, which is a paragraph, and the incomplete
    table produced a document the runtime rejects at load. `core/toml_text.py` records
    why one rule now serves both callers.
    """
    return toml_string(value)


def _render_config(
    *,
    workspace_root: str,
    gateway_url: str,
    model_selection: str,
    session_token: str,
) -> str:
    """The user-layer document, copied into the runtime's home before it starts.

    The model selection is placed first and joined as text rather than rebuilt here,
    because it opens with bare keys. TOML reads a bare key as belonging to whatever
    table header precedes it, so `model` written after the project table below would
    silently become that table's key and the runtime would start with no model named.

    What that fragment carries and why it carries it belongs to the module that renders
    it. What matters here is that it is the only place the provider table comes from:
    two renderers emitting `[model_providers.<id>]` into one document is not a merge but
    a duplicate table, and TOML refuses to parse it at all.

    The Session's token to the Tool Gateway rides in this document rather than in the
    pod's environment, and that is why the pod specification needs no new variable, no
    new Secret and no new volume. It also puts the value under CODEX_HOME, which this
    module's own profile denies to every confined command, instead of in a pod field
    anyone who can read the pod can read. The runtime builds its header map once at
    client construction, so what is written here is what every tool call carries for
    the life of the pod.
    """
    return "\n".join(
        (
            "# Compiled per Session. Nothing inside the pod writes this file.",
            model_selection,
            f"[projects.{_toml_string(workspace_root)}]",
            'trust_level = "untrusted"',
            "",
            f"[mcp_servers.{_toml_string(GATEWAY_SERVER_ID)}]",
            f"url = {_toml_string(gateway_url)}",
            "required = true",
            # Without this line the Gateway's tools are offered to the model and then
            # refused when it calls one, which reads to the model as the tool being
            # blocked -- measured, in those words, on a live Turn.
            #
            # The runtime auto-approves an MCP call only under the conditions in
            # codex-mcp/src/mcp/mod.rs:96-105: `approval_policy` is never (ours is), AND
            # the Permission Profile is Disabled, External, or Managed WITH FULL DISK
            # WRITE ACCESS. Ours is Managed and deliberately narrow, so the third arm is
            # false and always will be -- widening the sandbox to satisfy it would trade
            # the whole isolation story for a tool call. With nobody inside a pod
            # to ask, a call needing approval is simply refused.
            #
            # So the per-server override is the only door that does not cost the
            # sandbox, and it is not a widening: every tool behind this Gateway was
            # already named by a Grant and clamped to the Session's Scope before the
            # Gateway would forward it. Approving here re-states a decision the platform
            # already made one layer out, rather than making a new one.
            'default_tools_approval_mode = "approve"',
            "http_headers = { "
            f"{_toml_string(SESSION_TOKEN_HEADER_NAME)} = "
            f"{_toml_string(session_token)}"
            " }",
            "",
        )
    )


def _render_network(allowed_domains: Sequence[str]) -> list[str]:
    """The managed egress rules, or nothing at all when no domain was granted.

    **Nothing at all is not the same as "off by default and harmless to omit".** It is
    what keeps the default safe: the sandbox holds egress closed unless a profile opens
    it, so an absent block is a Session whose commands reach nothing, and there is no
    spelling of "off" here that would be safer than saying nothing.

    All three keys are written together because two of them without the third is the
    dangerous configuration rather than a partial one. `enabled` activates the managed
    proxy; the allowlist alone does not, and a document carrying only the list leaves
    the runtime taking DIRECT outbound access with the list restricting nothing -- which
    reads, to anyone auditing the document, exactly like a restricted egress that is
    working. `managed_allowed_domains_only` makes this list exclusive, so no config
    layer inside the pod can append to it; without it, a tenant-authored layer can widen
    what this document was written to bound.

    These keys are dotted rather than a `[experimental_network]` table because they are
    emitted BEFORE the first table header. Written as a table they would be fine here
    and would break the moment somebody moved them, and the two rules below them are top
    level -- a reader adding a key here should not have to notice which side of a header
    they are on.

    The profile's own `network.enabled` is separate and both are needed. The managed
    keys configure and start the proxy; the profile's key is what grants the agent's
    commands network access at all, and the managed side deliberately cannot grant it --
    an administrator can bound egress without turning it on for anybody.
    """
    if not allowed_domains:
        return []
    return [
        "experimental_network.enabled = true",
        "experimental_network.managed_allowed_domains_only = true",
        "experimental_network.allowed_domains = ["
        + ", ".join(_toml_string(domain) for domain in allowed_domains)
        + "]",
    ]


def _render_requirements(
    profile: PermissionProfile,
    *,
    gateway_url: str,
    allowed_domains: Sequence[str] = (),
) -> str:
    """The managed document, which a layer inside the pod cannot weaken.

    allowed_permission_profiles is written whether or not its contents restrict
    anything, because its mere presence is what forces profile mode; without it one
    stray older sandbox key anywhere in the loaded chain reverts the deployment with
    no error (ADR-005).

    deny_read is a second, independent guard on the same paths: while it is non-empty
    the runtime refuses full-access permissions for every profile, including ones
    added later.

    The egress block and the profile's `network.enabled` are emitted as a pair or not at
    all -- see `_render_network` for why either alone is worse than neither.

    `model_catalog_json` is emitted here rather than in `config.toml`, and on every
    Session rather than only the delegating ones. Here because a requirement both
    *sets* the value and pins it -- the runtime applies requirements to the parsed
    configuration before it loads the catalogue, then refuses any later layer that
    disagrees -- so a document inside the pod cannot swap the list of models the agent
    may delegate to.
    Unconditionally because the file is baked into the image rather than compiled per
    Session: it costs this document one line and the Session no bytes at all, and gating
    it would leave a non-delegating Session's own threads on the runtime's 10,000-token
    output default, against which the Evidence capture's margin is undefined (ADR-020,
    whose cap raise this closes).

    It takes a **path**, not the JSON. Measured on the pinned binary rather than
    assumed: its error strings are `model_catalog_json path` and `failed to parse
    model_catalog_json path`, the config field's type is `Option<AbsolutePathBuf>`, and
    the loader calls `read_to_string` on it. The runtime reads that path while loading
    its configuration and treats an unreadable one as fatal, so this line and the bake
    step are one mechanism: naming a file the image does not carry stops every pod.

    **There is deliberately no `additional_developer_instructions` here, and the reason
    is worth the paragraph.** That key is the administrator's channel for exactly what
    `core/pod/workspace_contract.py` holds -- the runtime renders it as a
    `developer`-role message wrapped in `<managed_developer_instructions>`, separate
    from the tenant's own text and attributed to whoever runs the platform. It is the
    right home for the contract in every way but one: **codex 0.149.0, which this
    platform pins, does not have it.**

    Measured rather than reasoned. The compiled document carried the key correctly, the
    pod's `/etc/codex/requirements.toml` held it verbatim, the runtime loaded that file
    (the permission profile in it was in force), and an agent asked which directory the
    platform collects finished work from answered `UNKNOWN` three times. `strings` over
    the shipped musl binary then settled it: `experimental_network` appears 5 times and
    `developer_instructions` 39, and `additional_developer_instructions` zero. It is a
    `main`-branch key, and the checkout under `.reference/` is `main`.

    So the contract rides `config.toml`'s `developer_instructions` instead, ahead of the
    tenant's own text -- see `control/pod_config/model_binding.py`. The cost is real and
    named there: one field now carries two authorships. Writing this key anyway "for a
    later codex" would be worse than the mixing, because a key that is present and inert
    is the thing a future reader takes as evidence the contract is delivered.
    """
    network = _render_network(allowed_domains)
    lines = [
        "# Compiled per Session. A layer inside the pod cannot weaken these.",
        "allow_managed_hooks_only = true",
        f"default_permissions = {_toml_string(profile.name)}",
        _CATALOG_REQUIREMENT,
        *network,
        "",
        "[allowed_permission_profiles]",
        f"{_toml_string(profile.name)} = true",
        "",
        "[permissions.filesystem]",
        "deny_read = [" + ", ".join(_toml_string(p) for p in profile.denied()) + "]",
        "",
        f"[permissions.{_toml_string(profile.name)}]",
        f"extends = {_toml_string(profile.extends)}",
        "",
        f"[permissions.{_toml_string(profile.name)}.filesystem]",
        *(
            f"{_toml_string(r.path)} = {_toml_string(r.access.value)}"
            for r in profile.rules
        ),
        "",
        *(
            [
                f"[permissions.{_toml_string(profile.name)}.network]",
                "enabled = true",
                "",
            ]
            if network
            else []
        ),
        f"[mcp_servers.{_toml_string(GATEWAY_SERVER_ID)}]",
        f"identity = {{ url = {_toml_string(gateway_url)} }}",
        "",
    ]
    return "\n".join(lines)


def session_profile() -> PermissionProfile:
    """The filesystem rules every Session's pod runs under.

    The parent is `:read-only` rather than `:workspace`, so what is writable is the
    one prefix named here and not whatever the runtime counts as a workspace root on
    the day.

    The two configuration directories are treated differently, and the difference is
    what each one holds. `$CODEX_HOME` holds the compiled `config.toml`, and that
    document carries this Session's own bearer token and the provider base URL — an
    agent that reads it can spend the tenant's budget under its own name, so it stays
    denied. `/etc/codex` holds the managed `requirements.toml` and the Session's
    skills, and it is readable. `requirements.toml` carries no credential: it is these
    permission rules plus an in-cluster gateway URL, and the kernel enforces the rules
    whether or not the confined process can read them, so denying it bought nothing.
    What denying it did cost was the skills. A Host skill's catalogue line gives the
    model a FILE PATH and expects it to open the file with its shell
    (`ext/skills/src/render.rs:195-215`), and the tool route that would read it for the
    model returns an empty list unless an orchestrator or executor provider is present
    (`ext/skills/src/tools/mod.rs:64-70`) — our pod has neither. So a denied
    `/etc/codex` meant every skill was delivered, catalogued, named to the model, and
    then unreadable by it, with nothing in any log saying why.

    `<workspace>/.codex` and `<workspace>/.agents` are denied although neither exists
    yet. Read off the runtime's source, a missing deny path *inside* a writable root
    is masked at its first non-existent component, so the confined process cannot
    create the hierarchy at all — which is what would make configuration written into
    the workspace inert rather than merely ignored. Stated as a source reading and not
    as a measurement: MAP-1's probe measured only the case outside every writable
    root, and measured it behaving differently from how the same source reads.

    `/run/codex` is denied as a directory, and the socket is NOT denied by name as
    well. The directory is what does the denying: access resolves to the
    prefix-matching rule with the most path components, so a rule over the directory
    already denies every path beneath it and a second rule at the leaf changes the
    resolved access for nothing. What the leaf rule did change was the compiled
    bubblewrap argv, and there it was fatal ONCE THE SOCKET EXISTED: an unreadable
    directory becomes a mode-000 tmpfs remounted read-only, unreadable roots are
    applied parent-first, and the leaf's own operation then needs a mkdir inside a
    filesystem already frozen -- so bwrap refused to build ANY sandbox and no Session
    could run a confined command. With the socket absent the same pair built a sandbox
    fine, because a missing leaf outside every writable root produces no bwrap
    operation at all; the runtime binds that socket on every start, so the refusal
    fired for every real Session. Measured with a real listener bound at that path:
    `bwrap: Can't mkdir parents for /run/codex/ctl/app-server-control.sock:
    Read-only file system`.

    The directory is also the stronger of the two masks, which is why it is the one
    kept rather than an accident of which failed. Measured from inside the sandbox
    with only the directory rule: listing, stat, lstat, open, chdir, relative and `..`
    traversal, symlinks, hardlinks, an `openat` through a dirfd, and
    `/proc/PID/{root,cwd}` for every visible pid are all EACCES. With only the leaf
    rule the directory is world-listable, the socket's exact filename is readable out
    of it, and three `/proc` routes reach the parent. See ADR-012's Status for what
    that means for its item 3.
    """
    return PermissionProfile(
        name=PROFILE_NAME,
        extends=":read-only",
        rules=(
            FsRule(path=WORKSPACE_ROOT, access=FsAccess.WRITE),
            FsRule(path=f"{WORKSPACE_ROOT}/.codex", access=FsAccess.DENY),
            FsRule(path=f"{WORKSPACE_ROOT}/.agents", access=FsAccess.DENY),
            FsRule(path=CONTROL_SOCKET_DIR, access=FsAccess.DENY),
            FsRule(path=CODEX_HOME, access=FsAccess.DENY),
            FsRule(path=SYSTEM_CONFIG_DIR, access=FsAccess.READ),
        ),
    )


def compile_session_config(
    record: SessionRecord,
    *,
    tool_gateway_url: str,
    model_gateway_url: str,
    environment: Environment,
    definition: AgentDefinition,
    session_token_key: bytes,
    session_token_expiry_epoch_s: int,
    skill_files: tuple[SkillFile, ...] = (),
    resuming: bool = False,
) -> CompiledConfig:
    """Compile the configuration this Session's pod runs under, in a named shape.

    The result is immutable for the Session's whole life, which is what lets the pod
    be started from it and then denied any path it was written from. It is checked on
    the way out, so a violated floor is a compilation failure rather than a document
    that reaches a pod and is graded later.

    The definition supplies exactly one thing: the model. Taking the whole value rather
    than a model string rules out one mistake and not the one it looks like it rules
    out: a caller cannot assemble a model from nowhere, but nothing here can tell
    whether this is the definition the Session pinned. `AgentDefinition` carries no id
    and no revision, so there is nothing to compare `record.definition_id` and
    `record.definition_revision` against, and no comparison is made. Resolving the
    current revision of a definition whose Session pins an older one would compile a
    model this Session did not choose, and it would compile cleanly. Closing that needs
    an identity on the value or the revision passed alongside it; until then the caller
    owns it.

    The provider is not the definition's to choose: `MODEL_PROVIDER_ID` says why. The
    address that provider is reached at is deployment configuration, which is why it
    arrives beside the Tool Gateway's rather than as a constant here -- one cluster's
    Model Gateway Service is not another's.

    The two token arguments have no defaults, and neither omission is an oversight. A
    default signing key would let every process that imports this mint a token every
    other process accepts. A default expiry would put a security parameter where no
    operator can see it -- and this token cannot be refreshed, because the document is
    copied into the pod at start and the runtime reads it once, so the expiry is a
    ceiling on how long the pod can use enterprise tools rather than a window that rolls
    forward. When it passes, the Session stops taking Turns: the Gateway is a `required`
    server, so a refusal fails the thread rather than losing one tool.

    The skills are carried, not read: nothing here parses or validates one. They were
    parsed where they were submitted and resolved against a definition before this was
    called, and passing them through is what keeps a Session's shape and its skills one
    value. Defaulted to none, so every caller written before skills existed still
    compiles -- and the one caller that must not silently pass none is graded where it
    resolves them, not by this signature.

    The environment supplies exactly two things and nothing else: the image the pod is
    started from, and additional deny rules appended to the platform's profile. Both
    only narrow. An entry can add a deny strictly under the writable root -- the
    registry that parses an Environment refuses anything else -- so no forbidden key
    appears, no second MCP server appears, the argv does not change, and every floor
    below still holds.

    **`allowed_domains` is the exception, and it widens.** Until 2026-08-23 this
    docstring said network reach was deliberately absent here because the Egress Policy
    is compiled from the Grant -- wrong in a way worth recording rather than quietly
    replacing. The Grant governs which TOOLS a Session may call through the Tool
    Gateway. It says nothing about the agent's own shell commands, which reach the
    network through the sandbox and not through any gateway, and for which the answer
    was not "compiled from the Grant" but "no network at all, undocumented". A skill
    telling the agent to install a package could not be followed, and nothing said why.

    So this is a second source for a rule the Grant never covered. Empty is still no
    network, which keeps the default the safe one; a non-empty list turns egress on and
    confines it to those names, with `managed_allowed_domains_only` set so no config
    layer inside the pod can add to the list this document names.
    """
    platform_profile = session_profile()
    profile = PermissionProfile(
        name=platform_profile.name,
        extends=platform_profile.extends,
        rules=platform_profile.rules
        + tuple(
            FsRule(path=path, access=FsAccess.DENY) for path in environment.denied_paths
        ),
    )
    # Re-raised as this module's own refusal, and not merely relabelled. A caller of
    # the compiler catches `FloorViolation` and nothing else, so a refusal escaping
    # under another name reaches a tenant as a bare 500 with no Turn closed behind it.
    # The cause is chained, so the message an operator reads is still the one that
    # names the base URL and why it would reach nothing.
    #
    # One token, minted once, presented to both gateways. They verify the same layout
    # with the same key, so a second mint here would be a second credential proving the
    # identical two facts -- and the pod would carry two things to leak instead of one.
    session_token = mint_session_token(
        session_id=record.id,
        tenant_id=record.tenant_id,
        expiry_epoch_s=session_token_expiry_epoch_s,
        key=session_token_key,
    )
    try:
        model_selection = render_model_selection(
            bind_session(record, definition),
            gateway_base_url=model_gateway_url,
            session_token=session_token,
        )
    except ModelBindingViolation as refused:
        raise FloorViolation(str(refused)) from refused
    compiled = CompiledConfig(
        session_id=record.id,
        tenant_id=record.tenant_id,
        resuming=resuming,
        runtime_image=environment.runtime_image,
        model=definition.model,
        model_provider=MODEL_PROVIDER_ID,
        config_toml=_render_config(
            workspace_root=WORKSPACE_ROOT,
            gateway_url=tool_gateway_url,
            model_selection=model_selection,
            session_token=session_token,
        ),
        requirements_toml=_render_requirements(
            profile,
            gateway_url=tool_gateway_url,
            allowed_domains=environment.allowed_domains,
        ),
        launch_argv=(
            "codex",
            "app-server",
            _LISTEN_FLAG,
            f"{_UNIX_SCHEME}{CONTROL_SOCKET}",
        ),
        skill_files=skill_files,
    )
    check_floors(compiled)
    return compiled
