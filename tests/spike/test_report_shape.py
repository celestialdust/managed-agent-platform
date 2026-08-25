"""Guard on the probe transcript: every finding present, and no two collapsed into one.

The probe answers one question — is the in-pod sandbox a security boundary on a hardened
node — and it answers it by emitting one `key=value` finding per check.
Two failure modes this guards against:

**A silently skipped check reads as a pass.** The probe is a shell script whose findings
are printed, not returned, so a check that never ran leaves no trace except an absent
key. Requiring the full key set means a dropped assertion fails here instead of quietly
shrinking the evidence behind the verdict.

**Two different failures wearing one name.** A missing `unshare` binary exits 127 and a
kernel that refuses user namespaces exits non-zero too. Collapsed into a single `FAIL`
they are indistinguishable, and the wrong reading of the pair says the architecture does
not work on a node that works. The probe emits `unshare-tool=MISSING-FROM-IMAGE` and
`unshare-userns=REFUSED-BY-KERNEL` as separate keys with separate values, and the last
test here is what stops them being merged back together.

The transcript under test is the real one the pod produced, kept verbatim in the spike
record's `## Probe transcript` section. Reading it from there rather than from a fixture
is deliberate: the record is what later slices read, so this asserts the shape of the
bytes they will actually see.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import pytest

RECORD = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "features"
    / "managed-agent-platform"
    / "spike-bubblewrap.md"
)

PROBE = Path(__file__).resolve().parents[2] / "deploy" / "spike" / "probe.sh"

# The words the probe uses to report that a boundary did not hold. Matched against the
# whole emission line rather than against a parsed value: a value computed in a subshell
# contains nested quotes, the value regex stops at the first of them, and the failure
# word ends up in the fragment the scan never sees. That is not a hypothetical — it is
# how `sandbox-backend=...not-bwrap` sat on a `say` with a test that could not see it.
_FAILURE_WORDS = (
    "ALLOWED",
    "DROPPED",
    "FAIL",
    "MISSING",
    "REFUSED-BY-KERNEL",
    "not-bwrap",
)


def _probe_source() -> str:
    """The probe script's text, for the tests that scan emissions rather than results.

    One reader for four scanners. They had three separate copies of this path, which is
    three chances for one of them to be reading a file that no longer exists while still
    reporting green.
    """
    return PROBE.read_text()


# One key per finding the verdict rests on. `unshare-tool` and `bwrap-tool` are separate
# from `unshare-userns` and `bwrap-usable` on purpose: the first of each pair says the
# tool is there, and only then is the second allowed to mean anything about the kernel.
REQUIRED_KEYS = frozenset(
    {
        # 4a — the host
        "seccomp-mode",
        "serviceaccount-token-absent",
        "max-user-namespaces",
        "unshare-tool",
        "unshare-userns",
        "unshare-userns-mount",
        "bwrap-tool",
        "bwrap-usable",
        # 4b — the runtime's own limits, read off the binary
        "codex-version",
        "sandbox-backend",
        "mcp-startup-timeout-sec",
        "mcp-tool-timeout-sec",
        "mcp-timeout-override-accepted",
        "tool-output-cap",
        "tool-output-cap-config-key",
        # 4c — the deny rule
        "deny-rule-in-argv",
        # The sound form of the assertion above. `deny-rule-in-argv` was a substring
        # match over the whole argv capture, which the socket path satisfies from inside
        # a --setenv value with no rule compiled at all; this one carries the three
        # adjacent tokens, so its shape can be checked rather than its presence assumed.
        "deny-rule-argv-form",
        "sandbox-cannot-read-own-policy",
        "sandbox-cannot-read-inbound-capture",
        "polarity-control",
        "polarity-control-cleanup",
        "write-grant-on-a-regular-file",
        "write-inside-profile",
        "write-outside-profile",
        "no-partial-write",
        "deny-rule-in-argv-target-absent",
        "write-outside-profile-target-absent",
        "sandbox-network",
        # 4d — p
        "rss-kb-idle",
        "rss-kb-peak-5mb",
        "cgroup-memory-peak-kb",
        "node-memory-total-kb",
    }
)

_FINDING = re.compile(r"^([a-z0-9][a-z0-9-]*)=(.*)$")


def parse_findings(transcript: str) -> dict[str, str]:
    """Read `key=value` findings out of a probe transcript, ignoring its prose.

    The probe interleaves section headings and raw command output with its findings, so
    anything that is not a `key=value` line at column zero is skipped rather than
    treated as a parse error — the transcript is a log a person reads as well as a
    record a test reads.
    """
    found: dict[str, str] = {}
    for line in transcript.splitlines():
        match = _FINDING.match(line.strip())
        if match:
            found[match.group(1)] = match.group(2).strip()
    return found


def extract_probe_transcript(record_text: str) -> str:
    """Return the fenced block under the record's `## Probe transcript` heading.

    Scoped to that one section because the record carries other fenced blocks — the
    provider-facing exchange among them — and a finding accidentally read out of one of
    those would not have come from the probe at all.
    """
    section = re.split(r"^## Probe transcript\s*$", record_text, maxsplit=1, flags=re.M)
    if len(section) != 2:
        raise AssertionError("the record has no '## Probe transcript' section")
    fenced = re.search(r"^```[a-z]*\n(.*?)^```", section[1], flags=re.M | re.S)
    if fenced is None:
        raise AssertionError("'## Probe transcript' carries no fenced block")
    return fenced.group(1)


@pytest.fixture(scope="module")
def findings() -> dict[str, str]:
    if not RECORD.exists():
        pytest.fail(f"the spike record does not exist at {RECORD}")
    return parse_findings(extract_probe_transcript(RECORD.read_text()))


def test_every_finding_is_present(findings: dict[str, str]) -> None:
    missing = sorted(REQUIRED_KEYS - findings.keys())
    assert not missing, (
        f"the transcript is missing {len(missing)} finding(s): {missing}. "
        "A check that did not run leaves no trace but an absent key, so an "
        "incomplete transcript must not read as a passing one."
    )


def test_no_finding_is_empty(findings: dict[str, str]) -> None:
    blank = sorted(k for k in REQUIRED_KEYS if not findings.get(k, "").strip())
    assert not blank, f"findings present but with no value: {blank}"


def test_the_tool_check_precedes_the_kernel_verdict(findings: dict[str, str]) -> None:
    """A kernel verdict means something only once its tool is known to be present.

    Recorded as an assertion rather than a comment because the ordering is the whole
    reason the two keys exist separately.
    """
    assert findings["unshare-tool"] == "present"
    assert findings["bwrap-tool"] == "present"


def test_a_missing_binary_and_a_refusing_kernel_are_distinguishable() -> None:
    """The two failures the original single check collapsed into one must stay apart.

    Synthetic on purpose: the node under test has both binaries and a permitting kernel,
    so neither failure can be provoked there, and the property being guarded is about
    the transcript's vocabulary rather than about that node.
    """
    missing_binary = parse_findings("unshare-tool=MISSING-FROM-IMAGE\n")
    refusing_kernel = parse_findings(
        "unshare-tool=present\nunshare-userns=REFUSED-BY-KERNEL rc=1\n"
    )

    assert missing_binary["unshare-tool"] == "MISSING-FROM-IMAGE"
    assert "unshare-userns" not in missing_binary, (
        "a report that never proved the tool present must not also carry a "
        "kernel verdict, or the two failures read as one again"
    )

    assert refusing_kernel["unshare-tool"] == "present"
    assert refusing_kernel["unshare-userns"].startswith("REFUSED-BY-KERNEL")

    assert missing_binary["unshare-tool"] != refusing_kernel["unshare-tool"]


def test_no_required_finding_can_only_ever_say_one_thing() -> None:
    """A finding with exactly one possible value is not measuring anything.

    This is the shape of a real defect. `tool-output-cap-is-a-config-key=no` was a
    string constant emitted unconditionally; read as evidence it said the Agent
    Runtime's output cap could not be raised, and it retired the mechanism ADR-019 rests
    on until a later run measured the opposite (`model_catalog_json` does raise it).

    The test is not "does the value interpolate" — `say "unshare-userns=ok"` is a
    literal and is a perfectly good measurement, because the `if` that guards it is the
    measurement and the `else` branch emits `REFUSED-BY-KERNEL`. What distinguishes the
    two is *arity*: a key the script can emit two different ways is deciding between
    them at runtime, and a key it can only emit one way has already decided.

    So each required key must either interpolate something, or have two or more distinct
    emission sites. Counting `die` as well as `say`: a finding whose other branch aborts
    the run is still a finding with two outcomes.
    """
    probe = _probe_source()
    emissions: dict[str, set[str]] = {}
    interpolated: set[str] = set()
    for line in probe.splitlines():
        for match in re.finditer(r'(?:say|die)\s+"([a-z0-9][a-z0-9-]*)=([^"]*)"', line):
            key, value = match.group(1), match.group(2)
            if key not in REQUIRED_KEYS:
                continue
            emissions.setdefault(key, set()).add(value)
            if "$" in value:
                interpolated.add(key)
    frozen = sorted(
        key
        for key, values in emissions.items()
        if key not in interpolated and len(values) < 2
    )
    assert frozen == [], (
        f"required findings the probe can only emit one way: {frozen}. "
        "A conclusion decided before the run is prose and belongs in the record's "
        "Conditions section, not in the key=value stream."
    )


def test_every_required_finding_has_an_emission_site() -> None:
    """A key required here that the probe never emits is a test asserting nothing.

    The pairing runs the other way too: `REQUIRED_KEYS` is edited by hand, and a key
    added without a matching `say` makes the transcript check fail for a reason that has
    nothing to do with the node under test.
    """
    probe = _probe_source()
    emitted = {
        m.group(1) for m in re.finditer(r'(?:say|die)\s+"([a-z0-9][a-z0-9-]*)=', probe)
    }
    missing = sorted(REQUIRED_KEYS - emitted)
    assert missing == [], (
        f"required by this test but never emitted by probe.sh: {missing}"
    )


# The verdict this slice exists to establish, as key=value pairs that must hold. Split
# from REQUIRED_KEYS deliberately: that set asks whether a check ran, and this one asks
# what it answered. A transcript can satisfy every presence check while saying the
# boundary leaked, which is exactly what a mutation test of this file found it doing.
SECURITY_OUTCOMES = {
    # The posture every other line here is relative to. Unpinned, this whole map read
    # the same at seccomp 0 and seccomp 2 — and the two are not comparable: under
    # RuntimeDefault the sandbox cannot be built at all, so a transcript claiming the
    # boundary held would be claiming it about a sandbox that never existed.
    "seccomp-mode": "0",
    # the pod holds no credential (I1), measured rather than trusted from the manifest
    "serviceaccount-token-absent": "ok",
    # the kernel permits the namespaces the sandbox is built from
    "unshare-userns": "ok",
    "unshare-userns-mount": "ok",
    "bwrap-usable": "ok",
    # the profile is enforcing, and is not merely refusing everything
    "write-inside-profile": "ok",
    "write-outside-profile": "refused",
    "no-partial-write": "ok",
    # the deny rule reaches the compiled argv, and still holds with its target absent
    "deny-rule-in-argv": "ok",
    "deny-rule-in-argv-target-absent": "present",
    "write-outside-profile-target-absent": "refused",
    # the confined process sees no interface but loopback
    "sandbox-network": "lo,",
    # the premise: the sandbox was built by bubblewrap at all
    "sandbox-backend": "bwrap",
    # the two routes around the recorder's redaction, rather than through it: the policy
    # confining the command, and the captured provider exchange with its bearer header
    "sandbox-cannot-read-own-policy": "ok",
    "sandbox-cannot-read-inbound-capture": "ok",
    # The negative control: the same write, under a profile that permits it, must
    # succeed. This is what pins polarity rather than vocabulary. Swapping the branch
    # bodies of write-outside-profile leaves every other check here green -- they grade
    # which words the probe can emit and which sit on a `die`, and a swap changes
    # neither -- but it contradicts this, because the two run under policies that differ
    # in exactly one rule.
    "polarity-control": "ok",
    "polarity-control-cleanup": "ok",
}

# Findings whose value is not a fixed string but still decides something. Each was a
# required key with its presence graded and its answer ungraded, and each of these
# predicates was written against a transcript mutation that passed the whole suite:
# seccomp-mode 0 -> 2 (a posture under which no confined command runs at all), tool-
# output-cap-config-key -> none-of-8-tried (ADR-019's lever does not exist), mcp-
# timeout-override-accepted -> REFUSED, max-user-namespaces -> 0 (a kernel permitting no
# user namespace, i.e. no sandbox), cgroup-memory-peak-kb -> 2000 (which turns p from 15
# into four figures).
OUTCOME_PREDICATES: dict[str, tuple[str, Callable[[str], bool]]] = {
    "max-user-namespaces": (
        "a positive count -- the sandbox is built from user namespaces",
        lambda v: v.isdigit() and int(v) > 0,
    ),
    "tool-output-cap-config-key": (
        "a key that was found, not none-of-N-tried",
        lambda v: bool(v) and not v.startswith("none-of-"),
    ),
    "mcp-timeout-override-accepted": ("ok", lambda v: v == "ok"),
    "cgroup-memory-peak-kb": (
        "a plausible cgroup peak, at least 32 MiB",
        lambda v: v.isdigit() and int(v) >= 32 * 1024,
    ),
    "node-memory-total-kb": (
        "a plausible node total, at least 1 GiB",
        lambda v: v.isdigit() and int(v) >= 1024 * 1024,
    ),
}

# The deny rule's shape, as three adjacent argv tokens. The source is not pinned — it is
# a file descriptor number the runtime picks — but the option must be a bind and the
# target must be the exact masked path, which is what makes this stronger than the
# substring match it replaces.
DENY_FORM = re.compile(r"^--[a-z-]*bind[a-z-]*\s+\S+\s+/run/map/app-server\.sock$")


def test_the_transcript_says_the_boundary_held(findings: dict[str, str]) -> None:
    """Every security finding has the value that means the boundary held.

    This is the test the suite was missing, and its absence was not theoretical: a
    mutation test rewrote the transcript to `write-outside-profile=ALLOWED`, `no-
    partial-write=FAIL bytes=1`, `deny-rule-in-argv=ABSENT` and `unshare-userns=REFUSED-
    BY-KERNEL rc=1`, and all seven tests still passed. They asked whether each check had
    run, never what it answered — so the one property this slice exists to establish was
    the one property nothing could fail on.

    Asserted as an exact map rather than a loop over substrings: `refused` and `REFUSED-
    BY-KERNEL` differ by which side of the boundary refused, and a substring match would
    read the second as the first.
    """
    wrong = {
        key: findings.get(key, "<absent>")
        for key, expected in SECURITY_OUTCOMES.items()
        if findings.get(key) != expected
    }
    assert wrong == {}, (
        f"the transcript does not say the boundary held: {wrong}. "
        f"expected { ({k: SECURITY_OUTCOMES[k] for k in wrong}) }"
    )


def test_every_security_outcome_is_also_a_required_finding() -> None:
    """The two sets must not drift: an outcome nobody requires can go missing silently.

    If a key leaves `REQUIRED_KEYS` but stays here, an absent finding reads as
    `<absent>` and still fails — which is correct. The reverse is the danger, so this
    pins the containment rather than trusting it.
    """
    stray = sorted(set(SECURITY_OUTCOMES) - REQUIRED_KEYS)
    assert stray == [], f"security outcomes not in REQUIRED_KEYS: {stray}"


def test_every_fail_open_branch_in_the_probe_aborts_the_run() -> None:
    """A fail-open finding must `die`, not `say`.

    A probe that prints `write-outside-profile-target-absent=ALLOWED` and exits 0 hands
    back a Succeeded pod whose transcript says the boundary leaked, and then whether
    anyone reads the transcript is the only thing standing between the leak and the
    plan. The pod's exit status is the one signal nothing has to remember to check.

    Detected by value, not by line number: any emission whose value is one of the words
    the probe uses for a failure must be on a `die`.
    """
    probe = _probe_source()
    leaked = [
        f"{m.group(2)} ({word})"
        for line in probe.splitlines()
        if (m := re.match(r'\s*(say|die)\s+"([a-z0-9-]+)=', line))
        and m.group(1) == "say"
        and (word := next((w for w in _FAILURE_WORDS if w in line), None))
    ]
    assert leaked == [], (
        f"fail-open findings emitted with `say` rather than `die`: {leaked}. "
        "A probe that reports a leak and exits 0 makes the pod's status a lie."
    )


def test_the_deny_rule_has_the_shape_of_a_rule(findings: dict[str, str]) -> None:
    """The compiled deny rule is a bind option, a source, and the masked path.

    `deny-rule-in-argv=ok` used to come from `grep -q "$SOCK" argv.txt`, and the socket
    path appears in that file for reasons that are not a deny rule — inside a --setenv
    value, and inside a bind of the directory containing it. So the slice's load-bearing
    assertion could report ok with no rule compiled. This checks the rule's shape, which
    the substring could not distinguish from its mention.
    """
    form = findings.get("deny-rule-argv-form", "<absent>")
    assert DENY_FORM.match(form), (
        f"deny-rule-argv-form is {form!r}, which is not a bind rule over "
        "/run/map/app-server.sock. A mention of the path is not a rule masking it."
    )


def test_every_outcome_predicate_holds(findings: dict[str, str]) -> None:
    """Findings whose answer is a number or an open string still have to be graded.

    SECURITY_OUTCOMES can only assert fixed strings. These five carry values that vary
    between runs and still decide something -- and each one was demonstrated, by
    rewriting the committed transcript, to be substitutable with a value meaning the
    opposite while the whole suite stayed green.
    """
    wrong = {
        key: (findings.get(key, "<absent>"), expectation)
        for key, (expectation, holds) in OUTCOME_PREDICATES.items()
        if not holds(findings.get(key, ""))
    }
    assert wrong == {}, (
        f"findings whose value does not support the record's conclusion: {wrong}. "
        "Each pair is (what the transcript says, what the conclusion needs)."
    )


def test_p_is_the_arithmetic_the_measurement_supports(findings: dict[str, str]) -> None:
    """`p` is re-derived rather than read beside its inputs.

    Two figures could be the denominator and only one is right. The transcript's `node-
    memory-total-kb` is `MemTotal` from `/proc/meminfo`, which is the node's *capacity*;
    Kubernetes places pods against *allocatable*, which is smaller by the system
    reservation and the eviction threshold. On this node that is 3931692 against 3376684
    — dividing by capacity gives 17 instead of 15 and overstates p by 16%. The probe
    cannot read allocatable: that is a Kubernetes figure and the pod deliberately holds
    no ServiceAccount token, so it comes from `kubectl` and is recorded with that
    provenance. This test is what stops the two being swapped, in either direction.
    """
    text = RECORD.read_text()
    alloc = re.search(r"allocatable[^0-9]{0,4}(\d{6,})Ki", text)
    assert alloc is not None, (
        "the record does not state the node's allocatable memory, so `p` has no "
        "denominator a reader can check"
    )
    allocatable = int(alloc.group(1))
    total = int(findings["node-memory-total-kb"])
    assert allocatable < total, (
        f"allocatable ({allocatable}) is not below MemTotal ({total}); one of the two "
        "figures is mislabelled and p would be derived from the wrong one"
    )
    peak = int(findings["cgroup-memory-peak-kb"])
    derived = allocatable // peak
    claimed = re.search(r"^\s*(?:\d+\.\s+)?\*\*`?p = (\d+)`?", text, flags=re.M)
    assert claimed is not None, "the record states no `p = <n>`"
    assert derived == int(claimed.group(1)), (
        f"record claims p = {claimed.group(1)}; allocatable / this run peak "
        f"is {allocatable} / {peak} = {derived}. One was measured on a different run."
    )


def test_the_capture_withholds_a_header_it_was_not_told_is_safe() -> None:
    """The recorder's redaction is an allowlist, not a denylist.

    The capture file is read back into a transcript that is committed, so a header
    value reaching it reaches the repository. A denylist of known secret-carrying names
    is wrong by default in the one case that matters -- the header nobody thought of. A
    provider adding `x-session-key`, a proxy adding `proxy-authorization`, or a runtime
    version bump adding anything at all would each write a live credential into a
    committed file, and no test would fail.

    Checked against the probe's source rather than against a transcript, because the
    transcript only shows the headers that *were* sent. It cannot show what would happen
    to one that was not, and that is the whole risk.
    """
    probe = _probe_source()
    assert "PRINTABLE_HEADERS" in probe, (
        "the recorder has no allowlist; a denylist admits every header nobody named"
    )
    assert "not in PRINTABLE_HEADERS" in probe, (
        "PRINTABLE_HEADERS exists but nothing tests non-membership, so the default is "
        "still to print"
    )
    # The withholding branch has to be the one that fires for an unlisted name.
    # Asserting the two appear together in that order is what separates "there is an
    # allowlist" from "there is an allowlist and it is what decides".
    guard = probe.index("not in PRINTABLE_HEADERS")
    withhold = probe.index("<withheld>", guard)
    assert withhold - guard < 120, (
        "the non-membership test is not followed by a withholding assignment; the "
        f"nearest <withheld> is {withhold - guard} characters away"
    )
    assert "SECRET_HEADERS" not in probe, (
        "the old denylist is still present; two redaction rules mean the weaker one "
        "wins for any header the other does not name"
    )
