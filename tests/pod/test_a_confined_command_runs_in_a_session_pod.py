"""A confined command running in a real Session pod, and what it still cannot reach.

Skipped unless MAP_CLUSTER_TESTS=1. SKIPPED MEANS NO CONTAINER RAN -- every other test
over this configuration reads TOML or YAML, and neither can say whether bubblewrap
accepted an argv.

Four pods, and three of them are controls, because without them this file is a list of
refusals that a broken pod would satisfy.

1. A real AF_UNIX listener is bound at the control-socket path and connected to from
   OUTSIDE the sandbox in the same run, in every pod. Every refusal measured inside is
   then a refusal over a real, reachable inode rather than over an absent path -- which
   is a different and much weaker finding, and it is also the difference that decides
   the refusal this slice removes: with the socket ABSENT the configuration that
   shipped before this slice builds a sandbox fine, and only once the leaf exists does
   bubblewrap refuse. The runtime binds it on every start, so a real Session is always
   in the second state.
2. A second AF_UNIX listener in the writable root, connected to from outside as well.
   Without it, `connect -> EPERM` from inside reads as evidence for the deny rules and
   is nothing of the kind: the runtime installs its own network seccomp filter that
   denies connect, bind, listen and sendto unconditionally for every address family.
   The same run measures that EPERM over a writable inode and over an
   abstract-namespace socket with no inode at all, so this file records the refusal as
   NOT attributable to the filesystem rules, and the control is what earns the right
   to say so.
3. A second pod mounting the document as the compiler emitted it BEFORE this slice --
   the leaf control-socket rule restored -- where bubblewrap must refuse to build any
   sandbox. Two of this file's findings are absences, and an absence is satisfied by a
   pod that never pulled its image.

The third and fourth pods are one pair, and they are about the FLOOR rather than about
the pod. `check_floors` refuses a deny set that nests one path inside another, and two
things about that refusal are claims rather than tastes. First, that a nested pair is
worth refusing at all -- and it is not always fatal: a deny at `/run/codex/deeper`
beside the deny at `/run/codex` runs a confined command perfectly well while nothing
has created `deeper`. What makes it fatal is the descendant EXISTING when the argv is
compiled, which is why the pair is measured in ONE pod either side of a mkdir. The
control socket is that same shape with the runtime creating the target on every start,
so refusing the class is the only decidable form of the rule: whether a path will exist
is not knowable in the control plane, where the compiler runs.

Second, that "inside" is decided by comparing path components and not string prefixes.
`/run//codex/deeper` does not start with the string `/run/codex/`, so a string
comparator reports that pair un-nested; the fourth pod carries exactly that spelling
and is fatal in the same state as the third, with bubblewrap naming the normalised path
in its own message. This repository's most expensive recurring defect is a guard that
grades one spelling of a value while the thing consuming it accepts several, so the
question is settled by measurement rather than by reading.

What this file cannot say: that a Turn works. It drives `codex sandbox`, which compiles
the same argv through the same helper, in the same container, at the same uid and under
the same seccomp profile -- and is not the caller a Turn uses. It says nothing about the
arg0 helper directory a Turn's patch tooling needs, which sits on PATH inside a tree
this platform denies wholesale and is a further refusal behind these.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import yaml

# The wrappers, imported rather than copied. The previous slice wrote them as the second
# copy of the seccomp suite's; a third would be the Rule of Three. Each takes the
# namespace it works in, so the two modules never share one. The coupling adds no
# ordering constraint that the Blocked-by edge between the two slices does not already
# impose.
#
# By bare module name because `tests/` carries no `__init__.py`: pytest imports a test
# file by inserting its own directory on sys.path, so `tests.pod.<module>` is not a
# name that resolves here and this one is.
from test_the_pod_materialises_its_sandbox_targets import (
    CODEX_VERSION,
    _compiled,
    _image,
    _kubectl,
    _probe_pod,
    _secret,
    _transcript,
    requires_the_cluster,
)

from managed_agent.control.pod_config import compiler as config_compiler

_NAMESPACE = "map-68-confined"

CONTROL_PATH = "Can't mkdir parents for /run/codex/ctl"
RAN = "CONFINED-COMMAND-RAN"
SOCK = config_compiler.CONTROL_SOCKET
WS = config_compiler.WORKSPACE_ROOT
SYSCONF = config_compiler.SYSTEM_CONFIG_DIR
CODEX_HOME = config_compiler.CODEX_HOME

# The two labels whose verdict is allowed to be REACHED, and each is a finding rather
# than an exemption. A7 reads the mode of the mask itself, which the confined process
# can see and must see as `d---------`; B10 is a sweep whose value is the list of pids
# that reached, and the assertion on it is that the list is empty.
_MAY_BE_REACHED = frozenset({"A7-stat-run-codex", "B10-proc-pid-sweep-reached"})

NESTED_TARGET = f"{config_compiler.CONTROL_SOCKET_DIR}/deeper"
"""A path one segment under the surviving control-path deny, which nothing creates.

The control socket is the same shape and the runtime creates it on every start; this
one is created by the probe, on demand, which is what makes the existence of a nested
deny target a variable a single pod can turn.
"""

DOUBLED_TARGET = config_compiler.CONTROL_SOCKET_DIR.replace("/codex", "//codex", 1) + (
    "/deeper"
)
"""The same path with a doubled separator, which is the spelling question.

`/run//codex/deeper` does not start with the string `/run/codex/`, so a nesting check
that compared strings would report the pair un-nested. Whether that is a real gap is
not a matter of opinion: it depends on what the runtime does with the spelling, and the
pod pair beside it measures exactly that.
"""

ABSENT_RAN = "RAN-WITH-TARGET-ABSENT"
PRESENT_RAN = "RAN-WITH-TARGET-PRESENT"

_BINDER = f'''
import os, socket, threading, time

for path in ("{SOCK}", "{WS}/probe.sock"):
    if os.path.lexists(path):
        os.unlink(path)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    server.listen(8)
    os.chmod(path, 0o755)

    def serve(sock=server):
        while True:
            conn, _ = sock.accept()
            conn.recv(64)
            conn.sendall(b"REACHED")
            conn.close()

    threading.Thread(target=serve, daemon=True).start()
    print("bound " + path, flush=True)

time.sleep(900)
'''

_OUTSIDE = f'''
import socket, stat, os


def say(label, call):
    try:
        value = call()
    except BaseException as err:
        code = getattr(err, "errno", None)
        print(f"{{label}}=REFUSED exc={{type(err).__name__}} errno={{code}}")
    else:
        print(f"{{label}}=REACHED value={{value!r}}")


def connect(path):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(5)
    sock.connect(path)
    sock.sendall(b"hello")
    answer = sock.recv(64)
    sock.close()
    return answer.decode()


say("P-stat-control-socket", lambda: stat.filemode(os.stat("{SOCK}").st_mode))
say("P-connect-control-socket", lambda: connect("{SOCK}"))
say("P-connect-workspace-socket", lambda: connect("{WS}/probe.sock"))
print("outside=complete")
'''

_INSIDE = f'''
import os, socket, stat

SOCK = "{SOCK}"
CTL = os.path.dirname(SOCK)
RUN = os.path.dirname(CTL)
WS = "{WS}"
SYSCONF = "{SYSCONF}"
CODEX_HOME = "{CODEX_HOME}"


def say(label, call):
    """One labelled line per call, and never a raise: a refusal is a finding here."""
    try:
        value = call()
    except BaseException as err:
        code = getattr(err, "errno", None)
        print(f"{{label}}=REFUSED exc={{type(err).__name__}} errno={{code}}")
    else:
        print(f"{{label}}=REACHED value={{value!r}}")


def connect(path):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(5)
    sock.connect(path)
    sock.sendall(b"hello")
    answer = sock.recv(64)
    sock.close()
    return answer.decode()


def link(kind, target, name):
    """A second name for the same inode, made in the one writable root."""
    where = f"{{WS}}/{{name}}"
    if os.path.lexists(where):
        os.unlink(where)
    if kind == "symlink":
        os.symlink(target, where)
    else:
        os.link(target, where)
    return where


# A -- the control path, through the routes that name it directly
say("A1-listdir-run-codex", lambda: sorted(os.listdir(RUN)))
say("A2-listdir-ctl", lambda: sorted(os.listdir(CTL)))
say("A3-stat-socket", lambda: stat.filemode(os.stat(SOCK).st_mode))
say("A4-lstat-socket", lambda: stat.filemode(os.lstat(SOCK).st_mode))
say("A5-open-socket", lambda: os.close(os.open(SOCK, os.O_RDONLY)) or "opened")
say("A6-connect-socket", lambda: connect(SOCK))
say("A7-stat-run-codex", lambda: stat.filemode(os.stat(RUN).st_mode))


# B -- the same inode reached by another name
def chdir_in():
    os.chdir(RUN)
    return os.getcwd()


say("B1-chdir", chdir_in)
os.chdir("/")


def relative_from_run():
    os.chdir("/run")
    try:
        return sorted(os.listdir("codex"))
    finally:
        os.chdir("/")


say("B2-relative", relative_from_run)
say("B3-dotdot", lambda: sorted(os.listdir(f"{{RUN}}/../codex")))
say(
    "B4-symlink-to-file",
    lambda: stat.filemode(os.stat(link("symlink", SOCK, "s-file")).st_mode),
)
say(
    "B5-symlink-to-directory",
    lambda: sorted(os.listdir(link("symlink", RUN, "s-dir"))),
)
say(
    "B6-hardlink",
    lambda: stat.filemode(os.stat(link("hard", SOCK, "h-file")).st_mode),
)


def openat_through_a_dirfd():
    fd = os.open("/run", os.O_RDONLY)
    try:
        return sorted(os.listdir(os.open("codex", os.O_RDONLY, dir_fd=fd)))
    finally:
        os.close(fd)


say("B7-openat-dirfd", openat_through_a_dirfd)
say("B8-proc-self-root", lambda: sorted(os.listdir(f"/proc/self/root{{RUN}}")))
say("B9-proc-1-root", lambda: sorted(os.listdir(f"/proc/1/root{{RUN}}")))

reached = []
pids = sorted(p for p in os.listdir("/proc") if p.isdigit())
for pid in pids:
    for name in ("root", "cwd"):
        try:
            os.listdir(f"/proc/{{pid}}/{{name}}{{RUN}}")
        except BaseException:
            continue
        reached.append(f"{{pid}}/{{name}}")
print(f"B10-proc-pid-sweep-reached={{reached!r}} over={{len(pids)}}")


# C -- the two workspace masks this slice must not move
for dot in (".agents", ".codex"):
    where = f"{{WS}}/{{dot}}"
    say(f"C-mode{{dot}}", lambda w=where: stat.filemode(os.stat(w).st_mode))
    say(f"C-listdir{{dot}}", lambda w=where: sorted(os.listdir(w)))
    say(f"C-read{{dot}}", lambda w=where: open(f"{{w}}/seeded").read())
    say(f"C-write{{dot}}", lambda w=where: open(f"{{w}}/written", "w").write("x"))


# D -- what actually refuses connect, and it is not a deny rule
def abstract_round_trip():
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind("\\0map-68-abstract")
    server.listen(1)
    return connect("\\0map-68-abstract")


say(
    "D1-socket-af-unix",
    lambda: socket.socket(socket.AF_UNIX, socket.SOCK_STREAM).fileno(),
)
say("D2-connect-a-socket-in-the-WRITABLE-root", lambda: connect(f"{{WS}}/probe.sock"))
say("D3-abstract-namespace-no-inode-at-all", abstract_round_trip)
say(
    "D4-write-a-plain-file-in-the-writable-root",
    lambda: open(f"{{WS}}/plain", "w").write("ok"),
)

# E -- the two configuration directories, which the profile treats differently. E1 and
# E2 are the assertion that MAP's skills are reachable at all: a Host skill's catalogue
# line hands the model a file path and the model opens it with its shell, so a refusal
# here is every skill silently unusable. E3 is what keeps that from being a hole -- the
# document E2 can read carries no credential, and the one that does is under CODEX_HOME.
say("E1-listdir-etc-codex", lambda: sorted(os.listdir(SYSCONF)))
REQUIREMENTS = f"{{SYSCONF}}/requirements.toml"
CONFIG = f"{{CODEX_HOME}}/config.toml"
say("E2-read-managed-requirements", lambda: len(open(REQUIREMENTS).read()))
say("E3-read-session-token-config", lambda: len(open(CONFIG).read()))

status = open("/proc/self/status").read()
for field in ("Seccomp:", "Seccomp_filters:", "NoNewPrivs:"):
    if field in status:
        print(f"{{field.strip(':').lower()}}={{status.split(field)[1].split()[0]}}")
print("inside=complete")
'''

# `set -u` and NOT `set -e`: every refusal below is a non-zero exit the run has to
# survive in order to report it. Each `codex sandbox` redirects to a file and the exit
# code is read on the next line -- never a pipe into `tail`, whose status is tail's.
_PRELUDE = f"""
set -u
cat > /tmp/binder.py <<'BINDER'
{_BINDER}
BINDER
cat > /tmp/outside.py <<'OUTSIDE'
{_OUTSIDE}
OUTSIDE
cat > /tmp/inside.py <<'INSIDE'
{_INSIDE}
INSIDE
# Seeded from outside so the read refusal inside is a refusal over real bytes. A mask
# whose mode is `d---------` over a directory with a readable file under it is a mask in
# name only, and the mode assertion alone cannot tell the two apart -- but neither can a
# refused read, which looks the same whether the bytes were denied or were never there.
# So the seed reports its own success and a case asserts it.
seeded=ok
for d in {WS}/.agents {WS}/.codex; do
  printf 'SEEDED-METADATA-BODY\\n' > "$d/seeded" 2>/dev/null || seeded=FAIL
  test -s "$d/seeded" || seeded=FAIL
done
echo "seeded=$seeded"
echo "codex-version=$(codex --version 2>&1 | head -1)"
python3 /tmp/binder.py > /tmp/binder.log 2>&1 &
i=0
while [ "$i" -lt 30 ]; do
  if [ -S {SOCK} ] && [ -S {WS}/probe.sock ]; then break; fi
  i=$((i + 1))
  sleep 1
done
cat /tmp/binder.log
python3 /tmp/outside.py 2>&1
"""

PROBE = f"""{_PRELUDE}
codex sandbox -P map-session --include-managed-config -C {WS} \\
  -- /bin/echo {RAN} > /tmp/echo.out 2>&1
echo "echo-rc=$?"
cat /tmp/echo.out
codex sandbox -P map-session --include-managed-config -C {WS} \\
  -- python3 /tmp/inside.py > /tmp/inside.out 2>&1
echo "inside-rc=$?"
cat /tmp/inside.out
echo probe=complete
"""

EXISTENCE_PROBE = f"""{_PRELUDE}
# The same document either side of one change on disk. Whether a deny rule's target
# EXISTS when the argv is compiled is the whole variable here, so it is varied inside a
# single pod rather than across two: two pods would also differ in their scheduling,
# their image pull and their Secret, and this differs in a mkdir.
echo "target-before=$(test -e {NESTED_TARGET} && echo present || echo absent)"
codex sandbox -P map-session --include-managed-config -C {WS} \\
  -- /bin/echo {ABSENT_RAN} > /tmp/a.out 2>&1
echo "absent-rc=$?"
cat /tmp/a.out
mkdir -p {NESTED_TARGET} && echo made-target=ok || echo made-target=FAIL
echo "target-after=$(test -d {NESTED_TARGET} && echo directory || echo missing)"
codex sandbox -P map-session --include-managed-config -C {WS} \\
  -- /bin/echo {PRESENT_RAN} > /tmp/b.out 2>&1
echo "present-rc=$?"
cat /tmp/b.out
echo probe=complete
"""


def _with_the_leaf_rule_restored(document: str) -> str:
    """The document as the compiler emitted it before this slice, and no other way.

    The mirror image of the patcher the previous slice carried, and it asserts each of
    its two additions in process rather than leaving a transcript to be debugged: the
    rule is written down twice, a row in the profile table and an entry in the managed
    deny_read, and every deny_read entry is pushed into the same policy the sandbox
    argv is compiled from. Adding only the row would leave the argv unchanged and this
    pod would silently become a duplicate of the first.
    """
    directory = config_compiler.CONTROL_SOCKET_DIR
    row = f'"{directory}" = "deny"'
    assert row in document, "the profile table does not name the control path directory"
    patched = document.replace(row, f'{row}\n"{SOCK}" = "deny"', 1)
    entry = f'"{directory}"'
    assert entry in patched, "the deny_read list does not name the control path"
    patched = patched.replace(entry, f'{entry}, "{SOCK}"', 1)
    assert patched.count(SOCK) == 2, "the leaf rule did not land in both lists"
    return patched


def _with_a_deny_added(document: str, path: str) -> str:
    """One more deny rule, in both lists, at a path spelled exactly as given.

    Spelled as given and never normalised, because the spelling is the subject: these
    two pods ask what the runtime does with `/run//codex/deeper`, and a helper that
    tidied the separator away would answer a different question.

    The anchor is only an insertion point -- any deny row this document already carries
    would do, and CODEX_HOME is chosen because it is the one deny rule with no bearing
    on what these pods measure. The path being added is the subject, and it is under
    the control path rather than under the anchor.
    """
    anchor = f'"{config_compiler.CODEX_HOME}"'
    row = f'{anchor} = "deny"'
    assert row in document, "the profile table does not name the codex home"
    patched = document.replace(row, f'{row}\n"{path}" = "deny"', 1)
    assert anchor in patched
    patched = patched.replace(anchor, f'{anchor}, "{path}"', 1)
    assert patched.count(f'"{path}"') == 2, "the added rule did not land in both lists"
    return patched


@pytest.fixture(scope="module")
def transcripts() -> Iterator[dict[str, str]]:
    """Four pods in a namespace this test creates and deletes, and their transcripts.

    The document is compiled here by calling the compiler, so what the first pod reads
    is the real thing rather than a copy free to differ from it. The other three mount
    a patched copy that differs from it in exactly one rule, rendered into a Secret in
    this namespace under the name the manifest already spells -- so the manifest needs
    no edit, no committed file changes, and nothing in map-dev is read or written.

    Two probe scripts, not one, and each comparison this file makes is between two pods
    that share theirs byte for byte: the first pair differs only in the document, the
    second only in the spelling of one path in it. The two scripts share their whole
    prelude, so the sockets are bound and the seeds are written identically in all four.
    """
    image = _image()
    compiled = _compiled(image)
    plans = {
        "real": (compiled.requirements_toml, PROBE),
        "leaf-restored": (
            _with_the_leaf_rule_restored(compiled.requirements_toml),
            PROBE,
        ),
        "nested-canonical": (
            _with_a_deny_added(compiled.requirements_toml, NESTED_TARGET),
            EXISTENCE_PROBE,
        ),
        "nested-doubled": (
            _with_a_deny_added(compiled.requirements_toml, DOUBLED_TARGET),
            EXISTENCE_PROBE,
        ),
    }
    _kubectl("create", "namespace", _NAMESPACE)
    try:
        _secret(
            "map-session-compiled-config",
            {"config.toml": compiled.config_toml},
            namespace=_NAMESPACE,
        )
        pods = {}
        for label, (document, probe) in plans.items():
            secret = f"map-session-requirements-{label}"
            _secret(
                secret, {"requirements.toml": document + "\n"}, namespace=_NAMESPACE
            )
            pod = _probe_pod(
                f"confined-{label}",
                image=image,
                namespace=_NAMESPACE,
                probe=probe,
            )
            for volume in pod["spec"]["volumes"]:
                if volume["name"] == "requirements":
                    volume["secret"]["secretName"] = secret
            pods[label] = pod
        for pod in pods.values():
            _kubectl("apply", "-n", _NAMESPACE, "-f", "-", stdin=yaml.safe_dump(pod))
        yield {
            label: _transcript(pod["metadata"]["name"], namespace=_NAMESPACE)
            for label, pod in pods.items()
        }
    finally:
        _kubectl("delete", "namespace", _NAMESPACE, "--ignore-not-found", check=False)


def _labelled(transcript: str, prefixes: tuple[str, ...]) -> list[str]:
    return [
        line
        for line in transcript.splitlines()
        if line.startswith(prefixes) and "=" in line
    ]


@requires_the_cluster
def test_every_pod_measured_the_runtime_version_these_claims_belong_to(
    transcripts: dict[str, str],
) -> None:
    """Every pod reports the codex version it ran, and it is the one recorded above.

    Asserted per pod rather than once, because the pods are built from separately
    resolved images in principle and a single check would let one of four disagree
    silently. The sibling `tests/deploy/test_session_image_runs_both_halves.py:140`
    already grades this on the image; nothing graded it on the pods that produce these
    transcripts, which are what the compiler's docstrings actually rest on.
    """
    for label, transcript in transcripts.items():
        line = next(
            (
                one
                for one in transcript.splitlines()
                if one.startswith("codex-version=")
            ),
            None,
        )
        assert line is not None, (
            f"{label} reported no codex-version line, so this transcript cannot say "
            f"which runtime it measured:\n{transcript}"
        )
        assert f"codex-cli {CODEX_VERSION}" in line, (
            f"{label} ran {line!r}, not codex-cli {CODEX_VERSION}. Every claim this "
            "file certifies was measured against that version and several are quoted "
            "as settled fact in the compiler's own docstrings. Re-measure and update "
            "CODEX_VERSION together, or pin the image."
        )


@requires_the_cluster
def test_every_pod_ran_its_probe(transcripts: dict[str, str]) -> None:
    """Guard the guard. Every case below reads a transcript, and a pod that never
    started produces an empty one that satisfies both of this file's absences by
    default."""
    for label, transcript in transcripts.items():
        assert "probe=complete" in transcript, f"{label}:\n{transcript}"


@requires_the_cluster
def test_the_control_socket_is_reachable_from_outside_the_sandbox(
    transcripts: dict[str, str],
) -> None:
    """Control 1, and what makes every refusal below a refusal over a real inode.

    Asserted for ALL FOUR pods, including the ones where bubblewrap refuses to build
    anything: their finding is about the argv, and it would be a different finding if
    their socket had never been bound -- with the leaf absent the pre-slice
    configuration builds a sandbox fine, so a pod whose binder failed would report the
    refusal missing and look like this slice's own result.
    """
    for label, transcript in transcripts.items():
        assert f"bound {SOCK}" in transcript, label
        assert "P-connect-control-socket=REACHED" in transcript, label
        assert "P-connect-workspace-socket=REACHED" in transcript, label
        assert "outside=complete" in transcript, label


@requires_the_cluster
def test_a_confined_command_runs_against_the_document_the_compiler_emits(
    transcripts: dict[str, str],
) -> None:
    """The whole point of the slice, in the state a Session is actually in.

    The socket is bound and connectable from outside in this same pod, so this is not
    the earlier finding that the configuration works while its leaf is absent.
    """
    transcript = transcripts["real"]
    assert "echo-rc=0" in transcript
    assert RAN in transcript
    assert CONTROL_PATH not in transcript


@requires_the_cluster
def test_restoring_the_leaf_rule_brings_the_refusal_back(
    transcripts: dict[str, str],
) -> None:
    """Control 3. Without it, "no `Can't mkdir parents` appeared" is a claim a pod that
    failed its image pull satisfies -- and this is also the only case in the file that
    measures what the removal was FOR."""
    transcript = transcripts["leaf-restored"]
    assert CONTROL_PATH in transcript
    assert "echo-rc=0" not in transcript
    assert RAN not in transcript


@requires_the_cluster
def test_the_control_path_is_unreachable_through_every_route_tried(
    transcripts: dict[str, str],
) -> None:
    """The whole of the safety claim, and it is about the routes rather than one call.

    A listdir refused while a symlink, a dirfd or a `/proc/PID/root` reaches the same
    inode is not a denied path. Fifteen routes are checked and the count is asserted,
    because the difference between "fifteen routes refused" and "one refused and
    fourteen never ran" is invisible in a transcript that only says REFUSED.
    """
    transcript = transcripts["real"]
    lines = _labelled(transcript, ("A", "B"))
    checked = [line for line in lines if line.partition("=")[0] not in _MAY_BE_REACHED]
    assert len(checked) >= 15, f"the battery did not run: {lines}"
    for line in checked:
        assert "=REFUSED" in line, line
    assert "B10-proc-pid-sweep-reached=[]" in transcript
    assert "A7-stat-run-codex=REACHED value='d---------'" in transcript


@requires_the_cluster
def test_the_two_workspace_masks_are_where_the_previous_slice_left_them(
    transcripts: dict[str, str],
) -> None:
    """This slice removes a control-path rule and must move nothing else.

    Mode, plus a listdir, a read of bytes the outer container seeded and a write --
    because `d---------` over a directory with a readable file under it is a mask in
    name only, and the seeded marker must not appear anywhere in the transcript.
    """
    transcript = transcripts["real"]
    for dot in (".agents", ".codex"):
        assert f"C-mode{dot}=REACHED value='d---------'" in transcript
        for call in ("listdir", "read", "write"):
            assert f"C-{call}{dot}=REFUSED" in transcript
    assert "SEEDED-METADATA-BODY" not in transcript


@requires_the_cluster
def test_a_delivered_skill_is_readable_and_the_session_token_is_not(
    transcripts: dict[str, str],
) -> None:
    """The two configuration directories, measured rather than reasoned about.

    This is the assertion that MAP's skills work at all. Codex renders a Host skill's
    catalogue entry as a FILE PATH and expects the model to open it with its shell
    (`ext/skills/src/render.rs:195-215`); the `skills.list`/`skills.read` tools that
    would read it for the model return an empty list unless an orchestrator or executor
    provider is registered (`ext/skills/src/tools/mod.rs:64-70`), and this pod registers
    neither. So the confined process being able to read under `/etc/codex` is the whole
    mechanism, and while it could not, every skill was delivered, catalogued, named to
    the model, and unopenable by it -- with nothing in any log saying so.

    E3 is why E1 and E2 are not a hole, and it is asserted here rather than trusted from
    the compiler's rule list. `requirements.toml` is these permission rules plus an
    in-cluster URL: the kernel enforces the rules whether or not the confined process
    can read them, so reading it gains an agent nothing. `config.toml` carries this
    Session's bearer token and the provider base URL, and an agent that read it could
    spend the tenant's budget under its own name. One is readable and the other is
    refused, in the same sandbox, in the same run.
    """
    transcript = transcripts["real"]
    assert "E1-listdir-etc-codex=REACHED" in transcript
    assert "E2-read-managed-requirements=REACHED" in transcript
    assert "E3-read-session-token-config=REFUSED" in transcript


@requires_the_cluster
def test_the_connect_refusal_is_not_attributable_to_the_deny_rule(
    transcripts: dict[str, str],
) -> None:
    """Control 2, asserted rather than described, and the reason ADR-012's Status says
    what it says.

    `connect` to the control socket is EPERM. So is `connect` to a socket in the
    writable root, and to an abstract-namespace socket that has no inode at all --
    while `socket()` succeeds and a plain file write into that same root succeeds. The
    runtime installs a second seccomp filter that denies the socket-addressing calls
    unconditionally, and the pod's own profile is one of the two the count reports.
    This case exists so that nobody reads the EPERM above as evidence for the deny
    rule, including a later author of this file.
    """
    transcript = transcripts["real"]
    assert "A6-connect-socket=REFUSED" in transcript
    assert "D1-socket-af-unix=REACHED" in transcript
    assert "D2-connect-a-socket-in-the-WRITABLE-root=REFUSED" in transcript
    assert "D3-abstract-namespace-no-inode-at-all=REFUSED" in transcript
    assert "D4-write-a-plain-file-in-the-writable-root=REACHED" in transcript
    assert "seccomp_filters=2" in transcript


@requires_the_cluster
def test_the_seeded_bytes_the_read_refusal_is_over_really_landed(
    transcripts: dict[str, str],
) -> None:
    """The positive control for the masks below, and it is not a formality.

    `C-read.agents=REFUSED` is produced identically by a mask that denies the bytes and
    by a seed that never wrote any -- a refusal proves nothing about a file that is not
    there. This is the line that says the file is there.
    """
    for label, transcript in transcripts.items():
        assert "seeded=ok" in transcript, f"{label}:\n{transcript}"


@requires_the_cluster
def test_a_nested_deny_is_fatal_when_its_target_exists_and_not_before(
    transcripts: dict[str, str],
) -> None:
    """The variable is the target's EXISTENCE, and this is the pod that turns it.

    One pod, one document, one mkdir between the two readings. With
    `/run/codex/deeper` denied and absent the confined command runs; the moment the
    directory exists, bubblewrap refuses to build any sandbox --
    `Can't mkdir /run/codex/deeper: Read-only file system`. The denied ancestor became
    a mode-000 tmpfs remounted read-only, and the descendant's own operation is then
    attempted inside a filesystem already frozen.

    This is the general form of the refusal this slice removes. The control socket is a
    nested deny whose target the runtime creates on every start, so it was always in
    the second state; `/run/codex/deeper` is the same pair with nothing creating it,
    and it runs clean until something does. That is why the compiler's floor refuses
    the whole nested class rather than only the pair it knows about: whether a path
    will exist is not decidable where the compiler runs -- the control plane, with no
    pod -- and a nested deny that is harmless today is fatal the moment anything
    creates its target.
    """
    for label in ("nested-canonical", "nested-doubled"):
        transcript = transcripts[label]
        assert "target-before=absent" in transcript, label
        assert "absent-rc=0" in transcript, label
        assert ABSENT_RAN in transcript, label
        assert "made-target=ok" in transcript, label
        assert "target-after=directory" in transcript, label
        assert "present-rc=1" in transcript, label
        assert PRESENT_RAN not in transcript, label
        assert "Can't mkdir /run/codex/deeper: Read-only file system" in transcript, (
            f"{label}:\n{transcript}"
        )


@requires_the_cluster
def test_the_runtime_reads_a_doubled_separator_as_a_separator(
    transcripts: dict[str, str],
) -> None:
    """Why the floor compares path components instead of string prefixes.

    The second pod denies `/run//codex/deeper`, which does not start with the string
    `/run/codex/`. A nesting check that compared strings would report that deny set
    clean. The runtime does not agree: it is fatal in exactly the same state as the
    canonical spelling, and bubblewrap's own message names the NORMALISED path, which
    is the runtime saying out loud which path it thinks the rule is about.

    So the string form is not a stylistic alternative to the component form here. It
    would have passed a document that starts no Session -- the guard-grades-one-
    spelling defect, on the exact value this slice's floor is written to catch.
    """
    doubled = transcripts["nested-doubled"]
    assert "present-rc=1" in doubled
    assert "Can't mkdir /run/codex/deeper: Read-only file system" in doubled, doubled
    assert DOUBLED_TARGET not in doubled, (
        "bubblewrap echoed the doubled spelling back, so it is not normalising the "
        "path and this case is measuring something other than what it claims"
    )
