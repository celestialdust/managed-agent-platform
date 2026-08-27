"""A runtime that stops talking is closed on its own report, in minutes.

Tier 3 (live cluster). This grades the one sweep signal that reads evidence from
inside the pod: `AbandonedTurnSweeper` closing a Turn because the pod's own
`turn.progress` said its runtime went quiet for `STUCK_IDLE_MS`.

This is the wedge the pod-gone signal cannot see, and the reason `turn.progress` exists
at all: the pod is present and `Ready`, so that signal never fires, and since the
hour-long ceiling was removed on 2026-08-26 there is no clock behind it either. What
has actually happened is that the process inside the pod stopped speaking to the shim,
the only thing in the platform that can observe that is the pod's own report, and this
is now the only signal that closes such a Turn at all.

**How the wedge is produced, and why this mechanism rather than a signal.** The Session
pod reaches the model gateway on 8080 through the shared `session-pod` NetworkPolicy,
which selects on `map.role`. Policies are additive, so an allow cannot be subtracted --
but the pod can be moved out of that selector and handed a replacement policy scoped to
its own unique `map.session-id`, granting back everything except that one egress. The
pod stays `Running` and `Ready`, the control plane still reaches its shim, and the
runtime blocks on a model call that will never answer. NetworkPolicy *drops* rather
than rejects, so there is no RST to fail fast on: the call hangs. That is a model
provider stalling mid-run, which is the production failure this signal exists for.

The control plane lists pods by `map.session-id` (`pod_runner.py:100`), never by
`map.role`, so the relabel leaves the pod fully visible to the sweep and the pod-gone
signal cannot fire on it. That is what keeps this a test of the idle signal.

Everything it touches is scoped to this Session: the replacement policy selects this
session id and the relabel names this pod. No shared object is edited, so a concurrent
Session in the same namespace is untouched.

**Four in-pod mechanisms were tried first and all four are closed.** Kept here because
each cost a real run to establish and because the next person to want a wedge will
reach for them in this order:

1. *Freeze the process the shim listens to.* It is PID 1. The measured tree, taken by
   asking a live agent to read `/proc`, is `codex app-server` at PID 1 holding the
   control socket the shim dials, with one live child (the vendored codex binary) and
   four zombies. A process that is PID 1 of its own namespace does not receive a signal
   whose disposition is the default, and SIGSTOP cannot be given a handler, so the
   kernel discards it. Nothing inside the pod can stop PID 1.
2. *Freeze its child instead.* Measured: the child was stopped and the Turn completed
   fifteen seconds later. It runs sandboxed tool executions; the model conversation
   that produces frames does not pass through it, so stopping it wedges nothing.
3. *Signal PID 1 from the sibling container.* The pod does not set
   `shareProcessNamespace`, so the shim container cannot see the runtime's processes at
   all, let alone signal them.
4. *Freeze the container's own cgroup.* `cgroup.freeze` exists, but `/sys/fs/cgroup` is
   mounted `ro` and the file is owned by `nobody` and not writable. Measured live.

The network route needs none of that, and it is the more faithful failure besides: a
stalled provider is a thing that happens, and a SIGSTOPped PID 1 is not.

**And it found that the platform does not close this Turn**, which is why the case is
`xfail(strict=True)` rather than green. The wedge is real and `idle_ms` climbs, but the
runtime retries about every 170 seconds and each attempt emits one frame carrying no
answer, resetting the clock -- across fifteen minutes `answer_bytes` never left 114,
`idle_ms` reset six times and never exceeded 165 359 against a 600 000 threshold. The
measurement and what it implies are written out in `abandoned_turns.py`'s module
docstring, next to the signal it is about.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Any, Final
from uuid import UUID, uuid4

import httpx
import pytest
from cluster_access import (
    NAMESPACE,
    forwarded,
    kubectl,
    open_a_bare_session,
    submit_a_turn,
)

from managed_agent.control.session.abandoned_turns import STUCK_IDLE_MS
from managed_agent.core.ids import SessionId

requires_the_cluster = pytest.mark.skipif(
    os.environ.get("MAP_CLUSTER_TESTS") != "1",
    reason="set MAP_CLUSTER_TESTS=1 to run against the live cluster",
)

_CONTROL_PLANE: Final = "deploy/control-plane"
_CONTROL_PLANE_PORT: Final = 8080
_REGION: Final = "us-east-1"
_REPOSITORY: Final = "map/session-shim"

_PATIENCE_S: Final = STUCK_IDLE_MS / 1000 + 6 * 60
"""How long to wait for the sweep: the threshold plus six minutes.

The margin covers placement, the run's own head start before the wedge, and the sweep's
thirty-second cadence -- and is generous because the alternative to waiting is a case
that fails on a slow afternoon and gets muted.

There is no longer a ceiling behind this to catch a wedge late, so this signal is the
only thing that closes a Turn whose pod is alive. A run that does not close inside this
window does not close at all.
"""

_POLL_EVERY_S: Final = 20

_HEAD_START_S: Final = 40
"""How long the run is left alone before the model path is cut.

Long enough that a model call is genuinely in flight and `frames` is climbing, so the
freeze is visible as a *change*: the first reports after the cut show `frames` frozen at
whatever it had reached while `idle_ms` climbs away from it. Cutting at zero would leave
no healthy reading to contrast against, and a wedge with nothing to compare it to is
indistinguishable from a pod that was never working.
"""

_LONG_RUN: Final = (
    "Work through these ten steps one at a time. After each step, tell me in one "
    "sentence what it printed, then start the next step. Do not batch them and do "
    "not skip ahead.\n"
    + "\n".join(f"{n}. Run `sleep 12 && echo step-{n}-done`." for n in range(1, 11))
)
"""A Turn that keeps the runtime talking to the model for minutes, not seconds.

Measured first with a single hard question, which came back complete in about twenty
seconds -- the Turn was over before the wedge could be applied, which measured the
prompt rather than the platform. What this needs is *repeated* model calls with real
gaps between them, because that is the shape the wedge has to catch: a provider that
stalls partway through an agent's run, not one that never answers at all. Ten sandboxed
sleeps give roughly two minutes of activity with a model call between each pair.
"""


def _replacement_policy(session_id: SessionId) -> str:
    """Everything the shared `session-pod` policy grants, minus the model gateway.

    Mirrored field for field from the live policy rather than trimmed to what seemed
    necessary. The first draft dropped the 443 egress too, which cuts a Session off from
    the internet as well and would have made a failure ambiguous between two causes. The
    only difference from the shared policy is the absent `map.component: model-gateway`
    selector in the 8080 egress rule.
    """
    return f"""
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: wedge-{str(session_id)[:8]}
  namespace: {NAMESPACE}
spec:
  podSelector:
    matchLabels:
      map.session-id: {session_id}
  policyTypes: [Ingress, Egress]
  ingress:
  - from:
    - podSelector:
        matchLabels:
          map.component: control-plane
    ports:
    - port: 8081
      protocol: TCP
  egress:
  - ports:
    - port: 53
      protocol: UDP
    - port: 53
      protocol: TCP
    to:
    - namespaceSelector:
        matchLabels:
          kubernetes.io/metadata.name: kube-system
      podSelector:
        matchLabels:
          k8s-app: kube-dns
  - ports:
    - port: 8080
      protocol: TCP
    to:
    - podSelector:
        matchLabels:
          map.component: tool-gateway
  - ports:
    - port: 443
      protocol: TCP
    to:
    - ipBlock:
        cidr: 0.0.0.0/0
        except:
        - 172.31.0.0/16
        - 10.100.0.0/16
        - 169.254.0.0/16
"""


def _cut_the_model_path(session_id: SessionId) -> None:
    """Move this one pod off the shared policy and onto one without the model gateway.

    The replacement is applied *before* the relabel and not after. Policies are
    additive, so while the pod still matches `session-pod` the new one grants nothing it
    did not already have and changes nothing. Applying it after the relabel would leave
    a window in which the pod matched no policy at all, and `default-deny` would cut its
    ingress from the control plane -- which is the pod-gone signal's failure, not this
    one's.
    """
    subprocess.run(
        ("kubectl", "apply", "-n", NAMESPACE, "-f", "-"),
        input=_replacement_policy(session_id),
        text=True,
        check=True,
        capture_output=True,
    )
    kubectl("label", "pod", _pod_name(session_id), "map.role=wedged", "--overwrite")


def _pod_name(session_id: SessionId) -> str:
    return f"map-session-{session_id}"


def _session_image() -> str:
    """The newest digest in the Session repository. Resolved, not pinned.

    This case genuinely depends on the image: the emitter that produces `turn.progress`
    runs in the shim, and a digest built before it reports nothing at all, which this
    sweep signal correctly ignores. So a stale image here does not fail loudly -- it
    produces a Turn that never closes and a case that times out saying the sweep never
    fired, pointing at the sweep rather than at the image.
    """
    done = subprocess.run(
        (
            "aws",
            "ecr",
            "describe-images",
            "--repository-name",
            _REPOSITORY,
            "--region",
            _REGION,
            "--output",
            "json",
        ),
        capture_output=True,
        text=True,
        check=True,
    )
    details = json.loads(done.stdout)["imageDetails"]
    assert details, f"{_REPOSITORY} holds no images, so no Session pod can start"
    newest = max(details, key=lambda one: str(one["imagePushedAt"]))
    return (
        f"{newest['registryId']}.dkr.ecr.{_REGION}.amazonaws.com/"
        f"{_REPOSITORY}@{newest['imageDigest']}"
    )


def _client(base: str, tenant_id: str, timeout: int = 90) -> httpx.Client:
    return httpx.Client(
        base_url=base, headers={"X-Tenant-Id": tenant_id}, timeout=timeout
    )


def _a_session(base: str, tenant_id: str, image: str, run: str) -> SessionId:
    """One bare Session on the newest Session image, through the REST API only.

    No files, no tools, no skills. The subject is what the sweep does with a wedged
    runtime, and every attachment a Session could carry is one more thing a reader would
    have to rule out before believing the result.

    The three posts live in `cluster_access` rather than here because their bodies are
    one half of a contract whose other half is a set of Pydantic models in `src/`, and
    nothing relates the two halves except a case that parses one with the other. Written
    inline, this drifted and was answered `400` several minutes into a live run.
    """
    with _client(base, tenant_id) as caller:
        created = open_a_bare_session(
            caller,
            f"wedge-{run}",
            image,
            "You run the shell commands you are given, exactly as given, one at a "
            "time, and you report what each one printed.",
        )
    return SessionId(UUID(created))


def _events(base: str, tenant_id: str, session_id: SessionId) -> list[dict[str, Any]]:
    with _client(base, tenant_id) as caller:
        answered = caller.get(f"/v1/sessions/{session_id}/events")
    assert answered.status_code == 200, answered.text
    listed: list[dict[str, Any]] = answered.json()["events"]
    return listed


def _clean_up(session_id: SessionId) -> None:
    """Delete this case's pod and its policy, whatever happened to the case.

    Both, and in that order: a policy left behind selects on a session id that no longer
    exists, which is harmless but accumulates, and a pod left behind holds a node slot
    and costs money. This has to run even when the case failed, because a case that
    times out is exactly the one that left a pod behind.

    Neither delete is conditional on the object existing. `--ignore-not-found` makes
    both idempotent, and a cleanup that first checks has a window in which the check and
    the delete disagree.
    """
    for argv in (
        ("delete", "pod", _pod_name(session_id), "--wait=false"),
        ("delete", "networkpolicy", f"wedge-{str(session_id)[:8]}"),
    ):
        subprocess.run(
            ["kubectl", "-n", NAMESPACE, *argv, "--ignore-not-found"],
            capture_output=True,
            text=True,
            timeout=120,
        )


_THE_RETRY_LOOP: Final = (
    "measured 2026-08-26 over fifteen minutes: a runtime cut off from the model "
    "gateway retries about every 170s, and each attempt emits one frame carrying no "
    "answer, which resets idle_ms. It reset six times, never exceeded 165359 against "
    "a 600000 threshold, and the Turn was never closed. The mechanism below is "
    "correct and the platform does not yet close this Turn; see the measurement in "
    "abandoned_turns.py's module docstring."
)


@pytest.mark.xfail(strict=True, reason=_THE_RETRY_LOOP)
@requires_the_cluster
def test_a_wedged_runtime_is_closed_in_minutes_on_its_own_report() -> None:
    """Cut a live pod off the model gateway and watch the deployed sweep close its Turn.

    One case rather than a fixture and several: everything worth asserting here is about
    a single timeline, and splitting it would mean either wedging a second pod or
    sharing a fixture whose failure mode is a sixteen-minute timeout in whichever case
    happened to run first.

    The assertions are ordered so the first one to fail is the most informative. A Turn
    that never closed says the sweep did not act; a Turn that took far longer than the
    threshold says something other than this signal closed it; a Turn whose `idle_ms`
    never climbed says the cut did not take, which is a defect in this case rather than
    in the platform.

    **`xfail(strict=True)` rather than skipped**, which is the opposite of what this
    file said until today. The mechanism works -- the wedge is real, the pod stays
    `Ready`, `idle_ms` climbs -- and the platform does not close the Turn, for the
    measured reason on the marker. A skip would say "we have no way to test this",
    which was true yesterday and is not true now; a deleted case would lose the
    mechanism it took five attempts to find. Strict is what makes it useful: when the
    gap is closed this case starts passing and fails the run for passing, which is the
    only reliable way anyone learns the behaviour changed.

    It costs a real pod for up to sixteen minutes per run, and it is in the tier that
    only runs when asked, so that cost is paid deliberately or not at all.
    """
    run = uuid4().hex[:8]
    tenant_id = str(uuid4())
    image = _session_image()
    session_id: SessionId | None = None
    try:
        with forwarded(_CONTROL_PLANE, _CONTROL_PLANE_PORT) as base:
            session_id = _a_session(base, tenant_id, image, run)
            with _client(base, tenant_id) as caller:
                submit_a_turn(caller, session_id, _LONG_RUN)

            subprocess.run(
                (
                    "kubectl",
                    "-n",
                    NAMESPACE,
                    "wait",
                    "--for=condition=Ready",
                    f"pod/{_pod_name(session_id)}",
                    f"--timeout={int(_PATIENCE_S)}s",
                ),
                check=True,
                capture_output=True,
            )
            time.sleep(_HEAD_START_S)
            _cut_the_model_path(session_id)
            began = time.monotonic()

            deadline = began + _PATIENCE_S
            events: list[dict[str, Any]] = []
            while time.monotonic() < deadline:
                time.sleep(_POLL_EVERY_S)
                events = _events(base, tenant_id, session_id)
                if any(
                    one["type"] in ("turn.completed", "turn.failed") for one in events
                ):
                    break
            closed_after = time.monotonic() - began

        types = [one["type"] for one in events]
        reports = [one for one in events if one["type"] == "turn.progress"]
        idles = [int(one.get("payload", {}).get("idle_ms", -1)) for one in reports]

        assert "turn.failed" in types or "turn.completed" in types, (
            f"no terminal event {closed_after:.0f}s after the cut. The Turn is still "
            f"open, so the sweep did not close it. Reports seen: {len(reports)}, "
            f"largest idle_ms {max(idles, default=-1)}. If that largest value is small "
            "the cut did not take and this case is at fault; if it is above "
            f"{STUCK_IDLE_MS} the sweep saw a stuck Turn and left it open."
        )

        assert closed_after * 1000 < STUCK_IDLE_MS * 2, (
            f"the Turn closed {closed_after:.0f}s after the cut, more than twice the "
            f"{STUCK_IDLE_MS // 60000}-minute threshold -- so whatever closed it was "
            "not this signal acting promptly on the report it reads"
        )

        assert max(idles, default=-1) >= STUCK_IDLE_MS, (
            f"the largest idle_ms reported was {max(idles, default=-1)}, below the "
            f"{STUCK_IDLE_MS} threshold, so whatever closed this Turn it was not this "
            "signal. Most likely the runtime found another way to make progress after "
            "the cut, which is a defect in this case rather than in the platform."
        )

        assert "turn.failed" in types, (
            f"the Turn ended as {types[-1]!r} rather than failing. A wedged runtime "
            "that reports completion means the model path was restored before the "
            "sweep acted."
        )
    finally:
        if session_id is not None:
            _clean_up(session_id)
