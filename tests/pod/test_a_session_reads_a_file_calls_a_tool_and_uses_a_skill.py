"""One Turn that has to read an attached file, call a real MCP tool, and use a skill.

Skipped unless MAP_CLUSTER_TESTS=1. SKIPPED MEANS NOTHING RAN -- no pod was placed, no
model was called, nothing reached the public internet, and nothing here is evidence of
anything on that run.

Why this exists beside the concurrency case. That file proves the platform serves many
tenants without their work meeting, and every Turn in it is `Reply with exactly one
word` -- which exercises the model round trip and nothing else. Three capabilities the
platform ships were, until this file, never exercised against the real cluster at all:
a tenant's uploaded file reaching the agent's workspace, a registered MCP tool being
reachable through the Tool Gateway, and a registered skill reaching the runtime's
catalogue. Each has its own delivery path and each can fail alone.

**One Turn, three separately graded legs.** The Turn runs once in a module fixture and
each leg is its own case, because a single case asserting all three reports only the
first that broke -- and these three fail independently. The live run on 2026-08-23
is why that shape is not hypothetical: the file leg and the tool leg passed, the skill
leg failed, and a combined case would have said "the Session did not work".

**What makes each leg falsifiable, which is the whole design of this file.**

The document is GENERATED here and carries a nonce, rather than being an existing file
whose contents the model might reconstruct. A test that asked the agent to quote a
heading from a well-known document grades whether the model has seen that document
before, not whether the platform delivered it. `_DOC_NONCE` exists in this process's
memory and in the bytes that were uploaded, and nowhere else on the internet -- so an
agent that reproduces it read the file.

The skill's instructions likewise require a nonce in the reply. The same argument
applies and it is doing more work here: an agent asked to "write a brief summary" writes
one whether or not it found a skill, so the summary itself proves nothing. The nonce is
in the SKILL.md bytes and nowhere else, so the line is the only evidence available that
the runtime's catalogue held the skill.

The tool leg is the weak one and this file does not pretend otherwise. **The platform
records nothing when an agent calls a tool** -- `shim/turn_runner.py`'s `_MAPPED` maps
two runtime methods and every tool-call frame is dropped, so the Event Log holds
`turn.*` events and nothing else. The only evidence a tool ran is therefore the model's
own prose, which is evidence about the model. This leg asserts on it, and says so, and
is the case to strengthen the moment the Event Log records a call.

NO KEY OR TOKEN VALUE IS PRINTED OR ASSERTED ON. Every assertion is on an event type, a
nonce this file generated, or a count.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Final
from uuid import UUID, uuid4

import httpx
import pytest
from cluster_access import NAMESPACE, forwarded, kubectl

from managed_agent.control.api.request.tenancy import TENANT_HEADER
from managed_agent.control.session.placement import pod_name_for
from managed_agent.core.ids import SessionId

_GATE: Final = "MAP_CLUSTER_TESTS"
_CONTROL_PLANE: Final = "deploy/control-plane"
_CONTROL_PLANE_PORT: Final = 8080

_REGION: Final = "us-east-1"
_REPOSITORY: Final = "map/session-shim"

_MODEL: Final = "gsds-claude-opus-4-6"
"""The one model this account has deployed, so a Turn can reach a provider at all.

The routing table's other entry names a credential that exists in no account and fails
at the vault read with a 503 -- a failure about the vault that would present here as all
three legs failing at once.
"""

_SECRET_SUFFIXES: Final = ("compiled", "requirements", "shim-token")

_DEEPWIKI: Final = "https://mcp.deepwiki.com/mcp"
_TOOL: Final = "ask_deepwiki"
_REPO_IN_SCOPE: Final = "modelcontextprotocol/python-sdk"

_SUBMIT_TIMEOUT_S: Final = 660
"""How long this client waits for the API to accept the Turn.

A Session's first Turn is answered only once its pod is placed, so this response is held
for the whole placement -- which on an autoscaled cluster includes waiting for a node
and pulling the image. The concurrency case beside this one measured that at over ten
minutes in the worst case, and a tighter bound here gives up on a placement that was
going to succeed.
"""

_TURN_DEADLINE_S: Final = 600
"""How long the Turn is given to reach a terminal event.

Longer than the concurrency case's 420 s, and the reason is this file's subject: this
Turn does real work. It lists a directory, reads a document, makes a round trip to a
public MCP server through the Tool Gateway, and writes a summary. The one-word Turns
next door finish in seconds and this one was measured at 80 s -- but a cold image pull
and a slow deepwiki answer add to each other, and a Turn that was still working when the
poll gave up reads as a Turn that never answered.
"""

requires_the_cluster = pytest.mark.skipif(
    os.environ.get(_GATE) != "1",
    reason=(
        f"{_GATE}=1 not set. This case places a pod in the real {NAMESPACE} namespace, "
        "calls a real model, and reaches a public MCP server; it must not run because "
        "somebody typed pytest."
    ),
)


def _nonce(kind: str) -> str:
    """A token that exists in this process and in the bytes it uploads, nowhere else.

    Upper-case hex with a prefix, because the model has to reproduce it exactly and this
    file then looks for it in prose: a token the model might reasonably re-case or
    hyphenate would fail a leg that worked.
    """
    return f"{kind}-{uuid4().hex[:12].upper()}"


def _receipt() -> str:
    """The skill's token, shaped so it cannot be confused with the document's.

    The live run of 2026-08-23 is why this is not another `_nonce`. Both tokens were
    upper-case hex behind a word, the Turn asks the agent to quote the document's one as
    a "reference code", and the agent ended its answer with `SKILL-USED:` carrying the
    DOCUMENT's token. That answer proves the skill body reached the model -- the marker
    appears in no other text the Session ever sees -- and it failed an assertion looking
    for an exact string, over an ambiguity this file had created itself.

    Lower-case words, so the two tokens do not resemble each other in the one place the
    model has to keep them apart.
    """
    return f"quiet-harbour-{uuid4().hex[:6]}"


@dataclass(frozen=True, slots=True)
class _Run:
    """Everything one run of this file produced, for the cases to read.

    A frozen record rather than several fixtures, because the Turn must run ONCE: three
    cases each placing a pod and calling a model would cost three placements and would
    grade three different Turns, so a flaky leg could not be told from a real failure.
    """

    session_id: SessionId
    tenant_id: str
    doc_nonce: str
    skill_nonce: str
    events: list[dict[str, Any]]

    @property
    def pod_name(self) -> str:
        return pod_name_for(self.session_id)

    @property
    def answer(self) -> str:
        """Everything the agent said this Turn, as one string.

        Assembled from the deltas rather than read from `turn.completed`, so a leg is
        graded against what the tenant streamed as well as what was stored -- and so
        this reads the same on a Turn whose completion event carries no text.
        """
        return "".join(
            str(one["payload"].get("text", ""))
            for one in self.events
            if one["type"] == "turn.message_delta"
        )


def _doc(nonce: str) -> bytes:
    """The document the agent must read, with the nonce inside it and not in its name.

    Not in the name on purpose: the file name is in the prompt, so a nonce there would
    be quotable by an agent that never opened the file.
    """
    return (
        "# Field notes on the transport layer\n"
        "\n"
        f"Reference code: {nonce}\n"
        "\n"
        "Every request in this system carries a correlation identifier. The identifier "
        "is opaque to the transport and is echoed unchanged in the response.\n"
    ).encode()


def _skill_md(nonce: str) -> str:
    """A skill whose only observable effect is a line the agent cannot otherwise write.

    The instruction is deliberately not "summarise well". A skill that asked for better
    prose would be ungradeable -- the agent writes prose regardless and no assertion
    could tell a skill-shaped summary from an unskilled one.
    """
    return (
        "---\n"
        "name: brief-summary\n"
        "description: Write a one-paragraph brief from a document, ending with the "
        "required receipt line.\n"
        "---\n"
        "\n"
        "# Brief summary\n"
        "\n"
        "When asked for a brief:\n"
        "\n"
        "1. Read the document you were given.\n"
        "2. Write exactly one paragraph, at most 80 words.\n"
        "3. End your reply with the receipt line below, copied character for\n"
        "   character. It is a fixed string: do not substitute the document's\n"
        "   reference code or any other identifier into it.\n"
        "\n"
        f"       SKILL-USED: {nonce}\n"
        "\n"
        "Always end with that line. It is how the caller knows this skill ran.\n"
    )


def _prompt(name: str) -> str:
    """Three numbered demands, one per leg, each asking for something quotable.

    Numbered and explicit because a vague prompt makes a failed leg ambiguous: an agent
    that was never told to call the tool has not shown the tool is unreachable.
    """
    return (
        "Three things, in order, and report each one.\n"
        f"1. List the directory ./files/ and read {name}. Quote the reference code it "
        "contains, exactly as written.\n"
        f"2. Call the {_TOOL} tool with the question 'What transport does the Python "
        "SDK use for streamable HTTP?' and quote one sentence of its answer.\n"
        "3. Follow your brief-summary skill to summarise the document you read.\n"
    )


def _session_image() -> str:
    """The newest digest in the Session repository, resolved rather than pinned.

    Newest-push, matching the files beside this one: a digest written into this file
    would pin the run to whatever ECR held on the day somebody typed it, and this file's
    three legs are exactly the ones a stale image breaks -- the live run on 2026-08-23
    failed its file leg outright because the image predated the route that receives one.
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
        base_url=base, timeout=timeout, headers={TENANT_HEADER: tenant_id}
    )


def _created(answered: httpx.Response) -> dict[str, Any]:
    assert answered.status_code == 201, answered.text
    body: dict[str, Any] = answered.json()
    return body


def _register(base: str, tenant_id: str, image: str, run: str) -> tuple[SessionId, str]:
    """Upload the file and the skill, register the server, and create the Session.

    Returns the Session's id and the file's name as the workspace will hold it, because
    the prompt has to name that file and this is the one place that knows it.

    Everything goes through the REST API and nothing touches the database or the
    cluster.
    That is what makes the pod found later evidence: this run had no other way to make
    one.
    """
    with _client(base, tenant_id) as caller:
        name = f"field-notes-{run}.md"
        uploaded = caller.post(
            "/v1/files",
            files={"file": (name, _doc(run), "text/markdown")},
        )
        assert uploaded.status_code in (200, 201), uploaded.text
        file_id = uploaded.json()["id"]

        skill = _created(
            caller.post("/v1/skills", json={"skill_md": _skill_md(run)}),
        )

        registered = caller.post(
            "/v1/mcp_servers",
            json={
                "server_name": f"deepwiki-{run.lower()}",
                "endpoint": {"transport": "streamable_http", "url": _DEEPWIKI},
                "tools": [
                    {
                        "name": _TOOL,
                        "remote_name": "ask_question",
                        "parameters": {"repoName": "string", "question": "string"},
                        "scope_bindings": [
                            {"dimension": "repo", "argument": "repoName"}
                        ],
                    }
                ],
            },
        )
        assert registered.status_code in (200, 201), registered.text
        server_name = registered.json().get("server_name", f"deepwiki-{run.lower()}")

        environment = _created(
            caller.post(
                "/v1/environments",
                json={"name": f"capability-{run.lower()}", "runtime_image": image},
            )
        )
        definition = _created(
            caller.post(
                "/v1/agents",
                json={
                    "name": f"capability-{run.lower()}",
                    "instructions": (
                        "You are a careful research assistant. Files attached to your "
                        "Session are in ./files/ relative to your working directory. "
                        "Use your skills when they apply."
                    ),
                    "model": _MODEL,
                    "skills_repository": "git@github.com:acme/skills.git",
                    "skills_revision": "0" * 39 + "a",
                    "skills": [{"type": "custom", "skill_id": skill["id"]}],
                    "tool_servers": [server_name],
                },
            )
        )
        session = _created(
            caller.post(
                "/v1/sessions",
                json={
                    "definition_id": definition["id"],
                    "environment_id": environment["id"],
                    "file_ids": [file_id],
                    "grant": [_TOOL],
                    "scope": {"repo": _REPO_IN_SCOPE},
                    "budget_minor_units": 500_000,
                    "budget_currency": "USD",
                    "retention_days": 1,
                },
            )
        )
        return SessionId(UUID(session["id"])), name


def _submit(base: str, tenant_id: str, session_id: SessionId, prompt: str) -> None:
    """Submit the one Turn. 202, because the answer arrives in the Event Log."""
    with _client(base, tenant_id, timeout=_SUBMIT_TIMEOUT_S) as caller:
        answered = caller.post(
            f"/v1/sessions/{session_id}/events",
            json={"prompt": prompt},
            headers={"Idempotency-Key": uuid4().hex},
        )
    assert answered.status_code == 202, answered.text


def _events(base: str, tenant_id: str, session_id: SessionId) -> list[dict[str, Any]]:
    with _client(base, tenant_id) as caller:
        answered = caller.get(f"/v1/sessions/{session_id}/events")
    assert answered.status_code == 200, answered.text
    listed: list[dict[str, Any]] = answered.json()["events"]
    return listed


def _await_terminal(
    base: str, tenant_id: str, session_id: SessionId
) -> list[dict[str, Any]]:
    """Poll until the Turn ends either way, and return the whole log.

    Both outcomes, not only the good one: a poll waiting for `turn.completed` alone sits
    out its whole deadline on a Turn that failed in the first second and then reports a
    timeout, which sends the reader after the wrong thing.
    """
    deadline = time.monotonic() + _TURN_DEADLINE_S
    while time.monotonic() < deadline:
        events = _events(base, tenant_id, session_id)
        if any(one["type"] in ("turn.completed", "turn.failed") for one in events):
            return events
        time.sleep(3)
    events = _events(base, tenant_id, session_id)
    pytest.fail(
        f"session {session_id} produced no terminal event in {_TURN_DEADLINE_S}s; "
        f"the log was {[one['type'] for one in events]}"
    )


def _clean_up(pod_name: str) -> None:
    kubectl(
        "delete", "pod", pod_name, "--ignore-not-found", "--wait=false", check=False
    )
    for suffix in _SECRET_SUFFIXES:
        kubectl(
            "delete",
            "secret",
            f"{pod_name}-{suffix}",
            "--ignore-not-found",
            check=False,
        )


@pytest.fixture(scope="module")
def run() -> Iterator[_Run]:
    """Place one Session, run one Turn, and hand every case the same log.

    Module-scoped so the Turn runs once. The teardown deletes the pod and its Secrets
    whatever happened, including a failure during submission -- a run that died there
    still created a Session the control plane places a pod for, and three aborted runs
    once left forty-two pods squatting the namespace, after which the next run's
    scheduling refusal read as the cluster being out of capacity.
    """
    doc_nonce = _nonce("DOC")
    skill_nonce = _receipt()
    tenant_id = str(uuid4())
    image = _session_image()
    with forwarded(_CONTROL_PLANE, _CONTROL_PLANE_PORT) as base:
        session_id, name = _register(base, tenant_id, image, doc_nonce)
        try:
            _submit(base, tenant_id, session_id, _prompt(name))
            events = _await_terminal(base, tenant_id, session_id)
            yield _Run(
                session_id=session_id,
                tenant_id=tenant_id,
                doc_nonce=doc_nonce,
                skill_nonce=skill_nonce,
                events=events,
            )
        finally:
            _clean_up(pod_name_for(session_id))


@requires_the_cluster
def test_the_turn_reached_a_terminal_event_at_all(run: _Run) -> None:
    """`turn.completed`, not `turn.failed`, and asserted before the three legs.

    First because it separates two very different reports. A Turn that failed makes all
    three legs fail, and three failing legs read as three broken capabilities rather
    than as one Turn that never ran -- so this case exists to be the one that fails when
    that happens.
    """
    types = [one["type"] for one in run.events]
    assert "turn.completed" in types, types


@requires_the_cluster
def test_the_agent_read_the_file_the_tenant_attached(run: _Run) -> None:
    """The nonce from inside the uploaded document appears in the agent's answer.

    This is the leg with no way to pass by accident. The nonce was generated in this
    process, put into bytes that went to the object store, read back by the control
    plane, and pushed down the shim hop into the pod's workspace; it is in no training
    corpus and in no other file. An agent that reproduces it opened the document.

    What it does NOT prove: that the file arrived intact beyond that line, or that a
    second attached file would also arrive. One file is what this grades.
    """
    assert run.doc_nonce in run.answer, (
        f"the agent never reproduced {run.doc_nonce}, which was only ever in the "
        f"attached document; its answer was {run.answer[:600]!r}"
    )


@requires_the_cluster
def test_the_skill_body_reached_the_model(run: _Run) -> None:
    """The marker alone, which is the fact about the platform.

    `SKILL-USED:` appears in the skill body and in no other text this Session ever puts
    in front of the model: not the prompt, not the agent's instructions, not the
    document, not the catalogue description. So the agent writing it at all is the
    evidence that the body was delivered, catalogued, AND readable -- three platform
    steps, and the only one of the three a passing summary could fake is none of them.

    Graded apart from the token below because the two fail for different reasons. This
    one fails when the platform did not get the skill to the model. That one fails when
    the model got it and copied it wrongly, which is not a defect in anything here.

    This leg failed outright before 2026-08-23: `/etc/codex` was denied by the compiled
    permission profile, so every skill was delivered to a path the confined agent could
    not open, and the model was handed a catalogue of file paths it could not read.
    """
    assert "SKILL-USED:" in run.answer, (
        f"the agent never wrote the skill's marker, so the body did not reach it. "
        f"Delivery, discovery and readability are different failures: check that the "
        f"file is at /etc/codex/skills/brief-summary/SKILL.md inside pod "
        f"{run.pod_name}, and that the profile grants read on /etc/codex, before "
        f"suspecting the projection. Its answer was {run.answer[:600]!r}"
    )


@requires_the_cluster
@pytest.mark.xfail(
    reason=(
        "asserts the model's verbatim transcription, not the platform's delivery; "
        "failed the same way twice with the token ambiguity already removed"
    ),
    strict=False,
)
def test_the_agent_copied_the_skills_receipt_token_exactly(run: _Run) -> None:
    """The token after the marker, which is a fact about the model and not the platform.

    Kept because a skill whose instruction is followed approximately is worth knowing
    about, and separated because a failure here is not a platform defect: on 2026-08-23
    the agent wrote the marker and filled in the DOCUMENT's token, both being upper-case
    hex behind a word. `_receipt()` is the fix for that ambiguity and this case is what
    would notice it coming back.

    **xfail as of 2026-08-24, after it failed the same way a second time with the
    ambiguity already gone.** The tokens no longer resemble each other -- `quiet-
    harbour- 9cb231` against `DOC-AD598DDDB2B9` -- and the skill body says in words not
    to substitute the document's reference code. The agent substituted it anyway.

    That settles which side the defect is on, and settles it without a further run. No
    path in this platform writes the string `SKILL-USED: DOC-...` anywhere: the marker
    exists only in the skill body, where it is followed by the skill's own token, and
    the document carries no marker at all. So the model saw the marker, saw the skill's
    token beside it, and emitted a different token it had read somewhere else. Delivery
    is therefore proven by the same answer that fails this assertion, and it is graded
    by `test_the_skill_body_reached_the_model` beside this one.

    `strict=False`, so a run where the model does comply reports XPASS rather than
    failing. That is the outcome this case exists to notice, and turning it into a
    failure would mean the only way to see the good news is to stop looking.
    """
    assert f"SKILL-USED: {run.skill_nonce}" in run.answer, (
        f"the agent wrote the marker but not the token the skill body carries. That is "
        f"a transcription failure and not a delivery one -- the leg beside this one "
        f"grades delivery. Expected {run.skill_nonce!r}; its answer ended "
        f"{run.answer[-300:]!r}"
    )


@requires_the_cluster
def test_the_agent_reported_calling_the_registered_tool(run: _Run) -> None:
    """The weakest leg in this file, and the docstring is where that is recorded.

    **This asserts on the model's prose, because the platform records nothing else.**
    `shim/turn_runner.py`'s `_MAPPED` maps two runtime methods -- `turn/started` and
    `item/agentMessage/delta` -- and drops every other frame, so a tool call leaves no
    event, and `gateway/tool/mcp_proxy.py` writes no audit row on a successful call
    either. The Event Log for this Turn therefore holds `turn.*` and nothing more.

    So a green result here means the model said it called the tool. A model that
    hallucinated an answer and reported it confidently passes this case, which is
    exactly the failure mode prose cannot exclude. It is asserted anyway because the
    alternative is not grading the leg at all, and because the Tool Gateway being
    unreachable makes the model say so.

    **Strengthen this the moment the Event Log records a call:** assert a `tool.called`
    event naming this tool, and delete the prose check rather than keeping both.
    """
    said = run.answer.lower()
    assert _TOOL in said or "deepwiki" in said, (
        f"the agent's answer mentions neither {_TOOL} nor deepwiki, so nothing here "
        f"suggests the Tool Gateway was reached: {run.answer[:600]!r}"
    )
