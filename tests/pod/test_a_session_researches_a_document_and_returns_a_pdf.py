"""The whole journey a tenant asked for, on the real cluster, in one Turn.

Skipped unless MAP_CLUSTER_TESTS=1. SKIPPED MEANS NOTHING RAN.

Five legs, and the point of the file is that they are five legs of ONE errand rather
than five features:

    1.  a document is uploaded to a Session and the agent reads it
    2.  the agent researches a feature it mentions, through a real MCP server
    3.  the agent renders a brief to PDF, following Anthropic's published `pdf` skill
    4.  the tenant downloads that PDF back out
    5.  the Session is archived and closed, and its history survives both

Each leg has a file of its own next door that grades it in isolation, and those files
are the better place to debug from. This one exists because passing them separately does
not mean the errand works: the PDF has to be about the document the tenant uploaded and
the research the agent did, the file id has to reach a tenant who was not watching the
pod, and the Session has to close without taking the evidence with it. Composition is
the subject.

**Nothing here is a toy.** The document is OpenAI's own Codex README, vendored verbatim
at `tests/fixtures/codex_blog.md`. The skill is Anthropic's published `pdf` skill,
vendored verbatim at `tests/fixtures/anthropic_pdf_skill.py`, registered byte for byte
-- not a SKILL.md written to suit this test. The MCP server is deepwiki on the public
internet, asked about the real `openai/codex` repository. A fixture we wrote ourselves
would only ever exercise the shapes we already thought of.

**The two nonces are what make this evidence and not a vibe.** `DOC-...` exists in this
process and inside the uploaded bytes, nowhere else, so quoting it proves the agent
opened
a file rather than recognised a README it may have seen in training. `pdf-receipt-...`
exists only inside the registered skill body, so its appearance proves the skill text
reached the model -- and it is worded unlike the document's code on purpose, because a
live run on 2026-08-23 had the agent copy the document's token into the skill's receipt
line, failing an assertion over an ambiguity a test had created itself.

**The strongest case here is `test_the_pdf_that_came_back_is_about_this_errand`.**
It downloads the bytes through the public API, confirms they are a real PDF by parsing
them, and looks for the document's reference code in the extracted text. That one string
can only be there if the upload, the mount, the read, the render, the ship-out to the
object store, the `output.produced` announcement and the download route all worked -- so
it is the leg that cannot be faked by any single piece behaving well.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final
from uuid import UUID, uuid4

import httpx
import pytest
from cluster_access import NAMESPACE, forwarded, kubectl
from fixtures.anthropic_pdf_skill import SKILL_MD
from fixtures.codex_blog import codex_blog

from managed_agent.control.api.request.tenancy import TENANT_HEADER
from managed_agent.control.session.placement import pod_name_for
from managed_agent.core.ids import SessionId
from managed_agent.core.registration.advertised_name import advertised_name_for
from managed_agent.core.vocabulary import tool_call, tool_server

_GATE: Final = "MAP_CLUSTER_TESTS"
_CONTROL_PLANE: Final = "deploy/control-plane"
_CONTROL_PLANE_PORT: Final = 8080

_REGION: Final = "us-east-1"
_REPOSITORY: Final = "map/session-shim"

_MODEL: Final = "gsds-claude-opus-4-6"
_SECRET_SUFFIXES: Final = ("compiled", "requirements", "shim-token")

_DEEPWIKI: Final = "https://mcp.deepwiki.com/mcp"
_TOOL: Final = "ask_deepwiki"
_REPO_IN_SCOPE: Final = "openai/codex"
"""The repository the agent researches, and the only one its Grant lets it ask about.

`openai/codex` rather than the SDK repository the capability file next door uses,
because
the document the tenant uploaded is Codex's own README: a Turn told to research "a
feature
this document mentions" and scoped to an unrelated repository would fail leg 2 for a
reason that is this file's own doing.
"""

_FEATURE: Final = "sandboxing"
"""The feature the agent must research, named in the prompt rather than left to it.

Named because leg 2 has to be gradeable. An agent free to pick would sometimes pick
something the document covers well enough that no tool call is needed, and a Turn that
answered without calling the tool cannot be told from a Turn whose tool was unreachable.
"""

_OUTPUT_NAME: Final = "codex-brief.pdf"
"""What the agent must call the PDF, at the root of its working directory.

Named in the prompt because ship-out takes "a regular file directly in the workspace
root" (`shim/serve.py`) and descends no directory: a PDF the agent tidied into `out/`
is a PDF nothing ships, and the Turn would look like a render failure.
"""

_SUBMIT_TIMEOUT_S: Final = 660
"""How long this client waits for the API to accept the Turn.

A Session's first Turn is answered only once its pod is placed, so this response is held
for the whole placement -- on an autoscaled cluster that includes waiting for a node and
pulling the image, measured at over ten minutes in the worst case.
"""

_TURN_DEADLINE_S: Final = 900
"""How long the Turn is given to reach a terminal event.

Longer than the three-leg file's 600 s because this Turn does strictly more: it reads a
3 KiB document, makes a round trip to a public MCP server through the Tool Gateway,
writes and runs Python that renders a PDF, and then ship-out transfers that PDF to the
object store before the log can settle. A cold image pull and a slow deepwiki answer add
to each other, and a Turn still working when the poll gave up reads as one that never
answered.
"""

_SHIPOUT_GRACE_S: Final = 120
"""How long to keep reading the log after the Turn ends, waiting for `output.produced`.

Kept, and no longer load-bearing, which is worth saying rather than deleting silently.

It was written when ship-out ran after `turn.completed` was appended, so the
announcement landed AFTER the terminal event and a poll that stopped there would race
the transfer and report a missing file on a Turn that produced one. Since 2026-08-26 the
marker is appended once the seam has returned, so `output.produced` is already in the
log by the time `turn.completed` is -- and this grace covers only the gap between the
append and this reader seeing it.

Bounded rather than unbounded either way: a Turn that genuinely wrote nothing must fail
this file rather than hang it.
"""

requires_the_cluster = pytest.mark.skipif(
    os.environ.get(_GATE) != "1",
    reason=(
        f"{_GATE}=1 not set. This case places a pod in the real {NAMESPACE} namespace, "
        "calls a real model, reaches a public MCP server, and writes to the object "
        "store; it must not run because somebody typed pytest."
    ),
)


def _doc_code() -> str:
    """A token that exists in this process and in the uploaded bytes, nowhere else.

    Upper-case hex behind a prefix, because the model has to reproduce it exactly in
    prose and inside a PDF, and a token it might reasonably re-case or hyphenate would
    fail a leg that worked.
    """
    return f"DOC-{uuid4().hex[:12].upper()}"


def _skill_receipt() -> str:
    """The skill's token, shaped so it cannot be confused with the document's.

    Lower-case words, unlike `_doc_code`'s upper-case hex. The live run of 2026-08-23 is
    why: both tokens were upper-case hex behind a word, and the agent ended its answer
    with the skill's receipt line carrying the DOCUMENT's token. That proved the skill
    body reached the model while failing an exact-string assertion, over an ambiguity
    test itself had introduced.
    """
    return f"pdf-receipt-{uuid4().hex[:6]}"


def _a_document(code: str) -> bytes:
    """The vendored README with this run's reference code appended.

    Appended rather than substituted in, so `codex_blog.md` stays byte-identical to
    upstream and the digest recorded beside it keeps being checkable. The code goes at
    the end under a heading of its own, where it reads as part of the document rather
    than as an artefact bolted on -- and it is NOT in the file name, which the prompt
    carries: a code in the name is quotable by an agent that never opened the file.
    """
    return (
        codex_blog().rstrip()
        + "\n\n## Document control\n\n"
        + f"Reference code: {code}\n"
    ).encode()


def _a_skill_with_a_receipt(receipt: str) -> str:
    """Anthropic's `pdf` skill, verbatim, with one receipt instruction appended.

    **The 8072 upstream bytes are not touched.** They are the subject: a 437-character
    description, a frontmatter key this platform does not model, and pointers to sibling
    files it cannot deliver, all of which it either handles or visibly does not.

    The appended paragraph is this file's own, and the only way leg 3 is gradeable.
    Nothing else the skill asks for leaves a trace an assertion can find: a PDF rendered
    by an agent that read this skill and one rendered by an agent that guessed reportlab
    from memory are the same PDF. A receipt line appearing in no other text the Session
    ever sees separates them.
    """
    return (
        SKILL_MD.rstrip()
        + "\n\n## Receipt\n\n"
        + "After you have written the PDF, end your reply with this line, copied\n"
        + "character for character. It is a fixed string: do not substitute the\n"
        + "document's reference code or any other identifier into it.\n"
        + "\n"
        + f"    PDF-SKILL-USED: {receipt}\n"
    )


def _prompt(name: str, tool: str, code_hint: str) -> str:
    """Four numbered demands, one per leg, each asking for something quotable.

    Numbered and explicit because a vague prompt makes a failed leg ambiguous: an agent
    that was never told to call the tool has not shown the tool is unreachable, and an
    agent that was never told where to write the PDF has not shown ship-out is broken.

    Leg 3 names the reference code as required PDF content, and that is the load-bearing
    sentence in this file. It is what makes the downloaded bytes prove the whole chain
    rather than prove that reportlab works.

    `tool` is passed in rather than spelled here, because what the model is shown is the
    server and the tool joined -- the bare registered name reaches it as nothing, and
    the agent reports a tool that is not available rather than a tool that failed.
    """
    return (
        "Four things, in order, and report each one as you finish it.\n"
        f"1. List the directory ./files/ and read {name}. Quote the reference code it "
        "contains, exactly as written.\n"
        f"2. Call the {tool} tool to ask the {_REPO_IN_SCOPE} repository how its "
        f"{_FEATURE} works. Quote one sentence of the answer you get back.\n"
        f"3. Follow your pdf skill to write {_OUTPUT_NAME} in your current working "
        "directory -- not in a subdirectory. It must be a real PDF of at least one "
        "page, and its text must include: a short brief of the document you read, the "
        f"sentence you quoted in step 2, and the line 'Reference code: {code_hint}' "
        "with the actual code in place of that placeholder.\n"
        f"4. Confirm that {_OUTPUT_NAME} exists and say how many bytes it is.\n"
    )


def _session_image() -> str:
    """The newest digest in the Session repository, resolved rather than pinned.

    Newest-push, matching the files beside this one: a digest written in here would pin
    the run to whatever ECR held the day somebody typed it, and this file's subject is
    a toolchain that arrived recently -- which is exactly what a stale image lacks.
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


@dataclass(frozen=True, slots=True)
class _Journey:
    """Everything one run of this file produced, for the cases to read.

    A frozen record rather than several fixtures, because the Turn must run ONCE. Six
    cases each placing a pod and calling a model would cost six placements and would
    grade six different Turns, so a flaky leg could not be told from a real failure.
    """

    session_id: SessionId
    tenant_id: str
    base: str
    doc_code: str
    receipt: str
    events: list[dict[str, Any]]
    census: list[str]

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

    @property
    def produced(self) -> list[dict[str, Any]]:
        return [one for one in self.events if one["type"] == "output.produced"]

    @property
    def tools_the_model_got(self) -> list[str]:
        """Every tool name the Model Gateway recorded handing this Session's model.

        Parsed out of the census lines rather than re-derived, because the point of
        reading them at all is to grade what the gateway itself observed sending. One
        Turn can take several requests -- the runtime re-asks after each tool result --
        so this is the union across them and not the last one's list.
        """
        got: list[str] = []
        for line in self.census:
            _, _, listed = line.partition("names=")
            for name in listed.strip().strip("[]").split(","):
                if bare := name.strip().strip("'\""):
                    got.append(bare)
        return sorted(set(got))

    @property
    def tools_the_wire_dropped(self) -> list[str]:
        """Every offered tool the request translator recorded refusing to carry.

        The other half of the subtraction above, and the half that leaves no other
        trace. A carried tool is named in the body the model receives; a dropped one is
        named nowhere except this line, so without it a reader holding "14 offered, 13
        translated" has to know the wire table by heart to work out which one went.
        """
        dropped: list[str] = []
        for line in self.census:
            _, _, listed = line.partition("dropped=")
            for name in listed.strip().strip("[]").split(","):
                if bare := name.strip().strip("'\""):
                    dropped.append(bare)
        return sorted(set(dropped))


def _register(
    base: str, tenant_id: str, image: str, run: str, receipt: str, code: str
) -> tuple[SessionId, str, str]:
    """Upload the document and the skill, register the server, create the Session.

    Returns the Session's id, the name the workspace will hold the file under, and the
    name the model will see for the tool -- all three because the prompt has to name
    them and this is the one place that knows any of them.

    Everything goes through the REST API; nothing touches the database or the cluster.
    That is what makes the pod found later evidence: this run had no other way to make
    one.

    No `allowed_domains` on the Environment, deliberately. The whole PDF toolchain is in
    the image, so this errand needs no egress -- and a Session that granted pypi.org
    would leave it unknown whether the skill worked or whether the agent fetched its way
    out of a gap.
    """
    with _client(base, tenant_id) as caller:
        name = f"codex-blog-{run}.md"
        uploaded = caller.post(
            "/v1/files",
            files={"file": (name, _a_document(code), "text/markdown")},
        )
        assert uploaded.status_code in (200, 201), uploaded.text
        file_id = uploaded.json()["id"]

        skill = _created(
            caller.post(
                "/v1/skills", json={"skill_md": _a_skill_with_a_receipt(receipt)}
            ),
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
        advertised = advertised_name_for(server_name, _TOOL)

        environment = _created(
            caller.post(
                "/v1/environments",
                json={"name": f"journey-{run.lower()}", "runtime_image": image},
            )
        )
        definition = _created(
            caller.post(
                "/v1/agents",
                json={
                    "name": f"journey-{run.lower()}",
                    "instructions": (
                        "You are a careful research assistant. Files attached to your "
                        "Session are in ./files/ relative to your working directory. "
                        "Use your skills when they apply, and follow their "
                        "instructions exactly."
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
                    "grant": [advertised],
                    "scope": {"repo": _REPO_IN_SCOPE},
                    "budget_minor_units": 1_000_000,
                    "budget_currency": "USD",
                    "retention_days": 1,
                },
            )
        )
        return SessionId(UUID(session["id"])), name, advertised


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


def _await_the_errand(
    base: str, tenant_id: str, session_id: SessionId
) -> list[dict[str, Any]]:
    """Poll to the terminal event, then keep reading until ship-out has spoken.

    Two phases because they end on different signals and conflating them loses a real
    failure. The first waits for `turn.completed` or `turn.failed` -- both, not only the
    good one, since a poll waiting for success alone sits out its whole deadline on a
    Turn that died in the first second and then reports a timeout, sending the reader
    after the wrong thing. The second keeps reading for a bounded grace period, because
    ship-out runs AT completion and its `output.produced` therefore lands after the
    terminal event: a poll that stopped at the terminal event races the object-store
    transfer and report a missing file on a Turn that produced one.

    Returns as soon as an announcement appears rather than always waiting out the grace,
    so a healthy run costs nothing for it.
    """
    deadline = time.monotonic() + _TURN_DEADLINE_S
    events: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        events = _events(base, tenant_id, session_id)
        if any(one["type"] in ("turn.completed", "turn.failed") for one in events):
            break
        time.sleep(3)
    else:
        pytest.fail(
            f"session {session_id} produced no terminal event in {_TURN_DEADLINE_S}s; "
            f"the log was {[one['type'] for one in events]}"
        )

    grace = time.monotonic() + _SHIPOUT_GRACE_S
    while time.monotonic() < grace:
        if any(one["type"] == "output.produced" for one in events):
            return events
        time.sleep(3)
        events = _events(base, tenant_id, session_id)
    return events


def _the_census_for(session_id: SessionId) -> list[str]:
    """The Model Gateway's tool-census lines for this Session, newest run included.

    Read from the gateway's own log rather than from anything this test computes,
    because the question these cases ask is what the platform ACTUALLY sent -- a value
    re-derived here would agree with the code under test by construction and would have
    agreed with it throughout the outage that prompted these cases.

    Filtered by Session id, which is why that id is on the line. The gateway serves
    every Session in the namespace at once, so a census matched on the tool name alone
    would pass this file on a line another tenant's Turn wrote.

    Never `check`ed. A gateway that rotated its pod mid-run, or whose log has aged out,
    yields no line -- and the cases below then fail saying the census was empty, which
    is the honest report. Turning that into a `kubectl` error would blame the wrong
    thing.
    """
    out = kubectl(
        "logs",
        "-n",
        NAMESPACE,
        "-l",
        "map.component=model-gateway",
        "--tail=-1",
        "--since=30m",
        check=False,
    )
    return [
        line
        for line in out.splitlines()
        if "tool census:" in line and str(session_id) in line
    ]


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
def journey() -> Iterator[_Journey]:
    """Place one Session, run one Turn, and hand every case the same log.

    The port-forward is held open for the whole module, because the download case has to
    reach the API after the Turn is over and the lifecycle case has to reach it after
    that.

    The teardown deletes the pod and its Secrets whatever happened, including a failure
    during submission -- a run that died there still created a Session the control plane
    places a pod for, and three aborted runs once left forty-two pods squatting the
    namespace, after which the next run's scheduling refusal read as the cluster being
    out of capacity.
    """
    doc_code = _doc_code()
    receipt = _skill_receipt()
    tenant_id = str(uuid4())
    image = _session_image()
    run = uuid4().hex[:8]
    with forwarded(_CONTROL_PLANE, _CONTROL_PLANE_PORT) as base:
        session_id, name, tool = _register(
            base, tenant_id, image, run, receipt, doc_code
        )
        try:
            _submit(base, tenant_id, session_id, _prompt(name, tool, "<the code>"))
            events = _await_the_errand(base, tenant_id, session_id)
            yield _Journey(
                session_id=session_id,
                tenant_id=tenant_id,
                base=base,
                doc_code=doc_code,
                receipt=receipt,
                events=events,
                census=_the_census_for(session_id),
            )
        finally:
            _clean_up(pod_name_for(session_id))


@requires_the_cluster
def test_the_turn_reached_a_terminal_event_at_all(journey: _Journey) -> None:
    """First, because every case below reads the same log this one proves exists.

    Printed rather than only asserted: when a leg below fails, the transcript is what
    says whether the agent misunderstood the errand or the platform denied it something,
    and reconstructing that from an assertion message is not possible.
    """
    types = [one["type"] for one in journey.events]
    print(f"\n--- event types ---\n{types}")
    print(f"\n--- what the agent said ---\n{journey.answer}")
    assert "turn.failed" not in types, journey.events
    assert "turn.completed" in types, types


@requires_the_cluster
def test_the_agent_read_the_document_the_tenant_uploaded(journey: _Journey) -> None:
    """Leg 1. The code is in the bytes and in this process, nowhere else.

    Not in the file name, which the prompt carries -- so this cannot be passed by an
    agent that read the prompt and never opened the file. And not anywhere in the Codex
    README itself, so it cannot be passed from training either.
    """
    assert journey.doc_code in journey.answer, (
        f"the agent never quoted {journey.doc_code}; it said: {journey.answer!r}"
    )


@requires_the_cluster
def test_the_model_was_handed_the_tool_this_tenant_granted(journey: _Journey) -> None:
    """Leg 2's precondition, graded on its own so the cascade reports its own head.

    Every other case in this file grades what the model DID. None of them could see what
    it was given, and for nineteen consecutive live Turns that was the whole defect: the
    Tool Gateway logged `registered=1 offered=1`, the model was handed twelve built-ins
    and none of them, and the four legs that failed each failed for a reason that had
    nothing to do with what they grade. This is the line that says so in one sentence.

    Read out of the Model Gateway's own log and not recomputed here. A value this test
    derived would agree with the translator by construction, which is exactly how the
    defect survived a suite that was otherwise green.

    The union across the Turn's requests, because the granted tool does not appear on
    the first one. The runtime offers `tool_search` and defers the rest, so the tool
    becomes callable only on the request after the search that found it -- and a census
    read from the first request alone would report this defect as still present on a
    healthy Turn.
    """
    got = journey.tools_the_model_got
    print(f"\n--- tools the model was handed ---\n{got}")
    assert journey.census, (
        "the Model Gateway logged no tool census for this Session, so what the model "
        "was handed is unknown; check that the gateway is serving and that its log "
        "level admits INFO"
    )
    assert any(name.endswith(_TOOL) for name in got), (
        f"the platform granted {_TOOL} and never handed it to the model; the model "
        f"was given {got}, so every leg below is grading a Turn that could not have "
        "run the errand"
    )


@requires_the_cluster
def test_the_search_tool_the_runtime_defers_every_other_tool_behind_arrived(
    journey: _Journey,
) -> None:
    """The single row whose loss costs the whole catalogue, pinned by name.

    codex puts no MCP tool in a request's tool list. It offers one `tool_search` tool
    and defers the rest, and in 0.149.0 that is not configurable -- the flag that would
    turn it off is at stage `Removed`, so the parser skips the key rather than honouring
    it. A wire that drops this tool therefore drops every tool the tenant granted, and
    it does so silently: the request is well-formed, the Turn completes, and the model
    explains at length that it has no such tool.

    Named separately from the case above because the two fail differently. That one says
    a granted tool did not arrive, which has many causes. This one says WHICH cause, and
    it is the cause that has already happened once.
    """
    got = journey.tools_the_model_got
    assert "tool_search" in got, (
        "the model was handed no tool_search, which is the tool codex defers every MCP "
        f"tool behind; it was given {got}, so no granted tool can be reached this Turn "
        "whatever the Tool Gateway offered"
    )


@requires_the_cluster
def test_the_editing_tool_the_runtime_teaches_is_dropped_and_says_so(
    journey: _Journey,
) -> None:
    """A drop that is deliberate has to look different from a drop that is a defect.

    The runtime offers `apply_patch` on every Turn as a freeform tool -- constrained by
    a Lark grammar, its call carried back as raw text -- and this wire has a field for
    neither, so it does not cross. That is a decision and not this file's to reverse.
    What this case pins is that the decision stays visible: the same shape of absence,
    unexplained, is what cost nineteen live Turns, and the only thing separating the two
    is whether the platform says out loud which tool it declined and why.

    So the absence is asserted together with the census line that accounts for it. An
    `apply_patch` that starts arriving means somebody built the grammar and the return
    item, which is real work and belongs here as a deliberate edit. A census that stops
    naming it means the accounting was lost while the drop stayed -- the same silent
    state the outage ran in, reached from the other direction.

    The cost is worth stating plainly, because nothing else in the suite states it: the
    runtime's own base_instructions teach the model to use this tool, so every Turn this
    platform serves arrives with the model instructed in an editing tool it does not
    have. Nothing fails, because editing also works through the shell -- which is
    exactly why this needs a test rather than a reader noticing.
    """
    got = journey.tools_the_model_got
    dropped = journey.tools_the_wire_dropped
    print(f"\n--- tools the wire dropped ---\n{dropped}")
    assert journey.census, (
        "the Model Gateway logged no tool census for this Session, so neither what the "
        "model was handed nor what was withheld can be read"
    )
    assert "apply_patch" not in got, (
        "apply_patch reached the model, which this wire has no way to serve: its call "
        "comes back as raw text against a Lark grammar and no item on this side "
        "carries that. If it was carried on purpose, the tool.custom row and the two "
        "item.custom_tool_call rows in the wire table have to move with it"
    )
    assert "apply_patch" in dropped, (
        "apply_patch is absent from the model's tools and the census does not say the "
        f"wire dropped it; it reported dropping {dropped}. An unexplained absence is "
        "the exact shape the tool_search outage held for nineteen runs"
    )


@requires_the_cluster
def test_the_turn_did_not_go_hunting_for_a_tool_it_should_have_been_handed(
    journey: _Journey,
) -> None:
    """The failure this file used to report as an intermittent second defect.

    It is not a second defect. When the granted tool is missing the model has two
    reasonable answers, and it picks between them run to run: work around the gap and
    finish the errand, or treat it as a blocker, refuse to fabricate a citation, and
    stop to ask the tenant for the tool. The first costs one failing leg and ninety
    seconds; the second costs four failing legs and three minutes, because steps three
    and four never run -- no skill receipt, no PDF, no `output.produced`.

    That bimodality read as flakiness for a while, and it is not: it is one cause with
    two model-side responses. Grading it here means the Turn that goes hunting says so
    under a name that points at the cause, instead of four downstream legs each
    reporting the absence of something they were never going to see.

    Graded on the Event Log. `list_mcp_resources` and `list_mcp_resource_templates` are
    the Agent Runtime's own introspection verbs, on a server it calls `codex`; this
    errand grants no MCP resources at all, so a Turn that reaches for them is a Turn
    looking for something it was told it had and cannot find.
    """
    hunting = [
        one
        for one in journey.events
        if one["type"] == tool_call.TOOL_CALLED
        and "list_mcp_resource" in str(one["payload"].get("tool", ""))
    ]
    assert not hunting, (
        "the model went looking for its granted tool through codex's own MCP "
        f"introspection verbs, which means it was not handed {_TOOL}; it called "
        f"{[one['payload'].get('tool') for one in hunting]} and then said: "
        f"{journey.answer!r}"
    )


@requires_the_cluster
def test_the_agent_researched_the_feature_through_the_tool_gateway(
    journey: _Journey,
) -> None:
    """Leg 2. Graded on the Event Log, not on the agent's prose.

    A tool call appears in the log as its own event whatever the agent then says about
    it, so this asks the platform rather than the model. An agent that claimed to have
    called the tool and did not would pass a prose assertion and fail this one.

    **It asks for THIS tool by name, and the earlier version did not.** It asserted only
    that some `tool.` event existed, and the Agent Runtime offers introspection tools of
    its own -- `list_mcp_resources`, `list_mcp_resource_templates` -- on a server it
    calls `codex`. An agent that went looking for the granted tool, failed to find it,
    and gave up therefore emitted two tool events and passed this case, on three
    consecutive live runs where the Tool Gateway received nothing at all. A leg that
    passes on the evidence of the agent noticing the tool is missing is not grading the
    leg it is named for.

    `tool.server_unavailable` is asserted absent for the same reason it exists: it is
    the platform's own statement that a granted server never came up, and a Turn
    carrying one has not exercised the Tool Gateway whatever else it did.
    """
    calls = [one for one in journey.events if one["type"].startswith("tool.")]
    print(f"\n--- tool events ---\n{[one['type'] for one in calls]}")
    down = [one for one in calls if one["type"] == tool_server.TOOL_SERVER_UNAVAILABLE]
    assert not down, (
        "a tool server this Session was granted never came up, so nothing reached the "
        f"Tool Gateway: {[one['payload'] for one in down]}"
    )
    reached = [
        one
        for one in calls
        if one["type"] == tool_call.TOOL_CALLED
        and str(one["payload"].get("tool", "")).endswith(_TOOL)
    ]
    assert reached, (
        f"no call to {_TOOL} reached the Tool Gateway; the tool events were "
        f"{[(one['type'], one['payload'].get('tool')) for one in calls]}"
    )
    assert not [one for one in calls if one["type"] == "tool.denied"], calls


@requires_the_cluster
def test_the_pdf_skills_own_text_reached_the_model(journey: _Journey) -> None:
    """Leg 3, first half: the skill was delivered AND read.

    The receipt token appears in no other text this Session ever sees -- not in the
    prompt, not in the document, not in the agent's instructions -- so it can only
    write it by having read the registered skill body. That is what separates "followed
    Anthropic's skill" from "knew reportlab already", which no assertion about the PDF
    itself can do.
    """
    assert journey.receipt in journey.answer, (
        f"the agent never wrote the receipt {journey.receipt}; it said: "
        f"{journey.answer!r}"
    )


@requires_the_cluster
def test_the_produced_pdf_was_announced_with_an_id_that_downloads_it(
    journey: _Journey,
) -> None:
    """Leg 4, first half: the tenant can find out the file exists.

    Without this event the bytes are unreachable. Ship-out stores them under a freshly
    minted id, no route lists a tenant's files, and `resources` reads `session.created`
    -- so a Turn that produced a document would end with the document safely in S3 and
    no answer to "where is my document".
    """
    print(f"\n--- produced ---\n{journey.produced}")
    assert journey.produced, (
        "nothing was announced as produced; the log was "
        f"{[one['type'] for one in journey.events]}"
    )
    paths = [one["payload"]["path"] for one in journey.produced]
    assert _OUTPUT_NAME in paths, paths
    announced = next(
        one for one in journey.produced if one["payload"]["path"] == _OUTPUT_NAME
    )
    assert int(announced["payload"]["byte_length"]) > 0, announced
    # Before the terminal event, not after it. `turn.completed` is appended once the
    # seam that ships out has returned, so the announcement that seam makes lands
    # first -- see the ordering case in
    # `tests/pod/test_a_session_ships_out_a_document_and_then_closes.py`, which is
    # where that claim is argued rather than merely used.
    types = [one["type"] for one in journey.events]
    assert types.index("output.produced") < types.index("turn.completed"), types


@requires_the_cluster
def test_the_pdf_that_came_back_is_about_this_errand(journey: _Journey) -> None:
    """**Leg 4, and the case this whole file is for.**

    The bytes come back through the public download route, are parsed as a PDF rather
    than sniffed, and the document's reference code is looked for in the extracted text.

    That one string can only be in there if every leg worked: the upload stored it, the
    mount delivered it, the agent read it, the skill's toolchain rendered it, ship-out
    transferred the render into the Session's `artifacts` lane, `output.produced` named
    the path, and the download route served the bytes to a caller that never saw the
    pod. No single piece behaving well produces this result.

    The path here is a bare filename and stays one: this errand asks for a document at
    the top of the output directory, so the separator case that the artifacts lane
    exists for is exercised by the file beside this one rather than here.

    `pypdf` parses rather than `%PDF-` alone, because a file can carry that header and
    still be unopenable -- and an unopenable PDF is a failed errand that a magic-byte
    check reports as a pass.
    """
    from pypdf import PdfReader

    announced = next(
        one for one in journey.produced if one["payload"]["path"] == _OUTPUT_NAME
    )
    path = announced["payload"]["path"]
    with _client(journey.base, journey.tenant_id) as caller:
        got = caller.get(f"/v1/sessions/{journey.session_id}/artifacts/{path}")
    assert got.status_code == 200, got.text
    assert got.content.startswith(b"%PDF-"), got.content[:32]
    assert len(got.content) == int(announced["payload"]["byte_length"])

    written = Path(os.environ.get("MAP_PDF_OUT", "/tmp")) / _OUTPUT_NAME
    written.write_bytes(got.content)
    print(f"\n--- the PDF, {len(got.content)} bytes, saved at {written} ---")

    reader = PdfReader(written)
    assert len(reader.pages) >= 1, "a PDF with no pages"
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    print(f"\n--- extracted text ---\n{text}")
    assert journey.doc_code in text, (
        f"the PDF does not carry {journey.doc_code}, so it is not a document about "
        f"the file the tenant uploaded; its text was {text!r}"
    )


@requires_the_cluster
def test_the_session_closes_without_taking_the_evidence_with_it(
    journey: _Journey,
) -> None:
    """Leg 5, run last because it ends the Session every case above reads.

    Archive is idempotent: a second call answers 200 rather than a conflict, so a caller
    that lost the first response can retry without having to distinguish "already
    archived" from "no such Session". `DELETE` stops the Session and KEEPS its history,
    which is the property that matters here -- the Event Log still reads afterwards, so
    closing a Session does not destroy the record of what it produced. A tenant who
    downloads the PDF next week still has the announcement that names it.
    """
    with _client(journey.base, journey.tenant_id) as caller:
        archived = caller.post(f"/v1/sessions/{journey.session_id}/archive")
        assert archived.status_code == 200, archived.text
        again = caller.post(f"/v1/sessions/{journey.session_id}/archive")
        assert again.status_code == 200, again.text
        closed = caller.delete(f"/v1/sessions/{journey.session_id}")
        assert closed.status_code == 200, closed.text

    after = _events(journey.base, journey.tenant_id, journey.session_id)
    assert [one for one in after if one["type"] == "output.produced"], (
        "the produced-file announcement did not survive the Session closing; the log "
        f"read {[one['type'] for one in after]}"
    )
