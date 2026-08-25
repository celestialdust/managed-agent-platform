"""One Turn that authenticates to an MCP server with a credential a tenant brought.

Skipped unless MAP_CLUSTER_TESTS=1. SKIPPED MEANS NOTHING RAN -- no pod was placed, no
model was called, nothing reached the public internet, and nothing here is evidence of
anything on that run.

Why this exists beside the other tool cases. Every one of them calls deepwiki, which
authenticates nobody: `for_http` returns `NoCredential` for a registration naming no
ref, so those Turns prove the Tool Gateway can reach a public server and prove exactly
nothing about the credential path. That path had never been exercised end to end at all
-- and when it was measured, the Tool Gateway's role was granted `map/dev/tools/*` while
`vault_name` composes `map/tool-credential/...`, so every authenticated server this
account could have registered would have failed at AWS. A public-server test cannot
find that, which is why it went unfound.

**The chain this grades, link by link.** A tenant writes a value through
`POST /v1/vaults/{id}/credentials`; the control plane composes
`map/tool-credential/<tenant>/<vault>/<credential>` and writes it to Secrets Manager;
the Tool Gateway, in another process under another role, composes the same name from
the ref in a server registration and reads it back; it becomes an `Authorization`
header on an outbound Streamable HTTP call. Four of the five cases below cut that chain
at a different link, so a break reports where it broke rather than "the Turn failed".

**The negative control is the point of the second server.** Tavily answers HTTP 200 to
`initialize` whatever key it is given and validates only at `tools/call`, where a wrong
key comes back as a 200 whose body says `Unauthorized: missing or invalid API key`. So
one server carrying a real key proves less than it looks: a platform that dropped the
header entirely would produce a 401 and a platform that sent a *stale* one would produce
that same body, and a single passing arm cannot tell either from success. Two servers in
one Turn -- one credential real, one deliberately wrong -- make the two arms differ, and
they can only differ if what this platform composed is what Tavily read.

NO KEY VALUE IS PRINTED, ASSERTED ON, OR PUT IN A FAILURE MESSAGE. The one case that
touches a value compares two of them and asserts a bool; the strings themselves reach no
assertion message, no log line and no report.
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

from managed_agent.control.api.request.tenancy import TENANT_HEADER
from managed_agent.control.session.placement import pod_name_for
from managed_agent.core.ids import SessionId, TenantId
from managed_agent.gateway.tool.credential_broker import vault_name

_GATE: Final = "MAP_CLUSTER_TESTS"
_CONTROL_PLANE: Final = "deploy/control-plane"
_CONTROL_PLANE_PORT: Final = 8080

_REGION: Final = "us-east-1"
_REPOSITORY: Final = "map/session-shim"
_TOOL_GATEWAY_ROLE: Final = "arn:aws:iam::062677866851:role/map-tool-gateway"

_MODEL: Final = "gsds-claude-opus-4-6"

_SECRET_SUFFIXES: Final = ("compiled", "requirements", "shim-token")

_TAVILY: Final = "https://mcp.tavily.com/mcp/"
_LIVE_TOOL: Final = "tavily_search"
_WRONG_TOOL: Final = "tavily_search_with_a_bad_key"
_TOPIC_IN_SCOPE: Final = "general"
_UNSCOPED_TOOL: Final = "tavily_search_outside_the_scope"
_DIMENSION_NOBODY_SCOPED: Final = "region"
"""A dimension this Session's Scope does not carry, bound by a third registration.

The third registration names the **working** credential and a healthy server, so the
only thing that can stop its call is the clamp. That is what makes it evidence rather
than decoration: without the clamp the call goes out with a good key and Tavily answers
it, and the arm reports OK.
"""

_WRONG_KEY: Final = "Bearer tvly-dev-000000000000000000000000000000"
"""A key shaped like Tavily's and belonging to nobody, for the negative arm.

Invented here rather than derived from the real one by mutating a character, which
would put a near-copy of a live credential in a tracked file. Tavily rejects it at
`tools/call` the same way it rejects any other key it does not hold.
"""

_ENV_KEY: Final = "TAVILY_API_KEY"

_SUBMIT_TIMEOUT_S: Final = 660
"""How long this client waits for the API to accept the Turn.

A Session's first Turn is answered only once its pod is placed, so this response is held
    for the whole placement -- which on an autoscaled cluster includes waiting for a
    node and pulling the image.
"""

_TURN_DEADLINE_S: Final = 600
"""How long the Turn is given to reach a terminal event.

Two round trips to a public MCP server through the Tool Gateway, one of which is
expected to come back an error the agent then has to report rather than retry.
"""

requires_the_cluster = pytest.mark.skipif(
    os.environ.get(_GATE) != "1",
    reason=(
        f"{_GATE}=1 not set. This case places a pod in the real {NAMESPACE} namespace, "
        "calls a real model, writes a credential to AWS Secrets Manager and reaches a "
        "public MCP server; it must not run because somebody typed pytest."
    ),
)


def _tavily_key() -> str:
    """The complete header value for the live arm, read from the environment or `.env`.

    Read rather than stored, and returned as the whole header value rather than the bare
    token: `HttpAttachment.into_headers` writes what it is given, so the scheme belongs
    with the value -- `Authorization` wants `Bearer x` and `X-Api-Key` wants `x`, and a
    platform deriving the scheme from the header name would produce a header only the
    far end could tell was wrong.

    `.env` is read as a fallback because that is where this repository's operator keeps
    it and an exported variable is one `env` away from a terminal recording. The value
    goes to one place -- the request body below -- and is never returned to a caller
    that prints.
    """
    from_environment = os.environ.get(_ENV_KEY)
    if from_environment:
        return f"Bearer {from_environment.strip()}"
    dotenv = Path(__file__).resolve().parents[2] / ".env"
    if dotenv.exists():
        for line in dotenv.read_text().splitlines():
            if line.startswith(f"{_ENV_KEY}="):
                return f"Bearer {line.split('=', 1)[1].strip().strip('"' + chr(39))}"
    pytest.fail(
        f"no {_ENV_KEY} in the environment or in .env, so the live arm would register "
        "a credential that authenticates nobody and this case would grade Tavily's "
        "refusal twice"
    )


@dataclass(frozen=True, slots=True)
class _Run:
    """Everything one run of this file produced, for the cases to read.

    A frozen record rather than several fixtures, because the Turn must run ONCE: five
    cases each placing a pod would cost five placements and would grade five different
    Turns, so a flaky arm could not be told from a real failure.
    """

    session_id: SessionId
    tenant_id: str
    live_entry: str
    wrong_entry: str
    events: list[dict[str, Any]]

    @property
    def pod_name(self) -> str:
        return pod_name_for(self.session_id)

    @property
    def answer(self) -> str:
        """Everything the agent said this Turn, as one string."""
        return "".join(
            str(one["payload"].get("text", ""))
            for one in self.events
            if one["type"] == "turn.message_delta"
        )


def _prompt() -> str:
    """Three calls and three marker lines, because each arm is graded separately.

    Explicit about not retrying and not substituting, because two of the three arms
    are supposed to fail: an agent that quietly falls back to the working server would
    produce three successes and both controls would evaporate.

    The three arms fail for three different reasons, which is why one prompt drives
    them all. `LIVE` must succeed. `WRONG` must fail at Tavily, because the key is
    nobody's. `SCOPED` must fail *here*, before anything is sent, because the tool is
    registered against a Scope dimension this Session does not carry -- and it names
    the working key, so a platform that did not clamp would report it OK.
    """
    return (
        "You have three search tools. Call each one exactly once, do not retry any of "
        "them, and do not use one in place of another.\n"
        f"1. Call {_LIVE_TOOL} with query 'capital city of France' and topic "
        "'general'.\n"
        f"2. Call {_WRONG_TOOL} with query 'capital city of France' and topic "
        "'general'.\n"
        f"3. Call {_UNSCOPED_TOOL} with query 'capital city of France' and topic "
        "'general'.\n"
        "Then reply with exactly these three lines and nothing after them:\n"
        f"LIVE: <OK followed by one fact from the results, or FAILED followed by the "
        "error text you received>\n"
        f"WRONG: <OK followed by one fact from the results, or FAILED followed by the "
        "error text you received>\n"
        f"SCOPED: <OK followed by one fact from the results, or FAILED followed by the "
        "error text you received>\n"
    )


def _session_image() -> str:
    """The newest digest in the Session repository, resolved rather than pinned."""
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


def _server(
    server_name: str, tool_name: str, ref: str, dimension: str = "topic"
) -> dict[str, Any]:
    """One Tavily registration, differing from the other only in which ref it names.

    `topic` carries the Scope Binding rather than `query`, and that is deliberate: a
    binding is required on every tool, and binding `query` would mean a Scope value
    standing where the model's question goes if bindings are ever applied to an
    outbound call. `topic` accepts `general` either way, so this registration is
    correct under both behaviours.
    """
    return {
        "server_name": server_name,
        "endpoint": {
            "transport": "streamable_http",
            "url": _TAVILY,
            "credential_ref": ref,
            "credential_header": "Authorization",
        },
        "tools": [
            {
                "name": tool_name,
                "remote_name": "tavily_search",
                "parameters": {"query": "string", "topic": "string"},
                "scope_bindings": [{"dimension": dimension, "argument": "topic"}],
            }
        ],
    }


def _register(base: str, tenant_id: str, image: str, run: str) -> tuple[SessionId, str]:
    """Build the whole precondition through the REST API: two credentials, two servers.

    Returns the Session's id and the vault's name, which the composed entry names below
    are built from.

    Nothing here touches the database, the cluster or Secrets Manager directly. That is
    what makes the vault entries found later evidence: this run had no other way to
    write one.
    """
    with _client(base, tenant_id) as caller:
        vault = _created(
            caller.post("/v1/vaults", json={"name": f"tavily-{run}"}),
        )
        live = _created(
            caller.post(
                f"/v1/vaults/{vault['id']}/credentials",
                json={
                    "name": "live",
                    "kind": "static_bearer",
                    "value": _tavily_key(),
                },
            )
        )
        wrong = _created(
            caller.post(
                f"/v1/vaults/{vault['id']}/credentials",
                json={
                    "name": "wrong",
                    "kind": "static_bearer",
                    "value": _WRONG_KEY,
                },
            )
        )
        assert live["ref"] == f"{vault['name']}/live", live
        assert wrong["ref"] == f"{vault['name']}/wrong", wrong

        for registration in (
            _server(f"tavily-live-{run}", _LIVE_TOOL, live["ref"]),
            _server(f"tavily-wrong-{run}", _WRONG_TOOL, wrong["ref"]),
            _server(
                f"tavily-unscoped-{run}",
                _UNSCOPED_TOOL,
                live["ref"],
                dimension=_DIMENSION_NOBODY_SCOPED,
            ),
        ):
            registered = caller.post("/v1/mcp_servers", json=registration)
            assert registered.status_code in (200, 201), registered.text

        environment = _created(
            caller.post(
                "/v1/environments",
                json={"name": f"credentialed-{run}", "runtime_image": image},
            )
        )
        definition = _created(
            caller.post(
                "/v1/agents",
                json={
                    "name": f"credentialed-{run}",
                    "instructions": (
                        "You are a careful assistant. When a tool returns an error, "
                        "report the error text verbatim rather than working around it."
                    ),
                    "model": _MODEL,
                    "skills_repository": "git@github.com:acme/skills.git",
                    "skills_revision": "0" * 39 + "a",
                    "skills": [],
                    "tool_servers": [
                        f"tavily-live-{run}",
                        f"tavily-wrong-{run}",
                        f"tavily-unscoped-{run}",
                    ],
                },
            )
        )
        session = _created(
            caller.post(
                "/v1/sessions",
                json={
                    "definition_id": definition["id"],
                    "environment_id": environment["id"],
                    "grant": [_LIVE_TOOL, _WRONG_TOOL, _UNSCOPED_TOOL],
                    "scope": {"topic": _TOPIC_IN_SCOPE},
                    "budget_minor_units": 500_000,
                    "budget_currency": "USD",
                    "retention_days": 1,
                },
            )
        )
        return SessionId(UUID(session["id"])), str(vault["name"])


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
    """Poll until the Turn ends either way, and return the whole log."""
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


def _entry_holds(name: str, expected: str) -> bool:
    """Whether the vault entry at this name holds exactly this value.

    Returns a bool and nothing else. Both strings are credentials, so neither is
    returned, logged, or put where a failure message could render it -- the caller
    learns that they matched or that they did not, which is the whole fact this case
    needs.

    An entry that is absent is `False` rather than an error, because absent and
    different are the same answer to "did the control plane write where the Tool
    Gateway reads".
    """
    done = subprocess.run(
        (
            "aws",
            "secretsmanager",
            "get-secret-value",
            "--secret-id",
            name,
            "--region",
            _REGION,
            "--query",
            "SecretString",
            "--output",
            "text",
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode != 0:
        return False
    return done.stdout.strip() == expected


def _may_read(name: str) -> str:
    """IAM's own answer for the Tool Gateway reading this exact composed name.

    The exact name rather than a pattern, and simulated rather than reasoned about.
    The defect this case exists for was a policy whose prefix looked plausible and
    matched nothing `vault_name` can build, and a test comparing two strings written by
    the same hand would have agreed with it.
    """
    done = subprocess.run(
        (
            "aws",
            "iam",
            "simulate-principal-policy",
            "--policy-source-arn",
            _TOOL_GATEWAY_ROLE,
            "--action-names",
            "secretsmanager:GetSecretValue",
            "--resource-arns",
            f"arn:aws:secretsmanager:{_REGION}:062677866851:secret:{name}-AbCdEf",
            "--query",
            "EvaluationResults[0].EvalDecision",
            "--output",
            "text",
        ),
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


def _clean_up(base: str, tenant_id: str, session_id: SessionId) -> None:
    """Delete the pod, its Secrets, and the two vault entries this run wrote.

    The vault goes through `DELETE` **without being emptied first**, which is the whole
    revocation path in one call: the route erases each value and removes each row before
    it can remove the vault, because `vault_credential`'s foreign key has no cascade and
    the store refuses a vault anything still points at. Deleting the credentials
    individually here would sidestep exactly the sequence worth proving -- and did,
    until a route case caught that the deployed path answered 500 on this.

    Best effort: a teardown that raised would replace a real failure with its own.
    """
    pod_name = pod_name_for(session_id)
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
    try:
        with _client(base, tenant_id) as caller:
            for vault in caller.get("/v1/vaults").json()["data"]:
                removed = caller.delete(f"/v1/vaults/{vault['id']}")
                assert removed.status_code == 204, removed.text
    except Exception as failed:  # noqa: BLE001 - teardown must not mask the real failure
        print(f"vault teardown for tenant {tenant_id} did not finish: {failed!r}")


@pytest.fixture(scope="module")
def run() -> Iterator[_Run]:
    """Place one Session, run one Turn, and hand every case the same log.

    The two entry names are composed with the Tool Gateway's own `vault_name` -- the
    function the reading side calls -- rather than with an f-string here. A name spelled
    a second time in this file would be a name that can agree with a broken writer.
    """
    stamp = uuid4().hex[:8]
    tenant_id = str(uuid4())
    image = _session_image()
    with forwarded(_CONTROL_PLANE, _CONTROL_PLANE_PORT) as base:
        session_id, vault = _register(base, tenant_id, image, stamp)
        tenant = TenantId(UUID(tenant_id))
        try:
            _submit(base, tenant_id, session_id, _prompt())
            events = _await_terminal(base, tenant_id, session_id)
            yield _Run(
                session_id=session_id,
                tenant_id=tenant_id,
                live_entry=vault_name(tenant, f"{vault}/live"),
                wrong_entry=vault_name(tenant, f"{vault}/wrong"),
                events=events,
            )
        finally:
            _clean_up(base, tenant_id, session_id)


@requires_the_cluster
def test_the_turn_reached_a_terminal_event_at_all(run: _Run) -> None:
    """`turn.completed`, not `turn.failed`, and asserted before the arms.

    First because it separates two very different reports. A Turn that failed makes both
    arms fail, and two failing arms read as a broken credential path rather than as one
    Turn that never ran.
    """
    types = [one["type"] for one in run.events]
    assert "turn.completed" in types, types


@requires_the_cluster
def test_the_written_credential_is_at_the_name_the_gateway_reads(run: _Run) -> None:
    """The value the tenant submitted is at the name `vault_name` composes.

    This is the link the offline suite can only half-prove. `test_a_written_credential_
    lands_where_the_gateway_reads.py` shows the two sides compose one string; this shows
    the control plane's writer actually put the value there, under the role it runs as,
    in the account this platform deploys to.

    Compared, never printed. A mismatch reports that it mismatched.
    """
    assert _entry_holds(run.live_entry, _tavily_key()), (
        f"nothing at {run.live_entry} matches what was submitted, so the credential "
        "the tenant registered is not the one the Tool Gateway would read. Either the "
        "control plane wrote elsewhere or it did not write at all -- check the "
        "WriteToolCredentialsNeverReadThem statement in deploy/iam/map-control-plane."
        "json. The value itself is withheld."
    )


@requires_the_cluster
def test_the_tool_gateway_may_read_that_exact_name(run: _Run) -> None:
    """IAM allows the reading role on the composed name, not on a plausible pattern.

    Separate from the case above because the two fail for different reasons and the
    fixes are in different files: that one fails when the write went somewhere else,
    this one when the write went to the right place and the reader is not permitted
    there. Both were true at once before 2026-08-25, and the pair of them read as "tool
    calls do not work".
    """
    decision = _may_read(run.live_entry)
    assert decision == "allowed", (
        f"IAM answers {decision} for map-tool-gateway reading {run.live_entry}. Every "
        "authenticated MCP server this tenant registers fails at the vault read, and "
        "the Session is told the tool is unavailable."
    )


@requires_the_cluster
def test_the_agent_searched_with_the_credential_the_tenant_brought(run: _Run) -> None:
    """The live arm came back with results rather than with Tavily's refusal.

    What this proves: the header this platform composed was one Tavily accepted. Tavily
    validates lazily -- `initialize` succeeds with any key at all -- so a 200 from the
    connection would have proven nothing, and the refusal only appears in the body of
    the call itself.

    What it does NOT prove, and the file does not pretend otherwise: nothing on this
    platform records a tool call, so the evidence is the agent's own report of what it
    received. The arm below is what makes that report load-bearing rather than merely
    plausible.
    """
    said = run.answer
    assert "LIVE:" in said, (
        "the agent never reported the live arm at all, so this case grades nothing; "
        f"its answer was {said[:800]!r}"
    )
    live = said.split("LIVE:", 1)[1].split("WRONG:", 1)[0]
    assert "Unauthorized" not in live and "401" not in live, (
        "Tavily refused the credential this platform composed for the live arm. The "
        "value reached the vault (see the case above), so what failed is between the "
        f"vault and the request. The agent reported: {live[:400]!r}"
    )
    assert "Paris" in live, (
        "the live arm reported neither Paris nor an authentication failure, so the "
        f"search did not return results. The agent reported: {live[:400]!r}"
    )


@requires_the_cluster
def test_a_wrong_credential_is_refused_by_the_far_end(run: _Run) -> None:
    """The negative control: the same path with a bad key comes back refused.

    This is what makes the arm above mean something. The two registrations differ in
    exactly one field -- which ref they name -- so if this arm ALSO succeeded, the
    credential would not be reaching Tavily and the live arm's results would be coming
    from somewhere this file cannot see. A platform that dropped the header entirely
    would fail both arms; a platform that sent one tenant's credential for every
    registration would pass both.

    Graded on Tavily's own words rather than on a status code, because the refusal
    arrives inside a 200: the key is checked at `tools/call`, never at the connection.
    """
    said = run.answer
    assert "WRONG:" in said, (
        "the agent never reported the second arm, so the control this case exists to "
        f"be did not run; its answer was {said[:800]!r}"
    )
    wrong = said.split("WRONG:", 1)[1]
    refused = any(
        marker in wrong
        for marker in ("Unauthorized", "401", "invalid API key", "FAILED")
    )
    assert refused, (
        "a credential belonging to nobody was accepted, which means the live arm's "
        "results did not come from the credential this platform composed. The agent "
        f"reported: {wrong[:400]!r}"
    )


@requires_the_cluster
def test_a_tool_bound_to_an_unscoped_dimension_never_reaches_the_far_end(
    run: _Run,
) -> None:
    """The clamp, graded where it cannot pass by accident.

    The other two arms are silent about it. Both name `topic` as their Scope dimension
    and this Session's Scope says `topic` is `general`, which is also the value the
    prompt tells the agent to send -- so the clamp writes the string that was already
    there, and both arms answer identically whether or not it ran. That is the whole
    reason this third arm exists.

    This registration names a dimension the Session's Scope does not carry, and it
    names the **working** credential against the same healthy server. So there is
    exactly one thing that can stop it: a platform without the clamp sends a good key
    to a server that answers, and this arm reports OK. It reporting a refusal is the
    clamp running.

    Graded on the refusal naming the Scope, not merely on failure. `FAILED` alone would
    also be produced by an outage, a timeout or a dropped header -- the three failures
    this case must not be satisfied by.
    """
    said = run.answer
    assert "SCOPED:" in said, (
        "the agent never reported the third arm, so the clamp was not exercised; its "
        f"answer was {said[:800]!r}"
    )
    scoped = said.split("SCOPED:", 1)[1]
    assert "OK" not in scoped.split("\n", 1)[0], (
        "a tool registered against a Scope dimension this Session does not carry was "
        "called successfully with a working credential. That is the clamp not running: "
        f"the agent reported {scoped[:400]!r}"
    )
    named_the_scope = any(
        marker in scoped
        for marker in ("Scope", "scope", _DIMENSION_NOBODY_SCOPED, "out_of_scope")
    )
    assert named_the_scope, (
        "the third arm failed, but not for the reason this case is about -- an outage, "
        "a timeout or a dropped header would look the same here. The refusal has to "
        f"name the Scope. The agent reported: {scoped[:400]!r}"
    )
