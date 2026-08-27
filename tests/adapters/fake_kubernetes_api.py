"""A stand-in Kubernetes API server, spoken to by the real generated client.

Real in the half that has to be proven here and fake in the half a unit test cannot
reach. The transport is genuine: `kubernetes_asyncio`'s own `CoreV1Api` builds its
credential from a real kubeconfig on disk, opens a real HTTP connection, serializes the
manifest the adapter hands it, and deserializes what comes back into `V1Pod` and
`V1Secret`. What is canned is the cluster's behaviour -- a pod does not really schedule,
so the status a read returns is whatever the test set.

This exists because `ensure` and `remove` are the two methods with the consequences and
neither could be exercised at all. Every other test of this adapter is over a pure
function -- `_pod_for`, `_phase_of`, `_why_it_will_not_start` -- and the orchestration
that decides whether to create, whether to adopt and what to delete on the way out was
covered only by a live-cluster tier that skips unless `MAP_CLUSTER_TESTS=1`. A guard
that skips in the ordinary suite is a guard a mutation walks straight past.

What it cannot show: that the real API server admits these bodies, that a real kubelet
reaches the states scripted here, or anything about timing. Those are the live tier's,
and the live tier stays.

Not a `conftest.py`: mypy refuses a second module named `conftest` in a tree with no
`__init__.py`, and the repository has one at `tests/`. So this is a context manager a
test opens, the same shape as `tests/session_shim/fake_agent_runtime.py`.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

_NOT_FOUND: Final = 404
_CONFLICT: Final = 409


def rfc3339(seconds_ago: int) -> str:
    """A creation timestamp the generated client will deserialize, `seconds_ago` old.

    Written as UTC with a `Z` suffix because that is the only form the API server emits
    and the only one the client's own deserializer is exercised on in production. A test
    that wants an old pod says how old in seconds rather than composing a datetime, so
    the ages in this suite read as durations instead of as instants.
    """
    when = datetime.now(UTC) - timedelta(seconds=seconds_ago)
    return when.strftime("%Y-%m-%dT%H:%M:%SZ")


# A kubeconfig pointing at this server, and nothing in it is a credential: the fake
# never looks at the bearer, and the field is present only because the loader wants a
# user entry.
_KUBECONFIG: Final = """\
apiVersion: v1
kind: Config
current-context: fake
clusters:
  - name: fake
    cluster: {{server: "http://127.0.0.1:{port}"}}
contexts:
  - name: fake
    context: {{cluster: fake, user: fake}}
users:
  - name: fake
    user: {{token: not-a-credential}}
"""


def _status(code: int, reason: str, message: str) -> JSONResponse:
    """A Kubernetes `Status` body, which is what the generated client parses a failure
    as.

    Returned rather than raised so the shape is identical to the real server's: the
    client reads the HTTP code, and `ApiException.status` is what every branch in the
    adapter tests against.
    """
    return JSONResponse(
        {
            "kind": "Status",
            "apiVersion": "v1",
            "status": "Failure",
            "code": code,
            "reason": reason,
            "message": message,
        },
        status_code=code,
    )


@dataclass
class FakeCluster:
    """What the fake API server holds, and the dials a test turns.

    `pods` and `secrets` are the store, keyed by name within one namespace, and they are
    the measurement: a test asserts on what is left in them rather than on which calls
    were made, so a cleanup that deletes the pod and forgets a Secret fails.

    `status_of` is how a pod's fate is scripted. Keyed by pod name and consulted on
    every read, so a test can hand back a pod that is Running with both containers
    ready, or one stuck Pending, or one whose init container is in `ImagePullBackOff`.
    Absent, a created pod reads back Pending with no container statuses, which is what a
    pod looks like the instant after it is created.

    A 409 on a create is the real server's answer to an object that already exists, and
    it is reachable by seeding the store rather than by a dial -- which is how the
    already-existing-Secret case below is set up.
    """

    pods: dict[str, dict[str, Any]] = field(default_factory=dict)
    secrets: dict[str, dict[str, Any]] = field(default_factory=dict)
    status_of: dict[str, dict[str, Any]] = field(default_factory=dict)
    lingering: set[str] = field(default_factory=set)
    created: int = 0

    def finish_deletion(self, name: str) -> None:
        """Complete a teardown the way kubelet does when the grace period runs out.

        Only meaningful for a pod named in `lingering`; for anything else the delete
        already removed the object and there is nothing left to finish.
        """
        self.pods.pop(name, None)
        self.status_of.pop(name, None)

    async def read_pod(self, request: Request) -> JSONResponse:
        name = request.path_params["name"]
        pod = self.pods.get(name)
        if pod is None:
            return _status(_NOT_FOUND, "NotFound", f'pods "{name}" not found')
        return JSONResponse(
            {**pod, "status": self.status_of.get(name, {"phase": "Pending"})}
        )

    async def create_pod(self, request: Request) -> JSONResponse:
        body: dict[str, Any] = await request.json()
        name = body["metadata"]["name"]
        if name in self.pods:
            return _status(_CONFLICT, "AlreadyExists", f'pods "{name}" already exists')
        # A fresh uid per creation, not one derived from the name. A real API server
        # never reissues one, and uid identity is the whole mechanism behind owner
        # references: a Secret owned by uid A is collected when uid A goes, and survives
        # a different object that merely reuses A's name. Deriving it from the name made
        # a replaced pod indistinguishable from the pod it replaced, which is exactly
        # the distinction the garbage-collection cases turn on.
        self.created += 1
        body["metadata"]["uid"] = f"uid-{self.created}-of-{name}"
        # Stamped for the same reason `uid` is: the API server sets both at admission,
        # so a caller that reads either off a pod it did not create would otherwise be
        # reading a field this fake had silently left out. A test that needs an *old*
        # pod seeds `pods` directly with an earlier stamp, which is the way the
        # already-exists case above is set up too.
        body["metadata"]["creationTimestamp"] = rfc3339(0)
        self.pods[name] = body
        return JSONResponse(body, status_code=201)

    async def list_pods(self, request: Request) -> JSONResponse:
        """The pods of one namespace, narrowed by an existence label selector.

        The selector is honoured rather than ignored, because the adapter passes one and
        a fake that returned everything could not tell a narrowed read from a read of
        the whole namespace -- which is the one thing the selector is there to do. Only
        the bare-key form is understood, which is the only form the adapter sends;
        anything else raises here rather than being read as "match everything", so a
        selector this fake cannot model fails loudly instead of widening the answer.
        """
        selector = request.query_params.get("labelSelector")
        if selector is not None and ("=" in selector or "," in selector):
            raise AssertionError(
                f"this fake models only a bare key selector: {selector}"
            )
        items = [
            {**pod, "status": self.status_of.get(name, {"phase": "Pending"})}
            for name, pod in self.pods.items()
            if selector is None or selector in (pod["metadata"].get("labels") or {})
        ]
        return JSONResponse(
            {"apiVersion": "v1", "kind": "PodList", "metadata": {}, "items": items}
        )

    async def delete_pod(self, request: Request) -> JSONResponse:
        """Delete, in either of the two shapes a real API server answers with.

        A pod with nothing running in it -- Pending, unscheduled, already terminal --
        really is removed by the time the call returns, and that is the default here.
        A pod whose containers are up is not: the API server stamps
        `deletionTimestamp`, answers 200, and the object stays addressable for its whole
        grace period while kubelet stops the containers. Naming a pod in `lingering`
        asks for that second shape, and `finish_deletion` is kubelet arriving.

        Both are real, and which one a pod gets depends on its state rather than on a
        preference -- so a test that needs the lingering shape has to say so. This
        started out modelling only the first, and the gap cost a live defect: a Turn
        arriving inside the grace window read the terminating pod as GONE and was
        refused, which no offline test could reproduce because deletion here had no
        duration at all.
        """
        name = request.path_params["name"]
        if name in self.lingering:
            pod = self.pods.get(name)
            if pod is None:
                return _status(_NOT_FOUND, "NotFound", f'pods "{name}" not found')
            pod["metadata"].setdefault("deletionTimestamp", rfc3339(0))
            return JSONResponse(pod)
        pod = self.pods.pop(name, None)
        if pod is None:
            return _status(_NOT_FOUND, "NotFound", f'pods "{name}" not found')
        # The scripted status goes with the object it describes. It is keyed by name,
        # and a name is reusable -- so leaving it behind would have a freshly created
        # pod read back the phase of the pod it replaced, which no real cluster does.
        self.status_of.pop(name, None)
        return JSONResponse(pod)

    async def create_secret(self, request: Request) -> JSONResponse:
        body: dict[str, Any] = await request.json()
        name = body["metadata"]["name"]
        if name in self.secrets:
            return _status(
                _CONFLICT, "AlreadyExists", f'secrets "{name}" already exists'
            )
        self.secrets[name] = body
        return JSONResponse(body, status_code=201)

    async def patch_secret(self, request: Request) -> JSONResponse:
        name = request.path_params["name"]
        secret = self.secrets.get(name)
        if secret is None:
            return _status(_NOT_FOUND, "NotFound", f'secrets "{name}" not found')
        patch: dict[str, Any] = await request.json()
        secret["metadata"].update(patch.get("metadata", {}))
        return JSONResponse(secret)

    async def delete_secret(self, request: Request) -> JSONResponse:
        name = request.path_params["name"]
        secret = self.secrets.pop(name, None)
        if secret is None:
            return _status(_NOT_FOUND, "NotFound", f'secrets "{name}" not found')
        return JSONResponse(secret)

    def app(self) -> Starlette:
        pods = "/api/v1/namespaces/{namespace}/pods"
        secrets = "/api/v1/namespaces/{namespace}/secrets"
        return Starlette(
            routes=[
                Route(f"{pods}/{{name}}", self.read_pod, methods=["GET"]),
                Route(pods, self.list_pods, methods=["GET"]),
                Route(pods, self.create_pod, methods=["POST"]),
                Route(f"{pods}/{{name}}", self.delete_pod, methods=["DELETE"]),
                Route(secrets, self.create_secret, methods=["POST"]),
                Route(f"{secrets}/{{name}}", self.patch_secret, methods=["PATCH"]),
                Route(f"{secrets}/{{name}}", self.delete_secret, methods=["DELETE"]),
            ]
        )


def _container_status(
    name: str, *, ready: bool, state: dict[str, Any]
) -> dict[str, Any]:
    """One container status, carrying the fields the generated model insists on.

    `image`, `imageID` and `restartCount` are not optional in `V1ContainerStatus` --
    the client raises `ValueError` while deserializing without them -- so the fake has
    to send them even though nothing in the adapter reads them. Which is itself the
    point of speaking to the real client: a hand-built stub would have accepted a body
    the API server's own schema does not.
    """
    return {
        "name": name,
        "ready": ready,
        "state": state,
        "image": "registry.invalid/map/session-shim@sha256:" + "ab" * 32,
        "imageID": "registry.invalid/map/session-shim@sha256:" + "ab" * 32,
        "restartCount": 0,
    }


def running(*containers: str) -> dict[str, Any]:
    """A pod status that `_phase_of` reads as RUNNING: phase Running, every one
    ready."""
    return {
        "phase": "Running",
        "containerStatuses": [
            _container_status(name, ready=True, state={"running": {}})
            for name in containers
        ],
    }


def stuck(reason: str, *, container: str = "seed-runtime-home") -> dict[str, Any]:
    """A pod status `_why_it_will_not_start` refuses: an init container that cannot
    pull.

    The init container's list and not the main one, because that is where the failure
    of a Session pod actually lands -- every container carries the same image, so the
    init container is always the first to try to pull it.
    """
    return {
        "phase": "Pending",
        "initContainerStatuses": [
            _container_status(
                container,
                ready=False,
                state={"waiting": {"reason": reason, "message": "no such manifest"}},
            )
        ],
    }


@asynccontextmanager
async def fake_kubernetes_api(
    cluster: FakeCluster, kubeconfig: Path
) -> AsyncIterator[FakeCluster]:
    """Serve `cluster` on a loopback port and write a kubeconfig naming it.

    Port 0, so nothing collides with a port a parallel test or the developer's own
    machine is using; the bound port is read back off the server after startup and only
    then written into the kubeconfig, which is why the file is written here rather than
    by the caller.

    Torn down on every exit including a failing assertion, because the server holds a
    listening socket and a task, and a suite that leaked one per test would run out of
    descriptors long before it ran out of tests.
    """
    config = uvicorn.Config(
        cluster.app(), host="127.0.0.1", port=0, log_level="warning"
    )
    server = uvicorn.Server(config)
    serving = asyncio.create_task(server.serve())
    try:
        while not server.started:
            await asyncio.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        kubeconfig.write_text(_KUBECONFIG.format(port=port))
        yield cluster
    finally:
        server.should_exit = True
        await serving
