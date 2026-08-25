"""Compare what the cluster is running against the newest image built for it.

`tools/terraform_drift.py` answers the same question about the AWS account and there
was no equivalent for the workloads, so the cluster could run code nobody had deployed
and nothing said so. It happened: on 2026-08-23 `control-plane` was on
`sha256:7e0b3153…`, `tool-gateway` on `sha256:cd704014…`, and the newest image in all
three repositories was `sha256:a39d16c5…` -- two workloads behind, and behind each
other. Every measurement taken against that cluster was a measurement of code that is
not what the tree says, which is a footnote nobody knew to write.

This does not decide whether the newest image is the right one to be running -- a
deliberate rollback is a legitimate state and looks identical from here. It reports the
difference and leaves the judgement to a person, exactly as the terraform tool prints a
plan and refuses to grep it.

Exit codes, kept apart for the same reason the terraform tool keeps its three apart:

  0  every workload runs the newest image in its repository.
  2  at least one does not. A finding, and the normal one after a merge.
  3  no comparison was attempted -- absent credentials, an unreachable cluster, an
     empty repository. Kept off 2 so that "could not be compared" can never be read as
     "is up to date", which is the direction that costs something.

The workload list is imported from `deploy/platform.py` rather than repeated. A second
list here would be free to disagree, and the way it would disagree is by omitting a
workload somebody added -- reporting "all up to date" over one it never looked at.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "deploy"))


def _platform() -> ModuleType:
    """`deploy/platform.py`, loaded by path.

    By path rather than `import platform`, because that name belongs to the standard
    library and `sys.path` order would decide which one wins. Registered in
    `sys.modules` **before** it is executed, which is not optional: the module defines
    a frozen dataclass, and `dataclasses` resolves a class's module through
    `sys.modules[cls.__module__]` -- absent, that lookup returns None and the failure
    surfaces as `'NoneType' object has no attribute '__dict__'`, which says nothing
    about the real cause.
    """
    name = "map_platform_for_drift"
    spec = importlib.util.spec_from_file_location(
        name, _ROOT / "deploy" / "platform.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("deploy/platform.py could not be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _run(*argv: str) -> str:
    done = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    if done.returncode != 0:
        raise RuntimeError(f"{' '.join(argv[:3])}: {done.stderr.strip()[:300]}")
    return done.stdout


def _deployed_digest(namespace: str, component: str) -> str:
    image = _run(
        "kubectl",
        "get",
        "deploy",
        component,
        "-n",
        namespace,
        "-o",
        "jsonpath={.spec.template.spec.containers[0].image}",
    ).strip()
    if "@" not in image:
        # A mutable tag is its own finding: two pulls of one tag can be two byte sets,
        # so there is no digest to compare and nothing here can say what is running.
        return f"NOT PINNED BY DIGEST: {image}"
    return image.split("@", 1)[1]


def _newest_digest(repository: str) -> str:
    out = _run(
        "aws",
        "ecr",
        "describe-images",
        "--repository-name",
        repository,
        "--region",
        "us-east-1",
        "--output",
        "json",
    )
    details = json.loads(out)["imageDetails"]
    if not details:
        raise RuntimeError(f"{repository} holds no image")
    newest = max(details, key=lambda d: d["imagePushedAt"])
    return str(newest["imageDigest"])


def main() -> int:
    try:
        platform = _platform()
        # The namespace comes from the same function the applier uses, which reads it
        # out of cluster-bootstrap.yaml. A literal here would be a fourth copy of a
        # name three files already agree on.
        namespace = platform.namespace(_ROOT)
        rows = [
            (
                workload.component,
                _deployed_digest(namespace, workload.component),
                _newest_digest(workload.repository),
            )
            for workload in platform.WORKLOADS
        ]
    except Exception as failure:  # noqa: BLE001 -- every failure here means "not compared"
        print(f"no comparison was attempted: {failure}", file=sys.stderr)
        return 3

    behind = [r for r in rows if r[1] != r[2]]
    for component, deployed, newest in rows:
        mark = "  " if deployed == newest else "->"
        print(f"{mark} {component:<16} running {deployed[:23]}  newest {newest[:23]}")
    if behind:
        print(
            f"\n{len(behind)} of {len(rows)} workloads are not running the newest "
            "image built for them. That is normal right after a merge and wrong right "
            "after a deploy; this tool does not know which of the two this is, and a "
            "deliberate rollback looks the same from here.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
