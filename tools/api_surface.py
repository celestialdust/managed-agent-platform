"""Print the REST surface the control plane actually serves, one operation per line.

Each line is a method and a path, sorted, in the form `POST /v1/sessions`. The list is
read out of the OpenAPI document the app publishes rather than by scanning the route
modules for decorators, because the document is what a client is generated from: a path
that appears here is one a caller can reach, and a decorator that never made it into an
included router is not.

Building the app needs a `Platform`, and this passes `None` for it. Every route reads
what it needs off `app.state.platform` when a request arrives, so nothing is read off it
while the routers are being attached -- which is what makes the surface printable with
no database, no cluster and no credentials.

`tests/test_readme_lists_every_endpoint.py` compares this output against the table in
README.md, so an endpoint added without documenting it fails the suite.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from managed_agent.control.api.app import create_app  # noqa: E402

_VERBS = ("get", "post", "put", "patch", "delete")


def operations() -> tuple[tuple[str, str], ...]:
    """Every (method, path) the app publishes, sorted by path then method."""
    document = create_app(cast(Any, None)).openapi()
    found = [
        (verb.upper(), path)
        for path, methods in document["paths"].items()
        for verb in methods
        if verb.lower() in _VERBS
    ]
    return tuple(sorted(found, key=lambda row: (row[1], row[0])))


def main() -> int:
    for method, path in operations():
        print(f"{method}\t{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
