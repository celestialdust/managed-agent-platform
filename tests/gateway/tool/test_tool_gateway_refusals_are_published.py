"""Every refusal the Tool Gateway can emit carries a code from the published set.

Tier 1 (local, no infrastructure). The sibling guard over `control/api/` globs exactly
one directory, so it grades nothing under `gateway/tool/` — and this slice adds a whole
new refusal-emitting surface. The check is structural rather than provoked for the same
reason the sibling's is: the codes worth catching are the ones nobody thought to
provoke.

Two shapes are refused here. A `{"code": ...}` dict literal is how an unversioned code
gets into a response body without ever passing through `ErrorCode` — ADR-013 makes the
published set the API version, so a string invented in a module is an addition to the
contract that no consumer can branch on. And a `raise HTTPException` is how a refusal
leaves as a status with a body nobody wrote down, bypassing `ErrorEnvelope` entirely.

The one deliberate exception is the pod-facing 401, which uses `{"error": ...}`: it is
seen by a Session pod rather than by a tenant, and putting a published code there would
commit the platform to a code no tenant can ever observe. It is not `{"code": ...}`, so
it is outside this walk by shape rather than by an exemption list.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from managed_agent.core.errors import ErrorCode

_PACKAGE = (
    Path(__file__).resolve().parents[3] / "src" / "managed_agent" / "gateway" / "tool"
)
_PUBLISHED = {code.value for code in ErrorCode}


def _modules() -> list[Path]:
    return sorted(path for path in _PACKAGE.glob("*.py") if path.name != "__init__.py")


def _trees() -> dict[Path, ast.Module]:
    return {path: ast.parse(path.read_text()) for path in _modules()}


def _error_code_attributes(tree: ast.Module) -> set[str]:
    """Every `ErrorCode.NAME` an module mentions, by attribute name."""
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "ErrorCode"
    }


def test_there_are_modules_and_refusals_here_to_check() -> None:
    """Guard the guard, twice.

    An empty glob, or a package that stopped naming `ErrorCode` at all, would satisfy
    every assertion below by having nothing to assert about — and this file has no other
    way to tell "every code is published" from "there are no codes".
    """
    assert _modules(), f"no modules found under {_PACKAGE}"

    named = set()
    for tree in _trees().values():
        named |= _error_code_attributes(tree)
    assert named, (
        "no module under gateway/tool/ names a member of ErrorCode, so the membership "
        "test below passes vacuously. Either the Gateway stopped refusing anything, or "
        "the shape it refuses with changed and this walk no longer recognises it."
    )


@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
def test_no_module_names_a_code_outside_the_published_set(path: Path) -> None:
    named = _error_code_attributes(ast.parse(path.read_text()))
    unpublished = {attr for attr in named if getattr(ErrorCode, attr, None) is None}

    assert unpublished == set(), (
        f"{path.name} names ErrorCode members that do not exist: {sorted(unpublished)}"
    )
    for attr in named:
        assert ErrorCode[attr].value in _PUBLISHED


@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
def test_no_module_builds_a_refusal_out_of_a_dict_literal(path: Path) -> None:
    """A `{"code": ...}` body is a code that never passed through the closed set."""
    offenders = [
        f"{path.name}:{value.lineno}"
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values, strict=True)
        if isinstance(key, ast.Constant) and key.value == "code"
    ]

    assert offenders == [], (
        f"{path.name} builds a refusal body by hand at {offenders}. Construct an "
        "ErrorEnvelope through error_map instead — its `code` field is typed as "
        "ErrorCode, so mypy refuses an unpublished code before this test would see it."
    )


@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
def test_no_module_refuses_with_a_bare_http_exception(path: Path) -> None:
    """A raw `HTTPException` carries a status and a body outside the published shape."""
    source = path.read_text()

    assert "HTTPException" not in source, (
        f"{path.name} names HTTPException. Every refusal this service emits goes out "
        "as an ErrorEnvelope, either inside a failed CallToolResult or inside an "
        "MCPError's `data` — a status code with an ad-hoc body is a second contract."
    )
