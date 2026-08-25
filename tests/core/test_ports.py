"""The ports are satisfiable by something that is not infrastructure, and know of none.

Tier 1 (local, no infrastructure). Two properties are graded here. First, an in-memory
fake satisfies each port, which is what makes the ports abstractions rather than a
description of the Postgres adapter. Second, `core/ports.py` names nothing outside the
standard library and `core` — asserted by parsing its imports rather than by importing
it, so the assertion still holds for a module that only imports infrastructure
conditionally.
"""

import ast
import sys
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from uuid import uuid4

from managed_agent.core.ids import FIRST_SEQ, Seq, SessionId
from managed_agent.core.ports import (
    Clock,
    CredentialVault,
    EventLogAppend,
    EventLogRange,
    EventRecord,
    ObjectStore,
    SequenceRace,
    SessionNotVisible,
)

_PORTS_SOURCE = Path(EventLogAppend.__module__.replace(".", "/")).with_suffix(".py")


class FakeRecord:
    def __init__(self, session_id: SessionId, seq: Seq) -> None:
        self.session_id = session_id
        self.seq = seq
        self.type = "test.event"
        self.payload: dict[str, object] = {}


class FakeAppend:
    def __init__(self) -> None:
        self._next = FIRST_SEQ

    async def append(
        self, session_id: SessionId, type_: str, payload: dict[str, object]
    ) -> Seq:
        seq = self._next
        self._next += 1
        return seq


class FakeRange:
    async def read(
        self, session_id: SessionId, start: Seq, end: Seq
    ) -> Sequence[EventRecord]:
        return []

    async def follow(
        self, session_id: SessionId, after: Seq
    ) -> AsyncIterator[EventRecord]:
        if False:  # pragma: no cover - an empty async generator
            yield FakeRecord(session_id, after)

    async def retained_floor(self, session_id: SessionId) -> Seq:
        return FIRST_SEQ


class FakeObjectStore:
    async def put(self, key: str, body: bytes) -> str:
        return key

    async def get(self, key: str) -> bytes:
        return b""

    async def delete_prefix(self, prefix: str) -> int:
        return 0


class FakeVault:
    async def fetch(self, name: str) -> str:
        return "not-a-secret"


class FakeClock:
    def now_epoch_ms(self) -> int:
        return 0


def test_behaviour_ports_are_satisfied_by_in_memory_fakes() -> None:
    assert issubclass(FakeAppend, EventLogAppend)
    assert issubclass(FakeRange, EventLogRange)
    assert issubclass(FakeObjectStore, ObjectStore)
    assert issubclass(FakeVault, CredentialVault)
    assert issubclass(FakeClock, Clock)


def test_event_record_is_satisfied_by_a_plain_object() -> None:
    record = FakeRecord(SessionId(uuid4()), FIRST_SEQ)
    assert isinstance(record, EventRecord)


def test_sequence_race_is_an_exception_callers_can_catch() -> None:
    assert issubclass(SequenceRace, Exception)


def test_session_not_visible_is_an_exception_callers_can_catch() -> None:
    """`Exception`, not `BaseException`, and the difference is not academic.

    A route catches this by name and would keep working either way, so the mutation
    that matters is invisible from the call site: derived from `BaseException` it slips
    past every `except Exception` between the raise and the client -- the framework's
    own handler included -- and a tenant asking for a Session that is not theirs gets
    an unhandled crash instead of a refusal. Asserted here beside `SequenceRace` for
    the same reason that one is.
    """
    assert issubclass(SessionNotVisible, Exception)


def test_ports_imports_nothing_outside_core_and_the_stdlib() -> None:
    source = Path(__file__).parents[2] / "src" / _PORTS_SOURCE
    tree = ast.parse(source.read_text())
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            roots.add(node.module.split(".")[0])

    assert roots, "parsed no imports at all — the path is probably wrong"
    outside = {
        root
        for root in roots
        if root != "managed_agent" and root not in sys.stdlib_module_names
    }
    assert outside == set(), f"core/ports.py reaches infrastructure: {sorted(outside)}"


def test_ports_names_no_adapter_module() -> None:
    source = Path(__file__).parents[2] / "src" / _PORTS_SOURCE
    tree = ast.parse(source.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            assert not node.module.startswith("managed_agent.adapters")
