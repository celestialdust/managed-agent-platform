"""The capture ledger cannot hold a row that lies about what was captured.

Tier 1 (testcontainers, real PostgreSQL 17). Asserted with raw SQL rather than through
the store adapter, because every property here has to survive a writer that never loads
our code -- a psql session, a later slice's adapter, a migration written in a hurry.

Every classified result gets a row, not only the captured ones: a reviewer who finds no
Evidence for a call has to be able to see that the output was small, and an absence
cannot say so. The constraints are what make the row unable to lie -- a row claiming a
capture with no digest, or a digest with no object behind it, is refused by the store
rather than by the care of whoever writes it.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

_HEX = "a" * 64

_INSERT = sa.text(
    "INSERT INTO evidence_capture (session_id, call_id, tool_name, capture_point,"
    " byte_length, threshold_bytes, passed_through_bytes, hash_algorithm, hash_hex,"
    " object_key, truncated_at_runtime_cap)"
    " VALUES (:sid, :call_id, :tool, :point, :len, :threshold, :through, :algo, :hex,"
    " :key, :cut)"
).bindparams(sa.bindparam("sid", type_=sa.Uuid()))

_INSERT_WITHOUT_PASSED_THROUGH = sa.text(
    "INSERT INTO evidence_capture (session_id, call_id, tool_name, capture_point,"
    " byte_length, threshold_bytes, hash_algorithm, hash_hex, object_key,"
    " truncated_at_runtime_cap)"
    " VALUES (:sid, :call_id, :tool, :point, :len, :threshold, :algo, :hex, :key, :cut)"
).bindparams(sa.bindparam("sid", type_=sa.Uuid()))

_READ = (
    sa.text(
        "SELECT hash_hex, object_key, byte_length, capture_point"
        " FROM evidence_capture WHERE session_id = :sid AND call_id = :call_id"
    )
    .bindparams(sa.bindparam("sid", type_=sa.Uuid()))
    .columns(
        hash_hex=sa.Text(),
        object_key=sa.Text(),
        byte_length=sa.BigInteger(),
        capture_point=sa.Text(),
    )
)


def _row(**overrides: Any) -> dict[str, Any]:
    """A captured row that satisfies every constraint, for one field to be spoiled."""
    row: dict[str, Any] = {
        "sid": uuid.uuid4(),
        "call_id": uuid.uuid4().hex,
        "tool": "acme__search",
        "point": "tool-gateway",
        "len": 200_000,
        "threshold": 65_536,
        "through": 0,
        "algo": "sha256",
        "hex": _HEX,
        "key": f"evidence/x/sha256-{_HEX}",
        "cut": False,
    }
    row.update(overrides)
    return row


async def _insert(engine: AsyncEngine, **overrides: Any) -> dict[str, Any]:
    row = _row(**overrides)
    async with engine.begin() as conn:
        await conn.execute(_INSERT, row)
    return row


async def test_a_capture_and_an_inline_return_both_get_a_row(
    engine: AsyncEngine,
) -> None:
    """The inline row is the half a reviewer needs: no Evidence, and a reason why."""
    captured = await _insert(engine)
    inline = await _insert(
        engine, len=10, algo=None, hex=None, key=None, point="session-shim"
    )

    async with engine.connect() as conn:
        one = (await conn.execute(_READ, captured)).one()
        two = (await conn.execute(_READ, inline)).one()

    assert one.hash_hex == _HEX
    assert one.object_key == captured["key"]
    assert two.hash_hex is None
    assert two.object_key is None
    assert two.byte_length == 10
    assert two.capture_point == "session-shim"


async def test_output_at_or_above_the_threshold_cannot_be_recorded_without_a_hash(
    engine: AsyncEngine,
) -> None:
    """Otherwise a large result reads as though it were returned inline."""
    with pytest.raises(IntegrityError, match="hash_iff_at_threshold"):
        await _insert(engine, algo=None, hex=None, key=None)


async def test_output_below_the_threshold_cannot_carry_a_hash(
    engine: AsyncEngine,
) -> None:
    """A digest on a small result claims an object the capture path never wrote."""
    with pytest.raises(IntegrityError, match="hash_iff_at_threshold"):
        await _insert(engine, len=10)


async def test_a_row_recording_the_boundary_exactly_is_a_capture(
    engine: AsyncEngine,
) -> None:
    """At the threshold exactly, output is Evidence -- the same boundary the code uses,
    restated by the constraint so the two cannot drift."""
    await _insert(engine, len=65_536)
    with pytest.raises(IntegrityError, match="hash_iff_at_threshold"):
        await _insert(engine, len=65_535)


async def test_a_hash_with_no_object_behind_it_is_refused(
    engine: AsyncEngine,
) -> None:
    with pytest.raises(IntegrityError, match="reference_is_whole"):
        await _insert(engine, key=None)


async def test_a_hash_with_no_algorithm_naming_it_is_refused(
    engine: AsyncEngine,
) -> None:
    """A digest whose algorithm nobody recorded is a digest nobody can reproduce."""
    with pytest.raises(IntegrityError, match="reference_is_whole"):
        await _insert(engine, algo=None)


@pytest.mark.parametrize("bad", [_HEX.upper(), "a" * 63, "z" * 64])
async def test_a_hash_no_reader_could_reproduce_is_refused(
    engine: AsyncEngine, bad: str
) -> None:
    with pytest.raises(IntegrityError, match="hash_is_lower_hex"):
        await _insert(engine, hex=bad)


@pytest.mark.parametrize("point", ["shim", "gateway", "TOOL-GATEWAY", ""])
async def test_a_capture_point_outside_the_two_that_exist_is_refused(
    engine: AsyncEngine, point: str
) -> None:
    """The column is what tells a reviewer which guarantee a piece of Evidence carries,
    so a third spelling would be an assurance nobody defined."""
    with pytest.raises(IntegrityError, match="point_is_known"):
        await _insert(engine, point=point)


async def test_both_known_capture_points_are_accepted(engine: AsyncEngine) -> None:
    await _insert(engine, point="tool-gateway")
    await _insert(engine, point="session-shim")


@pytest.mark.parametrize("threshold", [0, -5])
async def test_a_threshold_no_output_could_fall_below_is_refused(
    engine: AsyncEngine, threshold: int
) -> None:
    with pytest.raises(IntegrityError, match="sizes_are_positive"):
        await _insert(engine, threshold=threshold)


async def test_a_negative_byte_length_is_refused(engine: AsyncEngine) -> None:
    """Written as an inline row, so nothing but the size constraint can refuse it: a
    negative length against a positive threshold would otherwise be caught by
    `hash_iff_at_threshold` and this test would pass with the size rule gone."""
    with pytest.raises(IntegrityError, match="sizes_are_positive"):
        await _insert(engine, len=-1, algo=None, hex=None, key=None)


async def test_a_negative_count_of_what_passed_through_is_refused(
    engine: AsyncEngine,
) -> None:
    with pytest.raises(IntegrityError, match="sizes_are_positive"):
        await _insert(engine, through=-1)


async def test_a_row_that_does_not_say_what_passed_through_is_refused(
    engine: AsyncEngine,
) -> None:
    """The column has no default, so saying nothing is not the same as saying zero.

    `byte_length` records only the part of a result this capture point weighs. A writer
    that leaves the rest unstated -- which a `DEFAULT 0` would let it do silently --
    produces a row asserting that the weighed part was the whole output, and that is the
    one answer a reviewer asking "was this call's output small?" must never be handed
    wrongly. The refusal has to come from the store, because the writers this ledger has
    to survive include ones that never load our code.
    """
    row = _row()
    del row["through"]

    with pytest.raises(IntegrityError, match="passed_through_bytes"):
        async with engine.begin() as conn:
            await conn.execute(_INSERT_WITHOUT_PASSED_THROUGH, row)


async def test_nothing_returned_inline_can_be_cut_at_a_cap_it_never_reached(
    engine: AsyncEngine,
) -> None:
    with pytest.raises(IntegrityError, match="truncation_implies_capture"):
        await _insert(engine, len=10, algo=None, hex=None, key=None, cut=True)


async def test_a_captured_row_may_record_that_the_runtime_had_already_cut_it(
    engine: AsyncEngine,
) -> None:
    await _insert(engine, point="session-shim", cut=True)


async def test_a_second_capture_of_one_call_is_refused_rather_than_a_second_row(
    engine: AsyncEngine,
) -> None:
    """Two rows for one call would disagree with each other and nothing would say
    which of them describes the bytes the model was given."""
    first = await _insert(engine)
    with pytest.raises(IntegrityError):
        await _insert(engine, sid=first["sid"], call_id=first["call_id"])


async def test_one_call_id_under_two_sessions_is_two_rows(engine: AsyncEngine) -> None:
    """The Session leads the key, so a call id is only unique within its Session."""
    first = await _insert(engine)
    await _insert(engine, call_id=first["call_id"])


async def test_a_written_row_cannot_be_updated(engine: AsyncEngine) -> None:
    """A captured payload is addressed by the hash of its own bytes, so a row whose
    hash could be rewritten would stop describing the object it points at. The refusal
    has to raise rather than silently ignore: a writer told the update succeeded would
    go on believing the value it wrote."""
    written = await _insert(engine)

    with pytest.raises(DBAPIError):
        async with engine.begin() as conn:
            await conn.execute(
                sa.text(
                    "UPDATE evidence_capture SET hash_hex = :hex"
                    " WHERE session_id = :sid AND call_id = :call_id"
                ).bindparams(sa.bindparam("sid", type_=sa.Uuid())),
                {"hex": "b" * 64, "sid": written["sid"], "call_id": written["call_id"]},
            )

    async with engine.connect() as conn:
        assert (await conn.execute(_READ, written)).one().hash_hex == _HEX
