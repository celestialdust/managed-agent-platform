"""The published error set, its status mapping, and the envelope refusals travel in.

Tier 1 (local, no infrastructure). Realizes MAP-A51 (every error code a caller sees is
from the published closed set, so the caller can branch on it exhaustively).

The mypy test here is the unusual one and it earns its runtime. `_status` is a `match`
rather than a dict *because* the docstring claims mypy --strict fails the build when a
member is added with no arm. That is a claim about a tool, and a claim about a tool is
worth exactly what a run of the tool says it is worth — this repository has already
shipped a docstring asserting a line was load-bearing that turned out to be inert. So
the claim is executed: a member is added to a copy of the module and mypy is asked.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from managed_agent.core import errors
from managed_agent.core.errors import (
    PUBLIC_TYPE_FOR,
    STATUS_FOR,
    ErrorCode,
    ErrorEnvelope,
    PublicErrorType,
)

_LOAD_SHED_STATUS = 529
"""What a refusal for capacity answers with, since 2026-08-24."""

_QUOTA_STATUS = 429
"""Held free for a rate limiter that does not exist, so nothing may claim it yet."""

_STATUSES_FOR_CLASS: dict[PublicErrorType, frozenset[int]] = {
    PublicErrorType.NOT_FOUND: frozenset({404}),
    PublicErrorType.PERMISSION: frozenset({403}),
    PublicErrorType.AUTHENTICATION: frozenset({401}),
    PublicErrorType.OVERLOADED: frozenset({_LOAD_SHED_STATUS}),
    PublicErrorType.API: frozenset({500, 502, 504}),
    PublicErrorType.INVALID_REQUEST: frozenset({400, 402, 405, 409, 410}),
    PublicErrorType.RATE_LIMIT: frozenset(),
    PublicErrorType.REQUEST_TOO_LARGE: frozenset(),
}
"""Which statuses each published class may appear at, written out here on purpose.

`_status` and `_public_type` are two `match` statements over one enum, and mypy holds
each to being exhaustive without holding the two to agreeing with each other: a member
added to both, at 404 in one arm and `invalid_request_error` in the other, compiles.
This table is the third opinion that makes the disagreement fail — kept as data rather
than derived from either map, because a table derived from one of them could only ever
agree with it.

The two empty entries are claims, not gaps. Nothing here answers 429 or 413, so nothing
may be published as the class that promises one.

What this costs as the set grows, which is the reason it is a table of statuses and not
a list of members: a new code at a status already listed for its class costs nothing
here, and the set can grow indefinitely that way. A new code at a status no row carries
costs one row — and that edit is the decision being recorded, because a status the
published table has never carried is a new promise to every consumer, not a detail.
"""


def test_every_code_has_a_status() -> None:
    """No member can be reached without one, because the map is built from the enum."""
    assert set(STATUS_FOR) == set(ErrorCode)
    assert len(STATUS_FOR) == len(ErrorCode)


def test_the_status_map_cannot_be_mutated() -> None:
    """A caller re-pointing a code at another status would fork the contract in process.

    `MappingProxyType` is what makes the module-level map safe to hand out; asserted
    rather than assumed, because swapping it for a plain dict would change nothing
    visible until something wrote to it.
    """
    with pytest.raises(TypeError):
        STATUS_FOR[ErrorCode.INTERNAL] = 418  # type: ignore[index]


def test_every_code_has_a_public_class() -> None:
    """No member can be reached without one, because the map is built from the enum.

    The class is what a client generated against another surface's documentation
    branches on, so a code carrying none would be a refusal that client can only read
    as a transport fault — the one reading that tells it to stop retrying.
    """
    assert set(PUBLIC_TYPE_FOR) == set(ErrorCode)
    assert len(PUBLIC_TYPE_FOR) == len(ErrorCode)


def test_the_public_class_map_cannot_be_mutated() -> None:
    """Re-pointing a code at another class in process would fork the contract too.

    Asserted for the same reason `STATUS_FOR`'s immutability is: swapping the proxy for
    a plain dict changes nothing visible until something writes to it, and consumers
    already holding the published pairing would never see the write.
    """
    other_class = PublicErrorType.NOT_FOUND

    with pytest.raises(TypeError):
        PUBLIC_TYPE_FOR[ErrorCode.INTERNAL] = other_class  # type: ignore[index]


def test_an_expired_range_is_gone_rather_than_never_here() -> None:
    """410 and not 404, which is the entire reason this code exists separately.

    Those events were written and have since been swept. A 404 would say the same thing
    as an id that never existed, which is the collapse the code is defined to prevent.
    """
    assert STATUS_FOR[ErrorCode.EVENT_RANGE_EXPIRED] == 410
    assert STATUS_FOR[ErrorCode.SESSION_NOT_FOUND] == 404


def test_no_code_maps_to_a_success_status() -> None:
    """Every member of this set is a refusal, so none of them may read as an answer."""
    not_refusals = {code: status for code, status in STATUS_FOR.items() if status < 400}

    assert not_refusals == {}, (
        f"these codes carry a non-error status, so a caller branching on the status "
        f"would treat a refusal as an answer: {not_refusals}"
    )


def test_exactly_one_code_is_a_refusal_for_load() -> None:
    """Exactly one code means "we have no capacity", and it says so at 529.

    Exactly one, because two ways to say "come back later" would leave a caller
    choosing which to retry on. The status this reads moved from 429 to 529 on
    2026-08-24 and the invariant did not move with it: a load refusal is still the only
    member of the set that means the caller did nothing wrong, and still the only one
    that may mean it.

    Both mappings are read rather than just the status. A code handed a load-shedding
    status by one arm and a caller-fault class by the other would tell an SDK to stop
    retrying a fault that was ours and temporary, and no exhaustiveness check catches
    two arms that are each complete and disagree.
    """
    for_load = [
        code for code, status in STATUS_FOR.items() if status == _LOAD_SHED_STATUS
    ]
    published_as_load = [
        code
        for code, public in PUBLIC_TYPE_FOR.items()
        if public is PublicErrorType.OVERLOADED
    ]

    assert for_load == [ErrorCode.OVERLOADED], (
        f"expected exactly one load refusal in the closed set; found {for_load}"
    )
    assert published_as_load == for_load, (
        f"the code answering {_LOAD_SHED_STATUS} and the code published as "
        f"{PublicErrorType.OVERLOADED.value} have to be the same one: the status map "
        f"says {for_load} and the class map says {published_as_load}"
    )
    assert ErrorCode.OVERLOADED.value == "platform.overloaded"


def test_no_code_claims_a_quota_this_platform_does_not_publish() -> None:
    """429 and `rate_limit_error` stay unclaimed, which is what makes 529 readable.

    A caller decides whether and when to retry from these two, and they say different
    things: 429 says this caller asked too often and a quota governs when to come back,
    529 says the service has no capacity and the caller did nothing wrong. This
    platform publishes no per-caller quota, so any 429 it emitted today would be the
    second thing wearing the first thing's status — and a caller would sit out a quota
    window that never ticks, because there is none.

    When a rate limiter arrives it brings a member at 429 published as
    `PublicErrorType.RATE_LIMIT`, and this test fails until it is changed in that same
    commit. Failing then is the point: under ADR-013 the addition is a version event,
    and this is what stops it happening as a side effect of building the limiter.
    """
    at_quota_status = [
        code for code, status in STATUS_FOR.items() if status == _QUOTA_STATUS
    ]
    published_as_quota = [
        code
        for code, public in PUBLIC_TYPE_FOR.items()
        if public is PublicErrorType.RATE_LIMIT
    ]

    assert at_quota_status == [], (
        f"these codes answer {_QUOTA_STATUS}, which promises a quota window this "
        f"platform does not publish: {at_quota_status}"
    )
    assert published_as_quota == [], (
        f"these codes are published as {PublicErrorType.RATE_LIMIT.value} while no "
        f"rate limiter exists to have refused them: {published_as_quota}"
    )


def test_a_codes_status_and_its_public_class_cannot_disagree() -> None:
    """The two mappings over this enum are held to each other, not just to the enum.

    mypy makes each `match` exhaustive and has nothing to say about whether they agree.
    So a member added to both arms — 404 in `_status`, `invalid_request_error` in
    `_public_type` — type-checks, and publishes a body telling a client to fix its
    request about a resource that does not exist.

    Checked in both directions, and the second one is the half that earns its keep:
    "every not-found code answers 404" would still pass with a sixth 404 code
    classified as something else. What fails is the two sets not being the same set.
    """
    assert set(_STATUSES_FOR_CLASS) == set(PublicErrorType), (
        "a published class was added or removed without a row in _STATUSES_FOR_CLASS, "
        "so the class it names is not held to any status: "
        f"{sorted(set(PublicErrorType) ^ set(_STATUSES_FOR_CLASS))}"
    )

    for public, statuses in _STATUSES_FOR_CLASS.items():
        by_class = {code for code, found in PUBLIC_TYPE_FOR.items() if found is public}
        by_status = {code for code, status in STATUS_FOR.items() if status in statuses}

        assert by_class == by_status, (
            f"{public.value} is published for {sorted(by_class)} but "
            f"{sorted(statuses)} is carried by {sorted(by_status)}. A code whose "
            "status and published class disagree is a refusal a client cannot act on "
            "coherently: fix the arm that is wrong, or, if the pairing is deliberate, "
            "add the status to this class's row in _STATUSES_FOR_CLASS."
        )


def test_a_code_renders_as_its_published_dotted_string() -> None:
    """What goes on the wire is the dotted string, not the Python member name.

    A `StrEnum` serialising as `ErrorCode.SESSION_NOT_FOUND` would still branch
    correctly inside this process and be unusable to every consumer.
    """
    envelope = ErrorEnvelope(code=ErrorCode.SESSION_NOT_FOUND, message="nope")

    assert envelope.model_dump(mode="json")["code"] == "session.not_found"


def test_every_code_is_a_lowercase_dotted_pair() -> None:
    """One published shape, so a consumer can parse a family out of a code.

    Checked over the whole set rather than spot-checked: the set is written once and
    grown rarely, and a member added in another shape would be a second convention that
    nothing else notices.
    """
    malformed = [
        code.value
        for code in ErrorCode
        if not code.value.islower() or code.value.count(".") != 1
    ]

    assert malformed == [], f"codes not in `family.reason` form: {malformed}"


def test_the_envelope_carries_the_code_the_message_and_typed_detail() -> None:
    envelope = ErrorEnvelope(
        code=ErrorCode.EVENT_RANGE_EXPIRED,
        message="the requested range is no longer retained",
        detail={"from_seq": 1, "retained_floor": 3},
    )

    assert envelope.model_dump(mode="json") == {
        "code": "event_log.range_expired",
        "message": "the requested range is no longer retained",
        "detail": {"from_seq": 1, "retained_floor": 3},
    }


def test_the_envelope_refuses_an_unknown_field() -> None:
    """`extra="forbid"`, so a fact a consumer needs cannot be smuggled in unnamed.

    Anything actionable belongs in `detail` under a name. A top-level field invented at
    one call site is a contract only that call site knows about.
    """
    with pytest.raises(ValidationError):
        ErrorEnvelope(
            code=ErrorCode.INTERNAL,
            message="boom",
            retry_after=30,  # type: ignore[call-arg]
        )


def test_the_envelope_refuses_an_empty_message() -> None:
    """An empty sentence is worse than a missing one: it reads as a rendered message."""
    with pytest.raises(ValidationError):
        ErrorEnvelope(code=ErrorCode.INTERNAL, message="")


def test_the_envelope_refuses_a_code_outside_the_set() -> None:
    """The closure is enforced by the type, which is what makes it a closed set.

    Without this the enum would be a convention: a route could put any string in the
    field and every consumer branching exhaustively would be wrong.
    """
    with pytest.raises(ValidationError):
        ErrorEnvelope(code="runtime.codex_stream_disconnected", message="leaked")  # type: ignore[arg-type]


def test_the_envelope_is_frozen() -> None:
    """A refusal already handed to a caller must not be editable behind their back."""
    envelope = ErrorEnvelope(code=ErrorCode.INTERNAL, message="boom")

    with pytest.raises(ValidationError):
        envelope.message = "something else"


def _variant(source: str, target: Path, extra_member: str) -> Path:
    """A copy of `errors.py` with `extra_member` appended to the enum, and no arm."""
    anchor = '    INTERNAL = "platform.internal"\n'
    assert anchor in source, "the enum's last member moved; this probe needs updating"
    target.write_text(source.replace(anchor, anchor + extra_member))
    return target


def _mypy(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "mypy", "--strict", "--no-incremental", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_a_member_added_with_no_status_arm_fails_the_type_check(
    tmp_path: Path,
) -> None:
    """The claim in `_status`'s docstring, executed rather than believed.

    `_status` is a `match` with an `assert_never` tail instead of a dict literal, and
    the only thing that buys is a build failure when a member is added and its status
    is not. If that is not true the shape is pure cost, so it is measured.

    An unmodified copy is checked first, and that control is the half that matters. A
    temp file mypy could not analyse at all — a bad import path, a missing stub — would
    fail for reasons having nothing to do with exhaustiveness, and this test would pass
    while measuring nothing.
    """
    source = Path(errors.__file__).read_text()

    clean = _mypy(_variant(source, tmp_path / "unmodified.py", ""))
    assert clean.returncode == 0, (
        "an unmodified copy of errors.py does not type-check on its own, so a failure "
        f"on the mutated copy would prove nothing:\n{clean.stdout}{clean.stderr}"
    )

    added = _mypy(
        _variant(source, tmp_path / "extra.py", '    NEW_REFUSAL = "new.refusal"\n')
    )

    assert added.returncode != 0, (
        "a member was added to ErrorCode with no arm in `_status` and mypy --strict "
        "accepted it. The match/assert_never shape is then buying nothing, and a code "
        "can reach a caller with no committed status:\n" + added.stdout
    )
    assert added.stdout.count("assert_never") >= 2, (
        "mypy failed, but not on both exhaustiveness checks this shape exists for. "
        "`_status` and `_public_type` are each a match over ErrorCode with an "
        "`assert_never` tail, so one added member has to be rejected twice — a single "
        "complaint means one of the two accepted a member it has no arm for, and a "
        "code would reach a caller under a defaulted status or a defaulted class:\n"
        + added.stdout
    )
