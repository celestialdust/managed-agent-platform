"""The refusal a retired agent version answers with, and the status it carries.

A file of its own rather than an addition to `test_errors.py`, which grades the closed
set as a whole. What is asserted here is one member's published spelling and one status,
both of which a consumer branches on.

409 rather than 404 or 422, and neither of the other two is a near miss. 404 says the
version was never here, which erases the difference the 404/410 split was built to
preserve and tells a caller to go looking for a typo. 422 says the submitted body is
malformed, when the body is fine and the platform's own state is what refuses. 409 says
what actually happened: the request is well-formed and conflicts with the current state
of the thing it names.
"""

from managed_agent.core.errors import STATUS_FOR, ErrorCode


def test_a_retired_version_has_its_own_published_code() -> None:
    assert ErrorCode.DEFINITION_VERSION_ARCHIVED.value == "definition.version_archived"


def test_a_retired_version_is_a_conflict_rather_than_a_missing_thing() -> None:
    assert STATUS_FOR[ErrorCode.DEFINITION_VERSION_ARCHIVED] == 409
    assert STATUS_FOR[ErrorCode.DEFINITION_NOT_FOUND] == 404
