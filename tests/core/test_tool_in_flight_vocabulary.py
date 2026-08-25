"""The tool family's three in-flight notices are published, and the split is safe.

Tier 1 (local, no infrastructure). Four properties, and each one is about the decision
to put these three names in their own module rather than in the module that will hold
the family's terminal event.

The names are published, so the Tool Gateway can be handed them instead of holding its
own copies. They carry the same family label the terminal event will carry, so a tenant
reading the stream sees one family whatever the file layout is. A second declaration of
any of them is refused at import, which is what makes two modules under one family label
safe rather than merely tidy. And the registry needed no edit to admit the file, which
is the property that lets two slices add to one family without writing one file.
"""

from __future__ import annotations

import pkgutil
from importlib import import_module
from pathlib import Path

import pytest

from managed_agent.core import vocabulary
from managed_agent.core.vocabulary import tool_in_flight

_THREE = (
    tool_in_flight.TOOL_PROGRESS,
    tool_in_flight.TOOL_ELICITATION_REQUESTED,
    tool_in_flight.TOOL_ELICITATION_ANSWERED,
)


def test_all_three_names_are_published() -> None:
    """Published under the names a tenant reads, not just assigned to constants.

    The spelling is asserted too: these three strings are what the Event Log carries
    and what a consumer branches on, so renaming a constant is free and renaming the
    published string is not.
    """
    assert _THREE == (
        "tool.progress",
        "tool.elicitation_requested",
        "tool.elicitation_answered",
    )
    for name in _THREE:
        assert vocabulary.is_published(name), name


def test_they_share_the_family_label_the_terminal_event_will_use() -> None:
    """One prefix and one family label across two modules.

    `tool.py` is another slice's file and will declare the family's terminal event under
    the same label. A different label here would split one family in two for a reader
    grouping by it, which is the only thing the label is for.
    """
    assert tool_in_flight.FAMILY == "tool"
    for name in _THREE:
        assert vocabulary.PUBLISHED[name] == "tool"
        assert name.startswith("tool.")


@pytest.mark.parametrize("name", _THREE)
def test_a_second_declaration_of_one_of_these_names_is_refused(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercised against a copy of the registry, so the real published set is untouched.

    This is what makes the two-module split safe. If the module holding the terminal
    event ever declares one of these three, the package fails to import loudly, at
    start-up, in every process -- instead of two spellings of one event coexisting with
    one of them unreachable.

    The seal has to come off to reach the duplicate check at all: it runs during
    discovery, and discovery has already finished by the time any test can call in.
    """
    monkeypatch.setattr(vocabulary, "_types", dict(vocabulary.PUBLISHED))
    monkeypatch.setattr(vocabulary, "_sealed", False)

    with pytest.raises(ValueError, match="duplicate"):
        vocabulary.declare(name, "tool")


def test_the_registry_needed_no_edit_to_admit_this_file() -> None:
    """Discovery found the module; the package does not name it.

    The registry's own docstring says adding a family is a new file and never an edit to
    a switch statement there, and that this is what keeps two slices from writing one
    file. This asserts the claim rather than trusting it: the module is in the set
    `pkgutil` walks, and its name appears nowhere in the package's own source.
    """
    walked = {found.name for found in pkgutil.iter_modules(vocabulary.__path__)}
    assert "tool_in_flight" in walked

    package = Path(str(vocabulary.__file__)).read_text()
    assert "tool_in_flight" not in package

    reimported = import_module(f"{vocabulary.__name__}.tool_in_flight")
    assert reimported.TOOL_PROGRESS == "tool.progress"
