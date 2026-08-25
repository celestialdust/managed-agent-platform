"""A tool server that never came up is published, and only when it stays down.

Tier 1 (local, no infrastructure). Three properties, each about a decision this module
makes rather than about the registry mechanics `test_tool_in_flight_vocabulary.py`
already covers for the family's other split.

The name is published under the spelling a tenant branches on. It carries the family
label its two sibling modules carry, so a reader grouping by family sees one `tool`
family whatever the file layout is. And it is the family's third module, admitted by
discovery with no edit to the registry -- which is the property that lets a slice add
to a family without touching a file another slice owns.
"""

from __future__ import annotations

import pkgutil
from pathlib import Path

from managed_agent.core import vocabulary
from managed_agent.core.vocabulary import tool_call, tool_in_flight, tool_server


def test_the_name_is_published_under_the_spelling_a_tenant_reads() -> None:
    """The string is asserted, not just the constant.

    Renaming the constant is free and renaming the published string is not: the string
    is what the Event Log carries and what a consumer switches on.
    """
    assert tool_server.TOOL_SERVER_UNAVAILABLE == "tool.server_unavailable"
    assert vocabulary.is_published("tool.server_unavailable")


def test_it_joins_the_family_its_siblings_are_already_in() -> None:
    """One family across three modules, because the label is a reader's grouping.

    A different label here would split the tool family in two for a reader who groups
    by it, which is the only thing the label is for -- and it would do so silently,
    since nothing about a wrong label fails at import.
    """
    assert tool_server.FAMILY == tool_call.FAMILY == tool_in_flight.FAMILY == "tool"
    assert vocabulary.PUBLISHED["tool.server_unavailable"] == "tool"


def test_the_registry_needed_no_edit_to_admit_this_third_module() -> None:
    """Discovery found it; the package names it nowhere.

    Asserted rather than trusted, the same way the family's second module asserts it.
    A registry that had to be edited to admit a family would make every new family a
    change to a file some other slice owns, and this is the check that would fail if
    someone replaced discovery with a list.
    """
    walked = {found.name for found in pkgutil.iter_modules(vocabulary.__path__)}
    assert "tool_server" in walked
    assert "tool_server" not in Path(str(vocabulary.__file__)).read_text()
