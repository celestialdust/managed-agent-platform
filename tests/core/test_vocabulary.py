"""The published event vocabulary is discovered, complete, closed and sealed.

Tier 1 (local, no infrastructure). Four properties. Discovery really ran, so a family
module that exists is in the registry without anyone importing it by name. Every event
name a family module exposes went through `declare` — a module that assigned a constant
without registering it would publish a type the registry has never heard of, and that is
the drift this catches. A duplicate name is refused, because two families claiming one
type name means one of them is silently unreachable. And the registry is shut after
import, so a late import cannot widen what a tenant may be sent.
"""

import pkgutil
from importlib import import_module
from types import MappingProxyType, ModuleType

import pytest

from managed_agent.core import vocabulary


def _family_modules() -> list[ModuleType]:
    return [
        import_module(f"{vocabulary.__name__}.{found.name}")
        for found in pkgutil.iter_modules(vocabulary.__path__)
    ]


def _declared_constants(module: ModuleType) -> dict[str, str]:
    """The event-type constants a family module exposes, by constant name.

    Read off the module rather than taken from the registry, so the two can be compared.
    A constant is an upper-case name holding a dotted string; `FAMILY` is the
    family's own label and is not an event type.
    """
    return {
        name: value
        for name, value in vars(module).items()
        if name.isupper()
        and name != "FAMILY"
        and isinstance(value, str)
        and "." in value
    }


def test_discovery_found_the_family_modules() -> None:
    """Without this, every "is published" assertion below could pass vacuously."""
    assert _family_modules(), "the vocabulary package declares no family modules"
    assert vocabulary.PUBLISHED, "discovery ran but registered nothing"


def test_every_event_name_a_family_exposes_is_published_under_that_family() -> None:
    for module in _family_modules():
        family = module.FAMILY
        constants = _declared_constants(module)
        assert constants, f"{module.__name__} exposes no event types"
        for name, value in constants.items():
            assert vocabulary.is_published(value), (
                f"{module.__name__}.{name} is unpublished"
            )
            assert vocabulary.PUBLISHED[value] == family


def test_the_published_mapping_cannot_be_written_to() -> None:
    assert isinstance(vocabulary.PUBLISHED, MappingProxyType)
    with pytest.raises(TypeError):
        vocabulary.PUBLISHED["session.invented"] = "lifecycle"  # type: ignore[index]


def test_declaring_after_discovery_is_refused() -> None:
    """The seal is the closure: a late import may not widen the published set."""
    with pytest.raises(RuntimeError, match="sealed"):
        vocabulary.declare("session.invented", "lifecycle")
    assert not vocabulary.is_published("session.invented")


def test_two_families_may_not_claim_one_event_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercised against a copy of the registry, so the real published set is untouched.

    The seal has to come off to reach this guard at all, which is the only reason
    a test touches these two privates: the duplicate check runs during discovery,
    and discovery has already finished by the time any test can call in.
    """
    monkeypatch.setattr(vocabulary, "_types", dict(vocabulary.PUBLISHED))
    monkeypatch.setattr(vocabulary, "_sealed", False)

    assert vocabulary.declare("session.invented", "lifecycle") == "session.invented"
    with pytest.raises(ValueError, match="duplicate"):
        vocabulary.declare("session.invented", "takeover")
