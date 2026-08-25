"""The build step that gives the Agent Runtime a catalogue it will accept.

Asserted against a fixture catalogue of two entries rather than against the real 192 KB
document extracted from the shipped binary. The properties worth freezing are about the
*patch and the checks*, and those do not change when a runtime release renames a model
-- a test reading the real catalogue would fail on the next codex bump for a reason that
is not a defect.

One case does read the real binary, and it skips when the binary is absent. It exists
because every other case here would pass over a fixture whose shape had drifted from the
runtime's: the fixture is this file's idea of a catalogue, and only the binary knows the
real one.

Four of the six content checks fail *identically* to shipping no catalogue at all -- the
runtime answers the same `Unknown model` refusal either way -- so each is driven
separately below. A single "check refuses a bad document" case would pass with five of
the six clauses deleted.
"""

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from managed_agent.core.vfs.evidence import RUNTIME_OUTPUT_CAP_BYTES

_ROOT = Path(__file__).resolve().parents[2]

_ON_PATH = shutil.which("codex")
_REAL_BINARY = None if _ON_PATH is None else Path(_ON_PATH).resolve()
"""The shipped runtime, if this machine has one.

Discovered on `PATH` rather than named as a path, because the two places this file runs
put the binary in different locations and neither is stable: inside the session image it
is wherever npm installed it, and on a workstation it is wherever somebody unpacked it.
A hardcoded path under `/tmp` also silently stops covering anything the first time the
system cleans that directory -- which happened, mid-session, between two runs of this
file.

The real check is the image build: `deploy/docker/session.Dockerfile` runs the whole
step against the binary it just installed, and a document that fails any assertion fails
the build. The two cases below are the same check available offline when a runtime
happens to be at hand.
"""


def _load() -> ModuleType:
    """Import `tools/bake_model_catalog.py` by path.

    `tools/` is not an importable package, so the module is loaded by location. It is
    registered in `sys.modules` before execution for the same reason the sibling loader
    in `test_plan_waves.py` does it: a module that runs before it is registered cannot
    be looked up by name from inside itself.
    """
    spec = importlib.util.spec_from_file_location(
        "bake_model_catalog", _ROOT / "tools" / "bake_model_catalog.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bake = _load()


def _entry(slug: str, **over: Any) -> dict[str, Any]:
    """One catalogue entry carrying every field the checks read, and no more.

    The donor's real entry carries forty fields. Only the ones a check or the clone
    touches are here, because a fixture listing all forty invites the reader to believe
    the other thirty-four matter to this code, and they do not -- they are inherited
    verbatim, which is the whole point of patching rather than authoring.
    """
    base: dict[str, Any] = {
        "slug": slug,
        "visibility": "list",
        "supported_in_api": True,
        "multi_agent_version": "v2",
        "tool_mode": "code_mode_only",
        "availability_nux": {"headline": "new!"},
        "upgrade": {"slug": "something-else"},
        "truncation_policy": {"mode": "tokens", "limit": 10000},
        "display_name": f"Donor {slug}",
        "description": "somebody else's model",
        "context_window": 400000,
        "model_messages": {
            "instructions_template": (
                f"You are Codex, {bake.IDENTITY_CLAIM} Then a great deal about tools."
            ),
            "instructions_variables": {},
        },
    }
    base.update(over)
    return base


@pytest.fixture
def catalog() -> dict[str, Any]:
    """A runtime catalogue holding the donor and one other entry."""
    return {"models": [_entry(bake.DONOR_SLUG), _entry("gpt-other")]}


def test_the_runtime_entries_are_kept_and_hidden_rather_than_deleted(
    catalog: dict[str, Any],
) -> None:
    """Deleting them makes this platform the author of every model's manual.

    Hiding is the runtime's own way of saying resolvable-but-not-offered: `visibility`
    drives the picker, while `supported_in_api` is what decides whether an entry is
    filtered out before a spawn looks at it. So the count only ever grows.
    """
    patched = bake.patch(catalog, ("map-model",))

    slugs = [entry["slug"] for entry in patched["models"]]
    assert slugs == [bake.DONOR_SLUG, "gpt-other", "map-model"]
    hidden = {e["slug"]: e["visibility"] for e in patched["models"]}
    assert hidden == {bake.DONOR_SLUG: "hide", "gpt-other": "hide", "map-model": "list"}


def test_supported_in_api_is_left_at_whatever_each_entry_shipped_with(
    catalog: dict[str, Any],
) -> None:
    """Not an advertisement flag, and not ours to set on somebody else's entry.

    A false entry is filtered out of the model list before a spawn resolves anything, so
    flipping one would silently remove a model the runtime shipped as usable. The check
    downstream refuses a false value rather than correcting it, which is why nothing
    here writes the field.
    """
    catalog["models"][1]["supported_in_api"] = True
    patched = bake.patch(catalog, ())

    assert [e["supported_in_api"] for e in patched["models"]] == [True, True]


def test_every_entry_gets_the_raised_cap_not_only_the_platforms(
    catalog: dict[str, Any],
) -> None:
    """`truncation_policy` is per entry, so an unpatched entry has no defined margin.

    A Session running on one of the runtime's own models would otherwise keep the
    10,000-token default while a Session on a platform model got 5 MB, and Evidence
    capture would have two different margins depending on a value nobody set.
    """
    patched = bake.patch(catalog, ("map-model",))

    assert all(e["truncation_policy"] == bake.RAISED_CAP for e in patched["models"])
    assert bake.RAISED_CAP == {"mode": "bytes", "limit": RUNTIME_OUTPUT_CAP_BYTES}


def test_a_clone_inherits_capability_and_carries_none_of_the_donors_identity(
    catalog: dict[str, Any],
) -> None:
    """The three nulled fields each carry the donor's identity, not its capability.

    `tool_mode: code_mode_only` changes how tools are exposed and this platform exposes
    them over its own gateway; `availability_nux` is a launch announcement for somebody
    else's product release; `upgrade` would have this model advertise itself as retired
    in favour of one no Routing Entry declares. `context_window` is inherited untouched,
    which is the other half of the rule -- capability crosses over.
    """
    patched = bake.patch(catalog, ("map-model",))
    clone = next(e for e in patched["models"] if e["slug"] == "map-model")

    assert clone["tool_mode"] is None
    assert clone["availability_nux"] is None
    assert clone["upgrade"] is None
    assert clone["context_window"] == 400000
    assert clone["multi_agent_version"] == "v2"
    assert clone["display_name"] == "map-model"
    assert "somebody else" not in clone["description"]


def test_the_manuals_claim_about_which_model_it_is_gets_rewritten(
    catalog: dict[str, Any],
) -> None:
    """A Claude-family model handed this manual would be told it is based on GPT-5.

    Everything after that clause is tool protocol and workspace rules and is inherited
    unchanged, which is why only the clause is replaced rather than the template being
    written here.
    """
    patched = bake.patch(catalog, ("map-model",))
    clone = next(e for e in patched["models"] if e["slug"] == "map-model")
    template = clone["model_messages"]["instructions_template"]

    assert bake.IDENTITY_CLAIM not in template
    assert bake.IDENTITY_REPLACEMENT in template
    assert "Then a great deal about tools." in template
    assert "You are Codex," in template


def test_a_template_that_no_longer_carries_the_claim_is_refused_not_shipped(
    catalog: dict[str, Any],
) -> None:
    """A replacement that matches nothing is not an error, and that is the hazard.

    `str.replace` finding nothing returns the original string. So a reworded template in
    a future runtime would produce a catalogue that builds, ships, and hands this
    platform's model a manual claiming it is a different model -- with nothing anywhere
    reporting a problem. This is the assertion that turns that into a failed build.
    """
    donor = catalog["models"][0]
    donor["model_messages"]["instructions_template"] = "You are Codex. Reworded."

    with pytest.raises(SystemExit, match="identity substitution"):
        bake.patch(catalog, ("map-model",))


def test_a_runtime_without_the_donor_is_refused_naming_what_it_does_have(
    catalog: dict[str, Any],
) -> None:
    """The donor was chosen by measuring all eight; a renamed one needs measuring again.

    Refused rather than defaulted to whatever is first, because the properties that made
    this donor the right one -- v2 multi-agent, no launch announcement, no upgrade
    pointer -- are not properties of position in the list.
    """
    catalog["models"] = [_entry("something-renamed")]

    with pytest.raises(SystemExit, match="something-renamed"):
        bake.patch(catalog, ("map-model",))


def test_a_model_the_runtime_already_names_is_not_cloned_over(
    catalog: dict[str, Any],
) -> None:
    """A routed model that is already a runtime entry keeps the runtime's own entry.

    Cloning over it would replace a real operating manual with a copy of the donor's,
    which is strictly worse -- and would put two entries under one slug, which the
    runtime resolves by an order nothing here controls.
    """
    patched = bake.patch(catalog, ("gpt-other",))

    slugs = [e["slug"] for e in patched["models"]]
    assert slugs == [bake.DONOR_SLUG, "gpt-other"]
    assert slugs.count("gpt-other") == 1


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("slug", "", "no 'slug'"),
        ("supported_in_api", False, "filtered out"),
        ("multi_agent_version", "disabled", "disables the multi-agent"),
        ("visibility", "sometimes", "does not recognise"),
        ("truncation_policy", {"mode": "tokens", "limit": 10}, "raised cap"),
        ("model_messages", {}, "no instruction template"),
    ],
)
def test_each_condition_that_fails_like_having_no_catalogue_is_refused_separately(
    field: str, value: object, expected: str
) -> None:
    """Six clauses, six cases, because four of them fail indistinguishably.

    A spawn against a catalogue with any of the first four wrong gets the same `Unknown
    model` refusal it gets with no catalogue at all, and the refusal names none of them.
    A single case over one malformed document would pass with five clauses deleted.
    """
    entry = _entry("map-model")
    entry[field] = value
    entry["truncation_policy"] = entry.get("truncation_policy")
    if field != "truncation_policy":
        entry["truncation_policy"] = dict(bake.RAISED_CAP)

    with pytest.raises(SystemExit, match=expected):
        bake.check({"models": [entry]}, ("map-model",))


def test_a_catalogue_offering_something_the_gateway_cannot_route_is_refused() -> None:
    """A spawn resolves against the model the Session is bound to, and nothing else.

    So the set the catalogue offers and the set the routing table serves have to be one
    set. Offering more means a spawn succeeds and then fails at the model call, which
    surfaces as the platform accepting delegation and losing the answer.
    """
    entry = _entry("map-model", truncation_policy=dict(bake.RAISED_CAP))
    entry["model_messages"]["instructions_template"] = "clean manual"

    with pytest.raises(SystemExit, match="routing table"):
        bake.check({"models": [entry]}, ("a-different-model",))


def test_the_routing_table_is_read_off_the_manifest_this_deployment_applies(
    tmp_path: Path,
) -> None:
    """The models are read from the ConfigMap, in manifest order, parsed twice.

    Twice because the routing table is a JSON string inside a YAML document. Read from
    the manifest rather than restated here, so the catalogue cannot offer a model the
    gateway has no upstream for -- the two would be free to disagree, and the one that
    lost would be found as a spawn that succeeds and then cannot reach a model.
    """
    manifest = tmp_path / "gw.yaml"
    manifest.write_text(
        "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: routing\ndata:\n"
        '  routing.json: |\n    {"entries": [{"model": "one"}, {"model": "two"}]}\n'
    )

    assert bake.routed_models(manifest) == ("one", "two")


def test_a_manifest_naming_one_model_twice_is_refused(tmp_path: Path) -> None:
    """Two entries under one name means the catalogue would carry a duplicate slug.

    Which the runtime resolves by an order this build does not control, so the answer to
    "which upstream serves this model" would depend on parse order.
    """
    manifest = tmp_path / "gw.yaml"
    manifest.write_text(
        "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: routing\ndata:\n"
        '  routing.json: |\n    {"entries": [{"model": "one"}, {"model": "one"}]}\n'
    )

    with pytest.raises(SystemExit, match="names a model twice"):
        bake.routed_models(manifest)


def test_a_manifest_with_no_routing_configmap_is_refused(tmp_path: Path) -> None:
    """Zero tables and two tables are the same defect: no single answer.

    Refused rather than defaulted to empty, because an empty routing table would produce
    a catalogue offering nothing, which builds and ships and refuses every spawn.
    """
    manifest = tmp_path / "gw.yaml"
    manifest.write_text("apiVersion: v1\nkind: Service\nmetadata:\n  name: gw\n")

    with pytest.raises(SystemExit, match="0 routing tables"):
        bake.routed_models(manifest)


def test_a_directory_is_searched_for_the_file_that_actually_holds_a_catalogue(
    tmp_path: Path,
) -> None:
    """The `codex` on PATH after an npm install may be a launcher, not the executable.

    A launcher carries no catalogue, so a build trusting PATH would fail confusingly --
    or worse, succeed against a second copy of the runtime and ship a catalogue
    describing a binary it is not running beside. Searching by the marker rather than by
    filename means the layout of somebody else's package is not this build's business.

    The small file is skipped by size before it is read, which is why it can be a decoy
    carrying the marker: the bound decides what to open and the marker decides what is
    right, and this asserts that order.
    """
    body = b'{\n  "models": [\n    {"slug": "a"}\n  ]\n}'
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "codex").write_bytes(b"#!/usr/bin/env node\n" + body)
    real = tmp_path / "vendor" / "codex"
    real.parent.mkdir()
    real.write_bytes(b"\x00" * bake._MIN_BINARY_BYTES + body)

    assert bake.resolve_binary(tmp_path) == real
    assert bake.resolve_binary(real) == real


def test_a_directory_holding_two_runtimes_is_refused(tmp_path: Path) -> None:
    """Two candidates is a guess about which one the pod will execute.

    Refused rather than resolved by order, because sort order is a fact about filenames
    and the question is which binary runs.
    """
    body = b'{\n  "models": [\n    {"slug": "a"}\n  ]\n}'
    for name in ("one", "two"):
        path = tmp_path / name
        path.write_bytes(b"\x00" * bake._MIN_BINARY_BYTES + body)

    with pytest.raises(SystemExit, match="holds 2 files carrying"):
        bake.resolve_binary(tmp_path)


def test_a_directory_holding_no_runtime_is_refused(tmp_path: Path) -> None:
    """Zero is the same defect as two, and the one a wrong path produces."""
    (tmp_path / "readme").write_text("nothing here")

    with pytest.raises(SystemExit, match="holds 0 files carrying"):
        bake.resolve_binary(tmp_path)


def test_a_binary_carrying_no_catalogue_is_refused(tmp_path: Path) -> None:
    """Extraction has no subject, so it must not guess one.

    The count is asserted rather than the presence: two catalogues in one binary would
    make this pick whichever came first, and a build that picks between two documents
    without saying so is the failure this refuses.
    """
    fake = tmp_path / "codex"
    fake.write_bytes(b"not a codex binary")

    with pytest.raises(SystemExit, match="carries 0 model catalogues"):
        bake.extract_catalog(fake)


def test_a_binary_carrying_two_catalogues_is_refused(tmp_path: Path) -> None:
    """The other half of the same defect, and the one a real bump could produce."""
    fake = tmp_path / "codex"
    body = b'{\n  "models": [\n    {"slug": "a"}\n  ]\n}'
    fake.write_bytes(b"padding" + body + b"padding" + body)

    with pytest.raises(SystemExit, match="carries 2 model catalogues"):
        bake.extract_catalog(fake)


@pytest.mark.skipif(
    _REAL_BINARY is None, reason="no codex on PATH to read a real catalogue out of"
)
def test_the_real_binary_yields_the_catalogue_the_patch_expects() -> None:
    """The one case the fixture cannot cover: that the fixture's shape is the real
    shape.

    Every other case here is asserted over a document this file wrote, so all of them
    would pass over a fixture that had drifted from the runtime's actual catalogue. This
    reads the shipped binary and asserts the two facts the patch depends on -- that the
    donor is present, and that its manual still carries the identity clause the
    substitution targets.

    Skipped rather than failed when the binary is absent, because the binary is not in
    this repository. A skip here means this run did not check it.
    """
    document = bake.extract_catalog(_REAL_BINARY)
    slugs = {str(entry["slug"]) for entry in document["models"]}

    assert bake.DONOR_SLUG in slugs, f"donor absent; the runtime names {sorted(slugs)}"
    donor = next(e for e in document["models"] if e["slug"] == bake.DONOR_SLUG)
    template = donor["model_messages"]["instructions_template"]
    assert template.count(bake.IDENTITY_CLAIM) == 1
    assert donor["multi_agent_version"] == "v2"
    assert donor["availability_nux"] is None
    assert donor["upgrade"] is None


@pytest.mark.skipif(
    _REAL_BINARY is None, reason="no codex on PATH to read a real catalogue out of"
)
def test_the_whole_step_produces_a_document_that_passes_its_own_checks(
    tmp_path: Path,
) -> None:
    """End to end over the real inputs, because the parts passing is not the whole.

    Runs `main` the way the image build runs it, against the shipped binary and this
    repository's own routing manifest, and asserts the written document offers exactly
    the routed models. `check` runs inside `main`, so a document that would be refused
    never reaches the file -- which is why the assertion below is about what was written
    rather than about a return value.
    """
    out = tmp_path / "models.json"
    code = bake.main(
        [
            "--binary",
            str(_REAL_BINARY),
            "--routing-table",
            str(_ROOT / "deploy" / "k8s" / "model-gateway.yaml"),
            "--out",
            str(out),
        ]
    )

    assert code == 0
    written = json.loads(out.read_text())
    offered = [e["slug"] for e in written["models"] if e["visibility"] == "list"]
    assert offered == list(bake.routed_models(_ROOT / "deploy/k8s/model-gateway.yaml"))


def test_a_clone_does_not_ask_for_the_wire_the_gateway_does_not_speak() -> None:
    """The donor's `use_responses_lite` is overwritten, not inherited.

    Under it the runtime moves its tool list out of the top-level `tools` array and
    into an `additional_tools` item inside `input`. The Model Gateway reads the
    top-level array, so the clone would reach the provider with no tools bound -- and
    nothing would fail: the model is still told about its tools in its instructions, so
    it answers by describing calls it never made.
    """
    catalog = {"models": [_entry(bake.DONOR_SLUG, use_responses_lite=True)]}

    patched = bake.patch(catalog, ("map-model",))

    clone = next(e for e in patched["models"] if e["slug"] == "map-model")
    assert clone["use_responses_lite"] is False
    donor = next(e for e in patched["models"] if e["slug"] == bake.DONOR_SLUG)
    assert donor["use_responses_lite"] is True, (
        "the runtime's own entries are hidden and nothing selects them, so rewriting a "
        "field on one only makes the diff against the next runtime harder to read"
    )


def test_a_catalogue_whose_offered_entry_asks_for_that_wire_is_refused() -> None:
    """The forced field is checked on the way out, not only set on the way in.

    `patch` and `check` are separate entry points and the Dockerfile runs both. A clone
    built correctly today and a hand-edit tomorrow reach the image by the same path, so
    the assertion has to sit where the document is, not where the clone was made.
    """
    catalog = {"models": [_entry(bake.DONOR_SLUG, use_responses_lite=True)]}
    patched = bake.patch(catalog, ("map-model",))
    clone = next(e for e in patched["models"] if e["slug"] == "map-model")
    clone["use_responses_lite"] = True

    with pytest.raises(SystemExit, match="use_responses_lite"):
        bake.check(patched, ("map-model",))
