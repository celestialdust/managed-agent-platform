"""Build the model catalogue the Agent Runtime reads, from the binary being shipped.

The runtime validates a spawned agent's model against a catalogue compiled into its own
binary. That catalogue names eight OpenAI models and nothing else, so every attempt to
delegate to a model this platform actually serves is refused -- the agent tries five
times, tells the user it cannot, and does the work itself. Handing the runtime a
replacement catalogue is the only way to make delegation possible, and
`model_catalog_json` takes a filesystem path rather than inline configuration.

**Why this runs at image build time rather than per Session.** The eight entries weigh
192,663 bytes raw and 256,884 once the Kubernetes API server has base64'd them into a
Secret. One Session's requirements Secret has roughly 200 KiB left after skill delivery,
so the catalogue does not fit at any size -- not "does not fit today", but has no size
at which it would. Written into the image once, it costs a Session nothing and is read
from a path no volume mounts over.

**Why extraction rather than a copy checked into this repository.** A vendored copy can
describe a different runtime than the one shipped, and nothing about the file would say
so; the failure would appear as a spawn refused for a model the catalogue names. Reading
it out of the binary in the same build that installs the binary makes that state
unrepresentable rather than guarded.

The document is patched rather than authored. Authoring one means writing an operating
manual for every model from nothing, and an entry whose `model_messages` carries neither
an instructions template nor a top-level `base_instructions` is refused by the runtime's
deserializer as `InvalidData` at configuration load -- which stops every pod rather than
degrading one. So the eight are kept and hidden, and a platform model's entry is cloned
from one of them.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

from managed_agent.core.vfs.evidence import RUNTIME_OUTPUT_CAP_BYTES

CATALOG_MARKER = b'{\n  "models": [\n'
"""Where the catalogue starts inside the binary.

The literal opening of the pretty-printed document, which occurs exactly once in the
whole 258 MB of codex-cli 0.149.0 -- asserted below rather than assumed, because a
second occurrence would make the extraction pick one of two documents silently. Matching
the document's own text rather than a byte offset is what lets this survive a rebuild:
an offset is a fact about one compilation, and the opening brace of a JSON object is a
fact about the document.
"""

DONOR_SLUG = "gpt-5.6-terra"
"""The entry a platform model's entry is cloned from.

Chosen by measurement over all eight, not by preference. It is the only one that carries
`multi_agent_version: "v2"` -- proven capable of the multi-agent backend a spawn goes
through -- while carrying neither an `availability_nux` (a launch announcement, which
would have this platform's model introduce itself as somebody else's product release)
nor an `upgrade` pointer (which would have it advertise itself as retired in favour of a
model no Routing Entry declares). `gpt-5.6-sol` is also v2 and does carry a nux;
`gpt-5.2`'s instructions template names itself in prose.
"""

RAISED_CAP = {"mode": "bytes", "limit": RUNTIME_OUTPUT_CAP_BYTES}
"""The truncation policy every entry is given.

`truncation_policy` is a per-entry field, so a Session running on an entry that still
carries the runtime's 10,000-token default has an undefined margin for Evidence capture
-- which is why this is applied to all of them and not only to the platform's. See
`docs/adr/ADR-020`.

The limit is **imported** rather than written here, and that import is the reason this
script depends on `managed_agent` at all. A literal would be a second copy of the number
Evidence capture measures its margin against, and the two would be free to disagree --
which is exactly what happened: the first version of this held `5_000_000` and was kept
honest by a test that scraped this file's source text for the digits. A guard that reads
source text is not a guard, it is a second copy with a worse failure mode. The venv the
image build runs this with has the package installed, and so does every workstation that
can run the test suite.
"""

IDENTITY_CLAIM = "an agent based on GPT-5."
IDENTITY_REPLACEMENT = "an agent based on the model this session is bound to."
"""The one sentence of the donor's manual that is about who the model is.

The template's first line reads "You are Codex, an agent based on GPT-5." Everything
after it -- 17,730 characters of tool protocol, workspace rules and approval handling --
is capability and is inherited unchanged. That first clause is identity, and inherited
by a Claude-family model it is simply false, which is a poor way to open an operating
manual.

`Codex` is left in place: it is the runtime's own product name and the agent really is
running inside it.
"""

_ERASED_FIELDS = ("tool_mode", "availability_nux", "upgrade")
"""Donor fields nulled on a clone, because each one carries the donor's identity.

`tool_mode: "code_mode_only"` is not decoration -- it changes how tools are exposed to
the model, and this platform exposes them over its own gateway. The other two are
described above. All three are nullable: five of the eight entries already carry null in
`tool_mode`, so this is the runtime's own representation of "unset" rather than a value
invented here.
"""


_FORCED_FIELDS = {"use_responses_lite": False}
"""Donor fields overwritten on a clone, because the donor asks for a wire we do not run.

`use_responses_lite` is the only one, and it is not a preference. Measured against this
deployment: with it true the runtime sends its tool list inline, as an
`additional_tools` item inside `input`, and leaves the top-level `tools` array empty.
The Model Gateway implements the ordinary Responses wire, where the tool list is that
top-level array -- so it drops the inline item, finds no tools, and builds an upstream
request carrying none.

Nothing fails when that happens, which is what makes it worth a constant. The model is
still told about its tools in its instructions, so it answers by writing tool calls as
prose -- naming real tools, quoting file contents it never read and byte counts it
invented -- and the Turn completes. Sixteen live cases failed on the contents of that
answer before anyone read a request body.

So the wire the gateway implements is the wire the catalogue must ask for. The eight
entries this runtime ships are left exactly as they are: they are hidden, no Session
selects one, and rewriting a field on an entry nobody runs would only make the diff
against the next runtime harder to read.
"""


_MIN_BINARY_BYTES = 20 * 1024 * 1024
"""Below this, a file is not the runtime and is not worth reading into memory.

The npm package that installs codex puts a small launcher on `PATH` and the real
executable somewhere under its own tree, and the layout is the package's business rather
than ours. So a directory may be searched -- but a search that reads every file in it
would read a few hundred megabytes of JavaScript looking for a document that only ever
lives in a 258 MB executable. The bound is a filter on what to open, never a decision
about which file is right: that is settled by the marker.
"""


def resolve_binary(target: Path) -> Path:
    """The one file under `target` carrying a model catalogue.

    Accepts a file, in which case it is used as given, or a directory, which is
    searched. The directory case exists because the `codex` on `PATH` after an npm
    install may be a launcher rather than the executable, and a launcher carries no
    catalogue -- so a build that trusted `PATH` would either fail confusingly or, worse,
    succeed against a second copy of the runtime and ship a catalogue describing a
    binary it is not running beside.

    Refuses zero and refuses more than one. Two candidates means the tree holds two
    runtimes and picking either is a guess about which one the pod will execute.
    """
    if target.is_file():
        return target
    if not target.is_dir():
        raise SystemExit(f"{target} is neither a file nor a directory")
    found = [
        path
        for path in sorted(target.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and path.stat().st_size >= _MIN_BINARY_BYTES
        and CATALOG_MARKER in path.read_bytes()
    ]
    if len(found) != 1:
        raise SystemExit(
            f"{target} holds {len(found)} files carrying a model catalogue, not 1"
            + (f": {[str(p) for p in found]}" if found else "")
        )
    return found[0]


def extract_catalog(binary: Path) -> dict[str, Any]:
    """Read the model catalogue out of a codex binary.

    Finds the document's opening text and lets a JSON decoder find its end, so nothing
    here needs to know how long the document is. Refuses a binary carrying no catalogue
    or more than one, since either makes the result something other than "the catalogue
    this binary uses".

    Decoded with `surrogateescape` because the surrounding bytes are machine code and
    the slice handed to the decoder runs past the document's end; the escape keeps a
    non-UTF-8 byte beyond the document from failing a decode of the document itself.
    """
    data = binary.read_bytes()
    found = data.count(CATALOG_MARKER)
    if found != 1:
        raise SystemExit(
            f"{binary} carries {found} model catalogues, not 1; the extraction has no "
            "unambiguous subject and this build must not guess which one is live"
        )
    start = data.find(CATALOG_MARKER)
    text = data[start:].decode("utf-8", errors="surrogateescape")
    document, _ = json.JSONDecoder().raw_decode(text)
    if not isinstance(document, dict):
        raise SystemExit(f"{binary}'s catalogue decoded to {type(document)}, not a map")
    return document


def routed_models(manifest: Path) -> tuple[str, ...]:
    """Every model name this deployment can actually reach, in manifest order.

    Read from the Model Gateway's routing table rather than from anything about a
    tenant, because that is where the answer exists at build time: the table is
    deployment-level configuration naming which upstream serves which model, and a
    Session bound to a name outside it already fails at its first model call. So a
    catalogue built from this table refuses nothing that was not already refused, and it
    cannot name a model the platform could not serve.

    The routing table lives as a JSON string inside a YAML ConfigMap, which is why this
    parses twice.
    """
    documents = [d for d in yaml.safe_load_all(manifest.read_text()) if d]
    tables = [
        d["data"]["routing.json"]
        for d in documents
        if d.get("kind") == "ConfigMap" and "routing.json" in d.get("data", {})
    ]
    if len(tables) != 1:
        raise SystemExit(
            f"{manifest} holds {len(tables)} routing tables, not 1; which models this "
            "deployment serves has no single answer and the catalogue would guess"
        )
    entries = json.loads(tables[0])["entries"]
    models = tuple(str(entry["model"]) for entry in entries)
    if len(set(models)) != len(models):
        raise SystemExit(f"{manifest}'s routing table names a model twice: {models}")
    if not models:
        raise SystemExit(f"{manifest}'s routing table is empty")
    return models


def _cloned(donor: dict[str, Any], slug: str) -> dict[str, Any]:
    """One platform entry, inheriting the donor's capability and none of its identity.

    A new dict rather than a mutation, so the donor stays available as a donor for the
    next slug and the eight kept entries are not edited by having been read.

    The identity substitution is asserted rather than attempted. A replacement that
    matches nothing is not an error in Python -- it returns the original string -- so a
    reworded template in a future runtime would produce a catalogue that builds, ships,
    and tells this platform's model it is somebody else's.
    """
    entry = dict(donor)
    entry["slug"] = slug
    entry["display_name"] = slug
    entry["description"] = "Served by this platform's Model Gateway."
    entry["visibility"] = "list"
    entry["truncation_policy"] = dict(RAISED_CAP)
    for field in _ERASED_FIELDS:
        entry[field] = None
    entry.update(_FORCED_FIELDS)

    messages = dict(donor["model_messages"])
    template = str(messages["instructions_template"])
    if template.count(IDENTITY_CLAIM) != 1:
        raise SystemExit(
            f"the donor template does not contain {IDENTITY_CLAIM!r} exactly once, so "
            f"the identity substitution for {slug} would silently not apply and the "
            "model would be handed a manual claiming it is a different model"
        )
    messages["instructions_template"] = template.replace(
        IDENTITY_CLAIM, IDENTITY_REPLACEMENT
    )
    entry["model_messages"] = messages
    return entry


def patch(document: dict[str, Any], models: tuple[str, ...]) -> dict[str, Any]:
    """The document the runtime should read, built from the one it shipped with.

    The eight are kept and hidden rather than deleted. Deleting them makes this platform
    the author of every model's operating manual, and hiding is the runtime's own way of
    saying "resolvable but not offered" -- `visibility` drives the picker only, while
    `supported_in_api` decides whether an entry is filtered out before a spawn ever
    looks at it. So `supported_in_api` is left at whatever each entry shipped with.
    """
    kept = {str(entry["slug"]): entry for entry in document["models"]}
    if DONOR_SLUG not in kept:
        raise SystemExit(
            f"this runtime's catalogue has no {DONOR_SLUG!r} to clone from; it names "
            f"{sorted(kept)}. Pick a donor from that list with multi_agent_version v2, "
            "no availability_nux and no upgrade, and say why in DONOR_SLUG's docstring"
        )
    hidden = []
    for entry in document["models"]:
        shown = dict(entry)
        shown["visibility"] = "hide"
        shown["truncation_policy"] = dict(RAISED_CAP)
        hidden.append(shown)
    platform = [_cloned(kept[DONOR_SLUG], slug) for slug in models if slug not in kept]
    return {**document, "models": [*hidden, *platform]}


def check(document: dict[str, Any], models: tuple[str, ...]) -> None:
    """Refuse to ship a catalogue that would not do what it exists to do.

    Four of these fail *identically* to shipping no catalogue at all -- the runtime
    answers the same `Unknown model` refusal and says nothing about which condition was
    missed -- which is why each is a named assertion here rather than a comment. The
    fifth stops every pod instead of one spawn, so it is the most expensive to discover
    late.

    Checked at build time because that is the last moment the inputs are still present.
    A catalogue the deserializer rejects is `InvalidData` at configuration load, which
    means a Session pod that starts and immediately dies for a reason visible only in
    its container log.
    """
    entries = document["models"]
    for entry in entries:
        slug = entry.get("slug")
        if not slug:
            raise SystemExit(
                f"an entry carries no 'slug', so there is nothing for the runtime to "
                f"match a requested model against: {sorted(entry)}"
            )
        if entry.get("supported_in_api") is not True:
            raise SystemExit(
                f"entry {slug!r} is not supported_in_api, so it is filtered out of the "
                "model list before a spawn looks at it -- this platform's auth is not "
                "ChatGPT's, and that flag is not an advertisement"
            )
        if entry.get("multi_agent_version") == "disabled":
            raise SystemExit(
                f"entry {slug!r} disables the multi-agent backend, which is the one "
                "value that excludes it from the path a spawn takes"
            )
        if entry.get("visibility") not in ("list", "hide"):
            raise SystemExit(
                f"entry {slug!r} carries visibility {entry.get('visibility')!r}, which "
                "the runtime does not recognise"
            )
        if entry.get("truncation_policy") != RAISED_CAP:
            raise SystemExit(
                f"entry {slug!r} carries {entry.get('truncation_policy')!r} rather "
                f"than the raised cap {RAISED_CAP!r}, so Evidence capture has no "
                "defined margin against it"
            )
        template = (entry.get("model_messages") or {}).get("instructions_template")
        if not template:
            raise SystemExit(
                f"entry {slug!r} carries no instruction template, so a model selected "
                "on it would run with an empty operating manual"
            )

    listed = {str(e["slug"]) for e in entries if e.get("visibility") == "list"}
    if listed != set(models):
        raise SystemExit(
            f"the catalogue offers {sorted(listed)} but this deployment's routing "
            f"table serves {sorted(models)}; a spawn resolves against the model a "
            f"Session is "
            "bound to, so the two sets have to be the same one"
        )
    for slug in listed:
        entry = next(e for e in entries if e["slug"] == slug)
        template = entry["model_messages"]["instructions_template"]
        if IDENTITY_CLAIM in template:
            raise SystemExit(
                f"entry {slug!r} still claims {IDENTITY_CLAIM!r} in its instructions"
            )
        for field, wanted in _FORCED_FIELDS.items():
            if entry.get(field) != wanted:
                raise SystemExit(
                    f"entry {slug!r} carries {field}={entry.get(field)!r} rather than "
                    f"{wanted!r}; see _FORCED_FIELDS -- a Session on this entry would "
                    "reach the model with no tools bound and answer by describing "
                    "calls it never made"
                )


def main(argv: list[str] | None = None) -> int:
    """Extract, patch, check, write -- and write nothing if any check refuses."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        type=Path,
        default=None,
        help="the codex executable, or a directory holding it (default: PATH)",
    )
    parser.add_argument(
        "--routing-table",
        type=Path,
        default=Path("deploy/k8s/model-gateway.yaml"),
        help="the manifest holding this deployment's routing ConfigMap",
    )
    parser.add_argument("--out", type=Path, required=True, help="where to write it")
    args = parser.parse_args(argv)

    binary = args.binary
    if binary is None:
        resolved = shutil.which("codex")
        if resolved is None:
            raise SystemExit("no codex on PATH and no --binary given")
        binary = Path(resolved).resolve()

    models = routed_models(args.routing_table)
    binary = resolve_binary(binary)
    document = patch(extract_catalog(binary), models)
    check(document, models)

    body = json.dumps(document, indent=2, ensure_ascii=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(body + "\n")
    offered = sorted(e["slug"] for e in document["models"] if e["visibility"] == "list")
    print(
        f"wrote {args.out} from {binary}: {len(document['models'])} entries, "
        f"{len(body)} bytes, offering {offered}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
