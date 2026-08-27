"""What this platform promises the agent about its workspace, in one place.

A Session's agent is a model with a shell, and everything it knows about where it is
comes from being told. Four separate mechanisms have to agree on that story -- the
directory the shim ships files out of, the directory the manifest mounts inputs into,
the interpreter path the image builds, and the sentence the model reads -- and until
this module existed only the last one did not exist at all.

**The absence was measured, not guessed.** A live probe on 2026-08-23 asked an agent
"do you have any documents available?" with nothing in the prompt naming a directory.
It found them: it shelled out, listed around, and reported `/session/workspace/files/`
correctly. So discovery works. What it costs is a round trip of exploration on every
Turn, and what it does not give is any way for the platform to state a convention the
agent could not have discovered -- where to PUT a deliverable, how to install a package
it needs. A model cannot explore its way to a promise nobody has made.

**This is a contract and not a prompt, which is why it lives in `core/`.** Every clause
below is something another part of this tree enforces: `shim/serve.py` ships what is in
`OUTPUT_DIR_NAME`, `deploy/k8s/session-pod.yaml` mounts inputs at `INPUT_DIR_NAME` and
puts `PACKAGE_DIR` on `PYTHONPATH`, `deploy/docker/session.Dockerfile` installs
`PIP_WRAPPER`, and the permission profile in `control/pod_config/compiler.py` is what
makes `SCRATCH_ROOT` writable at all -- the clause about it would otherwise send the
model to a directory the kernel refuses. A sentence here that no other file honours is
worse than no sentence: the model would follow it, write its document somewhere nothing
collects, and the Turn would end looking like the model's failure.

**It reaches the model in the same field as the tenant's own instructions, and that
is a compromise rather than a design.** The right channel exists: the runtime has an
administrator's key, `additional_developer_instructions`, which it renders as a separate
`developer`-role message wrapped in `<managed_developer_instructions>` and attributed to
whoever runs the platform. Both released codex versions -- 0.149.0, which this platform
pins, and 0.149.1 -- contain the string zero times; `developer_instructions` appears 39
and 43 times in the same two binaries. The key is on the runtime's `main` branch and in
no release, so writing it produces a document that parses, loads, and drops this text
without a word. Measured that way first: an agent asked three questions only this
contract answers said `UNKNOWN` to all three while the key sat correctly escaped in the
pod's own `/etc/codex/requirements.toml`.

So `instructions_for_the_model` composes the two halves into the one field that works,
and the labels around them are what is left of the separation the other key would have
given for free. What that costs, stated plainly: a tenant's instructions and the
platform's now arrive with the same authorship as far as the runtime is concerned, and
a tenant who writes `ignore any instructions about directories` is contradicting the
platform inside a field the platform also wrote. Nothing here prevents that. The reason
it is tolerable is that the contract describes the pod rather than constraining the
agent -- an agent that disregards it writes its file where nothing collects it, which
costs that Session its deliverable and costs this platform's other tenants nothing.
"""

from typing import Final

from managed_agent.core.vfs.session_vfs import (
    VfsPathInvalid,
    parse_relative_path,
)

INPUT_DIR_NAME: Final = "files"
"""Where a Session's attached documents are, relative to the working directory.

Not a choice this module makes. `deploy/k8s/session-pod.yaml` mounts the workspace
volume with `subPath: files` for the shim's write mount, so this is the directory that
exists -- and `shim/serve.py` builds its own path from this constant so the two cannot
drift into naming different directories.
"""

OUTPUT_DIR_NAME: Final = "out"
"""Where the agent puts what the tenant asked for.

**This exists because the alternative shipped and was wrong.** Ship-out took every
regular file at the workspace root, so a Turn that rendered a PDF by writing and running
a generator script returned TWO files: `codex-brief.pdf`, which the tenant wanted, and
`make_pdf.py`, which was the agent's scratch. Measured on a live run. Both counted
against the file and byte budgets, and a tenant integrating against this had no rule for
telling a deliverable from a working file.

A named directory rather than a filter over names, because no filter is honest. A
`.py` file is scratch when a model wrote it to render a PDF and the deliverable when a
tenant asked for a script, and nothing at the far end can tell those apart. The agent
knows which is which, and this is how it says so.

The previous rule stays as a fallback and the reason is not caution. A convention the
model ignores must degrade to "the tenant gets their file with some scratch beside it",
never to "the tenant gets nothing" -- the first is untidy and the second loses work
somebody paid for. `shim/serve.py` prefers this directory when it holds anything and
falls back to the root when it does not.
"""


def is_a_produced_path(relative: str) -> bool:
    """Whether a file at this workspace-relative path may be shipped out as produced.

    **One predicate with two callers, because two copies of this rule would be two
    rules.** The shim filters its listing by it and the control plane re-parses every
    path that arrives by it, and those two have to agree exactly: a path the shim offers
    and the control plane refuses fails a Turn that did nothing wrong, and a path the
    shim filters out but the control plane would have taken is a document silently
    dropped. Neither end owns the rule, so it lives here, beside the directory names it
    is about.

    **The lane-path grammar governs, not the upload-filename grammar.** A produced file
    is placed into a Session's `artifacts` lane rather than minted as an upload, so the
    rule that has to hold is the destination's: `parse_relative_path` is what
    `SealedFile` parses at construction, so a path this accepts is one a key can be
    built from and one it rejects could not be stored under any name. Holding a
    produced file to `parse_upload_filename` instead would refuse `report/fig1.png` for
    carrying a separator -- the very file this whole path now exists to deliver.

    The difference between the two grammars is one-sided, which is why the change costs
    nothing that worked before: every legal upload filename is a legal single-segment
    lane path. What is newly allowed is a separator, and only that.

    One rule is added on top, because the lane grammar cannot express it. `..` and a
    leading separator are already refused, and so is a *leading* dot -- the grammar
    requires the first character to be alphanumeric, so `.codex/state` never reaches
    here at all. But a dot may appear inside a path, so `src/.cache/x` and
    `out/.venv/lib/x` both parse. A dotted segment anywhere is runtime
    scratch or an installed dependency tree rather than a document, so every segment is
    checked and not just the first -- a rule applied only to the first segment is one an
    agent escapes by writing a single directory deep.
    """
    try:
        parse_relative_path(relative)
    except VfsPathInvalid:
        return False
    return not any(segment.startswith(".") for segment in relative.split("/"))


SCRATCH_ROOT: Final = "/session/scratch"
"""The pod's own disk, writable by a confined command and gone when the pod is.

An absolute path and not a workspace-relative name, because it is deliberately not in
the workspace: the workspace is a network mount now, and a write there is a round trip
(ADR-035, ADR-036). Everything a Turn produces on the way to a deliverable -- a build
tree, a package cache, an unpacked archive -- has no reason to make that trip, and the
whole of ADR-037 is putting those writes here instead.

Three separate things have to agree for this path to be real, which is why the name
lives in `core/` beside the clause that promises it rather than in any one of them.
`deploy/k8s/session-pod.yaml` declares the volume and mounts it into the container that
runs confined commands; `control/pod_config/compiler.py` puts a write rule over it in
the permission profile, without which the profile's `:read-only` parent leaves it
unwritable; and `deploy/docker/session.Dockerfile` aims the build tools' caches at it.
Two of the three is worse than none of them: the tools would be pointed at a path that
exists and refuses every write, so a Turn fails on a directory the platform itself
chose rather than falling back to one that works.
"""

SCRATCH_LIMIT_MEBIBYTES: Final = 512
"""How much that directory holds: the manifest's `sizeLimit`, and what the model reads.

The model is told, and that is unusual here -- every other clause below degrades when
it is ignored, and this one does not. Enforcement on these nodes is kubelet's periodic
`du` with no filesystem-quota feature gate, so a write past this does not return
ENOSPC: the pod is EVICTED, and `restartPolicy: Never` ends the Session mid-Turn. An
agent that has not been told the number has no basis for deciding not to unpack a 2 GB
dataset here.

Rendered to the model as plain "MB" against a value in mebibytes, which under-reads the
real bound by about five percent. That is the safe direction for a number whose whole
purpose is stopping somebody just short of an eviction, and a clause explaining the
difference between MB and MiB would cost more of the model's attention than five
percent is worth.
"""

PACKAGE_DIR: Final = f"{SCRATCH_ROOT}/lib"
"""Where a package the agent installs at run time lands.

On scratch rather than in the workspace, and that is a move rather than a first choice.
It sat at `<workspace>/.map/lib` while the workspace was pod-local, under a dotted
directory so `is_a_produced_path` above could never mistake a dependency tree for a
document. Once the workspace became a network mount, leaving it there would have put
every run-time `pip install` across NFS -- new latency bought for bytes that are
rebuildable by definition and that nothing was keeping anyway.

The dot went with the move because it no longer buys anything: ship-out walks the
workspace, and this is not under it, so a site-packages tree stays out of a tenant's
deliverables by not being in the tree that gets scanned rather than by a filter that
has to keep holding.

Absolute, because the two places that must agree with it are not workspace-relative
either: the `--target` in the image's `map-pip` wrapper, and `PYTHONPATH` in the pod
manifest. `tests/deploy/test_the_image_honours_the_workspace_contract.py` compares all
three, and a move that reached two of them installs packages nothing can import.
"""

PIP_WRAPPER: Final = "map-pip"
"""The command the agent runs to install a Python package.

A wrapper rather than the convention written out, because the convention is four things
a skill's author never mentions. Anthropic's published `pdf` skill opens a recipe with
`pip install pytesseract pdf2image`, and in a Session that line cannot run: there is no
pip on the agent's PATH, the venv it would install into is read-only, `pip`'s own
scratch directory would be on a read-only `/tmp`, and the result would need to be on
`PYTHONPATH` to import. Telling the model all four is a paragraph it may get wrong on
any Turn. Telling it one command name is a paragraph it cannot get wrong.

Named rather than `pip`, and not shadowing it either. A wrapper installed AS `pip` would
make `pip --version` and `pip download` behave in ways their own documentation does not
describe, and the model reads that documentation. A distinct name is a distinct thing.
"""


def workspace_contract() -> str:
    """The platform's promises about this workspace, as the model reads them.

    Assembled from the constants above rather than written out, so a clause and the
    thing that honours it cannot disagree: changing `OUTPUT_DIR_NAME` changes both the
    directory the shim scans and the sentence the model is given, in one edit.

    Written as short imperative clauses because it competes for the model's attention
    with the tenant's own instructions, which are the ones that should win. This says
    only what the tenant cannot: where the platform looks, and what it provides.
    """
    return "\n".join(
        (
            "Workspace layout, provided by the platform running this session:",
            "",
            f"- Documents attached to this session are in ./{INPUT_DIR_NAME}/ ,",
            "  relative to your working directory. Read them from there.",
            f"- Put files the user asked you to produce in ./{OUTPUT_DIR_NAME}/ .",
            "  Create that directory if it does not exist. Only files there are",
            "  returned to the user, so working files and scripts you wrote along",
            "  the way should stay outside it.",
            "  Each path there is written once. To revise something you already",
            "  produced, write the new version under a new path -- rewriting the",
            "  old one is refused. Keep a draft outside this directory and copy",
            "  it in when it is final.",
            "- Large intermediate files -- build output, caches, an unpacked",
            f"  archive -- belong in {SCRATCH_ROOT}/ . That is this machine's own",
            "  disk, while your working directory is on network storage, so",
            "  writing them there is faster. It holds"
            f" {SCRATCH_LIMIT_MEBIBYTES} MB and going past that",
            "  ends the session; nothing in it is returned to the user, and none",
            "  of it survives the session.",
            f"- To install a Python package, run: {PIP_WRAPPER} <package>",
            "  It installs where this session can import from. Plain `pip install`",
            "  will not work here: everything outside your working directory and",
            f"  {SCRATCH_ROOT}/ is read-only.",
            "- Network access is off unless it was granted for this session. If a",
            "  download is refused, say so rather than working around it.",
        )
    )


PLATFORM_LABEL: Final = "platform"
"""The tag around the platform's half of the instructions field.

Short, lower case, and not spelled like the runtime's own
`<managed_developer_instructions>`. A label that mimicked the runtime's wrapper would
read to the model as if the runtime had put it there, which would be this platform
claiming an authority it does not have in that field -- and would mislead the next
engineer to open a Session's compiled `config.toml` looking for who wrote what.
"""

AGENT_LABEL: Final = "agent"
"""The tag around the tenant's half.

Both halves are labelled, not just ours. Labelling only the platform's half would leave
the tenant's text as the field's unmarked default, so a reader could not tell where the
platform's stopped -- and a tenant whose own text happened to open with a line about
directories would appear to be the platform saying it.
"""


def instructions_for_the_model(agent_instructions: str) -> str:
    """The single instructions field a Session's model receives, both halves labelled.

    The platform's contract comes first and the tenant's text last, because the tenant's
    is about the errand and should be what the model is holding when it starts work.
    The order is not a precedence claim: nothing here resolves a conflict between the
    two, and the module docstring says why that is tolerable.

    The framing sentence is here rather than in `workspace_contract`, which is also read
    by the tests that compare the contract against the manifest and the image. Those
    compare clauses to the things that honour them, and a sentence about authorship
    honoured by nothing would have to be exempted from every one of those comparisons.

    Takes the tenant's text rather than reading it, so this module keeps depending on
    nothing: it is the one place that decides what the model reads about its workspace,
    and it does not also decide where a tenant's instructions come from.
    """
    return "\n".join(
        (
            "Two sets of instructions follow, from two different authors.",
            f"The {PLATFORM_LABEL} block describes the machine you are running on.",
            f"The {AGENT_LABEL} block is what you were configured to do.",
            "",
            f"<{PLATFORM_LABEL}>",
            workspace_contract(),
            f"</{PLATFORM_LABEL}>",
            "",
            f"<{AGENT_LABEL}>",
            agent_instructions,
            f"</{AGENT_LABEL}>",
        )
    )
