"""OpenAI's Codex CLI README -- the document a tenant uploads in the whole-journey test.

The user's scenario begins "upload the codex blog to a session", and `codex_blog.md`
beside this module is that document. Vendored rather than fetched at run time for two
reasons: openai.com answers a plain fetch with HTTP 403, and a cluster test whose first
leg needs a third party's website to be up reports that outage as a platform defect.

Real prose rather than something written for the occasion, because the legs downstream
lean on it. Leg 2 asks deepwiki about a Codex *feature*, which only reads as research
if the document is genuinely about Codex; leg 3 renders a brief of it to PDF, and a
brief of invented filler cannot be told from a brief of nothing.

**Held as markdown beside this module rather than as a string inside it, and that is
not a style preference.** The upstream file has lines up to 290 characters, so a
verbatim copy inside a Python string fails ruff's line-length rule -- and both ways
out destroy something. Reflowing the text makes "verbatim" false. Exempting the file
turns the check off for this module's own prose too. A data file is not Python, so
nothing has to be switched off and nothing has to be reworded.

Upstream
    https://github.com/openai/codex/blob/main/README.md
    Read from a local checkout on 2026-08-23. 3334 bytes, sha256
    ba4e1f69ff48386e72a9c5e1edaf76aad64a475c2d51af79ccba6d1128261ba7.

Licence
    Apache-2.0, as the document's own last line says. Attribution above; the licence
    text is not reproduced, because this fixture carries the README and nothing else.

The reference code is NOT in the file. The test appends it, so the bytes on disk stay
identical to upstream and the digest above keeps being checkable. That code is what
proves the agent opened the document rather than recognised it -- which matters more
here than for most fixtures, since a README from a public repository is exactly the
kind of text a model may already have seen.
"""

from pathlib import Path
from typing import Final

_SOURCE: Final = Path(__file__).with_name("codex_blog.md")


def codex_blog() -> str:
    """The vendored README, read from disk on each call.

    A function rather than a module-level constant so importing this module costs no
    file read, and so a caller who mutated the returned string cannot reach the next
    caller. The file is 3 KiB and is read once per test run.
    """
    return _SOURCE.read_text(encoding="utf-8")
