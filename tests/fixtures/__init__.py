"""Real third-party documents, vendored verbatim, for cases needing the genuine article.

A package rather than a bare directory so a case can say `from fixtures.codex_blog
import codex_blog` and have both pytest and `mypy --strict` resolve it: pytest puts
`tests/` on `sys.path` because `tests/conftest.py` lives there, and mypy derives the
module name `fixtures.x` from this file's presence.

What belongs here is a document this repository did NOT write and must not paraphrase --
Anthropic's published `pdf` skill, OpenAI's Codex README. Provenance and licence go in
each module's own docstring, with the byte count and digest of what was fetched, because
a vendored copy whose origin is not recorded cannot be checked against upstream later.

What does not belong here is test data we invented. That reads better next to the case
that uses it, where a reader can see why it has the shape it has.
"""
