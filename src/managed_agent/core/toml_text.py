"""Quoting a value so TOML carries it back unchanged.

Here, and not private to a module, because there were two copies and they had diverged.
`control/pod_config/model_binding.py` escaped newlines because it emits a definition's
instructions, which run to many lines; `control/pod_config/compiler.py` escaped only
backslash and quote because it emitted paths and profile names, which never contain a
newline. Both were correct for what they emitted on the day they were written, and the
moment the compiler had to emit a paragraph the incomplete one produced a document the
runtime rejects at load -- after the pod has started.

That is the shape of the risk worth naming: neither copy was wrong, and the divergence
was invisible until a caller crossed from one kind of value to the other. A rule about
how to escape a string is one piece of knowledge, so it is written once.
"""

from typing import Final

_ESCAPES: Final = str.maketrans(
    {
        "\\": "\\\\",
        '"': '\\"',
        "\n": "\\n",
        "\r": "\\r",
        "\t": "\\t",
        "\b": "\\b",
        "\f": "\\f",
    }
)


def toml_string(value: str) -> str:
    """Quote `value` as a TOML basic string, escaping what TOML cannot carry raw.

    Raises `ValueError` if a control character survives the table. TOML forbids the
    whole C0 range and DEL inside a basic string, and there is no escape for most of
    them -- so a value carrying one cannot be expressed, and refusing is the only honest
    answer. Emitting it anyway would produce a document that fails to parse inside the
    pod, which is a failure with no line of provenance: the compiler that wrote it is
    long gone by then.

    Escaped rather than rendered as a multi-line literal, even for text with newlines. A
    literal string cannot express a quote or a backslash at all, so a value carrying one
    would have to be re-quoted by a rule that looked at the content -- and a quoting
    function whose output format depends on its input is one that has two behaviours to
    test and one of them to forget.
    """
    escaped = value.translate(_ESCAPES)
    if any(character < " " or character == "\x7f" for character in escaped):
        raise ValueError(
            f"a value carries a control character TOML cannot express: {value!r}"
        )
    return f'"{escaped}"'
