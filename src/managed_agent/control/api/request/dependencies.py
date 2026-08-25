"""What a handler in this package needs from the request, resolved in one place.

`tenancy.py` answers where a request's tenant comes from. This answers the other
question every handler here asks: where the wired `Platform` comes from. Both are
"how a handler gets its inputs", and both are here rather than in each route module
so that changing the answer is one edit.

Extracted at five copies rather than at three, which is later than the Rule of Three
asks for, and the delay is visible in the copies: the five docstrings had already
drifted into three different explanations of the same two lines. That drift is the
argument. Nothing had gone *wrong* -- the bodies were identical -- but a comment that
says something slightly different in five files is a comment nobody can correct, and
the next wave adds seven more route modules.
"""

from fastapi import Request

from managed_agent.composition import Platform


def platform_from_request(request: Request) -> Platform:
    """The `Platform` the app factory was handed.

    `app.state` is untyped by construction -- it is a namespace anything may put
    anything on -- so the one read of it is funnelled through here and typed once,
    rather than every handler quietly working against `Any`. Under `mypy --strict` an
    `Any` flowing out of `app.state` would silently disable checking of everything it
    touched downstream, which is the failure this exists to prevent and is invisible
    at the call site.

    The `assert` is a type narrowing rather than a validation. Nothing but
    `create_app` writes this attribute, so a wrong type here is a programming error in
    the composition root and not something a request can cause -- there is no
    caller-supplied value on this path to refuse.
    """
    platform = request.app.state.platform
    assert isinstance(platform, Platform)
    return platform
