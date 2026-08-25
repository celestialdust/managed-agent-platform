"""Compare the declared AWS account against the live one, and fail on a difference.

`terraform plan -detailed-exitcode` answers in three exit codes, and this tool keeps
all three apart rather than collapsing them, because two of them are findings of
different kinds and the third is not a finding at all:

  0  the account matches the configuration.
  2  a plan was produced and it is not empty -- something differs. A resource block
     deleted from a config whose resources are in state renders "will be destroyed"
     and lands here.
  1  no plan could be produced. Absent credentials and an uninitialised backend land
     here and are not findings. So does a plan that wanted to replace a resource
     carrying `lifecycle.prevent_destroy`, which prints "Instance cannot be
     destroyed" after rendering the whole plan and very much IS a finding. Terraform
     does not separate those two in its exit code, so the message names both and
     says which text tells them apart.

Exit 3 is this tool's own and means a precondition failed, so no comparison was
attempted at all. It is kept off 1 so that "the account could not be compared" can
never be misread as "a protected resource would have been destroyed".

`terraform fmt -check` runs first because a config terraform would reformat produces
a diff that is about whitespace rather than about the account. Both halves need the
same binary and the same directory; splitting them would put that directory in a
second place free to disagree with this one.

The exit code, not the output, is the verdict. The plan text is printed for a human
and is never parsed: a grep over plan output would be a second, weaker
implementation of the comparison terraform has just performed.

Two flags on every call, for reasons that are about the callers rather than about
terraform. `-no-color`, because terraform colours its output whenever stdout is a
terminal and this output is read in three places -- a terminal, a redirected log, and
`docs/progress.md` -- two of which then have to strip ANSI escapes out of the `Plan:`
line they quote. And `-input=false` on the plan, because a variable this configuration
does not have today would otherwise turn the gate into a process waiting on a prompt
nobody is at: a gate that hangs reports nothing, which is strictly worse than one that
fails. `terraform fmt` takes no `-input` flag -- it prints its usage and formats
nothing if given one -- and it has nothing to prompt for, so it gets only the first.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

_CONFIG = Path(__file__).resolve().parents[1] / "deploy" / "terraform"

# What terraform returns from `plan -detailed-exitcode`.
_TF_NO_CHANGES = 0
_TF_CHANGES_PRESENT = 2

# What this tool returns. 0 and 2 line up with terraform's on purpose; 3 has no
# terraform counterpart because terraform was never asked anything.
_AGREE = 0
_UNPLANNABLE = 1
_DRIFT = 2
_PRECONDITION = 3

_NO_TERRAFORM = """\
terraform is not on PATH, so the account was NOT compared. It is not a Python
dependency and `uv sync` does not install it: install it from your package manager,
or from https://developer.hashicorp.com/terraform/install.
"""

_NOT_FORMATTED = """\
{config} is not canonically formatted, so the account was NOT compared. Run
`terraform fmt -recursive` and try again -- a config terraform would reformat
produces a diff about whitespace instead of about the account.
"""

_DRIFTED = """\
The account and {config} disagree. Read the plan above: every line is either a
change somebody made outside this repository, or a line of the configuration that
is wrong. Reconcile one to the other -- do NOT edit the configuration to match an
unexplained change without recording why.
"""

_NO_PLAN = """\
terraform could not produce a plan, so nothing was compared -- UNLESS the output
above ends in "Instance cannot be destroyed", which is a finding: the plan wanted
to replace a resource carrying lifecycle.prevent_destroy. Otherwise the usual
causes are absent credentials and an uninitialised backend (`terraform init`).
"""


def _run(*args: str) -> int:
    """Run terraform in the config directory, letting its output through.

    Nothing is captured. The plan is for a human to read, and what its exit code
    means is the caller's decision rather than this function's.
    """
    return subprocess.run(["terraform", *args], cwd=_CONFIG, check=False).returncode


def main() -> int:
    """Return 0 if the account matches, 2 on a difference, 1 if no plan was made.

    Returns 3 without consulting terraform's account at all when a precondition
    fails, so that a run which compared nothing is never mistaken for a verdict.
    """
    if shutil.which("terraform") is None:
        sys.stderr.write(_NO_TERRAFORM)
        return _PRECONDITION

    if _run("fmt", "-check", "-recursive", "-diff", "-no-color") != 0:
        sys.stderr.write(_NOT_FORMATTED.format(config=_CONFIG))
        return _PRECONDITION

    code = _run(
        "plan",
        "-detailed-exitcode",
        "-lock-timeout=60s",
        "-no-color",
        "-input=false",
    )
    if code == _TF_NO_CHANGES:
        return _AGREE
    if code == _TF_CHANGES_PRESENT:
        sys.stderr.write(_DRIFTED.format(config=_CONFIG))
        return _DRIFT
    sys.stderr.write(_NO_PLAN)
    return _UNPLANNABLE


if __name__ == "__main__":
    raise SystemExit(main())
