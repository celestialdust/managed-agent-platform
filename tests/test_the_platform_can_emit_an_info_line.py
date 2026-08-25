"""The platform's INFO lines have to survive `uvicorn`, and once did not.

`uvicorn` configures logging from its own `LOGGING_CONFIG`, which names three loggers --
`uvicorn`, `uvicorn.error`, `uvicorn.access` -- and declares no `root` entry at all. A
`dictConfig` with no `root` key leaves the root logger exactly as Python built it: level
WARNING, holding no handler. So a `managed_agent` record at INFO propagated to a bare
root, found nothing willing to write it, and was dropped by `logging.lastResort`, whose
own level is WARNING.

The cost was not theoretical. Under `uvicorn` this package could not emit an INFO line
at all, which made a guard written at INFO indistinguishable -- from outside the
process -- from a guard that was never called. That is how a healthy tool census and a
census that never ran came to look identical in a live investigation, and it is why the
level now belongs in an assertion rather than in a reviewer's memory.
"""

from __future__ import annotations

import logging
import logging.config
from collections.abc import Iterator

import pytest
from uvicorn.config import LOGGING_CONFIG

from managed_agent.composition import install_platform_logging

_PACKAGE = "managed_agent"
_PROBE = "managed_agent.probe"
_LINE = "offer census: tenant=t registered=1 offered=1"


@pytest.fixture
def package_logger_restored() -> Iterator[None]:
    """Hand each case a bare package logger, and put the real one back afterwards.

    Logging configuration is process-wide and `dictConfig` is not scoped to a test, so a
    case that installs a handler and leaves would change what every later test in the
    session sees written.

    Clearing on the way *in* matters as much as restoring on the way out, and is not
    symmetry for its own sake. An app factory built earlier in the same session has
    already installed the platform handler, and a `StreamHandler` binds its stream when
    it is constructed -- so that handler holds the real `sys.stderr` from before
    `capsys` replaced it. Left in place it would make `install_platform_logging` return
    at its idempotence check and write the line somewhere this test cannot read, which
    looks exactly like the defect the file is about.
    """
    logger = logging.getLogger(_PACKAGE)
    handlers = list(logger.handlers)
    level, propagate = logger.level, logger.propagate
    logger.handlers.clear()
    logger.propagate = True
    logger.setLevel(logging.NOTSET)
    try:
        yield
    finally:
        logger.handlers[:] = handlers
        logger.setLevel(level)
        logger.propagate = propagate


def test_an_info_line_survives_uvicorns_own_configuration(
    package_logger_restored: None, capsys: pytest.CaptureFixture[str]
) -> None:
    logging.config.dictConfig(LOGGING_CONFIG)
    install_platform_logging()

    logging.getLogger(_PROBE).info(_LINE)

    assert _LINE in capsys.readouterr().err


def test_uvicorns_configuration_alone_drops_that_same_line(
    package_logger_restored: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """The defect itself, kept executable so it cannot come back unnoticed.

    Asserted on stderr rather than on `caplog`, because pytest attaches its own root
    handler and would catch the record that production drops -- the question here is
    what a reader of the pod's log sees, and that is stderr.
    """
    logging.config.dictConfig(LOGGING_CONFIG)
    assert not logging.getLogger(_PACKAGE).handlers, "the fixture owes a bare logger"

    logging.getLogger(_PROBE).info(_LINE)

    assert _LINE not in capsys.readouterr().err


def test_installing_twice_does_not_write_the_line_twice(
    package_logger_restored: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """A factory may be called twice in a process, and in tests it is called twice."""
    logging.config.dictConfig(LOGGING_CONFIG)
    install_platform_logging()
    install_platform_logging()

    logging.getLogger(_PROBE).info(_LINE)

    assert capsys.readouterr().err.count(_LINE) == 1


def test_the_level_comes_from_the_environment_and_a_bad_one_says_so(
    package_logger_restored: None,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreadable level falls back to INFO and complains, rather than bricking a pod.

    Refusing to start would be the usual fail-fast answer and is the wrong one here: the
    cost of a typo would be a platform that will not boot, against a benefit of catching
    that typo a few seconds earlier than the warning already does.
    """
    monkeypatch.setenv("MAP_LOG_LEVEL", "VERBOSE")
    logging.config.dictConfig(LOGGING_CONFIG)
    install_platform_logging()

    logging.getLogger(_PROBE).info(_LINE)

    written = capsys.readouterr().err
    assert "VERBOSE" in written
    assert _LINE in written
