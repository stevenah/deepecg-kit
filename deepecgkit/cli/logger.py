"""CLI logger with verbosity control."""

import logging
import sys
from typing import Any


class CLILogger:
    """Simple logger with verbosity control."""

    def __init__(self, verbose: bool = False, quiet: bool = False):
        self.verbose = verbose
        self.quiet = quiet

    def info(self, msg: str) -> None:
        if not self.quiet:
            print(msg)

    def debug(self, msg: str) -> None:
        if self.verbose and not self.quiet:
            print(f"[DEBUG] {msg}")

    def error(self, msg: str) -> None:
        print(f"[ERROR] {msg}", file=sys.stderr)

    def warning(self, msg: str) -> None:
        if not self.quiet:
            print(f"[WARNING] {msg}", file=sys.stderr)


class _LiveStreamHandler(logging.StreamHandler):
    """StreamHandler that resolves sys.stdout at emit time.

    Binding the stream lazily keeps output visible to pytest's capsys and any
    other stream redirection that happens after logging is configured.
    """

    def __init__(self) -> None:
        super().__init__(stream=sys.stdout)

    @property
    def stream(self) -> Any:
        return sys.stdout

    @stream.setter
    def stream(self, value: Any) -> None:
        pass


def configure_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Route the package's loggers to the console at the CLI's verbosity.

    Library modules log under the ``deepecgkit.*`` hierarchy; this attaches a
    plain-message handler so their output shows up in CLI runs, at DEBUG with
    --verbose, WARNING with --quiet, and INFO otherwise.

    Args:
        verbose: Show debug-level output
        quiet: Show only warnings and errors
    """
    level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    package_logger = logging.getLogger("deepecgkit")
    package_logger.setLevel(level)
    if not any(isinstance(handler, _LiveStreamHandler) for handler in package_logger.handlers):
        handler = _LiveStreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        package_logger.addHandler(handler)
