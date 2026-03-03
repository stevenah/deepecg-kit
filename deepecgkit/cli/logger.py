"""CLI logger with verbosity control."""

import sys


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
