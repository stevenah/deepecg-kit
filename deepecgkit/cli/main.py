"""Main CLI entry point: parser assembly, config merging, and dispatch.

Each subcommand's arguments live next to its implementation in
deepecgkit.cli.commands.<name> as an ``add_parser(subparsers)`` function that
also records how to dispatch it (``handler`` names the callable on the
deepecgkit.cli package — resolved late so tests can patch it — and
``to_kwargs`` maps parsed args to that callable's keyword arguments).
"""

import argparse
import importlib
import sys

import deepecgkit.cli as _cli
from deepecgkit.cli.config import load_config
from deepecgkit.cli.logger import CLILogger, configure_logging

_COMMAND_MODULES = (
    "train",
    "evaluate",
    "predict",
    "extract",
    "explain",
    "probe",
    "analysis",
    "info",
)

_EPILOG = """
Examples:
  deepecg train -m kanres -d af-classification
  deepecg train -m kanres -d af-classification --force-download
  deepecg evaluate --checkpoint model.pt -m kanres -d af-classification
  deepecg predict --checkpoint model.pt -m kanres --input ecg.npy
  deepecg explain --checkpoint model.pt -m kanres --input ecg.npy -o explanations/
  deepecg probe -m kanres --source-checkpoint af-classification=af.pt -o probe/
  deepecg list-models
  deepecg info -m kanres

Note: Datasets are automatically downloaded when first used.
"""


def build_parser() -> tuple[argparse.ArgumentParser, dict[str, argparse.ArgumentParser]]:
    """Build the CLI parser.

    Returns:
        Tuple of the top-level parser and a mapping of command name to its
        subparser (used by the config merge).
    """
    parser = argparse.ArgumentParser(
        prog="deepecg",
        description="DeepECG-Kit: Deep learning toolkit for ECG analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG,
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress non-essential output",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to configuration file (YAML or JSON)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    for name in _COMMAND_MODULES:
        module = importlib.import_module(f"deepecgkit.cli.commands.{name}")
        module.add_parser(subparsers)
    return parser, dict(subparsers.choices)


def _apply_config(
    parser: argparse.ArgumentParser,
    subparser_by_name: dict[str, argparse.ArgumentParser],
    argv: list | None,
) -> dict:
    """Merge a --config file into the chosen command's parser defaults.

    A first parse locates --config and the command; the command's config
    section is then installed as that subparser's defaults, so the precedence
    "explicit flags > config values > built-in defaults" falls out of argparse
    itself. Keys the command does not define are reported and ignored.

    Args:
        parser: The full parser from build_parser
        subparser_by_name: Command-name to subparser mapping
        argv: The command line being parsed

    Returns:
        The loaded configuration dictionary (empty without --config).
    """
    preliminary, _ = parser.parse_known_args(argv)
    config: dict = {}
    if preliminary.config:
        config = load_config(preliminary.config) or {}

    command = preliminary.command
    if command:
        section = config.get(command) or config.get(command.replace("-", "_")) or {}
        subparser = subparser_by_name[command]
        known = {action.dest for action in subparser._actions}
        unknown = sorted(set(section) - known)
        if unknown:
            print(
                f"[WARNING] Ignoring unknown config keys for '{command}': {', '.join(unknown)}",
                file=sys.stderr,
            )
        subparser.set_defaults(**{key: value for key, value in section.items() if key in known})
    return config


def main(argv: list | None = None) -> int:
    """Main CLI entry point."""
    parser, subparser_by_name = build_parser()

    try:
        config = _apply_config(parser, subparser_by_name, argv)
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}", file=sys.stderr)
        return 1

    args = parser.parse_args(argv)

    verbose = args.verbose or config.get("verbose", False)
    quiet = args.quiet or config.get("quiet", False)
    configure_logging(verbose=verbose, quiet=quiet)
    logger = CLILogger(verbose=verbose, quiet=quiet)

    if args.command is None:
        parser.print_help()
        return 1

    handler = getattr(_cli, args.handler)
    kwargs = args.to_kwargs(args)
    if getattr(args, "pass_logger", True):
        kwargs["logger"] = logger
    result = handler(**kwargs)
    return 0 if result is None else result
