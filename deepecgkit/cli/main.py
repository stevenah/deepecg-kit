"""Main CLI entry point: parser assembly, config merging, and dispatch.

Each subcommand's arguments live next to its implementation in
deepecgkit.cli.commands.<name> as an ``add_parser(subparsers)`` function that
also records how to dispatch it (``handler`` names the callable on the
deepecgkit.cli package — resolved late so tests can patch it — and
``to_kwargs`` maps parsed args to that callable's keyword arguments).
"""

import argparse
import importlib
import importlib.util
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

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

#: The module that defines each command. A run of one command imports only
#: its own module, so the analysis commands, for one, start without PyTorch.
_MODULE_OF_COMMAND = {
    "train": "train",
    "evaluate": "evaluate",
    "predict": "predict",
    "extract-features": "extract",
    "explain": "explain",
    "probe": "probe",
    "aggregate": "analysis",
    "rank": "analysis",
    "list-models": "info",
    "list-datasets": "info",
    "info": "info",
}

#: Environment variable naming extra modules to import before the parser is
#: built, comma-separated; the same as repeating --import.
PLUGINS_ENV = "DEEPECGKIT_PLUGINS"

_EPILOG = """
Examples:
  deepecg train -m kanres -d af-classification
  deepecg evaluate --checkpoint runs/<run-dir> -d af-classification
  deepecg predict --checkpoint runs/<run-dir> --input ecg.npy --sampling-rate 300
  deepecg explain --checkpoint runs/<run-dir> --input ecg.npy -o explanations/
  deepecg probe -m kanres --source-checkpoint af-classification=runs/<run-dir> -o probe/
  deepecg list-models
  deepecg list-datasets
  deepecg --import my_models.py train -m my-model -d af-classification

Datasets are downloaded on first use (sizes: deepecg list-datasets) into
~/.deepecgkit, or $DEEPECGKIT_HOME when that is set.
--verbose, --quiet and --config may be given before or after the command.
"""


def _package_version() -> str:
    try:
        return version("deepecgkit")
    except PackageNotFoundError:
        return "0.0.0.dev0"


def _add_global_options(parser: argparse.ArgumentParser, *, top_level: bool) -> None:
    """Register the options every command accepts.

    They are registered on the top-level parser and again on every subcommand
    so that they work on either side of the command name. On a subcommand the
    defaults are suppressed: an option left out there must not overwrite the
    value given before the command.

    Args:
        parser: The top-level parser or one subcommand's parser
        top_level: Whether parser is the top-level one
    """
    target = parser if top_level else parser.add_argument_group("global options")
    suppressed: dict = {} if top_level else {"default": argparse.SUPPRESS}
    target.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output, including tracebacks for errors",
        **suppressed,
    )
    target.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress non-essential output",
        **suppressed,
    )
    target.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to configuration file (YAML or JSON)",
        **suppressed,
    )
    target.add_argument(
        "--import",
        dest="plugins",
        action="append",
        metavar="MODULE",
        help="Import a module or .py file before running, so the models and datasets it "
        f"registers can be used by name; repeatable (also: ${PLUGINS_ENV})",
        **(suppressed or {"default": []}),
    )


def build_parser(
    command: str | None = None,
) -> tuple[argparse.ArgumentParser, dict[str, argparse.ArgumentParser]]:
    """Build the CLI parser.

    Args:
        command: The command about to run, when it is already known. Only the
            module defining it is then imported, and the parser knows that
            module's commands alone; anything else gets every command.

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
    parser.add_argument("--version", action="version", version=f"deepecg {_package_version()}")
    _add_global_options(parser, top_level=True)

    subparsers = parser.add_subparsers(dest="command", help="Available commands", metavar="COMMAND")
    wanted = _MODULE_OF_COMMAND.get(command or "")
    for name in (wanted,) if wanted else _COMMAND_MODULES:
        module = importlib.import_module(f"deepecgkit.cli.commands.{name}")
        module.add_parser(subparsers)
    for subparser in subparsers.choices.values():
        _add_global_options(subparser, top_level=False)
    return parser, dict(subparsers.choices)


def import_plugin(target: str) -> None:
    """Import one module so that its registrations take effect.

    Args:
        target: A dotted module name, or a path to a .py file

    Raises:
        ImportError: If the module cannot be found or fails to import.
    """
    path = Path(target)
    if target.endswith(".py") or os.sep in target:
        if not path.is_file():
            raise ImportError(f"No such file: {target}")
        # The file's own folder goes on the path so that it can import its
        # neighbours, and so that data loader workers can import it by name.
        folder = str(path.resolve().parent)
        if folder not in sys.path:
            sys.path.insert(0, folder)
        name = path.stem
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load a module from {target}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return

    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    importlib.import_module(target)


def _prescan(argv: list[str] | None) -> argparse.Namespace:
    """Find the options that must be known before the real parser exists.

    Plugins have to be imported before the parser is built, because the
    --model and --dataset choices are read from the registries they fill, and
    the config file has to be loaded before the real parse, because it can
    supply required arguments.

    Args:
        argv: The command line being parsed, or None for sys.argv

    Returns:
        Namespace with config, plugins, version, and command (the first
        positional).
    """
    scanner = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    scanner.add_argument("--config", "-c", default=None)
    scanner.add_argument("--import", dest="plugins", action="append", default=[])
    scanner.add_argument("--verbose", "-v", action="store_true")
    scanner.add_argument("--quiet", "-q", action="store_true")
    scanner.add_argument("--version", action="store_true")
    found, remaining = scanner.parse_known_args(sys.argv[1:] if argv is None else argv)
    found.command = next((token for token in remaining if not token.startswith("-")), None)
    return found


def _apply_config(
    subparser_by_name: dict[str, argparse.ArgumentParser],
    command: str | None,
    config_path: str | None,
) -> dict:
    """Merge a --config file into the chosen command's parser defaults.

    The command's config section is installed as that subparser's defaults,
    so the precedence "explicit flags > config values > built-in defaults"
    falls out of argparse itself. An argument the command requires stops
    being required once the config supplies it. Keys the command does not
    define are reported and ignored.

    Args:
        subparser_by_name: Command-name to subparser mapping
        command: The command named on the command line, if any
        config_path: The --config value, if any

    Returns:
        The loaded configuration dictionary (empty without --config).
    """
    config: dict = {}
    if config_path:
        config = load_config(config_path) or {}

    if command in subparser_by_name:
        section = config.get(command) or config.get(command.replace("-", "_")) or {}
        subparser = subparser_by_name[command]
        known = {action.dest for action in subparser._actions}
        unknown = sorted(set(section) - known)
        if unknown:
            print(
                f"[WARNING] Ignoring unknown config keys for '{command}': {', '.join(unknown)}",
                file=sys.stderr,
            )
        supplied = {key: value for key, value in section.items() if key in known}
        subparser.set_defaults(**supplied)
        for action in subparser._actions:
            if action.dest in supplied:
                action.required = False
        for group in subparser._mutually_exclusive_groups:
            if any(action.dest in supplied for action in group._group_actions):
                group.required = False
    return config


def main(argv: list | None = None) -> int:
    """Main CLI entry point."""
    early = _prescan(argv)
    if early.version and early.command is None:
        # Answered here, before the parser and with it every command is imported.
        # Exits as argparse's own --version does.
        print(f"deepecg {_package_version()}")
        sys.exit(0)

    plugins = list(early.plugins)
    plugins += [name.strip() for name in os.environ.get(PLUGINS_ENV, "").split(",") if name.strip()]
    for plugin in plugins:
        try:
            import_plugin(plugin)
        except Exception as e:
            print(f"[ERROR] Failed to import '{plugin}': {e}", file=sys.stderr)
            return 1

    parser, subparser_by_name = build_parser(early.command)

    try:
        config = _apply_config(subparser_by_name, early.command, early.config)
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
    try:
        result = handler(**kwargs)
    except KeyboardInterrupt:
        # Reached when the interruption lands outside a command's own
        # handling, typically during a dataset download.
        logger.warning("Interrupted. A download in progress resumes on the next run.")
        return 130
    return 0 if result is None else result
