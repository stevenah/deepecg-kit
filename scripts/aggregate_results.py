"""Aggregate per-run test metrics across random seeds.

Shim kept for documented invocations; the implementation lives in
``deepecgkit.cli.commands.analysis`` and is also available as
``deepecg aggregate``.
"""

import sys

from deepecgkit.cli.main import main

if __name__ == "__main__":
    sys.exit(main(["aggregate", *sys.argv[1:]]))
