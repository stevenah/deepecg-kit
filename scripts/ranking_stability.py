"""Report how stable architecture rankings are across random seeds.

Shim kept for documented invocations; the implementation lives in
``deepecgkit.cli.commands.analysis`` and is also available as
``deepecg rank``.
"""

import sys

from deepecgkit.cli.main import main

if __name__ == "__main__":
    sys.exit(main(["rank", *sys.argv[1:]]))
