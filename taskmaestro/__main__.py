"""Allow ``python -m taskmaestro`` as an alias for the ``taskmaestro`` command."""

import sys

from taskmaestro.cli import main

if __name__ == "__main__":
    sys.exit(main())
