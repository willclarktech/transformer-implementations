import sys

from transformer_implementations.lib import run

try:
    run()
# pylint: disable=broad-except
except Exception as exception:
    print(f"{type(exception).__name__}: {exception}")
    sys.exit(1)
