import sys

from transformer_implementations.lib import hello

try:
    hello()
# pylint: disable=broad-except
except Exception as exception:
    print(f"{type(exception).__name__}: {exception}")
    sys.exit(1)
