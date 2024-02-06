import sys

from transformer_implementations.transformer import run as run_transformer

try:
    run_transformer()
# pylint: disable=broad-except
except Exception as exception:
    print(f"{type(exception).__name__}: {exception}")
    sys.exit(1)
