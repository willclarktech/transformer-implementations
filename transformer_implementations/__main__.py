import sys

from transformer_implementations.rwkv import run as run_rwkv

run_rwkv()
# try:
#     run_rwkv()
# # pylint: disable=broad-except
# except Exception as exception:
#     print(f"{type(exception).__name__}: {exception}")
#     sys.exit(1)
