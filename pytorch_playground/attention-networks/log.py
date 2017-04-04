import sys


def log(*args, **kwargs):
    """use stdout.flush to allow streaming to file when used by IPython. IPython doesn't have -u option."""
    print(*args, **kwargs)
    sys.stdout.flush()
