import time
import contextlib


@contextlib.contextmanager
def timer(name=''):
    s = time.perf_counter()
    yield
    e = time.perf_counter()
    name = str(name)
    if name:
        name = str(name) + ': '
    print(name + '{}'.format(e-s))
