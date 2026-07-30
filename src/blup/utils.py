from contextlib import contextmanager
import time


@contextmanager
def timed(label: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        print(f"[timing] {label}: {dt:.6f}s", flush=True)
