from __future__ import annotations

from contextlib import contextmanager
import os
import time


VERBOSE: bool = os.environ.get("BLUP_VERBOSE", "").lower() in ("1", "true", "yes")

def set_verbose(enabled: bool) -> None:
    global VERBOSE
    VERBOSE = enabled


@contextmanager
def timed(label: str):
    if not VERBOSE:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        print(f"[timing] {label}: {dt:.6f}s", flush=True)


