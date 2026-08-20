from __future__ import annotations

from contextlib import contextmanager
import os
import time
from collections import OrderedDict
from threading import Lock
from typing import TYPE_CHECKING

from blup.types import TokenID, TokenKey, TokenKeyStr, TokenType

if TYPE_CHECKING:
    from blup.modules.interface import AnyWorkRequest

# TODO: organize this file!

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

def _describe_job(request: AnyWorkRequest, job: object) -> str:
    pipeline = getattr(request, "pipeline", None)
    module_id = getattr(pipeline, "module_id", "?")
    pid = f"{id(pipeline) & 0xFFFF:04x}" if pipeline is not None else "----"

    payload = getattr(job, "payload", job)
    parts = []
    kind = getattr(payload, "kind", None)
    if kind is not None:
        parts.append(str(kind))
    side = getattr(payload, "trace_side", None)
    if side is not None:
        parts.append(str(side))
    thread = getattr(payload, "thread_name", None)
    if thread is not None:
        parts.append(f"thread={thread}")

    detail = " ".join(parts) if parts else type(payload).__name__
    return f"{module_id}[{pid}] {detail}"

@contextmanager
def log_job(request: AnyWorkRequest, job: object):
    label = _describe_job(request, job)
    if not VERBOSE:
        yield
        return
    print(f"[job] start {label}", flush=True)
    try:
        yield
    except BaseException as exc:
        print(f"[job] FAIL  {label}: {exc!r}", flush=True)
        raise
    print(f"[job] done  {label}", flush=True)


def as_token_key(token_type: TokenType, token_id: TokenID) -> TokenKey:
    return (token_type, token_id)

def as_token_key_str(token_type: TokenType, token_id: TokenID) -> TokenKeyStr:
    return f"{token_type}:{token_id}"


def format_percent_diff(v1: float, v2: float) -> str:
    if v1 == 0:
        return "0.0%" if v2 == 0 else "—"
    pct = ((v2 - v1) / v1) * 100.0
    return f"{pct:+.1f}%"

def format_duration_ns(value: float) -> str:
    sign = "-" if value < 0 else ""
    x = abs(float(value))
    if x < 1_000:
        return f"{sign}{x:.0f} ns"
    if x < 1_000_000:
        return f"{sign}{x / 1_000:.3f} us"
    if x < 1_000_000_000:
        return f"{sign}{x / 1_000_000:.3f} ms"
    return f"{sign}{x / 1_000_000_000:.3f} s"

def format_duration_delta_ns(value: float) -> str:
    if value == 0:
        return "0 ns"
    sign = "+" if value > 0 else "-"
    x = abs(float(value))
    if x < 1_000:
        return f"{sign}{x:.0f} ns"
    if x < 1_000_000:
        return f"{sign}{x / 1_000:.3f} us"
    if x < 1_000_000_000:
        return f"{sign}{x / 1_000_000:.3f} ms"
    return f"{sign}{x / 1_000_000_000:.3f} s"


class DataCache[K, V]:

    def __init__(self, capacity: int = 32):
        if capacity < 0:
            raise ValueError(f"capacity must be >= 0, got {capacity}")

        self.capacity = capacity
        self._data: OrderedDict[K, V] = OrderedDict()
        self._lock = Lock()

    def get(self, key: K) -> V | None:
        with self._lock:
            value = self._data.get(key)
            if value is None:
                return None

            self._data.move_to_end(key)
            return value

    def put(self, key, value):
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)

            while len(self._data) > self.capacity:
                self._data.popitem(last=False)

    def clear(self):
        with self._lock:
            self._data.clear()


