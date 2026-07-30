from __future__ import annotations

from typing import TypeAlias


# NOTE: finish propagating new centralized types

TraceID: TypeAlias = str
ContextKey: TypeAlias = str
ThreadName: TypeAlias = str
TokenKey: TypeAlias = tuple[int, int]


