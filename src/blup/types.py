from __future__ import annotations

from typing import Literal


# NOTE: finish propagating new centralized types

# -------------------------------------------
# |             Trace Types                 |
# -------------------------------------------

# registry-assigned identifier for one loaded trace
type TraceID        = str
# user-assigned display tag for a trace
type TraceLabel     = str
# distinquishes two traces when in 'dual' mode
# NOTE: maybe considering renaming
type TraceSide      = Literal[
        "upper",
        "lower"
]
# trace display mode: direct display (1) vs comparison (2)
type TraceMode      = Literal[
        "dual",
        "single",
]

# -------------------------------------------
# |             Thread Types                |
# -------------------------------------------

# backend thread identifier used within a trace
type ThreadID       = int
# backend thread string based on trace data
type ThreadName     = str
# user-assigned display tag for a thread
type ThreadLabel    = str

# -------------------------------------------
# |              Token Types                |
# -------------------------------------------

# token category type (based on PALLAS)
type TokenType      = int
# token unique identifier within a type category
type TokenID        = int
# canonical identity for a distinct token
type TokenKey       = tuple[TokenType, TokenID]
# stringified version of TokenKey where applicable
type TokenKeyStr    = str

# -------------------------------------------
# |              Time Types                 |
# -------------------------------------------

# absolute point in trace time (nano-seconds)
type TimestampNS    = int
# interval length in trace time (nano-seconds)
type DurationNS     = int

# -------------------------------------------
# |               Job Types                 |
# -------------------------------------------

# opaque key for distinguishing unqique work jobs/requests
type RequestKey     = tuple

# -------------------------------------------
# |            Display Types                |
# -------------------------------------------

# color value in hex format (#rrggbb)
type ColorHex       = str
# shell panel 'context' identifier
type ContextKey     = str

# -------------------------------------------
# |                 Meta                    |
# -------------------------------------------

__all__ = [
    "TraceID",
    "TraceLabel",
    "TraceSide",
    "TraceMode",
    "ThreadID",
    "ThreadName",
    "TokenType",
    "TokenID",
    "TokenKey",
    "TokenKeyStr",
    "TimestampNS",
    "DurationNS",
    "RequestKey",
    "ColorHex",
    "ContextKey",
]


