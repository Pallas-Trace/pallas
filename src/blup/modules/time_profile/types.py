from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Hashable, Literal

from blup.data_model import FidelityMode, QuantaBundle, QuantaQuery, TokenMode
from blup.state import TimeProfileOrder
from blup.types import (
    ColorHex,
    ThreadID,
    ThreadName,
    TimestampNS,
    TokenKey,
    TokenName,
    TraceMode,
    TraceSide,
)


@dataclass(frozen=True)
class TimeProfileTraceContext:
    thread_name_to_id:      dict[ThreadName, ThreadID]
    token_name_by_key:      dict[TokenKey, TokenName]
    query_quanta:           Callable[[QuantaQuery], QuantaBundle]


@dataclass(frozen=True)
class TimeProfileUpdateContext:
    request_key:            Hashable
    trace_context:          dict[TraceSide, TimeProfileTraceContext]
    fidelity:               FidelityMode
    token_mode:             TokenMode
    bin_edges_ns:           tuple[TimestampNS, ...]
    order:                  TimeProfileOrder
    color_map:              dict[TokenKey, ColorHex]


@dataclass(frozen=True)
class TimeProfileUpdate:
    active_thread_names:    tuple[ThreadName, ...]
    start_ns:               TimestampNS
    end_ns:                 TimestampNS
    sync_range_to_fig:      bool
    trace_mode:             TraceMode
    context:                TimeProfileUpdateContext

    @property
    def request_key(self) -> Hashable:
        return self.context.request_key


@dataclass(frozen=True)
class TimeProfileJob:
    thread_name:            ThreadName
    thread_id:              ThreadID
    trace_side:             TraceSide
    thread_center:          float
    # ~~~
    update:                 TimeProfileUpdate


@dataclass(frozen=True)
class TimeProfileRequest:
    request_key:            Hashable
    jobs:                   tuple[TimeProfileJob, ...]


@dataclass(frozen=True)
class TimeProfileResult:
    thread_name:            str
    trace_side:             TraceSide
    # ~~~
    src:                    dict


