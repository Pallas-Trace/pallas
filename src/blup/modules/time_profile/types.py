from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Hashable, Literal

from blup.data_model import FidelityMode, QuantaBundle, QuantaQuery, TokenMode
from blup.state import TimeProfileOrder, TraceMode


TraceSide = Literal["lower", "upper"]
QueryQuantaFn = Callable[[QuantaQuery], QuantaBundle]
# NOTE: is this ^^ necessary ?


@dataclass(frozen=True)
class TimeProfileTraceContext:
    thread_name_to_id:      dict[str, int]
    token_name_by_key:      dict[str, str]
    query_quanta:           QueryQuantaFn


@dataclass(frozen=True)
class TimeProfileUpdateContext:
    request_key:            Hashable
    trace_context:          dict[TraceSide, TimeProfileTraceContext]
    fidelity:               FidelityMode
    token_mode:             TokenMode
    bin_edges_ns:           tuple[int, ...]
    order:                  TimeProfileOrder
    color_map:              dict[str, str]


@dataclass(frozen=True)
class TimeProfileUpdate:
    active_thread_names:    tuple[str, ...]
    start_ns:               int
    end_ns:                 int
    sync_range_to_fig:      bool
    trace_mode:             TraceMode
    context:                TimeProfileUpdateContext

    @property
    def request_key(self) -> Hashable:
        return self.context.request_key


@dataclass(frozen=True)
class TimeProfileJob:
    thread_name:            str
    thread_id:              int
    trace_side:             TraceSide
    thread_center:          float
    update:                 TimeProfileUpdate


@dataclass(frozen=True)
class TimeProfileRequest:
    request_key:            Hashable
    jobs:                   tuple[TimeProfileJob, ...]


@dataclass(frozen=True)
class TimeProfileResult:
    thread_name:            str
    trace_side:             TraceSide
    src:                    dict


