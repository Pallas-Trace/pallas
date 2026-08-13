from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Hashable, Literal, Protocol, Sequence, Union

from blup.data_model import (
    FidelityMode,
    SnapshotHistogram,
    SnapshotHistogramQuery,
    SummaryQuery,
    TokenMode,
    TraceSummary,
)
from blup.state import TokenDetailChartMode, TokenDetailTableMode, TraceMode

# NOTE: canonicalize this elsewhere
TraceSide = Literal["upper", "lower"]

TokenDetailJobKind = Literal["table", "histogram"]

# -----------------------------------------

def token_to_select_value(token: tuple[int, int] | None) -> str:
    if token is None:
        return ""
    return f"{token[0]}:{token[1]}"


def select_value_to_token(value: str) -> tuple[int, int] | None:
    if not value:
        return None
    raw_type, raw_id = value.split(":", 1)
    return (int(raw_type), int(raw_id))


@dataclass(frozen=True)
class TokenDetailDiffRow:
    token_type:             int
    token_id:               int
    name:                   str
    call_count_upper:       int
    call_count_lower:       int
    incl_total_ns_upper:    int
    incl_total_ns_lower:    int
    excl_total_ns_upper:    int
    excl_total_ns_lower:    int
    mean_incl_ns_upper:     float
    mean_incl_ns_lower:     float
    mean_excl_ns_upper:     float
    mean_excl_ns_lower:     float
    delta_call_count:       int
    delta_mean_incl_ns:     float
    delta_mean_excl_ns:     float
    delta_incl_total_ns:    int
    delta_excl_total_ns:    int
    thread_ids_upper:       tuple[int, ...]
    thread_ids_lower:       tuple[int, ...]
    contribution_abs_ns:    int = 0
    contribution_share_pct: float = 0.0
    contribution_rank:      int = 0

    @property
    def token(self) -> tuple[int, int]:
        return (self.token_type, self.token_id)


@dataclass(frozen=True)
class TokenDetailTableModel:
    title:                  str
    subtitle:               str
    metric:                 tuple[str, ...]
    upper:                  tuple[str, ...]
    lower:                  tuple[str, ...]
    delta:                  tuple[str, ...]
    percent:                tuple[str, ...]
    dual_mode:              bool
    upper_label:            str
    lower_label:            str
    options:                tuple[tuple[str, str], ...]
    selected_value:         str

# NOTE: the above ^^ should be considered temporary and
#       later modified or moved as appropraite
# -----------------------------------------


@dataclass(frozen=True)
class TokenDetailTraceContext:
    label:                  str
    # NOTE: check this param ^^
    thread_name_to_id:      dict[str, int]
    token_name_by_key:      dict[str, str]
    summarize_tokens:       Callable[[SummaryQuery], TraceSummary]
    query_histogram:        Callable[[SnapshotHistogramQuery], SnapshotHistogram]


@dataclass(frozen=True)
class TokenDetailUpdateContext:
    request_key:            Hashable
    trace_context:          dict[TraceSide, TokenDetailTraceContext]
    fidelity:               FidelityMode
    token_mode:             TokenMode
    top_k:                  int
    histogram_bins:         int


@dataclass(frozen=True)
class TokenDetailUpdate:
    active_thread_names:    tuple[str, ...]
    start_ns:               int
    end_ns:                 int
    selected_token:         tuple[int, int] | None
    trace_mode:             TraceMode
    chart_mode:             TokenDetailChartMode
    table_mode:             TokenDetailTableMode
    show_stats:             bool
    show_chart:             bool
    context:                TokenDetailUpdateContext

    @property
    def request_key(self) -> Hashable:
        return self.context.request_key


@dataclass(frozen=True)
class TokenDetailJob:
    kind:                   TokenDetailJobKind
    update:                 TokenDetailUpdate


@dataclass(frozen=True)
class TokenDetailRequest:
    request_key:            Hashable
    jobs:                   tuple[TokenDetailJob, ...]


@dataclass(frozen=True)
class TokenDetailTableResult:
    model:                  TokenDetailTableModel


@dataclass(frozen=True)
class TokenDetailHistogramResult:
    src:                    dict


TokenDetailResult = Union[TokenDetailTableResult, TokenDetailHistogramResult]


