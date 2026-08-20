from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Hashable, Literal, Protocol, Sequence, Union

from blup.data_model import (
    FidelityMode,
    HistogramBundle,
    HistogramQuery,
    OccurrenceBundle,
    OccurrenceQuery,
    SummaryQuery,
    TokenMode,
    SummaryBundle,
)
from blup.state import TokenDetailChartMode
from blup.types import (
    DurationNS,
    ThreadID,
    ThreadName,
    TimestampNS,
    TokenID,
    TokenKey,
    TokenKeyStr,
    TokenName,
    TokenType,
    TraceMode,
    TraceSide,
)


type TokenDetailJobKind = Literal["table", "histogram", "scatter"]

# -----------------------------------------

# NOTE: this functionality should be centralized elsewhere
def token_to_select_value(token: TokenKey | None) -> TokenKeyStr:
    if token is None:
        return ""
    return f"{token[0]}:{token[1]}"

def select_value_to_token(value: str) -> TokenKey | None:
    if not value:
        return None
    raw_type, raw_id = value.split(":", 1)
    return (int(raw_type), int(raw_id))


@dataclass(frozen=True)
class TokenDetailDiffRow:
    token_type:             TokenType
    token_id:               TokenID
    name:                   TokenName
    call_count_upper:       int
    call_count_lower:       int
    incl_total_ns_upper:    DurationNS
    incl_total_ns_lower:    DurationNS
    excl_total_ns_upper:    DurationNS
    excl_total_ns_lower:    DurationNS
    mean_incl_ns_upper:     float
    mean_incl_ns_lower:     float
    mean_excl_ns_upper:     float
    mean_excl_ns_lower:     float
    delta_call_count:       int
    delta_mean_incl_ns:     float
    delta_mean_excl_ns:     float
    delta_incl_total_ns:    DurationNS
    delta_excl_total_ns:    DurationNS
    thread_ids_upper:       tuple[ThreadID, ...]
    thread_ids_lower:       tuple[ThreadID, ...]
    contribution_abs_ns:    int = 0
    contribution_share_pct: float = 0.0
    contribution_rank:      int = 0

    @property
    def token(self) -> TokenKey:
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

# NOTE: the above ^^ should be considered temporary and
#       later modified or moved as appropraite
# -----------------------------------------


@dataclass(frozen=True)
class TokenDetailTraceContext:
    label:                  str
    # NOTE: check this param ^^
    thread_name_to_id:      dict[ThreadName, ThreadID]
    token_name_by_key:      dict[TokenKey, TokenName]
    summarize_tokens:       Callable[[SummaryQuery], SummaryBundle]
    query_histogram:        Callable[[HistogramQuery], HistogramBundle]
    query_occurrences:      Callable[[OccurrenceQuery], OccurrenceBundle]


@dataclass(frozen=True)
class TokenDetailUpdateContext:
    request_key:            Hashable
    trace_context:          dict[TraceSide, TokenDetailTraceContext]
    fidelity:               FidelityMode
    token_mode:             TokenMode
    top_k:                  int | None
    histogram_bins:         int


@dataclass(frozen=True)
class TokenDetailUpdate:
    active_thread_names:    tuple[ThreadName, ...]
    start_ns:               TimestampNS
    end_ns:                 TimestampNS
    selected_token:         TokenKey | None
    trace_mode:             TraceMode
    chart_mode:             TokenDetailChartMode
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
    trace_side:             TraceSide | None = None


@dataclass(frozen=True)
class TokenDetailRequest:
    request_key:            Hashable
    jobs:                   tuple[TokenDetailJob, ...]


@dataclass(frozen=True)
class TokenDetailTableResult:
    model:                  TokenDetailTableModel


@dataclass(frozen=True)
class TokenDetailHistogramResult:
    trace_side:             TraceSide
    # ~~~
    src:                    dict


@dataclass(frozen=True)
class TokenDetailScatterResult:
    trace_side:             TraceSide
    # ~~~
    src:                    dict


TokenDetailResult = Union[
    TokenDetailTableResult,
    TokenDetailHistogramResult,
    TokenDetailScatterResult,
]


