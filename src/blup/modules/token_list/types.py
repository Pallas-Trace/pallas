from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Hashable, Literal, Protocol, Sequence

from blup.data_model import FidelityMode, SummaryQuery, TokenMode, TraceSummary
from blup.state import TraceMode


# TODO: add these to state.py
TokenListOrder = Literal["delta", "excl", "calls", "name", "token"]
SortDirection = Literal["asc", "desc"]

# -----------------------------------------

@dataclass(frozen=True)
class TokenListRow:
    token_type: int
    token_id: int
    token_key: str
    name: str
    call_count_upper: int
    call_count_lower: int
    excl_total_ns_upper: int
    excl_total_ns_lower: int
    delta_excl_total_ns: int
    contribution_rank: int
    contribution_share_pct: float

    @property
    def token(self) -> tuple[int, int]:
        return (self.token_type, self.token_id)

# NOTE: the above ^^ should be considered temporary and
#       later modified or moved as appropraite
# -----------------------------------------


@dataclass(frozen=True)
class TokenListTraceContext:
    label:                  str
    thread_name_to_id:      dict[str, int]
    token_name_by_key:      dict[str, str]
    summarize_tokens:       Callable[[SummaryQuery], TraceSummary]


@dataclass(frozen=True)
class TokenListUpdateContext:
    request_key:            Hashable
    trace_context:          dict[TraceSide, TokenListTraceContext]
    fidelity:               FidelityMode
    token_mode:             TokenMode
    top_k:                  int
    order:                  TokenListOrder
    direction:              SortDirection
    color_map:              dict[str, str]


@dataclass(frozen=True)
class TokenListUpdate:
    active_thread_names:    tuple[str, ...]
    selected_token:         tuple[int, int] | None
    trace_mode:             TraceMode
    context:                TokenListUpdateContext

    @property
    def request_key(self) -> Hashable:
        return self.context.request_key


@dataclass(frozen=True)
class TokenListJob:
    update:                 TokenListUpdate


@dataclass(frozen=True)
class TokenListRequest:
    request_key:            Hashable
    jobs:                   tuple[TokenListJob, ...]


@dataclass(frozen=True)
class TokenListResult:
    src:                    dict


