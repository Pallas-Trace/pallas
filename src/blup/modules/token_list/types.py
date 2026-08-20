from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Hashable, Literal, Protocol, Sequence

from blup.data_model import FidelityMode, SummaryQuery, TokenMode, SummaryBundle
from blup.state import TokenListOrder, TokenListSortDirection
from blup.types import (
    ColorHex,
    DurationNS,
    ThreadID,
    ThreadName,
    TokenID,
    TokenKey,
    TokenKeyStr,
    TokenName,
    TokenType,
    TraceMode,
    TraceSide,
)

# -----------------------------------------

@dataclass(frozen=True)
class TokenListRow:
    token_type:             TokenType
    token_id:               TokenID
    token_key:              TokenKey
    name:                   TokenName
    call_count_upper:       int
    call_count_lower:       int
    excl_total_ns_upper:    DurationNS
    excl_total_ns_lower:    DurationNS
    delta_excl_total_ns:    DurationNS
    contribution_rank:      int
    contribution_share_pct: float

    @property
    def token(self) -> TokenKey:
        return self.token_key

# NOTE: the above ^^ should be considered temporary and
#       later modified or moved as appropraite
# -----------------------------------------


@dataclass(frozen=True)
class TokenListTraceContext:
    label:                  str
    thread_name_to_id:      dict[ThreadName, ThreadID]
    token_name_by_key:      dict[TokenKey, TokenName]
    summarize_tokens:       Callable[[SummaryQuery], SummaryBundle]


@dataclass(frozen=True)
class TokenListUpdateContext:
    request_key:            Hashable
    trace_context:          dict[TraceSide, TokenListTraceContext]
    fidelity:               FidelityMode
    token_mode:             TokenMode
    top_k:                  int | None
    order:                  TokenListOrder
    direction:              TokenListSortDirection
    color_map:              dict[TokenKey, ColorHex]


@dataclass(frozen=True)
class TokenListUpdate:
    active_thread_names:    tuple[ThreadName, ...]
    selected_token:         TokenKey | None
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


