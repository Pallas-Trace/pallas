from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias, get_args

import numpy as np
import numpy.typing as npt

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
)

# -------------------------------------------
# |           Domain Vocabulary             |
# -------------------------------------------

type FidelityMode       = Literal["fast", "balanced", "exact"]
type OccurrenceMode     = Literal["all", "first", "last", "nth", "median"]
type TokenMode          = Literal["raw", "named"]

# synthetic 'category' token type (TokenType = 3)
CATEGORY_TOKEN_TYPE:    TokenType = 3

# synthetic aggregated 'other' bucket token
OTHER_TOKEN_TYPE:       TokenType = 255
OTHER_TOKEN_ID:         TokenID = 0
OTHER_TOKEN_KEY:        TokenKey = (OTHER_TOKEN_TYPE, OTHER_TOKEN_ID)
OTHER_TOKEN_KEY_STR:    TokenKeyStr = "OTHER"
OTHER_TOKEN_NAME:       str = "OTHER"

# -------------------------------------------
# |         Tier 1: Trace Metadata          |
# -------------------------------------------

@dataclass(frozen=True)
class TraceMeta:
    path:               str
    start_ns:           TimestampNS
    end_ns:             TimestampNS

    thread_ids:         npt.NDArray[np.int64]
    thread_names:       npt.NDArray[np.object_]

    token_types:        npt.NDArray[np.uint8]
    token_ids:          npt.NDArray[np.int64]
    token_names:        npt.NDArray[np.object_]

    thread_name_to_id:  dict[ThreadName, ThreadID]
    thread_id_to_name:  dict[ThreadID, ThreadName]

    token_key_to_name:  dict[TokenKey, TokenName]

    token_cat_remap:    dict[TokenKey, TokenKey]
    cat_key_to_name:    dict[TokenKey, TokenName]

# -------------------------------------------
# |         Tier 2: Data Summaries          |
# -------------------------------------------

# >> query
@dataclass(frozen=True)
class SummaryQuery:
    thread_ids:         tuple[ThreadID, ...]
    fidelity:           FidelityMode = "fast"
    token_mode:         TokenMode = "raw"
    top_k:              int | None = 32     # none => no-limit
    block_only:         bool = False

    def canonicalize(self) -> SummaryQuery:
        top_k = self.top_k
        return SummaryQuery(
            thread_ids      = _normalize_thread_ids(self.thread_ids),
            fidelity        = _validate_fidelity(self.fidelity),
            token_mode      = _validate_token_mode(self.token_mode),
            top_k           = _normalize_count(self.top_k, name="top_k"),
            block_only      = bool(self.block_only),
        )

# >> result
@dataclass(frozen=True)
class TokenSummary:
    token_type:         TokenType
    token_id:           TokenID
    incl_total_ns:      DurationNS
    excl_total_ns:      DurationNS
    call_count:         int
    thread_ids:         tuple[ThreadID, ...]

@dataclass(frozen=True)
class SummaryBundle:
    fidelity:           FidelityMode
    tokens:             tuple[TokenSummary, ...]
    top_tokens:         tuple[TokenKey, ...]

    @classmethod
    def empty(cls, fidelity: FidelityMode) -> SummaryBundle:
        return cls(
            fidelity        = fidelity,
            tokens          = (),
            top_tokens      = (),
        )

# -------------------------------------------
# |       Tier 3: Bulk Data Streams         |
# -------------------------------------------

# Quanta Data Path
# ----------------

# >> query
@dataclass(frozen=True)
class QuantaQuery:
    thread_ids:         tuple[ThreadID, ...]
    bin_edges_ns:       tuple[TimestampNS, ...]
    fidelity:           FidelityMode = "fast"
    token_mode:         TokenMode = "raw"
    top_k:              int | None = None   # none => no-limit

    def canonicalize(self) -> QuantaQuery:
        bin_edges = tuple(int(x) for x in self.bin_edges_ns)
        for b0, b1 in zip(bin_edges, bin_edges[1:]):
            if b1 < b0:
                raise ValueError(
                    f"bin_edges_ns must be non-decreasing, got {bin_edges}"
                )
        return QuantaQuery(
            thread_ids      = _normalize_thread_ids(self.thread_ids),
            bin_edges_ns    = bin_edges,
            fidelity        = _validate_fidelity(self.fidelity),
            token_mode      = _validate_token_mode(self.token_mode),
            top_k           = _normalize_count(self.top_k, name="top_k"),
        )

# >> result
@dataclass
class QuantaBundle:
    fidelity:           FidelityMode
    start_ns:           npt.NDArray[np.int64]
    end_ns:             npt.NDArray[np.int64]
    thread_id:          npt.NDArray[np.int64]
    token_type:         npt.NDArray[np.uint8]
    token_id:           npt.NDArray[np.int64]
    excl_ns:            npt.NDArray[np.int64]
    proportion:         npt.NDArray[np.float64]

    @classmethod
    def empty(cls, fidelity: FidelityMode) -> QuantaBundle:
        empty_i64 = np.array([], dtype=np.int64)
        empty_u8 = np.array([], dtype=np.uint8)
        empty_f64 = np.array([], dtype=np.float64)
        return cls(
            fidelity        = fidelity,
            start_ns        = empty_i64,
            end_ns          = empty_i64.copy(),
            thread_id       = empty_i64.copy(),
            token_type      = empty_u8,
            token_id        = empty_i64.copy(),
            excl_ns         = empty_i64.copy(),
            proportion      = empty_f64,
        )

    @classmethod
    def from_pallas(cls, raw, fidelity: FidelityMode) -> QuantaBundle:
        n = len(raw.start_ns)
        if n == 0:
            return QuantaBundle.empty(fidelity)

        start_ns = np.asarray(raw.start_ns, dtype=np.int64)
        end_ns = np.asarray(raw.finish_ns, dtype=np.int64)
        thread_id = np.asarray(raw.thread_id, dtype=np.int64)
        token_type = np.asarray(raw.token_type, dtype=np.uint8)
        token_id = np.asarray(raw.token_id, dtype=np.int64)
        excl_ns = np.asarray(raw.excl_ns, dtype=np.int64)
        proportion = np.asarray(raw.proportion, dtype=np.float64)

        order = np.lexsort((token_id, token_type, thread_id, end_ns, start_ns))
        return cls(
            fidelity        = fidelity,
            start_ns        = start_ns[order],
            end_ns          = end_ns[order],
            thread_id       = thread_id[order],
            token_type      = token_type[order],
            token_id        = token_id[order],
            excl_ns         = excl_ns[order],
            proportion      = proportion[order],
        )

    def subset(self, keep: np.ndarray) -> QuantaBundle:
        return QuantaBundle(
            fidelity        = self.fidelity,
            start_ns        = self.start_ns[keep],
            end_ns          = self.end_ns[keep],
            thread_id       = self.thread_id[keep],
            token_type      = self.token_type[keep],
            token_id        = self.token_id[keep],
            excl_ns         = self.excl_ns[keep],
            proportion      = self.proportion[keep],
        )

# Span Data Path
# ----------------

# >> query
@dataclass(frozen=True)
class SpanQuery:
    thread_ids:         tuple[ThreadID, ...]
    t0_ns:              TimestampNS
    t1_ns:              TimestampNS
    fidelity:           FidelityMode = "fast"
    token_mode:         TokenMode = "raw"
    max_depth:          int | None = None
    token:              TokenKey | None = None

    def canonicalize(self) -> SpanQuery:
        t0, t1 = _validate_window(self.t0_ns, self.t1_ns)
        return SpanQuery(
            thread_ids      = _normalize_thread_ids(self.thread_ids),
            t0_ns           = t0,
            t1_ns           = t1,
            fidelity        = _validate_fidelity(self.fidelity),
            token_mode      = _validate_token_mode(self.token_mode),
            max_depth       = _normalize_count(self.max_depth, name="max_depth"),
            token           = _normalize_optional_token(self.token),
        )

# >> result
@dataclass
class SpanBundle:
    fidelity:           FidelityMode
    thread_id:          npt.NDArray[np.int64]
    token_type:         npt.NDArray[np.uint8]
    token_id:           npt.NDArray[np.int64]
    iteration:          npt.NDArray[np.int64]
    depth:              npt.NDArray[np.int64]
    start_ns:           npt.NDArray[np.int64]
    end_ns:             npt.NDArray[np.int64]
    dur_ns:             npt.NDArray[np.int64]
    excl_ns:            npt.NDArray[np.int64]

    @classmethod
    def empty(cls, fidelity: FidelityMode) -> SpanBundle:
        empty_i64 = np.array([], dtype=np.int64)
        empty_u8 = np.array([], dtype=np.uint8)
        return cls(
            fidelity        = fidelity,
            thread_id       = empty_i64,
            token_type      = empty_u8,
            token_id        = empty_i64.copy(),
            iteration       = empty_i64.copy(),
            depth           = empty_i64.copy(),
            start_ns        = empty_i64.copy(),
            end_ns          = empty_i64.copy(),
            dur_ns          = empty_i64.copy(),
            excl_ns         = empty_i64.copy(),
        )

    @classmethod
    def from_rows(
        cls,
        raw: list[tuple[int, int, int, int, int, int, int, int, int]],
        fidelity: FidelityMode,
    ) -> SpanBundle:
        if not raw:
            return SpanBundle.empty(fidelity)

        thread_id = np.array([r[0] for r in raw], dtype=np.int64)
        token_type = np.array([r[1] for r in raw], dtype=np.uint8)
        token_id = np.array([r[2] for r in raw], dtype=np.int64)
        iteration = np.array([r[3] for r in raw], dtype=np.int64)
        depth = np.array([r[4] for r in raw], dtype=np.int64)
        start_ns = np.array([r[5] for r in raw], dtype=np.int64)
        end_ns = np.array([r[6] for r in raw], dtype=np.int64)
        dur_ns = np.array([r[7] for r in raw], dtype=np.int64)
        excl_ns = np.array([r[8] for r in raw], dtype=np.int64)

        order = np.lexsort((iteration, -end_ns, start_ns, thread_id))
        return cls(
            fidelity        = fidelity,
            thread_id       = thread_id[order],
            token_type      = token_type[order],
            token_id        = token_id[order],
            iteration       = iteration[order],
            depth           = depth[order],
            start_ns        = start_ns[order],
            end_ns          = end_ns[order],
            dur_ns          = dur_ns[order],
            excl_ns         = excl_ns[order],
        )

    def subset(self, keep: np.ndarray) -> SpanBundle:
        return SpanBundle(
            fidelity        = self.fidelity,
            thread_id       = self.thread_id[keep],
            token_type      = self.token_type[keep],
            token_id        = self.token_id[keep],
            iteration       = self.iteration[keep],
            depth           = self.depth[keep],
            start_ns        = self.start_ns[keep],
            end_ns          = self.end_ns[keep],
            dur_ns          = self.dur_ns[keep],
            excl_ns         = self.excl_ns[keep],
        )

# Occurence Data Path
# ----------------

# >> query
@dataclass(frozen=True)
class OccurrenceQuery:
    thread_ids:         tuple[ThreadID, ...]
    token:              TokenKey
    fidelity:           FidelityMode = "exact"
    token_mode:         TokenMode = "raw"
    t0_ns:              TimestampNS | None = None
    t1_ns:              TimestampNS | None = None
    max_depth:          int | None = None
    mode:               OccurrenceMode = "all"
    index:              int | None = None
    max_points:         int | None = None   # none => no-limit

    def canonicalize(self) -> OccurrenceQuery:
        if self.mode not in _OCCURRENCE_MODES:
            raise ValueError(f"invalid occurrence mode: {self.mode!r}")
        if self.mode == "nth" and self.index is None:
            raise ValueError("occurrence mode='nth' requires index")
        index = None if self.index is None else int(self.index)
        if index is not None and index < 0:
            raise ValueError(f"index must be >= 0, got {index}")
        t0, t1 = _validate_optional_window(self.t0_ns, self.t1_ns)
        return OccurrenceQuery(
            thread_ids      = _normalize_thread_ids(self.thread_ids),
            token           = _normalize_token(self.token),
            fidelity        = _validate_fidelity(self.fidelity),
            token_mode      = _validate_token_mode(self.token_mode),
            t0_ns           = t0,
            t1_ns           = t1,
            max_depth       = _normalize_count(self.max_depth, name="max_depth"),
            mode            = self.mode,
            index           = index,
            max_points      = _normalize_count(self.max_points, name="max_points"),
        )


# >> result
@dataclass
class OccurrenceBundle:
    fidelity:           FidelityMode
    thread_id:          npt.NDArray[np.int64]
    iteration:          npt.NDArray[np.int64]
    start_ns:           npt.NDArray[np.int64]
    dur_ns:             npt.NDArray[np.int64]

    @classmethod
    def empty(cls, fidelity: FidelityMode) -> OccurrenceBundle:
        empty_i64 = np.array([], dtype=np.int64)
        return cls(
            fidelity        = fidelity,
            thread_id       = empty_i64,
            iteration       = empty_i64.copy(),
            start_ns        = empty_i64.copy(),
            dur_ns          = empty_i64.copy(),
        )

    # TODO: add if necessary
    @classmethod
    def from_pallas(cls, raw, fidelity: FidelityMode) -> OccurrenceBundle:
        raise NotImplementedError

    def subset(self, keep: np.ndarray) -> OccurrenceBundle:
        return OccurrenceBundle(
            fidelity        = self.fidelity,
            thread_id       = self.thread_id[keep],
            iteration       = self.iteration[keep],
            start_ns        = self.start_ns[keep],
            dur_ns          = self.dur_ns[keep],
        )


# Subtree Data Path
# ----------------

# >> query
@dataclass(frozen=True)
class NodeRef:
    thread_id:          ThreadID
    token_type:         TokenType
    token_id:           TokenID
    iteration:          int
    depth:              int
    start_ns:           TimestampNS
    end_ns:             TimestampNS

@dataclass(frozen=True)
class SubtreeQuery:
    root:               NodeRef
    fidelity:           FidelityMode = "fast"
    max_depth:          int | None = None
    normalize_time:     bool = False

    def canonicalize(self) -> SubtreeQuery:
        root_start, root_end = _validate_window(
            self.root.start_ns, self.root.end_ns
        )
        root = NodeRef(
            thread_id       = int(self.root.thread_id),
            token_type      = int(self.root.token_type),
            token_id        = int(self.root.token_id),
            iteration       = int(self.root.iteration),
            depth           = int(self.root.depth),
            start_ns        = root_start,
            end_ns          = root_end,
        )
        return SubtreeQuery(
            root            = root,
            fidelity        = _validate_fidelity(self.fidelity),
            max_depth       = _normalize_count(self.max_depth, name="max_depth"),
            normalize_time  = bool(self.normalize_time),
        )

# >> result
# NOTE: currently returns SpanBundle

# Snapshot-Histogram Data Path
# ----------------

# >> query
@dataclass(frozen=True)
class HistogramQuery:
    thread_ids:         tuple[ThreadID, ...]
    token:              TokenKey
    t0_ns:              TimestampNS
    t1_ns:              TimestampNS
    n_bins:             int = 20
    token_mode:         TokenMode = "raw"
    fidelity:           FidelityMode = "fast"

    def canonicalize(self) -> HistogramQuery:
        if int(self.n_bins) < 1:
            raise ValueError(
                f"histogram requires n_bins >= 1, got {self.n_bins}"
            )
        t0, t1 = _validate_window(self.t0_ns, self.t1_ns)
        return HistogramQuery(
            thread_ids      = _normalize_thread_ids(self.thread_ids),
            token           = _normalize_token(self.token),
            t0_ns           = t0,
            t1_ns           = t1,
            n_bins          = int(self.n_bins),
            token_mode      = _validate_token_mode(self.token_mode),
            fidelity        = _validate_fidelity(self.fidelity),
        )

# >> result
@dataclass(frozen=True)
class HistogramBundle:
    left_ns:            tuple[TimestampNS, ...]
    right_ns:           tuple[TimestampNS, ...]
    excl_ns:            tuple[DurationNS, ...]

    @classmethod
    def empty(cls) -> HistogramBundle:
        return cls(
            left_ns = (),
            right_ns = (),
            excl_ns = (),
        )

# -------------------------------------------
# |     Validation/Normalization Helpers    |
# -------------------------------------------

# some cursed type reflection because Python is silly
def _literal_args(alias) -> tuple:
    return get_args(getattr(alias, "__value__", alias))

_FIDELITIES: tuple = _literal_args(FidelityMode)
_TOKEN_MODES: tuple = _literal_args(TokenMode)
_OCCURRENCE_MODES: tuple = _literal_args(OccurrenceMode)

def _normalize_thread_ids(thread_ids) -> tuple[ThreadID, ...]:
    return tuple(sorted(int(t) for t in thread_ids))

def _validate_fidelity(fidelity) -> FidelityMode:
    if fidelity not in _FIDELITIES:
        raise ValueError(f"invalid fidelity: {fidelity!r}")
    return fidelity

def _validate_token_mode(token_mode) -> TokenMode:
    if token_mode not in _TOKEN_MODES:
        raise ValueError(f"invalid token_mode: {token_mode!r}")
    return token_mode

def _normalize_count(value, *, name: str) -> int | None:
    # NOTE: none => no-limt; 0 => valid (but empty); <0 => error
    if value is None:
        return None
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be None or >= 0, got {value}")
    return value

def _normalize_token(token) -> TokenKey:
    if token is None:
        raise ValueError("query requires a token")
    token_type, token_id = int(token[0]), int(token[1])
    if not 0 <= token_type <= 255:
        raise ValueError(f"token_type out of uint8 range: {token_type}")
    if token_id < 0:
        raise ValueError(f"token_id must be >= 0, got {token_id}")
    return (token_type, token_id)

def _normalize_optional_token(token) -> TokenKey | None:
    if token is None:
        return None
    return _normalize_token(token)

def _validate_window(t0_ns, t1_ns) -> tuple[TimestampNS, TimestampNS]:
    # NOTE: t0 == t1 => valid (but empty); t1 < t0 => error
    t0, t1 = int(t0_ns), int(t1_ns)
    if t1 < t0:
        raise ValueError(f"inverted time window: t0_ns={t0} > t1_ns={t1}")
    return t0, t1

def _validate_optional_window(t0_ns, t1_ns):
    t0 = None if t0_ns is None else int(t0_ns)
    t1 = None if t1_ns is None else int(t1_ns)
    if t0 is not None and t1 is not None and t1 < t0:
        raise ValueError(f"inverted time window: t0_ns={t0} > t1_ns={t1}")
    return t0, t1


