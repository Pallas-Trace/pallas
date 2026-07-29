from dataclasses import dataclass
from collections import OrderedDict
from typing import Optional, Literal, TypeAlias

import numpy as np

FidelityMode:       TypeAlias = Literal["fast", "balanced", "exact"]
OccurrenceMode:     TypeAlias = Literal["all", "first", "last", "nth", "median"]
TokenMode:          TypeAlias = Literal["raw", "named"]

CATEGORY_TOKEN_TYPE = 3

# Data Storage Cache structure

class DataCache:
    def __init__(self, capacity: int = 32):
        self.capacity = capacity
        self._data = OrderedDict()

    def get(self, key):
        if key not in self._data:
            return None
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key, value):
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self.capacity:
            self._data.popitem(last=False)

    def clear(self):
        self._data.clear()

# Tier 1: Internal Meta Indices

@dataclass(frozen=True)
class TraceInfo:
    path:               str
    start_ns:           int
    end_ns:             int

    thread_ids:         np.ndarray      # int64
    thread_names:       np.ndarray      # object[str]
    token_types:        np.ndarray      # uint8
    token_ids:          np.ndarray      # int64
    token_names:        np.ndarray      # object[str]

    thread_name_to_id:  dict[str, int]
    thread_id_to_name:  dict[int, str]

    token_key_to_name:  dict[str, str]

    token_cat_remap:    dict[tuple[int, int], tuple[int, int]]
    cat_key_to_name:    dict[str, str]

# Tier 2: Abbreviated Data Summaries

@dataclass(frozen=True)
class TokenSummary:
    token_type:         int
    token_id:           int
    incl_total_ns:      int
    excl_total_ns:      int
    call_count:         int
    thread_ids:         tuple[int, ...]

@dataclass(frozen=True)
class TraceSummary:
    fidelity:           FidelityMode
    tokens:             tuple[TokenSummary, ...]
    top_tokens:         tuple[str, ...]

@dataclass(frozen=True)
class SummaryQuery:
    thread_ids:         tuple[int, ...]
    fidelity:           FidelityMode = "fast"
    token_mode:         TokenMode = "raw"
    top_k:              int = 32
    block_only:         bool = False

# Tier 3: Large Data Streams and Queries

# Quanta Data Path

@dataclass
class QuantaBundle:
    fidelity:           FidelityMode
    start_ns:           np.ndarray      # int64
    end_ns:             np.ndarray      # int64
    thread_id:          np.ndarray      # int64
    token_type:         np.ndarray      # uint8
    token_id:           np.ndarray      # int64
    excl_ns:            np.ndarray      # int64
    proportion:         np.ndarray      # float32/float64

@dataclass(frozen=True)
class QuantaQuery:
    thread_ids:         tuple[int, ...]
    bin_edges_ns:       tuple[int, ...]
    fidelity:           FidelityMode = "fast"
    token_mode:         TokenMode = "raw"
    top_k:              Optional[int] = None

def empty_quanta_bundle(fidelity: FidelityMode) -> QuantaBundle:
    empty_i64 = np.array([], dtype=np.int64)
    empty_u8 = np.array([], dtype=np.uint8)
    empty_f64 = np.array([], dtype=np.float64)
    return QuantaBundle(
        fidelity        = fidelity,
        start_ns        = empty_i64,
        end_ns          = empty_i64.copy(),
        thread_id       = empty_i64.copy(),
        token_type      = empty_u8,
        token_id        = empty_i64.copy(),
        excl_ns         = empty_i64.copy(),
        proportion      = empty_f64,
    )

def normalize_quanta_result(raw, fidelity: FidelityMode) -> QuantaBundle:
    n = len(raw.start_ns)
    if n == 0:
        return empty_quanta_bundle(fidelity)

    start_ns = np.asarray(raw.start_ns, dtype=np.int64)
    end_ns = np.asarray(raw.finish_ns, dtype=np.int64)
    thread_id = np.asarray(raw.thread_id, dtype=np.int64)
    token_type = np.asarray(raw.token_type, dtype=np.uint8)
    token_id = np.asarray(raw.token_id, dtype=np.int64)
    excl_ns = np.asarray(raw.excl_ns, dtype=np.int64)
    proportion = np.asarray(raw.proportion, dtype=np.float64)

    order = np.lexsort((token_id, token_type, thread_id, end_ns, start_ns))
    return QuantaBundle(
        fidelity        = fidelity,
        start_ns        = start_ns[order],
        end_ns          = end_ns[order],
        thread_id       = thread_id[order],
        token_type      = token_type[order],
        token_id        = token_id[order],
        excl_ns         = excl_ns[order],
        proportion      = proportion[order],
    )

def subset_quanta_bundle(bundle: QuantaBundle, keep: np.ndarray) -> QuantaBundle:
    return QuantaBundle(
        fidelity        = bundle.fidelity,
        start_ns        = bundle.start_ns[keep],
        end_ns          = bundle.end_ns[keep],
        thread_id       = bundle.thread_id[keep],
        token_type      = bundle.token_type[keep],
        token_id        = bundle.token_id[keep],
        excl_ns         = bundle.excl_ns[keep],
        proportion      = bundle.proportion[keep],
    )

def canonicalize_quanta_query(query: QuantaQuery) -> QuantaQuery:
    return QuantaQuery(
        thread_ids      = tuple(sorted(int(t) for t in query.thread_ids)),
        bin_edges_ns    = tuple(int(x) for x in query.bin_edges_ns),
        fidelity        = query.fidelity,
        token_mode      = query.token_mode,
        top_k           = None if query.top_k is None else int(query.top_k),
    )

# Span Data Path

@dataclass
class SpanBundle:
    fidelity:           FidelityMode
    thread_id:          np.ndarray      # int64
    token_type:         np.ndarray      # uint8
    token_id:           np.ndarray      # int64
    iteration:          np.ndarray      # int64
    depth:              np.ndarray      # int32/int64
    start_ns:           np.ndarray      # int64
    end_ns:             np.ndarray      # int64
    dur_ns:             np.ndarray      # int64
    excl_ns:            np.ndarray      # int64

@dataclass(frozen=True)
class SpanQuery:
    thread_ids:         tuple[int, ...]
    t0_ns:              int
    t1_ns:              int
    fidelity:           FidelityMode = "fast"
    token_mode:         TokenMode = "raw"
    max_depth:          Optional[int] = None
    token:              Optional[tuple[int, int]] = None

def empty_span_bundle(fidelity: FidelityMode) -> SpanBundle:
    empty_i64 = np.array([], dtype=np.int64)
    empty_u8 = np.array([], dtype=np.uint8)
    return SpanBundle(
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

def normalize_span_rows(
    rows: list[tuple[int, int, int, int, int, int, int, int, int]],
    fidelity: FidelityMode,
) -> SpanBundle:
    if not rows:
        return empty_span_bundle(fidelity)

    thread_id = np.array([r[0] for r in rows], dtype=np.int64)
    token_type = np.array([r[1] for r in rows], dtype=np.uint8)
    token_id = np.array([r[2] for r in rows], dtype=np.int64)
    iteration = np.array([r[3] for r in rows], dtype=np.int64)
    depth = np.array([r[4] for r in rows], dtype=np.int64)
    start_ns = np.array([r[5] for r in rows], dtype=np.int64)
    end_ns = np.array([r[6] for r in rows], dtype=np.int64)
    dur_ns = np.array([r[7] for r in rows], dtype=np.int64)
    excl_ns = np.array([r[8] for r in rows], dtype=np.int64)

    order = np.lexsort((iteration, -end_ns, start_ns, thread_id))
    return SpanBundle(
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

def subset_span_bundle(bundle: SpanBundle, keep: np.ndarray, fidelity: FidelityMode) -> SpanBundle:
    return SpanBundle(
        fidelity        = fidelity,
        thread_id       = bundle.thread_id[keep],
        token_type      = bundle.token_type[keep],
        token_id        = bundle.token_id[keep],
        iteration       = bundle.iteration[keep],
        depth           = bundle.depth[keep],
        start_ns        = bundle.start_ns[keep],
        end_ns          = bundle.end_ns[keep],
        dur_ns          = bundle.dur_ns[keep],
        excl_ns         = bundle.excl_ns[keep],
    )

def canonicalize_span_query(query: SpanQuery) -> SpanQuery:
    tok = None if query.token is None else (int(query.token[0]), int(query.token[1]))
    return SpanQuery(
        thread_ids      = tuple(sorted(int(t) for t in query.thread_ids)),
        t0_ns           = int(query.t0_ns),
        t1_ns           = int(query.t1_ns),
        fidelity        = query.fidelity,
        token_mode      = query.token_mode,
        max_depth       = None if query.max_depth is None else int(query.max_depth),
        token           = tok,
    )

# Occurence Data Path

@dataclass(frozen=True)
class OccurrenceQuery:
    thread_ids:         tuple[int, ...]
    token:              tuple[int, int]
    fidelity:           FidelityMode = "exact"
    token_mode:         TokenMode = "raw"
    t0_ns:              Optional[int] = None
    t1_ns:              Optional[int] = None
    max_depth:          Optional[int] = None
    mode:               OccurrenceMode = "all"
    index:              Optional[int] = None

def canonicalize_occurrence_query(query: OccurrenceQuery) -> OccurrenceQuery:
    return OccurrenceQuery(
        thread_ids      = tuple(sorted(int(t) for t in query.thread_ids)),
        token           = (int(query.token[0]), int(query.token[1])),
        fidelity        = query.fidelity,
        token_mode      = query.token_mode,
        t0_ns           = None if query.t0_ns is None else int(query.t0_ns),
        t1_ns           = None if query.t1_ns is None else int(query.t1_ns),
        max_depth       = None if query.max_depth is None else int(query.max_depth),
        mode            = query.mode,
        index           = None if query.index is None else int(query.index),
    )

# Subtree Data Path

@dataclass(frozen=True)
class NodeRef:
    thread_id:          int
    token_type:         int
    token_id:           int
    iteration:          int
    depth:              int
    start_ns:           int
    end_ns:             int

@dataclass(frozen=True)
class SubtreeQuery:
    root:               NodeRef
    fidelity:           FidelityMode = "fast"
    max_depth:          Optional[int] = None
    normalize_time:     bool = False

def canonicalize_subtree_query(query: SubtreeQuery) -> SubtreeQuery:
    root = NodeRef(
        thread_id       = int(query.root.thread_id),
        token_type      = int(query.root.token_type),
        token_id        = int(query.root.token_id),
        iteration       = int(query.root.iteration),
        depth           = int(query.root.depth),
        start_ns        = int(query.root.start_ns),
        end_ns          = int(query.root.end_ns),
    )
    return SubtreeQuery(
        root            = root,
        fidelity        = query.fidelity,
        max_depth       = None if query.max_depth is None else int(query.max_depth),
        normalize_time  = bool(query.normalize_time),
    )

@dataclass(frozen=True)
class SnapshotHistogram:
    left_ns: tuple[int, ...]
    right_ns: tuple[int, ...]
    excl_ns: tuple[int, ...]

@dataclass(frozen=True)
class SnapshotHistogramQuery:
    thread_ids: tuple[int, ...]
    token: tuple[int, int]
    t0_ns: int
    t1_ns: int
    n_bins: int = 20
    token_mode: TokenMode = "raw"

def canonicalize_histogram_query(query: SnapshotHistogramQuery) -> SnapshotHistogramQuery:
    return SnapshotHistogramQuery(
        thread_ids      = tuple(sorted(int(t) for t in query.thread_ids)),
        token           = (int(query.token[0]), int(query.token[1])),
        t0_ns           = int(query.t0_ns),
        t1_ns           = int(query.t1_ns),
        n_bins          = int(query.n_bins),
        token_mode      =query.token_mode,
    )

# Misc Data Model helper functions

def as_token_key(token_type: int, token_id: int) -> str:
    return f"{token_type}:{token_id}"

def token_int_id(tok) -> int:
    try:
        return int(tok.id)
    except Exception:
        pass
    try:
        return int(tok.id.id)
    except Exception:
        pass
    return -1

