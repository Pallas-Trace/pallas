from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from typing import Optional

import numpy as np
import numpy.typing as npt
import pallas_trace as pallas

from blup.data_model import (
    CATEGORY_TOKEN_TYPE,
    DataCache,
    FidelityMode,
    NodeRef,
    OccurrenceQuery,
    QuantaBundle,
    QuantaQuery,
    SnapshotHistogram,
    SnapshotHistogramQuery,
    SpanBundle,
    SpanQuery,
    SubtreeQuery,
    SummaryQuery,
    TokenMode,
    TokenSummary,
    TraceMeta,
    TraceSummary,
    as_token_key,
    canonicalize_histogram_query,
    canonicalize_occurrence_query,
    canonicalize_quanta_query,
    canonicalize_span_query,
    canonicalize_subtree_query,
    empty_quanta_bundle,
    empty_span_bundle,
    normalize_quanta_result,
    normalize_span_rows,
    subset_span_bundle,
    token_int_id,
)
from blup.utils import timed


def sequence_block_depth(reader: pallas.ThreadReader) -> int:
    return len([
        t for (t, _) in reader.callstack[:-1]
        if isinstance(t, pallas.Sequence)
        and t.type == pallas.SequenceType.SEQUENCE_BLOCK
    ])


class TraceSession:
    def __init__(
        self,
        path: str,
        *,
        max_summary_cache: int = 16,
        max_quanta_cache: int = 32,
        max_span_cache: int = 32,
        max_occ_cache: int = 64,
        max_subtree_cache: int = 32,
        max_hist_cache: int = 32,
    ) -> None:
        self.path = path

        self._trace: Optional[pallas.Trace] = None
        self._meta: Optional[TraceMeta] = None
        self._summary: Optional[TraceSummary] = None

        self._summary_cache = DataCache(max_summary_cache)
        self._quanta_cache = DataCache(max_quanta_cache)
        self._span_cache = DataCache(max_span_cache)
        self._occ_cache = DataCache(max_occ_cache)
        self._subtree_cache = DataCache(max_subtree_cache)
        self._hist_cache = DataCache(max_hist_cache)

    # ---------- lifecycle ----------

    def open(self) -> None:
        if self._trace is not None:
            return

        with timed("pallas.open_trace"):
            self._trace = pallas.open_trace(self.path)

        with timed("build_metadata"):
            self._meta = self.build_metadata()

        assert self._meta is not None
        with timed("validate_trace"):
            self.validate_trace(self._meta)

        summary_query = SummaryQuery(
            thread_ids=tuple(int(id) for id in self._meta.thread_ids),
            fidelity="fast",
            top_k=32,
        )
        with timed("summarize_tokens"):
            self._summary = self.summarize_tokens(summary_query)

    def close(self) -> None:
        self._trace = None
        self._meta = None
        self._summary = None
        self.clear_caches()

    def clear_caches(self) -> None:
        self._summary_cache.clear()
        self._quanta_cache.clear()
        self._span_cache.clear()
        self._occ_cache.clear()
        self._subtree_cache.clear()
        self._hist_cache.clear()

    # ---------- metadata ----------

    @property
    def meta(self) -> TraceMeta:
        if self._meta is None:
            self.open()
        assert self._meta is not None
        return self._meta

    @property
    def summary(self) -> TraceSummary:
        if self._summary is None:
            self.open()
        assert self._summary is not None
        return self._summary

    def build_metadata(self) -> TraceMeta:
        assert self._trace is not None

        thread_rows: list[tuple[int, str, int, int]] = []
        for archive in self._trace.archives:
            for thread in archive.threads:
                tid = int(thread.id)
                tname = str(self._trace.locations[thread.id].name)
                t0 = int(thread.starting_timestamp)
                t1 = int(thread.finish_timestamp)
                thread_rows.append((tid, tname, t0, t1))

        thread_rows.sort(key=lambda x: x[1])
        thread_ids = np.array([r[0] for r in thread_rows], dtype=np.int64)
        thread_names = np.array([r[1] for r in thread_rows], dtype=object)

        token_res = self._trace.tokens()
        token_types = np.asarray(token_res.token_type, dtype=np.uint8)
        token_ids = np.asarray(token_res.token_id, dtype=np.int64)
        token_names = np.asarray([str(x) for x in token_res.display_name], dtype=object)

        raw_token_key_to_name = {
            as_token_key(int(ttype), int(tid)): str(name)
            for ttype, tid, name in zip(token_types, token_ids, token_names)
        }

        name_to_members: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for ttype, tid, name in zip(token_types, token_ids, token_names):
            name_to_members[str(name)].append((int(ttype), int(tid)))

        token_remap: dict[tuple[int, int], tuple[int, int]] = {}
        category_key_to_name: dict[str, str] = {}

        for cat_id, name in enumerate(sorted(name_to_members.keys())):
            cat_tok = (CATEGORY_TOKEN_TYPE, int(cat_id))
            cat_key = as_token_key(*cat_tok)
            category_key_to_name[cat_key] = name
            for src_tok in name_to_members[name]:
                token_remap[src_tok] = cat_tok

        token_key_to_name = dict(raw_token_key_to_name)
        token_key_to_name.update(category_key_to_name)

        return TraceMeta(
            path                = self.path,
            start_ns            = min(r[2] for r in thread_rows),
            end_ns              = max(r[3] for r in thread_rows),
            thread_ids          = thread_ids,
            thread_names        = thread_names,
            token_types         = token_types,
            token_ids           = token_ids,
            token_names         = token_names,
            thread_name_to_id   = {
                str(name): int(tid)
                for tid, name in zip(thread_ids, thread_names)
            },
            thread_id_to_name   = {
                int(tid): str(name)
                for tid, name in zip(thread_ids, thread_names)
            },
            token_key_to_name   = token_key_to_name,
            token_cat_remap     = token_remap,
            cat_key_to_name     = category_key_to_name,
        )

    def install_category_namespace(
        self,
        name_to_cat_token: dict[str, tuple[int, int]],
    ) -> None:
        meta = self.meta

        raw_key_to_name = {
            as_token_key(int(tt), int(tid)): str(name)
            for tt, tid, name in zip(meta.token_types, meta.token_ids, meta.token_names)
        }

        token_cat_remap = {
            (int(tt), int(tid)): name_to_cat_token[str(name)]
            for tt, tid, name in zip(meta.token_types, meta.token_ids, meta.token_names)
        }

        cat_key_to_name = {
            as_token_key(tok_type, tok_id): name
            for name, (tok_type, tok_id) in name_to_cat_token.items()
        }

        self._meta = replace(
            meta,
            token_key_to_name = raw_key_to_name | cat_key_to_name,
            token_cat_remap   = token_cat_remap,
            cat_key_to_name   = cat_key_to_name,
        )
        self.clear_caches()

    def validate_trace(self, meta: TraceMeta) -> None:
        thread_ids = [int(x) for x in meta.thread_ids]
        thread_names = [str(x) for x in meta.thread_names]
        token_keys = [
            as_token_key(int(ttype), int(tid))
            for ttype, tid in zip(meta.token_types, meta.token_ids)
        ]
        token_names = [str(x) for x in meta.token_names]

        def search_dupl_items(xs):
            counts = Counter(xs)
            return [x for x, n in counts.items() if n > 1]

        dup_thread_ids = search_dupl_items(thread_ids)
        if dup_thread_ids:
            raise RuntimeError(
                f"trace validation failed: duplicate thread ids: {dup_thread_ids[:8]}"
            )

        dup_thread_names = search_dupl_items(thread_names)
        if dup_thread_names:
            raise RuntimeError(
                f"trace validation failed: duplicate thread names: {dup_thread_names[:8]}"
            )

        dup_token_keys = search_dupl_items(token_keys)
        if dup_token_keys:
            raise RuntimeError(
                f"trace validation failed: duplicate token keys: {dup_token_keys[:8]}"
            )

        if len(meta.thread_name_to_id) != len(thread_ids):
            raise RuntimeError("trace validation failed: thread_name_to_id is non-injective")

        if len(meta.thread_id_to_name) != len(thread_ids):
            raise RuntimeError("trace validation failed: thread_id_to_name is non-injective")

        if len(set(token_keys)) != len(token_keys):
            raise RuntimeError("trace validation failed: duplicate raw token keys")

        for key, name in zip(token_keys, token_names):
            mapped = meta.token_key_to_name.get(key)
            if mapped != name:
                raise RuntimeError(
                    f"trace validation failed: inconsistent token name for key {key!r}: "
                    f"{mapped!r} != {name!r}"
                )

    def summarize_tokens(self, query: SummaryQuery) -> TraceSummary:
        cache_key = ("summary_tokens", query)
        cached = self._summary_cache.get(cache_key)
        if cached is not None:
            return cached

        if self._trace is None:
            self.open()
        assert self._trace is not None

        wanted_tids = set(int(t) for t in query.thread_ids)

        incl_totals: dict[tuple[int, int], int] = defaultdict(int)
        excl_totals: dict[tuple[int, int], int] = defaultdict(int)
        call_counts: dict[tuple[int, int], int] = defaultdict(int)
        tids_by_token: dict[tuple[int, int], set[int]] = defaultdict(set)

        for archive in self._trace.archives:
            for thread in archive.threads:
                tid = int(thread.id)
                if wanted_tids and tid not in wanted_tids:
                    continue

                for seq in thread.sequences:
                    if query.block_only and seq.type != pallas.SequenceType.SEQUENCE_BLOCK:
                        continue

                    token = seq.id
                    token_type = int(token.type)

                    token_id = int(token.id)
                    key = self.remap_token_pair(token_type, token_id, token_mode=query.token_mode)

                    n_iter = int(seq.n_iterations)
                    if n_iter <= 0:
                        continue

                    call_counts[key] += n_iter
                    tids_by_token[key].add(tid)

                    if query.fidelity == "fast":
                        incl_totals[key] += int(seq.mean_duration) * n_iter
                        excl_totals[key] += int(seq.mean_exclusive_duration) * n_iter
                    else:
                        incl_totals[key] += int(seq.durations.as_numpy_array().sum())
                        excl_totals[key] += int(seq.exclusive_durations.as_numpy_array().sum())

        keys = set(incl_totals) | set(excl_totals) | set(call_counts)
        rows = [
            TokenSummary(
                token_type      = token_type,
                token_id        = token_id,
                incl_total_ns   = int(incl_totals.get((token_type, token_id), 0)),
                excl_total_ns   = int(excl_totals.get((token_type, token_id), 0)),
                call_count      = int(call_counts.get((token_type, token_id), 0)),
                thread_ids      = tuple(sorted(tids_by_token.get((token_type, token_id), set()))),
            )
            for (token_type, token_id) in keys
        ]
        rows.sort(
            key=lambda r: (-r.excl_total_ns, -r.incl_total_ns, r.call_count, r.token_type, r.token_id)
        )
        top_tokens = tuple(as_token_key(r.token_type, r.token_id) for r in rows[: query.top_k])

        summary = TraceSummary(
            fidelity    = query.fidelity,
            tokens      = tuple(rows),
            top_tokens  = top_tokens,
        )
        self._summary_cache.put(cache_key, summary)
        return summary

    # ---------- queries ----------

    def query_quanta(self, query: QuantaQuery) -> QuantaBundle:
        query = canonicalize_quanta_query(query)

        cache_key = ("quanta", query)
        cached = self._quanta_cache.get(cache_key)
        if cached is not None:
            return cached

        if self._trace is None:
            self.open()
        assert self._trace is not None

        tids = np.asarray(query.thread_ids, dtype=np.uint32)
        bins = np.asarray(query.bin_edges_ns, dtype=np.uint64)

        if tids.size == 0 or bins.size < 2:
            bundle = empty_quanta_bundle(query.fidelity)
            self._quanta_cache.put(cache_key, bundle)
            return bundle

        with timed(f"trace.calc_quanta_base[{query.fidelity}]"):
            raw = self._trace.calc_quanta_base(
                tids, bins, query.fidelity,
                -1 if query.top_k is None else int(query.top_k)
            )

        with timed(f"normalize_quanta_result[{query.fidelity}]"):
            bundle = normalize_quanta_result(raw, query.fidelity)

        if query.token_mode == "named":
            tt, tid, groups, sums = self.remap_tokens(
                bundle.token_type,
                bundle.token_id,
                token_mode = query.token_mode,
                group_keys=(bundle.start_ns, bundle.end_ns, bundle.thread_id),
                sum_fields=(bundle.excl_ns, bundle.proportion),
            )
            assert groups is not None
            assert sums is not None

            start_ns, end_ns, thread_id = groups
            excl_ns, proportion = sums
            bundle = QuantaBundle(
                fidelity = bundle.fidelity,
                start_ns = start_ns,
                end_ns = end_ns,
                thread_id = thread_id,
                token_type = tt,
                token_id = tid,
                excl_ns = excl_ns,
                proportion = proportion,
            )

        self._quanta_cache.put(cache_key, bundle)
        return bundle

    def query_spans(self, query: SpanQuery) -> SpanBundle:
        query = canonicalize_span_query(query)

        cache_key = ("spans", query)
        cached = self._span_cache.get(cache_key)
        if cached is not None:
            return cached

        if self._trace is None:
            self.open()
        assert self._trace is not None

        req_thread_ids = set(query.thread_ids)
        token_filter = query.token

        rows: list[tuple[int, int, int, int, int, int, int, int, int]] = []
        for archive in self._trace.archives:
            for thread in archive.threads:
                tid = int(thread.id)
                if tid not in req_thread_ids:
                    continue

                reader = thread.reader()

                while not reader.isEndOfTrace():
                    token, iteration = reader.pollCurToken()
                    depth = sequence_block_depth(reader)

                    if query.max_depth is not None and depth > query.max_depth:
                        while reader.exitIfEndOfBlock(True, True):
                            pass
                        reader.moveToNextToken(False, False)
                        continue

                    if not isinstance(token, pallas.Sequence):
                        reader.moveToNextToken(True, True)
                        continue

                    reader.moveToNextToken(True, False)

                    if token.type != pallas.SequenceType.SEQUENCE_BLOCK:
                        reader.moveToNextToken(True, True)
                        continue

                    raw_token_type = int(token.type)
                    raw_token_id = token_int_id(token)
                    token_type, token_id = self.remap_token_pair(
                        raw_token_type,
                        raw_token_id,
                        token_mode = query.token_mode,
                    )

                    if token_filter is not None and (token_type, token_id) != token_filter:
                        reader.moveToNextToken(True, True)
                        continue

                    start_ns = int(token.timestamps[iteration])
                    dur_ns = int(token.durations[iteration])
                    excl_ns = int(token.exclusive_durations[iteration])
                    end_ns = start_ns + dur_ns

                    if end_ns < query.t0_ns:
                        reader.moveToNextToken(True, True)
                        continue
                    if start_ns > query.t1_ns:
                        reader.moveToNextToken(True, True)
                        continue

                    rows.append((
                        tid,
                        token_type,
                        token_id,
                        int(iteration),
                        int(depth),
                        start_ns,
                        end_ns,
                        dur_ns,
                        excl_ns,
                    ))

                    reader.moveToNextToken(True, True)

        bundle = normalize_span_rows(rows, fidelity="exact")
        self._span_cache.put(cache_key, bundle)
        return bundle

    def query_occurrences(self, query: OccurrenceQuery) -> tuple[NodeRef, ...]:
        query = canonicalize_occurrence_query(query)

        cache_key = ("occ", query)
        cached = self._occ_cache.get(cache_key)
        if cached is not None:
            return cached

        if self._trace is None:
            self.open()
        assert self._trace is not None

        if query.fidelity != "exact":
            raise NotImplementedError("query_occurrences: only fidelity='exact' is supported currently")

        wanted_thread_ids = set(query.thread_ids)
        want_token = query.token

        rows: list[NodeRef] = []
        for archive in self._trace.archives:
            for thread in archive.threads:
                tid = int(thread.id)
                if tid not in wanted_thread_ids:
                    continue

                reader = thread.reader()

                while not reader.isEndOfTrace():
                    token, iteration = reader.pollCurToken()
                    depth = sequence_block_depth(reader)

                    if query.max_depth is not None and depth > query.max_depth:
                        while reader.exitIfEndOfBlock(True, True):
                            pass
                        reader.moveToNextToken(False, False)
                        continue

                    if not isinstance(token, pallas.Sequence):
                        reader.moveToNextToken(True, True)
                        continue

                    reader.moveToNextToken(True, False)

                    if token.type != pallas.SequenceType.SEQUENCE_BLOCK:
                        reader.moveToNextToken(True, True)
                        continue

                    token_type = int(token.type)
                    token_id = token_int_id(token)
                    mapped_type, mapped_id = self.remap_token_pair(
                        token_type,
                        token_id,
                        token_mode=query.token_mode,
                    )

                    if (mapped_type, mapped_id) != want_token:
                        reader.moveToNextToken(True, True)
                        continue

                    start_ns = int(token.timestamps[iteration])
                    dur_ns = int(token.durations[iteration])
                    end_ns = start_ns + dur_ns

                    if query.t0_ns is not None and end_ns < query.t0_ns:
                        reader.moveToNextToken(True, True)
                        continue
                    if query.t1_ns is not None and start_ns > query.t1_ns:
                        reader.moveToNextToken(True, True)
                        continue

                    rows.append(
                        NodeRef(
                            thread_id   = tid,
                            token_type  = token_type,
                            token_id    = token_id,
                            iteration   = int(iteration),
                            depth       = int(depth),
                            start_ns    = start_ns,
                            end_ns      = end_ns,
                        )
                    )

                    reader.moveToNextToken(True, True)

        rows.sort(key=lambda r: (r.thread_id, r.start_ns, -r.end_ns, r.iteration))

        if query.mode == "all":
            out = tuple(rows)
        elif query.mode == "first":
            out = tuple(rows[:1])
        elif query.mode == "last":
            out = tuple(rows[-1:]) if rows else ()
        elif query.mode == "nth":
            if query.index is None:
                raise ValueError("query_occurrences(mode='nth') requires index")
            out = (rows[query.index],) if 0 <= query.index < len(rows) else ()
        elif query.mode == "median":
            if not rows:
                out = ()
            else:
                durations = np.array([r.end_ns - r.start_ns for r in rows], dtype=np.int64)
                median = np.median(durations)
                idx = int(np.argmin(np.abs(durations - median)))
                out = (rows[idx],)
        else:
            raise ValueError(f"unknown occurrence mode: {query.mode}")

        self._occ_cache.put(cache_key, out)
        return out

    def query_subtree(self, query: SubtreeQuery) -> SpanBundle:
        query = canonicalize_subtree_query(query)

        cache_key = ("subtree", query)
        cached = self._subtree_cache.get(cache_key)
        if cached is not None:
            return cached

        root = query.root
        if query.max_depth is None:
            span_max_depth = None
        else:
            span_max_depth = root.depth + query.max_depth

        base = self.query_spans(
            SpanQuery(
                thread_ids  = (root.thread_id,),
                t0_ns       = root.start_ns,
                t1_ns       = root.end_ns,
                fidelity    = query.fidelity,
                max_depth   = span_max_depth,
                token       = None,
            )
        )

        keep = (
            (base.thread_id == root.thread_id) &
            (base.start_ns >= root.start_ns) &
            (base.end_ns <= root.end_ns)
        )

        if query.max_depth is not None:
            keep &= (base.depth <= root.depth + query.max_depth)

        idx = np.nonzero(keep)[0]
        if idx.size == 0:
            bundle = empty_span_bundle(base.fidelity)
            self._subtree_cache.put(cache_key, bundle)
            return bundle

        bundle = subset_span_bundle(base, idx, fidelity=base.fidelity)

        bundle.depth = bundle.depth - int(root.depth)
        if query.normalize_time:
            bundle.start_ns = bundle.start_ns - int(root.start_ns)
            bundle.end_ns = bundle.end_ns - int(root.start_ns)

        self._subtree_cache.put(cache_key, bundle)
        return bundle

    def query_histogram(self, query: SnapshotHistogramQuery) -> SnapshotHistogram:
        query = canonicalize_histogram_query(query)

        cache_key = ("snapshot_hist", query)
        cached = self._hist_cache.get(cache_key)
        if cached is not None:
            return cached

        if self._trace is None:
            self.open()
        assert self._trace is not None

        if query.n_bins <= 0 or query.t1_ns <= query.t0_ns or not query.thread_ids:
            out = SnapshotHistogram(left_ns=(), right_ns=(), excl_ns=())
            self._hist_cache.put(cache_key, out)
            return out

        edges = np.linspace(query.t0_ns, query.t1_ns, query.n_bins + 1, dtype=np.int64)
        totals = np.zeros(query.n_bins, dtype=np.int64)
        wanted_tids = set(int(t) for t in query.thread_ids)

        for archive in self._trace.archives:
            for thread in archive.threads:
                tid = int(thread.id)
                if tid not in wanted_tids:
                    continue

                for i in range(query.n_bins):
                    start_ns = int(edges[i])
                    end_ns = int(edges[i + 1])
                    snap = thread.getSnapshotView(start_ns, end_ns)

                    bin_total = 0
                    for key, value in snap.items():
                        raw_type, raw_id = self._snapshot_key_to_token_pair(key)
                        mapped = self.remap_token_pair(raw_type, raw_id, token_mode=query.token_mode)
                        if mapped == query.token:
                            bin_total += int(value)

                    totals[i] += bin_total

        out = SnapshotHistogram(
            left_ns=tuple(int(x) for x in edges[:-1]),
            right_ns=tuple(int(x) for x in edges[1:]),
            excl_ns=tuple(int(x) for x in totals),
        )
        self._hist_cache.put(cache_key, out)
        return out

    # ------- helpers -------

    def remap_token_pair(
        self,
        token_type: int,
        token_id: int,
        *,
        token_mode: TokenMode,
    ) -> tuple[int, int]:
        token_type = int(token_type)
        token_id = int(token_id)

        if token_mode != "named":
            return (token_type, token_id)

        mapped = self.meta.token_cat_remap.get((token_type, token_id))
        if mapped is None:
            return (token_type, token_id)

        return (int(mapped[0]), int(mapped[1]))

    def remap_tokens(
        self,
        token_type: np.ndarray,
        token_id: np.ndarray,
        *,
        token_mode: TokenMode,
        group_keys: Optional[tuple[np.ndarray, ...]] = None,
        sum_fields: Optional[tuple[np.ndarray, ...]] = None,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        Optional[tuple[np.ndarray, ...]],
        Optional[tuple[np.ndarray, ...]],
    ]:
        out_type = np.asarray(token_type, dtype=np.uint8).copy()
        out_id = np.asarray(token_id, dtype=np.int64).copy()

        if token_mode == "named":
            remap = self.meta.token_cat_remap
            for i in range(len(out_id)):
                mapped = remap.get((int(out_type[i]), int(out_id[i])))
                if mapped is not None:
                    out_type[i] = np.uint8(mapped[0])
                    out_id[i] = np.int64(mapped[1])

        if group_keys is None or sum_fields is None:
            return out_type, out_id, None, None

        acc: dict[tuple[int, ...], list[object]] = {}
        order: list[tuple[int, ...]] = []

        for i in range(len(out_id)):
            key = tuple(int(g[i]) for g in group_keys) + (int(out_type[i]), int(out_id[i]))
            if key not in acc:
                acc[key] = [field[i] for field in sum_fields]
                order.append(key)
            else:
                for j in range(len(sum_fields)):
                    acc[key][j] += sum_fields[j][i]

        if not order:
            empty_groups = tuple(np.array([], dtype=g.dtype) for g in group_keys)
            empty_sums = tuple(np.array([], dtype=f.dtype) for f in sum_fields)
            return (
                np.array([], dtype=np.uint8),
                np.array([], dtype=np.int64),
                empty_groups,
                empty_sums,
            )

        n_group = len(group_keys)
        cols = list(zip(*order))

        out_group_keys = tuple(
            np.asarray(cols[i], dtype=group_keys[i].dtype)
            for i in range(n_group)
        )
        out_type2 = np.asarray(cols[n_group], dtype=np.uint8)
        out_id2 = np.asarray(cols[n_group + 1], dtype=np.int64)
        out_sum_fields = tuple(
            np.asarray([acc[key][j] for key in order], dtype=sum_fields[j].dtype)
            for j in range(len(sum_fields))
        )

        return out_type2, out_id2, out_group_keys, out_sum_fields

    def _snapshot_key_to_token_pair(self, key) -> tuple[int, int]:
        if isinstance(key, tuple):
            if len(key) >= 1:
                tok = key[0]
                return (int(tok.type), int(tok.id))
            raise ValueError(f"empty snapshot key tuple: {key!r}")
        return (int(key.type), int(key.id))


