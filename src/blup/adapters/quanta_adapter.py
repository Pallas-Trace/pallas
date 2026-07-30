from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import colorsys
import hashlib

import numpy as np

from state import TraceMode
from data_model import QuantaBundle, QuantaQuery, TraceMeta, FidelityMode, TokenMode, as_token_key
from trace_session import TraceSession


BASE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ab",
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
]


@dataclass(frozen=True)
class QuantaRequestSpec:
    active_threads:         tuple[str, ...]
    n_bins:                 int
    mode:                   FidelityMode
    trace_mode:             TraceMode
    token_mode:             TokenMode
    stack_order:            str
    start_ns:               int
    end_ns:                 int
    sync_range_to_fig:      bool
    bin_edges_ns:           tuple[int, ...]
    color_map:              dict[str, str]
    request_key:            tuple


@dataclass(frozen=True)
class QuantaThreadJobData:
    thread_name:            str
    thread_id_1:            int | None
    thread_id_2:            int | None
    active_threads:         tuple[str, ...]
    bin_edges_ns:           tuple[int, ...]
    mode:                   FidelityMode
    trace_mode:             TraceMode
    token_mode:             TokenMode
    stack_order:            str
    color_map:              dict[str, str]


@dataclass(frozen=True)
class QuantaThreadResultData:
    thread_name:            str
    src1:                   dict
    src2:                   dict


def empty_quanta_source() -> dict:
    return dict(
        left=[],
        right=[],
        top=[],
        bottom=[],
        color=[],
        token_key=[],
        token_name=[],
        token_type=[],
        token_id=[],
        thread=[],
        proportion=[],
        exclusive_s=[],
    )

def generated_color(key: str) -> str:
    h = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    hue = int.from_bytes(h[:2], "big") / 65535.0
    sat = 0.55 + (h[2] / 255.0) * 0.25
    val = 0.65 + (h[3] / 255.0) * 0.25
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

def build_color_map(token_keys: set[str]) -> dict[str, str]:
    keys = sorted(token_keys)
    cmap: dict[str, str] = {}
    for i, key in enumerate(keys):
        cmap[key] = BASE[i] if i < len(BASE) else generated_color(key)
    cmap["OTHER"] = "#b0b0b0"
    return cmap

def combine_thread_names(
    t1: TraceSession,
    t2: TraceSession | None
) -> tuple[str, ...]:
    names = set(map(str, t1.meta.thread_names))
    if t2 is not None:
        names |= set(map(str, t2.meta.thread_names))
    return tuple(sorted(names))

def get_shared_thread_names(
    t1: TraceSession,
    t2: TraceSession | None,
    requested: tuple[str, ...],
) -> tuple[str, ...]:
    available = set(combine_thread_names(t1, t2))
    chosen = tuple(name for name in requested if name in available)
    return chosen or combine_thread_names(t1, t2)

def build_quanta_request_spec(
    *,
    t1: TraceSession,
    t2: TraceSession | None,
    active_thread_names: tuple[str, ...],
    n_bins: int,
    mode: FidelityMode,
    trace_mode: TraceMode,
    token_mode: TokenMode,
    stack_order: str,
    window_t0_ns: int | None = None,
    window_t1_ns: int | None = None,
) -> QuantaRequestSpec:
    active_threads = get_shared_thread_names(t1, t2, active_thread_names)

    if t2 is None:
        full_start_ns = int(t1.meta.start_ns)
        full_end_ns = int(t1.meta.end_ns)
        all_token_keys = set(t1.meta.token_key_to_name.keys())
    else:
        full_start_ns = min(int(t1.meta.start_ns), int(t2.meta.start_ns))
        full_end_ns = max(int(t1.meta.end_ns), int(t2.meta.end_ns))
        all_token_keys = (
            set(t1.meta.token_key_to_name.keys())
            | set(t2.meta.token_key_to_name.keys())
        )

    if window_t0_ns is None or window_t1_ns is None:
        start_ns = full_start_ns
        end_ns = full_end_ns
        sync_range_to_fig = True
    else:
        start_ns = window_t0_ns
        end_ns = window_t1_ns
        sync_range_to_fig = False

    if end_ns <= start_ns:
        end_ns = start_ns + 1

    bin_edges_ns = tuple(
        int(x) for x in np.linspace(start_ns, end_ns, int(n_bins) + 1, dtype=np.int64)
    )
    color_map = build_color_map(all_token_keys)
    request_key = (
        active_threads,
        int(n_bins),
        mode,
        token_mode,
        stack_order,
        int(start_ns),
        int(end_ns),
    )

    return QuantaRequestSpec(
        active_threads      = active_threads,
        n_bins              = int(n_bins),
        mode                = mode,
        trace_mode          = trace_mode,
        token_mode          = token_mode,
        stack_order         = stack_order,
        start_ns            = int(start_ns),
        end_ns              = int(end_ns),
        sync_range_to_fig   = sync_range_to_fig,
        bin_edges_ns        = bin_edges_ns,
        color_map           = color_map,
        request_key         = request_key,
    )

def build_quanta_jobs(
    *,
    t1: TraceSession,
    t2: TraceSession | None,
    spec: QuantaRequestSpec,
) -> list[QuantaThreadJobData]:
    jobs: list[QuantaThreadJobData] = []

    for thread_name in spec.active_threads:
        t1_id = t1.meta.thread_name_to_id.get(thread_name) if t1 else None
        t2_id = t2.meta.thread_name_to_id.get(thread_name) if t2 else None
        jobs.append(
            QuantaThreadJobData(
                thread_name     = thread_name,
                thread_id_1     = t1_id,
                thread_id_2     = t2_id,
                active_threads  = spec.active_threads,
                bin_edges_ns    = spec.bin_edges_ns,
                mode            = spec.mode,
                trace_mode      = spec.trace_mode,
                token_mode      = spec.token_mode,
                stack_order     = spec.stack_order,
                color_map       = spec.color_map,
            )
        )

    return jobs

def compute_quanta_thread_result(
    *,
    t1: TraceSession,
    t2: TraceSession | None,
    job: QuantaThreadJobData,
) -> QuantaThreadResultData:
    if job.thread_id_1 is not None:
        q1 = t1.query_quanta(
            QuantaQuery(
                thread_ids      = (job.thread_id_1,),
                bin_edges_ns    = job.bin_edges_ns,
                fidelity        = job.mode,  # type: ignore[arg-type]
                token_mode      = job.token_mode,
                top_k           = -1,
            )
        )
        src1 = quanta_bundle_to_bokeh_source(
            q1,
            meta            = t1.meta,
            active_threads  = list(job.active_threads),
            trace_side      = "lower",
            color_map       = job.color_map,
            trace_mode      = job.trace_mode,
            stack_order     = job.stack_order,
        )
    else:
        src1 = empty_quanta_source()

    if ((job.thread_id_2 is not None)
        and (job.trace_mode != "single")):
        assert(t2 is not None)
        q2 = t2.query_quanta(
            QuantaQuery(
                thread_ids      = (job.thread_id_2,),
                bin_edges_ns    = job.bin_edges_ns,
                fidelity        = job.mode,  # type: ignore[arg-type]
                token_mode      = job.token_mode,
                top_k           = -1,
            )
        )
        src2 = quanta_bundle_to_bokeh_source(
            q2,
            meta            = t2.meta,
            active_threads  = list(job.active_threads),
            trace_side      = "upper",
            color_map       = job.color_map,
            trace_mode      = job.trace_mode,
            stack_order     = job.stack_order,
        )
    else:
        src2 = empty_quanta_source()

    return QuantaThreadResultData(
        thread_name = job.thread_name,
        src1        = src1,
        src2        = src2,
    )

def quanta_bundle_to_bokeh_source(
    bundle: QuantaBundle,
    *,
    meta: TraceMeta,
    active_threads: list[str],
    trace_side: str,
    color_map: dict[str, str],
    trace_mode: TraceMode = "dual",
    stack_order: str = "global",
) -> dict:
    if len(bundle.start_ns) == 0:
        return empty_quanta_source()

    if trace_mode == "dual":
        if trace_side not in {"lower", "upper"}:
            raise ValueError(f"invalid trace_side for dual mode: {trace_side!r}")
    else:
        if trace_side not in {"lower", "upper"}:
            raise ValueError(f"invalid trace_side: {trace_side!r}")

    thread_centers = {
        name: len(active_threads) - 0.5 - i
        for i, name in enumerate(active_threads)
    }

    start_ns_arr = bundle.start_ns
    end_ns_arr = bundle.end_ns
    thread_id_arr = bundle.thread_id
    token_type_arr = bundle.token_type
    token_id_arr = bundle.token_id
    proportion_arr = bundle.proportion
    excl_ns_arr = bundle.excl_ns

    lefts: list[float] = []
    rights: list[float] = []
    threads: list[str] = []
    token_keys: list[str] = []
    token_names: list[str] = []
    token_types: list[int] = []
    token_ids: list[int] = []
    proportions: list[float] = []
    exclusive_ss: list[float] = []

    OTHER_TOKEN_TYPE = 255
    OTHER_TOKEN_ID = 0
    OTHER_TOKEN_KEY = "OTHER"
    OTHER_TOKEN_NAME = "OTHER"

    dual_half = 0.45
    dual_padding = 0.02
    single_half = 0.45

    for i in range(len(start_ns_arr)):
        tid = int(thread_id_arr[i])
        thread_name = meta.thread_id_to_name.get(tid)
        if thread_name is None or thread_name not in thread_centers:
            continue

        token_type = int(token_type_arr[i])
        token_id = int(token_id_arr[i])

        if token_type == OTHER_TOKEN_TYPE and token_id == OTHER_TOKEN_ID:
            token_key = OTHER_TOKEN_KEY
            token_name = OTHER_TOKEN_NAME
        else:
            token_key = as_token_key(int(token_type), int(token_id))
            token_name = meta.token_key_to_name.get(token_key, token_key)

        lefts.append(int(start_ns_arr[i]) / 1e6)
        rights.append(int(end_ns_arr[i]) / 1e6)
        threads.append(thread_name)
        token_keys.append(token_key)
        token_names.append(token_name)
        token_types.append(token_type)
        token_ids.append(token_id)
        proportions.append(float(proportion_arr[i]))
        exclusive_ss.append(int(excl_ns_arr[i]) / 1e9)

    if not lefts:
        return empty_quanta_source()

    if stack_order == "global":
        totals: dict[str, float] = defaultdict(float)
        for token_key, exclusive_s in zip(token_keys, exclusive_ss):
            totals[token_key] += exclusive_s

        rank = {
            token_key: i
            for i, (token_key, _) in enumerate(
                sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
            )
        }

        sort_key_fn = lambda row: rank.get(row[0], 10**9)
    elif stack_order == "local":
        sort_key_fn = lambda row: (-row[4], row[0])
    else:
        raise ValueError(f"invalid stack_order: {stack_order!r}")

    grouped: dict[tuple[str, float, float], list[tuple[str, str, int, int, float, float]]] = defaultdict(list)

    for left, right, thread, token_key, token_name, token_type, token_id, proportion, exclusive_s in zip(
        lefts,
        rights,
        threads,
        token_keys,
        token_names,
        token_types,
        token_ids,
        proportions,
        exclusive_ss,
    ):
        grouped[(thread, left, right)].append(
            (token_key, token_name, token_type, token_id, proportion, exclusive_s)
        )

    out = empty_quanta_source()

    out_left = out["left"]
    out_right = out["right"]
    out_top = out["top"]
    out_bottom = out["bottom"]
    out_color = out["color"]
    out_token_key = out["token_key"]
    out_token_name = out["token_name"]
    out_token_type = out["token_type"]
    out_token_id = out["token_id"]
    out_thread = out["thread"]
    out_proportion = out["proportion"]
    out_exclusive_s = out["exclusive_s"]

    centered_mode = (trace_mode != "dual")

    for (thread, left, right), grp in grouped.items():
        center = thread_centers[thread]
        grp = sorted(grp, key=sort_key_fn)

        cumsum = 0.0
        for token_key, token_name, token_type, token_id, proportion, exclusive_s in grp:
            if centered_mode:
                lane_bottom = center - single_half
                lane_top = center + single_half
                lane_height = lane_top - lane_bottom

                bottom = lane_bottom + cumsum * lane_height
                top = lane_bottom + (cumsum + proportion) * lane_height
            else:
                if trace_side == "lower":
                    top = center - dual_padding - cumsum * dual_half
                    bottom = center - dual_padding - (cumsum + proportion) * dual_half
                else:
                    bottom = center + dual_padding + cumsum * dual_half
                    top = center + dual_padding + (cumsum + proportion) * dual_half

            out_left.append(left)
            out_right.append(right)
            out_top.append(top)
            out_bottom.append(bottom)
            out_color.append(color_map.get(token_key, "#999999"))
            out_token_key.append(token_key)
            out_token_name.append(token_name)
            out_token_type.append(token_type)
            out_token_id.append(token_id)
            out_thread.append(thread)
            out_proportion.append(proportion)
            out_exclusive_s.append(exclusive_s)

            cumsum += proportion

    return out


