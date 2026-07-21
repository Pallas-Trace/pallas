from __future__ import annotations

import os
import sys
import csv
import time
from typing import Any
from collections import defaultdict

import numpy as np
from bokeh.events import DocumentReady
from bokeh.io import curdoc
from bokeh.models.callbacks import CustomJS
import pallas_trace as pallas

from data_model import QuantaQuery
from trace_session import TraceSession
from controller import AppController, LoadedTrace
from utils import timed

DEBUG_EXACT_COMPARE = False
DEBUG_EXPORT_FAST_BALANCED = True
DEBUG_TRACE_SIDE = "t1"          # "t1" or "t2"
DEBUG_THREAD_NAME = None         # e.g. "Rank 0 main thread"; None => first thread
DEBUG_N_BINS = 2                # start with 
DEBUG_TOP_K = -1                 # disable truncation
DEBUG_TOP_N = 20                 # number of diff rows to print
DEBUG_WORST_BINS = 5
DEBUG_ROWS_PER_BIN = 5

def ns_to_ms(x: int) -> float:
    return float(x) / 1e6

def pct(x: float) -> str:
    return f"{100.0 * float(x):.2f}%"

def token_name_from_row(row: dict[str, Any], token_key_to_name: dict[str, str]) -> str:
    key = str(row["token_key"])
    return token_key_to_name.get(key, key)

def print_exact_compare_summary(report: dict[str, Any]) -> None:
    s = report["summary"]
    print("=== Exact implementation comparison ===")
    print(f"thread_id            : {s['thread_id']}")
    print(f"n_bins               : {s['n_bins']}")
    print(f"top_k                : {s['top_k']}")
    print(f"top_n                : {s['top_n']}")
    print(f"old_total_ns         : {s['old_total_ns']}")
    print(f"new_total_ns         : {s['new_total_ns']}")
    print(f"matched_total_ns     : {s['matched_total_ns']}")
    print(f"union_total_ns       : {s['union_total_ns']}")
    print(f"global_overlap_ratio : {pct(s['global_overlap_ratio'])}")
    print(f"old_row_count        : {s['old_row_count']}")
    print(f"new_row_count        : {s['new_row_count']}")
    print()

def print_diff_rows(
    rows: list[dict[str, Any]],
    token_key_to_name: dict[str, str],
    *,
    title: str,
    limit: int = 10,
) -> None:
    print(f"=== {title} ===")
    if not rows:
        print("(none)")
        print()
        return

    for i, row in enumerate(rows[:limit], start=1):
        name = token_name_from_row(row, token_key_to_name)
        print(
            f"{i:2d}. "
            f"{name:<30} "
            f"old={pct(row['old_prop']):>8} "
            f"new={pct(row['new_prop']):>8} "
            f"delta_ns={int(row['delta_ns']):>14} "
            f"abs_delta_prop={pct(row['abs_delta_prop']):>8}"
        )
    print()

def sort_bins_by_disagreement(per_bin: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        per_bin,
        key=lambda b: (
            float(b["overlap_ratio"]),
            -abs(int(b["old_total_ns"]) - int(b["new_total_ns"])),
        ),
    )

def print_worst_bins(
    report: dict[str, Any],
    token_key_to_name: dict[str, str],
    *,
    n_bins: int = 5,
    n_rows_per_bin: int = 5,
) -> None:
    bins = sort_bins_by_disagreement(report["per_bin"])
    print("=== Worst bins ===")
    if not bins:
        print("(none)")
        print()
        return

    for b in bins[:n_bins]:
        print(
            f"bin {b['bin_index']:>4d}  "
            f"[{ns_to_ms(int(b['start_ns'])):12.3f}, {ns_to_ms(int(b['finish_ns'])):12.3f}] ms  "
            f"overlap={pct(b['overlap_ratio'])}  "
            f"old_total={int(b['old_total_ns'])}  "
            f"new_total={int(b['new_total_ns'])}"
        )
        rows = b.get("largest_diffs", [])
        for row in rows[:n_rows_per_bin]:
            name = token_name_from_row(row, token_key_to_name)
            print(
                f"    {name:<30} "
                f"old={pct(row['old_prop']):>8} "
                f"new={pct(row['new_prop']):>8} "
                f"delta_ns={int(row['delta_ns']):>14}"
            )
        print()

def get_raw_trace(session: TraceSession) -> pallas.Trace:
    if session._trace is None:
        raise NotImplementedError
    return session._trace

def run_exact_debug(session: TraceSession) -> dict[str, Any]:
    meta = session.meta
    raw_trace = get_raw_trace(session)

    if DEBUG_THREAD_NAME is None:
        thread_id = int(meta.thread_ids[0])
        thread_name = meta.thread_id_to_name.get(thread_id, str(thread_id))
    else:
        thread_name = DEBUG_THREAD_NAME
        thread_id = int(meta.thread_name_to_id[thread_name])

    edges = np.linspace(
        int(meta.start_ns),
        int(meta.end_ns),
        int(DEBUG_N_BINS) + 1,
        dtype=np.uint64,
    )

    report = raw_trace.compare_exact_impls(
        thread_id=thread_id,
        bin_edges_ns=edges,
        top_k=DEBUG_TOP_K,
        top_n=DEBUG_TOP_N,
    )

    token_key_to_name = dict(meta.token_key_to_name)

    print()
    print(f"=== Debug trace side: {DEBUG_TRACE_SIDE} ===")
    print(f"=== Debug thread: {thread_name} ({thread_id}) ===")
    print_exact_compare_summary(report)
    print_diff_rows(
        report.get("largest_diffs", []), # type: ignore
        token_key_to_name,
        title="Largest whole-window differences",
        limit=DEBUG_TOP_N,
    )
    print_worst_bins(
        report,
        token_key_to_name,
        n_bins=min(DEBUG_WORST_BINS, DEBUG_N_BINS),
        n_rows_per_bin=DEBUG_ROWS_PER_BIN,
    )

    return report

def save_quanta_res_csv(res, path: str, token_key_to_name: dict[str, str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "start_ns", "finish_ns", "thread_id",
                "token_type", "token_id", "token_key",
                "token_name", "excl_ns", "proportion",
            ],
        )
        w.writeheader()
        for i in range(len(res)):
            token_type = int(res.token_type[i])
            token_id = int(res.token_id[i])
            tkey = f"{token_type}:{token_id}"
            w.writerow(
                {
                    "start_ns": int(res.start_ns[i]),
                    "finish_ns": int(res.finish_ns[i]),
                    "thread_id": int(res.thread_id[i]),
                    "token_type": token_type,
                    "token_id": token_id,
                    "token_key": tkey,
                    "token_name": token_key_to_name.get(tkey, tkey),
                    "excl_ns": int(res.excl_ns[i]),
                    "proportion": float(res.proportion[i]),
                }
            )

def save_quanta_bundle_csv(bundle, path: str, token_key_to_name: dict[str, str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "start_ns", "finish_ns", "thread_id",
                "token_type", "token_id", "token_key",
                "token_name", "excl_ns", "proportion",
            ],
        )
        w.writeheader()
        for i in range(len(bundle.start_ns)):
            token_type = int(bundle.token_type[i])
            token_id = int(bundle.token_id[i])
            tkey = f"{token_type}:{token_id}"
            w.writerow(
                {
                    "start_ns": int(bundle.start_ns[i]),
                    "finish_ns": int(bundle.end_ns[i]),
                    "thread_id": int(bundle.thread_id[i]),
                    "token_type": token_type,
                    "token_id": token_id,
                    "token_key": tkey,
                    "token_name": token_key_to_name.get(tkey, tkey),
                    "excl_ns": int(bundle.excl_ns[i]),
                    "proportion": float(bundle.proportion[i]),
                }
            )

def extract_bin_edges_from_quanta_res(res) -> tuple[int, ...]:
    bins = sorted(
        {
            (int(res.start_ns[i]), int(res.finish_ns[i]))
            for i in range(len(res))
        }
    )
    if not bins:
        return ()

    edges = [bins[0][0]]
    for start_ns, finish_ns in bins:
        if start_ns != edges[-1]:
            raise ValueError(
                f"Non-contiguous bins in debug exact result: got {start_ns}, expected {edges[-1]}"
            )
        if finish_ns <= start_ns:
            raise ValueError(f"Invalid bin [{start_ns}, {finish_ns})")
        edges.append(finish_ns)
    return tuple(edges)

def quanta_res_to_rows(res, token_key_to_name: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(len(res)):
        token_type = int(res.token_type[i])
        token_id = int(res.token_id[i])
        token_key = f"{token_type}:{token_id}"
        rows.append(
            {
                "start_ns": int(res.start_ns[i]),
                "finish_ns": int(res.finish_ns[i]),
                "thread_id": int(res.thread_id[i]),
                "token_key": token_key,
                "token_name": token_key_to_name.get(token_key, token_key),
                "excl_ns": int(res.excl_ns[i]),
            }
        )
    return rows

def quanta_bundle_to_rows(bundle, token_key_to_name: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(len(bundle.start_ns)):
        token_type = int(bundle.token_type[i])
        token_id = int(bundle.token_id[i])
        token_key = f"{token_type}:{token_id}"
        rows.append(
            {
                "start_ns": int(bundle.start_ns[i]),
                "finish_ns": int(bundle.end_ns[i]),
                "thread_id": int(bundle.thread_id[i]),
                "token_key": token_key,
                "token_name": token_key_to_name.get(token_key, token_key),
                "excl_ns": int(bundle.excl_ns[i]),
            }
        )
    return rows

def aggregate_rows_by_token(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for row in rows:
        out[str(row["token_name"])] += int(row["excl_ns"])
    return dict(out)

def rows_by_bin_and_key(rows: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, int]]:
    out: dict[tuple[int, int], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        bk = (int(row["start_ns"]), int(row["finish_ns"]))
        out[bk][str(row["token_key"])] += int(row["excl_ns"])
    return out

def compute_overlap_and_wmape(
    exact_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, float]:
    exact_bins = rows_by_bin_and_key(exact_rows)
    cand_bins = rows_by_bin_and_key(candidate_rows)
    all_bins = sorted(set(exact_bins) | set(cand_bins))

    weighted_overlap_num = 0.0
    weighted_overlap_den = 0.0
    abs_error_total = 0
    exact_total = 0

    for bk in all_bins:
        exact_map = exact_bins.get(bk, {})
        cand_map = cand_bins.get(bk, {})

        bin_exact_total = sum(exact_map.values())
        bin_matched = 0
        bin_union = 0
        bin_abs_error = 0

        for key in set(exact_map) | set(cand_map):
            a = int(exact_map.get(key, 0))
            b = int(cand_map.get(key, 0))
            bin_matched += min(a, b)
            bin_union += max(a, b)
            bin_abs_error += abs(b - a)

        exact_total += bin_exact_total
        abs_error_total += bin_abs_error

        if bin_union > 0:
            weighted_overlap_num += bin_exact_total * (bin_matched / bin_union)
            weighted_overlap_den += bin_exact_total

    overlap_ratio = (
        weighted_overlap_num / weighted_overlap_den if weighted_overlap_den else 1.0
    )
    wmape = (abs_error_total / exact_total) if exact_total else 0.0

    return {
        "exact_total_ns": float(exact_total),
        "abs_error_total_ns": float(abs_error_total),
        "overlap_ratio": float(overlap_ratio),
        "wmape": float(wmape),
    }


def print_fidelity_metrics(label: str, metrics: dict[str, float]) -> None:
    print(f"=== {label} vs exact ===")
    print(f"overlap_ratio : {pct(metrics['overlap_ratio'])}")
    print(f"wMAPE         : {pct(metrics['wmape'])}")
    print()

def main():

    if len(sys.argv) < 3:
        raise SystemExit("Usage: bokeh serve --show main.py --args TRACE1 TRACE2")

    js_startup_timer = CustomJS(code="""
        const t_ready = performance.now();
        const t0 = window.pallas_startup_begin ?? t_ready;
        const startup_ms = t_ready - t0;

        console.log("=========================================");
        console.log("[BENCHMARK] PALLAS Startup Metrics");
        console.log(`  - Browser startup to document_ready : ${startup_ms.toFixed(2)} ms`);
        console.log("=========================================");
    """)
    curdoc().js_on_event(DocumentReady, js_startup_timer)

    print(">> Reading Traces")

    paths = [p for p in sys.argv[1:] if p.strip()]
    if not paths:
        raise SystemExit("usage: bokeh serve --show main.py --args TRACE1 [TRACE2 ...]")

    loaded: list[LoadedTrace] = []
    for i, path in enumerate(paths):
        session = TraceSession(path)
        with timed(f"open {os.path.basename(path) or f'trace_{i}'}"):
            session.open()
        loaded.append(
            LoadedTrace(
                trace_id=f"trace_{i}",
                path=path,
                label=os.path.basename(path) or f"trace_{i}",
                session=session,
            )
        )

    print(">> Traces Read")

    # if DEBUG_EXACT_COMPARE:
    #     debug_session = t1 if DEBUG_TRACE_SIDE == "t1" else t2
    #     with timed("exact debug compare"):
    #         report = run_exact_debug(debug_session)

    #     token_key_to_name = dict(debug_session.meta.token_key_to_name)
    #     exact_rows = quanta_res_to_rows(report["new"], token_key_to_name)

    #     bin_edges_ns = extract_bin_edges_from_quanta_res(report["new"])
    #     if not bin_edges_ns:
    #         raise ValueError("Could not derive bin edges from exact debug output")

    #     thread_ids = tuple(sorted(set(int(x) for x in report["new"].thread_id)))
    #     if not thread_ids:
    #         raise ValueError("Could not derive thread ids from exact debug output")

    #     with timed("debug query fast"):
    #         fast_bundle = debug_session.query_quanta(
    #             QuantaQuery(
    #                 thread_ids=thread_ids,
    #                 bin_edges_ns=bin_edges_ns,
    #                 fidelity="fast",
    #                 top_k=None,
    #             )
    #         )

    #     with timed("debug query balanced"):
    #         balanced_bundle = debug_session.query_quanta(
    #             QuantaQuery(
    #                 thread_ids=thread_ids,
    #                 bin_edges_ns=bin_edges_ns,
    #                 fidelity="balanced",
    #                 top_k=None,
    #             )
    #         )

    #     fast_rows = quanta_bundle_to_rows(fast_bundle, token_key_to_name)
    #     balanced_rows = quanta_bundle_to_rows(balanced_bundle, token_key_to_name)

    #     fast_metrics = compute_overlap_and_wmape(exact_rows, fast_rows)
    #     balanced_metrics = compute_overlap_and_wmape(exact_rows, balanced_rows)

    #     print_fidelity_metrics("fast", fast_metrics)
    #     print_fidelity_metrics("balanced", balanced_metrics)

    with timed("build"):
        controller = AppController(loaded)
        root = controller.build()
        if not root:
            raise AssertionError
        curdoc().add_root(root)
        curdoc().title = "Blup"

main()
