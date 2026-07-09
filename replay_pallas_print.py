from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ENTER_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s+Enter\s+(\d+)\s+\(([^)]+)\)\s*$")
LEAVE_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s+Leave\s+(\d+)\s+\(([^)]+)\)\s*$")
THREAD_END_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s+THREAD_END\(\)\s*$")


@dataclass(frozen=True)
class Event:
    t_ns: int
    kind: str
    token_id: int | None
    token_name: str | None
    raw: str


@dataclass(frozen=True)
class Frame:
    token_id: int
    token_name: str


@dataclass(frozen=True)
class Interval:
    start_ns: int
    finish_ns: int
    dur_ns: int
    owner_token_id: int | None
    owner_token_name: str | None
    stack_names: tuple[str, ...]


def seconds_to_ns(x: str) -> int:
    return int(round(float(x) * 1e9))


def parse_events(txt_path: str | Path) -> list[Event]:
    events: list[Event] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            m = ENTER_RE.match(line)
            if m:
                events.append(
                    Event(
                        t_ns=seconds_to_ns(m.group(1)),
                        kind="enter",
                        token_id=int(m.group(2)),
                        token_name=m.group(3),
                        raw=line,
                    )
                )
                continue

            m = LEAVE_RE.match(line)
            if m:
                events.append(
                    Event(
                        t_ns=seconds_to_ns(m.group(1)),
                        kind="leave",
                        token_id=int(m.group(2)),
                        token_name=m.group(3),
                        raw=line,
                    )
                )
                continue

            m = THREAD_END_RE.match(line)
            if m:
                events.append(
                    Event(
                        t_ns=seconds_to_ns(m.group(1)),
                        kind="thread_end",
                        token_id=None,
                        token_name=None,
                        raw=line,
                    )
                )
                continue

    events.sort(key=lambda e: e.t_ns)
    return events


def is_mpi_like(name: str) -> bool:
    s = name.lower()
    return s.startswith("mpi_") or "mpi" in s


def choose_owner(stack: list[Frame], mode: str) -> Frame | None:
    if not stack:
        return None

    if mode == "innermost_active":
        return stack[-1]

    if mode == "outermost_active":
        return stack[0]

    if mode == "innermost_non_mpi":
        for fr in reversed(stack):
            if not is_mpi_like(fr.token_name):
                return fr
        return stack[-1]

    if mode == "outermost_non_mpi":
        for fr in stack:
            if not is_mpi_like(fr.token_name):
                return fr
        return stack[0]

    raise ValueError(f"Unknown owner mode: {mode}")


def apply_event(stack: list[Frame], ev: Event) -> None:
    if ev.kind == "enter":
        assert ev.token_id is not None and ev.token_name is not None
        stack.append(Frame(ev.token_id, ev.token_name))
        return

    if ev.kind == "leave":
        assert ev.token_id is not None and ev.token_name is not None
        for i in range(len(stack) - 1, -1, -1):
            if stack[i].token_id == ev.token_id and stack[i].token_name == ev.token_name:
                del stack[i]
                return
        return

    if ev.kind == "thread_end":
        stack.clear()
        return


def build_intervals(events: list[Event], owner_mode: str) -> list[Interval]:
    if not events:
        return []

    intervals: list[Interval] = []
    stack: list[Frame] = []

    i = 0
    n = len(events)
    while i < n:
        t0 = events[i].t_ns

        same_time: list[Event] = []
        while i < n and events[i].t_ns == t0:
            same_time.append(events[i])
            i += 1

        for ev in same_time:
            apply_event(stack, ev)

        if i >= n:
            break

        t1 = events[i].t_ns
        if t1 <= t0:
            continue

        owner = choose_owner(stack, owner_mode)
        intervals.append(
            Interval(
                start_ns=t0,
                finish_ns=t1,
                dur_ns=t1 - t0,
                owner_token_id=None if owner is None else owner.token_id,
                owner_token_name=None if owner is None else owner.token_name,
                stack_names=tuple(fr.token_name for fr in stack),
            )
        )

    return intervals


def accumulate_name_interval_into_bins(
    per_bin: list[dict[str, int]],
    edges: list[int],
    token_name: str,
    start_ns: int,
    end_ns: int,
) -> None:
    if end_ns <= start_ns or len(edges) < 2:
        return

    global_start = edges[0]
    global_end = edges[-1]
    if end_ns <= global_start or start_ns >= global_end:
        return

    s = max(start_ns, global_start)
    e = min(end_ns, global_end)
    if e <= s:
        return

    import bisect

    bin_idx = max(0, bisect.bisect_right(edges, s) - 1)
    while bin_idx + 1 < len(edges) and s < e:
        seg_end = min(e, edges[bin_idx + 1])
        if seg_end > s:
            per_bin[bin_idx][token_name] += seg_end - s
        s = seg_end
        bin_idx += 1


def load_backend_quanta_csv(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            token_name = row.get("token_name", "").strip()
            if not token_name:
                token_name = row.get("token_key", "").strip()
            rows.append(
                {
                    "start_ns": int(row["start_ns"]),
                    "finish_ns": int(row["finish_ns"]),
                    "thread_id": int(row["thread_id"]),
                    "token_type": int(row["token_type"]),
                    "token_id": int(row["token_id"]),
                    "token_key": row.get("token_key", ""),
                    "token_name": token_name,
                    "excl_ns": int(row["excl_ns"]),
                    "proportion": float(row["proportion"]),
                }
            )
    return rows


def derive_edges_from_backend_rows(rows: list[dict[str, Any]]) -> list[int]:
    bins = sorted({(int(r["start_ns"]), int(r["finish_ns"])) for r in rows})
    if not bins:
        raise ValueError("compare-csv has no rows")

    edges = [bins[0][0]]
    for start_ns, finish_ns in bins:
        if edges[-1] != start_ns:
            raise ValueError(
                f"Non-contiguous or mismatched backend bins: previous edge {edges[-1]}, next start {start_ns}"
            )
        edges.append(finish_ns)
    return edges


def replay_to_rows_by_name(
    intervals: list[Interval],
    edges: list[int],
    thread_id: int,
) -> list[dict[str, Any]]:
    n_bins = len(edges) - 1
    per_bin: list[dict[str, int]] = [defaultdict(int) for _ in range(n_bins)]

    for iv in intervals:
        if iv.owner_token_name is None:
            continue
        accumulate_name_interval_into_bins(
            per_bin,
            edges,
            iv.owner_token_name,
            iv.start_ns,
            iv.finish_ns,
        )

    rows: list[dict[str, Any]] = []
    for i in range(n_bins):
        b0 = edges[i]
        b1 = edges[i + 1]
        total = sum(per_bin[i].values())
        if total <= 0:
            continue
        for token_name, excl_ns in sorted(per_bin[i].items()):
            rows.append(
                {
                    "start_ns": int(b0),
                    "finish_ns": int(b1),
                    "thread_id": int(thread_id),
                    "token_name": token_name,
                    "excl_ns": int(excl_ns),
                    "proportion": float(excl_ns) / float(total),
                }
            )
    return rows


def backend_rows_to_name_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, int, str], int] = defaultdict(int)

    for r in rows:
        key = (
            int(r["start_ns"]),
            int(r["finish_ns"]),
            int(r["thread_id"]),
            str(r["token_name"]),
        )
        grouped[key] += int(r["excl_ns"])

    out: list[dict[str, Any]] = []
    totals_by_bin: dict[tuple[int, int, int], int] = defaultdict(int)

    for (start_ns, finish_ns, thread_id, token_name), excl_ns in grouped.items():
        totals_by_bin[(start_ns, finish_ns, thread_id)] += excl_ns

    for (start_ns, finish_ns, thread_id, token_name), excl_ns in sorted(grouped.items()):
        total = totals_by_bin[(start_ns, finish_ns, thread_id)]
        out.append(
            {
                "start_ns": start_ns,
                "finish_ns": finish_ns,
                "thread_id": thread_id,
                "token_name": token_name,
                "excl_ns": excl_ns,
                "proportion": (excl_ns / total) if total else 0.0,
            }
        )
    return out


def aggregate_name_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for r in rows:
        out[str(r["token_name"])] += int(r["excl_ns"])
    return dict(out)


def rows_by_bin_name(rows: list[dict[str, Any]]) -> dict[tuple[int, int], list[dict[str, Any]]]:
    out: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        out[(int(r["start_ns"]), int(r["finish_ns"]))].append(r)
    return dict(out)


def diff_name_maps(
    ref_map: dict[str, int],
    cand_map: dict[str, int],
    start_ns: int,
    finish_ns: int,
) -> list[dict[str, Any]]:
    keys = sorted(set(ref_map) | set(cand_map))
    old_total = sum(ref_map.values())
    new_total = sum(cand_map.values())

    rows: list[dict[str, Any]] = []
    for name in keys:
        old_excl = int(ref_map.get(name, 0))
        new_excl = int(cand_map.get(name, 0))
        old_prop = (old_excl / old_total) if old_total else 0.0
        new_prop = (new_excl / new_total) if new_total else 0.0
        rows.append(
            {
                "start_ns": int(start_ns),
                "finish_ns": int(finish_ns),
                "token_name": name,
                "old_excl_ns": old_excl,
                "new_excl_ns": new_excl,
                "delta_ns": new_excl - old_excl,
                "old_prop": old_prop,
                "new_prop": new_prop,
                "abs_delta_prop": abs(new_prop - old_prop),
            }
        )

    rows.sort(key=lambda r: (-r["abs_delta_prop"], -abs(r["delta_ns"]), r["token_name"]))
    return rows


def compare_name_rows(
    reference_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    top_n: int,
) -> dict[str, Any]:
    ref_all = aggregate_name_rows(reference_rows)
    cand_all = aggregate_name_rows(candidate_rows)

    old_total = sum(ref_all.values())
    new_total = sum(cand_all.values())

    matched_total = 0
    union_total = 0
    for k in set(ref_all) | set(cand_all):
        a = ref_all.get(k, 0)
        b = cand_all.get(k, 0)
        matched_total += min(a, b)
        union_total += max(a, b)

    whole = diff_name_maps(
        ref_all,
        cand_all,
        min([r["start_ns"] for r in reference_rows + candidate_rows], default=0),
        max([r["finish_ns"] for r in reference_rows + candidate_rows], default=0),
    )

    ref_bins = rows_by_bin_name(reference_rows)
    cand_bins = rows_by_bin_name(candidate_rows)
    all_bins = sorted(set(ref_bins) | set(cand_bins))

    per_bin: list[dict[str, Any]] = []
    for idx, bk in enumerate(all_bins):
        ref_map = aggregate_name_rows(ref_bins.get(bk, []))
        cand_map = aggregate_name_rows(cand_bins.get(bk, []))
        ref_total = sum(ref_map.values())
        cand_total = sum(cand_map.values())

        matched = 0
        uni = 0
        for k in set(ref_map) | set(cand_map):
            a = ref_map.get(k, 0)
            b = cand_map.get(k, 0)
            matched += min(a, b)
            uni += max(a, b)

        diffs = diff_name_maps(ref_map, cand_map, bk[0], bk[1])

        per_bin.append(
            {
                "bin_index": idx,
                "start_ns": int(bk[0]),
                "finish_ns": int(bk[1]),
                "old_total_ns": int(ref_total),
                "new_total_ns": int(cand_total),
                "matched_ns": int(matched),
                "union_ns": int(uni),
                "overlap_ratio": (matched / uni) if uni else 1.0,
                "largest_diffs": diffs[:top_n],
            }
        )

    return {
        "summary": {
            "n_bins": len(all_bins),
            "old_total_ns": int(old_total),
            "new_total_ns": int(new_total),
            "matched_total_ns": int(matched_total),
            "union_total_ns": int(union_total),
            "global_overlap_ratio": (matched_total / union_total) if union_total else 1.0,
            "old_row_count": len(reference_rows),
            "new_row_count": len(candidate_rows),
            "compare_key": "token_name",
        },
        "largest_diffs": whole[:top_n],
        "per_bin": per_bin,
    }


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("print_txt", help="pallas_print text file for one thread")
    ap.add_argument("--thread-id", type=int, default=0)
    ap.add_argument(
        "--owner-mode",
        default="innermost_active",
        choices=["innermost_active", "outermost_active", "innermost_non_mpi", "outermost_non_mpi"],
    )
    ap.add_argument("--compare-csv", required=True, help="backend quanta CSV; its bin edges will be reused")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--out-prefix", default="replay")
    args = ap.parse_args()

    events = parse_events(args.print_txt)
    if not events:
        raise SystemExit("No Enter/Leave/THREAD_END events parsed")

    backend_rows_raw = load_backend_quanta_csv(args.compare_csv)
    backend_rows = backend_rows_to_name_rows(backend_rows_raw)
    edges = derive_edges_from_backend_rows(backend_rows_raw)

    intervals = build_intervals(events, args.owner_mode)
    replay_rows = replay_to_rows_by_name(intervals, edges, args.thread_id)

    interval_rows = [
        {
            "start_ns": iv.start_ns,
            "finish_ns": iv.finish_ns,
            "dur_ns": iv.dur_ns,
            "owner_token_id": -1 if iv.owner_token_id is None else iv.owner_token_id,
            "owner_token_name": "" if iv.owner_token_name is None else iv.owner_token_name,
            "stack": " > ".join(iv.stack_names),
        }
        for iv in intervals
    ]

    summary_rows = []
    agg = aggregate_name_rows(replay_rows)
    total = sum(agg.values())
    for token_name, excl_ns in sorted(agg.items(), key=lambda kv: (-kv[1], kv[0])):
        summary_rows.append(
            {
                "token_name": token_name,
                "excl_ns": excl_ns,
                "proportion": (excl_ns / total) if total else 0.0,
            }
        )

    out_prefix = Path(args.out_prefix)

    write_csv(
        out_prefix.with_name(out_prefix.name + "_intervals.csv"),
        interval_rows,
        ["start_ns", "finish_ns", "dur_ns", "owner_token_id", "owner_token_name", "stack"],
    )
    write_csv(
        out_prefix.with_name(out_prefix.name + "_quanta_by_name.csv"),
        replay_rows,
        ["start_ns", "finish_ns", "thread_id", "token_name", "excl_ns", "proportion"],
    )
    write_csv(
        out_prefix.with_name(out_prefix.name + "_summary_by_name.csv"),
        summary_rows,
        ["token_name", "excl_ns", "proportion"],
    )
    write_csv(
        out_prefix.with_name(out_prefix.name + "_backend_by_name.csv"),
        backend_rows,
        ["start_ns", "finish_ns", "thread_id", "token_name", "excl_ns", "proportion"],
    )

    report = compare_name_rows(replay_rows, backend_rows, args.top_n)
    with open(out_prefix.with_name(out_prefix.name + "_compare.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(
        {
            "mode": args.owner_mode,
            "compare_key": "token_name",
            "backend_edges": edges,
            "n_events": len(events),
            "n_intervals": len(intervals),
            "n_replay_rows": len(replay_rows),
            "n_backend_rows": len(backend_rows),
            "summary": report["summary"],
            "top_diffs": report["largest_diffs"][:args.top_n],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
