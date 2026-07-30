from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from data_model import FidelityMode, SummaryQuery, as_token_key, TokenMode, SnapshotHistogramQuery
from trace_session import TraceSession


@dataclass(frozen=True)
class SequenceSummaryDiffRow:
    token_type: int
    token_id: int
    name: str
    call_count_1: int
    call_count_2: int
    incl_total_ns_1: int
    incl_total_ns_2: int
    excl_total_ns_1: int
    excl_total_ns_2: int
    mean_incl_ns_1: float
    mean_incl_ns_2: float
    mean_excl_ns_1: float
    mean_excl_ns_2: float
    delta_call_count: int
    delta_mean_incl_ns: float
    delta_mean_excl_ns: float
    delta_incl_total_ns: int
    delta_excl_total_ns: int
    thread_ids_1: tuple[int, ...]
    thread_ids_2: tuple[int, ...]
    contribution_abs_ns: int = 0
    contribution_share_pct: float = 0.0
    contribution_rank: int = 0

    @property
    def token(self) -> tuple[int, int]:
        return (self.token_type, self.token_id)

@dataclass(frozen=True)
class SequenceSummaryDisplayModel:
    title: str
    subtitle: str
    metric: tuple[str, ...]
    primary: tuple[str, ...]
    secondary: tuple[str, ...]
    delta: tuple[str, ...]
    percent: tuple[str, ...]
    hist_left_ns: tuple[int, ...] = ()
    hist_right_ns: tuple[int, ...] = ()
    hist_primary_excl_ns: tuple[int, ...] = ()
    hist_secondary_excl_ns: tuple[int, ...] = ()
    primary_label: str = "Primary"
    secondary_label: str = "Secondary"
    dual_mode: bool = True

def format_percent_diff(v1: float, v2: float) -> str:
    if v1 == 0:
        return "0.0%" if v2 == 0 else "—"
    pct = ((v2 - v1) / v1) * 100.0
    return f"{pct:+.1f}%"

def format_duration_ns(value: float) -> str:
    sign = "-" if value < 0 else ""
    x = abs(float(value))
    if x < 1_000:
        return f"{sign}{x:.0f} ns"
    if x < 1_000_000:
        return f"{sign}{x / 1_000:.3f} us"
    if x < 1_000_000_000:
        return f"{sign}{x / 1_000_000:.3f} ms"
    return f"{sign}{x / 1_000_000_000:.3f} s"

def format_duration_delta_ns(value: float) -> str:
    if value == 0:
        return "0 ns"
    sign = "+" if value > 0 else "-"
    x = abs(float(value))
    if x < 1_000:
        return f"{sign}{x:.0f} ns"
    if x < 1_000_000:
        return f"{sign}{x / 1_000:.3f} us"
    if x < 1_000_000_000:
        return f"{sign}{x / 1_000_000:.3f} ms"
    return f"{sign}{x / 1_000_000_000:.3f} s"

class SequenceSummaryDiffAdapter:
    def __init__(self, primary: TraceSession, secondary: TraceSession) -> None:
        self.primary = primary
        self.secondary = secondary

    def _thread_ids_for_names(
        self,
        trace: TraceSession,
        active_thread_names: tuple[str, ...],
    ) -> tuple[int, ...]:
        return tuple(
            trace.meta.thread_name_to_id[name]
            for name in active_thread_names
            if name in trace.meta.thread_name_to_id
        )

    def build_rows(
        self,
        *,
        token_mode: TokenMode,
        fidelity: FidelityMode = "fast",
        top_k: int = 32,
        active_thread_names: tuple[str, ...] = (),
    ) -> tuple[SequenceSummaryDiffRow, ...]:
        t1_ids = self._thread_ids_for_names(self.primary, active_thread_names)
        t2_ids = () if self.secondary is None else self._thread_ids_for_names(self.secondary, active_thread_names)

        q1 = SummaryQuery(
            thread_ids=tuple(sorted(t1_ids)),
            fidelity=fidelity,
            token_mode=token_mode,
            top_k=top_k,
            block_only=True,
        )
        q2 = SummaryQuery(
            thread_ids=tuple(sorted(t2_ids)),
            fidelity=fidelity,
            token_mode=token_mode,
            top_k=top_k,
            block_only=True,
        )

        s1 = self.primary.summarize_tokens(q1)
        s2 = None if self.secondary is None else self.secondary.summarize_tokens(q2)

        by1 = {(r.token_type, r.token_id): r for r in s1.tokens}
        by2 = {} if s2 is None else {(r.token_type, r.token_id): r for r in s2.tokens}
        keys = set(by1) | set(by2)

        base_rows: list[SequenceSummaryDiffRow] = []
        for token_type, token_id in keys:
            a = by1.get((token_type, token_id))
            b = by2.get((token_type, token_id))

            c1 = 0 if a is None else a.call_count
            c2 = 0 if b is None else b.call_count
            i1 = 0 if a is None else a.incl_total_ns
            i2 = 0 if b is None else b.incl_total_ns
            e1 = 0 if a is None else a.excl_total_ns
            e2 = 0 if b is None else b.excl_total_ns

            mi1 = i1 / c1 if c1 else 0.0
            mi2 = i2 / c2 if c2 else 0.0
            me1 = e1 / c1 if c1 else 0.0
            me2 = e2 / c2 if c2 else 0.0

            key = as_token_key(token_type, token_id)
            if key in self.primary.meta.token_key_to_name:
                name = self.primary.meta.token_key_to_name[key]
            elif self.secondary is not None and key in self.secondary.meta.token_key_to_name:
                name = self.secondary.meta.token_key_to_name[key]
            else:
                name = f"{token_type}:{token_id}"

            base_rows.append(
                SequenceSummaryDiffRow(
                    token_type=token_type,
                    token_id=token_id,
                    name=name,
                    call_count_1=c1,
                    call_count_2=c2,
                    incl_total_ns_1=i1,
                    incl_total_ns_2=i2,
                    excl_total_ns_1=e1,
                    excl_total_ns_2=e2,
                    mean_incl_ns_1=mi1,
                    mean_incl_ns_2=mi2,
                    mean_excl_ns_1=me1,
                    mean_excl_ns_2=me2,
                    delta_call_count=c2 - c1,
                    delta_mean_incl_ns=mi2 - mi1,
                    delta_mean_excl_ns=me2 - me1,
                    delta_incl_total_ns=i2 - i1,
                    delta_excl_total_ns=e2 - e1,
                    thread_ids_1=() if a is None else a.thread_ids,
                    thread_ids_2=() if b is None else b.thread_ids,
                )
            )

        base_rows.sort(
            key=lambda r: (
                -abs(r.delta_excl_total_ns),
                -abs(r.delta_incl_total_ns),
                -abs(r.delta_call_count),
                r.name,
                r.token_type,
                r.token_id,
            )
        )
        total_abs = sum(abs(r.delta_excl_total_ns) for r in base_rows)

        rows: list[SequenceSummaryDiffRow] = []
        for idx, r in enumerate(base_rows, start=1):
            contrib_abs = abs(r.delta_excl_total_ns)
            contrib_share_pct = 0.0 if total_abs == 0 else (100.0 * contrib_abs / total_abs)
            rows.append(
                SequenceSummaryDiffRow(
                    token_type=r.token_type,
                    token_id=r.token_id,
                    name=r.name,
                    call_count_1=r.call_count_1,
                    call_count_2=r.call_count_2,
                    incl_total_ns_1=r.incl_total_ns_1,
                    incl_total_ns_2=r.incl_total_ns_2,
                    excl_total_ns_1=r.excl_total_ns_1,
                    excl_total_ns_2=r.excl_total_ns_2,
                    mean_incl_ns_1=r.mean_incl_ns_1,
                    mean_incl_ns_2=r.mean_incl_ns_2,
                    mean_excl_ns_1=r.mean_excl_ns_1,
                    mean_excl_ns_2=r.mean_excl_ns_2,
                    delta_call_count=r.delta_call_count,
                    delta_mean_incl_ns=r.delta_mean_incl_ns,
                    delta_mean_excl_ns=r.delta_mean_excl_ns,
                    delta_incl_total_ns=r.delta_incl_total_ns,
                    delta_excl_total_ns=r.delta_excl_total_ns,
                    thread_ids_1=r.thread_ids_1,
                    thread_ids_2=r.thread_ids_2,
                    contribution_abs_ns=contrib_abs,
                    contribution_share_pct=contrib_share_pct,
                    contribution_rank=idx,
                )
            )

        return tuple(rows)

    def build_display_model(
        self,
        row: SequenceSummaryDiffRow | None,
        *,
        active_thread_names: tuple[str, ...],
        token_mode: TokenMode,
        t0_ns: int,
        t1_ns: int,
        histogram_bins: int = 20,
        primary_label: str = "Primary",
        secondary_label: str | None = None,
    ) -> SequenceSummaryDisplayModel:
        dual_mode = (self.secondary is not None) and (secondary_label is not None)

        if row is None:
            return SequenceSummaryDisplayModel(
                title="Sequence summary",
                subtitle="No sequence selected",
                metric=(),
                primary=(),
                secondary=(),
                delta=(),
                percent=(),
                hist_left_ns=(),
                hist_right_ns=(),
                hist_primary_excl_ns=(),
                hist_secondary_excl_ns=(),
                primary_label=primary_label,
                secondary_label=secondary_label or "",
                dual_mode=dual_mode,
            )

        t1_ids = self._thread_ids_for_names(self.primary, active_thread_names)
        token = (row.token_type, row.token_id)

        h1 = self.primary.query_histogram(
            SnapshotHistogramQuery(
                thread_ids=t1_ids,
                token=token,
                t0_ns=t0_ns,
                t1_ns=t1_ns,
                n_bins=histogram_bins,
                token_mode=token_mode,
            )
        )

        if self.secondary is not None:
            t2_ids = self._thread_ids_for_names(self.secondary, active_thread_names)
            h2 = self.secondary.query_histogram(
                SnapshotHistogramQuery(
                    thread_ids=t2_ids,
                    token=token,
                    t0_ns=t0_ns,
                    t1_ns=t1_ns,
                    n_bins=histogram_bins,
                    token_mode=token_mode,
                )
            )

            hist_left_ns = h1.left_ns if h1.left_ns else h2.left_ns
            hist_right_ns = h1.right_ns if h1.right_ns else h2.right_ns

            if len(hist_left_ns) != len(h1.excl_ns):
                raise RuntimeError(
                    f"snapshot histogram mismatch for primary trace: "
                    f"{len(hist_left_ns)=} {len(h1.excl_ns)=}"
                )
            if len(hist_left_ns) != len(h2.excl_ns):
                raise RuntimeError(
                    f"snapshot histogram mismatch for secondary trace: "
                    f"{len(hist_left_ns)=} {len(h2.excl_ns)=}"
                )

            hist_secondary_excl_ns = h2.excl_ns
        else:
            hist_left_ns = h1.left_ns
            hist_right_ns = h1.right_ns

            if len(hist_left_ns) != len(h1.excl_ns):
                raise RuntimeError(
                    f"snapshot histogram mismatch for primary trace: "
                    f"{len(hist_left_ns)=} {len(h1.excl_ns)=}"
                )

            hist_secondary_excl_ns = tuple(0 for _ in h1.excl_ns)

        return SequenceSummaryDisplayModel(
            title="Sequence summary",
            subtitle=f"{row.name} ({row.token_type}:{row.token_id})",
            metric=(
                "Contribution rank",
                "Contribution abs",
                "Contribution share",
                "Calls",
                "Mean inclusive",
                "Mean exclusive",
                "Total inclusive",
                "Total exclusive",
            ),
            primary=(
                "—",
                "—",
                "—",
                str(row.call_count_1),
                format_duration_ns(row.mean_incl_ns_1),
                format_duration_ns(row.mean_excl_ns_1),
                format_duration_ns(row.incl_total_ns_1),
                format_duration_ns(row.excl_total_ns_1),
            ),
            secondary=(
                "—",
                "—",
                "—",
                str(row.call_count_2),
                format_duration_ns(row.mean_incl_ns_2),
                format_duration_ns(row.mean_excl_ns_2),
                format_duration_ns(row.incl_total_ns_2),
                format_duration_ns(row.excl_total_ns_2),
            ),
            delta=(
                f"#{row.contribution_rank}",
                format_duration_ns(row.contribution_abs_ns),
                "—",
                f"{row.delta_call_count:+d}",
                format_duration_delta_ns(row.delta_mean_incl_ns),
                format_duration_delta_ns(row.delta_mean_excl_ns),
                format_duration_delta_ns(row.delta_incl_total_ns),
                format_duration_delta_ns(row.delta_excl_total_ns),
            ),
            percent=(
                "—",
                "—",
                f"{row.contribution_share_pct:.1f}%",
                format_percent_diff(row.call_count_1, row.call_count_2),
                format_percent_diff(row.mean_incl_ns_1, row.mean_incl_ns_2),
                format_percent_diff(row.mean_excl_ns_1, row.mean_excl_ns_2),
                format_percent_diff(row.incl_total_ns_1, row.incl_total_ns_2),
                format_percent_diff(row.excl_total_ns_1, row.excl_total_ns_2),
            ),
            hist_left_ns=hist_left_ns,
            hist_right_ns=hist_right_ns,
            hist_primary_excl_ns=h1.excl_ns,
            hist_secondary_excl_ns=hist_secondary_excl_ns,
            primary_label=primary_label,
            secondary_label=secondary_label or "",
            dual_mode=dual_mode,
        )


