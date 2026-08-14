from __future__ import annotations

from dataclasses import replace

from blup.data_model import (
    SnapshotHistogram,
    SnapshotHistogramQuery,
    SummaryQuery,
    TraceSummary,
    as_token_key,
)
from blup.modules.interface import WorkJob
from blup.modules.token_detail.types import (
    TokenDetailDiffRow,
    TokenDetailHistogramResult,
    TokenDetailJob,
    TokenDetailRequest,
    TokenDetailTableModel,
    TokenDetailTableResult,
    TokenDetailTraceContext,
    TokenDetailUpdate,
    token_to_select_value,
)
from blup.state import TokenDetailTableMode

_METRIC_NAMES = (
    "Contribution rank",
    "Contribution abs",
    "Contribution share",
    "Calls",
    "Mean inclusive",
    "Mean exclusive",
    "Total inclusive",
    "Total exclusive",
)
_COMPACT_METRIC_INDEXES = (2, 3, 5, 7)


def _format_percent_diff(v1: float, v2: float) -> str:
    if v1 == 0:
        return "0.0%" if v2 == 0 else "—"
    pct = ((v2 - v1) / v1) * 100.0
    return f"{pct:+.1f}%"

def _format_duration_ns(value: float) -> str:
    sign = "-" if value < 0 else ""
    x = abs(float(value))
    if x < 1_000:
        return f"{sign}{x:.0f} ns"
    if x < 1_000_000:
        return f"{sign}{x / 1_000:.3f} us"
    if x < 1_000_000_000:
        return f"{sign}{x / 1_000_000:.3f} ms"
    return f"{sign}{x / 1_000_000_000:.3f} s"

def _format_duration_delta_ns(value: float) -> str:
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

def _thread_ids_for(
    trace_ctx: TokenDetailTraceContext,
    active_thread_names: tuple[str, ...],
) -> tuple[int, ...]:
    return tuple(
        trace_ctx.thread_name_to_id[name]
        for name in active_thread_names
        if name in trace_ctx.thread_name_to_id
    )

def _empty_histogram_source() -> dict:
    return {
        "left": [],
        "right": [],
        "upper": [],
        "lower": [],
        "delta": [],
    }


class TokenDetailAssembler:
    def __init__(self) -> None:
        pass

    def prepare_request(self, update: TokenDetailUpdate) -> TokenDetailRequest:
        jobs: list[TokenDetailJob] = [
            TokenDetailJob(kind="table", update=update),
        ]
        if update.selected_token is not None:
            jobs.append(TokenDetailJob(kind="histogram", update=update))

        return TokenDetailRequest(
            request_key     = update.request_key,
            jobs            = tuple(jobs),
        )

    def unroll_job_list(
        self,
        request: TokenDetailRequest,
    ) -> list[WorkJob[TokenDetailJob]]:
        return [
            WorkJob(id=i, payload=job)
            for i, job in enumerate(request.jobs)
        ]

    def run_job(
        self,
        job: WorkJob[TokenDetailJob],
    ) -> TokenDetailTableResult | TokenDetailHistogramResult:
        payload = job.payload

        if payload.kind == "table":
            return self._run_table_job(payload.update)
        if payload.kind == "histogram":
            return self._run_histogram_job(payload.update)
        raise ValueError(f"invalid token_detail job kind: {payload.kind!r}")

    def _run_table_job(self, update: TokenDetailUpdate) -> TokenDetailTableResult:
        ctx = update.context
        upper_ctx = ctx.trace_context["upper"]
        lower_ctx = ctx.trace_context.get("lower")

        upper_summary = self._query_summary(upper_ctx, update)
        lower_summary = (
            None
            if lower_ctx is None
            else self._query_summary(lower_ctx, update)
        )

        rows = self._build_rows(
            upper_summary,
            lower_summary,
            upper_names=upper_ctx.token_name_by_key,
            lower_names=(
                {} if lower_ctx is None else lower_ctx.token_name_by_key
            ),
        )

        model = self._build_table_model(
            rows,
            selected_token=update.selected_token,
            table_mode=update.table_mode,
            dual_mode=lower_ctx is not None,
            upper_label=upper_ctx.label,
            lower_label="" if lower_ctx is None else lower_ctx.label,
        )
        return TokenDetailTableResult(model=model)

    def _query_summary(
        self,
        trace_ctx: TokenDetailTraceContext,
        update: TokenDetailUpdate,
    ) -> TraceSummary:
        ctx = update.context
        thread_ids = _thread_ids_for(trace_ctx, update.active_thread_names)
        query = SummaryQuery(
            thread_ids=tuple(sorted(thread_ids)),
            fidelity=ctx.fidelity,
            token_mode=ctx.token_mode,
            top_k=ctx.top_k,
            block_only=True,
        )
        return trace_ctx.summarize_tokens(query)

    def _build_rows(
        self,
        upper_summary: TraceSummary,
        lower_summary: TraceSummary | None,
        *,
        upper_names: dict[str, str],
        lower_names: dict[str, str],
    ) -> tuple[TokenDetailDiffRow, ...]:
        by_upper = {
            (r.token_type, r.token_id): r for r in upper_summary.tokens
        }
        by_lower = (
            {}
            if lower_summary is None
            else {
                (r.token_type, r.token_id): r for r in lower_summary.tokens
            }
        )
        keys = set(by_upper) | set(by_lower)

        base_rows: list[TokenDetailDiffRow] = []
        for token_type, token_id in keys:
            a = by_upper.get((token_type, token_id))
            b = by_lower.get((token_type, token_id))

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
            if key in upper_names:
                name = upper_names[key]
            elif key in lower_names:
                name = lower_names[key]
            else:
                name = f"{token_type}:{token_id}"

            base_rows.append(
                TokenDetailDiffRow(
                    token_type=token_type,
                    token_id=token_id,
                    name=name,
                    call_count_upper=c1,
                    call_count_lower=c2,
                    incl_total_ns_upper=i1,
                    incl_total_ns_lower=i2,
                    excl_total_ns_upper=e1,
                    excl_total_ns_lower=e2,
                    mean_incl_ns_upper=mi1,
                    mean_incl_ns_lower=mi2,
                    mean_excl_ns_upper=me1,
                    mean_excl_ns_lower=me2,
                    delta_call_count=c2 - c1,
                    delta_mean_incl_ns=mi2 - mi1,
                    delta_mean_excl_ns=me2 - me1,
                    delta_incl_total_ns=i2 - i1,
                    delta_excl_total_ns=e2 - e1,
                    thread_ids_upper=() if a is None else tuple(a.thread_ids),
                    thread_ids_lower=() if b is None else tuple(b.thread_ids),
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

        rows: list[TokenDetailDiffRow] = []
        for idx, r in enumerate(base_rows, start=1):
            contrib_abs = abs(r.delta_excl_total_ns)
            contrib_share_pct = (
                0.0 if total_abs == 0 else (100.0 * contrib_abs / total_abs)
            )
            rows.append(
                replace(
                    r,
                    contribution_abs_ns=contrib_abs,
                    contribution_share_pct=contrib_share_pct,
                    contribution_rank=idx,
                )
            )

        return tuple(rows)

    def _build_table_model(
        self,
        rows: tuple[TokenDetailDiffRow, ...],
        *,
        selected_token: tuple[int, int] | None,
        table_mode: TokenDetailTableMode,
        dual_mode: bool,
        upper_label: str,
        lower_label: str,
    ) -> TokenDetailTableModel:
        options = tuple(
            (
                token_to_select_value((row.token_type, row.token_id)),
                (
                    f"#{row.contribution_rank} "
                    f"{row.name} "
                    f"({_format_duration_ns(row.contribution_abs_ns)}, "
                    f"{row.contribution_share_pct:.1f}%) "
                    f"[{row.token_type}:{row.token_id}]"
                ),
            )
            for row in rows
        )

        selected_value = token_to_select_value(selected_token)
        row = next((r for r in rows if r.token == selected_token), None)

        if row is None:
            return TokenDetailTableModel(
                title="Token detail",
                subtitle="No token selected",
                metric=(),
                upper=(),
                lower=(),
                delta=(),
                percent=(),
                dual_mode=dual_mode,
                upper_label=upper_label,
                lower_label=lower_label,
                options=options,
                selected_value=selected_value,
            )

        if table_mode == "full":
            indexes = tuple(range(len(_METRIC_NAMES)))
        elif table_mode == "compact":
            indexes = _COMPACT_METRIC_INDEXES
        else:
            raise ValueError(f"invalid table_mode: {table_mode!r}")

        upper_all = (
            "—",
            "—",
            "—",
            str(row.call_count_upper),
            _format_duration_ns(row.mean_incl_ns_upper),
            _format_duration_ns(row.mean_excl_ns_upper),
            _format_duration_ns(row.incl_total_ns_upper),
            _format_duration_ns(row.excl_total_ns_upper),
        )
        lower_all = (
            "—",
            "—",
            "—",
            str(row.call_count_lower),
            _format_duration_ns(row.mean_incl_ns_lower),
            _format_duration_ns(row.mean_excl_ns_lower),
            _format_duration_ns(row.incl_total_ns_lower),
            _format_duration_ns(row.excl_total_ns_lower),
        )
        delta_all = (
            f"#{row.contribution_rank}",
            _format_duration_ns(row.contribution_abs_ns),
            "—",
            f"{row.delta_call_count:+d}",
            _format_duration_delta_ns(row.delta_mean_incl_ns),
            _format_duration_delta_ns(row.delta_mean_excl_ns),
            _format_duration_delta_ns(row.delta_incl_total_ns),
            _format_duration_delta_ns(row.delta_excl_total_ns),
        )
        percent_all = (
            "—",
            "—",
            f"{row.contribution_share_pct:.1f}%",
            _format_percent_diff(row.call_count_upper, row.call_count_lower),
            _format_percent_diff(row.mean_incl_ns_upper, row.mean_incl_ns_lower),
            _format_percent_diff(row.mean_excl_ns_upper, row.mean_excl_ns_lower),
            _format_percent_diff(row.incl_total_ns_upper, row.incl_total_ns_lower),
            _format_percent_diff(row.excl_total_ns_upper, row.excl_total_ns_lower),
        )

        return TokenDetailTableModel(
            title="Token detail",
            subtitle=f"{row.name} ({row.token_type}:{row.token_id})",
            metric=tuple(_METRIC_NAMES[i] for i in indexes),
            upper=tuple(upper_all[i] for i in indexes),
            lower=tuple(lower_all[i] for i in indexes),
            delta=tuple(delta_all[i] for i in indexes),
            percent=tuple(percent_all[i] for i in indexes),
            dual_mode=dual_mode,
            upper_label=upper_label,
            lower_label=lower_label,
            options=options,
            selected_value=selected_value,
        )

    def _run_histogram_job(
        self,
        update: TokenDetailUpdate,
    ) -> TokenDetailHistogramResult:
        token = update.selected_token
        if token is None:
            return TokenDetailHistogramResult(src=_empty_histogram_source())

        ctx = update.context
        upper_ctx = ctx.trace_context["upper"]
        lower_ctx = ctx.trace_context.get("lower")

        h_upper = self._query_histogram(upper_ctx, update, token)
        h_lower = (
            None
            if lower_ctx is None
            else self._query_histogram(lower_ctx, update, token)
        )

        if h_lower is not None:
            left_ns = (
                h_upper.left_ns if len(h_upper.left_ns) else h_lower.left_ns
            )
            right_ns = (
                h_upper.right_ns if len(h_upper.right_ns) else h_lower.right_ns
            )
            if len(left_ns) != len(h_upper.excl_ns):
                raise RuntimeError(
                    "snapshot histogram mismatch for upper trace: "
                    f"{len(left_ns)=} {len(h_upper.excl_ns)=}"
                )
            if len(left_ns) != len(h_lower.excl_ns):
                raise RuntimeError(
                    "snapshot histogram mismatch for lower trace: "
                    f"{len(left_ns)=} {len(h_lower.excl_ns)=}"
                )
            lower_excl_ns = h_lower.excl_ns
        else:
            left_ns = h_upper.left_ns
            right_ns = h_upper.right_ns
            if len(left_ns) != len(h_upper.excl_ns):
                raise RuntimeError(
                    "snapshot histogram mismatch for upper trace: "
                    f"{len(left_ns)=} {len(h_upper.excl_ns)=}"
                )
            lower_excl_ns = tuple(0 for _ in h_upper.excl_ns)

        src = _empty_histogram_source()
        for i in range(len(left_ns)):
            upper_ms = int(h_upper.excl_ns[i]) / 1e6
            lower_ms = int(lower_excl_ns[i]) / 1e6
            src["left"].append(int(left_ns[i]) / 1e6)
            src["right"].append(int(right_ns[i]) / 1e6)
            src["upper"].append(upper_ms)
            src["lower"].append(lower_ms)
            src["delta"].append(lower_ms - upper_ms)

        return TokenDetailHistogramResult(src=src)

    def _query_histogram(
        self,
        trace_ctx: TokenDetailTraceContext,
        update: TokenDetailUpdate,
        token: tuple[int, int],
    ) -> SnapshotHistogram:
        ctx = update.context
        thread_ids = _thread_ids_for(trace_ctx, update.active_thread_names)
        query = SnapshotHistogramQuery(
            thread_ids=thread_ids,
            token=token,
            t0_ns=update.start_ns,
            t1_ns=update.end_ns,
            n_bins=ctx.histogram_bins,
            token_mode=ctx.token_mode,
        )
        return trace_ctx.query_histogram(query)
