from __future__ import annotations

from dataclasses import replace

from blup.data_model import (
    HistogramBundle,
    HistogramQuery,
    OccurrenceBundle,
    OccurrenceQuery,
    SummaryQuery,
    SummaryBundle,
)
from blup.modules.interface import WorkJob
from blup.modules.token_detail.types import (
    TokenDetailDiffRow,
    TokenDetailHistogramResult,
    TokenDetailJob,
    TokenDetailRequest,
    TokenDetailResult,
    TokenDetailScatterResult,
    TokenDetailTableModel,
    TokenDetailTableResult,
    TokenDetailTraceContext,
    TokenDetailUpdate,
    token_to_select_value,
)
from blup.types import ThreadID, ThreadName, TokenKey, TokenName, TraceSide
from blup.utils import as_token_key, format_duration_delta_ns, format_duration_ns, format_percent_diff, timed


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


def _thread_ids_for(
    trace_ctx: TokenDetailTraceContext,
    active_thread_names: tuple[ThreadName, ...],
) -> tuple[ThreadID, ...]:
    return tuple(
        trace_ctx.thread_name_to_id[name]
        for name in active_thread_names
        if name in trace_ctx.thread_name_to_id
    )

def _empty_histogram_source() -> dict:
    return {"left": [], "right": [], "top": []}

def _empty_scatter_source() -> dict:
    return {"x": [], "y": []}


class TokenDetailAssembler:

    def __init__(self) -> None:
        pass

    def prepare_request(self, update: TokenDetailUpdate) -> TokenDetailRequest:
        jobs: list[TokenDetailJob] = [
            TokenDetailJob(kind="table", update=update),
        ]
        if update.selected_token is not None:
            for side in update.context.trace_context:
                jobs.append(
                    TokenDetailJob(
                        kind        = update.chart_mode,
                        trace_side  = side,
                        update      = update,
                    )
                )

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
    ) -> TokenDetailResult:
        payload = job.payload

        if payload.kind == "table":
            return self._run_table_job(payload.update)

        side = payload.trace_side
        if side is None:
            raise ValueError(
                f"token_detail {payload.kind} job requires a trace_side"
            )

        if payload.kind == "histogram":
            return self._run_histogram_job(payload.update, side)
        if payload.kind == "scatter":
            return self._run_scatter_job(payload.update, side)
        raise ValueError(f"invalid token_detail job kind: {payload.kind!r}")

    # ---------------- table job ----------------

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
            upper_names = upper_ctx.token_name_by_key,
            lower_names = (
                {} if lower_ctx is None else lower_ctx.token_name_by_key
            ),
        )

        model = self._build_table_model(
            rows,
            selected_token  = update.selected_token,
            dual_mode       = lower_ctx is not None,
            upper_label     = upper_ctx.label,
            lower_label     = "" if lower_ctx is None else lower_ctx.label,
        )
        return TokenDetailTableResult(model=model)

    def _query_summary(
        self,
        trace_ctx: TokenDetailTraceContext,
        update: TokenDetailUpdate,
    ) -> SummaryBundle:
        ctx = update.context
        thread_ids = _thread_ids_for(trace_ctx, update.active_thread_names)
        query = SummaryQuery(
            thread_ids      = tuple(sorted(thread_ids)),
            fidelity        = ctx.fidelity,
            token_mode      = ctx.token_mode,
            top_k           = ctx.top_k,
            block_only      = True,
        )
        return trace_ctx.summarize_tokens(query)

    def _build_rows(
        self,
        upper_summary: SummaryBundle,
        lower_summary: SummaryBundle | None,
        *,
        upper_names: dict[TokenKey, TokenName],
        lower_names: dict[TokenKey, TokenName],
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
        selected_token: TokenKey | None,
        dual_mode: bool,
        upper_label: str,
        lower_label: str,
    ) -> TokenDetailTableModel:
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
            )

        return TokenDetailTableModel(
            title="Token detail",
            subtitle=f"{row.name} ({row.token_type}:{row.token_id})",
            metric=_METRIC_NAMES,
            upper=(
                "—",
                "—",
                "—",
                str(row.call_count_upper),
                format_duration_ns(row.mean_incl_ns_upper),
                format_duration_ns(row.mean_excl_ns_upper),
                format_duration_ns(row.incl_total_ns_upper),
                format_duration_ns(row.excl_total_ns_upper),
            ),
            lower=(
                "—",
                "—",
                "—",
                str(row.call_count_lower),
                format_duration_ns(row.mean_incl_ns_lower),
                format_duration_ns(row.mean_excl_ns_lower),
                format_duration_ns(row.incl_total_ns_lower),
                format_duration_ns(row.excl_total_ns_lower),
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
                format_percent_diff(row.call_count_upper, row.call_count_lower),
                format_percent_diff(
                    row.mean_incl_ns_upper, row.mean_incl_ns_lower
                ),
                format_percent_diff(
                    row.mean_excl_ns_upper, row.mean_excl_ns_lower
                ),
                format_percent_diff(
                    row.incl_total_ns_upper, row.incl_total_ns_lower
                ),
                format_percent_diff(
                    row.excl_total_ns_upper, row.excl_total_ns_lower
                ),
            ),
            dual_mode=dual_mode,
            upper_label=upper_label,
            lower_label=lower_label,
        )

    def _run_histogram_job(
        self,
        update: TokenDetailUpdate,
        trace_side: TraceSide,
    ) -> TokenDetailHistogramResult:
        token = update.selected_token
        if token is None:
            return TokenDetailHistogramResult(
                trace_side  = trace_side,
                src         = _empty_histogram_source(),
            )

        ctx = update.context
        trace_ctx = ctx.trace_context[trace_side]

        h = self._query_histogram(trace_ctx, update, token)
        if len(h.left_ns) != len(h.excl_ns):
            raise RuntimeError(
                f"snapshot histogram mismatch for {trace_side} trace: "
                f"{len(h.left_ns)=} {len(h.excl_ns)=}"
            )


        src = _empty_histogram_source()
        for i in range(len(h.left_ns)):
            src["left"].append(int(h.left_ns[i]) / 1e6)
            src["right"].append(int(h.right_ns[i]) / 1e6)
            src["top"].append(int(h.excl_ns[i]) / 1e6)

        return TokenDetailHistogramResult(trace_side=trace_side, src=src)

    def _query_histogram(
        self,
        trace_ctx: TokenDetailTraceContext,
        update: TokenDetailUpdate,
        token: TokenKey,
    ) -> HistogramBundle:
        ctx = update.context
        thread_ids = _thread_ids_for(trace_ctx, update.active_thread_names)
        query = HistogramQuery(
            thread_ids      = thread_ids,
            token           = token,
            t0_ns           = update.start_ns,
            t1_ns           = update.end_ns,
            n_bins          = ctx.histogram_bins,
            token_mode      = ctx.token_mode,
        )
        with timed(f"query_histogram[{query.fidelity}]"):
            return trace_ctx.query_histogram(query)

# ---------------- scatter job ----------------

    def _run_scatter_job(
        self,
        update: TokenDetailUpdate,
        trace_side: TraceSide,
    ) -> TokenDetailScatterResult:
        token = update.selected_token
        if token is None:
            return TokenDetailScatterResult(
                trace_side=trace_side, src=_empty_scatter_source()
            )

        ctx = update.context
        trace_ctx = ctx.trace_context[trace_side]

        occ = self._query_occurrences(trace_ctx, update, token)
        if len(occ.start_ns) != len(occ.dur_ns):
            raise RuntimeError(
                f"occurrence stream mismatch for {trace_side} trace: "
                f"{len(occ.start_ns)=} {len(occ.dur_ns)=}"
            )

        src = _empty_scatter_source()
        for i in range(len(occ.start_ns)):
            src["x"].append(int(occ.start_ns[i]) / 1e6)
            src["y"].append(int(occ.dur_ns[i]) / 1e6)

        return TokenDetailScatterResult(trace_side=trace_side, src=src)

    def _query_occurrences(
        self,
        trace_ctx: TokenDetailTraceContext,
        update: TokenDetailUpdate,
        token: TokenKey,
    ) -> OccurrenceBundle:
        ctx = update.context
        thread_ids = _thread_ids_for(trace_ctx, update.active_thread_names)
        query = OccurrenceQuery(
            thread_ids      = thread_ids,
            token           = token,
            # NOTE: hardcoded for now
            fidelity        = "exact",
            token_mode      = ctx.token_mode,
            t0_ns           = update.start_ns,
            t1_ns           = update.end_ns,
            max_points      = None,
        )
        with timed(f"query_occurrences[{query.fidelity}]"):
            return trace_ctx.query_occurrences(query)


