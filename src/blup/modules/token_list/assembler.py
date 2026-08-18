from __future__ import annotations

from blup.data_model import SummaryQuery, TraceSummary, as_token_key
from blup.modules.interface import WorkJob
from blup.modules.token_list.types import (
    SortDirection,
    TokenListJob,
    TokenListOrder,
    TokenListRequest,
    TokenListResult,
    TokenListRow,
    TokenListTraceContext,
    TokenListUpdate,
)


# TODO: duplicated from token_detail.assembler; find a better home for both
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
    trace_ctx: TokenListTraceContext,
    active_thread_names: tuple[str, ...],
) -> tuple[int, ...]:
    return tuple(
        trace_ctx.thread_name_to_id[name]
        for name in active_thread_names
        if name in trace_ctx.thread_name_to_id
    )

def _empty_source() -> dict:
    return {
        "color":        [],
        "name":         [],
        "id_label":     [],
        "token_key":    [],
        "token_type":   [],
        "token_id":     [],
        "rank":         [],
        "excl_upper":   [],
        "excl_lower":   [],
        "delta_excl":   [],
        "share":        [],
    }

def _order_key_fn(order: TokenListOrder):
    if order == "delta":
        return lambda r: abs(r.delta_excl_total_ns)
    if order == "excl":
        return lambda r: r.excl_total_ns_upper
    if order == "calls":
        return lambda r: r.call_count_upper
    if order == "name":
        return lambda r: r.name.lower()
    if order == "token":
        return lambda r: (r.token_type, r.token_id)
    raise ValueError(f"invalid token_list order: {order!r}")


class TokenListAssembler:

    def __init__(self) -> None:
        pass

    def prepare_request(self, update: TokenListUpdate) -> TokenListRequest:
        return TokenListRequest(
            request_key     = update.request_key,
            jobs            = (TokenListJob(update=update),),
        )

    def unroll_job_list(
        self,
        request: TokenListRequest,
    ) -> list[WorkJob[TokenListJob]]:
        return [
            WorkJob(id=i, payload=job)
            for i, job in enumerate(request.jobs)
        ]

    def run_job(self, job: WorkJob[TokenListJob]) -> TokenListResult:
        update = job.payload.update
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
        rows = self._order_rows(rows, ctx.order, ctx.direction)

        src = self._build_source(
            rows,
            color_map   = ctx.color_map,
            dual_mode   = lower_ctx is not None,
        )
        return TokenListResult(src=src)

    def _query_summary(
        self,
        trace_ctx: TokenListTraceContext,
        update: TokenListUpdate,
    ) -> TraceSummary:
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
        upper_summary: TraceSummary,
        lower_summary: TraceSummary | None,
        *,
        upper_names: dict[str, str],
        lower_names: dict[str, str],
    ) -> list[TokenListRow]:
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

        merged: list[TokenListRow] = []
        for token_type, token_id in keys:
            a = by_upper.get((token_type, token_id))
            b = by_lower.get((token_type, token_id))

            c1 = 0 if a is None else a.call_count
            c2 = 0 if b is None else b.call_count
            e1 = 0 if a is None else a.excl_total_ns
            e2 = 0 if b is None else b.excl_total_ns

            token_key = as_token_key(token_type, token_id)
            if token_key in upper_names:
                name = upper_names[token_key]
            elif token_key in lower_names:
                name = lower_names[token_key]
            else:
                name = f"{token_type}:{token_id}"

            merged.append(
                TokenListRow(
                    token_type=token_type,
                    token_id=token_id,
                    token_key=token_key,
                    name=name,
                    call_count_upper=c1,
                    call_count_lower=c2,
                    excl_total_ns_upper=e1,
                    excl_total_ns_lower=e2,
                    delta_excl_total_ns=e2 - e1,
                    contribution_rank=0,
                    contribution_share_pct=0.0,
                )
            )

        ranked = sorted(
            merged,
            key=lambda r: (
                -abs(r.delta_excl_total_ns),
                r.name,
                r.token_type,
                r.token_id,
            ),
        )
        total_abs = sum(abs(r.delta_excl_total_ns) for r in ranked)

        out: list[TokenListRow] = []
        for idx, r in enumerate(ranked, start=1):
            share = (
                0.0
                if total_abs == 0
                else 100.0 * abs(r.delta_excl_total_ns) / total_abs
            )
            out.append(
                TokenListRow(
                    token_type              = r.token_type,
                    token_id                = r.token_id,
                    token_key               = r.token_key,
                    name                    = r.name,
                    call_count_upper        = r.call_count_upper,
                    call_count_lower        = r.call_count_lower,
                    excl_total_ns_upper     = r.excl_total_ns_upper,
                    excl_total_ns_lower     = r.excl_total_ns_lower,
                    delta_excl_total_ns     = r.delta_excl_total_ns,
                    contribution_rank       = idx,
                    contribution_share_pct  = share,
                )
            )
        return out

    def _order_rows(
        self,
        rows: list[TokenListRow],
        order: TokenListOrder,
        direction: SortDirection,
    ) -> list[TokenListRow]:
        if direction not in ("asc", "desc"):
            raise ValueError(f"invalid token_list direction: {direction!r}")
        rows = sorted(rows, key=lambda r: (r.token_type, r.token_id))
        return sorted(
            rows,
            key=_order_key_fn(order),
            reverse=(direction == "desc"),
        )

    def _build_source(
        self,
        rows: list[TokenListRow],
        *,
        color_map: dict[str, str],
        dual_mode: bool,
    ) -> dict:
        src = _empty_source()
        for r in rows:
            src["color"].append(color_map.get(r.token_key, "#999999"))
            src["name"].append(r.name)
            src["id_label"].append(f"{r.token_type}:{r.token_id}")
            src["token_key"].append(r.token_key)
            src["token_type"].append(r.token_type)
            src["token_id"].append(r.token_id)
            src["rank"].append(r.contribution_rank)
            src["excl_upper"].append(
                _format_duration_ns(r.excl_total_ns_upper)
            )
            if dual_mode:
                src["excl_lower"].append(
                    _format_duration_ns(r.excl_total_ns_lower)
                )
                src["delta_excl"].append(
                    _format_duration_delta_ns(r.delta_excl_total_ns)
                )
                src["share"].append(f"{r.contribution_share_pct:.1f}%")
            else:
                src["excl_lower"].append("")
                src["delta_excl"].append("")
                src["share"].append("")
        return src

