from __future__ import annotations

from typing import TYPE_CHECKING

from bokeh.models.layouts import LayoutDOM

from blup.data_model import FidelityMode
from blup.modules.interface import UIWorkRequest
from blup.modules.token_detail.assembler import TokenDetailAssembler
from blup.modules.token_detail.surface import TokenDetailSurface
from blup.modules.token_detail.types import (
    TokenDetailHistogramResult,
    TokenDetailResult,
    TokenDetailTableModel,
    TokenDetailTableResult,
    TokenDetailTraceContext,
    TokenDetailUpdate,
    TokenDetailUpdateContext,
    TraceSide,
    select_value_to_token,
    token_to_select_value,
)
from blup.state import (
    ContextPatch,
    ModuleID,
    TokenSelectionPatch,
    TraceMode,
)

if TYPE_CHECKING:
    from blup.controller import AppController
    from blup.traces.session import TraceSession


class TokenDetailPipeline:
    module_id: ModuleID = "token_detail"

    # TODO: lift these to state level options
    _TOP_K = 32
    _HISTOGRAM_BINS = 20
    _FIDELITY: FidelityMode = "fast"

    root: LayoutDOM | None
    host: "AppController | None"
    surface: TokenDetailSurface
    assembler: TokenDetailAssembler

    _pending_update: TokenDetailUpdate | None
    _active_update: TokenDetailUpdate | None

    def __init__(self, *, height: int = 700) -> None:
        self.root = None
        self.host = None

        self.surface = TokenDetailSurface(height=height)
        self.assembler = TokenDetailAssembler()

        self._pending_update = None
        self._active_update = None

    def build(self) -> LayoutDOM:
        self.root = self.surface.build()
        return self.root

    def bind(self, host: "AppController") -> None:
        self.host = host
        self.surface.on_token_selected = (
            lambda token: host.update_state(
                context=ContextPatch(
                    selection=TokenSelectionPatch(
                        token=token,
                    ),
                ),
            )
        )

    def refresh(self, host: "AppController") -> None:
        update = self.prepare_update(host)
        self._pending_update = update

        request = self.assembler.prepare_request(update)

        host.work_manager.submit(
            UIWorkRequest(
                request=request,
                pipeline=self,
            )
        )

    def prepare_update(self, host: "AppController") -> TokenDetailUpdate:
        app_ctx = host.state.context
        trace_ids = app_ctx.traces.trace_ids

        sessions = host.trace_registry.get_sessions(trace_ids)
        if not sessions:
            raise RuntimeError(
                "Token detail module requires at least one selected trace"
            )

        upper = sessions[0]
        lower = sessions[1] if len(sessions) >= 2 else None

        trace_mode: TraceMode = "dual" if lower is not None else "single"

        bounds = host.trace_registry.time_bounds_for(trace_ids)
        if bounds is None:
            raise RuntimeError(
                "No available time bounds in selected traces"
            )

        full_start_ns, full_end_ns = bounds

        available_threads = host.trace_registry.thread_names_for(
            trace_ids,
        )
        available_thread_set = set(available_threads)

        active_threads = tuple(
            thread_name
            for thread_name in app_ctx.active_threads
            if thread_name in available_thread_set
        )

        time_scope = app_ctx.time_scope
        if time_scope.t0_ns is None or time_scope.t1_ns is None:
            start_ns = full_start_ns
            end_ns = full_end_ns
        else:
            start_ns = max(full_start_ns, time_scope.t0_ns)
            end_ns = min(full_end_ns, time_scope.t1_ns)

        if end_ns <= start_ns:
            start_ns = full_start_ns
            end_ns = full_end_ns

        update_ctx = self._freeze_update_context(
            host=host,
            upper_session=upper,
            lower_session=lower,
            active_thread_names=active_threads,
            start_ns=int(start_ns),
            end_ns=int(end_ns),
            trace_mode=trace_mode,
        )

        mod_cfg = host.state.modules.token_detail

        return TokenDetailUpdate(
            active_thread_names=active_threads,
            start_ns=int(start_ns),
            end_ns=int(end_ns),
            selected_token=app_ctx.selection.token,
            trace_mode=trace_mode,
            chart_mode=mod_cfg.chart_mode,
            table_mode=mod_cfg.table_mode,
            show_stats=mod_cfg.show_stats,
            show_chart=mod_cfg.show_chart,
            context=update_ctx,
        )

    def start_update(self) -> None:
        if self._pending_update is None:
            raise RuntimeError(
                "token_detail start_update called without pending update"
            )

        update = self._pending_update
        self._active_update = update
        self._pending_update = None

        self.surface.prepare_display(
            start_ns=update.start_ns,
            end_ns=update.end_ns,
            trace_mode=update.trace_mode,
            chart_mode=update.chart_mode,
            show_stats=update.show_stats,
            show_chart=update.show_chart,
        )

    def apply_result(self, result: TokenDetailResult) -> None:
        update = self._active_update
        if update is None:
            return

        if isinstance(result, TokenDetailTableResult):
            self.surface.apply_table_result(result.model)
            self._maybe_fix_selection(update, result.model)
        elif isinstance(result, TokenDetailHistogramResult):
            self.surface.apply_histogram_result(result.src)
        else:
            raise TypeError(
                f"unexpected token_detail result: {type(result)!r}"
            )

    def finish_update(self, *, cancelled: bool) -> None:
        self._active_update = None

    def _freeze_update_context(
        self,
        host: "AppController",
        upper_session: "TraceSession",
        lower_session: "TraceSession | None",
        active_thread_names: tuple[str, ...],
        start_ns: int,
        end_ns: int,
        trace_mode: TraceMode,
    ) -> TokenDetailUpdateContext:
        app_ctx = host.state.context
        mod_cfg = host.state.modules.token_detail
        trace_ids = app_ctx.traces.trace_ids

        upper_trace_id = trace_ids[0]
        lower_trace_id = (
            trace_ids[1] if lower_session is not None else None
        )

        trace_context: dict[TraceSide, TokenDetailTraceContext] = {
            "upper": self._trace_context_for(upper_session, upper_trace_id),
        }
        if lower_session is not None and lower_trace_id is not None:
            trace_context["lower"] = self._trace_context_for(
                lower_session, lower_trace_id
            )

        request_key = (
            upper_trace_id,
            lower_trace_id,
            active_thread_names,
            app_ctx.selection.token,
            app_ctx.token_mode,
            self._FIDELITY,
            self._TOP_K,
            self._HISTOGRAM_BINS,
            mod_cfg.chart_mode,
            mod_cfg.table_mode,
            trace_mode,
            start_ns,
            end_ns,
        )

        return TokenDetailUpdateContext(
            request_key=request_key,
            trace_context=trace_context,
            fidelity=self._FIDELITY,
            token_mode=app_ctx.token_mode,
            top_k=self._TOP_K,
            histogram_bins=self._HISTOGRAM_BINS,
        )

    def _trace_context_for(
        self,
        session: "TraceSession",
        trace_id,
    ) -> TokenDetailTraceContext:
        return TokenDetailTraceContext(
            label=self._trace_label(session, trace_id),
            thread_name_to_id={
                str(k): int(v)
                for k, v in session.meta.thread_name_to_id.items()
            },
            token_name_by_key=dict(session.meta.token_key_to_name),
            summarize_tokens=(
                lambda query, s=session: s.summarize_tokens(query)
            ),
            query_histogram=(
                lambda query, s=session: s.query_histogram(query)
            ),
        )

    def _trace_label(self, session: "TraceSession", trace_id) -> str:
        # TODO: confirm the registry/session label API; falls back to the
        # trace id so table headers stay meaningful.
        label = getattr(session, "label", None)
        return str(label) if label else str(trace_id)

    def _maybe_fix_selection(
        self,
        update: TokenDetailUpdate,
        model: TokenDetailTableModel,
    ) -> None:
        host = self.host
        if host is None:
            return

        current_value = token_to_select_value(update.selected_token)
        if current_value == "":
            return

        valid_values = {value for value, _ in model.options}
        if current_value in valid_values:
            return

        next_value = model.options[0][0] if model.options else ""
        host.update_state(
            context=ContextPatch(
                selection=TokenSelectionPatch(
                    token=select_value_to_token(next_value),
                ),
            ),
        )
