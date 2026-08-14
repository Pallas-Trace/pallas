from __future__ import annotations

import math

import numpy as np
from bokeh.models.layouts import LayoutDOM
from typing import TYPE_CHECKING

from blup.modules.interface import UIWorkRequest
from blup.modules.time_profile.assembler import TimeProfileAssembler
from blup.modules.time_profile.chart import TimeProfileChartSurface
from blup.modules.time_profile.types import (
    TimeProfileTraceContext,
    TimeProfileUpdateContext,
    TimeProfileUpdate,
    TimeProfileResult,
    TraceSide,
)
from blup.state import (
    ContextPatch,
    ModuleID,
    TimeScopePatch,
    TokenSelectionPatch,
    TraceMode,
)

if TYPE_CHECKING:
    from blup.controller import AppController
    from blup.traces.session import TraceSession


class TimeProfilePipeline:
    module_id: ModuleID = "time_profile"

    root: LayoutDOM | None
    host: "AppController | None"
    assembler: TimeProfileAssembler

    chart: TimeProfileChartSurface

    _pending_update: TimeProfileUpdate | None
    _active_update: TimeProfileUpdate | None

    def __init__(self, *, height: int = 700) -> None:
        self.root = None
        self.host = None
        self.assembler = TimeProfileAssembler()

        self.chart = TimeProfileChartSurface(height=height)

        self._pending_update = None
        self._active_update = None

        self._callbacks_bound = False
        self._ignore_range_callbacks = False
        self._range_apply_scheduled = False

    @property
    def subscribed_state(self) -> tuple[str, ...]:
        return (
            "context.traces.trace_ids",
            "context.active_threads",
            "context.token_mode",
            "context.time_scope",
            "modules.time_profile",
        )

    def build(self) -> LayoutDOM:
        self.root = self.chart.build()
        return self.root

    def bind(self, host: "AppController") -> None:
        self.host = host

        self.chart.on_token_selected = (
            lambda token: host.update_state(
                context = ContextPatch(
                    selection = TokenSelectionPatch(
                        token=token,
                    ),
                ),
            )
        )

        # TODO: wire cursor_position into state as 'selected time'
        #       i.e. self.chart.on_cursor_position = 'set cursor position'

        # bind figure callbacks
        fig = self.chart.fig
        if fig is None or self._callbacks_bound:
            return
        fig.x_range.on_change("start", self._on_time_range_changed)         # type: ignore
        fig.x_range.on_change("end", self._on_time_range_changed)           # type: ignore
        self._callbacks_bound = True

    def refresh(self, host: "AppController") -> None:
        update = self.prepare_update(host)
        self._pending_update = update

        request = self.assembler.prepare_request(update)

        host.work_manager.submit(
            UIWorkRequest(
                request     = request,
                pipeline    = self,
            )
        )

    def prepare_update(self, host: "AppController") -> TimeProfileUpdate:
        app_ctx = host.state.context
        time_scope = app_ctx.time_scope
        trace_ids = app_ctx.traces.trace_ids

        # check trace sessions
        sessions = host.trace_registry.get_sessions(trace_ids)
        if not sessions:
            raise RuntimeError(
                "Time profile module requires at least one selected trace"
            )
        upper = sessions[0]
        lower = sessions[1] if len(sessions) >= 2 else None
        trace_mode: TraceMode = "dual" if lower is not None else "single"

        # check trace threads
        available_threads = host.trace_registry.thread_names_for(
            trace_ids,
        )
        available_thread_set = set(available_threads)
        active_threads = tuple(
            thread_name
            for thread_name in app_ctx.active_threads
            if thread_name in available_thread_set
        )

        # check time bounds
        bounds = host.trace_registry.time_bounds_for(trace_ids)
        if bounds is None:
            raise RuntimeError(
                "No available time bounds in selected traces"
            )
        full_start_ns, full_end_ns = bounds
        if time_scope.t0_ns is None or time_scope.t1_ns is None:
            start_ns = full_start_ns
            end_ns = full_end_ns
            sync_range_to_fig = True
        else:
            start_ns = max(full_start_ns, time_scope.t0_ns)
            end_ns = min(full_end_ns, time_scope.t1_ns)
            sync_range_to_fig = False

        if end_ns <= start_ns:
            start_ns = full_start_ns
            end_ns = full_end_ns
            sync_range_to_fig = True

        # prepare update context
        update_ctx = self._freeze_update_context(
            host                    = host,
            upper_session           = upper,
            lower_session           = lower,
            active_thread_names     = active_threads,
            start_ns                = int(start_ns),
            end_ns                  = int(end_ns),
            trace_mode              = trace_mode,
        )

        return TimeProfileUpdate(
            active_thread_names     = active_threads,
            start_ns                = int(start_ns),
            end_ns                  = int(end_ns),
            sync_range_to_fig       = sync_range_to_fig,
            trace_mode              = trace_mode,
            context                 = update_ctx,
        )

    def start_update(self) -> None:
        if self._pending_update is None:
            raise RuntimeError(
                "time_profile start_update called without pending update"
            )

        update = self._pending_update
        self._active_update = update
        self._pending_update = None

        self._ignore_range_callbacks = True
        try:
            self.chart.prepare_display(
                active_thread_names     = list(update.active_thread_names),
                start_ns                = update.start_ns,
                end_ns                  = update.end_ns,
                sync_range_to_fig       = update.sync_range_to_fig,
                trace_mode              = update.trace_mode,
            )
        finally:
            self._ignore_range_callbacks = False

    def apply_result(self, result: TimeProfileResult) -> None:
        update = self._active_update
        if update is None:
            return

        self.chart.apply_job_result(
            thread_name     = result.thread_name,
            trace_side      = result.trace_side,
            src             = result.src,
            trace_mode      = update.trace_mode,
        )

    def finish_update(self, *, cancelled: bool) -> None:
        self._active_update = None

    def _freeze_update_context(
        self,
        host: "AppController",
        upper_session: TraceSession,
        lower_session: TraceSession | None,
        active_thread_names: tuple[str, ...],
        start_ns: int,
        end_ns: int,
        trace_mode: str,
    ) -> TimeProfileUpdateContext:
        app_ctx = host.state.context
        mod_cfg = host.state.modules.time_profile
        trace_ids = app_ctx.traces.trace_ids

        upper_trace_id = trace_ids[0]
        lower_trace_id = (
            trace_ids[1]
            if lower_session is not None
            else None
        )

        bin_edges_ns = tuple(
            int(x)
            for x in np.linspace(
                start_ns,
                end_ns,
                int(mod_cfg.n_bins) + 1,
                dtype=np.int64,
            )
        )

        token_keys = set(upper_session.meta.token_key_to_name.keys())
        trace_context: dict[TraceSide, TimeProfileTraceContext] = {
            "upper": TimeProfileTraceContext(
                thread_name_to_id = {
                    str(k): int(v)
                    for k, v in upper_session.meta.thread_name_to_id.items()
                },
                token_name_by_key = dict(upper_session.meta.token_key_to_name),
                query_quanta = (
                    lambda query, session=upper_session
                        : session.query_quanta(query)
                ),
            )
        }
        if lower_session is not None:
            token_keys.update(lower_session.meta.token_key_to_name.keys())
            trace_context["lower"] = TimeProfileTraceContext(
                thread_name_to_id = {
                    str(k): int(v)
                    for k, v in lower_session.meta.thread_name_to_id.items()
                },
                token_name_by_key = dict(lower_session.meta.token_key_to_name),
                query_quanta = (
                    lambda query, session=lower_session
                        : session.query_quanta(query)
                ),
            )

        request_key = (
            upper_trace_id,
            lower_trace_id,
            active_thread_names,
            int(mod_cfg.n_bins),
            mod_cfg.fidelity,
            trace_mode,
            app_ctx.token_mode,
            mod_cfg.order,
            start_ns,
            end_ns,
        )

        color_map = dict(host.token_color.snapshot(token_keys).color_map)

        return TimeProfileUpdateContext(
            request_key     = request_key,
            trace_context   = trace_context,
            bin_edges_ns    = bin_edges_ns,
            fidelity        = mod_cfg.fidelity,
            token_mode      = app_ctx.token_mode,
            order           = mod_cfg.order,
            color_map       = color_map,
        )

    def _on_time_range_changed(self, attr, old, new) -> None:
        if self._ignore_range_callbacks:
            return
        if self._range_apply_scheduled:
            return
        if self.chart.doc is None:
            return

        self._range_apply_scheduled = True
        self.chart.doc.add_next_tick_callback(self._apply_time_range_change)

    def _apply_time_range_change(self) -> None:
        self._range_apply_scheduled = False

        host = self.host
        fig = self.chart.fig
        if host is None or fig is None:
            return

        start_ms = fig.x_range.start                                        # type: ignore
        end_ms = fig.x_range.end                                            # type: ignore
        if start_ms is None or end_ms is None:
            return
        if not math.isfinite(start_ms) or not math.isfinite(end_ms):
            return
        if end_ms <= start_ms:
            return

        full_t0_ns, full_t1_ns = host.full_time_bounds()
        new_t0_ns = max(full_t0_ns, int(start_ms * 1e6))
        new_t1_ns = min(full_t1_ns, int(end_ms * 1e6))
        if new_t1_ns <= new_t0_ns:
            return

        full_t0_ms = full_t0_ns / 1e6
        full_t1_ms = full_t1_ns / 1e6
        eps_ms = 1e-9

        if (
            abs(start_ms - full_t0_ms) <= eps_ms
            and abs(end_ms - full_t1_ms) <= eps_ms
        ):
            next_t0_ns = None
            next_t1_ns = None
        else:
            next_t0_ns = new_t0_ns
            next_t1_ns = new_t1_ns

        scope = host.state.context.time_scope
        if scope.t0_ns == next_t0_ns and scope.t1_ns == next_t1_ns:
            return

        host.update_state(
            context = ContextPatch(
                time_scope = TimeScopePatch(
                    t0_ns=next_t0_ns,
                    t1_ns=next_t1_ns,
                )
            )
        )


