from __future__ import annotations

import math

import numpy as np
from bokeh.models.layouts import LayoutDOM
from typing import TYPE_CHECKING

from module_interface import UIWorkRequest
from modules.time_profile.assembler import TimeProfileAssembler
from modules.time_profile.chart import TimeProfileChartSurface
from modules.time_profile.types import (
    TimeProfileTraceContext,
    TimeProfileUpdateContext,
    TimeProfileUpdate,
    TimeProfileResult,
    TraceSide,
)
from state import ModuleID

if TYPE_CHECKING:
    from controller import AppController


class TimeProfilePipeline:
    module_id: ModuleID = "time_profile"

    root: LayoutDOM | None
    host: "AppController | None"
    chart: TimeProfileChartSurface
    assembler: TimeProfileAssembler

    _pending_update: TimeProfileUpdate | None
    _active_update: TimeProfileUpdate | None

    def __init__(self, *, width: int, height: int) -> None:
        self.root = None
        self.host = None

        self.chart = TimeProfileChartSurface(width=width, height=height)
        self.assembler = TimeProfileAssembler()

        self._pending_update = None
        self._active_update = None

        self._callbacks_bound = False
        self._ignore_range_callbacks = False
        self._range_apply_scheduled = False

    def build(self) -> LayoutDOM:
        self.root = self.chart.build()
        return self.root

    def bind(self, host: "AppController") -> None:
        self.host = host
        self.chart.on_token_selected = host.set_highlight_token

        # bind figure callbacks
        fig = self.chart.fig
        if fig is None or self._callbacks_bound:
            return
        fig.x_range.on_change("start", self.on_time_range_changed)          # type: ignore
        fig.x_range.on_change("end", self.on_time_range_changed)            # type: ignore
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
        t1 = host.get_primary_session()
        assert t1 is not None
        t2 = host.get_secondary_session()
        app_ctx = host.state.context

        if t2 is None:
            full_start_ns = int(t1.meta.start_ns)
            full_end_ns = int(t1.meta.end_ns)
            available_threads = set(map(str, t1.meta.thread_names))
        else:
            full_start_ns = min(int(t1.meta.start_ns), int(t2.meta.start_ns))
            full_end_ns = max(int(t1.meta.end_ns), int(t2.meta.end_ns))
            available_threads = (
                set(map(str, t1.meta.thread_names))
                | set(map(str, t2.meta.thread_names))
            )

        requested_threads = tuple(
            name for name in app_ctx.active_threads if name in available_threads
        )
        active_threads = requested_threads or tuple(sorted(available_threads))

        if app_ctx.time_scope.t0_ns is None or app_ctx.time_scope.t1_ns is None:
            start_ns = full_start_ns
            end_ns = full_end_ns
            sync_range_to_fig = True
        else:
            start_ns = app_ctx.time_scope.t0_ns
            end_ns = app_ctx.time_scope.t1_ns
            sync_range_to_fig = False

        if end_ns <= start_ns:
            end_ns = start_ns + 1

        update_ctx = self._freeze_update_context(
            host                    = host,
            active_thread_names     = active_threads,
            start_ns                = int(start_ns),
            end_ns                  = int(end_ns),
            trace_mode              = app_ctx.trace_mode,
        )

        return TimeProfileUpdate(
            active_thread_names     = active_threads,
            start_ns                = int(start_ns),
            end_ns                  = int(end_ns),
            sync_range_to_fig       = sync_range_to_fig,
            trace_mode              = app_ctx.trace_mode,
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
        active_thread_names: tuple[str, ...],
        start_ns: int,
        end_ns: int,
        trace_mode: str,
    ) -> TimeProfileUpdateContext:
        t1 = host.get_primary_session()
        assert t1 is not None
        t2 = host.get_secondary_session()

        app_ctx = host.state.context
        mod_cfg = host.state.modules.time_profile

        primary_trace_id = app_ctx.primary_trace_id
        if primary_trace_id is None:
            raise RuntimeError("primary_trace_id is unset")
        secondary_trace_id = app_ctx.secondary_trace_id

        bin_edges_ns = tuple(
            int(x)
            for x in np.linspace(
                start_ns,
                end_ns,
                int(mod_cfg.n_bins) + 1,
                dtype=np.int64,
            )
        )

        token_keys = set(t1.meta.token_key_to_name.keys())
        trace_context: dict[TraceSide, TimeProfileTraceContext] = {
            "upper": TimeProfileTraceContext(
                thread_name_to_id = {
                    str(k): int(v)
                    for k, v in t1.meta.thread_name_to_id.items()
                },
                token_name_by_key = dict(t1.meta.token_key_to_name),
                query_quanta = (
                    lambda query, session=t1
                        : session.query_quanta(query)
                ),
            )
        }
        if (
            t2 is not None 
            and trace_mode == "dual" 
            and secondary_trace_id is not None
        ):
            token_keys |= set(t2.meta.token_key_to_name.keys())
            trace_context["lower"] = TimeProfileTraceContext(
                thread_name_to_id = {
                    str(k): int(v)
                    for k, v in t2.meta.thread_name_to_id.items()
                },
                token_name_by_key = dict(t2.meta.token_key_to_name),
                query_quanta = (
                    lambda query, session=t2
                        : session.query_quanta(query)
                ),
            )

        request_key = (
            primary_trace_id,
            secondary_trace_id,
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

    def on_time_range_changed(self, attr, old, new) -> None:
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

        if abs(start_ms - full_t0_ms) <= eps_ms and abs(end_ms - full_t1_ms) <= eps_ms:
            next_t0_ns = None
            next_t1_ns = None
        else:
            next_t0_ns = new_t0_ns
            next_t1_ns = new_t1_ns

        scope = host.state.context.time_scope
        if scope.t0_ns == next_t0_ns and scope.t1_ns == next_t1_ns:
            return

        scope.t0_ns = next_t0_ns
        scope.t1_ns = next_t1_ns
        host.schedule_refresh()


