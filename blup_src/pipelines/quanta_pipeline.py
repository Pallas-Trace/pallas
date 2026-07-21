from __future__ import annotations

import math
from typing import cast

from bokeh.models.layouts import LayoutDOM

from adapters.quanta_adapter import (
    QuantaRequestSpec,
    QuantaThreadJobData,
    QuantaThreadResultData,
    build_quanta_jobs,
    build_quanta_request_spec,
    compute_quanta_thread_result,
)
from state import ViewId
from trace_session import TraceSession
from views.quanta_view import QuantaView
from work_manager import WorkJob, WorkRequestBase, WorkResult


class QuantaWorkRequest(WorkRequestBase):
    scope_key = "quanta"

    def __init__(
        self,
        pipeline: "QuantaPipeline",
        *,
        t1: TraceSession,
        t2: TraceSession | None,
        spec: QuantaRequestSpec
    ) -> None:
        self.pipeline = pipeline
        self.t1 = t1
        self.t2 = t2
        self.spec = spec
        self.request_key = spec.request_key

    def begin_apply(self) -> None:
        self.pipeline.begin_request_apply(self.spec)

    def make_jobs(self) -> list[WorkJob]:
        jobs = build_quanta_jobs(
            t1      = self.t1,
            t2      = self.t2,
            spec    = self.spec,
        )
        return [
            WorkJob(id=i, payload=job)
            for i, job in enumerate(jobs)
        ]

    def run_job(self, job: WorkJob) -> WorkResult:
        payload = cast(QuantaThreadJobData, job.payload)
        result = compute_quanta_thread_result(
            t1      = self.t1,
            t2      = self.t2,
            job     = payload,
        )
        return WorkResult(id=job.id, payload=result)

    def apply_result(self, result: WorkResult) -> None:
        payload = cast(QuantaThreadResultData, result.payload)
        self.pipeline.view.apply_thread_result(
            thread_name = payload.thread_name,
            src1        = payload.src1,
            src2        = payload.src2,
            trace_mode  = self.spec.trace_mode,
        )

    def finish_apply(self, *, cancelled: bool) -> None:
        self.pipeline.finish_request_apply(cancelled=cancelled)


class QuantaPipeline:
    view_id: ViewId = "quanta"

    def __init__(
        self,
        *,
        width: int,
        height: int,
    ) -> None:
        self.view = QuantaView(width=width, height=height)

        self._root: LayoutDOM | None = None
        self._controller = None
        self._callbacks_bound = False
        self._ignore_range_callbacks = False
        self._range_apply_scheduled = False

    # -------------------------------------------
    # |        Base Pipeline Interface          |
    # -------------------------------------------

    def build(self) -> LayoutDOM:
        self._root = self.view.build()
        return self._root

    def root(self) -> LayoutDOM | None:
        return self._root

    def bind(self, controller) -> None:
        self._controller = controller

        if self.view.on_token_selected is None:
            self.view.on_token_selected = controller.set_highlight_token

        fig = self.view.fig
        if fig is None or self._callbacks_bound:
            return

        fig.x_range.on_change("start", self.on_time_range_changed)  # type: ignore
        fig.x_range.on_change("end", self.on_time_range_changed)    # type: ignore
        self._callbacks_bound = True

    def refresh(self, controller) -> None:
        t1 = controller.get_primary_session()
        assert(t1 is not None)
        t2 = controller.get_secondary_session()

        spec = build_quanta_request_spec(
            t1                      = t1,
            t2                      = t2,
            active_thread_names     = tuple(controller.state.context.active_threads),
            n_bins                  = controller.state.views.quanta.n_bins,
            mode                    = controller.state.views.quanta.mode,
            trace_mode              = controller.state.context.trace_mode,
            token_mode              = controller.state.context.token_mode,
            stack_order             = controller.state.views.quanta.order,
            window_t0_ns            = controller.state.context.time_scope.t0_ns,
            window_t1_ns            = controller.state.context.time_scope.t1_ns,
        )

        controller.work_manager.submit(
            QuantaWorkRequest(
                self,
                t1   = t1,
                t2   = t2,
                spec = spec
            )
        )

    def begin_request_apply(self, spec: QuantaRequestSpec) -> None:
        self._ignore_range_callbacks = True
        try:
            self.view.prepare_display(
                active_thread_names = list(spec.active_threads),
                start_ns            = spec.start_ns,
                end_ns              = spec.end_ns,
                sync_range_to_fig   = spec.sync_range_to_fig,
                trace_mode          = spec.trace_mode,
            )
        finally:
            self._ignore_range_callbacks = False

    def finish_request_apply(self, *, cancelled: bool) -> None:
        return

    # -------------------------------------------
    # |              Callbacks                  |
    # -------------------------------------------

    def on_time_range_changed(self, attr, old, new) -> None:
        if self._ignore_range_callbacks:
            return
        if self._range_apply_scheduled:
            return
        if self.view.doc is None:
            return

        self._range_apply_scheduled = True
        self.view.doc.add_next_tick_callback(self.apply_time_range_change)

    def apply_time_range_change(self) -> None:
        self._range_apply_scheduled = False

        controller = self._controller
        fig = self.view.fig
        if controller is None or fig is None:
            return

        start_ms = fig.x_range.start  # type: ignore
        end_ms = fig.x_range.end      # type: ignore
        if start_ms is None or end_ms is None:
            return
        if not math.isfinite(start_ms) or not math.isfinite(end_ms):
            return
        if end_ms <= start_ms:
            return

        full_t0_ns, full_t1_ns = controller.full_time_bounds()
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

        scope = controller.state.context.time_scope
        if scope.t0_ns == next_t0_ns and scope.t1_ns == next_t1_ns:
            return

        scope.t0_ns = next_t0_ns
        scope.t1_ns = next_t1_ns
        controller.schedule_refresh()


