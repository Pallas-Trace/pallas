from __future__ import annotations

import time
import json
from typing import Optional
from collections import deque
from dataclasses import dataclass

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models.tools import TapTool
from bokeh.models.callbacks import CustomJS
from bokeh.models.sources import ColumnDataSource
from bokeh.events import DocumentReady
from bokeh.models.tools import HoverTool
from bokeh.models.ranges import Range1d
from bokeh.models.widgets.inputs import TextInput
from bokeh.plotting import figure

from data_model import FidelityMode, QuantaQuery, TokenMode, as_token_key
from trace_session import TraceSession, QuantaQuery
from adapters.quanta_adapter import (
    empty_quanta_source,
    build_color_map,
    quanta_bundle_to_bokeh_source,
)
from utils import timed


@dataclass(frozen=True)
class ThreadLoadJob:
    generation:     int
    thread_name:    str
    thread_id_1:    Optional[int]
    thread_id_2:    Optional[int]
    bin_edges_ns:   tuple[int, ...]
    mode:           FidelityMode
    token_mode:     TokenMode
    stack_order:    str


@dataclass(frozen=True)
class ThreadLoadResult:
    generation:     int
    thread_name:    str
    src1:           dict
    src2:           dict


class QuantaView:

    def __init__(
        self,
        t1: TraceSession,
        t2: TraceSession,
        *,
        width: int = 1400,
        height: int = 900,
    ) -> None:
        self.t1 = t1
        self.t2 = t2
        self.width = width
        self.height = height

        self.fig = None
        self.doc = None

        self.sources1: dict[str, ColumnDataSource] = {}
        self.sources2: dict[str, ColumnDataSource] = {}
        self.renderers1: dict[str, object] = {}
        self.renderers2: dict[str, object] = {}

        self._request_key: tuple | None = None
        self._loading_generation: int = 0
        self._completed_results: deque[ThreadLoadResult] = deque()
        self._generation = 0
        self._pending_jobs: deque[ThreadLoadJob] = deque()
        self._loader_scheduled = False

        self._current_threads: list[str] = []
        self._current_edges: tuple[int, ...] = ()
        self._current_mode: FidelityMode = "fast"
        self._current_token_mode: TokenMode = "raw"
        self._current_stack_order: str = "global"
        self._color_map: dict[str, str] = {}

        self.full_start_ns = min(self.t1.meta.start_ns, self.t2.meta.start_ns)
        self.full_end_ns = max(self.t1.meta.end_ns, self.t2.meta.end_ns)

        self.on_token_selected = None
        self._ignore_selection_callbacks = False

        self._client_ready = False

        self._bench_generation = 0
        self._bench_py_compute_ms = 0.0
        self._bench_py_apply_ms = 0.0
        self._bench_py_total_ms = 0.0
        self._bench_batch_start_ms = 0.0

    def build(self, doc=None):
        self.doc = doc or curdoc()

        all_threads = self.get_all_thread_names()
        fig = figure(
            width           = self.width,
            height          = self.height,
            x_range         = Range1d(0, 1),
            y_range         = list(reversed(all_threads)),  # type: ignore
            tools           = ["tap", "box_zoom", "xwheel_pan", "xbox_zoom", "reset", "undo", "redo", "save"],
            active_drag     = "xbox_zoom",
            output_backend  = "webgl",
            title           = "Quanta comparison",
            x_axis_label    = "Time (ms)",
        )
        fig.toolbar.active_tap = fig.select_one(TapTool)  # type: ignore
        hover_renderers = []

        for thread_name in all_threads:
            s1 = ColumnDataSource(empty_quanta_source())
            s2 = ColumnDataSource(empty_quanta_source())
            s1.selected.on_change(
                "indices",
                lambda attr, old, new, source=s1: self._on_source_selected(source, new),
            )
            s2.selected.on_change(
                "indices",
                lambda attr, old, new, source=s2: self._on_source_selected(source, new),
            )
            self.sources1[thread_name] = s1
            self.sources2[thread_name] = s2

            r1 = fig.quad(
                left="left", right="right", top="top", bottom="bottom",
                color="color", line_color=None, fill_alpha=0.90,
                source=s1, name=f"quanta_t1_{thread_name}",
                visible=False,
            )
            r2 = fig.quad(
                left="left", right="right", top="top", bottom="bottom",
                color="color", line_color=None, fill_alpha=0.90,
                source=s2, name=f"quanta_t2_{thread_name}",
                visible=False,
            )
            r1.nonselection_glyph = r1.glyph
            r2.nonselection_glyph = r2.glyph
            self.renderers1[thread_name] = r1
            self.renderers2[thread_name] = r2
            hover_renderers.extend([r1, r2])

        hover = HoverTool(
            renderers=hover_renderers,
            tooltips=[
                ("thread", "@thread"),
                ("token", "@token_name"),
                ("token_key", "@token_key"),
                ("proportion", "@proportion{0.000}"),
                ("exclusive_s", "@exclusive_s{0.000000} s"),
            ],
            point_policy="follow_mouse",
        )
        fig.add_tools(hover)

        self._client_ready_input = TextInput(value="0", visible=False)
        self._benchmark_payload_input = TextInput(value="", visible=False)

        def _on_client_ready(attr, old, new):
            if new != "1" or self._client_ready:
                return
            self._client_ready = True
            if self._pending_jobs:
                self.schedule_next_tick()

        self._client_ready_input.on_change("value", _on_client_ready)

        js_client_ready = CustomJS(args=dict(ready=self._client_ready_input), code="""
            window.pallas_startup_begin = performance.now();
            ready.value = "1";
        """)
        self.doc.js_on_event(DocumentReady, js_client_ready)

        js_benchmark = CustomJS(code="""
            if (!cb_obj.value) return;

            const payload = JSON.parse(cb_obj.value);
            const t_recv = performance.now();

            requestAnimationFrame(() => {
                requestAnimationFrame(() => {
                    const t_paint = performance.now();
                    const client_render_ms = t_paint - t_recv;

                    const processing_ms = payload.py_compute_ms;
                    const render_ms = payload.py_apply_ms + client_render_ms;

                    console.log("=========================================");
                    console.log("[BENCHMARK] PALLAS Two-Way Split");
                    console.log(`  - Data Processing : ${processing_ms.toFixed(2)} ms`);
                    console.log(`  - Render Time     : ${render_ms.toFixed(2)} ms`);
                    console.log(`  - Total           : ${(processing_ms + render_ms).toFixed(2)} ms`);
                    console.log("=========================================");
                });
            });
        """)

        self._benchmark_payload_input.js_on_change("value", js_benchmark)

        self.fig = fig
        return column(
            fig,
            self._client_ready_input,
            self._benchmark_payload_input,
        )

    def update(
        self,
        *,
        active_thread_names: list[str],
        n_quanta: int,
        mode: FidelityMode,
        token_mode: TokenMode = "raw",
        stack_order: str,
        window_t0_ns: int | None = None,
        window_t1_ns: int | None = None,
    ) -> dict:
        with timed("setup metadata"):
            common_names = self.get_shared_thread_names(active_thread_names)

            start_ns = min(self.t1.meta.start_ns, self.t2.meta.start_ns)
            end_ns = max(self.t1.meta.end_ns, self.t2.meta.end_ns)

            if window_t0_ns is None or window_t1_ns is None:
                start_ns = self.full_start_ns
                end_ns = self.full_end_ns
                sync_range_to_fig = True
            else:
                start_ns = window_t0_ns
                end_ns = window_t1_ns
                sync_range_to_fig = False

            if end_ns <= start_ns:
                end_ns = start_ns + 1

            edges = tuple(
                int(x) for x in np.linspace(start_ns, end_ns, int(n_quanta) + 1, dtype=np.int64)
            )

            all_token_keys = (
                set(self.t1.meta.token_key_to_name.keys())
                | set(self.t2.meta.token_key_to_name.keys())
            )
            color_map = build_color_map(all_token_keys)

            request_key = (
                tuple(common_names),
                int(n_quanta),
                mode,
                token_mode,
                stack_order,
                int(start_ns),
                int(end_ns),
            )
            if request_key == self._request_key:
                return {
                    "generation": self._generation,
                    "threads": common_names,
                    "n_bins": int(n_quanta),
                    "mode": mode,
                    "token_mode": token_mode,
                    "stack_order": stack_order,
                    "color_map": color_map,
                }

            self._generation += 1
            generation = self._generation
            self._bench_generation = generation
            self._bench_py_compute_ms = 0.0
            self._bench_py_apply_ms = 0.0
            self._bench_py_total_ms = 0.0
            self._bench_batch_start_ms = time.perf_counter() * 1000.0

            self._request_key = request_key

            self._current_threads = common_names
            self._current_edges = edges
            self._current_mode = mode
            self._current_token_mode = token_mode
            self._current_stack_order = stack_order
            self._color_map = color_map

        with timed("setup ranges"):
            if self.fig is not None:
                self.fig.y_range.factors = list(reversed(common_names))  # type: ignore
                if sync_range_to_fig and len(edges) >= 2:
                    self.fig.x_range.start = edges[0] / 1e6  # type: ignore
                    self.fig.x_range.end = edges[-1] / 1e6   # type: ignore

        with timed("clear+queue jobs"):
            self.clear_all_sources()

            jobs: list[ThreadLoadJob] = []
            for thread_name in common_names:
                tid1 = self.t1.meta.thread_name_to_id.get(thread_name)
                tid2 = self.t2.meta.thread_name_to_id.get(thread_name)
                jobs.append(
                    ThreadLoadJob(
                        generation      = generation,
                        thread_name     = thread_name,
                        thread_id_1     = tid1,
                        thread_id_2     = tid2,
                        bin_edges_ns    = edges,
                        mode            = mode,
                        token_mode      = token_mode,
                        stack_order     = stack_order,
                    )
                )

            self._pending_jobs.clear()
            self._pending_jobs.extend(jobs)

        print("update gen", generation, "jobs", len(jobs))

        if self._client_ready:
            self.schedule_next_tick()

        return {
            "generation": generation,
            "threads": common_names,
            "n_bins": int(n_quanta),
            "mode": mode,
            "token_mode": token_mode,
            "stack_order": stack_order,
            "color_map": color_map,
        }

    def schedule_next_tick(self) -> None:
        if self.doc is None:
            self.doc = curdoc()
        if self._loader_scheduled:
            return
        self._loader_scheduled = True
        self.doc.add_next_tick_callback(self.drain_one_job)

    def drain_one_job(self) -> None:
        self._loader_scheduled = False
        print("drain gen", self._generation, "pending", len(self._pending_jobs))

        while self._pending_jobs:
            job = self._pending_jobs.popleft()
            if job.generation != self._generation:
                continue

            t0 = time.perf_counter() * 1000.0
            result = self.compute_thread_chunk(job)
            t1 = time.perf_counter() * 1000.0
            self._bench_py_compute_ms += (t1 - t0)

            t2 = time.perf_counter() * 1000.0
            self.apply_thread_chunk(result)
            t3 = time.perf_counter() * 1000.0
            self._bench_py_apply_ms += (t3 - t2)
            break

        if self._pending_jobs:
            self.schedule_next_tick()
        else:
            self._bench_py_total_ms = (time.perf_counter() * 1000.0) - self._bench_batch_start_ms
            payload = {
                "generation": self._generation,
                "py_compute_ms": self._bench_py_compute_ms,
                "py_apply_ms": self._bench_py_apply_ms,
                "py_total_ms": self._bench_py_total_ms,
            }
            self._benchmark_payload_input.value = json.dumps(payload)

    def compute_thread_chunk(self, job: ThreadLoadJob) -> ThreadLoadResult:
        if job.thread_id_1 is not None:
            q1 = self.t1.query_quanta(
                QuantaQuery(
                    thread_ids      = (job.thread_id_1,),
                    bin_edges_ns    = job.bin_edges_ns,
                    fidelity        = job.mode,  # type: ignore
                    token_mode      = job.token_mode,
                    top_k           = -1,
                )
            )
            src1 = quanta_bundle_to_bokeh_source(
                q1,
                meta            = self.t1.meta,
                active_threads  = self._current_threads,
                trace_side      = "lower",
                color_map       = self._color_map,
                stack_order     = job.stack_order,
            )
        else:
            src1 = empty_quanta_source()

        if job.thread_id_2 is not None:
            q2 = self.t2.query_quanta(
                QuantaQuery(
                    thread_ids      = (job.thread_id_2,),
                    bin_edges_ns    = job.bin_edges_ns,
                    fidelity        = job.mode,  # type: ignore
                    token_mode      = job.token_mode,
                    top_k           = -1,
                )
            )
            src2 = quanta_bundle_to_bokeh_source(
                q2,
                meta            = self.t2.meta,
                active_threads  = self._current_threads,
                trace_side      = "upper",
                color_map       = self._color_map,
                stack_order     = job.stack_order,
            )
        else:
            src2 = empty_quanta_source()

        return ThreadLoadResult(
            generation  = job.generation,
            thread_name = job.thread_name,
            src1        = src1,
            src2        = src2,
        )

    def apply_thread_chunk(self, result: ThreadLoadResult) -> None:
        print("apply gen", result.generation, result.thread_name,
            len(result.src1["left"]), len(result.src2["left"]))
        if result.generation != self._generation:
            return

        thread_name = result.thread_name
        self.sources1[thread_name].data = result.src1
        self.sources2[thread_name].data = result.src2

        self.renderers1[thread_name].visible = True  # type: ignore
        self.renderers2[thread_name].visible = True  # type: ignore

    def clear_all_sources(self) -> None:
        active = set(self._current_threads)
        for thread_name in self.get_all_thread_names():
            self.sources1[thread_name].data = empty_quanta_source()
            self.sources2[thread_name].data = empty_quanta_source()
            self.renderers1[thread_name].visible = thread_name in active  # type: ignore
            self.renderers2[thread_name].visible = thread_name in active  # type: ignore

    def bundle_token_keys(self, bundle) -> set[str]:
        return {
            as_token_key(int(tt), int(tid))
            for tt, tid in zip(bundle.token_type, bundle.token_id)
        }

    def get_all_thread_names(self) -> list[str]:
        names = set(map(str, self.t1.meta.thread_names)) | set(map(str, self.t2.meta.thread_names))
        return sorted(names)

    def get_shared_thread_names(self, requested: list[str]) -> list[str]:
        available = set(self.get_all_thread_names())
        chosen = [name for name in requested if name in available]
        return chosen or self.get_all_thread_names()

    def _on_source_selected(self, source: ColumnDataSource, indices) -> None:
        if self._ignore_selection_callbacks:
            return
        if not indices:
            return

        i = int(indices[0])
        data = source.data

        token_types = data.get("token_type")
        token_ids = data.get("token_id")
        if token_types is None or token_ids is None:
            return
        if i < 0 or i >= len(token_types) or i >= len(token_ids):
            return

        token = (int(token_types[i]), int(token_ids[i]))
        if self.on_token_selected is not None:
            self.on_token_selected(token)
