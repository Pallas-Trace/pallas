from __future__ import annotations

from typing import Callable

from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models.tools import TapTool
from bokeh.models.sources import ColumnDataSource
from bokeh.models.tools import HoverTool
from bokeh.models.ranges import Range1d
from bokeh.plotting import figure

from trace_session import TraceSession
from adapters.quanta_adapter import empty_quanta_source


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

        self.doc = None
        self.fig = None
        self.root = None

        self.sources1: dict[str, ColumnDataSource] = {}
        self.sources2: dict[str, ColumnDataSource] = {}
        self.renderers1: dict[str, object] = {}
        self.renderers2: dict[str, object] = {}

        self.full_start_ns = min(self.t1.meta.start_ns, self.t2.meta.start_ns)
        self.full_end_ns = max(self.t1.meta.end_ns, self.t2.meta.end_ns)

        self.on_token_selected: Callable[[tuple[int, int] | None], None] | None = None
        self._ignore_selection_callbacks = False

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
                lambda attr, old, new, source=s1: self.on_source_selected(source, new),
            )
            s2.selected.on_change(
                "indices",
                lambda attr, old, new, source=s2: self.on_source_selected(source, new),
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

        self.fig = fig
        self.root = column(fig)
        return self.root

    def prepare_display(
        self,
        *,
        active_thread_names: list[str],
        start_ns: int,
        end_ns: int,
        sync_range_to_fig: bool,
    ) -> None:
        if self.fig is not None:
            self.fig.y_range.factors = list(reversed(active_thread_names))  # type: ignore[attr-defined]
            if sync_range_to_fig:
                self.fig.x_range.start = start_ns / 1e6  # type: ignore[attr-defined]
                self.fig.x_range.end = end_ns / 1e6      # type: ignore[attr-defined]

        self.clear_all_sources(active_thread_names)

    def clear_all_sources(self, active_thread_names: list[str]) -> None:
        active = set(active_thread_names)
        for thread_name in self.get_all_thread_names():
            self.sources1[thread_name].data = empty_quanta_source()
            self.sources2[thread_name].data = empty_quanta_source()
            self.renderers1[thread_name].visible = thread_name in active  # type: ignore[attr-defined]
            self.renderers2[thread_name].visible = thread_name in active  # type: ignore[attr-defined]

    def apply_thread_result(self, *, thread_name: str, src1: dict, src2: dict) -> None:
        self.sources1[thread_name].data = src1
        self.sources2[thread_name].data = src2
        self.renderers1[thread_name].visible = True  # type: ignore[attr-defined]
        self.renderers2[thread_name].visible = True  # type: ignore[attr-defined]

    def get_all_thread_names(self) -> list[str]:
        names = set(map(str, self.t1.meta.thread_names)) | set(map(str, self.t2.meta.thread_names))
        return sorted(names)

    def on_source_selected(self, source: ColumnDataSource, indices) -> None:
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
