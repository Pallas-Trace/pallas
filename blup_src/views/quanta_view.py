from __future__ import annotations

from typing import Callable, Any

from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models.tools import TapTool
from bokeh.models.sources import ColumnDataSource
from bokeh.models.tools import HoverTool
from bokeh.models.ranges import Range1d
from bokeh.plotting import figure

from state import TraceMode
from trace_session import TraceSession
from adapters.quanta_adapter import empty_quanta_source


class QuantaView:

    def __init__(
        self,
        *,
        width: int = 1400,
        height: int = 900,
    ) -> None:
        self.width = width
        self.height = height

        self.doc = None
        self.fig = None
        self.root = None

        self.sources1: dict[str, ColumnDataSource] = {}
        self.sources2: dict[str, ColumnDataSource] = {}
        self.renderers1: dict[str, object] = {}
        self.renderers2: dict[str, object] = {}

        self.on_token_selected: Callable[[tuple[int, int] | None], None] | None = None
        self._ignore_selection_callbacks = False

    def build(self, doc=None):
        self.doc = doc or curdoc()

        fig = figure(
            width           = self.width,
            height          = self.height,
            x_range         = Range1d(0, 1),
            y_range         = [],  # type: ignore
            tools           = ["tap", "box_zoom", "xwheel_pan", "xbox_zoom", "reset", "undo", "redo", "save"],
            active_drag     = "xbox_zoom",
            output_backend  = "webgl",
            title           = "Quanta comparison",
            x_axis_label    = "Time (ms)",
        )
        fig.toolbar.active_tap = fig.select_one(TapTool)  # type: ignore

        hover = HoverTool(
            renderers=[],
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
        self.hover = hover
        self.root = column(fig)
        return self.root

    def ensure_threads(self, thread_names: list[str]) -> None:
        fig = self.fig
        if fig is None:
            raise RuntimeError("QuantaView.build() must be called before ensure_threads()")

        for thread_name in thread_names:
            if thread_name in self.sources1:
                continue

            s1 = ColumnDataSource(empty_quanta_source())
            s2 = ColumnDataSource(empty_quanta_source())

            s1.selected.on_change(
                "indices",
                lambda attr, old, new, source=s1: self.on_source_selected(source, new)
            )
            s2.selected.on_change(
                "indices",
                lambda attr, old, new, source=s2: self.on_source_selected(source, new)
            )

            self.sources1[thread_name] = s1
            self.sources2[thread_name] = s2

            r1 = fig.quad(
                left="left", right="right", top="top", bottom="bottom",
                color="color", line_color=None, fill_alpha=0.90,
                source=s1, visible=False, name=f"quanta_primary_{thread_name}")
            r2 = fig.quad(
                left="left", right="right", top="top", bottom="bottom",
                color="color", line_color=None, fill_alpha=0.90,
                source=s2, visible=False, name=f"quanta_secondary_{thread_name}")

            r1.nonselection_glyph = r1.glyph
            r2.nonselection_glyph = r2.glyph

            self.renderers1[thread_name] = r1
            self.renderers2[thread_name] = r2

            self.register_hover_renderers(r1, r2)

    def register_hover_renderers(self, *renderers: Any) -> None:
        if self.hover is None:
            return

        current = list(self.hover.renderers)
        current.extend(renderers)
        self.hover.renderers = current  # type: ignore

    def prepare_display(
        self,
        *,
        active_thread_names: list[str],
        start_ns: int,
        end_ns: int,
        sync_range_to_fig: bool,
        trace_mode: TraceMode
    ) -> None:
        fig = self.fig
        if fig is None:
            raise RuntimeError("QuantaView.build() must be called before prepare_display()")

        self.ensure_threads(active_thread_names)

        fig.y_range.factors = list(reversed(active_thread_names))  # type: ignore[attr-defined]
        if sync_range_to_fig:
            fig.x_range.start = start_ns / 1e6  # type: ignore[attr-defined]
            fig.x_range.end = end_ns / 1e6      # type: ignore[attr-defined]

        self.clear_all_sources(active_thread_names, trace_mode)

    def clear_all_sources(self, active_thread_names: list[str], trace_mode: TraceMode) -> None:
        active = set(active_thread_names)
        known_threads = set(self.sources1) | set(self.sources2)

        for thread_name in known_threads:
            self.sources1[thread_name].data = empty_quanta_source()
            self.sources2[thread_name].data = empty_quanta_source()
            self.renderers1[thread_name].visible = thread_name in active  # type: ignore
            self.renderers2[thread_name].visible = (trace_mode == "dual" and thread_name in active)  # type: ignore

    def apply_thread_result(
        self,
        *,
        thread_name: str,
        src1: dict,
        src2: dict,
        trace_mode: TraceMode
    ) -> None:
        self.ensure_threads([thread_name])

        self.sources1[thread_name].data = src1
        self.sources2[thread_name].data = src2
        self.renderers1[thread_name].visible = True  # type: ignore[attr-defined]
        self.renderers2[thread_name].visible = (trace_mode == "dual")  # type: ignore[attr-defined]

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
