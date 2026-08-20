from __future__ import annotations

from typing import Any, Callable

from blup.bokeh.theme import PALETTE
from blup.types import TraceMode
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models.annotations.geometry import Span
from bokeh.models.callbacks import CustomJS
from bokeh.models.css import InlineStyleSheet
from bokeh.models.formatters import CustomJSTickFormatter
from bokeh.models.layouts import LayoutDOM
from bokeh.models.ranges import Range1d
from bokeh.models.tools import HoverTool, TapTool
from bokeh.plotting import ColumnDataSource, figure

from blup.modules.time_profile.types import TraceSide


class TimeProfileChartSurface:

    def __init__(self, *, height: int = 700) -> None:
        self.height = height

        self.doc = None
        self.fig = None
        self.hover = None
        self.cursor_line: Span | None = None

        self.root: LayoutDOM | None = None

        self.sources1: dict[str, ColumnDataSource] = {}
        self.sources2: dict[str, ColumnDataSource] = {}
        self.renderers1: dict[str, object] = {}
        self.renderers2: dict[str, object] = {}

        self.on_token_selected: (
            Callable[[tuple[int, int] | None], None] | None
        ) = None
        self.on_cursor_position: (
            Callable[[float | None], None] | None
        ) = None

        self._ignore_selection_callbacks = False

    def build(self) -> LayoutDOM:
        self.doc = curdoc()

        p = PALETTE

        fig = figure(
            title               = "Time profile",
            min_width           = 420,
            min_height          = 450,
            toolbar_location    = None,
            # context_menu        = None,
            output_backend      = "webgl",
            sizing_mode         = "stretch_both",
            x_range             = Range1d(0, 1),
            y_range             = [],                                       # type: ignore
            active_drag         = "xbox_zoom",
            tools               = [
                "tap", "box_zoom", "xwheel_pan",
                "xbox_zoom", "reset", "undo", "redo", "save"
            ],
        )

        # record clicks on figure bars
        fig.toolbar.active_tap = fig.select_one(TapTool)                    # type: ignore
        # adaptive x-axis (time) formatting
        fig.xaxis.formatter = CustomJSTickFormatter(
            code = """
                const span = Math.max(...ticks) - Math.min(...ticks);

                let factor, unit, decimals;
                if (span >= 3600000) {        // >= 1 hour visible
                    factor = 3600000; unit = "h"; decimals = 1;
                } else if (span >= 60000) {   // >= 1 minute visible
                    factor = 60000; unit = "min"; decimals = 1;
                } else if (span >= 1000) {    // >= 1 second visible
                    factor = 1000; unit = "s"; decimals = 1;
                } else {
                    factor = 1; unit = "ms"; decimals = 0;
                }

                const val = tick / factor;
                let label = val.toFixed(decimals);
                label = label.replace(/\\.0+$/, "");
                return label + " " + unit;
            """
        )

        # custom hover popup styling
        hover = HoverTool(
            point_policy    = "follow_mouse",
            renderers       = [],
            tooltips        = f"""
                <div style="
                    position: relative;
                    font-family: monospace;
                    font-size: 11px;
                    color: {p.fg1};
                    background: {p.bg1};
                    padding: 6px 9px;
                    min-width: 160px;
                ">
                    <!-- color chip -->
                    <div style="
                        position: absolute;
                        top: 6px;
                        right: 6px;
                        width: 16px;
                        height: 16px;
                        background: @color;
                        border: 1px solid {p.bg3};
                    "></div>
                    <div style="
                        color: {p.muted};
                        font-size: 10px;
                        letter-spacing: 0.05em;
                        text-transform: uppercase;
                        padding-right: 16px;
                    ">
                        @thread
                    </div>
                    <div style="
                        color: {p.fg0};
                        font-weight: 700;
                        margin-top: 2px;
                        padding-right: 16px;
                    ">
                        @token_name
                    </div>
                    <div style="
                        margin-top: 6px;
                        padding-top: 5px;
                        border-top: 1px solid {p.bg3};
                        display: flex;
                        justify-content: space-between;
                    ">
                        <span style="color: {p.muted};">share</span>
                        <span style="color: {p.fg0};">@pct_display</span>
                    </div>
                    <div style="
                        display: flex;
                        justify-content: space-between;
                        margin-top: 2px;
                    ">
                        <span style="color: {p.muted};">active time</span>
                        <span style="color: {p.fg0};">@time_display</span>
                    </div>
                </div>
                """,
        )
        fig.add_tools(hover)

        # draw vertical dashed line following cursor
        self.cursor_line = Span(
            location    = 0,
            dimension   = "height",
            line_color  = PALETTE.fg2,
            line_width  = 3,
            line_dash   = "dashed",
            line_alpha  = 0.5,
        )
        fig.add_layout(self.cursor_line)
        fig.js_on_event(
            "mousemove",
            CustomJS(
                args = {"span": self.cursor_line, "fig": fig},
                code = """
                    const x = cb_obj.x;

                    if (typeof x !== "number" || !isFinite(x)) {
                        span.visible = false;
                        return;
                    }

                    const range = fig.x_range;
                    if (x < range.start || x > range.end) {
                        span.visible = false;
                        return;
                    }

                    span.location = x;
                    span.visible = true;
                """
            ),
        )
        fig.js_on_event(
            "mouseleave",
            CustomJS(
                args = {"span": self.cursor_line},
                code = "span.visible = false;"
            ),
        )

        self.fig = fig
        self.hover = hover
        self.root = column(
            fig,
            sizing_mode = "stretch_both",
        )
        return self.root

    def _ensure_threads(self, thread_names: list[str]) -> None:
        fig = self.fig
        if fig is None:
            raise RuntimeError("TimeProfileChartSurface.build() must be called before ensure_threads()")

        for thread_name in thread_names:
            if thread_name in self.sources1:
                continue

            s1 = ColumnDataSource(_empty_quanta_source())
            s2 = ColumnDataSource(_empty_quanta_source())

            s1.selected.on_change(
                "indices",
                (lambda attr, old, new, source
                    = s1: self.on_source_selected(source, new)),
            )
            s2.selected.on_change(
                "indices",
                (lambda attr, old, new, source
                    = s2: self.on_source_selected(source, new)),
            )

            self.sources1[thread_name] = s1
            self.sources2[thread_name] = s2

            r1 = fig.quad(
                source      = s1,
                name        = f"time_profile_primary_{thread_name}",
                line_color  = None,
                fill_alpha  = 0.90,
                visible     = False,
                left        = "left",
                right       = "right",
                top         = "top",
                bottom      = "bottom",
                color       = "color",
            )
            r2 = fig.quad(
                source      = s2,
                name        = f"time_profile_secondary_{thread_name}",
                line_color  = None,
                fill_alpha  = 0.90,
                visible     = False,
                left        = "left",
                right       = "right",
                top         = "top",
                bottom      = "bottom",
                color       = "color",
            )

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
        trace_mode: TraceMode,
    ) -> None:
        fig = self.fig
        if fig is None:
            raise RuntimeError(
                "TimeProfileChartSurface.build() "
                    + "must be called before prepare_display()"
            )

        self._ensure_threads(active_thread_names)

        fig.y_range.factors = list(reversed(active_thread_names))           # type: ignore[attr-defined]
        if sync_range_to_fig:
            fig.x_range.start = start_ns / 1e6                              # type: ignore[attr-defined]
            fig.x_range.end = end_ns / 1e6                                  # type: ignore[attr-defined]

        self.clear_all_sources(active_thread_names, trace_mode)

    def clear_all_sources(
        self,
        active_thread_names: list[str],
        trace_mode: TraceMode
    ) -> None:
        active = set(active_thread_names)
        known_threads = set(self.sources1) | set(self.sources2)

        for thread_name in known_threads:
            self.sources1[thread_name].data = _empty_quanta_source()
            self.sources2[thread_name].data = _empty_quanta_source()
            self.renderers1[thread_name].visible = thread_name in active    # type: ignore
            self.renderers2[thread_name].visible = (trace_mode == "dual"    # type: ignore
                                                    and thread_name in active)

    def apply_job_result(
        self,
        *,
        thread_name: str,
        trace_side: TraceSide,
        src: dict,
        trace_mode: TraceMode,
    ) -> None:
        self._ensure_threads([thread_name])

        if trace_side == "upper":
            self.sources1[thread_name].data = src
            self.renderers1[thread_name].visible = True                     # type: ignore[attr-defined]
        else:
            self.sources2[thread_name].data = src
            self.renderers2[thread_name].visible = (trace_mode == "dual")   # type: ignore[attr-defined]

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

def _empty_quanta_source() -> dict:
    return dict(
        left            = [],
        right           = [],
        top             = [],
        bottom          = [],
        color           = [],
        token_key       = [],
        token_name      = [],
        token_type      = [],
        token_id        = [],
        thread          = [],
        proportion      = [],
        exclusive_s     = [],
    )
