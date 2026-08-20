from __future__ import annotations

from typing import Callable

from blup.types import TimestampNS, TraceMode, TraceSide
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models.layouts import LayoutDOM
from bokeh.models.ranges import Range1d
from bokeh.models.tools import HoverTool
from bokeh.models.widgets.inputs import Select
from bokeh.models.widgets.markups import Div
from bokeh.plotting import ColumnDataSource, figure

from blup.bokeh.theme import PALETTE
from blup.modules.token_detail.types import (
    TokenDetailHistogramResult,
    TokenDetailScatterResult,
    TokenDetailTableModel,
    select_value_to_token,
)
from blup.state import TokenDetailChartMode


def _empty_histogram_source() -> dict:
    return {"left": [], "right": [], "top": []}

def _empty_scatter_source() -> dict:
    return {"x": [], "y": []}


class TokenDetailSurface:

    def __init__(self, *, height: int = 700) -> None:
        self.height = height
        self.doc = None
        self.root: LayoutDOM | None = None

        self.header: Div | None = None
        self.body: Div | None = None
        self.chart_fig = None

        self.hist_sources: dict[TraceSide, ColumnDataSource] = {}
        self.hist_renderers: dict[TraceSide, object] = {}
        self.scatter_sources: dict[TraceSide, ColumnDataSource] = {}
        self.scatter_renderers: dict[TraceSide, object] = {}

    def build(self) -> LayoutDOM:
        self.doc = curdoc()
        p = PALETTE

        self.header = Div(
            text="<b>Token detail</b><br>No token selected",
            sizing_mode="stretch_width",
        )
        self.body = Div(text="", sizing_mode="stretch_width")

        fig = figure(
            title               = "Exclusive total by time bin",
            min_width           = 320,
            min_height          = 240,
            sizing_mode         = "stretch_both",
            toolbar_location    = None,
            # context_menu        = None,
            output_backend      = "webgl",
            x_range             = Range1d(0, 1),
            active_drag="xbox_zoom",
            tools               = [
                "box_zoom", "xwheel_pan", "xbox_zoom",
                "reset", "save"
            ],
            # x_axis_label    = "Time (ms)",
            # y_axis_label    = "Exclusive (ms)",
        )

        side_colors: dict[TraceSide, str] = {
            "upper": p.blue,
            "lower": p.orange,
        }

        for side in ("upper", "lower"):
            hist_src = ColumnDataSource(data=_empty_histogram_source())
            self.hist_sources[side] = hist_src
            self.hist_renderers[side] = fig.quad(
                left="left",
                right="right",
                bottom=0,
                top="top",
                source=hist_src,
                fill_alpha=0.35,
                line_alpha=0.0,
                color=side_colors[side],
                visible=False,
                name=f"token_detail_hist_{side}",
            )

            scatter_src = ColumnDataSource(data=_empty_scatter_source())
            self.scatter_sources[side] = scatter_src
            self.scatter_renderers[side] = fig.scatter(
                x="x",
                y="y",
                source=scatter_src,
                size=4,
                alpha=0.6,
                color=side_colors[side],
                marker="circle",
                visible=False,
                name=f"token_detail_scatter_{side}",
            )

        hover = HoverTool(
            renderers=list(self.scatter_renderers.values()),                # type: ignore[attr-defined]
            tooltips=(
                f'<div style="font-family:monospace; font-size:11px;'
                f' color:{p.fg1}; background:{p.bg1}; padding:4px 8px;">'
                "<div>start: @x{0.000} ms</div>"
                "<div>duration: @y{0.000} ms</div>"
                "</div>"
            ),
            point_policy="snap_to_data",
        )
        fig.add_tools(hover)

        self.chart_fig = fig
        self.root = column(
            self.header,
            self.body,
            fig,
            sizing_mode="stretch_both",
        )
        return self.root

    def prepare_display(
        self,
        *,
        start_ns: TimestampNS,
        end_ns: TimestampNS,
        trace_mode: TraceMode,
        chart_mode: TokenDetailChartMode,
        show_stats: bool,
        show_chart: bool,
    ) -> None:
        fig = self.chart_fig
        if fig is None:
            raise RuntimeError(
                "TokenDetailSurface.build must be called before prepare_display"
            )
        if chart_mode not in ("histogram", "scatter"):
            raise ValueError(f"invalid chart_mode: {chart_mode!r}")

        dual = trace_mode == "dual"

        fig.x_range.start = start_ns / 1e6                                  # type: ignore[attr-defined]
        fig.x_range.end = end_ns / 1e6                                      # type: ignore[attr-defined]

        fig.visible = show_chart
        if self.body:
            self.body.visible = show_stats

        if chart_mode == "histogram":
            if fig.title:
                fig.title.text = "Exclusive total by time bin"              # type: ignore[attr-defined]
            fig.yaxis.axis_label = "Exclusive (ms)"
        else:
            if fig.title:
                fig.title.text = "Call durations over time"                 # type: ignore[attr-defined]
            fig.yaxis.axis_label = "Duration (ms)"

        for side in ("upper", "lower"):
            self.hist_sources[side].data = _empty_histogram_source()
            self.scatter_sources[side].data = _empty_scatter_source()

            side_active = side == "upper" or dual
            self.hist_renderers[side].visible = (                           # type: ignore[attr-defined]
                show_chart and chart_mode == "histogram" and side_active
            )
            self.scatter_renderers[side].visible = (                        # type: ignore[attr-defined]
                show_chart and chart_mode == "scatter" and side_active
            )

    def apply_table_result(self, model: TokenDetailTableModel) -> None:
        if self.header is not None:
            self.header.text = f"<b>{model.title}</b><br>{model.subtitle}"
        if self.body is not None:
            self.body.text = self._render_stats_html(model)

    def apply_histogram_result(self, result: TokenDetailHistogramResult) -> None:
        self.hist_sources[result.trace_side].data = result.src

    def apply_scatter_result(self, result: TokenDetailScatterResult) -> None:
        self.scatter_sources[result.trace_side].data = result.src

    def _render_stats_html(self, model: TokenDetailTableModel) -> str:
        if not model.metric:
            return ""

        if model.dual_mode:
            rows = "".join(
                f"<tr><td>{m}</td><td>{a}</td><td>{b}</td>"
                f"<td>{d}</td><td>{pc}</td></tr>"
                for m, a, b, d, pc in zip(
                    model.metric,
                    model.upper,
                    model.lower,
                    model.delta,
                    model.percent,
                )
            )
            return (
                '<table style="width:100%; border-collapse:collapse;">'
                "<thead><tr>"
                '<th align="left">Metric</th>'
                f'<th align="left">{model.upper_label}</th>'
                f'<th align="left">{model.lower_label}</th>'
                '<th align="left">Delta</th>'
                '<th align="left">% diff</th>'
                "</tr></thead>"
                f"<tbody>{rows}</tbody></table>"
            )

        rows = "".join(
            f"<tr><td>{m}</td><td>{a}</td><td>{d}</td><td>{pc}</td></tr>"
            for m, a, d, pc in zip(
                model.metric,
                model.upper,
                model.delta,
                model.percent,
            )
        )
        return (
            '<table style="width:100%; border-collapse:collapse;">'
            "<thead><tr>"
            '<th align="left">Metric</th>'
            f'<th align="left">{model.upper_label}</th>'
            '<th align="left">Delta</th>'
            '<th align="left">% diff</th>'
            "</tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )


