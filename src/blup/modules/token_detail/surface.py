from __future__ import annotations

from typing import Callable

from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models.layouts import LayoutDOM
from bokeh.models.ranges import Range1d
from bokeh.models.widgets.inputs import Select
from bokeh.models.widgets.markups import Div
from bokeh.plotting import ColumnDataSource, figure

from blup.bokeh.theme import PALETTE
from blup.modules.token_detail.types import (
    TokenDetailTableModel,
    select_value_to_token,
)
from blup.state import TokenDetailChartMode, TraceMode


def _empty_histogram_source() -> dict:
    return {
        "left": [],
        "right": [],
        "upper": [],
        "lower": [],
        "delta": [],
    }


class TokenDetailSurface:

    def __init__(self, *, height: int = 700) -> None:
        self.height = height
        self.doc = None
        self.root: LayoutDOM | None = None

        self.token_select: Select | None = None
        self.header: Div | None = None
        self.body: Div | None = None
        self.hist_fig = None
        self.hist_source = ColumnDataSource(data=_empty_histogram_source())
        self.upper_renderer = None
        self.lower_renderer = None
        self.delta_renderer = None

        self.on_token_selected: (
            Callable[[tuple[int, int] | None], None] | None
        ) = None

        self._ignore_widget_callbacks = False

    def build(self) -> LayoutDOM:
        self.doc = curdoc()
        p = PALETTE

        self.token_select = Select(
            title="Highlight token",
            value="",
            options=[("", "(none)")],
            sizing_mode="stretch_width",
        )
        self.token_select.on_change("value", self._on_select_changed)

        self.header = Div(
            text="<b>Token detail</b><br>No token selected",
            sizing_mode="stretch_width",
        )
        self.body = Div(text="", sizing_mode="stretch_width")

        fig = figure(
            title               = "Exclusive total by time bin",
            min_width           = 320,
            min_height          = 240,
            toolbar_location    = None,
            # context_menu        = None,
            output_backend      = "webgl",
            sizing_mode         = "stretch_both",
            x_range             = Range1d(0, 1),
            active_drag="xbox_zoom",
            tools               = [
                "box_zoom", "xwheel_pan", "xbox_zoom",
                "reset", "save"
            ],
            # x_axis_label    = "Time (ms)",
            # y_axis_label    = "Exclusive (ms)",
        )

        self.upper_renderer = fig.quad(
            left="left",
            right="right",
            bottom=0,
            top="upper",
            source=self.hist_source,
            fill_alpha=0.35,
            line_alpha=0.0,
            color=p.blue,
            name="token_detail_upper",
        )
        self.lower_renderer = fig.quad(
            left="left",
            right="right",
            bottom=0,
            top="lower",
            source=self.hist_source,
            fill_alpha=0.35,
            line_alpha=0.0,
            color=p.orange,
            name="token_detail_lower",
        )
        self.delta_renderer = fig.quad(
            left="left",
            right="right",
            bottom=0,
            top="delta",
            source=self.hist_source,
            fill_alpha=0.35,
            line_alpha=0.0,
            color=p.purple,
            visible=False,
            name="token_detail_delta",
        )

        self.hist_fig = fig
        self.root = column(
            self.token_select,
            self.header,
            self.body,
            fig,
            sizing_mode="stretch_both",
        )
        return self.root

    def prepare_display(
        self,
        *,
        start_ns: int,
        end_ns: int,
        trace_mode: TraceMode,
        chart_mode: TokenDetailChartMode,
        show_stats: bool,
        show_chart: bool,
    ) -> None:
        fig = self.hist_fig
        if fig is None:
            raise RuntimeError(
                "TokenDetailSurface.build must be called before prepare_display"
            )
        if chart_mode not in ("overlay", "delta"):
            raise ValueError(f"invalid chart_mode: {chart_mode!r}")

        dual = trace_mode == "dual"

        fig.x_range.start = start_ns / 1e6                                  # type: ignore[attr-defined]
        fig.x_range.end = end_ns / 1e6                                      # type: ignore[attr-defined]

        fig.visible = show_chart

        if self.body:
            self.body.visible = show_stats

        if self.upper_renderer:
            self.upper_renderer.visible = show_chart and chart_mode == "overlay"

        if self.lower_renderer:
            self.lower_renderer.visible = (
                show_chart and chart_mode == "overlay" and dual
            )

        if self.delta_renderer:
            self.delta_renderer.visible = (
                show_chart and chart_mode == "delta" and dual
            )

        self.hist_source.data = _empty_histogram_source()

    def apply_table_result(self, model: TokenDetailTableModel) -> None:
        if self.header is not None:
            self.header.text = f"<b>{model.title}</b><br>{model.subtitle}"
        if self.body is not None:
            self.body.text = self._render_stats_html(model)
        self._sync_token_select(model)

    def apply_histogram_result(self, src: dict) -> None:
        self.hist_source.data = src

    def _sync_token_select(self, model: TokenDetailTableModel) -> None:
        widget = self.token_select
        if widget is None:
            return

        options = [("", "(none)"), *model.options]
        valid_values = {value for value, _ in options}
        value = (
            model.selected_value if model.selected_value in valid_values else ""
        )

        self._ignore_widget_callbacks = True
        try:
            widget.options = options                                        # type: ignore[attr-defined]
            if widget.value != value:
                widget.value = value
        finally:
            self._ignore_widget_callbacks = False

    def _on_select_changed(self, attr: str, old: str, new: str) -> None:
        if self._ignore_widget_callbacks:
            return
        if self.on_token_selected is not None:
            self.on_token_selected(select_value_to_token(new))

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

