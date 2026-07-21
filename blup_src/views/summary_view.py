from __future__ import annotations

from bokeh.layouts import column
from bokeh.plotting import figure
from bokeh.models.sources import ColumnDataSource
from bokeh.models.widgets.markups import Div
from bokeh.models.widgets.tables import DataTable, TableColumn

from adapters.summary_adapter import SequenceSummaryDisplayModel

class SummaryView:
    def __init__(self, width: int = 360, height: int = 240) -> None:
        self.width = width
        self.height = height
        self.header: Div | None = None
        self.body: Div | None = None
        self.hist_source = ColumnDataSource(data=dict(
            left=[],
            right=[],
            trace1=[],
            trace2=[],
        ))
        self.root = None
        self.hist_fig = None
        self.primary_quad = None
        self.secondary_quad = None

    def build(self):
        self.header = Div(
            text="<b>Sequence summary</b><br>No sequence selected",
            width=self.width,
        )
        self.body = Div(text="", width=self.width)

        self.hist_fig = figure(width=self.width, height=600, title="Exclusive total by time bin")
        self.hist_fig.quad(
            left="left", right="right", bottom=0, top="primary",
            source=self.hist_source, fill_alpha=0.35, line_alpha=0.0, color="navy"
        )
        self.hist_fig.quad(
            left="left", right="right", bottom=0, top="secondary",
            source=self.hist_source, fill_alpha=0.35, line_alpha=0.0, color="firebrick"
        )

        self.root = column(self.header, self.body, self.hist_fig, width=self.width)
        return self.root

    def update(self, model: SequenceSummaryDisplayModel) -> None:
        if self.header is not None:
            self.header.text = f"<b>{model.title}</b><br>{model.subtitle}"

        if self.body is not None:
            rows = "".join(
                f"<tr><td>{m}</td><td>{a}</td><td>{b}</td><td>{d}</td><td>{p}</td></tr>"
                for m, a, b, d, p in zip(
                    model.metric, model.primary, model.secondary, model.delta, model.percent
                )
            )
            if model.dual_mode:
                rows = "".join(
                    f"<tr><td>{m}</td><td>{a}</td><td>{b}</td><td>{d}</td><td>{p}</td></tr>"
                    for m, a, b, d, p in zip(
                        model.metric, model.primary, model.secondary, model.delta, model.percent
                    )
                )
                self.body.text = f"""
                <table style="width:100%; border-collapse:collapse;">
                    <thead>
                        <tr>
                            <th align="left">Metric</th>
                            <th align="left">{model.primary_label}</th>
                            <th align="left">{model.secondary_label}</th>
                            <th align="left">Delta</th>
                            <th align="left">% diff</th>
                        </tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
                """
            else:
                rows = "".join(
                    f"<tr><td>{m}</td><td>{a}</td><td>{d}</td><td>{p}</td></tr>"
                    for m, a, d, p in zip(
                        model.metric, model.primary, model.delta, model.percent
                    )
                )
                self.body.text = f"""
                <table style="width:100%; border-collapse:collapse;">
                    <thead>
                        <tr>
                            <th align="left">Metric</th>
                            <th align="left">{model.primary_label}</th>
                            <th align="left">Delta</th>
                            <th align="left">% diff</th>
                        </tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
                """

        self.hist_source.data = dict(
            left=[x / 1e6 for x in model.hist_left_ns],
            right=[x / 1e6 for x in model.hist_right_ns],
            primary=[x / 1e6 for x in model.hist_primary_excl_ns],
            secondary=[x / 1e6 for x in model.hist_secondary_excl_ns],
        )

        if self.secondary_quad is not None:
            self.secondary_quad.visible = model.dual_mode

