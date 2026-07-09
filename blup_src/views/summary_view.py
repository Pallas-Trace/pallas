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

    def build(self):
        self.header = Div(
            text="<b>Sequence summary</b><br>No sequence selected",
            width=self.width,
        )
        self.body = Div(text="", width=self.width)

        self.hist_fig = figure(width=self.width, height=600, title="Exclusive total by time bin")
        self.hist_fig.quad(
            left="left", right="right", bottom=0, top="trace1",
            source=self.hist_source, fill_alpha=0.35, line_alpha=0.0, color="navy"
        )
        self.hist_fig.quad(
            left="left", right="right", bottom=0, top="trace2",
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
                    model.metric, model.trace1, model.trace2, model.delta, model.percent
                )
            )
            self.body.text = f"""
            <table style="width:100%; border-collapse:collapse;">
                <thead>
                    <tr>
                        <th align="left">Metric</th>
                        <th align="left">Trace 1</th>
                        <th align="left">Trace 2</th>
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
            trace1=[x / 1e6 for x in model.hist_trace1_excl_ns],
            trace2=[x / 1e6 for x in model.hist_trace2_excl_ns],
        )

