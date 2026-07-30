from __future__ import annotations

from dataclasses import dataclass

from bokeh.layouts import column
from bokeh.models.widgets.markups import Div

from data_model import SummaryQuery, TokenMode, as_token_key
from trace_session import TraceSession
from utils import timed


@dataclass(frozen=True)
class InspectorDisplayModel:
    title: str
    primary_label: str
    secondary_label: str | None
    thread_count: int
    n_quanta: int
    token_view_label: str
    snapshot_mode: str
    stack_order: str
    top_primary: tuple[str, ...]
    top_secondary: tuple[str, ...] = ()


class InspectorView:
    def __init__(self, width: int = 340) -> None:
        self.width = width
        self.div = Div(width=self.width, sizing_mode="fixed")
        self.root = column(self.div, width=self.width)

    def build(self):
        self.div.text = "<b>Inspector</b>"
        return self.root

    def update(self, model: InspectorDisplayModel) -> None:
        self.div.text = f"""
        <div style="padding:8px">
          <h3 style="margin-top:0">Inspector</h3>
          <p><b>Trace 1:</b> {model.primary_label}</p>
          <p><b>Trace 2:</b> {model.secondary_label if model.secondary_label is not None else ""}</p>
          <p><b>Threads:</b> {model.thread_count}</p>
          <p><b>Quanta bins:</b> {model.n_quanta}</p>
          <p><b>Token view:</b> {model.token_view_label}</p>
          <p><b>Snapshot mode:</b> {model.snapshot_mode}</p>
          <p><b>Stack order:</b> {model.stack_order}</p>
          <p><b>Top functions T1:</b> {", ".join(model.top_primary) if model.top_primary else "(none)"}</p>
          <p><b>Top functions T2:</b> {", ".join(model.top_secondary) if model.top_secondary else "(none)"}</p>
        </div>
        """


