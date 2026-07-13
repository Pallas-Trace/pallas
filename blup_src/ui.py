from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from bokeh.layouts import column, row
from bokeh.models.layouts import LayoutDOM
from bokeh.models.widgets.inputs import MultiSelect, Select, Spinner
from bokeh.models.widgets.markups import Div

from state import AppState

if TYPE_CHECKING:
    from controller import AppController


@dataclass
class UIElements:
    root: LayoutDOM
    title: Div

    thread_select: MultiSelect
    n_quanta_spinner: Spinner
    token_mode_select: Select
    quanta_mode_select: Select
    stack_order_select: Select
    highlight_token_select: Select

    primary_panel: LayoutDOM
    secondary_panel: LayoutDOM


class UIModel:
    def __init__(self, controller: "AppController") -> None:
        self.controller = controller

    def build(
        self,
        *,
        state: AppState,
        all_thread_names: list[str],
    ) -> UIElements:
        title = Div(text="## Pallas trace comparison")

        thread_select = MultiSelect(
            title="Threads",
            value=list(state.context.active_threads),
            options=[(name, name) for name in all_thread_names],
            size=min(max(len(all_thread_names), 8), 24),
            width=260,
        )
        thread_select.on_change("value", self.controller.on_threads_changed)

        n_quanta_spinner = Spinner(
            title="Quanta bins",
            low=4,
            high=2000,
            step=4,
            value=state.views.quanta.n_bins,
            width=140,
        )
        n_quanta_spinner.on_change("value", self.controller.on_n_quanta_changed)

        token_mode_select = Select(
            title="Token view",
            value=state.context.token_mode,
            options=[
                ("raw", "Raw"),
                ("named", "Named"),
            ],
            width=140,
        )
        token_mode_select.on_change("value", self.controller.on_token_mode_changed)

        quanta_mode_select = Select(
            title="Snapshot mode",
            value=state.views.quanta.mode,
            options=[
                ("fast", "Fast"),
                ("balanced", "Balanced"),
                ("exact", "Exact"),
            ],
            width=140,
        )
        quanta_mode_select.on_change("value", self.controller.on_quanta_mode_changed)

        stack_order_select = Select(
            title="Stack order",
            value=state.views.quanta.order,
            options=[
                ("global", "Global"),
                ("local", "Local"),
            ],
            width=140,
        )
        stack_order_select.on_change("value", self.controller.on_quanta_order_changed)

        highlight_token_select = Select(
            title="Highlight token",
            value="",
            options=[("", "(none)")],
            width=520,
        )
        highlight_token_select.on_change("value", self.controller.on_highlight_token_changed)

        controls = row(
            title,
            thread_select,
            n_quanta_spinner,
            token_mode_select,
            quanta_mode_select,
            stack_order_select,
            highlight_token_select,
            sizing_mode="stretch_width",
        )

        primary_panel = column(sizing_mode="fixed", width=1350)
        secondary_panel = column(sizing_mode="fixed", width=400)

        body = row(
            primary_panel,
            secondary_panel,
            sizing_mode="fixed",
        )

        root = column(
            controls,
            body,
            sizing_mode="stretch_width",
        )

        return UIElements(
            root=root,
            title=title,
            thread_select=thread_select,
            n_quanta_spinner=n_quanta_spinner,
            token_mode_select=token_mode_select,
            quanta_mode_select=quanta_mode_select,
            stack_order_select=stack_order_select,
            highlight_token_select=highlight_token_select,
            primary_panel=primary_panel,
            secondary_panel=secondary_panel,
        )
