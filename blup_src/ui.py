from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from bokeh.layouts import column, row
from bokeh.models.widgets.markups import Div
from bokeh.models.layouts import LayoutDOM
from bokeh.models.widgets.inputs import MultiSelect, Select, Spinner

from state import AppState

if TYPE_CHECKING:
    from controller import AppController


@dataclass
class UIElements:
    root:                       LayoutDOM
    title:                      Div

    thread_select:              MultiSelect
    n_quanta_spinner:           Spinner
    token_mode_select:          Select
    quanta_mode_select:         Select
    stack_order_select:         Select
    highlight_token_select:     Select

    primary_panel:              LayoutDOM
    secondary_panel:            LayoutDOM

class UIModel:

    def __init__(self, controller: "AppController") -> None:
        self.controller = controller

    def build(
        self,
        *,
        state: AppState,
        all_thread_names: list[str],
    ) -> UIElements:

        # Title

        title = Div(
            text    = "<h2 style='margin:0'>Blup</h2>",
            width   = 120,
        )

        # UI Widgets

        thread_select = MultiSelect(
            title   = "Threads",
            value   = list(state.context.active_threads),
            options = all_thread_names,  # type: ignore
            size    = 12,
            width   = 260,
        )
        n_quanta_spinner = Spinner(
            title   = "Quanta bins",
            low     = 1,
            high    = 500,
            step    = 1,
            value   = state.views.quanta.n_bins,
            width   = 130,
        )
        token_mode_select = Select(
            title   = "Token view",
            value   = state.context.token_mode,
            options = ["raw", "named"],
            width   = 120,
        )
        quanta_mode_select = Select(
            title   = "Snapshot mode",
            value   = state.views.quanta.mode,
            options = ["fast", "balanced", "exact"],
            width   = 120,
        )
        stack_order_select = Select(
            title   = "Stack order",
            value   = state.views.quanta.order,
            options = ["global", "local"],
            width   = 120,
        )
        highlight_token_select = Select(
            title   = "Sequence",
            value   = "",
            options = [],
            width   = 320,
        )


        # Document Layout

        primary_panel = column(sizing_mode="stretch_both")
        secondary_panel = column(width=360, sizing_mode="fixed")

        controls = row(
            title,
            thread_select,
            n_quanta_spinner,
            token_mode_select,
            quanta_mode_select,
            stack_order_select,
            highlight_token_select,
        )

        main_view = row(
            primary_panel,
            secondary_panel,
            sizing_mode="stretch_width",
        )

        root = column(
            controls,
            main_view,
            sizing_mode="stretch_width"
        )

        ui = UIElements(
            root = root,
            title = title,
            thread_select = thread_select,
            n_quanta_spinner = n_quanta_spinner,
            token_mode_select = token_mode_select,
            quanta_mode_select = quanta_mode_select,
            stack_order_select = stack_order_select,
            highlight_token_select = highlight_token_select,
            primary_panel = primary_panel,
            secondary_panel = secondary_panel,
        )

        # UI Callbacks

        self._wire_callbacks(ui)

        # return UI model

        return ui

    def _wire_callbacks(self, ui: UIElements) -> None:
        ui.thread_select.on_change("value", self.controller.on_threads_changed)
        ui.n_quanta_spinner.on_change("value", self.controller.on_n_quanta_changed)
        ui.token_mode_select.on_change("value", self.controller.on_token_mode_changed)
        ui.quanta_mode_select.on_change("value", self.controller.on_quanta_mode_changed)
        ui.stack_order_select.on_change("value", self.controller.on_quanta_order_changed)
        ui.highlight_token_select.on_change("value", self.controller.on_highlight_token_changed)


