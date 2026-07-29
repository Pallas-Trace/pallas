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

    trace_mode_select: Select
    primary_trace_select: Select
    secondary_trace_select: Select

    thread_select: MultiSelect
    n_bins_spinner: Spinner
    token_mode_select: Select
    quanta_mode_select: Select
    stack_order_select: Select
    highlight_token_select: Select

    center_panel: LayoutDOM
    left_panel: LayoutDOM
    right_panel: LayoutDOM


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

        trace_options = self.controller.loaded_trace_options()

        trace_mode_select = Select(
            title="Trace mode",
            value=state.context.trace_mode,
            options=[("single", "Single"), ("dual", "Dual")],
            width=120,
        )
        trace_mode_select.on_change("value", self.controller.on_trace_mode_changed)

        primary_trace_select = Select(
            title="Primary trace",
            value=state.context.primary_trace_id or "",
            options=trace_options,                                          # type: ignore
            width=220,
        )
        primary_trace_select.on_change("value", self.controller.on_primary_trace_changed)

        secondary_trace_select = Select(
            title="Secondary trace",
            value=state.context.secondary_trace_id or "",
            options=trace_options,                                          # type: ignore
            width=220,
            disabled=(state.context.trace_mode != "dual"),
        )
        secondary_trace_select.on_change("value", self.controller.on_secondary_trace_changed)

        thread_select = MultiSelect(
            title="Threads",
            value=list(state.context.active_threads),
            options=[(name, name) for name in all_thread_names],
            size=min(max(len(all_thread_names), 8), 24),
            width=260,
        )
        thread_select.on_change("value", self.controller.on_threads_changed)

        n_bins_spinner = Spinner(
            title="# of bins",
            low=4,
            high=2000,
            step=4,
            value=state.modules.time_profile.n_bins,
            width=140,
        )
        n_bins_spinner.on_change("value", self.controller.on_n_quanta_changed)

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

        time_profile_mode_select = Select(
            title="Snapshot mode",
            value=state.modules.time_profile.fidelity,
            options=[
                ("fast", "Fast"),
                ("balanced", "Balanced"),
                ("exact", "Exact"),
            ],
            width=140,
        )
        time_profile_mode_select.on_change("value", self.controller.on_time_profile_mode_changed)

        stack_order_select = Select(
            title="Stack order",
            value=state.modules.time_profile.order,
            options=[
                ("global", "Global"),
                ("local", "Local"),
            ],
            width=140,
        )
        stack_order_select.on_change("value", self.controller.on_time_profile_order_changed)

        highlight_token_select = Select(
            title="Highlight token",
            value="",
            options=[("", "(none)")],
            width=520,
        )
        highlight_token_select.on_change("value", self.controller.on_highlight_token_changed)

        controls = row(
            # title,
            trace_mode_select,
            primary_trace_select,
            secondary_trace_select,
            thread_select,
            n_bins_spinner,
            token_mode_select,
            time_profile_mode_select,
            stack_order_select,
            highlight_token_select,
            sizing_mode="stretch_width",
        )

        center_panel = column(sizing_mode="fixed", width=1350)
        left_panel = column(sizing_mode="fixed", width=400)
        right_panel = column(sizing_mode="fixed", width=400)

        body = row(
            left_panel,
            center_panel,
            right_panel,
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
            trace_mode_select=trace_mode_select,
            primary_trace_select=primary_trace_select,
            secondary_trace_select=secondary_trace_select,
            thread_select=thread_select,
            n_bins_spinner=n_bins_spinner,
            token_mode_select=token_mode_select,
            quanta_mode_select=time_profile_mode_select,
            stack_order_select=stack_order_select,
            highlight_token_select=highlight_token_select,
            center_panel=center_panel,
            left_panel=left_panel,
            right_panel=right_panel,
        )
