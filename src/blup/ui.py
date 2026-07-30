from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from bokeh.layouts import column, row
from bokeh.models.layouts import LayoutDOM
from bokeh.models.widgets.inputs import MultiSelect, Select, Spinner
from bokeh.models.widgets.markups import Div

from blup.state import AppState, TraceMode

if TYPE_CHECKING:
    from blup.controller import AppController


@dataclass
class UIElements:
    root: LayoutDOM
    title: Div

    trace_select: MultiSelect

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
        all_thread_names: tuple[str, ...],
    ) -> UIElements:
        title = Div(text="## Pallas trace comparison")

        trace_options = list(
            self.controller.trace_registry.trace_options()
        )
        trace_ids = state.context.traces.trace_ids

        primary_trace_id = trace_ids[0] if trace_ids else ""
        secondary_trace_id = trace_ids[1] if len(trace_ids) >= 2 else None
        trace_mode: TraceMode = (
                "dual" if secondary_trace_id is not None else "single"
        )

        trace_select = MultiSelect(
            title   = "Traces",
            value   = list(trace_ids),
            options = list(trace_options,),
            size    = min(
                        max(
                            len(self.controller.trace_registry.all_trace_ids()),
                            2,
                        ),
                        8,
                      ),
            width   = 260,
        )
        trace_select.on_change("value", self.controller.on_traces_changed)

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
            trace_select,
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
            root                    = root,
            title                   = title,
            trace_select            = trace_select,
            thread_select           = thread_select,
            n_bins_spinner          = n_bins_spinner,
            token_mode_select       = token_mode_select,
            quanta_mode_select      = time_profile_mode_select,
            stack_order_select      = stack_order_select,
            highlight_token_select  = highlight_token_select,
            center_panel            = center_panel,
            left_panel              = left_panel,
            right_panel             = right_panel,
        )
