from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from bokeh.layouts import column
from bokeh.models.layouts import LayoutDOM
from bokeh.models.widgets.inputs import MultiSelect, Select, Spinner
from bokeh.models.widgets.markups import Div

from blup.bokeh.app_shell import BokehAppShell
from blup.bokeh.styles import panel_title
from blup.state import AppState

if TYPE_CHECKING:
    from blup.controller import AppController


@dataclass
class UIElements:
    root: LayoutDOM
    title: Div
    status: Div

    trace_select: MultiSelect
    thread_select: MultiSelect
    token_mode_select: Select

    n_bins_spinner: Spinner
    time_profile_mode_select: Select
    stack_order_select: Select
    highlight_token_select: Select

    center_panel: LayoutDOM
    left_panel: LayoutDOM
    right_panel: LayoutDOM


@dataclass
class _ContextControls:
    root: LayoutDOM

    trace_select: MultiSelect
    thread_select: MultiSelect
    token_mode_select: Select

    n_bins_spinner: Spinner
    time_profile_mode_select: Select
    stack_order_select: Select
    highlight_token_select: Select


class UIModel:
    def __init__(self, controller: "AppController") -> None:
        self.controller = controller

    def build(
        self,
        *,
        state: AppState,
        all_thread_names: tuple[str, ...],
    ) -> UIElements:
        shell = BokehAppShell().build()

        controls = self._build_context_controls(
            state               = state,
            all_thread_names    = all_thread_names,
        )

        shell.context_host.children = [controls.root]                       # type: ignore

        shell.inspector_host.children = [                                   # type: ignore
            self._build_inspector_placeholder(),
        ]

        return UIElements(
            root                    = shell.root,
            title                   = shell.title,
            status                  = shell.status,

            trace_select            = controls.trace_select,
            thread_select           = controls.thread_select,
            token_mode_select       = controls.token_mode_select,

            n_bins_spinner          = controls.n_bins_spinner,
            time_profile_mode_select= controls.time_profile_mode_select,
            stack_order_select      = controls.stack_order_select,
            highlight_token_select  = controls.highlight_token_select,

            center_panel            = shell.center_host,
            left_panel              = shell.context_host,
            right_panel             = shell.inspector_host,
        )

    def _build_context_controls(
        self,
        *,
        state: AppState,
        all_thread_names: tuple[str, ...],
    ) -> "_ContextControls":
        context = state.context
        time_profile = state.modules.time_profile

        trace_options = list(
            self.controller.trace_registry.trace_options()
        )

        trace_select = MultiSelect(
            title="Traces",
            value=list(context.traces.trace_ids),
            options=trace_options,                                          # type: ignore
            size=min(
                max(
                    len(self.controller.trace_registry.all_trace_ids()),
                    2,
                ),
                8,
            ),
            sizing_mode="stretch_width",
        )
        trace_select.on_change(
            "value",
            self.controller.on_traces_changed,
        )

        thread_select = MultiSelect(
            title="Threads",
            value=list(context.active_threads),
            options=[(name, name) for name in all_thread_names],
            size=min(max(len(all_thread_names), 6), 14),
            sizing_mode="stretch_width",
        )
        thread_select.on_change(
            "value",
            self.controller.on_threads_changed,
        )

        token_mode_select = Select(
            title="Token view",
            value=context.token_mode,
            options=[
                ("raw", "Raw"),
                ("named", "Named"),
            ],
            sizing_mode="stretch_width",
        )
        token_mode_select.on_change(
            "value",
            self.controller.on_token_mode_changed,
        )

        n_bins_spinner = Spinner(
            title="Number of bins",
            low=4,
            high=2000,
            step=4,
            value=time_profile.n_bins,
            sizing_mode="stretch_width",
        )
        n_bins_spinner.on_change(
            "value",
            self.controller.on_n_quanta_changed,
        )

        time_profile_mode_select = Select(
            title="Snapshot mode",
            value=time_profile.fidelity,
            options=[
                ("fast", "Fast"),
                ("balanced", "Balanced"),
                ("exact", "Exact"),
            ],
            sizing_mode="stretch_width",
        )
        time_profile_mode_select.on_change(
            "value",
            self.controller.on_time_profile_mode_changed,
        )

        stack_order_select = Select(
            title="Stack order",
            value=time_profile.order,
            options=[
                ("global", "Global"),
                ("local", "Local"),
            ],
            sizing_mode="stretch_width",
        )
        stack_order_select.on_change(
            "value",
            self.controller.on_time_profile_order_changed,
        )

        highlight_token_select = Select(
            title="Highlight token",
            value="",
            options=[("", "(none)")],
            sizing_mode="stretch_width",
        )
        highlight_token_select.on_change(
            "value",
            self.controller.on_highlight_token_changed,
        )

        root = column(
            panel_title("Traces"),
            trace_select,

            panel_title("Scope"),
            thread_select,
            token_mode_select,

            panel_title("Time profile"),
            n_bins_spinner,
            time_profile_mode_select,
            stack_order_select,
            highlight_token_select,

            sizing_mode="stretch_width",
            spacing=10,
        )

        return _ContextControls(
            root=root,
            trace_select=trace_select,
            thread_select=thread_select,
            token_mode_select=token_mode_select,
            n_bins_spinner=n_bins_spinner,
            time_profile_mode_select=time_profile_mode_select,
            stack_order_select=stack_order_select,
            highlight_token_select=highlight_token_select,
        )

    def _build_inspector_placeholder(self) -> LayoutDOM:
        return column(
            panel_title("Selection"),
            Div(
                text=(
                    "<div class='blup-empty-state'>"
                    "Select a token, interval, or chart region to inspect it."
                    "</div>"
                ),
                sizing_mode="stretch_width",
            ),

            panel_title("Controls"),
            Div(
                text=(
                    "<div class='blup-empty-state'>"
                    "Module controls will appear here."
                    "</div>"
                ),
                sizing_mode="stretch_width",
            ),

            sizing_mode="stretch_width",
            spacing=10,
        )
