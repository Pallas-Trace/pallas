from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from blup.bokeh.styles import collapsible_section, panel_title, section_label, style_widget
from blup.bokeh.theme import PALETTE, make_widget_stylesheet
from blup.state import ContextPatch, ModuleID, ModulePatch, TimeProfilePatch, TokenSelectionPatch, TraceSelectionPatch
from blup.types import TraceID
from bokeh.layouts import column
from bokeh.models.layouts import LayoutDOM
from bokeh.models.widgets.inputs import MultiSelect, Select, Spinner

if TYPE_CHECKING:
    from blup.controller import AppController

@dataclass
class ContextControls:
    root:                       LayoutDOM

    trace_select:               MultiSelect
    thread_select:              MultiSelect
    token_mode_select:          Select
    n_bins_spinner:             Spinner
    time_profile_mode_select:   Select
    stack_order_select:         Select
    highlight_token_select:     Select


class ContextSelectionPipeline:
    module_id: ModuleID = "context_selection"

    root: LayoutDOM | None
    host: "AppController | None"

    controls: ContextControls | None

    def __init__(self) -> None:
        self.root = None
        self.host = None
        self.controls = None

        self._syncing = False

    @property
    def subscribed_state(self) -> tuple[str, ...]:
        return (
            "context.traces.trace_ids",
            "context.active_threads",
            "context.token_mode",
            "modules.time_profile.n_bins",
            "modules.time_profile.fidelity",
            "modules.time_profile.order",
        )

    def build(self) -> LayoutDOM:
        self.root = column(sizing_mode="stretch_height")
        return self.root

    def bind(self, host: "AppController") -> None:
        self.host = host

    def refresh(self, host: "AppController") -> None:
        if self.root is None:
            self.build()

        state = host.state
        context = state.context
        time_profile = state.modules.time_profile

        trace_options = list(host.trace_registry.trace_options())
        thread_options = [
            (name, name)
            for name in host.all_thread_names()
        ]

        self._syncing = True
        try:
            if self.controls is None:
                self.controls = self._build_controls(
                    host            = host,
                    trace_options   = trace_options,
                    thread_options  = thread_options,
                )
                self.root.children = [self.controls.root]                   # type: ignore
                return
            controls = self.controls

            controls.trace_select.options = list(
                host.trace_registry.trace_options()
            )
            controls.trace_select.value = list(
                context.traces.trace_ids
            )

            controls.thread_select.options = [
                (name, name)
                for name in host.all_thread_names()
            ]
            controls.thread_select.value = list(
                context.active_threads
            )

            controls.token_mode_select.value = context.token_mode

            controls.n_bins_spinner.value = (
                time_profile.n_bins
            )
            controls.time_profile_mode_select.value = (
                time_profile.fidelity
            )
            controls.stack_order_select.value = (
                time_profile.order
            )
        finally:
            self._syncing = False

    def _build_controls(
        self,
        *,
        host: "AppController",
        trace_options: list[tuple[TraceID, str]],
        thread_options: list[tuple[str, str]],
    ) -> ContextControls:
        if not self.host:
            raise RuntimeError(
                "host AppController must be bound before building controls"
            )

        state = host.state
        context = state.context
        time_profile = state.modules.time_profile

        # freeze host for lambda closures
        h = host

        # --- Traces ---

        trace_select = MultiSelect(
            title           = "Traces",
            value           = list(context.traces.trace_ids),
            options         = trace_options,                                # type: ignore
            size            = min(max(len(trace_options), 2), 6),
            sizing_mode     = "stretch_width",
            stylesheets     = [make_widget_stylesheet()],
        )
        style_widget(trace_select)
        trace_select.on_change(
            "value",
            lambda attr, old, new, h=h: (
                None
                if self._syncing or h is None
                else h.update_state(
                    context = ContextPatch(
                        traces = TraceSelectionPatch(
                            trace_ids=tuple(new),
                        ),
                    ),
                )
            ),
        )

        # --- Threads ---

        thread_select = MultiSelect(
            title           = "Threads",
            value           = list(context.active_threads),
            options         = thread_options,                               # type: ignore
            size            = min(max(len(thread_options), 5), 10),
            sizing_mode     = "stretch_width",
            stylesheets     = [make_widget_stylesheet()],
        )
        style_widget(thread_select)
        thread_select.on_change(
            "value",
            lambda attr, old, new, h=h: (
                None
                if self._syncing or h is None
                else h.update_state(
                    context = ContextPatch(
                        active_threads=tuple(new),
                    ),
                )
            ),
        )

        # --- Token View ---

        token_mode_select = Select(
            title           = "Token view",
            value           = context.token_mode,
            options         = [
                ("raw", "Raw"),
                ("named", "Named"),
            ],
            sizing_mode     = "stretch_width",
            stylesheets     = [make_widget_stylesheet()],
        )
        style_widget(token_mode_select)
        token_mode_select.on_change(
            "value",
            lambda attr, old, new, h=h: (
                None
                if self._syncing or h is None
                else h.update_state(
                    context = ContextPatch(
                        token_mode=new,
                    ),
                )
            ),
        )

        # --- Time Profile ---

        n_bins_spinner = Spinner(
            title           = "Number of bins",
            low             = 4,
            high            = 2000,
            step            = 4,
            value           = time_profile.n_bins,
            sizing_mode     = "stretch_width",
            stylesheets     = [make_widget_stylesheet()],
        )
        style_widget(n_bins_spinner)
        n_bins_spinner.on_change(
            "value",
            lambda attr, old, new, h=h: (
                None
                if self._syncing or h is None
                else h.update_state(
                    modules = ModulePatch(
                        time_profile = TimeProfilePatch(
                            n_bins=int(new),
                        ),
                    ),
                )
            ),
        )

        time_profile_mode_select = Select(
            title           = "Snapshot mode",
            value           = time_profile.fidelity,
            options         = [
                ("fast", "Fast"),
                ("balanced", "Balanced"),
                ("exact", "Exact"),
            ],
            sizing_mode     = "stretch_width",
            stylesheets     = [make_widget_stylesheet()],
        )
        style_widget(time_profile_mode_select)
        time_profile_mode_select.on_change(
            "value",
            lambda attr, old, new, h=h: (
                None
                if self._syncing or h is None
                else h.update_state(
                    modules = ModulePatch(
                        time_profile = TimeProfilePatch(
                            fidelity=new,
                        ),
                    ),
                )
            ),
        )

        stack_order_select = Select(
            title           = "Stack order",
            value           = time_profile.order,
            options         = [
                ("global", "Global"),
                ("local", "Local"),
            ],
            sizing_mode     = "stretch_width",
            stylesheets     = [make_widget_stylesheet()],
        )
        style_widget(stack_order_select)
        stack_order_select.on_change(
            "value",
            lambda attr, old, new, h=h: (
                None
                if self._syncing or h is None
                else h.update_state(
                    modules = ModulePatch(
                        time_profile = TimeProfilePatch(
                            order=new,
                        ),
                    ),
                )
            ),
        )

        highlight_token_select = Select(
            title           = "Highlight token",
            value           = "",
            options         = [("", "(none)")],
            sizing_mode     = "stretch_width",
            stylesheets     = [make_widget_stylesheet()],
        )
        style_widget(highlight_token_select)
        highlight_token_select.on_change(
            "value",
            lambda attr, old, new, h=h: (
                None
                if self._syncing or h is None
                else h.update_state(
                    context = ContextPatch(
                        selection = TokenSelectionPatch(
                            token=new,
                        ),
                    ),
                )
            ),
        )

        # --- Layout ---

        traces_section = collapsible_section(
            "Traces",
            column(trace_select, sizing_mode="stretch_width"),
        )

        scope_section = collapsible_section(
            "Scope",
            column(
                thread_select,
                token_mode_select,
                sizing_mode="stretch_width",
                spacing=6,
            ),
        )

        time_profile_section = collapsible_section(
            "Time Profile",
            column(
                n_bins_spinner,
                time_profile_mode_select,
                stack_order_select,
                highlight_token_select,
                sizing_mode="stretch_width",
                spacing=6,
            ),
        )

        root = column(
            traces_section,
            scope_section,
            time_profile_section,

            sizing_mode = "stretch_width",
            spacing     = 0,
            margin      = (0, 12, 12, 12),
        )

        return ContextControls(
            root=root,
            trace_select=trace_select,
            thread_select=thread_select,
            token_mode_select=token_mode_select,
            n_bins_spinner=n_bins_spinner,
            time_profile_mode_select=time_profile_mode_select,
            stack_order_select=stack_order_select,
            highlight_token_select=highlight_token_select,
        )
