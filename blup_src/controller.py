from __future__ import annotations

import re
import math
from dataclasses import dataclass

from bokeh.models.layouts import LayoutDOM

from ui import UIElements, UIModel
from data_model import FidelityMode, TokenMode, CATEGORY_TOKEN_TYPE
from state import (
    AppState,
    ContextState,
    DisplayState,
    PanelState,
    QuantaOrder,
    ViewId,
    ViewState,
)
from trace_session import TraceSession
from adapters.summary_adapter import SequenceSummaryDiffAdapter, SequenceSummaryDiffRow, format_duration_ns
from views.quanta_view import QuantaView
from views.inspector_view import InspectorView
from views.summary_view import SummaryView
from utils import timed
from pipelines.base import DisplayPipeline
from pipelines.quanta_pipeline import QuantaPipeline
from pipelines.inspector_pipeline import InspectorPipeline
from pipelines.summary_pipeline import SummaryPipeline


THREAD_RE = re.compile(r"^P#(\d+)T#(\d+)$")

def thread_sort_key(name: str):
    m = THREAD_RE.match(name)
    if m:
        return (0, int(m.group(1)), int(m.group(2)), name)
    nums = tuple(int(x) for x in re.findall(r"\d+", name))
    if nums:
        return (1, *nums, name)
    return (2, name)


@dataclass
class ControllerRuntime:
    ui:                 UIElements | None = None
    root:               LayoutDOM | None = None

    quanta_root:        LayoutDOM | None = None
    inspector_root:     LayoutDOM | None = None
    summary_root:       LayoutDOM | None = None

    sequence_rows:      tuple[SequenceSummaryDiffRow, ...] = ()


class AppController:
    t1:                 TraceSession
    t2:                 TraceSession
    state:              AppState
    ui_model:           UIModel
    runtime:            ControllerRuntime

    quanta_view:        QuantaView
    inspector_view:     InspectorView
    summary_view:       SummaryView

    def __init__(self, t1: TraceSession, t2: TraceSession):
        # install TraceSession objects
        self.t1 = t1
        self.t2 = t2

        # initial state setup
        self.install_merged_category_namespace()
        self.state = self.initial_state()

        self.ui_model = UIModel(self)
        self.runtime = ControllerRuntime()

        self.pipelines: dict[ViewId, DisplayPipeline] = {
            "quanta": QuantaPipeline(t1, t2, width=1350, height=950),
            "inspector": InspectorPipeline(t1, t2, width=360),
            "summary": SummaryPipeline(t1, t2, width=750, height=400),
        }

        # view initializers
        self.quanta_view = QuantaView(t1, t2, width=1350, height=950)
        self.quanta_view.on_token_selected = self.on_quanta_token_selected  # type: ignore

        self.inspector_view = InspectorView(t1, t2, width=360)

        self.summary_view = SummaryView(width=750, height=400)
        self.summary_adapter = SequenceSummaryDiffAdapter(t1, t2)

        # internal logic flags
        self._refresh_scheduled = False
        self._range_refresh_scheduled = False
        self._ignore_range_callbacks = False
        self._ignore_highlight_callbacks = False

    # -------------------------------------------
    # |              Lifecycle                  |
    # -------------------------------------------

    def build(self):
        with timed("build.view_roots"):
            self.build_view_roots()

        with timed("build.ui"):
            self.build_ui_shell()

        with timed("build.mount_current_displays"):
            self.mount_current_displays()

        with timed("build.bind_view_callbacks"):
            self.bind_view_callbacks()

        with timed("build.refresh_tick"):
            self.refresh_tick()

        return self.runtime.root

    def build_view_roots(self) -> None:
        self.runtime.quanta_root = self.quanta_view.build()
        self.runtime.inspector_root = self.inspector_view.build()
        self.runtime.summary_root = self.summary_view.build()

    def build_ui_shell(self) -> None:
        ui = self.ui_model.build(
            state = self.state,
            all_thread_names = self.all_thread_names(),
        )
        self.runtime.ui = ui
        self.runtime.root = ui.root

    def mount_current_displays(self) -> None:
        ui = self.runtime.ui
        if ui is None:
            return

        primary_root = self.get_view_root(self.state.display.primary.active_view)
        ui.primary_panel.children = [primary_root] if primary_root is not None else []  # type: ignore

        secondary_children: list[LayoutDOM] = []
        for panel in self.state.display.secondary:
            root = self.get_view_root(panel.active_view)
            if root is not None:
                secondary_children.append(root)
        ui.secondary_panel.children = secondary_children  # type: ignore

    def bind_view_callbacks(self) -> None:
        self.bind_time_range_callbacks()

    def schedule_refresh(self) -> None:
        if self._refresh_scheduled:
            return
        self._refresh_scheduled = True
        self.quanta_view.doc.add_next_tick_callback(self.refresh_tick)  # type: ignore

    def refresh_tick(self) -> None:
        self._refresh_scheduled = False
        self._ignore_range_callbacks = True
        try:
            self.mount_current_displays()
            self.bind_view_callbacks()
            self.refresh_current_views()
        finally:
            self._ignore_range_callbacks = False

    def get_view_root(self, view_id: ViewId) -> LayoutDOM | None:
        if view_id == "quanta":
            return self.runtime.quanta_root
        if view_id == "inspector":
            return self.runtime.inspector_root
        if view_id == "summary":
            return self.runtime.summary_root
        return None


    def active_view_ids(self) -> tuple[ViewId, ...]:
        ids = [self.state.display.primary.active_view]
        ids.extend(panel.active_view for panel in self.state.display.secondary)
        return tuple(dict.fromkeys(ids))  # type: ignore

    def refresh_current_views(self) -> None:
        for view_id in self.active_view_ids():
            self.refresh_view(view_id)

    def refresh_view(self, view_id: ViewId) -> None:
        if view_id == "quanta":
            self._refresh_quanta_view()
        elif view_id == "inspector":
            self._refresh_inspector_view()
        elif view_id == "summary":
            self._refresh_summary_view()

    def _refresh_quanta_view(self) -> None:
        with timed("quanta_view.update"):
            self.quanta_view.update(
                active_thread_names = list(self.state.context.active_threads),
                n_quanta            = self.state.views.quanta.n_bins,
                mode                = self.state.views.quanta.mode,
                token_mode          = self.state.context.token_mode,
                stack_order         = self.state.views.quanta.order,
                window_t0_ns        = self.state.context.time_scope.t0_ns,
                window_t1_ns        = self.state.context.time_scope.t1_ns,
            )

    def _refresh_inspector_view(self) -> None:
        with timed("inspector_view.update"):
            self.inspector_view.update(
                active_threads      = list(self.state.context.active_threads),
                n_quanta            = self.state.views.quanta.n_bins,
                mode                = self.state.views.quanta.mode,
                token_mode          = self.state.context.token_mode,
                stack_order         = self.state.views.quanta.order,
            )

    def _refresh_summary_view(self) -> None:
        with timed("summary_view.update"):
            rows = self.summary_adapter.build_rows(
                token_mode          = self.state.context.token_mode,
                fidelity            = "fast",
                top_k               = 32,
                active_thread_names = tuple(self.state.context.active_threads),
            )
            self.runtime.sequence_rows = rows
            self.refresh_highlight_token_select(rows)
            self.refresh_summary_view()

    # -------------------------------------------
    # |            State Management             |
    # -------------------------------------------

    def initial_state(self) -> AppState:
        names = tuple(self.all_thread_names())
        return AppState(
            views=ViewState(),
            context=ContextState(
                active_threads=names,
            ),
            display=DisplayState(
                primary=PanelState(
                    active_view="quanta",
                    context_key="main",
                ),
                secondary=(
                    PanelState(active_view="inspector", context_key="main"),
                    PanelState(active_view="summary", context_key="main"),
                ),
            ),
        )

    def set_active_threads(self, thread_names: tuple[str, ...]) -> None:
        chosen_threads = thread_names or tuple(self.all_thread_names())
        if self.state.context.active_threads == chosen_threads:
            return
        self.state.context.active_threads = chosen_threads
        self.schedule_refresh()

    def set_n_quanta(self, n: int) -> None:
        if self.state.views.quanta.n_bins == n:
            return
        self.state.views.quanta.n_bins = n
        self.schedule_refresh()

    def set_token_mode(self, mode: TokenMode) -> None:
        if self.state.context.token_mode == mode:
            return
        self.state.context.token_mode = mode
        self.schedule_refresh()

    def set_quanta_mode(self, mode: FidelityMode) -> None:
        if self.state.views.quanta.mode == mode:
            return
        self.state.views.quanta.mode = mode
        self.schedule_refresh()

    def set_quanta_order(self, order: QuantaOrder) -> None:
        if self.state.views.quanta.order == order:
            return
        self.state.views.quanta.order = order
        self.schedule_refresh()

    def set_highlight_token(
        self,
        token: tuple[int, int] | None,
        *,
        sync_widget: bool = True,
        refresh_summary: bool = True,
    ) -> None:
        if self.state.context.selection.token != token:
            self.state.context.selection.token = token
        if sync_widget:
            self.sync_highlight_widget(token)
        if refresh_summary:
            self.refresh_summary_view()

    # -------------------------------------------
    # |             Event Adapters              |
    # -------------------------------------------

    def on_threads_changed(self, attr, old, new):
        self.set_active_threads(tuple(new))

    def on_n_quanta_changed(self, attr, old, new):
        if new is None:
            return
        self.set_n_quanta(int(new))

    def on_token_mode_changed(self, attr, old, new):
        self.set_token_mode(new)

    def on_quanta_mode_changed(self, attr, old, new):
        self.set_quanta_mode(new)

    def on_quanta_order_changed(self, attr, old, new):
        self.set_quanta_order(new)

    def on_highlight_token_changed(self, attr, old, new):
        if self._ignore_highlight_callbacks:
            return
        self.set_highlight_token(
            self.select_value_to_token(new),
            sync_widget=False,
            refresh_summary=True,
        )

    def on_quanta_token_selected(self, token: tuple[int, int] | None) -> None:
        self.set_highlight_token(
            token,
            sync_widget=True,
            refresh_summary=True,
        )

    # -------------------------------------------
    # |         Selection Coordination          |
    # -------------------------------------------

    def refresh_highlight_token_select(
        self,
        rows: tuple[SequenceSummaryDiffRow, ...],
    ) -> None:
        if self.runtime.ui is None:
            return

        widget = self.runtime.ui.highlight_token_select
        options = self.highlight_token_options(rows)
        valid_values = {value for value, _ in options}

        current_value = self.token_to_select_value(self.state.context.selection.token)
        if current_value not in valid_values:
            current_value = options[1][0] if len(options) > 1 else ""
            self.state.context.selection.token = self.select_value_to_token(current_value)

        self._ignore_sequence_callbacks = True
        try:
            widget.options = options  # type: ignore
            if widget.value != current_value:
                widget.value = current_value
        finally:
            self._ignore_sequence_callbacks = False

    def highlight_token_options(
        self,
        rows: tuple[SequenceSummaryDiffRow, ...],
    ) -> list[tuple[str, str]]:
        opts: list[tuple[str, str]] = [("", "(none)")]
        for row in rows:
            value = self.token_to_select_value((row.token_type, row.token_id))
            label = (
                f"#{row.contribution_rank} "
                f"{row.name} "
                f"({format_duration_ns(row.contribution_abs_ns)}, "
                f"{row.contribution_share_pct:.1f}%) "
                f"[{row.token_type}:{row.token_id}]"
            )
            opts.append((value, label))
        return opts

    def sync_highlight_widget(self, token: tuple[int, int] | None) -> None:
        if self.runtime.ui is None:
            return

        widget = self.runtime.ui.highlight_token_select
        value = self.token_to_select_value(token)

        self._ignore_highlight_callbacks = True
        try:
            if widget.value != value:
                widget.value = value
        finally:
            self._ignore_highlight_callbacks = False

    def refresh_summary_view(self) -> None:
        token = self.state.context.selection.token
        row = next((r for r in self.runtime.sequence_rows if r.token == token), None)

        t0_ns = self.state.context.time_scope.t0_ns
        t1_ns = self.state.context.time_scope.t1_ns
        if t0_ns is None or t1_ns is None:
            t0_ns = self.quanta_view.full_start_ns
            t1_ns = self.quanta_view.full_end_ns

        model = self.summary_adapter.build_display_model(
            row,
            active_thread_names = tuple(self.state.context.active_threads),
            token_mode          = self.state.context.token_mode,
            t0_ns               = t0_ns,
            t1_ns               = t1_ns,
            histogram_bins      = 20,
        )
        self.summary_view.update(model)

    # -------------------------------------------
    # |               Viewport                  |
    # -------------------------------------------

    def bind_time_range_callbacks(self) -> None:
        fig = self.quanta_view.fig
        if fig is None:
            return
        fig.x_range.on_change("start", self.on_time_range_changed)  # type: ignore
        fig.x_range.on_change("end", self.on_time_range_changed)  # type: ignore

    def on_time_range_changed(self, attr, old, new) -> None:
        if self._ignore_range_callbacks:
            return
        if self._range_refresh_scheduled:
            return
        if self.quanta_view.doc is None:
            return

        self._range_refresh_scheduled = True
        self.quanta_view.doc.add_next_tick_callback(self.apply_time_range_change)

    def apply_time_range_change(self) -> None:
        self._range_refresh_scheduled = False

        fig = self.quanta_view.fig
        if fig is None:
            return

        start_ms = fig.x_range.start  # type: ignore
        end_ms = fig.x_range.end      # type: ignore
        if start_ms is None or end_ms is None:
            return
        if not math.isfinite(start_ms) or not math.isfinite(end_ms):
            return
        if end_ms <= start_ms:
            return

        full_t0_ns = self.quanta_view.full_start_ns
        full_t1_ns = self.quanta_view.full_end_ns

        new_t0_ns = max(full_t0_ns, int(start_ms * 1e6))
        new_t1_ns = min(full_t1_ns, int(end_ms * 1e6))
        if new_t1_ns <= new_t0_ns:
            return

        scope = self.state.context.time_scope

        full_t0_ms = full_t0_ns / 1e6
        full_t1_ms = full_t1_ns / 1e6
        eps_ms = 1e-9

        if abs(start_ms - full_t0_ms) <= eps_ms and abs(end_ms - full_t1_ms) <= eps_ms:
            next_t0_ns = None
            next_t1_ns = None
        else:
            next_t0_ns = new_t0_ns
            next_t1_ns = new_t1_ns

        if scope.t0_ns == next_t0_ns and scope.t1_ns == next_t1_ns:
            return

        scope.t0_ns = next_t0_ns
        scope.t1_ns = next_t1_ns
        self.schedule_refresh()

    # -------------------------------------------
    # |               Utilities                 |
    # -------------------------------------------

    def install_merged_category_namespace(self) -> None:
        names = sorted(
            set(str(name) for name in self.t1.meta.cat_key_to_name.values())
            | set(str(name) for name in self.t2.meta.cat_key_to_name.values())
        )

        name_to_cat_token = {
            name: (CATEGORY_TOKEN_TYPE, int(i))
            for i, name in enumerate(names)
        }

        self.t1.install_category_namespace(name_to_cat_token)
        self.t2.install_category_namespace(name_to_cat_token)

    def all_thread_names(self) -> list[str]:
        names = set(map(str, self.t1.meta.thread_names)) | set(map(str, self.t2.meta.thread_names))
        return sorted(names, key=thread_sort_key)

    def token_to_select_value(self, token: tuple[int, int] | None) -> str:
        if token is None:
            return ""
        return f"{token[0]}:{token[1]}"

    def select_value_to_token(self, value: str) -> tuple[int, int] | None:
        if not value:
            return None
        a, b = value.split(":", 1)
        return (int(a), int(b))


