from __future__ import annotations

import re
import math
from dataclasses import dataclass

from bokeh.io import curdoc
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
from pipelines.base import DisplayPipeline
from pipelines.quanta_pipeline import QuantaPipeline
from pipelines.inspector_pipeline import InspectorPipeline
from pipelines.summary_pipeline import SummaryPipeline
from work_manager import WorkManager
from utils import timed


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


class AppController:
    t1:                 TraceSession
    t2:                 TraceSession
    state:              AppState
    ui_model:           UIModel
    runtime:            ControllerRuntime

    def __init__(self, t1: TraceSession, t2: TraceSession):
        # install TraceSession objects
        self.t1 = t1
        self.t2 = t2

        # initial state setup
        self.doc = curdoc()
        self.install_merged_category_namespace()
        self.state = self.initial_state()

        self.ui_model = UIModel(self)
        self.runtime = ControllerRuntime()

        self.pipelines: dict[ViewId, DisplayPipeline] = {
            "quanta": QuantaPipeline(t1, t2, width=1650, height=950),
            "inspector": InspectorPipeline(t1, t2, width=360),
            "summary": SummaryPipeline(t1, t2, width=750, height=400),
        }

        self.work_manager = WorkManager(
            schedule_display_callback = self.doc.add_next_tick_callback,  # type: ignore
            max_workers = 8,
        )

        # internal logic flags
        self._refresh_scheduled = False

    # -------------------------------------------
    # |              Lifecycle                  |
    # -------------------------------------------

    def build(self):
        with timed("build.view_roots"):
            self.build_pipeline_roots()

        with timed("build.ui"):
            self.build_ui_shell()

        with timed("build.mount_current_displays"):
            self.mount_current_displays()

        with timed("build.bind_view_callbacks"):
            self.bind_active_pipelines()

        with timed("build.refresh_tick"):
            self.refresh_tick()

        return self.runtime.root

    def build_pipeline_roots(self) -> None:
        for pipeline in self.pipelines.values():
            pipeline.build()

    def build_ui_shell(self) -> None:
        ui = self.ui_model.build(
            state = self.state,
            all_thread_names = self.all_thread_names(),
        )
        self.runtime.ui = ui
        self.runtime.root = ui.root

    # -------------------------------------------
    # |               Refresh                   |
    # -------------------------------------------

    def schedule_refresh(self) -> None:
        if self._refresh_scheduled:
            return
        self._refresh_scheduled = True
        curdoc().add_next_tick_callback(self.refresh_tick)

    def refresh_tick(self) -> None:
        self._refresh_scheduled = False
        self.mount_current_displays()
        self.bind_active_pipelines()
        self.refresh_current_pipelines()

    def bind_active_pipelines(self) -> None:
        for pipeline in self.active_pipelines():
            pipeline.bind(self)

    def refresh_current_pipelines(self) -> None:
        for pipeline in self.active_pipelines():
            pipeline.refresh(self)

    # -------------------------------------------
    # |               Display                   |
    # -------------------------------------------

    def get_pipeline(self, view_id: ViewId) -> DisplayPipeline:
        return self.pipelines[view_id]

    def active_view_ids(self) -> tuple[ViewId, ...]:
        ids = [self.state.display.primary.active_view]
        ids.extend(panel.active_view for panel in self.state.display.secondary)
        return tuple(dict.fromkeys(ids))  # type: ignore

    def active_pipelines(self) -> tuple[DisplayPipeline, ...]:
        return tuple(self.get_pipeline(id) for id in self.active_view_ids())

    def mount_current_displays(self) -> None:
        ui = self.runtime.ui
        if ui is None:
            return

        primary_root = self.get_pipeline(self.state.display.primary.active_view).root()
        ui.primary_panel.children = [primary_root] if primary_root is not None else []  # type: ignore

        secondary_children: list[LayoutDOM] = []
        for panel in self.state.display.secondary:
            root = self.get_pipeline(panel.active_view).root()
            if root is not None:
                secondary_children.append(root)
        ui.secondary_panel.children = secondary_children  # type: ignore

    # -------------------------------------------
    # |            State Management             |
    # -------------------------------------------

    def initial_state(self) -> AppState:
        names = tuple(self.all_thread_names())
        return AppState(
            views=ViewState(),
            context=ContextState(
                active_threads=names,
                trace_mode="single",
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

    def set_highlight_token(self, token: tuple[int, int] | None) -> None:
        if self.state.context.selection.token == token:
            return
        self.state.context.selection.token = token
        self.schedule_refresh()

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
        summary = self.get_pipeline("summary")
        assert isinstance(summary, SummaryPipeline)
        summary.on_highlight_widget_changed(self, new)

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

    def full_time_bounds(self) -> tuple[int, int]:
        return (
            min(self.t1.meta.start_ns, self.t2.meta.start_ns),
            max(self.t1.meta.end_ns, self.t2.meta.end_ns),
        )


