from __future__ import annotations

import re
from dataclasses import dataclass

from bokeh.io import curdoc
from bokeh.models.layouts import LayoutDOM

from blup.colors import TokenColor
from blup.data_model import FidelityMode, TokenMode, CATEGORY_TOKEN_TYPE
from blup.modules.interface import Pipeline, WorkManager
from blup.modules.time_profile.pipeline import TimeProfilePipeline
from blup.ui import UIElements, UIModel
from blup.state import (
    ContextPatch,
    DisplayPatch,
    ModulePatch,
    StateManager,
    TimeProfilePatch,
    TokenSelectionPatch,
    TraceMode,
    ModuleID,
    TimeProfileOrder,
    PanelState,
    DisplayState,
    ContextState,
    ModuleState,
    AppState,
    TraceSelectionPatch,
    TraceSelectionState,
)
from blup.traces.session import TraceSession
from blup.utils import timed
from blup.traces.interface import TraceRecord, TraceRegistryAccess
from blup.traces.registry import TraceRegistry


@dataclass
class ControllerRuntime:
    ui:                 UIElements | None = None
    root:               LayoutDOM | None = None


class AppController:
    ui_model:           UIModel
    runtime:            ControllerRuntime

    trace_registry:     TraceRegistryAccess
    state_manager:      StateManager
    work_manager:       WorkManager

    def __init__(self, trace_records: list[TraceRecord]) -> None:
        self.doc = curdoc()

        # setup trace registry with loaded trace records
        self.trace_registry = TraceRegistry(trace_records)
        self._install_merged_category_namespace()

        # setup state manager with default initial state
        self.state_manager = StateManager(
            self.trace_registry,
            self.initial_state(),
        )

        # setup ui
        self.ui_model = UIModel(self)
        self.token_color = TokenColor()
        self._register_loaded_trace_tokens()

        # setup work manager and load modules
        self.work_manager = WorkManager(
            schedule_display_callback = self.doc.add_next_tick_callback,    # type: ignore
            max_workers = 8,
        )

        self.module_pipelines: dict[ModuleID, Pipeline] = {
            "time_profile": TimeProfilePipeline(width=1650, height=950)
        }

        # setup runtime
        self.runtime = ControllerRuntime()

        # set internal logic flags
        self._refresh_scheduled = False

    # -------------------------------------------
    # |              Lifecycle                  |
    # -------------------------------------------

    def build(self):
        with timed("build.module_roots"):
            self.build_module_roots()

        with timed("build.ui"):
            self.build_ui_shell()

        with timed("build.mount_current_displays"):
            self.mount_current_displays()

        with timed("build.bind_active_pipelines"):
            self.bind_active_pipelines()

        with timed("build.refresh_tick"):
            self.refresh_tick()

        return self.runtime.root

    def build_module_roots(self) -> None:
        for pipeline in self.module_pipelines.values():
            pipeline.build()

    def build_ui_shell(self) -> None:
        ui = self.ui_model.build(
            state = self.state,
            all_thread_names = self.trace_registry.thread_names_for(
                self.state.context.traces.trace_ids,
            ),
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

    def get_module_pipeline(self, module_id: ModuleID) -> Pipeline:
        return self.module_pipelines[module_id]

    def active_module_ids(self) -> tuple[ModuleID, ...]:
        ids = [self.state.display.center.active_module]
        if self.state.display.left is not None:
            ids.append(self.state.display.left.active_module)
        if self.state.display.right is not None:
            ids.append(self.state.display.right.active_module)
        return tuple(dict.fromkeys(ids))                                    # type: ignore

    def active_pipelines(self) -> tuple[Pipeline, ...]:
        return tuple(
            self.get_module_pipeline(id)
            for id in self.active_module_ids()
        )

    def mount_current_displays(self) -> None:
        ui = self.runtime.ui
        if ui is None:
            return

        center_root = (
            self
            .get_module_pipeline(self.state.display.center.active_module)
            .root
        )
        ui.center_panel.children = (                                        # type: ignore
            [center_root]
            if center_root is not None else []
        )

        left_children: list[LayoutDOM] = []
        if self.state.display.left is not None:
            left_root = (
                self
                .get_module_pipeline(self.state.display.left.active_module)
                .root
            )
            if left_root is not None:
                left_children.append(left_root)
        ui.left_panel.children = left_children                              # type: ignore

        right_children: list[LayoutDOM] = []
        if self.state.display.right is not None:
            right_root = (
                self
                .get_module_pipeline(self.state.display.right.active_module)
                .root
            )
            if right_root is not None:
                right_children.append(right_root)
        ui.right_panel.children = right_children                            # type: ignore

    # -------------------------------------------
    # |            State Management             |
    # -------------------------------------------

    @property
    def state(self) -> AppState:
        return self.state_manager.state

    def initial_state(self) -> AppState:
        trace_ids = self.trace_registry.all_trace_ids()

        if not trace_ids:
            raise ValueError(
                "AppController requires at least one trace"
            )

        return AppState(
            display = DisplayState(
                center = PanelState(
                    active_module   = "time_profile",
                    context_key     = "main"
                ),
                left   = None,
                right  = None,
            ),
            context = ContextState(
                traces = TraceSelectionState(
                    trace_ids = (trace_ids[0],),
                )
            ),
            modules = ModuleState(),
        )

    def update_state(
        self,
        *,
        display: DisplayPatch | None = None,
        context: ContextPatch | None = None,
        modules: ModulePatch | None = None,
    ) -> None:
        update_applied = self.state_manager.update(
            display = display,
            context = context,
            modules = modules,
        )

        if not update_applied:
            return

        self.sync_ui_controls()
        self.schedule_refresh()

    def sync_ui_controls(self) -> None:
        ui = self.runtime.ui
        if ui is None:
            return

        traces = self.state.context.traces

        ui.trace_select.options = list(
            self.trace_registry.trace_options(),
        )
        ui.trace_select.value = list(traces.trace_ids)

        ui.thread_select.options = [
            (name, name) 
            for name in self.trace_registry.thread_names_for(
                traces.trace_ids,
            )
        ]
        ui.thread_select.value = list(
            self.state.context.active_threads,
        )

    # -------------------------------------------
    # |             Event Adapters              |
    # -------------------------------------------

    def on_traces_changed(self, attr, old, new):
        self.update_state(
            context = ContextPatch(
                traces = TraceSelectionPatch(
                    trace_ids=tuple(new),
                ),
            ),
        )

    def on_threads_changed(self, attr, old, new):
        self.update_state(
            context = ContextPatch(
                active_threads=tuple(new),
            ),
        )

    def on_n_quanta_changed(self, attr, old, new):
        if new is None:
            return
        self.update_state(
            modules = ModulePatch(
                time_profile = TimeProfilePatch(
                    n_bins=int(new),
                ),
            ),
        )

    def on_token_mode_changed(self, attr, old, new):
        self.update_state(
            context = ContextPatch(
                token_mode=new,
            ),
        )

    def on_time_profile_mode_changed(self, attr, old, new):
        self.update_state(
            modules = ModulePatch(
                time_profile = TimeProfilePatch(
                    fidelity=new,
                ),
            ),
        )

    def on_time_profile_order_changed(self, attr, old, new):
        self.update_state(
            modules = ModulePatch(
                time_profile = TimeProfilePatch(
                    order=new,
                ),
            ),
        )

    def on_trace_ids_changed(self, attr, old, new):
        self.update_state(
            context = ContextPatch(
                traces = TraceSelectionPatch(
                    trace_ids=tuple(new),
                ),
            ),
        )

    def on_focus_trace_changed(self, attr, old, new):
        self.update_state(
            context = ContextPatch(
                traces = TraceSelectionPatch(
                    focus_id=new or None,
                ),
            ),
        )

    def on_highlight_token_changed(self, attr, old, new):
        self.update_state(
            context = ContextPatch(
                selection = TokenSelectionPatch(
                    token=new,
                ),
            ),
        )

    # -------------------------------------------
    # |               Utilities                 |
    # -------------------------------------------

    def get_selected_sessions(self) -> tuple[TraceSession, ...]:
        return self.trace_registry.get_sessions(
            self.state.context.traces.trace_ids
        )

    def get_focused_session(self) -> TraceSession | None:
        focus_id = self.state.context.traces.focus_id
        if focus_id is None:
            return None
        return self.trace_registry.get_session(focus_id)

    def all_thread_names(self) -> tuple[str, ...]:
        return self.trace_registry.thread_names_for(
            self.state.context.traces.trace_ids
        )

    def full_time_bounds(self) -> tuple[int, int]:
        bounds = self.trace_registry.time_bounds_for(
            self.state.context.traces.trace_ids
        )
        if bounds is None:
            raise RuntimeError(
                "No selected traces are available"
            )
        return bounds

    def _install_merged_category_namespace(self) -> None:
        names = sorted({
            str(name)
            for record in self.trace_registry.all_trace_records()
            for name in record.meta.cat_key_to_name.values()
        })

        name_to_cat_token = {
            name: (CATEGORY_TOKEN_TYPE, index)
            for index, name in enumerate(names)
        }

        for record in self.trace_registry.all_trace_records():
            record.session.install_category_namespace(
                name_to_cat_token,
            )

    def _register_loaded_trace_tokens(self) -> None:
        for record in self.trace_registry.all_trace_records():
            token_name_by_key = dict(
                record.meta.token_key_to_name,
            )

            self.token_color.register_tokens(
                token_name_by_key.keys(),
                token_names = token_name_by_key,
                namespace   = record.trace_id
            )


