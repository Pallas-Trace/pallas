from __future__ import annotations

import re
from dataclasses import dataclass

from blup.bokeh.app_shell import collapse_arrows
from blup.modules.context_selection.pipeline import ContextSelectionPipeline
from blup.modules.token_detail.pipeline import TokenDetailPipeline
from blup.shell.intents import IntentBus
from blup.shell.layout import ShellDimensions
from bokeh.io import curdoc
from bokeh.models.layouts import LayoutDOM

from blup.colors import TokenColor
from blup.data_model import FidelityMode, TokenMode, CATEGORY_TOKEN_TYPE
from blup.modules.interface import Pipeline, WorkPipeline, WorkManager
from blup.modules.time_profile.pipeline import TimeProfilePipeline
from blup.ui import UIElements, UIModel
from blup.state import (
    ContextPatch,
    DisplayPatch,
    ModulePatch,
    PanelPatch,
    StateManager,
    ModuleID,
    PanelState,
    DisplayState,
    ContextState,
    ModuleState,
    AppState,
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
    dims:               ShellDimensions = ShellDimensions()


class AppController:
    ui_model:           UIModel
    runtime:            ControllerRuntime

    trace_registry:     TraceRegistryAccess
    state_manager:      StateManager
    work_manager:       WorkManager
    intent_bus:         IntentBus

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

        # setup work manager and register modules
        self.work_manager = WorkManager(
            schedule_display_callback = self.doc.add_next_tick_callback,    # type: ignore
            max_workers = 8,
        )
        self.module_pipelines: dict[ModuleID, Pipeline] = {
            "context_selection": ContextSelectionPipeline(),
            "time_profile": TimeProfilePipeline(height=700),
            "token_detail": TokenDetailPipeline(height=700),
        }
        self.work_pipelines: dict[ModuleID, WorkPipeline] = {               # type: ignore[assignment]
            "time_profile": self.module_pipelines["time_profile"],          # type: ignore[assignment]
            "token_detail": self.module_pipelines["token_detail"]           # type: ignore[assignment]
        }

        # setup ui
        self.ui_model = UIModel(self)
        self.token_color = TokenColor()
        self._register_loaded_trace_tokens()

        # intents bus
        self.intent_bus = IntentBus()
        self.intent_bus.panel.on_change("data", self._on_panel_intent)
        print(f"[setup] subscribed to panel bus: {id(self.intent_bus.panel)}")

        # setup controller runtime
        self.runtime = ControllerRuntime()

        # set internal logic flags
        self._refresh_scheduled = False

    # -------------------------------------------
    # |           Lifecycle - Build             |
    # -------------------------------------------

    def build(self):
        with timed("build.module_roots"):
            self.build_module_roots()

        with timed("build.ui"):
            self.build_ui_shell()

        with timed("build.mount_current_displays"):
            self.mount_active_displays()

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
    # |          Lifecycle - Refresh            |
    # -------------------------------------------

    def schedule_refresh(self) -> None:
        if self._refresh_scheduled:
            return
        self._refresh_scheduled = True
        curdoc().add_next_tick_callback(self.refresh_tick)

    def refresh_tick(self) -> None:
        self._refresh_scheduled = False

        self.bind_active_pipelines()

        for pipeline in self.active_pipelines():
            if self.state_manager.refresh_needed(pipeline.subscribed_state):
                pipeline.refresh(self)

        self.mount_active_displays()
        self.sync_panel_layout()

        self.state_manager.mark_synced()

    def bind_active_pipelines(self) -> None:
        for pipeline in self.active_pipelines():
            pipeline.bind(self)

    def mount_active_displays(self) -> None:
        ui = self.runtime.ui
        if ui is None:
            return

        self._mount_panel(
            host    = ui.main_host,
            panel   = self.state.display.main,
        )
        self._mount_panel(
            host    = ui.context_host,
            panel   = self.state.display.context,
        )
        self._mount_panel(
            host    = ui.inspector_host,
            panel   = self.state.display.inspector,
        )

    def _mount_panel(
        self,
        *,
        host: LayoutDOM,
        panel: PanelState,
    ) -> None:
        if panel.active_module is None:
            host.children = []                                              # type: ignore[attr-defined]
            return

        root = self.get_module_pipeline(panel.active_module).root

        host.children = (                                                   # type: ignore[attr-defined]
            [root]
            if root is not None
            else []
        )

    def sync_panel_layout(self) -> None:
        ui = self.runtime.ui
        if ui is None:
            return

        for name in ("context", "inspector"):
            ps: PanelState = getattr(self.state.display, name)

            panel = getattr(ui, f"{name}_panel")
            host = getattr(ui, f"{name}_host")
            btn = ui.root.select_one({"name": f"blup-collapse-btn-{name}"})
            title = ui.root.select_one({"name": f"blup-panel-title-{name}"})

            collapse_arrow, expand_arrow = collapse_arrows(ps.side)

            panel.width = (
                self.runtime.dims.panel_rail_width
                if ps.collapsed
                else (ps.width or panel.width)
            )
            host.visible = not ps.collapsed
            title.visible = not ps.collapsed                                # type: ignore[attr-defined]
            btn.text = expand_arrow if ps.collapsed else collapse_arrow     # type: ignore[attr-defined]


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
                main = PanelState(
                    active_module   = "time_profile",
                    context_key     = "main",
                    side            = "center",
                ),
                context = PanelState(
                    active_module   = "context_selection",
                    context_key     = "selection",
                    side            = "left",
                    collapsed       = False,
                    width           = 280,
                ),
                inspector = PanelState(
                    active_module   = "token_detail",
                    context_key     = "detail",
                    side            = "right",
                    collapsed       = True,
                    width           = 320,
                )
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
        changed_branches = self.state_manager.update(
            display = display,
            context = context,
            modules = modules,
        )
        if not changed_branches:
            return

        if self.state_manager.layout_only_refresh():
            self.sync_panel_layout()
        else:
            self.schedule_refresh()

    # -------------------------------------------
    # |          Intent Bus Adapters            |
    # -------------------------------------------

    def _on_panel_intent(self, attr, old, new) -> None:
        print(f"[intent] raw: {new}")
        panel, action, width = (
                new["panel"][0], new["action"][0], new["width"][0]
        )

        if not panel:
            return
        match action:
            case "toggle":
                ps = getattr(self.state.display, panel)
                self.update_state(
                    display = self._build_display_patch(
                        panel, 
                        PanelPatch(
                            collapsed=not ps.collapsed
                        )
                    )
                )
            case "uncollapse":
                self.update_state(
                    display = self._build_display_patch(
                        panel,
                        PanelPatch(
                            collapsed=False,
                            width=width,
                        )
                    )
                )
            case "resize":
                if not getattr(self.state.display, panel).collapsed:
                    self.update_state(
                        display = self._build_display_patch(
                            panel,
                            PanelPatch(
                                width=width
                            )
                        )
                    )

    def _build_display_patch(self, panel: str, pp: PanelPatch) -> DisplayPatch:
        match panel:
            case "context":     return DisplayPatch(context=pp)
            case "inspector":   return DisplayPatch(inspector=pp)
            case _:             raise ValueError(
                f"unknown panel intent target{panel!r}"
            )

    # -------------------------------------------
    # |         Public Class Utilities          |
    # -------------------------------------------

    def get_module_pipeline(self, module_id: ModuleID) -> Pipeline:
        return self.module_pipelines[module_id]

    def get_work_pipeline(self, module_id: ModuleID) -> WorkPipeline:
        return self.work_pipelines[module_id]

    def active_module_ids(self) -> tuple[ModuleID, ...]:
        ids: list[ModuleID] = []

        for panel in (
            self.state.display.main,
            self.state.display.context,
            self.state.display.inspector,
        ):
            if panel.active_module is not None:
                ids.append(panel.active_module)

        return tuple(dict.fromkeys(ids))

    def active_pipelines(self) -> tuple[Pipeline, ...]:
        return tuple(
            self.get_module_pipeline(id)
            for id in self.active_module_ids()
        )

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

    # -------------------------------------------
    # |         Private Class Utilities         |
    # -------------------------------------------

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


