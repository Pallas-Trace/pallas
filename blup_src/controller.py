from __future__ import annotations

import re
from dataclasses import dataclass

from bokeh.io import curdoc
from bokeh.models.layouts import LayoutDOM

from colors import TokenColor
from modules.time_profile.pipeline import TimeProfilePipeline
from ui import UIElements, UIModel
from data_model import FidelityMode, TokenMode, CATEGORY_TOKEN_TYPE
from state import (
    TraceMode,
    ModuleID,
    TimeProfileOrder,
    PanelState,
    DisplayState,
    ContextState,
    ModuleState,
    AppState,
)
from trace_session import TraceSession
from module_interface import Pipeline
from module_interface import WorkManager
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


@dataclass(frozen=True)
class LoadedTrace:
    trace_id: str
    path: str
    label: str
    session: TraceSession


@dataclass
class ControllerRuntime:
    loaded_traces:      dict[str, LoadedTrace]
    trace_order:        list[str]
    ui:                 UIElements | None = None
    root:               LayoutDOM | None = None


class AppController:
    state:              AppState
    ui_model:           UIModel
    runtime:            ControllerRuntime

    def __init__(self, loaded_traces: list[LoadedTrace]):
        self.doc = curdoc()

        # runtime setup
        self.runtime = ControllerRuntime(
            loaded_traces = {t.trace_id: t for t in loaded_traces},
            trace_order   = [t.trace_id for t in loaded_traces]
        )

        # initial state setup
        self._install_merged_category_namespace()
        self.state = self.initial_state()

        self.ui_model = UIModel(self)

        self.token_color = TokenColor()
        self._register_loaded_trace_tokens()

        self.module_pipelines: dict[ModuleID, Pipeline] = {
            "time_profile": TimeProfilePipeline(width=1650, height=950)
        }

        self.work_manager = WorkManager(
            schedule_display_callback = self.doc.add_next_tick_callback,    # type: ignore
            max_workers = 8,
        )

        # internal logic flags
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

    def get_module_pipeline(self, module_id: ModuleID) -> Pipeline:
        return self.module_pipelines[module_id]

    def active_module_ids(self) -> tuple[ModuleID, ...]:
        ids = [self.state.display.center.active_module]
        if self.state.display.left is not None:
            ids.append(self.state.display.left.active_module)
        if self.state.display.right is not None:
            ids.append(self.state.display.right.active_module)
        return tuple(dict.fromkeys(ids))  # type: ignore

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

    def initial_state(self) -> AppState:
        trace_ids = self.all_trace_ids()
        n_loaded = len(trace_ids)
        if n_loaded == 0:
            raise ValueError("AppController requires at least one trace")

        primary_trace_id = trace_ids[0]
        secondary_trace_id = trace_ids[1] if n_loaded >= 2 else None
        trace_mode: TraceMode = "dual" if n_loaded >= 2 else "single"

        names = tuple(
            self._thread_names_for_trace_selection(
                trace_mode          = trace_mode,
                primary_trace_id    = primary_trace_id,
                secondary_trace_id  = secondary_trace_id,
            )
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
                active_threads      = names,
                trace_mode          = "single",
                primary_trace_id    = primary_trace_id,
                secondary_trace_id  = secondary_trace_id,
            ),
            modules = ModuleState(),
        )

    def set_trace_mode(self, mode: TraceMode) -> None:
        if self.state.context.trace_mode == mode:
            return
        self.state.context.trace_mode = mode
        self.normalize_trace_selection()
        self.sync_ui_controls()
        self.schedule_refresh()

    def set_primary_trace(self, trace_id: str) -> None:
        if self.state.context.primary_trace_id == trace_id:
            return
        self.state.context.primary_trace_id = trace_id
        self.normalize_trace_selection()
        self.sync_ui_controls()
        self.schedule_refresh()

    def set_secondary_trace(self, trace_id: str) -> None:
        if self.state.context.secondary_trace_id == trace_id:
            return
        self.state.context.secondary_trace_id = trace_id
        self.normalize_trace_selection()
        self.sync_ui_controls()
        self.schedule_refresh()

    # here for now pending refactor/cleanup
    def normalize_trace_selection(self) -> None:
        ctx = self.state.context
        trace_ids = self.all_trace_ids()

        if not trace_ids:
            raise ValueError("No loaded traces")

        if not self.has_trace(ctx.primary_trace_id):
            ctx.primary_trace_id = trace_ids[0]

        if ctx.trace_mode == "single":
            ctx.secondary_trace_id = None
        else:
            if (
                ctx.secondary_trace_id is None
                or not self.has_trace(ctx.secondary_trace_id)
                or ctx.secondary_trace_id == ctx.primary_trace_id
            ):
                for trace_id in trace_ids:
                    if trace_id != ctx.primary_trace_id:
                        ctx.secondary_trace_id = trace_id
                        break
                else:
                    ctx.secondary_trace_id = None
                    ctx.trace_mode = "single"

        available = tuple(self.all_thread_names())
        kept = tuple(t for t in ctx.active_threads if t in set(available))
        ctx.active_threads = kept or available
        ctx.selection.token = None
        ctx.time_scope.t0_ns = None
        ctx.time_scope.t1_ns = None

    def sync_ui_controls(self) -> None:
        ui = self.runtime.ui
        if ui is None:
            return

        trace_options = self.loaded_trace_options()
        all_threads = self.all_thread_names()

        ui.trace_mode_select.value = self.state.context.trace_mode
        ui.primary_trace_select.options = trace_options  # type: ignore
        ui.primary_trace_select.value = self.state.context.primary_trace_id or ""

        ui.secondary_trace_select.options = trace_options  # type: ignore
        ui.secondary_trace_select.disabled = (self.state.context.trace_mode != "dual")
        ui.secondary_trace_select.value = self.state.context.secondary_trace_id or ""

        ui.thread_select.options = [(name, name) for name in all_threads]
        ui.thread_select.value = list(self.state.context.active_threads)

    def set_active_threads(self, thread_names: tuple[str, ...]) -> None:
        chosen_threads = thread_names or tuple(self.all_thread_names())
        if self.state.context.active_threads == chosen_threads:
            return
        self.state.context.active_threads = chosen_threads
        self.schedule_refresh()

    def set_n_quanta(self, n: int) -> None:
        if self.state.modules.time_profile.n_bins == n:
            return
        self.state.modules.time_profile.n_bins = n
        self.schedule_refresh()

    def set_token_mode(self, mode: TokenMode) -> None:
        if self.state.context.token_mode == mode:
            return
        self.state.context.token_mode = mode
        self.schedule_refresh()

    def set_time_profile_mode(self, mode: FidelityMode) -> None:
        if self.state.modules.time_profile.fidelity == mode:
            return
        self.state.modules.time_profile.fidelity = mode
        self.schedule_refresh()

    def set_time_profile_order(self, order: TimeProfileOrder) -> None:
        if self.state.modules.time_profile.order == order:
            return
        self.state.modules.time_profile.order = order
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

    def on_time_profile_mode_changed(self, attr, old, new):
        self.set_time_profile_mode(new)

    def on_time_profile_order_changed(self, attr, old, new):
        self.set_time_profile_order(new)

    def on_trace_mode_changed(self, attr, old, new):
        self.set_trace_mode(new)

    def on_primary_trace_changed(self, attr, old, new):
        self.set_primary_trace(new)

    def on_secondary_trace_changed(self, attr, old, new):
        self.set_secondary_trace(new)

    def on_highlight_token_changed(self, attr, old, new):
        pass
        # summary = self.get_module_pipeline("summary")
        # assert isinstance(summary, SummaryPipeline)
        # summary.on_highlight_widget_changed(self, new)

    # -------------------------------------------
    # |               Utilities                 |
    # -------------------------------------------

    def all_trace_ids(self) -> list[str]:
        return list(self.runtime.trace_order)

    def get_trace(self, trace_id: str) -> LoadedTrace:
        return self.runtime.loaded_traces[trace_id]

    def has_trace(self, trace_id: str | None) -> bool:
        return trace_id is not None and trace_id in self.runtime.loaded_traces

    def loaded_trace_options(self) -> list[tuple[str, str]]:
        return [
            (trace_id, self.runtime.loaded_traces[trace_id].label)
            for trace_id in self.runtime.trace_order
        ]

    def iter_loaded_traces(self) -> tuple[LoadedTrace, ...]:
        return tuple(
            self.runtime.loaded_traces[trace_id]
            for trace_id in self.runtime.trace_order
        )

    def get_primary_trace(self) -> LoadedTrace | None:
        trace_id = self.state.context.primary_trace_id
        assert(trace_id is not None)
        if not self.has_trace(trace_id):
            return None
        return self.get_trace(trace_id)

    def get_secondary_trace(self) -> LoadedTrace | None:
        if self.state.context.trace_mode != "dual":
            return None
        trace_id = self.state.context.secondary_trace_id
        assert(trace_id is not None)
        if not self.has_trace(trace_id):
            return None
        if trace_id == self.state.context.primary_trace_id:
            return None
        return self.get_trace(trace_id)

    def get_primary_session(self) -> TraceSession | None:
        loaded = self.get_primary_trace()
        return None if loaded is None else loaded.session

    def get_secondary_session(self) -> TraceSession | None:
        loaded = self.get_secondary_trace()
        return None if loaded is None else loaded.session

    def _install_merged_category_namespace(self) -> None:
        names = sorted({
            str(name)
            for loaded in self.iter_loaded_traces()
            for name in loaded.session.meta.cat_key_to_name.values()
        })

        name_to_cat_token = {
            name: (CATEGORY_TOKEN_TYPE, int(i))
            for i, name in enumerate(names)
        }

        for loaded in self.iter_loaded_traces():
            loaded.session.install_category_namespace(name_to_cat_token)

    def _register_loaded_trace_tokens(self) -> None:
        for loaded in self.iter_loaded_traces():
            token_name_by_key = dict(loaded.session.meta.token_key_to_name)
            self.token_color.register_tokens(
                token_name_by_key.keys(),
                token_names = token_name_by_key,
                namespace   = loaded.trace_id
            )

    def _thread_names_for_trace_selection(
        self,
        *,
        trace_mode: TraceMode,
        primary_trace_id: str | None,
        secondary_trace_id: str | None,
    ) -> list[str]:
        trace_ids: list[str] = []

        if (primary_trace_id is not None 
                and primary_trace_id in self.runtime.loaded_traces):
            trace_ids.append(primary_trace_id)

        if (
            trace_mode == "dual"
            and secondary_trace_id is not None
            and secondary_trace_id in self.runtime.loaded_traces
            and secondary_trace_id != primary_trace_id
        ):
            trace_ids.append(secondary_trace_id)

        names: set[str] = set()
        for trace_id in trace_ids:
            loaded = self.runtime.loaded_traces[trace_id]
            names |= set(map(str, loaded.session.meta.thread_names))

        return sorted(names, key=thread_sort_key)

    def all_thread_names(self) -> list[str]:
        ctx = self.state.context
        return self._thread_names_for_trace_selection(
            trace_mode=ctx.trace_mode,
            primary_trace_id=ctx.primary_trace_id,
            secondary_trace_id=ctx.secondary_trace_id,
        )

    def full_time_bounds(self) -> tuple[int, int]:
        t1 = self.get_primary_session()
        if t1 is None:
            raise RuntimeError("primary trace is not available")

        t2 = self.get_secondary_session()
        if t2 is None:
            return (int(t1.meta.start_ns), int(t1.meta.end_ns))

        return (
            min(int(t1.meta.start_ns), int(t2.meta.start_ns)),
            max(int(t1.meta.end_ns), int(t2.meta.end_ns)),
        )


