from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Final, Literal, TypeAlias, TYPE_CHECKING, TypeGuard

from numpy import trace

from blup.data_model import FidelityMode, TokenMode
from blup.types import TraceID

if TYPE_CHECKING:
    from blup.traces.interface import TraceRegistryAccess


# -------------------------------------------
# |           State Constants               |
# -------------------------------------------


TraceMode:                  TypeAlias = Literal[
            "dual",
            "single",
]
PanelID:                    TypeAlias = Literal[
            "center",
            "left",
            "right",
]

ModuleID:                   TypeAlias = Literal[
            "time_profile",
            "token_detail",
            "inspector",
]

TimeProfileOrder:           TypeAlias = Literal[
            "global",
            "local",
]
TimeProfilePresentation:    TypeAlias = Literal[
            "auto",
            "binned",
            "flame",
]

TokenDetailChartMode:       TypeAlias = Literal[
            "histogram",
            "scatter",
]
TokenDetailTableMode:       TypeAlias = Literal[
            "compare",
            "summary",
]


# -------------------------------------------
# |              State Tree                 |
# -------------------------------------------

#   AppState
#      ├── display
#      │   ├── center
#      │   │   ├── active_module
#      │   │   └── context_key
#      │   ├── left
#      │   │   ├── active_module
#      │   │   └── context_key
#      │   └── right
#      │       ├── active_module
#      │       └── context_key
#      ├── context
#      │   ├── active_threads
#      │   ├── trace_mode
#      │   ├── primary_trace_id
#      │   ├── secondary_trace_id
#      │   ├── token_mode
#      │   ├── selection
#      │   │   └── token
#      │   └── time_scope
#      │       ├── t0_ns
#      │       └── t1_ns
#      └── modules
#          ├── time_profile
#          │   ├── presentation
#          │   ├── n_bins
#          │   ├── fidelity
#          │   └── order
#          ├── token_detail
#          │   ├── chart_mode
#          │   ├── table_mode
#          │   ├── show_stats
#          │   └── show_chart
#          └── inspector

# display layout state
# --------------------

@dataclass(frozen=True)
class PanelState:
    active_module:      ModuleID
    context_key:        str | None = None

# >>>

@dataclass(frozen=True)
class DisplayState:
    center:             PanelState
    left:               PanelState | None = None
    right:              PanelState | None = None

# global context state
# --------------------

@dataclass(frozen=True)
class TraceSelectionState:
    trace_ids:          tuple[TraceID, ...] = ()
    focus_id:           str | None = None

@dataclass(frozen=True)
class TokenSelectionState:
    token:              tuple[int, int] | None = None

@dataclass(frozen=True)
class TimeScopeState:
    t0_ns:              int | None = None
    t1_ns:              int | None = None

# >>>

@dataclass(frozen=True)
class ContextState:
    traces:             TraceSelectionState = field(
        default_factory=TraceSelectionState
    )
    active_threads:     tuple[str, ...] = ()
    token_mode:         TokenMode = "raw"
    selection:          TokenSelectionState = field(
        default_factory=TokenSelectionState
    )
    time_scope:         TimeScopeState = field(
        default_factory=TimeScopeState
    )

# module specific state
# ---------------------

@dataclass(frozen=True)
class TimeProfileState:
    presentation:       TimeProfilePresentation = "binned"
    n_bins:             int = 100
    fidelity:           FidelityMode = "fast"
    order:              TimeProfileOrder = "global"

@dataclass(frozen=True)
class TokenDetailState:
    chart_mode:         TokenDetailChartMode = "histogram"
    table_mode:         TokenDetailTableMode = "compare"
    show_stats:         bool = True
    show_chart:         bool = True

@dataclass(frozen=True)
class InspectorState:
    pass

# >>>

@dataclass(frozen=True)
class ModuleState:
    time_profile:       TimeProfileState = field(
        default_factory=TimeProfileState
    )
    token_detail:       TokenDetailState = field(
        default_factory=TokenDetailState
    )
    inspector:          InspectorState = field(
        default_factory=InspectorState
    )

# state tree root
# ---------------

@dataclass(frozen=True)
class AppState:
    display:            DisplayState
    context:            ContextState
    modules:            ModuleState = field(default_factory=ModuleState)

# -------------------------------------------
# |             State Patches               |
# -------------------------------------------

# patch typing helpers
class _Unset:
    """Sentinel"""

UNSET: Final = _Unset()
type PatchValue[T] = T | _Unset

def is_set[T](value: PatchValue[T]) -> TypeGuard[T]:
    return not isinstance(value, _Unset)

# ----------

@dataclass(frozen=True)
class PanelPatch:
    active_module:      PatchValue[ModuleID] = UNSET
    context_key:        PatchValue[str | None] = UNSET

    def apply(self, state: PanelState) -> PanelState:
        active_module = state.active_module
        if is_set(self.active_module):
            active_module = self.active_module

        context_key = state.context_key
        if is_set(self.context_key):
            context_key = self.context_key

        return replace(
            state,
            active_module   = active_module,
            context_key     = context_key,
        )


type SidePanelUpdate = PanelPatch | PanelState | None

@dataclass(frozen=True)
class DisplayPatch:
    center:             PatchValue[PanelPatch] = UNSET
    left:               PatchValue[SidePanelUpdate] = UNSET
    right:              PatchValue[SidePanelUpdate] = UNSET

    def apply(self, state: DisplayState) -> DisplayState:
        center = state.center
        if is_set(self.center):
            center = self.center.apply(center)

        return replace(
            state,
            center  = center,
            left    = self._apply_side(state.left, self.left),
            right   = self._apply_side(state.right, self.right),
        )

    @staticmethod
    def _apply_side(
        state: PanelState | None,
        update: PatchValue[SidePanelUpdate],
    ) -> PanelState | None:
        if not is_set(update):
            return state

        if update is None:
            return None

        if isinstance(update, PanelState):
            return update

        if state is None:
            raise ValueError(
                "Cannot apply PanelPatch to a missing side panel; "
                "call with PanelState(...) first to create it!"
            )

        return update.apply(state)


@dataclass(frozen=True)
class TraceSelectionPatch:
    trace_ids:          PatchValue[tuple[TraceID, ...]] = UNSET
    focus_id:           PatchValue[TraceID | None] = UNSET

    def apply(self, state: TraceSelectionState) -> TraceSelectionState:
        trace_ids = state.trace_ids
        if is_set(self.trace_ids):
            trace_ids = self.trace_ids

        focus_id = state.focus_id
        if is_set(self.focus_id):
            focus_id = self.focus_id

        return replace(
            state,
            trace_ids   = trace_ids,
            focus_id    = focus_id,
        )


@dataclass(frozen=True)
class TokenSelectionPatch:
    token:              PatchValue[tuple[int, int] | None] = UNSET

    def apply(self, state: TokenSelectionState) -> TokenSelectionState:
        token = state.token
        if is_set(self.token):
            token = self.token

        return replace(state, token=token)


@dataclass(frozen=True)
class TimeScopePatch:
    t0_ns:              PatchValue[int | None] = UNSET
    t1_ns:              PatchValue[int | None] = UNSET

    def apply(self, state: TimeScopeState) -> TimeScopeState:
        t0_ns = state.t0_ns
        if is_set(self.t0_ns):
            t0_ns = self.t0_ns

        t1_ns = state.t1_ns
        if is_set(self.t1_ns):
            t1_ns = self.t1_ns

        return replace(
            state,
            t0_ns   = t0_ns,
            t1_ns   = t1_ns,
        )


@dataclass(frozen=True)
class ContextPatch:
    traces:             PatchValue[TraceSelectionPatch] = UNSET
    active_threads:     PatchValue[tuple[str, ...]] = UNSET
    token_mode:         PatchValue[TokenMode] = UNSET
    selection:          PatchValue[TokenSelectionPatch] = UNSET
    time_scope:         PatchValue[TimeScopePatch] = UNSET

    def apply(self, state: ContextState) -> ContextState:
        traces = state.traces
        if is_set(self.traces):
            traces = self.traces.apply(traces)

        active_threads = state.active_threads
        if is_set(self.active_threads):
            active_threads = self.active_threads

        token_mode = state.token_mode
        if is_set(self.token_mode):
            token_mode = self.token_mode

        selection = state.selection
        if is_set(self.selection):
            selection = self.selection.apply(selection)

        time_scope = state.time_scope
        if is_set(self.time_scope):
            time_scope = self.time_scope.apply(time_scope)

        return replace(
            state,
            traces          = traces,
            active_threads  = active_threads,
            token_mode      = token_mode,
            selection       = selection,
            time_scope      = time_scope,
        )


@dataclass(frozen=True)
class TimeProfilePatch:
    presentation:       PatchValue[TimeProfilePresentation] = UNSET
    n_bins:             PatchValue[int] = UNSET
    fidelity:           PatchValue[FidelityMode] = UNSET
    order:              PatchValue[TimeProfileOrder] = UNSET

    def apply(self, state: TimeProfileState) -> TimeProfileState:
        presentation = state.presentation
        if is_set(self.presentation):
            presentation = self.presentation

        n_bins = state.n_bins
        if is_set(self.n_bins):
            n_bins = self.n_bins

        if n_bins < 1:
            raise ValueError(
                "time_profile.n_bins must >= 1"
            )

        fidelity = state.fidelity
        if is_set(self.fidelity):
            fidelity = self.fidelity

        order = state.order
        if is_set(self.order):
            order = self.order

        return replace(
            state,
            presentation    = presentation,
            n_bins          = n_bins,
            fidelity        = fidelity,
            order           = order,
        )


@dataclass(frozen=True)
class TokenDetailPatch:
    chart_mode:         PatchValue[TokenDetailChartMode] = UNSET
    table_mode:         PatchValue[TokenDetailTableMode] = UNSET
    show_stats:         PatchValue[bool] = UNSET
    show_chart:         PatchValue[bool] = UNSET

    def apply(self, state: TokenDetailState) -> TokenDetailState:
        chart_mode = state.chart_mode
        if is_set(self.chart_mode):
            chart_mode = self.chart_mode

        table_mode = state.table_mode
        if is_set(self.table_mode):
            table_mode = self.table_mode

        show_stats = state.show_stats
        if is_set(self.show_stats):
            show_stats = self.show_stats

        show_chart = state.show_chart
        if is_set(self.show_chart):
            show_chart = self.show_chart

        return replace(
            state,
            chart_mode  = chart_mode,
            table_mode  = table_mode,
            show_stats  = show_stats,
            show_chart  = show_chart,
        )


@dataclass(frozen=True)
class InspectorPatch:
    def apply(self, state: InspectorState) -> InspectorState:
        return state


@dataclass(frozen=True)
class ModulePatch:
    time_profile:       PatchValue[TimeProfilePatch] = UNSET
    token_detail:       PatchValue[TokenDetailPatch] = UNSET
    inspector:          PatchValue[InspectorPatch] = UNSET

    def apply(self, state: ModuleState) -> ModuleState:
        time_profile = state.time_profile
        if is_set(self.time_profile):
            time_profile = self.time_profile.apply(time_profile)

        token_detail = state.token_detail
        if is_set(self.token_detail):
            token_detail = self.token_detail.apply(token_detail)

        inspector = state.inspector
        if is_set(self.inspector):
            inspector = self.inspector.apply(inspector)

        return replace(
            state,
            time_profile    = time_profile,
            token_detail    = token_detail,
            inspector       = inspector,
        )

# -------------------------------------------
# |             State Manager               |
# -------------------------------------------

class StateManager:

    def __init__(
        self,
        trace_registry: "TraceRegistryAccess",
        initial_state: AppState,
    ) -> None:
        self._registry = trace_registry
        self._state = self._normalize(initial_state)

    @property
    def state(self) -> AppState:
        return self._state

    def update(
        self,
        *,
        display: DisplayPatch | None = None,
        context: ContextPatch | None = None,
        modules: ModulePatch | None = None,
    ) -> bool:
        """Apply provided state-tree patches and validate"""

        previous = self._state
        candidate = previous

        # apply provided patches to state-tree branches
        if display is not None:
            candidate = replace(
                candidate,
                display = display.apply(candidate.display),
            )
        if context is not None:
            candidate = replace(
                candidate,
                context = context.apply(candidate.context),
            )
        if modules is not None:
            candidate = replace(
                candidate,
                modules = modules.apply(candidate.modules),
            )

        candidate = self._normalize(candidate)

        # if trace selection changed reset relevant context
        trace_ids_changed = (
            candidate.context.traces.trace_ids
            != previous.context.traces.trace_ids
        )
        if trace_ids_changed:
            candidate = replace(
                candidate,
                context = replace(
                    candidate.context,
                    selection   = TokenSelectionState(),
                    time_scope  = TimeScopeState(),
                ),
            )

        if candidate == previous:
            # no actual update occured
            return False

        # commit state update
        self._state = candidate
        return True

    def _normalize(self, state: AppState) -> AppState:
        """Repair modified state-tree to closest valid state"""

        context = state.context
        trace_selection = context.traces

        # ----------

        # validate against loaded traces in registry
        trace_ids = self._registry.valid_trace_ids(
            trace_selection.trace_ids,
        )
        if not trace_ids:
            # fallback: select first available trace
            available = self._registry.all_trace_ids()
            if not available:
                raise RuntimeError(
                    "At least one trace must be loaded"
                )
            trace_ids = available[:1]

        # check that focused trace is loaded
        focus_id = trace_selection.focus_id
        if focus_id not in trace_ids:
            # fallback: focus first selected trace
            focus_id = trace_ids[0] if trace_ids else None

        traces = TraceSelectionState(
            trace_ids   = trace_ids,
            focus_id    = focus_id,
        )

        # ----------

        available_threads = self._registry.thread_names_for(trace_ids)
        available_thread_set = set(available_threads)

        # retain only valid active threads in selection
        active_threads = tuple(
            thread_name
            for thread_name in context.active_threads
            if thread_name in available_thread_set
        )
        if not active_threads:
            # fallback: select all available threads
            active_threads = available_threads

        # ----------

        return replace(
            state,
            context = replace(
                context,
                traces          = traces,
                active_threads  = active_threads,
            ),
        )


