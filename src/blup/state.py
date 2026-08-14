from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Final, Literal, TypeAlias, TYPE_CHECKING, TypeGuard

from bokeh.models.glyphs import Patch
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
            "main",
            "context",
            "inspector",
]
PanelSide:                  TypeAlias = Literal[
            "left",
            "right",
            "top",
            "bottom",
            "center",
]

ModuleID:                   TypeAlias = Literal[
            "time_profile",
            "token_detail",
            "context_selection",
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
            "overlay",
            "delta",
]
TokenDetailTableMode:       TypeAlias = Literal[
            "full",
            "compact",
]


# -------------------------------------------
# |              State Tree                 |
# -------------------------------------------

# display layout state
# --------------------

@dataclass(frozen=True)
class PanelState:
    active_module:      ModuleID | None = None
    context_key:        str | None = None
    side:               PanelSide = "center"
    width:              int | None = None
    collapsed:          bool = False

@dataclass(frozen=True)
class HeaderState:
    # TODO: flesh out state fields later
    status:             str = "READY"

@dataclass(frozen=True)
class FooterState:
    # TODO: flesh out state fields later
    status:             str = "READY"
    pending_jobs:       int = 0

# >>> branch root

@dataclass(frozen=True)
class DisplayState:
    main:               PanelState = field(
        default_factory=lambda: PanelState(
            active_module   = "time_profile",
            context_key     = "main",
            side            = "center",
        )
    )
    context:            PanelState = field(
        default_factory=lambda: PanelState(
            active_module   = "context_selection",
            context_key     = "selection",
            side            = "left",
            collapsed       = False,
            width           = 280,
        )
    )
    inspector:          PanelState = field(
        default_factory=lambda: PanelState(
            active_module   = "token_detail",
            context_key     = "detail",
            side            = "right",
            collapsed       = True,
            width           = 320,
        )
    )

    header:             HeaderState = field(
        default_factory=HeaderState
    )
    footer:             FooterState = field(
        default_factory=FooterState
    )

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

# >>> branch root

@dataclass(frozen=True)
class ContextState:
    traces:             TraceSelectionState = field(
        default_factory=TraceSelectionState
    )
    active_threads:     tuple[str, ...] = ()
    token_mode:         TokenMode = "named"
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
    n_bins:             int = 50
    fidelity:           FidelityMode = "balanced"
    order:              TimeProfileOrder = "global"

@dataclass(frozen=True)
class TokenDetailState:
    chart_mode:         TokenDetailChartMode = "overlay"
    table_mode:         TokenDetailTableMode = "compact"
    show_stats:         bool = True
    show_chart:         bool = True

@dataclass(frozen=True)
class InspectorState:
    pass

# >>> branch root

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

# state-tree root
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
    active_module:      PatchValue[ModuleID | None] = UNSET
    context_key:        PatchValue[str | None] = UNSET
    side:               PatchValue[PanelSide] = UNSET
    width:              PatchValue[int | None] = UNSET
    collapsed:          PatchValue[bool] = UNSET

    def apply(self, state: PanelState) -> PanelState:
        active_module = state.active_module
        if is_set(self.active_module):
            active_module = self.active_module

        context_key = state.context_key
        if is_set(self.context_key):
            context_key = self.context_key

        side = state.side
        if is_set(self.side):
            side = self.side

        width = state.width
        if is_set(self.width):
            width = self.width

        collapsed = state.collapsed
        if is_set(self.collapsed):
            collapsed = self.collapsed

        return replace(
            state,
            active_module   = active_module,
            context_key     = context_key,
            side            = side,
            width           = width,
            collapsed       = collapsed,
        )


@dataclass(frozen=True)
class HeaderPatch:
    status:             PatchValue[str] = UNSET

    def apply(self, state: HeaderState) -> HeaderState:
        status = state.status
        if is_set(self.status):
            status = self.status

        return replace(
            state,
            status  = status,
        )


@dataclass(frozen=True)
class FooterPatch:
    status:             PatchValue[str] = UNSET
    pending_jobs:       PatchValue[int] = UNSET

    def apply(self, state: FooterState) -> FooterState:
        status = state.status
        if is_set(self.status):
            status = self.status

        pending_jobs = state.pending_jobs
        if is_set(self.pending_jobs):
            pending_jobs = self.pending_jobs

        return replace(
            state,
            status          = status,
            pending_jobs    = pending_jobs,
        )


@dataclass(frozen=True)
class DisplayPatch:
    main:               PatchValue[PanelPatch] = UNSET
    context:            PatchValue[PanelPatch] = UNSET
    inspector:          PatchValue[PanelPatch] = UNSET

    header:             PatchValue[HeaderPatch] = UNSET
    footer:             PatchValue[FooterPatch] = UNSET

    def apply(self, state: DisplayState) -> DisplayState:
        main = state.main
        if is_set(self.main):
            main = self.main.apply(main)

        context = state.context
        if is_set(self.context):
            context = self.context.apply(context)

        inspector = state.inspector
        if is_set(self.inspector):
            inspector = self.inspector.apply(inspector)

        header = state.header
        if is_set(self.header):
            header = self.header.apply(header)

        footer = state.footer
        if is_set(self.footer):
            footer = self.footer.apply(footer)

        return replace(
            state,
            main        = main,
            context     = context,
            inspector   = inspector,
            header      = header,
            footer      = footer,
        )


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

_LAYOUT_PANELS = (
    "main",
    "context",
    "inspector",
)

def _without_layout(ps: PanelState) -> PanelState:
    return PanelState(
        active_module   = ps.active_module,
        context_key     = ps.context_key,
    )

def _strip_layout_state(state: AppState) -> AppState:
    return replace(
        state,
        display = replace(
            state.display,
            **{
                name: _without_layout(getattr(state.display, name))
                for name in _LAYOUT_PANELS
            },
        ),
    )


_NULL = object()

def _parse_state_path(state: AppState, path: str) -> object:
    obj = state
    for node in path.split("."):
        if obj is None:
            return _NULL
        obj = getattr(obj, node, _NULL)
    return obj


class StateManager:

    def __init__(
        self,
        trace_registry: "TraceRegistryAccess",
        initial_state: AppState,
    ) -> None:
        self._registry = trace_registry
        self._latest_state: AppState = self._normalize(initial_state)
        self._synced_state: AppState | None = None

    @property
    def state(self) -> AppState:
        return self._latest_state

    def refresh_needed(self, state_paths: tuple[str, ...]) -> bool:
        """
            Parses list of subsrcibed states to determine refresh status.
        """
        if self._synced_state is None:
            return True

        return any(
            _parse_state_path(self._latest_state, path) != _parse_state_path(self._synced_state, path)
            for path in state_paths
        )

    def layout_only_refresh(self) -> bool:
        """
            Returns true if the pending state concerns only app visual layout.
        """
        if self._synced_state is None:
            return False
        return (
            _strip_layout_state(self._latest_state)
            == _strip_layout_state(self._synced_state)
        )

    def mark_synced(self) -> None:
        """
            Advance synchronization mark after complete state propagation.
        """
        self._synced_state = self._latest_state

    def update(
        self,
        *,
        display: DisplayPatch | None = None,
        context: ContextPatch | None = None,
        modules: ModulePatch | None = None,
    ) -> tuple[str, ...]:
        """
            Apply provided state-tree patches and validate updated state.
            Returns list of altered top-level state branches.
        """

        previous = self._latest_state
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
            return ()

        # commit state update
        self._latest_state = candidate
        return tuple(
            branch
            for branch in ("display", "context", "modules")
            if getattr(candidate, branch) != getattr(previous, branch)
        )

    def _normalize(self, state: AppState) -> AppState:
        """
            Repair modified state-tree to closest valid state.
        """

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



