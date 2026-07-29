from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from data_model import FidelityMode, TokenMode


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

@dataclass
class PanelState:
    active_module:      ModuleID
    context_key:        str | None = None

# >>>

@dataclass
class DisplayState:
    center:             PanelState
    left:               PanelState | None = None
    right:              PanelState | None = None

# global context state
# --------------------

@dataclass
class SelectionState:
    token:              tuple[int, int] | None = None

@dataclass
class TimeScopeState:
    t0_ns:              int | None = None
    t1_ns:              int | None = None

# >>>

@dataclass
class ContextState:
    active_threads:     tuple[str, ...]
    trace_mode:         TraceMode = "single"
    primary_trace_id:   str | None = None
    secondary_trace_id: str | None = None
    token_mode:         TokenMode = "raw"
    selection:          SelectionState = field(
        default_factory=SelectionState
    )
    time_scope:         TimeScopeState = field(
        default_factory=TimeScopeState
    )

# module specific state
# ---------------------

@dataclass
class TimeProfileState:
    presentation:       TimeProfilePresentation = "binned"
    n_bins:             int = 100
    fidelity:           FidelityMode = "fast"
    order:              TimeProfileOrder = "global"

@dataclass
class TokenDetailState:
    chart_mode:         TokenDetailChartMode = "histogram"
    table_mode:         TokenDetailTableMode = "compare"
    show_stats:         bool = True
    show_chart:         bool = True

@dataclass
class InspectorState:
    pass

# >>>

@dataclass
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

@dataclass
class AppState:
    display:            DisplayState
    context:            ContextState
    modules:            ModuleState = field(default_factory=ModuleState)


