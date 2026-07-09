from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal, TypeAlias


# > state constants

from data_model import FidelityMode, TokenMode
QuantaOrder:    TypeAlias = Literal["global", "local"]
ViewId:         TypeAlias = Literal["quanta", "inspector", "summary"]

# > state tree:

#   AppState
#      ├── views
#      │   └── quanta
#      │       ├── n_bins
#      │       ├── mode
#      │       └── order
#      ├── context
#      │   ├── active_threads
#      │   ├── token_mode
#      │   ├── selection
#      │   │   └── token
#      │   └── time_scope
#      │       ├── t0_ns
#      │       └── t1_ns
#      └── display
#          ├── primary
#          │   ├── active_view
#          │   └── context_key
#          └── secondary[*]
#              ├── active_view
#              └── context_key

# >>> display layout state

@dataclass
class PanelState:
    active_view:        ViewId
    context_key:        str | None = None

@dataclass
class DisplayState:
    primary:            PanelState
    secondary:          tuple[PanelState, ...] = ()

# >>> global analysis context state

@dataclass
class SelectionState:
    token:              tuple[int, int] | None = None

@dataclass
class TimeScopeState:
    t0_ns:              int | None = None
    t1_ns:              int | None = None

@dataclass
class ContextState:
    active_threads:     tuple[str, ...]
    token_mode:         TokenMode = "raw"
    selection:          SelectionState = field(default_factory=SelectionState)
    time_scope:         TimeScopeState = field(default_factory=TimeScopeState)

# >>> view specific state

@dataclass
class QuantaState:
    n_bins:             int = 100
    mode:               FidelityMode = "fast"
    order:              QuantaOrder = "global"

@dataclass
class ViewState:
    quanta:             QuantaState = field(default_factory=QuantaState)

# >>> state root

@dataclass
class AppState:
    display:            DisplayState
    context:            ContextState
    views:              ViewState = field(default_factory=ViewState)


