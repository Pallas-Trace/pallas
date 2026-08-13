from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShellDimensions:
    context_width:          int = 230
    context_min_width:      int = 0

    inspector_width:        int = 400
    inspector_min_width:    int = 0

    header_height:          int = 44
    footer_height:          int = 26

    panel_gap:              int = 0
    panel_padding:          int = 12
    panel_rail_width:       int = 28

    splitter_width:         int = 4



