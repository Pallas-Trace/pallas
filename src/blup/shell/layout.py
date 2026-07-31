from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShellDimensions:
    context_width:          int = 280
    context_min_width:      int = 220

    inspector_width:        int = 320
    inspector_min_width:    int = 260

    header_height:          int = 52
    footer_height:          int = 26

    panel_gap:              int = 8
    panel_padding:          int = 12


