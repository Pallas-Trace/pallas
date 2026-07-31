from __future__ import annotations

from dataclasses import dataclass

from bokeh.themes import Theme


@dataclass(frozen=True)
class Palette:
    bg0:        str = "#1d2021"
    bg1:        str = "#282828"
    bg2:        str = "#3c3836"
    bg3:        str = "#504945"

    fg0:        str = "#fbf1c7"
    fg1:        str = "#ebdbb2"
    fg2:        str = "#d5c4a1"
    muted:      str = "#a89984"

    yellow:     str = "#d79921"
    orange:     str = "#d65d0e"
    red:        str = "#cc241d"
    green:      str = "#98971a"
    blue:       str = "#458588"
    purple:     str = "#b16286"

PALETTE = Palette()


def gruvbox_bokeh_theme() -> Theme:
    p = PALETTE

    return Theme(json={
        "attrs": {
            "Plot": {
                "background_fill_color":    p.bg0,
                "border_fill_color":        p.bg0,
                "outline_line_color":       p.bg3,
            },
            "Title": {
                "text_color":               p.fg1,
                "text_font":                "monospace",
                "text_font_style":          "normal",
            },
            "Axis": {
                "axis_line_color":          p.bg3,
                "major_tick_line_color":    p.bg3,
                "minor_tick_line_color":    None,
                "major_label_text_color":   p.muted,
                "major_label_text_font":    "monospace",
                "axis_label_text_color":    p.fg2,
                "axis_label_text_font":     "monospace",
            },
            "Grid": {
                "grid_line_color":          p.bg3,
                "grid_line_alpha":          0.35,
                "minor_grid_line_color":    None,
            },
            "Legend": {
                "background_fill_color":    p.bg1,
                "border_line_color":        p.bg3,
                "label_text_color":         p.fg2,
                "label_text_font":          "monospace",
            },
        },
    })


