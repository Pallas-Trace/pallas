from __future__ import annotations

from dataclasses import dataclass

from bokeh.models.css import GlobalInlineStyleSheet, InlineStyleSheet
from bokeh.themes import Theme


@dataclass(frozen=True)
class Glyphs:
    arrow_left:     str = "◀"
    arrow_right:    str = "▶"
    arrow_up:       str = "▲"
    arrow_down:     str = "▼"

GLYPHS = Glyphs()


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


def make_global_stylesheet() -> GlobalInlineStyleSheet:
    return GlobalInlineStyleSheet(
        css = f"""
            html, body {{
                width: 100%;
                height: 100%;
                margin: 0;
                background: {PALETTE.bg0};
            }}

            .bk-root {{
                width: 100%;
                height: 100%;
                background: {PALETTE.bg0};
            }}

            .bk-tooltip {{
                background: {PALETTE.bg1} !important;
                border: 1px solid {PALETTE.bg3} !important;
                border-radius: 0 !important;
                color: {PALETTE.fg1} !important;
                box-shadow: none !important;
                padding: 0 !important;
            }}

            bk-Tooltip.bk-left::after, .bk-Tooltip.bk-right::after {{
                border-color: transparent {PALETTE.bg3} transparent transparent !important;
            }}

            .bk-Tooltip.bk-right::after {{
                border-color: transparent transparent transparent {PALETTE.bg3} !important;
            }}

            .blup-split:hover {{
                background: {PALETTE.yellow} !important;
            }}

            :root {{
            --bokeh-base-font: monospace;
            --bokeh-font-size: 11px;
            --bokeh-icon-color: {PALETTE.muted};
            --bokeh-border-color: {PALETTE.bg3};
            --bokeh-background-color: {PALETTE.bg0};
            --bokeh-hover-color: {PALETTE.bg1};
            --bokeh-color: {PALETTE.fg1};
            --bokeh-disabled-color: {PALETTE.muted};
            --bokeh-disabled-background-color: {PALETTE.bg1};
            --bokeh-input-focus-border-color: {PALETTE.yellow};
            --tooltip-color: {PALETTE.bg1};
            --tooltip-border: {PALETTE.bg3};
            --tooltip-text: {PALETTE.fg1};
            }}
        """
    )


def make_widget_stylesheet() -> InlineStyleSheet:
    return InlineStyleSheet(
        css = f"""
            :host {{
                background: {PALETTE.bg1};
                color: {PALETTE.fg1};
                font-family: monospace;
                font-size: 12px;
            }}

            /* --- Buttons (section headers, toggle buttons) --- */
            :host .bk-btn {{
                background: {PALETTE.bg2};
                color: {PALETTE.fg1};
                border: 1px solid {PALETTE.bg3};
                border-radius: 0;
                font-family: monospace;
                font-size: 12px;
                font-weight: bold;
                padding: 2px 8px;
            }}
            :host .bk-btn:hover {{
                background: {PALETTE.bg3};
                color: {PALETTE.yellow};
            }}

            /* --- Input groups (Select, MultiChoice, TextInput, etc.) --- */
            :host .bk-input-group label {{
                color: {PALETTE.fg2};
                font-family: monospace;
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            :host .bk-input,
            :host select {{
                background: {PALETTE.bg0};
                color: {PALETTE.fg1};
                border: 1px solid {PALETTE.bg3};
                border-radius: 0;
                font-family: monospace;
                font-size: 12px;
            }}
            :host .bk-input:focus,
            :host select:focus {{
                border-color: {PALETTE.blue};
                outline: none;
                box-shadow: none;
            }}

            /* --- MultiChoice (choices.js internals) --- */
            :host .choices__inner {{
                background: {PALETTE.bg0};
                color: {PALETTE.red};
                border: 1px solid {PALETTE.bg3};
                border-radius: 0;
                font-family: monospace;
                font-size: 12px;
            }}
            :host .choices__list--dropdown {{
                background: {PALETTE.bg1};
                border: 1px solid {PALETTE.bg3};
                border-radius: 0;
            }}
            :host .choices__item--choice {{
                color: {PALETTE.fg1};
            }}
            :host .choices__item--choice:hover {{
                background: {PALETTE.bg3};
            }}
            :host .choices__item--selectable {{
                background: {PALETTE.blue};
                color: {PALETTE.bg0};
                border-radius: 0;
                font-size: 11px;
            }}

            /* --- Checkbox / Radio groups --- */
            :host .bk-checkbox-group label,
            :host .bk-radio-group label {{
                color: {PALETTE.fg1};
                font-family: monospace;
                font-size: 12px;
            }}

            /* --- Collapsible section toggle --- */
            :host(.blup-section-toggle) .bk-btn {{
                background: transparent;
                color: {PALETTE.yellow};
                border: none;
                border-bottom: 1px solid {PALETTE.bg3};
                border-radius: 0;
                font-family: monospace;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-align: left;
                justify-content: flex-start;
                padding: 4px 8px;
                box-shadow: none;
            }}
            :host(.blup-section-toggle) .bk-btn:hover {{
                background: {PALETTE.bg2};
                color: {PALETTE.yellow};
            }}
        """
    )


def make_bokeh_theme() -> Theme:
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


