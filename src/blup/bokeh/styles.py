from __future__ import annotations

from bokeh.layouts import column
from bokeh.models.layouts import LayoutDOM
from bokeh.models.widgets.markups import Div

from blup.bokeh.theme import PALETTE
from blup.shell.layout import ShellDimensions


def panel_title(text: str) -> Div:
    p = PALETTE
    return Div(
        text = f"""
        <div style="
            padding: 8px 10px;
            background: {p.bg2};
            border-bottom: 1px solid {p.bg3};
            color: {p.yellow};
            font-family: monospace;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.08em;
        ">{text.upper()}</div>
        """,
        sizing_mode = "stretch_width",
    )

def panel_frame(
    title: str,
    content: LayoutDOM,
    *,
    width: int | None = None,
) -> LayoutDOM:
    dims = ShellDimensions()
    p = PALETTE

    return column(
        panel_title(title),
        content,
        sizing_mode = "stretch_height",
        width       = width,
        margin      = 0,
        styles      = {
            "background": p.bg1,
            "border": f"1px solid {p.bg3}",
        },
    )

def status_text(text: str) -> Div:
    p = PALETTE
    return Div(
        text = f"""
        <div style="
            color: {p.muted};
            font-family: monospace;
            font-size: 11px;
            letter-spacing: 0.04em;
            padding: 6px 10px;
        ">{text}</div>
        """,
        sizing_mode = "stretch_width",
    )


