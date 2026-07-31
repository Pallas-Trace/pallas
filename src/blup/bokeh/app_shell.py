from __future__ import annotations

from dataclasses import dataclass

from bokeh.layouts import column, row
from bokeh.models.layouts import LayoutDOM
from bokeh.models.widgets.markups import Div

from blup.bokeh.styles import panel_frame, status_text
from blup.bokeh.theme import PALETTE
from blup.shell.layout import ShellDimensions


@dataclass
class BokehShellElements:
    root:               LayoutDOM
    context_host:       LayoutDOM
    center_host:        LayoutDOM
    inspector_host:     LayoutDOM
    status:             Div


class BokehAppShell:
    def build(self) -> BokehShellElements:
        dims = ShellDimensions()
        p = PALETTE

        header = Div(
            text = f"""
            <div style="
                height: {dims.header_height}px;
                display: flex;
                align-items: center;
                padding: 0 14px;
                box-sizing: border-box;
                background: {p.bg2};
                border: 1px solid {p.bg3};
                color: {p.fg0};
                font-family: monospace;
            ">
                <span style="color:{p.yellow}; font-weight:700;">
                    PALLAS
                </span>
                <span style="color:{p.muted}; margin-left:10px;">
                    TRACE LAB
                </span>
                <span style="margin-left:auto; color:{p.green}; font-size:11px;">
                    ● READY
                </span>
            </div>
            """,
            sizing_mode = "stretch_width",
            height      = dims.header_height,
        )

        context_host = column(
            sizing_mode = "stretch_height",
            width       = dims.context_width,
        )
        center_host = column(
            sizing_mode = "stretch_both",
        )
        inspector_host = column(
            sizing_mode = "stretch_height",
            width       = dims.inspector_width,
        )

        body = row(
            panel_frame("Context", context_host, width=dims.context_width),
            panel_frame("Analysis", center_host),
            panel_frame("Inspector", inspector_host, width=dims.inspector_width),
            sizing_mode = "stretch_both",
            spacing     = dims.panel_gap,
        )

        status = status_text("READY · NO ACTIVE JOBS")

        root = column(
            header,
            body,
            status,
            sizing_mode = "stretch_both",
            spacing     = dims.panel_gap,
            styles      = {
                "background": p.bg0,
                "padding": f"{dims.panel_gap}px",
            },
        )

        return BokehShellElements(
            root            = root,
            context_host    = context_host,
            center_host     = center_host,
            inspector_host  = inspector_host,
            status          = status,
        )


