from __future__ import annotations

from dataclasses import dataclass

from blup.state import PanelSide
from bokeh.layouts import Spacer, column, grid, row
from bokeh.models.callbacks import CustomJS
from bokeh.models.layouts import LayoutDOM
from bokeh.models.widgets.markups import Div

from blup.bokeh.styles import panel_frame, status_text
from blup.bokeh.theme import GLYPHS, PALETTE
from blup.shell.layout import ShellDimensions


@dataclass
class BokehShellElements:
    root:               LayoutDOM

    title:              Div
    status:             Div

    main_host:          LayoutDOM
    context_host:       LayoutDOM
    inspector_host:     LayoutDOM

    context_panel:      LayoutDOM
    inspector_panel:    LayoutDOM


def collapse_arrows(side: PanelSide) -> tuple[str, str]:
    match side:
        case "left":    return GLYPHS.arrow_left, GLYPHS.arrow_right
        case "right":   return GLYPHS.arrow_right, GLYPHS.arrow_left
        case "top":     return GLYPHS.arrow_up, GLYPHS.arrow_down
        case "bottom":  return GLYPHS.arrow_down, GLYPHS.arrow_up
        case _:         raise ValueError(f"panel at '{side}' is not collapsible")


_WIDTH_RECONCILER = """
    function find(s, r) {
        let els = Array.from(r.querySelectorAll(s));
        for (const h of r.querySelectorAll('*')) {
            if (h.shadowRoot) els = els.concat(find(s, h.shadowRoot));
        }
        return els;
    }
    const el = find(sel, document)[0];
    if (el) el.style.flex = "0 0 " + cb_obj.width + "px";
"""

def _bind_width_reconciler(panel: LayoutDOM, panel_id: str) -> None:
    panel.js_on_change("width", CustomJS(
        args={"sel": f".blup-panel-{panel_id}"},
        code=_WIDTH_RECONCILER,
    ))


class BokehAppShell:
    def build(self) -> BokehShellElements:
        dims = ShellDimensions()
        p = PALETTE

        title = Div(
            text = f"""
            <style>
                .bk-clearfix {{
                    display: block !important;
                    width: 100% !important;
                    height: 100%;
                }}
            </style>

            <div style="
                width: 100%;
                height: {dims.header_height}px;
                display: flex;
                align-items: center;
                padding: 0 16px;
                box-sizing: border-box;
                background: {p.bg0};
                border: 0px solid {p.bg3};
                color: {p.fg0};
                font-family: monospace;
                margin: 0;
            ">
                <span style="
                    color: {p.orange};
                    font-size: 15px;
                    font-weight: 800;
                    letter-spacing: 0.05em;
                ">
                    BLUP
                </span>
                <span style="
                    color: {p.muted};
                    font-size: 15px;
                    font-weight: 800;
                    letter-spacing: 0.05em;
                    margin-left: 6px;
                ">
                    TRACE
                </span>
                <span style="
                    margin-left: auto;
                    padding-left: 24px;
                    color: {p.green};
                    font-size: 11px;
                    letter-spacing: 0.06em;
                    white-space: nowrap;
                ">
                    ● READY
                </span>
            </div>
            """,
            sizing_mode     = "stretch_width",
            margin          = 0,
            width_policy    = "max",
            height          = dims.header_height,
        )

        # --- Panel hosts ---

        context_host = column(sizing_mode="stretch_both")
        main_host = column(sizing_mode="stretch_both", min_width=200)
        inspector_host = column(sizing_mode="stretch_both")

        context_panel = panel_frame(
            "Context",
            context_host,
            panel_id        = "context",
            collapsible     = True,
            arrow           = "◀",
            arrow_side      = "right",
        )
        context_panel.width = dims.context_width
        context_panel.min_width = dims.context_min_width
        context_panel.sizing_mode = "stretch_height"

        main_panel = panel_frame("Analysis", main_host)
        main_panel.sizing_mode = "stretch_both"

        inspector_panel = panel_frame(
            "Inspector",
            inspector_host,
            panel_id        = "inspector",
            collapsible     = True,
            arrow           = "▶",
            arrow_side      = "left",
        )
        inspector_panel.width = dims.inspector_width
        inspector_panel.min_width = dims.inspector_min_width
        inspector_panel.sizing_mode = "stretch_height"

        split_left = Spacer(
            width           = dims.splitter_width,
            sizing_mode     = "stretch_height",
            css_classes     = ["blup-split", "blup-split-left"],
            styles          = {
                "background": p.bg3,
                "cursor": "col-resize",
                "flex-shrink": "0",
                "transition": "background 0.1s",
            },
        )
        split_right = Spacer(
            width           = dims.splitter_width,
            sizing_mode     = "stretch_height",
            css_classes     = ["blup-split", "blup-split-right"],
            styles          = {
                "background": p.bg3,
                "cursor": "col-resize",
                "flex-shrink": "0",
                "transition": "background 0.1s",
            },
        )

        body = row(
            context_panel,
            split_left,
            main_panel,
            split_right,
            inspector_panel,
            sizing_mode     = "stretch_both",
            spacing         = 0,
        )

        status = status_text("READY · NO ACTIVE JOBS")

        root = column(
            title,
            body,
            status,
            sizing_mode     = "stretch_both",
            spacing         = 0,
            styles          = {
                "background": p.bg0, "box-sizing": "border-box"
            },
        )

        title.width_policy = "max"
        status.width_policy = "max"
        root.width_policy = "max"

        _bind_width_reconciler(context_panel, "context")
        _bind_width_reconciler(inspector_panel, "inspector")

        return BokehShellElements(
            root            = root,
            title           = title,
            status          = status,
            main_host       = main_host,
            context_host    = context_host,
            inspector_host  = inspector_host,
            context_panel   = context_panel,
            inspector_panel = inspector_panel,
        )


