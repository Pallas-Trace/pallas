from __future__ import annotations

import os
import sys

from blup.bokeh.custom_js import JSDispatcher
from blup.shell.layout import ShellDimensions
from blup.ui import UIElements
from bokeh.events import DocumentReady
from bokeh.io import curdoc
from bokeh.models.callbacks import CustomJS
from bokeh.models.css import GlobalInlineStyleSheet

from blup.bokeh.theme import PALETTE, gruvbox_bokeh_theme
from blup.controller import AppController
from blup.traces.interface import TraceRecord
from blup.traces.session import TraceSession
from blup.utils import timed
from bokeh.plotting import ColumnDataSource


def parse_args() -> list[str]:
    paths = [p for p in sys.argv[1:] if p.strip()]
    if not paths:
        raise SystemExit(
            "Usage: bokeh serve --show main.py --args TRACE [TRACE ...]"
        )
    return paths

def load_trace_records(
    paths: list[str],
) -> list[TraceRecord]:
    records: list[TraceRecord] = []

    for index, path in enumerate(paths):
        label = os.path.basename(path) or f"trace_{index}"
        session = TraceSession(path)

        with timed(f"open {label}"):
            session.open()

        records.append(
            TraceRecord(
                trace_id=f"trace_{index}",
                label=label,
                session=session,
            )
        )

    return records

def main() -> None:

    # parse inputs
    paths = parse_args()
    trace_records = load_trace_records(paths)

    # setup document styles
    doc = curdoc()
    doc.stylesheets = [                                                     # type: ignore
        GlobalInlineStyleSheet(
            css=f"""
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
    ]
    doc.theme = gruvbox_bokeh_theme()
    doc.title = "Blup Trace"

    # build app controllr
    with timed("build"):
        controller = AppController(trace_records)
        root = controller.build()
    if root is None:
        raise RuntimeError("unable to build bokeh root in startup")

    # mount app root
    root.sizing_mode = "stretch_both"                                       # type: ignore
    curdoc().add_root(root)                                                 # type: ignore

    # wire intent bus
    if controller.runtime.ui is None:
        raise RuntimeError("ui elements must be built")
    ui: UIElements = controller.runtime.ui
    controller.intent_bus.attach_to(ui.root)

    # load custom js scripts
    dims = ShellDimensions()
    js = JSDispatcher(ui, controller.intent_bus)
    doc.js_on_event(
        DocumentReady,
        js.tooltip_style_js(),
        js.drag_handler(),
        js.collapse_handler(),
        js.remove_canvas_focus(),
        # js.disable_context_menu(),
    )


main()
