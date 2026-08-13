from __future__ import annotations

import os
import sys

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
    paths = [
        path
        for path in sys.argv[1:]
        if path.strip()
    ]

    if not paths:
        raise SystemExit(
            "Usage: bokeh serve --show main.py "
            "--args TRACE [TRACE ...]"
        )

    trace_records = load_trace_records(paths)

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

    tooltip_style_js = CustomJS(
        args = {
            "css": f"""
                :host {{
                    background: {PALETTE.bg1} !important;
                    border: 1px solid {PALETTE.bg3} !important;
                    border-radius: 0 !important;
                    padding: 0 !important;
                    box-shadow: none !important;
                }}
            """
        },
        code = """
        const styled = new WeakSet();

        function styleTooltip(el) {
            if (styled.has(el) || !el.shadowRoot) return;
            styled.add(el);
            const sheet = new CSSStyleSheet();
            sheet.replaceSync(css);
            el.shadowRoot.adoptedStyleSheets = [
                ...el.shadowRoot.adoptedStyleSheets, sheet
            ];
        }

        // Style any that already exist
        document.querySelectorAll(".bk-Tooltip").forEach(styleTooltip);

        // Style new ones as they are inserted
        const observer = new MutationObserver((mutations) => {
            for (const m of mutations) {
                for (const node of m.addedNodes) {
                    if (node.nodeType !== 1) continue;
                    if (node.classList?.contains("bk-Tooltip")) {
                        styleTooltip(node);
                    }
                    node.querySelectorAll?.(".bk-Tooltip").forEach(styleTooltip);
                }
            }
        });
        observer.observe(document.body, { childList: true, subtree: true });
        """
    )
    curdoc().js_on_event(DocumentReady, tooltip_style_js)

    with timed("build"):
        controller = AppController(trace_records)
        root = controller.build()

    if root is None:
        raise RuntimeError("unable to build bokeh root in startup")

    root.sizing_mode = "stretch_both"                                       # type: ignore
    curdoc().add_root(root)                                                 # type: ignore

    if controller.runtime.ui is None:
        raise RuntimeError("ui elements must be built")
    ui: UIElements = controller.runtime.ui

    # bind intent_bus channels to root
    controller.intent_bus.attach_to(ui.root)

    dims = ShellDimensions()

    drag_handler = CustomJS(
        args = {
            "left_panel":       ui.context_panel,
            "left_name":        "context",
            "min_l":            dims.context_min_width,
            "max_l":            800,
            "right_panel":      ui.inspector_panel,
            "right_name":       "inspector",
            "min_r":            dims.inspector_min_width,
            "max_r":            800,
            "split_bg":         PALETTE.bg3,     # normal color
            "split_hover":      PALETTE.yellow,  # hover color
            "rail":             dims.panel_rail_width,
            "intent":           controller.intent_bus.panel,
        },
        code = """
        function findInShadowRoots(selector, root = document) {
            let els = Array.from(root.querySelectorAll(selector));
            for (const host of root.querySelectorAll('*')) {
                if (host.shadowRoot) {
                    els = els.concat(findInShadowRoots(selector, host.shadowRoot));
                }
            }
            return els;
        }

        function makeDraggable(splitSel, panelSel, panelModel,
                minW, maxW, invert, panelName) {
            const bindAll = () => {
                findInShadowRoots(splitSel).forEach(splitEl => {
                    if (splitEl._dragBound) return;
                    splitEl._dragBound = true;

                    splitEl.addEventListener("mouseenter", () => {
                        splitEl.style.background = split_hover;
                    });
                    splitEl.addEventListener("mouseleave", () => {
                        if (!splitEl._dragging) {
                            splitEl.style.background = split_bg;
                        }
                    });

                    splitEl.addEventListener("mousedown", (e) => {
                        e.preventDefault();
                        const panelEl = findInShadowRoots(panelSel)[0];
                        if (!panelEl) return;

                        splitEl._dragging = true;
                        const startX = e.clientX;
                        const startW = panelEl.getBoundingClientRect().width;
                        const prevSelect = document.body.style.userSelect;

                        document.body.style.cursor = "col-resize";
                        document.body.style.userSelect = "none";

                        const SYNC_INTERVAL = 500; // ms
                        let lastSync = 0;

                        const syncModel = () => {
                            panelModel.width = parseInt(panelEl.style.width);
                            lastSync = Date.now();
                        };

                        const wasCollapsed = startW <= rail + 4;
                        let uncollapsed = !wasCollapsed;

                        const move = (ev) => {
                            let dx = ev.clientX - startX;
                            if (invert) dx = -dx;
                            const effMin = wasCollapsed && !uncollapsed ? rail : minW;
                            const newW = Math.round(
                                Math.max(effMin, Math.min(maxW, startW + dx))
                            );

                            if (!uncollapsed && newW > rail + 4) {
                                uncollapsed = true;
                                intent.data = {
                                    seq: [Date.now()],
                                    panel: [panelName],
                                    action: ["uncollapse"],
                                    width: [newW]
                                };
                                intent.change.emit();
                            }

                            panelEl.style.width = newW + "px";
                            panelEl.style.flex = "0 0 " + newW + "px";
                            if (Date.now() - lastSync >= SYNC_INTERVAL) {
                                syncModel();
                            }
                        };

                        const up = () => {
                            syncModel();

                            intent.data = {
                                seq: [Date.now()],
                                panel: [panelName],
                                action: ["resize"],
                                width: [parseInt(panelEl.style.width)]
                            };
                            intent.change.emit();

                            splitEl._dragging = false;
                            splitEl.style.background = split_bg;
                            document.body.style.cursor = "";
                            document.body.style.userSelect = prevSelect;
                            document.removeEventListener("mousemove", move);
                            document.removeEventListener("mouseup", up);
                        };

                        document.addEventListener("mousemove", move);
                        document.addEventListener("mouseup", up);
                    });
                });
            };

            bindAll();
            new MutationObserver(bindAll).observe(
                document.body, { childList: true, subtree: true }
            );
            setTimeout(bindAll, 1000);
        }

        makeDraggable(".blup-split-left", ".blup-panel-context",
                    left_panel, min_l, max_l, false, left_name);
        makeDraggable(".blup-split-right", ".blup-panel-inspector",
                    right_panel, min_r, max_r, true, right_name);
        """
    )

    collapse_handler = CustomJS(
        args = {
            "btn_color":        PALETTE.fg2,
            "btn_hover":        PALETTE.yellow,
            "intent":           controller.intent_bus.panel,
        },
        code = """
        function findInShadowRoots(selector, root = document) {
            let els = Array.from(root.querySelectorAll(selector));
            for (const host of root.querySelectorAll('*')) {
                if (host.shadowRoot) {
                    els = els.concat(findInShadowRoots(selector, host.shadowRoot));
                }
            }
            return els;
        }

        function makeCollapsible(btnSel, panelName) {
            const bindAll = () => {
                findInShadowRoots(btnSel).forEach(btnEl => {
                    if (btnEl._collapseBound) return;
                    btnEl._collapseBound = true;

                    btnEl.style.transition = "color 0.1s, transform 0.1s";
                    btnEl.addEventListener("mouseenter", () => {
                        btnEl.style.color = btn_hover;
                        btnEl.style.transform = "scale(1.3)";
                    });
                    btnEl.addEventListener("mouseleave", () => {
                        btnEl.style.color = btn_color;
                        btnEl.style.transform = "scale(1)";
                    });

                    btnEl.addEventListener("click", () => {
                        console.log("[click] toggle", panelName);
                        console.log("[click] intent object:", intent);  
                        intent.data = {
                            seq: [Date.now()],
                            panel: [panelName],
                            action: ["toggle"],
                            width: [0]
                        };
                        intent.change.emit();
                        console.log("[click] emitted");
                    });
                });
            };

            bindAll();
            new MutationObserver(bindAll).observe(
                document.body, { childList: true, subtree: true }
            );
            setTimeout(bindAll, 1000);
        }

        makeCollapsible(".blup-collapse-context", "context");
        makeCollapsible(".blup-collapse-inspector", "inspector");
        """
    )

    print(f"[setup] handler args bus: {id(controller.intent_bus.panel)}")

    disable_context_menu = CustomJS(
        code = """
            document.addEventListener("contextmenu", (e) => {
                e.preventDefault();
            });
        """
    )

    remove_canvas_focus = CustomJS(
        code = """
            document.addEventListener("mousedown", (e) => {
                const path = e.composedPath();
                const onCanvas = path.some(
                    el => el.tagName === "CANVAS" ||
                        (el.classList && el.classList.contains("bk-Canvas"))
                );
                if (onCanvas) e.preventDefault();
            }, true);
        """
    )

    doc.js_on_event(
        DocumentReady,
        drag_handler,
        collapse_handler,
        # disable_context_menu,
        remove_canvas_focus,
    )


main()
