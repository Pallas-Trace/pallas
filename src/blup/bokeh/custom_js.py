from __future__ import annotations

from blup.bokeh.intents import IntentBus
from blup.bokeh.theme import PALETTE, Palette
from blup.controller import AppController
from blup.shell.layout import ShellDimensions
from blup.ui import UIElements
from bokeh.models.callbacks import CustomJS


class JSDispatcher:
    p: Palette
    ui: UIElements
    dims: ShellDimensions
    intent_bus: IntentBus

    def __init__(
        self,
        ui: UIElements,
        intent_bus: IntentBus,
    ) -> None:
        self.ui = ui
        self.intent_bus = intent_bus

        self.p = PALETTE
        self.dims = ShellDimensions()

    def tooltip_style_js(self) -> CustomJS:
        return CustomJS(
            args = {
                "css": f"""
                    :host {{
                        background: {self.p.bg1} !important;
                        border: 1px solid {self.p.bg3} !important;
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

    def drag_handler(self) -> CustomJS:
        return CustomJS(
            args = {
                "left_panel":       self.ui.context_panel,
                "left_name":        "context",
                "min_l":            self.dims.context_min_width,
                "max_l":            800,
                "right_panel":      self.ui.inspector_panel,
                "right_name":       "inspector",
                "min_r":            self.dims.inspector_min_width,
                "max_r":            800,
                "split_bg":         self.p.bg3,     # normal color
                "split_hover":      self.p.yellow,  # hover color
                "rail":             self.dims.panel_rail_width,
                "intent":           self.intent_bus.panel,
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

    def collapse_handler(self) -> CustomJS:
        return CustomJS(
            args = {
                "btn_color":        self.p.fg2,
                "btn_hover":        self.p.yellow,
                "intent":           self.intent_bus.panel,
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

    def disable_context_menu(self) -> CustomJS: 
        return CustomJS(
            code = """
                document.addEventListener("contextmenu", (e) => {
                    e.preventDefault();
                });
            """
        )

    def remove_canvas_focus(self) -> CustomJS:
        return CustomJS(
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
