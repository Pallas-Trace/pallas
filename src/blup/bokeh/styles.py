from __future__ import annotations

from bokeh.layouts import column, row
from bokeh.models.callbacks import CustomJS
from bokeh.models.layouts import LayoutDOM
from bokeh.models.widgets.buttons import Button
from bokeh.models.widgets.inputs import MultiSelect, Select, Spinner
from bokeh.models.widgets.markups import Div

from blup.bokeh.theme import PALETTE, make_widget_stylesheet
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
    panel_id: str | None = None,
    collapsible: bool = False,
    arrow: str = "◀",
    arrow_side: str = "right",
) -> LayoutDOM:
    children = []
    if collapsible:
        btn_classes = ["blup-collapse-btn"]
        if panel_id:
            btn_classes.append(f"blup-collapse-{panel_id}")

        title_div = Div(
            text            = title,
            name            = f"blup-panel-title-{panel_id}",
            css_classes     = ["blup-panel-header"],
            margin          = 0,
            styles          = {
                "background": PALETTE.bg2,
                "color": PALETTE.fg0,
                "font-family": "monospace",
                "font-size": "10px",
                "font-weight": "700",
                "text-transform": "uppercase",
                "letter-spacing": "0.5px",
                "padding": "4px 12px",
                "line-height": "20px",
                "flex-grow": "1",
                "text-align": "right" if arrow_side == "left" else "left",
            }
        )
        arrow_div = Div(
            text            = arrow,
            name            = f"blup-collapse-btn-{panel_id}",
            margin          = 0,
            css_classes     = btn_classes,
            styles          = {
                "color": PALETTE.fg2,
                "cursor": "pointer",
                "font-size": "14px",
                "font-family": "monospace",
                "padding": "4px 8px 0 8px",
                "line-height": "16px",
                "text-align": "center",
                "user-select": "none",
                "flex-shrink": "0",
            }
        )

        ordered = (
                [arrow_div, title_div]
                if arrow_side == "left"
                else [title_div, arrow_div]
        )

        header_row = row(
            *ordered,
            sizing_mode     = "stretch_width",
            height          = 28,
            align           = "center",
            styles          = {
                "background": PALETTE.bg2,
                "border-bottom": f"2px solid {PALETTE.bg3}",
                "flex-shrink": "0",
            },
        )
        children.append(header_row)
    else:
        header_row = row(
            Div(
                text        = title,
                name        = f"blup-panel-title-{panel_id}",
                css_classes = ["blup-panel-header"],
                margin      = 0,
                styles      = {
                    "background": PALETTE.bg2,
                    "color": PALETTE.fg0,
                    "font-family": "monospace",
                    "font-size": "10px",
                    "font-weight": "700",
                    "text-transform": "uppercase",
                    "letter-spacing": "0.5px",
                    "padding": "4px 12px",
                    "line-height": "20px",
                    "flex-grow": "1",
                },
                sizing_mode="stretch_width",
            ),
            sizing_mode     = "stretch_width",
            height          = 28,
            styles          = {
                "background": PALETTE.bg2,
                "border-bottom": f"2px solid {PALETTE.bg3}",
                "flex-shrink": "0",
            },
        )
        children.append(header_row)

    panel_classes = ["blup-panel"]
    if panel_id:
        panel_classes.append(f"blup-panel-{panel_id}")

    children.append(content)
    panel = column(
        *children,
        sizing_mode="stretch_both",
        css_classes=panel_classes,
        styles={
            "border": f"1px solid {PALETTE.bg3}",
            "border-radius": "0px",
            "overflow": "hidden",
            "background": PALETTE.bg1,
        },
    )
    return panel

def section_label(text: str) -> Div:
    p = PALETTE
    return Div(
        text=f"""
        <div style="
            color: {p.yellow};
            font-family: monospace;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.08em;
            padding: 8px 0 5px 0;
            border-bottom: 1px solid {p.bg3};
            text-transform: uppercase;
        ">
            {text}
        </div>
        """,
        sizing_mode="stretch_width",
        height=30,
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

def style_widget(widget) -> None:
    p = PALETTE
    widget.styles = {
        "font-family": "monospace",
        "font-size": "11px",
        "color": p.fg1,
        "background": p.bg0,
        "border": f"1px solid {p.bg3}",
        "border-radius": "0",
        "padding": "4px 6px",
        "box-sizing": "border-box",
    }

def collapsible_section(title: str, content: LayoutDOM) -> LayoutDOM:
    p = PALETTE

    toggle = Button(
        label=f"▾ {title.upper()}",
        button_type="default",
        sizing_mode="stretch_width",
        height=28,
        css_classes=["blup-section-toggle"],
        stylesheets=[make_widget_stylesheet()],
    )

    content_col = column(
        content,
        sizing_mode="stretch_width",
        visible=True,
    )

    toggle.js_on_click(
        CustomJS(
            args={"content_col": content_col, "toggle_btn": toggle},
            code="""
            content_col.visible = !content_col.visible;
            toggle_btn.label = content_col.visible
                ? "▾ " + toggle_btn.label.substring(2)
                : "▸ " + toggle_btn.label.substring(2);
            """,
        )
    )

    return column(
        toggle,
        content_col,
        sizing_mode="stretch_width",
        spacing=0,
    )


