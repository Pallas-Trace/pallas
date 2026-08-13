from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from bokeh.layouts import column, grid
from bokeh.models.layouts import LayoutDOM
from bokeh.models.widgets.inputs import MultiSelect, Select, Spinner
from bokeh.models.widgets.markups import Div

from blup.bokeh.app_shell import BokehAppShell
from blup.bokeh.styles import panel_title
from blup.state import AppState

if TYPE_CHECKING:
    from blup.controller import AppController


@dataclass
class UIElements:
    root:                       LayoutDOM

    title:                      Div
    status:                     Div

    main_host:                  LayoutDOM
    context_host:               LayoutDOM
    inspector_host:             LayoutDOM

    context_panel:              LayoutDOM
    inspector_panel:            LayoutDOM


class UIModel:
    def __init__(self, controller: "AppController") -> None:
        self.controller = controller

    def build(
        self,
        *,
        state: AppState,
        all_thread_names: tuple[str, ...],
    ) -> UIElements:
        shell = BokehAppShell().build()

        return UIElements(
            root                    = shell.root,
            title                   = shell.title,
            status                  = shell.status,
            main_host               = shell.main_host,
            context_host            = shell.context_host,
            inspector_host          = shell.inspector_host,
            context_panel           = shell.context_panel,
            inspector_panel         = shell.inspector_panel,
        )


