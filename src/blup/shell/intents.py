from __future__ import annotations

from blup.ui import LayoutDOM
from bokeh.models.sources import ColumnDataSource


class IntentBus:

    def __init__(self) -> None:

        self.panel = ColumnDataSource(
            data = {
                "seq":      [0],
                "panel":    [""],
                "action":   [""],
                "width":    [0]
            }
        )

    def channels(self) -> list[ColumnDataSource]:
        return [self.panel]

    def attach_to(self, root: LayoutDOM) -> None:
        root.tags = [*root.tags, *self.channels()]
