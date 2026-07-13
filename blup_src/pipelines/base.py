from __future__ import annotations

from typing import Protocol, TYPE_CHECKING
from bokeh.models.layouts import LayoutDOM
from state import ViewId

if TYPE_CHECKING:
    from controller import AppController


class DisplayPipeline(Protocol):
    view_id: ViewId

    def build(self) -> LayoutDOM: ...
    def root(self) -> LayoutDOM | None: ...
    def bind(self, controller: "AppController") -> None: ...
    def refresh(self, controller: "AppController") -> None: ...


