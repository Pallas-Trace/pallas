from __future__ import annotations

from bokeh.models.layouts import LayoutDOM

from state import ViewId
from trace_session import TraceSession
from views.inspector_view import InspectorView


class InspectorPipeline:
    view_id: ViewId = "inspector"

    def __init__(self, t1: TraceSession, t2: TraceSession, *, width: int) -> None:
        self.view = InspectorView(t1, t2, width=width)
        self._root: LayoutDOM | None = None

    # -------------------------------------------
    # |        Base Pipeline Interface          |
    # -------------------------------------------

    def build(self) -> LayoutDOM:
        self._root = self.view.build()
        return self._root

    def root(self) -> LayoutDOM | None:
        return self._root

    def bind(self, controller) -> None:
        # empty
        return

    def refresh(self, controller) -> None:
        self.view.update(
            active_threads      = list(controller.state.context.active_threads),
            n_quanta            = controller.state.views.quanta.n_bins,
            mode                = controller.state.views.quanta.mode,
            token_mode          = controller.state.context.token_mode,
            stack_order         = controller.state.views.quanta.order,
        )


