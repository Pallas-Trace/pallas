from __future__ import annotations

from bokeh.models.layouts import LayoutDOM

from data_model import SummaryQuery
from state import ViewId
from trace_session import TraceSession
from views.inspector_view import InspectorDisplayModel, InspectorView


class InspectorPipeline:
    view_id: ViewId = "inspector"

    def __init__(self, *, width: int) -> None:
        self.view = InspectorView(width=width)
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
        primary = controller.get_primary_trace()
        secondary = controller.get_secondary_trace()
        token_mode = controller.state.context.token_mode

        model = InspectorDisplayModel(
            title="Inspector",
            primary_label=primary.label,
            secondary_label=None if secondary is None else secondary.label,
            thread_count=len(controller.state.context.active_threads),
            n_quanta=controller.state.views.quanta.n_bins,
            token_view_label="named" if token_mode == "category" else "raw",
            snapshot_mode=controller.state.views.quanta.mode,
            stack_order=controller.state.views.quanta.order,
            top_primary=self.top_token_labels(primary.session, limit=8, token_mode=token_mode),
            top_secondary=() if secondary is None else self.top_token_labels(
                secondary.session, limit=8, token_mode=token_mode
            ),
        )

        self.view.update(model)

    def top_token_labels(self, session, *, limit: int, token_mode) -> tuple[str, ...]:
        summary = session.summarize_tokens(
            SummaryQuery(
                thread_ids=(),
                fidelity="fast",
                token_mode=token_mode,
                top_k=limit,
                block_only=True,
            )
        )
        labels = []
        for row in summary.tokens[:limit]:
            key = f"{row.token_type}:{row.token_id}"
            labels.append(session.meta.token_key_to_name.get(key, key))
        return tuple(labels)


