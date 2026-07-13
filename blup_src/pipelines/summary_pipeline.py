# pipelines/summary_pipeline.py
from __future__ import annotations

from bokeh.models.layouts import LayoutDOM

from adapters.summary_adapter import (
    SequenceSummaryDiffAdapter,
    SequenceSummaryDiffRow,
    format_duration_ns,
)
from state import ViewId
from trace_session import TraceSession
from views.summary_view import SummaryView


class SummaryPipeline:
    view_id: ViewId = "summary"

    def __init__(self, t1: TraceSession, t2: TraceSession, *, width: int, height: int) -> None:
        self.view = SummaryView(width=width, height=height)
        self.adapter = SequenceSummaryDiffAdapter(t1, t2)
        self._root: LayoutDOM | None = None
        self._rows: tuple[SequenceSummaryDiffRow, ...] = ()
        self._ignore_widget_callbacks = False

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
        self._rows = self.adapter.build_rows(
            token_mode              = controller.state.context.token_mode,
            fidelity                = "fast",
            top_k                   = 32,
            active_thread_names     = tuple(controller.state.context.active_threads),
        )
        self.sync_highlight_widget(controller)
        self.refresh_detail(controller)

    # -------------------------------------------
    # |              Callbacks                  |
    # -------------------------------------------

    def on_highlight_widget_changed(self, controller, value: str) -> None:
        if self._ignore_widget_callbacks:
            return
        controller.set_highlight_token(self.select_value_to_token(value))

    def sync_highlight_widget(self, controller) -> None:
        ui = controller.runtime.ui
        if ui is None:
            return

        widget = ui.highlight_token_select
        options = self.highlight_token_options(self._rows)
        valid_values = {value for value, _ in options}

        current_value = self.token_to_select_value(controller.state.context.selection.token)
        if current_value not in valid_values:
            current_value = options[1][0] if len(options) > 1 else ""
            controller.state.context.selection.token = self.select_value_to_token(current_value)

        self._ignore_widget_callbacks = True
        try:
            widget.options = options  # type: ignore
            if widget.value != current_value:
                widget.value = current_value
        finally:
            self._ignore_widget_callbacks = False

    def refresh_detail(self, controller) -> None:
        token = controller.state.context.selection.token
        row = next(
            (r for r in self._rows if (r.token_type, r.token_id) == token),
            None,
        )

        t0_ns = controller.state.context.time_scope.t0_ns
        t1_ns = controller.state.context.time_scope.t1_ns
        if t0_ns is None or t1_ns is None:
            t0_ns, t1_ns = controller.full_time_bounds()

        model = self.adapter.build_display_model(
            row,
            active_thread_names     = tuple(controller.state.context.active_threads),
            token_mode              = controller.state.context.token_mode,
            t0_ns                   = t0_ns,
            t1_ns                   = t1_ns,
            histogram_bins          = 20,
        )
        self.view.update(model)

    # -------------------------------------------
    # |               Helpers                   |
    # -------------------------------------------

    def highlight_token_options(
        self,
        rows: tuple[SequenceSummaryDiffRow, ...],
    ) -> list[tuple[str, str]]:
        opts: list[tuple[str, str]] = [("", "(none)")]
        for row in rows:
            value = self.token_to_select_value((row.token_type, row.token_id))
            label = (
                f"#{row.contribution_rank} "
                f"{row.name} "
                f"({format_duration_ns(row.contribution_abs_ns)}, "
                f"{row.contribution_share_pct:.1f}%) "
                f"[{row.token_type}:{row.token_id}]"
            )
            opts.append((value, label))
        return opts

    def token_to_select_value(self, token: tuple[int, int] | None) -> str:
        if token is None:
            return ""
        return f"{token[0]}:{token[1]}"

    def select_value_to_token(self, value: str) -> tuple[int, int] | None:
        if not value:
            return None
        a, b = value.split(":", 1)
        return (int(a), int(b))
