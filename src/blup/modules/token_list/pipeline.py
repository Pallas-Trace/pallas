from __future__ import annotations

from typing import TYPE_CHECKING

from blup.types import TraceMode
from bokeh.models.layouts import LayoutDOM

from blup.data_model import FidelityMode
from blup.modules.interface import UIWorkRequest
from blup.modules.token_list.assembler import TokenListAssembler
from blup.modules.token_list.table import TokenListTable
from blup.modules.token_list.types import (
    TokenListRequest,
    TokenListResult,
    TokenListTraceContext,
    TokenListUpdate,
    TokenListUpdateContext,
    TraceSide,
)
from blup.state import (
    ContextPatch,
    ModuleID,
    ModulePatch,
    TokenListPatch,
    TokenSelectionPatch,
)

if TYPE_CHECKING:
    from blup.controller import AppController
    from blup.traces.session import TraceSession


class TokenListPipeline:
    module_id: ModuleID = "token_list"

    root: LayoutDOM | None
    host: "AppController | None"
    assembler: TokenListAssembler

    table: TokenListTable

    _pending_update: TokenListUpdate | None
    _active_update: TokenListUpdate | None

    def __init__(self, *, height: int = 700) -> None:
        self.root = None
        self.host = None
        self.assembler = TokenListAssembler()

        self.table = TokenListTable(height=height)

        self._pending_update = None
        self._active_update = None
        self._completed_request_key = None

    @property
    def subscribed_state(self) -> tuple[str, ...]:
        return (
            "context.traces.trace_ids",
            "context.active_threads",
            "context.token_mode",
            "context.selection",
            "modules.token_list",
        )

    def build(self) -> LayoutDOM:
        self.root = self.table.build()
        return self.root

    def bind(self, host: "AppController") -> None:
        self.host = host
        self.table.on_token_selected = (
            lambda token: host.update_state(
                context = ContextPatch(
                    selection = TokenSelectionPatch(
                        token=token,
                    ),
                ),
            )
        )
        self.table.on_order_changed = (
            lambda order: host.update_state(
                modules = ModulePatch(
                    token_list = TokenListPatch(
                        order=order,
                    ),
                ),
            )
        )
        self.table.on_direction_changed = (
            lambda direction: host.update_state(
                modules = ModulePatch(
                    token_list = TokenListPatch(
                        direction=direction,
                    ),
                ),
            )
        )

    def refresh(self, host: "AppController") -> None:
        update = self.prepare_update(host)
        self._pending_update = update

        request = self.assembler.prepare_request(update)

        host.work_manager.submit(
            UIWorkRequest(
                request     = request,
                pipeline    = self,
            )
        )

    def prepare_update(self, host: "AppController") -> TokenListUpdate:
        app_ctx = host.state.context
        trace_ids = app_ctx.traces.trace_ids

        # check trace sessions
        sessions = host.trace_registry.get_sessions(trace_ids)
        if not sessions:
            raise RuntimeError(
                "Token list module requires at least one selected trace"
            )
        upper = sessions[0]
        lower = sessions[1] if len(sessions) >= 2 else None
        trace_mode: TraceMode = "dual" if lower is not None else "single"

        # check trace threads
        available_threads = host.trace_registry.thread_names_for(
            trace_ids,
        )
        available_thread_set = set(available_threads)
        active_threads = tuple(
            thread_name
            for thread_name in app_ctx.active_threads
            if thread_name in available_thread_set
        )

        # prepare update context
        update_ctx = self._freeze_update_context(
            host                    = host,
            upper_session           = upper,
            lower_session           = lower,
            active_thread_names     = active_threads,
            trace_mode              = trace_mode,
        )

        return TokenListUpdate(
            active_thread_names     = active_threads,
            selected_token          = app_ctx.selection.token,
            trace_mode              = trace_mode,
            context                 = update_ctx,
        )

    def start_update(self) -> None:
        if self._pending_update is None:
            raise RuntimeError(
                "token_list start_update called without pending update"
            )

        update = self._pending_update
        self._active_update = update
        self._pending_update = None

        self.table.prepare_display(
            trace_mode      = update.trace_mode,
            order           = update.context.order,
            direction       = update.context.direction,
        )

    def apply_result(self, result: TokenListResult) -> None:
        update = self._active_update
        if update is None:
            return

        self.table.apply_result(result)
        self.table.sync_selection(update.selected_token)

    def finish_update(self, *, cancelled: bool) -> None:
        self._active_update = None

    def _freeze_update_context(
        self,
        host: "AppController",
        upper_session: "TraceSession",
        lower_session: "TraceSession | None",
        active_thread_names: tuple[str, ...],
        trace_mode: TraceMode,
    ) -> TokenListUpdateContext:
        app_ctx = host.state.context
        mod_cfg = host.state.modules.token_list
        trace_ids = app_ctx.traces.trace_ids

        upper_trace_id = trace_ids[0]
        lower_trace_id = (
            trace_ids[1]
            if lower_session is not None
            else None
        )

        token_keys = set(upper_session.meta.token_key_to_name.keys())
        trace_context: dict[TraceSide, TokenListTraceContext] = {
            "upper": TokenListTraceContext(
                # TODO: set standard version for this call
                label = self._trace_label(upper_session, upper_trace_id),
                thread_name_to_id = {
                    str(k): int(v)
                    for k, v in upper_session.meta.thread_name_to_id.items()
                },
                token_name_by_key = dict(upper_session.meta.token_key_to_name),
                summarize_tokens = (
                    lambda query, session=upper_session
                        : session.query_summary(query)
                )
            )
        }
        if lower_session is not None and lower_trace_id is not None:
            token_keys.update(lower_session.meta.token_key_to_name.keys())
            trace_context["lower"] = TokenListTraceContext(
                # TODO: set standard version for this call
                label = self._trace_label(lower_session, lower_trace_id),
                thread_name_to_id = {
                    str(k): int(v)
                    for k, v in lower_session.meta.thread_name_to_id.items()
                },
                token_name_by_key = dict(lower_session.meta.token_key_to_name),
                summarize_tokens = (
                    lambda query, session=lower_session
                        : session.query_summary(query)
                )
            )

        request_key = (
            upper_trace_id,
            lower_trace_id,
            active_thread_names,
            trace_mode,
            mod_cfg.top_k,
            mod_cfg.fidelity,
            mod_cfg.order,
            mod_cfg.direction,
            app_ctx.token_mode,
            # TODO: later update to include recolors/nicknames/pins, etc.
        )

        color_map = dict(host.token_color.snapshot(token_keys).color_map)

        return TokenListUpdateContext(
            request_key     = request_key,
            trace_context   = trace_context,
            fidelity        = mod_cfg.fidelity,
            token_mode      = app_ctx.token_mode,
            top_k           = mod_cfg.top_k,
            order           = mod_cfg.order,
            direction       = mod_cfg.direction,
            color_map       = color_map,
        )

    def _trace_label(self, session: "TraceSession", trace_id) -> str:
        # TODO: implement standard app-wide way to store/access this
        label = getattr(session, "label", None)
        return str(label) if label else str(trace_id)


