from __future__ import annotations

from typing import Callable

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models.layouts import LayoutDOM
from bokeh.models.widgets.inputs import Select
from bokeh.models.widgets.tables import (
    DataTable,
    HTMLTemplateFormatter,
    TableColumn,
)
from bokeh.plotting import ColumnDataSource

from blup.modules.token_list.types import (
    SortDirection,
    TokenListOrder,
    TokenListResult,
)
from blup.state import TraceMode


_ORDER_OPTIONS: list[tuple[str, str]] = [
    ("delta", "Delta excl"),
    ("excl", "Excl (upper)"),
    ("calls", "Call count"),
    ("name", "Name"),
    ("token", "Token id"),
]

_DIRECTION_OPTIONS: list[tuple[str, str]] = [
    ("desc", "Desc"),
    ("asc", "Asc"),
]

_COLOR_CHIP_TEMPLATE = (
    '<div style="width:14px; height:14px; margin:2px;'
    "background:<%= color %>; border:1px solid #665c54;"
    '"></div>'
)


def _empty_source() -> dict:
    return {
        "color":        [],
        "name":         [],
        "id_label":     [],
        "token_key":    [],
        "token_type":   [],
        "token_id":     [],
        "rank":         [],
        "excl_upper":   [],
        "excl_lower":   [],
        "delta_excl":   [],
        "share":        [],
    }


class TokenListTable:

    def __init__(self, *, height: int = 700) -> None:
        self.height = height

        self.doc = None
        self.table: DataTable | None = None

        self.root: LayoutDOM | None = None

        self.source = ColumnDataSource(data=_empty_source())
        self.order_select: Select | None = None
        self.direction_select: Select | None = None

        self.on_token_selected: (
            Callable[[tuple[int, int] | None], None] | None
        ) = None
        self.on_order_changed: Callable[[str], None] | None = None
        self.on_direction_changed: Callable[[str], None] | None = None

        self._ignore_selection_callbacks = False
        self._ignore_widget_callbacks = False

        self._single_columns: list[TableColumn] = []
        self._dual_columns: list[TableColumn] = []

    def build(self) -> LayoutDOM:
        self.doc = curdoc()

        self.order_select = Select(
            title="Order by",
            value="delta",
            options=_ORDER_OPTIONS,                                         # type: ignore[attr-defined]
            width=150,
        )
        self.order_select.on_change("value", self._on_order_widget)

        self.direction_select = Select(
            title       = "Direction",
            value       = "desc",
            options     = _DIRECTION_OPTIONS,                                     # type: ignore[attr-defined]
            width       = 90,
        )
        self.direction_select.on_change("value", self._on_direction_widget)

        color_formatter = HTMLTemplateFormatter(template=_COLOR_CHIP_TEMPLATE)

        col_color = TableColumn(
            field       = "color",
            title       = "",
            width       = 30,
            formatter   = color_formatter,
        )
        col_name = TableColumn(field="name", title="Token")
        col_id = TableColumn(field="id_label", title="ID", width=70)
        col_excl_upper = TableColumn(
            field="excl_upper", title="Excl (upper)", width=90
        )
        col_excl_lower = TableColumn(
            field="excl_lower", title="Excl (lower)", width=90
        )
        col_delta = TableColumn(field="delta_excl", title="Delta excl", width=90)
        col_share = TableColumn(field="share", title="Share", width=70)

        self._single_columns = [col_color, col_name, col_id, col_excl_upper]
        self._dual_columns = [
            col_color,
            col_name,
            col_id,
            col_excl_upper,
            col_excl_lower,
            col_delta,
            col_share,
        ]

        self.table = DataTable(
            source          = self.source,
            columns         = self._single_columns,
            selectable      = True,
            sortable        = False,
            reorderable     = False,
            index_position  = None,
            row_height      = 26,
            sizing_mode     = "stretch_both",
        )
        self.source.selected.on_change("indices", self._on_source_selected)

        controls = row(
            self.order_select,
            self.direction_select,
            sizing_mode = "stretch_width",
        )
        self.root = column(
            controls,
            self.table,
            sizing_mode = "stretch_both",
        )
        return self.root

    def prepare_display(
        self,
        *,
        trace_mode: TraceMode,
        order: TokenListOrder,
        direction: SortDirection,
    ) -> None:
        table = self.table
        if table is None:
            raise RuntimeError(
                "TokenListSurface.build must be called before prepare_display"
            )

        dual = trace_mode == "dual"
        table.columns = list(
            self._dual_columns if dual else self._single_columns
        )

        self._ignore_widget_callbacks = True
        try:
            if self.order_select is not None and self.order_select.value != order:
                self.order_select.value = order
            if (
                self.direction_select is not None
                and self.direction_select.value != direction
            ):
                self.direction_select.value = direction
        finally:
            self._ignore_widget_callbacks = False

        self._ignore_selection_callbacks = True
        try:
            self.source.selected.indices = []
            self.source.data = _empty_source()
        finally:
            self._ignore_selection_callbacks = False

    def apply_result(self, result: TokenListResult) -> None:
        self._ignore_selection_callbacks = True
        try:
            self.source.selected.indices = []
            self.source.data = result.src
        finally:
            self._ignore_selection_callbacks = False

    def sync_selection(self, token: tuple[int, int] | None) -> None:
        indices: list[int] = []
        if token is not None:
            data = self.source.data
            token_types = data.get("token_type", [])
            token_ids = data.get("token_id", [])
            for i in range(len(token_types)):
                if (
                    int(token_types[i]) == token[0]
                    and int(token_ids[i]) == token[1]
                ):
                    indices = [i]
                    break

        self._ignore_selection_callbacks = True
        try:
            if list(self.source.selected.indices) != indices:
                self.source.selected.indices = indices
        finally:
            self._ignore_selection_callbacks = False

    def _on_source_selected(self, attr: str, old, new) -> None:
        if self._ignore_selection_callbacks:
            return
        indices = list(new)
        if not indices:
            if self.on_token_selected is not None:
                self.on_token_selected(None)
            return

        i = int(indices[0])
        data = self.source.data
        token_types = data.get("token_type", [])
        token_ids = data.get("token_id", [])
        if i < 0 or i >= len(token_types) or i >= len(token_ids):
            return

        token = (int(token_types[i]), int(token_ids[i]))
        if self.on_token_selected is not None:
            self.on_token_selected(token)

    def _on_order_widget(self, attr: str, old: str, new: str) -> None:
        if self._ignore_widget_callbacks:
            return
        if self.on_order_changed is not None:
            self.on_order_changed(new)

    def _on_direction_widget(self, attr: str, old: str, new: str) -> None:
        if self._ignore_widget_callbacks:
            return
        if self.on_direction_changed is not None:
            self.on_direction_changed(new)
