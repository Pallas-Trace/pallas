from __future__ import annotations

import os
import sys

from bokeh.io import curdoc

from blup.bokeh.theme import gruvbox_bokeh_theme
from blup.controller import AppController
from blup.traces.interface import TraceRecord
from blup.traces.session import TraceSession
from blup.utils import timed


def load_trace_records(
    paths: list[str],
) -> list[TraceRecord]:
    records: list[TraceRecord] = []

    for index, path in enumerate(paths):
        label = os.path.basename(path) or f"trace_{index}"
        session = TraceSession(path)

        with timed(f"open {label}"):
            session.open()

        records.append(
            TraceRecord(
                trace_id=f"trace_{index}",
                label=label,
                session=session,
            )
        )

    return records


def main() -> None:
    paths = [
        path
        for path in sys.argv[1:]
        if path.strip()
    ]

    if not paths:
        raise SystemExit(
            "Usage: bokeh serve --show main.py "
            "--args TRACE [TRACE ...]"
        )

    trace_records = load_trace_records(paths)

    with timed("build"):
        controller = AppController(trace_records)
        root = controller.build()

    curdoc().add_root(root)                                                 # type: ignore
    curdoc().theme = gruvbox_bokeh_theme()
    curdoc().title = "Blup"


main()
