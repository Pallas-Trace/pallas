# Architecture Design Document

## System Overview

```text
main.py
  │
  │ opens trace files
  ▼
TraceRecord and TraceRegistry
  │
  │ give access to loaded traces and sessions
  ▼
AppController
  ├── StateManager and AppState
  ├── UIModel and UIElements
  ├── TokenColor
  ├── WorkManager
  └── module_pipelines[ModuleID]
          │
          ├── Pipeline
          ├── Assembler
          ├── Request, Job, and Result types
          └── Bokeh chart surface
```

`AppController` is the main application boundary.

The controller owns the trace registry, state manager, UI model, work manager,
shared services, display layout, and pipeline lifecycle.

A module must not own global state. A module must not control application
lifecycle.

## Startup

`main.py` starts the application.

It does these tasks:

1. It reads the trace-file paths.
2. It opens a `TraceSession` for each trace file.
3. It creates a `TraceRecord` for each session.
4. It creates `AppController`.
5. It adds the Bokeh root to the current Bokeh document.

The project uses a `src/` package layout.

```text
pyproject.toml
uv.lock
src/
  blup/
    __init__.py
    main.py
    controller.py
    state.py
    ui.py
    utils.py
    traces/
    modules/
```

Use uv to start the Bokeh application:

```bash
uv run bokeh serve src/blup --show --args TRACE [TRACE ...]
```

The package must install from `src/`. This lets Bokeh use imports such as:

```python
from blup.controller import AppController
```

Put debug tools in `utils.py`, or in a future debug module. Do not put debug
tools in `main.py`.

## Trace Access

### Trace record

`TraceRecord` describes one loaded trace.

```text
TraceRecord
├── trace_id
├── label
└── session: TraceSession
```

`trace_id` identifies the trace in application state.

`label` gives the trace a name for the user interface.

`session` gives access to trace data.

### Trace registry

`TraceRegistry` owns all `TraceRecord` objects.

The registry is the supported way to find loaded traces. Other application
components must use `TraceRegistryAccess`. They must not read a controller
dictionary of traces directly.

The registry can:

- Get all loaded trace IDs.
- Check if a trace ID is valid.
- Get a trace record.
- Get one or more sessions.
- Give trace options for UI controls.
- Get thread names for selected traces.
- Get time bounds for selected traces.

### Trace session

`TraceSession` owns one Pallas trace object.

`TraceSession` uses query and result types from `data_model.py`.

`TraceSession` does these tasks:

- Open one trace.
- Build trace metadata.
- Build token indices.
- Cache query results.
- Call native Pallas queries.
- Convert native results to application result bundles.
- Apply token-mode changes when necessary.

`TraceSession` must not know about Bokeh.

`TraceSession` must not change application state.

`TraceSession` must not coordinate other traces.

## Application State

### State tree

`AppState` is the source of user-visible application state.

`AppState` is immutable. Code must create a new state object for each state
change.

```text
AppState
├── display
│   ├── center: PanelState
│   ├── left: PanelState | None
│   └── right: PanelState | None
├── context
│   ├── traces
│   │   ├── trace_ids: tuple[TraceID, ...]
│   │   └── focus_id: TraceID | None
│   ├── active_threads: tuple[str, ...]
│   ├── token_mode
│   ├── selection
│   │   └── token: tuple[int, int] | None
│   └── time_scope
│       ├── t0_ns: int | None
│       └── t1_ns: int | None
└── modules
    ├── time_profile
    │   ├── presentation
    │   ├── n_bins
    │   ├── fidelity
    │   └── order
    ├── token_detail
    │   ├── chart_mode
    │   ├── table_mode
    │   ├── show_stats
    │   └── show_chart
    └── inspector
```

`context.traces.trace_ids` contains the selected traces.

The application does not use `primary_trace_id`, `secondary_trace_id`, or
`trace_mode` in state.

`focus_id` identifies the selected trace that a view uses as its main trace.
If `focus_id` is not valid, the application uses the first selected trace.

### State patches

Use patch objects to change state.

```text
DisplayPatch
ContextPatch
ModulePatch
```

Use nested patches to change nested state.

```text
TraceSelectionPatch
TokenSelectionPatch
TimeScopePatch
TimeProfilePatch
TokenDetailPatch
```

Do not change fields in `AppState` directly.

Use the controller state-update method.

Example:

```python
host.update_state(
    context=ContextPatch(
        selection=TokenSelectionPatch(
            token=token,
        ),
    ),
)
```

### State normalization

`StateManager` applies patches and checks the resulting state.

`StateManager` makes these checks:

- Selected trace IDs must exist in the trace registry.
- If no selected trace exists, select the first loaded trace.
- `focus_id` must be in `trace_ids`.
- Active thread names must exist in the selected traces.
- If no active thread is selected, select all available threads.

When trace selection changes, `StateManager` clears:

- Token selection.
- Time scope.

The controller refreshes the display only when the state changes.

## User Interface

`UIModel` creates Bokeh controls.

`UIElements` gives access to the created controls.

The UI shows application state. The UI is not a second source of state.

The trace selector is a Bokeh `MultiSelect`.

Its selected values map to:

```python
state.context.traces.trace_ids
```

A trace-selection callback creates a state patch:

```python
ContextPatch(
    traces=TraceSelectionPatch(
        trace_ids=tuple(new),
    ),
)
```

The controller gets available thread names from:

```python
trace_registry.thread_names_for(
    state.context.traces.trace_ids,
)
```

The controller updates widget values after state normalization.

A Bokeh callback can occur during widget synchronization. The state manager
ignores the update if it causes no state change.

## Controller

`AppController` coordinates the main application components.

It owns:

- The trace registry.
- The state manager.
- The UI model.
- The work manager.
- Module pipelines.
- Display layout.
- Shared services.

The controller must not contain chart drawing code.

The controller must not contain low-level trace query code.

The controller has this main lifecycle:

```text
build()
├── build module roots
├── build UI shell
├── mount current displays
├── bind active pipelines
└── refresh active pipelines
```

The controller refreshes after a valid state change.

A pipeline can bind more than one time. Pipeline binding must be safe if it
runs again.

A pipeline must use a guard for callbacks that must be added only one time.
For example, a pipeline can bind Bokeh range callbacks only one time.

A pipeline can replace a chart callback reference on each bind. This action is
safe because it replaces one reference. It does not add another callback.

## Pipelines

A pipeline controls one module display.

A pipeline owns:

- One reusable Bokeh root.
- One chart surface.
- One assembler.
- Update request state.
- Chart callback bindings.

A pipeline does these tasks:

- Build its Bokeh root.
- Connect chart events to controller state updates.
- Read controller state.
- Make an update context.
- Make a work request.
- Send the request to `WorkManager`.
- Apply results on the Bokeh document thread.

A pipeline must not change `AppState` directly.

### Time profile pipeline

`TimeProfilePipeline` controls the time-profile chart.

It uses this update sequence:

```text
refresh(host)
  ↓
prepare_update(host)
  ↓
assembler.prepare_request(update)
  ↓
WorkManager.submit(UIWorkRequest(...))
  ↓
start_update()
  ↓
assembler.run_job(job) in a worker thread
  ↓
apply_result(result) in the Bokeh document thread
  ↓
finish_update(cancelled=...)
```

The time-profile pipeline binds time-range callbacks one time.

The callbacks combine multiple range changes into one next-tick update. The
pipeline checks the range, limits it to trace time bounds, and sends a time
scope patch through the controller.

The state model supports more than two selected traces.

The current time-profile chart still uses upper and lower trace sides. It does
not yet display all selected traces as separate visual lanes.

## Work Manager

`WorkManager` runs background jobs.

It keeps Bokeh changes on the Bokeh document thread.

A work request contains:

```text
scope_key
request_key
priority
request_kind
exec_kind
```

`scope_key` identifies the display or task group.

`request_key` identifies the request data.

If a new request has the same scope and request key, `WorkManager` uses the
active request.

If a new request has the same scope and a different request key,
`WorkManager` cancels the old request.

A cancelled request can finish in a worker thread. `WorkManager` ignores its
result.

The worker puts each result in a thread-safe queue.

`WorkManager` schedules a Bokeh document callback to drain the queue.

`Pipeline.apply_result()` runs in that Bokeh document callback.

### Thread lanes

`WorkManager` has two active thread lanes:

```text
Reserved lane
  UI update jobs

Shared lane
  Other UI update jobs
  Background jobs
```

UI work requests use threads by default.

The controller currently configures eight thread workers.

The process-work interface is for future use.

Process work is not ready for use because:

- Process dispatch is not complete.
- Current job contexts include session closures.
- Session closures cannot safely move to another process.

### CPU limitation

Many scheduled worker threads do not always use many CPU cores.

Python uses the Global Interpreter Lock, also called the GIL.

The native Pallas query can hold the GIL during its calculation. If it holds
the GIL, Python worker threads cannot run that calculation at the same time.

For this reason, process CPU use can stay near one CPU core even when the
application has many worker threads.

Future native work must:

- Copy Python arguments to native C++ data.
- Release the GIL only around native C++ work.
- Reacquire the GIL before code uses Python objects.
- Check that concurrent reads of `GlobalArchive` and `Thread` are safe.
- Test calls on different trace sessions first.
- Test calls on one shared trace archive after the first test.
- Use ThreadSanitizer when practical.

## Data Flow

### Query and result flow

```text
TraceSession
  ↓ normalized domain result
Assembler
  ↓ module job and result
WorkManager worker thread
  ↓ result queue
Bokeh document callback
  ↓
Pipeline.apply_result()
  ↓
Chart surface and ColumnDataSource update
```

### Interaction flow

```text
Bokeh control or chart event
  ↓
Controller callback or pipeline callback
  ↓
StateManager.update(patch)
  ↓
Valid AppState
  ↓
Controller refresh
  ↓
Pipeline work request
```

### Data levels

The application uses three data levels.

- **Level 1: Metadata and indices.** This data includes trace bounds, thread
  names, token metadata, and token mappings. `TraceSession` and
  `TraceRegistry` own this data.
- **Level 2: Small query results.** This data includes summaries and other
  compact results. The application can use this data for controls, state, and
  requests.
- **Level 3: Large display results.** This data includes quanta, spans, and
  other large result bundles. Worker threads produce this data. Python code
  must do as little processing as possible on this data.

Fidelity mode defines the trade between response time and result accuracy.

Fidelity mode is part of query contracts and module state. It is not an
uncontrolled chart setting.

## Module Boundaries

### `data_model.py`

`data_model.py` defines:

- Query types.
- Result bundle types.
- Fidelity modes.
- Token modes.
- Normalization helpers.
- Data caches.

`data_model.py` must not use Bokeh.

`data_model.py` must not control application state.

### `traces/`

The `traces/` directory contains trace access code.

```text
traces/
├── interface.py
├── registry.py
└── session.py
```

`interface.py` defines `TraceRecord` and `TraceRegistryAccess`.

`registry.py` manages loaded trace records.

`session.py` manages one trace and its data queries.

### `state.py`

`state.py` defines:

- Immutable state types.
- State patch types.
- `StateManager`.

`state.py` checks and normalizes state against the trace registry.

`state.py` must not create Bokeh controls.

`state.py` must not run heavy trace queries.

### `controller.py`

`controller.py` coordinates state, traces, UI, displays, pipelines, and
background work.

`controller.py` must not contain native Pallas query code.

`controller.py` must not contain detailed chart update code.

### `ui.py`

`ui.py` creates Bokeh controls and layout containers.

`ui.py` sends control changes to controller callbacks.

`ui.py` must not own application state.

### `modules/`

The `modules/` directory contains module code.

```text
modules/
├── interface.py
└── time_profile/
    ├── assembler.py
    ├── chart.py
    ├── pipeline.py
    └── types.py
```

`modules/interface.py` defines common pipeline, assembler, request, job, and
work-manager types.

A module contains only its display behavior and its data-shaping behavior.

### `utils.py`

`utils.py` contains general helper functions.

It can contain timing tools and temporary debug tools.

It must not start the application.

It must not contain Bokeh application setup.

## Code Rules

Use public names for supported class interfaces.

Use names that start with `_` for internal fields and methods.

Do not use internal names outside their class unless the design document
explicitly permits this use.

Do not change frozen state objects.

Do not update Bokeh models from a worker thread.

A worker job must return data only.

Only the Bokeh document thread can update Bokeh models.

Use `TraceRegistryAccess` and `TraceSession` methods for trace access.

Do not access trace data through controller implementation details.
