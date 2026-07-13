# Architecture Design-Doc

## Data Flow

Results path:

```text
TraceSession(s)
    ↓
data_model.py (provides: types + query contracts)
    ↓
AppController (provides: )
    ↓
Pipeline(s) (provides: )
    ↓
Adapter(s) (provides: interface specs + result shaping)
    ↓
View(s) (provides: Bokeh figure/widget logic)
```

Interaction path:

```text
Frontend UI
    ↓
AppController
    ↓
AppState
    ↓
Pipeline(s)
```

## Application State

State Tree:

```text
AppState
   ├── views
   │   ├── *
   │   ├── *
   │ [...]
   ├── context
   │   ├── active_threads
   │   ├── token_mode
   │   ├── selection
   │   │   └── token
   │   └── time_scope
   │       ├── t0_ns
   │       └── t1_ns
   └── display
       ├── primary
       │   ├── active_view
       │   └── context_key
       └── secondary[*]
           ├── active_view
           └── context_key

```

## Global Modules

### `data_model.py`

Defines the normalized domain types and query/result contracts used between
trace access code and higher-level application layers. It is the shared schema
layer for summaries, quanta, spans, occurrences, subtrees, and histograms.

Responsibilities:

- Define immutable query contracts
- Define normalized result containers

Constraints:

- Queries/results should be a generic interface with the trace
(informed by PALLAS semantics)
- Should not contain view, UI, or Bokeh specific logic

### `trace_session.py`

Acts as the boundary around trace-backed data access and exposes query methods
against a single trace using the shared `data_model` contracts.

Responsibilities:

- Wrap (and own) the underlying trace object
- Carry out queries and return result objects
- Carry out token-level data transformations/normalization

Constraints:

- Interface must follow the specification of `data_model` module
- Responsible only for a single trace at a time
- Does not carry out application-level logic or state changes

### `state.py`

Defines the application state tree and the enums/type aliases that
constrain user-visible modes. Serves as the ground source of truth for display
layout, current analysis context, and per-view settings.

Responsibilities:

- Defines the central state-tree that all application-level logic depends on

Constraints:

- Must maintain clear and consistent scope of state-tree entries, separating
view-specific states from broader global-context state
- Other modules define when state is mutated and how views respond to state changes

### `controller.py`

The central application orchestration layer. Owns sessions, state, pipelines,
UI model, and the work manager. Acts as the main coordinator between declarative
state and imperative refresh/mount/callback behavior carried out through views.

Responsibilities:

- Creates and owns the global application state
- Wires together individual `trace_session` objects
- Instantiates and registers individual data pipelines
- Builds the application UI and display panels and routes them together
- Serializes parallel data processing into a single async view-refresh loop

Constraints:

- Should be highly modular:
  - Should not contain any view-specific render-logic
  - Should not contain any low-level trace specific data-logic

### `work_manager.py`

Responsibilities:

-

Constraints:

-

### `pipelines/`

Pipelines are the application-facing units of display behavior. Each pipeline
owns one view, knows how to refresh it from controller state, and can submit
background work when needed. The controller treats pipelines uniformly through a
DisplayPipeline-style interface and activates them by ViewId.

Responsibilities:

-

Constraints:

-

## View-Specific Modules

### `adapters/`

Adapters convert domain-level query results into view-ready request specs, work
jobs, and Bokeh source dictionaries. They are the “shape translation” layer
between analysis data and rendering data.

Responsibilities:

-

Constraints:

-

### `views/`

Views are thin Bokeh-facing rendering objects. They own figures, sources, glyph
renderers, and widget callbacks, but they should not own application state
transitions or trace query logic.

Responsibilities:

-

Constraints:

-

## File Layout

```text
main.py
data_model.py
trace_session.py
state.py
controller.py
work_manager.py
ui.py
utils.py
adapters/
  *_adapter.py
pipelines/
  base.py
  *_pipeline.py
views/
  *_view.py
```
