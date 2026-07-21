# Architecture Design-Doc

## Data Flow

Multiple data flows exist among the application modules.

### Results path

Data 'results' are pulled from individual trace-sessions as defined by the API
in `data_model.py`, routed through the `AppController` into the requesting
pipeline which defines the requesting views data-contract with the application.
Data is subsequently shaped for the specific nature of the view inside the
adapter and served to the view file itself for rendering.

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

### Interaction path

```text
Frontend UI
    ↓
AppController
    ↓
AppState
    ↓
Pipeline(s)
```

### Data Classification

Data is organized into tiers based on its life-cycle, size, and responsibility

- Tier 1: direct trace meta-data and token information, primarily lives in
`TraceSession` objects to inform data query operations. Limited size structures
so can be highly processed/transformed.
- Tier 2: aggregated summary statistics that are directly transferred between
`TraceSession` and `AppController` to inform Application logic/state. Can also
be forwarded to views if needed. Data is limited in scope but excessive
calculation/analysis should be avoided (especially in Python).
- Tier 3: heavy result data-streams that are processed by thread workers and
\routed to views by the `AppContoller`. As these are the largest data structures
minimal processing should occur in Python code (as opposed to C++ PALLAS code)
besides app state dependent transformations (i.e. token replacement) and view
specific transformation (i.e. display preparation in adapters).

Data is also organized by 'fidelity' since the API design allows for multiple
data preparation paths featuring trade-offs of latency versus accuracy.
These differences are built into the architecture and considered different
bundles. The design is such that higher fidelity data can be prepared in the
background and substituted in as necessary (on demand or predicted need).

## Application State

Application state (i.e. mutable interaction based state differentiated from immutable
trace-session based state) is centralized in the `AppState` class (and
owned by/interfaced through the controller module) and is structured as a tree
to clarify distinct aspects of the state.

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

The work manager provides a generic interface implemented by individual
pipelines for interaction with the central application state through the
controller as well as participation in the multi-threaded data processing
job-queue system.

Responsibilities:

- Accepts data requests via `submit()`
- Cancels stale requests
- Runs jobs within the thread pool, queueing results for serialization

Constraints:

- Only manages background data processing
  - UI operations must be handled through the serial display loop in the controller

### `pipelines/`

Pipelines are the application-facing units of display behavior. Each pipeline
owns one view, knows how to refresh it from controller state, and can submit
background work when needed. The controller treats pipelines uniformly through a
common interface defined in `pipelines/base.py` and activates them by ID.

Responsibilities:

- Builds a reusable DOM root for its view that the controller can place/enable
- Binds callbacks between the view and controller
- Translates application state changes into data requests needed by the view

Constraints:

- Sits on the boundary between controller/application state and view internals
  - Should encapsulate only the necessary logic along this boundary
- Only contains high level architecture logic; data-shaping is done by adapters

## View-Specific Modules

### `adapters/`

Adapters convert domain-level query results into view-ready request specs, work
jobs, and Bokeh source dictionaries. They are the “shape translation” layer
between analysis data and rendering data.

Responsibilities:

- Translates generic data query results into view-specific form

Constraints:

- Should follow common patterns to be used interchangeably by pipelines

### `views/`

Views are thin Bokeh-facing rendering objects. They own figures, sources, glyph
renderers, and widget callbacks, but they should not own application state
transitions or trace query logic.

Responsibilities:

- Defines and initializes Bokeh rendering layout
- Updates Bokeh sources based on adapter preparation

Constraints:

- Should only contain immediate display logic
- Should not apply further processing to data; data is forwarded as-is

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
