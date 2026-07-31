# Architecture Design-Doc

## System Overview

```text
TraceSession(s)
    │
    │ domain queries and normalized results
    ▼
data_model.py
    │
    ▼
AppController
    ├── AppState
    ├── UIModel / UIElements
    ├── BokehAppShell
    ├── TokenColor
    ├── WorkManager
    └── module_pipelines[ModuleID]
            │
            ├── Pipeline
            ├── Assembler
            ├── Request / Job / Result contracts
            └── Bokeh Surface
```

The controller is the application integration boundary. It owns loaded traces,
shared state, shared services, the Bokeh shell, and the active module pipeline
registry.

The application has two UI layers:

```text
shell/
    Defines stable shell layout concepts and dimensions.
    Does not import Bokeh.

bokeh/
    Defines Bokeh layouts, themes, panel frames, and Bokeh-specific styling.
    Imports shell layout concepts.
```

The current shell is deliberately simple. It provides a static title bar,
context panel, center analysis panel, inspector panel, and status footer.
Dynamic docking, menus, persistent layouts, and flexible panel placement are
not part of the first shell implementation.

## Data Flow

### Results Path

```text
TraceSession(s)
    ↓
data_model.py
    ↓
AppController
    ↓
Pipeline
    ↓
Assembler
    ↓
Bokeh Surface
```

Trace sessions return normalized domain results defined by `data_model.py`.
Pipelines request data through the controller, assemblers convert domain
results to view-ready data, and Bokeh surfaces render the prepared result.

Heavy trace data should remain on the worker path. Bokeh views should receive
already prepared display data and should not perform large analysis operations.

### Interaction Path

```text
Bokeh widget event
    ↓
AppController callback
    ↓
AppState update
    ↓
UI control synchronization
    ↓
Scheduled refresh
    ↓
Active pipeline refresh
```

The first shell implementation keeps existing controller callback methods.
Widgets call controller event adapters directly. This avoids changing the
trace-query, pipeline, and work-manager interfaces while the visual shell is
being introduced.

Later, Bokeh callbacks may emit shell commands rather than call controller
methods directly. This is not required for the initial visual prototype.

## Data Classification

Data is organized by lifetime, size, and responsibility.

- Tier 1: Trace metadata and token information. This lives in `TraceSession`
  objects and is small enough for normal Python processing.
- Tier 2: Aggregated summary data. This moves between trace sessions,
  controller logic, and views when required. It should remain limited in size.
- Tier 3: Large result streams. These are prepared by worker jobs and passed
  to Bokeh surfaces with minimal additional Python-side processing.

Data is also organized by fidelity. Different query paths may trade accuracy
for latency. Higher-fidelity data may be prepared in the background and
replaced when it becomes available.

## Application State

`AppState` is the source of truth for mutable user-visible state. State is
separated by scope so that shared analysis context, module settings, and
display placement remain distinct.

```text
AppState
├── display
│   ├── center
│   │   ├── active_module
│   │   └── context_key
│   ├── left
│   │   ├── active_module
│   │   └── context_key
│   └── right
│       ├── active_module
│       └── context_key
├── context
│   ├── traces
│   │   ├── trace_ids
│   │   └── focus_id
│   ├── active_threads
│   ├── token_mode
│   ├── selection
│   │   └── token
│   └── time_scope
│       ├── t0_ns
│       └── t1_ns
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

Trace selection is represented by `context.traces.trace_ids`. The user selects
one or more traces through a single trace `MultiSelect`; the UI does not own
separate primary and secondary trace selector widgets.

`StateManager` owns state normalization and validation. It validates selected
traces against the trace registry, removes unavailable threads, applies
defaults when necessary, and resets dependent selection state when trace
selection changes.

The current `display` state still uses center, left, and right module slots.
This remains a compatibility model for the current controller and pipeline
mounting flow. It should not be expanded into a general docking model during
the first shell implementation.

## Shell Layout

The shell is the persistent frame around analysis modules. It is independent
from trace queries and module-specific data preparation.

```text
Bokeh document root
    │
    └── BokehAppShell.root
          ├── title/header
          ├── body
          │     ├── Context panel
          │     ├── Analysis panel
          │     └── Inspector panel
          └── status footer
```

The visible shell structure is:

```text
┌──────────────────────────────────────────────────────────────────┐
│ PALLAS / TRACE LAB                                      ● READY   │
├──────────────┬───────────────────────────────────┬───────────────┤
│ CONTEXT      │ ANALYSIS                          │ INSPECTOR     │
│              │                                   │               │
│ traces       │ active module Bokeh surface       │ placeholder   │
│ threads      │                                   │ or module UI  │
│ profile      │                                   │               │
├──────────────┴───────────────────────────────────┴───────────────┤
│ READY · NO ACTIVE JOBS                                            │
└──────────────────────────────────────────────────────────────────┘
```

The initial shell does not support moving, hiding, resizing, or persisting
panels. It is a visual and structural foundation only.

### Shell Dimensions

`shell/layout.py` contains stable layout constants in `ShellDimensions`.

```python
@dataclass(frozen=True)
class ShellDimensions:
    context_width: int = 280
    context_min_width: int = 220

    inspector_width: int = 320
    inspector_min_width: int = 260

    header_height: int = 52
    footer_height: int = 26

    panel_gap: int = 8
    panel_padding: int = 12
```

Sidebar widths are preferred dimensions, not a fixed total-page width.
Context and inspector panels have minimum widths. The center analysis host
uses the remaining available space.

The Bokeh root and body use `sizing_mode="stretch_both"`. The context and
inspector hosts use `stretch_height`; the center host uses `stretch_both`.
This allows the shell to fill normal desktop browser windows without a fixed
combined layout width.

At very narrow window sizes, three visible panels may still not fit. The
initial shell does not yet define a mobile or panel-hiding policy.

### Shell Ownership

`shell/` contains layout concepts that are not tied to Bokeh:

- Preferred and minimum panel dimensions.
- Future shell panel identifiers and layout state.
- Future commands for docking, collapsing, or resizing panels.

`bokeh/` contains Bokeh implementation code:

- `BokehAppShell` and its title, body, panel hosts, and status footer.
- Panel-frame and panel-title helpers.
- Bokeh layout construction using `row`, `column`, and `LayoutDOM`.
- Gruvbox palette, Bokeh `Theme`, and Bokeh-specific style helpers.
- Future Bokeh event adapters and browser-facing controls.

Dependency direction:

```text
shell/  ←  bokeh/
```

`shell/` must not import Bokeh. `bokeh/` may import shell dimensions and later
shell state types.

## UI Compatibility Layer

`ui.py` is currently a compatibility layer between `AppController` and the
new Bokeh shell.

```text
UIModel
    ├── builds trace and thread context controls
    ├── registers existing controller callbacks
    ├── mounts persistent controls into shell.context_host
    └── returns UIElements for controller compatibility
```

`UIElements` exposes the root, persistent widgets that the controller must
synchronize, and mutable module mount hosts.

```text
UIElements
├── root
├── title
├── status
├── trace_select
├── thread_select
├── token_mode_select
├── n_bins_spinner
├── quanta_mode_select
├── stack_order_select
├── highlight_token_select
├── center_panel
├── left_panel
└── right_panel
```

The trace selector is a `MultiSelect` bound to
`state.context.traces.trace_ids`. The controller updates its options and
selected values after trace state changes.

The time-profile widgets remain in `UIElements` during this transition so
existing controller callbacks can remain unchanged. They are not permanent
global-shell controls. They should later move to a controls surface owned by
the time-profile module.

The controller mounts active pipeline roots by replacing the `children` list
of `center_panel`, `left_panel`, and `right_panel`. Persistent context and
inspector content must not share those legacy module mount hosts, because a
pipeline mount would replace their children.

## Global Modules

### `data_model.py`

Defines normalized domain types and query/result contracts shared by trace
access code and higher application layers.

Responsibilities:

- Define immutable query contracts.
- Define normalized result containers.
- Remain independent of Bokeh and UI concerns.

### `traces/`

Trace code provides access to one or more trace sessions.

`TraceSession` wraps one underlying trace and executes domain queries using
`data_model.py` contracts. The trace registry owns the loaded-session
collection and provides trace options, valid trace IDs, and trace metadata.

Trace code must not own application state, Bokeh models, or display layout.

### `state.py`

Defines `AppState`, state branches, state patches, and validation rules.

Responsibilities:

- Define the central application state tree.
- Keep shared context separate from module-specific settings.
- Apply patches and normalize invalid selections.
- Remain independent of Bokeh models and callback objects.

### `controller.py`

The controller coordinates state, trace access, pipelines, Bokeh UI assembly,
and worker scheduling.

Responsibilities:

- Create initial application state.
- Own or access loaded traces and the trace registry.
- Build module roots and the UI shell.
- Mount active pipeline roots into UI panel hosts.
- Synchronize persistent UI controls after state changes.
- Schedule refresh work and bind active pipelines.
- Route Bokeh widget events to state updates.

Constraints:

- Must not contain module-specific rendering logic.
- Must not contain low-level trace query implementation.
- Must not construct detailed Bokeh shell layouts directly; that belongs in
  `bokeh/` and the temporary `ui.py` adapter.

### `work_manager.py`

The work manager executes background jobs and serializes result application
back onto the Bokeh display thread.

Responsibilities:

- Accept UI and background work requests.
- Cancel stale requests by scope and request key.
- Run jobs in worker pools.
- Schedule result delivery through the Bokeh document callback path.

Constraints:

- It manages background execution only.
- Bokeh model mutation must occur on the display-thread path.

### `modules/`

A module is an application feature such as `time_profile` or `token_detail`.

A module pipeline owns its feature behavior, reusable Bokeh surface, display
refresh logic, and request lifecycle. A pipeline is activated by module ID and
is mounted by the controller into a shell host.

Module pipelines should not own global shell geometry, global trace selection,
or Bokeh document registration.

## Module Interfaces

### `Pipeline`

A pipeline is the controller-facing unit of display behavior.

Responsibilities:

- Build a reusable Bokeh root.
- Bind to the controller when active.
- Convert current application state into update requests.
- Submit background work through the work manager.
- Apply completed results to its own surface.

### `Assembler`

An assembler translates a module update into requests, worker jobs, and
view-ready results.

Responsibilities:

- Prepare a request from module update state.
- Build jobs for the request.
- Execute worker jobs.
- Return prepared results for a pipeline surface.

### `Surface` and Views

A surface is a Bokeh-facing rendering object. It owns figures, data sources,
glyph renderers, and immediate display updates.

Responsibilities:

- Build Bokeh figures and layouts.
- Keep stable Bokeh model references.
- Update Bokeh sources from prepared result data.
- Handle display-local Bokeh behavior.

Constraints:

- Must not execute trace queries.
- Must not mutate global application state directly.
- Must not apply expensive data analysis after result preparation.

## Theme and Styling

`bokeh/theme.py` defines the shared Gruvbox-inspired palette and creates the
document-level Bokeh `Theme`.

```text
Palette
├── neutral backgrounds: bg0, bg1, bg2, bg3
├── foreground text: fg0, fg1, fg2, muted
└── semantic accents: yellow, orange, red, green, blue, purple
```

The document theme applies shared defaults to Bokeh plots, titles, axes,
grids, and legends. It is applied once in `main.py` before application roots
and module figures are built.

`bokeh/styles.py` provides visual helpers for shell content:

- `panel_title()` creates a large panel header.
- `panel_frame()` creates a bordered panel with a title and content.
- `status_text()` creates the shell footer content.
- `section_label()` may be added for smaller internal control sections.

The shell uses warm neutral backgrounds, one-pixel borders, square geometry,
compact spacing, and monospace system labels. Module charts may use semantic
data colors, but common shell and plot-neutral colors should come from the
shared palette.

## Startup and Document Setup

`main.py` is the Bokeh document composition entry point.

```text
main.py
    ├── load trace files
    ├── create TraceSession objects
    ├── apply document Bokeh theme
    ├── create AppController
    ├── build controller root
    ├── set root sizing mode
    └── add root to curdoc()
```

The document setup sequence is:

```python
doc = curdoc()
doc.theme = gruvbox_bokeh_theme()
doc.title = "Pallas Trace Lab"

controller = AppController(loaded)
root = controller.build()

root.sizing_mode = "stretch_both"
doc.add_root(root)
```

`BokehAppShell` constructs the root layout. `main.py` registers that completed
root with the Bokeh document. The shell must not call `doc.add_root()` itself.

## File Layout

```text
blup/
├── main.py
├── controller.py
├── data_model.py
├── state.py
├── colors.py
├── types.py
├── utils.py
│
├── shell/
│   ├── __init__.py
│   └── layout.py
│
├── bokeh/
│   ├── __init__.py
│   ├── app_shell.py
│   ├── theme.py
│   └── styles.py
│
├── ui.py
│
├── traces/
│   ├── __init__.py
│   ├── interface.py
│   ├── registry.py
│   └── session.py
│
└── modules/
    ├── __init__.py
    ├── interface.py
    ├── work_manager.py
    ├── time_profile/
    │   ├── __init__.py
    │   ├── pipeline.py
    │   ├── assembler.py
    │   └── chart.py
    └── token_detail/
        ├── __init__.py
        ├── pipeline.py
        ├── assembler.py
        └── surface.py
```

The exact module tree may evolve. The important boundaries are:

```text
traces/       trace-backed data access
data_model.py shared domain contracts
state.py      application state and validation
controller.py application coordination
shell/        framework-neutral shell concepts
bokeh/        Bokeh shell and visual implementation
ui.py         temporary controller compatibility adapter
modules/      feature-specific display behavior
```

## Planned Shell Evolution

The following are later work, not requirements for the initial shell:

- Move time-profile controls out of `ui.py` and into a module-owned controls
  surface.
- Replace fixed display slots with explicit shell layout state.
- Add an inspector contribution API for active modules.
- Add status updates for running jobs, errors, and exports.
- Add shell commands for showing, hiding, collapsing, and moving panels.
- Add saved shell layouts and theme selection.
- Add responsive small-window behavior, such as hiding side panels below a
  defined width.
- Replace direct controller widget callbacks with explicit command dispatch.
- Replace Bokeh shell rendering with another frontend implementation if
  required, while preserving `shell/` state and module contracts.

## Conventions

Use public names only for supported class interfaces. Other classes may depend
only on documented public names.

Use one leading underscore for internal fields and methods. Do not use private
names outside their owning class except for a documented exception.

Use a public method for an action or calculated value. Use a public field only
when it is a stable, direct part of the object interface.

Keep Bokeh objects, callback guards, caches, background-work state, and
temporary layout references private unless the controller must intentionally
synchronize or mount them.
