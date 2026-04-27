# Pallas Python API

> Before reading, it is strongly advised that you read the [introduction to the Pallas format](../../02-pallas.md).

Pallas's Python API only features the reading API.

## How to

### Loading a trace
```python
import pallas_trace as pallas
trace_name="/path/to/eztrace_log.pallas"
trace = pallas.open_trace(trace_name)
```

### Iterating over threads
```python
for process in trace.archives:
  for thread in process.threads:
    print(thread.id)

```

### Iterating over events
Per thread :
```python
for process in trace.archives:
  for thread in process.threads:
    for event, occurrence in thread:
      print(f"{event} ({occurrence}/{event.nb_occurrences})")
```

Whole trace :
```python
for thread, (event, occurrence) in trace:
  print(f"{thread} {event} ({occurrence}/{event.nb_occurrences})")
```

### Thread readers
Like in the C++ API, the Python API exposes `ThreadReader`s : a structure allowing more precise trace traversals. A `ThreadReader` can be thought of as a cursor allowing to move forward in the execution flow, it also allows moving up and down the "callstack" (in this case a stackframe would be a `Loop` or `Sequence`).

```python
class ThreadReader:
    @property
    def callstack(self) -> list[tuple]: ...
    def moveToNextToken(self, enter_sequence: bool, enter_loop: bool): ...
    def pollCurToken(self) -> tuple: ...
    def enterIfStartOfBlock(
        self, exit_sequence: bool, exit_loop: bool
    ) -> bool: ...
    def exitIfEndOfBlock(
        self, exit_sequence: bool, exit_loop: bool
    ) -> bool: ...
    def isEndOfCurrentBlock(self) -> bool: ...
    def isEndOfTrace(self) -> bool: ...
```

### Events
```python
class Event:
    def __repr__(self) -> str: ...
    def guessName(self) -> str: ...
    def getAttributes(self, occurrence: int) -> dict[str, int | float | str]: ...
    @property
    def data(self) -> dict: ...
    @property
    def id(self) -> Token: ...
    @property
    def nb_occurrences(self) -> int: ...
    @property
    def record(self) -> Record: ...
    @property
    def timestamps(self) -> Vector: ...
```

**WARNING** : Not all kinds of attributes are supported yet
