# Linked Vectors

## Purpose

Linked vectors are the runtime containers used for long metric streams in Pallas.

- event timestamps
- sequence timestamps
- inclusive durations
- exclusive durations

Construction happens in `pallas_write.cpp:104`, `pallas_write.cpp:759`, and
`pallas_write.cpp:830`.

## Mental Model

```terminal
Logical view seen by callers
  value[0] value[1] value[2] value[3] value[4] ...

Physical view used internally
  +-----------+    +-----------+    +-----------+
  | SubArray0 | -> | SubArray1 | -> | SubArray2 | -> ...
  +-----------+    +-----------+    +-----------+
     0..511           512..1023        1024..1535
```

The public contract stays vector-like:

- `add(value)` appends one logical value
- `at(i)` reads one logical value with bounds checks
- `operator[](i)` reads one logical value on the fast path

The storage layout behind that contract is free to change.

## Why This Exists

Pallas does not want one giant flat array per metric stream.

- writes are append-heavy
- traces can contain millions of values
- older chunks may be written out and reloaded later
- different chunks may use different internal encodings
- reads still need index-based access

## Main Types

```terminal
LVBase
├── TimeLinkedVector
└── DurationLinkedVector

SubArrayBase
├── TimeSubArray
└── DurationSubArray
```

`LinkedVectorBase` is declared in `pallas_linked_vector.h:106`.
`SubArrayBase` is declared in `pallas_subarray.h:308`.

---

## Logical And Physical View

### Logical

The rest of the runtime sees one ordered stream.

```terminal
timestamps = [ t0 t1 t2 t3 t4 ... ]
durations  = [ d0 d1 d2 d3 d4 ... ]
```

### Physical

The implementation stores that stream in pieces.

```terminal
LVBase
  first -------------------------------> last
    |                                     |
    v                                     v
  [Sub0] <-> [Sub1] <-> [Sub2] <-> ... <-> [SubN]
```

Each piece is a `SubArrayBase`-derived object.

### Why The Split Matters

- append can stay local to the tail subarray
- old subarrays can be unloaded without deleting the structure
- read path can locate one chunk, then decode locally
- codec-specific logic stays below the vector interface

## Core Responsibilities of `LinkedVectorBase`

`LinkedVectorBase` is the common container API used by `TimeLinkedVector` and `DurationLinkedVector`.

- owns `first`, `last`, `value_count`, and `subarray_index`
- serves `add()`, `at()`, `operator[]`, `front()`, and `back()`
- tracks loaded subarrays and recent-access helpers
- owns `hbuffer` in `pallas_linked_vector.h:125` for codec scratch space
- writes the common linked-vector header in `pallas_storage.cpp:830`

In the write path, `ThreadWriter::getOrCreateSequenceFromArray()` in
`pallas_write.cpp:104` creates:

- `TimeLinkedVector` for `sequence.timestamps`
- `DurationLinkedVector` for `sequence.durations`
- `DurationLinkedVector` for `sequence.exclusive_durations`

## Specialized Roles of `TimeLinkedVector` and `DurationLinkedVector`

Both inherit `LinkedVectorBase`, but they fix different value domains.

- `TimeLinkedVector` sets `ValueDomain::Timestamp` in `pallas_linked_vector.cpp:380`
- `DurationLinkedVector` sets `ValueDomain::Duration` in `pallas_linked_vector.cpp:543`

That domain choice drives subarray type, manager choice, and codec behavior.

### `TimeLinkedVector`

- used for timestamp streams
- creates `TimeSubArray`
- exposes timestamp helpers such as `getWeights()` and `getFirstOccurrenceBefore()`
- relies mainly on common vector metadata plus timestamp subarray headers

### `DurationLinkedVector`

- used for duration streams
- creates `DurationSubArray`
- keeps linked-vector stats: `min_duration`, `max_duration`, `mean_duration`
- writes extra duration metadata in `pallas_storage.cpp:984`

### Common + Specialized Headers

```terminal
LinkedVectorBase header
  value_count
  subarray_total
  storage_policy

then

TimeLinkedVector
  no extra vector-level fields

DurationLinkedVector
  min_duration
  max_duration
  mean_duration
```

Reconstruction follows the same split in `pallas_storage.cpp:910` and
`pallas_storage.cpp:1020`.

## Auxiliary Lookup Structures

Indexed reads should not linearly walk the full subarray chain on every access.

- `subarray_index` in `pallas_linked_vector.h:120` stores subarray pointers in
  logical order so `LinkedVectorBase::find_subarray()` can use binary search as the main
  lookup path.
- `recent_subarrays` in `pallas_linked_vector.h:119` is a tiny hot cache for
  recently used chunks, which helps when analysis code performs repeated local or
  sequential accesses.
- `recent_values` in `pallas_linked_vector.h:118` is a small ring buffer for very
  recent logical values, avoiding repeated decode work on immediate re-reads.

```terminal
read(i)
  -> recent value ring buffer?
  -> recent subarray cache?
  -> binary search in subarray_index?
  -> decode inside one subarray
```

These structures exist to make read-heavy analysis faster, especially when access
patterns are local, sequential, or otherwise likely to punish naive pointer chasing.

---

## What a `SubArray` Represents

A `SubArray` is one physical chunk of one logical stream.

- it is not a separate vector
- it covers a bounded logical range
- it is the unit of storage, loading, and eviction
- it is also the unit that owns one active manager

```terminal
Linked vector:
  [ logical stream ......................................... ]

Broken into chunks:
  [Sub0] [Sub1] [Sub2] [Sub3] ...
```

## The Metadata Carried by a `SubArray`

Important `SubArrayBase` metadata lives in `pallas_subarray.h:311`.

### Position

- `first_index`: where this chunk starts in the logical stream
- `value_count`: how many logical values it currently covers

### Storage

- `physical_size`: current in-memory payload size
- `file_offset`: where the persisted payload lives in the data file
- `subarray_phase`: runtime-write vs file-backed state

### Policy

- `storage_policy`: `None`, `Delta`, or generic `Lossy`
- `lossy_storage_policy`: resolved lossy variant
- `manager`: object that performs encode/decode logic

### Links

- `next` / `prev`: chain neighboring subarrays
- `parent_lv`: points back to the owning `LinkedVectorBase`

### Domain-Specific Examples

- `TimeSubArray` adds `first_timestamp` and `last_timestamp`
- `DurationSubArray` adds `min_duration`, `max_duration`, and `mean_duration`

---

## What a Manager Is Conceptually

The base `Manager` class in `pallas_subarray.h:95` is the object that actually
decides how a subarray behaves.

```terminal
LVBase
  -> SubArrayBase
       - logical range
       - buffer
       - file offset
       - links
       - policy tags
       -> Manager
            - add()
            - at()
            - write_data()
            - load_data()
```

From a design point of view, `SubArrayBase` is mostly a shell that owns placement
metadata and one buffer, while the `Manager` controls what happens to that data.

- `SubArrayBase` forwards append, indexed reads, serialization, and reload work
- `Manager` implements the storage-policy-specific logic behind that contract
- this is what allows `None`, `Delta`, `PLA`, and `DurationSpike` to share one
  common subarray shape while changing encoding and decoding behavior underneath

In short, subarrays own the data region; managers decide how that region is filled,
interpreted, persisted, and reconstructed.

- When and why a new subarray is created

- Why manager logic is separated from vector logic

- Manager families: `None`, `Delta`, `PLA`, and `DurationSpike`

- How manager selection depends on value domain and storage policy

- Generic lossy policy versus domain-specific codec resolution

- Differences between timestamp storage and duration storage

- How duration statistics are maintained and exposed

- How subarray payloads are written to disk

- What belongs in metadata headers versus data files

- On-demand loading of subarray payloads

- Eviction and freeing of loaded payloads

- Why subarray structure remains alive even after payload unloading

- Interaction between linked vectors and archive or thread storage code

- Runtime policy changes and policy application

- Hot-loop promotion in the write path

- Benchmarking hooks around vector and subarray operations

- Known limitations and non-goals of the current design
