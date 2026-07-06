# Overview

## Motivation

This page explains the idea behind linked vectors in Pallas.

The goal here is to first describe what a linked vector is at a conceptual level,
Why Pallas uses it, how it is implemented and what problems it is meant to solve.


## What Is A Linked Vector?

A linked vector is the internal container used by Pallas to store long streams of
performance values associated with a grammatical token such as timestamps and durations.

It should be presented as a structure that behaves like a logical vector from the
outside, while internally organizing values in smaller pieces instead of treating
the whole stream as one single flat block. This separation is useful because the
internal implementation can then be optimized for storage cost, memory usage,
loading behavior, and access time without changing the logical interface.

```text
Logical view:
  [ v0 | v1 | v2 | v3 | v4 | v5 | v6 | v7 | ... ]

High Level Physical view:
  [ v0 | v1 | v2 ] <-> [ v3 | v4 ] <-> [ v5 | v6 | v7 ] <-> ...
```

This section should explain the main intuition before discussing any detailed
storage or runtime mechanisms.

## Why Does Pallas Need It?

Pallas needs linked vectors because performance data does not behave like small,
fixed-size metadata.

- traces can contain very large value streams, so a single flat allocation quickly becomes awkward to grow and manage
- values are mostly appended during writing, so the container should make tail growth cheap and predictable
- values are often accessed by position during reading, so the logical interface should still feel like a vector
- memory usage matters, so old chunks should be easier to compress, unload, and reload than one giant contiguous block
- file-backed storage matters, so the in-memory layout should cooperate with incremental writing and on-demand reads

# Linked Vector Internals

- High-level purpose of linked vectors in Pallas
- Why Pallas does not store these value streams as one flat contiguous vector
- Logical view versus physical view
- Core responsibilities of `LVBase`
- Specialized roles of `TimeLV` and `DurationLV`
- Append-oriented growth and the runtime write path
- What a `SubArray` represents
- The metadata carried by a `SubArray`
- Logical indices versus physical storage layout
- How subarrays are linked together
- Why the vector also keeps an auxiliary lookup structure
- High-level indexed lookup path
- The role of the recent-subarray cache
- When and why a new subarray is created
- What a manager is conceptually
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
