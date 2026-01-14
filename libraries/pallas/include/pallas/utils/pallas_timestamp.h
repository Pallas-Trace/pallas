/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
/** @file
 * Functions to handle timestamps.
 */
#pragma once

#ifdef __cplusplus
#include <chrono>
#include <cstdint>
typedef std::chrono::time_point<std::chrono::high_resolution_clock> Timepoint;
#else
#include <limits.h>
#include <stdint.h>
#endif
typedef uint64_t pallas_timestamp_t;
#define PALLAS_TIMESTAMP_INVALID UINT64_MAX

typedef uint64_t pallas_duration_t;
#define PALLAS_DURATION_INVALID UINT64_MAX

/** return the time difference of two events. */
inline pallas_duration_t pallas_get_duration(pallas_timestamp_t t1, pallas_timestamp_t t2) {
  return t2 - t1;
}
