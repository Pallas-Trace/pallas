/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
/** @file
 * Internal serialisation bridge for compressed payload I/O helpers.
 *
 * This header exposes a narrow pair of helper declarations used by the
 * linked-vector and SubArray implementation to read and write persisted payload
 * data without duplicating backend compression logic in `pallas_subarray.cpp`.
 * The actual compression and decompression implementations live in the storage
 * layer, while this header makes those helpers available so SubArray read/write
 * paths can stay implemented in the subarray layer instead of being forced into
 * `pallas_storage.cpp`. This is not part of the public API surface.
 */
#pragma once

#ifdef __cplusplus

#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace pallas {
class ParameterHandler;
}

void _pallas_compress_write(uint64_t* src, size_t n, FILE* file, const pallas::ParameterHandler* parameter_handler);
uint64_t* _pallas_compress_read(size_t n, FILE* file, const pallas::ParameterHandler& parameter_handler);

#endif
