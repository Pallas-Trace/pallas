/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
/** @file
 * Internal serialisation helpers shared by storage-related implementation
 * code. This is not part of the stable public API surface.
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

