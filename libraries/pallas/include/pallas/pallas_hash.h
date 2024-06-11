/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
/** @file
 * Hashing functions, used for hashing pallas::Sequence.
 */
#pragma once

#include "pallas.h"

#ifdef __cplusplus
/** Seed used for the hasing algorithm. */
#define SEED 17
namespace pallas {
/** Writes a 32bits hash value to out.*/
uint32_t hash32(const void* key, const size_t len, const uint32_t seed);
/** Writes a 64bits hash value to out.*/
void hash64(const void* key, size_t len, uint32_t seed, uint64_t* out);
}  // namespace pallas
#endif

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */
