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
/** Seed used for the hashing algorithm. */
#define SEED 17
namespace pallas {
/** Writes a 32bits hash value to out.*/
uint32_t hash32(const byte * data, size_t len, uint32_t seed);

inline uint32_t hash32_Token(const Token* data, size_t len, uint32_t seed) {
    return hash32(reinterpret_cast<const uint8_t*>(data), len * sizeof(pallas::Token), SEED);
}
/** Writes a 64bits hash value to out.*/
uint64_t hash64(const byte* data, size_t len, uint32_t seed);
}  // namespace pallas
#endif

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */
