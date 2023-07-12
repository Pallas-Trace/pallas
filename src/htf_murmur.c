/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

//-----------------------------------------------------------------------------
// MurmurHash3 was written by Austin Appleby, and is placed in the public
// domain. The author hereby disclaims copyright to this source code.
// This is actually the C port of the Murmur3 Hash algorithm, which I have further modified
// So that it can hash efficiently the sequences.
// The OG C implementation can be found here: https://github.com/PeterScott/murmur3
#include "htf_murmur.h"

//-----------------------------------------------------------------------------
// Platform-specific functions and macros

#ifdef __GNUC__
#define FORCE_INLINE __attribute__((always_inline)) inline
#else
#define FORCE_INLINE inline
#endif

static FORCE_INLINE uint64_t rotl64(uint64_t x, int8_t r) {
  return (x << r) | (x >> (64 - r));
}

#define ROTL64(x, y) rotl64(x, y)

#define BIG_CONSTANT(x) (x##LLU)

//-----------------------------------------------------------------------------
// Block read - if your platform needs to do endian-swapping or can only
// handle aligned reads, do the conversion here

#define getblock(p, i) (p[i])

//-----------------------------------------------------------------------------
// Finalization mix - force all bits of a hash block to avalanche

static FORCE_INLINE uint64_t fmix64(uint64_t k) {
  k ^= k >> 33;
  k *= BIG_CONSTANT(0xff51afd7ed558ccd);
  k ^= k >> 33;
  k *= BIG_CONSTANT(0xc4ceb9fe1a85ec53);
  k ^= k >> 33;

  return k;
}

//-----------------------------------------------------------------------------
void htf_murmur_hash3_x64_32(const void* key, const int len, const uint32_t seed, uint32_t* out) {
  uint32_t buffer[2];
  htf_murmur_hash3_x64_64(key, len, seed, (uint64_t*)buffer);
  *out = buffer[0];
}

void htf_murmur_hash3_x64_64(const void* key, const int len, const uint32_t seed, uint64_t* out) {
  // Here's the issue: We'll be feeding it an array of htf_token
  // An htf_token is 32 bits long
  // So either we change every uint8_t here to a uint32_t
  // But that might lead to some unforeseen consequences
  // Or ! We simply take that into account, and change the "true_len" variable
  // For now we'll try the second one, and see what it does.
  const int true_len = len * (sizeof(htf_token_t) / sizeof(uint8_t));
  const uint8_t* data = (const uint8_t*)key;
  const int nblocks = true_len / 16;
  int i;

  uint64_t h1 = seed;
  uint64_t h2 = seed;

  uint64_t c1 = BIG_CONSTANT(0x87c37b91114253d5);
  uint64_t c2 = BIG_CONSTANT(0x4cf5ad432745937f);

  //----------
  // body

  const uint64_t* blocks = (const uint64_t*)(data);

  for (i = 0; i < nblocks; i++) {
    uint64_t k1 = getblock(blocks, i * 2 + 0);
    uint64_t k2 = getblock(blocks, i * 2 + 1);

    k1 *= c1;
    k1 = ROTL64(k1, 31);
    k1 *= c2;
    h1 ^= k1;

    h1 = ROTL64(h1, 27);
    h1 += h2;
    h1 = h1 * 5 + 0x52dce729;

    k2 *= c2;
    k2 = ROTL64(k2, 33);
    k2 *= c1;
    h2 ^= k2;

    h2 = ROTL64(h2, 31);
    h2 += h1;
    h2 = h2 * 5 + 0x38495ab5;
  }

  //----------
  // tail

  const uint8_t* tail = (const uint8_t*)(data + nblocks * 16);

  uint64_t k1 = 0;
  uint64_t k2 = 0;

  switch (true_len & 15) {
  case 15:
    k2 ^= (uint64_t)(tail[14]) << 48;
    [[fallthrough]];
  case 14:
    k2 ^= (uint64_t)(tail[13]) << 40;
    [[fallthrough]];
  case 13:
    k2 ^= (uint64_t)(tail[12]) << 32;
    [[fallthrough]];
  case 12:
    k2 ^= (uint64_t)(tail[11]) << 24;
    [[fallthrough]];
  case 11:
    k2 ^= (uint64_t)(tail[10]) << 16;
    [[fallthrough]];
  case 10:
    k2 ^= (uint64_t)(tail[9]) << 8;
    [[fallthrough]];
  case 9:
    k2 ^= (uint64_t)(tail[8]) << 0;
    k2 *= c2;
    k2 = ROTL64(k2, 33);
    k2 *= c1;
    h2 ^= k2;
    [[fallthrough]];

  case 8:
    k1 ^= (uint64_t)(tail[7]) << 56;
    [[fallthrough]];
  case 7:
    k1 ^= (uint64_t)(tail[6]) << 48;
    [[fallthrough]];
  case 6:
    k1 ^= (uint64_t)(tail[5]) << 40;
    [[fallthrough]];
  case 5:
    k1 ^= (uint64_t)(tail[4]) << 32;
    [[fallthrough]];
  case 4:
    k1 ^= (uint64_t)(tail[3]) << 24;
    [[fallthrough]];
  case 3:
    k1 ^= (uint64_t)(tail[2]) << 16;
    [[fallthrough]];
  case 2:
    k1 ^= (uint64_t)(tail[1]) << 8;
    [[fallthrough]];
  case 1:
    k1 ^= (uint64_t)(tail[0]) << 0;
    k1 *= c1;
    k1 = ROTL64(k1, 31);
    k1 *= c2;
    h1 ^= k1;
  };

  //----------
  // finalization

  h1 ^= true_len;
  h2 ^= true_len;

  h1 += h2;
  h2 += h1;

  h1 = fmix64(h1);
  h2 = fmix64(h2);

  h1 += h2;
  h2 += h1;

  // We don't need the 128 bit hash I think.
  ((uint64_t*)out)[0] = h1;
  //	((uint64_t*)out)[1] = h2;
}


/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */
