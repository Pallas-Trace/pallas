/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include <inttypes.h>
#include <malloc.h>
#include "pallas/pallas_hash.h"
#include "stdlib.h"

#define MAX_EVENT (1000000)
#define MAX_SEQUENCE (1 << 14)
#define MAX_LOOP (MAX_SEQUENCE >> 1)
#define MAX_SEQUENCE_LENGTH (1 << 8)
#define NUM_TEST_SEQUENCE (1 << 14)
#define NUM_HASHED (MAX_EVENT + MAX_SEQUENCE + MAX_LOOP + (NUM_TEST_SEQUENCE * MAX_SEQUENCE_LENGTH))
#define SIZE_COLLISION_ARRAY UINT32_MAX
using namespace std;
using namespace pallas;

int main(int argc __attribute__((unused)), char** argv __attribute__((unused))) {
  auto collisions = new short[SIZE_COLLISION_ARRAY];
  auto collision_indexes = vector<uint32_t>();
  int count_collision = 0;
  uint32_t hash[4] = {0};  // This is so that we can quickly test 64-bit and 128-bit hashing

  // Start with testing the single events and sequences and loops
  printf("Testing all the events up to %d\n", MAX_EVENT);
  struct timespec start_time;
  clock_gettime(CLOCK_MONOTONIC, &start_time);
  for (uint32_t i = 0; i < MAX_EVENT; i++) {
    Token token = Token(TypeEvent, i);
    hash[0]= hash32(&token, 1, SEED);
    uint32_t new_key = hash[0] % SIZE_COLLISION_ARRAY;
    int buffer = collisions[new_key]++;
    if (buffer) {
      count_collision++;
      if (buffer == 1) {
        collision_indexes.push_back(new_key);
      }
    }
  }
  printf("Testing all the sequences up to %d\n", MAX_SEQUENCE);
  for (uint32_t i = 0; i < MAX_SEQUENCE; i++) {
    Token token = Token(TypeSequence, i);
    hash[0] = hash32(&token, 1, SEED);
    uint32_t new_key = hash[0] % SIZE_COLLISION_ARRAY;
    int buffer = collisions[new_key]++;
    if (buffer) {
      count_collision++;
      if (buffer == 1) {
        collision_indexes.push_back(new_key);
      }
    }
  }
  printf("Testing all the loops up to %d\n", MAX_LOOP);
  for (uint32_t i = 0; i < MAX_LOOP; i++) {
    Token token = Token(TypeLoop, i);
    hash[0] = hash32(&token, 1, SEED);
    uint32_t new_key = hash[0] % SIZE_COLLISION_ARRAY;
    int buffer = collisions[new_key]++;
    if (buffer) {
      count_collision++;
      if (buffer == 1) {
        collision_indexes.push_back(new_key);
      }
    }
  }

  printf("Testing different sizes sequences, %d of them each\n", NUM_TEST_SEQUENCE);
  for (uint32_t sequence_size = 2; sequence_size < MAX_SEQUENCE_LENGTH; sequence_size++) {
    printf("\t Testing size %d\n", sequence_size);
    auto* token = new Token[sequence_size];
    for (uint32_t i = 0; i < NUM_TEST_SEQUENCE; i++) {
      // Constructing the sequence
      for (uint32_t j = 0; j < sequence_size; j++) {
        auto type = static_cast<TokenType>(1 + (rand() % 3));
        if (type == TypeEvent) {
          Token t = Token(type, rand() % MAX_EVENT);
          token[j] = t;
        }
        if (type == TypeSequence) {
          Token t = Token(type, rand() % MAX_SEQUENCE);
          token[j] = t;
        }
        if (type == TypeLoop) {
          Token t = Token(type, rand() % MAX_LOOP);
          token[j] = t;
        }
      }

      hash[0] = hash32(token, sequence_size, SEED);
      uint32_t new_key = hash[0] % SIZE_COLLISION_ARRAY;
      int buffer = collisions[new_key]++;
      if (buffer == 1)
        collision_indexes.push_back(new_key);
      if (buffer)
        count_collision++;
    }
    delete [] token;
  }

  struct timespec end_time;
  clock_gettime(CLOCK_MONOTONIC, &end_time);
  uint64_t time_per_hash =
    ((end_time.tv_sec - start_time.tv_sec) * 1000000000 + (end_time.tv_nsec - start_time.tv_nsec)) / NUM_HASHED;
  printf("Counted a total of %d collisions out of %d hits = %f%%\n", count_collision, NUM_HASHED,
         (count_collision * 100.0) / NUM_HASHED);
  printf("%lu ns/hash\n", time_per_hash);

  uint32_t max_index = 0;
  for (unsigned int index : collision_indexes) {
    if (collisions[index] > collisions[max_index]) {
      max_index = index;
    }
  }
  printf("Max collision: %d\n", collisions[max_index]);
  printf("Found at:\n");
  for (unsigned int index : collision_indexes) {
    if (collisions[index] == collisions[max_index]) {
      printf("\t%u\n", index);
    }
  }
}

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */
