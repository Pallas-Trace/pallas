/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include "htf/htf_timestamp.h"
#include <cstdlib>
#include <ctime>
#include <vector>

#define TIME_DIFF(t1, t2) ((t2.tv_sec - t1.tv_sec) * 1000000000 + (t2.tv_nsec - t1.tv_nsec))

static struct timespec first_timestamp = {0, 0};
thread_local static std::vector<htf_timestamp_t*> timestampsToDelta = std::vector<htf_timestamp_t*>();

htf_timestamp_t htf_get_timestamp() {
  struct timespec timestamp;
  clock_gettime(CLOCK_MONOTONIC, &timestamp);
  if (first_timestamp.tv_sec == 0 && first_timestamp.tv_nsec == 0) {
    first_timestamp.tv_sec = timestamp.tv_sec;
    first_timestamp.tv_nsec = timestamp.tv_nsec;
  }
  return TIME_DIFF(first_timestamp, timestamp);
}

htf_timestamp_t htf_timestamp(htf_timestamp_t t) {
  if (t == HTF_TIMESTAMP_INVALID)
    return htf_get_timestamp();
  return t;
}

void htf_delta_timestamp(htf_timestamp_t* t) {
  if (!timestampsToDelta.empty()) {
    htf_timestamp_t startingTimestamp = *timestampsToDelta.front();
    htf_timestamp_t startingDuration = 0;
    for (auto& timestamp : timestampsToDelta) {
      if (!startingDuration) {
        *timestamp = *t - *timestamp;
        startingDuration = *timestamp;
      } else {
        *timestamp += startingDuration;
        *timestamp -= startingTimestamp;
      }
    }
    timestampsToDelta.clear();
  }
  timestampsToDelta.push_back(t);
}

void htf_add_timestamp_to_delta(htf_timestamp_t* t) {
  timestampsToDelta.push_back(t);
}

void htf_finish_timestamp() {
  *timestampsToDelta.front() = 0;
}

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */
