/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
#include "pallas/pallas_timestamp.h"
#include "pallas/pallas_write.h"

#define NANOSECONDS(timestamp) std::chrono::duration_cast<std::chrono::nanoseconds>(timestamp).count()



pallas_timestamp_t pallas::ThreadWriter::getTimestamp() {
  Timepoint start = std::chrono::high_resolution_clock::now();
  if (NANOSECONDS(firstTimestamp.time_since_epoch()) == 0) {
    firstTimestamp = start;
  }
  //TODO test diff par rapport au prec
  return NANOSECONDS(start - firstTimestamp);
}

pallas_timestamp_t pallas::ThreadWriter::timestamp(pallas_timestamp_t t) {
  if (t == PALLAS_TIMESTAMP_INVALID)
    return getTimestamp();
  return t;
}

void pallas::ThreadWriter::completeDurations(pallas_duration_t duration) {
  for (auto it : incompleteDurations) {
    *it += duration;
  }
  incompleteDurations.resize(0);
}

void pallas::ThreadWriter::addDurationToComplete(pallas_duration_t* duration) {
  incompleteDurations.push_back(duration);
}

// void pallas_finish_timestamp() {
//   *timestampsToDelta.front() = 0;
// }

/* -*-
   mode: cpp;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */
