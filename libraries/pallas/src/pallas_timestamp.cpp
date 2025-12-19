/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include "pallas/pallas_write.h"

#include "pallas/utils/pallas_timestamp.h"

#define NANOSECONDS(timestamp) std::chrono::duration_cast<std::chrono::nanoseconds>(timestamp).count()

pallas_timestamp_t pallas::ThreadWriter::getTimestamp() {
  Timepoint start = std::chrono::high_resolution_clock::now();
  if (NANOSECONDS(firstTimestamp.time_since_epoch()) == 0) {
    firstTimestamp = start;
  }
  return NANOSECONDS(start - firstTimestamp);
}

pallas_timestamp_t pallas::ThreadWriter::timestamp(pallas_timestamp_t t) {
  if (t == PALLAS_TIMESTAMP_INVALID)
    return getTimestamp();
  return t;
}

/* -*-
   mode: cpp;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */
