/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include "pallas/utils/pallas_dbg.h"
#include "pallas/utils/pallas_linked_vector.h"
#include "pallas/utils/pallas_log.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    pallas_error("Not enough arguments ! 1 argument required.\n");
  }
  if (argc > 2) {
    pallas_error("Too many arguments ! 1 argument required.\n");
  }
  size_t TEST_SIZE = std::stoi(argv[1]);
    auto parameter_handler = pallas::ParameterHandler();

  pallas::LinkedDurationVector vector = pallas::LinkedDurationVector(parameter_handler);

  for (size_t i = 0; i < TEST_SIZE; i++) {
    vector.add(i);
  }
  vector.final_update_mean();
  pallas_assert_always(vector.size == TEST_SIZE);
  pallas_assert_always(vector.min == 0);
  pallas_assert_always(vector.max == TEST_SIZE - 1);
  // This is actually because the statistics are computed "one index late"
  // Because as always the fault lies in the fact we have to compute durations.
  return EXIT_SUCCESS;
}

/* -*-
   mode: cpp;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */
