/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>
#include "pallas/pallas.h"
#include "pallas/pallas_archive.h"
#include "pallas/pallas_log.h"
#include "pallas/pallas_read.h"
#include "pallas/pallas_storage.h"

static void printHeader() {
  std::cout << std::setw(4+1+ 6 + 1) << " "
            << std::right << std::setw(16) << "Stored" << " / "
    << std::right << std::setw(16) << "Computed" << std::endl;
}

static pallas_duration_t testCurrentTokenDuration(pallas::ThreadReader* reader) {
  auto token = reader->pollCurToken();
  switch (token.type) {
  case pallas::TypeEvent: {
    return reader->getEventOccurence(token, reader->currentState.currentFrame->tokenCount[token]).duration;
  }

  case pallas::TypeSequence: {
    pallas_duration_t sequence_duration = reader->getSequenceOccurence(token, reader->currentState.currentFrame->tokenCount[token]).duration;
    pallas_duration_t sum_of_durations_in_sequence = 0;
    reader->enterBlock();

    while (reader->pollNextToken().isValid()) {
      sum_of_durations_in_sequence += testCurrentTokenDuration(reader);
      reader->moveToNextToken();
    }
    sum_of_durations_in_sequence += testCurrentTokenDuration(reader);

    reader->leaveBlock();

    if (sequence_duration != sum_of_durations_in_sequence) {
      std::cout << "S" << std::left << std::setw(4) << token.id << "#" << std::left << std::setw(6) << reader->currentState.currentFrame->tokenCount[token]
                << std::right << std::setw(16) << sequence_duration
                << " / " << std::right << std::setw(16) << sum_of_durations_in_sequence << std::endl;
    }

    return sequence_duration;
  }

  case pallas::TypeLoop: {
    pallas_duration_t loop_duration = reader->getLoopOccurence(token, reader->currentState.currentFrame->tokenCount[token]).duration;
    pallas_duration_t sum_of_durations_in_loop = 0;
    reader->enterBlock();

    while (reader->pollNextToken().isValid()) {
      sum_of_durations_in_loop += testCurrentTokenDuration(reader);
      reader->moveToNextToken();
    }
    sum_of_durations_in_loop += testCurrentTokenDuration(reader);

    reader->leaveBlock();
    if (loop_duration != sum_of_durations_in_loop) {
      std::cout << "L" << std::left << std::setw(3) << token.id << "#" << std::left << std::setw(6) << reader->currentState.currentFrame->tokenCount[token]
                << std::right << std::setw(16) << loop_duration
                << " / " << std::right << std::setw(16) << sum_of_durations_in_loop << std::endl;
    }
    return loop_duration;
  }

  case pallas::TypeInvalid:
    pallas_error("This should not have happened");
  }
  return 0;
}

/* Print all the events of a thread */
static void testThreadDuration(pallas::Archive& trace, const pallas::Thread& thread) {
  printf("Testing durations for Thread %u (%s):\n", thread.id, thread.getName());
;
  auto* reader = new pallas::ThreadReader(&trace, thread.id, PALLAS_READ_FLAG_UNROLL_ALL);

  reader->leaveBlock();
  printHeader();
  testCurrentTokenDuration(reader);

  delete reader;
}

void usage(const char* prog_name) {
  printf("Usage: %s [OPTION] trace_file\n", prog_name);
  printf("\t-v          Verbose mode\n");
  printf("\t-?  -h      Display this help and exit\n");
}

int main(const int argc, char* argv[]) {
  int nb_opts = 0;
  char* trace_name = nullptr;

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-v")) {
      pallas_debug_level_set(pallas::DebugLevel::Debug);
      nb_opts++;
    } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "-?")) {
      usage(argv[0]);
      return EXIT_SUCCESS;
    } else {
      /* Unknown parameter name. It's probably the trace's path name. We can stop
       * parsing the parameter list.
       */
      break;
    }
  }

  struct separate_thousands : std::numpunct<char> {
    char_type do_thousands_sep() const override { return ','; }  // separate with commas
    string_type do_grouping() const override { return "\3"; } // groups of 3 digit
  };


  auto thousands = std::make_unique<separate_thousands>();
  std::cout.imbue(std::locale(std::cout.getloc(), thousands.release()));

  trace_name = argv[nb_opts + 1];
  if (trace_name == nullptr) {
    usage(argv[0]);
    return EXIT_SUCCESS;
  }

  /*auto trace = pallas::GlobalArchive();
  pallasReadGlobalArchive(&trace, trace_name);

  for (int i = 0; i < trace.nb_archives; i++) {
    for (int j = 0; j < trace.archive_list[i]->nb_threads; j++) {
      printf("\n");
      auto thread = trace.archive_list[i]->getThreadAt(j);
      if (thread != nullptr)
        testThreadDuration(*trace.archive_list[i], *thread);
    }
  }
*/
  return EXIT_SUCCESS;
}

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */
