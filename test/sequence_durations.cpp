/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 *
 * This is a test for the computation of durations during sequence creation.
 */
#include "pallas/pallas.h"
#include "pallas/pallas_write.h"
#include "pallas/pallas_record.h"
#include "pallas/pallas_log.h"

using namespace pallas;

static pallas_timestamp_t ts = 0;
static pallas_timestamp_t step = 1;
static pallas_timestamp_t get_timestamp() {
  ts += step;
  return ts;
}

static inline void check_event_allocation(Thread* thread_trace, unsigned id) {
  pallas_log(DebugLevel::Max, "Searching for event {.id=%d}\n", id);

  while (id > thread_trace->nb_allocated_events) {
    pallas_log(DebugLevel::Normal, "Doubling mem space of events for thread trace %p\n", (void*)thread_trace);
    doubleMemorySpaceConstructor(thread_trace->events, thread_trace->nb_allocated_events);
  }
  if (thread_trace->nb_events < id + 1) {
    thread_trace->nb_events = id + 1;
  }
}

int main(int argc __attribute__((unused)), char** argv __attribute__((unused))) {
  /* Make a dummy archive and a dummy thread writer. */
  Archive archive("sequence_duration_trace", "sequence_duration_trace", 0);
  ThreadWriter thread_writer(archive, 0);

  /* Here's what we're going to do: we'll define some sequences as the following:
   * S1 = E1 E2
   * S2 = E2 E3 E4
   * S3 = E3 E4 E5 E6
   * and L_i will be a repetition of S_i+1 (as S0 is always taken)
   * What we then need to do is a final repetition of
   * S_n = L1 L2 ... L_n-1
   * And then we'll check that the durations of all the sequences is oki doki
   */

  int OUTER_LOOP_SIZE = 2;
  int INNER_LOOP_SIZE = 10;
  int MAX_SUBSEQUENCE_NUMBER = 10;

  for (int outer_loop_number = 1; outer_loop_number <= OUTER_LOOP_SIZE; outer_loop_number++) {
    // Outer loop for S_n
    for (int sequence_number = 1; sequence_number <= MAX_SUBSEQUENCE_NUMBER; sequence_number++) {
      // Doing all the L_i containing the S_i
      for (int loop = 0; loop < INNER_LOOP_SIZE; loop++) {
        // Finally, doing the sequence
        for (int eid = 0; eid <= sequence_number; eid++)
          pallas_record_generic(&thread_writer, nullptr, get_timestamp(), sequence_number * MAX_SUBSEQUENCE_NUMBER + eid);
      }
    }
  }
  pallas_record_generic(&thread_writer, nullptr, get_timestamp(), 0);
  thread_writer.thread->events[0].durations->at(0) = 0;
  thread_writer.threadClose();
  archive.close();

  for (int sequence_number = 0; sequence_number <= MAX_SUBSEQUENCE_NUMBER; sequence_number++) {
    Sequence* s = thread_writer.thread->sequences[sequence_number];
    std::cout << "Information on sequence " << sequence_number << ":\n"
              << "\tNumber of tokens: " << s->tokens.size() << ": ";
    thread_writer.thread->printTokenVector(s->tokens);
    std::cout << "\tNumber of iterations: " << s->durations->size << "\n"
              << "\tDurations: ";
    s->durations->print();
    std::cout << std::endl;

    if (sequence_number > 0) {
      pallas_assert_always(s->tokens.size() == sequence_number + 1);
      pallas_assert_always(s->durations->size == INNER_LOOP_SIZE * OUTER_LOOP_SIZE);
      for (auto t : *s->durations) {
        pallas_assert_always(t == s->size());
      }
    } else {
      //      pallas_assert_always(s->tokens.size() == )
      //      pallas_assert_always(s->durations->back() == )
    }
  }

  return 0;
}

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */
