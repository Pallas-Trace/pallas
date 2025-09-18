/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 *
 * This is a test for the computation of durations during sequence creation.
 */
#include "pallas/pallas.h"
#include "pallas/pallas_log.h"
#include "pallas/pallas_record.h"
#include "pallas/pallas_write.h"

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
        pallas_log(DebugLevel::Debug, "Doubling mem space of events for thread trace %p\n", (void*)thread_trace);
        doubleMemorySpaceConstructor(thread_trace->events, thread_trace->nb_allocated_events);
    }
    if (thread_trace->nb_events < id + 1) {
        thread_trace->nb_events = id + 1;
    }
}

static inline void print_sequence_info(Sequence* s, Thread* t) {
    std::cout << "Information on sequence " << s->id << ":\n"
            << "\tNumber of tokens: " << s->tokens.size() << ": ";
    t->printTokenVector(s->tokens);
    std::cout << "\tNumber of iterations: " << s->durations->size << "\n"
            << "\tDurations: " << s->durations->to_string() << "\n"
            << "\tExclusive Durations:" << s->exclusive_durations->to_string() << "\n"
            << "\tTimestamps: " << s->timestamps->to_string() << std::endl;
}

int main(int argc __attribute__((unused)), char** argv __attribute__((unused))) {
    /* Make a dummy archive and a dummy thread writer. */
    Archive archive("sequence_duration_trace", 0);
    archive.addString(0, "main_thread");
    archive.defineLocation(0, 0, 0);
    archive.addRegion(0, 0);
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
    pallas_record_enter(&thread_writer, nullptr, get_timestamp(), 0);

    int OUTER_LOOP_SIZE = 4;
    int INNER_LOOP_SIZE = 5;
    int MAX_SUBSEQUENCE_NUMBER = 10;

    for (int outer_loop_number = 1; outer_loop_number <= OUTER_LOOP_SIZE; outer_loop_number++) {
        // Outer loop for S_n
        for (int sequence_number = 1; sequence_number <= MAX_SUBSEQUENCE_NUMBER; sequence_number++) {
            // Doing all the L_i containing the S_i
            for (int loop = 0; loop < INNER_LOOP_SIZE; loop++) {
                // Finally, doing the sequence
                for (int eid = 0; eid <= sequence_number; eid++) {
                    pallas_record_generic(&thread_writer, nullptr, get_timestamp(), sequence_number * MAX_SUBSEQUENCE_NUMBER + eid + 2);
                }
            }
        }
    }
    pallas_record_leave(&thread_writer, nullptr, get_timestamp(), 0);
    thread_writer.thread->events[0].timestamps->at(0) = 0;

    for (int sequence_number = 0; sequence_number <= MAX_SUBSEQUENCE_NUMBER; sequence_number++) {
        Sequence* s = thread_writer.thread->sequences[sequence_number];
        if (sequence_number > 0) {
            s->durations->final_update_mean();
            s->exclusive_durations->final_update_mean();
            print_sequence_info(s, thread_writer.thread);

            pallas_assert_equals_always(s->tokens.size(), sequence_number + 1);
            pallas_assert_equals_always(s->durations->size, INNER_LOOP_SIZE * OUTER_LOOP_SIZE);
            for (size_t i = 0; i < s->durations->size; i++) {
                auto& t = s->durations->at(i);
                pallas_assert_equals_always(t, s->size() - 1);
            }
            pallas_assert_equals_always(s->durations->min, s->size() - 1);
            pallas_assert_equals_always(s->durations->max, s->size() - 1);
            pallas_assert_equals_always(s->durations->mean, s->size() - 1);
        } else {
            //      pallas_assert_always(s->tokens.size() == )
            //      pallas_assert_always(s->durations->back() == )
        }
    }
    auto outer_sequence = thread_writer.thread->sequences[MAX_SUBSEQUENCE_NUMBER + 1];
    outer_sequence->durations->final_update_mean();
    outer_sequence->exclusive_durations->final_update_mean();
    print_sequence_info(outer_sequence, thread_writer.thread);
    pallas_assert_equals_always(outer_sequence->timestamps->size, OUTER_LOOP_SIZE);
    pallas_assert_equals_always(outer_sequence->tokens.size(), MAX_SUBSEQUENCE_NUMBER);
    // theoretical_length = INNER_LOOP_SIZE * sum(i=0;MAX_SUBSEQUENCE_NUMBER) { i +  2 }
    size_t theoretical_length = INNER_LOOP_SIZE * (2 * MAX_SUBSEQUENCE_NUMBER + (MAX_SUBSEQUENCE_NUMBER * (MAX_SUBSEQUENCE_NUMBER - 1) / 2)) - 1;
    pallas_assert_equals_always(theoretical_length, outer_sequence->durations->min);
    pallas_assert_equals_always(theoretical_length, outer_sequence->durations->max);

    return 0;
}

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */
