/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
#include <filesystem>
#if __cplusplus >= 202002L && (__GNUC__ >= 13 || __clang__ >= 14 || _MSC_VER >= 1929)
#include <format>
#define HAS_FORMAT
#endif
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

static inline std::string dummyTraceName = "find_loop_trace";

int main(int argc, char** argv __attribute__((unused))) {
  if (argc < 2) {
    pallas_error("Not enough arguments ! 2 argument required.\n");
  }
  if (argc > 3) {
    pallas_error("Too many arguments ! 3 argument required.\n");
  }
  int MAX_EVENT = std::stoi(argv[1]);
  int NUM_LOOPS = std::stoi(argv[2]);

  pallas_log(DebugLevel::Quiet, "Starting test:\n");

  /* Make a dummy archive and a dummy thread writer. */
  GlobalArchive trace(dummyTraceName.c_str(), dummyTraceName.c_str());
  Archive archive(trace, 0);
  ThreadWriter thread_writer(archive, 0);

  /* Start recording some events.*/

  pallas_log(DebugLevel::Quiet, "\tLoading definitions and creating first events\n");
  for (int eid = 0; eid < MAX_EVENT; eid++) {
#ifdef HAS_FORMAT
    archive.addString(eid, std::format("dummyEvent{}", eid).c_str());
#else
    trace.addString(eid, "dummyEvent");
#endif
    pallas_record_generic(&thread_writer, nullptr, get_timestamp(), eid);
  }
  /* Check they've been correctly registered. */
  pallas_assert_always(thread_writer.cur_depth == 0);
  pallas_assert_always(thread_writer.sequence_stack[0].size() == (unsigned int)MAX_EVENT);
  for (int eid = 0; eid < MAX_EVENT; eid++) {
    pallas_assert_always(thread_writer.sequence_stack[0][eid].type == TypeEvent);
    pallas_assert_always(thread_writer.sequence_stack[0][eid].id == eid);
  }

  pallas_log(DebugLevel::Quiet, "\tCreating our first loop\n");
  /* Start recording some more events. This should make a first loop. */
  for (int eid = 0; eid < MAX_EVENT; eid++)
    pallas_record_generic(&thread_writer, nullptr, get_timestamp(), eid);

  /* This should have been recognized as a loop, so now there should be some changes. */
  pallas_assert_always(thread_writer.cur_depth == 0);
  pallas_assert_always(thread_writer.sequence_stack[0].size() == 1);
  pallas_assert_always(thread_writer.sequence_stack[0][0].type == TypeLoop);
  /* Check that the loop is correct */
  auto& firstLoop = thread_writer.thread->loops[0];
  pallas_assert_always(firstLoop.nb_iterations == 2);

  /* Check that the sequence inside that loop is correct */
  struct Sequence* s = thread_writer.thread->getSequence(firstLoop.repeated_token);

  pallas_assert_always(s->size() == (unsigned int)MAX_EVENT);

  for (int eid = 0; eid < MAX_EVENT; eid++) {
    pallas_assert_always(s->tokens[eid].type == TypeEvent);
    pallas_assert_always(s->tokens[eid].id == eid);
  }

  pallas_log(DebugLevel::Quiet, "\tIncrementing L0 to 3 iterations\n");
  /* Start recording even more events. The first loop happens 3 times now.*/
  for (int eid = 0; eid < MAX_EVENT; eid++)
    pallas_record_generic(&thread_writer, nullptr, get_timestamp(), eid);
  pallas_assert_always(thread_writer.cur_depth == 0);
  pallas_assert_always(thread_writer.sequence_stack[0].size() == 1);
  pallas_assert_always(firstLoop.nb_iterations == 3);


  pallas_log(DebugLevel::Quiet, "\tAdding a dummy event\n");
  /* Now start recording one more event and then loop again. */
#ifdef HAS_FORMAT
  archive.addString(MAX_EVENT, std::format("dummyEvent{}", MAX_EVENT).c_str());
#else
  trace.addString(MAX_EVENT, "dummyEvent");
#endif
  pallas_record_generic(&thread_writer, nullptr, get_timestamp(), MAX_EVENT);


  pallas_log(DebugLevel::Quiet, "\tRunning the same loop with different iterations\n");
  DOFOR(loop_number, NUM_LOOPS) {
    DOFOR(eid, MAX_EVENT) {
      pallas_record_generic(&thread_writer, nullptr, get_timestamp(), eid);
    }
  }
  pallas_assert_always(thread_writer.cur_depth == 0);
  pallas_assert_always(thread_writer.sequence_stack[0].size() == 3); // L0 E L1
  pallas_assert_always(firstLoop.nb_iterations == 3);

  auto& secondLoop = thread_writer.thread->loops[1];
  pallas_assert_always(secondLoop.nb_iterations == NUM_LOOPS);

  thread_writer.threadClose();
  trace.store();
  // TODO Find a way for the test to clean this trace
  //      Because somehow the following does not remove the folder
  //      Only is content
  //      std::filesystem::remove_all(dummyTraceName);
  return 0;
}

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */
