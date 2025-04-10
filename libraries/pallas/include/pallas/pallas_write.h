/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
/** @file
 * Everything that has to do with writing a trace during an execution.
 */
#pragma once
#include "pallas.h"
#include "pallas_archive.h"
#include "pallas_attribute.h"
#ifdef __cplusplus
namespace pallas {
#endif
/**
 * Writes one thread to the Pallas trace format.
 */
typedef struct ThreadWriter {
  Thread* thread CXX({nullptr}); /**< Thread being written. */
  C_CXX(void, std::vector<Token>) *
    sequence_stack; /**< Stack of all the *incomplete* sequences the writer is currently in. */
  int cur_depth;    /**< Current depth in the callstack. */
  int max_depth;    /**< Maximum depth in the callstack. */

  pallas_timestamp_t last_timestamp; /**< Timestamp of the last encountered event */

  pallas_duration_t* last_duration; /**< Pointer to the last event duration (to be updated when the timestamp of the
                                    next event is known) */

  pallas_timestamp_t*
    sequence_start_timestamp; /**< Start date of each ongoing sequence (used for computing the sequence duration) */

  /**
   * This is a vector of durations that are incomplete.
   * Those are durations from sequence whose last duration was still a timestamp.
   * Next time a timestamp is registered, and thus the duration for the last event in those sequences is computed,
   * they'll be updated.
   */
  DEFINE_Vector(pallas_duration_t*, incompleteDurations);
  /** The first recorded timestamp for this thread.*/
  C_CXX(uint8_t firstTimestamp[TIMEPOINT_SIZE], Timepoint firstTimestamp = {});
#ifdef __cplusplus

 private:
  void findLoopBasic(size_t maxLoopLength);
  /** Tries to find a Loop in the current array of tokens.  */
  void findLoop();
  /** Tries to find and replace the last n tokens in the grammar sequence. */
  void findSequence(size_t n);
  /** Creates a Loop in the trace, and returns a pointer to it.
   * Does not change the current array of tokens.
   * @sequence_id ID of the sequence being repeated.
   * */
  Loop* createLoop(Token sequence_id);
  /** Create a Loop and change the current array of token to reflect that.
   *
   * For example, replaces `[E1, E2, E3, E4, E1, E2, E3, E4]` with `[L1]`,
   * where L1 contains 2 * S1 = `[E1, E2, E3, E4]`
   * @param loop_len Length of the sequence repeated in the loop.
   * @param index_first_iteration Starting index of the first iteration of the loop.
   * @param index_second_iteration Starting index of the second iteration of the loop.
   */
  void replaceTokensInLoop(int loop_len, size_t index_first_iteration, size_t index_second_iteration);
  /** Returns a reference to the current sequence of Tokens being written. */
  [[nodiscard]] std::vector<Token>& getCurrentTokenSequence() const { return sequence_stack[cur_depth]; };
  /** Stores the timestamp in the given EventSummary. */
  void storeTimestamp(EventSummary* es, pallas_timestamp_t ts);
  /** Stores the attribute list in the given EventSummary. */
  void storeAttributeList(EventSummary* es, AttributeList* attribute_list, size_t occurence_index);
  /** Stores the tokens in that Sequence's array of Tokens, then tries to find a Loop.*/
  void storeToken(std::vector<Token>& tokenSeq, Token t);
  /** Move up the callstack and create a new Sequence. */
  void recordEnterFunction();
  /** Close a Sequence and move down the callstack. */
  void recordExitFunction();
  /** Returns the current timestamp. */
  pallas_timestamp_t getTimestamp();
  /** Returns t if it's valid, of the current timestamp. */
  pallas_timestamp_t timestamp(pallas_timestamp_t t);
  /** Adds the given duration to all the stored addresses to complete.*/
  void completeDurations(pallas_duration_t duration);
  /** Adds the given address to a list of duration to complete. */
  void addDurationToComplete(pallas_duration_t* duration);

 public:
  ThreadWriter(Archive& archive, ThreadId thread_id);
  void threadClose();
  /** Creates the new Event and stores it. Returns the occurence index of that new Event. */
  size_t storeEvent(enum EventType event_type,
                    TokenId event_id,
                    pallas_timestamp_t ts,
                    struct AttributeList* attribute_list);
  ~ThreadWriter();
#endif
} ThreadWriter;
#ifdef __cplusplus
}
extern "C" {
#endif

/* C Callbacks */
/**
 * Allocates a new ThreadWriter and returns a pointer to that allocated memory.
 * @return Pointer to ThreadWriter
 */
extern PALLAS(ThreadWriter) * pallas_thread_writer_new(PALLAS(Archive)* archive, PALLAS(ThreadId) thread_id);
extern void pallas_global_archive_close(PALLAS(GlobalArchive) * archive);

extern void pallas_thread_writer_close(PALLAS(ThreadWriter) * thread_writer);
extern void pallas_thread_writer_delete(PALLAS(ThreadWriter) * thread_writer);
extern void pallas_archive_close(PALLAS(Archive) * archive);

extern void pallas_store_event(PALLAS(ThreadWriter) * thread_writer,
                               enum PALLAS(EventType) event_type,
                               PALLAS(TokenId) id,
                               pallas_timestamp_t ts,
                               PALLAS(AttributeList) * attribute_list);


#ifdef __cplusplus
};
#endif


/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */
