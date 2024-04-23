/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include <inttypes.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pallas/pallas.h"
#include "pallas/pallas_archive.h"
#include "pallas/pallas_hash.h"
#include "pallas/pallas_parameter_handler.h"
#include "pallas/pallas_storage.h"
#include "pallas/pallas_timestamp.h"
#include "pallas/pallas_write.h"
thread_local int pallas_recursion_shield = 0;

namespace pallas {
Token Thread::getSequenceId(pallas::Sequence* sequence) {
  return getSequenceIdFromArray(sequence->tokens.data(), sequence->size());
}
/**
 * Compares two arrays of tokens array1 and array2
 */
static inline bool _pallas_arrays_equal(Token* array1, size_t size1, Token* array2, size_t size2) {
  if (size1 != size2)
    return 0;
  return memcmp(array1, array2, sizeof(Token) * size1) == 0;
}

Token Thread::getSequenceIdFromArray(pallas::Token* token_array, size_t array_len) {
  uint32_t hash;
  hash32(token_array, array_len, SEED, &hash);
  pallas_log(DebugLevel::Debug, "Searching for sequence {.size=%zu, .hash=%x}\n", array_len, hash);

  for (unsigned i = 1; i < nb_sequences; i++) {
    if (sequences[i]->hash == hash) {
      if (_pallas_arrays_equal(token_array, array_len, sequences[i]->tokens.data(), sequences[i]->size())) {
        pallas_log(DebugLevel::Debug, "\t found with id=%u\n", i);
        return PALLAS_SEQUENCE_ID(i);
      } else {
        pallas_warn("Found two sequences with the same hash\n");
      }
    }
  }

  if (nb_sequences >= nb_allocated_sequences) {
    pallas_warn("Doubling mem space of sequence for thread trace %p\n", this);
    DOUBLE_MEMORY_SPACE(sequences, nb_allocated_sequences, Sequence*);
    for (uint i = nb_allocated_sequences / 2; i < nb_allocated_sequences; i++) {
      sequences[i] = new Sequence;
    }
  }

  size_t index = nb_sequences++;
  Token sid = PALLAS_SEQUENCE_ID(index);
  pallas_log(DebugLevel::Debug, "\tSequence not found. Adding it with id=S%zx\n", index);

  Sequence* s = getSequence(sid);
  s->tokens.resize(array_len);
  memcpy(s->tokens.data(), token_array, sizeof(Token) * array_len);
  s->hash = hash;

  return sid;
}

Loop* ThreadWriter::createLoop(int start_index, int loop_len) {
  if (thread_trace.nb_loops >= thread_trace.nb_allocated_loops) {
    pallas_warn("Doubling mem space of loops for thread writer %p's thread trace, cur=%d\n", this,
                thread_trace.nb_allocated_loops);
    DOUBLE_MEMORY_SPACE(thread_trace.loops, thread_trace.nb_allocated_loops, Loop);
  }

  auto* cur_seq = getCurrentSequence();
  Token sid = thread_trace.getSequenceIdFromArray(&cur_seq->tokens[start_index], loop_len);

  int index = -1;
  for (int i = 0; i < thread_trace.nb_loops; i++) {
    if (thread_trace.loops[i].repeated_token.id == sid.id) {
      index = i;
      pallas_log(DebugLevel::Debug, "\tLoop already exists: id=L%x containing S%x\n", index, sid.id);
      break;
    }
  }
  if (index == -1) {
    index = thread_trace.nb_loops++;
    pallas_log(DebugLevel::Debug, "\tLoop not found. Adding it with id=L%x containing S%x\n", index, sid.id);
  }

  Loop* l = &thread_trace.loops[index];
  l->nb_iterations.push_back(1);
  l->repeated_token = sid;
  l->self_id = PALLAS_LOOP_ID(index);
  return l;
}

void ThreadWriter::storeTimestamp(EventSummary* es, pallas_timestamp_t ts) {
#if 0
  // Not yet implemented
  if(store_event_timestamps) {
    es->timestamps->add(ts);
  }
#endif

  int store_event_durations = 1;  // TODO: make is optional
  if (store_event_durations) {
    // update the last event's duration
    if (last_duration) {
      pallas_timestamp_t delta = pallas_get_duration(last_timestamp, ts);
      *last_duration = delta;
      completeDurations(delta);
    }

    // allocate a new duration for the current event
    last_duration = es->durations->add(ts);
  }

  last_timestamp = ts;
}

void ThreadWriter::storeAttributeList(pallas::EventSummary* es,
                                      struct pallas::AttributeList* attribute_list,
                                      size_t occurence_index) {
  attribute_list->index = occurence_index;
  if (es->attribute_pos + attribute_list->struct_size >= es->attribute_buffer_size) {
    if (es->attribute_buffer_size == 0) {
      pallas_warn("Allocating attribute memory for event %u\n", es->id);
      es->attribute_buffer_size = NB_ATTRIBUTE_DEFAULT * sizeof(struct pallas::AttributeList);
      es->attribute_buffer = new uint8_t[es->attribute_buffer_size];
      pallas_assert(es->attribute_buffer != NULL);
    } else {
      pallas_warn("Doubling mem space of attributes for event %u\n", es->id);
      DOUBLE_MEMORY_SPACE(es->attribute_buffer, es->attribute_buffer_size, uint8_t);
    }
    pallas_assert(es->attribute_pos + attribute_list->struct_size < es->attribute_buffer_size);
  }

  memcpy(&es->attribute_buffer[es->attribute_pos], attribute_list, attribute_list->struct_size);
  es->attribute_pos += attribute_list->struct_size;

  pallas_log(DebugLevel::Debug, "store_attribute: {index: %d, struct_size: %d, nb_values: %d}\n", attribute_list->index,
             attribute_list->struct_size, attribute_list->nb_values);
}

void ThreadWriter::storeToken(pallas::Sequence* seq, pallas::Token t) {
  pallas_log(DebugLevel::Debug, "store_token: (%c%x) in %p (size: %zu)\n", PALLAS_TOKEN_TYPE_C(t), t.id, seq,
             seq->size() + 1);
  seq->tokens.push_back(t);
  findLoop();
}

/**
 * Adds an iteration of the given sequence to the loop.
 */
void Loop::addIteration() {
#if 0
  //	SID used to be an argument for this function, but it isn't actually required.
  //	It's only there for safety purposes.
  //	We shouldn't ask it, since	it actually takes some computing power to at it, most times.
  //	"Safety checks are made to prevent crashes, but if you program well, you don't need to prevent crashes"
  struct Sequence *s1 = pallas_get_sequence(&thread_writer->thread_trace, sid);
  struct Sequence *s2 = pallas_get_sequence(&thread_writer->thread_trace,
					     PALLAS_TOKEN_TO_SEQUENCE_ID(loop->repeated_token));
  pallas_assert(_pallas_sequences_equal(s1, s2));
#endif
  pallas_log(DebugLevel::Debug, "Adding an iteration to L%x n°%zu (to %u)\n", self_id.id, nb_iterations.size() - 1,
             nb_iterations.back() + 1);
  nb_iterations.back()++;
}

void ThreadWriter::replaceTokensInLoop(int loop_len, size_t index_first_iteration, size_t index_second_iteration) {
  if (index_first_iteration > index_second_iteration) {
    size_t tmp = index_second_iteration;
    index_second_iteration = index_first_iteration;
    index_first_iteration = tmp;
  }

  Loop* loop = createLoop(index_first_iteration, loop_len);
  Sequence* cur_seq = getCurrentSequence();

  // We need to go back in the current sequence in order to correctly calculate our durations
  Sequence* loop_seq = thread_trace.getSequence(loop->repeated_token);

  pallas_timestamp_t duration_first_iteration =
    thread_trace.getSequenceDuration(&cur_seq->tokens[index_first_iteration], 2 * loop_len, true);
  pallas_timestamp_t duration_second_iteration =
    thread_trace.getSequenceDuration(&cur_seq->tokens[index_second_iteration], loop_len, true);
  // We don't take into account the last token because it's not a duration yet

  loop_seq->durations->add(duration_first_iteration - duration_second_iteration);
  addDurationToComplete(loop_seq->durations->add(duration_second_iteration));

  // The current sequence last_timestamp does not need to be updated

  cur_seq->tokens.resize(index_first_iteration);
  cur_seq->tokens.push_back(loop->self_id);

  loop->addIteration();
}

/**
 * Finds a Loop in the current Sequence using a basic quadratic algorithm.
 *
 * For each correct correct possible loop length, this algorithm tries two things:
 *  - First, it checks if the array of tokens of that length is in front of a loop token
 *      whose repeating sequence is the same as ours. If it it, it replaces it.
 *       - Example: L0 = 2 * S1 = E1 E2 E3. L0 E1 E2 E3 -> L0 (= 3 * S1).
 *  - Secondly, it checks for any doubly repeating array of token, and replaces it with a Loop.
 *       - Example: E1 E2 E3 E1 E2 E3 -> L0. L0 = 2 * S1 = E1 E2 E3
 * @param maxLoopLength The maximum loop length that we try to find.
 */
void ThreadWriter::findLoopBasic(size_t maxLoopLength) {
  Sequence* currentSequence = getCurrentSequence();
  size_t currentIndex = currentSequence->size() - 1;
  for (int loopLength = 1; loopLength < maxLoopLength && loopLength <= currentIndex; loopLength++) {
    // search for a loop of loopLength tokens
    size_t s1Start = currentIndex + 1 - loopLength;
    size_t loopStart = s1Start - 1;
    // First, check if there's a loop that start at loopStart
    if (currentSequence->tokens[loopStart].type == TypeLoop) {
      Token l = currentSequence->tokens[loopStart];
      Loop* loop = thread_trace.getLoop(l);
      pallas_assert(loop);

      Sequence* seq = thread_trace.getSequence(loop->repeated_token);
      pallas_assert(seq);

      if (_pallas_arrays_equal(&currentSequence->tokens[s1Start], loopLength, seq->tokens.data(), seq->size())) {
        // The current sequence is just another iteration of the loop
        // remove the sequence, and increment the iteration count
        pallas_log(DebugLevel::Debug, "Last tokens were a sequence from L%x aka S%x\n", loop->self_id.id,
                   loop->repeated_token.id);
        loop->addIteration();
        // The current sequence last_timestamp does not need to be updated

        pallas_timestamp_t ts = thread_trace.getSequenceDuration(&currentSequence->tokens[s1Start], loopLength, true);
        addDurationToComplete(seq->durations->add(ts));
        currentSequence->tokens.resize(s1Start);
        return;
      }
    }

    if (currentIndex + 1 >= 2 * loopLength) {
      size_t s2Start = currentIndex + 1 - 2 * loopLength;
      /* search for a loop of loopLength tokens */
      int is_loop = 1;
      /* search for new loops */
      is_loop = _pallas_arrays_equal(&currentSequence->tokens[s1Start], loopLength, &currentSequence->tokens[s2Start],
                                     loopLength);

      if (is_loop) {
        if (debugLevel >= DebugLevel::Debug) {
          printf("Found a loop of len %d:\n", loopLength);
          thread_trace.printTokenArray(currentSequence->tokens.data(), s1Start, loopLength);
          thread_trace.printTokenArray(currentSequence->tokens.data(), s2Start, loopLength);
          printf("\n");
        }
        replaceTokensInLoop(loopLength, s1Start, s2Start);
        return;
      }
    }
  }
}

/**
 * Finds a Loop in the current Sequence by first filtering the correct Tokens.
 *
 * The idea is that since we always search for a Loop who will end on our last Token,
 * We only need to start searching arrays who end by that token.
 * We thus start by filtering the indexes of the correct tokens, and then we start searching for loops, using those
 * indexes.
 */
void ThreadWriter::findLoopFilter() {
  auto endingIndexes = std::vector<size_t>();
  auto loopIndexes = std::vector<size_t>();
  size_t i = 0;
  Sequence* currentSequence = getCurrentSequence();
  size_t curIndex = currentSequence->size() - 1;
  for (auto token : currentSequence->tokens) {
    if (token == currentSequence->tokens.back()) {
      endingIndexes.push_back(i);
    }
    if (token.type == TypeLoop) {
      loopIndexes.push_back(i);
    }
    i++;
  }
  for (auto endingIndex : endingIndexes) {
    size_t loopLength = curIndex - endingIndex;
    // If the loop can't exist, we skip it
    if (!loopLength || (endingIndex + 1) < loopLength)
      continue;
    if (_pallas_arrays_equal(&currentSequence->tokens[endingIndex + 1], loopLength,
                             &currentSequence->tokens[endingIndex + 1 - loopLength], loopLength)) {
      if (debugLevel >= DebugLevel::Debug) {
        printf("Found a loop of len %lu:\n", loopLength);
        thread_trace.printTokenArray(currentSequence->tokens.data(), endingIndex + 1, loopLength);
        thread_trace.printTokenArray(currentSequence->tokens.data(), endingIndex + 1 - loopLength, loopLength);
        printf("\n");
      }
      replaceTokensInLoop(loopLength, endingIndex + 1, endingIndex + 1 - loopLength);
    }
  }

  for (auto loopIndex : loopIndexes) {
    Token token = currentSequence->tokens[loopIndex];
    size_t loopLength = curIndex - loopIndex;
    auto* loop = thread_trace.getLoop(token);
    auto* sequence = thread_trace.getSequence(loop->repeated_token);
    if (_pallas_arrays_equal(&currentSequence->tokens[loopIndex + 1], loopLength, sequence->tokens.data(),
                             sequence->size())) {
      pallas_log(DebugLevel::Debug, "Last tokens were a sequence from L%x aka S%x\n", loop->self_id.id,
                 loop->repeated_token.id);
      loop->addIteration();
      // The current sequence last_timestamp does not need to be updated

      pallas_timestamp_t ts =
        thread_trace.getSequenceDuration(&currentSequence->tokens[loopIndex + 1], loopLength, true);
      addDurationToComplete(sequence->durations->add(ts));
      currentSequence->tokens.resize(loopIndex + 1);
      return;
    }
  }
}

void ThreadWriter::findLoop() {
  if (parameterHandler.getLoopFindingAlgorithm() == LoopFindingAlgorithm::None) {
    return;
  }

  Sequence* currentSequence = getCurrentSequence();
  size_t currentIndex = currentSequence->size() - 1;

  switch (parameterHandler.getLoopFindingAlgorithm()) {
  case LoopFindingAlgorithm::None:
    return;
  case LoopFindingAlgorithm::Basic:
  case LoopFindingAlgorithm::BasicTruncated: {
    size_t maxLoopLength = (parameterHandler.getLoopFindingAlgorithm() == LoopFindingAlgorithm::BasicTruncated)
                             ? parameterHandler.getMaxLoopLength()
                             : SIZE_MAX;
    if (debugLevel >= DebugLevel::Debug) {
      printf("Find loops using Basic Algorithm:\n");
      size_t start_index = (currentIndex >= maxLoopLength) ? currentIndex - maxLoopLength : 0;
      size_t len = (currentIndex <= maxLoopLength) ? currentIndex + 1 : maxLoopLength;
      thread_trace.printTokenArray(currentSequence->tokens.data(), start_index, len);
    }
    findLoopBasic(maxLoopLength);
  } break;
  case LoopFindingAlgorithm::Filter: {
    findLoopFilter();
    break;
  }
  default:
    pallas_error("Invalid LoopFinding algorithm\n");
  }
}

void ThreadWriter::recordEnterFunction() {
  cur_depth++;
  if (cur_depth >= max_depth) {
    pallas_error("Depth = %d >= max_depth (%d) \n", cur_depth, max_depth);
  }
}

void ThreadWriter::recordExitFunction() {
  Sequence* cur_seq = getCurrentSequence();

#ifdef DEBUG
  // check that the sequence is not bugous

  Token first_token = cur_seq->tokens[0];
  Token last_token = cur_seq->tokens.back();
  if (first_token.type != last_token.type) {
    /* If a sequence starts with an Event (eg Enter function foo), it
       should end with an Event too (eg. Exit function foo) */
    pallas_warn("When closing sequence %p: PALLAS_TOKEN_TYPE(%c%x) != PALLAS_TOKEN_TYPE(%c%x)\n", cur_seq,
                first_token.type, first_token.id, last_token.type, last_token.id);
  }

  if (first_token.type == TypeEvent) {
    Event* first_event = thread_trace.getEvent(first_token);
    Event* last_event = thread_trace.getEvent(last_token);

    enum Record expected_record;
    switch (first_event->record) {
    case PALLAS_EVENT_ENTER:
      expected_record = PALLAS_EVENT_LEAVE;
      break;
    case PALLAS_EVENT_MPI_COLLECTIVE_BEGIN:
      expected_record = PALLAS_EVENT_MPI_COLLECTIVE_END;
      break;
    case PALLAS_EVENT_OMP_FORK:
      expected_record = PALLAS_EVENT_OMP_JOIN;
      break;
    case PALLAS_EVENT_THREAD_FORK:
      expected_record = PALLAS_EVENT_THREAD_JOIN;
      break;
    case PALLAS_EVENT_THREAD_TEAM_BEGIN:
      expected_record = PALLAS_EVENT_THREAD_TEAM_END;
      break;
    case PALLAS_EVENT_THREAD_BEGIN:
      expected_record = PALLAS_EVENT_THREAD_END;
      break;
    case PALLAS_EVENT_PROGRAM_BEGIN:
      expected_record = PALLAS_EVENT_PROGRAM_END;
      break;
    default:
      pallas_warn("Unexpected start_sequence event:\n");
      thread_trace.printEvent(first_event);
      printf("\n");
      pallas_abort();
    }

    if (last_event->record != expected_record) {
      pallas_warn("Unexpected close event:\n");
      pallas_warn("\tStart_sequence event:\n");
      thread_trace.printEvent(first_event);
      printf("\n");
      pallas_warn("\tEnd_sequence event:\n");
      thread_trace.printEvent(last_event);
      printf("\n");
    }
  }

  if (cur_seq != og_seq[cur_depth]) {
    pallas_error("cur_seq=%p, but og_seq[%d] = %p\n", cur_seq, cur_depth, og_seq[cur_depth]);
  }
#endif

  Token seq_id = thread_trace.getSequenceId(cur_seq);
  auto* seq = thread_trace.sequences[seq_id.id];

  pallas_timestamp_t sequence_duration = last_timestamp - sequence_start_timestamp[cur_depth];
  // TODO: update statistics on the sequence (min/max/avg duration)
  seq->durations->add(sequence_duration);

  pallas_log(DebugLevel::Debug, "Exiting a function, closing sequence %d (%p)\n", seq_id.id, cur_seq);

  cur_depth--;
  /* upper_seq is the sequence that called cur_seq */
  Sequence* upper_seq = getCurrentSequence();
  if (!upper_seq) {
    pallas_error("upper_seq is NULL!\n");
  }

  storeToken(upper_seq, seq_id);
  cur_seq->tokens.resize(0);
  // We need to reset the token vector
  // Calling vector::clear() might be a better way to do that,
  // but depending on the implementation it might force a bunch of realloc, which isn't great.
}  // namespace pallas

size_t ThreadWriter::storeEvent(enum EventType event_type,
                                TokenId event_id,
                                pallas_timestamp_t ts,
                                AttributeList* attribute_list) {
  ts = timestamp(ts);
  if (event_type == PALLAS_BLOCK_START) {
    recordEnterFunction();
    sequence_start_timestamp[cur_depth] = ts;
  }

  Token token = Token(TypeEvent, event_id);
  auto* sequence = getCurrentSequence();

  EventSummary* es = &thread_trace.events[event_id];
  size_t occurrence_index = es->nb_occurences++;
  storeTimestamp(es, ts);
  storeToken(sequence, token);

  if (attribute_list)
    storeAttributeList(es, attribute_list, occurrence_index);

  if (event_type == PALLAS_BLOCK_END) {
    recordExitFunction();
  }
  return occurrence_index;
}

void ThreadWriter::threadClose() {
  while (cur_depth > 0) {
    pallas_warn("Closing unfinished sequence (lvl %d)\n", cur_depth);
    recordExitFunction();
  }
  thread_trace.finalizeThread();
}

void Archive::open(const char* dirname, const char* given_trace_name, LocationGroupId archive_id) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;
  pallas_debug_level_init();

  dir_name = strdup(dirname);
  trace_name = strdup(given_trace_name);
  fullpath = pallas_archive_fullpath(dir_name, trace_name);
  id = archive_id;
  global_archive = nullptr;

  pthread_mutex_init(&lock, nullptr);

  nb_allocated_threads = NB_THREADS_DEFAULT;
  nb_threads = 0;
  threads = new Thread*[nb_allocated_threads];

  pallas_storage_init(this);

  pallas_recursion_shield--;
}

void ThreadWriter::open(Archive* archive, ThreadId thread_id) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;

  pallas_assert(archive->getThread(thread_id) == nullptr);

  pallas_log(DebugLevel::Debug, "pallas_write_thread_open(%ux)\n", thread_id);

  thread_trace.initThread(archive, thread_id);
  max_depth = CALLSTACK_DEPTH_DEFAULT;
  og_seq = new Sequence*[max_depth];

  // the main sequence is in sequences[0]
  og_seq[0] = thread_trace.sequences[0];
  thread_trace.nb_sequences = 1;

  for (int i = 1; i < max_depth; i++) {
    og_seq[i] = new Sequence();
  }

  last_timestamp = PALLAS_TIMESTAMP_INVALID;
  last_duration = nullptr;
  sequence_start_timestamp = new pallas_timestamp_t[max_depth];

  cur_depth = 0;

  pallas_recursion_shield--;
}

/**
 * Creates a new LocationGroup and adds it to that Archive.
 */
void Archive::defineLocationGroup(LocationGroupId id, StringRef name, LocationGroupId parent) {
  pthread_mutex_lock(&lock);
  LocationGroup l = LocationGroup();
  l.id = id;
  l.name = name;
  l.parent = parent;
  location_groups.push_back(l);
  pthread_mutex_unlock(&lock);
}

/**
 * Creates a new Location and adds it to that Archive.
 */
void Archive::defineLocation(ThreadId id, StringRef name, LocationGroupId parent) {
  pthread_mutex_lock(&lock);
  Location l = Location();
  l.id = id;
  pallas_assert(l.id != PALLAS_THREAD_ID_INVALID);
  l.name = name;
  l.parent = parent;
  locations.push_back(l);
  pthread_mutex_unlock(&lock);
}

void Archive::close() {
  pallas_storage_finalize(this);
}

static inline void init_event(Event* e, enum Record record) {
  e->event_size = offsetof(Event, event_data);
  e->record = record;
  memset(&e->event_data[0], 0, sizeof(e->event_data));
}

static inline void push_data(Event* e, void* data, size_t data_size) {
  size_t o = e->event_size - offsetof(Event, event_data);
  pallas_assert(o < 256);
  pallas_assert(o + data_size < 256);
  memcpy(&e->event_data[o], data, data_size);
  e->event_size += data_size;
}

static inline void pop_data(Event* e, void* data, size_t data_size, byte*& cursor) {
  if (cursor == nullptr) {
    /* initialize the cursor to the begining of event data */
    cursor = &e->event_data[0];
  }

  uintptr_t last_event_byte = ((uintptr_t)e) + e->event_size;
  uintptr_t last_read_byte = ((uintptr_t)cursor) + data_size;
  pallas_assert(last_read_byte <= last_event_byte);

  memcpy(data, cursor, data_size);
  cursor += data_size;
}

void Thread::printEvent(pallas::Event* e) const {
  char output_str[1024];
  size_t buffer_size=1024;
  printEventToString(e, output_str, buffer_size);
  printf(output_str);
}

void Thread::printEventToString(pallas::Event* e, char* output_str, size_t buffer_size) const {
  byte* cursor = nullptr;
  switch (e->record) {
  case PALLAS_EVENT_ENTER: {
    RegionRef region_ref;
    pop_data(e, &region_ref, sizeof(region_ref), cursor);
    const Region* region = archive->getRegion(region_ref);
    const char* region_name = region ? archive->getString(region->string_ref)->str : "INVALID";
    snprintf(output_str, buffer_size, "Enter %d (%s)", region_ref, region_name);
    break;
  }
  case PALLAS_EVENT_LEAVE: {
    RegionRef region_ref;
    pop_data(e, &region_ref, sizeof(region_ref), cursor);
    const Region* region = archive->getRegion(region_ref);
    const char* region_name = region ? archive->getString(region->string_ref)->str : "INVALID";
    snprintf(output_str, buffer_size, "Leave %d (%s)", region_ref, region_name);
    break;
  }

  case PALLAS_EVENT_THREAD_BEGIN:
    snprintf(output_str, buffer_size, "THREAD_BEGIN()");
    break;

  case PALLAS_EVENT_THREAD_END:
    snprintf(output_str, buffer_size, "THREAD_END()");
    break;

  case PALLAS_EVENT_THREAD_TEAM_BEGIN:
    snprintf(output_str, buffer_size, "THREAD_TEAM_BEGIN()");
    break;

  case PALLAS_EVENT_THREAD_TEAM_END:
    snprintf(output_str, buffer_size, "THREAD_TEAM_END()");
    break;

  case PALLAS_EVENT_MPI_SEND: {
    uint32_t receiver;
    uint32_t communicator;
    uint32_t msgTag;
    uint64_t msgLength;

    pop_data(e, &receiver, sizeof(receiver), cursor);
    pop_data(e, &communicator, sizeof(communicator), cursor);
    pop_data(e, &msgTag, sizeof(msgTag), cursor);
    pop_data(e, &msgLength, sizeof(msgLength), cursor);
    snprintf(output_str, buffer_size, "MPI_SEND(dest=%d, comm=%x, tag=%x, len=%" PRIu64 ")", receiver, communicator, msgTag, msgLength);
    break;
  }
  case PALLAS_EVENT_MPI_ISEND: {
    uint32_t receiver;
    uint32_t communicator;
    uint32_t msgTag;
    uint64_t msgLength;
    uint64_t requestID;

    pop_data(e, &receiver, sizeof(receiver), cursor);
    pop_data(e, &communicator, sizeof(communicator), cursor);
    pop_data(e, &msgTag, sizeof(msgTag), cursor);
    pop_data(e, &msgLength, sizeof(msgLength), cursor);
    pop_data(e, &requestID, sizeof(requestID), cursor);
    snprintf(output_str, buffer_size, "MPI_ISEND(dest=%d, comm=%x, tag=%x, len=%" PRIu64 ", req=%" PRIx64 ")", receiver, communicator, msgTag,
	       msgLength, requestID);
    break;
  }
  case PALLAS_EVENT_MPI_ISEND_COMPLETE: {
    uint64_t requestID;
    pop_data(e, &requestID, sizeof(requestID), cursor);
    snprintf(output_str, buffer_size, "MPI_ISEND_COMPLETE(req=%" PRIx64 ")", requestID);
    break;
  }
  case PALLAS_EVENT_MPI_IRECV_REQUEST: {
    uint64_t requestID;
    pop_data(e, &requestID, sizeof(requestID), cursor);
    snprintf(output_str, buffer_size, "MPI_IRECV_REQUEST(req=%" PRIx64 ")", requestID);
    break;
  }
  case PALLAS_EVENT_MPI_RECV: {
    uint32_t sender;
    uint32_t communicator;
    uint32_t msgTag;
    uint64_t msgLength;

    pop_data(e, &sender, sizeof(sender), cursor);
    pop_data(e, &communicator, sizeof(communicator), cursor);
    pop_data(e, &msgTag, sizeof(msgTag), cursor);
    pop_data(e, &msgLength, sizeof(msgLength), cursor);

    snprintf(output_str, buffer_size, "MPI_RECV(src=%d, comm=%x, tag=%x, len=%" PRIu64 ")", sender, communicator, msgTag, msgLength);
    break;
  }
  case PALLAS_EVENT_MPI_IRECV: {
    uint32_t sender;
    uint32_t communicator;
    uint32_t msgTag;
    uint64_t msgLength;
    uint64_t requestID;
    pop_data(e, &sender, sizeof(sender), cursor);
    pop_data(e, &communicator, sizeof(communicator), cursor);
    pop_data(e, &msgTag, sizeof(msgTag), cursor);
    pop_data(e, &msgLength, sizeof(msgLength), cursor);
    pop_data(e, &requestID, sizeof(requestID), cursor);

    snprintf(output_str, buffer_size, "MPI_IRECV(src=%d, comm=%x, tag=%x, len=%" PRIu64 ", req=%" PRIu64 ")", sender, communicator, msgTag,
	       msgLength, requestID);
    break;
  }
  case PALLAS_EVENT_MPI_COLLECTIVE_BEGIN: {
    snprintf(output_str, buffer_size, "MPI_COLLECTIVE_BEGIN()");
    break;
  }
  case PALLAS_EVENT_MPI_COLLECTIVE_END: {
    uint32_t collectiveOp;
    uint32_t communicator;
    uint32_t root;
    uint64_t sizeSent;
    uint64_t sizeReceived;

    pop_data(e, &collectiveOp, sizeof(collectiveOp), cursor);
    pop_data(e, &communicator, sizeof(communicator), cursor);
    pop_data(e, &root, sizeof(root), cursor);
    pop_data(e, &sizeSent, sizeof(sizeSent), cursor);
    pop_data(e, &sizeReceived, sizeof(sizeReceived), cursor);

    snprintf(output_str, buffer_size, "MPI_COLLECTIVE_END(op=%x, comm=%x, root=%d, sent=%" PRIu64 ", recved=%" PRIu64 ")", collectiveOp,
	       communicator, root, sizeSent, sizeReceived);
    break;
  }
  default:
    snprintf(output_str, buffer_size, "{.record: %x, .size:%x}", e->record, e->event_size);
  }
}

void EventSummary::initEventSummary(TokenId token_id, const Event& e) {
  id = token_id;
  nb_occurences = 0;
  attribute_buffer = 0;
  attribute_buffer_size = 0;
  attribute_pos = 0;
  memcpy(&event, &e, sizeof(e));
}

TokenId Thread::getEventId(pallas::Event* e) {
  pallas_log(DebugLevel::Max, "Searching for event {.event_type=%d}\n", e->record);

  pallas_assert(e->event_size < 256);

  for (TokenId i = 0; i < nb_events; i++) {
    if (memcmp(e, &events[i].event, e->event_size) == 0) {
      pallas_log(DebugLevel::Max, "\t found with id=%u\n", i);
      return i;
    }
  }

  if (nb_events >= nb_allocated_events) {
    pallas_warn("Doubling mem space of events for thread trace %p\n", this);
    DOUBLE_MEMORY_SPACE(events, nb_allocated_events, EventSummary);
  }

  TokenId index = nb_events++;
  pallas_log(DebugLevel::Max, "\tNot found. Adding it with id=%x\n", index);
  auto* new_event = &events[index];
  new_event->initEventSummary(id, *e);

  return index;
}
pallas_duration_t Thread::getSequenceDuration(Token* array, size_t size, bool ignoreLast) {
  pallas_duration_t sum = 0;
  auto tokenCount = TokenCountMap();
  size_t i = size;
  do {
    i--;
    auto& token = array[i];
    tokenCount[token]++;
    if (ignoreLast && i == size - 1)
      continue;
    switch (token.type) {
    case TypeInvalid: {
      pallas_error("Error parsing the given array, a Token was invalid\n");
      break;
    }
    case TypeEvent: {
      auto summary = getEventSummary(token);
      sum += summary->durations->at(summary->durations->size - tokenCount[token]);
      break;
    }
    case TypeSequence: {
      auto sequence = getSequence(token);
      sum += sequence->durations->at(sequence->durations->size - tokenCount[token]);
      tokenCount += sequence->getTokenCount(this);
      break;
    }
    case TypeLoop: {
      auto loop = getLoop(token);
      auto nb_iterations = loop->nb_iterations[loop->nb_iterations.size() - tokenCount[token]];
      auto sequence = getSequence(loop->repeated_token);
      for (size_t j = 0; j < nb_iterations; j++) {
        tokenCount[loop->repeated_token]++;
        sum += sequence->durations->at(sequence->durations->size - tokenCount[loop->repeated_token]);
      }
      tokenCount += sequence->getTokenCount(this) * (size_t)nb_iterations;
      break;
    }
    }
  } while (i != 0);
  return sum;
}
}  // namespace pallas

/* C Callbacks */
pallas::ThreadWriter* pallas_thread_writer_new() {
  return new pallas::ThreadWriter();
}

extern void pallas_write_global_archive_open(pallas::Archive* archive, const char* dir_name, const char* trace_name) {
  archive->globalOpen(dir_name, trace_name);
};
extern void pallas_write_global_archive_close(pallas::Archive* archive) {
  archive->close();
};

extern void pallas_write_thread_open(pallas::Archive* archive,
                                     pallas::ThreadWriter* thread_writer,
                                     pallas::ThreadId thread_id) {
  thread_writer->open(archive, thread_id);
};

extern void pallas_write_thread_close(pallas::ThreadWriter* thread_writer) {
  thread_writer->threadClose();
};

extern void pallas_write_define_location_group(pallas::Archive* archive,
                                               pallas::LocationGroupId id,
                                               pallas::StringRef name,
                                               pallas::LocationGroupId parent) {
  archive->defineLocationGroup(id, name, parent);
};

extern void pallas_write_define_location(pallas::Archive* archive,
                                         pallas::ThreadId id,
                                         pallas::StringRef name,
                                         pallas::LocationGroupId parent) {
  archive->defineLocation(id, name, parent);
};

extern void pallas_write_archive_open(pallas::Archive* archive,
                                      const char* dir_name,
                                      const char* trace_name,
                                      pallas::LocationGroupId location_group) {
  archive->open(dir_name, trace_name, location_group);
};

extern void pallas_write_archive_close(PALLAS(Archive) * archive) {
  archive->close();
};

void pallas_store_event(PALLAS(ThreadWriter) * thread_writer,
                        enum PALLAS(EventType) event_type,
                        PALLAS(TokenId) id,
                        pallas_timestamp_t ts,
                        PALLAS(AttributeList) * attribute_list) {
  thread_writer->storeEvent(event_type, id, ts, attribute_list);
};

void pallas_record_enter(pallas::ThreadWriter* thread_writer,
                         struct pallas::AttributeList* attribute_list __attribute__((unused)),
                         pallas_timestamp_t time,
                         pallas::RegionRef region_ref) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;

  pallas::Event e;
  init_event(&e, pallas::PALLAS_EVENT_ENTER);

  push_data(&e, &region_ref, sizeof(region_ref));

  pallas::TokenId e_id = thread_writer->thread_trace.getEventId(&e);

  thread_writer->storeEvent(pallas::PALLAS_BLOCK_START, e_id, time, attribute_list);

  pallas_recursion_shield--;
}

void pallas_record_leave(pallas::ThreadWriter* thread_writer,
                         struct pallas::AttributeList* attribute_list __attribute__((unused)),
                         pallas_timestamp_t time,
                         pallas::RegionRef region_ref) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;

  pallas::Event e;
  init_event(&e, pallas::PALLAS_EVENT_LEAVE);

  push_data(&e, &region_ref, sizeof(region_ref));

  pallas::TokenId e_id = thread_writer->thread_trace.getEventId(&e);
  thread_writer->storeEvent(pallas::PALLAS_BLOCK_END, e_id, time, attribute_list);

  pallas_recursion_shield--;
}

void pallas_record_thread_begin(pallas::ThreadWriter* thread_writer,
                                struct pallas::AttributeList* attribute_list __attribute__((unused)),
                                pallas_timestamp_t time) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;

  pallas::Event e;
  init_event(&e, pallas::PALLAS_EVENT_THREAD_BEGIN);

  pallas::TokenId e_id = thread_writer->thread_trace.getEventId(&e);
  thread_writer->storeEvent(pallas::PALLAS_BLOCK_START, e_id, time, attribute_list);

  pallas_recursion_shield--;
}

void pallas_record_thread_end(pallas::ThreadWriter* thread_writer,
                              struct pallas::AttributeList* attribute_list __attribute__((unused)),
                              pallas_timestamp_t time) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;

  pallas::Event e;
  init_event(&e, pallas::PALLAS_EVENT_THREAD_END);
  pallas::TokenId e_id = thread_writer->thread_trace.getEventId(&e);
  thread_writer->storeEvent(pallas::PALLAS_BLOCK_END, e_id, time, attribute_list);

  pallas_recursion_shield--;
}

void pallas_record_thread_team_begin(pallas::ThreadWriter* thread_writer,
                                     struct pallas::AttributeList* attribute_list __attribute__((unused)),
                                     pallas_timestamp_t time) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;

  pallas::Event e;
  init_event(&e, pallas::PALLAS_EVENT_THREAD_TEAM_BEGIN);
  pallas::TokenId e_id = thread_writer->thread_trace.getEventId(&e);
  thread_writer->storeEvent(pallas::PALLAS_BLOCK_START, e_id, time, attribute_list);

  pallas_recursion_shield--;
}

void pallas_record_thread_team_end(pallas::ThreadWriter* thread_writer,
                                   struct pallas::AttributeList* attribute_list __attribute__((unused)),
                                   pallas_timestamp_t time) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;

  pallas::Event e;
  init_event(&e, pallas::PALLAS_EVENT_THREAD_TEAM_END);
  pallas::TokenId e_id = thread_writer->thread_trace.getEventId(&e);
  thread_writer->storeEvent(pallas::PALLAS_BLOCK_END, e_id, time, attribute_list);

  pallas_recursion_shield--;
}

void pallas_record_mpi_send(pallas::ThreadWriter* thread_writer,
                            struct pallas::AttributeList* attribute_list __attribute__((unused)),
                            pallas_timestamp_t time,
                            uint32_t receiver,
                            uint32_t communicator,
                            uint32_t msgTag,
                            uint64_t msgLength) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;

  pallas::Event e;
  init_event(&e, pallas::PALLAS_EVENT_MPI_SEND);

  push_data(&e, &receiver, sizeof(receiver));
  push_data(&e, &communicator, sizeof(communicator));
  push_data(&e, &msgTag, sizeof(msgTag));
  push_data(&e, &msgLength, sizeof(msgLength));

  pallas::TokenId e_id = thread_writer->thread_trace.getEventId(&e);
  thread_writer->storeEvent(pallas::PALLAS_SINGLETON, e_id, time, attribute_list);

  pallas_recursion_shield--;
  return;
}

void pallas_record_mpi_isend(pallas::ThreadWriter* thread_writer,
                             struct pallas::AttributeList* attribute_list __attribute__((unused)),
                             pallas_timestamp_t time,
                             uint32_t receiver,
                             uint32_t communicator,
                             uint32_t msgTag,
                             uint64_t msgLength,
                             uint64_t requestID) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;

  pallas::Event e;
  init_event(&e, pallas::PALLAS_EVENT_MPI_ISEND);

  push_data(&e, &receiver, sizeof(receiver));
  push_data(&e, &communicator, sizeof(communicator));
  push_data(&e, &msgTag, sizeof(msgTag));
  push_data(&e, &msgLength, sizeof(msgLength));
  push_data(&e, &requestID, sizeof(requestID));

  pallas::TokenId e_id = thread_writer->thread_trace.getEventId(&e);
  thread_writer->storeEvent(pallas::PALLAS_SINGLETON, e_id, time, attribute_list);

  pallas_recursion_shield--;
  return;
}

void pallas_record_mpi_isend_complete(pallas::ThreadWriter* thread_writer,
                                      struct pallas::AttributeList* attribute_list __attribute__((unused)),
                                      pallas_timestamp_t time,
                                      uint64_t requestID) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;

  pallas::Event e;
  init_event(&e, pallas::PALLAS_EVENT_MPI_ISEND_COMPLETE);

  push_data(&e, &requestID, sizeof(requestID));

  pallas::TokenId e_id = thread_writer->thread_trace.getEventId(&e);
  thread_writer->storeEvent(pallas::PALLAS_SINGLETON, e_id, time, attribute_list);

  pallas_recursion_shield--;
  return;
}

void pallas_record_mpi_irecv_request(pallas::ThreadWriter* thread_writer,
                                     struct pallas::AttributeList* attribute_list __attribute__((unused)),
                                     pallas_timestamp_t time,
                                     uint64_t requestID) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;

  pallas::Event e;
  init_event(&e, pallas::PALLAS_EVENT_MPI_IRECV_REQUEST);

  push_data(&e, &requestID, sizeof(requestID));

  pallas::TokenId e_id = thread_writer->thread_trace.getEventId(&e);
  thread_writer->storeEvent(pallas::PALLAS_SINGLETON, e_id, time, attribute_list);

  pallas_recursion_shield--;
  return;
}

void pallas_record_mpi_recv(pallas::ThreadWriter* thread_writer,
                            struct pallas::AttributeList* attribute_list __attribute__((unused)),
                            pallas_timestamp_t time,
                            uint32_t sender,
                            uint32_t communicator,
                            uint32_t msgTag,
                            uint64_t msgLength) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;

  pallas::Event e;
  init_event(&e, pallas::PALLAS_EVENT_MPI_RECV);

  push_data(&e, &sender, sizeof(sender));
  push_data(&e, &communicator, sizeof(communicator));
  push_data(&e, &msgTag, sizeof(msgTag));
  push_data(&e, &msgLength, sizeof(msgLength));

  pallas::TokenId e_id = thread_writer->thread_trace.getEventId(&e);
  thread_writer->storeEvent(pallas::PALLAS_SINGLETON, e_id, time, attribute_list);

  pallas_recursion_shield--;
  return;
}

void pallas_record_mpi_irecv(pallas::ThreadWriter* thread_writer,
                             struct pallas::AttributeList* attribute_list __attribute__((unused)),
                             pallas_timestamp_t time,
                             uint32_t sender,
                             uint32_t communicator,
                             uint32_t msgTag,
                             uint64_t msgLength,
                             uint64_t requestID) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;

  pallas::Event e;
  init_event(&e, pallas::PALLAS_EVENT_MPI_IRECV);

  push_data(&e, &sender, sizeof(sender));
  push_data(&e, &communicator, sizeof(communicator));
  push_data(&e, &msgTag, sizeof(msgTag));
  push_data(&e, &msgLength, sizeof(msgLength));
  push_data(&e, &requestID, sizeof(requestID));

  pallas::TokenId e_id = thread_writer->thread_trace.getEventId(&e);
  thread_writer->storeEvent(pallas::PALLAS_SINGLETON, e_id, time, attribute_list);

  pallas_recursion_shield--;
  return;
}

void pallas_record_mpi_collective_begin(pallas::ThreadWriter* thread_writer,
                                        struct pallas::AttributeList* attribute_list __attribute__((unused)),
                                        pallas_timestamp_t time) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;

  pallas::Event e;
  init_event(&e, pallas::PALLAS_EVENT_MPI_COLLECTIVE_BEGIN);

  pallas::TokenId e_id = thread_writer->thread_trace.getEventId(&e);
  thread_writer->storeEvent(pallas::PALLAS_SINGLETON, e_id, time, attribute_list);

  pallas_recursion_shield--;
  return;
}

void pallas_record_mpi_collective_end(pallas::ThreadWriter* thread_writer,
                                      struct pallas::AttributeList* attribute_list __attribute__((unused)),
                                      pallas_timestamp_t time,
                                      uint32_t collectiveOp,
                                      uint32_t communicator,
                                      uint32_t root,
                                      uint64_t sizeSent,
                                      uint64_t sizeReceived) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;

  pallas::Event e;
  init_event(&e, pallas::PALLAS_EVENT_MPI_COLLECTIVE_END);

  push_data(&e, &collectiveOp, sizeof(collectiveOp));
  push_data(&e, &communicator, sizeof(communicator));
  push_data(&e, &root, sizeof(root));
  push_data(&e, &sizeSent, sizeof(sizeSent));
  push_data(&e, &sizeReceived, sizeof(sizeReceived));

  pallas::TokenId e_id = thread_writer->thread_trace.getEventId(&e);
  thread_writer->storeEvent(pallas::PALLAS_SINGLETON, e_id, time, attribute_list);

  pallas_recursion_shield--;
}

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */
