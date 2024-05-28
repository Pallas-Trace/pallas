/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include <cinttypes>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "pallas/pallas.h"
#include "pallas/pallas_archive.h"
#include "pallas/pallas_hash.h"
#include "pallas/pallas_parameter_handler.h"
#include "pallas/pallas_storage.h"
#include "pallas/pallas_timestamp.h"
#include "pallas/pallas_write.h"
thread_local int pallas_recursion_shield = 0;
namespace pallas {
/**
 * Compares two arrays of tokens array1 and array2
 */
static inline bool _pallas_arrays_equal(Token* array1, size_t size1, Token* array2, size_t size2) {
  if (size1 != size2)
    return false;
  return memcmp(array1, array2, sizeof(Token) * size1) == 0;
}

Token Thread::getSequenceIdFromArray(pallas::Token* token_array, size_t array_len) {
  if (array_len == 1 && token_array[0].type == TypeSequence) {
    pallas_log(DebugLevel::Debug, "getSequenceIdFromArray: Searching for sequence {.size=1} containing sequence token\n");
    return token_array[0];
  }
  uint32_t hash;
  hash32(token_array, array_len, SEED, &hash);
  pallas_log(DebugLevel::Debug, "getSequenceIdFromArray: Searching for sequence {.size=%zu, .hash=%x}\n", array_len, hash);
  auto& sequencesWithSameHash = hashToSequence[hash];
  if (!sequencesWithSameHash.empty()) {
    if (sequencesWithSameHash.size() > 1) {
      pallas_warn("Found more than one sequence with the same hash\n");
    }
    for (const auto sid : sequencesWithSameHash) {
      if (_pallas_arrays_equal(token_array, array_len, sequences[sid]->tokens.data(), sequences[sid]->size())) {
        pallas_log(DebugLevel::Debug, "getSequenceIdFromArray: \t found with id=%u\n", sid);
        return PALLAS_SEQUENCE_ID(sid);
      }
    }
  }

  if (nb_sequences >= nb_allocated_sequences) {
    pallas_log(DebugLevel::Debug, "Doubling mem space of sequence for thread trace %p\n", this);
    DOUBLE_MEMORY_SPACE(sequences, nb_allocated_sequences, Sequence*);
    for (uint i = nb_allocated_sequences / 2; i < nb_allocated_sequences; i++) {
      sequences[i] = new Sequence;
    }
  }

  const auto index = nb_sequences++;
  const auto sid = PALLAS_SEQUENCE_ID(index);
  pallas_log(DebugLevel::Debug, "getSequenceIdFromArray: \tSequence not found. Adding it with id=S%x\n", index);

  Sequence* s = getSequence(sid);
  s->tokens.resize(array_len);
  memcpy(s->tokens.data(), token_array, sizeof(Token) * array_len);
  s->hash = hash;
  s->id = sid.id;
  sequencesWithSameHash.push_back(index);

  return sid;
}

Loop* ThreadWriter::createLoop(size_t start_index, size_t loop_len) {
  auto& curTokenSeq = getCurrentTokenSequence();
  const Token sid = thread_trace.getSequenceIdFromArray(&curTokenSeq[start_index], loop_len);

  int index = -1;
  for (int i = 0; i < thread_trace.nb_loops; i++) {
    if (thread_trace.loops[i].repeated_token.id == sid.id) {
      index = i;
      pallas_log(DebugLevel::Debug, "createLoop:\tLoop already exists: id=L%d containing S%d\n", index, sid.id);
      break;
    }
  }
  if (index == -1) {
    if (thread_trace.nb_loops >= thread_trace.nb_allocated_loops) {
      pallas_log(DebugLevel::Debug, "Doubling mem space of loops for thread writer %p's thread trace, cur=%d\n", this,
                  thread_trace.nb_allocated_loops);
      DOUBLE_MEMORY_SPACE_CONSTRUCTOR(thread_trace.loops, thread_trace.nb_allocated_loops, Loop);
    }
    index = thread_trace.nb_loops++;
    pallas_log(DebugLevel::Debug, "createLoop:\tLoop not found. Adding it with id=L%d containing S%d\n", index, sid.id);
  }

  Loop* l = &thread_trace.loops[index];
  l->nb_iterations.push_back(1);
  l->repeated_token = sid;
  l->self_id = PALLAS_LOOP_ID(index);
  return l;
}

void ThreadWriter::storeTimestamp(EventSummary* es, pallas_timestamp_t ts) {
  int store_event_durations = 1;
  if (store_event_durations) {
    // update the last event's duration
    if (last_duration) {
      const pallas_timestamp_t delta = pallas_get_duration(last_timestamp, ts);
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
                                      const size_t occurence_index) {
  attribute_list->index = occurence_index;
  if (es->attribute_pos + attribute_list->struct_size >= es->attribute_buffer_size) {
    if (es->attribute_buffer_size == 0) {
      pallas_log(DebugLevel::Debug, "Allocating attribute memory for event %u\n", es->id);
      es->attribute_buffer_size = NB_ATTRIBUTE_DEFAULT * sizeof(struct pallas::AttributeList);
      es->attribute_buffer = new uint8_t[es->attribute_buffer_size];
      pallas_assert(es->attribute_buffer != nullptr);
    } else {
      pallas_log(DebugLevel::Debug, "Doubling mem space of attributes for event %u\n", es->id);
      DOUBLE_MEMORY_SPACE(es->attribute_buffer, es->attribute_buffer_size, uint8_t);
    }
    pallas_assert(es->attribute_pos + attribute_list->struct_size < es->attribute_buffer_size);
  }

  memcpy(&es->attribute_buffer[es->attribute_pos], attribute_list, attribute_list->struct_size);
  es->attribute_pos += attribute_list->struct_size;

  pallas_log(DebugLevel::Debug, "storeAttributeList: {index: %d, struct_size: %d, nb_values: %d}\n", attribute_list->index,
             attribute_list->struct_size, attribute_list->nb_values);
}

void ThreadWriter::storeToken(std::vector<Token>& tokenSeq, Token t) {
  pallas_log(DebugLevel::Debug, "storeToken: (%c%d) in seq at callstack[%d] (size: %zu)\n",
             PALLAS_TOKEN_TYPE_C(t), t.id, cur_depth, tokenSeq.size() + 1);
  tokenSeq.push_back(t);
  findLoop();
}

/**
 * Adds an iteration of the given sequence to the loop.
 */
void Loop::addIteration() {
  pallas_log(DebugLevel::Debug, "addIteration: + 1 to L%d n°%zu (to %u)\n", self_id.id, nb_iterations.size() - 1,
             nb_iterations.back() + 1);
  nb_iterations.back()++;
}

void ThreadWriter::replaceTokensInLoop(int loop_len, size_t index_first_iteration, size_t index_second_iteration) {
  if (index_first_iteration > index_second_iteration) {
    const size_t tmp = index_second_iteration;
    index_second_iteration = index_first_iteration;
    index_first_iteration = tmp;
  }

  Loop* loop = createLoop(index_first_iteration, loop_len);
  auto& curTokenSeq = getCurrentTokenSequence();

  if (loop_len > 1 || curTokenSeq[index_first_iteration].type != TypeSequence) {
    // We need to go back in the current sequence in order to correctly calculate our durations
    // But only if those are new sequences
    const Sequence* loop_seq = thread_trace.getSequence(loop->repeated_token);

    const pallas_duration_t duration_first_iteration =
      thread_trace.getSequenceDuration(&curTokenSeq[index_first_iteration], 2 * loop_len, true);
    const pallas_duration_t duration_second_iteration =
      thread_trace.getSequenceDuration(&curTokenSeq[index_second_iteration], loop_len, true);
    // We don't take into account the last token because it's not a duration yet

    loop_seq->durations->add(duration_first_iteration - duration_second_iteration);
    addDurationToComplete(loop_seq->durations->add(duration_second_iteration));
  }

  // The current sequence last_timestamp does not need to be updated

  curTokenSeq.resize(index_first_iteration);
  curTokenSeq.push_back(loop->self_id);

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
  auto& curTokenSeq = getCurrentTokenSequence();
  const size_t currentIndex = curTokenSeq.size() - 1;
  for (int loopLength = 1; loopLength < maxLoopLength && loopLength <= currentIndex; loopLength++) {
    // search for a loop of loopLength tokens
    const size_t startS1 = currentIndex + 1 - loopLength;
    // First, check if there's a loop that start at loopStart
    if (const size_t loopStart = startS1 - 1;
      curTokenSeq[loopStart].type == TypeLoop) {
      const Token l = curTokenSeq[loopStart];
      Loop* loop = thread_trace.getLoop(l);
      pallas_assert(loop);

      Sequence* loopSeq = thread_trace.getSequence(loop->repeated_token);
      pallas_assert(loopSeq);

      // First check for repetitions of sequences
      if (loopLength == 1 && curTokenSeq[startS1] == loop->repeated_token) {
        pallas_log(DebugLevel::Debug, "findLoopBasic: Last token was the sequence from L%d: S%d\n", loop->self_id.id, loop->repeated_token.id);
        loop->addIteration();
        curTokenSeq.resize(startS1);
        return;
      }
      // Then check for actual iterations of tokens which correspond to the one in the loop
      if (_pallas_arrays_equal(&curTokenSeq[startS1], loopLength, loopSeq->tokens.data(), loopSeq->size())) {
        pallas_log(DebugLevel::Debug, "findLoopBasic: Last tokens were a sequence from L%d aka S%d\n", loop->self_id.id,
                   loop->repeated_token.id);
        loop->addIteration();
        const pallas_timestamp_t ts = thread_trace.getSequenceDuration(&curTokenSeq[startS1], loopLength, true);
        addDurationToComplete(loopSeq->durations->add(ts));
        curTokenSeq.resize(startS1);
        // Roundabount way to remove the tokens representing the loop
        return;
      }
    }

    if (currentIndex + 1 >= 2 * loopLength) {
      const size_t startS2 = currentIndex + 1 - 2 * loopLength;
      /* search for a loop of loopLength tokens */
      if (_pallas_arrays_equal(&curTokenSeq[startS1], loopLength, &curTokenSeq[startS2], loopLength)) {
        if (debugLevel >= DebugLevel::Debug) {
          pallas_log(DebugLevel::Debug, "findLoopBasic: Found a loop of len %d:", loopLength);
          thread_trace.printTokenArray(curTokenSeq.data(), startS2, loopLength * 2);
          printf("\n");
        }
        replaceTokensInLoop(loopLength, startS1, startS2);
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
  auto& curTokenSeq = getCurrentTokenSequence();
  size_t curIndex = curTokenSeq.size() - 1;
  for (auto token : curTokenSeq) {
    if (token == curTokenSeq.back()) {
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
    if (_pallas_arrays_equal(&curTokenSeq[endingIndex + 1], loopLength, &curTokenSeq[endingIndex + 1 - loopLength],
                             loopLength)) {
      if (debugLevel >= DebugLevel::Debug) {
        printf("findLoopFilter: Found a loop of len %lu:\n", loopLength);
        thread_trace.printTokenArray(curTokenSeq.data(), endingIndex + 1, loopLength);
        thread_trace.printTokenArray(curTokenSeq.data(), endingIndex + 1 - loopLength, loopLength);
        printf("\n");
      }
      replaceTokensInLoop(loopLength, endingIndex + 1, endingIndex + 1 - loopLength);
    }
  }

  for (auto loopIndex : loopIndexes) {
    Token token = curTokenSeq[loopIndex];
    size_t loopLength = curIndex - loopIndex;
    auto* loop = thread_trace.getLoop(token);
    auto* sequence = thread_trace.getSequence(loop->repeated_token);
    if (_pallas_arrays_equal(&curTokenSeq[loopIndex + 1], loopLength, sequence->tokens.data(), sequence->size())) {
      pallas_log(DebugLevel::Debug, "findLoopFilter: Last tokens were a sequence from L%d aka S%d\n", loop->self_id.id,
                 loop->repeated_token.id);
      loop->addIteration();
      // The current sequence last_timestamp does not need to be updated

      pallas_timestamp_t ts = thread_trace.getSequenceDuration(&curTokenSeq[loopIndex + 1], loopLength, true);
      addDurationToComplete(sequence->durations->add(ts));
      curTokenSeq.resize(loopIndex + 1);
      return;
    } else if (loopLength == 1 && curTokenSeq[loopIndex + 1].type == TypeSequence && curTokenSeq[loopIndex + 1].id == sequence->id) {
      pallas_log(DebugLevel::Debug, "findLoopFilter: Last token was the sequence from L%d: S%d\n", loop->self_id.id, loop->repeated_token.id);
      loop->addIteration();
      curTokenSeq.resize(loopIndex + 1);
      return;
    }
  }
}

void ThreadWriter::findLoop() {
  if (parameterHandler->getLoopFindingAlgorithm() == LoopFindingAlgorithm::None) {
    return;
  }

  auto& curTokenSeq = getCurrentTokenSequence();
  size_t currentIndex = curTokenSeq.size() - 1;

  switch (parameterHandler->getLoopFindingAlgorithm()) {
  case LoopFindingAlgorithm::None:
    return;
  case LoopFindingAlgorithm::Basic:
  case LoopFindingAlgorithm::BasicTruncated: {
    size_t maxLoopLength = (parameterHandler->getLoopFindingAlgorithm() == LoopFindingAlgorithm::BasicTruncated)
                             ? parameterHandler->getMaxLoopLength()
                             : SIZE_MAX;
    if (debugLevel >= DebugLevel::Debug) {
      printf("findLoop: Using Basic Algorithm:\n");
      size_t start_index = (currentIndex >= maxLoopLength) ? currentIndex - maxLoopLength : 0;
      size_t len = (currentIndex <= maxLoopLength) ? currentIndex + 1 : maxLoopLength;
      thread_trace.printTokenArray(curTokenSeq.data(), start_index, len);
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
  auto& curTokenSeq = getCurrentTokenSequence();

#ifdef DEBUG
  // check that the sequence is not bugous

  Token first_token = curTokenSeq.front();
  Token last_token = curTokenSeq.back();
  if (first_token.type != last_token.type) {
    /* If a sequence starts with an Event (eg Enter function foo), it
       should end with an Event too (eg. Exit function foo) */
    pallas_warn(
      "When closing sequence at callstack[%d]:"
      "PALLAS_TOKEN_TYPE(%c%d) != PALLAS_TOKEN_TYPE(%c%d)\n",
      cur_depth, first_token.type, first_token.id, last_token.type, last_token.id);
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

  if (curTokenSeq != sequence_stack[cur_depth]) {
    pallas_error("cur_seq=%p, but og_seq[%d] = %p\n", &curTokenSeq, cur_depth, &sequence_stack[cur_depth]);
  }
#endif

  const Token seq_id = thread_trace.getSequenceIdFromArray(curTokenSeq.data(), curTokenSeq.size());
  const auto* seq = thread_trace.sequences[seq_id.id];

  const pallas_timestamp_t sequence_duration = last_timestamp - sequence_start_timestamp[cur_depth];
  // I feel like it's a bad idea to simply... ignore the duration of the "Close" event
  addDurationToComplete(seq->durations->add(sequence_duration));

  pallas_log(DebugLevel::Debug, "Exiting a function, closing sequence %d\n", seq_id.id);

  cur_depth--;
  /* upper_seq is the sequence that called cur_seq */
  auto& upperTokenSeq = getCurrentTokenSequence();

  storeToken(upperTokenSeq, seq_id);
  curTokenSeq.resize(0);
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
  auto& curTokenSeq = getCurrentTokenSequence();

  EventSummary* es = &thread_trace.events[event_id];
  size_t occurrence_index = es->nb_occurences++;
  storeTimestamp(es, ts);
  storeToken(curTokenSeq, token);

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
  // Then we need to close the main sequence
  auto& mainSequence = thread_trace.sequences[0];
  mainSequence->tokens = sequence_stack[0];
  pallas_log(DebugLevel::Debug, "Last sequence token: (%d.%d)", mainSequence->tokens.back().type, mainSequence->tokens.back().id);
  *last_duration = 0;
  completeDurations(0);
  pallas_timestamp_t duration = last_timestamp;
  mainSequence->durations->add(duration);
  thread_trace.finalizeThread();
}

void Archive::open(const char* dirname, const char* given_trace_name, LocationGroupId archive_id) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;
  pallas_debug_level_init();
  if (!parameterHandler) {
    parameterHandler = new ParameterHandler();
  }
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

  pallas_log(DebugLevel::Debug, "ThreadWriter(%ux)::open\n", thread_id);

  thread_trace.initThread(archive, thread_id);
  max_depth = CALLSTACK_DEPTH_DEFAULT;
  sequence_stack = new std::vector<Token>[max_depth];

  // We need to initialize the main Sequence (Sequence 0)
  auto& mainSequence = thread_trace.sequences[0];
  mainSequence->id = 0;
  thread_trace.nb_sequences = 1;

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
  byte* cursor = nullptr;
  switch (e->record) {
  case PALLAS_EVENT_ENTER: {
    RegionRef region_ref;
    pop_data(e, &region_ref, sizeof(region_ref), cursor);
    const Region* region = archive->getRegion(region_ref);
    const char* region_name = region ? archive->getString(region->string_ref)->str : "INVALID";
    printf("Enter %d (%s)", region_ref, region_name);
    break;
  }
  case PALLAS_EVENT_LEAVE: {
    RegionRef region_ref;
    pop_data(e, &region_ref, sizeof(region_ref), cursor);
    const Region* region = archive->getRegion(region_ref);
    const char* region_name = region ? archive->getString(region->string_ref)->str : "INVALID";
    printf("Leave %d (%s)", region_ref, region_name);
    break;
  }

  case PALLAS_EVENT_THREAD_BEGIN:
    printf("THREAD_BEGIN()");
    break;

  case PALLAS_EVENT_THREAD_END:
    printf("THREAD_END()");
    break;

  case PALLAS_EVENT_THREAD_TEAM_BEGIN:
    printf("THREAD_TEAM_BEGIN()");
    break;

  case PALLAS_EVENT_THREAD_TEAM_END:
    printf("THREAD_TEAM_END()");
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
    printf("MPI_SEND(dest=%d, comm=%x, tag=%x, len=%" PRIu64 ")", receiver, communicator, msgTag, msgLength);
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
    printf("MPI_ISEND(dest=%d, comm=%x, tag=%x, len=%" PRIu64 ", req=%" PRIx64 ")", receiver, communicator, msgTag,
           msgLength, requestID);
    break;
  }
  case PALLAS_EVENT_MPI_ISEND_COMPLETE: {
    uint64_t requestID;
    pop_data(e, &requestID, sizeof(requestID), cursor);
    printf("MPI_ISEND_COMPLETE(req=%" PRIx64 ")", requestID);
    break;
  }
  case PALLAS_EVENT_MPI_IRECV_REQUEST: {
    uint64_t requestID;
    pop_data(e, &requestID, sizeof(requestID), cursor);
    printf("MPI_IRECV_REQUEST(req=%" PRIx64 ")", requestID);
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

    printf("MPI_RECV(src=%d, comm=%x, tag=%x, len=%" PRIu64 ")", sender, communicator, msgTag, msgLength);
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

    printf("MPI_IRECV(src=%d, comm=%x, tag=%x, len=%" PRIu64 ", req=%" PRIu64 ")", sender, communicator, msgTag,
           msgLength, requestID);
    break;
  }
  case PALLAS_EVENT_MPI_COLLECTIVE_BEGIN: {
    printf("MPI_COLLECTIVE_BEGIN()");
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

    printf("MPI_COLLECTIVE_END(op=%x, comm=%x, root=%d, sent=%" PRIu64 ", recved=%" PRIu64 ")", collectiveOp,
           communicator, root, sizeSent, sizeReceived);
    break;
  }
  default:
    printf("{.record: %x, .size:%x}", e->record, e->event_size);
  }
}

void EventSummary::initEventSummary(TokenId token_id, const Event& e) {
  id = token_id;
  nb_occurences = 0;
  attribute_buffer = nullptr;
  attribute_buffer_size = 0;
  attribute_pos = 0;
  durations = new LinkedVector();
  //TODO duration sans delta à écrire et lire 
  memcpy(&event, &e, sizeof(e));
}

TokenId Thread::getEventId(pallas::Event* e) {
  pallas_log(DebugLevel::Max, "getEventId: Searching for event {.event_type=%d}\n", e->record);

  pallas_assert(e->event_size < 256);

  for (TokenId i = 0; i < nb_events; i++) {
    if (memcmp(e, &events[i].event, e->event_size) == 0) {
      pallas_log(DebugLevel::Max, "getEventId: \t found with id=%u\n", i);
      return i;
    }
  }

  if (nb_events >= nb_allocated_events) {
    pallas_log(DebugLevel::Debug, "Doubling mem space of events for thread trace %p\n", this);
    DOUBLE_MEMORY_SPACE_CONSTRUCTOR(events, nb_allocated_events, EventSummary);
  }

  TokenId index = nb_events++;
  pallas_log(DebugLevel::Max, "getEventId: \tNot found. Adding it with id=%d\n", index);
  auto* new_event = &events[index];
  new_event->initEventSummary(id, *e);

  return index;
}
pallas_duration_t Thread::getSequenceDuration(const Token* array, size_t size, bool ignoreLastEvent) const {
  pallas_duration_t sum = 0;
  auto tokenCount = TokenCountMap();
  size_t i = size;
  do {
    i--;
    auto& token = array[i];
    tokenCount[token]++;
    if (ignoreLastEvent && i == size - 1 && token.type == TypeEvent)
      continue;
    switch (token.type) {
    case TypeInvalid: {
      pallas_error("Error parsing the given array, a Token was invalid\n");
      break;
    }
    case TypeEvent: {
      const auto summary = getEventSummary(token);
      sum += summary->durations->at(summary->durations->size - tokenCount[token]);
      break;
    }
    case TypeSequence: {
      const auto sequence = getSequence(token);
      sum += sequence->durations->at(sequence->durations->size - tokenCount[token]);
      tokenCount += sequence->getTokenCount(this);
      break;
    }
    case TypeLoop: {
      const auto loop = getLoop(token);
      const auto nb_iterations = loop->nb_iterations[loop->nb_iterations.size() - tokenCount[token]];
      const auto sequence = getSequence(loop->repeated_token);
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
