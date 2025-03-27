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
#include "pallas/pallas_log.h"
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

  uint32_t hash = hash32((uint8_t*)(token_array), array_len * sizeof(pallas::Token), SEED);
  pallas_log(DebugLevel::Debug, "getSequenceIdFromArray: Searching for sequence {.size=%zu, .hash=%x}\n", array_len, hash);
  auto& sequencesWithSameHash = hashToSequence[hash];
  if (!sequencesWithSameHash.empty()) {
    if (sequencesWithSameHash.size() > 1) {
      pallas_log(DebugLevel::Debug, "Found more than one sequence with the same hash\n");
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
    doubleMemorySpaceConstructor(sequences, nb_allocated_sequences);
    for (uint i = nb_allocated_sequences / 2; i < nb_allocated_sequences; i++) {
      sequences[i] = new Sequence;
      sequences[i]->durations = new LinkedDurationVector();
      sequences[i]->timestamps = new LinkedDurationVector();
    }
  }

  const auto index = nb_sequences++;
  const auto sid = PALLAS_SEQUENCE_ID(index);
  pallas_log(DebugLevel::Debug, "getSequenceIdFromArray: \tSequence not found. Adding it with id=S%lu\n", index);

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
  const Token sid = thread->getSequenceIdFromArray(&curTokenSeq[start_index], loop_len);

  // for (int i = 0; i < thread_trace->nb_loops; i++) {
  //   if (thread_trace->loops[i].repeated_token.id == sid.id) {
  //     index = i;
  //     pallas_log(DebugLevel::Debug, "createLoop:\tLoop already exists: id=L%d containing S%d\n", index, sid.id);
  //     break;
  //   }
  // }
  if (thread->nb_loops >= thread->nb_allocated_loops) {
    pallas_log(DebugLevel::Debug, "Doubling mem space of loops for thread writer %p's thread trace, cur=%lu\n", this, thread->nb_allocated_loops);
    doubleMemorySpaceConstructor(thread->loops, thread->nb_allocated_loops);
  }
  size_t index = thread->nb_loops++;
  pallas_log(DebugLevel::Debug, "createLoop:\tLoop not found. Adding it with id=L%lu containing S%d\n", index, sid.id);

  Loop* l = &thread->loops[index];
  l->nb_iterations = 1;
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

  if (thread->first_timestamp == PALLAS_TIMESTAMP_INVALID)
    thread->first_timestamp = ts;

  last_timestamp = ts;
}

void ThreadWriter::storeAttributeList(pallas::EventSummary* es, struct pallas::AttributeList* attribute_list, const size_t occurence_index) {
  attribute_list->index = occurence_index;
  if (es->attribute_pos + attribute_list->struct_size >= es->attribute_buffer_size) {
    if (es->attribute_buffer_size == 0) {
      pallas_log(DebugLevel::Debug, "Allocating attribute memory for event %u\n", es->id);
      es->attribute_buffer_size = NB_ATTRIBUTE_DEFAULT * sizeof(struct pallas::AttributeList);
      es->attribute_buffer = new uint8_t[es->attribute_buffer_size];
      pallas_assert(es->attribute_buffer != nullptr);
    } else {
      pallas_log(DebugLevel::Debug, "Doubling mem space of attributes for event %u\n", es->id);
      doubleMemorySpaceConstructor(es->attribute_buffer, es->attribute_buffer_size);
    }
    pallas_assert(es->attribute_pos + attribute_list->struct_size < es->attribute_buffer_size);
  }

  memcpy(&es->attribute_buffer[es->attribute_pos], attribute_list, attribute_list->struct_size);
  es->attribute_pos += attribute_list->struct_size;

  pallas_log(DebugLevel::Debug, "storeAttributeList: {index: %d, struct_size: %d, nb_values: %d}\n", attribute_list->index, attribute_list->struct_size, attribute_list->nb_values);
}

void ThreadWriter::storeToken(std::vector<Token>& tokenSeq, Token t) {
  pallas_log(DebugLevel::Debug, "storeToken: (%c%d) in seq at callstack[%d] (size: %zu)\n", PALLAS_TOKEN_TYPE_C(t), t.id, cur_depth, tokenSeq.size() + 1);
  tokenSeq.push_back(t);
  pallas_log(DebugLevel::Debug, "storeToken: %s\n", thread->getTokenArrayString(tokenSeq.data(), 0, tokenSeq.size()).c_str());
  findLoop();
}

/**
 * Adds an iteration of the given sequence to the loop.
 */
void Loop::addIteration() {
  pallas_log(DebugLevel::Debug, "addIteration: + 1 to L%d (to %u)\n", self_id.id, nb_iterations + 1);
  nb_iterations++;
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
    Sequence* loop_seq = thread->getSequence(loop->repeated_token);

    // Compute the durations
    const pallas_duration_t duration_first_iteration = thread->getLastSequenceDuration(loop_seq, 1);
    const pallas_duration_t duration_second_iteration = thread->getLastSequenceDuration(loop_seq, 0);
    // Reminded: when offset is 0, it doesn't take into account the duration of the last event
    // Because it's not a duration yet, it's a timestamp

    loop_seq->durations->add(duration_first_iteration);
    addDurationToComplete(loop_seq->durations->add(duration_second_iteration));

    // And add that timestamp to the vectors
    loop_seq->timestamps->add(last_timestamp - duration_first_iteration - duration_second_iteration);
    loop_seq->timestamps->add(last_timestamp - duration_second_iteration);
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
  // First, we check the case where there's a loop before a sequence containing it
  const size_t curIndex = curTokenSeq.size() - 1;
  if (curIndex >= 1) {
    if (curTokenSeq[curIndex - 1].type == TypeLoop) {
      auto l = thread->getLoop(curTokenSeq[curIndex - 1]);
      pallas_assert(l);
      if (l->repeated_token == curTokenSeq[curIndex]) {
        pallas_log(DebugLevel::Debug, "findLoopBasic: Last token was the sequence from L%d: S%d\n", l->self_id.id, l->repeated_token.id);
        l->addIteration();
        curTokenSeq.resize(curIndex);
        pallas_log(DebugLevel::Debug, "findLoopBasic: %s\n", thread->getTokenArrayString(curTokenSeq.data(), 0, curTokenSeq.size()).c_str());
        return;
      }
      // We need to find a way to tell if the Loop at [curIndex - 1] is finalized
      auto* s = thread->getSequence(l->repeated_token);
      if (s->getTokenCountWriting(thread)[curTokenSeq[curIndex]] == 0) {
        for (int i = 0; i < l->self_id.id; i++) {
          if (thread->loops[i].repeated_token == l->repeated_token && thread->loops[i].nb_iterations == l->nb_iterations) {
            curTokenSeq[curIndex - 1] = l->repeated_token;
            if (l->self_id.id == thread->nb_loops - 1) {
              thread->nb_loops--;
            }
            break;
          }
        }
      }
    }
  }
  for (int loopLength = 1; loopLength < maxLoopLength && loopLength <= curIndex; loopLength++) {
    // search for a loop of loopLength tokens
    const size_t startS1 = curIndex + 1 - loopLength;
    if (curIndex + 1 >= 2 * loopLength) {
      const size_t startS2 = curIndex + 1 - 2 * loopLength;
      /* search for a loop of loopLength tokens */
      if (_pallas_arrays_equal(&curTokenSeq[startS1], loopLength, &curTokenSeq[startS2], loopLength)) {
        pallas_log(DebugLevel::Debug, "findLoopBasic: Found a loop of len %d\n", loopLength);
        replaceTokensInLoop(loopLength, startS1, startS2);
        pallas_log(DebugLevel::Debug, "findLoopBasic: %s\n", thread->getTokenArrayString(curTokenSeq.data(), 0, curTokenSeq.size()).c_str());
        return;
      }
    }
  }
}

void ThreadWriter::findSequence(size_t n) {
  auto& curTokenSeq = getCurrentTokenSequence();
  size_t currentIndex = curTokenSeq.size() - 1;
  if (n >= currentIndex)
    n = currentIndex;

  unsigned found_sequence_id = 0;
  for (int array_len = 1; array_len <= n; array_len++) {
    auto token_array = &curTokenSeq[currentIndex - array_len + 1];
    uint32_t hash = hash32(reinterpret_cast<uint8_t*>(token_array), array_len * sizeof(Token), SEED);
    auto sequencesWithSameHash = thread->hashToSequence[hash];
    if (!sequencesWithSameHash.empty()) {
      for (const auto sid : sequencesWithSameHash) {
        if (_pallas_arrays_equal(token_array, array_len, thread->sequences[sid]->tokens.data(), thread->sequences[sid]->size())) {
          found_sequence_id = sid;
          break;
        }
      }
    }
    if (found_sequence_id) {
      pallas_log(DebugLevel::Debug, "Found S%d in %d last tokens\n", found_sequence_id, array_len);
      curTokenSeq.resize(curTokenSeq.size() - array_len);
      auto seqTok = Token(TypeSequence, found_sequence_id);
      storeToken(curTokenSeq, seqTok);
      pallas_log(DebugLevel::Debug, "findSequence: %s\n", thread->getTokenArrayString(curTokenSeq.data(), 0, curTokenSeq.size()).c_str());

      auto s = thread->getSequence(seqTok);

      const pallas_duration_t sequence_duration = thread->getLastSequenceDuration(s, 0);
      addDurationToComplete(s->durations->add(sequence_duration));
      s->timestamps->add(last_timestamp-sequence_duration);
      return;
    }
  }
}

void ThreadWriter::findLoop() {
  if (parameterHandler->getLoopFindingAlgorithm() == LoopFindingAlgorithm::None) {
    return;
  }

  size_t maxLoopLength = (parameterHandler->getLoopFindingAlgorithm() == LoopFindingAlgorithm::BasicTruncated) ? parameterHandler->getMaxLoopLength() : SIZE_MAX;
  // First we check if the last tokens are of a Sequence we already know
  findSequence(maxLoopLength);

  // Then we check for loops we haven't found yet
  switch (parameterHandler->getLoopFindingAlgorithm()) {
  case LoopFindingAlgorithm::None:
    return;
  case LoopFindingAlgorithm::Basic:
  case LoopFindingAlgorithm::BasicTruncated: {
    findLoopBasic(maxLoopLength);
  } break;
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

static Record getMatchingRecord(Record r) {
  switch (r) {
  case PALLAS_EVENT_ENTER:
    return PALLAS_EVENT_LEAVE;
  case PALLAS_EVENT_MPI_COLLECTIVE_BEGIN:
    return PALLAS_EVENT_MPI_COLLECTIVE_END;
  case PALLAS_EVENT_OMP_FORK:
    return PALLAS_EVENT_OMP_JOIN;
  case PALLAS_EVENT_THREAD_FORK:
    return PALLAS_EVENT_THREAD_JOIN;
  case PALLAS_EVENT_THREAD_TEAM_BEGIN:
    return PALLAS_EVENT_THREAD_TEAM_END;
  case PALLAS_EVENT_THREAD_BEGIN:
    return PALLAS_EVENT_THREAD_END;
  case PALLAS_EVENT_PROGRAM_BEGIN:
    return PALLAS_EVENT_PROGRAM_END;
  case PALLAS_EVENT_LEAVE:
    return PALLAS_EVENT_ENTER;
  case PALLAS_EVENT_MPI_COLLECTIVE_END:
    return PALLAS_EVENT_MPI_COLLECTIVE_BEGIN;
  case PALLAS_EVENT_OMP_JOIN:
    return PALLAS_EVENT_OMP_FORK;
  case PALLAS_EVENT_THREAD_JOIN:
    return PALLAS_EVENT_THREAD_FORK;
  case PALLAS_EVENT_THREAD_TEAM_END:
    return PALLAS_EVENT_THREAD_TEAM_BEGIN;
  case PALLAS_EVENT_THREAD_END:
    return PALLAS_EVENT_THREAD_BEGIN;
  case PALLAS_EVENT_PROGRAM_END:
    return PALLAS_EVENT_PROGRAM_BEGIN;
  default:
    return PALLAS_EVENT_MAX_ID;
  }
}

void ThreadWriter::recordExitFunction() {
  auto& curTokenSeq = getCurrentTokenSequence();

#ifdef DEBUG
  // check that the sequence is not weird

  Token first_token = curTokenSeq.front();
  Token last_token = curTokenSeq.back();

  if (first_token.type == TypeEvent) {
    Event* first_event = thread->getEvent(first_token);
    Event* last_event = thread->getEvent(last_token);

    enum Record expected_record = getMatchingRecord(first_event->record);
    if (expected_record == PALLAS_EVENT_MAX_ID) {
      pallas_warn("Unexpected start_event record: %s\n", thread->getEventString(first_event).c_str());
      pallas_abort();
    }

    if (last_event->record != expected_record) {
      pallas_warn("Unexpected close event:\n\tStart_sequence event: \t%s as E%d\n\tEnd_sequence event: \t%s as E%d\n",
        thread->getEventString(first_event).c_str(), first_token.id, thread->getEventString(last_event).c_str(), last_token.id);
      if (cur_depth > 1) {
        auto& underSequence = sequence_stack[cur_depth - 1];
        enum Record expected_start_record = getMatchingRecord(last_event->record);
        if (expected_start_record == PALLAS_EVENT_MAX_ID) {
          pallas_warn("Unexpected last_event record:\n\t%s\n", thread->getEventString(last_event).c_str());
          pallas_abort();
        }
        pallas_warn("Currently recorded last event is wrong by one layer, adding the correct Leave Event.\n");
        curTokenSeq.resize(curTokenSeq.size() - 1);
        Event e;
        e.event_size = offsetof(Event, event_data);
        e.record = expected_record;
        memcpy(e.event_data, first_event->event_data, first_event->event_size);
        e.event_size = first_event->event_size;
        TokenId e_id = thread->getEventId(&e);
        pallas_warn("\tInserting %s as E%d at end of curSequence\n", thread->getEventString(&e).c_str(), e_id);
        storeEvent(PALLAS_BLOCK_END, e_id, getTimestamp(), nullptr);
        pallas_warn("\tInserting %s as E%d at end of layer under curSequence\n", thread->getEventString(last_event).c_str(), last_token.id);
        underSequence.push_back(last_token);
        recordExitFunction();
        return;
      }
    }
  }

  if (curTokenSeq != sequence_stack[cur_depth]) {
    pallas_error("cur_seq=%p, but og_seq[%d] = %p\n", &curTokenSeq, cur_depth, &sequence_stack[cur_depth]);
  }
#endif

  const Token seq_id = thread->getSequenceIdFromArray(curTokenSeq.data(), curTokenSeq.size());
  auto* seq = thread->sequences[seq_id.id];

  const pallas_timestamp_t sequence_duration = last_timestamp - sequence_start_timestamp[cur_depth];
  addDurationToComplete(seq->durations->add(sequence_duration));
  seq->timestamps->add(sequence_start_timestamp[cur_depth]);

  pallas_log(DebugLevel::Debug, "Exiting function, closing %s, start=%lu\n", thread->getTokenString(seq_id).c_str(), sequence_start_timestamp[cur_depth]);

  cur_depth--;
  /* upper_seq is the sequence that called cur_seq */
  auto& upperTokenSeq = getCurrentTokenSequence();

  storeToken(upperTokenSeq, seq_id);
  curTokenSeq.clear();
  // We need to reset the token vector
  // Calling vector::clear() might be a better way to do that,
  // but depending on the implementation it might force a bunch of realloc, which isn't great.
}  // namespace pallas

size_t ThreadWriter::storeEvent(enum EventType event_type, TokenId event_id, pallas_timestamp_t ts, AttributeList* attribute_list) {
  ts = timestamp(ts);
  if (event_type == PALLAS_BLOCK_START) {
    recordEnterFunction();
    sequence_start_timestamp[cur_depth] = ts;
  }

  Token token = Token(TypeEvent, event_id);
  auto& curTokenSeq = getCurrentTokenSequence();

  EventSummary* es = &thread->events[event_id];
  size_t occurrence_index = es->nb_occurences++;
  pallas_log(DebugLevel::Debug, "storeEvent: %s @ %lu\n", thread->getTokenString(token).c_str(), ts);
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
  auto& mainSequence = thread->sequences[0];
  mainSequence->tokens = sequence_stack[0];
  pallas_log(DebugLevel::Debug, "Last sequence token: (%d.%d)\n", mainSequence->tokens.back().type, mainSequence->tokens.back().id);
  if (last_duration)
    *last_duration = 0;
  completeDurations(0);
  pallas_timestamp_t duration = last_timestamp - thread->first_timestamp;
  mainSequence->durations->add(duration);
  mainSequence->timestamps->add(thread->first_timestamp);
  thread->finalizeThread();
}
Archive::Archive(GlobalArchive& global_archive, LocationGroupId archive_id):
Archive(global_archive.dir_name, global_archive.trace_name, archive_id)
 {
  this->global_archive = &global_archive;
}


Archive::Archive(const char* dirname, const char* given_trace_name, LocationGroupId archive_id) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;
  pallas_debug_level_init();
  if (!parameterHandler) {
    parameterHandler = new ParameterHandler();
  }

  dir_name = strdup(dirname);
  trace_name = strdup(given_trace_name);
  id = archive_id;
  global_archive = nullptr;
  lock = {};
  pthread_mutex_init(&lock, nullptr);

  nb_allocated_threads = NB_THREADS_DEFAULT;
  nb_threads = 0;
  threads = new Thread*[nb_allocated_threads];
  pallas_recursion_shield--;
}


ThreadWriter::ThreadWriter(Archive& a, ThreadId thread_id) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;

  pallas_log(DebugLevel::Debug, "ThreadWriter(%ux)::open\n", thread_id);

  pthread_mutex_lock(&a.lock);
  while (a.nb_threads >= a.nb_allocated_threads) {
    doubleMemorySpaceConstructor(a.threads, a.nb_allocated_threads);
  }
  thread = new Thread;
  thread_rank = a.nb_threads;
  a.threads[a.nb_threads++] = thread;
  thread->archive = &a;
  thread->id = thread_id;

  thread->nb_allocated_events = NB_EVENT_DEFAULT;
  thread->events = new EventSummary[thread->nb_allocated_events]();
  thread->nb_events = 0;

  thread->nb_allocated_sequences = NB_SEQUENCE_DEFAULT;
  thread->sequences = new Sequence*[thread->nb_allocated_sequences]();
  thread->nb_sequences = 0;
  for (int i = 0; i < thread->nb_allocated_sequences; i++) {
    thread->sequences[i] = new Sequence();
    thread->sequences[i]->durations = new LinkedDurationVector();
    thread->sequences[i]->timestamps = new LinkedDurationVector();
  }

  thread->hashToSequence = std::unordered_map<uint32_t, std::vector<TokenId>>();
  thread->hashToEvent = std::unordered_map<uint32_t, std::vector<TokenId>>();

  thread->nb_allocated_loops = NB_LOOP_DEFAULT;
  thread->loops = new Loop[thread->nb_allocated_loops]();
  thread->nb_loops = 0;

  pthread_mutex_unlock(&a.lock);
  max_depth = CALLSTACK_DEPTH_DEFAULT;
  sequence_stack = new std::vector<Token>[max_depth];

  // We need to initialize the main Sequence (Sequence 0)
  auto& mainSequence = thread->sequences[0];
  mainSequence->id = 0;
  thread->nb_sequences = 1;

  last_timestamp = PALLAS_TIMESTAMP_INVALID;
  last_duration = nullptr;
  sequence_start_timestamp = new pallas_timestamp_t[max_depth];

  cur_depth = 0;

  pallas_recursion_shield--;
}


char* pallas_global_archive_fullpath(char* dir_name, char* trace_name) {
  int len = strlen(dir_name) + strlen(trace_name) + 2;
  char* fullpath = new char[len];
  snprintf(fullpath, len, "%s/%s", dir_name, trace_name);
  return fullpath;
}

GlobalArchive::GlobalArchive(const char* dirname, const char* given_trace_name) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;
  pallas_debug_level_init();
  if (!parameterHandler) {
    parameterHandler = new ParameterHandler();
  }
  dir_name = strdup(dirname);
  trace_name = strdup(given_trace_name);
  fullpath = pallas_global_archive_fullpath(dir_name, trace_name);
  nb_archives = 0;
  nb_allocated_archives = 0;
  lock = {};

  pthread_mutex_init(&lock, nullptr);

  pallas_recursion_shield--;
}

/**
 * Creates a new LocationGroup and adds it to that Archive.
 */
void GlobalArchive::defineLocationGroup(LocationGroupId lg_id, StringRef name, LocationGroupId parent) {
  pthread_mutex_lock(&lock);
  auto l = LocationGroup();
  l.id = lg_id;
  l.name = name;
  l.parent = parent;
  l.mainLoc = PALLAS_THREAD_ID_INVALID;
  location_groups.push_back(l);
  pthread_mutex_unlock(&lock);
}

/**
 * Creates a new Location and adds it to that Archive.
 */
void GlobalArchive::defineLocation(ThreadId l_id, StringRef name, LocationGroupId parent) {
  pthread_mutex_lock(&lock);
  Location l = Location();
  l.id = l_id;
  pallas_assert(l.id != PALLAS_THREAD_ID_INVALID);
  l.name = name;
  l.parent = parent;
  for (auto& locationGroup : location_groups) {
    if (locationGroup.id == parent && locationGroup.mainLoc == PALLAS_THREAD_ID_INVALID) {
      locationGroup.mainLoc = l_id;
      break;
    }
  }
  locations.push_back(l);
  pthread_mutex_unlock(&lock);
}

void Archive::close() {
  pallasStoreArchive(this);
}

void GlobalArchive ::close() {
  pallasStoreGlobalArchive(this);
}

TokenId Thread::getEventId(Event* e) {
  pallas_log(DebugLevel::Max, "getEventId: Searching for event {.event_type=%d}\n", e->record);

  uint32_t hash = hash32(reinterpret_cast<uint8_t*>(e), sizeof(Event), SEED);
  auto& eventWithSameHash = hashToEvent[hash];
  if (!eventWithSameHash.empty()) {
    if (eventWithSameHash.size() > 1) {
      pallas_log(DebugLevel::Debug, "Found more than one event with the same hash: %lu\n", eventWithSameHash.size());
    }
    for (const auto eid : eventWithSameHash) {
      if (memcmp(e, &events[eid].event, e->event_size) == 0) {
        pallas_log(DebugLevel::Debug, "getEventId: \t found with id=%u\n", eid);
        return eid;
      }
    }
  }

  if (nb_events >= nb_allocated_events) {
    pallas_log(DebugLevel::Debug, "Doubling mem space of events for thread trace %p\n", this);
    doubleMemorySpaceConstructor(events, nb_allocated_events);
  }

  TokenId index = nb_events++;
  pallas_log(DebugLevel::Max, "getEventId: \tNot found. Adding it with id=%d\n", index);
  auto* new_event = new (&events[index]) EventSummary(index, *e);
  hashToEvent[hash].push_back(index);

  return index;
}

pallas_duration_t Thread::getLastSequenceDuration(Sequence* sequence, size_t offset) const {
  pallas_duration_t sum = 0;
  auto tokenCount = sequence->getTokenCountWriting(this);
  if (offset == 0) {
    for (const auto& [token, count] : tokenCount) {
      if (token.type == TokenType::TypeEvent) {
        auto* event = getEventSummary(token);
        DOFOR(i, count) {
          sum += event->durations->at(event->durations->size - i - 1);
        }
      }
    }
    // We need to remove the duration of the last token, because it hasn't been calculated yet
    auto token = sequence->tokens.back();
    while (token.type != TokenType::TypeEvent) {
      if (token.type == TokenType::TypeSequence) {
        auto* seq = getSequence(token);
        token = seq->tokens.back();
        continue;
      } else {
        auto* loop = getLoop(token);
        auto* loop_seq = getSequence(loop->repeated_token);
        token = loop_seq->tokens.back();
        continue;
      }
    }
    auto* event = getEventSummary(token);
    sum -= event->durations->back();
  } else {
    auto secondTokenCount = sequence->getTokenCountWriting(this, &tokenCount);
    for (const auto& [token, count] : secondTokenCount) {
      if (token.type == TokenType::TypeEvent) {
        auto* event = getEventSummary(token);
        DOFOR(i, count) {
          sum += event->durations->at(event->durations->size - i - tokenCount[token] - 1);
        }
      }
    }
  }
  return sum;
}
}  // namespace pallas

/* C Callbacks */
pallas::ThreadWriter* pallas_thread_writer_new(pallas::Archive* archive, pallas::ThreadId thread_id) {
  return new pallas::ThreadWriter(*archive, thread_id);
}

extern void pallas_write_global_archive_close(pallas::GlobalArchive* archive) {
  archive->close();
};

extern void pallas_write_thread_close(pallas::ThreadWriter* thread_writer) {
  thread_writer->threadClose();
};

extern void pallas_write_define_location_group(pallas::GlobalArchive* archive, pallas::LocationGroupId id, pallas::StringRef name, pallas::LocationGroupId parent) {
  archive->defineLocationGroup(id, name, parent);
};

extern void pallas_write_define_location(pallas::GlobalArchive* archive, pallas::ThreadId id, pallas::StringRef name, pallas::LocationGroupId parent) {
  archive->defineLocation(id, name, parent);
};

extern void pallas_write_archive_close(PALLAS(Archive) * archive) {
  archive->close();
};

extern void pallas_store_event(PALLAS(ThreadWriter) * thread_writer,
                               enum PALLAS(EventType) event_type,
                               PALLAS(TokenId) id,
                               pallas_timestamp_t ts,
                               PALLAS(AttributeList) * attribute_list) {
  thread_writer->storeEvent(event_type, id, ts, attribute_list);
};

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */
