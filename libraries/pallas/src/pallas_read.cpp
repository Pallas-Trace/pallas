/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include "pallas/pallas_read.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "pallas/pallas_archive.h"

namespace pallas {
ThreadReader::ThreadReader(const Archive* archive, ThreadId threadId, int options) {
  // Setup the basic
  this->archive = archive;
  this->options = options;
  pallas_assert(threadId != PALLAS_THREAD_ID_INVALID);
  thread_trace = archive->getThread(threadId);
  pallas_assert(thread_trace != nullptr);

  if (debugLevel >= DebugLevel::Verbose) {
    pallas_log(DebugLevel::Verbose, "init callstack for thread %d\n", threadId);
    pallas_log(DebugLevel::Verbose, "The trace contains:\n");
    thread_trace->printSequence(Token(TypeSequence, 0));
  }

  // And initialize the callstack
  // ie set the cursor on the first event
  referential_timestamp = 0;
  current_frame = 0;
  callstack_index[0] = 0;
  callstack_loop_iteration[0] = 0;
  callstack_sequence[0].type = TypeSequence;
  callstack_sequence[0].id = 0;
}

const Token& ThreadReader::getFrameInCallstack(int frame_number) const {
  if (frame_number < 0 || frame_number >= MAX_CALLSTACK_DEPTH) {
    pallas_error("Frame number is too high or negative: %d\n", frame_number);
  }
  return callstack_sequence[frame_number];
}

const Token& ThreadReader::getTokenInCallstack(int frame_number) const {
  if (frame_number < 0 || frame_number >= MAX_CALLSTACK_DEPTH) {
    pallas_error("Frame number is too high or negative: %d\n", frame_number);
  }
  auto sequence = getFrameInCallstack(frame_number);
  pallas_assert(sequence.isIterable());
  return thread_trace->getToken(sequence, callstack_index[frame_number]);
}
const Token& ThreadReader::getCurToken() const {
  return getTokenInCallstack(current_frame);
}
void ThreadReader::printCurToken() const {
  thread_trace->printToken(getCurToken());
}
const Token& ThreadReader::getCurSequence() const {
  return getFrameInCallstack(current_frame);
}
void ThreadReader::printCurSequence() const {
  thread_trace->printSequence(getCurSequence());
}

void ThreadReader::printCallstack() const {
  printf("# Callstack (depth: %d) ------------\n", current_frame + 1);
  for (int i = 0; i < current_frame + 1; i++) {
    auto current_sequence_id = getFrameInCallstack(i);
    auto current_token = getTokenInCallstack(i);

    printf("%.*s[%d] ", i * 2, "                       ", i);
    thread_trace->printToken(current_sequence_id);

    if (current_sequence_id.type == TypeLoop) {
      auto* loop = thread_trace->getLoop(current_sequence_id);
      printf(" iter %d/%d", callstack_loop_iteration[i],
             loop->nb_iterations[tokenCount.get_value(current_sequence_id)]);
      pallas_assert(callstack_loop_iteration[i] < MAX_CALLSTACK_DEPTH);
    } else if (current_sequence_id.type == TypeSequence) {
      auto* sequence = thread_trace->getSequence(current_sequence_id);
      printf(" pos %d/%lu", callstack_index[i], sequence->size());
      pallas_assert(callstack_index[i] < MAX_CALLSTACK_DEPTH);
    }

    printf("\t-> ");
    pallas_print_token(thread_trace, current_token);
    printf("\n");
  }
}
EventSummary* ThreadReader::getEventSummary(Token event) const {
  pallas_assert(event.type == TypeEvent);
  if (event.id < thread_trace->nb_events) {
    return &thread_trace->events[event.id];
  }
  pallas_error("Given event (%d) was invalid\n", event.id);
}
pallas_timestamp_t ThreadReader::getEventTimestamp(Token event, int occurence_id) const {
  pallas_assert(event.type == TypeEvent);
  auto summary = getEventSummary(event);
  if (0 <= occurence_id && occurence_id < summary->nb_occurences) {
    return summary->durations->at(occurence_id);
  }
  pallas_error("Given occurence_id (%d) was invalid for event %d\n", occurence_id, event.id);
}
bool ThreadReader::isEndOfSequence(int current_index, Token sequence_id) const {
  if (sequence_id.type == TypeSequence) {
    auto* sequence = thread_trace->getSequence(sequence_id);
    return current_index >= sequence->size();
    // We are in a sequence and index is beyond the end of the sequence
  }
  pallas_error("The given sequence_id was the wrong type: %d\n", sequence_id.type);
}
bool ThreadReader::isEndOfLoop(int current_index, Token loop_id) const {
  if (loop_id.type == TypeLoop) {
    auto* loop = thread_trace->getLoop(loop_id);
    return current_index >= loop->nb_iterations[tokenCount.get_value(loop_id) - 1];
    // We are in a loop and index is beyond the number of iterations
  }
  pallas_error("The given loop_id was the wrong type: %d\n", loop_id.type);
}

pallas_duration_t ThreadReader::getLoopDuration(Token loop_id) const {
  pallas_assert(loop_id.type == TypeLoop);
  pallas_duration_t sum = 0;
  const auto* loop = thread_trace->getLoop(loop_id);
  const auto* sequence = thread_trace->getSequence(loop->repeated_token);

  const Token sequence_id = loop->repeated_token;

  const size_t loopIndex = tokenCount.get_value(loop_id);
  const size_t offset = tokenCount.get_value(sequence_id);
  const size_t nIterations = loop->nb_iterations.at(loopIndex);
  DOFOR(i, nIterations) {
    sum += sequence->durations->at(offset + i);
  }
  return sum;
}

EventOccurence ThreadReader::getEventOccurence(Token event_id, size_t occurence_id) const {
  auto eventOccurence = EventOccurence();
  auto* es = getEventSummary(event_id);
  eventOccurence.event = thread_trace->getEvent(event_id);

  if ((options & ThreadReaderOptions::NoTimestamps) == 0) {
    eventOccurence.timestamp = referential_timestamp;
    eventOccurence.duration = es->durations->at(occurence_id);
  }
  eventOccurence.attributes = getEventAttributeList(event_id, occurence_id);
  return eventOccurence;
}

SequenceOccurence ThreadReader::getSequenceOccurence(Token sequence_id,
                                                     size_t occurence_id,
                                                     bool saveReaderState) const {
  auto sequenceOccurence = SequenceOccurence();
  sequenceOccurence.sequence = thread_trace->getSequence(sequence_id);

  if ((options & ThreadReaderOptions::NoTimestamps) == 0) {
    sequenceOccurence.timestamp = referential_timestamp;
    sequenceOccurence.duration = sequenceOccurence.sequence->durations->at(occurence_id);
  }
  sequenceOccurence.full_sequence = nullptr;
  if (saveReaderState) {
    sequenceOccurence.savestate = new Savestate(this);
  } else {
    sequenceOccurence.savestate = nullptr;
  }

  //  auto localTokenCount = sequenceOccurence.sequence->getTokenCount(thread_trace, &this->tokenCount);
  return sequenceOccurence;
};

LoopOccurence ThreadReader::getLoopOccurence(Token loop_id, int occurence_id) const {
  auto loopOccurence = LoopOccurence();
  loopOccurence.loop = thread_trace->getLoop(loop_id);
  loopOccurence.nb_iterations = loopOccurence.loop->nb_iterations[occurence_id];
  loopOccurence.full_loop = nullptr;
  if ((options & ThreadReaderOptions::NoTimestamps) == 0) {
    loopOccurence.timestamp = referential_timestamp;
    loopOccurence.duration = getLoopDuration(loop_id);
  }
  return loopOccurence;
}

Occurence* ThreadReader::getOccurence(pallas::Token id, int occurence_id) const {
  auto occurence = new Occurence();
  switch (id.type) {
  case TypeInvalid: {
    pallas_error("Wrong token was given");
  }
  case TypeEvent:
    occurence->event_occurence = getEventOccurence(id, occurence_id);
    break;
  case TypeSequence:
    occurence->sequence_occurence = getSequenceOccurence(id, occurence_id, false);
    break;
  case TypeLoop:
    occurence->loop_occurence = getLoopOccurence(id, occurence_id);
    break;
  }
  return occurence;
}

AttributeList* ThreadReader::getEventAttributeList(Token event_id, int occurence_id) const {
  auto* summary = getEventSummary(event_id);
  if (summary->attribute_buffer == nullptr)
    return nullptr;

  //segfault in Debug mode here
  //pallas_assert(occurence_id < summary->nb_occurences);

  if (summary->attribute_pos < summary->attribute_buffer_size) {
    auto* l = (AttributeList*)&summary->attribute_buffer[summary->attribute_pos];

    while (l->index < occurence_id) { /* move to the next attribute until we reach the needed index */
      summary->attribute_pos += l->struct_size;
      l = (AttributeList*)&summary->attribute_buffer[summary->attribute_pos];
    }
    if (l->index == occurence_id) {
      return l;
    }
    if (l->index > occurence_id) {
      pallas_error("Error fetching attribute %d. We went too far (cur position: %d) !\n", occurence_id, l->index);
    }
  }
  return nullptr;
};

//******************* EXPLORATION FUNCTIONS ********************

void ThreadReader::enterBlock(Token new_block) {
  pallas_assert(new_block.isIterable());
  if (debugLevel >= DebugLevel::Debug) {
    pallas_log(DebugLevel::Debug, "[%d] Enter Block ", current_frame);
    printCurToken();
    printf("\n");
  }

  current_frame++;
  callstack_index[current_frame] = 0;
  callstack_loop_iteration[current_frame] = 0;
  callstack_sequence[current_frame] = new_block;
}

void ThreadReader::leaveBlock() {
  if (debugLevel >= DebugLevel::Debug) {
    pallas_log(DebugLevel::Debug, "[%d] Leave ", current_frame);
    printCurSequence();
    printf("\n");
  }

  callstack_index[current_frame] = INT16_MAX;
  callstack_sequence[current_frame] = Token();
  callstack_loop_iteration[current_frame] = INT16_MAX;
  current_frame--;  // pop frame

  if (debugLevel >= DebugLevel::Debug && current_frame >= 0) {
    auto current_sequence = getCurSequence();
    pallas_assert(current_sequence.type == TypeLoop || current_sequence.type == TypeSequence);
  }
}

void ThreadReader::moveToNextToken() {
  // Check if we've reached the end of the trace
  if (current_frame < 0) {
    pallas_log(DebugLevel::Debug, "End of trace %d!\n", __LINE__);
    return;
  }

  int current_index = callstack_index[current_frame];
  Token current_sequence_id = callstack_sequence[current_frame];
  int current_loop_iteration = callstack_loop_iteration[current_frame];
  pallas_assert(current_sequence_id.isIterable());

  /* First update the current loop / sequence. */
  if (current_sequence_id.type == TypeSequence) {
    if (isEndOfSequence(current_index + 1, current_sequence_id)) {
      /* We've reached the end of a sequence. Leave the block and give the next event. */
      leaveBlock();
      moveToNextToken();
    } else {
      /* Move to the next event in the Sequence */
      callstack_index[current_frame]++;
    }
  } else {
    if (isEndOfLoop(current_loop_iteration + 1, current_sequence_id)) {
      /* We've reached the end of the loop. Leave the block and give the next event. */
      leaveBlock();
      moveToNextToken();
    } else {
      /* just move to the next iteration in the loop */
      callstack_loop_iteration[current_frame]++;
    }
  }
}

void ThreadReader::updateReadCurToken() {
  auto current_token = getCurToken();
  switch (current_token.type) {
  case TypeSequence: {
    tokenCount[current_token]++;
    enterBlock(current_token);
    break;
  }
  case TypeLoop: {
    tokenCount[current_token]++;
    enterBlock(current_token);
    break;
  }
  case TypeEvent: {
    // Update the timestamps
    auto summary = getEventSummary(current_token);
    if ((options & ThreadReaderOptions::NoTimestamps) == 0) {
      referential_timestamp += summary->durations->at(tokenCount[current_token]);
      //lire timestamp abs
    }
    tokenCount[current_token]++;
    break;
  }
  default:
    break;
  }
}

Token ThreadReader::getNextToken() {
  moveToNextToken();
  updateReadCurToken();
  return getCurToken();
}
void ThreadReader::loadSavestate(Savestate* savestate) {
  if ((options & ThreadReaderOptions::NoTimestamps) == 0)
    referential_timestamp = savestate->referential_timestamp;
  memcpy(callstack_sequence, savestate->callstack_sequence, sizeof(int) * MAX_CALLSTACK_DEPTH);
  memcpy(callstack_index, savestate->callstack_index, sizeof(int) * MAX_CALLSTACK_DEPTH);
  memcpy(callstack_loop_iteration, savestate->callstack_loop_iteration, sizeof(int) * MAX_CALLSTACK_DEPTH);
  current_frame = savestate->current_frame;
  tokenCount = savestate->tokenCount;
}

std::vector<TokenOccurence> ThreadReader::readCurrentLevel() {
  const Token curSeqToken = getCurSequence();
  const auto* curSeq = thread_trace->getSequence(curSeqToken);
  pallas_assert(curSeq->size() > 0);
  auto outputVector = std::vector<TokenOccurence>();
  outputVector.resize(curSeq->size());

  DOFOR(i, curSeq->size()) {
    const Token token = curSeq->tokens[i];
    outputVector[i].occurence = new Occurence();
    outputVector[i].token = &curSeq->tokens[i];
    /// Three steps for every token type
    /// 1 - Grab the information we want, ie call getTypeOccurence
    /// 2 - Write that information to the occurence in the vector (outputVector[i].occurence->type_occurence)
    /// 3 - Update the reader
    switch (token.type) {
    case TypeEvent: {
      auto& occurence = outputVector[i].occurence->event_occurence;
      occurence = getEventOccurence(token, tokenCount[token]);
      if ((options & ThreadReaderOptions::NoTimestamps) == 0) {
        referential_timestamp += occurence.duration;
      }
      break;
    }
    case TypeLoop: {
      auto& occurence = outputVector[i].occurence->loop_occurence;
      auto* loop = &thread_trace->loops[token.id];

      // Write it to the occurence
      occurence.loop = loop;
      occurence.nb_iterations = loop->nb_iterations.at(tokenCount[token]);
      if ((options & ThreadReaderOptions::NoTimestamps) == 0)
        occurence.timestamp = referential_timestamp;

      // Write the loop
      enterBlock(token);

      occurence.full_loop = new SequenceOccurence[occurence.nb_iterations];
      auto& sequenceTokenCount = thread_trace->getSequence(loop->repeated_token)->getTokenCount(thread_trace);
      occurence.duration = 0;
      DOFOR(j, occurence.nb_iterations) {
        occurence.full_loop[j] = getSequenceOccurence(loop->repeated_token, tokenCount[loop->repeated_token], true);
        if ((options & ThreadReaderOptions::NoTimestamps) == 0) {
          occurence.duration += occurence.full_loop[j].duration;
          referential_timestamp += occurence.full_loop[j].duration;
        }
        tokenCount[loop->repeated_token]++;
        tokenCount += sequenceTokenCount;
      }
      leaveBlock();
      break;
    }
    case TypeSequence: {
      // Get the info
      outputVector[i].occurence->sequence_occurence = getSequenceOccurence(token, tokenCount[token], true);
      if ((options & ThreadReaderOptions::NoTimestamps) == 0) {
        referential_timestamp += outputVector[i].occurence->sequence_occurence.duration;
      }
      tokenCount += thread_trace->getSequence(token)->getTokenCount(thread_trace);;
      break;
    }
    default:
      pallas_error("Invalid token type\n;");
    }
    tokenCount[token]++;
  }
  return outputVector;
}
ThreadReader::~ThreadReader() {
  bool hasStilThreads = false;
  DOFOR(i, archive->nb_threads) {
    hasStilThreads = hasStilThreads || archive->threads[i] != nullptr;
    if (archive->threads[i] == thread_trace) {
      archive->threads[i] = nullptr;
    }
  }
  delete thread_trace;
  if (!hasStilThreads)
    delete archive;
}

Savestate::Savestate(const ThreadReader* reader) {
  if ((reader->options & ThreadReaderOptions::NoTimestamps) == 0) {
    referential_timestamp = reader->referential_timestamp;
  }

  callstack_sequence = new Token[reader->current_frame];
  memcpy(callstack_sequence, reader->callstack_sequence, sizeof(Token) * reader->current_frame);

  callstack_index = new int[reader->current_frame];
  memcpy(callstack_index, reader->callstack_index, sizeof(int) * reader->current_frame);

  callstack_loop_iteration = new int[reader->current_frame];
  memcpy(callstack_loop_iteration, reader->callstack_loop_iteration, sizeof(int) * reader->current_frame);

  current_frame = reader->current_frame;

  tokenCount = reader->tokenCount;
#ifdef DEBUG
  savestate_memory += sizeof(Savestate);
  savestate_memory += reader->current_frame * sizeof(Token);
  savestate_memory += reader->current_frame * sizeof(int) * 2;
  savestate_memory += sizeof(tokenCount) + (tokenCount.size() * (sizeof(Token) + sizeof(size_t)));
  pallas_log(DebugLevel::Debug, "New savestate created, memory usage for savestates: %lu bytes\n", savestate_memory );
#endif
}
Savestate::~Savestate() {
#ifdef DEBUG
  savestate_memory -= sizeof(Savestate);
  savestate_memory -= current_frame * sizeof(Token);
  savestate_memory -= current_frame * sizeof(int) * 2;
  savestate_memory -= sizeof(tokenCount) + (tokenCount.size() * (sizeof(Token) + sizeof(size_t)));
#endif
  delete[] callstack_index;
  delete[] callstack_loop_iteration;
  delete[] callstack_sequence;
  pallas_log(DebugLevel::Debug, "Savestate deleted, memory usage for savestates: %lu bytes\n", savestate_memory );

}
TokenOccurence::~TokenOccurence() {
  if (token == nullptr || occurence == nullptr) {
    return;
  }
  if (token->type == TypeSequence) {
//    delete[] occurence->sequence_occurence.full_sequence;
    delete occurence->sequence_occurence.savestate;
  }
  if (token->type == TypeLoop) {
    auto& loopOccurence = occurence->loop_occurence;
    if (loopOccurence.full_loop) {
      for (int i = 0; i < loopOccurence.nb_iterations; i++) {
        delete loopOccurence.full_loop[i].savestate;
      }
      delete[] loopOccurence.full_loop;
    }
  }
  delete occurence;
}
} /* namespace pallas */

pallas::ThreadReader* pallas_new_thread_reader(const pallas::Archive* archive,
                                               pallas::ThreadId thread_id,
                                               int options) {
  return new pallas::ThreadReader(archive, thread_id, options);
}

void pallas_thread_reader_enter_block(pallas::ThreadReader* reader, pallas::Token new_block) {
  reader->enterBlock(new_block);
}

void pallas_thread_reader_leave_block(pallas::ThreadReader* reader) {
  reader->leaveBlock();
}

void pallas_thread_reader_move_to_next_token(pallas::ThreadReader* reader) {
  return reader->moveToNextToken();
}

void pallas_thread_reader_update_reader_cur_token(pallas::ThreadReader* reader) {
  return reader->updateReadCurToken();
}

pallas::Token pallas_thread_reader_get_next_token(pallas::ThreadReader* reader) {
  return reader->getNextToken();
}

pallas::Token pallas_read_thread_cur_token(const pallas::ThreadReader* reader) {
  return reader->getCurToken();
}

pallas::Occurence* pallas_thread_reader_get_occurence(const pallas::ThreadReader* reader,
                                                      pallas::Token id,
                                                      int occurence_id) {
  return reader->getOccurence(id, occurence_id);
}

C_CXX(_Thread_local, thread_local) size_t savestate_memory = 0;

struct pallas::Savestate* create_savestate(pallas::ThreadReader* reader) {
  return new pallas::Savestate(reader);
}

void load_savestate(pallas::ThreadReader* reader, pallas::Savestate* savestate) {
  reader->loadSavestate(savestate);
}

void skip_sequence(pallas::ThreadReader* reader, pallas::Token token) {
  reader->skipSequence(token);
}

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */
