/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include <cinttypes>
#include <set>
#include <sstream>

#include "pallas/pallas.h"
#include "pallas/pallas_archive.h"
#include "pallas/pallas_log.h"
__thread uint64_t thread_rank = 0;
unsigned int pallas_mpi_rank = 0;

namespace pallas {
void Thread::loadTimestamps() {
  DOFOR(i, nb_events) {
    size_t loaded_duration = events[i].durations->front();
  }
  DOFOR(i, nb_sequences) {
    size_t loaded_duration = sequences[i]->durations->front();
    size_t loaded_timestamps = sequences[i]->timestamps->front();
  }
}

Event* Thread::getEvent(Token token) const {
  return &getEventSummary(token)->event;
}

void EventSummary::cleanEventSummary() {
  delete durations;
  delete attribute_buffer;
  durations = nullptr;
  attribute_buffer = nullptr;
}

EventSummary::EventSummary(TokenId token_id, const Event& e) {
  id = token_id;
  nb_occurences = 0;
  attribute_buffer = nullptr;
  attribute_buffer_size = 0;
  attribute_pos = 0;
  durations = new LinkedDurationVector();
  event = e;
}

EventSummary* Thread::getEventSummary(Token token) const {
  if (token.type != TokenType::TypeEvent) {
    pallas_error("Trying to getEventSummary of (%c%d)\n", PALLAS_TOKEN_TYPE_C(token), token.id);
  }
  pallas_assert(token.id < this->nb_events);
  return &this->events[token.id];
}

Sequence* Thread::getSequence(Token token) const {
  if (token.type != TokenType::TypeSequence) {
    pallas_error("Trying to getSequence of (%c%d)\n", PALLAS_TOKEN_TYPE_C(token), token.id);
  }
  pallas_assert(token.id < this->nb_sequences);
  return this->sequences[token.id];
}

Loop* Thread::getLoop(Token token) const {
  if (token.type != TypeLoop) {
    pallas_error("Trying to getLoop of (%c%d)\n", PALLAS_TOKEN_TYPE_C(token), token.id);
  }
  pallas_assert(token.id < this->nb_loops);
  auto* l = &this->loops[token.id];
  pallas_assert(l->repeated_token.isValid());
  pallas_assert(l->self_id.isValid());
  return l;
}

Token& Thread::getToken(Token sequenceToken, int index) const {
  if (sequenceToken.type == TypeSequence) {
    auto* sequence = getSequence(sequenceToken);
    if (!sequence) {
      pallas_error("Invalid sequence ID: %d\n", sequenceToken.id);
    }
    if (index >= sequence->size()) {
      pallas_error("Invalid index (%d) in sequence %d\n", index, sequenceToken.id);
    }
    return sequence->tokens[index];
  } else if (sequenceToken.type == TypeLoop) {
    auto* loop = getLoop(sequenceToken);
    if (!loop) {
      pallas_error("Invalid loop ID: %d\n", sequenceToken.id);
    }
    return loop->repeated_token;
  }
  pallas_error("Invalid parameter to getToken\n");
}

std::string Thread::getTokenString(Token token) const {
  std::ostringstream tempString;
  switch (token.type) {
  case TypeInvalid:
    tempString << "U";
    break;
  case TypeEvent:
    tempString << "E";
    break;
  case TypeSequence:
    tempString << "S";
    break;
  case TypeLoop:
    tempString << "L";
    break;
  }
  tempString << token.id;
  if (token.type == TypeEvent) {
    Event* event = getEvent(token);
    tempString << ((event->record) == PALLAS_EVENT_ENTER ? "E" : (event->record) == PALLAS_EVENT_LEAVE ? "L" : "S");
  }
  return tempString.str();
}

pallas_duration_t Thread::getDuration() const {
  return sequences[0]->durations->at(0);
}
pallas_duration_t get_duration(PALLAS(Thread) * t) {
  return t->getDuration();
}

pallas_timestamp_t Thread::getFirstTimestamp() const {
  return first_timestamp;
}
pallas_timestamp_t get_first_timestamp(PALLAS(Thread) * t) {
  return t->getFirstTimestamp();
}

pallas_timestamp_t Thread::getLastTimestamp() const {
  return getFirstTimestamp() + getDuration();
}
pallas_timestamp_t get_last_timestamp(PALLAS(Thread) * t) {
  return t->getLastTimestamp();
}

size_t Thread::getEventCount() const {
  size_t ret = 0;
  for (unsigned i = 0; i < this->nb_events; i++) {
    ret += this->events[i].nb_occurences;
  }
  return ret;
}
size_t get_event_count(PALLAS(Thread) * t) {
  return t->getEventCount();
}

std::string Thread::getTokenArrayString(const Token* array, size_t start_index, size_t len) const {
  std::string out("[");
  for (int i = 0; i < len; i++) {
    out += getTokenString(array[start_index + i]);
    if (i != len - 1)
      out += ", ";
  }
  out += "]";
  return out;
};

void Thread::printTokenVector(const std::vector<Token>& vector) const {
  std::cout << getTokenArrayString(vector.data(), 0, vector.size()) << std::endl;
}

void Thread::printSequence(pallas::Token token) const {
  Sequence* sequence = getSequence(token);
  printf("#Sequence %d (%zu tokens)-------------\n", token.id, sequence->tokens.size());
  printTokenVector(sequence->tokens);
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

const char* Thread::getRegionStringFromEvent(pallas::Event* e) const {
  const Region* region = NULL;
  byte* cursor = nullptr;
  switch (e->record) {
  case PALLAS_EVENT_ENTER: {
    RegionRef region_ref;
    pop_data(e, &region_ref, sizeof(region_ref), cursor);
    region = archive->getRegion(region_ref);
    break;
  }
  case PALLAS_EVENT_LEAVE: {
    RegionRef region_ref;
    pop_data(e, &region_ref, sizeof(region_ref), cursor);
    region = archive->getRegion(region_ref);
    break;
  }
  default:
    region = NULL;
  }

  return region ? archive->getString(region->string_ref)->str : "INVALID";
}
std::string Thread::getEventString(Event* e) const {
  byte* cursor = nullptr;
  switch (e->record) {
  case PALLAS_EVENT_ENTER: {
    RegionRef region_ref;
    pop_data(e, &region_ref, sizeof(region_ref), cursor);
    if (archive->global_archive) {
      const Region* region = archive->getRegion(region_ref);
      const char* region_name = region ? archive->getString(region->string_ref)->str : "INVALID";
      return "Enter " + std::to_string(region_ref) + "(" + region_name + ")";
    } else {
      return "Enter" + std::to_string(region_ref);
    }
  }
  case PALLAS_EVENT_LEAVE: {
    RegionRef region_ref;
    pop_data(e, &region_ref, sizeof(region_ref), cursor);
    if (archive->global_archive) {
      const Region* region = archive->getRegion(region_ref);
      const char* region_name = region ? archive->getString(region->string_ref)->str : "INVALID";
      return "Leave " + std::to_string(region_ref) + "(" + region_name + ")";
    } else {
      return "Leave " + std::to_string(region_ref);
    }
  }
  case PALLAS_EVENT_THREAD_BEGIN:
    return "THREAD_BEGIN()";
  case PALLAS_EVENT_THREAD_END:
    return "THREAD_END()";
  case PALLAS_EVENT_THREAD_TEAM_BEGIN:
    return "THREAD_TEAM_BEGIN()";
  case PALLAS_EVENT_THREAD_TEAM_END:
    return "THREAD_TEAM_END()";
  case PALLAS_EVENT_THREAD_FORK: {
    uint32_t numberOfRequestedThreads;
    pop_data(e, &numberOfRequestedThreads, sizeof(numberOfRequestedThreads), cursor);
    return "THREAD_FORK(nThreads= " + std::to_string(numberOfRequestedThreads) + ")";
  }
  case PALLAS_EVENT_THREAD_JOIN:
    return "THREAD_JOIN";

  case PALLAS_EVENT_MPI_SEND: {
    uint32_t receiver;
    uint32_t communicator;
    uint32_t msgTag;
    uint64_t msgLength;

    pop_data(e, &receiver, sizeof(receiver), cursor);
    pop_data(e, &communicator, sizeof(communicator), cursor);
    pop_data(e, &msgTag, sizeof(msgTag), cursor);
    pop_data(e, &msgLength, sizeof(msgLength), cursor);
    return "MPI_SEND("
           "dest=" + std::to_string(receiver) +
           ", comm=" + std::to_string(communicator) +
           ", tag=" + std::to_string(msgTag) +
           ", len=" + std::to_string(msgLength) + ")";
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
    return "MPI_ISEND("
                "dest=" + std::to_string(receiver) +
                ", comm=" + std::to_string(communicator) +
                ", tag=" + std::to_string(msgTag) +
                ", len=" + std::to_string(msgLength) +
                ", req=" + std::to_string(requestID)+ ")";
  }
  case PALLAS_EVENT_MPI_ISEND_COMPLETE: {
    uint64_t requestID;
    pop_data(e, &requestID, sizeof(requestID), cursor);
    return "MPI_ISEND_COMPLETE(req=" + std::to_string(requestID) + ")";
  }
  case PALLAS_EVENT_MPI_IRECV_REQUEST: {
    uint64_t requestID;
    pop_data(e, &requestID, sizeof(requestID), cursor);
    return "MPI_IRECV_REQUEST(req=" + std::to_string(requestID) + ")";
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
    return "MPI_RECV("
               "src=" + std::to_string(sender) +
               ", comm=" + std::to_string(communicator) +
               ", tag=" + std::to_string(msgTag) +
               ", len=" + std::to_string(msgLength) + ")";
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
    return "MPI_IRECV("
           "src=" + std::to_string(sender) +
           ", comm=" + std::to_string(communicator) +
           ", tag=" + std::to_string(msgTag) +
           ", len=" + std::to_string(msgLength) +
           ", tag=" + std::to_string(msgTag) + ")";
  }
  case PALLAS_EVENT_MPI_COLLECTIVE_BEGIN: {
    return "MPI_COLLECTIVE_BEGIN()";
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

    return "MPI_COLLECTIVE_END(op=" + std::to_string(collectiveOp) +
      ", comm=" + std::to_string(communicator) +
      ", root=" + std::to_string(root) +
      ", sent=" + std::to_string(sizeSent) +
      ", recv=" + std::to_string(sizeReceived) + ")";
  }
  case PALLAS_EVENT_OMP_FORK: {
    uint32_t numberOfRequestedThreads;
    pop_data(e, &numberOfRequestedThreads, sizeof(numberOfRequestedThreads), cursor);
    return "OMP_FORK(nThreads=" + std::to_string(numberOfRequestedThreads) + ")";
  }
  case PALLAS_EVENT_OMP_JOIN:
    return "OMP_JOIN()";
  case PALLAS_EVENT_OMP_ACQUIRE_LOCK: {
    uint32_t lockID;
    uint32_t acquisitionOrder;
    pop_data(e, &lockID, sizeof(lockID), cursor);
    // pop_data(e, &acquisitionOrder, sizeof(acquisitionOrder), cursor);
    return "OMP_ACQUIRE_LOCK(lockID="+ std::to_string(lockID) + "";
  }
  case PALLAS_EVENT_THREAD_ACQUIRE_LOCK: {
    uint32_t lockID;
    uint32_t acquisitionOrder;
    pop_data(e, &lockID, sizeof(lockID), cursor);
    // pop_data(e, &acquisitionOrder, sizeof(acquisitionOrder), cursor);
    return "THREAD_ACQUIRE_LOCK(lockID="+ std::to_string(lockID) + "";
  }
  case PALLAS_EVENT_OMP_RELEASE_LOCK: {
    uint32_t lockID;
    uint32_t acquisitionOrder;
    pop_data(e, &lockID, sizeof(lockID), cursor);
    // pop_data(e, &acquisitionOrder, sizeof(acquisitionOrder), cursor);
    return "OMP_RELEASE_LOCK(lockID="+ std::to_string(lockID) + "";
  }
  case PALLAS_EVENT_THREAD_RELEASE_LOCK: {
    uint32_t lockID;
    uint32_t acquisitionOrder;
    pop_data(e, &lockID, sizeof(lockID), cursor);
    // pop_data(e, &acquisitionOrder, sizeof(acquisitionOrder), cursor);
    return "THREAD_RELEASE_LOCK(lockID="+ std::to_string(lockID) + "";
  }
  case PALLAS_EVENT_OMP_TASK_CREATE: {
    uint64_t taskID;
    pop_data(e, &taskID, sizeof(taskID), cursor);
    return "OMP_TASK_CREATE(taskID="+ std::to_string(taskID) + ")";
  }
  case PALLAS_EVENT_OMP_TASK_SWITCH: {
    uint64_t taskID;
    pop_data(e, &taskID, sizeof(taskID), cursor);
    return "OMP_TASK_SWITCH(taskID="+ std::to_string(taskID) + ")";
  }
  case PALLAS_EVENT_OMP_TASK_COMPLETE: {
    uint64_t taskID;
    pop_data(e, &taskID, sizeof(taskID), cursor);
    return "OMP_TASK_COMPLETE(taskID="+ std::to_string(taskID) + ")";
  }
  case PALLAS_EVENT_THREAD_TASK_CREATE: {
    return "THREAD_TASK_CREATE()";
  }
  case PALLAS_EVENT_THREAD_TASK_SWITCH: {
    return "THREAD_TASK_SWITCH()";
  }
  case PALLAS_EVENT_THREAD_TASK_COMPLETE: {
    return "THREAD_TASK_COMPLETE()";
  }
  case PALLAS_EVENT_GENERIC: {
    StringRef eventNameRef;
    pop_data(e, &eventNameRef, sizeof(eventNameRef), cursor);
    auto eventName = archive->getString(eventNameRef);
    return eventName->str;
  }
  default:
    return "{.record=" + std::to_string(e->record) + ", .size=" + std::to_string(e->event_size) + "}";
  }
}

Thread::Thread() {
  archive = nullptr;
  id = PALLAS_THREAD_ID_INVALID;

  events = nullptr;
  nb_allocated_events = 0;
  nb_events = 0;

  sequences = nullptr;
  nb_allocated_sequences = 0;
  nb_sequences = 0;

  loops = nullptr;
  nb_allocated_loops = 0;
  nb_loops = 0;

  first_timestamp = PALLAS_TIMESTAMP_INVALID;
}

Thread::~Thread() {
  for (size_t i = 0; i < nb_allocated_events; i++) {
    events[i].cleanEventSummary();
  }
  delete[] events;
  for (size_t i = 0; i < nb_allocated_sequences; i++) {
    delete sequences[i];
  }
  delete[] sequences;
  delete[] loops;
}

const char* Thread::getName() const {
  return archive->getString(archive->getLocation(id)->name)->str;
}

bool Sequence::isFunctionSequence(const struct Thread* thread) const {
  if (tokens.front().type == TypeEvent && tokens.back().type == TypeEvent) {
    auto frontToken = thread->getEvent(tokens.front());
    auto backToken = thread->getEvent(tokens.back());
    return frontToken->record == PALLAS_EVENT_ENTER && backToken->record == PALLAS_EVENT_LEAVE;
  }
  return false;
};

 Group::~Group() {
  delete[] this->members;
}
 String::~String() {
  free(this->str);
}


std::string Sequence::guessName(const pallas::Thread* thread) {
  if (this->size() < 4) {
    Token t_start = this->tokens[0];
    if (t_start.type == TypeEvent) {
      Event* event = thread->getEvent(t_start);
      if (event->record == PALLAS_EVENT_ENTER) {
        const char* event_name = thread->getRegionStringFromEvent(event);
        std::string prefix(event_name);

        if (this->size() == 3) {
          // that's probably an MPI call. To differentiate calls (eg
          // MPI_Send(dest=5) vs MPI_Send(dest=0)), we can add the
          // the second token to the name
          Token t_second = this->tokens[1];

          std::string res = prefix + "_" + thread->getTokenString(t_second);
          return res;
        }
        return prefix;
      }
    }
  }
  char buff[128];
  snprintf(buff, sizeof(buff), "Sequence_%d", this->id);

  return std::string(buff);
}

size_t Sequence::getEventCount(const struct Thread* thread) {
  // TODO This function doesn't really makes sense, since the number of event is dependant on iteration of the loops
  // inside of it.
  return 0;
  // TokenCountMap tokenCount = getTokenCount(thread);
  return tokenCount.getEventCount();
}

void _sequenceGetTokenCountReading(Sequence* seq, const Thread* thread, TokenCountMap& readerTokenCountMap, TokenCountMap& sequenceTokenCountMap, bool isReversedOrder);

TokenCountMap tempSeen;
void _loopGetTokenCountReading(const Loop* loop, const Thread* thread, TokenCountMap& readerTokenCountMap, TokenCountMap& sequenceTokenCountMap, bool isReversedOrder) {
  size_t loop_nb_iterations = loop->nb_iterations;
  auto* loop_sequence = thread->getSequence(loop->repeated_token);
  // This creates bug idk why ?????
  TokenCountMap temp = loop_sequence->getTokenCountReading(thread, readerTokenCountMap, isReversedOrder);
  temp *= loop_nb_iterations;
  readerTokenCountMap += temp;
  sequenceTokenCountMap += temp;
  readerTokenCountMap[loop->repeated_token] += loop_nb_iterations;
  sequenceTokenCountMap[loop->repeated_token] += loop_nb_iterations;
}

std::string Loop::guessName(const Thread* t) {
  Sequence* s = t->getSequence(this->repeated_token);
  return s->guessName(t);
}
void _sequenceGetTokenCountReading(Sequence* seq, const Thread* thread, TokenCountMap& readerTokenCountMap, TokenCountMap& sequenceTokenCountMap, bool isReversedOrder) {
  for (auto& token : seq->tokens) {
    if (token.type == TypeSequence) {
      auto* s = thread->getSequence(token);
      _sequenceGetTokenCountReading(s, thread, readerTokenCountMap, sequenceTokenCountMap, isReversedOrder);
    }
    if (token.type == TypeLoop) {
      auto* loop = thread->getLoop(token);
      _loopGetTokenCountReading(loop, thread, readerTokenCountMap, sequenceTokenCountMap, isReversedOrder);
    }
    readerTokenCountMap[token]++;
    sequenceTokenCountMap[token]++;
  }
}

TokenCountMap Sequence::getTokenCountReading(const Thread* thread, const TokenCountMap& threadReaderTokenCountMap, bool isReversedOrder) {
  if (tokenCount.empty()) {
    auto tokenCountMapCopy = TokenCountMap(threadReaderTokenCountMap);
    auto tempTokenCount = TokenCountMap();
    _sequenceGetTokenCountReading(this, thread, tokenCountMapCopy, tempTokenCount, isReversedOrder);
    tokenCount = tempTokenCount;
  }
  return tokenCount;
}

static void _loopGetTokenCountWriting(const Loop* loop, const Thread* thread, TokenCountMap& tokenCount) {
  size_t loop_nb_iterations = loop->nb_iterations;
  auto* loop_sequence = thread->getSequence(loop->repeated_token);
  auto temp = loop_sequence->getTokenCountWriting(thread);
  DOFOR(i, loop->nb_iterations) {
    tokenCount += temp;
  }
  if ( tokenCount.find(loop->repeated_token) == tokenCount.end() ) {
    tokenCount[loop->repeated_token] = 0;
  }
  tokenCount[loop->repeated_token] += loop_nb_iterations;
}

TokenCountMap Sequence::getTokenCountWriting(const Thread* thread) {
   if (tokenCount.empty()) {
    for (auto& token : tokens) {
      if (tokenCount.find(token) == tokenCount.end()) {
        tokenCount[token] = 0;
      }
      tokenCount[token]++;
      if (token.type == TypeSequence) {
        auto* s = thread->getSequence(token);
        tokenCount += s->getTokenCountWriting(thread);
      }
      if (token.type == TypeLoop) {
        const auto* loop = thread->getLoop(token);
        _loopGetTokenCountWriting(loop, thread, tokenCount);
      }
    }
  }
  return tokenCount;
}
}  // namespace pallas

void* pallas_realloc(void* buffer, int cur_size, int new_size, size_t datatype_size) {
  void* new_buffer = (void*)realloc(buffer, new_size * datatype_size);
  if (new_buffer == NULL) {
    new_buffer = (void*)calloc(new_size, datatype_size);
    if (new_buffer == NULL) {
      pallas_error("Failed to allocate memory using realloc AND malloc\n");
    }
    memmove(new_buffer, buffer, cur_size * datatype_size);
    free(buffer);
  } else {
    /* realloc changed the size of the buffer, leaving some bytes */
    /* uninitialized. Let's fill the rest of the buffer with zeros to*/
    /* prevent problems. */

    if (new_size > cur_size) {
      uintptr_t old_end_addr = (uintptr_t)(new_buffer) + (cur_size * datatype_size);
      uintptr_t rest_size = (new_size - cur_size) * datatype_size;
      memset((void*)old_end_addr, 0, rest_size);
    }
  }
  return new_buffer;
}

/* C bindings now */

pallas::Thread* pallas_thread_new() {
  return new pallas::Thread();
};

const char* pallas_thread_get_name(pallas::Thread* thread) {
  return thread->getName();
}

void pallas_print_sequence(pallas::Thread* thread, pallas::Token seq_id) {
  thread->printSequence(seq_id);
}

pallas::Loop* pallas_get_loop(pallas::Thread* thread, pallas::Token id) {
  return thread->getLoop(id);
}
pallas::Sequence* pallas_get_sequence(pallas::Thread* thread, pallas::Token id) {
  return thread->getSequence(id);
}
pallas::Event* pallas_get_event(pallas::Thread* thread, pallas::Token id) {
  return thread->getEvent(id);
}
pallas::Token pallas_get_token(pallas::Thread* thread, pallas::Token sequence, int index) {
  return thread->getToken(sequence, index);
}

size_t pallas_sequence_get_size(pallas::Sequence* sequence) {
  return sequence->size();
}
pallas::Token pallas_sequence_get_token(pallas::Sequence* sequence, int index) {
  return sequence->tokens[index];
}

/* -*-
  mode: cpp;
  c-file-style: "k&r";
  c-basic-offset 2;
  tab-width 2 ;
  indent-tabs-mode nil
  -*- */
