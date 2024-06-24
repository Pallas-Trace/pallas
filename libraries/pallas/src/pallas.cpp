/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include "pallas/pallas.h"
#include "pallas/pallas_archive.h"

namespace pallas {
void Thread::loadTimestamps() {
  DOFOR(i, nb_events) {
    size_t loaded_duration = events[i].durations->front();
  }
  DOFOR(i, nb_sequences) {
    size_t loaded_duration = sequences[i]->durations->front();
  }
}

Event* Thread::getEvent(Token token) const {
  return &getEventSummary(token)->event;
}

EventSummary* Thread::getEventSummary(Token token) const {
  pallas_assert(token.type == TokenType::TypeEvent);
  pallas_assert(token.id < this->nb_events);
  return &this->events[token.id];
}

Sequence* Thread::getSequence(Token token) const {
  pallas_assert(token.type == TokenType::TypeSequence);
  pallas_assert(token.id < this->nb_sequences);
  return this->sequences[token.id];
}

Loop* Thread::getLoop(Token token) const {
  pallas_assert(token.type == TokenType::TypeLoop);
  pallas_assert(token.id < this->nb_loops);
  return &this->loops[token.id];
}

Token& Thread::getToken(Token sequenceToken, int index) const {
  if (sequenceToken.type == TypeSequence) {
    auto sequence = getSequence(sequenceToken);
    if (!sequence) {
      pallas_error("Invalid sequence ID: %d\n", sequenceToken.id);
    }
    if (index >= sequence->size()) {
      pallas_error("Invalid index (%d) in sequence %d\n", index, sequenceToken.id);
    }
    return sequence->tokens[index];
  } else if (sequenceToken.type == TypeLoop) {
    auto loop = getLoop(sequenceToken);
    if (!loop) {
      pallas_error("Invalid loop ID: %d\n", sequenceToken.id);
    }
    return loop->repeated_token;
  }
  pallas_error("Invalid parameter to getToken\n");
}

void Thread::printToken(Token token) const {
  switch (token.type) {
  case TypeEvent: {
#define ET2C(et) (((et) == PALLAS_EVENT_ENTER ? 'E' : (et) == PALLAS_EVENT_LEAVE ? 'L' : 'S'))
    Event* event = getEvent(token);
    printf("E%d_%c", token.id, ET2C(event->record));
    break;
  }
  case TypeSequence:
    printf("S%d", token.id);
    break;
  case TypeLoop:
    printf("L%d", token.id);
    break;
  default:
    printf("U%d_%d", token.type, token.id);
    break;
  }
}

void Thread::printTokenArray(const Token* array, size_t start_index, size_t len) const {
  printf("[");
  for (int i = 0; i < len; i++) {
    printToken(array[start_index + i]);
    printf(" ");
  }
  printf("]\n");
}

void Thread::printTokenVector(const std::vector<Token>& vector) const {
  printf("[");
  for (auto& token : vector) {
    printToken(token);
    printf(" ");
  }
  printf("]\n");
}

void Thread::printSequence(pallas::Token token) const {
  Sequence* sequence = getSequence(token);
  printf("#Sequence %d (%zu tokens)-------------\n", token.id, sequence->tokens.size());
  printTokenVector(sequence->tokens);
}

void fonctionBasique(){
  volatile char test = 1;
  while(test){
    printf(" ///////\n");
  }

}

Thread::Thread() {
  fonctionBasique();
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
  
}

void Thread::initThread(Archive* a, ThreadId thread_id) {
  
  archive = a;
  id = thread_id;

  nb_allocated_events = NB_EVENT_DEFAULT;
  events = new EventSummary[nb_allocated_events];
  nb_events = 0;

  nb_allocated_sequences = NB_SEQUENCE_DEFAULT;
  sequences = new Sequence*[nb_allocated_sequences];
  nb_sequences = 0;
  hashToSequence = std::unordered_map<uint32_t, std::vector<TokenId>>();

  nb_allocated_loops = NB_LOOP_DEFAULT;
  loops = new Loop[nb_allocated_loops];
  nb_loops = 0;

  pthread_mutex_lock(&archive->lock);
  while (archive->nb_threads >= archive->nb_allocated_threads) {
    DOUBLE_MEMORY_SPACE(archive->threads, archive->nb_allocated_threads, Thread*);
  }
  for (int i = 0; i < nb_allocated_sequences; i++) {
    sequences[i] = new Sequence();
  }
  archive->threads[archive->nb_threads++] = this;
  pthread_mutex_unlock(&archive->lock);
  
}

Archive::~Archive() {
  delete[] dir_name;
  delete[] trace_name;
  delete[] fullpath;
  delete[] threads;
  delete[] archive_list;
}

Thread::~Thread() {
  DOFOR(i, nb_events) {
    delete events[i].durations;
    delete[] events[i].attribute_buffer;
  }
  delete[] events;
  DOFOR(i, nb_sequences) {
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

const TokenCountMap& Sequence::getTokenCount(const Thread* thread, const TokenCountMap* alreadyReadTokens) {
  if (tokenCount.empty()) {
    // We need to count the tokens
    for (auto t = tokens.rbegin(); t != tokens.rend(); ++t) {
      tokenCount[*t]++;
      switch (t->type) {
      case TypeSequence: {
        auto* s = thread->getSequence(*t);
        tokenCount += s->getTokenCount(thread);
        break;
      }
      case TypeLoop: {
        auto* l = thread->getLoop(*t);
        auto loopTokenCount = thread->getSequence(l->repeated_token)->getTokenCount(thread);
        // Vrai à l'écriture, pas à la lecture
        if (alreadyReadTokens) {
          tokenCount += loopTokenCount * l->nb_iterations[alreadyReadTokens->get_value(*t) + tokenCount[*t]];
        } else {
          tokenCount += loopTokenCount * l->nb_iterations[l->nb_iterations.size() - tokenCount[*t]];
        }
        break;
      }
      default:
        break;
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

void pallas_print_token_array(pallas::Thread* thread, pallas::Token* token_array, int index_start, int index_stop) {
  thread->printTokenArray(token_array, index_start, index_stop);
}

void pallas_print_token(pallas::Thread* thread, pallas::Token token) {
  thread->printToken(token);
}

void pallas_print_event(pallas::Thread* thread, pallas::Event* e) {
  thread->printEvent(e);
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

size_t pallas_loop_count(pallas::Loop* loop) {
  return loop->nb_iterations.size();
};
size_t pallas_loop_get_count(PALLAS(Loop) * loop, size_t index) {
  return loop->nb_iterations[index];
};

/* -*-
  mode: cpp;
  c-file-style: "k&r";
  c-basic-offset 2;
  tab-width 2 ;
  indent-tabs-mode nil
  -*- */
