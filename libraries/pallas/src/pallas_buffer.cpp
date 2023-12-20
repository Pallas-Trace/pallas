/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include "pallas/pallas_buffer.h"
#include <cstring>
using namespace pallas;

#define NB_BUFFERS_DEFAULT 10

PallasBufferManager::PallasBufferManager(size_t n_elem, size_t elem_size) {
  this->n_elem = n_elem;
  this->elem = elem_size;

  // create a bunch of ready buffers
  allocBuffer(NB_BUFFERS_DEFAULT);
}

PallasBuffer* PallasBufferManager::allocBuffer(int nb_buffers = 1) {
  PallasBuffer* b = nullptr;
  for(int i=0; i<NB_BUFFERS_DEFAULT; i++) {
    b = new PallasBuffer(n_elem, elem_size);
    initializedBuffers.push_front(b);
  }
  return b;
}

PallasBuffer* PallasBufferManager::assign(FILE* output_file, size_t sequence_number) {
  PallasBuffer* b = nullptr;
  if(initializedBuffers.empty()) {
    b = allocBuffer();
  } else {
    b = initializedBuffers.pop_front(b);
  }

  pendingBuffers.push_front(b);
  b->assign(output_file, sequence_number);
  pallas_assert(buffer->status == Pending);
  return b;
}

void PallasBufferManager::setFull(PallasBuffer* buffer) {
  pallas_assert(buffer->status == Pending);
  // move the buffer from the Pending list to the Full list
  auto ret = pendingBuffers.remove(buffer);
  pallas_assert(ret == 1);

  buffer->setFull();

  pallas_assert(buffer->status == Full);
  fullBuffers.push_front(buffer); 
}


LinkedVector::LinkedVector() {
  first = new SubVector(defaultSize);
  last = first;
}

uint64_t* LinkedVector::add(uint64_t val) {
  if (this->last->size >= this->last->allocated) {
    pallas_log(DebugLevel::Debug, "Adding a new tail to an array: %p\n", this);
    last = new SubVector(defaultSize, last);
  }
  size++;
  return last->add(val);
}



uint64_t& LinkedVector::at(size_t pos) const {
  if (pos >= size) {
    pallas_error("Getting an element whose index (%lu) is bigger than vector size (%lu)\n", pos, size);
  }
  struct SubVector* correct_sub = last;
  while (pos < correct_sub->starting_index) {
    correct_sub = correct_sub->previous;
  }
  return correct_sub->at(pos);
}

uint64_t& LinkedVector::operator[](size_t pos) const {
  struct SubVector* correct_sub = last;
  while (pos < correct_sub->starting_index) {
    correct_sub = correct_sub->previous;
  }
  return (*correct_sub)[pos];
}

uint64_t& LinkedVector::front() const {
  return first->at(0);
}

uint64_t& LinkedVector::back() const {
  return last->at(size - 1);
}

void LinkedVector::print() const {
  std::cout << "[";
  if (size) {
    for (auto& i : *this) {
      std::cout << i << ((&i != &this->back()) ? ", " : "]");
    }
  } else
    std::cout << "]";
}

/* C++ Callbacks for C Usage */
LinkedVector* linked_vector_new() {
  return new LinkedVector();
}
uint64_t* linked_vector_add(LinkedVector* linkedVector, uint64_t val) {
  return linkedVector->add(val);
}
uint64_t* linked_vector_get(LinkedVector* linkedVector, size_t pos) {
  return &linkedVector->at(pos);
}
uint64_t* linked_vector_get_last(LinkedVector* linkedVector) {
  return &linkedVector->back();
}
void print(LinkedVector linkedVector) {
  return linkedVector.print();
}
