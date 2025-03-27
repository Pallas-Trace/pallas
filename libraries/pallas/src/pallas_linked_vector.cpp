/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include "pallas/pallas_linked_vector.h"
#include "pallas/pallas_log.h"
#include <cstring>
using namespace pallas;

LinkedVector::LinkedVector() {
  first = new SubVector(defaultSize);
  last = first;
}

LinkedDurationVector::LinkedDurationVector() = default;

void LinkedDurationVector::updateStats() {
  if (size > 1) {
    auto& val = at(size - 2);
    max = std::max(max, val);
    min = std::min(min, val);
    mean += val;
  }
}

void LinkedDurationVector::finalUpdateStats() {
  auto& val = back();
  max = std::max(max, val);
  min = std::min(min, val);
  mean = (mean + val) / size;
}

uint64_t* LinkedDurationVector::add(uint64_t val) {
  if (this->last->size >= this->last->allocated) {
    pallas_log(DebugLevel::Debug, "Adding a new tail to an array: %p\n", this);
    last = new SubVector(defaultSize, last);
  }
  size++;
  updateStats();
  return last->add(val);
}
uint64_t* LinkedVector::add(uint64_t val) {
  if (this->last->size >= this->last->allocated) {
    pallas_log(DebugLevel::Debug, "Adding a new tail to an array: %p\n", this);
    last = new SubVector(defaultSize, last);
  }
  size++;
  return last->add(val);
}

uint64_t& LinkedVector::at(size_t pos) {
  if (pos >= size) {
    pallas_error("Getting an element whose index (%lu) is bigger than vector size (%lu)\n", pos, size);
  }
  if (first == nullptr) {
    load_timestamps();
  }
  struct SubVector* correct_sub = last;
  while (pos < correct_sub->starting_index) {
    correct_sub = correct_sub->previous;
  }
  return correct_sub->at(pos);
}

uint64_t& LinkedVector::operator[](size_t pos) {
  if (first == nullptr) {
    load_timestamps();
  }
  struct SubVector* correct_sub = last;
  while (pos < correct_sub->starting_index) {
    correct_sub = correct_sub->previous;
  }
  return (*correct_sub)[pos];
}

uint64_t& LinkedVector::front() {
  return at(0);
}

uint64_t& LinkedVector::back() {
  return last->at(size - 1);
}


void LinkedVector::deleteTimestamps() {
  if (first == nullptr)
    return;
  pallas_log(DebugLevel::Debug, "Freeing timestamps from %p\n", this);
  auto* sub = first;
  while (sub) {
    delete[] sub->array;
    auto* temp = sub->next;
    delete sub;
    sub = temp;
  }
  first = nullptr;
  last = nullptr;
}
void LinkedVector::print() {
  std::cout << "[";
  if (size) {
    for (auto& i : *this) {
      std::cout << i << ((&i != &this->back()) ? ", " : "]");
    }
  } else
    std::cout << "]";
}

LinkedVector::~LinkedVector() {
  deleteTimestamps();
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

// Sub-vector methods

uint64_t* LinkedVector::SubVector::add(uint64_t val) {
  array[size] = val;
  return &array[size++];
}

uint64_t& LinkedVector::SubVector::at(size_t pos) const {
  if (pos >= starting_index && pos < size + starting_index) {
    return array[pos - starting_index];
  }
  pallas_error("Wrong index (%lu) compared to starting index (%lu) and size (%lu)\n", pos, starting_index, size);
}

uint64_t& LinkedVector::SubVector::operator[](size_t pos) const {
  return array[pos - starting_index];
}

LinkedVector::SubVector::SubVector(size_t new_array_size, LinkedVector::SubVector* previous_subvector) {
  previous = previous_subvector;
  starting_index = 0;
  if (previous) {
    previous->next = this;
    starting_index = previous->starting_index + previous->size;
  }
  allocated = new_array_size;
  array = new uint64_t[new_array_size];
}

LinkedVector::SubVector::SubVector(size_t size, uint64_t* array) {
  previous = nullptr;
  starting_index = 0;
  allocated = size;
  this->size = size;
  this->array = array;
}

void LinkedVector::SubVector::copyToArray(uint64_t* given_array) const {
  memcpy(given_array, array, size * sizeof(uint64_t));
}
