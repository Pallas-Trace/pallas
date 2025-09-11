/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
#include "pallas/pallas_linked_vector.h"
#include <sstream>
#include "pallas/pallas_log.h"

#define SAME_FOR_BOTH_VECTORS(return_type, function_core) return_type LinkedVector::function_core return_type LinkedDurationVector::function_core

namespace pallas {

std::string LinkedVector::to_string() {
    if (size == 0)
        return "[ ]";
    std::ostringstream output;
    output << "[";
    for (size_t i = 0; i < size; i++) {
        if (i != size - 1) {
            output << this->at(i) << ", ";
        }
        else {
            output << this->at(i) << "]";
        }
    }
    return output.str();
}

std::string LinkedDurationVector::to_string() {
    if (size == 0)
        return "[ ]";
    std::ostringstream output;
    output << "[";
    for (size_t i = 0; i < size; i++) {
        if (i != size - 1) {
            output << this->at(i) << ", ";
        }
        else {
            output << this->at(i) << "]";
        }
    }
    if (size >= 2) {
        output << " { " << min << ", " << mean << ", " << max << " }";
    }
    return output.str();
}

LinkedVector::LinkedVector(ParameterHandler& p ) : parameter_handler(p) {
    first = new SubArray(DEFAULT_VECTOR_SIZE);
    last = first;
}

LinkedDurationVector::LinkedDurationVector(ParameterHandler& p ) : parameter_handler(p) {
    first = new SubArray(DEFAULT_VECTOR_SIZE);
    last = first;
}

uint64_t* LinkedVector::SubArray::add(uint64_t val) {
    array[size] = val;
    return &array[size++];
}

uint64_t* LinkedDurationVector::SubArray::add(uint64_t val) {
    array[size++] = val;
    update_statistics();
    return &array[size-1];
}

SAME_FOR_BOTH_VECTORS(
  uint64_t&,
  SubArray::at(size_t pos) const {
      if (pos >= starting_index && pos < size + starting_index) {
          return array[pos - starting_index];
      }
      pallas_error("Wrong index (%lu) compared to starting index (%lu) and size (%lu)\n", pos, starting_index, size);
  })

SAME_FOR_BOTH_VECTORS(uint64_t&, SubArray::operator[](size_t pos) const { return array[pos - starting_index]; })

LinkedVector::SubArray::SubArray(size_t size, LinkedVector::SubArray* previous) {
    this->previous = previous;
    starting_index = 0;
    if (previous) {
        previous->next = this;
        starting_index = previous->starting_index + previous->size;
    }
    allocated = size;
    array = new uint64_t[size];
}

LinkedDurationVector::SubArray::SubArray(size_t size, LinkedDurationVector::SubArray* previous) {
    this->previous = previous;
    starting_index = 0;
    if (previous) {
        previous->next = this;
        starting_index = previous->starting_index + previous->size;
    }
    allocated = size;
    array = new uint64_t[size];
}


SAME_FOR_BOTH_VECTORS(, SubArray::~SubArray() { delete[] array; })

SAME_FOR_BOTH_VECTORS(void, SubArray::copy_to_array(uint64_t* given_array) const { memcpy(given_array, array, size * sizeof(uint64_t)); })

void LinkedDurationVector::update_statistics() {
    auto& val = at(size - 1);
    max = std::max(max, val);
    min = std::min(min, val);
    mean += val;
}

void LinkedDurationVector::SubArray::update_statistics() {
    auto& val = at(size - 1 + starting_index);
        max = std::max(max, val);
        min = std::min(min, val);
        mean += val;
}

uint64_t* LinkedDurationVector::add(uint64_t val) {
    if (this->last->size >= this->last->allocated) {
        last->final_update_mean();
        last = new SubArray(DEFAULT_VECTOR_SIZE, last);
    }
    size++;
    auto* out = last->add(val);
    update_statistics();
    return out;
}

uint64_t* LinkedVector::add(uint64_t val) {
    if (this->last->size >= this->last->allocated) {
        last = new SubArray(DEFAULT_VECTOR_SIZE, last);
    }
    size++;
    return last->add(val);
}

SAME_FOR_BOTH_VECTORS(void, load_all_data() {
    auto* v = first;
    while (v) {
        load_data(v);
        v = v->next;
    }
})


SAME_FOR_BOTH_VECTORS(
    uint64_t&,
    at(size_t pos) {
      if (pos >= size) {
          pallas_error("Getting an element whose index (%lu) is bigger than LinkedVector size (%lu)\n", pos, size);
      }
      SubArray* correct_sub = last;
      while (pos < correct_sub->starting_index) {
          correct_sub = correct_sub->previous;
      }
      if (correct_sub->array == nullptr) {
          while (parameter_handler.loaded_durations_size > parameter_handler.max_memory_durations) {
            auto * temp = (SubArray*) parameter_handler.subvector_queue.front();
            parameter_handler.subvector_queue.pop_front();
            delete temp->array;
            temp->array = nullptr;
            parameter_handler.loaded_durations_size -= temp->size * sizeof(uint64_t);
        }
          load_data(correct_sub);
      }
      return correct_sub->at(pos);
  })

SAME_FOR_BOTH_VECTORS(
  uint64_t&,
  operator[](size_t pos) {
      SubArray* correct_sub = last;
      while (pos < correct_sub->starting_index) {
          correct_sub = correct_sub->previous;
      }
      if (correct_sub->array == nullptr) {
          while (parameter_handler.loaded_durations_size > parameter_handler.max_memory_durations) {
            auto * temp = (SubArray*) parameter_handler.subvector_queue.front();
            parameter_handler.subvector_queue.pop_front();
            delete temp->array;
            temp->array = nullptr;
            parameter_handler.loaded_durations_size -= temp->size * sizeof(uint64_t);
        }
          load_data(correct_sub);
      }
      return (*correct_sub)[pos];
  })

size_t LinkedVector::getFirstOccurrenceBefore(pallas_timestamp_t ts) {
    auto current_subarray = first;
    // First, we find the correct subarray
    while (current_subarray->last_value < ts) {
        if (current_subarray->next &&  ts < current_subarray->next->first_value) {
            return current_subarray->next->starting_index - 1;
        }
        current_subarray = current_subarray->next;
        if (current_subarray == nullptr) {
            return -1;
        }
    }
    // Then we do a dichotomy on it.
    size_t start = 0;
    size_t end = current_subarray->size;
    if (current_subarray->array == nullptr) {
        load_data(current_subarray);
    }
    while (start + 1 < end) {
        size_t middle = (start + end ) / 2;
        if (current_subarray->array[middle] <= ts) {
            start = middle;
        } else {
            end = middle;
        }
    }
    if (ts == current_subarray->array[end]) {
        return end + current_subarray->starting_index;
    }
    return start+ current_subarray->starting_index;
}

size_t LinkedVector::getFirstOccurrenceAfter(pallas_timestamp_t ts) {
    auto current_subarray = first;
    // First, we find the correct subarray
    while (current_subarray->last_value < ts) {
        if (current_subarray->next && ts < current_subarray->next->first_value) {
            return current_subarray->next->starting_index;
        }
        current_subarray = current_subarray->next;
        if (current_subarray == nullptr) {
            return -1;
        }
    }
    // Then we do a dichotomy on it.
    size_t start = 0;
    size_t end = current_subarray->size;
    if (current_subarray->array == nullptr) {
        load_data(current_subarray);
    }
    while (start + 1 < end) {
        size_t middle = (start + end ) / 2;
        if (current_subarray->array[middle] < ts) {
            start = middle;
        } else {
            end = middle;
        }
    }
    if (ts == current_subarray->array[start]) {
        return start+ current_subarray->starting_index;
    }
    return end+ current_subarray->starting_index;
}

pallas_duration_t LinkedDurationVector::computeDurationBetween(size_t start_index, size_t end_index) {
    auto* start_subarray = first;
    while (start_subarray->starting_index + start_subarray->size < start_index) {
        start_subarray = start_subarray->next;
    }
    pallas_duration_t sum = 0;
    if ( start_subarray->starting_index != start_index ) {
        for (size_t i = start_index; i < start_subarray->starting_index + start_subarray->size && i <= end_index; i++) {
            sum += at(i);
        }
        start_subarray = start_subarray->next;
    }
    while (start_subarray != nullptr && start_subarray->starting_index + start_subarray->size < end_index) {
        sum += start_subarray->mean * start_subarray->size;
        start_subarray = start_subarray->next;
    }
    if (start_subarray == nullptr) return sum;
    for (size_t i = start_subarray->starting_index; i < end_index; i++) {
        sum += at(i);
    }
    return sum;
}




uint64_t& LinkedVector::front() {
    return first->first_value;
}

uint64_t& LinkedVector::back() {
    return last->last_value;
}


uint64_t& LinkedDurationVector::front() {
    return at(0);
}

uint64_t& LinkedDurationVector::back() {
    return at(size - 1);
}

SAME_FOR_BOTH_VECTORS(
  void,
  free_data() {
      if (first == nullptr)
          return;
      pallas_log(DebugLevel::Debug, "Freeing timestamps from %p\n", this);
      SubArray* sub = first;
      while (sub) {
          auto* temp = sub->next;
          delete sub;
          sub = temp;
      }
      first = nullptr;
      last = nullptr;
  })

LinkedVector::~LinkedVector() {
    free_data();
}

LinkedDurationVector::~LinkedDurationVector() {
    free_data();
}

SAME_FOR_BOTH_VECTORS(void, reset_offsets() {
    auto* v = first;
    while (v != nullptr) {
        v->offset = 0;
        v = v->next;
    }
})

// Sub-LinkedVector methods

}  // namespace pallas