/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
/** @file
 * A custom type of Linked-List. This takes into account the fact that we never go remove anything from timestamps
 * vector.
 */
#pragma once

#include "pallas_dbg.h"
#ifndef __cplusplus
#include <stdint.h>
#endif
#ifdef __cplusplus
#include <cstring>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>
/** Default size for creating Vectors and SubVectors.*/
#define DEFAULT_VECTOR_SIZE 1000
namespace pallas {
/**
 * An hybrid between a LinkedList and a Vector.
 *
 * Contains many sub-arrays organized in a linked list fashion.
 * Does not implement any methods to remove items from itself.
 */
typedef struct LinkedVector {
  size_t size{0}; /**< Number of element stored in the vector.  */
 protected:
  const char* filePath {nullptr}; /**< Path to the file storing the durations. */
  long offset {0};          /**< Offset in the file. */
  /**
   * A fixed-sized array functionning as a node in a LinkedList.
   *
   * We call it a SubVector since it's the sub-structure of our LinkedVector struct.
   */
  struct SubVector {
   public:
    size_t size{0};               /**< Number of elements stored in the vector. */
    size_t allocated;             /**< Number of elements this vector has allocated. */
    uint64_t* array;              /**< Array of elements. Currently only used on uint64_t */
    SubVector* next{nullptr};     /**< Next SubVector in the LinkedVector. nullptr if last. */
    SubVector* previous{nullptr}; /**< Previous SubVector in the LinkedVector. nullptr if first. */
    size_t starting_index;        /**< Starting index of this SubVector. */

    /**
     * Adds a new element at the end of the vector, after its current last element.
     * The content of `val` is copied to the new element.
     *
     * @param val Value to be copied to the new element.
     * @return Reference to the new element.
     */
    uint64_t* add(uint64_t val);

    /**
     * Returns a reference to the element at specified location `pos`, with bounds checking.
     * @param pos Position of the element in the LinkedVector.
     * @return Reference to the requested element.
     */
    [[nodiscard]] uint64_t& at(size_t pos) const;

    /**
     * Returns a reference to the element at specified location `pos`, without bounds checking.
     * @param pos Position of the element in the LinkedVector.
     * @return Reference to the requested element.
     */
    [[nodiscard]] uint64_t& operator[](size_t pos) const;

    /**
     * Construct a SubVector of a given size.
     * @param new_array_size Size of the SubVector.
     * @param previous_subvector Previous SubVector in the LinkedVector.
     */
    SubVector(size_t new_array_size, SubVector* previous_subvector = nullptr);

    /**
     * Construct a SubVector from a given already allocated array, and its size.
     * @param size Size of `array`.
     * @param array Allocated array of values.
     */
    SubVector(size_t size, uint64_t* array);

    /**
     * Copies the values in array to given_array.
     * @param given_array An allocated array of correct size.
     */
    void copyToArray(uint64_t* given_array) const;
    ~SubVector();
  };
  size_t defaultSize CXX({DEFAULT_VECTOR_SIZE}); /**< Default size of the newly created SubVectors.*/
  C_CXX(void, SubVector) * first;                /**< First SubVector in the LinkedList structure.*/
  C_CXX(void, SubVector) * last;                 /**< Last SubVector in the LinkedList structure.*/
  /**
   * Loads the timestamps / durations from filePath.
   */
  void load_timestamps();

 public:
  /**
   * Creates a new LinkedVector, with a SubVector of size `defaultSize`.
   */
  LinkedVector();
  /** Creates a new LinkedVector from a file. Doesn't actually load it until and element is accessed. */
  LinkedVector(FILE* vectorFile, const char* valueFilePath);
  /**
   * Adds a new element at the end of the vector, after its current last element.
   * The content of `val` is copied to the new element.
   *
   * @param val Value to be copied to the new element.
   * @return Reference to the new element.
   */
  uint64_t* add(uint64_t val);
  /**
   * Returns a reference to the element at specified location `pos`, with bounds checking.
   *
   * To do so, parses the LinkedList from the last SubVector to the first one, stopping once the condition
   * `starting_index` <= `pos` < `starting_index` + `size`
   * @param pos Position of the element in the LinkedVector.
   * @return Reference to the requested element.
   */
  [[nodiscard]] uint64_t& at(size_t pos);
  /**
   * Returns a reference to the element at specified location `pos`, without bounds checking.
   *
   * To do so, parses the LinkedList from the last SubVector to the first one, stopping once the condition
   * `starting_index` <= `pos` < `starting_index` + `size`
   * @param pos Position of the element in the LinkedVector.
   * @return Reference to the requested element.
   */
  [[nodiscard]] uint64_t& operator[](size_t pos);
  /**
   * Returns a reference to the first element in the LinkedVector.
   * @return Reference to the first element.
   */
  [[nodiscard]] uint64_t& front();
  /**
   * Returns a reference to the last element in the LinkedVector.
   * @return Reference to the last element.
   */
  [[nodiscard]] uint64_t& back();
  /**
   * Does the inverse of loadTimestamps.
   * This is used when you want to free some memory,
   * but don't want to  lose access the actual data
   */
  void deleteTimestamps();

  /**
   * Prints the content of the LinkedVector to stdout.
   */
  void print();
  /**
   * Writes the vector to the given files.
   * @param vectorFile File where metadata is stored.
   * @param valueFile  File where data is stored (most of the time).
   */
  void writeToFile(FILE* vectorFile, FILE* valueFile);

  /**
   * Classic ForwardIterator for LinkedVector.
   */
  struct Iterator {
    /// @cond NONE
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = uint64_t;
    using pointer = uint64_t*;
    using reference = uint64_t&;
    Iterator(SubVector* s) {
      if (s) {
        cur_sub = s;
        ptr = &s->array[0];
      } else {
        cur_sub = nullptr;
        ptr = nullptr;
      }
    }
    reference operator*() const { return *ptr; }
    pointer operator->() { return ptr; }

    // Prefix increment
    Iterator& operator++() {
      i++;
      if (i < cur_sub->starting_index + cur_sub->size) {
        ptr++;
      } else {
        cur_sub = cur_sub->next;
        ptr = (cur_sub) ? &(*cur_sub)[i] : nullptr;
      }
      return *this;
    }

    // Postfix increment
    Iterator operator++(int) {
      Iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    friend bool operator==(const Iterator& a, const Iterator& b) { return a.ptr == b.ptr; };
    friend bool operator!=(const Iterator& a, const Iterator& b) { return a.ptr != b.ptr; };

   private:
    SubVector* cur_sub;
    pointer ptr;
    size_t i{0};
    /// @endcond
  };
  /** Returns an iterator pointing to the first element in the LinkedVector. */
  [[nodiscard]] Iterator begin() const { return {first}; };
  /** Returns an iterator pointing to the **past-the*end** element in the LinkedVector.
   * Here, it means nullptr.*/
  [[nodiscard]] Iterator end() const { return {nullptr}; };
  ~LinkedVector();
} LinkedVector;

typedef struct LinkedDurationVector : LinkedVector {
 private:
  /**
   * Updates the min/max/mean, taking into account all the items from 0 to size-1.
   *
   * This is because we assume the last element isn't a duration, but a timestamp.
   */
  void updateStats();
public:
 /** Does the final calculation for updating the statistics in that vector.*/
 void finalUpdateStats();
  /**
   * Adds a new element at the end of the vector, after its current last element.
   * The content of `val` is copied to the new element.
   * Updates mean, min and max.
   *
   * @param val Value to be copied to the new element.
   * @return Reference to the new element.
   */
  uint64_t* add(uint64_t val);
  uint64_t min{UINT64_MAX}; /**< Max element stored in the vector. */
  uint64_t max{0};          /**< Min element stored in the vector. */
  uint64_t mean{0};         /**< Mean of all the elements in the vector. */

  /**
   * Writes the vector to the given files.
   * If size >= 4, we do the following:\n
   *    - To vectorFile, we write [size, min, max, mean, offset] in that order.\n
   *    - To valueFile, we write the array.\n
   * If size <= 3, we don't write anything to valueFile.
   * Instead, we write [size] + array to vectorFile.\n
   * @param vectorFile File where metadata is stored.
   * @param valueFile  File where data is stored (most of the time).
   */
  void writeToFile(FILE* vectorFile, FILE* valueFile);
  /** Loads a LinkedDurationVector from a file.
   * Only loads the statistics, doesn't load the timestamps until they're accessed.
   */
  LinkedDurationVector(FILE* vectorFile, const char* valueFilePath);

  /**
   * Creates a new LinkedDurationVector, with a SubVector of size `defaultSize`.
   */
  LinkedDurationVector();
} LinkedDurationVector;
#else
typedef struct LinkedVector {
  size_t size;          /**< Number of element stored in the vector.  */
  const char* filePath; /**< Path to the file storing the durations. */
  long offset;          /**< Offset in the file. */
  size_t defaultSize;   /**< Default size of the newly created SubVectors.*/
  void* first;          /**< First SubVector in the LinkedList structure.*/
  void* last;           /**< First SubVector in the LinkedList structure.*/
} LinkedVector;

typedef struct LinkedDurationVector {
  LinkedVector linked_vector; /**< Actual vector storing the information. */
  uint64_t min;         /**< Max element stored in the vector. */
  uint64_t max;         /**< Min element stored in the vector. */
  uint64_t mean;        /**< Mean of all the elements in the vector. */
} LinkedDurationVector;
#endif

#ifdef __cplusplus
}  // namespace pallas
extern "C" {
#endif
/** Allocates and returns a new LinkedVector. */
  extern PALLAS(LinkedVector)* linked_vector_new(void);
  /**
   * Adds a new element at the end of the vector, after its current last element.
   * The content of `val` is copied to the new element.
   *
   * @param linkedVector Pointer to the vector.
   * @param val Value to be copied to the new element.
   * @return Pointer to the new element.
   */
  extern uint64_t* linked_vector_add(PALLAS(LinkedVector) * linkedVector, uint64_t val);
  /**
   * Returns a pointer to the element at specified location `pos`, with bounds checking.
   *
   * To do so, parses the LinkedList from the last SubVector to the first one, stopping once the condition
   * `starting_index` <= `pos` < `starting_index` + `size`
   * @param linkedVector Pointer to the vector.
   * @param pos Position of the element in the LinkedVector.
   * @return Pointer to the requested element.
   */
  extern uint64_t* linked_vector_get(PALLAS(LinkedVector) * linkedVector, size_t pos);
  /**
   * Returns a pointer to the last element in the LinkedVector.
   * @param linkedVector Pointer to the vector.
   * @return Pointer to the last element.
   */
  extern uint64_t* linked_vector_get_last(PALLAS(LinkedVector) * linkedVector);
  /**
   * Prints the content of the LinkedVector to stdout.
   */
  extern void print(PALLAS(LinkedVector));
CXX(
};)

/* -*-
   mode: c++;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */
