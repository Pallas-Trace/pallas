/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */


#pragma once

#include "pallas_dbg.h"
#ifndef __cplusplus
#include <stdint-gcc.h>
#endif
#ifdef __cplusplus
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

namespace pallas {
#endif
  typedef enum PallasBufferStatus {
    Invalid,		/**< The buffer is invalid. */
    Initialized,	/**< The buffer is empty and ready to be used. */
    Pending,		/**< The buffer is being used. */
    Full,		/**< The buffer is full and can be copied to disk. */
    FlushPending,  	/**< The buffer is being copied to disk. */
  };

  typedef struct PallasBuffer {

    /** Status of the PallasBuffer  */
    enum PallasBufferStatus status CXX({Invalid});

    /** Number of element stored in the buffer. */
    size_t n_elem CXX({0});

    /** Size of each element stored in the buffer. */
    size_t elem_size CXX({0});

    /** Buffer that contains the elements. */
    void* buffer CXX({nullptr});

    /** File where the buffer will be written.  */
    FILE* output_file CXX({0});

    /** Sequence number of the buffer in the file. Buffers are written to disk in sequential order
	starting from zero.  */   
    size_t sequence_number CXX({0});

#ifdef __cplusplus
  public:
    PallasBuffer(size_t n_elem, size_t elem_size);

    /** Start using a buffer.
	The buffer status changes from Initialized to Pending, and data can be written to it.
    */
    void assign(FILE* output_file, size_t sequence_number);

    /** Mark the buffer as Full. The buffer can now be copied to disk.*/
    void setFull();
#endif
  };

  typedef struct PallasBufferManager {
    /** Number of element stored in the buffer. */
    size_t n_elem CXX({0});

    /** Size of each element stored in the buffer. */
    size_t elem_size CXX({0});

    
#ifdef __cplusplus

    std::List<PallasBuffer*> initializedBuffers CXX({});
    std::List<PallasBuffer*> pendingBuffers CXX({});
    std::List<PallasBuffer*> fullBuffers CXX({});
    std::List<PallasBuffer*> flushPendingBuffers CXX({});
    
  public:
    PallasBufferManager(size_t n_elem, size_t elem_size);

    PallasBuffer* allocBuffer(int nb_buffers);

    /** Start using a buffer.
	The buffer status changes from Initialized to Pending, and data can be written to it.
    */
    PallasBuffer* assign(FILE* output_file, size_t sequence_number);

    /** Mark the buffer as Full. The buffer can now be copied to disk.*/
    void setFull(PallasBuffer* buffer);
#endif

  };


#ifdef __cplusplus
  private:
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
      uint64_t* add(uint64_t val) {
	array[size] = val;
	return &array[size++];
      }
    };

  public:
    /**
     * Adds a new element at the end of the vector, after its current last element.
     * The content of `val` is copied to the new element.
     *
     * @param val Value to be copied to the new element.
     * @return Reference to the new element.
     */
    uint64_t* add(uint64_t val);

    void update_last_duration(uint64_t val);

  }
    
}
    
#endif
