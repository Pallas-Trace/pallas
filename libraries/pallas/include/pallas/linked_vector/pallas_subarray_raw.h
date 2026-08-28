/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
#pragma once

#include "pallas_subarray.h"


#ifndef DEFAULT_VECTOR_SIZE
#define DEFAULT_VECTOR_SIZE 1000
#endif


#ifndef VECTOR_SIZE_32 
#define VECTOR_SIZE_32 32
#endif

#ifndef VECTOR_SIZE_2048 
#define VECTOR_SIZE_2048 2048
#endif

namespace pallas {

    /**
     * SubArrayRaw is the basic implementation of a subArray.
     * It simply stores timestamps/durations in a contiguous buffer
     */
    class SubArrayRaw : public SubArrayBase {
        protected:
            /** Backing buffer that contains the data */
            uint64_t* _buffer = nullptr;

            /** Size of the buffer
             */
            size_t _buffer_capacity;
        
        public:
            explicit SubArrayRaw(ValueDomain domain,
                                StoragePolicy policy = StoragePolicy::None,
                                SubArrayBase* previous = nullptr,
                                const ParameterHandler* parameter_handler = nullptr,
                                LinkedVectorBase* parent = nullptr);

            /** File-backed constructor used while reconstructing archived SubArrays. */
            explicit SubArrayRaw(File* info_file, ValueDomain domain, StoragePolicy policy = StoragePolicy::None, SubArrayBase* previous = nullptr,
                                const ParameterHandler* parameter_handler = nullptr, LinkedVectorBase* parent = nullptr);

            ~SubArrayRaw();            
            
            /********* Functions for accessing the SubArray data *********/
            /** Reconstruct the logical value stored at absolute index `pos`. */
            [[nodiscard]] uint64_t at(size_t pos) const;

            /** Materialise all logical values into `given_array` in logical order. */
            void copy_values(uint64_t* given_array) const;

            /** @returns Maximum physical occupancy allowed by the attached manager. */
            [[nodiscard]] size_t capacity() const;

            /** @retval true - The SubArray payload is currently resident in memory.
             *  @retval false - The SubArray payload has been freed and must be reloaded. */
            [[nodiscard]] bool has_values() const;

            /** @returns Physical occupancy of the current in-memory representation. */
            [[nodiscard]] size_t mem_size() const;
            /** Release the resident payload and let the manager drop transient state. */
            void free_values(); // useless ?
        /********* Functions for adding data to the SubArray *********/

            /** Append one logical value to the manager-controlled representation.
             * A child class should implement this function to actually store val
             * This function is called by the SubArrayBase::add() function before
             * updating the statistics and the size of the SubArray. 
             * */
            AddStatus add_value(uint64_t val);
            /** Finalize the current subArray once it is full. */
            void finalize_block(); // todo: return statistics on the data block to decide if we could use a lossy encoding or not

            
            /** Write the SubArray to a file. */
            void write_values(File* info_file) const;
            /** Read the SubArray data from a file. */
            void load_values(File* info_file);
    };

}