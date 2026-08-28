/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include <cstring>
#include <iostream>

#include "pallas/utils/pallas_dbg.h"
#include "pallas/utils/pallas_log.h"
#include "pallas/utils/pallas_storage.h"
#include "pallas/linked_vector/pallas_linked_vector.h"
#include "pallas/linked_vector/pallas_subarray.h"
#include "pallas/linked_vector/pallas_subarray_raw.h"

namespace pallas {

    SubArrayRaw::SubArrayRaw(ValueDomain domain,
                                    StoragePolicy policy,
                                    SubArrayBase* previous,
                                    const ParameterHandler* parameter_handler,
                                    LinkedVectorBase* parent) : SubArrayBase(domain, policy, previous, parameter_handler, parent) {
        _buffer_capacity = VECTOR_SIZE_2048;
        _buffer = new uint64_t[_buffer_capacity];
    }

    SubArrayRaw::SubArrayRaw(File* info_file, ValueDomain domain, StoragePolicy policy, SubArrayBase* previous, 
                            const ParameterHandler* parameter_handler, LinkedVectorBase* parent) 
                            : SubArrayBase(info_file, domain, policy, previous, parameter_handler, parent) {
        load_values(info_file);
    };

    SubArrayRaw::~SubArrayRaw(){
        delete[] _buffer;
    }
    

    /** Reconstruct the logical value stored at absolute index `pos`. */
    [[nodiscard]] uint64_t SubArrayRaw::at(size_t pos) const {
        if (!contains(pos)) {
            pallas_error("Wrong index (%lu) compared to starting index (%lu) and size (%lu)\n", pos, _starting_index, _size);
        }
        return _buffer[local_index(pos)];
    };

    /** Materialise all logical values into `given_array` in logical order. */
    void SubArrayRaw::copy_values(uint64_t* given_array) const {
        std::memcpy(given_array, _buffer, _size * sizeof(uint64_t));
    }

    /** @returns Maximum physical occupancy allowed by the attached manager. */
    size_t SubArrayRaw::capacity() const { return _buffer_capacity; }

     /** @retval true - The SubArray payload is currently resident in memory.
     *  @retval false - The SubArray payload has been freed and must be reloaded. */
     bool SubArrayRaw::has_values() const { return _buffer != nullptr; }

    /** @returns Physical occupancy of the current in-memory representation. */
    size_t SubArrayRaw::mem_size() const { return _buffer_capacity * sizeof(uint64_t); }

    /** Release the resident payload and let the manager drop transient state. */
    void SubArrayRaw::free_values() {
        delete[] _buffer;
        _buffer = nullptr;
    }
    /** Append one logical value to the subarray. */
    AddStatus SubArrayRaw::add_value(uint64_t val) {
        _buffer[_size] = val;

        if(_size >= _buffer_capacity - 1) {
            return AddStatus::Full;
        }
        return AddStatus::Ok;
    };
    
    /** Finalize the current block. */    
    void SubArrayRaw::finalize_block() {
        // Nothing to do here
    }

        
    /** Write the SubArray to a file. */
    void SubArrayRaw::write_values(File* info_file) const {
        info_file->begin_block(__func__);
        size_t size = mem_size(); // TODO: this will store the whole vector, even if it is mostly empty
        info_file->write(&size, sizeof(size), 1);
        info_file->write(_buffer, size, 1);
        info_file->end_block(__func__);
    }

    /** Read the SubArray data from a file. */
    void SubArrayRaw::load_values(File* info_file) {
        info_file->begin_block(__func__);
        size_t size;
        info_file->read(&size, sizeof(size), 1);
        _size = size/sizeof(uint64_t);
        _buffer_capacity = _size;
        _buffer = new uint64_t[size];
        info_file->read(_buffer, size, 1);
        info_file->end_block(__func__);
    };            
}