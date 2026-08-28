/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include <cstring>
#include <iostream>

#ifdef BMARK
#include "pallas/linked_vector/pallas_bmark.h"
#endif

#include "pallas/utils/pallas_dbg.h"
#include "pallas/utils/pallas_log.h"
#include "pallas/utils/pallas_storage.h"
#include "pallas/linked_vector/pallas_linked_vector.h"
#include "pallas/linked_vector/pallas_subarray.h"




namespace pallas {
    SubArrayStats::SubArrayStats(pallas::ValueDomain value_domain) {
        _value_domain = value_domain;
        _count = 0;
    }

    SubArrayDurationStats::SubArrayDurationStats(ValueDomain value_domain) : SubArrayStats(value_domain){ }
    SubArrayTimestampStats::SubArrayTimestampStats(ValueDomain value_domain) : SubArrayStats(value_domain){ }
}

namespace pallas {
    


    /********* Getters/Setter functions *********/
    SubArrayBase* SubArrayBase::next_subarray() const { return _next_subarray; }
    SubArrayBase* SubArrayBase::previous_subarray() const { return _prev_subarray; }
    LinkedVectorBase* SubArrayBase::parent_linked_vector() const { return _parent_linked_vector; }

    ValueDomain SubArrayBase::value_domain() const {return _value_domain; };
    StoragePolicy SubArrayBase::storage_policy() const { return _storage_policy; }

    size_t SubArrayBase::size() const { return _size; }
    size_t SubArrayBase::starting_index() const { return _starting_index; }
    size_t SubArrayBase::details_offset() const { return _details_offset;}
    size_t SubArrayBase::details_size() const { return _details_size;}

    bool SubArrayBase::contains(size_t pos) const {
        return pos >= _starting_index && pos < _starting_index + _size;
    }

    size_t SubArrayBase::local_index(size_t pos) const {
        return pos - _starting_index;
    }

#if 1
    //void SubArrayBase::set_offset(size_t offset) { _file_offset = offset; } // useless ?
#endif   

    /********* Functions for accessing the SubArray data *********/

    uint64_t SubArrayBase::operator[](size_t pos) const { return at(pos); }
    SubArrayStats& SubArrayBase::subarray_stats() const {return *_subarray_stats;}
    
    /** @returns First logical timestamp stored in this SubArray. */
    uint64_t SubArrayBase::first_value() const { return at(_starting_index); }
    /** @returns Last logical timestamp stored in this SubArray. */
    uint64_t SubArrayBase::last_value() const  { return at(_starting_index+_size); }


    AddStatus SubArrayBase::add(uint64_t val) {
        AddStatus status = add_value(val);
        _subarray_stats->add(val);
        _size++;

        if(status == AddStatus::Full) {
            finalize_block();
        }
        return status;
    }
    
    /********* Constructor/Destructor functions *********/
    
    SubArrayBase::SubArrayBase(ValueDomain domain,
                           StoragePolicy policy,
                           SubArrayBase* previous,
                           const ParameterHandler* parameter_handler,
                           LinkedVectorBase* parent)
    : _prev_subarray(previous),
      _parent_linked_vector(parent),
      _value_domain(domain),
      _storage_policy(policy),
      _subarray_phase(SubArrayPhase::RuntimeWrite) {

        if(domain == ValueDomain::Duration) _subarray_stats = new SubArrayTimestampStats(domain);
        else _subarray_stats = new SubArrayDurationStats(domain);

        if (_prev_subarray != nullptr) {
            _prev_subarray->_next_subarray = this;
            _starting_index = _prev_subarray->_starting_index + _prev_subarray->_size;
        }
    }
    
    /**
     * @brief Reconstruct the common SubArray shell from the info stream.
     *
     * This constructor restores the metadata shared by timestamp and duration
     * subarrays, recreates the appropriate manager, and relinks the SubArray into
     * the in-memory chain during analysis-time loading.
     */
    pallas::SubArrayBase::SubArrayBase(File* info_file, ValueDomain domain, StoragePolicy policy, SubArrayBase* previous,
                                    const ParameterHandler* parameter_handler, LinkedVectorBase* parent)
        : _prev_subarray(previous),
        _parent_linked_vector(parent),
        _value_domain(domain),
        _storage_policy(policy),
        _subarray_phase(SubArrayPhase::AnalysisRead) {
        
            //read_common_header(info_file);
          //  std::cout<<"not implemented!\n";
        //abort();
        if (_prev_subarray != nullptr) {
            _prev_subarray->_next_subarray = this;
            _starting_index = _prev_subarray->_starting_index + _prev_subarray->_size;
        }
    }

    SubArrayBase* SubArrayBase::create_subarray(SubArrayBase* previous,
                                                    const ParameterHandler* parameter_handler,
                                                    LinkedVectorBase* parent) {
        StoragePolicy policy = StoragePolicy::None; // TODO: automatically decide the storage policy to apply
        
        switch(policy) {
#if 0
            case StoragePolicy::Delta:
                return new SubArrayDelta(domain, previous);
                break;
            case StoragePolicy::Lossy:
                return new SubArrayPLA(domain, previous);
                break;
#endif
            default:
                return new SubArrayRaw(parent->domain(), policy, previous, parameter_handler, parent);
                break;
        }
    }

    void log_io(File*f, std::string &msg) {
        std::cout<<msg<<": "<<f->path<<":"<<f->offset()<<"\n";
    }

    SubArrayBase* SubArrayBase::load_subarray(File* data_file,
                                        SubArrayBase* previous,
                                        const ParameterHandler* parameter_handler,
                                        LinkedVectorBase* parent) {
        ValueDomain domain;
        StoragePolicy policy;
        size_t size;
        if(!data_file->isOpen)
            data_file->open("r");

        data_file->begin_block(__func__);
        data_file->read(&policy, sizeof(policy), 1);
        data_file->read(&domain, sizeof(domain), 1);
        data_file->read(&size, sizeof(size), 1);

        switch(policy) {
#if 0
            case StoragePolicy::Delta:
                return new SubArrayDelta(domain, previous);
                break;
            case StoragePolicy::Lossy:
                return new SubArrayPLA(domain, previous);
                break;
#endif
            default:
                data_file->end_block(__func__);

                return new SubArrayRaw(data_file, domain, policy, previous, parameter_handler, parent);
                
                break;
            
        }
        data_file->end_block(__func__);
        return NULL;
    }

    /** Write/read the SubArray summary (eg. stats) */

    
    void SubArrayTimestampStats::write_summary(File* info_file) {
        info_file->begin_block(__func__);
        info_file->write(&_first_timestamp, sizeof(_first_timestamp), 1);
        info_file->write(&_last_timestamp, sizeof(_last_timestamp), 1);
        info_file->end_block(__func__);
    }
    void SubArrayTimestampStats::read_summary(File* info_file) {
        std::cout<<"Read summary for subarray TimeStats\n";
        info_file->begin_block(__func__);
        info_file->read(&_first_timestamp, sizeof(_first_timestamp), 1);
        info_file->read(&_last_timestamp, sizeof(_last_timestamp), 1);
        info_file->end_block(__func__);
    }

    void SubArrayDurationStats::write_summary(File* info_file) {
        info_file->begin_block(__func__);
        info_file->write(&_min_duration, sizeof(_min_duration), 1);
        info_file->write(&_max_duration, sizeof(_max_duration), 1);
        info_file->write(&_mean_duration, sizeof(_mean_duration), 1);
        info_file->write(&_mean_duration_is_finalized, sizeof(_mean_duration_is_finalized), 1);
        info_file->end_block(__func__);
    }
    void SubArrayDurationStats::read_summary(File* info_file) {
        info_file->begin_block(__func__);
        std::cout<<"Read summary for subarray DurationStats\n";
        info_file->read(&_min_duration, sizeof(_min_duration), 1);
        info_file->read(&_max_duration, sizeof(_max_duration), 1);
        info_file->read(&_mean_duration, sizeof(_mean_duration), 1);
        info_file->read(&_mean_duration_is_finalized, sizeof(_mean_duration_is_finalized), 1);
        info_file->end_block(__func__);
    }

    #if 0
    void pallas::SubArrayBase::write_summary(File* info_file) const {
        info_file->begin_block(__func__);
        info_file->write(&_storage_policy, sizeof(_storage_policy), 1);
        info_file->write(&_value_domain, sizeof(_value_domain), 1);
        info_file->write(&_size, sizeof(_size), 1);
        info_file->write(&_starting_index, sizeof(_starting_index), 1);
        _subarray_stats->write_summary(info_file);
//        info_file->write(&_subarray_stats->_stats, sizeof(_subarray_stats_stats), 1);
        info_file->end_block(__func__);
    }
#endif
    void pallas::SubArrayBase::read_summary(File* info_file) {
        info_file->begin_block(__func__);
        std::cout<<"Read summary for subarray\n";
        info_file->read(&_storage_policy, sizeof(_storage_policy), 1);
        info_file->read(&_value_domain, sizeof(_value_domain), 1);
        info_file->read(&_size, sizeof(_size), 1);
        info_file->read(&_starting_index, sizeof(_starting_index), 1);
        _subarray_stats->read_summary(info_file);
   //     info_file->read(&_subarray_stats->_stats, sizeof(_subarray_stats_stats), 1);
        info_file->end_block(__func__);
    }

    /** Write/read the SubArray data (eg. timestamps) and calls the child class write_values */
    void pallas::SubArrayBase::write_details(File* details_file, size_t* data_size, off_t *data_offset) {
        _details_offset = details_file->seek(0, SEEK_END);
        if(data_offset)
            *data_offset = _details_offset;

        details_file->begin_block(__func__);                
        details_file->write(&_storage_policy, sizeof(_storage_policy), 1);
        details_file->write(&_value_domain, sizeof(_value_domain), 1);
        details_file->write(&_size, sizeof(_size), 1);

        // Call the child write_values function to actually write the timestamps
        write_values(details_file);
        details_file->end_block(__func__);
        off_t end_offset = details_file->seek(0, SEEK_END);
        _details_size = end_offset - _details_offset;
        if(data_size) {
            *data_size = _details_size;
        }
    }

    void pallas::SubArrayBase::read_data(File* data_file) {
        data_file->begin_block(__func__);
        std::cout<<"Read data for subarray\n";
        data_file->read(&_storage_policy, sizeof(_storage_policy), 1);
        data_file->read(&_value_domain, sizeof(_value_domain), 1);
        data_file->read(&_size, sizeof(_size), 1);

        // Call the child write_values function to actually write the timestamps
        load_values(data_file);
        data_file->end_block(__func__);
    }        

    SubArrayBase::~SubArrayBase() {
        delete _subarray_stats;
        //free_values();
    }

}

#if 0
extern size_t numberPreRawBytes;
extern size_t numberRawBytes;

// These functions are defined in pallas_storage.cpp
extern void _pallas_compress_write(uint64_t* src, size_t n, File* file, const pallas::ParameterHandler* parameter_handler);
extern uint64_t* _pallas_compress_read(size_t n, File* file, const pallas::ParameterHandler& parameter_handler);

/** Methods Pertaining to the policy manager of the SubArray */
namespace pallas {

namespace {

constexpr uint8_t kStoragePolicyMask = 0x03;

LossyPolicy resolve_lossy_policy(ValueDomain domain,
                                 StoragePolicy policy,
                                 const ParameterHandler* parameter_handler) {
    const auto default_lossy =
            (domain == ValueDomain::Timestamp) ? DEFAULT_LOSSY_TIME : DEFAULT_LOSSY_DURATION;
    if (policy != StoragePolicy::Lossy) {
        return default_lossy;
    }
    if (parameter_handler == nullptr) {
        return default_lossy;
    }
    return (domain == ValueDomain::Timestamp)
           ? parameter_handler->getTimeLossyPolicy()
           : parameter_handler->getDurationLossyPolicy();
}

#ifdef BMARK
void record_subarray_write_metrics(BmarkFamily family,
                                   File* data_file,
                                   long start_offset,
                                   uint64_t pre_raw_bytes,
                                   uint64_t raw_bytes) {
    if (data_file == nullptr || start_offset < 0 || family == BmarkFamily::Unknown) {
        return;
    }

    const long end_offset = std::ftell(data_file);
    if (end_offset < start_offset) {
        return;
    }

    bmark_note_subarray_write(
            family,
            pre_raw_bytes,
            raw_bytes,
            static_cast<uint64_t>(end_offset - start_offset));
}

void record_subarray_error_metrics(BmarkFamily family,
                                   const uint64_t* exact_values,
                                   const uint64_t* observed_values,
                                   size_t _size) {
    if (family == BmarkFamily::Unknown) {
        return;
    }
    bmark_note_error_values(family, exact_values, observed_values, _size);
}
#endif

}  // namespace

/**
 * @brief Pack the SubArray storage-policy header into one persisted byte.
 *
 * On disk the policy byte is laid out as:
 * @code
 *   bit index:  7 6 5 4 3 2 1 0
 *               +-----------+---+
 *               | lossy id  |sp |
 *               +-----------+---+
 *
 *   sp bits:
 *     00 -> StoragePolicy::None
 *     01 -> StoragePolicy::Delta
 *     10 -> StoragePolicy::Lossy
 *
 *   lossy id bits:
 *     valid only when `sp == StoragePolicy::Lossy`
 *     stores the `LossyPolicy` enum value in the upper 6 bits
 * @endcode
 *
 * This keeps the common SubArray header compact while still preserving both
 * the coarse storage family and the concrete lossy variant needed to rebuild
 * the right manager during analysis-time reconstruction.
 */
uint8_t SubArrayBase::pack_subarray_flags() const {
    const auto storage_bits = static_cast<uint8_t>(storage_policy) & kStoragePolicyMask;
    const auto lossy_bits = static_cast<uint8_t>(lossy_storage_policy) << 2;
    return static_cast<uint8_t>(storage_bits | lossy_bits);
}

/** @brief Decode the packed one-byte storage-policy header produced by `pack_subarray_flags()`. */
void SubArrayBase::unpack_subarray_flags(uint8_t encoded_policy) {
    const auto storage_bits = static_cast<uint8_t>(encoded_policy & kStoragePolicyMask);
    if (storage_bits <= static_cast<uint8_t>(StoragePolicy::Lossy)) {
        storage_policy = static_cast<StoragePolicy>(storage_bits);
    } else {
        storage_policy = StoragePolicy::None;
    }

    const auto lossy_bits = static_cast<uint8_t>(encoded_policy >> 2);
    if (storage_policy == StoragePolicy::Lossy &&
        lossy_bits <= static_cast<uint8_t>(LossyPolicy::Spike32)) {
        lossy_storage_policy = static_cast<LossyPolicy>(lossy_bits);
    }
}

namespace {

/** Methods Peratining to NoneManger Class */
namespace pallas {

size_t NoneManager::_capacity() const {
    return DEFAULT_VECTOR_SIZE;
}

AddStatus NoneManager::add(uint64_t val) {
    if (parent.physical_size >= parent.capacity()) {
        return AddStatus::Full;
    }

    parent.buffer[parent.physical_size] = val;
    parent._size++;
    parent.physical_size++;
    return AddStatus::Ok;
}

uint64_t NoneManager::at(size_t pos) const {
    if (!parent.contains(pos)) {
        pallas_error("Wrong index (%lu) compared to starting index (%lu) and size (%lu)\n", pos, parent._starting_index, parent._size);
    }
    return parent.buffer[parent.local_index(pos)];
}

void NoneManager::copy_to_array(uint64_t* given_array) const {
    std::memcpy(given_array, parent.buffer, parent._size * sizeof(uint64_t));
}

void NoneManager::write_data(File* data_file, const ParameterHandler* parameter_handler) {
    if (data_file == nullptr || parameter_handler == nullptr || parent.buffer == nullptr) {
        return;
    }

    std::cout<<"Write a None chunk\n";
    const long current_offset = std::ftell(data_file);
    if (current_offset >= 0) {
        parent.file_offset = static_cast<size_t>(current_offset);
    }

    const auto pre_raw_bytes = static_cast<uint64_t>(parent.size() * sizeof(uint64_t));
    const auto raw_bytes = static_cast<uint64_t>(parent.mem_size() * sizeof(uint64_t));
    numberPreRawBytes += pre_raw_bytes;
    numberRawBytes += raw_bytes;
    _pallas_compress_write(parent.buffer, parent.mem_size(), data_file, parameter_handler);
#ifdef BMARK
    const auto family =
            (parent.parent_lv != nullptr) ? parent.parent_lv->get_bmark_family() : BmarkFamily::Unknown;
    record_subarray_error_metrics(family, parent.buffer, parent.buffer, parent.size());
    record_subarray_write_metrics(family, data_file, current_offset, pre_raw_bytes, raw_bytes);
#endif
    parent.free_values();
}

void NoneManager::load_data(File* data_file, const ParameterHandler& parameter_handler) {
    if (data_file == nullptr) {
        return;
    }

    delete[] parent.buffer;
    parent.buffer = _pallas_compress_read(parent.mem_size(), data_file, parameter_handler);
}

void NoneManager::on_values_freed() {}
} 




/** Methods Pertaining to the base SubArray Class */
namespace pallas {



SubArrayBase::~SubArrayBase() {
    free_values();
}



uint64_t* SubArrayBase::raw_buffer() {
    return buffer;
}

void SubArrayBase::free_values() {
    delete[] buffer;
    buffer = nullptr;
    if (manager != nullptr) {
        manager->on_values_freed();
    }
}

void SubArrayBase::rebuild_manager() {
    manager = make_manager(*this, value_domain, storage_policy, lossy_storage_policy);
    manager->on_subarray_initialized();
}

uint64_t SubArrayBase::at(size_t pos) const {
    return manager->at(pos);
}

uint64_t SubArrayBase::operator[](size_t pos) const {
    return at(pos);
}

void SubArrayBase::copy_values(uint64_t* given_array) const {
    manager->copy_to_array(given_array);
}

ValueDomain SubArrayBase::domain() const {
    return value_domain;
}

StoragePolicy SubArrayBase::policy() const {
    return storage_policy;
}

LossyPolicy SubArrayBase::lossy_policy() const {
    return lossy_storage_policy;
}

SubArrayPhase SubArrayBase::phase() const {
    return subarray_phase;
}

size_t SubArrayBase::size() const {
    return _size;
}

size_t SubArrayBase::mem_size() const {
    return physical_size;
}

size_t SubArrayBase::capacity() const {
    if (manager == nullptr) {
        return DEFAULT_VECTOR_SIZE;
    }
    return manager->_capacity();
}

size_t SubArrayBase::starting_index() const {
    return _starting_index;
}

size_t SubArrayBase::offset() const {
    return file_offset;
}



bool SubArrayBase::has_values() const {
    return buffer != nullptr;
}

#ifdef BMARK
BmarkFamily SubArrayBase::get_bmark_family() const {
    return (parent_lv != nullptr) ? parent_lv->get_bmark_family() : BmarkFamily::Unknown;
}
#endif

void SubArrayBase::load_data(File* data_file, const ParameterHandler& parameter_handler) {
    manager->load_data(data_file, parameter_handler);
}

}

/** Methods Peratining to the TimeSubArray Class */
namespace pallas {

TimeSubArray::TimeSubArray(StoragePolicy policy,
                           TimeSubArray* previous,
                           const ParameterHandler* parameter_handler,
                           LinkedVectorBase* parent)
    : SubArrayBase(ValueDomain::Timestamp, policy, previous, parameter_handler, parent) {}

AddStatus TimeSubArray::add(uint64_t val) {
    const bool is_first_value = (_size == 0);
    auto status = manager->add(val);
    if (status == AddStatus::Ok) {
        if (is_first_value) {
            first_timestamp = val;
            last_timestamp = val;
        }
        last_timestamp = val;
    }
    return status;
}

void TimeSubArray::write_data(File* file, const ParameterHandler* parameter_handler) {
    manager->write_data(file, parameter_handler);
}

uint64_t TimeSubArray::first_value() const {
    return first_timestamp;
}

uint64_t TimeSubArray::last_value() const {
    return last_timestamp;
}

}

/** Methods Peratining to the DurationSubArray Class */
namespace pallas {

DurationSubArray::DurationSubArray(StoragePolicy policy,
                                   DurationSubArray* previous,
                                   const ParameterHandler* parameter_handler,
                                   LinkedVectorBase* parent)
    : SubArrayBase(ValueDomain::Duration, policy, previous, parameter_handler, parent) {}

AddStatus DurationSubArray::add(uint64_t val) {
    auto status = manager->add(val);
    if (status == AddStatus::Ok) {
        update_statistics(val);
    }
    return status;
}

void DurationSubArray::write_data(File* file, const ParameterHandler* parameter_handler) {
    manager->write_data(file, parameter_handler);
}

void DurationSubArray::update_statistics(uint64_t current_value) {
    min_duration = (current_value < min_duration) ? current_value : min_duration;
    max_duration = (current_value > max_duration) ? current_value : max_duration;
    if (mean_duration_is_finalized && _size > 1) {
        mean_duration *= (_size - 1);
    }
    mean_duration_is_finalized = false;
    mean_duration += current_value;
}

void DurationSubArray::final_update_mean() {
    if (_size == 0 || mean_duration_is_finalized) {
        return;
    }
    mean_duration /= _size;
    mean_duration_is_finalized = true;
    pallas_assert_inferior_equal(mean_duration, max_duration);
    pallas_assert_inferior_equal(min_duration, mean_duration);
}

uint64_t DurationSubArray::min_value() const {
    return min_duration;
}

uint64_t DurationSubArray::max_value() const {
    return max_duration;
}

uint64_t DurationSubArray::mean_value() const {
    return mean_duration;
}

}  // namespace pallas

#endif