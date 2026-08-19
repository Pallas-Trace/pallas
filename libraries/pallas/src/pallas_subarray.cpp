/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include <cstring>

#ifdef BMARK
#include "pallas/linked_vector/pallas_bmark.h"
#endif

#include "pallas/utils/pallas_dbg.h"
#include "pallas/utils/pallas_log.h"
#include "pallas/linked_vector/pallas_linked_vector.h"
#include "pallas/linked_vector/pallas_subarray.h"

extern size_t numberPreRawBytes;
extern size_t numberRawBytes;

// These functions are defined in pallas_storage.cpp
extern void _pallas_compress_write(uint64_t* src, size_t n, FILE* file, const pallas::ParameterHandler* parameter_handler);
extern uint64_t* _pallas_compress_read(size_t n, FILE* file, const pallas::ParameterHandler& parameter_handler);

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
                                   FILE* data_file,
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
                                   size_t value_count) {
    if (family == BmarkFamily::Unknown) {
        return;
    }
    bmark_note_error_values(family, exact_values, observed_values, value_count);
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

/**
 * @brief Factory that selects the concrete manager attached to one SubArray.
 *
 * The choice depends on the logical value domain, the coarse `StoragePolicy`,
 * and, for lossy paths, the resolved `LossyPolicy` variant. This keeps manager
 * selection in one place for both runtime-created and file-reconstructed
 * SubArrays.
 */
std::unique_ptr<Manager> make_manager(SubArrayBase& parent,
                                      ValueDomain domain,
                                      StoragePolicy policy,
                                      LossyPolicy lossy_policy) {
    switch (policy) {
        case StoragePolicy::None:
            return std::make_unique<NoneManager>(parent);
        case StoragePolicy::Lossy:
            if (domain == ValueDomain::Timestamp) {
                switch (lossy_policy) {
                    case LossyPolicy::PLA8:
                        return std::make_unique<PLAManager>(parent, 8);
                    case LossyPolicy::PLA16:
                        return std::make_unique<PLAManager>(parent, 16);
                    case LossyPolicy::PLA32:
                        return std::make_unique<PLAManager>(parent, 32);
                    case LossyPolicy::PLA4:
                        return std::make_unique<PLAManager>(parent, 4);
                    case LossyPolicy::Spike4:
                    case LossyPolicy::Spike8:
                    case LossyPolicy::Spike16:
                    case LossyPolicy::Spike32:
                        return std::make_unique<DeltaManager>(parent, domain);
                }
                return std::make_unique<DeltaManager>(parent, domain);
            }
            if (domain == ValueDomain::Duration) {
                switch (lossy_policy) {
                    case LossyPolicy::Spike4:
                        return std::make_unique<DurationSpikeManager>(parent, 4);
                    case LossyPolicy::Spike8:
                        return std::make_unique<DurationSpikeManager>(parent, 8);
                    case LossyPolicy::Spike16:
                        return std::make_unique<DurationSpikeManager>(parent, 16);
                    case LossyPolicy::Spike32:
                        return std::make_unique<DurationSpikeManager>(parent, 32);
                    case LossyPolicy::PLA4:
                    case LossyPolicy::PLA8:
                    case LossyPolicy::PLA16:
                    case LossyPolicy::PLA32:
                        return std::make_unique<DeltaManager>(parent, domain);
                }
                return std::make_unique<DeltaManager>(parent, domain);
            }
            return std::make_unique<NoneManager>(parent);
        case StoragePolicy::Delta:
            if (domain == ValueDomain::Timestamp) {
                return std::make_unique<DeltaManager>(parent, domain);
            }
            if (domain == ValueDomain::Duration) {
                return std::make_unique<DeltaManager>(parent, domain);
            }
            return std::make_unique<NoneManager>(parent);
    }

    return std::make_unique<NoneManager>(parent);
}

}  // namespace

Manager::~Manager() = default;

void Manager::on_subarray_initialized() {}

}

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
    parent.value_count++;
    parent.physical_size++;
    return AddStatus::Ok;
}

uint64_t NoneManager::at(size_t pos) const {
    if (!parent.contains(pos)) {
        pallas_error("Wrong index (%lu) compared to starting index (%lu) and size (%lu)\n", pos, parent.first_index, parent.value_count);
    }
    return parent.buffer[parent.local_index(pos)];
}

void NoneManager::copy_to_array(uint64_t* given_array) const {
    std::memcpy(given_array, parent.buffer, parent.value_count * sizeof(uint64_t));
}

void NoneManager::write_data(FILE* data_file, const ParameterHandler* parameter_handler) {
    if (data_file == nullptr || parameter_handler == nullptr || parent.buffer == nullptr) {
        return;
    }

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

void NoneManager::load_data(FILE* data_file, const ParameterHandler& parameter_handler) {
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

SubArrayBase::SubArrayBase(ValueDomain domain,
                           StoragePolicy policy,
                           SubArrayBase* previous,
                           const ParameterHandler* parameter_handler,
                           LinkedVectorBase* parent)
    : prev(previous),
      manager(nullptr),
      parent_lv(parent),
      value_domain(domain),
      storage_policy(policy),
      lossy_storage_policy(resolve_lossy_policy(domain, policy, parameter_handler)),
      subarray_phase(SubArrayPhase::RuntimeWrite) {
    manager = make_manager(*this, domain, policy, lossy_storage_policy);
    buffer = new uint64_t[manager->_capacity()];
    manager->on_subarray_initialized();
    if (prev != nullptr) {
        prev->next = this;
        first_index = prev->first_index + prev->value_count;
    }
}

SubArrayBase::~SubArrayBase() {
    free_values();
}

bool SubArrayBase::contains(size_t pos) const {
    return pos >= first_index && pos < first_index + value_count;
}

size_t SubArrayBase::local_index(size_t pos) const {
    return pos - first_index;
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
    return value_count;
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
    return first_index;
}

size_t SubArrayBase::offset() const {
    return file_offset;
}

SubArrayBase* SubArrayBase::next_subarray() const {
    return next;
}

SubArrayBase* SubArrayBase::previous_subarray() const {
    return prev;
}

bool SubArrayBase::has_values() const {
    return buffer != nullptr;
}

void SubArrayBase::set_offset(size_t offset) {
    file_offset = offset;
}

#ifdef BMARK
BmarkFamily SubArrayBase::get_bmark_family() const {
    return (parent_lv != nullptr) ? parent_lv->get_bmark_family() : BmarkFamily::Unknown;
}
#endif

void SubArrayBase::load_data(FILE* data_file, const ParameterHandler& parameter_handler) {
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
    const bool is_first_value = (value_count == 0);
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

void TimeSubArray::write_data(FILE* file, const ParameterHandler* parameter_handler) {
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

void DurationSubArray::write_data(FILE* file, const ParameterHandler* parameter_handler) {
    manager->write_data(file, parameter_handler);
}

void DurationSubArray::update_statistics(uint64_t current_value) {
    min_duration = (current_value < min_duration) ? current_value : min_duration;
    max_duration = (current_value > max_duration) ? current_value : max_duration;
    if (mean_duration_is_finalized && value_count > 1) {
        mean_duration *= (value_count - 1);
    }
    mean_duration_is_finalized = false;
    mean_duration += current_value;
}

void DurationSubArray::final_update_mean() {
    if (value_count == 0 || mean_duration_is_finalized) {
        return;
    }
    mean_duration /= value_count;
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
