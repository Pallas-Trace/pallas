/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "pallas/utils/pallas_dbg.h"
#include "pallas/utils/pallas_log.h"
#include "pallas/utils/pallas_serialisation.h"
#include "pallas/utils/pallas_subarray.h"

extern size_t numberPreRawBytes;
extern size_t numberRawBytes;

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

}  // namespace

uint8_t SubArrayBase::encode_policy_byte() const {
    const auto storage_bits = static_cast<uint8_t>(storage_policy) & kStoragePolicyMask;
    const auto lossy_bits = static_cast<uint8_t>(lossy_storage_policy) << 2;
    return static_cast<uint8_t>(storage_bits | lossy_bits);
}

void SubArrayBase::decode_policy_byte(uint8_t encoded_policy) {
    const auto storage_bits = static_cast<uint8_t>(encoded_policy & kStoragePolicyMask);
    if (storage_bits <= static_cast<uint8_t>(StoragePolicy::Lossy)) {
        storage_policy = static_cast<StoragePolicy>(storage_bits);
    } else {
        storage_policy = StoragePolicy::None;
    }

    const auto lossy_bits = static_cast<uint8_t>(encoded_policy >> 2);
    if (storage_policy == StoragePolicy::Lossy &&
        lossy_bits <= static_cast<uint8_t>(LossyPolicy::PLA32)) {
        lossy_storage_policy = static_cast<LossyPolicy>(lossy_bits);
    }
}

namespace {

std::unique_ptr<Manager> make_manager(ValueDomain domain, StoragePolicy policy, LossyPolicy lossy_policy) {
    switch (policy) {
        case StoragePolicy::None:
            return std::make_unique<NoneManager>();
        case StoragePolicy::Lossy:
            if (domain == ValueDomain::Timestamp) {
                switch (lossy_policy) {
                    case LossyPolicy::PLA4:
                    case LossyPolicy::PLA8:
                    case LossyPolicy::PLA16:
                    case LossyPolicy::PLA32:
                        return std::make_unique<TimeDeltaManager>();
                    case LossyPolicy::NormalSample:
                        return std::make_unique<TimeDeltaManager>();
                }
                return std::make_unique<TimeDeltaManager>();
            }
            if (domain == ValueDomain::Duration) {
                return std::make_unique<DurationDeltaManager>();
            }
            return std::make_unique<NoneManager>();
        case StoragePolicy::Delta:
            if (domain == ValueDomain::Timestamp) {
                return std::make_unique<TimeDeltaManager>();
            }
            if (domain == ValueDomain::Duration) {
                return std::make_unique<DurationDeltaManager>();
            }
            return std::make_unique<NoneManager>();
    }

    return std::make_unique<NoneManager>();
}

}  // namespace

Manager::~Manager() = default;

}

/** Methods Peratining to NoneManger Class */
namespace pallas {

size_t NoneManager::recommended_capacity(ValueDomain, StoragePolicy, SubArrayPhase) const {
    return DEFAULT_VECTOR_SIZE;
}

AddStatus NoneManager::add(SubArrayBase& subarray, uint64_t val) {
    if (subarray.physical_size >= subarray.allocated_count) {
        return AddStatus::Full;
    }

    subarray.values[subarray.physical_size] = val;
    subarray.value_count++;
    subarray.physical_size++;
    return AddStatus::Ok;
}

uint64_t NoneManager::at(const SubArrayBase& subarray, size_t pos) const {
    if (!subarray.contains(pos)) {
        pallas_error("Wrong index (%lu) compared to starting index (%lu) and size (%lu)\n", pos, subarray.first_index, subarray.value_count);
    }
    return subarray.values[subarray.local_index(pos)];
}

void NoneManager::copy_values(const SubArrayBase& subarray, uint64_t* given_array) const {
    std::memcpy(given_array, subarray.values, subarray.value_count * sizeof(uint64_t));
}

void NoneManager::write_data(SubArrayBase& subarray, FILE* data_file, const ParameterHandler* parameter_handler) {
    if (data_file == nullptr || parameter_handler == nullptr || subarray.values == nullptr) {
        return;
    }

    const long current_offset = std::ftell(data_file);
    if (current_offset >= 0) {
        subarray.file_offset = static_cast<size_t>(current_offset);
    }

    numberPreRawBytes += subarray.size() * sizeof(uint64_t);
    numberRawBytes += subarray.mem_size() * sizeof(uint64_t);
    _pallas_compress_write(subarray.values, subarray.mem_size(), data_file, parameter_handler);
    subarray.free_values();
}

void NoneManager::load_data(SubArrayBase& subarray, FILE* data_file, const ParameterHandler& parameter_handler) {
    if (data_file == nullptr) {
        return;
    }

    delete[] subarray.values;
    subarray.values = _pallas_compress_read(subarray.mem_size(), data_file, parameter_handler);
}

void NoneManager::on_values_freed(SubArrayBase&) {}
} 

/** Methods Pertaining to the DeltaManager (Time and Duration) */
namespace pallas {

size_t TimeDeltaManager::recommended_capacity(ValueDomain, StoragePolicy, SubArrayPhase) const {
    return DEFAULT_VECTOR_SIZE;
}

AddStatus TimeDeltaManager::add(SubArrayBase& subarray, uint64_t val) {
    auto& time_subarray = static_cast<TimeSubArray&>(subarray);
    if (max_payload_bytes == 0) {
        max_payload_bytes = subarray.allocated_count * sizeof(uint64_t);
    }
    if (payload == nullptr) {
        payload = reinterpret_cast<uint8_t*>(subarray.values);
    }

    uint64_t encoded_value = 0;
    uint64_t next_previous_delta = previous_delta;
    if (subarray.value_count == 0) {
        encoded_value = val;
    } else if (subarray.value_count == 1) {
        pallas_assert_inferior_equal(time_subarray.last_timestamp, val);
        const uint64_t current_delta = val - time_subarray.last_timestamp;
        encoded_value = current_delta;
        next_previous_delta = current_delta;
    } else {
        pallas_assert_inferior_equal(time_subarray.last_timestamp, val);
        const uint64_t current_delta = val - time_subarray.last_timestamp;
        const int64_t delta_of_delta =
                static_cast<int64_t>(current_delta) - static_cast<int64_t>(previous_delta);
        encoded_value = zigzag_encode(delta_of_delta);
        next_previous_delta = current_delta;
    }

    uint8_t packet[10];
    uint8_t* packet_out = packet;
    write_varint(encoded_value, packet_out);
    const size_t packet_bytes = static_cast<size_t>(packet_out - packet);
    const size_t next_payload_bytes = payload_bytes + packet_bytes;
    if (next_payload_bytes > max_payload_bytes) {
        return AddStatus::Full;
    }

    std::memcpy(payload + payload_bytes, packet, packet_bytes);
    payload_bytes = next_payload_bytes;
    previous_delta = next_previous_delta;
    subarray.value_count++;
    subarray.physical_size = (payload_bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t);

    const size_t logical_index = subarray.value_count - 1;
    if (logical_index % kCheckpointStride == 0) {
        checkpoints.push_back(Checkpoint{
                logical_index,
                payload_bytes,
                val,
                previous_delta
        });
    }
    return AddStatus::Ok;
}

uint64_t TimeDeltaManager::at(const SubArrayBase& subarray, size_t pos) const {
    if (subarray.phase() == SubArrayPhase::AnalysisRead) {
        if (!subarray.contains(pos)) {
            pallas_error("Wrong index (%lu) compared to starting index (%lu) and size (%lu)\n",
                         pos, subarray.first_index, subarray.value_count);
        }
        return subarray.values[subarray.local_index(pos)];
    }

    if (!subarray.contains(pos)) {
        pallas_error("Wrong index (%lu) compared to starting index (%lu) and size (%lu)\n",
                     pos, subarray.first_index, subarray.value_count);
    }
    if (payload == nullptr || subarray.values == nullptr) {
        pallas_warn("TimeDeltaManager::at cannot decode without payload.\n");
        return 0;
    }

    const size_t target_index = subarray.local_index(pos);
    const uint8_t* begin = payload;
    const uint8_t* end = begin + payload_bytes;
    const Checkpoint* checkpoint = nullptr;
    for (const auto& candidate : checkpoints) {
        if (candidate.logical_index > target_index) {
            break;
        }
        checkpoint = &candidate;
    }

    size_t current_index = 0;
    uint64_t current_value = 0;
    uint64_t current_previous_delta = 0;
    const uint8_t* cursor = begin;

    if (checkpoint != nullptr) {
        if (checkpoint->logical_index == target_index) {
            return checkpoint->value;
        }
        current_index = checkpoint->logical_index;
        current_value = checkpoint->value;
        current_previous_delta = checkpoint->previous_delta;
        cursor = begin + checkpoint->byte_offset;
    } else {
        current_value = read_varint(cursor, end);
        if (target_index == 0) {
            return current_value;
        }
    }

    while (current_index < target_index) {
        if (current_index == 0) {
            current_previous_delta = read_varint(cursor, end);
            current_value += current_previous_delta;
            current_index = 1;
            continue;
        }

        const int64_t delta_of_delta = zigzag_decode(read_varint(cursor, end));
        const uint64_t current_delta =
                static_cast<uint64_t>(static_cast<int64_t>(current_previous_delta) + delta_of_delta);
        current_value += current_delta;
        current_previous_delta = current_delta;
        current_index++;
    }

    return current_value;
}

void TimeDeltaManager::copy_to_array(const SubArrayBase& subarray, uint64_t* given_array) const {
    if (subarray.phase() == SubArrayPhase::AnalysisRead) {
        if (given_array != nullptr && subarray.values != nullptr) {
            std::memcpy(given_array, subarray.values, subarray.size() * sizeof(uint64_t));
        }
        return;
    }

    if (given_array == nullptr || payload == nullptr || subarray.values == nullptr) {
        return;
    }

    const uint8_t* cursor = payload;
    const uint8_t* end = payload + payload_bytes;
    if (subarray.size() == 0) {
        return;
    }

    given_array[0] = read_varint(cursor, end);
    if (subarray.size() >= 2) {
        uint64_t current_previous_delta = read_varint(cursor, end);
        given_array[1] = given_array[0] + current_previous_delta;

        for (size_t logical_index = 2; logical_index < subarray.size(); ++logical_index) {
            const int64_t delta_of_delta = zigzag_decode(read_varint(cursor, end));
            const uint64_t current_delta =
                    static_cast<uint64_t>(static_cast<int64_t>(current_previous_delta) + delta_of_delta);
            given_array[logical_index] = given_array[logical_index - 1] + current_delta;
            current_previous_delta = current_delta;
        }
    }
}

void TimeDeltaManager::write_data(SubArrayBase& subarray, FILE* data_file, const ParameterHandler* parameter_handler) {
    if (data_file == nullptr || parameter_handler == nullptr || subarray.values == nullptr) {
        return;
    }

    const long current_offset = std::ftell(data_file);
    if (current_offset >= 0) {
        subarray.file_offset = static_cast<size_t>(current_offset);
    }

    numberPreRawBytes += subarray.size() * sizeof(uint64_t);
    numberRawBytes += subarray.mem_size() * sizeof(uint64_t);
    _pallas_compress_write(subarray.values, subarray.mem_size(), data_file, parameter_handler);
    subarray.free_values();
}

void TimeDeltaManager::load_data(SubArrayBase& subarray, FILE* data_file, const ParameterHandler& parameter_handler) {
    if (data_file == nullptr) {
        return;
    }

    delete[] subarray.values;
    if (subarray.phase() == SubArrayPhase::AnalysisRead) {
        const size_t packed_word_count = subarray.mem_size();
        uint64_t* packed_values = _pallas_compress_read(packed_word_count, data_file, parameter_handler);
        const size_t logical_value_count = subarray.size();
        uint64_t* decoded_values = new uint64_t[logical_value_count];

        if (logical_value_count > 0) {
            const uint8_t* p = reinterpret_cast<const uint8_t*>(packed_values);
            const uint8_t* end = p + packed_word_count * sizeof(uint64_t);

            decoded_values[0] = read_varint(p, end);
            if (logical_value_count >= 2) {
                uint64_t prev_delta = read_varint(p, end);
                decoded_values[1] = decoded_values[0] + prev_delta;

                for (size_t i = 2; i < logical_value_count; ++i) {
                    const int64_t delta_of_delta = zigzag_decode(read_varint(p, end));
                    const uint64_t current_delta =
                            static_cast<uint64_t>(static_cast<int64_t>(prev_delta) + delta_of_delta);
                    decoded_values[i] = decoded_values[i - 1] + current_delta;
                    prev_delta = current_delta;
                }
            }
        }

        delete[] packed_values;
        subarray.values = decoded_values;
        subarray.allocated_count = logical_value_count;
        subarray.physical_size = logical_value_count;
        payload = nullptr;
        payload_bytes = 0;
        max_payload_bytes = 0;
        previous_delta = 0;
        checkpoints.clear();
        return;
    }

    subarray.values = _pallas_compress_read(subarray.mem_size(), data_file, parameter_handler);
    payload = reinterpret_cast<uint8_t*>(subarray.values);
    if (max_payload_bytes == 0) {
        max_payload_bytes = subarray.allocated_count * sizeof(uint64_t);
    }
}

void TimeDeltaManager::on_values_freed(SubArrayBase&) {
    payload = nullptr;
}

size_t DurationDeltaManager::recommended_capacity(ValueDomain, StoragePolicy, SubArrayPhase) const {
    return DEFAULT_VECTOR_SIZE;
}

AddStatus DurationDeltaManager::add(SubArrayBase& subarray, uint64_t val) {
    if (max_payload_bytes == 0) {
        max_payload_bytes = subarray.allocated_count * sizeof(uint64_t);
    }
    if (payload == nullptr) {
        payload = reinterpret_cast<uint8_t*>(subarray.values);
    }

    uint64_t encoded_value = 0;
    int64_t next_previous_delta = previous_delta;
    if (subarray.value_count == 0) {
        encoded_value = val;
    } else if (subarray.value_count == 1) {
        const int64_t current_delta =
                static_cast<int64_t>(val) - static_cast<int64_t>(last_value);
        encoded_value = zigzag_encode(current_delta);
        next_previous_delta = current_delta;
    } else {
        const int64_t current_delta =
                static_cast<int64_t>(val) - static_cast<int64_t>(last_value);
        const int64_t delta_of_delta = current_delta - previous_delta;
        encoded_value = zigzag_encode(delta_of_delta);
        next_previous_delta = current_delta;
    }

    uint8_t packet[10];
    uint8_t* packet_out = packet;
    write_varint(encoded_value, packet_out);
    const size_t packet_bytes = static_cast<size_t>(packet_out - packet);
    const size_t next_payload_bytes = payload_bytes + packet_bytes;
    if (next_payload_bytes > max_payload_bytes) {
        return AddStatus::Full;
    }

    std::memcpy(payload + payload_bytes, packet, packet_bytes);
    payload_bytes = next_payload_bytes;
    last_value = val;
    previous_delta = next_previous_delta;
    subarray.value_count++;
    subarray.physical_size = (payload_bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t);

    const size_t logical_index = subarray.value_count - 1;
    if (logical_index % kCheckpointStride == 0) {
        checkpoints.push_back(Checkpoint{
                logical_index,
                payload_bytes,
                val,
                previous_delta
        });
    }
    return AddStatus::Ok;
}

uint64_t DurationDeltaManager::at(const SubArrayBase& subarray, size_t pos) const {
    if (subarray.phase() == SubArrayPhase::AnalysisRead) {
        if (!subarray.contains(pos)) {
            pallas_error("Wrong index (%lu) compared to starting index (%lu) and size (%lu)\n",
                         pos, subarray.first_index, subarray.value_count);
        }
        return subarray.values[subarray.local_index(pos)];
    }

    if (!subarray.contains(pos)) {
        pallas_error("Wrong index (%lu) compared to starting index (%lu) and size (%lu)\n",
                     pos, subarray.first_index, subarray.value_count);
    }
    if (payload == nullptr || subarray.values == nullptr) {
        pallas_warn("DurationDeltaManager::at cannot decode without payload.\n");
        return 0;
    }

    const size_t target_index = subarray.local_index(pos);
    const uint8_t* begin = payload;
    const uint8_t* end = begin + payload_bytes;
    const Checkpoint* checkpoint = nullptr;
    for (const auto& candidate : checkpoints) {
        if (candidate.logical_index > target_index) {
            break;
        }
        checkpoint = &candidate;
    }

    size_t current_index = 0;
    uint64_t current_value = 0;
    int64_t current_previous_delta = 0;
    const uint8_t* cursor = begin;

    if (checkpoint != nullptr) {
        if (checkpoint->logical_index == target_index) {
            return checkpoint->value;
        }
        current_index = checkpoint->logical_index;
        current_value = checkpoint->value;
        current_previous_delta = checkpoint->previous_delta;
        cursor = begin + checkpoint->byte_offset;
    } else {
        current_value = read_varint(cursor, end);
        if (target_index == 0) {
            return current_value;
        }
    }

    while (current_index < target_index) {
        if (current_index == 0) {
            current_previous_delta = zigzag_decode(read_varint(cursor, end));
            current_value = static_cast<uint64_t>(static_cast<int64_t>(current_value) + current_previous_delta);
            current_index = 1;
            continue;
        }

        const int64_t delta_of_delta = zigzag_decode(read_varint(cursor, end));
        const int64_t current_delta = current_previous_delta + delta_of_delta;
        current_value = static_cast<uint64_t>(static_cast<int64_t>(current_value) + current_delta);
        current_previous_delta = current_delta;
        current_index++;
    }

    return current_value;
}

void DurationDeltaManager::copy_to_array(const SubArrayBase& subarray, uint64_t* given_array) const {
    if (subarray.phase() == SubArrayPhase::AnalysisRead) {
        if (given_array != nullptr && subarray.values != nullptr) {
            std::memcpy(given_array, subarray.values, subarray.size() * sizeof(uint64_t));
        }
        return;
    }

    if (given_array == nullptr || payload == nullptr || subarray.values == nullptr) {
        return;
    }

    const uint8_t* cursor = payload;
    const uint8_t* end = payload + payload_bytes;
    if (subarray.size() == 0) {
        return;
    }

    given_array[0] = read_varint(cursor, end);
    if (subarray.size() >= 2) {
        int64_t current_previous_delta = zigzag_decode(read_varint(cursor, end));
        given_array[1] = static_cast<uint64_t>(static_cast<int64_t>(given_array[0]) + current_previous_delta);

        for (size_t logical_index = 2; logical_index < subarray.size(); ++logical_index) {
            const int64_t delta_of_delta = zigzag_decode(read_varint(cursor, end));
            const int64_t current_delta = current_previous_delta + delta_of_delta;
            given_array[logical_index] =
                    static_cast<uint64_t>(static_cast<int64_t>(given_array[logical_index - 1]) + current_delta);
            current_previous_delta = current_delta;
        }
    }
}

void DurationDeltaManager::write_data(SubArrayBase& subarray, FILE* data_file, const ParameterHandler* parameter_handler) {
    if (data_file == nullptr || parameter_handler == nullptr || subarray.values == nullptr) {
        return;
    }

    const long current_offset = std::ftell(data_file);
    if (current_offset >= 0) {
        subarray.file_offset = static_cast<size_t>(current_offset);
    }

    numberPreRawBytes += subarray.size() * sizeof(uint64_t);
    numberRawBytes += subarray.mem_size() * sizeof(uint64_t);
    _pallas_compress_write(subarray.values, subarray.mem_size(), data_file, parameter_handler);
    subarray.free_values();
}

void DurationDeltaManager::load_data(SubArrayBase& subarray, FILE* data_file, const ParameterHandler& parameter_handler) {
    if (data_file == nullptr) {
        return;
    }

    delete[] subarray.values;
    if (subarray.phase() == SubArrayPhase::AnalysisRead) {
        const size_t packed_word_count = subarray.mem_size();
        uint64_t* packed_values = _pallas_compress_read(packed_word_count, data_file, parameter_handler);
        const size_t logical_value_count = subarray.size();
        uint64_t* decoded_values = new uint64_t[logical_value_count];

        if (logical_value_count > 0) {
            const uint8_t* p = reinterpret_cast<const uint8_t*>(packed_values);
            const uint8_t* end = p + packed_word_count * sizeof(uint64_t);

            decoded_values[0] = read_varint(p, end);
            if (logical_value_count >= 2) {
                int64_t prev_delta = zigzag_decode(read_varint(p, end));
                decoded_values[1] = static_cast<uint64_t>(static_cast<int64_t>(decoded_values[0]) + prev_delta);

                for (size_t i = 2; i < logical_value_count; ++i) {
                    const int64_t delta_of_delta = zigzag_decode(read_varint(p, end));
                    const int64_t current_delta = prev_delta + delta_of_delta;
                    decoded_values[i] = static_cast<uint64_t>(static_cast<int64_t>(decoded_values[i - 1]) + current_delta);
                    prev_delta = current_delta;
                }
            }
        }

        delete[] packed_values;
        subarray.values = decoded_values;
        subarray.allocated_count = logical_value_count;
        subarray.physical_size = logical_value_count;
        payload = nullptr;
        payload_bytes = 0;
        max_payload_bytes = 0;
        last_value = 0;
        previous_delta = 0;
        checkpoints.clear();
        return;
    }

    subarray.values = _pallas_compress_read(subarray.mem_size(), data_file, parameter_handler);
    payload = reinterpret_cast<uint8_t*>(subarray.values);
    if (max_payload_bytes == 0) {
        max_payload_bytes = subarray.allocated_count * sizeof(uint64_t);
    }
}

void DurationDeltaManager::on_values_freed(SubArrayBase&) {
    payload = nullptr;
}

}

/** Methods Pertaining to the base SubArray Class */
namespace pallas {

SubArrayBase::SubArrayBase(ValueDomain domain,
                           StoragePolicy policy,
                           SubArrayBase* previous,
                           const ParameterHandler* parameter_handler)
    : prev(previous),
      value_domain(domain),
      storage_policy(policy),
      lossy_storage_policy(resolve_lossy_policy(domain, policy, parameter_handler)),
      subarray_phase(SubArrayPhase::RuntimeWrite),
      manager(make_manager(domain, policy, lossy_storage_policy)),
      allocated_count(manager->recommended_capacity(domain, policy, subarray_phase)),
      values(new uint64_t[allocated_count]),
      configuration(parameter_handler) {
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
    return values;
}

void SubArrayBase::free_values() {
    delete[] values;
    values = nullptr;
    if (manager != nullptr) {
        manager->on_values_freed(*this);
    }
}

void SubArrayBase::rebuild_manager() {
    manager = make_manager(value_domain, storage_policy, lossy_storage_policy);
}

uint64_t SubArrayBase::at(size_t pos) const {
    return manager->at(*this, pos);
}

uint64_t SubArrayBase::operator[](size_t pos) const {
    return at(pos);
}

void SubArrayBase::copy_to_array(uint64_t* given_array) const {
    manager->copy_to_array(*this, given_array);
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
    return allocated_count;
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
    return values != nullptr;
}

void SubArrayBase::set_offset(size_t offset) {
    file_offset = offset;
}

void SubArrayBase::load_data(FILE* data_file, const ParameterHandler& parameter_handler) {
    manager->load_data(*this, data_file, parameter_handler);
}

}

/** Methods Peratining to the TimeSubArray Class */
namespace pallas {

TimeSubArray::TimeSubArray(StoragePolicy policy,
                           TimeSubArray* previous,
                           const ParameterHandler* parameter_handler)
    : SubArrayBase(ValueDomain::Timestamp, policy, previous, parameter_handler) {}

AddStatus TimeSubArray::add(uint64_t val) {
    const bool is_first_value = (value_count == 0);
    auto status = manager->add(*this, val);
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
    manager->write_data(*this, file, parameter_handler);
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
                                   const ParameterHandler* parameter_handler)
    : SubArrayBase(ValueDomain::Duration, policy, previous, parameter_handler) {}

AddStatus DurationSubArray::add(uint64_t val) {
    auto status = manager->add(*this, val);
    if (status == AddStatus::Ok) {
        update_statistics(val);
    }
    return status;
}

void DurationSubArray::write_data(FILE* file, const ParameterHandler* parameter_handler) {
    manager->write_data(*this, file, parameter_handler);
}

void DurationSubArray::update_statistics(uint64_t current_value) {
    min_duration = (current_value < min_duration) ? current_value : min_duration;
    max_duration = (current_value > max_duration) ? current_value : max_duration;
    mean_duration += current_value;
}

void DurationSubArray::final_update_mean() {
    if (value_count == 0) {
        return;
    }
    mean_duration /= value_count;
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
