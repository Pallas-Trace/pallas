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

std::unique_ptr<Manager> make_manager(ValueDomain domain, StoragePolicy policy) {
    switch (policy) {
        case StoragePolicy::None:
            return std::make_unique<NoneManager>();
        case StoragePolicy::Lossy:
            if (domain == ValueDomain::Timestamp) {
                return std::make_unique<LinearTimeManager>();
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

void NoneManager::copy_to_array(const SubArrayBase& subarray, uint64_t* given_array) const {
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
    subarray.resident = true;
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
        subarray.resident = true;
        payload = nullptr;
        payload_bytes = 0;
        max_payload_bytes = 0;
        previous_delta = 0;
        checkpoints.clear();
        return;
    }

    subarray.values = _pallas_compress_read(subarray.mem_size(), data_file, parameter_handler);
    subarray.resident = true;
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
        subarray.resident = true;
        payload = nullptr;
        payload_bytes = 0;
        max_payload_bytes = 0;
        last_value = 0;
        previous_delta = 0;
        checkpoints.clear();
        return;
    }

    subarray.values = _pallas_compress_read(subarray.mem_size(), data_file, parameter_handler);
    subarray.resident = true;
    payload = reinterpret_cast<uint8_t*>(subarray.values);
    if (max_payload_bytes == 0) {
        max_payload_bytes = subarray.allocated_count * sizeof(uint64_t);
    }
}

void DurationDeltaManager::on_values_freed(SubArrayBase&) {
    payload = nullptr;
}

size_t LinearTimeManager::recommended_capacity(ValueDomain, StoragePolicy, SubArrayPhase) const {
    return kSeedValueCount;
}

void LinearTimeManager::set_epsilon(uint64_t new_epsilon) {
    epsilon = new_epsilon;
}

bool LinearTimeManager::model_active() const {
    return serialized_mode == SerializedMode::LinearModel;
}

uint64_t LinearTimeManager::predict_value(size_t logical_index) const {
    if (!model_active()) {
        if (logical_index < raw_prefix_count) {
            return raw_prefix_values[logical_index];
        }
        return anchor_value;
    }

    const double prediction = static_cast<double>(anchor_value) + slope * static_cast<double>(logical_index);
    if (prediction <= 0.0) {
        return 0;
    }
    return static_cast<uint64_t>(std::llround(prediction));
}

const LinearTimeManager::OutlierEntry* LinearTimeManager::find_outlier(size_t logical_index) const {
    for (size_t i = 0; i < outlier_count; ++i) {
        if (outliers[i].logical_index == logical_index) {
            return &outliers[i];
        }
    }
    return nullptr;
}

void LinearTimeManager::fit_model(const TimeSubArray& subarray) {
    pallas_assert(subarray.value_count == kSeedValueCount);
    pallas_assert(subarray.values != nullptr);

    long double sum_x = 0.0;
    long double sum_y = 0.0;
    long double sum_xx = 0.0;
    long double sum_xy = 0.0;
    for (size_t i = 0; i < kSeedValueCount; ++i) {
        const long double x = static_cast<long double>(i);
        const long double y = static_cast<long double>(subarray.values[i]);
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }

    const long double n = static_cast<long double>(kSeedValueCount);
    const long double denominator = n * sum_xx - sum_x * sum_x;
    long double fitted_slope = 0.0;
    if (denominator != 0.0) {
        fitted_slope = (n * sum_xy - sum_x * sum_y) / denominator;
    }
    if (fitted_slope < 0.0) {
        fitted_slope = 0.0;
    }

    anchor_value = subarray.values[0];
    slope = static_cast<double>(fitted_slope);
    serialized_mode = SerializedMode::LinearModel;
    raw_prefix_count = 0;
}

void LinearTimeManager::clear_state() {
    serialized_mode = SerializedMode::RawPrefixOnly;
    anchor_value = 0;
    slope = 0.0;
    raw_prefix_count = 0;
    outlier_count = 0;
    raw_prefix_values.fill(0);
    for (auto& outlier : outliers) {
        outlier = OutlierEntry{};
    }
}

AddStatus LinearTimeManager::add(SubArrayBase& subarray, uint64_t val) {
    auto& time_subarray = static_cast<TimeSubArray&>(subarray);
    const size_t logical_index = subarray.value_count;

    if (!model_active()) {
        if (subarray.value_count < kSeedValueCount) {
            if (subarray.physical_size >= subarray.allocated_count) {
                return AddStatus::Full;
            }
            subarray.values[subarray.physical_size] = val;
            raw_prefix_values[subarray.value_count] = val;
            raw_prefix_count = subarray.value_count + 1;
            subarray.value_count++;
            subarray.physical_size++;
            if (subarray.value_count == kSeedValueCount) {
                fit_model(time_subarray);
                delete[] subarray.values;
                subarray.values = nullptr;
                subarray.physical_size = 0;
            }
            return AddStatus::Ok;
        }

        fit_model(time_subarray);
        delete[] subarray.values;
        subarray.values = nullptr;
        subarray.physical_size = 0;
    }

    const uint64_t predicted_value = predict_value(logical_index);
    const uint64_t absolute_error =
            (val >= predicted_value) ? (val - predicted_value) : (predicted_value - val);
    if (absolute_error <= epsilon) {
        subarray.value_count++;
        return AddStatus::Ok;
    }

    if (outlier_count >= kOutlierCapacity) {
        return AddStatus::Outlier;
    }

    outliers[outlier_count++] = OutlierEntry{logical_index, val};
    subarray.value_count++;
    return AddStatus::Ok;
}

uint64_t LinearTimeManager::at(const SubArrayBase& subarray, size_t pos) const {
    if (!subarray.contains(pos)) {
        pallas_error("Wrong index (%lu) compared to starting index (%lu) and size (%lu)\n",
                     pos, subarray.first_index, subarray.value_count);
    }

    const size_t logical_index = subarray.local_index(pos);
    if (!model_active()) {
        if (subarray.values != nullptr) {
            return subarray.values[logical_index];
        }
        return raw_prefix_values[logical_index];
    }

    if (const auto* outlier = find_outlier(logical_index); outlier != nullptr) {
        return outlier->value;
    }
    return predict_value(logical_index);
}

void LinearTimeManager::copy_to_array(const SubArrayBase& subarray, uint64_t* given_array) const {
    if (given_array == nullptr) {
        return;
    }

    for (size_t logical_index = 0; logical_index < subarray.size(); ++logical_index) {
        given_array[logical_index] = at(subarray, subarray.starting_index() + logical_index);
    }
}

void LinearTimeManager::write_data(SubArrayBase& subarray, FILE* data_file, const ParameterHandler* parameter_handler) {
    if (data_file == nullptr || parameter_handler == nullptr || !subarray.has_values()) {
        return;
    }

    const long current_offset = std::ftell(data_file);
    if (current_offset >= 0) {
        subarray.file_offset = static_cast<size_t>(current_offset);
    }

    numberPreRawBytes += subarray.size() * sizeof(uint64_t);

    uint64_t* serialized_words = nullptr;
    size_t serialized_word_count = 0;
    if (!model_active()) {
        serialized_word_count = 2 + subarray.size();
        serialized_words = new uint64_t[serialized_word_count]();
        serialized_words[0] = static_cast<uint64_t>(SerializedMode::RawPrefixOnly);
        serialized_words[1] = subarray.size();
        for (size_t i = 0; i < subarray.size(); ++i) {
            serialized_words[2 + i] = (subarray.values != nullptr) ? subarray.values[i] : raw_prefix_values[i];
        }
    } else {
        serialized_word_count = 4 + 2 * outlier_count;
        serialized_words = new uint64_t[serialized_word_count]();
        serialized_words[0] = static_cast<uint64_t>(SerializedMode::LinearModel);
        serialized_words[1] = anchor_value;
        static_assert(sizeof(double) == sizeof(uint64_t));
        std::memcpy(&serialized_words[2], &slope, sizeof(slope));
        serialized_words[3] = outlier_count;
        for (size_t i = 0; i < outlier_count; ++i) {
            serialized_words[4 + 2 * i] = outliers[i].logical_index;
            serialized_words[5 + 2 * i] = outliers[i].value;
        }
    }

    subarray.physical_size = serialized_word_count;
    numberRawBytes += subarray.mem_size() * sizeof(uint64_t);
    _pallas_compress_write(serialized_words, subarray.mem_size(), data_file, parameter_handler);
    delete[] serialized_words;
    subarray.free_values();
}

void LinearTimeManager::load_data(SubArrayBase& subarray, FILE* data_file, const ParameterHandler& parameter_handler) {
    if (data_file == nullptr) {
        return;
    }

    delete[] subarray.values;
    subarray.values = nullptr;
    clear_state();

    const size_t serialized_word_count = subarray.mem_size();
    uint64_t* serialized_words = _pallas_compress_read(serialized_word_count, data_file, parameter_handler);
    if (serialized_word_count == 0) {
        delete[] serialized_words;
        subarray.resident = true;
        return;
    }

    const auto mode = static_cast<SerializedMode>(serialized_words[0]);
    if (mode == SerializedMode::RawPrefixOnly) {
        raw_prefix_count = serialized_words[1];
        pallas_assert(raw_prefix_count <= kSeedValueCount);
        for (size_t i = 0; i < raw_prefix_count; ++i) {
            raw_prefix_values[i] = serialized_words[2 + i];
        }
        serialized_mode = SerializedMode::RawPrefixOnly;
        if (raw_prefix_count > 0) {
            anchor_value = raw_prefix_values[0];
        }
    } else if (mode == SerializedMode::LinearModel) {
        serialized_mode = SerializedMode::LinearModel;
        anchor_value = serialized_words[1];
        std::memcpy(&slope, &serialized_words[2], sizeof(slope));
        outlier_count = serialized_words[3];
        pallas_assert(outlier_count <= kOutlierCapacity);
        for (size_t i = 0; i < outlier_count; ++i) {
            outliers[i].logical_index = serialized_words[4 + 2 * i];
            outliers[i].value = serialized_words[5 + 2 * i];
        }
    } else {
        delete[] serialized_words;
        pallas_error("Invalid LinearTimeManager serialized mode.\n");
    }

    delete[] serialized_words;
    subarray.resident = true;
}

void LinearTimeManager::on_values_freed(SubArrayBase&) {
    clear_state();
}

}

/** Methods Pertaining to the base SubArray Class */
namespace pallas {

SubArrayBase::SubArrayBase(ValueDomain domain, StoragePolicy policy, SubArrayBase* previous)
    : prev(previous),
      value_domain(domain),
      storage_policy(policy),
      subarray_phase(SubArrayPhase::RuntimeWrite),
      manager(make_manager(domain, policy)),
      allocated_count(manager->recommended_capacity(domain, policy, subarray_phase)),
      values(new uint64_t[allocated_count]),
      resident(true) {
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

uint64_t* SubArrayBase::raw_values() {
    return values;
}

void SubArrayBase::free_values() {
    delete[] values;
    values = nullptr;
    resident = false;
    if (manager != nullptr) {
        manager->on_values_freed(*this);
    }
}

void SubArrayBase::rebuild_manager() {
    manager = make_manager(value_domain, storage_policy);
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
    return resident;
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

TimeSubArray::TimeSubArray(StoragePolicy policy, TimeSubArray* previous, uint64_t linear_epsilon)
    : SubArrayBase(ValueDomain::Timestamp, policy, previous) {
    if (policy == StoragePolicy::Lossy) {
        auto* linear_manager = dynamic_cast<LinearTimeManager*>(manager.get());
        pallas_assert(linear_manager != nullptr);
        linear_manager->set_epsilon(linear_epsilon);
    }
}

AddStatus TimeSubArray::add(uint64_t val) {
    if (value_count == 0) {
        first_timestamp = val;
    }

    auto status = manager->add(*this, val);
    if (status == AddStatus::Ok) {
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

DurationSubArray::DurationSubArray(StoragePolicy policy, DurationSubArray* previous)
    : SubArrayBase(ValueDomain::Duration, policy, previous) {}

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
