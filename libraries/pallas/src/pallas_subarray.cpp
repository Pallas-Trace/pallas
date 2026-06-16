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

/** Methods Pertaining to LinearTimeManager */
namespace pallas {

size_t LinearTimeManager::representative_capacity() const {
    return 2 + 2 * kOutlierCapacity;
}

size_t LinearTimeManager::recommended_capacity(ValueDomain, StoragePolicy, SubArrayPhase) const {
    return representative_capacity();
}

void LinearTimeManager::set_epsilon(uint64_t new_epsilon) {
    epsilon = new_epsilon;
}

bool LinearTimeManager::model_active() const {
    return prediction_model_active;
}

size_t LinearTimeManager::outlier_count(const SubArrayBase& subarray) const {
    if (!model_active() || subarray.physical_size < 2) {
        return 0;
    }
    return (subarray.physical_size - 2) / 2;
}

uint64_t LinearTimeManager::prediction_from_fit(const uint64_t* fit_values, size_t fit_count, size_t logical_index) const {
    if (fit_values == nullptr || fit_count == 0) {
        return 0;
    }
    if (fit_count == 1) {
        return fit_values[0];
    }

    long double sum_x = 0.0;
    long double sum_y = 0.0;
    long double sum_xx = 0.0;
    long double sum_xy = 0.0;
    for (size_t i = 0; i < fit_count; ++i) {
        const long double x = static_cast<long double>(i);
        const long double y = static_cast<long double>(fit_values[i]);
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }

    const long double n = static_cast<long double>(fit_count);
    const long double denominator = n * sum_xx - sum_x * sum_x;
    long double fitted_slope = 0.0;
    if (denominator != 0.0) {
        fitted_slope = (n * sum_xy - sum_x * sum_y) / denominator;
    }
    if (fitted_slope < 0.0) {
        fitted_slope = 0.0;
    }

    const double prediction = static_cast<double>(fit_values[0]) + static_cast<double>(fitted_slope) * static_cast<double>(logical_index);
    if (prediction <= 0.0) {
        return 0;
    }
    return static_cast<uint64_t>(std::llround(prediction));
}

uint64_t LinearTimeManager::predict_value(size_t logical_index) const {
    const double prediction = static_cast<double>(anchor_value) + slope * static_cast<double>(logical_index);
    if (prediction <= 0.0) {
        return 0;
    }
    return static_cast<uint64_t>(std::llround(prediction));
}

bool LinearTimeManager::find_outlier(const SubArrayBase& subarray, size_t logical_index, uint64_t& value) const {
    if (!model_active() || subarray.values == nullptr) {
        return false;
    }

    const size_t count = outlier_count(subarray);
    for (size_t i = 0; i < count; ++i) {
        const size_t offset = 2 + 2 * i;
        if (subarray.values[offset] == logical_index) {
            value = subarray.values[offset + 1];
            return true;
        }
    }
    return false;
}

void LinearTimeManager::fit_model(const TimeSubArray& subarray, size_t fit_count) {
    pallas_assert(subarray.values != nullptr);
    pallas_assert(fit_count > 0);

    if (fit_count == 1) {
        anchor_value = subarray.values[0];
        slope = 0.0;
        return;
    }

    long double sum_x = 0.0;
    long double sum_y = 0.0;
    long double sum_xx = 0.0;
    long double sum_xy = 0.0;
    for (size_t i = 0; i < fit_count; ++i) {
        const long double x = static_cast<long double>(i);
        const long double y = static_cast<long double>(subarray.values[i]);
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }

    const long double n = static_cast<long double>(fit_count);
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
}

void LinearTimeManager::sync_model_to_values(SubArrayBase& subarray) const {
    if (subarray.values == nullptr) {
        return;
    }

    static_assert(sizeof(double) == sizeof(uint64_t));
    std::memcpy(&subarray.values[0], &slope, sizeof(slope));
    subarray.values[1] = anchor_value;
}

void LinearTimeManager::activate_prediction_model(TimeSubArray& subarray, size_t fit_count) {
    fit_model(subarray, fit_count);
    prediction_model_active = true;
    sync_model_to_values(subarray);
    subarray.physical_size = 2;
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
            subarray.value_count++;
            subarray.physical_size++;
            if (subarray.value_count == kSeedValueCount) {
                activate_prediction_model(time_subarray, kSeedValueCount);
            }
            return AddStatus::Ok;
        }

        activate_prediction_model(time_subarray, std::min(subarray.value_count, kSeedValueCount));
    }

    const uint64_t predicted_value = predict_value(logical_index);
    const uint64_t absolute_error =
            (val >= predicted_value) ? (val - predicted_value) : (predicted_value - val);
    if (absolute_error <= epsilon) {
        subarray.value_count++;
        return AddStatus::Ok;
    }

    const size_t current_outlier_count = outlier_count(subarray);
    if (current_outlier_count >= kOutlierCapacity || subarray.physical_size + 2 > subarray.allocated_count) {
        return AddStatus::Outlier;
    }

    subarray.values[subarray.physical_size] = logical_index;
    subarray.values[subarray.physical_size + 1] = val;
    subarray.physical_size += 2;
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
        return prediction_from_fit(subarray.values, subarray.value_count, logical_index);
    }

    uint64_t outlier_value = 0;
    if (find_outlier(subarray, logical_index, outlier_value)) {
        return outlier_value;
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
    if (data_file == nullptr || parameter_handler == nullptr || subarray.values == nullptr) {
        return;
    }

    const long current_offset = std::ftell(data_file);
    if (current_offset >= 0) {
        subarray.file_offset = static_cast<size_t>(current_offset);
    }

    numberPreRawBytes += subarray.size() * sizeof(uint64_t);
    if (!model_active() && subarray.value_count > 0) {
        activate_prediction_model(static_cast<TimeSubArray&>(subarray),
                                  std::min(subarray.value_count, kSeedValueCount));
    }

    numberRawBytes += subarray.mem_size() * sizeof(uint64_t);
    _pallas_compress_write(subarray.values, subarray.mem_size(), data_file, parameter_handler);
    subarray.free_values();
}

// Since double bytes are memcpy, wrapped into a helper

void LinearTimeManager::refresh_model_from_values(const SubArrayBase& subarray) {
    if (subarray.values == nullptr || subarray.physical_size < 2) {
        anchor_value = 0;
        slope = 0.0;
        return;
    }

    static_assert(sizeof(double) == sizeof(uint64_t));
    std::memcpy(&slope, &subarray.values[0], sizeof(slope));
    anchor_value = subarray.values[1];
}

void LinearTimeManager::load_data(SubArrayBase& subarray, FILE* data_file, const ParameterHandler& parameter_handler) {
    if (data_file == nullptr) {
        return;
    }

    delete[] subarray.values;
    clear_state();
    subarray.values = _pallas_compress_read(subarray.mem_size(), data_file, parameter_handler);
    
    prediction_model_active = (subarray.value_count > 0 && subarray.physical_size >= 2);
    if (prediction_model_active) {
        refresh_model_from_values(subarray);
    }
}

// Clear State clears the anchor and slope value, called before loading data and upon freeing of data 

void LinearTimeManager::clear_state() {
    prediction_model_active = false;
    anchor_value = 0;
    slope = 0.0;
}

void LinearTimeManager::on_values_freed(SubArrayBase&) {
    clear_state();
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
      subarray_phase(SubArrayPhase::RuntimeWrite),
      manager(make_manager(domain, policy)),
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

uint64_t* SubArrayBase::raw_values() {
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
    : SubArrayBase(ValueDomain::Timestamp, policy, previous, parameter_handler) {
    if (policy == StoragePolicy::Lossy) {
        auto* linear_manager = dynamic_cast<LinearTimeManager*>(manager.get());
        pallas_assert(linear_manager != nullptr);
        if (configuration != nullptr) {
            linear_manager->set_epsilon(configuration->getTimeLinearEpsilon());
        }
    }
}

AddStatus TimeSubArray::add(uint64_t val) {
    const bool is_first_value = (value_count == 0);
    auto status = manager->add(*this, val);
    if (status == AddStatus::Ok) {
        if (is_first_value) {
            first_timestamp = val;
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
