/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include <cstring>

#include "pallas/utils/pallas_dbg.h"
#include "pallas/utils/pallas_log.h"
#include "pallas/linked_vector/pallas_linked_vector.h"
#include "pallas/linked_vector/pallas_subarray.h"
#include "pallas/linked_vector/pallas_subarray_delta.h"

extern size_t numberPreRawBytes;
extern size_t numberRawBytes;

// These functions are defined in pallas_storage.cpp
extern void _pallas_compress_write(uint64_t* src, size_t n, FILE* file, const pallas::ParameterHandler* parameter_handler);
extern uint64_t* _pallas_compress_read(size_t n, FILE* file, const pallas::ParameterHandler& parameter_handler);


/** Shared delta-codec helpers used by the packed exact and lossy payload paths. */
/** Map a signed delta to an unsigned integer with small magnitudes near zero. */
[[nodiscard]] static uint64_t zigzag_encode(int64_t x) {
    return (static_cast<uint64_t>(x) << 1) ^ static_cast<uint64_t>(x >> 63);
}

/** Recover the original signed delta from its zigzag-encoded unsigned form. */
[[nodiscard]] static int64_t zigzag_decode(uint64_t x) {
    return static_cast<int64_t>((x >> 1) ^ static_cast<uint64_t>(-static_cast<int64_t>(x & 1)));
}

/** Methods Pertaining to the DeltaManager (Time and Duration) */
namespace pallas {

    size_t DeltaManager::_capacity() const {
        return VECTOR_SIZE_2048;
    }

    AddStatus DeltaManager::add(uint64_t val) {
        return is_time_domain() ? add_time(val) : add_duration(val);
    }

    /**
     * @brief Append one timestamp into the packed delta payload.
     *
     * The timestamp codec uses a small prefix-style scheme:
     * @code
     *   logical value 0 : store absolute timestamp as varint
     *   logical value 1 : store first delta          as varint
     *   logical value 2+: store delta-of-delta       as zigzag(varint)
     * @endcode
     *
     * Each encoded item is first staged in a local `packet[10]` buffer, which is large enough for the worst-case 64-bit 
     * varint, and then copied into the SubArray payload if capacity still permits. Periodic checkpoints record the
     * reconstructed logical value plus the decoder state needed to resume random access without replaying the whole prefix.
     */
    AddStatus DeltaManager::add_time(uint64_t val) {
        auto& time_subarray = static_cast<TimeSubArray&>(parent);
        if (cap_bytes == 0) {
            cap_bytes = parent.capacity() * sizeof(uint64_t);
        }
        if (payload == nullptr) {
            payload = reinterpret_cast<uint8_t*>(parent.buffer);
        }

        uint64_t encoded_value = 0;
        uint64_t next_previous_delta = st.prev.u;
        if (parent.value_count == 0) {
            encoded_value = val;
        } else if (parent.value_count == 1) {
            pallas_assert_inferior_equal(time_subarray.last_timestamp, val);
            const uint64_t current_delta = val - time_subarray.last_timestamp;
            encoded_value = current_delta;
            next_previous_delta = current_delta;
        } else {
            pallas_assert_inferior_equal(time_subarray.last_timestamp, val);
            const uint64_t current_delta = val - time_subarray.last_timestamp;
            const int64_t delta_of_delta =
                    static_cast<int64_t>(current_delta) - static_cast<int64_t>(st.prev.u);
            encoded_value = zigzag_encode(delta_of_delta);
            next_previous_delta = current_delta;
        }

        uint8_t packet[10];
        uint8_t* packet_out = packet;
        write_varint(encoded_value, packet_out);
        const size_t packet_bytes = static_cast<size_t>(packet_out - packet);
        const size_t next_payload_bytes = bytes + packet_bytes;
        if (next_payload_bytes > cap_bytes) {
            return AddStatus::Full;
        }

        std::memcpy(payload + bytes, packet, packet_bytes);
        bytes = next_payload_bytes;
        st.prev.u = next_previous_delta;
        parent.value_count++;
        parent.physical_size = (bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t);
    #ifdef BMARK
        shadow_values.push_back(val);
    #endif

        const size_t logical_index = parent.value_count - 1;
        if (logical_index % kCheckpointStride == 0) {
            Checkpoint checkpoint{};
            checkpoint.idx = logical_index;
            checkpoint.off = bytes;
            checkpoint.val = val;
            checkpoint.prev.u = st.prev.u;
            cps.push_back(checkpoint);
        }
        return AddStatus::Ok;
    }

    /** @brief Signed-duration variant of `add_time()`, using zigzag-coded signed deltas. */
    AddStatus DeltaManager::add_duration(uint64_t val) {
        if (cap_bytes == 0) {
            cap_bytes = parent.capacity() * sizeof(uint64_t);
        }
        if (payload == nullptr) {
            payload = reinterpret_cast<uint8_t*>(parent.buffer);
        }

        uint64_t encoded_value = 0;
        int64_t next_previous_delta = st.prev.i;
        if (parent.value_count == 0) {
            encoded_value = val;
        } else if (parent.value_count == 1) {
            const int64_t current_delta =
                    static_cast<int64_t>(val) - static_cast<int64_t>(st.last);
            encoded_value = zigzag_encode(current_delta);
            next_previous_delta = current_delta;
        } else {
            const int64_t current_delta =
                    static_cast<int64_t>(val) - static_cast<int64_t>(st.last);
            const int64_t delta_of_delta = current_delta - st.prev.i;
            encoded_value = zigzag_encode(delta_of_delta);
            next_previous_delta = current_delta;
        }

        uint8_t packet[10];
        uint8_t* packet_out = packet;
        write_varint(encoded_value, packet_out);
        const size_t packet_bytes = static_cast<size_t>(packet_out - packet);
        const size_t next_payload_bytes = bytes + packet_bytes;
        if (next_payload_bytes > cap_bytes) {
            return AddStatus::Full;
        }

        std::memcpy(payload + bytes, packet, packet_bytes);
        bytes = next_payload_bytes;
        st.last = val;
        st.prev.i = next_previous_delta;
        parent.value_count++;
        parent.physical_size = (bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t);
    #ifdef BMARK
        shadow_values.push_back(val);
    #endif

        const size_t logical_index = parent.value_count - 1;
        if (logical_index % kCheckpointStride == 0) {
            Checkpoint checkpoint{};
            checkpoint.idx = logical_index;
            checkpoint.off = bytes;
            checkpoint.val = val;
            checkpoint.prev.i = st.prev.i;
            cps.push_back(checkpoint);
        }
        return AddStatus::Ok;
    }

    uint64_t DeltaManager::at(size_t pos) const {
        if (parent.phase() == SubArrayPhase::AnalysisRead) {
            if (!parent.contains(pos)) {
                pallas_error("Wrong index (%lu) compared to starting index (%lu) and size (%lu)\n",
                            pos, parent.first_index, parent.value_count);
            }
            return parent.buffer[parent.local_index(pos)];
        }

        if (!parent.contains(pos)) {
            pallas_error("Wrong index (%lu) compared to starting index (%lu) and size (%lu)\n",
                        pos, parent.first_index, parent.value_count);
        }
        if (payload == nullptr || parent.buffer == nullptr) {
            pallas_warn("DeltaManager::at cannot decode without payload.\n");
            return 0;
        }

        return is_time_domain() ? at_time(pos) : at_duration(pos);
    }

    /**
     * @brief Decode one timestamp value from the packed delta payload.
     *
     * Random access does not restart from the beginning of the SubArray unless it has to. The decoder
     * first finds the nearest checkpoint whose logical index is at or before the target, restores the 
     * saved value and previous-delta state, then replays only the remaining suffix packets up to `target_index`.
     */
    uint64_t DeltaManager::at_time(size_t pos) const {
        const size_t target_index = parent.local_index(pos);
        const uint8_t* begin = payload;
        const uint8_t* end = begin + bytes;
        const Checkpoint* checkpoint = nullptr;
        for (const auto& candidate : cps) {
            if (candidate.idx > target_index) {
                break;
            }
            checkpoint = &candidate;
        }

        size_t current_index = 0;
        uint64_t current_value = 0;
        uint64_t current_previous_delta = 0;
        const uint8_t* cursor = begin;

        if (checkpoint != nullptr) {
            if (checkpoint->idx == target_index) {
                return checkpoint->val;
            }
            current_index = checkpoint->idx;
            current_value = checkpoint->val;
            current_previous_delta = checkpoint->prev.u;
            cursor = begin + checkpoint->off;
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

    /** @brief Signed-duration variant of `at_time()`, replaying the signed delta stream from the nearest checkpoint. */
    uint64_t DeltaManager::at_duration(size_t pos) const {
        const size_t target_index = parent.local_index(pos);
        const uint8_t* begin = payload;
        const uint8_t* end = begin + bytes;
        const Checkpoint* checkpoint = nullptr;
        for (const auto& candidate : cps) {
            if (candidate.idx > target_index) {
                break;
            }
            checkpoint = &candidate;
        }

        size_t current_index = 0;
        uint64_t current_value = 0;
        int64_t current_previous_delta = 0;
        const uint8_t* cursor = begin;

        if (checkpoint != nullptr) {
            if (checkpoint->idx == target_index) {
                return checkpoint->val;
            }
            current_index = checkpoint->idx;
            current_value = checkpoint->val;
            current_previous_delta = checkpoint->prev.i;
            cursor = begin + checkpoint->off;
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

    void DeltaManager::copy_to_array(uint64_t* given_array) const {
        if (parent.phase() == SubArrayPhase::AnalysisRead) {
            if (given_array != nullptr && parent.buffer != nullptr) {
                std::memcpy(given_array, parent.buffer, parent.size() * sizeof(uint64_t));
            }
            return;
        }

        if (given_array == nullptr || payload == nullptr || parent.buffer == nullptr) {
            return;
        }

        if (is_time_domain()) {
            copy_time_to_array(given_array);
        } else {
            copy_duration_to_array(given_array);
        }
    }

    void DeltaManager::copy_time_to_array(uint64_t* given_array) const {
        const uint8_t* cursor = payload;
        const uint8_t* end = payload + bytes;
        if (parent.size() == 0) {
            return;
        }

        given_array[0] = read_varint(cursor, end);
        if (parent.size() >= 2) {
            uint64_t current_previous_delta = read_varint(cursor, end);
            given_array[1] = given_array[0] + current_previous_delta;

            for (size_t logical_index = 2; logical_index < parent.size(); ++logical_index) {
                const int64_t delta_of_delta = zigzag_decode(read_varint(cursor, end));
                const uint64_t current_delta =
                        static_cast<uint64_t>(static_cast<int64_t>(current_previous_delta) + delta_of_delta);
                given_array[logical_index] = given_array[logical_index - 1] + current_delta;
                current_previous_delta = current_delta;
            }
        }
    }

    void DeltaManager::copy_duration_to_array(uint64_t* given_array) const {
        const uint8_t* cursor = payload;
        const uint8_t* end = payload + bytes;
        if (parent.size() == 0) {
            return;
        }

        given_array[0] = read_varint(cursor, end);
        if (parent.size() >= 2) {
            int64_t current_previous_delta = zigzag_decode(read_varint(cursor, end));
            given_array[1] = static_cast<uint64_t>(static_cast<int64_t>(given_array[0]) + current_previous_delta);

            for (size_t logical_index = 2; logical_index < parent.size(); ++logical_index) {
                const int64_t delta_of_delta = zigzag_decode(read_varint(cursor, end));
                const int64_t current_delta = current_previous_delta + delta_of_delta;
                given_array[logical_index] =
                        static_cast<uint64_t>(static_cast<int64_t>(given_array[logical_index - 1]) + current_delta);
                current_previous_delta = current_delta;
            }
        }
    }

    void DeltaManager::write_data(FILE* data_file, const ParameterHandler* parameter_handler) {
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
        std::vector<uint64_t> reconstructed_values(parent.size());
        if (parent.size() > 0) {
            copy_to_array(reconstructed_values.data());
            const auto family = (parent.parent_lv != nullptr) ? parent.parent_lv->get_bmark_family()
                                                            : BmarkFamily::Unknown;
            if (shadow_values.size() == parent.size()) {
                record_subarray_error_metrics(
                        family, shadow_values.data(), reconstructed_values.data(), parent.size());
            }
            record_subarray_write_metrics(family, data_file, current_offset, pre_raw_bytes, raw_bytes);
        } else {
            const auto family = (parent.parent_lv != nullptr) ? parent.parent_lv->get_bmark_family()
                                                            : BmarkFamily::Unknown;
            record_subarray_write_metrics(family, data_file, current_offset, pre_raw_bytes, raw_bytes);
        }
    #endif
        parent.free_values();
    }

    void DeltaManager::load_data(FILE* data_file, const ParameterHandler& parameter_handler) {
        if (data_file == nullptr) {
            return;
        }

        delete[] parent.buffer;
        if (is_time_domain()) {
            load_time_data(data_file, parameter_handler);
        } else {
            load_duration_data(data_file, parameter_handler);
        }
    }

    void DeltaManager::load_time_data(FILE* data_file, const ParameterHandler& parameter_handler) {
        if (parent.phase() == SubArrayPhase::AnalysisRead) {
            const size_t packed_word_count = parent.mem_size();
            uint64_t* packed_values = _pallas_compress_read(packed_word_count, data_file, parameter_handler);
            const size_t logical_value_count = parent.size();
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
            parent.buffer = decoded_values;
            parent.physical_size = logical_value_count;
            payload = nullptr;
            bytes = 0;
            cap_bytes = 0;
            st = State{};
            cps.clear();
            return;
        }

        parent.buffer = _pallas_compress_read(parent.mem_size(), data_file, parameter_handler);
        payload = reinterpret_cast<uint8_t*>(parent.buffer);
        if (cap_bytes == 0) {
            cap_bytes = parent.capacity() * sizeof(uint64_t);
        }
        bytes = parent.mem_size() * sizeof(uint64_t);
    }

    void DeltaManager::load_duration_data(FILE* data_file, const ParameterHandler& parameter_handler) {
        if (parent.phase() == SubArrayPhase::AnalysisRead) {
            const size_t packed_word_count = parent.mem_size();
            uint64_t* packed_values = _pallas_compress_read(packed_word_count, data_file, parameter_handler);
            const size_t logical_value_count = parent.size();
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
                        decoded_values[i] =
                                static_cast<uint64_t>(static_cast<int64_t>(decoded_values[i - 1]) + current_delta);
                        prev_delta = current_delta;
                    }
                }
            }

            delete[] packed_values;
            parent.buffer = decoded_values;
            parent.physical_size = logical_value_count;
            payload = nullptr;
            bytes = 0;
            cap_bytes = 0;
            st = State{};
            cps.clear();
            return;
        }

        parent.buffer = _pallas_compress_read(parent.mem_size(), data_file, parameter_handler);
        payload = reinterpret_cast<uint8_t*>(parent.buffer);
        if (cap_bytes == 0) {
            cap_bytes = parent.capacity() * sizeof(uint64_t);
        }
        bytes = parent.mem_size() * sizeof(uint64_t);
    }

    void DeltaManager::on_values_freed() {
        payload = nullptr;
        bytes = 0;
    #ifdef BMARK
        shadow_values.clear();
    #endif
    }

}