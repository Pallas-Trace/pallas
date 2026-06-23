/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>

#include "pallas/utils/pallas_dbg.h"
#include "pallas/utils/pallas_log.h"
#include "pallas/utils/pallas_lv.h"
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

uint8_t SubArrayBase::pack_subarray_flags() const {
    const auto storage_bits = static_cast<uint8_t>(storage_policy) & kStoragePolicyMask;
    const auto lossy_bits = static_cast<uint8_t>(lossy_storage_policy) << 2;
    return static_cast<uint8_t>(storage_bits | lossy_bits);
}

void SubArrayBase::unpack_subarray_flags(uint8_t encoded_policy) {
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
                    case LossyPolicy::NormalSample:
                        return std::make_unique<DeltaManager>(parent, domain);
                }
                return std::make_unique<DeltaManager>(parent, domain);
            }
            if (domain == ValueDomain::Duration) {
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

    numberPreRawBytes += parent.size() * sizeof(uint64_t);
    numberRawBytes += parent.mem_size() * sizeof(uint64_t);
    _pallas_compress_write(parent.buffer, parent.mem_size(), data_file, parameter_handler);
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

/** Methods Pertaining to the DeltaManager (Time and Duration) */
namespace pallas {

size_t DeltaManager::_capacity() const {
    return DEFAULT_VECTOR_SIZE;
}

AddStatus DeltaManager::add(uint64_t val) {
    return is_time_domain() ? add_time(val) : add_duration(val);
}

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

    numberPreRawBytes += parent.size() * sizeof(uint64_t);
    numberRawBytes += parent.mem_size() * sizeof(uint64_t);
    _pallas_compress_write(parent.buffer, parent.mem_size(), data_file, parameter_handler);
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
}

}

/** Methods Pertaining to the PLAManager */
namespace pallas {

namespace {

size_t pla_index_bytes(uint8_t anchor_count) {
    return (static_cast<size_t>(anchor_count) * 10 + 7) / 8;
}

size_t pla_payload_bytes(uint8_t anchor_count) {
    return sizeof(anchor_count) +
           static_cast<size_t>(anchor_count) * sizeof(uint64_t) +
           static_cast<size_t>(anchor_count) * sizeof(int32_t) +
           pla_index_bytes(anchor_count);
}

void pack_10bit_indices(const PLAAnchor* anchors, uint8_t anchor_count, uint8_t* out) {
    size_t bit_offset = 0;
    for (uint8_t i = 0; i < anchor_count; ++i) {
        const uint16_t idx = anchors[i].idx;
        for (size_t bit = 0; bit < 10; ++bit) {
            if ((idx >> bit) & 1U) {
                out[(bit_offset + bit) / 8] |= static_cast<uint8_t>(1U << ((bit_offset + bit) % 8));
            }
        }
        bit_offset += 10;
    }
}

void unpack_10bit_indices(PLAAnchor* anchors, uint8_t anchor_count, const uint8_t* in) {
    size_t bit_offset = 0;
    for (uint8_t i = 0; i < anchor_count; ++i) {
        uint16_t idx = 0;
        for (size_t bit = 0; bit < 10; ++bit) {
            const uint8_t byte = in[(bit_offset + bit) / 8];
            if ((byte >> ((bit_offset + bit) % 8)) & 1U) {
                idx |= static_cast<uint16_t>(1U << bit);
            }
        }
        anchors[i].idx = idx;
        bit_offset += 10;
    }
}

}  // namespace

size_t PLAManager::_capacity() const {
    return kPLABlockSize;
}

void PLAManager::ensure_staging() {
    if (parent.parent_lv == nullptr) {
        pallas_error("PLAManager requires a parent LVBase for PLA scratch staging.\n");
    }
    parent.parent_lv->ensure_hbuffer(GammaBlockStats::helper_buffer_bytes());
    stats = GammaBlockStats::bind(parent.parent_lv->helper_buffer());
}

void PLAManager::on_subarray_initialized() {
    if (parent.parent_lv == nullptr) {
        return;
    }
    ensure_staging();
}

void PLAManager::clear_state() {
    compact_ready = false;
    anchor_count = 0;
    stats = GammaBlockStats{};
}

void PLAManager::write_packed_payload() {
    const size_t payload_bytes = pla_payload_bytes(anchor_count);
    const size_t payload_words = (payload_bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t);
    auto* packed_words = new uint64_t[payload_words]();
    auto* out = reinterpret_cast<uint8_t*>(packed_words);

    *out++ = anchor_count;

    for (uint8_t i = 0; i < anchor_count; ++i) {
        std::memcpy(out, &anchor_storage[i].val, sizeof(anchor_storage[i].val));
        out += sizeof(anchor_storage[i].val);
    }
    for (uint8_t i = 0; i < anchor_count; ++i) {
        std::memcpy(out, &anchor_storage[i].dprev, sizeof(anchor_storage[i].dprev));
        out += sizeof(anchor_storage[i].dprev);
    }

    const size_t index_bytes = pla_index_bytes(anchor_count);
    std::memset(out, 0, index_bytes);
    pack_10bit_indices(anchor_storage, anchor_count, out);

    delete[] parent.buffer;
    parent.buffer = packed_words;
    parent.physical_size = payload_words;
}

void PLAManager::load_packed_payload() {
    clear_state();
    auto* in = reinterpret_cast<const uint8_t*>(parent.buffer);
    anchor_count = *in++;

    for (uint8_t i = 0; i < anchor_count; ++i) {
        std::memcpy(&anchor_storage[i].val, in, sizeof(anchor_storage[i].val));
        in += sizeof(anchor_storage[i].val);
    }
    for (uint8_t i = 0; i < anchor_count; ++i) {
        std::memcpy(&anchor_storage[i].dprev, in, sizeof(anchor_storage[i].dprev));
        in += sizeof(anchor_storage[i].dprev);
    }
    unpack_10bit_indices(anchor_storage, anchor_count, in);
    compact_ready = true;
}

void PLAManager::finalize_block() {
    if (compact_ready) {
        return;
    }
    if (parent.value_count == 0) {
        clear_state();
        compact_ready = true;
        write_packed_payload();
        return;
    }
    if (parent.value_count < static_cast<size_t>(k_max) + 2) {
        anchor_count = static_cast<uint8_t>(
                build_all_interior_anchor_block(parent.buffer,
                                                parent.value_count,
                                                anchor_storage,
                                                kPLAMaxAnchors));
        compact_ready = true;
        write_packed_payload();
        return;
    }

    anchor_count = static_cast<uint8_t>(
            (k_max == 4)
                    ? build_pla4_alpha_block(parent.buffer, parent.value_count, stats, anchor_storage, k_max)
                    : build_gamma_anchor_block(parent.buffer, parent.value_count, stats, anchor_storage, k_max));
    compact_ready = true;
    write_packed_payload();
}

AddStatus PLAManager::add(uint64_t val) {
    if (compact_ready) {
        return AddStatus::Full;
    }
    if (parent.value_count >= kPLABlockSize) {
        return AddStatus::Full;
    }

    parent.buffer[parent.value_count] = val;
    parent.value_count++;
    parent.physical_size = parent.value_count;
    if (parent.value_count == kPLABlockSize) {
        finalize_block();
    }
    return AddStatus::Ok;
}

uint64_t PLAManager::interpolate_value(const TimeSubArray& subarray, size_t logical_index) const {
    if (!compact_ready) {
        return 0;
    }
    const size_t size = subarray.size();
    if (size == 0) {
        return 0;
    }
    if (logical_index == 0) {
        // First value
        return subarray.first_value();
    }
    if (logical_index + 1 >= size) {
        // Last value
        return subarray.last_value();
    }
    if (anchor_count == 0) {
        // Worse case fallback
        const size_t span = size - 1;
        if (span == 0) {
            return subarray.first_value();
        }
        const int64_t delta = static_cast<int64_t>(subarray.last_value()) - static_cast<int64_t>(subarray.first_value());
        return static_cast<uint64_t>(static_cast<int64_t>(subarray.first_value()) +
                                     delta * static_cast<int64_t>(logical_index) / static_cast<int64_t>(span));
    }

    size_t left = 0;
    size_t right = anchor_count; // Binary Search on Anchors
    while (left < right) {
        const size_t mid = left + (right - left) / 2;
        if (anchor_storage[mid].idx <= logical_index) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    size_t segment_begin = 0;
    uint64_t begin_value = subarray.first_value();
    size_t segment_end = size - 1;
    uint64_t end_value = subarray.last_value();

    if (left == 0) {
        segment_end = anchor_storage[0].idx - 1;
        end_value = static_cast<uint64_t>(static_cast<int64_t>(anchor_storage[0].val) -
                                          static_cast<int64_t>(anchor_storage[0].dprev));
    } else {
        const auto& current = anchor_storage[left - 1];
        segment_begin = current.idx;
        begin_value = current.val;
        if (left < anchor_count) {
            const auto& next = anchor_storage[left];
            segment_end = next.idx - 1;
            end_value = static_cast<uint64_t>(static_cast<int64_t>(next.val) -
                                              static_cast<int64_t>(next.dprev));
        }
    }

    if (segment_end <= segment_begin) {
        return begin_value;
    }

    const int64_t delta = static_cast<int64_t>(end_value) - static_cast<int64_t>(begin_value);
    const int64_t offset = static_cast<int64_t>(logical_index - segment_begin);
    const int64_t span = static_cast<int64_t>(segment_end - segment_begin);
    return static_cast<uint64_t>(static_cast<int64_t>(begin_value) + delta * offset / span);
}

uint64_t PLAManager::at(size_t pos) const {
    if (!parent.contains(pos)) {
        pallas_error("Wrong index (%lu) compared to starting index (%lu) and size (%lu)\n",
                     pos, parent.first_index, parent.value_count);
    }
    const size_t local = parent.local_index(pos);
    if (!compact_ready) {
        if (parent.buffer == nullptr) {
            pallas_error("PLAManager missing staging buffer for runtime access.\n");
        }
        return parent.buffer[local];
    }
    return interpolate_value(static_cast<const TimeSubArray&>(parent), local);
}

void PLAManager::copy_to_array(uint64_t* given_array) const {
    if (given_array == nullptr) {
        return;
    }
    if (!compact_ready) {
        if (parent.buffer != nullptr) {
            std::memcpy(given_array, parent.buffer, parent.size() * sizeof(uint64_t));
        }
        return;
    }
    const auto& time_subarray = static_cast<const TimeSubArray&>(parent);
    for (size_t i = 0; i < parent.size(); ++i) {
        given_array[i] = interpolate_value(time_subarray, i);
    }
}

void PLAManager::write_data(FILE* data_file, const ParameterHandler* parameter_handler) {
    if (data_file == nullptr || parameter_handler == nullptr) {
        return;
    }
    if (!compact_ready) {
        finalize_block();
    }
    if (parent.buffer == nullptr) {
        return;
    }

    const long current_offset = std::ftell(data_file);
    if (current_offset >= 0) {
        parent.file_offset = static_cast<size_t>(current_offset);
    }

    numberPreRawBytes += parent.size() * sizeof(uint64_t);
    numberRawBytes += parent.mem_size() * sizeof(uint64_t);
    _pallas_compress_write(parent.buffer, parent.mem_size(), data_file, parameter_handler);
    parent.free_values();
}

void PLAManager::load_data(FILE* data_file, const ParameterHandler& parameter_handler) {
    if (data_file == nullptr) {
        return;
    }
    delete[] parent.buffer;
    parent.buffer = _pallas_compress_read(parent.mem_size(), data_file, parameter_handler);
    load_packed_payload();
}

void PLAManager::on_values_freed() {
    clear_state();
}

}

/** Methods Pertaining to the base SubArray Class */
namespace pallas {

SubArrayBase::SubArrayBase(ValueDomain domain,
                           StoragePolicy policy,
                           SubArrayBase* previous,
                           const ParameterHandler* parameter_handler,
                           LVBase* parent)
    : prev(previous),
      value_domain(domain),
      storage_policy(policy),
      lossy_storage_policy(resolve_lossy_policy(domain, policy, parameter_handler)),
      subarray_phase(SubArrayPhase::RuntimeWrite),
      manager(nullptr),
      parent_lv(parent) {
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

void SubArrayBase::load_data(FILE* data_file, const ParameterHandler& parameter_handler) {
    manager->load_data(data_file, parameter_handler);
}

}

/** Methods Peratining to the TimeSubArray Class */
namespace pallas {

TimeSubArray::TimeSubArray(StoragePolicy policy,
                           TimeSubArray* previous,
                           const ParameterHandler* parameter_handler,
                           LVBase* parent)
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
                                   LVBase* parent)
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
