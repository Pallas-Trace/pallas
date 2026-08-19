/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include <cstring>
#include <cmath>
#include <iostream>

#include "pallas/utils/pallas_dbg.h"
#include "pallas/utils/pallas_log.h"
#include "pallas/linked_vector/pallas_linked_vector.h"
#include "pallas/linked_vector/pallas_subarray.h"
#include "pallas/linked_vector/pallas_subarray_pla.h"

extern size_t numberPreRawBytes;
extern size_t numberRawBytes;

// These functions are defined in pallas_storage.cpp
extern void _pallas_compress_write(uint64_t* src, size_t n, FILE* file, const pallas::ParameterHandler* parameter_handler);
extern uint64_t* _pallas_compress_read(size_t n, FILE* file, const pallas::ParameterHandler& parameter_handler);


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
            pallas_error("PLAManager requires a parent LinkedVectorBase for PLA scratch staging.\n");
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

    /**
     * @brief Pack the compact PLA anchor representation back into the SubArray buffer.
     *
     * The packed layout is:
     * @code
     *   [anchor_count]
     *   [anchor values...]
     *   [anchor dprev values...]
     *   [10-bit packed anchor indices...]
     * @endcode
     *
     * The 10-bit index packing is the non-obvious part here: anchor positions are
     * dense enough to fit in 10 bits for one PLA block, so `pack_10bit_indices()`
     * reduces the metadata footprint without changing reconstruction semantics.
     */
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

    /**
     * @brief Perform the delayed PLA compaction step for one full timestamp block.
     *
     * This manager is only pseudo-online: `add()` stores raw values until the SubArray-sized block is available, then
     * this routine selects anchors and rewrites the raw block into the compact PLA form. Small blocks fall back to a
     * simple all-interior-anchor path, while larger blocks use the specialised PLA-4 or gamma-anchor builders 
     * before serialising the compact payload.
     */
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
    #ifdef BMARK
            std::vector<uint64_t> reconstructed_values(parent.size());
            copy_to_array(reconstructed_values.data());
            const auto family =
                    (parent.parent_lv != nullptr) ? parent.parent_lv->get_bmark_family() : BmarkFamily::Unknown;
            record_subarray_error_metrics(family, parent.buffer, reconstructed_values.data(), parent.size());
    #endif
            write_packed_payload();
            return;
        }

        anchor_count = static_cast<uint8_t>(
                (k_max == 4)
                        ? build_pla4_alpha_block(parent.buffer, parent.value_count, stats, anchor_storage, k_max)
                        : build_gamma_anchor_block(parent.buffer, parent.value_count, stats, anchor_storage, k_max));
        compact_ready = true;
    #ifdef BMARK
        std::vector<uint64_t> reconstructed_values(parent.size());
        copy_to_array(reconstructed_values.data());
        const auto family =
                (parent.parent_lv != nullptr) ? parent.parent_lv->get_bmark_family() : BmarkFamily::Unknown;
        record_subarray_error_metrics(family, parent.buffer, reconstructed_values.data(), parent.size());
    #endif
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

        std::cout<<"Write a PLA chunk\n";
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
        record_subarray_write_metrics(family, data_file, current_offset, pre_raw_bytes, raw_bytes);
    #endif
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

/** Methods Pertaining to the DurationSpikeManager */
namespace pallas {

    namespace {

        size_t duration_spike_varint_bytes(uint64_t value) {
            size_t bytes = 1;
            while (value >= 0x80) {
                value >>= 7;
                ++bytes;
            }
            return bytes;
        }

        uint64_t splitmix64(uint64_t value) {
            value += 0x9e3779b97f4a7c15ULL;
            value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
            value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
            return value ^ (value >> 31);
        }

        double uniform_unit_from_u64(uint64_t value) {
            constexpr double kScale = 1.0 / static_cast<double>(1ULL << 53);
            return static_cast<double>((value >> 11) & ((1ULL << 53) - 1)) * kScale;
        }

        struct DurationSpikeCandidate {
            uint16_t idx = 0;
            int64_t residual = 0;
        };

        double median_from_sorted(double* values, size_t count) {
            if (count == 0) {
                return 0.0;
            }
            if ((count & 1U) != 0U) {
                return values[count / 2];
            }
            return 0.5 * (values[count / 2 - 1] + values[count / 2]);
        }

        double median_from_u64(const uint64_t* values, size_t count) {
            if (count == 0) {
                return 0.0;
            }

            std::array<double, VECTOR_SIZE_2048> sorted_values{};
            for (size_t idx = 0; idx < count; ++idx) {
                sorted_values[idx] = static_cast<double>(values[idx]);
            }
            std::sort(sorted_values.begin(), sorted_values.begin() + static_cast<std::ptrdiff_t>(count));
            return median_from_sorted(sorted_values.data(), count);
        }

        double median_from_double_buffer(const double* values, size_t count) {
            if (count == 0) {
                return 0.0;
            }

            std::array<double, VECTOR_SIZE_2048> sorted_values{};
            for (size_t idx = 0; idx < count; ++idx) {
                sorted_values[idx] = values[idx];
            }
            std::sort(sorted_values.begin(), sorted_values.begin() + static_cast<std::ptrdiff_t>(count));
            return median_from_sorted(sorted_values.data(), count);
        }

    }

    size_t DurationSpikeManager::_capacity() const {
        return VECTOR_SIZE_2048;
    }

    void DurationSpikeManager::on_subarray_initialized() {
        clear_state();
    }

    void DurationSpikeManager::clear_state() {
        compact_ready = false;
        packed_payload_ready = false;
        baseline_mean = 0;
        baseline_stddev = 0;
        exact_count = 0;
        group_count = 0;
        exact_spikes = {};
        spike_groups = {};
    }

    size_t DurationSpikeManager::packed_payload_bytes() const {
        size_t payload_bytes =
                sizeof(baseline_mean) + sizeof(baseline_stddev) + sizeof(exact_count) + sizeof(group_count);

        payload_bytes += static_cast<size_t>(exact_count) * (sizeof(uint16_t) + sizeof(uint64_t));
        for (uint8_t group_idx = 0; group_idx < group_count; ++group_idx) {
            const auto& group = spike_groups[group_idx];
            payload_bytes += sizeof(group.value) + sizeof(group.index_count);

            uint16_t previous_index = 0;
            for (uint8_t index_idx = 0; index_idx < group.index_count; ++index_idx) {
                const uint16_t current_index = group.indices[index_idx];
                const uint16_t index_delta =
                        (index_idx == 0) ? current_index : static_cast<uint16_t>(current_index - previous_index);
                payload_bytes += duration_spike_varint_bytes(index_delta);
                previous_index = current_index;
            }
        }
        return payload_bytes;
    }

    /**
     * @brief Pack the duration baseline-and-spikes model into the SubArray buffer.
     *
     * The compact layout is:
     * @code
     *   [baseline_mean][baseline_stddev][exact_count][group_count]
     *   [exact spike entries: (idx, value)...]
     *   [group entries: value, count, delta-coded indices...]
     * @endcode
     *
     * Group member indices are stored as varint-coded deltas inside each group so
     * that repeated spike locations cost much less than storing every logical value
     * exactly.
     */
    void DurationSpikeManager::write_packed_payload() {
        const size_t payload_bytes = packed_payload_bytes();
        const size_t payload_words = (payload_bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t);
        auto* packed_words = new uint64_t[payload_words]();
        auto* out = reinterpret_cast<uint8_t*>(packed_words);

        std::memcpy(out, &baseline_mean, sizeof(baseline_mean));
        out += sizeof(baseline_mean);
        std::memcpy(out, &baseline_stddev, sizeof(baseline_stddev));
        out += sizeof(baseline_stddev);
        *out++ = exact_count;
        *out++ = group_count;

        for (uint8_t exact_idx = 0; exact_idx < exact_count; ++exact_idx) {
            std::memcpy(out, &exact_spikes[exact_idx].idx, sizeof(exact_spikes[exact_idx].idx));
            out += sizeof(exact_spikes[exact_idx].idx);
            std::memcpy(out, &exact_spikes[exact_idx].value, sizeof(exact_spikes[exact_idx].value));
            out += sizeof(exact_spikes[exact_idx].value);
        }

        for (uint8_t group_idx = 0; group_idx < group_count; ++group_idx) {
            const auto& group = spike_groups[group_idx];
            std::memcpy(out, &group.value, sizeof(group.value));
            out += sizeof(group.value);
            *out++ = group.index_count;

            uint16_t previous_index = 0;
            for (uint8_t index_idx = 0; index_idx < group.index_count; ++index_idx) {
                const uint16_t current_index = group.indices[index_idx];
                const uint16_t index_delta =
                        (index_idx == 0) ? current_index : static_cast<uint16_t>(current_index - previous_index);
                write_varint(index_delta, out);
                previous_index = current_index;
            }
        }

        delete[] parent.buffer;
        parent.buffer = packed_words;
        parent.physical_size = payload_words;
        packed_payload_ready = true;
        compact_ready = true;
    }

    void DurationSpikeManager::load_packed_payload() {
        clear_state();

        const auto* in = reinterpret_cast<const uint8_t*>(parent.buffer);
        const auto* end = in + parent.mem_size() * sizeof(uint64_t);
        std::memcpy(&baseline_mean, in, sizeof(baseline_mean));
        in += sizeof(baseline_mean);
        std::memcpy(&baseline_stddev, in, sizeof(baseline_stddev));
        in += sizeof(baseline_stddev);
        exact_count = *in++;
        group_count = *in++;

        for (uint8_t exact_idx = 0; exact_idx < exact_count; ++exact_idx) {
            std::memcpy(&exact_spikes[exact_idx].idx, in, sizeof(exact_spikes[exact_idx].idx));
            in += sizeof(exact_spikes[exact_idx].idx);
            std::memcpy(&exact_spikes[exact_idx].value, in, sizeof(exact_spikes[exact_idx].value));
            in += sizeof(exact_spikes[exact_idx].value);
        }

        for (uint8_t group_idx = 0; group_idx < group_count; ++group_idx) {
            auto& group = spike_groups[group_idx];
            std::memcpy(&group.value, in, sizeof(group.value));
            in += sizeof(group.value);
            group.index_count = *in++;

            uint16_t previous_index = 0;
            for (uint8_t index_idx = 0; index_idx < group.index_count; ++index_idx) {
                const auto index_delta = static_cast<uint16_t>(read_varint(in, end));
                group.indices[index_idx] =
                        (index_idx == 0) ? index_delta : static_cast<uint16_t>(previous_index + index_delta);
                previous_index = group.indices[index_idx];
            }
        }

        compact_ready = true;
        packed_payload_ready = true;
    }

    uint64_t DurationSpikeManager::reconstructed_value(size_t logical_index) const {
        for (uint8_t exact_idx = 0; exact_idx < exact_count; ++exact_idx) {
            if (exact_spikes[exact_idx].idx == logical_index) {
                return exact_spikes[exact_idx].value;
            }
        }

        for (uint8_t group_idx = 0; group_idx < group_count; ++group_idx) {
            const auto& group = spike_groups[group_idx];
            for (uint8_t index_idx = 0; index_idx < group.index_count; ++index_idx) {
                if (group.indices[index_idx] == logical_index) {
                    return group.value;
                }
            }
        }

        if (baseline_stddev == 0) {
            return baseline_mean;
        }

        const uint64_t absolute_index = static_cast<uint64_t>(parent.first_index + logical_index);
        const uint64_t seed_a = splitmix64(absolute_index ^ baseline_mean ^ static_cast<uint64_t>(baseline_stddev));
        const uint64_t seed_b = splitmix64(seed_a ^ 0xd6e8feb86659fd93ULL);
        const double u1 = std::max(1e-12, uniform_unit_from_u64(seed_a));
        const double u2 = uniform_unit_from_u64(seed_b);
        const double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
        const double sigma = static_cast<double>(baseline_stddev);
        const double mean = static_cast<double>(baseline_mean);
        const double low = mean - 3.0 * sigma;
        const double high = mean + 3.0 * sigma;
        const double sampled = std::min(std::max(mean + sigma * z, low), high);
        return static_cast<uint64_t>(std::max(0.0, std::round(sampled)));
    }

    /**
     * @brief Fit the lossy duration model for one full SubArray block.
     *
     * The fitting pipeline is staged on purpose:
     * @code
     *   1. estimate a robust baseline from the median
     *   2. mark strong positive residuals as spike candidates
     *   3. keep the strongest candidates exactly
     *   4. cluster similar remaining spikes into grouped buckets
     *   5. fit a clipped baseline mean/stddev on the non-spike remainder
     *   6. emit the compact payload and, under BMARK, compare reconstruction error
     * @endcode
     *
     * This is the main post-processing step that turns the pseudo-online raw block
     * into the compact baseline-plus-exceptions representation used at read time.
     */
    void DurationSpikeManager::finalize_block() {
        // Stage 0: Reset any prior decoded state and handle the empty-block case.
        clear_state();
        if (parent.value_count == 0 || parent.buffer == nullptr) {
            compact_ready = true;
            packed_payload_ready = true;
            write_packed_payload();
            return;
        }

        const size_t value_count = parent.value_count;
        const auto* raw_values = parent.buffer;

        // Stage 1: Build a robust initial baseline and derive the spike threshold.
        const double initial_baseline = median_from_u64(raw_values, value_count);

        std::array<double, VECTOR_SIZE_2048> abs_residuals{};
        for (size_t idx = 0; idx < value_count; ++idx) {
            abs_residuals[idx] = std::abs(static_cast<double>(raw_values[idx]) - initial_baseline);
        }
        const double median_abs_residual = median_from_double_buffer(abs_residuals.data(), value_count);
        const double spike_threshold = std::max(kMinSpikeResidual, 3.0 * median_abs_residual);

        // Stage 2: Gather and sort the strongest positive residual spike candidates.
        std::array<DurationSpikeCandidate, VECTOR_SIZE_2048> all_candidates{};
        size_t all_candidate_count = 0;
        for (size_t idx = 0; idx < value_count; ++idx) {
            const int64_t residual =
                    static_cast<int64_t>(raw_values[idx]) - static_cast<int64_t>(std::llround(initial_baseline));
            if (static_cast<double>(residual) >= spike_threshold) {
                all_candidates[all_candidate_count++] = {
                        static_cast<uint16_t>(idx),
                        residual,
                };
            }
        }

        std::sort(
                all_candidates.begin(),
                all_candidates.begin() + static_cast<std::ptrdiff_t>(all_candidate_count),
                [](const DurationSpikeCandidate& lhs, const DurationSpikeCandidate& rhs) {
                    return lhs.residual > rhs.residual;
                });

        const size_t candidate_count = std::min(static_cast<size_t>(k_max), all_candidate_count);
        const uint8_t exact_target =
                static_cast<uint8_t>(std::min(candidate_count, static_cast<size_t>(std::min<uint8_t>(kMaxExactSpikes, std::max<uint8_t>(2, k_max / 2)))));

        // Stage 3: Preserve the top spike candidates exactly.
        std::array<bool, VECTOR_SIZE_2048> selected_positions{};
        exact_count = exact_target;
        for (uint8_t exact_idx = 0; exact_idx < exact_count; ++exact_idx) {
            const auto& candidate = all_candidates[exact_idx];
            exact_spikes[exact_idx].idx = candidate.idx;
            exact_spikes[exact_idx].value = raw_values[candidate.idx];
            selected_positions[candidate.idx] = true;
        }

        std::array<bool, VECTOR_SIZE_2048> candidate_used{};
        for (size_t candidate_idx = 0; candidate_idx < candidate_count; ++candidate_idx) {
            if (candidate_idx < exact_count) {
                candidate_used[candidate_idx] = true;
            }
        }

        // Stage 4: Cluster the remaining candidates into grouped spike buckets.
        group_count = 0;
        for (size_t candidate_idx = exact_count; candidate_idx < candidate_count && group_count < kMaxSpikeGroups; ++candidate_idx) {
            if (candidate_used[candidate_idx]) {
                continue;
            }

            const auto& seed = all_candidates[candidate_idx];
            const double tolerance = std::max(
                    kAbsoluteGroupTolerance,
                    std::abs(static_cast<double>(seed.residual)) * kRelativeGroupTolerance);

            std::array<uint16_t, 64> group_positions{};
            std::array<size_t, 64> group_candidate_indices{};
            size_t group_member_count = 0;
            int64_t residual_sum = 0;

            for (size_t inner_idx = candidate_idx; inner_idx < candidate_count; ++inner_idx) {
                if (candidate_used[inner_idx]) {
                    continue;
                }
                const auto& candidate = all_candidates[inner_idx];
                if (std::abs(static_cast<double>(candidate.residual - seed.residual)) <= tolerance) {
                    if (group_member_count < group_positions.size()) {
                        group_positions[group_member_count++] = candidate.idx;
                        group_candidate_indices[group_member_count - 1] = inner_idx;
                        residual_sum += candidate.residual;
                    }
                }
            }

            if (group_member_count < kMinGroupSize) {
                continue;
            }

            std::sort(group_positions.begin(), group_positions.begin() + static_cast<std::ptrdiff_t>(group_member_count));
            auto& group = spike_groups[group_count];
            group.index_count = static_cast<uint8_t>(group_member_count);
            group.value = static_cast<uint64_t>(
                    static_cast<int64_t>(std::llround(initial_baseline)) +
                    static_cast<int64_t>(std::llround(static_cast<double>(residual_sum) / static_cast<double>(group_member_count))));

            for (size_t member_idx = 0; member_idx < group_member_count; ++member_idx) {
                group.indices[member_idx] = group_positions[member_idx];
                candidate_used[group_candidate_indices[member_idx]] = true;
                selected_positions[group_positions[member_idx]] = true;
            }
            ++group_count;
        }

        // Stage 5: Fit the clipped baseline model on values not claimed by spikes.
        std::array<double, VECTOR_SIZE_2048> baseline_values{};
        size_t baseline_value_count = 0;
        for (size_t idx = 0; idx < value_count; ++idx) {
            if (!selected_positions[idx]) {
                baseline_values[baseline_value_count++] = static_cast<double>(raw_values[idx]);
            }
        }

        if (baseline_value_count == 0) {
            baseline_mean = static_cast<uint64_t>(std::max(0.0, std::round(initial_baseline)));
            baseline_stddev = 0;
    #ifdef BMARK
            std::vector<uint64_t> reconstructed_values(value_count);
            for (size_t idx = 0; idx < value_count; ++idx) {
                reconstructed_values[idx] = reconstructed_value(idx);
            }
            const auto family =
                    (parent.parent_lv != nullptr) ? parent.parent_lv->get_bmark_family() : BmarkFamily::Unknown;
            record_subarray_error_metrics(family, raw_values, reconstructed_values.data(), value_count);
    #endif
            write_packed_payload();
            return;
        }

        const double robust_center = median_from_double_buffer(baseline_values.data(), baseline_value_count);
        std::array<double, VECTOR_SIZE_2048> abs_deviations{};
        for (size_t idx = 0; idx < baseline_value_count; ++idx) {
            abs_deviations[idx] = std::abs(baseline_values[idx] - robust_center);
        }
        const double mad = median_from_double_buffer(abs_deviations.data(), baseline_value_count);
        const double robust_sigma = 1.4826 * mad;
        const double clip_radius = std::max(kBaselineMinClipRadius, robust_sigma * kBaselineClipSigma);

        double clipped_sum = 0.0;
        size_t clipped_count = 0;
        for (size_t idx = 0; idx < baseline_value_count; ++idx) {
            if (std::abs(baseline_values[idx] - robust_center) <= clip_radius) {
                clipped_sum += baseline_values[idx];
                ++clipped_count;
            }
        }
        if (clipped_count == 0) {
            for (size_t idx = 0; idx < baseline_value_count; ++idx) {
                clipped_sum += baseline_values[idx];
            }
            clipped_count = baseline_value_count;
        }

        const double mean_value = clipped_sum / static_cast<double>(clipped_count);
        double variance = 0.0;
        if (clipped_count > 1) {
            for (size_t idx = 0; idx < baseline_value_count; ++idx) {
                if (std::abs(baseline_values[idx] - robust_center) <= clip_radius || baseline_value_count == clipped_count) {
                    const double centered = baseline_values[idx] - mean_value;
                    variance += centered * centered;
                }
            }
            variance /= static_cast<double>(clipped_count);
        }

        // Stage 6: Materialize the compact payload header and spike sections.
        baseline_mean = static_cast<uint64_t>(std::max(0.0, std::round(mean_value)));
        baseline_stddev = static_cast<uint32_t>(std::max(0.0, std::round(std::sqrt(std::max(0.0, variance)))));
    #ifdef BMARK
        std::vector<uint64_t> reconstructed_values(value_count);
        for (size_t idx = 0; idx < value_count; ++idx) {
            reconstructed_values[idx] = reconstructed_value(idx);
        }
        const auto family =
                (parent.parent_lv != nullptr) ? parent.parent_lv->get_bmark_family() : BmarkFamily::Unknown;
        record_subarray_error_metrics(family, raw_values, reconstructed_values.data(), value_count);
    #endif
        write_packed_payload();
    }

    AddStatus DurationSpikeManager::add(uint64_t val) {
        if (compact_ready) {
            return AddStatus::Full;
        }
        if (parent.physical_size >= _capacity()) {
            return AddStatus::Full;
        }

        parent.buffer[parent.physical_size] = val;
        parent.value_count++;
        parent.physical_size++;
        if (parent.value_count == _capacity()) {
            finalize_block();
        }
        return AddStatus::Ok;
    }

    uint64_t DurationSpikeManager::at(size_t pos) const {
        if (!parent.contains(pos)) {
            pallas_error("Wrong index (%lu) compared to starting index (%lu) and size (%lu)\n",
                        pos, parent.first_index, parent.value_count);
        }
        if (packed_payload_ready) {
            return reconstructed_value(parent.local_index(pos));
        }
        return parent.buffer[parent.local_index(pos)];
    }

    void DurationSpikeManager::copy_to_array(uint64_t* given_array) const {
        if (given_array == nullptr) {
            return;
        }
        if (packed_payload_ready) {
            for (size_t logical_index = 0; logical_index < parent.size(); ++logical_index) {
                given_array[logical_index] = reconstructed_value(logical_index);
            }
            return;
        }
        if (parent.buffer == nullptr) {
            return;
        }
        std::memcpy(given_array, parent.buffer, parent.value_count * sizeof(uint64_t));
    }

    void DurationSpikeManager::write_data(FILE* data_file, const ParameterHandler* parameter_handler) {
        if (data_file == nullptr || parameter_handler == nullptr || parent.buffer == nullptr) {
            return;
        }
        if (!compact_ready) {
            finalize_block();
        }

        if (compact_ready && !packed_payload_ready) {
            write_packed_payload();
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
        record_subarray_write_metrics(family, data_file, current_offset, pre_raw_bytes, raw_bytes);
    #endif
        parent.free_values();
    }

    void DurationSpikeManager::load_data(FILE* data_file, const ParameterHandler& parameter_handler) {
        if (data_file == nullptr) {
            return;
        }

        clear_state();
        delete[] parent.buffer;
        parent.buffer = _pallas_compress_read(parent.mem_size(), data_file, parameter_handler);
        if (parent.mem_size() < _capacity()) {
            load_packed_payload();
            return;
        }
        compact_ready = true;
    }

    void DurationSpikeManager::on_values_freed() {
        clear_state();
    }

}
