/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
#pragma once

#include "pallas_subarray.h"

namespace pallas {


/**
 * Lossy timestamp manager based on blockwise PLA-style approximation.
 *
 * This path is only pseudo-online: values are accepted through `add()` as if they were being encoded incrementally, but the actual PLA compaction 
 * happens once a full SubArray-sized block has been collected. That design keeps the public linked-vector interface online while still letting the 
 * manager run a stronger block analysis (some time) before serialization. The tradeoff is that `add()` carries extra staging and bookkeeping overhead 
 * compared to exact policies.
 */
class PLAManager : public Manager {
   public:
    PLAManager(SubArrayBase& parent, uint8_t k_max)
        : Manager(parent), k_max(k_max) {}

    [[nodiscard]] size_t _capacity() const override;
    AddStatus add(uint64_t val) override;
    [[nodiscard]] uint64_t at(size_t pos) const override;
    void copy_to_array(uint64_t* given_array) const override;
    void write_data(FILE* data_file, const ParameterHandler* parameter_handler) override;
    void load_data(FILE* data_file, const ParameterHandler& parameter_handler) override;
    void on_subarray_initialized() override;
    void on_values_freed() override;

   private:
    /** Allocate and bind the temporary helper buffers needed by block compaction. */
    void ensure_staging();
    /** Build the compact PLA representation once the staging block is full. */
    void finalize_block();
    /** Pack the compact anchor representation into the SubArray buffer. */
    void write_packed_payload();
    /** Reload anchors and helper state from the packed SubArray payload. */
    void load_packed_payload();
    /** Reconstruct one logical timestamp from the compact anchor representation. */
    [[nodiscard]] uint64_t interpolate_value(const TimeSubArray& subarray, size_t logical_index) const;
    /** Drop transient staging and compaction state after values are freed. */
    void clear_state();

    /** Policy knob selecting the maximum anchor budget used by the PLA builder. */
    uint8_t k_max = 0;
    /** Number of anchors emitted into `anchor_storage` for the current block. */
    uint8_t anchor_count = 0;
    /** Whether the current full block has already been compacted. */
    bool compact_ready = false;
    /** ScratchPad views and statistics used while analysing one full PLA block. */
    GammaBlockStats stats{};
    /** Fixed storage for the anchors produced by the block compaction step. */
    PLAAnchor anchor_storage[kPLAMaxAnchors]{};
};

/**
 * Lossy duration manager that models a block around a clipped baseline and a bounded set of spike exceptions.
 *
 * Like `PLAManager`, this path is only pseudo-online. Values are accepted through `add()` during runtime, but the actual model fitting and spike
 * selection happen as a post-processing step once the current SubArray block is full. This keeps the outer linked-vector API online while allowing the
 * manager to inspect the complete block before deciding which values become exact spikes, grouped spikes, or baseline-covered samples.
 */
class DurationSpikeManager : public Manager {
   public:
    DurationSpikeManager(SubArrayBase& parent, uint8_t k_max)
        : Manager(parent), k_max(k_max) {}

    [[nodiscard]] size_t _capacity() const override;
    AddStatus add(uint64_t val) override;
    [[nodiscard]] uint64_t at(size_t pos) const override;
    void copy_to_array(uint64_t* given_array) const override;
    void write_data(FILE* data_file, const ParameterHandler* parameter_handler) override;
    void load_data(FILE* data_file, const ParameterHandler& parameter_handler) override;
    void on_subarray_initialized() override;
    void on_values_freed() override;

   private:
    /** One logical sample kept exactly because it does not fit the baseline well. */
    struct ExactSpike {
        /** Logical index of the exact exception. */
        uint16_t idx = 0;
        /** Exact stored duration value. */
        uint64_t value = 0;
    };

    /** Small group of samples that share the same representative spike value. */
    struct SpikeGroup {
        /** Representative value assigned to every member of the group. */
        uint64_t value = 0;
        /** Number of logical indices stored in `indices`. */
        uint8_t index_count = 0;
        /** Logical indices that reconstruct to `value`. */
        std::array<uint16_t, 64> indices{};
    };

    /** Hard cap on individually stored exact spikes per compacted block. */
    static constexpr uint8_t kMaxExactSpikes = 8;
    /** Hard cap on grouped spike buckets per compacted block. */
    static constexpr uint8_t kMaxSpikeGroups = 4;
    /** Minimum population required before a spike group is worth keeping. */
    static constexpr uint8_t kMinGroupSize = 2;
    /** Relative tolerance used when clustering residual spikes into groups. */
    static constexpr double kRelativeGroupTolerance = 0.20;
    /** Absolute tolerance used when clustering residual spikes into groups. */
    static constexpr double kAbsoluteGroupTolerance = 96.0;
    /** Minimum residual magnitude required before a value is treated as a spike. */
    static constexpr double kMinSpikeResidual = 64.0;
    /** Sigma-based clipping factor used while estimating the baseline window. */
    static constexpr double kBaselineClipSigma = 2.5;
    /** Minimum absolute clipping radius retained around the baseline mean. */
    static constexpr double kBaselineMinClipRadius = 32.0;

    /** Analyse the staged block and build the baseline-plus-spikes model. */
    void finalize_block();
    /** Reset transient block-analysis state after values are freed. */
    void clear_state();
    /** Pack the compact baseline and spike metadata into the SubArray buffer. */
    void write_packed_payload();
    /** Reload the packed baseline and spike metadata from persisted bytes. */
    void load_packed_payload();
    /** Return the exact payload size needed for the packed compact form. */
    [[nodiscard]] size_t packed_payload_bytes() const;
    /** Reconstruct one logical duration from the baseline and spike metadata. */
    [[nodiscard]] uint64_t reconstructed_value(size_t logical_index) const;

    /** Whether the current staged block has already been compacted. */
    bool compact_ready = false;
    /** Whether packed payload bytes are ready to be written or replayed. */
    bool packed_payload_ready = false;
    /** Policy knob controlling the spike budget chosen by `LossyPolicy::Spike*`. */
    uint8_t k_max = 0;
    /** Mean of the clipped baseline used for non-spike samples. */
    uint64_t baseline_mean = 0;
    /** Baseline spread encoded alongside the compacted block. */
    uint32_t baseline_stddev = 0;
    /** Number of populated entries in `exact_spikes`. */
    uint8_t exact_count = 0;
    /** Number of populated entries in `spike_groups`. */
    uint8_t group_count = 0;
    /** Individually stored exceptions for values that remain too important to merge. */
    std::array<ExactSpike, kMaxExactSpikes> exact_spikes{};
    /** Grouped spike buckets used to represent repeated exceptional values compactly. */
    std::array<SpikeGroup, kMaxSpikeGroups> spike_groups{};
};

}