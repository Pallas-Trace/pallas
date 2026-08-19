/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
/** @file
 * Fixed-size helpers used by the standalone PLA manager implementation.
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace pallas {

/** Logical block size analysed by the standalone PLA compaction helpers. */
constexpr size_t kPLABlockSize = 2048;
/** Maximum number of anchors emitted for one compacted PLA block. */
constexpr size_t kPLAMaxAnchors = 32;

/** Per-position state used while selecting and pruning PLA anchor candidates. */
enum class CandidateState : uint8_t {
    /** Position is not currently participating in the candidate set. */
    Null = 0,
    /** Position is available to become an anchor candidate. */
    Free = 1,
    /** Position was considered but suppressed by a neighbouring choice. */
    Suppressed = 2,
    /** Position is retained as an emitted anchor. */
    Anchor = 3,
};

/** One emitted PLA anchor used to reconstruct a compacted timestamp block. */
struct PLAAnchor {
    /** Reconstructed value stored at the anchor position. */
    uint64_t val = 0;
    /** Previous-delta state associated with this anchor. */
    int32_t dprev = 0;
    /** Logical index of the anchor inside the PLA block. */
    uint16_t idx = 0;
};

/**
 * Scratch views and derived statistics used while analysing one PLA block.
 *
 * The fields are bound onto one caller-provided helper buffer so the block
 * builders can reuse temporary arrays without repeated heap allocation.
 */
struct GammaBlockStats {
    /** Raw logical values of the current block. */
    uint64_t* raw = nullptr;
    /** First-order signed deltas derived from `raw`. */
    int64_t* delta = nullptr;
    /** Absolute delta-of-delta magnitudes used by the scoring heuristics. */
    uint64_t* abs_delta_of_delta = nullptr;
    /** Absolute deviation from the baseline delta trend. */
    double* abs_delta_deviation = nullptr;
    /** Prefix sum of values. */
    double* sum_y = nullptr;
    /** Prefix sum of squared values. */
    double* sum_y2 = nullptr;
    /** Prefix sum of index-value products. */
    double* sum_xy = nullptr;
    /** Prefix sum of first-order deltas. */
    double* sum_d = nullptr;
    /** Prefix sum of squared first-order deltas. */
    double* sum_d2 = nullptr;
    /** Prefix sum of absolute delta-of-delta magnitudes. */
    double* sum_abs_dd = nullptr;
    /** Prefix sum of absolute delta deviations. */
    double* sum_abs_delta_deviation = nullptr;
    /** Per-position candidate score produced by the PLA heuristics. */
    double* score = nullptr;
    /** Sorted candidate order used during anchor selection. */
    uint16_t* order = nullptr;
    /** Packed per-position `CandidateState` words. */
    uint64_t* state_words = nullptr;
    /** Baseline first-order delta estimated for the current block. */
    double baseline_delta = 0.0;
    /** Global smoothness score used by the anchor builders. */
    double global_smoothness = 0.0;
    /** Threshold used to classify locally jarring positions. */
    double jarring_threshold = 0.0;
    /** Number of logical values currently bound into this stats view. */
    size_t value_count = 0;

    /** @returns Size in bytes required for the shared helper buffer. */
    [[nodiscard]] static size_t helper_buffer_bytes();
    /** Bind all scratch views onto one caller-provided helper buffer. */
    [[nodiscard]] static GammaBlockStats bind(void* buffer);
};

/**
 * PLA builder entry points.
 *
 * An anchor is a logical sample that is kept explicitly so the rest of the block can be 
 * reconstructed by interpolation between retained points. The first and last values of a 
 * block always act as the natural start and end anchors, while the interior anchors are 
 * chosen by the helper below. The concrete PLA algorithms should be treated as black-box 
 * selectors that return a compact set of anchors giving good prediction quality for the block.
 */

/** Simple PLA-4 heuristic that ranks spike-like interior positions and keeps the strongest few as anchors. */
size_t build_pla4_alpha_block(const uint64_t* values, size_t n, GammaBlockStats& stats,
                              PLAAnchor* anchors, size_t anchor_capacity);

/**
 * Multi-stage "gamma" anchor builder used by the richer PLA variants.
 *
 * The name is only a local naming convention. In the implementation
 * (`build_gamma_anchor_block()` in `pallas_pla.cpp`) this path first ranks
 * spike-like interior positions, builds an initial seed pool, suppresses
 * candidates that look too flat or overlap too heavily with already chosen
 * anchors, then scores segments and repeatedly refines the worst-fitting
 * segment by inserting a better split anchor.
 */
size_t build_gamma_anchor_block(const uint64_t* values, size_t n, GammaBlockStats& stats,
                                PLAAnchor* anchors, size_t anchor_capacity);

/** Conservative fallback that emits all interior positions as anchors when richer PLA heuristics cannot be applied / are not worth applying. */
size_t build_all_interior_anchor_block(const uint64_t* values, size_t n,PLAAnchor* anchors,
                                       size_t anchor_capacity);

}
