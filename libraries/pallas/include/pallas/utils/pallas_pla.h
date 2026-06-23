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

constexpr size_t kPLABlockSize = 1000;
constexpr size_t kPLAMaxAnchors = 32;

enum class CandidateState : uint8_t {
    Null = 0,
    Free = 1,
    Suppressed = 2,
    Anchor = 3,
};

struct PLAAnchor {
    uint64_t val = 0;
    int32_t dprev = 0;
    uint16_t idx = 0;
};

struct GammaBlockStats {
    uint64_t* raw = nullptr;
    int64_t* delta = nullptr;
    uint64_t* abs_delta_of_delta = nullptr;
    double* abs_delta_deviation = nullptr;
    double* sum_y = nullptr;
    double* sum_y2 = nullptr;
    double* sum_xy = nullptr;
    double* sum_d = nullptr;
    double* sum_d2 = nullptr;
    double* sum_abs_dd = nullptr;
    double* sum_abs_delta_deviation = nullptr;
    double* score = nullptr;
    uint16_t* order = nullptr;
    uint64_t* state_words = nullptr;
    double baseline_delta = 0.0;
    double global_smoothness = 0.0;
    double jarring_threshold = 0.0;
    size_t value_count = 0;

    [[nodiscard]] static size_t helper_buffer_bytes();
    [[nodiscard]] static GammaBlockStats bind(void* buffer);
};

size_t build_pla4_alpha_block(const uint64_t* values,
                              size_t n,
                              GammaBlockStats& stats,
                              PLAAnchor* anchors,
                              size_t anchor_capacity);
size_t build_gamma_anchor_block(const uint64_t* values,
                                size_t n,
                                GammaBlockStats& stats,
                                PLAAnchor* anchors,
                                size_t anchor_capacity);
size_t build_all_interior_anchor_block(const uint64_t* values,
                                       size_t n,
                                       PLAAnchor* anchors,
                                       size_t anchor_capacity);

}
