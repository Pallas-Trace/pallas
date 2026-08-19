/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>

#include "pallas/linked_vector/pallas_pla.h"

namespace pallas {

namespace {

// Gamma tuning constants shared across the pipeline stages.
constexpr size_t kGammaLocalRadius = 8;
constexpr size_t kGammaSeedPoolSize = 64;
constexpr size_t kGammaSeedCount = 4;
constexpr size_t kGammaCandidatesPerSegment = 4;
constexpr uint16_t kGammaRightWindows[] = {4, 16, 32, 64, 128, 256, 512};
constexpr uint16_t kGammaLeftWindows[] = {4, 16, 32, 64};
constexpr double kGammaSuppressedOverlapThreshold = 0.6;
constexpr uint16_t kGammaMinSegmentLength = 8;
constexpr double kGammaMinGainRatio = 0.05;
constexpr double kGammaMinGainAbs = 1e-6;
constexpr size_t kStateWordCount = (kPLABlockSize + 31) / 32;

}  // namespace

// [Gamma] Pure numeric helpers and compact-state accessors.
namespace {

static uint64_t abs_i64(int64_t value) {
    return static_cast<uint64_t>(value < 0 ? -value : value);
}

static double abs_f64(double value) {
    return value < 0.0 ? -value : value;
}

static double range_sum(const double* prefix, uint16_t start, uint16_t end_inclusive) {
    if (end_inclusive < start) {
        return 0.0;
    }
    return prefix[end_inclusive + 1] - prefix[start];
}

static double sum_of_squares(uint16_t end_inclusive) {
    const double end_value = static_cast<double>(end_inclusive);
    return end_value * (end_value + 1.0) * (2.0 * end_value + 1.0) / 6.0;
}

static double range_sum_x(uint16_t start, uint16_t end_inclusive) {
    const double length = static_cast<double>(end_inclusive - start + 1);
    return length * static_cast<double>(start + end_inclusive) / 2.0;
}

static double range_sum_x2(uint16_t start, uint16_t end_inclusive) {
    if (end_inclusive < start) {
        return 0.0;
    }
    const double low = (start > 0) ? sum_of_squares(static_cast<uint16_t>(start - 1)) : 0.0;
    return sum_of_squares(end_inclusive) - low;
}

static double segment_fit_score(const GammaBlockStats& stats, uint16_t start, uint16_t end_inclusive) {
    if (end_inclusive <= start) {
        return 0.0;
    }

    const double count = static_cast<double>(end_inclusive - start + 1);
    const double sum_x = range_sum_x(start, end_inclusive);
    const double sum_x2 = range_sum_x2(start, end_inclusive);
    const double sum_y = range_sum(stats.sum_y, start, end_inclusive);
    const double sum_y2 = range_sum(stats.sum_y2, start, end_inclusive);
    const double sum_xy = range_sum(stats.sum_xy, start, end_inclusive);

    const double centered_xx = sum_x2 - (sum_x * sum_x) / count;
    const double centered_xy = sum_xy - (sum_x * sum_y) / count;
    const double centered_yy = sum_y2 - (sum_y * sum_y) / count;

    if (centered_xx <= 0.0) {
        return std::max(0.0, centered_yy) / count;
    }

    const double sse = centered_yy - (centered_xy * centered_xy) / centered_xx;
    return std::max(0.0, sse) / count;
}

static double segment_smoothness_score(const GammaBlockStats& stats, uint16_t start, uint16_t end_inclusive) {
    if (end_inclusive <= start || stats.value_count <= 1) {
        return 0.0;
    }

    const uint16_t delta_start = start;
    const uint16_t delta_end = static_cast<uint16_t>(end_inclusive - 1);
    const double delta_count = static_cast<double>(delta_end - delta_start + 1);
    const double sum_delta = range_sum(stats.sum_d, delta_start, delta_end);
    const double sum_delta2 = range_sum(stats.sum_d2, delta_start, delta_end);
    const double mean_delta = sum_delta / delta_count;
    const double variance = std::max(0.0, (sum_delta2 / delta_count) - (mean_delta * mean_delta));
    const double std_delta = std::sqrt(variance);

    const uint16_t delta_value_count = static_cast<uint16_t>(stats.value_count > 0 ? stats.value_count - 1 : 0);
    if (delta_value_count < 2) {
        return 0.5 * std_delta;
    }

    const uint16_t dod_start = start;
    const uint16_t dod_end = static_cast<uint16_t>(end_inclusive - 2);
    if (dod_end < dod_start || dod_start >= delta_value_count - 1) {
        return 0.5 * std_delta;
    }

    const double dod_count = static_cast<double>(dod_end - dod_start + 1);
    const double mean_abs_dd = range_sum(stats.sum_abs_dd, dod_start, dod_end) / dod_count;
    return mean_abs_dd + 0.5 * std_delta;
}

static CandidateState get_state(const GammaBlockStats& stats, uint16_t position) {
    const size_t word_index = position / 32;
    const size_t shift = static_cast<size_t>(position % 32) * 2;
    const uint64_t bits = (stats.state_words[word_index] >> shift) & 0x3ULL;
    return static_cast<CandidateState>(bits);
}

static void set_state(const GammaBlockStats& stats, uint16_t position, CandidateState state) {
    const size_t word_index = position / 32;
    const size_t shift = static_cast<size_t>(position % 32) * 2;
    const uint64_t mask = 0x3ULL << shift;
    stats.state_words[word_index] =
            (stats.state_words[word_index] & ~mask) | (static_cast<uint64_t>(state) << shift);
}

static void initialize_states(const GammaBlockStats& stats, size_t n) {
    std::memset(stats.state_words, 0, kStateWordCount * sizeof(uint64_t));
    if (n == 0) {
        return;
    }
    set_state(stats, 0, CandidateState::Null);
    for (size_t i = 1; i + 1 < n; ++i) {
        set_state(stats, static_cast<uint16_t>(i), CandidateState::Free);
    }
    if (n >= 2) {
        set_state(stats, static_cast<uint16_t>(n - 1), CandidateState::Null);
    }
}

static void insert_top_position(uint16_t idx, double score, uint16_t* positions,
                         double* scores, size_t limit, size_t& count) {
    size_t insert_at = count;
    while (insert_at > 0) {
        const size_t prev = insert_at - 1;
        if (score < scores[prev]) {
            break;
        }
        if (score == scores[prev] && idx > positions[prev]) {
            break;
        }
        if (insert_at < limit) {
            positions[insert_at] = positions[prev];
            scores[insert_at] = scores[prev];
        }
        insert_at = prev;
    }

    if (insert_at < limit) {
        positions[insert_at] = idx;
        scores[insert_at] = score;
        if (count < limit) {
            ++count;
        }
    }
}

static void insert_sorted_position(uint16_t position, uint16_t* positions, size_t& count) {
    size_t insert_at = count;
    while (insert_at > 0 && positions[insert_at - 1] > position) {
        positions[insert_at] = positions[insert_at - 1];
        --insert_at;
    }
    positions[insert_at] = position;
    ++count;
}

}  // namespace

// [Gamma] precomputation and score materialization helpers.
namespace {

static double compute_delta_median(const int64_t* delta, size_t delta_count) {
    if (delta_count == 0) {
        return 0.0;
    }

    std::array<int64_t, kPLABlockSize> sorted_delta{};
    for (size_t i = 0; i < delta_count; ++i) {
        sorted_delta[i] = delta[i];
    }
    std::sort(sorted_delta.begin(), sorted_delta.begin() + static_cast<std::ptrdiff_t>(delta_count));

    if ((delta_count & 1U) != 0U) {
        return static_cast<double>(sorted_delta[delta_count / 2]);
    }
    const size_t upper = delta_count / 2;
    const size_t lower = upper - 1;
    return (static_cast<double>(sorted_delta[lower]) + static_cast<double>(sorted_delta[upper])) / 2.0;
}

static void prepare_gamma_stats(const uint64_t* values, size_t n, GammaBlockStats& stats) {
    assert(values != nullptr);
    assert(n <= kPLABlockSize);

    stats.value_count = n;
    if (n == 0) {
        stats.baseline_delta = 0.0;
        stats.global_smoothness = 0.0;
        stats.jarring_threshold = 0.0;
        return;
    }

    const size_t delta_count = (n > 0) ? (n - 1) : 0;
    const size_t dod_count = (delta_count > 0) ? (delta_count - 1) : 0;

    for (size_t i = 0; i < delta_count; ++i) {
        stats.delta[i] = static_cast<int64_t>(values[i + 1]) - static_cast<int64_t>(values[i]);
    }
    stats.baseline_delta = compute_delta_median(stats.delta, delta_count);

    stats.sum_y[0] = 0.0;
    stats.sum_y2[0] = 0.0;
    stats.sum_xy[0] = 0.0;
    stats.sum_d[0] = 0.0;
    stats.sum_d2[0] = 0.0;
    stats.sum_abs_dd[0] = 0.0;
    stats.sum_abs_delta_deviation[0] = 0.0;
    stats.score[0] = 0.0;

    for (size_t i = 0; i < n; ++i) {
        const double value = static_cast<double>(values[i]);
        stats.sum_y[i + 1] = stats.sum_y[i] + value;
        stats.sum_y2[i + 1] = stats.sum_y2[i] + value * value;
        stats.sum_xy[i + 1] = stats.sum_xy[i] + static_cast<double>(i) * value;

        if (i < delta_count) {
            const double delta_value = static_cast<double>(stats.delta[i]);
            stats.sum_d[i + 1] = stats.sum_d[i] + delta_value;
            stats.sum_d2[i + 1] = stats.sum_d2[i] + delta_value * delta_value;

            const double abs_deviation = abs_f64(delta_value - stats.baseline_delta);
            stats.abs_delta_deviation[i] = abs_deviation;
            stats.sum_abs_delta_deviation[i + 1] =
                    stats.sum_abs_delta_deviation[i] + abs_deviation;
        } else {
            stats.sum_d[i + 1] = stats.sum_d[i];
            stats.sum_d2[i + 1] = stats.sum_d2[i];
            stats.sum_abs_delta_deviation[i + 1] = stats.sum_abs_delta_deviation[i];
        }

        if (i < dod_count) {
            const double abs_dd = static_cast<double>(abs_i64(stats.delta[i + 1] - stats.delta[i]));
            stats.abs_delta_of_delta[i] = static_cast<uint64_t>(abs_dd);
            stats.sum_abs_dd[i + 1] = stats.sum_abs_dd[i] + abs_dd;
        } else {
            stats.sum_abs_dd[i + 1] = stats.sum_abs_dd[i];
        }
    }

    for (size_t i = 1; i + 1 < n; ++i) {
        const uint16_t delta_idx = static_cast<uint16_t>(i - 1);
        const uint16_t local_delta_start = (delta_idx > kGammaLocalRadius)
                                           ? static_cast<uint16_t>(delta_idx - kGammaLocalRadius)
                                           : 0;
        const uint16_t local_delta_end =
                static_cast<uint16_t>(std::min(delta_count - 1, i - 1 + kGammaLocalRadius));
        const double local_delta_count =
                static_cast<double>(local_delta_end - local_delta_start + 1);
        const double local_mean_abs_deviation =
                range_sum(stats.sum_abs_delta_deviation, local_delta_start, local_delta_end) /
                local_delta_count;

        double local_mean_abs_dod = 0.0;
        if (dod_count > 0) {
            const uint16_t local_dod_start = (delta_idx > kGammaLocalRadius)
                                             ? static_cast<uint16_t>(delta_idx - kGammaLocalRadius)
                                             : 0;
            const uint16_t local_dod_end = static_cast<uint16_t>(
                    std::min(static_cast<size_t>(dod_count - 1), static_cast<size_t>(delta_idx) + kGammaLocalRadius));
            if (local_dod_end >= local_dod_start) {
                const double local_dod_count =
                        static_cast<double>(local_dod_end - local_dod_start + 1);
                local_mean_abs_dod =
                        range_sum(stats.sum_abs_dd, local_dod_start, local_dod_end) / local_dod_count;
            }
        }

        double candidate_sharpness = 0.0;
        if (delta_idx > 0 && static_cast<size_t>(delta_idx - 1) < dod_count) {
            candidate_sharpness = std::max(
                    candidate_sharpness,
                    static_cast<double>(stats.abs_delta_of_delta[delta_idx - 1]));
        }
        if (static_cast<size_t>(delta_idx) < dod_count) {
            candidate_sharpness = std::max(
                    candidate_sharpness,
                    static_cast<double>(stats.abs_delta_of_delta[delta_idx]));
        }

        stats.score[i] =
                stats.abs_delta_deviation[delta_idx] /
                        (1.0 + local_mean_abs_deviation) +
                0.75 * candidate_sharpness / (1.0 + local_mean_abs_dod);
    }
    if (n >= 2) {
        stats.score[n - 1] = 0.0;
    }

    for (size_t i = 0; i < n; ++i) {
        stats.order[i] = static_cast<uint16_t>(i);
    }
    if (n > 2) {
        std::sort(
                stats.order + 1,
                stats.order + static_cast<std::ptrdiff_t>(n - 1),
                [&stats](uint16_t left, uint16_t right) {
                    if (stats.score[left] != stats.score[right]) {
                        return stats.score[left] > stats.score[right];
                    }
                    return left < right;
                });
    }

    stats.global_smoothness = segment_smoothness_score(stats, 0, static_cast<uint16_t>(n - 1));

    const size_t interior_count = (n > 2) ? (n - 2) : 0;
    const size_t seed_pool_count = std::min(kGammaSeedPoolSize, interior_count);
    if (seed_pool_count == 0) {
        stats.jarring_threshold = 0.0;
        return;
    }

    std::array<double, kGammaSeedPoolSize> seed_pool_scores{};
    for (size_t i = 0; i < seed_pool_count; ++i) {
        seed_pool_scores[i] = stats.score[stats.order[i + 1]];
    }
    std::sort(seed_pool_scores.begin(),
              seed_pool_scores.begin() + static_cast<std::ptrdiff_t>(seed_pool_count));
    const size_t pivot_index = static_cast<size_t>(0.75 * static_cast<double>(seed_pool_count - 1));
    stats.jarring_threshold = seed_pool_scores[pivot_index];
}

static bool is_jarring_position(const GammaBlockStats& stats, uint16_t position) {
    return stats.score[position] >= stats.jarring_threshold;
}

static void build_seed_pool(const GammaBlockStats& stats, size_t n, uint16_t* seed_pool, size_t& seed_pool_count) {
    seed_pool_count = 0;
    for (size_t i = 1; i + 1 < n && seed_pool_count < kGammaSeedPoolSize; ++i) {
        const uint16_t position = stats.order[i];
        if (get_state(stats, position) == CandidateState::Free) {
            seed_pool[seed_pool_count++] = position;
        }
    }
}

static size_t select_initial_seeds(const GammaBlockStats& stats, const uint16_t* seed_pool,
                            size_t seed_pool_count, size_t target_anchor_count, uint16_t* anchors) {
    if (seed_pool_count == 0 || target_anchor_count == 0) {
        return 0;
    }

    const size_t max_seed_count = std::min({kGammaSeedCount, target_anchor_count, seed_pool_count});
    std::array<bool, kGammaSeedPoolSize> used{};
    anchors[0] = seed_pool[0];
    used[0] = true;
    size_t selected_count = 1;

    while (selected_count < max_seed_count) {
        int best_index = -1;
        uint16_t best_distance = 0;
        double best_score = -1.0;

        for (size_t i = 0; i < seed_pool_count; ++i) {
            if (used[i]) {
                continue;
            }
            const uint16_t candidate = seed_pool[i];
            uint16_t min_distance = UINT16_MAX;
            for (size_t j = 0; j < selected_count; ++j) {
                const uint16_t anchor = anchors[j];
                const uint16_t distance = static_cast<uint16_t>(
                        candidate > anchor ? candidate - anchor : anchor - candidate);
                min_distance = std::min(min_distance, distance);
            }

            const double candidate_score = stats.score[candidate];
            if (best_index < 0 ||
                min_distance > best_distance ||
                (min_distance == best_distance && candidate_score > best_score)) {
                best_index = static_cast<int>(i);
                best_distance = min_distance;
                best_score = candidate_score;
            }
        }

        if (best_index < 0) {
            break;
        }
        used[best_index] = true;
        anchors[selected_count++] = seed_pool[best_index];
    }

    std::sort(anchors, anchors + static_cast<std::ptrdiff_t>(selected_count));
    return selected_count;
}

}  // namespace

// [Gamma] refinement, suppression, and anchor-emission helpers.
namespace {

static double suppressed_overlap(const GammaBlockStats& stats, uint16_t start, uint16_t end_inclusive) {
    if (end_inclusive < start) {
        return 0.0;
    }
    const double length = static_cast<double>(end_inclusive - start + 1);
    size_t suppressed_count = 0;
    for (uint16_t position = start; position <= end_inclusive; ++position) {
        if (get_state(stats, position) == CandidateState::Suppressed) {
            ++suppressed_count;
        }
    }
    return static_cast<double>(suppressed_count) / length;
}

static int first_blocking_position(const GammaBlockStats& stats, uint16_t anchor_position, uint16_t start,
                            uint16_t end_inclusive, int direction) {
    if (direction > 0) {
        for (uint16_t position = start; position <= end_inclusive; ++position) {
            if (position == anchor_position) {
                continue;
            }
            const auto state = get_state(stats, position);
            if (state == CandidateState::Anchor || is_jarring_position(stats, position)) {
                return static_cast<int>(position);
            }
        }
        return -1;
    }

    for (int position = static_cast<int>(end_inclusive); position >= static_cast<int>(start); --position) {
        if (static_cast<uint16_t>(position) == anchor_position) {
            continue;
        }
        const auto state = get_state(stats, static_cast<uint16_t>(position));
        if (state == CandidateState::Anchor ||
            is_jarring_position(stats, static_cast<uint16_t>(position))) {
            return position;
        }
    }
    return -1;
}

static void suppress_direction(const GammaBlockStats& stats, size_t n, uint16_t anchor_position,
                        int direction, const uint16_t* windows, size_t window_count) {
    const double smoothness_limit = std::max(stats.global_smoothness, 1e-6);

    for (size_t window_index = 0; window_index < window_count; ++window_index) {
        const uint16_t window_size = windows[window_index];
        uint16_t raw_start = 0;
        uint16_t raw_end = 0;

        if (direction > 0) {
            raw_start = static_cast<uint16_t>(anchor_position + 1);
            raw_end = static_cast<uint16_t>(std::min(n - 2, static_cast<size_t>(anchor_position) + window_size));
        } else {
            raw_start = static_cast<uint16_t>(
                    (anchor_position > window_size) ? (anchor_position - window_size) : 1);
            raw_end = static_cast<uint16_t>(anchor_position - 1);
        }

        if (raw_end < raw_start) {
            break;
        }

        const int blocked = first_blocking_position(stats, anchor_position, raw_start, raw_end, direction);
        uint16_t effective_start = raw_start;
        uint16_t effective_end = raw_end;
        if (blocked >= 0) {
            if (direction > 0) {
                effective_end = static_cast<uint16_t>(blocked - 1);
            } else {
                effective_start = static_cast<uint16_t>(blocked + 1);
            }
        }

        if (effective_end < effective_start) {
            break;
        }

        const uint16_t smooth_start = (direction > 0) ? anchor_position : effective_start;
        const uint16_t smooth_end = (direction > 0) ? effective_end : anchor_position;
        if (segment_smoothness_score(stats, smooth_start, smooth_end) > smoothness_limit) {
            break;
        }
        if (suppressed_overlap(stats, effective_start, effective_end) > kGammaSuppressedOverlapThreshold) {
            break;
        }

        for (uint16_t position = effective_start; position <= effective_end; ++position) {
            if (get_state(stats, position) == CandidateState::Free) {
                set_state(stats, position, CandidateState::Suppressed);
            }
        }

        if (blocked >= 0) {
            break;
        }
    }
}

static void apply_anchor_suppression(const GammaBlockStats& stats, size_t n, uint16_t anchor_position) {
    suppress_direction(stats,
                       n,
                       anchor_position,
                       1,
                       kGammaRightWindows,
                       sizeof(kGammaRightWindows) / sizeof(kGammaRightWindows[0]));
    suppress_direction(stats,
                       n,
                       anchor_position,
                       -1,
                       kGammaLeftWindows,
                       sizeof(kGammaLeftWindows) / sizeof(kGammaLeftWindows[0]));
}

static void build_segments(const uint16_t* anchors, size_t anchor_count, size_t n,
                    uint16_t* starts, uint16_t* ends, size_t& segment_count) {
    segment_count = 0;
    uint16_t segment_start = 0;
    for (size_t i = 0; i < anchor_count; ++i) {
        starts[segment_count] = segment_start;
        ends[segment_count] = static_cast<uint16_t>(anchors[i] - 1);
        ++segment_count;
        segment_start = anchors[i];
    }
    starts[segment_count] = segment_start;
    ends[segment_count] = static_cast<uint16_t>(n - 1);
    ++segment_count;
}

static bool choose_best_refinement(const GammaBlockStats& stats, size_t n, const uint16_t* anchors,
                            size_t anchor_count, uint16_t& chosen_split) {
    std::array<uint16_t, kPLAMaxAnchors + 1> starts{};
    std::array<uint16_t, kPLAMaxAnchors + 1> ends{};
    size_t segment_count = 0;
    build_segments(anchors, anchor_count, n, starts.data(), ends.data(), segment_count);

    bool found = false;
    double best_gain = 0.0;
    double best_threshold = 0.0;
    uint16_t best_split = 0;

    for (size_t segment_index = 0; segment_index < segment_count; ++segment_index) {
        const uint16_t start = starts[segment_index];
        const uint16_t end_inclusive = ends[segment_index];
        if (end_inclusive < start) {
            continue;
        }
        if (static_cast<uint16_t>(end_inclusive - start + 1) < kGammaMinSegmentLength) {
            continue;
        }

        std::array<uint16_t, kGammaCandidatesPerSegment> candidates{};
        std::array<double, kGammaCandidatesPerSegment> candidate_scores{};
        size_t candidate_count = 0;

        const uint16_t scan_start = std::max<uint16_t>(1, static_cast<uint16_t>(start + 1));
        const uint16_t scan_end = std::min<uint16_t>(static_cast<uint16_t>(n - 2), end_inclusive);
        for (int state_pass = static_cast<int>(CandidateState::Free);
             state_pass <= static_cast<int>(CandidateState::Suppressed) && candidate_count == 0;
             ++state_pass) {
            for (uint16_t position = scan_start; position <= scan_end; ++position) {
                if (get_state(stats, position) != static_cast<CandidateState>(state_pass)) {
                    continue;
                }
                insert_top_position(position,
                                    stats.score[position],
                                    candidates.data(),
                                    candidate_scores.data(),
                                    kGammaCandidatesPerSegment,
                                    candidate_count);
            }
        }

        if (candidate_count == 0) {
            continue;
        }

        const double parent_score = segment_fit_score(stats, start, end_inclusive);
        const double gain_threshold = std::max(kGammaMinGainAbs, parent_score * kGammaMinGainRatio);

        for (size_t candidate_index = 0; candidate_index < candidate_count; ++candidate_index) {
            const uint16_t split = candidates[candidate_index];
            const double left_score = segment_fit_score(stats, start, static_cast<uint16_t>(split - 1));
            const double right_score = segment_fit_score(stats, split, end_inclusive);
            const double gain = parent_score - (left_score + right_score);

            if (!found || gain > best_gain) {
                found = true;
                best_gain = gain;
                best_threshold = gain_threshold;
                best_split = split;
            }
        }
    }

    if (!found || best_gain <= best_threshold) {
        return false;
    }
    chosen_split = best_split;
    return true;
}

static void fill_remaining_anchors(const GammaBlockStats& stats, size_t n, uint16_t* anchors, size_t& anchor_count,
                            size_t target_anchor_count) {
    if (anchor_count >= target_anchor_count) {
        return;
    }

    auto try_fill = [&](CandidateState allowed_state) {
        for (size_t i = 1; i + 1 < n && anchor_count < target_anchor_count; ++i) {
            const uint16_t position = stats.order[i];
            if (get_state(stats, position) != allowed_state) {
                continue;
            }
            insert_sorted_position(position, anchors, anchor_count);
            set_state(stats, position, CandidateState::Anchor);
        }
    };

    try_fill(CandidateState::Free);
    try_fill(CandidateState::Suppressed);

    for (size_t i = 1; i + 1 < n && anchor_count < target_anchor_count; ++i) {
        const uint16_t position = stats.order[i];
        if (get_state(stats, position) == CandidateState::Anchor) {
            continue;
        }
        insert_sorted_position(position, anchors, anchor_count);
        set_state(stats, position, CandidateState::Anchor);
    }
}

static size_t emit_anchor_records(const uint64_t* values, const uint16_t* anchor_positions,
                           size_t anchor_count, PLAAnchor* anchors) {
    for (size_t i = 0; i < anchor_count; ++i) {
        const uint16_t idx = anchor_positions[i];
        anchors[i].idx = idx;
        anchors[i].val = values[idx];
        anchors[i].dprev = static_cast<int32_t>(
                static_cast<int64_t>(values[idx]) - static_cast<int64_t>(values[idx - 1]));
    }
    return anchor_count;
}

}  // namespace

/**
 * @brief Return the size of the shared scratch buffer needed by `GammaBlockStats`.
 *
 * The returned size covers all temporary arrays required by the PLA helper
 * pipeline for one full `kPLABlockSize` block, including prefix sums, per-point
 * scores, candidate order, and packed candidate-state words.
 */
size_t GammaBlockStats::helper_buffer_bytes() {
    const size_t value_count = kPLABlockSize;
    const size_t prefix_count = kPLABlockSize + 1;
    return value_count * sizeof(uint64_t) +                    // raw
           value_count * sizeof(int64_t) +                     // delta
           value_count * sizeof(uint64_t) +                    // abs_delta_of_delta
           value_count * sizeof(double) +                      // abs_delta_deviation
           prefix_count * sizeof(double) +                     // sum_y
           prefix_count * sizeof(double) +                     // sum_y2
           prefix_count * sizeof(double) +                     // sum_xy
           prefix_count * sizeof(double) +                     // sum_d
           prefix_count * sizeof(double) +                     // sum_d2
           prefix_count * sizeof(double) +                     // sum_abs_dd
           prefix_count * sizeof(double) +                     // sum_abs_delta_deviation
           value_count * sizeof(double) +                      // score
           value_count * sizeof(uint16_t) +                    // order
           kStateWordCount * sizeof(uint64_t);                 // packed candidate state
}

/**
 * @brief Bind a caller-provided scratch buffer to the typed views in `GammaBlockStats`.
 *
 * No allocation happens here: the method simply carves the single flat helper
 * buffer into the ordered array views expected by the gamma and PLA4 builders.
 */
GammaBlockStats GammaBlockStats::bind(void* buffer) {
    auto* bytes = static_cast<uint8_t*>(buffer);
    GammaBlockStats stats{};
    stats.raw = reinterpret_cast<uint64_t*>(bytes);  bytes += kPLABlockSize * sizeof(uint64_t);
    stats.delta = reinterpret_cast<int64_t*>(bytes); bytes += kPLABlockSize * sizeof(int64_t);
    stats.abs_delta_of_delta = reinterpret_cast<uint64_t*>(bytes);   bytes += kPLABlockSize * sizeof(uint64_t);
    stats.abs_delta_deviation = reinterpret_cast<double*>(bytes);    bytes += kPLABlockSize * sizeof(double);
    stats.sum_y = reinterpret_cast<double*>(bytes);  bytes += (kPLABlockSize + 1) * sizeof(double);
    stats.sum_y2 = reinterpret_cast<double*>(bytes); bytes += (kPLABlockSize + 1) * sizeof(double);
    stats.sum_xy = reinterpret_cast<double*>(bytes); bytes += (kPLABlockSize + 1) * sizeof(double);
    stats.sum_d = reinterpret_cast<double*>(bytes);  bytes += (kPLABlockSize + 1) * sizeof(double);
    stats.sum_d2 = reinterpret_cast<double*>(bytes); bytes += (kPLABlockSize + 1) * sizeof(double);
    stats.sum_abs_dd = reinterpret_cast<double*>(bytes); bytes += (kPLABlockSize + 1) * sizeof(double);
    stats.sum_abs_delta_deviation = reinterpret_cast<double*>(bytes);bytes += (kPLABlockSize + 1) * sizeof(double);
    stats.score = reinterpret_cast<double*>(bytes);  bytes += kPLABlockSize * sizeof(double);
    stats.order = reinterpret_cast<uint16_t*>(bytes);bytes += kPLABlockSize * sizeof(uint16_t);
    stats.state_words = reinterpret_cast<uint64_t*>(bytes);
    return stats;
}

size_t build_pla4_alpha_block(const uint64_t* values, size_t n,GammaBlockStats& stats,
                              PLAAnchor* anchors, size_t anchor_capacity) {
    assert(values != nullptr);
    assert(anchors != nullptr);
    assert(anchor_capacity >= 4);
    assert(n >= 6);

    const size_t max_candidates = 4;
    std::array<uint16_t, 4> best_idx{};
    std::array<double, 4> best_score{};
    size_t best_count = 0;

    // Stage 1: Materialize the first-order delta stream for the block.
    const size_t delta_count = n - 1;
    for (size_t i = 0; i < delta_count; ++i) {
        stats.delta[i] = static_cast<int64_t>(values[i + 1]) - static_cast<int64_t>(values[i]);
    }

    // Stage 2: Score each interior position by how spike-like its local delta pattern looks.
    for (size_t position = 1; position + 1 < n; ++position) {
        const size_t delta_idx = position - 1;
        const size_t left = (delta_idx > 5) ? (delta_idx - 5) : 0;
        const size_t right = std::min(delta_count - 1, delta_idx + 5);

        int64_t sum = 0;
        size_t count = 0;
        for (size_t j = left; j <= right; ++j) {
            if (j == delta_idx) {
                continue;
            }
            sum += stats.delta[j];
            ++count;
        }

        const int64_t baseline = (count != 0) ? (sum / static_cast<int64_t>(count)) : 0;
        const int64_t current_delta = stats.delta[delta_idx];
        const int64_t left_delta = (delta_idx > 0) ? stats.delta[delta_idx - 1] : current_delta;
        const int64_t right_delta =
                (delta_idx + 1 < delta_count) ? stats.delta[delta_idx + 1] : current_delta;
        const double score = static_cast<double>(
                abs_i64(current_delta - baseline) +
                abs_i64(current_delta - left_delta) +
                abs_i64(right_delta - current_delta));

        stats.score[position] = score;
        insert_top_position(static_cast<uint16_t>(position),
                            score,
                            best_idx.data(),
                            best_score.data(),
                            max_candidates,
                            best_count);
    }

    // Stage 3: Sort the strongest candidates and emit them as interior anchors.
    const size_t emitted_count = std::min(best_count, max_candidates);
    std::sort(best_idx.begin(), best_idx.begin() + static_cast<std::ptrdiff_t>(emitted_count));
    return emit_anchor_records(values, best_idx.data(), emitted_count, anchors);
}

size_t build_gamma_anchor_block(const uint64_t* values, size_t n, GammaBlockStats& stats,
                                PLAAnchor* anchors, size_t anchor_capacity) {
    assert(values != nullptr);
    assert(anchors != nullptr);
    assert(anchor_capacity <= kPLAMaxAnchors);
    assert(n <= kPLABlockSize);

    if (n <= 2) {
        return 0;
    }

    const size_t target_anchor_count = std::min(anchor_capacity, n - 2);
    if (target_anchor_count == 0) {
        return 0;
    }

    // Stage 1: Precompute block statistics and initialize the compact candidate-state map.
    prepare_gamma_stats(values, n, stats);
    initialize_states(stats, n);

    // Stage 2: Build and rank the initial pool of seed candidates.
    std::array<uint16_t, kGammaSeedPoolSize> seed_pool{};
    size_t seed_pool_count = 0;
    build_seed_pool(stats, n, seed_pool.data(), seed_pool_count);

    std::array<uint16_t, kPLAMaxAnchors> anchor_positions{};
    size_t anchor_count = select_initial_seeds(
            stats, seed_pool.data(), seed_pool_count, target_anchor_count, anchor_positions.data());

    // Stage 3: Mark the chosen seeds as anchors and suppress nearby flat/overlapping candidates.
    for (size_t i = 0; i < anchor_count; ++i) {
        set_state(stats, anchor_positions[i], CandidateState::Anchor);
    }
    for (size_t i = 0; i < anchor_count; ++i) {
        apply_anchor_suppression(stats, n, anchor_positions[i]);
    }

    // Stage 4: Repeatedly refine the current worst segment by inserting a better split anchor.
    while (anchor_count < target_anchor_count) {
        uint16_t split_position = 0;
        if (!choose_best_refinement(stats, n, anchor_positions.data(), anchor_count, split_position)) {
            break;
        }

        insert_sorted_position(split_position, anchor_positions.data(), anchor_count);
        set_state(stats, split_position, CandidateState::Anchor);
        if (is_jarring_position(stats, split_position)) {
            apply_anchor_suppression(stats, n, split_position);
        }
    }

    // Stage 5: Fill any remaining capacity conservatively, then emit the final anchor records.
    fill_remaining_anchors(stats, n, anchor_positions.data(), anchor_count, target_anchor_count);
    return emit_anchor_records(values, anchor_positions.data(), anchor_count, anchors);
}

size_t build_all_interior_anchor_block(const uint64_t* values, size_t n, PLAAnchor* anchors, size_t anchor_capacity) {
    assert(values != nullptr);
    assert(anchors != nullptr);

    if (n <= 2) {
        return 0;
    }

    const size_t interior_count = n - 2;
    assert(anchor_capacity >= interior_count);
    for (size_t i = 1; i + 1 < n; ++i) {
        const size_t anchor_index = i - 1;
        anchors[anchor_index].idx = static_cast<uint16_t>(i);
        anchors[anchor_index].val = values[i];
        anchors[anchor_index].dprev =
                static_cast<int32_t>(static_cast<int64_t>(values[i]) - static_cast<int64_t>(values[i - 1]));
    }
    return interior_count;
}

}  // namespace pallas
