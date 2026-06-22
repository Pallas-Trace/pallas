/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include <algorithm>
#include <cassert>
#include <cstring>
#include <vector>

#include "pallas/utils/pallas_pla.h"

namespace pallas {

namespace {

uint64_t abs_i64(int64_t value) {
    return static_cast<uint64_t>(value < 0 ? -value : value);
}

void insert_best_candidate(uint16_t idx, uint64_t score, std::vector<uint16_t>& best_idx, 
                           std::vector<uint64_t>& best_score, size_t& best_count) {
    size_t insert_at = best_count;
    while (insert_at > 0) {
        const size_t prev = insert_at - 1;
        if (score < best_score[prev]) {
            break;
        }
        if (score == best_score[prev] && idx > best_idx[prev]) {
            break;
        }
        if (insert_at < best_idx.size()) {
            best_idx[insert_at] = best_idx[prev];
            best_score[insert_at] = best_score[prev];
        }
        insert_at = prev;
    }

    if (insert_at < best_idx.size()) {
        best_idx[insert_at] = idx;
        best_score[insert_at] = score;
        if (best_count < best_idx.size()) {
            ++best_count;
        }
    }
}

}  // namespace

size_t pla_helper_buffer_bytes() {
    return kPLABlockSize * sizeof(uint64_t) + kPLABlockSize * sizeof(int64_t) +
           kPLABlockSize * sizeof(uint64_t) + kPLABlockSize * sizeof(uint16_t);
}

PLAWorkspace bind_pla_workspace(void* buffer) {
    auto* bytes = static_cast<uint8_t*>(buffer);
    PLAWorkspace workspace{};
    workspace.raw = reinterpret_cast<uint64_t*>(bytes);
    bytes += kPLABlockSize * sizeof(uint64_t);
    workspace.delta = reinterpret_cast<int64_t*>(bytes);
    bytes += kPLABlockSize * sizeof(int64_t);
    workspace.score = reinterpret_cast<uint64_t*>(bytes);
    bytes += kPLABlockSize * sizeof(uint64_t);
    workspace.order = reinterpret_cast<uint16_t*>(bytes);
    return workspace;
}

size_t build_pla4_alpha_block(const uint64_t* values, size_t n,
                              PLAWorkspace& workspace, PLAAnchor* anchors,
                              size_t anchor_capacity) {
    assert(values != nullptr);
    assert(anchors != nullptr);
    assert(anchor_capacity >= 4);
    assert(n >= 6);

    for (size_t i = 1; i < n; ++i) {
        workspace.delta[i] = static_cast<int64_t>(values[i]) - static_cast<int64_t>(values[i - 1]);
    }

    const size_t max_candidates = 4;
    std::vector<uint16_t> best_idx(max_candidates, 0);
    std::vector<uint64_t> best_score(max_candidates, 0);
    size_t best_count = 0;

    for (size_t i = 1; i + 1 < n; ++i) {
        const size_t left = (i > 5) ? (i - 5) : 1;
        const size_t right = ((i + 5) < n) ? (i + 5) : (n - 1);
        int64_t sum = 0;
        size_t count = 0;
        for (size_t j = left; j <= right; ++j) {
            if (j == i) {
                continue;
            }
            sum += workspace.delta[j];
            ++count;
        }

        int64_t baseline = 0;
        if (count != 0) {
            baseline = sum / static_cast<int64_t>(count);
        }

        const int64_t current_delta = workspace.delta[i];
        const int64_t left_delta = workspace.delta[i - 1];
        const int64_t right_delta = workspace.delta[i + 1];
        const uint64_t score =
                abs_i64(current_delta - baseline) +
                abs_i64(current_delta - left_delta) +
                abs_i64(right_delta - current_delta);
        workspace.score[i] = score;
        if (max_candidates > 0) {
            insert_best_candidate(static_cast<uint16_t>(i), score, best_idx, best_score, best_count);
        }
    }

    const size_t emitted_count = std::min(best_count, max_candidates);
    std::sort(best_idx.begin(), best_idx.begin() + emitted_count);
    for (size_t i = 0; i < emitted_count; ++i) {
        const uint16_t idx = best_idx[i];
        anchors[i].idx = idx;
        anchors[i].val = values[idx];
        anchors[i].dprev =
                static_cast<int32_t>(static_cast<int64_t>(values[idx]) - static_cast<int64_t>(values[idx - 1]));
    }
    return emitted_count;
}

}
