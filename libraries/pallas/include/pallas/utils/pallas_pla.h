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

struct PLAAnchor {
    uint64_t val = 0;
    int32_t dprev = 0;
    uint16_t idx = 0;
};

struct PLAWorkspace {
    uint64_t* raw = nullptr;
    int64_t* delta = nullptr;
    uint64_t* score = nullptr;
    uint16_t* order = nullptr;
};

size_t pla_helper_buffer_bytes();
PLAWorkspace bind_pla_workspace(void* buffer);

size_t build_pla4_alpha_block(const uint64_t* values,
                              size_t n,
                              PLAWorkspace& workspace,
                              PLAAnchor* anchors,
                              size_t anchor_capacity);

}
