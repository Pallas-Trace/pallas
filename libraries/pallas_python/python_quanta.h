/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <pallas/pallas.h>
#include <pallas/pallas_archive.h>
#include "pallas/utils/pallas_timestamp.h"

namespace py = pybind11;

namespace quanta {

enum class Mode {
    Fast,      // getSnapshotViewFast
    Balanced,  // getSnapshotView
    Exact,     // reader walk
};

constexpr const char* MODE_LIST = "fast, balanced, exact";

constexpr uint8_t OTHER_TOKEN_TYPE = 255;
constexpr uint64_t OTHER_TOKEN_ID = 0;

using ThreadIdsArg = py::array_t<uint32_t, py::array::c_style | py::array::forcecast>;
using BinEdgesArg = py::array_t<uint64_t, py::array::c_style | py::array::forcecast>;

using TokenTotals = std::map<pallas::Token, uint64_t>;

struct ResultRow {
    uint64_t start_ns;
    uint64_t finish_ns;
    uint32_t thread_id;
    uint8_t token_type;
    uint64_t token_id;
    uint64_t excl_ns;
    float proportion;
};

struct Result {
    std::vector<pallas::ThreadId> thread_id;
    std::vector<pallas_timestamp_t> start_ns;
    std::vector<pallas_timestamp_t> finish_ns;
    std::vector<uint8_t> token_type;
    std::vector<pallas::TokenId> token_id;
    std::vector<pallas_timestamp_t> excl_ns;
    std::vector<float> proportion;

    size_t size() const { return thread_id.size(); }

    void reserve(size_t n) {
        thread_id.reserve(n);
        start_ns.reserve(n);
        finish_ns.reserve(n);
        token_type.reserve(n);
        token_id.reserve(n);
        excl_ns.reserve(n);
        proportion.reserve(n);
    }
};

Result calc(pallas::GlobalArchive& trace, ThreadIdsArg thread_ids, BinEdgesArg bin_edges_ns, const std::string& mode = "fast", int top_k = -1);

}  // namespace quanta

void setup_quanta(py::module_& m, py::class_<pallas::GlobalArchive>& trace_cls);

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */
