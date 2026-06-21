/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <pallas/pallas.h>
#include <pallas/pallas_archive.h>
#include "pallas/utils/pallas_timestamp.h"

namespace py = pybind11;

struct QuantaRes {
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

QuantaRes calc_quanta_base(pallas::GlobalArchive& trace,
                           py::array_t<uint32_t, py::array::c_style | py::array::forcecast> thread_ids,
                           py::array_t<uint64_t, py::array::c_style | py::array::forcecast> bin_edges_ns,
                           const std::string& mode = "fast");

void setup_quanta(py::module_& m, py::class_<pallas::GlobalArchive>& trace_cls);

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */
