/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <pallas/pallas.h>
#include <pallas/pallas_archive.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

struct PySequence {
    pallas::Sequence* self;
    pallas::Thread* thread;
};

struct PyLoop {
    pallas::Loop* self;
    pallas::Thread* thread;
};

struct PyEvent {
    pallas::Event* self;
    pallas::Thread* thread;
};

struct TokenMetaRes {
    std::vector<uint8_t> token_type;
    std::vector<pallas::TokenId> token_id;
    std::vector<std::string> display_name;

    size_t size() const { return token_id.size(); }

    void reserve(size_t n) {
        token_type.reserve(n);
        token_id.reserve(n);
        display_name.reserve(n);
    }
};

std::string Token_toString(pallas::Token t);
py::dict& EventData_get_data(pallas::EventData* data);

TokenMetaRes Trace_get_tokens(pallas::GlobalArchive& trace);
void setup_tokens(py::module_& m, py::class_<pallas::GlobalArchive>& trace_cls);
