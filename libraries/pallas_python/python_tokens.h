/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#pragma once
#include <pallas/pallas.h>

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

std::string Token_toString(pallas::Token t);
py::dict& EventData_get_data(pallas::EventData* data);
