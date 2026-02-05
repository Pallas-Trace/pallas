/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#pragma once
#include <pallas/pallas.h>
#include <pallas/pallas_archive.h>
#include <pallas/pallas_record.h>

#include "python_read.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array_t<uint64_t> get_communication_matrix(pallas::GlobalArchive& trace);

py::array_t<uint64_t> get_communication_matrix_timed(pallas::GlobalArchive& trace, pallas_timestamp_t start, pallas_timestamp_t end);

std::map<uint64_t, uint64_t> get_message_size_histogram(pallas::GlobalArchive& trace, bool count_data_amount = false);

std::map<uint64_t, uint64_t> get_message_size_histogram_local(pallas::Archive& archive, bool count_data_amount = false);

py::array_t<uint64_t> get_communication_over_time(pallas::GlobalArchive& trace, py::array_t<uint64_t> timestamps, bool count_messages = false);

py::array_t<uint64_t> get_communication_over_time_archive(pallas::Archive& archive, py::array_t<uint64_t> timestamps, bool count_messages = false);

py::object get_sequences_statistics(pallas::Thread& thread);
