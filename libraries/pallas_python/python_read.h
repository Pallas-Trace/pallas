/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#pragma once
#include <pallas/pallas.h>
#include <pallas/pallas_archive.h>
#include <pallas/pallas_record.h>

#include "pallas/pallas_read.h"
#include "python_tokens.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

struct PyLocationGroup {
    const pallas::LocationGroupId id;
    const std::string name;
    const pallas::LocationGroup* parent = {nullptr};
};

struct PyLocation {
    const pallas::ThreadId id;
    const std::string name;
    const pallas::LocationGroup* parent;
};

struct PyRegion {
    const pallas::RegionRef id;
    const std::string name;
};

struct PyMetricMember {
    const pallas::MetricMemberRef id;
    const std::string name;
    const std::string description;
    const std::string unit;
    const pallas::MetricType type;
    const pallas::MetricMode mode;
    const pallas::Base base;
    const int64_t exponent;
};

struct PyMetric {
    const pallas::MetricRef id;
    const std::vector<pallas::MetricMemberRef> members;
};

struct PyLinkedVector {
    pallas::LinkedVector* linked_vector;
    pallas::LinkedDurationVector* linked_duration_vector;
};

struct PyLinkedVectorIterator {
    pallas::LinkedVector* linked_vector;
    pallas::LinkedDurationVector* linked_duration_vector;
    size_t index;
};

struct PyThreadIterator {
    pallas::ThreadReader *inner;
    bool is_first_event;
    ~PyThreadIterator() {
        delete inner;
    }
};

struct PyTraceIterator {
    pallas::MultiThreadReader *inner;
    ~PyTraceIterator() {
        delete inner;
    }
};


std::vector<pallas::Thread*> Archive_get_threads(pallas::Archive& archive);

std::map<pallas::StringRef, std::string> Archive_get_strings(pallas::Archive& archive);

std::map<pallas::ThreadId, PyLocation> Archive_get_locations(pallas::Archive& archive);

std::map<pallas::RegionRef, PyRegion> Archive_get_regions(pallas::Archive& archive);

std::map<pallas::ThreadId, PyLocation> Trace_get_locations(pallas::GlobalArchive& trace);

std::map<pallas::LocationGroupId, PyLocationGroup> Trace_get_location_groups(pallas::GlobalArchive& trace);

std::map<pallas::StringRef, std::string> Trace_get_strings(pallas::GlobalArchive& trace);

std::map<pallas::RegionRef, PyRegion> Trace_get_regions(pallas::GlobalArchive& trace);

std::map<pallas::MetricMemberRef, PyMetricMember> Archive_get_metric_members(pallas::Archive& archive);

std::map<pallas::MetricRef, PyMetric> Archive_get_metrics(pallas::Archive& archive);

std::map<pallas::MetricMemberRef, PyMetricMember> Trace_get_metric_members(pallas::GlobalArchive& trace);

std::map<pallas::MetricRef, PyMetric> Trace_get_metrics(pallas::GlobalArchive& trace);

py::dict threadGetMetrics(pallas::Thread& thread);

py::list MetricValues_to_list(uint8_t nb_values, const pallas::MetricValue* values);

py::list* Trace_get_archives(pallas::GlobalArchive& trace);

pallas::GlobalArchive* open_trace(const std::string& path);

std::vector<PySequence> threadGetSequences(pallas::Thread& self);

std::vector<PyLoop> threadGetLoops(pallas::Thread& self);

std::vector<PyEvent> threadGetEvents(pallas::Thread& self);

pybind11::list sequenceGetContent(const PySequence& self);

bool doesSequenceContains(const PySequence& self, pallas::Token t);

std::vector<PyEvent> threadGetEventsMatching(pallas::Thread& t, pallas::Record record);

std::vector<PyEvent> threadGetEventsMatchingList(pallas::Thread& t, std::vector<pallas::Record> records);

py::tuple makePyObjectFromToken(pallas::Token t, pallas::ThreadReader& thread_reader);

py::array_t<uint64_t> linked_vector_to_numpy(PyLinkedVector& self);

std::vector<py::tuple> thread_reader_get_callstack(pallas::ThreadReader& self);

int get_read_flags_from_bools(bool enter_sequence, bool enter_loop);

py::dict get_attributes(PyEvent &event, size_t occurrence);
