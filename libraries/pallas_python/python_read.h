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

struct PyLinkedVector {
    pallas::TimeLV* linked_vector;
    pallas::DurationLV* linked_duration_vector;

    [[nodiscard]] size_t size() const {
        return linked_vector ? linked_vector->size() : linked_duration_vector->size();
    }

    [[nodiscard]] pallas::StoragePolicy preferred_subarray_policy() const {
        return linked_vector ? linked_vector->get_storage_policy()
                             : linked_duration_vector->get_storage_policy();
    }

    [[nodiscard]] std::vector<pallas::StoragePolicy> subarray_policies() const {
        return linked_vector ? linked_vector->get_sub_array_policies()
                             : linked_duration_vector->get_sub_array_policies();
    }

    [[nodiscard]] std::vector<pallas::StoragePolicy> loaded_subarray_policies() const {
        return linked_vector ? linked_vector->get_loaded_sub_array_policies()
                             : linked_duration_vector->get_loaded_sub_array_policies();
    }

    [[nodiscard]] uint64_t at(size_t index) const {
        return linked_vector ? linked_vector->at(index) : linked_duration_vector->at(index);
    }
};

struct PyLinkedVectorIterator {
    pallas::TimeLV* linked_vector;
    pallas::DurationLV* linked_duration_vector;
    size_t index;
};

struct PyThreadIterator {
    pallas::ThreadReader *inner;
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
