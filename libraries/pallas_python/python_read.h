/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#pragma once
#include <pallas/pallas.h>
#include <pallas/pallas_archive.h>
#include <pallas/pallas_record.h>

#include "python_tokens.h"

#include <pybind11/pybind11.h>

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
    pallas::LinkedVector* linked_vector;
    pallas::LinkedDurationVector* linked_duration_vector;
};

struct PyThreadIterator {
    pallas::ThreadReader inner;
};


std::vector<pallas::Thread*> Archive_get_threads(pallas::Archive& archive);

std::map<pallas::StringRef, std::string>& Archive_get_strings(pallas::Archive& archive);

std::map<pallas::ThreadId, PyLocation>& Archive_get_locations(pallas::Archive& archive);

std::map<pallas::RegionRef, PyRegion>& Archive_get_regions(pallas::Archive& archive);

std::map<pallas::ThreadId, PyLocation>& Trace_get_locations(pallas::GlobalArchive& trace);

std::map<pallas::LocationGroupId, PyLocationGroup>& Trace_get_location_groups(pallas::GlobalArchive& trace);

std::map<pallas::StringRef, std::string>& Trace_get_strings(pallas::GlobalArchive& trace);

std::map<pallas::RegionRef, PyRegion>& Trace_get_regions(pallas::GlobalArchive& trace);

py::list& Trace_get_archives(pallas::GlobalArchive& trace);

pallas::GlobalArchive* open_trace(const std::string& path);

std::vector<PySequence> threadGetSequences(pallas::Thread& self);

std::vector<PyLoop> threadGetLoops(pallas::Thread& self);

std::vector<PyEvent> threadGetEvents(pallas::Thread& self);

pybind11::list sequenceGetContent(const PySequence& self);

bool doesSequenceContains(const PySequence& self, pallas::Token t);

std::vector<PyEvent> threadGetEventsMatching(pallas::Thread& t, pallas::Record record);

std::vector<PyEvent> threadGetEventsMatchingList(pallas::Thread& t, std::vector<pallas::Record> records);

py::tuple makePyObjectFromToken(pallas::Token t, pallas::ThreadReader& thread_reader);
