#include "python_read.h"

#include <pallas/utils/pallas_storage.h>

std::vector<pallas::Thread*> Archive_get_threads(pallas::Archive& archive) {
    auto vector = std::vector<pallas::Thread*>();
    for (size_t i = 0; i < archive.nb_threads; ++i) {
        vector.push_back(archive.getThreadAt(i));
    }
    return vector;
}

std::map<pallas::StringRef, std::string>& Archive_get_strings(pallas::Archive& archive) {
    auto& map = *new std::map<pallas::StringRef, std::string>();
    for (auto& [key, r] : archive.definitions.strings) {
        map.insert(std::pair(key, archive.getString(r.string_ref)->str));
    }
    return map;
}

std::map<pallas::ThreadId, PyLocation>& Archive_get_locations(pallas::Archive& archive) {
    auto& map = *new std::map<pallas::ThreadId, PyLocation>();
    for (auto& loc : archive.locations) {
        map.insert(std::pair(loc.id, PyLocation{loc.id, archive.getString(loc.name)->str, archive.getLocationGroup(loc.parent)}));
    }
    return map;
}

std::map<pallas::RegionRef, PyRegion>& Archive_get_regions(pallas::Archive& archive) {
    auto& map = *new std::map<pallas::RegionRef, PyRegion>();
    for (auto& [key, r] : archive.definitions.regions) {
        map.insert(std::pair(key, PyRegion{r.region_ref, archive.getString(r.string_ref)->str}));
    }
    return map;
}

std::map<pallas::ThreadId, PyLocation>& Trace_get_locations(pallas::GlobalArchive& trace) {
    auto& map = *new std::map<pallas::ThreadId, PyLocation>();
    for (auto& lg : trace.location_groups) {
        auto a = trace.getArchive(lg.id);
        for (const auto& [key, value] : Archive_get_locations(*a)) {
            map.insert(std::pair(key, value));
        }
    }
    return map;
}

std::map<pallas::LocationGroupId, PyLocationGroup>& Trace_get_location_groups(pallas::GlobalArchive& trace) {
    auto& map = *new std::map<pallas::LocationGroupId, PyLocationGroup>();
    for (auto& lg : trace.location_groups) {
        map.insert(std::pair(lg.id, PyLocationGroup{lg.id, trace.getString(lg.name)->str, trace.getLocationGroup(lg.parent)}));
    }
    return map;
}

std::map<pallas::StringRef, std::string>& Trace_get_strings(pallas::GlobalArchive& trace) {
    auto& map = *new std::map<pallas::StringRef, std::string>();
    for (auto& [key, r] : trace.definitions.strings) {
        map.insert(std::pair(key, trace.getString(r.string_ref)->str));
    }
    return map;
}

std::map<pallas::RegionRef, PyRegion>& Trace_get_regions(pallas::GlobalArchive& trace) {
    auto& map = *new std::map<pallas::RegionRef, PyRegion>();
    for (auto& [key, r] : trace.definitions.regions) {
        map.insert(std::pair(key, PyRegion{r.region_ref, trace.getString(r.string_ref)->str}));
    }
    return map;
}

py::list& Trace_get_archives(pallas::GlobalArchive& trace) {
    auto& list = *new py::list(trace.location_groups.size());
    int i = 0;
    for (auto& locationGroup : trace.location_groups) {
        list[i++] = trace.getArchive(locationGroup.id);
    }
    return list;
}

pallas::GlobalArchive* open_trace(const std::string& path) {
    return pallas_open_trace(path.c_str());
}


std::vector<PySequence> threadGetSequences(pallas::Thread& self) {
    auto output = std::vector<PySequence>(self.nb_sequences);
    for (size_t i = 0; i < self.nb_sequences; i++) {
        output[i].self = &self.sequences[i];
        output[i].thread = &self;
    }
    return output;
}

std::vector<PyLoop> threadGetLoops(pallas::Thread& self) {
    auto output = std::vector<PyLoop>(self.nb_loops);
    for (size_t i = 0; i < self.nb_loops; i++) {
        output[i].self = &self.loops[i];
        output[i].thread = &self;
    }
    return output;
}

std::vector<PyEvent> threadGetEvents(pallas::Thread& self) {
    auto output = std::vector<PyEvent>(self.nb_events);
    for (size_t i = 0; i < self.nb_events; i++) {
        output[i].self = &self.events[i];
        output[i].thread = &self;
    }
    return output;
}

pybind11::list sequenceGetContent(const PySequence& self) {
    auto output = pybind11::list();
    // auto output = std::vector<PySomething>(self.self->tokens.size());
    for (size_t i = 0; i < self.self->tokens.size(); i++) {
        auto t = self.self->tokens[i];
        if (t.type == pallas::TypeEvent) {
            PyEvent temp = {self.thread->getEvent(t), self.thread};
            output.append(temp);
        }
        if (t.type == pallas::TypeSequence) {
            PySequence temp = {self.thread->getSequence(t), self.thread};
            output.append(temp);
        }
        if (t.type == pallas::TypeLoop) {
            PyLoop temp = {self.thread->getLoop(t), self.thread};
            output.append(temp);
        }
    }
    return output;
}

bool doesSequenceContains(const PySequence& self, pallas::Token t) {
    for (auto token : self.self->tokens) {
        if (t == token) {
            return true;
        }
    }
    for (auto token : self.self->tokens) {
        if (token.type == pallas::TypeSequence) {
            PySequence temp = {self.thread->getSequence(token), self.thread};
            if (doesSequenceContains(temp, t)) {
                return true;
            }
        }
        if (token.type == pallas::TypeLoop) {
            auto* loop = self.thread->getLoop(token);
            PySequence temp = {self.thread->getSequence(loop->repeated_token), self.thread};
            if (doesSequenceContains(temp, t)) {
                return true;
            }
        }
    }
    return false;
}

std::vector<PyEvent> threadGetEventsMatching(pallas::Thread& t, pallas::Record record) {
    auto output = std::vector<PyEvent>();
    for (size_t i = 0; i < t.nb_events; i++) {
        if (t.events[i].data.record == record) {
            output.push_back({&t.events[i], &t});
        }
    }
    return output;
}


std::vector<PyEvent> threadGetEventsMatchingList(pallas::Thread& t, std::vector<pallas::Record> records) {
    auto output = std::vector<PyEvent>();
    for (size_t i = 0; i < t.nb_events; i++) {
        for (auto record : records) {
            if (t.events[i].data.record == record) {
                output.push_back({&t.events[i], &t});
                continue;
            }
        }
    }
    return output;
}

py::tuple makePyObjectFromToken(pallas::Token t, pallas::ThreadReader& thread_reader) {
    switch (t.type) {
    case pallas::TypeEvent: {
        return py::make_tuple(
                std::variant<PyEvent, PySequence, PyLoop, pallas::Token>(
                        PyEvent{thread_reader.thread_trace->getEvent(t), thread_reader.thread_trace}),
                thread_reader.currentState.currentFrame->tokenCount[t]
                );
    }
    case pallas::TypeSequence: {
        return py::make_tuple(
                std::variant<PyEvent, PySequence, PyLoop, pallas::Token>(
                        PySequence{thread_reader.thread_trace->getSequence(t), thread_reader.thread_trace}),
                thread_reader.currentState.currentFrame->tokenCount[t]
                );
    }
    case pallas::TypeLoop: {
        return py::make_tuple(
                std::variant<PyEvent, PySequence, PyLoop, pallas::Token>(
                        PyLoop{thread_reader.thread_trace->getLoop(t), thread_reader.thread_trace}),
                thread_reader.currentState.currentFrame->tokenCount[t]
                );
    }
    default: {
        // pallas::TypeInvalid
        return py::make_tuple(
                std::variant<PyEvent, PySequence, PyLoop, pallas::Token>(
                        pallas::Token()),
                0
                );
    }
    }
}
