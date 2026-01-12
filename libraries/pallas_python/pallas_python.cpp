/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
#include "pallas_python.h"

std::string Token_toString(pallas::Token t) {
    std::string out;
    switch (t.type) {
    case pallas::TypeInvalid:
        out += "U";
        break;
    case pallas::TypeEvent:
        out += "E";
        break;
    case pallas::TypeSequence:
        out += "S";
        break;
    case pallas::TypeLoop:
        out += "L";
        break;
    }
    out += std::to_string(t.id);
    return out;
}

#define READ(data, cursor, type, name)                              \
    {                                                               \
        type name;                                                  \
        pallas_event_pop_data(data, &name, sizeof(type), &cursor);  \
        dict[#name] = name;                                         \
    }

static py::dict& EventData_get_data(pallas::EventData* data) {
    auto& dict = *new py::dict();
    byte* cursor = nullptr;

    switch (data->record) {
    case pallas::PALLAS_EVENT_ENTER:
    case pallas::PALLAS_EVENT_LEAVE: {
        READ(data, cursor, pallas::RegionRef, region_ref);
        break;
    }
    case pallas::PALLAS_EVENT_THREAD_FORK:
    case pallas::PALLAS_EVENT_OMP_FORK: {
        READ(data, cursor, uint32_t, numberOfRequestedThreads);
        break;
    }
    case pallas::PALLAS_EVENT_MPI_SEND:
    case pallas::PALLAS_EVENT_MPI_ISEND: {
        READ(data, cursor, uint32_t, receiver);
        READ(data, cursor, uint32_t, communicator);
        READ(data, cursor, uint32_t, msgTag);
        READ(data, cursor, uint64_t, msgLength);
        if (data->record == pallas::PALLAS_EVENT_MPI_ISEND)
            READ(data, cursor, uint64_t, requestID);
        break;
    }
    case pallas::PALLAS_EVENT_MPI_RECV:
    case pallas::PALLAS_EVENT_MPI_IRECV: {
        READ(data, cursor, uint32_t, sender);
        READ(data, cursor, uint32_t, communicator);
        READ(data, cursor, uint32_t, msgTag);
        READ(data, cursor, uint64_t, msgLength);
        if (data->record == pallas::PALLAS_EVENT_MPI_IRECV)
            READ(data, cursor, uint64_t, requestID);
        break;
    }
    case pallas::PALLAS_EVENT_MPI_ISEND_COMPLETE:
    case pallas::PALLAS_EVENT_MPI_IRECV_REQUEST: {
        READ(data, cursor, uint64_t, requestID);
        break;
    }
    case pallas::PALLAS_EVENT_THREAD_ACQUIRE_LOCK:
    case pallas::PALLAS_EVENT_THREAD_RELEASE_LOCK:
    case pallas::PALLAS_EVENT_OMP_ACQUIRE_LOCK:
    case pallas::PALLAS_EVENT_OMP_RELEASE_LOCK: {
        READ(data, cursor, uint32_t, lockID);
        READ(data, cursor, uint32_t, acquisitionOrder);
        break;
    }
    case pallas::PALLAS_EVENT_MPI_COLLECTIVE_END: {
        READ(data, cursor, uint32_t, collectiveOp);
        READ(data, cursor, uint32_t, communicator);
        READ(data, cursor, uint32_t, root);
        READ(data, cursor, uint64_t, sizeSent);
        READ(data, cursor, uint64_t, sizeReceived);
        break;
    }
    case pallas::PALLAS_EVENT_OMP_TASK_CREATE:
    case pallas::PALLAS_EVENT_OMP_TASK_SWITCH:
    case pallas::PALLAS_EVENT_OMP_TASK_COMPLETE: {
        READ(data, cursor, uint64_t, taskID);
        break;
    }
    case pallas::PALLAS_EVENT_GENERIC: {
        pallas::StringRef event_name;
        pallas_event_pop_data(data, &event_name, sizeof(event_name), &cursor);
        dict["event_name"] = event_name;
        break;
    }
    case pallas::PALLAS_EVENT_THREAD_BEGIN:
    case pallas::PALLAS_EVENT_THREAD_END:
    case pallas::PALLAS_EVENT_THREAD_TEAM_BEGIN:
    case pallas::PALLAS_EVENT_THREAD_TEAM_END:
    case pallas::PALLAS_EVENT_THREAD_JOIN:
    case pallas::PALLAS_EVENT_MPI_COLLECTIVE_BEGIN:
    case pallas::PALLAS_EVENT_THREAD_TASK_CREATE:
    case pallas::PALLAS_EVENT_THREAD_TASK_COMPLETE:
    case pallas::PALLAS_EVENT_THREAD_TASK_SWITCH:
        // No additional data for these events
    default:
        break;
    }

    return dict;
}

void setupEnums(const py::module_& m) {
    py::enum_<pallas::TokenType>(m, "TokenType")
            .value("INVALID", pallas::TypeInvalid)
            .value("EVENT", pallas::TypeEvent)
            .value("SEQUENCE", pallas::TypeSequence)
            .value("LOOP", pallas::TypeLoop)
            .export_values();

    py::enum_<pallas::Record>(m, "Record")
            .value("BUFFER_FLUSH", pallas::PALLAS_EVENT_BUFFER_FLUSH)
            .value("MEASUREMENT_ON_OFF", pallas::PALLAS_EVENT_MEASUREMENT_ON_OFF)
            .value("ENTER", pallas::PALLAS_EVENT_ENTER)
            .value("LEAVE", pallas::PALLAS_EVENT_LEAVE)
            .value("MPI_SEND", pallas::PALLAS_EVENT_MPI_SEND)
            .value("MPI_ISEND", pallas::PALLAS_EVENT_MPI_ISEND)
            .value("MPI_ISEND_COMPLETE", pallas::PALLAS_EVENT_MPI_ISEND_COMPLETE)
            .value("MPI_IRECV_REQUEST", pallas::PALLAS_EVENT_MPI_IRECV_REQUEST)
            .value("MPI_RECV", pallas::PALLAS_EVENT_MPI_RECV)
            .value("MPI_IRECV", pallas::PALLAS_EVENT_MPI_IRECV)
            .value("MPI_REQUEST_TEST", pallas::PALLAS_EVENT_MPI_REQUEST_TEST)
            .value("MPI_REQUEST_CANCELLED", pallas::PALLAS_EVENT_MPI_REQUEST_CANCELLED)
            .value("MPI_COLLECTIVE_BEGIN", pallas::PALLAS_EVENT_MPI_COLLECTIVE_BEGIN)
            .value("MPI_COLLECTIVE_END", pallas::PALLAS_EVENT_MPI_COLLECTIVE_END)
            .value("OMP_FORK", pallas::PALLAS_EVENT_OMP_FORK)
            .value("OMP_JOIN", pallas::PALLAS_EVENT_OMP_JOIN)
            .value("OMP_ACQUIRE_LOCK", pallas::PALLAS_EVENT_OMP_ACQUIRE_LOCK)
            .value("OMP_RELEASE_LOCK", pallas::PALLAS_EVENT_OMP_RELEASE_LOCK)
            .value("OMP_TASK_CREATE", pallas::PALLAS_EVENT_OMP_TASK_CREATE)
            .value("OMP_TASK_SWITCH", pallas::PALLAS_EVENT_OMP_TASK_SWITCH)
            .value("OMP_TASK_COMPLETE", pallas::PALLAS_EVENT_OMP_TASK_COMPLETE)
            .value("METRIC", pallas::PALLAS_EVENT_METRIC)
            .value("PARAMETER_STRING", pallas::PALLAS_EVENT_PARAMETER_STRING)
            .value("PARAMETER_INT", pallas::PALLAS_EVENT_PARAMETER_INT)
            .value("PARAMETER_UNSIGNED_INT", pallas::PALLAS_EVENT_PARAMETER_UNSIGNED_INT)
            .value("THREAD_FORK", pallas::PALLAS_EVENT_THREAD_FORK)
            .value("THREAD_JOIN", pallas::PALLAS_EVENT_THREAD_JOIN)
            .value("THREAD_TEAM_BEGIN", pallas::PALLAS_EVENT_THREAD_TEAM_BEGIN)
            .value("THREAD_TEAM_END", pallas::PALLAS_EVENT_THREAD_TEAM_END)
            .value("THREAD_ACQUIRE_LOCK", pallas::PALLAS_EVENT_THREAD_ACQUIRE_LOCK)
            .value("THREAD_RELEASE_LOCK", pallas::PALLAS_EVENT_THREAD_RELEASE_LOCK)
            .value("THREAD_TASK_CREATE", pallas::PALLAS_EVENT_THREAD_TASK_CREATE)
            .value("THREAD_TASK_SWITCH", pallas::PALLAS_EVENT_THREAD_TASK_SWITCH)
            .value("THREAD_TASK_COMPLETE", pallas::PALLAS_EVENT_THREAD_TASK_COMPLETE)
            .value("THREAD_CREATE", pallas::PALLAS_EVENT_THREAD_CREATE)
            .value("THREAD_BEGIN", pallas::PALLAS_EVENT_THREAD_BEGIN)
            .value("THREAD_WAIT", pallas::PALLAS_EVENT_THREAD_WAIT)
            .value("THREAD_END", pallas::PALLAS_EVENT_THREAD_END)
            .value("IO_CREATE_HANDLE", pallas::PALLAS_EVENT_IO_CREATE_HANDLE)
            .value("IO_DESTROY_HANDLE", pallas::PALLAS_EVENT_IO_DESTROY_HANDLE)
            .value("IO_DUPLICATE_HANDLE", pallas::PALLAS_EVENT_IO_DUPLICATE_HANDLE)
            .value("IO_SEEK", pallas::PALLAS_EVENT_IO_SEEK)
            .value("IO_CHANGE_STATUS_FLAGS", pallas::PALLAS_EVENT_IO_CHANGE_STATUS_FLAGS)
            .value("IO_DELETE_FILE", pallas::PALLAS_EVENT_IO_DELETE_FILE)
            .value("IO_OPERATION_BEGIN", pallas::PALLAS_EVENT_IO_OPERATION_BEGIN)
            .value("IO_OPERATION_TEST", pallas::PALLAS_EVENT_IO_OPERATION_TEST)
            .value("IO_OPERATION_ISSUED", pallas::PALLAS_EVENT_IO_OPERATION_ISSUED)
            .value("IO_OPERATION_COMPLETE", pallas::PALLAS_EVENT_IO_OPERATION_COMPLETE)
            .value("IO_OPERATION_CANCELLED", pallas::PALLAS_EVENT_IO_OPERATION_CANCELLED)
            .value("IO_ACQUIRE_LOCK", pallas::PALLAS_EVENT_IO_ACQUIRE_LOCK)
            .value("IO_RELEASE_LOCK", pallas::PALLAS_EVENT_IO_RELEASE_LOCK)
            .value("IO_TRY_LOCK", pallas::PALLAS_EVENT_IO_TRY_LOCK)
            .value("PROGRAM_BEGIN", pallas::PALLAS_EVENT_PROGRAM_BEGIN)
            .value("PROGRAM_END", pallas::PALLAS_EVENT_PROGRAM_END)
            .value("NON_BLOCKING_COLLECTIVE_REQUEST", pallas::PALLAS_EVENT_NON_BLOCKING_COLLECTIVE_REQUEST)
            .value("NON_BLOCKING_COLLECTIVE_COMPLETE", pallas::PALLAS_EVENT_NON_BLOCKING_COLLECTIVE_COMPLETE)
            .value("COMM_CREATE", pallas::PALLAS_EVENT_COMM_CREATE)
            .value("COMM_DESTROY", pallas::PALLAS_EVENT_COMM_DESTROY)
            .value("GENERIC", pallas::PALLAS_EVENT_GENERIC)
            .export_values();
}

std::vector<pallas::Thread*> Archive_get_threads(pallas::Archive& archive) {
    auto vector = std::vector<pallas::Thread*>();
    for (size_t i = 0; i < archive.nb_threads; ++i) {
        vector.push_back(archive.getThreadAt(i));
    }
    return vector;
}

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

auto makePyObjectFromToken(pallas::Token t, pallas::ThreadReader& thread_reader) {
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
    default: { // pallas::TypeInvalid
        return py::make_tuple(
                std::variant<PyEvent, PySequence, PyLoop, pallas::Token>(
                    pallas::Token()),
                    0
                );
    }
    }
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "Python API for the Pallas library";

    setupEnums(m);

    py::class_<pallas::Token>(m, "Token", "A Pallas token")
            .def_property_readonly("id", [](pallas::Token t) { return t.id; })
            .def_property_readonly("type", [](pallas::Token t) { return t.type; })
            .def("__repr__", &Token_toString)
            .def("__hash__", [](pallas::Token& self) { return *reinterpret_cast<uint32_t*>(&self); })
            .def("__eq__", [](pallas::Token& self, pallas::Token& other) { return self.type == other.type && self.id == other.id; });

    py::class_<PyLinkedVector>(m, "Vector", "A Pallas custom vector")
            .def_property_readonly("size", [](PyLinkedVector self) { return self.linked_vector ? self.linked_vector->size : self.linked_duration_vector->size; })
            .def("__getitem__", [](PyLinkedVector self, int i) { return self.linked_vector ? self.linked_vector->at(i) : self.linked_duration_vector->at(i); })
            .def("as_numpy_array", [](PyLinkedVector& self) {
                py::capsule free_when_done(&self, [](void* f) {
                });

                if (self.linked_vector)
                    return py::array_t<uint64_t>(
                            {self.linked_vector->size},
                            {sizeof(uint64_t)},
                            &self.linked_vector->at(0),
                            free_when_done
                            );
                else
                    return py::array_t<uint64_t>(
                            {self.linked_duration_vector->size},
                            {sizeof(uint64_t)},
                            &self.linked_duration_vector->at(0),
                            free_when_done
                            );
            });

    py::class_<PySequence>(m, "Sequence", "A Pallas Sequence, ie a group of tokens.")
            .def_property_readonly("id", [](const PySequence& self) { return self.self->id; })
            .def_property_readonly("tokens", [](const PySequence& self) { return self.self->tokens; })
            .def_property_readonly("content", [](const PySequence& self) { return sequenceGetContent(self); })
            .def_property_readonly("n_iterations", [](const PySequence& self) { return self.self->durations->size; })
            .def_property_readonly("timestamps", [](const PySequence& self) { return PyLinkedVector{self.self->timestamps, nullptr}; })
            .def_property_readonly("durations", [](const PySequence& self) { return PyLinkedVector{nullptr, self.self->durations}; })
            .def_property_readonly("exclusive_durations", [](const PySequence& self) { return PyLinkedVector{nullptr, self.self->exclusive_durations}; })
            .def_property_readonly("max_duration", [](const PySequence& self) { return self.self->durations->max; })
            .def_property_readonly("min_duration", [](const PySequence& self) { return self.self->durations->min; })
            .def_property_readonly("mean_duration", [](const PySequence& self) { return self.self->durations->mean; })
            .def("contains", [](const PySequence& self, const PySequence& other) { return doesSequenceContains(self, other.self->id); })
            .def("contains", [](const PySequence& self, const PyLoop& other) { return doesSequenceContains(self, other.self->self_id); })
            .def("contains", [](const PySequence& self, const PyEvent& other) { return doesSequenceContains(self, {pallas::TokenType::TypeEvent, other.self->id}); })
            .def("contains", [](const PySequence& self, const pallas::Token& other) { return doesSequenceContains(self, other); })
            .def("guessName", [](const PySequence& self) { return self.self->guessName(self.thread); })
            .def("__repr__", [](const PySequence& self) { return "<pallas_python.Sequence " + std::to_string(self.self->id.id) + ">"; });

    py::class_<PyLoop>(m, "Loop", "A Pallas Loop, ie a repetition of a Sequence token.")
            .def_property_readonly("id", [](const PyLoop& self) { return self.self->self_id; })
            .def_property_readonly("sequence", [](const PyLoop& self) { return PySequence{self.thread->getSequence(self.self->repeated_token), self.thread}; })
            .def_property_readonly("nb_iterations", [](const PyLoop& self) { return self.self->nb_iterations; })
            .def("__repr__", [](const PyLoop& self) { return "<pallas_python.Loop " + std::to_string(self.self->self_id.id) + ">"; });

    py::class_<PyEvent>(m, "Event", "A Pallas Event.")
            .def_property_readonly("id", [](const PyEvent& self) { return pallas::Token(pallas::TypeEvent, self.self->id); })
            .def_property_readonly("record", [](const PyEvent& self) { return self.self->data.record; })
            .def_property_readonly("data", [](const PyEvent& self) { return EventData_get_data(&self.self->data); })
            .def_property_readonly("nb_occurrences", [](const PyEvent& self) { return self.self->nb_occurrences; })
            .def_property_readonly("timestamps", [](const PyEvent& self) { return PyLinkedVector{self.self->timestamps, nullptr}; })
            .def("guessName",[](const PyEvent& self) {return self.thread->getEventString(&self.self->data);})
            .def("__repr__", [](const PyEvent& self) { return "<pallas_python.Event " + std::to_string(self.self->id) + ">"; });

    py::class_<pallas::Thread>(m, "Thread", "A Pallas thread.")
            .def_readonly("id", &pallas::Thread::id)
            .def_property_readonly("starting_timestamp", [](const pallas::Thread& self) { return self.first_timestamp; })
            .def_property_readonly("finish_timestamp", [](const pallas::Thread& self) { return (self.first_timestamp + self.sequences[0].durations->at(0)); })
            .def_property_readonly("events", [](pallas::Thread& self) { return threadGetEvents(self); })
            .def_property_readonly("sequences", [](pallas::Thread& self) { return threadGetSequences(self); })
            .def_property_readonly("loops", [](pallas::Thread& self) { return threadGetLoops(self); })
            .def("get_events_from_record", threadGetEventsMatching)
            .def("get_events_from_record", threadGetEventsMatchingList)
            .def("__repr__", [](const pallas::Thread& self) { return "<pallas_python.Thread " + std::to_string(self.id) + ">"; })
            .def("getSnapshotView", &pallas::Thread::getSnapshotView)
            .def("getSnapshotViewFast", &pallas::Thread::getSnapshotViewFast)
            .def("__iter__", [](const pallas::Thread& self) {
                auto inner = pallas::ThreadReader(self.archive, self.id, PALLAS_READ_FLAG_UNROLL_ALL);
                return PyThreadIterator{inner};
            })
            .def("reader", [](const pallas::Thread& self) {
                return new pallas::ThreadReader(self.archive, self.id, PALLAS_READ_FLAG_UNROLL_ALL);
            });

    py::class_<PyThreadIterator>(m, "Thread_Iterator", "An iterator over the thread.")
            .def("__next__", [](PyThreadIterator& self) {
                if (self.inner.moveToNextToken()) {
                    auto t = self.inner.pollCurToken();
                    if (!t.isValid()) {
                        throw py::stop_iteration();
                    }
                    return makePyObjectFromToken(t, self.inner);
                }
                throw py::stop_iteration();
            });
    py::class_<pallas::ThreadReader>(m, "ThreadReader", "A helper structure to read a thread")
            .def_property_readonly("callstack", [](pallas::ThreadReader& self) {
                std::vector<py::tuple> res;
                res.reserve(self.currentState.current_frame_index);
                for (int i = 1; i <= self.currentState.current_frame_index; i++) {
                    res.push_back(
                        makePyObjectFromToken(
                            self.currentState.callstack[i].callstack_iterable,
                            self
                        )
                    );
                }
                res.push_back(makePyObjectFromToken(self.pollCurToken(), self));
                return res;
            }, py::keep_alive<0, 1>())
            .def("moveToNextToken", [](pallas::ThreadReader& self, bool enter_sequence = true, bool enter_loop = true) {
                int flags = PALLAS_READ_FLAG_NONE;
                if (enter_sequence) {flags |= PALLAS_READ_FLAG_UNROLL_SEQUENCE;}
                if (enter_loop) {flags |= PALLAS_READ_FLAG_UNROLL_LOOP;}
                if (!flags) {flags = PALLAS_READ_FLAG_NO_UNROLL;}
                self.moveToNextToken(flags);
            })
            .def("pollCurToken", [](pallas::ThreadReader& self) {
                return makePyObjectFromToken(self.pollCurToken(), self);
            })
            .def("enterIfStartOfBlock", [](pallas::ThreadReader& self, bool enter_sequence = true, bool enter_loop = true) {
                int flags = PALLAS_READ_FLAG_NONE;
                if (enter_sequence) {flags |= PALLAS_READ_FLAG_UNROLL_SEQUENCE;}
                if (enter_loop) {flags |= PALLAS_READ_FLAG_UNROLL_LOOP;}
                if (!flags) {flags = PALLAS_READ_FLAG_NO_UNROLL;}
                return self.enterIfStartOfBlock(flags);
            })
            .def("exitIfEndOfBlock", [](pallas::ThreadReader& self, bool exit_sequence = true, bool exit_loop = true) {
                int flags = PALLAS_READ_FLAG_NONE;
                if (exit_sequence) {flags |= PALLAS_READ_FLAG_UNROLL_SEQUENCE;}
                if (exit_loop) {flags |= PALLAS_READ_FLAG_UNROLL_LOOP;}
                if (!flags) {flags = PALLAS_READ_FLAG_NO_UNROLL;}
                return self.exitIfEndOfBlock(flags);
            })
            .def("isEndOfCurrentBlock", &pallas::ThreadReader::isEndOfCurrentBlock)
            .def("isEndOfTrace", &pallas::ThreadReader::isEndOfTrace);

    py::class_<PyLocationGroup>(m, "LocationGroup", "A group of Pallas locations. Usually means a process.")
            .def_readonly("id", &PyLocationGroup::id)
            .def_readonly("name", &PyLocationGroup::name)
            .def_readonly("parent", &PyLocationGroup::parent)
            .def("__repr__", [](const PyLocationGroup& self) { return "<pallas_python.LocationGroup " + std::to_string(self.id) + ": '" + self.name + "'>"; });
    py::class_<PyLocation>(m, "Location", "A Pallas location. Usually means an execution thread.")
            .def_readonly("id", &PyLocation::id)
            .def_readonly("name", &PyLocation::name)
            .def_readonly("parent", &PyLocation::parent)
            .def("__repr__", [](const PyLocation& self) { return "<pallas_python.Location " + std::to_string(self.id) + ": '" + self.name + "'>"; });
    py::class_<PyRegion>(m, "Region", "A Pallas region.")
    .def_readonly("id", &PyRegion::id)
    .def_readonly("name", &PyRegion::name)
    .def("__repr__", [](const PyRegion& self) {
        return "<pallas_python.Region " + std::to_string(self.id) + ": '" + self.name + "'>";
    });

    py::class_<pallas::Archive>(m, "Archive", "A Pallas archive. If it exists, it's already been loaded.")
            .def_readonly("dir_name", &pallas::Archive::dir_name)
            .def_readonly("id", &pallas::Archive::id)
            .def_readonly("metadata", &pallas::Archive::metadata)
            .def_property_readonly("locations", &Archive_get_locations)
            .def_property_readonly("strings", &Archive_get_strings)
            .def_property_readonly("regions", &Archive_get_regions)
            .def_property_readonly("threads", &Archive_get_threads)
    ;

    m.def("open_trace", &open_trace, "Open a Pallas trace")
            .def("get_ABI", []() { return PALLAS_ABI_VERSION; });

    py::class_<pallas::GlobalArchive>(m, "Trace", "A Pallas Trace file.")
            .def(py::init(&open_trace), "Open a trace file and read its structure.")
            .def_readonly("dir_name", &pallas::GlobalArchive::dir_name)
            .def_readonly("trace_name", &pallas::GlobalArchive::trace_name)
            .def_readonly("fullpath", &pallas::GlobalArchive::fullpath)
            .def_readonly("metadata", &pallas::GlobalArchive::metadata)
            .def_property_readonly("locations", &Trace_get_locations)
            .def_property_readonly("location_groups", &Trace_get_location_groups)
            .def_property_readonly("strings", &Trace_get_strings)
            .def_property_readonly("regions", &Trace_get_regions)
            .def_property_readonly("archives", &Trace_get_archives);
}

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */
