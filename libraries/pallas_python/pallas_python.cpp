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

static inline void pop_data(pallas::Event* e, void* data, size_t data_size, byte*& cursor) {
    if (cursor == nullptr) {
        cursor = &e->event_data[0];
    }
    memcpy(data, cursor, data_size);
    cursor += data_size;
}

#define READ(type, name)                          \
    {                                             \
        type name;                                \
        pop_data(e, &name, sizeof(type), cursor); \
        dict[#name] = name;                       \
    }

static py::dict& Event_get_data(pallas::Event* e) {
    auto& dict = *new py::dict();
    byte* cursor = nullptr;

    switch (e->record) {
    case pallas::PALLAS_EVENT_ENTER:
    case pallas::PALLAS_EVENT_LEAVE: {
        READ(pallas::RegionRef, region_ref);
        break;
    }
    case pallas::PALLAS_EVENT_THREAD_FORK:
    case pallas::PALLAS_EVENT_OMP_FORK: {
        READ(uint32_t, numberOfRequestedThreads);
        break;
    }
    case pallas::PALLAS_EVENT_MPI_SEND:
    case pallas::PALLAS_EVENT_MPI_RECV: {
        READ(uint32_t, receiver);
        READ(uint32_t, communicator);
        READ(uint32_t, msgTag);
        READ(uint64_t, msgLength);
        break;
    }
    case pallas::PALLAS_EVENT_MPI_ISEND:
    case pallas::PALLAS_EVENT_MPI_IRECV: {
        READ(uint32_t, receiver);
        READ(uint32_t, communicator);
        READ(uint32_t, msgTag);
        READ(uint64_t, msgLength);
        READ(uint64_t, requestID);
        break;
    }
    case pallas::PALLAS_EVENT_MPI_ISEND_COMPLETE:
    case pallas::PALLAS_EVENT_MPI_IRECV_REQUEST: {
        READ(uint64_t, requestID);
        break;
    }
    case pallas::PALLAS_EVENT_THREAD_ACQUIRE_LOCK:
    case pallas::PALLAS_EVENT_THREAD_RELEASE_LOCK:
    case pallas::PALLAS_EVENT_OMP_ACQUIRE_LOCK:
    case pallas::PALLAS_EVENT_OMP_RELEASE_LOCK: {
        READ(uint32_t, lockID);
        READ(uint32_t, acquisitionOrder);
        break;
    }
    case pallas::PALLAS_EVENT_MPI_COLLECTIVE_END: {
        READ(uint32_t, collectiveOp);
        READ(uint32_t, communicator);
        READ(uint32_t, root);
        READ(uint64_t, sizeSent);
        READ(uint64_t, sizeReceived);
        break;
    }
    case pallas::PALLAS_EVENT_OMP_TASK_CREATE:
    case pallas::PALLAS_EVENT_OMP_TASK_SWITCH:
    case pallas::PALLAS_EVENT_OMP_TASK_COMPLETE: {
        READ(uint64_t, taskID);
        break;
    }
    case pallas::PALLAS_EVENT_GENERIC: {
        pallas::StringRef event_name;
        pop_data(e, &event_name, sizeof(event_name), cursor);
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

template <class VectorType>
class DataHolder {
   private:
    VectorType* data;

   public:
    explicit DataHolder(VectorType* data_) { data = data_; };
    py::array_t<uint64_t> get_array() {
        /* Since we're currently still using np.array, and not out-of-cores custom ones, we need to actually
         * get all the values as a single flat array. This single flat array is owned by the Numpy Array.
         * This whole DataHolder class is a relic from when we didn't have LinkedArray like that when reading.
         */
        // data->ref++;
        return py::array_t({data->size}, {sizeof(uint64_t)}, data->as_flat_array() //
                           // py::capsule(this, [](void* p) {
                           //     auto* holder = reinterpret_cast<DataHolder*>(p);
                           //     if (--holder->data->ref == 0) {
                           //         //holder->data->free_data();
                           //     }
                           //     // delete holder;
                           //     // TODO Python is shit so I had to remove these lines to make sure I don't have issues with Pallas.
                           // })
                           );
    }
};

struct PySequence {
    pallas::Sequence* self;
    pallas::Thread* thread;
};
struct PyLoop {
    pallas::Loop* self;
    pallas::Thread* thread;
};
struct PyEventSummary {
    pallas::EventSummary* self;
    pallas::Thread* thread;
};

std::vector<PySequence> threadGetSequences(pallas::Thread& self) {
    auto output = std::vector<PySequence>(self.nb_sequences);
    for (size_t i = 0; i < self.nb_sequences; i++) {
        output[i].self = self.sequences[i];
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
std::vector<PyEventSummary> threadGetEventsSummary(pallas::Thread& self) {
    auto output = std::vector<PyEventSummary>(self.nb_events);
    for (size_t i = 0; i < self.nb_events; i++) {
        output[i].self = &self.events[i];
        output[i].thread = &self;
    }
    return output;
}

pybind11::list sequenceGetContent(const PySequence& self) {
    auto output = pybind11::list();
    // auto output = std::vector<PySomething>(self.self->tokens.size());
    for (size_t i = 0; i < self.self->tokens.size(); i ++) {
        auto t = self.self->tokens[i];
        if (t.type == pallas::TypeEvent) {
            PyEventSummary temp = {self.thread->getEventSummary(t), self.thread};
            output.append(temp);
        } if (t.type == pallas::TypeSequence) {
            PySequence temp = {self.thread->getSequence(t), self.thread};
            output.append(temp);
        } if (t.type == pallas::TypeLoop) {
            PyLoop temp = {self.thread->getLoop(t), self.thread};
            output.append(temp);
        }
    }
    return output;
}


PYBIND11_MODULE(pallas_trace, m) {
    m.doc() = "Python API for the Pallas library";

    setupEnums(m);

    py::class_<pallas::Token>(m, "Token", "A Pallas token")
      .def_property_readonly("id", [](pallas::Token t) { return t.id; })
      .def_property_readonly("type", [](pallas::Token t) { return t.type; })
      .def("__repr__", &Token_toString);

    py::class_<PySequence>(m, "Sequence", "A Pallas Sequence, ie a group of tokens.")
      .def_property_readonly("id", [](const PySequence& self) { return self.self->id;})
      .def_property_readonly("tokens", [](const PySequence& self) {return self.self->tokens;})
      .def_property_readonly("content", [](const PySequence& self) { return sequenceGetContent(self);})
      .def_property_readonly("timestamps", [](const PySequence& self) { return (new DataHolder(self.self->timestamps))->get_array(); })
      .def_property_readonly("durations", [](const PySequence& self) { return (new DataHolder(self.self->durations))->get_array(); })
      .def_property_readonly("exclusive_durations", [](const PySequence& self) { return (new DataHolder(self.self->exclusive_durations))->get_array(); })
      .def_property_readonly("max_duration", [](const PySequence& self) { return self.self->durations->max; })
      .def_property_readonly("min_duration", [](const PySequence& self) { return self.self->durations->min; })
      .def_property_readonly("mean_duration", [](const PySequence& self) { return self.self->durations->mean; })
      .def("guessName", [](const PySequence& self) { return self.self->guessName(self.thread); })
      .def("__repr__", [](const PySequence& self) { return "<pallas_python.Sequence " + std::to_string(self.self->id) + ">"; });

    py::class_<PyLoop>(m, "Loop", "A Pallas Loop, ie a repetition of a Sequence token.")
      .def_property_readonly("id", [](const PyLoop& self) { return self.self->self_id; })
      .def_property_readonly("repeated_token", [](const PyLoop& self) { return self.self->repeated_token; })
      .def_property_readonly("nb_iterations", [](const PyLoop& self) { return self.self->nb_iterations; })
      .def("__repr__", [](const PyLoop& self) { return "<pallas_python.Loop " + std::to_string(self.self->self_id.id) + ">"; });

    py::class_<PyEventSummary>(m, "EventSummary", "A Pallas Event Summary, that stores info about an event.")
      .def_property_readonly("id", [](const PyEventSummary& self) { return self.self->id; })
      .def_property_readonly("event", [](const PyEventSummary& self) { return self.self->event; })
      .def_property_readonly("nb_occurrences", [](const PyEventSummary& self) { return self.self->nb_occurences; })
      .def_property_readonly("timestamps", [](const PyEventSummary& self) { return (new DataHolder(self.self->timestamps))->get_array(); })
      .def("__repr__", [](const PyEventSummary& self) { return "<pallas_python.EventSummary " + std::to_string(self.self->id) + ">"; });

    py::class_<pallas::Event>(m, "Event", "A Pallas Event.").def_readonly("record", &pallas::Event::record).def_property_readonly("data", &Event_get_data);

    py::class_<pallas::Thread>(m, "Thread", "A Pallas thread.")
      .def_readonly("id", &pallas::Thread::id)
      .def_property_readonly("events", [](pallas::Thread& self) { return threadGetEventsSummary(self); })
      .def_property_readonly("sequences", [](pallas::Thread& self) { return threadGetSequences(self); })
      .def_property_readonly("loops", [](pallas::Thread& self) { return threadGetLoops(self); })
      .def("__repr__", [](const pallas::Thread& self) { return "<pallas_python.Thread " + std::to_string(self.id) + ">"; })
      .def("getSnapshotView", &pallas::Thread::getSnapshotView);

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
    py::class_<PyRegion>(m, "Region", "A Pallas region.").def_readonly("id", &PyRegion::id).def_readonly("name", &PyRegion::name).def("__repr__", [](const PyRegion& self) {
        return "<pallas_python.Region " + std::to_string(self.id) + ": '" + self.name + "'>";
    });

    py::class_<pallas::Archive>(m, "Archive", "A Pallas archive. If it exists, it's already been loaded.")
      .def_readonly("dir_name", &pallas::Archive::dir_name)
      .def_property_readonly("locations", &Archive_get_locations)
      .def_property_readonly("strings", &Archive_get_strings)
      .def_property_readonly("regions", &Archive_get_regions)
      .def_property_readonly("threads", &Archive_get_threads);

    m.def("open_trace", &open_trace, "Open a Pallas trace");

    py::class_<pallas::GlobalArchive>(m, "Trace", "A Pallas Trace file.")
      .def(py::init(&open_trace), "Open a trace file and read its structure.")
      .def_readonly("dir_name", &pallas::GlobalArchive::dir_name)
      .def_readonly("trace_name", &pallas::GlobalArchive::trace_name)
      .def_readonly("fullpath", &pallas::GlobalArchive::fullpath)
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
